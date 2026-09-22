import json
import random
from pathlib import Path

import torch

from cs336_alignment.checkpoint import (
    get_model_and_tokenizer,
)
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
)
from cs336_alignment.grpo import (
    grpo_train_step,
)
from cs336_alignment.vllm_utils import (
    VLLMServer,
)


# ============================================================
# Config
# ============================================================

MODEL_ID = "allenai/OLMo-2-0425-1B"

TRAIN_PATH = Path(
    "data/gsm8k/train.jsonl"
)

VAL_PATH = Path(
    "data/gsm8k/test.jsonl"
)

PROMPT_PATH = Path(
    "cs336_alignment/prompts/r1_zero.prompt"
)


# Dataset
N_TRAIN_EXAMPLES = 64
N_VAL_EXAMPLES = 32


# Training
NUM_ROLLOUT_STEPS = 2

LEARNING_RATE = 1e-5

ROLLOUT_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32

GROUP_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 4

MAX_GRAD_NORM = 1.0


# Sampling
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_MAX_TOKENS = 128


# Logging
VAL_EVERY = 1
LOG_ROLLOUTS_EVERY = 1


# Seed
SEED = 0


# Devices
TRAIN_DEVICE = "cuda:0"
VLLM_GPU = 1


# Output
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path(
    f"outputs/grpo_seed_{SEED}"
)

METRICS_PATH = (
    RESULTS_DIR
    / f"grpo_seed_{SEED}.json"
)


# ============================================================
# Data helpers
# ============================================================

def load_gsm8k(
    path: Path,
):
    samples = []

    with path.open("r") as f:
        for line in f:
            sample = json.loads(line)

            question = sample["question"]

            ground_truth = (
                sample["answer"]
                .split("####")[-1]
                .strip()
            )

            samples.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )

    return samples


def build_prompt(
    template: str,
    question: str,
):
    return template.format(
        question=question
    )


# ============================================================
# Rollout batch helpers
# ============================================================

def prepare_rollout_batch(
    batch,
    prompt_template,
    group_size,
):
    repeated_prompts = []
    repeated_ground_truths = []

    for sample in batch:
        prompt = build_prompt(
            prompt_template,
            sample["question"],
        )

        for _ in range(group_size):
            repeated_prompts.append(
                prompt
            )

            repeated_ground_truths.append(
                sample["ground_truth"]
            )

    return (
        repeated_prompts,
        repeated_ground_truths,
    )


# ============================================================
# Rollout generation
# ============================================================

def generate_rollouts(
    server,
    repeated_prompts,
):
    sampling_params = {
        "temperature": (
            SAMPLING_TEMPERATURE
        ),
        "top_p": (
            SAMPLING_TOP_P
        ),
        "max_tokens": (
            SAMPLING_MAX_TOKENS
        ),
        "stop": [
            "</answer>"
        ],
        "include_stop_str_in_output": True,
    }

    completions = (
        server.generate_completions(
            prompts=repeated_prompts,
            sampling_params=(
                sampling_params
            ),
        )
    )

    rollout_responses = [
        completion.text
        for completion in completions
    ]

    return rollout_responses


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    server,
    dataset,
    prompt_template,
    n_examples,
):
    n_examples = min(
        n_examples,
        len(dataset),
    )

    batch = random.sample(
        dataset,
        n_examples,
    )

    prompts = []
    ground_truths = []

    for sample in batch:
        prompt = build_prompt(
            prompt_template,
            sample["question"],
        )

        prompts.append(
            prompt
        )

        ground_truths.append(
            sample["ground_truth"]
        )

    sampling_params = {
        "temperature": (
            SAMPLING_TEMPERATURE
        ),
        "top_p": (
            SAMPLING_TOP_P
        ),
        "max_tokens": (
            SAMPLING_MAX_TOKENS
        ),
        "stop": [
            "</answer>"
        ],
        "include_stop_str_in_output": True,
    }

    completions = (
        server.generate_completions(
            prompts=prompts,
            sampling_params=(
                sampling_params
            ),
        )
    )

    total_rewards = []
    format_rewards = []
    answer_rewards = []
    response_lengths = []

    for completion, ground_truth in zip(
        completions,
        ground_truths,
    ):
        result = r1_zero_reward_fn(
            response=completion.text,
            ground_truth=ground_truth,
        )

        total_rewards.append(
            result["reward"]
        )

        format_rewards.append(
            result["format_reward"]
        )

        answer_rewards.append(
            result["answer_reward"]
        )

        response_lengths.append(
            len(completion.token_ids)
        )

    metrics = {
        "val_reward": (
            sum(total_rewards)
            / len(total_rewards)
        ),
        "val_format_reward": (
            sum(format_rewards)
            / len(format_rewards)
        ),
        "val_answer_reward": (
            sum(answer_rewards)
            / len(answer_rewards)
        ),
        "val_response_length": (
            sum(response_lengths)
            / len(response_lengths)
        ),
    }

    return metrics


# ============================================================
# Serialization helper
# ============================================================

def to_python_value(
    value,
):
    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.numel() == 1:
            return value.item()

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    return value


def serialize_metadata(
    metadata,
):
    return {
        key: to_python_value(value)
        for key, value
        in metadata.items()
    }


# ============================================================
# Save helpers
# ============================================================

def save_history(
    history,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        json.dumps(
            history,
            indent=2,
        )
    )


def save_model(
    policy,
    tokenizer,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert (
        ROLLOUT_BATCH_SIZE
        % GROUP_SIZE
        == 0
    )

    assert (
        ROLLOUT_BATCH_SIZE
        == TRAIN_BATCH_SIZE
    ), (
        "This script currently implements "
        "standard on-policy GRPO, so "
        "ROLLOUT_BATCH_SIZE must equal "
        "TRAIN_BATCH_SIZE."
    )

    assert (
        TRAIN_BATCH_SIZE
        % GRADIENT_ACCUMULATION_STEPS
        == 0
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_data = load_gsm8k(
        TRAIN_PATH
    )

    val_data = load_gsm8k(
        VAL_PATH
    )

    train_data = train_data[
        :N_TRAIN_EXAMPLES
    ]

    val_data = val_data[
        :N_VAL_EXAMPLES
    ]

    # Shuffle train examples once.
    random.shuffle(
        train_data
    )

    prompt_template = (
        PROMPT_PATH.read_text()
    )

    # --------------------------------------------------------
    # Derived sizes
    # --------------------------------------------------------

    n_prompts_per_rollout_batch = (
        ROLLOUT_BATCH_SIZE
        // GROUP_SIZE
    )

    required_train_examples = (
        NUM_ROLLOUT_STEPS
        * n_prompts_per_rollout_batch
    )

    if (
        required_train_examples
        > len(train_data)
    ):
        raise ValueError(
            "Not enough training examples. "
            f"Need {required_train_examples}, "
            f"but only have {len(train_data)}."
        )

    print(
        "\n===== CONFIG ====="
    )

    print(
        f"Model: {MODEL_ID}"
    )

    print(
        f"Train examples: "
        f"{len(train_data)}"
    )

    print(
        f"Validation examples: "
        f"{len(val_data)}"
    )

    print(
        f"Rollout batch size: "
        f"{ROLLOUT_BATCH_SIZE}"
    )

    print(
        f"Group size: "
        f"{GROUP_SIZE}"
    )

    print(
        f"Prompts per rollout batch: "
        f"{n_prompts_per_rollout_batch}"
    )

    print(
        f"Gradient accumulation steps: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Seed: {SEED}"
    )

    # --------------------------------------------------------
    # Load Hugging Face policy
    # --------------------------------------------------------

    print(
        "\nLoading policy model..."
    )

    policy, tokenizer = (
        get_model_and_tokenizer(
            MODEL_ID,
            TRAIN_DEVICE,
        )
    )

    policy.train()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        betas=(
            0.9,
            0.95,
        ),
        weight_decay=0.0,
    )

    # --------------------------------------------------------
    # Start vLLM server
    # --------------------------------------------------------

    print(
        "\nStarting vLLM..."
    )

    server = VLLMServer(
        model_id=MODEL_ID,
        gpu=VLLM_GPU,
        seed=SEED,
    )

    server.start()

    # Create NCCL weight-sync connection
    server.init_weight_sync(
        policy_device=TRAIN_DEVICE,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for step in range(
        1,
        NUM_ROLLOUT_STEPS + 1,
    ):

        print(
            "\n"
            "===================================="
        )

        print(
            f"ROLLOUT STEP {step}"
        )

        print(
            "===================================="
        )

        # ----------------------------------------------------
        # 1. Sync latest policy weights to vLLM
        # ----------------------------------------------------

        server.sync_policy_weights(
            policy
        )

        # ----------------------------------------------------
        # 2. Select next 32 unique prompts
        # ----------------------------------------------------

        start_idx = (
            (step - 1)
            * n_prompts_per_rollout_batch
        )

        end_idx = (
            start_idx
            + n_prompts_per_rollout_batch
        )

        batch = train_data[
            start_idx:end_idx
        ]

        (
            repeated_prompts,
            repeated_ground_truths,
        ) = prepare_rollout_batch(
            batch=batch,
            prompt_template=(
                prompt_template
            ),
            group_size=GROUP_SIZE,
        )

        assert (
            len(repeated_prompts)
            == ROLLOUT_BATCH_SIZE
        )

        # ----------------------------------------------------
        # 3. Generate rollouts
        # ----------------------------------------------------

        rollout_responses = (
            generate_rollouts(
                server,
                repeated_prompts,
            )
        )

        assert (
            len(rollout_responses)
            == ROLLOUT_BATCH_SIZE
        )

        # ----------------------------------------------------
        # 4. One standard on-policy GRPO update
        # ----------------------------------------------------

        loss, metadata = (
            grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=(
                    GRADIENT_ACCUMULATION_STEPS
                ),
                max_grad_norm=(
                    MAX_GRAD_NORM
                ),
                reward_fn=(
                    r1_zero_reward_fn
                ),
                repeated_prompts=(
                    repeated_prompts
                ),
                rollout_responses=(
                    rollout_responses
                ),
                repeated_ground_truths=(
                    repeated_ground_truths
                ),
                group_size=GROUP_SIZE,

                # Standard GRPO
                baseline="mean",
                advantage_eps=1e-6,
                advantage_normalizer="std",

                # Fully on-policy
                importance_reweighting_method=(
                    "none"
                ),
                old_log_probs=None,
                cliprange=None,

                # Standard sequence normalization
                loss_normalization=(
                    "sequence"
                ),
                normalization_constant=None,
            )
        )

        # ----------------------------------------------------
        # 5. Record train metrics
        # ----------------------------------------------------

        record = {
            "step": step,
            "seed": SEED,
            "loss": loss.item(),
        }

        record.update(
            serialize_metadata(
                metadata
            )
        )

        print(
            f"loss = "
            f"{loss.item():.6f}"
        )

        for key, value in (
            serialize_metadata(
                metadata
            ).items()
        ):
            print(
                f"{key} = {value}"
            )

        # ----------------------------------------------------
        # 6. Validation
        # ----------------------------------------------------

        if (
            step % VAL_EVERY
            == 0
        ):
            print(
                "\nRunning validation..."
            )

            # vLLM currently still has pre-update
            # weights from the start of this step,
            # so sync again after optimizer.step().
            server.sync_policy_weights(
                policy
            )

            val_metrics = evaluate(
                server=server,
                dataset=val_data,
                prompt_template=(
                    prompt_template
                ),
                n_examples=(
                    N_VAL_EXAMPLES
                ),
            )

            record.update(
                val_metrics
            )

            print(
                "\n===== VALIDATION ====="
            )

            for key, value in (
                val_metrics.items()
            ):
                print(
                    f"{key} = {value}"
                )

        # ----------------------------------------------------
        # 7. Qualitative rollout logging
        # ----------------------------------------------------

        if (
            step
            % LOG_ROLLOUTS_EVERY
            == 0
        ):
            print(
                "\n===== SAMPLE ROLLOUTS ====="
            )

            for i in range(
                min(
                    3,
                    len(
                        rollout_responses
                    ),
                )
            ):
                print(
                    f"\n--- rollout {i} ---"
                )

                print(
                    rollout_responses[i]
                )

                print(
                    "\nGround truth:"
                )

                print(
                    repeated_ground_truths[i]
                )

        # ----------------------------------------------------
        # 8. Save metrics after every step
        # ----------------------------------------------------

        history.append(
            record
        )

        save_history(
            history
        )

    # --------------------------------------------------------
    # Save final trained model
    # --------------------------------------------------------

    print(
        "\nSaving final model..."
    )

    save_model(
        policy,
        tokenizer,
    )

    print(
        "\n===== TRAINING COMPLETE ====="
    )

    print(
        f"Metrics saved to:"
        f"\n{METRICS_PATH}"
    )

    print(
        f"Model saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()