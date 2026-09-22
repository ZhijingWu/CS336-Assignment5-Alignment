import json
import random
from pathlib import Path

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


# ============================================================
# Config
# ============================================================

MODEL_ID = "allenai/OLMo-2-0425-1B"

TRAIN_PATH = Path("data/gsm8k/train.jsonl")
VAL_PATH = Path("data/gsm8k/test.jsonl")

PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")

N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024

NUM_ROLLOUT_STEPS = 200

LEARNING_RATE = 1e-5

ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256

GROUP_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 32
MAX_GRAD_NORM = 1.0

SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_MAX_TOKENS = 512

VAL_EVERY = 10
LOG_ROLLOUTS_EVERY = 40

SEED = 0

TRAIN_DEVICE = "cuda:0"
VLLM_GPU = 1


# ============================================================
# Data helpers
# ============================================================

def load_gsm8k(path: Path):
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


def build_prompt(template: str, question: str):
    return template.format(
        question=question
    )


# ============================================================
# Rollout helpers
# ============================================================

def sample_rollout_batch(
    dataset,
    prompt_template,
    n_prompts,
    group_size,
):
    batch = random.sample(
        dataset,
        n_prompts,
    )

    repeated_prompts = []
    repeated_ground_truths = []

    for sample in batch:
        prompt = build_prompt(
            prompt_template,
            sample["question"],
        )

        for _ in range(group_size):
            repeated_prompts.append(prompt)
            repeated_ground_truths.append(
                sample["ground_truth"]
            )

    return (
        repeated_prompts,
        repeated_ground_truths,
    )


def generate_rollouts(
    server,
    repeated_prompts,
):
    sampling_params = {
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "max_tokens": SAMPLING_MAX_TOKENS,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    completions = server.generate_completions(
        prompts=repeated_prompts,
        sampling_params=sampling_params,
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
    batch = random.sample(
        dataset,
        min(n_examples, len(dataset)),
    )

    prompts = []
    ground_truths = []

    for sample in batch:
        prompts.append(
            build_prompt(
                prompt_template,
                sample["question"],
            )
        )

        ground_truths.append(
            sample["ground_truth"]
        )

    sampling_params = {
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "max_tokens": SAMPLING_MAX_TOKENS,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    total_rewards = []
    format_rewards = []
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
        "val_response_length": (
            sum(response_lengths)
            / len(response_lengths)
        ),
    }

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Load data
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

    prompt_template = (
        PROMPT_PATH.read_text()
    )

    # --------------------------------------------------------
    # Load policy model
    # --------------------------------------------------------

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
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # --------------------------------------------------------
    # vLLM server
    # --------------------------------------------------------

    server = VLLMServer(
        model_id=MODEL_ID,
        gpu=VLLM_GPU,
        seed=SEED,
    )

    server.start()

    server.init_weight_sync(
        policy_device=TRAIN_DEVICE,
    )

    # --------------------------------------------------------
    # GRPO loop
    # --------------------------------------------------------

    n_prompts_per_rollout_batch = (
        ROLLOUT_BATCH_SIZE
        // GROUP_SIZE
    )

    for step in range(
        1,
        NUM_ROLLOUT_STEPS + 1,
    ):
        # 1. Sync current policy into vLLM
        server.sync_policy_weights(
            policy
        )

        # 2. Sample prompts
        (
            repeated_prompts,
            repeated_ground_truths,
        ) = sample_rollout_batch(
            train_data,
            prompt_template,
            n_prompts_per_rollout_batch,
            GROUP_SIZE,
        )

        # 3. Generate rollouts
        rollout_responses = (
            generate_rollouts(
                server,
                repeated_prompts,
            )
        )

        # 4. Train on this rollout batch
        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
            max_grad_norm=MAX_GRAD_NORM,
            reward_fn=r1_zero_reward_fn,
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
            baseline="mean",
            advantage_eps=1e-6,
            advantage_normalizer="std",
            importance_reweighting_method=(
                "none"
            ),
            old_log_probs=None,
            cliprange=None,
            loss_normalization="sequence",
            normalization_constant=None,
        )

        print(
            f"\nStep {step}"
        )

        print(
            f"loss = {loss.item():.6f}"
        )

        print(
            f"metadata = {metadata}"
        )

        # 5. Periodic validation
        if step % VAL_EVERY == 0:
            server.sync_policy_weights(
                policy
            )

            val_metrics = evaluate(
                server,
                val_data,
                prompt_template,
                N_VAL_EXAMPLES,
            )

            print(
                f"validation = "
                f"{val_metrics}"
            )

        # 6. Periodic rollout inspection
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


if __name__ == "__main__":
    main()