import json
from pathlib import Path

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMServer

prompt_dir = Path("cs336_alignment/prompts")

question_only_template = (
    prompt_dir / "question_only.prompt"
).read_text()

r1_zero_template = (
    prompt_dir / "r1_zero.prompt"
).read_text()

three_shot_template = (
    prompt_dir / "r1_zero_three_shot_gsm8k.prompt"
).read_text()

data_path = Path("data/gsm8k/test.jsonl")

samples = []

with data_path.open("r") as f:
    for line in f:
        sample = json.loads(line)
        samples.append(sample)


print(f"Loaded {len(samples)} samples")


question_only_prompts = []
r1_zero_prompts = []
three_shot_prompts = []

ground_truths = []

for sample in samples:
    question = sample["question"]
    ground_truth = sample["answer"].split("####")[-1].strip()

    question_only_prompt = question_only_template.format(question=question)
    r1_zero_prompt = r1_zero_template.format(question=question)
    three_shot_prompt = three_shot_template.format(question=question)

    question_only_prompts.append(question_only_prompt)
    r1_zero_prompts.append(r1_zero_prompt)
    three_shot_prompts.append(three_shot_prompt)

    ground_truths.append(ground_truth)

def evaluate_responses(responses, ground_truths, reward_fn):
    assert len(responses) == len(ground_truths)

    stats = {
        "correct_and_formatted": 0,
        "wrong_but_formatted": 0,
        "bad_format": 0
    }

    results = []

    for response, ground_truth in zip(responses, ground_truths):
        result = reward_fn(
            response=response,
            ground_truth=ground_truth,
        )

        results.append(result)

        format_reward = result["format_reward"]
        answer_reward = result["answer_reward"]

        if format_reward == 1.0 and answer_reward == 1.0:
            stats["correct_and_formatted"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            stats["wrong_but_formatted"] += 1
        else:
            stats["bad_format"] += 1

    return results, stats


fake_responses = [
    "The answer is \\boxed{18}.",
    "The answer is \\boxed{17}.",
    "I think the answer is 18.",
]

fake_ground_truths = [
    "18",
    "18",
    "18",
]

results, stats = evaluate_responses(
    responses=fake_responses,
    ground_truths=fake_ground_truths,
    reward_fn=question_only_reward_fn,
)

print("\n===== LOCAL GRADER TEST =====")

for i, result in enumerate(results):
    print(f"\nSample {i}")
    print("Response:")
    print(fake_responses[i])

    print("Ground truth:")
    print(fake_ground_truths[i])

    print("Reward result:")
    print(result)

print("\n===== STATS =====")
print(stats)


# ============================================================
# 6. Real vLLM inference
#    Enable this part on a machine with NVIDIA GPU
# ============================================================

# server = VLLMServer(
#     model_id="allenai/OLMo-2-0425-1B"
# )
#
# server.start()


# ----------------------------
# question_only
# ----------------------------

# question_only_sampling_params = {
#     "temperature": 1.0,
#     "top_p": 1.0,
#     "max_tokens": 512,
# }
#
# question_only_completions = (
#     server.generate_completions(
#         prompts=question_only_prompts,
#         sampling_params=question_only_sampling_params,
#     )
# )
#
# question_only_responses = [
#     completion.text
#     for completion in question_only_completions
# ]
#
# question_only_results, question_only_stats = (
#     evaluate_responses(
#         responses=question_only_responses,
#         ground_truths=ground_truths,
#         reward_fn=question_only_reward_fn,
#     )
# )
#
# print("\n===== QUESTION ONLY =====")
# print(question_only_stats)


# ----------------------------
# r1_zero
# ----------------------------

# r1_sampling_params = {
#     "temperature": 1.0,
#     "top_p": 1.0,
#     "max_tokens": 512,
#     "stop": ["</answer>"],
#     "include_stop_str_in_output": True,
# }
#
# r1_zero_completions = (
#     server.generate_completions(
#         prompts=r1_zero_prompts,
#         sampling_params=r1_sampling_params,
#     )
# )
#
# r1_zero_responses = [
#     completion.text
#     for completion in r1_zero_completions
# ]
#
# r1_zero_results, r1_zero_stats = (
#     evaluate_responses(
#         responses=r1_zero_responses,
#         ground_truths=ground_truths,
#         reward_fn=r1_zero_reward_fn,
#     )
# )
#
# print("\n===== R1 ZERO =====")
# print(r1_zero_stats)


# ----------------------------
# r1_zero_three_shot
# ----------------------------

# three_shot_completions = (
#     server.generate_completions(
#         prompts=three_shot_prompts,
#         sampling_params=r1_sampling_params,
#     )
# )
#
# three_shot_responses = [
#     completion.text
#     for completion in three_shot_completions
# ]
#
# three_shot_results, three_shot_stats = (
#     evaluate_responses(
#         responses=three_shot_responses,
#         ground_truths=ground_truths,
#         reward_fn=r1_zero_reward_fn,
#     )
# )
#
# print("\n===== R1 ZERO THREE SHOT =====")
# print(three_shot_stats)