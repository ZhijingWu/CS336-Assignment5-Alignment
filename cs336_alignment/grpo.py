import torch
from torch.nn.utils.rnn import pad_sequence

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer 
):
    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
    )["input_ids"]

    output_ids = tokenizer(
        output_strs,
        add_special_tokens=False
    )["input_ids"]

    batch_all_ids = []
    batch_masks = []

    for prompt_id, output_id in zip(prompt_ids, output_ids):
        all_id = prompt_id + output_id

        mask = [0] * len(prompt_id) + [1] * len(output_id)

        batch_all_ids.append(torch.tensor(all_id))
        batch_masks.append(torch.tensor(mask))

    all_ids = pad_sequence(
        batch_all_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    padded_masks = pad_sequence(
        batch_masks,
        batch_first=True,
        padding_value=0,
    )

    input_ids = all_ids[:, :-1]
    labels = all_ids[:, 1:]
    response_mask = padded_masks[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_log_probs(
    model,
    input_ids,
    labels,
    return_token_entropy=False,
):
    logits = model(input_ids).logits

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs": token_log_probs,
    }

    if return_token_entropy:
        probs = torch.softmax(
            logits,
            dim=-1
        )

        token_entropy = -(
            probs*log_probs
        ).sum(dim=-1)

        result["token_entropy"] = token_entropy

    return result


def compute_rollout_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
):
    rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(
        rollout_responses,
        repeated_ground_truths,
    ):
        result = reward_fn(
            response,
            ground_truth,
        )

        rewards.append(result["reward"])
        format_rewards.append(result["format_reward"])
        answer_rewards.append(result["answer_reward"])

    raw_rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    metadata = {
        "rewards": sum(rewards) / len(rewards),
        "forward_rewards": sum(format_rewards) / len(format_rewards),
        "answer_rewards": sum(answer_rewards) / len(answer_rewards),
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards,
    group_size,
    baseline="mean",
    advantage_eps=1e-6,
    advantage_normalizer="std",
):
    grouped_rewards = raw_rewards.reshape(-1, group_size)

    if baseline == "mean":
        group_mean = grouped_rewards.mean(
            dim=1,
            keepdim=True,
        )

        advantages = (
            grouped_rewards - group_mean
        )
    else:
        raise NotImplementedError

    if advantage_normalizer == "std":
        group_std = grouped_rewards.std(
            dim=1,
            keepdim=True,
        )

        advantages = (
            advantages
            / (group_std + advantage_eps)
        )
    else:
        raise NotImplementedError

    advantages = advantages.reshape(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(unbiased=False).item(),
    }

    return advantages, metadata