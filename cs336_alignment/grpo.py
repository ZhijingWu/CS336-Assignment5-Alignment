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

    group_mean = grouped_rewards.mean(
        dim=1,
        keepdim=True,
    )

    if baseline == "mean":
        advantages = (
            grouped_rewards - group_mean
        )
    elif baseline == "none":
        advantages = grouped_rewards

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
    elif advantage_normalizer == "mean":
        advantages = (
            advantages / (group_mean + advantage_eps)
        )

    elif advantage_normalizer == "none":
        pass

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


def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    response_mask=None,
):
    advantages = raw_rewards_or_advantages

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    
    if importance_reweighting_method == "none":
        per_token_loss = (
            - advantages * policy_log_probs
        )
    elif importance_reweighting_method == "noclip":
        ratio = torch.exp(
            policy_log_probs - old_log_probs
        )

        per_token_loss = (
            - advantages * ratio
        )
    elif importance_reweighting_method == "grpo":
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required for off-policy GRPO."
            )
        
        if cliprange is None:
            raise ValueError(
                "cliprange is required for GRPO clipping."
            )
        
        ratio = torch.exp(
            policy_log_probs - old_log_probs
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        objective = torch.minimum(
            advantages * ratio,
            advantages * clipped_ratio,
        )

        per_token_loss = -objective
    elif importance_reweighting_method == "gspo":
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required for off-policy GSPO."
            )
                
        if cliprange is None:
            raise ValueError(
                "cliprange is required for GSPO clipping."
            )

        if response_mask is None:
            raise ValueError(
                "response_mask i required for GSPO clipping"
            )

        log_ratio = (
            policy_log_probs - old_log_probs
        )

        response_length = response_mask.sum(
            dim=-1,
            keepdim=True
        )

        sequence_log_ratio = (
            (log_ratio * response_mask)
            .sum(
                dim=1,
                keepdim=True,
            )
            / response_length
        )

        sequence_ratio = torch.exp(
            sequence_log_ratio
        )

        clipped_ratio = torch.clamp(
            sequence_ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        objective = torch.minimum(
            advantages * sequence_ratio,
            advantages * clipped_ratio,
        )

        per_token_loss = -objective.expand_as(
            policy_log_probs
        )

    metadata = {}

    return per_token_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss,
    mask: torch.Tensor,
    loss_normalization,
    normalization_constant=None,
):
    masked_loss = (
        per_token_policy_gradient_loss * mask
    )

    if loss_normalization == "sequence":
        sequence_loss = (
            masked_loss.sum(dim=1)
            / mask.sum(dim=1)
        )

        loss = sequence_loss.mean()

    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required "
                "when loss_normalization='constant'"
            )

        loss = (
            masked_loss.sum()
            / normalization_constant
        )

    else:
        raise NotImplementedError

    return loss


def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps,
    max_grad_norm,
    reward_fn,
    repeated_prompts,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    # Reward normalization
    baseline,
    advantage_eps,
    advantage_normalizer,
    # Importance reweighting and clipping
    importance_reweighting_method,
    old_log_probs,
    cliprange,
    # Loss normalization
    loss_normalization,
    normalization_constant,
):
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )

    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )

    original_batch_size = len(rollout_responses)

    keep_indices = torch.nonzero(
        advantages != 0,
        as_tuple=False,
    ).squeeze(-1)

    advantages = advantages[keep_indices]

    tokenized = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer=tokenizer
    )

    input_ids = tokenized["input_ids"][keep_indices]
    labels = tokenized["labels"][keep_indices]
    response_mask = tokenized["response_mask"][keep_indices]

    batch_size = len(rollout_responses)

    microbatch_size = (
        original_batch_size // gradient_accumulation_steps
    )

    effective_batch_size = input_ids.shape[0]

    if old_log_probs is not None:
        old_log_probs = old_log_probs[keep_indices]

    device = next(model.parameters()).device

    optimizer.zero_grad()

    if effective_batch_size == 0:
        zero = torch.tensor(
            0.0,
            device=device,
        )

        metadata = {
            **reward_metadata,
            **advantage_metadata,
            "loss": zero,
            "grad_norm": zero,
            "token_entropy": zero,
        }

        return zero, metadata

    total_loss = torch.zeros(
        (),
        device=device,
    )

    entropy_sum = torch.zeros(
        (),
        device=device,
    )

    entropy_count = torch.zeros(
        (),
        device=device,
    )

    for start in range(
        0,
        effective_batch_size,
        microbatch_size,
    ):
        end = min(start + microbatch_size, effective_batch_size)

        input_ids_mb = input_ids[start:end].to(device)
        labels_mb = labels[start:end].to(device)
        mask_mb = response_mask[start:end].to(device)
        advantages_mb = advantages[start:end].to(device)
        if old_log_probs is not None:
            old_log_probs_mb = old_log_probs[start:end].to(device)
        else:
            old_log_probs_mb = None

        response_info = get_response_log_probs(
            model,
            input_ids_mb,
            labels_mb,
            return_token_entropy=True,
        )

        policy_log_probs = response_info["log_probs"]
        token_entropy = response_info["token_entropy"]

        entropy_sum += (
            token_entropy * mask_mb
        ).sum().detach()

        entropy_count += mask_mb.sum().detach()

        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            advantages_mb,
            policy_log_probs,
            importance_reweighting_method,
            old_log_probs_mb,
            cliprange,
            mask_mb,
        )

        loss = aggregate_loss_across_microbatch(
            per_token_loss,
            mask_mb,
            loss_normalization,
            normalization_constant,
        )

        current_microbatch_size = end - start

        if loss_normalization == "sequence":
            scaled_loss = loss * (current_microbatch_size / effective_batch_size)
        elif loss_normalization == "constant":
            scaled_loss = loss

        scaled_loss.backward()

        total_loss += scaled_loss.detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float("inf"),
        )

    optimizer.step()

    optimizer.zero_grad()

    mean_token_entropy = entropy_sum / entropy_count

    metadata = {
        **reward_metadata,
        **advantage_metadata,
        "loss": total_loss,
        "grad_norm": grad_norm.detach(),
        "token_entropy": mean_token_entropy,
    }

    return total_loss, metadata


