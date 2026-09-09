# Copyright 2025 Individual Contributor: NTHR example contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure PyTorch implementation of the NTHR group computation.

This module deliberately has no Ray, Hydra, or VERL worker dependencies.  Keeping
the numerical kernel separate makes it possible to test the method on CPU.
"""

from dataclasses import dataclass

import torch


@dataclass
class NTHRGroupResult:
    """NTHR output for one prompt and its rollout group."""

    selected: list[torch.Tensor]
    scores: list[torch.Tensor]
    eta: float
    tau: torch.Tensor | None
    active: bool


def chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute a last-dimension logsumexp in FP32 with bounded scratch memory."""

    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty final dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    running_max = torch.full(logits.shape[:-1], -torch.inf, dtype=torch.float32, device=logits.device)
    running_sum = torch.zeros_like(running_max)
    for start in range(0, logits.shape[-1], chunk_size):
        chunk = logits[..., start : start + chunk_size].to(torch.float32)
        chunk_max = chunk.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * torch.exp(running_max - new_max)
        running_sum += torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        running_max = new_max
    return running_max + torch.log(running_sum)


def _validate_binary_rewards(response_scores: torch.Tensor, atol: float) -> None:
    close_to_zero = torch.isclose(response_scores, torch.zeros_like(response_scores), atol=atol, rtol=0.0)
    close_to_one = torch.isclose(response_scores, torch.ones_like(response_scores), atol=atol, rtol=0.0)
    if not torch.all(close_to_zero | close_to_one):
        values = response_scores.detach().cpu().tolist()
        raise ValueError(
            "NTHR currently requires binary response rewards (0 or 1); "
            f"received {values}. Set algorithm.nthr.require_binary_rewards=false "
            "only if reward_threshold is a meaningful positive/negative boundary."
        )


def prediction_error_on_vstar(
    probabilities: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    vocabulary: torch.Tensor,
) -> torch.Tensor:
    """Return ``one_hot(sample) - probabilities`` on the sampled vocabulary.

    ``vocabulary`` must be the sorted unique set of sampled tokens for the whole
    prompt group.  The sliced probabilities are intentionally not renormalized.
    """

    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must have shape [tokens, V*], got {probabilities.shape}")
    if sampled_token_ids.ndim != 1 or sampled_token_ids.shape[0] != probabilities.shape[0]:
        raise ValueError("sampled_token_ids must have one entry per probability row")
    if vocabulary.ndim != 1 or vocabulary.numel() != probabilities.shape[1]:
        raise ValueError("vocabulary must have one entry per probability column")

    sampled_columns = torch.searchsorted(vocabulary, sampled_token_ids)
    if vocabulary.numel() == 0 or torch.any(sampled_columns >= vocabulary.numel()):
        raise ValueError("Every sampled token must be present in vocabulary")
    if not torch.equal(vocabulary[sampled_columns], sampled_token_ids):
        raise ValueError("Every sampled token must be present in vocabulary")

    prediction_error = -probabilities.to(torch.float32)
    prediction_error = prediction_error.clone()
    rows = torch.arange(prediction_error.shape[0], device=prediction_error.device)
    prediction_error[rows, sampled_columns] += 1.0
    return prediction_error


@torch.no_grad()
def compute_nthr_group(
    *,
    response_scores: torch.Tensor,
    token_ids: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    probabilities_on_vstar: list[torch.Tensor],
    vocabulary: torch.Tensor,
    beta: float = 1.0,
    reward_threshold: float = 0.5,
    require_binary_rewards: bool = True,
    binary_reward_atol: float = 1e-6,
) -> NTHRGroupResult:
    """Compute selected negative tokens for one prompt's rollout group.

    Args:
        response_scores: Scalar reward for each response, shape ``[G]``.
        token_ids: Valid completion token ids for every response.
        hidden_states: Old-policy final hidden states, one ``[T_i, d]`` tensor
            per response.  State ``k`` must predict token ``k``.
        probabilities_on_vstar: Old-policy probabilities on ``vocabulary``, one
            ``[T_i, V*]`` tensor per response.  These probabilities must not be
            renormalized after slicing the vocabulary.
        vocabulary: Sorted unique completion token ids for this prompt.
        beta: Positive-anchor threshold multiplier.
        reward_threshold: Rewards at or above this value are positive.
        require_binary_rewards: Validate that rewards are approximately 0/1.
        binary_reward_atol: Absolute tolerance used by the binary validation.
    """

    group_size = response_scores.numel()
    if response_scores.ndim != 1:
        raise ValueError(f"response_scores must have shape [G], got {response_scores.shape}")
    if not (len(token_ids) == len(hidden_states) == len(probabilities_on_vstar) == group_size):
        raise ValueError("All feature lists must contain exactly one tensor per response")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if binary_reward_atol < 0:
        raise ValueError(f"binary_reward_atol must be non-negative, got {binary_reward_atol}")
    if group_size == 0:
        raise ValueError("NTHR requires a non-empty rollout group")
    if require_binary_rewards:
        if not 0.0 < reward_threshold <= 1.0:
            raise ValueError("reward_threshold must be in (0, 1] when binary rewards are required")
        _validate_binary_rewards(response_scores, binary_reward_atol)

    device = hidden_states[0].device
    scores_on_device = response_scores.to(device=device)
    positive = scores_on_device >= reward_threshold
    num_positive = int(positive.sum().item())
    num_negative = group_size - num_positive
    p = num_positive / group_size
    eta = 2.0 * abs(0.5 - p)

    empty_selection = [torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device) for ids in token_ids]
    empty_scores = [torch.zeros(ids.shape[0], dtype=torch.float32, device=ids.device) for ids in token_ids]
    if num_positive == 0 or num_negative == 0:
        return NTHRGroupResult(
            selected=empty_selection,
            scores=empty_scores,
            eta=1.0,
            tau=None,
            active=False,
        )

    prediction_errors = []
    hidden_fp32 = []
    hidden_size = None
    for response_idx, (ids, hidden, probabilities) in enumerate(
        zip(token_ids, hidden_states, probabilities_on_vstar, strict=True)
    ):
        if ids.ndim != 1:
            raise ValueError(f"token_ids[{response_idx}] must have shape [T_i]")
        if hidden.ndim != 2 or hidden.shape[0] != ids.shape[0]:
            raise ValueError(f"hidden_states[{response_idx}] must have shape [T_i, d]")
        if probabilities.shape != (ids.shape[0], vocabulary.numel()):
            raise ValueError(f"probabilities_on_vstar[{response_idx}] must have shape [T_i, V*]")
        if ids.numel() == 0:
            raise ValueError("NTHR does not support an empty completion inside a mixed group")
        if hidden_size is None:
            hidden_size = hidden.shape[1]
        elif hidden.shape[1] != hidden_size:
            raise ValueError("All hidden-state tensors must use the same hidden dimension")
        prediction_errors.append(prediction_error_on_vstar(probabilities, ids, vocabulary))
        hidden_fp32.append(hidden.to(torch.float32))

    positive_errors = torch.cat(
        [prediction_errors[i] for i in range(group_size) if bool(positive[i])], dim=0
    )
    positive_hidden = torch.cat([hidden_fp32[i] for i in range(group_size) if bool(positive[i])], dim=0)

    # M+ = (G+)^T H+.  All accumulation is FP32 even when the model uses BF16.
    positive_aggregate = positive_errors.transpose(0, 1) @ positive_hidden

    per_response_scores = [
        ((error @ positive_aggregate) * hidden).sum(dim=-1)
        for error, hidden in zip(prediction_errors, hidden_fp32, strict=True)
    ]
    positive_anchor_scores = torch.stack(
        [per_response_scores[i].mean() for i in range(group_size) if bool(positive[i])]
    )
    tau = positive_anchor_scores.min() * beta

    selected = []
    for response_idx in range(group_size):
        if bool(positive[response_idx]):
            selected.append(torch.zeros_like(per_response_scores[response_idx], dtype=torch.bool))
        else:
            selected.append(per_response_scores[response_idx] > tau)

    return NTHRGroupResult(
        selected=selected,
        scores=per_response_scores,
        eta=eta,
        tau=tau,
        active=True,
    )
