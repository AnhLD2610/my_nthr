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

import pytest
import torch

from examples.nthr_trainer.nthr import chunked_logsumexp, compute_nthr_group, prediction_error_on_vstar


def _brute_force_scores(errors, hidden, positive_rows):
    output = []
    for target_error, target_hidden in zip(errors, hidden, strict=True):
        target_scores = []
        for target_g, target_h in zip(target_error, target_hidden, strict=True):
            score = torch.tensor(0.0)
            for positive_row in positive_rows:
                for positive_g, positive_h in zip(errors[positive_row], hidden[positive_row], strict=True):
                    score += torch.dot(positive_g, target_g) * torch.dot(positive_h, target_h)
            target_scores.append(score)
        output.append(torch.stack(target_scores))
    return output


def test_nthr_matrix_form_matches_pairwise_definition():
    vocabulary = torch.tensor([2, 4, 7, 9])
    token_ids = [torch.tensor([2, 7]), torch.tensor([4]), torch.tensor([9, 2])]
    probabilities = [
        torch.tensor([[0.10, 0.20, 0.30, 0.05], [0.20, 0.05, 0.15, 0.10]]),
        torch.tensor([[0.30, 0.10, 0.20, 0.05]]),
        torch.tensor([[0.05, 0.10, 0.25, 0.20], [0.15, 0.20, 0.10, 0.05]]),
    ]
    hidden = [
        torch.tensor([[1.0, 0.5], [0.0, 2.0]]),
        torch.tensor([[0.5, -1.0]]),
        torch.tensor([[1.5, 0.25], [-0.5, 1.0]]),
    ]
    rewards = torch.tensor([1.0, 1.0, 0.0])

    result = compute_nthr_group(
        response_scores=rewards,
        token_ids=token_ids,
        hidden_states=hidden,
        probabilities_on_vstar=probabilities,
        vocabulary=vocabulary,
    )
    errors = [
        prediction_error_on_vstar(probability, ids, vocabulary)
        for probability, ids in zip(probabilities, token_ids, strict=True)
    ]
    expected = _brute_force_scores(errors, hidden, positive_rows=[0, 1])

    assert result.active
    assert result.eta == pytest.approx(1.0 / 3.0)
    for actual_scores, expected_scores in zip(result.scores, expected, strict=True):
        torch.testing.assert_close(actual_scores, expected_scores)
    expected_tau = min(expected[0].mean(), expected[1].mean())
    torch.testing.assert_close(result.tau, expected_tau)
    assert torch.equal(result.selected[2], expected[2] > expected_tau)
    assert not result.selected[0].any()
    assert not result.selected[1].any()


@pytest.mark.parametrize("rewards", [torch.zeros(2), torch.ones(2)])
def test_nthr_falls_back_for_non_mixed_groups(rewards):
    result = compute_nthr_group(
        response_scores=rewards,
        token_ids=[torch.tensor([1]), torch.tensor([1])],
        hidden_states=[torch.ones(1, 2), torch.ones(1, 2)],
        probabilities_on_vstar=[torch.full((1, 1), 0.25), torch.full((1, 1), 0.25)],
        vocabulary=torch.tensor([1]),
    )

    assert not result.active
    assert result.eta == 1.0
    assert result.tau is None
    assert not any(mask.any() for mask in result.selected)


def test_nthr_rejects_non_binary_rewards_by_default():
    with pytest.raises(ValueError, match="binary response rewards"):
        compute_nthr_group(
            response_scores=torch.tensor([1.0, 0.25]),
            token_ids=[torch.tensor([1]), torch.tensor([1])],
            hidden_states=[torch.ones(1, 2), torch.ones(1, 2)],
            probabilities_on_vstar=[torch.full((1, 1), 0.25), torch.full((1, 1), 0.25)],
            vocabulary=torch.tensor([1]),
        )


def test_signed_correctness_rewards_match_zero_one_split_when_validation_is_disabled():
    inputs = {
        "token_ids": [torch.tensor([1]), torch.tensor([1])],
        "hidden_states": [torch.ones(1, 2), torch.ones(1, 2)],
        "probabilities_on_vstar": [torch.full((1, 1), 0.25), torch.full((1, 1), 0.25)],
        "vocabulary": torch.tensor([1]),
        "reward_threshold": 0.5,
    }
    zero_one = compute_nthr_group(response_scores=torch.tensor([1.0, 0.0]), **inputs)
    signed = compute_nthr_group(
        response_scores=torch.tensor([1.0, -1.0]),
        require_binary_rewards=False,
        **inputs,
    )

    assert zero_one.active and signed.active
    assert zero_one.eta == signed.eta
    torch.testing.assert_close(zero_one.tau, signed.tau)
    for zero_one_mask, signed_mask in zip(zero_one.selected, signed.selected, strict=True):
        assert torch.equal(zero_one_mask, signed_mask)


def test_nthr_rejects_binary_threshold_that_classifies_zero_as_positive():
    with pytest.raises(ValueError, match="reward_threshold"):
        compute_nthr_group(
            response_scores=torch.tensor([1.0, 0.0]),
            token_ids=[torch.tensor([1]), torch.tensor([1])],
            hidden_states=[torch.ones(1, 2), torch.ones(1, 2)],
            probabilities_on_vstar=[torch.full((1, 1), 0.25), torch.full((1, 1), 0.25)],
            vocabulary=torch.tensor([1]),
            reward_threshold=0.0,
        )


def test_prediction_error_does_not_renormalize_vocabulary_slice():
    probability_slice = torch.tensor([[0.2, 0.3]])
    error = prediction_error_on_vstar(
        probability_slice,
        sampled_token_ids=torch.tensor([5]),
        vocabulary=torch.tensor([5, 8]),
    )
    torch.testing.assert_close(error, torch.tensor([[0.8, -0.3]]))


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_chunked_logsumexp_matches_torch(chunk_size):
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 4, 11, generator=generator, dtype=torch.bfloat16)
    actual = chunked_logsumexp(logits, chunk_size=chunk_size)
    expected = torch.logsumexp(logits.float(), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
