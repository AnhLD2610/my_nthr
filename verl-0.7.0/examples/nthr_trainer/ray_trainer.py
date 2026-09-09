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
"""Ray trainer extension that applies NTHR to existing GRPO advantages."""

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class NTHRRayPPOTrainer(RayPPOTrainer):
    """Apply an actor-computed NTHR mask immediately before each actor update."""

    def _update_actor(self, batch: DataProto) -> DataProto:
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("NTHR is implemented only for algorithm.adv_estimator=grpo")

        nthr_config = self.config.algorithm.get("nthr", {})
        nthr_input = batch.select(
            batch_keys=[
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "response_mask",
                "token_level_scores",
            ],
            non_tensor_batch_keys=["uid"],
        )
        # DataProto.select shares meta_info by default; keep NTHR RPC metadata
        # from leaking into the actor-update batch.
        nthr_input.meta_info = dict(nthr_input.meta_info)
        nthr_input.meta_info.update(
            {
                "nthr_beta": float(nthr_config.get("beta", 1.0)),
                "nthr_reward_threshold": float(nthr_config.get("reward_threshold", 0.5)),
                "nthr_require_binary_rewards": bool(nthr_config.get("require_binary_rewards", True)),
                "nthr_binary_reward_atol": float(nthr_config.get("binary_reward_atol", 1e-6)),
                "nthr_forward_micro_batch_size": int(nthr_config.get("forward_micro_batch_size", 1)),
                "nthr_softmax_vocab_chunk_size": int(nthr_config.get("softmax_vocab_chunk_size", 16384)),
            }
        )
        nthr_output = self.actor_rollout_wg.compute_nthr_mask(nthr_input)

        advantages = batch.batch["advantages"]
        selected = nthr_output.batch["nthr_mask"].to(device=advantages.device, dtype=torch.bool)
        eta = nthr_output.batch["nthr_eta"].to(device=advantages.device, dtype=advantages.dtype)
        scale = torch.where(selected, eta.expand_as(advantages), 1.0)
        batch.batch["advantages"] = advantages * scale

        actor_output = super()._update_actor(batch)

        response_scores = batch.batch["token_level_scores"].sum(dim=-1)
        negative_rows = response_scores < float(nthr_config.get("reward_threshold", 0.5))
        negative_tokens = negative_rows.unsqueeze(-1) & batch.batch["response_mask"].bool()
        negative_tokens = negative_tokens.to(selected.device)
        selected_count = (selected & negative_tokens).sum().item()
        negative_count = negative_tokens.sum().item()
        active = nthr_output.batch["nthr_active"].squeeze(-1)
        finite_tau = nthr_output.batch["nthr_tau"].squeeze(-1)
        finite_tau = finite_tau[torch.isfinite(finite_tau)]
        nthr_metrics = {
            "nthr/selected_negative_tokens": selected_count,
            "nthr/negative_tokens": negative_count,
            "nthr/selected_fraction": selected_count / max(negative_count, 1),
            "nthr/active_rollout_fraction": active.float().mean().item(),
            "nthr/eta_mean": nthr_output.batch["nthr_eta"][active].float().mean().item()
            if active.any()
            else 1.0,
            "nthr/tau_mean": finite_tau.float().mean().item() if finite_tau.numel() else 0.0,
        }
        actor_output.meta_info.setdefault("metrics", {}).update(nthr_metrics)
        return actor_output
