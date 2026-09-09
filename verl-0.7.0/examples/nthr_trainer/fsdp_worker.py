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
"""Legacy FSDP worker extension that computes NTHR masks on actor ranks."""

from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from examples.nthr_trainer.nthr import chunked_logsumexp, compute_nthr_group
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.fsdp_workers import (
    AsyncActorRolloutRefWorker,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)


def _get_output_embedding(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = getattr(model, "module", model)
    if not hasattr(unwrapped, "get_output_embeddings"):
        raise TypeError("NTHR requires a Hugging Face causal LM with get_output_embeddings()")
    output_embedding = unwrapped.get_output_embeddings()
    if output_embedding is None:
        raise TypeError("The actor model did not expose an output embedding module")
    return output_embedding


def _any_rank_has(local_error: bool) -> bool:
    """Synchronize pre-forward validation so one bad rank cannot deadlock FSDP."""

    error = torch.tensor(int(local_error), dtype=torch.int32, device=get_device_id())
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return bool(error.item())


class NTHRAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Add an RPC that returns a fixed NTHR mask for an actor batch."""

    def _validate_nthr_worker_config(self, data: DataProto) -> None:
        if not self._is_actor:
            raise RuntimeError("compute_nthr_mask may only run on the actor worker")
        if self.ulysses_sequence_parallel_size != 1:
            raise NotImplementedError("The NTHR example currently requires actor.ulysses_sequence_parallel_size=1")
        if self.actor.use_fused_kernels:
            raise NotImplementedError("The NTHR example currently requires actor.use_fused_kernels=false")
        if "multi_modal_inputs" in data.non_tensor_batch:
            raise NotImplementedError("The NTHR example currently supports text-only causal language models")
        if self.config.rollout.multi_turn.enable:
            raise NotImplementedError("The NTHR example currently supports single-turn rollouts only")
        if data.batch["position_ids"].ndim != 2:
            raise NotImplementedError("The NTHR example does not yet support multimodal position ids")

    def _forward_nthr_features(
        self,
        data: DataProto,
        response_indices: list[int],
        vocabulary: torch.Tensor,
        micro_batch_size: int,
        softmax_vocab_chunk_size: int,
        collect_features: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Collect valid completion hidden states and probabilities for a group."""

        actor_module = self.actor.actor_module
        actor_module.eval()
        response_length = data.batch["responses"].shape[1]
        temperature = float(self.config.rollout.temperature)
        if temperature <= 0:
            raise ValueError(f"rollout temperature must be positive, got {temperature}")

        hidden_by_response: list[torch.Tensor] = []
        probabilities_by_response: list[torch.Tensor] = []
        output_embedding = _get_output_embedding(actor_module) if collect_features else None
        device = get_device_id()
        vocabulary_device = vocabulary.to(device=device)

        for offset in range(0, len(response_indices), micro_batch_size):
            chunk_indices = response_indices[offset : offset + micro_batch_size]
            index = torch.tensor(chunk_indices, dtype=torch.long)
            input_ids = data.batch["input_ids"].index_select(0, index).to(device)
            attention_mask = data.batch["attention_mask"].index_select(0, index).to(device)
            position_ids = data.batch["position_ids"].index_select(0, index).to(device)
            response_mask = data.batch["response_mask"].index_select(0, index).to(device).bool()

            captured_hidden: list[torch.Tensor] = []

            def capture_lm_head_input(_module, args):
                if not args:
                    raise RuntimeError("The actor output embedding received no hidden-state input")
                captured_hidden.append(args[0])

            hook = (
                output_embedding.register_forward_pre_hook(capture_lm_head_input)
                if output_embedding is not None
                else None
            )
            try:
                autocast = (
                    torch.autocast(device_type=self.actor.device_name, dtype=self.actor.param_dtype)
                    if self.actor.device_name != "cpu"
                    else nullcontext()
                )
                with torch.no_grad(), autocast:
                    output = actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
            finally:
                if hook is not None:
                    hook.remove()

            # Homogeneous reward groups use ordinary GRPO. They still execute
            # the forward so every FSDP rank makes the same collective calls.
            if not collect_features:
                del output, input_ids, attention_mask, position_ids, response_mask, captured_hidden
                continue

            if len(captured_hidden) != 1:
                raise RuntimeError(
                    "Expected the language-model output embedding to run exactly once, "
                    f"but captured {len(captured_hidden)} calls"
                )
            logits = output.logits[:, -response_length - 1 : -1, :]
            prediction_hidden = captured_hidden[0][:, -response_length - 1 : -1, :]
            logits.div_(temperature)

            # q[V*] = exp(logit[V*] - logsumexp(all logits)).  This keeps the
            # full-vocabulary denominator without materializing full probs. The
            # denominator is accumulated in FP32 chunks to limit peak memory.
            log_normalizer = chunked_logsumexp(logits, softmax_vocab_chunk_size).unsqueeze(-1)
            selected_logits = logits.index_select(dim=-1, index=vocabulary_device).to(torch.float32)
            probabilities = torch.exp(selected_logits - log_normalizer)

            for row in range(len(chunk_indices)):
                valid = response_mask[row]
                hidden_by_response.append(prediction_hidden[row, valid].detach())
                probabilities_by_response.append(probabilities[row, valid].detach())

            del (
                output,
                logits,
                prediction_hidden,
                log_normalizer,
                selected_logits,
                probabilities,
                captured_hidden,
                input_ids,
                attention_mask,
                position_ids,
                response_mask,
            )

        return hidden_by_response, probabilities_by_response

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_nthr_mask(self, data: DataProto) -> DataProto:
        """Compute the NTHR selection mask from the current (old) actor snapshot."""

        self._validate_nthr_worker_config(data)
        required_batch_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "token_level_scores",
        }
        missing = required_batch_keys - set(data.batch.keys())
        if missing:
            raise KeyError(f"NTHR input is missing batch keys: {sorted(missing)}")
        if "uid" not in data.non_tensor_batch:
            raise KeyError("NTHR requires data.non_tensor_batch['uid'] to identify rollout groups")

        beta = float(data.meta_info.get("nthr_beta", 1.0))
        reward_threshold = float(data.meta_info.get("nthr_reward_threshold", 0.5))
        require_binary_rewards = bool(data.meta_info.get("nthr_require_binary_rewards", True))
        binary_reward_atol = float(data.meta_info.get("nthr_binary_reward_atol", 1e-6))
        micro_batch_size = int(data.meta_info.get("nthr_forward_micro_batch_size", 1))
        softmax_vocab_chunk_size = int(data.meta_info.get("nthr_softmax_vocab_chunk_size", 16384))
        if beta < 0:
            raise ValueError("algorithm.nthr.beta must be non-negative")
        if binary_reward_atol < 0:
            raise ValueError("algorithm.nthr.binary_reward_atol must be non-negative")
        if micro_batch_size <= 0:
            raise ValueError("algorithm.nthr.forward_micro_batch_size must be positive")
        if softmax_vocab_chunk_size <= 0:
            raise ValueError("algorithm.nthr.softmax_vocab_chunk_size must be positive")

        uids = np.asarray(data.non_tensor_batch["uid"])
        grouped_indices: OrderedDict[object, list[int]] = OrderedDict()
        for row, uid in enumerate(uids):
            grouped_indices.setdefault(uid, []).append(row)

        expected_group_size = int(self.config.rollout.n)
        malformed = {uid: len(rows) for uid, rows in grouped_indices.items() if len(rows) != expected_group_size}
        if _any_rank_has(bool(malformed)):
            raise RuntimeError(
                "NTHR rollout groups were split across actor data-parallel ranks. "
                f"Expected {expected_group_size} rows per uid; local malformed groups: {malformed}. "
                "Use trainer.balance_batch=false and make data.train_batch_size divisible by actor DP size."
            )

        batch_size, response_length = data.batch["responses"].shape
        nthr_mask = torch.zeros((batch_size, response_length), dtype=torch.bool)
        nthr_eta = torch.ones((batch_size, 1), dtype=torch.float32)
        nthr_active = torch.zeros((batch_size, 1), dtype=torch.bool)
        nthr_tau = torch.full((batch_size, 1), torch.nan, dtype=torch.float32)
        response_scores = data.batch["token_level_scores"].sum(dim=-1).to(torch.float32)

        has_empty_response = (~data.batch["response_mask"].bool().any(dim=-1)).any().item()
        if _any_rank_has(has_empty_response):
            raise ValueError("NTHR encountered a response with no valid completion tokens")
        if require_binary_rewards:
            if not 0.0 < reward_threshold <= 1.0:
                raise ValueError("reward_threshold must be in (0, 1] when binary rewards are required")
            close_to_zero = torch.isclose(
                response_scores, torch.zeros_like(response_scores), atol=binary_reward_atol, rtol=0.0
            )
            close_to_one = torch.isclose(
                response_scores, torch.ones_like(response_scores), atol=binary_reward_atol, rtol=0.0
            )
            local_nonbinary = not torch.all(close_to_zero | close_to_one).item()
            if _any_rank_has(local_nonbinary):
                local_values = response_scores[~(close_to_zero | close_to_one)].tolist()
                raise ValueError(
                    "NTHR requires binary response rewards (0 or 1); "
                    f"local invalid values: {local_values}"
                )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        try:
            # Every rank executes the same number of forwards (one full group at
            # a time), even for non-mixed groups.  This is required by FSDP.
            with self.ulysses_sharding_manager:
                for group_rows in grouped_indices.values():
                    group_index = torch.tensor(group_rows, dtype=torch.long)
                    group_responses = data.batch["responses"].index_select(0, group_index)
                    group_mask = data.batch["response_mask"].index_select(0, group_index).bool()
                    valid_group_tokens = group_responses[group_mask]
                    vocabulary = torch.unique(valid_group_tokens, sorted=True)
                    group_scores = response_scores.index_select(0, group_index)
                    positive = group_scores >= reward_threshold
                    is_mixed_group = positive.any().item() and not positive.all().item()

                    hidden, probabilities = self._forward_nthr_features(
                        data=data,
                        response_indices=group_rows,
                        vocabulary=vocabulary,
                        micro_batch_size=micro_batch_size,
                        softmax_vocab_chunk_size=softmax_vocab_chunk_size,
                        collect_features=is_mixed_group,
                    )
                    if not is_mixed_group:
                        continue

                    token_ids = [
                        group_responses[row, group_mask[row]].to(hidden[row].device)
                        for row in range(len(group_rows))
                    ]
                    result = compute_nthr_group(
                        response_scores=group_scores.to(hidden[0].device),
                        token_ids=token_ids,
                        hidden_states=hidden,
                        probabilities_on_vstar=probabilities,
                        vocabulary=vocabulary.to(hidden[0].device),
                        beta=beta,
                        reward_threshold=reward_threshold,
                        require_binary_rewards=require_binary_rewards,
                        binary_reward_atol=binary_reward_atol,
                    )

                    for local_row, batch_row in enumerate(group_rows):
                        valid_positions = torch.nonzero(group_mask[local_row], as_tuple=False).squeeze(-1)
                        nthr_mask[batch_row, valid_positions] = result.selected[local_row].cpu()
                        nthr_eta[batch_row, 0] = result.eta
                        nthr_active[batch_row, 0] = result.active
                        if result.tau is not None:
                            nthr_tau[batch_row, 0] = result.tau.cpu()
                    del hidden, probabilities, token_ids, result
        finally:
            if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
                self.actor.actor_module._handle.reshard(True)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()

        return DataProto.from_dict(
            tensors={
                "nthr_mask": nthr_mask,
                "nthr_eta": nthr_eta,
                "nthr_active": nthr_active,
                "nthr_tau": nthr_tau,
            }
        )
