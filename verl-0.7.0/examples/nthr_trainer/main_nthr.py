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
"""Entry point for the NTHR-on-GRPO example."""

import hydra
import ray

from examples.nthr_trainer.fsdp_worker import NTHRAsyncActorRolloutRefWorker
from examples.nthr_trainer.ray_trainer import NTHRRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer import main_ppo
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.ray_trainer import Role
from verl.utils.device import auto_set_device


class NTHRTaskRunner(TaskRunner):
    """Use the NTHR worker and trainer while reusing VERL's normal setup."""

    def add_actor_rollout_worker(self, config):
        worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if worker_impl not in {"auto", "enable"}:
            raise NotImplementedError("NTHR currently requires trainer.use_legacy_worker_impl=enable")
        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("NTHR currently supports only the fsdp and fsdp2 actor strategies")
        if config.trainer.balance_batch:
            raise ValueError("NTHR requires trainer.balance_batch=false so rollout groups stay intact")
        if config.algorithm.adv_estimator != "grpo":
            raise ValueError("NTHR requires algorithm.adv_estimator=grpo")
        if config.algorithm.use_kl_in_reward:
            raise ValueError("NTHR requires algorithm.use_kl_in_reward=false so GRPO uses binary outcome rewards")
        rollout_correction = config.algorithm.get("rollout_correction")
        if rollout_correction and rollout_correction.get("bypass_mode", False):
            raise NotImplementedError("NTHR does not support rollout-correction bypass mode")
        if config.actor_rollout_ref.rollout.n < 2:
            raise ValueError("NTHR requires actor_rollout_ref.rollout.n >= 2")

        actor_dp_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.data.train_batch_size % actor_dp_size != 0:
            raise ValueError(
                "NTHR requires data.train_batch_size to be divisible by the actor data-parallel size "
                f"({actor_dp_size})"
            )

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(NTHRAsyncActorRolloutRefWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return NTHRAsyncActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        # TaskRunner.run currently constructs RayPPOTrainer directly rather than
        # accepting a trainer class.  Patch that module-level binding only for
        # the lifetime of this remote task runner.
        original_trainer = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = NTHRRayPPOTrainer
        try:
            return super().run(config)
        finally:
            main_ppo.RayPPOTrainer = original_trainer


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(NTHRTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
