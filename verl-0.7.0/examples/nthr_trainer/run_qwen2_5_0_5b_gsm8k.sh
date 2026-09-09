#!/usr/bin/env bash
#
# NTHR run — Qwen2.5-Math-7B on DAPO-Math-17k, ported from the GCCR-GRPO baseline
# (run_qwen3_4b_grpo_baseline.sh) settings onto examples.nthr_trainer.main_nthr.
#
set -xeuo pipefail

# ─────────────────────────── Environment ─────────────────────────
export NCCL_NVLS_ENABLE=0
export TENSORBOARD_DIR="/prj/corp/airesearch/lasvegas/vol20-scratch/lducanh/diverse_grpo/verl-0.7.0/examples/coverage_weighted_nash_grpo/tensorboard/qwen25_7b_math_nthr_8192_nopen"

# ─────────────────────── Pre-run GPU cleanup ─────────────────────
if [ "${SKIP_CLEANUP:-0}" != "1" ]; then
  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f 'ray::'      2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm'       2>/dev/null || true
  pkill -9 -f 'main_nthr'  2>/dev/null || true
  rm -rf /tmp/torchinductor_tranao 2>/dev/null || true
  sleep 3
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv 2>/dev/null || true
fi

# ─────────────────────────── Model ───────────────────────────
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}

# ──────────────────── Algorithm Setting ──────────────────────
n=8
lr=1e-6
clip_ratio_low=0.2
clip_ratio_high=0.28
use_kl_loss=false
kl_loss_coef=0.000
entropy_coeff=0
use_kl_in_reward=false

# Sequence lengths
max_prompt_length=2048
max_response_length=8192
val_response_length=20480
total_training_steps=2000

# Keep both DAPO overlong mechanisms disabled, matching the reference GRPO run.
# The values are still passed because DAPORewardManager requires a complete
# overlong-buffer configuration even when it is disabled.
enable_overlong_buffer=false
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=0.0

# ──────────────────── Training Setting ───────────────────────
NGPUS=8
train_prompt_bsz=256
train_prompt_mini_bsz=64
tp_size=2
gpu_mem_util=0.8

ppo_max_token_len_per_gpu=12000  # = max_prompt_length + max_response_length

total_epochs=32
save_freq=20
test_freq=20
val_before_train=false
max_ckpt_to_keep=80

# ─────────────────────────── Logging ─────────────────────────
project_name='cwn_grpo_dapo17k'
experiment_name='qwen25_7b_math_nthr_8192_nopen'

LOG_DIR=${LOG_DIR:-/prj/corp/airesearch/lasvegas/vol20-scratch/lducanh/diverse_grpo/verl-0.7.0/examples/coverage_weighted_nash_grpo/logs}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${experiment_name}_$(date +%Y%m%d_%H%M%S).txt}
echo "=== training log -> ${LOG_FILE} ==="

# ──────────────────── Training Data ──────────────────────────
train_path=${TRAIN_FILE:-/prj/corp/airesearch/lasvegas/vol20-scratch/lducanh/diverse_grpo/verl-0.7.0/examples/coverage_weighted_nash_grpo/data/dapo-math-17k.boxed.parquet}
test_path=${TEST_FILE:-/prj/corp/airesearch/lasvegas/vol20-scratch/lducanh/diverse_grpo/verl-0.7.0/examples/coverage_weighted_nash_grpo/data/aime-2024.boxed.parquet}
train_files="['${train_path}']"
test_files="['${test_path}']"

# ─────────────────────── Reward ───────────────────────
# Use VERL's built-in math_dapo scorer with strict \boxed{} verification and
# the same (disabled) DAPO overlong configuration as the reference GRPO run.
# math_dapo emits +1/-1, so reward_threshold=0.5 still separates correct and
# incorrect responses while the literal 0/1 validation must be disabled.

# ═════════════════════════════════════════════════════════════
# Launch
# ═════════════════════════════════════════════════════════════
python3 -m examples.nthr_trainer.main_nthr \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.nthr.beta=1.0 \
    +algorithm.nthr.reward_threshold=0.5 \
    +algorithm.nthr.require_binary_rewards=false \
    +algorithm.nthr.binary_reward_atol=1e-6 \
    +algorithm.nthr.forward_micro_batch_size=1 \
    +algorithm.nthr.softmax_vocab_chunk_size=16384 \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.val_kwargs.response_length=${val_response_length} \
    reward_manager.name=dapo \
    +reward_model.reward_kwargs.strict_box_verify=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=true \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.use_legacy_worker_impl=enable \
    trainer.balance_batch=false \
    trainer.critic_warmup=0 \
    trainer.val_before_train=${val_before_train} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=auto \
    trainer.total_epochs=${total_epochs} \
    "$@" 2>&1 | tee "${LOG_FILE}"
