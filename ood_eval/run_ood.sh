#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# BENCHMARKS may be "all" or a space-separated subset such as "gpqa mmlu_pro".
MODEL="${MODEL:-Elliott/LUFFY-Qwen-Math-7B-Zero}"
MODEL_NAME="${MODEL_NAME:-luffy-qwen-math-7b-zero}"
BENCHMARKS="${BENCHMARKS:-all}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
N="${N:-1}"
SEED="${SEED:-0}"
TP_SIZE="${TP_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
read -r -a BENCHMARK_ARGS <<< "$BENCHMARKS"

exec "${PYTHON_BIN:-python}" "$SCRIPT_DIR/eval_ood.py" \
  --model "$MODEL" \
  --model-name "$MODEL_NAME" \
  --benchmarks "${BENCHMARK_ARGS[@]}" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --n "$N" \
  --seed "$SEED" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "$@"
