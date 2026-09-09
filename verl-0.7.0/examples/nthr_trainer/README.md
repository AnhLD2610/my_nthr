# NTHR for VERL GRPO

This example implements the method in [`examples/nthr.md`](../nthr.md) without
changing VERL's standard GRPO loss. It computes a fixed token-selection mask from
the old actor snapshot, scales only selected negative-token advantages by
`eta = 2 * abs(0.5 - p)`, and then calls the normal actor update.

## Run

The provided launcher mirrors the Qwen2.5-Math-7B/DAPO-Math-17k GRPO
configuration used by the comparison run:

```bash
bash examples/nthr_trainer/run_qwen2_5_0_5b_gsm8k.sh
```

The checked-in data defaults point to the original experiment filesystem. Set
the training and validation parquet paths for the machine that runs the job:

```bash
TRAIN_FILE=/data/dapo-math-17k.boxed.parquet \
TEST_FILE=/data/aime-2024.boxed.parquet \
MODEL_PATH=/models/Qwen2.5-Math-7B \
  bash examples/nthr_trainer/run_qwen2_5_0_5b_gsm8k.sh trainer.total_epochs=1
```

The launcher uses VERL's built-in `math_dapo` scorer with strict boxed-answer
verification. It does not load a custom reward function. Since that scorer
returns `+1/-1`, the launcher sets `require_binary_rewards=false` and uses
`reward_threshold=0.5`; this keeps the positive/negative split identical to a
correctness reward. The DAPO overlong penalty is configured but disabled to
match the comparison run.

## Design

The NTHR numerical kernel is in `nthr.py` and is CPU-testable. The custom FSDP
worker performs an additional no-grad actor forward immediately before the actor
update. At that point the actor parameters are still the same snapshot used to
compute `old_log_probs`; doing the computation on the worker avoids transferring
hidden states or vocabulary probability matrices through Ray.

For every prompt group the worker:

1. Builds `V*` from valid generated completion tokens.
2. Captures the final hidden states entering the language-model head.
3. Computes `q[V*]` with the full-vocabulary softmax denominator, without
   materializing a full probability tensor.
4. Accumulates `M+`, positive anchors, `tau`, and negative scores in FP32.
5. Returns only the boolean selection mask and per-group `eta` to the trainer.

`forward_micro_batch_size=1` bounds the peak full-logits allocation. Increasing
it can improve throughput when memory permits. The full-vocabulary softmax
denominator is accumulated in FP32 blocks controlled by
`softmax_vocab_chunk_size` (default `16384`).

## Current scope and correctness constraints

- Binary outcome rewards (0/1) are required by default. Set
  `algorithm.nthr.require_binary_rewards=false` only when
  `algorithm.nthr.reward_threshold` is a valid classification boundary.
- `algorithm.use_kl_in_reward=false` is required so the GRPO group advantage is
  computed from the same binary outcome used to identify positive responses.
- Text-only Hugging Face causal LMs with legacy VERL FSDP/FSDP2 workers are
  supported. Megatron, the new model-engine worker, multimodal models, fused
  kernels, multi-turn rollouts, and Ulysses sequence parallelism are rejected
  explicitly.
- `trainer.balance_batch=false` is required. A complete rollout group must land
  on one actor data-parallel rank, so `data.train_batch_size` must also be
  divisible by the actor DP size.
- NTHR uses completion tokens only. Prompt and padding positions never enter the
  positive aggregate or anchor averages.
- The vocabulary slice is not renormalized, matching the method note.
