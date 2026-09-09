# LUFFY OOD evaluation with vLLM

This is a small evaluator for the three multiple-choice OOD benchmarks bundled with LUFFY:

- ARC-Challenge: `data/valid.arc_c.parquet`
- GPQA-Diamond: `data/valid.gpqa.parquet`
- MMLU-Pro: `data/valid.mmlu_pro.parquet`

It loads the model only once, evaluates each selected benchmark, and reports both per-benchmark accuracy and overall macro/micro averages.

## Install

Use the CUDA/PyTorch environment in which vLLM works:

```bash
cd /home/anhld48/nthr/eval_arc/LUFFY/ood_eval
pip install -r requirements.txt
```

The full LUFFY environment already contains these dependencies.

## Run all three benchmarks

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_ood.py \
  --model Elliott/LUFFY-Qwen-Math-7B-Zero \
  --model-name luffy-7b \
  --benchmarks all \
  --temperature 0.6 \
  --top-p 1.0 \
  --max-tokens 8192
```

The shell launcher exposes the same common settings as environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=Elliott/LUFFY-Qwen-Math-7B-Zero \
MODEL_NAME=luffy-7b \
BENCHMARKS="arc_c gpqa mmlu_pro" \
TEMPERATURE=0.6 TOP_P=1.0 MAX_TOKENS=8192 \
bash run_ood.sh
```

Evaluate a subset with, for example, `--benchmarks gpqa mmlu_pro`. Accepted aliases include `arc-c`, `gpqa-diamond`, and `mmlu-pro`. `--limit 10` evaluates the first ten rows of each selected benchmark for a quick smoke test.

`--tensor-parallel-size 0` uses every GPU visible through `CUDA_VISIBLE_DEVICES`. Run `python eval_ood.py --help` for the other vLLM parameters, including `top_k`, `min_p`, `n`, seed, dtype, context length, and GPU-memory utilization.

The default `chat` prompt mode uses the model tokenizer's chat template. Use `--prompt-template qwen-math` for the fixed Qwen-Math-style prompt or `--prompt-template plain` for a base model without a chat template. The stored system message is removed by default to match LUFFY's published evaluation command; pass `--keep-system` to retain it.

## Output

Results go to `results/`:

- `<model-name>.ood.jsonl` contains every generation, its benchmark, extracted answer, gold answer, and correctness.
- `<model-name>.ood.summary.json` contains per-benchmark scores, macro/micro averages, parse rate, and the complete generation configuration.

The scorer handles answers `A`-`J` or `1`-`10` in `\\boxed{}`, `<answer>`, `final answer`, `answer is`, or as a standalone final line. This covers MMLU-Pro's ten choices without the heavyweight math grader used by LUFFY's general evaluator.
