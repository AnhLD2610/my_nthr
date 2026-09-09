#!/usr/bin/env python3
"""Evaluate a model on LUFFY's ARC-C, GPQA, and MMLU-Pro splits with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_FILES = {
    "arc_c": "valid.arc_c.parquet",
    "gpqa": "valid.gpqa.parquet",
    "mmlu_pro": "valid.mmlu_pro.parquet",
}
ALIASES = {
    "arc": "arc_c",
    "arc-c": "arc_c",
    "arc_challenge": "arc_c",
    "gpqa-diamond": "gpqa",
    "gpqa_diamond": "gpqa",
    "mmlu-pro": "mmlu_pro",
}
CHOICE = r"(?:10|[1-9]|[A-J])"


def native(value: Any) -> Any:
    """Turn numpy/pyarrow values read from parquet into normal Python values."""
    if hasattr(value, "as_py"):
        return native(value.as_py())
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        return native(value.tolist())
    return value


def extract_choice(text: Any) -> str | None:
    """Return a choice A-J or 1-10, preferring the last explicit marker."""
    if text is None:
        return None
    text = str(text)
    patterns = (
        rf"\\boxed\s*\{{\s*\\(?:text|mathrm)\s*\{{\s*({CHOICE})\s*\}}\s*\}}",
        rf"\\boxed\s*\{{\s*({CHOICE})\s*\}}",
        rf"<answer>\s*(?:\\boxed\s*\{{\s*)?({CHOICE})(?:\s*\}})?\s*</answer>",
        rf"\bfinal\s+answer\s*(?:is|:|=)\s*(?:choice|option)?\s*[\[(]?({CHOICE})[\])]?(?=\W|$)",
        rf"\b(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:|=)\s*(?:choice|option)?\s*[\[(]?({CHOICE})[\])]?(?=\W|$)",
    )
    matches = [
        (match.start(), match.group(1).upper())
        for pattern in patterns
        for match in re.finditer(pattern, text, re.I | re.S)
    ]
    if matches:
        return max(matches, key=lambda item: item[0])[1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last_line = lines[-1] if lines else text.strip()
    last_line = re.sub(r"^(?:choice|option)\s+", "", last_line, flags=re.I)
    match = re.fullmatch(
        rf"[\s`*$]*[\[(]?\s*({CHOICE})\s*[\])]?[\s`*$.!]*", last_line, re.I
    )
    return match.group(1).upper() if match else None


def gold_choices(reward_model: Any) -> set[str]:
    value = native(reward_model)
    if isinstance(value, dict):
        value = value.get("ground_truth")
    values = value if isinstance(value, list) else [value]
    choices = {choice for item in values if (choice := extract_choice(item)) is not None}
    if not choices:
        raise ValueError(f"Could not parse multiple-choice ground truth from {value!r}")
    return choices


def prompt_messages(cell: Any, keep_system: bool) -> list[dict[str, str]]:
    value = native(cell)
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    elif isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Unsupported prompt type: {type(value).__name__}")

    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in value
    ]
    if not keep_system and messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if not messages:
        raise ValueError("Prompt is empty after removing its system message")
    return messages


def render_prompt(messages: list[dict[str, str]], tokenizer: Any, template: str) -> str:
    if template == "chat":
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Tokenizer has no usable chat template; pass --prompt-template plain"
            ) from exc

    if template == "qwen-math":
        question = next(
            (item["content"] for item in reversed(messages) if item["role"] == "user"),
            messages[-1]["content"],
        )
        return (
            "<|im_start|>system\nPlease reason step by step, and put your final answer "
            "within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            f"{question}<|im_end|>\n<|im_start|>assistant\n"
        )

    text = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )
    return f"{text}\n\nASSISTANT:"


def selected_benchmarks(names: list[str]) -> list[str]:
    if "all" in [name.lower() for name in names]:
        return list(DATA_FILES)
    selected = []
    for name in names:
        normalized = ALIASES.get(name.lower(), name.lower())
        if normalized not in DATA_FILES:
            valid = ", ".join(("all", *DATA_FILES))
            raise ValueError(f"Unknown benchmark {name!r}; choose from {valid}")
        if normalized not in selected:
            selected.append(normalized)
    return selected


def file_safe_name(name: str) -> str:
    name = name.rstrip("/").rsplit("/", 1)[-1] or "model"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def average(values: list[bool] | list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face ID or local path")
    parser.add_argument("--model-name", help="Name used for result files")
    parser.add_argument("--tokenizer", help="Optional tokenizer ID/path")
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--limit", type=int, default=0, help="First N rows per benchmark")
    parser.add_argument("--prompt-template", choices=("chat", "qwen-math", "plain"), default="chat")
    parser.add_argument("--keep-system", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop", action="append", default=[])

    parser.add_argument(
        "--tensor-parallel-size", type=int, default=0,
        help="0 uses all GPUs in CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--max-model-len", type=int, default=0,
        help="0 lets vLLM infer the context length",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    benchmarks = selected_benchmarks(args.benchmarks)
    if args.limit < 0 or args.n < 1 or args.max_tokens < 1:
        raise ValueError("--limit must be >= 0, and --n/--max-tokens must be >= 1")

    try:
        import pandas as pd
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            f"Missing {exc.name!r}; install ood_eval/requirements.txt in the vLLM environment"
        ) from exc

    jobs = {}
    for benchmark in benchmarks:
        input_file = args.data_dir / DATA_FILES[benchmark]
        if not input_file.is_file():
            raise FileNotFoundError(input_file)
        data = pd.read_parquet(input_file)
        missing = {"prompt", "reward_model"} - set(data.columns)
        if missing:
            raise ValueError(f"{input_file} is missing columns: {sorted(missing)}")
        if args.limit:
            data = data.head(args.limit)
        if data.empty:
            raise ValueError(f"{input_file} has no rows")
        jobs[benchmark] = {
            "messages": [prompt_messages(cell, args.keep_system) for cell in data["prompt"].tolist()],
            "answers": [gold_choices(cell) for cell in data["reward_model"].tolist()],
        }

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    tp_size = args.tensor_parallel_size or max(torch.cuda.device_count(), 1)
    engine = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len:
        engine["max_model_len"] = args.max_model_len
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "seed": args.seed,
        "stop": args.stop or None,
    }

    print(f"Model: {args.model}")
    print(f"Benchmarks: {', '.join(benchmarks)} | tensor parallel: {tp_size}")
    print("Sampling:", json.dumps(sampling))
    llm = LLM(**engine)
    params = SamplingParams(**sampling)

    name = file_safe_name(args.model_name or args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_file = args.output_dir / f"{name}.ood.jsonl"
    summary_file = args.output_dir / f"{name}.ood.summary.json"
    benchmark_results = {}
    all_correct: list[bool] = []
    all_parsed: list[bool] = []
    all_any_correct: list[bool] = []

    with details_file.open("w", encoding="utf-8") as handle:
        for benchmark, job in jobs.items():
            prompts = [
                render_prompt(messages, tokenizer, args.prompt_template)
                for messages in job["messages"]
            ]
            print(f"\nGenerating {benchmark}: {len(prompts)} questions")
            outputs = llm.generate(prompts, params)
            if len(outputs) != len(prompts):
                raise RuntimeError(f"{benchmark}: output count does not match prompt count")

            correct: list[bool] = []
            parsed: list[bool] = []
            any_correct: list[bool] = []
            for question_id, output in enumerate(outputs):
                if len(output.outputs) != args.n:
                    raise RuntimeError(f"{benchmark} question {question_id} did not return n={args.n}")
                question_scores = []
                for sample_id, completion in enumerate(output.outputs):
                    prediction = extract_choice(completion.text)
                    score = prediction in job["answers"][question_id]
                    correct.append(score)
                    parsed.append(prediction is not None)
                    question_scores.append(score)
                    record = {
                        "benchmark": benchmark,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "expected": sorted(job["answers"][question_id]),
                        "prediction": prediction,
                        "correct": score,
                        "prompt": output.prompt,
                        "generated_text": completion.text,
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                any_correct.append(any(question_scores))

            benchmark_results[benchmark] = {
                "questions": len(prompts),
                "generations": len(correct),
                "accuracy": average(correct),
                "pass_at_n": average(any_correct),
                "answer_parse_rate": average(parsed),
            }
            all_correct.extend(correct)
            all_parsed.extend(parsed)
            all_any_correct.extend(any_correct)
            print(f"{benchmark} accuracy: {average(correct):.4f} ({sum(correct)}/{len(correct)})")

    summary = {
        "model": args.model,
        "model_name": args.model_name or file_safe_name(args.model),
        "benchmarks": benchmark_results,
        "overall": {
            "questions": sum(result["questions"] for result in benchmark_results.values()),
            "generations": len(all_correct),
            "macro_accuracy": average([result["accuracy"] for result in benchmark_results.values()]),
            "micro_accuracy": average(all_correct),
            "macro_pass_at_n": average([result["pass_at_n"] for result in benchmark_results.values()]),
            "micro_pass_at_n": average(all_any_correct),
            "answer_parse_rate": average(all_parsed),
        },
        "samples_per_question": args.n,
        "data_dir": str(args.data_dir.resolve()),
        "prompt_template": args.prompt_template,
        "keep_system": args.keep_system,
        "sampling": sampling,
        "engine": {key: value for key, value in engine.items() if key != "model"},
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nMacro accuracy: {summary['overall']['macro_accuracy']:.4f}")
    print(f"Micro accuracy: {summary['overall']['micro_accuracy']:.4f}")
    print(f"Answer parse rate: {summary['overall']['answer_parse_rate']:.4f}")
    print(f"Predictions: {details_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
