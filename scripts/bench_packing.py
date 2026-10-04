"""Phase 1 benchmark: one state, k questions (docs/designs/minijev-plan.md).

For k = 1, 4, 16, 64 questions on one state, times:
- separate:      k plain calls of "state + question" (the baseline);
- packed:        one pass, block mask (src/minijev/packing.py);
- branches:      prefill the state, then one pass over all questions;
- branches_warm: the same with the state already cached (a serving cache hit);
- rows:          prefill the state, then each question as its own row.

Also checks, in fp16 on the real model, how far packed features drift from the
separate calls. Gate: latency grows sub-linearly in k.

Runs the bare backbone (no vocabulary head): Phase 2 reads hidden states only.

Usage (from the repo root):
    python -m scripts.bench_packing --output-dir eval_runs/packing
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from src.config import load_task_config
from src.data import load_banking77_splits
from src.minijev.packing import (
    branch_features,
    pack,
    packed_features,
    prefill_state,
    row_features,
)

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
KS = (1, 4, 16, 64)
REPS = 5


def build_request(tokenizer, n_state_tokens: int, n_questions: int) -> tuple[list[int], list[list[int]]]:
    """A support-chat state of about n_state_tokens built from Banking77 val
    messages, and yes/no questions built from the label names."""
    config = load_task_config()
    splits = load_banking77_splits(config)
    lines, state_ids = [], []
    for row in splits.val:
        lines.append(f"Customer: {row['text']}")
        state_ids = tokenizer.encode("\n".join(lines) + "\n", add_special_tokens=False)
        if len(state_ids) >= n_state_tokens:
            break
    labels = [label.replace("_", " ") for label in config.labels]
    questions = [
        tokenizer.encode(
            f"Question: is the customer's main issue \"{labels[i % len(labels)]}\"? Answer yes or no.",
            add_special_tokens=False,
        )
        for i in range(n_questions)
    ]
    return state_ids, questions


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device: torch.device) -> float:
    """Median wall time over REPS runs after one warm-up, GPU-synchronised."""
    fn()
    samples = []
    for _ in range(REPS):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(1000 * (time.perf_counter() - start))
    return statistics.median(samples)


def separate_features(model, state, questions):
    out = []
    with torch.no_grad():
        for q in questions:
            hidden = model(input_ids=torch.tensor([state + q], device=model.device)).last_hidden_state
            out.append(hidden[0, len(state) :])
    return out


def max_drift(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Largest |packed - separate| over question tokens, relative to the feature scale."""
    diff = max((x.float() - y.float()).abs().max().item() for x, y in zip(a, b))
    scale = max(y.float().abs().max().item() for y in b)
    return diff / scale


def bench_k(model, state, questions) -> dict:
    device = model.device
    packed = pack(state, questions)
    warm_cache = prefill_state(model, state)
    methods = {
        "separate": lambda: separate_features(model, state, questions),
        "packed": lambda: packed_features(model, packed),
        "branches": lambda: branch_features(model, prefill_state(model, state), packed),
        "branches_warm": lambda: branch_features(model, warm_cache, packed),
        "rows": lambda: row_features(model, prefill_state(model, state), len(state), questions),
    }
    result: dict = {"k": len(questions), "packed_tokens": len(packed.ids), "ms": {}}
    for name, fn in methods.items():
        try:
            result["ms"][name] = time_ms(fn, device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["ms"][name] = "oom"
    result["drift_packed_vs_separate"] = max_drift(
        packed_features(model, packed), separate_features(model, state, questions)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--state-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/packing"))
    args = parser.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16  # T4: no bf16
    else:
        device, dtype = "cpu", torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()

    state, questions = build_request(tokenizer, args.state_tokens, max(KS))
    report: dict = {
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model": args.model,
            "dtype": str(dtype),
        },
        "state_tokens": len(state),
        "question_tokens_mean": sum(len(q) for q in questions) / len(questions),
        "results": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "packing_bench.json"

    for k in KS:
        report["results"].append(bench_k(model, state, questions[:k]))
        print(json.dumps(report["results"][-1]), flush=True)
        out_path.write_text(json.dumps(report, indent=2))  # after every k: a late crash keeps the rest

    # Gate: how latency grows from k=1 to the largest k, per method (linear = max(KS)).
    first, last = report["results"][0]["ms"], report["results"][-1]["ms"]
    report["growth_k1_to_kmax"] = {
        name: last[name] / first[name]
        for name in first
        if isinstance(first[name], float) and isinstance(last[name], float)
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
