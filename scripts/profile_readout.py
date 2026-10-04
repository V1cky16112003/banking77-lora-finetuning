"""Phase 0c: where does the logit read-out spend its time?
(docs/designs/minijev-phase0c-plan.md)

Measures the GPU's achieved fp16 matmul throughput, per-stage read-out timings
at several batch sizes, token positions per message, efficiency against the
roofline, the effect of merging LoRA into the base weights, and the top CUDA
kernels for one batch.

Usage (from the repo root):
    python -m scripts.profile_readout --adapter-dir <lora_dir> --limit 48
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from scripts.compare_readout import _load_model
from src.config import load_task_config
from src.data import load_banking77_splits
from src.readout import LabelScorer

BATCH_SIZES = (1, 2, 4, 8)
MERGED_BATCH_SIZE = 4
TOP_KERNELS = 15


def matmul_tflops(device: torch.device, size: int = 8192, reps: int = 20) -> float:
    """Achieved fp16 TFLOP/s on a large square matmul, the practical roofline."""
    if device.type != "cuda":
        return float("nan")
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    a @ b  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return 2 * size**3 * reps / (time.perf_counter() - start) / 1e12


def token_positions_per_message(scorer: LabelScorer, messages: list[str]) -> dict[str, float]:
    """Token positions the model processes per message, counted from the inputs."""
    suffix = sum(len(scorer._suffix_ids(m)) for m in messages) / len(messages)
    n_labels, label_len = scorer.labels.ids.shape
    return {
        "suffix": suffix,
        "label_pass_padded": n_labels * (label_len - 1),
        "label_pass_real": int(scorer.labels.mask[:, 1:].sum()),
        "prefix_once_per_run": len(scorer.prefix_ids),
    }


def profile_batches(scorer: LabelScorer, messages: list[str], batch_size: int) -> dict:
    """Timings for one batch size. An out-of-memory error is a result too
    (it bounds the usable batch size), so it is recorded rather than raised."""
    try:
        return _profile_batches(scorer, messages, batch_size)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "oom": str(exc).split(".")[0]}


def _profile_batches(scorer: LabelScorer, messages: list[str], batch_size: int) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scorer.score(messages[:batch_size])  # warm-up: kernels, allocator
    timings: dict[str, float] = {}
    start = time.perf_counter()
    for i in range(0, len(messages), batch_size):
        scorer.score(messages[i : i + batch_size], timings=timings)
    total = time.perf_counter() - start
    n = len(messages)
    return {
        "batch_size": batch_size,
        "ms_per_example": 1000 * total / n,
        "stage_ms_per_example": {k: 1000 * v / n for k, v in timings.items()},
        "stage_share": {k: v / sum(timings.values()) for k, v in timings.items()},
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }


def top_kernels(scorer: LabelScorer, messages: list[str]) -> list[dict]:
    if not torch.cuda.is_available():
        return []
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        scorer.score(messages)
        torch.cuda.synchronize()
    events = sorted(prof.key_averages(), key=lambda e: e.cuda_time_total, reverse=True)
    total = sum(e.self_cuda_time_total for e in events) or 1
    return [
        {
            "name": e.key[:90],
            "cuda_ms": e.cuda_time_total / 1000,
            "self_cuda_share": e.self_cuda_time_total / total,
            "calls": e.count,
        }
        for e in events[:TOP_KERNELS]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/profile"))
    args = parser.parse_args()

    config = load_task_config()
    splits = load_banking77_splits(config)
    messages = [r["text"] for r in splits.val][: args.limit]

    model, tokenizer = _load_model(args.adapter_dir)
    device = model.device
    report: dict = {
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
            "dtype": str(next(model.parameters()).dtype),
            "n_messages": len(messages),
        },
        "roofline_tflops": matmul_tflops(device),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "profile.json"

    def save() -> None:
        # Written after every step, so a crash late in the run keeps the earlier results.
        out_path.write_text(json.dumps(report, indent=2))

    save()
    scorer = LabelScorer(model, tokenizer, config)
    tokens = token_positions_per_message(scorer, messages)
    report["token_positions_per_message"] = tokens
    n_params = sum(p.numel() for p in model.parameters())
    report["model_params_billion"] = n_params / 1e9

    report["unmerged"] = []
    for batch_size in BATCH_SIZES:
        report["unmerged"].append(profile_batches(scorer, messages, batch_size))
        print(json.dumps(report["unmerged"][-1]), flush=True)
        save()
    report["top_kernels_b4"] = top_kernels(scorer, messages[:MERGED_BATCH_SIZE])
    save()

    if args.adapter_dir:
        merged = model.merge_and_unload().eval()
        report["merged_lora"] = profile_batches(
            LabelScorer(merged, tokenizer, config), messages, MERGED_BATCH_SIZE
        )
        save()

    # Efficiency: forward FLOPs ~ 2 * params per token position (attention ignored).
    completed = [r for r in report["unmerged"] if "oom" not in r]
    best = min(completed, key=lambda r: r["ms_per_example"])
    flops = 2 * n_params * (tokens["suffix"] + tokens["label_pass_padded"])
    achieved = flops / (best["ms_per_example"] / 1000) / 1e12
    report["efficiency"] = {
        "tflop_per_message": flops / 1e12,
        "achieved_tflops_at_best_batch": achieved,
        "fraction_of_roofline": achieved / report["roofline_tflops"],
        "best_batch_size": best["batch_size"],
    }

    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
