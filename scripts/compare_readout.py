"""Phase 0 gate check: logit read-out vs constrained generation on the same
split, same model, same prompt (docs/designs/minijev-phase0-plan.md).

Reports accuracy, macro-F1, calibration (ECE, Brier, AURC, Cov@5%) and
per-example latency. Generation yields no probabilities, so it only gets the
accuracy and latency columns.

Usage (from the repo root):
    python -m scripts.compare_readout --adapter-dir <lora_dir> --split val --limit 500
    python -m scripts.compare_readout --split val --limit 50      # base model, smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_task_config
from src.data import load_banking77_splits
from src.evaluate import _GENERATE_BATCH_SIZE, evaluate_dataset
from src.evaluate_core import (
    aurc,
    brier_score,
    compute_macro_f1,
    coverage_at_risk,
    expected_calibration_error,
)
from src.readout import readout_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # must match src/train.py
COVERAGE_MAX_RISK = 0.05


def _load_model(adapter_dir: str | None):
    if torch.cuda.is_available():
        # T4 has no bf16; fp16 matches what train.py used on Kaggle.
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), tokenizer


def _timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", default=None, help="LoRA adapter; omit for base model")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None, help="first N examples only")
    parser.add_argument("--batch-size", type=int, default=4, help="messages per read-out batch")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/readout"))
    args = parser.parse_args()

    config = load_task_config()
    splits = load_banking77_splits(config)
    dataset = splits.val if args.split == "val" else splits.test
    examples = [{"text": r["text"], "label": splits.labels[r["label"]]} for r in dataset]
    if args.limit:
        examples = examples[: args.limit]

    model, tokenizer = _load_model(args.adapter_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readout_path = args.output_dir / f"{args.split}_readout.jsonl"
    generate_path = args.output_dir / f"{args.split}_generate.jsonl"
    # Fresh run each time: resumed files would make the latency numbers meaningless.
    readout_path.unlink(missing_ok=True)
    generate_path.unlink(missing_ok=True)

    records, readout_secs = _timed(
        lambda: readout_dataset(model, tokenizer, examples, config, readout_path, args.batch_size)
    )
    generated, generate_secs = _timed(
        lambda: evaluate_dataset(model, tokenizer, examples, config, generate_path)
    )

    y_true = [r["true_label"] for r in records]
    ro_pred = [r["predicted_label"] for r in records]
    ro_conf = [r["confidence"] for r in records]
    ro_correct = [p == t for p, t in zip(ro_pred, y_true)]
    true_idx = [config.labels.index(t) for t in y_true]
    gen_pred = [p for _, p in generated]
    n = len(examples)

    summary = {
        "n": n,
        "split": args.split,
        "adapter_dir": args.adapter_dir,
        # Both batch sizes are recorded because ms_per_example depends on them:
        # generation batches 16 prompts, the read-out --batch-size messages.
        "readout_batch_size": args.batch_size,
        "generate_batch_size": _GENERATE_BATCH_SIZE,
        "readout": {
            "accuracy": sum(ro_correct) / n,
            "macro_f1": compute_macro_f1(y_true, ro_pred, config.labels),
            "ece": expected_calibration_error(ro_conf, ro_correct),
            "brier": brier_score([r["probs"] for r in records], true_idx),
            "aurc": aurc(ro_conf, ro_correct),
            f"coverage_at_{COVERAGE_MAX_RISK:.0%}_risk": coverage_at_risk(
                ro_conf, ro_correct, COVERAGE_MAX_RISK
            ),
            "ms_per_example": 1000 * readout_secs / n,
        },
        "generate": {
            "accuracy": sum(p == t for p, t in zip(gen_pred, y_true)) / n,
            "macro_f1": compute_macro_f1(y_true, gen_pred, config.labels),
            "ms_per_example": 1000 * generate_secs / n,
        },
        "agreement": sum(a == b for a, b in zip(ro_pred, gen_pred)) / n,
    }
    summary["speedup"] = summary["generate"]["ms_per_example"] / summary["readout"]["ms_per_example"]

    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
