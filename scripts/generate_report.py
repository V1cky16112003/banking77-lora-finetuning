"""Generates the README results tables, confusion heatmap, and bootstrap CI
from saved eval JSONL output (item 2, item 3, outside-voice E1). Avoids
hand-copying numbers into the README — eliminates transcription errors
between what the eval harness measured and what the README claims.

Usage: python scripts/generate_report.py <predictions.jsonl> --labels-config configs/task.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_task_config
from src.evaluate_core import (
    bootstrap_confidence_interval,
    compute_confusion_ranking,
    compute_macro_f1,
    compute_per_class_f1,
)

CONFUSION_TOP_N = 15  # top-15 to 20 classes by error volume, per D2's Temporal Interrogation decision


def load_predictions(path: Path) -> tuple[list[str], list[str | None]]:
    y_true, y_pred = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            y_true.append(record["true_label"])
            y_pred.append(record["predicted_label"])
    return y_true, y_pred


def render_results_table(y_true, y_pred, labels) -> str:
    macro_f1 = compute_macro_f1(y_true, y_pred, labels)
    point, lo, hi = bootstrap_confidence_interval(y_true, y_pred, labels)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Outside-voice E1: bootstrap CI on the headline metric, not just a point
    # estimate — the same ~40-examples/class fragility the heatmap caveat
    # (D9) covers applies to macro-F1 too.
    return (
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Accuracy | {accuracy:.3f} |\n"
        f"| Macro-F1 | {macro_f1:.3f} (95% CI: {lo:.3f}-{hi:.3f}, bootstrap n=1000) |\n"
    )


def render_per_class_table(y_true, y_pred, labels) -> str:
    per_class = compute_per_class_f1(y_true, y_pred, labels)
    rows = "\n".join(f"| {label} | {f1:.3f} |" for label, f1 in sorted(per_class.items()))
    return f"| Label | F1 |\n|---|---|\n{rows}\n"


def render_confusion_heatmap(y_true, y_pred, output_path: Path) -> None:
    ranking = compute_confusion_ranking(y_true, y_pred, top_n=CONFUSION_TOP_N)
    if not ranking:
        return

    involved_labels = sorted({t for t, _, _ in ranking} | {p for _, p, _ in ranking})
    matrix = np.zeros((len(involved_labels), len(involved_labels)))
    index = {label: i for i, label in enumerate(involved_labels)}
    for true_label, pred_label, count in ranking:
        matrix[index[true_label], index[pred_label]] = count

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(involved_labels)))
    ax.set_yticks(range(len(involved_labels)))
    ax.set_xticklabels(involved_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(involved_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Top-{CONFUSION_TOP_N} confused classes (by total misclassification involvement)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("--labels-config", type=Path, default=Path("configs/task.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/report"))
    args = parser.parse_args()

    config = load_task_config(args.labels_config)
    y_true, y_pred = load_predictions(args.predictions_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_table = render_results_table(y_true, y_pred, config.labels)
    (args.output_dir / "results_table.md").write_text(results_table)

    per_class_table = render_per_class_table(y_true, y_pred, config.labels)
    (args.output_dir / "per_class_f1_table.md").write_text(per_class_table)

    render_confusion_heatmap(y_true, y_pred, args.output_dir / "confusion_heatmap.png")

    print(f"Report written to {args.output_dir}/")


if __name__ == "__main__":
    main()
