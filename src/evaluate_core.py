"""Pure evaluation logic: label resolution, metrics, confusion ranking.

Zero torch/transformers imports, by design (Architecture finding 1 / D6). This
is what CI (item 5) imports and tests — GitHub Actions has no GPU and no model
weights, so only this file's logic is CI-testable. evaluate.py (ML
orchestration) imports this module plus torch/transformers for the actual
generation calls.
"""
from __future__ import annotations

import difflib
import random
from collections import Counter

from sklearn.metrics import f1_score

# Fuzzy-match threshold for difflib.get_close_matches (0-1, higher = stricter).
_FUZZY_CUTOFF = 0.75


def parse_label(generated_text: str, labels: list[str]) -> str | None:
    """Resolve a model's generated text to one of the canonical labels.

    Per D3: this fallback chain (exact match -> fuzzy match -> unresolved)
    exists for the case where the constrained-decoding mechanism itself fails
    to initialize and generation falls back to unconstrained text — NOT for
    "malformed constrained output", which cannot occur when the constraint
    successfully applies (constrained generation can only emit label tokens).

    Returns None if unresolved. Callers must count an unresolved result as
    incorrect against the true label — never drop it from the denominator.
    """
    cleaned = generated_text.strip().lower()

    for label in labels:
        if cleaned == label.strip().lower():
            return label

    close = difflib.get_close_matches(
        cleaned, [l.lower() for l in labels], n=1, cutoff=_FUZZY_CUTOFF
    )
    if close:
        matched_lower = close[0]
        for label in labels:
            if label.lower() == matched_lower:
                return label

    return None


def compute_macro_f1(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> float:
    """Macro-F1 (headline metric, per the design doc's Technical Notes) —
    treats all 77 classes equally regardless of size. Unresolved predictions
    (None) are passed through as a distinct wrong label so they always count
    against accuracy/F1, never silently dropped from the denominator."""
    y_pred_filled = [p if p is not None else "__UNRESOLVED__" for p in y_pred]
    return float(f1_score(y_true, y_pred_filled, labels=labels, average="macro", zero_division=0))


def compute_per_class_f1(
    y_true: list[str], y_pred: list[str | None], labels: list[str]
) -> dict[str, float]:
    """Per-class F1 breakdown for the README appendix table."""
    y_pred_filled = [p if p is not None else "__UNRESOLVED__" for p in y_pred]
    scores = f1_score(y_true, y_pred_filled, labels=labels, average=None, zero_division=0)
    return dict(zip(labels, (float(s) for s in scores)))


def compute_confusion_ranking(
    y_true: list[str], y_pred: list[str | None], top_n: int = 15
) -> list[tuple[str, str, int]]:
    """Top-N confused (true_label, predicted_label) pairs, ranked by total
    misclassification involvement (false positives + false negatives per
    class) — resolved decision from the outside-voice review: ranks classes
    causing the most overall confusion, not just false-negative count, so a
    frequently over-predicted class shows up too."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for true_label, pred_label in zip(y_true, y_pred):
        pred = pred_label if pred_label is not None else "__UNRESOLVED__"
        if pred != true_label:
            pair_counts[(true_label, pred)] += 1

    ranked = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
    return [(t, p, c) for (t, p), c in ranked[:top_n]]


def bootstrap_confidence_interval(
    y_true: list[str],
    y_pred: list[str | None],
    labels: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI on macro-F1 (outside-voice E1). Returns
    (point_estimate, ci_low, ci_high) at a 95% interval. Addresses the same
    statistical-fragility concern the confusion heatmap caveat (D9) covers —
    ~40 test examples/class means the headline metric needs an honest error
    bar, not just a point estimate."""
    rng = random.Random(seed)
    n = len(y_true)
    point_estimate = compute_macro_f1(y_true, y_pred, labels)

    resampled_scores = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        resample_true = [y_true[i] for i in indices]
        resample_pred = [y_pred[i] for i in indices]
        resampled_scores.append(compute_macro_f1(resample_true, resample_pred, labels))

    resampled_scores.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples) - 1
    return point_estimate, resampled_scores[lo_idx], resampled_scores[hi_idx]


def format_label_for_display(label: str) -> str:
    """Cosmetic-only reformatting for the demo UI (design review Pass 7) —
    the raw underscored label is still what's used internally and in eval;
    this only changes what's rendered on screen."""
    return label.replace("_", " ").title()
