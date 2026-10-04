"""Pure evaluation logic: label resolution, metrics, confusion ranking.

Zero torch/transformers imports, by design (Architecture finding 1 / D6). This
is what CI (item 5) imports and tests — GitHub Actions has no GPU and no model
weights, so only this file's logic is CI-testable. evaluate.py (ML
orchestration) imports this module plus torch/transformers for the actual
generation calls.
"""
from __future__ import annotations

import difflib
import json
import math
import random
from collections import Counter
from pathlib import Path

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


# --- Calibration metrics (MiniJev Phase 0, docs/designs/minijev-phase0-plan.md) ---
# Pure-Python on purpose: these take plain lists of probabilities produced by
# src/readout.py, so CI can test them without torch.


def softmax_scores(scores: list[float]) -> list[float]:
    """Turn per-label log-likelihoods into a distribution over the label set.
    Subtracting the max first keeps exp() from overflowing on large-magnitude
    log-probs (sequence log-likelihoods are often -50 or lower)."""
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def brier_score(probs: list[list[float]], true_indices: list[int]) -> float:
    """Mean multiclass Brier score, ||p - e_y||^2, averaged over examples.
    Strictly proper: its expectation is uniquely minimised by reporting the
    true distribution, which is why RLCD-style training uses it. Range [0, 2]."""
    total = 0.0
    for p, y in zip(probs, true_indices):
        total += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
    return total / len(probs)


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 15
) -> float:
    """Top-label ECE: bin examples by confidence, then average |accuracy -
    mean confidence| per bin, weighted by bin size. 0 means "when it says 80%,
    it is right 80% of the time". Bins are (lo, hi], with confidence 0 put in
    the first bin."""
    n = len(confidences)
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        idx = min(n_bins - 1, max(0, int(c * n_bins - 1e-12)))
        bins[idx].append(i)

    ece = 0.0
    for members in bins:
        if not members:
            continue
        acc = sum(correct[i] for i in members) / len(members)
        conf = sum(confidences[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    return ece


def risk_coverage_curve(
    confidences: list[float], correct: list[bool]
) -> list[tuple[float, float]]:
    """(coverage, risk) points from accepting the k most-confident predictions,
    for k = 1..n. Risk is the error rate among the accepted predictions. This is
    the "how much can I automate at a given error budget?" view."""
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    n = len(order)
    points = []
    errors = 0
    for k, i in enumerate(order, start=1):
        errors += not correct[i]
        points.append((k / n, errors / k))
    return points


def aurc(confidences: list[float], correct: list[bool]) -> float:
    """Area under the risk-coverage curve (lower is better). Mean of the risk
    over all k, i.e. the uniform-coverage-step integral."""
    curve = risk_coverage_curve(confidences, correct)
    return sum(risk for _, risk in curve) / len(curve)


def coverage_at_risk(
    confidences: list[float], correct: list[bool], max_risk: float
) -> float:
    """Largest fraction of examples that can be auto-decided (most-confident
    first) while keeping the error rate among them <= max_risk. Cov@5% is the
    headline "delegable decisions" number from the OpenJev paper."""
    best = 0.0
    for coverage, risk in risk_coverage_curve(confidences, correct):
        if risk <= max_risk:
            best = coverage
    return best


def load_resumable_jsonl(path: Path) -> list[dict]:
    """Records already written to a predictions JSONL, for resuming a run.

    A Kaggle disconnect can kill the process halfway through writing a line.
    That torn last line is dropped and the file truncated back to the last
    whole record, so the resumed run appends cleanly instead of crashing on
    json.loads. A bad line anywhere else is real corruption and still raises."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for n, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return records
