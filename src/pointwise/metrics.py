"""Decision metrics and temperature fitting (Pointwise Phase 4). Pure Python on
plain lists of logits, so CI tests it without torch.

Every question is scored against its target distribution q:
    correct      argmax p == argmax q
    nll          -sum q_i log p_i          (log score; the training loss)
    brier        sum (p_i - q_i)^2
    confidence   max p                     (top-label ECE bins on this)
"""
from __future__ import annotations

import math

from src.evaluate_core import expected_calibration_error


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(z - peak) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def question_scores(logits: list[float], target: list[float], temperature: float = 1.0) -> dict:
    p = softmax(logits, temperature)
    best = max(range(len(p)), key=p.__getitem__)
    gold = max(range(len(target)), key=target.__getitem__)
    return {
        "correct": best == gold,
        "nll": -sum(q * math.log(max(pi, 1e-12)) for pi, q in zip(p, target) if q > 0),
        "brier": sum((pi - q) ** 2 for pi, q in zip(p, target)),
        "confidence": p[best],
    }


def summarise(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "accuracy": sum(s["correct"] for s in scores) / n,
        "nll": sum(s["nll"] for s in scores) / n,
        "brier": sum(s["brier"] for s in scores) / n,
        "ece": expected_calibration_error([s["confidence"] for s in scores], [s["correct"] for s in scores]),
    }


def mean_nll(logits: list[list[float]], targets: list[list[float]], temperature: float) -> float:
    return sum(question_scores(z, q, temperature)["nll"] for z, q in zip(logits, targets)) / len(logits)


def fit_temperature(
    logits: list[list[float]], targets: list[list[float]], lo: float = 0.2, hi: float = 10.0, steps: int = 40
) -> float:
    """The single T minimising mean NLL on (dev) questions, by golden-section
    search over log T. Dividing every logit by one T never changes an argmax,
    so accuracy is untouched; only the confidence is rescaled."""
    a, b = math.log(lo), math.log(hi)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = mean_nll(logits, targets, math.exp(c)), mean_nll(logits, targets, math.exp(d))
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = mean_nll(logits, targets, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = mean_nll(logits, targets, math.exp(d))
    return math.exp((a + b) / 2)
