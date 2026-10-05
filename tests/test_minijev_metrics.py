"""Decision metrics and temperature fitting. Pure Python, runs in CI."""
import math
import random

import pytest

from src.minijev import metrics as M


def test_softmax_and_temperature():
    assert M.softmax([0.0, 0.0]) == [0.5, 0.5]
    hot = M.softmax([2.0, 0.0], temperature=2.0)
    assert hot[0] == pytest.approx(1 / (1 + math.exp(-1)))


def test_question_scores_one_hot_and_soft():
    s = M.question_scores([math.log(3), 0.0], [1.0, 0.0])  # p = [0.75, 0.25]
    assert s["correct"] and s["confidence"] == pytest.approx(0.75)
    assert s["nll"] == pytest.approx(-math.log(0.75))
    assert s["brier"] == pytest.approx(0.25**2 * 2)
    soft = M.question_scores([0.0, 0.0], [0.5, 0.5])
    assert soft["nll"] == pytest.approx(math.log(2)) and soft["brier"] == pytest.approx(0.0)


def test_summarise():
    scores = [M.question_scores([1.0, 0.0], [1.0, 0.0]), M.question_scores([1.0, 0.0], [0.0, 1.0])]
    out = M.summarise(scores)
    assert out["n"] == 2 and out["accuracy"] == 0.5
    assert M.summarise([]) == {"n": 0}


def test_fit_temperature_recovers_overconfidence():
    """Labels drawn from softmax(z); the model reports 3z. The NLL-optimal fix is T = 3."""
    rng = random.Random(0)
    logits, targets = [], []
    for _ in range(4000):
        z = [rng.gauss(0, 1) for _ in range(4)]
        p = M.softmax(z)
        gold = rng.choices(range(4), weights=p)[0]
        logits.append([3 * v for v in z])
        targets.append([1.0 if i == gold else 0.0 for i in range(4)])
    assert M.fit_temperature(logits, targets) == pytest.approx(3.0, rel=0.1)


def test_temperature_never_changes_accuracy():
    logits = [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]
    targets = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    for t in (0.5, 1.0, 4.0):
        assert M.summarise([M.question_scores(z, q, t) for z, q in zip(logits, targets)])["accuracy"] == 0.5
