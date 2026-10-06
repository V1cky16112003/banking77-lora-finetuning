"""The /v1/systemone contract: validation, rendering, confidence, answers.
Pure pydantic, no torch, so it runs in CI."""
import pytest
from pydantic import ValidationError

from src.pointwise.schema import (
    MAX_OPTIONS,
    SystemOneRequest,
    choice_confidence,
    question_options,
    render,
    round_prob,
    score_confidence,
    to_answers,
)


def _request(**questions):
    return SystemOneRequest.model_validate({"state": "s", "questions": questions})


def test_choice_option_count_is_capped():
    ok = {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS)}}
    _request(q=ok)
    too_many = {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS + 1)}}
    with pytest.raises(ValidationError):
        _request(q=too_many)
    with pytest.raises(ValidationError):
        _request(q={"type": "choice", "criteria": {}})


def test_unknown_question_type_and_empty_questions_rejected():
    with pytest.raises(ValidationError):
        _request(q={"type": "freeform", "criteria": {}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {}})


def test_render_flattens_objects_and_lists():
    state = {"ticket": "late shoes", "items": ["shoes", "socks"], "meta": {"priority": 2}}
    assert render(state) == (
        "ticket: late shoes\nitems:\n  - shoes\n  - socks\nmeta:\n  priority: 2"
    )
    assert render(None) == ""
    assert render(["a", "b"]) == "- a\n- b"


def test_question_options_per_type():
    req = _request(
        c={"type": "choice", "criteria": {"returns": "Refunds", "billing": None}},
        n={"type": "noul", "criteria": {"true": "it is urgent"}},
        s={"type": "score", "criteria": ["Calm", "Angry"]},
    )
    c, n, s = req.questions.values()
    assert question_options(c) == (["returns", "billing"], ["returns: Refunds", "billing"])
    assert question_options(n) == (["false", "true"], ["no", "yes: it is urgent"])
    assert question_options(s) == (["0", "1"], ["Calm", "Angry"])


@pytest.mark.parametrize(
    "p, want",
    [([0.25] * 4, 0.0), ([1.0, 0, 0, 0], 1.0), ([1.0], 1.0), ([0, 0], 0.0), ([0.5, 0.5, 0], 0.25)],
)
def test_choice_confidence(p, want):
    assert choice_confidence(p) == pytest.approx(want)


@pytest.mark.parametrize(
    "p, want", [([0, 1.0, 0], 1.0), ([1 / 3] * 3, 0.0), ([1.0], 1.0), ([0.5, 0, 0.5], 0.0)]
)
def test_score_confidence(p, want):
    assert score_confidence(p) == pytest.approx(want)


def test_round_prob_keeps_255_option_sum_within_tolerance():
    p = [1 / 255] * 255
    assert abs(sum(round_prob(x) for x in p) - 1) < 0.02


def test_to_answers_shapes():
    req = _request(
        c={"type": "choice", "criteria": {"a": None, "b": None, "c": None}},
        n={"type": "noul"},
        s={"type": "score", "criteria": ["low", "mid", "high"]},
    )
    answers = to_answers(req, [[0.1, 0.7, 0.2], [0.3, 0.7], [0.0, 0.5, 0.5]])
    assert answers["c"] == {
        "type": "choice",
        "choice": "b",
        "confidence": round_prob((0.7 - 1 / 3) / (1 - 1 / 3)),
        "probabilities": {"a": 0.1, "b": 0.7, "c": 0.2},
    }
    assert answers["n"] == {"type": "noul", "noul": 0.7}
    assert answers["s"]["score"] == 1.5
    assert answers["s"]["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert answers["s"]["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
