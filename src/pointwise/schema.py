"""The /v1/systemone request and response contract (Pointwise Phase 2).

Matches TypeSafe's public System One API, as reproduced by Kev's kev/api.py
against TypeSafe's reference adapter (system-one-adapter 0.2.1), so a Jev or
Kev client can call a Pointwise server unchanged. Pure Python + pydantic, no
torch, so CI can test it.

Every question type becomes one primitive, "pick one of these options":
    choice -> its criteria names (option text "name" or "name: description")
    noul   -> two options, [no, yes]; the answer is p(yes)
    score  -> the ordered level descriptions; the answer is the expected level
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None  # optional {"true": ..., "false": ...}


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]  # {option name: description or None}

    @model_validator(mode="after")
    def _option_count(self) -> Choice:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"choice criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)  # ordered levels


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "pointwise-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(value: JSONContent, indent: int = 0) -> str:
    """Flatten a str / object / array state into the text the model reads.
    Object keys stay as labels ("key: value"); nesting is indented."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def _option_text(name: str, description: JSONContent) -> str:
    return name if description in (None, "") else f"{name}: {render(description)}"


def question_options(question: Question) -> tuple[list[str], list[str]]:
    """(keys the probabilities are reported under, option texts the model reads),
    both in option order."""
    if question.type == "choice":
        keys = list(question.criteria)
        return keys, [_option_text(k, v) for k, v in question.criteria.items()]
    if question.type == "noul":
        criteria = question.criteria or {}
        return ["false", "true"], [
            _option_text("no", criteria.get("false")),
            _option_text("yes", criteria.get("true")),
        ]
    texts = [render(level) for level in question.criteria]
    return [str(i) for i in range(len(texts))], texts


# Both confidence formulas follow TypeSafe's reference adapter (via Kev): p is
# renormalised first (all zeros -> uniform), and a single option is certain.
def _normalise(p: list[float]) -> list[float]:
    total = sum(p)
    return [1 / len(p)] * len(p) if total == 0 else [x / total for x in p]


def choice_confidence(p: list[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty. Unlike max(p),
    it means the same thing for 2 options as for 200."""
    k = len(p)
    return 1.0 if k == 1 else (max(_normalise(p)) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """max(0, 1 - E|level - mode| / D), where D is the mean absolute deviation
    of a uniform distribution over the levels. 1 when all mass is on one level,
    0 at uniform or anything as spread out."""
    n = len(p)
    if n == 1:
        return 1.0
    p = _normalise(p)
    mode = max(range(n), key=p.__getitem__)
    spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / spread)


def round_prob(x: float) -> float:
    """4 decimals keeps a rounded distribution's sum within TypeSafe's tolerance
    (|sum - 1| < 0.02) even at 255 options: 255 * 0.00005 < 0.02."""
    return round(float(x), 4)


def to_answers(request: SystemOneRequest, probs: list[list[float]]) -> dict:
    """Per-question probabilities (in request order) -> the response's answers."""
    answers = {}
    for (key, question), p in zip(request.questions.items(), probs):
        names, texts = question_options(question)
        if question.type == "noul":
            answers[key] = {"type": "noul", "noul": round_prob(p[1])}
        elif question.type == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            answers[key] = {
                "type": "choice",
                "choice": names[best],
                "confidence": round_prob(choice_confidence(p)),
                "probabilities": {n: round_prob(v) for n, v in zip(names, p)},
            }
        else:
            answers[key] = {
                "type": "score",
                "score": round_prob(sum(i * pi for i, pi in enumerate(p))),
                "confidence": round_prob(score_confidence(p)),
                "legend": dict(zip(names, texts)),
                "probabilities": {n: round_prob(v) for n, v in zip(names, p)},
            }
    return answers
