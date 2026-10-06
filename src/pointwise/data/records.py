"""Record format, question builders and the label report (pure Python, CI-safe).

A record is one /v1/systemone request plus a target distribution per question:

    {"id": "agnews/train/17", "source": "agnews", "split": "train",
     "request": {"state": ..., "questions": {key: {type, instructions, criteria}}},
     "targets": {key: [p over the question's options, in option order]}}

Option order for a target is schema.question_options(): choice criteria order,
[false, true] for noul, level order for score. Targets are one-hot here; the
format allows soft targets (Phase 5 ChaosNLI, teacher labels).

Every builder takes the record's random.Random, so a build is reproducible
from one seed.
"""
from __future__ import annotations

import random
from collections import Counter

from src.pointwise.schema import SystemOneRequest, question_options


def one_hot(n: int, index: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(n)]


def choice(
    rng: random.Random,
    instructions: str,
    options: list[str],
    gold: str,
    k: int | None = None,
    descriptions: dict[str, str] | None = None,
) -> tuple[dict, list[float]]:
    """A Choice question with the options shuffled. With k, keep only k options:
    the gold plus k-1 random distractors. Shuffling every time is what stops the
    gold from sitting at a predictable position."""
    if gold not in options:
        raise ValueError(f"gold {gold!r} is not an option")
    if len(set(options)) != len(options):
        raise ValueError("options must be unique")
    pool = [o for o in options if o != gold]
    if k is not None:
        pool = rng.sample(pool, k - 1)
    picked = pool + [gold]
    rng.shuffle(picked)
    descriptions = descriptions or {}
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": {o: descriptions.get(o) for o in picked},
    }
    return question, one_hot(len(picked), picked.index(gold))


def noul(instructions: str, answer: bool, criteria: dict | None = None) -> tuple[dict, list[float]]:
    question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question, one_hot(2, int(answer))


def score(instructions: str, levels: list[str], gold_level: int) -> tuple[dict, list[float]]:
    """An ordered Score question. Levels are never shuffled: their order is the scale."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}, one_hot(
        len(levels), gold_level
    )


def record(source: str, split: str, index: int, state, questions: dict[str, tuple[dict, list[float]]]) -> dict:
    return {
        "id": f"{source}/{split}/{index}",
        "source": source,
        "split": split,
        "request": {"state": state, "questions": {key: q for key, (q, _) in questions.items()}},
        "targets": {key: t for key, (_, t) in questions.items()},
    }


def validate(rec: dict) -> None:
    """Raises unless the request is a valid /v1/systemone request and every
    target is a distribution over exactly that question's options."""
    request = SystemOneRequest.model_validate(rec["request"])
    if set(rec["targets"]) != set(request.questions):
        raise ValueError(f"{rec['id']}: target keys differ from question keys")
    for key, question in request.questions.items():
        target = rec["targets"][key]
        names, _ = question_options(question)
        if len(target) != len(names):
            raise ValueError(f"{rec['id']}/{key}: {len(target)} targets for {len(names)} options")
        if abs(sum(target) - 1) > 1e-6 or min(target) < 0:
            raise ValueError(f"{rec['id']}/{key}: target is not a distribution")


def pick(rng: random.Random, templates: list[str], **fields) -> str:
    """A random paraphrase of an instruction."""
    return rng.choice(templates).format(**fields)


def label_report(records: list[dict]) -> dict:
    """Per source: records, questions, question types, option counts, and where
    the gold option sits. For unordered choices the gold position should be
    close to uniform; a skew would let a model score by position alone."""
    report: dict = {}
    for rec in records:
        src = report.setdefault(
            rec["source"],
            {
                "records": 0,
                "questions": 0,
                "types": Counter(),
                "options": Counter(),
                "questions_per_record": Counter(),
                "choice_total": 0,
                "gold_first": 0,  # gold is the first option
                "expected_first": 0.0,  # sum of 1/K: gold_first if position were random
                "gold_relative_position": 0.0,  # sum of index/(K-1); mean 0.5 if random
                "noul_true": 0,
                "noul_total": 0,
            },
        )
        src["records"] += 1
        qs = rec["request"]["questions"]
        src["questions_per_record"][len(qs)] += 1
        for key, q in qs.items():
            target = rec["targets"][key]
            gold = max(range(len(target)), key=target.__getitem__)
            src["questions"] += 1
            src["types"][q["type"]] += 1
            src["options"][_bucket(len(target))] += 1
            if q["type"] == "choice" and len(target) > 1:
                src["choice_total"] += 1
                src["gold_first"] += gold == 0
                src["expected_first"] += 1 / len(target)
                src["gold_relative_position"] += gold / (len(target) - 1)
            elif q["type"] == "noul":
                src["noul_total"] += 1
                src["noul_true"] += gold
    for src in report.values():
        for key in ("types", "options", "questions_per_record"):
            src[key] = dict(sorted(src[key].items()))
        if src["choice_total"]:
            n = src["choice_total"]
            src["gold_first_rate"] = src.pop("gold_first") / n
            src["expected_first_rate"] = src.pop("expected_first") / n
            src["gold_mean_relative_position"] = src.pop("gold_relative_position") / n
        else:
            for key in ("gold_first", "expected_first", "gold_relative_position"):
                src.pop(key)
    return report


def _bucket(n: int) -> str:
    for hi in (2, 3, 5, 10, 25, 50, 100):
        if n <= hi:
            return f"<= {hi}"
    return "> 100"
