"""Rule-based policy records: Jev-shaped requests with labels computed by code.

A record is a generated business object (an order, a loan application, a
support ticket) rendered as JSON or prose, with 1-6 questions about it:
thresholds, value bands, field extraction, ordered levels, and multi-condition
rules stated in the question. Every label is evaluated from the object, so it
is exact and costs nothing (Kev trains on similar generated policies; this
replaces a paid teacher set in v1).

No dates: the backbone can't subtract them reliably (Kev notes the same), so
durations are given as counts ("delivered 12 days ago").
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.minijev.data import records as R


@dataclass(frozen=True)
class Numeric:
    name: str
    phrase: str
    lo: int
    hi: int
    unit: str = ""  # "$" prefix for money, otherwise a suffix like "days"
    money: bool = False  # money values carry cents, so integer thresholds never tie

    def sample(self, rng: random.Random):
        whole = rng.randint(self.lo, self.hi)
        return round(whole + rng.randint(1, 99) / 100, 2) if self.money else whole

    def fmt(self, value) -> str:
        if self.money:
            return f"${value:,.2f}" if isinstance(value, float) else f"${value:,}"
        return f"{value} {self.unit}".strip()

    def threshold(self, rng: random.Random) -> int:
        span = self.hi - self.lo
        return rng.randint(self.lo + span // 10, self.hi - span // 10)


@dataclass(frozen=True)
class Categorical:
    name: str
    phrase: str
    values: tuple[str, ...]
    ordered: bool = False  # values listed low -> high

    def sample(self, rng: random.Random) -> str:
        return rng.choice(self.values)

    def fmt(self, value) -> str:
        return value


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str
    id_prefix: str
    fields: tuple
    outcomes: tuple[str, str, str]  # decision options: (first rule, second rule, otherwise)


DOMAINS = (
    Domain(
        "orders",
        "order",
        "ORD",
        (
            Categorical("customer_tier", "customer tier", ("basic", "silver", "gold"), ordered=True),
            Categorical("category", "item category", ("electronics", "clothing", "books", "home", "toys")),
            Numeric("price", "order total", 5, 800, money=True),
            Numeric("days_since_delivery", "number of days since delivery", 0, 90, unit="days"),
            Categorical("condition", "item condition", ("unopened", "opened", "damaged")),
            Numeric("previous_refunds", "number of previous refunds", 0, 6),
            Categorical("payment_method", "payment method", ("card", "paypal", "gift card")),
        ),
        ("full refund", "store credit", "reject"),
    ),
    Domain(
        "loans",
        "loan application",
        "APP",
        (
            Numeric("annual_income", "annual income", 12000, 180000, money=True),
            Numeric("loan_amount", "requested loan amount", 1000, 60000, money=True),
            Numeric("credit_score", "credit score", 450, 850),
            Numeric("years_employed", "years in current job", 0, 30, unit="years"),
            Categorical("employment", "employment type", ("salaried", "self-employed", "contract", "unemployed")),
            Categorical("risk_band", "risk band", ("low", "medium", "high"), ordered=True),
            Numeric("missed_payments", "number of missed payments in the last year", 0, 8),
        ),
        ("approve", "refer to underwriter", "decline"),
    ),
    Domain(
        "tickets",
        "support ticket",
        "TCK",
        (
            Categorical("priority", "priority", ("low", "normal", "high", "urgent"), ordered=True),
            Categorical("product", "product", ("mobile app", "website", "billing", "hardware", "api")),
            Numeric("hours_open", "hours the ticket has been open", 0, 240, unit="hours"),
            Numeric("customer_replies", "number of customer replies", 0, 12),
            Categorical("plan", "customer plan", ("free", "pro", "enterprise"), ordered=True),
            Categorical("channel", "contact channel", ("email", "chat", "phone")),
            Numeric("affected_users", "number of affected users", 1, 5000),
        ),
        ("escalate", "assign to an agent", "send a self-help article"),
    ),
)

NUMBERED = ("first", "second", "third")


@dataclass(frozen=True)
class Condition:
    text: str
    field: str
    op: str  # ">", "<=", "in"
    arg: object

    def holds(self, obj: dict) -> bool:
        value = obj[self.field]
        if self.op == ">":
            return value > self.arg
        if self.op == "<=":
            return value <= self.arg
        return value in self.arg


def condition(rng: random.Random, domain: Domain, avoid: frozenset = frozenset()) -> Condition:
    """A random condition on a field not in `avoid`, so one rule never tests a field twice."""
    field = rng.choice([f for f in domain.fields if f.name not in avoid])
    if isinstance(field, Numeric):
        t = field.threshold(rng)
        if rng.random() < 0.5:
            return Condition(f"the {field.phrase} is more than {field.fmt(t)}", field.name, ">", t)
        return Condition(f"the {field.phrase} is at most {field.fmt(t)}", field.name, "<=", t)
    allowed = tuple(rng.sample(field.values, rng.randint(1, len(field.values) - 1)))
    shown = " or ".join(allowed)
    return Condition(f"the {field.phrase} is {shown}", field.name, "in", allowed)


def render_state(rng: random.Random, domain: Domain, obj: dict, obj_id: str):
    """JSON half of the time, prose the other half; field order shuffled."""
    fields = list(domain.fields)
    rng.shuffle(fields)
    if rng.random() < 0.5:
        return {"id": obj_id, "type": domain.noun, **{f.name: obj[f.name] for f in fields}}
    lines = [f"{domain.noun.capitalize()} {obj_id}."] + [
        f"The {f.phrase} is {f.fmt(obj[f.name])}." for f in fields
    ]
    return " ".join(lines)


# --- question templates: (rng, domain, obj) -> (question, target) ---


def _threshold(rng, domain, obj):
    cond = condition(rng, domain)
    return R.noul(f"Is it true that {cond.text}?", cond.holds(obj))


def _band(rng, domain, obj):
    field = rng.choice([f for f in domain.fields if isinstance(f, Numeric)])
    a, b = sorted(rng.sample(range(field.lo + 1, field.hi), 2))
    value = obj[field.name]
    if field.money:
        levels = [f"under {field.fmt(a)}", f"{field.fmt(a)} to {field.fmt(b)}", f"over {field.fmt(b)}"]
        gold = 0 if value < a else 1 if value <= b else 2
    else:
        levels = [f"{field.fmt(a)} or less", f"{field.fmt(a + 1)} to {field.fmt(b)}", f"more than {field.fmt(b)}"]
        gold = 0 if value <= a else 1 if value <= b else 2
    return R.score(f"Which range is the {field.phrase} in?", levels, gold)


def _extract(rng, domain, obj):
    field = rng.choice([f for f in domain.fields if isinstance(f, Categorical)])
    if field.ordered:
        return R.score(f"What is the {field.phrase}?", list(field.values), field.values.index(obj[field.name]))
    return R.choice(rng, f"What is the {field.phrase}?", list(field.values), obj[field.name])


def _distinct(rng, domain, n):
    conds = []
    for _ in range(n):
        conds.append(condition(rng, domain, frozenset(c.field for c in conds)))
    return conds


def _a(noun: str) -> str:
    return f"an {noun}" if noun[0] in "aeiou" else f"a {noun}"


def _all_of(rng, domain, obj):
    conds = _distinct(rng, domain, rng.randint(2, 3))
    rule = " and ".join(c.text for c in conds)
    return R.noul(
        f"Policy: {_a(domain.noun)} qualifies only if {rule}. Does this {domain.noun} qualify?",
        all(c.holds(obj) for c in conds),
    )


def _any_of(rng, domain, obj):
    conds = _distinct(rng, domain, 2)
    rule = " or ".join(c.text for c in conds)
    return R.noul(
        f"Flag the {domain.noun} for review if {rule}. Should it be flagged?",
        any(c.holds(obj) for c in conds),
    )


def _decision(rng, domain, obj):
    first, second, otherwise = domain.outcomes
    c1, c2 = _distinct(rng, domain, 2)
    c3 = condition(rng, domain)
    rules = (
        f"Choose {first} if {c1.text} and {c2.text}. "
        f"Otherwise choose {second} if {c3.text}. "
        f"In every other case choose {otherwise}."
    )
    gold = first if c1.holds(obj) and c2.holds(obj) else second if c3.holds(obj) else otherwise
    return R.choice(rng, f"Apply the rules in order. {rules} What is the outcome?", list(domain.outcomes), gold)


TEMPLATES = (_threshold, _band, _extract, _all_of, _any_of, _decision)


def policy_record(rng: random.Random, split: str, index: int) -> dict:
    domain = rng.choice(DOMAINS)
    obj = {f.name: f.sample(rng) for f in domain.fields}
    obj_id = f"{domain.id_prefix}-{rng.randint(10000, 99999)}"
    n = rng.randint(1, 6)
    questions = {}
    for i in range(n):
        template = rng.choice(TEMPLATES)
        questions[f"{template.__name__.strip('_')}_{i + 1}"] = template(rng, domain, obj)
    return R.record("policies", split, index, render_state(rng, domain, obj, obj_id), questions)


def policy_records(seed: int, split: str, n: int) -> list[dict]:
    rng = random.Random(f"policies:{split}:{seed}")
    return [policy_record(rng, split, i) for i in range(n)]
