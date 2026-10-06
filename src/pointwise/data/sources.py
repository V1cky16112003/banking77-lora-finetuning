"""Public datasets -> decision records.

Converters are pure: plain row dicts (and label names) in, records out, so tests
can feed them hand-made rows. Each record carries its raw text under "_text";
build.py uses it for the leakage checks and strips it before writing.

Loading (load_rows) is the only part that touches the Hub, at pinned revisions.
"""
from __future__ import annotations

import html
import random

from src.pointwise.data import records as R


def _state(rng: random.Random, text: str, json_key: str, json_rate: float = 0.25):
    """The raw text, or the same text inside a small JSON object."""
    return {json_key: text} if rng.random() < json_rate else text


def clean(text: str) -> str:
    """Undo source scraping artifacts: AG News's broken HTML entities ("Woods #39;
    record") and stray backslashes, Yelp's literal "\\n" sequences."""
    text = text.replace("\\n", "\n").replace("\\", "")
    text = text.replace(" #39;", "'").replace("#39;", "'").replace("quot;", '"')
    return html.unescape(text).strip()


def _with_text(rec: dict, text: str) -> dict:
    rec["_text"] = text
    return rec


# --- MNLI ---------------------------------------------------------------------

NLI_OPTIONS = ["entailment", "neutral", "contradiction"]
NLI_DESCRIPTIONS = {
    "entailment": "the hypothesis must be true if the premise is true",
    "neutral": "the hypothesis might or might not be true",
    "contradiction": "the hypothesis cannot be true if the premise is true",
}
NLI_CHOICE = [
    "How does the hypothesis relate to the premise?",
    "Given the premise, what is the status of the hypothesis?",
    "Classify the relation between premise and hypothesis.",
]
NLI_NOUL = [
    "Does the premise imply that the hypothesis is true?",
    "If the premise is true, must the hypothesis also be true?",
    "Is the hypothesis entailed by the premise?",
]


def mnli(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        if row["label"] < 0:
            continue
        gold = names[row["label"]]
        if rng.random() < 0.25:
            state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
        else:
            state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
        kinds = ["choice", "noul"] if rng.random() < 0.3 else [rng.choice(["choice", "choice", "noul"])]
        qs = {}
        for kind in kinds:
            if kind == "choice":
                qs["relation"] = R.choice(rng, R.pick(rng, NLI_CHOICE), NLI_OPTIONS, gold, descriptions=NLI_DESCRIPTIONS)
            else:
                qs["entailed"] = R.noul(R.pick(rng, NLI_NOUL), gold == "entailment")
        out.append(_with_text(R.record("mnli", split, i, state, qs), row["premise"] + " || " + row["hypothesis"]))
    return out


# --- BoolQ --------------------------------------------------------------------


def boolq(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        question = row["question"].strip()
        question = question[0].upper() + question[1:] + ("" if question.endswith("?") else "?")
        state = _state(rng, row["passage"], "passage")
        qs = {"answer": R.noul(question, bool(row["answer"]))}
        out.append(_with_text(R.record("boolq", split, i, state, qs), row["passage"] + " || " + row["question"]))
    return out


# --- AG News ------------------------------------------------------------------

AG_TOPICS = {"World": "world", "Sports": "sports", "Business": "business", "Sci/Tech": "science and technology"}
AG_CHOICE = [
    "What is this news article about?",
    "Which section of a newspaper does this article belong in?",
    "Pick the topic of the article.",
]


def agnews(rows, names, rng, split, start=0):
    topics = [AG_TOPICS[n] for n in names]
    out = []
    for i, row in enumerate(rows, start):
        gold = topics[row["label"]]
        text = clean(row["text"])
        qs = {"topic": R.choice(rng, R.pick(rng, AG_CHOICE), topics, gold, k=rng.randint(2, len(topics)))}
        if rng.random() < 0.5:
            asked = gold if rng.random() < 0.5 else rng.choice([t for t in topics if t != gold])
            qs["is_topic"] = R.noul(f"Is this article mainly about {asked}?", asked == gold)
        out.append(_with_text(R.record("agnews", split, i, _state(rng, text, "article"), qs), text))
    return out


# --- SST-5 and Yelp: ordered sentiment ----------------------------------------

SENTIMENT_SCALES = [
    ["very negative", "negative", "neutral", "positive", "very positive"],
    ["hated it", "disliked it", "mixed feelings", "liked it", "loved it"],
    ["strongly unfavourable", "unfavourable", "neither", "favourable", "strongly favourable"],
]
SENTIMENT_SCORE = [
    "How positive is the writer about the subject?",
    "Rate the sentiment of the text.",
    "How does the author feel about it?",
]
POLARITY = ["negative", "negative", "neutral", "positive", "positive"]


def sst5(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        level = int(row["label"])
        qs = {"sentiment": R.score(R.pick(rng, SENTIMENT_SCORE), rng.choice(SENTIMENT_SCALES), level)}
        if rng.random() < 0.4:
            qs["polarity"] = R.choice(
                rng, "Is the overall sentiment positive, negative or neutral?", ["negative", "neutral", "positive"], POLARITY[level]
            )
        out.append(_with_text(R.record("sst5", split, i, _state(rng, row["text"], "review"), qs), row["text"]))
    return out


STAR_SCALE = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def yelp(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        stars = int(row["label"])  # 0..4 = 1..5 stars
        text = clean(row["text"])
        scale = STAR_SCALE if rng.random() < 0.5 else rng.choice(SENTIMENT_SCALES)
        qs = {"rating": R.score("How many stars did the reviewer give?" if scale is STAR_SCALE else R.pick(rng, SENTIMENT_SCORE), scale, stars)}
        if stars != 2 and rng.random() < 0.3:
            qs["recommend"] = R.noul("Would the reviewer recommend this business?", stars >= 3)
        state = {"review": text, "site": "Yelp"} if rng.random() < 0.25 else text
        out.append(_with_text(R.record("yelp", split, i, state, qs), text))
    return out


# --- CLINC150 -----------------------------------------------------------------

# CLINC's banking and credit_cards domains (15 intents each) overlap Banking77's
# label space, so they are dropped to keep the Banking77 eval zero-shot.
CLINC_BANKING_DOMAINS = frozenset({
    "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance",
    "bill_due", "interest_rate", "routing", "min_payment", "order_checks", "pin_change",
    "report_fraud", "account_blocked", "spending_history",
    "credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
    "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
    "expiration_date",
})
CLINC_CHOICE = [
    "Which intent best describes this request to a virtual assistant?",
    "What does the user want?",
    "Classify the user's request.",
]


def clinc_excluded(names: list[str], banking77_labels: list[str]) -> frozenset[str]:
    """Banking/credit-card domains, any intent named exactly like a Banking77
    label, and out-of-scope. Raises if a listed domain intent is missing from
    the dataset, so a typo can't silently let one through."""
    missing = CLINC_BANKING_DOMAINS - set(names)
    if missing:
        raise ValueError(f"CLINC intents not found: {sorted(missing)}")
    return CLINC_BANKING_DOMAINS | (set(names) & set(banking77_labels)) | {"oos"}


def clinc(rows, names, rng, split, start=0, excluded=frozenset(), copies=1):
    intents = [n.replace("_", " ") for n in names if n not in excluded]
    out = []
    for i, row in enumerate(rows, start):
        name = names[row["intent"]]
        if name in excluded:
            continue
        gold = name.replace("_", " ")
        for copy in range(copies):  # same message, a different option subset each time
            # Log-uniform option count: many small sets, some up to the full ~119.
            k = min(len(intents), max(2, round(2 ** rng.uniform(1, 7))))
            qs = {"intent": R.choice(rng, R.pick(rng, CLINC_CHOICE), intents, gold, k=k)}
            if rng.random() < 0.3:
                asked = gold if rng.random() < 0.5 else rng.choice([t for t in intents if t != gold])
                qs["is_intent"] = R.noul(f"Is the user asking about \"{asked}\"?", asked == gold)
            state = _state(rng, row["text"], "user_message")
            out.append(_with_text(R.record("clinc", split, i * copies + copy, state, qs), row["text"]))
    return out


# --- Multiple-choice QA: options differ for every question -----------------------

MCQA_CHOICE = ["Which answer is correct?", "Choose the best answer to the question.", "What is the answer?"]


def _mcqa(source, question, texts, labels, answer_key, rng, split, i):
    if len(set(texts)) != len(texts) or answer_key not in labels:
        return None
    gold = texts[labels.index(answer_key)]
    qs = {"answer": R.choice(rng, R.pick(rng, MCQA_CHOICE), list(texts), gold)}
    if rng.random() < 0.5:
        asked = gold if rng.random() < 0.5 else rng.choice([t for t in texts if t != gold])
        qs["is_correct"] = R.noul(f"Is \"{asked}\" a correct answer to the question?", asked == gold)
    state = {"question": question} if rng.random() < 0.25 else question
    return _with_text(R.record(source, split, i, state, qs), question + " || " + " | ".join(texts))


def mcqa(source: str, question_key: str):
    def convert(rows, names, rng, split, start=0):
        out = []
        for i, row in enumerate(rows, start):
            rec = _mcqa(source, row[question_key], row["choices"]["text"], row["choices"]["label"], row["answerKey"], rng, split, i)
            if rec:
                out.append(rec)
        return out

    return convert


arc = mcqa("arc", "question")
openbookqa = mcqa("openbookqa", "question_stem")
commonsenseqa = mcqa("commonsenseqa", "question")


# --- Held-out sources (evaluation only, fixed instructions) --------------------


def banking77(rows, names, rng, split, start=0):
    labels = [n.replace("_", " ") for n in names]
    out = []
    for i, row in enumerate(rows, start):
        qs = {"intent": R.choice(rng, "Which banking intent best describes this customer message?", labels, labels[row["label"]])}
        out.append(_with_text(R.record("banking77", split, i, row["text"], qs), row["text"]))
    return out


def emotion(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        qs = {"emotion": R.choice(rng, "Which emotion does the writer express?", list(names), names[row["label"]])}
        out.append(_with_text(R.record("emotion", split, i, row["text"], qs), row["text"]))
    return out


def tweet_sentiment(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        qs = {"sentiment": R.choice(rng, "What is the sentiment of this tweet?", list(names), names[row["label"]])}
        out.append(_with_text(R.record("tweet_sentiment", split, i, row["text"], qs), row["text"]))
    return out


def qnli(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        state = {"question": row["question"], "sentence": row["sentence"]}
        qs = {"answers": R.noul("Does the sentence contain the answer to the question?", row["label"] == 0)}
        out.append(_with_text(R.record("qnli", split, i, state, qs), row["question"] + " || " + row["sentence"]))
    return out


def paws(rows, names, rng, split, start=0):
    out = []
    for i, row in enumerate(rows, start):
        state = {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}
        qs = {"paraphrase": R.noul("Do the two sentences mean the same thing?", row["label"] == 1)}
        out.append(_with_text(R.record("paws", split, i, state, qs), row["sentence1"] + " || " + row["sentence2"]))
    return out


# --- Loading --------------------------------------------------------------------

# (repo, config, revision): pinned 2026-10-06, so a rebuild sees the same rows.
REPOS = {
    "mnli": ("nyu-mll/multi_nli", None, "da70db2af9d09693783c3320c4249840212ee221"),
    "boolq": ("google/boolq", None, "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "agnews": ("fancyzhx/ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4"),
    "sst5": ("SetFit/sst5", None, "e51bdcd8cd3a30da231967c1a249ba59361279a3"),
    "yelp": ("Yelp/yelp_review_full", None, "c1f9ee939b7d05667af864ee1cb066393154bf85"),
    "clinc": ("clinc/clinc_oos", "plus", "155b9c710419136e17307b80d0a13e68cd46b4ec"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453"),
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454"),
    "commonsenseqa": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155"),
    "qnli": ("nyu-mll/glue", "qnli", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
    "paws": ("google-research-datasets/paws", "labeled_final", "161ece9501cf0a11f3e48bd356eaa82de46d6a09"),
    "emotion": ("dair-ai/emotion", "split", "cab853a1dbdf4c42c2b3ef2173804746df8825fe"),
    "tweet_sentiment": ("cardiffnlp/tweet_eval", "sentiment", "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"),
}


def load_rows(name: str, split: str) -> tuple[list[dict], list[str] | None]:
    """Rows of a pinned dataset split, and its label names when the label is a ClassLabel."""
    from datasets import load_dataset

    repo, config, revision = REPOS[name]
    ds = load_dataset(repo, config, split=split, revision=revision)
    names = None
    for column in ("label", "intent"):
        feature = ds.features.get(column)
        if feature is not None and hasattr(feature, "names"):
            names = feature.names
    return list(ds), names
