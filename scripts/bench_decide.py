"""Phase 2 benchmark: the decision API on a T4 (docs/designs/pointwise-phase2-plan.md).

Untrained pointer head on the Qwen3-1.7B-Base backbone, so the answers are
random but must be well typed. Measures:
- banking77: state = one Banking77 message, one Choice over all 77 labels.
  Reported next to generation, not gated: all 77 options are input tokens.
- shared_state: a ~520-token chat state with k = 1..8 three-option questions.
  Gate: each extra question costs <= 25% of a one-question request.
- fp16 consistency: probabilities of questions packed together vs each alone.

Usage (from the repo root):
    python -m scripts.bench_decide --output-dir eval_runs/decide
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import transformers

from scripts.bench_packing import build_request, time_ms
from src.config import load_task_config
from src.data import load_banking77_splits
from src.pointwise.model import DecisionModel
from src.pointwise.schema import SystemOneRequest

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
N_BANKING_MESSAGES = 32
KS = (1, 2, 4, 8)
MAX_EXTRA_QUESTION_SHARE = 0.25
# Phase 0b, same T4: constrained generation, Qwen2.5-1.5B + LoRA, batch 16.
GENERATION_MS_PER_EXAMPLE = 167

QUESTIONS = [
    ("department", "Which team should handle this?", ["cards", "transfers", "account"]),
    ("urgency", "How urgent is this for the customer?", ["low", "medium", "high"]),
    ("sentiment", "What is the customer's overall mood?", ["calm", "annoyed", "angry"]),
    ("channel", "How should we reply?", ["in-app chat", "email", "phone call"]),
    ("fraud", "Could this involve fraud?", ["no", "possibly", "yes"]),
    ("refund", "Is the customer asking for money back?", ["no", "partly", "yes"]),
    ("repeat", "Has the customer contacted us before about this?", ["no", "unclear", "yes"]),
    ("resolved", "Can a bot resolve this without a human?", ["no", "maybe", "yes"]),
]


def shared_state_request(state: str, k: int) -> dict:
    return {
        "state": state,
        "questions": {
            key: {"type": "choice", "instructions": instr, "criteria": {o: None for o in opts}}
            for key, instr, opts in QUESTIONS[:k]
        },
    }


def bench_banking77(model: DecisionModel, labels: list[str], messages: list[str]) -> dict:
    criteria = {label: None for label in labels}
    requests = [
        SystemOneRequest.model_validate(
            {"state": m, "questions": {"intent": {"type": "choice", "instructions": "What is the customer's intent?", "criteria": criteria}}}
        )
        for m in messages
    ]
    encoded = [model.encode(r) for r in requests]
    per_request = [time_ms(lambda e=e: model.probs(e), model.backbone.device) for e in encoded[:8]]
    tokens = statistics.mean(len(e.state_ids) + len(e.questions[0].ids) for e in encoded)
    # Typing check over every message: exactly 77 probabilities summing to 1.
    for e in encoded:
        (p,) = model.probs(e)
        assert len(p) == len(labels) and abs(sum(p) - 1) < 1e-3
    ms = statistics.median(per_request)
    return {
        "tokens_per_request": tokens,
        "ms_per_request": ms,
        "generation_ms_per_example": GENERATION_MS_PER_EXAMPLE,
        "speedup_vs_generation": GENERATION_MS_PER_EXAMPLE / ms,
    }


def bench_shared_state(model: DecisionModel, state: str) -> dict:
    device = model.backbone.device
    rows = []
    for k in KS:
        encoded = model.encode(SystemOneRequest.model_validate(shared_state_request(state, k)))
        rows.append(
            {
                "k": k,
                "tokens": len(encoded.state_ids) + sum(len(q.ids) for q in encoded.questions),
                "ms": time_ms(lambda: model.probs(encoded), device),
            }
        )
    first, last = rows[0]["ms"], rows[-1]["ms"]
    extra = (last - first) / (KS[-1] - 1)
    return {
        "rows": rows,
        "ms_per_extra_question": extra,
        "extra_question_share_of_first": extra / first,
        "gate_pass": extra / first <= MAX_EXTRA_QUESTION_SHARE,
    }


def fp16_consistency(model: DecisionModel, state: str) -> dict:
    """Packed vs each question alone, in the model's own dtype."""
    request = shared_state_request(state, KS[-1])
    together = model.probs(model.encode(SystemOneRequest.model_validate(request)))
    max_diff, flips = 0.0, 0
    for i, key in enumerate(request["questions"]):
        alone_req = {**request, "questions": {key: request["questions"][key]}}
        (alone,) = model.probs(model.encode(SystemOneRequest.model_validate(alone_req)))
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(alone, together[i])))
        flips += max(range(3), key=alone.__getitem__) != max(range(3), key=together[i].__getitem__)
    return {"max_abs_prob_diff": max_diff, "argmax_flips": flips, "questions": len(together)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_runs/decide"))
    args = parser.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16  # T4: no bf16
    else:
        device, dtype = "cpu", torch.float32
    torch.manual_seed(0)  # the untrained head's random init
    model = DecisionModel.from_pretrained(args.model, device, dtype)

    config = load_task_config()
    messages = [r["text"] for r in load_banking77_splits(config).val][:N_BANKING_MESSAGES]
    state_ids, _ = build_request(model.tokenizer, 512, 1)
    state = model.tokenizer.decode(state_ids)

    report: dict = {
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model": args.model,
            "dtype": str(dtype),
            "head": "untrained (random init)",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "decide_bench.json"

    def step(name: str, fn) -> None:
        # Saved after every step, so a late crash keeps the earlier results.
        try:
            report[name] = fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            report[name] = {"oom": True}
        print(name, json.dumps(report[name]), flush=True)
        out_path.write_text(json.dumps(report, indent=2))

    step("example_response", lambda: model.decide(shared_state_request(state, 3)))
    step("banking77", lambda: bench_banking77(model, config.labels, messages))
    step("shared_state", lambda: bench_shared_state(model, state))
    step("fp16_consistency", lambda: fp16_consistency(model, state))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
