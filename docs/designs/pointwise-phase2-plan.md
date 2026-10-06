# Pointwise Phase 2: Pointer head and the `/v1/systemone` API (execution plan)

Parent: `pointwise-plan.md` Phase 2. Date: 2026-10-05.

## Goal
A typed decision API on the Phase 1 packing: a TypeSafe-compatible request in,
calibration-ready probabilities over the declared options out, every question
in one packed forward pass. Phase 2 only builds and checks the plumbing; the head
stays untrained until Phase 4.

## Files (pure logic separate from the ML code, as in Phase 0)
| File | What | Imports torch? |
|---|---|---|
| `src/pointwise/schema.py` | pydantic `Choice` / `Noul` / `Score` / `SystemOneRequest`; `render`; TypeSafe's confidence formulas; 4-decimal rounding; `to_answers` | No (CI-tested) |
| `src/pointwise/encoding.py` | token layout `<state> … / <q> instr <opt> o </opt> … <decide>` on reused Qwen special tokens; unforgeable boundaries; 32k state limit | No (tokenizer only) |
| `src/pointwise/model.py` | `PointerHead` (fp32, `d_p = 256`, eval-only temperature); `DecisionModel.decide()` on a bare backbone | Yes |
| `scripts/bench_decide.py` | T4 benchmark, the Kaggle job | Yes |

## Decisions
- **Delimiters:** `<|fim_prefix|>` state, `<|fim_middle|>` question,
  `<|box_start|>` / `<|box_end|>` option open/close, `<|fim_suffix|>` decide.
  Single tokens in the Qwen vocabulary (checked), so no embedding rows are added.
- **Read-out positions:** `<decide>` and each option's `</opt>`. Under causal
  attention `</opt>` has read the whole option text; `<decide>` has read the
  instruction and every option.
- **Boundary safety:** caller text is tokenised with `split_special_tokens=True`,
  which keeps the text exact. If a tokenizer ignores the flag, `<|x|>` is
  rewritten to `<¦x¦>`. If a delimiter id still appears, it raises. Never silent.
- **Confidence:** choice `(p_max − 1/K)/(1 − 1/K)`, score `1 − E|level − mode|/D`
  (TypeSafe's reference adapter via Kev), not `max(p)`, so 0.5 means the same thing
  for 2 options as for 200.
- **Temperature** is applied only in eval mode, so training always sees T = 1
  and a fitted T stays meaningful; dividing by T never changes the argmax.

## Local verification
- `tests/test_pointwise_schema.py` (15 tests, no torch): validation limits (1..255
  options, unknown types, empty requests), rendering, both confidence formulas at
  their fixed points, rounding tolerance at 255 options, response shapes.
  Also run in a CI-like Python 3.11 env with `requirements-core.txt` + pydantic 2.9.2.
- `tests/test_pointwise_model.py` (8 tests, tiny random Qwen3 + cached Qwen tokenizer):
  encoding layout; a state/instruction/option containing delimiter strings can't
  add or close options (the test fails if splitting is removed); state limit;
  `decide()` end to end, well typed for choice / noul / score; packed questions ==
  each question alone; a secret in one question doesn't move the others;
  temperature flattens without changing the argmax and is ignored in training;
  a 255-option choice runs.

## Gate and the Kaggle run (`scripts/bench_decide.py`)
Qwen3-1.7B-Base, fp16, untrained head (random but must be well typed).
1. **banking77:** one Banking77 message + one 77-option Choice (~480 tokens).
   Reported next to generation (167 ms/example, Phase 0b). Not gated: the 77
   options are input tokens after the per-message state, so they can't be
   shared. Expect roughly generation's prefill cost without its decode loop.
2. **shared_state (gate):** ~520-token state, k = 1, 2, 4, 8 three-option
   questions. Each extra question must cost ≤ 25% of a one-question request.
3. **fp16 consistency:** probabilities with 8 questions packed vs each alone.

**Why the gate changed:** the first Phase 2 gate asked for the extra-question
cost on Banking77-shaped requests. There the state is a ~15-token message and a
question is ~400 tokens of options, so a second question costs about as much as
the first, and the gate fails by construction. Sharing the state only pays when
the state is large next to the questions, which is the shape Jev is built for.

## Status
- [x] schema, encoding, model; 23 new tests green (87 passed, 1 skipped overall).
- [x] Benchmark smoke-tested on CPU with a tiny model.
- [x] Kaggle T4 run, 2026-10-05. **Gate PASS.**

## Results (T4, fp16, Qwen3-1.7B-Base, transformers 5.14.1, untrained head)

**Well typed:** the example response has every question key, each choice is one of its
declared options and is the argmax of its probabilities, probabilities sum to 1, and
`output_tokens` is 0. The values are random (untrained head) and should not be read.

**Shared state (gate):** ~520-token state, three-option questions, median of 5.

| k | tokens | ms |
|---|---|---|
| 1 | 542 | 101 |
| 2 | 561 | 103 |
| 4 | 603 | 109 |
| 8 | 680 | 137 |

Extra question: (137 − 101) / 7 ≈ **5.2 ms = 5% of a one-question request**
(gate ≤ 25%). **PASS.** Eight questions cost 1.36× one question.

**Banking77 shape (reported, not gated):** one message + one 77-option Choice,
~487 tokens: **90 ms per request, 1.85× faster than generation** (167 ms/example,
Phase 0b). That matches the Phase 0c prediction of ~1.5–3×: the read-out pays
the same ~480-token prefill as generation but drops the decode loop. Not a like-for-
like comparison yet: different backbone (Qwen3-1.7B vs Qwen2.5-1.5B + LoRA), batch 1
vs batch 16, untrained head.

**fp16 consistency:** 8 questions packed vs each alone: max |Δp| 0.0038, **0 argmax
flips** out of 8. The Phase 1 fp16 drift does not change answers here; recheck on a
trained head, where probabilities will be sharper.

**Side note:** the session showed RUNNING for ~40 minutes, but the notebook log
spans only ~3 minutes: ~90 s clone and pip install, ~60 s model download and load,
~10 s of benchmark. The rest was Kaggle's queue and version saving, outside the
code, so there is nothing to optimise in the job itself.

Phase 2 is done. Next: Phase 3, the synthetic data generator.
