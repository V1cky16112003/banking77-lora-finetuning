# Pointwise Phase 1: Shared-prefix parallel questions (execution plan)

Parent: `pointwise-plan.md` Phase 1. Date: 2026-10-04.

## Goal
One state, k questions, the state computed once. Each question must get exactly
what a separate "state + this question" call would give it, and must not see
its siblings.

## Design: three equivalent forms (`src/pointwise/packing.py`)
| Form | How | When |
|---|---|---|
| packed | state + all questions in one sequence; block attention mask; positions restart at `len(state)` per question | one-shot requests |
| branches | state prefilled into a KV cache; all questions in one pass on top; cache cropped back to the state afterwards | serving, a state reused across requests |
| rows | state prefilled; each question its own causal row on a copy of the cache, rows per pass capped by a token budget | backbones that can't take a block mask (recurrent/hybrid), or very many questions |

Mask: additive, `finfo(dtype).min` rather than `-inf` (a softmax row can never
become NaN), and additive rather than boolean because eager attention adds it to
the scores. The functions return logits for a `*ForCausalLM` and last hidden
states for a bare backbone, which is what the Phase 2 pointer head reads.

Learn: restarted positions are why the context budget is "state + longest
question". Every question sits at positions `len(state) .. len(state)+len(q)`,
however many questions there are.

## Dependency bump
`requirements.txt` on `jev`: transformers 4.46.3 → **5.14.1** (Qwen3 needs ≥ 4.51),
peft 0.13.2 → **0.21.2**. Checked locally in a venv matching the Kaggle install
(torch 2.12, transformers 5.14.1, peft 0.21.2, datasets 3.1.0, huggingface-hub 1.26):
full test suite green, every `src/` and `scripts/` module imports, Banking77
loads with the same split sizes, and LoRA wrap + `merge_and_unload` is exact
on a tiny Qwen3. `main` keeps the old pins.

## Local verification
`tests/test_packing.py`, tiny random Qwen3 (eager, SDPA) and Qwen2 (SDPA), fp32:
- packed == separate calls, atol 1e-4;
- branches == separate, and the cache is cropped back and reused for a second request;
- rows == separate, at one pass and at one row per pass;
- changing one question leaves every other question's features unchanged;
- a bare backbone returns hidden states equal to a separate call.

Checked that the tests can fail: replacing the block mask with a plain causal
mask, or using unrestarted positions, makes the equivalence and isolation tests fail.

## Kaggle run (one): `scripts/bench_packing.py`
Qwen3-1.7B-Base backbone, fp16, SDPA, a ~520-token Banking77 chat as the state,
~18-token yes/no questions, k = 1, 4, 16, 64. Times separate / packed / branches
/ branches_warm / rows (median of 5), and the fp16 drift between packed and
separate features. Output `pointwise-output/packing_bench.json`.

**Gate:** equivalence tests pass (done locally); latency grows sub-linearly in k,
i.e. `growth_k1_to_kmax` well below 64 for packed and branches, while separate
grows about 64×.

## Status
- [x] `packing.py` with the three forms; 24 tests green.
- [x] Dependency bump checked locally.
- [x] Kaggle T4 benchmark, 2026-10-05. **Gate PASS.**

## Results (T4, fp16, SDPA, Qwen3-1.7B-Base, transformers 5.14.1)
State 523 tokens, questions 18.5 tokens on average. Median of 5, ms per request.

| k | separate | packed | branches (cold) | branches (warm cache) | rows | drift |
|---|---|---|---|---|---|---|
| 1 | 133 | 98 | 165 | 41 | 173 | 0.0019 |
| 4 | 539 | 106 | 169 | 41 | 178 | 0.0027 |
| 16 | 2,258 | 175 | 207 | 70 | 257 | 0.0032 |
| 64 | 10,422 | **473** | 500 | 320 | 788 | 0.0054 |
| growth k=1→64 | **78×** | **4.8×** | 3.0× | 7.8× | 4.6× | |

Gate: latency sub-linear in k. **PASS.** Separate calls grow about linearly (78×); packed
grows 4.8× for 64× the questions. At k=64 the packed request is **22× faster** than 64
separate calls.

What else the numbers say:
- **Marginal cost of a question:** packed (473 − 98) / 63 ≈ 6 ms, about 6% of a whole
  one-question request. Warm cache: (320 − 41) / 63 ≈ 4.4 ms. That is the shape the
  Phase 2 gate asks for (extra question ≤ 25% of the first), but here with 18-token
  questions; a 77-option Banking77 question is ~400 tokens and has to be measured in Phase 2.
- **A cached state is the big serving win:** one question on a warm cache costs 41 ms
  against 98 ms packed, because the 523 state tokens are not recomputed.
- **Cold branches is slower than packed at small k** (165 vs 98 ms at k=1): two forward
  passes plus building the cache, against one pass. Use packed for one-off requests and
  branches only when the state is reused.
- **Rows is the slowest cached form** (788 ms at k=64): every row carries its own copy of
  the 523-token cache. Kev reports the same cost. Keep rows for backbones that can't take
  a block mask.
- **fp16 drift** (max |packed − separate| relative to the feature scale) is 0.2–0.5% and
  grows with sequence length, while the fp32 tests agree to ~1e-7. It is fp16 rounding,
  not a masking error. Phase 2 should check that it never flips an argmax.
- Unexplained: separate at k=1 (133 ms) is slower than packed at k=1 (98 ms) for almost
  the same tokens. Possibly SDPA choosing a different kernel for an implicit causal mask
  than for an explicit one. Not worth chasing for the gate.

Phase 1 is done. Next: Phase 2, the pointer head and the `/v1/systemone` API.
