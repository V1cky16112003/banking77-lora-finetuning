# MiniJev Phase 1: Shared-prefix parallel questions (execution plan)

Parent: `minijev-plan.md` Phase 1. Date: 2026-10-04.

## Goal
One state, k questions, the state computed once. Each question must get exactly
what a separate "state + this question" call would give it, and must not see
its siblings.

## Design: three equivalent forms (`src/minijev/packing.py`)
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
separate features. Output `minijev-output/packing_bench.json`.

**Gate:** equivalence tests pass (done locally); latency grows sub-linearly in k,
i.e. `growth_k1_to_kmax` well below 64 for packed and branches, while separate
grows about 64×.

## Status
- [x] `packing.py` with the three forms; 24 tests green.
- [x] Dependency bump checked locally.
- [ ] Kaggle T4 benchmark.
