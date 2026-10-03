# MiniJev Phase 0: Logit Read-out (execution plan)

Parent: `minijev-plan.md` Phase 0. Date: 2026-10-03.

## Goal
Get a decision on Banking77 without calling `generate()`. Score all 77 labels from
the model's logits and return a calibrated probability distribution. Then measure
calibration (ECE, Brier, AURC, Cov@ε) next to accuracy.

## Design decision: full-sequence label scoring
Banking77 labels span several tokens (`card_payment_not_recognised`), so a
single first token can't tell all of them apart. For each label `y` we compute

    s(y) = Σ_t log p(y_t | prompt, y_<t)        # exact sequence log-likelihood
    p(y | x) = softmax_y(s(y))                   # renormalised over the 77 labels

This is the exact probability of each label under the model, restricted to the
label set. It matches how constrained decoding scores labels, but covers all 77,
not just the greedy path.

Efficiency (a small version of Jev's "state once, questions in parallel"):
1. Prefill the prompt once (`use_cache=True`) to get its KV cache.
2. Expand the KV cache across 77 rows and run one forward pass over the
   right-padded label token sequences.
3. Gather the token log-probs and mask the padding.

One prompt therefore costs 1 prefill + 1 batched pass, instead of 77 full passes.

## Files (code discipline: pure logic separate from the ML code)
| File | Change | Imports torch? |
|---|---|---|
| `src/evaluate_core.py` | add `softmax_scores`, `expected_calibration_error`, `brier_score`, `risk_coverage_curve`, `aurc`, `coverage_at_risk` | **No** (CI-safe, stdlib + sklearn only) |
| `src/readout.py` | new: `tokenize_labels`, `score_labels` (KV-cache scorer), `readout_dataset` (resumable JSONL, same as `evaluate_dataset`) | yes |
| `scripts/compare_readout.py` | new: run generation and read-out on the same split; print accuracy, macro-F1, ECE, Brier, AURC, Cov@5%, latency | yes |
| `tests/test_evaluate_core.py` | tests for the new metrics (known closed-form values, edge cases) | no |

Unchanged: `train.py`, `evaluate.py`, `serve.py`, `configs/task.yaml`. The prompt
template is reused exactly, so the read-out sees the training distribution.

## Label tokenisation detail
`train.py` appends the bare label straight after the prompt's trailing newline,
followed by EOS. The scorer encodes `label` with `add_special_tokens=False`
and **appends EOS**. Without EOS, a label that is a prefix of another label
would be scored unfairly.

## Gates (from parent plan)
- Read-out accuracy within 1 pt of constrained generation on the same split.
- At least 5× faster per example on a T4.
- ECE, Brier, AURC and Cov@5% reported.
- `pytest` green locally, and `evaluate_core.py` still has no torch import.

## Steps
1. Add the metrics to `evaluate_core.py` and their tests; run pytest. ← local
2. Write `src/readout.py`; syntax/import check locally; test it against a tiny
   model (`sshleifer/tiny-gpt2`, or skip if not cached) to confirm that
   KV-cache scoring equals naive full-sequence scoring.
3. Write `scripts/compare_readout.py`.
4. Run on Kaggle T4 with the existing LoRA checkpoint (needs user go-ahead,
   because it uses GPU quota).

## Status (2026-10-03)
- [x] Step 1: metrics in `evaluate_core.py` plus 8 tests (31/31 pass, still no torch import).
- [x] Step 2: `src/readout.py`. `tests/test_readout.py` shows that KV-cache scoring equals naive per-label scoring (atol 1e-4, tiny random Qwen2).
- [x] Step 3: `scripts/compare_readout.py`. Data loading verified locally. Full run not possible locally (Qwen weights not cached).
- [ ] Step 4: Kaggle T4 run with the LoRA checkpoint → gate results.
