# Pointwise Phase 0: Logit Read-out (execution plan)

Parent: `pointwise-plan.md` Phase 0. Date: 2026-10-03.

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
- [x] Step 4: Kaggle T4 run (2026-10-04, `r16_a32_all`, val n=500). First run crashed (tuple KV cache on transformers 4.46), fixed in `300ae7f`.

## Results (val, n=500, T4 fp16)
| | Read-out | Generation |
|---|---|---|
| Accuracy | **0.918** | 0.914 |
| Macro-F1 | 0.911 | 0.905 |
| ECE | 0.024 | n/a |
| Brier | 0.121 | n/a |
| AURC | 0.0085 | n/a |
| Cov@5% risk | **0.932** | n/a |
| ms / example | 478 | **177** |

Agreement 99.4%. Gates: accuracy PASS, calibration PASS, **speed FAIL (0.37×, i.e. 2.7× slower)**.

### Why read-out is slower (diagnosis)
1. **No batching across examples.** Read-out scores one message at a time; generation batches 16.
2. **Full-vocab log-softmax on 77 rows.** `log_softmax(out.logits.float())` materialises
   77 × (label_len−1) × 151,936 fp32 values per example (~0.5 GB of traffic), only to gather ~400 of them.
3. **The 400-token label-list prefix is recomputed per example**, the same as in generation. Only the
   message differs between examples, so this prefix can be cached once for the whole run (Jev's "state once").

### Fix plan (Phase 0b)
- Cache the static prompt prefix (everything before the message) once per run.
- Replace the full log-softmax with `gather(target logits) − logsumexp(logits)`, computed in fp32 chunks.
- Batch B messages × 77 labels per forward pass.
- Gate unchanged: ≥5× faster than generation, accuracy unchanged (agreement ≥ 99%).
