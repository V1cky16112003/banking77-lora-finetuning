# MiniJev Phase 0b: Fast read-out (execution plan)

Parent: `minijev-phase0-plan.md` (Results section). Date: 2026-10-04.

## Problem
Phase 0 read-out is exact (99.4% agreement with generation) and well calibrated
(ECE 0.024), but takes 478 ms/example against 177 ms for generation. The gate
needs it to be **≥5× faster than generation**.

Per example, the current scorer:
- prefills the full ~470-token prompt, ~400 tokens of which are the fixed label
  list and identical for every example;
- copies that prompt's KV cache into 77 rows (~1 GB of copying per example);
- runs a full-vocabulary log-softmax in fp32 over 77 × ~11 × 151,936 logits;
- handles one message at a time.

## Design: three changes, same exact scores

### 1. Static prefix cache, computed once per run (Jev's "state once")
Split each prompt into
- **prefix**: everything up to and including the last newline before
  `{message}`, i.e. the instructions and the 77-label list. It's the same for
  every example.
- **suffix**: `Customer message: <message>\nIntent:\n`.

The prefix is prefilled **once** and its KV cache reused for every example.

*Tokenisation risk:* BPE might merge tokens across the split point, which would
silently change the input the model sees. So the split is at a newline,
`encode(prefix) + encode(suffix) == encode(full)` is checked **for every example
at runtime** (it raises on a mismatch, never silently drifts), and a local test
checks it on all Banking77 val + test messages.

### 2. Two-stage batched scoring
For a batch of B messages:
- **Stage A, the messages:** B rows of suffix tokens (right-padded) on top of
  the prefix cache. The last real position gives each label's first-token
  log-prob. Its cache holds prefix + suffix.
- **Stage B, the labels:** repeat the stage-A cache 77× per message (B·77
  rows) and feed `label[:-1]` tokens. The attention mask hides suffix padding,
  and explicit `position_ids = P + L_i + t` keep positions identical to an
  unpadded run.

The suffix is computed once per message rather than once per label.

### 3. Log-probs without materialising a full-vocab fp32 softmax
`logprob(target) = logit[target] − logsumexp(logits)`, computed in fp32 over
row chunks, gathering only the label tokens.

## Files
| File | Change |
|---|---|
| `src/readout.py` | replace `score_labels` with `LabelScorer` (prefix cache + `score(messages)`); `split_prompt`; `readout_dataset` takes `batch_size` |
| `scripts/compare_readout.py` | `--batch-size` flag (default 4) |
| `tests/test_readout.py` | batched scores == naive per-label scores for messages of different lengths (padding path); prefix-split invariant on real Banking77 messages |
| `notebooks/kaggle_minijev_phase0.ipynb` | unchanged (picks up new code from the branch) |

## Local verification (before any GPU time)
1. Equivalence: `LabelScorer.score` == naive full-sequence scoring, atol 1e-4,
   tiny random Qwen2, batch of messages of different lengths.
2. Split invariant holds for all Banking77 val + test messages with the real
   Qwen tokenizer and template.
3. Both on **transformers 4.46.3** (Kaggle pin) and 5.14.

## Kaggle run (one)
Same notebook, same adapter, val n=500, batch size 4.
Gates: agreement ≥ 99% with generation, accuracy within 1 pt, **≥5× faster**.

## Memory budget (T4, 15 GB)
Stage B holds a KV cache of B·77 × ~500 tokens: ~13 MB per row, about 4 GB at
B=4. Model weights take ~3 GB in fp16. B is a flag, so an OOM means lowering
it, not changing code.

## Honest risk
Read-out still does more total work than greedy generation: 77 labels × ~10
tokens against ~8 sequential decode steps. The win comes from replacing
sequential decode steps with parallel ones and from dropping the repeated
prefix. If it lands below 5×, the report will show the measured breakdown, and
a token-trie over shared label prefixes (`card_…`) is the next lever.

## Status
- [x] `LabelScorer` with prefix cache, two-stage batching and chunked logsumexp.
- [x] Local: batched == naive (atol 1e-4), batch-of-1 == batch-of-3, split invariant on all
      13,083 Banking77 messages. Green on transformers 4.46.3 and 5.14.
- [x] Kaggle T4 run, 2026-10-04 12:43 UTC, val n=500, batch size 4, `r16_a32_all`.

## Results
| | Phase 0 read-out | **Phase 0b read-out** | Generation |
|---|---|---|---|
| Accuracy | 0.918 | **0.918** | 0.914 |
| ECE | 0.024 | **0.021** | n/a |
| Cov@5% risk | 0.932 | **0.932** | n/a |
| ms / example | 478 | **336** | 167 |

Agreement with generation 99.4% (unchanged, so the scores really are unchanged).
Gates: accuracy PASS, calibration PASS, **speed FAIL (0.50× generation, only 1.4× faster than Phase 0)**.

### What this tells us
The three planned changes saved 142 ms/example, far less than estimated. Back-of-envelope
cost for one batch of 4 (lm_head over 3,080 label positions ~0.7 TFLOP, ~4 GB of KV copies)
should be ~100 ms, i.e. ~25 ms/example. The measured 336 ms means something else dominates,
and guessing again is not good engineering. **Next step is a profiling run**, not another
optimisation: time prefix, stage A, the KV repeat, stage B forward, and logsumexp separately
(CUDA-synchronised), plus check whether 4.46 drops from SDPA to eager attention when it is
given a padded 2D mask.
