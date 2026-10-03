# MiniJev — Implementation Plan

Companion to `minijev-architecture.md`. Hardware: Kaggle T4 (fp16, ~30 GPU-h/week).
Each phase ends with a measurable gate; don't start the next phase until it passes.

## Phase 0 — Logit read-out on Banking77 (≈1 day, no new training)
Goal: prove "decide without generating" on the existing pipeline.
- `src/readout.py`: score each of the 77 labels by the logit of its first token
  after `Intent:` (labels → unique single tokens or first-token disambiguation).
- Reuse the existing LoRA checkpoint; compare against `generate()` + `parse_label`.
- Add `ece`, `brier`, `aurc`, `coverage_at` to `src/evaluate_core.py` (+ tests).
- **Gate:** read-out accuracy within 1 pt of generation; ≥5× faster; ECE reported.

Learn: why reading logits gives a probability distribution for free.

## Phase 1 — Shared-prefix parallel questions (≈2 days)
Goal: one state, many questions, one prefill.
- `src/minijev/packing.py`: build packed sequence + block attention mask +
  restarted position ids (state, then k question branches).
- Unit test: packed logits == logits from k separate "state + Qi" runs (atol 1e-3).
- Benchmark on T4: k = 1, 4, 16, 64 questions; compare vs k separate calls.
- **Gate:** equivalence test passes; latency grows sub-linearly in k.

Learn: KV caching, attention masks, why context budget = state + longest question.

## Phase 2 — Slot tokens & the `decide()` API (≈3 days)
- Add 255 `[S_i]` tokens + `[ANS]`; resize embeddings; init slot embeddings
  from mean embedding + noise.
- `src/minijev/schema.py`: `Choice`, `Noul`, `Score`, `Decision` dataclasses;
  serializer that writes options as `[S_i] text` with shuffled slot order.
- `src/minijev/model.py`: `decide(state, questions)` → masked softmax over slots
  read at each `[ANS]`.
- Tests: output always within declared options; probs sum to 1; slot-order
  shuffle invariance (after training).
- **Gate:** API works end-to-end on the untrained model (random but well-typed).

## Phase 3 — Synthetic data generator (≈4 days)
- `src/minijev/data/`: converters for CLINC150, MNLI, BoolQ, SST-2, AG News,
  ChaosNLI (soft labels) → `{state, questions, targets}` JSONL.
- Augment: random option subsets (2–255), distractors, paraphrased prompts,
  JSON vs prose state, multi-question records (1–8 per state).
- Teacher-labeled set (optional, budgeted): ~5–10k synthetic states labeled
  with option probabilities from a frontier LLM; cache results to disk.
- **Hold out Banking77** (zero-shot eval) and a slice of each source (dev/test).
- **Gate:** ≥200k questions; label-distribution report; no Banking77 leakage
  (test asserts it).

## Phase 4 — RLCD-lite training (≈1 week of Kaggle runs)
- `src/minijev/train.py`: LoRA (r=16) on Qwen3-1.7B, fp16; loss = Brier
  (config flag for log score); slot-head tied to slot embeddings.
- Resume-from-checkpoint across Kaggle sessions (reuse current notebook
  mechanism).
- Run matrix: {Brier, log} × {with, without teacher soft labels}.
- **Gate:** on held-out sources, accuracy ≥ SFT-cross-entropy baseline and
  ECE ≤ 0.05 with fitted T in [0.8, 1.3].

Learn: proper scoring rules — why Brier's optimum is the true distribution.

## Phase 5 — Evaluation report (≈2 days)
- Zero-shot Banking77 (never trained on) vs the repo's fine-tuned classifier.
- Reliability diagrams, risk–coverage curves, Cov@5%.
- Latency table on T4: MiniJev vs generate-and-parse, k questions per state.
- Write `reports/minijev-results.md`.
- **Gate:** report shows where MiniJev wins and loses, with bootstrap CIs.

## Phase 6 — Serving (≈2 days)
- Extend `src/serve.py` with `POST /v1/systemone` matching the `decide()` schema.
- Optional: vLLM prefix caching path for throughput.
- **Gate:** p95 latency < 150 ms for 1 state + 8 questions on T4.

## Phase 7 — Stretch
- RLCD stage 2 with rationales (OpenJev Algorithm 1) on a reasoning task.
- >255 options via 2-stage Noul scoring → Choice.
- Qwen3-4B backbone.

## Risks
| Risk | Mitigation |
|---|---|
| Slot binding fails at 1.7B | Fall back to scoring option text tokens directly (Phase 0 style) |
| T4 fp16 overflow | Qwen3 (not Gemma); loss in fp32; grad clipping |
| Kaggle quota | Small run matrix; resume checkpoints; short eval sets during dev |
| Teacher cost | Teacher set is optional; real datasets alone suffice for v1 |
| Overclaiming vs Jev | Report results as "MiniJev", never as Jev parity |

## Out of scope
Pre-training, multimodal input, multi-label outputs, matching Jev's benchmark claims.
