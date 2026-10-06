# Pointwise Phase 4: RLCD-lite training (execution plan)

Parent: `pointwise-plan.md` Phase 4. Date: 2026-10-06.

## Goal
Train the pointer head and a LoRA adapter so the Phase 2 decision API gives
calibrated answers, and measure them on sources it never trained on.

## Budget
Kaggle quota at the start: 29.5 GPU-hours left until 2026-10-10. Sessions are
capped at 12h, and a session that times out or is cancelled saves **no output**.
- Session 1: ~11h (setup ~10 min, training 10.5h, evaluation ~10 min).
- Session 2: same, resuming from session 1.
- ~7h left for Phase 5 evaluation.
Both T4s are used (DDP): quota counts session time, not GPUs, so two GPUs train
about twice as much for the same quota.

## Design (`src/pointwise/train.py`)
| Choice | Value | Why |
|---|---|---|
| Base | Qwen3-1.7B-Base, fp16, frozen | T4 has no bf16; Phase 1–2 ran on it |
| Adapter | LoRA r=16, α=32, dropout 0.05, q/k/v/o/gate/up/down, fp32 | Kev's recipe |
| Head | pointer head, fp32 (autocast disabled around it) | small logits differences matter for calibration |
| Loss | log score −Σ q log p (+ optional Brier, off) | strictly proper; equals CE for one-hot targets |
| Optimiser | AdamW, LR 2e-4 (LoRA) / 1e-3 (head), no weight decay, grad clip 1.0, fp16 GradScaler | standard LoRA settings |
| Schedule | 200-step warmup, cosine to 10% over the planned steps | length set from measured step time × 21 planned hours |
| Batches | ≤ 4096 padded tokens per GPU, length-sorted within chunks of 1000 | little padding; largest batch first so OOM appears at once |
| Context | state ≤ 384, state + question ≤ 1024, packed ≤ 2048 tokens; longer records dropped whole | cutting a state could cut the evidence a label depends on |
| Memory | gradient checkpointing (non-reentrant) | fits T4's 15 GB |

Kaggle safety:
- Stops after `--max-hours 10.5` of training and saves; checkpoints every 30 min
  (written to a temp file and renamed, so a kill mid-save can't corrupt it).
- Resumes from the newest `pointwise-train/` (or pre-rename `minijev-train/`) checkpoint under
  `/kaggle/input` (attach the previous version) or its own output dir:
  adapter, head, optimiser, GradScaler, step and schedule length.
- `scripts/kaggle_job.py` builds the data into `/tmp` (~2 min, not saved as
  output), runs torchrun on every GPU, and retries once with half the token
  budget if training fails in its first 20 minutes.

End of session (rank 0): 1,500 dev records and up to 500 records per held-out
source are scored; T is fitted on dev (golden-section search on mean NLL);
`metrics.json` has accuracy / NLL / Brier / ECE per source, raw and scaled, plus
chance accuracy; `heldout_predictions.jsonl` keeps every logit.

## Local verification
- `tests/test_pointwise_metrics.py` (5, CI-safe): softmax, per-question scores,
  temperature fitting recovers a known overconfidence (T = 3), T never changes accuracy.
- `tests/test_pointwise_train.py` (8, tiny random Qwen3 saved to disk, CPU):
  batched forward == one at a time; batches cover every example once within budget,
  largest first; batch stream crosses epochs and gives each rank its own batch;
  LR schedule; a full `main()` run checkpoints and evaluates; a second run resumes
  at the saved step; resume from a "previous session" directory; loss falls on a
  small set; **a real two-process DDP run** (gloo) completes.
- Data build checked with the Kaggle stack (Python 3.11, datasets 3.1.0).
- Bug found by the tests: the training loop passed `Example` wrappers instead of
  encoded requests to the model.

## Gate
On held-out out-of-domain sources (QNLI, PAWS, Emotion, TweetEval sentiment):
ECE ≤ 0.05 after temperature scaling, and accuracy clearly above chance per
source. Reported, not gated: raw ECE, fitted T (Kev's adapters land at 2.2–2.4),
dev accuracy per training source, zero-shot Banking77 accuracy.

## How to run
Session 1: notebook *mini-jev-phase-0*, Accelerator **GPU T4 x2**, Internet on,
**Save & Run All**.
Session 2: **Add Input → Your Work → this notebook's session-1 version**, then
**Save & Run All**. The log should say `resumed from … at step N`.

## Status
- [x] Training code, tests, Kaggle job.
- [ ] Session 1.
- [ ] Session 2.

## Run log
- **2026-10-06 05:25 UTC, session 1, attempt 1: failed in 4 minutes, ~4.5 GPU-min used.**
  The data build passed on Kaggle; training died while applying LoRA. Kaggle's image
  ships torchao 0.10.0, and peft 0.21.2 raises on any torchao below 0.16 (an absent
  torchao is fine; it wasn't installed locally, so the tests passed). The OOM retry
  then repeated the same error. Fix: the job uninstalls torchao (Pointwise doesn't
  quantise) and runs a LoRA preflight on a tiny model first, so environment errors
  stop in seconds with a clear message. Reproduced locally with a stub torchao 0.10.0:
  same error with it, preflight passes without.

