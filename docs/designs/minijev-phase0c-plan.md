# MiniJev Phase 0c: Profile the read-out (execution plan)

Parent: `minijev-phase0b-plan.md`. Date: 2026-10-04.

## Hypothesis (from local token counting, before any GPU time)
The read-out is **compute-bound by design**, not slowed by a bug.

| Per message | Token positions through the 1.5B model |
|---|---|
| Phase 0b stage B (77 labels padded to 10) | **693** (309 real, 55% padding) |
| Same labels as a shared-prefix tree (trie) | **255** |
| Message suffix | ~11 |
| Static prefix (now cached once per run) | 397, amortised to ~0 |

At ~3 GFLOP per token, 704 positions ≈ 2.1 TFLOP per message. A T4 sustains
roughly 20–25 TFLOP/s in fp16, so ≈ 90–100 ms of pure matmul per message even
before attention. Phase 0b measured 336 ms.

**Prediction:** stage B is ≥80% of the time; achieved throughput is within ~3×
of the matmul roofline; LoRA (unmerged, all-linear) adds a measurable
overhead; and even a perfect trie (255 positions) cannot reach 5× faster than
generation (~33 ms/example).

## What the profiling run measures
`scripts/profile_readout.py` (n=48 val messages, adapter `r16_a32_all`):
1. Environment: GPU, torch/transformers versions, attention implementation in use.
2. Roofline: achieved fp16 TFLOP/s on a large matmul on this GPU.
3. Per-stage CUDA-synchronised timings from `LabelScorer.score(..., timings=)`:
   suffix pass, KV repeat, label pass, logsumexp. Batch sizes 1, 4 and 8.
4. Token positions per message, and model FLOPs → efficiency against the roofline.
5. The same at B=4 with LoRA merged into the base weights (`merge_and_unload`).
6. `torch.profiler` top-15 CUDA kernels for one B=4 batch.

Output: `minijev-profile/profile.json`, plus a printed report.

## Kaggle job runner (do this once)
The Kaggle notebook's cells live on Kaggle, not in git, so every new job meant
editing the notebook. `notebooks/kaggle_minijev_job.ipynb` is a thin runner:
clone `jev`, install, find the adapter, run `python -m scripts.kaggle_job`.
The job is chosen in the repo (`scripts/kaggle_job.py`), so future runs need
only **Save & Run All**.

## Decision rule after the run
- **Compute-bound confirmed:** exact multi-token scoring can't pass the 5× gate
  on a T4. Re-scope the gates: Phase 0 passes on exactness + calibration (done),
  and the speed gate moves to Phase 2, where every option is a single slot
  token. That makes the read-out ~11 positions per message. This is the core
  reason Jev answers with fixed option slots instead of strings.
  Optional Phase 0d: trie + tree attention (693 → 255 positions, ~2.7×) as an
  exact intermediate step.
- **Not compute-bound** (low efficiency, one kernel dominating): fix that
  specific kernel and re-run the Phase 0 gate.
