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
  on a T4. Phase 0 passes on exactness + calibration (done), and the speed gate
  is **re-derived, not just moved** (see correction below).
  Optional Phase 0d: trie + tree attention (693 → 255 positions, ~2.7×) as an
  exact intermediate step. Borderline by this plan's own numbers: 255 + 11
  positions × 3.1 GFLOP at 25 TFLOP/s is ~33 ms, exactly the 5× target.

> **Correction (2026-10-04, after comparing with Kev).** An earlier version of
> this rule said a slot/pointer read-out is "~11 positions per message". It is
> not: the 77 options must still be *in the input*, after the per-message state,
> so they cannot be prefix-cached across messages. A Jev/Kev-style read-out
> processes ~11 (message) + ~400 (77 option spans) ≈ 410 positions per message,
> against generation's ~470 prefill + ~10 sequential decode steps. Its speed-up
> over generation comes from deleting the decode steps, not from fewer
> positions, so expect ~1.5–3×, not 5×. The Phase 2 gate in `minijev-plan.md`
> is rewritten accordingly: per-question cost and questions-per-state scaling,
> not "5× faster than generating one Banking77 label".
- **Not compute-bound** (low efficiency, one kernel dominating): fix that
  specific kernel and re-run the Phase 0 gate.

## Run log
- **2026-10-04 21:21 UTC, run 1: CUDA OOM at batch size 8** in the label pass, inside transformers'
  `repeat_kv`. Qwen2.5-1.5B has 12 query heads over 2 KV heads, and 4.46's SDPA path materialises
  the KV cache 6x per row per layer (616 rows x ~470 positions). That 6x copy also runs at B=4, so it
  is extra memory traffic worth measuring, not only an OOM. The profiler lost the B=1/B=4 results
  because it only saved at the end. Fixed: per-batch-size OOM is recorded as a result, batch
  sizes are now 1/2/4/8, and the JSON is saved after every step.
- **Profiler review fixes (not in run 2's code if it was already cloned):** the top-ops table
  ranked by inclusive CUDA time, which lists nested ops (`aten::linear` > `aten::matmul` >
  `aten::mm`) as separate entries counting the same kernel time several times. It now ranks by
  self GPU time (`self_gpu_ms`, `self_gpu_share`) and reads `self_device_time_total` on newer torch.
  Read run 2's `top_kernels_b4` by `self_cuda_share`, not `cuda_ms`.
