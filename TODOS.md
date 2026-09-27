# TODOS

## P0 — Run the real Colab training + eval (all 4 configs) and populate README results
**What:** Execute `notebooks/kaggle_orchestration.ipynb` end to end: frozen split + dataset integrity check, baseline eval, all 4 training configs (3 bf16 LoRA + 1 QLoRA), fine-tuned re-eval, `scripts/generate_report.py` to fill the README results table and "What Worked/What Didn't" section.
**Why:** Every number in the README is currently a placeholder. This is the one remaining step between "scaffold" and "portfolio artifact."
**Context:** Confirmed still outstanding as of the 2026-09-18 `/office-hours` review; QLoRA arm added to `src/train.py` (`r16_a32_all_qlora`) same session — see `docs/designs/lora-finetuning-banking-intent-classifier.md` Decision Log.
**Effort estimate:** L (human) → depends on GPU queue/Colab session limits, not CC time.
**Priority:** P0
**Depends on / blocked by:** None — unblocked, next action.

## P3 — /health + /model-info endpoint on FastAPI
**What:** Add `/health` and `/model-info` routes to `serve.py` showing loaded adapter config (base model, rank, alpha, target modules).
**Why:** Lets you (or an interviewer) verify what's actually loaded without reading server logs.
**Pros:** Cheap, useful for debugging and demoing config transparency.
**Cons:** Not required for the core CV story (fine-tuning evidence + before/after numbers).
**Context:** Surfaced as expansion candidate #6 during the `/office-hours` Selective Expansion cherry-pick ceremony (2026-09-12), deferred as lower-priority. No dependency on any accepted scope item.
**Effort estimate:** XS (human) → XS (CC+gstack)
**Priority:** P3
**Depends on / blocked by:** None.

## P3 — Cost-equivalent line item in README
**What:** One README line translating the free Colab T4 GPU-hours used into an equivalent paid-GPU dollar cost.
**Why:** Gives interviewers a tangible sense of accessibility/efficiency for the project.
**Pros:** Cheap, concrete number, reinforces the "built on free tier" story.
**Cons:** Not required for the core CV story.
**Context:** Surfaced as expansion candidate #7 during the `/office-hours` Selective Expansion cherry-pick ceremony (2026-09-12), deferred as lower-priority. No dependency on any accepted scope item.
**Effort estimate:** XS (human) → XS (CC+gstack)
**Priority:** P3
**Depends on / blocked by:** None.
