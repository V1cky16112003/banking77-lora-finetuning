# Project Logbook — Banking Intent Classifier (LoRA/QLoRA Fine-Tuning)

Running record of every decision, scope change, and milestone on this project, in chronological order. Purpose: a single place to answer "why is it built this way" and "what's left" without re-deriving it from git history or memory search.

**How to use this file:** append new entries at the bottom of the relevant date, under `## Status` for the current snapshot. Don't rewrite history — if a decision is later reversed, add a new entry that references and supersedes it (see the 2026-09-18 entry for the pattern).

---

## Status (as of 2026-09-21)

- **Stage:** Scaffold complete, all 23 tests passing, design re-validated against 2026 market data, architecture diagram published. Real Colab training run has **not** happened yet — README results are still placeholders.
- **Scope:** Banking77 intent classification (77 classes), Qwen2.5-1.5B-Instruct, 4 fine-tuning configs (3 bf16 LoRA + 1 QLoRA, added 2026-09-18), config-driven eval harness, FastAPI + Gradio demo (stretch/cuttable).
- **Immediate next step:** Run `notebooks/colab_orchestration.ipynb` end to end (see TODOS.md P0) — dataset integrity check, baseline eval, all 4 training configs, fine-tuned re-eval, generate README tables.
- **What's accomplished:** see [Accomplished](#accomplished) below.
- **What's next / not done:** see [Open Items](#open-items) below.

---

## 2026-09-12 — Problem selection and design

**Context:** Building an AI/ML Engineer CV for the London market (2026). Market research on job postings identified a specific gap: no hands-on evidence of adapting a pretrained model's weights. Existing portfolio covers RAG, agentic systems, retrieval, observability — nothing on fine-tuning.

**Key decisions:**
- **Task chosen: Banking77 intent classification** (77 single-label classes, ~13k examples, public HF dataset `PolyAI/banking77`) — chosen over two rejected alternatives:
  - *GoEmotions (multi-label emotion)* — rejected: multi-label eval (threshold tuning, label-parsing fragility on a decoder model) adds risk not worth it on a 1-2 week budget.
  - *Synthetic structured-JSON extraction* (category+urgency+sentiment) — rejected: no existing public dataset with this label set; synthesizing one introduces label-quality risk and harder eval logic. Most differentiated story of the three, but too risky for the timeline.
- **Model: Qwen2.5-1.5B-Instruct** — small, open-weight, fits a free Colab T4 (16GB VRAM) in plain bf16 without 4-bit quantization.
- **Method: plain LoRA (bf16), not QLoRA, as primary** (later revised — see 2026-09-18). Rationale at the time: QLoRA's 4-bit quantization is calibrated for 7B+ models; unnecessary complexity for 1.5B on a T4. Researched via 2026 LoRA/QLoRA best-practice check: Unsloth identified as the dominant 2026 fine-tuning framework generally, but the project deliberately stayed on vanilla HF `transformers`+`peft` to demonstrate HF-internals fluency rather than an abstraction layer.
- **Task adaptation: decoder-only generation, not encoder classifier head** — matches "LLM fine-tuning" as it appears in job specs, the specific gap being closed.
- **Held-out test set methodology:** frozen before training/hyperparameter selection, touched only once for final evaluation, identical for baseline and fine-tuned eval.
- **Architecture: modular pipeline** (`data.py`, `train.py`, `evaluate.py`, `serve.py`) over notebook-only — 2026 hiring signals reward demonstrable evaluation rigor and reusable tooling over one-off notebooks.
- **`evaluate.py` built config-driven, not Banking77-specific** — label set and prompt template live in `configs/task.yaml`; the eval harness itself is a reusable capability across any single-label classification task. Generalization boundary explicitly capped at single-label (multi-label deferred — reintroduces the complexity GoEmotions was rejected for).
- **Constrained decoding: custom `PrefixConstrainedLogitsProcessor` in vanilla `transformers`**, not the `outlines` library — no new dependency, no Colab version-compatibility risk, demonstrates HF internals fluency.
- **Confusion-matrix ranking:** top 15-20 classes ranked by total misclassification involvement (FP + FN combined), not FN count alone — a recruiter-scannable heatmap, not a full 77×77 matrix.
- **3 starting LoRA configs specified:** r=8/α=16 (`q_proj,v_proj`), r=16/α=16 (`q_proj,v_proj`), r=16/α=32 (`all-linear`); shared schedule LR 1e-4 to 2e-4, 2-3 epochs, effective batch size ~16-32.
- **Metric: macro-F1 as headline** (treats all 77 classes equally), per-class F1 as README appendix.
- **Scope caps accepted:** auto-generated README tables from eval JSON, confusion heatmap (top 15-20 classes), Gradio/static-HTML demo (contingent on FastAPI shipping), GitHub Actions CI on non-model logic only (label parsing + metrics — no GPU on runners).
- **Cut priority if behind schedule:** CI → demo → confusion heatmap → auto-tables. Core `evaluate.py` rigor is never cut. FastAPI endpoint flagged as the Success Criterion most tangential to the actual skill gap (fine-tuning, not deployment) — downgrade to stretch before cutting eval rigor.
- **Deferred to `TODOS.md`:** `/health` + `/model-info` FastAPI endpoints (P3), README cost-equivalent line item (P3).

**Process:** Design went through 3 rounds of adversarial spec review (16 issues found round 1-2, all fixed, 9/10 quality score), then a separate CEO-plan review cycle (3 iterations, 21 issues found and fixed, 9/10 quality score). Design doc committed to `docs/designs/lora-finetuning-banking-intent-classifier.md`.

---

## 2026-09-13 — CEO plan, engineering review, outside-voice review

**Key decisions/changes:**
- CEO plan formalized the 5 scope expansions from the design phase (auto-tables, confusion heatmap, demo, CI, config-driven `evaluate.py`) with explicit effort estimates, dependencies, and cut order.
- **GitHub remote setup elevated to explicit "Step 0"** (~10-15 min) — blocks GitHub Actions CI; without it, CI stays permanently blocked, not merely deferred.
- **`evaluate.py` generalization scope capped** at single-label decoder-classification only — effort re-estimated M (45-60 min) to reflect enforcing that cap rather than open-ended generalization.
- Engineering review (`/plan-eng-review`, FULL_REVIEW): 18 findings resolved, zero critical gaps. Plus a second outside-voice review round after a Codex connection failure (5 reconnect attempts, delegated to a Claude subagent instead) surfaced 7 more findings (E1-E7), all resolved — covering eval robustness, torch version pinning strategy, repo hygiene, and few-shot prompt strategy.
- **Decision D7 reversed:** the demo prompt must keep the *full* 77-label list rather than dropping labels for latency — engineering review flagged that trimming labels would cause distribution shift between the demo and the frozen-eval methodology. Demo now matches the eval prompt exactly.
- **TODOS.md created** to hold the two deferred P3 items (`/health`/`/model-info`, cost-equivalent line) with explicit rationale and no dependency on accepted scope.
- Durable engineering pitfalls logged: Python import isolation, prompt-format correctness risk, Colab torch-pin/CUDA conflict risk.

---

## 2026-09-14 — Design review (demo UI)

**Key decisions:**
- Demo UI classified as **OPERATE mode** (user completes a task directly, not persuaded to act) — informs a plain, task-first layout: input primary/auto-focused, submit + example chips secondary, result area tertiary.
- **Success state shows only the top-1 predicted label, no confidence score** — the model isn't calibrated for reliable confidence estimates, and showing one would misrepresent the actual eval methodology (top-1 accuracy/macro-F1).
- Loading state pairs a spinner with explicit copy ("Running inference on CPU — this can take a few seconds") to manage expectations for multi-second CPU-only latency.
- Visual styling deliberately uses Gradio's stock theme / unstyled static HTML — explicit choice, not oversight: project priority is fine-tuning rigor over demo polish.
- 7-pass design review raised the demo's design score from 5/10 to 9/10 (information architecture pass was the largest single jump, 4/10 → 9/10).
- Visual mockup generation failed 3/3 times (OpenAI API key returned 401) — review proceeded text-based; not blocking.

---

## 2026-09-17/18 — Post-approval premise challenge (`/office-hours` grill session)

**Context:** With the scaffold built and 23/23 tests passing, before running the real Colab training, the design was re-examined against **fresh 2026 London AI/ML market data** rather than treating the 2026-09-12 approval as final.

**Research findings that drove the change:**
1. 2026 London job postings list **QLoRA as the default PEFT method**, independent of model size — this project's "plain LoRA only" premise reads as a gap unless explicitly defended with a comparison.
2. **Banking77 + LoRA is a common existing portfolio pattern** — multiple public repos already do FLAN-T5/RoBERTa/DeBERTa/BERT + LoRA on this exact dataset. The task choice itself isn't being reversed (re-litigating it now would reintroduce the dataset/eval risk that ruled out GoEmotions and structured-JSON in the first place) — but it means the task alone buys no differentiation.
3. Banking77 has a **published label-quality flaw** (some labels are aggregations with off-topic examples, e.g. `card_about_to_expire` reportedly containing ~30 off-topic examples) — a live risk to the project's "defend every number" pitch if undetected.
4. Job specs also list quantization/pruning/distillation, vLLM/llama.cpp serving, W&B/MLflow tracking, DPO/ORPO alignment — none of which this project touches; acknowledged as an accepted scope boundary, not fixed this round.

**Decisions taken (superseding earlier scope where noted):**
- **QLoRA promoted from "optional stretch" (2026-09-12 Open Question) to a required 4th comparison config.** Implemented same session in `src/train.py`: new `LoraRunConfig(use_qlora=True)` field, 4th `CANDIDATE_CONFIGS` entry `r16_a32_all_qlora` (mirrors config 3's rank/alpha/targets so the LoRA-vs-QLoRA delta isolates the quantization effect only), 4-bit NF4 loading via `BitsAndBytesConfig` + `prepare_model_for_kbit_training`. `bitsandbytes==0.44.1` added to `requirements.txt`.
- **Dataset Integrity Check added** as an explicit step (design doc Technical Notes, Success Criteria, Next Steps #2; new README section) — per-class counts and label spot-checks now a required, documented part of data prep, not an implicit assumption.
- **README pitch reframed:** headline claim shifted from "I fine-tuned a model" to "I built a config-driven eval harness with a frozen-test-set methodology that produced an evidence-based LoRA-vs-QLoRA comparison and caught a real dataset flaw" — same underlying work, leading with the part that's actually rare rather than the part that's now table-stakes.
- **Option considered and rejected:** swapping Banking77 for a less-genericized dataset — rejected, would reintroduce label-quality/effort risk the original design explicitly avoided, no timeline slack to absorb it.
- Design doc (`docs/designs/lora-finetuning-banking-intent-classifier.md`) updated in place: Premises 1 and 5, Constraints, Approach A open question, Technical Notes (LoRA configs + new dataset-integrity note), Success Criteria, Next Steps, and a new **Decision Log** section recording this entry's reasoning inline for future reference.
- `TODOS.md`: added **P0 — Run the real Colab training + eval and populate README results** (the one concrete remaining step); the two P3 items from 2026-09-12 remain deferred, unaffected.
- All 23 existing tests re-verified passing after the `train.py` change (no CI-testable coverage for the QLoRA path itself — it's a GPU-only code path, consistent with the existing "not CI-testable, no GPU" note on `train.py`).

---

## 2026-09-21 — Architecture diagram

Created the first architecture diagram for the project (none existed previously — decisions were documented in prose only, in the design doc and this logbook). Generated via the `/diagram` skill as a Mermaid source + rendered SVG/PNG + Excalidraw scene, all committed under `diagrams/`.

- `diagrams/architecture.mmd` — source of truth, edit this and re-render.
- `diagrams/architecture.svg` / `.png` — rendered, embedded in README under a new "Architecture" section.
- `diagrams/architecture.excalidraw` — opens at excalidraw.com; flattened to a single image element rather than per-node-editable, because Mermaid `subgraph` blocks don't convert cleanly to native Excalidraw elements (noted in README so it's not mistaken for a bug).
- Diagram shows the 6-layer split (Config, Data, Training, Evaluation, Serving, Reporting) and data flow already described in the design doc's Repo Structure/Technical Notes — no new architectural decisions, just the first visual representation of existing ones.
- Referenced from both `README.md` (new Architecture section) and the design doc (new Architecture Diagram section, before Distribution Plan).

## Accomplished

- [x] Design doc written, reviewed (3-round adversarial spec review + CEO plan review + engineering review + outside-voice review + design review), all rounds converged to 9/10 quality, zero unresolved findings.
- [x] Modular scaffold: `src/config.py`, `src/data.py`, `src/evaluate_core.py` (CI-tested, zero ML deps), `src/evaluate.py` (ML orchestration), `src/train.py`, `src/serve.py` (FastAPI), `demo/app.py` (Gradio).
- [x] Config-driven task definition (`configs/task.yaml`) — single source of truth for label list, prompt template, decoding config.
- [x] `scripts/generate_report.py` (results table + confusion heatmap + bootstrap CI) and `scripts/smoke_test.py` (post-deploy verification).
- [x] `notebooks/colab_orchestration.ipynb` scaffolded.
- [x] Test suite: 23/23 passing (`tests/test_config.py`, `tests/test_evaluate_core.py`).
- [x] Git repo initialized, design doc and scaffold committed.
- [x] Demo UI design reviewed (7-pass, 9/10 score), OPERATE-mode interaction model documented.
- [x] 2026-09-18: 4th QLoRA comparison config added to `src/train.py`; `bitsandbytes` dependency added; README and design doc updated to reflect the LoRA-vs-QLoRA comparison and the evaluation-first framing.
- [x] 2026-09-21: Architecture diagram created (`diagrams/architecture.{mmd,svg,png,excalidraw}`) covering Config/Data/Training/Evaluation/Serving/Reporting layers and data flow; embedded in README and referenced from the design doc.

## Open Items

- [ ] **P0:** Run the actual Colab training + eval (all 4 configs) and populate the README results table, confusion heatmap, and "What Worked/What Didn't" section — nothing here is real yet, everything is a placeholder pending this run.
- [ ] Dataset Integrity Check: verify per-class counts and spot-check label noise (`card_about_to_expire` and others) against the actually-loaded `PolyAI/banking77` dataset; document findings in the README.
- [ ] Pin the exact `PolyAI/banking77` HF dataset revision hash in `configs/task.yaml` (currently `null`, marked TODO).
- [ ] Populate `configs/task.yaml`'s `labels: []` from the loaded dataset's feature names (currently a placeholder, by design — never hand-transcribed).
- [ ] GitHub remote + GitHub Actions CI (blocked on remote creation; CI itself scoped to unit tests on `evaluate_core.py` only, no model inference).
- [ ] FastAPI + Gradio demo — build and verify live; this is the most cut-able item if time runs short (see design doc risk note); consider a recorded GIF as a live-demo fallback given CPU-only multi-second latency.
- [ ] P3 (deferred, low priority): `/health` + `/model-info` endpoints on `serve.py`.
- [ ] P3 (deferred, low priority): cost-equivalent line item in README (free Colab T4-hours → paid-GPU dollar equivalent).
