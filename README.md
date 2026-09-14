# Banking Support Intent Classifier — LoRA Fine-Tuning

**A rigorous before/after story: LoRA fine-tuning a 1.5B instruction model to classify banking customer-support messages, with a frozen held-out test set and honest, defensible numbers.**

## Problem

I was building an AI/ML Engineer CV for the London market (2026) and identified a gap through market research on job postings: no hands-on evidence of adapting a pretrained model's weights to a task. This project closes that gap with a concrete artifact — not a toy notebook, but a methodologically rigorous fine-tune with a frozen test set, a reusable eval harness, and a deployed demo.

## Results

*(Populate this table by running `scripts/generate_report.py` against the real eval output — never hand-copy these numbers.)*

| Metric | Baseline (zero-shot) | Baseline (few-shot) | Fine-tuned (LoRA) |
|---|---|---|---|
| Accuracy | — | — | — |
| Macro-F1 (95% CI) | — | — | — |
| Latency/call (CPU) | — | — | — |

*Confusion heatmap and per-class F1 table generated at `eval_runs/report/`. Note: confusion-pair counts rest on ~40 test examples/class on average (Banking77's 3,080-example test split ÷ 77 classes) — read pair rankings as illustrative, not statistically precise.*

## The Reusable Eval Harness

`src/evaluate_core.py` is config-driven, not Banking77-specific: the label set and prompt template come from `configs/task.yaml`, and the parsing/metrics/confusion-ranking logic works for any single-label decoder-classification fine-tune. This is a deliberate design choice (see `docs/designs/`) — the eval harness itself is a reusable capability, not a one-off script for this project alone. It's also the file GitHub Actions CI actually tests (zero `torch`/`transformers` imports), separated from `evaluate.py`'s ML orchestration layer.

## Training Configuration

- **Model:** Qwen2.5-1.5B-Instruct (open-weight, small enough for plain bf16 LoRA on a free Colab T4 — no QLoRA/4-bit quantization needed).
- **Method:** LoRA via HF `peft` + `transformers` (not QLoRA — see the design doc's premise 1 for why).
- **Configs compared (3):** r=8/α=16 (`q_proj,v_proj`), r=16/α=16 (`q_proj,v_proj`), r=16/α=32 (`all-linear`). Shared schedule: LR 1e-4 to 2e-4, 2-3 epochs, effective batch size ~16-32.
- **Decoding:** greedy (`do_sample=False`) for all scoring — deterministic, reproducible numbers.
- **Label resolution:** constrained decoding (`PrefixConstrainedLogitsProcessor`) restricts generation to the label set; exact/fuzzy-match fallback covers only constraint-initialization failure, not "malformed constrained output" (which can't occur when the constraint applies).
- **Hardware:** free-tier Google Colab T4 (16GB VRAM) for training/eval; local machine (CPU) for serving.
- **Cost:** $0 — entirely free-tier.

## What Worked / What Didn't

*(Fill in honestly after running the real experiment — this is the section that makes the before/after story credible in an interview.)*

## Repo Structure

```
configs/task.yaml       # single source of truth: label list, prompt template, decoding config
src/config.py           # shared config loader
src/data.py             # dataset loading, frozen train/val/test split
src/evaluate_core.py    # pure eval logic (parsing, metrics, confusion ranking) — CI-tested
src/evaluate.py         # ML orchestration: constrained decoding, batched generation
src/train.py            # LoRA fine-tuning
src/serve.py            # FastAPI /predict endpoint
demo/app.py             # Gradio demo UI
scripts/smoke_test.py   # post-deploy verification against known examples
scripts/generate_report.py  # results table + confusion heatmap + bootstrap CI, from eval JSONL
notebooks/colab_orchestration.ipynb  # Colab notebook tying it all together
tests/                  # pytest suite for evaluate_core.py and config.py
```

## Running It

**Training (Colab):** open `notebooks/colab_orchestration.ipynb` in Colab, mount Drive, run cells top to bottom.

**Serving (local):**
```bash
pip install -r requirements.txt
python -c "from src.serve import load_model; load_model('path/to/downloaded/checkpoint')"
uvicorn src.serve:app --reload
python scripts/smoke_test.py   # verify before demoing live
python demo/app.py             # launches the Gradio UI on top of the running API
```

**Tests:**
```bash
pip install -r requirements-core.txt
pytest tests/ -v
```

## Known Limitations

- Single-run results, no seed-variation re-runs (portfolio-scope tradeoff, documented in the design doc).
- Deployment (FastAPI + demo) is the most cuttable component of this project if time runs short — see `docs/designs/` for the explicit cut-order.
- CPU-only serving is multi-second per response — a "works but slow" demo, not a production-latency claim.

## Design Process

The full design doc, CEO scope review, engineering review, and design review — including every decision and its rationale — live in `docs/designs/lora-finetuning-banking-intent-classifier.md` and the linked review artifacts.
