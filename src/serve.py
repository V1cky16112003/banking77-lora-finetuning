"""FastAPI demo endpoint. No auth, no scaling, no production infra (design doc
premise 4) — a local demo for interview purposes.

Applies: fail-fast on missing/corrupted adapter (1A), structured error
responses (2A), input validation with length cap (3A), metadata-only logging
that never persists raw input text (3B).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_task_config
from src.evaluate import generate_labels_batch
from src.evaluate_core import format_label_for_display, parse_label
from src.train import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 500

_config = None
_model = None
_tokenizer = None


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_MESSAGE_LENGTH)


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_label_display: str


def load_model(adapter_dir: str) -> None:
    """1A: fails fast and refuses to start if the adapter checkpoint is
    missing/corrupted — never silently falls back to serving the base model,
    which would look like a working demo while actually showing unmodified
    base-model behavior."""
    global _config, _model, _tokenizer

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        logger.error(f"Adapter checkpoint not found or empty at {adapter_dir}")
        sys.exit(
            f"FATAL: adapter checkpoint missing or empty at '{adapter_dir}'. "
            f"Refusing to start rather than silently serving the base model. "
            f"Download the checkpoint from Drive before starting serve.py."
        )

    _config = load_task_config()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    try:
        _model = PeftModel.from_pretrained(base_model, adapter_dir)
    except Exception as exc:
        logger.error(f"Failed to load adapter from {adapter_dir}: {exc}")
        sys.exit(f"FATAL: adapter at '{adapter_dir}' failed to load: {exc}")

    logger.info(f"Model loaded successfully from adapter at {adapter_dir}")


@asynccontextmanager
async def _lifespan(_: FastAPI):
    adapter_dir = os.environ.get("ADAPTER_DIR")
    if not adapter_dir:
        sys.exit("FATAL: set ADAPTER_DIR to the downloaded adapter checkpoint directory.")
    load_model(adapter_dir)
    yield


app = FastAPI(title="Banking Support Intent Classifier", lifespan=_lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.monotonic()
    # 3B: log request metadata only — never the raw input text, since a live
    # demo could have real-looking sensitive text typed into it.
    logger.info(f"predict request received, length={len(request.message)}")

    try:
        # Demo keeps the full label list in the prompt (D7 reversed) —
        # matches the fine-tuning distribution exactly, per config.format_prompt.
        prompt = _config.format_prompt(request.message)
        generated_texts = generate_labels_batch(_model, _tokenizer, [prompt], _config)
        predicted_label = parse_label(generated_texts[0], _config.labels)
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error(f"predict generation failed after {elapsed:.2f}s: {exc}")
        raise HTTPException(status_code=500, detail={"error": "generation_failed"}) from exc

    if predicted_label is None:
        elapsed = time.monotonic() - start
        logger.info(f"predict unresolved after {elapsed:.2f}s")
        raise HTTPException(status_code=500, detail={"error": "unresolved_prediction"})

    elapsed = time.monotonic() - start
    logger.info(f"predict succeeded in {elapsed:.2f}s")
    return PredictResponse(
        predicted_label=predicted_label,
        predicted_label_display=format_label_for_display(predicted_label),
    )


# Note: a /health + /model-info endpoint was explicitly deferred to TODOS.md
# (item 6, from the office-hours cherry-pick ceremony) — not built here.
