"""LoRA fine-tuning (plain bf16 LoRA, per design doc premise 1 — no QLoRA needed
for a 1.5B model on a T4). Run in Colab; not CI-testable (no GPU).

Checkpoints to Google Drive after each epoch (design doc: "Colab session
continuity"). Checks for an existing checkpoint before overwriting (finding
4B) rather than silently clobbering a completed run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.config import TaskConfig
from src.data import BankingSplits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class LoraRunConfig:
    """One of the 2-3 candidate configs (design doc Technical Notes)."""

    name: str
    rank: int
    alpha: int
    target_modules: list[str]
    learning_rate: float = 1.5e-4  # midpoint of the 1e-4 to 2e-4 range
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch size ~16


CANDIDATE_CONFIGS = [
    LoraRunConfig(name="r8_a16_qv", rank=8, alpha=16, target_modules=["q_proj", "v_proj"]),
    LoraRunConfig(name="r16_a16_qv", rank=16, alpha=16, target_modules=["q_proj", "v_proj"]),
    LoraRunConfig(name="r16_a32_all", rank=16, alpha=32, target_modules=["all-linear"]),
]


def checkpoint_exists(output_dir: Path) -> bool:
    """4B: check before overwrite, rather than silently clobbering a
    completed run (e.g. after an accidental re-run following a Colab
    disconnect/reconnect)."""
    return output_dir.exists() and any(output_dir.iterdir())


def train_lora(
    config: TaskConfig,
    splits: BankingSplits,
    run_config: LoraRunConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    if checkpoint_exists(output_dir) and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {output_dir} for run '{run_config.name}'. "
            f"Pass overwrite=True to intentionally replace it, or choose a different "
            f"output_dir. Refusing to silently overwrite a completed training run."
        )

    logger.info(
        json.dumps(
            {
                "event": "train_start",
                "run_name": run_config.name,
                "rank": run_config.rank,
                "alpha": run_config.alpha,
                "target_modules": run_config.target_modules,
                "learning_rate": run_config.learning_rate,
                "epochs": run_config.epochs,
                "batch_size": run_config.batch_size,
                "train_examples": len(splits.train),
            }
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=run_config.rank,
        lora_alpha=run_config.alpha,
        target_modules=run_config.target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def _format_example(example: dict) -> dict:
        true_label = splits.labels[example["label"]]
        prompt = config.format_prompt(example["text"])
        full_text = prompt + " " + true_label
        return tokenizer(full_text, truncation=True, max_length=512)

    tokenized_train = splits.train.map(_format_example, remove_columns=splits.train.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=run_config.epochs,
        per_device_train_batch_size=run_config.batch_size,
        gradient_accumulation_steps=run_config.gradient_accumulation_steps,
        learning_rate=run_config.learning_rate,
        save_strategy="epoch",  # checkpoint after each epoch (design doc: per-epoch is sufficient)
        logging_steps=10,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    train_result = trainer.train()

    logger.info(
        json.dumps(
            {
                "event": "train_end",
                "run_name": run_config.name,
                "final_loss": train_result.training_loss,
            }
        )
    )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
