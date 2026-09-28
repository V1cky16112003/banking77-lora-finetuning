"""LoRA fine-tuning. Primary method is plain bf16 LoRA (design doc premise 1 —
no QLoRA needed for a 1.5B model on a T4 purely to fit VRAM). A 4th config adds
QLoRA (4-bit NF4 + LoRA) as an explicit comparison arm, not because this model
needs the memory savings, but to report the LoRA-vs-QLoRA quality/speed/memory
trade-off directly — see docs/designs/lora-finetuning-banking-intent-classifier.md,
"Decision Log" 2026-09-18 (office-hours premise challenge: 2026 job specs treat
QLoRA as the default, so the README needs to show, not just argue, when plain
LoRA is the better call). Run in Colab; not CI-testable (no GPU).

Checkpoints to Google Drive after each epoch (design doc: "Colab session
continuity"). A finished run is skipped and a partial run resumes from its last
epoch checkpoint, rather than silently clobbering either (finding 4B).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config import TaskConfig
from src.data import BankingSplits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _use_bf16() -> bool:
    """bf16 needs Ampere+ (compute capability 8.x). The T4 (7.5) only emulates
    it, which makes training crawl, so fall back to fp16 there."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def compute_dtype() -> torch.dtype:
    return torch.bfloat16 if _use_bf16() else torch.float16


class _PrintProgressCallback(TrainerCallback):
    """tqdm's in-place progress bar never reaches a Kaggle commit run's log, so
    print one plain line per logging step instead."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.start = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        elapsed = time.monotonic() - self.start
        eta = elapsed / max(state.global_step, 1) * (state.max_steps - state.global_step)
        print(
            f"[{self.run_name}] step {state.global_step}/{state.max_steps} "
            f"loss={logs['loss']:.4f} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
            flush=True,
        )

# Masked out of the loss (PyTorch cross-entropy's ignore_index).
_IGNORE_INDEX = -100


@dataclass(frozen=True)
class LoraRunConfig:
    """One of the 4 candidate configs (design doc Technical Notes + the
    2026-09-18 QLoRA-arm decision)."""

    name: str
    rank: int
    alpha: int
    # PEFT only special-cases "all-linear" as a bare string; inside a list it
    # is looked up as a literal module name and fails.
    target_modules: list[str] | str
    learning_rate: float = 1.5e-4  # midpoint of the 1e-4 to 2e-4 range
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch size ~16
    use_qlora: bool = False  # 4-bit NF4 base + LoRA, instead of plain bf16 LoRA


CANDIDATE_CONFIGS = [
    LoraRunConfig(name="r8_a16_qv", rank=8, alpha=16, target_modules=["q_proj", "v_proj"]),
    LoraRunConfig(name="r16_a16_qv", rank=16, alpha=16, target_modules=["q_proj", "v_proj"]),
    LoraRunConfig(name="r16_a32_all", rank=16, alpha=32, target_modules="all-linear"),
    LoraRunConfig(
        name="r16_a32_all_qlora",
        rank=16,
        alpha=32,
        target_modules="all-linear",
        use_qlora=True,
    ),
]


def load_base_model(use_qlora: bool) -> PreTrainedModel:
    """Base model on GPU 0, plain 16-bit (see compute_dtype) or 4-bit NF4. Shared by training and
    eval so a QLoRA adapter is evaluated on the same quantized base it was
    trained on."""
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
        )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=compute_dtype(), device_map={"": 0}
    )


def is_training_complete(output_dir: Path) -> bool:
    """The final save_pretrained writes adapter_config.json at the top level;
    per-epoch checkpoints only write it inside checkpoint-*/ subdirectories."""
    return (output_dir / "adapter_config.json").exists()


def _has_epoch_checkpoint(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.glob("checkpoint-*"))


def build_training_example(tokenizer: PreTrainedTokenizer, prompt: str, label: str) -> dict:
    """Prompt tokens are masked out of the loss so training signal comes only
    from the label + EOS. Otherwise the ~450-token prompt (mostly the fixed
    77-label list) dominates the loss and the model mostly learns to recite it.

    The label is encoded bare and appended directly after the prompt's
    trailing newline — exactly the token ids evaluate.py's constrained
    decoding allows, so training and eval see the same label tokenization."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [_IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def train_lora(
    config: TaskConfig,
    splits: BankingSplits,
    run_config: LoraRunConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    if is_training_complete(output_dir) and not overwrite:
        logger.info(
            json.dumps(
                {
                    "event": "train_skipped_already_complete",
                    "run_name": run_config.name,
                    "output_dir": str(output_dir),
                }
            )
        )
        return output_dir

    resume = _has_epoch_checkpoint(output_dir) and not overwrite

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
                "use_qlora": run_config.use_qlora,
                "train_examples": len(splits.train),
                "resuming_from_checkpoint": resume,
            }
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(run_config.use_qlora)
    if run_config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=run_config.rank,
        lora_alpha=run_config.alpha,
        target_modules=run_config.target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def _format_example(example: dict) -> dict:
        prompt = config.format_prompt(example["text"])
        return build_training_example(tokenizer, prompt, splits.labels[example["label"]])

    tokenized_train = splits.train.map(_format_example, remove_columns=splits.train.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=run_config.epochs,
        per_device_train_batch_size=run_config.batch_size,
        gradient_accumulation_steps=run_config.gradient_accumulation_steps,
        learning_rate=run_config.learning_rate,
        save_strategy="epoch",  # checkpoint after each epoch (design doc: per-epoch is sufficient)
        logging_steps=10,
        bf16=_use_bf16(),
        fp16=not _use_bf16(),
        disable_tqdm=True,
        # ~500-token sequences x batch 4 on a 15GB T4 is at the edge of OOM
        # without this; non-reentrant avoids PEFT's frozen-input grad error.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )

    # Pads input_ids with pad_token and labels with -100, so padding never
    # contributes to the loss.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=_IGNORE_INDEX)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        callbacks=[_PrintProgressCallback(run_config.name)],
    )
    # torch>=2.6 defaults torch.load to weights_only=True, and transformers
    # 4.46.3 can't restore the numpy RNG state in epoch checkpoints under that
    # default, so resuming crashes. Allowlist just the numpy array primitives
    # that rng_state.pth contains.
    with torch.serialization.safe_globals(
        [np.ndarray, np.dtype, np.dtypes.UInt32DType, np._core.multiarray._reconstruct]
    ):
        train_result = trainer.train(resume_from_checkpoint=True if resume else None)

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
