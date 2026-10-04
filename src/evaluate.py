"""ML orchestration for evaluation: constrained decoding, batched generation,
incremental prediction persistence. Imports evaluate_core.py (pure logic) plus
torch/transformers for the actual generation calls — this is the file CI does
NOT import (Architecture finding 1 / D6).

Batching + incremental JSONL persistence (outside-voice E2): the eval loop
runs ~12,800 generation calls total (baseline + 3 LoRA configs + final test),
each carrying the full label list (D7 reversal). Un-batched and without
incremental persistence, a Colab disconnect near the end of a pass would mean
restarting the whole pass from zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

from src.config import TaskConfig
from src.evaluate_core import load_resumable_jsonl, parse_label

_GENERATE_BATCH_SIZE = 16


def _build_prefix_allowed_tokens_fn(
    tokenizer: PreTrainedTokenizer, labels: list[str], prompt_len: int
):
    """Constrains generation to only ever produce tokens that extend a valid
    prefix of one of the configured labels, then EOS once a label is complete.
    This is authoritative (D3): when it successfully applies, the model CANNOT
    emit anything outside the label set.

    `prompt_len` is the padded prompt length: with left-padding every row's
    prompt ends at the same index, so everything after it is generated text.
    Label token ids come from encoding the bare label, which matches how
    train.py tokenizes the target (label appended directly after the prompt's
    trailing newline) — a leading space would tokenize differently."""
    label_token_sequences = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    eos_id = tokenizer.eos_token_id

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[prompt_len:].tolist()
        n = len(generated)
        allowed: set[int] = set()
        for seq in label_token_sequences:
            if seq[:n] != generated:
                continue
            if n < len(seq):
                allowed.add(seq[n])
            else:
                allowed.add(eos_id)
        return list(allowed) if allowed else [eos_id]

    return prefix_allowed_tokens_fn


def generate_labels_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: TaskConfig,
) -> list[str]:
    """Batched, greedy (D4), constrained-decoding generation for a batch of
    prompts. Falls back to unconstrained generation only if the constraint
    processor itself fails to build (D3's documented failure mode) —
    exact/fuzzy matching in evaluate_core.parse_label then resolves the
    unconstrained output, or marks it unresolved."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        prefix_fn = _build_prefix_allowed_tokens_fn(
            tokenizer, config.labels, prompt_len=inputs["input_ids"].shape[1]
        )
        logits_processor = LogitsProcessorList(
            [PrefixConstrainedLogitsProcessor(prefix_fn, num_beams=1)]
        )
    except Exception:
        logits_processor = None  # constraint-init failure (D3) — unconstrained fallback

    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,  # greedy, per D4
        # Qwen2.5-Instruct's generation_config ships repetition_penalty=1.1,
        # which would penalize label tokens simply for appearing in the prompt.
        repetition_penalty=1.0,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated_texts = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return generated_texts


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: list[dict],  # each: {"text": str, "label": str}
    config: TaskConfig,
    predictions_path: Path,
) -> list[tuple[str, str | None]]:
    """Runs generation over `examples` in batches, writing each prediction to
    `predictions_path` (JSONL, one line per example) as it's produced —
    a disconnect resumes from the last written line instead of restarting
    the whole pass (E2)."""
    results: list[tuple[str, str | None]] = [
        (record["true_label"], record["predicted_label"])
        for record in load_resumable_jsonl(predictions_path)
    ]
    remaining = examples[len(results) :]

    with predictions_path.open("a", encoding="utf-8") as f:
        for batch_start in range(0, len(remaining), _GENERATE_BATCH_SIZE):
            batch = remaining[batch_start : batch_start + _GENERATE_BATCH_SIZE]
            prompts = [config.format_prompt(ex["text"]) for ex in batch]
            generated_texts = generate_labels_batch(model, tokenizer, prompts, config)

            for example, generated_text in zip(batch, generated_texts):
                predicted_label = parse_label(generated_text, config.labels)
                f.write(
                    json.dumps(
                        {
                            "true_label": example["label"],
                            "predicted_label": predicted_label,
                            "raw_generated_text": generated_text,
                        }
                    )
                    + "\n"
                )
                f.flush()
                results.append((example["label"], predicted_label))

    return results
