"""Logit read-out: decide among the configured labels without generate().

MiniJev Phase 0 (docs/designs/minijev-phase0-plan.md). For each label y we
compute the exact sequence log-likelihood

    s(y) = sum_t log p(y_t | prompt, y_<t)     (label tokens + EOS)

and renormalise over the label set, giving a full probability distribution
instead of one greedy string. The prompt is prefilled once and its KV cache is
shared by all labels, scored together in one batched forward pass. That is the
"ingest the state once, evaluate every option in parallel" idea behind Jev,
at the scale of a single question.

Label tokenisation matches train.build_training_example exactly: bare label
appended after the prompt's trailing newline, then EOS. The EOS term is what
stops a label that is a prefix of another from winning by being shorter.

Pure metric logic lives in evaluate_core.py; this file holds only the torch
side, mirroring the evaluate.py / evaluate_core.py split.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from src.config import TaskConfig
from src.evaluate_core import softmax_scores


@dataclass(frozen=True)
class LabelTokens:
    """Right-padded label token ids (each ending in EOS), built once per run."""

    ids: torch.Tensor  # (n_labels, max_len), long
    mask: torch.Tensor  # (n_labels, max_len), bool — True on real tokens


def tokenize_labels(tokenizer: PreTrainedTokenizer, labels: list[str]) -> LabelTokens:
    sequences = [
        tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
        for label in labels
    ]
    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = True
    return LabelTokens(ids=ids, mask=mask)


def repeat_cache(cache, n: int):
    """Copy a batch-1 KV cache across `n` rows, whichever format the model returns.

    The pinned transformers (4.46, requirements.txt) returns the legacy format,
    a tuple of (key, value) tensors per layer, while newer releases return a
    Cache object. A legacy tuple is expanded and wrapped in a DynamicCache,
    because 4.46 only accepts a tuple when use_cache=True. expand() makes a
    zero-copy view; the cache concatenates new keys onto it rather than writing
    in place, so the prompt cache is not changed.
    """
    if isinstance(cache, tuple):
        return DynamicCache.from_legacy_cache(
            tuple(tuple(t.expand(n, *t.shape[1:]) for t in layer) for layer in cache)
        )
    # A Cache object is mutated in place by the next forward pass, so copy it first.
    cache = copy.deepcopy(cache)
    cache.batch_repeat_interleave(n)
    return cache


@torch.no_grad()
def score_labels(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    label_tokens: LabelTokens,
) -> list[float]:
    """Sequence log-likelihood of every label given `prompt`.

    1. Prefill the prompt once. Its last-position logits score each label's
       first token, and its KV cache is kept.
    2. Repeat the cache across labels and feed label tokens [:-1] in one batch;
       position t's logits score label token t+1.
    Right padding is safe: causal attention means pad positions never affect
    the real ones, and their log-probs are masked out of the sum.
    """
    device = model.device
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], device=device
    )
    ids = label_tokens.ids.to(device)
    mask = label_tokens.mask.to(device)
    n_labels, max_len = ids.shape

    prefill = model(input_ids=prompt_ids, use_cache=True)
    first_logprobs = torch.log_softmax(prefill.logits[0, -1].float(), dim=-1)
    token_logprobs = [first_logprobs[ids[:, 0]]]  # (n_labels,)

    if max_len > 1:
        cache = repeat_cache(prefill.past_key_values, n_labels)
        attention_mask = torch.ones(
            (n_labels, prompt_ids.shape[1] + max_len - 1), dtype=torch.long, device=device
        )
        out = model(
            input_ids=ids[:, :-1],
            past_key_values=cache,
            attention_mask=attention_mask,
            use_cache=False,
        )
        rest = torch.log_softmax(out.logits.float(), dim=-1)  # (n_labels, max_len-1, vocab)
        token_logprobs.extend(
            rest[:, t].gather(1, ids[:, t + 1 : t + 2]).squeeze(1) for t in range(max_len - 1)
        )

    stacked = torch.stack(token_logprobs, dim=1)  # (n_labels, max_len)
    return (stacked * mask).sum(dim=1).tolist()


def readout_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: list[dict],  # each: {"text": str, "label": str}
    config: TaskConfig,
    predictions_path: Path,
) -> list[dict]:
    """Read-out over `examples`, appending one JSONL record per example so a
    Kaggle disconnect resumes where it stopped (same contract as
    evaluate.evaluate_dataset). Records carry the full distribution so
    calibration metrics can be recomputed offline."""
    records: list[dict] = []
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

    label_tokens = tokenize_labels(tokenizer, config.labels)
    with predictions_path.open("a", encoding="utf-8") as f:
        for example in examples[len(records) :]:
            scores = score_labels(model, tokenizer, config.format_prompt(example["text"]), label_tokens)
            probs = softmax_scores(scores)
            best = max(range(len(probs)), key=probs.__getitem__)
            record = {
                "true_label": example["label"],
                "predicted_label": config.labels[best],
                "confidence": probs[best],
                "probs": probs,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)
    return records
