"""Logit read-out: decide among the configured labels without generate().

Pointwise Phase 0 (docs/designs/pointwise-phase0-plan.md). For each label y we
compute the exact sequence log-likelihood

    s(y) = sum_t log p(y_t | prompt, y_<t)     (label tokens + EOS)

and renormalise over the label set, giving a full probability distribution
instead of one greedy string.

Phase 0b (docs/designs/pointwise-phase0b-plan.md) makes it fast without changing
the scores:
- The prompt's static prefix (instructions + the 77-label list) is prefilled
  once per run and its KV cache reused for every message. This is Jev's
  "ingest the state once" idea.
- Messages are scored in batches, in two stages: the message suffix once per
  message, then all labels for all messages in one forward pass.
- Log-probs are logit[target] - logsumexp(logits) over row chunks, never a
  full-vocabulary fp32 softmax tensor.

Label tokenisation matches train.build_training_example exactly: bare label
appended after the prompt's trailing newline, then EOS. The EOS term is what
stops a label that is a prefix of another from winning by being shorter.

Pure metric logic lives in evaluate_core.py; this file holds only the torch
side, mirroring the evaluate.py / evaluate_core.py split.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from src.config import TaskConfig
from src.evaluate_core import load_resumable_jsonl, softmax_scores

_MESSAGE_SENTINEL = "\x00MESSAGE\x00"
# Rows per fp32 logsumexp chunk: 32 rows x ~12 positions x 151,936 vocab ~ 230 MB.
_LOGSUMEXP_ROW_CHUNK = 32


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
    ids = torch.full((len(sequences), max_len), _pad_id(tokenizer), dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = True
    return LabelTokens(ids=ids, mask=mask)


def split_prompt(config: TaskConfig, message: str) -> tuple[str, str]:
    """Split the formatted prompt into (static prefix, per-message suffix).

    The cut is at the last newline before the message, so the prefix holds the
    instructions and label list and the suffix starts "Customer message: ".
    Cutting at a newline keeps BPE from merging tokens across the boundary;
    LabelScorer still checks that per message.
    """
    template = config.format_prompt(_MESSAGE_SENTINEL)
    cut = template.rindex("\n", 0, template.index(_MESSAGE_SENTINEL)) + 1
    full = config.format_prompt(message)
    return full[:cut], full[cut:]


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


class LabelScorer:
    """Scores every configured label for a batch of messages.

    Built once per run: tokenises the labels and prefills the static prompt
    prefix. score() then costs one forward pass over the message suffixes and
    one over (messages x labels) label tokens.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: TaskConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.labels = tokenize_labels(tokenizer, config.labels)
        self.pad_id = _pad_id(tokenizer)
        prefix, _ = split_prompt(config, "")
        self.prefix_ids = self._encode(prefix)
        with torch.no_grad():
            self.prefix_cache = model(
                input_ids=torch.tensor([self.prefix_ids], device=model.device), use_cache=True
            ).past_key_values

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _suffix_ids(self, message: str) -> list[int]:
        prefix, suffix = split_prompt(self.config, message)
        suffix_ids = self._encode(suffix)
        if self.prefix_ids + suffix_ids != self._encode(prefix + suffix):
            # Never score a different token sequence than training saw.
            raise ValueError(f"Prompt prefix/suffix split changes tokenisation for message {message!r}")
        return suffix_ids

    @torch.no_grad()
    def score(self, messages: list[str], timings: dict[str, float] | None = None) -> list[list[float]]:
        """Sequence log-likelihood of every label, per message: (len(messages), n_labels).

        Pass a dict as `timings` to accumulate per-stage seconds into it
        (CUDA-synchronised, so slower; for profiling only)."""
        device = self.model.device
        mark = _StageTimer(timings, device)
        suffixes = [self._suffix_ids(m) for m in messages]
        n_msgs, n_labels = len(suffixes), self.labels.ids.shape[0]
        prefix_len = len(self.prefix_ids)
        lengths = torch.tensor([len(s) for s in suffixes], device=device)
        suffix_len = int(lengths.max())

        # Stage A: message suffixes, right-padded, on top of the shared prefix.
        suffix_ids = torch.full((n_msgs, suffix_len), self.pad_id, dtype=torch.long, device=device)
        for row, ids in enumerate(suffixes):
            suffix_ids[row, : len(ids)] = torch.tensor(ids, device=device)
        suffix_mask = torch.arange(suffix_len, device=device)[None, :] < lengths[:, None]
        mask_a = torch.cat(
            [torch.ones((n_msgs, prefix_len), dtype=torch.long, device=device), suffix_mask.long()], dim=1
        )
        out_a = self.model(
            input_ids=suffix_ids,
            past_key_values=repeat_cache(self.prefix_cache, n_msgs),
            attention_mask=mask_a,
            position_ids=(prefix_len + torch.arange(suffix_len, device=device)).expand(n_msgs, -1),
            use_cache=True,
        )
        mark("suffix_pass")
        # Each message's last real position scores the first token of every label.
        last_logits = out_a.logits[torch.arange(n_msgs, device=device), lengths - 1].float()
        label_ids = self.labels.ids.to(device)
        first = last_logits.log_softmax(dim=-1)[:, label_ids[:, 0]]  # (n_msgs, n_labels)

        # Stage B: all labels for all messages. Row r = message r // n_labels, label r % n_labels.
        cache = out_a.past_key_values
        cache.batch_repeat_interleave(n_labels)
        mark("kv_repeat")
        label_len = label_ids.shape[1]
        mask_b = torch.cat(
            [
                mask_a.repeat_interleave(n_labels, dim=0),
                torch.ones((n_msgs * n_labels, label_len - 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        # Explicit positions skip the suffix padding, so each row matches an unpadded prompt.
        positions = (prefix_len + lengths).repeat_interleave(n_labels)[:, None] + torch.arange(
            label_len - 1, device=device
        )
        out_b = self.model(
            input_ids=label_ids[:, :-1].repeat(n_msgs, 1),
            past_key_values=cache,
            attention_mask=mask_b,
            position_ids=positions,
            use_cache=False,
        )
        mark("label_pass")
        rest = _token_logprobs(out_b.logits, label_ids[:, 1:].repeat(n_msgs, 1))
        mark("logsumexp")

        token_logprobs = torch.cat([first.reshape(-1, 1), rest], dim=1)
        mask = self.labels.mask.to(device).repeat(n_msgs, 1)
        return (token_logprobs * mask).sum(dim=1).view(n_msgs, n_labels).tolist()


class _StageTimer:
    """Adds the seconds since the previous call to timings[stage]. A no-op when
    timings is None, so the normal path never synchronises the GPU."""

    def __init__(self, timings: dict[str, float] | None, device: torch.device):
        self.timings = timings
        self.cuda = device.type == "cuda"
        self.last = self._now() if timings is not None else 0.0

    def _now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def __call__(self, stage: str) -> None:
        if self.timings is None:
            return
        now = self._now()
        self.timings[stage] = self.timings.get(stage, 0.0) + now - self.last
        self.last = now


def _token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) at every (row, position), computed in fp32 row chunks so
    the full (rows, positions, vocab) fp32 tensor is never materialised."""
    out = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[0], _LOGSUMEXP_ROW_CHUNK):
        chunk = logits[start : start + _LOGSUMEXP_ROW_CHUNK].float()
        picked = chunk.gather(-1, targets[start : start + _LOGSUMEXP_ROW_CHUNK, :, None]).squeeze(-1)
        out[start : start + _LOGSUMEXP_ROW_CHUNK] = picked - chunk.logsumexp(dim=-1)
    return out


def _pad_id(tokenizer: PreTrainedTokenizer) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def readout_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: list[dict],  # each: {"text": str, "label": str}
    config: TaskConfig,
    predictions_path: Path,
    batch_size: int = 4,
) -> list[dict]:
    """Read-out over `examples`, appending one JSONL record per example so a
    Kaggle disconnect resumes where it stopped (same contract as
    evaluate.evaluate_dataset). Records carry the full distribution so
    calibration metrics can be recomputed offline."""
    records = load_resumable_jsonl(predictions_path)
    scorer = LabelScorer(model, tokenizer, config)
    remaining = examples[len(records) :]
    with predictions_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            for example, scores in zip(batch, scorer.score([ex["text"] for ex in batch])):
                probs = softmax_scores(scores)
                best = max(range(len(probs)), key=probs.__getitem__)
                record = {
                    "true_label": example["label"],
                    "predicted_label": config.labels[best],
                    "confidence": probs[best],
                    "probs": probs,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
            f.flush()
    return records
