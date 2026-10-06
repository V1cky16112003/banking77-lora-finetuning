"""Shared-prefix parallel questions (Pointwise Phase 1, docs/designs/pointwise-plan.md).

One state, k questions. Each question must see exactly what it would see in
a separate "state + this question" call, and nothing of its siblings. Three
equivalent ways to run it:

- packed: state then every question in ONE sequence, with a block attention
  mask (a question token attends to the state and to earlier tokens of its own
  question only) and position ids that restart at len(state) for each
  question. One forward pass for the whole request.
- branches: the state prefilled once into a KV cache, then all questions in one
  pass on top of it (same mask, minus the state rows). This is the serving
  path: a cached state is reused across requests.
- rows: the state prefilled once, then each question as its own causal row
  continuing a copy of the cache. No custom mask, so it works on any backbone
  (including recurrent/hybrid ones that can't honour a block mask), at the cost
  of one state-cache copy per row.

Why the context budget is "state + longest question": with restarted positions
every question sits at positions len(state) .. len(state)+len(question), so
position ids never grow with the number of questions.

All three return the model's features at each question's tokens: logits for a
*ForCausalLM model, last hidden states for a bare backbone (AutoModel), which
is what the Phase 2 pointer head reads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache, PreTrainedModel

from src.readout import repeat_cache

# Tokens per rows-form forward pass, counting the state each row carries.
# Bounds memory however many questions a request has (Kev uses the same rule).
ROW_PASS_TOKENS = 16384


@dataclass(frozen=True)
class Packed:
    """One request laid out as a single sequence: state, then each question."""

    ids: list[int]
    positions: list[int]  # questions restart at state_len
    segments: list[int]  # 0 = state, k = question k (1-based)
    state_len: int
    question_spans: list[tuple[int, int]]  # [start, end) of each question in ids


def pack(state_ids: list[int], questions: list[list[int]]) -> Packed:
    if not state_ids:
        raise ValueError("state must have at least one token")
    if not questions or any(not q for q in questions):
        raise ValueError("need at least one question, and no empty question")
    state_len = len(state_ids)
    ids, positions, segments = list(state_ids), list(range(state_len)), [0] * state_len
    spans = []
    for k, q in enumerate(questions, start=1):
        spans.append((len(ids), len(ids) + len(q)))
        ids += q
        positions += range(state_len, state_len + len(q))
        segments += [k] * len(q)
    return Packed(ids, positions, segments, state_len, spans)


def block_mask(segments: list[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive [1, 1, L, L] mask: token i may attend to j iff j <= i and j is
    state or in i's own question. Allowed = 0, blocked = finfo(dtype).min
    (not -inf, so a softmax over a row can never become NaN). Additive rather
    than boolean because eager attention adds the mask to the scores."""
    return block_mask_batch([segments], len(segments), dtype, device)


def block_mask_batch(
    rows: list[list[int]], length: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """block_mask for a right-padded batch: [B, 1, length, length]. Padding keys
    are blocked for every query; a padding query keeps only its own diagonal, so
    no row is ever fully blocked. Real tokens never see padding (it sits after
    them, and belongs to no segment)."""
    seg = torch.full((len(rows), length), -1, device=device)
    for b, row in enumerate(rows):
        seg[b, : len(row)] = torch.tensor(row, device=device)
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    visible = (seg[:, None, :] == 0) | (seg[:, None, :] == seg[:, :, None])
    allowed = causal[None] & visible & (seg != -1)[:, None, :]
    allowed |= torch.eye(length, dtype=torch.bool, device=device)[None]
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]


def _features(out) -> torch.Tensor:
    return out.logits if hasattr(out, "logits") else out.last_hidden_state


def _model_dtype(model: PreTrainedModel) -> torch.dtype:
    """The backbone's compute dtype. Read from the embeddings, not the first
    parameter: with LoRA the adapter weights are fp32 on an fp16 backbone."""
    return model.get_input_embeddings().weight.dtype


def packed_forward(model: PreTrainedModel, batch: list[Packed], pad_id: int) -> list[list[torch.Tensor]]:
    """One forward pass over a right-padded batch of packed requests, with
    gradients (training). -> per request, per question [len_q, F]."""
    device = model.device
    length = max(len(p.ids) for p in batch)
    ids = torch.full((len(batch), length), pad_id, dtype=torch.long, device=device)
    positions = torch.zeros((len(batch), length), dtype=torch.long, device=device)
    for b, p in enumerate(batch):
        ids[b, : len(p.ids)] = torch.tensor(p.ids, device=device)
        positions[b, : len(p.ids)] = torch.tensor(p.positions, device=device)
    out = model(
        input_ids=ids,
        position_ids=positions,
        attention_mask=block_mask_batch([p.segments for p in batch], length, _model_dtype(model), device),
    )
    feats = _features(out)
    return [[feats[b, start:end] for start, end in p.question_spans] for b, p in enumerate(batch)]


@torch.no_grad()
def packed_features(model: PreTrainedModel, packed: Packed) -> list[torch.Tensor]:
    """One forward pass over state + all questions. -> per question [len_q, F]."""
    return packed_forward(model, [packed], pad_id=0)[0]


@torch.no_grad()
def prefill_state(model: PreTrainedModel, state_ids: list[int]) -> DynamicCache:
    """KV cache of the state alone. Exact to reuse for any questions: the state
    never attends to what comes after it (causal), so its keys and values are
    the same with or without questions."""
    out = model(input_ids=torch.tensor([state_ids], device=model.device), use_cache=True)
    return out.past_key_values


@torch.no_grad()
def branch_features(
    model: PreTrainedModel, cache: DynamicCache, packed: Packed
) -> list[torch.Tensor]:
    """All questions in one pass on top of a cached state. The cache is cropped
    back to the state afterwards, so the same cache serves the next request."""
    device, start = model.device, packed.state_len
    if cache.get_seq_length() != start:
        raise ValueError(f"cache holds {cache.get_seq_length()} tokens, state has {start}")
    mask = block_mask(packed.segments, _model_dtype(model), device)[:, :, start:, :]
    try:
        out = model(
            input_ids=torch.tensor([packed.ids[start:]], device=device),
            position_ids=torch.tensor([packed.positions[start:]], device=device),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )
    finally:
        cache.crop(start)
    feats = _features(out)[0]
    return [feats[s - start : e - start] for s, e in packed.question_spans]


@torch.no_grad()
def row_features(
    model: PreTrainedModel,
    cache: DynamicCache,
    state_len: int,
    questions: list[list[int]],
    token_budget: int = ROW_PASS_TOKENS,
) -> list[torch.Tensor]:
    """Each question as its own causal row continuing the cached state, right-
    padded and run as many rows per pass as fit `token_budget`. The cache is
    copied per pass, never modified."""
    device = model.device
    rows_per_pass = max(1, token_budget // (state_len + max(len(q) for q in questions)))
    results: list[torch.Tensor] = []
    for first in range(0, len(questions), rows_per_pass):
        chunk = questions[first : first + rows_per_pass]
        width = max(len(q) for q in chunk)
        ids = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        real = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        for row, q in enumerate(chunk):
            ids[row, : len(q)] = torch.tensor(q, device=device)
            real[row, : len(q)] = 1
        # Padding sits after every real token of its row, so causal attention
        # already keeps it out of the real tokens; the mask covers the rest.
        out = model(
            input_ids=ids,
            position_ids=(state_len + torch.arange(width, device=device)).expand(len(chunk), -1),
            attention_mask=torch.cat(
                [torch.ones((len(chunk), state_len), dtype=torch.long, device=device), real], dim=1
            ),
            past_key_values=repeat_cache(cache, len(chunk)),
            use_cache=True,
        )
        feats = _features(out)
        results += [feats[row, : len(q)] for row, q in enumerate(chunk)]
    return results
