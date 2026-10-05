"""Decision model: Qwen backbone + packed questions + pointer head (MiniJev Phase 2).

    q   = W_q h[<decide>]                 what this question is asking
    k_j = W_k h[</opt> of option j]       what option j says, in context
    z_j = k_j . q / sqrt(d_p)
    p   = softmax(z / T)                  over this question's options only

Each option is represented by the backbone's own hidden state at the end of
its text, so nothing has to learn what "option 7" means: unseen label sets work
by construction, and 255 is only an API cap (architecture §3.3). The answer is
always one of the declared options, because the softmax only runs over them.

All questions of a request run in one packed pass (src/minijev/packing.py), so
they share the state's computation and can't see each other.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from src.minijev.encoding import EncodedRequest, encode_request
from src.minijev.packing import pack, packed_features
from src.minijev.schema import SystemOneRequest, to_answers

HEAD_DIM = 256


class PointerHead(nn.Module):
    """Scores a question's <decide> state against each of its option states.
    Kept in fp32 even when the backbone is fp16."""

    def __init__(self, hidden_size: int, head_dim: int = HEAD_DIM):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)
        self.scale = 1 / math.sqrt(head_dim)
        # Fitted on dev data after training (Phase 4) and applied at inference
        # only, so training always sees T = 1. Dividing by T never changes the argmax.
        self.temperature = 1.0

    def forward(self, h_decide: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
        """[d], [K, d] -> logits [K]."""
        z = (self.k(h_options) @ self.q(h_decide)) * self.scale
        return z if self.training else z / self.temperature


class DecisionModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, tokenizer, head_dim: int = HEAD_DIM):
        super().__init__()
        self.backbone = backbone  # bare backbone (AutoModel): no vocabulary head, we never generate
        self.tokenizer = tokenizer
        self.head = PointerHead(backbone.config.hidden_size, head_dim).to(backbone.device)

    @classmethod
    def from_pretrained(cls, name: str, device: str, dtype: torch.dtype) -> DecisionModel:
        tokenizer = AutoTokenizer.from_pretrained(name)
        backbone = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
        return cls(backbone.to(device).eval(), tokenizer).eval()

    def encode(self, request: SystemOneRequest) -> EncodedRequest:
        return encode_request(self.tokenizer, request)

    def question_logits(self, encoded: EncodedRequest) -> list[torch.Tensor]:
        """Option logits per question, from one packed forward pass.
        (packed_features runs without gradients; Phase 4 training needs a grad path.)"""
        features = packed_features(
            self.backbone, pack(encoded.state_ids, [q.ids for q in encoded.questions])
        )
        logits = []
        for question, hidden in zip(encoded.questions, features):
            hidden = hidden.float()
            logits.append(self.head(hidden[question.decide_offset], hidden[question.option_offsets]))
        return logits

    @torch.no_grad()
    def probs(self, encoded: EncodedRequest) -> list[list[float]]:
        return [z.softmax(-1).tolist() for z in self.question_logits(encoded)]

    def decide(self, request: SystemOneRequest | dict) -> dict:
        """A /v1/systemone request -> its response body."""
        if isinstance(request, dict):
            request = SystemOneRequest.model_validate(request)
        encoded = self.encode(request)
        return {
            "model": request.model,
            "answers": to_answers(request, self.probs(encoded)),
            "usage": {
                "input_tokens": len(encoded.state_ids) + sum(len(q.ids) for q in encoded.questions),
                "output_tokens": 0,  # nothing is generated
            },
        }
