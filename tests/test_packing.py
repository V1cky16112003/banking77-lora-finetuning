"""Packed, branch and row forms must give the same features as running each
question separately on "state + question" (MiniJev Phase 1 gate).

Tiny randomly initialised Qwen3 / Qwen2 models in fp32 with random token ids,
so nothing is downloaded. Skipped where torch is absent (CI installs
requirements-core.txt only).
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.minijev.packing import (  # noqa: E402
    block_mask,
    branch_features,
    pack,
    packed_features,
    prefill_state,
    row_features,
)

VOCAB = 101
ATOL = 1e-4
STATE = [5, 17, 42, 8, 99, 3, 61]
# Different lengths on purpose: exercises restarted positions and row padding.
QUESTIONS = [[11, 12, 13], [21], [31, 32, 33, 34, 35], [41, 42]]


def _tiny(arch: str, attn: str, head: bool = True):
    torch.manual_seed(0)
    shape = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation=attn,
    )
    if arch == "qwen3":
        config = transformers.Qwen3Config(head_dim=16, **shape)
        cls = transformers.Qwen3ForCausalLM if head else transformers.Qwen3Model
    else:
        config = transformers.Qwen2Config(**shape)
        cls = transformers.Qwen2ForCausalLM if head else transformers.Qwen2Model
    return cls(config).eval()


MODELS = [("qwen3", "eager"), ("qwen3", "sdpa"), ("qwen2", "sdpa")]


@pytest.fixture(scope="module", params=MODELS, ids=["-".join(m) for m in MODELS])
def model(request):
    return _tiny(*request.param)


def _separate(model, state, questions):
    """Reference: one plain causal forward pass per "state + question"."""
    out = []
    for q in questions:
        with torch.no_grad():
            feats = model(input_ids=torch.tensor([state + q])).logits[0]
        out.append(feats[len(state) :])
    return out


def _assert_close(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, atol=ATOL, rtol=0)


def test_pack_layout():
    p = pack([1, 2, 3], [[4, 5], [6]])
    assert p.ids == [1, 2, 3, 4, 5, 6]
    assert p.positions == [0, 1, 2, 3, 4, 3]
    assert p.segments == [0, 0, 0, 1, 1, 2]
    assert p.state_len == 3
    assert p.question_spans == [(3, 5), (5, 6)]


@pytest.mark.parametrize("state, questions", [([], [[1]]), ([1], []), ([1], [[2], []])])
def test_pack_rejects_empty_parts(state, questions):
    with pytest.raises(ValueError):
        pack(state, questions)


def test_block_mask_pattern():
    mask = block_mask([0, 0, 1, 1, 2], torch.float32, torch.device("cpu"))[0, 0]
    allowed = (mask == 0).int().tolist()
    assert allowed == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],  # question 1 sees the state and itself
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1],  # question 2 sees the state, never question 1
    ]
    assert mask.min() == torch.finfo(torch.float32).min  # finite, not -inf


def test_packed_matches_separate(model):
    _assert_close(packed_features(model, pack(STATE, QUESTIONS)), _separate(model, STATE, QUESTIONS))


def test_branches_on_cached_state_match_separate_and_leave_cache_reusable(model):
    cache = prefill_state(model, STATE)
    packed = pack(STATE, QUESTIONS)
    first = branch_features(model, cache, packed)
    assert cache.get_seq_length() == len(STATE)
    _assert_close(first, _separate(model, STATE, QUESTIONS))
    # The same cache serves a second request with different questions.
    other = [[7, 7], [8, 9, 10]]
    _assert_close(branch_features(model, cache, pack(STATE, other)), _separate(model, STATE, other))


def test_branches_reject_cache_of_another_state(model):
    cache = prefill_state(model, STATE[:-1])
    with pytest.raises(ValueError):
        branch_features(model, cache, pack(STATE, QUESTIONS))


@pytest.mark.parametrize("token_budget", [16384, len(STATE) + 5])  # one pass; one row per pass
def test_rows_match_separate(model, token_budget):
    cache = prefill_state(model, STATE)
    rows = row_features(model, cache, len(STATE), QUESTIONS, token_budget=token_budget)
    _assert_close(rows, _separate(model, STATE, QUESTIONS))
    assert cache.get_seq_length() == len(STATE)  # rows work on copies


def test_a_question_cannot_see_its_siblings(model):
    base = packed_features(model, pack(STATE, QUESTIONS))
    changed = packed_features(model, pack(STATE, [QUESTIONS[0], [77, 78, 79, 80], *QUESTIONS[2:]]))
    for k in (0, 2, 3):
        torch.testing.assert_close(base[k], changed[k], atol=ATOL, rtol=0)


def test_bare_backbone_returns_hidden_states():
    backbone = _tiny("qwen3", "sdpa", head=False)
    feats = packed_features(backbone, pack(STATE, QUESTIONS))
    assert [f.shape for f in feats] == [(len(q), 64) for q in QUESTIONS]
    with torch.no_grad():
        alone = backbone(input_ids=torch.tensor([STATE + QUESTIONS[2]])).last_hidden_state[0, len(STATE) :]
    torch.testing.assert_close(feats[2], alone, atol=ATOL, rtol=0)
