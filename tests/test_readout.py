"""KV-cache label scorer must equal naive per-label full-sequence scoring.

Uses a tiny randomly initialised Qwen2 model plus the cached Qwen tokenizer, so
no weights download is needed. Skipped where torch is absent (CI installs
requirements-core.txt only) or the tokenizer isn't cached.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.readout import repeat_cache, score_labels, tokenize_labels  # noqa: E402

LABELS = ["card_arrival", "card_delivery_estimate", "card_lost", "age_limit"]
PROMPT = "Customer message: where is my new card?\nIntent:\n"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not cached locally")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    return model, tokenizer


def _naive_scores(model, tokenizer, prompt, labels):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores = []
    for label in labels:
        target = tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = torch.tensor([prompt_ids + target])
        with torch.no_grad():
            logprobs = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        start = len(prompt_ids)
        scores.append(sum(logprobs[start - 1 + t, tok].item() for t, tok in enumerate(target)))
    return scores


def test_kv_cache_scoring_matches_naive(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    fast = score_labels(model, tokenizer, PROMPT, tokenize_labels(tokenizer, LABELS))
    slow = _naive_scores(model, tokenizer, PROMPT, LABELS)
    assert fast == pytest.approx(slow, abs=1e-4)


def test_label_tokens_end_in_eos_and_mask_padding(tiny_model_and_tokenizer):
    _, tokenizer = tiny_model_and_tokenizer
    tokens = tokenize_labels(tokenizer, LABELS)
    for row in range(len(LABELS)):
        last = int(tokens.mask[row].sum()) - 1
        assert tokens.ids[row, last].item() == tokenizer.eos_token_id


@pytest.mark.skipif(
    not hasattr(transformers.DynamicCache, "from_legacy_cache"),
    reason="transformers 5.x removed the legacy tuple cache format",
)
def test_repeat_cache_handles_legacy_tuple_format():
    # transformers 4.46 (the pinned version on Kaggle) returns this format.
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    repeated = repeat_cache(((key, value), (key, value)), 3)
    assert isinstance(repeated, transformers.DynamicCache)
    assert len(repeated) == 2
    for layer_key, layer_value in repeated.to_legacy_cache():
        assert layer_key.shape == (3, 2, 5, 8)
        assert torch.equal(layer_key[2], key[0])
        assert torch.equal(layer_value[1], value[0])
