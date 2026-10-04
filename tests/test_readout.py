"""The fast label scorer must give the same scores as naive scoring.

Uses a tiny randomly initialised Qwen2 model plus the cached Qwen tokenizer, so
no weights download is needed. Skipped where torch is absent (CI installs
requirements-core.txt only) or the tokenizer isn't cached.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.config import load_task_config  # noqa: E402
from src.readout import LabelScorer, repeat_cache, split_prompt, tokenize_labels  # noqa: E402

# Different lengths on purpose: exercises the suffix-padding and position-id path.
MESSAGES = [
    "where is my new card?",
    "I was charged twice for the same coffee this morning and the app still shows pending",
    "top up failed",
]


@pytest.fixture(scope="module")
def config():
    return load_task_config()


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not cached locally")


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    model_config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return transformers.Qwen2ForCausalLM(model_config).eval()


def _naive_scores(model, tokenizer, prompt, labels):
    """Reference: one full forward pass per label over the whole prompt."""
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


def test_batched_scores_match_naive(tiny_model, tokenizer, config):
    fast = LabelScorer(tiny_model, tokenizer, config).score(MESSAGES)
    for message, row in zip(MESSAGES, fast):
        slow = _naive_scores(tiny_model, tokenizer, config.format_prompt(message), config.labels)
        assert row == pytest.approx(slow, abs=1e-4)


def test_batch_of_one_matches_batch_of_many(tiny_model, tokenizer, config):
    scorer = LabelScorer(tiny_model, tokenizer, config)
    together = scorer.score(MESSAGES)
    alone = [scorer.score([m])[0] for m in MESSAGES]
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-4)


def test_split_prompt_reassembles_and_cuts_at_newline(config):
    prefix, suffix = split_prompt(config, "hello")
    assert prefix + suffix == config.format_prompt("hello")
    assert prefix.endswith("\n")
    assert suffix.startswith("Customer message: hello")


def test_split_keeps_tokenisation_on_every_banking77_message(tokenizer, config):
    datasets = pytest.importorskip("datasets")
    try:
        splits = datasets.load_dataset(config.dataset_hf_name, revision=config.dataset_revision)
    except Exception:
        pytest.skip("Banking77 not cached locally")
    prefix_ids = tokenizer.encode(split_prompt(config, "")[0], add_special_tokens=False)
    for split in ("train", "test"):
        for text in splits[split]["text"]:
            prefix, suffix = split_prompt(config, text)
            full_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
            assert prefix_ids + tokenizer.encode(suffix, add_special_tokens=False) == full_ids, text


def test_label_tokens_end_in_eos_and_mask_padding(tokenizer, config):
    tokens = tokenize_labels(tokenizer, config.labels)
    for row in range(len(config.labels)):
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
