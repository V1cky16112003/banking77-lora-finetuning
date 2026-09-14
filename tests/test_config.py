import pytest
import yaml

from src.config import TaskConfig, load_task_config

VALID_CONFIG = {
    "task_name": "test_task",
    "dataset": {"hf_name": "PolyAI/banking77", "revision": "abc123"},
    "labels": ["card_lost", "balance_not_updated"],
    "prompt_template": "Labels: {label_list}\nMessage: {message}\nIntent:",
    "few_shot": {"total_examples": 12},
    "decoding": {"do_sample": False, "max_new_tokens": 24},
}


def _write_config(tmp_path, data):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_valid_config(tmp_path):
    path = _write_config(tmp_path, VALID_CONFIG)
    config = load_task_config(path)

    assert isinstance(config, TaskConfig)
    assert config.task_name == "test_task"
    assert config.dataset_hf_name == "PolyAI/banking77"
    assert config.dataset_revision == "abc123"
    assert config.labels == ["card_lost", "balance_not_updated"]
    assert config.few_shot_total_examples == 12
    assert config.do_sample is False
    assert config.max_new_tokens == 24


@pytest.mark.parametrize("missing_key", ["task_name", "dataset", "labels", "prompt_template", "few_shot", "decoding"])
def test_missing_required_key_raises_clear_error(tmp_path, missing_key):
    data = {k: v for k, v in VALID_CONFIG.items() if k != missing_key}
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError, match=missing_key):
        load_task_config(path)


def test_missing_dataset_hf_name_raises(tmp_path):
    data = dict(VALID_CONFIG)
    data["dataset"] = {"revision": "abc123"}
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError, match="hf_name"):
        load_task_config(path)


def test_format_prompt_substitutes_labels_and_message(tmp_path):
    path = _write_config(tmp_path, VALID_CONFIG)
    config = load_task_config(path)

    prompt = config.format_prompt("my card is lost")

    assert "card_lost" in prompt
    assert "balance_not_updated" in prompt
    assert "my card is lost" in prompt
