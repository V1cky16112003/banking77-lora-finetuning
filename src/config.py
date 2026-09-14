"""Shared task-config loader (finding 5A). No ML dependencies — safe to import
from evaluate_core.py, evaluate.py, train.py, and serve.py alike, and safe for
CI (D6/Architecture finding 1) since it only depends on pyyaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_KEYS = ("task_name", "dataset", "labels", "prompt_template", "few_shot", "decoding")


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    dataset_hf_name: str
    dataset_revision: str | None
    labels: list[str]
    prompt_template: str
    few_shot_total_examples: int
    do_sample: bool
    max_new_tokens: int

    def format_prompt(self, message: str) -> str:
        label_list = ", ".join(self.labels)
        return self.prompt_template.format(label_list=label_list, message=message)


def load_task_config(path: str | Path = "configs/task.yaml") -> TaskConfig:
    """Load and validate the task config. Raises a clear error on a missing
    required key rather than returning silent Nones (per the design doc's
    error-handling discipline)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"Task config at {path} is missing required key(s): {missing}. "
            f"Expected all of: {_REQUIRED_KEYS}"
        )

    dataset = raw["dataset"]
    if "hf_name" not in dataset:
        raise ValueError(f"Task config at {path}: dataset section missing 'hf_name'.")

    return TaskConfig(
        task_name=raw["task_name"],
        dataset_hf_name=dataset["hf_name"],
        dataset_revision=dataset.get("revision"),
        labels=raw["labels"],
        prompt_template=raw["prompt_template"],
        few_shot_total_examples=raw["few_shot"]["total_examples"],
        do_sample=raw["decoding"]["do_sample"],
        max_new_tokens=raw["decoding"]["max_new_tokens"],
    )
