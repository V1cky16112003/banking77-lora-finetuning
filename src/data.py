"""Dataset loading and the frozen train/val/held-out-test split.

Requires the `datasets` library (not imported by evaluate_core.py — this file
is part of the ML-dependent side, run in Colab, not CI).

Split discipline (design doc premise 3): the held-out test set is Banking77's
own provided test split, used identically for baseline and fine-tuned
evaluation. It is built once, frozen, and never touched during training or
hyperparameter selection. The validation split (for comparing the 2-3 LoRA
configs) is carved out of the TRAINING data only.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

from src.config import TaskConfig

# Validation split size as a fraction of the training data (design doc
# Technical Notes: "10-15% of the training data").
_VAL_FRACTION = 0.12
_SPLIT_SEED = 42  # fixed seed for the train/val split (design doc: "fix a random seed")


@dataclass(frozen=True)
class BankingSplits:
    train: Dataset
    val: Dataset
    test: Dataset  # frozen held-out test set — never touched until final eval
    labels: list[str]
    dataset_revision: str | None


def load_banking77_splits(config: TaskConfig) -> BankingSplits:
    """Load Banking77 and carve the frozen train/val/test split.

    Populates config.labels the first time this runs, if it was left empty —
    callers should persist the returned `labels` (and `dataset_revision`)
    back into configs/task.yaml so the label list and dataset pin (D11) are
    never hand-transcribed, per the design doc's Approach A verification note.
    """
    dataset: DatasetDict = load_dataset(config.dataset_hf_name, revision=config.dataset_revision)

    label_names = dataset["train"].features["label"].names
    if config.labels and config.labels != label_names:
        raise ValueError(
            "configs/task.yaml's `labels` list does not match the loaded dataset's "
            "label names. Re-verify and update the config rather than silently "
            "proceeding with a mismatched label set."
        )

    # Stratified train/val split, carved from the training data only.
    split = dataset["train"].train_test_split(
        test_size=_VAL_FRACTION,
        seed=_SPLIT_SEED,
        stratify_by_column="label",
    )
    train_split = split["train"]
    val_split = split["test"]

    return BankingSplits(
        train=train_split,
        val=val_split,
        test=dataset["test"],  # Banking77's own test split — frozen, untouched until final eval
        labels=label_names,
        dataset_revision=getattr(dataset["train"], "info", None)
        and dataset["train"].info.version
        and str(dataset["train"].info.version)
        or config.dataset_revision,
    )
