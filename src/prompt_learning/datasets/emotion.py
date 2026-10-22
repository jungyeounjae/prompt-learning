"""Balanced, reproducible emotion samples with isolated content identities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from prompt_learning.config import ExperimentConfig, SplitConfig
from prompt_learning.models import DatasetSplits

from .base import content_identity, find_column, require_sample_count, split_ratios


LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


def split_emotion(
    frame: pd.DataFrame, *, seed: int, samples_per_label: int = 20,
    ratios: SplitConfig = SplitConfig(),
) -> DatasetSplits:
    """Sample equally per label and stratify (default 60/20/20) without mutation.

    If a label has fewer rows than requested, cap all six labels to that count.
    At least five unique examples per label are required. Content hashes replace
    source IDs so repeated IDs or repeated text cannot leak between subsets.
    """
    require_sample_count(samples_per_label, "samples_per_label")
    train_ratio, validation_ratio, test_ratio = split_ratios(ratios)
    text_column = find_column(frame, ("text",))
    label_column = find_column(frame, ("expected_label", "label"))
    normalized = frame[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "expected_label"}
    ).copy()
    valid_text = normalized.text.map(lambda value: isinstance(value, str) and bool(value.strip()))
    normalized = normalized.loc[valid_text].copy()
    normalized["text"] = normalized.text.str.strip()
    mapping = {**dict(enumerate(LABELS)), **dict(zip(LABELS, LABELS))}
    labels = normalized.expected_label.map(mapping)
    if labels.isna().any():
        unknown = normalized.loc[labels.isna(), "expected_label"].tolist()
        raise ValueError(f"Unknown emotion label: {unknown}")
    normalized["expected_label"] = labels
    normalized["example_id"] = normalized.text.map(content_identity)
    if normalized.groupby("example_id").expected_label.nunique().gt(1).any():
        raise ValueError("Duplicate emotion text has conflicting labels")
    normalized = normalized.drop_duplicates("example_id")
    counts = normalized.expected_label.value_counts()
    for label in LABELS:
        if counts.get(label, 0) < 5:
            raise ValueError(f"Emotion label {label} needs at least 5 unique rows")
    count = min(samples_per_label, int(counts.min()))
    sampled = pd.concat([
        normalized.loc[normalized.expected_label.eq(label)].sample(n=count, random_state=seed)
        for label in LABELS
    ])[["example_id", "text", "expected_label"]]
    empty = sampled.iloc[:0].copy()
    if train_ratio == 0:
        train, remainder = empty, sampled
    elif validation_ratio + test_ratio == 0:
        train, remainder = sampled, empty
    else:
        train, remainder = train_test_split(
            sampled, train_size=train_ratio, stratify=sampled.expected_label, random_state=seed
        )
    if validation_ratio == 0:
        validation, test = empty, remainder
    elif test_ratio == 0:
        validation, test = remainder, empty
    else:
        validation, test = train_test_split(
            remainder, train_size=validation_ratio / (validation_ratio + test_ratio),
            stratify=remainder.expected_label, random_state=seed
        )
    return DatasetSplits(*(part.reset_index(drop=True) for part in (train, validation, test)))


def load_emotion_dataset(config: ExperimentConfig) -> DatasetSplits:
    """Fetch the configured HF revision only when explicitly loading data."""
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset.name, revision=config.dataset.revision, split=config.dataset.split
    )
    return split_emotion(
        dataset.to_pandas(), seed=config.seed, samples_per_label=config.dataset.samples_per_label,
        ratios=config.split,
    )
