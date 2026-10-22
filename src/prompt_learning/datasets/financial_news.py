"""Financial-news normalization and reproducible, duplicate-free splits."""

from math import floor

import pandas as pd

from prompt_learning.config import ExperimentConfig, SplitConfig
from prompt_learning.models import DatasetSplits

from .base import content_identity, find_column, require_sample_count, split_ratios


def split_financial(
    frame: pd.DataFrame, *, seed: int, sample_count: int = 60,
    ratios: SplitConfig = SplitConfig(),
) -> DatasetSplits:
    """Deduplicate before sampling and allocate all rows by largest remainder.

    Equal fractional remainders are resolved in train/validation/test order.
    Sample at most the requested number of unique articles; require five.
    """
    require_sample_count(sample_count, "sample_count")
    proportions = split_ratios(ratios)
    article = find_column(frame, ("article", "content", "text", "document", "source"))
    summary = find_column(frame, ("reference_summary", "summary", "highlights", "target", "abstract"))
    normalized = frame[[article, summary]].rename(
        columns={article: "article", summary: "reference_summary"}
    ).copy()
    for column in ("article", "reference_summary"):
        valid = normalized[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        normalized = normalized.loc[valid].copy()
        normalized[column] = normalized[column].str.strip()
    normalized["example_id"] = normalized.article.map(content_identity)
    normalized = normalized.drop_duplicates("example_id")
    if len(normalized) < 5:
        raise ValueError("Financial dataset needs at least 5 unique articles with reference summaries")
    sampled = normalized.sample(frac=1, random_state=seed).head(sample_count)
    sampled = sampled[["example_id", "article", "reference_summary"]].reset_index(drop=True)
    quotas = [len(sampled) * ratio for ratio in proportions]
    counts = [floor(quota) for quota in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:len(sampled) - sum(counts)]:
        counts[index] += 1
    train_end, validation_end = counts[0], counts[0] + counts[1]
    return DatasetSplits(
        sampled.iloc[:train_end].reset_index(drop=True),
        sampled.iloc[train_end:validation_end].reset_index(drop=True),
        sampled.iloc[validation_end:].reset_index(drop=True),
    )


def load_financial_dataset(config: ExperimentConfig) -> DatasetSplits:
    """Load a full configured HF split so deduplication precedes sampling."""
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset.name, revision=config.dataset.revision, split=config.dataset.split
    )
    return split_financial(
        dataset.to_pandas(), seed=config.seed, sample_count=config.dataset.sample_count,
        ratios=config.split,
    )
