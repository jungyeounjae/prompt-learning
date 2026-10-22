from itertools import combinations

import pandas as pd
import pytest

from prompt_learning.datasets.financial_news import split_financial


def financial_frame(count=80):
    return pd.DataFrame({
        "article": [f"Synthetic financial article {i}" for i in range(count)],
        "reference_summary": [f"Synthetic reference {i}" for i in range(count)],
    })


def subsets(splits):
    return (splits.train, splits.validation, splits.test)


def test_financial_split_is_reproducible():
    frame = financial_frame()
    original = frame.copy(deep=True)
    first = split_financial(frame, seed=42)
    second = split_financial(frame, seed=42)
    assert [len(part) for part in subsets(first)] == [36, 12, 12]
    for part, repeated in zip(subsets(first), subsets(second)):
        pd.testing.assert_frame_equal(part, repeated)
    pd.testing.assert_frame_equal(frame, original)
    assert first.train.example_id.tolist() != split_financial(frame, seed=7).train.example_id.tolist()


def test_financial_configurable_ratios_are_reproducible():
    from prompt_learning.config import SplitConfig

    ratios = SplitConfig(train=0.5, validation=0.25, test=0.25)
    first = split_financial(financial_frame(), seed=42, ratios=ratios)
    repeated = split_financial(financial_frame(), seed=42, ratios=ratios)
    assert [len(part) for part in subsets(first)] == [30, 15, 15]
    for part, other in zip(subsets(first), subsets(repeated)):
        pd.testing.assert_frame_equal(part, other)


def test_financial_supports_zero_ratio_subsets():
    from prompt_learning.config import SplitConfig

    parts = subsets(split_financial(financial_frame(), seed=42, ratios=SplitConfig(1, 0, 0)))
    assert [len(part) for part in parts] == [60, 0, 0]
    assert pd.concat(parts).example_id.nunique() == 60


@pytest.mark.parametrize("ratios", [(0.6, 0.3, 0.3), (-0.1, 0.5, 0.6), (float("nan"), 0.2, 0.2)])
def test_financial_rejects_invalid_ratios(ratios):
    from prompt_learning.config import SplitConfig

    with pytest.raises(ValueError, match="ratios"):
        split_financial(financial_frame(), seed=42, ratios=SplitConfig(*ratios))


def test_financial_split_prevents_duplicate_leakage():
    frame = financial_frame(5)
    frame["example_id"] = "repeated-source-id"
    duplicates = pd.DataFrame({"article": ["  SYNTHETIC   financial article 0 ", "Ｓynthetic financial article 1"],
                               "reference_summary": ["Other reference", "Other reference"]})
    parts = subsets(split_financial(pd.concat([frame, duplicates]), seed=42))
    assert [len(part) for part in parts] == [3, 1, 1]
    assert pd.concat(parts).example_id.nunique() == 5
    for left, right in combinations(parts, 2):
        assert set(left.example_id).isdisjoint(right.example_id)


def test_financial_split_rejects_fewer_than_five_unique_articles():
    frame = pd.concat([financial_frame(4)] * 3)
    with pytest.raises(ValueError, match="5 unique articles"):
        split_financial(frame, seed=42)


@pytest.mark.parametrize("article,summary", [("Content", "Summary"), ("TEXT", "Highlights"),
                                             ("document", "target"), ("source", "abstract")])
def test_financial_alias_columns_are_normalized(article, summary):
    frame = financial_frame(5).rename(columns={"article": article, "reference_summary": summary})
    combined = pd.concat(subsets(split_financial(frame, seed=42)))
    assert set(combined.columns) == {"example_id", "article", "reference_summary"}
    assert set(combined.article) == set(frame[article])
    assert set(combined.reference_summary) == set(frame[summary])


@pytest.mark.parametrize("count,wanted", [(6, [4, 1, 1]), (7, [4, 2, 1]), (8, [5, 2, 1]),
                                        (9, [5, 2, 2]), (11, [7, 2, 2])])
def test_financial_largest_remainder_allocates_every_row(count, wanted):
    parts = subsets(split_financial(financial_frame(count), seed=42))
    assert [len(part) for part in parts] == wanted
    assert pd.concat(parts).example_id.nunique() == count


@pytest.mark.parametrize("count", [0, 4, True, 5.5])
def test_financial_rejects_sample_count_below_split_minimum(count):
    with pytest.raises(ValueError, match="sample_count"):
        split_financial(financial_frame(), seed=42, sample_count=count)


def test_financial_filters_missing_and_blank_source_rows():
    invalid = pd.DataFrame({"article": [None, "", "  ", "valid"],
                            "reference_summary": ["summary", "summary", "summary", None]})
    combined = pd.concat(subsets(split_financial(pd.concat([financial_frame(5), invalid]), seed=42)))
    assert len(combined) == 5


def test_financial_missing_reference_column_is_actionable():
    with pytest.raises(ValueError, match="reference_summary"):
        split_financial(financial_frame().drop(columns="reference_summary"), seed=42)


def test_financial_loader_respects_revision_split_seed_and_sample_count(monkeypatch):
    import sys
    from types import SimpleNamespace

    from prompt_learning.config import DatasetConfig, ExperimentConfig, ModelConfig, SplitConfig
    from prompt_learning.datasets.financial_news import load_financial_dataset

    def fake_load(name, *, revision, split):
        assert (name, revision, split) == ("synthetic/financial", "fixed-revision", "test")
        return SimpleNamespace(to_pandas=financial_frame)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load))
    config = ExperimentConfig(task="financial", seed=7, split=SplitConfig(0.5, 0.25, 0.25),
                              dataset=DatasetConfig("synthetic/financial", "fixed-revision",
                                                    split="test", sample_count=10),
                              models=ModelConfig("fake", "fake", "fake"))
    actual = load_financial_dataset(config)
    expected = split_financial(financial_frame(), seed=7, sample_count=10, ratios=config.split)
    assert [len(part) for part in subsets(actual)] == [5, 3, 2]
    for part, wanted in zip(subsets(actual), subsets(expected)):
        pd.testing.assert_frame_equal(part, wanted)
