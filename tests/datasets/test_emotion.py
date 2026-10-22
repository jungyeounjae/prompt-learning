from itertools import combinations

import pandas as pd
import pytest

from prompt_learning.datasets.emotion import split_emotion


LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


def emotion_frame(count=25):
    return pd.DataFrame([
        {"text": f"synthetic emotion {label} example {i}", "label": label}
        for label in range(6) for i in range(count)
    ])


def subsets(splits):
    return (splits.train, splits.validation, splits.test)


def test_emotion_split_is_reproducible_and_stratified():
    frame = emotion_frame()
    original = frame.copy(deep=True)
    first = split_emotion(frame, seed=42)
    second = split_emotion(frame, seed=42)
    for actual, repeated, size, per_label in zip(
        subsets(first), subsets(second), (72, 24, 24), (12, 4, 4)
    ):
        pd.testing.assert_frame_equal(actual, repeated)
        assert len(actual) == size
        assert actual.expected_label.value_counts().to_dict() == dict.fromkeys(LABELS, per_label)
        assert set(actual.columns) == {"example_id", "text", "expected_label"}
    pd.testing.assert_frame_equal(frame, original)
    assert first.train.example_id.tolist() != split_emotion(frame, seed=7).train.example_id.tolist()


def test_emotion_subsets_are_disjoint():
    frame = emotion_frame(5)
    duplicate = frame.iloc[[0]].copy()
    duplicate["text"] = "  SYNTHETIC   emotion 0 example 0  "
    parts = subsets(split_emotion(pd.concat([frame, duplicate]), seed=42))
    for left, right in combinations(parts, 2):
        assert set(left.example_id).isdisjoint(right.example_id)
    assert sum(len(part) for part in parts) == 30
    assert pd.concat(parts).example_id.nunique() == 30


def test_emotion_split_rejects_underrepresented_label():
    frame = emotion_frame(5)
    frame = frame.drop(frame[frame.label.eq(5)].index[-1])
    with pytest.raises(ValueError, match="surprise.*5"):
        split_emotion(frame, seed=42)


def test_emotion_minimum_and_balanced_available_sample():
    parts = subsets(split_emotion(emotion_frame(5), seed=42))
    assert [len(part) for part in parts] == [18, 6, 6]
    assert all(set(part.expected_label) == set(LABELS) for part in parts)


def test_emotion_caps_every_label_to_smallest_available_class():
    frame = emotion_frame(5)
    extra = emotion_frame(20).loc[lambda rows: rows.label.ne(5)]
    parts = subsets(split_emotion(pd.concat([frame, extra]), seed=42))
    assert pd.concat(parts).expected_label.value_counts().to_dict() == dict.fromkeys(LABELS, 5)


def test_emotion_configurable_ratios_are_reproducible_and_stratified():
    from prompt_learning.config import SplitConfig

    ratios = SplitConfig(train=0.5, validation=0.25, test=0.25)
    first = split_emotion(emotion_frame(), seed=42, ratios=ratios)
    repeated = split_emotion(emotion_frame(), seed=42, ratios=ratios)
    for part, other, size, per_label in zip(subsets(first), subsets(repeated),
                                          (60, 30, 30), (10, 5, 5)):
        pd.testing.assert_frame_equal(part, other)
        assert len(part) == size
        assert part.expected_label.value_counts().to_dict() == dict.fromkeys(LABELS, per_label)


@pytest.mark.parametrize("ratios,wanted", [((1, 0, 0), [120, 0, 0]),
                                         ((0, 0.5, 0.5), [0, 60, 60]),
                                         ((0.5, 0, 0.5), [60, 0, 60]),
                                         ((0.5, 0.5, 0), [60, 60, 0])])
def test_emotion_supports_zero_ratio_subsets(ratios, wanted):
    from prompt_learning.config import SplitConfig

    parts = subsets(split_emotion(emotion_frame(), seed=42, ratios=SplitConfig(*ratios)))
    assert [len(part) for part in parts] == wanted
    assert pd.concat(parts).example_id.nunique() == 120


@pytest.mark.parametrize("ratios", [(0.6, 0.3, 0.3), (-0.1, 0.5, 0.6), (float("nan"), 0.2, 0.2)])
def test_emotion_rejects_invalid_ratios(ratios):
    from prompt_learning.config import SplitConfig

    with pytest.raises(ValueError, match="ratios"):
        split_emotion(emotion_frame(), seed=42, ratios=SplitConfig(*ratios))


@pytest.mark.parametrize("count", [0, 4, True, 5.5])
def test_emotion_rejects_sample_size_below_split_minimum(count):
    with pytest.raises(ValueError, match="samples_per_label"):
        split_emotion(emotion_frame(), seed=42, samples_per_label=count)


def test_emotion_rejects_unknown_label():
    frame = emotion_frame()
    frame.loc[0, "label"] = 99
    with pytest.raises(ValueError, match="99"):
        split_emotion(frame, seed=42)


def test_emotion_accepts_canonical_labels():
    frame = emotion_frame(5).rename(columns={"label": "expected_label"})
    frame["expected_label"] = frame.expected_label.map(dict(enumerate(LABELS)))
    assert len(split_emotion(frame, seed=42).train) == 18


def test_emotion_rejects_conflicting_duplicate_labels():
    frame = emotion_frame()
    conflicting = frame.iloc[[0]].assign(label=1)
    with pytest.raises(ValueError, match="conflicting"):
        split_emotion(pd.concat([frame, conflicting]), seed=42)


def test_emotion_loader_respects_revision_split_seed_and_sample_size(monkeypatch):
    import sys
    from types import SimpleNamespace

    from prompt_learning.config import DatasetConfig, ExperimentConfig, ModelConfig, SplitConfig
    from prompt_learning.datasets.emotion import load_emotion_dataset

    def fake_load(name, *, revision, split):
        assert (name, revision, split) == ("synthetic/emotion", "fixed-revision", "validation")
        return SimpleNamespace(to_pandas=lambda: emotion_frame(25))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load))
    config = ExperimentConfig(task="emotion", seed=7, split=SplitConfig(0.5, 0.25, 0.25),
                              dataset=DatasetConfig("synthetic/emotion", "fixed-revision",
                                                    split="validation", samples_per_label=10),
                              models=ModelConfig("fake", "fake", "fake"))
    actual = load_emotion_dataset(config)
    expected = split_emotion(emotion_frame(), seed=7, samples_per_label=10, ratios=config.split)
    assert [len(part) for part in subsets(actual)] == [30, 15, 15]
    for part, wanted in zip(subsets(actual), subsets(expected)):
        pd.testing.assert_frame_equal(part, wanted)
