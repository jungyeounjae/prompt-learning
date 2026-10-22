"""Artifact files are reproducible, readable, and safe to publish."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from prompt_learning.artifacts import ArtifactWriter
from prompt_learning.config import DatasetConfig, ExperimentConfig, ModelConfig
from prompt_learning.models import MetricResult, PromptCandidate, RunManifest, SelectionDecision


def manifest(path: Path, task: str) -> RunManifest:
    config = ExperimentConfig(
        task=task, dataset=DatasetConfig("sample", "revision-1"),
        models=ModelConfig("inference", "evaluator", "meta"),
        evaluation={"private_article": "DO NOT PUBLISH ARTICLE"},
    )
    return RunManifest(
        "run-1", task, config, "revision-1", {"inference": "inference"}, 42,
        datetime(2026, 10, 22, tzinfo=timezone.utc), path,
        datetime(2026, 10, 22, 1, tzinfo=timezone.utc),
        {"validation": ("a" * 64,)},
    )


def emotion_metrics() -> MetricResult:
    return MetricResult(
        {"accuracy": 0.5, "macro_f1": 0.4, "invalid_output_rate": 0.0,
         "per_class": {"joy": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "support": 2}},
         "confusion_matrix": {"labels": ["joy", "sadness"], "matrix": [[1, 1], [0, 1]]},
         "secret": "DO NOT PUBLISH ARTICLE"},
        pd.DataFrame([{"example_id": "id-1", "text": "DO NOT PUBLISH ARTICLE",
                       "reference_summary": "DO NOT PUBLISH REFERENCE", "output": "DO NOT PUBLISH OUTPUT"}]),
        "DO NOT PUBLISH FEEDBACK",
    )


def financial_metrics() -> MetricResult:
    return MetricResult(
        {"overall_score": 80.0, "mean_financial_figures": 4.0,
         "mean_conciseness": 4.0, "mean_faithfulness": 4.0,
         "financial_figures_not_applicable_count": 2,
         "severe_hallucination_count": 0, "failure_count": 0},
        pd.DataFrame([{"example_id": "id-1", "article": "DO NOT PUBLISH ARTICLE",
                       "reference_summary": "DO NOT PUBLISH REFERENCE"}]),
    )


def test_start_run_rejects_duplicate_id_and_unsafe_paths(tmp_path):
    writer = ArtifactWriter(tmp_path)
    path = writer.start_run("emotion", "run-1")
    assert path == tmp_path / "emotion" / "run-1"
    with pytest.raises(FileExistsError):
        ArtifactWriter(tmp_path).start_run("emotion", "run-1")
    with pytest.raises(ValueError):
        ArtifactWriter(tmp_path).start_run("emotion", "../outside")


def test_emotion_run_writes_required_files_and_redacts_source_text(tmp_path):
    writer = ArtifactWriter(tmp_path)
    path = writer.start_run("emotion", "run-1")
    baseline, candidate = PromptCandidate("Classify {text}", 0, 0), PromptCandidate("Classify carefully {text}", 1, 1)
    metrics = emotion_metrics()
    writer.write_iteration(0, baseline, metrics, None)
    writer.write_iteration(1, candidate, metrics, SelectionDecision(
        False, ("macro_f1_decreased",),
        ({"example_id": "id-1", "expected_label": "joy", "baseline_output": "joy",
          "candidate_output": "sadness", "text": "DO NOT PUBLISH ARTICLE"},),
    ))
    assert writer.finalize(manifest(path, "emotion"), baseline, baseline, metrics, metrics) == path
    expected = {"manifest.json", "baseline_prompt.txt", "selected_prompt.txt", "iteration_history.csv",
                "validation_metrics.json", "test_metrics.json", "regressions.csv", "report.md",
                "confusion_matrix.png", "iteration_1_prompt.txt", "iteration_1_validation_metrics.json"}
    assert expected <= {file.name for file in path.iterdir()}
    assert (path / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest_data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["seed"] == 42
    assert len(manifest_data["evaluation_sha256"]) == 64
    assert json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))["accuracy"] == 0.5
    assert (path / "selected_prompt.txt").read_text(encoding="utf-8") == baseline.text
    with (path / "iteration_history.csv").open(encoding="utf-8", newline="") as file:
        history = list(csv.DictReader(file))
    assert [row["iteration"] for row in history] == ["0", "1"]
    assert history[0]["accepted"] == "" and history[1]["accepted"] == "False"
    with (path / "regressions.csv").open(encoding="utf-8", newline="") as file:
        regressions = list(csv.DictReader(file))
    assert regressions == [{"iteration": "1", "example_id": "id-1", "expected_label": "joy",
                            "baseline_output": "joy", "candidate_output": "sadness"}]
    published = "\n".join(file.read_text(encoding="utf-8") for file in path.iterdir() if file.suffix != ".png")
    for secret in ("DO NOT PUBLISH ARTICLE", "DO NOT PUBLISH REFERENCE", "DO NOT PUBLISH OUTPUT",
                   "DO NOT PUBLISH FEEDBACK"):
        assert secret not in published


def test_financial_report_shows_dimension_scores_and_na_count_without_png(tmp_path):
    writer = ArtifactWriter(tmp_path)
    path = writer.start_run("financial", "run-1")
    prompt, metrics = PromptCandidate("Summarize {article}", 0, 0), financial_metrics()
    writer.write_iteration(0, prompt, metrics, None)
    writer.finalize(manifest(path, "financial"), prompt, prompt, metrics, metrics)
    report = (path / "report.md").read_text(encoding="utf-8")
    assert "mean_financial_figures" in report and "mean_conciseness" in report
    assert "mean_faithfulness" in report and "financial_figures_not_applicable_count" in report
    assert "2" in report
    assert not (path / "confusion_matrix.png").exists()
    assert "DO NOT PUBLISH ARTICLE" not in report


def test_write_failure_propagates_and_preserves_existing_file(tmp_path, monkeypatch):
    writer = ArtifactWriter(tmp_path)
    path = writer.start_run("emotion", "run-1")
    original = path / "iteration_history.csv"
    original.write_text("old contents", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr("prompt_learning.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        writer.write_iteration(0, PromptCandidate("Classify {text}", 0, 0), emotion_metrics(), None)
    assert original.read_text(encoding="utf-8") == "old contents"
    assert not [file for file in path.iterdir() if file.name.endswith(".tmp")]
