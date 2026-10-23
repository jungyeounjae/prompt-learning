import json

from prompt_learning.comparison import collect_runs, write_comparison


def _run(path, seed, score):
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"run_id": path.name, "task": "emotion", "seed": seed, "dataset": {"samples_per_label": 100, "sample_count": 60}}))
    (path / "validation_metrics.json").write_text(json.dumps({"accuracy": score, "macro_f1": score}))
    (path / "test_metrics.json").write_text(json.dumps({"accuracy": score - .1, "macro_f1": score - .1}))


def test_comparison_uses_only_completed_artifacts(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    _run(first, 42, .8)
    _run(second, 123, .7)
    rows = collect_runs([first, second])
    assert [row["seed"] for row in rows] == [42, 123]
    output = write_comparison([first, second], tmp_path / "comparison.csv")
    assert output.exists() and (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison-summary.csv").exists()
    assert "validation_macro_f1" in (tmp_path / "comparison.md").read_text()
