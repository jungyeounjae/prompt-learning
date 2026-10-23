"""Build reproducible comparison tables from completed run artifacts."""

import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid run artifact: {path}") from exc


def collect_runs(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        manifest = _read(path / "manifest.json")
        validation = _read(path / "validation_metrics.json")
        test = _read(path / "test_metrics.json")
        row: dict[str, Any] = {
            "run_id": manifest.get("run_id", path.name),
            "task": manifest.get("task", ""),
            "seed": manifest.get("seed"),
            "samples_per_label": manifest.get("dataset", {}).get("samples_per_label"),
            "sample_count": manifest.get("dataset", {}).get("sample_count"),
            "validation_accuracy": validation.get("accuracy"),
            "validation_macro_f1": validation.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "validation_overall_score": validation.get("overall_score"),
            "test_overall_score": test.get("overall_score"),
            "artifact_path": str(path),
        }
        rows.append(row)
    return rows


def write_comparison(paths: list[Path], output: Path) -> Path:
    rows = collect_runs(paths)
    if not rows:
        raise ValueError("At least one completed run is required")
    fields = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = output.with_suffix(".md")
    shown = [field for field in fields if field != "artifact_path"]
    lines = ["# Experiment comparison", "", "| " + " | ".join(shown) + " |", "| " + " | ".join("---" for _ in shown) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row[field] is None else str(row[field]) for field in shown) + " |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
