"""Persist reproducible experiment results without copying dataset text."""

import csv
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any
import zlib

from prompt_learning.models import MetricResult, PromptCandidate, RunManifest, SelectionDecision


_RUN_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_EMOTION_METRICS = ("accuracy", "macro_f1", "invalid_output_rate")
_FINANCIAL_METRICS = (
    "overall_score", "mean_financial_figures", "mean_conciseness", "mean_faithfulness",
    "financial_figures_not_applicable_count", "severe_hallucination_count", "failure_count",
)
_REGRESSION_FIELDS = {
    "emotion": ("example_id", "expected_label", "baseline_output", "candidate_output"),
    "financial": ("example_id", "baseline_score", "candidate_score", "delta"),
}
_HISTORY_FIELDS = ("iteration", "prompt_version", "prompt_hash", "accepted", "reasons")


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(fields: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _prompt_text(prompt: PromptCandidate | str) -> str:
    return prompt.text if isinstance(prompt, PromptCandidate) else prompt


def _public_metrics(task: str, result: MetricResult) -> dict[str, Any]:
    aggregate = result.aggregate
    allowed = _EMOTION_METRICS if task == "emotion" else _FINANCIAL_METRICS
    public = {key: aggregate[key] for key in allowed if key in aggregate}
    if task == "emotion":
        if "per_class" in aggregate:
            public["per_class"] = {
                str(label): {key: scores[key] for key in ("precision", "recall", "f1", "support") if key in scores}
                for label, scores in aggregate["per_class"].items()
            }
        if "confusion_matrix" in aggregate:
            matrix = aggregate["confusion_matrix"]
            public["confusion_matrix"] = {"labels": matrix["labels"], "matrix": matrix["matrix"]}
    return public


def _manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    config = manifest.config
    return {
        "run_id": manifest.run_id,
        "task": manifest.task,
        "dataset": {"name": config.dataset.name, "revision": config.dataset.revision,
                    "split": config.dataset.split, "samples_per_label": config.dataset.samples_per_label,
                    "sample_count": config.dataset.sample_count},
        "dataset_revision": manifest.dataset_revision,
        "models": dict(manifest.models),
        "seed": manifest.seed,
        "split": {"train": config.split.train, "validation": config.split.validation,
                  "test": config.split.test},
        "retry_count": config.retry.count,
        "failure_rate": config.failure_rate,
        "max_iterations": config.max_iterations,
        "temperature": config.models.temperature,
        "pricing": {name: {"input_per_million": str(price.input_per_million),
                           "output_per_million": str(price.output_per_million)}
                    for name, price in config.pricing.items()},
        "evaluation_sha256": sha256(json.dumps(
            config.evaluation, ensure_ascii=False, sort_keys=True, default=str,
        ).encode("utf-8")).hexdigest(),
        "started_at": manifest.started_at.isoformat(),
        "finished_at": manifest.finished_at.isoformat() if manifest.finished_at else None,
        "artifact_path": str(manifest.artifact_path),
        "input_hashes": {split: list(hashes) for split, hashes in manifest.input_hashes.items()},
    }


def _matrix_png(matrix: dict[str, Any]) -> bytes:
    values = matrix["matrix"]
    side = len(matrix["labels"])
    if not side or len(values) != side or any(len(row) != side for row in values):
        raise ValueError("Confusion matrix must be square")
    cell = 40
    width = height = side * cell
    maximum = max(1, *(count for row in values for count in row))
    scanlines = bytearray()
    for y in range(height):
        scanlines.append(0)  # PNG filter: none
        for x in range(width):
            count = values[y // cell][x // cell]
            intensity = round(220 * count / maximum)
            color = (255 - intensity, 255 - intensity // 2, 255)
            scanlines.extend((80, 95, 115) if x % cell == 0 or y % cell == 0 else color)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(scanlines)))
            + chunk(b"IEND", b""))


class ArtifactWriter:
    """Own one run directory and publish only approved result fields."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path: Path | None = None
        self.task: str | None = None
        self._history: list[dict[str, object]] = []
        self._regressions: list[dict[str, object]] = []

    def start_run(self, task: str, run_id: str) -> Path:
        if task not in _REGRESSION_FIELDS or not _RUN_COMPONENT.fullmatch(run_id) or run_id in (".", ".."):
            raise ValueError("Invalid task or run ID")
        if self.path is not None:
            raise RuntimeError("This writer already owns a run")
        path = self.root / task / run_id
        path.mkdir(parents=True, exist_ok=False)
        self.path, self.task = path, task
        return path

    def _active(self) -> tuple[Path, str]:
        if self.path is None or self.task is None:
            raise RuntimeError("Call start_run before writing artifacts")
        return self.path, self.task

    def write_iteration(
        self, iteration: int, prompt: PromptCandidate, metrics: MetricResult,
        decision: SelectionDecision | None,
    ) -> None:
        """Record a baseline pass with ``decision=None`` or a selected candidate."""
        path, task = self._active()
        if iteration < 0 or any(row["iteration"] == iteration for row in self._history):
            raise ValueError("Iteration must be non-negative and unique")
        history = [*self._history, {
            "iteration": iteration, "prompt_version": prompt.version, "prompt_hash": prompt.hash,
            "accepted": "" if decision is None else decision.accepted,
            "reasons": "" if decision is None else ";".join(decision.reasons),
        }]
        regressions = [*self._regressions]
        if decision is not None:
            for regression in decision.regressions:
                regressions.append({"iteration": iteration, **{
                    key: regression.get(key, "") for key in _REGRESSION_FIELDS[task]
                }})
        _atomic_write(path / "iteration_history.csv", _csv_bytes(_HISTORY_FIELDS, history))
        _atomic_write(path / f"iteration_{iteration}_prompt.txt", prompt.text.encode("utf-8"))
        _atomic_write(path / f"iteration_{iteration}_validation_metrics.json", _json_bytes(_public_metrics(task, metrics)))
        _atomic_write(path / "regressions.csv", _csv_bytes(("iteration", *_REGRESSION_FIELDS[task]), regressions))
        self._history, self._regressions = history, regressions

    def write_optimization_error(self, iteration: int, reason: str) -> None:
        """Archive a failed generation attempt without inventing a candidate."""
        path, _ = self._active()
        if iteration < 0 or any(row["iteration"] == iteration for row in self._history):
            raise ValueError("Iteration must be non-negative and unique")
        category = reason if reason in ("empty_output", "missing_placeholder") else "invalid_candidate"
        history = [*self._history, {
            "iteration": iteration, "prompt_version": "", "prompt_hash": "", "accepted": False,
            "reasons": f"optimization_error:{category}",
        }]
        _atomic_write(path / "iteration_history.csv", _csv_bytes(_HISTORY_FIELDS, history))
        self._history = history

    def finalize(
        self, manifest: RunManifest, baseline_prompt: PromptCandidate | str,
        selected_prompt: PromptCandidate | str, validation_metrics: MetricResult,
        test_metrics: MetricResult,
    ) -> Path:
        path, task = self._active()
        if manifest.task != task or manifest.run_id != path.name or manifest.artifact_path != path:
            raise ValueError("Manifest does not match the active run")
        validation = _public_metrics(task, validation_metrics)
        test = _public_metrics(task, test_metrics)
        report_lines = [
            f"# Run {manifest.run_id}", "", f"Task: {task}",
            f"Dataset: {manifest.config.dataset.name} @ {manifest.dataset_revision}",
            f"Seed: {manifest.seed}", "", "## Validation metrics", "",
        ]
        report_lines.extend(f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in validation.items()
                            if key != "confusion_matrix")
        report_lines.extend(["", "## Test metrics", ""])
        report_lines.extend(f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in test.items()
                            if key != "confusion_matrix")
        report_lines.extend(["", "## Iterations", ""])
        for entry in self._history:
            report_lines.append(
                f"- {entry['iteration']}: accepted={entry['accepted']}, reasons={entry['reasons'] or 'none'}, "
                f"prompt_sha256={entry['prompt_hash']}"
            )
        _atomic_write(path / "manifest.json", _json_bytes(_manifest_payload(manifest)))
        _atomic_write(path / "baseline_prompt.txt", _prompt_text(baseline_prompt).encode("utf-8"))
        _atomic_write(path / "selected_prompt.txt", _prompt_text(selected_prompt).encode("utf-8"))
        _atomic_write(path / "validation_metrics.json", _json_bytes(validation))
        _atomic_write(path / "test_metrics.json", _json_bytes(test))
        _atomic_write(path / "iteration_history.csv", _csv_bytes(_HISTORY_FIELDS, self._history))
        _atomic_write(path / "regressions.csv", _csv_bytes(("iteration", *_REGRESSION_FIELDS[task]), self._regressions))
        _atomic_write(path / "report.md", ("\n".join(report_lines) + "\n").encode("utf-8"))
        if task == "emotion" and "confusion_matrix" in test:
            _atomic_write(path / "confusion_matrix.png", _matrix_png(test["confusion_matrix"]))
        return path
