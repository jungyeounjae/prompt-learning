"""Deprecated financial command; use prompt-learning run --config configs/financial.yaml."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from prompt_learning import cli
from prompt_learning.config import ConfigOverrides, merge_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "financial.yaml"


def financial_legacy_args(argv: Sequence[str] | None = None) -> ConfigOverrides:
    """Translate supported legacy flags, leaving omitted values to shared defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--samples", type=int, help="total sample count (minimum 5)")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--summarizer-model")
    parser.add_argument("--evaluator-model")
    parser.add_argument("--meta-model")
    parser.add_argument("--output-dir", help="artifact root containing per-task, per-run directories")
    parser.add_argument("--final-prompt", help="unsupported; each run artifact contains selected_prompt.txt")
    args = parser.parse_args(argv)
    if args.final_prompt is not None:
        parser.error("--final-prompt is unsupported: use --output-dir; each run writes selected_prompt.txt")
    if args.samples is not None and args.samples < 5:
        parser.error("--samples must be at least 5 for isolated Train/Validation/Test subsets")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be at least 1")
    overrides: ConfigOverrides = {}
    dataset = {
        key: value for key, value in (
            ("name", args.dataset), ("split", args.split), ("sample_count", args.samples),
        ) if value is not None
    }
    if dataset:
        overrides["dataset"] = dataset
    models = {
        key: value for key, value in (
            ("inference", args.summarizer_model), ("evaluator", args.evaluator_model),
            ("meta", args.meta_model),
        ) if value is not None
    }
    if models:
        overrides["models"] = models
    if args.iterations is not None:
        overrides["max_iterations"] = args.iterations
    if args.output_dir is not None:
        overrides["artifact_root"] = args.output_dir
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    print("Deprecated: use prompt-learning run --config configs/financial.yaml", file=sys.stderr)
    overrides = financial_legacy_args(argv)
    with merge_config(DEFAULT_CONFIG, overrides) as path:
        return cli.main(["run", "--config", str(path)])


if __name__ == "__main__":
    raise SystemExit(main())
