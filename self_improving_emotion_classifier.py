"""Deprecated emotion command; use prompt-learning run --config configs/emotion.yaml."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from prompt_learning import cli
from prompt_learning.config import ConfigOverrides, merge_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "emotion.yaml"


def emotion_legacy_args(argv: Sequence[str] | None = None) -> ConfigOverrides:
    """Translate supported legacy flags, leaving omitted values to shared defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int,
        help="total sample budget, rounded up across six balanced labels (minimum 5 per label)",
    )
    parser.add_argument("--iterations", type=int, help="maximum optimization iterations")
    parser.add_argument("--target-accuracy", help="unsupported; selection now uses Validation gates")
    parser.add_argument("--output", help="unsupported; the run artifact contains selected_prompt.txt")
    args = parser.parse_args(argv)
    if args.target_accuracy is not None:
        parser.error("--target-accuracy is unsupported: selection uses Validation gates; use --iterations for the run budget")
    if args.output is not None:
        parser.error("--output is unsupported: use artifact_root in configs/emotion.yaml; each run writes selected_prompt.txt")
    overrides: ConfigOverrides = {}
    if args.samples is not None:
        if args.samples < 30:
            parser.error("--samples must be at least 30 (5 per label) for isolated Train/Validation/Test subsets")
        per_label = (args.samples + 5) // 6
        overrides["dataset"] = {"samples_per_label": per_label}
        print(f"Legacy --samples={args.samples} maps to {per_label} per label ({per_label * 6} balanced examples).", file=sys.stderr)
    if args.iterations is not None:
        if args.iterations < 1:
            parser.error("--iterations must be at least 1")
        overrides["max_iterations"] = args.iterations
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    print("Deprecated: use prompt-learning run --config configs/emotion.yaml", file=sys.stderr)
    overrides = emotion_legacy_args(argv)
    with merge_config(DEFAULT_CONFIG, overrides) as path:
        return cli.main(["run", "--config", str(path)])


if __name__ == "__main__":
    raise SystemExit(main())
