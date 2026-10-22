"""Shared, pure normalization helpers; source text is never persisted here."""

from hashlib import sha256
from math import isclose, isfinite
import unicodedata

import pandas as pd

from prompt_learning.config import SplitConfig


def content_identity(text: str) -> str:
    """Hash Unicode-normalized, case-insensitive text with collapsed whitespace."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return sha256(normalized.encode("utf-8")).hexdigest()


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    raise ValueError(f"Missing required column; expected one of {aliases}")


def require_sample_count(value: int, name: str) -> None:
    if type(value) is not int or value < 5:
        raise ValueError(f"{name} must be an integer >= 5 for dataset splitting")


def split_ratios(ratios: SplitConfig) -> tuple[float, float, float]:
    values = (ratios.train, ratios.validation, ratios.test)
    if (any(isinstance(value, bool) or not isinstance(value, (float, int))
            or not isfinite(value) or not 0 <= value <= 1 for value in values)
            or not isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError(f"Split ratios must be finite values in [0, 1] summing to 1: {values}")
    return values
