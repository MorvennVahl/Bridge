"""Data loaders. Cached at module scope so requests are cheap.

Tolerant by design: bad rows are skipped, missing columns tolerated, missing
files return an empty DataFrame with the expected schema. Nothing in this
module should raise on malformed input — the frontend must stay demoable.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _read_csv_tolerant(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV, skipping malformed rows and returning empty on file errors."""
    if not path.is_file():
        log.warning("data file missing: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)
    except Exception as e:
        log.warning("failed to read %s: %s", path, e)
        return pd.DataFrame()


def _coerce_int_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce ``col`` to int64, dropping rows where coercion fails."""
    if col not in df.columns:
        log.warning("column %s missing from dataframe (cols=%s)", col, list(df.columns))
        return df.iloc[0:0]
    before = len(df)
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype("int64")
    dropped = before - len(df)
    if dropped:
        log.info("dropped %d rows with unparseable %s", dropped, col)
    return df


def _coerce_str_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce ``col`` to string, dropping rows where it's null/empty."""
    if col not in df.columns:
        log.warning("column %s missing from dataframe (cols=%s)", col, list(df.columns))
        return df.iloc[0:0]
    before = len(df)
    df = df.copy()
    df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=[col])
    df = df.loc[df[col] != ""]
    dropped = before - len(df)
    if dropped:
        log.info("dropped %d rows with empty %s", dropped, col)
    return df


@lru_cache(maxsize=1)
def load_ingredients() -> pd.DataFrame:
    df = _read_csv_tolerant(DATA_DIR / "cem_ingredients.csv")
    df = _coerce_int_column(df, "ingredient_concept_id")
    df = _coerce_str_column(df, "ingredient_name")
    if df.empty:
        return df
    return df.sort_values("ingredient_name").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_conditions() -> pd.DataFrame:
    """Distinct conditions extracted from the associations file."""
    df = _read_csv_tolerant(
        DATA_DIR / "cem_ingredient_condition_associations.csv",
        usecols=lambda c: c in {"condition_concept_id", "condition_name"},
    )
    df = _coerce_int_column(df, "condition_concept_id")
    df = _coerce_str_column(df, "condition_name")
    if df.empty:
        return df
    return (
        df.drop_duplicates("condition_concept_id")
        .sort_values("condition_name")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def load_associations() -> pd.DataFrame:
    """Full associations table. Large (~121 MB). Loaded on first call."""
    df = _read_csv_tolerant(DATA_DIR / "cem_ingredient_condition_associations.csv")
    df = _coerce_int_column(df, "ingredient_concept_id")
    df = _coerce_int_column(df, "condition_concept_id")
    return df


def data_files_present() -> dict[str, bool]:
    return {
        "cem_ingredients.csv": (DATA_DIR / "cem_ingredients.csv").is_file(),
        "cem_ingredient_condition_associations.csv": (
            DATA_DIR / "cem_ingredient_condition_associations.csv"
        ).is_file(),
        "ingredient_features.csv": (DATA_DIR / "ingredient_features.csv").is_file(),
    }
