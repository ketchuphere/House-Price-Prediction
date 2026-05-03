"""
data/data_loader.py
────────────────────
Responsible for:
  1. Reading the raw CSV from disk.
  2. Cleaning obvious data quality issues (zero-price rows, extreme outliers).
  3. Returning a clean DataFrame that the feature-engineering pipeline can consume.

Design principle: keep this module *stateless* — no sklearn transformers, no
global side-effects.  Everything is a pure function operating on DataFrames.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

#Default dataset path (overridable via env-var for Docker/CI)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "data.csv"
DATA_PATH = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))


#define data loading and cleaning functions
def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the raw house-price CSV from *path* (defaults to DATA_PATH).

    Returns
    -------
    pd.DataFrame
        Raw dataframe — no transformations applied yet.
    """
    csv_path = Path(path) if path else DATA_PATH
    logger.info("Loading raw data", extra={"path": str(csv_path)})
    df = pd.read_csv(csv_path)
    logger.info("Raw data loaded", extra={"rows": len(df), "cols": df.shape[1]})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based data cleaning:

    * Drop rows with null prices.
    * Drop rows where price == 0 (likely data-entry errors).
    * Drop rows where bedrooms == 0 (uninhabitable listings).
    * Cap extreme outliers in ``price`` using the IQR fence (3× IQR above Q3).
    * Parse the ``date`` column into datetime and extract ``sale_year`` /
      ``sale_month`` / ``sale_dayofweek`` features directly here so the
      original string column can be dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy (original is never mutated).
    """
    df = df.copy()
    original_rows = len(df)

    #Drop missing targets
    df.dropna(subset=["price"], inplace=True)

    # ── Drop invalid rows ─────────────────────────────────────────────────────
    df = df[df["price"] > 0]
    df = df[df["bedrooms"] > 0]

    #Outlier removal: IQR fence on price
    q1, q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 3.0 * iqr
    before = len(df)
    df = df[df["price"] <= upper_fence]
    removed = before - len(df)
    if removed:
        logger.info(
            "Outlier rows removed",
            extra={"removed": removed, "upper_fence": round(upper_fence, 2)},
        )

    #Date engineering
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year.fillna(2014).astype(int)
    df["sale_month"] = df["date"].dt.month.fillna(1).astype(int)
    df["sale_dayofweek"] = df["date"].dt.dayofweek.fillna(0).astype(int)

    #Drop columns not useful for ML
    #street: too high-cardinality; country/statezip: near-constant in this dataset
    df.drop(columns=["date", "street", "country"], inplace=True, errors="ignore")

    #Reset index after row drops
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Data cleaning complete",
        extra={"original_rows": original_rows, "clean_rows": len(df)},
    )
    return df


def load_and_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Convenience wrapper: load → clean in one call."""
    return clean_data(load_raw_data(path))
