"""Data preparation for the Heart Disease Risk Classifier (prototype).

Loads the UCI Heart Disease dataset (Cleveland subset: 303 patients, 13 features):
  1. Primary source:  ucimlrepo.fetch_ucirepo(id=45)   (requires internet once)
  2. Fallback source: a local CSV at ``data/heart.csv`` with the standard 13 UCI
     columns (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
     oldpeak, slope, ca, thal, target).

This module only loads + cleans + maps columns to report groups.  It does not
learn/train anything.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Standard UCI Heart Disease column names (Cleveland, 13 features + target).
UCI_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Which medical-report group each input column belongs to.  Keys match the
# form sections in app.py.
REPORT_GROUPS: dict[str, list[str]] = {
    "Lipid / Blood Report": ["chol", "fbs"],
    "ECG Report": ["restecg", "thalach", "oldpeak", "slope", "exang"],
    "Echocardiography Report": ["ca", "thal"],
    "Demographics / Clinical": ["age", "sex", "cp", "trestbps"],
}

# Categorical / ordered-categorical columns (kept out of the numeric scaler,
# fed to one-hot / categorical handling downstream).
CATEGORICAL_COLUMNS: list[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

# In the raw Cleveland file missing values are written as '?' (mostly ca, thal).
MISSING_MARKER = "?"

# UCI ML repo dataset id for "Heart Disease" (Cleveland).
HEART_UCI_ID = 45


def load_heart_data(data_dir: str | os.PathLike = "data") -> pd.DataFrame:
    """Return the 14-column UCI Heart Disease frame with standard names.

    Tries the online ucimlrepo API (HEART_UCI_ID) first; on any failure it
    reads a local CSV at ``data_dir/heart.csv``.  Raises a clear RuntimeError
    only if both paths fail.
    """
    frame: pd.DataFrame | None = None
    source = ""

    # --- Attempt 1: online fetch ---
    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=HEART_UCI_ID)
        frame = heart.data.features.copy()
        targets = heart.data.targets.copy()
        if isinstance(targets, pd.DataFrame):
            targets = targets.iloc[:, 0]
        frame["target"] = pd.Series(targets).values
        source = f"ucimlrepo fetch_ucirepo(id={HEART_UCI_ID})"
    except Exception as exc:  # noqa: BLE001 - deliberate graceful fallback
        print(f"[data_prep] online fetch failed ({type(exc).__name__}: {exc}); trying local CSV ...")
        frame = None

    # --- Attempt 2: local CSV ---
    if frame is None:
        path = Path(data_dir) / "heart.csv"
        if not path.exists():
            raise RuntimeError(
                f"Cannot load the dataset.\n"
                f"- online fetch (ucimlrepo id={HEART_UCI_ID}) failed, and\n"
                f"- no CSV at expected path: {path}\n"
                f"Please either enable internet once, or place the standard UCI heart "
                f"disease CSV at: {path}\nExpected columns: {UCI_COLUMNS}"
            )
        frame = pd.read_csv(path)
        source = f"local CSV {path}"

    frame = _normalise_columns(frame)
    missing = [c for c in UCI_COLUMNS if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Data source is missing required columns: {missing}\nExpected: {UCI_COLUMNS}"
        )
    frame = frame[UCI_COLUMNS]

    print(
        f"[data_prep] Loaded from {source} -> {len(frame)} rows, {len(frame.columns)} cols\n"
        f"[data_prep] Raw target counts:\n{frame['target'].value_counts().sort_index().to_string()}"
    )
    return frame


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def clean(df: pd.DataFrame):
    """Convert '?' -> NaN, enforce numeric dtypes, binarise target.

    Returns a simple namespace with:
      .df          cleaned features (numeric; NaN still present for ca/thal)
      .target      binarised target (1 = disease, 0 = none)
      .report_map  DataFrame: column -> report_group + dtype kind
    """
    df = df.copy()

    # '?' -> NaN (Cleveland stores missing ca/thal as '?').
    df = df.replace(MISSING_MARKER, np.nan)

    feature_cols = [c for c in UCI_COLUMNS if c != "target"]
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarise target: any disease category > 0 => present.
    target = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)

    report_map = pd.DataFrame(
        {
            "column": feature_cols,
            "report_group": [_report_group(c) for c in feature_cols],
            "dtype": [
                "categorical" if c in CATEGORICAL_COLUMNS else "continuous"
                for c in feature_cols
            ],
        }
    )

    class PreparedData:
        __slots__ = ("df", "target", "report_map")

    out = PreparedData()
    out.df = df[feature_cols]
    out.target = target
    out.report_map = report_map
    return out


def _report_group(col: str) -> str:
    for group, cols in REPORT_GROUPS.items():
        if col in cols:
            return group
    return "untagged"


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count + percentage of missing values."""
    n = len(df)
    return pd.DataFrame({"nulls": df.isna().sum()}).assign(
        pct=lambda d: (d["nulls"] / n * 100).round(1)
    )


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute remaining NaN columns (mostly ca / thal in Cleveland)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            med = float(df[col].median())
            if np.isnan(med):
                raise RuntimeError(f"Column '{col}' is entirely missing; cannot impute.")
            n_before = int(df[col].isna().sum())
            df[col] = df[col].fillna(med)
            print(f"[data_prep] imputed '{col}': {n_before} missing -> median {med:.2f}")
    return df
