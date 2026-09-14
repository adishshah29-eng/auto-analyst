"""Stage 1: Load & Profile.

Deliberately *not* LLM-generated code: profiling a dataframe by dtype is a
mechanical operation with no judgment calls, so it runs as plain trusted
pandas in-process (no sandbox needed — we wrote this code, it isn't
attacker-influenced). Every later stage (clean/explore/chart) is where
the agent actually writes and executes code, because those steps require
reasoning about what this specific dataset needs.

Nothing here assumes particular column names — only dtypes, null counts,
cardinality, and generic stats, so it works on any schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent.state import AnalysisState

MAX_TOP_CATEGORIES = 8
MAX_CATEGORY_VALUE_LEN = 80  # truncate long/adversarial cell values before they ever reach a prompt


def _infer_datetime_columns(df: pd.DataFrame, min_parse_rate: float = 0.9) -> pd.DataFrame:
    """Generic heuristic, not tied to any column name: an object column
    whose non-null values mostly parse as dates gets converted to
    datetime64, so it profiles/charts as a time column instead of a
    high-cardinality categorical full of near-unique date strings."""
    for col in df.columns:
        # pandas >= 3 defaults string columns to a "str" dtype rather than
        # object, so check is_string_dtype (which covers both) rather than
        # is_object_dtype alone.
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if parsed.notna().mean() >= min_parse_rate:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return _infer_datetime_columns(df)


def _safe_str(v: Any) -> str:
    s = str(v)
    return s if len(s) <= MAX_CATEGORY_VALUE_LEN else s[:MAX_CATEGORY_VALUE_LEN] + "…"


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    columns: dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isna().sum())
        col_info: dict[str, Any] = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(100 * null_count / n_rows, 2) if n_rows else 0.0,
            "n_unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info["kind"] = "numeric"
            col_info["stats"] = {
                k: (None if pd.isna(v) else round(float(v), 4))
                for k, v in desc.to_dict().items()
            }
            non_null = series.dropna()
            if len(non_null) > 2:
                col_info["skew"] = round(float(non_null.skew()), 4)
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["kind"] = "datetime"
            non_null = series.dropna()
            if len(non_null):
                col_info["min"] = _safe_str(non_null.min())
                col_info["max"] = _safe_str(non_null.max())
        else:
            col_info["kind"] = "categorical"
            top = series.value_counts(dropna=True).head(MAX_TOP_CATEGORIES)
            col_info["top_values"] = {_safe_str(k): int(v) for k, v in top.items()}

        columns[_safe_str(col)] = col_info

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def run(state: AnalysisState, df: pd.DataFrame) -> None:
    state["dataset_schema"] = profile_dataset(df)
