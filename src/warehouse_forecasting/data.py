from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASET_COLUMNS = {
    "retail_supply_chain": (["Order Date", "order_date", "date"], ["Quantity", "quantity", "Sales", "sales"]),
    "historical_product_demand": (["Date", "date"], ["Order_Demand", "order_demand", "demand"]),
    "store_item": (["date", "Date"], ["sales", "Sales"]),
}


def _select(columns, candidates, kind):
    match = next((name for name in candidates if name in columns), None)
    if match is None:
        raise ValueError(f"Could not find {kind} column; expected one of {candidates}")
    return match


def load_demand_series(path: str | Path, dataset: str) -> pd.Series:
    """Load, clean, aggregate, and interpolate one of the paper's datasets."""
    if dataset not in DATASET_COLUMNS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASET_COLUMNS)}")
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path)
    dates, targets = DATASET_COLUMNS[dataset]
    date_col = _select(frame.columns, dates, "date")
    target_col = _select(frame.columns, targets, "target")
    values = frame[target_col]
    if values.dtype == object:
        values = values.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    clean = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "demand": pd.to_numeric(values, errors="coerce")})
    clean = clean.dropna(subset=["date"]).groupby("date", as_index=True)["demand"].sum(min_count=1).sort_index()
    clean = clean.resample("D").sum(min_count=1).interpolate("time").ffill().bfill().clip(lower=0)
    if len(clean) < 40:
        raise ValueError("At least 40 daily observations are required")
    return clean.rename("demand")


def demo_series(days: int = 730, seed: int = 42) -> pd.Series:
    """Create a deterministic seasonal demand series for a dependency-free demo."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    demand = 120 + 0.04 * t + 22 * np.sin(2 * np.pi * t / 7) + 12 * np.sin(2 * np.pi * t / 365.25)
    demand += rng.normal(0, 8, days)
    return pd.Series(np.maximum(demand, 0), index=pd.date_range("2023-01-01", periods=days), name="demand")


def supervised_windows(series: pd.Series, lookback: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = series.to_numpy(dtype=float)
    if lookback < 1 or len(values) <= lookback:
        raise ValueError("lookback must be positive and shorter than the series")
    x = np.array([values[i - lookback:i] for i in range(lookback, len(values))])
    y = values[lookback:]
    return x, y, series.index[lookback:]
