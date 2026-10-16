from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_market_history(path: str | Path, date_column: str = "Date") -> pd.DataFrame:
    """Load a wide market-history CSV with one row per business date."""
    frame = pd.read_csv(path)
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    frame = frame.set_index(date_column).sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Market history contains duplicate dates")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric observations in: {bad}")
    return numeric


def absolute_shocks(levels: pd.DataFrame, holding_period_days: int = 1) -> pd.DataFrame:
    """Absolute changes, suitable for rates and normal volatilities."""
    if holding_period_days < 1:
        raise ValueError("holding_period_days must be positive")
    return levels.diff(holding_period_days).dropna()


def validate_current_state(current: pd.Series, factor_names: list[str]) -> None:
    missing = [name for name in factor_names if name not in current.index]
    if missing:
        raise ValueError(f"Current market state is missing factors: {missing}")
    if current[factor_names].isna().any():
        raise ValueError("Current market state contains missing values")

