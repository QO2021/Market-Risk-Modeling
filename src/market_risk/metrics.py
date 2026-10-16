from __future__ import annotations

import numpy as np
import pandas as pd


def value_at_risk(losses: pd.Series | np.ndarray, confidence: float) -> float:
    values = np.asarray(losses, dtype=float)
    return float(np.quantile(values, confidence, method="linear"))


def expected_shortfall(losses: pd.Series | np.ndarray, confidence: float) -> float:
    values = np.asarray(losses, dtype=float)
    var = value_at_risk(values, confidence)
    tail = values[values >= var]
    return float(tail.mean())


def risk_summary(
    losses: pd.Series | np.ndarray,
    var_levels: tuple[float, ...] = (0.95, 0.99),
    es_levels: tuple[float, ...] = (0.975, 0.99),
) -> pd.Series:
    values = np.asarray(losses, dtype=float)
    output: dict[str, float] = {"observations": float(len(values))}
    for level in var_levels:
        output[f"VaR_{100 * level:g}"] = value_at_risk(values, level)
    for level in es_levels:
        output[f"ES_{100 * level:g}"] = expected_shortfall(values, level)
    output["maximum_loss"] = float(values.max())
    output["mean_loss"] = float(values.mean())
    output["loss_std"] = float(values.std(ddof=1))
    return pd.Series(output, name="risk_measure")

