from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def exception_indicators(realized_losses: pd.Series, var_forecasts: pd.Series) -> pd.Series:
    aligned = pd.concat([realized_losses, var_forecasts], axis=1, join="inner").dropna()
    return (aligned.iloc[:, 0] > aligned.iloc[:, 1]).astype(int).rename("exception")


def kupiec_test(exceptions: pd.Series, confidence: float) -> dict[str, float]:
    x = int(exceptions.sum())
    n = int(len(exceptions))
    if n == 0:
        raise ValueError("No observations")
    p = 1.0 - confidence
    phat = x / n
    eps = np.finfo(float).eps
    phat_clip = np.clip(phat, eps, 1 - eps)
    null_ll = (n - x) * np.log(1 - p) + x * np.log(p)
    alt_ll = (n - x) * np.log(1 - phat_clip) + x * np.log(phat_clip)
    lr = max(0.0, -2.0 * (null_ll - alt_ll))
    return {
        "observations": n,
        "exceptions": x,
        "exception_rate": phat,
        "expected_rate": p,
        "lr_uc": lr,
        "p_value": float(chi2.sf(lr, 1)),
    }


def christoffersen_independence_test(exceptions: pd.Series) -> dict[str, float]:
    values = np.asarray(exceptions, dtype=int)
    if len(values) < 2:
        raise ValueError("At least two observations are required")
    n00 = int(np.sum((values[:-1] == 0) & (values[1:] == 0)))
    n01 = int(np.sum((values[:-1] == 0) & (values[1:] == 1)))
    n10 = int(np.sum((values[:-1] == 1) & (values[1:] == 0)))
    n11 = int(np.sum((values[:-1] == 1) & (values[1:] == 1)))
    eps = np.finfo(float).eps
    pi0 = np.clip(n01 / max(n00 + n01, 1), eps, 1 - eps)
    pi1 = np.clip(n11 / max(n10 + n11, 1), eps, 1 - eps)
    pi = np.clip((n01 + n11) / max(n00 + n01 + n10 + n11, 1), eps, 1 - eps)
    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "lr_ind": lr, "p_value": float(chi2.sf(lr, 1)),
    }


def es_diagnostics(realized_losses: pd.Series, var_forecasts: pd.Series, es_forecasts: pd.Series) -> dict[str, float]:
    data = pd.concat([realized_losses, var_forecasts, es_forecasts], axis=1, join="inner").dropna()
    data.columns = ["loss", "var", "es"]
    breaches = data[data["loss"] > data["var"]]
    return {
        "observations": float(len(data)),
        "tail_observations": float(len(breaches)),
        "mean_tail_loss": float(breaches["loss"].mean()) if len(breaches) else np.nan,
        "mean_es_forecast_on_breach": float(breaches["es"].mean()) if len(breaches) else np.nan,
        "mean_tail_minus_es": float((breaches["loss"] - breaches["es"]).mean()) if len(breaches) else np.nan,
    }

