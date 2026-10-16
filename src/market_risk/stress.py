from __future__ import annotations

from collections.abc import Mapping
import pandas as pd

from .valuation import Pricer


def standard_stress_scenarios(
    rate_columns: list[str],
    vol_columns: list[str],
    spread_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Starter shocks in decimal rate units (100 bp = 0.01)."""
    spread_columns = spread_columns or []
    def shock(rate_bp=0.0, vol_bp=0.0, spread_bp=0.0):
        d = {c: rate_bp / 10_000 for c in rate_columns}
        d.update({c: vol_bp / 10_000 for c in vol_columns})
        d.update({c: spread_bp / 10_000 for c in spread_columns})
        return d
    scenarios = {
        "rates_up_50bp": shock(rate_bp=50),
        "rates_up_150bp": shock(rate_bp=150),
        "rates_down_100bp": shock(rate_bp=-100),
        "vol_up_25bp": shock(vol_bp=25),
        "credit_up_100bp": shock(spread_bp=100),
        "combined_crisis": shock(rate_bp=150, vol_bp=30, spread_bp=150),
    }
    # Nonparallel examples use tenor-name matching and leave unmatched nodes unchanged.
    steepener = {c: 0.0 for c in rate_columns + vol_columns + spread_columns}
    flattener = steepener.copy()
    for c in rate_columns:
        if "2Y" in c:
            steepener[c], flattener[c] = 50 / 10_000, 125 / 10_000
        elif "10Y" in c or "30Y" in c:
            steepener[c], flattener[c] = 125 / 10_000, 50 / 10_000
    scenarios["bear_steepener"] = steepener
    scenarios["bear_flattener"] = flattener
    return scenarios


def run_stress_tests(
    current: pd.Series,
    pricer: Pricer,
    scenarios: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    base_value = float(pricer(current.copy()))
    rows = []
    for name, shocks in scenarios.items():
        state = current.copy()
        for factor, change in shocks.items():
            if factor not in state.index:
                raise ValueError(f"Stress factor not present in current state: {factor}")
            state.loc[factor] += change
        value = float(pricer(state))
        rows.append({"scenario": name, "scenario_value": value, "pnl": value - base_value, "loss": base_value - value})
    return pd.DataFrame(rows).set_index("scenario")


def scenario_matrix(
    current: pd.Series,
    pricer: Pricer,
    rate_columns: list[str],
    vol_columns: list[str],
    rate_shocks_bp: tuple[int, ...] = (-150, -75, 0, 75, 150),
    vol_shocks_bp: tuple[int, ...] = (-20, 0, 20, 40),
) -> pd.DataFrame:
    base_value = float(pricer(current.copy()))
    matrix = pd.DataFrame(index=rate_shocks_bp, columns=vol_shocks_bp, dtype=float)
    for rate_bp in rate_shocks_bp:
        for vol_bp in vol_shocks_bp:
            state = current.copy()
            state.loc[rate_columns] += rate_bp / 10_000
            state.loc[vol_columns] += vol_bp / 10_000
            matrix.loc[rate_bp, vol_bp] = base_value - float(pricer(state))
    matrix.index.name = "rate_shock_bp"
    matrix.columns.name = "vol_shock_bp"
    return matrix

