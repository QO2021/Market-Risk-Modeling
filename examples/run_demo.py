from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_risk.data import absolute_shocks
from market_risk.metrics import risk_summary
from market_risk.scenarios import historical_scenarios, pca_monte_carlo_scenarios
from market_risk.stress import scenario_matrix, standard_stress_scenarios, run_stress_tests
from market_risk.valuation import revalue_scenarios
from mock_pricer import price_callable_cms


def synthetic_history(seed: int = 42, days: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = ["OIS_1Y", "OIS_2Y", "OIS_5Y", "OIS_10Y", "OIS_30Y", "VOL_1Y2Y", "VOL_2Y2Y", "VOL_5Y2Y", "VOL_10Y2Y", "ISSUER_SPREAD"]
    base = np.array([.038, .036, .037, .040, .043, .0065, .0068, .0072, .0075, .012])
    common = rng.standard_t(6, size=(days, 3))
    loadings = rng.normal(size=(3, len(names)))
    shocks = common @ loadings * np.array([.00012] * 5 + [.00008] * 4 + [.00015])
    levels = base + np.cumsum(shocks, axis=0)
    return pd.DataFrame(levels, index=pd.bdate_range("2023-01-02", periods=days), columns=names)


def main() -> None:
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    history = synthetic_history()
    current = history.iloc[-1]
    shocks = absolute_shocks(history)

    results = []
    start = time.perf_counter()
    hs = historical_scenarios(current, shocks, lookback_days=500)
    hs_pnl = revalue_scenarios(current, hs, price_callable_cms)
    summary = risk_summary(hs_pnl["loss"])
    summary["runtime_seconds"] = time.perf_counter() - start
    summary.name = "Historical Simulation"
    results.append(summary)

    for distribution in ("gaussian", "student_t"):
        start = time.perf_counter()
        mc, diagnostics = pca_monte_carlo_scenarios(current, shocks.tail(500), distribution=distribution, n_scenarios=10_000)
        mc_pnl = revalue_scenarios(current, mc, price_callable_cms)
        summary = risk_summary(mc_pnl["loss"])
        summary["runtime_seconds"] = time.perf_counter() - start
        summary.name = f"PCA MC {distribution}"
        results.append(summary)
        diagnostics["loadings"].to_csv(output / f"pca_loadings_{distribution}.csv")

    pd.DataFrame(results).to_csv(output / "model_comparison.csv")
    rate_cols = [c for c in history if c.startswith("OIS_")]
    vol_cols = [c for c in history if c.startswith("VOL_")]
    stresses = standard_stress_scenarios(rate_cols, vol_cols, ["ISSUER_SPREAD"])
    run_stress_tests(current, price_callable_cms, stresses).to_csv(output / "stress_results.csv")
    scenario_matrix(current, price_callable_cms, rate_cols, vol_cols).to_csv(output / "scenario_matrix.csv")
    print(pd.DataFrame(results).round(6))
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()

