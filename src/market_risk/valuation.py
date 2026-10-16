from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pandas as pd

Pricer = Callable[[pd.Series], float]


def revalue_scenarios(
    current: pd.Series,
    scenarios: pd.DataFrame,
    pricer: Pricer,
) -> pd.DataFrame:
    """Full revaluation under each scenario; positive loss means value decreased."""
    base_value = float(pricer(current.copy()))
    values = np.empty(len(scenarios), dtype=float)
    for i, (_, row) in enumerate(scenarios.iterrows()):
        shocked = current.copy()
        shocked.loc[row.index] = row
        values[i] = float(pricer(shocked))
    result = pd.DataFrame(index=scenarios.index)
    result["scenario_value"] = values
    result["pnl"] = values - base_value
    result["loss"] = base_value - values
    result.attrs["base_value"] = base_value
    return result

