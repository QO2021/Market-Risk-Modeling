import numpy as np
import pandas as pd

from market_risk.backtesting import kupiec_test
from market_risk.metrics import expected_shortfall, value_at_risk
from market_risk.stress import run_stress_tests


def test_var_and_es_ordering():
    losses = np.arange(1.0, 101.0)
    var = value_at_risk(losses, 0.95)
    es = expected_shortfall(losses, 0.95)
    assert es >= var


def test_kupiec_counts_exceptions():
    result = kupiec_test(pd.Series([0, 0, 1, 0, 1]), 0.95)
    assert result["exceptions"] == 2
    assert result["observations"] == 5


def test_stress_loss_sign():
    current = pd.Series({"x": 1.0})
    pricer = lambda state: 100.0 - 10.0 * state["x"]
    result = run_stress_tests(current, pricer, {"up": {"x": 0.1}})
    assert np.isclose(result.loc["up", "loss"], 1.0)

