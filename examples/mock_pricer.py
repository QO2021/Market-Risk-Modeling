"""A nonlinear placeholder pricer. Replace this file with the pricing team's adapter."""

import numpy as np
import pandas as pd


def price_callable_cms(state: pd.Series) -> float:
    rate = float(state["OIS_10Y"])
    cms = float(state["OIS_2Y"])
    vol = float(state["VOL_5Y2Y"])
    spread = float(state.get("ISSUER_SPREAD", 0.0))
    duration_component = -650.0 * (rate - 0.04)
    convexity_component = 18_000.0 * (rate - 0.04) ** 2
    inverse_coupon_component = -240.0 * (cms - 0.035)
    option_component = -90.0 * max(vol - 0.007, 0.0) - 35.0 * np.sqrt(max(vol, 0.0))
    credit_component = -700.0 * spread
    return 100.0 + duration_component + convexity_component + inverse_coupon_component + option_component + credit_component

