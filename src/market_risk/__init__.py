"""Market-risk tools for callable CMS notes."""

from .metrics import risk_summary
from .scenarios import historical_scenarios, pca_monte_carlo_scenarios
from .stress import standard_stress_scenarios, run_stress_tests

__all__ = [
    "historical_scenarios",
    "pca_monte_carlo_scenarios",
    "risk_summary",
    "standard_stress_scenarios",
    "run_stress_tests",
]

