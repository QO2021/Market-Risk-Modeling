from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RiskConfig:
    holding_period_days: int = 1
    lookback_days: int = 500
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    es_levels: tuple[float, ...] = (0.975, 0.99)
    mc_scenarios: int = 10_000
    pca_components: int = 3
    student_t_df: float = 5.0
    random_seed: int = 42
    rate_columns: tuple[str, ...] = field(default_factory=tuple)
    vol_columns: tuple[str, ...] = field(default_factory=tuple)
    spread_columns: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        for key in ("confidence_levels", "es_levels", "rate_columns", "vol_columns", "spread_columns"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

