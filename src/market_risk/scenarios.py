from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def historical_scenarios(
    current: pd.Series,
    shocks: pd.DataFrame,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Apply joint historical absolute shocks to today's market state."""
    selected = shocks.tail(lookback_days) if lookback_days else shocks
    missing = selected.columns.difference(current.index)
    if len(missing):
        raise ValueError(f"Current state is missing: {missing.tolist()}")
    scenarios = selected.add(current[selected.columns], axis="columns")
    scenarios.index = pd.Index([f"HS_{d:%Y%m%d}" for d in selected.index], name="scenario")
    return scenarios


def _draw_scores(
    covariance: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
    distribution: str,
    student_t_df: float,
) -> np.ndarray:
    n = covariance.shape[0]
    if distribution == "gaussian":
        return rng.multivariate_normal(np.zeros(n), covariance, size=n_scenarios)
    if distribution == "student_t":
        if student_t_df <= 2:
            raise ValueError("student_t_df must exceed 2 for finite covariance")
        normal = rng.multivariate_normal(np.zeros(n), covariance, size=n_scenarios)
        chi2 = rng.chisquare(student_t_df, size=n_scenarios)
        # Scale so the requested covariance equals covariance, not the t scale matrix.
        return normal * np.sqrt((student_t_df - 2.0) / chi2)[:, None]
    raise ValueError("distribution must be 'gaussian' or 'student_t'")


def pca_monte_carlo_scenarios(
    current: pd.Series,
    shocks: pd.DataFrame,
    n_scenarios: int = 10_000,
    n_components: int = 3,
    distribution: str = "gaussian",
    student_t_df: float = 5.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Simulate correlated absolute shocks in PCA space and reconstruct levels."""
    if n_components < 1 or n_components > min(shocks.shape):
        raise ValueError("Invalid number of PCA components")
    x = shocks.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    centered = x - mean
    pca = PCA(n_components=n_components).fit(centered)
    scores = pca.transform(centered)
    covariance = np.cov(scores, rowvar=False)
    covariance = np.atleast_2d(covariance)
    rng = np.random.default_rng(random_seed)
    simulated_scores = _draw_scores(
        covariance, n_scenarios, rng, distribution, student_t_df
    )
    simulated_shocks = simulated_scores @ pca.components_ + mean
    simulated_levels = simulated_shocks + current[shocks.columns].to_numpy()
    scenarios = pd.DataFrame(
        simulated_levels,
        columns=shocks.columns,
        index=pd.Index([f"MC_{i:06d}" for i in range(n_scenarios)], name="scenario"),
    )
    diagnostics = {
        "distribution": distribution,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=shocks.columns,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        ),
    }
    return scenarios, diagnostics

