"""
Phase 11: Bootstrap and Jackknife confidence intervals for lag estimates.

Why bootstrap for lag estimates:
  The sampling distribution of the optimal CCF lag or the Granger minimum
  significant lag is non-standard (discrete, bounded) and cannot be
  approximated by a t-distribution.  The block bootstrap preserves
  the temporal dependence structure of financial returns.

Methods:
  - IID bootstrap: assumes i.i.d. residuals (anti-conservative for returns).
  - Circular block bootstrap: preserves autocorrelation, recommended.
  - Stationary bootstrap (Politis & Romano): random block lengths.
  - Jackknife: delete-one; useful when bootstrap is too slow.

Monte Carlo:
  Used to assess robustness of the lead-lag signal under data-generating
  process uncertainty (simulate from a VAR and check if estimated lag
  recovers the true lag with correct coverage).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BootstrapResult:
    statistic_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic_name,
            "estimate": round(self.point_estimate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
        }


class BootstrapAnalyzer:
    """Phase 11: Block bootstrap for lead-lag estimation uncertainty."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(settings.random_seed)

    # ── Block bootstrap ───────────────────────────────────────────────────────

    def _circular_block_resample(
        self,
        data: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Circular block bootstrap resample of a 1-D array."""
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))
        starts = self.rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_size)) % n for start in starts]
        )[:n]
        return data[indices]

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        n_bootstrap: int | None = None,
        block_size: int = 10,
        ci_level: float = 0.95,
        statistic_name: str = "statistic",
    ) -> BootstrapResult:
        """
        Circular block bootstrap confidence interval.

        block_size ≈ sqrt(n) is a common heuristic for financial returns.
        For daily returns over 252 days, block_size ≈ 16.
        """
        n_boot = n_bootstrap or self.settings.n_bootstrap
        point = statistic_fn(data)
        boot_dist = np.array([
            statistic_fn(self._circular_block_resample(data, block_size))
            for _ in range(n_boot)
        ])
        alpha = 1 - ci_level
        ci_lower = float(np.percentile(boot_dist, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            statistic_name=statistic_name,
            point_estimate=float(point),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            n_bootstrap=n_boot,
            bootstrap_distribution=boot_dist,
        )

    def bootstrap_lag_ci(
        self,
        source: pd.Series,
        target: pd.Series,
        lag_fn: Callable[[pd.Series, pd.Series], float],
        n_bootstrap: int | None = None,
        block_size: int = 10,
        ci_level: float = 0.95,
    ) -> BootstrapResult:
        """
        Confidence interval for an estimated lead-lag statistic.
        lag_fn(source, target) → scalar lag estimate.
        """
        idx = source.dropna().index.intersection(target.dropna().index)
        s = source[idx].values
        t = target[idx].values
        n_boot = n_bootstrap or self.settings.n_bootstrap

        point_lag = lag_fn(source[idx], target[idx])
        boot_lags = []
        for _ in range(n_boot):
            perm_idx = self._circular_block_resample(np.arange(len(s)), block_size)
            s_b = pd.Series(s[perm_idx], index=idx[:len(perm_idx)], name=source.name)
            t_b = pd.Series(t[perm_idx], index=idx[:len(perm_idx)], name=target.name)
            try:
                boot_lags.append(lag_fn(s_b, t_b))
            except Exception:
                boot_lags.append(np.nan)

        boot_arr = np.array([x for x in boot_lags if np.isfinite(x)])
        alpha = 1 - ci_level
        ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            statistic_name="optimal_lag",
            point_estimate=float(point_lag),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            n_bootstrap=len(boot_arr),
            bootstrap_distribution=boot_arr,
        )

    # ── Jackknife ─────────────────────────────────────────────────────────────

    def jackknife_se(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
    ) -> float:
        """Jackknife standard error estimate."""
        n = len(data)
        jack_stats = np.array([
            statistic_fn(np.delete(data, i))
            for i in range(n)
        ])
        return float(np.sqrt((n - 1) / n * np.sum((jack_stats - jack_stats.mean()) ** 2)))
