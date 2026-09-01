"""
Phase 2: Missing value imputation and outlier treatment.

Design decisions:
- Missing values in price series: forward-fill (max 5 days), then
  interpolate linearly for short interior gaps.  Long gaps (> 5 days)
  are left as NaN and flagged in logs; they are dropped before modelling.
- Outlier detection: we use z-score |z| > threshold on log-returns
  (NOT on levels) because return distributions are stationary.
- Winsorisation is applied at the return level when explicitly requested.
  We do NOT winsorise prices because that would distort compounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


class DataCleaner:
    """
    Missing value imputation and outlier treatment for financial price series.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ── Missing values ────────────────────────────────────────────────────────

    def fill_missing(
        self,
        prices: pd.DataFrame,
        max_ffill: int = 5,
        interpolate: bool = True,
    ) -> pd.DataFrame:
        """
        Fill missing prices.

        Strategy:
          1. Forward-fill up to max_ffill days (handles weekends/holidays
             already excluded by the aligner, but covers genuine suspension).
          2. Linear interpolation for interior gaps ≤ max_ffill.
          3. Remaining NaN: logged and left as-is for the caller to decide.
        """
        n_before = prices.isna().sum().sum()
        filled = prices.ffill(limit=max_ffill)
        if interpolate:
            filled = filled.interpolate(method="linear", limit=max_ffill, limit_area="inside")
        n_after = filled.isna().sum().sum()
        log.info("Missing values: %d -> %d (filled %d).", n_before, n_after, n_before - n_after)
        for col in filled.columns[filled.isna().any()]:
            n = filled[col].isna().sum()
            log.warning("Column '%s' still has %d NaN after imputation.", col, n)
        return filled

    # ── Outlier detection ─────────────────────────────────────────────────────

    def detect_outliers(
        self,
        returns: pd.DataFrame,
        z_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """
        Return a boolean mask (True = outlier) on log-return series.

        z-score is computed relative to the rolling 252-day window
        (not the full-sample mean) so that regime changes are not
        mis-classified as outliers.
        """
        roll_mean = returns.rolling(252, min_periods=60).mean()
        roll_std = returns.rolling(252, min_periods=60).std()
        z = (returns - roll_mean) / roll_std
        mask = z.abs() > z_threshold
        n = mask.sum().sum()
        log.info("Detected %d outliers (|z| > %.1f) across %d series.", n, z_threshold, returns.shape[1])
        return mask

    def winsorise(
        self,
        returns: pd.DataFrame,
        quantile: float | None = None,
    ) -> pd.DataFrame:
        """
        Winsorise return series at the given two-sided quantile.

        When to use: transfer-entropy and rank-based methods are robust
        to outliers; winsorisation is most useful before OLS regressions.
        It should NOT be applied before computing summary statistics because
        it changes the distribution shape.
        """
        q = quantile if quantile is not None else self.settings.winsorise_quantile
        lower = returns.quantile(q)
        upper = returns.quantile(1 - q)
        winsorised = returns.clip(lower=lower, upper=upper, axis=1)
        n_clipped = (returns != winsorised).sum().sum()
        log.info("Winsorised %d observations (q=%.4f).", n_clipped, q)
        return winsorised
