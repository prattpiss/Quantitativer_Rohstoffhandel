"""
Phase 2: Return calculations.

Design decision:
- Logarithmic returns are the default because they are time-additive,
  approximately normally distributed, and better suited for statistical
  inference (stationarity tests, VAR models, regression).
- Simple returns are provided for performance attribution and strategies
  where compounding must be exact (e.g. portfolio P&L).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


class ReturnCalculator:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def log_returns(self, prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Compute log-returns: r_t = ln(P_t / P_{t-periods})."""
        lr = np.log(prices / prices.shift(periods))
        lr = lr.replace([np.inf, -np.inf], np.nan)
        log.debug("Computed log-returns (periods=%d) for %d series.", periods, prices.shape[1])
        return lr

    def simple_returns(self, prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Compute simple returns: r_t = (P_t - P_{t-periods}) / P_{t-periods}."""
        sr = prices.pct_change(periods=periods)
        sr = sr.replace([np.inf, -np.inf], np.nan)
        return sr

    def excess_returns(
        self,
        returns: pd.DataFrame,
        risk_free: pd.Series,
    ) -> pd.DataFrame:
        """Subtract daily risk-free rate (aligned by index)."""
        rf_aligned = risk_free.reindex(returns.index).ffill()
        return returns.subtract(rf_aligned, axis=0)

    def rolling_returns(
        self,
        prices: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """Rolling cumulative log-return over a window."""
        lr = self.log_returns(prices)
        return lr.rolling(window).sum()

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return log or simple returns according to project settings."""
        if self.settings.return_type == "log":
            return self.log_returns(prices)
        return self.simple_returns(prices)
