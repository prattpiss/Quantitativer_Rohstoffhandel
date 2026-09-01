"""
Phase 2: Seasonality decomposition.

Methods and when to use them:
─────────────────────────────
STL (Seasonal-Trend decomposition using LOESS)
  Best for: high-frequency data with stable but possibly time-varying
  seasonality (e.g. natural gas prices with winter peaks).
  Use when: the seasonal pattern is roughly periodic but can evolve.

X-13ARIMA-SEATS
  Best for: monthly macroeconomic series (CPI, PPI, Industrial Production).
  Use when: you need a Census-Bureau-grade seasonal adjustment.
  Requires: the x13as binary (statsmodels can call it if installed).

Kalman Filter (state-space decomposition)
  Best for: unobserved components models; handles irregular data,
  missing values, and slowly changing trends.
  Use when: the trend/cycle separation is more important than seasonality.

Wavelets (discrete wavelet transform)
  Best for: multi-resolution analysis; decomposing signals into
  frequency bands without assuming fixed periods.
  Use when: you want to isolate short/medium/long-term components
  simultaneously or study cross-frequency correlation.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult
import pywt

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    method: str


class SeasonalDecomposer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def stl(
        self,
        series: pd.Series,
        period: int = 252,     # annual cycle on business days
        robust: bool = True,
    ) -> DecompositionResult:
        """
        STL decomposition.

        period=252  → annual seasonality for daily data
        period=21   → monthly cycle
        robust=True → down-weights outliers (recommended for commodity data)
        """
        series_clean = series.dropna()
        stl = STL(series_clean, period=period, robust=robust)
        res: DecomposeResult = stl.fit()
        return DecompositionResult(
            trend=pd.Series(res.trend, index=series_clean.index),
            seasonal=pd.Series(res.seasonal, index=series_clean.index),
            residual=pd.Series(res.resid, index=series_clean.index),
            method="STL",
        )

    def x13(self, series: pd.Series) -> DecompositionResult:
        """
        X-13ARIMA-SEATS decomposition via statsmodels.

        Requires the x13as binary to be on PATH.  If unavailable, falls
        back to STL with a monthly period.
        """
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            res = x13_arima_analysis(series.dropna(), x12path=None)
            trend = pd.Series(res.trend.values, index=series.dropna().index)
            seasonal = pd.Series(res.seasonal.values, index=series.dropna().index)
            residual = pd.Series(res.irregular.values, index=series.dropna().index)
            return DecompositionResult(trend, seasonal, residual, "X13")
        except Exception as exc:
            log.warning("X-13 failed (%s) – falling back to STL(period=21).", exc)
            return self.stl(series, period=21)

    def kalman(self, series: pd.Series) -> DecompositionResult:
        """
        Unobserved-components model via statsmodels (level + trend + irregular).

        This is a local-level model (random walk + noise), which provides a
        smooth trend without assuming any fixed seasonal period.
        """
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        series_clean = series.dropna()
        model = UnobservedComponents(
            series_clean,
            level="local linear trend",
            irregular=True,
        )
        result = model.fit(disp=False)
        trend = result.level.filtered[0]
        irregular = series_clean.values - trend
        return DecompositionResult(
            trend=pd.Series(trend, index=series_clean.index),
            seasonal=pd.Series(np.zeros(len(series_clean)), index=series_clean.index),
            residual=pd.Series(irregular, index=series_clean.index),
            method="Kalman",
        )

    def wavelet(
        self,
        series: pd.Series,
        wavelet: str = "db4",
        level: int = 4,
    ) -> dict[str, pd.Series]:
        """
        Discrete Wavelet Transform decomposition.

        Returns a dict of detail coefficients (D1…D{level}) and
        approximation (A{level}), each reconstructed to original length.
        These represent progressively longer-term frequency components.

        Interpretation:
          D1 = highest frequency (noise / daily noise)
          D2 = weekly component
          D3 = monthly component
          D4 = quarterly component
          A4 = long-term trend / approximation
        """
        series_clean = series.dropna().values
        coeffs = pywt.wavedec(series_clean, wavelet=wavelet, level=level)
        components: dict[str, pd.Series] = {}
        idx = series.dropna().index
        for i, coeff in enumerate(coeffs):
            name = f"A{level}" if i == 0 else f"D{level - i + 1}"
            # Reconstruct the component at original scale
            template = [np.zeros_like(c) for c in coeffs]
            template[i] = coeff
            reconstructed = pywt.waverec(template, wavelet=wavelet)
            # Trim to original length (waverec may add one sample)
            reconstructed = reconstructed[: len(idx)]
            components[name] = pd.Series(reconstructed, index=idx)
        return components
