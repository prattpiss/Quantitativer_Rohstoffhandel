"""
Phase 2: Standardisation and normalisation of return series.

Methods provided:
- Z-score standardisation (mean=0, std=1) – for factor models, PCA, regression.
- Min-max normalisation – rarely used for returns; included for completeness.
- Rolling Z-score – for time-varying standardisation in online analyses.

Note: standardisation is applied to RETURNS, never to price levels,
because price levels are non-stationary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


class Normalizer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def z_score(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Full-sample z-score standardisation."""
        scaler = StandardScaler()
        arr = scaler.fit_transform(returns.dropna())
        return pd.DataFrame(arr, index=returns.dropna().index, columns=returns.columns)

    def rolling_z_score(
        self,
        returns: pd.DataFrame,
        window: int | None = None,
    ) -> pd.DataFrame:
        """
        Rolling z-score: (r_t - μ_{t,window}) / σ_{t,window}.

        Avoids look-ahead bias; suitable for online/walk-forward analysis.
        """
        w = window or self.settings.rolling_long
        mu = returns.rolling(w, min_periods=w // 2).mean()
        sigma = returns.rolling(w, min_periods=w // 2).std()
        return (returns - mu) / sigma

    def min_max(
        self,
        returns: pd.DataFrame,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> pd.DataFrame:
        """Min-max normalisation. Use with caution for returns."""
        scaler = MinMaxScaler(feature_range=feature_range)
        clean = returns.dropna()
        arr = scaler.fit_transform(clean)
        return pd.DataFrame(arr, index=clean.index, columns=returns.columns)
