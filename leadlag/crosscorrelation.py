"""
Phase 6: Cross-Correlation Function (CCF) analysis for lead-lag detection.

The CCF corr(X_t, Y_{t-h}) is the primary screening tool:
  - Quick to compute for all pairs.
  - Identifies the sign and approximate timing of the lead-lag relationship.
  - Provides starting points for VAR lag selection and Granger tests.

Limitation: CCF assumes linearity and is symmetric by definition;
it cannot distinguish true causality from contemporaneous correlation.
Use together with Granger/transfer entropy for robustness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class LeadLagResult:
    source: str
    target: str
    optimal_lag: int            # lag h at which CCF peaks (positive → source leads)
    peak_ccf: float             # CCF value at optimal lag
    ccf_series: pd.Series       # full CCF across lags
    pvalue_at_peak: float
    significant: bool
    # Information-delay metrics (filled from VAR IRFs if available)
    time_to_25pct: float = field(default=float("nan"))
    time_to_50pct: float = field(default=float("nan"))
    time_to_75pct: float = field(default=float("nan"))
    time_to_90pct: float = field(default=float("nan"))
    time_to_95pct: float = field(default=float("nan"))

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "optimal_lag": self.optimal_lag,
            "peak_ccf": round(self.peak_ccf, 4),
            "pvalue": round(self.pvalue_at_peak, 6),
            "significant": self.significant,
            "time_to_25pct": self.time_to_25pct,
            "time_to_50pct": self.time_to_50pct,
            "time_to_75pct": self.time_to_75pct,
            "time_to_90pct": self.time_to_90pct,
            "time_to_95pct": self.time_to_95pct,
        }


class CrossCorrelationAnalyzer:
    """Phase 6: Systematic lead-lag screening via CCF."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _ccf_pvalue(self, r: float, n: int) -> float:
        """Approximate p-value using the Fisher z-transform."""
        if abs(r) >= 1.0:
            return 0.0
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        return 2 * stats.norm.sf(abs(z) / se)

    def compute(
        self,
        source: pd.Series,
        target: pd.Series,
        max_lag: int | None = None,
        search_positive_lags_only: bool = True,
    ) -> LeadLagResult:
        """
        Compute CCF between source and target.

        Positive lag h: source at time t leads target at time t+h.
        search_positive_lags_only: if True, only search h >= 0
        (i.e. test whether source leads target, not the reverse).
        """
        max_lag = max_lag or self.settings.max_lag
        idx = source.dropna().index.intersection(target.dropna().index)
        n = len(idx)
        s = (source[idx] - source[idx].mean()) / (source[idx].std() + 1e-12)
        t = (target[idx] - target[idx].mean()) / (target[idx].std() + 1e-12)

        lag_range = range(0, max_lag + 1) if search_positive_lags_only else range(-max_lag, max_lag + 1)
        ccf_vals: dict[int, float] = {}
        for lag in lag_range:
            if lag == 0:
                r = float(np.corrcoef(s, t)[0, 1])
            elif lag > 0:
                r = float(np.corrcoef(s.iloc[:-lag], t.iloc[lag:])[0, 1])
            else:
                r = float(np.corrcoef(s.iloc[-lag:], t.iloc[:lag])[0, 1])
            ccf_vals[lag] = r if np.isfinite(r) else 0.0

        ccf_series = pd.Series(ccf_vals, name=f"CCF({source.name},{target.name})")
        optimal_lag = int(ccf_series.abs().idxmax())
        peak_ccf = ccf_series[optimal_lag]
        pval = self._ccf_pvalue(peak_ccf, n)

        return LeadLagResult(
            source=str(source.name),
            target=str(target.name),
            optimal_lag=optimal_lag,
            peak_ccf=peak_ccf,
            ccf_series=ccf_series,
            pvalue_at_peak=pval,
            significant=pval < self.settings.significance_level,
        )

    def screen_commodity_to_equities(
        self,
        commodity_returns: pd.DataFrame,
        equity_returns: pd.DataFrame,
        max_lag: int | None = None,
    ) -> pd.DataFrame:
        """
        All commodity × equity pairs.
        Returns a ranked DataFrame of lead-lag results.
        """
        rows = []
        for comm in commodity_returns.columns:
            for eq in equity_returns.columns:
                result = self.compute(
                    commodity_returns[comm].rename(comm),
                    equity_returns[eq].rename(eq),
                    max_lag=max_lag,
                )
                rows.append(result.to_dict())
        df = pd.DataFrame(rows)
        if df.empty or "optimal_lag" not in df.columns:
            return df
        return df.sort_values("optimal_lag")

    def rolling_lead_lag(
        self,
        source: pd.Series,
        target: pd.Series,
        window: int | None = None,
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """
        Compute optimal lead-lag in a rolling window.
        Returns a Series of optimal lags over time.
        """
        w = window or self.settings.rolling_long
        idx = source.dropna().index.intersection(target.dropna().index)
        s = source[idx]
        t = target[idx]
        optimal_lags = []
        dates = []
        for i in range(w, len(idx) + 1):
            res = self.compute(
                s.iloc[i - w : i].rename(source.name),
                t.iloc[i - w : i].rename(target.name),
                max_lag=max_lag,
            )
            optimal_lags.append(res.optimal_lag)
            dates.append(idx[i - 1])
        return pd.Series(optimal_lags, index=pd.DatetimeIndex(dates), name=f"lag({source.name},{target.name})")
