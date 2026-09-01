"""
Phase 5: Correlation analysis.

Methods and their appropriate use cases:
────────────────────────────────────────
Pearson:
  Linear relationships, assumes normality.  Misleading for heavy-tailed
  financial returns (inflated by outliers).

Spearman:
  Rank-based; captures monotone (not just linear) relationships.
  Robust to outliers and non-normality.  Preferred for returns.

Kendall τ:
  Also rank-based; more robust than Spearman for small samples.
  Computationally O(n²) – use on subsampled data for n > 10,000.

Distance Correlation (dcor):
  Detects ANY dependence (including non-linear), not just monotone.
  Zero dcor implies independence (unlike Pearson/Spearman).
  Use when you suspect non-linear commodity–equity relationships.

Mutual Information:
  Information-theoretic measure.  Captures all statistical dependence.
  Non-negative; zero iff independent (continuous case).
  Use to rank the strength of any association without parametric assumptions.

Rolling Correlation:
  Time-varying Pearson correlation over a sliding window.
  Use to detect structural breaks and regime changes.

Cross-Correlation Function (CCF):
  Correlation at various lags h: corr(X_t, Y_{t-h}).
  Foundational for lead-lag analysis (Phase 6).
"""

from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

try:
    import dcor
    HAS_DCOR = True
except ImportError:
    HAS_DCOR = False

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


class CorrelationAnalyzer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ── Pairwise correlation matrices ─────────────────────────────────────────

    def pearson(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.corr(method="pearson")

    def spearman(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.corr(method="spearman")

    def kendall(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.corr(method="kendall")

    def distance_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Pairwise distance correlation matrix.
        Requires dcor package (pip install dcor).
        """
        if not HAS_DCOR:
            log.warning("dcor not installed – skipping distance correlation.")
            return pd.DataFrame()
        cols = returns.columns.tolist()
        n = len(cols)
        mat = np.eye(n)
        clean = returns.dropna()
        for i in range(n):
            for j in range(i + 1, n):
                val = dcor.distance_correlation(clean.iloc[:, i], clean.iloc[:, j])
                mat[i, j] = mat[j, i] = val
        return pd.DataFrame(mat, index=cols, columns=cols)

    def mutual_information_matrix(self, returns: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Pairwise mutual information matrix (normalised to [0, 1]).
        Uses k-NN estimator from sklearn.
        """
        clean = returns.dropna()
        cols = clean.columns.tolist()
        n = len(cols)
        mat = np.zeros((n, n))
        for i, col_i in enumerate(cols):
            mi_vals = mutual_info_regression(
                clean.drop(columns=[col_i]),
                clean[col_i],
                n_neighbors=n_neighbors,
                random_state=self.settings.random_seed,
            )
            # Normalize by max to get [0, 1] scale
            mi_max = mi_vals.max() if mi_vals.max() > 0 else 1.0
            other_cols = [c for c in cols if c != col_i]
            for j_idx, col_j in enumerate(other_cols):
                j = cols.index(col_j)
                mat[i, j] = mi_vals[j_idx] / mi_max
        np.fill_diagonal(mat, 1.0)
        return pd.DataFrame(mat, index=cols, columns=cols)

    # ── Rolling correlation ───────────────────────────────────────────────────

    def rolling_pearson(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: int | None = None,
    ) -> pd.Series:
        w = window or self.settings.rolling_medium
        return series_a.rolling(w, min_periods=w // 2).corr(series_b)

    def rolling_correlation_matrix(
        self,
        returns: pd.DataFrame,
        target: str,
        window: int | None = None,
    ) -> pd.DataFrame:
        w = window or self.settings.rolling_medium
        others = [c for c in returns.columns if c != target]
        result = {
            col: returns[col].rolling(w, min_periods=w // 2).corr(returns[target])
            for col in others
        }
        return pd.DataFrame(result)

    # ── Cross-correlation function ────────────────────────────────────────────

    def cross_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        max_lag: int | None = None,
    ) -> pd.Series:
        """
        CCF: corr(A_t, B_{t-h}) for h in [-max_lag, +max_lag].
        Positive h → B leads A (or equivalently A lags B).
        """
        max_lag = max_lag or self.settings.max_lag
        a = series_a.dropna()
        b = series_b.dropna()
        idx = a.index.intersection(b.index)
        a, b = a[idx], b[idx]
        a_z = (a - a.mean()) / a.std()
        b_z = (b - b.mean()) / b.std()
        ccf_vals = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = a_z.iloc[-lag:].corr(b_z.iloc[:lag])
            elif lag == 0:
                corr = a_z.corr(b_z)
            else:
                corr = a_z.iloc[:-lag].corr(b_z.iloc[lag:])
            ccf_vals[lag] = corr
        return pd.Series(ccf_vals, name=f"CCF({series_a.name},{series_b.name})")

    # ── Significance tests ────────────────────────────────────────────────────

    def correlation_pvalue(
        self,
        r: float,
        n: int,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> float:
        """Two-tailed p-value for a correlation coefficient."""
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
        return 2 * stats.t.sf(abs(t_stat), df=n - 2)

    def significant_correlations(
        self,
        returns: pd.DataFrame,
        method: str = "spearman",
        min_abs_corr: float = 0.1,
    ) -> pd.DataFrame:
        """Return pairwise correlations that are statistically significant."""
        corr = returns.corr(method=method)
        n = len(returns.dropna())
        rows = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) < min_abs_corr:
                    continue
                p = self.correlation_pvalue(r, n)
                if p < self.settings.significance_level:
                    rows.append({
                        "asset_a": cols[i],
                        "asset_b": cols[j],
                        "method": method,
                        "correlation": round(r, 4),
                        "pvalue": round(p, 6),
                    })
        return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)
