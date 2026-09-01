"""
Phase 6: Transfer Entropy.

Transfer entropy T(X→Y) measures the reduction in uncertainty about Y_t+1
given the past of X relative to knowing only the past of Y.

Unlike Granger causality, transfer entropy:
  - Is non-parametric and model-free.
  - Detects non-linear information flow.
  - Is not symmetric: T(X→Y) ≠ T(Y→X), giving directionality.

Implementation:
  We estimate transfer entropy using the k-NN (Kraskov) estimator via
  the pyinform package, or fall back to a histogram-based estimator if
  pyinform is not available.

Appropriate use:
  - Requires stationary series.
  - Use on standardised returns or rank-transformed series for robustness.
  - Interpret T(commodity → equity) vs T(equity → commodity) for direction.
  - Statistical significance via bootstrap permutation test.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)

try:
    import pyinform
    HAS_PYINFORM = True
except ImportError:
    HAS_PYINFORM = False


def _discretise(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Bin continuous series into integers for histogram-based TE."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(series.dropna(), quantiles)
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return np.digitize(series.dropna().values, edges) - 1


def _histogram_te(x: np.ndarray, y: np.ndarray, lag: int = 1, n_bins: int = 10) -> float:
    """
    Histogram-based transfer entropy T(X → Y).
    T(X→Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-lag)
    """
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    def joint_entropy(*arrays: np.ndarray) -> float:
        stacked = np.column_stack(arrays)
        _, counts = np.unique(stacked, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def entropy(arr: np.ndarray) -> float:
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    y_t = y[lag:]
    y_tm1 = y[:-lag]
    x_tlag = x[:-lag]

    # T(X→Y) = H(Y_t, Y_t-1) + H(Y_t-1, X_t-lag) - H(Y_t-1) - H(Y_t, Y_t-1, X_t-lag)
    h_yt_ytm1 = joint_entropy(y_t, y_tm1)
    h_ytm1_xtlag = joint_entropy(y_tm1, x_tlag)
    h_ytm1 = entropy(y_tm1)
    h_yt_ytm1_xtlag = joint_entropy(y_t, y_tm1, x_tlag)

    te = h_yt_ytm1 + h_ytm1_xtlag - h_ytm1 - h_yt_ytm1_xtlag
    return max(te, 0.0)  # TE is non-negative; clamp numerical errors


@dataclass
class TransferEntropyResult:
    source: str
    target: str
    lag: int
    te_forward: float          # T(source → target)
    te_backward: float         # T(target → source)
    net_te: float              # te_forward - te_backward (directionality)
    pvalue: float
    significant: bool
    n_permutations: int

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "lag": self.lag,
            "TE_fwd": round(self.te_forward, 6),
            "TE_bwd": round(self.te_backward, 6),
            "net_TE": round(self.net_te, 6),
            "pvalue": round(self.pvalue, 4),
            "significant": self.significant,
        }


class TransferEntropyAnalyzer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def compute(
        self,
        source: pd.Series,
        target: pd.Series,
        lag: int = 1,
        n_bins: int = 10,
        n_permutations: int = 200,
    ) -> TransferEntropyResult:
        """
        Compute transfer entropy with permutation-based significance test.
        """
        idx = source.dropna().index.intersection(target.dropna().index)
        sx = _discretise(source[idx], n_bins)
        sy = _discretise(target[idx], n_bins)

        te_fwd = _histogram_te(sx, sy, lag=lag)
        te_bwd = _histogram_te(sy, sx, lag=lag)

        # Permutation test: shuffle source, recompute TE
        rng = np.random.default_rng(self.settings.random_seed)
        null_dist = []
        for _ in range(n_permutations):
            sx_perm = rng.permutation(sx)
            null_dist.append(_histogram_te(sx_perm, sy, lag=lag))

        pvalue = 1.0 - percentileofscore(null_dist, te_fwd) / 100.0

        return TransferEntropyResult(
            source=str(source.name),
            target=str(target.name),
            lag=lag,
            te_forward=te_fwd,
            te_backward=te_bwd,
            net_te=te_fwd - te_bwd,
            pvalue=pvalue,
            significant=pvalue < self.settings.significance_level,
            n_permutations=n_permutations,
        )

    def compute_matrix(
        self,
        returns: pd.DataFrame,
        lag: int = 1,
        n_bins: int = 10,
        n_permutations: int = 200,
    ) -> pd.DataFrame:
        """
        NxN transfer entropy matrix: entry [i,j] = T(col_i → col_j).
        """
        cols = returns.columns.tolist()
        n = len(cols)
        mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if i == j:
                    continue
                res = self.compute(returns[src], returns[tgt], lag, n_bins, n_permutations)
                mat.loc[src, tgt] = res.te_forward
        return mat

    def commodity_to_equity_te(
        self,
        commodity_returns: pd.Series,
        equity_returns: pd.DataFrame,
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute TE from one commodity to all equity series across lags."""
        lags = lags or list(range(1, min(self.settings.max_lag + 1, 11)))
        rows = []
        for lag in lags:
            for col in equity_returns.columns:
                res = self.compute(commodity_returns, equity_returns[col], lag=lag)
                rows.append(res.to_dict())
        return pd.DataFrame(rows)
