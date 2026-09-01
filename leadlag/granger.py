"""
Phase 6: Granger Causality Testing.

Granger causality tests whether past values of X improve the forecast of Y
beyond what Y's own past provides.  It is a statistical (not economic)
notion of causality.

Appropriate use:
  - Requires stationary series (run Phase 4 first).
  - Use on log-returns, NOT price levels.
  - Report F-statistic and p-value for each lag specification.
  - Combine with transfer entropy (transfer_entropy.py) for confirmation,
    as Granger is linear while transfer entropy is non-parametric.

Limitations:
  - Granger ≠ structural causality; spurious results are possible.
  - Confounders (VIX, USD) should be included as controls in a multivariate VAR.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GrangerResult:
    cause: str
    effect: str
    lag: int
    f_stat: float
    pvalue: float
    significant: bool

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "lag": self.lag,
            "f_stat": round(self.f_stat, 4),
            "pvalue": round(self.pvalue, 6),
            "significant": self.significant,
        }


class GrangerAnalyzer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def test_pair(
        self,
        cause: pd.Series,
        effect: pd.Series,
        max_lag: int | None = None,
        test: str = "ssr_ftest",
    ) -> list[GrangerResult]:
        """
        Test whether `cause` Granger-causes `effect` at each lag up to max_lag.

        test options (statsmodels): 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
        """
        max_lag = max_lag or self.settings.granger_max_lag
        idx = cause.dropna().index.intersection(effect.dropna().index)
        if len(idx) < 2 * max_lag + 10:
            log.warning("Insufficient observations for Granger test: %d", len(idx))
            return []

        data = pd.concat(
            [effect.rename("y"), cause.rename("x")], axis=1
        ).dropna()

        try:
            raw = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        except Exception as exc:
            log.error("Granger test failed (%s -> %s): %s", cause.name, effect.name, exc)
            return []

        results = []
        for lag, (tests, _) in raw.items():
            f_stat = tests[test][0]
            p = tests[test][1]
            results.append(
                GrangerResult(
                    cause=str(cause.name),
                    effect=str(effect.name),
                    lag=lag,
                    f_stat=f_stat,
                    pvalue=p,
                    significant=p < self.settings.significance_level,
                )
            )
        return results

    def test_commodity_to_equities(
        self,
        commodity_returns: pd.Series,
        equity_returns: pd.DataFrame,
        max_lag: int | None = None,
    ) -> pd.DataFrame:
        """
        Test all equity series against one commodity series.
        Returns a table of results at each lag.
        """
        all_results = []
        for col in equity_returns.columns:
            results = self.test_pair(commodity_returns, equity_returns[col], max_lag)
            all_results.extend(r.to_dict() for r in results)
        return pd.DataFrame(all_results)

    def minimum_significant_lag(
        self,
        results: pd.DataFrame,
    ) -> pd.Series:
        """
        For each (cause, effect) pair, return the minimum lag at which
        Granger causality is statistically significant.
        """
        sig = results[results["significant"]]
        if sig.empty:
            return pd.Series(dtype=float)
        return (
            sig.groupby(["cause", "effect"])["lag"].min()
        )

    def build_causality_matrix(
        self,
        returns: pd.DataFrame,
        max_lag: int | None = None,
        optimal_lag: int | None = None,
    ) -> pd.DataFrame:
        """
        NxN matrix: entry [i,j] = minimum Granger-causal lag from column i to j.
        NaN if no significant causality detected.
        """
        max_lag = max_lag or self.settings.granger_max_lag
        cols = returns.columns.tolist()
        n = len(cols)
        mat = pd.DataFrame(np.nan, index=cols, columns=cols)
        for i, cause in enumerate(cols):
            for j, effect in enumerate(cols):
                if i == j:
                    continue
                results = self.test_pair(
                    returns[cause], returns[effect], max_lag=max_lag
                )
                sig = [r for r in results if r.significant]
                if sig:
                    mat.loc[cause, effect] = min(r.lag for r in sig)
        return mat
