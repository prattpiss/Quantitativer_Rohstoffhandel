"""
Phase 6: VAR (Vector Autoregression) and VECM (Vector Error Correction Model).

Decision rule: VAR vs VECM
──────────────────────────
- If series are I(1) and NOT cointegrated → VAR on first differences.
- If series are I(1) and cointegrated     → VECM (preserves long-run relationship).
- If series are I(0)                      → VAR on levels.

This module:
  1. Selects optimal lag order using information criteria (AIC, BIC, HQIC).
  2. Fits VAR or VECM as appropriate.
  3. Computes Impulse Response Functions (IRFs).
  4. Computes Forecast Error Variance Decomposition (FEVD).
  5. Tests residuals for autocorrelation (Portmanteau) and normality.

FEVD interpretation:
  The fraction of variance in equity Y at horizon h explained by a shock
  to commodity X directly answers: "How dominant is the commodity factor?"
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.stattools import coint

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class VARSummary:
    lag_order: int
    aic: float
    bic: float
    hqic: float
    model_type: str  # "VAR" or "VECM"
    n_obs: int


class VARAnalyzer:
    """
    Fit VAR/VECM models and extract IRFs and FEVD.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ── Cointegration check ───────────────────────────────────────────────────

    def test_cointegration(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> tuple[float, float]:
        """Engle-Granger cointegration test. Returns (stat, pvalue)."""
        idx = series_a.dropna().index.intersection(series_b.dropna().index)
        stat, p, _ = coint(series_a[idx], series_b[idx])
        return stat, p

    # ── Lag selection ─────────────────────────────────────────────────────────

    def select_lag(
        self,
        data: pd.DataFrame,
        max_lag: int | None = None,
        criterion: str = "aic",
    ) -> int:
        max_lag = max_lag or self.settings.var_max_lag
        model = VAR(data.dropna())
        try:
            result = model.select_order(maxlags=max_lag)
            return getattr(result, criterion)
        except Exception:
            log.warning("Lag selection failed; defaulting to lag=1.")
            return 1

    # ── VAR ───────────────────────────────────────────────────────────────────

    def fit_var(
        self,
        returns: pd.DataFrame,
        lag: int | None = None,
    ):
        """Fit a VAR(p) model. Returns the fitted VARResults object."""
        p = lag or self.select_lag(returns)
        log.info("Fitting VAR(%d) on %d series × %d obs.", p, returns.shape[1], len(returns))
        model = VAR(returns.dropna())
        result = model.fit(maxlags=p, ic=None)
        return result

    # ── Impulse Response ─────────────────────────────────────────────────────

    def impulse_response(
        self,
        var_result,
        periods: int = 20,
        orth: bool = True,
    ) -> pd.DataFrame:
        """
        Compute IRFs.
        orth=True → orthogonalised IRFs (Cholesky decomposition).
        Ordering: commodities first, then equities (economic prior).
        Returns a tidy DataFrame: columns = (response, impulse, period, irf).
        """
        irf = var_result.irf(periods=periods)
        cum_irf = irf.orth_irfs if orth else irf.irfs
        names = var_result.names
        rows = []
        for impulse_idx, impulse_name in enumerate(names):
            for response_idx, response_name in enumerate(names):
                for t in range(periods + 1):
                    rows.append({
                        "impulse": impulse_name,
                        "response": response_name,
                        "period": t,
                        "irf": cum_irf[t, response_idx, impulse_idx],
                    })
        return pd.DataFrame(rows)

    # ── Forecast Error Variance Decomposition ─────────────────────────────────

    def fevd(self, var_result, periods: int = 20) -> pd.DataFrame:
        """
        FEVD: fraction of forecast error variance in each variable
        explained by shocks to each other variable.
        Returns a tidy DataFrame.
        """
        fevd_obj = var_result.fevd(periods=periods)
        names = var_result.names
        rows = []
        for response_idx, response_name in enumerate(names):
            for impulse_idx, impulse_name in enumerate(names):
                for t in range(1, periods + 1):
                    rows.append({
                        "response": response_name,
                        "impulse": impulse_name,
                        "horizon": t,
                        "variance_share": fevd_obj.decomp[response_idx, t - 1, impulse_idx],
                    })
        return pd.DataFrame(rows)

    # ── Information delay metrics ─────────────────────────────────────────────

    def information_delay(
        self,
        irf_df: pd.DataFrame,
        impulse: str,
        response: str,
        quantiles: list[float] | None = None,
    ) -> dict[str, int]:
        """
        For a given impulse-response pair, compute the number of periods
        to reach a given fraction of the cumulative total IRF.

        Returns: {
          "time_to_25pct": h,
          "time_to_50pct": h,
          "time_to_75pct": h,
          "time_to_90pct": h,
          "time_to_95pct": h,
        }
        """
        quantiles = quantiles or [0.25, 0.50, 0.75, 0.90, 0.95]
        subset = irf_df[(irf_df["impulse"] == impulse) & (irf_df["response"] == response)]
        subset = subset.sort_values("period")
        cum = subset["irf"].cumsum().abs()
        total = cum.iloc[-1]
        if total == 0:
            return {f"time_to_{int(q*100)}pct": np.nan for q in quantiles}

        result = {}
        for q in quantiles:
            threshold = q * total
            idx = (cum >= threshold).idxmax()
            period = subset.loc[idx, "period"]
            result[f"time_to_{int(q*100)}pct"] = int(period)
        return result

    # ── Residual diagnostics ──────────────────────────────────────────────────

    def residual_diagnostics(self, var_result) -> dict:
        """Portmanteau test for autocorrelation in residuals."""
        try:
            pt = var_result.test_whiteness(nlags=10)
            return {
                "portmanteau_stat": pt.test_statistic,
                "portmanteau_pvalue": pt.pvalue,
                "white_noise": pt.pvalue > self.settings.significance_level,
            }
        except Exception as exc:
            log.warning("Residual diagnostics failed: %s", exc)
            return {}
