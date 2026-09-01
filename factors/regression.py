"""
Phase 10: Factor Regression Models.

Model:
  R_equity = α + β_commodity·R_commodity + β_market·R_SPY
           + β_dollar·R_USD + β_vix·ΔVIX + β_rates·Δrates
           + β_sector·R_sector_ETF + ε

Purpose:
  1. Measure how much of equity return variance each factor explains (R²).
  2. Isolate the pure commodity effect (residual after controlling for other factors).
  3. Test whether the commodity coefficient β is statistically significant.
  4. Compute rolling regressions to detect regime changes.

Method:
  - OLS for a single estimation.
  - Ridge/Lasso when regressors are collinear.
  - Newey-West standard errors for serial correlation in residuals.
  - Rolling OLS window = 252 days for time-varying betas.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class RegressionResult:
    asset: str
    alpha: float
    betas: dict[str, float]
    r_squared: float
    adj_r_squared: float
    pvalues: dict[str, float]
    residuals: pd.Series
    aic: float
    bic: float
    nw_pvalues: dict[str, float]    # Newey-West corrected p-values

    def to_dict(self) -> dict:
        d = {
            "asset": self.asset,
            "alpha": round(self.alpha, 6),
            "r_squared": round(self.r_squared, 4),
            "adj_r_squared": round(self.adj_r_squared, 4),
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
        }
        d.update({f"beta_{k}": round(v, 4) for k, v in self.betas.items()})
        d.update({f"pvalue_{k}": round(v, 6) for k, v in self.pvalues.items()})
        d.update({f"nw_pvalue_{k}": round(v, 6) for k, v in self.nw_pvalues.items()})
        return d


class FactorRegressor:
    """
    Phase 10: OLS factor regression with Newey-West SE and rolling estimation.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fit(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        nw_lags: int = 5,
    ) -> RegressionResult:
        """
        OLS with Newey-West heteroscedasticity and autocorrelation consistent SE.
        """
        data = pd.concat([y, X], axis=1).dropna()
        y_clean = data.iloc[:, 0]
        X_clean = sm.add_constant(data.iloc[:, 1:])

        model = sm.OLS(y_clean, X_clean)
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

        regressors = X.columns.tolist()
        betas = {reg: float(result.params.get(reg, np.nan)) for reg in regressors}
        pvalues = {reg: float(result.pvalues.get(reg, np.nan)) for reg in regressors}
        nw_pvalues = pvalues  # statsmodels applies NW when cov_type="HAC"

        return RegressionResult(
            asset=str(y.name),
            alpha=float(result.params.get("const", np.nan)),
            betas=betas,
            r_squared=float(result.rsquared),
            adj_r_squared=float(result.rsquared_adj),
            pvalues=pvalues,
            residuals=pd.Series(result.resid, index=y_clean.index, name=f"resid_{y.name}"),
            aic=float(result.aic),
            bic=float(result.bic),
            nw_pvalues=nw_pvalues,
        )

    def fit_all(
        self,
        equity_returns: pd.DataFrame,
        factors: pd.DataFrame,
        nw_lags: int = 5,
    ) -> pd.DataFrame:
        """Fit factor model for every equity column. Returns summary table."""
        rows = []
        residuals = {}
        for col in equity_returns.columns:
            try:
                res = self.fit(equity_returns[col].rename(col), factors, nw_lags)
                rows.append(res.to_dict())
                residuals[col] = res.residuals
            except Exception as exc:
                log.error("Regression failed for %s: %s", col, exc)
        self.last_residuals = pd.DataFrame(residuals)
        return pd.DataFrame(rows).set_index("asset")

    def rolling_betas(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        window: int | None = None,
        regressor: str | None = None,
    ) -> pd.DataFrame:
        """
        Rolling OLS beta estimates.
        Returns DataFrame with columns = X columns, index = date.
        """
        w = window or self.settings.rolling_long
        data = pd.concat([y, X], axis=1).dropna()
        results = []
        dates = []
        for i in range(w, len(data) + 1):
            window_data = data.iloc[i - w : i]
            y_w = window_data.iloc[:, 0]
            X_w = sm.add_constant(window_data.iloc[:, 1:])
            try:
                ols = sm.OLS(y_w, X_w).fit()
                row = ols.params.drop("const", errors="ignore").to_dict()
            except Exception:
                row = {col: np.nan for col in X.columns}
            results.append(row)
            dates.append(data.index[i - 1])
        return pd.DataFrame(results, index=pd.DatetimeIndex(dates))

    def variance_decomposition(
        self,
        equity_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each equity, compute the fraction of variance explained by each factor.
        Returns a DataFrame (equities × factors).
        """
        rows = {}
        for col in equity_returns.columns:
            y = equity_returns[col].dropna()
            data = pd.concat([y, factors], axis=1).dropna()
            y_c = data.iloc[:, 0]
            X_c = data.iloc[:, 1:]
            total_var = y_c.var()
            decomp = {}
            for factor in X_c.columns:
                x_f = sm.add_constant(X_c[[factor]])
                try:
                    r = sm.OLS(y_c, x_f).fit()
                    decomp[factor] = r.rsquared * total_var
                except Exception:
                    decomp[factor] = np.nan
            rows[col] = decomp
        df = pd.DataFrame(rows).T
        df = df.div(df.sum(axis=1), axis=0)  # normalise to sum to 1
        return df
