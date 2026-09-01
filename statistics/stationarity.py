"""
Phase 4: Stationarity testing.

Tests implemented:
- ADF  (Augmented Dickey-Fuller)  – null: unit root exists (non-stationary)
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin) – null: series IS stationary
- Phillips-Perron – similar to ADF but non-parametric correction for serial correlation

Recommendation:
  Use ADF + KPSS together for robustness ("confirmatory testing"):
  - Both agree stationary  → stationary
  - Both agree non-stationary → non-stationary; differentiate
  - Conflict → inconclusive; inspect ACF/PACF manually

Order of integration:
  If level is non-stationary, test first differences.  Most financial
  return series are I(0); price levels are I(1).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class StationarityResult:
    ticker: str
    test: str
    statistic: float
    pvalue: float
    critical_1pct: float
    critical_5pct: float
    critical_10pct: float
    reject_null: bool
    conclusion: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "test": self.test,
            "statistic": round(self.statistic, 4),
            "pvalue": round(self.pvalue, 4),
            "crit_1%": round(self.critical_1pct, 4),
            "crit_5%": round(self.critical_5pct, 4),
            "crit_10%": round(self.critical_10pct, 4),
            "reject_null": self.reject_null,
            "conclusion": self.conclusion,
        }


class StationarityTester:
    """
    Phase 4: Unit-root and stationarity tests for all series.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ── ADF ──────────────────────────────────────────────────────────────────

    def adf(
        self,
        series: pd.Series,
        regression: Literal["c", "ct", "ctt", "n"] = "c",
        autolag: str = "AIC",
    ) -> StationarityResult:
        """
        ADF test.
        regression: "c"=constant, "ct"=constant+trend, "n"=none.
        Use "c" for most return series; "ct" if a trend is visible.
        """
        s = series.dropna()
        adf_stat, p, _, _, crits, _ = adfuller(s, regression=regression, autolag=autolag)
        reject = p < self.settings.significance_level
        return StationarityResult(
            ticker=series.name or "",
            test="ADF",
            statistic=adf_stat,
            pvalue=p,
            critical_1pct=crits["1%"],
            critical_5pct=crits["5%"],
            critical_10pct=crits["10%"],
            reject_null=reject,
            conclusion="Stationary (I(0))" if reject else "Non-stationary (unit root)",
        )

    # ── KPSS ─────────────────────────────────────────────────────────────────

    def kpss_test(
        self,
        series: pd.Series,
        regression: Literal["c", "ct"] = "c",
        nlags: str = "auto",
    ) -> StationarityResult:
        """
        KPSS test. Null: series is stationary.
        Reject null → evidence of non-stationarity.
        """
        s = series.dropna()
        kpss_stat, p, _, crits = kpss(s, regression=regression, nlags=nlags)
        reject = p < self.settings.significance_level
        return StationarityResult(
            ticker=series.name or "",
            test="KPSS",
            statistic=kpss_stat,
            pvalue=p,
            critical_1pct=crits["1%"],
            critical_5pct=crits["5%"],
            critical_10pct=crits["10%"],
            reject_null=reject,
            conclusion="Non-stationary" if reject else "Stationary",
        )

    # ── Phillips-Perron ───────────────────────────────────────────────────────

    def phillips_perron(self, series: pd.Series) -> StationarityResult:
        """
        Phillips-Perron test via arch package.
        Preferred over ADF when autocorrelation structure is unclear.
        """
        s = series.dropna()
        pp = PhillipsPerron(s)
        reject = pp.pvalue < self.settings.significance_level
        return StationarityResult(
            ticker=series.name or "",
            test="PP",
            statistic=pp.stat,
            pvalue=pp.pvalue,
            critical_1pct=pp.critical_values["1%"],
            critical_5pct=pp.critical_values["5%"],
            critical_10pct=pp.critical_values["10%"],
            reject_null=reject,
            conclusion="Stationary" if reject else "Non-stationary",
        )

    # ── Batch testing ─────────────────────────────────────────────────────────

    def test_all(
        self,
        data: pd.DataFrame,
        run_adf: bool = True,
        run_kpss: bool = True,
        run_pp: bool = True,
    ) -> pd.DataFrame:
        """
        Run selected tests for every column.  Returns consolidated DataFrame.
        """
        rows = []
        for col in data.columns:
            series = data[col].rename(col)
            if run_adf:
                rows.append(self.adf(series).to_dict())
            if run_kpss:
                rows.append(self.kpss_test(series).to_dict())
            if run_pp:
                try:
                    rows.append(self.phillips_perron(series).to_dict())
                except Exception as exc:
                    log.warning("PP test failed for %s: %s", col, exc)
        return pd.DataFrame(rows)

    def recommend_integration_order(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        For each series, test both level and first-difference.
        Returns a summary table with recommended integration order.
        """
        rows = []
        for col in prices.columns:
            level_adf = self.adf(prices[col].rename(col))
            ret_adf = self.adf(returns[col].dropna().rename(col))
            if level_adf.reject_null:
                order = 0
            elif ret_adf.reject_null:
                order = 1
            else:
                order = 2  # rare for financial series
            rows.append({
                "ticker": col,
                "level_adf_p": level_adf.pvalue,
                "return_adf_p": ret_adf.pvalue,
                "integration_order": order,
            })
        return pd.DataFrame(rows).set_index("ticker")
