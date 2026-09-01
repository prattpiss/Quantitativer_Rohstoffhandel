"""
Phase 3 & 4: Descriptive statistics and distribution analysis.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DescriptiveStats:
    mean: pd.Series
    median: pd.Series
    std: pd.Series
    skewness: pd.Series
    kurtosis: pd.Series
    min: pd.Series
    max: pd.Series
    percentile_5: pd.Series
    percentile_95: pd.Series
    annualised_vol: pd.Series
    sharpe: pd.Series
    sortino: pd.Series
    max_drawdown: pd.Series
    hit_rate: pd.Series         # fraction of positive returns
    var_95: pd.Series           # 95 % historical VaR
    cvar_95: pd.Series          # 95 % CVaR / Expected Shortfall

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Mean (ann.)": self.mean * 252,
                "Median": self.median,
                "Vol (ann.)": self.annualised_vol,
                "Skewness": self.skewness,
                "Excess Kurtosis": self.kurtosis,
                "Min": self.min,
                "Max": self.max,
                "P5": self.percentile_5,
                "P95": self.percentile_95,
                "Sharpe": self.sharpe,
                "Sortino": self.sortino,
                "Max Drawdown": self.max_drawdown,
                "Hit Rate": self.hit_rate,
                "VaR 95%": self.var_95,
                "CVaR 95%": self.cvar_95,
            }
        )


class DescriptiveAnalyzer:
    """
    Phase 3: Compute comprehensive descriptive statistics for return series.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def compute(self, returns: pd.DataFrame) -> DescriptiveStats:
        clean = returns.dropna(how="all")

        ann_vol = clean.std() * np.sqrt(252)
        ann_mean = clean.mean() * 252
        sharpe = ann_mean / ann_vol
        downside = clean[clean < 0].std() * np.sqrt(252)
        sortino = ann_mean / downside.replace(0, np.nan)
        var_95 = clean.quantile(0.05)
        cvar_95 = clean[clean <= var_95].mean()

        # Max drawdown on cumulative returns
        cum = (1 + clean).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()

        return DescriptiveStats(
            mean=clean.mean(),
            median=clean.median(),
            std=clean.std(),
            skewness=clean.skew(),
            kurtosis=clean.kurt(),
            min=clean.min(),
            max=clean.max(),
            percentile_5=clean.quantile(0.05),
            percentile_95=clean.quantile(0.95),
            annualised_vol=ann_vol,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            hit_rate=(clean > 0).mean(),
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def rolling_volatility(
        self,
        returns: pd.DataFrame,
        window: int | None = None,
        annualise: bool = True,
    ) -> pd.DataFrame:
        w = window or self.settings.rolling_medium
        vol = returns.rolling(w, min_periods=w // 2).std()
        if annualise:
            vol = vol * np.sqrt(252)
        return vol

    def rolling_correlation(
        self,
        returns: pd.DataFrame,
        target: str,
        window: int | None = None,
    ) -> pd.DataFrame:
        """Rolling Pearson correlation of all columns vs. target."""
        w = window or self.settings.rolling_medium
        other = returns.drop(columns=[target], errors="ignore")
        result = {}
        for col in other.columns:
            result[col] = (
                returns[col]
                .rolling(w, min_periods=w // 2)
                .corr(returns[target])
            )
        return pd.DataFrame(result)

    def normality_tests(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Jarque-Bera and Shapiro-Wilk tests for normality."""
        rows = []
        for col in returns.columns:
            s = returns[col].dropna()
            jb_stat, jb_p = stats.jarque_bera(s)
            # Shapiro-Wilk is expensive for n > 5000; sample if necessary
            sample = s.sample(min(len(s), 5_000), random_state=self.settings.random_seed)
            sw_stat, sw_p = stats.shapiro(sample)
            rows.append({
                "ticker": col,
                "jb_stat": jb_stat,
                "jb_pvalue": jb_p,
                "sw_stat": sw_stat,
                "sw_pvalue": sw_p,
                "normal_jb": jb_p > self.settings.significance_level,
                "normal_sw": sw_p > self.settings.significance_level,
            })
        return pd.DataFrame(rows).set_index("ticker")
