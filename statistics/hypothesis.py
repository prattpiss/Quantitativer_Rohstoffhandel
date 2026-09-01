"""
Phase 12: Hypothesis testing framework.

H1: Mega Caps react faster than Small Caps (shorter information delay).
H2: Small Caps react more strongly (larger impulse-response magnitude).
H3: Proximity to commodity → stronger and faster effect.
H4: Commodity explains the largest fraction of equity return variance.
H5: Macro releases generate systematic, predictable lags.
H6: Estimated lags are stable across sub-periods (no structural break).
H7: Strategy based on detected lags has positive expected value.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class HypothesisResult:
    hypothesis: str
    test_name: str
    statistic: float
    pvalue: float
    reject: bool
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "test": self.test_name,
            "statistic": round(self.statistic, 4),
            "pvalue": round(self.pvalue, 6),
            "reject_H0": self.reject,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "CI_lower": round(self.confidence_interval[0], 4) if self.confidence_interval else None,
            "CI_upper": round(self.confidence_interval[1], 4) if self.confidence_interval else None,
            "notes": self.notes,
        }


class HypothesisTester:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alpha = settings.significance_level

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        return (a.mean() - b.mean()) / (pooled_std + 1e-12)

    # ── H1: Mega Caps react faster ───────────────────────────────────────────

    def test_h1_mega_faster(
        self,
        mega_lags: Sequence[float],
        small_lags: Sequence[float],
    ) -> HypothesisResult:
        """
        One-sided Mann-Whitney U test:
        H0: lag(mega) >= lag(small)
        H1: lag(mega) < lag(small)  (mega reacts faster = smaller lag)
        """
        mega = np.array(mega_lags)
        small = np.array(small_lags)
        stat, p = stats.mannwhitneyu(mega, small, alternative="less")
        d = self._cohens_d(mega, small)
        return HypothesisResult(
            hypothesis="H1: Mega Caps react faster than Small Caps",
            test_name="Mann-Whitney U (one-sided)",
            statistic=stat,
            pvalue=p,
            reject=p < self.alpha,
            effect_size=d,
            notes="Positive d: mega has smaller lag (faster).",
        )

    # ── H2: Small Caps react stronger ────────────────────────────────────────

    def test_h2_small_stronger(
        self,
        mega_magnitudes: Sequence[float],
        small_magnitudes: Sequence[float],
    ) -> HypothesisResult:
        """
        One-sided Mann-Whitney U:
        H0: |effect(small)| <= |effect(mega)|
        H1: |effect(small)| > |effect(mega)|
        """
        mega = np.array(mega_magnitudes)
        small = np.array(small_magnitudes)
        stat, p = stats.mannwhitneyu(small, mega, alternative="greater")
        d = self._cohens_d(small, mega)
        return HypothesisResult(
            hypothesis="H2: Small Caps react more strongly",
            test_name="Mann-Whitney U (one-sided)",
            statistic=stat,
            pvalue=p,
            reject=p < self.alpha,
            effect_size=d,
        )

    # ── H3: Proximity → stronger effect ──────────────────────────────────────

    def test_h3_proximity(
        self,
        proximity_levels: Sequence[int],
        effect_magnitudes: Sequence[float],
    ) -> HypothesisResult:
        """
        Spearman rank correlation between proximity level and effect magnitude.
        H0: ρ = 0  (no monotone relationship)
        H1: ρ < 0  (higher proximity → larger effect, since level 0 is commodity)
        """
        rho, p = stats.spearmanr(proximity_levels, effect_magnitudes, alternative="less")
        return HypothesisResult(
            hypothesis="H3: Proximity to commodity → stronger effect",
            test_name="Spearman rank correlation",
            statistic=rho,
            pvalue=p,
            reject=p < self.alpha,
            effect_size=rho,
            notes="Negative rho expected: lower level = closer proximity = larger effect.",
        )

    # ── H4: Commodity explains the most variance ──────────────────────────────

    def test_h4_commodity_dominates(
        self,
        r_squared_commodity: Sequence[float],
        r_squared_other_factors: dict[str, Sequence[float]],
    ) -> HypothesisResult:
        """
        Non-parametric: Wilcoxon signed-rank vs each other factor.
        H0: R²(commodity) <= R²(factor)
        H1: R²(commodity) > R²(factor)
        Takes the minimum p-value across all factor comparisons.
        """
        rc = np.array(r_squared_commodity)
        min_p = 1.0
        for factor_name, rf_vals in r_squared_other_factors.items():
            rf = np.array(rf_vals)
            _, p = stats.wilcoxon(rc - rf, alternative="greater")
            min_p = min(min_p, p)
        return HypothesisResult(
            hypothesis="H4: Commodity factor explains most return variance",
            test_name="Wilcoxon signed-rank (min p across factors)",
            statistic=float("nan"),
            pvalue=min_p,
            reject=min_p < self.alpha,
        )

    # ── H6: Lags are stable across sub-periods ───────────────────────────────

    def test_h6_lag_stability(
        self,
        lags_period1: Sequence[float],
        lags_period2: Sequence[float],
    ) -> HypothesisResult:
        """
        Levene test for equality of variances + t-test for equality of means.
        H0: lag distribution unchanged across periods.
        """
        l1 = np.array(lags_period1)
        l2 = np.array(lags_period2)
        lev_stat, lev_p = stats.levene(l1, l2)
        t_stat, t_p = stats.ttest_ind(l1, l2, equal_var=lev_p > self.alpha)
        combined_p = max(lev_p, t_p)  # conservative: both must confirm stability
        return HypothesisResult(
            hypothesis="H6: Lag estimates are stable across sub-periods",
            test_name="Levene variance + Welch t-test",
            statistic=t_stat,
            pvalue=combined_p,
            reject=combined_p < self.alpha,
            notes="Reject → lags are unstable (structural break).",
        )

    # ── Multiple testing correction ───────────────────────────────────────────

    def fdr_correction(
        self,
        results: list[HypothesisResult],
        method: str = "fdr_bh",
    ) -> list[HypothesisResult]:
        """
        Apply Benjamini-Hochberg FDR correction to a list of test results.
        Returns updated results with corrected reject decisions.
        """
        pvalues = [r.pvalue for r in results]
        reject_arr, pvals_corrected, _, _ = multipletests(
            pvalues, alpha=self.alpha, method=method
        )
        for i, r in enumerate(results):
            r.reject = bool(reject_arr[i])
            r.pvalue = pvals_corrected[i]
            r.notes += f" [FDR-corrected p={pvals_corrected[i]:.4f}]"
        return results

    def run_all(
        self,
        mega_lags: Sequence[float],
        small_lags: Sequence[float],
        mega_effects: Sequence[float],
        small_effects: Sequence[float],
        proximity_levels: Sequence[int],
        proximity_effects: Sequence[float],
    ) -> pd.DataFrame:
        results = [
            self.test_h1_mega_faster(mega_lags, small_lags),
            self.test_h2_small_stronger(mega_effects, small_effects),
            self.test_h3_proximity(proximity_levels, proximity_effects),
        ]
        results = self.fdr_correction(results)
        return pd.DataFrame([r.to_dict() for r in results])
