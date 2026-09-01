"""
Phase 10: PCA and Factor Analysis.

Goal: identify latent factors that drive cross-sectional variation in
commodity and equity returns.

PCA:
  - Determines how many factors explain most of the variance.
  - Factor 1 is typically the market factor (broad risk-on/risk-off).
  - Factor 2 is often a commodity factor.
  - Scree plot + cumulative explained variance guides factor selection.

Factor Analysis (FA):
  - Similar to PCA but optimises for common factor structure.
  - Factors are interpretable as latent constructs (e.g. "energy risk").
  - Rotation (Varimax) improves interpretability.

Why PCA before regression:
  - Reduces multicollinearity among regressors (commodity, market, FX, rates).
  - The first K principal components summarise most information.
  - Residuals from the factor model represent the idiosyncratic commodity effect.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PCAResult:
    components: pd.DataFrame          # shape: (n_components, n_features)
    loadings: pd.DataFrame             # shape: (n_features, n_components)
    explained_variance: pd.Series      # individual explained variance ratio
    cumulative_variance: pd.Series
    scores: pd.DataFrame               # shape: (n_obs, n_components) – factor scores
    n_components_90pct: int            # min components for 90% explained variance


class FactorModeler:
    """
    Phase 10: PCA and Factor Analysis on return panels.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_pca(
        self,
        returns: pd.DataFrame,
        n_components: int | None = None,
        standardise: bool = True,
    ) -> PCAResult:
        """
        Fit PCA on the return matrix.

        standardise=True is recommended because assets have different volatilities;
        without standardisation, high-vol assets (small caps) dominate PC1.
        """
        clean = returns.dropna()
        if standardise:
            scaler = StandardScaler()
            X = scaler.fit_transform(clean)
        else:
            X = clean.values

        n = n_components or min(self.settings.n_pca_components, clean.shape[1])
        pca = PCA(n_components=n, random_state=self.settings.random_seed)
        scores_arr = pca.fit_transform(X)

        comp_labels = [f"PC{i+1}" for i in range(n)]
        feat_labels = clean.columns.tolist()

        components = pd.DataFrame(
            pca.components_, index=comp_labels, columns=feat_labels
        )
        loadings = pd.DataFrame(
            pca.components_.T * np.sqrt(pca.explained_variance_),
            index=feat_labels,
            columns=comp_labels,
        )
        ev = pd.Series(pca.explained_variance_ratio_, index=comp_labels)
        cum_ev = ev.cumsum()
        n90 = int((cum_ev < 0.90).sum()) + 1

        log.info(
            "PCA: %d components explain %.1f%% variance. "
            "Components for 90%%: %d.",
            n, cum_ev.iloc[-1] * 100, n90,
        )

        return PCAResult(
            components=components,
            loadings=loadings,
            explained_variance=ev,
            cumulative_variance=cum_ev,
            scores=pd.DataFrame(scores_arr, index=clean.index, columns=comp_labels),
            n_components_90pct=n90,
        )

    def run_factor_analysis(
        self,
        returns: pd.DataFrame,
        n_factors: int | None = None,
        rotation: str = "varimax",
    ) -> pd.DataFrame:
        """
        Factor Analysis with Varimax rotation.
        Returns a DataFrame of factor loadings (n_features × n_factors).
        """
        clean = returns.dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(clean)
        n = n_factors or self.settings.n_pca_components
        fa = FactorAnalysis(
            n_components=n,
            rotation=rotation,
            random_state=self.settings.random_seed,
        )
        fa.fit(X)
        cols = [f"Factor{i+1}" for i in range(n)]
        return pd.DataFrame(fa.components_.T, index=clean.columns, columns=cols)

    def interpret_factors(
        self,
        loadings: pd.DataFrame,
        top_n: int = 5,
    ) -> dict[str, list[str]]:
        """
        For each factor, return the top_n assets by absolute loading.
        Useful for economic interpretation (e.g. Factor 1 ≈ energy complex).
        """
        interpretation = {}
        for factor in loadings.columns:
            top = loadings[factor].abs().nlargest(top_n).index.tolist()
            interpretation[factor] = top
        return interpretation
