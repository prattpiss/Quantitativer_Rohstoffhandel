"""
Visualisation: Correlation heatmaps, cluster dendrograms.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


def _save(fig: go.Figure, output_dir: Path, name: str) -> None:
    path = output_dir / "figures" / f"{name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    try:
        fig.write_image(str(path.with_suffix(".png")), width=1200, height=1000)
    except Exception:
        pass
    log.info("Saved heatmap -> %s", path)


class HeatmapBuilder:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = settings.output_dir

    def correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        cluster: bool = True,
    ) -> go.Figure:
        """Annotated correlation heatmap, optionally with hierarchical clustering."""
        labels = corr_matrix.columns.tolist()
        z = corr_matrix.values

        if cluster:
            dist = 1 - np.abs(z)
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, None)
            link = linkage(squareform(dist), method="ward")
            dend = dendrogram(link, no_plot=True)
            order = dend["leaves"]
            z = z[np.ix_(order, order)]
            labels = [labels[i] for i in order]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(z, 2),
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig.update_layout(title=title, height=max(600, 30 * len(labels)))
        _save(fig, self.output_dir, f"heatmap_{title.replace(' ', '_').lower()}")
        return fig

    def rolling_correlation_heatmap(
        self,
        rolling_corr: pd.DataFrame,
        title: str = "Rolling Correlations",
    ) -> go.Figure:
        """Heatmap of rolling correlation over time (cols = assets, index = date)."""
        fig = go.Figure(
            go.Heatmap(
                z=rolling_corr.T.values,
                x=rolling_corr.index.astype(str),
                y=rolling_corr.columns.tolist(),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Asset")
        _save(fig, self.output_dir, f"rolling_corr_{title.replace(' ', '_').lower()}")
        return fig

    def lead_lag_heatmap(
        self,
        lag_matrix: pd.DataFrame,
        title: str = "Lead-Lag Matrix (days)",
    ) -> go.Figure:
        """Heatmap of minimum significant lag for all pairs."""
        fig = go.Figure(
            go.Heatmap(
                z=lag_matrix.values,
                x=lag_matrix.columns.tolist(),
                y=lag_matrix.index.tolist(),
                colorscale="Viridis_r",
                text=lag_matrix.round(1).astype(str).values,
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Effect (target)",
            yaxis_title="Cause (source)",
            height=max(600, 30 * len(lag_matrix)),
        )
        _save(fig, self.output_dir, "lead_lag_heatmap")
        return fig

    def fevd_heatmap(self, fevd_df: pd.DataFrame, horizon: int = 10) -> go.Figure:
        """FEVD stacked bar chart at a given horizon."""
        subset = fevd_df[fevd_df["horizon"] == horizon]
        pivot = subset.pivot(index="response", columns="impulse", values="variance_share")
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col]))
        fig.update_layout(
            barmode="stack",
            title=f"FEVD at Horizon {horizon}d",
            yaxis_title="Fraction of Variance",
        )
        _save(fig, self.output_dir, f"fevd_h{horizon}")
        return fig
