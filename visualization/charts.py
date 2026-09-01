"""
Visualisation: standard time-series and return charts.
All charts are built with Plotly (interactive) and saved as HTML + PNG.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


def _save(fig: go.Figure, output_dir: Path, name: str) -> None:
    path_html = output_dir / "figures" / f"{name}.html"
    path_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path_html))
    try:
        fig.write_image(str(path_html.with_suffix(".png")), width=1400, height=700)
    except Exception:
        pass  # kaleido optional
    log.info("Saved figure -> %s", path_html)


class ChartBuilder:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = settings.output_dir

    def price_history(self, prices: pd.DataFrame, title: str = "Price History") -> go.Figure:
        fig = go.Figure()
        for col in prices.columns:
            fig.add_trace(go.Scatter(x=prices.index, y=prices[col], name=col, mode="lines"))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        _save(fig, self.output_dir, f"price_history_{title.replace(' ', '_').lower()}")
        return fig

    def rolling_volatility(self, vol: pd.DataFrame, title: str = "Rolling Volatility") -> go.Figure:
        fig = go.Figure()
        for col in vol.columns:
            fig.add_trace(go.Scatter(x=vol.index, y=vol[col], name=col, mode="lines"))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Annualised Vol")
        _save(fig, self.output_dir, "rolling_volatility")
        return fig

    def return_distribution(self, returns: pd.DataFrame, bins: int = 100) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=len(returns.columns),
            subplot_titles=list(returns.columns),
        )
        for i, col in enumerate(returns.columns, 1):
            s = returns[col].dropna()
            fig.add_trace(go.Histogram(x=s, nbinsx=bins, name=col, showlegend=False), row=1, col=i)
        fig.update_layout(title="Return Distributions")
        _save(fig, self.output_dir, "return_distributions")
        return fig

    def ccf_plot(
        self,
        ccf: pd.Series,
        title: str = "Cross-Correlation Function",
        significance_band: float | None = None,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ccf.index, y=ccf.values, name="CCF"))
        if significance_band is not None:
            n = len(ccf)
            band = significance_band / np.sqrt(n)
            fig.add_hline(y=band, line_dash="dash", line_color="red", annotation_text="95% CI")
            fig.add_hline(y=-band, line_dash="dash", line_color="red")
        fig.update_layout(title=title, xaxis_title="Lag (days)", yaxis_title="Correlation")
        _save(fig, self.output_dir, f"ccf_{title.replace(' ', '_').lower()}")
        return fig

    def irf_plot(self, irf_df: pd.DataFrame, impulse: str, response: str) -> go.Figure:
        subset = irf_df[(irf_df["impulse"] == impulse) & (irf_df["response"] == response)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset["period"], y=subset["irf"], mode="lines+markers"))
        fig.add_hline(y=0, line_color="black")
        fig.update_layout(
            title=f"IRF: {impulse} → {response}",
            xaxis_title="Horizon (days)",
            yaxis_title="Response",
        )
        _save(fig, self.output_dir, f"irf_{impulse}_{response}")
        return fig

    def event_study_plot(self, cars: list[float], event_type: str, asset: str) -> go.Figure:
        arr = np.array(cars)
        fig = go.Figure()
        fig.add_trace(go.Box(y=arr, name=f"{event_type} → {asset}"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title=f"CAR Distribution: {event_type} → {asset}", yaxis_title="CAR")
        _save(fig, self.output_dir, f"event_{event_type}_{asset}")
        return fig

    def cumulative_returns(self, returns: pd.DataFrame) -> go.Figure:
        cum = (1 + returns.fillna(0)).cumprod()
        fig = go.Figure()
        for col in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col))
        fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1")
        _save(fig, self.output_dir, "cumulative_returns")
        return fig
