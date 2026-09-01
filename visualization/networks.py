"""
Visualisation: Network graphs using NetworkX + Plotly.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from config.settings import Settings
from config.symbols import UNIVERSE
from utils.logging_utils import get_logger

log = get_logger(__name__)

# Colour palette by asset class
ASSET_CLASS_COLORS = {
    "commodity": "#FF6B35",
    "etf": "#004E89",
    "equity": "#1A936F",
    "index": "#88498F",
    "control": "#C6C5B9",
}

# Node size proxy by proximity level (closer = larger)
PROXIMITY_SIZE = {0: 30, 1: 25, 2: 20, 3: 16, 4: 14, 5: 12, 6: 20, -1: 10}


def _save_html(fig: go.Figure, output_dir: Path, name: str) -> None:
    path = output_dir / "figures" / f"{name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    log.info("Saved network -> %s", path)


class NetworkVisualizer:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = settings.output_dir

    def plot_information_network(
        self,
        graph: nx.DiGraph,
        title: str = "Information Flow Network",
        layout: str = "kamada_kawai",
    ) -> go.Figure:
        """
        Interactive directed network.
        Edge thickness ∝ weight; node color = asset class; node size ∝ proximity.
        Arrow direction shows information flow.
        """
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph.to_undirected(), weight="weight")
        elif layout == "spring":
            pos = nx.spring_layout(graph, seed=self.settings.random_seed, weight="weight")
        elif layout == "hierarchical":
            pos = self._hierarchical_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=self.settings.random_seed)

        edge_traces = self._build_edge_traces(graph, pos)
        node_trace = self._build_node_trace(graph, pos)

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
        )
        _save_html(fig, self.output_dir, f"network_{title.replace(' ', '_').lower()}")
        return fig

    def _hierarchical_layout(self, graph: nx.DiGraph) -> dict:
        """Layout based on proximity_level (y-axis) and PageRank (x-axis)."""
        try:
            pr = nx.pagerank(graph, weight="weight", max_iter=500)
        except Exception:
            pr = {n: 0.5 for n in graph.nodes}

        pos = {}
        level_nodes: dict[int, list] = {}
        for node in graph.nodes:
            lv = graph.nodes[node].get("proximity_level", 6)
            level_nodes.setdefault(lv, []).append(node)

        for lv, nodes in sorted(level_nodes.items()):
            nodes_sorted = sorted(nodes, key=lambda n: pr.get(n, 0), reverse=True)
            for i, node in enumerate(nodes_sorted):
                x = i / max(len(nodes_sorted) - 1, 1) if len(nodes_sorted) > 1 else 0.5
                pos[node] = (x, -lv)  # y decreases with proximity level
        return pos

    def _build_edge_traces(self, graph: nx.DiGraph, pos: dict) -> list[go.Scatter]:
        traces = []
        weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
        max_w = max(weights) if weights else 1.0
        for u, v, data in graph.edges(data=True):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = data.get("weight", 1.0)
            width = 1 + 4 * (w / max_w)
            traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=width, color="rgba(80,80,80,0.4)"),
                    hoverinfo="none",
                )
            )
        return traces

    def _build_node_trace(self, graph: nx.DiGraph, pos: dict) -> go.Scatter:
        x_nodes, y_nodes, text, colors, sizes = [], [], [], [], []
        for node in graph.nodes:
            if node not in pos:
                continue
            x, y = pos[node]
            x_nodes.append(x)
            y_nodes.append(y)
            attr = graph.nodes[node]
            ac = attr.get("asset_class", "equity")
            prox = attr.get("proximity_level", -1)
            name = attr.get("name", node)
            text.append(f"{node}<br>{name}<br>Level: {prox}")
            colors.append(ASSET_CLASS_COLORS.get(ac, "#888888"))
            sizes.append(PROXIMITY_SIZE.get(prox, 10))

        return go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            hoverinfo="text",
            text=[t.split("<br>")[0] for t in text],
            hovertext=text,
            textposition="top center",
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=1, color="white"),
            ),
        )
