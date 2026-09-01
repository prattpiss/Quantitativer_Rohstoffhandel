"""
Phase 13: Information Flow Network Analysis.

Builds a directed weighted graph where:
  - Nodes = assets (commodities, ETFs, equities, macro variables)
  - Edges = statistically significant Granger causality or transfer entropy
  - Edge weight = strength of information flow (TE value or 1/min_lag)
  - Edge direction = source → target (information flow direction)

Node attributes:
  - proximity_level: distance to commodity (0=commodity, 6=market index)
  - cap_tier: mega/large/mid/small
  - sector: Energy, Metals, Agriculture, …

Network metrics computed:
  - In-degree / Out-degree (how many markets a node influences / is influenced by)
  - Betweenness centrality (information bottlenecks)
  - PageRank (overall influence in the network)
  - Average shortest path length (speed of information diffusion)
  - Community detection (which assets cluster together)

Visualisation: directed layout with Kamada-Kawai or hierarchical by proximity_level.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from config.settings import Settings
from config.symbols import UNIVERSE
from utils.logging_utils import get_logger

log = get_logger(__name__)


class InformationFlowNetwork:
    """
    Phase 13: Build and analyse directed information-flow network.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph: nx.DiGraph = nx.DiGraph()
        self._add_nodes()

    def _add_nodes(self) -> None:
        for ticker, meta in UNIVERSE.items():
            self.graph.add_node(
                ticker,
                name=meta.name,
                asset_class=meta.asset_class,
                sector=meta.sector,
                cap_tier=meta.cap_tier,
                proximity_level=meta.proximity_level,
            )

    # ── Build from Granger causality ──────────────────────────────────────────

    def add_granger_edges(
        self,
        granger_df: pd.DataFrame,
        weight_col: str = "f_stat",
    ) -> None:
        """
        Add edges from Granger causality results.
        granger_df must have columns: cause, effect, significant, weight_col.
        """
        sig = granger_df[granger_df["significant"]]
        for _, row in sig.iterrows():
            src, tgt = row["cause"], row["effect"]
            w = float(row.get(weight_col, 1.0))
            if self.graph.has_edge(src, tgt):
                self.graph[src][tgt]["weight"] = max(self.graph[src][tgt]["weight"], w)
            else:
                self.graph.add_edge(src, tgt, weight=w, source="granger",
                                    lag=row.get("lag", np.nan))

    # ── Build from Transfer Entropy ───────────────────────────────────────────

    def add_te_edges(
        self,
        te_matrix: pd.DataFrame,
        te_threshold: float | None = None,
    ) -> None:
        """
        Add edges from a transfer entropy matrix.
        te_matrix[i, j] = T(i → j); edge added if value > threshold.
        """
        threshold = te_threshold or self.settings.network_threshold
        for src in te_matrix.index:
            for tgt in te_matrix.columns:
                if src == tgt:
                    continue
                val = float(te_matrix.loc[src, tgt])
                if val > threshold:
                    if self.graph.has_edge(src, tgt):
                        self.graph[src][tgt]["te"] = val
                    else:
                        self.graph.add_edge(src, tgt, weight=val, source="transfer_entropy")

    # ── Network metrics ───────────────────────────────────────────────────────

    def compute_metrics(self) -> pd.DataFrame:
        """
        Compute node-level network centrality measures.
        """
        G = self.graph
        in_deg = dict(G.in_degree(weight="weight"))
        out_deg = dict(G.out_degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        try:
            pagerank = nx.pagerank(G, weight="weight", max_iter=500)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: np.nan for n in G.nodes}

        rows = []
        for node in G.nodes:
            attr = G.nodes[node]
            rows.append({
                "ticker": node,
                "name": attr.get("name", ""),
                "sector": attr.get("sector", ""),
                "cap_tier": attr.get("cap_tier", ""),
                "proximity_level": attr.get("proximity_level", -1),
                "in_degree": in_deg.get(node, 0),
                "out_degree": out_deg.get(node, 0),
                "net_flow": out_deg.get(node, 0) - in_deg.get(node, 0),
                "betweenness": betweenness.get(node, 0),
                "pagerank": pagerank.get(node, np.nan),
            })
        df = pd.DataFrame(rows).set_index("ticker")
        return df.sort_values("pagerank", ascending=False)

    def detect_communities(self) -> dict[str, int]:
        """
        Community detection via Louvain (greedy modularity on undirected projection).
        Returns dict: {ticker: community_id}.
        """
        undirected = self.graph.to_undirected()
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(undirected, weight="weight")
            mapping = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    mapping[node] = i
            return mapping
        except Exception as exc:
            log.warning("Community detection failed: %s", exc)
            return {}

    def get_information_hierarchy(self) -> pd.DataFrame:
        """
        Sort nodes by proximity_level and net outflow (out_degree - in_degree).
        High outflow + low proximity = information source.
        """
        metrics = self.compute_metrics()
        metrics = metrics.sort_values(["proximity_level", "net_flow"], ascending=[True, False])
        return metrics

    def to_adjacency(self, weight: str = "weight") -> pd.DataFrame:
        """Return weighted adjacency matrix as DataFrame."""
        nodes = list(self.graph.nodes)
        mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for u, v, data in self.graph.edges(data=True):
            mat.loc[u, v] = data.get(weight, 1.0)
        return mat

    def summary(self) -> dict[str, Any]:
        G = self.graph
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "avg_clustering": nx.average_clustering(G.to_undirected()),
        }
