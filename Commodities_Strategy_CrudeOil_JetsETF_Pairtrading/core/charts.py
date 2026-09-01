"""Wiederverwendbare Chart-Bausteine: Rohdaten, Trade-Marker, Equity-Dashboard.

Ziel ist Nachvollziehbarkeit: jede Backtest-Aussage muss visuell an der
tatsächlich geladenen Kursreihe und an den einzelnen Trades prüfbar sein.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T

GREEN = "#3fb950"
RED = "#ff7b72"
AMBER = "#d29922"
BLUE = "#58a6ff"
GREY = "#8b949e"


# ── Rohdaten ────────────────────────────────────────────────────────────────
def price_panel(panel: pd.DataFrame, title: str = "Geladene Kursreihen",
                normalize: bool = True, log: bool = True,
                groups: dict[str, str] | None = None) -> go.Figure:
    """Alle tatsächlich geladenen Kursreihen in einem Chart."""
    fig = go.Figure()
    if panel is None or panel.empty:
        return fig
    df = panel.ffill()
    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        if s.empty:
            continue
        y = s / s.iloc[0] * 100.0 if normalize else s
        grp = groups.get(col) if groups else None
        fig.add_trace(go.Scatter(
            x=s.index, y=y, name=str(col), mode="lines",
            legendgroup=grp, legendgrouptitle_text=grp,
            line=dict(width=1.3, color=T.PAL[i % len(T.PAL)]),
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Index (Start = 100)" if normalize else "Kurs",
                   type="log" if log else "linear"),
        xaxis=dict(title="", rangeslider=dict(visible=False)),
        hovermode="x unified",
    )
    return fig


def series_grid(series: dict[str, pd.Series], title: str = "Eingangsreihen",
                cols: int = 2, row_height: int = 190) -> tuple[go.Figure, int]:
    """Kleine Multiples — je Reihe eine eigene Achse in Originaleinheiten."""
    items = [(k, pd.to_numeric(v, errors="coerce").dropna())
             for k, v in series.items()]
    items = [(k, v) for k, v in items if len(v)]
    if not items:
        return go.Figure(), 200
    rows = int(np.ceil(len(items) / cols))
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False,
                        subplot_titles=[k for k, _ in items],
                        vertical_spacing=0.13 / max(rows - 1, 1) if rows > 1 else 0.1,
                        horizontal_spacing=0.07)
    for i, (name, s) in enumerate(items):
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(go.Scatter(
            x=s.index, y=s.to_numpy(float), name=name, mode="lines",
            line=dict(width=1.2, color=T.PAL[i % len(T.PAL)]), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>"), row=r, col=c)
    fig.update_annotations(font_size=11)
    fig.update_layout(title=title)
    return fig, rows * row_height + 60


def data_table(panel: pd.DataFrame, n: int = 6) -> str:
    """Erste und letzte Zeilen der geladenen Matrix — Sichtprüfung der Rohwerte."""
    if panel is None or panel.empty:
        return '<p class="text-muted small">Keine Daten.</p>'
    head, tail = panel.head(n).round(3), panel.tail(n).round(3)
    sep = pd.DataFrame([["…"] * panel.shape[1]], columns=panel.columns, index=["…"])
    both = pd.concat([head.astype(object), sep, tail.astype(object)])
    both.index = [i if isinstance(i, str) else str(pd.Timestamp(i).date())
                  for i in both.index]
    return (T.df_html(both, max_rows=2 * n + 1)
            + f'<p class="small" style="color:#8b949e;margin-top:.4rem;">'
              f'{panel.shape[0]:,} Zeilen &times; {panel.shape[1]} Spalten · '
              f'{panel.index[0].date()} bis {panel.index[-1].date()}</p>'
            .replace(",", "."))


# ── Trades im Kurschart ─────────────────────────────────────────────────────
def _signal_shading(price: pd.Series, signal: pd.Series | None) -> go.Scatter | None:
    if signal is None:
        return None
    sig = signal.reindex(price.index).fillna(False).astype(bool).to_numpy()
    if not sig.any():
        return None
    top = float(np.nanmax(price.to_numpy(float))) * 1.06
    y = np.where(sig, top, np.nan)
    return go.Scatter(x=price.index, y=y, name="Signal aktiv", mode="lines",
                      line=dict(width=0), fill="tozeroy",
                      fillcolor="rgba(88,166,255,0.10)",
                      hoverinfo="skip", showlegend=True)


def trade_chart(price: pd.Series, trades: pd.DataFrame,
                signal: pd.Series | None = None,
                title: str = "Einstiege &amp; Ausstiege im Kursverlauf",
                overlays: dict[str, pd.Series] | None = None) -> go.Figure:
    """Kursreihe mit hinterlegtem Signalfenster und jedem einzelnen Trade."""
    px = pd.to_numeric(price, errors="coerce").dropna()
    fig = go.Figure()
    shade = _signal_shading(px, signal)
    if shade is not None:
        fig.add_trace(shade)
    fig.add_trace(go.Scatter(
        x=px.index, y=px.to_numpy(float), name=str(px.name or "Kurs"), mode="lines",
        line=dict(color="#c9d1d9", width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} USD<extra></extra>"))
    for i, (nm, s) in enumerate((overlays or {}).items()):
        s = s.reindex(px.index)
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(float), name=nm, mode="lines",
                                 line=dict(width=1.1, dash="dot",
                                           color=T.PAL[(i + 3) % len(T.PAL)])))

    if trades is not None and not trades.empty:
        tr = trades.copy()
        txt = [f"Entry {pd.Timestamp(e).date()} @ {ep:.2f}<br>"
               f"Exit {pd.Timestamp(x).date()} @ {xp:.2f}<br>"
               f"Rendite {r:+.2%} · {d} Tage · {g}"
               for e, x, ep, xp, r, d, g in zip(
                   tr["Entry"], tr["Exit"], tr["EntryPx"], tr["ExitPx"],
                   tr["Return"], tr["Tage"], tr["Grund"])]
        fig.add_trace(go.Scatter(
            x=tr["Entry"], y=tr["EntryPx"], name="Einstieg", mode="markers",
            marker=dict(symbol="triangle-up", size=9, color=GREEN,
                        line=dict(width=1, color="#0d1117")),
            text=txt, hovertemplate="<b>Einstieg</b><br>%{text}<extra></extra>"))
        for grund, col, sym in (("Stop-Loss", RED, "x"),
                                ("Signal-Exit", AMBER, "triangle-down"),
                                ("offen", BLUE, "circle-open")):
            m = tr["Grund"] == grund
            if not m.any():
                continue
            fig.add_trace(go.Scatter(
                x=tr.loc[m, "Exit"], y=tr.loc[m, "ExitPx"], name=f"Ausstieg ({grund})",
                mode="markers",
                marker=dict(symbol=sym, size=9, color=col,
                            line=dict(width=1, color="#0d1117")),
                text=[t for t, k in zip(txt, m) if k],
                hovertemplate="<b>Ausstieg</b><br>%{text}<extra></extra>"))

    fig.update_layout(title=title, hovermode="closest",
                      yaxis=dict(title="Kurs (USD)"),
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    return fig


def trade_return_bars(trades: pd.DataFrame,
                      title: str = "Rendite je Trade") -> go.Figure:
    fig = go.Figure()
    if trades is None or trades.empty:
        return fig
    tr = trades.reset_index(drop=True)
    fig.add_trace(go.Bar(
        x=tr["Entry"], y=tr["Return"] * 100,
        marker_color=[GREEN if v > 0 else RED for v in tr["Return"]],
        name="Trade-Rendite",
        customdata=np.stack([tr["Tage"], tr["Grund"]], axis=-1),
        hovertemplate=("Entry %{x|%Y-%m-%d}<br>%{y:.2f} %"
                       "<br>%{customdata[0]} Tage · %{customdata[1]}<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.update_layout(title=title, yaxis=dict(title="Rendite (%)"), bargap=0.1)
    return fig


# ── Equity ──────────────────────────────────────────────────────────────────
def _dd(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def equity_dashboard(curves: dict[str, pd.Series],
                     exposure: pd.Series | None = None,
                     trades: pd.DataFrame | None = None,
                     title: str = "Kapitalkurve, Drawdown und Investitionsgrad",
                     log: bool = True, roll: int = 126) -> tuple[go.Figure, int]:
    """Vier gestapelte Panels: Equity, Drawdown, Exposure, rollierende Sharpe."""
    curves = {k: pd.to_numeric(v, errors="coerce").dropna()
              for k, v in curves.items() if v is not None and len(v)}
    if not curves:
        return go.Figure(), 300
    has_exp = exposure is not None and len(exposure) > 0
    rows = 3 + int(has_exp)
    heights = [0.44, 0.20, 0.14, 0.22] if has_exp else [0.50, 0.24, 0.26]
    titles = ["Kapitalkurve (Start = 1)", "Drawdown"] + \
             (["Investitionsgrad"] if has_exp else []) + \
             [f"Rollierende Sharpe-Ratio ({roll} Handelstage)"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.045,
                        subplot_titles=titles)

    main = next(iter(curves))
    for i, (name, eq) in enumerate(curves.items()):
        eq = eq / eq.iloc[0]
        col = T.PAL[i % len(T.PAL)]
        dash = "solid" if name == main else "dot"
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.to_numpy(float), name=name, mode="lines",
            line=dict(color=col, width=2.0 if name == main else 1.3, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>"
                          f"%{{y:.3f}}&times;<extra></extra>"), row=1, col=1)
        d = _dd(eq)
        fig.add_trace(go.Scatter(
            x=d.index, y=d.to_numpy(float) * 100, name=f"DD {name}", mode="lines",
            line=dict(color=col, width=1.1, dash=dash),
            fill="tozeroy" if name == main else None,
            fillcolor=T.hex_rgba(col, 0.20), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} %<extra></extra>"), row=2, col=1)

        r = eq.pct_change()
        sd = r.rolling(roll).std()
        rs = (r.rolling(roll).mean() / sd.where(sd > 0)) * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.to_numpy(float), name=f"Sharpe {name}", mode="lines",
            line=dict(color=col, width=1.2, dash=dash), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe %{y:.2f}<extra></extra>"),
            row=rows, col=1)

    if has_exp:
        ex = exposure.reindex(curves[main].index).fillna(0.0)
        fig.add_trace(go.Scatter(
            x=ex.index, y=ex.to_numpy(float), name="Investitionsgrad", mode="lines",
            line=dict(width=0.8, color=BLUE), fill="tozeroy",
            fillcolor=T.hex_rgba(BLUE, 0.30), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0%}<extra></extra>"), row=3, col=1)
        fig.update_yaxes(title_text="Anteil", tickformat=".0%", row=3, col=1)

    if trades is not None and not trades.empty and main in curves:
        eq = curves[main] / curves[main].iloc[0]
        ent = eq.reindex(pd.DatetimeIndex(trades["Entry"])).dropna()
        fig.add_trace(go.Scatter(
            x=ent.index, y=ent.to_numpy(float), name="Einstieg", mode="markers",
            marker=dict(symbol="triangle-up", size=7, color=GREEN),
            hovertemplate="Einstieg %{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)

    fig.add_hline(y=0, line=dict(color=GREY, width=1), row=rows, col=1)
    fig.update_yaxes(title_text="Faktor", type="log" if log else "linear", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe", row=rows, col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(title=title, hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0))
    return fig, 260 + 150 * rows


def underwater_vs_benchmark(strat: pd.Series, bench: pd.Series,
                            title: str = "Relative Wertentwicklung") -> go.Figure:
    a = strat / strat.iloc[0]
    b = bench.reindex(a.index).ffill()
    b = b / b.iloc[0]
    rel = a / b
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rel.index, y=rel.to_numpy(float), mode="lines",
                             name="Strategie / Benchmark",
                             line=dict(color=BLUE, width=1.6),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=GREY, width=1, dash="dash"))
    fig.update_layout(title=title, yaxis=dict(title="Verhältnis", type="log"))
    return fig
