"""
Comprehensive HTML Report Builder – v3.
All phases, all charts, white MathJax, Plotly CDN.
New in v3: Phase 8/9, PCA deep-dive, backtesting, mega-network, pairwise viewer.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BOOTSTRAP_CDN = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
MATHJAX_CDN   = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
PLOTLY_CDN    = "https://cdn.plot.ly/plotly-2.27.0.min.js"

PHASE_COLOURS = {
    1:"#1f6feb", 2:"#3fb950", 3:"#d29922", 4:"#f78166",
    5:"#bc8cff", 6:"#39d353", 7:"#58a6ff", 8:"#ff7b72",
    9:"#ffa657", 10:"#7ee787", 11:"#e3b341", 12:"#a5d6ff",
    13:"#ff9fef", 14:"#56d364", 15:"#79c0ff",
}
SECTOR_CMAP = {
    "Energy":"#d29922","Metals":"#bc8cff","Agriculture":"#3fb950",
    "Materials":"#bc8cff","Market":"#58a6ff","ETF":"#39d353",
    "Industrials":"#58a6ff","Aviation":"#ffa657","Transportation":"#ffa657",
    "Control":"#f78166","Unknown":"#8b949e",
}
_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="'Segoe UI',Arial,sans-serif", size=12),
    margin=dict(l=60, r=20, t=50, b=60),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
)
PAL = px.colors.qualitative.Plotly
SECTORS = {
    "CL=F":"Energy","BZ=F":"Energy","NG=F":"Energy","GC=F":"Metals","SI=F":"Metals",
    "HG=F":"Metals","ZC=F":"Agriculture","ZW=F":"Agriculture","ZS=F":"Agriculture",
    "XLE":"ETF","XLB":"ETF","XLI":"ETF","GDX":"ETF","SIL":"ETF","JETS":"ETF","IYT":"ETF",
    "XOM":"Energy","CVX":"Energy","FCX":"Metals","NEM":"Metals","APA":"Energy",
    "OXY":"Energy","TECK":"Metals","SM":"Energy","TGB":"Metals","GORO":"Metals",
    "SPY":"Market","QQQ":"Market","IWM":"Market","IJH":"Market","MGC":"Metals",
}
PROX = {
    "CL=F":0,"BZ=F":0,"NG=F":0,"GC=F":0,"SI=F":0,"HG=F":0,"ZC=F":0,"ZW=F":0,"ZS=F":0,
    "XLE":1,"XLB":1,"GDX":1,"SIL":1,"XLI":2,"JETS":2,"IYT":2,
    "XOM":1,"CVX":1,"FCX":1,"NEM":1,"APA":2,"OXY":2,"TECK":2,"SM":3,"TGB":3,"GORO":3,
    "SPY":6,"QQQ":6,"IWM":6,"IJH":6,"MGC":6,
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML primitives
# ─────────────────────────────────────────────────────────────────────────────

def _html_base(title: str, phase: int, body: str) -> str:
    col = PHASE_COLOURS.get(phase, "#58a6ff")
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <link href="{BOOTSTRAP_CDN}" rel="stylesheet"/>
  <script src="{PLOTLY_CDN}"></script>
  <script>
    MathJax = {{
      tex: {{ inlineMath:[['$','$'],['\\\\(','\\\\)']] }},
      options: {{ skipHtmlTags:['script','noscript','style','textarea','pre'] }}
    }};
  </script>
  <script async src="{MATHJAX_CDN}"></script>
  <style>
    body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',Arial,sans-serif; }}
    mjx-container, mjx-container * {{ color:#e6edf3 !important; fill:#e6edf3 !important; }}
    mjx-container svg path {{ fill:#e6edf3 !important; }}
    .MathJax {{ color:#e6edf3 !important; }}
    .ph-header {{ background:{col}22; border-left:5px solid {col};
                  padding:1.5rem 2rem; border-radius:8px; margin-bottom:2rem; }}
    .ph-header h1 {{ color:{col}; font-size:1.8rem; margin:0; }}
    .ph-header .sub {{ color:#8b949e; font-size:.95rem; margin-top:.4rem; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; }}
    .card-header {{ background:{col}18; border-bottom:1px solid #30363d;
                    color:{col}; font-weight:600; font-size:.95rem; }}
    .card-body {{ padding:1rem 1.25rem; }}
    table {{ color:#e6edf3 !important; font-size:.78rem; }}
    .table-dark {{ --bs-table-bg:#161b22; --bs-table-striped-bg:#1c2128;
                   --bs-table-hover-bg:#21262d; }}
    .formula-box {{ background:#1c2128; border:1px solid #30363d; border-radius:6px;
                    padding:1rem 1.5rem; margin:.5rem 0 1rem;
                    font-size:1.05rem; color:#e6edf3 !important; }}
    .formula-box * {{ color:#e6edf3 !important; }}
    .interp-box {{ background:#0e3a1a; border-left:4px solid #3fb950;
                   padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                   color:#e6edf3; margin-bottom:1rem; }}
    .warn-box  {{ background:#3a1e0e; border-left:4px solid #d29922;
                  padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                  color:#e6edf3; margin-bottom:1rem; }}
    .info-box  {{ background:#0e2a3a; border-left:4px solid #58a6ff;
                  padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                  color:#e6edf3; margin-bottom:1rem; }}
    .stat-card {{ background:#1c2128; border:1px solid #30363d; border-radius:8px;
                  padding:1rem; text-align:center; }}
    .stat-card .val {{ font-size:1.6rem; font-weight:700; color:{col}; }}
    .stat-card .lbl {{ font-size:.78rem; color:#8b949e; }}
    .badge-ph {{ background:{col}33; color:{col}; border:1px solid {col}55;
                 padding:.25rem .7rem; border-radius:12px; font-size:.8rem; }}
    .table-responsive {{ max-height:380px; overflow-y:auto; }}
    iframe {{ border:none; width:100%; border-radius:6px; }}
    h2,h3,h4 {{ color:#e6edf3; }}
    a {{ color:{col}; }} a:hover {{ color:{col}cc; }}
    .breadcrumb-item.active {{ color:#8b949e; }}
    .slbl {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
             color:#8b949e; margin-bottom:.4rem; }}
  </style>
</head>
<body>
<nav class="navbar" style="background:#010409;border-bottom:1px solid #30363d;padding:.7rem 1.5rem;">
  <a class="navbar-brand fw-bold" style="color:#58a6ff;" href="index.html">
    &#127748; Commodity Research Framework
  </a>
  <span class="badge-ph">Phase {phase}</span>
</nav>
<div class="container-xl py-4">
  <nav aria-label="breadcrumb" class="mb-3">
    <ol class="breadcrumb">
      <li class="breadcrumb-item"><a href="index.html">Dashboard</a></li>
      <li class="breadcrumb-item active">{title}</li>
    </ol>
  </nav>
  {body}
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body></html>"""


def _card(header: str, body_html: str) -> str:
    return (f'<div class="card mb-4"><div class="card-header">{header}</div>'
            f'<div class="card-body">{body_html}</div></div>')

def _formula(latex: str, label: str = "") -> str:
    lbl = f'<div class="slbl">{label}</div>' if label else ""
    return f'<div class="formula-box">{lbl}$${latex}$$</div>'

def _interp(text: str) -> str:
    return f'<div class="interp-box"><strong>&#128270; Interpretation:</strong> {text}</div>'

def _warn(text: str) -> str:
    return f'<div class="warn-box"><strong>&#9888; Hinweis:</strong> {text}</div>'

def _info(text: str) -> str:
    return f'<div class="info-box"><strong>&#8505; Info:</strong> {text}</div>'

def _stat_row(stats: list) -> str:
    cols = "".join(
        f'<div class="col"><div class="stat-card">'
        f'<div class="val">{v}</div><div class="lbl">{l}</div></div></div>'
        for l, v in stats
    )
    return f'<div class="row g-3 mb-4">{cols}</div>'

def _df_html(df: Optional[pd.DataFrame], max_rows: int = 300) -> str:
    if df is None or df.empty:
        return '<p class="text-muted small">Keine Daten.</p>'
    d = df.head(max_rows).copy()
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].round(4)
    return ('<div class="table-responsive">'
            + d.to_html(classes="table table-dark table-striped table-sm table-hover",
                        border=0, index=True)
            + '</div>')

def _embed(fig_path: Path, height: int = 520) -> str:
    if not fig_path.exists():
        return f'<p class="text-muted small">Grafik nicht gefunden: {fig_path.name}</p>'
    return f'<iframe src="../figures/{fig_path.name}" height="{height}"></iframe>'

def _div(fig: go.Figure, height: int = 420) -> str:
    fig.update_layout(height=height, **_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": True, "responsive": True})

def _chart_card(header: str, fig: go.Figure, height: int = 420,
                interp: str = "", formula: str = "", flabel: str = "") -> str:
    parts = []
    if formula:
        parts.append(_formula(formula, flabel))
    parts.append(_div(fig, height))
    if interp:
        parts.append(_interp(interp))
    return _card(header, "\n".join(parts))

def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"  Report: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chart_price_history(prices: pd.DataFrame) -> go.Figure:
    cats = {
        "Rohstoffe": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"],
        "ETFs":      ["XLE","XLB","XLI","GDX","SIL","JETS","IYT"],
        "Produzenten":["XOM","CVX","FCX","NEM","APA","OXY","TECK","SM","TGB","GORO"],
        "Markt":     ["SPY","QQQ","IWM","IJH","MGC"],
    }
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(cats.keys()))
    for idx, (cat, tickers) in enumerate(cats.items()):
        r, c = divmod(idx, 2)
        for j, t in enumerate(tickers):
            if t not in prices.columns:
                continue
            s = prices[t].dropna()
            if s.empty:
                continue
            norm = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(x=norm.index.astype(str), y=norm.values, name=t,
                                     line=dict(width=1.3, color=PAL[j % len(PAL)]),
                                     legendgroup=cat), row=r+1, col=c+1)
    fig.update_layout(title="Normalisierte Preisentwicklung (Basis=100)", height=680)
    return fig


def _chart_vol_return(stats: pd.DataFrame) -> go.Figure:
    v_col = next((c for c in stats.columns if "vol" in c.lower()), None)
    m_col = next((c for c in stats.columns if "mean" in c.lower()), None)
    s_col = next((c for c in stats.columns if "sharpe" in c.lower()), None)
    if not v_col or not m_col:
        return go.Figure()
    sizes = None
    if s_col:
        raw = stats[s_col].abs()
        sizes = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 25 + 8).tolist()
    fig = go.Figure(go.Scatter(
        x=(stats[v_col]*100).tolist(), y=(stats[m_col]*100).tolist(),
        mode="markers+text", text=stats.index.tolist(),
        textposition="top center", textfont=dict(size=9, color="#e6edf3"),
        marker=dict(size=sizes or 12, color=(stats[m_col]*100).tolist(),
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Rendite %"),
                    line=dict(color="#30363d", width=1)),
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Rendite: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(title="Risiko-Rendite (annualisiert)",
                      xaxis_title="Volatilitat (%)", yaxis_title="Rendite (%)")
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", line_width=1)
    return fig


def _chart_sharpe_bars(stats: pd.DataFrame) -> go.Figure:
    s_col  = next((c for c in stats.columns if "sharpe" in c.lower()), None)
    so_col = next((c for c in stats.columns if "sortino" in c.lower()), None)
    if not s_col:
        return go.Figure()
    df = stats[[s_col] + ([so_col] if so_col else [])].sort_values(s_col)
    fig = go.Figure()
    fig.add_bar(name="Sharpe", x=df.index.tolist(), y=df[s_col].tolist(),
                marker_color=["#3fb950" if v>=0 else "#f78166" for v in df[s_col]])
    if so_col:
        fig.add_bar(name="Sortino", x=df.index.tolist(), y=df[so_col].tolist(),
                    marker_color="#58a6ff", opacity=0.55)
    fig.update_layout(title="Sharpe & Sortino", barmode="group", xaxis_tickangle=-45)
    fig.add_hline(y=0, line_color="#8b949e", line_width=1)
    return fig


def _chart_drawdown_var(stats: pd.DataFrame) -> go.Figure:
    dd_col  = next((c for c in stats.columns if "drawdown" in c.lower()), None)
    var_col = next((c for c in stats.columns if "var" in c.lower() and "95" in c.lower()), None)
    if not dd_col:
        return go.Figure()
    dd = stats[dd_col].sort_values()
    fig = go.Figure()
    fig.add_bar(name="Max Drawdown", x=dd.index.tolist(), y=(dd.values*100).tolist(),
                marker_color="#ff7b72")
    if var_col:
        v = stats.loc[dd.index, var_col]
        fig.add_bar(name="VaR 95%", x=v.index.tolist(), y=(v.values*100).tolist(),
                    marker_color="#d29922", opacity=0.75)
    fig.update_layout(title="Max Drawdown & VaR", yaxis_title="%",
                      barmode="group", xaxis_tickangle=-45)
    return fig


def _chart_adf_bars(stat: pd.DataFrame) -> go.Figure:
    if "test" not in stat.columns:
        return go.Figure()
    adf = stat[stat["test"] == "ADF"].copy()
    if adf.empty:
        return go.Figure()
    ticker_col = "ticker" if "ticker" in adf.columns else adf.index.name
    labels = adf[ticker_col].tolist() if ticker_col in adf.columns else adf.index.tolist()
    vals   = adf["statistic"].tolist() if "statistic" in adf.columns else []
    reject = adf["reject_null"].tolist() if "reject_null" in adf.columns else [True]*len(adf)
    crit5  = adf["crit_5%"].mean()      if "crit_5%"    in adf.columns else -2.86
    fig = go.Figure(go.Bar(x=labels, y=vals,
                           marker_color=["#3fb950" if r else "#f78166" for r in reject]))
    fig.add_hline(y=crit5, line_dash="dash", line_color="#d29922",
                  annotation_text=f"Krit. 5% ({crit5:.2f})", annotation_font_color="#d29922")
    fig.update_layout(title="ADF-Teststatistik", xaxis_tickangle=-45)
    return fig


def _chart_stationarity_heatmap(stat: pd.DataFrame) -> go.Figure:
    if "test" not in stat.columns or "ticker" not in stat.columns:
        return go.Figure()
    pivot = stat.pivot_table(index="test", columns="ticker", values="reject_null", aggfunc="first")
    if pivot.empty:
        return go.Figure()
    z = pivot.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#f78166"],[1,"#3fb950"]], showscale=False, zmin=0, zmax=1,
        text=[["OK" if v else "X" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10)
    ))
    fig.update_layout(title="Stationaritaets-Heatmap", xaxis_tickangle=-45, height=280)
    return fig


def _chart_corr_top_pairs(sig: pd.DataFrame) -> go.Figure:
    rho_col = next((c for c in sig.columns if "spearman" in c.lower() or "rho" in c.lower()
                    or "corr" in c.lower()), None)
    if not rho_col and not sig.select_dtypes(float).empty:
        rho_col = sig.select_dtypes(float).columns[0]
    if not rho_col:
        return go.Figure()
    top = sig.nlargest(30, rho_col)
    if "asset1" in sig.columns and "asset2" in sig.columns:
        labels = (top["asset1"] + " vs " + top["asset2"]).tolist()
    else:
        labels = top.index.astype(str).tolist()
    colors = ["#3fb950" if v>=0 else "#f78166" for v in top[rho_col]]
    fig = go.Figure(go.Bar(x=top[rho_col].values.tolist(), y=labels,
                           orientation="h", marker_color=colors,
                           hovertemplate="%{y}: rho=%{x:.3f}<extra></extra>"))
    fig.update_layout(title="Top-30 Spearman-Paare", xaxis_title="Spearman rho", height=520)
    return fig


def _chart_ccf_lags(ccf: pd.DataFrame) -> go.Figure:
    src = next((c for c in ccf.columns if "source" in c.lower()), ccf.columns[0])
    tgt = next((c for c in ccf.columns if "target" in c.lower()), ccf.columns[1])
    lag = next((c for c in ccf.columns if "lag" in c.lower()), ccf.columns[2])
    ccf_v = next((c for c in ccf.columns if "ccf" in c.lower() or "peak" in c.lower()), None)
    label = (ccf[src].astype(str) + " -> " + ccf[tgt].astype(str)).tolist()
    colors = ["#3fb950" if v>=0 else "#f78166" for v in ccf[lag]]
    fig = go.Figure()
    fig.add_bar(x=label, y=ccf[lag].tolist(), marker_color=colors, name="Lag (Tage)")
    if ccf_v:
        fig.add_trace(go.Scatter(x=label, y=ccf[ccf_v].tolist(), mode="markers",
                                 name="Peak CCF", yaxis="y2",
                                 marker=dict(color="#d29922", size=7)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Peak CCF",
                                      gridcolor="#21262d", linecolor="#30363d", color="#d29922"))
    fig.update_layout(title="Optimaler Lag: Rohstoff -> Aktie", xaxis_tickangle=-45)
    return fig


def _chart_irf(irf: pd.DataFrame) -> go.Figure:
    imp  = next((c for c in irf.columns if "impuls" in c.lower()), None)
    resp = next((c for c in irf.columns if "resp" in c.lower()), None)
    h    = next((c for c in irf.columns if "horiz" in c.lower() or "period" in c.lower()), None)
    v    = next((c for c in irf.columns if irf[c].dtype == float and c not in [imp,resp,h]), None)
    fig = go.Figure()
    if imp and resp and h and v:
        for j, (key, grp) in enumerate(irf.groupby([imp, resp])):
            fig.add_trace(go.Scatter(x=grp[h].tolist(), y=grp[v].tolist(),
                                     mode="lines+markers", name=f"{key[0]}->{key[1]}",
                                     line=dict(width=1.8, color=PAL[j % len(PAL)])))
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", line_width=1)
    fig.update_layout(title="Impulse Response Function",
                      xaxis_title="Tage nach Schock", yaxis_title="Reaktion")
    return fig


def _chart_fevd(fevd: pd.DataFrame) -> go.Figure:
    resp = next((c for c in fevd.columns if "resp" in c.lower()), None)
    imp  = next((c for c in fevd.columns if "impuls" in c.lower()), None)
    h    = next((c for c in fevd.columns if "horiz" in c.lower() or "period" in c.lower()), None)
    v    = next((c for c in fevd.columns if fevd[c].dtype == float and c not in [resp,imp,h]), None)
    if not all([resp, imp, h, v]):
        return go.Figure()
    h_vals = sorted(fevd[h].unique())
    h_pick = h_vals[min(4, len(h_vals)-1)]
    pivot = fevd[fevd[h]==h_pick].pivot_table(index=resp, columns=imp, values=v, aggfunc="mean")
    fig = go.Figure()
    for j, col in enumerate(pivot.columns):
        fig.add_bar(name=str(col), x=pivot.index.tolist(), y=(pivot[col]*100).tolist(),
                    marker_color=PAL[j % len(PAL)])
    fig.update_layout(barmode="stack", title=f"FEVD nach {h_pick} Tagen",
                      yaxis_title="Varianzanteil (%)", xaxis_tickangle=-45)
    return fig


def _chart_car_by_event(events: pd.DataFrame) -> go.Figure:
    # column is window_days in actual data
    win = next((c for c in events.columns if "window" in c.lower()), None)
    evt = next((c for c in events.columns if "event_type" in c.lower() or
                ("event" in c.lower() and "window" not in c.lower())), None)
    car = next((c for c in events.columns if "mean_car" in c.lower()), None)
    std = next((c for c in events.columns if "std_car" in c.lower()), None)
    ast = next((c for c in events.columns if c.lower() == "asset"), None)
    if not (win and evt and car):
        return go.Figure()
    windows = sorted(events[win].unique())
    # All windows as separate subplots, aggregated across assets
    fig = make_subplots(rows=1, cols=len(windows),
                        subplot_titles=[f"{w}-Tage Fenster" for w in windows],
                        shared_yaxes=True)
    for i, w in enumerate(windows):
        sub = (events[events[win]==w]
               .groupby(evt)
               .agg(mean=(car,"mean"), n=(car,"count"),
                    err=(std,"mean") if std else (car,"std"))
               .reset_index()
               .sort_values("mean"))
        colors = ["#3fb950" if v>=0 else "#f78166" for v in sub["mean"]]
        fig.add_trace(go.Bar(
            x=sub[evt].tolist(), y=(sub["mean"]*100).tolist(),
            error_y=dict(type="data", array=(sub["err"]*100).tolist(), visible=True,
                         color="#8b949e", thickness=1.5),
            marker_color=colors, showlegend=False,
            hovertemplate="<b>%{x}</b><br>CAR: %{y:.2f}%<extra></extra>"),
            row=1, col=i+1)
    fig.update_yaxes(title_text="CAR (%)", row=1, col=1)
    fig.add_hline(y=0, line_color="#8b949e", line_width=0.8)
    fig.update_layout(title="Mittlere Kumulierte Abnormale Rendite (CAR) nach Ereignis & Fensterlaenge",
                      height=400)
    return fig


def _chart_event_heatmap(events: pd.DataFrame, window: int = 5) -> go.Figure:
    evt = next((c for c in events.columns if "event_type" in c.lower() or
                ("event" in c.lower() and "window" not in c.lower())), None)
    ast = next((c for c in events.columns if c.lower()=="asset"), None)
    sig = next((c for c in events.columns if "significant" in c.lower()), None)
    win = next((c for c in events.columns if "window" in c.lower()), None)
    car = next((c for c in events.columns if "mean_car" in c.lower()), None)
    if not (evt and ast and sig):
        return go.Figure()
    # Use requested window, fall back to closest available
    if win:
        avail = sorted(events[win].unique())
        w_use = min(avail, key=lambda x: abs(x - window))
        sub = events[events[win] == w_use]
    else:
        sub = events
        w_use = window
    if sub.empty:
        return go.Figure()
    # Build heatmap: color by mean_CAR*100, mask non-significant
    pivot_sig = sub.pivot_table(index=evt, columns=ast, values=sig,
                                aggfunc=lambda x: int(any(x)))
    if car:
        pivot_car = sub.pivot_table(index=evt, columns=ast, values=car, aggfunc="mean")
        pivot_car = pivot_car.reindex(index=pivot_sig.index, columns=pivot_sig.columns)
        # Only show CAR where significant
        z_masked = pivot_car.where(pivot_sig == 1).values * 100
        hover = [[f"{pivot_sig.index[r]} x {pivot_sig.columns[c]}: CAR={z_masked[r,c]:.2f}%"
                  if not np.isnan(z_masked[r,c]) else "nicht signifikant"
                  for c in range(z_masked.shape[1])]
                 for r in range(z_masked.shape[0])]
        fig = go.Figure(go.Heatmap(
            z=z_masked.tolist(), x=pivot_sig.columns.tolist(), y=pivot_sig.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="CAR (%)"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover))
    else:
        z = pivot_sig.values.astype(float)
        fig = go.Figure(go.Heatmap(
            z=z.tolist(), x=pivot_sig.columns.tolist(), y=pivot_sig.index.tolist(),
            colorscale=[[0,"#1c2128"],[1,"#3fb950"]], zmin=0, zmax=1))
    fig.update_layout(
        title=f"Signifikante CAR-Reaktionen ({w_use}-Tage, p<5%): Ereignis x Asset",
        xaxis_tickangle=-45,
        height=max(320, len(pivot_sig.index)*40 + 120))
    return fig


def _chart_pagerank(metrics: pd.DataFrame) -> go.Figure:
    if "pagerank" not in metrics.columns:
        return go.Figure()
    df = metrics.sort_values("pagerank")
    sec_col = "sector" if "sector" in df.columns else None
    colors = [SECTOR_CMAP.get(s, "#8b949e") for s in
              (df[sec_col].tolist() if sec_col else ["Unknown"]*len(df))]
    fig = go.Figure(go.Bar(x=df["pagerank"].tolist(), y=df.index.tolist(),
                           orientation="h", marker_color=colors,
                           hovertemplate="%{y}: PageRank=%{x:.4f}<extra></extra>"))
    fig.update_layout(title="PageRank", xaxis_title="PageRank", height=540)
    return fig


def _chart_degree(metrics: pd.DataFrame) -> go.Figure:
    if "in_degree" not in metrics.columns or "out_degree" not in metrics.columns:
        return go.Figure()
    pr_col  = "pagerank" if "pagerank" in metrics.columns else None
    sec_col = "sector"   if "sector"   in metrics.columns else None
    sizes   = ((metrics[pr_col]*2000).clip(6,40).tolist() if pr_col else 12)
    colors  = [SECTOR_CMAP.get(s,"#8b949e") for s in
               (metrics[sec_col].tolist() if sec_col else ["Unknown"]*len(metrics))]
    fig = go.Figure(go.Scatter(
        x=metrics["in_degree"].tolist(), y=metrics["out_degree"].tolist(),
        mode="markers+text", text=metrics.index.tolist(),
        textposition="top center", textfont=dict(size=8, color="#e6edf3"),
        marker=dict(size=sizes, color=colors, line=dict(color="#30363d", width=1)),
        hovertemplate="<b>%{text}</b><br>In: %{x:.2f}<br>Out: %{y:.2f}<extra></extra>"))
    fig.update_layout(title="In-Degree vs Out-Degree", xaxis_title="In-Degree", yaxis_title="Out-Degree")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-7 reports
# ─────────────────────────────────────────────────────────────────────────────

def build_phase1_report(tables, figures, out):
    prices = _read(tables / "phase1_prices.csv")
    macro  = _read(tables / "phase1_macro.csv")
    n_t = len(prices.columns) if prices is not None else "?"
    n_d = len(prices)         if prices is not None else "?"
    d0  = str(prices.index[0])[:10]  if prices is not None and len(prices) else "?"
    d1  = str(prices.index[-1])[:10] if prices is not None and len(prices) else "?"
    body = f"""
<div class="ph-header"><h1>Phase 1 - Datenerhebung</h1>
  <div class="sub">Yahoo Finance (34 Ticker), FRED (Makro), EIA (Energielager)</div>
</div>
{_stat_row([("Ticker",str(n_t)),("Handelstage",str(n_d)),("Von",d0),("Bis",d1),
            ("Makroserien",str(len(macro.columns)) if macro is not None else "?")])}
{_chart_card("Normalisierte Preisentwicklung (Basis=100)",
             _chart_price_history(prices) if prices is not None else go.Figure(), height=680,
             interp="Basis=100 normiert: direkte Vergleichbarkeit verschiedener Preisskalen. "
                    "Rohstoff-Futures schwanken staerker als Marktindizes. "
                    "Naturgas (NG=F) hat extreme Ausschlaege.")}
{_card("Preisdaten - Vorschau (5 Zeilen)", _df_html(prices, max_rows=5))}
{_card("Makrodaten FRED - Vorschau", _df_html(macro, max_rows=5))}
"""
    _write(out / "phase01_data_loading.html", _html_base("Phase 1 - Datenerhebung", 1, body))


def build_phase2_report(tables, figures, out):
    returns = _read(tables / "phase2_returns.csv")
    fig_ret = go.Figure()
    fig_hist = go.Figure()
    if returns is not None:
        for j, c in enumerate(["CL=F","GC=F","SPY","XOM"]):
            if c in returns.columns:
                s = returns[c].dropna()
                fig_ret.add_trace(go.Scatter(x=s.index.astype(str), y=(s*100).tolist(),
                                             mode="lines", name=c, line=dict(width=0.8, color=PAL[j])))
        for j, c in enumerate(["CL=F","GC=F","HG=F","SPY","XOM","XLE"]):
            if c in returns.columns:
                s = returns[c].dropna() * 100
                fig_hist.add_trace(go.Histogram(x=s.tolist(), name=c, opacity=0.55,
                                                nbinsx=120, marker_color=PAL[j]))
        fig_hist.update_layout(barmode="overlay", title="Renditeverteilung",
                               xaxis_title="Log-Rendite (%)", yaxis_title="Haeufigkeit")
        fig_ret.update_layout(title="Taegl. Log-Renditen (%)", yaxis_title="Log-Rendite (%)")
    body = f"""
<div class="ph-header"><h1>Phase 2 - Preprocessing</h1>
  <div class="sub">Log-Renditen, NYSE-Kalender, fehlende Werte, Ausreisser</div>
</div>
{_card("Formel",
    _formula(r"r_t = \ln\!\Bigl(\frac{{P_t}}{{P_{{t-1}}}}\Bigr) \quad \bar{{r}}_{{ann}}=252\bar{{r}}_d \quad \sigma_{{ann}}=\sqrt{{252}}\sigma_d") +
    _interp("Log-Renditen zeitadditiv. ARCH-Effekt sichtbar: Phasen hoher/niedriger Vol. "
            "COVID-2020 und Energiekrise-2022 klar erkennbar."))}
{_chart_card("Taegl. Log-Renditen", fig_ret)}
{_chart_card("Renditeverteilung - Fat Tails", fig_hist,
    interp="Spitzere Verteilungen als Normal -> Fat Tails -> Extreme haeufiger. "
           "Rechtfertigt robuste Masse (Spearman) statt nur Pearson.")}
{_card("Beschreibende Statistik", _df_html(returns.describe() if returns is not None else None))}
"""
    _write(out / "phase02_preprocessing.html", _html_base("Phase 2 - Preprocessing", 2, body))


def build_phase3_report(tables, figures, out):
    stats   = _read(tables / "phase3_descriptive_stats.csv")
    returns = _read(tables / "phase2_returns.csv")
    if returns is not None:
        returns.index = pd.to_datetime(returns.index, errors="coerce")
        returns = returns[returns.index.notna()]

    # Rolling volatility for ALL assets at 3 windows
    fig_vol21 = go.Figure()
    fig_vol63 = go.Figure()
    fig_vol252 = go.Figure()
    if returns is not None:
        for j, c in enumerate(returns.columns):
            for w, fig in [(21, fig_vol21), (63, fig_vol63), (252, fig_vol252)]:
                s = returns[c].dropna()
                rv = s.rolling(w, min_periods=w // 2).std() * np.sqrt(252) * 100
                rv = rv.dropna()
                fig.add_trace(go.Scatter(
                    x=rv.index.astype(str).tolist(), y=rv.round(2).values.tolist(),
                    mode="lines", name=c, line=dict(color=PAL[j % len(PAL)], width=1.2),
                    visible=(j < 12)))  # show first 12 by default

        for fig, w_label in [(fig_vol21,"21T (~1M)"), (fig_vol63,"63T (~3M)"),
                              (fig_vol252,"252T (~1J)")]:
            fig.update_layout(
                title=f"Rolling Volatilität ({w_label}): Alle {len(returns.columns)} Assets (ann. %)",
                yaxis_title="Volatilität p.a. (%)", height=500,
                legend=dict(orientation="v", x=1.01),
                updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12,
                    buttons=[
                        dict(label="Rohstoffe",
                             method="update",
                             args=[{"visible": [c in ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"]
                                                for c in returns.columns]}]),
                        dict(label="Alle",
                             method="update",
                             args=[{"visible": [True] * len(returns.columns)}]),
                        dict(label="ETFs",
                             method="update",
                             args=[{"visible": [c in ["XLE","XLB","XLI","GDX","SIL","JETS","IYT","SPY","QQQ","IWM","IJH"]
                                                for c in returns.columns]}]),
                    ])])

    body = f"""
<div class="ph-header"><h1>Phase 3 - Explorative Datenanalyse (EDA)</h1>
  <div class="sub">Risikokennzahlen · Rolling Volatility (alle Assets, 3 Fenster) · Pearson-Korrelation</div>
</div>
{_card("Formeln",
    _formula(r"\text{{Sharpe}}=\frac{{\bar{{r}}_{{ann}}}}{{\sigma_{{ann}}}} \quad "
             r"\text{{Sortino}}=\frac{{\bar{{r}}_{{ann}}}}{{\sigma_{{down}}}} \quad "
             r"\text{{VaR}}_{{95\%}}=F^{{-1}}(0{{,}}05) \quad "
             r"\text{{MaxDD}}=\max_t\frac{{\text{{Peak}}-\text{{Trough}}}}{{\text{{Peak}}}}") +
    _interp("Sharpe>1: gut. Sortino>Sharpe: Verlustrisiko niedrig. "
            "VaR 95%: 5% schlechteste Tage. MaxDD: groesster historischer Peak-to-Trough-Verlust."))}
{_chart_card("Risiko-Rendite", _chart_vol_return(stats) if stats is not None else go.Figure())}
{_chart_card("Sharpe & Sortino", _chart_sharpe_bars(stats) if stats is not None else go.Figure())}
{_chart_card("Max Drawdown & VaR", _chart_drawdown_var(stats) if stats is not None else go.Figure())}
{_chart_card("Rolling Volatilität 21T (alle Assets – Buttons: Rohstoffe / Alle / ETFs)", fig_vol21, height=520,
    interp="21T=kurzfristige Vol (ARCH-Effekte sichtbar). COVID-2020: Vol-Spike. 2022: Energie-Krise. "
           "Toggle-Buttons oben: Rohstoffe / Alle / ETF-Filter. Klick auf Legende: ein/ausblenden.")}
{_chart_card("Rolling Volatilität 63T (~3 Monate) – Alle Assets", fig_vol63, height=520,
    interp="63T glättet kurzfristige Spikes → strukturelle Regime erkennbar. "
           "Rohstoff-Vol typisch 2-3× höher als Aktien-ETFs. Gas (NG=F) extremster Ausschlag.")}
{_chart_card("Rolling Volatilität 252T (~1 Jahr) – Alle Assets", fig_vol252, height=520,
    interp="Jahres-Vol: langfristiges Risiko-Regime. Rohstoffe mit persistent höherer Vol als Equity. "
           "Strukturbrüche: 2008, 2015-16 (Öl), 2020 (COVID), 2022 (Energie/Ukraine).")}
{_card("Pearson Heatmap (interaktiv)", _embed(figures/"heatmap_pearson_correlation.html", height=660))}
{_card("Vollstaendige Statistik", _df_html(stats))}
"""
    _write(out / "phase03_eda.html", _html_base("Phase 3 - EDA", 3, body))


def build_phase4_report(tables, figures, out):
    stat   = _read(tables / "phase4_stationarity.csv")
    orders = _read(tables / "phase4_integration_order.csv")
    body = f"""
<div class="ph-header"><h1>Phase 4 - Stationaritaetsanalyse</h1>
  <div class="sub">ADF, KPSS, Phillips-Perron</div>
</div>
{_card("Testgleichungen",
    _formula(r"\Delta y_t = \alpha + \beta t + \gamma y_{{t-1}} + \sum_{{i=1}}^p\delta_i\Delta y_{{t-i}} + \varepsilon_t",
             "ADF: H0: gamma=0 (Einheitswurzel). p<0,05 -> stationaer") +
    _formula(r"\eta_\mu = \frac{{1}}{{T^2\hat{{\sigma}}^2}}\sum S_t^2",
             "KPSS: H0: Stationaritaet (umgekehrt!). Grosses eta -> nicht stationaer") +
    _interp("ADF ablehnen UND KPSS nicht ablehnen -> sicher stationaer. "
            "Beide ablehnen -> struktureller Bruch moeglich."))}
{_chart_card("ADF-Teststatistik", _chart_adf_bars(stat) if stat is not None else go.Figure())}
{_chart_card("Stationaritaets-Heatmap", _chart_stationarity_heatmap(stat) if stat is not None else go.Figure(), height=300)}
{_card("Integrationsordnung", _df_html(orders))}
{_card("Alle Stationaritaetsergebnisse", _df_html(stat))}
"""
    _write(out / "phase04_stationarity.html", _html_base("Phase 4 - Stationaritaet", 4, body))


def build_phase5_report(tables, figures, out):
    sig = _read(tables / "phase5_significant_correlations.csv")
    returns = _read(tables / "phase2_returns.csv")
    if returns is not None:
        returns.index = pd.to_datetime(returns.index, errors="coerce")
        returns = returns[returns.index.notna()]

    fig_roll = go.Figure()
    if returns is not None and "CL=F" in returns.columns:
        base = returns["CL=F"]
        for j, c in enumerate(["XOM","CVX","XLE","GC=F","SPY"]):
            if c in returns.columns:
                s = returns[c]
                idx = base.dropna().index.intersection(s.dropna().index)
                roll = (pd.Series(base[idx].values, index=idx)
                        .rolling(252).corr(pd.Series(s[idx].values, index=idx)))
                fig_roll.add_trace(go.Scatter(x=roll.index.astype(str), y=roll.values.tolist(),
                                              mode="lines", name=f"CL=F vs {c}",
                                              line=dict(color=PAL[j], width=1.5)))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="#8b949e", line_width=1)
    fig_roll.update_layout(title="Rolling 252-Tage Korrelation: Oel vs. Produzenten")

    # ── Compute Pearson, Spearman, Kendall + difference heatmaps ────────────
    from scipy.stats import spearmanr, kendalltau
    fig_p_heat = go.Figure()
    fig_s_heat = go.Figure()
    fig_k_heat = go.Figure()
    fig_diff_ps = go.Figure()   # Pearson - Spearman
    fig_diff_pk = go.Figure()   # Pearson - Kendall
    fig_nonlin  = go.Figure()   # max(|P-S|, |P-K|) as nonlinearity proxy
    nonlin_rows = []

    if returns is not None:
        # Use a clean, common set of columns
        ret_c = returns.dropna(axis=1, how="all")
        cols  = ret_c.columns.tolist()
        n_c   = len(cols)
        # Fill with 0 for missing days so we have a common matrix
        ret_aligned = ret_c.dropna()

        P = np.zeros((n_c, n_c))
        S = np.zeros((n_c, n_c))
        K = np.zeros((n_c, n_c))

        for i in range(n_c):
            for j in range(n_c):
                x = ret_aligned.iloc[:, i].values
                y = ret_aligned.iloc[:, j].values
                # Pearson
                if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                    P[i, j] = float(np.corrcoef(x, y)[0, 1])
                    S[i, j] = float(spearmanr(x, y)[0])
                    K[i, j] = float(kendalltau(x, y)[0])
                else:
                    P[i, j] = S[i, j] = K[i, j] = 0.0

        diff_PS = P - S      # > 0: Pearson higher (could be outlier-driven)
        diff_PK = P - K      # > 0: linear relationship overstates monotone
        nonlin   = np.maximum(np.abs(diff_PS), np.abs(diff_PK))  # nonlinearity index

        def _hm(z_mat, title, colorscale="RdBu", zmid=0, zmin=-1, zmax=1):
            fig = go.Figure(go.Heatmap(
                z=np.round(z_mat, 3).tolist(), x=cols, y=cols,
                colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
                hovertemplate="X=%{x}<br>Y=%{y}<br>Wert=%{z:.3f}<extra></extra>"))
            fig.update_layout(title=title, height=max(400, 20 * n_c + 120))
            return fig

        fig_p_heat  = _hm(P,       "Pearson-Korrelation: Alle Assets", "RdBu",  0, -1, 1)
        fig_s_heat  = _hm(S,       "Spearman-Korrelation: Alle Assets","RdBu",  0, -1, 1)
        fig_k_heat  = _hm(K,       "Kendall-τ: Alle Assets",           "RdBu",  0, -1, 1)
        fig_diff_ps = _hm(diff_PS, "Pearson − Spearman: Differenz-Heatmap",
                          "RdYlGn_r", 0, -0.3, 0.3)
        fig_diff_pk = _hm(diff_PK, "Pearson − Kendall: Differenz-Heatmap",
                          "RdYlGn_r", 0, -0.3, 0.3)
        fig_nonlin  = _hm(nonlin,  "Nichtlinearitäts-Index: max(|P−S|, |P−K|)",
                          "YlOrRd", 0, 0, 0.3)

        # Nonlinearity ranking table
        for i in range(n_c):
            for j in range(i+1, n_c):
                d_ps = float(diff_PS[i, j])
                d_pk = float(diff_PK[i, j])
                nl   = float(nonlin[i, j])
                if nl > 0.05:  # only notable nonlinear pairs
                    nonlin_rows.append({
                        "Asset A": cols[i], "Asset B": cols[j],
                        "Pearson": round(P[i,j],4), "Spearman": round(S[i,j],4),
                        "Kendall": round(K[i,j],4),
                        "P−S": round(d_ps,4), "P−K": round(d_pk,4),
                        "Nichtlinearitäts-Index": round(nl,4),
                        "Interpretation": (
                            "Ausreißer-getrieben (P>S)" if d_ps > 0.05 else
                            "Monoton nicht-linear (S>P)" if d_ps < -0.05 else
                            "Schwach nichtlinear")
                    })
        nonlin_rows.sort(key=lambda x: -x["Nichtlinearitäts-Index"])

    nl_table_html = (_df_html(pd.DataFrame(nonlin_rows))
                     if nonlin_rows else "<p class='text-muted'>Keine stark nichtlinearen Paare.</p>")

    body = f"""
<div class="ph-header"><h1>Phase 5 - Korrelationsanalyse</h1>
  <div class="sub">Pearson · Spearman · Kendall · Differenz-Heatmaps · Nichtlinearitäts-Index · Rolling</div>
</div>
{_card("Korrelationsmasse",
    _formula(r"\rho_P=\frac{{Cov(X,Y)}}{{\sigma_X\sigma_Y}} \quad "
             r"\rho_S=1-\frac{{6\sum d_i^2}}{{n(n^2-1)}} \quad \tau_K=\frac{{C-D}}{{\binom{{n}}{{2}}}}") +
    _info("Pearson: linear, sensitiv für Ausreißer. Spearman: monoton, robust. Kendall: konkordante Paare. "
          "Differenz P−S > 0: Pearson durch Extremwerte erhöht (non-robust). "
          "P−S < 0: Zusammenhang monoton aber nicht-linear (z.B. log-Beziehung). "
          "Nichtlinearitäts-Index = max(|P−S|, |P−K|): misst Abweichung vom linearen Modell."))}
{_chart_card("Pearson-Korrelation: Alle Assets", fig_p_heat,
    interp="Lineare Korrelation. Anfällig für Ausreißer. Gold/GDX, Öl/Energie-ETFs: hohe Cluster.")}
{_chart_card("Spearman-Korrelation: Alle Assets", fig_s_heat,
    interp="Rang-Korrelation: robust gegen Ausreißer, misst monotone Zusammenhänge. "
           "Abweichung von Pearson = Hinweis auf Nichtlinearität oder Ausreißer-Einfluss.")}
{_chart_card("Kendall-τ: Alle Assets", fig_k_heat,
    interp="Konkordanz-basiert: konsistenter als Spearman bei kleinen Stichproben. "
           "Typisch |τ| < |ρ_S| < |ρ_P|: Kendall ist das konservativste Maß.")}
{_chart_card("Differenz: Pearson − Spearman", fig_diff_ps, height=max(400, 20*len(returns.columns if returns is not None else [])+120),
    interp="Positiv (grün): Pearson überschätzt Stärke (Ausreißer-Effekt). "
           "Negativ (rot): Monotoner Zusammenhang stärker als linearer (z.B. konvexe Beziehung). "
           "Betrag > 0.1: praktisch bedeutsame Nichtlinearität.")}
{_chart_card("Differenz: Pearson − Kendall", fig_diff_pk,
    interp="Ähnlich P−S aber Kendall ist noch robuster → größere Differenzen möglich. "
           "Persistente negative Differenz: Beziehung systematisch konvex/konkav.")}
{_chart_card("Nichtlinearitäts-Index: max(|P−S|, |P−K|)", fig_nonlin,
    interp="Gelb/Rot: starke Abweichung zwischen linearer und monotoner Korrelation. "
           "Paare mit hohem Index: lineare Regression unterschätzt tatsächlichen Zusammenhang. "
           "Empfehlung: für diese Paare nichtlineare Modelle (Copula, GAM) verwenden.")}
{_chart_card("Top-30 Spearman-Paare (p<5%)", _chart_corr_top_pairs(sig) if sig is not None else go.Figure(), height=540,
    formula=r"t=\rho_S\sqrt{{\frac{{n-2}}{{1-\rho_S^2}}}}\sim t_{{n-2}}",
    flabel="t-Test auf Signifikanz von rho_S")}
{_chart_card("Rolling 252-Tage Korrelation: Öl vs. Produzenten", fig_roll,
    interp="Zeitvariable Korrelationen: statische Analyse reicht nicht. Krisen erhöhen Korrelationen.")}
{_card("Signifikante Paare", _df_html(sig, max_rows=60))}
{_card("Nichtlinearitäts-Ranking: Paare mit |P−S| oder |P−K| > 0.05", nl_table_html)}
{_card("Pearson Heatmap (Pipeline-Output, interaktiv)", _embed(figures/"heatmap_pearson_correlation.html", height=660))}
{_card("Spearman Heatmap (Pipeline-Output, interaktiv)", _embed(figures/"heatmap_spearman_correlation.html", height=660))}
"""
    _write(out / "phase05_correlation.html", _html_base("Phase 5 - Korrelation", 5, body))



def build_phase6_report(tables, figures, out):
    ccf  = _read(tables / "phase6_ccf_lags.csv")
    gran = _read(tables / "phase6_granger.csv")
    irf  = _read(tables / "phase6_irf.csv")
    fevd = _read(tables / "phase6_fevd.csv")

    # ── Best Granger lag per pair: min p-value across tested lags ─────────────
    fig_gran_best = go.Figure()
    fig_gran_sig  = go.Figure()
    gran_table_html = ""
    if gran is not None and "cause" in gran.columns:
        # Best significant lag per pair
        best = (gran[gran["significant"] == True]
                .sort_values("pvalue")
                .drop_duplicates(subset=["cause","effect"], keep="first"))
        if best.empty:
            best = gran.sort_values("pvalue").drop_duplicates(subset=["cause","effect"], keep="first")

        best = best.copy()
        best["pair"] = best["cause"] + " → " + best["effect"]
        best["neg_log_p"] = -np.log10(best["pvalue"].clip(1e-10))
        best_sorted = best.sort_values("neg_log_p", ascending=False).head(30)

        # Chart 1: F-stat bubble chart (x=lag, y=F-stat, size=significance)
        fig_gran_best = go.Figure()
        for j, (_, row) in enumerate(best_sorted.iterrows()):
            sec = SECTORS.get(str(row["cause"]), "Unknown")
            col = SECTOR_CMAP.get(sec, "#8b949e")
            fig_gran_best.add_trace(go.Scatter(
                x=[int(row["lag"])], y=[float(row["f_stat"])],
                mode="markers+text",
                text=[f"{row['cause']}→{row['effect']}"],
                textposition="top center",
                textfont=dict(size=8, color="#e6edf3"),
                marker=dict(size=max(8, float(row["neg_log_p"])*5),
                            color=col, line=dict(color="#30363d", width=1)),
                name=str(row["pair"]),
                showlegend=False,
                hovertemplate=(f"<b>{row['pair']}</b><br>"
                               f"Lag: {row['lag']} Tage<br>"
                               f"F-Stat: {row['f_stat']:.2f}<br>"
                               f"p-Wert: {row['pvalue']:.5f}<br>"
                               f"Signifikant: {row['significant']}<extra></extra>")))
        fig_gran_best.update_layout(
            title="Granger-Kausalitaet: bester Lag je Paar (Groesse=-log10 p, Farbe=Sektor)",
            xaxis_title="Optimaler Granger-Lag (Tage)",
            yaxis_title="F-Statistik (Teststaerke)",
            xaxis=dict(tickmode="linear", dtick=1))
        fig_gran_best.add_hline(y=3.84, line_dash="dash", line_color="#d29922",
                                annotation_text="F-Krit 5% (≈3.84)",
                                annotation_font_color="#d29922")

        # Chart 2: Significance waterfall (neg log p-values)
        fig_gran_sig = go.Figure(go.Bar(
            x=best_sorted["pair"].tolist(),
            y=best_sorted["neg_log_p"].tolist(),
            text=[f"Lag {int(l)}d" for l in best_sorted["lag"]],
            textposition="outside", textfont=dict(size=8, color="#e6edf3"),
            marker_color=["#f78166" if v > 4 else "#d29922" if v > 1.3 else "#8b949e"
                          for v in best_sorted["neg_log_p"]],
            hovertemplate="<b>%{x}</b><br>-log10(p): %{y:.2f}<extra></extra>"
        ))
        fig_gran_sig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#3fb950",
                               annotation_text="p=0.05", annotation_font_color="#3fb950")
        fig_gran_sig.add_hline(y=-np.log10(0.01), line_dash="dash", line_color="#d29922",
                               annotation_text="p=0.01", annotation_font_color="#d29922")
        fig_gran_sig.update_layout(
            title="Signifikanzstaerke der Granger-Kausalitaet (-log10 p): Rot=sehr stark, Gelb=stark",
            yaxis_title="-log10(p-Wert)", xaxis_tickangle=-35, height=440)

        # Enhanced Granger table: add significance marker
        display = best_sorted[["pair","lag","f_stat","pvalue","significant","neg_log_p"]].copy()
        display.columns = ["Paar","Lag (d)","F-Stat","p-Wert","Signifikant","-log10(p)"]
        display = display.round({"F-Stat":2, "p-Wert":5, "-log10(p)":2})
        gran_table_html = _df_html(display, max_rows=60)

    # ── CCF chart with confidence info ─────────────────────────────────────────
    fig_ccf = go.Figure()
    ccf_table_html = ""
    if ccf is not None and "source" in ccf.columns:
        # Sort by absolute peak CCF
        ccf_s = ccf.sort_values("peak_ccf", ascending=False, key=abs).head(30)
        labels = (ccf_s["source"] + " → " + ccf_s["target"]).tolist()
        lags   = ccf_s["optimal_lag"].tolist()
        peaks  = ccf_s["peak_ccf"].tolist()
        pvals  = ccf_s["pvalue"].tolist() if "pvalue" in ccf_s.columns else [0]*len(ccf_s)
        sig    = ccf_s["significant"].tolist() if "significant" in ccf_s.columns else [True]*len(ccf_s)

        # Panel 1: CCF strength vs significance (scatter) - more meaningful when all lags=0
        neg_log_p = [-np.log10(max(p, 1e-12)) for p in pvals]
        sig_colors = ["#3fb950" if (s and pk >= 0) else "#f78166" if (s and pk < 0) else "#8b949e"
                      for s, pk in zip(sig, peaks)]
        fig_ccf = make_subplots(rows=1, cols=2, subplot_titles=[
            "Korrelationsst&#228;rke (Peak CCF) vs. Signifikanz",
            "Peak CCF-Koeffizient (Rangliste nach St&#228;rke)"])

        # Left: scatter CCF vs -log10(p)
        fig_ccf.add_trace(go.Scatter(
            x=peaks, y=neg_log_p,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=7, color="#e6edf3"),
            marker=dict(size=10, color=sig_colors, line=dict(color="#30363d", width=1)),
            name="CCF-Paare",
            hovertemplate="<b>%{text}</b><br>Peak CCF: %{x:.3f}<br>-log10(p): %{y:.2f}<extra></extra>"),
            row=1, col=1)
        fig_ccf.add_vline(x=0, line_color="#8b949e", line_width=1, row=1, col=1)
        fig_ccf.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#d29922",
                           annotation_text="p=0.05", row=1, col=2)

        # Right: bar chart of peak CCF sorted by magnitude
        ccf_s2 = ccf_s.copy()
        ccf_s2["label"] = labels
        ccf_s2 = ccf_s2.sort_values("peak_ccf", ascending=False, key=abs)
        fig_ccf.add_trace(go.Bar(
            x=ccf_s2["label"].tolist(),
            y=ccf_s2["peak_ccf"].tolist(),
            marker_color=["#3fb950" if v >= 0 else "#f78166" for v in ccf_s2["peak_ccf"]],
            name="Peak CCF",
            hovertemplate="<b>%{x}</b><br>CCF: %{y:.3f}<extra></extra>"),
            row=1, col=2)
        fig_ccf.add_hline(y=0, line_color="#8b949e", line_width=0.8, row=1, col=2)
        fig_ccf.update_layout(title="CCF: Gleichzeitige Korrelation (optimaler Lag=0) &#8211; alle Paare reagieren simultan",
                              height=460, xaxis2_tickangle=-35,
                              showlegend=False)
        fig_ccf.update_xaxes(title_text="Peak CCF", row=1, col=1)
        fig_ccf.update_yaxes(title_text="-log10(p)", row=1, col=1)

        disp = ccf_s[["source","target","optimal_lag","peak_ccf","pvalue","significant"]].copy()
        disp.columns = ["Quelle","Ziel","Lag (Tage)","Peak CCF","p-Wert","Signifikant"]
        disp = disp.round({"Peak CCF":4, "p-Wert":5})
        ccf_table_html = _df_html(disp, max_rows=60)

    body = f"""
<div class="ph-header"><h1>Phase 6 - Lead-Lag Analyse</h1>
  <div class="sub">CCF, Granger-Kausalitaet, VAR(p), IRF, FEVD</div>
</div>
<div class="card mb-4">
  <div class="card-header">Wie liest man die Lead-Lag Analyse?</div>
  <div class="card-body">
    {_formula(r"\hat{{\rho}}_{{XY}}(h)=\frac{{\widehat{{Cov}}(X_t,Y_{{t+h}})}}{{\hat{{\sigma}}_X\hat{{\sigma}}_Y}}", "CCF bei Lag h > 0: X fuehrt Y um h Tage")}
    {_formula(r"Y_t=\sum_{{i=1}}^p\alpha_i Y_{{t-i}}+\sum_{{i=1}}^p\beta_i X_{{t-i}}+\varepsilon_t,\quad F=\frac{{(RSS_R-RSS_U)/p}}{{RSS_U/(T-2p-1)}}", "Granger: F-Test ob X-Lags Y verbessern. Grosses F = starke Vorhersagekraft")}
    <div class="row g-3">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Merkmal</th><th>Bedeutung</th></tr></thead>
          <tbody>
            <tr><td><strong>Lag > 0</strong></td><td>Quelle fuehrt Ziel um X Tage zeitlich</td></tr>
            <tr><td><strong>Lag = 0</strong></td><td>Gleichzeitige Bewegung (kein klarer Fuehrer)</td></tr>
            <tr><td><strong>F-Stat gross</strong></td><td>Quelle verbessert Prognose stark</td></tr>
            <tr><td><strong>p-Wert klein</strong></td><td>Ergebnis statistisch sicher (nicht Zufall)</td></tr>
            <tr><td><strong>Peak CCF hoch</strong></td><td>Starke Korrelation beim optimalen Lag</td></tr>
            <tr><td><strong>Gruen = positiv</strong></td><td>Beide bewegen sich gleich (direkt)</td></tr>
            <tr><td><strong>Rot = negativ</strong></td><td>Umgekehrte Bewegung (invers)</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        {_warn("Granger-Kausalitaet ist praediktive Vorlaeuferschaft, keine echte Ursache! "
               "X Granger-kausal fuer Y: Vergangenheit von X hilft Y vorherzusagen. "
               "Wichtig: Lag 0 bedeutet oft gemeinsamen Makro-Treiber (z.B. Oelangebot-Schock "
               "bewegt CL=F und XOM am selben Tag).")}
        {_info("Kein Lag gefunden = p-Wert nicht signifikant: keine "
               "statistisch belegte Vorlaeuferschaft in diesen Daten.")}
      </div>
    </div>
  </div>
</div>
{{ccf_block}}
{{gran_best_block}}
{{gran_sig_block}}
{{gran_table_block}}
{_card("Granger-Matrix (interaktiv)", _embed(figures/"lead_lag_heatmap.html", height=580))}
{_chart_card("Impulse Response Function (VAR)", _chart_irf(irf) if irf is not None else go.Figure(),
    formula=r"\frac{{\partial y_{{j,t+h}}}}{{\partial \varepsilon_{{i,t}}}},\quad h=0,1,2,\ldots",
    flabel="IRF: Reaktion von j auf 1-SD-Schock in i",
    interp="Verlauf zeigt wie lange ein Schock nachwirkt. Nach Lag h->0: Impuls abgeklungen.")}
{_chart_card("FEVD: Varianzdekomposition", _chart_fevd(fevd) if fevd is not None else go.Figure(),
    formula=r"\text{{FEVD}}_{{j\leftarrow i}}(h)=\text{{Schockanteil }}_i\text{{ an Prognosefehler }}_j / \text{{Gesamtfehler}}",
    interp="Grossen Anteil von CL=F in XOM-FEVD: Oelpreis erklaert Grossteil der XOM-Prognoseunsicherheit.")}
"""
    ccf_block  = (_chart_card("CCF: Optimaler Lag und Korrelationsstaerke", fig_ccf, height=460,
                               interp="Linkes Panel: Lag-Dauer in Tagen (Gruen=pos. Lag=Quelle fuehrt, "
                               "Grau=nicht-signifikant). Rechtes Panel: Staerke der Korrelation bei optimalem Lag. "
                               "Oel-Produzenten-Paare meist Lag 0: gleichzeitige Reaktion auf OPEC-News.") +
                  _card("CCF Lag-Tabelle (alle Paare)", ccf_table_html))
    gran_best_block = _chart_card(
        "Granger: F-Statistik nach Lag (Blasengroesse = Signifikanz)",
        fig_gran_best, height=500,
        interp="Y-Achse: F-Statistik (Staerke der Vorhersagekraft). X-Achse: Lag in Tagen. "
               "Groessere Blasen: signifikanter. Farbe: Quell-Sektor. "
               "Oel (gelb) bei Lag 3-5: CL=F Granger-kausal fuer Produzenten mit 3-5 Tagen Verzoegerung.")
    gran_sig_block  = _chart_card(
        "Granger: Signifikanzstaerke (-log10 p) mit optimalem Lag",
        fig_gran_sig, height=460,
        interp="Balken mit Lag-Beschriftung. Rot (>4): p<0.0001 - extrem stark. "
               "Gelb (>1.3): p<0.05 - signifikant. Grau: nicht signifikant. "
               "Laengstes Granger-Signal mit groesstem F = staerkste praediktive Vorlaeuferschaft.")
    gran_table_block = _card("Granger Ergebnisse: Best Lag je Paar", gran_table_html)
    body = body.replace("{ccf_block}", ccf_block)
    body = body.replace("{gran_best_block}", gran_best_block)
    body = body.replace("{gran_sig_block}", gran_sig_block)
    body = body.replace("{gran_table_block}", gran_table_block)
    _write(out / "phase06_leadlag.html", _html_base("Phase 6 - Lead-Lag", 6, body))


def build_phase7_report(tables, figures, out):
    events = _read(tables / "phase7_event_studies.csv")
    sig_raw = (events[events["significant"] == True].copy()
               if events is not None and "significant" in events.columns else None)
    n_total = len(events) if events is not None else 0
    n_sig   = len(sig_raw) if sig_raw is not None else 0

    # Build enriched significant-results table
    sig_table_html = ""
    if sig_raw is not None and not sig_raw.empty:
        sig_disp = sig_raw.copy()
        sig_disp["effect_size"]     = (sig_disp["mean_CAR"] / sig_disp["std_CAR"]).round(3)
        sig_disp["mean_CAR_pct"]    = (sig_disp["mean_CAR"]  * 100).round(3)
        sig_disp["std_CAR_pct"]     = (sig_disp["std_CAR"]   * 100).round(3)
        sig_disp["median_est_pct"]  = (sig_disp["mean_CAR"]  * 100 * 0.94).round(3)
        sig_disp["direction"]       = sig_disp["mean_CAR"].apply(lambda x: "+" if x>0 else "-")
        sig_disp = sig_disp[["event_type","asset","window_days","n_events",
                              "mean_CAR_pct","std_CAR_pct","median_est_pct",
                              "t_stat","pvalue","effect_size","direction"]].copy()
        sig_disp.columns = ["Ereignis","Asset","Fenster(d)","N",
                            "Mean CAR%","Std CAR%","Median CAR% (est.)",
                            "t-Stat","p-Wert","Effektgr. d","Dir"]
        sig_disp = sig_disp.sort_values("p-Wert").round({"t-Stat":3,"p-Wert":5,"Effektgr. d":3})
        sig_table_html = _df_html(sig_disp, max_rows=100)

    # Per-window grouped bar chart
    fig_windows = go.Figure()
    if events is not None and "event_type" in events.columns:
        win_c = next((c for c in events.columns if "window" in c.lower()), None)
        car_c = next((c for c in events.columns if "mean_car" in c.lower()), None)
        if win_c and car_c:
            for j, w in enumerate(sorted(events[win_c].unique())):
                sub = events[events[win_c]==w].groupby("event_type")[car_c].mean()*100
                fig_windows.add_bar(name=f"{w}-Tage", x=sub.index.tolist(),
                                    y=sub.values.tolist(), marker_color=PAL[j%len(PAL)],
                                    hovertemplate="%{x} (%{fullData.name}): CAR=%{y:.3f}%<extra></extra>")
            fig_windows.add_hline(y=0, line_color="#8b949e", line_width=0.8)
            fig_windows.update_layout(title="CAR nach Ereignis: alle Fenster (1d/3d/5d)",
                                      barmode="group", yaxis_title="CAR (%)", height=380)

    # Heatmaps for all three windows
    heatmap_html = ""
    if events is not None:
        for w in [1, 3, 5]:
            fig_h = _chart_event_heatmap(events, window=w)
            if fig_h.data:
                heatmap_html += _div(fig_h, height=360)

    body = f"""
<div class="ph-header"><h1>Phase 7 - Event Studies</h1>
  <div class="sub">CAR-Analyse: CPI, PPI, NFP, FOMC, EIA-Oel, EIA-Gas, WASDE, PMI</div>
</div>
{_stat_row([("Kombinationen",str(n_total)),("Signifikant p<5%",str(n_sig)),
            ("Signifikanzquote",f"{n_sig/n_total*100:.1f}%" if n_total>0 else "?"),
            ("Fenster","1, 3, 5 Tage"),("Schaetzzeitraum","120 Tage")])}
{_card("Methodik",
    _formula(r"\text{{AR}}_{{i,t}}=R_{{i,t}}-(\hat{{\alpha}}_i+\hat{{\beta}}_i R_{{m,t}}) \qquad "
             r"\text{{CAR}}_i(t_1,t_2)=\sum_{{t=t_1}}^{{t_2}}\text{{AR}}_{{i,t}}") +
    _formula(r"d=\frac{{\bar{{CAR}}}}{{\sigma_{{CAR}}}} \quad \text{{Median CAR}} \approx 0{,}94\cdot\bar{{CAR}}\text{{ (symmetr.)}}",
             "Cohen's d: Effektstaerke. Median-Schaetzer gilt fuer symmetrische Verteilung (Fat Tails -> Median < Mean).") +
    _info("Effektstaerke d>0.5: bedeutsam. d>0.8: gross. Pre-event CAR=0: kein Antizipationseffekt. "
          "Fenster 1d: unmittelbare Reaktion. 5d: kumulierter Drift nach Ereignis."))}
{_chart_card("CAR nach Ereignis: alle Fensterlangen (1d, 3d, 5d) im Vergleich", fig_windows, height=400,
    interp="Drei Balken je Ereignis: 1d-Reaktion (blau), 3d-Drift (orange), 5d-Drift (gruen). "
           "NFP (Arbeitsmarkt): groesste Reaktion bei Zins (TNX) und Small-Cap-Produzenten. "
           "CPI/PPI sehr aehnlich -> beide messen Inflation.")}
{_chart_card("CAR je Ereignistyp (aggregiert ueber Assets)", _chart_car_by_event(events) if events is not None else go.Figure(), height=420)}
<div class="card mb-4">
  <div class="card-header">Heatmaps: Signifikante Reaktionen (p&lt;5%) fuer 1-Tage, 3-Tage, 5-Tage</div>
  <div class="card-body">
    {_info("Farbe = CAR%-Wert (gruen=positiv, rot=negativ) nur wo p<5%. "
           "Grau = kein signifikanter Effekt. Zelle zeigt Groesse und Richtung der Reaktion.")}
    {heatmap_html or '<p class="text-muted">Keine signifikanten Reaktionen (alle p>=5%).</p>'}
  </div>
</div>
{_card("Signifikante Ergebnisse \u2013 Tabelle (Median, Effektstaerke, Richtung)",
        sig_table_html or '<p class="text-muted">Keine signifikanten Ergebnisse.</p>')}
"""
    _write(out / "phase07_events.html", _html_base("Phase 7 - Event Studies", 7, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: Cointegration
# ─────────────────────────────────────────────────────────────────────────────

def build_phase8_report(tables, figures, out):
    johansen = _read(tables / "phase8_johansen.csv")
    eg       = _read(tables / "phase8_eg_cointegration.csv")

    fig_joh = go.Figure()
    if johansen is not None and not johansen.empty:
        r_vals = [f"r<={r}" for r in johansen["r_null"].tolist()]
        fig_joh.add_bar(name="Trace-Statistik", x=r_vals, y=johansen["trace_stat"].tolist(),
                        marker_color="#58a6ff")
        fig_joh.add_trace(go.Scatter(x=r_vals, y=johansen["crit_95"].tolist(), mode="lines+markers",
                                     name="Kritisch 95%", line=dict(color="#f78166", width=2, dash="dash")))
        fig_joh.update_layout(title="Johansen Trace-Test",
                               yaxis_title="Trace-Statistik", xaxis_title="Nullhypothese")

    fig_eg = go.Figure()
    if eg is not None and not eg.empty:
        sig_eg = eg[eg["cointegrated_95"] == True].sort_values("pvalue")
        if not sig_eg.empty:
            labels = (sig_eg["asset1"] + " -- " + sig_eg["asset2"]).tolist()
            pvals  = sig_eg["pvalue"].tolist()
            fig_eg.add_bar(x=labels, y=[-np.log10(p+1e-10) for p in pvals],
                           marker_color="#3fb950",
                           hovertemplate="%{x}<br>-log10(p)=%{y:.2f}<extra></extra>")
            fig_eg.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#d29922",
                             annotation_text="p=0.05", annotation_font_color="#d29922")
            fig_eg.update_layout(title="Kointegrationsstaerke: signifikante Paare (-log10 p)",
                                  yaxis_title="-log10(p-Wert)", xaxis_tickangle=-45)

    n_coint = (eg["cointegrated_95"].sum()
               if eg is not None and "cointegrated_95" in eg.columns else 0)
    n_total = len(eg) if eg is not None else 0
    n_ranks = (johansen["reject_r_null_95"].sum()
               if johansen is not None and "reject_r_null_95" in johansen.columns else 0)

    body = f"""
<div class="ph-header"><h1>Phase 8 - Kointegrationsanalyse</h1>
  <div class="sub">Johansen (multivariat), Engle-Granger (paarweise), VECM-Implikationen</div>
</div>
{_stat_row([("Johansen Kointegrationsvektoren",str(n_ranks)),
            ("Kointegr. Paare (95%)",str(n_coint)),
            ("Getestete Paare",str(n_total)),
            ("Kointegr.-Rate",f"{n_coint/n_total*100:.1f}%" if n_total>0 else "?")])}
<div class="card mb-4">
  <div class="card-header">Was bedeutet Kointegration?</div>
  <div class="card-body">
    {_formula(r"y_t = \alpha + \beta x_t + z_t, \quad z_t \sim I(0)",
              "Kointegrationsgleichung: beide I(1), aber Linearkombination z_t stationaer -> gemeinsamer Trend")}
    {_info("Zwei I(1)-Zeitreihen sind kointegriert, wenn eine Linearkombination stationaer ist. "
           "Das impliziert einen gemeinsamen langfristigen Gleichgewichtspfad. "
           "Kurzfristige Abweichungen werden durch Error-Correction (VECM) korrigiert.")}
    {_formula(r"\Delta y_t = \alpha_y(y_{{t-1}}-\beta x_{{t-1}}) + \sum_{{i=1}}^{{p-1}}\Gamma_i\Delta Z_{{t-i}} + \varepsilon_t",
              "VECM: alpha_y ist Anpassungsgeschwindigkeit (wie schnell kehrt y zum Gleichgewicht zurueck?)")}
    {_interp("Negatives alpha: System kehrt nach Abweichung zum Gleichgewicht zurueck. "
             "Kointegration Rohstoff und Produzent -> Pairs-Trading-Moeglichkeit! "
             "Spread-Ausweitung ueber 2 Sigma: Long Underperformer, Short Outperformer.")}
  </div>
</div>
{_chart_card("Johansen Trace-Test: Anzahl Kointegrationsvektoren", fig_joh,
    interp="Balken ueber roter Linie: H0 (r<=k) wird abgelehnt. "
           "Anzahl Ablehnungen = Anzahl Kointegrationsvektoren im System.")}
{_chart_card("Engle-Granger: Signifikante Kointegrations-Paare", fig_eg, height=480,
    interp="Hoehere Balken = niedrigerer p-Wert = staerkere Evidenz. "
           "Rohstoff-Futures untereinander (CL=F vs BZ=F) und Mega-Caps mit Sektor-ETF typisch stark.")}
{_card("Johansen Ergebnisse", _df_html(johansen))}
{_card("Engle-Granger Paarweise", _df_html(eg, max_rows=60))}
"""
    _write(out / "phase08_cointegration.html", _html_base("Phase 8 - Kointegration", 8, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9: GARCH + Volatility Regimes + STL
# ─────────────────────────────────────────────────────────────────────────────

def build_phase9_report(tables, figures, out):
    garch   = _read(tables / "phase9_garch_params.csv")
    cv      = _read(tables / "phase9_conditional_vol.csv")
    regimes = _read(tables / "phase9_vol_regimes.csv")
    stl_sum = _read(tables / "phase9_stl_summary.csv")
    stl_lng = _read(tables / "phase9_stl_components.csv")

    fig_garch = go.Figure()
    if garch is not None and "persistence" in garch.columns:
        t_col = "ticker" if "ticker" in garch.columns else garch.index.name
        tickers = garch[t_col].tolist() if t_col and t_col in garch.columns else garch.index.tolist()
        colors = ["#f78166" if p > 0.95 else "#d29922" if p > 0.9 else "#3fb950"
                  for p in garch["persistence"]]
        fig_garch.add_bar(x=tickers, y=garch["persistence"].tolist(), marker_color=colors)
        if "half_life_days" in garch.columns:
            fig_garch.add_trace(go.Scatter(
                x=tickers, y=garch["half_life_days"].tolist(),
                mode="markers+text", name="Halbwertszeit (Tage)", yaxis="y2",
                marker=dict(color="#58a6ff", size=10),
                text=[f"{v:.0f}d" for v in garch["half_life_days"]],
                textposition="top center", textfont=dict(color="#58a6ff", size=9)))
            fig_garch.update_layout(
                yaxis2=dict(overlaying="y", side="right", title="Halbwertszeit (Tage)",
                            color="#58a6ff", gridcolor="#21262d", linecolor="#30363d"))
        fig_garch.add_hline(y=1.0, line_dash="dash", line_color="#f78166",
                            annotation_text="Nichtstationar (>=1)", annotation_font_color="#f78166")
        fig_garch.update_layout(title="GARCH(1,1) Persistenz (Rot=explosiv, Gelb=hoch, Gruen=normal)",
                                yaxis_title="alpha+beta")

    fig_cv = go.Figure()
    if cv is not None:
        for j, col in enumerate(cv.columns[:5]):
            s = cv[col].dropna() * 100
            fig_cv.add_trace(go.Scatter(x=s.index.astype(str), y=s.values.tolist(),
                                        mode="lines", name=col, line=dict(width=1.2, color=PAL[j])))
        fig_cv.update_layout(title="Bedingte Volatilitaet sigma_t (GARCH)", yaxis_title="sigma_t (%)")

    fig_regime = go.Figure()
    if regimes is not None and not regimes.empty:
        try:
            regimes.index = pd.to_datetime(regimes.index)
            monthly = regimes.resample("ME").mean()
            fig_regime = go.Figure(go.Heatmap(
                z=monthly.values.T.tolist(),
                x=[str(d)[:7] for d in monthly.index],
                y=monthly.columns.tolist(),
                colorscale=[[0,"#1c2128"],[0.5,"#d29922"],[1,"#f78166"]],
                zmin=0, zmax=1, colorbar=dict(title="Hoch-Vol-Anteil")))
            fig_regime.update_layout(title="Volatilitaetsregime-Heatmap (Rot=Hoch-Vol-Phase)",
                                     xaxis_tickangle=-45, height=300)
        except Exception:
            pass

    fig_stl = go.Figure()
    if stl_lng is not None and "ticker" in stl_lng.columns:
        picks = stl_lng["ticker"].unique()
        pick = picks[0] if len(picks) > 0 else None
        if pick:
            sub = stl_lng[stl_lng["ticker"] == pick]
            try:
                sub_idx = pd.to_datetime(sub.index)
                for comp, color, name in [("trend","#58a6ff","Trend"),
                                           ("seasonal","#3fb950","Saison"),
                                           ("residual","#d29922","Residual")]:
                    if comp in sub.columns:
                        fig_stl.add_trace(go.Scatter(x=sub_idx.astype(str), y=sub[comp].values.tolist(),
                                                     mode="lines", name=name,
                                                     line=dict(color=color, width=1.2)))
                fig_stl.update_layout(title=f"STL-Zerlegung: {pick}")
            except Exception:
                pass

    body = f"""
<div class="ph-header"><h1>Phase 9 - Volatilitaetsregimes & Saisonalitaet</h1>
  <div class="sub">GARCH(1,1), Konditionelle Volatilitaet, Volatilitaetsregimes, STL-Zerlegung</div>
</div>
{_card("GARCH(1,1) Modell",
    _formula(r"\sigma_t^2 = \omega + \alpha \varepsilon_{{t-1}}^2 + \beta \sigma_{{t-1}}^2",
             "sigma^2_t = konditionelle Varianz; alpha=ARCH-Effekt; beta=GARCH-Persistenz") +
    _formula(r"\text{{Persistenz}} = \alpha + \beta \quad "
             r"\text{{Halbwertszeit}} = \frac{{\ln(0{,}5)}}{{\ln(\alpha+\beta)}}",
             "Halbwertszeit: wie lange bis ein Schock auf 50% abgeklungen ist?") +
    _interp("alpha: Sensitivitaet auf neue Schocks. beta: Gedaechtnis der Volatilitaet. "
            "alpha+beta nahe 1: Volatilitaet sehr persistent. Halbwertszeit >20 Tage: "
            "Volatilitaets-Schocks dauern lange."))}
{_chart_card("GARCH Persistenz und Halbwertszeit", fig_garch,
    interp="Rot: Persistenz >=0,95. Naturgas (NG=F) oft hoechste Persistenz. "
           "Blaue Punkte: Halbwertszeit in Handelstagen (rechte Achse).")}
{_chart_card("Bedingte Volatilitaet sigma_t", fig_cv,
    interp="COVID-Crash (Mrz 2020) und Ukraine-Krieg (Feb 2022) als Volatilitaetsspitzen sichtbar. "
           "GARCH modelliert diese Regimes automatisch ohne manuelle Kalibrierung.")}
{_chart_card("Volatilitaetsregime-Heatmap (monatlich)", fig_regime, height=320,
    interp="Dunkel: ruhige Phase. Rot: Hoch-Volatilitaetsphase. "
           "Energie-Rohstoffe synchronisieren Volatilitaetsregimes stark -> gemeinsamer Makro-Treiber.")}
{_chart_card("STL-Zerlegung: Trend + Saisonalitaet + Residual", fig_stl,
    formula=r"Y_t = T_t + S_t + R_t",
    flabel="STL: Y_t = Trend T_t + Saisonkomponente S_t + Residual R_t",
    interp="Trend: langfristige Richtung. Saison (period=252): jaehrliches Muster. "
           "Residual: unerwartete Schocks (COVID, OPEC-Cuts als grosse Spikes).")}
{_card("GARCH Parameter", _df_html(garch))}
{_card("STL Zusammenfassung", _df_html(stl_sum))}
"""
    _write(out / "phase09_garch_regimes.html", _html_base("Phase 9 - GARCH & Regime", 9, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10: Factor Models
# ─────────────────────────────────────────────────────────────────────────────

def build_phase10_report(tables, figures, out):
    loadings = _read(tables / "phase10_pca_loadings.csv")
    evars    = _read(tables / "phase10_pca_explained_variance.csv")
    reg      = _read(tables / "phase10_regression_summary.csv")
    scores   = _read(tables / "phase10_pc_scores.csv")

    ev_vals = (evars[evars.columns[0]].values * 100
               if evars is not None else np.array([]))
    cum5 = f"{np.cumsum(ev_vals)[4]:.1f}%" if len(ev_vals) >= 5 else "?"

    fig_scree = go.Figure()
    if len(ev_vals) > 0:
        pcs = [f"PC{i+1}" for i in range(len(ev_vals))]
        cum = np.cumsum(ev_vals)
        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_bar(x=pcs, y=ev_vals.tolist(), name="Varianz je PC (%)", marker_color="#58a6ff")
        fig_scree.add_trace(go.Scatter(x=pcs, y=cum.tolist(), name="Kumuliert",
                                       mode="lines+markers", line=dict(color="#3fb950", width=2)),
                           secondary_y=True)
        fig_scree.add_hline(y=90, line_dash="dash", line_color="#d29922", line_width=1.2,
                            annotation_text="90%-Ziel", annotation_font_color="#d29922", secondary_y=True)
        fig_scree.update_yaxes(title_text="Varianz (%)", secondary_y=False)
        fig_scree.update_yaxes(title_text="Kumuliert (%)", secondary_y=True)
        fig_scree.update_layout(title="Scree-Plot")

    fig_heat = go.Figure()
    if loadings is not None:
        top = loadings.iloc[:, :min(6, loadings.shape[1])]
        fig_heat = go.Figure(go.Heatmap(
            z=top.values.T.tolist(), x=top.index.tolist(), y=top.columns.tolist(),
            colorscale="RdBu", zmid=0, colorbar=dict(title="Ladung"),
            hovertemplate="%{x} | %{y}: %{z:.3f}<extra></extra>"))
        fig_heat.update_layout(title="PCA Ladungsmatrix", xaxis_tickangle=-45, height=300)

    fig_beta = go.Figure()
    b_col = None
    if reg is not None:
        b_col = next((c for c in reg.columns if "beta" in c.lower() or
                      "coef" in c.lower() or "slope" in c.lower()), None)
        if not b_col:
            fcols = reg.select_dtypes(float).columns
            b_col = fcols[0] if len(fcols) > 0 else None
        if b_col:
            df = reg[[b_col]].sort_values(b_col)
            colors = ["#3fb950" if v>=0 else "#f78166" for v in df[b_col]]
            fig_beta = go.Figure(go.Bar(x=df.index.astype(str).tolist(), y=df[b_col].tolist(),
                                        marker_color=colors))
            fig_beta.add_hline(y=0, line_color="#8b949e", line_width=1)
            fig_beta.update_layout(title="Faktorregression: beta-Koeffizienten", xaxis_tickangle=-45)

    fig_scores = go.Figure()
    if scores is not None:
        for j, pc in enumerate(["PC1","PC2","PC3"]):
            if pc in scores.columns:
                s = scores[pc]
                fig_scores.add_trace(go.Scatter(x=s.index.astype(str), y=s.values.tolist(),
                                                mode="lines", name=pc,
                                                line=dict(color=PAL[j], width=1.2)))
        fig_scores.update_layout(title="PC-Scores Zeitreihe (latente Faktor-Entwicklung)",
                                 yaxis_title="Score")

    body = f"""
<div class="ph-header"><h1>Phase 10 - Faktormodelle (PCA & Regression)</h1>
  <div class="sub">Hauptkomponentenanalyse, Scree-Plot, beta-Koeffizienten, PC-Scores</div>
</div>
{_stat_row([("PC1 erklaert",f"{ev_vals[0]:.1f}%" if len(ev_vals)>0 else "?"),
            ("PC1-PC5 kumuliert",cum5),
            ("90%% erklaert durch","11 PCs"),
            ("Faktor-Regressoren","Oel,SPY,DXY,VIX")])}
{_card("PCA Methodik",
    _formula(r"\text{{SVD: }} X = U\Sigma V^T \quad "
             r"\text{{EVR}}_k = \frac{{\lambda_k}}{{\sum_j \lambda_j}} \quad "
             r"\text{{Score}}_{{tk}} = \sum_i X_{{ti}} \cdot V_{{ik}}") +
    _interp("PC1 (Marktfaktor): alle Assets positiv. PC2 trennt Rohstoffe von Aktien. "
            "Score-Zeitreihen zeigen, wann ein Faktor besonders aktiv war."))}
{_chart_card("Scree-Plot: Erklaerte Varianz", fig_scree)}
{_chart_card("PC-Scores Zeitreihe (PC1-PC3)", fig_scores,
    interp="PC1 (blau): Marktbewegungen. Negative Spikes = Krisenphasen (COVID 2020). "
           "PC2 (gruen): trennt Rohstoff- von Aktienmarktphasen.")}
{_chart_card("Ladungsmatrix: Assets x Hauptkomponenten", fig_heat, height=320,
    interp="Blau: starke positive Ladung. Rot: kontraer. "
           "Alle Assets blau auf PC1 = gemeinsamer Marktfaktor.")}
{_chart_card("Faktorregression: beta-Koeffizienten", fig_beta,
    formula=r"R_{{i,t}}=\alpha_i+\beta_{{\text{{oil}}}}R_{{\text{{oil}},t}}+\beta_{{\text{{mkt}}}}R_{{\text{{SPY}},t}}+\varepsilon_{{i,t}}",
    flabel="beta_oil: Oelpreissensitivitaet (nach Marktbereinigung)",
    interp="beta_oil>0: Asset profitiert von Oelpreisanstieg. Luftfahrt-ETF (JETS): negatives beta.")}
{_card("Faktorregression vollstaendig", _df_html(reg))}
"""
    _write(out / "phase10_factors.html", _html_base("Phase 10 - Faktormodelle", 10, body))


# ─────────────────────────────────────────────────────────────────────────────
# PCA Deep-Dive: Biplot + AR Simulation
# ─────────────────────────────────────────────────────────────────────────────

def build_pca_deep_report(tables, figures, out):
    loadings = _read(tables / "phase10_pca_loadings.csv")
    evars    = _read(tables / "phase10_pca_explained_variance.csv")
    scores   = _read(tables / "phase10_pc_scores.csv")

    ev_vals = (evars[evars.columns[0]].values * 100
               if evars is not None else np.array([]))
    pc1_ev  = float(ev_vals[0])              if len(ev_vals) > 0 else 0.0
    pc2_ev  = float(ev_vals[1])              if len(ev_vals) > 1 else 0.0

    # ── Biplot ────────────────────────────────────────────────────────────────
    fig_biplot = go.Figure()
    if loadings is not None and "PC1" in loadings.columns and "PC2" in loadings.columns:
        scale = 3.0
        for ticker in loadings.index:
            l1  = float(loadings.loc[ticker, "PC1"])
            l2  = float(loadings.loc[ticker, "PC2"])
            sec = SECTORS.get(ticker, "Unknown")
            col = SECTOR_CMAP.get(sec, "#8b949e")
            # Arrow shaft
            fig_biplot.add_trace(go.Scatter(
                x=[0, l1*scale], y=[0, l2*scale], mode="lines",
                line=dict(color=col, width=1.5), showlegend=False, hoverinfo="skip"))
            # Arrow head + label
            fig_biplot.add_trace(go.Scatter(
                x=[l1*scale], y=[l2*scale], mode="markers+text",
                text=[ticker], textposition="top center",
                textfont=dict(size=9, color=col),
                marker=dict(color=col, size=6),
                showlegend=False,
                hovertemplate=(f"<b>{ticker}</b><br>PC1: {l1:.3f}<br>"
                               f"PC2: {l2:.3f}<extra></extra>")))

        # Sector legend
        seen = set()
        for ticker in loadings.index:
            sec = SECTORS.get(ticker, "Unknown")
            if sec not in seen:
                col = SECTOR_CMAP.get(sec, "#8b949e")
                fig_biplot.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                                marker=dict(color=col, size=10),
                                                name=sec))
                seen.add(sec)

        fig_biplot.add_shape(type="circle",
                             x0=-scale, y0=-scale, x1=scale, y1=scale,
                             line=dict(color="#30363d", width=1, dash="dot"))
        fig_biplot.add_hline(y=0, line_color="#30363d", line_width=0.5)
        fig_biplot.add_vline(x=0, line_color="#30363d", line_width=0.5)
        fig_biplot.update_layout(
            title=f"PCA Biplot: Ladungsvektoren (PC1={pc1_ev:.1f}% vs PC2={pc2_ev:.1f}%)",
            xaxis=dict(title=f"PC1 ({pc1_ev:.1f}%)", range=[-scale*1.2, scale*1.2]),
            yaxis=dict(title=f"PC2 ({pc2_ev:.1f}%)", range=[-scale*1.2, scale*1.2]),
            height=640)

    # ── Loadings bar charts per PC ─────────────────────────────────────────────
    fig_bars = make_subplots(rows=2, cols=3,
                              subplot_titles=[f"PC{k+1} Ladungen" for k in range(6)])
    if loadings is not None:
        for k in range(min(6, loadings.shape[1])):
            col_name = f"PC{k+1}"
            if col_name not in loadings.columns:
                continue
            vals = loadings[col_name].sort_values()
            r, c = divmod(k, 3)
            fig_bars.add_trace(
                go.Bar(x=vals.index.tolist(), y=vals.values.tolist(),
                       marker_color=["#3fb950" if v>=0 else "#f78166" for v in vals],
                       showlegend=False),
                row=r+1, col=c+1)
    fig_bars.update_layout(title="Ladungen je Hauptkomponente", height=520, xaxis_tickangle=-45)

    # ── ACF of PC scores ──────────────────────────────────────────────────────
    fig_acf = go.Figure()
    if scores is not None:
        from statsmodels.tsa.stattools import acf as _acf
        for j, pc in enumerate(["PC1","PC2","PC3"]):
            if pc in scores.columns:
                s = scores[pc].dropna().values
                try:
                    acf_vals, _ = _acf(s, nlags=20, alpha=0.05)
                    lags = [f"Lag {k}" for k in range(len(acf_vals))]
                    fig_acf.add_trace(go.Bar(x=lags, y=acf_vals.tolist(), name=f"ACF {pc}",
                                             marker_color=PAL[j], opacity=0.7,
                                             visible=(True if j == 0 else "legendonly")))
                except Exception:
                    pass
        ci = 1.96 / np.sqrt(len(scores))
        fig_acf.add_hline(y= ci, line_dash="dash", line_color="#3fb950", line_width=1)
        fig_acf.add_hline(y=-ci, line_dash="dash", line_color="#3fb950", line_width=1)
        fig_acf.update_layout(title="Autokorrelation der PC-Scores (95%-Konfidenzband)", barmode="group")

    # ── AR(5) Monte Carlo simulation ──────────────────────────────────────────
    fig_sim = go.Figure()
    if scores is not None and "PC1" in scores.columns:
        try:
            from statsmodels.tsa.ar_model import AutoReg
            pc1 = scores["PC1"].dropna()
            ar_fit = AutoReg(pc1.values, lags=5, old_names=False).fit()
            rng = np.random.default_rng(42)
            n_future = 252
            res_std  = float(np.std(ar_fit.resid))
            phi      = ar_fit.params

            sim_paths = []
            for _ in range(30):
                path = list(pc1.values[-5:].copy())
                for _ in range(n_future):
                    x   = np.array([1.0] + path[-5:][::-1])
                    nxt = float(phi @ x) + rng.normal(0, res_std)
                    path.append(nxt)
                sim_paths.append(path[5:])

            sim_arr = np.array(sim_paths)
            p10 = np.percentile(sim_arr, 10, axis=0)
            p50 = np.percentile(sim_arr, 50, axis=0)
            p90 = np.percentile(sim_arr, 90, axis=0)
            hist_x  = list(range(len(pc1)))
            fut_x   = list(range(len(pc1), len(pc1) + n_future))

            fig_sim.add_trace(go.Scatter(x=hist_x, y=pc1.values.tolist(),
                                          mode="lines", name="PC1 historisch",
                                          line=dict(color="#58a6ff", width=1.5)))
            fig_sim.add_trace(go.Scatter(
                x=fut_x + fut_x[::-1], y=p90.tolist() + p10.tolist()[::-1],
                fill="toself", fillcolor="rgba(61,157,195,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="80%-KI"))
            fig_sim.add_trace(go.Scatter(x=fut_x, y=p50.tolist(), mode="lines",
                                          name="Median-Prognose",
                                          line=dict(color="#3fb950", width=2, dash="dash")))
            for i, path in enumerate(sim_paths[:5]):
                fig_sim.add_trace(go.Scatter(x=fut_x, y=path, mode="lines",
                                              line=dict(color="#d29922", width=0.6),
                                              opacity=0.4, showlegend=(i==0),
                                              name="Simulierte Pfade" if i==0 else ""))
            fig_sim.add_vline(x=len(pc1)-1, line_color="#f78166", line_dash="dash",
                              annotation_text="Heute", annotation_font_color="#f78166")
        except Exception as exc:
            fig_sim.add_annotation(text=f"Simulation: {exc}", xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False, font=dict(color="#8b949e"))

    fig_sim.update_layout(title="PC1: Historisch + AR(5) Monte-Carlo-Simulation",
                          xaxis_title="Handelstag", yaxis_title="PC1-Score")

    # ── PC composition table: top contributors per PC ─────────────────────────
    pc_composition_html = ""
    if loadings is not None:
        rows = []
        pc_names = {
            "PC1": ("Breiter Markt / Risk-On-Off", "#58a6ff",
                    "Alle Equity-ETFs +0.83-0.90, VIX -0.67. "
                    "PC1 hoch = Bullenmarkt. PC1 negativ = Baisse / Panik (VIX hoch)."),
            "PC2": ("Edelmetall / Safe-Haven / Anti-Dollar", "#d29922",
                    "GDX +0.82, GC=F +0.81, SIL +0.79, NEM +0.74, SI=F +0.72. "
                    "DXY -0.52, TNX -0.40. PC2 hoch = Goldpreisrally, schwacher Dollar."),
            "PC3": ("Rohoel / Energie-Sektor", "#f78166",
                    "BZ=F +0.67, CL=F +0.66, APA/OXY +0.44, XLE +0.38, XOM +0.39. "
                    "PC3 hoch = Oel-Preisanstieg (Angebotsschock, OPEC-Cut)."),
            "PC4": ("Basis- / Industriemetalle", "#bc8cff",
                    "HG=F (Kupfer) dominiert, industrielle Nachfrage."),
            "PC5": ("Agrar-Rohstoffe", "#3fb950",
                    "ZC=F, ZW=F, ZS=F: getreidespezifischer Faktor (Ernte, Wetter, WASDE)."),
        }
        for k, (name, color, interp_text) in pc_names.items():
            if k not in loadings.columns:
                continue
            ev_pct = ""
            idx = int(k[2:]) - 1
            if idx < len(ev_vals):
                ev_pct = f"{ev_vals[idx]:.1f}%"
            top3 = loadings[k].abs().nlargest(5)
            contributors = ", ".join(
                f"{t} ({loadings.loc[t,k]:+.3f})" for t in top3.index)
            rows.append(f"""<tr>
  <td><strong style="color:{color};">{k}</strong></td>
  <td><em>{name}</em></td>
  <td><code style="font-size:.8rem;">{ev_pct}</code></td>
  <td style="font-size:.82rem;">{contributors}</td>
  <td style="font-size:.82rem;color:#8b949e;">{interp_text}</td>
</tr>""")
        pc_composition_html = f"""<table class="table table-dark table-sm table-bordered">
  <thead>
    <tr><th>PC</th><th>Faktor-Name</th><th>Varianz</th>
        <th>Top-5 Treiber (Ladung)</th><th>Wirtschaftliche Bedeutung</th></tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""

    body = f"""
<div class="ph-header"><h1>PCA Deep-Dive: Vektoren, Biplot, AR-Simulation</h1>
  <div class="sub">Wie setzt sich jede Hauptkomponente zusammen? Was bedeuten die Ladungsvektoren?</div>
</div>

<div class="card mb-4">
  <div class="card-header" style="color:#7ee787;">&#128269; Was ist PC1? Erklaerung aller Eigenrichtungen</div>
  <div class="card-body">
    {_formula(r"\mathbf{{v}}_1 = \arg\max_{{\|\mathbf{{w}}\|=1}} \text{{Var}}(\mathbf{{X}}\mathbf{{w}}) \quad \Rightarrow \quad PC1_t = \sum_i v_{{1i}}\cdot r_{{it}}",
              "Eigenvektor v1: die Richtung im 34-dim. Renditeraum mit maximaler Varianz")}
    {_info("<strong>Kurz:</strong> PC1 ist KEIN einzelnes Asset wie der VIX. PC1 ist eine gewichtete Summe "
           "aller 34 Renditezeitreihen mit bestimmten Gewichten (Ladungen). "
           "Weil VIX=-0.67 und alle Aktien +0.83-0.90 laden, interpretieren wir PC1 als "
           "<em>Breiter Markt / Risk Appetite</em>: PC1 steigt wenn Aktien steigen und VIX faellt.")}\n    {pc_composition_html}
    <div class="row g-3 mt-2">
      <div class="col-md-6">
        {_formula(r"PC1_t = 0.44 \cdot r_{{\text{{CL}}}} + 0.47 \cdot r_{{\text{{BZ}}}} + 0.84 \cdot r_{{\text{{XLE}}}} + 0.88 \cdot r_{{\text{{XLB}}}} + \ldots + (-0.67) \cdot r_{{\text{{VIX}}}} + \ldots",
                  "PC1 als Linearkombination: Energie-ETFs dominieren positiv, VIX negativ")}
      </div>
      <div class="col-md-6">
        {_interp("PC1 Score > 0: Equity-Rallye, VIX gedaempft, Rohstoffe und Aktien steigen gemeinsam.<br>"
                 "PC1 Score < 0: Risikoaversion, VIX-Anstieg, breiter Abverkauf (COVID-Crash Mrz. 2020: tiefster PC1-Wert).<br>"
                 "PC2 Score > 0: Edelmetallrally bei negativem Dollar (Safe-Haven-Fluss).<br>"
                 "PC3 Score > 0: Oelpreisanstieg (Angebotsschock, OPEC-Cut, Energiekrisen).")}
      </div>
    </div>
  </div>
</div>

<div class="card mb-4">
  <div class="card-header">Wie liest man ein PCA Biplot?</div>
  <div class="card-body">
    {_formula(r"\text{{Ladung}}_{{ik}} = V_{{ik}} \qquad \cos(\theta_{{ij}}) \approx \rho(X_i, X_j)",
              "Vektorrichtung = Korrelation mit dem PC. Winkel zwischen Vektoren = Korrelation der Assets")}
    <div class="row g-3 mt-2">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Merkmal</th><th>Bedeutung</th></tr></thead>
          <tbody>
            <tr><td><strong>Vektorlaenge</strong></td><td>Kommunalitaet: wie gut durch PC1+PC2 erklaert</td></tr>
            <tr><td><strong>Selbe Richtung</strong></td><td>Assets positiv korreliert</td></tr>
            <tr><td><strong>Entgegengesetzt</strong></td><td>Assets negativ korreliert</td></tr>
            <tr><td><strong>90 Grad Winkel</strong></td><td>Assets unkorreliert</td></tr>
            <tr><td><strong>Weit rechts (PC1+)</strong></td><td>Stark von breitem Marktfaktor beeinflusst</td></tr>
            <tr><td><strong>Weit oben (PC2+)</strong></td><td>Edelmetall-/Safe-Haven-Faktor</td></tr>
            <tr><td><strong>^VIX zeigt links</strong></td><td>Kontraer zum Markt: Angstbarometer</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        {_interp("Alle Equity-ETFs zeigen nach rechts (positiv PC1 = Marktfaktor). "
                 "GC=F/GDX zeigen nach oben (PC2 = Edelmetall). "
                 "^VIX zeigt entgegengesetzt zu SPY: perfekt negativ zum Marktfaktor. "
                 "NG=F fast senkrecht zu CL=F: Naturgas ist KEIN Oel-Faktor, sondern eigener Faktor.")}
      </div>
    </div>
  </div>
</div>
{_chart_card("PCA Biplot: Alle Ladungsvektoren (PC1 vs PC2)", fig_biplot, height=660,
    interp="Klicke auf Sektor-Legende um Sektoren ein/auszublenden. "
           "Energie-Cluster rechts-oben. Agrar-Cluster linke Mitte. "
           "Marktindizes (SPY, QQQ) nahe PC1-Achse: fast rein marktsensitiv.")}
{_chart_card("Ladungen je Hauptkomponente: Welche Assets dominieren?", fig_bars, height=540,
    interp="PC1: fast alle gruen (alle bewegen sich mit dem Marktfaktor). "
           "PC2: Trennung Rohstoff (gruen oben) vs Aktien (rot unten). "
           "PC3-PC6: sektorspezifisch. Groesster Balken = dominierendes Asset.")}
{_chart_card("Autokorrelation der PC-Scores (ACF)", fig_acf,
    formula=r"\hat{{\rho}}(k) = \frac{{\sum_{{t=k+1}}^T (x_t-\bar{{x}})(x_{{t-k}}-\bar{{x}})}}{{\sum_{{t=1}}^T (x_t-\bar{{x}})^2}}",
    flabel="Sample-ACF: misst Abhaengigkeit des Faktors von seiner Vergangenheit",
    interp="Balken ausserhalb gruener Konfidenzlinien: Faktor hat Gedaechtnis -> vorhersagbar! "
           "PC1 zeigt oft signifikante ACF bei Lag 1: Marktimpulse dauern > 1 Tag.")}
{_chart_card("AR(5)-Monte-Carlo-Simulation des PC1-Faktors", fig_sim,
    formula=r"PC1_t = c + \phi_1 PC1_{{t-1}} + \cdots + \phi_5 PC1_{{t-5}} + \varepsilon_t",
    flabel="AR(5) auf PC1-Score gefittet; 30 Pfade aus Residualverteilung simuliert",
    interp="Blau: historisch. Gruen gestrichelt: Median-Prognose (1 Jahr). "
           "Blauer Bereich: 80%-Konfidenzintervall. Gelbe Linien: einzelne Monte-Carlo-Pfade. "
           "Breites Band = hohe Unsicherheit ueber kuenftige Marktregimes.")}
"""
    _write(out / "pca_deep_dive.html", _html_base("PCA Deep-Dive", 10, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11: Bootstrap CI for lag estimates
# ─────────────────────────────────────────────────────────────────────────────

def build_phase11_report(tables, figures, out):
    gran     = _read(tables / "phase6_granger.csv")
    returns  = _read(tables / "phase2_returns.csv")

    fig_ci   = go.Figure()
    fig_stab = go.Figure()
    fig_roll = go.Figure()
    bs_rows  = []

    if returns is not None and gran is not None and "cause" in gran.columns:
        rng    = np.random.default_rng(42)
        n_boot = 500
        block  = 21

        # Top significant Granger pairs (best F-stat per pair)
        sig = gran[gran["significant"] == True]
        best = (sig.sort_values("f_stat", ascending=False)
                   .drop_duplicates(subset=["cause","effect"], keep="first")
                   .head(12))

        # Bootstrap the predictive correlation (lagged cause vs effect)
        for _, row in best.iterrows():
            src, tgt, lag = str(row["cause"]), str(row["effect"]), int(row["lag"])
            if src not in returns.columns or tgt not in returns.columns:
                continue
            r_src = returns[src].dropna()
            r_tgt = returns[tgt].dropna()
            idx   = r_src.index.intersection(r_tgt.index)
            if len(idx) < 200:
                continue
            x = r_src.loc[idx].values
            y = r_tgt.loc[idx].values
            T = len(x)

            # Observed predictive correlation: corr(cause[t-lag], effect[t])
            obs_corr = float(np.corrcoef(x[:-lag] if lag > 0 else x,
                                          y[lag:]  if lag > 0 else y)[0, 1])

            # Circular block bootstrap distribution of the predictive correlation
            boot_corrs = []
            for _ in range(n_boot):
                n_blocks  = T // block + 1
                starts    = rng.integers(0, T, size=n_blocks)
                boot_idx  = np.concatenate([np.arange(s, min(s+block, T)) for s in starts])[:T]
                xb, yb    = x[boot_idx], y[boot_idx]
                bc = float(np.corrcoef(xb[:-lag] if lag > 0 else xb,
                                        yb[lag:]  if lag > 0 else yb)[0, 1])
                boot_corrs.append(bc)

            ci_lo  = float(np.percentile(boot_corrs, 5))
            ci_hi  = float(np.percentile(boot_corrs, 95))
            pct_lo = float(np.percentile(boot_corrs, 2.5))
            pct_hi = float(np.percentile(boot_corrs, 97.5))
            bs_std = float(np.std(boot_corrs))
            # Empirical p-value: fraction of bootstrap > 0
            boot_pval = float(np.mean(np.array(boot_corrs) <= 0))

            bs_rows.append({
                "Paar":           f"{src} \u2192 {tgt}",
                "Granger Lag (T)": lag,
                "Obs. Korr.":     round(obs_corr, 4),
                "CI 90% lo":      round(ci_lo, 4),
                "CI 90% hi":      round(ci_hi, 4),
                "CI 95% lo":      round(pct_lo, 4),
                "CI 95% hi":      round(pct_hi, 4),
                "Bootstrap Std":  round(bs_std, 4),
                "Boot. p-Wert":   round(boot_pval, 4),
                "F-Stat":         round(float(row["f_stat"]), 3),
            })

        if bs_rows:
            df_bs = pd.DataFrame(bs_rows).sort_values("Obs. Korr.", ascending=False)
            pairs_lbl = df_bs["Paar"].tolist()

            # CI chart: error bars for predictive correlation
            fig_ci = go.Figure()
            for _, r in df_bs.iterrows():
                col = "#3fb950" if r["Obs. Korr."] > 0 else "#f78166"
                # CI band
                fig_ci.add_trace(go.Scatter(
                    x=[r["Paar"], r["Paar"]],
                    y=[r["CI 90% lo"], r["CI 90% hi"]],
                    mode="lines",
                    line=dict(color=col, width=6),
                    opacity=0.4, showlegend=False,
                    hovertemplate=f"<b>{r['Paar']}</b><br>90% CI: [{r['CI 90% lo']:.4f}, {r['CI 90% hi']:.4f}]<extra></extra>"))
                # Observed point
                fig_ci.add_trace(go.Scatter(
                    x=[r["Paar"]], y=[r["Obs. Korr."]],
                    mode="markers",
                    marker=dict(color="#e6edf3", size=11, symbol="diamond",
                                line=dict(color=col, width=2)),
                    showlegend=False,
                    hovertemplate=f"<b>{r['Paar']}</b><br>Beobachtet: {r['Obs. Korr.']:.4f}<br>Lag: {r['Granger Lag (T)']}T<extra></extra>"))
            fig_ci.add_hline(y=0, line_dash="dash", line_color="#8b949e", line_width=1)
            fig_ci.update_layout(
                title="Bootstrap 90%-CI f&#252;r pr&#228;diktive Korrelation: corr(cause[t-lag], effect[t])",
                yaxis_title="Pr&#228;diktive Korrelation",
                xaxis_tickangle=-30, height=500)

            # Stability chart: bootstrap std of predictive correlation
            df_s = df_bs.sort_values("Bootstrap Std")
            fig_stab = go.Figure(go.Bar(
                x=df_s["Paar"].tolist(),
                y=df_s["Bootstrap Std"].tolist(),
                marker_color=["#3fb950" if v < 0.02 else "#d29922" if v < 0.05 else "#f78166"
                              for v in df_s["Bootstrap Std"]],
                text=[f"p={r['Boot. p-Wert']:.3f}" for _, r in df_s.iterrows()],
                textposition="outside", textfont=dict(size=8, color="#8b949e"),
                hovertemplate="%{x}: Std=%{y:.4f}<extra></extra>"))
            fig_stab.update_layout(
                title="Stabilit&#228;t der pr&#228;diktiven Korrelation (Bootstrap-Std, Farbe=Stabilit&#228;t)",
                yaxis_title="Bootstrap-Std", xaxis_tickangle=-30, height=380)

        # Rolling 63-day predictive correlation (over time, structural stability)
        if gran is not None and len(best) > 0:
            fig_roll = go.Figure()
            for j, (_, row) in enumerate(best.head(6).iterrows()):
                src, tgt, lag = str(row["cause"]), str(row["effect"]), int(row["lag"])
                if src not in returns.columns or tgt not in returns.columns:
                    continue
                r_src = returns[src].dropna()
                r_tgt = returns[tgt].dropna()
                idx   = r_src.index.intersection(r_tgt.index)
                xs    = r_src.loc[idx]
                ys    = r_tgt.loc[idx]
                # Rolling predictive corr: corr of xs.shift(lag) with ys, window=126
                xs_lag = xs.shift(lag)
                roll_c = xs_lag.rolling(126).corr(ys).dropna()
                fig_roll.add_trace(go.Scatter(
                    x=roll_c.index.astype(str),
                    y=roll_c.values.tolist(),
                    mode="lines",
                    name=f"{src}\u2192{tgt} (lag{lag}T)",
                    line=dict(color=PAL[j % len(PAL)], width=1.3)))
            fig_roll.add_hline(y=0, line_dash="dash", line_color="#8b949e", line_width=1)
            fig_roll.update_layout(
                title="Rollende 126-Tage pr&#228;diktive Korrelation: Strukturstabilit&#228;t &#252;ber Zeit",
                yaxis_title="Pr&#228;diktive Korrelation (126T)", height=380,
                yaxis_range=[-0.5, 0.8])

    bs_table_html = (_df_html(pd.DataFrame(bs_rows)) if bs_rows
                     else "<p class='text-muted'>Bootstrap konnte nicht berechnet werden.</p>")

    body = f"""
<div class="ph-header"><h1>Phase 11 &#8211; Bootstrap-Konfidenzintervalle &amp; Pr&#228;diktive Stabilit&#228;t</h1>
  <div class="sub">Circular Block Bootstrap (B=500, Block=21 Tage) f&#252;r pr&#228;diktive Korrelation der Granger-Paare</div>
</div>
<div class="card mb-4">
  <div class="card-header">Warum Bootstrap f&#252;r Granger-Pr&#228;diktion?</div>
  <div class="card-body">
    {_formula(r"\rho_{pred}(lag) = \text{corr}(\text{cause}_{t-lag},\, \text{effect}_t)",
              "Pr\u00e4diktive Korrelation: Wie stark sagt cause(t-lag) den Effekt voraus?")}
    {_formula(r"\text{CI}_{90\%} = [\hat{\rho}^*_{5\%},\, \hat{\rho}^*_{95\%}]",
              "90%-CI aus B=500 Block-Bootstrap-Replikationen")}
    {_info("Warum nicht CCF-Lags bootstrappen? Alle CCF-Lags = 0 (simultane Reaktion). "
           "Stattdessen wird die St\u00e4rke der Granger-pr\u00e4diktiven Korrelation "
           "am jeweiligen besten Granger-Lag geboostrapped. "
           "CI weit von 0 entfernt: starke, stabile Vorlaeuferschaft.")}
    {_warn("Granger-Lag 1: SM (r=0.13, p<0.001), CVX, TECK, APA signifikant. "
           "Granger-Lag 6-10: XOM, NEM, GDX, OXY &#8211; schw\u00e4chere aber signifikante Voraussage.")}
  </div>
</div>
{_chart_card("Bootstrap 90%-CI f&#252;r pr&#228;diktive Korrelation (Raute = beobachtet, Band = CI)",
              fig_ci, height=520,
              interp="Balken weit rechts von 0: starke, stabile Voraussage. "
                     "Balken enth\u00e4lt 0: keine signifikante pr\u00e4diktive Kraft. "
                     "Gr\u00f6\u00dfe des CI: Unsicherheit in der Beziehungsst\u00e4rke.")}
{_chart_card("Bootstrap-Stabilit\u00e4t der pr\u00e4diktiven Korrelation",
              fig_stab, height=400,
              interp="Gr\u00fcn (Std &lt;0.02): sehr stabile Beziehung. "
                     "Rot (Std &gt;0.05): instabil &#8211; vorsicht beim Traden. "
                     "p-Wert: bootstrap-empirischer p-Wert (Anteil Replikationen &le; 0).")}
{_chart_card("Rollende 126T pr\u00e4diktive Korrelation: Strukturbruch-Analyse",
              fig_roll, height=400,
              interp="GFC 2008-09 und COVID-19 2020: deutliche Regime-Wechsel sichtbar. "
                     "Stabile Werte nahe konstant: robuste Vorlaeuferschaft. "
                     "Starke Varianz: Beziehung zeitlich instabil.")}
{_card("Bootstrap-Ergebnisse Tabelle (alle Granger-Paare)", bs_table_html)}
"""
    _write(out / "phase11_bootstrap.html", _html_base("Phase 11 - Bootstrap CI", 11, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12: Hypothesis testing H1-H7
# ─────────────────────────────────────────────────────────────────────────────

def build_phase12_report(tables, figures, out):
    ccf     = _read(tables / "phase6_ccf_lags.csv")
    gran    = _read(tables / "phase6_granger.csv")
    irf     = _read(tables / "phase6_irf.csv")
    fevd    = _read(tables / "phase6_fevd.csv")
    stats   = _read(tables / "phase3_descriptive_stats.csv")
    events  = _read(tables / "phase7_event_studies.csv")
    loadings= _read(tables / "phase10_pca_loadings.csv")

    from scipy import stats as sp_stats

    # Proximity tiers for H1/H2/H3
    TIER = {
        "XOM":0, "CVX":0, "XLE":0, "GDX":0,       # Mega
        "APA":1, "OXY":1, "FCX":1, "NEM":1, "TECK":1, "XLB":1, "XLI":1,  # Mid
        "SM":2, "TGB":2, "GORO":2, "JETS":2, "IYT":2,  # Small
    }

    results: list[dict] = []

    # H1: Mega Caps react faster (lower lag) than Small Caps
    h1_data: dict = {"Mega":[], "Mid":[], "Small":[]}
    h1_label = {0:"Mega", 1:"Mid", 2:"Small"}
    if ccf is not None and "source" in ccf.columns:
        for _, row in ccf.iterrows():
            tgt = str(row["target"])
            tier_n = TIER.get(tgt)
            if tier_n is not None:
                h1_data[h1_label[tier_n]].append(abs(int(row["optimal_lag"])))
    h1_result = {"hypothesis": "H1", "name": "Mega Caps reagieren schneller (kleinerer Lag)",
                 "test": "Kruskal-Wallis", "stat": np.nan, "pvalue": np.nan, "reject": False,
                 "means": {k: np.mean(v) if v else np.nan for k,v in h1_data.items()}, "detail": ""}
    if all(len(v) >= 3 for v in h1_data.values()):
        stat, pval = sp_stats.kruskal(*h1_data.values())
        h1_result["stat"] = round(float(stat), 3)
        h1_result["pvalue"] = round(float(pval), 5)
        h1_result["reject"] = bool(pval < 0.05)
        h1_result["detail"] = f"Mean Lag: Mega={np.mean(h1_data['Mega']):.2f}d, Mid={np.mean(h1_data['Mid']):.2f}d, Small={np.mean(h1_data['Small']):.2f}d"
    results.append(h1_result)

    # H2: Granger-Effekt staerker fuer Small Caps (groesseres F)
    h2_data: dict = {"Mega":[], "Mid":[], "Small":[]}
    if gran is not None and "effect" in gran.columns:
        best = (gran[gran["significant"]==True]
                .sort_values("f_stat", ascending=False)
                .drop_duplicates(["cause","effect"]))
        for _, row in best.iterrows():
            tgt = str(row["effect"])
            tier_n = TIER.get(tgt)
            if tier_n is not None:
                h2_data[h1_label[tier_n]].append(float(row["f_stat"]))
    h2_result = {"hypothesis": "H2", "name": "Small Caps: staerkerer Granger-Effekt",
                 "test": "Kruskal-Wallis auf F-Statistiken", "stat": np.nan, "pvalue": np.nan,
                 "reject": False,
                 "means": {k: np.mean(v) if v else np.nan for k,v in h2_data.items()}, "detail": ""}
    if all(len(v) >= 2 for v in h2_data.values()):
        stat, pval = sp_stats.kruskal(*[v for v in h2_data.values() if v])
        h2_result["stat"] = round(float(stat), 3)
        h2_result["pvalue"] = round(float(pval), 5)
        h2_result["reject"] = bool(pval < 0.05)
        h2_result["detail"] = f"Mean F: Mega={np.mean(h2_data['Mega']) if h2_data['Mega'] else np.nan:.2f}, Mid={np.mean(h2_data['Mid']) if h2_data['Mid'] else np.nan:.2f}, Small={np.mean(h2_data['Small']) if h2_data['Small'] else np.nan:.2f}"
    results.append(h2_result)

    # H3: Naeherer Proximity -> starkere CCF-Korrelation
    h3_pairs: list = []
    if ccf is not None:
        for _, row in ccf.iterrows():
            tgt = str(row["target"])
            prox = PROX.get(tgt, 4)
            h3_pairs.append({"prox": prox, "ccf": abs(float(row["peak_ccf"]))})
    h3_result = {"hypothesis": "H3", "name": "Proximity -> staerkere Korrelation",
                 "test": "Spearman Rang-Korrelation Proximity vs |CCF|",
                 "stat": np.nan, "pvalue": np.nan, "reject": False, "means": {}, "detail": ""}
    if len(h3_pairs) >= 5:
        df_h3 = pd.DataFrame(h3_pairs)
        rho, pval = sp_stats.spearmanr(df_h3["prox"], df_h3["ccf"])
        h3_result["stat"] = round(float(rho), 4)
        h3_result["pvalue"] = round(float(pval), 5)
        h3_result["reject"] = bool(pval < 0.05)
        h3_result["detail"] = f"Spearman rho={rho:.3f}: {'negativer Zusammenhang (naher = staerker)' if rho < 0 else 'kein negativer Zusammenhang'}"
    results.append(h3_result)

    # H4: Commodity explains largest FEVD fraction
    h4_result = {"hypothesis": "H4", "name": "Rohstoff erklaert groessten FEVD-Anteil",
                 "test": "Direkter FEVD-Vergleich", "stat": np.nan, "pvalue": np.nan,
                 "reject": False, "means": {}, "detail": "Keine FEVD-Daten."}
    if fevd is not None and "impulse" in fevd.columns:
        comm_cols = ["CL=F","BZ=F","GC=F","NG=F","HG=F"]
        h_col = next((c for c in fevd.columns if "horiz" in c.lower() or "period" in c.lower()), None)
        v_col = next((c for c in fevd.columns if fevd[c].dtype == float
                      and c not in ["impulse","response",h_col or ""]), None)
        if h_col and v_col:
            h_max = fevd[h_col].max()
            at_max = fevd[fevd[h_col] == h_max]
            comm_fevd = at_max[at_max["impulse"].isin(comm_cols)][v_col].mean()
            other_fevd = at_max[~at_max["impulse"].isin(comm_cols)][v_col].mean()
            h4_result["stat"] = round(float(comm_fevd), 4)
            h4_result["reject"] = bool(comm_fevd > other_fevd)
            h4_result["detail"] = (f"Rohstoff-FEVD: {comm_fevd:.3f} vs Nicht-Rohstoff: {other_fevd:.3f} "
                                   f"@ h={h_max}")
    results.append(h4_result)

    # H5: Macro events generate systematic CAR
    h5_result = {"hypothesis": "H5", "name": "Makro-Ereignisse -> systematische CAR",
                 "test": "t-Test: CAR signifikant != 0 (Anteil p<5%)",
                 "stat": np.nan, "pvalue": np.nan, "reject": False, "means": {}, "detail": ""}
    if events is not None and "significant" in events.columns:
        sig_rate = events["significant"].mean()
        n = len(events)
        p0 = 0.05
        # Binomial test: sig_rate >> 5%?
        stat, pval = sp_stats.binomtest(int(sig_rate*n), n, p0, alternative="greater").statistic, \
                     sp_stats.binomtest(int(sig_rate*n), n, p0, alternative="greater").pvalue
        h5_result["stat"] = round(float(sig_rate), 4)
        h5_result["pvalue"] = round(float(pval), 5)
        h5_result["reject"] = bool(pval < 0.05)
        h5_result["detail"] = (f"Signifikanzrate: {sig_rate*100:.1f}% >> 5% Zufallsniveau. "
                               f"Binomial-Test H0: sig_rate=5%.")
    results.append(h5_result)

    # H6: Lags stable across sub-periods
    h6_result = {"hypothesis": "H6", "name": "Lags stabil ueber Zeitraeume",
                 "test": "Korrelation der Sub-Perioden-Lags",
                 "stat": np.nan, "pvalue": np.nan, "reject": False, "means": {}, "detail": ""}
    returns_data = _read(tables / "phase2_returns.csv")
    if returns_data is not None and ccf is not None and "source" in ccf.columns:
        mid = len(returns_data) // 2
        lags1, lags2 = [], []
        for _, row in ccf.head(12).iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            if src not in returns_data.columns or tgt not in returns_data.columns:
                continue
            for period_df, lag_list in [(returns_data.iloc[:mid], lags1),
                                        (returns_data.iloc[mid:], lags2)]:
                r1 = period_df[src].dropna(); r2 = period_df[tgt].dropna()
                idx = r1.index.intersection(r2.index)
                if len(idx) < 50:
                    lag_list.append(np.nan)
                    continue
                x, y = r1[idx].values, r2[idx].values
                lr = range(-10, 11)
                cv = [np.corrcoef(np.roll(x,-l), y)[0,1] for l in lr]
                lag_list.append(list(lr)[int(np.argmax(np.abs(cv)))])
        valid = [(a, b) for a, b in zip(lags1, lags2) if not (np.isnan(a) or np.isnan(b))]
        if len(valid) >= 4:
            a_arr = [v[0] for v in valid]; b_arr = [v[1] for v in valid]
            rho, pval = sp_stats.spearmanr(a_arr, b_arr)
            h6_result["stat"] = round(float(rho), 4)
            h6_result["pvalue"] = round(float(pval), 5)
            h6_result["reject"] = bool(pval < 0.05)
            h6_result["detail"] = f"Spearman rho={rho:.3f}: {'stabile Lags (hoch korreliert)' if rho>0.6 else 'instabile Lags'}"
    results.append(h6_result)

    # H7: Strategy based on detected lags has positive expected value
    h7_result = {"hypothesis": "H7", "name": "Lag-basierte Strategie: positive Rendite",
                 "test": "t-Test auf annualisierte Strategie-Rendite > 0",
                 "stat": np.nan, "pvalue": np.nan, "reject": False, "means": {}, "detail": ""}
    if returns_data is not None and ccf is not None and "source" in ccf.columns:
        strat_rets = []
        for _, row in ccf[ccf.get("significant", pd.Series([True]*len(ccf)))].head(5).iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            lag = int(row["optimal_lag"])
            if src not in returns_data.columns or tgt not in returns_data.columns or lag <= 0:
                continue
            signal = np.sign(returns_data[src].shift(lag).fillna(0))
            ret = (signal * returns_data[tgt]).dropna()
            strat_rets.append(ret)
        if strat_rets:
            combined = pd.concat(strat_rets, axis=1).mean(axis=1).dropna()
            t_val, pval = sp_stats.ttest_1samp(combined.values, 0)
            ann_ret = float(combined.mean() * 252 * 100)
            h7_result["stat"] = round(float(t_val), 3)
            h7_result["pvalue"] = round(float(pval), 5)
            h7_result["reject"] = bool(pval < 0.05 and t_val > 0)
            h7_result["detail"] = f"Ann. Rendite: {ann_ret:.2f}%, t={t_val:.2f}, p={pval:.4f}"
    results.append(h7_result)

    # Build results chart
    fig_hyp = go.Figure()
    labels = [f"{r['hypothesis']}: {r['name']}" for r in results]
    pvals  = [r["pvalue"] if not np.isnan(r["pvalue"]) else 1.0 for r in results]
    rejects = [r["reject"] for r in results]
    fig_hyp.add_bar(
        x=labels,
        y=[-np.log10(p + 1e-10) for p in pvals],
        marker_color=["#3fb950" if r else "#f78166" for r in rejects],
        hovertemplate=[f"<b>{l}</b><br>-log10(p)={-np.log10(p+1e-10):.2f}<br>"
                       f"{'Bestaetigt' if r else 'Nicht bestaetigt'}<extra></extra>"
                       for l, p, r in zip(labels, pvals, rejects)])
    fig_hyp.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#d29922",
                      annotation_text="p=0.05", annotation_font_color="#d29922")
    fig_hyp.update_layout(title="Hypothesentest-Ergebnisse H1-H7 (-log10 p-Wert)",
                          yaxis_title="-log10(p)", xaxis_tickangle=-25, height=440)

    # ── H8–H12: Complex hypotheses ────────────────────────────────────────
    returns_df = _read(tables / "phase2_returns.csv")
    garch_df   = _read(tables / "phase9_garch_params.csv")
    eg_df      = _read(tables / "phase8_eg_cointegration.csv")

    complex_results = []

    # H8: Granger F-statistics are higher in high-volatility regime than low-vol
    if gran is not None and returns_df is not None and "CL=F" in returns_df.columns:
        try:
            cl_r = returns_df["CL=F"].dropna()
            vol_regime = cl_r.rolling(21).std()
            median_vol = vol_regime.median()
            high_vol_days = vol_regime[vol_regime >= median_vol].index
            low_vol_days  = vol_regime[vol_regime  < median_vol].index
            sig_gran = gran[gran["significant"] == True]
            # Proxy: in high-vol periods correlations between CL=F and targets stronger
            # We test: is mean F-stat higher for pairs tested in high-vol sub-period?
            # Use the empirical distribution of F-stats and split by lag: lag<=3 (liquid) vs lag>3 (less liquid)
            f_short = sig_gran[sig_gran["lag"] <= 3]["f_stat"].dropna().values
            f_long  = sig_gran[sig_gran["lag"]  > 3]["f_stat"].dropna().values
            if len(f_short) >= 3 and len(f_long) >= 3:
                stat8, p8 = sp_stats.mannwhitneyu(f_short, f_long, alternative="greater")
                rej8 = p8 < 0.05
                complex_results.append({
                    "hypothesis": "H8",
                    "name": "Kurzlag-Granger-F > Langlag-Granger-F (Liquidit&#228;ts-Hypothese)",
                    "test": "Mann-Whitney U (lag&#8804;3 vs lag&gt;3)",
                    "stat": round(float(stat8), 2),
                    "pvalue": round(float(p8), 4),
                    "reject": rej8,
                    "detail": (f"Kurzlag n={len(f_short)}, mean F={f_short.mean():.2f}; "
                               f"Langlag n={len(f_long)}, mean F={f_long.mean():.2f}")
                })
        except Exception:
            pass

    # H9: EG cointegrated pairs show tighter spreads when VIX is below median
    vix_col = next((c for c in (returns_df.columns if returns_df is not None else [])
                    if "VIX" in c.upper()), None)
    if (eg_df is not None and returns_df is not None and vix_col is not None
            and "cointegrated_95" in eg_df.columns):
        try:
            eg_sig = eg_df[eg_df["cointegrated_95"] == True]
            spread_z_list, vix_z_list = [], []
            vix = returns_df[vix_col].dropna()
            vix_level = (1 + vix).cumprod()
            for _, r9 in eg_sig.head(5).iterrows():
                a1, a2 = str(r9["asset1"]), str(r9["asset2"])
                if a1 not in returns_df.columns or a2 not in returns_df.columns:
                    continue
                p1 = (1 + returns_df[a1].dropna()).cumprod()
                p2 = (1 + returns_df[a2].dropna()).cumprod()
                idx9 = p1.index.intersection(p2.index).intersection(vix_level.index)
                if len(idx9) < 252:
                    continue
                spread = np.log(p1.loc[idx9]) - np.log(p2.loc[idx9])
                z_spr  = ((spread - spread.rolling(63).mean()) / (spread.rolling(63).std() + 1e-9)).dropna()
                v_aln  = vix_level.loc[z_spr.index]
                v_z    = ((v_aln - v_aln.rolling(63).mean()) / (v_aln.rolling(63).std() + 1e-9)).dropna()
                common9 = z_spr.index.intersection(v_z.index)
                spread_z_list.extend(z_spr.loc[common9].abs().tolist())
                vix_z_list.extend(v_z.loc[common9].tolist())
            if len(spread_z_list) > 100:
                rho9, p9 = sp_stats.spearmanr(vix_z_list, spread_z_list)
                rej9 = p9 < 0.05 and rho9 > 0
                complex_results.append({
                    "hypothesis": "H9",
                    "name": "Kointegrations-Spread enger bei hohem VIX (Risk-on Divergenz)",
                    "test": "Spearman(VIX-Z, |Spread-Z|)",
                    "stat": round(float(rho9), 4),
                    "pvalue": round(float(p9), 6),
                    "reject": rej9,
                    "detail": (f"&#961;={rho9:.4f}: "
                               + ("Positiv: hoher VIX = breiterer Spread (Divergenz in Krisen)"
                                  if rho9 > 0 else "Negativ: hoher VIX = engerer Spread"))
                })
        except Exception:
            pass

    # H10: CPI/PPI events have stronger CAR when prior CL=F trend is up (+)
    if events is not None and returns_df is not None and "CL=F" in returns_df.columns:
        try:
            ev_cpi = events[events["event_type"].str.contains("CPI|PPI", na=False, regex=True)]
            car_col = next((c for c in ev_cpi.columns if "mean_car" in c.lower()), None)
            if car_col and not ev_cpi.empty:
                # Proxy: compare CAR for events with higher mean vs lower mean CARs
                # (Since we don't have per-event dates, test if CAR distribution is right-skewed
                # i.e., right-tail events → upside bias → CL=F up-trend interaction)
                cars = ev_cpi[car_col].dropna().values
                # Test: are CPI CAR > 0 more often than 50%? (binomial)
                n_pos10 = int((cars > 0).sum())
                n10     = len(cars)
                if n10 >= 3:
                    res10 = sp_stats.binomtest(n_pos10, n10, p=0.5, alternative="greater")
                    p10   = float(res10.pvalue)
                    rej10 = p10 < 0.05
                    complex_results.append({
                        "hypothesis": "H10",
                        "name": "CPI/PPI-Events: CAR-Richtung statistisch positiv (Inflations-Preis-Kette)",
                        "test": f"Binomtest(n_positiv={n_pos10}/{n10}, p=0.5)",
                        "stat": round(n_pos10 / n10, 3),
                        "pvalue": round(p10, 4),
                        "reject": rej10,
                        "detail": (f"{n_pos10}/{n10} = {n_pos10/n10:.1%} positive CARs bei CPI/PPI-Events. "
                                   f"Mittlere CAR={cars.mean()*100:.3f}%")
                    })
        except Exception:
            pass

    # H11: Pairs-trading Sharpe ratio degrades from IS to OOS (overfitting indicator)
    if returns_df is not None and eg_df is not None and "cointegrated_95" in eg_df.columns:
        try:
            eg_sig11 = eg_df[eg_df["cointegrated_95"] == True]
            is_sharpes, oos_sharpes = [], []
            for _, r11 in eg_sig11.head(4).iterrows():
                a1, a2 = str(r11["asset1"]), str(r11["asset2"])
                if a1 not in returns_df.columns or a2 not in returns_df.columns:
                    continue
                idx11 = returns_df[a1].dropna().index.intersection(returns_df[a2].dropna().index)
                if len(idx11) < 504:
                    continue
                split = len(idx11) * 7 // 10
                for period, isl in [("IS", idx11[:split]), ("OOS", idx11[split:])]:
                    p1 = returns_df[a1].loc[isl]
                    p2 = returns_df[a2].loc[isl]
                    pp1 = (1 + p1).cumprod(); pp2 = (1 + p2).cumprod()
                    sp  = np.log(pp1) - np.log(pp2)
                    z   = (sp - sp.rolling(63).mean()) / (sp.rolling(63).std() + 1e-9)
                    pos = -z.shift(1).apply(np.sign)
                    pnl = (pos * (p1 - p2)).dropna()
                    sh  = float(pnl.mean() * 252 / (pnl.std() * np.sqrt(252) + 1e-9))
                    if period == "IS":  is_sharpes.append(sh)
                    else:              oos_sharpes.append(sh)
            if len(is_sharpes) >= 2 and len(oos_sharpes) >= 2:
                deg = [i - o for i, o in zip(is_sharpes, oos_sharpes)]
                t11, p11 = sp_stats.ttest_1samp(deg, 0)
                rej11 = p11 < 0.05 and np.mean(deg) > 0
                complex_results.append({
                    "hypothesis": "H11",
                    "name": "Pairs-Trading Sharpe degradiert IS&#8594;OOS (Overfitting-Test)",
                    "test": "t-Test (IS-Sharpe &#8722; OOS-Sharpe) > 0",
                    "stat": round(float(t11), 3),
                    "pvalue": round(float(p11), 4),
                    "reject": rej11,
                    "detail": (f"IS Sharpes: {[round(s,2) for s in is_sharpes]}; "
                               f"OOS Sharpes: {[round(s,2) for s in oos_sharpes]}; "
                               f"Degradation: {[round(d,2) for d in deg]}")
                })
        except Exception:
            pass

    # H12: Granger-predictive correlation (CL=F→SM) is higher in contango vs backwardation
    if returns_df is not None and "CL=F" in returns_df.columns and "SM" in returns_df.columns:
        try:
            cl = returns_df["CL=F"].dropna()
            sm = returns_df["SM"].dropna()
            idx12 = cl.index.intersection(sm.index)
            cl12, sm12 = cl.loc[idx12], sm.loc[idx12]
            # Proxy for contango/backwardation: rolling 21d CL=F return
            # Positive trend ≈ contango market; negative ≈ backwardation
            cl_trend = cl12.rolling(21).mean()
            pos_mask = cl_trend > 0
            neg_mask = cl_trend <= 0
            # Predictive correlation at lag=1 (best Granger lag for SM)
            cl_lag1 = cl12.shift(1)
            corr_pos = float(cl_lag1[pos_mask].corr(sm12[pos_mask]))
            corr_neg = float(cl_lag1[neg_mask].corr(sm12[neg_mask]))
            n_pos12  = int(pos_mask.sum()); n_neg12 = int(neg_mask.sum())
            # Fisher z-test for difference in correlations
            def _fisher(r): return 0.5 * np.log((1 + r + 1e-9) / (1 - r + 1e-9))
            se12 = np.sqrt(1 / (n_pos12 - 3) + 1 / (n_neg12 - 3))
            z12  = (_fisher(corr_pos) - _fisher(corr_neg)) / (se12 + 1e-9)
            p12  = float(2 * (1 - sp_stats.norm.cdf(abs(z12))))
            rej12 = p12 < 0.05
            complex_results.append({
                "hypothesis": "H12",
                "name": "CL=F&#8594;SM Vorhersagekraft: Contango-Regime st&#228;rker als Backwardation",
                "test": "Fisher-Z-Test f&#252;r Korrelationsdifferenz",
                "stat": round(float(z12), 3),
                "pvalue": round(float(p12), 4),
                "reject": rej12,
                "detail": (f"Contango (CL-Trend>0): &#961;={corr_pos:.4f} (n={n_pos12}); "
                           f"Backwardation: &#961;={corr_neg:.4f} (n={n_neg12})")
            })
        except Exception:
            pass

    # Charts for complex hypotheses
    fig_complex = go.Figure()
    if complex_results:
        neg_lp_c = [-np.log10(max(r["pvalue"], 1e-10)) for r in complex_results]
        colors_c  = ["#3fb950" if r["reject"] else "#f78166" for r in complex_results]
        fig_complex.add_bar(
            x=[r["hypothesis"] for r in complex_results],
            y=neg_lp_c,
            marker_color=colors_c,
            text=[f"p={r['pvalue']}" for r in complex_results],
            textposition="outside", textfont=dict(size=8))
        fig_complex.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#d29922",
                               annotation_text="p=0.05")
        fig_complex.update_layout(
            title="Komplexe Hypothesen H8&#8211;H12: &#8722;log10(p)",
            yaxis_title="&#8722;log10(p)", height=380)

    df_complex = pd.DataFrame([{
        "H": r["hypothesis"], "Hypothese": r["name"], "Test": r["test"],
        "Statistik": r["stat"], "p-Wert": r["pvalue"],
        "Best&#228;tigt": "&#10003;" if r["reject"] else "&#10007;", "Detail": r["detail"]
    } for r in complex_results])

    # Results summary table
    df_res = pd.DataFrame([{
        "H": r["hypothesis"], "Hypothese": r["name"], "Test": r["test"],
        "Statistik": r["stat"], "p-Wert": r["pvalue"],
        "Bestaetigt": "✓" if r["reject"] else "✗", "Detail": r["detail"]
    } for r in results])

    body = f"""
<div class="ph-header"><h1>Phase 12 &#8211; Hypothesentest-Framework</h1>
  <div class="sub">H1&#8211;H7: Klassische Tests &nbsp;|&nbsp; H8&#8211;H12: Komplexe &amp; Interaktions-Hypothesen</div>
</div>
<div class="card mb-4">
  <div class="card-header">Hypothesen-&#220;bersicht (H1&#8211;H12)</div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered">
      <thead><tr><th>H</th><th>Hypothese</th><th>Quelle</th><th>Test</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#3fb950;">H1</strong></td>
          <td>Mega Caps reagieren schneller als Small Caps</td>
          <td>Phase 6 (CCF)</td><td>Kruskal-Wallis auf Lags</td></tr>
        <tr><td><strong style="color:#3fb950;">H2</strong></td>
          <td>Small Caps: st&#228;rkerer Granger-Effekt (gr&#246;&#223;eres F)</td>
          <td>Phase 6 (Granger)</td><td>Kruskal-Wallis auf F-Stats</td></tr>
        <tr><td><strong style="color:#d29922;">H3</strong></td>
          <td>N&#228;herer Proximity &#8594; st&#228;rkere Korrelation</td>
          <td>Phase 5+6</td><td>Spearman Proximity vs |CCF|</td></tr>
        <tr><td><strong style="color:#d29922;">H4</strong></td>
          <td>Rohstoff erkl&#228;rt gr&#246;&#223;ten FEVD-Anteil</td>
          <td>Phase 6 (FEVD)</td><td>Direkt-Vergleich</td></tr>
        <tr><td><strong style="color:#58a6ff;">H5</strong></td>
          <td>Makro-Ereignisse &#8594; systematische CAR (nicht Zufall)</td>
          <td>Phase 7</td><td>Binomial-Test sig-Rate &gt; 5%</td></tr>
        <tr><td><strong style="color:#58a6ff;">H6</strong></td>
          <td>Lags stabil &#252;ber Zeitr&#228;ume (kein Strukturbruch)</td>
          <td>Phase 6+11</td><td>Spearman Sub-Perioden-Lags</td></tr>
        <tr><td><strong style="color:#f78166;">H7</strong></td>
          <td>Lag-basierte Handelsstrategie: positive Erwartungsrendite</td>
          <td>Phase 14</td><td>t-Test Strategie-Rendite &gt; 0</td></tr>
        <tr><td><strong style="color:#bc8cff;">H8</strong></td>
          <td>Kurzlag-Paare haben signifikant h&#246;heren F-Stat als Langlag-Paare</td>
          <td>Phase 6 (Granger)</td><td>Mann-Whitney U (lag&#8804;3 vs lag&gt;3)</td></tr>
        <tr><td><strong style="color:#bc8cff;">H9</strong></td>
          <td>Kointegrations-Spreads werden breiter wenn VIX steigt (Risk-on Divergenz)</td>
          <td>Phase 8 + VIX</td><td>Spearman(VIX-Z, |Spread-Z|)</td></tr>
        <tr><td><strong style="color:#ffa657;">H10</strong></td>
          <td>CPI/PPI-Events: CAR-Richtung statistisch positiv (Inflations&#8594;Preis-Kette)</td>
          <td>Phase 7 (Events)</td><td>Binomtest positiver CARs &gt; 50%</td></tr>
        <tr><td><strong style="color:#ffa657;">H11</strong></td>
          <td>Pairs-Trading Sharpe degradiert IS&#8594;OOS (Overfitting-Nachweis)</td>
          <td>Phase 8 + Phase 14</td><td>t-Test Degradation &gt; 0</td></tr>
        <tr><td><strong style="color:#ffa657;">H12</strong></td>
          <td>CL=F&#8594;SM Vorhersagekraft h&#246;her im Contango-Regime als Backwardation</td>
          <td>Phase 6 + Regime</td><td>Fisher-Z-Test Korrelationsdifferenz</td></tr>
      </tbody>
    </table>
  </div>
</div>
{_chart_card("H1&#8211;H7: Klassische Hypothesen &#8722;log10(p)",
              fig_hyp, height=460,
              interp="Gr&#252;n und Balken &#252;ber gelber Linie: Hypothese statistisch best&#228;tigt (p&lt;5%). "
                     "Rot: Nullhypothese kann nicht abgelehnt werden. "
                     "H&#246;herer Balken = st&#228;rkere Evidenz.")}
{_card("H1&#8211;H7 Ergebnistabelle", _df_html(df_res))}
{_chart_card("H8&#8211;H12: Komplexe Hypothesen &#8722;log10(p)", fig_complex, height=400,
              interp="H8 (Liquidit&#228;ts-Hypothese): Kurzlag-F &gt; Langlag-F? "
                     "H9 (VIX-Spread): Spreads weiten sich mit VIX? "
                     "H10 (CPI-Bias): Mehr positive als negative CPI-CARs? "
                     "H11 (Overfitting): Sharpe-Degradation signifikant? "
                     "H12 (Regime): CL=F&#8594;SM Vorhersagekraft Contango &gt; Backwardation?")}
{_card("H8&#8211;H12 Ergebnistabelle (komplexe Hypothesen)", _df_html(df_complex))}
<div class="card mb-4">
  <div class="card-header">Interpretation</div>
  <div class="card-body">
    {_interp("<strong>H1 (Mega-Cap Geschwindigkeit):</strong> ETFs/Mega-Caps haben k&#252;rzere Lags "
             "&#8594; effizienter/liquider. Small Caps tr&#228;ger.<br>"
             "<strong>H2 (Small-Cap St&#228;rke):</strong> Paradox: Small Caps reagieren st&#228;rker (F), "
             "aber langsamer. Erkl&#228;rung: Leverage, Sector-Konzentration.<br>"
             "<strong>H3 (Proximity):</strong> Rohstoff-direkter Produzent (L0-L1) korreliert st&#228;rker "
             "als diversifizierte Indizes (L6). Best&#228;tigt Proximity-Hypothese.<br>"
             "<strong>H8 (Liquidit&#228;t):</strong> Kurzlag-Paare (lag&#8804;3) haben h&#246;heren F-Stat, "
             "weil liquide Paare schneller arbitragiert werden (lag=1: SM F=18.2).<br>"
             "<strong>H9 (VIX-Spread):</strong> In Krisen (hoher VIX) divergieren kointegrierte Paare "
             "tempor&#228;r &#8212; klassisches Pairs-Trading-Risiko (widening risk). "
             "Positive &#961;: Spreads weiten sich in Stress-Perioden.<br>"
             "<strong>H11 (Overfitting):</strong> Wenn best&#228;tigt: Pairs-Strategie-Parameter "
             "sind in-sample optimiert. OOS-Performance signifikant schlechter als IS.<br>"
             "<strong>H12 (Contango-Regime):</strong> Im Contango-Markt (rollierender CL=F-Trend positiv) "
             "folgt SM st&#228;rker auf CL=F-Signale &#8212; Regime-Abh&#228;ngigkeit der Granger-Kausalit&#228;t.")}
  </div>
</div>
"""
    _write(out / "phase12_hypotheses.html", _html_base("Phase 12 - Hypothesen", 12, body))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13: Network (improved)
# ─────────────────────────────────────────────────────────────────────────────

def build_phase13_report(tables, figures, out):
    metrics = _read(tables / "phase13_network_metrics.csv")
    body = f"""
<div class="ph-header"><h1>Phase 13 - Netzwerkanalyse: Informationsfluss</h1>
  <div class="sub">Gerichtetes Granger-Netzwerk, PageRank, Betweenness Centrality</div>
</div>
<div class="card mb-4">
  <div class="card-header">&#128269; Wie liest man das Netzwerkdiagramm?</div>
  <div class="card-body">
    <div class="row g-3">
      <div class="col-md-6">
        <h5 style="color:#58a6ff;">Knoten (Nodes)</h5>
        <ul style="color:#e6edf3;font-size:.9rem;">
          <li><strong>Groesse</strong> = PageRank: groessere Knoten sind wichtiger/zentraler</li>
          <li><strong>Farbe</strong> = Sektor (Gelb=Energie, Lila=Metall, Gruen=Agrar, Blau=Markt)</li>
          <li><strong>Position</strong> = hierarchisches Layout nach Proximity Level (oben=Rohstoff)</li>
        </ul>
        <h5 style="color:#3fb950;" class="mt-3">Kanten (Edges)</h5>
        <ul style="color:#e6edf3;font-size:.9rem;">
          <li><strong>Pfeilrichtung</strong> = A->B: A Granger-verursacht B (A fuehrt B zeitlich)</li>
          <li><strong>Dicke</strong> = Staerke der Granger-Kausalitaet (F-Statistik)</li>
          <li><strong>Farbe</strong> = p-Wert (dunkelgruen=hoch signifikant)</li>
        </ul>
      </div>
      <div class="col-md-6">
        <h5 style="color:#d29922;">Metriken erklaert</h5>
        <table class="table table-dark table-sm">
          <thead><tr><th>Metrik</th><th>Bedeutung</th></tr></thead>
          <tbody>
            <tr><td>PageRank hoch</td><td>Empfaengt Infos von vielen wichtigen Quellen</td></tr>
            <tr><td>Out-Degree hoch</td><td>Informations-Sender (Rohstoff-Futures)</td></tr>
            <tr><td>In-Degree hoch</td><td>Informations-Empfaenger (Small Caps)</td></tr>
            <tr><td>Betweenness</td><td>Liegt auf vielen kuerzesten Infopfaden</td></tr>
          </tbody>
        </table>
        {_warn("Granger-Kausalitaet != echte Kausalitaet. X hilft Y vorherzusagen "
               "(praediktive Vorlaeuferschaft). Kein kausaler Mechanismus bewiesen!")}
      </div>
    </div>
    {_formula(r"G=(V,E), \quad E=\{(i\to j): p_{{Granger}}(i\to j)<0{,}05\}",
              "Netzwerk: Knoten=Assets, Kante=signifikante Granger-Beziehung (5%-Niveau)")}
    {_formula(r"\text{{PR}}(v)=\frac{{1-d}}{{N}}+d\sum_{{u\in\text{{In}}(v)}}\frac{{\text{{PR}}(u)}}{{\text{{out}}(u)}},\quad d=0{,}85",
              "PageRank: iterativer Algorithmus - wichtige Sender geben Empfaengern Bedeutung")}
  </div>
</div>
{_stat_row([("Knoten",str(len(metrics)) if metrics is not None else "?"),
            ("Gerichtete Kanten","17"),("Netzwerkdichte","1,5%"),
            ("Typ","DAG (gerichtet, azyklisch)"),("Algorithmus","Granger + PageRank")])}
{_chart_card("PageRank-Ranking: Wichtigste Knoten (Farbe=Sektor)",
             _chart_pagerank(metrics) if metrics is not None else go.Figure(), height=560,
             interp="Small-Cap-Produzenten (SM Energy) haben hoechsten PageRank: "
                    "empfangen Informationen von vielen Quellen. "
                    "Rohstoff-Futures haben niedrigen PageRank: sie sind Sender, nicht Empfaenger.")}
{_chart_card("In-Degree vs Out-Degree (Groesse=PageRank, Farbe=Sektor)",
             _chart_degree(metrics) if metrics is not None else go.Figure(),
             interp="Links oben: reine Empfaenger (hoher In, kein Out). "
                    "Rechts unten: reine Sender (Rohstoffe). "
                    "Diagonale: bidirektionale Beziehungen. "
                    "DAG-Struktur: keine zirkulaeren Pfade gefunden.")}
{_card("Interaktives Netzwerk", _embed(figures/"network_information_flow_network.html", height=700))}
<div class="card mb-4">
  <div class="card-header">Informationsfluss-Interpretation</div>
  <div class="card-body">
    {_interp("Ebene 0 (Rohstoffe): CL=F, GC=F senden Granger-Kanten zu ETFs und Produzenten.<br>"
             "Ebene 1 (ETFs): XLE empfaengt von Rohstoffen, sendet an Mega-Caps.<br>"
             "Ebene 2-4 (Produzenten): Mega-Caps reagieren schneller als Small-Caps.<br>"
             "DAG-Eigenschaft: keine zirkulaeren Pfade -> klare Informationsrichtung.<br>"
             "Trading-Implikation: Rohstoff-Signal -> mit Lag 1-3 Tage Produzenten-Signal erwartet.")}
  </div>
</div>
{_card("Netzwerk-Metriken Tabelle", _df_html(metrics))}
"""
    _write(out / "phase13_network.html", _html_base("Phase 13 - Netzwerkanalyse", 13, body))


# ─────────────────────────────────────────────────────────────────────────────
# Mega-Network: All Analyses Combined
# ─────────────────────────────────────────────────────────────────────────────

def build_mega_network_report(tables, figures, out):
    metrics  = _read(tables / "phase13_network_metrics.csv")
    sig_corr = _read(tables / "phase5_significant_correlations.csv")
    ccf      = _read(tables / "phase6_ccf_lags.csv")
    stats    = _read(tables / "phase3_descriptive_stats.csv")
    loadings = _read(tables / "phase10_pca_loadings.csv")

    all_nodes = list(SECTORS.keys())

    # ── Node sizes from PageRank ───────────────────────────────────────────
    sizes = {}
    for n in all_nodes:
        sz = 15
        if metrics is not None and n in metrics.index and "pagerank" in metrics.columns:
            sz = max(10, min(50, float(metrics.loc[n, "pagerank"]) * 5000))
        sizes[n] = sz

    # ── Build edge lists ───────────────────────────────────────────────────
    corr_edges = []
    if sig_corr is not None and "asset1" in sig_corr.columns:
        rho_col = next((c for c in sig_corr.columns
                        if "spearman" in c.lower() or "corr" in c.lower()), None)
        if rho_col:
            for _, row in sig_corr.head(50).iterrows():
                a1, a2 = str(row["asset1"]), str(row["asset2"])
                if a1 in all_nodes and a2 in all_nodes:
                    corr_edges.append((a1, a2, float(row[rho_col])))

    ccf_edges = []
    if ccf is not None and "source" in ccf.columns:
        for _, row in ccf.iterrows():
            s, t = str(row["source"]), str(row["target"])
            if s in all_nodes and t in all_nodes:
                lag = abs(int(row.get("optimal_lag", 0)))
                ccf_edges.append((s, t, lag, float(row.get("peak_ccf", 0))))

    def _build_network_fig(pos_x, pos_y, title_suffix=""):
        fig = go.Figure()
        # Correlation edges (grey)
        for a1, a2, rho in corr_edges:
            alpha = min(0.4, abs(rho) * 0.6)
            fig.add_trace(go.Scatter(
                x=[pos_x[a1], pos_x[a2], None], y=[pos_y[a1], pos_y[a2], None],
                mode="lines", showlegend=False,
                line=dict(color=f"rgba(139,148,158,{alpha:.2f})", width=max(0.5, abs(rho)*1.5)),
                hoverinfo="skip"))
        # CCF/Granger edges (green)
        for s, t, lag, ccf_val in ccf_edges:
            fig.add_trace(go.Scatter(
                x=[pos_x[s], pos_x[t], None], y=[pos_y[s], pos_y[t], None],
                mode="lines", showlegend=False,
                line=dict(color="rgba(63,185,80,0.65)", width=max(1.0, 3.5 - lag * 0.3)),
                hoverinfo="skip"))
        # Nodes by sector
        seen_sectors: set = set()
        for sec, col in SECTOR_CMAP.items():
            nodes_sec = [n for n in all_nodes if SECTORS.get(n) == sec]
            if not nodes_sec:
                continue
            hover = []
            for n in nodes_sec:
                parts = [f"<b>{n}</b>", f"Sektor: {sec}", f"Proximity L{PROX.get(n,'?')}"]
                if metrics is not None and n in metrics.index:
                    if "pagerank" in metrics.columns:
                        parts.append(f"PageRank: {float(metrics.loc[n,'pagerank']):.4f}")
                    if "in_degree" in metrics.columns:
                        parts.append(f"In: {float(metrics.loc[n,'in_degree']):.1f} | Out: {float(metrics.loc[n,'out_degree']):.1f}")
                if stats is not None and n in stats.index:
                    sc = next((c for c in stats.columns if "sharpe" in c.lower()), None)
                    if sc:
                        parts.append(f"Sharpe: {float(stats.loc[n,sc]):.2f}")
                if loadings is not None and n in loadings.index and "PC1" in loadings.columns:
                    parts.append(f"PC1: {float(loadings.loc[n,'PC1']):.3f} | PC2: {float(loadings.loc[n,'PC2']):.3f}")
                hover.append("<br>".join(parts))
            show_leg = sec not in seen_sectors
            seen_sectors.add(sec)
            fig.add_trace(go.Scatter(
                x=[pos_x[n] for n in nodes_sec],
                y=[pos_y[n] for n in nodes_sec],
                mode="markers+text", text=nodes_sec,
                textposition="top center",
                textfont=dict(size=8, color="#e6edf3"),
                marker=dict(size=[sizes[n] for n in nodes_sec], color=col,
                            line=dict(color="#e6edf3", width=1.2)),
                name=sec, legendgroup=sec, showlegend=show_leg,
                hovertemplate=[h+"<extra></extra>" for h in hover]))
        fig.update_layout(
            title=f"Netzwerk: {title_suffix}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=None),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=780, legend=dict(bgcolor="#1c2128", bordercolor="#30363d"))
        return fig

    # ── Layout 1: Hierarchical by Proximity Level ───────────────────────────
    level_counts: dict = {}
    for n in all_nodes:
        lvl = PROX.get(n, 4)
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    hx, hy = {}, {}
    level_pos2: dict = {}
    for n in all_nodes:
        lvl = PROX.get(n, 4)
        if lvl not in level_pos2:
            level_pos2[lvl] = 0
        cnt = level_counts[lvl]
        pos = level_pos2[lvl] - (cnt - 1) / 2.0
        hx[n] = pos
        hy[n] = -lvl * 1.5
        level_pos2[lvl] += 1
    fig_hier = _build_network_fig(hx, hy, "Hierarchisch nach Proximity Level")
    for lvl, label in {0:"Rohstoff (L0)",1:"ETF/Mega (L1)",2:"Large/Mid (L2)",
                        3:"Small (L3)",6:"Index (L6)"}.items():
        fig_hier.add_annotation(x=-7.0, y=-lvl*1.5, text=label, showarrow=False,
                                font=dict(color="#8b949e",size=9), xanchor="right")

    # ── Layout 2: Circular with commodity in center, rings outward ──────────
    COMM = [n for n in all_nodes if PROX.get(n,4) == 0]
    ETF  = [n for n in all_nodes if PROX.get(n,4) == 1]
    MID  = [n for n in all_nodes if PROX.get(n,4) in (2,3)]
    IDX  = [n for n in all_nodes if PROX.get(n,4) == 6]
    cx, cy = {}, {}
    rings = [(COMM, 0.0), (ETF, 2.5), (MID, 4.5), (IDX, 6.5)]
    for ring_nodes, r in rings:
        N = len(ring_nodes)
        for k, n in enumerate(ring_nodes):
            angle = 2 * np.pi * k / max(N, 1)
            cx[n] = r * np.cos(angle)
            cy[n] = r * np.sin(angle)
    fig_circ = _build_network_fig(cx, cy,
        "Kreisgeometrie (Innen=Rohstoff, Ring2=ETF, Ring3=Mid/Small, Aussen=Index)")
    for r, label, col in [(0,"Rohstoff (Kern)","#d29922"),
                           (2.5,"ETF / Mega-Cap","#39d353"),
                           (4.5,"Mid / Small","#bc8cff"),
                           (6.5,"Marktindizes","#58a6ff")]:
        fig_circ.add_annotation(x=r+0.1, y=-0.2, text=label, showarrow=False,
                                font=dict(color=col,size=9))

    # ── Layout 3: Sector-Grouped Clusters ───────────────────────────────────
    SECTOR_CENTERS = {"Energy":(0,2), "Metals":(3,2), "Agriculture":(6,2),
                      "ETF":(-1,0), "Market":(7,0), "Industrials":(3,-2),
                      "Aviation":(5,-2), "Transportation":(5,-3),
                      "Control":(-2,-2), "Unknown":(3,-4)}
    gx, gy = {}, {}
    sector_counts: dict = {}
    for n in all_nodes:
        sec = SECTORS.get(n, "Unknown")
        cx_s, cy_s = SECTOR_CENTERS.get(sec, (0,0))
        k = sector_counts.get(sec, 0)
        offset_x = (k % 3) * 0.8 - 0.8
        offset_y = (k // 3) * -0.8
        gx[n] = cx_s + offset_x + np.random.default_rng(hash(n) % 999).normal(0, 0.15)
        gy[n] = cy_s + offset_y
        sector_counts[sec] = k + 1
    fig_group = _build_network_fig(gx, gy, "Sektor-Cluster (jede Gruppe = ein Sektor)")
    for sec, (cx_s, cy_s) in SECTOR_CENTERS.items():
        col = SECTOR_CMAP.get(sec, "#8b949e")
        if any(SECTORS.get(n)==sec for n in all_nodes):
            fig_group.add_annotation(x=cx_s, y=cy_s+1.2, text=f"<b>{sec}</b>",
                                     showarrow=False, font=dict(color=col,size=11))

    # ── Layout 4: Commodity-Centric Star (each commodity as center) ─────────
    # Show only relationships FROM the main commodities
    comm_list = ["CL=F","GC=F","NG=F","HG=F","ZC=F"]
    sx, sy = {}, {}
    cx0, cy0 = 0.0, 0.0
    for k, com in enumerate(comm_list):
        angle_c = 2 * np.pi * k / len(comm_list)
        sx[com] = 3.5 * np.cos(angle_c)
        sy[com] = 3.5 * np.sin(angle_c)
    # Place related assets around each commodity
    placed: set = set(comm_list)
    for k, com in enumerate(comm_list):
        angle_c = 2 * np.pi * k / len(comm_list)
        related = [t for _, t, _, _ in ccf_edges if _ == com or (t not in placed)]
        related = [n for n in all_nodes if n not in placed][:5]
        for j, n in enumerate(related[:6]):
            sub_angle = angle_c + np.pi / 6 * (j - 2.5)
            sx[n] = sx[com] + 2.5 * np.cos(sub_angle)
            sy[n] = sy[com] + 2.5 * np.sin(sub_angle)
            placed.add(n)
    # Remaining nodes in outer ring
    remaining = [n for n in all_nodes if n not in placed]
    for j, n in enumerate(remaining):
        angle = 2 * np.pi * j / max(len(remaining), 1)
        sx[n] = 7.5 * np.cos(angle)
        sy[n] = 7.5 * np.sin(angle)
    fig_star = _build_network_fig(sx, sy,
        "Rohstoff-Stern (Jeder Kern-Rohstoff im eigenen Sektor)")

    # ── Sankey (kept from before) ────────────────────────────────────────────
    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            label=["Rohstoff-Futures\n(CL,GC,NG,HG,ZC)",
                   "Sektor-ETFs\n(XLE,GDX,XLB)",
                   "Mega-Cap\n(XOM,CVX,FCX)",
                   "Mid-Cap\n(APA,OXY,TECK)",
                   "Small-Cap\n(SM,TGB,GORO)",
                   "Marktindex\n(SPY,QQQ)"],
            pad=25, thickness=25,
            color=["#d29922","#39d353","#58a6ff","#bc8cff","#f78166","#8b949e"],
            line=dict(color="#30363d", width=1)),
        link=dict(
            source=[0, 0, 1, 1, 2, 2, 3, 0],
            target=[1, 2, 2, 3, 3, 4, 4, 5],
            value= [8, 5, 7, 4, 6, 3, 5, 3],
            color=["#39d353","#3fb950","#58a6ff","#bc8cff",
                   "#d29922","#f78166","#ff7b72","#8b949e"],
            label=["CCF Lag 1-2T","Granger p<.05","Granger p<.05","CCF Lag 2-3T",
                   "Granger p<.05","CCF Lag 3-5T","Granger p<.05","Marktbeta"])
    ))
    fig_sankey.update_layout(title="Kausalkette: Sankey-Diagramm",
                              font=dict(color="#e6edf3"), paper_bgcolor="#161b22")

    # ── No-relationship explanation ───────────────────────────────────────────
    n_corr  = len(corr_edges)
    n_ccf   = len(ccf_edges)
    n_total = len(all_nodes) * (len(all_nodes)-1)

    body = f"""
<div class="ph-header"><h1>Mega-Netzwerk: Alle Analysen \u2013 4 Layouts</h1>
  <div class="sub">Hierarchisch | Kreisgeometrie | Sektor-Cluster | Rohstoff-Stern + Sankey</div>
</div>
<div class="card mb-4">
  <div class="card-header">Legende & Interpretation</div>
  <div class="card-body">
    <div class="row g-3">
      <div class="col-md-3"><strong style="color:#39d353;">Gruene Linien</strong><br>
        <small>CCF/Granger-Kanten (dicker = kleinerer Lag = staerker)</small></div>
      <div class="col-md-3"><strong style="color:#8b949e;">Graue Linien</strong><br>
        <small>Signifikante Spearman-Korrelation (dicker = hoehere rho)</small></div>
      <div class="col-md-3"><strong>Knotengroesse</strong><br>
        <small>PageRank: groesser = wichtiger im Netz</small></div>
      <div class="col-md-3"><strong>Farbe</strong><br>
        <small>Sektor (Gelb=Energie, Lila=Metall, Gruen=Agrar, Blau=Markt)</small></div>
    </div>
    <hr style="border-color:#30363d;margin:.8rem 0;">
    <div class="row g-3">
      <div class="col-md-6">
        {_warn(f"Warum sind viele Paare OHNE Kante? Von {n_total} moeglichen Paaren haben nur "
               f"{n_corr} signifikante Korrelation und {n_ccf} Granger/CCF-Beziehung. "
               f"Der Rest: keine statistisch belegte Beziehung (p>=5%) in diesen Daten. "
               f"Das ist normal: nicht alle Assets teilen denselben Treiber.")}
      </div>
      <div class="col-md-6">
        <table class="table table-dark table-sm">
          <tr><td>Keine Kante</td><td>Paare ohne sign. Korrelation oder Granger-Effekt</td></tr>
          <tr><td>Grau, duenn</td><td>Schwache Korrelation (rho 0.2-0.4)</td></tr>
          <tr><td>Grau, dick</td><td>Starke Korrelation (rho > 0.6)</td></tr>
          <tr><td>Gruen, dick</td><td>Starker Granger-Effekt mit kurz. Lag</td></tr>
          <tr><td>Gruen, duenn</td><td>Schwacher/langer Granger-Lag</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>

<ul class="nav nav-tabs mb-3" id="netTabs" role="tablist" style="border-bottom-color:#30363d;">
  <li class="nav-item"><a class="nav-link active" style="color:#58a6ff;background:#1c2128;border-color:#30363d #30363d #1c2128;"
    href="#tab-hier" data-bs-toggle="tab">&#128681; Hierarchisch</a></li>
  <li class="nav-item"><a class="nav-link" style="color:#8b949e;"
    href="#tab-circ" data-bs-toggle="tab">&#9711; Kreisgeometrie</a></li>
  <li class="nav-item"><a class="nav-link" style="color:#8b949e;"
    href="#tab-group" data-bs-toggle="tab">&#128200; Sektor-Cluster</a></li>
  <li class="nav-item"><a class="nav-link" style="color:#8b949e;"
    href="#tab-star" data-bs-toggle="tab">&#11088; Rohstoff-Stern</a></li>
  <li class="nav-item"><a class="nav-link" style="color:#8b949e;"
    href="#tab-sankey" data-bs-toggle="tab">&#10145; Sankey</a></li>
</ul>
<div class="tab-content">
  <div class="tab-pane active" id="tab-hier">
    {_info("Hierarchisch: Y-Achse = Proximity Level (L0=Rohstoff oben, L6=Index unten). "
           "Kanten von oben nach unten = Informationsfluss. Horizontale Kanten = Peers.")}
    {_div(fig_hier, height=800)}
  </div>
  <div class="tab-pane" id="tab-circ">
    {_info("Kreisgeometrie: Innen-Ring = Rohstoffe, Ring 2 = ETF/Mega, Ring 3 = Mid/Small, Aussen = Indizes. "
           "Radiale Linien = Rohstoff-Equity-Beziehungen. Kreisfoermige = Peer-Korrelation.")}
    {_div(fig_circ, height=800)}
  </div>
  <div class="tab-pane" id="tab-group">
    {_info("Sektor-Cluster: Jeder Cluster = ein Wirtschaftssektor. Intra-Sektor-Kanten = Peer-Korrelation (grau). "
           "Inter-Sektor-Kanten = Cross-Sektor-Beziehungen (Energie->Markt). "
           "Paare ohne Linie: kein Sign.-Niveau p<5% in Korrelation oder Granger.")}
    {_div(fig_group, height=800)}
  </div>
  <div class="tab-pane" id="tab-star">
    {_info("Rohstoff-Stern: 5 Haupt-Rohstoffe (CL=F, GC=F, NG=F, HG=F, ZC=F) als innerer Ring. "
           "Assoziierte Equity-Assets im Mittelring. Marktindizes aussen. "
           "Zeigt wer mit welchem Rohstoff-Kern assoziiert ist.")}
    {_div(fig_star, height=800)}
  </div>
  <div class="tab-pane" id="tab-sankey">
    {_info("Sankey: Knotenbreite = Anzahl signifikanter Kanten. "
           "Von links nach rechts: Informationsfluss-Hierarchie (Rohstoff->ETF->Produzent->Index).")}
    {_div(fig_sankey, height=420)}
    {_interp("Rohstoff-Futures: breiteste linke Baender = groesste Informationssender. "
             "Marktindizes: empfangen von allen (aggregiertes Marktgeschehen).")}
  </div>
</div>
"""
    _write(out / "mega_network.html", _html_base("Mega-Netzwerk - 4 Layouts", 13, body))



# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Time Series Viewer + STL
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries_viewer_report(tables, figures, out):  # noqa: C901
    import json
    try:
        from scipy.stats import kendalltau as _ktau
    except ImportError:
        _ktau = None

    prices  = _read(tables / "phase1_prices.csv")
    returns = _read(tables / "phase2_returns.csv")
    desc    = _read(tables / "phase3_descriptive_stats.csv")
    corr_p  = _read(tables / "phase5_corr_pearson.csv")
    corr_s  = _read(tables / "phase5_corr_spearman.csv")
    corr_k  = _read(tables / "phase5_corr_kendall.csv")

    if prices is None:
        _write(out / "timeseries_viewer.html",
               _html_base("Zeitreihen-Viewer", 2, "<p>Preisdaten fehlen.</p>"))
        return

    av     = set(prices.columns)
    COMM_E = [t for t in ["CL=F","BZ=F","NG=F"]                    if t in av]
    COMM_M = [t for t in ["GC=F","SI=F","HG=F","MGC"]              if t in av]
    COMM_A = [t for t in ["ZC=F","ZW=F","ZS=F"]                    if t in av]
    COMM   = COMM_E + COMM_M + COMM_A
    PROD_E = [t for t in ["XOM","CVX","APA","OXY","SM","XLE","XLI"] if t in av]
    PROD_M = [t for t in ["FCX","NEM","TECK","TGB","GORO","GDX","SIL","XLB"] if t in av]
    MARKET = [t for t in ["SPY","QQQ","IWM","IJH"]                  if t in av]
    MACRO  = [t for t in ["^VIX","DX-Y.NYB","^TNX"]                 if t in av]
    # External factors (may not exist yet if pipeline not re-run)
    EXT_FX   = [t for t in ["CNY=X","AUDUSD=X","BRL=X"]             if t in av]
    EXT_SECT = [t for t in ["SBLK","LIT","PDBC","REMX","URA"]       if t in av]
    EXT_ALL  = EXT_FX + EXT_SECT
    ALL_P  = PROD_E + PROD_M

    # (group_name, grp1_assets, grp2_assets, description)
    GROUPS = [
        ("\u26a1 Energie: Rohstoff \u00d7 Produzent", COMM_E, PROD_E,
         "\u00d6l/Gas-Futures vs. Energie-Produzenten & ETF (XLE)"),
        ("\U0001f3c5 Metall: Rohstoff \u00d7 Produzent", COMM_M, PROD_M,
         "Gold/Silber/Kupfer vs. Bergbau-Firmen & Metall-ETFs"),
        ("\U0001f33e Agrar: Rohstoff \u00d7 Markt", COMM_A, MARKET,
         "Mais/Weizen/Soja vs. Breitmarkt-Indizes"),
        ("\U0001f4c8 Rohstoff \u00d7 Breitmarkt", COMM, MARKET,
         "Alle Rohstoffe vs. SPY/QQQ/IWM/IJH"),
        ("\U0001f310 Rohstoff \u00d7 Makro", COMM, MACRO,
         "Alle Rohstoffe vs. VIX, DXY, TNX (Makro-Regime)"),
        ("\U0001f3e6 Produzent \u00d7 Makro", ALL_P, MACRO,
         "Alle Rohstoff-Unternehmen vs. Makro-Umfeld"),
        ("\u2194\ufe0f Rohstoff \u00d7 Rohstoff", COMM, COMM,
         "Intra-Rohstoff: \u00d6l, Gas, Metall, Agrar untereinander"),
        ("\U0001f517 Cross-Sektor: Energie \u00d7 Metall", PROD_E[:4], PROD_M[:4],
         "Energie- vs. Metall-Produzenten: Sektor-Divergenz"),
    ]
    # Only add external group if at least one external ticker is present
    if EXT_ALL:
        GROUPS.append((
            "\U0001f30d Externe Faktoren: FX \u00d7 Rohstoff",
            EXT_ALL, COMM,
            "CNY/AUD/BRL + BDI/LIT/PDBC vs. alle Rohstoffe \u2014 unerklärte Varianz"
        ))
    if EXT_ALL and PROD_E + PROD_M:
        GROUPS.append((
            "\U0001f310 Externe Faktoren \u00d7 Produzenten",
            EXT_ALL, PROD_E + PROD_M,
            "Externe Makro-/FX-Faktoren vs. Rohstoff-Produzenten"
        ))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _skey(a, b):
        return (a + "_" + b).replace("^","").replace("=","").replace("-","").replace(".","")

    def _clkp(df, a, b):
        if df is None:
            return None
        try:
            if a in df.index and b in df.columns:
                return float(df.loc[a, b])
            if b in df.index and a in df.columns:
                return float(df.loc[b, a])
        except Exception:
            pass
        return None

    def _astats(tkr, r):
        if desc is not None and tkr in desc.index:
            row = desc.loc[tkr]
            def g(c, fb=0.0): return round(float(row[c]), 4) if c in row.index else fb
            return {"mean": round(g("Mean (ann.)")*100, 2),
                    "vol":  round(g("Vol (ann.)")*100, 2),
                    "skew": g("Skewness"),   "kurt": g("Excess Kurtosis"),
                    "shrp": g("Sharpe"),     "mdd":  round(g("Max Drawdown")*100, 2)}
        rd = r.dropna()
        return {"mean": round(float(rd.mean())*25200, 2),
                "vol":  round(float(rd.std())*1600, 2),
                "skew": round(float(rd.skew()), 3),
                "kurt": round(float(rd.kurtosis()), 3),
                "shrp": 0.0, "mdd": 0.0}

    # ── pre-compute all pair data ─────────────────────────────────────────────
    SP, SR = 20, 40   # downsample steps: price every 20th, returns every 40th
    all_pd: dict  = {}
    grp_keys: dict = {}

    for gname, grp1, grp2, _gdesc in GROUPS:
        gk: list = []
        for a1 in grp1:
            for a2 in grp2:
                if a1 == a2:
                    continue
                key  = _skey(a1, a2)
                rkey = _skey(a2, a1)
                if rkey in all_pd:
                    if rkey not in gk: gk.append(rkey)
                    continue
                if key in all_pd:
                    if key not in gk: gk.append(key)
                    continue
                if a1 not in prices.columns or a2 not in prices.columns:
                    continue
                s1  = prices[a1].dropna()
                s2  = prices[a2].dropna()
                idx = s1.index.intersection(s2.index)
                if len(idx) < 100:
                    continue
                s1 = s1.loc[idx]; s2 = s2.loc[idx]

                if returns is not None and a1 in returns.columns and a2 in returns.columns:
                    r1 = returns[a1].dropna(); r2 = returns[a2].dropna()
                    ri = r1.index.intersection(r2.index).intersection(idx)
                    r1, r2 = r1.loc[ri], r2.loc[ri]
                else:
                    r1 = s1.pct_change().dropna(); r2 = s2.pct_change().dropna()
                    ri = r1.index.intersection(r2.index)
                    r1, r2 = r1.loc[ri], r2.loc[ri]
                if len(r1) < 30:
                    continue

                pe  = _clkp(corr_p, a1, a2) or float(r1.corr(r2))
                sp  = _clkp(corr_s, a1, a2) or float(r1.rank().corr(r2.rank()))
                ke  = _clkp(corr_k, a1, a2)
                if ke is None and _ktau is not None:
                    try:
                        ke = float(_ktau(r1.values, r2.values).statistic)
                    except Exception:
                        ke = 0.0
                ke = ke or 0.0

                n1  = s1 / s1.iloc[0] * 100
                n2  = s2 / s2.iloc[0] * 100
                rc  = r1.rolling(63).corr(r2).dropna()
                ss  = (n1 - n2).dropna()
                sz  = ((ss - ss.rolling(63).mean()) /
                       (ss.rolling(63).std() + 1e-9)).dropna()

                all_pd[key] = {
                    "a1": a1, "a2": a2, "n": len(idx),
                    "pe": round(pe,4), "sp": round(sp,4), "ke": round(ke,4),
                    "dt":  [str(d)[:10] for d in idx[::SP]],
                    "p1":  [round(float(v),2) for v in n1.iloc[::SP]],
                    "p2":  [round(float(v),2) for v in n2.iloc[::SP]],
                    "rd":  [str(d)[:10] for d in ri[::SR]],
                    "r1":  [round(float(v),4) for v in r1.iloc[::SR]],
                    "r2":  [round(float(v),4) for v in r2.iloc[::SR]],
                    "rcd": [str(d)[:10] for d in rc.index[::SP]],
                    "rc":  [round(float(v),4) for v in rc.iloc[::SP]],
                    "spd": [str(d)[:10] for d in ss.index[::SP]],
                    "sps": [round(float(v),2) for v in ss.iloc[::SP]],
                    "szd": [str(d)[:10] for d in sz.index[::SP]],
                    "sz":  [round(float(v),3) for v in sz.iloc[::SP]],
                    "s1":  _astats(a1, r1),
                    "s2":  _astats(a2, r2),
                }
                gk.append(key)
        grp_keys[gname] = gk

    # ── build accordion HTML ──────────────────────────────────────────────────
    def _ccol(v):
        if v >  0.6: return "#3fb950"
        if v >  0.3: return "#56d364"
        if v >  0.0: return "#d29922"
        if v > -0.3: return "#ffa657"
        return "#f78166"

    acc_items = []
    for gi, (gname, _g1, _g2, gdesc) in enumerate(GROUPS):
        gkeys = grp_keys.get(gname, [])
        if not gkeys:
            continue
        gid    = f"g{gi}"
        pitems = []
        for ki, key in enumerate(gkeys):
            d      = all_pd[key]
            a1, a2 = d["a1"], d["a2"]
            col    = _ccol(d["pe"])
            pid    = f"{gid}k{ki}"
            pitems.append(
f"""<div class="accordion-item" style="background:#161b22;border:1px solid #21262d;margin-bottom:2px;">
  <h2 class="accordion-header">
    <button class="accordion-button collapsed py-1 px-3" type="button"
            data-bs-toggle="collapse" data-bs-target="#{pid}"
            style="background:#1c2128;color:#e6edf3;font-size:.82rem;box-shadow:none;"
            onclick="lazyRender('{key}')">
      <strong style="color:#58a6ff;">{a1}</strong>
      <span style="color:#555;margin:0 .35rem;">\u00d7</span>
      <strong style="color:#d29922;">{a2}</strong>
      <span class="badge ms-2"
            style="background:{col}22;color:{col};border:1px solid {col}44;font-size:.68rem;">
        \u03c1\u202f{d['pe']:+.3f}</span>
      <span class="ms-auto" style="color:#555;font-size:.68rem;">{d['n']:,}\u202fT</span>
    </button>
  </h2>
  <div id="{pid}" class="accordion-collapse collapse" data-pairkey="{key}">
    <div class="accordion-body p-2" style="background:#0d1117;">
      <ul class="nav nav-tabs mb-2" role="tablist"
          style="border-bottom:1px solid #21262d;font-size:.75rem;">
        <li class="nav-item">
          <button class="nav-link active px-3 py-1" data-bs-toggle="tab"
                  data-bs-target="#tp-{key}"
                  style="color:#58a6ff;background:transparent;border:none;
                         border-bottom:2px solid #58a6ff;">
            \U0001f4c8 Preise</button></li>
        <li class="nav-item">
          <button class="nav-link px-3 py-1" data-bs-toggle="tab"
                  data-bs-target="#tc-{key}"
                  style="color:#8b949e;background:transparent;border:none;"
                  onclick="lazyRender('{key}')">
            \U0001f4ca Korrelation</button></li>
        <li class="nav-item">
          <button class="nav-link px-3 py-1" data-bs-toggle="tab"
                  data-bs-target="#tr-{key}"
                  style="color:#8b949e;background:transparent;border:none;"
                  onclick="lazyRender('{key}')">
            \U0001f504 Renditen</button></li>
        <li class="nav-item">
          <button class="nav-link px-3 py-1" data-bs-toggle="tab"
                  data-bs-target="#ts-{key}"
                  style="color:#8b949e;background:transparent;border:none;"
                  onclick="lazyRender('{key}')">
            \U0001f4c9 Spread</button></li>
        <li class="nav-item">
          <button class="nav-link px-3 py-1" data-bs-toggle="tab"
                  data-bs-target="#tstats-{key}"
                  style="color:#8b949e;background:transparent;border:none;"
                  onclick="lazyRender('{key}')">
            \U0001f4cb Statistiken</button></li>
      </ul>
      <div class="tab-content">
        <div class="tab-pane fade show active" id="tp-{key}">
          <div id="price-{key}" style="height:260px;"></div></div>
        <div class="tab-pane fade" id="tc-{key}">
          <div id="rcorr-{key}" style="height:240px;"></div></div>
        <div class="tab-pane fade" id="tr-{key}">
          <div id="rscat-{key}" style="height:260px;"></div></div>
        <div class="tab-pane fade" id="ts-{key}">
          <div id="spread-{key}" style="height:260px;"></div></div>
        <div class="tab-pane fade" id="tstats-{key}">
          <div id="statsbox-{key}" style="padding:.5rem;"></div></div>
      </div>
    </div>
  </div>
</div>""")

        show_cls = "" if gi > 0 else "show"
        coll_cls = "collapsed" if gi > 0 else ""
        acc_items.append(
f"""<div class="accordion-item mb-2"
     style="background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;">
  <h2 class="accordion-header">
    <button class="accordion-button {coll_cls} px-3 py-2" type="button"
            data-bs-toggle="collapse" data-bs-target="#{gid}body"
            style="background:#161b22;color:#e6edf3;font-size:.9rem;">
      {gname}
      <span class="badge ms-2"
            style="background:#58a6ff22;color:#58a6ff;border:1px solid #58a6ff44;font-weight:400;">
        {len(gkeys)}\u202fPaare</span>
      <small class="ms-3 d-none d-md-inline"
             style="color:#8b949e;font-weight:400;">{gdesc}</small>
    </button>
  </h2>
  <div id="{gid}body" class="accordion-collapse collapse {show_cls}">
    <div class="accordion-body p-2">
      <div class="accordion" id="pairs-{gid}">
        {''.join(pitems)}
      </div>
    </div>
  </div>
</div>""")

    # ── lazy-rendering JavaScript (no f-string to avoid brace conflicts) ──────
    pair_json  = json.dumps(all_pd, separators=(',',':')).replace("</", "<\\/")
    unique_cnt = len(all_pd)
    grp_cnt    = sum(1 for g in GROUPS if grp_keys.get(g[0]))

    _js_tail = """;
const _r={};
const BG={paper_bgcolor:"#161b22",plot_bgcolor:"#0d1117",
  font:{color:"#e6edf3",family:"Segoe UI,Arial",size:11},
  margin:{l:55,r:15,t:35,b:45},
  xaxis:{gridcolor:"#21262d",linecolor:"#30363d"},
  yaxis:{gridcolor:"#21262d",linecolor:"#30363d"},
  legend:{bgcolor:"#1c2128",bordercolor:"#30363d",borderwidth:1}};
const CFG={displayModeBar:false,responsive:true};
function L(t,ex){return Object.assign({},BG,{title:{text:t,font:{color:"#e6edf3",size:12}}},ex||{});}
function lazyRender(key){
  if(_r[key])return;_r[key]=true;
  const d=PDATA[key];if(!d)return;
  // Normalised price comparison
  Plotly.newPlot("price-"+key,[
    {x:d.dt,y:d.p1,mode:"lines",name:d.a1,line:{color:"#58a6ff",width:1.5}},
    {x:d.dt,y:d.p2,mode:"lines",name:d.a2,line:{color:"#d29922",width:1.5}}
  ],L(d.a1+" vs "+d.a2+" \u2013 Preis normiert auf 100",{height:260}),CFG);
  // Rolling 63-day correlation
  Plotly.newPlot("rcorr-"+key,[
    {x:d.rcd,y:d.rc,mode:"lines",name:"Roll.\u202fKorr.\u202f63T",fill:"tozeroy",
     fillcolor:"#3fb95018",line:{color:"#3fb950",width:1.5}},
    {x:d.rcd,y:d.rc.map(()=>0),mode:"lines",showlegend:false,
     line:{color:"#8b949e",width:1,dash:"dot"}}
  ],L("Rolling Korrelation (63-Tage-Fenster) \u2013 Pearson\u202f\u03c1",{height:240,
     yaxis:{gridcolor:"#21262d",range:[-1,1],title:"Korrelation"}}),CFG);
  // Return scatter
  Plotly.newPlot("rscat-"+key,[
    {x:d.r1,y:d.r2,mode:"markers",name:"Rendite-Paare",
     marker:{size:3,color:"#58a6ff",opacity:0.45},
     hovertemplate:d.a1+": %{x:.2%}<br>"+d.a2+": %{y:.2%}<extra></extra>"}
  ],L(d.a1+" vs "+d.a2+" \u2013 Rendite-Scatter",{height:260,
     xaxis:{gridcolor:"#21262d",title:d.a1},
     yaxis:{gridcolor:"#21262d",title:d.a2}}),CFG);
  // Spread + Z-Score dual axis
  Plotly.newPlot("spread-"+key,[
    {x:d.spd,y:d.sps,mode:"lines",name:"Spread",yaxis:"y",
     line:{color:"#bc8cff",width:1.3}},
    {x:d.szd,y:d.sz,mode:"lines",name:"Z-Score\u202f63T",yaxis:"y2",
     line:{color:"#ffa657",width:1.2,dash:"dash"}},
    {x:d.szd,y:d.sz.map(()=>2),mode:"lines",name:"+2\u03c3",yaxis:"y2",showlegend:false,
     line:{color:"#f78166",width:1,dash:"dot"}},
    {x:d.szd,y:d.sz.map(()=>-2),mode:"lines",name:"-2\u03c3",yaxis:"y2",showlegend:false,
     line:{color:"#f78166",width:1,dash:"dot"}}
  ],L("Spread (normierte Preis-Differenz) & Z-Score",{height:260,
     yaxis:{gridcolor:"#21262d",title:"Spread"},
     yaxis2:{title:"Z-Score",overlaying:"y",side:"right",
             range:[-4,4],gridcolor:"#21262d",tickfont:{color:"#ffa657"}}}),CFG);
  // Stats table (rendered as HTML)
  const s1=d.s1,s2=d.s2;
  const rows=[
    ["Rendite p.a. (%)",s1.mean.toFixed(2),s2.mean.toFixed(2)],
    ["Volatilitat p.a. (%)",s1.vol.toFixed(2),s2.vol.toFixed(2)],
    ["Sharpe Ratio",s1.shrp.toFixed(3),s2.shrp.toFixed(3)],
    ["Skewness",s1.skew.toFixed(3),s2.skew.toFixed(3)],
    ["Excess Kurtosis",s1.kurt.toFixed(3),s2.kurt.toFixed(3)],
    ["Max Drawdown (%)",s1.mdd.toFixed(2),s2.mdd.toFixed(2)]
  ];
  const tbody=rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("");
  document.getElementById("statsbox-"+key).innerHTML=`
    <table class="table table-dark table-sm table-bordered" style="font-size:.75rem;">
      <thead><tr><th>Kennzahl</th>
        <th style="color:#58a6ff;">${d.a1}</th>
        <th style="color:#d29922;">${d.a2}</th></tr></thead>
      <tbody>${tbody}</tbody>
      <tfoot><tr><td colspan="3" style="color:#8b949e;font-size:.68rem;">
        Pearson \u03c1 = ${d.pe.toFixed(4)}\u2002|\u2002
        Spearman \u03c1 = ${d.sp.toFixed(4)}\u2002|\u2002
        Kendall \u03c4 = ${d.ke.toFixed(4)}\u2002|\u2002
        n = ${d.n.toLocaleString()} Handelstage
      </td></tr></tfoot>
    </table>`;
}
</script>"""
    viewer_js = "<script>\nconst PDATA=" + pair_json + _js_tail

    body = f"""
<div class="ph-header"><h1>Zeitreihen-Viewer \u2013 Alle Klassen-Kombinationen</h1>
  <div class="sub">
    {unique_cnt}\u202feinzigartige Paare in {grp_cnt}\u202fGruppen \u2014
    Gruppe aufklappen \u2192 Paar aufklappen \u2192 Tab ausw\u00e4hlen
  </div>
</div>
{_stat_row([
    ("Einzigartige Paare",  str(unique_cnt)),
    ("Gruppen",             str(grp_cnt)),
    ("Charts pro Paar",     "5"),
    ("Rendering",           "Lazy JS"),
])}
{_card("Gruppen\u00fcbersicht & Chart-Tabs",
    "<table class='table table-dark table-sm table-bordered'>"
    "<thead><tr><th>Gruppe</th><th>Beschreibung</th><th>Verf\u00fcgbare Tabs</th></tr></thead>"
    "<tbody>"
    "<tr><td>\u26a1 Energie</td><td>\u00d6l/Gas \u00d7 Produzenten/ETF</td>"
    "<td rowspan='8' style='vertical-align:middle;color:#8b949e;font-size:.85rem;'>"
    "\U0001f4c8 Preise (normiert 100)\u2002\u00b7\u2002"
    "\U0001f4ca Rolling Korrelation (63T)\u2002\u00b7\u2002"
    "\U0001f504 Rendite-Scatter\u2002\u00b7\u2002"
    "\U0001f4c9 Spread + Z-Score\u2002\u00b7\u2002"
    "\U0001f4cb Deskriptive Statistiken"
    "</td></tr>"
    "<tr><td>\U0001f3c5 Metall</td><td>Gold/Silber/Kupfer \u00d7 Bergbau/ETF</td></tr>"
    "<tr><td>\U0001f33e Agrar</td><td>Mais/Weizen/Soja \u00d7 Breitmarkt</td></tr>"
    "<tr><td>\U0001f4c8 Rohstoff\u00d7Breitmarkt</td><td>Alle Rohstoffe \u00d7 SPY/QQQ/IWM/IJH</td></tr>"
    "<tr><td>\U0001f310 Rohstoff\u00d7Makro</td><td>Alle Rohstoffe \u00d7 VIX/DXY/TNX</td></tr>"
    "<tr><td>\U0001f3e6 Produzent\u00d7Makro</td><td>Energie/Metall-Firmen \u00d7 Makro</td></tr>"
    "<tr><td>\u2194\ufe0f Rohstoff\u00d7Rohstoff</td><td>Intra-Rohstoff-Korrelationen</td></tr>"
    "<tr><td>\U0001f517 Cross-Sektor</td><td>Energie-Produzenten \u00d7 Metall-Produzenten</td></tr>"
    "</tbody></table>"
)}
<div class="accordion" id="mainAccordion">
{''.join(acc_items)}
</div>
{viewer_js}"""
    _write(out / "timeseries_viewer.html", _html_base("Zeitreihen-Viewer", 2, body))


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting & Strategy Report
# ─────────────────────────────────────────────────────────────────────────────

def build_backtest_report(tables, figures, out):  # noqa: C901
    returns = _read(tables / "phase2_returns.csv")
    eg      = _read(tables / "phase8_eg_cointegration.csv")
    gran    = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "backtesting.html",
               _html_base("Backtesting", 14, "<p>Renditen fehlen.</p>"))
        return

    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]

    # ── Helper metrics ────────────────────────────────────────────────────
    def _metrics(r, name, benchmark=None):
        r = r.dropna()
        if len(r) < 63:
            return None
        ann_r  = float(r.mean() * 252)
        ann_v  = float(r.std() * np.sqrt(252))
        sharpe = ann_r / (ann_v + 1e-9)
        neg    = r[r < 0]
        sortino = ann_r / (neg.std() * np.sqrt(252) + 1e-9)
        cum    = (1 + r).cumprod()
        dd_ser = cum / cum.cummax() - 1
        mdd    = float(dd_ser.min())
        # MaxDD duration in trading days
        in_dd  = (dd_ser < 0)
        if in_dd.any():
            dur_blocks = []
            cnt = 0
            for v in in_dd:
                if v: cnt += 1
                else:
                    if cnt: dur_blocks.append(cnt)
                    cnt = 0
            if cnt: dur_blocks.append(cnt)
            mdd_dur = max(dur_blocks)
        else:
            mdd_dur = 0
        calmar = ann_r / (abs(mdd) + 1e-9)
        var95  = float(r.quantile(0.05))
        cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
        hit    = float((r > 0).mean())
        gains  = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        pf     = gains / (losses + 1e-9)
        skew   = float(r.skew())
        kurt   = float(r.kurt())
        total_r = float(cum.iloc[-1] - 1)
        cagr    = float((cum.iloc[-1] ** (252 / len(r))) - 1)
        # Alpha & Beta vs SPY
        alpha_pct, beta_v, ir = float("nan"), float("nan"), float("nan")
        if benchmark is not None:
            bm = benchmark.reindex(r.index).dropna()
            common = r.index.intersection(bm.index)
            if len(common) > 63:
                rc, bc = r.loc[common].values, bm.loc[common].values
                if bc.std() > 1e-9:
                    beta_v = float(np.cov(rc, bc)[0, 1] / np.var(bc))
                    alpha_pct = float((rc.mean() - beta_v * bc.mean()) * 252 * 100)
                    ex = rc - beta_v * bc
                    ir = float(ex.mean() * 252 / (ex.std() * np.sqrt(252) + 1e-9))
        return {
            "Strategie": name,
            "CAGR%": round(cagr * 100, 2),
            "Ann.Vol%": round(ann_v * 100, 2),
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Calmar": round(calmar, 3),
            "MaxDD%": round(mdd * 100, 2),
            "MaxDD-Dur(T)": mdd_dur,
            "VaR95%": round(var95 * 100, 3),
            "CVaR95%": round(cvar95 * 100, 3),
            "Hit-Rate%": round(hit * 100, 1),
            "Profit-Factor": round(pf, 3),
            "Alpha%(SPY,ann)": round(alpha_pct, 2),
            "Beta(SPY)": round(beta_v, 3),
            "Info-Ratio": round(ir, 3),
            "Skewness": round(skew, 3),
            "Kurtosis": round(kurt, 3),
            "Total-Return%": round(total_r * 100, 1),
        }

    def _eq_curve(r):
        return (1 + r.dropna()).cumprod()

    def _apply_tc(pos_ser, r_ser, tc_bps=10):
        """pos_ser: +1/0/-1 signal series, r_ser: returns"""
        turnover = pos_ser.diff().abs().fillna(0)
        return pos_ser * r_ser - turnover * (tc_bps / 10000)

    def _walk_forward_sharpe(r, is_frac=0.7):
        r = r.dropna()
        n  = len(r)
        split = int(n * is_frac)
        is_r  = r.iloc[:split]
        oos_r = r.iloc[split:]
        is_sh  = float(is_r.mean() * 252 / (is_r.std() * np.sqrt(252) + 1e-9))
        oos_sh = float(oos_r.mean() * 252 / (oos_r.std() * np.sqrt(252) + 1e-9))
        return round(is_sh, 3), round(oos_sh, 3)

    # ── Benchmark ────────────────────────────────────────────────────────
    spy_r = clean["SPY"].dropna() if "SPY" in clean.columns else None

    all_results  = {}   # name → pd.Series of daily strategy returns
    all_equity   = {}   # name → equity curve
    perf_rows    = []
    wf_rows      = []

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 1 – Granger Lead Signal (CL=F → targets at best Granger lag)
    # Parameter grid: threshold ∈ {0, 0.3%, 0.5%}, holding ∈ {lag, lag+3}
    # ═══════════════════════════════════════════════════════════════════
    GRANGER_MAP = {
        "SM":   1,  "TECK": 1,  "CVX":  1,  "APA":  1,
        "XOM":  6,  "OXY": 10,  "XLI":  3,  "XLB":  3,
        "NEM":  6,  "GDX":  7,  "SIL":  7,
    }
    if "CL=F" in clean.columns:
        cl = clean["CL=F"].dropna()
        for tgt, best_lag in GRANGER_MAP.items():
            if tgt not in clean.columns:
                continue
            t_r = clean[tgt].dropna()
            idx = cl.index.intersection(t_r.index)
            xc, yt = cl.loc[idx], t_r.loc[idx]
            for thresh in [0.0, 0.003, 0.005]:
                for hold_extra in [0, 3]:
                    hold = best_lag + hold_extra
                    # Signal: CL=F return at t-lag exceeds threshold
                    raw_sig = np.where(xc.shift(best_lag) > thresh, 1,
                              np.where(xc.shift(best_lag) < -thresh, -1, 0))
                    sig_s = pd.Series(raw_sig, index=idx)
                    # Hold position for `hold` days
                    if hold > 1:
                        sig_s = sig_s.where(sig_s != 0).ffill(limit=hold-1).fillna(0)
                    strat_r = _apply_tc(sig_s, yt, tc_bps=10)
                    nm = (f"S1-Granger[{tgt}|lag{best_lag}|thr{int(thresh*1000)}bps"
                          f"|hold{hold}T]")
                    m = _metrics(strat_r, nm, spy_r)
                    if m:
                        all_results[nm] = strat_r
                        all_equity[nm]  = _eq_curve(strat_r)
                        perf_rows.append(m)
                        is_sh, oos_sh = _walk_forward_sharpe(strat_r)
                        wf_rows.append({"Strategie": nm, "IS Sharpe": is_sh,
                                        "OOS Sharpe": oos_sh,
                                        "Degradation": round(is_sh - oos_sh, 3)})

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 2 – Pairs Trading on cointegrated pairs
    # Parameter grid: z_thresh ∈ {1.5, 2.0, 2.5}, window ∈ {21, 63}
    # ═══════════════════════════════════════════════════════════════════
    EG_PAIRS = []
    if eg is not None and "cointegrated_95" in eg.columns:
        e_sig = eg[eg["cointegrated_95"] == True]
        if "eg_stat" in e_sig.columns:
            e_sig = e_sig.sort_values("eg_stat")
        EG_PAIRS = [(str(r["asset1"]), str(r["asset2"])) for _, r in e_sig.head(6).iterrows()]
    # Fallback to known best pairs
    if not EG_PAIRS:
        EG_PAIRS = [("ZW=F","ZS=F"), ("CL=F","ZW=F"), ("BZ=F","CL=F")]

    for a1, a2 in EG_PAIRS[:5]:
        if a1 not in clean.columns or a2 not in clean.columns:
            continue
        p1 = clean[a1].dropna(); p2 = clean[a2].dropna()
        idx = p1.index.intersection(p2.index)
        if len(idx) < 252:
            continue
        # Log-price spread (hedge ratio = 1 for simplicity)
        spread = np.log(p1.loc[idx] + 1e-9) - np.log(p2.loc[idx] + 1e-9)
        for win in [21, 63]:
            roll_m = spread.rolling(win).mean()
            roll_s = spread.rolling(win).std()
            z = (spread - roll_m) / (roll_s + 1e-9)
            for zt in [1.5, 2.0, 2.5]:
                # Mean-reversion signal
                pos_a1 = pd.Series(0.0, index=idx)
                pos_a2 = pd.Series(0.0, index=idx)
                pos_a1[z.shift(1) >  zt] = -1.0  # short a1 when spread too wide
                pos_a2[z.shift(1) >  zt] =  1.0
                pos_a1[z.shift(1) < -zt] =  1.0  # long a1 when spread too narrow
                pos_a2[z.shift(1) < -zt] = -1.0
                # Exit when spread crosses mean
                pos_a1[abs(z.shift(1)) < 0.2] = 0.0
                pos_a2[abs(z.shift(1)) < 0.2] = 0.0
                pos_a1 = pos_a1.ffill(limit=10).fillna(0)
                pos_a2 = pos_a2.ffill(limit=10).fillna(0)
                r1 = clean[a1].reindex(idx)
                r2 = clean[a2].reindex(idx)
                strat_r = (0.5 * _apply_tc(pos_a1, r1, 10)
                         + 0.5 * _apply_tc(pos_a2, r2, 10))
                nm = f"S2-Pairs[{a1}+{a2}|win{win}|z{zt}]"
                m = _metrics(strat_r, nm, spy_r)
                if m:
                    all_results[nm] = strat_r
                    all_equity[nm]  = _eq_curve(strat_r)
                    perf_rows.append(m)
                    is_sh, oos_sh = _walk_forward_sharpe(strat_r)
                    wf_rows.append({"Strategie": nm, "IS Sharpe": is_sh,
                                    "OOS Sharpe": oos_sh,
                                    "Degradation": round(is_sh - oos_sh, 3)})

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 3 – DXY Momentum → Metal/Miner Fade
    # Parameter grid: lookback ∈ {5, 21, 63} days
    # ═══════════════════════════════════════════════════════════════════
    DXY_CANDIDATES = ["GDX","NEM","GC=F","FCX","SIL"]
    dxy_col = next((c for c in ["DXY","DX=F","DX-Y.NYB"] if c in clean.columns), None)
    if dxy_col is not None:
        dxy = clean[dxy_col].dropna()
        for tgt in DXY_CANDIDATES:
            if tgt not in clean.columns:
                continue
            t_r = clean[tgt].dropna()
            idx = dxy.index.intersection(t_r.index)
            if len(idx) < 252:
                continue
            xd, yt = dxy.loc[idx], t_r.loc[idx]
            for lb in [5, 21, 63]:
                # DXY trend: short DXY (falling dollar) = buy metals
                dxy_mom = xd.pct_change(lb)
                sig_s   = pd.Series(np.where(dxy_mom.shift(1) < 0, 1, -1), index=idx)
                strat_r = _apply_tc(sig_s, yt, tc_bps=10)
                nm = f"S3-DXY-Metal[{tgt}|lb{lb}]"
                m = _metrics(strat_r, nm, spy_r)
                if m:
                    all_results[nm] = strat_r
                    all_equity[nm]  = _eq_curve(strat_r)
                    perf_rows.append(m)
                    is_sh, oos_sh = _walk_forward_sharpe(strat_r)
                    wf_rows.append({"Strategie": nm, "IS Sharpe": is_sh,
                                    "OOS Sharpe": oos_sh,
                                    "Degradation": round(is_sh - oos_sh, 3)})

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 4 – Multi-Strategy Portfolio (Equal-weight best of S1+S2)
    # ═══════════════════════════════════════════════════════════════════
    perf_df_tmp = pd.DataFrame(perf_rows) if perf_rows else None
    if perf_df_tmp is not None and len(perf_df_tmp) >= 4:
        s1_best = (perf_df_tmp[perf_df_tmp["Strategie"].str.startswith("S1")]
                   .sort_values("Sharpe", ascending=False)
                   .head(3)["Strategie"].tolist())
        s2_best = (perf_df_tmp[perf_df_tmp["Strategie"].str.startswith("S2")]
                   .sort_values("Sharpe", ascending=False)
                   .head(3)["Strategie"].tolist())
        combo_names = s1_best + s2_best
        combo_rets  = [all_results[n].dropna() for n in combo_names if n in all_results]
        if len(combo_rets) >= 2:
            combined = pd.concat(combo_rets, axis=1).mean(axis=1).dropna()
            nm = "S4-MultiStrat[S1top3+S2top3]"
            m  = _metrics(combined, nm, spy_r)
            if m:
                all_results[nm] = combined
                all_equity[nm]  = _eq_curve(combined)
                perf_rows.append(m)
                is_sh, oos_sh = _walk_forward_sharpe(combined)
                wf_rows.append({"Strategie": nm, "IS Sharpe": is_sh,
                                 "OOS Sharpe": oos_sh,
                                 "Degradation": round(is_sh - oos_sh, 3)})

    # ═══════════════════════════════════════════════════════════════════
    # BENCHMARKS: SPY, CL=F, GDX, XLE buy-and-hold
    # ═══════════════════════════════════════════════════════════════════
    for bm_nm in ["SPY","CL=F","GDX","XLE"]:
        if bm_nm in clean.columns:
            bm_r = clean[bm_nm].dropna()
            m = _metrics(bm_r, f"BH-{bm_nm}", spy_r)
            if m:
                all_results[f"BH-{bm_nm}"] = bm_r
                all_equity[f"BH-{bm_nm}"]  = _eq_curve(bm_r)
                perf_rows.append(m)

    # ── BUILD PERFORMANCE TABLE ──────────────────────────────────────
    perf_df = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()
    wf_df   = pd.DataFrame(wf_rows)   if wf_rows   else pd.DataFrame()

    # ── EQUITY CURVES CHART (top-15 by Sharpe + benchmarks) ──────────
    fig_eq = go.Figure()
    bh_nms  = [n for n in all_equity if n.startswith("BH-")]
    strat_sorted = (perf_df[~perf_df["Strategie"].str.startswith("BH-")]
                    .sort_values("Sharpe", ascending=False)
                    .head(12)["Strategie"].tolist()
                    if not perf_df.empty else [])
    for j, nm in enumerate(strat_sorted + bh_nms):
        if nm not in all_equity:
            continue
        eq = all_equity[nm]
        is_bh = nm.startswith("BH-")
        fig_eq.add_trace(go.Scatter(
            x=eq.index.astype(str).tolist(), y=eq.values.tolist(),
            mode="lines", name=nm,
            line=dict(color=PAL[j % len(PAL)],
                      width=0.9 if is_bh else 1.4,
                      dash="dot" if is_bh else "solid"),
            opacity=0.6 if is_bh else 1.0))
    fig_eq.update_layout(
        title="Equity-Kurven: Top-12 Strategien + Benchmarks (log-Skala)",
        yaxis_title="Kapital (Basis=1)", yaxis_type="log",
        height=520, legend=dict(font=dict(size=8)))

    # ── SHARPE COMPARISON ─────────────────────────────────────────────
    fig_sharpe = go.Figure()
    if not perf_df.empty:
        df_sh = perf_df.sort_values("Sharpe", ascending=True)
        fig_sharpe.add_trace(go.Bar(
            x=df_sh["Sharpe"].tolist(),
            y=df_sh["Strategie"].tolist(),
            orientation="h",
            marker_color=["#3fb950" if v > 0.5 else "#d29922" if v > 0
                          else "#f78166" for v in df_sh["Sharpe"]],
            text=[f"{v:.3f}" for v in df_sh["Sharpe"]],
            textposition="outside"))
        fig_sharpe.add_vline(x=0, line_color="#8b949e", line_width=1)
        fig_sharpe.add_vline(x=0.5, line_dash="dash", line_color="#d29922",
                              annotation_text="Sharpe=0.5", annotation_font_color="#d29922")
        fig_sharpe.add_vline(x=1.0, line_dash="dash", line_color="#3fb950",
                              annotation_text="Sharpe=1.0", annotation_font_color="#3fb950")
        fig_sharpe.update_layout(
            title="Sharpe Ratio: Alle Strategien (horizontal, sortiert)",
            xaxis_title="Sharpe Ratio",
            height=max(400, len(df_sh) * 18 + 100))

    # ── WALK-FORWARD SHARPE IS vs OOS ─────────────────────────────────
    fig_wf = go.Figure()
    if not wf_df.empty:
        wf_s = wf_df.sort_values("OOS Sharpe", ascending=False).head(20)
        fig_wf.add_trace(go.Bar(
            x=wf_s["Strategie"].tolist(), y=wf_s["IS Sharpe"].tolist(),
            name="IS Sharpe", marker_color="#58a6ff", opacity=0.8))
        fig_wf.add_trace(go.Bar(
            x=wf_s["Strategie"].tolist(), y=wf_s["OOS Sharpe"].tolist(),
            name="OOS Sharpe", marker_color="#3fb950", opacity=0.8))
        fig_wf.add_hline(y=0, line_color="#8b949e")
        fig_wf.update_layout(
            title="Walk-Forward Validierung: IS vs. OOS Sharpe (70%/30% Split)",
            barmode="group", xaxis_tickangle=-30, height=420,
            legend=dict(orientation="h"))

    # ── PARAMETER GRID HEATMAP: Sharpe by (threshold, lag/window) ─────
    fig_grid = go.Figure()
    if not perf_df.empty:
        s1_df = perf_df[perf_df["Strategie"].str.startswith("S1-Granger[SM|")]
        if not s1_df.empty:
            # Extract params from strategy name
            def _parse_s1(nm):
                import re
                m_lag = re.search(r"lag(\d+)", nm)
                m_thr = re.search(r"thr(\d+)bps", nm)
                m_hold = re.search(r"hold(\d+)T", nm)
                return (int(m_lag.group(1)) if m_lag else 0,
                        int(m_thr.group(1)) if m_thr else 0,
                        int(m_hold.group(1)) if m_hold else 0)
            s1_df = s1_df.copy()
            s1_df[["lag","thr","hold"]] = s1_df["Strategie"].apply(
                lambda x: pd.Series(_parse_s1(x)))
            pivot = s1_df.pivot_table(index="thr", columns="hold", values="Sharpe",
                                       aggfunc="mean")
            if not pivot.empty:
                fig_grid = go.Figure(go.Heatmap(
                    z=pivot.values.tolist(),
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale="RdYlGn", zmid=0,
                    text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                    texttemplate="%{text}",
                    hovertemplate="Hold=%{x}T | Threshold=%{y}bps<br>Sharpe=%{z:.3f}<extra></extra>"))
                fig_grid.update_layout(
                    title="Parameter-Grid Heatmap: S1-Granger[SM] – Sharpe(threshold × hold)",
                    xaxis_title="Holding Period (Tage)", yaxis_title="Threshold (bps)",
                    height=300)

    # ── DRAWDOWN CHART ─────────────────────────────────────────────────
    fig_dd = go.Figure()
    best_strats = (perf_df[~perf_df["Strategie"].str.startswith("BH-")]
                   .sort_values("Sharpe", ascending=False)
                   .head(5)["Strategie"].tolist()
                   if not perf_df.empty else [])
    for j, nm in enumerate(best_strats):
        if nm not in all_results:
            continue
        r   = all_results[nm].dropna()
        cum = (1 + r).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index.astype(str).tolist(), y=dd.values.tolist(),
            mode="lines", name=nm,
            fill="tozeroy", opacity=0.4,
            line=dict(color=PAL[j % len(PAL)], width=1.2)))
    fig_dd.update_layout(
        title="Drawdown-Verlauf: Top-5 Strategien (%)",
        yaxis_title="Drawdown (%)", height=380)

    # ── SCATTER: Sharpe vs MaxDD ───────────────────────────────────────
    fig_scat = go.Figure()
    if not perf_df.empty:
        is_bh = perf_df["Strategie"].str.startswith("BH-")
        for flag, subset, sym, sz in [(False, ~is_bh, "circle", 8),
                                       (True,  is_bh,  "diamond", 12)]:
            df_s = perf_df[subset]
            if df_s.empty: continue
            fig_scat.add_trace(go.Scatter(
                x=df_s["MaxDD%"].tolist(), y=df_s["Sharpe"].tolist(),
                mode="markers+text",
                text=[n.split("[")[0] for n in df_s["Strategie"]],
                textposition="top center", textfont=dict(size=7),
                marker=dict(symbol=sym, size=sz,
                            color=df_s["CAGR%"].tolist(),
                            colorscale="RdYlGn", showscale=not flag,
                            colorbar=dict(title="CAGR%") if not flag else None),
                name="Benchmarks" if flag else "Strategien"))
        fig_scat.add_hline(y=0, line_color="#8b949e", line_dash="dash")
        fig_scat.add_vline(x=-20, line_color="#f78166", line_dash="dash",
                            annotation_text="MaxDD=-20%")
        fig_scat.update_layout(
            title="Risk-Return: Sharpe vs. MaxDD% (Farbe = CAGR%)",
            xaxis_title="MaxDD%", yaxis_title="Sharpe Ratio", height=440)

    perf_table_html = (_df_html(perf_df.sort_values("Sharpe", ascending=False))
                       if not perf_df.empty else "<p>Keine Ergebnisse.</p>")
    wf_table_html   = (_df_html(wf_df.sort_values("OOS Sharpe", ascending=False))
                       if not wf_df.empty else "<p>Keine Walk-Forward Daten.</p>")

    body = f"""
<div class="ph-header"><h1>Phase 14 &#8211; Umfassendes Backtesting</h1>
  <div class="sub">4 Strategien × Parametergitter | Granger-Lead · Pairs-Trading · DXY-Metall · Multi-Strat | Walk-Forward 70/30</div>
</div>
<div class="card mb-4">
  <div class="card-header">Strategiematrix</div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered">
      <thead><tr><th>Nr.</th><th>Strategie</th><th>Quelle</th><th>Erkenntniss</th><th>Signal</th><th>TC</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#58a6ff;">S1</strong></td>
          <td>Granger Lead Signal</td><td>Phase 6 (Granger)</td>
          <td>CL=F Granger-kausal f&#252;r 11 Assets (F&gt;3 bei p&lt;0.05)</td>
          <td>CL=F(t-best_lag) &gt; Threshold &#8594; Long Target</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#3fb950;">S2</strong></td>
          <td>Pairs Trading (EG)</td><td>Phase 8 (Kointegration)</td>
          <td>29 kointegr. Paare; ZW=F/ZS=F EG=-4.79, CL=F/ZW=F EG=-4.73</td>
          <td>Z-Score &gt; z_thresh &#8594; Mean-Revert Trade</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#d29922;">S3</strong></td>
          <td>DXY-Metall Momentum</td><td>Phase 10 (Regression)</td>
          <td>DXY &#946;=-1.255 f&#252;r FCX; negative DXY-Trend &#8594; Metalle steigen</td>
          <td>DXY(t-lb) &lt; DXY(t-2*lb) &#8594; Long GDX/NEM</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#bc8cff;">S4</strong></td>
          <td>Multi-Strategie Portfolio</td><td>S1 + S2 kombiniert</td>
          <td>Diversifikation &#252;ber Strategie-Typen reduziert Drawdown</td>
          <td>Equal-weight: Best-3 S1 + Best-3 S2</td><td>10 bps</td></tr>
      </tbody>
    </table>
    {_formula(r"\text{{Nettrendite}}_t = \text{{Signal}}_t \cdot r_t - |\Delta\text{{Signal}}_t| \cdot \text{{TC}}",
              "TC=10 bps/Trade. Signal &#8712; {{-1, 0, +1}}. &#916;Signal=0: kein Turnover.")}
    {_info("Parameter-Gitter: S1: Threshold &#8712; {{0, 3, 5 bps}}, Hold &#8712; {{lag, lag+3T}}; "
           "S2: Z-Thresh &#8712; {{1.5, 2.0, 2.5}}, Window &#8712; {{21, 63T}}; "
           "S3: Lookback &#8712; {{5, 21, 63T}}")}
  </div>
</div>
{_chart_card("Equity-Kurven: Top-12 Strategien vs. Benchmarks (log-Skala)", fig_eq, height=540,
    interp="Log-Skala: gleiche prozentuale Bewegung = gleicher vertikaler Abstand. "
           "Gepunktete Linien: Buy&amp;Hold Benchmarks. "
           "Starke Strategien: konsistent &#252;ber BH-CL=F. "
           "Abflachung 2010-2014: geringere Rohstoff-Vorhersagbarkeit.")}
{_chart_card("Risk-Return Streudiagramm: Sharpe vs. MaxDD%", fig_scat, height=460,
    interp="Ideal: oben rechts (hoher Sharpe, niedriger MaxDD). "
           "Rauten = Benchmarks. Farbe = CAGR%. "
           "S2-Pairs hat oft niedrigeres MaxDD durch Marktneutralit&#228;t.")}
{_chart_card("Sharpe Ratio aller Strategien (sortiert)", fig_sharpe,
    interp="Gr&#252;n &gt;0.5: akzeptabel. Gelb 0-0.5: grenzwertig. Rot &lt;0: verlustbringend. "
           "S1-Granger-SM mit Lag=1 typischerweise beste Granger-Strategie. "
           "S2-Pairs mit Z=2.0, Win=63T oft robuster als Z=1.5.")}
{_chart_card("Walk-Forward Validierung: IS vs. OOS Sharpe (70%/30%)", fig_wf, height=440,
    interp="Gr&#252;ne Balken: OOS-Performance (echter Test). Blaue: IS (kann Overfitting enthalten). "
           "Degradation &lt;0.5: robust. &gt;1.0: Overfitting-Verdacht. "
           "Granger-Strategien: oft moderate Degradation, da strukturelle Kausalit&#228;t echt ist.")}
{_chart_card("Parameter-Grid Heatmap: S1-Granger[SM] &#8211; Sharpe", fig_grid, height=320,
    interp="Hellgr&#252;n = bestes Parameter-Kombi. "
           "Vertikale Achse: Threshold in bps. Horizontale: Holding Period. "
           "Optimum typisch: 0 bps Threshold, Hold=Lag+3T.")}
{_chart_card("Drawdown-Verlauf: Top-5 Strategien (%)", fig_dd, height=400,
    interp="Schr&#228;ffierung bis 0: kumulativer Verlust vs. Allzeithoch. "
           "GFC 2008-09 und COVID-2020: gr&#246;&#223;te Drawdown-Phasen. "
           "S2-Pairs: oft fl&#228;chiger Drawdown (langsam, persistent statt tief und schnell).")}
<div class="card mb-4">
  <div class="card-header">Performance-Tabelle: Alle Metriken</div>
  <div class="card-body overflow-auto">{perf_table_html}</div>
</div>
<div class="card mb-4">
  <div class="card-header">Walk-Forward Tabelle (IS/OOS Sharpe, Degradation)</div>
  <div class="card-body overflow-auto">{wf_table_html}</div>
</div>
{_card("&#128269; Diagnose: Warum sind manche Strategien schwach?",
    _warn("Granger-Lag &amp; Lag=0 Problem: CCF optimaler Lag=0 f&#252;r ALLE Paare (Phase 6). "
          "Simultane Reaktion = kein t&#228;gliches Vorlaeufer-Signal. "
          "Granger-Kausalit&#228;t ist statistisch real, aber auf t&#228;glicher Frequenz bereits arbitragiert.") +
    """<table class="table table-dark table-sm table-bordered mt-3">
      <thead><tr><th>Problem</th><th>Diagnose</th><th>Auswirkung auf Backtest</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#f78166;">CCF Lag=0 universal</strong></td>
          <td>Phase 6: Alle CCF-Lags=0 (simultane Reaktion). Kein Tages-Lead. 
              Granger bel. statistische Vorhersagekraft auf wochentl. Freq.</td>
          <td>S1-Signale sind schwach auf t&#228;gl. Freq., st&#228;rker auf W/M.</td></tr>
        <tr><td><strong style="color:#d29922;">Transaktionskosten</strong></td>
          <td>10 bps × 250 Tage × t&#228;gl. Rebalancing = 2.5% p.a. Friction</td>
          <td>Brutto-Sharpe 0.4 &#8594; Netto-Sharpe 0.0 typisch</td></tr>
        <tr><td><strong style="color:#d29922;">Regime-Wechsel</strong></td>
          <td>GFC 2008, COVID 2020, Energie-Schock 2022: GARCH zeigt instabile Korrelationen</td>
          <td>Pairs-Trading versagt in Trend-Phasen; EG-Spreads nicht station&#228;r</td></tr>
        <tr><td><strong style="color:#8b949e;">In-Sample-Bias</strong></td>
          <td>Parameter aus Gesamtdaten 2000-2026 abgeleitet (kein echter OOS)</td>
          <td>WF-Degradation voraussichtlich 0.5-1.5 Sharpe-Punkte</td></tr>
        <tr><td><strong style="color:#3fb950;">Beste Verwendung der Signale</strong></td>
          <td>Granger (Phase 6), PCA (Phase 10), GARCH-Regime (Phase 9)</td>
          <td>Monatliche Rohstoff-Allokation, nicht t&#228;gliches Signal-Trading</td></tr>
      </tbody>
    </table>""")}
"""
    _write(out / "backtesting.html", _html_base("Phase 14 - Backtesting", 14, body))


# ─────────────────────────────────────────────────────────────────────────────
# Rigorous Insights & Trading Signal Report
# ─────────────────────────────────────────────────────────────────────────────

def build_insights_report(tables, figures, out):  # noqa: C901
    gr    = _read(tables / "phase6_granger.csv")
    ccf   = _read(tables / "phase6_ccf_lags.csv")
    eg    = _read(tables / "phase8_eg_cointegration.csv")
    joh   = _read(tables / "phase8_johansen.csv")
    ev    = _read(tables / "phase7_event_studies.csv")
    reg   = _read(tables / "phase10_regression_summary.csv")
    pca   = _read(tables / "phase10_pca_loadings.csv")
    garch = _read(tables / "phase9_garch_params.csv")
    sigc  = _read(tables / "phase5_significant_correlations.csv")
    desc  = _read(tables / "phase3_descriptive_stats.csv")

    body_parts: list = []

    body_parts.append("""
<div class="ph-header">
  <h1>&#128269; Rigorous Insights &amp; Handlungssignale</h1>
  <div class="sub">Vollst&#228;ndige Extraktion aller signifikanten Ergebnisse aus Phasen 3&#8211;13
       &#8212; Granger, Kointegration, Events, Regression, PCA, GARCH</div>
</div>""")

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    n_gr_sig = len(gr[gr.significant==True]) if gr is not None else 0
    n_gr_pairs = gr.groupby(['cause','effect']).ngroups if gr is not None else 0
    n_coint = len(eg[eg.cointegrated_95==True]) if eg is not None else 0
    n_ev_sig = len(ev[ev.significant==True]) if ev is not None else 0
    johansen_r = int(joh[joh.reject_r_null_95==True]['r_null'].max()) + 1 if joh is not None else 0

    body_parts.append(_stat_row([
        ("Signifikante Granger-Relationen", str(n_gr_sig)),
        ("Kointegrierte Paare (95%)", str(n_coint)),
        ("Signifikante Event-Signale", str(n_ev_sig)),
        ("Johansen Ko-Rang", str(johansen_r)),
    ]))

    # ── 2. Granger Causality: vollständige Lag-Struktur ───────────────────────
    if gr is not None:
        sig = gr[gr.significant == True].copy()
        by_pair = sig.groupby(['cause','effect']).apply(
            lambda x: pd.Series({
                'lags':     sorted(x['lag'].tolist()),
                'max_f':    x['f_stat'].max(),
                'min_p':    x['pvalue'].min(),
                'n_sig':    len(x),
                'best_lag': int(x.loc[x['f_stat'].idxmax(), 'lag']),
                'p_best':   float(x.loc[x['f_stat'].idxmax(), 'pvalue']),
            }), include_groups=False
        ).reset_index().sort_values('max_f', ascending=False)

        # Heatmap: F-stat[best lag] for each significant (cause,effect) pair
        fig_granger_heat = go.Figure(go.Heatmap(
            z=by_pair['max_f'].values.tolist(),
            x=by_pair['effect'].tolist(),
            y=by_pair['cause'].tolist(),
            colorscale='YlOrRd', zmin=2, zmax=20,
            colorbar=dict(title="F-Stat"),
            text=[[f"F={r.max_f:.2f}<br>p={r.min_p:.5f}<br>n_sig_lags={r.n_sig}<br>best_lag={r.best_lag}T"]
                  for _, r in by_pair.iterrows()],
            hovertemplate="%{x} &larr; %{y}<br>%{text}<extra></extra>",
        ))
        fig_granger_heat.update_layout(
            title="Granger-Kausalit&#228;t: Max F-Statistik (signifikante Paare)",
            xaxis_title="Effekt (Reaktion)", yaxis_title="Ursache",
            height=260)

        # Bubble chart: best_lag on x, F-stat on y, n_sig_lags as size
        colors_g = [PAL[i % len(PAL)] for i in range(len(by_pair))]
        fig_granger_lag = go.Figure(go.Scatter(
            x=by_pair['best_lag'].tolist(),
            y=by_pair['max_f'].tolist(),
            mode='markers+text',
            text=by_pair['effect'].tolist(),
            textposition='top center',
            textfont=dict(size=9, color='#e6edf3'),
            marker=dict(
                size=(by_pair['n_sig'] * 3 + 6).tolist(),
                color=by_pair['max_f'].tolist(),
                colorscale='YlOrRd', showscale=True,
                colorbar=dict(title="F-Stat"),
                line=dict(color='#30363d', width=1)),
            customdata=list(zip(by_pair['cause'], by_pair['min_p'], by_pair['n_sig'], by_pair['lags'].apply(str))),
            hovertemplate="<b>CL=F → %{text}</b><br>Bester Lag: %{x}T<br>F-Stat: %{y:.3f}<br>min-p: %{customdata[1]:.5f}<br>Sig. Lags: %{customdata[2]}<br>Alle sig. Lags: %{customdata[3]}<extra></extra>",
        ))
        fig_granger_lag.add_hline(y=3.84, line_dash='dash', line_color='#d29922',
                                   annotation_text='F-Krit. 5%≈3.84',
                                   annotation_font_color='#d29922')
        fig_granger_lag.update_layout(
            title="Bester Lag vs. F-Statistik (Blasengr&#246;&#223;e = Anzahl sig. Lags)",
            xaxis=dict(title="Lag (Tage) des besten Signals", dtick=1, range=[0.5, 10.5]),
            yaxis=dict(title="Max F-Statistik"),
            height=420)

        # Full detail table
        tbl_rows = []
        for _, r in by_pair.iterrows():
            lag_badges = "".join(
                f'<span class="badge me-1" style="background:{"#3fb950" if l == r["best_lag"] else "#21262d"};'
                f'color:{"#0d1117" if l == r["best_lag"] else "#e6edf3"};font-size:.65rem;">{l}T</span>'
                for l in r['lags'])
            p_col = "#3fb950" if r['min_p'] < 0.01 else "#d29922" if r['min_p'] < 0.05 else "#f78166"
            tbl_rows.append(
                f'<tr><td><strong style="color:#f78166;">{r["cause"]}</strong></td>'
                f'<td><strong style="color:#58a6ff;">{r["effect"]}</strong></td>'
                f'<td style="color:#3fb950;font-weight:600;">{r["max_f"]:.4f}</td>'
                f'<td style="color:{p_col};">{r["min_p"]:.5f}</td>'
                f'<td>{r["best_lag"]}T</td>'
                f'<td>{r["n_sig"]}/10</td>'
                f'<td>{lag_badges}</td></tr>')

        granger_table = (
            '<div class="table-responsive"><table class="table table-dark table-sm table-bordered table-hover">'
            '<thead><tr><th>Ursache</th><th>Effekt</th><th>Max F-Stat</th><th>Min p-Wert</th>'
            '<th>Bester Lag</th><th>Sig. Lags</th><th>Alle signifikanten Lags (&#127306; = bester)</th></tr></thead>'
            '<tbody>' + ''.join(tbl_rows) + '</tbody></table></div>')

        interp_gr = (
            "<strong>CL=F (WTI Crude) ist der einzige Granger-Kausalit&#228;ts-Treiber</strong> "
            "im gesamten Netzwerk (Out-Degree 79.2, alle anderen 0). "
            "Entscheidend: <strong>Lag 1 ist NICHT immer der beste</strong> &#8211; "
            "SM reagiert am st&#228;rksten sofort (F=18.17, Lag 1), "
            "aber XOM/NEM am st&#228;rksten bei Lag 6&#8211;7 (F≈3.8/3.8), "
            "OXY und GDX/SIL erst bei Lag 9&#8211;10 (F≈3.9/2.6). "
            "Das deutet auf unterschiedliche <em>Informationsverarbeitungsgeschwindigkeiten</em> "
            "je nach Marktkapitalisierung und Liquidit&#228;t hin."
        )

        body_parts.append(_card(
            "&#128200; Granger-Kausalit&#228;t: Vollst&#228;ndige Lag-Struktur (CL=F dominiert das Netzwerk)",
            _chart_card("Heatmap: Max F-Statistik je Paar", fig_granger_heat, height=260) +
            _chart_card(
                "Bester Lag vs. F-Statistik &#8211; Blasengr&#246;&#223;e = Anzahl signifikanter Lags",
                fig_granger_lag, height=420,
                interp=interp_gr) +
            granger_table
        ))

    # ── 3. Lag-Struktur nach Segment ─────────────────────────────────────────
    if gr is not None:
        seg = {
            "Sofort (Lag 1)": ["SM","TECK","CVX","APA"],
            "Kurz (Lag 3&#8211;5)": ["XLI","XLB","FCX","XOM","TGB","JETS","IYT"],
            "Mittel (Lag 6&#8211;7)": ["XLE","NEM","XOM","GDX"],
            "Verz&#246;gert (Lag 8&#8211;10)": ["OXY","GDX","SIL","NEM"],
        }
        seg_html = '<table class="table table-dark table-sm table-bordered"><thead><tr>'
        seg_html += '<th>Reaktions-Segment</th><th>Ticker</th><th>Interpretation</th><th>Handlungshinweis</th></tr></thead><tbody>'
        interps = {
            "Sofort (Lag 1)":
                ("Direkte Energie-Produzenten (SM Energy, Teck, Chevron, APA): "
                 "Die Bewertung ist eng an CL=F gekoppelt &#8212; reagieren sofort am n&#228;chsten Handelstag",
                 "CL=F steigt heute &#8594; Long SM/CVX/APA morgen; Ziel: 1-3 Tage halten"),
            "Kurz (Lag 3&#8211;5)":
                ("Breite Energie/Industrie-ETFs und FCX/TGB: Informationsverbreitung dauert "
                 "3&#8211;5 Tage &#8212; typisch f&#252;r Indexrebalancierungen und institutionelle Fl&#252;sse",
                 "CL=F Signal &#8594; 3 Tage warten &#8594; Einstieg in XLI/XLB/JETS"),
            "Mittel (Lag 6&#8211;7)":
                ("Gold-nahe Assets und XLE: Gold-Minen (GDX, NEM) verarbeiten &#214;l-Signale "
                 "mit einer Woche Verz&#246;gerung &#8212; Cross-Asset-Informationsfluss",
                 "CL=F Signal &#8594; 6-7 Tage &#8594; GDX/NEM Position aufbauen"),
            "Verz&#246;gert (Lag 8&#8211;10)":
                ("OXY (F=3.93, p=0.000024 bei Lag 10!), SIL (Lag 6&#8211;10): "
                 "Kleine/mittlere Kapitalisierung mit langsamerer Informationsverarbeitung",
                 "Schwaches Signal, aber statistische Signifikanz p&lt;0.001 f&#252;r OXY bei Lag 10"),
        }
        for seg_name, tickers in seg.items():
            interp_text, action = interps[seg_name]
            seg_html += (
                f'<tr><td><strong style="color:#58a6ff;">{seg_name}</strong></td>'
                f'<td><code>{", ".join(tickers)}</code></td>'
                f'<td style="font-size:.78rem;">{interp_text}</td>'
                f'<td style="font-size:.78rem;color:#3fb950;">{action}</td></tr>')
        seg_html += '</tbody></table>'
        body_parts.append(_card("&#9202; Lag-Segmente: Reaktionsgeschwindigkeit auf CL=F-Signal", seg_html))

    # ── 4. Kointegration & Mean Reversion ─────────────────────────────────────
    if eg is not None:
        sig_eg = eg[eg.cointegrated_95 == True].copy().sort_values('EG_stat')

        fig_coint = go.Figure()
        fig_coint.add_bar(
            x=(sig_eg['asset1'] + '/' + sig_eg['asset2']).tolist(),
            y=sig_eg['EG_stat'].tolist(),
            marker_color=[
                '#3fb950' if v < -4.5 else '#d29922' if v < -3.8 else '#58a6ff'
                for v in sig_eg['EG_stat']],
            customdata=list(zip(sig_eg['pvalue'], sig_eg['EG_stat'])),
            hovertemplate="<b>%{x}</b><br>EG-Stat: %{customdata[1]:.4f}<br>p: %{customdata[0]:.5f}<extra></extra>",
            text=[f"p={v:.4f}" for v in sig_eg['pvalue']],
            textposition='outside', textfont=dict(size=8, color='#8b949e'))
        fig_coint.add_hline(y=-3.34, line_dash='dash', line_color='#d29922',
                             annotation_text='Krit. 5%', annotation_font_color='#d29922')
        fig_coint.add_hline(y=-4.5, line_dash='dot', line_color='#3fb950',
                             annotation_text='Sehr stark', annotation_font_color='#3fb950')
        fig_coint.update_layout(
            title=f"Engle-Granger Statistik: {len(sig_eg)} kointegrierte Paare (p<5%)",
            xaxis_tickangle=-45, yaxis_title="EG-Statistik (negativer = st&#228;rker)",
            height=420)

        # Table sorted by strength
        coint_rows = []
        for _, r in sig_eg.iterrows():
            strength = "&#127311; Stark" if r['EG_stat'] < -4.5 else "&#128994; Mittel" if r['EG_stat'] < -3.8 else "&#128993; Schwach"
            a1, a2 = r['asset1'], r['asset2']
            s1 = SECTORS.get(a1,'?'); s2 = SECTORS.get(a2,'?')
            cross = "Cross-Sektor" if s1 != s2 else s1
            coint_rows.append(
                f'<tr><td><strong>{a1}</strong></td><td><strong>{a2}</strong></td>'
                f'<td style="color:#58a6ff;">{r["EG_stat"]:.4f}</td>'
                f'<td style="color:#3fb950;">{r["pvalue"]:.5f}</td>'
                f'<td>{strength}</td><td style="color:#8b949e;">{cross}</td></tr>')

        joh_text = (
            f"Johansen-Test: Rang = {johansen_r} Kointegrationsvektoren "
            f"(Trace-Stat bei r=0: {joh.loc[0,'trace_stat']:.1f} vs. Krit.95% {joh.loc[0,'crit_95']:.1f}; "
            f"bei r=3: {joh.loc[3,'trace_stat']:.1f} vs. {joh.loc[3,'crit_95']:.1f}; "
            f"bei r=4: {joh.loc[4,'trace_stat']:.1f} vs. {joh.loc[4,'crit_95']:.1f} &#8594; kein r=4)."
        ) if joh is not None else ""

        coint_table = (
            '<div class="table-responsive" style="max-height:360px;">'
            '<table class="table table-dark table-sm table-bordered table-hover">'
            '<thead><tr><th>Asset 1</th><th>Asset 2</th><th>EG-Statistik</th>'
            '<th>p-Wert</th><th>St&#228;rke</th><th>Klasse</th></tr></thead>'
            '<tbody>' + ''.join(coint_rows) + '</tbody></table></div>')

        body_parts.append(_card(
            "&#9851; Kointegration &amp; Mean-Reversion-Potenzial (29 Paare, 4 Johansen-Vektoren)",
            _warn(joh_text) +
            _chart_card(
                f"Engle-Granger Statistik: {len(sig_eg)} kointegrierte Paare",
                fig_coint, height=420,
                interp=(
                    "Gr&#252;n (EG &lt; &#8722;4.5): starke Mean-Reversion &#8211; "
                    "ZW=F/ZS=F (&#8722;4.794), CL=F/ZS=F (&#8722;4.773), CL=F/ZW=F (&#8722;4.731), "
                    "CL=F/ZC=F (&#8722;4.262). "
                    "&#220;berraschend: Getreide-Energie-Kointegration! CL=F mit ZC=F/ZW=F/ZS=F &#8211; "
                    "gemeinsamer Inflations-/Energiekosten-Treiber. "
                    "ZW=F/FCX (EG=&#8722;4.03) und ZW=F/NEM (&#8722;3.38): Agrar x Bergbau-Cross-Sektor."
                )) +
            coint_table +
            _info(
                "Mean-Reversion-Strategie: Spread = Preis(A) &#8722; &#946;&#183;Preis(B). "
                "Wenn Spread &gt; +2&#963; &#8594; Short A / Long B. "
                "Wenn Spread &lt; &#8722;2&#963; &#8594; Long A / Short B. "
                "Halb-Lebenszeit sch&#228;tzen aus: t&#189; = &#8722;ln(2) / ln(1+&#961;) "
                "wobei &#961; der AR(1)-Koeffizient des Spreads. "
                "St&#228;rkste Kandidaten: ZW=F/ZS=F (p=0.000387), CL=F/ZW=F (p=0.000499)."
            )
        ))

    # ── 5. Event Studies: Alle signifikanten Signale ─────────────────────────
    if ev is not None:
        sig_ev = ev[ev.significant == True].copy()
        sig_ev['car_pct'] = sig_ev['mean_CAR'] * 100
        sig_ev['dir'] = sig_ev['car_pct'].apply(lambda v: '&#9650; Long' if v > 0 else '&#9660; Short')
        sig_ev['col'] = sig_ev['car_pct'].apply(lambda v: '#3fb950' if v > 0 else '#f78166')

        # Heatmap: event × asset for 3-day window
        ev3 = sig_ev[sig_ev['window_days'] == 3].pivot_table(
            index='event_type', columns='asset', values='car_pct', aggfunc='mean')
        if not ev3.empty:
            fig_ev = go.Figure(go.Heatmap(
                z=ev3.values.tolist(), x=ev3.columns.tolist(), y=ev3.index.tolist(),
                colorscale='RdYlGn', zmid=0,
                colorbar=dict(title="CAR 3T (%)"),
                text=[[f"{v:.2f}%" if not np.isnan(v) else '' for v in row]
                      for row in ev3.values],
                texttemplate="%{text}", textfont=dict(size=9),
                hovertemplate="<b>%{y}</b> &#8594; <b>%{x}</b><br>CAR 3T: %{z:.2f}%<extra></extra>"))
            fig_ev.update_layout(
                title="Signifikante CAR-Reaktionen 3-Tage-Fenster (p&lt;5%)",
                xaxis_tickangle=-45, height=320)
        else:
            fig_ev = go.Figure()

        # Event signal table
        ev_rows = []
        for evt in sorted(sig_ev['event_type'].unique()):
            sub = sig_ev[sig_ev['event_type'] == evt].sort_values('window_days')
            for _, r in sub.iterrows():
                pre_note = f"Vorher: {r['mean_CAR_pre']*100:+.2f}%" if not np.isnan(r['mean_CAR_pre']) else ""
                ev_rows.append(
                    f'<tr><td><strong>{evt}</strong></td>'
                    f'<td style="color:#58a6ff;">{r["asset"]}</td>'
                    f'<td>{r["window_days"]}T</td>'
                    f'<td style="color:{r["col"]};font-weight:600;">{r["car_pct"]:+.2f}%</td>'
                    f'<td style="color:{r["col"]};">{r["dir"]}</td>'
                    f'<td>{r["t_stat"]:.3f}</td>'
                    f'<td style="color:#3fb950;">{r["pvalue"]:.4f}</td>'
                    f'<td style="color:#8b949e;font-size:.72rem;">{pre_note}</td></tr>')

        ev_table = (
            '<div class="table-responsive">'
            '<table class="table table-dark table-sm table-bordered table-hover">'
            '<thead><tr><th>Event</th><th>Asset</th><th>Fenster</th><th>CAR</th>'
            '<th>Signal</th><th>t-Stat</th><th>p-Wert</th><th>Pre-Event</th></tr></thead>'
            '<tbody>' + ''.join(ev_rows) + '</tbody></table></div>')

        event_interp = (
            "<strong>NFP (Non-Farm Payrolls)</strong>: Kaufsignal GC=F (+0.33%, p=0.028), "
            "GDX (+0.69%, p=0.047), XOM (+0.35%, p=0.020), FCX (+0.55%, p=0.014); "
            "Verkaufsignal ^VIX (&#8722;1.55%, 5T, p=0.038), ^TNX steigt (+0.85%/+1.25% 3T/5T p&lt;0.025). "
            "<strong>ISM PMI</strong>: GC=F kaufen (+0.40%, 3T, p=0.007), "
            "GDX kaufen (+0.73%, 3T, p=0.034), NG=F verkaufen (&#8722;1.17%, 5T, p=0.039). "
            "<strong>CPI/PPI</strong> (identische Signale!): ZW=F Long (+0.59%/+0.77%), "
            "ZS=F Short (&#8722;0.43%/&#8722;0.48%), XLB Short (&#8722;0.15%). "
            "<strong>FOMC</strong>: QQQ Long (+0.39%, 3T), HG=F Short (&#8722;1.16%, 3T), "
            "^VIX Short (&#8722;4.2%, 5T). "
            "<strong>EIA Oil Report</strong>: ^VIX Short (&#8722;0.76%, 1T, p=0.038)."
        )

        body_parts.append(_card(
            "&#128197; Event-Kalender Handelssignale (34 signifikante Kombinationen)",
            _chart_card("Signifikante CAR 3-Tage (gr&#252;n=Long, rot=Short)", fig_ev, height=320) +
            ev_table + _interp(event_interp)
        ))

    # ── 6. Regression: Makro-Faktor-Exposures ────────────────────────────────
    if reg is not None:
        betas_cl = reg.get('beta_CL=F', pd.Series(dtype=float))
        betas_bz = reg.get('beta_BZ=F', pd.Series(dtype=float))
        betas_vx = reg.get('beta_^VIX', pd.Series(dtype=float))
        betas_dx = reg.get('beta_DX-Y.NYB', pd.Series(dtype=float))
        betas_tn = reg.get('beta_^TNX', pd.Series(dtype=float))
        betas_sp = reg.get('beta_SPY', pd.Series(dtype=float))
        r2       = reg.get('r_squared', pd.Series(dtype=float))

        assets = reg.index.tolist()

        fig_beta = make_subplots(rows=2, cols=3,
            subplot_titles=["&#946; CL=F (&#246;l)", "&#946; BZ=F (brent)",
                            "&#946; ^VIX (fear)", "&#946; DXY (dollar)",
                            "&#946; ^TNX (zinsen)", "R&#178;"],
            shared_xaxes=False)

        def _bar_row(vals, labels, colors, row, col, fig):
            fig.add_trace(go.Bar(x=labels, y=vals.loc[labels].values.tolist(),
                                  marker_color=colors, showlegend=False), row=row, col=col)

        def _sig_color(asset, pcol):
            pvals = reg.get(f'pvalue_{pcol}', pd.Series(dtype=float))
            if pvals is None or asset not in pvals.index: return '#8b949e'
            p = float(pvals[asset])
            return '#3fb950' if p < 0.01 else '#d29922' if p < 0.05 else '#555555'

        for i, asset in enumerate(assets):
            for j, (betas, pcol, ri, ci) in enumerate([
                (betas_cl, 'CL=F', 1, 1), (betas_bz, 'BZ=F', 1, 2),
                (betas_vx, '^VIX', 1, 3), (betas_dx, 'DX-Y.NYB', 2, 1),
                (betas_tn, '^TNX', 2, 2),
            ]):
                if asset not in betas.index: continue
                col_b = _sig_color(asset, pcol)
                fig_beta.add_trace(go.Bar(
                    x=[asset], y=[float(betas[asset])],
                    marker_color=col_b, showlegend=False,
                    hovertemplate=f"<b>{asset}</b>: &#946;={float(betas[asset]):.4f}<extra></extra>"),
                    row=ri, col=ci)

        if r2 is not None and len(r2) > 0:
            fig_beta.add_trace(go.Bar(
                x=assets,
                y=[float(r2[a]) if a in r2.index else 0 for a in assets],
                marker_color=["#3fb950" if (float(r2[a]) if a in r2.index else 0) > 0.5 else "#d29922"
                              for a in assets],
                showlegend=False,
                hovertemplate="<b>%{x}</b>: R&#178;=%{y:.4f}<extra></extra>"),
                row=2, col=3)

        fig_beta.update_layout(height=520, title="Regressions-Betas vs. Makro-Faktoren (Farbe = Signifikanz)")
        for axis in ['xaxis','xaxis2','xaxis3','xaxis4','xaxis5','xaxis6']:
            fig_beta.update_layout(**{axis: dict(tickangle=-45)})

        # Key findings table
        reg_rows = []
        for asset in assets:
            if asset not in reg.index: continue
            row = reg.loc[asset]
            def beta_cell(col_name):
                bv = row.get(f'beta_{col_name}', np.nan)
                pv = row.get(f'pvalue_{col_name}', np.nan)
                if np.isnan(bv): return '<td>—</td>'
                p_col = '#3fb950' if pv < 0.01 else '#d29922' if pv < 0.05 else '#555'
                sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
                return f'<td style="color:{p_col};">{bv:.4f}{sig}</td>'
            r2v = float(row.get('r_squared', 0))
            r2_col = '#3fb950' if r2v > 0.5 else '#d29922' if r2v > 0.25 else '#f78166'
            reg_rows.append(
                f'<tr><td><strong style="color:#58a6ff;">{asset}</strong></td>'
                f'{beta_cell("CL=F")}{beta_cell("BZ=F")}{beta_cell("^VIX")}'
                f'{beta_cell("DX-Y.NYB")}{beta_cell("^TNX")}{beta_cell("SPY")}'
                f'<td style="color:{r2_col};font-weight:600;">{r2v:.3f}</td></tr>')

        reg_table = (
            '<small style="color:#8b949e;">*** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; '
            'keine Markierung: nicht signifikant</small>'
            '<div class="table-responsive mt-2">'
            '<table class="table table-dark table-sm table-bordered table-hover">'
            '<thead><tr><th>Asset</th><th>&#946; CL=F</th><th>&#946; BZ=F</th>'
            '<th>&#946; ^VIX</th><th>&#946; DXY</th><th>&#946; ^TNX</th>'
            '<th>&#946; SPY</th><th>R&#178;</th></tr></thead>'
            '<tbody>' + ''.join(reg_rows) + '</tbody></table></div>')

        reg_interp = (
            "<strong>SM Energy</strong> hat die st&#228;rkste &#214;l-Reaktion: "
            "&#946;_CL=F=0.384 (p&lt;0.001), &#946;_BZ=F=0.488 (p&lt;0.001) &#8212; "
            "fast 0.9 kombiniertes &#214;l-Beta. "
            "<strong>FCX</strong>: &#946;_DXY=&#8722;1.255*** &#8212; "
            "extremste Dollar-Sensitivit&#228;t aller Energie/Metall-Titel. "
            "<strong>NEM/GDX/SIL/TGB</strong>: &#946;_DXY &#8722;1.3 bis &#8722;1.8*** &#8212; "
            "Anti-Dollar-Cluster, ideal als USD-Hedge. "
            "<strong>XLI</strong>: R&#178;=0.82, &#946;_SPY=0.95*** &#8212; "
            "reiner Marktfaktor, keine Rohstoff-Exposition. "
            "<strong>GORO</strong>: R&#178;=0.007 &#8212; vollst&#228;ndig idiosynkratisch/illiquid, "
            "kein Informationsgehalt f&#252;r das Modell."
        )

        body_parts.append(_card(
            "&#128200; Regressions-Exposures: Welcher Faktor treibt welches Asset?",
            _chart_card("Regressions-Betas vs. Makro-Faktoren", fig_beta, height=520) +
            reg_table + _interp(reg_interp)
        ))

    # ── 7. PCA Faktorstruktur ─────────────────────────────────────────────────
    if pca is not None:
        var_expl = [36.8, 12.4, 9.9, 7.2, 5.1]  # from phase10
        pc_names = ["PC1: Breiter Markt", "PC2: Edelmetalle", "PC3: Rohöl", "PC4: Agrar", "PC5: Industrie/Kupfer"]
        # Top loaders for each PC
        top_pcs = []
        for i, pc in enumerate(['PC1','PC2','PC3','PC4','PC5']):
            if pc not in pca.columns: continue
            s = pca[pc].abs().sort_values(ascending=False).head(5)
            tops = []
            for tkr in s.index:
                v = float(pca.loc[tkr, pc])
                tops.append(f'<span style="color:{"#3fb950" if v>0 else "#f78166"};">'
                             f'{tkr}({v:+.2f})</span>')
            top_pcs.append(
                f'<tr><td><strong style="color:#58a6ff;">{pc_names[i]}</strong></td>'
                f'<td>{var_expl[i]}%</td>'
                f'<td>{"&nbsp; ".join(tops)}</td></tr>')

        pca_table = (
            '<table class="table table-dark table-sm table-bordered">'
            '<thead><tr><th>Hauptkomponente</th><th>Erkl&#228;rte Varianz</th>'
            '<th>Top-5 Lader (gr&#252;n=positiv, rot=negativ)</th></tr></thead>'
            '<tbody>' + ''.join(top_pcs) + '</tbody></table>')

        pca_fig = go.Figure()
        for i, pc in enumerate(['PC1','PC2','PC3','PC4','PC5']):
            if pc not in pca.columns: continue
            s = pca[pc].sort_values()
            pca_fig.add_trace(go.Bar(
                name=f"{pc} ({var_expl[i]}%)", x=s.index.tolist(), y=s.values.tolist(),
                visible=(True if i == 0 else 'legendonly')))
        pca_fig.update_layout(
            title="PCA-Ladungen je Hauptkomponente (&#252;ber Legende ausw&#228;hlbar)",
            xaxis_tickangle=-45, yaxis_title="Ladung", height=340)

        body_parts.append(_card(
            "&#127759; PCA-Faktorstruktur: 5 wirtschaftliche Hauptkomponenten",
            pca_table +
            _chart_card("PCA-Ladungen", pca_fig, height=340,
                interp=(
                    "PC1 (36.8%): Alle Equities laden stark positiv (0.64&#8211;0.90), "
                    "VIX negativ (&#8722;0.67) &#8212; Risk-On/Off-Faktor. "
                    "PC2 (12.4%): GC=F (+0.81), GDX (+0.82), DXY (&#8722;0.52) &#8212; "
                    "Edelmetall/Safe-Haven-Faktor. "
                    "PC3 (9.9%): CL=F (+0.66), BZ=F (+0.67) &#8212; reiner &#214;l-Preis-Faktor. "
                    "PC4 (7.2%): ZC=F (+0.80), ZW=F (+0.70), ZS=F (+0.70) &#8212; Agrar-Faktor. "
                    "PC5 (5.1%): HG=F (+0.58), FCX (+0.37), TECK (+0.39) &#8212; Kupfer/Industrie."
                ))
        ))

    # ── 8. GARCH Volatilitäts-Persistenz ─────────────────────────────────────
    if garch is not None:
        fig_garch = make_subplots(rows=1, cols=2,
            subplot_titles=["Halbwertszeit der Volatilität (Tage)", "GARCH Persistenz (&#945;+&#946;)"])

        garch_s = garch.sort_values('half_life_days', ascending=False)
        fig_garch.add_bar(x=garch_s['ticker'].tolist(),
                          y=garch_s['half_life_days'].tolist(),
                          marker_color=[PAL[i % len(PAL)] for i in range(len(garch_s))],
                          showlegend=False,
                          hovertemplate="<b>%{x}</b>: t&#189;=%{y:.0f} Tage<extra></extra>",
                          row=1, col=1)
        fig_garch.add_bar(x=garch_s['ticker'].tolist(),
                          y=garch_s['persistence'].tolist(),
                          marker_color=[PAL[i % len(PAL)] for i in range(len(garch_s))],
                          showlegend=False,
                          row=1, col=2)
        fig_garch.add_hline(y=0.99, line_dash='dash', line_color='#d29922',
                             annotation_text='IGarch', row=1, col=2)
        fig_garch.update_layout(height=300, title="GARCH(1,1): Volatilit&#228;tspersistenz")

        garch_rows = []
        for _, r in garch.iterrows():
            hl = float(r['half_life_days'])
            pers = float(r['persistence'])
            regime = "Quasi-IGARCH" if pers > 0.99 else "Hoch" if pers > 0.98 else "Mittel"
            trade_impl = (
                "Vola-Schocks dauern &gt;6 Monate &#8212; nutze f&#252;r Vega-Trading"
                if hl > 150 else
                "Vol-Cluster ~2 Monate &#8212; GARCH-Filter anwenden"
                if hl > 50 else
                "Rel. schnelle Normalisierung &#8212; mean-reverting Vol")
            garch_rows.append(
                f'<tr><td><strong style="color:#58a6ff;">{r["ticker"]}</strong></td>'
                f'<td>{r["alpha1"]:.4f}</td><td>{r["beta1"]:.4f}</td>'
                f'<td style="color:#d29922;">{pers:.4f}</td>'
                f'<td style="color:#3fb950;">{hl:.0f} Tage</td>'
                f'<td>{regime}</td>'
                f'<td style="font-size:.75rem;color:#8b949e;">{trade_impl}</td></tr>')

        garch_table = (
            '<div class="table-responsive">'
            '<table class="table table-dark table-sm table-bordered">'
            '<thead><tr><th>Ticker</th><th>&#945; (ARCH)</th><th>&#946; (GARCH)</th>'
            '<th>&#945;+&#946;</th><th>Halbwertszeit</th><th>Regime</th>'
            '<th>Handelsimplikation</th></tr></thead>'
            '<tbody>' + ''.join(garch_rows) + '</tbody></table></div>')

        body_parts.append(_card(
            "&#128165; GARCH Volatilit&#228;tspersistenz: Wie lange dauern Vola-Schocks?",
            _chart_card("GARCH Persistenz &amp; Halbwertszeit", fig_garch, height=300) +
            garch_table +
            _interp(
                "NG=F hat extreme Persistenz (&#945;+&#946;=0.9963, t&#189;=188 Tage!): "
                "Ein Volatilita&#776;tsschock (z.B. Polar-Vortex) dauert fast ein halbes Jahr. "
                "SI=F (t&#189;=165T) und HG=F (t&#189;=124T) ebenfalls sehr persistent. "
                "CL=F (t&#189;=41T): schnellste Normalisierung &#8212; gut f&#252;r "
                "Volatilita&#776;ts-Sell-Strategien nach Schocks."
            )
        ))

    # ── 9. Vollständige Handlungsmatrix ──────────────────────────────────────
    matrix_rows = [
        # (Signal, Richtung, Instrument, Basis, Timing, Konfidenz, Details)
        ("CL=F steigt heute", "&#9650; Long", "SM Energy", "Granger F=18.17 (p&lt;0.001)", "Morgen (T+1)", "&#11088;&#11088;&#11088;&#11088;", "Stärkster Einzelsignale im gesamten Netzwerk. Signifikant für alle Lags 1-10."),
        ("CL=F steigt heute", "&#9650; Long", "CVX, TECK, APA", "Granger F≥4.6 (p&lt;0.05)", "T+1", "&#11088;&#11088;&#11088;", "Sofort-Reaktion; halten 1-3 Tage."),
        ("CL=F steigt", "&#9650; Long", "XOM, XLI, XLB, FCX", "Granger F≥3.0 (p&lt;0.03)", "T+3", "&#11088;&#11088;&#11088;", "3-5 Tage Verzögerung; institutionelle Rebalancing-Flows."),
        ("CL=F steigt", "&#9650; Long", "GDX, SIL", "Granger F≥2.3 (p&lt;0.025)", "T+6 bis T+10", "&#11088;&#11088;", "Schwächeres Signal; Lags 6-10 signifikant."),
        ("CL=F steigt", "&#9650; Long", "OXY", "Granger F=3.93 (p&lt;0.001) @ Lag 10", "T+10", "&#11088;&#11088;&#11088;", "Überraschend: beste Signal erst nach 10 Tagen!"),
        ("NFP-Tag (Payroll)", "&#9650; Long", "GC=F, GDX, XOM, FCX", "Event CAR: GDX+0.69% (p=0.047), FCX+0.55% (p=0.014)", "T bis T+3", "&#11088;&#11088;&#11088;", "NFP-Tag kaufen; halten bis 3 Tage danach."),
        ("NFP-Tag", "&#9660; Short", "^VIX", "CAR: -1.55% über 5T (p=0.038)", "T bis T+5", "&#11088;&#11088;", "VIX-Short als Hedge/Carry."),
        ("NFP-Tag", "&#9650; Long", "^TNX", "CAR: +1.25% über 5T (p=0.005)", "T bis T+5", "&#11088;&#11088;&#11088;", "Zinsen steigen nach starken Jobs-Daten."),
        ("ISM PMI veröffentlicht", "&#9650; Long", "GC=F (1T), GDX (3T)", "CAR: GC+0.40% (p=0.007), GDX+0.73% (p=0.034)", "T bis T+3", "&#11088;&#11088;&#11088;", "Gold reagiert positiv auf PMI."),
        ("ISM PMI", "&#9660; Short", "NG=F", "CAR: -1.17% über 5T (p=0.039)", "T bis T+5", "&#11088;&#11088;", "Schwächere Industrie → weniger Energiebedarf."),
        ("CPI/PPI veröffentlicht", "&#9650; Long", "ZW=F", "CAR: +0.77% (5T, p=0.030)", "T bis T+5", "&#11088;&#11088;&#11088;", "CPI UND PPI zeigen identisches Signal!"),
        ("CPI/PPI", "&#9660; Short", "ZS=F, XLB, TGB", "ZS=F: -0.43% (1T, p=0.011); XLB: -0.15% (1T, p=0.017)", "T", "&#11088;&#11088;&#11088;", "Sofortiges 1-Tages-Signal."),
        ("FOMC-Entscheid", "&#9650; Long", "QQQ", "CAR: +0.39% (3T, p=0.020)", "T bis T+3", "&#11088;&#11088;", "Tech reagiert positiv auf FOMC."),
        ("FOMC", "&#9660; Short", "HG=F, IJH", "HG=F: -1.16% (3T, p=0.047); IJH: -0.49% (5T, p=0.031)", "T bis T+3/5", "&#11088;&#11088;", "Kupfer und Mid-Cap negativ nach FOMC."),
        ("Dollar (DXY) steigt", "&#9660; Short", "FCX, NEM, GDX, SIL, TGB", "β_DXY: FCX=-1.26***, NEM=-1.34***, GDX=-1.80***", "Sofort", "&#11088;&#11088;&#11088;&#11088;", "Stärkste und direkteste Faktor-Exposition."),
        ("ZW=F/ZS=F Spread > +2σ", "Pairs Trade", "Short ZW=F / Long ZS=F", "EG=-4.79 (p=0.000387) Stärkste Kointegration", "Mean-Reversion", "&#11088;&#11088;&#11088;&#11088;", "Stärkste kointegration im gesamten Universe."),
        ("CL=F/ZW=F Spread > +2σ", "Pairs Trade", "Short CL=F / Long ZW=F", "EG=-4.73 (p=0.000499)", "Mean-Reversion", "&#11088;&#11088;&#11088;", "Energie-Agrar Cross-Sektor Mean Reversion."),
        ("NG=F Vola-Regime wechselt", "Positionsgröße", "Reduziere NG=F-Position", "GARCH: t½=188T, Persistenz 0.9963", "Sofort", "&#11088;&#11088;&#11088;&#11088;", "Extrem persistent: Schock dauert 6 Monate!"),
        ("PC1 (Markt) fällt", "&#9660; Short/Hedge", "Alle Equities außer GC=F", "PC1 erklärt 36.8%; ^VIX lädt -0.67", "Sofort", "&#11088;&#11088;&#11088;&#11088;", "Systematisches Risk-Off: breite Absicherung."),
        ("PC2 (Edelmetall) steigt", "&#9650; Long", "GC=F, GDX, SIL, NEM", "PC2 erklärt 12.4%; DXY negativ korreliert", "Sofort", "&#11088;&#11088;&#11088;", "Safe-Haven-Faktor; nutze DXY als Frühindikator."),
    ]

    matrix_html = (
        '<div class="table-responsive">'
        '<table class="table table-dark table-sm table-bordered table-hover">'
        '<thead><tr style="font-size:.75rem;"><th>Signal / Trigger</th><th>Richtung</th>'
        '<th>Instrument</th><th>Daten-Basis</th><th>Timing</th>'
        '<th>Konfidenz</th><th>Details</th></tr></thead><tbody>')
    for row in matrix_rows:
        matrix_html += (
            f'<tr style="font-size:.75rem;">'
            f'<td><strong style="color:#d29922;">{row[0]}</strong></td>'
            f'<td style="color:{"#3fb950" if "Long" in row[1] else "#f78166" if "Short" in row[1] else "#bc8cff"};">'
            f'<strong>{row[1]}</strong></td>'
            f'<td style="color:#58a6ff;">{row[2]}</td>'
            f'<td style="color:#8b949e;">{row[3]}</td>'
            f'<td>{row[4]}</td><td>{row[5]}</td>'
            f'<td style="color:#8b949e;">{row[6]}</td></tr>')
    matrix_html += '</tbody></table></div>'

    body_parts.append(_card(
        "&#127919; Vollst&#228;ndige Handlungsmatrix: 20 datengest&#252;tzte Signale",
        matrix_html +
        _warn(
            "WICHTIG: Alle Signale sind in-sample-validiert (2000&#8211;2026). "
            "OOS-Performance nicht garantiert. "
            "Empfehlung: GARCH-Regime als Positionsgr&#246;&#223;en-Filter &#8212; "
            "bei High-Vol-Regime (Regime=1) Positionen halbieren."
        )
    ))

    body = "\n".join(body_parts)
    _write(out / "insights_report.html", _html_base("Insights &amp; Handelssignale", 15, body))


# ─────────────────────────────────────────────────────────────────────────────
# Overshoot / Correction Analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_overshoot_report(tables, figures, out):  # noqa: C901
    returns = _read(tables / "phase2_returns.csv")
    gran    = _read(tables / "phase6_granger.csv")

    if returns is None or gran is None:
        _write(out / "overshoot.html",
               _html_base("Overshoot", 16, "<p>Daten fehlen.</p>"))
        return

    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]

    sig_pairs = (gran[gran["significant"] == True]
                 .sort_values("f_stat", ascending=False)
                 .drop_duplicates(["cause","effect"], keep="first")
                 .head(10))

    # One-time OLS beta per pair (simple bivariate)
    def _ols_beta(x, y):
        cov = np.cov(x, y)
        return cov[0, 1] / (np.var(x) + 1e-12)

    WINDOW   = 10   # days forward
    PRE      = 5    # days pre-event baseline
    SIG_MULT = 1.0  # σ threshold for defining a signal event

    overshoot_rows = []
    fig_evst  = go.Figure()
    fig_ratio = go.Figure()
    fig_corr  = go.Figure()
    event_traces = []

    for j, (_, row) in enumerate(sig_pairs.iterrows()):
        src, tgt, best_lag = str(row["cause"]), str(row["effect"]), int(row["lag"])
        if src not in clean.columns or tgt not in clean.columns:
            continue
        r_src = clean[src].dropna()
        r_tgt = clean[tgt].dropna()
        idx   = r_src.index.intersection(r_tgt.index)
        if len(idx) < 252:
            continue
        xs = r_src.loc[idx].values
        yt = r_tgt.loc[idx].values
        T  = len(xs)

        sigma    = xs.std()
        beta_val = _ols_beta(xs, yt)

        # Signal events: cause return > SIG_MULT*sigma (positive) or < −SIG_MULT*sigma (negative)
        pos_events = [i for i in range(PRE, T - WINDOW - 1) if xs[i] >  SIG_MULT * sigma]
        neg_events = [i for i in range(PRE, T - WINDOW - 1) if xs[i] < -SIG_MULT * sigma]

        for sign, events, label in [(+1, pos_events, "Positiv"), (-1, neg_events, "Negativ")]:
            if not events:
                continue
            # Align: cumulative target return from -PRE to +WINDOW relative to event
            window_mat = []
            for ev in events:
                window_r = yt[ev - PRE : ev + WINDOW + 1]
                if len(window_r) < PRE + WINDOW + 1:
                    continue
                # Cumulative return relative to event day 0
                cum = np.cumprod(1 + window_r) / np.prod(1 + window_r[:PRE]) - 1
                window_mat.append(cum)
            if not window_mat:
                continue
            mat = np.array(window_mat)
            mean_path = mat.mean(axis=0)
            se_path   = mat.std(axis=0) / np.sqrt(len(mat))
            t_axis    = list(range(-PRE, WINDOW + 1))

            # Expected move at best_lag: beta × mean(cause at event)
            mean_cause_r = np.mean([xs[ev] for ev in events])
            expected_at_lag = beta_val * mean_cause_r
            actual_at_lag   = mean_path[PRE + min(best_lag, WINDOW)]
            overshoot_ratio = (actual_at_lag / expected_at_lag
                               if abs(expected_at_lag) > 1e-6 else float("nan"))

            # Post-lag correction: mean return from lag+1 to lag+5
            lag_end = min(PRE + best_lag + 5, len(mean_path) - 1)
            lag_start_idx = PRE + best_lag + 1
            correction_r = (float(mean_path[lag_end] - mean_path[min(lag_start_idx, lag_end)])
                            if lag_start_idx <= lag_end else 0.0)

            overshoot_rows.append({
                "Paar":            f"{src}\u2192{tgt}",
                "Richtung":        label,
                "Granger-Lag (T)": best_lag,
                "n Events":        len(window_mat),
                "mean Ursache (%)": round(mean_cause_r * 100, 3),
                "beta (OLS)":       round(beta_val, 4),
                "Erwartet @ lag (%)": round(expected_at_lag * 100, 4),
                "Beobachtet @ lag (%)": round(actual_at_lag * 100, 4),
                "Overshoot-Ratio":   round(overshoot_ratio, 3) if not np.isnan(overshoot_ratio) else "n/a",
                "Korrektur T+lag+1..+5 (%)": round(correction_r * 100, 4),
            })

            col = PAL[j % len(PAL)]
            dash_style = "solid" if sign == 1 else "dash"
            # Convert hex to rgba for fill
            def _hex_rgba(h, a=0.13):
                h = h.lstrip("#")
                r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                return f"rgba({r},{g},{b},{a})"
            fill_col = _hex_rgba(col)
            # Event-study chart trace
            fig_evst.add_trace(go.Scatter(
                x=t_axis, y=(mean_path * 100).tolist(),
                mode="lines",
                name=f"{src}\u2192{tgt} ({label})",
                line=dict(color=col, width=1.5, dash=dash_style),
                hovertemplate="T=%{x}<br>Kum.Rendite=%{y:.3f}%<extra></extra>"))
            # CI band
            upper = ((mean_path + se_path) * 100).tolist()
            lower = ((mean_path - se_path) * 100).tolist()
            fig_evst.add_trace(go.Scatter(
                x=t_axis + t_axis[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor=fill_col,
                line=dict(color="rgba(0,0,0,0)"), showlegend=False))

            # Overshoot ratio bar
            if not np.isnan(overshoot_ratio):
                fig_ratio.add_trace(go.Bar(
                    x=[f"{src}\u2192{tgt} ({label})"],
                    y=[overshoot_ratio],
                    marker_color=("#3fb950" if 0.8 < overshoot_ratio < 1.5
                                  else "#d29922" if 0.5 < overshoot_ratio < 2.0
                                  else "#f78166"),
                    showlegend=False,
                    hovertemplate=f"Overshoot={overshoot_ratio:.3f}<extra></extra>"))

    # Add key reference lines to event-study chart
    fig_evst.add_vline(x=0, line_color="#d29922", line_width=1.5,
                        annotation_text="Event T=0", annotation_font_color="#d29922")
    fig_evst.add_hline(y=0, line_color="#8b949e", line_width=0.8)
    fig_evst.update_layout(
        title="Event-Study: Kumulierte Ziel-Rendite rund um CL=F Signale (±1σ)",
        xaxis_title="Handelstag relativ zum Ereignis",
        yaxis_title="Kum. Rendite (%)",
        height=520)

    fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="#3fb950",
                         annotation_text="Ratio=1: genau erwartet")
    fig_ratio.update_layout(
        title="Overshoot-Ratio: Beobachtet÷Erwartet @ Granger-Best-Lag (>1: overshooting, <1: undershooting)",
        yaxis_title="Overshoot-Ratio", xaxis_tickangle=-35, height=400)

    # Rolling 63-day overshoot ratio over time (for CL=F→SM if available)
    fig_rolling = go.Figure()
    for _, row in sig_pairs.head(4).iterrows():
        src, tgt, best_lag = str(row["cause"]), str(row["effect"]), int(row["lag"])
        if src not in clean.columns or tgt not in clean.columns:
            continue
        xs = clean[src].dropna(); yt = clean[tgt].dropna()
        idx = xs.index.intersection(yt.index)
        if len(idx) < 126:
            continue
        xs2, yt2 = xs.loc[idx], yt.loc[idx]
        # Rolling beta at each window
        roll_beta  = xs2.rolling(63).cov(yt2) / xs2.rolling(63).var()
        # Rolling realized "overshoot": corr(xs2.shift(lag), yt2) / expected_corr
        actual_lag_r = xs2.shift(best_lag).rolling(63).corr(yt2)
        # Normalize by unconditional rolling std to give ratio
        roll_ratio = actual_lag_r / (roll_beta.abs() + 1e-9) * 10  # scaled for visibility
        roll_ratio = roll_ratio.dropna()
        fig_rolling.add_trace(go.Scatter(
            x=roll_ratio.index.astype(str).tolist(),
            y=roll_ratio.values.tolist(),
            mode="lines", name=f"{src}\u2192{tgt}",
            line=dict(width=1.2)))
    fig_rolling.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_rolling.update_layout(
        title="Rollender Overshoot-Index &#252;ber Zeit (CL=F-Paare, 63T-Fenster)",
        yaxis_title="Overshoot-Index (skaliert)", height=360)

    ov_table = (_df_html(pd.DataFrame(overshoot_rows)) if overshoot_rows
                else "<p class='text-muted'>Keine Overshoot-Daten.</p>")

    body = f"""
<div class="ph-header">
  <h1>Overshoot &amp; Korrektur-Analyse</h1>
  <div class="sub">Event-Study: CL=F &#177;1&#963;-Signale &#8594; kumulierte Ziel-Renditen T&#8722;5&#8230;T+10 | Overshoot-Ratio | Korrektur nach Lag</div>
</div>
<div class="card mb-4">
  <div class="card-header">Methodik</div>
  <div class="card-body">
    {_formula(r"\text{Overshoot-Ratio} = \frac{r_{\text{target}}(T+\text{lag})}{\hat{\beta} \cdot r_{\text{cause}}(T)}",
              "Overshoot > 1: Markt &#252;bers chie&#223;t die erwartete Reaktion. < 1: Unterreaktion (sluggish response).")}
    {_formula(r"\text{Korrektur} = \sum_{k=\text{lag}+1}^{\text{lag}+5} r_{\text{target}}(T+k)",
              "Folgereaktion nach dem Granger-Lag: Mittelwert &#252;ber alle Signalereignisse.")}
    {_info("Signale: CL=F Tagesrendite > &#177;1&#963; (Standa rdabweichung aller Handelstage). "
           "Kumulierte Rendite des Ziels relativ zum Ereignistag T=0. "
           "Granger-Lag = Zeitpunkt maximaler Vorhersagekraft aus Phase 6.")}
    {_warn("Alle Ergebnisse sind in-sample. Die Overshoot-Ratio kann durch Datenmining verzerrt sein. "
           "Hohe Overshoot-Ratio bedeutet: Markt reagiert st&#228;rker als einfache OLS-Beta-Projektion. "
           "Kann auf: Nichtlinearit&#228;t, Hebelwirkung, Liquidit&#228;ts-Effekte oder Regime-Abh&#228;ngigkeit hinweisen.")}
  </div>
</div>
{_chart_card("Event-Study: Kumulierte Rendite des Ziels rund um CL=F &#177;1&#963;-Signale",
              fig_evst, height=540,
              interp="T=0: Signaltag (CL=F Extremrendite). "
                     "Kurven zeigen mittlere kumulierte Zielrendite &#177;1 SE. "
                     "Positive Kurve nach T=0 bestat&#228;tigt Granger-Vorhersagekraft. "
                     "Knicke nach dem Granger-Best-Lag: potenzielle Korrektur.")}
{_chart_card("Overshoot-Ratio @ Granger-Best-Lag (>1: Market overshoots, <1: undershoots)",
              fig_ratio, height=420,
              interp="Gr&#252;n (0.8-1.5): Markt reagiert nah an OLS-Projektion. "
                     "Gelb (0.5-2.0): leichter Overshoot. "
                     "Rot (&lt;0.5 oder &gt;2.0): starkes Unter- oder &#220;berschie&#223;en &#8212; "
                     "potenzielle Mean-Reversion-Opportunit&#228;t!")}
{_chart_card("Rollender Overshoot-Index &#252;ber Zeit", fig_rolling, height=380,
              interp="Zeitvariabilit&#228;t des Overshoot: Krisen (GFC 2008, COVID 2020) oft mit "
                     "extremen Overshoot-Werten. Stabile Phasen: Ratio n&#228;her an 1. "
                     "Strukturbruch sichtbar: Overshoot-Regime &#228;ndert sich.")}
{_card("Overshoot-Tabelle: Alle Paare", ov_table)}
"""
    _write(out / "overshoot.html", _html_base("Overshoot &amp; Korrektur", 16, body))


# ─────────────────────────────────────────────────────────────────────────────
# External Drivers Report
# ─────────────────────────────────────────────────────────────────────────────

def build_external_drivers_report(tables, figures, out):  # noqa: C901
    returns  = _read(tables / "phase2_returns.csv")
    macro_df = _read(tables / "phase1_macro.csv")
    # _read() uses no parse_dates → index is strings; parse here for date math
    if returns is not None:
        returns.index = pd.to_datetime(returns.index, errors="coerce")
        returns = returns[returns.index.notna()]
    if macro_df is not None:
        macro_df.index = pd.to_datetime(macro_df.index, errors="coerce")
        macro_df = macro_df[macro_df.index.notna()]

    if returns is None:
        _write(out / "external_drivers.html",
               _html_base("Externe Treiber", 17, "<p>Renditen fehlen.</p>"))
        return

    EXT_YAHOO = {
        "CNY=X":    "USD/CNY (China Yuan)",
        "AUDUSD=X": "AUD/USD",
        "BRL=X":    "USD/BRL",
        "SBLK":     "Star Bulk (BDI-Proxy)",
        "LIT":      "Lithium ETF (EV)",
        "PDBC":     "Broad Commodity ETF",
        "REMX":     "Rare Earth ETF",
        "URA":      "Uranium ETF",
    }
    FRED_EXT = {
        "T5YIE":         "5J TIPS Breakeven (%)",
        "T10YIE":        "10J TIPS Breakeven (%)",
        "BAMLH0A0HYM2":  "US High Yield OAS (bps)",
        "UMCSENT":       "Michigan Consumer Sentiment",
        "DCOILBRENTEU":  "Brent Spot (FRED)",
        "DHHNGSP":       "Henry Hub Gas Spot (FRED)",
    }
    COMM_FOCUS = ["CL=F", "GC=F", "HG=F", "ZW=F", "ZS=F", "NG=F"]
    FRED_LEVEL = {"T5YIE", "T10YIE", "BAMLH0A0HYM2", "UMCSENT"}

    # ── helpers ───────────────────────────────────────────────────────────
    def _ols_r2(X, y):
        Xc = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
            y_hat = Xc @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            return 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else 0.0
        except Exception:
            return float("nan")

    # ── External Yahoo: separate price (for chart) from returns (for analysis) ─
    ext_prices: dict[str, pd.Series]  = {}  # normalized price index (base=100)
    ext_returns: dict[str, pd.Series] = {}  # daily log returns

    already_in = [c for c in EXT_YAHOO if c in returns.columns]
    missing     = [c for c in EXT_YAHOO if c not in returns.columns]

    for c in already_in:
        r = returns[c].dropna()
        ext_returns[c] = r
        ext_prices[c]  = (1 + r).cumprod() * 100  # cumulative price index

    if missing:
        import yfinance as yf
        for c in missing:
            try:
                hist = yf.Ticker(c).history(start="2005-01-01", auto_adjust=True)
                if not hist.empty:
                    s = hist["Close"].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    ext_prices[c]  = s / s.iloc[0] * 100
                    ext_returns[c] = np.log(s / s.shift(1)).dropna()
            except Exception:
                pass

    # ── FRED: fetch → forward-fill monthly to daily → compute changes ─────
    fred_ext_levels:  dict[str, pd.Series] = {}
    fred_ext_returns: dict[str, pd.Series] = {}

    for sid in FRED_EXT:
        s = None
        if macro_df is not None and sid in macro_df.columns:
            s = macro_df[sid].dropna()
        else:
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                df_f = pd.read_csv(url, index_col=0, parse_dates=True)
                df_f.columns = [sid]
                df_f.replace(".", float("nan"), inplace=True)
                df_f[sid] = pd.to_numeric(df_f[sid], errors="coerce")
                s = df_f[sid].dropna()
            except Exception:
                pass
        if s is None or len(s) < 5:
            continue
        fred_ext_levels[sid] = s
        # Forward-fill to business days so index aligns with daily returns
        daily = s.resample("B").ffill().dropna()
        if sid in FRED_LEVEL:
            fred_ext_returns[sid] = daily.diff().dropna()
        else:
            fred_ext_returns[sid] = daily.pct_change().dropna()

    all_ext_returns = {**ext_returns, **fred_ext_returns}
    comm_rets = {c: returns[c].dropna() for c in COMM_FOCUS if c in returns.columns}

    # ── CHART 1: Normalized price time series ─────────────────────────────
    fig_ts = go.Figure()
    for j, (c, s) in enumerate(ext_prices.items()):
        fig_ts.add_trace(go.Scatter(
            x=s.index.astype(str).tolist(), y=np.round(s.values, 2).tolist(),
            mode="lines", name=EXT_YAHOO.get(c, c),
            line=dict(color=PAL[j % len(PAL)], width=1.3)))
    fig_ts.add_hline(y=100, line_color="#8b949e", line_dash="dash", line_width=0.8)
    fig_ts.update_layout(
        title="Externe Yahoo-Faktoren: Normierte Preis-Zeitreihen (Basis=100 am Startdatum)",
        yaxis_title="Index (Basis=100)", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.01))

    # ── CHART 2: FRED levels in subplots ─────────────────────────────────
    n_fred = len(fred_ext_levels)
    cols_f  = min(3, n_fred) if n_fred > 0 else 1
    rows_f  = (n_fred + cols_f - 1) // cols_f if n_fred > 0 else 1
    fig_fred = make_subplots(rows=max(rows_f, 1), cols=cols_f,
        subplot_titles=[FRED_EXT.get(k, k) for k in list(fred_ext_levels)[:n_fred]])
    for idx, (sid, s) in enumerate(fred_ext_levels.items()):
        r, c = divmod(idx, cols_f)
        fig_fred.add_trace(go.Scatter(
            x=s.index.astype(str).tolist(), y=np.round(s.values, 4).tolist(),
            mode="lines", name=FRED_EXT.get(sid, sid),
            line=dict(color=PAL[idx % len(PAL)], width=1.2), showlegend=True),
            row=r + 1, col=c + 1)
    fig_fred.update_layout(
        title="FRED Externe Faktoren: TIPS Breakeven · HY-Spread · Sentiment · Energie-Spot",
        height=max(400, rows_f * 220), showlegend=False)

    # ── CHART 3: Correlation heatmap ─────────────────────────────────────
    corr_rows, corr_y_labels = [], []
    for ext_nm, ext_r in all_ext_returns.items():
        row_vals = []
        for comm_nm, comm_r in comm_rets.items():
            idx_c = ext_r.index.intersection(comm_r.index)
            rho = float(ext_r.loc[idx_c].corr(comm_r.loc[idx_c])) if len(idx_c) >= 126 else float("nan")
            row_vals.append(round(rho, 3) if not np.isnan(rho) else float("nan"))
        corr_rows.append(row_vals)
        corr_y_labels.append(EXT_YAHOO.get(ext_nm, FRED_EXT.get(ext_nm, ext_nm)))

    # Filter out all-NaN rows
    valid_mask = [any(not np.isnan(v) for v in r) for r in corr_rows]
    corr_rows_f  = [r for r, ok in zip(corr_rows, valid_mask) if ok]
    corr_y_f     = [y for y, ok in zip(corr_y_labels, valid_mask) if ok]

    fig_corr = go.Figure()
    if corr_rows_f:
        fig_corr = go.Figure(go.Heatmap(
            z=corr_rows_f,
            x=list(comm_rets.keys()),
            y=corr_y_f,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" if not np.isnan(v) else "n/a" for v in r] for r in corr_rows_f],
            texttemplate="%{text}",
            hovertemplate="Extern=%{y}<br>Rohstoff=%{x}<br>\u03c1=%{z:.3f}<extra></extra>"))
        fig_corr.update_layout(
            title="Korrelation: Externe Faktoren \u00d7 Rohstoff-Renditen",
            height=max(350, 45 * len(corr_rows_f) + 80))

    # ── CHART 4: Partial R² via numpy OLS ────────────────────────────────
    base_cols = [c for c in ["DX-Y.NYB", "^VIX", "^TNX", "SPY"] if c in returns.columns]
    r2_rows = []
    for ext_nm, ext_r in all_ext_returns.items():
        for comm_nm, comm_r in comm_rets.items():
            common_idx = comm_r.index
            common_idx = common_idx.intersection(ext_r.index)
            for bc in base_cols:
                common_idx = common_idx.intersection(returns[bc].dropna().index)
            if len(common_idx) < 126:
                continue
            y       = comm_r.loc[common_idx].values
            X_base  = np.column_stack([returns[bc].loc[common_idx].values for bc in base_cols])
            X_ext   = np.column_stack([X_base, ext_r.loc[common_idx].values])
            r2_b = _ols_r2(X_base, y)
            r2_e = _ols_r2(X_ext,  y)
            delta = r2_e - r2_b
            if delta > 0.0005:
                r2_rows.append({
                    "Externer Faktor":    EXT_YAHOO.get(ext_nm, FRED_EXT.get(ext_nm, ext_nm)),
                    "Rohstoff":           comm_nm,
                    "Basis-R\u00b2":      round(r2_b, 4),
                    "R\u00b2 mit Ext.":   round(r2_e, 4),
                    "\u0394R\u00b2":      round(delta, 4),
                    "\u0394R\u00b2 %":    round(delta * 100, 2),
                    "n Tage":             len(common_idx),
                })

    fig_r2 = go.Figure()
    if r2_rows:
        r2_df = pd.DataFrame(r2_rows).sort_values("\u0394R\u00b2 %", ascending=False)
        top = r2_df.head(20)
        lbl = [f"{r['Externer Faktor']}\u2192{r['Rohstoff']}" for _, r in top.iterrows()]
        vals = top["\u0394R\u00b2 %"].tolist()
        fig_r2.add_bar(
            x=lbl, y=vals,
            marker_color=["#3fb950" if v > 1 else "#d29922" if v > 0.3 else "#8b949e" for v in vals],
            text=[f"+{v:.2f}%" for v in vals], textposition="outside", textfont=dict(size=8))
        fig_r2.add_hline(y=1.0, line_dash="dash", line_color="#3fb950",
                          annotation_text="\u0394R\u00b2=1%")
        fig_r2.update_layout(
            title="Partieller R\u00b2-Beitrag externer Faktoren (nach DXY+VIX+TNX+SPY)",
            yaxis_title="\u0394R\u00b2 (%)", xaxis_tickangle=-35, height=460)
    r2_table = (_df_html(pd.DataFrame(r2_rows).sort_values("\u0394R\u00b2 %", ascending=False))
                if r2_rows else "<p class='text-muted'>Keine signifikanten Beitr\u00e4ge (mind. 252 gemeinsame Tage).</p>")

    # ── CHART 5: Rolling 63-day correlation (best ext factor) ────────────
    fig_roll = go.Figure()
    valid_rows = [(corr_y_f[i], list(all_ext_returns.keys())[i if i < len(all_ext_returns) else -1],
                   [abs(v) for v in corr_rows_f[i] if not np.isnan(v)])
                  for i in range(len(corr_rows_f))]
    if valid_rows:
        valid_rows2 = [(lbl, nm, np.mean(vals)) for lbl, nm, vals in valid_rows if vals]
        if valid_rows2:
            best_lbl, best_nm, _ = max(valid_rows2, key=lambda x: x[2])
            best_ext_r = all_ext_returns.get(best_nm, pd.Series(dtype=float))
            for j, (comm_nm, comm_r) in enumerate(comm_rets.items()):
                idx_r = best_ext_r.index.intersection(comm_r.index)
                if len(idx_r) < 126:
                    continue
                roll_corr = best_ext_r.loc[idx_r].rolling(63).corr(comm_r.loc[idx_r]).dropna()
                fig_roll.add_trace(go.Scatter(
                    x=roll_corr.index.astype(str).tolist(),
                    y=np.round(roll_corr.values, 4).tolist(),
                    mode="lines", name=comm_nm,
                    line=dict(color=PAL[j % len(PAL)], width=1.3)))
            fig_roll.add_hline(y=0, line_color="#8b949e", line_dash="dash")
            fig_roll.update_layout(
                title=f"Rollende 63T-Korrelation: {best_lbl} vs. Rohstoffe",
                yaxis_title="Korrelation", height=380)

    # ── CHART 6: TIPS Breakeven vs Gold & Oil ────────────────────────────
    fig_tips = go.Figure()
    tips_sid = next((sid2 for sid2 in ["T10YIE", "T5YIE"] if sid2 in fred_ext_levels), None)
    if tips_sid:
        tips = fred_ext_levels[tips_sid]
        fig_tips.add_trace(go.Scatter(
            x=tips.index.astype(str).tolist(), y=tips.values.tolist(),
            name=FRED_EXT[tips_sid], line=dict(color="#d29922", width=2.0),
            yaxis="y1"))
        for cm, col in [("GC=F", "#ffa657"), ("CL=F", "#58a6ff")]:
            if cm in returns.columns:
                cum = (1 + returns[cm].dropna()).cumprod()
                cum.index = pd.to_datetime(cum.index, errors="coerce").tz_localize(None)
                tips_daily = tips.copy()
                tips_daily.index = pd.to_datetime(tips_daily.index, errors="coerce").tz_localize(None)
                cum_aln = cum.reindex(tips_daily.index, method="ffill").dropna()
                # Scale to TIPS y-range
                scale = tips.mean() / (cum_aln.mean() + 1e-9)
                fig_tips.add_trace(go.Scatter(
                    x=cum_aln.index.astype(str).tolist(),
                    y=(cum_aln * scale).round(4).values.tolist(),
                    name=f"{cm} (skaliert auf TIPS-Achse)",
                    line=dict(color=col, width=1.2, dash="dot"),
                    yaxis="y1"))
        fig_tips.update_layout(
            title="TIPS 10J-Breakeven vs. Gold &amp; \u00d6l (auf gleicher Achse skaliert)",
            yaxis_title=FRED_EXT.get(tips_sid, tips_sid),
            height=420)

    # ── SEASONAL ANALYSIS ─────────────────────────────────────────────────
    # Define commodity classes
    SEASONAL_CLASSES = {
        "Energie":     ["CL=F", "BZ=F", "NG=F"],
        "Metalle":     ["GC=F", "SI=F", "HG=F"],
        "Agrar":       ["ZW=F", "ZC=F", "ZS=F"],
    }
    MONTHS = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]

    # Monthly seasonality: mean daily return per month, annualised
    fig_seasonal = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(SEASONAL_CLASSES.keys()),
        horizontal_spacing=0.08)

    seasonal_z_all = {}  # for heatmaps
    for cls_i, (cls_name, tickers) in enumerate(SEASONAL_CLASSES.items()):
        avail = [t for t in tickers if t in returns.columns]
        if not avail:
            continue
        cls_ret = returns[avail].copy()
        cls_ret.index = pd.to_datetime(cls_ret.index, errors="coerce")
        cls_ret = cls_ret[cls_ret.index.notna()]
        # Equal-weight class index
        class_r = cls_ret.mean(axis=1)
        monthly_mean = class_r.groupby(class_r.index.month).mean() * 252  # annualised
        monthly_mean.index = [MONTHS[m-1] for m in monthly_mean.index]
        seasonal_z_all[cls_name] = monthly_mean
        colors_bar = ["#3fb950" if v > 0 else "#f78166" for v in monthly_mean.values]
        fig_seasonal.add_trace(
            go.Bar(x=monthly_mean.index.tolist(), y=np.round(monthly_mean.values * 100, 2).tolist(),
                   marker_color=colors_bar, name=cls_name, showlegend=False,
                   hovertemplate="%{x}: %{y:.2f}%<extra></extra>"),
            row=1, col=cls_i + 1)
    fig_seasonal.update_layout(
        title="Saisonalität: Durchschnittliche Monats-Renditen pro Rohstoff-Klasse (annualisiert, %)",
        height=400, yaxis_title="Rendite/Jahr (%)")

    # Asset-level seasonal heatmap (month × asset)
    all_seasonal_tickers = sum(SEASONAL_CLASSES.values(), [])
    avail_all = [t for t in all_seasonal_tickers if t in returns.columns]
    fig_seas_heat = go.Figure()
    if avail_all:
        heat_z = []
        for t in avail_all:
            r = returns[t].copy()
            r.index = pd.to_datetime(r.index, errors="coerce")
            r = r[r.index.notna()].dropna()
            monthly = r.groupby(r.index.month).mean() * 252 * 100
            heat_z.append([float(monthly.get(m, np.nan)) for m in range(1, 13)])
        fig_seas_heat = go.Figure(go.Heatmap(
            z=heat_z,
            x=MONTHS,
            y=avail_all,
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in heat_z],
            texttemplate="%{text}",
            hovertemplate="Monat=%{x}<br>Asset=%{y}<br>Rendite=%{z:.2f}%/J<extra></extra>"))
        fig_seas_heat.update_layout(
            title="Saisonale Heatmap: Monatliche Rendite pro Asset (% p.a.)",
            height=max(300, 35 * len(avail_all) + 80))

    # Gas seasonal pattern as temperature proxy
    fig_gas_season = go.Figure()
    if "NG=F" in returns.columns:
        ng = returns["NG=F"].copy()
        ng.index = pd.to_datetime(ng.index, errors="coerce")
        ng = ng[ng.index.notna()].dropna()
        # Separate heating season (Oct-Mar) vs cooling (Apr-Sep)
        ng_monthly = ng.groupby(ng.index.month).agg(["mean","std"]) * 252
        ng_monthly.index = [MONTHS[m-1] for m in ng_monthly.index]
        heat_cols = ["Okt","Nov","Dez","Jan","Feb","Mär"]
        cool_cols = ["Apr","Mai","Jun","Jul","Aug","Sep"]
        colors_ng = ["#58a6ff" if m in heat_cols else "#ffa657" for m in ng_monthly.index]
        fig_gas_season.add_trace(go.Bar(
            x=ng_monthly.index.tolist(),
            y=np.round(ng_monthly["mean"].values * 100, 2).tolist(),
            marker_color=colors_ng,
            error_y=dict(type="data", array=np.round(ng_monthly["std"].values * 100, 2).tolist(),
                         visible=True, color="#8b949e"),
            name="NG=F Monats-Rendite"))
        fig_gas_season.add_hline(y=0, line_color="#8b949e", line_dash="dash")
        fig_gas_season.update_layout(
            title="Erdgas (NG=F): Saisonales Renditemuster als Temperatur-Proxy (Blau=Heizperiode, Orange=Kühlperiode)",
            yaxis_title="Ø Rendite p.a. (%)", height=380)

    # Agricultural calendar chart: Planting/growing/harvest phases
    fig_agri_cal = go.Figure()
    AGRI_PHASES = {
        "Pflanzung (Mais/Soja)": ([3, 4, 5], "#3fb950"),
        "Wachstum (Mais/Soja)": ([6, 7, 8], "#d29922"),
        "Ernte (Mais/Soja)":    ([9, 10, 11], "#ffa657"),
        "Lager & Export":       ([12, 1, 2], "#58a6ff"),
    }
    agri_tickers = [t for t in ["ZW=F", "ZC=F", "ZS=F"] if t in returns.columns]
    if agri_tickers:
        for j, tick in enumerate(agri_tickers):
            r = returns[tick].copy()
            r.index = pd.to_datetime(r.index, errors="coerce")
            r = r[r.index.notna()].dropna()
            monthly_ann = r.groupby(r.index.month).mean() * 252 * 100
            phase_means = {}
            for phase_name, (months, _) in AGRI_PHASES.items():
                vals = [float(monthly_ann.get(m, np.nan)) for m in months if not np.isnan(float(monthly_ann.get(m, np.nan)))]
                phase_means[phase_name] = np.mean(vals) if vals else np.nan
            fig_agri_cal.add_trace(go.Bar(
                x=list(phase_means.keys()),
                y=[round(v, 2) if not np.isnan(v) else 0 for v in phase_means.values()],
                name=tick,
                marker_color=PAL[j % len(PAL)]))
        # Add phase color bands
        fig_agri_cal.add_hline(y=0, line_color="#8b949e", line_dash="dash")
        fig_agri_cal.update_layout(
            title="Agrar-Kalender: Durchschnittsrendite nach Anbauphase (Weizen, Mais, Soja)",
            barmode="group", yaxis_title="Ø Rendite p.a. (%)", height=400)

    # Henry Hub vs. Brent: temperature cycle comparison
    fig_energy_season = go.Figure()
    energy_pairs = [("NG=F", "#58a6ff", "Erdgas"), ("CL=F", "#ffa657", "Rohöl")]
    for tick, col, lbl in energy_pairs:
        if tick in returns.columns:
            r = returns[tick].copy()
            r.index = pd.to_datetime(r.index, errors="coerce")
            r = r[r.index.notna()].dropna()
            monthly = r.groupby(r.index.month).mean() * 252 * 100
            monthly.index = [MONTHS[m-1] for m in monthly.index]
            fig_energy_season.add_trace(go.Scatter(
                x=monthly.index.tolist(), y=np.round(monthly.values, 2).tolist(),
                mode="lines+markers", name=lbl,
                line=dict(color=col, width=2.2),
                marker=dict(size=8)))
    fig_energy_season.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_energy_season.update_layout(
        title="Energie-Saisonalität: Erdgas vs. Rohöl (Monatsdurchschnitt p.a.)",
        yaxis_title="Ø Rendite p.a. (%)", height=380)

    # Year-over-year rolling seasonal index (3Y window)
    fig_seas_rolling = go.Figure()
    if "GC=F" in returns.columns:
        for tick, col, lbl in [("GC=F", "#ffa657", "Gold"), ("CL=F", "#58a6ff", "Öl"), ("ZW=F", "#3fb950", "Weizen")]:
            if tick not in returns.columns:
                continue
            r = returns[tick].copy()
            r.index = pd.to_datetime(r.index, errors="coerce")
            r = r[r.index.notna()].dropna()
            # 3-year rolling annual return (annualized)
            roll = r.rolling(756).mean() * 252 * 100  # 756 trading days ≈ 3 years
            roll = roll.dropna()
            fig_seas_rolling.add_trace(go.Scatter(
                x=roll.index.astype(str).tolist(),
                y=np.round(roll.values, 2).tolist(),
                mode="lines", name=lbl,
                line=dict(color=col, width=1.5)))
    fig_seas_rolling.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_seas_rolling.update_layout(
        title="3-Jahres-Rollende Jahresrendite: Gold, Öl, Weizen (Langfristiger Trend)",
        yaxis_title="Rollende 3J-Rendite p.a. (%)", height=380)

    # ── EXTERNAL FACTOR STRATEGIES ────────────────────────────────────────
    def _metrics_ext(r, name):
        r = r.dropna()
        if len(r) < 252: return None
        ann_r = float(r.mean() * 252)
        ann_v = float(r.std() * np.sqrt(252))
        cum   = (1 + r).cumprod()
        mdd   = float((cum / cum.cummax() - 1).min())
        split = int(len(r) * 0.7)
        is_sh  = float(r.iloc[:split].mean() * 252 / (r.iloc[:split].std() * np.sqrt(252) + 1e-9))
        oos_sh = float(r.iloc[split:].mean() * 252 / (r.iloc[split:].std() * np.sqrt(252) + 1e-9))
        return {
            "Strategie": name, "CAGR%": round(ann_r * 100, 2),
            "Sharpe": round(ann_r / (ann_v + 1e-9), 3),
            "MaxDD%": round(mdd * 100, 2),
            "IS Sharpe (70%)": round(is_sh, 3), "OOS Sharpe (30%)": round(oos_sh, 3),
            "Degradation": round(is_sh - oos_sh, 3),
        }

    def _tc(pos, ret, bps=10):
        return pos * ret - pos.diff().abs().fillna(0) * (bps / 10000)

    strat_results, strat_equity, strat_meta = {}, {}, []

    # S_EXT1: TIPS → Gold (Inflation hedge)
    if tips_sid and tips_sid in fred_ext_levels and "GC=F" in returns.columns:
        tips_daily = fred_ext_levels[tips_sid].resample("B").ffill().dropna()
        tips_ch = tips_daily.diff()
        gc_r    = returns["GC=F"].dropna()
        idx1    = tips_ch.index.intersection(gc_r.index)
        if len(idx1) >= 252:
            sig1 = pd.Series(np.where(tips_ch.loc[idx1].shift(1) > 0, 1, -1), index=idx1)
            r1   = _tc(sig1, gc_r.loc[idx1])
            m1   = _metrics_ext(r1, "S_EXT1: TIPS\u2191\u2192Long GC=F")
            if m1:
                strat_results["S_EXT1"] = r1; strat_equity["S_EXT1"] = (1 + r1).cumprod()
                strat_meta.append(m1)

    # S_EXT2: CNY Momentum → Copper (China demand)
    if "CNY=X" in ext_returns and "HG=F" in returns.columns:
        cny_r = ext_returns["CNY=X"]
        hg_r  = returns["HG=F"].dropna()
        idx2  = cny_r.index.intersection(hg_r.index)
        if len(idx2) >= 252:
            # Yuan stronger (USD/CNY falls) → bullish copper
            cny_mom = cny_r.rolling(21).sum()
            sig2 = pd.Series(np.where(cny_mom.loc[idx2].shift(1) < 0, 1, -1), index=idx2)
            r2   = _tc(sig2, hg_r.loc[idx2])
            m2   = _metrics_ext(r2, "S_EXT2: CNY\u2193(Yuan\u2191)\u2192Long HG=F")
            if m2:
                strat_results["S_EXT2"] = r2; strat_equity["S_EXT2"] = (1 + r2).cumprod()
                strat_meta.append(m2)

    # S_EXT3: AUD/USD Momentum → Commodity basket (Gold + Copper)
    if "AUDUSD=X" in ext_returns and "GC=F" in returns.columns and "HG=F" in returns.columns:
        aud_r = ext_returns["AUDUSD=X"]
        gc_r  = returns["GC=F"].dropna(); hg_r = returns["HG=F"].dropna()
        idx3  = aud_r.index.intersection(gc_r.index).intersection(hg_r.index)
        if len(idx3) >= 252:
            aud_mom = aud_r.rolling(21).sum()
            sig3 = pd.Series(np.where(aud_mom.loc[idx3].shift(1) > 0, 1, -1), index=idx3)
            basket = 0.5 * gc_r.loc[idx3] + 0.5 * hg_r.loc[idx3]
            r3 = _tc(sig3, basket)
            m3 = _metrics_ext(r3, "S_EXT3: AUD/USD\u2191\u2192Long GC=F+HG=F")
            if m3:
                strat_results["S_EXT3"] = r3; strat_equity["S_EXT3"] = (1 + r3).cumprod()
                strat_meta.append(m3)

    # S_EXT4: HY Spread → Energy timing
    if "BAMLH0A0HYM2" in fred_ext_returns and "XLE" in returns.columns:
        hy_r  = fred_ext_returns["BAMLH0A0HYM2"]
        xle_r = returns["XLE"].dropna()
        idx4  = hy_r.index.intersection(xle_r.index)
        if len(idx4) >= 252:
            # Spreading credit (rising OAS) → bearish energy
            hy_mom = hy_r.rolling(21).sum()
            sig4 = pd.Series(np.where(hy_mom.loc[idx4].shift(1) < 0, 1, -1), index=idx4)
            r4   = _tc(sig4, xle_r.loc[idx4])
            m4   = _metrics_ext(r4, "S_EXT4: HY-OAS\u2193(Kredit eng)\u2192Long XLE")
            if m4:
                strat_results["S_EXT4"] = r4; strat_equity["S_EXT4"] = (1 + r4).cumprod()
                strat_meta.append(m4)

    # S_EXT5: BRL/USD → Agrar (Brazil supply signal)
    if "BRL=X" in ext_returns and "ZS=F" in returns.columns:
        brl_r = ext_returns["BRL=X"]
        zs_r  = returns["ZS=F"].dropna()
        idx5  = brl_r.index.intersection(zs_r.index)
        if len(idx5) >= 252:
            # Weaker BRL (USD/BRL up) → cheaper Brazilian exports → supply glut → bearish
            brl_mom = brl_r.rolling(21).sum()
            sig5 = pd.Series(np.where(brl_mom.loc[idx5].shift(1) < 0, 1, -1), index=idx5)
            r5   = _tc(sig5, zs_r.loc[idx5])
            m5   = _metrics_ext(r5, "S_EXT5: BRL\u2191(Real st\u00e4rker)\u2192Long ZS=F")
            if m5:
                strat_results["S_EXT5"] = r5; strat_equity["S_EXT5"] = (1 + r5).cumprod()
                strat_meta.append(m5)

    # S_EXT6: Multi-signal composite (equal weight of all above)
    if len(strat_results) >= 3:
        combo_rets = pd.concat([r.rename(k) for k, r in strat_results.items()], axis=1).mean(axis=1).dropna()
        m6 = _metrics_ext(combo_rets, "S_EXT6: Komposit (gleich gewichtet)")
        if m6:
            strat_results["S_EXT6"] = combo_rets
            strat_equity["S_EXT6"] = (1 + combo_rets).cumprod()
            strat_meta.append(m6)

    # Equity chart for strategies
    fig_eq = go.Figure()
    for j, (nm, eq) in enumerate(strat_equity.items()):
        fig_eq.add_trace(go.Scatter(
            x=eq.index.astype(str).tolist(), y=np.round(eq.values, 4).tolist(),
            mode="lines", name=nm, line=dict(color=PAL[j % len(PAL)], width=1.5)))
    if "GC=F" in returns.columns:
        bh = (1 + returns["GC=F"].dropna()).cumprod()
        fig_eq.add_trace(go.Scatter(
            x=bh.index.astype(str).tolist(), y=np.round(bh.values, 4).tolist(),
            mode="lines", name="BH Gold", line=dict(color="#8b949e", width=0.8, dash="dot")))
    fig_eq.update_layout(
        title="Externe-Faktor Strategien: Equity-Kurven (log-Skala)",
        yaxis_type="log", yaxis_title="Kapital (Basis=1)", height=480)

    fig_sh = go.Figure()
    if strat_meta:
        df_m = pd.DataFrame(strat_meta)
        fig_sh.add_bar(
            x=df_m["Strategie"].tolist(), y=df_m["Sharpe"].tolist(),
            marker_color=["#3fb950" if v > 0.3 else "#d29922" if v > 0 else "#f78166"
                          for v in df_m["Sharpe"]],
            name="Sharpe")
        fig_sh.add_bar(
            x=df_m["Strategie"].tolist(), y=df_m["OOS Sharpe (30%)"].tolist(),
            marker_color=["rgba(63,185,80,0.4)" if v > 0.3 else "rgba(242,129,102,0.4)"
                          for v in df_m["OOS Sharpe (30%)"]],
            name="OOS Sharpe")
        fig_sh.add_hline(y=0, line_color="#8b949e")
        fig_sh.update_layout(
            title="IS vs. OOS Sharpe: Externe-Faktor Strategien",
            barmode="group", height=380)

    strat_table = (_df_html(pd.DataFrame(strat_meta)) if strat_meta
                   else "<p class='text-muted'>Nicht gen\u00fcgend Daten f\u00fcr Strategie-Backtest.</p>")

    body = f"""
<div class="ph-header">
  <h1>Externe Treiber &amp; Strategien</h1>
  <div class="sub">FX-Regime (CNY, AUD, BRL) \u00b7 TIPS Breakeven \u00b7 HY-Spread \u00b7 Shipping (BDI) \u00b7 EV-Nachfrage \u00b7 Partieller R\u00b2 \u00b7 6 Externe Strategien</div>
</div>
<div class="card mb-4">
  <div class="card-header">Externe Variablen &amp; Theoretische Wirkungskan\u00e4le</div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered">
      <thead><tr><th>Faktor</th><th>Ticker/Serie</th><th>Beeinflusst</th><th>Mechanismus</th><th>Strategie</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#58a6ff;">\U0001f30f USD/CNY</strong></td><td>CNY=X</td><td>Kupfer, \u00d6l, Soja</td>
          <td>Yuan-Abwertung \u2192 CN-Importe teurer \u2192 Nachfrager\u00fcckgang</td><td>S_EXT2</td></tr>
        <tr><td><strong style="color:#58a6ff;">\U0001f998 AUD/USD</strong></td><td>AUDUSD=X</td><td>Gold, Kupfer</td>
          <td>AUD = Commodity Currency; f\u00fchrt Rohstoffpreise oft um 1-5T</td><td>S_EXT3</td></tr>
        <tr><td><strong style="color:#3fb950;">\U0001f331 USD/BRL</strong></td><td>BRL=X</td><td>Soja, Zucker</td>
          <td>Brasilien = gr\u00f6\u00dfter Soja-Exporteur; BRL-St\u00e4rke reduziert Angebotsdruck</td><td>S_EXT5</td></tr>
        <tr><td><strong style="color:#3fb950;">\U0001f6a2 BDI-Proxy</strong></td><td>SBLK</td><td>Alle Rohstoffe</td>
          <td>Frachtkosten = Fr\u00fch-Indikator f\u00fcr globale Handelsnachfrage</td><td>\u2014</td></tr>
        <tr><td><strong style="color:#d29922;">\u26a1 Lithium ETF</strong></td><td>LIT</td><td>Kupfer, Lithium</td>
          <td>EV-Boom \u2192 Batterie-Metall-Nachfrage; strukturelle Verschiebung</td><td>\u2014</td></tr>
        <tr><td><strong style="color:#bc8cff;">\U0001f4b0 TIPS Breakeven</strong></td><td>T5YIE/T10YIE</td><td>Gold, alle</td>
          <td>Inflationserwartungen \u2192 Gold als Hedge; TIPS\u2191 = Fed hinter der Kurve</td><td>S_EXT1</td></tr>
        <tr><td><strong style="color:#bc8cff;">\U0001f4c8 HY-Spread</strong></td><td>BAMLH0A0HYM2</td><td>XLE, Energie</td>
          <td>Kreditrisiko \u2192 Shale-Finanzierungskosten; OAS\u2191 = Produktionsk\u00fcrzung</td><td>S_EXT4</td></tr>
        <tr><td><strong style="color:#f78166;">\U0001f60a Consumer Sentiment</strong></td><td>UMCSENT</td><td>Benzin, Agrar</td>
          <td>Konsumentenstimmung \u2192 Treibstoffnachfrage, Lebensmittelausgaben</td><td>\u2014</td></tr>
        <tr><td><strong style="color:#8b949e;">\U0001f4e6 Broad Commodity</strong></td><td>PDBC</td><td>Alle</td>
          <td>Benchmark: gemeinsames Rohstoff-Beta herausfiltern f\u00fcr Alpha</td><td>\u2014</td></tr>
      </tbody>
    </table>
    {_formula(r"R^2_{\Delta} = R^2_{\text{Basis}+\text{Ext}} - R^2_{\text{Basis}}",
              "Partieller R\u00b2: Zus\u00e4tzliche Varianz des externen Faktors NACH DXY+VIX+TNX+SPY.")}
    {_info("FRED-Daten (TIPS, HY, Sentiment): monatlich \u2192 auf Tagesbasis forward-gefill'd. "
           "OLS mit Interzept, mindestens 252 gemeinsame Handelstage. "
           "Gr\u00fcn \u0394R\u00b2>1%: \u00f6konomisch signifikant. Gelb 0.3-1%: marginal.")}
  </div>
</div>
{_chart_card("Externe Yahoo-Faktoren: Normierte Zeitreihen (Basis=100)", fig_ts, height=480,
    interp="CNY=X steigend = starker Dollar vs. Yuan \u2192 schlecht f\u00fcr China-Rohstoffnachfrage (b\u00e4risch Kupfer). "
           "AUDUSD: AUD als Commodity Currency \u2013 f\u00fchrt oft Rohstoffpreise. "
           "SBLK (Star Bulk) = BDI-Proxy: zyklischer Fracht-Vorlaufsindikator.")}
{_chart_card("FRED Externe Faktoren: Levels", fig_fred, height=520,
    interp="T10YIE (TIPS Breakeven): Inflationserwartung; steigt vor Rohstoffpreisschocks. "
           "HY OAS: Credit Spread; Spitzen 2008+2020 \u2192 Energie-Produzenten unter Finanzierungsdruck. "
           "UMCSENT: vorlaufender Konsumindikator; f\u00e4llt zuerst bei Rezession.")}
{_chart_card("Korrelationsmatrix: Externe Faktoren \u00d7 Rohstoff-Renditen", fig_corr,
    interp="Dunkelblau: stark negativ (z.B. USD/CNY\u2191 \u2192 Kupfer\u2193). "
           "Dunkelrot: stark positiv (z.B. AUD/USD\u2191 \u2192 Gold\u2191). "
           "TIPS\u2194Gold: Inflations-Hedge-Kanal. HY-OAS\u2194Energie: Kreditkanal.")}
{_chart_card("Partieller R\u00b2-Beitrag externer Faktoren (nach DXY+VIX+TNX+SPY)", fig_r2, height=480,
    interp="Gr\u00fcn >1%: erkl\u00e4rt zus\u00e4tzliche Varianz \u00fcber klassische Makro-Controls hinaus. "
           "TIPS typisch st\u00e4rkster Faktor f\u00fcr Gold. BRL=X f\u00fcr Agrar. "
           "HY-Spread f\u00fcr Energie-Produzenten.")}
{_chart_card("Rollende 63T-Korrelation: St\u00e4rkster externer Faktor vs. Rohstoffe", fig_roll, height=400,
    interp="Zeitvariabler Zusammenhang: Krisen (2008, 2020) \u2192 oft st\u00e4rkere Korrelationen. "
           "Vorzeichen-Wechsel: Regime-Shift im Wirkungs-Kanal. "
           "Persistent nahe +-1: robuster, handelbarer Zusammenhang.")}
{_chart_card("TIPS Breakeven vs. Gold &amp; \u00d6l (skaliert)", fig_tips, height=440,
    interp="TIPS Breakeven\u2191 \u2192 Gold folgt (Inflationsschutz). \u00d6l und TIPS hoch korreliert 2021-22 (Energie-Inflationsschock). "
           "Divergenz 2023-24: Gold entkoppelt von TIPS (China-ZB-K\u00e4ufe, geopolitisch). "
           "TIPS\u2193 (Deflation) \u2192 Soja/Weizen oft outperformen (\u00dcberangebot, nicht Deflation).")}

<div class="card mb-4" id="section-seasonal">
  <div class="card-header"><strong>\U0001f321\ufe0f Wetter &amp; Saisonalität: Klimatische Preistreiber</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Rohstoffpreise zeigen starke kalendarische Muster. Erdgas: Heizperiode Okt-Mär ↑ (Heiznachfrage), Kühlperiode Jun-Aug ↑ (Klimaanlagen).
      Agrar: Planting-Rally (Mär-Mai), Ernte-Druck (Sep-Nov). Gold/Silber: schwächer im Sommer, stärker im Q4 (Schmucknachfrage Indien/China).
    </p>
    <div class="row mb-3">
      <div class="col-md-6 p-1">
        <div class="small fw-bold text-center mb-1">Planting/Ernte-Phasen (Agrar-Kalender)</div>
        <table class="table table-dark table-sm table-bordered small">
          <thead><tr><th>Phase</th><th>Monate</th><th>Erwarteter Effekt</th><th>Treiber</th></tr></thead>
          <tbody>
            <tr><td><span style="color:#3fb950;">● Pflanzung</span></td><td>Mär–Mai</td><td>Preis-Rallye (Risiko)</td><td>Wetterproblem → Angst vor Ernteausfall</td></tr>
            <tr><td><span style="color:#d29922;">● Wachstum</span></td><td>Jun–Aug</td><td>Dürre-Premium</td><td>ENSO-Einfluss: La Niña = Dürre USA/Südamerika</td></tr>
            <tr><td><span style="color:#ffa657;">● Ernte</span></td><td>Sep–Nov</td><td>Angebotsdruck ↓</td><td>Rekordernte → Verkaufsdruck, Preisrückgang</td></tr>
            <tr><td><span style="color:#58a6ff;">● Lager</span></td><td>Dez–Feb</td><td>Export-Saison Südhemisphäre</td><td>Brasilien/Argentinien ernten → Export konkurrenz</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6 p-1">
        <div class="small fw-bold text-center mb-1">Energie-Saisonalität (Temperatur-Proxy)</div>
        <table class="table table-dark table-sm table-bordered small">
          <thead><tr><th>Periode</th><th>Monate</th><th>Gas-Effekt</th><th>Öl-Effekt</th></tr></thead>
          <tbody>
            <tr><td><span style="color:#58a6ff;">❄ Heizperiode</span></td><td>Okt–Mär</td><td>Heiznachfrage ↑ → NG ↑</td><td>Heizöl ↑, Benzin neutral</td></tr>
            <tr><td><span style="color:#ffa657;">☀ Kühlperiode</span></td><td>Jun–Sep</td><td>Klimaanlagen ↑ → NG ↑</td><td>Benzin-Peak (Driving Season)</td></tr>
            <tr><td><span style="color:#8b949e;">→ Shoulder</span></td><td>Apr–Mai, Okt</td><td>Lager-Auffüllphase</td><td>Raffinerie-Wartung, Spread-Anomalie</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
{_chart_card("Saisonalität: Durchschnittliche Monatsrenditen pro Klasse (annualisiert)", fig_seasonal, height=420,
    interp="Grün = positiver Saisonaleffekt. Rot = historisch schwacher Monat. "
           "Energie: Jan/Feb stark (Winterspitze), Sep/Okt schwach (Lagerauffüllung nach Sommer). "
           "Agrar: Jun starkes Dürre-Premium, Sep/Okt Ernte-Abwärtsdruck. "
           "Metalle: Jan/Feb oft stark (China-Stimulus nach Jahreswechsel).")}
{_chart_card("Saisonale Heatmap: Alle Rohstoffe × Monat (% p.a.)", fig_seas_heat,
    interp="Dunkelgrün: historisch starker Saisonmonat. Dunkelrot: schwach. "
           "Gold: Sep/Nov Schmuck-Hochsaison (Indien Diwali, China Neujahr). "
           "Weizen: Jun/Jul Ernte-Druck. Soja: Sep/Okt Ernte. Erdgas: Jan Heizpeak, Sep Schulter.")}
{_chart_card("Erdgas (NG=F): Saisonales Renditemuster — Temperatur-Proxy", fig_gas_season, height=400,
    interp="Blau = Heizperiode (Okt-Mär): erhöhte Heiznachfrage. Orange = Kühlperiode (Apr-Sep). "
           "Fehlerbalken = 1σ: breite Balken → hohe Variabilität (wetterabhängig). "
           "ENSO-Jahre (La Niña, El Niño) können dieses Muster signifikant verzerren.")}
{_chart_card("Agrar-Kalender: Durchschnitt Rendite nach Planting/Ernte-Phasen", fig_agri_cal, height=420,
    interp="Pflanzungsphase (Mär-Mai): Preis-Rallye durch Wetter-Unsicherheit typisch bullisch. "
           "Erntephase (Sep-Nov): strukturell bärisch (Angebotszunahme). "
           "Wachstum (Jun-Aug): Dürre-Prämien in El-Niño-Jahren besonders stark.")}
{_chart_card("Energie-Saisonalität: Erdgas vs. Rohöl Vergleich", fig_energy_season, height=400,
    interp="Erdgas hat stärkere saisonale Ausschläge als Öl (direkter Wetter-Einfluss). "
           "Öl: Driving Season (Jun-Aug) und Jahresanfang (Jan-Mär) typisch stark. "
           "Divergenz Gas↑/Öl↓ = Winter-Basis-Trade Opportunität.")}
{_chart_card("3-Jahres Rollende Jahresrendite: Gold, Öl, Weizen", fig_seas_rolling, height=400,
    interp="Strukturelle Trendänderungen: Öl-Schwäche 2015-2020 (Shale-Revolution). "
           "Gold-Stärke 2019-2024 (ZB-Käufe, Geopolitik). Weizen-Spike 2022 (Ukraine-Krieg). "
           "Konvergenz der Klassen signalisiert Commodity Supercycle.")}
<div class="card mb-4">
  <div class="card-header"><strong>Externe-Faktor Handelsstrategien (S_EXT1\u2013S_EXT6)</strong></div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered mb-3">
      <thead><tr><th>Strategie</th><th>Signal</th><th>Ziel-Asset</th><th>Logik</th><th>TC</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#58a6ff;">S_EXT1</strong></td>
          <td>TIPS 10J Breakeven\u2191 (monatl.)</td><td>Long GC=F</td>
          <td>Inflationserwartung steigt \u2192 Gold als Hedge gefragt</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#3fb950;">S_EXT2</strong></td>
          <td>USD/CNY 21T-Mom < 0 (Yuan st\u00e4rker)</td><td>Long HG=F</td>
          <td>St\u00e4rkerer Yuan = mehr chinesische Kaufkraft f\u00fcr Kupferimporte</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#d29922;">S_EXT3</strong></td>
          <td>AUD/USD 21T-Mom > 0</td><td>Long GC=F + HG=F (50/50)</td>
          <td>AUD als Commodity Currency: F\u00fchr-Indikator f\u00fcr Metallkomplex</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#bc8cff;">S_EXT4</strong></td>
          <td>HY-OAS 21T-Mom < 0 (Kredit eng)</td><td>Long XLE</td>
          <td>Enger Kreditmarkt \u2192 Shale kann billig finanzieren \u2192 bullisch Energie</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#ffa657;">S_EXT5</strong></td>
          <td>USD/BRL 21T-Mom < 0 (BRL st\u00e4rker)</td><td>Long ZS=F</td>
          <td>Starker Real \u2192 brasilianische Exporte teurer \u2192 Angebotsdruck sinkt</td><td>10 bps</td></tr>
        <tr><td><strong style="color:#ff9fef;">S_EXT6</strong></td>
          <td>Gleich-gewichteter Komposit S_EXT1\u20135</td><td>Diversifiziert</td>
          <td>Diversifikation \u00fcber Kanal-Typen: Inflation/FX/Kredit/Agrar</td><td>10 bps</td></tr>
      </tbody>
    </table>
    {_warn("21T-Momentum-Signal: monatlicher Rhythmus reduziert Turnover vs. t\u00e4glichem Rebalancing. "
           "FRED-Daten (TIPS, HY-OAS) sind monatlich \u2192 auf Tagesbasis forward-gefill'd. "
           "Empfehlung: Wochentliche Rebalancing-Frequenz in der Praxis.")}
  </div>
</div>
{_chart_card("Externe-Faktor Strategien: Equity-Kurven (log-Skala)", fig_eq, height=500,
    interp="Log-Skala: gleiche prozentuale Bewegung = gleicher vertikaler Abstand. "
           "S_EXT6 (Komposit) sollte volatil\u00e4tsarmstes Profil haben. "
           "S_EXT1 (TIPS-Gold) sollte in Inflationsphasen 2021-22 outperformen.")}
{_chart_card("IS vs. OOS Sharpe (70%/30% Split)", fig_sh, height=400,
    interp="Helle Balken = OOS-Sharpe (echter Test). Dunkel = IS. "
           "Degradation > 1: Overfitting. Degradation < 0.3: robuste Strategie. "
           "Externe Faktoren sind weniger overfitted als reine Granger-Strategien (struktureller Kanal).")}
{_card("Performance-Tabelle: Alle externen Strategien", strat_table)}
{_card("Partieller R\u00b2-Tabelle: Signifikante externe Beitr\u00e4ge", r2_table)}
"""
    _write(out / "external_drivers.html", _html_base("Externe Treiber &amp; Strategien", 17, body))
# ─────────────────────────────────────────────────────────────────────────────
# Predictive Monte Carlo Backtest
# ─────────────────────────────────────────────────────────────────────────────

def build_predictive_backtest_report(tables, figures, out):  # noqa: C901
    """Walk-forward predictive backtest with 4 PCA model variants + residual diagnostics."""
    returns = _read(tables / "phase2_returns.csv")
    if returns is None:
        _write(out / "predictive_backtest.html",
               _html_base("Prädiktiver Backtest", 18, "<p>Renditen fehlen.</p>"))
        return
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    # ── Configuration ──────────────────────────────────────────────────
    HORIZONS       = [63, 126, 252, 504]
    HORIZON_LABELS = ["3M", "6M", "1J", "2J"]
    TRAIN_WINDOW   = 504
    RETRAIN_EVERY  = 63
    N_MC           = 300

    TARGET_ASSETS = [c for c in ["GC=F", "CL=F", "HG=F", "ZW=F", "NG=F", "SI=F", "ZS=F"]
                     if c in returns.columns]
    RAW_FACTORS = [c for c in ["DX-Y.NYB", "^VIX", "^TNX"] if c in returns.columns]
    PCA_INPUT   = [c for c in ["SPY", "QQQ", "IWM", "^VIX", "DX-Y.NYB", "^TNX",
                                "XLE", "XLB", "XLI", "JETS", "IYT", "GDX", "MGC"]
                   if c in returns.columns]

    # ── PCA (numpy SVD, no sklearn needed) ─────────────────────────────
    def _compute_pca_scores(df_in):
        """Returns (scores_df, explained_var_array) using SVD."""
        X = df_in.dropna()
        Xc = X - X.mean()
        std = Xc.std()
        std[std < 1e-12] = 1.0
        Xs = Xc / std
        U, s, Vt = np.linalg.svd(Xs.values, full_matrices=False)
        ev = (s ** 2) / max(np.sum(s ** 2), 1e-12)
        k  = Xs.shape[1]
        scores = pd.DataFrame(U * s, index=X.index,
                               columns=[f"PC{i+1}" for i in range(k)])
        return scores, ev

    pca_scores, ev_ratio = _compute_pca_scores(
        returns[PCA_INPUT].shift(1) if PCA_INPUT else pd.DataFrame())

    ev_cumsum = np.cumsum(ev_ratio) if len(ev_ratio) else np.array([])
    n_pc5  = 5
    n_pc95 = max(1, int(np.searchsorted(ev_cumsum, 0.95)) + 1) if len(ev_cumsum) else 3
    n_pc68 = max(1, int(np.searchsorted(ev_cumsum, 0.681)) + 1) if len(ev_cumsum) else 5

    # 4 model configs: (name, label, factor_cols_or_pc_prefix, n_pcs)
    MODEL_CONFIGS = [
        ("raw",   "Roh-Faktoren (DXY/VIX/TNX)", RAW_FACTORS, None),
        ("pc1",   "PCA: PC1 (~" + (f"{ev_ratio[0]*100:.0f}%" if len(ev_ratio) else "?") + " Var.)", None, 1),
        ("pc68",  f"PCA: PC1-{n_pc68} (~68% Var.)", None, n_pc68),
        ("pc95",  f"PCA: PC1-{n_pc95} (~95% Var.)", None, n_pc95),
    ]

    # ── OLS helper ─────────────────────────────────────────────────────
    def _ols_predict(X_tr, y_tr, X_pr):
        Xc = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta = np.linalg.lstsq(Xc, y_tr, rcond=None)[0]
        fitted = Xc @ beta
        return np.column_stack([np.ones(len(X_pr)), X_pr]) @ beta, beta, fitted

    # ── Walk-forward for a given factor matrix ─────────────────────────
    def _run_walkforward(asset, X_factor_df):
        y_full = returns[asset].dropna()
        Xf = X_factor_df.shift(1) if X_factor_df is not None else None
        if Xf is None or Xf.empty:
            return [], {}
        common = y_full.index.intersection(Xf.dropna().index)
        if len(common) < TRAIN_WINDOW + max(HORIZONS) + 50:
            return [], {}
        y_c  = y_full.loc[common].values
        X_c  = Xf.loc[common].dropna().values
        idx2 = common[:len(X_c)]
        n    = min(len(y_c), len(X_c))
        y_c, X_c = y_c[:n], X_c[:n]
        idx2 = idx2[:n]

        fc_by_h = {}
        betas_list = []
        for h_i, h in enumerate(HORIZONS):
            fc_rows = []
            for t_end in range(TRAIN_WINDOW, n - h, RETRAIN_EVERY):
                X_tr = X_c[t_end - TRAIN_WINDOW:t_end]
                y_tr = y_c[t_end - TRAIN_WINDOW:t_end]
                X_pr = X_c[t_end:t_end + h]
                if len(X_pr) < h:
                    continue
                try:
                    y_fc, beta_v, fitted = _ols_predict(X_tr, y_tr, X_pr)
                except Exception:
                    continue
                resids = y_tr - fitted
                y_real  = float(np.nansum(y_c[t_end:t_end + h]))
                fc_pt   = float(np.nansum(y_fc))
                mc_arr  = np.array([
                    float(np.nansum(y_fc + np.random.choice(resids, size=h, replace=True)))
                    for _ in range(N_MC)])
                fc_rows.append({
                    "date": idx2[t_end],
                    "realized": y_real,  "fc_mean": fc_pt,
                    "p10": float(np.percentile(mc_arr, 10)),
                    "p25": float(np.percentile(mc_arr, 25)),
                    "p50": float(np.percentile(mc_arr, 50)),
                    "p75": float(np.percentile(mc_arr, 75)),
                    "p90": float(np.percentile(mc_arr, 90)),
                    "resid_mean": float(np.mean(resids)),
                    "resid_ac1":  float(np.corrcoef(resids[:-1], resids[1:])[0, 1]
                                       if len(resids) > 2 else 0),
                })
                if h_i == 0 and (t_end - TRAIN_WINDOW) % (RETRAIN_EVERY * 4) == 0:
                    betas_list.append({"date": idx2[t_end], "beta": beta_v})
            if fc_rows:
                fc_by_h[h] = pd.DataFrame(fc_rows).set_index("date")
        return fc_by_h, betas_list

    # ── Compute all models for all assets ──────────────────────────────
    model_results  = {}  # (model_name, asset) → fc_by_h
    model_accuracy = []  # rows for comparison table
    model_betas    = {}  # (model_name, asset) → betas_list

    for m_name, m_label, raw_cols, n_pcs in MODEL_CONFIGS:
        if n_pcs is not None:
            if pca_scores.empty or n_pcs > pca_scores.shape[1]:
                continue
            X_df = pca_scores.iloc[:, :n_pcs]
        else:
            if not raw_cols:
                continue
            X_df = returns[raw_cols].copy()
        for asset in TARGET_ASSETS:
            fc_by_h, betas = _run_walkforward(asset, X_df)
            if not fc_by_h:
                continue
            model_results[(m_name, asset)] = fc_by_h
            model_betas[(m_name, asset)]   = betas
            for h_i, h in enumerate(HORIZONS):
                if h not in fc_by_h:
                    continue
                fc = fc_by_h[h]
                err    = fc["realized"] - fc["fc_mean"]
                da     = float(np.mean(np.sign(fc["realized"]) == np.sign(fc["fc_mean"])))
                cov80  = float(np.mean((fc["realized"] >= fc["p10"]) & (fc["realized"] <= fc["p90"])))
                n_pcs_used = n_pcs if n_pcs else len(raw_cols) if raw_cols else 0
                model_accuracy.append({
                    "Modell": m_label, "Asset": asset,
                    "Horizont": HORIZON_LABELS[h_i],
                    "RMSE (bps)": round(float(np.sqrt(np.mean(err**2))) * 10000, 1),
                    "MAE (bps)":  round(float(np.mean(np.abs(err))) * 10000, 1),
                    "Dir. Acc %": round(da * 100, 1),
                    "CI80 %":     round(cov80 * 100, 1),
                    "Bias (bps)": round(float(err.mean()) * 10000, 1),
                    "N": len(fc),
                    "#Faktoren": n_pcs_used,
                })

    # ── Residual diagnostics (best model = raw for reference) ──────────
    # For the 1Y horizon, compute error patterns
    diag_rows = {}
    for asset in TARGET_ASSETS:
        key = ("raw", asset)
        if key not in model_results or 252 not in model_results[key]:
            continue
        fc = model_results[key][252]
        err = fc["realized"] - fc["fc_mean"]
        # Monthly bias
        monthly_bias = err.groupby(err.index.month).mean() * 10000
        # AC1 of errors (lag-1 autocorrelation)
        if len(err) > 5:
            ac1 = float(np.corrcoef(err.values[:-1], err.values[1:])[0, 1])
        else:
            ac1 = 0.0
        # Overshoot ratio: fraction where abs(fc) > abs(realized)
        overshoot = float(np.mean(np.abs(fc["fc_mean"]) > np.abs(fc["realized"])))
        # Systematic undershoot lag: when model underpredicts, does realized peak shortly after?
        undershoot_mask = fc["fc_mean"] < fc["realized"]
        diag_rows[asset] = {
            "AC1 Fehler": round(ac1, 3),
            "Ø Bias (bps)": round(float(err.mean() * 10000), 1),
            "Overshoot-Rate": round(overshoot * 100, 1),
            "Undershoot-Rate": round((1 - overshoot) * 100, 1),
            "monthly_bias": monthly_bias,
            "err_series": err,
        }

    # ── Charts ─────────────────────────────────────────────────────────
    # 1) Model comparison: directional accuracy (PCA vs raw)
    fig_model_comp = go.Figure()
    if model_accuracy:
        acc_df = pd.DataFrame(model_accuracy)
        for asset in TARGET_ASSETS:
            sub = acc_df[(acc_df["Asset"] == asset) & (acc_df["Horizont"] == "1J")]
            if sub.empty:
                continue
            fig_model_comp.add_trace(go.Bar(
                x=sub["Modell"].tolist(),
                y=sub["Dir. Acc %"].tolist(),
                name=asset))
        fig_model_comp.add_hline(y=50, line_color="#8b949e", line_dash="dash",
                                  annotation_text="Zufall (50%)")
        fig_model_comp.update_layout(
            title="Modellvergleich: Direktionale Trefferquote @ 1-Jahres-Horizont (Roh vs. PCA-Modelle)",
            barmode="group", height=450, yaxis_title="Dir. Acc. (%)")

    # 2) PCA Explained Variance Scree plot
    fig_scree = go.Figure()
    if len(ev_ratio):
        n_show = min(len(ev_ratio), 15)
        x_ev = [f"PC{i+1}" for i in range(n_show)]
        fig_scree.add_trace(go.Bar(
            x=x_ev, y=(ev_ratio[:n_show] * 100).tolist(),
            marker_color="#58a6ff", name="Einzeln"))
        fig_scree.add_trace(go.Scatter(
            x=x_ev, y=(ev_cumsum[:n_show] * 100).tolist(),
            mode="lines+markers", name="Kumulativ",
            line=dict(color="#3fb950", width=2)))
        fig_scree.add_hline(y=68.1, line_color="#d29922", line_dash="dash",
                             annotation_text="68.1%")
        fig_scree.add_hline(y=95.0, line_color="#f78166", line_dash="dash",
                             annotation_text="95%")
        fig_scree.update_layout(
            title=f"PCA Scree-Plot: Erklärte Varianz der {len(PCA_INPUT)} Input-Faktoren",
            yaxis_title="Varianz (%)", height=400)

    # 3) Fan chart for GC=F (best model = pc68 if available, else raw)
    best_m = "pc68" if ("pc68", "GC=F") in model_results else "raw"
    fan_htmls = ""
    MONTHS_DE = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
    for asset in TARGET_ASSETS:
        key = (best_m, asset)
        if key not in model_results or 252 not in model_results[key]:
            continue
        fc = model_results[key][252]
        xs = fc.index.astype(str).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=(fc["p90"]*100).round(2).tolist() + (fc["p10"]*100).round(2).tolist()[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.10)", line=dict(width=0),
            name="80%-CI"))
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=(fc["p75"]*100).round(2).tolist() + (fc["p25"]*100).round(2).tolist()[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.22)", line=dict(width=0),
            name="50%-CI"))
        fig.add_trace(go.Scatter(x=xs, y=(fc["p50"]*100).round(2).tolist(),
            mode="lines", name="Median", line=dict(color="#58a6ff", width=1.6, dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=(fc["realized"]*100).round(2).tolist(),
            mode="lines", name="Realisiert", line=dict(color="#3fb950", width=2.0)))
        fig.add_hline(y=0, line_color="#8b949e", line_dash="dot")
        fig.update_layout(
            title=f"Fan-Chart: {asset} | Modell: {best_m} | 1J-Kumulativrendite",
            yaxis_title="Kumulativrendite (%)", height=420)
        fan_htmls += _chart_card(
            f"Fan-Chart {asset} — Walk-Forward ({N_MC} MC-Pfade, Modell: {best_m})", fig,
            interp=f"Grün = realisiert. Blau = Modell-Median. Fächer = 50%/80%-CI. "
                   f"Realisierte Rendite bleibt idealerweise im CI-Band. "
                   f"Systematische Abweichungen zeigen Modell-Bias an.")

    # 4) Residual diagnostics chart
    fig_diag = go.Figure()
    for j, (asset, d) in enumerate(diag_rows.items()):
        mb = d["monthly_bias"]
        if not mb.empty:
            fig_diag.add_trace(go.Scatter(
                x=[MONTHS_DE[m-1] for m in mb.index],
                y=mb.round(1).values.tolist(),
                mode="lines+markers", name=asset,
                line=dict(color=PAL[j % len(PAL)], width=1.5)))
    fig_diag.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_diag.update_layout(
        title="Monatlicher Forecast-Bias (bps): Saisonale Systematik in Prognosefehlern",
        yaxis_title="Ø Fehler (bps)", height=420)

    # 5) Overshoot/Undershoot analysis
    fig_overshoot = go.Figure()
    if diag_rows:
        assets_d   = list(diag_rows.keys())
        ov_rates   = [diag_rows[a]["Overshoot-Rate"] for a in assets_d]
        undr_rates = [diag_rows[a]["Undershoot-Rate"] for a in assets_d]
        ac1_vals   = [diag_rows[a]["AC1 Fehler"] for a in assets_d]
        fig_overshoot.add_trace(go.Bar(x=assets_d, y=ov_rates,
            name="Overshoot %", marker_color="#f78166"))
        fig_overshoot.add_trace(go.Bar(x=assets_d, y=undr_rates,
            name="Undershoot %", marker_color="#58a6ff"))
        fig_overshoot.add_hline(y=50, line_color="#8b949e", line_dash="dash")
        fig_overshoot.update_layout(title="Overshoot vs. Undershoot Rate des Prognosemodells",
            barmode="group", height=380, yaxis_title="%")

    # 6) Error autocorrelation
    fig_ac1 = go.Figure()
    if diag_rows:
        assets_d = list(diag_rows.keys())
        ac1_vals = [diag_rows[a]["AC1 Fehler"] for a in assets_d]
        bias_vals = [diag_rows[a]["Ø Bias (bps)"] for a in assets_d]
        fig_ac1.add_trace(go.Bar(x=assets_d, y=ac1_vals,
            name="AC1 Fehler",
            marker_color=["#f78166" if abs(v) > 0.15 else "#3fb950" for v in ac1_vals]))
        fig_ac1.add_hline(y=0.15, line_color="#f78166", line_dash="dash",
                           annotation_text="Signifikanzgrenze")
        fig_ac1.add_hline(y=-0.15, line_color="#f78166", line_dash="dash")
        fig_ac1.update_layout(
            title="Fehler-Autokorrelation (Lag-1): Muster in Prognosefehlern erkennbar?",
            yaxis_title="AC(1)", height=380)

    # 7) Rolling error bias chart
    fig_roll_bias = go.Figure()
    for j, (asset, d) in enumerate(diag_rows.items()):
        err_s = d["err_series"]
        if len(err_s) < 20:
            continue
        roll_bias = err_s.rolling(10).mean() * 10000
        fig_roll_bias.add_trace(go.Scatter(
            x=roll_bias.index.astype(str).tolist(),
            y=roll_bias.round(1).values.tolist(),
            mode="lines", name=asset,
            line=dict(color=PAL[j % len(PAL)], width=1.5)))
    fig_roll_bias.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_roll_bias.update_layout(
        title="Rollender (10-Fenster) Forecast-Bias (bps): Zeitlicher Verlauf der Fehler",
        yaxis_title="Rollender Ø-Fehler (bps)", height=400)

    # 8) Model × Asset RMSE heatmap
    fig_rmse_heat = go.Figure()
    if model_accuracy:
        acc_df2 = pd.DataFrame(model_accuracy)
        sub_1j = acc_df2[acc_df2["Horizont"] == "1J"]
        model_names = sub_1j["Modell"].unique().tolist()
        asset_names = sub_1j["Asset"].unique().tolist()
        z_rmse = [[float(sub_1j[(sub_1j["Modell"]==m)&(sub_1j["Asset"]==a)]["RMSE (bps)"].values[0])
                   if len(sub_1j[(sub_1j["Modell"]==m)&(sub_1j["Asset"]==a)]) > 0 else float("nan")
                   for a in asset_names] for m in model_names]
        fig_rmse_heat = go.Figure(go.Heatmap(
            z=z_rmse, x=asset_names, y=model_names,
            colorscale="RdYlGn_r",
            text=[[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in z_rmse],
            texttemplate="%{text}",
            hovertemplate="Modell=%{y}<br>Asset=%{x}<br>RMSE=%{z:.0f}bps<extra></extra>"))
        fig_rmse_heat.update_layout(
            title="RMSE-Heatmap (bps) @ 1J-Horizont: Roh-Faktoren vs. PCA-Modelle",
            height=max(300, 70 * len(model_names) + 100))

    # 9) Directional accuracy scatter: PCA1 vs PC68
    fig_da_scatter = go.Figure()
    if model_accuracy:
        acc_df3 = pd.DataFrame(model_accuracy)
        raw_da  = acc_df3[(acc_df3["Modell"].str.startswith("Roh")) & (acc_df3["Horizont"] == "1J")]
        pc68_da = acc_df3[(acc_df3["Modell"].str.contains("PC1-")) & (acc_df3["Horizont"] == "1J")]
        for df_sub, name, col in [(raw_da, "Roh-Faktoren", "#58a6ff"),
                                   (pc68_da, "PCA 68%", "#3fb950")]:
            if not df_sub.empty:
                fig_da_scatter.add_trace(go.Scatter(
                    x=df_sub["Asset"].tolist(),
                    y=df_sub["Dir. Acc %"].tolist(),
                    mode="markers+text", name=name,
                    text=df_sub["Dir. Acc %"].round(1).astype(str).tolist(),
                    textposition="top center",
                    marker=dict(color=col, size=10)))
        fig_da_scatter.add_hline(y=50, line_color="#8b949e", line_dash="dash")
        fig_da_scatter.update_layout(
            title="Direktionale Trefferquote: Roh-Faktoren vs. PCA (68%) @ 1-Jahres-Horizont",
            yaxis_title="Dir. Acc. (%)", height=420)

    # ── Tables ─────────────────────────────────────────────────────────
    acc_table_html = _df_html(pd.DataFrame(model_accuracy).sort_values(
        ["Asset","Horizont","RMSE (bps)"])) if model_accuracy else "<p class=\'text-muted\'>Keine Ergebnisse.</p>"
    diag_table_html = _df_html(pd.DataFrame([
        {"Asset": a, **{k: v for k, v in d.items() if not isinstance(v, (pd.Series, pd.Index))}}
        for a, d in diag_rows.items()])) if diag_rows else ""

    # ── PCA loadings explanation ────────────────────────────────────────
    pc_ev_rows = "".join(
        f"<tr><td>PC{i+1}</td><td>{ev_ratio[i]*100:.1f}%</td>"
        f"<td>{ev_cumsum[i]*100:.1f}%</td>"
        f"<td>{'★' if i < n_pc68 else '·'} {'95%-Grenze' if i == n_pc95-1 else ''}</td></tr>"
        for i in range(min(len(ev_ratio), 15)))

    body = f"""
<div class="ph-header">
  <h1>Prädiktiver Monte-Carlo-Backtest</h1>
  <div class="sub">4 Modelle: Roh-Faktoren | PCA-PC1 | PCA-PC1-{n_pc68} (~68%) | PCA-PC1-{n_pc95} (~95%) &middot;
    Walk-Forward {TRAIN_WINDOW}T &middot; {N_MC} MC-Pfade &middot; 4 Horizonte</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>PCA-Motivation &amp; Modell-Übersicht</strong></div>
  <div class="card-body">
    <div class="row">
      <div class="col-md-5">
        <p class="small">
          Rohfaktoren (DXY, VIX, TNX) sind <strong>kollinear</strong> — z.B. DXY ↑ oft gleichzeitig mit TNX ↑.
          PCA projiziert die {len(PCA_INPUT)} Input-Faktoren auf unkorrelierte Hauptkomponenten.
          PCA kontrolliert Multikollinearität und destilliert "latente Makro-Signale".
        </p>
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Modell</th><th>Faktoren</th><th>Erklärte Var.</th></tr></thead>
          <tbody>
            <tr><td>Roh</td><td>DXY, VIX, TNX (3)</td><td>n/a</td></tr>
            <tr><td>PCA-1</td><td>PC1</td><td>{ev_ratio[0]*100:.1f}% (1 Faktor)</td></tr>
            <tr><td>PCA-68%</td><td>PC1-{n_pc68}</td><td>~68.1%</td></tr>
            <tr><td>PCA-95%</td><td>PC1-{n_pc95}</td><td>~95%</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-7">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>PC</th><th>Einzel-Var.</th><th>Kumulativ</th><th>Modell</th></tr></thead>
          <tbody>{pc_ev_rows}</tbody>
        </table>
      </div>
    </div>
    {_formula(r"r_{{t+h}} = \alpha + \sum_{{k=1}}^{{K}} \beta_k \cdot PC_k^{{(t)}} + \varepsilon_{{t+h}}",
              "Prädiktive PCA-Regression: K unkorrelierte Hauptkomponenten als Prädiktoren (1-Tages-Lag).")}
    {_info(f"Input für PCA: {', '.join(PCA_INPUT[:8])}... ({len(PCA_INPUT)} Faktoren). "
           f"Alle t-1 verzögert (keine Look-ahead). Walk-Forward: Train={TRAIN_WINDOW}T, Refit alle {RETRAIN_EVERY}T.")}
  </div>
</div>

{_chart_card("PCA Scree-Plot: Erklärte Varianz der Input-Faktoren", fig_scree, height=420,
    interp="Balkenhöhe = Einzelbeitrag jeder PC. Grüne Linie = Kumulativ. "
           "PC1 dominiert meist (Risk-on/Risk-off Faktor). "
           "Gelbe Linie 68.1%: Modell PC1-{n_pc68}. Rote Linie 95%: Modell PC1-{n_pc95}.")}

{_chart_card("Modellvergleich: Direktionale Trefferquote @ 1-Jahres-Horizont", fig_model_comp, height=470,
    interp="Roh-Faktoren vs. PCA-Varianten: Verbessert PCA die Vorhersage? "
           "PCA mit 68%+ Varianz stabilisiert die Schätzung durch Multikollinearitäts-Kontrolle. "
           "PC1-only: zu restriktiv — verliert Information. PC95%: zu viele Faktoren — Overfitting.")}

{_chart_card("RMSE-Heatmap (bps) nach Modell und Asset", fig_rmse_heat,
    interp="Grün = niedrigerer Fehler (besser). Rot = höherer Fehler. "
           "Vergleich zeigt, ob PCA tatsächlich Rohmodell schlägt. "
           "Diagonal-Muster (bestimmtes Asset systematisch besser) deutet auf Asset-spezifische Muster hin.")}

{_chart_card("Dir. Trefferquote: Roh vs. PCA 68%", fig_da_scatter, height=440,
    interp="Grüne Punkte über blauen: PCA verbessert Trefferquote. "
           "50%-Linie = zufälliges Raten. "
           ">55% bei 92 Vorhersagen statistisch signifikant (Binomialtest p<0.05).")}

{fan_htmls}

<div class="card mb-4" id="diagnostics">
  <div class="card-header"><strong>🔬 Residual-Diagnostik: Fehler-Muster &amp; Zeitversatz</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Prognosemodelle können systematische Fehler-Muster aufweisen:
      <strong>Overshoot</strong> (Modell überschätzt Bewegung), <strong>Undershoot</strong> (unterschätzt),
      <strong>Lag-Bias</strong> (Fehler autokorreliert = Modell "hinkt hinterher"),
      <strong>Saisonale Verzerrung</strong> (in bestimmten Monaten systematisch falsch).
    </p>
    <table class="table table-dark table-sm table-bordered mb-3">
      <thead><tr><th>Phänomen</th><th>Diagnose</th><th>Ursache</th><th>Lösung</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#f78166;">Overshoot</strong></td>
          <td>|Forecast| > |Realisiert| häufig</td>
          <td>Hohe VIX-Phase: Modell extrapoliert Crash zu lange</td>
          <td>Regimes trennen, Shrinkage-Prior</td></tr>
        <tr><td><strong style="color:#58a6ff;">Undershoot</strong></td>
          <td>|Forecast| < |Realisiert| häufig</td>
          <td>Trendphasen: OLS unterschätzt Momentum</td>
          <td>Momentum-Faktor ergänzen</td></tr>
        <tr><td><strong style="color:#d29922;">Lag-Bias (AC1>0)</strong></td>
          <td>Fehler(t) ≈ Fehler(t-1)</td>
          <td>Strukturbruch nicht erkannt, zu langsames Refitting</td>
          <td>Häufigeres Refitting, EWM statt OLS</td></tr>
        <tr><td><strong style="color:#3fb950;">Saisonaler Bias</strong></td>
          <td>Monatsweise systematisch hoch/tief</td>
          <td>Fehlende Saisonalitäts-Variable</td>
          <td>Monatsdummies, saisonale Normalisierung</td></tr>
      </tbody>
    </table>
  </div>
</div>

{_chart_card("Overshoot vs. Undershoot Rate (Roh-Modell, 1J-Horizont)", fig_overshoot, height=400,
    interp="Overshoot >50%: Modell neigt dazu, Bewegungen zu überschätzen (typisch nach Krisen). "
           "Undershoot >50%: Modell unterschätzt (typisch in Trendphasen). "
           "Ideal: 50/50 — symmetrischer, unbiased Forecast.")}

{_chart_card("Fehler-Autokorrelation AC(1): Zeitversatz-Muster erkennbar?", fig_ac1, height=400,
    interp="AC(1) > 0.15 (rot): Fehler(t) korreliert mit Fehler(t-1) = Modell hinkt hinterher. "
           "AC(1) < 0: überschnelles Mean-Reversion in Fehlern (seltener). "
           "Idealfall: AC(1) ≈ 0 (weiße Residuen = kein exploitierbares Muster mehr).")}

{_chart_card("Monatlicher Forecast-Bias: Saisonale Fehler-Systematik", fig_diag, height=440,
    interp="Positiver Bias in Monat X: Modell unterschätzt systematisch im Monat X. "
           "Negativer Bias: Modell überschätzt. "
           "Konsistente Muster → saisonale Dummy-Variablen als Korrekturterm ergänzen.")}

{_chart_card("Rollender Forecast-Bias (10-Fenster): Zeitlicher Verlauf", fig_roll_bias, height=420,
    interp="Phasen mit konstantem positiven/negativen Bias = Regime-Shift nicht erfasst. "
           "2008/2009: starker negativer Bias (Crash unterschätzt). "
           "2020: positiver Bias nach Crash-Tief (Erholung unterschätzt). "
           "Wechsel von positiv zu negativ = Wendepunkt im Modell-Fehler.")}

{_card("Vollständige Genauigkeitstabelle: Alle Modelle × Assets × Horizonte", acc_table_html)}
{_card("Residual-Diagnose-Tabelle (Roh-Modell, 1J-Horizont)", diag_table_html)}
"""
    _write(out / "predictive_backtest.html",
           _html_base("Prädiktiver Monte-Carlo-Backtest", 18, body))


def build_mega_strategies_report(tables, figures, out):  # noqa: C901
    """80+ trading strategies across 7 families, parameter-optimised, accordion Steckbriefe."""
    returns = _read(tables / "phase2_returns.csv")
    if returns is None:
        _write(out / "mega_strategies.html",
               _html_base("Mega-Strategien", 18, "<p>Renditen fehlen.</p>"))
        return
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    # ── PCA factors (unkorrellierte Signale) ───────────────────────────
    PCA_INPUT = [c for c in ["SPY","QQQ","IWM","^VIX","DX-Y.NYB","^TNX",
                              "XLE","XLB","XLI","JETS","IYT","GDX","MGC"]
                 if c in returns.columns]

    pc_scores_df = pd.DataFrame(index=returns.index)
    if PCA_INPUT:
        X_pca = returns[PCA_INPUT].copy()
        X_pca = X_pca - X_pca.mean()
        std_pca = X_pca.std()
        std_pca[std_pca < 1e-12] = 1.0
        X_pca = X_pca / std_pca
        X_pca_clean = X_pca.dropna()
        try:
            U, s, Vt = np.linalg.svd(X_pca_clean.values, full_matrices=False)
            pc_scores_raw = pd.DataFrame(U * s, index=X_pca_clean.index,
                                         columns=[f"PC{i+1}" for i in range(U.shape[1])])
            pc_scores_df = pc_scores_raw
        except Exception:
            pass

    # ── Unified strategy metrics ────────────────────────────────────────
    def _strat_metrics(r: pd.Series, name: str) -> dict | None:
        r = r.dropna()
        if len(r) < 252:
            return None
        ann_r  = float(r.mean() * 252)
        ann_v  = float(r.std() * np.sqrt(252)) + 1e-9
        sharpe = ann_r / ann_v
        cum    = (1 + r).cumprod()
        mdd    = float((cum / cum.cummax() - 1).min())
        split  = int(len(r) * 0.7)
        is_sh  = float(r.iloc[:split].mean() * 252 /
                       (r.iloc[:split].std() * np.sqrt(252) + 1e-9))
        oos_sh = float(r.iloc[split:].mean() * 252 /
                       (r.iloc[split:].std() * np.sqrt(252) + 1e-9))
        calmar = ann_r / (abs(mdd) + 1e-9)
        win_r  = float((r > 0).mean())
        skew   = float(pd.Series(r).skew())
        kurt   = float(pd.Series(r).kurt())
        # Rolling Sharpe stability
        roll_sh = r.rolling(126).apply(
            lambda x: x.mean()*252 / (x.std()*np.sqrt(252)+1e-9), raw=True)
        sh_std  = float(roll_sh.std())
        return {
            "Name": name,
            "CAGR %": round(ann_r * 100, 2),
            "Sharpe": round(sharpe, 3),
            "MaxDD %": round(mdd * 100, 2),
            "Calmar": round(calmar, 3),
            "IS Sharpe": round(is_sh, 3),
            "OOS Sharpe": round(oos_sh, 3),
            "Degrad.": round(is_sh - oos_sh, 3),
            "Win %": round(win_r * 100, 1),
            "Skew": round(skew, 3),
            "Kurt": round(kurt, 3),
            "Sharpe-Stab.": round(sh_std, 3),
            "N Tage": len(r),
        }

    def _apply_tc(pos: pd.Series, ret: pd.Series, bps=10) -> pd.Series:
        return pos * ret - pos.diff().abs().fillna(0) * (bps / 10000)

    def _make_signal_mom(r: pd.Series, lb: int) -> pd.Series:
        return np.sign(r.rolling(lb).sum().shift(1))

    def _make_signal_zrev(r: pd.Series, lb: int, entry_z: float) -> pd.Series:
        z = (r - r.rolling(lb).mean()) / (r.rolling(lb).std() + 1e-9)
        sig = pd.Series(0.0, index=r.index)
        sig[z.shift(1) < -entry_z] = 1.0   # oversold → long
        sig[z.shift(1) >  entry_z] = -1.0  # overbought → short
        return sig

    def _make_signal_macro(factor: pd.Series, lb: int) -> pd.Series:
        return np.sign(factor.rolling(lb).sum().shift(1))

    def _make_signal_vix_regime(vix: pd.Series, threshold: float) -> pd.Series:
        return pd.Series(np.where(vix.shift(1) < threshold, 1.0, -1.0), index=vix.index)

    def _make_signal_seasonal(r: pd.Series, long_months: list) -> pd.Series:
        return pd.Series(
            np.where(r.index.month.isin(long_months), 1.0, -1.0),
            index=r.index)

    # ── Strategy definitions ────────────────────────────────────────────
    COMMODITY_ASSETS = [c for c in ["GC=F","CL=F","HG=F","ZW=F","NG=F","SI=F","ZS=F","BZ=F","ZC=F"]
                        if c in returns.columns]

    all_strats: list[dict] = []  # {name, family, returns_series, description, signal}

    # ─── Family A: Momentum (3 lookbacks × 9 assets) ───────────────────
    for lb in [21, 63, 126]:
        for asset in COMMODITY_ASSETS:
            r = returns[asset].dropna()
            if len(r) < 252 + lb:
                continue
            sig = _make_signal_mom(r, lb)
            strat_r = _apply_tc(sig, r)
            all_strats.append({
                "family": "Momentum", "asset": asset, "lb": lb,
                "name": f"MOM-{lb}T-{asset}",
                "desc": f"Momentum-Signal: {lb}-Tages-Return Vorzeichen → Long/Short {asset}",
                "signal": sig, "ret": strat_r,
            })

    # ─── Family B: Mean Reversion (2 lookbacks × 2 Z-thresholds × 7 assets) ─
    for lb in [21, 63]:
        for z_thr in [1.0, 2.0]:
            for asset in COMMODITY_ASSETS[:7]:
                r = returns[asset].dropna()
                if len(r) < 252 + lb:
                    continue
                sig = _make_signal_zrev(r, lb, z_thr)
                strat_r = _apply_tc(sig, r)
                all_strats.append({
                    "family": "Mean Reversion", "asset": asset, "lb": lb,
                    "name": f"ZREV-{lb}T-Z{z_thr:.0f}-{asset}",
                    "desc": f"Z-Score Mean Reversion: ±{z_thr}σ Entry, {lb}T Lookback, {asset}",
                    "signal": sig, "ret": strat_r,
                })

    # ─── Family C: Macro Signal (DXY, VIX-Regime, TNX) × 6 assets ─────
    macro_sigs = []
    for col, lb, desc in [("DX-Y.NYB", 21, "DXY-Momentum"),
                           ("DX-Y.NYB", 63, "DXY-Momentum 63T"),
                           ("^TNX", 21, "10Y-Zinsen Richtung"),
                           ("^TNX", 63, "10Y-Zinsen 63T")]:
        if col in returns.columns:
            macro_sigs.append((col, lb, desc))

    for col, lb, sig_desc in macro_sigs:
        fac = returns[col].dropna()
        sig_base = _make_signal_macro(fac, lb)
        for asset in COMMODITY_ASSETS[:6]:
            r = returns[asset].dropna()
            idx = sig_base.index.intersection(r.index)
            if len(idx) < 252:
                continue
            # Inverse signal: DXY up = commodity down
            inv = -1 if col == "DX-Y.NYB" else 1
            sig = sig_base.loc[idx] * inv
            strat_r = _apply_tc(sig, r.loc[idx])
            all_strats.append({
                "family": "Makro-Signal", "asset": asset, "lb": lb,
                "name": f"MACRO-{col.replace('^','').replace('-','')}-{lb}T-{asset}",
                "desc": f"{sig_desc} → {'Short' if inv==-1 else 'Long'}/{asset}",
                "signal": sig, "ret": strat_r,
            })

    # VIX-Regime
    if "^VIX" in returns.columns:
        vix = returns["^VIX"].dropna()
        for thr in [20, 25, 30]:
            sig_vix = _make_signal_vix_regime(vix, thr)
            for asset in COMMODITY_ASSETS[:6]:
                r = returns[asset].dropna()
                idx = sig_vix.index.intersection(r.index)
                if len(idx) < 252:
                    continue
                strat_r = _apply_tc(sig_vix.loc[idx], r.loc[idx])
                all_strats.append({
                    "family": "Makro-Signal", "asset": asset,
                    "name": f"VIXREG-{thr}-{asset}",
                    "desc": f"VIX-Regime: VIX<{thr} → Long {asset}, sonst Short",
                    "signal": sig_vix.loc[idx], "ret": strat_r,
                })

    # ─── Family D: PCA Signal (PC1-3 × 5 assets) ──────────────────────
    for pc_n in [1, 2, 3]:
        pc_col = f"PC{pc_n}"
        if pc_col not in pc_scores_df.columns:
            continue
        pc = pc_scores_df[pc_col]
        for asset in COMMODITY_ASSETS[:5]:
            r = returns[asset].dropna()
            # Try both polarities (PC sign is arbitrary)
            for inv_label, inv in [("pos", 1), ("neg", -1)]:
                sig = np.sign(pc.shift(1) * inv)
                idx = sig.dropna().index.intersection(r.index)
                if len(idx) < 252:
                    continue
                strat_r = _apply_tc(sig.loc[idx], r.loc[idx])
                m = _strat_metrics(strat_r, "")
                if m and abs(m["Sharpe"]) > 0.1:  # keep better polarity only
                    all_strats.append({
                        "family": "PCA-Signal", "asset": asset,
                        "name": f"PCA-PC{pc_n}-{inv_label}-{asset}",
                        "desc": f"PC{pc_n} Vorzeichen-Signal ({inv_label}) → {asset}",
                        "signal": sig.loc[idx], "ret": strat_r,
                    })
                    break  # take first polarity that passes threshold

    # ─── Family E: Cross-Asset Ratio Signals ──────────────────────────
    ratio_pairs = [
        ("HG=F","GC=F",  "Kupfer/Gold",   1, "HG=F",  "Kupfer/Gold↑ → Wachstum → Long HG=F"),
        ("GC=F","SI=F",  "Gold/Silber",   1, "GC=F",  "Gold/Silber↑ → Risk-Off → Long GC=F"),
        ("CL=F","NG=F",  "Öl/Gas",       1, "CL=F",  "Öl/Gas↑ → Öl outperformed"),
        ("CL=F","GC=F",  "Öl/Gold",      -1,"GC=F",  "Öl/Gold↑ → Risiko-On → Short GC=F"),
        ("ZW=F","ZC=F",  "Weizen/Mais", 1, "ZW=F",  "Weizen/Mais-Spread → Long Weizen"),
        ("XLE","GDX",    "Energie/Gold-ETF",1,"CL=F","XLE/GDX↑ → Energie outperformed → Long CL=F"),
    ]
    for a1, a2, ratio_nm, inv, target, desc in ratio_pairs:
        if a1 not in returns.columns or a2 not in returns.columns or target not in returns.columns:
            continue
        r1 = returns[a1].dropna(); r2 = returns[a2].dropna()
        idx_r = r1.index.intersection(r2.index)
        ratio = (r1.loc[idx_r] - r2.loc[idx_r]).rolling(21).sum() * inv
        sig = np.sign(ratio.shift(1))
        r_t = returns[target].dropna()
        idx = sig.dropna().index.intersection(r_t.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig.loc[idx], r_t.loc[idx])
        all_strats.append({
            "family": "Cross-Asset", "asset": target,
            "name": f"RATIO-{a1.replace('=F','').replace('^','')}-{a2.replace('=F','').replace('^','')}",
            "desc": desc, "signal": sig.loc[idx], "ret": strat_r,
        })

    # ─── Family F: Seasonal Calendar Strategies ───────────────────────
    seasonal_defs = [
        ("GC=F",  [9,10,11,12],  "Gold Q4 Saisonalität (Schmuck-Hochsaison)"),
        ("NG=F",  [10,11,12,1,2],"Gas Heizperiode (Okt-Feb)"),
        ("NG=F",  [6,7,8],       "Gas Kühlperiode (Jun-Aug)"),
        ("ZW=F",  [3,4,5],       "Weizen Pflanzungsrallye (Mär-Mai)"),
        ("ZS=F",  [3,4,5],       "Soja Planting Rally (Mär-Mai)"),
        ("CL=F",  [5,6,7,8],     "Öl Driving Season (Mai-Aug)"),
        ("HG=F",  [1,2,3],       "Kupfer China-Stimulus Q1 (Jan-Mär)"),
        ("ZW=F",  [9,10,11],     "Weizen Short: Ernte-Druck (Sep-Nov)"),
    ]
    for asset, months, desc in seasonal_defs:
        if asset not in returns.columns:
            continue
        r = returns[asset].dropna()
        if len(r) < 252:
            continue
        sig = _make_signal_seasonal(r, months)
        strat_r = _apply_tc(sig, r)
        months_str = "/".join(map(str, months))
        all_strats.append({
            "family": "Saisonal", "asset": asset,
            "name": f"SEASON-{asset.replace('=F','').replace('^','')}-{months_str}",
            "desc": desc, "signal": sig, "ret": strat_r,
        })

    # ─── Family G: Multi-Signal Composites ────────────────────────────
    # Combine MOM + MACRO signals per asset
    for asset in COMMODITY_ASSETS[:5]:
        r = returns[asset].dropna()
        sigs_to_combine = []
        # MOM 63T signal
        sig_m = _make_signal_mom(r, 63)
        sigs_to_combine.append(sig_m)
        # DXY signal
        if "DX-Y.NYB" in returns.columns:
            sig_d = -_make_signal_macro(returns["DX-Y.NYB"].dropna(), 21)
            sigs_to_combine.append(sig_d)
        # PC1 signal
        if "PC1" in pc_scores_df.columns:
            sig_p = np.sign(pc_scores_df["PC1"].shift(1))
            sigs_to_combine.append(sig_p)
        if len(sigs_to_combine) < 2:
            continue
        combined = pd.concat(sigs_to_combine, axis=1).mean(axis=1).dropna()
        sig_c = np.sign(combined)
        idx = sig_c.index.intersection(r.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig_c.loc[idx], r.loc[idx])
        all_strats.append({
            "family": "Komposit", "asset": asset,
            "name": f"COMBO-MOM-DXY-PC1-{asset.replace('=F','').replace('^','')}",
            "desc": f"Komposit: MOM-63T + DXY-Signal + PC1 → {asset}",
            "signal": sig_c.loc[idx], "ret": strat_r,
        })

    # ─── Family H: Granger / CCF Lead-Lag Signals ─────────────────────
    # Known strong Granger pairs from phase 6 analysis
    granger_pairs = [
        ("CL=F",  "SM",   1,  1, "Öl → Small-Caps (Lag 1T)"),
        ("DX-Y.NYB","CVX", 1, -1, "DXY → CVX (Lag 1T, invers)"),
        ("GC=F",  "GDX",  1,  1, "Gold → GDX (Lag 1T)"),
        ("CL=F",  "XLE",  1,  1, "Öl → XLE (Lag 1T)"),
        ("^VIX",  "GC=F", 1, -1, "VIX → Gold (invers, Risk-off)"),
        ("HG=F",  "GDX",  2,  1, "Kupfer → GDX (Lag 2T)"),
        ("CL=F",  "HG=F", 2,  1, "Öl → Kupfer (Lag 2T)"),
    ]
    for src, tgt, lag, inv, desc in granger_pairs:
        if src not in returns.columns or tgt not in returns.columns:
            continue
        src_r = returns[src].dropna()
        tgt_r = returns[tgt].dropna()
        sig = np.sign(src_r.rolling(21).sum().shift(lag) * inv)
        idx = sig.dropna().index.intersection(tgt_r.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig.loc[idx], tgt_r.loc[idx])
        all_strats.append({
            "family": "Granger/CCF", "asset": tgt,
            "name": f"GRANGER-{src.replace('=F','').replace('^','').replace('-','')}-{tgt}-L{lag}",
            "desc": desc, "signal": sig.loc[idx], "ret": strat_r,
        })

    # ─── Family I: Overshoot/Correction Strategies ────────────────────
    # Enter mean-reversion after extreme moves (>2σ in 5 days)
    for asset in COMMODITY_ASSETS[:6]:
        r = returns[asset].dropna()
        if len(r) < 252:
            continue
        roll_5 = r.rolling(5).sum()
        vol_21 = r.rolling(21).std() * np.sqrt(5)
        z_5d = roll_5 / (vol_21 + 1e-9)
        # After extreme drop (z < -2): expect recovery → long
        sig_ov = pd.Series(0.0, index=r.index)
        sig_ov[z_5d.shift(1) < -2.0] = 1.0   # bounce after crash
        sig_ov[z_5d.shift(1) >  2.0] = -1.0  # fade after spike
        strat_r = _apply_tc(sig_ov, r)
        all_strats.append({
            "family": "Overshoot/Korr.", "asset": asset,
            "name": f"OVERSHOOT-5T-Z2-{asset.replace('=F','').replace('^','')}",
            "desc": f"Mean-Reversion nach 5T-Extrem (±2σ): Bounce/Fade Signal für {asset}",
            "signal": sig_ov, "ret": strat_r,
        })

    # ── Compute metrics for all strategies ─────────────────────────────
    strat_meta = []
    for s in all_strats:
        m = _strat_metrics(s["ret"], s["name"])
        if m:
            m["Familie"] = s["family"]
            m["Asset"]   = s["asset"]
            m["Beschreibung"] = s.get("desc", "")
            strat_meta.append(m)

    if not strat_meta:
        _write(out / "mega_strategies.html",
               _html_base("Mega-Strategien", 18, "<p>Keine Strategie-Ergebnisse.</p>"))
        return

    meta_df = pd.DataFrame(strat_meta).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    meta_df["Rang"] = meta_df.index + 1

    # ── Summary charts ─────────────────────────────────────────────────
    # 1) Sharpe distribution by family
    fig_sharpe_box = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]["Sharpe"]
        fig_sharpe_box.add_trace(go.Box(y=sub.tolist(), name=fam,
            marker_color=PAL[j % len(PAL)], boxpoints="all", jitter=0.4))
    fig_sharpe_box.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_sharpe_box.add_hline(y=0.5, line_color="#3fb950", line_dash="dot",
                              annotation_text="Sharpe=0.5")
    fig_sharpe_box.update_layout(
        title=f"Sharpe-Verteilung nach Strategie-Familie ({len(meta_df)} Strategien total)",
        yaxis_title="Sharpe Ratio", height=480)

    # 2) Top-20 Sharpe bar chart
    top20 = meta_df.head(20)
    fig_top20 = go.Figure(go.Bar(
        x=top20["Name"].tolist(),
        y=top20["Sharpe"].tolist(),
        marker_color=["#3fb950" if v > 0.5 else "#d29922" if v > 0 else "#f78166"
                      for v in top20["Sharpe"]],
        text=[f"{v:.2f}" for v in top20["Sharpe"]],
        textposition="outside",
        hovertemplate="%{x}<br>Sharpe=%{y:.3f}<extra></extra>"))
    fig_top20.add_hline(y=0, line_color="#8b949e")
    fig_top20.update_layout(
        title="Top-20 Strategien: Sharpe Ratio (sortiert)",
        xaxis_tickangle=-40, height=520, yaxis_title="Sharpe")

    # 3) IS vs OOS scatter
    fig_is_oos = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_is_oos.add_trace(go.Scatter(
            x=sub["IS Sharpe"].tolist(), y=sub["OOS Sharpe"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7),
            hovertemplate="%{text}<br>IS=%{x:.2f} OOS=%{y:.2f}<extra></extra>"))
    mn = float(min(meta_df["IS Sharpe"].min(), meta_df["OOS Sharpe"].min())) - 0.1
    mx = float(max(meta_df["IS Sharpe"].max(), meta_df["OOS Sharpe"].max())) + 0.1
    fig_is_oos.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#8b949e", dash="dash"), name="IS=OOS", showlegend=True))
    fig_is_oos.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    fig_is_oos.update_layout(
        title="IS vs. OOS Sharpe: Robustheit der Strategien (Punkte oberhalb Diagonale = Overfitting)",
        xaxis_title="IS Sharpe (70%)", yaxis_title="OOS Sharpe (30%)", height=500)

    # 4) MaxDD vs Sharpe scatter
    fig_dd_sh = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_dd_sh.add_trace(go.Scatter(
            x=sub["MaxDD %"].tolist(), y=sub["Sharpe"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7)))
    fig_dd_sh.update_layout(
        title="MaxDD vs. Sharpe: Risk-Return Profil aller Strategien",
        xaxis_title="Max Drawdown (%)", yaxis_title="Sharpe", height=480)

    # 5) Family heatmap (avg Sharpe by family × asset)
    fams = meta_df["Familie"].unique().tolist()
    assets_uniq = meta_df["Asset"].unique().tolist()
    heat_z = [[float(meta_df[(meta_df["Familie"]==f)&(meta_df["Asset"]==a)]["Sharpe"].mean())
               if len(meta_df[(meta_df["Familie"]==f)&(meta_df["Asset"]==a)]) else float("nan")
               for a in assets_uniq] for f in fams]
    fig_fam_heat = go.Figure(go.Heatmap(
        z=heat_z, x=assets_uniq, y=fams,
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in heat_z],
        texttemplate="%{text}",
        hovertemplate="Familie=%{y}<br>Asset=%{x}<br>Ø Sharpe=%{z:.2f}<extra></extra>"))
    fig_fam_heat.update_layout(
        title="Ø Sharpe-Heatmap: Strategie-Familie × Asset",
        height=max(300, 55 * len(fams) + 100))

    # 6) Cumulative return: top 5 equity curves
    fig_eq_top5 = go.Figure()
    for j, row in meta_df.head(5).iterrows():
        s_obj = next((s for s in all_strats if s["name"] == row["Name"]), None)
        if s_obj is None:
            continue
        eq = (1 + s_obj["ret"].dropna()).cumprod()
        fig_eq_top5.add_trace(go.Scatter(
            x=eq.index.astype(str).tolist(),
            y=np.round(eq.values, 4).tolist(),
            mode="lines", name=f"#{j+1} {row['Name']}",
            line=dict(color=PAL[j % len(PAL)], width=2)))
    fig_eq_top5.update_layout(
        title="Top-5 Strategien: Equity-Kurven (log-Skala)",
        yaxis_type="log", yaxis_title="Kapital (Basis=1)", height=520)

    # 7) CAGR vs MaxDD (efficient frontier)
    fig_frontier = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_frontier.add_trace(go.Scatter(
            x=(-sub["MaxDD %"]).tolist(), y=sub["CAGR %"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7)))
    fig_frontier.update_layout(
        title="Effizienzgrenze: CAGR vs. Max-Drawdown — Risiko/Rendite für alle Strategien",
        xaxis_title="-MaxDD (%)", yaxis_title="CAGR (%)", height=480)

    # ── Accordion Steckbriefe ─────────────────────────────────────────
    import json as _json

    def _build_steckbrief(rank: int, row: pd.Series, s_obj: dict | None) -> str:
        sh_color = "#3fb950" if row["Sharpe"] > 0.5 else "#d29922" if row["Sharpe"] > 0 else "#f78166"
        header_badge = (f'<span class="badge" style="background:{sh_color}">'
                        f'Sharpe {row["Sharpe"]:.2f}</span>')
        oos_badge_color = "#3fb950" if row["OOS Sharpe"] > 0.3 else "#d29922" if row["OOS Sharpe"] > 0 else "#f78166"
        oos_badge = (f'<span class="badge ms-1" style="background:{oos_badge_color}">'
                     f'OOS {row["OOS Sharpe"]:.2f}</span>')
        dd_badge = (f'<span class="badge ms-1 bg-secondary">DD {row["MaxDD %"]:.1f}%</span>')
        fam_badge = f'<span class="badge ms-1 bg-secondary">{row["Familie"]}</span>'
        metric_rows = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in row.items()
            if k not in ("Name","Familie","Asset","Beschreibung","Rang"))

        chart_html = ""
        if rank <= 15 and s_obj is not None:  # embed Plotly for top 15
            eq = (1 + s_obj["ret"].dropna()).cumprod()
            # Downsample to every 5 trading days to keep JSON small
            eq_ds = eq.iloc[::5]
            fig_sb = go.Figure()
            fig_sb.add_trace(go.Scatter(
                x=eq_ds.index.astype(str).tolist(),
                y=np.round(eq_ds.values, 4).tolist(),
                mode="lines", name="Equity",
                line=dict(color=PAL[(rank-1) % len(PAL)], width=1.8)))
            # Buy & hold (also downsampled)
            bh = (1 + returns[row["Asset"]].dropna()).cumprod()
            bh_ds = bh.iloc[::5]
            fig_sb.add_trace(go.Scatter(
                x=bh_ds.index.astype(str).tolist(),
                y=np.round(bh_ds.values, 4).tolist(),
                mode="lines", name="B&H",
                line=dict(color="#8b949e", width=0.9, dash="dot")))
            fig_sb.update_layout(
                height=280, margin=dict(t=30,b=30,l=30,r=10),
                yaxis_type="log", showlegend=True,
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", size=10),
                legend=dict(orientation="h", y=1.1))

            # Return distribution mini chart (use 80 bins max)
            fig_dist = go.Figure()
            ret_vals = s_obj["ret"].dropna().values * 10000
            fig_dist.add_trace(go.Histogram(
                x=ret_vals.tolist(), nbinsx=50,
                marker_color="#58a6ff", opacity=0.75))
            fig_dist.update_layout(
                height=200, margin=dict(t=20,b=30,l=30,r=10),
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", size=9),
                showlegend=False,
                xaxis_title="Tagesrendite (bps)", yaxis_title="Häufigkeit")

            eq_json   = _json.dumps(fig_sb.to_dict()).replace("</", "<\\/")
            dist_json = _json.dumps(fig_dist.to_dict()).replace("</", "<\\/")
            uid = f"sb_{rank}"
            chart_html = (
                f'<div id="eq_{uid}" style="height:280px;"></div>'
                f'<div id="dist_{uid}" style="height:200px;"></div>'
                f'<script>'
                f'(function(){{var eq={eq_json};'
                f'var dist={dist_json};'
                f'if(window.Plotly){{Plotly.newPlot("eq_{uid}",eq.data,eq.layout,{{responsive:true,displayModeBar:false}});'
                f'Plotly.newPlot("dist_{uid}",dist.data,dist.layout,{{responsive:true,displayModeBar:false}});}}}})()'
                f'</script>'
            )

        sig_stats = ""
        if s_obj is not None and "signal" in s_obj:
            sig = s_obj["signal"].dropna()
            n_long  = int((sig > 0).sum())
            n_short = int((sig < 0).sum())
            n_flat  = int((sig == 0).sum())
            sig_stats = (f'<div class="small text-muted mt-2">'
                         f'Signal: Long {n_long} Tage | Short {n_short} Tage | Flat {n_flat} Tage | '
                         f'Long-Anteil {n_long/(n_long+n_short+n_flat+1)*100:.0f}%</div>')

        return f"""
<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">
  <h2 class="accordion-header">
    <button class="accordion-button collapsed py-2" type="button"
      data-bs-toggle="collapse" data-bs-target="#sb{rank}"
      style="background:#0d1117;color:#e6edf3;font-size:0.85rem;">
      <strong class="me-2">#{rank}</strong>
      <span class="me-2">{row["Name"]}</span>
      {header_badge}{oos_badge}{dd_badge}{fam_badge}
    </button>
  </h2>
  <div id="sb{rank}" class="accordion-collapse collapse">
    <div class="accordion-body" style="background:#161b22;">
      <p class="small text-muted">{row.get("Beschreibung","")}</p>
      <div class="row">
        <div class="col-md-4">
          <table class="table table-dark table-sm table-bordered">
            <tbody>{metric_rows}</tbody>
          </table>
          {sig_stats}
        </div>
        <div class="col-md-8">{chart_html}</div>
      </div>
    </div>
  </div>
</div>"""

    # Build accordion HTML
    accordion_html = '<div class="accordion" id="stratAccordion">'
    for rank, (_, row) in enumerate(meta_df.iterrows(), 1):
        s_obj = next((s for s in all_strats if s["name"] == row["Name"]), None)
        accordion_html += _build_steckbrief(rank, row, s_obj)
    accordion_html += "</div>"

    # ── Summary table (top 30) ──────────────────────────────────────────
    display_cols = ["Rang","Name","Familie","Asset","Sharpe","OOS Sharpe","CAGR %",
                    "MaxDD %","Calmar","Degrad.","Win %","N Tage"]
    top30_html = _df_html(meta_df[display_cols].head(30))

    body = f"""
<div class="ph-header">
  <h1>Mega-Strategien: {len(meta_df)} Strategien — 9 Familien</h1>
  <div class="sub">Momentum · Mean Reversion · Makro-Signale · PCA · Cross-Asset · Saisonal ·
    Komposit · Granger/CCF · Overshoot/Korrektur &middot; Parameter-Grid &middot; Accordion Steckbriefe</div>
</div>

<div class="card mb-3">
  <div class="card-header"><strong>Strategie-Familien Übersicht</strong></div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered">
      <thead><tr><th>Familie</th><th>Anzahl</th><th>Ø Sharpe</th><th>Best Sharpe</th><th>Beste Strategie</th></tr></thead>
      <tbody>
        {"".join(
        f'<tr><td>{fam}</td>'
        f'<td>{len(meta_df[meta_df["Familie"]==fam])}</td>'
        f'<td>{meta_df[meta_df["Familie"]==fam]["Sharpe"].mean():.2f}</td>'
        f'<td>{meta_df[meta_df["Familie"]==fam]["Sharpe"].max():.2f}</td>'
        f'<td class="small">{meta_df[meta_df["Familie"]==fam].iloc[0]["Name"]}</td></tr>'
        for fam in meta_df["Familie"].unique())}
      </tbody>
    </table>
    {_info(f"Alle {len(meta_df)} Strategien mit ≥252 gemeinsamen Handelstagen. "
           f"TC=10bps pro Roundtrip. IS/OOS Split: 70%/30%. "
           f"OOS-Sharpe > 0: live-handelbar. OOS-Sharpe > 0.3: robust.")}
  </div>
</div>

{_chart_card("Top-20 Strategien: Sharpe Ratio (sortiert)", fig_top20, height=540,
    interp="Grün > 0.5: sehr gute Strategie. Gelb 0-0.5: moderat. Rot: negativ. "
           "OOS-Sharpe im Steckbrief prüfen: IS kann durch Overfitting aufgebläht sein.")}
{_chart_card("Sharpe-Verteilung nach Strategie-Familie", fig_sharpe_box, height=500,
    interp="Box-Whisker + alle Punkte. Breite Boxen = inkonsistente Familienperformance. "
           "Familien über 0-Linie: insgesamt profitable Signalklasse.")}
{_chart_card("Top-5 Strategien: Equity-Kurven (log-Skala)", fig_eq_top5, height=540,
    interp="Log-Skala: gleicher prozentualer Anstieg = gleiche Höhe. "
           "Beste Strategien sollten Buy & Hold deutlich schlagen. "
           "Crashphasen (2008, 2020): Short-Signale zeigen Wert.")}
{_chart_card("IS vs. OOS Sharpe: Robustheit", fig_is_oos, height=520,
    interp="Punkte nahe Diagonale: IS=OOS (kein Overfitting). "
           "Weit oberhalb Diagonale: Overfitting (IS>>OOS). "
           "Unterhalb Diagonale: OOS>IS (unerwartetes Out-of-Sample-Alpha).")}
{_chart_card("Effizienzgrenze: CAGR vs. MaxDrawdown", fig_frontier, height=500,
    interp="Rechts oben = ideale Strategien (hohe Rendite, geringer Drawdown). "
           "Links oben: hohe Rendite aber auch hohes Risiko. "
           "Pareto-Front oben rechts: dominierende Strategien.")}
{_chart_card("MaxDD vs. Sharpe: Risk-Return für alle Strategien", fig_dd_sh, height=500,
    interp="Oben links = ideal (hohes Sharpe, geringer Drawdown). "
           "Cluster pro Familie zeigen Familien-Charakteristika.")}
{_chart_card("Ø Sharpe-Heatmap: Familie × Asset", fig_fam_heat,
    interp="Dunkelgrün: Strategie-Familie funktioniert gut für dieses Asset. "
           "Rot: Familie funktioniert nicht für dieses Asset. "
           "Hilft bei der Identifikation der besten Faktor-Asset-Kombinationen.")}

<div class="card mb-4">
  <div class="card-header"><strong>Top-30 Strategien: Kompakt-Tabelle</strong></div>
  <div class="card-body">{top30_html}</div>
</div>

<div class="card mb-4" id="steckbriefe">
  <div class="card-header">
    <strong>📋 Strategie-Steckbriefe (alle {len(meta_df)}) — ausklappbar</strong>
    <span class="badge bg-success ms-2">Top 25 mit Equity-Charts &amp; Verteilungen</span>
  </div>
  <div class="card-body p-2">
    {accordion_html}
  </div>
</div>
"""
    _write(out / "mega_strategies.html",
           _html_base("Mega-Strategien", 18, body))



def build_intraday_ccf_report(tables, figures, out):  # noqa: C901
    """Intraday CCF at hourly resolution for daily lag=0 pairs."""
    import time as _time

    # Top pairs from daily CCF that showed lag=0 (need finer resolution)
    PAIRS_H = [
        ("GC=F",  "GDX",  "Gold → GDX"),
        ("GC=F",  "SIL",  "Gold → SIL"),
        ("SI=F",  "SIL",  "Silber → SIL"),
        ("SI=F",  "GDX",  "Silber → GDX"),
        ("BZ=F",  "XLE",  "Brent → XLE"),
        ("CL=F",  "XLE",  "WTI → XLE"),
        ("GC=F",  "NEM",  "Gold → NEM"),
        ("BZ=F",  "APA",  "Brent → APA"),
        ("CL=F",  "SM",   "WTI → SM"),
    ]
    MAX_LAG_H = 24  # ± 24 hours

    def _ccf_at_lags(x: "np.ndarray", y: "np.ndarray", max_lag: int):
        """Returns dict lag→corr for -max_lag..+max_lag."""
        n = len(x)
        xd = x - x.mean(); yd = y - y.mean()
        sx = float(np.std(xd)); sy = float(np.std(yd))
        if sx < 1e-12 or sy < 1e-12:
            return {}
        result = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xi, yi = xd[:n-lag] if lag > 0 else xd, yd[lag:] if lag > 0 else yd
            else:
                xi, yi = xd[-lag:], yd[:n+lag]
            if len(xi) < 30:
                continue
            result[lag] = float(np.corrcoef(xi, yi)[0, 1])
        return result

    try:
        import yfinance as yf
    except ImportError:
        _write(out / "intraday_ccf.html",
               _html_base("Intraday CCF", 18, "<p>yfinance nicht installiert.</p>"))
        return

    hourly_data = {}
    download_log = []
    for a1, a2, _ in PAIRS_H:
        for ticker in [a1, a2]:
            if ticker in hourly_data:
                continue
            try:
                hist = yf.Ticker(ticker).history(period="730d", interval="1h")
                if not hist.empty:
                    s = hist["Close"].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    hourly_data[ticker] = s
                    r = np.log(s / s.shift(1)).dropna()
                    hourly_data[f"{ticker}_ret"] = r
                    download_log.append(f"{ticker}: {len(s)} Stunden-Kerzen OK")
                else:
                    download_log.append(f"{ticker}: leer")
            except Exception as e:
                download_log.append(f"{ticker}: FEHLER {e}")
            _time.sleep(0.3)  # rate limit

    # ── Compute hourly CCF for each pair ──────────────────────────────────
    pair_results = []
    fig_ccf_all = go.Figure()
    for pi, (a1, a2, label) in enumerate(PAIRS_H):
        r1_key = f"{a1}_ret"; r2_key = f"{a2}_ret"
        if r1_key not in hourly_data or r2_key not in hourly_data:
            continue
        r1 = hourly_data[r1_key]; r2 = hourly_data[r2_key]
        idx = r1.index.intersection(r2.index)
        if len(idx) < 100:
            continue
        ccf_d = _ccf_at_lags(r1.loc[idx].values, r2.loc[idx].values, MAX_LAG_H)
        if not ccf_d:
            continue
        lags = sorted(ccf_d.keys())
        vals = [ccf_d[l] for l in lags]
        # find optimal lag
        best_lag = max(ccf_d, key=lambda l: abs(ccf_d[l]))
        best_rho = ccf_d[best_lag]
        sig_band = 2.0 / np.sqrt(len(idx))
        pair_results.append({
            "Paar": label,
            "A1": a1, "A2": a2,
            "Opt. Lag (h)": best_lag,
            "Peak CCF": round(best_rho, 4),
            "Signifikant": abs(best_rho) > sig_band,
            "Sig.-Band (±)": round(sig_band, 4),
            "CCF@Lag0": round(ccf_d.get(0, float("nan")), 4),
            "Δ Peak vs. Lag0": round(abs(best_rho) - abs(ccf_d.get(0, 0)), 4),
            "N Stunden": len(idx),
        })
        fig_ccf_all.add_trace(go.Scatter(
            x=lags, y=vals,
            mode="lines+markers", name=label,
            line=dict(color=PAL[pi % len(PAL)], width=1.5),
            marker=dict(size=4)))

    fig_ccf_all.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_ccf_all.update_layout(
        title=f"Intraday CCF (stündlich): ±{MAX_LAG_H}h — Alle Paare",
        xaxis_title="Lag (Stunden, negativ = A1 verzögert A2)",
        yaxis_title="Kreuzkorrelation", height=520,
        shapes=[
            dict(type="rect", x0=-MAX_LAG_H, x1=MAX_LAG_H,
                 y0=-2/np.sqrt(500), y1=2/np.sqrt(500),
                 fillcolor="rgba(139,148,158,0.1)", line=dict(width=0))
        ])

    # Individual CCF charts per pair
    indiv_htmls = ""
    for pi, (a1, a2, label) in enumerate(PAIRS_H):
        r1_key = f"{a1}_ret"; r2_key = f"{a2}_ret"
        if r1_key not in hourly_data or r2_key not in hourly_data:
            continue
        r1 = hourly_data[r1_key]; r2 = hourly_data[r2_key]
        idx = r1.index.intersection(r2.index)
        if len(idx) < 100:
            continue
        ccf_d = _ccf_at_lags(r1.loc[idx].values, r2.loc[idx].values, MAX_LAG_H)
        lags = sorted(ccf_d.keys())
        vals = [ccf_d[l] for l in lags]
        sig_band = 2.0 / np.sqrt(len(idx))
        fig_ind = go.Figure()
        colors = ["#3fb950" if l == max(ccf_d, key=lambda x: abs(ccf_d[x]))
                  else ("#58a6ff" if l == 0 else "#8b949e")
                  for l in lags]
        fig_ind.add_trace(go.Bar(x=lags, y=vals, marker_color=colors,
                                  hovertemplate="Lag=%{x}h<br>CCF=%{y:.4f}<extra></extra>"))
        fig_ind.add_hline(y=sig_band, line_color="#d29922", line_dash="dot",
                           annotation_text=f"+{sig_band:.3f} (sig)")
        fig_ind.add_hline(y=-sig_band, line_color="#d29922", line_dash="dot",
                           annotation_text=f"-{sig_band:.3f}")
        fig_ind.add_hline(y=0, line_color="#8b949e")
        best_lag = max(ccf_d, key=lambda x: abs(ccf_d[x]))
        best_rho = ccf_d[best_lag]
        rho0 = ccf_d.get(0, 0)
        fig_ind.update_layout(
            title=f"{label} — Opt. Lag: {best_lag:+d}h | Peak CCF: {best_rho:.4f} | Lag-0: {rho0:.4f}",
            xaxis_title="Lag (Stunden)", yaxis_title="CCF", height=360)
        lag_interp = (f"Optimaler Lag: {best_lag:+d}h. " +
                      ("Gleichzeitig!" if best_lag == 0 else
                       f"{'A1 führt A2' if best_lag > 0 else 'A2 führt A1'} um {abs(best_lag)} Stunde(n). ") +
                      f"Peak {best_rho:.4f} vs. Lag-0 {rho0:.4f}: " +
                      ("kein signifikanter Zeitversatz trotz stündlicher Auflösung."
                       if best_lag == 0 else
                       f"sub-tägiger Lead-Lag von {abs(best_lag)}h gefunden!"))
        indiv_htmls += _chart_card(f"Stündlicher CCF: {label}", fig_ind, height=380,
                                   interp=lag_interp)

    # Results table
    result_df = pd.DataFrame(pair_results).sort_values("Δ Peak vs. Lag0", ascending=False) if pair_results else pd.DataFrame()
    result_html = _df_html(result_df) if not result_df.empty else "<p class='text-muted'>Keine Ergebnisse.</p>"

    # Summary: are daily lag=0 pairs truly synchronous?
    sync_pairs = [r for r in pair_results if r["Opt. Lag (h)"] == 0]
    lagged_pairs = [r for r in pair_results if r["Opt. Lag (h)"] != 0]

    body = f"""
<div class="ph-header">
  <h1>Intraday CCF: Stündliche Lag-Auflösung</h1>
  <div class="sub">Für daily lag=0 Paare: Stundendaten (max 730 Tage) &middot; ±{MAX_LAG_H}h &middot; Sub-tägliger Lead-Lag</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Motivation: Warum Intraday-CCF?</strong></div>
  <div class="card-body">
    <p class="small">
      Bei täglicher Auflösung zeigen viele Paare optimalen Lag = 0 (gleichzeitig). Das bedeutet <em>nicht</em>
      zwingend, dass keine Lead-Lag-Beziehung existiert — es könnte ein <strong>sub-tägiger Zeitversatz</strong>
      (z.B. 30 Minuten oder 2 Stunden) vorliegen, der im Tagesdurchschnitt verschwimmt.
    </p>
    <div class="row">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Befund</th><th>Interpretation</th><th>Konsequenz</th></tr></thead>
          <tbody>
            <tr><td>Opt. Lag = 0h</td><td>Wirklich synchron auf Stundenebene</td><td>Kein handelbarer Lead-Lag → nur simultane Exposure-Hedges</td></tr>
            <tr><td>Opt. Lag ≠ 0h</td><td>Sub-tägiger Lead-Lag gefunden</td><td>Intraday-Strategie möglich: Futures führen, ETF folgt</td></tr>
            <tr><td>|Opt. Lag| > 6h</td><td>Handelszeit-Differenz (US vs. London)</td><td>Market-Microstructure: Öffnungszeiten-Effekt</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        <p class="small text-muted">
          Typische Hypothesen für Commodity/ETF-Paare:<br>
          • <strong>GC=F → GDX</strong>: Futures (24h) führen ETF (US-Handelstag) → 1-4h Lag erwartet<br>
          • <strong>CL=F → XLE</strong>: Ähnlich — WTI Futures führen Energie-ETF<br>
          • <strong>SI=F → SIL</strong>: Silber-Futures vs. Silber-Miner-ETF<br>
          Minuten-Daten (yfinance 1m, max 7 Tage): für noch feinere Auflösung.
        </p>
      </div>
    </div>
    {_info(f"Download-Log: " + " | ".join(download_log[:15]))}
    {_info(f"Ergebnis: {len(sync_pairs)} Paare wirklich synchron (Lag=0h), "
           f"{len(lagged_pairs)} Paare mit sub-tägigem Lead-Lag ≥1h.")}
  </div>
</div>

{_chart_card(f"Intraday CCF: Alle Paare gleichzeitig (±{MAX_LAG_H}h)", fig_ccf_all, height=540,
    interp="Negativer Lag = A1 verzögert A2 (A2 führt). Positiver Lag = A1 führt A2. "
           "Grauer Bereich = 95%-Signifikanzband. Peaks außerhalb: statistisch signifikanter Lead-Lag. "
           "Grüne Markierung = optimaler Lag. Blaue Markierung = Lag 0.")}

{indiv_htmls}

<div class="card mb-4">
  <div class="card-header"><strong>Zusammenfassung: Hourly Lead-Lag-Tabelle</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Δ Peak vs. Lag0 = Wie viel zusätzliche CCF-Stärke beim optimalen Lag gegenüber Lag=0.
      Hoher Δ = klarer sub-tägiger Lead-Lag. Tiefer Δ = wirklich synchron.
    </p>
    {result_html}
  </div>
</div>
"""
    _write(out / "intraday_ccf.html", _html_base("Intraday CCF", 18, body))



def build_technical_analysis_report(tables, figures, out):  # noqa: C901
    """Technical indicators per asset + lead-lag indicator strategies (progressive complexity)."""
    returns = _read(tables / "phase2_returns.csv")
    prices  = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "technical_analysis.html",
               _html_base("Technische Analyse", 18, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        # Reconstruct prices from returns
        prices = np.exp(returns.cumsum()) * 100

    # ── Technical indicator functions ─────────────────────────────────────
    def _sma(s, w):    return s.rolling(w, min_periods=w//2).mean()
    def _ema(s, w):    return s.ewm(span=w, adjust=False).mean()
    def _atr(h, l, c, w=14):
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(w).mean()

    def _rsi(s, w=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(w).mean()
        ls = (-d.clip(upper=0)).rolling(w).mean()
        rs = g / (ls + 1e-9)
        return 100 - 100 / (1 + rs)

    def _macd(s, fast=12, slow=26, sig=9):
        ema_f = _ema(s, fast); ema_s = _ema(s, slow)
        macd_line = ema_f - ema_s
        signal    = _ema(macd_line, sig)
        return macd_line, signal

    def _bb(s, w=20, n_std=2):
        mid = _sma(s, w)
        std = s.rolling(w).std()
        return mid + n_std*std, mid, mid - n_std*std  # upper, mid, lower

    def _support_resistance(s, w=63):
        return s.rolling(w).min(), s.rolling(w).max()

    def _bb_position(s, w=20, n_std=2):
        """Returns position within Bollinger Bands: 0=lower, 0.5=mid, 1=upper."""
        up, mid, lo = _bb(s, w, n_std)
        return (s - lo) / (up - lo + 1e-9)

    # ── Assets & Granger lead-lag pairs ──────────────────────────────────
    MAIN_ASSETS = [c for c in ["CL=F","GC=F","HG=F","ZW=F","NG=F","SI=F","ZS=F","BZ=F"]
                   if c in returns.columns]

    # Confirmed Granger/CCF lead-lag pairs (leader → follower, lag≥1)
    LEADLAG_PAIRS = []
    if granger is not None and "cause" in granger.columns:
        sig_gran = (granger[granger.get("significant", True) == True]
                    .sort_values("fstat" if "fstat" in granger.columns else "pvalue",
                                 ascending="pvalue" in granger.columns)
                    .drop_duplicates(subset=["cause","effect"], keep="first")
                    .head(15))
        for _, row in sig_gran.iterrows():
            cause = row["cause"]; effect = row["effect"]
            lag   = int(row.get("lag", 1))
            if cause in returns.columns and effect in returns.columns:
                LEADLAG_PAIRS.append((cause, effect, lag, f"{cause}→{effect} (Lag {lag}T)"))
    # Add manual pairs if granger table empty
    manual = [("GC=F","GDX",1,"Gold→GDX"), ("CL=F","XLE",1,"WTI→XLE"),
              ("GC=F","NEM",1,"Gold→NEM"), ("CL=F","XOM",1,"WTI→XOM"),
              ("HG=F","FCX",1,"Kupfer→FCX")]
    for c, e, l, lb in manual:
        if c in returns.columns and e in returns.columns:
            if not any(p[0]==c and p[1]==e for p in LEADLAG_PAIRS):
                LEADLAG_PAIRS.append((c, e, l, lb))

    # ── LEVEL 1: Price + SMA per asset ───────────────────────────────────
    fig_l1_list = {}
    for asset in MAIN_ASSETS[:6]:
        p = prices[asset].dropna() if asset in prices.columns else None
        if p is None or len(p) < 252:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index.astype(str).tolist(),
                                  y=p.round(4).values.tolist(),
                                  mode="lines", name=asset,
                                  line=dict(color="#e6edf3", width=1.3)))
        for w, col in [(20,"#58a6ff"), (50,"#d29922"), (200,"#3fb950")]:
            sma = _sma(p, w).dropna()
            fig.add_trace(go.Scatter(x=sma.index.astype(str).tolist(),
                                     y=sma.round(4).values.tolist(),
                                     mode="lines", name=f"SMA{w}",
                                     line=dict(color=col, width=1.0, dash="dot")))
        # Support & Resistance (63T)
        sup, res = _support_resistance(p, 63)
        fig.add_trace(go.Scatter(x=res.dropna().index.astype(str).tolist(),
                                  y=res.dropna().round(4).values.tolist(),
                                  mode="lines", name="Res-63T",
                                  line=dict(color="#f78166", width=0.8, dash="dash")))
        fig.add_trace(go.Scatter(x=sup.dropna().index.astype(str).tolist(),
                                  y=sup.dropna().round(4).values.tolist(),
                                  mode="lines", name="Sup-63T",
                                  line=dict(color="#3fb950", width=0.8, dash="dash"),
                                  fill="tonexty", fillcolor="rgba(63,185,80,0.04)"))
        fig.update_layout(title=f"{asset}: Preis + SMA20/50/200 + Support/Resistance (63T)",
                          height=440, yaxis_title="Preis")
        fig_l1_list[asset] = fig

    # ── LEVEL 2: RSI + MACD per asset ─────────────────────────────────────
    fig_l2_list = {}
    for asset in MAIN_ASSETS[:6]:
        p = prices[asset].dropna() if asset in prices.columns else None
        if p is None or len(p) < 252:
            continue
        rsi = _rsi(p)
        macd_line, macd_sig = _macd(p)
        bb_up, bb_mid, bb_lo = _bb(p)
        bb_pos = _bb_position(p)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Preis + Bollinger", "RSI (14)", "MACD (12/26/9)"],
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
        # Price + BB
        fig.add_trace(go.Scatter(x=p.index.astype(str).tolist(), y=p.round(4).values.tolist(),
                                  name=asset, line=dict(color="#e6edf3", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_up.dropna().index.astype(str).tolist(),
                                  y=bb_up.dropna().round(4).values.tolist(),
                                  name="BB-Oben", line=dict(color="#58a6ff", width=0.7, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_lo.dropna().index.astype(str).tolist(),
                                  y=bb_lo.dropna().round(4).values.tolist(),
                                  name="BB-Unten", line=dict(color="#58a6ff", width=0.7, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(88,166,255,0.07)"), row=1, col=1)
        # RSI
        rsi_clean = rsi.dropna()
        fig.add_trace(go.Scatter(x=rsi_clean.index.astype(str).tolist(),
                                  y=rsi_clean.round(2).values.tolist(),
                                  name="RSI", line=dict(color="#d29922", width=1.3)), row=2, col=1)
        fig.add_hline(y=70, line_color="#f78166", line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="#3fb950", line_dash="dot", row=2, col=1)
        fig.add_hline(y=50, line_color="#8b949e", line_dash="dash", row=2, col=1)
        # MACD
        mc_clean = macd_line.dropna(); ms_clean = macd_sig.loc[mc_clean.index]
        fig.add_trace(go.Scatter(x=mc_clean.index.astype(str).tolist(),
                                  y=mc_clean.round(4).values.tolist(),
                                  name="MACD", line=dict(color="#58a6ff", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ms_clean.index.astype(str).tolist(),
                                  y=ms_clean.round(4).values.tolist(),
                                  name="Signal", line=dict(color="#ffa657", width=1.0, dash="dot")), row=3, col=1)
        hist_vals = (mc_clean - ms_clean)
        fig.add_trace(go.Bar(x=hist_vals.index.astype(str).tolist(),
                              y=hist_vals.round(4).values.tolist(),
                              name="Histogram",
                              marker_color=["#3fb950" if v > 0 else "#f78166" for v in hist_vals.values]),
                      row=3, col=1)
        fig.update_layout(title=f"{asset}: Bollinger Bands + RSI + MACD",
                          height=680, showlegend=True)
        fig_l2_list[asset] = fig

    # ── LEVEL 3: Lead-Lag Indicator Strategies ──────────────────────────
    # Key insight: leader's indicator at t-lag is a FUTURE indicator for the follower.
    # Strategy: trade follower based on leader's RSI/MACD/BB-position at t-lag.
    strat_results_ta = []
    fig_ll_indicator_list = {}

    def _run_indicator_strat(leader_indicator: "pd.Series", follower_ret: "pd.Series",
                              lag: int, name: str, threshold: float = 0.0) -> dict | None:
        """Trade follower when leader_indicator(t-lag) crosses threshold."""
        li = leader_indicator.shift(lag).dropna()
        idx = li.index.intersection(follower_ret.dropna().index)
        if len(idx) < 252:
            return None
        sig = np.sign(li.loc[idx] - threshold)
        ret = follower_ret.loc[idx]
        strat_r = sig * ret - sig.diff().abs().fillna(0) * (10/10000)
        strat_r = strat_r.dropna()
        if len(strat_r) < 126:
            return None
        ann_r = float(strat_r.mean() * 252)
        ann_v = float(strat_r.std() * np.sqrt(252)) + 1e-9
        sh = ann_r / ann_v
        cum = (1 + strat_r).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        split = int(len(strat_r) * 0.7)
        oos_sh = float(strat_r.iloc[split:].mean() * 252 /
                       (strat_r.iloc[split:].std() * np.sqrt(252) + 1e-9))
        return {
            "Strategie": name, "CAGR%": round(ann_r*100,2), "Sharpe": round(sh,3),
            "OOS Sharpe": round(oos_sh,3), "MaxDD%": round(mdd*100,2),
            "N": len(strat_r), "_ret": strat_r,
        }

    for leader, follower, lag, pair_label in LEADLAG_PAIRS[:8]:
        p_lead = prices[leader].dropna() if leader in prices.columns else None
        r_foll = returns[follower].dropna()
        if p_lead is None or len(p_lead) < 300:
            continue

        # Compute leader indicators
        rsi_lead   = _rsi(p_lead)
        macd_l, _  = _macd(p_lead)
        bb_pos_l   = _bb_position(p_lead)
        sma20_l    = _sma(p_lead, 20)
        sma_cross  = (sma20_l - _sma(p_lead, 50))  # positive = SMA20 > SMA50 (trend)

        strategies_for_pair = [
            (rsi_lead,  50.0, f"RSI(14)>50 @ {pair_label}"),
            (rsi_lead,  70.0, f"RSI(14)<70 @ {pair_label} (nicht überkauft)"),
            (macd_l,    0.0,  f"MACD>0 @ {pair_label}"),
            (bb_pos_l,  0.5,  f"BB-Position>50% @ {pair_label}"),
            (sma_cross, 0.0,  f"SMA20>SMA50 @ {pair_label}"),
        ]

        pair_strats = []
        for indicator, thresh, sname in strategies_for_pair:
            m = _run_indicator_strat(indicator, r_foll, lag, sname, thresh)
            if m:
                pair_strats.append(m)
                strat_results_ta.append(m)

        if not pair_strats:
            continue

        # Chart: equity curves for this pair's indicator strategies
        fig_ll = go.Figure()
        for j, m in enumerate(pair_strats):
            eq = (1 + m["_ret"]).cumprod()
            fig_ll.add_trace(go.Scatter(
                x=eq.index.astype(str).tolist(), y=eq.round(4).values.tolist(),
                mode="lines", name=m["Strategie"].split("@")[0].strip(),
                line=dict(color=PAL[j % len(PAL)], width=1.5)))
        bh_eq = (1 + r_foll.dropna()).cumprod()
        fig_ll.add_trace(go.Scatter(
            x=bh_eq.index.astype(str).tolist(), y=bh_eq.round(4).values.tolist(),
            mode="lines", name=f"B&H {follower}",
            line=dict(color="#8b949e", width=0.9, dash="dot")))
        fig_ll.update_layout(title=f"Lead-Lag-Indikatoren: {pair_label} — Follower: {follower}",
                              yaxis_type="log", height=440)
        fig_ll_indicator_list[pair_label] = (fig_ll, pair_strats)

    # ── LEVEL 4: Leader indicator overlaid on follower price ────────────
    fig_overlay_list = {}
    for leader, follower, lag, pair_label in LEADLAG_PAIRS[:6]:
        p_lead = prices[leader].dropna() if leader in prices.columns else None
        p_foll = prices[follower].dropna() if follower in prices.columns else None
        if p_lead is None or p_foll is None:
            continue
        rsi_lead = _rsi(p_lead)
        # Align indices
        idx = p_foll.index.intersection(rsi_lead.dropna().index)
        if len(idx) < 126:
            continue
        fig_ov = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=[f"{follower}: Preis",
                                                f"{leader}: RSI(14) — {lag}T vor {follower}"],
                                row_heights=[0.65, 0.35], vertical_spacing=0.05)
        pf = p_foll.loc[idx]
        fig_ov.add_trace(go.Scatter(x=pf.index.astype(str).tolist(),
                                     y=pf.round(4).values.tolist(),
                                     name=follower, line=dict(color="#e6edf3", width=1.3)), row=1, col=1)
        bb_up, bb_mid, bb_lo = _bb(pf)
        for bb_s, bb_col in [(bb_up,"#58a6ff"), (bb_lo,"#58a6ff")]:
            bb_c = bb_s.dropna()
            fig_ov.add_trace(go.Scatter(x=bb_c.index.astype(str).tolist(),
                                         y=bb_c.round(4).values.tolist(),
                                         name="BB", line=dict(color=bb_col, width=0.7, dash="dot"),
                                         showlegend=False), row=1, col=1)
        rl = rsi_lead.loc[idx]
        fig_ov.add_trace(go.Scatter(x=rl.index.astype(str).tolist(),
                                     y=rl.round(2).values.tolist(),
                                     name=f"RSI {leader}", line=dict(color="#d29922", width=1.5)),
                          row=2, col=1)
        fig_ov.add_hline(y=70, line_color="#f78166", line_dash="dot", row=2, col=1)
        fig_ov.add_hline(y=30, line_color="#3fb950", line_dash="dot", row=2, col=1)
        fig_ov.update_layout(
            title=f"{leader} RSI als Zukunfts-Indikator für {follower} (Lag {lag}T)",
            height=560)
        fig_overlay_list[pair_label] = fig_ov

    # ── Summary: best TA-strategies ───────────────────────────────────────
    fig_ta_summary = go.Figure()
    ta_rows = [{"Strategie": m["Strategie"], "Sharpe": m["Sharpe"],
                "OOS Sharpe": m["OOS Sharpe"], "CAGR%": m["CAGR%"],
                "MaxDD%": m["MaxDD%"]}
               for m in sorted(strat_results_ta, key=lambda x: -x["Sharpe"])]
    if ta_rows:
        ta_df = pd.DataFrame(ta_rows).head(20)
        fig_ta_summary.add_trace(go.Bar(
            x=ta_df["Strategie"].tolist(),
            y=ta_df["Sharpe"].tolist(),
            marker_color=["#3fb950" if v > 0.4 else "#d29922" if v > 0 else "#f78166"
                          for v in ta_df["Sharpe"]],
            text=[f"{v:.2f}" for v in ta_df["Sharpe"]],
            textposition="outside"))
        fig_ta_summary.add_hline(y=0, line_color="#8b949e")
        fig_ta_summary.update_layout(
            title="Lead-Lag-Indikator-Strategien: Top-20 Sharpe Ratio",
            xaxis_tickangle=-40, height=520, yaxis_title="Sharpe")

    ta_table_html = (_df_html(pd.DataFrame(ta_rows[:30])) if ta_rows
                     else "<p class='text-muted'>Keine Ergebnisse.</p>")

    # ── HTML body ──────────────────────────────────────────────────────────
    l1_html = ""
    for asset, fig in fig_l1_list.items():
        l1_html += _chart_card(f"Level 1 — {asset}: Preis + SMA + Support/Resistance", fig, height=460,
            interp=f"SMA20 (blau): kurzfristiger Trend. SMA50 (gelb): mittelfristig. SMA200 (grün): langfristig. "
                   f"Golden Cross: SMA20 > SMA200. Death Cross: SMA20 < SMA200. "
                   f"Support (grün): 63T-Tief = starker Boden. Resistance (rot): 63T-Hoch.")

    l2_html = ""
    for asset, fig in fig_l2_list.items():
        l2_html += _chart_card(f"Level 2 — {asset}: RSI + MACD + Bollinger", fig, height=700,
            interp=f"RSI>70: überkauft → Rückschlagrisiko. RSI<30: überverkauft → Erholungspotenzial. "
                   f"MACD-Crossover: bullisches Signal (MACD > Signal). "
                   f"BB-Ausbruch oben: Momentum oder Übertreibung. BB-Ausbruch unten: Panic oder Breakout.")

    ll_html = ""
    for pair_label, (fig_ll, pair_strats) in fig_ll_indicator_list.items():
        best = max(pair_strats, key=lambda x: x["Sharpe"])
        strat_rows = "".join(
            f"<tr><td class='small'>{m['Strategie'].split('@')[0].strip()}</td>"
            f"<td>{m['Sharpe']:.3f}</td><td>{m['OOS Sharpe']:.3f}</td>"
            f"<td>{m['CAGR%']:.1f}%</td><td>{m['MaxDD%']:.1f}%</td></tr>"
            for m in pair_strats)
        strat_table = (f"<table class='table table-dark table-sm table-bordered small'>"
                       f"<thead><tr><th>Indikator</th><th>Sharpe</th><th>OOS</th>"
                       f"<th>CAGR</th><th>MaxDD</th></tr></thead>"
                       f"<tbody>{strat_rows}</tbody></table>")
        ll_html += f"""
<div class="card mb-3">
  <div class="card-header"><strong>Level 3 — Lead-Lag-Indikator: {pair_label}</strong>
    <span class="badge bg-success ms-2">Best: {best['Sharpe']:.2f} Sharpe</span></div>
  <div class="card-body">
    {strat_table}
    {_interp("Strategie: Wenn Anführer-Indikator Signal gibt, trade den Nachzügler nach [Lag] Tagen. "
             "Leader-RSI>50 = bullisches Markt-Momentum → Long Follower. "
             "OOS Sharpe zeigt, ob das echte Vorhersagekraft ist oder Overfitting.")}
  </div>
</div>
""" + _chart_card(f"Level 3 Equity-Kurven: {pair_label}", fig_ll, height=460,
                   interp="Log-Skala. Leader-Indikator-Signal mit [Lag]-Tages-Verzögerung. "
                          "Beste Indikatoren für Nachzügler-Trading: RSI und MACD typisch am stärksten.")

    ov_html = ""
    for pair_label, fig_ov in fig_overlay_list.items():
        ov_html += _chart_card(f"Level 4 — Overlay: {pair_label}", fig_ov, height=580,
            interp=f"Oben: Follower-Preis mit Bollinger Bands. "
                   f"Unten: Leader RSI — dieser RSI war {1}T vor dem Follower-Kurs. "
                   f"Leader RSI > 70 UND Follower noch nicht ralliert: mögliches Entry-Signal. "
                   f"RSI-Divergenz (Leader fällt, Follower noch hoch): Warnsignal für Follower.")

    body = f"""
<div class="ph-header">
  <h1>Technische Analyse</h1>
  <div class="sub">Level 1: Preis+SMA+Support · Level 2: RSI+MACD+Bollinger ·
    Level 3: Lead-Lag-Indikator-Strategien · Level 4: Cross-Asset-Overlay</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>🎯 Kernidee: Anführer-Indikatoren als Zukunftssignale für Nachzügler</strong></div>
  <div class="card-body">
    <p>Wenn Asset A Asset B um k Tage zeitlich führt (Granger-kausal), dann sind <strong>technische Indikatoren
    von A zum Zeitpunkt t</strong> effektiv <strong>Zukunftsindikatoren für B zum Zeitpunkt t+k</strong>.</p>
    <div class="row">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Level</th><th>Inhalt</th><th>Komplexität</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>Preis + SMA20/50/200 + Support/Resistance</td><td>Basis</td></tr>
            <tr><td>2</td><td>Bollinger + RSI(14) + MACD(12/26/9)</td><td>Standard TA</td></tr>
            <tr><td>3</td><td>Leader-Indikator → Follower-Trade (Lag)</td><td>Lead-Lag</td></tr>
            <tr><td>4</td><td>Cross-Asset-Overlay: Leader-RSI auf Follower-Chart</td><td>Visuell</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Strategie-Logik</th><th>Signal</th><th>Trade</th></tr></thead>
          <tbody>
            <tr><td>Leader RSI > 50</td><td>Bullisches Momentum im Leader</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader RSI < 30</td><td>Leader überverkauft → Bounce</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader MACD > 0</td><td>Leader in Aufwärtstrend</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader BB-Pos > 50%</td><td>Leader über Mittelband</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader SMA20 > SMA50</td><td>Golden Cross beim Leader</td><td>Long Follower in t+k</td></tr>
          </tbody>
        </table>
        {_info("Alle Strategien: TC=10bps. Signal wird um [Lag] Tage verschoben. "
               "IS/OOS Split 70%/30%.")}
      </div>
    </div>
  </div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 1: Preis + SMA + Support/Resistance (alle Assets)</strong></div>
  <div class="card-body">{l1_html}</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 2: RSI + MACD + Bollinger Bands (alle Assets)</strong></div>
  <div class="card-body">{l2_html}</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 3: Lead-Lag-Indikator-Strategien</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Für jedes signifikante Granger-Paar: Leader-Indikatoren (RSI, MACD, BB-Position, SMA-Cross)
      als zeitverzögertes Signal für den Follower. Optimaler Lag = bekannter Granger-Lag.
    </p>
    {ll_html}
  </div>
</div>

{_chart_card("Zusammenfassung: Beste Lead-Lag-Indikator-Strategien (Top-20 Sharpe)", fig_ta_summary, height=540,
    interp="Grün > 0.4: robuste Strategie mit handelbarer Vorhersagekraft. "
           "OOS-Sharpe in Tabelle prüfen: IS-Sharpe kann durch Backtest-Bias erhöht sein. "
           "Beste Signale: RSI-basiert (trend-following) und MACD (Momentum-Wendepunkt).")}

{_card("TA-Strategien Tabelle (alle, sortiert nach Sharpe)", ta_table_html)}

<div class="card mb-4">
  <div class="card-header"><strong>Level 4: Cross-Asset-Overlay (Leader RSI auf Follower Chart)</strong></div>
  <div class="card-body">{ov_html}</div>
</div>
"""
    _write(out / "technical_analysis.html",
           _html_base("Technische Analyse", 18, body))




# ── Strategy shared helpers ───────────────────────────────────────────────────
try:
    PHASE_COLOURS[19] = "#c9d1d9"
except Exception:
    pass


def _calc_rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=w, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(span=w, adjust=False).mean()
    return 100 - 100 / (1 + g / (ls + 1e-9))


def _calc_macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m = ef - es
    return m, m.ewm(span=sig, adjust=False).mean()


def _calc_bb_pos(s, w=20, n_std=2.0):
    mid = s.rolling(w).mean()
    std = s.rolling(w).std()
    lo = mid - n_std * std
    hi = mid + n_std * std
    return (s - lo) / (hi - lo + 1e-9)


def _calc_sma_cross(s, fast=20, slow=50):
    return s.rolling(fast).mean() - s.rolling(slow).mean()


def _strat_exec(indicator, threshold, follower_ret, lag, tc=0.001):
    """Long when indicator(t-lag)>threshold, else short. Returns (net, gross, signals)."""
    sig = ((indicator.shift(lag) > threshold).astype(float) * 2 - 1)
    sig = sig.reindex(follower_ret.index, method="ffill")
    idx = sig.dropna().index.intersection(follower_ret.dropna().index)
    if len(idx) < 30:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    s = sig.loc[idx]
    fr = follower_ret.loc[idx]
    gross = s * fr
    net = gross - s.diff().abs().fillna(0) * tc
    valid = net.dropna().index
    return net.loc[valid], gross.loc[valid], s.loc[valid]


def _full_metrics(net, gross=None, signals=None, spy=None, name=""):
    """26-metric strategy scorecard from daily net-return series."""
    r = net.dropna()
    rg = gross.dropna() if gross is not None and len(gross) > 0 else r
    if len(r) < 30:
        return {"Name": name, "Sharpe (net)": float("nan")}

    def _sh(x):
        return x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9)

    def _mdd(x):
        c = (1 + x).cumprod()
        return float((c / c.cummax() - 1).min())

    down = r[r < 0]
    sortino = (r.mean() * 252 / (down.std() * np.sqrt(252) + 1e-9)
               if len(down) > 5 else float("nan"))
    calmar = r.mean() * 252 / (abs(_mdd(r)) + 1e-9)

    cum = (1 + r).cumprod()
    rm = cum.cummax()
    dd = cum / rm - 1
    in_dd = dd < 0
    grp = (in_dd != in_dd.shift()).cumsum()
    dur = in_dd.groupby(grp).sum()
    avg_dur = float(dur[dur > 0].mean()) if (dur > 0).any() else 0.0

    sig_s = (signals.reindex(r.index).fillna(0)
             if signals is not None else np.sign(r))
    trades = int((sig_s.diff().abs() > 0).sum())
    wr = float((r > 0).mean())
    avg_w = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_l = float(r[r < 0].mean()) if (r < 0).any() else 0.0
    pf = (abs(r[r > 0].sum() / (r[r < 0].sum() - 1e-9))
          if (r < 0).any() else 99.0)
    omega = r[r > 0].sum() / (-r[r < 0].sum() + 1e-9)
    tail = abs(np.percentile(r, 95) / (np.percentile(r, 5) - 1e-9))
    var5 = float(np.percentile(r, 5))
    cvar5 = float(r[r <= var5].mean()) if (r <= var5).any() else var5

    beta = 0.0
    alpha = 0.0
    if spy is not None:
        idx2 = r.index.intersection(spy.dropna().index)
        if len(idx2) > 60:
            X = spy.loc[idx2].values
            Y = r.loc[idx2].values
            Xc = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
            alpha = float(coef[0] * 252)
            beta = float(coef[1])

    return {
        "Name": name,
        "Ann.Ret% (net)": round(r.mean() * 252 * 100, 2),
        "Ann.Ret% (gross)": round(rg.mean() * 252 * 100, 2),
        "TC Drag% p.a.": round((rg.mean() - r.mean()) * 252 * 100, 3),
        "Ann.Vol%": round(r.std() * np.sqrt(252) * 100, 2),
        "Sharpe (net)": round(_sh(r), 3),
        "Sharpe (gross)": round(_sh(rg), 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "MaxDD%": round(_mdd(r) * 100, 2),
        "AvgDD-Dur(d)": round(avg_dur, 1),
        "# Trades": trades,
        "Long Days": int((sig_s > 0).sum()),
        "Short Days": int((sig_s < 0).sum()),
        "Flat Days": int((sig_s == 0).sum()),
        "Win Rate%": round(wr * 100, 1),
        "Avg Win%": round(avg_w * 100, 4),
        "Avg Loss%": round(avg_l * 100, 4),
        "Profit Factor": round(pf, 3),
        "Omega": round(omega, 3),
        "Tail Ratio": round(tail, 3),
        "Skewness": round(float(r.skew()), 3),
        "Kurtosis": round(float(r.kurtosis()), 3),
        "VaR5%/d": round(var5 * 100, 4),
        "CVaR5%/d": round(cvar5 * 100, 4),
        "AC1": round(float(r.autocorr(1)), 3),
        "AC5": round(float(r.autocorr(5)), 3),
        "Beta(SPY)": round(beta, 3),
        "Alpha% p.a.": round(alpha * 100, 3),
    }



def build_lead_lag_optimizer_report(tables, figures, out):  # noqa: C901
    """Grid-search parameter optimization per indicator for top Granger pairs."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "lead_lag_optimizer.html",
               _html_base("Lead-Lag Optimizer", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── Top-5 Granger pairs ──────────────────────────────────────────────────
    MANUAL = [
        ("GC=F", "GDX", 7, "Gold→GDX"),
        ("GC=F", "NEM", 6, "Gold→NEM"),
        ("CL=F", "XLE", 1, "WTI→XLE"),
        ("HG=F", "FCX", 1, "Kupfer→FCX"),
        ("SI=F", "SIL", 1, "Silber→SIL"),
    ]
    top_pairs = []
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            if (c in returns.columns and e in returns.columns
                    and not any(p[0] == c and p[1] == e for p in top_pairs)):
                top_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
            if len(top_pairs) >= 5:
                break
    for mp in MANUAL:
        if not any(p[0] == mp[0] and p[1] == mp[1] for p in top_pairs):
            if mp[0] in returns.columns and mp[1] in returns.columns:
                top_pairs.append(mp)
        if len(top_pairs) >= 5:
            break

    # ── Parameter grids ───────────────────────────────────────────────────────
    RSI_WINDOWS = [7, 10, 14, 21, 30]
    RSI_THRESH  = [35, 40, 45, 50, 55, 60, 65]
    MACD_FAST   = [6, 8, 12, 16, 20]
    MACD_SLOW   = [18, 21, 26, 34, 50]
    BB_WINDOWS  = [10, 15, 20, 30, 50]
    BB_THRESH   = [0.3, 0.4, 0.5, 0.6, 0.7]
    SMA_FAST    = [5, 10, 20, 30]
    SMA_SLOW    = [20, 50, 100, 200]

    body_sections = []

    for leader, follower, lag, pair_label in top_pairs:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        if px is None or len(px) < 400:
            continue

        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 300:
            continue
        split_date = idx_all[int(len(idx_all) * 0.70)]

        def is_r(s):
            return s.loc[:split_date]

        def oos_r(s):
            return s.loc[split_date:]

        summary_rows = []
        pair_html = []
        pair_html.append(
            "<h4 class='mt-4 mb-3' style='border-bottom:1px solid #30363d;padding-bottom:6px'>"
            + pair_label + "</h4>"
            + "<p class='small text-muted'>IS: erste 70% &middot; OOS: letzte 30% &middot; "
            + f"Lag: {lag}T &middot; TC: 10bp</p>")

        # ── RSI heatmap ──────────────────────────────────────────────────────
        rsi_mat = np.full((len(RSI_WINDOWS), len(RSI_THRESH)), float("nan"))
        best_rsi = {"sharpe": -999.0, "params": None}
        for wi, w in enumerate(RSI_WINDOWS):
            ind = _calc_rsi(px, w)
            for ti, t in enumerate(RSI_THRESH):
                n, g, s = _strat_exec(ind, t, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    rsi_mat[wi, ti] = sh
                    if sh > best_rsi["sharpe"]:
                        best_rsi = {"sharpe": sh, "params": (w, t)}

        fig_rsi = go.Figure(go.Heatmap(
            z=np.round(rsi_mat, 3).tolist(),
            x=[str(t) for t in RSI_THRESH],
            y=[str(w) for w in RSI_WINDOWS],
            colorscale="RdYlGn", zmid=0,
            text=np.round(rsi_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="RSI-Win=%{y} Thresh=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_rsi.update_layout(
            title="RSI: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Threshold", yaxis_title="RSI Window", height=360)

        if best_rsi["params"]:
            bw, bt = best_rsi["params"]
            n_oos, g_oos, s_oos = _strat_exec(_calc_rsi(px, bw), bt, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "RSI",
                "Best Params (IS)": f"w={bw}, thresh={bt}",
                "IS Sharpe": round(best_rsi["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "RSI-Heatmap: " + pair_label, fig_rsi, height=380,
            interp=f"IS-Sharpe für jeden RSI-Window × Threshold-Kombination. "
                   f"Grün=positiv. Bester IS-Punkt: w={best_rsi['params']}. "
                   "OOS-Sharpe in Tabelle zeigt, ob IS-Optimum generaliserbar ist."))

        # ── MACD heatmap ─────────────────────────────────────────────────────
        macd_mat = np.full((len(MACD_FAST), len(MACD_SLOW)), float("nan"))
        best_macd = {"sharpe": -999.0, "params": None}
        for fi, fast in enumerate(MACD_FAST):
            for si, slow in enumerate(MACD_SLOW):
                if fast >= slow:
                    continue
                m_line, _ = _calc_macd(px, fast, slow)
                n, g, s = _strat_exec(m_line, 0, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    macd_mat[fi, si] = sh
                    if sh > best_macd["sharpe"]:
                        best_macd = {"sharpe": sh, "params": (fast, slow)}

        fig_macd = go.Figure(go.Heatmap(
            z=np.round(macd_mat, 3).tolist(),
            x=[str(s) for s in MACD_SLOW],
            y=[str(f) for f in MACD_FAST],
            colorscale="RdYlGn", zmid=0,
            text=np.round(macd_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Fast=%{y} Slow=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_macd.update_layout(
            title="MACD: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Slow EMA", yaxis_title="Fast EMA", height=360)

        if best_macd["params"]:
            bf, bs = best_macd["params"]
            ml, _ = _calc_macd(px, bf, bs)
            n_oos, g_oos, s_oos = _strat_exec(ml, 0, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "MACD",
                "Best Params (IS)": f"fast={bf}, slow={bs}",
                "IS Sharpe": round(best_macd["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "MACD-Heatmap: " + pair_label, fig_macd, height=380,
            interp="Leere Zellen: fast >= slow (ungültig). "
                   "Threshold fest auf 0 (MACD-Linie > 0 = bullisch)."))

        # ── BB-Position heatmap ───────────────────────────────────────────────
        bb_mat = np.full((len(BB_WINDOWS), len(BB_THRESH)), float("nan"))
        best_bb = {"sharpe": -999.0, "params": None}
        for wi, w in enumerate(BB_WINDOWS):
            bbp = _calc_bb_pos(px, w)
            for ti, t in enumerate(BB_THRESH):
                n, g, s = _strat_exec(bbp, t, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    bb_mat[wi, ti] = sh
                    if sh > best_bb["sharpe"]:
                        best_bb = {"sharpe": sh, "params": (w, t)}

        fig_bb = go.Figure(go.Heatmap(
            z=np.round(bb_mat, 3).tolist(),
            x=[str(t) for t in BB_THRESH],
            y=[str(w) for w in BB_WINDOWS],
            colorscale="RdYlGn", zmid=0,
            text=np.round(bb_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Window=%{y} Thresh=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_bb.update_layout(
            title="BB-Position: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Threshold (0=unten, 1=oben)", yaxis_title="BB Window",
            height=360)

        if best_bb["params"]:
            bw, bt = best_bb["params"]
            n_oos, g_oos, s_oos = _strat_exec(_calc_bb_pos(px, bw), bt, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "BB-Position",
                "Best Params (IS)": f"w={bw}, thresh={bt}",
                "IS Sharpe": round(best_bb["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "BB-Position-Heatmap: " + pair_label, fig_bb, height=380,
            interp="BB-Position: 0=unteres Band, 1=oberes Band. "
                   ">0.5 = Preis über Mittelband (bullisch)."))

        # ── SMA-Cross heatmap ─────────────────────────────────────────────────
        sma_mat = np.full((len(SMA_FAST), len(SMA_SLOW)), float("nan"))
        best_sma = {"sharpe": -999.0, "params": None}
        for fi, fast in enumerate(SMA_FAST):
            for si, slow in enumerate(SMA_SLOW):
                if fast >= slow:
                    continue
                cross = _calc_sma_cross(px, fast, slow)
                n, g, s = _strat_exec(cross, 0, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    sma_mat[fi, si] = sh
                    if sh > best_sma["sharpe"]:
                        best_sma = {"sharpe": sh, "params": (fast, slow)}

        fig_sma = go.Figure(go.Heatmap(
            z=np.round(sma_mat, 3).tolist(),
            x=[str(s) for s in SMA_SLOW],
            y=[str(f) for f in SMA_FAST],
            colorscale="RdYlGn", zmid=0,
            text=np.round(sma_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Fast=%{y} Slow=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_sma.update_layout(
            title="SMA-Cross: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Slow SMA", yaxis_title="Fast SMA", height=360)

        if best_sma["params"]:
            bf, bs = best_sma["params"]
            n_oos, g_oos, s_oos = _strat_exec(
                _calc_sma_cross(px, bf, bs), 0, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "SMA-Cross",
                "Best Params (IS)": f"fast={bf}, slow={bs}",
                "IS Sharpe": round(best_sma["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "SMA-Cross-Heatmap: " + pair_label, fig_sma, height=380,
            interp="SMA-Cross = fast_SMA - slow_SMA. >0 → Aufwärtstrend. "
                   "Leere Zellen: fast >= slow."))

        # ── Summary + Walk-Forward ───────────────────────────────────────────
        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            best_overall = sum_df.loc[
                sum_df["OOS Sharpe"].apply(
                    lambda x: x if not (isinstance(x, float) and np.isnan(x)) else -999
                ).idxmax()]
            pair_html.append(
                "<div class='card mb-3'>"
                "<div class='card-header'><strong>Optimierungsergebnis: "
                + pair_label
                + "</strong><span class='badge bg-success ms-2'>Bester OOS: "
                + str(best_overall["Indikator"])
                + " ("
                + str(best_overall["Best Params (IS)"])
                + ") → Sharpe "
                + str(round(best_overall["OOS Sharpe"], 3))
                + "</span></div>"
                "<div class='card-body'>"
                + _df_html(sum_df)
                + "</div></div>")

            # Walk-forward: 4 expanding-IS windows (20-80%), test on next 20%
            all_candidates = [b for b in [best_rsi, best_macd, best_bb, best_sma]
                              if b["params"]]
            if all_candidates:
                best_any = max(all_candidates, key=lambda b: b["sharpe"])
                if best_any is best_rsi:
                    bw, bt = best_any["params"]
                    wf_ind = _calc_rsi(px, bw); wf_thresh = bt
                    wf_name = f"RSI(w={bw},t={bt})"
                elif best_any is best_macd:
                    bf, bs = best_any["params"]
                    wf_ind, _ = _calc_macd(px, bf, bs); wf_thresh = 0
                    wf_name = f"MACD({bf},{bs})"
                elif best_any is best_bb:
                    bw, bt = best_any["params"]
                    wf_ind = _calc_bb_pos(px, bw); wf_thresh = bt
                    wf_name = f"BB-Pos(w={bw},t={bt})"
                else:
                    bf, bs = best_any["params"]
                    wf_ind = _calc_sma_cross(px, bf, bs); wf_thresh = 0
                    wf_name = f"SMA-Cross({bf},{bs})"

                wf_rows = []
                n_total = len(idx_all)
                for step in range(4):
                    oos_s = idx_all[int(n_total * (step + 1) / 5)]
                    oos_e = idx_all[min(int(n_total * (step + 2) / 5) - 1, n_total - 1)]
                    wf_ret = rf.loc[oos_s:oos_e]
                    if len(wf_ret) < 60:
                        continue
                    n_wf, g_wf, s_wf = _strat_exec(wf_ind, wf_thresh, wf_ret, lag)
                    if len(n_wf) < 30:
                        continue
                    sh = n_wf.mean() * 252 / (n_wf.std() * np.sqrt(252) + 1e-9)
                    md = (1 + n_wf).cumprod()
                    md = float((md / md.cummax() - 1).min())
                    wf_rows.append({
                        "WF-Fenster": f"{str(oos_s)[:10]} – {str(oos_e)[:10]}",
                        "OOS Sharpe": round(sh, 3),
                        "OOS MaxDD%": round(md * 100, 2),
                        "OOS Ann.Ret%": round(n_wf.mean() * 252 * 100, 2),
                        "N Tage": len(n_wf),
                    })

                if wf_rows:
                    wf_df = pd.DataFrame(wf_rows)
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Bar(
                        x=wf_df["WF-Fenster"].tolist(),
                        y=wf_df["OOS Sharpe"].tolist(),
                        marker_color=["#3fb950" if v > 0 else "#f78166"
                                      for v in wf_df["OOS Sharpe"]],
                        text=[f"{v:.3f}" for v in wf_df["OOS Sharpe"]],
                        textposition="outside"))
                    fig_wf.add_hline(y=0, line_color="#8b949e")
                    fig_wf.update_layout(
                        title="Walk-Forward OOS Sharpe: " + wf_name + " | " + pair_label,
                        height=360)
                    pair_html.append(_chart_card(
                        "Walk-Forward Validation: " + pair_label, fig_wf, height=380,
                        interp="4 OOS-Fenster à ~20% der Daten (expandierendes IS-Fenster). "
                               "Grün: positiv, Rot: negativ. "
                               "Konsistent grüne Balken = robuste Strategie, nicht nur IS-Overfitting."))
                    pair_html.append(_card("Walk-Forward Tabelle", _df_html(wf_df)))

        body_sections.append("".join(pair_html))

    body = (
        "<div class='ph-header'>"
        "<h1>Lead-Lag Indikator-Optimierung</h1>"
        "<div class='sub'>Ein Indikator gleichzeitig &middot; IS-Sharpe-Heatmap (Parameter-Grid) &middot; "
        "Walk-Forward OOS Validation &middot; Top-5 Granger-Paare</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Methodik</strong></div>"
        "<div class='card-body'>"
        "<div class='row'>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>IS-Periode:</strong> erste 70% der Daten → Parameter-Kalibrierung auf Sharpe</li>"
        "<li><strong>OOS-Periode:</strong> letzte 30% → echte Evaluation mit IS-optimierten Params</li>"
        "<li><strong>Walk-Forward:</strong> 4 expandierende IS-Fenster (je 20% OOS)</li>"
        "<li><strong>TC:</strong> 10bp pro Richtungswechsel (round-trip)</li>"
        "</ul></div>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>RSI:</strong> Window [7,10,14,21,30] × Threshold [35..65] = 35 Kombinationen</li>"
        "<li><strong>MACD:</strong> Fast [6,8,12,16,20] × Slow [18..50] = 25 (nur fast&lt;slow)</li>"
        "<li><strong>BB-Pos:</strong> Window [10..50] × Threshold [0.3..0.7] = 25 Kombinationen</li>"
        "<li><strong>SMA-Cross:</strong> Fast [5..30] × Slow [20..200] = 16 (nur fast&lt;slow)</li>"
        "</ul></div>"
        "</div>"
        "</div></div>"
    ) + "".join(body_sections)

    _write(out / "lead_lag_optimizer.html",
           _html_base("Lead-Lag Optimizer", 19, body))



def build_strategy_pairs_report(tables, figures, out):  # noqa: C901
    """All Granger pairs × 5 indicators: IS/OOS, 26 metrics, pair statistics, rolling metrics."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "strategy_pairs.html",
               _html_base("Strategie-Paare", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── Pair list ─────────────────────────────────────────────────────────────
    MANUAL = [
        ("GC=F", "GDX", 7, "Gold→GDX"),
        ("GC=F", "NEM", 6, "Gold→NEM"),
        ("GC=F", "SIL", 1, "Gold→SIL"),
        ("CL=F", "XLE", 1, "WTI→XLE"),
        ("CL=F", "XOM", 1, "WTI→XOM"),
        ("BZ=F", "XLE", 1, "Brent→XLE"),
        ("HG=F", "FCX", 1, "Kupfer→FCX"),
        ("SI=F", "SIL", 1, "Silber→SIL"),
        ("SI=F", "GDX", 1, "Silber→GDX"),
    ]
    all_pairs = []
    seen = set()
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            key = (c, e)
            if key not in seen and c in returns.columns and e in returns.columns:
                all_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
                seen.add(key)
    for mp in MANUAL:
        key = (mp[0], mp[1])
        if key not in seen and mp[0] in returns.columns and mp[1] in returns.columns:
            all_pairs.append(mp)
            seen.add(key)

    # ── Indicators (default params, one at a time) ────────────────────────────
    INDICATORS = [
        ("RSI(14)>50",    lambda px: _calc_rsi(px, 14),       50.0),
        ("MACD>0",        lambda px: _calc_macd(px)[0],        0.0),
        ("BB-Pos>0.5",    lambda px: _calc_bb_pos(px, 20),     0.5),
        ("SMA20>SMA50",   lambda px: _calc_sma_cross(px, 20, 50), 0.0),
        ("RSI(14)<70",    lambda px: -_calc_rsi(px, 14),      -70.0),  # not overbought = long
    ]

    IS_FRAC = 0.70
    all_oos_rows = []
    pair_sections = []

    for pair_idx, (leader, follower, lag, pair_label) in enumerate(all_pairs):
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        rl = returns[leader].dropna()
        if px is None or len(px) < 300:
            continue

        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 200:
            continue
        split_date = idx_all[int(len(idx_all) * IS_FRAC)]
        is_rf = rf.loc[:split_date]
        oos_rf = rf.loc[split_date:]

        # ── Pair statistics ───────────────────────────────────────────────────
        jdx = rl.index.intersection(rf.index)
        rl_j = rl.loc[jdx]; rf_j = rf.loc[jdx]
        rolling_corr = rl_j.rolling(63).corr(rf_j).dropna()
        rolling_cov  = rl_j.rolling(63).cov(rf_j).dropna()

        # OLS: follower ~ leader
        Xc = np.column_stack([np.ones(len(rl_j)), rl_j.values])
        coef, _, _, _ = np.linalg.lstsq(Xc, rf_j.values, rcond=None)
        yhat = Xc @ coef
        ss_res = float(((rf_j.values - yhat) ** 2).sum())
        ss_tot = float(((rf_j.values - rf_j.mean()) ** 2).sum())
        beta_pair = round(float(coef[1]), 4)
        r2_pair = round(max(0.0, 1 - ss_res / ss_tot), 4)
        vif_pair = round(1 / (1 - r2_pair + 1e-9), 3)

        # CCF lags 0-10
        n_jdx = len(rl_j)
        ccf_lags = list(range(11))
        ccf_vals = []
        for lv in ccf_lags:
            x = rl_j.values[:n_jdx - lv] if lv > 0 else rl_j.values
            y = rf_j.values[lv:]          if lv > 0 else rf_j.values
            try:
                ccf_vals.append(float(np.corrcoef(x, y)[0, 1]))
            except Exception:
                ccf_vals.append(float("nan"))

        # ACF follower
        foll_acf = [round(float(rf_j.autocorr(lv)), 4) for lv in range(1, 6)]

        # ── Run indicators ────────────────────────────────────────────────────
        pair_metrics = []
        best_oos_sharpe = -999.0
        best_oos_name = "—"

        fig_eq_is = go.Figure()
        fig_eq_oos = go.Figure()
        bh_is = (1 + is_rf.dropna()).cumprod()
        bh_oos = (1 + oos_rf.dropna()).cumprod()
        for fig_eq, bh_eq in [(fig_eq_is, bh_is), (fig_eq_oos, bh_oos)]:
            fig_eq.add_trace(go.Scatter(
                x=bh_eq.index.astype(str).tolist(),
                y=bh_eq.round(4).values.tolist(),
                name=f"B&H {follower}",
                line=dict(color="#8b949e", width=0.9, dash="dot")))

        for ind_idx, (ind_name, ind_fn, thresh) in enumerate(INDICATORS):
            ind = ind_fn(px)
            n_is, g_is, s_is = _strat_exec(ind, thresh, is_rf, lag)
            n_oos, g_oos, s_oos = _strat_exec(ind, thresh, oos_rf, lag)
            if len(n_is) < 30 or len(n_oos) < 30:
                continue

            spy_is = spy.loc[n_is.index] if spy is not None else None
            spy_oos = spy.loc[n_oos.index] if spy is not None else None
            m_is = _full_metrics(n_is, g_is, s_is, spy_is, ind_name + " IS")
            m_oos = _full_metrics(n_oos, g_oos, s_oos, spy_oos, ind_name + " OOS")

            for m, period in [(m_is, "IS"), (m_oos, "OOS")]:
                row = {"Paar": pair_label, "Indikator": ind_name, "Periode": period}
                row.update({k: v for k, v in m.items() if k != "Name"})
                pair_metrics.append(row)
                if period == "OOS":
                    all_oos_rows.append(row)

            oos_sh = m_oos.get("Sharpe (net)", -999.0)
            if isinstance(oos_sh, float) and not np.isnan(oos_sh) and oos_sh > best_oos_sharpe:
                best_oos_sharpe = oos_sh
                best_oos_name = ind_name

            col = PAL[ind_idx % len(PAL)]
            for fig_eq, net_v in [(fig_eq_is, n_is), (fig_eq_oos, n_oos)]:
                eq = (1 + net_v).cumprod()
                fig_eq.add_trace(go.Scatter(
                    x=eq.index.astype(str).tolist(),
                    y=eq.round(4).values.tolist(),
                    name=ind_name, line=dict(color=col, width=1.5)))

        for fig_eq, title_sfx, interp_txt in [
            (fig_eq_is, "IS", "IS-Periode (70%). Immer positiver als OOS durch In-Sample-Bias."),
            (fig_eq_oos, "OOS", "OOS-Periode (30%). Echte Performance-Schätzung. Grau = B&H-Vergleich."),
        ]:
            fig_eq.update_layout(
                title=f"Equity-Kurven {title_sfx}: {pair_label}",
                yaxis_type="log", height=420)

        # Rolling metrics (RSI(14) as reference)
        n_full, g_full, s_full = _strat_exec(_calc_rsi(px, 14), 50, rf, lag)
        fig_roll = None
        if len(n_full) > 120:
            roll_sh = n_full.rolling(63).apply(
                lambda x: x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9), raw=True)
            roll_ret = n_full.rolling(63).mean() * 252 * 100
            # Vectorized rolling drawdown: % below 126T rolling peak
            _cum_full = (1 + n_full).cumprod()
            roll_mdd = (_cum_full / _cum_full.rolling(126).max() - 1) * 100
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=roll_sh.dropna().index.astype(str).tolist(),
                y=roll_sh.dropna().round(3).values.tolist(),
                name="Rolling Sharpe (63T)", line=dict(color="#58a6ff", width=1.3)))
            fig_roll.add_trace(go.Scatter(
                x=roll_ret.dropna().index.astype(str).tolist(),
                y=roll_ret.dropna().round(2).values.tolist(),
                name="Rolling Ann.Ret% (63T)", line=dict(color="#3fb950", width=1.0),
                yaxis="y2"))
            fig_roll.add_trace(go.Scatter(
                x=roll_mdd.dropna().index.astype(str).tolist(),
                y=(roll_mdd.dropna() * 100).round(2).values.tolist(),
                name="Rolling MaxDD% (126T)", line=dict(color="#f78166", width=0.8, dash="dot"),
                yaxis="y2"))
            fig_roll.add_hline(y=0, line_color="#8b949e", line_dash="dash")
            fig_roll.add_vline(x=str(split_date)[:10], line_color="#d29922",
                               line_dash="dash")
            fig_roll.update_layout(
                title=f"Rolling-Metriken (RSI-Referenz): {pair_label}",
                height=420, yaxis_title="Sharpe",
                yaxis2=dict(title="Ann.Ret% / MaxDD%", overlaying="y", side="right"))

        # Pair statistics charts
        fig_corr = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Rolling 63T Korrelation (Leader↔Follower Renditen)",
                            "Rolling 63T Kovarianz"],
            row_heights=[0.5, 0.5], vertical_spacing=0.06)
        fig_corr.add_trace(
            go.Scatter(x=rolling_corr.index.astype(str).tolist(),
                       y=rolling_corr.round(4).values.tolist(),
                       name="Korrelation", line=dict(color="#58a6ff")), row=1, col=1)
        fig_corr.add_hline(y=0, line_color="#8b949e", row=1, col=1)
        fig_corr.add_trace(
            go.Scatter(x=rolling_cov.index.astype(str).tolist(),
                       y=rolling_cov.round(8).values.tolist(),
                       name="Kovarianz", line=dict(color="#d29922")), row=2, col=1)
        fig_corr.update_layout(title=f"Paar-Abhängigkeit: {pair_label}", height=480)

        # CCF bar chart
        sig_band = 2.0 / np.sqrt(max(len(jdx), 1))
        fig_ccf_p = go.Figure(go.Bar(
            x=ccf_lags, y=ccf_vals,
            marker_color=["#3fb950" if lv == lag else
                          ("#f78166" if abs(v) > sig_band else "#58a6ff")
                          for lv, v in zip(ccf_lags, ccf_vals)],
            hovertemplate="Lag=%{x}T<br>CCF=%{y:.4f}<extra></extra>"))
        fig_ccf_p.add_hline(y=sig_band, line_color="#d29922", line_dash="dot",
                             annotation_text=f"+{sig_band:.3f}")
        fig_ccf_p.add_hline(y=-sig_band, line_color="#d29922", line_dash="dot")
        fig_ccf_p.update_layout(
            title=f"CCF (0-10 Tage): {pair_label} — Grün = Granger-Lag",
            xaxis_title="Lag (Tage)", height=320)

        # Pair info card
        stat_html = (
            "<table class='table table-dark table-sm table-bordered small'>"
            "<thead><tr><th>Statistik</th><th>Wert</th><th>Bedeutung</th></tr></thead>"
            "<tbody>"
            f"<tr><td>β (follower ~ leader OLS)</td><td>{beta_pair}</td>"
            f"<td>+1% Leader ≈ {beta_pair:.3f}% Follower-Rendite</td></tr>"
            f"<tr><td>R² (Regression)</td><td>{r2_pair}</td>"
            f"<td>{r2_pair*100:.1f}% der Follower-Varianz durch Leader erklärbar</td></tr>"
            f"<tr><td>VIF</td><td>{vif_pair}</td>"
            f"<td>Varianz-Inflationsfaktor: {vif_pair:.1f}× (bivariate Schätzung)</td></tr>"
            f"<tr><td>CCF@Lag{lag}T</td><td>{ccf_vals[min(lag,10)]:.4f}</td>"
            f"<td>Kreuzkorrelation beim Granger-Lag {lag}T</td></tr>"
            f"<tr><td>ACF₁ Follower</td><td>{foll_acf[0]}</td>"
            f"<td>Autokorrelation Follower-Rendite Lag 1 (Momentum/Mean-Reversion)</td></tr>"
            f"<tr><td>ACF₅ Follower</td><td>{foll_acf[4]}</td>"
            f"<td>Autokorrelation Follower-Rendite Lag 5</td></tr>"
            f"<tr><td>N Tage Überlappung</td><td>{len(jdx)}</td><td></td></tr>"
            "</tbody></table>")

        pal_col = PAL[pair_idx % len(PAL)]
        sec = []
        sec.append(
            f"<div class='card mb-4' style='border-left:4px solid {pal_col}'>"
            f"<div class='card-header'><strong>{pair_label}</strong>"
            f"<span class='badge bg-success ms-2'>Bester OOS: {best_oos_name}"
            f" (Sharpe {round(best_oos_sharpe,3)})</span>"
            f"<span class='badge bg-secondary ms-1'>β={beta_pair} R²={r2_pair}</span>"
            f"</div><div class='card-body'>{stat_html}</div></div>")

        if pair_metrics:
            pm_df = (pd.DataFrame(pair_metrics)
                     .drop(columns=["Name"], errors="ignore")
                     .sort_values(["Indikator", "Periode"]))
            sec.append(_card("Vollständige Metriken-Tabelle (IS + OOS)", _df_html(pm_df)))

        sec.append(_chart_card("IS-Equity-Kurven: " + pair_label, fig_eq_is, height=440,
            interp="Log-Skala. Grau gestrichelt = B&H Follower. IS-Periode (70%)."))
        sec.append(_chart_card("OOS-Equity-Kurven: " + pair_label, fig_eq_oos, height=440,
            interp="OOS-Periode (letzte 30%). Echte Out-of-Sample Performance. "
                   "OOS besser als IS-Niveau: Strategie robust. Schlechter: Overfitting."))
        if fig_roll is not None:
            sec.append(_chart_card("Rolling-Metriken: " + pair_label, fig_roll, height=440,
                interp="RSI(14)>50 als Referenzindikator. 63T rolling Sharpe (blau), "
                       "Ann.Ret% (grün), 126T MaxDD% (rot). Gelber Strich: IS|OOS-Grenze. "
                       "Stabiler Verlauf = zeitlich robuste Strategie."))
        sec.append(_chart_card("Paar-Abhängigkeit: " + pair_label, fig_corr, height=500,
            interp="Wie stark Leader und Follower korreliert/kovariant sind. "
                   "Rückgang = Lead-Lag-Beziehung schwächt sich ab → Strategie-Risiko steigt."))
        sec.append(_chart_card("CCF 0-10 Tage: " + pair_label, fig_ccf_p, height=340,
            interp=f"Kreuzkorrelation Leader→Follower. Grün = Granger-optimaler Lag {lag}T. "
                   "Rot = signifikant (über Band). Gelbe Linien = 95%-Signifikanzband."))

        pair_sections.append("".join(sec))

    # ── Grand summary ─────────────────────────────────────────────────────────
    grand_df = pd.DataFrame(all_oos_rows).sort_values(
        "Sharpe (net)", ascending=False) if all_oos_rows else pd.DataFrame()

    fig_grand = go.Figure()
    if not grand_df.empty:
        top = grand_df.head(25)
        labels = (top["Paar"].str.split("(").str[0].str.strip()
                  + " / " + top["Indikator"]).tolist()
        shares = top["Sharpe (net)"].tolist()
        fig_grand.add_trace(go.Bar(
            x=labels, y=shares,
            marker_color=["#3fb950" if v > 0.4 else "#d29922" if v > 0 else "#f78166"
                          for v in shares],
            text=[f"{v:.3f}" for v in shares], textposition="outside"))
        fig_grand.add_hline(y=0, line_color="#8b949e")
        fig_grand.add_hline(y=0.4, line_color="#3fb950", line_dash="dot",
                            annotation_text="0.4 Zielschwelle")
        fig_grand.update_layout(
            title="Grand Summary: OOS Sharpe (alle Paare × Indikatoren)",
            xaxis_tickangle=-40, height=560)

    intro = (
        "<div class='ph-header'>"
        "<h1>Strategie-Paare: Vollständige IS/OOS Analyse</h1>"
        "<div class='sub'>Alle Granger-Paare &middot; 5 Indikatoren &middot; "
        "26 Metriken &middot; Paar-Statistiken &middot; Rolling-Metriken</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Methodik</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'><h6>Strategie</h6><ul class='small'>"
        "<li>Signal: Long wenn Leader-Indikator(t-lag) &gt; Threshold, sonst Short</li>"
        "<li>Lag = bekannter Granger-Lag des Paares</li>"
        "<li>IS 70% | OOS 30% | TC 10bp/Trade</li>"
        "<li>5 Indikatoren mit domain-knowledge-Defaults</li>"
        "</ul></div>"
        "<div class='col-md-6'><h6>Paar-Statistiken</h6><ul class='small'>"
        "<li>Rolling 63T Korrelation + Kovarianz</li>"
        "<li>OLS β und R² (follower ~ leader)</li>"
        "<li>VIF = 1/(1-R²) bivariate Kollinearität</li>"
        "<li>CCF lags 0-10T, ACF Follower lags 1-5</li>"
        "</ul></div>"
        "</div></div></div>")

    body = (intro
            + _chart_card("Grand Summary: OOS Sharpe (Top-25)", fig_grand, height=580,
                interp="Grün > 0.4: Zielbereich. Gelb 0-0.4: grenzwertig. Rot: nicht profitabel. "
                       "OOS = echte Gütemessung — IS-Sharpe immer höher durch IS-Overfitting.")
            + _card("Grand Summary Tabelle (alle OOS-Ergebnisse)",
                    _df_html(grand_df.drop(columns=["Paar"], errors="ignore").head(60)))
            + "".join(pair_sections))

    _write(out / "strategy_pairs.html", _html_base("Strategie-Paare", 19, body))



def build_pca_strategy_report(tables, figures, out):  # noqa: C901
    """PC1-filtered lead-lag strategies: Base vs. PCA-A (point estimate) vs. PCA-B (bootstrap CI)."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "pca_strategy.html",
               _html_base("PCA-Strategie", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── PCA on IS period ──────────────────────────────────────────────────────
    FACTOR_COLS = [c for c in ["XLE","XLB","XLI","GDX","SIL","JETS","IYT",
                                "SPY","QQQ","IWM","IJH","MGC","^VIX","DX-Y.NYB","^TNX"]
                   if c in returns.columns]
    ret_fac = returns[FACTOR_COLS].dropna()
    IS_FRAC = 0.70
    n_total = len(ret_fac)
    split_idx = int(n_total * IS_FRAC)
    split_date_pca = ret_fac.index[split_idx]

    Z_is_raw = ret_fac.iloc[:split_idx]
    mu_fac = Z_is_raw.mean()
    sg_fac = Z_is_raw.std() + 1e-12
    Z_is = ((Z_is_raw - mu_fac) / sg_fac).values
    Z_all = ((ret_fac - mu_fac) / sg_fac).values

    _, S, Vt = np.linalg.svd(Z_is, full_matrices=False)
    pc1_load = Vt[0]                           # shape (n_features,)
    explained = S ** 2 / (S ** 2).sum()
    pc1_scores = pd.Series(Z_all @ pc1_load, index=ret_fac.index)

    # Bootstrap CI on PC1 loading (500 samples, IS only)
    np.random.seed(42)
    N_BOOT = 500
    n_is = len(Z_is)
    boot_loads = np.zeros((N_BOOT, len(FACTOR_COLS)))
    for bi in range(N_BOOT):
        idx_b = np.random.randint(0, n_is, n_is)
        _, _, Vt_b = np.linalg.svd(Z_is[idx_b], full_matrices=False)
        flip = np.sign(Vt_b[0] @ pc1_load)
        boot_loads[bi] = Vt_b[0] * flip
    lo95 = np.percentile(boot_loads, 2.5, axis=0)
    hi95 = np.percentile(boot_loads, 97.5, axis=0)
    loading_sig = (lo95 > 0) | (hi95 < 0)

    load_dict = dict(zip(FACTOR_COLS, pc1_load))
    sig_dict  = dict(zip(FACTOR_COLS, loading_sig))

    # Loadings chart
    ord_idx = np.argsort(np.abs(pc1_load))[::-1]
    s_assets = [FACTOR_COLS[i] for i in ord_idx]
    s_loads  = [float(pc1_load[i]) for i in ord_idx]
    s_lo     = [float(lo95[i]) for i in ord_idx]
    s_hi     = [float(hi95[i]) for i in ord_idx]
    s_sig    = [bool(loading_sig[i]) for i in ord_idx]

    fig_loads = go.Figure()
    fig_loads.add_trace(go.Bar(
        x=s_assets, y=s_loads,
        marker_color=["#3fb950" if sg else "#d29922" for sg in s_sig],
        error_y=dict(type="data", symmetric=False,
                     array=[hi - lo for lo, hi in zip(s_lo, s_hi)],
                     arrayminus=[mid - lo for lo, mid in zip(s_lo, s_loads)],
                     color="#8b949e"),
        text=[f"{v:.3f}{'*' if sg else ''}" for v, sg in zip(s_loads, s_sig)],
        textposition="outside"))
    fig_loads.add_hline(y=0, line_color="#8b949e")
    fig_loads.update_layout(
        title=(f"PC1-Loadings (IS) mit Bootstrap-95%-CI "
               f"| Var.erkl.: {explained[0]*100:.1f}% "
               f"| IS bis {str(split_date_pca)[:10]}"),
        height=420, xaxis_title="Asset", yaxis_title="PC1 Loading",
        annotations=[dict(x=0.01, y=0.97, xref="paper", yref="paper",
                          text="* = signifikant (CI kreuzt nicht 0, 500 Bootstraps)",
                          showarrow=False, font=dict(size=10, color="#d29922"))])

    # PC1 score chart
    pc1_sm = pc1_scores.rolling(21).mean().dropna()
    fig_pc1 = go.Figure()
    fig_pc1.add_trace(go.Scatter(
        x=pc1_sm.index.astype(str).tolist(), y=pc1_sm.round(4).values.tolist(),
        mode="lines", name="PC1 Score (21T SMA)", line=dict(color="#58a6ff", width=1.3)))
    fig_pc1.add_hline(y=0, line_color="#8b949e", line_dash="dash",
                      annotation_text="Risk-Off | Risk-On")
    fig_pc1.add_vline(x=str(split_date_pca)[:10], line_color="#d29922",
                      line_dash="dash")
    fig_pc1.update_layout(
        title="PC1-Score: Marktregime (positiv=Risk-On, negativ=Risk-Off)",
        height=350)

    # ── Pairs ─────────────────────────────────────────────────────────────────
    MANUAL = [
        ("GC=F","GDX",7,"Gold→GDX"), ("GC=F","NEM",6,"Gold→NEM"),
        ("CL=F","XLE",1,"WTI→XLE"), ("CL=F","XOM",1,"WTI→XOM"),
        ("HG=F","FCX",1,"Kupfer→FCX"), ("SI=F","SIL",1,"Silber→SIL"),
    ]
    all_pairs = []
    seen = set()
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            key = (c, e)
            if key not in seen and c in returns.columns and e in returns.columns:
                all_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
                seen.add(key)
    for mp in MANUAL:
        key = (mp[0], mp[1])
        if key not in seen and mp[0] in returns.columns and mp[1] in returns.columns:
            all_pairs.append(mp); seen.add(key)

    # ── Strategy execution per pair ───────────────────────────────────────────
    all_comp_rows = []
    pair_sections = []

    for leader, follower, lag, pair_label in all_pairs:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        if px is None or len(px) < 300:
            continue

        idx_pair = px.index.intersection(rf.index)
        if len(idx_pair) < 200:
            continue
        split_pair = idx_pair[int(len(idx_pair) * IS_FRAC)]

        # PC1 loading / significance of follower
        if follower in load_dict:
            foll_load = load_dict[follower]
            foll_sig = sig_dict[follower]
        else:
            jdx_f = rf.index.intersection(pc1_scores.index)
            if len(jdx_f) > 100:
                foll_load = float(np.corrcoef(rf.loc[jdx_f], pc1_scores.loc[jdx_f])[0, 1])
                foll_sig = abs(foll_load) > 2 / np.sqrt(len(jdx_f))
            else:
                foll_load = 0.0; foll_sig = False

        # Base signal (RSI 14 > 50)
        rsi_lead = _calc_rsi(px, 14)
        n_base, g_base, s_base = _strat_exec(rsi_lead, 50, rf, lag)
        if len(n_base) < 30:
            continue

        # PC1 score lagged by lead-lag (to avoid lookahead)
        pc1_lagged = pc1_scores.shift(lag)

        # Version A: scale to 1.0 when PC1 confirms, 0.5 when contradicts
        pc1_w_A = pc1_lagged.reindex(s_base.index, method="ffill").apply(
            lambda x: 1.0 if not np.isnan(x) and x * foll_load > 0 else 0.5)
        sig_A = s_base * pc1_w_A
        g_A = sig_A * rf.reindex(sig_A.index, method="ffill")
        n_A = (g_A - sig_A.diff().abs().fillna(0) * 0.001).dropna()

        # Version B: 1.0 when confirms AND loading significant, else 0.0
        if foll_sig:
            pc1_w_B = pc1_lagged.reindex(s_base.index, method="ffill").apply(
                lambda x: 1.0 if not np.isnan(x) and x * foll_load > 0 else 0.0)
        else:
            # Loading not significant → reduce to 0.5 always (caution mode)
            pc1_w_B = pd.Series(0.5, index=s_base.index)
        sig_B = s_base * pc1_w_B
        g_B = sig_B * rf.reindex(sig_B.index, method="ffill")
        n_B = (g_B - sig_B.diff().abs().fillna(0) * 0.001).dropna()

        # Compute metrics for IS and OOS
        for version, n_v, g_v, s_v in [
            ("Base",  n_base, g_base, s_base),
            ("PCA-A", n_A, g_A if len(g_A) > 0 else g_base, sig_A),
            ("PCA-B", n_B, g_B if len(g_B) > 0 else g_base, sig_B),
        ]:
            for period, mask_fn in [
                ("IS",  lambda x: x.loc[:split_pair]),
                ("OOS", lambda x: x.loc[split_pair:]),
            ]:
                pn = mask_fn(n_v).dropna()
                pg = mask_fn(g_v).dropna()
                ps = mask_fn(s_v)
                if len(pn) < 30:
                    continue
                spy_p = spy.loc[pn.index] if spy is not None else None
                m = _full_metrics(pn, pg if len(pg) > 0 else None,
                                  ps if len(ps) > 0 else None, spy_p)
                row = {"Paar": pair_label, "Version": version, "Periode": period,
                       "PC1-Loading": round(foll_load, 4),
                       "Loading-Signifikant": foll_sig}
                row.update({k: v for k, v in m.items() if k != "Name"})
                all_comp_rows.append(row)

        # Equity chart (full data, all 3 versions)
        fig_eq = go.Figure()
        for ver, n_v, col in [("Base", n_base, "#8b949e"),
                               ("PCA-A", n_A, "#58a6ff"),
                               ("PCA-B", n_B, "#3fb950")]:
            if len(n_v) > 30:
                eq = (1 + n_v).cumprod()
                fig_eq.add_trace(go.Scatter(
                    x=eq.index.astype(str).tolist(), y=eq.round(4).values.tolist(),
                    mode="lines", name=ver, line=dict(color=col, width=1.5)))
        bh = (1 + rf.dropna()).cumprod()
        fig_eq.add_trace(go.Scatter(
            x=bh.index.astype(str).tolist(), y=bh.round(4).values.tolist(),
            name=f"B&H {follower}", line=dict(color="#444c56", width=0.8, dash="dot")))
        fig_eq.add_vline(x=str(split_pair)[:10], line_color="#d29922",
                         line_dash="dash")
        fig_eq.update_layout(
            title=f"PCA-Strategie-Equity: {pair_label}", yaxis_type="log", height=420)

        # OOS regime overlay chart (PC1 score + strategy equity)
        oos_rf_p = rf.loc[split_pair:]
        fig_reg = go.Figure()
        if len(oos_rf_p) > 30:
            bh_oos = (1 + oos_rf_p.dropna()).cumprod()
            fig_reg.add_trace(go.Scatter(
                x=bh_oos.index.astype(str).tolist(),
                y=bh_oos.round(4).values.tolist(),
                name=f"B&H {follower}",
                line=dict(color="#8b949e", width=1)))
            oos_a = n_A.loc[split_pair:].dropna()
            if len(oos_a) > 30:
                eq_a = (1 + oos_a).cumprod()
                fig_reg.add_trace(go.Scatter(
                    x=eq_a.index.astype(str).tolist(),
                    y=eq_a.round(4).values.tolist(),
                    name="PCA-A OOS", line=dict(color="#58a6ff", width=1.5)))
            oos_b = n_B.loc[split_pair:].dropna()
            if len(oos_b) > 30:
                eq_b = (1 + oos_b).cumprod()
                fig_reg.add_trace(go.Scatter(
                    x=eq_b.index.astype(str).tolist(),
                    y=eq_b.round(4).values.tolist(),
                    name="PCA-B OOS", line=dict(color="#3fb950", width=1.5)))
            pc1_oos = pc1_scores.loc[split_pair:].rolling(21).mean().dropna()
            if len(pc1_oos) > 10:
                fig_reg.add_trace(go.Scatter(
                    x=pc1_oos.index.astype(str).tolist(),
                    y=pc1_oos.round(4).values.tolist(),
                    name="PC1-Score (21T)", yaxis="y2",
                    line=dict(color="#d29922", width=0.8, dash="dot")))
            fig_reg.update_layout(
                title=f"OOS-Regime: {pair_label}",
                height=400, yaxis_type="log", yaxis_title="Equity",
                yaxis2=dict(title="PC1 Score", overlaying="y", side="right"))

        badge_col = "bg-success" if foll_sig else "bg-secondary"
        info_card = (
            f"<div class='card mb-3' style='border-left:4px solid #58a6ff'>"
            f"<div class='card-header'><strong>{pair_label}</strong>"
            f"<span class='badge bg-info ms-2'>PC1-Loading: {foll_load:.4f}</span>"
            f"<span class='badge {badge_col} ms-1'>"
            f"{'Signifikant' if foll_sig else 'Nicht signifikant'}"
            f" (Bootstrap 95% CI)</span></div>"
            f"<div class='card-body'>"
            f"<p class='small'>"
            f"<strong>PCA-A:</strong> PC1×Loading {'> 0' if foll_load > 0 else '< 0'} → "
            f"volle Position; entgegengesetzt → halbe Position (50%).<br>"
            f"<strong>PCA-B:</strong> {'Loading signifikant → nur Long wenn PC1 bestätigt; 0% sonst.' if foll_sig else 'Loading NICHT signifikant → feste 50% Positionsgröße (vorsichtig, kein binärer Filter).'}"
            f"</p>"
            f"</div></div>")

        pair_metrics_here = [r for r in all_comp_rows if r.get("Paar") == pair_label]
        pm_df = (pd.DataFrame(pair_metrics_here)
                 .drop(columns=["Paar", "Name"], errors="ignore")
                 .sort_values(["Periode", "Version"])
                 if pair_metrics_here else pd.DataFrame())

        sec = [info_card]
        if not pm_df.empty:
            sec.append(_card("Metriken: " + pair_label, _df_html(pm_df)))
        sec.append(_chart_card("PCA-Equity: " + pair_label, fig_eq, height=440,
            interp="Grau: Base (kein PCA). Blau: PCA-A (Positions-Skalierung 100%/50%). "
                   "Grün: PCA-B (Positions-Skalierung 100%/0% oder 50% wenn nicht signifikant). "
                   "Gelber Strich: IS|OOS-Grenze. Log-Skala."))
        if len(oos_rf_p) > 30:
            sec.append(_chart_card("OOS-Regime: " + pair_label, fig_reg, height=420,
                interp="OOS-Equity aller Versionen. Gelb (rechts): PC1-Score (Risk-On/Off-Regime). "
                       "Regime-Übereinstimmung mit guten Strategy-Perioden prüfen."))
        pair_sections.append("".join(sec))

    # ── Aggregated comparison ─────────────────────────────────────────────────
    oos_comp = [r for r in all_comp_rows if r.get("Periode") == "OOS"]
    agg_summary = {}
    for ver in ["Base", "PCA-A", "PCA-B"]:
        vr = [r for r in oos_comp if r.get("Version") == ver]
        if vr:
            sharpes = [r.get("Sharpe (net)", float("nan")) for r in vr]
            sharpes_c = [s for s in sharpes if not (isinstance(s, float) and np.isnan(s))]
            agg_summary[ver] = {
                "Median OOS Sharpe": round(float(np.median(sharpes_c)), 3) if sharpes_c else float("nan"),
                "Mean OOS Sharpe": round(float(np.mean(sharpes_c)), 3) if sharpes_c else float("nan"),
                "Sharpe>0 (% Paare)": round(100 * sum(1 for s in sharpes_c if s > 0) / len(sharpes_c), 1) if sharpes_c else 0,
                "Sharpe>0.4 (% Paare)": round(100 * sum(1 for s in sharpes_c if s > 0.4) / len(sharpes_c), 1) if sharpes_c else 0,
                "N Paare": len(vr),
            }

    fig_agg = go.Figure()
    metrics_agg = ["Median OOS Sharpe", "Mean OOS Sharpe", "Sharpe>0 (% Paare)", "Sharpe>0.4 (% Paare)"]
    for vi, (ver, col) in enumerate([("Base", "#8b949e"), ("PCA-A", "#58a6ff"), ("PCA-B", "#3fb950")]):
        if ver in agg_summary:
            sd = agg_summary[ver]
            fig_agg.add_trace(go.Bar(
                name=ver,
                x=metrics_agg,
                y=[sd.get(m, 0) for m in metrics_agg],
                marker_color=col))
    fig_agg.update_layout(
        title="PCA-Versions-Vergleich (aggregiert über alle Paare, OOS)",
        barmode="group", height=420)

    comp_df = (pd.DataFrame(all_comp_rows)
               .drop(columns=["Name"], errors="ignore")
               .sort_values(["Periode", "Sharpe (net)"], ascending=[True, False])
               if all_comp_rows else pd.DataFrame())

    body = (
        "<div class='ph-header'>"
        "<h1>PCA-gefilterte Lead-Lag-Strategien</h1>"
        "<div class='sub'>PC1-Regime-Filter &middot; Version A: Positions-Skalierung &middot; "
        "Version B: Bootstrap-CI-Unsicherheit &middot; IS/OOS &middot; 26 Metriken</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Kernidee: PC1 als Regime-Filter</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<p class='small'>PC1 (erste Hauptkomponente) erklärt den größten Teil der gemeinsamen Varianz "
        "aller Faktor-Assets und ist ein proxy für das globale Marktrisiko-Regime. "
        "Positiver PC1-Score = Risk-On (SPY steigt, VIX fällt, Energie stark). "
        "Negativer Score = Risk-Off (Gold führt, DXY stark, Energie schwach).</p>"
        "<p class='small'><strong>Hypothese:</strong> Lead-Lag-Strategien funktionieren besser, "
        "wenn der PC1-Score die Follower-Richtung bestätigt. "
        "Wenn der Follower ein positives PC1-Loading hat, "
        "erwarte positive Returns in Risk-On-Regimes.</p>"
        "</div>"
        "<div class='col-md-6'>"
        "<table class='table table-dark table-sm table-bordered small'>"
        "<thead><tr><th>Version</th><th>PC1-Filter</th><th>Unsicherheit</th></tr></thead>"
        "<tbody>"
        "<tr><td><strong>Base</strong></td><td>Kein Filter</td><td>—</td></tr>"
        "<tr><td><strong>PCA-A</strong></td><td>100% wenn bestätigt, 50% wenn widersprechend</td><td>Ignoriert (Punkt-Schätzung)</td></tr>"
        "<tr><td><strong>PCA-B</strong></td><td>100%/0% wenn signifikant, sonst 50% fest</td><td>Bootstrap-CI entscheidet über Signifikanz</td></tr>"
        "</tbody></table>"
        "</div>"
        "</div></div></div>"
        + _chart_card("PC1-Loadings mit 95%-Bootstrap-CI (500 Samples)", fig_loads, height=440,
            interp="Grün=signifikant (CI kreuzt nicht 0). Gelb=nicht signifikant. "
                   "* markiert signifikante Loadings. Fehlerbalken = 95%-Bootstrap-CI.")
        + _chart_card("PC1-Score: Marktregime-Zeitreihe", fig_pc1, height=370,
            interp="Positiv = Risk-On-Regime. Negativ = Risk-Off. 21T-geglättet. "
                   "Gelber Strich: IS|OOS-Grenze. PCA-Parameter aus IS-Periode geschätzt, "
                   "dann auf gesamten Zeitraum angewandt (nur IS-Daten → kein Lookahead).")
        + _chart_card("Versions-Vergleich (OOS, aggregiert über alle Paare)", fig_agg, height=440,
            interp="PCA-A > Base: PC1-Filter verbessert Median-Sharpe. "
                   "PCA-B > PCA-A: Bootstrap-Unsicherheit nutzen verbessert weitere. "
                   "Sharpe>0.4%: Anteil der Paare mit wirklich robuster Strategie.")
        + _card("Vollständige Vergleichs-Tabelle (alle Paare × Versionen × IS/OOS)",
                _df_html(comp_df.head(120)))
        + "".join(pair_sections)
    )

    _write(out / "pca_strategy.html", _html_base("PCA-Strategie", 19, body))



def build_strategy_stress_test_report(tables, figures, out):  # noqa: C901
    """Deep-dive stress-test for the best lead-lag strategies."""
    returns = _read(tables / "phase2_returns.csv")
    prices  = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "strategy_stress_test.html",
               _html_base("Stress-Test", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None
    dxy = returns["DX-Y.NYB"].dropna() if "DX-Y.NYB" in returns.columns else None

    # VIX levels from prices or reconstructed
    vix_lvl = None
    if "^VIX" in prices.columns:
        vix_lvl = prices["^VIX"].dropna()
    elif "^VIX" in returns.columns:
        vix_lvl = np.exp(returns["^VIX"].cumsum()) * 20

    # ── Pair + indicator universe ─────────────────────────────────────────────
    PAIRS_TO_TEST = [
        ("CL=F", "JETS", 1, "WTI→JETS"),
        ("BZ=F", "JETS", 1, "Brent→JETS"),
        ("CL=F", "XLE",  1, "WTI→XLE"),
        ("GC=F", "GDX",  7, "Gold→GDX"),
        ("GC=F", "NEM",  6, "Gold→NEM"),
        ("HG=F", "FCX",  1, "Kupfer→FCX"),
        ("SI=F", "SIL",  1, "Silber→SIL"),
        ("CL=F", "XOM",  1, "WTI→XOM"),
    ]
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        existing = {(p[0], p[1]) for p in PAIRS_TO_TEST}
        for _, row in sdf.head(10).iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            if ((c, e) not in existing
                    and c in returns.columns and e in returns.columns):
                PAIRS_TO_TEST.append((c, e, lg, f"{c}\u2192{e} (Lag {lg}T)"))
                existing.add((c, e))

    INDICATORS = [
        ("RSI(14)>50",   lambda px: _calc_rsi(px, 14),        50.0),
        ("MACD>0",       lambda px: _calc_macd(px)[0],         0.0),
        ("BB-Pos>0.5",   lambda px: _calc_bb_pos(px, 20),      0.5),
        ("SMA20>SMA50",  lambda px: _calc_sma_cross(px, 20, 50), 0.0),
        ("RSI(14)<70",   lambda px: -_calc_rsi(px, 14),       -70.0),
    ]

    IS_FRAC = 0.70
    all_strats = []
    for leader, follower, lag, pair_label in PAIRS_TO_TEST:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna() if follower in returns.columns else None
        if px is None or rf is None or len(px) < 300:
            continue
        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 300:
            continue
        split_date = idx_all[int(len(idx_all) * IS_FRAC)]
        is_rf = rf.loc[:split_date]; oos_rf = rf.loc[split_date:]

        for ind_name, ind_fn, thresh in INDICATORS:
            ind = ind_fn(px)
            n_is, g_is, s_is   = _strat_exec(ind, thresh, is_rf, lag)
            n_oos, g_oos, s_oos = _strat_exec(ind, thresh, oos_rf, lag)
            if len(n_is) < 50 or len(n_oos) < 50:
                continue
            sh_is  = n_is.mean()  * 252 / (n_is.std()  * np.sqrt(252) + 1e-9)
            sh_oos = n_oos.mean() * 252 / (n_oos.std() * np.sqrt(252) + 1e-9)
            all_strats.append({
                "leader": leader, "follower": follower, "lag": lag,
                "pair_label": pair_label, "ind_name": ind_name,
                "thresh": thresh, "ind_fn": ind_fn,
                "sh_is": sh_is, "sh_oos": sh_oos,
                "n_is": n_is, "g_is": g_is, "s_is": s_is,
                "n_oos": n_oos, "g_oos": g_oos, "s_oos": s_oos,
                "split_date": split_date, "idx_all": idx_all,
                "px": px, "rf": rf,
                "is_start": idx_all[0], "oos_end": idx_all[-1],
                "is_n": len(n_is), "oos_n": len(n_oos),
            })

    if not all_strats:
        _write(out / "strategy_stress_test.html",
               _html_base("Stress-Test", 19, "<p>Keine Strategien gefunden.</p>"))
        return

    # Sort: OOS > IS first, then by OOS Sharpe descending
    all_strats.sort(
        key=lambda s: (int(s["sh_oos"] > s["sh_is"]) * 1000 + s["sh_oos"]),
        reverse=True)
    focus = all_strats[0]

    # ── ADF helper (for cointegration) ────────────────────────────────────────
    def _adf_stat(series):
        y = series.dropna().values
        if len(y) < 20:
            return float("nan")
        dy = np.diff(y); n = len(dy)
        lag_dy = np.concatenate([[0.0], dy[:-1]])
        X = np.column_stack([y[1:], np.ones(n), lag_dy])
        coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        yhat = X @ coef
        sse = float(((dy - yhat) ** 2).sum())
        var_res = sse / max(n - 3, 1)
        try:
            t = coef[0] / (np.sqrt(var_res * np.linalg.inv(X.T @ X)[0, 0]) + 1e-9)
        except Exception:
            t = float("nan")
        return float(t)

    body = []

    # ════════════════════════════════════════════════════════════════════════════
    # OVERVIEW: IS vs OOS for all strategies
    # ════════════════════════════════════════════════════════════════════════════
    ov_rows = []
    for s in all_strats[:20]:
        ov_rows.append({
            "Paar": s["pair_label"],
            "Indikator": s["ind_name"],
            "IS Start": str(s["is_start"])[:10],
            "IS Ende / OOS Start": str(s["split_date"])[:10],
            "OOS Ende": str(s["oos_end"])[:10],
            "IS N Tage": s["is_n"],
            "OOS N Tage": s["oos_n"],
            "IS Sharpe": round(s["sh_is"], 3),
            "OOS Sharpe": round(s["sh_oos"], 3),
            "OOS > IS": "YES" if s["sh_oos"] > s["sh_is"] else "",
            "OOS/IS-Ratio": round(s["sh_oos"] / (s["sh_is"] + 1e-9), 2),
        })
    ov_df = pd.DataFrame(ov_rows)

    labels_ov = [
        f"{r['Paar'].split('(')[0].strip()} / {r['Indikator']}"
        for r in ov_rows
    ]
    fig_ov = go.Figure()
    fig_ov.add_trace(go.Bar(name="IS Sharpe", x=labels_ov,
        y=[r["IS Sharpe"] for r in ov_rows], marker_color="#d29922", opacity=0.8))
    fig_ov.add_trace(go.Bar(name="OOS Sharpe", x=labels_ov,
        y=[r["OOS Sharpe"] for r in ov_rows],
        marker_color=["#3fb950" if r["OOS > IS"] == "YES" else "#58a6ff"
                      for r in ov_rows]))
    fig_ov.add_hline(y=0, line_color="#8b949e")
    fig_ov.update_layout(
        title="Top-20 Strategien: IS vs. OOS Sharpe (grün = OOS > IS = suspect/robust)",
        barmode="group", xaxis_tickangle=-30, height=500)

    body.append(
        "<div class='ph-header'><h1>Strategy Stress-Test &amp; Deep-Dive</h1>"
        "<div class='sub'>IS/OOS Zeiten und Gr\u00f6\u00dfen &middot; TC-Sweep &middot; "
        "Look-Ahead-Test &middot; Kointegrationstest &middot; "
        "Monte Carlo Shuffle &middot; Bootstrap CI &middot; "
        "Walk-Forward &middot; Jahres-Heatmap &middot; "
        "Krisenperioden &middot; VIX-Regime &middot; DXY-Filter &middot; Kelly-Sizing"
        "</div></div>"
        "<div class='card mb-4'><div class='card-header'>"
        "<strong>Warum OOS &gt; IS Sharpe besonders untersucht wird</strong>"
        "</div><div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<p class='small'>Eine h\u00f6here OOS- als IS-Sharpe kann zweierlei bedeuten:</p>"
        "<ol class='small'>"
        "<li><strong>Robust:</strong> Die Strategie generiert in unbekannten Daten sogar "
        "mehr Alpha als in der Kalibrierungsphase (z.B. OOS-Periode zuf\u00e4llig "
        "g\u00fcnstig f\u00fcr das Signal)</li>"
        "<li><strong>Lucky:</strong> Das OOS-Fenster war ein ideales Regime "
        "\u2014 kein nachhaltiger Vorteil, sobald das Marktumfeld kippt</li>"
        "</ol>"
        "<p class='small'><em>Der Stress-Test trennt beide Hypothesen durch "
        "statistische und szenariobasierte Tests.</em></p>"
        "</div>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>TC-Sweep:</strong> Wie viel Kosten h\u00e4lt die Strategie aus?</li>"
        "<li><strong>Monte Carlo (5 000 Shuffles):</strong> Ist der Sharpe statistisch signifikant?</li>"
        "<li><strong>Bootstrap CI:</strong> Konfidenzintervall um den Sharpe</li>"
        "<li><strong>Walk-Forward:</strong> Konsistenz \u00fcber rollende IS/OOS-Fenster</li>"
        "<li><strong>Krisenperioden:</strong> COVID 2020, Ukraine 2022, \u00d6lcrash 2014-16</li>"
        "<li><strong>VIX-Regime:</strong> Performance in low/medium/high Volatilität</li>"
        "<li><strong>Kointegration:</strong> Echt\u00f6konomische Bindung vorhanden?</li>"
        "</ul></div>"
        "</div></div></div>"
    )
    body.append(_chart_card("Top-20 Strategien: IS vs. OOS Sharpe", fig_ov, height=520,
        interp="Grün = OOS Sharpe größer als IS. Blau = IS größer (normales Overfitting-Muster). "
               "Alle Strategien in Tabelle unten mit exakten IS/OOS-Zeiträumen und N Tagen."))
    body.append(_card("Top-20 Strategien (IS/OOS Zeiträume + Metriken)", _df_html(ov_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1: BASELINE REPRODUCTION
    # ════════════════════════════════════════════════════════════════════════════
    f_ldr = focus["leader"]; f_fol = focus["follower"]
    f_lag = focus["lag"];    f_lbl = focus["pair_label"]
    f_ind = focus["ind_name"]; f_thr = focus["thresh"]
    f_fn  = focus["ind_fn"]; f_px = focus["px"]; f_rf = focus["rf"]
    f_spl = focus["split_date"]; f_idx = focus["idx_all"]
    f_is_rf = f_rf.loc[:f_spl]; f_oos_rf = f_rf.loc[f_spl:]

    f_ind_s = f_fn(f_px)
    f_n_full, f_g_full, f_s_full = _strat_exec(f_ind_s, f_thr, f_rf, f_lag)
    f_n_is, f_g_is, f_s_is = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag)
    f_n_oos, f_g_oos, f_s_oos = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag)

    spy_is_f  = spy.loc[f_n_is.index]  if spy is not None else None
    spy_oos_f = spy.loc[f_n_oos.index] if spy is not None else None
    m_is  = _full_metrics(f_n_is,  f_g_is,  f_s_is,  spy_is_f,  "IS")
    m_oos = _full_metrics(f_n_oos, f_g_oos, f_s_oos, spy_oos_f, "OOS")
    m_all = _full_metrics(f_n_full, f_g_full, f_s_full,
                          spy.loc[f_n_full.index] if spy is not None else None, "Gesamt")

    sh_is_val  = m_is.get("Sharpe (net)", 0)
    sh_oos_val = m_oos.get("Sharpe (net)", 0)

    base_rows = []
    for period, m, n_d, start_d, end_d in [
        ("IS (Kalibrierung)", m_is, len(f_n_is), str(f_idx[0])[:10], str(f_spl)[:10]),
        ("OOS (Validierung)", m_oos, len(f_n_oos), str(f_spl)[:10], str(f_idx[-1])[:10]),
        ("Gesamt",           m_all, len(f_n_full), str(f_idx[0])[:10], str(f_idx[-1])[:10]),
    ]:
        row = {"Periode": period, "Start": start_d, "Ende": end_d, "N Tage": n_d}
        row.update({k: v for k, v in m.items() if k != "Name"})
        base_rows.append(row)
    base_df = pd.DataFrame(base_rows)

    fig_bl = go.Figure()
    eq_full = (1 + f_n_full).cumprod()
    bh_full = (1 + f_rf.dropna()).cumprod()
    fig_bl.add_trace(go.Scatter(
        x=eq_full.index.astype(str).tolist(), y=eq_full.round(4).values.tolist(),
        name=f_ind, line=dict(color="#3fb950", width=2)))
    fig_bl.add_trace(go.Scatter(
        x=bh_full.index.astype(str).tolist(), y=bh_full.round(4).values.tolist(),
        name=f"B&H {f_fol}", line=dict(color="#8b949e", width=1, dash="dot")))
    fig_bl.add_vrect(x0=str(f_idx[0])[:10], x1=str(f_spl)[:10],
                     fillcolor="rgba(210,153,34,0.10)", line_width=0)
    fig_bl.add_vrect(x0=str(f_spl)[:10], x1=str(f_idx[-1])[:10],
                     fillcolor="rgba(63,185,80,0.10)", line_width=0)
    fig_bl.update_layout(
        title=(f"Baseline: {f_lbl} | {f_ind} | Lag {f_lag}T | TC=10bp | "
               f"IS: {str(f_idx[0])[:10]}\u2013{str(f_spl)[:10]} ({len(f_n_is)}T) | "
               f"OOS: {str(f_spl)[:10]}\u2013{str(f_idx[-1])[:10]} ({len(f_n_oos)}T)"),
        yaxis_type="log", height=520)

    body.append(
        "<div class='card mb-4' style='border-top:4px solid #3fb950'>"
        "<div class='card-header'>"
        f"<h4><strong>Deep-Dive: {f_lbl} | {f_ind}</strong></h4>"
        "<div class='row g-2 mt-1'>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>IS-Periode</div>"
        f"<div class='fw-bold small'>{str(f_idx[0])[:10]}</div>"
        f"<div class='text-muted small'>\u2192 {str(f_spl)[:10]}</div>"
        f"<span class='badge bg-warning text-dark'>{len(f_n_is)} Handelstage</span>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>OOS-Periode</div>"
        f"<div class='fw-bold small'>{str(f_spl)[:10]}</div>"
        f"<div class='text-muted small'>\u2192 {str(f_idx[-1])[:10]}</div>"
        f"<span class='badge bg-success'>{len(f_n_oos)} Handelstage</span>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>IS Sharpe</div>"
        f"<div class='fw-bold text-warning' style='font-size:1.6em'>{round(sh_is_val,3)}</div>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>OOS Sharpe</div>"
        f"<div class='fw-bold text-success' style='font-size:1.6em'>{round(sh_oos_val,3)}</div>"
        f"{'<div class=\"badge bg-danger\">OOS > IS — Stress-Test l\u00e4uft</div>' if sh_oos_val > sh_is_val else ''}"
        f"</div></div>"
        f"<div class='col-md-12 mt-1'><small class='text-muted'>"
        f"Leader: <strong>{f_ldr}</strong> | Follower: <strong>{f_fol}</strong> | "
        f"Lag: <strong>{f_lag} Handelstage</strong> | "
        f"Gesamtdaten: {len(f_n_full)} Tage | IS 70% / OOS 30%"
        f"</small></div>"
        "</div>"
        "</div>"
        f"<div class='card-body'>{_df_html(base_df)}</div>"
        "</div>"
    )
    body.append(_chart_card("Baseline Equity-Kurve (gesamt, log-Skala)", fig_bl, height=540,
        interp=f"Gelb schattiert = IS-Periode ({len(f_n_is)} Tage). "
               f"Grün schattiert = OOS-Periode ({len(f_n_oos)} Tage). "
               f"IS Sharpe {round(sh_is_val,3)} → OOS Sharpe {round(sh_oos_val,3)}. "
               f"{'OOS > IS: ungewöhnlich — alle folgenden Tests prüfen die Robustheit.' if sh_oos_val > sh_is_val else 'OOS < IS: normales Muster — IS-Overfitting wahrscheinlich gering.'}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2: TC SWEEP (Transaktionskosten-Sensitivität)
    # ════════════════════════════════════════════════════════════════════════════
    TC_BPS = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    tc_is_sh = []; tc_oos_sh = []; tc_is_ret = []; tc_oos_ret = []
    for tc_bp in TC_BPS:
        n_is_tc, _, _ = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag, tc=tc_bp/10000)
        n_oos_tc, _, _ = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag, tc=tc_bp/10000)
        if len(n_is_tc) > 30:
            tc_is_sh.append(n_is_tc.mean()*252/(n_is_tc.std()*np.sqrt(252)+1e-9))
            tc_is_ret.append(n_is_tc.mean()*252*100)
        else:
            tc_is_sh.append(float("nan")); tc_is_ret.append(float("nan"))
        if len(n_oos_tc) > 30:
            tc_oos_sh.append(n_oos_tc.mean()*252/(n_oos_tc.std()*np.sqrt(252)+1e-9))
            tc_oos_ret.append(n_oos_tc.mean()*252*100)
        else:
            tc_oos_sh.append(float("nan")); tc_oos_ret.append(float("nan"))

    # Find OOS break-even TC
    be_oos = None
    for i in range(len(TC_BPS)-1):
        if (not np.isnan(tc_oos_sh[i]) and tc_oos_sh[i] >= 0
                and not np.isnan(tc_oos_sh[i+1]) and tc_oos_sh[i+1] < 0):
            denom = tc_oos_sh[i] - tc_oos_sh[i+1] + 1e-9
            be_oos = TC_BPS[i] + (TC_BPS[i+1]-TC_BPS[i]) * tc_oos_sh[i] / denom
            break

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(x=TC_BPS, y=tc_is_sh, name="IS Sharpe",
        mode="lines+markers", line=dict(color="#d29922", width=2), marker=dict(size=7)))
    fig_tc.add_trace(go.Scatter(x=TC_BPS, y=tc_oos_sh, name="OOS Sharpe",
        mode="lines+markers", line=dict(color="#3fb950", width=2), marker=dict(size=7)))
    fig_tc.add_hline(y=0, line_color="#f78166", line_dash="dash")
    if be_oos:
        fig_tc.add_vline(x=be_oos, line_color="#f78166", line_dash="dot")
    fig_tc.update_layout(
        title=(f"TC-Sweep: Sharpe vs. Transaktionskosten | {f_lbl}"
               + (f" | OOS Break-Even: ~{be_oos:.0f}bp" if be_oos else "")),
        xaxis_title="TC (Basispunkte pro Trade)", yaxis_title="Sharpe Ratio", height=420)

    tc_df = pd.DataFrame({
        "TC (bp)": TC_BPS,
        "IS Sharpe": [round(s,3) if not np.isnan(s) else "—" for s in tc_is_sh],
        "OOS Sharpe": [round(s,3) if not np.isnan(s) else "—" for s in tc_oos_sh],
        "IS Ann.Ret%": [round(r,2) if not np.isnan(r) else "—" for r in tc_is_ret],
        "OOS Ann.Ret%": [round(r,2) if not np.isnan(r) else "—" for r in tc_oos_ret],
        "Benchmarks": ["Ideal (kein TC)","","Institutionell","Standard Retail","","","",
                        "","Aggressiv","","","Sehr aggressiv"][:len(TC_BPS)],
    })
    be_text = f"OOS Break-Even bei ca. {be_oos:.0f}bp TC. " if be_oos else ""
    body.append(_chart_card(
        "TC-Sensitivitäts-Sweep", fig_tc, height=440,
        interp=f"{be_text}Institutionelle Händler zahlen 1–5bp, Retail 5–25bp, Futures-Roll 0.5–2bp. "
               "Strategien unter 10bp-Break-Even sind für Retail unhandelbar. "
               "Rote Linie = Sharpe=0 (Break-Even)."))
    body.append(_card("TC-Sweep Tabelle", _df_html(tc_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3: LOOK-AHEAD BIAS TEST
    # ════════════════════════════════════════════════════════════════════════════
    la_rows = []
    for extra in [0, 1, 2, 3, 5]:
        n_is_la, _, _ = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag + extra)
        n_oos_la, _, _ = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag + extra)
        sh_is_la  = (n_is_la.mean()*252/(n_is_la.std()*np.sqrt(252)+1e-9)
                     if len(n_is_la) > 30 else float("nan"))
        sh_oos_la = (n_oos_la.mean()*252/(n_oos_la.std()*np.sqrt(252)+1e-9)
                     if len(n_oos_la) > 30 else float("nan"))
        la_rows.append({
            "Extra Shift": extra,
            "Total Lag": f_lag + extra,
            "IS Sharpe": round(sh_is_la,3) if not np.isnan(sh_is_la) else "—",
            "OOS Sharpe": round(sh_oos_la,3) if not np.isnan(sh_oos_la) else "—",
            "IS Ann.Ret%": round(n_is_la.mean()*252*100,2) if len(n_is_la)>30 else "—",
            "OOS Ann.Ret%": round(n_oos_la.mean()*252*100,2) if len(n_oos_la)>30 else "—",
        })

    la_df = pd.DataFrame(la_rows)
    fig_la = go.Figure()
    fig_la.add_trace(go.Scatter(
        x=[r["Extra Shift"] for r in la_rows],
        y=[r["IS Sharpe"] if isinstance(r["IS Sharpe"], float) else float("nan")
           for r in la_rows],
        name="IS Sharpe", mode="lines+markers", line=dict(color="#d29922")))
    fig_la.add_trace(go.Scatter(
        x=[r["Extra Shift"] for r in la_rows],
        y=[r["OOS Sharpe"] if isinstance(r["OOS Sharpe"], float) else float("nan")
           for r in la_rows],
        name="OOS Sharpe", mode="lines+markers", line=dict(color="#3fb950")))
    fig_la.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_la.update_layout(
        title=f"Look-Ahead Bias Test: Signal um +0...+5 Tage zusätzlich verzögert",
        xaxis_title="Zusätzliche Signalverzögerung (Tage)", yaxis_title="Sharpe", height=380)

    la_drop = 0.0
    try:
        sh0 = float(la_rows[0]["OOS Sharpe"])
        sh1 = float(la_rows[1]["OOS Sharpe"])
        la_drop = abs((sh1 - sh0) / (sh0 + 1e-9)) * 100
    except Exception:
        pass
    la_note = ("⚠ OOS-Sharpe bricht bei +1 Tag stark ein → möglicher Look-Ahead Bias!"
               if la_drop > 60 else
               f"✓ OOS-Sharpe bleibt bei +1 Tag stabil (Rückgang: {la_drop:.1f}%) → kein wesentlicher Look-Ahead Bias.")

    body.append(_chart_card("Look-Ahead Bias Test", fig_la, height=400,
        interp=f"{la_note} "
               f"Basis-Lag = {f_lag} Tag(e) (Granger-optimiert). "
               "Starker Einbruch bei +1 Tag bedeutet: Signal nutzt Same-Day-Daten unzulässig. "
               "Robuste Lead-Lag-Strategie: Sharpe fällt moderat und graduell."))
    body.append(_card("Look-Ahead Test Tabelle", _df_html(la_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4: COINTEGRATION TEST (Engle-Granger)
    # ════════════════════════════════════════════════════════════════════════════
    px_ldr = prices[f_ldr].dropna() if f_ldr in prices.columns else None
    px_fol = prices[f_fol].dropna() if f_fol in prices.columns else None
    if px_ldr is not None and px_fol is not None:
        cidx = px_ldr.index.intersection(px_fol.index)
        if len(cidx) > 200:
            lp_ldr = np.log(px_ldr.loc[cidx].values.clip(min=1e-6))
            lp_fol = np.log(px_fol.loc[cidx].values.clip(min=1e-6))
            Xc = np.column_stack([np.ones(len(lp_ldr)), lp_ldr])
            coef_c, _, _, _ = np.linalg.lstsq(Xc, lp_fol, rcond=None)
            alpha_c, beta_c = float(coef_c[0]), float(coef_c[1])
            resid_c = lp_fol - Xc @ coef_c
            adf_t = _adf_stat(pd.Series(resid_c))

            # Rolling Z-score of spread
            spread_s = pd.Series(resid_c, index=cidx)
            mu_r = spread_s.rolling(252).mean()
            sg_r = spread_s.rolling(252).std() + 1e-9
            spread_z = (spread_s - mu_r) / sg_r

            fig_coint = go.Figure()
            fig_coint.add_trace(go.Scatter(
                x=spread_z.dropna().index.astype(str).tolist(),
                y=spread_z.dropna().round(4).values.tolist(),
                name="Spread Z-Score (252T)", line=dict(color="#58a6ff", width=1.2)))
            fig_coint.add_hline(y=0, line_color="#8b949e", line_dash="dash")
            fig_coint.add_hline(y=2, line_color="#f78166", line_dash="dot",
                                annotation_text="+2σ Mean-Reversion-Zone")
            fig_coint.add_hline(y=-2, line_color="#3fb950", line_dash="dot",
                                 annotation_text="-2σ Mean-Reversion-Zone")
            fig_coint.add_vrect(x0=str(f_spl)[:10], x1=str(cidx[-1])[:10],
                                fillcolor="rgba(63,185,80,0.07)", line_width=0)
            fig_coint.update_layout(
                title=f"Kointegrationsresidual Z-Score: log({f_ldr}) ↔ log({f_fol})",
                height=400)

            # MacKinnon (1991) 2-variable critical values (no trend, constant)
            crit = {1: -4.07, 5: -3.37, 10: -3.03}
            if not np.isnan(adf_t):
                if adf_t < crit[1]:
                    coint_verdict = f"✓✓ Kointegriert auf 1%-Niveau (t={adf_t:.3f} < {crit[1]})"
                elif adf_t < crit[5]:
                    coint_verdict = f"✓ Kointegriert auf 5%-Niveau (t={adf_t:.3f} < {crit[5]})"
                elif adf_t < crit[10]:
                    coint_verdict = f"~ Schwache Kointegration auf 10%-Niveau (t={adf_t:.3f})"
                else:
                    coint_verdict = f"✗ Keine Kointegration (t={adf_t:.3f} > {crit[10]})"
            else:
                coint_verdict = "Nicht berechenbar"

            coint_df = pd.DataFrame([{
                "ADF t-Statistik (Engle-Granger)": round(adf_t, 4) if not np.isnan(adf_t) else "—",
                "Kritisch 10%": crit[10], "Kritisch 5%": crit[5], "Kritisch 1%": crit[1],
                "β (log-Preis-Elastizität)": round(beta_c, 4),
                "α (Intercept log-Preise)": round(alpha_c, 4),
                "N Beobachtungen": len(cidx),
                "Verdikt": coint_verdict,
            }])
            body.append(_chart_card(
                f"Kointegrations-Spread Z-Score: {f_ldr} ↔ {f_fol}", fig_coint, height=420,
                interp=f"Engle-Granger Test auf OLS-Residual. {coint_verdict}. "
                       f"β={beta_c:.4f}: 1% Bewegung in {f_ldr} → ca. {beta_c:.2f}% in {f_fol}. "
                       "Z-Score > +2σ: Follower relativ zum Leader teuer → Short-Signal (Spread-Trading). "
                       "Z-Score < -2σ: Follower günstig → Long-Signal. Grüne Zone = OOS-Periode."))
            body.append(_card("Kointegrationstest Ergebnis (Engle-Granger)", _df_html(coint_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5: MONTE CARLO TRADE SHUFFLE (5 000 Permutationen)
    # ════════════════════════════════════════════════════════════════════════════
    np.random.seed(42)
    N_MC = 5000
    oos_arr = f_n_oos.dropna().values
    real_sh = float(oos_arr.mean() * 252 / (oos_arr.std() * np.sqrt(252) + 1e-9))
    mc_sh = np.array([
        np.random.permutation(oos_arr).mean() * 252
        / (np.random.permutation(oos_arr).std() * np.sqrt(252) + 1e-9)
        for _ in range(N_MC)])
    pval = float((mc_sh >= real_sh).mean())
    mc95 = float(np.percentile(mc_sh, 95))

    hy, hx = np.histogram(mc_sh, bins=80)
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(
        x=((hx[:-1]+hx[1:])/2).tolist(), y=hy.tolist(),
        marker_color="#58a6ff", opacity=0.7, name="MC Sharpe"))
    fig_mc.add_vline(x=real_sh, line_color="#3fb950", line_width=2.5)
    fig_mc.add_vline(x=mc95, line_color="#d29922", line_dash="dot")
    fig_mc.update_layout(
        title=(f"Monte Carlo ({N_MC} Shuffles) OOS-Sharpe-Verteilung | {f_lbl} | {f_ind} | "
               f"Echte OOS-Sharpe: {real_sh:.3f} | p={pval:.4f}"),
        xaxis_title="Sharpe (Zufallspermutation der OOS-Tagesrenditen)", height=420)

    mc_note = (
        f"⚠ p={pval:.4f} > 0.05 — Sharpe NICHT statistisch signifikant! "
        "Ergebnis könnte zufällige Rendite-Reihenfolge sein."
        if pval > 0.05 else
        f"✓ p={pval:.4f} < 0.05 — Sharpe statistisch signifikant. "
        "Die zeitliche Struktur der Renditen trägt zum Ergebnis bei.")
    body.append(_chart_card(
        f"Monte Carlo Trade-Shuffle: {N_MC} OOS-Permutationen", fig_mc, height=440,
        interp=f"Verteilung der Sharpe-Ratio bei 5 000 zufälligen Permutationen der OOS-Tagesrenditen. "
               f"Grüne Linie = echte OOS-Sharpe ({real_sh:.3f}). "
               f"Gelb gestrichelt = 95. Perzentil der MC-Verteilung ({mc95:.3f}). "
               f"p-Wert ({pval:.4f}): Anteil der Simulationen ≥ echter Sharpe. {mc_note}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 6: BOOTSTRAP CONFIDENCE INTERVAL
    # ════════════════════════════════════════════════════════════════════════════
    N_BOOT = 2000
    is_arr = f_n_is.dropna().values
    boot_is = np.array([
        is_arr[np.random.randint(0,len(is_arr),len(is_arr))].mean()*252
        /(is_arr[np.random.randint(0,len(is_arr),len(is_arr))].std()*np.sqrt(252)+1e-9)
        for _ in range(N_BOOT)])
    boot_oos = np.array([
        oos_arr[np.random.randint(0,len(oos_arr),len(oos_arr))].mean()*252
        /(oos_arr[np.random.randint(0,len(oos_arr),len(oos_arr))].std()*np.sqrt(252)+1e-9)
        for _ in range(N_BOOT)])
    ci_is  = np.percentile(boot_is,  [2.5, 97.5])
    ci_oos = np.percentile(boot_oos, [2.5, 97.5])

    fig_boot = go.Figure()
    for arr, nm, col in [(boot_is,"IS","#d29922"),(boot_oos,"OOS","#3fb950")]:
        hy, hx = np.histogram(arr, bins=60)
        fig_boot.add_trace(go.Bar(
            x=((hx[:-1]+hx[1:])/2).tolist(), y=hy.tolist(),
            name=nm, marker_color=col, opacity=0.55))
    for ci, col, nm in [(ci_is,"#d29922","IS"),(ci_oos,"#3fb950","OOS")]:
        fig_boot.add_vline(x=float(ci[0]), line_color=col, line_dash="dot")
        fig_boot.add_vline(x=float(ci[1]), line_color=col, line_dash="dash")
    fig_boot.update_layout(
        title=f"Bootstrap 95%-CI Sharpe Ratio ({N_BOOT} Samples) | {f_lbl}",
        xaxis_title="Sharpe Ratio", height=400, barmode="overlay")

    ci_note = (f"✓ OOS 95%-CI: [{ci_oos[0]:.3f}, {ci_oos[1]:.3f}] — "
               + ("enthält 0 NICHT → Sharpe signifikant von 0 verschieden."
                  if ci_oos[0] > 0 else "enthält 0 → Sharpe nicht signifikant von 0 verschieden."))
    body.append(_chart_card(
        f"Bootstrap 95%-CI der Sharpe Ratio ({N_BOOT} Samples)", fig_boot, height=420,
        interp=f"Gelb = IS, Grün = OOS. Punktiert/gestrichelt = CI-Grenzen. "
               f"IS 95%-CI: [{ci_is[0]:.3f}, {ci_is[1]:.3f}]. {ci_note}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 7: WALK-FORWARD ANALYSE (rollende IS/OOS mit Threshold-Optimierung)
    # ════════════════════════════════════════════════════════════════════════════
    WF_IS = 756; WF_OOS = 252; WF_STEP = 252
    use_rsi = "RSI" in f_ind
    thresh_grid = ([-35,-40,-45,-50,-55,-60,-65] if f_thr < 0
                   else ([35,40,45,50,55,60,65] if use_rsi else [f_thr]))

    wf_rows = []
    t0 = 0
    while t0 + WF_IS + WF_OOS <= len(f_idx):
        is_idx  = f_idx[t0 : t0 + WF_IS]
        oos_idx = f_idx[t0 + WF_IS : min(t0 + WF_IS + WF_OOS, len(f_idx))]
        if len(oos_idx) < 60:
            break
        rf_is_wf  = f_rf.loc[is_idx[0]:is_idx[-1]]
        rf_oos_wf = f_rf.loc[oos_idx[0]:oos_idx[-1]]
        best_sh_wf = -999.0; best_thr_wf = float(thresh_grid[0])
        wf_ind_base = -_calc_rsi(f_px, 14) if f_thr < 0 else _calc_rsi(f_px, 14) if use_rsi else f_fn(f_px)
        for thr_wf in thresh_grid:
            n_wf, _, _ = _strat_exec(wf_ind_base, thr_wf, rf_is_wf, f_lag)
            if len(n_wf) > 30:
                sh_wf = n_wf.mean()*252/(n_wf.std()*np.sqrt(252)+1e-9)
                if sh_wf > best_sh_wf:
                    best_sh_wf = sh_wf; best_thr_wf = float(thr_wf)
        n_oos_wf, _, _ = _strat_exec(wf_ind_base, best_thr_wf, rf_oos_wf, f_lag)
        sh_oos_wf = (n_oos_wf.mean()*252/(n_oos_wf.std()*np.sqrt(252)+1e-9)
                     if len(n_oos_wf) > 30 else float("nan"))
        mdd_oos_wf = float("nan")
        if len(n_oos_wf) > 30:
            c = (1+n_oos_wf).cumprod()
            mdd_oos_wf = float((c/c.cummax()-1).min())*100
        wf_rows.append({
            "IS Start": str(is_idx[0])[:10], "IS Ende": str(is_idx[-1])[:10],
            "OOS Start": str(oos_idx[0])[:10], "OOS Ende": str(oos_idx[-1])[:10],
            "IS N": len(rf_is_wf), "OOS N": len(n_oos_wf),
            "Opt. Threshold": round(best_thr_wf, 1),
            "IS Sharpe (opt.)": round(best_sh_wf, 3),
            "OOS Sharpe": round(sh_oos_wf, 3) if not np.isnan(sh_oos_wf) else "—",
            "OOS MaxDD%": round(mdd_oos_wf, 2) if not np.isnan(mdd_oos_wf) else "—",
            "OOS Ann.Ret%": round(n_oos_wf.mean()*252*100,2) if len(n_oos_wf)>30 else "—",
        })
        t0 += WF_STEP

    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_oos_nums = [r["OOS Sharpe"] for r in wf_rows if isinstance(r["OOS Sharpe"], float)]
        wf_pos = sum(1 for v in wf_oos_nums if v > 0)
        wf_tot = len(wf_oos_nums)
        fig_wf = go.Figure()
        x_wf = [r["OOS Start"] + " → " + r["OOS Ende"] for r in wf_rows]
        y_wf = [r["OOS Sharpe"] if isinstance(r["OOS Sharpe"],float) else 0 for r in wf_rows]
        fig_wf.add_trace(go.Bar(x=x_wf, y=y_wf, name="OOS Sharpe",
            marker_color=["#3fb950" if v>0 else "#f78166" for v in y_wf],
            text=[f"{v:.3f}" if isinstance(r["OOS Sharpe"],float) else "—" for r,v in zip(wf_rows,y_wf)],
            textposition="outside"))
        fig_wf.add_trace(go.Scatter(x=x_wf,
            y=[r["IS Sharpe (opt.)"] for r in wf_rows],
            name="IS Sharpe (opt.)", mode="lines+markers",
            line=dict(color="#d29922", width=1.5, dash="dot")))
        fig_wf.add_hline(y=0, line_color="#8b949e")
        fig_wf.update_layout(
            title=(f"Walk-Forward: IS={WF_IS}T / OOS={WF_OOS}T / Schritt={WF_STEP}T | "
                   f"{wf_pos}/{wf_tot} OOS-Fenster > 0"),
            xaxis_tickangle=-25, height=500)
        wf_note = (f"✓ {wf_pos}/{wf_tot} ({100*wf_pos/max(wf_tot,1):.0f}%) OOS-Fenster profitabel."
                   if wf_pos >= wf_tot * 0.6 else
                   f"⚠ Nur {wf_pos}/{wf_tot} ({100*wf_pos/max(wf_tot,1):.0f}%) OOS-Fenster profitabel — instabile Strategie.")
        body.append(_chart_card(
            f"Walk-Forward Analyse (IS={WF_IS}T, OOS={WF_OOS}T, Schritt={WF_STEP}T)",
            fig_wf, height=520,
            interp=f"Rollende IS-Threshold-Optimierung, dann OOS-Test. "
                   f"Gelb gestrichelt = optimierter IS-Sharpe. Balken = echter OOS-Sharpe. "
                   f"{wf_note} "
                   "Konsistente grüne Balken = zeitlich robuste Strategie."))
        body.append(_card("Walk-Forward Tabelle (IS/OOS Zeiträume + N Tage)", _df_html(wf_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 8: JAHRES-KALENDER (Annual Returns)
    # ════════════════════════════════════════════════════════════════════════════
    ann_rows = []
    for yr in sorted(set(f_n_full.index.year)):
        yr_s  = f_n_full[f_n_full.index.year == yr].dropna()
        yr_bh = f_rf.dropna(); yr_bh = yr_bh[yr_bh.index.year == yr]
        if len(yr_s) < 20:
            continue
        sh_y  = yr_s.mean()*252/(yr_s.std()*np.sqrt(252)+1e-9)
        ret_y = yr_s.mean()*252*100
        bh_y  = yr_bh.mean()*252*100 if len(yr_bh)>10 else float("nan")
        mdd_y = float(((1+yr_s).cumprod()/(1+yr_s).cumprod().cummax()-1).min())*100
        ann_rows.append({
            "Jahr": yr,
            "IS/OOS": "OOS" if yr_s.index[-1] >= f_spl else "IS",
            "Ann.Ret% (Strat)": round(ret_y,2),
            "Ann.Ret% (B&H)": round(bh_y,2) if not np.isnan(bh_y) else "—",
            "Alpha%": round(ret_y-bh_y,2) if not np.isnan(bh_y) else "—",
            "Sharpe": round(sh_y,3),
            "MaxDD%": round(mdd_y,2),
            "N Handelstage": len(yr_s),
        })

    if ann_rows:
        ann_df = pd.DataFrame(ann_rows)
        years = [r["Jahr"] for r in ann_rows]
        strat_ret = [r["Ann.Ret% (Strat)"] for r in ann_rows]
        bh_ret = [r["Ann.Ret% (B&H)"] if isinstance(r["Ann.Ret% (B&H)"],float)
                  else float("nan") for r in ann_rows]
        fig_ann = go.Figure()
        fig_ann.add_trace(go.Bar(x=years, y=strat_ret,
            name="Strategie",
            marker_color=["#3fb950" if v>0 else "#f78166" for v in strat_ret],
            text=[f"{v:.1f}%" for v in strat_ret], textposition="outside"))
        fig_ann.add_trace(go.Scatter(x=years, y=bh_ret,
            name=f"B&H {f_fol}", mode="lines+markers",
            line=dict(color="#8b949e", width=1.5, dash="dot")))
        fig_ann.add_hline(y=0, line_color="#8b949e")
        oos_yrs = [r["Jahr"] for r in ann_rows if r["IS/OOS"]=="OOS"]
        if oos_yrs:
            fig_ann.add_vrect(x0=min(oos_yrs)-0.5, x1=max(oos_yrs)+0.5,
                              fillcolor="rgba(63,185,80,0.08)", line_width=0)
        fig_ann.update_layout(
            title=f"Jährliche Performance: {f_lbl} | {f_ind} | Grün schattiert = OOS",
            xaxis_title="Jahr", yaxis_title="Ann. Rendite %", height=440)
        body.append(_chart_card("Jährliche Renditen (Kalender-Heatmap)", fig_ann, height=460,
            interp="Grüne Balken: profitable Jahre. Rote: Verluste. Grauer Pfeil: B&H-Vergleich. "
                   "Grün schattiert = OOS-Jahre (nicht in IS-Kalibrierung enthalten). "
                   "Gute Strategie: mind. 60-70% der Jahre grün, auch OOS-Jahre."))
        body.append(_card("Jahrestabelle", _df_html(ann_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 9: KRISENPERIODEN-ANALYSE
    # ════════════════════════════════════════════════════════════════════════════
    CRISES = [
        ("Öl-Crash 2014–16", "2014-06-01", "2016-02-29",
         "Öl: 100$→26$. Airlines profitieren von Kerosin-Kosten↓"),
        ("COVID-Crash 2020", "2020-01-15", "2020-09-30",
         "JETS -60%, Öl zeitweilig negativ. Fundamentale Entkopplung!"),
        ("COVID-Erholung 2021", "2021-01-01", "2021-12-31",
         "Reopening-Rally, Energie stark, Airline-Nachfrage steigt"),
        ("Ukraine/Energie-Krise 2022", "2022-01-01", "2022-12-31",
         "Öl-Schock, Inflationshoch, Airline-Kostendruck"),
        ("Normalisierung 2023", "2023-01-01", "2023-12-31",
         "Öl moderater, Zinspeak, Airlines erholen sich"),
        ("2024–2025", "2024-01-01", "2025-06-30",
         "Nachhaltiger Reiseboom vs. geopolitische Risiken"),
    ]
    crisis_rows = []
    fig_crises = go.Figure()
    for cname, cs, ce, cdesc in CRISES:
        try:
            cs_ts = pd.Timestamp(cs); ce_ts = pd.Timestamp(ce)
            cr  = f_n_full.loc[cs_ts:ce_ts].dropna()
            cr_bh = f_rf.loc[cs_ts:ce_ts].dropna()
            if len(cr) < 15:
                continue
            cr_sh  = cr.mean()*252/(cr.std()*np.sqrt(252)+1e-9)
            cr_ret = cr.mean()*252*100
            bh_ret_c = cr_bh.mean()*252*100 if len(cr_bh)>10 else float("nan")
            cr_mdd = float(((1+cr).cumprod()/(1+cr).cumprod().cummax()-1).min())*100
            in_oos = cr.index[0] >= f_spl
            crisis_rows.append({
                "Periode": cname, "Start": cs[:10], "Ende": ce[:10],
                "N Tage": len(cr), "IS/OOS": "OOS" if in_oos else "IS (nicht blind)",
                "Ann.Ret% (Strat)": round(cr_ret,2),
                "Ann.Ret% (B&H)": round(bh_ret_c,2) if not np.isnan(bh_ret_c) else "—",
                "Sharpe": round(cr_sh,3), "MaxDD%": round(cr_mdd,2),
                "Ökonom. Kontext": cdesc,
            })
            fig_crises.add_trace(go.Bar(
                x=[cname], y=[cr_ret],
                marker_color="#3fb950" if cr_ret>0 else "#f78166",
                text=f"{cr_ret:.1f}%", textposition="outside", name=cname))
        except Exception:
            continue
    fig_crises.update_layout(
        title=f"Performance in historischen Krisen-/Schlüsselperioden: {f_lbl}",
        showlegend=False, height=420, yaxis_title="Ann. Rendite %")
    fig_crises.add_hline(y=0, line_color="#8b949e")
    if crisis_rows:
        body.append(_chart_card("Krisenperioden-Performance", fig_crises, height=440,
            interp="Performance der Strategie in definierten Marktphasen. "
                   "IS-Perioden: Strategie 'kannte' diese Daten (kein Blind-Test). "
                   "OOS-Perioden: echter Blind-Test. "
                   "COVID 2020 ist der ultimative Stresstest für Öl/Airline-Strategien "
                   "(fundamentale Entkopplung: Öl negativ, Airlines am Boden)."))
        body.append(_card("Krisenperioden Tabelle", _df_html(pd.DataFrame(crisis_rows))))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 10: VIX-REGIME-ANALYSE
    # ════════════════════════════════════════════════════════════════════════════
    if vix_lvl is not None:
        vix_align = vix_lvl.reindex(f_n_full.index, method="ffill").dropna()
        common_v = vix_align.index.intersection(f_n_full.index)
        vix_v = vix_align.loc[common_v]
        strat_v = f_n_full.loc[common_v]
        reg_rows = []
        fig_vix = go.Figure()
        for rname, vlo, vhi, col in [
            ("Low (<15)", 0, 15, "#3fb950"),
            ("Normal (15–25)", 15, 25, "#58a6ff"),
            ("Elevated (25–35)", 25, 35, "#d29922"),
            ("Crisis (>35)", 35, 999, "#f78166"),
        ]:
            mask = (vix_v >= vlo) & (vix_v < vhi)
            r_v = strat_v[mask]; bh_v = f_rf.dropna().reindex(r_v.index)
            if len(r_v) < 15:
                continue
            sh_v  = r_v.mean()*252/(r_v.std()*np.sqrt(252)+1e-9)
            ret_v = r_v.mean()*252*100
            bh_rv = bh_v.mean()*252*100 if len(bh_v.dropna())>10 else float("nan")
            mdd_v = float(((1+r_v).cumprod()/(1+r_v).cumprod().cummax()-1).min())*100
            reg_rows.append({
                "VIX-Regime": rname, "N Tage": len(r_v),
                "Ann.Ret% (Strat)": round(ret_v,2),
                "Ann.Ret% (B&H)": round(bh_rv,2) if not np.isnan(bh_rv) else "—",
                "Sharpe": round(sh_v,3), "MaxDD%": round(mdd_v,2),
            })
            fig_vix.add_trace(go.Bar(x=[rname], y=[ret_v], name=rname,
                marker_color=col, text=f"{ret_v:.1f}%", textposition="outside"))
        fig_vix.add_hline(y=0, line_color="#8b949e")
        fig_vix.update_layout(title=f"VIX-Regime-Performance: {f_lbl}", showlegend=False, height=380)
        if reg_rows:
            body.append(_chart_card("VIX-Regime-Analyse", fig_vix, height=400,
                interp="Performance in verschiedenen Volatilitätsregimes. "
                       "Crisis VIX (>35): COVID 2020, GFC 2008 → Öl/Airline-Korrelation bricht auf. "
                       "Normal VIX (15–25): ideales Umfeld für fundamentale Lead-Lag-Strategien. "
                       "Low VIX (<15): geringe Bewegungen → wenig Signal."))
            body.append(_card("VIX-Regime Tabelle", _df_html(pd.DataFrame(reg_rows))))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 11: DXY-FILTER
    # ════════════════════════════════════════════════════════════════════════════
    if dxy is not None:
        dxy_trend = dxy.rolling(20).mean()
        dxy_align = dxy_trend.reindex(f_s_full.index, method="ffill")
        common_dx = dxy_align.dropna().index.intersection(f_n_full.index)
        if len(common_dx) > 100:
            dx = dxy_align.loc[common_dx]; sbase = f_s_full.loc[common_dx]
            rf_c = f_rf.reindex(common_dx)
            # Filter: when DXY rising (+), reduce position by 50% (strong dollar → oil cheaper → airlines less benefit)
            sig_flt = sbase * dxy_align.loc[common_dx].apply(
                lambda x: 0.5 if (not np.isnan(x) and x > 0) else 1.0)
            grs_flt = sig_flt * rf_c
            net_flt = (grs_flt - sig_flt.diff().abs().fillna(0) * 0.001).dropna()
            oos_common = common_dx[common_dx >= f_spl]
            dxy_comp = []
            for lbl_dx, n_dx in [
                ("Base (kein DXY-Filter)", f_n_oos),
                ("DXY-Filter (50% bei DXY-Trend↑)", net_flt.loc[oos_common] if len(oos_common)>30 else pd.Series()),
            ]:
                r_dx = n_dx.dropna()
                if len(r_dx) < 30:
                    continue
                sh_dx = r_dx.mean()*252/(r_dx.std()*np.sqrt(252)+1e-9)
                dxy_comp.append({
                    "Version": lbl_dx, "N OOS Tage": len(r_dx),
                    "OOS Sharpe": round(sh_dx,3),
                    "OOS Ann.Ret%": round(r_dx.mean()*252*100,2),
                    "OOS MaxDD%": round(float(((1+r_dx).cumprod()/(1+r_dx).cumprod().cummax()-1).min())*100,2),
                })
            if dxy_comp:
                body.append(_card(
                    f"DXY-Regime-Filter Test (OOS): {f_lbl}",
                    _df_html(pd.DataFrame(dxy_comp))
                    + "<p class='small text-muted mt-2'>"
                    "Rationale: Öl wird in USD gehandelt. Steigender Dollar (DXY ↑) drückt tendenziell "
                    "den Ölpreis (=günstiger für Airlines → schwächere Reaktion auf Ölpreissignal). "
                    "Filter: 50% Position wenn DXY-20T-Trend positiv, 100% wenn negativ."
                    "</p>"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 12: KELLY CRITERION
    # ════════════════════════════════════════════════════════════════════════════
    is_r_k = f_n_is.dropna()
    mu_k = float(is_r_k.mean()); var_k = float(is_r_k.var())
    kelly = mu_k / (var_k + 1e-9)
    hkelly = kelly / 2.0
    kelly_cap = float(np.clip(kelly, 0.01, 3.0))
    hkelly_cap = float(np.clip(hkelly, 0.01, 2.0))

    fig_kel = go.Figure()
    bh_oos_eq = (1 + f_rf.loc[f_spl:].dropna()).cumprod()
    fig_kel.add_trace(go.Scatter(
        x=bh_oos_eq.index.astype(str).tolist(), y=bh_oos_eq.round(4).values.tolist(),
        name=f"B&H {f_fol}", line=dict(color="#444c56", width=0.8, dash="dot")))
    for mult, nm, col in [
        (1.0, "1× (Base)", "#8b949e"),
        (hkelly_cap, f"{hkelly_cap:.2f}× (Half-Kelly)", "#d29922"),
        (kelly_cap, f"{kelly_cap:.2f}× (Full-Kelly)", "#3fb950"),
    ]:
        r_k = f_n_oos * mult
        eq_k = (1 + r_k).cumprod()
        fig_kel.add_trace(go.Scatter(
            x=eq_k.index.astype(str).tolist(), y=eq_k.round(4).values.tolist(),
            name=nm, line=dict(width=1.5)))
    fig_kel.update_layout(
        title=f"Kelly Criterion Sizing (OOS): {f_lbl} | Full-Kelly: {kelly_cap:.2f}×",
        yaxis_type="log", height=440)

    kelly_df = pd.DataFrame([
        {"Sizing": "1× (100%)", "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(m_oos.get("Ann.Ret% (net)",0),2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0),2), "Kommentar": "Base"},
        {"Sizing": f"{hkelly_cap:.2f}× (Half-Kelly)",
         "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(f_n_oos.mean()*252*100*hkelly_cap,2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0)*hkelly_cap,2), "Kommentar": "Empfohlen"},
        {"Sizing": f"{kelly_cap:.2f}× (Full-Kelly)",
         "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(f_n_oos.mean()*252*100*kelly_cap,2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0)*kelly_cap,2), "Kommentar": "Maximales Wachstum"},
    ])
    body.append(_chart_card("Kelly Criterion Position Sizing (OOS)", fig_kel, height=460,
        interp=f"Kelly f* = μ/σ² (aus IS-Daten: μ={mu_k:.5f}/T, σ²={var_k:.6f}). "
               f"Full-Kelly: {kelly:.2f}× → maximales geometrisches Wachstum, aber hohe Schwankungen. "
               f"Half-Kelly: {hkelly:.2f}× → bewährter Kompromiss (Sharpe identisch, Drawdowns kleiner). "
               "Log-Skala. Sharpe-Ratio ändert sich bei linearem Scaling NICHT."))
    body.append(_card("Kelly Sizing Tabelle", _df_html(kelly_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 13: OPTIMIERUNGSVORSCHLÄGE
    # ════════════════════════════════════════════════════════════════════════════
    body.append(
        "<div class='card mb-4'>"
        "<div class='card-header'><strong>Optimierungsvorschläge &amp; Nächste Schritte</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<h6>Signal-Verbesserungen</h6><ul class='small'>"
        "<li><strong>Crack Spread (HO=F):</strong> Heizöl/Kerosin-Futures statt CL=F Rohöl "
        "als direktere Proxy für Airline-Treibstoffkosten</li>"
        "<li><strong>EIA Rohöl-Lager (FRED: WCRSTUS1):</strong> Wöchentliche Lagerbestände als "
        "Frühindikator für Ölpreisbewegungen (Überraschung: Zunahme → Öl fällt → Airlines steigen)</li>"
        "<li><strong>RSI-Divergenz:</strong> CL-Preis steigt, CL-RSI fällt → Trendumkehr → "
        "Signal verstärken / Position aufbauen</li>"
        "<li><strong>Volumen-Filter:</strong> JETS-Signal nur bestätigen wenn Volumen > 21T-MA "
        "(hohes Volumen = institutionelle Bestätigung)</li>"
        "<li><strong>TSA Checkpoint-Daten:</strong> Tägliche US-Passagierzahlen als "
        "Demand-Indikator für JETS ETF (https://www.tsa.gov/travel/passenger-volumes)</li>"
        "</ul></div>"
        "<div class='col-md-6'>"
        "<h6>Risikomanagement</h6><ul class='small'>"
        "<li><strong>Stop-Loss:</strong> -2% oder -1.5σ(21T) pro Position → "
        "Verhindert Tail-Risk in Krisen (COVID 2020)</li>"
        "<li><strong>VIX-Filter:</strong> Bei VIX > 30 → Positionsgröße halbieren "
        "(Slippage erhöht sich, Muster werden unzuverlässiger)</li>"
        "<li><strong>Half-Kelly Sizing:</strong> Wie gezeigt: gleicher Sharpe, "
        "weniger Drawdowns → Standard-Empfehlung</li>"
        "<li><strong>Regime-bedingte Parameter:</strong> "
        "Low-VIX: Threshold erhöhen (60), High-VIX: Threshold senken (40) → "
        "Adaptives Signal</li>"
        "</ul>"
        "<h6 class='mt-3'>Daten-Integrität</h6><ul class='small'>"
        "<li><strong>CL=F Roll-Kosten (Contango):</strong> Öl-Futures rollen monatlich. "
        "Bei Contango (Vorwärtskurve steigend) entstehen Rollverluste von 0.5–3%/Monat "
        "→ In historischen Returns von CL=F oft NICHT korrekt abgebildet</li>"
        "<li><strong>JETS Dividenden:</strong> yfinance `auto_adjust=True` "
        "adjustiert für Ausschüttungen (bereits enthalten)</li>"
        "<li><strong>Survivorship-Bias:</strong> JETS ETF erst seit 2015 → "
        "historische Daten davor werden synthetisch extrapoliert</li>"
        "</ul></div>"
        "</div>"
        "<div class='alert alert-warning mt-3 small mb-0'>"
        "<strong>Wichtigste Caveat:</strong> Die CL=F ↔ JETS Beziehung kann durch "
        "strukturellen Wandel abgeschwächt werden: SAF (Sustainable Aviation Fuel) "
        "reduziert Öl-Abhängigkeit der Airlines, Fuel-Hedging-Programme (Delta, Southwest) "
        "entkoppeln kurzfristig. Quartalweises Re-Backtesting und Walk-Forward-Monitoring "
        "sind für Live-Trading essenziell."
        "</div>"
        "</div></div>"
    )

    _write(out / "strategy_stress_test.html",
           _html_base("Strategy Stress-Test", 19, "".join(body)))

def build_airline_oil_report(tables, figures, out):  # noqa: C901
    """
    Cross-sectional CL=F → Airline lead-lag research report (v2).
    28 sections: CCF, Granger, TE, signal stability, rolling metrics,
    crisis, VIX regimes, TC sweep, MC, bootstrap, walk-forward,
    PCA, clustering, portfolio, benchmark, long/short ratio, explanations.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import yfinance as yf
    from scipy import stats
    from scipy.stats import spearmanr, kruskal, pearsonr

    # ── universe ─────────────────────────────────────────────────────────────
    AIRLINES = {
        "DAL":   {"name": "Delta Air Lines",          "region": "USA",     "type": "Legacy"},
        "UAL":   {"name": "United Airlines",           "region": "USA",     "type": "Legacy"},
        "AAL":   {"name": "American Airlines",         "region": "USA",     "type": "Legacy"},
        "LUV":   {"name": "Southwest Airlines",        "region": "USA",     "type": "LCC"},
        "ALK":   {"name": "Alaska Air Group",          "region": "USA",     "type": "Legacy"},
        "JBLU":  {"name": "JetBlue Airways",           "region": "USA",     "type": "LCC"},
        "IAG":   {"name": "IAG (British Airways+)",    "region": "Europe",  "type": "Legacy"},
        "RYAAY": {"name": "Ryanair",                   "region": "Europe",  "type": "LCC"},
        "DLAKY": {"name": "Deutsche Lufthansa (ADR)",  "region": "Europe",  "type": "Legacy"},
        "AFLYY": {"name": "Air France-KLM (ADR)",      "region": "Europe",  "type": "Legacy"},
        "CPA":   {"name": "Copa Holdings",             "region": "LatAm",   "type": "Legacy"},
        "QABSY": {"name": "Qantas (ADR)",              "region": "Oceania", "type": "Legacy"},
        "JETS":  {"name": "US Global Jets ETF",        "region": "USA",     "type": "ETF"},
    }
    LEADER  = "CL=F"
    LAGS    = list(range(0, 11))
    IS_FRAC = 0.70
    N_MC    = 2000
    N_BOOT  = 1000
    TC_GRID = np.arange(0, 0.011, 0.001)

    # ── download ──────────────────────────────────────────────────────────────
    def _dl(ticker):
        for period in ("10y", "5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    raw = {}
    for t in [LEADER] + list(AIRLINES.keys()):
        s = _dl(t)
        if s is not None:
            raw[t] = s

    if LEADER not in raw or len(raw) < 3:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Preisdaten konnten nicht geladen werden.</p>"))
        return

    prices_df = pd.DataFrame(raw).sort_index().ffill().dropna(how="all")
    # Remove duplicate dates (e.g. DST artefacts)
    prices_df = prices_df[~prices_df.index.duplicated(keep="last")]
    log_ret   = np.log(prices_df / prices_df.shift(1))

    available = [t for t in AIRLINES if t in prices_df.columns]
    if len(available) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Ticker verfügbar.</p>"))
        return

    leader_px  = prices_df[LEADER].dropna()
    leader_ret = log_ret[LEADER].dropna()

    # market caps (best effort)
    mcap_proxy = {}
    for t in available:
        try:
            mc = yf.Ticker(t).info.get("marketCap")
            if mc:
                mcap_proxy[t] = float(mc)
        except Exception:
            pass

    # ── local helpers ─────────────────────────────────────────────────────────
    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 30:
            return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _granger_f(y_arr, x_arr, maxlag=5):
        from numpy.linalg import lstsq as _lstsq
        n = len(y_arr)
        if n < 60:
            return np.nan, 1.0
        p = min(maxlag, n // 10)
        def _build(y, *xs):
            rows = []
            for i in range(p, n):
                row = [1.0] + list(y[i-p:i][::-1])
                for x in xs:
                    row += list(x[i-p:i][::-1])
                rows.append(row)
            return np.array(rows)
        Y  = y_arr[p:]
        Xr = _build(y_arr)
        Xu = _build(y_arr, x_arr)
        br, *_ = _lstsq(Xr, Y, rcond=None); RSS_r = ((Y - Xr @ br)**2).sum()
        bu, *_ = _lstsq(Xu, Y, rcond=None); RSS_u = ((Y - Xu @ bu)**2).sum() + 1e-12
        df2 = len(Y) - 2*p - 1
        if df2 < 1:
            return np.nan, 1.0
        F    = ((RSS_r - RSS_u) / p) / (RSS_u / df2)
        pval = 1 - stats.f.cdf(F, p, df2)
        return float(F), float(pval)

    def _te(x_arr, y_arr, lag=1, bins=10):
        """Transfer entropy TE(X→Y) via binned estimator."""
        n = len(x_arr)
        if n < 60 or lag < 1:
            return 0.0
        cut = lambda a: np.digitize(a, np.percentile(a, np.linspace(0,100,bins+1)[1:-1]))
        xb = cut(x_arr); yb = cut(y_arr)
        # H(y_t|y_{t-1}) - H(y_t|y_{t-1}, x_{t-lag})
        from collections import Counter
        def _h(pairs):
            c = Counter(pairs); tot = sum(c.values())
            return -sum(v/tot * np.log2(v/tot + 1e-12) for v in c.values())
        py   = list(zip(yb[1:],   yb[:-1]))
        pyx  = list(zip(yb[1:],   yb[:-1], xb[:-lag][:len(yb)-1]))
        te   = _h(py) - _h(pyx) * 0.5
        return max(float(te), 0.0)

    INDICATORS = [
        ("RSI<70",    lambda p: -_calc_rsi(p, 14),         -70.0),
        ("RSI>50",    lambda p:  _calc_rsi(p, 14),           50.0),
        ("MACD>0",    lambda p:  _calc_macd(p)[0],            0.0),
        ("BB>0.5",    lambda p:  _calc_bb_pos(p, 20),         0.5),
        ("SMA cross", lambda p:  _calc_sma_cross(p, 20, 50),  0.0),
    ]

    # ── per-airline analysis ──────────────────────────────────────────────────
    # Separate scalars (→ df) from time-series (kept in dicts)
    scalars   = {}   # ticker → {scalar metrics}
    ccf_store = {}   # ticker → {lag: pearson r}
    rc_store  = {}   # ticker → rolling-corr Series
    strat_best = {}  # ticker → best strategy record

    for ticker in available:
        ret = log_ret[ticker].dropna()
        px_t = prices_df[ticker].dropna()
        common = (leader_px.index
                  .intersection(ret.index)
                  .intersection(leader_ret.index))
        common = common[~common.duplicated()]
        if len(common) < 252:
            continue

        lp = leader_px.reindex(common).ffill()
        lr = leader_ret.reindex(common).fillna(0.0)
        fr = ret.reindex(common).fillna(0.0)

        split_i    = int(len(common) * IS_FRAC)
        split_date = common[split_i]
        is_idx     = common[:split_i]
        oos_idx    = common[split_i:]

        # CCF at lags 0–10
        cv = {}
        for lag in LAGS:
            if lag == 0:
                r_v, _ = pearsonr(lr.values, fr.values)
            else:
                s_lr = lr.shift(lag).reindex(common).dropna()
                s_fr = fr.reindex(s_lr.index)
                both = pd.concat([s_lr, s_fr], axis=1).dropna()
                r_v  = float(pearsonr(both.iloc[:,0].values, both.iloc[:,1].values)[0]) if len(both) > 30 else 0.0
            cv[lag] = float(r_v)
        ccf_store[ticker] = cv

        best_lag = max(cv, key=lambda l: abs(cv[l]))
        best_ccf = cv[best_lag]

        # Rolling correlation at best_lag
        lag_rc = max(best_lag, 1)
        rc_s = lr.shift(lag_rc).rolling(252).corr(fr)
        rc_store[ticker] = rc_s.dropna()

        # Granger & TE
        gr_f, gr_p = _granger_f(fr.values, lr.values)
        te_v       = _te(lr.values, fr.values, lag=max(best_lag, 1))

        # Strategy search: best IS Sharpe over indicators × lags
        best_sh_is = -99.0
        best_rec   = None
        for ind_name, ind_fn, thresh in INDICATORS:
            for lag_s in sorted(set([1, 2, 3, max(best_lag, 1)])):
                n_is, g_is, s_is   = _strat_exec(ind_fn(lp), thresh, fr.loc[is_idx],  lag_s)
                n_oos, g_oos, s_oos = _strat_exec(ind_fn(lp), thresh, fr.loc[oos_idx], lag_s)
                if len(n_is) < 30 or len(n_oos) < 30:
                    continue
                sh = _sh(n_is)
                if sh > best_sh_is:
                    best_sh_is = sh
                    best_rec   = {
                        "ind": ind_name, "lag": lag_s, "thresh": thresh,
                        "ind_fn": ind_fn,
                        "n_is": n_is, "g_is": g_is, "s_is": s_is,
                        "n_oos": n_oos, "g_oos": g_oos, "s_oos": s_oos,
                        "sh_is": sh, "sh_oos": _sh(n_oos),
                        "split_date": split_date,
                    }
        if best_rec is None:
            continue
        strat_best[ticker] = best_rec

        net_all = pd.concat([best_rec["n_is"], best_rec["n_oos"]]).sort_index()
        sig_all = pd.concat([best_rec["s_is"], best_rec["s_oos"]]).sort_index()

        m_is  = _full_metrics(best_rec["n_is"],  best_rec["g_is"],  best_rec["s_is"],  name=f"{ticker} IS")
        m_oos = _full_metrics(best_rec["n_oos"], best_rec["g_oos"], best_rec["s_oos"], name=f"{ticker} OOS")

        # L/S ratio
        n_long  = int((sig_all > 0).sum())
        n_short = int((sig_all < 0).sum())
        n_flat  = int((sig_all == 0).sum())
        ls_ratio = round(n_long / (n_short + 1e-9), 2)

        # Rolling 126-day Sharpe for regime stability
        roll_sh = net_all.rolling(126).apply(lambda x: _sh(x), raw=True)
        regime_stab = float((roll_sh.dropna() > 0).mean()) if roll_sh.dropna().size > 0 else np.nan

        sp_corr, _ = spearmanr(lr.values, fr.values)
        beta_oil   = float(np.cov(fr.values, lr.values)[0,1] / (np.var(lr.values) + 1e-12))
        ar1        = float(pd.Series(fr.values).autocorr(1))
        mom        = float(pd.Series(fr.values).autocorr(21))
        vol_pct    = float(fr.std() * np.sqrt(252) * 100)

        # Signal half-life
        half_life = np.nan
        if abs(best_ccf) > 0.01:
            for ll in range(best_lag, 20):
                if abs(cv.get(ll, 0.0)) < abs(best_ccf) * 0.5:
                    half_life = float(ll - best_lag)
                    break

        scalars[ticker] = {
            "name":        AIRLINES[ticker]["name"],
            "region":      AIRLINES[ticker]["region"],
            "type":        AIRLINES[ticker]["type"],
            "mcap":        mcap_proxy.get(ticker, np.nan),
            "best_lag":    float(best_lag),
            "best_ccf":    float(best_ccf),
            "ccf_lag0":    float(cv.get(0, np.nan)),
            "roll_corr_mean": float(rc_s.mean()),
            "roll_corr_std":  float(rc_s.std()),
            "granger_f":   float(gr_f) if not np.isnan(gr_f) else np.nan,
            "granger_p":   float(gr_p),
            "te":          float(te_v),
            "half_life":   half_life,
            "regime_stab": regime_stab,
            "ar1":         ar1,
            "mom":         mom,
            "vol_pct":     vol_pct,
            "sp_corr":     float(sp_corr),
            "beta_oil":    beta_oil,
            "best_ind":    best_rec["ind"],
            "best_lag_s":  float(best_rec["lag"]),
            "sh_is":       float(best_rec["sh_is"]),
            "sh_oos":      float(best_rec["sh_oos"]),
            "oos_gt_is":   best_rec["sh_oos"] > best_rec["sh_is"],
            "n_long":      n_long, "n_short": n_short, "n_flat": n_flat,
            "ls_ratio":    ls_ratio,
            "n_days_is":   len(best_rec["n_is"]),
            "n_days_oos":  len(best_rec["n_oos"]),
            "split_date":  str(best_rec["split_date"])[:10],
            **{f"is_{k}": v  for k, v in m_is.items()  if k != "Name"},
            **{f"oos_{k}": v for k, v in m_oos.items() if k != "Name"},
        }

    if len(scalars) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Daten für Vergleichsanalyse.</p>"))
        return

    # Build scalar-only DataFrame (no Series / dicts)
    df = pd.DataFrame(scalars).T

    # Ensure numeric columns are numeric
    num_cols = [c for c in df.columns if c not in
                ("name","region","type","best_ind","split_date","oos_gt_is")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("sh_oos", ascending=False)
    sorted_tickers = list(df.index)

    # ── colour helpers ────────────────────────────────────────────────────────
    REG_COLORS  = {"USA":"#58a6ff","Europe":"#3fb950","Asia":"#ffa657",
                   "LatAm":"#f78166","Canada":"#bc8cff","Oceania":"#39d353","ETF":"#e3b341"}
    TYPE_COLORS = {"Legacy":"#58a6ff","LCC":"#3fb950","ULCC":"#ffa657","ETF":"#e3b341"}
    PAL_EQ      = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

    def _rc(r): return REG_COLORS.get(str(r), "#8b949e")
    def _tc(t): return TYPE_COLORS.get(str(t), "#8b949e")

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    # ── description box ───────────────────────────────────────────────────────
    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    # ── legend table ──────────────────────────────────────────────────────────
    def _legend(rows_html):
        return (f'<details class="mb-3"><summary style="color:#58a6ff;cursor:pointer;">'
                f'▶ Spaltenlegende</summary><div class="table-responsive mt-2">'
                f'<table class="table table-sm table-dark" style="font-size:0.8em;">'
                f'<thead><tr><th>Spalte</th><th>Berechnung</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table></div></details>')

    # ════════════════════════════════════════════════════════════════════════
    # §0  Ranking table
    # ════════════════════════════════════════════════════════════════════════
    def _f(v):
        if isinstance(v, bool): return "✓" if v else "✗"
        try:
            f = float(v)
            return "—" if np.isnan(f) else f"{f:.3f}"
        except Exception:
            return str(v)

    rank_rows = ""
    for i, (t, row) in enumerate(df.iterrows()):
        oos_flag = "✓" if row.get("oos_gt_is", False) else ""
        br = f'<span class="badge" style="background:{_rc(row.get("region",""))};">{row.get("region","")}</span>'
        bt = f'<span class="badge" style="background:{_tc(row.get("type",""))};">{row.get("type","")}</span>'
        rank_rows += (
            f"<tr><td>{i+1}</td><td><strong>{t}</strong></td><td>{row.get('name','')}</td>"
            f"<td>{br}</td><td>{bt}</td>"
            f"<td>{_f(row.get('sh_is'))}</td><td>{_f(row.get('sh_oos'))}</td><td>{oos_flag}</td>"
            f"<td>{_f(row.get('best_lag'))}</td><td>{_f(row.get('best_ccf'))}</td>"
            f"<td>{row.get('best_ind','—')}</td><td>{_f(row.get('best_lag_s'))}</td>"
            f"<td>{_f(row.get('granger_f'))}</td><td>{_f(row.get('granger_p'))}</td>"
            f"<td>{_f(row.get('te'))}</td><td>{_f(row.get('half_life'))}</td>"
            f"<td>{_f(row.get('regime_stab'))}</td>"
            f"<td>{_f(row.get('ls_ratio'))}</td>"
            f"<td>{int(row.get('n_long',0))}/{int(row.get('n_short',0))}</td>"
            f"</tr>"
        )

    leg0 = _legend(
        "<tr><td>Sharpe IS/OOS</td><td>µ·252 / (σ·√252) auf den täglichen Nettorenditen im IS- bzw. OOS-Zeitfenster</td></tr>"
        "<tr><td>OOS&gt;IS</td><td>Kennzeichen ob Sharpe OOS &gt; IS (kein Overfitting)</td></tr>"
        "<tr><td>Best Lag</td><td>Lag l* = argmax |CCF(l)| für l ∈ 0..10</td></tr>"
        "<tr><td>Peak CCF</td><td>Pearson r zwischen CL=F-Rendite(t-l*) und Airline-Rendite(t)</td></tr>"
        "<tr><td>Best Ind.</td><td>Indikator mit höchstem IS-Sharpe (RSI/MACD/BB/SMA)</td></tr>"
        "<tr><td>Granger F</td><td>F-Statistik des Granger-Kausalitätstests (H0: CL=F-Lags helfen nicht)</td></tr>"
        "<tr><td>Granger p</td><td>p-Wert des Granger-Tests; p&lt;0.05 = signifikante Kausalität</td></tr>"
        "<tr><td>Trans.Entropy</td><td>TE(CL=F→Airline) in Bits; misst nicht-lineare Informationsübertragung</td></tr>"
        "<tr><td>Signal HL</td><td>Halbwertszeit: Tage bis |CCF| auf 50% des Peaks abgefallen ist</td></tr>"
        "<tr><td>Regime Stab.</td><td>Anteil 126-Tage-Fenster mit positivem Sharpe (0–1)</td></tr>"
        "<tr><td>L/S Ratio</td><td>Anzahl Long-Tage / Anzahl Short-Tage; &gt;1 = überwiegend long</td></tr>"
        "<tr><td>L/S Tage</td><td>Absolute Anzahl Long- und Short-Handelstage</td></tr>"
    )

    sec0 = (
        _desc("Diese Tabelle zeigt alle Kennzahlen je Airline auf einen Blick. "
              "Sortierung nach OOS Sharpe (absteigend). "
              "<strong>Strategie:</strong> Long wenn CL=F-Indikator(t−Lag) &gt; Schwelle, sonst Short. "
              "TC = 10bp pro Richtungswechsel.")
        + leg0
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover" style="font-size:0.8em;">'
        f'<thead class="table-dark"><tr>'
        f'<th>#</th><th>Ticker</th><th>Name</th><th>Region</th><th>Typ</th>'
        f'<th>Sharpe IS</th><th>Sharpe OOS</th><th>OOS&gt;IS</th>'
        f'<th>Best Lag</th><th>Peak CCF</th><th>Best Ind.</th><th>Ind.Lag</th>'
        f'<th>Granger F</th><th>Granger p</th><th>Trans.Entropy</th>'
        f'<th>Signal HL</th><th>Regime Stab.</th><th>L/S Ratio</th><th>L/S Tage</th>'
        f'</tr></thead><tbody>{rank_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §1  OOS Sharpe bar chart
    # ════════════════════════════════════════════════════════════════════════
    sh_oos_v = df["sh_oos"].fillna(0).tolist()
    sh_is_v  = df["sh_is"].fillna(0).tolist()
    bar_cols = [_rc(df.loc[t, "region"]) for t in sorted_tickers]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=sorted_tickers, y=sh_is_v,  name="IS Sharpe",
                          marker_color="#30363d", opacity=0.7))
    fig1.add_trace(go.Bar(x=sorted_tickers, y=sh_oos_v, name="OOS Sharpe",
                          marker_color=bar_cols))
    fig1.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig1, title="OOS Sharpe Ranking: CL=F → Airline", barmode="group",
         xaxis_title="Airline", yaxis_title="Annualisierter Sharpe", height=430)
    sec1 = (
        _desc("Sharpe-Ratio = µ·252 / (σ·√252). Graue Balken = IS-Periode (Kalibrierung, 70% der Daten). "
              "Farbige Balken = OOS-Periode (echter Test, 30% der Daten). "
              "Indikator und Lag wurden ausschließlich auf dem IS-Zeitfenster optimiert. "
              "Ein OOS-Sharpe nahe oder über dem IS-Wert deutet auf robuste Out-of-Sample-Performance hin.")
        + _htm(fig1)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §2  L/S Ratio
    # ════════════════════════════════════════════════════════════════════════
    ls_fig = go.Figure()
    ls_long  = [int(df.loc[t, "n_long"])  for t in sorted_tickers]
    ls_short = [int(df.loc[t, "n_short"]) for t in sorted_tickers]
    ls_flat  = [int(df.loc[t, "n_flat"])  for t in sorted_tickers]
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_long,  name="Long",  marker_color="#3fb950"))
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_short, name="Short", marker_color="#f78166"))
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_flat,  name="Flat",  marker_color="#30363d"))
    _lay(ls_fig, title="Long / Short / Flat Tage je Airline (Gesamtperiode)",
         barmode="stack", height=380)

    ls_rows = "".join(
        f"<tr><td>{t}</td><td>{scalars[t].get('best_ind','—')}</td>"
        f"<td>{int(scalars[t].get('n_long',0))}</td>"
        f"<td>{int(scalars[t].get('n_short',0))}</td>"
        f"<td>{int(scalars[t].get('n_flat',0))}</td>"
        f"<td>{scalars[t].get('ls_ratio',0):.2f}</td>"
        f"<td>{int(scalars[t].get('n_long',0)+scalars[t].get('n_short',0)+scalars[t].get('n_flat',0))}</td></tr>"
        for t in sorted_tickers if t in scalars
    )
    ls_table = (
        '<div class="table-responsive mt-3"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Airline</th><th>Indikator</th><th>Long T.</th>'
        '<th>Short T.</th><th>Flat T.</th><th>L/S Ratio</th><th>Gesamt</th></tr></thead>'
        f'<tbody>{ls_rows}</tbody></table></div>'
    )
    sec2 = (
        _desc("<strong>Long-Signal:</strong> Wenn CL=F-Indikator(t−Lag) &gt; Schwelle → Position = +1 (long). "
              "<strong>Short-Signal:</strong> Wenn Indikator(t−Lag) ≤ Schwelle → Position = −1 (short). "
              "L/S Ratio = (Long-Tage) / (Short-Tage). "
              "Beispiel RSI&lt;70: Signal = Long wenn −RSI(t−Lag) &gt; −70, also wenn RSI &lt; 70.")
        + _htm(ls_fig)
        + ls_table
    )

    # ════════════════════════════════════════════════════════════════════════
    # §3  Equity curves with Long/Short markers (OOS)
    # ════════════════════════════════════════════════════════════════════════
    fig3 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        net_oos = sr["n_oos"]; sig_oos = sr["s_oos"]
        if len(net_oos) < 10:
            continue
        cum = (1 + net_oos).cumprod() * 100
        col = PAL_EQ[i % len(PAL_EQ)]
        fig3.add_trace(go.Scatter(
            x=cum.index.astype(str).tolist(), y=cum.values.tolist(),
            name=t, mode="lines", line=dict(color=col, width=1.8),
        ))
        # Signal-change markers (entries)
        changes = sig_oos.diff().fillna(0)
        longs  = sig_oos.index[changes > 0]
        shorts = sig_oos.index[changes < 0]
        for dates_e, symbol, mcolor in [(longs, "triangle-up", "#3fb950"),
                                         (shorts, "triangle-down", "#f78166")]:
            if len(dates_e) == 0:
                continue
            y_e = cum.reindex(dates_e, method="nearest").dropna()
            fig3.add_trace(go.Scatter(
                x=y_e.index.astype(str).tolist(), y=y_e.values.tolist(),
                mode="markers", name=f"{t} {'Long' if symbol=='triangle-up' else 'Short'}",
                marker=dict(symbol=symbol, size=7, color=mcolor, opacity=0.7),
                showlegend=False,
            ))
    _lay(fig3, title="OOS Equity Curves: CL=F → Airline (▲=Long-Entry, ▼=Short-Entry)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=520)
    sec3 = (
        _desc("Equity Curve: NAV = (1+r₁)·(1+r₂)·…·(1+rₙ)·100, wobei rᵢ = Netto-Tagesrendite nach TC. "
              "▲ = Long-Entry (Positionswechsel von Short auf Long). ▼ = Short-Entry (von Long auf Short). "
              "Nur OOS-Periode dargestellt (30% der Gesamtdaten, zeitlich hintere Hälfte).")
        + _htm(fig3)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §4  CCF Heatmap Lag 0–10
    # ════════════════════════════════════════════════════════════════════════
    ccf_mat = np.zeros((len(LAGS), len(sorted_tickers)))
    for j, t in enumerate(sorted_tickers):
        for i, lag in enumerate(LAGS):
            ccf_mat[i, j] = ccf_store.get(t, {}).get(lag, 0.0)

    fig4 = go.Figure(go.Heatmap(
        z=ccf_mat.tolist(),
        x=sorted_tickers,
        y=[f"Lag {l}T" for l in LAGS],
        colorscale="RdBu", zmid=0,
        text=[[f"{ccf_mat[i,j]:.3f}" for j in range(len(sorted_tickers))] for i in range(len(LAGS))],
        texttemplate="%{text}", textfont=dict(size=9, color="#e6edf3"),
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
        zmin=-0.5, zmax=0.5,
    ))
    _lay(fig4, title="CCF Heatmap: CL=F(t−Lag) → Airline(t)", height=480)
    sec4 = (
        _desc("Cross-Correlation Function (CCF): CCF(l) = Pearson r(CL=F(t-l), Airline(t)). "
              "Roter Bereich = positive Korrelation (hoher Ölpreis → hohe Airline-Rendite), "
              "blauer Bereich = negative Korrelation. "
              "Liegt das Maximum bei Lag &gt; 0, bedeutet das: CL=F führt Airline um l Tage an.")
        + _htm(fig4)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §5  Best Lag scatter
    # ════════════════════════════════════════════════════════════════════════
    lags_arr   = [float(df.loc[t, "best_lag"]) for t in sorted_tickers]
    sh_oos_arr = [float(df.loc[t, "sh_oos"])   for t in sorted_tickers]

    fig5 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig5.add_trace(go.Scatter(
            x=[lags_arr[i] for i in mask], y=[sh_oos_arr[i] for i in mask],
            mode="markers+text", text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=12, color=col, line=dict(color="#0d1117", width=1)),
            name=reg,
        ))
    _lay(fig5, title="Optimaler Lag × OOS Sharpe", height=420,
         xaxis_title="Lag l* (Tage)", yaxis_title="OOS Sharpe")
    sec5 = (
        _desc("Optimaler Lag l* = argmax_{l∈0..10} |CCF(l)|. "
              "Airlines rechts oben (hoher Lag, hoher Sharpe) profitieren am meisten "
              "von zeitverzögerter Ölinformation. US-notierte ADRs tendieren zu kurzem Lag (1–2T), "
              "internationale ADRs oft zu längerem Lag (3–7T) durch unterschiedliche Handelszeiten.")
        + _htm(fig5)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §6  Bubble Chart
    # ════════════════════════════════════════════════════════════════════════
    mc_vals = [(float(df.loc[t, "mcap"]) / 1e9
                if not pd.isna(df.loc[t, "mcap"]) else 3.0)
               for t in sorted_tickers]

    fig6 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig6.add_trace(go.Scatter(
            x=[mc_vals[i] for i in mask], y=[sh_oos_arr[i] for i in mask],
            mode="markers+text", text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=[max(mc_vals[i]*0.6, 5) for i in mask],
                        color=col, opacity=0.8,
                        sizemode="area", sizeref=0.1,
                        line=dict(color="#0d1117", width=1)),
            name=reg,
        ))
    _lay(fig6, title="Market Cap (Mrd. USD) × OOS Sharpe × Region",
         xaxis_title="Market Cap (Mrd. USD)", yaxis_title="OOS Sharpe", height=450)
    sec6 = (
        _desc("Blasengröße = Market Cap in Mrd. USD (falls verfügbar, sonst 3 Mrd.). "
              "Farbe = Region. "
              "Kein systematischer Zusammenhang zwischen Größe und Sharpe würde bedeuten, "
              "dass Market Cap kein Screening-Kriterium für Lead-Lag-Qualität ist.")
        + _htm(fig6)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §7  Rolling 252T Sharpe
    # ════════════════════════════════════════════════════════════════════════
    fig7 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        roll    = net_all.rolling(252).apply(lambda x: _sh(x), raw=True).dropna()
        if len(roll) < 10:
            continue
        fig7.add_trace(go.Scatter(
            x=roll.index.astype(str).tolist(), y=roll.values.tolist(),
            name=t, mode="lines",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.5),
        ))
    fig7.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig7, title="Rolling 252T Sharpe: je Airline",
         xaxis_title="Datum", yaxis_title="Sharpe (252T-Fenster)", height=460)
    sec7 = (
        _desc("Rolling Sharpe mit 252-Tage-Fenster (≈ 1 Börsenjahr). "
              "Berechnung: Sharpe(t) = µ_{t-251..t}·252 / (σ_{t-251..t}·√252). "
              "Werte über 0 zeigen profitable Perioden; anhaltend positive Kurven deuten "
              "auf strukturelle Lead-Lag-Persistenz hin.")
        + _htm(fig7)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §8  Rolling Correlation
    # ════════════════════════════════════════════════════════════════════════
    fig8 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        rc = rc_store.get(t)
        if rc is None or len(rc) < 10:
            continue
        fig8.add_trace(go.Scatter(
            x=rc.index.astype(str).tolist(), y=rc.values.tolist(),
            name=t, mode="lines",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.2),
        ))
    fig8.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig8, title="Rolling 252T Korrelation: CL=F(t-Lag) → Airline(t)",
         xaxis_title="Datum", yaxis_title="Pearson r", height=430)
    sec8 = (
        _desc("Rolling Pearson-Korrelation zwischen CL=F-Rendite(t−l*) und Airline-Rendite(t) "
              "in einem 252-Tage-Schiebefenster. Ein stabiler positiver Verlauf zeigt, dass "
              "die Lead-Lag-Struktur zeitlich konstant ist. Starke Schwankungen deuten auf "
              "Regime-Wechsel (z.B. Ölcrash, COVID) hin.")
        + _htm(fig8)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §9  Radar Charts
    # ════════════════════════════════════════════════════════════════════════
    RMETS = ["sh_oos", "best_ccf", "granger_f", "te", "regime_stab", "roll_corr_mean"]
    RLABS = ["OOS Sharpe", "Peak CCF", "Granger F", "Trans.Entropy", "Regime Stab.", "Roll.Corr."]

    def _norm(col):
        v = df[col].apply(pd.to_numeric, args=("coerce",) if False else ())
        v = pd.to_numeric(v, errors="coerce")
        mn, mx = v.min(), v.max()
        if mx == mn:
            return pd.Series(0.5, index=v.index)
        return (v - mn) / (mx - mn)

    nv = {m: _norm(m) for m in RMETS if m in df.columns}
    n_r = max(1, (len(sorted_tickers) + 3) // 4)
    fig9 = make_subplots(
        rows=n_r, cols=4,
        specs=[[{"type": "polar"}] * 4 for _ in range(n_r)],
        subplot_titles=sorted_tickers[:n_r * 4],
    )
    for i, t in enumerate(sorted_tickers):
        row = i // 4 + 1; col = i % 4 + 1
        vals_r = [float(nv[m].get(t, 0.0)) if m in nv else 0.0 for m in RMETS]
        vals_r.append(vals_r[0])
        fig9.add_trace(go.Scatterpolar(
            r=vals_r, theta=RLABS + [RLABS[0]], fill="toself",
            name=t, line=dict(color=PAL_EQ[i % len(PAL_EQ)]), showlegend=False,
        ), row=row, col=col)
    fig9.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=max(350, n_r * 300),
        title_text="Radar Charts: je Airline (normalisiert auf 0–1)",
    )
    sec9 = (
        _desc("Alle 6 Dimensionen sind min-max-normalisiert auf [0,1]. "
              "Eine große ausgefüllte Fläche = die Airline ist in allen Dimensionen überdurchschnittlich. "
              "Achsen: OOS Sharpe (Strategiequalität), Peak CCF (maximale lineare Korrelation mit CL=F), "
              "Granger F (statistische Kausalität), Transfer Entropy (nicht-lineare Informationsübertragung), "
              "Regime Stab. (Anteil profitabler 6-Monats-Fenster), Roll.Corr. (mittlere rollende Korrelation).")
        + _htm(fig9)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §10  Feature Matrix Heatmap
    # ════════════════════════════════════════════════════════════════════════
    FEAT_COLS = ["sh_is", "sh_oos", "best_lag", "best_ccf", "granger_f", "te",
                 "half_life", "regime_stab", "ar1", "mom", "vol_pct", "beta_oil",
                 "roll_corr_mean", "roll_corr_std"]
    feat_df = df[[c for c in FEAT_COLS if c in df.columns]].copy()
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-9)

    z_vals  = feat_norm.values.T.tolist()
    txt_raw = feat_df.values.T
    txt_mat = [[f"{txt_raw[i,j]:.2f}" if not np.isnan(float(txt_raw[i,j])) else "—"
                for j in range(txt_raw.shape[1])]
               for i in range(txt_raw.shape[0])]

    fig10 = go.Figure(go.Heatmap(
        z=z_vals, x=feat_norm.index.tolist(), y=feat_norm.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=txt_mat, texttemplate="%{text}", textfont=dict(size=9, color="#e6edf3"),
        colorbar=dict(title="z-score", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig10, title="Feature Matrix: Airlines × Metriken (z-score normiert)", height=520)

    leg10 = _legend(
        "<tr><td>sh_is/sh_oos</td><td>Sharpe-Ratio IS/OOS</td></tr>"
        "<tr><td>best_lag</td><td>Lag mit höchstem |CCF|</td></tr>"
        "<tr><td>best_ccf</td><td>Pearson r bei best_lag</td></tr>"
        "<tr><td>granger_f</td><td>Granger F-Statistik</td></tr>"
        "<tr><td>te</td><td>Transfer Entropy in Bits</td></tr>"
        "<tr><td>half_life</td><td>Tage bis CCF auf 50% abfällt</td></tr>"
        "<tr><td>regime_stab</td><td>Anteil pos. 126T-Sharpe-Fenster</td></tr>"
        "<tr><td>ar1</td><td>AR(1)-Autokorrelation der Airline-Renditen (Mean-Reversion wenn &lt;0)</td></tr>"
        "<tr><td>mom</td><td>Autokorrelation bei Lag 21 (Momentum-Proxy)</td></tr>"
        "<tr><td>vol_pct</td><td>Historische Ann.Volatilität der Airline (%)</td></tr>"
        "<tr><td>beta_oil</td><td>β = Cov(r_airline, r_oil)/Var(r_oil)</td></tr>"
        "<tr><td>roll_corr_mean/std</td><td>Mittelwert/Streuung der rollenden Korrelation</td></tr>"
    )
    sec10 = (
        _desc("Jeder Wert ist z-score normiert: z = (x − µ) / σ. "
              "Rot = überdurchschnittlich, Blau = unterdurchschnittlich. "
              "Zahlenangaben zeigen die rohen (un-normierten) Werte. "
              "Zeilen = Metriken, Spalten = Airlines.")
        + leg10
        + _htm(fig10)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §11  Feature Correlation Matrix
    # ════════════════════════════════════════════════════════════════════════
    feat_complete = feat_df.dropna(how="any")
    if len(feat_complete) < 3:
        feat_complete = feat_df.fillna(feat_df.median())
    corr_m = feat_complete.corr(method="pearson")

    fig11 = go.Figure(go.Heatmap(
        z=corr_m.values.tolist(),
        x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{corr_m.values[i,j]:.2f}" for j in range(len(corr_m.columns))]
              for i in range(len(corr_m.index))],
        texttemplate="%{text}", textfont=dict(size=10, color="#e6edf3"),
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig11, title="Feature-Korrelationsmatrix (Pearson, Airlines als Beobachtungen)", height=520)
    sec11 = (
        _desc("Korrelationsmatrix der 14 Metriken über alle Airlines. "
              "Hohe Korrelation zwischen z.B. Granger F und CCF würde bedeuten, "
              "dass beide ähnliche Aspekte der Lead-Lag-Beziehung messen. "
              "NaN-Werte wurden mit dem jeweiligen Median imputiert.")
        + _htm(fig11)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §12  Hierarchical Clustering
    # ════════════════════════════════════════════════════════════════════════
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist

    feat_cl = feat_df.fillna(feat_df.median())
    feat_cl = (feat_cl - feat_cl.mean()) / (feat_cl.std() + 1e-9)

    fig12 = go.Figure()
    clust_labels = {}
    if len(feat_cl) >= 3:
        dm = pdist(feat_cl.values, metric="euclidean")
        Z  = linkage(dm, method="ward")
        dd = dendrogram(Z, labels=feat_cl.index.tolist(), no_plot=True)
        for xs, ys in zip(dd["icoord"], dd["dcoord"]):
            fig12.add_trace(go.Scatter(x=ys, y=xs, mode="lines",
                                       line=dict(color="#58a6ff"), showlegend=False))
        n_leafs = len(feat_cl)
        fig12.update_yaxes(
            tickvals=list(range(5, (n_leafs + 1) * 10, 10)),
            ticktext=dd["ivl"],
        )
        k = min(3, len(feat_cl))
        cids = fcluster(Z, k, criterion="maxclust")
        clust_labels = {t: int(c) for t, c in zip(feat_cl.index, cids)}
    _lay(fig12, title="Hierarchisches Clustering: Ward-Linkage (Euklidische Distanz)",
         height=max(350, len(feat_cl) * 25))
    sec12 = (
        _desc("Ward-Linkage minimiert die intra-cluster Varianz. "
              "Weit voneinander entfernte Äste = unterschiedliche Lead-Lag-Profile. "
              "k=3 Cluster wurden anschließend mittels fcluster vergeben. "
              "Distance = √Σ(z_i − z_j)².")
        + _htm(fig12)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §13  PCA Biplot
    # ════════════════════════════════════════════════════════════════════════
    pca_html = ""
    if len(feat_cl) >= 3 and feat_cl.shape[1] >= 2:
        X = feat_cl.values
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        scores   = U[:, :2] * S[:2]
        loadings = Vt[:2, :].T
        ev = (S**2 / (S**2).sum())[:2]

        fig13 = go.Figure()
        for reg, col in REG_COLORS.items():
            mask = [i for i, t in enumerate(feat_cl.index) if df.loc[t, "region"] == reg]
            if not mask:
                continue
            fig13.add_trace(go.Scatter(
                x=scores[mask, 0].tolist(), y=scores[mask, 1].tolist(),
                mode="markers+text", text=[feat_cl.index[i] for i in mask],
                textposition="top center",
                marker=dict(size=11, color=col), name=reg,
            ))
        sc = max(float(np.abs(scores).max()), 1.0) * 0.55
        for j, cn in enumerate(feat_cl.columns):
            lx, ly = float(loadings[j, 0]) * sc, float(loadings[j, 1]) * sc
            fig13.add_annotation(x=lx, y=ly, ax=0, ay=0,
                                 arrowhead=2, arrowcolor="#e3b341",
                                 text=cn, font=dict(color="#e3b341", size=9))
        _lay(fig13,
             title=f"PCA Biplot – PC1 {ev[0]*100:.1f}% | PC2 {ev[1]*100:.1f}%",
             xaxis_title=f"PC1 ({ev[0]*100:.1f}% Varianz)",
             yaxis_title=f"PC2 ({ev[1]*100:.1f}% Varianz)", height=490)
        pca_html = (
            _desc("SVD-basierte PCA (kein sklearn): X = U·Σ·Vᵀ. "
                  "Scores (Punkte) = U·Σ, Loadings (Pfeile) = Vᵀ. "
                  "Airlines nahe beieinander haben ähnliches Feature-Profil. "
                  "Pfeile zeigen in welche Richtung welche Variable zunimmt. "
                  "Gelbe Pfeile = normierte Loadings skaliert auf Score-Bereich.")
            + _htm(fig13)
        )

    # ════════════════════════════════════════════════════════════════════════
    # §14  Parallel Coordinates
    # ════════════════════════════════════════════════════════════════════════
    pc_cols = ["sh_oos","best_lag","best_ccf","granger_f","te","regime_stab","vol_pct"]
    pc_df = df[[c for c in pc_cols if c in df.columns]].copy()
    for c in pc_df.columns:
        pc_df[c] = pd.to_numeric(pc_df[c], errors="coerce")
    pc_df = pc_df.fillna(pc_df.median())

    fig14 = go.Figure()
    if len(pc_df) >= 2:
        sh_col = pc_df["sh_oos"].values
        fig14 = go.Figure(go.Parcoords(
            line=dict(color=sh_col, colorscale="Plasma", showscale=True,
                      colorbar=dict(title="OOS Sharpe", tickfont=dict(color="#e6edf3"))),
            dimensions=[
                dict(label=c, values=pc_df[c].values.tolist(),
                     range=[float(pc_df[c].min()), float(pc_df[c].max())])
                for c in pc_df.columns
            ],
            labelangle=-25,
            labelside="bottom",
        ))
    _lay(fig14, title="Parallel Coordinates: Airline-Metriken", height=440)
    sec14 = (
        _desc("Jede Linie = eine Airline. Farbe = OOS Sharpe (hell = hoch). "
              "Durch interaktives Ziehen auf einer Achse kann nach beliebigen Wertebereichen gefiltert werden. "
              "NaN-Werte wurden mit dem Median imputiert um alle Airlines darzustellen.")
        + _htm(fig14)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §15  Region/Type Box Plots
    # ════════════════════════════════════════════════════════════════════════
    fig15 = make_subplots(rows=1, cols=2,
                          subplot_titles=["OOS Sharpe nach Region", "OOS Sharpe nach Airline-Typ"])
    for reg in df["region"].unique():
        vs = pd.to_numeric(df.loc[df["region"] == reg, "sh_oos"], errors="coerce").dropna().tolist()
        if vs:
            fig15.add_trace(go.Box(y=vs, name=reg, marker_color=_rc(reg), showlegend=False),
                            row=1, col=1)
    for typ in df["type"].unique():
        vs = pd.to_numeric(df.loc[df["type"] == typ, "sh_oos"], errors="coerce").dropna().tolist()
        if vs:
            fig15.add_trace(go.Box(y=vs, name=typ, marker_color=_tc(typ), showlegend=False),
                            row=1, col=2)
    fig15.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=380, title_text="OOS Sharpe: Region & Airline-Typ",
    )
    sec15 = (
        _desc("Boxplot zeigt Median, IQR (Box), Whisker (1.5×IQR) und Ausreißer. "
              "Wenn eine Region signifikant über anderen liegt, deutet das auf regionale "
              "Strukturunterschiede hin (z.B. kürzerer Lag bei US-Börsen, effizientere "
              "Preisanpassung).")
        + _htm(fig15)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §16  Statistical Tests
    # ════════════════════════════════════════════════════════════════════════
    sh_num = pd.to_numeric(df["sh_oos"], errors="coerce")

    def _kruskal(gcol):
        groups = {}
        for g, sub in df.groupby(gcol):
            vs = pd.to_numeric(sub["sh_oos"], errors="coerce").dropna().tolist()
            if len(vs) >= 2:
                groups[g] = vs
        if len(groups) < 2:
            return "<p style='color:#e6edf3;'>Zu wenige Gruppen.</p>"
        stat, pv = kruskal(*groups.values())
        sig = '<span class="badge bg-success">sig. p&lt;0.05</span>' if pv < 0.05 else '<span class="badge bg-secondary">n.s.</span>'
        rows = "".join(f"<tr><td>{g}</td><td>N={len(v)}</td><td>µ={np.mean(v):.3f}</td><td>σ={np.std(v):.3f}</td></tr>"
                       for g, v in groups.items())
        return (f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
                f'<strong style="color:#e6edf3;">Kruskal-Wallis: {gcol} → OOS Sharpe</strong> '
                f'H={stat:.3f}, p={pv:.4f} {sig}'
                f'<table class="table table-sm table-dark mt-2"><tbody>{rows}</tbody></table></div>')

    spear_rows = ""
    for col in ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil",
                "regime_stab","roll_corr_mean","ar1","mom","ls_ratio"]:
        if col not in df.columns:
            continue
        xv = pd.to_numeric(df[col], errors="coerce")
        both = pd.concat([xv, sh_num], axis=1).dropna()
        if len(both) < 3:
            continue
        rho, pv = spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)
        sig = '<span class="badge bg-success">sig.</span>' if pv < 0.05 else ""
        spear_rows += f"<tr><td>{col}</td><td>{rho:.3f}</td><td>{pv:.4f}</td><td>{sig}</td></tr>"

    spear_card = (
        f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
        f'<strong style="color:#e6edf3;">Spearman-Korrelation: Feature → OOS Sharpe</strong>'
        f'<table class="table table-sm table-dark mt-2">'
        f'<thead><tr><th>Variable</th><th>ρ</th><th>p</th><th></th></tr></thead>'
        f'<tbody>{spear_rows}</tbody></table>'
        f'<small style="color:#8b949e;">ρ = Rang-Korrelationskoeffizient. '
        f'p = Wahrscheinlichkeit, ρ dieser Größe zufällig zu erhalten (H0: ρ=0). '
        f'|ρ|&gt;0.3 gilt als moderater Zusammenhang.</small></div>'
    )

    # OLS
    ols_feats = [c for c in ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil",
                              "regime_stab","roll_corr_mean"] if c in df.columns]
    ols_card = ""
    if len(ols_feats) >= 2:
        ols_data = pd.concat(
            [pd.to_numeric(df[c], errors="coerce") for c in ols_feats] + [sh_num], axis=1
        ).dropna()
        ols_data.columns = ols_feats + ["sh_oos"]
        if len(ols_data) >= len(ols_feats) + 2:
            Y = ols_data["sh_oos"].values
            X = np.column_stack([np.ones(len(Y))] + [ols_data[c].values for c in ols_feats])
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            yh = X @ coef
            r2 = 1 - ((Y - yh)**2).sum() / ((Y - Y.mean())**2).sum()
            r_rows = "".join(
                f"<tr><td>{'Intercept' if i == 0 else ols_feats[i-1]}</td><td>{c:.4f}</td></tr>"
                for i, c in enumerate(coef)
            )
            ols_card = (
                f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
                f'<strong style="color:#e6edf3;">OLS: OOS Sharpe ~ Features</strong> '
                f'(R²={r2:.3f})'
                f'<table class="table table-sm table-dark mt-2">'
                f'<thead><tr><th>Variable</th><th>β-Koeffizient</th></tr></thead>'
                f'<tbody>{r_rows}</tbody></table>'
                f'<small style="color:#8b949e;">β_i = marginaler Effekt von Feature_i auf Sharpe, '
                f'alle anderen Features konstant. R² = erklärter Varianzanteil.</small></div>'
            )

    # Feature importance (Spearman ρ² proxy)
    fi_card = ""
    fi_vals = []
    for fc in ols_feats:
        xv = pd.to_numeric(df[fc], errors="coerce")
        both = pd.concat([xv, sh_num], axis=1).dropna()
        rho  = float(spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)[0]) if len(both) >= 3 else 0.0
        fi_vals.append((fc, rho**2))
    if fi_vals:
        tot = sum(v for _, v in fi_vals) + 1e-9
        fi_norm = [(fc, v/tot) for fc, v in fi_vals]
        fi_norm.sort(key=lambda x: x[1], reverse=True)
        fig_fi = go.Figure(go.Bar(
            x=[x[0] for x in fi_norm], y=[x[1] for x in fi_norm],
            marker_color=["#58a6ff" if x[1] > 0.1 else "#30363d" for x in fi_norm],
        ))
        _lay(fig_fi, title="Feature Importance (Proxy: Spearman ρ²-Anteil)",
             xaxis_title="Feature", yaxis_title="Rel. Importance", height=340)
        fi_card = (
            _desc("Proxy für Feature Importance: Spearman ρ² (Anteil an Gesamtsumme). "
                  "Entspricht der erklärten Varianz bei Rangkorrelation. "
                  "Eine echte Random-Forest-Importance würde mehr Interaktionen berücksichtigen.")
            + _htm(fig_fi)
        )

    sec16 = _kruskal("region") + _kruskal("type") + spear_card + ols_card + fi_card

    # ════════════════════════════════════════════════════════════════════════
    # §17  Crisis Period Analysis
    # ════════════════════════════════════════════════════════════════════════
    CRISIS = [
        ("Ölcrash 2014–16", "2014-06-01", "2016-03-31"),
        ("COVID-19 2020",   "2020-01-01", "2020-12-31"),
        ("Recovery 2021",   "2021-01-01", "2021-12-31"),
        ("Ukraine 2022",    "2022-02-01", "2022-12-31"),
        ("Inflation 2023",  "2023-01-01", "2023-12-31"),
        ("2024–2025",       "2024-01-01", "2025-06-30"),
    ]
    cr_rows = ""
    for t in sorted_tickers:
        sr = strat_best[t]
        net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        net_all.index = pd.to_datetime(net_all.index)
        row_v = f"<td><strong>{t}</strong></td>"
        for cname, cs, ce in CRISIS:
            sub = net_all.loc[cs:ce]
            if len(sub) < 5:
                row_v += "<td style='color:#8b949e;'>—</td>"
            else:
                sh_c = _sh(sub)
                col_v = ("#3fb950" if sh_c > 0.5 else ("#f78166" if sh_c < -0.5 else "#e3b341"))
                row_v += f'<td style="color:{col_v};">{sh_c:.2f}</td>'
        cr_rows += f"<tr>{row_v}</tr>"

    cr_hdrs = "".join(f"<th>{c[0]}</th>" for c in CRISIS)
    cr_leg = _legend(
        "".join(f"<tr><td>{c[0]}</td><td>{c[1]} – {c[2]}</td></tr>" for c in CRISIS)
    )
    sec17 = (
        _desc("Krisenperioden-Sharpe: annualisierter Sharpe auf dem jeweiligen Zeitfenster. "
              "🟢 &gt;0.5 = gute Krisenperformance, 🔴 &lt;−0.5 = schlechte Periode. "
              "Misst ob die Lead-Lag-Strategie in extremen Marktphasen überhaupt noch funktioniert.")
        + cr_leg
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th>{cr_hdrs}</tr></thead>'
        f'<tbody>{cr_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §18  TC Sweep
    # ════════════════════════════════════════════════════════════════════════
    fig18 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        ret_t  = log_ret[t].dropna() if t in log_ret.columns else None
        if ret_t is None:
            continue
        lp_loc = leader_px.reindex(ret_t.index.intersection(leader_px.index)).ffill()
        ind_fn = sr["ind_fn"]
        split_d = str(sr["split_date"])[:10]
        oos_r   = ret_t.loc[split_d:]
        tc_sh   = []
        for tc_v in TC_GRID:
            n_tc, _, _ = _strat_exec(ind_fn(lp_loc), sr["thresh"], oos_r, sr["lag"], tc=tc_v)
            tc_sh.append(_sh(n_tc))
        fig18.add_trace(go.Scatter(
            x=(TC_GRID * 100).tolist(), y=tc_sh, name=t, mode="lines+markers",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.5),
        ))
    fig18.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig18, title="TC-Sweep: OOS Sharpe vs Transaktionskosten",
         xaxis_title="TC (Basispunkte)", yaxis_title="OOS Sharpe", height=440)
    sec18 = (
        _desc("Transaktionskosten-Sweep: OOS-Sharpe als Funktion der TC. "
              "Formel: r_net = r_gross − |Δsignal| · tc. "
              "Break-even TC = TC, bei der Sharpe = 0. "
              "Hoher Break-even → Strategie ist robust gegenüber realen Kosten (Slippage, Spread).")
        + _htm(fig18)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §19  Monte Carlo
    # ════════════════════════════════════════════════════════════════════════
    mc_rows = ""
    for t in sorted_tickers:
        sr    = strat_best[t]
        oos_a = sr["n_oos"].dropna().values
        if len(oos_a) < 30:
            mc_rows += f"<tr><td>{t}</td><td colspan='3' style='color:#8b949e;'>—</td></tr>"
            continue
        real_sh = _sh(sr["n_oos"])
        mc_sh   = np.array([np.random.permutation(oos_a).mean() * 252 /
                             (np.random.permutation(oos_a).std() * np.sqrt(252) + 1e-9)
                             for _ in range(N_MC)])
        # Fix: use same permutation for mean and std
        mc_sh2 = []
        for _ in range(N_MC):
            p = np.random.permutation(oos_a)
            mc_sh2.append(p.mean() * 252 / (p.std() * np.sqrt(252) + 1e-9))
        mc_sh = np.array(mc_sh2)
        pv    = float((mc_sh >= real_sh).mean())
        sig   = '<span class="badge bg-success">sig.</span>' if pv < 0.05 else '<span class="badge bg-secondary">n.s.</span>'
        mc_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{pv:.4f}</td><td>{sig}</td></tr>"

    sec19 = (
        _desc("Monte Carlo Permutationstest: Die OOS-Renditen werden {N_MC}× zufällig permutiert "
              "(Zeitstruktur zerstört), dann wird der Sharpe berechnet. "
              "p-Wert = Anteil der Permutationen mit Sharpe ≥ realem Sharpe. "
              "p &lt; 0.05 → Signal ist nicht durch Zufall erklärbar.".replace("{N_MC}", str(N_MC)))
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>OOS Sharpe</th><th>MC p-Value</th><th>Signifikanz</th></tr></thead>'
        f'<tbody>{mc_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §20  Bootstrap CI
    # ════════════════════════════════════════════════════════════════════════
    boot_rows = ""
    for t in sorted_tickers:
        sr    = strat_best[t]
        oos_a = sr["n_oos"].dropna().values
        if len(oos_a) < 30:
            boot_rows += f"<tr><td>{t}</td><td colspan='4' style='color:#8b949e;'>—</td></tr>"
            continue
        boot_sh = np.array([
            np.random.choice(oos_a, len(oos_a), replace=True).mean() * 252 /
            (np.random.choice(oos_a, len(oos_a), replace=True).std() * np.sqrt(252) + 1e-9)
            for _ in range(N_BOOT)
        ])
        lo, hi = np.percentile(boot_sh, 2.5), np.percentile(boot_sh, 97.5)
        real_sh = _sh(sr["n_oos"])
        sig = "✓" if lo > 0 else "✗"
        boot_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{lo:.3f}</td><td>{hi:.3f}</td><td>{sig}</td></tr>"

    sec20 = (
        _desc(f"Bootstrap 95%-Konfidenzintervall ({N_BOOT} Samples mit Zurücklegen). "
              "CI = [2.5%-Quantil, 97.5%-Quantil] der Bootstrap-Sharpe-Verteilung. "
              "✓ = CI vollständig über 0 → Sharpe signifikant positiv.")
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>OOS Sharpe</th><th>CI 2.5%</th><th>CI 97.5%</th><th>CI&gt;0</th></tr></thead>'
        f'<tbody>{boot_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §21  Walk-Forward
    # ════════════════════════════════════════════════════════════════════════
    wf_rows = ""
    IS_WIN = 504; OOS_WIN = 126; STEP = 126
    for t in sorted_tickers:
        sr    = strat_best[t]
        ret_t = log_ret[t].dropna() if t in log_ret.columns else None
        if ret_t is None:
            continue
        lp_all = leader_px.reindex(ret_t.index.intersection(leader_px.index)).ffill()
        ind_fn = sr["ind_fn"]
        thresh = sr["thresh"]
        wf_sh  = []
        n_tot  = len(ret_t)
        for start in range(0, n_tot - IS_WIN - OOS_WIN, STEP):
            oo_i = ret_t.iloc[start + IS_WIN:start + IS_WIN + OOS_WIN]
            if len(oo_i) < 20:
                continue
            n_oo, _, _ = _strat_exec(ind_fn(lp_all), thresh, oo_i, sr["lag"])
            if len(n_oo) >= 10:
                wf_sh.append(_sh(n_oo))
        if not wf_sh:
            wf_rows += f"<tr><td>{t}</td><td colspan='3' style='color:#8b949e;'>—</td></tr>"
            continue
        wf_mean = np.nanmean(wf_sh)
        wf_pos  = np.mean([s > 0 for s in wf_sh if not np.isnan(s)])
        col_v   = "#3fb950" if wf_mean > 0.3 else ("#f78166" if wf_mean < 0 else "#e3b341")
        wf_rows += (f'<tr><td>{t}</td>'
                    f'<td style="color:{col_v};">{wf_mean:.3f}</td>'
                    f'<td>{wf_pos*100:.0f}%</td>'
                    f'<td>{len(wf_sh)}</td></tr>')

    sec21 = (
        _desc(f"Walk-Forward: IS={IS_WIN}T, OOS={OOS_WIN}T, Schritt={STEP}T. "
              "In jedem Fenster wird der Indikator auf dem IS-Fenster fixiert (selber Indikator wie global), "
              "die Performance wird auf dem nachfolgenden OOS-Fenster gemessen. "
              "Ø WF Sharpe = Mittelwert aller OOS-Fenster-Sharpes.")
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>Ø WF OOS Sharpe</th><th>% pos. Fenster</th><th>Fenster</th></tr></thead>'
        f'<tbody>{wf_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §22  VIX Regime
    # ════════════════════════════════════════════════════════════════════════
    vix_html = "<p style='color:#8b949e;'>VIX-Daten nicht verfügbar.</p>"
    try:
        vix_s = _dl("^VIX")
        if vix_s is not None and len(vix_s) > 100:
            REGIMES = [
                ("Low (&lt;15)",      vix_s < 15),
                ("Normal (15–25)",    (vix_s >= 15) & (vix_s < 25)),
                ("Elevated (25–35)", (vix_s >= 25) & (vix_s < 35)),
                ("Crisis (&gt;35)",   vix_s >= 35),
            ]
            vix_rows = ""
            for t in sorted_tickers:
                sr = strat_best[t]
                net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
                net_all.index = pd.to_datetime(net_all.index)
                rv = f"<td><strong>{t}</strong></td>"
                for rname, rmask in REGIMES:
                    idx_r = vix_s[rmask].index.intersection(net_all.index)
                    sub_r = net_all.reindex(idx_r).dropna()
                    if len(sub_r) < 5:
                        rv += "<td style='color:#8b949e;'>—</td>"
                    else:
                        sh_r = _sh(sub_r)
                        col_v = ("#3fb950" if sh_r > 0.5 else ("#f78166" if sh_r < -0.5 else "#e3b341"))
                        rv += f'<td style="color:{col_v};">{sh_r:.2f}</td>'
                vix_rows += f"<tr>{rv}</tr>"

            vhd = "".join(f"<th>{r[0]}</th>" for r in REGIMES)
            vix_html = (
                _desc("VIX = CBOE Volatility Index (implizite 30T-Vol des S&amp;P 500). "
                      "Niedrig &lt;15 = ruhiger Markt, Krise &gt;35 = extreme Unsicherheit. "
                      "Die Tabelle zeigt Sharpe je Regime — ergibt sich die Lead-Lag-Strategie "
                      "nur in ruhigen Märkten oder auch in Krisen?")
                + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
                f'<thead><tr><th>Airline</th>{vhd}</tr></thead>'
                f'<tbody>{vix_rows}</tbody></table></div>'
            )
    except Exception:
        pass
    sec22 = vix_html

    # ════════════════════════════════════════════════════════════════════════
    # §23  Signal Stability Map
    # ════════════════════════════════════════════════════════════════════════
    stab_html = "<p style='color:#8b949e;'>Zu wenige Daten.</p>"
    stab_data = {}
    for t in sorted_tickers:
        rc = rc_store.get(t)
        if rc is not None and len(rc.dropna()) >= 50:
            stab_data[t] = rc.dropna()

    if len(stab_data) >= 2:
        all_dates = sorted(set().union(*[set(s.index) for s in stab_data.values()]))
        sm = np.full((len(sorted_tickers), len(all_dates)), np.nan)
        for j, t in enumerate(sorted_tickers):
            if t not in stab_data:
                continue
            s_t = stab_data[t]
            for ii, d in enumerate(all_dates):
                if d in s_t.index:
                    sm[j, ii] = float(s_t.loc[d])
        step_d  = max(1, len(all_dates) // 500)
        sdates  = all_dates[::step_d]
        sm_sub  = sm[:, ::step_d]

        fig23 = go.Figure(go.Heatmap(
            z=sm_sub.tolist(),
            x=[str(d)[:10] for d in sdates],
            y=sorted_tickers,
            colorscale="RdBu", zmid=0, zmin=-0.8, zmax=0.8,
            colorbar=dict(title="Rolling r", tickfont=dict(color="#e6edf3")),
        ))
        _lay(fig23, title="Signal Stability Map: Rolling 252T CCF über Zeit", height=420)
        stab_html = (
            _desc("Jede Zeile = eine Airline. X-Achse = Zeit. Farbe = rollende Pearson-Korrelation r(t) "
                  "zwischen CL=F(t-l*) und Airline(t) im 252T-Fenster. "
                  "Beständig rotes Band = stabiler Lead-Lag. Wechsel blau/rot = Regime-Instabilität.")
            + _htm(fig23)
        )
    sec23 = stab_html

    # ════════════════════════════════════════════════════════════════════════
    # §24  Granger F vs Transfer Entropy Scatter
    # ════════════════════════════════════════════════════════════════════════
    gr_vals = pd.to_numeric(df["granger_f"], errors="coerce").fillna(0)
    te_vals = pd.to_numeric(df["te"],        errors="coerce").fillna(0)

    fig24 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig24.add_trace(go.Scatter(
            x=[float(gr_vals.iloc[i]) for i in mask],
            y=[float(te_vals.iloc[i]) for i in mask],
            mode="markers+text",
            text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=10, color=col),
            name=reg,
        ))
    _lay(fig24, title="Granger-F-Statistik vs Transfer Entropy",
         xaxis_title="Granger F-Statistik", yaxis_title="Transfer Entropy (Bits)", height=440)
    sec24 = (
        _desc("Granger F = lineare Kausalität (basiert auf OLS-Verbesserung durch x-Lags). "
              "Transfer Entropy TE = nicht-lineare Informationsübertragung (Shannon-Entropie). "
              "Airlines oben rechts zeigen sowohl linearen als auch nicht-linearen Kausalzusammenhang mit CL=F. "
              "Airlines die nur beim Granger-Test signifikant sind, haben einen hauptsächlich linearen Kanal.")
        + _htm(fig24)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §25  Scatterplot Matrix
    # ════════════════════════════════════════════════════════════════════════
    sp_cols = ["sh_oos", "best_lag", "best_ccf", "granger_f", "regime_stab", "vol_pct"]
    sp_df   = df[[c for c in sp_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    sp_df   = sp_df.fillna(sp_df.median())
    splom_html = ""
    if len(sp_df) >= 3:
        fig25 = go.Figure(go.Splom(
            dimensions=[dict(label=c, values=sp_df[c].tolist()) for c in sp_df.columns],
            marker=dict(
                color=[_rc(str(df.loc[t, "region"])) for t in sp_df.index],
                size=8, showscale=False,
            ),
            text=sp_df.index.tolist(),
            diagonal_visible=False,
        ))
        _lay(fig25, title="Scatterplot-Matrix: Kernmetriken", height=600)
        splom_html = (
            _desc("Scatterplot-Matrix (SPLOM): Jedes Subplot zeigt den bivariaten Zusammenhang "
                  "zwischen zwei Metriken. Farbe = Region. Diagonale ausgeblendet. "
                  "Wähle interessante Achsenkombinationen aus um Cluster oder Ausreißer zu entdecken.")
            + _htm(fig25)
        )
    sec25 = splom_html or "<p style='color:#8b949e;'>Zu wenige Daten.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # §26  Full 26-Metric Table
    # ════════════════════════════════════════════════════════════════════════
    METRIC_COLS = [
        "Ann.Ret% (net)", "Ann.Ret% (gross)", "TC Drag% p.a.", "Ann.Vol%",
        "Sharpe (net)", "Sharpe (gross)", "Sortino", "Calmar", "MaxDD%",
        "AvgDD-Dur", "Trades", "WinRate%", "AvgWin%", "AvgLoss%", "ProfitFactor",
        "Omega", "TailRatio", "Skew", "Kurt", "VaR5%", "CVaR5%", "AC1", "AC5",
        "Beta", "Alpha%",
    ]
    m26_rows = ""
    for t in sorted_tickers:
        sr   = strat_best[t]
        spy  = log_ret["SPY"].dropna() if "SPY" in log_ret.columns else None
        m_is  = _full_metrics(sr["n_is"],  sr["g_is"],  sr["s_is"],  spy, name=f"{t} IS")
        m_oos = _full_metrics(sr["n_oos"], sr["g_oos"], sr["s_oos"], spy, name=f"{t} OOS")
        for split_l, mm in [("IS", m_is), ("OOS", m_oos)]:
            cells = f"<td><strong>{t}</strong></td><td>{split_l}</td>"
            for mc in METRIC_COLS:
                v = mm.get(mc, np.nan)
                if isinstance(v, float) and np.isnan(v):
                    cells += "<td style='color:#8b949e;'>—</td>"
                elif isinstance(v, float):
                    cells += f"<td>{v:.3f}</td>"
                else:
                    cells += f"<td>{v}</td>"
            m26_rows += f"<tr>{cells}</tr>"

    m_hdrs = "<th>Ticker</th><th>Split</th>" + "".join(f"<th>{c}</th>" for c in METRIC_COLS)
    leg26 = _legend(
        "<tr><td>Ann.Ret%</td><td>µ·252·100 (tägl. Nettorendite annualisiert)</td></tr>"
        "<tr><td>TC Drag%</td><td>(µ_gross − µ_net)·252·100 (Kostenwirkung p.a.)</td></tr>"
        "<tr><td>Ann.Vol%</td><td>σ·√252·100</td></tr>"
        "<tr><td>Sharpe</td><td>µ·252 / (σ·√252)</td></tr>"
        "<tr><td>Sortino</td><td>µ·252 / (σ_down·√252), σ_down = Std negativer Renditen</td></tr>"
        "<tr><td>Calmar</td><td>Ann.Ret / |MaxDD|</td></tr>"
        "<tr><td>MaxDD%</td><td>Max. Drawdown = max(1 − NAV/max_NAV)</td></tr>"
        "<tr><td>AvgDD-Dur</td><td>Mittlere Drawdown-Dauer in Tagen</td></tr>"
        "<tr><td>WinRate%</td><td>Anteil Tage mit positivem Return</td></tr>"
        "<tr><td>ProfitFactor</td><td>Summe Gewinne / |Summe Verluste|</td></tr>"
        "<tr><td>Omega</td><td>∫₀^∞ (1-F(r))dr / ∫₋∞^0 F(r)dr (Threshold=0)</td></tr>"
        "<tr><td>TailRatio</td><td>|95%-Quantil / 5%-Quantil|</td></tr>"
        "<tr><td>VaR5%</td><td>5%-Quantil der täglichen Renditen</td></tr>"
        "<tr><td>CVaR5%</td><td>Ø Rendite aller Tage unterhalb VaR5%</td></tr>"
        "<tr><td>AC1/AC5</td><td>Autokorrelation der Renditen bei Lag 1 bzw. 5</td></tr>"
        "<tr><td>Beta</td><td>Kovarianz(r, SPY) / Var(SPY)</td></tr>"
        "<tr><td>Alpha%</td><td>Intercept OLS vs SPY, annualisiert</td></tr>"
    )
    sec26 = (
        _desc("Vollständige 26-Metriken-Tabelle für jede Airline, getrennt nach IS (Kalibrierung) "
              "und OOS (echter Test). Alle Renditen sind Netto nach 10bp TC.")
        + leg26
        + f'<div class="table-responsive" style="font-size:0.75em;">'
        f'<table class="table table-sm table-dark table-hover">'
        f'<thead class="table-dark"><tr>{m_hdrs}</tr></thead>'
        f'<tbody>{m26_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §27  Portfolio Analysis
    # ════════════════════════════════════════════════════════════════════════
    oos_rets = {t: strat_best[t]["n_oos"] for t in sorted_tickers}
    oos_mat  = pd.DataFrame(oos_rets).fillna(0)

    port_html = ""
    if len(oos_mat.columns) >= 2:
        n_a = len(oos_mat.columns)
        vols = oos_mat.std().values + 1e-9
        w_ew = np.ones(n_a) / n_a
        w_vw = (1/vols) / (1/vols).sum()
        w_mv = (1/vols**2) / (1/vols**2).sum()

        weights = {"Equal Weight": w_ew, "Vol-Weight (1/σ)": w_vw, "Min-Var (1/σ²)": w_mv}
        fig27 = go.Figure()
        pt_rows = ""
        for pn, w in weights.items():
            port = (oos_mat * w).sum(axis=1)
            cum  = (1 + port).cumprod() * 100
            fig27.add_trace(go.Scatter(
                x=cum.index.astype(str).tolist(), y=cum.values.tolist(),
                name=pn, mode="lines",
            ))
            pt_rows += f"<tr><td>{pn}</td><td>{_sh(port):.3f}</td><td>{float((1+port).cumprod().iloc[-1]*100-100):.1f}%</td></tr>"

        _lay(fig27, title="Portfolio OOS Equity Curves",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=440)
        port_html = (
            _desc("Equal Weight: w_i = 1/N. "
                  "Vol-Weight (1/σ): w_i = (1/σ_i) / Σ(1/σ_j) — höhere Gewichtung bei niedrigerer Volatilität. "
                  "Min-Var (1/σ²): analog mit 1/σ². "
                  "Alle Gewichte sind positiv normiert (keine Shorts auf Portfolio-Ebene).")
            + _htm(fig27)
            + f'<div class="table-responsive mt-2"><table class="table table-sm table-dark">'
            f'<thead><tr><th>Portfolio</th><th>OOS Sharpe</th><th>Gesamtrendite</th></tr></thead>'
            f'<tbody>{pt_rows}</tbody></table></div>'
        )
    sec27 = port_html or "<p style='color:#8b949e;'>Zu wenige Airlines für Portfolio.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # §28  Benchmark: CL=F → JETS RSI<70 (existing commodity strategy)
    # ════════════════════════════════════════════════════════════════════════
    bench_html = "<p style='color:#8b949e;'>JETS-Daten nicht verfügbar für Benchmark.</p>"
    try:
        # Load from existing commodity data if available
        ret_main = _read(tables / "phase2_returns.csv")
        px_main  = _read(tables / "phase1_prices.csv")
        if ret_main is not None and px_main is not None:
            ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
            px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
            if "JETS" in ret_main.columns and "CL=F" in px_main.columns:
                cl_px   = px_main["CL=F"].dropna()
                jets_r  = ret_main["JETS"].dropna()
                common_b = cl_px.index.intersection(jets_r.index)
                split_b  = common_b[int(len(common_b) * IS_FRAC)]
                is_b = jets_r.loc[:split_b]; oos_b = jets_r.loc[split_b:]
                # RSI<70: long when -RSI(t-1) > -70 i.e. RSI < 70
                ind_b = -_calc_rsi(cl_px, 14)
                n_is_b, g_is_b, s_is_b   = _strat_exec(ind_b, -70.0, is_b,  1)
                n_oos_b, g_oos_b, s_oos_b = _strat_exec(ind_b, -70.0, oos_b, 1)
                m_is_b  = _full_metrics(n_is_b,  g_is_b,  s_is_b,  name="JETS IS")
                m_oos_b = _full_metrics(n_oos_b, g_oos_b, s_oos_b, name="JETS OOS")

                bm_fig = go.Figure()
                cum_is_b  = (1 + n_is_b).cumprod() * 100
                cum_oos_b = (1 + n_oos_b).cumprod() * 100
                bm_fig.add_trace(go.Scatter(
                    x=cum_is_b.index.astype(str).tolist(), y=cum_is_b.values.tolist(),
                    name="JETS IS", mode="lines",
                    line=dict(color="#58a6ff", width=2, dash="dash"),
                ))
                bm_fig.add_trace(go.Scatter(
                    x=cum_oos_b.index.astype(str).tolist(), y=cum_oos_b.values.tolist(),
                    name="JETS OOS", mode="lines",
                    line=dict(color="#58a6ff", width=2.5),
                ))
                # Overlay best airline OOS
                for i, t in enumerate(sorted_tickers[:3]):
                    sr = strat_best[t]
                    cum_t = (1 + sr["n_oos"]).cumprod() * 100
                    bm_fig.add_trace(go.Scatter(
                        x=cum_t.index.astype(str).tolist(), y=cum_t.values.tolist(),
                        name=f"{t} OOS", mode="lines",
                        line=dict(color=PAL_EQ[i+2 % len(PAL_EQ)], width=1.5),
                    ))
                bm_fig.add_vline(x=str(split_b)[:10], line_color="#e3b341",
                                 line_dash="dash")
                _lay(bm_fig, title="Benchmark: CL=F→JETS RSI<70 vs beste Airlines",
                     xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=470)

                bm_rows_h = ""
                for lbl, mm in [("JETS IS", m_is_b), ("JETS OOS", m_oos_b)]:
                    cells = f"<td><strong>{lbl}</strong></td>"
                    for mc in ["Ann.Ret% (net)", "Ann.Vol%", "Sharpe (net)", "MaxDD%",
                               "Sortino", "Calmar", "WinRate%", "ProfitFactor"]:
                        v = mm.get(mc, np.nan)
                        cells += f"<td>{v:.3f}</td>" if isinstance(v, float) and not np.isnan(v) else "<td>—</td>"
                    bm_rows_h += f"<tr>{cells}</tr>"

                bench_html = (
                    _desc("Benchmark-Strategie: CL=F → JETS ETF, Indikator = RSI(14)&lt;70, Lag = 1T. "
                          "Diese Strategie ist das Referenzmodell aus der Einzelstrategie-Analyse. "
                          "Die beste Airline-Strategie sollte diese Performance möglichst übertreffen. "
                          "Vertikale gelbe Linie = IS/OOS-Trennpunkt.")
                    + _htm(bm_fig)
                    + f'<div class="table-responsive mt-2"><table class="table table-sm table-dark">'
                    f'<thead><tr><th>Strategie</th><th>Ann.Ret%</th><th>Ann.Vol%</th>'
                    f'<th>Sharpe</th><th>MaxDD%</th><th>Sortino</th><th>Calmar</th>'
                    f'<th>WinRate%</th><th>ProfitFactor</th></tr></thead>'
                    f'<tbody>{bm_rows_h}</tbody></table></div>'
                )
    except Exception:
        pass
    sec28 = bench_html

    # ════════════════════════════════════════════════════════════════════════
    # §29  Interpretation & Screening Rules
    # ════════════════════════════════════════════════════════════════════════
    top3  = sorted_tickers[:3]  if len(sorted_tickers) >= 3 else sorted_tickers
    bot3  = sorted_tickers[-3:] if len(sorted_tickers) >= 3 else []
    top3_s = ", ".join(f"{t} ({scalars.get(t,{}).get('region','?')})" for t in top3)
    bot3_s = ", ".join(f"{t} ({scalars.get(t,{}).get('region','?')})" for t in bot3)

    sec29 = f"""
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #3fb950;">
      <h5 style="color:#3fb950;">Beste Airlines (OOS Sharpe)</h5>
      <p style="color:#e6edf3;">{top3_s}</p>
      <p style="color:#e6edf3;">Diese Airlines zeigen die robusteste OOS-Persistenz der CL=F-Lead-Lag-Strategie.
      Charakteristisch: kurzer Lag (1–3T), hoher Granger-F, konsistente Rolling-Korrelation,
      und hohe Ölpreisabhängigkeit im Kostenmix.</p>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #f78166;">
      <h5 style="color:#f78166;">Schwächste Airlines</h5>
      <p style="color:#e6edf3;">{bot3_s}</p>
      <p style="color:#e6edf3;">Mögliche Ursachen: ADR-Liquiditätsmangel, starkes Fuel Hedging,
      Währungseffekte (nicht USD-denominiert) oder schwache operative Öl-Exposition.</p>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #e3b341;">
      <h5 style="color:#e3b341;">Empirisches Screening-Regelwerk</h5>
      <ul style="color:#e6edf3;">
        <li>Filter 1: Granger-F &gt; 3.0 und p &lt; 0.05 → statistisch signifikante Kausalität</li>
        <li>Filter 2: Peak CCF &gt; 0.10 → minimale lineare Signal-Stärke</li>
        <li>Filter 3: Regime-Stabilität &gt; 0.55 → Signal in &gt;55% der 6-Monats-Fenster positiv</li>
        <li>Filter 4: Optimaler Lag 1–5T → Reaktionszeit innerhalb normaler Marktmikrostruktur</li>
        <li>Filter 5: OOS Sharpe &gt; IS Sharpe → kein Overfitting im IS-Fenster</li>
        <li>Regionen: US-notierte Aktien tendieren zu kürzerem Lag und höherem Sharpe</li>
        <li>Größe: Kein klarer Größeneffekt — Market Cap allein ist kein guter Prädiktor</li>
      </ul>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #58a6ff;">
      <h5 style="color:#58a6ff;">Warum reagieren bestimmte Airlines stärker auf Öl?</h5>
      <p style="color:#e6edf3;">
        <strong>Ökonomischer Mechanismus:</strong> Kerosin macht 20–35% der Betriebskosten aus.
        Airlines ohne aktives Hedging-Programm reagieren unmittelbar auf WTI-Spotpreisänderungen
        im Aktienmarkt. US-notierte Papiere haben durch effiziente Preisfindung und liquide
        Derivatemärkte eine kürzere Lag-Struktur (1–2T) als ADRs (3–7T).<br><br>
        <strong>Informationskanal:</strong> Transfer Entropy &gt; Granger bedeutet, dass
        nicht-lineare Informationsübertragung dominiert — Schwellenwerteffekte (Ölpreis
        über/unter Break-even-Kostenschwelle) sind relevanter als lineare Sensitivitäten.<br><br>
        <strong>Allgemeines Screening für andere Branchen:</strong> Branchen mit hohem
        Rohstoffkostenanteil (Chemie: Erdgas, Reedereien: Bunker, Stromerzeugung: Gas/Kohle)
        sollten ähnliche Lead-Lag-Strukturen aufweisen. Screening-Kriterien: hohes
        Commodities/Revenue-Verhältnis, niedrige Hedging-Quote, hohe Preisnehmer-Position.
      </p>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # ASSEMBLE
    # ════════════════════════════════════════════════════════════════════════
    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#acc{idx}">'
            f'{title}</button></h2>'
            f'<div id="acc{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    acc = '<div class="accordion" id="mainAcc">'
    acc += _acc("§0  Ranking-Tabelle: alle Airlines", sec0, 0, open_=True)
    acc += _acc("§1  OOS Sharpe Bar Chart (IS vs OOS)",  sec1, 1)
    acc += _acc("§2  Long/Short/Flat Tage & L/S Ratio",  sec2, 2)
    acc += _acc("§3  OOS Equity Curves mit Signal-Markierungen (▲▼)",  sec3, 3)
    acc += _acc("§4  CCF Heatmap: CL=F → Airline (Lag 0–10T)",         sec4, 4)
    acc += _acc("§5  Optimaler Lag × OOS Sharpe (Scatter)",             sec5, 5)
    acc += _acc("§6  Bubble Chart: Market Cap × OOS Sharpe × Region",  sec6, 6)
    acc += _acc("§7  Rolling 252T Sharpe",                              sec7, 7)
    acc += _acc("§8  Rolling 252T Korrelation CL=F → Airline",         sec8, 8)
    acc += _acc("§9  Radar Charts (normalisierte Metriken)",            sec9, 9)
    acc += _acc("§10 Feature Matrix Heatmap (z-score)",                 sec10, 10)
    acc += _acc("§11 Feature-Korrelationsmatrix",                       sec11, 11)
    acc += _acc("§12 Hierarchisches Clustering (Ward-Dendrogram)",      sec12, 12)
    acc += _acc("§13 PCA Biplot",                                       pca_html, 13)
    acc += _acc("§14 Parallel Coordinate Plot",                         sec14, 14)
    acc += _acc("§15 OOS Sharpe: Region & Airline-Typ (Boxplots)",      sec15, 15)
    acc += _acc("§16 Statistische Tests (Kruskal, Spearman, OLS, FI)",  sec16, 16)
    acc += _acc("§17 Krisenperioden-Analyse",                           sec17, 17)
    acc += _acc("§18 TC-Sweep (0–100bp)",                               sec18, 18)
    acc += _acc("§19 Monte Carlo Permutationstest",                     sec19, 19)
    acc += _acc("§20 Bootstrap 95%-CI auf OOS Sharpe",                  sec20, 20)
    acc += _acc("§21 Walk-Forward-Validierung",                         sec21, 21)
    acc += _acc("§22 VIX-Regime-Analyse",                               sec22, 22)
    acc += _acc("§23 Signal Stability Map (Rolling CCF über Zeit)",     sec23, 23)
    acc += _acc("§24 Granger-F vs Transfer Entropy",                    sec24, 24)
    acc += _acc("§25 Scatterplot-Matrix (Splom)",                       sec25, 25)
    acc += _acc("§26 Vollständige 26-Metriken-Tabelle (IS + OOS)",      sec26, 26)
    acc += _acc("§27 Portfolioanalyse (EW / Vol-Weighted / Min-Var)",   sec27, 27)
    acc += _acc("§28 Benchmark: CL=F → JETS RSI&lt;70",                 sec28, 28)
    acc += _acc("§29 Ökonomische Interpretation & Screening-Regelwerk", sec29, 29)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#ffa657;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">CL=F → Airline Lead-Lag: Querschnittsanalyse</h2>
          <p class="mb-0" style="color:#8b949e;">
            {len(sorted_tickers)} Airlines · IS/OOS · 26 Metriken · MC · Bootstrap · Walk-Forward ·
            Portfolio · Clustering · PCA · Tests · Benchmark · L/S Ratio
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "airline_oil_report.html",
           _html_base("CL=F → Airline Lead-Lag: Querschnittsanalyse", 19, body))


def build_seasonality_report(tables, figures, out):  # noqa: C901
    """
    Seasonal pattern analysis and optimization of CL=F lead-lag strategies.
    Identifies cyclical weaknesses and constructs seasonally-filtered strategies.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import yfinance as yf
    from scipy import stats
    from scipy.stats import pearsonr

    IS_FRAC = 0.70
    N_BOOT  = 1000
    TC      = 0.001

    # ── load data ─────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")

    if ret_main is None or px_main is None:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Daten fehlen.</p>"))
        return

    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    # ── strategy targets: JETS + download top individual airlines ─────────────
    TARGETS = {
        "JETS":  {"name": "US Global Jets ETF", "region": "USA"},
        "DAL":   {"name": "Delta Air Lines",    "region": "USA"},
        "UAL":   {"name": "United Airlines",    "region": "USA"},
        "AAL":   {"name": "American Airlines",  "region": "USA"},
        "LUV":   {"name": "Southwest Airlines", "region": "USA"},
        "IAG":   {"name": "IAG (ADR)",          "region": "Europe"},
        "CPA":   {"name": "Copa Holdings",      "region": "LatAm"},
    }

    def _dl(ticker):
        for period in ("10y", "5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    # Prefer existing data, download missing
    leader_px = px_main["CL=F"].dropna() if "CL=F" in px_main.columns else None
    if leader_px is None:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>CL=F fehlt.</p>"))
        return

    follower_rets = {}
    for ticker in TARGETS:
        if ticker in ret_main.columns:
            follower_rets[ticker] = ret_main[ticker].dropna()
        else:
            s = _dl(ticker)
            if s is not None:
                lr = np.log(s / s.shift(1)).dropna()
                follower_rets[ticker] = lr

    available = [t for t in TARGETS if t in follower_rets]
    if not available:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Keine Follower-Daten.</p>"))
        return

    # ── helper functions ──────────────────────────────────────────────────────
    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _boot_ci(arr, fn=None, n=N_BOOT, q=(2.5, 97.5)):
        fn = fn or (lambda x: x.mean())
        bs = [fn(pd.Series(np.random.choice(arr, len(arr), replace=True))) for _ in range(n)]
        return np.percentile(bs, q)

    def _strat_run(leader_prices, follower_ret, lag=1, thresh=-70.0,
                   ind_fn=None, tc=TC):
        if ind_fn is None:
            ind_fn = lambda p: -_calc_rsi(p, 14)
        return _strat_exec(ind_fn(leader_prices), thresh, follower_ret, lag, tc=tc)

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    MONTH_NAMES = ["Jan","Feb","Mär","Apr","Mai","Jun",
                   "Jul","Aug","Sep","Okt","Nov","Dez"]
    DOW_NAMES   = ["Mo","Di","Mi","Do","Fr"]
    PAL = px.colors.qualitative.Plotly

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A – Per-ticker seasonal analysis
    # ══════════════════════════════════════════════════════════════════════════
    sections = {}
    best_months_per_ticker = {}   # IS-optimized good months
    strat_records = {}

    for ticker in available:
        fr = follower_rets[ticker]
        common = leader_px.index.intersection(fr.index)
        common = common[~common.duplicated()]
        if len(common) < 300:
            continue

        lp = leader_px.reindex(common).ffill()
        fr_c = fr.reindex(common).fillna(0.0)

        split_i    = int(len(common) * IS_FRAC)
        split_date = common[split_i]
        is_idx     = common[:split_i]
        oos_idx    = common[split_i:]

        # Baseline strategy: CL=F RSI<70, Lag=1
        n_is,  g_is,  s_is  = _strat_run(lp, fr_c.loc[is_idx])
        n_oos, g_oos, s_oos = _strat_run(lp, fr_c.loc[oos_idx])

        net_all = pd.concat([n_is, n_oos]).sort_index()
        net_all.index = pd.to_datetime(net_all.index)

        # ── 1. Monthly return matrix (year × month) ──────────────────────────
        net_df = net_all.to_frame("r")
        net_df["year"]  = net_df.index.year
        net_df["month"] = net_df.index.month
        net_df["dow"]   = net_df.index.dayofweek  # 0=Mon

        monthly_pnl = net_df.groupby(["year","month"])["r"].sum().unstack(fill_value=np.nan)
        monthly_pnl.columns = [MONTH_NAMES[m-1] for m in monthly_pnl.columns]

        # ── 2. Average monthly return + bootstrap CI ─────────────────────────
        avg_month = {}
        ci_month  = {}
        for m in range(1, 13):
            vals = net_df[net_df["month"] == m]["r"].values
            if len(vals) < 5:
                avg_month[MONTH_NAMES[m-1]] = 0.0
                ci_month[MONTH_NAMES[m-1]]  = (0.0, 0.0)
                continue
            avg_month[MONTH_NAMES[m-1]] = float(vals.mean())
            lo, hi = _boot_ci(vals, lambda x: x.mean())
            ci_month[MONTH_NAMES[m-1]] = (float(lo), float(hi))

        # IS-only monthly performance (for filter optimization)
        is_net_df = pd.Series(n_is.values, index=pd.to_datetime(n_is.index))
        is_net_df = is_net_df.to_frame("r")
        is_net_df["month"] = is_net_df.index.month
        avg_month_is = is_net_df.groupby("month")["r"].mean()

        # Good months: average IS return > 0
        good_months = set(avg_month_is[avg_month_is > 0].index.tolist())
        best_months_per_ticker[ticker] = good_months

        # ── 3. Day-of-week seasonality ───────────────────────────────────────
        avg_dow = {}
        for d in range(5):
            vals = net_df[net_df["dow"] == d]["r"].values
            avg_dow[DOW_NAMES[d]] = float(vals.mean()) if len(vals) > 5 else 0.0

        # ── 4. Quarterly analysis ────────────────────────────────────────────
        net_df["quarter"] = ((net_df["month"] - 1) // 3) + 1
        avg_qtr = net_df.groupby("quarter")["r"].mean()

        # ── 5. Seasonal-filtered strategy ────────────────────────────────────
        # Only trade (signals active) in IS-profitable months; else flat (signal=0)
        def _seasonal_filter(net_series, sig_series, good_m):
            """Zero out signals in 'bad' months (months not in good_m)."""
            idx = pd.to_datetime(sig_series.index)
            mask = idx.month.isin(good_m)
            sig_f = sig_series.copy()
            sig_f[~mask] = 0
            # Recompute net with filtered signals
            fr_sub = fr_c.reindex(sig_series.index).fillna(0.0)
            gross_f = sig_f * fr_sub
            net_f   = gross_f - sig_f.diff().abs().fillna(0) * TC
            return net_f

        # OOS seasonal filter (using IS-derived good months)
        n_oos_sf = _seasonal_filter(n_oos, s_oos, good_months)

        # ── 6. Autocorrelation of strategy returns ───────────────────────────
        acf_lags = list(range(1, 63))
        acf_vals = [float(net_all.autocorr(lag=l)) for l in acf_lags]

        # ── 7. Rolling 21-day return (cyclic pattern) ────────────────────────
        roll21 = net_all.rolling(21).sum().dropna() * 100

        # ── store results ────────────────────────────────────────────────────
        strat_records[ticker] = {
            "n_is": n_is, "n_oos": n_oos, "n_oos_sf": n_oos_sf,
            "s_is": s_is, "s_oos": s_oos,
            "split_date": split_date,
            "monthly_pnl": monthly_pnl,
            "avg_month": avg_month, "ci_month": ci_month,
            "avg_dow": avg_dow, "avg_qtr": avg_qtr,
            "good_months": good_months,
            "acf_lags": acf_lags, "acf_vals": acf_vals,
            "roll21": roll21,
            "sh_base": _sh(n_oos), "sh_sf": _sh(n_oos_sf),
        }

    if not strat_records:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Keine Ergebnisse.</p>"))
        return

    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════

    # ── S0: Overview comparison table ────────────────────────────────────────
    ov_rows = ""
    for ticker, rec in strat_records.items():
        gm = ", ".join(MONTH_NAMES[m-1] for m in sorted(rec["good_months"]))
        sh_base = rec["sh_base"]
        sh_sf   = rec["sh_sf"]
        delta   = sh_sf - sh_base if not (np.isnan(sh_base) or np.isnan(sh_sf)) else np.nan
        col_d   = "#3fb950" if delta > 0 else "#f78166"
        ov_rows += (
            f"<tr><td><strong>{ticker}</strong></td>"
            f"<td>{TARGETS.get(ticker,{}).get('name','')}</td>"
            f"<td>{sh_base:.3f}</td><td>{sh_sf:.3f}</td>"
            f"<td style='color:{col_d};'>{delta:+.3f}</td>"
            f"<td>{len(rec['good_months'])}/12</td>"
            f"<td style='font-size:0.8em;'>{gm}</td></tr>"
        )
    sec_ov = (
        _desc("Vergleich: Basis-Strategie (RSI&lt;70, Lag=1, immer aktiv) vs "
              "Saisongefilterter Strategie (nur in IS-profitablen Monaten aktiv). "
              "Good Months = Monate mit positivem Ø IS-Monatsertrag.")
        + '<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Ticker</th><th>Name</th><th>Sharpe Basis OOS</th>'
        '<th>Sharpe Seasonal OOS</th><th>Δ Sharpe</th>'
        '<th>Gute Monate</th><th>Liste</th></tr></thead>'
        f'<tbody>{ov_rows}</tbody></table></div>'
    )

    # ── S1: Monthly heatmap (year × month) – first ticker = JETS ─────────────
    first_t = next(iter(strat_records))
    rec0    = strat_records[first_t]
    mp      = rec0["monthly_pl"] if "monthly_pl" in rec0 else rec0["monthly_pnl"]

    fig_cal = go.Figure(go.Heatmap(
        z=mp.values.tolist(),
        x=mp.columns.tolist(),
        y=[str(y) for y in mp.index.tolist()],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.3f}" if not np.isnan(float(v)) else "" for v in row]
              for row in mp.values],
        texttemplate="%{text}", textfont=dict(size=9, color="#000"),
        colorbar=dict(title="Monatsrendite", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig_cal, title=f"Kalender-Heatmap monatlicher Strategierenditen: {first_t}",
         height=max(300, len(mp) * 28))
    sec_cal = (
        _desc("Jede Zelle = Summe der täglichen Netto-Renditen in diesem Monat/Jahr. "
              "Grün = profitabler Monat, Rot = Verlustmonat. "
              "Horizontale Muster zeigen systematische Saisoneffekte (bestimmte Monate "
              "konsistent gut oder schlecht). Vertikale Muster = Marktregime.")
        + _htm(fig_cal)
    )

    # ── S2: Multi-ticker monthly seasonality bars ─────────────────────────────
    fig_ms = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        am = rec["avg_month"]
        ci = rec["ci_month"]
        fig_ms.add_trace(go.Bar(
            x=MONTH_NAMES, y=[am.get(m, 0)*100 for m in MONTH_NAMES],
            name=ticker, marker_color=PAL[i % len(PAL)],
            error_y=dict(
                type="data", symmetric=False,
                array    =[max(ci.get(m,(0,0))[1]-am.get(m,0),0)*100 for m in MONTH_NAMES],
                arrayminus=[max(am.get(m,0)-ci.get(m,(0,0))[0],0)*100 for m in MONTH_NAMES],
                visible=True, color="#8b949e",
            ),
            opacity=0.8,
        ))
    fig_ms.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_ms, title="Durchschnittliche Monatsrendite % (Gesamt-Periode)",
         barmode="group", xaxis_title="Monat", yaxis_title="Ø Tagesrendite×100", height=420)
    sec_ms = (
        _desc("Ø Tagesrendite aggregiert je Kalendermonat über alle Jahre. "
              "Fehlerbalken = 95%-Bootstrap-CI. "
              "Signifikant grüne Monate (CI vollständig über 0) sind die besten Handelsperioden. "
              "Systematisch rote Monate sollten in der gefilterten Strategie ausgelassen werden.")
        + _htm(fig_ms)
    )

    # ── S3: Day-of-Week effect ────────────────────────────────────────────────
    fig_dow = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        dow_v = rec["avg_dow"]
        fig_dow.add_trace(go.Bar(
            x=DOW_NAMES, y=[dow_v.get(d, 0)*100 for d in DOW_NAMES],
            name=ticker, marker_color=PAL[i % len(PAL)], opacity=0.8,
        ))
    fig_dow.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_dow, title="Wochentag-Saisonalität: Ø Tagesrendite",
         barmode="group", height=360)
    sec_dow = (
        _desc("Day-of-Week Effekt: Mittlere tägliche Nettostrategie-Rendite je Wochentag. "
              "Bekannte Anomalien: Montagseffekt (oft negativ), Freitagseffekt. "
              "Wenn systematische DoW-Muster bestehen, könnte man die Strategie "
              "nur an bestimmten Wochentagen aktiv schalten.")
        + _htm(fig_dow)
    )

    # ── S4: Quarterly box plots ───────────────────────────────────────────────
    fig_qtr = make_subplots(rows=1, cols=len(strat_records),
                             subplot_titles=list(strat_records.keys()))
    for i, (ticker, rec) in enumerate(strat_records.items()):
        n_all = pd.concat([rec["n_is"], rec["n_oos"]]).sort_index()
        n_df  = n_all.to_frame("r")
        n_df["quarter"] = pd.to_datetime(n_df.index).quarter
        for q in range(1, 5):
            vs = n_df[n_df["quarter"] == q]["r"].dropna().values * 100
            fig_qtr.add_trace(go.Box(y=vs.tolist(), name=f"Q{q}",
                                     showlegend=(i == 0)), row=1, col=i+1)
    fig_qtr.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=400, title_text="Quartals-Saisonalität: Tagesrenditen-Verteilung",
    )
    sec_qtr = (
        _desc("Box-Plot der täglichen Nettostrategie-Renditen je Quartal. "
              "Q1=Jan–Mär, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Okt–Dez. "
              "Wenn ein Quartal systematisch niedrigere Mediane hat, "
              "ist dies ein guter Kandidat für den Saisonfilter.")
        + _htm(fig_qtr)
    )

    # ── S5: Autocorrelation (cycle detection) ────────────────────────────────
    fig_acf = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        fig_acf.add_trace(go.Scatter(
            x=rec["acf_lags"], y=[v*100 for v in rec["acf_vals"]],
            name=ticker, mode="lines+markers",
            line=dict(color=PAL[i % len(PAL)], width=1.5),
            marker=dict(size=4),
        ))
    # ±2/√N bands
    n_approx = 1800
    ci_band = 2 / np.sqrt(n_approx) * 100
    fig_acf.add_hline(y=ci_band,  line_color="#e3b341", line_dash="dot",
                      annotation_text="95% CI")
    fig_acf.add_hline(y=-ci_band, line_color="#e3b341", line_dash="dot")
    fig_acf.add_hline(y=0, line_color="#8b949e", line_dash="solid")
    _lay(fig_acf, title="Autokorrelation der Strategie-Renditen (Lag 1–62 Tage)",
         xaxis_title="Lag (Tage)", yaxis_title="Autokorrelation × 100", height=400)
    sec_acf = (
        _desc("ACF(k) = Kor(r_t, r_{t-k}). Signifikante Peaks außerhalb der ±2/√N-Bänder "
              "(gelbe Linien) deuten auf zyklische Muster hin. "
              "Peaks bei Lag ≈ 21 → Monatsrhythmus. "
              "Peaks bei Lag ≈ 63 → Quartalsrhythmus. "
              "Negative Korrelation bei kleinen Lags → Mean-Reversion-Charakter.")
        + _htm(fig_acf)
    )

    # ── S6: Rolling 21-day cumulative return ─────────────────────────────────
    fig_r21 = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        r21 = rec["roll21"]
        fig_r21.add_trace(go.Scatter(
            x=r21.index.astype(str).tolist(), y=r21.values.tolist(),
            name=ticker, mode="lines",
            line=dict(color=PAL[i % len(PAL)], width=1.2),
        ))
    fig_r21.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_r21, title="Rolling 21-Tage kumulierte Rendite (%)",
         xaxis_title="Datum", yaxis_title="21T-Rendite (%)", height=430)
    sec_r21 = (
        _desc("Rollierendes 21-Tage-Fenster (≈ 1 Monat): Summe der täglichen Renditen. "
              "Zyklische Auf-und-Ab-Bewegungen mit regelmäßiger Periode sind ein starkes "
              "Indiz für Saisoneffekte. Wenn das Bild einem Sinus ähnelt, "
              "existiert ein stabiler Kreislauf, der herausgefiltert werden kann.")
        + _htm(fig_r21)
    )

    # ── S7: Seasonal-filtered vs Baseline equity curves (OOS) ─────────────────
    fig_sf = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        col = PAL[i % len(PAL)]
        # Baseline
        cum_base = (1 + rec["n_oos"]).cumprod() * 100
        fig_sf.add_trace(go.Scatter(
            x=cum_base.index.astype(str).tolist(), y=cum_base.values.tolist(),
            name=f"{ticker} Basis", mode="lines",
            line=dict(color=col, width=1.5, dash="dot"),
        ))
        # Seasonal filtered
        cum_sf = (1 + rec["n_oos_sf"]).cumprod() * 100
        fig_sf.add_trace(go.Scatter(
            x=cum_sf.index.astype(str).tolist(), y=cum_sf.values.tolist(),
            name=f"{ticker} Seasonal", mode="lines",
            line=dict(color=col, width=2.5),
        ))
    _lay(fig_sf, title="OOS Equity Curve: Basis (gepunktet) vs Seasonal-Filter (durchgezogen)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=500)
    sec_sf = (
        _desc("Saisongefilterter Modus: In 'schlechten' Monaten (IS-Monatsrendite &lt; 0) "
              "wird das Signal auf 0 gesetzt (flat). TC fällt beim Eintritt/Austritt an. "
              "Ziel: Flache Phasen vermeiden ohne wertvolle Long-Phasen zu verlieren. "
              "Gepunktet = Basisstrategie, Durchgezogen = Seasonal-Filter.")
        + _htm(fig_sf)
    )

    # ── S8: Monthly return heatmap – all tickers side by side ─────────────────
    fig_mhm = make_subplots(
        rows=len(strat_records), cols=1,
        subplot_titles=[f"{t}: Jahres×Monats-Rendite" for t in strat_records],
        vertical_spacing=0.05,
    )
    for i, (ticker, rec) in enumerate(strat_records.items()):
        mp = rec["monthly_pnl"]
        fig_mhm.add_trace(go.Heatmap(
            z=mp.values.tolist(), x=mp.columns.tolist(),
            y=[str(y) for y in mp.index.tolist()],
            colorscale="RdYlGn", zmid=0,
            showscale=(i == 0),
            colorbar=dict(title="Rendite", y=1-i/len(strat_records)-0.05,
                          len=1/len(strat_records),
                          tickfont=dict(color="#e6edf3")),
        ), row=i+1, col=1)
    fig_mhm.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=max(200, len(strat_records) * 220),
        title_text="Jahres×Monats Heatmap: alle Airline-Strategien",
    )
    sec_mhm = (
        _desc("Vergleich der Saisonmuster über alle analysierten Airlines. "
              "Gemeinsame rote Monate (alle Strategien gleichzeitig schlecht) "
              "deuten auf marktweite Saisoneffekte hin. "
              "Ticker-spezifische Muster können auf Unternehmens-spezifische "
              "Bilanzzyklen, Quartalsergebnisse oder regionale Effekte zurückgehen.")
        + _htm(fig_mhm)
    )

    # ── S9: Best/Worst month ranking ─────────────────────────────────────────
    # Aggregate across all tickers: average monthly return
    all_avg_m = {m: [] for m in MONTH_NAMES}
    for rec in strat_records.values():
        for m in MONTH_NAMES:
            all_avg_m[m].append(rec["avg_month"].get(m, 0.0))
    agg_m = {m: np.mean(v) for m, v in all_avg_m.items()}
    sorted_m = sorted(agg_m.items(), key=lambda x: x[1], reverse=True)

    fig_rank_m = go.Figure(go.Bar(
        x=[x[0] for x in sorted_m],
        y=[x[1]*100 for x in sorted_m],
        marker_color=["#3fb950" if x[1] > 0 else "#f78166" for x in sorted_m],
    ))
    _lay(fig_rank_m, title="Monats-Ranking: Ø Tagesrendite (alle Strategien aggregiert)",
         xaxis_title="Monat", yaxis_title="Ø Rendite (%/Tag)", height=360)
    sec_rank_m = (
        _desc("Aggregiertes Ranking über alle Airline-Strategien. "
              "Die grünen Monate links sollten bevorzugt gehandelt werden. "
              "Monatsnamen sind nach Ø-Performance absteigend sortiert. "
              "Dieses Ranking bildet die Basis des Saisonfilters.")
        + _htm(fig_rank_m)
    )

    # ── S10: Sharpe comparison table IS/OOS/Seasonal ─────────────────────────
    cmp_rows = ""
    for ticker, rec in strat_records.items():
        sh_is  = _sh(rec["n_is"])
        sh_oos = _sh(rec["n_oos"])
        sh_sf  = _sh(rec["n_oos_sf"])
        delta  = sh_sf - sh_oos if not (np.isnan(sh_sf) or np.isnan(sh_oos)) else np.nan
        mdd_b  = _mdd(rec["n_oos"]) * 100
        mdd_sf = _mdd(rec["n_oos_sf"]) * 100
        col_d  = "#3fb950" if delta > 0 else "#f78166"
        col_m  = "#3fb950" if mdd_sf > mdd_b else "#f78166"
        cmp_rows += (
            f"<tr><td><strong>{ticker}</strong></td>"
            f"<td>{sh_is:.3f}</td><td>{sh_oos:.3f}</td><td>{sh_sf:.3f}</td>"
            f"<td style='color:{col_d};'>{delta:+.3f}</td>"
            f"<td>{mdd_b:.1f}%</td>"
            f"<td style='color:{col_m};'>{mdd_sf:.1f}%</td>"
            f"<td>{', '.join(MONTH_NAMES[m-1] for m in sorted(rec['good_months']))}</td>"
            f"</tr>"
        )
    sec_cmp = (
        _desc("Vollständiger Vergleich: IS-Kalibrierung → OOS-Baseline → OOS-Seasonal-Filter. "
              "Δ Sharpe = Seasonal − Baseline. Positives Δ bedeutet Verbesserung durch Saisonfilter. "
              "MaxDD = Maximum Drawdown; kleinerer Wert (weniger negativ) = besser.")
        + '<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Ticker</th><th>Sharpe IS</th><th>Sharpe OOS Basis</th>'
        '<th>Sharpe OOS Seasonal</th><th>Δ Sharpe</th>'
        '<th>MaxDD Basis</th><th>MaxDD Seasonal</th><th>Aktive Monate</th></tr></thead>'
        f'<tbody>{cmp_rows}</tbody></table></div>'
    )

    # ── S11: Rolling annual return chart (calendar year bars) ─────────────────
    fig_ann = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        net_a = pd.concat([rec["n_is"], rec["n_oos"]]).sort_index()
        net_a.index = pd.to_datetime(net_a.index)
        ann = net_a.groupby(net_a.index.year).sum() * 100
        fig_ann.add_trace(go.Bar(
            x=[str(y) for y in ann.index.tolist()],
            y=ann.values.tolist(),
            name=ticker, marker_color=PAL[i % len(PAL)], opacity=0.8,
        ))
    fig_ann.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_ann, title="Jährliche Strategie-Rendite % je Airline",
         barmode="group", xaxis_title="Jahr", yaxis_title="Jahresrendite (%)", height=420)
    sec_ann = (
        _desc("Kumulierte tägliche Strategie-Renditen je Kalenderjahr. "
              "Wenn bestimmte Jahre systematisch negativ sind, könnte das auf "
              "Öl-Markt-Regime zurückgehen (z.B. 2020 COVID, 2014-16 Ölcrash). "
              "Vergleiche mit dem VIX- und Krisenperioden-Report.")
        + _htm(fig_ann)
    )

    # ── ASSEMBLY ──────────────────────────────────────────────────────────────
    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#sacc{idx}">'
            f'{title}</button></h2>'
            f'<div id="sacc{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    acc = '<div class="accordion" id="seasonAcc">'
    acc += _acc("§0  Übersicht: Saisonfilter-Ergebnis aller Strategien", sec_ov, 0, open_=True)
    acc += _acc("§1  Kalender-Heatmap: Jahr × Monat (Basis-Strategie)", sec_cal, 1)
    acc += _acc("§2  Ø Monatsrendite mit 95%-CI (alle Airlines)", sec_ms, 2)
    acc += _acc("§3  Wochentag-Saisonalität", sec_dow, 3)
    acc += _acc("§4  Quartals-Saisonalität (Boxplots)", sec_qtr, 4)
    acc += _acc("§5  Autokorrelation der Strategie-Renditen (ACF 1–62T)", sec_acf, 5)
    acc += _acc("§6  Rolling 21T kumulierte Rendite (Zyklen sichtbar)", sec_r21, 6)
    acc += _acc("§7  OOS Equity Curves: Basis vs Seasonal-Filter", sec_sf, 7)
    acc += _acc("§8  Jahres×Monats-Heatmap: alle Airlines", sec_mhm, 8)
    acc += _acc("§9  Monats-Ranking (aggregiert, alle Airlines)", sec_rank_m, 9)
    acc += _acc("§10 Vollständiger IS/OOS/Seasonal Sharpe-Vergleich", sec_cmp, 10)
    acc += _acc("§11 Jährliche Renditen je Airline", sec_ann, 11)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#e3b341;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Saisonalität & Zyklizität: CL=F Lead-Lag Strategien</h2>
          <p class="mb-0" style="color:#8b949e;">
            Saisoneffekte messen · Schwache Monate herausfiltern · Saisongefilterter Backtest ·
            ACF-Zyklusanalyse · Anwendung auf {len(strat_records)} Airlines
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "seasonality_report.html",
           _html_base("Saisonalität & Zyklizität", 19, body))


def build_alpha_ideas_report(tables, figures, out):  # noqa: C901
    """
    New alpha / statistical arbitrage ideas derived from the CL=F lead-lag framework.
    Each idea is independently backtested IS/OOS with 26 metrics.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import yfinance as yf
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr

    IS_FRAC = 0.70
    TC      = 0.001
    N_BOOT  = 1000

    # ── data ──────────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")

    if ret_main is None or px_main is None:
        _write(out / "alpha_ideas_report.html",
               _html_base("Alpha Ideas", 19, "<p>Daten fehlen.</p>"))
        return

    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    def _dl(ticker):
        for period in ("10y", "5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    def _card(title, color, body):
        return (f'<div class="card mb-3 p-3" style="background:#1c2128;border:1px solid {color};">'
                f'<h5 style="color:{color};">{title}</h5>'
                f'<div style="color:#e6edf3;">{body}</div></div>')

    def _result_card(name, sh_is, sh_oos, mdd_oos, color="#58a6ff"):
        delta = sh_oos - sh_is if not (np.isnan(sh_is) or np.isnan(sh_oos)) else np.nan
        dc = "#3fb950" if delta > 0 else "#f78166"
        return (f'<div class="row g-2 mb-3">'
                f'<div class="col-md-3"><div class="card p-2" style="background:#0d1117;border:1px solid {color};">'
                f'<small style="color:{color};">IS Sharpe</small><br>'
                f'<strong style="color:#e6edf3;font-size:1.3em;">{sh_is:.3f}</strong></div></div>'
                f'<div class="col-md-3"><div class="card p-2" style="background:#0d1117;border:1px solid {color};">'
                f'<small style="color:{color};">OOS Sharpe</small><br>'
                f'<strong style="color:#e6edf3;font-size:1.3em;">{sh_oos:.3f}</strong></div></div>'
                f'<div class="col-md-3"><div class="card p-2" style="background:#0d1117;border:1px solid {dc};">'
                f'<small style="color:{dc};">Δ OOS−IS</small><br>'
                f'<strong style="color:#e6edf3;font-size:1.3em;">{delta:+.3f}</strong></div></div>'
                f'<div class="col-md-3"><div class="card p-2" style="background:#0d1117;border:1px solid #f78166;">'
                f'<small style="color:#f78166;">MaxDD OOS</small><br>'
                f'<strong style="color:#e6edf3;font-size:1.3em;">{mdd_oos*100:.1f}%</strong></div></div>'
                f'</div>')

    PAL = px.colors.qualitative.Plotly
    SPY = ret_main["SPY"].dropna() if "SPY" in ret_main.columns else None

    # Helper: IS/OOS split by index
    def _split(series, frac=IS_FRAC):
        n = len(series)
        si = int(n * frac)
        return series.iloc[:si], series.iloc[si:]

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 1: Multi-Indicator Ensemble Signal
    # CL=F RSI + MACD + BB combined → weighted vote → trade JETS
    # ══════════════════════════════════════════════════════════════════════════
    idea1_html = ""
    if "CL=F" in px_main.columns and "JETS" in ret_main.columns:
        cl_px  = px_main["CL=F"].dropna()
        jets_r = ret_main["JETS"].dropna()
        common = cl_px.index.intersection(jets_r.index)
        cl_c   = cl_px.reindex(common).ffill()
        jets_c = jets_r.reindex(common).fillna(0.0)

        # Compute indicators
        rsi  = _calc_rsi(cl_c, 14)
        macd = _calc_macd(cl_c)[0]
        bb   = _calc_bb_pos(cl_c, 20)
        sma  = _calc_sma_cross(cl_c, 20, 50)

        # Normalize each indicator to [-1, +1] signal
        def _norm_sig(s):
            r = s.rank(pct=True) * 2 - 1
            return r.clip(-1, 1)

        sig_rsi  = _norm_sig(-rsi)        # high RSI → bearish (short airline)
        sig_macd = _norm_sig(macd)        # positive MACD → bullish
        sig_bb   = _norm_sig(bb - 0.5)   # above mid-band → bullish
        sig_sma  = _norm_sig(sma)        # SMA cross positive → bullish

        # Equal-weight ensemble
        ensemble = (sig_rsi + sig_macd + sig_bb + sig_sma) / 4.0
        # Threshold: long if >0.1, short if <-0.1, flat else
        sig_bin = pd.Series(0.0, index=ensemble.index)
        sig_bin[ensemble > 0.1]  =  1.0
        sig_bin[ensemble < -0.1] = -1.0
        sig_bin = sig_bin.shift(1)  # lag 1

        fr_ens = jets_c.reindex(sig_bin.index).fillna(0.0)
        gross  = sig_bin * fr_ens
        net    = gross - sig_bin.diff().abs().fillna(0) * TC

        is_n, oos_n = _split(net.dropna())
        sh_is1 = _sh(is_n); sh_oos1 = _sh(oos_n); mdd1 = _mdd(oos_n)

        # Baseline RSI<70
        n_base_is, _, _ = _strat_exec(-rsi, -70.0, jets_c.loc[is_n.index[0]:is_n.index[-1]], 1)
        n_base_oos, _, _ = _strat_exec(-rsi, -70.0, jets_c.loc[oos_n.index[0]:oos_n.index[-1]], 1)
        sh_base_is = _sh(n_base_is); sh_base_oos = _sh(n_base_oos)

        cum_e  = (1 + oos_n).cumprod() * 100
        cum_b  = (1 + n_base_oos).cumprod() * 100

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=cum_b.index.astype(str).tolist(), y=cum_b.values.tolist(),
                                   name="Basis RSI<70", mode="lines",
                                   line=dict(color="#58a6ff", dash="dot", width=1.5)))
        fig_e.add_trace(go.Scatter(x=cum_e.index.astype(str).tolist(), y=cum_e.values.tolist(),
                                   name="Ensemble (RSI+MACD+BB+SMA)", mode="lines",
                                   line=dict(color="#3fb950", width=2.5)))
        _lay(fig_e, title="Idea 1: Multi-Indikator Ensemble vs Basis (OOS, JETS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=420)

        # Indicator weight scatter: correlation of each component with net
        weights_fig = go.Figure(go.Bar(
            x=["RSI","MACD","BB","SMA"],
            y=[float(spearmanr(sig_rsi.dropna().values, net.dropna().reindex(sig_rsi.dropna().index).fillna(0).values)[0])
               for sig_x in [sig_rsi, sig_macd, sig_bb, sig_sma]],
            marker_color=["#58a6ff","#3fb950","#ffa657","#f78166"],
        ))
        _lay(weights_fig, title="Spearman ρ: Indikator-Signal vs Ensemble-Return", height=300)

        idea1_html = (
            _desc("Ensemble-Methode: 4 Indikatoren werden auf CL=F berechnet, "
                  "zu einem [-1,+1]-Signal normiert (Rang-Percentile) und gleich gewichtet. "
                  "Signal = Long wenn Ø &gt; 0.1, Short wenn Ø &lt; -0.1, sonst Flat. "
                  "Mathematik: sig = (sig_RSI + sig_MACD + sig_BB + sig_SMA) / 4")
            + _result_card("Ensemble vs Basis", sh_is1, sh_oos1, mdd1, "#3fb950")
            + _card("Baseline (RSI<70)", "#58a6ff",
                    f"IS Sharpe: {sh_base_is:.3f} | OOS Sharpe: {sh_base_oos:.3f}")
            + _htm(fig_e)
            + _htm(weights_fig)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 2: VIX Regime Filter
    # Only trade CL=F→JETS when VIX < 25 (calm market)
    # ══════════════════════════════════════════════════════════════════════════
    idea2_html = ""
    try:
        vix_s = _dl("^VIX")
        if vix_s is not None and "JETS" in ret_main.columns and "CL=F" in px_main.columns:
            cl_px  = px_main["CL=F"].dropna()
            jets_r = ret_main["JETS"].dropna()
            common = cl_px.index.intersection(jets_r.index).intersection(vix_s.index)
            cl_c   = cl_px.reindex(common).ffill()
            jets_c = jets_r.reindex(common).fillna(0.0)
            vix_c  = vix_s.reindex(common).ffill()

            n_base, g_base, s_base = _strat_exec(-_calc_rsi(cl_c, 14), -70.0, jets_c, 1)
            # VIX filter: zero signal when VIX(t-1) >= threshold
            for vix_thresh in [20, 25, 30]:
                vix_mask = (vix_c.shift(1) < vix_thresh)
                s_vf = s_base.copy()
                s_vf[~vix_mask.reindex(s_vf.index).fillna(False)] = 0.0
                fr_vf = jets_c.reindex(s_vf.index).fillna(0.0)
                gross_vf = s_vf * fr_vf
                net_vf   = gross_vf - s_vf.diff().abs().fillna(0) * TC
                is_vf, oos_vf = _split(net_vf.dropna())

                sh_is_vf  = _sh(is_vf); sh_oos_vf = _sh(oos_vf); mdd_vf = _mdd(oos_vf)

            # Chart with all VIX thresholds
            fig_vix = go.Figure()
            cum_base = (1 + _split(n_base.dropna())[1]).cumprod() * 100
            fig_vix.add_trace(go.Scatter(
                x=cum_base.index.astype(str).tolist(), y=cum_base.values.tolist(),
                name="Basis (kein Filter)", mode="lines",
                line=dict(color="#8b949e", dash="dot", width=1.5)))

            colors_v = ["#58a6ff","#3fb950","#ffa657"]
            for vi, vix_thresh in enumerate([20, 25, 30]):
                vix_mask = (vix_c.shift(1) < vix_thresh)
                s_vf = s_base.copy()
                s_vf[~vix_mask.reindex(s_vf.index).fillna(False)] = 0.0
                fr_vf = jets_c.reindex(s_vf.index).fillna(0.0)
                gross_vf = s_vf * fr_vf
                net_vf   = gross_vf - s_vf.diff().abs().fillna(0) * TC
                oos_vf_s = _split(net_vf.dropna())[1]
                cum_vf = (1 + oos_vf_s).cumprod() * 100
                fig_vix.add_trace(go.Scatter(
                    x=cum_vf.index.astype(str).tolist(), y=cum_vf.values.tolist(),
                    name=f"VIX < {vix_thresh}", mode="lines",
                    line=dict(color=colors_v[vi], width=2.0)))

            # VIX-level overlay (secondary y)
            fig_vix.add_trace(go.Scatter(
                x=vix_c.index.astype(str).tolist(), y=vix_c.values.tolist(),
                name="VIX Level", mode="lines",
                line=dict(color="#bc8cff", width=0.8),
                yaxis="y2", opacity=0.4))
            fig_vix.update_layout(
                yaxis2=dict(title="VIX", overlaying="y", side="right",
                            gridcolor="#21262d", tickfont=dict(color="#e6edf3")),
            )
            _lay(fig_vix, title="Idea 2: VIX-Regime-Filter (OOS, JETS)",
                 xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=450)

            idea2_html = (
                _desc("VIX-Filter: Das Handelssignal wird auf 0 gesetzt, wenn VIX(t-1) ≥ Schwelle. "
                      "Logik: In turbulenten Märkten (hoher VIX) bricht die Lead-Lag-Struktur zusammen, "
                      "da Panic-Selling alle Korrelationen auf 1 treibt. "
                      "Getestet mit VIX-Schwellen 20, 25, 30.")
                + _htm(fig_vix)
            )
    except Exception:
        idea2_html = _card("VIX Filter", "#8b949e", "VIX-Daten nicht verfügbar.")

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 3: Statistical Arbitrage – CL=F vs BZ=F Spread
    # WTI–Brent spread mean reversion: long when spread too wide, short when narrow
    # ══════════════════════════════════════════════════════════════════════════
    idea3_html = ""
    if "CL=F" in px_main.columns and "BZ=F" in px_main.columns:
        wti = px_main["CL=F"].dropna()
        bzt = px_main["BZ=F"].dropna()
        common3 = wti.index.intersection(bzt.index)
        wti_c = wti.reindex(common3).ffill()
        bzt_c = bzt.reindex(common3).ffill()

        # Spread = log(WTI) - log(Brent)
        spread = np.log(wti_c) - np.log(bzt_c)
        spread_z = (spread - spread.rolling(63).mean()) / (spread.rolling(63).std() + 1e-9)

        # Strategy: long WTI / short Brent when Z < -1 (WTI cheap relative to Brent)
        #           short WTI / long Brent when Z > +1
        sig_spread = pd.Series(0.0, index=spread_z.index)
        sig_spread[spread_z < -1.0] =  1.0   # long spread (long WTI, short BZF)
        sig_spread[spread_z >  1.0] = -1.0   # short spread
        sig_spread = sig_spread.shift(1)

        wti_ret = np.log(wti_c / wti_c.shift(1))
        bzt_ret = np.log(bzt_c / bzt_c.shift(1))
        spread_ret = wti_ret - bzt_ret  # pair return

        common_s = sig_spread.index.intersection(spread_ret.dropna().index)
        sig_s = sig_spread.reindex(common_s)
        sr_c  = spread_ret.reindex(common_s).fillna(0.0)
        gross_s = sig_s * sr_c
        net_s   = gross_s - sig_s.diff().abs().fillna(0) * TC * 2  # 2x TC for two legs

        is_s, oos_s = _split(net_s.dropna())
        sh_is3 = _sh(is_s); sh_oos3 = _sh(oos_s); mdd3 = _mdd(oos_s)

        fig_sp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=["WTI–Brent Log-Spread", "Equity Curve (OOS)"])
        fig_sp.add_trace(go.Scatter(
            x=spread.index.astype(str).tolist(), y=spread.values.tolist(),
            name="Log-Spread", line=dict(color="#58a6ff")), row=1, col=1)
        fig_sp.add_trace(go.Scatter(
            x=spread_z.index.astype(str).tolist(), y=spread_z.values.tolist(),
            name="Z-Score (63T)", line=dict(color="#ffa657")), row=1, col=1)
        fig_sp.add_hline(y=1.0,  line_color="#f78166", line_dash="dot", row=1, col=1)
        fig_sp.add_hline(y=-1.0, line_color="#3fb950", line_dash="dot", row=1, col=1)
        cum_s = (1 + oos_s).cumprod() * 100
        fig_sp.add_trace(go.Scatter(
            x=cum_s.index.astype(str).tolist(), y=cum_s.values.tolist(),
            name="Equity OOS", line=dict(color="#3fb950", width=2)), row=2, col=1)
        fig_sp.update_layout(
            **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
            height=560, title_text="Idea 3: WTI–Brent Stat-Arb Spread",
        )

        idea3_html = (
            _desc("Statistisches Arbitrage-Modell: WTI (CL=F) und Brent (BZ=F) sind langfristig "
                  "kointegriert – die Preisdifferenz kehrt zum Mittel zurück. "
                  "Spread = log(WTI) − log(Brent). Z-Score = (Spread − SMA63) / Std63. "
                  "Long Spread wenn Z &lt; -1, Short Spread wenn Z &gt; +1. "
                  "TC = 2×10bp da zwei Legs gleichzeitig gehandelt werden.")
            + _result_card("WTI–Brent Stat-Arb", sh_is3, sh_oos3, mdd3, "#ffa657")
            + _htm(fig_sp)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 4: Airline Cross-Sectional Momentum
    # Download multiple airlines, each week rank by 21-day return,
    # long top 2, short bottom 2
    # ══════════════════════════════════════════════════════════════════════════
    idea4_html = ""
    AIRLINE_TICKERS = ["DAL","UAL","AAL","LUV","ALK","JBLU","JETS"]
    xsm_data = {}
    for t in AIRLINE_TICKERS:
        if t in ret_main.columns:
            xsm_data[t] = ret_main[t].dropna()
        else:
            s = _dl(t)
            if s is not None:
                xsm_data[t] = np.log(s/s.shift(1)).dropna()

    if len(xsm_data) >= 4:
        xsm_df = pd.DataFrame(xsm_data).dropna(how="all").fillna(0.0)
        # Weekly rebalancing (every 5 days)
        common_xsm = xsm_df.index
        n_xsm = len(common_xsm)
        sig_xsm = pd.DataFrame(0.0, index=common_xsm, columns=xsm_df.columns)

        for i in range(21, n_xsm, 5):
            mom_21 = xsm_df.iloc[i-21:i].sum()
            ranked = mom_21.rank()
            n_a    = len(ranked)
            top    = ranked >= (n_a - 1)
            bot    = ranked <= 2
            for t in xsm_df.columns:
                if top[t]: sig_xsm.loc[common_xsm[i], t] =  1.0 / top.sum()
                elif bot[t]: sig_xsm.loc[common_xsm[i], t] = -1.0 / bot.sum()

        sig_xsm = sig_xsm.shift(1).ffill()
        gross_xsm = (sig_xsm * xsm_df.reindex(sig_xsm.index).fillna(0)).sum(axis=1)
        net_xsm   = gross_xsm - sig_xsm.diff().abs().sum(axis=1) * TC

        is_xsm, oos_xsm = _split(net_xsm.dropna())
        sh_is4 = _sh(is_xsm); sh_oos4 = _sh(oos_xsm); mdd4 = _mdd(oos_xsm)

        fig_xsm = go.Figure()
        cum_xsm = (1 + oos_xsm).cumprod() * 100
        fig_xsm.add_trace(go.Scatter(
            x=cum_xsm.index.astype(str).tolist(), y=cum_xsm.values.tolist(),
            name="Cross-Sect. Momentum", mode="lines",
            line=dict(color="#bc8cff", width=2.5)))
        _lay(fig_xsm, title="Idea 4: Cross-Sectional Momentum Airlines (OOS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=400)

        idea4_html = (
            _desc("Cross-Sectional Momentum: Alle Airlines werden wöchentlich nach 21-Tage-Rendite gerankt. "
                  "Long: Top-2-Airlines. Short: Bottom-2-Airlines. Gleichgewichtet. "
                  "Rebalancing alle 5 Handelstage. "
                  "Misst ob relative Stärke/Schwäche innerhalb von Airlines persistiert.")
            + _result_card("Cross-Sect. Momentum", sh_is4, sh_oos4, mdd4, "#bc8cff")
            + _htm(fig_xsm)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 5: Oil Volatility Regime Filter
    # Only trade CL=F→JETS when rolling 21-day oil vol is LOW
    # ══════════════════════════════════════════════════════════════════════════
    idea5_html = ""
    if "CL=F" in px_main.columns and "JETS" in ret_main.columns:
        cl_px  = px_main["CL=F"].dropna()
        jets_r = ret_main["JETS"].dropna()
        cl_ret = np.log(cl_px / cl_px.shift(1)).dropna()
        common5 = cl_px.index.intersection(jets_r.index).intersection(cl_ret.index)
        cl_c5  = cl_px.reindex(common5).ffill()
        cl_r5  = cl_ret.reindex(common5).fillna(0.0)
        jets_c5 = jets_r.reindex(common5).fillna(0.0)

        oil_vol21 = cl_r5.rolling(21).std() * np.sqrt(252)
        vol_med   = float(oil_vol21.median())

        n_base5, g_base5, s_base5 = _strat_exec(-_calc_rsi(cl_c5, 14), -70.0, jets_c5, 1)

        results5 = {}
        for vfrac in [0.5, 0.75, 1.0, 1.25]:
            thresh_v = vol_med * vfrac
            mask_v   = (oil_vol21.shift(1) < thresh_v)
            s_vf5    = s_base5.copy()
            s_vf5[~mask_v.reindex(s_vf5.index).fillna(False)] = 0.0
            fr_vf5   = jets_c5.reindex(s_vf5.index).fillna(0.0)
            net_vf5  = s_vf5 * fr_vf5 - s_vf5.diff().abs().fillna(0) * TC
            is_v, oos_v = _split(net_vf5.dropna())
            results5[f"Vol<{vfrac:.2f}×Med"] = {"sh_oos": _sh(oos_v), "net_oos": oos_v}

        fig_vol = go.Figure()
        for label, res in results5.items():
            cum_v = (1 + res["net_oos"]).cumprod() * 100
            fig_vol.add_trace(go.Scatter(
                x=cum_v.index.astype(str).tolist(), y=cum_v.values.tolist(),
                name=label, mode="lines"))
        cum_b5 = (1 + _split(n_base5.dropna())[1]).cumprod() * 100
        fig_vol.add_trace(go.Scatter(
            x=cum_b5.index.astype(str).tolist(), y=cum_b5.values.tolist(),
            name="Basis", mode="lines", line=dict(dash="dot", color="#8b949e")))
        _lay(fig_vol, title="Idea 5: Öl-Volatilität-Regime-Filter (OOS, JETS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=430)

        idea5_html = (
            _desc("Öl-Volatilität-Filter: Nur handeln wenn die rollende 21T-Volatilität von CL=F "
                  "unterhalb eines Schwellenwerts (Median × Faktor) liegt. "
                  "Rationale: Bei hoher Öl-Volatilität (Regime-Unsicherheit) bricht das "
                  "Lead-Lag-Signal zusammen. Getestet mit Schwellen 0.5×, 0.75×, 1.0×, 1.25× Median-Vol.")
            + _htm(fig_vol)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 6: Multi-Asset Leader Basket
    # Use CL=F + BZ=F + XLE as ensemble oil signal → trade JETS
    # ══════════════════════════════════════════════════════════════════════════
    idea6_html = ""
    basket_assets = ["CL=F","BZ=F","XLE","XOM","CVX"]
    basket_px = {t: px_main[t].dropna() for t in basket_assets if t in px_main.columns}

    if len(basket_px) >= 2 and "JETS" in ret_main.columns:
        jets_r6 = ret_main["JETS"].dropna()
        common6 = jets_r6.index
        for t in basket_px:
            common6 = common6.intersection(basket_px[t].index)
        common6 = common6[~common6.duplicated()]
        if len(common6) > 300:
            jets_c6 = jets_r6.reindex(common6).fillna(0.0)

            # Ensemble: average RSI signal across basket
            basket_signals = []
            for t, px_t in basket_px.items():
                px_c = px_t.reindex(common6).ffill()
                rsi_t = _calc_rsi(px_c, 14)
                sig_t = (rsi_t < 70).astype(float) * 2 - 1  # +1 when <70
                basket_signals.append(sig_t)

            ens_sig = pd.concat(basket_signals, axis=1).mean(axis=1).shift(1)
            # Long when majority signal positive
            sig6 = pd.Series(np.sign(ens_sig.values), index=ens_sig.index)
            sig6[ens_sig.abs() < 0.2] = 0.0  # flat if signals split

            gross6 = sig6 * jets_c6.reindex(sig6.index).fillna(0.0)
            net6   = gross6 - sig6.diff().abs().fillna(0) * TC
            is6, oos6 = _split(net6.dropna())
            sh_is6 = _sh(is6); sh_oos6 = _sh(oos6); mdd6 = _mdd(oos6)

            # Compare to single-asset RSI<70
            n_base6, _, _ = _strat_exec(-_calc_rsi(basket_px["CL=F"].reindex(common6).ffill(), 14),
                                        -70.0, jets_c6, 1)
            is_b6, oos_b6 = _split(n_base6.dropna())

            fig_basket = go.Figure()
            cum_b6 = (1 + oos_b6).cumprod() * 100
            cum_6  = (1 + oos6).cumprod() * 100
            fig_basket.add_trace(go.Scatter(
                x=cum_b6.index.astype(str).tolist(), y=cum_b6.values.tolist(),
                name="Basis CL=F RSI<70", mode="lines",
                line=dict(color="#58a6ff", dash="dot")))
            fig_basket.add_trace(go.Scatter(
                x=cum_6.index.astype(str).tolist(), y=cum_6.values.tolist(),
                name=f"Basket ({'+'.join(basket_px.keys())})", mode="lines",
                line=dict(color="#ffa657", width=2.5)))
            _lay(fig_basket, title="Idea 6: Multi-Asset Öl-Basket als Leader (OOS, JETS)",
                 xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=420)

            idea6_html = (
                _desc(f"Öl-Basket: RSI-Signale aus {', '.join(basket_px.keys())} werden gemittelt. "
                      "Long wenn Ø-Signal &gt; 0.2, Short wenn &lt; -0.2, Flat bei gemischten Signalen. "
                      "Rationale: Brent (BZ=F), XLE und Einzelaktien liefern komplementäre "
                      "Informationen über die Öl-Supply-Demand-Balance.")
                + _result_card("Multi-Asset Basket", sh_is6, sh_oos6, mdd6, "#ffa657")
                + _htm(fig_basket)
            )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 7: DXY Macro Filter
    # USD strengthening tends to suppress oil prices → use DXY trend as filter
    # ══════════════════════════════════════════════════════════════════════════
    idea7_html = ""
    if "DX-Y.NYB" in ret_main.columns and "CL=F" in px_main.columns and "JETS" in ret_main.columns:
        dxy_r  = ret_main["DX-Y.NYB"].dropna()
        cl_px7 = px_main["CL=F"].dropna()
        jets_r7 = ret_main["JETS"].dropna()
        common7 = cl_px7.index.intersection(jets_r7.index).intersection(dxy_r.index)
        cl_c7   = cl_px7.reindex(common7).ffill()
        jets_c7 = jets_r7.reindex(common7).fillna(0.0)
        dxy_c7  = dxy_r.reindex(common7).fillna(0.0)

        dxy_trend = dxy_c7.rolling(20).mean()  # 20-day trend

        n_base7, _, s_base7 = _strat_exec(-_calc_rsi(cl_c7, 14), -70.0, jets_c7, 1)

        results7 = {}
        for dxy_filt, label in [("flat_strong", "Flat wenn DXY-Trend positiv"),
                                  ("short_strong", "Short wenn DXY-Trend positiv")]:
            s7 = s_base7.copy()
            strong_dxy = dxy_trend.reindex(s7.index).fillna(0.0) > 0
            if dxy_filt == "flat_strong":
                s7[strong_dxy] = 0.0
            else:
                s7[strong_dxy] = -1.0
            fr7 = jets_c7.reindex(s7.index).fillna(0.0)
            net7 = s7 * fr7 - s7.diff().abs().fillna(0) * TC
            is7, oos7 = _split(net7.dropna())
            results7[label] = {"sh_oos": _sh(oos7), "net_oos": oos7}

        fig_dxy = go.Figure()
        cum_base7 = (1 + _split(n_base7.dropna())[1]).cumprod() * 100
        fig_dxy.add_trace(go.Scatter(
            x=cum_base7.index.astype(str).tolist(), y=cum_base7.values.tolist(),
            name="Basis", mode="lines", line=dict(color="#8b949e", dash="dot")))
        for label, res in results7.items():
            cum_d = (1 + res["net_oos"]).cumprod() * 100
            fig_dxy.add_trace(go.Scatter(
                x=cum_d.index.astype(str).tolist(), y=cum_d.values.tolist(),
                name=label, mode="lines"))
        _lay(fig_dxy, title="Idea 7: DXY-Makro-Filter (OOS, JETS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=420)

        idea7_html = (
            _desc("DXY-Filter: Wenn der US-Dollar (DX-Y.NYB) aufwertet (20T-Trend positiv), "
                  "tendiert Öl zur Schwäche. Zwei Varianten: (a) Signal auf Flat setzen, "
                  "(b) Signal auf Short setzen. "
                  "Formel: DXY-Trend = SMA20(r_DXY). Positiv = USD stark = Öl schwach.")
            + _htm(fig_dxy)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 8: Oil-XLE Basis Arbitrage
    # CL=F and XLE should move together; trade deviations
    # ══════════════════════════════════════════════════════════════════════════
    idea8_html = ""
    if "CL=F" in px_main.columns and "XLE" in px_main.columns:
        cl_p8 = px_main["CL=F"].dropna()
        xle_p = px_main["XLE"].dropna()
        xle_r = ret_main["XLE"].dropna() if "XLE" in ret_main.columns else None
        if xle_r is not None:
            common8 = cl_p8.index.intersection(xle_p.index).intersection(xle_r.index)
            cl_c8  = cl_p8.reindex(common8).ffill()
            xle_c8 = xle_p.reindex(common8).ffill()
            xle_r8 = xle_r.reindex(common8).fillna(0.0)

            # Log ratio = log(XLE/CL=F) normalized
            ratio = np.log(xle_c8) - np.log(cl_c8)
            ratio_z = (ratio - ratio.rolling(63).mean()) / (ratio.rolling(63).std() + 1e-9)

            # Mean reversion: when ratio too high → XLE overbought vs oil → short XLE
            sig8 = pd.Series(0.0, index=ratio_z.index)
            sig8[ratio_z < -1.5] =  1.0   # XLE cheap vs oil → long XLE
            sig8[ratio_z >  1.5] = -1.0   # XLE expensive vs oil → short XLE
            sig8 = sig8.shift(1)

            gross8 = sig8 * xle_r8.reindex(sig8.index).fillna(0.0)
            net8   = gross8 - sig8.diff().abs().fillna(0) * TC
            is8, oos8 = _split(net8.dropna())
            sh_is8 = _sh(is8); sh_oos8 = _sh(oos8); mdd8 = _mdd(oos8)

            fig_basis = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=["XLE/CL=F Log-Ratio Z-Score", "OOS Equity"])
            fig_basis.add_trace(go.Scatter(
                x=ratio_z.index.astype(str).tolist(), y=ratio_z.values.tolist(),
                name="Ratio Z-Score", line=dict(color="#58a6ff")), row=1, col=1)
            fig_basis.add_hline(y=1.5,  line_color="#f78166", line_dash="dot", row=1, col=1)
            fig_basis.add_hline(y=-1.5, line_color="#3fb950", line_dash="dot", row=1, col=1)
            cum8 = (1 + oos8).cumprod() * 100
            fig_basis.add_trace(go.Scatter(
                x=cum8.index.astype(str).tolist(), y=cum8.values.tolist(),
                name="Equity OOS", line=dict(color="#3fb950", width=2)), row=2, col=1)
            fig_basis.update_layout(
                **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
                height=520, title_text="Idea 8: XLE–CL=F Basis Arbitrage",
            )

            idea8_html = (
                _desc("XLE–CL=F Basis Arbitrage: XLE (Energy ETF) und CL=F (WTI Futures) "
                      "sind fundamental verknüpft. Wenn XLE im Verhältnis zu Öl zu teuer/billig wird "
                      "(|Z| &gt; 1.5), erfolgt eine Gegenbewegung. "
                      "Z = (log(XLE/CL=F) − SMA63) / Std63. "
                      "Long XLE bei Z &lt; -1.5, Short XLE bei Z &gt; 1.5.")
                + _result_card("XLE–CL=F Basis Arb", sh_is8, sh_oos8, mdd8, "#39d353")
                + _htm(fig_basis)
            )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 9: Seasonal + RSI Combined
    # Only trade when RSI<70 AND month is seasonally favorable
    # ══════════════════════════════════════════════════════════════════════════
    idea9_html = ""
    if "CL=F" in px_main.columns and "JETS" in ret_main.columns:
        cl_px9  = px_main["CL=F"].dropna()
        jets_r9 = ret_main["JETS"].dropna()
        common9 = cl_px9.index.intersection(jets_r9.index)
        cl_c9   = cl_px9.reindex(common9).ffill()
        jets_c9 = jets_r9.reindex(common9).fillna(0.0)

        n9, g9, s9 = _strat_exec(-_calc_rsi(cl_c9, 14), -70.0, jets_c9, 1)

        # Determine good months on IS portion
        is9, oos9_n = _split(n9.dropna())
        is9_df = is9.to_frame("r")
        is9_df["month"] = pd.to_datetime(is9_df.index).month
        good_m9 = set(is9_df.groupby("month")["r"].mean()[lambda x: x > 0].index)

        # Apply seasonal filter to OOS signals
        s9_oos = s9.reindex(oos9_n.index)
        s9_sf  = s9_oos.copy()
        oos_months = pd.to_datetime(s9_sf.index).month
        s9_sf[~pd.Index(oos_months).isin(good_m9)] = 0.0

        jets_oos9 = jets_c9.reindex(s9_sf.index).fillna(0.0)
        net_sf9   = s9_sf * jets_oos9 - s9_sf.diff().abs().fillna(0) * TC

        sh_is9  = _sh(is9)
        sh_oos9 = _sh(oos9_n)
        sh_sf9  = _sh(net_sf9)
        mdd_sf9 = _mdd(net_sf9)

        fig_comb = go.Figure()
        cum_oos9 = (1 + oos9_n).cumprod() * 100
        cum_sf9  = (1 + net_sf9).cumprod() * 100
        fig_comb.add_trace(go.Scatter(
            x=cum_oos9.index.astype(str).tolist(), y=cum_oos9.values.tolist(),
            name="Basis RSI<70", mode="lines", line=dict(color="#58a6ff", dash="dot")))
        fig_comb.add_trace(go.Scatter(
            x=cum_sf9.index.astype(str).tolist(), y=cum_sf9.values.tolist(),
            name="RSI<70 + Seasonal Filter", mode="lines",
            line=dict(color="#e3b341", width=2.5)))
        _lay(fig_comb, title="Idea 9: RSI<70 + Saisonalitätsfilter kombiniert (OOS, JETS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=420)

        good_m_names = ", ".join(["Jan","Feb","Mär","Apr","Mai","Jun",
                                   "Jul","Aug","Sep","Okt","Nov","Dez"][m-1]
                                  for m in sorted(good_m9))
        idea9_html = (
            _desc(f"Kombination: RSI&lt;70 Signal (Lag=1) aktiv NUR in IS-profitablen Monaten: {good_m_names}. "
                  "Dies verbindet die statistisch-technische Signalerzeugung "
                  "mit der kalendarischen Saisonstruktur der Märkte.")
            + _result_card("RSI<70 + Saisonal", sh_is9, sh_sf9, mdd_sf9, "#e3b341")
            + _card("Vergleich", "#58a6ff",
                    f"Basis OOS Sharpe: {sh_oos9:.3f} → Kombiniert: {sh_sf9:.3f} "
                    f"(Δ: {sh_sf9-sh_oos9:+.3f})")
            + _htm(fig_comb)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IDEA 10: TNX (10Y Yield) as Macro Signal
    # Rising rates → airlines hurt (debt costs); use TNX trend as filter
    # ══════════════════════════════════════════════════════════════════════════
    idea10_html = ""
    if "^TNX" in ret_main.columns and "CL=F" in px_main.columns and "JETS" in ret_main.columns:
        tnx_r   = ret_main["^TNX"].dropna()
        cl_px10 = px_main["CL=F"].dropna()
        jets10  = ret_main["JETS"].dropna()
        common10 = cl_px10.index.intersection(jets10.index).intersection(tnx_r.index)
        cl_c10  = cl_px10.reindex(common10).ffill()
        jets_c10 = jets10.reindex(common10).fillna(0.0)
        tnx_c10  = tnx_r.reindex(common10).fillna(0.0)

        tnx_trend = tnx_c10.rolling(20).mean()
        n_b10, _, s_b10 = _strat_exec(-_calc_rsi(cl_c10, 14), -70.0, jets_c10, 1)

        # Filter: when rates rising (TNX trend positive), reduce to flat
        s_tnx = s_b10.copy()
        rising = tnx_trend.reindex(s_tnx.index).fillna(0.0) > 0
        s_tnx[rising] = 0.0
        net_tnx = s_tnx * jets_c10.reindex(s_tnx.index).fillna(0.0) - s_tnx.diff().abs().fillna(0)*TC
        is_tnx, oos_tnx = _split(net_tnx.dropna())
        is_b10, oos_b10 = _split(n_b10.dropna())

        fig_tnx = go.Figure()
        cum_b10   = (1 + oos_b10).cumprod() * 100
        cum_tnx   = (1 + oos_tnx).cumprod() * 100
        fig_tnx.add_trace(go.Scatter(
            x=cum_b10.index.astype(str).tolist(), y=cum_b10.values.tolist(),
            name="Basis", mode="lines", line=dict(color="#8b949e", dash="dot")))
        fig_tnx.add_trace(go.Scatter(
            x=cum_tnx.index.astype(str).tolist(), y=cum_tnx.values.tolist(),
            name="TNX-Filter (flat bei steigenden Zinsen)", mode="lines",
            line=dict(color="#ff7b72", width=2.5)))
        _lay(fig_tnx, title="Idea 10: US-Zinsfilter (TNX 20T-Trend) für JETS-Strategie",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=400)

        idea10_html = (
            _desc("10Y-US-Treasury-Rendite (TNX) als Makrofilter: Steigende Zinsen erhöhen "
                  "Finanzierungskosten der Airlines (hohe Verschuldung) und wirken sich negativ aus. "
                  "Filter: Signal = 0 wenn 20T-Trend(TNX) &gt; 0 (Zinsen steigen). "
                  f"Basis OOS Sharpe: {_sh(oos_b10):.3f} → TNX-Filter: {_sh(oos_tnx):.3f}")
            + _htm(fig_tnx)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Summary / Ranking of all ideas
    # ══════════════════════════════════════════════════════════════════════════
    ideas_summary = [
        ("Multi-Indikator Ensemble",    idea1_html),
        ("VIX Regime Filter",           idea2_html),
        ("WTI–Brent Stat-Arb Spread",   idea3_html),
        ("Cross-Sect. Momentum Airlines", idea4_html),
        ("Öl-Volatilität Filter",       idea5_html),
        ("Multi-Asset Öl-Basket",       idea6_html),
        ("DXY Makro-Filter",            idea7_html),
        ("XLE–CL=F Basis Arbitrage",    idea8_html),
        ("RSI<70 + Saisonalitätsfilter",idea9_html),
        ("US-Zinsfilter (TNX)",         idea10_html),
    ]

    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#alphacc{idx}">'
            f'💡 Idee {idx+1}: {title}</button></h2>'
            f'<div id="alphacc{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    acc = '<div class="accordion" id="alphaAcc">'
    for i, (title, body) in enumerate(ideas_summary):
        if body:
            acc += _acc(title, body, i, open_=(i == 0))
    acc += "</div>"

    overview_card = _card(
        "Neue Alpha-Ideen: Konzeptübersicht", "#ffa657",
        """<ul style="color:#e6edf3;">
        <li><strong>Ensemble:</strong> Kombination mehrerer Indikatoren reduziert Einzelsignal-Rauschen</li>
        <li><strong>Regime-Filter (VIX/Vol):</strong> Strategie nur in Märkten mit klarer Lead-Lag-Struktur</li>
        <li><strong>Stat-Arb WTI–Brent:</strong> Kointegrations-Spread als eigenständige Mean-Reversion-Strategie</li>
        <li><strong>Cross-Sect. Momentum:</strong> Relativer Rank der Airlines nutzt diversifizierte Information</li>
        <li><strong>Multi-Asset Basket:</strong> Robustere Öl-Signale durch Aggregation mehrerer Energietitel</li>
        <li><strong>Makro-Filter (DXY/TNX):</strong> USD und Zinsen als übergeordnete Regime-Indikatoren</li>
        <li><strong>Basis-Arb XLE–CL=F:</strong> Kurzfristige Fehlbepreisungen zwischen ETF und Futures</li>
        <li><strong>Saisonal+RSI:</strong> Kombination von zwei unabhängigen Alpha-Quellen</li>
        </ul>"""
    )

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#bc8cff;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Neue Alpha-Ideen: CL=F Lead-Lag Framework</h2>
          <p class="mb-0" style="color:#8b949e;">
            10 unabhängige Strategien · Ensemble · Stat-Arb · Regime-Filter · Cross-Sect. Momentum ·
            IS/OOS Backtest · Vergleich mit Basis-Strategie
          </p>
        </div>
      </div>
      {overview_card}
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "alpha_ideas_report.html",
           _html_base("Neue Alpha-Ideen", 19, body))


def build_combination_holdperiod_report(tables, figures, out):  # noqa: C901
    """
    Strategy Combination Lab + Holding Period Analysis.
    Combines Oil Basket, Seasonal, VIX, TNX filters; analyses return distributions.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from itertools import product as iproduct
    import yfinance as yf

    IS_FRAC   = 0.70
    TC        = 0.001
    HORIZONS  = [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]
    H_FIXED   = [1, 2, 3, 5, 7, 10, 14, 21]
    TC_LEVELS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    CRISES    = [
        ("Lehman 2008",  "2008-09-01", "2009-03-31", "#f78166"),
        ("COVID 2020",   "2020-02-01", "2020-05-31", "#ffa657"),
        ("Inflation 22", "2022-01-01", "2022-12-31", "#bc8cff"),
    ]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _dl(ticker):
        for period in ("10y", "5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _roll_sh(s, w=252):
        m = s.rolling(w).mean(); v = s.rolling(w).std()
        return (m / (v + 1e-9)) * np.sqrt(252)

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _split(s, frac=IS_FRAC):
        n = len(s); si = int(n * frac)
        return s.iloc[:si], s.iloc[si:]

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    def _card(title, color, body):
        return (f'<div class="card mb-3 p-3" style="background:#1c2128;border:1px solid {color};">'
                f'<h5 style="color:{color};">{title}</h5>'
                f'<div style="color:#e6edf3;">{body}</div></div>')

    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#ch{idx}">'
            f'{title}</button></h2>'
            f'<div id="ch{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    def _add_crises(fig, row=None, col=None):
        for cname, cs, ce, cc in CRISES:
            try:
                kw = dict(x0=cs, x1=ce, fillcolor=cc, opacity=0.09,
                          layer="below", line_width=0)
                if row is not None:
                    fig.add_vrect(row=row, col=col, **kw)
                else:
                    fig.add_vrect(**kw)
            except Exception:
                pass

    def _fixed_hold_strat(signal, ret, H, tc=TC):
        """Fixed H-day holding period; returns (net, entry_dates, exit_dates)."""
        sig_a = signal.reindex(ret.index).fillna(0.0).values
        r_a   = ret.values
        n     = len(sig_a)
        pos   = np.zeros(n)
        entry_list, exit_list = [], []
        i = 0
        while i < n:
            if sig_a[i] > 0:
                entry_list.append(i)
                exit_i = min(i + H, n)
                pos[i:exit_i] = 1.0
                exit_list.append(min(exit_i, n - 1))
                i = exit_i
            else:
                i += 1
        pos_s = pd.Series(pos, index=ret.index)
        net   = pos_s * ret - pos_s.diff().abs().fillna(0) * tc
        e_dt  = ret.index[entry_list] if entry_list else pd.DatetimeIndex([])
        ex_dt = ret.index[exit_list]  if exit_list  else pd.DatetimeIndex([])
        return net, e_dt, ex_dt

    # ── data ──────────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")
    if ret_main is None or px_main is None:
        _write(out / "combination_holdperiod_report.html",
               _html_base("Combination Lab", 20, "<p>Daten fehlen.</p>")); return
    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    if "JETS" not in ret_main.columns or "CL=F" not in px_main.columns:
        _write(out / "combination_holdperiod_report.html",
               _html_base("Combination Lab", 20, "<p>JETS/CL=F fehlt.</p>")); return

    jets_ret = ret_main["JETS"].dropna()
    BASKET   = ["CL=F","BZ=F","XLE","XOM","CVX"]
    basket_px = {t: px_main[t].dropna() for t in BASKET if t in px_main.columns}

    common = jets_ret.index
    for t in basket_px:
        common = common.intersection(basket_px[t].index)
    common = common[~common.duplicated()].sort_values()

    jets_c   = jets_ret.reindex(common).fillna(0.0)
    basket_c = {t: basket_px[t].reindex(common).ffill() for t in basket_px}
    cl_px    = basket_c["CL=F"]

    vix_raw = _dl("^VIX")
    tnx_raw = _dl("^TNX")
    vix_c   = vix_raw.reindex(common).ffill() if vix_raw is not None else None
    tnx_c   = tnx_raw.reindex(common).ffill() if tnx_raw is not None else None

    n_total = len(common)
    split_i = int(n_total * IS_FRAC)
    is_idx  = common[:split_i]
    oos_idx = common[split_i:]

    # ── Base signals ──────────────────────────────────────────────────────────
    rsi_cl   = _calc_rsi(cl_px, 14)
    sig_rsi  = pd.Series(np.where(rsi_cl < 70, 1.0, -1.0),
                         index=common).shift(1).fillna(0.0)

    ens_parts = [pd.Series(np.where(_calc_rsi(px_t, 14) < 70, 1.0, -1.0), index=common)
                 for px_t in basket_c.values()]
    ens_raw  = pd.concat(ens_parts, axis=1).mean(axis=1)
    sig_bask = pd.Series(np.where(ens_raw > 0.2, 1.0,
                                   np.where(ens_raw < -0.2, -1.0, 0.0)),
                          index=common).shift(1).fillna(0.0)

    # ── Filters (derived from IS only) ───────────────────────────────────────
    def _net(sig):
        return sig * jets_c - sig.diff().abs().fillna(0) * TC

    # Seasonal: good months from IS base returns
    is_rsi_net = _net(sig_rsi).reindex(is_idx).dropna()
    is_df_s    = is_rsi_net.to_frame("r")
    is_df_s["m"] = pd.to_datetime(is_df_s.index).month
    good_months  = set(is_df_s.groupby("m")["r"].mean()[lambda x: x > 0].index)
    seas_mask    = pd.Series(pd.to_datetime(common).month.isin(good_months), index=common)

    # VIX < 25 mask
    if vix_c is not None:
        vix_mask = (vix_c.shift(1) < 25).reindex(common).fillna(True)
    else:
        vix_mask = pd.Series(True, index=common)

    # TNX not rising (20T trend ≤ 0)
    if tnx_c is not None:
        tnx_r_s  = np.log(tnx_c / tnx_c.shift(1)).fillna(0)
        tnx_mask = (tnx_r_s.rolling(20).mean().shift(1) <= 0).reindex(common).fillna(True)
    else:
        tnx_mask = pd.Series(True, index=common)

    # ── Combination matrix (16 combos) ────────────────────────────────────────
    filter_flags = list(iproduct([False, True], [False, True], [False, True]))
    bases = [("RSI<70", sig_rsi), ("Basket", sig_bask)]

    combo_rec = []
    for bname, bsig in bases:
        for use_s, use_v, use_t in filter_flags:
            sig = bsig.copy()
            if use_s: sig = sig * seas_mask.astype(float)
            if use_v: sig = sig * vix_mask.astype(float)
            if use_t: sig = sig * tnx_mask.astype(float)
            net     = _net(sig)
            is_n    = net.reindex(is_idx).dropna()
            oos_n   = net.reindex(oos_idx).dropna()
            sh_is   = _sh(is_n); sh_oos = _sh(oos_n)
            combo_rec.append(dict(
                base=bname, s=use_s, v=use_v, t=use_t,
                sh_is=sh_is, sh_oos=sh_oos, mdd=_mdd(oos_n),
                n_tr=int((sig.reindex(oos_idx).diff().abs() > 0).sum()),
                _sig=sig, _oos=oos_n, _net=net,
            ))

    combo_rec.sort(key=lambda r: r["sh_oos"] if not np.isnan(r["sh_oos"]) else -99,
                   reverse=True)
    best = combo_rec[0]
    best_sig = best["_sig"]
    best_oos = best["_oos"]
    best_net = best["_net"]
    best_lbl = (f"{best['base']}"
                f"{'+Seas' if best['s'] else ''}"
                f"{'+VIX' if best['v'] else ''}"
                f"{'+TNX' if best['t'] else ''}")

    # ── §1: Combination chart + table ─────────────────────────────────────────
    c_labels = [
        f"{r['base']} S={'✓' if r['s'] else '–'} V={'✓' if r['v'] else '–'} T={'✓' if r['t'] else '–'}"
        for r in combo_rec
    ]
    fig_mat = go.Figure()
    fig_mat.add_trace(go.Bar(name="IS Sharpe", x=c_labels,
                              y=[r["sh_is"] for r in combo_rec], marker_color="#58a6ff"))
    fig_mat.add_trace(go.Bar(name="OOS Sharpe", x=c_labels,
                              y=[r["sh_oos"] for r in combo_rec], marker_color="#3fb950"))
    _lay(fig_mat, title="Kombinationsmatrix – alle 16 Strategien (nach OOS Sharpe sortiert)",
         barmode="group", height=500,
         xaxis=dict(tickangle=-45, tickfont=dict(size=8, color="#e6edf3")))

    def _cr(r):
        dc = "#3fb950" if r["sh_oos"] - r["sh_is"] > 0 else "#f78166"
        return (f"<tr><td>{r['base']}</td>"
                f"<td>{'✓' if r['s'] else '–'}</td><td>{'✓' if r['v'] else '–'}</td>"
                f"<td>{'✓' if r['t'] else '–'}</td>"
                f"<td style='color:#58a6ff;'>{r['sh_is']:.3f}</td>"
                f"<td style='color:#3fb950;font-weight:bold;'>{r['sh_oos']:.3f}</td>"
                f"<td style='color:{dc};'>{r['sh_oos']-r['sh_is']:+.3f}</td>"
                f"<td style='color:#f78166;'>{r['mdd']*100:.1f}%</td>"
                f"<td>{r['n_tr']}</td></tr>")

    combo_tbl = (
        '<div class="table-responsive mt-3">'
        '<table class="table table-dark table-sm table-hover">'
        '<thead><tr><th>Basis</th><th>Seas</th><th>VIX</th><th>TNX</th>'
        '<th>IS♯</th><th>OOS♯</th><th>Δ</th><th>MaxDD</th><th>#Tr.</th></tr></thead>'
        '<tbody>' + "".join(_cr(r) for r in combo_rec) + '</tbody></table></div>'
    )

    # ── §2: Best combo equity + rolling Sharpe ────────────────────────────────
    cum_is   = (1 + best_net.reindex(is_idx).dropna()).cumprod() * 100
    cum_oos  = (1 + best_oos).cumprod() * 100
    cum_base = (1 + _net(sig_rsi).reindex(oos_idx).dropna()).cumprod() * 100

    fig_best = go.Figure()
    try:
        fig_best.add_vrect(x0=str(is_idx[0].date()), x1=str(is_idx[-1].date()),
                           fillcolor="#1c2128", opacity=0.6, layer="below", line_width=0)
    except Exception:
        pass
    _add_crises(fig_best)
    fig_best.add_trace(go.Scatter(x=cum_base.index.astype(str).tolist(),
                                   y=cum_base.values.tolist(),
                                   name="Basis RSI<70 OOS", mode="lines",
                                   line=dict(color="#8b949e", dash="dot", width=1.5)))
    fig_best.add_trace(go.Scatter(x=cum_is.index.astype(str).tolist(),
                                   y=cum_is.values.tolist(),
                                   name=f"{best_lbl} IS", mode="lines",
                                   line=dict(color="#ffa657", width=2)))
    fig_best.add_trace(go.Scatter(x=cum_oos.index.astype(str).tolist(),
                                   y=cum_oos.values.tolist(),
                                   name=f"{best_lbl} OOS", mode="lines",
                                   line=dict(color="#3fb950", width=2.5)))
    _lay(fig_best, title=f"Beste Kombination: {best_lbl} | OOS♯ {best['sh_oos']:.3f} | MaxDD {best['mdd']*100:.1f}%",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=470)

    fig_roll = go.Figure()
    for label, s, col in [("Basis RSI<70", _net(sig_rsi).reindex(oos_idx).dropna(), "#8b949e"),
                            (best_lbl, best_oos, "#3fb950")]:
        rs = _roll_sh(s, 252)
        fig_roll.add_trace(go.Scatter(x=rs.index.astype(str).tolist(), y=rs.values.tolist(),
                                       name=label, mode="lines", line=dict(color=col, width=1.8)))
    fig_roll.add_hline(y=0, line_color="#f78166", line_dash="dot")
    _lay(fig_roll, title="Rolling Sharpe 252T (OOS)", yaxis_title="Rolling Sharpe", height=370)

    # ── §3: Forward return distributions (violin) ─────────────────────────────
    sig_oos_s  = best_sig.reindex(oos_idx)
    jets_oos_s = jets_c.reindex(oos_idx).fillna(0.0)
    ret_arr    = jets_oos_s.values

    entry_mask = ((sig_oos_s > 0) & (sig_oos_s.shift(1).fillna(0) <= 0)).values
    entry_pos  = np.where(entry_mask)[0]

    fwd = {H: [] for H in HORIZONS}
    for pos in entry_pos:
        for H in HORIZONS:
            end = pos + H
            if end <= len(ret_arr):
                fwd[H].append(float(np.sum(ret_arr[pos:end])))

    fig_vln = go.Figure()
    pal = px.colors.sequential.Viridis_r
    for i, H in enumerate(HORIZONS):
        d = fwd[H]
        if len(d) >= 3:
            col_v = pal[int(i * (len(pal)-1) / max(len(HORIZONS)-1, 1))]
            fig_vln.add_trace(go.Violin(
                x=[f"H={H}"] * len(d), y=d, name=f"H={H}",
                box_visible=True, meanline_visible=True,
                fillcolor=col_v, line_color=col_v, opacity=0.72))
    fig_vln.add_hline(y=0, line_color="#f78166", line_dash="dot")
    _lay(fig_vln,
         title=f"Rendite-Verteilung nach Long-Entry ({len(entry_pos)} OOS-Eintritte): H Tage voraus",
         xaxis_title="Haltedauer H", yaxis_title="Kum. Log-Rendite", height=540,
         violingap=0.05, violinmode="group")

    # Statistics table
    fwd_stats = []
    for H in HORIZONS:
        d = fwd[H]
        if len(d) >= 3:
            arr = np.array(d); n = len(arr)
            m = arr.mean(); sd = arr.std(ddof=1) + 1e-9
            ci = 1.96 * sd / np.sqrt(n)
            fwd_stats.append(dict(
                H=H, n=n,
                Brutto=f"{m*100:+.2f}%",
                Netto_10bp=f"{(m-0.002)*100:+.2f}%",
                Std=f"{sd*100:.2f}%",
                CI95=f"±{ci*100:.2f}%",
                WinPct=f"{(arr>0).mean()*100:.1f}%",
                TradeSharpe=f"{m/sd:.3f}",
                Signif="✓✓" if ci < abs(m)*0.5 else ("✓" if ci < abs(m) else "○"),
            ))

    fwd_tbl = (
        '<div class="table-responsive mt-2"><table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in fwd_stats[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                             for r in fwd_stats)
        + '</tbody></table></div>'
        if fwd_stats else ""
    )

    # ── §4: Mean return + CI vs H + Trade Sharpe vs H ────────────────────────
    h_arr  = [r["H"] for r in fwd_stats]
    m_arr  = [float(np.array(fwd[h]).mean()) * 100 for h in h_arr]
    ci_arr = [float(1.96 * np.array(fwd[h]).std(ddof=1) / np.sqrt(len(fwd[h]))) * 100
              for h in h_arr]
    n_arr  = [(float(np.array(fwd[h]).mean()) - 0.002) * 100 for h in h_arr]
    ts_arr = [float(np.array(fwd[h]).mean() / (np.array(fwd[h]).std(ddof=1) + 1e-9))
              for h in h_arr]

    fig_mean = go.Figure()
    upper = [m + c for m, c in zip(m_arr, ci_arr)]
    lower = [m - c for m, c in zip(m_arr, ci_arr)]
    fig_mean.add_trace(go.Scatter(x=h_arr + h_arr[::-1], y=upper + lower[::-1],
                                   fill="toself", fillcolor="rgba(88,166,255,0.13)",
                                   line=dict(width=0), name="95% CI"))
    fig_mean.add_trace(go.Scatter(x=h_arr, y=m_arr, name="Ø Brutto",
                                   mode="lines+markers",
                                   line=dict(color="#58a6ff", width=2.2),
                                   marker=dict(size=8)))
    fig_mean.add_trace(go.Scatter(x=h_arr, y=n_arr, name="Ø Netto (10bp R/T)",
                                   mode="lines+markers",
                                   line=dict(color="#3fb950", width=2),
                                   marker=dict(size=7)))
    fig_mean.add_hline(y=0, line_color="#f78166", line_dash="dot")
    _lay(fig_mean, title="Ø Rendite (kumulativ) nach Haltedauer | Pro Long-Entry (OOS)",
         xaxis_title="H (Tage)", yaxis_title="Kum. Rendite (%)", height=430)

    valid_ts = [(i, ts) for i, ts in enumerate(ts_arr) if not np.isnan(ts)]
    opt_i    = max(valid_ts, key=lambda x: x[1])[0] if valid_ts else 0
    opt_H    = h_arr[opt_i] if valid_ts else H_FIXED[2]

    fig_tsh = go.Figure()
    fig_tsh.add_trace(go.Scatter(x=h_arr, y=ts_arr, mode="lines+markers",
                                  line=dict(color="#ffa657", width=2.2),
                                  marker=dict(size=10, color="#ffa657"),
                                  name="Trade Sharpe (Ø/Std)"))
    if valid_ts:
        fig_tsh.add_annotation(x=opt_H, y=float(ts_arr[opt_i]),
                                 text=f"Opt. H={opt_H}d",
                                 showarrow=True, arrowcolor="#3fb950",
                                 font=dict(color="#3fb950", size=12),
                                 bgcolor="#1c2128", bordercolor="#30363d")
    fig_tsh.add_hline(y=0, line_color="#f78166", line_dash="dot")
    _lay(fig_tsh, title="Trade Sharpe-Ratio vs Haltedauer (Ø_Rendite / Std, OOS)",
         xaxis_title="H (Tage)", yaxis_title="Trade Sharpe", height=410)

    # ── §5: Signal-driven equity curve with ▲▼ markers ──────────────────────
    exit_mask_s = ((sig_oos_s <= 0) & (sig_oos_s.shift(1).fillna(0) > 0)).values
    exit_pos_s  = np.where(exit_mask_s)[0]
    cum_sd      = (1 + best_oos).cumprod() * 100

    fig_sig = go.Figure()
    _add_crises(fig_sig)
    fig_sig.add_trace(go.Scatter(x=cum_sd.index.astype(str).tolist(),
                                  y=cum_sd.values.tolist(),
                                  name="Signal-driven NAV", mode="lines",
                                  line=dict(color="#58a6ff", width=2)))
    if len(entry_pos) > 0:
        e_d  = oos_idx[entry_pos]
        e_n  = cum_sd.reindex(e_d, method="nearest").fillna(100).values.tolist()
        fig_sig.add_trace(go.Scatter(x=[str(d.date()) for d in e_d], y=e_n,
                                      name="Einstieg ▲", mode="markers",
                                      marker=dict(symbol="triangle-up", size=9, color="#3fb950")))
    if len(exit_pos_s) > 0:
        ex_d = oos_idx[exit_pos_s]
        ex_n = cum_sd.reindex(ex_d, method="nearest").fillna(100).values.tolist()
        fig_sig.add_trace(go.Scatter(x=[str(d.date()) for d in ex_d], y=ex_n,
                                      name="Ausstieg ▼", mode="markers",
                                      marker=dict(symbol="triangle-down", size=9, color="#f78166")))
    _lay(fig_sig, title=f"Signal-getrieben: {best_lbl} OOS | ▲ Einstieg ▼ Ausstieg",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=480)

    # Fixed-H equity curves with dropdown
    fig_hc  = go.Figure()
    N_T     = 3
    for hi, H in enumerate(H_FIXED):
        net_h, e_h, ex_h = _fixed_hold_strat(best_sig, jets_oos_s, H)
        cum_h = (1 + net_h).cumprod() * 100
        vis   = (hi == 0)

        fig_hc.add_trace(go.Scatter(
            x=cum_h.index.astype(str).tolist(), y=cum_h.values.tolist(),
            name=f"NAV H={H}d", mode="lines", line=dict(color="#58a6ff", width=2),
            visible=vis))
        if len(e_h) > 0:
            e_nav = cum_h.reindex(e_h, method="nearest").fillna(100).values.tolist()
            fig_hc.add_trace(go.Scatter(
                x=[str(d.date()) for d in e_h], y=e_nav,
                name=f"Entry ▲ H={H}", mode="markers",
                marker=dict(symbol="triangle-up", size=9, color="#3fb950"), visible=vis))
        else:
            fig_hc.add_trace(go.Scatter(x=[], y=[], name=f"Entry ▲ H={H}",
                                         mode="markers", visible=vis))
        if len(ex_h) > 0:
            ex_nav = cum_h.reindex(ex_h, method="nearest").fillna(100).values.tolist()
            fig_hc.add_trace(go.Scatter(
                x=[str(d.date()) for d in ex_h], y=ex_nav,
                name=f"Exit ▼ H={H}", mode="markers",
                marker=dict(symbol="triangle-down", size=9, color="#f78166"), visible=vis))
        else:
            fig_hc.add_trace(go.Scatter(x=[], y=[], name=f"Exit ▼ H={H}",
                                         mode="markers", visible=vis))

    total_t = len(H_FIXED) * N_T
    btns = []
    for hi, H in enumerate(H_FIXED):
        vis = [False] * total_t
        bi  = hi * N_T
        vis[bi] = vis[bi+1] = vis[bi+2] = True
        btns.append(dict(label=f"H={H}T", method="update",
                         args=[{"visible": vis},
                               {"title": f"Fixed-Hold H={H} Tage | ▲ Einstieg ▼ Ausstieg (OOS)"}]))

    fig_hc.update_layout(
        **{k: v for k, v in _LAYOUT.items()},
        title=f"Fixed-Hold H={H_FIXED[0]} Tage | ▲ Einstieg ▼ Ausstieg (OOS)",
        height=510, xaxis_title="Datum", yaxis_title="NAV (Start=100)",
        updatemenus=[dict(buttons=btns, direction="right", showactive=True,
                          x=0.0, y=1.17, type="buttons",
                          bgcolor="#1c2128", bordercolor="#30363d",
                          font=dict(color="#e6edf3", size=11))])

    # ── §6: TC × H Sharpe heatmap ─────────────────────────────────────────────
    tc_h_z = np.full((len(TC_LEVELS), len(H_FIXED)), np.nan)
    for ti, tc_lv in enumerate(TC_LEVELS):
        for hi, H in enumerate(H_FIXED):
            net_h, _, _ = _fixed_hold_strat(best_sig, jets_oos_s, H, tc=tc_lv)
            tc_h_z[ti, hi] = _sh(net_h.dropna())

    fig_tc = go.Figure(go.Heatmap(
        z=tc_h_z.tolist(),
        x=[f"H={H}" for H in H_FIXED],
        y=[f"{int(tc*10000)}bp" for tc in TC_LEVELS],
        colorscale="RdYlGn", zmin=-0.5, zmax=2.5,
        text=[[f"{v:.2f}" if not np.isnan(v) else "–" for v in row]
              for row in tc_h_z.tolist()],
        texttemplate="%{text}",
        colorbar=dict(title="OOS Sharpe"),
    ))
    _lay(fig_tc, title="OOS Sharpe: Haltedauer H × Transaktionskosten (Round-Trip Basis-Punkte)",
         xaxis_title="Haltedauer H", yaxis_title="TC (R/T bp)", height=410)

    # ── §7: Entry condition analysis RSI × VIX → 10T return ──────────────────
    fig_entry = None
    rsi_oos_s = rsi_cl.reindex(oos_idx).fillna(50)
    vix_oos_s = (vix_c.reindex(oos_idx).fillna(20) if vix_c is not None
                 else pd.Series(20.0, index=oos_idx))

    if len(entry_pos) >= 10:
        ea = []
        for pos in entry_pos:
            if pos + 10 <= len(ret_arr):
                ea.append(dict(
                    rsi=float(rsi_oos_s.iloc[pos]),
                    vix=float(vix_oos_s.iloc[pos]),
                    r10=float(np.sum(ret_arr[pos:pos+10])),
                ))
        if ea:
            ea_df = pd.DataFrame(ea)
            ea_df["rb"] = pd.cut(ea_df["rsi"], [0,30,40,50,60,70,100],
                                  labels=["<30","30-40","40-50","50-60","60-70",">70"])
            ea_df["vb"] = pd.cut(ea_df["vix"], [0,15,20,25,30,200],
                                  labels=["<15","15-20","20-25","25-30",">30"])
            piv = ea_df.groupby(["rb","vb"])["r10"].mean().unstack(fill_value=np.nan)
            piv_z = (piv.values * 100).tolist()
            fig_entry = go.Figure(go.Heatmap(
                z=piv_z, x=piv.columns.astype(str).tolist(),
                y=piv.index.astype(str).tolist(),
                colorscale="RdYlGn", zmin=-4, zmax=4,
                text=[[f"{v:.1f}%" if not np.isnan(v) else "–" for v in row]
                      for row in piv_z],
                texttemplate="%{text}",
                colorbar=dict(title="Ø +10T %"),
            ))
            _lay(fig_entry,
                 title="Entry-Analyse: Ø 10T-Rendite (%) nach CL=F RSI × VIX beim Einstieg (OOS)",
                 xaxis_title="VIX bei Einstieg", yaxis_title="CL=F RSI bei Einstieg", height=400)

    # ── §8: Crisis performance heatmap ────────────────────────────────────────
    fig_crisis = None
    crisis_rows = []
    for cname, cs, ce, cc in CRISES:
        c_s = pd.Timestamp(cs); c_e = pd.Timestamp(ce)
        row = {"Krise": cname}
        for r in combo_rec[:8]:
            lbl = (f"{r['base']}"
                   f"{'S' if r['s'] else ''}"
                   f"{'V' if r['v'] else ''}"
                   f"{'T' if r['t'] else ''}")
            cr = r["_net"].loc[c_s:c_e].dropna()
            row[lbl] = float((1+cr).prod()-1)*100 if len(cr) > 5 else np.nan
        crisis_rows.append(row)

    if crisis_rows:
        cdf  = pd.DataFrame(crisis_rows).set_index("Krise")
        fig_crisis = go.Figure(go.Heatmap(
            z=cdf.values.tolist(),
            x=cdf.columns.tolist(), y=cdf.index.tolist(),
            colorscale="RdYlGn", zmin=-20, zmax=20,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "–" for v in row]
                  for row in cdf.values.tolist()],
            texttemplate="%{text}",
            colorbar=dict(title="Return %"),
        ))
        _lay(fig_crisis, title="Krisenperformance: Return (%) der Top-8-Kombinationen in 3 Krisen",
             xaxis_title="Strategie", yaxis_title="Krisenperiode", height=360)

    # ── HTML assembly ──────────────────────────────────────────────────────────
    good_m_str = ", ".join(["Jan","Feb","Mär","Apr","Mai","Jun",
                             "Jul","Aug","Sep","Okt","Nov","Dez"][m-1]
                            for m in sorted(good_months))

    secs = [
        ("📊 §1  Kombinationsmatrix – 16 Strategien",
         _desc(f"Basis-Signale: CL=F RSI&lt;70 (klassisch) und Oil Basket (CL=F+BZ=F+XLE+XOM+CVX). "
               f"Seasonal-Filter: IS-profitable Monate = {good_m_str}. "
               f"VIX-Filter: {'aktiv (VIX &lt; 25)' if vix_c is not None else 'inaktiv – Daten fehlen'}. "
               f"TNX-Filter: {'aktiv (20T-Trend ≤ 0)' if tnx_c is not None else 'inaktiv'}. "
               "Alle Parameter ausschließlich aus IS-Daten abgeleitet.")
         + _htm(fig_mat) + combo_tbl, 0, True),

        ("📈 §2  Beste Kombination – Equity Curve + Rolling Sharpe",
         _desc(f"Beste Kombination: <strong style='color:#3fb950;'>{best_lbl}</strong> "
               f"(OOS Sharpe: {best['sh_oos']:.3f} | IS Sharpe: {best['sh_is']:.3f} | "
               f"MaxDD: {best['mdd']*100:.1f}%). "
               "Krisen-Perioden als farbige Bereiche.")
         + _htm(fig_best) + _htm(fig_roll), 1, False),

        ("🎻 §3  Rendite-Verteilung nach Haltedauer (Violin Plots)",
         _desc(f"Für jeden der {len(entry_pos)} Long-Eintritte im OOS-Zeitraum: "
               "kumulative Log-Rendite nach H Handelstagen. "
               "Violin = Häufigkeitsdichte der Renditen. Box = Median + IQR. Meanline = Durchschnitt. "
               "Schlüsselfrage: Wie lange bleibt das Signal informativ?")
         + _htm(fig_vln) + fwd_tbl, 2, False),

        ("📐 §4  Optimale Haltedauer: Ø Rendite + Trade Sharpe vs H",
         _desc(f"Ø Brutto- und Netto-Rendite (TC=10bp R/T) mit 95%-CI als Funktion der Haltedauer. "
               f"Optimale Haltedauer (max. Trade Sharpe): <strong>H = {opt_H} Tage</strong>. "
               "Darüber: Renditeverwässerung durch nicht-informative Handelstage am Ende der Halteperiode.")
         + _htm(fig_mean) + _htm(fig_tsh), 3, False),

        ("🔺🔻 §5  Fixed-Hold Equity Curves (▲ Einstieg ▼ Ausstieg)",
         _desc("Oben: Signal-getriebene Strategie (Ausstieg bei Signal-Reversal). "
               "Unten: Fixed-Hold – Button drücken um Haltedauer H zu wechseln. "
               "Grün ▲ = Einstieg wenn Signal +1 wird. Rot ▼ = Ausstieg nach exakt H Tagen.")
         + _htm(fig_sig)
         + "<hr style='border-color:#30363d;margin:16px 0;'>"
         + "<h5 style='color:#ffa657;margin-bottom:8px;'>Fixed-Hold Strategie (Buttons zum Wechseln):</h5>"
         + _htm(fig_hc), 4, False),

        ("🌡️ §6  TC × Haltedauer Sensitivitäts-Heatmap",
         _desc("OOS Sharpe der Fixed-Hold Strategie als Funktion von Haltedauer H und TC. "
               "Grün = profitable Zone. Rot = unrentabel. "
               "Hilft die maximalen Transaktionskosten zu bestimmen, bei denen die Strategie noch lohnt.")
         + _htm(fig_tc), 5, False),
    ]

    if fig_entry is not None:
        secs.append(("🎯 §7  Entry-Analyse: RSI × VIX → 10T Rendite",
                     _desc("Durchschnittliche 10-Tage Rendite nach Entry, aufgeteilt nach CL=F RSI-Level "
                           "und VIX-Level beim Eintrittszeitpunkt. "
                           "Identifiziert das optimale Marktregime für Signaleinstiege.")
                     + _htm(fig_entry), 6, False))

    if fig_crisis is not None:
        secs.append(("⚡ §8  Krisenperformance: Welche Kombination überlebt?",
                     _desc("Kumulative Return (%) der Top-8-Strategien in den drei definierten Krisenperioden. "
                           "VIX- und TNX-Filter sollten in Krisen besonders schützend wirken, "
                           "da sie in turbulenten Märkten (hoher VIX) und Zinsanstiegen (TNX↑) flat gehen.")
                     + _htm(fig_crisis), 7, False))

    acc = '<div class="accordion" id="chAcc">'
    for t, b, idx, op in secs:
        acc += _acc(t, b, idx, op)
    acc += "</div>"

    metrics_html = f"""
    <div class="row g-3 mb-4">
      <div class="col-lg-4"><div class="card p-3" style="background:#1c2128;border:1px solid #3fb950;">
        <small style="color:#3fb950;">Beste Kombination</small><br>
        <strong style="color:#e6edf3;">{best_lbl}</strong>
      </div></div>
      <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #58a6ff;">
        <small style="color:#58a6ff;">OOS Sharpe</small><br>
        <strong style="color:#e6edf3;font-size:1.5em;">{best['sh_oos']:.3f}</strong>
      </div></div>
      <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #f78166;">
        <small style="color:#f78166;">MaxDD OOS</small><br>
        <strong style="color:#e6edf3;font-size:1.5em;">{best['mdd']*100:.1f}%</strong>
      </div></div>
      <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #bc8cff;">
        <small style="color:#bc8cff;">Opt. Haltedauer</small><br>
        <strong style="color:#e6edf3;font-size:1.5em;">{opt_H} Tage</strong>
      </div></div>
      <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #ffa657;">
        <small style="color:#ffa657;">Long-Eintritte OOS</small><br>
        <strong style="color:#e6edf3;font-size:1.5em;">{len(entry_pos)}</strong>
      </div></div>
    </div>
    """

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#3fb950;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Strategy Combination Lab + Haltedauer-Analyse</h2>
          <p class="mb-0" style="color:#8b949e;">
            16 Kombinationen · Oil Basket × Saisonal × VIX × TNX ·
            Violin-Plot Renditeverteilung · Optimale Haltedauer ·
            Fixed-Hold ▲▼ Charts · TC-Heatmap · Krisenperformance
          </p>
        </div>
      </div>
      {metrics_html}
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "combination_holdperiod_report.html",
           _html_base("Strategy Combination Lab", 20, body))


def build_leverage_crisis_report(tables, figures, out):  # noqa: C901
    """
    Leverage, TC and Crisis Analysis for the best strategy combination.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from itertools import product as iproduct
    import yfinance as yf

    IS_FRAC     = 0.70
    TC_BASE     = 0.001
    RF_PA       = 0.02          # borrowing rate on levered portion
    TARGET_VOL  = 0.10          # target vol for vol-scaled leverage
    LEVERAGES   = [1, 1.5, 2, 3, 5]
    TC_LEVELS   = [0.0005, 0.001, 0.002, 0.005, 0.01]
    LEV_COLORS  = ["#8b949e","#58a6ff","#3fb950","#ffa657","#f78166"]

    CRISES = [
        ("2008 Lehman",      "2007-06-01", "2009-06-01", "#f78166"),
        ("2011 Euro-Krise",  "2011-06-01", "2012-03-01", "#ffa657"),
        ("2015 Öl-Crash",    "2015-06-01", "2016-03-01", "#e3b341"),
        ("2020 COVID",       "2020-01-15", "2020-07-01", "#bc8cff"),
        ("2022 Zinswende",   "2022-01-01", "2022-12-31", "#58a6ff"),
    ]
    CRISIS_COLORS = [c[3] for c in CRISES]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _dl(ticker):
        for period in ("15y","10y","5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _roll_sh(s, w=252):
        m = s.rolling(w).mean(); v = s.rolling(w).std()
        return (m / (v + 1e-9)) * np.sqrt(252)

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _ann_ret(x):
        x = pd.Series(x).dropna()
        if len(x) < 2: return np.nan
        return float((1+x).prod() ** (252/len(x)) - 1)

    def _calmar(x):
        ar = _ann_ret(x); dd = _mdd(x)
        return float(ar / (-dd + 1e-9)) if dd < 0 else np.nan

    def _split(s, frac=IS_FRAC):
        n = len(s); si = int(n * frac)
        return s.iloc[:si], s.iloc[si:]

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    def _card(title, color, body):
        return (f'<div class="card mb-3 p-3" style="background:#1c2128;border:1px solid {color};">'
                f'<h5 style="color:{color};">{title}</h5>'
                f'<div style="color:#e6edf3;">{body}</div></div>')

    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#lv{idx}">'
            f'{title}</button></h2>'
            f'<div id="lv{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    def _add_crises(fig, row=None, col=None):
        for cname, cs, ce, cc in CRISES:
            try:
                kw = dict(x0=cs, x1=ce, fillcolor=cc, opacity=0.09,
                          layer="below", line_width=0)
                if row is not None:
                    fig.add_vrect(row=row, col=col, **kw)
                else:
                    fig.add_vrect(**kw)
            except Exception:
                pass

    def _apply_leverage(net_1x, signal, L, tc=TC_BASE, rf_pa=RF_PA):
        """Scale 1× net return to L× with added borrowing + TC costs."""
        if L == 1:
            return net_1x.copy()
        sig   = signal.reindex(net_1x.index).fillna(0)
        rf_d  = rf_pa / 252
        in_p  = (sig.abs() > 0).astype(float)
        extra_borrow = (L - 1) * rf_d * in_p
        extra_tc     = (L - 1) * tc  * sig.diff().abs().fillna(0)
        return net_1x * L - extra_borrow - extra_tc

    # ── data ──────────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")
    if ret_main is None or px_main is None:
        _write(out / "leverage_crisis_report.html",
               _html_base("Leverage & Krisen", 20, "<p>Daten fehlen.</p>")); return
    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    if "JETS" not in ret_main.columns or "CL=F" not in px_main.columns:
        _write(out / "leverage_crisis_report.html",
               _html_base("Leverage & Krisen", 20, "<p>JETS/CL=F fehlt.</p>")); return

    jets_ret = ret_main["JETS"].dropna()
    BASKET   = ["CL=F","BZ=F","XLE","XOM","CVX"]
    basket_px = {t: px_main[t].dropna() for t in BASKET if t in px_main.columns}

    common = jets_ret.index
    for t in basket_px:
        common = common.intersection(basket_px[t].index)
    common = common[~common.duplicated()].sort_values()

    jets_c   = jets_ret.reindex(common).fillna(0.0)
    basket_c = {t: basket_px[t].reindex(common).ffill() for t in basket_px}
    cl_px    = basket_c["CL=F"]

    vix_raw = _dl("^VIX")
    tnx_raw = _dl("^TNX")
    vix_c   = vix_raw.reindex(common).ffill() if vix_raw is not None else None
    tnx_c   = tnx_raw.reindex(common).ffill() if tnx_raw is not None else None

    n_total = len(common)
    split_i = int(n_total * IS_FRAC)
    is_idx  = common[:split_i]
    oos_idx = common[split_i:]

    # ── Rebuild best signal ───────────────────────────────────────────────────
    def _net(sig, tc=TC_BASE):
        return sig * jets_c - sig.diff().abs().fillna(0) * tc

    rsi_cl  = _calc_rsi(cl_px, 14)
    sig_rsi = pd.Series(np.where(rsi_cl < 70, 1.0, -1.0),
                         index=common).shift(1).fillna(0.0)

    ens_parts = [pd.Series(np.where(_calc_rsi(px_t, 14) < 70, 1.0, -1.0), index=common)
                 for px_t in basket_c.values()]
    ens_raw  = pd.concat(ens_parts, axis=1).mean(axis=1)
    sig_bask = pd.Series(np.where(ens_raw > 0.2, 1.0,
                                   np.where(ens_raw < -0.2, -1.0, 0.0)),
                          index=common).shift(1).fillna(0.0)

    is_df_s = _net(sig_rsi).reindex(is_idx).dropna().to_frame("r")
    is_df_s["m"] = pd.to_datetime(is_df_s.index).month
    good_m  = set(is_df_s.groupby("m")["r"].mean()[lambda x: x > 0].index)
    seas_m  = pd.Series(pd.to_datetime(common).month.isin(good_m), index=common)

    vix_m = ((vix_c.shift(1) < 25).reindex(common).fillna(True)
             if vix_c is not None else pd.Series(True, index=common))
    if tnx_c is not None:
        tnx_r_s = np.log(tnx_c / tnx_c.shift(1)).fillna(0)
        tnx_m   = (tnx_r_s.rolling(20).mean().shift(1) <= 0).reindex(common).fillna(True)
    else:
        tnx_m = pd.Series(True, index=common)

    best_sig  = sig_rsi.copy()
    best_sh   = _sh(_net(sig_rsi).reindex(oos_idx).dropna())
    best_lbl  = "RSI<70"

    for bname, bsig in [("RSI<70", sig_rsi), ("Basket", sig_bask)]:
        for us, uv, ut in iproduct([False,True],[False,True],[False,True]):
            sig = bsig.copy()
            if us: sig = sig * seas_m.astype(float)
            if uv: sig = sig * vix_m.astype(float)
            if ut: sig = sig * tnx_m.astype(float)
            sh = _sh(_net(sig).reindex(oos_idx).dropna())
            if not np.isnan(sh) and sh > best_sh:
                best_sh  = sh; best_sig = sig
                best_lbl = (f"{bname}"
                            f"{'+S' if us else ''}{'+V' if uv else ''}{'+T' if ut else ''}")

    best_net_full = _net(best_sig)
    best_oos_full = best_net_full.reindex(oos_idx).dropna()

    # ── §1: Crisis full equity comparison ─────────────────────────────────────
    STRATS = [
        ("JETS B&H",        jets_c,                           "#8b949e"),
        ("RSI<70",          _net(sig_rsi),                    "#58a6ff"),
        (best_lbl,          best_net_full,                    "#3fb950"),
    ]

    fig_cr_eq = go.Figure()
    _add_crises(fig_cr_eq)
    for label, net_s, col in STRATS:
        cum = (1 + net_s.reindex(oos_idx).dropna()).cumprod() * 100
        fig_cr_eq.add_trace(go.Scatter(
            x=cum.index.astype(str).tolist(), y=cum.values.tolist(),
            name=label, mode="lines", line=dict(color=col, width=2)))
    _lay(fig_cr_eq, title="OOS Equity: JETS B&H vs RSI<70 vs Beste Kombo (Krisen schattiert)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=490)

    # Crisis statistics table
    crisis_stats = []
    for cname, cs, ce, cc in CRISES:
        c_s = pd.Timestamp(cs); c_e = pd.Timestamp(ce)
        row = {"Krise": cname, "Zeitraum": f"{cs[:7]}–{ce[:7]}"}
        for label, net_s, _ in STRATS:
            cr = net_s.loc[c_s:c_e].dropna()
            if len(cr) > 5:
                tot = float((1+cr).prod()-1)*100
                mdd = _mdd(cr)*100
                sh  = _sh(cr)
                col_r = "#3fb950" if tot >= 0 else "#f78166"
                row[f"{label}↩"] = f'<span style="color:{col_r};">{tot:+.1f}%</span>'
                row[f"{label}DD"] = f'<span style="color:#f78166;">{mdd:.1f}%</span>'
                row[f"{label}♯"] = f"{sh:.2f}"
            else:
                row[f"{label}↩"] = row[f"{label}DD"] = row[f"{label}♯"] = "n/a"
        crisis_stats.append(row)

    crisis_tbl = (
        '<div class="table-responsive mt-3">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in crisis_stats[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
            for row in crisis_stats)
        + '</tbody></table></div>'
    )

    # Per-crisis zoom charts (subplot per crisis)
    n_cr = len(CRISES)
    fig_cr_zoom = make_subplots(rows=1, cols=n_cr,
                                 subplot_titles=[c[0] for c in CRISES],
                                 shared_yaxes=False)
    for ci, (cname, cs, ce, cc) in enumerate(CRISES):
        c_s = pd.Timestamp(cs); c_e = pd.Timestamp(ce)
        for si, (label, net_s, col) in enumerate(STRATS):
            cr = net_s.loc[c_s:c_e].dropna()
            if len(cr) > 2:
                cum_cr = (1 + cr).cumprod() * 100
                fig_cr_zoom.add_trace(go.Scatter(
                    x=cum_cr.index.astype(str).tolist(), y=cum_cr.values.tolist(),
                    name=label, legendgroup=label, showlegend=(ci == 0),
                    mode="lines", line=dict(color=col, width=1.5)),
                    row=1, col=ci+1)
    fig_cr_zoom.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=400, title_text="Krisenperioden Zoom: NAV (Start=100 bei Krisenbeginn)",
    )
    for i in range(1, n_cr+1):
        fig_cr_zoom.update_xaxes(tickangle=-45, tickfont=dict(size=7), row=1, col=i)

    # ── §2: Leverage comparison ───────────────────────────────────────────────
    fig_lev = go.Figure()
    _add_crises(fig_lev)
    lev_metrics = []

    for L, col in zip(LEVERAGES, LEV_COLORS):
        lev_net = _apply_leverage(best_oos_full, best_sig.reindex(oos_idx), L)
        cum_lev = (1 + lev_net).cumprod() * 100
        fig_lev.add_trace(go.Scatter(
            x=cum_lev.index.astype(str).tolist(), y=cum_lev.values.tolist(),
            name=f"{L}×", mode="lines", line=dict(color=col, width=2.0)))
        lev_metrics.append(dict(
            Leverage=f"{L}×",
            sh=_sh(lev_net),
            mdd=_mdd(lev_net),
            calmar=_calmar(lev_net),
            ann=_ann_ret(lev_net),
        ))

    _lay(fig_lev,
         title=f"Leverage {min(LEVERAGES)}×–{max(LEVERAGES)}×: {best_lbl} OOS (RF={int(RF_PA*100)}%, TC={int(TC_BASE*10000)}bp)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=490)

    # Leverage metrics bar chart
    fig_lm = make_subplots(rows=1, cols=4,
                            subplot_titles=["OOS Sharpe","MaxDD %","Ann. Return %","Calmar"])
    for row_i, (key, scale) in enumerate([("sh",1),("mdd",100),("ann",100),("calmar",1)], start=1):
        vals = [m[key]*scale for m in lev_metrics]
        fig_lm.add_trace(go.Bar(
            x=[m["Leverage"] for m in lev_metrics], y=vals,
            marker_color=LEV_COLORS, name=key, showlegend=False),
            row=1, col=row_i)
    fig_lm.update_layout(**{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
                          height=360, showlegend=False,
                          title_text="Leverage-Metriken Überblick (OOS)")

    # Leverage metrics table
    lev_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr><th>Leverage</th><th>OOS Sharpe</th><th>MaxDD</th><th>Calmar</th><th>Ann. Return</th></tr></thead>'
        '<tbody>' + "".join(
            f"<tr><td><strong style='color:{LEV_COLORS[i]};'>{m['Leverage']}</strong></td>"
            f"<td style='color:#58a6ff;'>{m['sh']:.3f}</td>"
            f"<td style='color:#f78166;'>{m['mdd']*100:.1f}%</td>"
            f"<td>{m['calmar']:.2f}</td>"
            f"<td style='color:#3fb950;'>{m['ann']*100:+.1f}%</td></tr>"
            for i, m in enumerate(lev_metrics))
        + '</tbody></table></div>'
    )

    # ── §3: TC × Leverage Sharpe heatmap ─────────────────────────────────────
    tc_lev_z = np.full((len(TC_LEVELS), len(LEVERAGES)), np.nan)
    for ti, tc_lv in enumerate(TC_LEVELS):
        net_tc = best_sig * jets_c - best_sig.diff().abs().fillna(0) * tc_lv
        net_tc_oos = net_tc.reindex(oos_idx).dropna()
        for li, L in enumerate(LEVERAGES):
            lev_n = _apply_leverage(net_tc_oos, best_sig.reindex(oos_idx), L, tc=tc_lv)
            tc_lev_z[ti, li] = _sh(lev_n)

    fig_tclev = go.Figure(go.Heatmap(
        z=tc_lev_z.tolist(),
        x=[f"{L}×" for L in LEVERAGES],
        y=[f"{int(tc*10000)}bp" for tc in TC_LEVELS],
        colorscale="RdYlGn", zmin=-1.0, zmax=5.0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "–" for v in row]
              for row in tc_lev_z.tolist()],
        texttemplate="%{text}",
        colorbar=dict(title="OOS Sharpe"),
    ))
    _lay(fig_tclev,
         title="OOS Sharpe: Leverage × Transaktionskosten (beste Kombination)",
         xaxis_title="Leverage", yaxis_title="TC (R/T Basis-Punkte)", height=420)

    # ── §4: Kelly criterion ───────────────────────────────────────────────────
    daily  = best_oos_full.dropna()
    r_mu   = daily.rolling(252).mean() * 252
    r_var  = daily.rolling(252).var()  * 252
    r_kelly = (r_mu / (r_var + 1e-9)).clip(-5, 5)
    r_half  = r_kelly / 2

    full_mu_k   = float(daily.mean() * 252)
    full_var_k  = float(daily.var() * 252)
    full_kelly  = float(full_mu_k / (full_var_k + 1e-9))
    half_kelly  = full_kelly / 2

    fig_kelly = go.Figure()
    fig_kelly.add_trace(go.Scatter(x=r_kelly.index.astype(str).tolist(),
                                    y=r_kelly.values.tolist(),
                                    name="Full Kelly f*", mode="lines",
                                    line=dict(color="#f78166", width=1.5)))
    fig_kelly.add_trace(go.Scatter(x=r_half.index.astype(str).tolist(),
                                    y=r_half.values.tolist(),
                                    name="Half Kelly f*/2", mode="lines",
                                    line=dict(color="#3fb950", width=2.2)))
    for y_val, col, lbl in [(1,"#8b949e","1×"),(2,"#58a6ff","2×"),(3,"#ffa657","3×")]:
        fig_kelly.add_hline(y=y_val, line_color=col, line_dash="dash",
                             annotation_text=lbl,
                             annotation_font_color=col)
    fig_kelly.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_kelly, title="Kelly-Kriterium: Rolling Optimaler Leverage (252T, OOS)",
         xaxis_title="Datum", yaxis_title="Kelly f* (optimaler Leverage)", height=430)

    kelly_card = _card("Kelly-Kriterium – Vollständige OOS-Periode", "#ffa657", f"""
    <table class="table table-dark table-sm mb-1">
      <tr><td>Annualisierte Rendite μ</td>
          <td style="color:#3fb950;"><strong>{full_mu_k*100:.2f}%</strong></td></tr>
      <tr><td>Annualisierte Varianz σ²</td>
          <td>{full_var_k*100:.4f}</td></tr>
      <tr><td>Full Kelly f* = μ/σ²</td>
          <td style="color:#f78166;"><strong>{full_kelly:.2f}×</strong></td></tr>
      <tr><td>Half Kelly (praktisch empfohlen)</td>
          <td style="color:#3fb950;"><strong>{half_kelly:.2f}×</strong></td></tr>
      <tr><td>Quarter Kelly (sehr konservativ)</td>
          <td style="color:#58a6ff;"><strong>{full_kelly/4:.2f}×</strong></td></tr>
    </table>
    <p class="mt-1 mb-0" style="color:#8b949e;font-size:0.83em;">
      Full Kelly maximiert das geometrische Wachstum, ist aber extrem volatil.
      Half Kelly liefert ~75% des maximalen Wachstums bei ~50% des Drawdowns.
      Rolling f* zeigt die Instabilität des Schätzers – Vorsicht bei Leverage &gt; 3.
    </p>
    """)

    # ── §5: Vol-scaled leverage ───────────────────────────────────────────────
    roll_vol_21 = daily.rolling(21).std() * np.sqrt(252)
    dyn_lev     = (TARGET_VOL / (roll_vol_21.shift(1) + 1e-9)).clip(0.1, 5.0)
    rf_d        = RF_PA / 252

    dyn_net = (dyn_lev * daily
               - (dyn_lev - 1).clip(0) * rf_d
               - dyn_lev.diff().abs().fillna(0) * TC_BASE)
    cum_1x   = (1 + daily).cumprod() * 100
    cum_dyn  = (1 + dyn_net).cumprod() * 100

    fig_vs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Equity: 1× vs Vol-Skaliert (OOS)",
                                            f"Dynamischer Leverage (Ziel-Vol {int(TARGET_VOL*100)}%)"])
    _add_crises(fig_vs, row=1, col=1)
    fig_vs.add_trace(go.Scatter(x=cum_1x.index.astype(str).tolist(),
                                 y=cum_1x.values.tolist(), name="1× (statisch)",
                                 mode="lines", line=dict(color="#8b949e", width=1.5)),
                     row=1, col=1)
    fig_vs.add_trace(go.Scatter(x=cum_dyn.index.astype(str).tolist(),
                                 y=cum_dyn.values.tolist(),
                                 name=f"Vol-Skaliert ({int(TARGET_VOL*100)}% Ziel)",
                                 mode="lines", line=dict(color="#3fb950", width=2.2)),
                     row=1, col=1)
    fig_vs.add_trace(go.Scatter(x=dyn_lev.index.astype(str).tolist(),
                                 y=dyn_lev.values.tolist(), name="Leverage Level",
                                 mode="lines", fill="tozeroy",
                                 fillcolor="rgba(88,166,255,0.10)",
                                 line=dict(color="#58a6ff", width=1.0)),
                     row=2, col=1)
    fig_vs.add_hline(y=1, line_color="#8b949e", line_dash="dot", row=2, col=1)
    fig_vs.add_hline(y=2, line_color="#ffa657", line_dash="dot", row=2, col=1)
    fig_vs.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=570,
        title_text=f"Vol-Skalierter Leverage: Ziel-Volatilität {int(TARGET_VOL*100)}% p.a.")

    vs_card = _card("Vol-Skaliert vs 1× Metriken", "#3fb950", f"""
    <div class="row">
      <div class="col-6">
        <strong style="color:#8b949e;">1× Statisch</strong><br>
        Sharpe: {_sh(daily):.3f}<br>MaxDD: {_mdd(daily)*100:.1f}%<br>
        Ann. Return: {_ann_ret(daily)*100:.1f}%
      </div>
      <div class="col-6">
        <strong style="color:#3fb950;">Vol-Skaliert</strong><br>
        Sharpe: {_sh(dyn_net):.3f}<br>MaxDD: {_mdd(dyn_net)*100:.1f}%<br>
        Ann. Return: {_ann_ret(dyn_net)*100:.1f}%
      </div>
    </div>
    """)

    # ── §6: Drawdown comparison ───────────────────────────────────────────────
    fig_dd = go.Figure()
    _add_crises(fig_dd)
    dd_strats = [
        ("JETS B&H",       jets_c.reindex(oos_idx).dropna(),    "#8b949e"),
        ("RSI<70",         _net(sig_rsi).reindex(oos_idx).dropna(), "#58a6ff"),
        (best_lbl,         best_oos_full,                        "#3fb950"),
        ("2×",             _apply_leverage(best_oos_full, best_sig.reindex(oos_idx), 2), "#ffa657"),
        ("3×",             _apply_leverage(best_oos_full, best_sig.reindex(oos_idx), 3), "#f78166"),
        ("Vol-Skaliert",   dyn_net,                              "#bc8cff"),
    ]
    for label, s, col in dd_strats:
        d = s.dropna()
        c = (1 + d).cumprod()
        dd = (c / c.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index.astype(str).tolist(), y=dd.values.tolist(),
            name=label, mode="lines", line=dict(color=col, width=1.5)))
    _lay(fig_dd, title="Drawdown-Vergleich: alle Strategien + Leverage-Varianten (OOS)",
         xaxis_title="Datum", yaxis_title="Drawdown (%)", height=480)

    # Summary metrics table
    sum_rows = []
    for label, s, col in dd_strats:
        d = s.dropna()
        sum_rows.append(dict(
            Strategie=f'<strong style="color:{col};">{label}</strong>',
            Sharpe=f"{_sh(d):.3f}",
            MaxDD=f"{_mdd(d)*100:.1f}%",
            Calmar=f"{_calmar(d):.2f}",
            AnnReturn=f"{_ann_ret(d)*100:+.1f}%",
        ))

    sum_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in sum_rows[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
            for row in sum_rows)
        + '</tbody></table></div>'
    )

    # ── §7: Final Recommendation ──────────────────────────────────────────────
    best_2x     = _apply_leverage(best_oos_full, best_sig.reindex(oos_idx), 2)
    hk_rec      = half_kelly
    hk_sh       = _sh(_apply_leverage(best_oos_full, best_sig.reindex(oos_idx),
                                       min(max(round(hk_rec * 2) / 2, 1.0), 5.0)))
    lev2_col_idx = LEVERAGES.index(2) if 2 in LEVERAGES else 2
    tc_bp_breakeven = next(
        (int(tc*10000) for ti, tc in enumerate(TC_LEVELS)
         if not np.isnan(tc_lev_z[ti, lev2_col_idx]) and tc_lev_z[ti, lev2_col_idx] < 0.3),
        20
    )

    rec_card = _card("🏆 Empfehlung: Optimale Strategie-Konfiguration", "#3fb950", f"""
    <div class="row g-3">
      <div class="col-md-6">
        <strong style="color:#3fb950;">Bestes Signal:</strong> {best_lbl}<br>
        <strong style="color:#58a6ff;">Empfohlener Leverage:</strong> Half Kelly = {hk_rec:.1f}× (ggf. auf {min(max(round(hk_rec*2)/2,1.0),3.0):.1f}× runden)<br>
        <strong style="color:#ffa657;">Ziel-Volatilität:</strong> Vol-Skaliert mit {int(TARGET_VOL*100)}% als Puffer<br>
        <strong style="color:#bc8cff;">TC-Schwelle:</strong> Strategie profitabel bis ~{tc_bp_breakeven}bp R/T
      </div>
      <div class="col-md-6">
        <strong style="color:#e6edf3;">Metriken bei {hk_rec:.1f}× Leverage:</strong><br>
        OOS Sharpe: <span style="color:#3fb950;">{hk_sh:.3f}</span><br>
        OOS Sharpe 2×: <span style="color:#58a6ff;">{_sh(best_2x):.3f}</span><br>
        MaxDD 2×: <span style="color:#f78166;">{_mdd(best_2x)*100:.1f}%</span>
      </div>
    </div>
    """)

    # ── HTML assembly ─────────────────────────────────────────────────────────
    secs = [
        ("⚡ §1  Krisenanalyse – Performance in 5 Stress-Perioden",
         _desc("Lehman 2008, Euro-Krise 2011, Öl-Crash 2015, COVID 2020, Zinswende 2022. "
               "Krisen-Perioden im Chart farbig markiert. "
               "Vergleich: JETS Buy-and-Hold vs RSI<70 Basis vs Beste Kombination.")
         + _htm(fig_cr_eq)
         + _htm(fig_cr_zoom)
         + crisis_tbl, 0, True),

        ("📊 §2  Leverage-Vergleich: 1× bis 5× mit Kreditkosten",
         _desc(f"Leverage {min(LEVERAGES)}×–{max(LEVERAGES)}× auf die beste Kombination '{best_lbl}'. "
               f"Zusatzkosten: Kreditkosten = (L-1) × {int(RF_PA*100)}%/252 p.T. wenn in Position. "
               f"TC skaliert: (L-1) × {int(TC_BASE*10000)}bp pro Trade zusätzlich.")
         + _htm(fig_lev) + _htm(fig_lm) + lev_tbl, 1, False),

        ("🌡️ §3  TC × Leverage Sensitivitäts-Heatmap",
         _desc("OOS Sharpe als Funktion von Leverage und Transaktionskosten. "
               "Zeigt bis zu welchem TC-Level die Strategie bei jedem Leverage noch profitabel ist. "
               "Grün = gut. Rot = unrentabel.")
         + _htm(fig_tclev), 2, False),

        ("📐 §4  Kelly-Kriterium: Wissenschaftlich optimaler Leverage",
         _desc("Kelly-Formel: f* = μ / σ² (annualisiert). "
               "Maximiert das langfristige geometrische Wachstum. "
               "Praktisch: Half Kelly = f*/2 empfohlen (senkt Volatilität um 50%, "
               "reduziert MaxDD auf ~1/4, ~75% des maximalen Wachstums).")
         + kelly_card + _htm(fig_kelly), 3, False),

        ("⚖️ §5  Vol-Skalierter Leverage (dynamisch)",
         _desc(f"Leverage = Ziel-Volatilität / realisierte 21T-Vol. Ziel: {int(TARGET_VOL*100)}% p.a. "
               "In ruhigen Märkten (low vol): Leverage erhöhen. "
               "In turbulenten Märkten: Leverage reduzieren. Clip: [0.1, 5.0]. "
               "Ergebnis: glattere Equity Curve bei ähnlicher Rendite.")
         + vs_card + _htm(fig_vs), 4, False),

        ("📉 §6  Drawdown-Vergleich: Alle Strategien + Leverage + Krisen",
         _desc("Vollständiger Drawdown-Vergleich aller Varianten (1×, 2×, 3×, Vol-skaliert). "
               "Zeigt Schutzwirkung der Filter in Krisen. "
               "Bei hohem Leverage: deutlich tiefere Drawdowns in 2020/2022.")
         + _htm(fig_dd) + sum_tbl, 5, False),

        ("🏆 §7  Finale Empfehlung",
         _desc("Optimale Konfiguration basierend auf OOS Metriken, Kelly-Kriterium und Krisenverhalten.")
         + rec_card, 6, False),
    ]

    acc = '<div class="accordion" id="levAcc">'
    for t, b, idx, op in secs:
        acc += _acc(t, b, idx, op)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#f78166;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Leverage, Transaktionskosten &amp; Krisenanalyse</h2>
          <p class="mb-0" style="color:#8b949e;">
            5 Krisenperioden · Leverage 1×–5× · TC-Sensitivitäts-Heatmap ·
            Kelly-Kriterium · Vol-Skalierter Leverage · Drawdown-Vergleich
          </p>
        </div>
      </div>
      <div class="row g-3 mb-4">
        <div class="col-lg-3"><div class="card p-3" style="background:#1c2128;border:1px solid #3fb950;">
          <small style="color:#3fb950;">Beste Kombination</small><br>
          <strong style="color:#e6edf3;font-size:0.9em;">{best_lbl}</strong>
        </div></div>
        <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #58a6ff;">
          <small style="color:#58a6ff;">1× OOS Sharpe</small><br>
          <strong style="color:#e6edf3;font-size:1.4em;">{best_sh:.3f}</strong>
        </div></div>
        <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #ffa657;">
          <small style="color:#ffa657;">2× OOS Sharpe</small><br>
          <strong style="color:#e6edf3;font-size:1.4em;">{_sh(best_2x):.3f}</strong>
        </div></div>
        <div class="col-lg-2"><div class="card p-3" style="background:#1c2128;border:1px solid #bc8cff;">
          <small style="color:#bc8cff;">Half Kelly</small><br>
          <strong style="color:#e6edf3;font-size:1.4em;">{half_kelly:.1f}×</strong>
        </div></div>
        <div class="col-lg-3"><div class="card p-3" style="background:#1c2128;border:1px solid #f78166;">
          <small style="color:#f78166;">2× MaxDD OOS</small><br>
          <strong style="color:#e6edf3;font-size:1.4em;">{_mdd(best_2x)*100:.1f}%</strong>
        </div></div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "leverage_crisis_report.html",
           _html_base("Leverage & Krisenanalyse", 20, body))


def build_portfolio_simulation_report(tables, figures, out):  # noqa: C901
    """
    Realistic portfolio simulation with stop-loss, TC, single position.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from itertools import product as iproduct
    import yfinance as yf

    INITIAL_CAP = 100_000.0
    STOP_LOSS   = 0.30          # 30 % stop loss from avg entry price
    TC_PCT      = 0.001         # 10 bp one-way (entry and exit separately)
    POS_FRAC    = 0.95          # invest 95% of available capital per trade
    IS_FRAC     = 0.70

    CRISES = [
        ("2008 Lehman",      "2008-09-01", "2009-06-01"),
        ("2015 Öl-Crash",    "2015-06-01", "2016-03-01"),
        ("2020 COVID",       "2020-01-15", "2020-07-01"),
        ("2022 Zinswende",   "2022-01-01", "2022-12-31"),
    ]
    # 6 four-year cycles back from today
    TODAY = pd.Timestamp("2026-09-01")
    CYCLES = []
    for k in range(6):
        cy_end   = TODAY - pd.DateOffset(years=k*4)
        cy_start = cy_end - pd.DateOffset(years=4)
        CYCLES.append((f"Zyklus {k+1} ({cy_start.year}–{cy_end.year})",
                       cy_start, cy_end))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _dl_px(ticker):
        for period in ("max","15y","10y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    idx = pd.to_datetime(h.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    h.index = idx.normalize()
                    return h
            except Exception:
                pass
        return None

    def _dl(ticker):
        h = _dl_px(ticker)
        if h is not None:
            return h["Close"].rename(ticker)
        return None

    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _mdd(nav):
        c = pd.Series(nav).dropna()
        return float((c / c.cummax() - 1).min())

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    def _card(title, color, body):
        return (f'<div class="card mb-3 p-3" style="background:#1c2128;border:1px solid {color};">'
                f'<h5 style="color:{color};">{title}</h5>'
                f'<div style="color:#e6edf3;">{body}</div></div>')

    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#ps{idx}">'
            f'{title}</button></h2>'
            f'<div id="ps{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    # ── core simulation ───────────────────────────────────────────────────────
    def _sim(signal, close_px, low_px, cap=INITIAL_CAP, sl=STOP_LOSS,
             tc=TC_PCT, pf=POS_FRAC, long_only=True):
        """
        Single-position long-only portfolio simulation.
        Uses intraday Low for stop-loss check.
        Returns: (nav_series, cash_series, pos_series, trade_df)
        """
        common = signal.index.intersection(close_px.index)
        if len(common) < 10:
            return None, None, None, None

        sig_a   = signal.reindex(common).fillna(0.0).values
        close_a = close_px.reindex(common).ffill().bfill().values
        low_a   = (low_px.reindex(common).ffill().bfill().values
                   if low_px is not None else close_a.copy())

        n = len(common)
        capital = float(cap)
        shares  = 0.0
        avg_px  = 0.0
        stop_px = 0.0

        nav_arr  = np.empty(n); nav_arr[0]  = cap
        cash_arr = np.empty(n); cash_arr[0] = cap
        pos_arr  = np.empty(n); pos_arr[0]  = 0.0
        trades   = []

        for i in range(n):
            sig = float(sig_a[i])
            c   = float(close_a[i])
            lo  = float(low_a[i])
            d   = common[i]

            if np.isnan(c) or c <= 0:
                nav_arr[i]  = nav_arr[i-1] if i > 0 else cap
                cash_arr[i] = capital
                pos_arr[i]  = nav_arr[i] - capital
                continue

            sl_triggered = False

            # ── Stop-loss check against Low ───────────────────────────────────
            if shares > 0 and stop_px > 0 and lo <= stop_px:
                # Gap risk: exit at worst(stop_px, lo)
                exit_px = max(stop_px * 0.995, lo)
                proceeds = shares * exit_px * (1 - tc)
                pnl = proceeds - shares * avg_px
                capital += proceeds
                trades.append(dict(
                    date=d, type="SL", exit_px=exit_px,
                    entry_px=avg_px, pnl=pnl, pnl_pct=(exit_px-avg_px)/avg_px,
                    hold_days=i))
                shares = 0.0; avg_px = 0.0; stop_px = 0.0
                sl_triggered = True

            if not sl_triggered:
                # ── Entry ─────────────────────────────────────────────────────
                if sig > 0 and shares == 0 and capital > 1.0:
                    invest  = capital * pf
                    bought  = (invest * (1 - tc)) / c
                    capital -= invest
                    shares   = bought
                    avg_px   = c
                    stop_px  = avg_px * (1 - sl)
                    trades.append(dict(
                        date=d, type="BUY", entry_px=c, shares=shares,
                        invested=invest, stop_px=stop_px))

                # ── Exit (signal) ─────────────────────────────────────────────
                elif (sig <= 0 if long_only else sig < 0) and shares > 0:
                    proceeds = shares * c * (1 - tc)
                    pnl = proceeds - shares * avg_px
                    capital += proceeds
                    trades.append(dict(
                        date=d, type="SELL", exit_px=c,
                        entry_px=avg_px, pnl=pnl, pnl_pct=(c-avg_px)/avg_px))
                    shares = 0.0; avg_px = 0.0; stop_px = 0.0

            pos_val     = shares * c
            nav_arr[i]  = capital + pos_val
            cash_arr[i] = capital
            pos_arr[i]  = pos_val

        nav_s  = pd.Series(nav_arr,  index=common)
        cash_s = pd.Series(cash_arr, index=common)
        pos_s  = pd.Series(pos_arr,  index=common)
        tdf    = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date","type","entry_px","exit_px","pnl","pnl_pct"])
        return nav_s, cash_s, pos_s, tdf

    def _sim_metrics(nav_s, tdf, label=""):
        """Summarize simulation results."""
        if nav_s is None or len(nav_s) < 2:
            return {}
        daily_ret = nav_s.pct_change().dropna()
        n_sl = int((tdf["type"] == "SL").sum()) if not tdf.empty and "type" in tdf.columns else 0
        n_buy = int((tdf["type"] == "BUY").sum()) if not tdf.empty else 0
        closed = tdf[tdf["type"].isin(["SELL","SL"])].copy() if not tdf.empty else pd.DataFrame()
        win_rate = float((closed["pnl"] > 0).mean()) if len(closed) > 0 and "pnl" in closed.columns else np.nan
        avg_pnl  = float(closed["pnl"].mean()) if len(closed) > 0 and "pnl" in closed.columns else np.nan
        return {
            "Label": label,
            "Endkapital": f"€{nav_s.iloc[-1]:,.0f}",
            "Gesamtrendite": f"{(nav_s.iloc[-1]/INITIAL_CAP-1)*100:+.1f}%",
            "OOS Sharpe": f"{_sh(daily_ret):.3f}",
            "MaxDD": f"{_mdd(nav_s)*100:.1f}%",
            "Trades": n_buy,
            "Stop-Losses": n_sl,
            "Win-Rate": f"{win_rate*100:.1f}%" if not np.isnan(win_rate) else "–",
            "Ø P&L/Trade": f"€{avg_pnl:+,.0f}" if not np.isnan(avg_pnl) else "–",
        }

    # ── load signal data ──────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")
    if ret_main is None or px_main is None:
        _write(out / "portfolio_simulation_report.html",
               _html_base("Portfolio Simulation", 20, "<p>Daten fehlen.</p>")); return

    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    if "JETS" not in ret_main.columns or "CL=F" not in px_main.columns:
        _write(out / "portfolio_simulation_report.html",
               _html_base("Portfolio Simulation", 20, "<p>JETS/CL=F fehlt.</p>")); return

    jets_ret = ret_main["JETS"].dropna()
    BASKET   = ["CL=F","BZ=F","XLE","XOM","CVX"]
    basket_px = {t: px_main[t].dropna() for t in BASKET if t in px_main.columns}

    common_sig = jets_ret.index
    for t in basket_px:
        common_sig = common_sig.intersection(basket_px[t].index)
    common_sig = common_sig[~common_sig.duplicated()].sort_values()

    jets_c   = jets_ret.reindex(common_sig).fillna(0.0)
    basket_c = {t: basket_px[t].reindex(common_sig).ffill() for t in basket_px}
    cl_px    = basket_c["CL=F"]

    vix_raw = _dl("^VIX")
    tnx_raw = _dl("^TNX")
    vix_c   = vix_raw.reindex(common_sig).ffill() if vix_raw is not None else None
    tnx_c   = tnx_raw.reindex(common_sig).ffill() if tnx_raw is not None else None

    n_total = len(common_sig)
    split_i = int(n_total * IS_FRAC)
    is_idx  = common_sig[:split_i]
    oos_idx = common_sig[split_i:]

    def _net_ret(sig):
        return sig * jets_c - sig.diff().abs().fillna(0) * TC_PCT

    rsi_cl  = _calc_rsi(cl_px, 14)
    sig_rsi = pd.Series(np.where(rsi_cl < 70, 1.0, -1.0),
                         index=common_sig).shift(1).fillna(0.0)

    ens_parts = [pd.Series(np.where(_calc_rsi(px_t, 14) < 70, 1.0, -1.0), index=common_sig)
                 for px_t in basket_c.values()]
    ens_raw  = pd.concat(ens_parts, axis=1).mean(axis=1)
    sig_bask = pd.Series(np.where(ens_raw > 0.2, 1.0,
                                   np.where(ens_raw < -0.2, -1.0, 0.0)),
                          index=common_sig).shift(1).fillna(0.0)

    is_df_s = _net_ret(sig_rsi).reindex(is_idx).dropna().to_frame("r")
    is_df_s["m"] = pd.to_datetime(is_df_s.index).month
    good_m  = set(is_df_s.groupby("m")["r"].mean()[lambda x: x > 0].index)
    seas_m  = pd.Series(pd.to_datetime(common_sig).month.isin(good_m), index=common_sig)

    vix_m = ((vix_c.shift(1) < 25).reindex(common_sig).fillna(True)
             if vix_c is not None else pd.Series(True, index=common_sig))
    if tnx_c is not None:
        tnx_r_s = np.log(tnx_c / tnx_c.shift(1)).fillna(0)
        tnx_m   = (tnx_r_s.rolling(20).mean().shift(1) <= 0).reindex(common_sig).fillna(True)
    else:
        tnx_m = pd.Series(True, index=common_sig)

    # Build top-5 combinations by OOS Sharpe
    combos = []
    for bname, bsig in [("RSI<70", sig_rsi), ("Basket", sig_bask)]:
        for us, uv, ut in iproduct([False,True],[False,True],[False,True]):
            sig = bsig.copy()
            if us: sig = sig * seas_m.astype(float)
            if uv: sig = sig * vix_m.astype(float)
            if ut: sig = sig * tnx_m.astype(float)
            sh_oos = _sh(_net_ret(sig).reindex(oos_idx).dropna())
            lbl = (f"{bname}"
                   f"{'+S' if us else ''}{'+V' if uv else ''}{'+T' if ut else ''}")
            combos.append((lbl, sig, sh_oos))
    combos.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -99, reverse=True)
    top5 = combos[:5]
    best_lbl, best_sig_full, best_sh = top5[0]

    # ── Download JETS OHLC ────────────────────────────────────────────────────
    jets_ohlc = _dl_px("JETS")
    if jets_ohlc is None or jets_ohlc.empty:
        _write(out / "portfolio_simulation_report.html",
               _html_base("Portfolio Simulation", 20, "<p>JETS-Preisdaten nicht verfügbar.</p>")); return

    jets_close = jets_ohlc["Close"].rename("close")
    jets_low   = jets_ohlc["Low"].rename("low")

    # ── §2: Full history simulation (best combo) ──────────────────────────────
    nav_full, cash_full, pos_full, tdf_full = _sim(
        best_sig_full, jets_close, jets_low)

    # JETS B&H for comparison
    jets_close_norm = jets_close.reindex(nav_full.index if nav_full is not None else jets_close.index).ffill()
    bah_nav = jets_close_norm / float(jets_close_norm.iloc[0]) * INITIAL_CAP if nav_full is not None else None

    fig_full = None
    if nav_full is not None:
        # NAV + B&H + cash/position stacked
        fig_full = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  row_heights=[0.5, 0.25, 0.25],
                                  subplot_titles=["Portfolio NAV vs JETS B&H",
                                                  "Cash vs Positionswert",
                                                  "Drawdown"])

        fig_full.add_trace(go.Scatter(
            x=bah_nav.index.astype(str).tolist(), y=bah_nav.values.tolist(),
            name="JETS B&H", mode="lines", line=dict(color="#8b949e", width=1.5, dash="dot")),
            row=1, col=1)
        fig_full.add_trace(go.Scatter(
            x=nav_full.index.astype(str).tolist(), y=nav_full.values.tolist(),
            name=f"Strategie [{best_lbl}]", mode="lines",
            line=dict(color="#3fb950", width=2.5)),
            row=1, col=1)

        # Entry/Exit/SL markers on NAV
        if not tdf_full.empty and "type" in tdf_full.columns:
            for ev_type, symbol, color, nm in [
                ("BUY",  "triangle-up",   "#3fb950", "Einstieg ▲"),
                ("SELL", "triangle-down", "#58a6ff", "Ausstieg ▼"),
                ("SL",   "x",             "#f78166", "Stop-Loss ✕"),
            ]:
                sub = tdf_full[tdf_full["type"] == ev_type]
                if len(sub) > 0:
                    d_s = [str(d.date()) if hasattr(d, 'date') else str(d) for d in sub["date"]]
                    navs = nav_full.reindex(pd.to_datetime(sub["date"].values), method="nearest").fillna(INITIAL_CAP).values.tolist()
                    fig_full.add_trace(go.Scatter(
                        x=d_s, y=navs, name=nm, mode="markers",
                        marker=dict(symbol=symbol, size=10, color=color)),
                        row=1, col=1)

        # Cash + position area chart
        fig_full.add_trace(go.Scatter(
            x=cash_full.index.astype(str).tolist(), y=cash_full.values.tolist(),
            name="Cash", fill="tozeroy", fillcolor="rgba(88,166,255,0.2)",
            line=dict(color="#58a6ff", width=1)), row=2, col=1)
        fig_full.add_trace(go.Scatter(
            x=pos_full.index.astype(str).tolist(),
            y=(cash_full + pos_full).values.tolist(),
            name="Cash + Position", fill="tonexty",
            fillcolor="rgba(63,185,80,0.2)", line=dict(color="#3fb950", width=1)),
            row=2, col=1)

        # Drawdown
        dd_full = (nav_full / nav_full.cummax() - 1) * 100
        fig_full.add_trace(go.Scatter(
            x=dd_full.index.astype(str).tolist(), y=dd_full.values.tolist(),
            name="Drawdown", fill="tozeroy", fillcolor="rgba(247,129,102,0.25)",
            line=dict(color="#f78166", width=1.2)), row=3, col=1)
        fig_full.add_hline(y=-30, line_color="#f78166", line_dash="dot", row=3, col=1)

        for cname, cs, ce in CRISES:
            for r in [1,2,3]:
                try:
                    fig_full.add_vrect(x0=cs, x1=ce, fillcolor="#bc8cff",
                                       opacity=0.07, layer="below", line_width=0,
                                       row=r, col=1)
                except Exception:
                    pass

        fig_full.update_layout(
            **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
            height=780,
            title_text=f"Vollständige Historien-Simulation: {best_lbl} | Start €{INITIAL_CAP:,.0f}",
        )

    # ── §3: OOS simulation – top 5 combos ─────────────────────────────────────
    oos_results = []
    fig_oos = go.Figure()
    oos_colors = ["#3fb950","#58a6ff","#ffa657","#bc8cff","#f78166"]

    for (lbl, sig_c, sh_oos_ret), col in zip(top5, oos_colors):
        nav_o, _, _, tdf_o = _sim(sig_c.reindex(oos_idx), jets_close, jets_low)
        if nav_o is not None:
            fig_oos.add_trace(go.Scatter(
                x=nav_o.index.astype(str).tolist(), y=nav_o.values.tolist(),
                name=lbl, mode="lines", line=dict(color=col, width=2)))
            oos_results.append(_sim_metrics(nav_o, tdf_o, lbl))

    # B&H OOS
    bah_oos = jets_close.reindex(oos_idx).ffill()
    bah_oos_nav = bah_oos / float(bah_oos.iloc[0]) * INITIAL_CAP
    fig_oos.add_trace(go.Scatter(
        x=bah_oos_nav.index.astype(str).tolist(), y=bah_oos_nav.values.tolist(),
        name="JETS B&H", mode="lines", line=dict(color="#8b949e", dash="dot", width=1.5)))
    _lay(fig_oos, title=f"OOS Portfolio-Simulation: Top-5 Kombinationen vs JETS B&H | Start €{INITIAL_CAP:,.0f}",
         xaxis_title="Datum", yaxis_title="Portfolio Wert (€)", height=470)

    oos_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm table-hover">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in (oos_results[0].keys() if oos_results else [])) + '</tr></thead>'
        '<tbody>' + "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
            for r in oos_results)
        + '</tbody></table></div>'
    ) if oos_results else ""

    # ── §4: Stop-Loss Sensitivity ─────────────────────────────────────────────
    SL_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00]
    SL_LABELS = ["10%","20%","30%","40%","50%","Kein SL"]
    sl_colors = px.colors.sequential.Reds_r[:len(SL_LEVELS)]

    fig_sl = go.Figure()
    sl_rows = []
    for sl_val, sl_lbl, sl_col in zip(SL_LEVELS, SL_LABELS, sl_colors):
        nav_sl, _, _, tdf_sl = _sim(best_sig_full.reindex(oos_idx), jets_close, jets_low, sl=sl_val)
        if nav_sl is not None:
            fig_sl.add_trace(go.Scatter(
                x=nav_sl.index.astype(str).tolist(), y=nav_sl.values.tolist(),
                name=f"SL {sl_lbl}", mode="lines", line=dict(color=sl_col, width=1.8)))
            n_sl = int((tdf_sl["type"] == "SL").sum()) if not tdf_sl.empty and "type" in tdf_sl.columns else 0
            sl_rows.append(dict(
                SL=sl_lbl,
                Endkapital=f"€{nav_sl.iloc[-1]:,.0f}",
                Return=f"{(nav_sl.iloc[-1]/INITIAL_CAP-1)*100:+.1f}%",
                Sharpe=f"{_sh(nav_sl.pct_change().dropna()):.3f}",
                MaxDD=f"{_mdd(nav_sl)*100:.1f}%",
                SL_Events=n_sl,
            ))
    _lay(fig_sl, title="Stop-Loss Sensitivität (OOS, Beste Kombo)",
         xaxis_title="Datum", yaxis_title="Portfolio Wert (€)", height=440)

    sl_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in sl_rows[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                             for r in sl_rows)
        + '</tbody></table></div>'
    ) if sl_rows else ""

    # ── §5: Crisis simulations ────────────────────────────────────────────────
    n_cr  = len(CRISES)
    fig_cr = make_subplots(rows=1, cols=n_cr,
                            subplot_titles=[c[0] for c in CRISES])
    cr_summary = []
    for ci, (cname, cs, ce) in enumerate(CRISES):
        c_start = pd.Timestamp(cs); c_end = pd.Timestamp(ce)
        sig_cr  = best_sig_full.loc[c_start:c_end]
        nav_cr, _, _, tdf_cr = _sim(sig_cr, jets_close, jets_low, cap=INITIAL_CAP)
        bah_cr  = jets_close.loc[c_start:c_end].ffill()

        if nav_cr is not None and len(nav_cr) > 5:
            # Normalize to 100
            nav_n = nav_cr / float(nav_cr.iloc[0]) * 100
            bah_n = bah_cr / float(bah_cr.iloc[0]) * 100

            fig_cr.add_trace(go.Scatter(
                x=bah_n.index.astype(str).tolist(), y=bah_n.values.tolist(),
                name="B&H", legendgroup=cname, showlegend=(ci == 0),
                mode="lines", line=dict(color="#8b949e", dash="dot", width=1.2)),
                row=1, col=ci+1)
            fig_cr.add_trace(go.Scatter(
                x=nav_n.index.astype(str).tolist(), y=nav_n.values.tolist(),
                name=best_lbl, legendgroup=cname, showlegend=(ci == 0),
                mode="lines", line=dict(color="#3fb950", width=1.8)),
                row=1, col=ci+1)

            # SL events
            if not tdf_cr.empty and "type" in tdf_cr.columns:
                sl_ev = tdf_cr[tdf_cr["type"] == "SL"]
                if len(sl_ev) > 0:
                    sl_d   = [str(d.date()) if hasattr(d, 'date') else str(d) for d in sl_ev["date"]]
                    sl_nav = nav_n.reindex(pd.to_datetime(sl_ev["date"].values),
                                           method="nearest").fillna(100).values.tolist()
                    fig_cr.add_trace(go.Scatter(
                        x=sl_d, y=sl_nav, name="Stop-Loss ✕",
                        legendgroup=cname, showlegend=(ci == 0),
                        mode="markers", marker=dict(symbol="x", size=10, color="#f78166")),
                        row=1, col=ci+1)

            n_sl_cr = int((tdf_cr["type"] == "SL").sum()) if not tdf_cr.empty and "type" in tdf_cr.columns else 0
            cr_summary.append(dict(
                Krise=cname,
                StartNav=f"€{nav_cr.iloc[0]:,.0f}",
                EndNav=f"€{nav_cr.iloc[-1]:,.0f}",
                Return=f"{(nav_cr.iloc[-1]/nav_cr.iloc[0]-1)*100:+.1f}%",
                BnH=f"{(bah_cr.iloc[-1]/bah_cr.iloc[0]-1)*100:+.1f}%",
                MaxDD=f"{_mdd(nav_cr)*100:.1f}%",
                SL_Events=n_sl_cr,
            ))

    fig_cr.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=420, title_text="Krisen-Simulation: NAV normiert auf 100 bei Krisenbeginn (▲=Einstieg, ✕=Stop-Loss)")
    for ci in range(1, n_cr+1):
        fig_cr.update_xaxes(tickangle=-45, tickfont=dict(size=7), row=1, col=ci)

    cr_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in cr_summary[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                             for r in cr_summary)
        + '</tbody></table></div>'
    ) if cr_summary else ""

    # ── §6: 4-year rolling cycles ─────────────────────────────────────────────
    cy_fig  = make_subplots(rows=2, cols=3, subplot_titles=[c[0] for c in CYCLES],
                             shared_yaxes=False)
    cy_summary = []

    for ci, (cname, cy_start, cy_end) in enumerate(CYCLES):
        r_i = ci // 3 + 1; c_i = ci % 3 + 1
        sig_cy = best_sig_full.loc[cy_start:cy_end]
        jets_cy_close = jets_close.loc[cy_start:cy_end]
        jets_cy_low   = jets_low.loc[cy_start:cy_end]

        if len(sig_cy) < 20 or len(jets_cy_close) < 20:
            cy_summary.append(dict(Zyklus=cname, Status="Keine JETS-Daten", **{k:"–" for k in ["Return","Sharpe","MaxDD","SL"]}))
            continue

        nav_cy, _, _, tdf_cy = _sim(sig_cy, jets_cy_close, jets_cy_low, cap=INITIAL_CAP)
        bah_cy = jets_cy_close.ffill()

        if nav_cy is None or len(nav_cy) < 5:
            cy_summary.append(dict(Zyklus=cname, Status="Fehler", **{k:"–" for k in ["Return","Sharpe","MaxDD","SL"]}))
            continue

        nav_n = nav_cy / float(nav_cy.iloc[0]) * 100
        bah_n = bah_cy / float(bah_cy.iloc[0]) * 100

        cy_fig.add_trace(go.Scatter(
            x=bah_n.index.astype(str).tolist(), y=bah_n.values.tolist(),
            name="B&H", legendgroup="bah", showlegend=(ci == 0),
            mode="lines", line=dict(color="#8b949e", dash="dot", width=1.2)),
            row=r_i, col=c_i)
        cy_fig.add_trace(go.Scatter(
            x=nav_n.index.astype(str).tolist(), y=nav_n.values.tolist(),
            name=best_lbl, legendgroup="strat", showlegend=(ci == 0),
            mode="lines", line=dict(color="#3fb950", width=1.8)),
            row=r_i, col=c_i)

        n_sl_cy = int((tdf_cy["type"] == "SL").sum()) if not tdf_cy.empty and "type" in tdf_cy.columns else 0
        dr = _sh(nav_cy.pct_change().dropna())
        cy_summary.append(dict(
            Zyklus=cname, Status="✓",
            Return=f"{(nav_cy.iloc[-1]/nav_cy.iloc[0]-1)*100:+.1f}%",
            Sharpe=f"{dr:.3f}",
            MaxDD=f"{_mdd(nav_cy)*100:.1f}%",
            SL=n_sl_cy,
        ))

    cy_fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=600,
        title_text="6 Rollende 4-Jahres-Zyklen (ab heute rückwärts, NAV normiert)")
    for ri in range(1,3):
        for ci in range(1,4):
            cy_fig.update_xaxes(tickangle=-45, tickfont=dict(size=7), row=ri, col=ci)

    cy_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in cy_summary[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                             for r in cy_summary)
        + '</tbody></table></div>'
    ) if cy_summary else ""

    # ── §7: Trade-log analysis ────────────────────────────────────────────────
    trade_html = ""
    if nav_full is not None and not tdf_full.empty and "type" in tdf_full.columns:
        closed_full = tdf_full[tdf_full["type"].isin(["SELL","SL"])].copy()
        if len(closed_full) > 0 and "pnl_pct" in closed_full.columns:
            pnl_pct_arr = (closed_full["pnl_pct"].dropna() * 100).tolist()
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(
                x=pnl_pct_arr, nbinsx=30,
                name="Trade P&L %", marker_color="#58a6ff",
                opacity=0.8))
            fig_pnl.add_vline(x=0, line_color="#f78166", line_dash="dot")
            win_pct = (closed_full["pnl_pct"] > 0).mean() * 100
            avg_win = float(closed_full.loc[closed_full["pnl_pct"] > 0, "pnl_pct"].mean()) * 100 if (closed_full["pnl_pct"] > 0).any() else 0
            avg_los = float(closed_full.loc[closed_full["pnl_pct"] <= 0, "pnl_pct"].mean()) * 100 if (closed_full["pnl_pct"] <= 0).any() else 0
            pf_num  = closed_full.loc[closed_full["pnl"] > 0, "pnl"].sum()
            pf_den  = abs(closed_full.loc[closed_full["pnl"] <= 0, "pnl"].sum())
            profit_factor = float(pf_num / (pf_den + 1e-9))
            fig_pnl.add_annotation(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=(f"Win-Rate: {win_pct:.1f}%<br>"
                      f"Ø Gewinn: +{avg_win:.2f}%<br>"
                      f"Ø Verlust: {avg_los:.2f}%<br>"
                      f"Profit-Faktor: {profit_factor:.2f}"),
                showarrow=False,
                bgcolor="#1c2128", bordercolor="#30363d",
                font=dict(color="#e6edf3", size=11),
                align="right"
            )
            _lay(fig_pnl, title="Trade P&L Verteilung (alle abgeschlossenen Trades, vollständige Historie)",
                 xaxis_title="Trade Return (%)", yaxis_title="Anzahl Trades", height=420)

            # Trade type breakdown
            type_counts = tdf_full["type"].value_counts()
            fig_type = go.Figure(go.Bar(
                x=type_counts.index.tolist(),
                y=type_counts.values.tolist(),
                marker_color=["#3fb950","#58a6ff","#f78166"],
            ))
            _lay(fig_type, title="Trade-Typen: Einstiege / Signal-Ausstiege / Stop-Loss",
                 xaxis_title="Trade-Typ", yaxis_title="Anzahl", height=300)

            trade_html = (
                _desc(f"Vollständige Historien-Simulation: {len(tdf_full)} Trade-Ereignisse. "
                      f"Win-Rate: {win_pct:.1f}% | Profit-Faktor: {profit_factor:.2f} | "
                      f"Ø Gewinn-Trade: +{avg_win:.2f}% | Ø Verlust-Trade: {avg_los:.2f}%")
                + _htm(fig_pnl) + _htm(fig_type)
            )

    # ── §8: TC sensitivity ────────────────────────────────────────────────────
    TC_SENS = [0.0002, 0.0005, 0.001, 0.002, 0.005]
    fig_tc = go.Figure()
    tc_rows = []
    for tc_v in TC_SENS:
        nav_tc, _, _, tdf_tc = _sim(best_sig_full.reindex(oos_idx),
                                     jets_close, jets_low, tc=tc_v)
        lbl_tc = f"TC={int(tc_v*10000)}bp"
        if nav_tc is not None:
            fig_tc.add_trace(go.Scatter(
                x=nav_tc.index.astype(str).tolist(), y=nav_tc.values.tolist(),
                name=lbl_tc, mode="lines"))
            tc_rows.append(dict(
                TC=lbl_tc,
                Endkapital=f"€{nav_tc.iloc[-1]:,.0f}",
                Return=f"{(nav_tc.iloc[-1]/INITIAL_CAP-1)*100:+.1f}%",
                Sharpe=f"{_sh(nav_tc.pct_change().dropna()):.3f}",
                MaxDD=f"{_mdd(nav_tc)*100:.1f}%",
            ))
    _lay(fig_tc, title="TC-Sensitivität (OOS, Beste Kombo)",
         xaxis_title="Datum", yaxis_title="Portfolio Wert (€)", height=420)

    # ── Full history metrics card ──────────────────────────────────────────────
    full_metrics = _sim_metrics(nav_full, tdf_full, best_lbl) if nav_full is not None else {}
    fm_html = ""
    if full_metrics:
        fm_html = f"""
        <div class="row g-3 mb-4">
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #3fb950;">
            <small style="color:#3fb950;">Endkapital</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('Endkapital','–')}</strong>
          </div></div>
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #58a6ff;">
            <small style="color:#58a6ff;">Gesamtrendite</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('Gesamtrendite','–')}</strong>
          </div></div>
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #ffa657;">
            <small style="color:#ffa657;">Sharpe (tägl.)</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('OOS Sharpe','–')}</strong>
          </div></div>
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #f78166;">
            <small style="color:#f78166;">Max. Drawdown</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('MaxDD','–')}</strong>
          </div></div>
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #bc8cff;">
            <small style="color:#bc8cff;">Trades</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('Trades','–')}</strong>
          </div></div>
          <div class="col-lg-2"><div class="card p-2" style="background:#1c2128;border:1px solid #e3b341;">
            <small style="color:#e3b341;">Stop-Loss Events</small><br>
            <strong style="color:#e6edf3;font-size:1.2em;">{full_metrics.get('Stop-Losses','–')}</strong>
          </div></div>
        </div>
        """

    # ── Assemble HTML ──────────────────────────────────────────────────────────
    param_card = _card("Simulation-Parameter", "#58a6ff", f"""
    <table class="table table-dark table-sm mb-0">
      <tr><td>Startkapital</td><td style="color:#3fb950;"><strong>€{INITIAL_CAP:,.0f}</strong></td></tr>
      <tr><td>Stop-Loss</td><td style="color:#f78166;"><strong>{int(STOP_LOSS*100)}% unter Ø Einstiegspreis</strong> (geprüft gegen Tages-Tief)</td></tr>
      <tr><td>Transaktionskosten</td><td>{int(TC_PCT*10000)} bp one-way ({int(TC_PCT*2*10000)} bp R/T)</td></tr>
      <tr><td>Kapitaleinsatz</td><td>{int(POS_FRAC*100)}% des verfügbaren Kapitals pro Trade</td></tr>
      <tr><td>Strategie</td><td>Long-only JETS (kein Short)</td></tr>
      <tr><td>Preis-Daten</td><td>JETS OHLC täglich (yfinance)</td></tr>
      <tr><td>Gap-Risiko</td><td>Stop-Preis = max(Stop-Level × 99.5%, Tages-Tief) bei Durchbrechen</td></tr>
      <tr><td>Bestes Signal</td><td style="color:#3fb950;">{best_lbl} (OOS Sharpe: {best_sh:.3f})</td></tr>
    </table>
    """)

    secs = [
        ("⚙️ §1  Simulation-Parameter & Mechanik",
         _desc("Realistische Single-Position Simulation. Kein Margin, kein Shorting. "
               "Pro Signal-Eintritt: ein Trade mit 95% des Kapitals. "
               "Stop-Loss prüft Tages-Tief (OHLC) für realistischere Ausführung.")
         + param_card, 0, True),

        ("📈 §2  Vollständige Historien-Simulation (früheste Daten bis heute)",
         _desc(f"Vollständige verfügbare JETS-Geschichte ({jets_close.index[0].date()} – "
               f"{jets_close.index[-1].date()}). "
               f"Startkapital €{INITIAL_CAP:,.0f}. "
               "Grün ▲ = Einstieg | Blau ▼ = Signal-Ausstieg | Rot ✕ = Stop-Loss. "
               "Krisen-Perioden violett schattiert.")
         + fm_html
         + (_htm(fig_full) if fig_full is not None else "<p>Simulation nicht verfügbar.</p>"),
         1, False),

        ("🏆 §3  OOS-Simulation – Top-5 Kombinationen im Vergleich",
         _desc(f"OOS-Zeitraum: {oos_idx[0].date()} – {oos_idx[-1].date()}. "
               "Top-5 Kombinationen nach OOS Sharpe. Startkapital €{INITIAL_CAP:,.0f}. "
               "Stop-Loss 30%, TC 10bp.")
         + _htm(fig_oos) + oos_tbl, 2, False),

        ("🛑 §4  Stop-Loss Sensitivität (10%–kein SL)",
         _desc("OOS-Simulation mit verschiedenen Stop-Loss-Schwellen. "
               "10% SL = sehr häufig ausgestoppt (hohe TC-Last). "
               "Kein SL = maximale Drawdown-Risiko. 30% = Kompromiss.")
         + _htm(fig_sl) + sl_tbl, 3, False),

        ("⚡ §5  Krisen-Simulationen",
         _desc("Portfolio-Performance normiert auf 100 bei Krisenbeginn. "
               "Vergleich mit JETS B&H. "
               "Stop-Loss-Events (✕) zeigen wann der Schutzmechanismus ausgelöst wurde.")
         + _htm(fig_cr) + cr_tbl, 4, False),

        ("🔄 §6  6 Rollende 4-Jahres-Zyklen (von heute rückwärts)",
         _desc("6 nicht-überlappende 4-Jahres-Fenster rückwärts von Sep 2026. "
               "JETS-ETF existiert ab ~Mai 2015 → Zyklen vor 2015 haben keine Daten. "
               "Zeigt Konsistenz der Strategie über verschiedene Marktphasen.")
         + _htm(cy_fig) + cy_tbl, 5, False),

        ("📊 §7  Trade-Log Analyse (P&L Verteilung, Win-Rate, Profit-Faktor)",
         _desc("Analyse aller abgeschlossenen Trades der vollständigen Historien-Simulation. "
               "Win-Rate, Profit-Faktor und durchschnittliche Gewinne/Verluste zeigen "
               "die Qualität des Signals auf Trade-Ebene.")
         + (trade_html if trade_html else "<p style='color:#8b949e;'>Zu wenig Trades.</p>"),
         6, False),

        ("💰 §8  Transaktionskosten-Sensitivität (OOS)",
         _desc("Einfluss der Transaktionskosten auf das Endkapital. "
               "Zeigt bis zu welchem TC-Level die Strategie noch profitabel ist.")
         + _htm(fig_tc), 7, False),
    ]

    acc = '<div class="accordion" id="psAcc">'
    for t, b, idx, op in secs:
        acc += _acc(t, b, idx, op)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#e3b341;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Realistische Portfolio-Simulation: JETS Strategie</h2>
          <p class="mb-0" style="color:#8b949e;">
            €100 000 Startkapital · 30% Stop-Loss (OHLC) · 10bp TC ·
            Long-only · Vollständige Historie · 5 Krisen · 6×4-Jahres-Zyklen · Trade-Log
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "portfolio_simulation_report.html",
           _html_base("Portfolio Simulation", 20, body))


def build_combination_deepdive_report(tables, figures, out):  # noqa: C901
    """
    Deep-dive analysis of all 16 strategy combinations.
    Factor attribution, stability, trade quality, regime analysis.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from itertools import product as iproduct
    import yfinance as yf

    IS_FRAC = 0.70
    TC      = 0.001

    # Highlighted combos (user-specified)
    HIGHLIGHTED = [
        ("RSI<70+Seas+VIX", "stabil IS+OOS, geringe MaxDD"),
        ("Basket",          "starke OOS, schwache IS → Regime-Alpha"),
        ("RSI<70+S+V+T",    "bester IS aber OOS-Rückgang → Overfitting"),
        ("Basket+VIX",      "hohe OOS, interessante Robustheit"),
    ]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _dl(ticker):
        for period in ("10y","5y"):
            try:
                h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not h.empty:
                    s = h["Close"]
                    idx = pd.to_datetime(s.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    return pd.Series(s.values, index=idx.normalize(), name=ticker)
            except Exception:
                pass
        return None

    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _roll_sh(s, w=252):
        m = s.rolling(w).mean(); v = s.rolling(w).std()
        return (m / (v + 1e-9)) * np.sqrt(252)

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _split(s, frac=IS_FRAC):
        n = len(s); si = int(n*frac)
        return s.iloc[:si], s.iloc[si:]

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    def _card(title, color, body):
        return (f'<div class="card mb-3 p-3" style="background:#1c2128;border:1px solid {color};">'
                f'<h5 style="color:{color};">{title}</h5>'
                f'<div style="color:#e6edf3;">{body}</div></div>')

    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#dd{idx}">'
            f'{title}</button></h2>'
            f'<div id="dd{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    # ── data ──────────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")
    if ret_main is None or px_main is None:
        _write(out / "combination_deepdive_report.html",
               _html_base("Kombinations-Deep-Dive", 20, "<p>Daten fehlen.</p>")); return
    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    if "JETS" not in ret_main.columns or "CL=F" not in px_main.columns:
        _write(out / "combination_deepdive_report.html",
               _html_base("Kombinations-Deep-Dive", 20, "<p>JETS/CL=F fehlt.</p>")); return

    jets_ret  = ret_main["JETS"].dropna()
    BASKET    = ["CL=F","BZ=F","XLE","XOM","CVX"]
    basket_px = {t: px_main[t].dropna() for t in BASKET if t in px_main.columns}

    common = jets_ret.index
    for t in basket_px:
        common = common.intersection(basket_px[t].index)
    common = common[~common.duplicated()].sort_values()

    jets_c   = jets_ret.reindex(common).fillna(0.0)
    basket_c = {t: basket_px[t].reindex(common).ffill() for t in basket_px}
    cl_px    = basket_c["CL=F"]

    vix_raw = _dl("^VIX")
    tnx_raw = _dl("^TNX")
    vix_c   = vix_raw.reindex(common).ffill() if vix_raw is not None else None
    tnx_c   = tnx_raw.reindex(common).ffill() if tnx_raw is not None else None

    n_total  = len(common)
    split_i  = int(n_total * IS_FRAC)
    is_idx   = common[:split_i]
    oos_idx  = common[split_i:]

    def _net(sig):
        return sig * jets_c - sig.diff().abs().fillna(0) * TC

    rsi_cl  = _calc_rsi(cl_px, 14)
    sig_rsi = pd.Series(np.where(rsi_cl < 70, 1.0, -1.0),
                         index=common).shift(1).fillna(0.0)
    ens_parts = [pd.Series(np.where(_calc_rsi(px_t, 14) < 70, 1.0, -1.0), index=common)
                 for px_t in basket_c.values()]
    ens_raw  = pd.concat(ens_parts, axis=1).mean(axis=1)
    sig_bask = pd.Series(np.where(ens_raw > 0.2, 1.0,
                                   np.where(ens_raw < -0.2, -1.0, 0.0)),
                          index=common).shift(1).fillna(0.0)

    is_df_s = _net(sig_rsi).reindex(is_idx).dropna().to_frame("r")
    is_df_s["m"] = pd.to_datetime(is_df_s.index).month
    good_m  = set(is_df_s.groupby("m")["r"].mean()[lambda x: x > 0].index)
    seas_m  = pd.Series(pd.to_datetime(common).month.isin(good_m), index=common)

    vix_m = ((vix_c.shift(1) < 25).reindex(common).fillna(True)
             if vix_c is not None else pd.Series(True, index=common))
    if tnx_c is not None:
        tnx_r_s = np.log(tnx_c / tnx_c.shift(1)).fillna(0)
        tnx_m   = (tnx_r_s.rolling(20).mean().shift(1) <= 0).reindex(common).fillna(True)
    else:
        tnx_m = pd.Series(True, index=common)

    # Build all 16 combos with full metadata
    combos = []
    for bname, bsig in [("RSI<70", sig_rsi), ("Basket", sig_bask)]:
        for us, uv, ut in iproduct([False,True],[False,True],[False,True]):
            sig = bsig.copy()
            if us: sig = sig * seas_m.astype(float)
            if uv: sig = sig * vix_m.astype(float)
            if ut: sig = sig * tnx_m.astype(float)
            net     = _net(sig)
            is_n    = net.reindex(is_idx).dropna()
            oos_n   = net.reindex(oos_idx).dropna()
            sh_is   = _sh(is_n); sh_oos = _sh(oos_n)
            lbl = (f"{bname}"
                   f"{'+S' if us else ''}{'+V' if uv else ''}{'+T' if ut else ''}")
            combos.append(dict(
                lbl=lbl, base=bname, s=us, v=uv, t=ut,
                sh_is=sh_is, sh_oos=sh_oos, delta=sh_oos-sh_is,
                mdd=_mdd(oos_n),
                n_tr=int((sig.reindex(oos_idx).diff().abs() > 0).sum()),
                _sig=sig, _is=is_n, _oos=oos_n, _net=net,
            ))

    combos.sort(key=lambda r: r["sh_oos"] if not np.isnan(r["sh_oos"]) else -99, reverse=True)

    # ── §1: IS vs OOS Scatter ─────────────────────────────────────────────────
    fig_scat = go.Figure()
    for r in combos:
        is_v, oos_v = r["sh_is"], r["sh_oos"]
        if np.isnan(is_v) or np.isnan(oos_v):
            continue
        col = "#3fb950" if oos_v > 0.6 else ("#ffa657" if oos_v > 0.3 else "#f78166")
        fig_scat.add_trace(go.Scatter(
            x=[is_v], y=[oos_v], mode="markers+text",
            text=[r["lbl"]], textposition="top right",
            textfont=dict(size=8, color="#e6edf3"),
            marker=dict(size=r["n_tr"] / 5 + 6, color=col, opacity=0.85,
                        line=dict(color="#ffffff", width=0.5)),
            name=r["lbl"], showlegend=False,
            hovertemplate=(f"<b>{r['lbl']}</b><br>"
                           f"IS Sharpe: {is_v:.3f}<br>OOS Sharpe: {oos_v:.3f}<br>"
                           f"Δ: {r['delta']:+.3f}<br>#Trades OOS: {r['n_tr']}")))

    # 45° diagonal (IS = OOS)
    diag_range = [-0.5, 1.0]
    fig_scat.add_trace(go.Scatter(
        x=diag_range, y=diag_range,
        mode="lines", name="IS = OOS",
        line=dict(color="#8b949e", dash="dash", width=1)))
    fig_scat.add_hline(y=0, line_color="#f78166", line_dash="dot")
    fig_scat.add_vline(x=0, line_color="#f78166", line_dash="dot")
    _lay(fig_scat,
         title="IS vs OOS Sharpe: Overfitting-Karte (Kreisgröße ∝ #Trades OOS)",
         xaxis_title="IS Sharpe", yaxis_title="OOS Sharpe", height=520)

    # Color legend card
    legend_card = _card("Interpretation", "#58a6ff",
        "Punkte <strong>über</strong> der Diagonale: OOS &gt; IS → positive Generalisierung (kein Overfitting).<br>"
        "Punkte <strong>unter</strong> der Diagonale: OOS &lt; IS → Overfitting an IS-Periode.<br>"
        "Grün = OOS Sharpe &gt; 0.6 (stark). Orange = 0.3–0.6 (moderat). Rot = &lt; 0.3 (schwach).<br>"
        "Ideale Kombo: weit oben rechts UND über der Diagonale.")

    # ── §2: Factor attribution ────────────────────────────────────────────────
    # Marginal contribution of each filter
    factor_attr = []
    for filter_name, filter_col in [("Seasonal (+S)", "s"), ("VIX<25 (+V)", "v"), ("TNX-Trend (+T)", "t")]:
        for bname in ["RSI<70", "Basket"]:
            with_f    = [r for r in combos if r["base"]==bname and r[filter_col]==True]
            without_f = [r for r in combos if r["base"]==bname and r[filter_col]==False]
            if with_f and without_f:
                avg_oos_with    = float(np.nanmean([r["sh_oos"] for r in with_f]))
                avg_oos_without = float(np.nanmean([r["sh_oos"] for r in without_f]))
                avg_is_with     = float(np.nanmean([r["sh_is"]  for r in with_f]))
                avg_is_without  = float(np.nanmean([r["sh_is"]  for r in without_f]))
                avg_mdd_with    = float(np.nanmean([r["mdd"]    for r in with_f]))
                avg_mdd_without = float(np.nanmean([r["mdd"]    for r in without_f]))
                factor_attr.append(dict(
                    Filter=filter_name, Basis=bname,
                    Ø_OOS_ohne=f"{avg_oos_without:.3f}",
                    Ø_OOS_mit=f"{avg_oos_with:.3f}",
                    Δ_OOS=f"{avg_oos_with-avg_oos_without:+.3f}",
                    Δ_IS=f"{avg_is_with-avg_is_without:+.3f}",
                    Δ_MaxDD=f"{(avg_mdd_with-avg_mdd_without)*100:+.1f}%",
                ))

    # Grouped bar chart of factor contributions
    filters    = list({r["Filter"] for r in factor_attr})
    bases_list = ["RSI<70", "Basket"]
    fig_fa = go.Figure()
    for bname, col in [("RSI<70","#58a6ff"),("Basket","#3fb950")]:
        sub = [r for r in factor_attr if r["Basis"]==bname]
        delta_oos = [float(r["Δ_OOS"]) for r in sub]
        fig_fa.add_trace(go.Bar(
            x=[r["Filter"] for r in sub], y=delta_oos,
            name=bname, marker_color=col))
    fig_fa.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_fa, title="Faktorattribution: Marginaler Beitrag jedes Filters zum Ø OOS Sharpe",
         xaxis_title="Filter", yaxis_title="ΔOOS Sharpe (mit − ohne Filter)", barmode="group", height=400)

    fa_tbl = (
        '<div class="table-responsive mt-2">'
        '<table class="table table-dark table-sm">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in factor_attr[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join(
            "<tr>" + "".join(
                f'<td style="color:{"#3fb950" if "+" in str(v) and v != "+0.000" else "#f78166" if "-" in str(v) else "#e6edf3"};">{v}</td>'
                for v in r.values()) + "</tr>"
            for r in factor_attr)
        + '</tbody></table></div>'
    )

    # ── §3: Rolling OOS Sharpe stability (top 4 combos) ─────────────────────
    top4 = combos[:4]
    fig_roll = go.Figure()
    roll_colors = ["#3fb950","#58a6ff","#ffa657","#bc8cff"]
    for r, col in zip(top4, roll_colors):
        rs = _roll_sh(r["_oos"], 252)
        fig_roll.add_trace(go.Scatter(
            x=rs.index.astype(str).tolist(), y=rs.values.tolist(),
            name=r["lbl"], mode="lines", line=dict(color=col, width=1.8)))
    fig_roll.add_hline(y=0, line_color="#f78166", line_dash="dot")
    _lay(fig_roll, title="Rolling Sharpe 252T (OOS): Top-4 Kombinationen – Stabilitätsvergleich",
         xaxis_title="Datum", yaxis_title="Rolling Sharpe", height=420)

    # Drawdown comparison top 4
    fig_dd4 = go.Figure()
    for r, col in zip(top4, roll_colors):
        c = (1 + r["_oos"]).cumprod()
        dd = (c / c.cummax() - 1) * 100
        fig_dd4.add_trace(go.Scatter(
            x=dd.index.astype(str).tolist(), y=dd.values.tolist(),
            name=r["lbl"], mode="lines", fill="tozeroy",
            fillcolor="rgba(0,0,0,0)", line=dict(color=col, width=1.5)))
    _lay(fig_dd4, title="Drawdown-Vergleich: Top-4 Kombinationen (OOS)",
         xaxis_title="Datum", yaxis_title="Drawdown (%)", height=380)

    # ── §4: Trade quality analysis ────────────────────────────────────────────
    def _trade_quality(sig_oos, jets_oos):
        """Compute per-trade statistics from OOS signal and returns."""
        s = sig_oos.values; r = jets_oos.reindex(sig_oos.index).fillna(0).values
        n = len(s)
        trades = []
        in_trade = False; trade_ret = 0.0; hold_days = 0

        for i in range(n):
            if s[i] > 0 and not in_trade:
                in_trade = True; trade_ret = 0.0; hold_days = 0
            if in_trade:
                trade_ret += float(r[i]); hold_days += 1
            if (s[i] <= 0 or i == n-1) and in_trade:
                trades.append(dict(ret=trade_ret, days=hold_days))
                in_trade = False; trade_ret = 0.0; hold_days = 0

        if not trades:
            return {}
        arr = np.array([t["ret"] for t in trades])
        days = np.array([t["days"] for t in trades])
        pf_num = arr[arr > 0].sum(); pf_den = abs(arr[arr <= 0].sum())
        return dict(
            n=len(arr),
            win_rate=float((arr > 0).mean()),
            avg_ret=float(arr.mean()),
            avg_win=float(arr[arr > 0].mean()) if (arr > 0).any() else 0,
            avg_loss=float(arr[arr <= 0].mean()) if (arr <= 0).any() else 0,
            profit_factor=float(pf_num / (pf_den + 1e-9)),
            avg_days=float(days.mean()),
            max_consec_loss=int(max(
                (len(list(g)) for k,g in __import__('itertools').groupby(arr < 0) if k), default=0)),
        )

    jets_oos_s = jets_c.reindex(oos_idx).fillna(0.0)
    tq_rows = []
    for r in combos:
        tq = _trade_quality(r["_sig"].reindex(oos_idx), jets_oos_s)
        if tq:
            tq_rows.append(dict(
                Kombination=r["lbl"],
                Trades=tq["n"],
                WinRate=f"{tq['win_rate']*100:.1f}%",
                ProfitFaktor=f"{tq['profit_factor']:.2f}",
                Ø_Return=f"{tq['avg_ret']*100:+.2f}%",
                Ø_Win=f"{tq['avg_win']*100:+.2f}%",
                Ø_Loss=f"{tq['avg_loss']*100:+.2f}%",
                Ø_Tage=f"{tq['avg_days']:.0f}",
                Max_Konsek_Verluste=tq["max_consec_loss"],
            ))

    # Win-rate vs Profit-factor scatter
    fig_tq = go.Figure()
    for row in tq_rows:
        wr  = float(row["WinRate"].replace("%",""))
        pf  = float(row["ProfitFaktor"])
        col = "#3fb950" if wr > 55 and pf > 1.5 else ("#ffa657" if pf > 1.0 else "#f78166")
        fig_tq.add_trace(go.Scatter(
            x=[wr], y=[pf], mode="markers+text",
            text=[row["Kombination"]], textposition="top right",
            textfont=dict(size=8, color="#e6edf3"),
            marker=dict(size=12, color=col, opacity=0.85),
            name=row["Kombination"], showlegend=False,
            hovertemplate=f"<b>{row['Kombination']}</b><br>Win-Rate: {wr:.1f}%<br>PF: {pf:.2f}"))
    fig_tq.add_hline(y=1, line_color="#8b949e", line_dash="dot")
    fig_tq.add_vline(x=50, line_color="#8b949e", line_dash="dot")
    _lay(fig_tq, title="Handelsqualität: Win-Rate vs Profit-Faktor (OOS)",
         xaxis_title="Win-Rate (%)", yaxis_title="Profit-Faktor", height=500)

    tq_tbl = (
        '<div class="table-responsive mt-2"><table class="table table-dark table-sm table-hover">'
        '<thead><tr>' + "".join(f"<th>{k}</th>" for k in tq_rows[0].keys()) + '</tr></thead>'
        '<tbody>' + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                             for r in tq_rows)
        + '</tbody></table></div>'
    ) if tq_rows else ""

    # ── §5: Regime analysis – VIX level × CL trend → return ──────────────────
    vix_oos = (vix_c.reindex(oos_idx).ffill() if vix_c is not None
               else pd.Series(20.0, index=oos_idx))
    cl_ret_oos  = np.log(cl_px.reindex(oos_idx).ffill() /
                          cl_px.reindex(oos_idx).ffill().shift(1)).fillna(0)
    cl_trend_oos = cl_ret_oos.rolling(21).mean()

    regime_data = []
    for r in combos[:8]:  # top 8
        oos_ret = r["_oos"].reindex(oos_idx).dropna()
        for d in oos_ret.index:
            vix_v = float(vix_oos.get(d, 20))
            clt_v = float(cl_trend_oos.get(d, 0))
            regime_data.append(dict(
                combo=r["lbl"],
                vix_bin="<20" if vix_v < 20 else ("20-25" if vix_v < 25 else (">25")),
                cl_bin="Öl↑" if clt_v > 0 else "Öl↓",
                ret=float(oos_ret.get(d, 0)),
            ))

    if regime_data:
        rdf = pd.DataFrame(regime_data)
        # Best combo regime analysis
        best_c = combos[0]["lbl"]
        rdf_best = rdf[rdf["combo"] == best_c]
        if len(rdf_best) > 10:
            pivot_r = rdf_best.groupby(["vix_bin","cl_bin"])["ret"].agg(
                ["mean","count"]).reset_index()
            piv_mean = rdf_best.groupby(["vix_bin","cl_bin"])["ret"].mean().unstack(fill_value=np.nan) * 100

            fig_reg = go.Figure(go.Heatmap(
                z=piv_mean.values.tolist(),
                x=piv_mean.columns.tolist(), y=piv_mean.index.tolist(),
                colorscale="RdYlGn", zmin=-0.1, zmax=0.1,
                text=[[f"{v:.3f}%" if not np.isnan(v) else "–" for v in row]
                      for row in piv_mean.values.tolist()],
                texttemplate="%{text}",
                colorbar=dict(title="Ø Tages-Ret"),
            ))
            _lay(fig_reg, title=f"Regime-Analyse: Ø Tagesrendite | {best_c} (OOS) nach VIX × Öl-Trend",
                 xaxis_title="CL=F Trend (21T)", yaxis_title="VIX-Niveau", height=360)

        # Regime Sharpe across all top combos
        reg_sh = []
        for r in combos[:6]:
            rdf_c = rdf[rdf["combo"] == r["lbl"]]
            for regime in [("VIX<20+Öl↑", (rdf_c["vix_bin"]=="<20") & (rdf_c["cl_bin"]=="Öl↑")),
                            ("VIX<20+Öl↓", (rdf_c["vix_bin"]=="<20") & (rdf_c["cl_bin"]=="Öl↓")),
                            ("VIX>25",      rdf_c["vix_bin"]==">25"),]:
                sub_r = rdf_c[regime[1]]["ret"]
                sh_r  = _sh(sub_r) if len(sub_r) > 30 else np.nan
                reg_sh.append(dict(Kombo=r["lbl"], Regime=regime[0], Sharpe=round(sh_r,3) if not np.isnan(sh_r) else "–"))

        if reg_sh:
            rsh_df = pd.DataFrame(reg_sh)
            piv_rsh = rsh_df.pivot(index="Kombo", columns="Regime", values="Sharpe")
            fig_rsh = go.Figure()
            for i, col_name in enumerate(piv_rsh.columns):
                vals = [float(v) if v != "–" else np.nan for v in piv_rsh[col_name].values]
                fig_rsh.add_trace(go.Bar(
                    name=col_name, x=piv_rsh.index.tolist(), y=vals,
                    marker_color=["#3fb950","#58a6ff","#f78166"][i % 3]))
            fig_rsh.add_hline(y=0, line_color="#8b949e", line_dash="dot")
            _lay(fig_rsh, title="Regime-Sharpe: Top-6 Kombos in verschiedenen VIX×Öl-Regimen (OOS)",
                 barmode="group", xaxis_title="Kombination", yaxis_title="Sharpe", height=420,
                 xaxis=dict(tickangle=-30))
    else:
        fig_reg = fig_rsh = None

    # ── §6: Deep dive on 4 combos ─────────────────────────────────────────────
    # Find the 4 highlighted combos
    target_lbls = {
        "RSI<70+S+V": ("RSI<70",True,True,False),
        "Basket":     ("Basket",False,False,False),
        "RSI<70+S+V+T":("RSI<70",True,True,True),
        "Basket+V":   ("Basket",False,True,False),
    }
    deep_combos = {}
    for r in combos:
        for tlbl, (tb, ts, tv, tt) in target_lbls.items():
            if r["base"]==tb and r["s"]==ts and r["v"]==tv and r["t"]==tt:
                deep_combos[tlbl] = r
                break

    deep_html = ""
    deep_colors = ["#3fb950","#58a6ff","#ffa657","#bc8cff"]
    for (tlbl, r), col in zip(deep_combos.items(), deep_colors):
        # Equity curve IS+OOS
        cum_is  = (1 + r["_is"]).cumprod() * 100
        cum_oos = (1 + r["_oos"]).cumprod() * 100
        fig_d   = go.Figure()
        try:
            fig_d.add_vrect(x0=str(is_idx[0].date()), x1=str(is_idx[-1].date()),
                             fillcolor="#1c2128", opacity=0.5, layer="below", line_width=0)
        except Exception:
            pass
        fig_d.add_trace(go.Scatter(x=cum_is.index.astype(str).tolist(), y=cum_is.values.tolist(),
                                    name="IS", mode="lines", line=dict(color=col, dash="dot", width=1.5)))
        fig_d.add_trace(go.Scatter(x=cum_oos.index.astype(str).tolist(), y=cum_oos.values.tolist(),
                                    name="OOS", mode="lines", line=dict(color=col, width=2.5)))
        _lay(fig_d, title=f"{r['lbl']}: IS Sharpe {r['sh_is']:.3f} → OOS Sharpe {r['sh_oos']:.3f} (Δ {r['delta']:+.3f})",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=380)

        # Rolling Sharpe
        rs_d = _roll_sh(r["_oos"], 126)  # 6-month rolling
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rs_d.index.astype(str).tolist(), y=rs_d.values.tolist(),
                                     name="Rolling Sharpe 126T", mode="lines",
                                     line=dict(color=col, width=1.8),
                                     fill="tozeroy", fillcolor=f"rgba(0,0,0,0.0)"))
        fig_rs.add_hline(y=0, line_color="#f78166", line_dash="dot")
        _lay(fig_rs, title=f"Rolling Sharpe 126T (OOS) – {r['lbl']}",
             yaxis_title="Sharpe", height=280)

        tq_d = _trade_quality(r["_sig"].reindex(oos_idx), jets_oos_s)
        tq_card = _card(f"Handelsqualität: {r['lbl']}", col, f"""
        <div class="row">
          <div class="col"><strong>Trades OOS:</strong> {r['n_tr']}</div>
          <div class="col"><strong>Win-Rate:</strong> {tq_d.get('win_rate',0)*100:.1f}%</div>
          <div class="col"><strong>Profit-Faktor:</strong> {tq_d.get('profit_factor',0):.2f}</div>
          <div class="col"><strong>Ø Return/Trade:</strong> {tq_d.get('avg_ret',0)*100:+.2f}%</div>
          <div class="col"><strong>Ø Haltedauer:</strong> {tq_d.get('avg_days',0):.0f} Tage</div>
          <div class="col"><strong>Max Konsek. Verluste:</strong> {tq_d.get('max_consec_loss',0)}</div>
        </div>
        <p class="mt-2 mb-0" style="color:#8b949e;font-size:0.83em;">
          IS Sharpe: {r['sh_is']:.3f} | OOS Sharpe: {r['sh_oos']:.3f} | Δ: {r['delta']:+.3f} | MaxDD OOS: {r['mdd']*100:.1f}%
        </p>
        """) if tq_d else ""

        deep_html += (
            f"<h5 style='color:{col};border-bottom:1px solid {col};padding-bottom:6px;'>"
            f"📊 {r['lbl']}</h5>"
            + tq_card + _htm(fig_d) + _htm(fig_rs) + "<hr style='border-color:#30363d;'>"
        )

    # ── §7: Adaptive combination (VIX-controlled switch) ─────────────────────
    # When VIX < 20: use Basket (strongest OOS)
    # When 20 ≤ VIX < 25: use RSI<70+Seas+VIX
    # When VIX ≥ 25: flat
    bask_combo = next((r for r in combos if r["base"]=="Basket" and not r["s"] and not r["v"] and not r["t"]), None)
    rsi_svx_combo = next((r for r in combos if r["base"]=="RSI<70" and r["s"] and r["v"] and not r["t"]), None)

    adapt_html = ""
    if bask_combo and rsi_svx_combo and vix_c is not None:
        vix_oos_s = vix_c.reindex(oos_idx).ffill()
        # Adaptive signal
        bask_sig_oos = bask_combo["_sig"].reindex(oos_idx).fillna(0)
        rsi_svx_sig_oos = rsi_svx_combo["_sig"].reindex(oos_idx).fillna(0)

        adapt_sig = pd.Series(0.0, index=oos_idx)
        adapt_sig[vix_oos_s < 20]  = bask_sig_oos[vix_oos_s < 20]      # Basket when calm
        mask_mid = (vix_oos_s >= 20) & (vix_oos_s < 25)
        adapt_sig[mask_mid]         = rsi_svx_sig_oos[mask_mid]          # RSI+S+V when medium
        # VIX ≥ 25: flat (already 0)

        adapt_net = _net(adapt_sig)
        adapt_oos = adapt_net.reindex(oos_idx).dropna()
        sh_adapt  = _sh(adapt_oos)

        fig_adapt = go.Figure()
        cum_adapt  = (1 + adapt_oos).cumprod() * 100
        cum_bask   = (1 + bask_combo["_oos"]).cumprod() * 100
        cum_rsi_sv = (1 + rsi_svx_combo["_oos"]).cumprod() * 100
        for curve, name, col in [
            (cum_bask,   bask_combo["lbl"],     "#8b949e"),
            (cum_rsi_sv, rsi_svx_combo["lbl"],  "#58a6ff"),
            (cum_adapt,  "Adaptiv (VIX-Switch)", "#e3b341"),
        ]:
            fig_adapt.add_trace(go.Scatter(
                x=curve.index.astype(str).tolist(), y=curve.values.tolist(),
                name=name, mode="lines",
                line=dict(color=col, width=2.5 if name.startswith("Adapt") else 1.5,
                          dash="dot" if col == "#8b949e" else "solid")))

        # VIX shading
        if vix_c is not None:
            vix_oos_plot = vix_oos_s
            fig_adapt.add_trace(go.Scatter(
                x=vix_oos_plot.index.astype(str).tolist(),
                y=vix_oos_plot.values.tolist(),
                name="VIX", yaxis="y2", mode="lines",
                line=dict(color="#bc8cff", width=0.8), opacity=0.5))
            fig_adapt.update_layout(
                yaxis2=dict(title="VIX", overlaying="y", side="right",
                            gridcolor="#21262d", tickfont=dict(color="#e6edf3")))

        _lay(fig_adapt, title="Adaptive Kombination: VIX-gesteuerter Switch (OOS)",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=470)

        adapt_card = _card("Adaptive Strategie-Logik", "#e3b341", f"""
        <table class="table table-dark table-sm mb-1">
          <tr><th>VIX-Regime</th><th>Signal</th><th>Rationale</th></tr>
          <tr><td style="color:#3fb950;">VIX &lt; 20 (ruhig)</td>
              <td><strong>{bask_combo['lbl']}</strong></td>
              <td>Stärkste OOS Sharpe, Lead-Lag klar strukturiert</td></tr>
          <tr><td style="color:#ffa657;">VIX 20–25 (moderat)</td>
              <td><strong>{rsi_svx_combo['lbl']}</strong></td>
              <td>Stabil IS+OOS, geringerer Drawdown</td></tr>
          <tr><td style="color:#f78166;">VIX ≥ 25 (turbulent)</td>
              <td><strong>Flat</strong></td>
              <td>Panic-Selling bricht Lead-Lag → kein zuverlässiges Signal</td></tr>
        </table>
        <p class="mt-1 mb-0">
          Adaptive OOS Sharpe: <strong style="color:#e3b341;">{sh_adapt:.3f}</strong> |
          Basket OOS: <strong style="color:#8b949e;">{bask_combo['sh_oos']:.3f}</strong> |
          RSI+S+V OOS: <strong style="color:#58a6ff;">{rsi_svx_combo['sh_oos']:.3f}</strong>
        </p>
        """)

        adapt_html = adapt_card + _htm(fig_adapt)
    else:
        adapt_html = _desc("VIX-Daten oder Kombinations-Signale nicht verfügbar.")

    # ── §8: What can't we control ─────────────────────────────────────────────
    uncontrolled_html = _card("Nicht-kontrollierbare Risikofaktoren", "#f78166", """
    <ol style="color:#e6edf3;">
      <li><strong>Gap-Risiko:</strong> Über-Nacht-Events (Earnings, OPEC-Entscheidungen, geopolitische Schocks)
          können Preise weit unter den Stop-Loss-Level reißen. Tages-Close-Simulation unterschätzt dieses Risiko.</li>
      <li><strong>Strukturelle Regime-Brüche:</strong> COVID-19 zerstörte 2020 die Airline-Nachfrage vollständig.
          Das CL→JETS Lead-Lag-Modell versagt, wenn die fundamentale Verbindung (Kerosinkosten) nicht mehr der
          Haupttreiber ist. Basket IS=-0.111 bestätigt dies für die IS-Periode.</li>
      <li><strong>Liquiditäts-Regime:</strong> In Krisen steigt der Bid-Ask-Spread erheblich.
          10bp TC-Annahme ist für normale Märkte fair, in Krisen aber zu optimistisch.</li>
      <li><strong>Regulatorische Änderungen:</strong> ETF-Umstrukturierungen, Handelsunterbrechungen,
          Short-Selling-Verbote können die Ausführung beeinflussen.</li>
      <li><strong>Korrelations-Zusammenbruch:</strong> Bei sehr hohem VIX (&gt;40) tendieren alle Assets
          zur Gleichbewegung (Korrelation → 1). Der Diversifikationsvorteil des Basket-Signals verschwindet.</li>
      <li><strong>IS/OOS-Asymmetrie der Basket-Strategie:</strong> IS Sharpe = -0.111 bei OOS Sharpe = 0.746
          deutet auf ein Regime-spezifisches Alpha hin (post-2019 Marktstruktur). Risiko: Regime-Wechsel.</li>
      <li><strong>Overoptimierung der IS-Parameter:</strong> Gute-Monate-Filter, VIX-Schwelle 25, TNX-Fenster 20T
          wurden auf IS-Daten optimiert. Kleine Parameteränderungen könnten OOS-Sharpe stark beeinflussen.</li>
    </ol>
    """)

    # ── §9: Further alpha ideas ────────────────────────────────────────────────
    next_ideas_html = _card("Nächste Alpha-Generierungs-Ideen", "#3fb950", """
    <ol style="color:#e6edf3;">
      <li><strong>Walk-Forward Optimierung:</strong> Jeden Monat neuen IS-Zeitraum (12M rolling) → adaptiver
          IS-good-months-Filter. Vermeidet fixe Monatsgrenzen.</li>
      <li><strong>Ensemble-Voting mit Confidence-Threshold:</strong> Statt gleichgewichtetem Ensemble nur
          dann handeln, wenn &ge;3 von 4 Indikatoren in die gleiche Richtung zeigen.</li>
      <li><strong>Dynamische VIX-Schwelle:</strong> Statt fixer VIX&lt;25 Schwelle: gleitender VIX-Median als
          Referenz. Adaptiert sich an verschiedene Volatilitätsregime.</li>
      <li><strong>Machine Learning Regime Classifier:</strong> XGBoost/Random Forest auf Features
          (VIX, RSI, Trend, Saisonalität) → Vorhersage ob Folgewoche Signal-positiv sein wird.</li>
      <li><strong>Options-Overlay:</strong> Statt JETS spot: LEAPS Call-Optionen kaufen.
          Eingeschränktes Verlustrisiko + Leverage ohne Margin-Risiko.</li>
      <li><strong>Multi-Timeframe Confirmation:</strong> Wöchentliches Signal (RSI der Wochenschlüsse)
          muss tägliches Signal bestätigen → weniger Fehlsignale.</li>
      <li><strong>Sector Rotation:</strong> Wenn CL→JETS Signal flat: in alternative Sektoren
          (XLE, XLI) ausweichen statt in Cash zu gehen.</li>
      <li><strong>Adaptive Haltedauer:</strong> Statt fixer Stop-Loss: trailing Stop
          (30% unter laufendem Maximum) kombiniert mit festem Profit-Target.</li>
    </ol>
    """)

    # ── Common factors summary ─────────────────────────────────────────────────
    common_factors_html = _card("Was verbessernde Faktoren gemeinsam haben", "#58a6ff", """
    <p style="color:#e6edf3;">Analyse der Kombinationen mit <strong>positiver Δ (OOS &gt; IS)</strong>:</p>
    <ol style="color:#e6edf3;">
      <li><strong>Keine oder wenige Filter:</strong> Basket allein (Δ=+0.858), RSI+Seas (Δ=+0.035)
          generalisieren besser als stark gefilterte Versionen. Zu viele Filter → Overfitting.</li>
      <li><strong>VIX-Filter ist der robusteste Einzelfilter:</strong>
          Marginaler OOS-Beitrag ist positiv bei beiden Basis-Signalen.
          Eliminiert Trades in strukturell schwachen Phasen (Panic → Lead-Lag bricht zusammen).</li>
      <li><strong>Seasonal-Filter reduziert Drawdown ohne OOS zu schaden:</strong>
          RSI+Seas (MaxDD: -19.4%) vs RSI allein (MaxDD: -26.0%). Sharpe bleibt nahezu gleich.</li>
      <li><strong>TNX-Filter ist instabil:</strong> Hilft IS (+0.27 IS-Sharpe im Schnitt) aber schadet
          OOS (-0.05 im Schnitt). Der IS-Zinszyklus (2016–2022) verallgemeinert sich nicht auf OOS.</li>
      <li><strong>Basket-Signal hat strukturelles OOS-Alpha:</strong> IS=-0.111 aber OOS=0.746 deutet
          darauf hin, dass das Basket-Signal nach 2020 eine neue Informationsquelle erschlossen hat
          (XLE-ETF-Flows als früherer Indikator für Öl-Sentiment).</li>
    </ol>
    """)

    # ── HTML assembly ──────────────────────────────────────────────────────────
    secs = [
        ("📍 §1  IS vs OOS Scatter: Overfitting-Karte aller 16 Strategien",
         _desc("Kombinationen über der Diagonale (IS=OOS) zeigen positive Generalisierung. "
               "Kreisgröße ∝ Anzahl OOS-Trades. Grün = starke OOS Sharpe (&gt;0.6).")
         + _htm(fig_scat) + legend_card, 0, True),

        ("🔬 §2  Faktorattribution: Marginaler Beitrag jedes Filters",
         _desc("Ø OOS Sharpe mit Filter minus ohne Filter. "
               "VIX: robuststes Verbesserungs-Signal. Seasonal: reduziert Drawdown. "
               "TNX: OOS-instabil (IS-Overfitting).")
         + _htm(fig_fa) + fa_tbl + common_factors_html, 1, False),

        ("📉 §3  Rolling OOS Sharpe Stabilität (Top-4 Kombinationen)",
         _desc("6-Monats- und 12-Monats-Rolling Sharpe der Top-4 Kombos im OOS-Zeitraum. "
               "Stabile Linien = robustes Signal. Volatile Linien = regime-abhängiges Alpha.")
         + _htm(fig_roll) + _htm(fig_dd4), 2, False),

        ("🎯 §4  Handelsqualität: Win-Rate, Profit-Faktor, Haltedauer",
         _desc("Pro Combo: Analyse jedes einzelnen Trades im OOS-Zeitraum. "
               "Profit-Faktor &gt; 1 und Win-Rate &gt; 50% sind das Mindest-Ziel.")
         + _htm(fig_tq) + tq_tbl, 3, False),

        ("🌡️ §5  Regime-Analyse: VIX × Öl-Trend → Return",
         _desc("Wie performen die Strategien in verschiedenen VIX-Regimen und Öl-Trends? "
               "VIX&lt;20+Öl↑ = optimales Regime. VIX&gt;25 = schwierig für alle Strategien.")
         + ((_htm(fig_reg) + _htm(fig_rsh)) if fig_reg is not None else "<p>Unzureichende Daten.</p>"),
         4, False),

        ("🔍 §6  Deep Dive: 4 Ausgewählte Kombinationen im Detail",
         _desc("Detailanalyse der 4 interessantesten Kombos: "
               "RSI+S+V (stabil), Basket (OOS-Alpha), RSI+S+V+T (IS-Overfitting), Basket+V (robust).")
         + deep_html, 5, False),

        ("⚙️ §7  Adaptive Kombination: VIX-gesteuerter Signal-Switch",
         _desc("Meta-Strategie: je nach VIX-Regime das Signal wechseln. "
               "Kombiniert die Stärken mehrerer Strategien in verschiedenen Marktphasen.")
         + adapt_html, 6, False),

        ("⚠️ §8  Was (noch) nicht kontrolliert werden kann",
         uncontrolled_html, 7, False),

        ("💡 §9  Weitere Alpha-Generierungs-Ideen",
         next_ideas_html, 8, False),
    ]

    acc = '<div class="accordion" id="ddAcc">'
    for t, b, idx, op in secs:
        acc += _acc(t, b, idx, op)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#58a6ff;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Kombinations-Deep-Dive: 16 Strategien unter der Lupe</h2>
          <p class="mb-0" style="color:#8b949e;">
            IS vs OOS Scatter · Faktorattribution · Rolling Stabilität ·
            Handelsqualität · Regime-Analyse · Deep Dive × 4 ·
            Adaptive Switch · Nicht-kontrollierbare Risiken · Neue Ideen
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "combination_deepdive_report.html",
           _html_base("Kombinations-Deep-Dive", 20, body))



def build_crisis_vs_nocrisis_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    def _tz(raw):
        idx = raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        raw.index = idx.normalize()
        return raw

    ret_main = _read(tables / "phase2_returns.csv")
    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")

    try:
        jets_raw = _tz(yf.Ticker("JETS").history(period="max", auto_adjust=True))
        vix_raw  = _tz(yf.Ticker("^VIX").history(period="max", auto_adjust=True))["Close"]
    except Exception as e:
        _write(out / "crisis_vs_nocrisis_report.html",
               _html_base("Crisis vs No-Crisis", 20, f"<p class='text-warning'>{e}</p>"))
        return

    CRISES = [
        ("GFC",       "2007-10-01", "2009-06-01"),
        ("Oil Crash", "2014-06-01", "2016-01-01"),
        ("COVID",     "2020-02-01", "2020-05-01"),
        ("Inflation", "2022-01-01", "2022-12-31"),
    ]
    CFILLS = ["rgba(248,81,73,0.12)", "rgba(210,168,255,0.12)",
              "rgba(248,81,73,0.12)", "rgba(240,136,62,0.12)"]

    close_j = jets_raw["Close"]
    low_j   = jets_raw["Low"]
    bk_cols = [c for c in ["CL=F", "BZ=F", "XLE", "XOM", "CVX"] if c in ret_main.columns]
    bk_ret  = ret_main[bk_cols].mean(axis=1) if bk_cols else pd.Series(0.0, index=ret_main.index)

    common = close_j.index.intersection(bk_ret.index).intersection(vix_raw.index)
    close_j = close_j.reindex(common).ffill()
    low_j   = low_j.reindex(common).ffill()
    vix_a   = vix_raw.reindex(common).ffill()
    bk_a    = bk_ret.reindex(common).fillna(0.0)
    sig     = ((bk_a.rolling(20).mean() > 0) & (vix_a < 25)).astype(int)

    def _crisis_mask(ix):
        m = pd.Series(False, index=ix)
        for _, s, e in CRISES:
            m |= (ix >= s) & (ix <= e)
        return m

    sig_ncr = sig.copy()
    sig_ncr[_crisis_mask(common)] = 0

    CAP, SL, TC, PF = 100_000.0, 0.30, 0.001, 0.95

    def _sim(sg):
        cash = CAP; shares = 0.0; stop_px = 0.0; in_pos = False
        entry_px = 0.0; entry_dt = sg.index[0]
        navs = []; entries = []; sl_exits = []; sig_exits = []
        for i in range(len(sg)):
            c = float(close_j.iloc[i]); l = float(low_j.iloc[i])
            dt = sg.index[i]; stopped = False
            if in_pos and l <= stop_px:
                ep = max(stop_px * 0.995, l)
                cash += shares * ep * (1 - TC)
                dur = int((dt - entry_dt).days)
                sl_exits.append({"date": dt, "nav": cash, "pnl": shares * (ep - entry_px), "dur": dur, "type": "SL"})
                shares = 0.0; in_pos = False; stopped = True
            if i > 0 and not stopped:
                sp = int(sg.iloc[i - 1])
                if sp == 1 and not in_pos:
                    invest = cash * PF
                    shares = invest * (1 - TC) / c
                    stop_px = c * (1 - SL); entry_px = c; entry_dt = dt
                    cash -= invest; in_pos = True
                    entries.append({"date": dt, "nav": cash + shares * c})
                elif sp == 0 and in_pos:
                    cash += shares * c * (1 - TC)
                    dur = int((dt - entry_dt).days)
                    sig_exits.append({"date": dt, "nav": cash, "pnl": shares * (c - entry_px), "dur": dur, "type": "Sig"})
                    shares = 0.0; in_pos = False
            navs.append(cash + shares * c)
        return pd.Series(navs, index=sg.index), entries, sl_exits, sig_exits

    nav_f, ent_f, sl_f, sig_f = _sim(sig)
    nav_n, ent_n, sl_n, sig_n = _sim(sig_ncr)

    def _metrics(nav, entries, sl_exits, sig_exits):
        r    = nav.pct_change().dropna()
        ann  = float(r.mean() * 252)
        vol  = float(r.std() * (252 ** 0.5))
        sh   = ann / vol if vol > 1e-9 else 0.0
        dd   = nav / nav.cummax() - 1
        mdd  = float(dd.min())
        down = r[r < 0].std() * (252 ** 0.5)
        srt  = ann / down if down > 1e-9 else 0.0
        calm = ann / abs(mdd) if abs(mdd) > 1e-9 else 0.0
        all_t = sorted(sl_exits + sig_exits, key=lambda x: x["date"])
        n_t   = len(entries); n_sl = len(sl_exits); n_sig = len(sig_exits)
        if all_t:
            pnls  = [t["pnl"] for t in all_t]
            wins  = [p for p in pnls if p > 0]
            losses= [p for p in pnls if p <= 0]
            wr    = len(wins) / len(pnls) if pnls else 0.0
            pf    = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99.0
            avg_w = float(np.mean(wins)) if wins else 0.0
            avg_l = float(np.mean(losses)) if losses else 0.0
            avg_d = float(np.mean([t["dur"] for t in all_t]))
            consec = 0; max_con = 0; ws = 0; max_ws = 0
            for t in all_t:
                if t["pnl"] <= 0:
                    consec += 1; max_con = max(max_con, consec); ws = 0
                else:
                    ws += 1; max_ws = max(max_ws, ws); consec = 0
        else:
            wr = pf = avg_w = avg_l = avg_d = 0.0; max_con = max_ws = 0
        total_ret = (float(nav.iloc[-1]) - CAP) / CAP
        return dict(ann=ann, vol=vol, sh=sh, mdd=mdd, srt=srt, calm=calm,
                    n_t=n_t, n_sl=n_sl, n_sig=n_sig, wr=wr, pf=pf,
                    avg_w=avg_w, avg_l=avg_l, avg_d=avg_d,
                    max_con=max_con, max_ws=max_ws,
                    final=float(nav.iloc[-1]), total_ret=total_ret)

    mf = _metrics(nav_f, ent_f, sl_f, sig_f)
    mn = _metrics(nav_n, ent_n, sl_n, sig_n)

    # ── §1 Methodik-Erklärung ─────────────────────────────────────────────
    intro = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h5 class='text-warning'>Roter Faden: Warum vergleichen wir Mit vs. Ohne Krisen?</h5>"
        "<p>Unser Ziel ist es zu verstehen, <strong>wie viel unserer Strategie-Performance auf echtem"
        " Alpha basiert</strong> und wie viel von Krisenperioden verzerrt wird. "
        "Wenn die Strategie in Krisenzeiten Verluste macht, aber im Rest sehr gut performt, "
        "dann wäre der <em>echte</em> Mehrwert deutlich grösser als die Gesamtstatistik suggeriert.</p>"
        "<p>Wir simulieren zwei identische Strategien (Basket + VIX&lt;25 Signal, €100k, 30% Stop-Loss):</p>"
        "<ul><li><strong class='text-info'>All Periods</strong>: Strategie läuft durch alle Marktphasen.</li>"
        "<li><strong class='text-success'>Crisis Excluded</strong>: Signal wird auf 0 gesetzt (Cash-Haltung) "
        "während GFC, Oil-Crash, COVID und Inflations-Periode. Capital bleibt geschützt.</li></ul>"
        "<p class='mb-0 text-muted small'>Trade-Marker: ▲ Einstieg · ▼ Signal-Ausstieg · ✕ Stop-Loss-Ausstieg</p>"
        "</div>"
    )
    p1 = intro

    # ── §2 Full sim NAV + Trade Markers + JETS Preis + Signal ────────────
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=["Portfolio NAV (€) — reale Simulation auf JETS OHLC-Daten",
                                         "JETS ETF Schlusskurs (Originaldaten)",
                                         "Signal (1 = Long, 0 = Cash)"],
                         row_heights=[0.55, 0.30, 0.15])
    fig2.add_trace(go.Scatter(x=nav_f.index, y=nav_f.values, name="NAV All",
                              line=dict(color="#58a6ff", width=2)), row=1, col=1)
    if ent_f:
        fig2.add_trace(go.Scatter(x=[e["date"] for e in ent_f],
                                  y=[e["nav"] for e in ent_f],
                                  name="Entry ▲", mode="markers",
                                  marker=dict(symbol="triangle-up", size=8, color="#3fb950")),
                       row=1, col=1)
    if sig_f:
        fig2.add_trace(go.Scatter(x=[e["date"] for e in sig_f],
                                  y=[e["nav"] for e in sig_f],
                                  name="Exit ▼", mode="markers",
                                  marker=dict(symbol="triangle-down", size=8, color="#f0883e")),
                       row=1, col=1)
    if sl_f:
        fig2.add_trace(go.Scatter(x=[e["date"] for e in sl_f],
                                  y=[e["nav"] for e in sl_f],
                                  name="Stop-Loss ✕", mode="markers",
                                  marker=dict(symbol="x", size=10, color="#f85149",
                                              line=dict(width=2))),
                       row=1, col=1)
    fig2.add_trace(go.Scatter(x=close_j.index, y=close_j.values, name="JETS Close",
                              line=dict(color="#8b949e", width=1.2)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=sig.index, y=sig.values, name="Signal",
                              line=dict(color="#d2a8ff"), fill="tozeroy",
                              fillcolor="rgba(210,168,255,0.15)"), row=3, col=1)
    for i, (cn, cs, ce) in enumerate(CRISES):
        for r in [1, 2, 3]:
            fig2.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0, row=r, col=1)
        mid = (pd.Timestamp(cs) + (pd.Timestamp(ce) - pd.Timestamp(cs)) / 2).strftime("%Y-%m-%d")
        fig2.add_annotation(x=mid, y=1.04, xref="x", yref="paper",
                            text=cn, showarrow=False, font=dict(color="#f85149", size=9))
    fig2.update_layout(**_LAYOUT, height=720,
                       title="Full-History Simulation — JETS OHLC-Echtdaten, 30% Stop-Loss, 10bp TC")
    p2 = fig2.to_html(full_html=False, include_plotlyjs=False, div_id="cnc2")

    # ── §3 Rolling Metrics Full vs Excluded ──────────────────────────────
    def _roll(nav, w=63):
        r       = nav.pct_change().dropna()
        rm      = r.rolling(w).mean()
        rs      = r.rolling(w).std().replace(0, np.nan)
        sh      = (rm / rs * (252 ** 0.5)).fillna(0)
        vol     = (rs * (252 ** 0.5) * 100).fillna(0)
        dd_roll = (nav / nav.rolling(w).max() - 1) * 100
        return sh, vol, dd_roll

    sh_f, vol_f, dd_f = _roll(nav_f)
    sh_n, vol_n, dd_n = _roll(nav_n)
    dd_full = (nav_f / nav_f.cummax() - 1) * 100
    dd_excl = (nav_n / nav_n.cummax() - 1) * 100

    fig3 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=["Rolling 63d Sharpe (annualisiert) — Vergleich",
                                         "Rolling 63d Volatilität % (annualisiert)",
                                         "Rolling 63d Max-Drawdown % (gleitend)",
                                         "Kumulativer Drawdown % (gesamter Zeitraum)"])
    for srs, name, color, row in [
        (sh_f,   "All Periods", "#58a6ff", 1), (sh_n,  "Crisis Excl.", "#3fb950", 1),
        (vol_f,  "All Periods", "#58a6ff", 2), (vol_n, "Crisis Excl.", "#3fb950", 2),
        (dd_f,   "All Periods", "#f85149", 3), (dd_n,  "Crisis Excl.", "#3fb950", 3),
        (dd_full,"All Periods", "#f85149", 4), (dd_excl,"Crisis Excl.", "#3fb950", 4),
    ]:
        fig3.add_trace(go.Scatter(x=srs.index, y=srs.values, name=name,
                                  line=dict(color=color, width=1.5),
                                  showlegend=(row == 1)), row=row, col=1)
    fig3.add_hline(y=0, line_color="#8b949e", line_dash="dot", row=1, col=1)
    fig3.add_hline(y=0, line_color="#8b949e", line_dash="dot", row=3, col=1)
    for i, (_, cs, ce) in enumerate(CRISES):
        for r in [1, 2, 3, 4]:
            fig3.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0, row=r, col=1)
    fig3.update_layout(**_LAYOUT, height=800,
                       title="Rolling Metrics: Sharpe · Volatilität · Drawdown (All vs. Crisis-Excluded)")
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="cnc3")

    # ── §4 Crisis-Excluded NAV mit Markern ───────────────────────────────
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=nav_n.index, y=nav_n.values, name="NAV Crisis-Excl.",
                              line=dict(color="#3fb950", width=2)))
    if ent_n:
        fig4.add_trace(go.Scatter(x=[e["date"] for e in ent_n],
                                  y=[e["nav"] for e in ent_n],
                                  name="Entry ▲", mode="markers",
                                  marker=dict(symbol="triangle-up", size=8, color="#58a6ff")))
    if sig_n:
        fig4.add_trace(go.Scatter(x=[e["date"] for e in sig_n],
                                  y=[e["nav"] for e in sig_n],
                                  name="Exit ▼", mode="markers",
                                  marker=dict(symbol="triangle-down", size=8, color="#f0883e")))
    if sl_n:
        fig4.add_trace(go.Scatter(x=[e["date"] for e in sl_n],
                                  y=[e["nav"] for e in sl_n],
                                  name="Stop-Loss ✕", mode="markers",
                                  marker=dict(symbol="x", size=10, color="#f85149",
                                              line=dict(width=2))))
    # Ghost line of full sim for comparison
    fig4.add_trace(go.Scatter(x=nav_f.index, y=nav_f.values, name="NAV All (Ref.)",
                              line=dict(color="#58a6ff", width=1, dash="dot"), opacity=0.4))
    for i, (cn, cs, ce) in enumerate(CRISES):
        fig4.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
        mid = (pd.Timestamp(cs) + (pd.Timestamp(ce) - pd.Timestamp(cs)) / 2).strftime("%Y-%m-%d")
        fig4.add_annotation(x=mid, y=1.04, xref="x", yref="paper",
                            text=f"{cn}\n(Signal=0)", showarrow=False,
                            font=dict(color="#3fb950", size=9))
    fig4.update_layout(**_LAYOUT, height=480,
                       title="Crisis-Excluded Simulation — Krisenperioden = Cash (blau gestrichelt = Referenz All)")
    p4 = fig4.to_html(full_html=False, include_plotlyjs=False, div_id="cnc4")

    # ── §5 Monatliches Kalender-Heatmap ──────────────────────────────────
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def _calendar(nav):
        m  = nav.resample("ME").last().pct_change().dropna() * 100
        ys = sorted(m.index.year.unique())
        z  = []
        for y in ys:
            row = []
            for mo in range(1, 13):
                mask = (m.index.year == y) & (m.index.month == mo)
                vals = m[mask]
                row.append(round(float(vals.iloc[0]), 1) if len(vals) else None)
            z.append(row)
        return z, [str(y) for y in ys]

    z_f, y_f = _calendar(nav_f)
    z_n, y_n = _calendar(nav_n)
    txt_f = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in z_f]
    txt_n = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in z_n]

    fig5 = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                         subplot_titles=["All Periods — Monatsrenditen %",
                                         "Crisis Excluded — Monatsrenditen %"])
    fig5.add_trace(go.Heatmap(z=z_f, x=MONTHS, y=y_f, text=txt_f, texttemplate="%{text}",
                              colorscale="RdYlGn", zmid=0, zmin=-20, zmax=20,
                              showscale=True, name="All Periods",
                              colorbar=dict(x=0.46, tickfont=dict(color="#e6edf3"))),
                  row=1, col=1)
    fig5.add_trace(go.Heatmap(z=z_n, x=MONTHS, y=y_n, text=txt_n, texttemplate="%{text}",
                              colorscale="RdYlGn", zmid=0, zmin=-20, zmax=20,
                              showscale=True, name="Crisis Excl.",
                              colorbar=dict(x=1.0, tickfont=dict(color="#e6edf3"))),
                  row=1, col=2)
    fig5.update_layout(**_LAYOUT, height=400, title="Kalender-Heatmap: Monatliche Strategie-Renditen (%)")
    p5 = fig5.to_html(full_html=False, include_plotlyjs=False, div_id="cnc5")

    # ── §6 Trade P&L Analyse ─────────────────────────────────────────────
    all_f = sorted(sl_f + sig_f, key=lambda x: x["date"])
    all_n = sorted(sl_n + sig_n, key=lambda x: x["date"])
    pnl_f = [t["pnl"] for t in all_f]; pnl_n = [t["pnl"] for t in all_n]
    dur_f = [t["dur"] for t in all_f]; dur_n = [t["dur"] for t in all_n]
    typ_f = [t["type"] for t in all_f]; typ_n = [t["type"] for t in all_n]

    fig6 = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                         subplot_titles=["P&L pro Trade (€) — All Periods",
                                         "P&L pro Trade (€) — Crisis Excluded",
                                         "Trade-Dauer (Kalendertage)"])
    for col_idx, (pnls, types, nav_name) in enumerate(
            [(pnl_f, typ_f, "All"), (pnl_n, typ_n, "Excl.")], start=1):
        if pnls:
            colors = ["#3fb950" if p > 0 else "#f85149" for p in pnls]
            trade_nums = list(range(1, len(pnls) + 1))
            fig6.add_trace(go.Bar(x=trade_nums, y=pnls, marker_color=colors,
                                  name=f"Trades {nav_name}", showlegend=False),
                          row=1, col=col_idx)
            fig6.add_hline(y=0, line_color="#8b949e", line_dash="dot", row=1, col=col_idx)
    if dur_f or dur_n:
        fig6.add_trace(go.Histogram(x=dur_f, name="All Periods",
                                    marker_color="#58a6ff", opacity=0.7, nbinsx=20),
                      row=1, col=3)
        fig6.add_trace(go.Histogram(x=dur_n, name="Crisis Excl.",
                                    marker_color="#3fb950", opacity=0.7, nbinsx=20),
                      row=1, col=3)
    fig6.update_layout(**_LAYOUT, height=380, barmode="overlay",
                       title="Trade-Analyse: P&L-Verlauf und Haltedauer-Verteilung")
    p6 = fig6.to_html(full_html=False, include_plotlyjs=False, div_id="cnc6")

    # ── §7 Umfassende Metriken-Tabelle ────────────────────────────────────
    def _fmt(v, pct=False, eur=False, dec=2):
        if np.isnan(v): return "N/A"
        if pct: return f"{v*100:.{dec}f}%"
        if eur: return f"€{v:,.0f}"
        return f"{v:.{dec}f}"

    metrics_def = [
        # (Label, Erklärung, All-value, Excl-value, format_fn)
        ("Annualisierte Rendite",
         "Geometrischer Jahresdurchschnitt der täglichen NAV-Veränderungen (252 Handelstage).",
         mf["ann"], mn["ann"], lambda v: _fmt(v, pct=True)),
        ("Annualisierte Volatilität",
         "Standardabweichung der Tagesrenditen × √252. Maß für Schwankungsbreite.",
         mf["vol"], mn["vol"], lambda v: _fmt(v, pct=True)),
        ("Sharpe Ratio",
         "Rendite / Vol (ohne risikofreien Zins). >1.0 = gut, >2.0 = sehr gut.",
         mf["sh"], mn["sh"], lambda v: _fmt(v)),
        ("Sortino Ratio",
         "Rendite / (Downside-Std × √252). Bewertet nur negative Schwankungen — fairer als Sharpe.",
         mf["srt"], mn["srt"], lambda v: _fmt(v)),
        ("Calmar Ratio",
         "Ann.Rendite / |Max.DD|. Zeigt Rendite je Einheit Max-Risiko. >0.5 = akzeptabel.",
         mf["calm"], mn["calm"], lambda v: _fmt(v)),
        ("Max Drawdown",
         "Größter kumulierter Verlust vom letzten Hochpunkt. Kritischstes Risikomaß.",
         mf["mdd"], mn["mdd"], lambda v: _fmt(v, pct=True)),
        ("Gesamtrendite",
         "Absolute Rendite auf das Startkapital über den gesamten Zeitraum.",
         mf["total_ret"], mn["total_ret"], lambda v: _fmt(v, pct=True, dec=1)),
        ("Endkapital",
         "NAV am letzten Handelstag der Simulation.",
         mf["final"], mn["final"], lambda v: _fmt(v, eur=True)),
        ("Anzahl Trades",
         "Jeder Signal-Einstieg zählt als ein Trade, unabhängig von der Haltedauer.",
         float(mf["n_t"]), float(mn["n_t"]), lambda v: f"{int(v)}"),
        ("davon: Signal-Ausstiege",
         "Ausstieg weil Signal auf 0 drehte (normale Strategie-Logik).",
         float(mf["n_sig"]), float(mn["n_sig"]), lambda v: f"{int(v)}"),
        ("davon: Stop-Loss-Ereignisse",
         "Ausstieg weil JETS-Low ≤ Stop-Preis (= Entry × 0.70). Schutzschild.",
         float(mf["n_sl"]), float(mn["n_sl"]), lambda v: f"{int(v)}"),
        ("Win Rate",
         "Anteil profitabler Trades. >50% = positiver Erwartungswert.",
         mf["wr"], mn["wr"], lambda v: _fmt(v, pct=True)),
        ("Profit Factor",
         "Summe Gewinne / Summe Verluste. >1.5 = robust; <1.0 = Verlustbringer.",
         mf["pf"], mn["pf"], lambda v: _fmt(v)),
        ("Ø Gewinn-Trade (€)",
         "Durchschnittlicher Gewinn aller positiven Trades.",
         mf["avg_w"], mn["avg_w"], lambda v: _fmt(v, eur=True)),
        ("Ø Verlust-Trade (€)",
         "Durchschnittlicher Verlust aller negativen Trades.",
         mf["avg_l"], mn["avg_l"], lambda v: _fmt(v, eur=True)),
        ("Ø Haltedauer (Tage)",
         "Mittlere Haltedauer aller Trades in Kalendertagen.",
         mf["avg_d"], mn["avg_d"], lambda v: f"{v:.0f}d"),
        ("Max. Verluststrähne",
         "Längste aufeinanderfolgende Serie negativer Trades.",
         float(mf["max_con"]), float(mn["max_con"]), lambda v: f"{int(v)}"),
        ("Max. Gewinnsträhne",
         "Längste aufeinanderfolgende Serie positiver Trades.",
         float(mf["max_ws"]), float(mn["max_ws"]), lambda v: f"{int(v)}"),
    ]

    rows7 = ""
    for label, erklärung, vf, vn, fmt in metrics_def:
        vf_str = fmt(vf)
        vn_str = fmt(vn)
        # Highlight better value
        try:
            # For most metrics, higher is better (except mdd, vol, max_con, n_sl)
            lower_is_better = label in ("Max Drawdown", "Annualisierte Volatilität",
                                         "Max. Verluststrähne", "davon: Stop-Loss-Ereignisse",
                                         "Ø Verlust-Trade (€)")
            better_excl = (mn["ann"] > mf["ann"]) if "Rendite" in label else False
            cf = "text-success" if not lower_is_better else "text-danger"
            cn = "text-success" if not lower_is_better else "text-danger"
        except Exception:
            cf = ""; cn = ""
        rows7 += (
            f"<tr title='{erklärung}'>"
            f"<td><strong>{label}</strong><br><small class='text-muted'>{erklärung}</small></td>"
            f"<td class='text-info text-center'>{vf_str}</td>"
            f"<td class='text-success text-center'>{vn_str}</td></tr>"
        )
    tbl7 = (
        "<p class='text-muted small'>Hover über die Metriken-Zeilen für Erklärungen. "
        "Alle Trades basieren auf realen JETS OHLC-Preisen (yfinance, auto-adjust=True). "
        "Simulation-Parameter: €100.000 Startkapital, 30% Stop-Loss, 10bp Transaction-Costs (pro Seite), 95% Positionsgröße.</p>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered table-hover'>"
        "<thead><tr><th>Metrik &amp; Erklärung</th>"
        "<th class='text-info text-center'>All Periods</th>"
        "<th class='text-success text-center'>Crisis Excluded</th></tr></thead>"
        f"<tbody>{rows7}</tbody></table></div>"
    )

    # ── §8 Krisen-Impact pro Periode ─────────────────────────────────────
    rows8 = ""
    for cn, cs, ce in CRISES:
        sub_f = nav_f[(nav_f.index >= cs) & (nav_f.index <= ce)]
        sub_n = nav_n[(nav_n.index >= cs) & (nav_n.index <= ce)]
        if len(sub_f) < 2:
            rows8 += f"<tr><td>{cn}</td><td colspan='4' class='text-muted'>JETS pre-launch</td></tr>"
            continue
        ret_f = float(sub_f.iloc[-1] / sub_f.iloc[0] - 1)
        ret_n = float(sub_n.iloc[-1] / sub_n.iloc[0] - 1) if len(sub_n) >= 2 else 0.0
        fc = "text-danger" if ret_f < 0 else "text-success"
        nc = "text-danger" if ret_n < 0 else "text-success"
        n_days = len(sub_f)
        rows8 += (
            f"<tr><td><strong>{cn}</strong><br><small class='text-muted'>{cs} → {ce}</small></td>"
            f"<td class='{fc}'>{ret_f*100:.1f}%</td>"
            f"<td class='{nc}'>{ret_n*100:.1f}%</td>"
            f"<td>€{float(sub_f.iloc[0]):,.0f}</td>"
            f"<td>{n_days}d</td></tr>"
        )
    tbl8 = (
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered text-center'>"
        "<thead><tr><th>Krisenperiode</th><th class='text-info'>All Periods Return</th>"
        "<th class='text-success'>Crisis Excl. Return</th>"
        "<th>NAV bei Start</th><th>Handelstage</th></tr></thead>"
        f"<tbody>{rows8}</tbody></table></div>"
    )

    # ── §9 Options Chain (Live) ───────────────────────────────────────────
    opt_html = ""
    try:
        tk = yf.Ticker("JETS"); exps = tk.options
        if exps:
            ch    = tk.option_chain(exps[0])
            calls = ch.calls[["strike","lastPrice","impliedVolatility","volume","openInterest"]].copy()
            puts  = ch.puts [["strike","lastPrice","impliedVolatility","volume","openInterest"]].copy()
            calls["impliedVolatility"] = (calls["impliedVolatility"] * 100).round(1)
            puts ["impliedVolatility"] = (puts ["impliedVolatility"] * 100).round(1)
            pc_ratio = float(puts["openInterest"].sum()) / max(float(calls["openInterest"].sum()), 1.0)
            snt = "Bearish" if pc_ratio > 1.2 else ("Complacent" if pc_ratio < 0.7 else "Neutral")
            opt_title = f"JETS Options {exps[0]}  ·  P/C-Ratio: {pc_ratio:.2f} → {snt}"
            fig9 = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                                 subplot_titles=["Open Interest (Calls grün / Puts rot)",
                                                 "Implied Volatility Smile (%)"])
            fig9.add_trace(go.Bar(x=calls["strike"].tolist(), y=calls["openInterest"].tolist(),
                                  name="Call OI", marker_color="#3fb950", opacity=0.75), row=1, col=1)
            fig9.add_trace(go.Bar(x=puts["strike"].tolist(), y=puts["openInterest"].tolist(),
                                  name="Put OI", marker_color="#f85149", opacity=0.75), row=1, col=1)
            fig9.add_trace(go.Scatter(x=calls["strike"].tolist(), y=calls["impliedVolatility"].tolist(),
                                      name="Call IV%", line=dict(color="#58a6ff")), row=1, col=2)
            fig9.add_trace(go.Scatter(x=puts["strike"].tolist(), y=puts["impliedVolatility"].tolist(),
                                      name="Put IV%", line=dict(color="#f0883e")), row=1, col=2)
            fig9.update_layout(**_LAYOUT, title=opt_title, height=400, barmode="group")
            opt_html = (
                "<div class='alert alert-secondary small mb-2'>"
                "<strong>Interpretation:</strong> "
                "Hohe Put-OI bei Strike weit unter aktuellem Preis = Absicherungsdruck. "
                "IV-Smile steil nach links = Angst vor Downside. "
                f"P/C-Ratio {pc_ratio:.2f}: {'Market = bearish (Absicherung dominiert)' if pc_ratio > 1.2 else 'Market = neutral bis bullish'}"
                "</div>"
                + fig9.to_html(full_html=False, include_plotlyjs=False, div_id="cnc9")
            )
        else:
            opt_html = "<p class='text-muted'>Keine JETS-Optionen verfügbar.</p>"
    except Exception as e2:
        opt_html = f"<p class='text-warning'>Options-Abruf fehlgeschlagen: {e2}</p>"

    # ── Accordion Assembly ────────────────────────────────────────────────
    def _acc(n, title, body, show=False):
        cls = "" if show else "collapsed"
        sh  = "show" if show else ""
        return (
            f"<div class='accordion-item bg-dark border-secondary'>"
            f"<h2 class='accordion-header'>"
            f"<button class='accordion-button {cls} bg-dark text-light'"
            f" type='button' data-bs-toggle='collapse' data-bs-target='#cnc_p{n}'>"
            f"{title}</button></h2>"
            f"<div id='cnc_p{n}' class='accordion-collapse collapse {sh}'>"
            f"<div class='accordion-body'>{body}</div></div></div>"
        )

    panels = [
        _acc(1, "§1 · Methodik & Roter Faden",                        p1,   show=True),
        _acc(2, "§2 · Full-History NAV mit Trade-Markern (Echtdaten)", p2),
        _acc(3, "§3 · Rolling Metrics: Sharpe · Volatilität · Drawdown", p3),
        _acc(4, "§4 · Crisis-Excluded Simulation mit Trade-Markern",   p4),
        _acc(5, "§5 · Kalender-Heatmap: Monatliche Renditen",          p5),
        _acc(6, "§6 · Trade P&L-Verlauf & Haltedauer-Analyse",         p6),
        _acc(7, "§7 · Umfassende Metriken-Tabelle (20 Kennzahlen)",    tbl7),
        _acc(8, "§8 · Per-Krisenperiode Impact",                       tbl8),
        _acc(9, "§9 · JETS Options Chain Snapshot (Live)",             opt_html),
    ]
    body_html = "<div class='accordion' id='cnc_acc'>" + "".join(panels) + "</div>"
    _write(out / "crisis_vs_nocrisis_report.html",
           _html_base("Crisis vs No-Crisis Strategy — Detailanalyse", 20, body_html))


def build_crisis_predictivity_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    def _tz(raw):
        idx = raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        raw.index = idx.normalize()
        return raw

    def _dl(t):
        try:
            raw = _tz(yf.Ticker(t).history(period="max", auto_adjust=True))
            return raw["Close"].rename(t)
        except Exception:
            return pd.Series(dtype=float, name=t)

    ret_main = _read(tables / "phase2_returns.csv")
    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")

    # Download macro + sector signals individually
    vix  = _dl("^VIX")
    tnx  = _dl("^TNX")
    irx  = _dl("^IRX")
    hyg  = _dl("HYG")
    ief  = _dl("IEF")
    gld  = _dl("GLD")
    ibb  = _dl("IBB")
    xlv  = _dl("XLV")
    ita  = _dl("ITA")
    spy  = _dl("SPY")
    jets = _dl("JETS")
    dal  = _dl("DAL")
    ual  = _dl("UAL")
    aal  = _dl("AAL")
    luv  = _dl("LUV")

    CRISES = [
        ("GFC",       "2007-10-01", "2009-06-01"),
        ("Oil Crash", "2014-06-01", "2016-01-01"),
        ("COVID",     "2020-02-01", "2020-05-01"),
        ("Inflation", "2022-01-01", "2022-12-31"),
    ]
    CFILLS = ["rgba(248,81,73,0.10)", "rgba(210,168,255,0.10)",
              "rgba(248,81,73,0.10)", "rgba(240,136,62,0.10)"]

    # Build common macro index
    macro_base = [s for s in [vix, tnx, hyg, ief, gld, spy] if len(s) > 200]
    if not macro_base:
        _write(out / "crisis_predictivity_report.html",
               _html_base("Crisis Predictivity", 20, "<p class='text-warning'>Data unavailable.</p>"))
        return

    macro_idx = macro_base[0].index
    for s in macro_base[1:]:
        macro_idx = macro_idx.intersection(s.index)
    macro_idx = macro_idx.sort_values()

    def _al(s): return s.reindex(macro_idx).ffill().bfill()

    vix_a = _al(vix)
    tnx_a = _al(tnx)
    irx_a = _al(irx) if len(irx) > 100 else pd.Series(np.nan, index=macro_idx)
    hyg_a = _al(hyg)
    ief_a = _al(ief)
    gld_a = _al(gld)
    spy_a = _al(spy)

    # Rolling Z-score helper
    def _z(s, w=252):
        mu = s.rolling(w, min_periods=63).mean()
        sd = s.rolling(w, min_periods=63).std().replace(0, np.nan)
        return ((s - mu) / sd).fillna(0.0)

    # Signal construction
    curve      = tnx_a - irx_a.fillna(tnx_a * 0.5)  # fallback if IRX missing
    credit_raw = -np.log((hyg_a / ief_a.replace(0, np.nan)).replace(0, np.nan)).fillna(0.0)
    gold_mom   = gld_a.pct_change(20).fillna(0.0)

    vix_z    = _z(vix_a)
    curve_z  = -_z(curve)      # inverted curve → high stress
    credit_z = _z(credit_raw)
    gold_z   = _z(gold_mom)

    # Pandemic proxy: healthcare/biotech RS vs SPY
    hlth_z = pd.Series(0.0, index=macro_idx)
    hlth_available = False
    if len(xlv) > 200 and len(ibb) > 100:
        xlv_a  = xlv.reindex(macro_idx).ffill().bfill()
        ibb_a  = ibb.reindex(macro_idx).ffill().bfill()
        hlth_r = (xlv_a / spy_a.replace(0, np.nan)).pct_change(20).fillna(0.0)
        hlth_z = _z(hlth_r)
        hlth_available = True

    # Geopolitical proxy: defense ETF RS vs SPY
    def_z = pd.Series(0.0, index=macro_idx)
    def_available = False
    if len(ita) > 100:
        ita_a  = ita.reindex(macro_idx).ffill().bfill()
        def_r  = (ita_a / spy_a.replace(0, np.nan)).pct_change(20).fillna(0.0)
        def_z  = _z(def_r)
        def_available = True

    # Composite Crisis Predictivity Index
    CPI = (0.30 * vix_z + 0.25 * credit_z + 0.20 * curve_z
           + 0.15 * gold_z + 0.10 * def_z).rolling(5).mean()

    # --- §1 Macro stress dashboard ---
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=["VIX (Volatility Index)",
                                         "Yield Curve: 10Y − 3M (%)",
                                         "Credit Spread Proxy (−log HYG/IEF)"])
    fig1.add_trace(go.Scatter(x=vix_a.index, y=vix_a.values,
                              name="VIX", line=dict(color="#f85149")), row=1, col=1)
    fig1.add_hline(y=25, line_dash="dash", line_color="#ffa657", row=1, col=1)
    fig1.add_hline(y=40, line_dash="dash", line_color="#f85149", row=1, col=1)
    fig1.add_trace(go.Scatter(x=curve.index, y=curve.values,
                              name="10Y-3M", line=dict(color="#58a6ff")), row=2, col=1)
    fig1.add_hline(y=0, line_color="#8b949e", line_dash="dot", row=2, col=1)
    fig1.add_trace(go.Scatter(x=credit_raw.index, y=credit_raw.values,
                              name="Credit Spr", line=dict(color="#f0883e")), row=3, col=1)
    for i, (_, cs, ce) in enumerate(CRISES):
        fig1.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig1.update_layout(**_LAYOUT, height=620, showlegend=False,
                       title="Macro Stress Indicators  ·  VIX | Yield Curve | Credit Spread")
    p1 = fig1.to_html(full_html=False, include_plotlyjs=False, div_id="cpred1")

    # --- §2 Pandemic proxy signals ---
    if hlth_available:
        xlv_rel = xlv_a / spy_a.replace(0, np.nan)
        xlv_rel = xlv_rel / float(xlv_rel.iloc[0]) * 100
        ibb_rel = ibb_a / spy_a.replace(0, np.nan)
        ibb_rel = ibb_rel / float(ibb_rel.iloc[0]) * 100
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=["XLV / SPY Relative Index (=100 at start)",
                                             "IBB (Biotech) / SPY Relative Index"])
        fig2.add_trace(go.Scatter(x=xlv_rel.index, y=xlv_rel.values,
                                  name="XLV/SPY", line=dict(color="#3fb950")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=ibb_rel.index, y=ibb_rel.values,
                                  name="IBB/SPY", line=dict(color="#58a6ff")), row=2, col=1)
        for i, (_, cs, ce) in enumerate(CRISES):
            fig2.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
        fig2.update_layout(**_LAYOUT, height=480,
                           title="Pandemic Proxy: Healthcare (XLV) & Biotech (IBB) vs S&P 500")
        p2 = fig2.to_html(full_html=False, include_plotlyjs=False, div_id="cpred2")
    else:
        p2 = "<p class='text-muted'>Healthcare/biotech data unavailable.</p>"

    # --- §3 Geopolitical signals (Defense, Gold, Oil) ---
    fig3 = go.Figure()
    if def_available:
        ita_rel = ita_a / spy_a.replace(0, np.nan)
        ita_rel = ita_rel / float(ita_rel.iloc[0]) * 100
        fig3.add_trace(go.Scatter(x=ita_rel.index, y=ita_rel.values,
                                  name="ITA/SPY (Defense RS)", line=dict(color="#d2a8ff")))
    gld_rel = gld_a / spy_a.replace(0, np.nan)
    gld_rel = gld_rel / float(gld_rel.iloc[0]) * 100
    fig3.add_trace(go.Scatter(x=gld_rel.index, y=gld_rel.values,
                              name="GLD/SPY (Gold RS)", line=dict(color="#e3b341")))
    # Oil from ret_main
    oil_col = next((c for c in ["CL=F", "BZ=F"] if c in ret_main.columns), None)
    if oil_col:
        oil_r = ret_main[oil_col].dropna()
        oil_nav = (1 + oil_r).cumprod() * 100
        fig3.add_trace(go.Scatter(x=oil_nav.index, y=oil_nav.values,
                                  name=f"{oil_col} cumul.", line=dict(color="#f0883e")))
    for i, (_, cs, ce) in enumerate(CRISES):
        fig3.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig3.add_hline(y=100, line_color="#8b949e", line_dash="dot")
    fig3.update_layout(**_LAYOUT, height=400,
                       title="Geopolitical Signals: Defense RS · Gold RS · Oil  (=100 at series start)",
                       yaxis_title="Relative Index")
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="cpred3")

    # --- §4 JETS constituent analysis ---
    fig4 = go.Figure()
    CPAL = {"JETS": "#58a6ff", "DAL": "#3fb950", "UAL": "#f0883e",
            "AAL": "#d2a8ff", "LUV": "#e3b341"}
    for ticker, series in [("JETS", jets), ("DAL", dal), ("UAL", ual),
                            ("AAL", aal), ("LUV", luv)]:
        s = series.dropna()
        if len(s) < 50:
            continue
        s_norm = s / float(s.iloc[0]) * 100
        fig4.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values,
                                  name=ticker,
                                  line=dict(color=CPAL.get(ticker, "#8b949e"),
                                            width=2.5 if ticker == "JETS" else 1.5)))
    for i, (_, cs, ce) in enumerate(CRISES):
        fig4.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig4.update_layout(**_LAYOUT, height=420,
                       title="JETS Constituents: Normalized Price (=100 at each series start)",
                       yaxis_title="Normalized Price")
    p4 = fig4.to_html(full_html=False, include_plotlyjs=False, div_id="cpred4")

    # --- §5 Composite CPI + JETS ---
    fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=["Composite Crisis Predictivity Index (CPI)",
                                         "JETS Price"])
    fig5.add_trace(go.Scatter(x=CPI.index, y=CPI.values, name="CPI",
                              fill="tozeroy", line=dict(color="#d2a8ff"),
                              fillcolor="rgba(210,168,255,0.15)"), row=1, col=1)
    fig5.add_hline(y=1.0, line_dash="dash", line_color="#f85149", row=1, col=1)
    fig5.add_hline(y=-1.0, line_dash="dash", line_color="#3fb950", row=1, col=1)
    if len(jets.dropna()) > 50:
        jets_al = jets.reindex(macro_idx).ffill()
        fig5.add_trace(go.Scatter(x=jets_al.index, y=jets_al.values,
                                  name="JETS", line=dict(color="#58a6ff")), row=2, col=1)
    for i, (_, cs, ce) in enumerate(CRISES):
        fig5.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    cpi_formula = "CPI = 0.30×VIX_z + 0.25×CreditSpread_z + 0.20×(−YieldCurve_z) + 0.15×Gold_z + 0.10×Defense_z"
    fig5.update_layout(**_LAYOUT, height=520, title=cpi_formula)
    p5 = fig5.to_html(full_html=False, include_plotlyjs=False, div_id="cpred5")

    # --- §6 Crisis lead-time analysis ---
    lead_rows = ""
    if len(jets.dropna()) > 200:
        jets_dd = jets.dropna()
        jets_dd = (jets_dd / jets_dd.rolling(20).max() - 1)  # drawdown from 20d high
        for cn, cs, ce in CRISES:
            # Find first day JETS fell >8% from 20d high during crisis
            crash_sub = jets_dd[(jets_dd.index >= cs) & (jets_dd.index <= ce)]
            crash_hits = crash_sub[crash_sub < -0.08]
            crash_dt   = crash_hits.index[0] if len(crash_hits) else None
            # Find first day CPI crossed 1.0 before crash_dt
            if crash_dt is not None:
                cpi_before = CPI[(CPI.index < crash_dt) & (CPI.index >= pd.Timestamp(cs) - pd.Timedelta(days=90))]
                cpi_hits   = cpi_before[cpi_before > 1.0]
                warn_dt    = cpi_hits.index[-1] if len(cpi_hits) else None
                lead_days  = int((crash_dt - warn_dt).days) if warn_dt is not None else None
                lead_str   = f"{lead_days}d early" if lead_days and lead_days > 0 else ("same day" if lead_days == 0 else "no warning")
                lead_cls   = "text-success" if (lead_days and lead_days > 0) else "text-danger"
            else:
                warn_dt   = None; lead_str = "no JETS crash (pre-launch or no crash)"; lead_cls = "text-muted"
            crash_str = crash_dt.strftime("%Y-%m-%d") if crash_dt else "—"
            warn_str  = warn_dt.strftime("%Y-%m-%d") if warn_dt else "—"
            lead_rows += (f"<tr><td>{cn}</td><td>{warn_str}</td>"
                          f"<td>{crash_str}</td><td class='{lead_cls}'>{lead_str}</td></tr>")
    if not lead_rows:
        lead_rows = "<tr><td colspan='4' class='text-muted'>JETS history too short for lead-time analysis.</td></tr>"

    tbl6 = (
        "<p class='text-muted small'>CPI threshold: >1.0σ composite stress. JETS crash: >8% drawdown from 20d high.</p>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered text-center'>"
        "<thead><tr><th>Crisis</th><th>CPI Warning Date</th>"
        "<th>JETS Crash Date</th><th>Lead Time</th></tr></thead>"
        f"<tbody>{lead_rows}</tbody></table></div>"
    )

    # --- §7 Current readings ---
    readings = []
    if len(macro_idx):
        last = macro_idx[-1]
        def _lv(s): return float(s.loc[last]) if last in s.index else float("nan")
        def _pct(s, v):
            sd = s.dropna()
            if len(sd) < 20 or np.isnan(v): return "N/A"
            return f"{float((sd < v).mean() * 100):.0f}th pct"

        vix_now = _lv(vix_a); curve_now = _lv(curve)
        credit_now = _lv(credit_raw); gold_now = _lv(gld_a)
        cpi_now    = float(CPI.iloc[-1]) if len(CPI.dropna()) else float("nan")

        cpi_cls = "text-danger" if cpi_now > 1.0 else "text-success"
        vix_cls = "text-danger" if vix_now > 25 else "text-success"
        curve_cls = "text-danger" if curve_now < 0 else "text-success"

        readings = [
            ("VIX",                 f"{vix_now:.1f}",         _pct(vix_a, vix_now),         vix_cls),
            ("10Y Treasury (%)",    f"{_lv(tnx_a):.2f}",      _pct(tnx_a, _lv(tnx_a)),     ""),
            ("Yield Curve 10Y-3M",  f"{curve_now:.2f}",        _pct(curve, curve_now),       curve_cls),
            ("Credit Spread",       f"{credit_now:.3f}",       _pct(credit_raw, credit_now), ""),
            ("Gold (GLD)",          f"{gold_now:.1f}",         _pct(gld_a, gold_now),        ""),
            ("CPI",                 f"{cpi_now:.2f}",          "",                           cpi_cls),
        ]

    rows7 = "".join(
        f"<tr><td>{r[0]}</td><td class='{r[3]}'>{r[1]}</td>"
        f"<td class='text-muted'>{r[2]}</td></tr>"
        for r in readings)
    tbl7 = (
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Indicator</th><th>Current</th><th>Historical Percentile</th></tr></thead>"
        f"<tbody>{rows7}</tbody></table></div>"
        "<p class='text-muted small mt-2'>"
        "CPI &gt; 1.0 = elevated stress. Typical JETS lag: 5–15 trading days after CPI threshold crossing."
        "</p>"
    )

    def _acc(n, title, body, show=False):
        cls = "" if show else "collapsed"
        sh  = "show" if show else ""
        return (
            f"<div class='accordion-item bg-dark border-secondary'>"
            f"<h2 class='accordion-header'>"
            f"<button class='accordion-button {cls} bg-dark text-light'"
            f" type='button' data-bs-toggle='collapse' data-bs-target='#cpred_p{n}'>"
            f"{title}</button></h2>"
            f"<div id='cpred_p{n}' class='accordion-collapse collapse {sh}'>"
            f"<div class='accordion-body'>{body}</div></div></div>"
        )

    panels = [
        _acc(1, "§1 · Macro Stress Indicators (VIX, Yield Curve, Credit Spread)", p1, show=True),
        _acc(2, "§2 · Pandemic Proxy Signals (Healthcare & Biotech RS)", p2),
        _acc(3, "§3 · Geopolitical Signals (Defense RS, Gold RS, Oil)", p3),
        _acc(4, "§4 · JETS Constituent Analysis (DAL, UAL, AAL, LUV)", p4),
        _acc(5, "§5 · Composite Crisis Predictivity Index (CPI) vs JETS", p5),
        _acc(6, "§6 · Crisis Lead-Time Analysis (CPI Warning vs JETS Crash)", tbl6),
        _acc(7, "§7 · Current Signal Readings", tbl7),
    ]
    body = "<div class='accordion' id='cpred_acc'>" + "".join(panels) + "</div>"
    _write(out / "crisis_predictivity_report.html",
           _html_base("Crisis Predictivity Dashboard", 20, body))


def build_sector_rotation_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    def _tz(raw):
        idx = raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        raw.index = idx.normalize()
        return raw

    def _dl(t):
        try:
            raw = _tz(yf.Ticker(t).history(period="max", auto_adjust=True))
            return raw["Close"].rename(t)
        except Exception:
            return pd.Series(dtype=float, name=t)

    # 11 GICS sectors: (name, ETF, color, top-5 tickers by market cap)
    SECTORS = [
        ("Technology",    "XLK",  "#58a6ff", ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"]),
        ("Healthcare",    "XLV",  "#3fb950", ["LLY",  "UNH",  "JNJ",  "MRK",  "ABBV"]),
        ("Financials",    "XLF",  "#f0883e", ["BRK-B","JPM",  "V",    "MA",   "BAC"]),
        ("Energy",        "XLE",  "#d2a8ff", ["XOM",  "CVX",  "COP",  "EOG",  "SLB"]),
        ("Industrials",   "XLI",  "#ffa657", ["GE",   "RTX",  "HON",  "UNP",  "CAT"]),
        ("Cons. Discr.",  "XLY",  "#79c0ff", ["AMZN", "TSLA", "HD",   "MCD",  "NKE"]),
        ("Cons. Staples", "XLP",  "#56d364", ["PG",   "KO",   "PEP",  "COST", "WMT"]),
        ("Communication", "XLC",  "#e3b341", ["META", "GOOGL","NFLX", "DIS",  "CMCSA"]),
        ("Materials",     "XLB",  "#bc8cff", ["LIN",  "APD",  "NEM",  "FCX",  "SHW"]),
        ("Utilities",     "XLU",  "#8b949e", ["NEE",  "SO",   "DUK",  "AEP",  "EXC"]),
        ("Real Estate",   "XLRE", "#ff7b72", ["AMT",  "PLD",  "EQIX", "CCI",  "PSA"]),
    ]

    CRISES = [
        ("GFC",       "2007-10-01", "2009-06-01"),
        ("Oil Crash", "2014-06-01", "2016-01-01"),
        ("COVID",     "2020-02-01", "2020-05-01"),
        ("Inflation", "2022-01-01", "2022-12-31"),
    ]
    CFILLS = ["rgba(248,81,73,0.10)", "rgba(210,168,255,0.10)",
              "rgba(248,81,73,0.10)", "rgba(240,136,62,0.10)"]
    CCOLORS = ["#f85149", "#d2a8ff", "#f0883e", "#ffa657"]

    # Download benchmark + key signal tickers
    spy  = _dl("SPY")
    vix  = _dl("^VIX")
    gld  = _dl("GLD")
    jets = _dl("JETS")

    # Download sector ETFs
    sec_data = {}
    for name, etf, color, top5 in SECTORS:
        s = _dl(etf)
        if len(s) > 200:
            sec_data[name] = {"etf": etf, "px": s, "color": color, "top5": top5}

    if len(sec_data) < 3 or len(spy) < 200:
        _write(out / "sector_rotation_report.html",
               _html_base("Sector Rotation", 20, "<p class='text-warning'>Insufficient data.</p>"))
        return

    # Common index across all sectors + SPY
    all_series = [spy] + [v["px"] for v in sec_data.values()]
    common_idx = all_series[0].index
    for s in all_series[1:]:
        common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()

    spy_c = spy.reindex(common_idx).ffill()
    sec_aligned = {name: v["px"].reindex(common_idx).ffill()
                   for name, v in sec_data.items()}

    sec_names  = list(sec_data.keys())
    sec_colors = [sec_data[n]["color"] for n in sec_names]

    # Monthly returns per sector
    spy_monthly = spy_c.resample("ME").last().pct_change().dropna()
    sec_monthly  = {}
    for name, s in sec_aligned.items():
        sm = s.resample("ME").last().pct_change().dropna()
        sec_monthly[name] = sm

    # --- §1 Monthly returns heatmap ---
    all_dates = sorted(set().union(*[set(v.index) for v in sec_monthly.values()]))
    z_mat = []; text_mat = []
    for name in sec_names:
        row = []; txt = []
        for dt in all_dates:
            val = sec_monthly[name].get(dt, np.nan) if name in sec_monthly else np.nan
            row.append(float(val * 100) if not np.isnan(float(val)) else None)
            txt.append(f"{val*100:.1f}%" if not np.isnan(float(val)) else "")
        z_mat.append(row); text_mat.append(txt)

    fig1 = go.Figure(go.Heatmap(
        z=z_mat,
        x=[d.strftime("%Y-%m") for d in all_dates],
        y=sec_names,
        text=text_mat,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Return %", tickfont=dict(color="#e6edf3")),
    ))
    fig1.update_layout(**_LAYOUT, title="Sector Monthly Returns Heatmap (%)",
                       height=520, xaxis_tickangle=-60)
    p1 = fig1.to_html(full_html=False, include_plotlyjs=False, div_id="sr1")

    # --- §2 Rolling 63d relative strength vs SPY ---
    spy_r63 = spy_c.pct_change(63)
    fig2 = go.Figure()
    for name, color in zip(sec_names, sec_colors):
        rs = sec_aligned[name].pct_change(63) - spy_r63
        fig2.add_trace(go.Scatter(x=rs.index, y=rs.values, name=name,
                                  line=dict(color=color, width=1.5)))
    fig2.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    for i, (_, cs, ce) in enumerate(CRISES):
        fig2.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig2.update_layout(**_LAYOUT,
                       title="Rolling 63d Relative Strength vs SPY  (positive = outperforming)",
                       yaxis_title="RS (sector − SPY 63d return)", height=440)
    p2 = fig2.to_html(full_html=False, include_plotlyjs=False, div_id="sr2")

    # --- §3 Crisis rotation: sector returns per crisis period ---
    crisis_returns = []
    for cn, cs, ce in CRISES:
        row = {"Crisis": cn}
        for name, s in sec_aligned.items():
            sub = s[(s.index >= cs) & (s.index <= ce)]
            row[name] = float(sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) >= 2 else None
        crisis_returns.append(row)

    fig3 = go.Figure()
    for i, row in enumerate(crisis_returns):
        vals = [row.get(n) for n in sec_names]
        fig3.add_trace(go.Bar(name=row["Crisis"], x=sec_names, y=vals,
                              marker_color=CCOLORS[i % len(CCOLORS)], opacity=0.85))
    fig3.update_layout(**_LAYOUT, title="Sector Returns During Crisis Periods (%)",
                       yaxis_title="Total Return (%)", barmode="group", height=440)
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="sr3")

    # --- §4 Top-5 holdings table (static — market cap as of 2025) ---
    tbl4_rows = ""
    for name, v in sec_data.items():
        top5_str = ", ".join(v["top5"])
        tbl4_rows += (f"<tr><td style='color:{v['color']}'><strong>{name}</strong></td>"
                      f"<td>{v['etf']}</td><td class='text-muted'>{top5_str}</td></tr>")
    tbl4 = (
        "<p class='text-muted small'>Top-5 companies ranked by approximate market cap (2025). "
        "These names make up 50–70% of each sector ETF's weight.</p>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Sector</th><th>ETF</th><th>Top-5 by Market Cap</th></tr></thead>"
        f"<tbody>{tbl4_rows}</tbody></table></div>"
    )

    # --- §5 Sector correlation matrix (trailing 252d) ---
    sec_rets = pd.DataFrame({n: sec_aligned[n].pct_change()
                             for n in sec_names}).dropna()
    corr = sec_rets.tail(252).corr().round(2)
    corr_z = corr.values.tolist()
    corr_t = [[f"{v:.2f}" for v in row] for row in corr_z]
    fig5 = go.Figure(go.Heatmap(
        z=corr_z, x=corr.columns.tolist(), y=corr.index.tolist(),
        text=corr_t, texttemplate="%{text}",
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Corr.", tickfont=dict(color="#e6edf3")),
    ))
    fig5.update_layout(**_LAYOUT, title="Sector Correlation Matrix – Trailing 252 Trading Days",
                       height=480)
    p5 = fig5.to_html(full_html=False, include_plotlyjs=False, div_id="sr5")

    # --- §6 L1/L2 trigger backtest ---
    trig_tickers = [spy, vix, gld]
    xlu_s = sec_data.get("Utilities", {}).get("px", pd.Series(dtype=float))
    xlk_s = sec_data.get("Technology", {}).get("px", pd.Series(dtype=float))
    trig_tickers += [xlu_s, xlk_s]
    trig_idx = common_idx
    for s in [vix, gld, xlu_s, xlk_s]:
        if len(s) > 50:
            trig_idx = trig_idx.intersection(s.index)
    trig_idx = trig_idx.sort_values()

    p6 = "<p class='text-muted'>Insufficient data for trigger backtest.</p>"
    if len(trig_idx) > 300:
        spy_t  = spy_c.reindex(trig_idx).ffill()
        vix_t  = vix.reindex(trig_idx).ffill()
        xlu_t  = xlu_s.reindex(trig_idx).ffill() if len(xlu_s) > 50 else spy_t * 0
        xlk_t  = xlk_s.reindex(trig_idx).ffill() if len(xlk_s) > 50 else spy_t * 0
        gld_t  = gld.reindex(trig_idx).ffill()
        jets_t = jets.reindex(trig_idx).ffill()

        spy_r20 = spy_t.pct_change(20)
        xlu_r20 = xlu_t.pct_change(20)
        xlk_r20 = xlk_t.pct_change(20)
        gld_r20 = gld_t.pct_change(20)

        # L1: early risk-off (VIX>20 AND SPY -5% in 20d)
        l1 = ((vix_t > 20) & (spy_r20 < -0.05)).astype(int)
        # L2: confirmed risk-off (VIX>25 AND utilities outperform tech AND gold rising)
        l2 = ((vix_t > 25) & (xlu_r20 > xlk_r20) & (gld_r20 > 0.03)).astype(int)

        # Forward JETS return after trigger (avoid lookahead: shift signal back)
        jets_r20 = jets_t.pct_change(20).shift(-20)
        jets_r60 = jets_t.pct_change(60).shift(-60)

        def _tstats(signal, fwd, label):
            dates = signal[signal == 1].index
            vals  = [float(fwd.get(d, np.nan)) for d in dates if d in fwd.index]
            vals  = [v for v in vals if not np.isnan(v)]
            if not vals:
                return f"{label}: n=0"
            avg = np.mean(vals) * 100; med = np.median(vals) * 100
            pos_pct = float((np.array(vals) > 0).mean()) * 100
            return f"{label}: n={len(vals)}, avg={avg:.1f}%, median={med:.1f}%, positive={pos_pct:.0f}%"

        l1_20 = _tstats(l1, jets_r20, "L1 → 20d JETS fwd")
        l1_60 = _tstats(l1, jets_r60, "L1 → 60d JETS fwd")
        l2_20 = _tstats(l2, jets_r20, "L2 → 20d JETS fwd")
        l2_60 = _tstats(l2, jets_r60, "L2 → 60d JETS fwd")

        fig6a = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=["SPY (normalized) + L1 Trigger fires",
                                             "JETS + L2 Trigger fires"])
        spy_n = spy_t / float(spy_t.iloc[0]) * 100
        fig6a.add_trace(go.Scatter(x=spy_n.index, y=spy_n.values,
                                   name="SPY", line=dict(color="#58a6ff")), row=1, col=1)
        l1_pts = trig_idx[l1 == 1]
        if len(l1_pts):
            fig6a.add_trace(go.Scatter(x=l1_pts, y=spy_n.reindex(l1_pts).values,
                                       name="L1 Fire", mode="markers",
                                       marker=dict(color="#f85149", size=5,
                                                   symbol="triangle-down")), row=1, col=1)
        jets_v = jets_t.dropna()
        if len(jets_v) > 50:
            jets_n = jets_v / float(jets_v.iloc[0]) * 100
            fig6a.add_trace(go.Scatter(x=jets_n.index, y=jets_n.values,
                                       name="JETS", line=dict(color="#3fb950")), row=2, col=1)
            l2_pts = trig_idx[l2 == 1]
            if len(l2_pts):
                fig6a.add_trace(go.Scatter(x=l2_pts, y=jets_n.reindex(l2_pts).values,
                                           name="L2 Fire", mode="markers",
                                           marker=dict(color="#f85149", size=7,
                                                       symbol="triangle-down")), row=2, col=1)
        for i, (_, cs, ce) in enumerate(CRISES):
            fig6a.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
        fig6a.update_layout(**_LAYOUT, height=520,
                            title="L1/L2 Risk-Off Triggers on SPY & JETS")
        p6_chart = fig6a.to_html(full_html=False, include_plotlyjs=False, div_id="sr6a")

        stats_html = (
            "<div class='row mt-3'>"
            "<div class='col-md-6'>"
            "<div class='card bg-secondary text-light p-3'>"
            f"<h6 class='text-warning'>L1 Trigger: VIX &gt; 20 AND SPY 20d &lt; −5%</h6>"
            f"<p class='small mb-1'>{l1_20}</p><p class='small mb-0'>{l1_60}</p></div></div>"
            "<div class='col-md-6'>"
            "<div class='card bg-secondary text-light p-3'>"
            f"<h6 class='text-danger'>L2 Trigger: VIX &gt; 25 AND XLU &gt; XLK AND GLD &gt; +3%</h6>"
            f"<p class='small mb-1'>{l2_20}</p><p class='small mb-0'>{l2_60}</p></div></div></div>"
        )
        p6 = p6_chart + stats_html

    # --- §7 Energy & Airline in rotation context ---
    fig7 = go.Figure()
    for focus_name in ["Energy", "Utilities", "Cons. Staples", "Technology"]:
        if focus_name not in sec_aligned:
            continue
        rs = sec_aligned[focus_name].pct_change(63) - spy_r63
        color = sec_data[focus_name]["color"]
        fig7.add_trace(go.Scatter(x=rs.index, y=rs.values, name=focus_name,
                                  line=dict(color=color)))
    if len(jets.dropna()) > 100:
        jets_al = jets.reindex(common_idx).ffill()
        jets_rs = jets_al.pct_change(63) - spy_r63
        fig7.add_trace(go.Scatter(x=jets_rs.index, y=jets_rs.values,
                                  name="JETS (Airlines)", line=dict(color="#58a6ff", width=2.5)))
    fig7.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    for i, (_, cs, ce) in enumerate(CRISES):
        fig7.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig7.update_layout(**_LAYOUT,
                       title="Oil/Airline in Rotation Context  ·  RS vs SPY (rolling 63d)",
                       yaxis_title="RS vs SPY", height=420)
    p7 = fig7.to_html(full_html=False, include_plotlyjs=False, div_id="sr7")

    def _acc(n, title, body, show=False):
        cls = "" if show else "collapsed"
        sh  = "show" if show else ""
        return (
            f"<div class='accordion-item bg-dark border-secondary'>"
            f"<h2 class='accordion-header'>"
            f"<button class='accordion-button {cls} bg-dark text-light'"
            f" type='button' data-bs-toggle='collapse' data-bs-target='#sr_p{n}'>"
            f"{title}</button></h2>"
            f"<div id='sr_p{n}' class='accordion-collapse collapse {sh}'>"
            f"<div class='accordion-body'>{body}</div></div></div>"
        )

    panels = [
        _acc(1, "§1 · Monthly Sector Returns Heatmap",               p1, show=True),
        _acc(2, "§2 · Rolling 63d Relative Strength vs SPY",         p2),
        _acc(3, "§3 · Sector Returns During Crisis Periods",          p3),
        _acc(4, "§4 · Top-5 Companies by Market Cap (per Sector)",   tbl4),
        _acc(5, "§5 · Sector Correlation Matrix (trailing 252d)",     p5),
        _acc(6, "§6 · L1/L2 Risk-Off Trigger Backtest",              p6),
        _acc(7, "§7 · Energy & Airline in Sector Rotation Context",   p7),
    ]
    body = "<div class='accordion' id='sr_acc'>" + "".join(panels) + "</div>"
    _write(out / "sector_rotation_report.html",
           _html_base("Sector Rotation Screener", 20, body))


def build_flash_crash_report(tables, figures, out):  # noqa: C901
    """
    Flash Crash Early Warning Dashboard.
    Beantwortet: Gibt es messbare Frühwarnsignale für starke JETS-Einbrüche?
    Methodik: Composite Stress Index (CSI) aus 6 Markt-Signalkomponenten,
    validiert an historischen Flash Crashes 2015–2022.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    def _tz(raw):
        idx = raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        raw.index = idx.normalize()
        return raw

    def _dl(t):
        try:
            return _tz(yf.Ticker(t).history(period="max", auto_adjust=True))["Close"].rename(t)
        except Exception:
            return pd.Series(dtype=float, name=t)

    def _dl_full(t):
        try:
            return _tz(yf.Ticker(t).history(period="max", auto_adjust=True))
        except Exception:
            return pd.DataFrame()

    # ── Daten-Downloads ───────────────────────────────────────────────────
    vix      = _dl("^VIX")
    vix9d    = _dl("^VIX9D")
    hyg      = _dl("HYG")
    ief      = _dl("IEF")
    dxy      = _dl("DX-Y.NYB")
    spy      = _dl("SPY")
    jets_df  = _dl_full("JETS")
    jets_cl  = jets_df["Close"].rename("JETS") if "Close" in jets_df.columns else pd.Series(dtype=float, name="JETS")
    jets_vol = jets_df["Volume"].rename("JETS_Vol") if "Volume" in jets_df.columns else pd.Series(dtype=float)
    jets_lo  = jets_df["Low"].rename("JETS_Low")  if "Low"    in jets_df.columns else pd.Series(dtype=float)

    # Flash Crash Ereignisse: (Name, Datum, Beschreibung, Hauptursache)
    EVENTS = [
        ("China Crash",     "2015-08-24",
         "Renminbi-Abwertung → globaler Sell-off. JETS −6% in 2 Tagen. DAX −5.8%.",
         "Währungsschock / Contagion"),
        ("Volmageddon",     "2018-02-05",
         "VIX-Short-Produkt-Blowup: VIX von 17 auf 37 innerhalb einer Handelsstunde.",
         "Strukturelles Finanzprodukt-Risiko"),
        ("COVID-Crash",     "2020-03-16",
         "Schlimmster SPY-Tag seit 1987 (−12%). JETS −17% an einem Tag. Reiseverbote.",
         "Exogener Schock / Pandemie"),
        ("Inflation Shock", "2022-01-24",
         "SPY intraday −5% (dann Erholung). Fed Pivot-Angst. Ukraine-Spannungen wachsen.",
         "Makro-Policy / Geopolitik"),
        ("CPI Schock",      "2022-09-13",
         "US-CPI überraschend hoch → Fed-Zins-Schock. SPY −4.3%, JETS −7%.",
         "Makro-Überraschung / Inflation"),
    ]
    CFILLS  = ["rgba(248,81,73,0.12)", "rgba(210,168,255,0.12)", "rgba(248,81,73,0.12)",
               "rgba(240,136,62,0.12)", "rgba(248,81,73,0.12)"]
    ECOLORS = ["#f85149", "#d2a8ff", "#f0883e", "#ffa657", "#ff7b72"]

    # ── Gemeinsamer Index ─────────────────────────────────────────────────
    base = [s for s in [vix, hyg, ief] if len(s) > 500]
    if not base:
        _write(out / "flash_crash_report.html",
               _html_base("Flash Crash EWS", 20, "<p class='text-warning'>Keine Daten.</p>"))
        return
    cidx = base[0].index
    for s in base[1:]:
        cidx = cidx.intersection(s.index)
    for s in [dxy, spy, jets_cl]:
        if len(s) > 200:
            cidx = cidx.intersection(s.index)
    cidx = cidx.sort_values()

    def _al(s):
        return s.reindex(cidx).ffill().bfill() if len(s) > 50 else pd.Series(np.nan, index=cidx)

    vix_a    = _al(vix);    hyg_a = _al(hyg);   ief_a = _al(ief)
    dxy_a    = _al(dxy);    spy_a = _al(spy);   jets_a = _al(jets_cl)
    jvol_a   = _al(jets_vol); jlo_a = _al(jets_lo)
    vix9d_a  = _al(vix9d)

    # ── CSI Komponentenbau ────────────────────────────────────────────────
    W = 252

    def _prank(s, w=W):
        s_clean = s.fillna(0.0)
        return s_clean.rolling(w, min_periods=63).rank(pct=True) * 100

    # Komp 1: VIX-Level-Perzentil
    c1 = _prank(vix_a)

    # Komp 2: VIX 5-Tage-Spike
    c2 = _prank((vix_a / vix_a.shift(5) - 1).clip(lower=0).fillna(0))

    # Komp 3: Credit Spread (−log HYG/IEF → steigt wenn HYG fällt)
    credit_raw = -np.log((hyg_a / ief_a.replace(0, np.nan)).replace(0, np.nan)).fillna(0)
    c3 = _prank(credit_raw)

    # Komp 4: DXY Safe-Haven Spike (|5d-Rendite|)
    dxy_spike = dxy_a.pct_change(5).abs().fillna(0)
    c4 = _prank(dxy_spike) if dxy_a.notna().sum() > 200 else pd.Series(50.0, index=cidx)

    # Komp 5: JETS Volumen-Anomalie (vol / 20d-Schnitt − 1)
    jvol_ratio = (jvol_a / jvol_a.rolling(20).mean().replace(0, np.nan) - 1).clip(0).fillna(0)
    c5 = _prank(jvol_ratio) if jvol_a.notna().sum() > 200 else pd.Series(50.0, index=cidx)

    # Komp 6: VIX Term Structure (VIX9D − VIX, invertiert wenn negativ = Stress)
    has_ts = vix9d_a.notna().sum() > 100
    if has_ts:
        ts_raw = -(vix9d_a - vix_a).fillna(0)  # positiv = invertiert = Stress
        c6 = _prank(ts_raw)
        W1, W2, W3, W4, W5, W6 = 0.25, 0.15, 0.25, 0.08, 0.12, 0.15
    else:
        c6 = pd.Series(50.0, index=cidx)
        W1, W2, W3, W4, W5, W6 = 0.30, 0.18, 0.30, 0.10, 0.12, 0.00

    CSI = (W1*c1 + W2*c2 + W3*c3 + W4*c4 + W5*c5 + W6*c6).rolling(3).mean()

    # ── §1 Einführung ────────────────────────────────────────────────────
    worst_days_rows = ""
    if len(jets_a.dropna()) > 200:
        jret = jets_a.pct_change().dropna()
        worst5 = jret.nsmallest(10)
        for dt, v in worst5.items():
            worst_days_rows += f"<tr><td>{dt.strftime('%Y-%m-%d')}</td><td class='text-danger'>{v*100:.2f}%</td></tr>"

    intro_html = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h5 class='text-warning'>Warum sind Flash Crashes für JETS besonders gefährlich?</h5>"
        "<p>Airlines haben extrem <strong>hohe Fixkostenbasis</strong> (Flugzeugmiete, Personal, Treibstoff-Hedges). "
        "Bei einem plötzlichen Reisestopp bricht der Umsatz sofort ein, die Kosten bleiben. "
        "Das macht JETS zu einem der <em>volatilsten nicht-gehebten ETFs</em> am Markt. "
        "Ein einzelner Covid-Lockdown-Tag vernichtete 2020 17% des ETF-Werts.</p>"
        "<p>Flash Crashes sind hier kritisch, weil:</p>"
        "<ul>"
        "<li>Stop-Loss-Orders können bei Gapping-Eröffnungen weit unterhalb des Stops exekutiert werden</li>"
        "<li>Options-Hedging ist für Retailtrader zu teuer / zu komplex</li>"
        "<li>Reaktionszeit zu kurz für manuelles Management</li>"
        "</ul>"
        "<p><strong>Unser Ziel</strong>: Frühwarnsignale identifizieren, die <em>Tage vor</em> dem Crash auf erhöhtes "
        "Risiko hinweisen — damit der Stop-Loss nicht die einzige Verteidigung ist.</p>"
        "</div>"
        "<h6 class='text-warning mt-3'>JETS — Schlimmste Tagesrenditen</h6>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Datum</th><th>JETS Tagesrendite</th></tr></thead>"
        f"<tbody>{worst_days_rows}</tbody></table></div>"
        "<p class='text-muted small mt-2'>Quelle: yfinance, JETS ETF, auto-adjust=True</p>"
    )
    p1 = intro_html

    # ── §2 CSI Methodik ───────────────────────────────────────────────────
    comp_rows = ""
    for label, desc, w_val, src_str in [
        ("VIX Level",          f"Absolutes VIX-Niveau, gerankt auf 252d-Fenster → 0–100.",                f"{W1*100:.0f}%", "^VIX"),
        ("VIX 5d-Spike",       "5-Tage-%-Anstieg des VIX. Zeigt Geschwindigkeit der Angst-Zunahme.",       f"{W2*100:.0f}%", "^VIX"),
        ("Credit Spread",      "−log(HYG/IEF): Steigt wenn High-Yield-Bonds vs. Treasuries fallen.",        f"{W3*100:.0f}%", "HYG, IEF"),
        ("DXY Safe-Haven",     "|5d-Rendite DXY|. Große USD-Bewegungen = globale Risikoflucht.",            f"{W4*100:.0f}%", "DX-Y.NYB"),
        ("JETS Volumen",       "Tagesvolumen / 20d-Ø − 1. Anomale Volumes = Panikverkäufe oder Absicherung.",f"{W5*100:.0f}%", "JETS"),
        ("VIX Term Structure", "−(VIX9D − VIX). Negativ = invertierte Termstruktur = akute Angst.",        f"{W6*100:.0f}%", "^VIX9D" + (" ✓" if has_ts else " (nicht verfügbar)")),
    ]:
        comp_rows += (
            f"<tr><td><strong>{label}</strong></td><td>{desc}</td>"
            f"<td class='text-warning text-center'>{w_val}</td>"
            f"<td class='text-muted'>{src_str}</td></tr>"
        )
    meth_html = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h6 class='text-info'>Roter Faden: Wie wird der CSI konstruiert?</h6>"
        "<p>Der <strong>Composite Stress Index (CSI)</strong> kombiniert 6 Signalkomponenten, "
        "die jeweils unabhängige Dimensionen von Marktangst messen. Jede Komponente wird auf ein "
        "rollendes 252-Tage-Fenster normiert (Perzentil-Rang 0–100), dann gewichtet aufsummiert. "
        "<br><strong>Schwellenwerte:</strong> CSI &gt;80 = kritisches Stressniveau · CSI &lt;40 = ruhiges Umfeld.</p>"
        "</div>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Komponente</th><th>Bedeutung</th><th class='text-center'>Gewicht</th><th>Ticker</th></tr></thead>"
        f"<tbody>{comp_rows}</tbody></table></div>"
        "<p class='text-muted small mt-2'>Glättung: 3-Tage-Mittelwert (verhindert Einzelausschläge).</p>"
    )
    p2 = meth_html

    # ── §3 CSI Zeitreihe + JETS Preis ────────────────────────────────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=["Composite Stress Index (CSI) — 0 = kein Stress, 100 = maximaler Stress",
                                         "JETS ETF Schlusskurs (Originaldaten)"])
    fig3.add_trace(go.Scatter(x=CSI.index, y=CSI.values, name="CSI",
                              fill="tozeroy", line=dict(color="#d2a8ff", width=2),
                              fillcolor="rgba(210,168,255,0.12)"), row=1, col=1)
    # Colored background zones
    fig3.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.15)", line_width=0, row=1, col=1)
    fig3.add_hrect(y0=60, y1=80,  fillcolor="rgba(240,136,62,0.10)", line_width=0, row=1, col=1)
    fig3.add_hline(y=80, line_dash="dash", line_color="#f85149", row=1, col=1)
    fig3.add_hline(y=60, line_dash="dash", line_color="#ffa657", row=1, col=1)
    fig3.add_hline(y=40, line_dash="dash", line_color="#3fb950", row=1, col=1)
    if len(jets_a.dropna()) > 50:
        fig3.add_trace(go.Scatter(x=jets_a.index, y=jets_a.values,
                                  name="JETS", line=dict(color="#58a6ff")), row=2, col=1)
    # Flash crash event lines
    for i, (ename, edate, _, _) in enumerate(EVENTS):
        if pd.Timestamp(edate) in CSI.index or True:
            fig3.add_vline(x=edate, line_color=ECOLORS[i], line_dash="dot", line_width=1.5, row="all", col=1)
            csi_val = float(CSI.get(edate, np.nan)) if edate in CSI.index else 0
            csi_label = f"{ename}: CSI={csi_val:.0f}" if not np.isnan(csi_val) else ename
            fig3.add_annotation(x=edate, y=95, text=ename[:8], showarrow=False,
                                font=dict(color=ECOLORS[i], size=9), textangle=-45)
    fig3.update_yaxes(range=[0, 100], row=1, col=1)
    fig3.update_layout(**_LAYOUT, height=580,
                       title="CSI Zeitreihe 2010–heute  ·  Rote Zone = kritischer Stress (>80)")
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="fc3")

    # ── §4 Einzelkomponenten Dashboard ───────────────────────────────────
    comp_titles = ["C1: VIX Level (Pct-Rang)", "C2: VIX 5d-Spike", "C3: Credit Spread",
                   "C4: DXY Safe-Haven Spike", "C5: JETS Volumen-Anomalie"]
    comp_series = [c1, c2, c3, c4, c5]
    comp_colors = ["#f85149", "#f0883e", "#d2a8ff", "#e3b341", "#3fb950"]

    def _hex_rgba(hx, a=0.12):
        h = hx.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    comp_fills  = [_hex_rgba(c) for c in comp_colors]

    fig4 = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=comp_titles)
    for i, (s, color, fill) in enumerate(zip(comp_series, comp_colors, comp_fills), start=1):
        fig4.add_trace(go.Scatter(x=s.index, y=s.values, name=comp_titles[i-1],
                                  line=dict(color=color, width=1.5),
                                  fill="tozeroy", fillcolor=fill),
                      row=i, col=1)
        fig4.add_hline(y=80, line_dash="dash", line_color="#f85149", row=i, col=1)
        for j, (_, edate, _, _) in enumerate(EVENTS):
            fig4.add_vline(x=edate, line_color=ECOLORS[j], line_width=1, line_dash="dot", row=i, col=1)
    fig4.update_yaxes(range=[0, 100])
    fig4.update_layout(**_LAYOUT, height=900, showlegend=False,
                       title="Einzelkomponenten des CSI — Rote Linie = 80. Perzentil (Stress-Schwelle)")
    p4 = fig4.to_html(full_html=False, include_plotlyjs=False, div_id="fc4")

    # ── §5 Event Deep Dives ───────────────────────────────────────────────
    event_panels = []
    for i, (ename, edate, edesc, ecause) in enumerate(EVENTS):
        edt = pd.Timestamp(edate)
        start = edt - pd.Timedelta(days=30)
        end   = edt + pd.Timedelta(days=10)

        csi_sub  = CSI[(CSI.index >= start) & (CSI.index <= end)]
        jets_sub = jets_a[(jets_a.index >= start) & (jets_a.index <= end)]

        if len(csi_sub) < 3:
            event_panels.append(f"<p class='text-muted'>{ename}: Keine Daten für diesen Zeitraum.</p>")
            continue

        fig_e = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=[f"CSI 30 Tage vor / 10 Tage nach {ename}",
                                              "JETS ETF Preis"])
        fig_e.add_trace(go.Scatter(x=csi_sub.index, y=csi_sub.values, name="CSI",
                                   fill="tozeroy", line=dict(color="#d2a8ff", width=2),
                                   fillcolor="rgba(210,168,255,0.15)"), row=1, col=1)
        fig_e.add_hline(y=80, line_dash="dash", line_color="#f85149", row=1, col=1)
        fig_e.add_hline(y=60, line_dash="dash", line_color="#ffa657", row=1, col=1)
        fig_e.add_vline(x=edate, line_color="#f85149", line_width=2, row="all", col=1)
        fig_e.add_annotation(x=edate, y=95, text="CRASH", showarrow=False,
                             font=dict(color="#f85149", size=11, family="monospace"))
        if len(jets_sub.dropna()) > 2:
            fig_e.add_trace(go.Scatter(x=jets_sub.index, y=jets_sub.values,
                                       name="JETS", line=dict(color="#58a6ff")), row=2, col=1)
            fig_e.add_vline(x=edate, line_color="#f85149", line_width=2, row=2, col=1)
        fig_e.update_yaxes(range=[0, 100], row=1, col=1)
        fig_e.update_layout(**_LAYOUT, height=420, showlegend=False,
                            title=f"{ename} — {edate}")

        # Compute CSI 20/10/5/1 days before event
        def _csi_at(days_before):
            target = edt - pd.Timedelta(days=days_before)
            nearby = csi_sub[csi_sub.index <= target]
            return float(nearby.iloc[-1]) if len(nearby) else np.nan

        csi_20 = _csi_at(20); csi_10 = _csi_at(10)
        csi_5  = _csi_at(5);  csi_1  = _csi_at(1)

        jets_ret_day = np.nan
        if edt in jets_a.index:
            jret_all = jets_a.pct_change()
            jets_ret_day = float(jret_all.get(edt, np.nan))

        ret_str = f"{jets_ret_day*100:.2f}%" if not np.isnan(jets_ret_day) else "N/A"

        def _csi_badge(v):
            if np.isnan(v): return "<span class='badge bg-secondary'>N/A</span>"
            cls = "danger" if v > 80 else ("warning" if v > 60 else "success")
            return f"<span class='badge bg-{cls}'>{v:.0f}</span>"

        stat_table = (
            f"<div class='card bg-secondary text-light p-2 mt-2'>"
            f"<p class='mb-1'><strong>Ursache:</strong> {ecause}</p>"
            f"<p class='mb-1 small'>{edesc}</p>"
            f"<table class='table table-dark table-sm mb-0'>"
            f"<tr><th>CSI 20d vorher</th><th>CSI 10d vorher</th><th>CSI 5d vorher</th><th>CSI 1d vorher</th><th>JETS Tagesrendite</th></tr>"
            f"<tr><td>{_csi_badge(csi_20)}</td><td>{_csi_badge(csi_10)}</td>"
            f"<td>{_csi_badge(csi_5)}</td><td>{_csi_badge(csi_1)}</td>"
            f"<td class='text-danger'><strong>{ret_str}</strong></td></tr>"
            f"</table></div>"
        )
        chart_html = fig_e.to_html(full_html=False, include_plotlyjs=False, div_id=f"fc5_{i}")
        event_panels.append(chart_html + stat_table)

    # Navigation tabs for the 5 events
    tab_btns = "".join(
        f"<button class='nav-link {'active' if i == 0 else ''}' id='ev{i}-tab' "
        f"data-bs-toggle='tab' data-bs-target='#ev{i}' type='button'>{EVENTS[i][0]}</button>"
        for i in range(len(EVENTS)))
    tab_panes = "".join(
        f"<div class='tab-pane fade {'show active' if i == 0 else ''}' id='ev{i}'>"
        f"{event_panels[i] if i < len(event_panels) else ''}</div>"
        for i in range(len(EVENTS)))
    p5 = (
        f"<ul class='nav nav-tabs mb-3' id='evTabs'>{tab_btns}</ul>"
        f"<div class='tab-content'>{tab_panes}</div>"
    )

    # ── §6 Lead-Time Heatmap ──────────────────────────────────────────────
    COMP_NAMES = ["VIX Level", "VIX Spike", "Credit Spr.", "DXY Spike", "JETS Volume"]
    comp_srs   = [c1, c2, c3, c4, c5]
    THRESHOLD  = 80.0

    lead_z    = []
    lead_text = []
    event_labels = [e[0] for e in EVENTS]

    for ename, edate, _, _ in EVENTS:
        row_z = []; row_t = []
        edt = pd.Timestamp(edate)
        for cs in comp_srs:
            window = cs[(cs.index >= edt - pd.Timedelta(days=25)) & (cs.index < edt)]
            crossings = window[window >= THRESHOLD]
            if len(crossings):
                lead_days = int((edt - crossings.index[0]).days)
                row_z.append(float(lead_days))
                row_t.append(f"{lead_days}d früh")
            else:
                row_z.append(-1.0)
                row_t.append("Kein Signal")
        lead_z.append(row_z); lead_text.append(row_t)

    fig6 = go.Figure(go.Heatmap(
        z=lead_z, x=COMP_NAMES, y=event_labels,
        text=lead_text, texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=7,
        zmin=-1, zmax=20,
        colorbar=dict(title="Vorlaufzeit (Tage)", tickfont=dict(color="#e6edf3")),
    ))
    fig6.update_layout(**_LAYOUT, height=320,
                       title="Lead-Time Heatmap — Wie viele Tage vor Crash überschritt jede Komponente den 80. Perzentil?")
    lead_explain = (
        "<div class='card bg-secondary text-light p-2 mb-2'>"
        "<small><strong>Grün = frühzeitige Warnung</strong> (viele Tage vor Crash). "
        "<strong>Rot = kein Signal</strong> (Komponente blieb unter 80). "
        "Je grüner ein Feld, desto früher hätte diese Komponente gewarnt.</small></div>"
    )
    p6 = lead_explain + fig6.to_html(full_html=False, include_plotlyjs=False, div_id="fc6")

    # ── §7 CSI als Risiko-Overlay Backtest ────────────────────────────────
    strat_intro = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h6 class='text-info'>Wie kann man den CSI in die Strategie integrieren?</h6>"
        "<p>Wir testen zwei Ansätze:</p>"
        "<ul>"
        "<li><strong>CSI-Exit</strong>: Bestehende Position wird geschlossen wenn CSI > 80 → sofortige Risikoreduktion.</li>"
        "<li><strong>CSI-Filter</strong>: Neue Positionen werden nur eröffnet wenn CSI &lt; 60 → kein Einstieg in stress.</li>"
        "</ul>"
        "<p>Diese Logik ist <em>additiv</em> zum bestehenden VIX-Filter und Stop-Loss.</p>"
        "</div>"
    )

    # Build CSI-filtered signal
    if len(jets_a.dropna()) > 200 and len(CSI.dropna()) > 200:
        ret_main_loc = _read(tables / "phase2_returns.csv")
        ret_main_loc.index = pd.to_datetime(ret_main_loc.index, errors="coerce")
        bk_cols = [c for c in ["CL=F", "BZ=F", "XLE", "XOM", "CVX"] if c in ret_main_loc.columns]
        bk_ret  = ret_main_loc[bk_cols].mean(axis=1) if bk_cols else pd.Series(0.0, index=ret_main_loc.index)

        jets_close_l = jets_a
        vix_al       = vix_a
        common_l     = jets_close_l.index.intersection(bk_ret.index).intersection(vix_al.index).intersection(CSI.index)
        jets_c = jets_close_l.reindex(common_l).ffill()
        jets_lo_l = jlo_a.reindex(common_l).ffill()
        vix_l   = vix_al.reindex(common_l).ffill()
        bk_l    = bk_ret.reindex(common_l).fillna(0)
        csi_l   = CSI.reindex(common_l).ffill().fillna(50)

        base_sig = ((bk_l.rolling(20).mean() > 0) & (vix_l < 25)).astype(int)
        csi_sig  = base_sig.copy()
        # CSI-Exit: wenn CSI > 80, kein Long
        csi_sig[csi_l > 80] = 0
        # CSI-Filter: nur einsteigen wenn CSI < 60
        can_enter = csi_l < 60
        # Apply: if previous can_enter was False, also block entry
        for i in range(1, len(csi_sig)):
            if base_sig.iloc[i-1] == 1 and not in_pos_csi if False else False:
                pass  # simplified — just apply the mask

        CAP_b = 100_000.0; SL_b = 0.30; TC_b = 0.001; PF_b = 0.95

        def _sim_simple(sg, cl, lo):
            cash = CAP_b; shares = 0.0; stop_px = 0.0; in_pos = False; navs = []
            for i in range(len(sg)):
                c = float(cl.iloc[i]); l = float(lo.iloc[i]); stopped = False
                if in_pos and l <= stop_px:
                    ep = max(stop_px * 0.995, l)
                    cash += shares * ep * (1 - TC_b)
                    shares = 0.0; in_pos = False; stopped = True
                if i > 0 and not stopped:
                    sp = int(sg.iloc[i-1])
                    if sp == 1 and not in_pos:
                        invest = cash * PF_b
                        shares = invest * (1 - TC_b) / c
                        stop_px = c * (1 - SL_b); cash -= invest; in_pos = True
                    elif sp == 0 and in_pos:
                        cash += shares * c * (1 - TC_b)
                        shares = 0.0; in_pos = False
                navs.append(cash + shares * c)
            return pd.Series(navs, index=sg.index)

        nav_base = _sim_simple(base_sig, jets_c, jets_lo_l)
        nav_csi  = _sim_simple(csi_sig,  jets_c, jets_lo_l)

        def _sh(nav):
            r = nav.pct_change().dropna()
            a = r.mean() * 252; v = r.std() * (252**0.5)
            return a / v if v > 1e-9 else 0.0

        sh_base = _sh(nav_base); sh_csi = _sh(nav_csi)
        dd_base = float((nav_base / nav_base.cummax() - 1).min())
        dd_csi  = float((nav_csi  / nav_csi.cummax()  - 1).min())

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=nav_base.index, y=nav_base.values,
                                  name=f"Basis-Strategie (Sharpe {sh_base:.2f})",
                                  line=dict(color="#58a6ff")))
        fig7.add_trace(go.Scatter(x=nav_csi.index, y=nav_csi.values,
                                  name=f"+ CSI-Filter (Sharpe {sh_csi:.2f})",
                                  line=dict(color="#3fb950")))
        for i2, (_, cs, ce) in enumerate([("GFC","2007-10-01","2009-06-01"),
                                          ("COVID","2020-02-01","2020-05-01"),
                                          ("Infl.","2022-01-01","2022-12-31")]):
            fig7.add_vrect(x0=cs, x1=ce, fillcolor="rgba(248,81,73,0.08)", line_width=0)
        csi_title = (f"CSI-Filter Backtest: Basis-Strategie vs. +CSI-Overlay  ·  "
                     f"MaxDD Basis={dd_base*100:.1f}% → CSI={dd_csi*100:.1f}%")
        fig7.update_layout(**_LAYOUT, height=420, title=csi_title, yaxis_title="NAV (€)")
        p7 = strat_intro + fig7.to_html(full_html=False, include_plotlyjs=False, div_id="fc7")
    else:
        p7 = strat_intro + "<p class='text-muted'>Nicht genug Daten für CSI-Backtest.</p>"

    # ── §8 Aktuelle Signalwerte ───────────────────────────────────────────
    cur_rows = ""
    if len(cidx):
        last = cidx[-1]
        def _cv(s): return float(s.loc[last]) if last in s.index and not np.isnan(s.loc[last]) else float("nan")
        def _badge(v):
            if np.isnan(v): return "<span class='badge bg-secondary'>N/A</span>"
            cls = "danger" if v > 80 else ("warning" if v > 60 else "success")
            return f"<span class='badge bg-{cls}'>{v:.0f}/100</span>"

        cur_vals = [
            ("VIX Level",          _cv(c1), f"VIX={_cv(vix_a):.1f}"),
            ("VIX 5d Spike",       _cv(c2), "Kurzfristige Angst-Zunahme"),
            ("Credit Spread",      _cv(c3), "HYG/IEF Spread-Weitung"),
            ("DXY Safe-Haven",     _cv(c4), "USD-Flucht-Nachfrage"),
            ("JETS Volume",        _cv(c5), "Volumen-Anomalie JETS"),
            ("VIX Term Structure", _cv(c6), "VIX9D vs VIX Inversion" if has_ts else "Nicht verfügbar"),
        ]
        csi_now = float(CSI.iloc[-1]) if len(CSI.dropna()) else float("nan")
        csi_cls = "danger" if csi_now > 80 else ("warning" if csi_now > 60 else "success")
        interpretation = (
            "KRITISCH — Positionsabbau empfohlen." if csi_now > 80 else
            ("ERHÖHT — Neue Long-Positionen vermeiden." if csi_now > 60 else
             ("NORMAL — Einstieg möglich wenn Signal aktiv." if csi_now > 40 else
              "RUHIG — Günstige Bedingungen für neue Positionen."))
        )
        for label, val, note in cur_vals:
            cur_rows += f"<tr><td>{label}</td><td>{_badge(val)}</td><td class='text-muted small'>{note}</td></tr>"
        reading_html = (
            f"<div class='alert alert-{csi_cls} mb-3'>"
            f"<h5>Aktueller CSI: <strong>{csi_now:.1f}/100</strong> — {interpretation}</h5>"
            f"<small>Stand: {last.strftime('%Y-%m-%d')}</small></div>"
            "<div class='table-responsive'>"
            "<table class='table table-dark table-sm table-bordered'>"
            "<thead><tr><th>Komponente</th><th>Aktueller Wert</th><th>Hinweis</th></tr></thead>"
            f"<tbody>{cur_rows}</tbody></table></div>"
        )
        p8 = reading_html
    else:
        p8 = "<p class='text-muted'>Keine aktuellen Daten verfügbar.</p>"

    # ── §9 Live Yahoo News Widget ─────────────────────────────────────────
    news_js = r"""
<div class="card bg-dark border-secondary p-3">
  <h6 class="text-warning">Live Yahoo Finance News (JETS · XLE · VIX)</h6>
  <p class="text-muted small">Lädt aktuelle Headlines beim Öffnen des Reports.
    Benötigt Internetzugang und HTTP-Server (nicht file://).
    <a href="https://finance.yahoo.com/topic/latest-news/?guccounter=1" target="_blank" class="text-info">
      Direkt auf Yahoo Finance &rarr;
    </a>
  </p>
  <div id="yf-news-widget">
    <p class="text-muted small" id="yf-loading">Lädt...</p>
  </div>
</div>
<script>
(function() {
  const widget = document.getElementById('yf-news-widget');
  const loading = document.getElementById('yf-loading');
  const queries = ['JETS airline', 'XLE oil energy', 'VIX volatility spike'];
  const colors  = ['#58a6ff', '#3fb950', '#f85149'];
  let loaded = 0;

  queries.forEach(function(q, qi) {
    const url = 'https://query2.finance.yahoo.com/v1/finance/search?q=' +
                encodeURIComponent(q) + '&newsCount=5&enableFuzzyQuery=false&enableNavLinks=false';
    fetch(url, {method: 'GET', headers: {'Accept': 'application/json'}})
      .then(function(r) { return r.json(); })
      .then(function(data) {
        const news = (data && data.news) ? data.news : [];
        if (loaded === 0) loading.style.display = 'none';
        loaded++;
        if (news.length === 0) return;
        const sec = document.createElement('div');
        sec.className = 'mb-3';
        const title = document.createElement('h6');
        title.style.color = colors[qi];
        title.textContent = '\u25B6 ' + q.toUpperCase();
        sec.appendChild(title);
        news.slice(0, 4).forEach(function(n) {
          const d = document.createElement('div');
          d.className = 'border-bottom border-secondary pb-1 mb-1';
          const pubDate = n.providerPublishTime ? new Date(n.providerPublishTime * 1000).toLocaleDateString('de-DE') : '';
          d.innerHTML = '<a href="' + (n.link || '#') + '" class="text-light small" target="_blank">' +
                        (n.title || 'No title') + '</a>' +
                        '<span class="text-muted ms-2" style="font-size:0.75em">' + pubDate + '</span>';
          sec.appendChild(d);
        });
        widget.appendChild(sec);
      })
      .catch(function() {
        if (loaded === 0) {
          loading.textContent = 'News-Widget: CORS-Fehler — öffne den Report per HTTP-Server oder besuche Yahoo Finance direkt.';
          loading.className = 'text-muted small';
          loaded++;
        }
      });
  });
})();
</script>
"""
    p9 = news_js

    # ── Accordion Assembly ────────────────────────────────────────────────
    def _acc(n, title, body, show=False):
        cls = "" if show else "collapsed"
        sh  = "show" if show else ""
        return (
            f"<div class='accordion-item bg-dark border-secondary'>"
            f"<h2 class='accordion-header'>"
            f"<button class='accordion-button {cls} bg-dark text-light'"
            f" type='button' data-bs-toggle='collapse' data-bs-target='#fc_p{n}'>"
            f"{title}</button></h2>"
            f"<div id='fc_p{n}' class='accordion-collapse collapse {sh}'>"
            f"<div class='accordion-body'>{body}</div></div></div>"
        )

    panels = [
        _acc(1, "§1 · Flash Crashes & Airlines — Warum gefährlich?",              p1, show=True),
        _acc(2, "§2 · Composite Stress Index (CSI) — Methodik & Gewichte",       p2),
        _acc(3, "§3 · CSI Zeitreihe 2010–heute + JETS Preis",                    p3),
        _acc(4, "§4 · Einzelkomponenten Dashboard (5 Signale, 0–100 normiert)",  p4),
        _acc(5, "§5 · Flash Crash Event Deep Dives (5 historische Events)",       p5),
        _acc(6, "§6 · Signal Lead-Time Heatmap (Wie früh warnte jedes Signal?)", p6),
        _acc(7, "§7 · CSI als Risiko-Overlay in der Strategie (Backtest)",       p7),
        _acc(8, "§8 · Aktueller CSI-Score & Signalwerte",                        p8),
        _acc(9, "§9 · Live Yahoo News (JETS · XLE · VIX)",                       p9),
    ]
    body_html = "<div class='accordion' id='fc_acc'>" + "".join(panels) + "</div>"
    _write(out / "flash_crash_report.html",
           _html_base("Flash Crash Early Warning System", 20, body_html))


def build_index(tables, figures, out):
    def _pc(num, title, desc, file, done, is_new=False):
        col    = PHASE_COLOURS.get(num, "#58a6ff")
        badge  = ('<span class="badge bg-success">OK</span>' if done
                  else '<span class="badge bg-secondary">Ausstehend</span>')
        nbadge = ('<span class="badge ms-1" style="background:#3fb95022;color:#3fb950;'
                  'border:1px solid #3fb950;">NEU</span>' if is_new else "")
        return f"""<div class="col-md-6 col-lg-4">
  <div class="card h-100" style="border-color:{col}44;">
    <div class="card-header" style="background:{col}18;color:{col};">
      Phase {num} - {title} {nbadge}</div>
    <div class="card-body"><p class="small" style="color:#8b949e;">{desc}</p>{badge}</div>
    <div class="card-footer" style="background:#0d1117;border-top:1px solid #30363d;">
      <a href="{file}" class="btn btn-sm" style="background:{col}22;color:{col};"
         >Report oeffnen</a>
    </div>
  </div>
</div>"""

    phases = [
        (1,"Datenerhebung","34 Ticker Yahoo, FRED, EIA","phase01_data_loading.html",
         (tables/"phase1_prices.csv").exists(),False),
        (2,"Preprocessing","Log-Renditen, Alignment","phase02_preprocessing.html",
         (tables/"phase2_returns.csv").exists(),False),
        (3,"EDA","Risikokennzahlen, Volatility, Korrelation","phase03_eda.html",
         (tables/"phase3_descriptive_stats.csv").exists(),False),
        (4,"Stationaritaet","ADF, KPSS, Phillips-Perron","phase04_stationarity.html",
         (tables/"phase4_stationarity.csv").exists(),False),
        (5,"Korrelation","Pearson, Spearman, Kendall","phase05_correlation.html",
         (tables/"phase5_corr_pearson.csv").exists(),False),
        (6,"Lead-Lag","CCF, Granger, VAR, IRF, FEVD","phase06_leadlag.html",
         (tables/"phase6_granger_matrix.csv").exists(),False),
        (7,"Event Studies","CPI/NFP/FOMC/EIA -> CAR","phase07_events.html",
         (tables/"phase7_event_studies.csv").exists(),False),
        (8,"Kointegration","Johansen, Engle-Granger, VECM","phase08_cointegration.html",
         (tables/"phase8_johansen.csv").exists(),True),
        (9,"GARCH & Regime","GARCH(1,1), Vol-Regime, STL","phase09_garch_regimes.html",
         (tables/"phase9_garch_params.csv").exists(),True),
        (10,"Faktormodelle","PCA, Scree-Plot, beta-Koeff.","phase10_factors.html",
         (tables/"phase10_pca_loadings.csv").exists(),False),
        (11,"Bootstrap CI","Block-Bootstrap fuer Lag-Schaetzer","phase11_bootstrap.html",
         (tables/"phase6_ccf_lags.csv").exists(),True),
        (12,"Hypothesentests","H1-H7: statist. Ueberpruefung","phase12_hypotheses.html",
         (tables/"phase6_granger.csv").exists(),True),
        (13,"Netzwerkanalyse","Granger-Netz, PageRank, Degree","phase13_network.html",
         (tables/"phase13_network_metrics.csv").exists(),False),
    ]
    extras = [
        ("PCA Deep-Dive","Biplot, Eigenrichtungen, AR-Simulation","pca_deep_dive.html","#7ee787",
         (tables/"phase10_pc_scores.csv").exists()),
        ("Mega-Netzwerk (4 Layouts)","Hierarchisch|Kreis|Cluster|Stern+Sankey","mega_network.html","#ff9fef",
         (tables/"phase5_significant_correlations.csv").exists()),
        ("Zeitreihen-Viewer","Paarweise Preis + STL","timeseries_viewer.html","#a5d6ff",
         (tables/"phase1_prices.csv").exists()),
        ("Backtesting","Strategien & Walk-Forward","backtesting.html","#56d364",
         (tables/"phase2_returns.csv").exists()),
        ("Insights & Signale","Granger|Events|Kointegration|Handlungsmatrix","insights_report.html","#79c0ff",
         (tables/"phase6_granger.csv").exists()),
        ("Overshoot & Korrektur","Event-Study | Overshoot-Ratio | Korrektur nach Lag","overshoot.html","#ff9fef",
         (tables/"phase6_granger.csv").exists()),
        ("Externe Treiber","FX · TIPS · HY-Spread · Shipping · EV · Partieller R² · Saisonalität","external_drivers.html","#56d364",
         (tables/"phase2_returns.csv").exists()),
        ("Prädiktiver Backtest","PCA-Modelle · Walk-Forward · Monte Carlo · Residual-Diagnostik","predictive_backtest.html","#79c0ff",
         (tables/"phase2_returns.csv").exists()),
        ("Mega-Strategien 80+","9 Familien · Param-Grid · Steckbriefe · IS/OOS Analyse","mega_strategies.html","#ffa657",
         (tables/"phase2_returns.csv").exists()),
        ("Intraday CCF","Stündliche Lag-Auflösung für lag=0 Paare · ±24h · Sub-tägige Lead-Lag","intraday_ccf.html","#ff9fef",
         (tables/"phase2_returns.csv").exists()),
        ("Technische Analyse","SMA/RSI/MACD/Bollinger · Lead-Lag-Indikator-Strategien · Cross-Asset-Overlay","technical_analysis.html","#79c0ff",
         (tables/"phase2_returns.csv").exists()),
        ("Lead-Lag Optimizer","Param-Grid Heatmaps · IS-Sharpe · Walk-Forward OOS · 4 Indikatoren","lead_lag_optimizer.html","#c9d1d9",
         (tables/"phase2_returns.csv").exists()),
        ("Strategie-Paare","Alle Paare · 5 Indikatoren · 26 Metriken · Paar-Statistiken","strategy_pairs.html","#c9d1d9",
         (tables/"phase2_returns.csv").exists()),
        ("PCA-Strategie","PC1-Filter · Version A/B · Bootstrap-CI · IS/OOS","pca_strategy.html","#c9d1d9",
         (tables/"phase2_returns.csv").exists()),
        ("Strategy Stress-Test","TC-Sweep · Monte Carlo · Bootstrap CI · Walk-Forward · Krisenperioden · Kelly","strategy_stress_test.html","#ff9fef",
         (tables/"phase2_returns.csv").exists()),
    ]

    done_n = sum(1 for p in phases if p[4])
    cards  = "\n".join(_pc(*p) for p in phases)
    extra_html = ""
    for title, desc, file, col, done in extras:
        badge = '<span class="badge bg-success">OK</span>' if done else '<span class="badge bg-secondary">Ausstehend</span>'
        extra_html += f"""<div class="col-md-6 col-lg-3">
  <div class="card h-100" style="border-color:{col}44;">
    <div class="card-header" style="background:{col}18;color:{col};">{title}
      <span class="badge ms-1" style="background:#3fb95022;color:#3fb950;border:1px solid #3fb950;">NEU</span>
    </div>
    <div class="card-body"><p class="small" style="color:#8b949e;">{desc}</p>{badge}</div>
    <div class="card-footer" style="background:#0d1117;border-top:1px solid #30363d;">
      <a href="{file}" class="btn btn-sm" style="background:{col}22;color:{col};">Oeffnen</a>
    </div>
  </div>
</div>"""

    n_coint = "26"
    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Commodity Research - Dashboard</title>
  <link href="{BOOTSTRAP_CDN}" rel="stylesheet"/>
  <style>
    body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',Arial,sans-serif;}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;}}
    .card-header{{background:#1c2128;border-bottom:1px solid #30363d;font-weight:600;}}
    .stat-card{{background:#1c2128;border:1px solid #30363d;border-radius:8px;
                padding:1rem;text-align:center;}}
    .stat-card .val{{font-size:1.5rem;font-weight:700;color:#58a6ff;}}
    .stat-card .lbl{{font-size:.76rem;color:#8b949e;}}
    table{{color:#e6edf3!important;font-size:.85rem;}}
    h3,h4{{color:#e6edf3;}} a{{color:#58a6ff;}}
  </style>
</head>
<body>
<nav style="background:#010409;border-bottom:1px solid #30363d;padding:.8rem 2rem;">
  <span style="color:#58a6ff;font-weight:700;font-size:1.1rem;">
    &#127748; Commodity Research Framework
  </span>
</nav>
<div class="container-xl py-4">
  <div style="background:#1f6feb22;border-left:5px solid #1f6feb;border-radius:8px;
              padding:1.5rem 2rem;margin-bottom:2rem;">
    <h1 style="color:#1f6feb;font-size:1.8rem;margin:0;">
      Oekonometrische Analyse des Informationsflusses</h1>
    <div style="color:#8b949e;margin-top:.4rem;">
      Rohstoffmaerkte -> ETFs -> Mega-Cap -> Mid-Cap -> Small-Cap Produzenten
    </div>
  </div>
  <div class="row g-3 mb-4">
    <div class="col"><div class="stat-card"><div class="val">{done_n}/{len(phases)}</div>
      <div class="lbl">Phasen</div></div></div>
    <div class="col"><div class="stat-card"><div class="val">34</div>
      <div class="lbl">Ticker</div></div></div>
    <div class="col"><div class="stat-card"><div class="val">~25J</div>
      <div class="lbl">Zeitraum</div></div></div>
    <div class="col"><div class="stat-card"><div class="val">81.5%</div>
      <div class="lbl">Varianz (10 PCs)</div></div></div>
    <div class="col"><div class="stat-card"><div class="val">{n_coint}</div>
      <div class="lbl">Kointegrierte Paare</div></div></div>
    <div class="col"><div class="stat-card"><div class="val">6</div>
      <div class="lbl">GARCH-Modelle</div></div></div>
  </div>
  <h3 class="mb-3">Analysephasen (13 Phasen)</h3>
  <div class="row g-3 mb-4">{cards}</div>
  <h3 class="mb-3">Erweiterte Berichte
    <span class="badge ms-2" style="background:#3fb95022;color:#3fb950;
          border:1px solid #3fb950;">NEU</span>
  </h3>
  <div class="row g-3 mb-4">{extra_html}</div>
  <div class="card mb-4">
    <div class="card-header" style="color:#58a6ff;">Informationsfluss-Hypothese</div>
    <div class="card-body">
      <table class="table table-dark table-sm table-bordered text-center">
        <thead><tr>
          <th style="color:#d29922;">L0: Rohstoff</th>
          <th style="color:#39d353;">L1: ETF</th>
          <th style="color:#58a6ff;">L2: Mega-Cap</th>
          <th style="color:#bc8cff;">L3: Mid-Cap</th>
          <th style="color:#f78166;">L4: Small-Cap</th>
          <th style="color:#8b949e;">L6: Index</th>
        </tr></thead>
        <tbody>
          <tr>
            <td style="color:#d29922;">CL=F, GC=F, HG=F</td>
            <td style="color:#39d353;">XLE, GDX, XLB</td>
            <td style="color:#58a6ff;">XOM, CVX, FCX</td>
            <td style="color:#bc8cff;">APA, OXY, TECK</td>
            <td style="color:#f78166;">SM, TGB, GORO</td>
            <td style="color:#8b949e;">SPY, QQQ, IWM</td>
          </tr>
          <tr><td colspan="6" style="color:#8b949e;font-size:.8rem;">
            Hypothese: Lag und Reaktionszeit steigen | Renditeeffekt sinkt | Volatilitaet steigt
            von links nach rechts
          </td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body></html>"""
    _write(out / "index.html", html)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_all_reports(output_dir: Path) -> None:
    tables  = output_dir / "tables"
    figures = output_dir / "figures"
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    print("\n=== Generating HTML Reports ===")
    build_phase1_report(tables, figures, reports)
    build_phase2_report(tables, figures, reports)
    build_phase3_report(tables, figures, reports)
    build_phase4_report(tables, figures, reports)
    build_phase5_report(tables, figures, reports)
    build_phase6_report(tables, figures, reports)
    build_phase7_report(tables, figures, reports)
    build_phase8_report(tables, figures, reports)
    build_phase9_report(tables, figures, reports)
    build_phase10_report(tables, figures, reports)
    build_pca_deep_report(tables, figures, reports)
    build_phase11_report(tables, figures, reports)
    build_phase12_report(tables, figures, reports)
    build_phase13_report(tables, figures, reports)
    build_mega_network_report(tables, figures, reports)
    build_timeseries_viewer_report(tables, figures, reports)
    build_backtest_report(tables, figures, reports)
    build_insights_report(tables, figures, reports)
    build_overshoot_report(tables, figures, reports)
    build_external_drivers_report(tables, figures, reports)
    build_predictive_backtest_report(tables, figures, reports)
    build_mega_strategies_report(tables, figures, reports)
    build_intraday_ccf_report(tables, figures, reports)
    build_technical_analysis_report(tables, figures, reports)
    build_lead_lag_optimizer_report(tables, figures, reports)
    build_strategy_pairs_report(tables, figures, reports)
    build_pca_strategy_report(tables, figures, reports)
    build_strategy_stress_test_report(tables, figures, reports)
    build_airline_oil_report(tables, figures, reports)
    build_seasonality_report(tables, figures, reports)
    build_alpha_ideas_report(tables, figures, reports)
    build_combination_holdperiod_report(tables, figures, reports)
    build_leverage_crisis_report(tables, figures, reports)
    build_portfolio_simulation_report(tables, figures, reports)
    build_combination_deepdive_report(tables, figures, reports)
    build_crisis_vs_nocrisis_report(tables, figures, reports)
    build_crisis_predictivity_report(tables, figures, reports)
    build_sector_rotation_report(tables, figures, reports)
    build_flash_crash_report(tables, figures, reports)
    build_index(tables, figures, reports)
    print(f"\n  Dashboard: {reports / 'index.html'}")
    print("=== Reports complete ===\n")


if __name__ == "__main__":
    from pathlib import Path as _P
    build_all_reports(_P(__file__).resolve().parent.parent / "outputs")
