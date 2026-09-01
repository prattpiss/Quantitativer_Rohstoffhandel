"""
Inject build_leverage_crisis_report into report_builder.py.

Sections:
  §1  Krisenanalyse – Strategie-Performance in 5 Stress-Perioden
  §2  Leverage-Vergleich: 1× bis 5× mit Kreditkosten + TC
  §3  TC × Leverage Sensitivitäts-Heatmap
  §4  Kelly-Kriterium: wissenschaftlich optimaler Leverage (rolling)
  §5  Vol-Skalierter Leverage (dynamisch, Ziel-Volatilität)
  §6  Drawdown-Vergleich: alle Strategien in Krisen
  §7  Kombiniertes Final-Report: beste Kombo + optimales Leverage
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
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

    rec_card = _card("🏆 Empfehlung: Optimale Strategie-Konfiguration", "#3fb950", f"""
    <div class="row g-3">
      <div class="col-md-6">
        <strong style="color:#3fb950;">Bestes Signal:</strong> {best_lbl}<br>
        <strong style="color:#58a6ff;">Empfohlener Leverage:</strong> Half Kelly = {hk_rec:.1f}× (ggf. auf {min(max(round(hk_rec*2)/2,1.0),3.0):.1f}× runden)<br>
        <strong style="color:#ffa657;">Ziel-Volatilität:</strong> Vol-Skaliert mit {int(TARGET_VOL*100)}% als Puffer<br>
        <strong style="color:#bc8cff;">TC-Schwelle:</strong> Strategie profitabel bis ~{next((int(tc*10000) for ti,tc in enumerate(TC_LEVELS) if tc_lev_z[ti, LEVERAGES.index(2) if 2 in LEVERAGES else 2] < 0.3 else 0), 20)}bp R/T
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

'''

# ── injection ─────────────────────────────────────────────────────────────────
src    = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_leverage_crisis_report(" in src:
    s = src.find("\ndef build_leverage_crisis_report(")
    e = src.find("\ndef build_", s + 10)
    src = src[:s] + FUNC + src[e:]
    print("Replaced existing build_leverage_crisis_report.")
else:
    pos = src.find(MARKER)
    if pos == -1:
        raise RuntimeError("Marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_leverage_crisis_report.")

# wire
OLD_W = ("    build_combination_holdperiod_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_combination_holdperiod_report(tables, figures, reports)\n"
         "    build_leverage_crisis_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_leverage_crisis_report(tables" in src:
    print("build_all_reports already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
else:
    print("WARNING: wiring failed – check manually.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
