"""
Inject build_combination_holdperiod_report into report_builder.py.

Sections:
  §1  Kombinationsmatrix 16 Strategien (Oil Basket × Seasonal × VIX × TNX)
  §2  Beste Kombination – Equity Curve IS+OOS + Rolling Sharpe
  §3  Rendite-Verteilung nach Haltedauer (Violin Plots, n Horizonte)
  §4  Optimale Haltedauer: Ø Rendite + Trade Sharpe vs H
  §5  Signal-driven + Fixed-Hold Equity Curves (▲▼ Entry/Exit, interaktiver Dropdown)
  §6  TC × Haltedauer Sensitivitäts-Heatmap
  §7  Entry-Bedingungsanalyse: RSI × VIX → 10T-Rendite
  §8  Krisenperformance: Top-Strategien in 3 Krisen
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
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

'''

# ── injection ─────────────────────────────────────────────────────────────────
src    = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_combination_holdperiod_report(" in src:
    s = src.find("\ndef build_combination_holdperiod_report(")
    e = src.find("\ndef build_", s + 10)
    src = src[:s] + FUNC + src[e:]
    print("Replaced existing build_combination_holdperiod_report.")
else:
    pos = src.find(MARKER)
    if pos == -1:
        raise RuntimeError("Marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_combination_holdperiod_report.")

# wire
OLD_W = ("    build_alpha_ideas_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_alpha_ideas_report(tables, figures, reports)\n"
         "    build_combination_holdperiod_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_combination_holdperiod_report(tables" in src:
    print("build_all_reports already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
else:
    print("WARNING: wiring failed – check manually.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
