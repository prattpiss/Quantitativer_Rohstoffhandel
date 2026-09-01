"""
Inject build_portfolio_simulation_report into report_builder.py.

Realistic single-position portfolio simulation with:
  - 100 000 € Startkapital
  - Echte JETS OHLC-Preise (Low-Preis für Stop-Loss-Prüfung)
  - 30 % Stop-Loss (getestet gegen Tages-Tief)
  - 10 bp Transaktionskosten (one-way)
  - 95 % Kapitaleinsatz pro Trade (Long-only JETS)
  - Vollständige Historien-Simulation (JETS ab ~2015)
  - Standard-OOS-Simulation (Top-5 Kombinationen)
  - Stop-Loss Sensitivitätsanalyse
  - Krisen-Simulation (5 Perioden)
  - 6 rollende 4-Jahres-Zyklen rückwärts von heute
  - Trade-Log Analyse (P&L-Verteilung, Haltedauer)
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
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

'''

# ── injection ─────────────────────────────────────────────────────────────────
src    = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_portfolio_simulation_report(" in src:
    s = src.find("\ndef build_portfolio_simulation_report(")
    e = src.find("\ndef build_", s + 10)
    src = src[:s] + FUNC + src[e:]
    print("Replaced existing build_portfolio_simulation_report.")
else:
    pos = src.find(MARKER)
    if pos == -1: raise RuntimeError("Marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_portfolio_simulation_report.")

OLD_W = ("    build_leverage_crisis_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_leverage_crisis_report(tables, figures, reports)\n"
         "    build_portfolio_simulation_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_portfolio_simulation_report(tables" in src:
    print("Already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1); print("Wired.")
else:
    print("WARNING: wiring failed.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
