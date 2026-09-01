#!/usr/bin/env python3
"""Replace build_crisis_vs_nocrisis_report with comprehensive v2 (real data, rolling metrics, trade tracking)"""
from pathlib import Path

SRC = Path("reports/report_builder.py")

NEW_FUNC = r'''
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

'''


def main():
    src   = SRC.read_text(encoding="utf-8")
    START = "\ndef build_crisis_vs_nocrisis_report("
    END   = "\ndef build_crisis_predictivity_report("
    i_s   = src.find(START)
    i_e   = src.find(END)
    if i_s == -1 or i_e == -1:
        print("ERROR: function boundaries not found"); return
    src = src[:i_s] + "\n" + NEW_FUNC.strip() + "\n\n" + src[i_e:]
    SRC.write_text(src, encoding="utf-8")
    print(f"Replaced build_crisis_vs_nocrisis_report. {len(src.splitlines())} lines")


if __name__ == "__main__":
    main()
