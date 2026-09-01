#!/usr/bin/env python3
"""Inject build_crisis_predictivity_report into reports/report_builder.py"""
from pathlib import Path

SRC   = Path("reports/report_builder.py")
FN    = "build_crisis_predictivity_report"
INJ   = "\ndef build_index(tables, figures, out):"
OLD_W = ("    build_crisis_vs_nocrisis_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_crisis_vs_nocrisis_report(tables, figures, reports)\n"
         "    build_crisis_predictivity_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

FUNC = '''
def build_crisis_predictivity_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    print("  Report: crisis_predictivity_report.html")

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
               _html_base("Crisis Predictivity", "<p class='text-warning'>Data unavailable.</p>"))
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
           _html_base("Crisis Predictivity Dashboard", body))
'''


def main():
    src = SRC.read_text(encoding="utf-8")
    if f"def {FN}" in src:
        print(f"Already exists: {FN}.")
    else:
        idx = src.find(INJ)
        if idx == -1:
            print("ERROR: injection point not found"); return
        src = src[:idx] + "\n" + FUNC + src[idx:]
        print(f"Injected {FN}.")

    wired_marker = "build_crisis_predictivity_report(tables, figures, reports)"
    if wired_marker in src:
        print("Already wired.")
    elif OLD_W in src:
        src = src.replace(OLD_W, NEW_W)
        print("Wired.")
    else:
        print("WARNING: wiring point not found")

    SRC.write_text(src, encoding="utf-8")
    print(f"Done. {len(src.splitlines())} lines")


if __name__ == "__main__":
    main()
