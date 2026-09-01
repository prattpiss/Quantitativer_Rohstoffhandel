#!/usr/bin/env python3
"""Inject build_crisis_vs_nocrisis_report into reports/report_builder.py"""
from pathlib import Path

SRC   = Path("reports/report_builder.py")
FN    = "build_crisis_vs_nocrisis_report"
INJ   = "\ndef build_index(tables, figures, out):"
OLD_W = "    build_combination_deepdive_report(tables, figures, reports)\n    build_index(tables, figures, reports)"
NEW_W = ("    build_combination_deepdive_report(tables, figures, reports)\n"
         "    build_crisis_vs_nocrisis_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

FUNC = '''
def build_crisis_vs_nocrisis_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    print("  Report: crisis_vs_nocrisis_report.html")

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
               _html_base("Crisis vs No-Crisis", f"<p class='text-warning'>{e}</p>"))
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

    idx = close_j.index.intersection(bk_ret.index).intersection(vix_raw.index)
    close_j = close_j.reindex(idx).ffill()
    low_j   = low_j.reindex(idx).ffill()
    vix_a   = vix_raw.reindex(idx).ffill()
    bk_a    = bk_ret.reindex(idx).fillna(0.0)
    sig     = ((bk_a.rolling(20).mean() > 0) & (vix_a < 25)).astype(int)

    def _crisis_mask(ix):
        m = pd.Series(False, index=ix)
        for _, s, e in CRISES:
            m |= (ix >= s) & (ix <= e)
        return m

    sig_ncr = sig.copy()
    sig_ncr[_crisis_mask(idx)] = 0

    CAP, SL, TC, PF = 100_000.0, 0.30, 0.001, 0.95

    def _sim(sg):
        cash = CAP; shares = 0.0; stop_px = 0.0; in_pos = False; navs = []
        for i in range(len(sg)):
            c = float(close_j.iloc[i]); l = float(low_j.iloc[i]); stopped = False
            if in_pos and l <= stop_px:
                ep = max(stop_px * 0.995, l)
                cash += shares * ep * (1 - TC)
                shares = 0.0; in_pos = False; stopped = True
            if i > 0 and not stopped:
                sp = int(sg.iloc[i - 1])
                if sp == 1 and not in_pos:
                    invest = cash * PF
                    shares = invest * (1 - TC) / c
                    stop_px = c * (1 - SL)
                    cash -= invest; in_pos = True
                elif sp == 0 and in_pos:
                    cash += shares * c * (1 - TC)
                    shares = 0.0; in_pos = False
            navs.append(cash + shares * c)
        return pd.Series(navs, index=sg.index)

    nav_f = _sim(sig)
    nav_n = _sim(sig_ncr)

    def _m(nav):
        r   = nav.pct_change().dropna()
        ann = float(r.mean() * 252)
        vol = float(r.std() * (252 ** 0.5))
        sh  = ann / vol if vol > 1e-9 else 0.0
        mdd = float((nav / nav.cummax() - 1).min())
        wr  = float((r > 0).mean())
        return ann, vol, sh, mdd, wr, float(nav.iloc[-1])

    mf = _m(nav_f)
    mn = _m(nav_n)

    # §1 NAV comparison chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=nav_f.index, y=nav_f.values,
                              name="All Periods", line=dict(color="#58a6ff", width=2)))
    fig1.add_trace(go.Scatter(x=nav_n.index, y=nav_n.values,
                              name="Crisis-Excluded", line=dict(color="#3fb950", width=2)))
    for i, (cn, cs, ce) in enumerate(CRISES):
        fig1.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
        mid = (pd.Timestamp(cs) + (pd.Timestamp(ce) - pd.Timestamp(cs)) / 2).strftime("%Y-%m-%d")
        fig1.add_annotation(x=mid, y=1.04, xref="x", yref="paper",
                            text=cn, showarrow=False, font=dict(color="#f85149", size=9))
    fig1.update_layout(**_LAYOUT, title="NAV: All Periods vs Crisis-Excluded (Basket + VIX<25 Signal)",
                       yaxis_title="Portfolio NAV (€)", height=440)
    p1 = fig1.to_html(full_html=False, include_plotlyjs=False, div_id="cnc1")

    # §2 Metrics table
    labels  = ["Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Win Rate", "Final NAV"]
    vf_str  = [f"{mf[0]*100:.1f}%", f"{mf[1]*100:.1f}%", f"{mf[2]:.2f}",
               f"{mf[3]*100:.1f}%", f"{mf[4]*100:.1f}%", f"€{mf[5]:,.0f}"]
    vn_str  = [f"{mn[0]*100:.1f}%", f"{mn[1]*100:.1f}%", f"{mn[2]:.2f}",
               f"{mn[3]*100:.1f}%", f"{mn[4]*100:.1f}%", f"€{mn[5]:,.0f}"]
    rows2 = "".join(
        f"<tr><td>{l}</td><td class='text-info'>{vf}</td><td class='text-success'>{vn}</td></tr>"
        for l, vf, vn in zip(labels, vf_str, vn_str))
    tbl2 = (
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered text-center'>"
        "<thead><tr><th>Metric</th><th class='text-info'>All Periods</th>"
        "<th class='text-success'>Crisis Excluded</th></tr></thead>"
        f"<tbody>{rows2}</tbody></table></div>"
        "<p class='text-muted small mt-2'>"
        "Crisis Excluded = signal forced to 0 (cash) during GFC / Oil-Crash / COVID / Inflation windows."
        "</p>"
    )

    # §3 Drawdown comparison
    dd_f = (nav_f / nav_f.cummax() - 1) * 100
    dd_n = (nav_n / nav_n.cummax() - 1) * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dd_f.index, y=dd_f.values, fill="tozeroy",
                              name="All Periods", line=dict(color="#f85149"),
                              fillcolor="rgba(248,81,73,0.2)"))
    fig3.add_trace(go.Scatter(x=dd_n.index, y=dd_n.values, fill="tozeroy",
                              name="Crisis-Excluded", line=dict(color="#3fb950"),
                              fillcolor="rgba(63,185,80,0.15)"))
    for i, (_, cs, ce) in enumerate(CRISES):
        fig3.add_vrect(x0=cs, x1=ce, fillcolor=CFILLS[i], line_width=0)
    fig3.update_layout(**_LAYOUT, title="Drawdown Comparison (%)", yaxis_title="DD %", height=320)
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="cnc3")

    # §4 JETS options chain snapshot
    opt_html = ""
    try:
        tk = yf.Ticker("JETS")
        exps = tk.options
        if exps:
            ch    = tk.option_chain(exps[0])
            calls = ch.calls[["strike", "lastPrice", "impliedVolatility",
                               "volume", "openInterest"]].copy()
            puts  = ch.puts [["strike", "lastPrice", "impliedVolatility",
                               "volume", "openInterest"]].copy()
            calls["impliedVolatility"] = (calls["impliedVolatility"] * 100).round(1)
            puts ["impliedVolatility"] = (puts ["impliedVolatility"] * 100).round(1)
            pc_ratio = float(puts["openInterest"].sum()) / max(float(calls["openInterest"].sum()), 1.0)
            snt_label = "Bearish" if pc_ratio > 1.2 else ("Complacent" if pc_ratio < 0.7 else "Neutral")
            opt_title = f"JETS Options {exps[0]}  ·  P/C Ratio: {pc_ratio:.2f}  →  {snt_label}"
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=calls["strike"].tolist(), y=calls["openInterest"].tolist(),
                                  name="Call OI", marker_color="#3fb950", opacity=0.7))
            fig4.add_trace(go.Bar(x=puts["strike"].tolist(), y=puts["openInterest"].tolist(),
                                  name="Put OI", marker_color="#f85149", opacity=0.7))
            fig4.add_trace(go.Scatter(x=calls["strike"].tolist(),
                                      y=calls["impliedVolatility"].tolist(),
                                      name="Call IV%", line=dict(color="#58a6ff"), yaxis="y2"))
            fig4.add_trace(go.Scatter(x=puts["strike"].tolist(),
                                      y=puts["impliedVolatility"].tolist(),
                                      name="Put IV%", line=dict(color="#f0883e"), yaxis="y2"))
            fig4.update_layout(**_LAYOUT, title=opt_title,
                               yaxis_title="Open Interest", height=420, barmode="group",
                               yaxis2=dict(title="IV %", overlaying="y", side="right",
                                           showgrid=False, color="#58a6ff"))
            opt_html = fig4.to_html(full_html=False, include_plotlyjs=False, div_id="cnc4")
        else:
            opt_html = "<p class='text-muted'>No JETS options listed at this time.</p>"
    except Exception as e2:
        opt_html = f"<p class='text-warning'>Options data unavailable: {e2}</p>"

    # §5 Realized volatility percentile
    jr    = close_j.pct_change().dropna()
    rv21  = jr.rolling(21).std() * (252 ** 0.5) * 100
    rvpct = rv21.rolling(252, min_periods=63).rank(pct=True) * 100
    fig5  = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=["21d Realized Vol % (ann.)",
                                          "RV Percentile – rolling 252d window"])
    fig5.add_trace(go.Scatter(x=rv21.index, y=rv21.values,
                              name="RV21", line=dict(color="#f0883e")), row=1, col=1)
    fig5.add_trace(go.Scatter(x=rvpct.index, y=rvpct.values,
                              name="Percentile", fill="tozeroy",
                              line=dict(color="#d2a8ff"),
                              fillcolor="rgba(210,168,255,0.15)"), row=2, col=1)
    fig5.add_hline(y=80, line_dash="dash", line_color="#f85149", row=2, col=1)
    fig5.add_annotation(x=rvpct.dropna().index[-1] if len(rvpct.dropna()) else rvpct.index[-1],
                        y=83, text="Crisis zone (>80th pct)",
                        showarrow=False, font=dict(color="#f85149", size=10))
    for _, cs, ce in CRISES:
        fig5.add_vrect(x0=cs, x1=ce, fillcolor="rgba(248,81,73,0.08)", line_width=0)
    fig5.update_layout(**_LAYOUT, height=480, showlegend=True)
    p5 = fig5.to_html(full_html=False, include_plotlyjs=False, div_id="cnc5")

    # §6 Per-crisis impact table
    rows6 = ""
    for cn, cs, ce in CRISES:
        sub = nav_f[(nav_f.index >= cs) & (nav_f.index <= ce)]
        if len(sub) < 2:
            rows6 += (f"<tr><td>{cn}</td>"
                      f"<td colspan='3' class='text-muted'>JETS pre-launch (no data)</td></tr>")
            continue
        ret_c = float(sub.iloc[-1] / sub.iloc[0] - 1)
        cc = "text-danger" if ret_c < 0 else "text-success"
        rows6 += (f"<tr><td>{cn}</td><td class='{cc}'>{ret_c*100:.1f}%</td>"
                  f"<td>€{float(sub.iloc[0]):,.0f}</td><td>€{float(sub.iloc[-1]):,.0f}</td></tr>")
    tbl6 = (
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered text-center'>"
        "<thead><tr><th>Crisis</th><th>Strategy Return</th>"
        "<th>NAV at Start</th><th>NAV at End</th></tr></thead>"
        f"<tbody>{rows6}</tbody></table></div>"
    )

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
        _acc(1, "§1 · NAV: All Periods vs Crisis-Excluded", p1, show=True),
        _acc(2, "§2 · Performance Metrics Comparison",      tbl2),
        _acc(3, "§3 · Drawdown Comparison",                 p3),
        _acc(4, "§4 · JETS Options Chain Snapshot (Live)",  opt_html),
        _acc(5, "§5 · JETS Realized Volatility & Percentile", p5),
        _acc(6, "§6 · Per-Crisis P&L Impact",               tbl6),
    ]
    body = "<div class='accordion' id='cnc_acc'>" + "".join(panels) + "</div>"
    _write(out / "crisis_vs_nocrisis_report.html",
           _html_base("Crisis vs No-Crisis Strategy Comparison", body))
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

    wired_marker = "build_crisis_vs_nocrisis_report(tables, figures, reports)"
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
