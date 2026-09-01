#!/usr/bin/env python3
"""Inject build_sector_rotation_report into reports/report_builder.py"""
from pathlib import Path

SRC   = Path("reports/report_builder.py")
FN    = "build_sector_rotation_report"
INJ   = "\ndef build_index(tables, figures, out):"
OLD_W = ("    build_crisis_predictivity_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_crisis_predictivity_report(tables, figures, reports)\n"
         "    build_sector_rotation_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

FUNC = '''
def build_sector_rotation_report(tables, figures, out):  # noqa: C901
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    print("  Report: sector_rotation_report.html")

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
               _html_base("Sector Rotation", "<p class='text-warning'>Insufficient data.</p>"))
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
           _html_base("Sector Rotation Screener", body))
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

    wired_marker = "build_sector_rotation_report(tables, figures, reports)"
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
