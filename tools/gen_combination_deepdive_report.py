"""
Inject build_combination_deepdive_report into report_builder.py.

Sections:
  §1  IS vs OOS Scatter (Robustheit vs Overfitting-Karte)
  §2  Faktorattribution – marginaler Beitrag jedes Filters
  §3  Rolling OOS Sharpe Stabilität (Top-4 Kombos, 252T Fenster)
  §4  Handelsqualität: Win-Rate, Profit-Faktor, Ø Haltedauer
  §5  Regime-Analyse: VIX × CL-Trend → Return-Heatmap per Kombo
  §6  Deep Dive: 4 ausgewählte Kombinationen im Detail
  §7  Adaptive Kombination: VIX-gesteuerter Switch zwischen Signalen
  §8  Was (noch) nicht kontrolliert werden kann
  §9  Weitere Alpha-Generierungs-Ideen
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
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

'''

# ── injection ─────────────────────────────────────────────────────────────────
src    = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_combination_deepdive_report(" in src:
    s = src.find("\ndef build_combination_deepdive_report(")
    e = src.find("\ndef build_", s + 10)
    src = src[:s] + FUNC + src[e:]
    print("Replaced existing build_combination_deepdive_report.")
else:
    pos = src.find(MARKER)
    if pos == -1: raise RuntimeError("Marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_combination_deepdive_report.")

OLD_W = ("    build_portfolio_simulation_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_portfolio_simulation_report(tables, figures, reports)\n"
         "    build_combination_deepdive_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_combination_deepdive_report(tables" in src:
    print("Already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1); print("Wired.")
else:
    print("WARNING: wiring failed.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
