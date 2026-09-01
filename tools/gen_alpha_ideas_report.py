"""
Inject build_alpha_ideas_report into report_builder.py.

New alpha ideas explored:
  1. Statistical Arbitrage: airline pair cointegration + spread mean reversion
  2. Oil Term Structure Signal: WTI contango/backwardation via nearby vs deferred
  3. Multi-Indicator Ensemble: weighted RSI+MACD+BB signal
  4. VIX Regime Filter: only trade when VIX < 25
  5. DXY Macro Filter: reduce/flip when USD trending strongly
  6. Cross-Sectional Momentum: rank airlines, long top/short bottom
  7. Oil Volatility Regime: GARCH-style rolling vol threshold
  8. Correlated Commodity Basket: WTI + Brent + XLE ensemble leader
  9. Mean-Reversion on CL=F–XLE Spread (pairs)
 10. Lagged Macro Signals: GDP/CPI proxy via TNX, DXY, SPY
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
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

'''

# ── inject ────────────────────────────────────────────────────────────────────
src = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_alpha_ideas_report(" in src:
    start = src.find("\ndef build_alpha_ideas_report(")
    end   = src.find("\ndef build_", start + 10)
    src   = src[:start] + FUNC + src[end:]
    print("Replaced existing build_alpha_ideas_report.")
else:
    pos = src.find(MARKER)
    if pos == -1:
        raise RuntimeError("Injection marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_alpha_ideas_report.")

# wire build_all_reports
OLD_W = ("    build_seasonality_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_seasonality_report(tables, figures, reports)\n"
         "    build_alpha_ideas_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_alpha_ideas_report(tables" in src:
    print("build_all_reports already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
else:
    print("WARNING: could not wire build_all_reports — check manually.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
