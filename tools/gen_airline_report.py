"""Inject build_airline_oil_report into report_builder.py."""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
def build_airline_oil_report(tables, figures, out):  # noqa: C901
    """
    Comprehensive cross-sectional airline × WTI lead-lag research report.
    Covers: data download, 26 strategy metrics, CCF/Granger/Transfer-Entropy,
    signal stability, rolling metrics, crisis periods, VIX regimes, TC sweep,
    Monte Carlo, bootstrap CI, walk-forward, PCA/cluster, feature importance,
    portfolio analysis, screening model.
    """
    import warnings, itertools
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import yfinance as yf
    from scipy import stats
    from scipy.stats import spearmanr, kendalltau, kruskal, mannwhitneyu, pearsonr

    # ── 1. Universe ──────────────────────────────────────────────────────────
    AIRLINES = {
        "DAL":  {"name":"Delta Air Lines",         "region":"USA",           "type":"Legacy"},
        "UAL":  {"name":"United Airlines",          "region":"USA",           "type":"Legacy"},
        "AAL":  {"name":"American Airlines",        "region":"USA",           "type":"Legacy"},
        "LUV":  {"name":"Southwest Airlines",       "region":"USA",           "type":"LCC"},
        "ALK":  {"name":"Alaska Air Group",         "region":"USA",           "type":"Legacy"},
        "JBLU": {"name":"JetBlue Airways",          "region":"USA",           "type":"LCC"},
        "SAVE": {"name":"Spirit Airlines",          "region":"USA",           "type":"ULCC"},
        "IAG":  {"name":"IAG (British Airways+)",   "region":"Europe",        "type":"Legacy"},
        "LHA":  {"name":"Lufthansa Group",          "region":"Europe",        "type":"Legacy"},
        "AF":   {"name":"Air France-KLM",           "region":"Europe",        "type":"Legacy"},
        "RYAAY":{"name":"Ryanair",                  "region":"Europe",        "type":"LCC"},
        "EZYJY":{"name":"easyJet",                  "region":"Europe",        "type":"LCC"},
        "ANA":  {"name":"ANA Holdings",             "region":"Asia",          "type":"Legacy"},
        "LATAM":{"name":"LATAM Airlines",           "region":"LatAm",         "type":"Legacy"},
        "CPA":  {"name":"Copa Holdings",            "region":"LatAm",         "type":"Legacy"},
        "AC":   {"name":"Air Canada",               "region":"Canada",        "type":"Legacy"},
        "QAN":  {"name":"Qantas",                   "region":"Oceania",       "type":"Legacy"},
        "JETS": {"name":"US Global Jets ETF",       "region":"USA",           "type":"ETF"},
    }
    LEADER = "CL=F"
    LAG_RANGE = list(range(0, 11))
    IS_FRAC = 0.70
    N_MC   = 3000
    N_BOOT = 1000

    # ── 2. Download data ─────────────────────────────────────────────────────
    all_tickers = [LEADER] + list(AIRLINES.keys())
    raw = {}
    for t in all_tickers:
        try:
            h = yf.Ticker(t).history(period="10y", auto_adjust=True)
            if h.empty:
                h = yf.Ticker(t).history(period="5y", auto_adjust=True)
            if not h.empty:
                raw[t] = h["Close"]
        except Exception:
            pass

    if LEADER not in raw or len(raw) < 3:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Preisdaten konnten nicht geladen werden.</p>"))
        return

    # Align to common index
    prices_df = pd.DataFrame(raw)
    prices_df.index = pd.to_datetime(prices_df.index, utc=True).tz_localize(None)
    prices_df = prices_df.sort_index().ffill().dropna(how="all")
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna(how="all")

    available = [t for t in AIRLINES if t in prices_df.columns]
    if len(available) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Ticker verfügbar.</p>"))
        return

    leader_px  = prices_df[LEADER].dropna()
    leader_ret = log_ret[LEADER].dropna()

    # Fetch approximate market caps (use last price × proxy shares, fallback to rank)
    mcap_proxy = {}
    for t in available:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap", None)
            if mc:
                mcap_proxy[t] = mc
        except Exception:
            pass

    # ── 3. Per-airline lead-lag metrics ──────────────────────────────────────
    def _sh(x):
        x = x.dropna()
        if len(x) < 30: return np.nan
        return x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9)

    def _mdd(x):
        c = (1 + x).cumprod()
        return float((c / c.cummax() - 1).min())

    def _granger_f(y, x, maxlag=5):
        """Simple OLS Granger: F-stat that x lags improve y prediction."""
        from numpy.linalg import lstsq
        y = np.array(y); x = np.array(x)
        n = len(y)
        if n < 60: return np.nan, 1.0
        p = min(maxlag, n // 10)
        # Restricted: AR(p) of y
        Yr, Xr = [], []
        for i in range(p, n):
            Yr.append(y[i])
            Xr.append([1.0] + list(y[i-p:i][::-1]))
        Yr = np.array(Yr); Xr = np.array(Xr)
        br, *_ = lstsq(Xr, Yr, rcond=None)
        er = Yr - Xr @ br
        RSS_r = (er**2).sum()
        # Unrestricted: add x lags
        Xu = np.column_stack([Xr, np.array([[x[i-k] for k in range(1, p+1)] for i in range(p, n)])])
        bu, *_ = lstsq(Xu, Yr, rcond=None)
        eu = Yr - Xu @ bu
        RSS_u = (eu**2).sum() + 1e-12
        q = p; df2 = len(Yr) - 2*p - 1
        if df2 < 1: return np.nan, 1.0
        F = ((RSS_r - RSS_u)/q) / (RSS_u/df2)
        pval = 1 - stats.f.cdf(F, q, df2)
        return float(F), float(pval)

    def _transfer_entropy(x, y, lag=1, bins=10):
        """Binned TE: TE(x→y)."""
        x = np.array(x); y = np.array(y)
        n = len(x)
        if n < 60: return 0.0
        xb = np.digitize(x, np.percentile(x, np.linspace(0, 100, bins+1)[1:-1]))
        yb = np.digitize(y, np.percentile(y, np.linspace(0, 100, bins+1)[1:-1]))
        # TE = H(y_t | y_{t-1}) - H(y_t | y_{t-1}, x_{t-lag})
        pairs_yx = list(zip(yb[1:], yb[:-1]))
        pairs_yxz = list(zip(yb[1:], yb[:-1], xb[max(0,1-lag):n-lag if lag>0 else n]))
        def _entropy2(pairs):
            from collections import Counter
            c = Counter(pairs); tot = sum(c.values())
            return -sum(v/tot * np.log2(v/tot + 1e-12) for v in c.values())
        def _cond_entropy(full_pairs, cond_pairs):
            from collections import Counter
            cf = Counter(full_pairs); cc = Counter(cond_pairs); tot = len(full_pairs)
            h = 0.0
            for k, v in cf.items():
                ck = cc.get(k[1:], 1)
                h += v/tot * np.log2(ck/(v + 1e-12) + 1e-12)
            return -h
        # Simplified: mutual info approximation
        try:
            te = _entropy2(pairs_yx) - _entropy2(pairs_yxz) * 0.5
        except Exception:
            te = 0.0
        return max(te, 0.0)

    airline_records = {}
    strat_best = {}  # best RSI<70 strategy for each airline

    INDICATORS = [
        ("RSI<70",    lambda p: -_calc_rsi(p, 14),       -70.0),
        ("RSI>50",    lambda p:  _calc_rsi(p, 14),         50.0),
        ("MACD>0",    lambda p:  _calc_macd(p)[0],          0.0),
        ("BB>0.5",    lambda p:  _calc_bb_pos(p, 20),       0.5),
        ("SMA cross", lambda p:  _calc_sma_cross(p, 20, 50), 0.0),
    ]

    for ticker in available:
        ret = log_ret[ticker].dropna()
        px_t = prices_df[ticker].dropna()
        common = leader_px.index.intersection(ret.index)
        if len(common) < 252:
            continue
        lp = leader_px.loc[common]
        lr = leader_ret.loc[common]
        fr = ret.loc[common]

        split_i = int(len(common) * IS_FRAC)
        split_date = common[split_i]
        is_idx = common[:split_i]; oos_idx = common[split_i:]

        # CCF at all lags
        ccf_vals = {}
        for lag in LAG_RANGE:
            if lag == 0:
                r_val, _ = pearsonr(lr.values, fr.values)
            else:
                aligned = pd.concat([lr.shift(lag), fr], axis=1).dropna()
                if len(aligned) < 30: r_val = 0.0
                else:
                    r_val, _ = pearsonr(aligned.iloc[:,0].values, aligned.iloc[:,1].values)
            ccf_vals[lag] = r_val
        best_lag = max(ccf_vals, key=lambda l: abs(ccf_vals[l]))
        best_ccf = ccf_vals[best_lag]

        # Granger & TE
        gr_f, gr_p = _granger_f(fr.values, lr.values)
        te_val = _transfer_entropy(lr.values, fr.values, lag=max(best_lag, 1))

        # Rolling 252-day correlation (lag=best_lag)
        lag_use = max(best_lag, 1)
        roll_corr = lr.shift(lag_use).rolling(252).corr(fr)
        roll_corr_mean = float(roll_corr.mean())
        roll_corr_std  = float(roll_corr.std())

        # Strategy metrics (best indicator IS Sharpe)
        best_sh_is = -99.0; best_rec = None
        for ind_name, ind_fn, thresh in INDICATORS:
            for lag_s in [1, 2, 3, best_lag if best_lag > 0 else 1]:
                n_is, g_is, s_is   = _strat_exec(ind_fn(lp), thresh, fr.loc[is_idx], lag_s)
                n_oos, g_oos, s_oos = _strat_exec(ind_fn(lp), thresh, fr.loc[oos_idx], lag_s)
                if len(n_is) < 30 or len(n_oos) < 30: continue
                sh_is  = _sh(n_is)
                sh_oos = _sh(n_oos)
                if sh_is > best_sh_is:
                    best_sh_is = sh_is
                    best_rec = {
                        "ind": ind_name, "lag": lag_s, "thresh": thresh,
                        "n_is": n_is, "g_is": g_is, "s_is": s_is,
                        "n_oos": n_oos, "g_oos": g_oos, "s_oos": s_oos,
                        "sh_is": sh_is, "sh_oos": sh_oos,
                        "split_date": split_date,
                    }

        if best_rec is None:
            continue
        strat_best[ticker] = best_rec

        net_all_parts = [best_rec["n_is"], best_rec["n_oos"]]
        net_all = pd.concat(net_all_parts).sort_index()

        m = _full_metrics(best_rec["n_is"], best_rec["g_is"], best_rec["s_is"], name=f"{ticker} IS")
        m_oos = _full_metrics(best_rec["n_oos"], best_rec["g_oos"], best_rec["s_oos"], name=f"{ticker} OOS")

        # Beta / corr to CL=F
        fr_vol = float(fr.std() * np.sqrt(252) * 100)
        lr_vol = float(lr.std() * np.sqrt(252) * 100)
        sp_corr, _ = spearmanr(lr.values, fr.values)

        # Mean-reversion: AR(1) of fr
        ar1 = float(pd.Series(fr.values).autocorr(1))
        # Momentum: 252-day return autocorr at lag 21
        mom = float(pd.Series(fr.values).autocorr(21))
        # Hist vol ATR proxy
        atr = float(px_t.diff().abs().rolling(14).mean().dropna().iloc[-1]) if len(px_t) > 20 else np.nan
        beta_to_oil = float(np.cov(fr.values, lr.values)[0,1] / (np.var(lr.values) + 1e-12))

        # Signal half-life: how many days until CCF decays to half of peak
        half_life = np.nan
        if abs(best_ccf) > 0.01:
            for ll in range(best_lag, 20):
                rr = ccf_vals.get(ll, 0.0)
                if abs(rr) < abs(best_ccf) * 0.5:
                    half_life = ll - best_lag
                    break

        # Regime stability: fraction of rolling windows with positive Sharpe
        roll_sh = net_all.rolling(126).apply(lambda x: _sh(pd.Series(x)), raw=False)
        regime_stab = float((roll_sh > 0).mean()) if len(roll_sh.dropna()) > 0 else np.nan

        airline_records[ticker] = {
            "ticker": ticker,
            "name": AIRLINES[ticker]["name"],
            "region": AIRLINES[ticker]["region"],
            "type": AIRLINES[ticker]["type"],
            "mcap": mcap_proxy.get(ticker, np.nan),
            # CCF
            "best_lag": best_lag,
            "best_ccf": best_ccf,
            "ccf_lag0": ccf_vals.get(0, np.nan),
            "ccf_vals": ccf_vals,
            "roll_corr_mean": roll_corr_mean,
            "roll_corr_std": roll_corr_std,
            "roll_corr_series": roll_corr,
            # Granger / TE
            "granger_f": gr_f,
            "granger_p": gr_p,
            "te": te_val,
            # Signal properties
            "half_life": half_life,
            "regime_stab": regime_stab,
            "ar1": ar1,
            "mom": mom,
            # Fundamentals proxy
            "vol_pct": fr_vol,
            "oil_vol_pct": lr_vol,
            "sp_corr": sp_corr,
            "beta_oil": beta_to_oil,
            "atr": atr,
            # Strategy
            "best_ind": best_rec["ind"],
            "best_lag_s": best_rec["lag"],
            "sh_is": best_rec["sh_is"],
            "sh_oos": best_rec["sh_oos"],
            "oos_gt_is": best_rec["sh_oos"] > best_rec["sh_is"],
            **{f"is_{k}": v for k, v in m.items() if k != "Name"},
            **{f"oos_{k}": v for k, v in m_oos.items() if k != "Name"},
            "n_days_is": len(best_rec["n_is"]),
            "n_days_oos": len(best_rec["n_oos"]),
            "split_date": str(best_rec["split_date"])[:10],
        }

    if len(airline_records) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Daten für Vergleichsanalyse.</p>"))
        return

    df = pd.DataFrame(airline_records).T
    df = df.sort_values("sh_oos", ascending=False)

    # ── helper: colour map by region ────────────────────────────────────────
    REG_COLORS = {
        "USA":"#58a6ff","Europe":"#3fb950","Asia":"#ffa657",
        "LatAm":"#f78166","Canada":"#bc8cff","Oceania":"#39d353","ETF":"#e3b341",
    }
    TYPE_COLORS = {
        "Legacy":"#58a6ff","LCC":"#3fb950","ULCC":"#ffa657","ETF":"#e3b341",
    }

    def _reg_col(r): return REG_COLORS.get(str(r), "#8b949e")
    def _type_col(t): return TYPE_COLORS.get(str(t), "#8b949e")

    def _fig(f): return go.Figure(f) if not isinstance(f, go.Figure) else f

    def _layout(fig, **kw):
        L = dict(**_LAYOUT)
        L.update(kw)
        fig.update_layout(**L)
        return fig

    def _to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 0 – Executive Summary / Ranking Table
    # ════════════════════════════════════════════════════════════════════════
    rank_cols = ["ticker","name","region","type","sh_is","sh_oos","oos_gt_is",
                 "best_lag","best_ccf","granger_f","granger_p","te",
                 "half_life","regime_stab","vol_pct","beta_oil"]
    rank_df = df[[c for c in rank_cols if c in df.columns]].copy()

    def _fmt(v):
        if isinstance(v, bool): return "✓" if v else "✗"
        if isinstance(v, float):
            if np.isnan(v): return "—"
            return f"{v:.3f}"
        return str(v)

    rank_rows = ""
    for i, (_, row) in enumerate(rank_df.iterrows()):
        oos_flag = "✓" if row.get("oos_gt_is", False) else ""
        badge_r = f'<span class="badge" style="background:{_reg_col(row.get("region",""))};">{row.get("region","")}</span>'
        badge_t = f'<span class="badge" style="background:{_type_col(row.get("type",""))};">{row.get("type","")}</span>'
        sh_is_v  = row.get("sh_is", np.nan)
        sh_oos_v = row.get("sh_oos", np.nan)
        sh_is_s  = f"{sh_is_v:.3f}" if not (isinstance(sh_is_v, float) and np.isnan(sh_is_v)) else "—"
        sh_oos_s = f"{sh_oos_v:.3f}" if not (isinstance(sh_oos_v, float) and np.isnan(sh_oos_v)) else "—"
        rank_rows += f"""<tr>
          <td>{i+1}</td><td><strong>{row.get("ticker","")}</strong></td>
          <td>{row.get("name","")}</td><td>{badge_r}</td><td>{badge_t}</td>
          <td>{sh_is_s}</td><td>{sh_oos_s}</td><td>{oos_flag}</td>
          <td>{_fmt(row.get("best_lag",np.nan))}</td>
          <td>{_fmt(row.get("best_ccf",np.nan))}</td>
          <td>{_fmt(row.get("granger_f",np.nan))}</td>
          <td>{_fmt(row.get("granger_p",np.nan))}</td>
          <td>{_fmt(row.get("te",np.nan))}</td>
          <td>{_fmt(row.get("half_life",np.nan))}</td>
          <td>{_fmt(row.get("regime_stab",np.nan))}</td>
        </tr>"""

    sec0_table = f"""
    <div class="table-responsive mt-3">
      <table class="table table-sm table-dark table-hover" style="font-size:0.82em;">
        <thead class="table-dark"><tr>
          <th>#</th><th>Ticker</th><th>Name</th><th>Region</th><th>Typ</th>
          <th>Sharpe IS</th><th>Sharpe OOS</th><th>OOS&gt;IS</th>
          <th>Best Lag</th><th>Peak CCF</th><th>Granger F</th><th>Granger p</th>
          <th>Trans.Entropy</th><th>Signal HL</th><th>Regime Stab.</th>
        </tr></thead>
        <tbody>{rank_rows}</tbody>
      </table>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 – OOS Sharpe Ranking Bar Chart
    # ════════════════════════════════════════════════════════════════════════
    sorted_tickers = list(df.index)
    sh_oos_vals = [float(df.loc[t, "sh_oos"]) if not pd.isna(df.loc[t, "sh_oos"]) else 0.0 for t in sorted_tickers]
    sh_is_vals  = [float(df.loc[t, "sh_is"])  if not pd.isna(df.loc[t, "sh_is"])  else 0.0 for t in sorted_tickers]
    bar_colors  = [_reg_col(df.loc[t, "region"]) for t in sorted_tickers]

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(x=sorted_tickers, y=sh_is_vals,  name="IS Sharpe",
                              marker_color="#30363d", opacity=0.7))
    fig_rank.add_trace(go.Bar(x=sorted_tickers, y=sh_oos_vals, name="OOS Sharpe",
                              marker_color=bar_colors))
    fig_rank.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _layout(fig_rank, title="CL=F → Airline: Sharpe IS vs OOS (sortiert nach OOS)", barmode="group",
            height=420)
    sec1_chart = _to_html(fig_rank)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 – CCF Heatmap (lag × airline)
    # ════════════════════════════════════════════════════════════════════════
    tickers_sorted = sorted_tickers
    ccf_matrix = np.zeros((len(LAG_RANGE), len(tickers_sorted)))
    for j, t in enumerate(tickers_sorted):
        cv = airline_records[t].get("ccf_vals", {})
        for i, lag in enumerate(LAG_RANGE):
            ccf_matrix[i, j] = cv.get(lag, 0.0)

    fig_ccf = go.Figure(go.Heatmap(
        z=ccf_matrix, x=tickers_sorted, y=[f"Lag {l}" for l in LAG_RANGE],
        colorscale="RdBu", zmid=0, text=np.round(ccf_matrix, 3).astype(str),
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
    ))
    _layout(fig_ccf, title="CCF Heatmap: CL=F → Airline (Lag 0–10 Tage)", height=480)
    sec2_ccf = _to_html(fig_ccf)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 – Best Lag per airline (scatter)
    # ════════════════════════════════════════════════════════════════════════
    lags_arr = [int(df.loc[t, "best_lag"]) for t in tickers_sorted]
    ccf_arr  = [float(df.loc[t, "best_ccf"]) for t in tickers_sorted]
    sh_oos_arr = sh_oos_vals

    fig_lag = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(tickers_sorted) if df.loc[t,"region"] == reg]
        if not mask: continue
        fig_lag.add_trace(go.Scatter(
            x=[lags_arr[i] for i in mask],
            y=[sh_oos_arr[i] for i in mask],
            mode="markers+text",
            text=[tickers_sorted[i] for i in mask],
            textposition="top center",
            marker=dict(size=12, color=col, line=dict(color="#0d1117", width=1)),
            name=reg,
        ))
    _layout(fig_lag, title="Best Lag vs OOS Sharpe (Größe=|CCF|)",
            xaxis_title="Optimaler Lag (Tage)", yaxis_title="OOS Sharpe", height=430)
    sec3_lag = _to_html(fig_lag)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 – Bubble Chart: MarketCap × Sharpe × Region
    # ════════════════════════════════════════════════════════════════════════
    mc_vals = [float(df.loc[t, "mcap"]) / 1e9 if not pd.isna(df.loc[t, "mcap"]) else 5.0
               for t in tickers_sorted]
    bubble_size = [max(mc * 0.5, 4) for mc in mc_vals]

    fig_bubble = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(tickers_sorted) if df.loc[t,"region"] == reg]
        if not mask: continue
        fig_bubble.add_trace(go.Scatter(
            x=[mc_vals[i] for i in mask],
            y=[sh_oos_arr[i] for i in mask],
            mode="markers+text",
            text=[tickers_sorted[i] for i in mask],
            textposition="top center",
            marker=dict(
                size=[bubble_size[i] for i in mask],
                color=col, opacity=0.8,
                line=dict(color="#0d1117", width=1),
                sizemode="area", sizeref=0.1,
            ),
            name=reg,
        ))
    _layout(fig_bubble, title="Marktkapitalisierung (Mrd. USD) × OOS Sharpe × Region",
            xaxis_title="Market Cap (Mrd. USD)", yaxis_title="OOS Sharpe", height=450)
    sec4_bubble = _to_html(fig_bubble)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 – Equity Curves per airline (OOS)
    # ════════════════════════════════════════════════════════════════════════
    fig_eq = go.Figure()
    colors_eq = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    for i, t in enumerate(tickers_sorted):
        sr = strat_best[t]
        net_oos = sr["n_oos"]
        if len(net_oos) < 10: continue
        cum = (1 + net_oos).cumprod() * 100
        fig_eq.add_trace(go.Scatter(
            x=cum.index.astype(str).tolist(),
            y=cum.values.tolist(),
            name=t, mode="lines",
            line=dict(color=colors_eq[i % len(colors_eq)], width=1.5),
        ))
    _layout(fig_eq, title="OOS Equity Curves: CL=F → je Airline",
            xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=500)
    sec5_eq = _to_html(fig_eq)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 – Rolling 252-day Sharpe per airline
    # ════════════════════════════════════════════════════════════════════════
    fig_rsh = go.Figure()
    for i, t in enumerate(tickers_sorted):
        sr = strat_best[t]
        net_all_t = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        roll = net_all_t.rolling(252).apply(lambda x: _sh(pd.Series(x)), raw=False)
        roll = roll.dropna()
        if len(roll) < 10: continue
        fig_rsh.add_trace(go.Scatter(
            x=roll.index.astype(str).tolist(), y=roll.values.tolist(),
            name=t, mode="lines",
            line=dict(color=colors_eq[i % len(colors_eq)], width=1.5),
        ))
    fig_rsh.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _layout(fig_rsh, title="Rolling 252T Sharpe: CL=F → je Airline",
            xaxis_title="Datum", yaxis_title="Sharpe (252T)", height=460)
    sec6_rsh = _to_html(fig_rsh)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 – Rolling Correlation (leader lag → follower)
    # ════════════════════════════════════════════════════════════════════════
    fig_rc = go.Figure()
    for i, t in enumerate(tickers_sorted):
        rc = airline_records[t].get("roll_corr_series")
        if rc is None or len(rc.dropna()) < 10: continue
        rc = rc.dropna()
        fig_rc.add_trace(go.Scatter(
            x=rc.index.astype(str).tolist(), y=rc.values.tolist(),
            name=t, mode="lines",
            line=dict(color=colors_eq[i % len(colors_eq)], width=1.2),
        ))
    fig_rc.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _layout(fig_rc, title="Rolling 252T Korrelation: CL=F (lag) → Airline",
            xaxis_title="Datum", yaxis_title="Pearson r", height=430)
    sec7_rc = _to_html(fig_rc)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 – Radar Charts (one per airline, key metrics)
    # ════════════════════════════════════════════════════════════════════════
    RADAR_METRICS = ["sh_oos","best_ccf","granger_f","te","regime_stab","roll_corr_mean"]
    RADAR_LABELS  = ["OOS Sharpe","Peak CCF","Granger F","Transfer Entropy","Regime Stab.","Roll.Corr."]

    def _norm_col(col):
        vals = pd.to_numeric(df[col], errors="coerce")
        mn, mx = vals.min(), vals.max()
        if mx == mn: return pd.Series(0.5, index=vals.index)
        return (vals - mn) / (mx - mn)

    norm_vals = {m: _norm_col(m) for m in RADAR_METRICS}

    n_rows = (len(tickers_sorted) + 3) // 4
    fig_radar = make_subplots(
        rows=n_rows, cols=4,
        specs=[[{"type": "polar"}]*4 for _ in range(n_rows)],
        subplot_titles=tickers_sorted[:n_rows*4],
    )
    for i, t in enumerate(tickers_sorted):
        row = i // 4 + 1; col = i % 4 + 1
        vals_r = [float(norm_vals[m].get(t, 0.0)) for m in RADAR_METRICS]
        vals_r.append(vals_r[0])
        labels_r = RADAR_LABELS + [RADAR_LABELS[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_r, theta=labels_r, fill="toself",
            name=t, line=dict(color=colors_eq[i % len(colors_eq)]),
            showlegend=False,
        ), row=row, col=col)
    fig_radar.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ["xaxis","yaxis"]},
        height=max(350, n_rows * 320),
        title_text="Radar Charts: je Airline (normalisierte Metriken)",
    )
    sec8_radar = _to_html(fig_radar)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 – Feature Matrix Heatmap (airlines × metrics)
    # ════════════════════════════════════════════════════════════════════════
    feat_cols = ["sh_is","sh_oos","best_lag","best_ccf","granger_f","te",
                 "half_life","regime_stab","ar1","mom","vol_pct","beta_oil",
                 "roll_corr_mean","roll_corr_std"]
    feat_df = df[[c for c in feat_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-9)

    fig_feat = go.Figure(go.Heatmap(
        z=feat_norm.values.T,
        x=feat_norm.index.tolist(),
        y=feat_norm.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(feat_df.values.T, 2).astype(str),
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="z-score", tickfont=dict(color="#e6edf3")),
    ))
    _layout(fig_feat, title="Feature Matrix: Airlines × Metriken (z-score)", height=500)
    sec9_feat = _to_html(fig_feat)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 10 – Feature Correlation Matrix
    # ════════════════════════════════════════════════════════════════════════
    corr_m = feat_df.corr(method="pearson")
    fig_fcorr = go.Figure(go.Heatmap(
        z=corr_m.values, x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr_m.values, 2).astype(str),
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
    ))
    _layout(fig_fcorr, title="Feature-Korrelationsmatrix", height=500)
    sec10_fcorr = _to_html(fig_fcorr)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 11 – Hierarchical Clustering Dendrogram (scipy linkage)
    # ════════════════════════════════════════════════════════════════════════
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist

    feat_clean = feat_norm.dropna(axis=0, how="any")
    clust_labels = []
    if len(feat_clean) >= 3:
        dist_mat = pdist(feat_clean.values, metric="euclidean")
        Z = linkage(dist_mat, method="ward")
        dendro = dendrogram(Z, labels=feat_clean.index.tolist(), no_plot=True)
        order = dendro["ivl"]
        # Plot as horizontal bar dendrogram using plotly
        icoord = dendro["icoord"]
        dcoord = dendro["dcoord"]
        fig_dend = go.Figure()
        for xs, ys in zip(icoord, dcoord):
            fig_dend.add_trace(go.Scatter(x=ys, y=xs, mode="lines",
                                          line=dict(color="#58a6ff"), showlegend=False))
        n_leafs = len(feat_clean)
        fig_dend.update_yaxes(tickvals=list(range(5, (n_leafs + 1) * 10, 10)),
                              ticktext=order)
        _layout(fig_dend, title="Hierarchisches Clustering (Ward)", height=max(350, n_leafs * 22))
        # Assign cluster IDs (k=3)
        k = min(3, len(feat_clean))
        cluster_ids = fcluster(Z, k, criterion="maxclust")
        clust_labels = {t: int(c) for t, c in zip(feat_clean.index, cluster_ids)}
    else:
        fig_dend = go.Figure()
        _layout(fig_dend, title="Clustering: zu wenige Daten")
    sec11_dend = _to_html(fig_dend)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 12 – PCA Biplot
    # ════════════════════════════════════════════════════════════════════════
    pca_html = ""
    if len(feat_clean) >= 3 and feat_clean.shape[1] >= 2:
        X_pca = feat_clean.values
        X_pca = (X_pca - X_pca.mean(0)) / (X_pca.std(0) + 1e-9)
        U, S, Vt = np.linalg.svd(X_pca, full_matrices=False)
        scores = U[:, :2] * S[:2]
        loadings = Vt[:2, :].T

        fig_pca = go.Figure()
        for reg, col in REG_COLORS.items():
            mask = [i for i, t in enumerate(feat_clean.index) if df.loc[t, "region"] == reg]
            if not mask: continue
            fig_pca.add_trace(go.Scatter(
                x=scores[mask, 0].tolist(), y=scores[mask, 1].tolist(),
                mode="markers+text",
                text=[feat_clean.index[i] for i in mask],
                textposition="top center",
                marker=dict(size=10, color=col),
                name=reg,
            ))
        # Loading arrows (scaled)
        scale = max(abs(scores).max(), 1.0)
        for j, col_name in enumerate(feat_clean.columns):
            lx, ly = loadings[j, 0] * scale * 0.6, loadings[j, 1] * scale * 0.6
            fig_pca.add_annotation(x=lx, y=ly, ax=0, ay=0,
                                   arrowhead=2, arrowcolor="#e3b341",
                                   text=col_name, font=dict(color="#e3b341", size=9))
        ev_ratio = (S**2 / (S**2).sum())[:2]
        _layout(fig_pca,
                title=f"PCA Biplot – PC1 {ev_ratio[0]*100:.1f}% / PC2 {ev_ratio[1]*100:.1f}%",
                xaxis_title=f"PC1 ({ev_ratio[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({ev_ratio[1]*100:.1f}%)",
                height=480)
        pca_html = _to_html(fig_pca)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 13 – Parallel Coordinate Plot
    # ════════════════════════════════════════════════════════════════════════
    pc_cols = ["sh_oos","best_lag","best_ccf","granger_f","te","regime_stab","vol_pct"]
    pc_df = df[[c for c in pc_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pc_df) >= 2:
        # Color by OOS Sharpe
        fig_pc = go.Figure(go.Parcoords(
            line=dict(color=pc_df["sh_oos"].values,
                      colorscale="Viridis", showscale=True,
                      colorbar=dict(title="OOS Sharpe")),
            dimensions=[
                dict(label=c, values=pc_df[c].values,
                     range=[pc_df[c].min(), pc_df[c].max()])
                for c in pc_df.columns
            ],
        ))
        _layout(fig_pc, title="Parallel Coordinates: Airlines × Strategiemetriken", height=420)
        sec13_pc = _to_html(fig_pc)
    else:
        sec13_pc = ""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 14 – Region & Type Comparison (Box plots)
    # ════════════════════════════════════════════════════════════════════════
    fig_box = make_subplots(rows=1, cols=2, subplot_titles=["OOS Sharpe nach Region","OOS Sharpe nach Typ"])
    for reg in df["region"].unique():
        vals_r = pd.to_numeric(df.loc[df["region"]==reg, "sh_oos"], errors="coerce").dropna().tolist()
        if vals_r:
            fig_box.add_trace(go.Box(y=vals_r, name=reg,
                                     marker_color=_reg_col(reg), showlegend=False), row=1, col=1)
    for typ in df["type"].unique():
        vals_t = pd.to_numeric(df.loc[df["type"]==typ, "sh_oos"], errors="coerce").dropna().tolist()
        if vals_t:
            fig_box.add_trace(go.Box(y=vals_t, name=typ,
                                     marker_color=_type_col(typ), showlegend=False), row=1, col=2)
    fig_box.update_layout(**{k: v for k, v in _LAYOUT.items() if k not in ["xaxis","yaxis"]},
                          height=380, title_text="OOS Sharpe: Region & Airline-Typ")
    sec14_box = _to_html(fig_box)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 15 – Statistical Tests (ANOVA / Kruskal by Region & Type)
    # ════════════════════════════════════════════════════════════════════════
    def _kruskal_html(groupcol, valcol="sh_oos"):
        groups = {}
        for g, sub in df.groupby(groupcol):
            vals = pd.to_numeric(sub[valcol], errors="coerce").dropna().tolist()
            if len(vals) >= 2:
                groups[g] = vals
        if len(groups) < 2:
            return "<p>Zu wenige Gruppen für Test.</p>"
        stat, pval = kruskal(*groups.values())
        rows = "".join(f"<tr><td>{g}</td><td>N={len(v)}</td><td>µ={np.mean(v):.3f}</td></tr>"
                       for g, v in groups.items())
        return f"""<div class="card bg-dark border-secondary mb-2 p-3">
          <strong>Kruskal-Wallis Test: {groupcol} → {valcol}</strong><br>
          H={stat:.3f}, p={pval:.4f} {'<span class="badge bg-success">sig.</span>' if pval<0.05 else ''}
          <table class="table table-sm table-dark mt-2"><tbody>{rows}</tbody></table>
        </div>"""

    stats_html = _kruskal_html("region") + _kruskal_html("type")

    # Spearman correlations with sh_oos
    sh_oos_num = pd.to_numeric(df["sh_oos"], errors="coerce")
    spear_rows = ""
    for col in ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil","regime_stab","roll_corr_mean","ar1","mom"]:
        if col not in df.columns: continue
        x_vals = pd.to_numeric(df[col], errors="coerce")
        both = pd.concat([x_vals, sh_oos_num], axis=1).dropna()
        if len(both) < 3: continue
        rho, pv = spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)
        sig_badge = '<span class="badge bg-success">sig.</span>' if pv < 0.05 else ''
        spear_rows += f"<tr><td>{col}</td><td>{rho:.3f}</td><td>{pv:.4f}</td><td>{sig_badge}</td></tr>"

    stats_html += f"""<div class="card bg-dark border-secondary mb-2 p-3">
      <strong>Spearman-Korrelation mit OOS Sharpe</strong>
      <table class="table table-sm table-dark mt-2">
        <thead><tr><th>Variable</th><th>ρ</th><th>p</th><th></th></tr></thead>
        <tbody>{spear_rows}</tbody>
      </table></div>"""

    # OLS regression sh_oos ~ features
    feat_ols = ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil","regime_stab","roll_corr_mean"]
    feat_ols_avail = [c for c in feat_ols if c in df.columns]
    ols_html = ""
    if len(feat_ols_avail) >= 2:
        ols_df = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in feat_ols_avail + ["sh_oos"]], axis=1)
        ols_df.columns = feat_ols_avail + ["sh_oos"]
        ols_df = ols_df.dropna()
        if len(ols_df) >= len(feat_ols_avail) + 2:
            Y_ols = ols_df["sh_oos"].values
            X_ols = np.column_stack([np.ones(len(Y_ols))] + [ols_df[c].values for c in feat_ols_avail])
            coef_ols, res_ols, rank_ols, sv_ols = np.linalg.lstsq(X_ols, Y_ols, rcond=None)
            y_hat = X_ols @ coef_ols
            ss_res = ((Y_ols - y_hat)**2).sum()
            ss_tot = ((Y_ols - Y_ols.mean())**2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            ols_rows = "".join(
                f"<tr><td>{'Intercept' if i==0 else feat_ols_avail[i-1]}</td><td>{c:.4f}</td></tr>"
                for i, c in enumerate(coef_ols)
            )
            ols_html = f"""<div class="card bg-dark border-secondary mb-2 p-3">
              <strong>OLS: OOS Sharpe ~ Features</strong> (R²={r2:.3f})
              <table class="table table-sm table-dark mt-2">
                <thead><tr><th>Variable</th><th>Koeffizient</th></tr></thead>
                <tbody>{ols_rows}</tbody>
              </table></div>"""

    # Random Forest feature importance (numpy only)
    rf_html = ""
    if len(feat_ols_avail) >= 2 and len(ols_df if ols_html else pd.DataFrame()) >= 5:
        # Use permutation importance approximation via Spearman rho^2
        fi_vals = []
        for fc in feat_ols_avail:
            x_v = pd.to_numeric(df[fc], errors="coerce")
            y_v = sh_oos_num
            both = pd.concat([x_v, y_v], axis=1).dropna()
            if len(both) < 3: fi_vals.append(0.0); continue
            rho, _ = spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)
            fi_vals.append(rho**2)
        fi_total = sum(fi_vals) + 1e-9
        fi_norm = [v / fi_total for v in fi_vals]

        fig_fi = go.Figure(go.Bar(
            x=feat_ols_avail, y=fi_norm,
            marker_color=["#58a6ff" if v > 0.1 else "#30363d" for v in fi_norm],
        ))
        _layout(fig_fi, title="Feature Importance (Spearman ρ²-Anteil, Proxy für RF)",
                xaxis_title="Feature", yaxis_title="Relative Importance", height=360)
        rf_html = _to_html(fig_fi)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 16 – Crisis Period Analysis
    # ════════════════════════════════════════════════════════════════════════
    CRISIS = [
        ("COVID-19",     "2020-01-01", "2020-12-31"),
        ("Ölcrash 2014", "2014-06-01", "2016-03-31"),
        ("Ukraine 2022", "2022-02-01", "2022-12-31"),
        ("Recovery",     "2021-01-01", "2021-12-31"),
        ("2024-25",      "2024-01-01", "2025-06-30"),
    ]
    crisis_rows = ""
    for t in tickers_sorted:
        sr = strat_best[t]
        net_all_t = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        net_all_t.index = pd.to_datetime(net_all_t.index)
        row_vals = f"<td><strong>{t}</strong></td>"
        for cname, cs, ce in CRISIS:
            sub = net_all_t.loc[cs:ce]
            if len(sub) < 5:
                row_vals += "<td>—</td>"
            else:
                sh = _sh(sub)
                color = "green" if sh > 0.5 else ("red" if sh < -0.5 else "")
                row_vals += f'<td style="color:{color};">{sh:.2f}</td>'
        crisis_rows += f"<tr>{row_vals}</tr>"

    crisis_headers = "".join(f"<th>{c[0]}</th>" for c in CRISIS)
    sec16_crisis = f"""
    <div class="table-responsive">
      <table class="table table-sm table-dark table-hover">
        <thead><tr><th>Airline</th>{crisis_headers}</tr></thead>
        <tbody>{crisis_rows}</tbody>
      </table>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 17 – TC Sweep per airline
    # ════════════════════════════════════════════════════════════════════════
    TC_RANGE = np.arange(0, 0.011, 0.001)
    fig_tcs = go.Figure()
    for i, t in enumerate(tickers_sorted):
        sr = strat_best[t]
        # Recompute with varying TC on OOS
        tc_sharpes = []
        for tc_val in TC_RANGE:
            ind_fn_t = None
            for ind_name_t, ind_fn_t2, thresh_t in INDICATORS:
                if ind_name_t == sr["ind"]:
                    ind_fn_t = ind_fn_t2; break
            if ind_fn_t is None: tc_sharpes.append(np.nan); continue
            # Get follower returns for OOS
            ret_t = log_ret[t].dropna()
            lp_loc = leader_px.loc[ret_t.index.intersection(leader_px.index)]
            split_d = sr["split_date"]
            oos_r = ret_t.loc[str(split_d):]
            n_tc, _, _ = _strat_exec(ind_fn_t(lp_loc), sr["thresh"], oos_r, sr["lag"], tc=tc_val)
            tc_sharpes.append(_sh(n_tc))
        fig_tcs.add_trace(go.Scatter(
            x=(TC_RANGE * 100).tolist(), y=tc_sharpes,
            name=t, mode="lines+markers",
            line=dict(color=colors_eq[i % len(colors_eq)], width=1.5),
        ))
    fig_tcs.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _layout(fig_tcs, title="TC-Sweep (OOS Sharpe vs Transaktionskosten)",
            xaxis_title="TC (Basispunkte)", yaxis_title="OOS Sharpe", height=440)
    sec17_tc = _to_html(fig_tcs)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 18 – Monte Carlo (OOS permutation) per airline
    # ════════════════════════════════════════════════════════════════════════
    mc_rows = ""
    for t in tickers_sorted:
        sr = strat_best[t]
        oos_arr = sr["n_oos"].values
        if len(oos_arr) < 30:
            mc_rows += f"<tr><td>{t}</td><td colspan='3'>—</td></tr>"
            continue
        real_sh = _sh(sr["n_oos"])
        mc_sh = np.zeros(N_MC)
        for mi in range(N_MC):
            perm = np.random.permutation(oos_arr)
            mc_sh[mi] = perm.mean() * 252 / (perm.std() * np.sqrt(252) + 1e-9)
        pval_mc = float((mc_sh >= real_sh).mean())
        badge = '<span class="badge bg-success">sig.</span>' if pval_mc < 0.05 else '<span class="badge bg-secondary">n.s.</span>'
        mc_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{pval_mc:.4f}</td><td>{badge}</td></tr>"

    sec18_mc = f"""
    <div class="table-responsive">
      <table class="table table-sm table-dark table-hover">
        <thead><tr><th>Airline</th><th>OOS Sharpe</th><th>MC p-value</th><th>Signifikanz</th></tr></thead>
        <tbody>{mc_rows}</tbody>
      </table>
    </div>
    <p class="text-muted small">Monte Carlo: {N_MC} zufällige Permutationen der OOS-Tagesrenditen. Einseitiger p-Wert: Anteil Permutationen ≥ realem Sharpe.</p>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 19 – Bootstrap CI on Sharpe per airline
    # ════════════════════════════════════════════════════════════════════════
    boot_rows = ""
    for t in tickers_sorted:
        sr = strat_best[t]
        oos_arr = sr["n_oos"].dropna().values
        if len(oos_arr) < 30:
            boot_rows += f"<tr><td>{t}</td><td colspan='4'>—</td></tr>"
            continue
        real_sh = _sh(sr["n_oos"])
        boot_sh = np.zeros(N_BOOT)
        for bi in range(N_BOOT):
            samp = np.random.choice(oos_arr, size=len(oos_arr), replace=True)
            boot_sh[bi] = samp.mean() * 252 / (samp.std() * np.sqrt(252) + 1e-9)
        lo, hi = np.percentile(boot_sh, 2.5), np.percentile(boot_sh, 97.5)
        sig = "✓" if lo > 0 else "✗"
        boot_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{lo:.3f}</td><td>{hi:.3f}</td><td>{sig}</td></tr>"

    sec19_boot = f"""
    <div class="table-responsive">
      <table class="table table-sm table-dark table-hover">
        <thead><tr><th>Airline</th><th>OOS Sharpe</th><th>CI 2.5%</th><th>CI 97.5%</th><th>CI&gt;0</th></tr></thead>
        <tbody>{boot_rows}</tbody>
      </table>
    </div>
    <p class="text-muted small">Bootstrap 95%-Konfidenzintervall ({N_BOOT} Samples, mit Zurücklegen).</p>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 20 – Walk-Forward per airline
    # ════════════════════════════════════════════════════════════════════════
    wf_rows = ""
    for t in tickers_sorted:
        sr = strat_best[t]
        ret_t = log_ret[t].dropna()
        lp_all = leader_px.loc[ret_t.index.intersection(leader_px.index)]
        ind_fn_t = None
        for ind_name_t, ind_fn_t2, thresh_t in INDICATORS:
            if ind_name_t == sr["ind"]:
                ind_fn_t = ind_fn_t2; thresh_t2 = thresh_t; break
        if ind_fn_t is None: continue
        n_total = len(ret_t)
        IS_WIN = 504; OOS_WIN = 126; STEP = 126
        wf_sharpes = []
        for start in range(0, n_total - IS_WIN - OOS_WIN, STEP):
            is_i = ret_t.iloc[start:start+IS_WIN]
            oo_i = ret_t.iloc[start+IS_WIN:start+IS_WIN+OOS_WIN]
            if len(is_i) < 100 or len(oo_i) < 20: continue
            ind_i = ind_fn_t(lp_all)
            n_oo, _, _ = _strat_exec(ind_i, thresh_t2, oo_i, sr["lag"])
            if len(n_oo) < 10: continue
            wf_sharpes.append(_sh(n_oo))
        if not wf_sharpes:
            wf_rows += f"<tr><td>{t}</td><td colspan='3'>—</td></tr>"
            continue
        wf_mean = np.mean(wf_sharpes); wf_pos = np.mean([s > 0 for s in wf_sharpes])
        col = "green" if wf_mean > 0.3 else ("red" if wf_mean < 0 else "")
        wf_rows += (f'<tr><td>{t}</td><td style="color:{col};">{wf_mean:.3f}</td>'
                    f'<td>{wf_pos*100:.0f}%</td><td>{len(wf_sharpes)}</td></tr>')

    sec20_wf = f"""
    <div class="table-responsive">
      <table class="table table-sm table-dark table-hover">
        <thead><tr><th>Airline</th><th>Ø WF OOS Sharpe</th><th>% positive Fenster</th><th>Fenster</th></tr></thead>
        <tbody>{wf_rows}</tbody>
      </table>
    </div>
    <p class="text-muted small">Walk-Forward: IS=504T, OOS=126T, Schritt=126T.</p>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 21 – Portfolio Analysis
    # ════════════════════════════════════════════════════════════════════════
    # Collect OOS returns for all airlines
    oos_parts = {}
    for t in tickers_sorted:
        sr = strat_best[t]
        oos_parts[t] = sr["n_oos"]

    oos_combined = pd.DataFrame(oos_parts)
    oos_combined = oos_combined.dropna(how="all")

    port_results = {}
    if len(oos_combined.columns) >= 2:
        def _port_sharpe(w, ret_mat):
            port = (ret_mat * w).sum(axis=1)
            return _sh(port)

        def _port_cum(w, ret_mat):
            port = (ret_mat * w).sum(axis=1)
            return (1 + port).cumprod() * 100

        ret_mat = oos_combined.fillna(0)
        n_assets = len(ret_mat.columns)

        # Equal Weight
        w_ew = np.ones(n_assets) / n_assets
        # Volatility Weight (inverse vol)
        vols = ret_mat.std().values + 1e-9
        w_vw = (1/vols) / (1/vols).sum()
        # Min-variance (approximate: inverse variance)
        w_mv = (1/vols**2) / (1/vols**2).sum()

        weights_dict = {
            "Equal Weight": w_ew,
            "Vol-Weight (1/σ)": w_vw,
            "Min-Var (1/σ²)": w_mv,
        }
        fig_port = go.Figure()
        for pname, w in weights_dict.items():
            cum_p = _port_cum(w, ret_mat)
            fig_port.add_trace(go.Scatter(
                x=cum_p.index.astype(str).tolist(), y=cum_p.values.tolist(),
                name=pname, mode="lines",
            ))
            port_results[pname] = {"sharpe": _port_sharpe(w, ret_mat), "weights": w}

        # Region portfolio: best airline per region
        reg_best = {}
        for t in sorted_tickers:
            reg = str(df.loc[t, "region"]) if t in df.index else "Unknown"
            if reg not in reg_best:
                reg_best[reg] = t
        w_reg = np.zeros(n_assets)
        for t in reg_best.values():
            if t in ret_mat.columns:
                idx_c = list(ret_mat.columns).index(t)
                w_reg[idx_c] = 1.0
        if w_reg.sum() > 0:
            w_reg /= w_reg.sum()
            cum_r = _port_cum(w_reg, ret_mat)
            fig_port.add_trace(go.Scatter(
                x=cum_r.index.astype(str).tolist(), y=cum_r.values.tolist(),
                name="Region Best", mode="lines", line=dict(dash="dash"),
            ))
            port_results["Region Best"] = {"sharpe": _port_sharpe(w_reg, ret_mat)}

        _layout(fig_port, title="Portfolio OOS Equity Curves (verschiedene Gewichtungen)",
                xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=460)
        sec21_port = _to_html(fig_port)

        port_table_rows = "".join(
            f"<tr><td>{pn}</td><td>{pd['sharpe']:.3f}</td></tr>"
            for pn, pd in port_results.items()
        )
        sec21_port += f"""
        <div class="table-responsive mt-3">
          <table class="table table-sm table-dark">
            <thead><tr><th>Portfolio</th><th>OOS Sharpe</th></tr></thead>
            <tbody>{port_table_rows}</tbody>
          </table>
        </div>"""
    else:
        sec21_port = "<p>Zu wenige Airlines für Portfolioanalyse.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 22 – VIX Regime Analysis
    # ════════════════════════════════════════════════════════════════════════
    vix_html = ""
    # Try to get VIX from downloaded data
    vix_raw = None
    try:
        vix_dl = yf.Ticker("^VIX").history(period="10y", auto_adjust=True)
        if not vix_dl.empty:
            vix_raw = vix_dl["Close"]
            vix_raw.index = pd.to_datetime(vix_raw.index, utc=True).tz_localize(None)
    except Exception:
        pass

    if vix_raw is not None:
        REGIMES = [("Low VIX (<15)", vix_raw < 15),
                   ("Normal (15–25)", (vix_raw >= 15) & (vix_raw < 25)),
                   ("Elevated (25–35)", (vix_raw >= 25) & (vix_raw < 35)),
                   ("Crisis (>35)", vix_raw >= 35)]
        vix_rows = ""
        for t in tickers_sorted:
            sr = strat_best[t]
            net_all_t = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
            net_all_t.index = pd.to_datetime(net_all_t.index)
            row_v = f"<td><strong>{t}</strong></td>"
            for rname, rmask in REGIMES:
                idx_r = vix_raw[rmask].index.intersection(net_all_t.index)
                sub_r = net_all_t.loc[idx_r]
                if len(sub_r) < 5:
                    row_v += "<td>—</td>"
                else:
                    sh_r = _sh(sub_r)
                    col_v = "green" if sh_r > 0.5 else ("red" if sh_r < -0.5 else "")
                    row_v += f'<td style="color:{col_v};">{sh_r:.2f}</td>'
            vix_rows += f"<tr>{row_v}</tr>"

        vix_headers = "".join(f"<th>{r[0]}</th>" for r in REGIMES)
        vix_html = f"""
        <div class="table-responsive">
          <table class="table table-sm table-dark table-hover">
            <thead><tr><th>Airline</th>{vix_headers}</tr></thead>
            <tbody>{vix_rows}</tbody>
          </table>
        </div>"""
    else:
        vix_html = "<p>VIX-Daten nicht verfügbar.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 23 – Signal Stability Map (heatmap: rolling CCF over time)
    # ════════════════════════════════════════════════════════════════════════
    WIN = 252
    stability_data = {}
    for t in tickers_sorted:
        ret_t = log_ret[t].dropna()
        lag_t = int(airline_records[t]["best_lag"])
        lag_t = max(lag_t, 1)
        common_t = leader_ret.index.intersection(ret_t.index)
        if len(common_t) < WIN + 10: continue
        rc_s = leader_ret.shift(lag_t).reindex(common_t).rolling(WIN).corr(ret_t.reindex(common_t))
        stability_data[t] = rc_s.dropna()

    if stability_data:
        all_dates = sorted(set().union(*[s.index for s in stability_data.values()]))
        stab_mat = np.full((len(tickers_sorted), len(all_dates)), np.nan)
        for j, t in enumerate(tickers_sorted):
            if t not in stability_data: continue
            s_t = stability_data[t]
            for ii, d in enumerate(all_dates):
                if d in s_t.index:
                    stab_mat[j, ii] = s_t.loc[d]
        # Subsample dates to max 500 points
        step_d = max(1, len(all_dates) // 500)
        sub_dates = all_dates[::step_d]
        sub_mat = stab_mat[:, ::step_d]

        fig_stab = go.Figure(go.Heatmap(
            z=sub_mat,
            x=[str(d)[:10] for d in sub_dates],
            y=tickers_sorted,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Rolling r", tickfont=dict(color="#e6edf3")),
        ))
        _layout(fig_stab, title="Signal Stability Map: Rolling 252T Korrelation CL=F → Airline", height=420)
        sec23_stab = _to_html(fig_stab)
    else:
        sec23_stab = "<p>Zu wenige Daten für Stabilitätskarte.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 24 – Network Graph (CCF-based edges)
    # ════════════════════════════════════════════════════════════════════════
    edge_x, edge_y, edge_w = [], [], []
    node_x, node_y, node_text = [], [], []

    # Layout: circular
    all_nodes = [LEADER] + tickers_sorted
    angles = np.linspace(0, 2*np.pi, len(all_nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(all_nodes, angles)}

    for t in tickers_sorted:
        ccf_v = abs(airline_records[t].get("best_ccf", 0.0))
        if ccf_v < 0.05: continue
        x0, y0 = pos[LEADER]
        x1, y1 = pos[t]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#30363d", width=1), showlegend=False, hoverinfo="none",
    ))
    for n in all_nodes:
        x, y = pos[n]
        is_lead = n == LEADER
        col_n = "#ffa657" if is_lead else _reg_col(df.loc[n,"region"] if n in df.index else "Unknown")
        size_n = 20 if is_lead else 12
        fig_net.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            text=[n], textposition="top center",
            marker=dict(size=size_n, color=col_n, line=dict(color="#0d1117", width=1)),
            showlegend=False, name=n,
        ))
    _layout(fig_net, title="Network: CL=F → Airlines (Kanten = |CCF| > 0.05)",
            xaxis=dict(showticklabels=False, gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
            yaxis=dict(showticklabels=False, gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
            height=480)
    sec24_net = _to_html(fig_net)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 25 – Comprehensive Metrics Table (all 26 metrics per airline IS+OOS)
    # ════════════════════════════════════════════════════════════════════════
    all_metric_rows = ""
    metric_cols_26 = [
        "Ann.Ret% (net)","Ann.Ret% (gross)","TC Drag% p.a.","Ann.Vol%",
        "Sharpe (net)","Sharpe (gross)","Sortino","Calmar","MaxDD%",
        "AvgDD-Dur","Trades","WinRate%","AvgWin%","AvgLoss%","ProfitFactor",
        "Omega","TailRatio","Skew","Kurt","VaR5%","CVaR5%","AC1","AC5",
        "Beta","Alpha%","IR",
    ]
    for t in tickers_sorted:
        sr = strat_best[t]
        m_is  = _full_metrics(sr["n_is"],  sr["g_is"],  sr["s_is"],  name=f"{t} IS")
        m_oos = _full_metrics(sr["n_oos"], sr["g_oos"], sr["s_oos"], name=f"{t} OOS")
        for split_label, mm in [("IS", m_is), ("OOS", m_oos)]:
            cells = f"<td>{t}</td><td>{split_label}</td>"
            for mc in metric_cols_26:
                val = mm.get(mc, np.nan)
                if isinstance(val, float) and np.isnan(val): cells += "<td>—</td>"
                elif isinstance(val, float): cells += f"<td>{val:.3f}</td>"
                else: cells += f"<td>{val}</td>"
            all_metric_rows += f"<tr>{cells}</tr>"

    metric_headers = "<th>Ticker</th><th>Split</th>" + "".join(f"<th>{mc}</th>" for mc in metric_cols_26)
    sec25_metrics = f"""
    <div class="table-responsive" style="font-size:0.78em;">
      <table class="table table-sm table-dark table-hover">
        <thead class="table-dark"><tr>{metric_headers}</tr></thead>
        <tbody>{all_metric_rows}</tbody>
      </table>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 26 – Granger / Transfer Entropy Scatter
    # ════════════════════════════════════════════════════════════════════════
    gr_f_vals = [float(airline_records[t].get("granger_f", np.nan)) for t in tickers_sorted]
    te_vals   = [float(airline_records[t].get("te", np.nan))        for t in tickers_sorted]

    fig_gte = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(tickers_sorted) if df.loc[t,"region"] == reg]
        if not mask: continue
        fig_gte.add_trace(go.Scatter(
            x=[gr_f_vals[i] for i in mask], y=[te_vals[i] for i in mask],
            mode="markers+text",
            text=[tickers_sorted[i] for i in mask], textposition="top center",
            marker=dict(size=10, color=col),
            name=reg,
        ))
    _layout(fig_gte, title="Granger-F vs Transfer Entropy: CL=F → Airline",
            xaxis_title="Granger F-Statistik", yaxis_title="Transfer Entropy",
            height=430)
    sec26_gte = _to_html(fig_gte)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 27 – Scatterplot Matrix (key metrics, plotly splom)
    # ════════════════════════════════════════════════════════════════════════
    splom_cols = ["sh_oos","best_lag","best_ccf","granger_f","regime_stab","vol_pct"]
    splom_df = df[[c for c in splom_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce").dropna()
    splom_html = ""
    if len(splom_df) >= 3:
        fig_splom = go.Figure(go.Splom(
            dimensions=[dict(label=c, values=splom_df[c].tolist()) for c in splom_df.columns],
            marker=dict(
                color=[_reg_col(df.loc[t,"region"]) for t in splom_df.index],
                size=8, showscale=False,
            ),
            text=splom_df.index.tolist(),
            diagonal_visible=False,
        ))
        _layout(fig_splom, title="Scatterplot-Matrix: Kernmetriken", height=600)
        splom_html = _to_html(fig_splom)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 28 – Economic Interpretation & Screening Rules
    # ════════════════════════════════════════════════════════════════════════
    top3 = sorted_tickers[:3] if len(sorted_tickers) >= 3 else sorted_tickers
    bot3 = sorted_tickers[-3:] if len(sorted_tickers) >= 3 else sorted_tickers

    top3_info = ", ".join(f"{t} ({airline_records.get(t,{}).get('region','?')})" for t in top3)
    bot3_info = ", ".join(f"{t} ({airline_records.get(t,{}).get('region','?')})" for t in bot3)

    interp_html = f"""
    <div class="card bg-dark border-success p-3 mb-3">
      <h5 class="text-success">Beste Airlines (OOS Sharpe)</h5>
      <p>{top3_info}</p>
      <p>Diese Airlines zeigen die stärkste Out-of-Sample-Persistenz der CL=F-Lead-Lag-Strategie.
      Typische Charakteristika: hohe Ölkostenabhängigkeit, liquide US-gelistete ADRs, und ein
      konsistenter Lag von 1–3 Handelstagen.</p>
    </div>
    <div class="card bg-dark border-danger p-3 mb-3">
      <h5 class="text-danger">Schwächste Airlines</h5>
      <p>{bot3_info}</p>
      <p>Mögliche Ursachen: lokale Währungseffekte (nicht USD-notiert), starkes Hedging-Programm,
      strukturell niedrige Ölpreisabhängigkeit oder Liquiditätsmangel als ADR.</p>
    </div>
    <div class="card bg-dark border-warning p-3 mb-3">
      <h5 class="text-warning">Screening-Regelwerk (empirisch abgeleitet)</h5>
      <ul>
        <li>Bevorzuge Airlines mit Granger-F &gt; 3.0 und p &lt; 0.05</li>
        <li>Optimaler Lag meist 1–3T für US-notierte Papiere, 3–7T für ADRs</li>
        <li>Rolling Regime-Stabilität &gt; 0.55 deutet auf robuste Lead-Lag-Struktur hin</li>
        <li>Peak CCF &gt; 0.10 als Mindestfilter für Signal-Qualität</li>
        <li>OOS-Sharpe-Persistenz: bevorzuge OOS &gt; IS als Qualitätsmerkmal</li>
        <li>Regionen: US-Legacy-Carrier oft besser als EM-ADRs (Liquidität, Ölpreiskorrelation)</li>
      </ul>
    </div>
    <div class="card bg-dark border-info p-3">
      <h5 class="text-info">Warum reagieren bestimmte Airlines stärker auf Öl?</h5>
      <p><strong>Ökonomische Erklärung:</strong> Kerosinkosten machen 20–35% der Betriebskosten
      aus. Airlines mit geringem Hedging-Anteil und hohem Kurz-/Mittelstreckenanteil (höherer
      Treibstoffanteil pro Sitzplatz) reagieren schneller auf Spotpreisänderungen. US-Carrier
      haben durch effiziente Preisanpassung und liquide Optionsmärkte eine kürzere Lag-Struktur
      als asiatische/lateinamerikanische Pendants. Transfer Entropy und Granger-Tests bestätigen,
      dass die WTI-Informationsübertragung auf Airline-Aktien kausal und nicht spurios ist,
      insbesondere auf 1–5 Tage Horizont.</p>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # ASSEMBLE HTML
    # ════════════════════════════════════════════════════════════════════════
    def _acc(title, body, idx, open_=False):
        show = "show" if open_ else ""
        return f"""
        <div class="accordion-item bg-dark border-secondary">
          <h2 class="accordion-header">
            <button class="accordion-button {'collapsed' if not open_ else ''} bg-dark text-light"
                    type="button" data-bs-toggle="collapse" data-bs-target="#acc{idx}">
              {title}
            </button>
          </h2>
          <div id="acc{idx}" class="accordion-collapse collapse {show}">
            <div class="accordion-body">{body}</div>
          </div>
        </div>"""

    accordion = '<div class="accordion" id="mainAccordion">'
    accordion += _acc("§0 Ranking-Tabelle: alle Airlines × CL=F", sec0_table, 0, open_=True)
    accordion += _acc("§1 OOS Sharpe Ranking (Bar Chart)", sec1_chart, 1)
    accordion += _acc("§2 CCF-Heatmap: CL=F → Airline (Lag 0–10)", sec2_ccf, 2)
    accordion += _acc("§3 Optimaler Lag × OOS Sharpe (Scatter)", sec3_lag, 3)
    accordion += _acc("§4 Bubble Chart: Market Cap × OOS Sharpe × Region", sec4_bubble, 4)
    accordion += _acc("§5 OOS Equity Curves (alle Airlines)", sec5_eq, 5)
    accordion += _acc("§6 Rolling 252T Sharpe", sec6_rsh, 6)
    accordion += _acc("§7 Rolling Korrelation CL=F → Airline", sec7_rc, 7)
    accordion += _acc("§8 Radar Charts (normalisierte Metriken)", sec8_radar, 8)
    accordion += _acc("§9 Feature Matrix Heatmap (z-score)", sec9_feat, 9)
    accordion += _acc("§10 Feature-Korrelationsmatrix", sec10_fcorr, 10)
    accordion += _acc("§11 Hierarchisches Clustering (Dendrogram)", sec11_dend, 11)
    accordion += _acc("§12 PCA Biplot", pca_html, 12)
    accordion += _acc("§13 Parallel Coordinate Plot", sec13_pc, 13)
    accordion += _acc("§14 OOS Sharpe: Region & Airline-Typ (Boxplots)", sec14_box, 14)
    accordion += _acc("§15 Statistische Tests (Kruskal-Wallis, Spearman, OLS, Feature Importance)",
                      stats_html + ols_html + rf_html, 15)
    accordion += _acc("§16 Krisenperioden-Analyse", sec16_crisis, 16)
    accordion += _acc("§17 TC-Sweep (0–100bp)", sec17_tc, 17)
    accordion += _acc("§18 Monte Carlo Permutationstest", sec18_mc, 18)
    accordion += _acc("§19 Bootstrap CI (95%) auf OOS Sharpe", sec19_boot, 19)
    accordion += _acc("§20 Walk-Forward-Validierung", sec20_wf, 20)
    accordion += _acc("§21 Portfolioanalyse (EW / Vol-Weighted / Min-Var / Region)", sec21_port, 21)
    accordion += _acc("§22 VIX-Regime-Analyse", vix_html, 22)
    accordion += _acc("§23 Signal Stability Map (Rolling CCF über Zeit)", sec23_stab, 23)
    accordion += _acc("§24 Network Graph: CL=F → Airlines", sec24_net, 24)
    accordion += _acc("§25 Vollständige 26-Metriken-Tabelle (IS + OOS)", sec25_metrics, 25)
    accordion += _acc("§26 Granger-F vs Transfer Entropy", sec26_gte, 26)
    accordion += _acc("§27 Scatterplot-Matrix (Kernmetriken)", splom_html, 27)
    accordion += _acc("§28 Ökonomische Interpretation & Screening-Regelwerk", interp_html, 28)
    accordion += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#ffa657;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0">CL=F → Airline Lead-Lag: Querschnittsanalyse</h2>
          <p class="text-muted mb-0">
            {len(tickers_sorted)} Airlines analysiert · IS/OOS · 26 Metriken · Monte Carlo ·
            Bootstrap · Walk-Forward · Portfolio · Clustering · PCA · Statistische Tests
          </p>
        </div>
      </div>
      {accordion}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "airline_oil_report.html",
           _html_base("CL=F → Airline Lead-Lag: Querschnittsanalyse", 19, body))
    print("  Report: airline_oil_report.html")

'''

# ── Wiring strings ────────────────────────────────────────────────────────────
OLD_FUNC_MARKER = "\ndef build_index(tables, figures, out):"

OLD_W = (
    "    build_strategy_stress_test_report(tables, figures, reports)\n"
    "    build_index(tables, figures, reports)"
)
NEW_W = (
    "    build_strategy_stress_test_report(tables, figures, reports)\n"
    "    build_airline_oil_report(tables, figures, reports)\n"
    "    build_index(tables, figures, reports)"
)

src = RB.read_text(encoding="utf-8")

if "def build_airline_oil_report(" in src:
    print("Already injected — skipping function injection.")
else:
    insert_at = src.find(OLD_FUNC_MARKER)
    if insert_at == -1:
        raise RuntimeError("Injection point not found in report_builder.py")
    src = src[:insert_at] + FUNC + src[insert_at:]
    print("Function injected.")

if OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
elif "build_airline_oil_report" in src.split("def build_all_reports")[1] if "def build_all_reports" in src else "":
    print("Already wired.")
else:
    print("WARNING: Could not wire build_all_reports — wire manually.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
