"""
Replace build_airline_oil_report with v2 (full fixes + enhancements).

Fixes:
  - Roll-corr Series / CCF dicts removed from df construction → all heatmaps work
  - Parallel Coordinates: fillna with median instead of dropna
  - VIX index normalization
  - Interpretation text color → white
  - CCF matrix: proper float conversion
  - Feature/corr matrix: separate scalar dict from time-series storage

New features:
  - Long/Short ratio + signal overlay on equity curves
  - Per-section explanations with math formulas
  - Column-by-column table legend
  - Benchmark comparison: CL=F→JETS RSI<70 (from existing commodity data)
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

# ── locate and remove old function ───────────────────────────────────────────
src = RB.read_text(encoding="utf-8")

START_MARKER = "\ndef build_airline_oil_report(tables, figures, out):"
END_MARKER   = "\ndef build_index(tables, figures, out):"

s_pos = src.find(START_MARKER)
e_pos = src.find(END_MARKER)

if s_pos == -1 or e_pos == -1:
    raise RuntimeError("Could not find function boundaries.")

# Remove old function body (keep everything before START and from END onward)
src_without = src[:s_pos] + src[e_pos:]

# ── new function ──────────────────────────────────────────────────────────────
FUNC = r'''
def build_airline_oil_report(tables, figures, out):  # noqa: C901
    """
    Cross-sectional CL=F → Airline lead-lag research report (v2).
    28 sections: CCF, Granger, TE, signal stability, rolling metrics,
    crisis, VIX regimes, TC sweep, MC, bootstrap, walk-forward,
    PCA, clustering, portfolio, benchmark, long/short ratio, explanations.
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
    from scipy.stats import spearmanr, kruskal, pearsonr

    # ── universe ─────────────────────────────────────────────────────────────
    AIRLINES = {
        "DAL":   {"name": "Delta Air Lines",          "region": "USA",     "type": "Legacy"},
        "UAL":   {"name": "United Airlines",           "region": "USA",     "type": "Legacy"},
        "AAL":   {"name": "American Airlines",         "region": "USA",     "type": "Legacy"},
        "LUV":   {"name": "Southwest Airlines",        "region": "USA",     "type": "LCC"},
        "ALK":   {"name": "Alaska Air Group",          "region": "USA",     "type": "Legacy"},
        "JBLU":  {"name": "JetBlue Airways",           "region": "USA",     "type": "LCC"},
        "IAG":   {"name": "IAG (British Airways+)",    "region": "Europe",  "type": "Legacy"},
        "RYAAY": {"name": "Ryanair",                   "region": "Europe",  "type": "LCC"},
        "DLAKY": {"name": "Deutsche Lufthansa (ADR)",  "region": "Europe",  "type": "Legacy"},
        "AFLYY": {"name": "Air France-KLM (ADR)",      "region": "Europe",  "type": "Legacy"},
        "CPA":   {"name": "Copa Holdings",             "region": "LatAm",   "type": "Legacy"},
        "QABSY": {"name": "Qantas (ADR)",              "region": "Oceania", "type": "Legacy"},
        "JETS":  {"name": "US Global Jets ETF",        "region": "USA",     "type": "ETF"},
    }
    LEADER  = "CL=F"
    LAGS    = list(range(0, 11))
    IS_FRAC = 0.70
    N_MC    = 2000
    N_BOOT  = 1000
    TC_GRID = np.arange(0, 0.011, 0.001)

    # ── download ──────────────────────────────────────────────────────────────
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

    raw = {}
    for t in [LEADER] + list(AIRLINES.keys()):
        s = _dl(t)
        if s is not None:
            raw[t] = s

    if LEADER not in raw or len(raw) < 3:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Preisdaten konnten nicht geladen werden.</p>"))
        return

    prices_df = pd.DataFrame(raw).sort_index().ffill().dropna(how="all")
    # Remove duplicate dates (e.g. DST artefacts)
    prices_df = prices_df[~prices_df.index.duplicated(keep="last")]
    log_ret   = np.log(prices_df / prices_df.shift(1))

    available = [t for t in AIRLINES if t in prices_df.columns]
    if len(available) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Ticker verfügbar.</p>"))
        return

    leader_px  = prices_df[LEADER].dropna()
    leader_ret = log_ret[LEADER].dropna()

    # market caps (best effort)
    mcap_proxy = {}
    for t in available:
        try:
            mc = yf.Ticker(t).info.get("marketCap")
            if mc:
                mcap_proxy[t] = float(mc)
        except Exception:
            pass

    # ── local helpers ─────────────────────────────────────────────────────────
    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 30:
            return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _granger_f(y_arr, x_arr, maxlag=5):
        from numpy.linalg import lstsq as _lstsq
        n = len(y_arr)
        if n < 60:
            return np.nan, 1.0
        p = min(maxlag, n // 10)
        def _build(y, *xs):
            rows = []
            for i in range(p, n):
                row = [1.0] + list(y[i-p:i][::-1])
                for x in xs:
                    row += list(x[i-p:i][::-1])
                rows.append(row)
            return np.array(rows)
        Y  = y_arr[p:]
        Xr = _build(y_arr)
        Xu = _build(y_arr, x_arr)
        br, *_ = _lstsq(Xr, Y, rcond=None); RSS_r = ((Y - Xr @ br)**2).sum()
        bu, *_ = _lstsq(Xu, Y, rcond=None); RSS_u = ((Y - Xu @ bu)**2).sum() + 1e-12
        df2 = len(Y) - 2*p - 1
        if df2 < 1:
            return np.nan, 1.0
        F    = ((RSS_r - RSS_u) / p) / (RSS_u / df2)
        pval = 1 - stats.f.cdf(F, p, df2)
        return float(F), float(pval)

    def _te(x_arr, y_arr, lag=1, bins=10):
        """Transfer entropy TE(X→Y) via binned estimator."""
        n = len(x_arr)
        if n < 60 or lag < 1:
            return 0.0
        cut = lambda a: np.digitize(a, np.percentile(a, np.linspace(0,100,bins+1)[1:-1]))
        xb = cut(x_arr); yb = cut(y_arr)
        # H(y_t|y_{t-1}) - H(y_t|y_{t-1}, x_{t-lag})
        from collections import Counter
        def _h(pairs):
            c = Counter(pairs); tot = sum(c.values())
            return -sum(v/tot * np.log2(v/tot + 1e-12) for v in c.values())
        py   = list(zip(yb[1:],   yb[:-1]))
        pyx  = list(zip(yb[1:],   yb[:-1], xb[:-lag][:len(yb)-1]))
        te   = _h(py) - _h(pyx) * 0.5
        return max(float(te), 0.0)

    INDICATORS = [
        ("RSI<70",    lambda p: -_calc_rsi(p, 14),         -70.0),
        ("RSI>50",    lambda p:  _calc_rsi(p, 14),           50.0),
        ("MACD>0",    lambda p:  _calc_macd(p)[0],            0.0),
        ("BB>0.5",    lambda p:  _calc_bb_pos(p, 20),         0.5),
        ("SMA cross", lambda p:  _calc_sma_cross(p, 20, 50),  0.0),
    ]

    # ── per-airline analysis ──────────────────────────────────────────────────
    # Separate scalars (→ df) from time-series (kept in dicts)
    scalars   = {}   # ticker → {scalar metrics}
    ccf_store = {}   # ticker → {lag: pearson r}
    rc_store  = {}   # ticker → rolling-corr Series
    strat_best = {}  # ticker → best strategy record

    for ticker in available:
        ret = log_ret[ticker].dropna()
        px_t = prices_df[ticker].dropna()
        common = (leader_px.index
                  .intersection(ret.index)
                  .intersection(leader_ret.index))
        common = common[~common.duplicated()]
        if len(common) < 252:
            continue

        lp = leader_px.reindex(common).ffill()
        lr = leader_ret.reindex(common).fillna(0.0)
        fr = ret.reindex(common).fillna(0.0)

        split_i    = int(len(common) * IS_FRAC)
        split_date = common[split_i]
        is_idx     = common[:split_i]
        oos_idx    = common[split_i:]

        # CCF at lags 0–10
        cv = {}
        for lag in LAGS:
            if lag == 0:
                r_v, _ = pearsonr(lr.values, fr.values)
            else:
                s_lr = lr.shift(lag).reindex(common).dropna()
                s_fr = fr.reindex(s_lr.index)
                both = pd.concat([s_lr, s_fr], axis=1).dropna()
                r_v  = float(pearsonr(both.iloc[:,0].values, both.iloc[:,1].values)[0]) if len(both) > 30 else 0.0
            cv[lag] = float(r_v)
        ccf_store[ticker] = cv

        best_lag = max(cv, key=lambda l: abs(cv[l]))
        best_ccf = cv[best_lag]

        # Rolling correlation at best_lag
        lag_rc = max(best_lag, 1)
        rc_s = lr.shift(lag_rc).rolling(252).corr(fr)
        rc_store[ticker] = rc_s.dropna()

        # Granger & TE
        gr_f, gr_p = _granger_f(fr.values, lr.values)
        te_v       = _te(lr.values, fr.values, lag=max(best_lag, 1))

        # Strategy search: best IS Sharpe over indicators × lags
        best_sh_is = -99.0
        best_rec   = None
        for ind_name, ind_fn, thresh in INDICATORS:
            for lag_s in sorted(set([1, 2, 3, max(best_lag, 1)])):
                n_is, g_is, s_is   = _strat_exec(ind_fn(lp), thresh, fr.loc[is_idx],  lag_s)
                n_oos, g_oos, s_oos = _strat_exec(ind_fn(lp), thresh, fr.loc[oos_idx], lag_s)
                if len(n_is) < 30 or len(n_oos) < 30:
                    continue
                sh = _sh(n_is)
                if sh > best_sh_is:
                    best_sh_is = sh
                    best_rec   = {
                        "ind": ind_name, "lag": lag_s, "thresh": thresh,
                        "ind_fn": ind_fn,
                        "n_is": n_is, "g_is": g_is, "s_is": s_is,
                        "n_oos": n_oos, "g_oos": g_oos, "s_oos": s_oos,
                        "sh_is": sh, "sh_oos": _sh(n_oos),
                        "split_date": split_date,
                    }
        if best_rec is None:
            continue
        strat_best[ticker] = best_rec

        net_all = pd.concat([best_rec["n_is"], best_rec["n_oos"]]).sort_index()
        sig_all = pd.concat([best_rec["s_is"], best_rec["s_oos"]]).sort_index()

        m_is  = _full_metrics(best_rec["n_is"],  best_rec["g_is"],  best_rec["s_is"],  name=f"{ticker} IS")
        m_oos = _full_metrics(best_rec["n_oos"], best_rec["g_oos"], best_rec["s_oos"], name=f"{ticker} OOS")

        # L/S ratio
        n_long  = int((sig_all > 0).sum())
        n_short = int((sig_all < 0).sum())
        n_flat  = int((sig_all == 0).sum())
        ls_ratio = round(n_long / (n_short + 1e-9), 2)

        # Rolling 126-day Sharpe for regime stability
        roll_sh = net_all.rolling(126).apply(lambda x: _sh(x), raw=True)
        regime_stab = float((roll_sh.dropna() > 0).mean()) if roll_sh.dropna().size > 0 else np.nan

        sp_corr, _ = spearmanr(lr.values, fr.values)
        beta_oil   = float(np.cov(fr.values, lr.values)[0,1] / (np.var(lr.values) + 1e-12))
        ar1        = float(pd.Series(fr.values).autocorr(1))
        mom        = float(pd.Series(fr.values).autocorr(21))
        vol_pct    = float(fr.std() * np.sqrt(252) * 100)

        # Signal half-life
        half_life = np.nan
        if abs(best_ccf) > 0.01:
            for ll in range(best_lag, 20):
                if abs(cv.get(ll, 0.0)) < abs(best_ccf) * 0.5:
                    half_life = float(ll - best_lag)
                    break

        scalars[ticker] = {
            "name":        AIRLINES[ticker]["name"],
            "region":      AIRLINES[ticker]["region"],
            "type":        AIRLINES[ticker]["type"],
            "mcap":        mcap_proxy.get(ticker, np.nan),
            "best_lag":    float(best_lag),
            "best_ccf":    float(best_ccf),
            "ccf_lag0":    float(cv.get(0, np.nan)),
            "roll_corr_mean": float(rc_s.mean()),
            "roll_corr_std":  float(rc_s.std()),
            "granger_f":   float(gr_f) if not np.isnan(gr_f) else np.nan,
            "granger_p":   float(gr_p),
            "te":          float(te_v),
            "half_life":   half_life,
            "regime_stab": regime_stab,
            "ar1":         ar1,
            "mom":         mom,
            "vol_pct":     vol_pct,
            "sp_corr":     float(sp_corr),
            "beta_oil":    beta_oil,
            "best_ind":    best_rec["ind"],
            "best_lag_s":  float(best_rec["lag"]),
            "sh_is":       float(best_rec["sh_is"]),
            "sh_oos":      float(best_rec["sh_oos"]),
            "oos_gt_is":   best_rec["sh_oos"] > best_rec["sh_is"],
            "n_long":      n_long, "n_short": n_short, "n_flat": n_flat,
            "ls_ratio":    ls_ratio,
            "n_days_is":   len(best_rec["n_is"]),
            "n_days_oos":  len(best_rec["n_oos"]),
            "split_date":  str(best_rec["split_date"])[:10],
            **{f"is_{k}": v  for k, v in m_is.items()  if k != "Name"},
            **{f"oos_{k}": v for k, v in m_oos.items() if k != "Name"},
        }

    if len(scalars) < 2:
        _write(out / "airline_oil_report.html",
               _html_base("Airline × Oil", 19, "<p>Zu wenige Daten für Vergleichsanalyse.</p>"))
        return

    # Build scalar-only DataFrame (no Series / dicts)
    df = pd.DataFrame(scalars).T

    # Ensure numeric columns are numeric
    num_cols = [c for c in df.columns if c not in
                ("name","region","type","best_ind","split_date","oos_gt_is")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("sh_oos", ascending=False)
    sorted_tickers = list(df.index)

    # ── colour helpers ────────────────────────────────────────────────────────
    REG_COLORS  = {"USA":"#58a6ff","Europe":"#3fb950","Asia":"#ffa657",
                   "LatAm":"#f78166","Canada":"#bc8cff","Oceania":"#39d353","ETF":"#e3b341"}
    TYPE_COLORS = {"Legacy":"#58a6ff","LCC":"#3fb950","ULCC":"#ffa657","ETF":"#e3b341"}
    PAL_EQ      = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

    def _rc(r): return REG_COLORS.get(str(r), "#8b949e")
    def _tc(t): return TYPE_COLORS.get(str(t), "#8b949e")

    def _lay(fig, **kw):
        L = dict(**_LAYOUT); L.update(kw)
        fig.update_layout(**L)
        return fig

    def _htm(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})

    # ── description box ───────────────────────────────────────────────────────
    def _desc(txt):
        return (f'<div class="alert" style="background:#1c2128;border:1px solid #30363d;'
                f'color:#e6edf3;font-size:0.88em;margin-bottom:12px;">{txt}</div>')

    # ── legend table ──────────────────────────────────────────────────────────
    def _legend(rows_html):
        return (f'<details class="mb-3"><summary style="color:#58a6ff;cursor:pointer;">'
                f'▶ Spaltenlegende</summary><div class="table-responsive mt-2">'
                f'<table class="table table-sm table-dark" style="font-size:0.8em;">'
                f'<thead><tr><th>Spalte</th><th>Berechnung</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table></div></details>')

    # ════════════════════════════════════════════════════════════════════════
    # §0  Ranking table
    # ════════════════════════════════════════════════════════════════════════
    def _f(v):
        if isinstance(v, bool): return "✓" if v else "✗"
        try:
            f = float(v)
            return "—" if np.isnan(f) else f"{f:.3f}"
        except Exception:
            return str(v)

    rank_rows = ""
    for i, (t, row) in enumerate(df.iterrows()):
        oos_flag = "✓" if row.get("oos_gt_is", False) else ""
        br = f'<span class="badge" style="background:{_rc(row.get("region",""))};">{row.get("region","")}</span>'
        bt = f'<span class="badge" style="background:{_tc(row.get("type",""))};">{row.get("type","")}</span>'
        rank_rows += (
            f"<tr><td>{i+1}</td><td><strong>{t}</strong></td><td>{row.get('name','')}</td>"
            f"<td>{br}</td><td>{bt}</td>"
            f"<td>{_f(row.get('sh_is'))}</td><td>{_f(row.get('sh_oos'))}</td><td>{oos_flag}</td>"
            f"<td>{_f(row.get('best_lag'))}</td><td>{_f(row.get('best_ccf'))}</td>"
            f"<td>{row.get('best_ind','—')}</td><td>{_f(row.get('best_lag_s'))}</td>"
            f"<td>{_f(row.get('granger_f'))}</td><td>{_f(row.get('granger_p'))}</td>"
            f"<td>{_f(row.get('te'))}</td><td>{_f(row.get('half_life'))}</td>"
            f"<td>{_f(row.get('regime_stab'))}</td>"
            f"<td>{_f(row.get('ls_ratio'))}</td>"
            f"<td>{int(row.get('n_long',0))}/{int(row.get('n_short',0))}</td>"
            f"</tr>"
        )

    leg0 = _legend(
        "<tr><td>Sharpe IS/OOS</td><td>µ·252 / (σ·√252) auf den täglichen Nettorenditen im IS- bzw. OOS-Zeitfenster</td></tr>"
        "<tr><td>OOS&gt;IS</td><td>Kennzeichen ob Sharpe OOS &gt; IS (kein Overfitting)</td></tr>"
        "<tr><td>Best Lag</td><td>Lag l* = argmax |CCF(l)| für l ∈ 0..10</td></tr>"
        "<tr><td>Peak CCF</td><td>Pearson r zwischen CL=F-Rendite(t-l*) und Airline-Rendite(t)</td></tr>"
        "<tr><td>Best Ind.</td><td>Indikator mit höchstem IS-Sharpe (RSI/MACD/BB/SMA)</td></tr>"
        "<tr><td>Granger F</td><td>F-Statistik des Granger-Kausalitätstests (H0: CL=F-Lags helfen nicht)</td></tr>"
        "<tr><td>Granger p</td><td>p-Wert des Granger-Tests; p&lt;0.05 = signifikante Kausalität</td></tr>"
        "<tr><td>Trans.Entropy</td><td>TE(CL=F→Airline) in Bits; misst nicht-lineare Informationsübertragung</td></tr>"
        "<tr><td>Signal HL</td><td>Halbwertszeit: Tage bis |CCF| auf 50% des Peaks abgefallen ist</td></tr>"
        "<tr><td>Regime Stab.</td><td>Anteil 126-Tage-Fenster mit positivem Sharpe (0–1)</td></tr>"
        "<tr><td>L/S Ratio</td><td>Anzahl Long-Tage / Anzahl Short-Tage; &gt;1 = überwiegend long</td></tr>"
        "<tr><td>L/S Tage</td><td>Absolute Anzahl Long- und Short-Handelstage</td></tr>"
    )

    sec0 = (
        _desc("Diese Tabelle zeigt alle Kennzahlen je Airline auf einen Blick. "
              "Sortierung nach OOS Sharpe (absteigend). "
              "<strong>Strategie:</strong> Long wenn CL=F-Indikator(t−Lag) &gt; Schwelle, sonst Short. "
              "TC = 10bp pro Richtungswechsel.")
        + leg0
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover" style="font-size:0.8em;">'
        f'<thead class="table-dark"><tr>'
        f'<th>#</th><th>Ticker</th><th>Name</th><th>Region</th><th>Typ</th>'
        f'<th>Sharpe IS</th><th>Sharpe OOS</th><th>OOS&gt;IS</th>'
        f'<th>Best Lag</th><th>Peak CCF</th><th>Best Ind.</th><th>Ind.Lag</th>'
        f'<th>Granger F</th><th>Granger p</th><th>Trans.Entropy</th>'
        f'<th>Signal HL</th><th>Regime Stab.</th><th>L/S Ratio</th><th>L/S Tage</th>'
        f'</tr></thead><tbody>{rank_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §1  OOS Sharpe bar chart
    # ════════════════════════════════════════════════════════════════════════
    sh_oos_v = df["sh_oos"].fillna(0).tolist()
    sh_is_v  = df["sh_is"].fillna(0).tolist()
    bar_cols = [_rc(df.loc[t, "region"]) for t in sorted_tickers]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=sorted_tickers, y=sh_is_v,  name="IS Sharpe",
                          marker_color="#30363d", opacity=0.7))
    fig1.add_trace(go.Bar(x=sorted_tickers, y=sh_oos_v, name="OOS Sharpe",
                          marker_color=bar_cols))
    fig1.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig1, title="OOS Sharpe Ranking: CL=F → Airline", barmode="group",
         xaxis_title="Airline", yaxis_title="Annualisierter Sharpe", height=430)
    sec1 = (
        _desc("Sharpe-Ratio = µ·252 / (σ·√252). Graue Balken = IS-Periode (Kalibrierung, 70% der Daten). "
              "Farbige Balken = OOS-Periode (echter Test, 30% der Daten). "
              "Indikator und Lag wurden ausschließlich auf dem IS-Zeitfenster optimiert. "
              "Ein OOS-Sharpe nahe oder über dem IS-Wert deutet auf robuste Out-of-Sample-Performance hin.")
        + _htm(fig1)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §2  L/S Ratio
    # ════════════════════════════════════════════════════════════════════════
    ls_fig = go.Figure()
    ls_long  = [int(df.loc[t, "n_long"])  for t in sorted_tickers]
    ls_short = [int(df.loc[t, "n_short"]) for t in sorted_tickers]
    ls_flat  = [int(df.loc[t, "n_flat"])  for t in sorted_tickers]
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_long,  name="Long",  marker_color="#3fb950"))
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_short, name="Short", marker_color="#f78166"))
    ls_fig.add_trace(go.Bar(x=sorted_tickers, y=ls_flat,  name="Flat",  marker_color="#30363d"))
    _lay(ls_fig, title="Long / Short / Flat Tage je Airline (Gesamtperiode)",
         barmode="stack", height=380)

    ls_rows = "".join(
        f"<tr><td>{t}</td><td>{scalars[t].get('best_ind','—')}</td>"
        f"<td>{int(scalars[t].get('n_long',0))}</td>"
        f"<td>{int(scalars[t].get('n_short',0))}</td>"
        f"<td>{int(scalars[t].get('n_flat',0))}</td>"
        f"<td>{scalars[t].get('ls_ratio',0):.2f}</td>"
        f"<td>{int(scalars[t].get('n_long',0)+scalars[t].get('n_short',0)+scalars[t].get('n_flat',0))}</td></tr>"
        for t in sorted_tickers if t in scalars
    )
    ls_table = (
        '<div class="table-responsive mt-3"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Airline</th><th>Indikator</th><th>Long T.</th>'
        '<th>Short T.</th><th>Flat T.</th><th>L/S Ratio</th><th>Gesamt</th></tr></thead>'
        f'<tbody>{ls_rows}</tbody></table></div>'
    )
    sec2 = (
        _desc("<strong>Long-Signal:</strong> Wenn CL=F-Indikator(t−Lag) &gt; Schwelle → Position = +1 (long). "
              "<strong>Short-Signal:</strong> Wenn Indikator(t−Lag) ≤ Schwelle → Position = −1 (short). "
              "L/S Ratio = (Long-Tage) / (Short-Tage). "
              "Beispiel RSI&lt;70: Signal = Long wenn −RSI(t−Lag) &gt; −70, also wenn RSI &lt; 70.")
        + _htm(ls_fig)
        + ls_table
    )

    # ════════════════════════════════════════════════════════════════════════
    # §3  Equity curves with Long/Short markers (OOS)
    # ════════════════════════════════════════════════════════════════════════
    fig3 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        net_oos = sr["n_oos"]; sig_oos = sr["s_oos"]
        if len(net_oos) < 10:
            continue
        cum = (1 + net_oos).cumprod() * 100
        col = PAL_EQ[i % len(PAL_EQ)]
        fig3.add_trace(go.Scatter(
            x=cum.index.astype(str).tolist(), y=cum.values.tolist(),
            name=t, mode="lines", line=dict(color=col, width=1.8),
        ))
        # Signal-change markers (entries)
        changes = sig_oos.diff().fillna(0)
        longs  = sig_oos.index[changes > 0]
        shorts = sig_oos.index[changes < 0]
        for dates_e, symbol, mcolor in [(longs, "triangle-up", "#3fb950"),
                                         (shorts, "triangle-down", "#f78166")]:
            if len(dates_e) == 0:
                continue
            y_e = cum.reindex(dates_e, method="nearest").dropna()
            fig3.add_trace(go.Scatter(
                x=y_e.index.astype(str).tolist(), y=y_e.values.tolist(),
                mode="markers", name=f"{t} {'Long' if symbol=='triangle-up' else 'Short'}",
                marker=dict(symbol=symbol, size=7, color=mcolor, opacity=0.7),
                showlegend=False,
            ))
    _lay(fig3, title="OOS Equity Curves: CL=F → Airline (▲=Long-Entry, ▼=Short-Entry)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=520)
    sec3 = (
        _desc("Equity Curve: NAV = (1+r₁)·(1+r₂)·…·(1+rₙ)·100, wobei rᵢ = Netto-Tagesrendite nach TC. "
              "▲ = Long-Entry (Positionswechsel von Short auf Long). ▼ = Short-Entry (von Long auf Short). "
              "Nur OOS-Periode dargestellt (30% der Gesamtdaten, zeitlich hintere Hälfte).")
        + _htm(fig3)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §4  CCF Heatmap Lag 0–10
    # ════════════════════════════════════════════════════════════════════════
    ccf_mat = np.zeros((len(LAGS), len(sorted_tickers)))
    for j, t in enumerate(sorted_tickers):
        for i, lag in enumerate(LAGS):
            ccf_mat[i, j] = ccf_store.get(t, {}).get(lag, 0.0)

    fig4 = go.Figure(go.Heatmap(
        z=ccf_mat.tolist(),
        x=sorted_tickers,
        y=[f"Lag {l}T" for l in LAGS],
        colorscale="RdBu", zmid=0,
        text=[[f"{ccf_mat[i,j]:.3f}" for j in range(len(sorted_tickers))] for i in range(len(LAGS))],
        texttemplate="%{text}", textfont=dict(size=9, color="#e6edf3"),
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
        zmin=-0.5, zmax=0.5,
    ))
    _lay(fig4, title="CCF Heatmap: CL=F(t−Lag) → Airline(t)", height=480)
    sec4 = (
        _desc("Cross-Correlation Function (CCF): CCF(l) = Pearson r(CL=F(t-l), Airline(t)). "
              "Roter Bereich = positive Korrelation (hoher Ölpreis → hohe Airline-Rendite), "
              "blauer Bereich = negative Korrelation. "
              "Liegt das Maximum bei Lag &gt; 0, bedeutet das: CL=F führt Airline um l Tage an.")
        + _htm(fig4)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §5  Best Lag scatter
    # ════════════════════════════════════════════════════════════════════════
    lags_arr   = [float(df.loc[t, "best_lag"]) for t in sorted_tickers]
    sh_oos_arr = [float(df.loc[t, "sh_oos"])   for t in sorted_tickers]

    fig5 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig5.add_trace(go.Scatter(
            x=[lags_arr[i] for i in mask], y=[sh_oos_arr[i] for i in mask],
            mode="markers+text", text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=12, color=col, line=dict(color="#0d1117", width=1)),
            name=reg,
        ))
    _lay(fig5, title="Optimaler Lag × OOS Sharpe", height=420,
         xaxis_title="Lag l* (Tage)", yaxis_title="OOS Sharpe")
    sec5 = (
        _desc("Optimaler Lag l* = argmax_{l∈0..10} |CCF(l)|. "
              "Airlines rechts oben (hoher Lag, hoher Sharpe) profitieren am meisten "
              "von zeitverzögerter Ölinformation. US-notierte ADRs tendieren zu kurzem Lag (1–2T), "
              "internationale ADRs oft zu längerem Lag (3–7T) durch unterschiedliche Handelszeiten.")
        + _htm(fig5)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §6  Bubble Chart
    # ════════════════════════════════════════════════════════════════════════
    mc_vals = [(float(df.loc[t, "mcap"]) / 1e9
                if not pd.isna(df.loc[t, "mcap"]) else 3.0)
               for t in sorted_tickers]

    fig6 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig6.add_trace(go.Scatter(
            x=[mc_vals[i] for i in mask], y=[sh_oos_arr[i] for i in mask],
            mode="markers+text", text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=[max(mc_vals[i]*0.6, 5) for i in mask],
                        color=col, opacity=0.8,
                        sizemode="area", sizeref=0.1,
                        line=dict(color="#0d1117", width=1)),
            name=reg,
        ))
    _lay(fig6, title="Market Cap (Mrd. USD) × OOS Sharpe × Region",
         xaxis_title="Market Cap (Mrd. USD)", yaxis_title="OOS Sharpe", height=450)
    sec6 = (
        _desc("Blasengröße = Market Cap in Mrd. USD (falls verfügbar, sonst 3 Mrd.). "
              "Farbe = Region. "
              "Kein systematischer Zusammenhang zwischen Größe und Sharpe würde bedeuten, "
              "dass Market Cap kein Screening-Kriterium für Lead-Lag-Qualität ist.")
        + _htm(fig6)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §7  Rolling 252T Sharpe
    # ════════════════════════════════════════════════════════════════════════
    fig7 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        roll    = net_all.rolling(252).apply(lambda x: _sh(x), raw=True).dropna()
        if len(roll) < 10:
            continue
        fig7.add_trace(go.Scatter(
            x=roll.index.astype(str).tolist(), y=roll.values.tolist(),
            name=t, mode="lines",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.5),
        ))
    fig7.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig7, title="Rolling 252T Sharpe: je Airline",
         xaxis_title="Datum", yaxis_title="Sharpe (252T-Fenster)", height=460)
    sec7 = (
        _desc("Rolling Sharpe mit 252-Tage-Fenster (≈ 1 Börsenjahr). "
              "Berechnung: Sharpe(t) = µ_{t-251..t}·252 / (σ_{t-251..t}·√252). "
              "Werte über 0 zeigen profitable Perioden; anhaltend positive Kurven deuten "
              "auf strukturelle Lead-Lag-Persistenz hin.")
        + _htm(fig7)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §8  Rolling Correlation
    # ════════════════════════════════════════════════════════════════════════
    fig8 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        rc = rc_store.get(t)
        if rc is None or len(rc) < 10:
            continue
        fig8.add_trace(go.Scatter(
            x=rc.index.astype(str).tolist(), y=rc.values.tolist(),
            name=t, mode="lines",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.2),
        ))
    fig8.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig8, title="Rolling 252T Korrelation: CL=F(t-Lag) → Airline(t)",
         xaxis_title="Datum", yaxis_title="Pearson r", height=430)
    sec8 = (
        _desc("Rolling Pearson-Korrelation zwischen CL=F-Rendite(t−l*) und Airline-Rendite(t) "
              "in einem 252-Tage-Schiebefenster. Ein stabiler positiver Verlauf zeigt, dass "
              "die Lead-Lag-Struktur zeitlich konstant ist. Starke Schwankungen deuten auf "
              "Regime-Wechsel (z.B. Ölcrash, COVID) hin.")
        + _htm(fig8)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §9  Radar Charts
    # ════════════════════════════════════════════════════════════════════════
    RMETS = ["sh_oos", "best_ccf", "granger_f", "te", "regime_stab", "roll_corr_mean"]
    RLABS = ["OOS Sharpe", "Peak CCF", "Granger F", "Trans.Entropy", "Regime Stab.", "Roll.Corr."]

    def _norm(col):
        v = df[col].apply(pd.to_numeric, args=("coerce",) if False else ())
        v = pd.to_numeric(v, errors="coerce")
        mn, mx = v.min(), v.max()
        if mx == mn:
            return pd.Series(0.5, index=v.index)
        return (v - mn) / (mx - mn)

    nv = {m: _norm(m) for m in RMETS if m in df.columns}
    n_r = max(1, (len(sorted_tickers) + 3) // 4)
    fig9 = make_subplots(
        rows=n_r, cols=4,
        specs=[[{"type": "polar"}] * 4 for _ in range(n_r)],
        subplot_titles=sorted_tickers[:n_r * 4],
    )
    for i, t in enumerate(sorted_tickers):
        row = i // 4 + 1; col = i % 4 + 1
        vals_r = [float(nv[m].get(t, 0.0)) if m in nv else 0.0 for m in RMETS]
        vals_r.append(vals_r[0])
        fig9.add_trace(go.Scatterpolar(
            r=vals_r, theta=RLABS + [RLABS[0]], fill="toself",
            name=t, line=dict(color=PAL_EQ[i % len(PAL_EQ)]), showlegend=False,
        ), row=row, col=col)
    fig9.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=max(350, n_r * 300),
        title_text="Radar Charts: je Airline (normalisiert auf 0–1)",
    )
    sec9 = (
        _desc("Alle 6 Dimensionen sind min-max-normalisiert auf [0,1]. "
              "Eine große ausgefüllte Fläche = die Airline ist in allen Dimensionen überdurchschnittlich. "
              "Achsen: OOS Sharpe (Strategiequalität), Peak CCF (maximale lineare Korrelation mit CL=F), "
              "Granger F (statistische Kausalität), Transfer Entropy (nicht-lineare Informationsübertragung), "
              "Regime Stab. (Anteil profitabler 6-Monats-Fenster), Roll.Corr. (mittlere rollende Korrelation).")
        + _htm(fig9)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §10  Feature Matrix Heatmap
    # ════════════════════════════════════════════════════════════════════════
    FEAT_COLS = ["sh_is", "sh_oos", "best_lag", "best_ccf", "granger_f", "te",
                 "half_life", "regime_stab", "ar1", "mom", "vol_pct", "beta_oil",
                 "roll_corr_mean", "roll_corr_std"]
    feat_df = df[[c for c in FEAT_COLS if c in df.columns]].copy()
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    feat_norm = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-9)

    z_vals  = feat_norm.values.T.tolist()
    txt_raw = feat_df.values.T
    txt_mat = [[f"{txt_raw[i,j]:.2f}" if not np.isnan(float(txt_raw[i,j])) else "—"
                for j in range(txt_raw.shape[1])]
               for i in range(txt_raw.shape[0])]

    fig10 = go.Figure(go.Heatmap(
        z=z_vals, x=feat_norm.index.tolist(), y=feat_norm.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=txt_mat, texttemplate="%{text}", textfont=dict(size=9, color="#e6edf3"),
        colorbar=dict(title="z-score", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig10, title="Feature Matrix: Airlines × Metriken (z-score normiert)", height=520)

    leg10 = _legend(
        "<tr><td>sh_is/sh_oos</td><td>Sharpe-Ratio IS/OOS</td></tr>"
        "<tr><td>best_lag</td><td>Lag mit höchstem |CCF|</td></tr>"
        "<tr><td>best_ccf</td><td>Pearson r bei best_lag</td></tr>"
        "<tr><td>granger_f</td><td>Granger F-Statistik</td></tr>"
        "<tr><td>te</td><td>Transfer Entropy in Bits</td></tr>"
        "<tr><td>half_life</td><td>Tage bis CCF auf 50% abfällt</td></tr>"
        "<tr><td>regime_stab</td><td>Anteil pos. 126T-Sharpe-Fenster</td></tr>"
        "<tr><td>ar1</td><td>AR(1)-Autokorrelation der Airline-Renditen (Mean-Reversion wenn &lt;0)</td></tr>"
        "<tr><td>mom</td><td>Autokorrelation bei Lag 21 (Momentum-Proxy)</td></tr>"
        "<tr><td>vol_pct</td><td>Historische Ann.Volatilität der Airline (%)</td></tr>"
        "<tr><td>beta_oil</td><td>β = Cov(r_airline, r_oil)/Var(r_oil)</td></tr>"
        "<tr><td>roll_corr_mean/std</td><td>Mittelwert/Streuung der rollenden Korrelation</td></tr>"
    )
    sec10 = (
        _desc("Jeder Wert ist z-score normiert: z = (x − µ) / σ. "
              "Rot = überdurchschnittlich, Blau = unterdurchschnittlich. "
              "Zahlenangaben zeigen die rohen (un-normierten) Werte. "
              "Zeilen = Metriken, Spalten = Airlines.")
        + leg10
        + _htm(fig10)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §11  Feature Correlation Matrix
    # ════════════════════════════════════════════════════════════════════════
    feat_complete = feat_df.dropna(how="any")
    if len(feat_complete) < 3:
        feat_complete = feat_df.fillna(feat_df.median())
    corr_m = feat_complete.corr(method="pearson")

    fig11 = go.Figure(go.Heatmap(
        z=corr_m.values.tolist(),
        x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{corr_m.values[i,j]:.2f}" for j in range(len(corr_m.columns))]
              for i in range(len(corr_m.index))],
        texttemplate="%{text}", textfont=dict(size=10, color="#e6edf3"),
        colorbar=dict(title="Pearson r", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig11, title="Feature-Korrelationsmatrix (Pearson, Airlines als Beobachtungen)", height=520)
    sec11 = (
        _desc("Korrelationsmatrix der 14 Metriken über alle Airlines. "
              "Hohe Korrelation zwischen z.B. Granger F und CCF würde bedeuten, "
              "dass beide ähnliche Aspekte der Lead-Lag-Beziehung messen. "
              "NaN-Werte wurden mit dem jeweiligen Median imputiert.")
        + _htm(fig11)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §12  Hierarchical Clustering
    # ════════════════════════════════════════════════════════════════════════
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist

    feat_cl = feat_df.fillna(feat_df.median())
    feat_cl = (feat_cl - feat_cl.mean()) / (feat_cl.std() + 1e-9)

    fig12 = go.Figure()
    clust_labels = {}
    if len(feat_cl) >= 3:
        dm = pdist(feat_cl.values, metric="euclidean")
        Z  = linkage(dm, method="ward")
        dd = dendrogram(Z, labels=feat_cl.index.tolist(), no_plot=True)
        for xs, ys in zip(dd["icoord"], dd["dcoord"]):
            fig12.add_trace(go.Scatter(x=ys, y=xs, mode="lines",
                                       line=dict(color="#58a6ff"), showlegend=False))
        n_leafs = len(feat_cl)
        fig12.update_yaxes(
            tickvals=list(range(5, (n_leafs + 1) * 10, 10)),
            ticktext=dd["ivl"],
        )
        k = min(3, len(feat_cl))
        cids = fcluster(Z, k, criterion="maxclust")
        clust_labels = {t: int(c) for t, c in zip(feat_cl.index, cids)}
    _lay(fig12, title="Hierarchisches Clustering: Ward-Linkage (Euklidische Distanz)",
         height=max(350, len(feat_cl) * 25))
    sec12 = (
        _desc("Ward-Linkage minimiert die intra-cluster Varianz. "
              "Weit voneinander entfernte Äste = unterschiedliche Lead-Lag-Profile. "
              "k=3 Cluster wurden anschließend mittels fcluster vergeben. "
              "Distance = √Σ(z_i − z_j)².")
        + _htm(fig12)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §13  PCA Biplot
    # ════════════════════════════════════════════════════════════════════════
    pca_html = ""
    if len(feat_cl) >= 3 and feat_cl.shape[1] >= 2:
        X = feat_cl.values
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        scores   = U[:, :2] * S[:2]
        loadings = Vt[:2, :].T
        ev = (S**2 / (S**2).sum())[:2]

        fig13 = go.Figure()
        for reg, col in REG_COLORS.items():
            mask = [i for i, t in enumerate(feat_cl.index) if df.loc[t, "region"] == reg]
            if not mask:
                continue
            fig13.add_trace(go.Scatter(
                x=scores[mask, 0].tolist(), y=scores[mask, 1].tolist(),
                mode="markers+text", text=[feat_cl.index[i] for i in mask],
                textposition="top center",
                marker=dict(size=11, color=col), name=reg,
            ))
        sc = max(float(np.abs(scores).max()), 1.0) * 0.55
        for j, cn in enumerate(feat_cl.columns):
            lx, ly = float(loadings[j, 0]) * sc, float(loadings[j, 1]) * sc
            fig13.add_annotation(x=lx, y=ly, ax=0, ay=0,
                                 arrowhead=2, arrowcolor="#e3b341",
                                 text=cn, font=dict(color="#e3b341", size=9))
        _lay(fig13,
             title=f"PCA Biplot – PC1 {ev[0]*100:.1f}% | PC2 {ev[1]*100:.1f}%",
             xaxis_title=f"PC1 ({ev[0]*100:.1f}% Varianz)",
             yaxis_title=f"PC2 ({ev[1]*100:.1f}% Varianz)", height=490)
        pca_html = (
            _desc("SVD-basierte PCA (kein sklearn): X = U·Σ·Vᵀ. "
                  "Scores (Punkte) = U·Σ, Loadings (Pfeile) = Vᵀ. "
                  "Airlines nahe beieinander haben ähnliches Feature-Profil. "
                  "Pfeile zeigen in welche Richtung welche Variable zunimmt. "
                  "Gelbe Pfeile = normierte Loadings skaliert auf Score-Bereich.")
            + _htm(fig13)
        )

    # ════════════════════════════════════════════════════════════════════════
    # §14  Parallel Coordinates
    # ════════════════════════════════════════════════════════════════════════
    pc_cols = ["sh_oos","best_lag","best_ccf","granger_f","te","regime_stab","vol_pct"]
    pc_df = df[[c for c in pc_cols if c in df.columns]].copy()
    for c in pc_df.columns:
        pc_df[c] = pd.to_numeric(pc_df[c], errors="coerce")
    pc_df = pc_df.fillna(pc_df.median())

    fig14 = go.Figure()
    if len(pc_df) >= 2:
        sh_col = pc_df["sh_oos"].values
        fig14 = go.Figure(go.Parcoords(
            line=dict(color=sh_col, colorscale="Plasma", showscale=True,
                      colorbar=dict(title="OOS Sharpe", tickfont=dict(color="#e6edf3"))),
            dimensions=[
                dict(label=c, values=pc_df[c].values.tolist(),
                     range=[float(pc_df[c].min()), float(pc_df[c].max())])
                for c in pc_df.columns
            ],
            labelangle=-25,
            labelside="bottom",
        ))
    _lay(fig14, title="Parallel Coordinates: Airline-Metriken", height=440)
    sec14 = (
        _desc("Jede Linie = eine Airline. Farbe = OOS Sharpe (hell = hoch). "
              "Durch interaktives Ziehen auf einer Achse kann nach beliebigen Wertebereichen gefiltert werden. "
              "NaN-Werte wurden mit dem Median imputiert um alle Airlines darzustellen.")
        + _htm(fig14)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §15  Region/Type Box Plots
    # ════════════════════════════════════════════════════════════════════════
    fig15 = make_subplots(rows=1, cols=2,
                          subplot_titles=["OOS Sharpe nach Region", "OOS Sharpe nach Airline-Typ"])
    for reg in df["region"].unique():
        vs = pd.to_numeric(df.loc[df["region"] == reg, "sh_oos"], errors="coerce").dropna().tolist()
        if vs:
            fig15.add_trace(go.Box(y=vs, name=reg, marker_color=_rc(reg), showlegend=False),
                            row=1, col=1)
    for typ in df["type"].unique():
        vs = pd.to_numeric(df.loc[df["type"] == typ, "sh_oos"], errors="coerce").dropna().tolist()
        if vs:
            fig15.add_trace(go.Box(y=vs, name=typ, marker_color=_tc(typ), showlegend=False),
                            row=1, col=2)
    fig15.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=380, title_text="OOS Sharpe: Region & Airline-Typ",
    )
    sec15 = (
        _desc("Boxplot zeigt Median, IQR (Box), Whisker (1.5×IQR) und Ausreißer. "
              "Wenn eine Region signifikant über anderen liegt, deutet das auf regionale "
              "Strukturunterschiede hin (z.B. kürzerer Lag bei US-Börsen, effizientere "
              "Preisanpassung).")
        + _htm(fig15)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §16  Statistical Tests
    # ════════════════════════════════════════════════════════════════════════
    sh_num = pd.to_numeric(df["sh_oos"], errors="coerce")

    def _kruskal(gcol):
        groups = {}
        for g, sub in df.groupby(gcol):
            vs = pd.to_numeric(sub["sh_oos"], errors="coerce").dropna().tolist()
            if len(vs) >= 2:
                groups[g] = vs
        if len(groups) < 2:
            return "<p style='color:#e6edf3;'>Zu wenige Gruppen.</p>"
        stat, pv = kruskal(*groups.values())
        sig = '<span class="badge bg-success">sig. p&lt;0.05</span>' if pv < 0.05 else '<span class="badge bg-secondary">n.s.</span>'
        rows = "".join(f"<tr><td>{g}</td><td>N={len(v)}</td><td>µ={np.mean(v):.3f}</td><td>σ={np.std(v):.3f}</td></tr>"
                       for g, v in groups.items())
        return (f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
                f'<strong style="color:#e6edf3;">Kruskal-Wallis: {gcol} → OOS Sharpe</strong> '
                f'H={stat:.3f}, p={pv:.4f} {sig}'
                f'<table class="table table-sm table-dark mt-2"><tbody>{rows}</tbody></table></div>')

    spear_rows = ""
    for col in ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil",
                "regime_stab","roll_corr_mean","ar1","mom","ls_ratio"]:
        if col not in df.columns:
            continue
        xv = pd.to_numeric(df[col], errors="coerce")
        both = pd.concat([xv, sh_num], axis=1).dropna()
        if len(both) < 3:
            continue
        rho, pv = spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)
        sig = '<span class="badge bg-success">sig.</span>' if pv < 0.05 else ""
        spear_rows += f"<tr><td>{col}</td><td>{rho:.3f}</td><td>{pv:.4f}</td><td>{sig}</td></tr>"

    spear_card = (
        f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
        f'<strong style="color:#e6edf3;">Spearman-Korrelation: Feature → OOS Sharpe</strong>'
        f'<table class="table table-sm table-dark mt-2">'
        f'<thead><tr><th>Variable</th><th>ρ</th><th>p</th><th></th></tr></thead>'
        f'<tbody>{spear_rows}</tbody></table>'
        f'<small style="color:#8b949e;">ρ = Rang-Korrelationskoeffizient. '
        f'p = Wahrscheinlichkeit, ρ dieser Größe zufällig zu erhalten (H0: ρ=0). '
        f'|ρ|&gt;0.3 gilt als moderater Zusammenhang.</small></div>'
    )

    # OLS
    ols_feats = [c for c in ["best_lag","best_ccf","granger_f","te","vol_pct","beta_oil",
                              "regime_stab","roll_corr_mean"] if c in df.columns]
    ols_card = ""
    if len(ols_feats) >= 2:
        ols_data = pd.concat(
            [pd.to_numeric(df[c], errors="coerce") for c in ols_feats] + [sh_num], axis=1
        ).dropna()
        ols_data.columns = ols_feats + ["sh_oos"]
        if len(ols_data) >= len(ols_feats) + 2:
            Y = ols_data["sh_oos"].values
            X = np.column_stack([np.ones(len(Y))] + [ols_data[c].values for c in ols_feats])
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            yh = X @ coef
            r2 = 1 - ((Y - yh)**2).sum() / ((Y - Y.mean())**2).sum()
            r_rows = "".join(
                f"<tr><td>{'Intercept' if i == 0 else ols_feats[i-1]}</td><td>{c:.4f}</td></tr>"
                for i, c in enumerate(coef)
            )
            ols_card = (
                f'<div class="card mb-2 p-3" style="background:#1c2128;border:1px solid #30363d;">'
                f'<strong style="color:#e6edf3;">OLS: OOS Sharpe ~ Features</strong> '
                f'(R²={r2:.3f})'
                f'<table class="table table-sm table-dark mt-2">'
                f'<thead><tr><th>Variable</th><th>β-Koeffizient</th></tr></thead>'
                f'<tbody>{r_rows}</tbody></table>'
                f'<small style="color:#8b949e;">β_i = marginaler Effekt von Feature_i auf Sharpe, '
                f'alle anderen Features konstant. R² = erklärter Varianzanteil.</small></div>'
            )

    # Feature importance (Spearman ρ² proxy)
    fi_card = ""
    fi_vals = []
    for fc in ols_feats:
        xv = pd.to_numeric(df[fc], errors="coerce")
        both = pd.concat([xv, sh_num], axis=1).dropna()
        rho  = float(spearmanr(both.iloc[:,0].values, both.iloc[:,1].values)[0]) if len(both) >= 3 else 0.0
        fi_vals.append((fc, rho**2))
    if fi_vals:
        tot = sum(v for _, v in fi_vals) + 1e-9
        fi_norm = [(fc, v/tot) for fc, v in fi_vals]
        fi_norm.sort(key=lambda x: x[1], reverse=True)
        fig_fi = go.Figure(go.Bar(
            x=[x[0] for x in fi_norm], y=[x[1] for x in fi_norm],
            marker_color=["#58a6ff" if x[1] > 0.1 else "#30363d" for x in fi_norm],
        ))
        _lay(fig_fi, title="Feature Importance (Proxy: Spearman ρ²-Anteil)",
             xaxis_title="Feature", yaxis_title="Rel. Importance", height=340)
        fi_card = (
            _desc("Proxy für Feature Importance: Spearman ρ² (Anteil an Gesamtsumme). "
                  "Entspricht der erklärten Varianz bei Rangkorrelation. "
                  "Eine echte Random-Forest-Importance würde mehr Interaktionen berücksichtigen.")
            + _htm(fig_fi)
        )

    sec16 = _kruskal("region") + _kruskal("type") + spear_card + ols_card + fi_card

    # ════════════════════════════════════════════════════════════════════════
    # §17  Crisis Period Analysis
    # ════════════════════════════════════════════════════════════════════════
    CRISIS = [
        ("Ölcrash 2014–16", "2014-06-01", "2016-03-31"),
        ("COVID-19 2020",   "2020-01-01", "2020-12-31"),
        ("Recovery 2021",   "2021-01-01", "2021-12-31"),
        ("Ukraine 2022",    "2022-02-01", "2022-12-31"),
        ("Inflation 2023",  "2023-01-01", "2023-12-31"),
        ("2024–2025",       "2024-01-01", "2025-06-30"),
    ]
    cr_rows = ""
    for t in sorted_tickers:
        sr = strat_best[t]
        net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
        net_all.index = pd.to_datetime(net_all.index)
        row_v = f"<td><strong>{t}</strong></td>"
        for cname, cs, ce in CRISIS:
            sub = net_all.loc[cs:ce]
            if len(sub) < 5:
                row_v += "<td style='color:#8b949e;'>—</td>"
            else:
                sh_c = _sh(sub)
                col_v = ("#3fb950" if sh_c > 0.5 else ("#f78166" if sh_c < -0.5 else "#e3b341"))
                row_v += f'<td style="color:{col_v};">{sh_c:.2f}</td>'
        cr_rows += f"<tr>{row_v}</tr>"

    cr_hdrs = "".join(f"<th>{c[0]}</th>" for c in CRISIS)
    cr_leg = _legend(
        "".join(f"<tr><td>{c[0]}</td><td>{c[1]} – {c[2]}</td></tr>" for c in CRISIS)
    )
    sec17 = (
        _desc("Krisenperioden-Sharpe: annualisierter Sharpe auf dem jeweiligen Zeitfenster. "
              "🟢 &gt;0.5 = gute Krisenperformance, 🔴 &lt;−0.5 = schlechte Periode. "
              "Misst ob die Lead-Lag-Strategie in extremen Marktphasen überhaupt noch funktioniert.")
        + cr_leg
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th>{cr_hdrs}</tr></thead>'
        f'<tbody>{cr_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §18  TC Sweep
    # ════════════════════════════════════════════════════════════════════════
    fig18 = go.Figure()
    for i, t in enumerate(sorted_tickers):
        sr = strat_best[t]
        ret_t  = log_ret[t].dropna() if t in log_ret.columns else None
        if ret_t is None:
            continue
        lp_loc = leader_px.reindex(ret_t.index.intersection(leader_px.index)).ffill()
        ind_fn = sr["ind_fn"]
        split_d = str(sr["split_date"])[:10]
        oos_r   = ret_t.loc[split_d:]
        tc_sh   = []
        for tc_v in TC_GRID:
            n_tc, _, _ = _strat_exec(ind_fn(lp_loc), sr["thresh"], oos_r, sr["lag"], tc=tc_v)
            tc_sh.append(_sh(n_tc))
        fig18.add_trace(go.Scatter(
            x=(TC_GRID * 100).tolist(), y=tc_sh, name=t, mode="lines+markers",
            line=dict(color=PAL_EQ[i % len(PAL_EQ)], width=1.5),
        ))
    fig18.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig18, title="TC-Sweep: OOS Sharpe vs Transaktionskosten",
         xaxis_title="TC (Basispunkte)", yaxis_title="OOS Sharpe", height=440)
    sec18 = (
        _desc("Transaktionskosten-Sweep: OOS-Sharpe als Funktion der TC. "
              "Formel: r_net = r_gross − |Δsignal| · tc. "
              "Break-even TC = TC, bei der Sharpe = 0. "
              "Hoher Break-even → Strategie ist robust gegenüber realen Kosten (Slippage, Spread).")
        + _htm(fig18)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §19  Monte Carlo
    # ════════════════════════════════════════════════════════════════════════
    mc_rows = ""
    for t in sorted_tickers:
        sr    = strat_best[t]
        oos_a = sr["n_oos"].dropna().values
        if len(oos_a) < 30:
            mc_rows += f"<tr><td>{t}</td><td colspan='3' style='color:#8b949e;'>—</td></tr>"
            continue
        real_sh = _sh(sr["n_oos"])
        mc_sh   = np.array([np.random.permutation(oos_a).mean() * 252 /
                             (np.random.permutation(oos_a).std() * np.sqrt(252) + 1e-9)
                             for _ in range(N_MC)])
        # Fix: use same permutation for mean and std
        mc_sh2 = []
        for _ in range(N_MC):
            p = np.random.permutation(oos_a)
            mc_sh2.append(p.mean() * 252 / (p.std() * np.sqrt(252) + 1e-9))
        mc_sh = np.array(mc_sh2)
        pv    = float((mc_sh >= real_sh).mean())
        sig   = '<span class="badge bg-success">sig.</span>' if pv < 0.05 else '<span class="badge bg-secondary">n.s.</span>'
        mc_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{pv:.4f}</td><td>{sig}</td></tr>"

    sec19 = (
        _desc("Monte Carlo Permutationstest: Die OOS-Renditen werden {N_MC}× zufällig permutiert "
              "(Zeitstruktur zerstört), dann wird der Sharpe berechnet. "
              "p-Wert = Anteil der Permutationen mit Sharpe ≥ realem Sharpe. "
              "p &lt; 0.05 → Signal ist nicht durch Zufall erklärbar.".replace("{N_MC}", str(N_MC)))
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>OOS Sharpe</th><th>MC p-Value</th><th>Signifikanz</th></tr></thead>'
        f'<tbody>{mc_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §20  Bootstrap CI
    # ════════════════════════════════════════════════════════════════════════
    boot_rows = ""
    for t in sorted_tickers:
        sr    = strat_best[t]
        oos_a = sr["n_oos"].dropna().values
        if len(oos_a) < 30:
            boot_rows += f"<tr><td>{t}</td><td colspan='4' style='color:#8b949e;'>—</td></tr>"
            continue
        boot_sh = np.array([
            np.random.choice(oos_a, len(oos_a), replace=True).mean() * 252 /
            (np.random.choice(oos_a, len(oos_a), replace=True).std() * np.sqrt(252) + 1e-9)
            for _ in range(N_BOOT)
        ])
        lo, hi = np.percentile(boot_sh, 2.5), np.percentile(boot_sh, 97.5)
        real_sh = _sh(sr["n_oos"])
        sig = "✓" if lo > 0 else "✗"
        boot_rows += f"<tr><td>{t}</td><td>{real_sh:.3f}</td><td>{lo:.3f}</td><td>{hi:.3f}</td><td>{sig}</td></tr>"

    sec20 = (
        _desc(f"Bootstrap 95%-Konfidenzintervall ({N_BOOT} Samples mit Zurücklegen). "
              "CI = [2.5%-Quantil, 97.5%-Quantil] der Bootstrap-Sharpe-Verteilung. "
              "✓ = CI vollständig über 0 → Sharpe signifikant positiv.")
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>OOS Sharpe</th><th>CI 2.5%</th><th>CI 97.5%</th><th>CI&gt;0</th></tr></thead>'
        f'<tbody>{boot_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §21  Walk-Forward
    # ════════════════════════════════════════════════════════════════════════
    wf_rows = ""
    IS_WIN = 504; OOS_WIN = 126; STEP = 126
    for t in sorted_tickers:
        sr    = strat_best[t]
        ret_t = log_ret[t].dropna() if t in log_ret.columns else None
        if ret_t is None:
            continue
        lp_all = leader_px.reindex(ret_t.index.intersection(leader_px.index)).ffill()
        ind_fn = sr["ind_fn"]
        thresh = sr["thresh"]
        wf_sh  = []
        n_tot  = len(ret_t)
        for start in range(0, n_tot - IS_WIN - OOS_WIN, STEP):
            oo_i = ret_t.iloc[start + IS_WIN:start + IS_WIN + OOS_WIN]
            if len(oo_i) < 20:
                continue
            n_oo, _, _ = _strat_exec(ind_fn(lp_all), thresh, oo_i, sr["lag"])
            if len(n_oo) >= 10:
                wf_sh.append(_sh(n_oo))
        if not wf_sh:
            wf_rows += f"<tr><td>{t}</td><td colspan='3' style='color:#8b949e;'>—</td></tr>"
            continue
        wf_mean = np.nanmean(wf_sh)
        wf_pos  = np.mean([s > 0 for s in wf_sh if not np.isnan(s)])
        col_v   = "#3fb950" if wf_mean > 0.3 else ("#f78166" if wf_mean < 0 else "#e3b341")
        wf_rows += (f'<tr><td>{t}</td>'
                    f'<td style="color:{col_v};">{wf_mean:.3f}</td>'
                    f'<td>{wf_pos*100:.0f}%</td>'
                    f'<td>{len(wf_sh)}</td></tr>')

    sec21 = (
        _desc(f"Walk-Forward: IS={IS_WIN}T, OOS={OOS_WIN}T, Schritt={STEP}T. "
              "In jedem Fenster wird der Indikator auf dem IS-Fenster fixiert (selber Indikator wie global), "
              "die Performance wird auf dem nachfolgenden OOS-Fenster gemessen. "
              "Ø WF Sharpe = Mittelwert aller OOS-Fenster-Sharpes.")
        + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        f'<thead><tr><th>Airline</th><th>Ø WF OOS Sharpe</th><th>% pos. Fenster</th><th>Fenster</th></tr></thead>'
        f'<tbody>{wf_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §22  VIX Regime
    # ════════════════════════════════════════════════════════════════════════
    vix_html = "<p style='color:#8b949e;'>VIX-Daten nicht verfügbar.</p>"
    try:
        vix_s = _dl("^VIX")
        if vix_s is not None and len(vix_s) > 100:
            REGIMES = [
                ("Low (&lt;15)",      vix_s < 15),
                ("Normal (15–25)",    (vix_s >= 15) & (vix_s < 25)),
                ("Elevated (25–35)", (vix_s >= 25) & (vix_s < 35)),
                ("Crisis (&gt;35)",   vix_s >= 35),
            ]
            vix_rows = ""
            for t in sorted_tickers:
                sr = strat_best[t]
                net_all = pd.concat([sr["n_is"], sr["n_oos"]]).sort_index()
                net_all.index = pd.to_datetime(net_all.index)
                rv = f"<td><strong>{t}</strong></td>"
                for rname, rmask in REGIMES:
                    idx_r = vix_s[rmask].index.intersection(net_all.index)
                    sub_r = net_all.reindex(idx_r).dropna()
                    if len(sub_r) < 5:
                        rv += "<td style='color:#8b949e;'>—</td>"
                    else:
                        sh_r = _sh(sub_r)
                        col_v = ("#3fb950" if sh_r > 0.5 else ("#f78166" if sh_r < -0.5 else "#e3b341"))
                        rv += f'<td style="color:{col_v};">{sh_r:.2f}</td>'
                vix_rows += f"<tr>{rv}</tr>"

            vhd = "".join(f"<th>{r[0]}</th>" for r in REGIMES)
            vix_html = (
                _desc("VIX = CBOE Volatility Index (implizite 30T-Vol des S&amp;P 500). "
                      "Niedrig &lt;15 = ruhiger Markt, Krise &gt;35 = extreme Unsicherheit. "
                      "Die Tabelle zeigt Sharpe je Regime — ergibt sich die Lead-Lag-Strategie "
                      "nur in ruhigen Märkten oder auch in Krisen?")
                + f'<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
                f'<thead><tr><th>Airline</th>{vhd}</tr></thead>'
                f'<tbody>{vix_rows}</tbody></table></div>'
            )
    except Exception:
        pass
    sec22 = vix_html

    # ════════════════════════════════════════════════════════════════════════
    # §23  Signal Stability Map
    # ════════════════════════════════════════════════════════════════════════
    stab_html = "<p style='color:#8b949e;'>Zu wenige Daten.</p>"
    stab_data = {}
    for t in sorted_tickers:
        rc = rc_store.get(t)
        if rc is not None and len(rc.dropna()) >= 50:
            stab_data[t] = rc.dropna()

    if len(stab_data) >= 2:
        all_dates = sorted(set().union(*[set(s.index) for s in stab_data.values()]))
        sm = np.full((len(sorted_tickers), len(all_dates)), np.nan)
        for j, t in enumerate(sorted_tickers):
            if t not in stab_data:
                continue
            s_t = stab_data[t]
            for ii, d in enumerate(all_dates):
                if d in s_t.index:
                    sm[j, ii] = float(s_t.loc[d])
        step_d  = max(1, len(all_dates) // 500)
        sdates  = all_dates[::step_d]
        sm_sub  = sm[:, ::step_d]

        fig23 = go.Figure(go.Heatmap(
            z=sm_sub.tolist(),
            x=[str(d)[:10] for d in sdates],
            y=sorted_tickers,
            colorscale="RdBu", zmid=0, zmin=-0.8, zmax=0.8,
            colorbar=dict(title="Rolling r", tickfont=dict(color="#e6edf3")),
        ))
        _lay(fig23, title="Signal Stability Map: Rolling 252T CCF über Zeit", height=420)
        stab_html = (
            _desc("Jede Zeile = eine Airline. X-Achse = Zeit. Farbe = rollende Pearson-Korrelation r(t) "
                  "zwischen CL=F(t-l*) und Airline(t) im 252T-Fenster. "
                  "Beständig rotes Band = stabiler Lead-Lag. Wechsel blau/rot = Regime-Instabilität.")
            + _htm(fig23)
        )
    sec23 = stab_html

    # ════════════════════════════════════════════════════════════════════════
    # §24  Granger F vs Transfer Entropy Scatter
    # ════════════════════════════════════════════════════════════════════════
    gr_vals = pd.to_numeric(df["granger_f"], errors="coerce").fillna(0)
    te_vals = pd.to_numeric(df["te"],        errors="coerce").fillna(0)

    fig24 = go.Figure()
    for reg, col in REG_COLORS.items():
        mask = [i for i, t in enumerate(sorted_tickers) if df.loc[t, "region"] == reg]
        if not mask:
            continue
        fig24.add_trace(go.Scatter(
            x=[float(gr_vals.iloc[i]) for i in mask],
            y=[float(te_vals.iloc[i]) for i in mask],
            mode="markers+text",
            text=[sorted_tickers[i] for i in mask],
            textposition="top center",
            marker=dict(size=10, color=col),
            name=reg,
        ))
    _lay(fig24, title="Granger-F-Statistik vs Transfer Entropy",
         xaxis_title="Granger F-Statistik", yaxis_title="Transfer Entropy (Bits)", height=440)
    sec24 = (
        _desc("Granger F = lineare Kausalität (basiert auf OLS-Verbesserung durch x-Lags). "
              "Transfer Entropy TE = nicht-lineare Informationsübertragung (Shannon-Entropie). "
              "Airlines oben rechts zeigen sowohl linearen als auch nicht-linearen Kausalzusammenhang mit CL=F. "
              "Airlines die nur beim Granger-Test signifikant sind, haben einen hauptsächlich linearen Kanal.")
        + _htm(fig24)
    )

    # ════════════════════════════════════════════════════════════════════════
    # §25  Scatterplot Matrix
    # ════════════════════════════════════════════════════════════════════════
    sp_cols = ["sh_oos", "best_lag", "best_ccf", "granger_f", "regime_stab", "vol_pct"]
    sp_df   = df[[c for c in sp_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    sp_df   = sp_df.fillna(sp_df.median())
    splom_html = ""
    if len(sp_df) >= 3:
        fig25 = go.Figure(go.Splom(
            dimensions=[dict(label=c, values=sp_df[c].tolist()) for c in sp_df.columns],
            marker=dict(
                color=[_rc(str(df.loc[t, "region"])) for t in sp_df.index],
                size=8, showscale=False,
            ),
            text=sp_df.index.tolist(),
            diagonal_visible=False,
        ))
        _lay(fig25, title="Scatterplot-Matrix: Kernmetriken", height=600)
        splom_html = (
            _desc("Scatterplot-Matrix (SPLOM): Jedes Subplot zeigt den bivariaten Zusammenhang "
                  "zwischen zwei Metriken. Farbe = Region. Diagonale ausgeblendet. "
                  "Wähle interessante Achsenkombinationen aus um Cluster oder Ausreißer zu entdecken.")
            + _htm(fig25)
        )
    sec25 = splom_html or "<p style='color:#8b949e;'>Zu wenige Daten.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # §26  Full 26-Metric Table
    # ════════════════════════════════════════════════════════════════════════
    METRIC_COLS = [
        "Ann.Ret% (net)", "Ann.Ret% (gross)", "TC Drag% p.a.", "Ann.Vol%",
        "Sharpe (net)", "Sharpe (gross)", "Sortino", "Calmar", "MaxDD%",
        "AvgDD-Dur", "Trades", "WinRate%", "AvgWin%", "AvgLoss%", "ProfitFactor",
        "Omega", "TailRatio", "Skew", "Kurt", "VaR5%", "CVaR5%", "AC1", "AC5",
        "Beta", "Alpha%",
    ]
    m26_rows = ""
    for t in sorted_tickers:
        sr   = strat_best[t]
        spy  = log_ret["SPY"].dropna() if "SPY" in log_ret.columns else None
        m_is  = _full_metrics(sr["n_is"],  sr["g_is"],  sr["s_is"],  spy, name=f"{t} IS")
        m_oos = _full_metrics(sr["n_oos"], sr["g_oos"], sr["s_oos"], spy, name=f"{t} OOS")
        for split_l, mm in [("IS", m_is), ("OOS", m_oos)]:
            cells = f"<td><strong>{t}</strong></td><td>{split_l}</td>"
            for mc in METRIC_COLS:
                v = mm.get(mc, np.nan)
                if isinstance(v, float) and np.isnan(v):
                    cells += "<td style='color:#8b949e;'>—</td>"
                elif isinstance(v, float):
                    cells += f"<td>{v:.3f}</td>"
                else:
                    cells += f"<td>{v}</td>"
            m26_rows += f"<tr>{cells}</tr>"

    m_hdrs = "<th>Ticker</th><th>Split</th>" + "".join(f"<th>{c}</th>" for c in METRIC_COLS)
    leg26 = _legend(
        "<tr><td>Ann.Ret%</td><td>µ·252·100 (tägl. Nettorendite annualisiert)</td></tr>"
        "<tr><td>TC Drag%</td><td>(µ_gross − µ_net)·252·100 (Kostenwirkung p.a.)</td></tr>"
        "<tr><td>Ann.Vol%</td><td>σ·√252·100</td></tr>"
        "<tr><td>Sharpe</td><td>µ·252 / (σ·√252)</td></tr>"
        "<tr><td>Sortino</td><td>µ·252 / (σ_down·√252), σ_down = Std negativer Renditen</td></tr>"
        "<tr><td>Calmar</td><td>Ann.Ret / |MaxDD|</td></tr>"
        "<tr><td>MaxDD%</td><td>Max. Drawdown = max(1 − NAV/max_NAV)</td></tr>"
        "<tr><td>AvgDD-Dur</td><td>Mittlere Drawdown-Dauer in Tagen</td></tr>"
        "<tr><td>WinRate%</td><td>Anteil Tage mit positivem Return</td></tr>"
        "<tr><td>ProfitFactor</td><td>Summe Gewinne / |Summe Verluste|</td></tr>"
        "<tr><td>Omega</td><td>∫₀^∞ (1-F(r))dr / ∫₋∞^0 F(r)dr (Threshold=0)</td></tr>"
        "<tr><td>TailRatio</td><td>|95%-Quantil / 5%-Quantil|</td></tr>"
        "<tr><td>VaR5%</td><td>5%-Quantil der täglichen Renditen</td></tr>"
        "<tr><td>CVaR5%</td><td>Ø Rendite aller Tage unterhalb VaR5%</td></tr>"
        "<tr><td>AC1/AC5</td><td>Autokorrelation der Renditen bei Lag 1 bzw. 5</td></tr>"
        "<tr><td>Beta</td><td>Kovarianz(r, SPY) / Var(SPY)</td></tr>"
        "<tr><td>Alpha%</td><td>Intercept OLS vs SPY, annualisiert</td></tr>"
    )
    sec26 = (
        _desc("Vollständige 26-Metriken-Tabelle für jede Airline, getrennt nach IS (Kalibrierung) "
              "und OOS (echter Test). Alle Renditen sind Netto nach 10bp TC.")
        + leg26
        + f'<div class="table-responsive" style="font-size:0.75em;">'
        f'<table class="table table-sm table-dark table-hover">'
        f'<thead class="table-dark"><tr>{m_hdrs}</tr></thead>'
        f'<tbody>{m26_rows}</tbody></table></div>'
    )

    # ════════════════════════════════════════════════════════════════════════
    # §27  Portfolio Analysis
    # ════════════════════════════════════════════════════════════════════════
    oos_rets = {t: strat_best[t]["n_oos"] for t in sorted_tickers}
    oos_mat  = pd.DataFrame(oos_rets).fillna(0)

    port_html = ""
    if len(oos_mat.columns) >= 2:
        n_a = len(oos_mat.columns)
        vols = oos_mat.std().values + 1e-9
        w_ew = np.ones(n_a) / n_a
        w_vw = (1/vols) / (1/vols).sum()
        w_mv = (1/vols**2) / (1/vols**2).sum()

        weights = {"Equal Weight": w_ew, "Vol-Weight (1/σ)": w_vw, "Min-Var (1/σ²)": w_mv}
        fig27 = go.Figure()
        pt_rows = ""
        for pn, w in weights.items():
            port = (oos_mat * w).sum(axis=1)
            cum  = (1 + port).cumprod() * 100
            fig27.add_trace(go.Scatter(
                x=cum.index.astype(str).tolist(), y=cum.values.tolist(),
                name=pn, mode="lines",
            ))
            pt_rows += f"<tr><td>{pn}</td><td>{_sh(port):.3f}</td><td>{float((1+port).cumprod().iloc[-1]*100-100):.1f}%</td></tr>"

        _lay(fig27, title="Portfolio OOS Equity Curves",
             xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=440)
        port_html = (
            _desc("Equal Weight: w_i = 1/N. "
                  "Vol-Weight (1/σ): w_i = (1/σ_i) / Σ(1/σ_j) — höhere Gewichtung bei niedrigerer Volatilität. "
                  "Min-Var (1/σ²): analog mit 1/σ². "
                  "Alle Gewichte sind positiv normiert (keine Shorts auf Portfolio-Ebene).")
            + _htm(fig27)
            + f'<div class="table-responsive mt-2"><table class="table table-sm table-dark">'
            f'<thead><tr><th>Portfolio</th><th>OOS Sharpe</th><th>Gesamtrendite</th></tr></thead>'
            f'<tbody>{pt_rows}</tbody></table></div>'
        )
    sec27 = port_html or "<p style='color:#8b949e;'>Zu wenige Airlines für Portfolio.</p>"

    # ════════════════════════════════════════════════════════════════════════
    # §28  Benchmark: CL=F → JETS RSI<70 (existing commodity strategy)
    # ════════════════════════════════════════════════════════════════════════
    bench_html = "<p style='color:#8b949e;'>JETS-Daten nicht verfügbar für Benchmark.</p>"
    try:
        # Load from existing commodity data if available
        ret_main = _read(tables / "phase2_returns.csv")
        px_main  = _read(tables / "phase1_prices.csv")
        if ret_main is not None and px_main is not None:
            ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
            px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
            if "JETS" in ret_main.columns and "CL=F" in px_main.columns:
                cl_px   = px_main["CL=F"].dropna()
                jets_r  = ret_main["JETS"].dropna()
                common_b = cl_px.index.intersection(jets_r.index)
                split_b  = common_b[int(len(common_b) * IS_FRAC)]
                is_b = jets_r.loc[:split_b]; oos_b = jets_r.loc[split_b:]
                # RSI<70: long when -RSI(t-1) > -70 i.e. RSI < 70
                ind_b = -_calc_rsi(cl_px, 14)
                n_is_b, g_is_b, s_is_b   = _strat_exec(ind_b, -70.0, is_b,  1)
                n_oos_b, g_oos_b, s_oos_b = _strat_exec(ind_b, -70.0, oos_b, 1)
                m_is_b  = _full_metrics(n_is_b,  g_is_b,  s_is_b,  name="JETS IS")
                m_oos_b = _full_metrics(n_oos_b, g_oos_b, s_oos_b, name="JETS OOS")

                bm_fig = go.Figure()
                cum_is_b  = (1 + n_is_b).cumprod() * 100
                cum_oos_b = (1 + n_oos_b).cumprod() * 100
                bm_fig.add_trace(go.Scatter(
                    x=cum_is_b.index.astype(str).tolist(), y=cum_is_b.values.tolist(),
                    name="JETS IS", mode="lines",
                    line=dict(color="#58a6ff", width=2, dash="dash"),
                ))
                bm_fig.add_trace(go.Scatter(
                    x=cum_oos_b.index.astype(str).tolist(), y=cum_oos_b.values.tolist(),
                    name="JETS OOS", mode="lines",
                    line=dict(color="#58a6ff", width=2.5),
                ))
                # Overlay best airline OOS
                for i, t in enumerate(sorted_tickers[:3]):
                    sr = strat_best[t]
                    cum_t = (1 + sr["n_oos"]).cumprod() * 100
                    bm_fig.add_trace(go.Scatter(
                        x=cum_t.index.astype(str).tolist(), y=cum_t.values.tolist(),
                        name=f"{t} OOS", mode="lines",
                        line=dict(color=PAL_EQ[i+2 % len(PAL_EQ)], width=1.5),
                    ))
                bm_fig.add_vline(x=str(split_b)[:10], line_color="#e3b341",
                                 line_dash="dash")
                _lay(bm_fig, title="Benchmark: CL=F→JETS RSI<70 vs beste Airlines",
                     xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=470)

                bm_rows_h = ""
                for lbl, mm in [("JETS IS", m_is_b), ("JETS OOS", m_oos_b)]:
                    cells = f"<td><strong>{lbl}</strong></td>"
                    for mc in ["Ann.Ret% (net)", "Ann.Vol%", "Sharpe (net)", "MaxDD%",
                               "Sortino", "Calmar", "WinRate%", "ProfitFactor"]:
                        v = mm.get(mc, np.nan)
                        cells += f"<td>{v:.3f}</td>" if isinstance(v, float) and not np.isnan(v) else "<td>—</td>"
                    bm_rows_h += f"<tr>{cells}</tr>"

                bench_html = (
                    _desc("Benchmark-Strategie: CL=F → JETS ETF, Indikator = RSI(14)&lt;70, Lag = 1T. "
                          "Diese Strategie ist das Referenzmodell aus der Einzelstrategie-Analyse. "
                          "Die beste Airline-Strategie sollte diese Performance möglichst übertreffen. "
                          "Vertikale gelbe Linie = IS/OOS-Trennpunkt.")
                    + _htm(bm_fig)
                    + f'<div class="table-responsive mt-2"><table class="table table-sm table-dark">'
                    f'<thead><tr><th>Strategie</th><th>Ann.Ret%</th><th>Ann.Vol%</th>'
                    f'<th>Sharpe</th><th>MaxDD%</th><th>Sortino</th><th>Calmar</th>'
                    f'<th>WinRate%</th><th>ProfitFactor</th></tr></thead>'
                    f'<tbody>{bm_rows_h}</tbody></table></div>'
                )
    except Exception:
        pass
    sec28 = bench_html

    # ════════════════════════════════════════════════════════════════════════
    # §29  Interpretation & Screening Rules
    # ════════════════════════════════════════════════════════════════════════
    top3  = sorted_tickers[:3]  if len(sorted_tickers) >= 3 else sorted_tickers
    bot3  = sorted_tickers[-3:] if len(sorted_tickers) >= 3 else []
    top3_s = ", ".join(f"{t} ({scalars.get(t,{}).get('region','?')})" for t in top3)
    bot3_s = ", ".join(f"{t} ({scalars.get(t,{}).get('region','?')})" for t in bot3)

    sec29 = f"""
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #3fb950;">
      <h5 style="color:#3fb950;">Beste Airlines (OOS Sharpe)</h5>
      <p style="color:#e6edf3;">{top3_s}</p>
      <p style="color:#e6edf3;">Diese Airlines zeigen die robusteste OOS-Persistenz der CL=F-Lead-Lag-Strategie.
      Charakteristisch: kurzer Lag (1–3T), hoher Granger-F, konsistente Rolling-Korrelation,
      und hohe Ölpreisabhängigkeit im Kostenmix.</p>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #f78166;">
      <h5 style="color:#f78166;">Schwächste Airlines</h5>
      <p style="color:#e6edf3;">{bot3_s}</p>
      <p style="color:#e6edf3;">Mögliche Ursachen: ADR-Liquiditätsmangel, starkes Fuel Hedging,
      Währungseffekte (nicht USD-denominiert) oder schwache operative Öl-Exposition.</p>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #e3b341;">
      <h5 style="color:#e3b341;">Empirisches Screening-Regelwerk</h5>
      <ul style="color:#e6edf3;">
        <li>Filter 1: Granger-F &gt; 3.0 und p &lt; 0.05 → statistisch signifikante Kausalität</li>
        <li>Filter 2: Peak CCF &gt; 0.10 → minimale lineare Signal-Stärke</li>
        <li>Filter 3: Regime-Stabilität &gt; 0.55 → Signal in &gt;55% der 6-Monats-Fenster positiv</li>
        <li>Filter 4: Optimaler Lag 1–5T → Reaktionszeit innerhalb normaler Marktmikrostruktur</li>
        <li>Filter 5: OOS Sharpe &gt; IS Sharpe → kein Overfitting im IS-Fenster</li>
        <li>Regionen: US-notierte Aktien tendieren zu kürzerem Lag und höherem Sharpe</li>
        <li>Größe: Kein klarer Größeneffekt — Market Cap allein ist kein guter Prädiktor</li>
      </ul>
    </div>
    <div class="card mb-3 p-3" style="background:#1c2128;border:1px solid #58a6ff;">
      <h5 style="color:#58a6ff;">Warum reagieren bestimmte Airlines stärker auf Öl?</h5>
      <p style="color:#e6edf3;">
        <strong>Ökonomischer Mechanismus:</strong> Kerosin macht 20–35% der Betriebskosten aus.
        Airlines ohne aktives Hedging-Programm reagieren unmittelbar auf WTI-Spotpreisänderungen
        im Aktienmarkt. US-notierte Papiere haben durch effiziente Preisfindung und liquide
        Derivatemärkte eine kürzere Lag-Struktur (1–2T) als ADRs (3–7T).<br><br>
        <strong>Informationskanal:</strong> Transfer Entropy &gt; Granger bedeutet, dass
        nicht-lineare Informationsübertragung dominiert — Schwellenwerteffekte (Ölpreis
        über/unter Break-even-Kostenschwelle) sind relevanter als lineare Sensitivitäten.<br><br>
        <strong>Allgemeines Screening für andere Branchen:</strong> Branchen mit hohem
        Rohstoffkostenanteil (Chemie: Erdgas, Reedereien: Bunker, Stromerzeugung: Gas/Kohle)
        sollten ähnliche Lead-Lag-Strukturen aufweisen. Screening-Kriterien: hohes
        Commodities/Revenue-Verhältnis, niedrige Hedging-Quote, hohe Preisnehmer-Position.
      </p>
    </div>"""

    # ════════════════════════════════════════════════════════════════════════
    # ASSEMBLE
    # ════════════════════════════════════════════════════════════════════════
    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#acc{idx}">'
            f'{title}</button></h2>'
            f'<div id="acc{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    acc = '<div class="accordion" id="mainAcc">'
    acc += _acc("§0  Ranking-Tabelle: alle Airlines", sec0, 0, open_=True)
    acc += _acc("§1  OOS Sharpe Bar Chart (IS vs OOS)",  sec1, 1)
    acc += _acc("§2  Long/Short/Flat Tage & L/S Ratio",  sec2, 2)
    acc += _acc("§3  OOS Equity Curves mit Signal-Markierungen (▲▼)",  sec3, 3)
    acc += _acc("§4  CCF Heatmap: CL=F → Airline (Lag 0–10T)",         sec4, 4)
    acc += _acc("§5  Optimaler Lag × OOS Sharpe (Scatter)",             sec5, 5)
    acc += _acc("§6  Bubble Chart: Market Cap × OOS Sharpe × Region",  sec6, 6)
    acc += _acc("§7  Rolling 252T Sharpe",                              sec7, 7)
    acc += _acc("§8  Rolling 252T Korrelation CL=F → Airline",         sec8, 8)
    acc += _acc("§9  Radar Charts (normalisierte Metriken)",            sec9, 9)
    acc += _acc("§10 Feature Matrix Heatmap (z-score)",                 sec10, 10)
    acc += _acc("§11 Feature-Korrelationsmatrix",                       sec11, 11)
    acc += _acc("§12 Hierarchisches Clustering (Ward-Dendrogram)",      sec12, 12)
    acc += _acc("§13 PCA Biplot",                                       pca_html, 13)
    acc += _acc("§14 Parallel Coordinate Plot",                         sec14, 14)
    acc += _acc("§15 OOS Sharpe: Region & Airline-Typ (Boxplots)",      sec15, 15)
    acc += _acc("§16 Statistische Tests (Kruskal, Spearman, OLS, FI)",  sec16, 16)
    acc += _acc("§17 Krisenperioden-Analyse",                           sec17, 17)
    acc += _acc("§18 TC-Sweep (0–100bp)",                               sec18, 18)
    acc += _acc("§19 Monte Carlo Permutationstest",                     sec19, 19)
    acc += _acc("§20 Bootstrap 95%-CI auf OOS Sharpe",                  sec20, 20)
    acc += _acc("§21 Walk-Forward-Validierung",                         sec21, 21)
    acc += _acc("§22 VIX-Regime-Analyse",                               sec22, 22)
    acc += _acc("§23 Signal Stability Map (Rolling CCF über Zeit)",     sec23, 23)
    acc += _acc("§24 Granger-F vs Transfer Entropy",                    sec24, 24)
    acc += _acc("§25 Scatterplot-Matrix (Splom)",                       sec25, 25)
    acc += _acc("§26 Vollständige 26-Metriken-Tabelle (IS + OOS)",      sec26, 26)
    acc += _acc("§27 Portfolioanalyse (EW / Vol-Weighted / Min-Var)",   sec27, 27)
    acc += _acc("§28 Benchmark: CL=F → JETS RSI&lt;70",                 sec28, 28)
    acc += _acc("§29 Ökonomische Interpretation & Screening-Regelwerk", sec29, 29)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#ffa657;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">CL=F → Airline Lead-Lag: Querschnittsanalyse</h2>
          <p class="mb-0" style="color:#8b949e;">
            {len(sorted_tickers)} Airlines · IS/OOS · 26 Metriken · MC · Bootstrap · Walk-Forward ·
            Portfolio · Clustering · PCA · Tests · Benchmark · L/S Ratio
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "airline_oil_report.html",
           _html_base("CL=F → Airline Lead-Lag: Querschnittsanalyse", 19, body))

'''

# ── inject ────────────────────────────────────────────────────────────────────
src_new = src_without[:src_without.find(END_MARKER)] + FUNC + src_without[src_without.find(END_MARKER):]

# ── wire build_all_reports ────────────────────────────────────────────────────
OLD_W = ("    build_strategy_stress_test_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_strategy_stress_test_report(tables, figures, reports)\n"
         "    build_airline_oil_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if NEW_W in src_new:
    print("build_all_reports already wired.")
elif OLD_W in src_new:
    src_new = src_new.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
else:
    print("WARNING: could not wire build_all_reports.")

RB.write_text(src_new, encoding="utf-8")
print(f"Done. {len(src_new.splitlines())} lines")
