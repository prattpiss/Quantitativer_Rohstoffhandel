"""
Inject build_seasonality_report into report_builder.py.

Report covers:
  - Calendar heatmap of monthly strategy returns (year × month)
  - Day-of-week seasonality
  - Monthly & quarterly average returns with bootstrap CI
  - Autocorrelation of strategy returns (cycles)
  - Seasonal filter: only trade in profitable months (IS-optimized), else flat
  - IS/OOS comparison: baseline vs seasonal-filtered strategy
  - Rolling seasonal decomposition
  - Annual return bar charts per airline
  - Best/worst month ranking
  - Apply seasonal filter to all top airlines from cross-sectional study
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

FUNC = r'''
def build_seasonality_report(tables, figures, out):  # noqa: C901
    """
    Seasonal pattern analysis and optimization of CL=F lead-lag strategies.
    Identifies cyclical weaknesses and constructs seasonally-filtered strategies.
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
    from scipy.stats import pearsonr

    IS_FRAC = 0.70
    N_BOOT  = 1000
    TC      = 0.001

    # ── load data ─────────────────────────────────────────────────────────────
    ret_main = _read(tables / "phase2_returns.csv")
    px_main  = _read(tables / "phase1_prices.csv")

    if ret_main is None or px_main is None:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Daten fehlen.</p>"))
        return

    ret_main.index = pd.to_datetime(ret_main.index, errors="coerce")
    px_main.index  = pd.to_datetime(px_main.index,  errors="coerce")
    ret_main = ret_main[ret_main.index.notna()]
    px_main  = px_main[px_main.index.notna()]

    # ── strategy targets: JETS + download top individual airlines ─────────────
    TARGETS = {
        "JETS":  {"name": "US Global Jets ETF", "region": "USA"},
        "DAL":   {"name": "Delta Air Lines",    "region": "USA"},
        "UAL":   {"name": "United Airlines",    "region": "USA"},
        "AAL":   {"name": "American Airlines",  "region": "USA"},
        "LUV":   {"name": "Southwest Airlines", "region": "USA"},
        "IAG":   {"name": "IAG (ADR)",          "region": "Europe"},
        "CPA":   {"name": "Copa Holdings",      "region": "LatAm"},
    }

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

    # Prefer existing data, download missing
    leader_px = px_main["CL=F"].dropna() if "CL=F" in px_main.columns else None
    if leader_px is None:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>CL=F fehlt.</p>"))
        return

    follower_rets = {}
    for ticker in TARGETS:
        if ticker in ret_main.columns:
            follower_rets[ticker] = ret_main[ticker].dropna()
        else:
            s = _dl(ticker)
            if s is not None:
                lr = np.log(s / s.shift(1)).dropna()
                follower_rets[ticker] = lr

    available = [t for t in TARGETS if t in follower_rets]
    if not available:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Keine Follower-Daten.</p>"))
        return

    # ── helper functions ──────────────────────────────────────────────────────
    def _sh(x):
        x = pd.Series(x).dropna()
        if len(x) < 20: return np.nan
        return float(x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9))

    def _mdd(x):
        c = (1 + pd.Series(x)).cumprod()
        return float((c / c.cummax() - 1).min())

    def _boot_ci(arr, fn=None, n=N_BOOT, q=(2.5, 97.5)):
        fn = fn or (lambda x: x.mean())
        bs = [fn(pd.Series(np.random.choice(arr, len(arr), replace=True))) for _ in range(n)]
        return np.percentile(bs, q)

    def _strat_run(leader_prices, follower_ret, lag=1, thresh=-70.0,
                   ind_fn=None, tc=TC):
        if ind_fn is None:
            ind_fn = lambda p: -_calc_rsi(p, 14)
        return _strat_exec(ind_fn(leader_prices), thresh, follower_ret, lag, tc=tc)

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

    MONTH_NAMES = ["Jan","Feb","Mär","Apr","Mai","Jun",
                   "Jul","Aug","Sep","Okt","Nov","Dez"]
    DOW_NAMES   = ["Mo","Di","Mi","Do","Fr"]
    PAL = px.colors.qualitative.Plotly

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A – Per-ticker seasonal analysis
    # ══════════════════════════════════════════════════════════════════════════
    sections = {}
    best_months_per_ticker = {}   # IS-optimized good months
    strat_records = {}

    for ticker in available:
        fr = follower_rets[ticker]
        common = leader_px.index.intersection(fr.index)
        common = common[~common.duplicated()]
        if len(common) < 300:
            continue

        lp = leader_px.reindex(common).ffill()
        fr_c = fr.reindex(common).fillna(0.0)

        split_i    = int(len(common) * IS_FRAC)
        split_date = common[split_i]
        is_idx     = common[:split_i]
        oos_idx    = common[split_i:]

        # Baseline strategy: CL=F RSI<70, Lag=1
        n_is,  g_is,  s_is  = _strat_run(lp, fr_c.loc[is_idx])
        n_oos, g_oos, s_oos = _strat_run(lp, fr_c.loc[oos_idx])

        net_all = pd.concat([n_is, n_oos]).sort_index()
        net_all.index = pd.to_datetime(net_all.index)

        # ── 1. Monthly return matrix (year × month) ──────────────────────────
        net_df = net_all.to_frame("r")
        net_df["year"]  = net_df.index.year
        net_df["month"] = net_df.index.month
        net_df["dow"]   = net_df.index.dayofweek  # 0=Mon

        monthly_pnl = net_df.groupby(["year","month"])["r"].sum().unstack(fill_value=np.nan)
        monthly_pnl.columns = [MONTH_NAMES[m-1] for m in monthly_pnl.columns]

        # ── 2. Average monthly return + bootstrap CI ─────────────────────────
        avg_month = {}
        ci_month  = {}
        for m in range(1, 13):
            vals = net_df[net_df["month"] == m]["r"].values
            if len(vals) < 5:
                avg_month[MONTH_NAMES[m-1]] = 0.0
                ci_month[MONTH_NAMES[m-1]]  = (0.0, 0.0)
                continue
            avg_month[MONTH_NAMES[m-1]] = float(vals.mean())
            lo, hi = _boot_ci(vals, lambda x: x.mean())
            ci_month[MONTH_NAMES[m-1]] = (float(lo), float(hi))

        # IS-only monthly performance (for filter optimization)
        is_net_df = pd.Series(n_is.values, index=pd.to_datetime(n_is.index))
        is_net_df = is_net_df.to_frame("r")
        is_net_df["month"] = is_net_df.index.month
        avg_month_is = is_net_df.groupby("month")["r"].mean()

        # Good months: average IS return > 0
        good_months = set(avg_month_is[avg_month_is > 0].index.tolist())
        best_months_per_ticker[ticker] = good_months

        # ── 3. Day-of-week seasonality ───────────────────────────────────────
        avg_dow = {}
        for d in range(5):
            vals = net_df[net_df["dow"] == d]["r"].values
            avg_dow[DOW_NAMES[d]] = float(vals.mean()) if len(vals) > 5 else 0.0

        # ── 4. Quarterly analysis ────────────────────────────────────────────
        net_df["quarter"] = ((net_df["month"] - 1) // 3) + 1
        avg_qtr = net_df.groupby("quarter")["r"].mean()

        # ── 5. Seasonal-filtered strategy ────────────────────────────────────
        # Only trade (signals active) in IS-profitable months; else flat (signal=0)
        def _seasonal_filter(net_series, sig_series, good_m):
            """Zero out signals in 'bad' months (months not in good_m)."""
            idx = pd.to_datetime(sig_series.index)
            mask = idx.month.isin(good_m)
            sig_f = sig_series.copy()
            sig_f[~mask] = 0
            # Recompute net with filtered signals
            fr_sub = fr_c.reindex(sig_series.index).fillna(0.0)
            gross_f = sig_f * fr_sub
            net_f   = gross_f - sig_f.diff().abs().fillna(0) * TC
            return net_f

        # OOS seasonal filter (using IS-derived good months)
        n_oos_sf = _seasonal_filter(n_oos, s_oos, good_months)

        # ── 6. Autocorrelation of strategy returns ───────────────────────────
        acf_lags = list(range(1, 63))
        acf_vals = [float(net_all.autocorr(lag=l)) for l in acf_lags]

        # ── 7. Rolling 21-day return (cyclic pattern) ────────────────────────
        roll21 = net_all.rolling(21).sum().dropna() * 100

        # ── store results ────────────────────────────────────────────────────
        strat_records[ticker] = {
            "n_is": n_is, "n_oos": n_oos, "n_oos_sf": n_oos_sf,
            "s_is": s_is, "s_oos": s_oos,
            "split_date": split_date,
            "monthly_pnl": monthly_pnl,
            "avg_month": avg_month, "ci_month": ci_month,
            "avg_dow": avg_dow, "avg_qtr": avg_qtr,
            "good_months": good_months,
            "acf_lags": acf_lags, "acf_vals": acf_vals,
            "roll21": roll21,
            "sh_base": _sh(n_oos), "sh_sf": _sh(n_oos_sf),
        }

    if not strat_records:
        _write(out / "seasonality_report.html",
               _html_base("Saisonalität", 19, "<p>Keine Ergebnisse.</p>"))
        return

    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════

    # ── S0: Overview comparison table ────────────────────────────────────────
    ov_rows = ""
    for ticker, rec in strat_records.items():
        gm = ", ".join(MONTH_NAMES[m-1] for m in sorted(rec["good_months"]))
        sh_base = rec["sh_base"]
        sh_sf   = rec["sh_sf"]
        delta   = sh_sf - sh_base if not (np.isnan(sh_base) or np.isnan(sh_sf)) else np.nan
        col_d   = "#3fb950" if delta > 0 else "#f78166"
        ov_rows += (
            f"<tr><td><strong>{ticker}</strong></td>"
            f"<td>{TARGETS.get(ticker,{}).get('name','')}</td>"
            f"<td>{sh_base:.3f}</td><td>{sh_sf:.3f}</td>"
            f"<td style='color:{col_d};'>{delta:+.3f}</td>"
            f"<td>{len(rec['good_months'])}/12</td>"
            f"<td style='font-size:0.8em;'>{gm}</td></tr>"
        )
    sec_ov = (
        _desc("Vergleich: Basis-Strategie (RSI&lt;70, Lag=1, immer aktiv) vs "
              "Saisongefilterter Strategie (nur in IS-profitablen Monaten aktiv). "
              "Good Months = Monate mit positivem Ø IS-Monatsertrag.")
        + '<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Ticker</th><th>Name</th><th>Sharpe Basis OOS</th>'
        '<th>Sharpe Seasonal OOS</th><th>Δ Sharpe</th>'
        '<th>Gute Monate</th><th>Liste</th></tr></thead>'
        f'<tbody>{ov_rows}</tbody></table></div>'
    )

    # ── S1: Monthly heatmap (year × month) – first ticker = JETS ─────────────
    first_t = next(iter(strat_records))
    rec0    = strat_records[first_t]
    mp      = rec0["monthly_pl"] if "monthly_pl" in rec0 else rec0["monthly_pnl"]

    fig_cal = go.Figure(go.Heatmap(
        z=mp.values.tolist(),
        x=mp.columns.tolist(),
        y=[str(y) for y in mp.index.tolist()],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.3f}" if not np.isnan(float(v)) else "" for v in row]
              for row in mp.values],
        texttemplate="%{text}", textfont=dict(size=9, color="#000"),
        colorbar=dict(title="Monatsrendite", tickfont=dict(color="#e6edf3")),
    ))
    _lay(fig_cal, title=f"Kalender-Heatmap monatlicher Strategierenditen: {first_t}",
         height=max(300, len(mp) * 28))
    sec_cal = (
        _desc("Jede Zelle = Summe der täglichen Netto-Renditen in diesem Monat/Jahr. "
              "Grün = profitabler Monat, Rot = Verlustmonat. "
              "Horizontale Muster zeigen systematische Saisoneffekte (bestimmte Monate "
              "konsistent gut oder schlecht). Vertikale Muster = Marktregime.")
        + _htm(fig_cal)
    )

    # ── S2: Multi-ticker monthly seasonality bars ─────────────────────────────
    fig_ms = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        am = rec["avg_month"]
        ci = rec["ci_month"]
        fig_ms.add_trace(go.Bar(
            x=MONTH_NAMES, y=[am.get(m, 0)*100 for m in MONTH_NAMES],
            name=ticker, marker_color=PAL[i % len(PAL)],
            error_y=dict(
                type="data", symmetric=False,
                array    =[max(ci.get(m,(0,0))[1]-am.get(m,0),0)*100 for m in MONTH_NAMES],
                arrayminus=[max(am.get(m,0)-ci.get(m,(0,0))[0],0)*100 for m in MONTH_NAMES],
                visible=True, color="#8b949e",
            ),
            opacity=0.8,
        ))
    fig_ms.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_ms, title="Durchschnittliche Monatsrendite % (Gesamt-Periode)",
         barmode="group", xaxis_title="Monat", yaxis_title="Ø Tagesrendite×100", height=420)
    sec_ms = (
        _desc("Ø Tagesrendite aggregiert je Kalendermonat über alle Jahre. "
              "Fehlerbalken = 95%-Bootstrap-CI. "
              "Signifikant grüne Monate (CI vollständig über 0) sind die besten Handelsperioden. "
              "Systematisch rote Monate sollten in der gefilterten Strategie ausgelassen werden.")
        + _htm(fig_ms)
    )

    # ── S3: Day-of-Week effect ────────────────────────────────────────────────
    fig_dow = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        dow_v = rec["avg_dow"]
        fig_dow.add_trace(go.Bar(
            x=DOW_NAMES, y=[dow_v.get(d, 0)*100 for d in DOW_NAMES],
            name=ticker, marker_color=PAL[i % len(PAL)], opacity=0.8,
        ))
    fig_dow.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_dow, title="Wochentag-Saisonalität: Ø Tagesrendite",
         barmode="group", height=360)
    sec_dow = (
        _desc("Day-of-Week Effekt: Mittlere tägliche Nettostrategie-Rendite je Wochentag. "
              "Bekannte Anomalien: Montagseffekt (oft negativ), Freitagseffekt. "
              "Wenn systematische DoW-Muster bestehen, könnte man die Strategie "
              "nur an bestimmten Wochentagen aktiv schalten.")
        + _htm(fig_dow)
    )

    # ── S4: Quarterly box plots ───────────────────────────────────────────────
    fig_qtr = make_subplots(rows=1, cols=len(strat_records),
                             subplot_titles=list(strat_records.keys()))
    for i, (ticker, rec) in enumerate(strat_records.items()):
        n_all = pd.concat([rec["n_is"], rec["n_oos"]]).sort_index()
        n_df  = n_all.to_frame("r")
        n_df["quarter"] = pd.to_datetime(n_df.index).quarter
        for q in range(1, 5):
            vs = n_df[n_df["quarter"] == q]["r"].dropna().values * 100
            fig_qtr.add_trace(go.Box(y=vs.tolist(), name=f"Q{q}",
                                     showlegend=(i == 0)), row=1, col=i+1)
    fig_qtr.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=400, title_text="Quartals-Saisonalität: Tagesrenditen-Verteilung",
    )
    sec_qtr = (
        _desc("Box-Plot der täglichen Nettostrategie-Renditen je Quartal. "
              "Q1=Jan–Mär, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Okt–Dez. "
              "Wenn ein Quartal systematisch niedrigere Mediane hat, "
              "ist dies ein guter Kandidat für den Saisonfilter.")
        + _htm(fig_qtr)
    )

    # ── S5: Autocorrelation (cycle detection) ────────────────────────────────
    fig_acf = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        fig_acf.add_trace(go.Scatter(
            x=rec["acf_lags"], y=[v*100 for v in rec["acf_vals"]],
            name=ticker, mode="lines+markers",
            line=dict(color=PAL[i % len(PAL)], width=1.5),
            marker=dict(size=4),
        ))
    # ±2/√N bands
    n_approx = 1800
    ci_band = 2 / np.sqrt(n_approx) * 100
    fig_acf.add_hline(y=ci_band,  line_color="#e3b341", line_dash="dot",
                      annotation_text="95% CI")
    fig_acf.add_hline(y=-ci_band, line_color="#e3b341", line_dash="dot")
    fig_acf.add_hline(y=0, line_color="#8b949e", line_dash="solid")
    _lay(fig_acf, title="Autokorrelation der Strategie-Renditen (Lag 1–62 Tage)",
         xaxis_title="Lag (Tage)", yaxis_title="Autokorrelation × 100", height=400)
    sec_acf = (
        _desc("ACF(k) = Kor(r_t, r_{t-k}). Signifikante Peaks außerhalb der ±2/√N-Bänder "
              "(gelbe Linien) deuten auf zyklische Muster hin. "
              "Peaks bei Lag ≈ 21 → Monatsrhythmus. "
              "Peaks bei Lag ≈ 63 → Quartalsrhythmus. "
              "Negative Korrelation bei kleinen Lags → Mean-Reversion-Charakter.")
        + _htm(fig_acf)
    )

    # ── S6: Rolling 21-day cumulative return ─────────────────────────────────
    fig_r21 = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        r21 = rec["roll21"]
        fig_r21.add_trace(go.Scatter(
            x=r21.index.astype(str).tolist(), y=r21.values.tolist(),
            name=ticker, mode="lines",
            line=dict(color=PAL[i % len(PAL)], width=1.2),
        ))
    fig_r21.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_r21, title="Rolling 21-Tage kumulierte Rendite (%)",
         xaxis_title="Datum", yaxis_title="21T-Rendite (%)", height=430)
    sec_r21 = (
        _desc("Rollierendes 21-Tage-Fenster (≈ 1 Monat): Summe der täglichen Renditen. "
              "Zyklische Auf-und-Ab-Bewegungen mit regelmäßiger Periode sind ein starkes "
              "Indiz für Saisoneffekte. Wenn das Bild einem Sinus ähnelt, "
              "existiert ein stabiler Kreislauf, der herausgefiltert werden kann.")
        + _htm(fig_r21)
    )

    # ── S7: Seasonal-filtered vs Baseline equity curves (OOS) ─────────────────
    fig_sf = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        col = PAL[i % len(PAL)]
        # Baseline
        cum_base = (1 + rec["n_oos"]).cumprod() * 100
        fig_sf.add_trace(go.Scatter(
            x=cum_base.index.astype(str).tolist(), y=cum_base.values.tolist(),
            name=f"{ticker} Basis", mode="lines",
            line=dict(color=col, width=1.5, dash="dot"),
        ))
        # Seasonal filtered
        cum_sf = (1 + rec["n_oos_sf"]).cumprod() * 100
        fig_sf.add_trace(go.Scatter(
            x=cum_sf.index.astype(str).tolist(), y=cum_sf.values.tolist(),
            name=f"{ticker} Seasonal", mode="lines",
            line=dict(color=col, width=2.5),
        ))
    _lay(fig_sf, title="OOS Equity Curve: Basis (gepunktet) vs Seasonal-Filter (durchgezogen)",
         xaxis_title="Datum", yaxis_title="NAV (Start=100)", height=500)
    sec_sf = (
        _desc("Saisongefilterter Modus: In 'schlechten' Monaten (IS-Monatsrendite &lt; 0) "
              "wird das Signal auf 0 gesetzt (flat). TC fällt beim Eintritt/Austritt an. "
              "Ziel: Flache Phasen vermeiden ohne wertvolle Long-Phasen zu verlieren. "
              "Gepunktet = Basisstrategie, Durchgezogen = Seasonal-Filter.")
        + _htm(fig_sf)
    )

    # ── S8: Monthly return heatmap – all tickers side by side ─────────────────
    fig_mhm = make_subplots(
        rows=len(strat_records), cols=1,
        subplot_titles=[f"{t}: Jahres×Monats-Rendite" for t in strat_records],
        vertical_spacing=0.05,
    )
    for i, (ticker, rec) in enumerate(strat_records.items()):
        mp = rec["monthly_pnl"]
        fig_mhm.add_trace(go.Heatmap(
            z=mp.values.tolist(), x=mp.columns.tolist(),
            y=[str(y) for y in mp.index.tolist()],
            colorscale="RdYlGn", zmid=0,
            showscale=(i == 0),
            colorbar=dict(title="Rendite", y=1-i/len(strat_records)-0.05,
                          len=1/len(strat_records),
                          tickfont=dict(color="#e6edf3")),
        ), row=i+1, col=1)
    fig_mhm.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis","yaxis")},
        height=max(200, len(strat_records) * 220),
        title_text="Jahres×Monats Heatmap: alle Airline-Strategien",
    )
    sec_mhm = (
        _desc("Vergleich der Saisonmuster über alle analysierten Airlines. "
              "Gemeinsame rote Monate (alle Strategien gleichzeitig schlecht) "
              "deuten auf marktweite Saisoneffekte hin. "
              "Ticker-spezifische Muster können auf Unternehmens-spezifische "
              "Bilanzzyklen, Quartalsergebnisse oder regionale Effekte zurückgehen.")
        + _htm(fig_mhm)
    )

    # ── S9: Best/Worst month ranking ─────────────────────────────────────────
    # Aggregate across all tickers: average monthly return
    all_avg_m = {m: [] for m in MONTH_NAMES}
    for rec in strat_records.values():
        for m in MONTH_NAMES:
            all_avg_m[m].append(rec["avg_month"].get(m, 0.0))
    agg_m = {m: np.mean(v) for m, v in all_avg_m.items()}
    sorted_m = sorted(agg_m.items(), key=lambda x: x[1], reverse=True)

    fig_rank_m = go.Figure(go.Bar(
        x=[x[0] for x in sorted_m],
        y=[x[1]*100 for x in sorted_m],
        marker_color=["#3fb950" if x[1] > 0 else "#f78166" for x in sorted_m],
    ))
    _lay(fig_rank_m, title="Monats-Ranking: Ø Tagesrendite (alle Strategien aggregiert)",
         xaxis_title="Monat", yaxis_title="Ø Rendite (%/Tag)", height=360)
    sec_rank_m = (
        _desc("Aggregiertes Ranking über alle Airline-Strategien. "
              "Die grünen Monate links sollten bevorzugt gehandelt werden. "
              "Monatsnamen sind nach Ø-Performance absteigend sortiert. "
              "Dieses Ranking bildet die Basis des Saisonfilters.")
        + _htm(fig_rank_m)
    )

    # ── S10: Sharpe comparison table IS/OOS/Seasonal ─────────────────────────
    cmp_rows = ""
    for ticker, rec in strat_records.items():
        sh_is  = _sh(rec["n_is"])
        sh_oos = _sh(rec["n_oos"])
        sh_sf  = _sh(rec["n_oos_sf"])
        delta  = sh_sf - sh_oos if not (np.isnan(sh_sf) or np.isnan(sh_oos)) else np.nan
        mdd_b  = _mdd(rec["n_oos"]) * 100
        mdd_sf = _mdd(rec["n_oos_sf"]) * 100
        col_d  = "#3fb950" if delta > 0 else "#f78166"
        col_m  = "#3fb950" if mdd_sf > mdd_b else "#f78166"
        cmp_rows += (
            f"<tr><td><strong>{ticker}</strong></td>"
            f"<td>{sh_is:.3f}</td><td>{sh_oos:.3f}</td><td>{sh_sf:.3f}</td>"
            f"<td style='color:{col_d};'>{delta:+.3f}</td>"
            f"<td>{mdd_b:.1f}%</td>"
            f"<td style='color:{col_m};'>{mdd_sf:.1f}%</td>"
            f"<td>{', '.join(MONTH_NAMES[m-1] for m in sorted(rec['good_months']))}</td>"
            f"</tr>"
        )
    sec_cmp = (
        _desc("Vollständiger Vergleich: IS-Kalibrierung → OOS-Baseline → OOS-Seasonal-Filter. "
              "Δ Sharpe = Seasonal − Baseline. Positives Δ bedeutet Verbesserung durch Saisonfilter. "
              "MaxDD = Maximum Drawdown; kleinerer Wert (weniger negativ) = besser.")
        + '<div class="table-responsive"><table class="table table-sm table-dark table-hover">'
        '<thead><tr><th>Ticker</th><th>Sharpe IS</th><th>Sharpe OOS Basis</th>'
        '<th>Sharpe OOS Seasonal</th><th>Δ Sharpe</th>'
        '<th>MaxDD Basis</th><th>MaxDD Seasonal</th><th>Aktive Monate</th></tr></thead>'
        f'<tbody>{cmp_rows}</tbody></table></div>'
    )

    # ── S11: Rolling annual return chart (calendar year bars) ─────────────────
    fig_ann = go.Figure()
    for i, (ticker, rec) in enumerate(strat_records.items()):
        net_a = pd.concat([rec["n_is"], rec["n_oos"]]).sort_index()
        net_a.index = pd.to_datetime(net_a.index)
        ann = net_a.groupby(net_a.index.year).sum() * 100
        fig_ann.add_trace(go.Bar(
            x=[str(y) for y in ann.index.tolist()],
            y=ann.values.tolist(),
            name=ticker, marker_color=PAL[i % len(PAL)], opacity=0.8,
        ))
    fig_ann.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    _lay(fig_ann, title="Jährliche Strategie-Rendite % je Airline",
         barmode="group", xaxis_title="Jahr", yaxis_title="Jahresrendite (%)", height=420)
    sec_ann = (
        _desc("Kumulierte tägliche Strategie-Renditen je Kalenderjahr. "
              "Wenn bestimmte Jahre systematisch negativ sind, könnte das auf "
              "Öl-Markt-Regime zurückgehen (z.B. 2020 COVID, 2014-16 Ölcrash). "
              "Vergleiche mit dem VIX- und Krisenperioden-Report.")
        + _htm(fig_ann)
    )

    # ── ASSEMBLY ──────────────────────────────────────────────────────────────
    def _acc(title, body, idx, open_=False):
        sh = "show" if open_ else ""
        return (
            f'<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">'
            f'<h2 class="accordion-header">'
            f'<button class="accordion-button {"" if open_ else "collapsed"}" '
            f'style="background:#1c2128;color:#e6edf3;" '
            f'type="button" data-bs-toggle="collapse" data-bs-target="#sacc{idx}">'
            f'{title}</button></h2>'
            f'<div id="sacc{idx}" class="accordion-collapse collapse {sh}">'
            f'<div class="accordion-body" style="background:#161b22;color:#e6edf3;">{body}</div>'
            f'</div></div>'
        )

    acc = '<div class="accordion" id="seasonAcc">'
    acc += _acc("§0  Übersicht: Saisonfilter-Ergebnis aller Strategien", sec_ov, 0, open_=True)
    acc += _acc("§1  Kalender-Heatmap: Jahr × Monat (Basis-Strategie)", sec_cal, 1)
    acc += _acc("§2  Ø Monatsrendite mit 95%-CI (alle Airlines)", sec_ms, 2)
    acc += _acc("§3  Wochentag-Saisonalität", sec_dow, 3)
    acc += _acc("§4  Quartals-Saisonalität (Boxplots)", sec_qtr, 4)
    acc += _acc("§5  Autokorrelation der Strategie-Renditen (ACF 1–62T)", sec_acf, 5)
    acc += _acc("§6  Rolling 21T kumulierte Rendite (Zyklen sichtbar)", sec_r21, 6)
    acc += _acc("§7  OOS Equity Curves: Basis vs Seasonal-Filter", sec_sf, 7)
    acc += _acc("§8  Jahres×Monats-Heatmap: alle Airlines", sec_mhm, 8)
    acc += _acc("§9  Monats-Ranking (aggregiert, alle Airlines)", sec_rank_m, 9)
    acc += _acc("§10 Vollständiger IS/OOS/Seasonal Sharpe-Vergleich", sec_cmp, 10)
    acc += _acc("§11 Jährliche Renditen je Airline", sec_ann, 11)
    acc += "</div>"

    body = f"""
    <div class="container-fluid px-4 py-3">
      <div class="d-flex align-items-center mb-4">
        <div style="width:6px;height:50px;background:#e3b341;border-radius:3px;" class="me-3"></div>
        <div>
          <h2 class="mb-0" style="color:#e6edf3;">Saisonalität & Zyklizität: CL=F Lead-Lag Strategien</h2>
          <p class="mb-0" style="color:#8b949e;">
            Saisoneffekte messen · Schwache Monate herausfiltern · Saisongefilterter Backtest ·
            ACF-Zyklusanalyse · Anwendung auf {len(strat_records)} Airlines
          </p>
        </div>
      </div>
      {acc}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    _write(out / "seasonality_report.html",
           _html_base("Saisonalität & Zyklizität", 19, body))

'''

# ── inject ────────────────────────────────────────────────────────────────────
src = RB.read_text(encoding="utf-8")
MARKER = "\ndef build_index(tables, figures, out):"

if "def build_seasonality_report(" in src:
    # replace existing
    start = src.find("\ndef build_seasonality_report(")
    end   = src.find("\ndef build_", start + 10)
    src   = src[:start] + FUNC + src[end:]
    print("Replaced existing build_seasonality_report.")
else:
    pos = src.find(MARKER)
    if pos == -1:
        raise RuntimeError("Injection marker not found.")
    src = src[:pos] + FUNC + src[pos:]
    print("Injected build_seasonality_report.")

# wire build_all_reports
OLD_W = ("    build_airline_oil_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_airline_oil_report(tables, figures, reports)\n"
         "    build_seasonality_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

if "build_seasonality_report(tables" in src:
    print("build_all_reports already wired.")
elif OLD_W in src:
    src = src.replace(OLD_W, NEW_W, 1)
    print("build_all_reports wired.")
else:
    print("WARNING: could not wire build_all_reports.")

RB.write_text(src, encoding="utf-8")
print(f"Done. {len(src.splitlines())} lines")
