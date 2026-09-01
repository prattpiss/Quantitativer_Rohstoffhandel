"""Injects build_intraday_ccf_report + build_technical_analysis_report into report_builder.py"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

INTRADAY_CCF_FN = r'''
def build_intraday_ccf_report(tables, figures, out):  # noqa: C901
    """Intraday CCF at hourly resolution for daily lag=0 pairs."""
    import time as _time

    # Top pairs from daily CCF that showed lag=0 (need finer resolution)
    PAIRS_H = [
        ("GC=F",  "GDX",  "Gold → GDX"),
        ("GC=F",  "SIL",  "Gold → SIL"),
        ("SI=F",  "SIL",  "Silber → SIL"),
        ("SI=F",  "GDX",  "Silber → GDX"),
        ("BZ=F",  "XLE",  "Brent → XLE"),
        ("CL=F",  "XLE",  "WTI → XLE"),
        ("GC=F",  "NEM",  "Gold → NEM"),
        ("BZ=F",  "APA",  "Brent → APA"),
        ("CL=F",  "SM",   "WTI → SM"),
    ]
    MAX_LAG_H = 24  # ± 24 hours

    def _ccf_at_lags(x: "np.ndarray", y: "np.ndarray", max_lag: int):
        """Returns dict lag→corr for -max_lag..+max_lag."""
        n = len(x)
        xd = x - x.mean(); yd = y - y.mean()
        sx = float(np.std(xd)); sy = float(np.std(yd))
        if sx < 1e-12 or sy < 1e-12:
            return {}
        result = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xi, yi = xd[:n-lag] if lag > 0 else xd, yd[lag:] if lag > 0 else yd
            else:
                xi, yi = xd[-lag:], yd[:n+lag]
            if len(xi) < 30:
                continue
            result[lag] = float(np.corrcoef(xi, yi)[0, 1])
        return result

    try:
        import yfinance as yf
    except ImportError:
        _write(out / "intraday_ccf.html",
               _html_base("Intraday CCF", 18, "<p>yfinance nicht installiert.</p>"))
        return

    hourly_data = {}
    download_log = []
    for a1, a2, _ in PAIRS_H:
        for ticker in [a1, a2]:
            if ticker in hourly_data:
                continue
            try:
                hist = yf.Ticker(ticker).history(period="730d", interval="1h")
                if not hist.empty:
                    s = hist["Close"].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    hourly_data[ticker] = s
                    r = np.log(s / s.shift(1)).dropna()
                    hourly_data[f"{ticker}_ret"] = r
                    download_log.append(f"{ticker}: {len(s)} Stunden-Kerzen OK")
                else:
                    download_log.append(f"{ticker}: leer")
            except Exception as e:
                download_log.append(f"{ticker}: FEHLER {e}")
            _time.sleep(0.3)  # rate limit

    # ── Compute hourly CCF for each pair ──────────────────────────────────
    pair_results = []
    fig_ccf_all = go.Figure()
    for pi, (a1, a2, label) in enumerate(PAIRS_H):
        r1_key = f"{a1}_ret"; r2_key = f"{a2}_ret"
        if r1_key not in hourly_data or r2_key not in hourly_data:
            continue
        r1 = hourly_data[r1_key]; r2 = hourly_data[r2_key]
        idx = r1.index.intersection(r2.index)
        if len(idx) < 100:
            continue
        ccf_d = _ccf_at_lags(r1.loc[idx].values, r2.loc[idx].values, MAX_LAG_H)
        if not ccf_d:
            continue
        lags = sorted(ccf_d.keys())
        vals = [ccf_d[l] for l in lags]
        # find optimal lag
        best_lag = max(ccf_d, key=lambda l: abs(ccf_d[l]))
        best_rho = ccf_d[best_lag]
        sig_band = 2.0 / np.sqrt(len(idx))
        pair_results.append({
            "Paar": label,
            "A1": a1, "A2": a2,
            "Opt. Lag (h)": best_lag,
            "Peak CCF": round(best_rho, 4),
            "Signifikant": abs(best_rho) > sig_band,
            "Sig.-Band (±)": round(sig_band, 4),
            "CCF@Lag0": round(ccf_d.get(0, float("nan")), 4),
            "Δ Peak vs. Lag0": round(abs(best_rho) - abs(ccf_d.get(0, 0)), 4),
            "N Stunden": len(idx),
        })
        fig_ccf_all.add_trace(go.Scatter(
            x=lags, y=vals,
            mode="lines+markers", name=label,
            line=dict(color=PAL[pi % len(PAL)], width=1.5),
            marker=dict(size=4)))

    fig_ccf_all.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_ccf_all.update_layout(
        title=f"Intraday CCF (stündlich): ±{MAX_LAG_H}h — Alle Paare",
        xaxis_title="Lag (Stunden, negativ = A1 verzögert A2)",
        yaxis_title="Kreuzkorrelation", height=520,
        shapes=[
            dict(type="rect", x0=-MAX_LAG_H, x1=MAX_LAG_H,
                 y0=-2/np.sqrt(500), y1=2/np.sqrt(500),
                 fillcolor="rgba(139,148,158,0.1)", line=dict(width=0))
        ])

    # Individual CCF charts per pair
    indiv_htmls = ""
    for pi, (a1, a2, label) in enumerate(PAIRS_H):
        r1_key = f"{a1}_ret"; r2_key = f"{a2}_ret"
        if r1_key not in hourly_data or r2_key not in hourly_data:
            continue
        r1 = hourly_data[r1_key]; r2 = hourly_data[r2_key]
        idx = r1.index.intersection(r2.index)
        if len(idx) < 100:
            continue
        ccf_d = _ccf_at_lags(r1.loc[idx].values, r2.loc[idx].values, MAX_LAG_H)
        lags = sorted(ccf_d.keys())
        vals = [ccf_d[l] for l in lags]
        sig_band = 2.0 / np.sqrt(len(idx))
        fig_ind = go.Figure()
        colors = ["#3fb950" if l == max(ccf_d, key=lambda x: abs(ccf_d[x]))
                  else ("#58a6ff" if l == 0 else "#8b949e")
                  for l in lags]
        fig_ind.add_trace(go.Bar(x=lags, y=vals, marker_color=colors,
                                  hovertemplate="Lag=%{x}h<br>CCF=%{y:.4f}<extra></extra>"))
        fig_ind.add_hline(y=sig_band, line_color="#d29922", line_dash="dot",
                           annotation_text=f"+{sig_band:.3f} (sig)")
        fig_ind.add_hline(y=-sig_band, line_color="#d29922", line_dash="dot",
                           annotation_text=f"-{sig_band:.3f}")
        fig_ind.add_hline(y=0, line_color="#8b949e")
        best_lag = max(ccf_d, key=lambda x: abs(ccf_d[x]))
        best_rho = ccf_d[best_lag]
        rho0 = ccf_d.get(0, 0)
        fig_ind.update_layout(
            title=f"{label} — Opt. Lag: {best_lag:+d}h | Peak CCF: {best_rho:.4f} | Lag-0: {rho0:.4f}",
            xaxis_title="Lag (Stunden)", yaxis_title="CCF", height=360)
        lag_interp = (f"Optimaler Lag: {best_lag:+d}h. " +
                      ("Gleichzeitig!" if best_lag == 0 else
                       f"{'A1 führt A2' if best_lag > 0 else 'A2 führt A1'} um {abs(best_lag)} Stunde(n). ") +
                      f"Peak {best_rho:.4f} vs. Lag-0 {rho0:.4f}: " +
                      ("kein signifikanter Zeitversatz trotz stündlicher Auflösung."
                       if best_lag == 0 else
                       f"sub-tägiger Lead-Lag von {abs(best_lag)}h gefunden!"))
        indiv_htmls += _chart_card(f"Stündlicher CCF: {label}", fig_ind, height=380,
                                   interp=lag_interp)

    # Results table
    result_df = pd.DataFrame(pair_results).sort_values("Δ Peak vs. Lag0", ascending=False) if pair_results else pd.DataFrame()
    result_html = _df_html(result_df) if not result_df.empty else "<p class='text-muted'>Keine Ergebnisse.</p>"

    # Summary: are daily lag=0 pairs truly synchronous?
    sync_pairs = [r for r in pair_results if r["Opt. Lag (h)"] == 0]
    lagged_pairs = [r for r in pair_results if r["Opt. Lag (h)"] != 0]

    body = f"""
<div class="ph-header">
  <h1>Intraday CCF: Stündliche Lag-Auflösung</h1>
  <div class="sub">Für daily lag=0 Paare: Stundendaten (max 730 Tage) &middot; ±{MAX_LAG_H}h &middot; Sub-tägliger Lead-Lag</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Motivation: Warum Intraday-CCF?</strong></div>
  <div class="card-body">
    <p class="small">
      Bei täglicher Auflösung zeigen viele Paare optimalen Lag = 0 (gleichzeitig). Das bedeutet <em>nicht</em>
      zwingend, dass keine Lead-Lag-Beziehung existiert — es könnte ein <strong>sub-tägiger Zeitversatz</strong>
      (z.B. 30 Minuten oder 2 Stunden) vorliegen, der im Tagesdurchschnitt verschwimmt.
    </p>
    <div class="row">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Befund</th><th>Interpretation</th><th>Konsequenz</th></tr></thead>
          <tbody>
            <tr><td>Opt. Lag = 0h</td><td>Wirklich synchron auf Stundenebene</td><td>Kein handelbarer Lead-Lag → nur simultane Exposure-Hedges</td></tr>
            <tr><td>Opt. Lag ≠ 0h</td><td>Sub-tägiger Lead-Lag gefunden</td><td>Intraday-Strategie möglich: Futures führen, ETF folgt</td></tr>
            <tr><td>|Opt. Lag| > 6h</td><td>Handelszeit-Differenz (US vs. London)</td><td>Market-Microstructure: Öffnungszeiten-Effekt</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        <p class="small text-muted">
          Typische Hypothesen für Commodity/ETF-Paare:<br>
          • <strong>GC=F → GDX</strong>: Futures (24h) führen ETF (US-Handelstag) → 1-4h Lag erwartet<br>
          • <strong>CL=F → XLE</strong>: Ähnlich — WTI Futures führen Energie-ETF<br>
          • <strong>SI=F → SIL</strong>: Silber-Futures vs. Silber-Miner-ETF<br>
          Minuten-Daten (yfinance 1m, max 7 Tage): für noch feinere Auflösung.
        </p>
      </div>
    </div>
    {_info(f"Download-Log: " + " | ".join(download_log[:15]))}
    {_info(f"Ergebnis: {len(sync_pairs)} Paare wirklich synchron (Lag=0h), "
           f"{len(lagged_pairs)} Paare mit sub-tägigem Lead-Lag ≥1h.")}
  </div>
</div>

{_chart_card(f"Intraday CCF: Alle Paare gleichzeitig (±{MAX_LAG_H}h)", fig_ccf_all, height=540,
    interp="Negativer Lag = A1 verzögert A2 (A2 führt). Positiver Lag = A1 führt A2. "
           "Grauer Bereich = 95%-Signifikanzband. Peaks außerhalb: statistisch signifikanter Lead-Lag. "
           "Grüne Markierung = optimaler Lag. Blaue Markierung = Lag 0.")}

{indiv_htmls}

<div class="card mb-4">
  <div class="card-header"><strong>Zusammenfassung: Hourly Lead-Lag-Tabelle</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Δ Peak vs. Lag0 = Wie viel zusätzliche CCF-Stärke beim optimalen Lag gegenüber Lag=0.
      Hoher Δ = klarer sub-tägiger Lead-Lag. Tiefer Δ = wirklich synchron.
    </p>
    {result_html}
  </div>
</div>
"""
    _write(out / "intraday_ccf.html", _html_base("Intraday CCF", 18, body))

'''

TECHNICAL_ANALYSIS_FN = r'''
def build_technical_analysis_report(tables, figures, out):  # noqa: C901
    """Technical indicators per asset + lead-lag indicator strategies (progressive complexity)."""
    returns = _read(tables / "phase2_returns.csv")
    prices  = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "technical_analysis.html",
               _html_base("Technische Analyse", 18, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        # Reconstruct prices from returns
        prices = np.exp(returns.cumsum()) * 100

    # ── Technical indicator functions ─────────────────────────────────────
    def _sma(s, w):    return s.rolling(w, min_periods=w//2).mean()
    def _ema(s, w):    return s.ewm(span=w, adjust=False).mean()
    def _atr(h, l, c, w=14):
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(w).mean()

    def _rsi(s, w=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(w).mean()
        ls = (-d.clip(upper=0)).rolling(w).mean()
        rs = g / (ls + 1e-9)
        return 100 - 100 / (1 + rs)

    def _macd(s, fast=12, slow=26, sig=9):
        ema_f = _ema(s, fast); ema_s = _ema(s, slow)
        macd_line = ema_f - ema_s
        signal    = _ema(macd_line, sig)
        return macd_line, signal

    def _bb(s, w=20, n_std=2):
        mid = _sma(s, w)
        std = s.rolling(w).std()
        return mid + n_std*std, mid, mid - n_std*std  # upper, mid, lower

    def _support_resistance(s, w=63):
        return s.rolling(w).min(), s.rolling(w).max()

    def _bb_position(s, w=20, n_std=2):
        """Returns position within Bollinger Bands: 0=lower, 0.5=mid, 1=upper."""
        up, mid, lo = _bb(s, w, n_std)
        return (s - lo) / (up - lo + 1e-9)

    # ── Assets & Granger lead-lag pairs ──────────────────────────────────
    MAIN_ASSETS = [c for c in ["CL=F","GC=F","HG=F","ZW=F","NG=F","SI=F","ZS=F","BZ=F"]
                   if c in returns.columns]

    # Confirmed Granger/CCF lead-lag pairs (leader → follower, lag≥1)
    LEADLAG_PAIRS = []
    if granger is not None and "cause" in granger.columns:
        sig_gran = (granger[granger.get("significant", True) == True]
                    .sort_values("fstat" if "fstat" in granger.columns else "pvalue",
                                 ascending="pvalue" in granger.columns)
                    .drop_duplicates(subset=["cause","effect"], keep="first")
                    .head(15))
        for _, row in sig_gran.iterrows():
            cause = row["cause"]; effect = row["effect"]
            lag   = int(row.get("lag", 1))
            if cause in returns.columns and effect in returns.columns:
                LEADLAG_PAIRS.append((cause, effect, lag, f"{cause}→{effect} (Lag {lag}T)"))
    # Add manual pairs if granger table empty
    manual = [("GC=F","GDX",1,"Gold→GDX"), ("CL=F","XLE",1,"WTI→XLE"),
              ("GC=F","NEM",1,"Gold→NEM"), ("CL=F","XOM",1,"WTI→XOM"),
              ("HG=F","FCX",1,"Kupfer→FCX")]
    for c, e, l, lb in manual:
        if c in returns.columns and e in returns.columns:
            if not any(p[0]==c and p[1]==e for p in LEADLAG_PAIRS):
                LEADLAG_PAIRS.append((c, e, l, lb))

    # ── LEVEL 1: Price + SMA per asset ───────────────────────────────────
    fig_l1_list = {}
    for asset in MAIN_ASSETS[:6]:
        p = prices[asset].dropna() if asset in prices.columns else None
        if p is None or len(p) < 252:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index.astype(str).tolist(),
                                  y=p.round(4).values.tolist(),
                                  mode="lines", name=asset,
                                  line=dict(color="#e6edf3", width=1.3)))
        for w, col in [(20,"#58a6ff"), (50,"#d29922"), (200,"#3fb950")]:
            sma = _sma(p, w).dropna()
            fig.add_trace(go.Scatter(x=sma.index.astype(str).tolist(),
                                     y=sma.round(4).values.tolist(),
                                     mode="lines", name=f"SMA{w}",
                                     line=dict(color=col, width=1.0, dash="dot")))
        # Support & Resistance (63T)
        sup, res = _support_resistance(p, 63)
        fig.add_trace(go.Scatter(x=res.dropna().index.astype(str).tolist(),
                                  y=res.dropna().round(4).values.tolist(),
                                  mode="lines", name="Res-63T",
                                  line=dict(color="#f78166", width=0.8, dash="dash")))
        fig.add_trace(go.Scatter(x=sup.dropna().index.astype(str).tolist(),
                                  y=sup.dropna().round(4).values.tolist(),
                                  mode="lines", name="Sup-63T",
                                  line=dict(color="#3fb950", width=0.8, dash="dash"),
                                  fill="tonexty", fillcolor="rgba(63,185,80,0.04)"))
        fig.update_layout(title=f"{asset}: Preis + SMA20/50/200 + Support/Resistance (63T)",
                          height=440, yaxis_title="Preis")
        fig_l1_list[asset] = fig

    # ── LEVEL 2: RSI + MACD per asset ─────────────────────────────────────
    fig_l2_list = {}
    for asset in MAIN_ASSETS[:6]:
        p = prices[asset].dropna() if asset in prices.columns else None
        if p is None or len(p) < 252:
            continue
        rsi = _rsi(p)
        macd_line, macd_sig = _macd(p)
        bb_up, bb_mid, bb_lo = _bb(p)
        bb_pos = _bb_position(p)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Preis + Bollinger", "RSI (14)", "MACD (12/26/9)"],
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
        # Price + BB
        fig.add_trace(go.Scatter(x=p.index.astype(str).tolist(), y=p.round(4).values.tolist(),
                                  name=asset, line=dict(color="#e6edf3", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_up.dropna().index.astype(str).tolist(),
                                  y=bb_up.dropna().round(4).values.tolist(),
                                  name="BB-Oben", line=dict(color="#58a6ff", width=0.7, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_lo.dropna().index.astype(str).tolist(),
                                  y=bb_lo.dropna().round(4).values.tolist(),
                                  name="BB-Unten", line=dict(color="#58a6ff", width=0.7, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(88,166,255,0.07)"), row=1, col=1)
        # RSI
        rsi_clean = rsi.dropna()
        fig.add_trace(go.Scatter(x=rsi_clean.index.astype(str).tolist(),
                                  y=rsi_clean.round(2).values.tolist(),
                                  name="RSI", line=dict(color="#d29922", width=1.3)), row=2, col=1)
        fig.add_hline(y=70, line_color="#f78166", line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="#3fb950", line_dash="dot", row=2, col=1)
        fig.add_hline(y=50, line_color="#8b949e", line_dash="dash", row=2, col=1)
        # MACD
        mc_clean = macd_line.dropna(); ms_clean = macd_sig.loc[mc_clean.index]
        fig.add_trace(go.Scatter(x=mc_clean.index.astype(str).tolist(),
                                  y=mc_clean.round(4).values.tolist(),
                                  name="MACD", line=dict(color="#58a6ff", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ms_clean.index.astype(str).tolist(),
                                  y=ms_clean.round(4).values.tolist(),
                                  name="Signal", line=dict(color="#ffa657", width=1.0, dash="dot")), row=3, col=1)
        hist_vals = (mc_clean - ms_clean)
        fig.add_trace(go.Bar(x=hist_vals.index.astype(str).tolist(),
                              y=hist_vals.round(4).values.tolist(),
                              name="Histogram",
                              marker_color=["#3fb950" if v > 0 else "#f78166" for v in hist_vals.values]),
                      row=3, col=1)
        fig.update_layout(title=f"{asset}: Bollinger Bands + RSI + MACD",
                          height=680, showlegend=True)
        fig_l2_list[asset] = fig

    # ── LEVEL 3: Lead-Lag Indicator Strategies ──────────────────────────
    # Key insight: leader's indicator at t-lag is a FUTURE indicator for the follower.
    # Strategy: trade follower based on leader's RSI/MACD/BB-position at t-lag.
    strat_results_ta = []
    fig_ll_indicator_list = {}

    def _run_indicator_strat(leader_indicator: "pd.Series", follower_ret: "pd.Series",
                              lag: int, name: str, threshold: float = 0.0) -> dict | None:
        """Trade follower when leader_indicator(t-lag) crosses threshold."""
        li = leader_indicator.shift(lag).dropna()
        idx = li.index.intersection(follower_ret.dropna().index)
        if len(idx) < 252:
            return None
        sig = np.sign(li.loc[idx] - threshold)
        ret = follower_ret.loc[idx]
        strat_r = sig * ret - sig.diff().abs().fillna(0) * (10/10000)
        strat_r = strat_r.dropna()
        if len(strat_r) < 126:
            return None
        ann_r = float(strat_r.mean() * 252)
        ann_v = float(strat_r.std() * np.sqrt(252)) + 1e-9
        sh = ann_r / ann_v
        cum = (1 + strat_r).cumprod()
        mdd = float((cum / cum.cummax() - 1).min())
        split = int(len(strat_r) * 0.7)
        oos_sh = float(strat_r.iloc[split:].mean() * 252 /
                       (strat_r.iloc[split:].std() * np.sqrt(252) + 1e-9))
        return {
            "Strategie": name, "CAGR%": round(ann_r*100,2), "Sharpe": round(sh,3),
            "OOS Sharpe": round(oos_sh,3), "MaxDD%": round(mdd*100,2),
            "N": len(strat_r), "_ret": strat_r,
        }

    for leader, follower, lag, pair_label in LEADLAG_PAIRS[:8]:
        p_lead = prices[leader].dropna() if leader in prices.columns else None
        r_foll = returns[follower].dropna()
        if p_lead is None or len(p_lead) < 300:
            continue

        # Compute leader indicators
        rsi_lead   = _rsi(p_lead)
        macd_l, _  = _macd(p_lead)
        bb_pos_l   = _bb_position(p_lead)
        sma20_l    = _sma(p_lead, 20)
        sma_cross  = (sma20_l - _sma(p_lead, 50))  # positive = SMA20 > SMA50 (trend)

        strategies_for_pair = [
            (rsi_lead,  50.0, f"RSI(14)>50 @ {pair_label}"),
            (rsi_lead,  70.0, f"RSI(14)<70 @ {pair_label} (nicht überkauft)"),
            (macd_l,    0.0,  f"MACD>0 @ {pair_label}"),
            (bb_pos_l,  0.5,  f"BB-Position>50% @ {pair_label}"),
            (sma_cross, 0.0,  f"SMA20>SMA50 @ {pair_label}"),
        ]

        pair_strats = []
        for indicator, thresh, sname in strategies_for_pair:
            m = _run_indicator_strat(indicator, r_foll, lag, sname, thresh)
            if m:
                pair_strats.append(m)
                strat_results_ta.append(m)

        if not pair_strats:
            continue

        # Chart: equity curves for this pair's indicator strategies
        fig_ll = go.Figure()
        for j, m in enumerate(pair_strats):
            eq = (1 + m["_ret"]).cumprod()
            fig_ll.add_trace(go.Scatter(
                x=eq.index.astype(str).tolist(), y=eq.round(4).values.tolist(),
                mode="lines", name=m["Strategie"].split("@")[0].strip(),
                line=dict(color=PAL[j % len(PAL)], width=1.5)))
        bh_eq = (1 + r_foll.dropna()).cumprod()
        fig_ll.add_trace(go.Scatter(
            x=bh_eq.index.astype(str).tolist(), y=bh_eq.round(4).values.tolist(),
            mode="lines", name=f"B&H {follower}",
            line=dict(color="#8b949e", width=0.9, dash="dot")))
        fig_ll.update_layout(title=f"Lead-Lag-Indikatoren: {pair_label} — Follower: {follower}",
                              yaxis_type="log", height=440)
        fig_ll_indicator_list[pair_label] = (fig_ll, pair_strats)

    # ── LEVEL 4: Leader indicator overlaid on follower price ────────────
    fig_overlay_list = {}
    for leader, follower, lag, pair_label in LEADLAG_PAIRS[:6]:
        p_lead = prices[leader].dropna() if leader in prices.columns else None
        p_foll = prices[follower].dropna() if follower in prices.columns else None
        if p_lead is None or p_foll is None:
            continue
        rsi_lead = _rsi(p_lead)
        # Align indices
        idx = p_foll.index.intersection(rsi_lead.dropna().index)
        if len(idx) < 126:
            continue
        fig_ov = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=[f"{follower}: Preis",
                                                f"{leader}: RSI(14) — {lag}T vor {follower}"],
                                row_heights=[0.65, 0.35], vertical_spacing=0.05)
        pf = p_foll.loc[idx]
        fig_ov.add_trace(go.Scatter(x=pf.index.astype(str).tolist(),
                                     y=pf.round(4).values.tolist(),
                                     name=follower, line=dict(color="#e6edf3", width=1.3)), row=1, col=1)
        bb_up, bb_mid, bb_lo = _bb(pf)
        for bb_s, bb_col in [(bb_up,"#58a6ff"), (bb_lo,"#58a6ff")]:
            bb_c = bb_s.dropna()
            fig_ov.add_trace(go.Scatter(x=bb_c.index.astype(str).tolist(),
                                         y=bb_c.round(4).values.tolist(),
                                         name="BB", line=dict(color=bb_col, width=0.7, dash="dot"),
                                         showlegend=False), row=1, col=1)
        rl = rsi_lead.loc[idx]
        fig_ov.add_trace(go.Scatter(x=rl.index.astype(str).tolist(),
                                     y=rl.round(2).values.tolist(),
                                     name=f"RSI {leader}", line=dict(color="#d29922", width=1.5)),
                          row=2, col=1)
        fig_ov.add_hline(y=70, line_color="#f78166", line_dash="dot", row=2, col=1)
        fig_ov.add_hline(y=30, line_color="#3fb950", line_dash="dot", row=2, col=1)
        fig_ov.update_layout(
            title=f"{leader} RSI als Zukunfts-Indikator für {follower} (Lag {lag}T)",
            height=560)
        fig_overlay_list[pair_label] = fig_ov

    # ── Summary: best TA-strategies ───────────────────────────────────────
    fig_ta_summary = go.Figure()
    ta_rows = [{"Strategie": m["Strategie"], "Sharpe": m["Sharpe"],
                "OOS Sharpe": m["OOS Sharpe"], "CAGR%": m["CAGR%"],
                "MaxDD%": m["MaxDD%"]}
               for m in sorted(strat_results_ta, key=lambda x: -x["Sharpe"])]
    if ta_rows:
        ta_df = pd.DataFrame(ta_rows).head(20)
        fig_ta_summary.add_trace(go.Bar(
            x=ta_df["Strategie"].tolist(),
            y=ta_df["Sharpe"].tolist(),
            marker_color=["#3fb950" if v > 0.4 else "#d29922" if v > 0 else "#f78166"
                          for v in ta_df["Sharpe"]],
            text=[f"{v:.2f}" for v in ta_df["Sharpe"]],
            textposition="outside"))
        fig_ta_summary.add_hline(y=0, line_color="#8b949e")
        fig_ta_summary.update_layout(
            title="Lead-Lag-Indikator-Strategien: Top-20 Sharpe Ratio",
            xaxis_tickangle=-40, height=520, yaxis_title="Sharpe")

    ta_table_html = (_df_html(pd.DataFrame(ta_rows[:30])) if ta_rows
                     else "<p class='text-muted'>Keine Ergebnisse.</p>")

    # ── HTML body ──────────────────────────────────────────────────────────
    l1_html = ""
    for asset, fig in fig_l1_list.items():
        l1_html += _chart_card(f"Level 1 — {asset}: Preis + SMA + Support/Resistance", fig, height=460,
            interp=f"SMA20 (blau): kurzfristiger Trend. SMA50 (gelb): mittelfristig. SMA200 (grün): langfristig. "
                   f"Golden Cross: SMA20 > SMA200. Death Cross: SMA20 < SMA200. "
                   f"Support (grün): 63T-Tief = starker Boden. Resistance (rot): 63T-Hoch.")

    l2_html = ""
    for asset, fig in fig_l2_list.items():
        l2_html += _chart_card(f"Level 2 — {asset}: RSI + MACD + Bollinger", fig, height=700,
            interp=f"RSI>70: überkauft → Rückschlagrisiko. RSI<30: überverkauft → Erholungspotenzial. "
                   f"MACD-Crossover: bullisches Signal (MACD > Signal). "
                   f"BB-Ausbruch oben: Momentum oder Übertreibung. BB-Ausbruch unten: Panic oder Breakout.")

    ll_html = ""
    for pair_label, (fig_ll, pair_strats) in fig_ll_indicator_list.items():
        best = max(pair_strats, key=lambda x: x["Sharpe"])
        strat_rows = "".join(
            f"<tr><td class='small'>{m['Strategie'].split('@')[0].strip()}</td>"
            f"<td>{m['Sharpe']:.3f}</td><td>{m['OOS Sharpe']:.3f}</td>"
            f"<td>{m['CAGR%']:.1f}%</td><td>{m['MaxDD%']:.1f}%</td></tr>"
            for m in pair_strats)
        strat_table = (f"<table class='table table-dark table-sm table-bordered small'>"
                       f"<thead><tr><th>Indikator</th><th>Sharpe</th><th>OOS</th>"
                       f"<th>CAGR</th><th>MaxDD</th></tr></thead>"
                       f"<tbody>{strat_rows}</tbody></table>")
        ll_html += f"""
<div class="card mb-3">
  <div class="card-header"><strong>Level 3 — Lead-Lag-Indikator: {pair_label}</strong>
    <span class="badge bg-success ms-2">Best: {best['Sharpe']:.2f} Sharpe</span></div>
  <div class="card-body">
    {strat_table}
    {_interp("Strategie: Wenn Anführer-Indikator Signal gibt, trade den Nachzügler nach [Lag] Tagen. "
             "Leader-RSI>50 = bullisches Markt-Momentum → Long Follower. "
             "OOS Sharpe zeigt, ob das echte Vorhersagekraft ist oder Overfitting.")}
  </div>
</div>
""" + _chart_card(f"Level 3 Equity-Kurven: {pair_label}", fig_ll, height=460,
                   interp="Log-Skala. Leader-Indikator-Signal mit [Lag]-Tages-Verzögerung. "
                          "Beste Indikatoren für Nachzügler-Trading: RSI und MACD typisch am stärksten.")

    ov_html = ""
    for pair_label, fig_ov in fig_overlay_list.items():
        ov_html += _chart_card(f"Level 4 — Overlay: {pair_label}", fig_ov, height=580,
            interp=f"Oben: Follower-Preis mit Bollinger Bands. "
                   f"Unten: Leader RSI — dieser RSI war {1}T vor dem Follower-Kurs. "
                   f"Leader RSI > 70 UND Follower noch nicht ralliert: mögliches Entry-Signal. "
                   f"RSI-Divergenz (Leader fällt, Follower noch hoch): Warnsignal für Follower.")

    body = f"""
<div class="ph-header">
  <h1>Technische Analyse</h1>
  <div class="sub">Level 1: Preis+SMA+Support · Level 2: RSI+MACD+Bollinger ·
    Level 3: Lead-Lag-Indikator-Strategien · Level 4: Cross-Asset-Overlay</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>🎯 Kernidee: Anführer-Indikatoren als Zukunftssignale für Nachzügler</strong></div>
  <div class="card-body">
    <p>Wenn Asset A Asset B um k Tage zeitlich führt (Granger-kausal), dann sind <strong>technische Indikatoren
    von A zum Zeitpunkt t</strong> effektiv <strong>Zukunftsindikatoren für B zum Zeitpunkt t+k</strong>.</p>
    <div class="row">
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Level</th><th>Inhalt</th><th>Komplexität</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>Preis + SMA20/50/200 + Support/Resistance</td><td>Basis</td></tr>
            <tr><td>2</td><td>Bollinger + RSI(14) + MACD(12/26/9)</td><td>Standard TA</td></tr>
            <tr><td>3</td><td>Leader-Indikator → Follower-Trade (Lag)</td><td>Lead-Lag</td></tr>
            <tr><td>4</td><td>Cross-Asset-Overlay: Leader-RSI auf Follower-Chart</td><td>Visuell</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-6">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Strategie-Logik</th><th>Signal</th><th>Trade</th></tr></thead>
          <tbody>
            <tr><td>Leader RSI > 50</td><td>Bullisches Momentum im Leader</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader RSI < 30</td><td>Leader überverkauft → Bounce</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader MACD > 0</td><td>Leader in Aufwärtstrend</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader BB-Pos > 50%</td><td>Leader über Mittelband</td><td>Long Follower in t+k</td></tr>
            <tr><td>Leader SMA20 > SMA50</td><td>Golden Cross beim Leader</td><td>Long Follower in t+k</td></tr>
          </tbody>
        </table>
        {_info("Alle Strategien: TC=10bps. Signal wird um [Lag] Tage verschoben. "
               "IS/OOS Split 70%/30%.")}
      </div>
    </div>
  </div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 1: Preis + SMA + Support/Resistance (alle Assets)</strong></div>
  <div class="card-body">{l1_html}</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 2: RSI + MACD + Bollinger Bands (alle Assets)</strong></div>
  <div class="card-body">{l2_html}</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>Level 3: Lead-Lag-Indikator-Strategien</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Für jedes signifikante Granger-Paar: Leader-Indikatoren (RSI, MACD, BB-Position, SMA-Cross)
      als zeitverzögertes Signal für den Follower. Optimaler Lag = bekannter Granger-Lag.
    </p>
    {ll_html}
  </div>
</div>

{_chart_card("Zusammenfassung: Beste Lead-Lag-Indikator-Strategien (Top-20 Sharpe)", fig_ta_summary, height=540,
    interp="Grün > 0.4: robuste Strategie mit handelbarer Vorhersagekraft. "
           "OOS-Sharpe in Tabelle prüfen: IS-Sharpe kann durch Backtest-Bias erhöht sein. "
           "Beste Signale: RSI-basiert (trend-following) und MACD (Momentum-Wendepunkt).")}

{_card("TA-Strategien Tabelle (alle, sortiert nach Sharpe)", ta_table_html)}

<div class="card mb-4">
  <div class="card-header"><strong>Level 4: Cross-Asset-Overlay (Leader RSI auf Follower Chart)</strong></div>
  <div class="card-body">{ov_html}</div>
</div>
"""
    _write(out / "technical_analysis.html",
           _html_base("Technische Analyse", 18, body))

'''

# ──────────────────────────────────────────────────────────────────────────────
with open(RB, "r", encoding="utf-8") as f:
    src = f.read()

# Find build_index (where to insert before)
INSERT_BEFORE = "\ndef build_index(tables, figures, out):"
idx = src.find(INSERT_BEFORE)
if idx < 0:
    raise RuntimeError("build_index not found")

new_src = src[:idx] + "\n" + INTRADAY_CCF_FN + "\n" + TECHNICAL_ANALYSIS_FN + src[idx:]
with open(RB, "w", encoding="utf-8") as f:
    f.write(new_src)
print(f"Done. New size: {len(new_src.splitlines())} lines")
