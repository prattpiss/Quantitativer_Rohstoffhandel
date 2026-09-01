"""Injects build_strategy_stress_test_report into report_builder.py."""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

STRESS_TEST_FN = '''
def build_strategy_stress_test_report(tables, figures, out):  # noqa: C901
    """Deep-dive stress-test for the best lead-lag strategies."""
    returns = _read(tables / "phase2_returns.csv")
    prices  = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "strategy_stress_test.html",
               _html_base("Stress-Test", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None
    dxy = returns["DX-Y.NYB"].dropna() if "DX-Y.NYB" in returns.columns else None

    # VIX levels from prices or reconstructed
    vix_lvl = None
    if "^VIX" in prices.columns:
        vix_lvl = prices["^VIX"].dropna()
    elif "^VIX" in returns.columns:
        vix_lvl = np.exp(returns["^VIX"].cumsum()) * 20

    # ── Pair + indicator universe ─────────────────────────────────────────────
    PAIRS_TO_TEST = [
        ("CL=F", "JETS", 1, "WTI→JETS"),
        ("BZ=F", "JETS", 1, "Brent→JETS"),
        ("CL=F", "XLE",  1, "WTI→XLE"),
        ("GC=F", "GDX",  7, "Gold→GDX"),
        ("GC=F", "NEM",  6, "Gold→NEM"),
        ("HG=F", "FCX",  1, "Kupfer→FCX"),
        ("SI=F", "SIL",  1, "Silber→SIL"),
        ("CL=F", "XOM",  1, "WTI→XOM"),
    ]
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        existing = {(p[0], p[1]) for p in PAIRS_TO_TEST}
        for _, row in sdf.head(10).iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            if ((c, e) not in existing
                    and c in returns.columns and e in returns.columns):
                PAIRS_TO_TEST.append((c, e, lg, f"{c}\\u2192{e} (Lag {lg}T)"))
                existing.add((c, e))

    INDICATORS = [
        ("RSI(14)>50",   lambda px: _calc_rsi(px, 14),        50.0),
        ("MACD>0",       lambda px: _calc_macd(px)[0],         0.0),
        ("BB-Pos>0.5",   lambda px: _calc_bb_pos(px, 20),      0.5),
        ("SMA20>SMA50",  lambda px: _calc_sma_cross(px, 20, 50), 0.0),
        ("RSI(14)<70",   lambda px: -_calc_rsi(px, 14),       -70.0),
    ]

    IS_FRAC = 0.70
    all_strats = []
    for leader, follower, lag, pair_label in PAIRS_TO_TEST:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna() if follower in returns.columns else None
        if px is None or rf is None or len(px) < 300:
            continue
        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 300:
            continue
        split_date = idx_all[int(len(idx_all) * IS_FRAC)]
        is_rf = rf.loc[:split_date]; oos_rf = rf.loc[split_date:]

        for ind_name, ind_fn, thresh in INDICATORS:
            ind = ind_fn(px)
            n_is, g_is, s_is   = _strat_exec(ind, thresh, is_rf, lag)
            n_oos, g_oos, s_oos = _strat_exec(ind, thresh, oos_rf, lag)
            if len(n_is) < 50 or len(n_oos) < 50:
                continue
            sh_is  = n_is.mean()  * 252 / (n_is.std()  * np.sqrt(252) + 1e-9)
            sh_oos = n_oos.mean() * 252 / (n_oos.std() * np.sqrt(252) + 1e-9)
            all_strats.append({
                "leader": leader, "follower": follower, "lag": lag,
                "pair_label": pair_label, "ind_name": ind_name,
                "thresh": thresh, "ind_fn": ind_fn,
                "sh_is": sh_is, "sh_oos": sh_oos,
                "n_is": n_is, "g_is": g_is, "s_is": s_is,
                "n_oos": n_oos, "g_oos": g_oos, "s_oos": s_oos,
                "split_date": split_date, "idx_all": idx_all,
                "px": px, "rf": rf,
                "is_start": idx_all[0], "oos_end": idx_all[-1],
                "is_n": len(n_is), "oos_n": len(n_oos),
            })

    if not all_strats:
        _write(out / "strategy_stress_test.html",
               _html_base("Stress-Test", 19, "<p>Keine Strategien gefunden.</p>"))
        return

    # Sort: OOS > IS first, then by OOS Sharpe descending
    all_strats.sort(
        key=lambda s: (int(s["sh_oos"] > s["sh_is"]) * 1000 + s["sh_oos"]),
        reverse=True)
    focus = all_strats[0]

    # ── ADF helper (for cointegration) ────────────────────────────────────────
    def _adf_stat(series):
        y = series.dropna().values
        if len(y) < 20:
            return float("nan")
        dy = np.diff(y); n = len(dy)
        lag_dy = np.concatenate([[0.0], dy[:-1]])
        X = np.column_stack([y[1:], np.ones(n), lag_dy])
        coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        yhat = X @ coef
        sse = float(((dy - yhat) ** 2).sum())
        var_res = sse / max(n - 3, 1)
        try:
            t = coef[0] / (np.sqrt(var_res * np.linalg.inv(X.T @ X)[0, 0]) + 1e-9)
        except Exception:
            t = float("nan")
        return float(t)

    body = []

    # ════════════════════════════════════════════════════════════════════════════
    # OVERVIEW: IS vs OOS for all strategies
    # ════════════════════════════════════════════════════════════════════════════
    ov_rows = []
    for s in all_strats[:20]:
        ov_rows.append({
            "Paar": s["pair_label"],
            "Indikator": s["ind_name"],
            "IS Start": str(s["is_start"])[:10],
            "IS Ende / OOS Start": str(s["split_date"])[:10],
            "OOS Ende": str(s["oos_end"])[:10],
            "IS N Tage": s["is_n"],
            "OOS N Tage": s["oos_n"],
            "IS Sharpe": round(s["sh_is"], 3),
            "OOS Sharpe": round(s["sh_oos"], 3),
            "OOS > IS": "YES" if s["sh_oos"] > s["sh_is"] else "",
            "OOS/IS-Ratio": round(s["sh_oos"] / (s["sh_is"] + 1e-9), 2),
        })
    ov_df = pd.DataFrame(ov_rows)

    labels_ov = [
        f"{r['Paar'].split('(')[0].strip()} / {r['Indikator']}"
        for r in ov_rows
    ]
    fig_ov = go.Figure()
    fig_ov.add_trace(go.Bar(name="IS Sharpe", x=labels_ov,
        y=[r["IS Sharpe"] for r in ov_rows], marker_color="#d29922", opacity=0.8))
    fig_ov.add_trace(go.Bar(name="OOS Sharpe", x=labels_ov,
        y=[r["OOS Sharpe"] for r in ov_rows],
        marker_color=["#3fb950" if r["OOS > IS"] == "YES" else "#58a6ff"
                      for r in ov_rows]))
    fig_ov.add_hline(y=0, line_color="#8b949e")
    fig_ov.update_layout(
        title="Top-20 Strategien: IS vs. OOS Sharpe (grün = OOS > IS = suspect/robust)",
        barmode="group", xaxis_tickangle=-30, height=500)

    body.append(
        "<div class='ph-header'><h1>Strategy Stress-Test &amp; Deep-Dive</h1>"
        "<div class='sub'>IS/OOS Zeiten und Gr\\u00f6\\u00dfen &middot; TC-Sweep &middot; "
        "Look-Ahead-Test &middot; Kointegrationstest &middot; "
        "Monte Carlo Shuffle &middot; Bootstrap CI &middot; "
        "Walk-Forward &middot; Jahres-Heatmap &middot; "
        "Krisenperioden &middot; VIX-Regime &middot; DXY-Filter &middot; Kelly-Sizing"
        "</div></div>"
        "<div class='card mb-4'><div class='card-header'>"
        "<strong>Warum OOS &gt; IS Sharpe besonders untersucht wird</strong>"
        "</div><div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<p class='small'>Eine h\\u00f6here OOS- als IS-Sharpe kann zweierlei bedeuten:</p>"
        "<ol class='small'>"
        "<li><strong>Robust:</strong> Die Strategie generiert in unbekannten Daten sogar "
        "mehr Alpha als in der Kalibrierungsphase (z.B. OOS-Periode zuf\\u00e4llig "
        "g\\u00fcnstig f\\u00fcr das Signal)</li>"
        "<li><strong>Lucky:</strong> Das OOS-Fenster war ein ideales Regime "
        "\\u2014 kein nachhaltiger Vorteil, sobald das Marktumfeld kippt</li>"
        "</ol>"
        "<p class='small'><em>Der Stress-Test trennt beide Hypothesen durch "
        "statistische und szenariobasierte Tests.</em></p>"
        "</div>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>TC-Sweep:</strong> Wie viel Kosten h\\u00e4lt die Strategie aus?</li>"
        "<li><strong>Monte Carlo (5 000 Shuffles):</strong> Ist der Sharpe statistisch signifikant?</li>"
        "<li><strong>Bootstrap CI:</strong> Konfidenzintervall um den Sharpe</li>"
        "<li><strong>Walk-Forward:</strong> Konsistenz \\u00fcber rollende IS/OOS-Fenster</li>"
        "<li><strong>Krisenperioden:</strong> COVID 2020, Ukraine 2022, \\u00d6lcrash 2014-16</li>"
        "<li><strong>VIX-Regime:</strong> Performance in low/medium/high Volatilität</li>"
        "<li><strong>Kointegration:</strong> Echt\\u00f6konomische Bindung vorhanden?</li>"
        "</ul></div>"
        "</div></div></div>"
    )
    body.append(_chart_card("Top-20 Strategien: IS vs. OOS Sharpe", fig_ov, height=520,
        interp="Grün = OOS Sharpe größer als IS. Blau = IS größer (normales Overfitting-Muster). "
               "Alle Strategien in Tabelle unten mit exakten IS/OOS-Zeiträumen und N Tagen."))
    body.append(_card("Top-20 Strategien (IS/OOS Zeiträume + Metriken)", _df_html(ov_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1: BASELINE REPRODUCTION
    # ════════════════════════════════════════════════════════════════════════════
    f_ldr = focus["leader"]; f_fol = focus["follower"]
    f_lag = focus["lag"];    f_lbl = focus["pair_label"]
    f_ind = focus["ind_name"]; f_thr = focus["thresh"]
    f_fn  = focus["ind_fn"]; f_px = focus["px"]; f_rf = focus["rf"]
    f_spl = focus["split_date"]; f_idx = focus["idx_all"]
    f_is_rf = f_rf.loc[:f_spl]; f_oos_rf = f_rf.loc[f_spl:]

    f_ind_s = f_fn(f_px)
    f_n_full, f_g_full, f_s_full = _strat_exec(f_ind_s, f_thr, f_rf, f_lag)
    f_n_is, f_g_is, f_s_is = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag)
    f_n_oos, f_g_oos, f_s_oos = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag)

    spy_is_f  = spy.loc[f_n_is.index]  if spy is not None else None
    spy_oos_f = spy.loc[f_n_oos.index] if spy is not None else None
    m_is  = _full_metrics(f_n_is,  f_g_is,  f_s_is,  spy_is_f,  "IS")
    m_oos = _full_metrics(f_n_oos, f_g_oos, f_s_oos, spy_oos_f, "OOS")
    m_all = _full_metrics(f_n_full, f_g_full, f_s_full,
                          spy.loc[f_n_full.index] if spy is not None else None, "Gesamt")

    sh_is_val  = m_is.get("Sharpe (net)", 0)
    sh_oos_val = m_oos.get("Sharpe (net)", 0)

    base_rows = []
    for period, m, n_d, start_d, end_d in [
        ("IS (Kalibrierung)", m_is, len(f_n_is), str(f_idx[0])[:10], str(f_spl)[:10]),
        ("OOS (Validierung)", m_oos, len(f_n_oos), str(f_spl)[:10], str(f_idx[-1])[:10]),
        ("Gesamt",           m_all, len(f_n_full), str(f_idx[0])[:10], str(f_idx[-1])[:10]),
    ]:
        row = {"Periode": period, "Start": start_d, "Ende": end_d, "N Tage": n_d}
        row.update({k: v for k, v in m.items() if k != "Name"})
        base_rows.append(row)
    base_df = pd.DataFrame(base_rows)

    fig_bl = go.Figure()
    eq_full = (1 + f_n_full).cumprod()
    bh_full = (1 + f_rf.dropna()).cumprod()
    fig_bl.add_trace(go.Scatter(
        x=eq_full.index.astype(str).tolist(), y=eq_full.round(4).values.tolist(),
        name=f_ind, line=dict(color="#3fb950", width=2)))
    fig_bl.add_trace(go.Scatter(
        x=bh_full.index.astype(str).tolist(), y=bh_full.round(4).values.tolist(),
        name=f"B&H {f_fol}", line=dict(color="#8b949e", width=1, dash="dot")))
    fig_bl.add_vrect(x0=str(f_idx[0])[:10], x1=str(f_spl)[:10],
                     fillcolor="rgba(210,153,34,0.10)", line_width=0)
    fig_bl.add_vrect(x0=str(f_spl)[:10], x1=str(f_idx[-1])[:10],
                     fillcolor="rgba(63,185,80,0.10)", line_width=0)
    fig_bl.update_layout(
        title=(f"Baseline: {f_lbl} | {f_ind} | Lag {f_lag}T | TC=10bp | "
               f"IS: {str(f_idx[0])[:10]}\\u2013{str(f_spl)[:10]} ({len(f_n_is)}T) | "
               f"OOS: {str(f_spl)[:10]}\\u2013{str(f_idx[-1])[:10]} ({len(f_n_oos)}T)"),
        yaxis_type="log", height=520)

    body.append(
        "<div class='card mb-4' style='border-top:4px solid #3fb950'>"
        "<div class='card-header'>"
        f"<h4><strong>Deep-Dive: {f_lbl} | {f_ind}</strong></h4>"
        "<div class='row g-2 mt-1'>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>IS-Periode</div>"
        f"<div class='fw-bold small'>{str(f_idx[0])[:10]}</div>"
        f"<div class='text-muted small'>\\u2192 {str(f_spl)[:10]}</div>"
        f"<span class='badge bg-warning text-dark'>{len(f_n_is)} Handelstage</span>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>OOS-Periode</div>"
        f"<div class='fw-bold small'>{str(f_spl)[:10]}</div>"
        f"<div class='text-muted small'>\\u2192 {str(f_idx[-1])[:10]}</div>"
        f"<span class='badge bg-success'>{len(f_n_oos)} Handelstage</span>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>IS Sharpe</div>"
        f"<div class='fw-bold text-warning' style='font-size:1.6em'>{round(sh_is_val,3)}</div>"
        f"</div></div>"
        f"<div class='col-md-3'><div class='border rounded p-2 text-center'>"
        f"<div class='text-muted small'>OOS Sharpe</div>"
        f"<div class='fw-bold text-success' style='font-size:1.6em'>{round(sh_oos_val,3)}</div>"
        f"{'<div class=\\\"badge bg-danger\\\">OOS > IS — Stress-Test l\\u00e4uft</div>' if sh_oos_val > sh_is_val else ''}"
        f"</div></div>"
        f"<div class='col-md-12 mt-1'><small class='text-muted'>"
        f"Leader: <strong>{f_ldr}</strong> | Follower: <strong>{f_fol}</strong> | "
        f"Lag: <strong>{f_lag} Handelstage</strong> | "
        f"Gesamtdaten: {len(f_n_full)} Tage | IS 70% / OOS 30%"
        f"</small></div>"
        "</div>"
        "</div>"
        f"<div class='card-body'>{_df_html(base_df)}</div>"
        "</div>"
    )
    body.append(_chart_card("Baseline Equity-Kurve (gesamt, log-Skala)", fig_bl, height=540,
        interp=f"Gelb schattiert = IS-Periode ({len(f_n_is)} Tage). "
               f"Grün schattiert = OOS-Periode ({len(f_n_oos)} Tage). "
               f"IS Sharpe {round(sh_is_val,3)} → OOS Sharpe {round(sh_oos_val,3)}. "
               f"{'OOS > IS: ungewöhnlich — alle folgenden Tests prüfen die Robustheit.' if sh_oos_val > sh_is_val else 'OOS < IS: normales Muster — IS-Overfitting wahrscheinlich gering.'}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2: TC SWEEP (Transaktionskosten-Sensitivität)
    # ════════════════════════════════════════════════════════════════════════════
    TC_BPS = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    tc_is_sh = []; tc_oos_sh = []; tc_is_ret = []; tc_oos_ret = []
    for tc_bp in TC_BPS:
        n_is_tc, _, _ = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag, tc=tc_bp/10000)
        n_oos_tc, _, _ = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag, tc=tc_bp/10000)
        if len(n_is_tc) > 30:
            tc_is_sh.append(n_is_tc.mean()*252/(n_is_tc.std()*np.sqrt(252)+1e-9))
            tc_is_ret.append(n_is_tc.mean()*252*100)
        else:
            tc_is_sh.append(float("nan")); tc_is_ret.append(float("nan"))
        if len(n_oos_tc) > 30:
            tc_oos_sh.append(n_oos_tc.mean()*252/(n_oos_tc.std()*np.sqrt(252)+1e-9))
            tc_oos_ret.append(n_oos_tc.mean()*252*100)
        else:
            tc_oos_sh.append(float("nan")); tc_oos_ret.append(float("nan"))

    # Find OOS break-even TC
    be_oos = None
    for i in range(len(TC_BPS)-1):
        if (not np.isnan(tc_oos_sh[i]) and tc_oos_sh[i] >= 0
                and not np.isnan(tc_oos_sh[i+1]) and tc_oos_sh[i+1] < 0):
            denom = tc_oos_sh[i] - tc_oos_sh[i+1] + 1e-9
            be_oos = TC_BPS[i] + (TC_BPS[i+1]-TC_BPS[i]) * tc_oos_sh[i] / denom
            break

    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(x=TC_BPS, y=tc_is_sh, name="IS Sharpe",
        mode="lines+markers", line=dict(color="#d29922", width=2), marker=dict(size=7)))
    fig_tc.add_trace(go.Scatter(x=TC_BPS, y=tc_oos_sh, name="OOS Sharpe",
        mode="lines+markers", line=dict(color="#3fb950", width=2), marker=dict(size=7)))
    fig_tc.add_hline(y=0, line_color="#f78166", line_dash="dash")
    if be_oos:
        fig_tc.add_vline(x=be_oos, line_color="#f78166", line_dash="dot")
    fig_tc.update_layout(
        title=(f"TC-Sweep: Sharpe vs. Transaktionskosten | {f_lbl}"
               + (f" | OOS Break-Even: ~{be_oos:.0f}bp" if be_oos else "")),
        xaxis_title="TC (Basispunkte pro Trade)", yaxis_title="Sharpe Ratio", height=420)

    tc_df = pd.DataFrame({
        "TC (bp)": TC_BPS,
        "IS Sharpe": [round(s,3) if not np.isnan(s) else "—" for s in tc_is_sh],
        "OOS Sharpe": [round(s,3) if not np.isnan(s) else "—" for s in tc_oos_sh],
        "IS Ann.Ret%": [round(r,2) if not np.isnan(r) else "—" for r in tc_is_ret],
        "OOS Ann.Ret%": [round(r,2) if not np.isnan(r) else "—" for r in tc_oos_ret],
        "Benchmarks": ["Ideal (kein TC)","","Institutionell","Standard Retail","","","",
                        "","Aggressiv","","","Sehr aggressiv"][:len(TC_BPS)],
    })
    be_text = f"OOS Break-Even bei ca. {be_oos:.0f}bp TC. " if be_oos else ""
    body.append(_chart_card(
        "TC-Sensitivitäts-Sweep", fig_tc, height=440,
        interp=f"{be_text}Institutionelle Händler zahlen 1–5bp, Retail 5–25bp, Futures-Roll 0.5–2bp. "
               "Strategien unter 10bp-Break-Even sind für Retail unhandelbar. "
               "Rote Linie = Sharpe=0 (Break-Even)."))
    body.append(_card("TC-Sweep Tabelle", _df_html(tc_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3: LOOK-AHEAD BIAS TEST
    # ════════════════════════════════════════════════════════════════════════════
    la_rows = []
    for extra in [0, 1, 2, 3, 5]:
        n_is_la, _, _ = _strat_exec(f_ind_s, f_thr, f_is_rf, f_lag + extra)
        n_oos_la, _, _ = _strat_exec(f_ind_s, f_thr, f_oos_rf, f_lag + extra)
        sh_is_la  = (n_is_la.mean()*252/(n_is_la.std()*np.sqrt(252)+1e-9)
                     if len(n_is_la) > 30 else float("nan"))
        sh_oos_la = (n_oos_la.mean()*252/(n_oos_la.std()*np.sqrt(252)+1e-9)
                     if len(n_oos_la) > 30 else float("nan"))
        la_rows.append({
            "Extra Shift": extra,
            "Total Lag": f_lag + extra,
            "IS Sharpe": round(sh_is_la,3) if not np.isnan(sh_is_la) else "—",
            "OOS Sharpe": round(sh_oos_la,3) if not np.isnan(sh_oos_la) else "—",
            "IS Ann.Ret%": round(n_is_la.mean()*252*100,2) if len(n_is_la)>30 else "—",
            "OOS Ann.Ret%": round(n_oos_la.mean()*252*100,2) if len(n_oos_la)>30 else "—",
        })

    la_df = pd.DataFrame(la_rows)
    fig_la = go.Figure()
    fig_la.add_trace(go.Scatter(
        x=[r["Extra Shift"] for r in la_rows],
        y=[r["IS Sharpe"] if isinstance(r["IS Sharpe"], float) else float("nan")
           for r in la_rows],
        name="IS Sharpe", mode="lines+markers", line=dict(color="#d29922")))
    fig_la.add_trace(go.Scatter(
        x=[r["Extra Shift"] for r in la_rows],
        y=[r["OOS Sharpe"] if isinstance(r["OOS Sharpe"], float) else float("nan")
           for r in la_rows],
        name="OOS Sharpe", mode="lines+markers", line=dict(color="#3fb950")))
    fig_la.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_la.update_layout(
        title=f"Look-Ahead Bias Test: Signal um +0...+5 Tage zusätzlich verzögert",
        xaxis_title="Zusätzliche Signalverzögerung (Tage)", yaxis_title="Sharpe", height=380)

    la_drop = 0.0
    try:
        sh0 = float(la_rows[0]["OOS Sharpe"])
        sh1 = float(la_rows[1]["OOS Sharpe"])
        la_drop = abs((sh1 - sh0) / (sh0 + 1e-9)) * 100
    except Exception:
        pass
    la_note = ("⚠ OOS-Sharpe bricht bei +1 Tag stark ein → möglicher Look-Ahead Bias!"
               if la_drop > 60 else
               f"✓ OOS-Sharpe bleibt bei +1 Tag stabil (Rückgang: {la_drop:.1f}%) → kein wesentlicher Look-Ahead Bias.")

    body.append(_chart_card("Look-Ahead Bias Test", fig_la, height=400,
        interp=f"{la_note} "
               f"Basis-Lag = {f_lag} Tag(e) (Granger-optimiert). "
               "Starker Einbruch bei +1 Tag bedeutet: Signal nutzt Same-Day-Daten unzulässig. "
               "Robuste Lead-Lag-Strategie: Sharpe fällt moderat und graduell."))
    body.append(_card("Look-Ahead Test Tabelle", _df_html(la_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4: COINTEGRATION TEST (Engle-Granger)
    # ════════════════════════════════════════════════════════════════════════════
    px_ldr = prices[f_ldr].dropna() if f_ldr in prices.columns else None
    px_fol = prices[f_fol].dropna() if f_fol in prices.columns else None
    if px_ldr is not None and px_fol is not None:
        cidx = px_ldr.index.intersection(px_fol.index)
        if len(cidx) > 200:
            lp_ldr = np.log(px_ldr.loc[cidx].values.clip(min=1e-6))
            lp_fol = np.log(px_fol.loc[cidx].values.clip(min=1e-6))
            Xc = np.column_stack([np.ones(len(lp_ldr)), lp_ldr])
            coef_c, _, _, _ = np.linalg.lstsq(Xc, lp_fol, rcond=None)
            alpha_c, beta_c = float(coef_c[0]), float(coef_c[1])
            resid_c = lp_fol - Xc @ coef_c
            adf_t = _adf_stat(pd.Series(resid_c))

            # Rolling Z-score of spread
            spread_s = pd.Series(resid_c, index=cidx)
            mu_r = spread_s.rolling(252).mean()
            sg_r = spread_s.rolling(252).std() + 1e-9
            spread_z = (spread_s - mu_r) / sg_r

            fig_coint = go.Figure()
            fig_coint.add_trace(go.Scatter(
                x=spread_z.dropna().index.astype(str).tolist(),
                y=spread_z.dropna().round(4).values.tolist(),
                name="Spread Z-Score (252T)", line=dict(color="#58a6ff", width=1.2)))
            fig_coint.add_hline(y=0, line_color="#8b949e", line_dash="dash")
            fig_coint.add_hline(y=2, line_color="#f78166", line_dash="dot",
                                annotation_text="+2σ Mean-Reversion-Zone")
            fig_coint.add_hline(y=-2, line_color="#3fb950", line_dash="dot",
                                 annotation_text="-2σ Mean-Reversion-Zone")
            fig_coint.add_vrect(x0=str(f_spl)[:10], x1=str(cidx[-1])[:10],
                                fillcolor="rgba(63,185,80,0.07)", line_width=0)
            fig_coint.update_layout(
                title=f"Kointegrationsresidual Z-Score: log({f_ldr}) ↔ log({f_fol})",
                height=400)

            # MacKinnon (1991) 2-variable critical values (no trend, constant)
            crit = {1: -4.07, 5: -3.37, 10: -3.03}
            if not np.isnan(adf_t):
                if adf_t < crit[1]:
                    coint_verdict = f"✓✓ Kointegriert auf 1%-Niveau (t={adf_t:.3f} < {crit[1]})"
                elif adf_t < crit[5]:
                    coint_verdict = f"✓ Kointegriert auf 5%-Niveau (t={adf_t:.3f} < {crit[5]})"
                elif adf_t < crit[10]:
                    coint_verdict = f"~ Schwache Kointegration auf 10%-Niveau (t={adf_t:.3f})"
                else:
                    coint_verdict = f"✗ Keine Kointegration (t={adf_t:.3f} > {crit[10]})"
            else:
                coint_verdict = "Nicht berechenbar"

            coint_df = pd.DataFrame([{
                "ADF t-Statistik (Engle-Granger)": round(adf_t, 4) if not np.isnan(adf_t) else "—",
                "Kritisch 10%": crit[10], "Kritisch 5%": crit[5], "Kritisch 1%": crit[1],
                "β (log-Preis-Elastizität)": round(beta_c, 4),
                "α (Intercept log-Preise)": round(alpha_c, 4),
                "N Beobachtungen": len(cidx),
                "Verdikt": coint_verdict,
            }])
            body.append(_chart_card(
                f"Kointegrations-Spread Z-Score: {f_ldr} ↔ {f_fol}", fig_coint, height=420,
                interp=f"Engle-Granger Test auf OLS-Residual. {coint_verdict}. "
                       f"β={beta_c:.4f}: 1% Bewegung in {f_ldr} → ca. {beta_c:.2f}% in {f_fol}. "
                       "Z-Score > +2σ: Follower relativ zum Leader teuer → Short-Signal (Spread-Trading). "
                       "Z-Score < -2σ: Follower günstig → Long-Signal. Grüne Zone = OOS-Periode."))
            body.append(_card("Kointegrationstest Ergebnis (Engle-Granger)", _df_html(coint_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5: MONTE CARLO TRADE SHUFFLE (5 000 Permutationen)
    # ════════════════════════════════════════════════════════════════════════════
    np.random.seed(42)
    N_MC = 5000
    oos_arr = f_n_oos.dropna().values
    real_sh = float(oos_arr.mean() * 252 / (oos_arr.std() * np.sqrt(252) + 1e-9))
    mc_sh = np.array([
        np.random.permutation(oos_arr).mean() * 252
        / (np.random.permutation(oos_arr).std() * np.sqrt(252) + 1e-9)
        for _ in range(N_MC)])
    pval = float((mc_sh >= real_sh).mean())
    mc95 = float(np.percentile(mc_sh, 95))

    hy, hx = np.histogram(mc_sh, bins=80)
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(
        x=((hx[:-1]+hx[1:])/2).tolist(), y=hy.tolist(),
        marker_color="#58a6ff", opacity=0.7, name="MC Sharpe"))
    fig_mc.add_vline(x=real_sh, line_color="#3fb950", line_width=2.5)
    fig_mc.add_vline(x=mc95, line_color="#d29922", line_dash="dot")
    fig_mc.update_layout(
        title=(f"Monte Carlo ({N_MC} Shuffles) OOS-Sharpe-Verteilung | {f_lbl} | {f_ind} | "
               f"Echte OOS-Sharpe: {real_sh:.3f} | p={pval:.4f}"),
        xaxis_title="Sharpe (Zufallspermutation der OOS-Tagesrenditen)", height=420)

    mc_note = (
        f"⚠ p={pval:.4f} > 0.05 — Sharpe NICHT statistisch signifikant! "
        "Ergebnis könnte zufällige Rendite-Reihenfolge sein."
        if pval > 0.05 else
        f"✓ p={pval:.4f} < 0.05 — Sharpe statistisch signifikant. "
        "Die zeitliche Struktur der Renditen trägt zum Ergebnis bei.")
    body.append(_chart_card(
        f"Monte Carlo Trade-Shuffle: {N_MC} OOS-Permutationen", fig_mc, height=440,
        interp=f"Verteilung der Sharpe-Ratio bei 5 000 zufälligen Permutationen der OOS-Tagesrenditen. "
               f"Grüne Linie = echte OOS-Sharpe ({real_sh:.3f}). "
               f"Gelb gestrichelt = 95. Perzentil der MC-Verteilung ({mc95:.3f}). "
               f"p-Wert ({pval:.4f}): Anteil der Simulationen ≥ echter Sharpe. {mc_note}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 6: BOOTSTRAP CONFIDENCE INTERVAL
    # ════════════════════════════════════════════════════════════════════════════
    N_BOOT = 2000
    is_arr = f_n_is.dropna().values
    boot_is = np.array([
        is_arr[np.random.randint(0,len(is_arr),len(is_arr))].mean()*252
        /(is_arr[np.random.randint(0,len(is_arr),len(is_arr))].std()*np.sqrt(252)+1e-9)
        for _ in range(N_BOOT)])
    boot_oos = np.array([
        oos_arr[np.random.randint(0,len(oos_arr),len(oos_arr))].mean()*252
        /(oos_arr[np.random.randint(0,len(oos_arr),len(oos_arr))].std()*np.sqrt(252)+1e-9)
        for _ in range(N_BOOT)])
    ci_is  = np.percentile(boot_is,  [2.5, 97.5])
    ci_oos = np.percentile(boot_oos, [2.5, 97.5])

    fig_boot = go.Figure()
    for arr, nm, col in [(boot_is,"IS","#d29922"),(boot_oos,"OOS","#3fb950")]:
        hy, hx = np.histogram(arr, bins=60)
        fig_boot.add_trace(go.Bar(
            x=((hx[:-1]+hx[1:])/2).tolist(), y=hy.tolist(),
            name=nm, marker_color=col, opacity=0.55))
    for ci, col, nm in [(ci_is,"#d29922","IS"),(ci_oos,"#3fb950","OOS")]:
        fig_boot.add_vline(x=float(ci[0]), line_color=col, line_dash="dot")
        fig_boot.add_vline(x=float(ci[1]), line_color=col, line_dash="dash")
    fig_boot.update_layout(
        title=f"Bootstrap 95%-CI Sharpe Ratio ({N_BOOT} Samples) | {f_lbl}",
        xaxis_title="Sharpe Ratio", height=400, barmode="overlay")

    ci_note = (f"✓ OOS 95%-CI: [{ci_oos[0]:.3f}, {ci_oos[1]:.3f}] — "
               + ("enthält 0 NICHT → Sharpe signifikant von 0 verschieden."
                  if ci_oos[0] > 0 else "enthält 0 → Sharpe nicht signifikant von 0 verschieden."))
    body.append(_chart_card(
        f"Bootstrap 95%-CI der Sharpe Ratio ({N_BOOT} Samples)", fig_boot, height=420,
        interp=f"Gelb = IS, Grün = OOS. Punktiert/gestrichelt = CI-Grenzen. "
               f"IS 95%-CI: [{ci_is[0]:.3f}, {ci_is[1]:.3f}]. {ci_note}"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 7: WALK-FORWARD ANALYSE (rollende IS/OOS mit Threshold-Optimierung)
    # ════════════════════════════════════════════════════════════════════════════
    WF_IS = 756; WF_OOS = 252; WF_STEP = 252
    use_rsi = "RSI" in f_ind
    thresh_grid = ([-35,-40,-45,-50,-55,-60,-65] if f_thr < 0
                   else ([35,40,45,50,55,60,65] if use_rsi else [f_thr]))

    wf_rows = []
    t0 = 0
    while t0 + WF_IS + WF_OOS <= len(f_idx):
        is_idx  = f_idx[t0 : t0 + WF_IS]
        oos_idx = f_idx[t0 + WF_IS : min(t0 + WF_IS + WF_OOS, len(f_idx))]
        if len(oos_idx) < 60:
            break
        rf_is_wf  = f_rf.loc[is_idx[0]:is_idx[-1]]
        rf_oos_wf = f_rf.loc[oos_idx[0]:oos_idx[-1]]
        best_sh_wf = -999.0; best_thr_wf = float(thresh_grid[0])
        wf_ind_base = -_calc_rsi(f_px, 14) if f_thr < 0 else _calc_rsi(f_px, 14) if use_rsi else f_fn(f_px)
        for thr_wf in thresh_grid:
            n_wf, _, _ = _strat_exec(wf_ind_base, thr_wf, rf_is_wf, f_lag)
            if len(n_wf) > 30:
                sh_wf = n_wf.mean()*252/(n_wf.std()*np.sqrt(252)+1e-9)
                if sh_wf > best_sh_wf:
                    best_sh_wf = sh_wf; best_thr_wf = float(thr_wf)
        n_oos_wf, _, _ = _strat_exec(wf_ind_base, best_thr_wf, rf_oos_wf, f_lag)
        sh_oos_wf = (n_oos_wf.mean()*252/(n_oos_wf.std()*np.sqrt(252)+1e-9)
                     if len(n_oos_wf) > 30 else float("nan"))
        mdd_oos_wf = float("nan")
        if len(n_oos_wf) > 30:
            c = (1+n_oos_wf).cumprod()
            mdd_oos_wf = float((c/c.cummax()-1).min())*100
        wf_rows.append({
            "IS Start": str(is_idx[0])[:10], "IS Ende": str(is_idx[-1])[:10],
            "OOS Start": str(oos_idx[0])[:10], "OOS Ende": str(oos_idx[-1])[:10],
            "IS N": len(rf_is_wf), "OOS N": len(n_oos_wf),
            "Opt. Threshold": round(best_thr_wf, 1),
            "IS Sharpe (opt.)": round(best_sh_wf, 3),
            "OOS Sharpe": round(sh_oos_wf, 3) if not np.isnan(sh_oos_wf) else "—",
            "OOS MaxDD%": round(mdd_oos_wf, 2) if not np.isnan(mdd_oos_wf) else "—",
            "OOS Ann.Ret%": round(n_oos_wf.mean()*252*100,2) if len(n_oos_wf)>30 else "—",
        })
        t0 += WF_STEP

    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_oos_nums = [r["OOS Sharpe"] for r in wf_rows if isinstance(r["OOS Sharpe"], float)]
        wf_pos = sum(1 for v in wf_oos_nums if v > 0)
        wf_tot = len(wf_oos_nums)
        fig_wf = go.Figure()
        x_wf = [r["OOS Start"] + " → " + r["OOS Ende"] for r in wf_rows]
        y_wf = [r["OOS Sharpe"] if isinstance(r["OOS Sharpe"],float) else 0 for r in wf_rows]
        fig_wf.add_trace(go.Bar(x=x_wf, y=y_wf, name="OOS Sharpe",
            marker_color=["#3fb950" if v>0 else "#f78166" for v in y_wf],
            text=[f"{v:.3f}" if isinstance(r["OOS Sharpe"],float) else "—" for r,v in zip(wf_rows,y_wf)],
            textposition="outside"))
        fig_wf.add_trace(go.Scatter(x=x_wf,
            y=[r["IS Sharpe (opt.)"] for r in wf_rows],
            name="IS Sharpe (opt.)", mode="lines+markers",
            line=dict(color="#d29922", width=1.5, dash="dot")))
        fig_wf.add_hline(y=0, line_color="#8b949e")
        fig_wf.update_layout(
            title=(f"Walk-Forward: IS={WF_IS}T / OOS={WF_OOS}T / Schritt={WF_STEP}T | "
                   f"{wf_pos}/{wf_tot} OOS-Fenster > 0"),
            xaxis_tickangle=-25, height=500)
        wf_note = (f"✓ {wf_pos}/{wf_tot} ({100*wf_pos/max(wf_tot,1):.0f}%) OOS-Fenster profitabel."
                   if wf_pos >= wf_tot * 0.6 else
                   f"⚠ Nur {wf_pos}/{wf_tot} ({100*wf_pos/max(wf_tot,1):.0f}%) OOS-Fenster profitabel — instabile Strategie.")
        body.append(_chart_card(
            f"Walk-Forward Analyse (IS={WF_IS}T, OOS={WF_OOS}T, Schritt={WF_STEP}T)",
            fig_wf, height=520,
            interp=f"Rollende IS-Threshold-Optimierung, dann OOS-Test. "
                   f"Gelb gestrichelt = optimierter IS-Sharpe. Balken = echter OOS-Sharpe. "
                   f"{wf_note} "
                   "Konsistente grüne Balken = zeitlich robuste Strategie."))
        body.append(_card("Walk-Forward Tabelle (IS/OOS Zeiträume + N Tage)", _df_html(wf_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 8: JAHRES-KALENDER (Annual Returns)
    # ════════════════════════════════════════════════════════════════════════════
    ann_rows = []
    for yr in sorted(set(f_n_full.index.year)):
        yr_s  = f_n_full[f_n_full.index.year == yr].dropna()
        yr_bh = f_rf.dropna(); yr_bh = yr_bh[yr_bh.index.year == yr]
        if len(yr_s) < 20:
            continue
        sh_y  = yr_s.mean()*252/(yr_s.std()*np.sqrt(252)+1e-9)
        ret_y = yr_s.mean()*252*100
        bh_y  = yr_bh.mean()*252*100 if len(yr_bh)>10 else float("nan")
        mdd_y = float(((1+yr_s).cumprod()/(1+yr_s).cumprod().cummax()-1).min())*100
        ann_rows.append({
            "Jahr": yr,
            "IS/OOS": "OOS" if yr_s.index[-1] >= f_spl else "IS",
            "Ann.Ret% (Strat)": round(ret_y,2),
            "Ann.Ret% (B&H)": round(bh_y,2) if not np.isnan(bh_y) else "—",
            "Alpha%": round(ret_y-bh_y,2) if not np.isnan(bh_y) else "—",
            "Sharpe": round(sh_y,3),
            "MaxDD%": round(mdd_y,2),
            "N Handelstage": len(yr_s),
        })

    if ann_rows:
        ann_df = pd.DataFrame(ann_rows)
        years = [r["Jahr"] for r in ann_rows]
        strat_ret = [r["Ann.Ret% (Strat)"] for r in ann_rows]
        bh_ret = [r["Ann.Ret% (B&H)"] if isinstance(r["Ann.Ret% (B&H)"],float)
                  else float("nan") for r in ann_rows]
        fig_ann = go.Figure()
        fig_ann.add_trace(go.Bar(x=years, y=strat_ret,
            name="Strategie",
            marker_color=["#3fb950" if v>0 else "#f78166" for v in strat_ret],
            text=[f"{v:.1f}%" for v in strat_ret], textposition="outside"))
        fig_ann.add_trace(go.Scatter(x=years, y=bh_ret,
            name=f"B&H {f_fol}", mode="lines+markers",
            line=dict(color="#8b949e", width=1.5, dash="dot")))
        fig_ann.add_hline(y=0, line_color="#8b949e")
        oos_yrs = [r["Jahr"] for r in ann_rows if r["IS/OOS"]=="OOS"]
        if oos_yrs:
            fig_ann.add_vrect(x0=min(oos_yrs)-0.5, x1=max(oos_yrs)+0.5,
                              fillcolor="rgba(63,185,80,0.08)", line_width=0)
        fig_ann.update_layout(
            title=f"Jährliche Performance: {f_lbl} | {f_ind} | Grün schattiert = OOS",
            xaxis_title="Jahr", yaxis_title="Ann. Rendite %", height=440)
        body.append(_chart_card("Jährliche Renditen (Kalender-Heatmap)", fig_ann, height=460,
            interp="Grüne Balken: profitable Jahre. Rote: Verluste. Grauer Pfeil: B&H-Vergleich. "
                   "Grün schattiert = OOS-Jahre (nicht in IS-Kalibrierung enthalten). "
                   "Gute Strategie: mind. 60-70% der Jahre grün, auch OOS-Jahre."))
        body.append(_card("Jahrestabelle", _df_html(ann_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 9: KRISENPERIODEN-ANALYSE
    # ════════════════════════════════════════════════════════════════════════════
    CRISES = [
        ("Öl-Crash 2014–16", "2014-06-01", "2016-02-29",
         "Öl: 100$→26$. Airlines profitieren von Kerosin-Kosten↓"),
        ("COVID-Crash 2020", "2020-01-15", "2020-09-30",
         "JETS -60%, Öl zeitweilig negativ. Fundamentale Entkopplung!"),
        ("COVID-Erholung 2021", "2021-01-01", "2021-12-31",
         "Reopening-Rally, Energie stark, Airline-Nachfrage steigt"),
        ("Ukraine/Energie-Krise 2022", "2022-01-01", "2022-12-31",
         "Öl-Schock, Inflationshoch, Airline-Kostendruck"),
        ("Normalisierung 2023", "2023-01-01", "2023-12-31",
         "Öl moderater, Zinspeak, Airlines erholen sich"),
        ("2024–2025", "2024-01-01", "2025-06-30",
         "Nachhaltiger Reiseboom vs. geopolitische Risiken"),
    ]
    crisis_rows = []
    fig_crises = go.Figure()
    for cname, cs, ce, cdesc in CRISES:
        try:
            cs_ts = pd.Timestamp(cs); ce_ts = pd.Timestamp(ce)
            cr  = f_n_full.loc[cs_ts:ce_ts].dropna()
            cr_bh = f_rf.loc[cs_ts:ce_ts].dropna()
            if len(cr) < 15:
                continue
            cr_sh  = cr.mean()*252/(cr.std()*np.sqrt(252)+1e-9)
            cr_ret = cr.mean()*252*100
            bh_ret_c = cr_bh.mean()*252*100 if len(cr_bh)>10 else float("nan")
            cr_mdd = float(((1+cr).cumprod()/(1+cr).cumprod().cummax()-1).min())*100
            in_oos = cr.index[0] >= f_spl
            crisis_rows.append({
                "Periode": cname, "Start": cs[:10], "Ende": ce[:10],
                "N Tage": len(cr), "IS/OOS": "OOS" if in_oos else "IS (nicht blind)",
                "Ann.Ret% (Strat)": round(cr_ret,2),
                "Ann.Ret% (B&H)": round(bh_ret_c,2) if not np.isnan(bh_ret_c) else "—",
                "Sharpe": round(cr_sh,3), "MaxDD%": round(cr_mdd,2),
                "Ökonom. Kontext": cdesc,
            })
            fig_crises.add_trace(go.Bar(
                x=[cname], y=[cr_ret],
                marker_color="#3fb950" if cr_ret>0 else "#f78166",
                text=f"{cr_ret:.1f}%", textposition="outside", name=cname))
        except Exception:
            continue
    fig_crises.update_layout(
        title=f"Performance in historischen Krisen-/Schlüsselperioden: {f_lbl}",
        showlegend=False, height=420, yaxis_title="Ann. Rendite %")
    fig_crises.add_hline(y=0, line_color="#8b949e")
    if crisis_rows:
        body.append(_chart_card("Krisenperioden-Performance", fig_crises, height=440,
            interp="Performance der Strategie in definierten Marktphasen. "
                   "IS-Perioden: Strategie 'kannte' diese Daten (kein Blind-Test). "
                   "OOS-Perioden: echter Blind-Test. "
                   "COVID 2020 ist der ultimative Stresstest für Öl/Airline-Strategien "
                   "(fundamentale Entkopplung: Öl negativ, Airlines am Boden)."))
        body.append(_card("Krisenperioden Tabelle", _df_html(pd.DataFrame(crisis_rows))))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 10: VIX-REGIME-ANALYSE
    # ════════════════════════════════════════════════════════════════════════════
    if vix_lvl is not None:
        vix_align = vix_lvl.reindex(f_n_full.index, method="ffill").dropna()
        common_v = vix_align.index.intersection(f_n_full.index)
        vix_v = vix_align.loc[common_v]
        strat_v = f_n_full.loc[common_v]
        reg_rows = []
        fig_vix = go.Figure()
        for rname, vlo, vhi, col in [
            ("Low (<15)", 0, 15, "#3fb950"),
            ("Normal (15–25)", 15, 25, "#58a6ff"),
            ("Elevated (25–35)", 25, 35, "#d29922"),
            ("Crisis (>35)", 35, 999, "#f78166"),
        ]:
            mask = (vix_v >= vlo) & (vix_v < vhi)
            r_v = strat_v[mask]; bh_v = f_rf.dropna().reindex(r_v.index)
            if len(r_v) < 15:
                continue
            sh_v  = r_v.mean()*252/(r_v.std()*np.sqrt(252)+1e-9)
            ret_v = r_v.mean()*252*100
            bh_rv = bh_v.mean()*252*100 if len(bh_v.dropna())>10 else float("nan")
            mdd_v = float(((1+r_v).cumprod()/(1+r_v).cumprod().cummax()-1).min())*100
            reg_rows.append({
                "VIX-Regime": rname, "N Tage": len(r_v),
                "Ann.Ret% (Strat)": round(ret_v,2),
                "Ann.Ret% (B&H)": round(bh_rv,2) if not np.isnan(bh_rv) else "—",
                "Sharpe": round(sh_v,3), "MaxDD%": round(mdd_v,2),
            })
            fig_vix.add_trace(go.Bar(x=[rname], y=[ret_v], name=rname,
                marker_color=col, text=f"{ret_v:.1f}%", textposition="outside"))
        fig_vix.add_hline(y=0, line_color="#8b949e")
        fig_vix.update_layout(title=f"VIX-Regime-Performance: {f_lbl}", showlegend=False, height=380)
        if reg_rows:
            body.append(_chart_card("VIX-Regime-Analyse", fig_vix, height=400,
                interp="Performance in verschiedenen Volatilitätsregimes. "
                       "Crisis VIX (>35): COVID 2020, GFC 2008 → Öl/Airline-Korrelation bricht auf. "
                       "Normal VIX (15–25): ideales Umfeld für fundamentale Lead-Lag-Strategien. "
                       "Low VIX (<15): geringe Bewegungen → wenig Signal."))
            body.append(_card("VIX-Regime Tabelle", _df_html(pd.DataFrame(reg_rows))))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 11: DXY-FILTER
    # ════════════════════════════════════════════════════════════════════════════
    if dxy is not None:
        dxy_trend = dxy.rolling(20).mean()
        dxy_align = dxy_trend.reindex(f_s_full.index, method="ffill")
        common_dx = dxy_align.dropna().index.intersection(f_n_full.index)
        if len(common_dx) > 100:
            dx = dxy_align.loc[common_dx]; sbase = f_s_full.loc[common_dx]
            rf_c = f_rf.reindex(common_dx)
            # Filter: when DXY rising (+), reduce position by 50% (strong dollar → oil cheaper → airlines less benefit)
            sig_flt = sbase * dxy_align.loc[common_dx].apply(
                lambda x: 0.5 if (not np.isnan(x) and x > 0) else 1.0)
            grs_flt = sig_flt * rf_c
            net_flt = (grs_flt - sig_flt.diff().abs().fillna(0) * 0.001).dropna()
            oos_common = common_dx[common_dx >= f_spl]
            dxy_comp = []
            for lbl_dx, n_dx in [
                ("Base (kein DXY-Filter)", f_n_oos),
                ("DXY-Filter (50% bei DXY-Trend↑)", net_flt.loc[oos_common] if len(oos_common)>30 else pd.Series()),
            ]:
                r_dx = n_dx.dropna()
                if len(r_dx) < 30:
                    continue
                sh_dx = r_dx.mean()*252/(r_dx.std()*np.sqrt(252)+1e-9)
                dxy_comp.append({
                    "Version": lbl_dx, "N OOS Tage": len(r_dx),
                    "OOS Sharpe": round(sh_dx,3),
                    "OOS Ann.Ret%": round(r_dx.mean()*252*100,2),
                    "OOS MaxDD%": round(float(((1+r_dx).cumprod()/(1+r_dx).cumprod().cummax()-1).min())*100,2),
                })
            if dxy_comp:
                body.append(_card(
                    f"DXY-Regime-Filter Test (OOS): {f_lbl}",
                    _df_html(pd.DataFrame(dxy_comp))
                    + "<p class='small text-muted mt-2'>"
                    "Rationale: Öl wird in USD gehandelt. Steigender Dollar (DXY ↑) drückt tendenziell "
                    "den Ölpreis (=günstiger für Airlines → schwächere Reaktion auf Ölpreissignal). "
                    "Filter: 50% Position wenn DXY-20T-Trend positiv, 100% wenn negativ."
                    "</p>"))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 12: KELLY CRITERION
    # ════════════════════════════════════════════════════════════════════════════
    is_r_k = f_n_is.dropna()
    mu_k = float(is_r_k.mean()); var_k = float(is_r_k.var())
    kelly = mu_k / (var_k + 1e-9)
    hkelly = kelly / 2.0
    kelly_cap = float(np.clip(kelly, 0.01, 3.0))
    hkelly_cap = float(np.clip(hkelly, 0.01, 2.0))

    fig_kel = go.Figure()
    bh_oos_eq = (1 + f_rf.loc[f_spl:].dropna()).cumprod()
    fig_kel.add_trace(go.Scatter(
        x=bh_oos_eq.index.astype(str).tolist(), y=bh_oos_eq.round(4).values.tolist(),
        name=f"B&H {f_fol}", line=dict(color="#444c56", width=0.8, dash="dot")))
    for mult, nm, col in [
        (1.0, "1× (Base)", "#8b949e"),
        (hkelly_cap, f"{hkelly_cap:.2f}× (Half-Kelly)", "#d29922"),
        (kelly_cap, f"{kelly_cap:.2f}× (Full-Kelly)", "#3fb950"),
    ]:
        r_k = f_n_oos * mult
        eq_k = (1 + r_k).cumprod()
        fig_kel.add_trace(go.Scatter(
            x=eq_k.index.astype(str).tolist(), y=eq_k.round(4).values.tolist(),
            name=nm, line=dict(width=1.5)))
    fig_kel.update_layout(
        title=f"Kelly Criterion Sizing (OOS): {f_lbl} | Full-Kelly: {kelly_cap:.2f}×",
        yaxis_type="log", height=440)

    kelly_df = pd.DataFrame([
        {"Sizing": "1× (100%)", "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(m_oos.get("Ann.Ret% (net)",0),2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0),2), "Kommentar": "Base"},
        {"Sizing": f"{hkelly_cap:.2f}× (Half-Kelly)",
         "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(f_n_oos.mean()*252*100*hkelly_cap,2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0)*hkelly_cap,2), "Kommentar": "Empfohlen"},
        {"Sizing": f"{kelly_cap:.2f}× (Full-Kelly)",
         "OOS Sharpe": round(m_oos.get("Sharpe (net)",0),3),
         "OOS Ann.Ret%": round(f_n_oos.mean()*252*100*kelly_cap,2),
         "OOS MaxDD%": round(m_oos.get("MaxDD%",0)*kelly_cap,2), "Kommentar": "Maximales Wachstum"},
    ])
    body.append(_chart_card("Kelly Criterion Position Sizing (OOS)", fig_kel, height=460,
        interp=f"Kelly f* = μ/σ² (aus IS-Daten: μ={mu_k:.5f}/T, σ²={var_k:.6f}). "
               f"Full-Kelly: {kelly:.2f}× → maximales geometrisches Wachstum, aber hohe Schwankungen. "
               f"Half-Kelly: {hkelly:.2f}× → bewährter Kompromiss (Sharpe identisch, Drawdowns kleiner). "
               "Log-Skala. Sharpe-Ratio ändert sich bei linearem Scaling NICHT."))
    body.append(_card("Kelly Sizing Tabelle", _df_html(kelly_df)))

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 13: OPTIMIERUNGSVORSCHLÄGE
    # ════════════════════════════════════════════════════════════════════════════
    body.append(
        "<div class='card mb-4'>"
        "<div class='card-header'><strong>Optimierungsvorschläge &amp; Nächste Schritte</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<h6>Signal-Verbesserungen</h6><ul class='small'>"
        "<li><strong>Crack Spread (HO=F):</strong> Heizöl/Kerosin-Futures statt CL=F Rohöl "
        "als direktere Proxy für Airline-Treibstoffkosten</li>"
        "<li><strong>EIA Rohöl-Lager (FRED: WCRSTUS1):</strong> Wöchentliche Lagerbestände als "
        "Frühindikator für Ölpreisbewegungen (Überraschung: Zunahme → Öl fällt → Airlines steigen)</li>"
        "<li><strong>RSI-Divergenz:</strong> CL-Preis steigt, CL-RSI fällt → Trendumkehr → "
        "Signal verstärken / Position aufbauen</li>"
        "<li><strong>Volumen-Filter:</strong> JETS-Signal nur bestätigen wenn Volumen > 21T-MA "
        "(hohes Volumen = institutionelle Bestätigung)</li>"
        "<li><strong>TSA Checkpoint-Daten:</strong> Tägliche US-Passagierzahlen als "
        "Demand-Indikator für JETS ETF (https://www.tsa.gov/travel/passenger-volumes)</li>"
        "</ul></div>"
        "<div class='col-md-6'>"
        "<h6>Risikomanagement</h6><ul class='small'>"
        "<li><strong>Stop-Loss:</strong> -2% oder -1.5σ(21T) pro Position → "
        "Verhindert Tail-Risk in Krisen (COVID 2020)</li>"
        "<li><strong>VIX-Filter:</strong> Bei VIX > 30 → Positionsgröße halbieren "
        "(Slippage erhöht sich, Muster werden unzuverlässiger)</li>"
        "<li><strong>Half-Kelly Sizing:</strong> Wie gezeigt: gleicher Sharpe, "
        "weniger Drawdowns → Standard-Empfehlung</li>"
        "<li><strong>Regime-bedingte Parameter:</strong> "
        "Low-VIX: Threshold erhöhen (60), High-VIX: Threshold senken (40) → "
        "Adaptives Signal</li>"
        "</ul>"
        "<h6 class='mt-3'>Daten-Integrität</h6><ul class='small'>"
        "<li><strong>CL=F Roll-Kosten (Contango):</strong> Öl-Futures rollen monatlich. "
        "Bei Contango (Vorwärtskurve steigend) entstehen Rollverluste von 0.5–3%/Monat "
        "→ In historischen Returns von CL=F oft NICHT korrekt abgebildet</li>"
        "<li><strong>JETS Dividenden:</strong> yfinance `auto_adjust=True` "
        "adjustiert für Ausschüttungen (bereits enthalten)</li>"
        "<li><strong>Survivorship-Bias:</strong> JETS ETF erst seit 2015 → "
        "historische Daten davor werden synthetisch extrapoliert</li>"
        "</ul></div>"
        "</div>"
        "<div class='alert alert-warning mt-3 small mb-0'>"
        "<strong>Wichtigste Caveat:</strong> Die CL=F ↔ JETS Beziehung kann durch "
        "strukturellen Wandel abgeschwächt werden: SAF (Sustainable Aviation Fuel) "
        "reduziert Öl-Abhängigkeit der Airlines, Fuel-Hedging-Programme (Delta, Southwest) "
        "entkoppeln kurzfristig. Quartalweises Re-Backtesting und Walk-Forward-Monitoring "
        "sind für Live-Trading essenziell."
        "</div>"
        "</div></div>"
    )

    _write(out / "strategy_stress_test.html",
           _html_base("Strategy Stress-Test", 19, "".join(body)))
'''

# ─────────────────────────────────────────────────────────────────────────────
with open(RB, "r", encoding="utf-8") as f:
    src = f.read()

INSERT_BEFORE = "\ndef build_index(tables, figures, out):"
idx = src.find(INSERT_BEFORE)
if idx < 0:
    raise RuntimeError("build_index not found")

new_src = src[:idx] + "\n" + STRESS_TEST_FN + src[idx:]

# Wire into build_all_reports
OLD_W = ("    build_pca_strategy_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_pca_strategy_report(tables, figures, reports)\n"
         "    build_strategy_stress_test_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
if OLD_W in new_src:
    new_src = new_src.replace(OLD_W, NEW_W, 1)
else:
    print("WARNING: build_all_reports wiring failed")

# Wire into build_index
OLD_I = ('        ("PCA-Strategie","PC1-Filter \u00b7 Version A/B \u00b7 Bootstrap-CI \u00b7 IS/OOS","pca_strategy.html","#c9d1d9",\n'
         '         (tables/"phase2_returns.csv").exists()),\n'
         '    ]')
NEW_I = ('        ("PCA-Strategie","PC1-Filter \u00b7 Version A/B \u00b7 Bootstrap-CI \u00b7 IS/OOS","pca_strategy.html","#c9d1d9",\n'
         '         (tables/"phase2_returns.csv").exists()),\n'
         '        ("Strategy Stress-Test","TC-Sweep \u00b7 Monte Carlo \u00b7 Bootstrap CI \u00b7 Walk-Forward \u00b7 Krisenperioden \u00b7 Kelly","strategy_stress_test.html","#ff9fef",\n'
         '         (tables/"phase2_returns.csv").exists()),\n'
         '    ]')
if OLD_I in new_src:
    new_src = new_src.replace(OLD_I, NEW_I, 1)
else:
    print("WARNING: build_index wiring failed")

with open(RB, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"Done. {len(new_src.splitlines())} lines")
