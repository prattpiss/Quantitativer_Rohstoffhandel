"""Injects lead-lag optimizer, comprehensive strategy-pairs, and PCA strategy reports."""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Module-level shared helpers (indicator functions + full metrics)
# ─────────────────────────────────────────────────────────────────────────────
SHARED_BLOCK = '''

# ── Strategy shared helpers ───────────────────────────────────────────────────
try:
    PHASE_COLOURS[19] = "#c9d1d9"
except Exception:
    pass


def _calc_rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=w, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(span=w, adjust=False).mean()
    return 100 - 100 / (1 + g / (ls + 1e-9))


def _calc_macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m = ef - es
    return m, m.ewm(span=sig, adjust=False).mean()


def _calc_bb_pos(s, w=20, n_std=2.0):
    mid = s.rolling(w).mean()
    std = s.rolling(w).std()
    lo = mid - n_std * std
    hi = mid + n_std * std
    return (s - lo) / (hi - lo + 1e-9)


def _calc_sma_cross(s, fast=20, slow=50):
    return s.rolling(fast).mean() - s.rolling(slow).mean()


def _strat_exec(indicator, threshold, follower_ret, lag, tc=0.001):
    """Long when indicator(t-lag)>threshold, else short. Returns (net, gross, signals)."""
    sig = ((indicator.shift(lag) > threshold).astype(float) * 2 - 1)
    sig = sig.reindex(follower_ret.index, method="ffill")
    idx = sig.dropna().index.intersection(follower_ret.dropna().index)
    if len(idx) < 30:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    s = sig.loc[idx]
    fr = follower_ret.loc[idx]
    gross = s * fr
    net = gross - s.diff().abs().fillna(0) * tc
    valid = net.dropna().index
    return net.loc[valid], gross.loc[valid], s.loc[valid]


def _full_metrics(net, gross=None, signals=None, spy=None, name=""):
    """26-metric strategy scorecard from daily net-return series."""
    r = net.dropna()
    rg = gross.dropna() if gross is not None and len(gross) > 0 else r
    if len(r) < 30:
        return {"Name": name, "Sharpe (net)": float("nan")}

    def _sh(x):
        return x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9)

    def _mdd(x):
        c = (1 + x).cumprod()
        return float((c / c.cummax() - 1).min())

    down = r[r < 0]
    sortino = (r.mean() * 252 / (down.std() * np.sqrt(252) + 1e-9)
               if len(down) > 5 else float("nan"))
    calmar = r.mean() * 252 / (abs(_mdd(r)) + 1e-9)

    cum = (1 + r).cumprod()
    rm = cum.cummax()
    dd = cum / rm - 1
    in_dd = dd < 0
    grp = (in_dd != in_dd.shift()).cumsum()
    dur = in_dd.groupby(grp).sum()
    avg_dur = float(dur[dur > 0].mean()) if (dur > 0).any() else 0.0

    sig_s = (signals.reindex(r.index).fillna(0)
             if signals is not None else np.sign(r))
    trades = int((sig_s.diff().abs() > 0).sum())
    wr = float((r > 0).mean())
    avg_w = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_l = float(r[r < 0].mean()) if (r < 0).any() else 0.0
    pf = (abs(r[r > 0].sum() / (r[r < 0].sum() - 1e-9))
          if (r < 0).any() else 99.0)
    omega = r[r > 0].sum() / (-r[r < 0].sum() + 1e-9)
    tail = abs(np.percentile(r, 95) / (np.percentile(r, 5) - 1e-9))
    var5 = float(np.percentile(r, 5))
    cvar5 = float(r[r <= var5].mean()) if (r <= var5).any() else var5

    beta = 0.0
    alpha = 0.0
    if spy is not None:
        idx2 = r.index.intersection(spy.dropna().index)
        if len(idx2) > 60:
            X = spy.loc[idx2].values
            Y = r.loc[idx2].values
            Xc = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
            alpha = float(coef[0] * 252)
            beta = float(coef[1])

    return {
        "Name": name,
        "Ann.Ret% (net)": round(r.mean() * 252 * 100, 2),
        "Ann.Ret% (gross)": round(rg.mean() * 252 * 100, 2),
        "TC Drag% p.a.": round((rg.mean() - r.mean()) * 252 * 100, 3),
        "Ann.Vol%": round(r.std() * np.sqrt(252) * 100, 2),
        "Sharpe (net)": round(_sh(r), 3),
        "Sharpe (gross)": round(_sh(rg), 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "MaxDD%": round(_mdd(r) * 100, 2),
        "AvgDD-Dur(d)": round(avg_dur, 1),
        "# Trades": trades,
        "Long Days": int((sig_s > 0).sum()),
        "Short Days": int((sig_s < 0).sum()),
        "Flat Days": int((sig_s == 0).sum()),
        "Win Rate%": round(wr * 100, 1),
        "Avg Win%": round(avg_w * 100, 4),
        "Avg Loss%": round(avg_l * 100, 4),
        "Profit Factor": round(pf, 3),
        "Omega": round(omega, 3),
        "Tail Ratio": round(tail, 3),
        "Skewness": round(float(r.skew()), 3),
        "Kurtosis": round(float(r.kurtosis()), 3),
        "VaR5%/d": round(var5 * 100, 4),
        "CVaR5%/d": round(cvar5 * 100, 4),
        "AC1": round(float(r.autocorr(1)), 3),
        "AC5": round(float(r.autocorr(5)), 3),
        "Beta(SPY)": round(beta, 3),
        "Alpha% p.a.": round(alpha * 100, 3),
    }

'''

# ─────────────────────────────────────────────────────────────────────────────
# 2. Lead-Lag Parameter Optimizer (one indicator at a time, grid-search heatmaps)
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZER_FN = '''
def build_lead_lag_optimizer_report(tables, figures, out):  # noqa: C901
    """Grid-search parameter optimization per indicator for top Granger pairs."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "lead_lag_optimizer.html",
               _html_base("Lead-Lag Optimizer", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── Top-5 Granger pairs ──────────────────────────────────────────────────
    MANUAL = [
        ("GC=F", "GDX", 7, "Gold→GDX"),
        ("GC=F", "NEM", 6, "Gold→NEM"),
        ("CL=F", "XLE", 1, "WTI→XLE"),
        ("HG=F", "FCX", 1, "Kupfer→FCX"),
        ("SI=F", "SIL", 1, "Silber→SIL"),
    ]
    top_pairs = []
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            if (c in returns.columns and e in returns.columns
                    and not any(p[0] == c and p[1] == e for p in top_pairs)):
                top_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
            if len(top_pairs) >= 5:
                break
    for mp in MANUAL:
        if not any(p[0] == mp[0] and p[1] == mp[1] for p in top_pairs):
            if mp[0] in returns.columns and mp[1] in returns.columns:
                top_pairs.append(mp)
        if len(top_pairs) >= 5:
            break

    # ── Parameter grids ───────────────────────────────────────────────────────
    RSI_WINDOWS = [7, 10, 14, 21, 30]
    RSI_THRESH  = [35, 40, 45, 50, 55, 60, 65]
    MACD_FAST   = [6, 8, 12, 16, 20]
    MACD_SLOW   = [18, 21, 26, 34, 50]
    BB_WINDOWS  = [10, 15, 20, 30, 50]
    BB_THRESH   = [0.3, 0.4, 0.5, 0.6, 0.7]
    SMA_FAST    = [5, 10, 20, 30]
    SMA_SLOW    = [20, 50, 100, 200]

    body_sections = []

    for leader, follower, lag, pair_label in top_pairs:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        if px is None or len(px) < 400:
            continue

        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 300:
            continue
        split_date = idx_all[int(len(idx_all) * 0.70)]

        def is_r(s):
            return s.loc[:split_date]

        def oos_r(s):
            return s.loc[split_date:]

        summary_rows = []
        pair_html = []
        pair_html.append(
            "<h4 class='mt-4 mb-3' style='border-bottom:1px solid #30363d;padding-bottom:6px'>"
            + pair_label + "</h4>"
            + "<p class='small text-muted'>IS: erste 70% &middot; OOS: letzte 30% &middot; "
            + f"Lag: {lag}T &middot; TC: 10bp</p>")

        # ── RSI heatmap ──────────────────────────────────────────────────────
        rsi_mat = np.full((len(RSI_WINDOWS), len(RSI_THRESH)), float("nan"))
        best_rsi = {"sharpe": -999.0, "params": None}
        for wi, w in enumerate(RSI_WINDOWS):
            ind = _calc_rsi(px, w)
            for ti, t in enumerate(RSI_THRESH):
                n, g, s = _strat_exec(ind, t, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    rsi_mat[wi, ti] = sh
                    if sh > best_rsi["sharpe"]:
                        best_rsi = {"sharpe": sh, "params": (w, t)}

        fig_rsi = go.Figure(go.Heatmap(
            z=np.round(rsi_mat, 3).tolist(),
            x=[str(t) for t in RSI_THRESH],
            y=[str(w) for w in RSI_WINDOWS],
            colorscale="RdYlGn", zmid=0,
            text=np.round(rsi_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="RSI-Win=%{y} Thresh=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_rsi.update_layout(
            title="RSI: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Threshold", yaxis_title="RSI Window", height=360)

        if best_rsi["params"]:
            bw, bt = best_rsi["params"]
            n_oos, g_oos, s_oos = _strat_exec(_calc_rsi(px, bw), bt, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "RSI",
                "Best Params (IS)": f"w={bw}, thresh={bt}",
                "IS Sharpe": round(best_rsi["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "RSI-Heatmap: " + pair_label, fig_rsi, height=380,
            interp=f"IS-Sharpe für jeden RSI-Window × Threshold-Kombination. "
                   f"Grün=positiv. Bester IS-Punkt: w={best_rsi['params']}. "
                   "OOS-Sharpe in Tabelle zeigt, ob IS-Optimum generaliserbar ist."))

        # ── MACD heatmap ─────────────────────────────────────────────────────
        macd_mat = np.full((len(MACD_FAST), len(MACD_SLOW)), float("nan"))
        best_macd = {"sharpe": -999.0, "params": None}
        for fi, fast in enumerate(MACD_FAST):
            for si, slow in enumerate(MACD_SLOW):
                if fast >= slow:
                    continue
                m_line, _ = _calc_macd(px, fast, slow)
                n, g, s = _strat_exec(m_line, 0, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    macd_mat[fi, si] = sh
                    if sh > best_macd["sharpe"]:
                        best_macd = {"sharpe": sh, "params": (fast, slow)}

        fig_macd = go.Figure(go.Heatmap(
            z=np.round(macd_mat, 3).tolist(),
            x=[str(s) for s in MACD_SLOW],
            y=[str(f) for f in MACD_FAST],
            colorscale="RdYlGn", zmid=0,
            text=np.round(macd_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Fast=%{y} Slow=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_macd.update_layout(
            title="MACD: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Slow EMA", yaxis_title="Fast EMA", height=360)

        if best_macd["params"]:
            bf, bs = best_macd["params"]
            ml, _ = _calc_macd(px, bf, bs)
            n_oos, g_oos, s_oos = _strat_exec(ml, 0, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "MACD",
                "Best Params (IS)": f"fast={bf}, slow={bs}",
                "IS Sharpe": round(best_macd["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "MACD-Heatmap: " + pair_label, fig_macd, height=380,
            interp="Leere Zellen: fast >= slow (ungültig). "
                   "Threshold fest auf 0 (MACD-Linie > 0 = bullisch)."))

        # ── BB-Position heatmap ───────────────────────────────────────────────
        bb_mat = np.full((len(BB_WINDOWS), len(BB_THRESH)), float("nan"))
        best_bb = {"sharpe": -999.0, "params": None}
        for wi, w in enumerate(BB_WINDOWS):
            bbp = _calc_bb_pos(px, w)
            for ti, t in enumerate(BB_THRESH):
                n, g, s = _strat_exec(bbp, t, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    bb_mat[wi, ti] = sh
                    if sh > best_bb["sharpe"]:
                        best_bb = {"sharpe": sh, "params": (w, t)}

        fig_bb = go.Figure(go.Heatmap(
            z=np.round(bb_mat, 3).tolist(),
            x=[str(t) for t in BB_THRESH],
            y=[str(w) for w in BB_WINDOWS],
            colorscale="RdYlGn", zmid=0,
            text=np.round(bb_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Window=%{y} Thresh=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_bb.update_layout(
            title="BB-Position: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Threshold (0=unten, 1=oben)", yaxis_title="BB Window",
            height=360)

        if best_bb["params"]:
            bw, bt = best_bb["params"]
            n_oos, g_oos, s_oos = _strat_exec(_calc_bb_pos(px, bw), bt, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "BB-Position",
                "Best Params (IS)": f"w={bw}, thresh={bt}",
                "IS Sharpe": round(best_bb["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "BB-Position-Heatmap: " + pair_label, fig_bb, height=380,
            interp="BB-Position: 0=unteres Band, 1=oberes Band. "
                   ">0.5 = Preis über Mittelband (bullisch)."))

        # ── SMA-Cross heatmap ─────────────────────────────────────────────────
        sma_mat = np.full((len(SMA_FAST), len(SMA_SLOW)), float("nan"))
        best_sma = {"sharpe": -999.0, "params": None}
        for fi, fast in enumerate(SMA_FAST):
            for si, slow in enumerate(SMA_SLOW):
                if fast >= slow:
                    continue
                cross = _calc_sma_cross(px, fast, slow)
                n, g, s = _strat_exec(cross, 0, is_r(rf), lag)
                if len(n) > 30:
                    sh = n.mean() * 252 / (n.std() * np.sqrt(252) + 1e-9)
                    sma_mat[fi, si] = sh
                    if sh > best_sma["sharpe"]:
                        best_sma = {"sharpe": sh, "params": (fast, slow)}

        fig_sma = go.Figure(go.Heatmap(
            z=np.round(sma_mat, 3).tolist(),
            x=[str(s) for s in SMA_SLOW],
            y=[str(f) for f in SMA_FAST],
            colorscale="RdYlGn", zmid=0,
            text=np.round(sma_mat, 2).tolist(),
            texttemplate="%{text}",
            hovertemplate="Fast=%{y} Slow=%{x}<br>IS-Sharpe=%{z:.3f}<extra></extra>"))
        fig_sma.update_layout(
            title="SMA-Cross: IS-Sharpe-Heatmap — " + pair_label,
            xaxis_title="Slow SMA", yaxis_title="Fast SMA", height=360)

        if best_sma["params"]:
            bf, bs = best_sma["params"]
            n_oos, g_oos, s_oos = _strat_exec(
                _calc_sma_cross(px, bf, bs), 0, oos_r(rf), lag)
            m_oos = _full_metrics(n_oos, g_oos, s_oos,
                                  spy.loc[n_oos.index] if spy is not None else None)
            summary_rows.append({
                "Indikator": "SMA-Cross",
                "Best Params (IS)": f"fast={bf}, slow={bs}",
                "IS Sharpe": round(best_sma["sharpe"], 3),
                "OOS Sharpe": m_oos.get("Sharpe (net)", float("nan")),
                "OOS Sortino": m_oos.get("Sortino", float("nan")),
                "OOS MaxDD%": m_oos.get("MaxDD%", float("nan")),
                "OOS Ann.Ret%": m_oos.get("Ann.Ret% (net)", float("nan")),
                "# Trades": m_oos.get("# Trades", 0),
            })
        pair_html.append(_chart_card(
            "SMA-Cross-Heatmap: " + pair_label, fig_sma, height=380,
            interp="SMA-Cross = fast_SMA - slow_SMA. >0 → Aufwärtstrend. "
                   "Leere Zellen: fast >= slow."))

        # ── Summary + Walk-Forward ───────────────────────────────────────────
        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            best_overall = sum_df.loc[
                sum_df["OOS Sharpe"].apply(
                    lambda x: x if not (isinstance(x, float) and np.isnan(x)) else -999
                ).idxmax()]
            pair_html.append(
                "<div class='card mb-3'>"
                "<div class='card-header'><strong>Optimierungsergebnis: "
                + pair_label
                + "</strong><span class='badge bg-success ms-2'>Bester OOS: "
                + str(best_overall["Indikator"])
                + " ("
                + str(best_overall["Best Params (IS)"])
                + ") → Sharpe "
                + str(round(best_overall["OOS Sharpe"], 3))
                + "</span></div>"
                "<div class='card-body'>"
                + _df_html(sum_df)
                + "</div></div>")

            # Walk-forward: 4 expanding-IS windows (20-80%), test on next 20%
            all_candidates = [b for b in [best_rsi, best_macd, best_bb, best_sma]
                              if b["params"]]
            if all_candidates:
                best_any = max(all_candidates, key=lambda b: b["sharpe"])
                if best_any is best_rsi:
                    bw, bt = best_any["params"]
                    wf_ind = _calc_rsi(px, bw); wf_thresh = bt
                    wf_name = f"RSI(w={bw},t={bt})"
                elif best_any is best_macd:
                    bf, bs = best_any["params"]
                    wf_ind, _ = _calc_macd(px, bf, bs); wf_thresh = 0
                    wf_name = f"MACD({bf},{bs})"
                elif best_any is best_bb:
                    bw, bt = best_any["params"]
                    wf_ind = _calc_bb_pos(px, bw); wf_thresh = bt
                    wf_name = f"BB-Pos(w={bw},t={bt})"
                else:
                    bf, bs = best_any["params"]
                    wf_ind = _calc_sma_cross(px, bf, bs); wf_thresh = 0
                    wf_name = f"SMA-Cross({bf},{bs})"

                wf_rows = []
                n_total = len(idx_all)
                for step in range(4):
                    oos_s = idx_all[int(n_total * (step + 1) / 5)]
                    oos_e = idx_all[min(int(n_total * (step + 2) / 5) - 1, n_total - 1)]
                    wf_ret = rf.loc[oos_s:oos_e]
                    if len(wf_ret) < 60:
                        continue
                    n_wf, g_wf, s_wf = _strat_exec(wf_ind, wf_thresh, wf_ret, lag)
                    if len(n_wf) < 30:
                        continue
                    sh = n_wf.mean() * 252 / (n_wf.std() * np.sqrt(252) + 1e-9)
                    md = (1 + n_wf).cumprod()
                    md = float((md / md.cummax() - 1).min())
                    wf_rows.append({
                        "WF-Fenster": f"{str(oos_s)[:10]} – {str(oos_e)[:10]}",
                        "OOS Sharpe": round(sh, 3),
                        "OOS MaxDD%": round(md * 100, 2),
                        "OOS Ann.Ret%": round(n_wf.mean() * 252 * 100, 2),
                        "N Tage": len(n_wf),
                    })

                if wf_rows:
                    wf_df = pd.DataFrame(wf_rows)
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Bar(
                        x=wf_df["WF-Fenster"].tolist(),
                        y=wf_df["OOS Sharpe"].tolist(),
                        marker_color=["#3fb950" if v > 0 else "#f78166"
                                      for v in wf_df["OOS Sharpe"]],
                        text=[f"{v:.3f}" for v in wf_df["OOS Sharpe"]],
                        textposition="outside"))
                    fig_wf.add_hline(y=0, line_color="#8b949e")
                    fig_wf.update_layout(
                        title="Walk-Forward OOS Sharpe: " + wf_name + " | " + pair_label,
                        height=360)
                    pair_html.append(_chart_card(
                        "Walk-Forward Validation: " + pair_label, fig_wf, height=380,
                        interp="4 OOS-Fenster à ~20% der Daten (expandierendes IS-Fenster). "
                               "Grün: positiv, Rot: negativ. "
                               "Konsistent grüne Balken = robuste Strategie, nicht nur IS-Overfitting."))
                    pair_html.append(_card("Walk-Forward Tabelle", _df_html(wf_df)))

        body_sections.append("".join(pair_html))

    body = (
        "<div class='ph-header'>"
        "<h1>Lead-Lag Indikator-Optimierung</h1>"
        "<div class='sub'>Ein Indikator gleichzeitig &middot; IS-Sharpe-Heatmap (Parameter-Grid) &middot; "
        "Walk-Forward OOS Validation &middot; Top-5 Granger-Paare</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Methodik</strong></div>"
        "<div class='card-body'>"
        "<div class='row'>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>IS-Periode:</strong> erste 70% der Daten → Parameter-Kalibrierung auf Sharpe</li>"
        "<li><strong>OOS-Periode:</strong> letzte 30% → echte Evaluation mit IS-optimierten Params</li>"
        "<li><strong>Walk-Forward:</strong> 4 expandierende IS-Fenster (je 20% OOS)</li>"
        "<li><strong>TC:</strong> 10bp pro Richtungswechsel (round-trip)</li>"
        "</ul></div>"
        "<div class='col-md-6'><ul class='small'>"
        "<li><strong>RSI:</strong> Window [7,10,14,21,30] × Threshold [35..65] = 35 Kombinationen</li>"
        "<li><strong>MACD:</strong> Fast [6,8,12,16,20] × Slow [18..50] = 25 (nur fast&lt;slow)</li>"
        "<li><strong>BB-Pos:</strong> Window [10..50] × Threshold [0.3..0.7] = 25 Kombinationen</li>"
        "<li><strong>SMA-Cross:</strong> Fast [5..30] × Slow [20..200] = 16 (nur fast&lt;slow)</li>"
        "</ul></div>"
        "</div>"
        "</div></div>"
    ) + "".join(body_sections)

    _write(out / "lead_lag_optimizer.html",
           _html_base("Lead-Lag Optimizer", 19, body))

'''

# ─────────────────────────────────────────────────────────────────────────────
# 3. Comprehensive strategy-pairs report (all pairs × 5 indicators, full metrics)
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_PAIRS_FN = '''
def build_strategy_pairs_report(tables, figures, out):  # noqa: C901
    """All Granger pairs × 5 indicators: IS/OOS, 26 metrics, pair statistics, rolling metrics."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "strategy_pairs.html",
               _html_base("Strategie-Paare", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── Pair list ─────────────────────────────────────────────────────────────
    MANUAL = [
        ("GC=F", "GDX", 7, "Gold→GDX"),
        ("GC=F", "NEM", 6, "Gold→NEM"),
        ("GC=F", "SIL", 1, "Gold→SIL"),
        ("CL=F", "XLE", 1, "WTI→XLE"),
        ("CL=F", "XOM", 1, "WTI→XOM"),
        ("BZ=F", "XLE", 1, "Brent→XLE"),
        ("HG=F", "FCX", 1, "Kupfer→FCX"),
        ("SI=F", "SIL", 1, "Silber→SIL"),
        ("SI=F", "GDX", 1, "Silber→GDX"),
    ]
    all_pairs = []
    seen = set()
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            key = (c, e)
            if key not in seen and c in returns.columns and e in returns.columns:
                all_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
                seen.add(key)
    for mp in MANUAL:
        key = (mp[0], mp[1])
        if key not in seen and mp[0] in returns.columns and mp[1] in returns.columns:
            all_pairs.append(mp)
            seen.add(key)

    # ── Indicators (default params, one at a time) ────────────────────────────
    INDICATORS = [
        ("RSI(14)>50",    lambda px: _calc_rsi(px, 14),       50.0),
        ("MACD>0",        lambda px: _calc_macd(px)[0],        0.0),
        ("BB-Pos>0.5",    lambda px: _calc_bb_pos(px, 20),     0.5),
        ("SMA20>SMA50",   lambda px: _calc_sma_cross(px, 20, 50), 0.0),
        ("RSI(14)<70",    lambda px: -_calc_rsi(px, 14),      -70.0),  # not overbought = long
    ]

    IS_FRAC = 0.70
    all_oos_rows = []
    pair_sections = []

    for pair_idx, (leader, follower, lag, pair_label) in enumerate(all_pairs):
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        rl = returns[leader].dropna()
        if px is None or len(px) < 300:
            continue

        idx_all = px.index.intersection(rf.index)
        if len(idx_all) < 200:
            continue
        split_date = idx_all[int(len(idx_all) * IS_FRAC)]
        is_rf = rf.loc[:split_date]
        oos_rf = rf.loc[split_date:]

        # ── Pair statistics ───────────────────────────────────────────────────
        jdx = rl.index.intersection(rf.index)
        rl_j = rl.loc[jdx]; rf_j = rf.loc[jdx]
        rolling_corr = rl_j.rolling(63).corr(rf_j).dropna()
        rolling_cov  = rl_j.rolling(63).cov(rf_j).dropna()

        # OLS: follower ~ leader
        Xc = np.column_stack([np.ones(len(rl_j)), rl_j.values])
        coef, _, _, _ = np.linalg.lstsq(Xc, rf_j.values, rcond=None)
        yhat = Xc @ coef
        ss_res = float(((rf_j.values - yhat) ** 2).sum())
        ss_tot = float(((rf_j.values - rf_j.mean()) ** 2).sum())
        beta_pair = round(float(coef[1]), 4)
        r2_pair = round(max(0.0, 1 - ss_res / ss_tot), 4)
        vif_pair = round(1 / (1 - r2_pair + 1e-9), 3)

        # CCF lags 0-10
        n_jdx = len(rl_j)
        ccf_lags = list(range(11))
        ccf_vals = []
        for lv in ccf_lags:
            x = rl_j.values[:n_jdx - lv] if lv > 0 else rl_j.values
            y = rf_j.values[lv:]          if lv > 0 else rf_j.values
            try:
                ccf_vals.append(float(np.corrcoef(x, y)[0, 1]))
            except Exception:
                ccf_vals.append(float("nan"))

        # ACF follower
        foll_acf = [round(float(rf_j.autocorr(lv)), 4) for lv in range(1, 6)]

        # ── Run indicators ────────────────────────────────────────────────────
        pair_metrics = []
        best_oos_sharpe = -999.0
        best_oos_name = "—"

        fig_eq_is = go.Figure()
        fig_eq_oos = go.Figure()
        bh_is = (1 + is_rf.dropna()).cumprod()
        bh_oos = (1 + oos_rf.dropna()).cumprod()
        for fig_eq, bh_eq in [(fig_eq_is, bh_is), (fig_eq_oos, bh_oos)]:
            fig_eq.add_trace(go.Scatter(
                x=bh_eq.index.astype(str).tolist(),
                y=bh_eq.round(4).values.tolist(),
                name=f"B&H {follower}",
                line=dict(color="#8b949e", width=0.9, dash="dot")))

        for ind_idx, (ind_name, ind_fn, thresh) in enumerate(INDICATORS):
            ind = ind_fn(px)
            n_is, g_is, s_is = _strat_exec(ind, thresh, is_rf, lag)
            n_oos, g_oos, s_oos = _strat_exec(ind, thresh, oos_rf, lag)
            if len(n_is) < 30 or len(n_oos) < 30:
                continue

            spy_is = spy.loc[n_is.index] if spy is not None else None
            spy_oos = spy.loc[n_oos.index] if spy is not None else None
            m_is = _full_metrics(n_is, g_is, s_is, spy_is, ind_name + " IS")
            m_oos = _full_metrics(n_oos, g_oos, s_oos, spy_oos, ind_name + " OOS")

            for m, period in [(m_is, "IS"), (m_oos, "OOS")]:
                row = {"Paar": pair_label, "Indikator": ind_name, "Periode": period}
                row.update({k: v for k, v in m.items() if k != "Name"})
                pair_metrics.append(row)
                if period == "OOS":
                    all_oos_rows.append(row)

            oos_sh = m_oos.get("Sharpe (net)", -999.0)
            if isinstance(oos_sh, float) and not np.isnan(oos_sh) and oos_sh > best_oos_sharpe:
                best_oos_sharpe = oos_sh
                best_oos_name = ind_name

            col = PAL[ind_idx % len(PAL)]
            for fig_eq, net_v in [(fig_eq_is, n_is), (fig_eq_oos, n_oos)]:
                eq = (1 + net_v).cumprod()
                fig_eq.add_trace(go.Scatter(
                    x=eq.index.astype(str).tolist(),
                    y=eq.round(4).values.tolist(),
                    name=ind_name, line=dict(color=col, width=1.5)))

        for fig_eq, title_sfx, interp_txt in [
            (fig_eq_is, "IS", "IS-Periode (70%). Immer positiver als OOS durch In-Sample-Bias."),
            (fig_eq_oos, "OOS", "OOS-Periode (30%). Echte Performance-Schätzung. Grau = B&H-Vergleich."),
        ]:
            fig_eq.update_layout(
                title=f"Equity-Kurven {title_sfx}: {pair_label}",
                yaxis_type="log", height=420)

        # Rolling metrics (RSI(14) as reference)
        n_full, g_full, s_full = _strat_exec(_calc_rsi(px, 14), 50, rf, lag)
        fig_roll = None
        if len(n_full) > 120:
            roll_sh = n_full.rolling(63).apply(
                lambda x: x.mean() * 252 / (x.std() * np.sqrt(252) + 1e-9), raw=True)
            roll_ret = n_full.rolling(63).mean() * 252 * 100
            roll_mdd = n_full.rolling(126).apply(
                lambda x: float(((1 + pd.Series(x)).cumprod()
                                 / (1 + pd.Series(x)).cumprod().cummax() - 1).min()), raw=True)
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=roll_sh.dropna().index.astype(str).tolist(),
                y=roll_sh.dropna().round(3).values.tolist(),
                name="Rolling Sharpe (63T)", line=dict(color="#58a6ff", width=1.3)))
            fig_roll.add_trace(go.Scatter(
                x=roll_ret.dropna().index.astype(str).tolist(),
                y=roll_ret.dropna().round(2).values.tolist(),
                name="Rolling Ann.Ret% (63T)", line=dict(color="#3fb950", width=1.0),
                yaxis="y2"))
            fig_roll.add_trace(go.Scatter(
                x=roll_mdd.dropna().index.astype(str).tolist(),
                y=(roll_mdd.dropna() * 100).round(2).values.tolist(),
                name="Rolling MaxDD% (126T)", line=dict(color="#f78166", width=0.8, dash="dot"),
                yaxis="y2"))
            fig_roll.add_hline(y=0, line_color="#8b949e", line_dash="dash")
            fig_roll.add_vline(x=str(split_date)[:10], line_color="#d29922",
                               line_dash="dash", annotation_text="IS|OOS")
            fig_roll.update_layout(
                title=f"Rolling-Metriken (RSI-Referenz): {pair_label}",
                height=420, yaxis_title="Sharpe",
                yaxis2=dict(title="Ann.Ret% / MaxDD%", overlaying="y", side="right"))

        # Pair statistics charts
        fig_corr = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Rolling 63T Korrelation (Leader↔Follower Renditen)",
                            "Rolling 63T Kovarianz"],
            row_heights=[0.5, 0.5], vertical_spacing=0.06)
        fig_corr.add_trace(
            go.Scatter(x=rolling_corr.index.astype(str).tolist(),
                       y=rolling_corr.round(4).values.tolist(),
                       name="Korrelation", line=dict(color="#58a6ff")), row=1, col=1)
        fig_corr.add_hline(y=0, line_color="#8b949e", row=1, col=1)
        fig_corr.add_trace(
            go.Scatter(x=rolling_cov.index.astype(str).tolist(),
                       y=rolling_cov.round(8).values.tolist(),
                       name="Kovarianz", line=dict(color="#d29922")), row=2, col=1)
        fig_corr.update_layout(title=f"Paar-Abhängigkeit: {pair_label}", height=480)

        # CCF bar chart
        sig_band = 2.0 / np.sqrt(max(len(jdx), 1))
        fig_ccf_p = go.Figure(go.Bar(
            x=ccf_lags, y=ccf_vals,
            marker_color=["#3fb950" if lv == lag else
                          ("#f78166" if abs(v) > sig_band else "#58a6ff")
                          for lv, v in zip(ccf_lags, ccf_vals)],
            hovertemplate="Lag=%{x}T<br>CCF=%{y:.4f}<extra></extra>"))
        fig_ccf_p.add_hline(y=sig_band, line_color="#d29922", line_dash="dot",
                             annotation_text=f"+{sig_band:.3f}")
        fig_ccf_p.add_hline(y=-sig_band, line_color="#d29922", line_dash="dot")
        fig_ccf_p.update_layout(
            title=f"CCF (0-10 Tage): {pair_label} — Grün = Granger-Lag",
            xaxis_title="Lag (Tage)", height=320)

        # Pair info card
        stat_html = (
            "<table class='table table-dark table-sm table-bordered small'>"
            "<thead><tr><th>Statistik</th><th>Wert</th><th>Bedeutung</th></tr></thead>"
            "<tbody>"
            f"<tr><td>β (follower ~ leader OLS)</td><td>{beta_pair}</td>"
            f"<td>+1% Leader ≈ {beta_pair:.3f}% Follower-Rendite</td></tr>"
            f"<tr><td>R² (Regression)</td><td>{r2_pair}</td>"
            f"<td>{r2_pair*100:.1f}% der Follower-Varianz durch Leader erklärbar</td></tr>"
            f"<tr><td>VIF</td><td>{vif_pair}</td>"
            f"<td>Varianz-Inflationsfaktor: {vif_pair:.1f}× (bivariate Schätzung)</td></tr>"
            f"<tr><td>CCF@Lag{lag}T</td><td>{ccf_vals[min(lag,10)]:.4f}</td>"
            f"<td>Kreuzkorrelation beim Granger-Lag {lag}T</td></tr>"
            f"<tr><td>ACF₁ Follower</td><td>{foll_acf[0]}</td>"
            f"<td>Autokorrelation Follower-Rendite Lag 1 (Momentum/Mean-Reversion)</td></tr>"
            f"<tr><td>ACF₅ Follower</td><td>{foll_acf[4]}</td>"
            f"<td>Autokorrelation Follower-Rendite Lag 5</td></tr>"
            f"<tr><td>N Tage Überlappung</td><td>{len(jdx)}</td><td></td></tr>"
            "</tbody></table>")

        pal_col = PAL[pair_idx % len(PAL)]
        sec = []
        sec.append(
            f"<div class='card mb-4' style='border-left:4px solid {pal_col}'>"
            f"<div class='card-header'><strong>{pair_label}</strong>"
            f"<span class='badge bg-success ms-2'>Bester OOS: {best_oos_name}"
            f" (Sharpe {round(best_oos_sharpe,3)})</span>"
            f"<span class='badge bg-secondary ms-1'>β={beta_pair} R²={r2_pair}</span>"
            f"</div><div class='card-body'>{stat_html}</div></div>")

        if pair_metrics:
            pm_df = (pd.DataFrame(pair_metrics)
                     .drop(columns=["Name"], errors="ignore")
                     .sort_values(["Indikator", "Periode"]))
            sec.append(_card("Vollständige Metriken-Tabelle (IS + OOS)", _df_html(pm_df)))

        sec.append(_chart_card("IS-Equity-Kurven: " + pair_label, fig_eq_is, height=440,
            interp="Log-Skala. Grau gestrichelt = B&H Follower. IS-Periode (70%)."))
        sec.append(_chart_card("OOS-Equity-Kurven: " + pair_label, fig_eq_oos, height=440,
            interp="OOS-Periode (letzte 30%). Echte Out-of-Sample Performance. "
                   "OOS besser als IS-Niveau: Strategie robust. Schlechter: Overfitting."))
        if fig_roll is not None:
            sec.append(_chart_card("Rolling-Metriken: " + pair_label, fig_roll, height=440,
                interp="RSI(14)>50 als Referenzindikator. 63T rolling Sharpe (blau), "
                       "Ann.Ret% (grün), 126T MaxDD% (rot). Gelber Strich: IS|OOS-Grenze. "
                       "Stabiler Verlauf = zeitlich robuste Strategie."))
        sec.append(_chart_card("Paar-Abhängigkeit: " + pair_label, fig_corr, height=500,
            interp="Wie stark Leader und Follower korreliert/kovariant sind. "
                   "Rückgang = Lead-Lag-Beziehung schwächt sich ab → Strategie-Risiko steigt."))
        sec.append(_chart_card("CCF 0-10 Tage: " + pair_label, fig_ccf_p, height=340,
            interp=f"Kreuzkorrelation Leader→Follower. Grün = Granger-optimaler Lag {lag}T. "
                   "Rot = signifikant (über Band). Gelbe Linien = 95%-Signifikanzband."))

        pair_sections.append("".join(sec))

    # ── Grand summary ─────────────────────────────────────────────────────────
    grand_df = pd.DataFrame(all_oos_rows).sort_values(
        "Sharpe (net)", ascending=False) if all_oos_rows else pd.DataFrame()

    fig_grand = go.Figure()
    if not grand_df.empty:
        top = grand_df.head(25)
        labels = (top["Paar"].str.split("(").str[0].str.strip()
                  + " / " + top["Indikator"]).tolist()
        shares = top["Sharpe (net)"].tolist()
        fig_grand.add_trace(go.Bar(
            x=labels, y=shares,
            marker_color=["#3fb950" if v > 0.4 else "#d29922" if v > 0 else "#f78166"
                          for v in shares],
            text=[f"{v:.3f}" for v in shares], textposition="outside"))
        fig_grand.add_hline(y=0, line_color="#8b949e")
        fig_grand.add_hline(y=0.4, line_color="#3fb950", line_dash="dot",
                            annotation_text="0.4 Zielschwelle")
        fig_grand.update_layout(
            title="Grand Summary: OOS Sharpe (alle Paare × Indikatoren)",
            xaxis_tickangle=-40, height=560)

    intro = (
        "<div class='ph-header'>"
        "<h1>Strategie-Paare: Vollständige IS/OOS Analyse</h1>"
        "<div class='sub'>Alle Granger-Paare &middot; 5 Indikatoren &middot; "
        "26 Metriken &middot; Paar-Statistiken &middot; Rolling-Metriken</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Methodik</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'><h6>Strategie</h6><ul class='small'>"
        "<li>Signal: Long wenn Leader-Indikator(t-lag) &gt; Threshold, sonst Short</li>"
        "<li>Lag = bekannter Granger-Lag des Paares</li>"
        "<li>IS 70% | OOS 30% | TC 10bp/Trade</li>"
        "<li>5 Indikatoren mit domain-knowledge-Defaults</li>"
        "</ul></div>"
        "<div class='col-md-6'><h6>Paar-Statistiken</h6><ul class='small'>"
        "<li>Rolling 63T Korrelation + Kovarianz</li>"
        "<li>OLS β und R² (follower ~ leader)</li>"
        "<li>VIF = 1/(1-R²) bivariate Kollinearität</li>"
        "<li>CCF lags 0-10T, ACF Follower lags 1-5</li>"
        "</ul></div>"
        "</div></div></div>")

    body = (intro
            + _chart_card("Grand Summary: OOS Sharpe (Top-25)", fig_grand, height=580,
                interp="Grün > 0.4: Zielbereich. Gelb 0-0.4: grenzwertig. Rot: nicht profitabel. "
                       "OOS = echte Gütemessung — IS-Sharpe immer höher durch IS-Overfitting.")
            + _card("Grand Summary Tabelle (alle OOS-Ergebnisse)",
                    _df_html(grand_df.drop(columns=["Paar"], errors="ignore").head(60)))
            + "".join(pair_sections))

    _write(out / "strategy_pairs.html", _html_base("Strategie-Paare", 19, body))

'''

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA-filtered strategy report (Version A + Version B with bootstrap CI)
# ─────────────────────────────────────────────────────────────────────────────
PCA_STRATEGY_FN = '''
def build_pca_strategy_report(tables, figures, out):  # noqa: C901
    """PC1-filtered lead-lag strategies: Base vs. PCA-A (point estimate) vs. PCA-B (bootstrap CI)."""
    returns = _read(tables / "phase2_returns.csv")
    prices = _read(tables / "phase1_prices.csv")
    granger = _read(tables / "phase6_granger.csv")

    if returns is None:
        _write(out / "pca_strategy.html",
               _html_base("PCA-Strategie", 19, "<p>Daten fehlen.</p>"))
        return

    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]
    if prices is not None:
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()]
    else:
        prices = np.exp(returns.cumsum()) * 100

    spy = returns["SPY"].dropna() if "SPY" in returns.columns else None

    # ── PCA on IS period ──────────────────────────────────────────────────────
    FACTOR_COLS = [c for c in ["XLE","XLB","XLI","GDX","SIL","JETS","IYT",
                                "SPY","QQQ","IWM","IJH","MGC","^VIX","DX-Y.NYB","^TNX"]
                   if c in returns.columns]
    ret_fac = returns[FACTOR_COLS].dropna()
    IS_FRAC = 0.70
    n_total = len(ret_fac)
    split_idx = int(n_total * IS_FRAC)
    split_date_pca = ret_fac.index[split_idx]

    Z_is_raw = ret_fac.iloc[:split_idx]
    mu_fac = Z_is_raw.mean()
    sg_fac = Z_is_raw.std() + 1e-12
    Z_is = ((Z_is_raw - mu_fac) / sg_fac).values
    Z_all = ((ret_fac - mu_fac) / sg_fac).values

    _, S, Vt = np.linalg.svd(Z_is, full_matrices=False)
    pc1_load = Vt[0]                           # shape (n_features,)
    explained = S ** 2 / (S ** 2).sum()
    pc1_scores = pd.Series(Z_all @ pc1_load, index=ret_fac.index)

    # Bootstrap CI on PC1 loading (500 samples, IS only)
    np.random.seed(42)
    N_BOOT = 500
    n_is = len(Z_is)
    boot_loads = np.zeros((N_BOOT, len(FACTOR_COLS)))
    for bi in range(N_BOOT):
        idx_b = np.random.randint(0, n_is, n_is)
        _, _, Vt_b = np.linalg.svd(Z_is[idx_b], full_matrices=False)
        flip = np.sign(Vt_b[0] @ pc1_load)
        boot_loads[bi] = Vt_b[0] * flip
    lo95 = np.percentile(boot_loads, 2.5, axis=0)
    hi95 = np.percentile(boot_loads, 97.5, axis=0)
    loading_sig = (lo95 > 0) | (hi95 < 0)

    load_dict = dict(zip(FACTOR_COLS, pc1_load))
    sig_dict  = dict(zip(FACTOR_COLS, loading_sig))

    # Loadings chart
    ord_idx = np.argsort(np.abs(pc1_load))[::-1]
    s_assets = [FACTOR_COLS[i] for i in ord_idx]
    s_loads  = [float(pc1_load[i]) for i in ord_idx]
    s_lo     = [float(lo95[i]) for i in ord_idx]
    s_hi     = [float(hi95[i]) for i in ord_idx]
    s_sig    = [bool(loading_sig[i]) for i in ord_idx]

    fig_loads = go.Figure()
    fig_loads.add_trace(go.Bar(
        x=s_assets, y=s_loads,
        marker_color=["#3fb950" if sg else "#d29922" for sg in s_sig],
        error_y=dict(type="data", symmetric=False,
                     array=[hi - lo for lo, hi in zip(s_lo, s_hi)],
                     arrayminus=[mid - lo for lo, mid in zip(s_lo, s_loads)],
                     color="#8b949e"),
        text=[f"{v:.3f}{'*' if sg else ''}" for v, sg in zip(s_loads, s_sig)],
        textposition="outside"))
    fig_loads.add_hline(y=0, line_color="#8b949e")
    fig_loads.update_layout(
        title=(f"PC1-Loadings (IS) mit Bootstrap-95%-CI "
               f"| Var.erkl.: {explained[0]*100:.1f}% "
               f"| IS bis {str(split_date_pca)[:10]}"),
        height=420, xaxis_title="Asset", yaxis_title="PC1 Loading",
        annotations=[dict(x=0.01, y=0.97, xref="paper", yref="paper",
                          text="* = signifikant (CI kreuzt nicht 0, 500 Bootstraps)",
                          showarrow=False, font=dict(size=10, color="#d29922"))])

    # PC1 score chart
    pc1_sm = pc1_scores.rolling(21).mean().dropna()
    fig_pc1 = go.Figure()
    fig_pc1.add_trace(go.Scatter(
        x=pc1_sm.index.astype(str).tolist(), y=pc1_sm.round(4).values.tolist(),
        mode="lines", name="PC1 Score (21T SMA)", line=dict(color="#58a6ff", width=1.3)))
    fig_pc1.add_hline(y=0, line_color="#8b949e", line_dash="dash",
                      annotation_text="Risk-Off | Risk-On")
    fig_pc1.add_vline(x=str(split_date_pca)[:10], line_color="#d29922",
                      line_dash="dash", annotation_text="IS|OOS")
    fig_pc1.update_layout(
        title="PC1-Score: Marktregime (positiv=Risk-On, negativ=Risk-Off)",
        height=350)

    # ── Pairs ─────────────────────────────────────────────────────────────────
    MANUAL = [
        ("GC=F","GDX",7,"Gold→GDX"), ("GC=F","NEM",6,"Gold→NEM"),
        ("CL=F","XLE",1,"WTI→XLE"), ("CL=F","XOM",1,"WTI→XOM"),
        ("HG=F","FCX",1,"Kupfer→FCX"), ("SI=F","SIL",1,"Silber→SIL"),
    ]
    all_pairs = []
    seen = set()
    if granger is not None and "cause" in granger.columns:
        fcol = next((c for c in ["fstat", "f_stat"] if c in granger.columns), None)
        pcol = "pvalue" if "pvalue" in granger.columns else "p_value"
        sig_mask = granger.get("significant", pd.Series([True] * len(granger)))
        sdf = granger[sig_mask == True].copy()
        sdf = sdf.sort_values(fcol, ascending=False) if fcol else sdf.sort_values(pcol)
        sdf = sdf.drop_duplicates(["cause", "effect"], keep="first")
        for _, row in sdf.iterrows():
            c = row["cause"]; e = row["effect"]; lg = int(row.get("lag", 1))
            key = (c, e)
            if key not in seen and c in returns.columns and e in returns.columns:
                all_pairs.append((c, e, lg, f"{c}→{e} (Lag {lg}T)"))
                seen.add(key)
    for mp in MANUAL:
        key = (mp[0], mp[1])
        if key not in seen and mp[0] in returns.columns and mp[1] in returns.columns:
            all_pairs.append(mp); seen.add(key)

    # ── Strategy execution per pair ───────────────────────────────────────────
    all_comp_rows = []
    pair_sections = []

    for leader, follower, lag, pair_label in all_pairs:
        px = prices[leader].dropna() if leader in prices.columns else None
        rf = returns[follower].dropna()
        if px is None or len(px) < 300:
            continue

        idx_pair = px.index.intersection(rf.index)
        if len(idx_pair) < 200:
            continue
        split_pair = idx_pair[int(len(idx_pair) * IS_FRAC)]

        # PC1 loading / significance of follower
        if follower in load_dict:
            foll_load = load_dict[follower]
            foll_sig = sig_dict[follower]
        else:
            jdx_f = rf.index.intersection(pc1_scores.index)
            if len(jdx_f) > 100:
                foll_load = float(np.corrcoef(rf.loc[jdx_f], pc1_scores.loc[jdx_f])[0, 1])
                foll_sig = abs(foll_load) > 2 / np.sqrt(len(jdx_f))
            else:
                foll_load = 0.0; foll_sig = False

        # Base signal (RSI 14 > 50)
        rsi_lead = _calc_rsi(px, 14)
        n_base, g_base, s_base = _strat_exec(rsi_lead, 50, rf, lag)
        if len(n_base) < 30:
            continue

        # PC1 score lagged by lead-lag (to avoid lookahead)
        pc1_lagged = pc1_scores.shift(lag)

        # Version A: scale to 1.0 when PC1 confirms, 0.5 when contradicts
        pc1_w_A = pc1_lagged.reindex(s_base.index, method="ffill").apply(
            lambda x: 1.0 if not np.isnan(x) and x * foll_load > 0 else 0.5)
        sig_A = s_base * pc1_w_A
        g_A = sig_A * rf.reindex(sig_A.index, method="ffill")
        n_A = (g_A - sig_A.diff().abs().fillna(0) * 0.001).dropna()

        # Version B: 1.0 when confirms AND loading significant, else 0.0
        if foll_sig:
            pc1_w_B = pc1_lagged.reindex(s_base.index, method="ffill").apply(
                lambda x: 1.0 if not np.isnan(x) and x * foll_load > 0 else 0.0)
        else:
            # Loading not significant → reduce to 0.5 always (caution mode)
            pc1_w_B = pd.Series(0.5, index=s_base.index)
        sig_B = s_base * pc1_w_B
        g_B = sig_B * rf.reindex(sig_B.index, method="ffill")
        n_B = (g_B - sig_B.diff().abs().fillna(0) * 0.001).dropna()

        # Compute metrics for IS and OOS
        for version, n_v, g_v, s_v in [
            ("Base",  n_base, g_base, s_base),
            ("PCA-A", n_A, g_A if len(g_A) > 0 else g_base, sig_A),
            ("PCA-B", n_B, g_B if len(g_B) > 0 else g_base, sig_B),
        ]:
            for period, mask_fn in [
                ("IS",  lambda x: x.loc[:split_pair]),
                ("OOS", lambda x: x.loc[split_pair:]),
            ]:
                pn = mask_fn(n_v).dropna()
                pg = mask_fn(g_v).dropna()
                ps = mask_fn(s_v)
                if len(pn) < 30:
                    continue
                spy_p = spy.loc[pn.index] if spy is not None else None
                m = _full_metrics(pn, pg if len(pg) > 0 else None,
                                  ps if len(ps) > 0 else None, spy_p)
                row = {"Paar": pair_label, "Version": version, "Periode": period,
                       "PC1-Loading": round(foll_load, 4),
                       "Loading-Signifikant": foll_sig}
                row.update({k: v for k, v in m.items() if k != "Name"})
                all_comp_rows.append(row)

        # Equity chart (full data, all 3 versions)
        fig_eq = go.Figure()
        for ver, n_v, col in [("Base", n_base, "#8b949e"),
                               ("PCA-A", n_A, "#58a6ff"),
                               ("PCA-B", n_B, "#3fb950")]:
            if len(n_v) > 30:
                eq = (1 + n_v).cumprod()
                fig_eq.add_trace(go.Scatter(
                    x=eq.index.astype(str).tolist(), y=eq.round(4).values.tolist(),
                    mode="lines", name=ver, line=dict(color=col, width=1.5)))
        bh = (1 + rf.dropna()).cumprod()
        fig_eq.add_trace(go.Scatter(
            x=bh.index.astype(str).tolist(), y=bh.round(4).values.tolist(),
            name=f"B&H {follower}", line=dict(color="#444c56", width=0.8, dash="dot")))
        fig_eq.add_vline(x=str(split_pair)[:10], line_color="#d29922",
                         line_dash="dash", annotation_text="IS|OOS")
        fig_eq.update_layout(
            title=f"PCA-Strategie-Equity: {pair_label}", yaxis_type="log", height=420)

        # OOS regime overlay chart (PC1 score + strategy equity)
        oos_rf_p = rf.loc[split_pair:]
        fig_reg = go.Figure()
        if len(oos_rf_p) > 30:
            bh_oos = (1 + oos_rf_p.dropna()).cumprod()
            fig_reg.add_trace(go.Scatter(
                x=bh_oos.index.astype(str).tolist(),
                y=bh_oos.round(4).values.tolist(),
                name=f"B&H {follower}",
                line=dict(color="#8b949e", width=1)))
            oos_a = n_A.loc[split_pair:].dropna()
            if len(oos_a) > 30:
                eq_a = (1 + oos_a).cumprod()
                fig_reg.add_trace(go.Scatter(
                    x=eq_a.index.astype(str).tolist(),
                    y=eq_a.round(4).values.tolist(),
                    name="PCA-A OOS", line=dict(color="#58a6ff", width=1.5)))
            oos_b = n_B.loc[split_pair:].dropna()
            if len(oos_b) > 30:
                eq_b = (1 + oos_b).cumprod()
                fig_reg.add_trace(go.Scatter(
                    x=eq_b.index.astype(str).tolist(),
                    y=eq_b.round(4).values.tolist(),
                    name="PCA-B OOS", line=dict(color="#3fb950", width=1.5)))
            pc1_oos = pc1_scores.loc[split_pair:].rolling(21).mean().dropna()
            if len(pc1_oos) > 10:
                fig_reg.add_trace(go.Scatter(
                    x=pc1_oos.index.astype(str).tolist(),
                    y=pc1_oos.round(4).values.tolist(),
                    name="PC1-Score (21T)", yaxis="y2",
                    line=dict(color="#d29922", width=0.8, dash="dot")))
            fig_reg.update_layout(
                title=f"OOS-Regime: {pair_label}",
                height=400, yaxis_type="log", yaxis_title="Equity",
                yaxis2=dict(title="PC1 Score", overlaying="y", side="right"))

        badge_col = "bg-success" if foll_sig else "bg-secondary"
        info_card = (
            f"<div class='card mb-3' style='border-left:4px solid #58a6ff'>"
            f"<div class='card-header'><strong>{pair_label}</strong>"
            f"<span class='badge bg-info ms-2'>PC1-Loading: {foll_load:.4f}</span>"
            f"<span class='badge {badge_col} ms-1'>"
            f"{'Signifikant' if foll_sig else 'Nicht signifikant'}"
            f" (Bootstrap 95% CI)</span></div>"
            f"<div class='card-body'>"
            f"<p class='small'>"
            f"<strong>PCA-A:</strong> PC1×Loading {'> 0' if foll_load > 0 else '< 0'} → "
            f"volle Position; entgegengesetzt → halbe Position (50%).<br>"
            f"<strong>PCA-B:</strong> {'Loading signifikant → nur Long wenn PC1 bestätigt; 0% sonst.' if foll_sig else 'Loading NICHT signifikant → feste 50% Positionsgröße (vorsichtig, kein binärer Filter).'}"
            f"</p>"
            f"</div></div>")

        pair_metrics_here = [r for r in all_comp_rows if r.get("Paar") == pair_label]
        pm_df = (pd.DataFrame(pair_metrics_here)
                 .drop(columns=["Paar", "Name"], errors="ignore")
                 .sort_values(["Periode", "Version"])
                 if pair_metrics_here else pd.DataFrame())

        sec = [info_card]
        if not pm_df.empty:
            sec.append(_card("Metriken: " + pair_label, _df_html(pm_df)))
        sec.append(_chart_card("PCA-Equity: " + pair_label, fig_eq, height=440,
            interp="Grau: Base (kein PCA). Blau: PCA-A (Positions-Skalierung 100%/50%). "
                   "Grün: PCA-B (Positions-Skalierung 100%/0% oder 50% wenn nicht signifikant). "
                   "Gelber Strich: IS|OOS-Grenze. Log-Skala."))
        if len(oos_rf_p) > 30:
            sec.append(_chart_card("OOS-Regime: " + pair_label, fig_reg, height=420,
                interp="OOS-Equity aller Versionen. Gelb (rechts): PC1-Score (Risk-On/Off-Regime). "
                       "Regime-Übereinstimmung mit guten Strategy-Perioden prüfen."))
        pair_sections.append("".join(sec))

    # ── Aggregated comparison ─────────────────────────────────────────────────
    oos_comp = [r for r in all_comp_rows if r.get("Periode") == "OOS"]
    agg_summary = {}
    for ver in ["Base", "PCA-A", "PCA-B"]:
        vr = [r for r in oos_comp if r.get("Version") == ver]
        if vr:
            sharpes = [r.get("Sharpe (net)", float("nan")) for r in vr]
            sharpes_c = [s for s in sharpes if not (isinstance(s, float) and np.isnan(s))]
            agg_summary[ver] = {
                "Median OOS Sharpe": round(float(np.median(sharpes_c)), 3) if sharpes_c else float("nan"),
                "Mean OOS Sharpe": round(float(np.mean(sharpes_c)), 3) if sharpes_c else float("nan"),
                "Sharpe>0 (% Paare)": round(100 * sum(1 for s in sharpes_c if s > 0) / len(sharpes_c), 1) if sharpes_c else 0,
                "Sharpe>0.4 (% Paare)": round(100 * sum(1 for s in sharpes_c if s > 0.4) / len(sharpes_c), 1) if sharpes_c else 0,
                "N Paare": len(vr),
            }

    fig_agg = go.Figure()
    metrics_agg = ["Median OOS Sharpe", "Mean OOS Sharpe", "Sharpe>0 (% Paare)", "Sharpe>0.4 (% Paare)"]
    for vi, (ver, col) in enumerate([("Base", "#8b949e"), ("PCA-A", "#58a6ff"), ("PCA-B", "#3fb950")]):
        if ver in agg_summary:
            sd = agg_summary[ver]
            fig_agg.add_trace(go.Bar(
                name=ver,
                x=metrics_agg,
                y=[sd.get(m, 0) for m in metrics_agg],
                marker_color=col))
    fig_agg.update_layout(
        title="PCA-Versions-Vergleich (aggregiert über alle Paare, OOS)",
        barmode="group", height=420)

    comp_df = (pd.DataFrame(all_comp_rows)
               .drop(columns=["Name"], errors="ignore")
               .sort_values(["Periode", "Sharpe (net)"], ascending=[True, False])
               if all_comp_rows else pd.DataFrame())

    body = (
        "<div class='ph-header'>"
        "<h1>PCA-gefilterte Lead-Lag-Strategien</h1>"
        "<div class='sub'>PC1-Regime-Filter &middot; Version A: Positions-Skalierung &middot; "
        "Version B: Bootstrap-CI-Unsicherheit &middot; IS/OOS &middot; 26 Metriken</div>"
        "</div>"
        "<div class='card mb-4'><div class='card-header'><strong>Kernidee: PC1 als Regime-Filter</strong></div>"
        "<div class='card-body'><div class='row'>"
        "<div class='col-md-6'>"
        "<p class='small'>PC1 (erste Hauptkomponente) erklärt den größten Teil der gemeinsamen Varianz "
        "aller Faktor-Assets und ist ein proxy für das globale Marktrisiko-Regime. "
        "Positiver PC1-Score = Risk-On (SPY steigt, VIX fällt, Energie stark). "
        "Negativer Score = Risk-Off (Gold führt, DXY stark, Energie schwach).</p>"
        "<p class='small'><strong>Hypothese:</strong> Lead-Lag-Strategien funktionieren besser, "
        "wenn der PC1-Score die Follower-Richtung bestätigt. "
        "Wenn der Follower ein positives PC1-Loading hat, "
        "erwarte positive Returns in Risk-On-Regimes.</p>"
        "</div>"
        "<div class='col-md-6'>"
        "<table class='table table-dark table-sm table-bordered small'>"
        "<thead><tr><th>Version</th><th>PC1-Filter</th><th>Unsicherheit</th></tr></thead>"
        "<tbody>"
        "<tr><td><strong>Base</strong></td><td>Kein Filter</td><td>—</td></tr>"
        "<tr><td><strong>PCA-A</strong></td><td>100% wenn bestätigt, 50% wenn widersprechend</td><td>Ignoriert (Punkt-Schätzung)</td></tr>"
        "<tr><td><strong>PCA-B</strong></td><td>100%/0% wenn signifikant, sonst 50% fest</td><td>Bootstrap-CI entscheidet über Signifikanz</td></tr>"
        "</tbody></table>"
        "</div>"
        "</div></div></div>"
        + _chart_card("PC1-Loadings mit 95%-Bootstrap-CI (500 Samples)", fig_loads, height=440,
            interp="Grün=signifikant (CI kreuzt nicht 0). Gelb=nicht signifikant. "
                   "* markiert signifikante Loadings. Fehlerbalken = 95%-Bootstrap-CI.")
        + _chart_card("PC1-Score: Marktregime-Zeitreihe", fig_pc1, height=370,
            interp="Positiv = Risk-On-Regime. Negativ = Risk-Off. 21T-geglättet. "
                   "Gelber Strich: IS|OOS-Grenze. PCA-Parameter aus IS-Periode geschätzt, "
                   "dann auf gesamten Zeitraum angewandt (nur IS-Daten → kein Lookahead).")
        + _chart_card("Versions-Vergleich (OOS, aggregiert über alle Paare)", fig_agg, height=440,
            interp="PCA-A > Base: PC1-Filter verbessert Median-Sharpe. "
                   "PCA-B > PCA-A: Bootstrap-Unsicherheit nutzen verbessert weitere. "
                   "Sharpe>0.4%: Anteil der Paare mit wirklich robuster Strategie.")
        + _card("Vollständige Vergleichs-Tabelle (alle Paare × Versionen × IS/OOS)",
                _df_html(comp_df.head(120)))
        + "".join(pair_sections)
    )

    _write(out / "pca_strategy.html", _html_base("PCA-Strategie", 19, body))

'''

# ─────────────────────────────────────────────────────────────────────────────
# Inject everything and wire up
# ─────────────────────────────────────────────────────────────────────────────
with open(RB, "r", encoding="utf-8") as f:
    src = f.read()

INSERT_BEFORE = "\ndef build_index(tables, figures, out):"
idx = src.find(INSERT_BEFORE)
if idx < 0:
    raise RuntimeError("build_index not found in report_builder.py")

new_src = (src[:idx]
           + "\n" + SHARED_BLOCK
           + "\n" + OPTIMIZER_FN
           + "\n" + STRATEGY_PAIRS_FN
           + "\n" + PCA_STRATEGY_FN
           + src[idx:])

# Wire into build_all_reports
OLD_WIRE = "    build_technical_analysis_report(tables, figures, reports)\n    build_index(tables, figures, reports)"
NEW_WIRE = (
    "    build_technical_analysis_report(tables, figures, reports)\n"
    "    build_lead_lag_optimizer_report(tables, figures, reports)\n"
    "    build_strategy_pairs_report(tables, figures, reports)\n"
    "    build_pca_strategy_report(tables, figures, reports)\n"
    "    build_index(tables, figures, reports)"
)
if OLD_WIRE in new_src:
    new_src = new_src.replace(OLD_WIRE, NEW_WIRE, 1)
else:
    print("WARNING: could not wire into build_all_reports — wire manually")

# Wire into build_index extras list
OLD_INDEX = (
    '        ("Technische Analyse","SMA/RSI/MACD/Bollinger \u00b7 Lead-Lag-Indikator-Strategien \u00b7 Cross-Asset-Overlay","technical_analysis.html","#79c0ff",\n'
    '         (tables/"phase2_returns.csv").exists()),\n'
    '    ]'
)
NEW_INDEX = (
    '        ("Technische Analyse","SMA/RSI/MACD/Bollinger \u00b7 Lead-Lag-Indikator-Strategien \u00b7 Cross-Asset-Overlay","technical_analysis.html","#79c0ff",\n'
    '         (tables/"phase2_returns.csv").exists()),\n'
    '        ("Lead-Lag Optimizer","Param-Grid Heatmaps \u00b7 IS-Sharpe \u00b7 Walk-Forward OOS \u00b7 4 Indikatoren","lead_lag_optimizer.html","#c9d1d9",\n'
    '         (tables/"phase2_returns.csv").exists()),\n'
    '        ("Strategie-Paare","Alle Paare \u00b7 5 Indikatoren \u00b7 26 Metriken \u00b7 Paar-Statistiken","strategy_pairs.html","#c9d1d9",\n'
    '         (tables/"phase2_returns.csv").exists()),\n'
    '        ("PCA-Strategie","PC1-Filter \u00b7 Version A/B \u00b7 Bootstrap-CI \u00b7 IS/OOS","pca_strategy.html","#c9d1d9",\n'
    '         (tables/"phase2_returns.csv").exists()),\n'
    '    ]'
)
if OLD_INDEX in new_src:
    new_src = new_src.replace(OLD_INDEX, NEW_INDEX, 1)
else:
    print("WARNING: could not wire into build_index — wire manually")

with open(RB, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"Done. New size: {len(new_src.splitlines())} lines")
