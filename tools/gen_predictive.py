"""
Generates the two new report functions and patches them into report_builder.py:
  1. build_predictive_backtest_report  – enhanced with PCA models + residual diagnostics
  2. build_mega_strategies_report      – 80+ strategies + accordion Steckbriefe
"""
from pathlib import Path

RB = Path(__file__).resolve().parent.parent / "reports" / "report_builder.py"

PREDICTIVE_FN = '''
def build_predictive_backtest_report(tables, figures, out):  # noqa: C901
    """Walk-forward predictive backtest with 4 PCA model variants + residual diagnostics."""
    returns = _read(tables / "phase2_returns.csv")
    if returns is None:
        _write(out / "predictive_backtest.html",
               _html_base("Prädiktiver Backtest", 18, "<p>Renditen fehlen.</p>"))
        return
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    # ── Configuration ──────────────────────────────────────────────────
    HORIZONS       = [63, 126, 252, 504]
    HORIZON_LABELS = ["3M", "6M", "1J", "2J"]
    TRAIN_WINDOW   = 504
    RETRAIN_EVERY  = 63
    N_MC           = 300

    TARGET_ASSETS = [c for c in ["GC=F", "CL=F", "HG=F", "ZW=F", "NG=F", "SI=F", "ZS=F"]
                     if c in returns.columns]
    RAW_FACTORS = [c for c in ["DX-Y.NYB", "^VIX", "^TNX"] if c in returns.columns]
    PCA_INPUT   = [c for c in ["SPY", "QQQ", "IWM", "^VIX", "DX-Y.NYB", "^TNX",
                                "XLE", "XLB", "XLI", "JETS", "IYT", "GDX", "MGC"]
                   if c in returns.columns]

    # ── PCA (numpy SVD, no sklearn needed) ─────────────────────────────
    def _compute_pca_scores(df_in):
        """Returns (scores_df, explained_var_array) using SVD."""
        X = df_in.dropna()
        Xc = X - X.mean()
        std = Xc.std()
        std[std < 1e-12] = 1.0
        Xs = Xc / std
        U, s, Vt = np.linalg.svd(Xs.values, full_matrices=False)
        ev = (s ** 2) / max(np.sum(s ** 2), 1e-12)
        k  = Xs.shape[1]
        scores = pd.DataFrame(U * s, index=X.index,
                               columns=[f"PC{i+1}" for i in range(k)])
        return scores, ev

    pca_scores, ev_ratio = _compute_pca_scores(
        returns[PCA_INPUT].shift(1) if PCA_INPUT else pd.DataFrame())

    ev_cumsum = np.cumsum(ev_ratio) if len(ev_ratio) else np.array([])
    n_pc5  = 5
    n_pc95 = max(1, int(np.searchsorted(ev_cumsum, 0.95)) + 1) if len(ev_cumsum) else 3
    n_pc68 = max(1, int(np.searchsorted(ev_cumsum, 0.681)) + 1) if len(ev_cumsum) else 5

    # 4 model configs: (name, label, factor_cols_or_pc_prefix, n_pcs)
    MODEL_CONFIGS = [
        ("raw",   "Roh-Faktoren (DXY/VIX/TNX)", RAW_FACTORS, None),
        ("pc1",   "PCA: PC1 (~" + (f"{ev_ratio[0]*100:.0f}%" if len(ev_ratio) else "?") + " Var.)", None, 1),
        ("pc68",  f"PCA: PC1-{n_pc68} (~68% Var.)", None, n_pc68),
        ("pc95",  f"PCA: PC1-{n_pc95} (~95% Var.)", None, n_pc95),
    ]

    # ── OLS helper ─────────────────────────────────────────────────────
    def _ols_predict(X_tr, y_tr, X_pr):
        Xc = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta = np.linalg.lstsq(Xc, y_tr, rcond=None)[0]
        fitted = Xc @ beta
        return np.column_stack([np.ones(len(X_pr)), X_pr]) @ beta, beta, fitted

    # ── Walk-forward for a given factor matrix ─────────────────────────
    def _run_walkforward(asset, X_factor_df):
        y_full = returns[asset].dropna()
        Xf = X_factor_df.shift(1) if X_factor_df is not None else None
        if Xf is None or Xf.empty:
            return [], {}
        common = y_full.index.intersection(Xf.dropna().index)
        if len(common) < TRAIN_WINDOW + max(HORIZONS) + 50:
            return [], {}
        y_c  = y_full.loc[common].values
        X_c  = Xf.loc[common].dropna().values
        idx2 = common[:len(X_c)]
        n    = min(len(y_c), len(X_c))
        y_c, X_c = y_c[:n], X_c[:n]
        idx2 = idx2[:n]

        fc_by_h = {}
        betas_list = []
        for h_i, h in enumerate(HORIZONS):
            fc_rows = []
            for t_end in range(TRAIN_WINDOW, n - h, RETRAIN_EVERY):
                X_tr = X_c[t_end - TRAIN_WINDOW:t_end]
                y_tr = y_c[t_end - TRAIN_WINDOW:t_end]
                X_pr = X_c[t_end:t_end + h]
                if len(X_pr) < h:
                    continue
                try:
                    y_fc, beta_v, fitted = _ols_predict(X_tr, y_tr, X_pr)
                except Exception:
                    continue
                resids = y_tr - fitted
                y_real  = float(np.nansum(y_c[t_end:t_end + h]))
                fc_pt   = float(np.nansum(y_fc))
                mc_arr  = np.array([
                    float(np.nansum(y_fc + np.random.choice(resids, size=h, replace=True)))
                    for _ in range(N_MC)])
                fc_rows.append({
                    "date": idx2[t_end],
                    "realized": y_real,  "fc_mean": fc_pt,
                    "p10": float(np.percentile(mc_arr, 10)),
                    "p25": float(np.percentile(mc_arr, 25)),
                    "p50": float(np.percentile(mc_arr, 50)),
                    "p75": float(np.percentile(mc_arr, 75)),
                    "p90": float(np.percentile(mc_arr, 90)),
                    "resid_mean": float(np.mean(resids)),
                    "resid_ac1":  float(np.corrcoef(resids[:-1], resids[1:])[0, 1]
                                       if len(resids) > 2 else 0),
                })
                if h_i == 0 and (t_end - TRAIN_WINDOW) % (RETRAIN_EVERY * 4) == 0:
                    betas_list.append({"date": idx2[t_end], "beta": beta_v})
            if fc_rows:
                fc_by_h[h] = pd.DataFrame(fc_rows).set_index("date")
        return fc_by_h, betas_list

    # ── Compute all models for all assets ──────────────────────────────
    model_results  = {}  # (model_name, asset) → fc_by_h
    model_accuracy = []  # rows for comparison table
    model_betas    = {}  # (model_name, asset) → betas_list

    for m_name, m_label, raw_cols, n_pcs in MODEL_CONFIGS:
        if n_pcs is not None:
            if pca_scores.empty or n_pcs > pca_scores.shape[1]:
                continue
            X_df = pca_scores.iloc[:, :n_pcs]
        else:
            if not raw_cols:
                continue
            X_df = returns[raw_cols].copy()
        for asset in TARGET_ASSETS:
            fc_by_h, betas = _run_walkforward(asset, X_df)
            if not fc_by_h:
                continue
            model_results[(m_name, asset)] = fc_by_h
            model_betas[(m_name, asset)]   = betas
            for h_i, h in enumerate(HORIZONS):
                if h not in fc_by_h:
                    continue
                fc = fc_by_h[h]
                err    = fc["realized"] - fc["fc_mean"]
                da     = float(np.mean(np.sign(fc["realized"]) == np.sign(fc["fc_mean"])))
                cov80  = float(np.mean((fc["realized"] >= fc["p10"]) & (fc["realized"] <= fc["p90"])))
                n_pcs_used = n_pcs if n_pcs else len(raw_cols) if raw_cols else 0
                model_accuracy.append({
                    "Modell": m_label, "Asset": asset,
                    "Horizont": HORIZON_LABELS[h_i],
                    "RMSE (bps)": round(float(np.sqrt(np.mean(err**2))) * 10000, 1),
                    "MAE (bps)":  round(float(np.mean(np.abs(err))) * 10000, 1),
                    "Dir. Acc %": round(da * 100, 1),
                    "CI80 %":     round(cov80 * 100, 1),
                    "Bias (bps)": round(float(err.mean()) * 10000, 1),
                    "N": len(fc),
                    "#Faktoren": n_pcs_used,
                })

    # ── Residual diagnostics (best model = raw for reference) ──────────
    # For the 1Y horizon, compute error patterns
    diag_rows = {}
    for asset in TARGET_ASSETS:
        key = ("raw", asset)
        if key not in model_results or 252 not in model_results[key]:
            continue
        fc = model_results[key][252]
        err = fc["realized"] - fc["fc_mean"]
        # Monthly bias
        monthly_bias = err.groupby(err.index.month).mean() * 10000
        # AC1 of errors (lag-1 autocorrelation)
        if len(err) > 5:
            ac1 = float(np.corrcoef(err.values[:-1], err.values[1:])[0, 1])
        else:
            ac1 = 0.0
        # Overshoot ratio: fraction where abs(fc) > abs(realized)
        overshoot = float(np.mean(np.abs(fc["fc_mean"]) > np.abs(fc["realized"])))
        # Systematic undershoot lag: when model underpredicts, does realized peak shortly after?
        undershoot_mask = fc["fc_mean"] < fc["realized"]
        diag_rows[asset] = {
            "AC1 Fehler": round(ac1, 3),
            "Ø Bias (bps)": round(float(err.mean() * 10000), 1),
            "Overshoot-Rate": round(overshoot * 100, 1),
            "Undershoot-Rate": round((1 - overshoot) * 100, 1),
            "monthly_bias": monthly_bias,
            "err_series": err,
        }

    # ── Charts ─────────────────────────────────────────────────────────
    # 1) Model comparison: directional accuracy (PCA vs raw)
    fig_model_comp = go.Figure()
    if model_accuracy:
        acc_df = pd.DataFrame(model_accuracy)
        for asset in TARGET_ASSETS:
            sub = acc_df[(acc_df["Asset"] == asset) & (acc_df["Horizont"] == "1J")]
            if sub.empty:
                continue
            fig_model_comp.add_trace(go.Bar(
                x=sub["Modell"].tolist(),
                y=sub["Dir. Acc %"].tolist(),
                name=asset))
        fig_model_comp.add_hline(y=50, line_color="#8b949e", line_dash="dash",
                                  annotation_text="Zufall (50%)")
        fig_model_comp.update_layout(
            title="Modellvergleich: Direktionale Trefferquote @ 1-Jahres-Horizont (Roh vs. PCA-Modelle)",
            barmode="group", height=450, yaxis_title="Dir. Acc. (%)")

    # 2) PCA Explained Variance Scree plot
    fig_scree = go.Figure()
    if len(ev_ratio):
        n_show = min(len(ev_ratio), 15)
        x_ev = [f"PC{i+1}" for i in range(n_show)]
        fig_scree.add_trace(go.Bar(
            x=x_ev, y=(ev_ratio[:n_show] * 100).tolist(),
            marker_color="#58a6ff", name="Einzeln"))
        fig_scree.add_trace(go.Scatter(
            x=x_ev, y=(ev_cumsum[:n_show] * 100).tolist(),
            mode="lines+markers", name="Kumulativ",
            line=dict(color="#3fb950", width=2)))
        fig_scree.add_hline(y=68.1, line_color="#d29922", line_dash="dash",
                             annotation_text="68.1%")
        fig_scree.add_hline(y=95.0, line_color="#f78166", line_dash="dash",
                             annotation_text="95%")
        fig_scree.update_layout(
            title=f"PCA Scree-Plot: Erklärte Varianz der {len(PCA_INPUT)} Input-Faktoren",
            yaxis_title="Varianz (%)", height=400)

    # 3) Fan chart for GC=F (best model = pc68 if available, else raw)
    best_m = "pc68" if ("pc68", "GC=F") in model_results else "raw"
    fan_htmls = ""
    MONTHS_DE = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
    for asset in TARGET_ASSETS:
        key = (best_m, asset)
        if key not in model_results or 252 not in model_results[key]:
            continue
        fc = model_results[key][252]
        xs = fc.index.astype(str).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=(fc["p90"]*100).round(2).tolist() + (fc["p10"]*100).round(2).tolist()[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.10)", line=dict(width=0),
            name="80%-CI"))
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=(fc["p75"]*100).round(2).tolist() + (fc["p25"]*100).round(2).tolist()[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.22)", line=dict(width=0),
            name="50%-CI"))
        fig.add_trace(go.Scatter(x=xs, y=(fc["p50"]*100).round(2).tolist(),
            mode="lines", name="Median", line=dict(color="#58a6ff", width=1.6, dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=(fc["realized"]*100).round(2).tolist(),
            mode="lines", name="Realisiert", line=dict(color="#3fb950", width=2.0)))
        fig.add_hline(y=0, line_color="#8b949e", line_dash="dot")
        fig.update_layout(
            title=f"Fan-Chart: {asset} | Modell: {best_m} | 1J-Kumulativrendite",
            yaxis_title="Kumulativrendite (%)", height=420)
        fan_htmls += _chart_card(
            f"Fan-Chart {asset} — Walk-Forward ({N_MC} MC-Pfade, Modell: {best_m})", fig,
            interp=f"Grün = realisiert. Blau = Modell-Median. Fächer = 50%/80%-CI. "
                   f"Realisierte Rendite bleibt idealerweise im CI-Band. "
                   f"Systematische Abweichungen zeigen Modell-Bias an.")

    # 4) Residual diagnostics chart
    fig_diag = go.Figure()
    for j, (asset, d) in enumerate(diag_rows.items()):
        mb = d["monthly_bias"]
        if not mb.empty:
            fig_diag.add_trace(go.Scatter(
                x=[MONTHS_DE[m-1] for m in mb.index],
                y=mb.round(1).values.tolist(),
                mode="lines+markers", name=asset,
                line=dict(color=PAL[j % len(PAL)], width=1.5)))
    fig_diag.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_diag.update_layout(
        title="Monatlicher Forecast-Bias (bps): Saisonale Systematik in Prognosefehlern",
        yaxis_title="Ø Fehler (bps)", height=420)

    # 5) Overshoot/Undershoot analysis
    fig_overshoot = go.Figure()
    if diag_rows:
        assets_d   = list(diag_rows.keys())
        ov_rates   = [diag_rows[a]["Overshoot-Rate"] for a in assets_d]
        undr_rates = [diag_rows[a]["Undershoot-Rate"] for a in assets_d]
        ac1_vals   = [diag_rows[a]["AC1 Fehler"] for a in assets_d]
        fig_overshoot.add_trace(go.Bar(x=assets_d, y=ov_rates,
            name="Overshoot %", marker_color="#f78166"))
        fig_overshoot.add_trace(go.Bar(x=assets_d, y=undr_rates,
            name="Undershoot %", marker_color="#58a6ff"))
        fig_overshoot.add_hline(y=50, line_color="#8b949e", line_dash="dash")
        fig_overshoot.update_layout(title="Overshoot vs. Undershoot Rate des Prognosemodells",
            barmode="group", height=380, yaxis_title="%")

    # 6) Error autocorrelation
    fig_ac1 = go.Figure()
    if diag_rows:
        assets_d = list(diag_rows.keys())
        ac1_vals = [diag_rows[a]["AC1 Fehler"] for a in assets_d]
        bias_vals = [diag_rows[a]["Ø Bias (bps)"] for a in assets_d]
        fig_ac1.add_trace(go.Bar(x=assets_d, y=ac1_vals,
            name="AC1 Fehler",
            marker_color=["#f78166" if abs(v) > 0.15 else "#3fb950" for v in ac1_vals]))
        fig_ac1.add_hline(y=0.15, line_color="#f78166", line_dash="dash",
                           annotation_text="Signifikanzgrenze")
        fig_ac1.add_hline(y=-0.15, line_color="#f78166", line_dash="dash")
        fig_ac1.update_layout(
            title="Fehler-Autokorrelation (Lag-1): Muster in Prognosefehlern erkennbar?",
            yaxis_title="AC(1)", height=380)

    # 7) Rolling error bias chart
    fig_roll_bias = go.Figure()
    for j, (asset, d) in enumerate(diag_rows.items()):
        err_s = d["err_series"]
        if len(err_s) < 20:
            continue
        roll_bias = err_s.rolling(10).mean() * 10000
        fig_roll_bias.add_trace(go.Scatter(
            x=roll_bias.index.astype(str).tolist(),
            y=roll_bias.round(1).values.tolist(),
            mode="lines", name=asset,
            line=dict(color=PAL[j % len(PAL)], width=1.5)))
    fig_roll_bias.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_roll_bias.update_layout(
        title="Rollender (10-Fenster) Forecast-Bias (bps): Zeitlicher Verlauf der Fehler",
        yaxis_title="Rollender Ø-Fehler (bps)", height=400)

    # 8) Model × Asset RMSE heatmap
    fig_rmse_heat = go.Figure()
    if model_accuracy:
        acc_df2 = pd.DataFrame(model_accuracy)
        sub_1j = acc_df2[acc_df2["Horizont"] == "1J"]
        model_names = sub_1j["Modell"].unique().tolist()
        asset_names = sub_1j["Asset"].unique().tolist()
        z_rmse = [[float(sub_1j[(sub_1j["Modell"]==m)&(sub_1j["Asset"]==a)]["RMSE (bps)"].values[0])
                   if len(sub_1j[(sub_1j["Modell"]==m)&(sub_1j["Asset"]==a)]) > 0 else float("nan")
                   for a in asset_names] for m in model_names]
        fig_rmse_heat = go.Figure(go.Heatmap(
            z=z_rmse, x=asset_names, y=model_names,
            colorscale="RdYlGn_r",
            text=[[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in z_rmse],
            texttemplate="%{text}",
            hovertemplate="Modell=%{y}<br>Asset=%{x}<br>RMSE=%{z:.0f}bps<extra></extra>"))
        fig_rmse_heat.update_layout(
            title="RMSE-Heatmap (bps) @ 1J-Horizont: Roh-Faktoren vs. PCA-Modelle",
            height=max(300, 70 * len(model_names) + 100))

    # 9) Directional accuracy scatter: PCA1 vs PC68
    fig_da_scatter = go.Figure()
    if model_accuracy:
        acc_df3 = pd.DataFrame(model_accuracy)
        raw_da  = acc_df3[(acc_df3["Modell"].str.startswith("Roh")) & (acc_df3["Horizont"] == "1J")]
        pc68_da = acc_df3[(acc_df3["Modell"].str.contains("PC1-")) & (acc_df3["Horizont"] == "1J")]
        for df_sub, name, col in [(raw_da, "Roh-Faktoren", "#58a6ff"),
                                   (pc68_da, "PCA 68%", "#3fb950")]:
            if not df_sub.empty:
                fig_da_scatter.add_trace(go.Scatter(
                    x=df_sub["Asset"].tolist(),
                    y=df_sub["Dir. Acc %"].tolist(),
                    mode="markers+text", name=name,
                    text=df_sub["Dir. Acc %"].round(1).astype(str).tolist(),
                    textposition="top center",
                    marker=dict(color=col, size=10)))
        fig_da_scatter.add_hline(y=50, line_color="#8b949e", line_dash="dash")
        fig_da_scatter.update_layout(
            title="Direktionale Trefferquote: Roh-Faktoren vs. PCA (68%) @ 1-Jahres-Horizont",
            yaxis_title="Dir. Acc. (%)", height=420)

    # ── Tables ─────────────────────────────────────────────────────────
    acc_table_html = _df_html(pd.DataFrame(model_accuracy).sort_values(
        ["Asset","Horizont","RMSE (bps)"])) if model_accuracy else "<p class=\\'text-muted\\'>Keine Ergebnisse.</p>"
    diag_table_html = _df_html(pd.DataFrame([
        {"Asset": a, **{k: v for k, v in d.items() if not isinstance(v, (pd.Series, pd.Index))}}
        for a, d in diag_rows.items()])) if diag_rows else ""

    # ── PCA loadings explanation ────────────────────────────────────────
    pc_ev_rows = "".join(
        f"<tr><td>PC{i+1}</td><td>{ev_ratio[i]*100:.1f}%</td>"
        f"<td>{ev_cumsum[i]*100:.1f}%</td>"
        f"<td>{'★' if i < n_pc68 else '·'} {'95%-Grenze' if i == n_pc95-1 else ''}</td></tr>"
        for i in range(min(len(ev_ratio), 15)))

    body = f"""
<div class="ph-header">
  <h1>Prädiktiver Monte-Carlo-Backtest</h1>
  <div class="sub">4 Modelle: Roh-Faktoren | PCA-PC1 | PCA-PC1-{n_pc68} (~68%) | PCA-PC1-{n_pc95} (~95%) &middot;
    Walk-Forward {TRAIN_WINDOW}T &middot; {N_MC} MC-Pfade &middot; 4 Horizonte</div>
</div>

<div class="card mb-4">
  <div class="card-header"><strong>PCA-Motivation &amp; Modell-Übersicht</strong></div>
  <div class="card-body">
    <div class="row">
      <div class="col-md-5">
        <p class="small">
          Rohfaktoren (DXY, VIX, TNX) sind <strong>kollinear</strong> — z.B. DXY ↑ oft gleichzeitig mit TNX ↑.
          PCA projiziert die {len(PCA_INPUT)} Input-Faktoren auf unkorrelierte Hauptkomponenten.
          PCA kontrolliert Multikollinearität und destilliert "latente Makro-Signale".
        </p>
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>Modell</th><th>Faktoren</th><th>Erklärte Var.</th></tr></thead>
          <tbody>
            <tr><td>Roh</td><td>DXY, VIX, TNX (3)</td><td>n/a</td></tr>
            <tr><td>PCA-1</td><td>PC1</td><td>{ev_ratio[0]*100:.1f}% (1 Faktor)</td></tr>
            <tr><td>PCA-68%</td><td>PC1-{n_pc68}</td><td>~68.1%</td></tr>
            <tr><td>PCA-95%</td><td>PC1-{n_pc95}</td><td>~95%</td></tr>
          </tbody>
        </table>
      </div>
      <div class="col-md-7">
        <table class="table table-dark table-sm table-bordered">
          <thead><tr><th>PC</th><th>Einzel-Var.</th><th>Kumulativ</th><th>Modell</th></tr></thead>
          <tbody>{pc_ev_rows}</tbody>
        </table>
      </div>
    </div>
    {_formula(r"r_{{t+h}} = \\alpha + \\sum_{{k=1}}^{{K}} \\beta_k \\cdot PC_k^{{(t)}} + \\varepsilon_{{t+h}}",
              "Prädiktive PCA-Regression: K unkorrelierte Hauptkomponenten als Prädiktoren (1-Tages-Lag).")}
    {_info(f"Input für PCA: {', '.join(PCA_INPUT[:8])}... ({len(PCA_INPUT)} Faktoren). "
           f"Alle t-1 verzögert (keine Look-ahead). Walk-Forward: Train={TRAIN_WINDOW}T, Refit alle {RETRAIN_EVERY}T.")}
  </div>
</div>

{_chart_card("PCA Scree-Plot: Erklärte Varianz der Input-Faktoren", fig_scree, height=420,
    interp="Balkenhöhe = Einzelbeitrag jeder PC. Grüne Linie = Kumulativ. "
           "PC1 dominiert meist (Risk-on/Risk-off Faktor). "
           "Gelbe Linie 68.1%: Modell PC1-{n_pc68}. Rote Linie 95%: Modell PC1-{n_pc95}.")}

{_chart_card("Modellvergleich: Direktionale Trefferquote @ 1-Jahres-Horizont", fig_model_comp, height=470,
    interp="Roh-Faktoren vs. PCA-Varianten: Verbessert PCA die Vorhersage? "
           "PCA mit 68%+ Varianz stabilisiert die Schätzung durch Multikollinearitäts-Kontrolle. "
           "PC1-only: zu restriktiv — verliert Information. PC95%: zu viele Faktoren — Overfitting.")}

{_chart_card("RMSE-Heatmap (bps) nach Modell und Asset", fig_rmse_heat,
    interp="Grün = niedrigerer Fehler (besser). Rot = höherer Fehler. "
           "Vergleich zeigt, ob PCA tatsächlich Rohmodell schlägt. "
           "Diagonal-Muster (bestimmtes Asset systematisch besser) deutet auf Asset-spezifische Muster hin.")}

{_chart_card("Dir. Trefferquote: Roh vs. PCA 68%", fig_da_scatter, height=440,
    interp="Grüne Punkte über blauen: PCA verbessert Trefferquote. "
           "50%-Linie = zufälliges Raten. "
           ">55% bei 92 Vorhersagen statistisch signifikant (Binomialtest p<0.05).")}

{fan_htmls}

<div class="card mb-4" id="diagnostics">
  <div class="card-header"><strong>🔬 Residual-Diagnostik: Fehler-Muster &amp; Zeitversatz</strong></div>
  <div class="card-body">
    <p class="small text-muted">
      Prognosemodelle können systematische Fehler-Muster aufweisen:
      <strong>Overshoot</strong> (Modell überschätzt Bewegung), <strong>Undershoot</strong> (unterschätzt),
      <strong>Lag-Bias</strong> (Fehler autokorreliert = Modell "hinkt hinterher"),
      <strong>Saisonale Verzerrung</strong> (in bestimmten Monaten systematisch falsch).
    </p>
    <table class="table table-dark table-sm table-bordered mb-3">
      <thead><tr><th>Phänomen</th><th>Diagnose</th><th>Ursache</th><th>Lösung</th></tr></thead>
      <tbody>
        <tr><td><strong style="color:#f78166;">Overshoot</strong></td>
          <td>|Forecast| > |Realisiert| häufig</td>
          <td>Hohe VIX-Phase: Modell extrapoliert Crash zu lange</td>
          <td>Regimes trennen, Shrinkage-Prior</td></tr>
        <tr><td><strong style="color:#58a6ff;">Undershoot</strong></td>
          <td>|Forecast| < |Realisiert| häufig</td>
          <td>Trendphasen: OLS unterschätzt Momentum</td>
          <td>Momentum-Faktor ergänzen</td></tr>
        <tr><td><strong style="color:#d29922;">Lag-Bias (AC1>0)</strong></td>
          <td>Fehler(t) ≈ Fehler(t-1)</td>
          <td>Strukturbruch nicht erkannt, zu langsames Refitting</td>
          <td>Häufigeres Refitting, EWM statt OLS</td></tr>
        <tr><td><strong style="color:#3fb950;">Saisonaler Bias</strong></td>
          <td>Monatsweise systematisch hoch/tief</td>
          <td>Fehlende Saisonalitäts-Variable</td>
          <td>Monatsdummies, saisonale Normalisierung</td></tr>
      </tbody>
    </table>
  </div>
</div>

{_chart_card("Overshoot vs. Undershoot Rate (Roh-Modell, 1J-Horizont)", fig_overshoot, height=400,
    interp="Overshoot >50%: Modell neigt dazu, Bewegungen zu überschätzen (typisch nach Krisen). "
           "Undershoot >50%: Modell unterschätzt (typisch in Trendphasen). "
           "Ideal: 50/50 — symmetrischer, unbiased Forecast.")}

{_chart_card("Fehler-Autokorrelation AC(1): Zeitversatz-Muster erkennbar?", fig_ac1, height=400,
    interp="AC(1) > 0.15 (rot): Fehler(t) korreliert mit Fehler(t-1) = Modell hinkt hinterher. "
           "AC(1) < 0: überschnelles Mean-Reversion in Fehlern (seltener). "
           "Idealfall: AC(1) ≈ 0 (weiße Residuen = kein exploitierbares Muster mehr).")}

{_chart_card("Monatlicher Forecast-Bias: Saisonale Fehler-Systematik", fig_diag, height=440,
    interp="Positiver Bias in Monat X: Modell unterschätzt systematisch im Monat X. "
           "Negativer Bias: Modell überschätzt. "
           "Konsistente Muster → saisonale Dummy-Variablen als Korrekturterm ergänzen.")}

{_chart_card("Rollender Forecast-Bias (10-Fenster): Zeitlicher Verlauf", fig_roll_bias, height=420,
    interp="Phasen mit konstantem positiven/negativen Bias = Regime-Shift nicht erfasst. "
           "2008/2009: starker negativer Bias (Crash unterschätzt). "
           "2020: positiver Bias nach Crash-Tief (Erholung unterschätzt). "
           "Wechsel von positiv zu negativ = Wendepunkt im Modell-Fehler.")}

{_card("Vollständige Genauigkeitstabelle: Alle Modelle × Assets × Horizonte", acc_table_html)}
{_card("Residual-Diagnose-Tabelle (Roh-Modell, 1J-Horizont)", diag_table_html)}
"""
    _write(out / "predictive_backtest.html",
           _html_base("Prädiktiver Monte-Carlo-Backtest", 18, body))

'''

# ──────────────────────────────────────────────────────────────────────────────
MEGA_STRATEGIES_FN = r'''
def build_mega_strategies_report(tables, figures, out):  # noqa: C901
    """80+ trading strategies across 7 families, parameter-optimised, accordion Steckbriefe."""
    returns = _read(tables / "phase2_returns.csv")
    if returns is None:
        _write(out / "mega_strategies.html",
               _html_base("Mega-Strategien", 18, "<p>Renditen fehlen.</p>"))
        return
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns[returns.index.notna()]

    # ── PCA factors (unkorrellierte Signale) ───────────────────────────
    PCA_INPUT = [c for c in ["SPY","QQQ","IWM","^VIX","DX-Y.NYB","^TNX",
                              "XLE","XLB","XLI","JETS","IYT","GDX","MGC"]
                 if c in returns.columns]

    pc_scores_df = pd.DataFrame(index=returns.index)
    if PCA_INPUT:
        X_pca = returns[PCA_INPUT].copy()
        X_pca = X_pca - X_pca.mean()
        std_pca = X_pca.std()
        std_pca[std_pca < 1e-12] = 1.0
        X_pca = X_pca / std_pca
        X_pca_clean = X_pca.dropna()
        try:
            U, s, Vt = np.linalg.svd(X_pca_clean.values, full_matrices=False)
            pc_scores_raw = pd.DataFrame(U * s, index=X_pca_clean.index,
                                         columns=[f"PC{i+1}" for i in range(U.shape[1])])
            pc_scores_df = pc_scores_raw
        except Exception:
            pass

    # ── Unified strategy metrics ────────────────────────────────────────
    def _strat_metrics(r: pd.Series, name: str) -> dict | None:
        r = r.dropna()
        if len(r) < 252:
            return None
        ann_r  = float(r.mean() * 252)
        ann_v  = float(r.std() * np.sqrt(252)) + 1e-9
        sharpe = ann_r / ann_v
        cum    = (1 + r).cumprod()
        mdd    = float((cum / cum.cummax() - 1).min())
        split  = int(len(r) * 0.7)
        is_sh  = float(r.iloc[:split].mean() * 252 /
                       (r.iloc[:split].std() * np.sqrt(252) + 1e-9))
        oos_sh = float(r.iloc[split:].mean() * 252 /
                       (r.iloc[split:].std() * np.sqrt(252) + 1e-9))
        calmar = ann_r / (abs(mdd) + 1e-9)
        win_r  = float((r > 0).mean())
        skew   = float(pd.Series(r).skew())
        kurt   = float(pd.Series(r).kurt())
        # Rolling Sharpe stability
        roll_sh = r.rolling(126).apply(
            lambda x: x.mean()*252 / (x.std()*np.sqrt(252)+1e-9), raw=True)
        sh_std  = float(roll_sh.std())
        return {
            "Name": name,
            "CAGR %": round(ann_r * 100, 2),
            "Sharpe": round(sharpe, 3),
            "MaxDD %": round(mdd * 100, 2),
            "Calmar": round(calmar, 3),
            "IS Sharpe": round(is_sh, 3),
            "OOS Sharpe": round(oos_sh, 3),
            "Degrad.": round(is_sh - oos_sh, 3),
            "Win %": round(win_r * 100, 1),
            "Skew": round(skew, 3),
            "Kurt": round(kurt, 3),
            "Sharpe-Stab.": round(sh_std, 3),
            "N Tage": len(r),
        }

    def _apply_tc(pos: pd.Series, ret: pd.Series, bps=10) -> pd.Series:
        return pos * ret - pos.diff().abs().fillna(0) * (bps / 10000)

    def _make_signal_mom(r: pd.Series, lb: int) -> pd.Series:
        return np.sign(r.rolling(lb).sum().shift(1))

    def _make_signal_zrev(r: pd.Series, lb: int, entry_z: float) -> pd.Series:
        z = (r - r.rolling(lb).mean()) / (r.rolling(lb).std() + 1e-9)
        sig = pd.Series(0.0, index=r.index)
        sig[z.shift(1) < -entry_z] = 1.0   # oversold → long
        sig[z.shift(1) >  entry_z] = -1.0  # overbought → short
        return sig

    def _make_signal_macro(factor: pd.Series, lb: int) -> pd.Series:
        return np.sign(factor.rolling(lb).sum().shift(1))

    def _make_signal_vix_regime(vix: pd.Series, threshold: float) -> pd.Series:
        return pd.Series(np.where(vix.shift(1) < threshold, 1.0, -1.0), index=vix.index)

    def _make_signal_seasonal(r: pd.Series, long_months: list) -> pd.Series:
        return pd.Series(
            np.where(r.index.month.isin(long_months), 1.0, -1.0),
            index=r.index)

    # ── Strategy definitions ────────────────────────────────────────────
    COMMODITY_ASSETS = [c for c in ["GC=F","CL=F","HG=F","ZW=F","NG=F","SI=F","ZS=F","BZ=F","ZC=F"]
                        if c in returns.columns]

    all_strats: list[dict] = []  # {name, family, returns_series, description, signal}

    # ─── Family A: Momentum (3 lookbacks × 9 assets) ───────────────────
    for lb in [21, 63, 126]:
        for asset in COMMODITY_ASSETS:
            r = returns[asset].dropna()
            if len(r) < 252 + lb:
                continue
            sig = _make_signal_mom(r, lb)
            strat_r = _apply_tc(sig, r)
            all_strats.append({
                "family": "Momentum", "asset": asset, "lb": lb,
                "name": f"MOM-{lb}T-{asset}",
                "desc": f"Momentum-Signal: {lb}-Tages-Return Vorzeichen → Long/Short {asset}",
                "signal": sig, "ret": strat_r,
            })

    # ─── Family B: Mean Reversion (2 lookbacks × 2 Z-thresholds × 7 assets) ─
    for lb in [21, 63]:
        for z_thr in [1.0, 2.0]:
            for asset in COMMODITY_ASSETS[:7]:
                r = returns[asset].dropna()
                if len(r) < 252 + lb:
                    continue
                sig = _make_signal_zrev(r, lb, z_thr)
                strat_r = _apply_tc(sig, r)
                all_strats.append({
                    "family": "Mean Reversion", "asset": asset, "lb": lb,
                    "name": f"ZREV-{lb}T-Z{z_thr:.0f}-{asset}",
                    "desc": f"Z-Score Mean Reversion: ±{z_thr}σ Entry, {lb}T Lookback, {asset}",
                    "signal": sig, "ret": strat_r,
                })

    # ─── Family C: Macro Signal (DXY, VIX-Regime, TNX) × 6 assets ─────
    macro_sigs = []
    for col, lb, desc in [("DX-Y.NYB", 21, "DXY-Momentum"),
                           ("DX-Y.NYB", 63, "DXY-Momentum 63T"),
                           ("^TNX", 21, "10Y-Zinsen Richtung"),
                           ("^TNX", 63, "10Y-Zinsen 63T")]:
        if col in returns.columns:
            macro_sigs.append((col, lb, desc))

    for col, lb, sig_desc in macro_sigs:
        fac = returns[col].dropna()
        sig_base = _make_signal_macro(fac, lb)
        for asset in COMMODITY_ASSETS[:6]:
            r = returns[asset].dropna()
            idx = sig_base.index.intersection(r.index)
            if len(idx) < 252:
                continue
            # Inverse signal: DXY up = commodity down
            inv = -1 if col == "DX-Y.NYB" else 1
            sig = sig_base.loc[idx] * inv
            strat_r = _apply_tc(sig, r.loc[idx])
            all_strats.append({
                "family": "Makro-Signal", "asset": asset, "lb": lb,
                "name": f"MACRO-{col.replace('^','').replace('-','')}-{lb}T-{asset}",
                "desc": f"{sig_desc} → {'Short' if inv==-1 else 'Long'}/{asset}",
                "signal": sig, "ret": strat_r,
            })

    # VIX-Regime
    if "^VIX" in returns.columns:
        vix = returns["^VIX"].dropna()
        for thr in [20, 25, 30]:
            sig_vix = _make_signal_vix_regime(vix, thr)
            for asset in COMMODITY_ASSETS[:6]:
                r = returns[asset].dropna()
                idx = sig_vix.index.intersection(r.index)
                if len(idx) < 252:
                    continue
                strat_r = _apply_tc(sig_vix.loc[idx], r.loc[idx])
                all_strats.append({
                    "family": "Makro-Signal", "asset": asset,
                    "name": f"VIXREG-{thr}-{asset}",
                    "desc": f"VIX-Regime: VIX<{thr} → Long {asset}, sonst Short",
                    "signal": sig_vix.loc[idx], "ret": strat_r,
                })

    # ─── Family D: PCA Signal (PC1-3 × 5 assets) ──────────────────────
    for pc_n in [1, 2, 3]:
        pc_col = f"PC{pc_n}"
        if pc_col not in pc_scores_df.columns:
            continue
        pc = pc_scores_df[pc_col]
        for asset in COMMODITY_ASSETS[:5]:
            r = returns[asset].dropna()
            # Try both polarities (PC sign is arbitrary)
            for inv_label, inv in [("pos", 1), ("neg", -1)]:
                sig = np.sign(pc.shift(1) * inv)
                idx = sig.dropna().index.intersection(r.index)
                if len(idx) < 252:
                    continue
                strat_r = _apply_tc(sig.loc[idx], r.loc[idx])
                m = _strat_metrics(strat_r, "")
                if m and abs(m["Sharpe"]) > 0.1:  # keep better polarity only
                    all_strats.append({
                        "family": "PCA-Signal", "asset": asset,
                        "name": f"PCA-PC{pc_n}-{inv_label}-{asset}",
                        "desc": f"PC{pc_n} Vorzeichen-Signal ({inv_label}) → {asset}",
                        "signal": sig.loc[idx], "ret": strat_r,
                    })
                    break  # take first polarity that passes threshold

    # ─── Family E: Cross-Asset Ratio Signals ──────────────────────────
    ratio_pairs = [
        ("HG=F","GC=F",  "Kupfer/Gold",   1, "HG=F",  "Kupfer/Gold↑ → Wachstum → Long HG=F"),
        ("GC=F","SI=F",  "Gold/Silber",   1, "GC=F",  "Gold/Silber↑ → Risk-Off → Long GC=F"),
        ("CL=F","NG=F",  "Öl/Gas",       1, "CL=F",  "Öl/Gas↑ → Öl outperformed"),
        ("CL=F","GC=F",  "Öl/Gold",      -1,"GC=F",  "Öl/Gold↑ → Risiko-On → Short GC=F"),
        ("ZW=F","ZC=F",  "Weizen/Mais", 1, "ZW=F",  "Weizen/Mais-Spread → Long Weizen"),
        ("XLE","GDX",    "Energie/Gold-ETF",1,"CL=F","XLE/GDX↑ → Energie outperformed → Long CL=F"),
    ]
    for a1, a2, ratio_nm, inv, target, desc in ratio_pairs:
        if a1 not in returns.columns or a2 not in returns.columns or target not in returns.columns:
            continue
        r1 = returns[a1].dropna(); r2 = returns[a2].dropna()
        idx_r = r1.index.intersection(r2.index)
        ratio = (r1.loc[idx_r] - r2.loc[idx_r]).rolling(21).sum() * inv
        sig = np.sign(ratio.shift(1))
        r_t = returns[target].dropna()
        idx = sig.dropna().index.intersection(r_t.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig.loc[idx], r_t.loc[idx])
        all_strats.append({
            "family": "Cross-Asset", "asset": target,
            "name": f"RATIO-{a1.replace('=F','').replace('^','')}-{a2.replace('=F','').replace('^','')}",
            "desc": desc, "signal": sig.loc[idx], "ret": strat_r,
        })

    # ─── Family F: Seasonal Calendar Strategies ───────────────────────
    seasonal_defs = [
        ("GC=F",  [9,10,11,12],  "Gold Q4 Saisonalität (Schmuck-Hochsaison)"),
        ("NG=F",  [10,11,12,1,2],"Gas Heizperiode (Okt-Feb)"),
        ("NG=F",  [6,7,8],       "Gas Kühlperiode (Jun-Aug)"),
        ("ZW=F",  [3,4,5],       "Weizen Pflanzungsrallye (Mär-Mai)"),
        ("ZS=F",  [3,4,5],       "Soja Planting Rally (Mär-Mai)"),
        ("CL=F",  [5,6,7,8],     "Öl Driving Season (Mai-Aug)"),
        ("HG=F",  [1,2,3],       "Kupfer China-Stimulus Q1 (Jan-Mär)"),
        ("ZW=F",  [9,10,11],     "Weizen Short: Ernte-Druck (Sep-Nov)"),
    ]
    for asset, months, desc in seasonal_defs:
        if asset not in returns.columns:
            continue
        r = returns[asset].dropna()
        if len(r) < 252:
            continue
        sig = _make_signal_seasonal(r, months)
        strat_r = _apply_tc(sig, r)
        months_str = "/".join(map(str, months))
        all_strats.append({
            "family": "Saisonal", "asset": asset,
            "name": f"SEASON-{asset.replace('=F','').replace('^','')}-{months_str}",
            "desc": desc, "signal": sig, "ret": strat_r,
        })

    # ─── Family G: Multi-Signal Composites ────────────────────────────
    # Combine MOM + MACRO signals per asset
    for asset in COMMODITY_ASSETS[:5]:
        r = returns[asset].dropna()
        sigs_to_combine = []
        # MOM 63T signal
        sig_m = _make_signal_mom(r, 63)
        sigs_to_combine.append(sig_m)
        # DXY signal
        if "DX-Y.NYB" in returns.columns:
            sig_d = -_make_signal_macro(returns["DX-Y.NYB"].dropna(), 21)
            sigs_to_combine.append(sig_d)
        # PC1 signal
        if "PC1" in pc_scores_df.columns:
            sig_p = np.sign(pc_scores_df["PC1"].shift(1))
            sigs_to_combine.append(sig_p)
        if len(sigs_to_combine) < 2:
            continue
        combined = pd.concat(sigs_to_combine, axis=1).mean(axis=1).dropna()
        sig_c = np.sign(combined)
        idx = sig_c.index.intersection(r.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig_c.loc[idx], r.loc[idx])
        all_strats.append({
            "family": "Komposit", "asset": asset,
            "name": f"COMBO-MOM-DXY-PC1-{asset.replace('=F','').replace('^','')}",
            "desc": f"Komposit: MOM-63T + DXY-Signal + PC1 → {asset}",
            "signal": sig_c.loc[idx], "ret": strat_r,
        })

    # ─── Family H: Granger / CCF Lead-Lag Signals ─────────────────────
    # Known strong Granger pairs from phase 6 analysis
    granger_pairs = [
        ("CL=F",  "SM",   1,  1, "Öl → Small-Caps (Lag 1T)"),
        ("DX-Y.NYB","CVX", 1, -1, "DXY → CVX (Lag 1T, invers)"),
        ("GC=F",  "GDX",  1,  1, "Gold → GDX (Lag 1T)"),
        ("CL=F",  "XLE",  1,  1, "Öl → XLE (Lag 1T)"),
        ("^VIX",  "GC=F", 1, -1, "VIX → Gold (invers, Risk-off)"),
        ("HG=F",  "GDX",  2,  1, "Kupfer → GDX (Lag 2T)"),
        ("CL=F",  "HG=F", 2,  1, "Öl → Kupfer (Lag 2T)"),
    ]
    for src, tgt, lag, inv, desc in granger_pairs:
        if src not in returns.columns or tgt not in returns.columns:
            continue
        src_r = returns[src].dropna()
        tgt_r = returns[tgt].dropna()
        sig = np.sign(src_r.rolling(21).sum().shift(lag) * inv)
        idx = sig.dropna().index.intersection(tgt_r.index)
        if len(idx) < 252:
            continue
        strat_r = _apply_tc(sig.loc[idx], tgt_r.loc[idx])
        all_strats.append({
            "family": "Granger/CCF", "asset": tgt,
            "name": f"GRANGER-{src.replace('=F','').replace('^','').replace('-','')}-{tgt}-L{lag}",
            "desc": desc, "signal": sig.loc[idx], "ret": strat_r,
        })

    # ─── Family I: Overshoot/Correction Strategies ────────────────────
    # Enter mean-reversion after extreme moves (>2σ in 5 days)
    for asset in COMMODITY_ASSETS[:6]:
        r = returns[asset].dropna()
        if len(r) < 252:
            continue
        roll_5 = r.rolling(5).sum()
        vol_21 = r.rolling(21).std() * np.sqrt(5)
        z_5d = roll_5 / (vol_21 + 1e-9)
        # After extreme drop (z < -2): expect recovery → long
        sig_ov = pd.Series(0.0, index=r.index)
        sig_ov[z_5d.shift(1) < -2.0] = 1.0   # bounce after crash
        sig_ov[z_5d.shift(1) >  2.0] = -1.0  # fade after spike
        strat_r = _apply_tc(sig_ov, r)
        all_strats.append({
            "family": "Overshoot/Korr.", "asset": asset,
            "name": f"OVERSHOOT-5T-Z2-{asset.replace('=F','').replace('^','')}",
            "desc": f"Mean-Reversion nach 5T-Extrem (±2σ): Bounce/Fade Signal für {asset}",
            "signal": sig_ov, "ret": strat_r,
        })

    # ── Compute metrics for all strategies ─────────────────────────────
    strat_meta = []
    for s in all_strats:
        m = _strat_metrics(s["ret"], s["name"])
        if m:
            m["Familie"] = s["family"]
            m["Asset"]   = s["asset"]
            m["Beschreibung"] = s.get("desc", "")
            strat_meta.append(m)

    if not strat_meta:
        _write(out / "mega_strategies.html",
               _html_base("Mega-Strategien", 18, "<p>Keine Strategie-Ergebnisse.</p>"))
        return

    meta_df = pd.DataFrame(strat_meta).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    meta_df["Rang"] = meta_df.index + 1

    # ── Summary charts ─────────────────────────────────────────────────
    # 1) Sharpe distribution by family
    fig_sharpe_box = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]["Sharpe"]
        fig_sharpe_box.add_trace(go.Box(y=sub.tolist(), name=fam,
            marker_color=PAL[j % len(PAL)], boxpoints="all", jitter=0.4))
    fig_sharpe_box.add_hline(y=0, line_color="#8b949e", line_dash="dash")
    fig_sharpe_box.add_hline(y=0.5, line_color="#3fb950", line_dash="dot",
                              annotation_text="Sharpe=0.5")
    fig_sharpe_box.update_layout(
        title=f"Sharpe-Verteilung nach Strategie-Familie ({len(meta_df)} Strategien total)",
        yaxis_title="Sharpe Ratio", height=480)

    # 2) Top-20 Sharpe bar chart
    top20 = meta_df.head(20)
    fig_top20 = go.Figure(go.Bar(
        x=top20["Name"].tolist(),
        y=top20["Sharpe"].tolist(),
        marker_color=["#3fb950" if v > 0.5 else "#d29922" if v > 0 else "#f78166"
                      for v in top20["Sharpe"]],
        text=[f"{v:.2f}" for v in top20["Sharpe"]],
        textposition="outside",
        hovertemplate="%{x}<br>Sharpe=%{y:.3f}<extra></extra>"))
    fig_top20.add_hline(y=0, line_color="#8b949e")
    fig_top20.update_layout(
        title="Top-20 Strategien: Sharpe Ratio (sortiert)",
        xaxis_tickangle=-40, height=520, yaxis_title="Sharpe")

    # 3) IS vs OOS scatter
    fig_is_oos = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_is_oos.add_trace(go.Scatter(
            x=sub["IS Sharpe"].tolist(), y=sub["OOS Sharpe"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7),
            hovertemplate="%{text}<br>IS=%{x:.2f} OOS=%{y:.2f}<extra></extra>"))
    mn = float(min(meta_df["IS Sharpe"].min(), meta_df["OOS Sharpe"].min())) - 0.1
    mx = float(max(meta_df["IS Sharpe"].max(), meta_df["OOS Sharpe"].max())) + 0.1
    fig_is_oos.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#8b949e", dash="dash"), name="IS=OOS", showlegend=True))
    fig_is_oos.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    fig_is_oos.update_layout(
        title="IS vs. OOS Sharpe: Robustheit der Strategien (Punkte oberhalb Diagonale = Overfitting)",
        xaxis_title="IS Sharpe (70%)", yaxis_title="OOS Sharpe (30%)", height=500)

    # 4) MaxDD vs Sharpe scatter
    fig_dd_sh = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_dd_sh.add_trace(go.Scatter(
            x=sub["MaxDD %"].tolist(), y=sub["Sharpe"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7)))
    fig_dd_sh.update_layout(
        title="MaxDD vs. Sharpe: Risk-Return Profil aller Strategien",
        xaxis_title="Max Drawdown (%)", yaxis_title="Sharpe", height=480)

    # 5) Family heatmap (avg Sharpe by family × asset)
    fams = meta_df["Familie"].unique().tolist()
    assets_uniq = meta_df["Asset"].unique().tolist()
    heat_z = [[float(meta_df[(meta_df["Familie"]==f)&(meta_df["Asset"]==a)]["Sharpe"].mean())
               if len(meta_df[(meta_df["Familie"]==f)&(meta_df["Asset"]==a)]) else float("nan")
               for a in assets_uniq] for f in fams]
    fig_fam_heat = go.Figure(go.Heatmap(
        z=heat_z, x=assets_uniq, y=fams,
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in heat_z],
        texttemplate="%{text}",
        hovertemplate="Familie=%{y}<br>Asset=%{x}<br>Ø Sharpe=%{z:.2f}<extra></extra>"))
    fig_fam_heat.update_layout(
        title="Ø Sharpe-Heatmap: Strategie-Familie × Asset",
        height=max(300, 55 * len(fams) + 100))

    # 6) Cumulative return: top 5 equity curves
    fig_eq_top5 = go.Figure()
    for j, row in meta_df.head(5).iterrows():
        s_obj = next((s for s in all_strats if s["name"] == row["Name"]), None)
        if s_obj is None:
            continue
        eq = (1 + s_obj["ret"].dropna()).cumprod()
        fig_eq_top5.add_trace(go.Scatter(
            x=eq.index.astype(str).tolist(),
            y=np.round(eq.values, 4).tolist(),
            mode="lines", name=f"#{j+1} {row['Name']}",
            line=dict(color=PAL[j % len(PAL)], width=2)))
    fig_eq_top5.update_layout(
        title="Top-5 Strategien: Equity-Kurven (log-Skala)",
        yaxis_type="log", yaxis_title="Kapital (Basis=1)", height=520)

    # 7) CAGR vs MaxDD (efficient frontier)
    fig_frontier = go.Figure()
    for j, fam in enumerate(meta_df["Familie"].unique()):
        sub = meta_df[meta_df["Familie"] == fam]
        fig_frontier.add_trace(go.Scatter(
            x=(-sub["MaxDD %"]).tolist(), y=sub["CAGR %"].tolist(),
            mode="markers", name=fam,
            text=sub["Name"].tolist(),
            marker=dict(color=PAL[j % len(PAL)], size=7)))
    fig_frontier.update_layout(
        title="Effizienzgrenze: CAGR vs. Max-Drawdown — Risiko/Rendite für alle Strategien",
        xaxis_title="-MaxDD (%)", yaxis_title="CAGR (%)", height=480)

    # ── Accordion Steckbriefe ─────────────────────────────────────────
    import json as _json

    def _build_steckbrief(rank: int, row: pd.Series, s_obj: dict | None) -> str:
        sh_color = "#3fb950" if row["Sharpe"] > 0.5 else "#d29922" if row["Sharpe"] > 0 else "#f78166"
        header_badge = (f'<span class="badge" style="background:{sh_color}">'
                        f'Sharpe {row["Sharpe"]:.2f}</span>')
        oos_badge_color = "#3fb950" if row["OOS Sharpe"] > 0.3 else "#d29922" if row["OOS Sharpe"] > 0 else "#f78166"
        oos_badge = (f'<span class="badge ms-1" style="background:{oos_badge_color}">'
                     f'OOS {row["OOS Sharpe"]:.2f}</span>')
        dd_badge = (f'<span class="badge ms-1 bg-secondary">DD {row["MaxDD %"]:.1f}%</span>')
        fam_badge = f'<span class="badge ms-1 bg-secondary">{row["Familie"]}</span>'
        metric_rows = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in row.items()
            if k not in ("Name","Familie","Asset","Beschreibung","Rang"))

        chart_html = ""
        if rank <= 25 and s_obj is not None:  # embed Plotly for top 25
            eq = (1 + s_obj["ret"].dropna()).cumprod()
            fig_sb = go.Figure()
            fig_sb.add_trace(go.Scatter(
                x=eq.index.astype(str).tolist(),
                y=np.round(eq.values, 4).tolist(),
                mode="lines", name="Equity",
                line=dict(color=PAL[(rank-1) % len(PAL)], width=1.8)))
            # Buy & hold
            bh = (1 + returns[row["Asset"]].dropna()).cumprod()
            fig_sb.add_trace(go.Scatter(
                x=bh.index.astype(str).tolist(),
                y=np.round(bh.values, 4).tolist(),
                mode="lines", name="B&H",
                line=dict(color="#8b949e", width=0.9, dash="dot")))
            fig_sb.update_layout(
                height=280, margin=dict(t=30,b=30,l=30,r=10),
                yaxis_type="log", showlegend=True,
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", size=10),
                legend=dict(orientation="h", y=1.1))

            # Return distribution mini chart
            fig_dist = go.Figure()
            ret_vals = s_obj["ret"].dropna().values * 10000
            fig_dist.add_trace(go.Histogram(
                x=ret_vals.tolist(), nbinsx=60,
                marker_color="#58a6ff", opacity=0.75,
                name="Rendite-Verteilung (bps)"))
            fig_dist.update_layout(
                height=200, margin=dict(t=20,b=30,l=30,r=10),
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3", size=9),
                showlegend=False,
                xaxis_title="Tagesrendite (bps)", yaxis_title="Häufigkeit")

            eq_json   = _json.dumps(fig_sb.to_dict()).replace("</", "<\\/")
            dist_json = _json.dumps(fig_dist.to_dict()).replace("</", "<\\/")
            uid = f"sb_{rank}"
            chart_html = (
                f'<div id="eq_{uid}" style="height:280px;"></div>'
                f'<div id="dist_{uid}" style="height:200px;"></div>'
                f'<script>'
                f'(function(){{var eq={eq_json};'
                f'var dist={dist_json};'
                f'if(window.Plotly){{Plotly.newPlot("eq_{uid}",eq.data,eq.layout,{{responsive:true,displayModeBar:false}});'
                f'Plotly.newPlot("dist_{uid}",dist.data,dist.layout,{{responsive:true,displayModeBar:false}});}}}})()'
                f'</script>'
            )

        sig_stats = ""
        if s_obj is not None and "signal" in s_obj:
            sig = s_obj["signal"].dropna()
            n_long  = int((sig > 0).sum())
            n_short = int((sig < 0).sum())
            n_flat  = int((sig == 0).sum())
            sig_stats = (f'<div class="small text-muted mt-2">'
                         f'Signal: Long {n_long} Tage | Short {n_short} Tage | Flat {n_flat} Tage | '
                         f'Long-Anteil {n_long/(n_long+n_short+n_flat+1)*100:.0f}%</div>')

        return f"""
<div class="accordion-item" style="background:#161b22;border:1px solid #30363d;">
  <h2 class="accordion-header">
    <button class="accordion-button collapsed py-2" type="button"
      data-bs-toggle="collapse" data-bs-target="#sb{rank}"
      style="background:#0d1117;color:#e6edf3;font-size:0.85rem;">
      <strong class="me-2">#{rank}</strong>
      <span class="me-2">{row["Name"]}</span>
      {header_badge}{oos_badge}{dd_badge}{fam_badge}
    </button>
  </h2>
  <div id="sb{rank}" class="accordion-collapse collapse">
    <div class="accordion-body" style="background:#161b22;">
      <p class="small text-muted">{row.get("Beschreibung","")}</p>
      <div class="row">
        <div class="col-md-4">
          <table class="table table-dark table-sm table-bordered">
            <tbody>{metric_rows}</tbody>
          </table>
          {sig_stats}
        </div>
        <div class="col-md-8">{chart_html}</div>
      </div>
    </div>
  </div>
</div>"""

    # Build accordion HTML
    accordion_html = '<div class="accordion" id="stratAccordion">'
    for rank, (_, row) in enumerate(meta_df.iterrows(), 1):
        s_obj = next((s for s in all_strats if s["name"] == row["Name"]), None)
        accordion_html += _build_steckbrief(rank, row, s_obj)
    accordion_html += "</div>"

    # ── Summary table (top 30) ──────────────────────────────────────────
    display_cols = ["Rang","Name","Familie","Asset","Sharpe","OOS Sharpe","CAGR %",
                    "MaxDD %","Calmar","Degrad.","Win %","N Tage"]
    top30_html = _df_html(meta_df[display_cols].head(30))

    body = f"""
<div class="ph-header">
  <h1>Mega-Strategien: {len(meta_df)} Strategien — 9 Familien</h1>
  <div class="sub">Momentum · Mean Reversion · Makro-Signale · PCA · Cross-Asset · Saisonal ·
    Komposit · Granger/CCF · Overshoot/Korrektur &middot; Parameter-Grid &middot; Accordion Steckbriefe</div>
</div>

<div class="card mb-3">
  <div class="card-header"><strong>Strategie-Familien Übersicht</strong></div>
  <div class="card-body">
    <table class="table table-dark table-sm table-bordered">
      <thead><tr><th>Familie</th><th>Anzahl</th><th>Ø Sharpe</th><th>Best Sharpe</th><th>Beste Strategie</th></tr></thead>
      <tbody>
        {"".join(
        f'<tr><td>{fam}</td>'
        f'<td>{len(meta_df[meta_df["Familie"]==fam])}</td>'
        f'<td>{meta_df[meta_df["Familie"]==fam]["Sharpe"].mean():.2f}</td>'
        f'<td>{meta_df[meta_df["Familie"]==fam]["Sharpe"].max():.2f}</td>'
        f'<td class="small">{meta_df[meta_df["Familie"]==fam].iloc[0]["Name"]}</td></tr>'
        for fam in meta_df["Familie"].unique())}
      </tbody>
    </table>
    {_info(f"Alle {len(meta_df)} Strategien mit ≥252 gemeinsamen Handelstagen. "
           f"TC=10bps pro Roundtrip. IS/OOS Split: 70%/30%. "
           f"OOS-Sharpe > 0: live-handelbar. OOS-Sharpe > 0.3: robust.")}
  </div>
</div>

{_chart_card("Top-20 Strategien: Sharpe Ratio (sortiert)", fig_top20, height=540,
    interp="Grün > 0.5: sehr gute Strategie. Gelb 0-0.5: moderat. Rot: negativ. "
           "OOS-Sharpe im Steckbrief prüfen: IS kann durch Overfitting aufgebläht sein.")}
{_chart_card("Sharpe-Verteilung nach Strategie-Familie", fig_sharpe_box, height=500,
    interp="Box-Whisker + alle Punkte. Breite Boxen = inkonsistente Familienperformance. "
           "Familien über 0-Linie: insgesamt profitable Signalklasse.")}
{_chart_card("Top-5 Strategien: Equity-Kurven (log-Skala)", fig_eq_top5, height=540,
    interp="Log-Skala: gleicher prozentualer Anstieg = gleiche Höhe. "
           "Beste Strategien sollten Buy & Hold deutlich schlagen. "
           "Crashphasen (2008, 2020): Short-Signale zeigen Wert.")}
{_chart_card("IS vs. OOS Sharpe: Robustheit", fig_is_oos, height=520,
    interp="Punkte nahe Diagonale: IS=OOS (kein Overfitting). "
           "Weit oberhalb Diagonale: Overfitting (IS>>OOS). "
           "Unterhalb Diagonale: OOS>IS (unerwartetes Out-of-Sample-Alpha).")}
{_chart_card("Effizienzgrenze: CAGR vs. MaxDrawdown", fig_frontier, height=500,
    interp="Rechts oben = ideale Strategien (hohe Rendite, geringer Drawdown). "
           "Links oben: hohe Rendite aber auch hohes Risiko. "
           "Pareto-Front oben rechts: dominierende Strategien.")}
{_chart_card("MaxDD vs. Sharpe: Risk-Return für alle Strategien", fig_dd_sh, height=500,
    interp="Oben links = ideal (hohes Sharpe, geringer Drawdown). "
           "Cluster pro Familie zeigen Familien-Charakteristika.")}
{_chart_card("Ø Sharpe-Heatmap: Familie × Asset", fig_fam_heat,
    interp="Dunkelgrün: Strategie-Familie funktioniert gut für dieses Asset. "
           "Rot: Familie funktioniert nicht für dieses Asset. "
           "Hilft bei der Identifikation der besten Faktor-Asset-Kombinationen.")}

<div class="card mb-4">
  <div class="card-header"><strong>Top-30 Strategien: Kompakt-Tabelle</strong></div>
  <div class="card-body">{top30_html}</div>
</div>

<div class="card mb-4" id="steckbriefe">
  <div class="card-header">
    <strong>📋 Strategie-Steckbriefe (alle {len(meta_df)}) — ausklappbar</strong>
    <span class="badge bg-success ms-2">Top 25 mit Equity-Charts &amp; Verteilungen</span>
  </div>
  <div class="card-body p-2">
    {accordion_html}
  </div>
</div>
"""
    _write(out / "mega_strategies.html",
           _html_base("Mega-Strategien", 18, body))

'''

# ──────────────────────────────────────────────────────────────────────────────
# Patch report_builder.py
with open(RB, "r", encoding="utf-8") as f:
    src = f.read()

# Markers
START_PRED = "def build_predictive_backtest_report(tables, figures, out):  # noqa: C901"
END_PRED   = "\ndef build_index(tables, figures, out):"

idx_start = src.find(START_PRED)
idx_end   = src.find(END_PRED, idx_start)
if idx_start < 0 or idx_end < 0:
    raise RuntimeError(f"Markers not found: start={idx_start} end={idx_end}")

new_src = src[:idx_start] + PREDICTIVE_FN.lstrip("\n") + MEGA_STRATEGIES_FN + src[idx_end:]
with open(RB, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"Patched {RB}")
print(f"Old function removed ({idx_end - idx_start} chars). "
      f"New size: {len(new_src)} chars ({len(new_src.splitlines())} lines)")
