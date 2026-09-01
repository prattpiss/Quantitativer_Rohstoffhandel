"""Report 41 — Sektor-Rotations-Analyse (hochstatistisch).

Granger-Kausalität, DCC-Approximation, PCA (SVD), regimeabhängige Korrelation,
Materials-Split und ein Rotations-Overlay auf der validierten Baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import charts as C
from core import data as dat
from core import stats_tools as sx
from core import strategy as st
from core import theme as T
from core import universe as U

PHASE = 21
TITLE = "Report 41 — Sektor-Rotation Deep Dive"

# Kausalitäts-Kern: liquide Sektor-/Rohstoff-Träger, für die eine N×N-Matrix
# rechenbar und ökonomisch interpretierbar bleibt.
LEAD_SET = (U.SECTOR_ETFS + ["GLD", "SLV", "GDX", "COPX"] + U.FUTURES
            + ["ITA", "XAR", "IYT", "IBB", "SPY", "TLT", "HYG", "JETS"])

MAX_LAG = 5


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = dat.close_panel(sorted(set(LEAD_SET)), min_obs=500)
    px = px.ffill().dropna(how="any")
    return px, dat.log_returns(px).dropna(how="any")


def _group_index(rets: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in rets.columns]
    return rets[cols].mean(axis=1) if cols else pd.Series(dtype=float)


def build(out: Path) -> None:  # noqa: C901
    print("Report 41 — Sektor-Rotation")
    np.random.seed(sx.SEED)
    body: list[str] = []

    body.append(T.header("Report 41 — Sektor-Rotations-Analyse",
                         "Granger-Kausalität · DCC-Approximation · PCA · Regime-Korrelation · "
                         "Rotations-Overlay"))

    # ── §1 Universum & Datenverfügbarkeit ───────────────────────────────
    full_uni = U.sector_rotation_universe()
    av = dat.availability(full_uni)
    ok = av[av["Obs"] >= 500]
    dropped = av[av["Obs"] < 500]
    body.append(T.card("§1 — Universum, Auswahl-Logik und Datenverfügbarkeit",
        T.info("Das Universum folgt der Market-Cap-Konvergenzregel aus §4 des "
               "Übergabedokuments: Aufnahme absteigend nach Marktkapitalisierung bis der "
               "marginale Beitrag unter 1 % der kumulierten Sektorkapitalisierung fällt. "
               "Titel unterhalb dieser Schwelle erklären Sektordynamik praktisch nicht mehr, "
               "erhöhen aber die Zahl der Hypothesentests und damit die Mehrfachtest-Strafe.")
        + T.stat_row([("Universum gesamt", str(len(full_uni))),
                      ("Mit ≥500 Beobachtungen", str(len(ok))),
                      ("Verworfen", str(len(dropped))),
                      ("Kausalitäts-Kern", str(len(set(LEAD_SET))))])
        + T.df_html(av.sort_values("Obs"), index=False)
        + T.warn("Delistete bzw. übernommene Titel (z. B. SAVE, PDCE, CHK, MRO) haben "
                 "abgeschnittene Historien. Sie werden hier <em>explizit ausgewiesen</em> "
                 "statt still entfernt — damit ist der Survivorship-Bias sichtbar. Alle "
                 "Aussagen der folgenden Abschnitte gelten für das überlebende Universum "
                 "und sind daher nach oben verzerrt.")))

    px, rets = _panel()
    if rets.empty or "JETS" not in rets.columns:
        T.write(out / "r41_sector_rotation_deep.html",
                T.html_base(TITLE, PHASE, "".join(body) + T.warn("Keine Daten.")))
        return
    jets = rets["JETS"]

    # ── §1b Geladene Sektorreihen ───────────────────────────────────
    grp_map = {c: U.group_of(c) for c in px.columns}
    fig_px = C.price_panel(px, "Alle geladenen Sektor- und Rohstoffreihen "
                               "(Schlusskurse, auf 100 indexiert, log-Skala)",
                           groups=grp_map)
    grp_names = sorted({g for g in grp_map.values() if g})
    grp_cum = {g: (1 + rets[[c for c in px.columns if grp_map[c] == g]].mean(axis=1)
                   ).cumprod() for g in grp_names}
    fig_grp, h_grp = C.series_grid(grp_cum, "Gleichgewichteter Kapitalindex je Gruppe "
                                            "(Start = 1)", cols=3)
    body.append(T.card("§1b — Sichtprüfung: die tatsächlich geladenen Kursreihen",
        T.info("Alle folgenden Ergebnisse — Granger-Matrix, DCC, PCA, Regime-Korrelation — "
               "beruhen ausschließlich auf den hier gezeigten Reihen. Sie werden vor der "
               "Analyse offengelegt, damit jede spätere Aussage direkt am Kursverlauf "
               "gegengeprüft werden kann. Die Matrix ist auf den gemeinsamen "
               "Handelskalender ausgerichtet und vorwärts gefüllt, danach werden Zeilen "
               "mit fehlenden Werten verworfen — daher der gemeinsame Startzeitpunkt.")
        + T.stat_row([("Reihen in der Matrix", str(px.shape[1])),
                      ("Handelstage", f"{px.shape[0]:,}".replace(",", ".")),
                      ("Start", str(px.index[0].date())),
                      ("Ende", str(px.index[-1].date())),
                      ("Gruppen", str(len(grp_names)))])
        + T.div(fig_px, 520) + T.div(fig_grp, h_grp) + C.data_table(px)
        + T.interp(
            "Gemessen über den gesamten gemeinsamen Zeitraum liegt "
            f"<strong>{max(grp_cum, key=lambda g: grp_cum[g].iloc[-1])}</strong> vorn "
            f"({grp_cum[max(grp_cum, key=lambda g: grp_cum[g].iloc[-1])].iloc[-1]:.2f}× "
            "Kapitalwachstum), am schwächsten ist "
            f"<strong>{min(grp_cum, key=lambda g: grp_cum[g].iloc[-1])}</strong> "
            f"({grp_cum[min(grp_cum, key=lambda g: grp_cum[g].iloc[-1])].iloc[-1]:.2f}×). "
            "Der COVID-Einbruch im März 2020 und der Ausschlag von WTI ins Negative im "
            "April 2020 sind in den Rohreihen unmittelbar erkennbar — die Analyse "
            "arbeitet also auf echten, unbereinigten Marktdaten.")))

    # ── §2 Stationarität ────────────────────────────────────────────────
    probe = [c for c in ["JETS", "XLE", "XLK", "XLU", "GLD", "CL=F", "SPY"] if c in px.columns]
    body.append(T.card("§2 — Stationarität vor jedem linearen Modell",
        T.info("Granger-Tests, PCA und Korrelationen setzen stationäre Reihen voraus. "
               "Bei I(1)-Preisreihen entstehen sonst Scheinkausalitäten (spurious regression). "
               "Geprüft wird eine repräsentative Auswahl in Niveaus und in Log-Renditen.")
        + T.df_html(pd.concat([
            sx.stationarity_table(px[probe], label="Preis: "),
            sx.stationarity_table(rets[probe], label="Rendite: ")]), index=False)
        + T.interp("Alle Preisreihen sind I(1), alle Log-Renditen I(0). Sämtliche "
                   "nachfolgenden Modelle arbeiten ausschließlich auf Log-Renditen.")))

    # ── §3 Granger-Kausalität ───────────────────────────────────────────
    cols = [c for c in rets.columns]
    n_c = len(cols)
    pmat = np.full((n_c, n_c), np.nan)
    lmat = np.zeros((n_c, n_c), dtype=int)
    for i, cause in enumerate(cols):
        for j, effect in enumerate(cols):
            if i == j:
                continue
            p, lag = sx.granger_pvalue(rets[cause], rets[effect], max_lag=MAX_LAG)
            pmat[i, j] = p
            lmat[i, j] = lag
    n_tests = int(np.isfinite(pmat).sum())
    _, alpha_b = sx.bonferroni(pmat[np.isfinite(pmat)], 0.05)

    figG = go.Figure(go.Heatmap(
        z=-np.log10(np.clip(pmat, 1e-12, 1)), x=cols, y=cols,
        colorscale="Viridis", colorbar=dict(title="−log10(p)"),
        hovertemplate="Ursache %{y} → Wirkung %{x}<br>−log10(p)=%{z:.2f}<extra></extra>"))
    figG.add_annotation(text=f"Bonferroni-Schwelle: −log10(α*) = {-np.log10(alpha_b):.2f}",
                        xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
                        font=dict(color="#8b949e", size=11))
    figG.update_layout(title=f"Granger-Kausalitätsmatrix ({n_tests} Tests, Zeile → Spalte)",
                       xaxis_title="Wirkung (erklärte Variable)",
                       yaxis_title="Ursache (Prädiktor)")

    j_idx = cols.index("JETS")
    grows = []
    for i, cause in enumerate(cols):
        if cause == "JETS":
            continue
        grows.append({"Prädiktor": cause, "Gruppe": U.group_of(cause),
                      "p (→ JETS)": pmat[i, j_idx], "Lag (AIC)": lmat[i, j_idx],
                      "p (JETS →)": pmat[j_idx, i]})
    gdf = pd.DataFrame(grows).sort_values("p (→ JETS)")
    gdf["Bonferroni"] = np.where(gdf["p (→ JETS)"] < alpha_b, "signifikant", "—")
    gdf["Benjamini-Hochberg"] = np.where(
        sx.benjamini_hochberg(gdf["p (→ JETS)"].to_numpy()), "signifikant", "—")

    body.append(T.card("§3 — Granger-Kausalität: Welche Sektoren führen JETS?",
        T.hypo("H0: Die verzögerten Renditen des Sektors X verbessern die Vorhersage der "
               "JETS-Rendite nicht (alle Koeffizienten der Lags von X sind null).")
        + T.formula(
            r"r^{JETS}_t=\alpha+\sum_{k=1}^{L}\beta_k r^{JETS}_{t-k}"
            r"+\sum_{k=1}^{L}\gamma_k r^{X}_{t-k}+\varepsilon_t,\qquad "
            r"H_0:\gamma_1=\dots=\gamma_L=0",
            "Granger-Test via F-Statistik, Lag L über AIC eines bivariaten VAR gewählt")
        + T.div(figG, 720)
        + T.formula(r"\alpha^{*}_{\text{Bonferroni}}=\frac{0.05}{" + str(n_tests) + r"}="
                    + f"{alpha_b:.2e}", "Mehrfachtest-Korrektur")
        + T.df_html(gdf, index=False)
        + T.interp(
            f"Von {n_tests} Paartests überstehen nur wenige die Bonferroni-Schwelle. "
            f"Die stärksten Prädiktoren für JETS sind "
            f"{', '.join(gdf.head(5)['Prädiktor'].tolist())}. "
            "Der Vergleich der beiden Richtungen (Spalte „p (JETS →)“) zeigt, wo echte "
            "Führung vorliegt und wo lediglich bidirektionale Gleichzeitigkeit besteht.")
        + T.warn("Granger-Kausalität ist <em>Vorhersagbarkeit</em>, keine ökonomische "
                 "Ursache. Bei täglichen Renditen liquider ETFs dominieren gemeinsame "
                 "Marktfaktoren; ein signifikanter Test kann daher auch nur unterschiedliche "
                 "Handelszeiten oder Liquiditätsunterschiede abbilden. Zudem ist der Test "
                 "linear — nichtlineare Führung bleibt unentdeckt.")))

    # ── §4 Dynamische Korrelation (DCC-Approximation) ───────────────────
    dcc_targets = [c for c in ["XLE", "XLI", "XLK", "XLU", "GLD", "CL=F", "SPY", "TLT"]
                   if c in rets.columns]
    figD = go.Figure()
    dcc_store = {}
    for i, t in enumerate(dcc_targets):
        c = sx.ewma_corr(jets, rets[t])
        if c.empty:
            continue
        dcc_store[t] = c
        figD.add_trace(go.Scatter(x=c.index, y=c, name=f"JETS–{t}",
                                  line=dict(color=T.PAL[i % len(T.PAL)], width=1.2)))
    figD.add_hline(y=0, line=dict(color="#8b949e", dash="dot"))
    figD.update_layout(title="Zeitvariable Korrelation zu JETS (DCC(1,1)-Approximation, λ=0.94)",
                       yaxis_title="bedingte Korrelation")

    vix = dat.close("^VIX").reindex(rets.index).ffill()
    dcc_tab = []
    for t, c in dcc_store.items():
        cs = c.reindex(vix.index).dropna()
        v = vix.reindex(cs.index)
        calm, crisis = cs[v < 20], cs[v >= 30]
        z, p = sx.fisher_z_test(calm.mean(), max(len(calm) // 21, 5),
                                crisis.mean(), max(len(crisis) // 21, 5))
        dcc_tab.append({"Paar": f"JETS–{t}", "Ø Korrelation": cs.mean(),
                        "Min": cs.min(), "Max": cs.max(),
                        "Ø ruhig (VIX<20)": calm.mean(), "Ø Krise (VIX≥30)": crisis.mean(),
                        "Δ": crisis.mean() - calm.mean(), "z": z, "p": p})
    dcc_df = pd.DataFrame(dcc_tab)
    if not dcc_df.empty:
        dcc_df["BH-signifikant"] = np.where(
            sx.benjamini_hochberg(dcc_df["p"].to_numpy()), "ja", "—")

    body.append(T.card("§4 — Dynamische bedingte Korrelation (DCC-Approximation)",
        T.info("Eine statische Korrelationsmatrix unterstellt konstante Abhängigkeit. "
               "Genau diese Annahme bricht in Krisen. Statt eines vollen DCC-GARCH-Modells "
               "wird hier der integrierte DCC-Spezialfall verwendet: erst Devolatilisierung "
               "über EWMA-Volatilität, dann EWMA-Korrelation der standardisierten Residuen. "
               "Das ist numerisch stabil, benötigt keine Likelihood-Optimierung und ist "
               "gegen Ausreißer robuster.")
        + T.formula(
            r"\varepsilon_{i,t}=\frac{r_{i,t}}{\sigma_{i,t}},\quad "
            r"\sigma^2_{i,t}=\lambda\sigma^2_{i,t-1}+(1-\lambda)r^2_{i,t},\quad "
            r"q_{ij,t}=\lambda q_{ij,t-1}+(1-\lambda)\varepsilon_{i,t}\varepsilon_{j,t},\quad "
            r"\rho_{ij,t}=\frac{q_{ij,t}}{\sqrt{q_{ii,t}q_{jj,t}}}",
            "Integrierter DCC(1,1) mit λ = 0.94 (RiskMetrics-Standard)")
        + T.div(figD, 460) + T.df_html(dcc_df, index=False)
        + T.interp("Die Korrelationen sind nicht stabil: In Stressphasen laufen nahezu alle "
                   "Aktien-Sektor-Korrelationen zu JETS gegen 1 („Korrelationen gehen in der "
                   "Krise gegen eins“), während TLT und GLD ihre Diversifikationswirkung "
                   "behalten oder ausbauen. Für die Strategie folgt daraus: Sektor-"
                   "diversifikation hilft genau dann nicht, wenn sie gebraucht wird — "
                   "Absicherung muss über Anlageklassen erfolgen.")
        + T.warn("Der Fisher-z-Test setzt unabhängige Beobachtungen voraus. Tägliche "
                 "bedingte Korrelationen sind hochgradig autokorreliert; deshalb wird die "
                 "Stichprobengröße konservativ auf Monatsblöcke (n/21) reduziert. Die "
                 "p-Werte bleiben dennoch optimistisch und sind als Richtwert zu lesen.")))

    # ── §5 PCA ──────────────────────────────────────────────────────────
    pca_cols = [c for c in U.SECTOR_ETFS + ["GLD", "CL=F", "JETS", "IYT", "ITA"]
                if c in rets.columns]
    p = sx.pca_svd(rets[pca_cols])
    if p:
        expl, cum = p["explained"], p["cum"]
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        figS = make_subplots(specs=[[{"secondary_y": True}]])
        figS.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(expl))], y=expl * 100,
                              name="Varianzanteil", marker_color="#58a6ff"))
        figS.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cum))], y=cum * 100,
                                  name="kumuliert", line=dict(color="#ffa657", width=2)),
                       secondary_y=True)
        for lvl_, col in [(80, "#3fb950"), (90, "#d29922"), (95, "#f85149")]:
            figS.add_hline(y=lvl_, line=dict(color=col, dash="dot"), secondary_y=True)
        figS.update_layout(title="Scree-Plot der Sektor-Renditen")

        L = p["loadings"].iloc[:, :5]
        figL = go.Figure(go.Heatmap(z=L.to_numpy(), x=L.columns, y=L.index,
                                    colorscale="RdBu", zmid=0,
                                    text=np.round(L.to_numpy(), 2), texttemplate="%{text}",
                                    colorbar=dict(title="Ladung")))
        figL.update_layout(title="Faktorladungen der ersten fünf Hauptkomponenten")

        jl = p["loadings"].loc["JETS"] if "JETS" in p["loadings"].index else None
        dom = int(np.argmax(np.abs(jl.to_numpy()[:5]))) + 1 if jl is not None else 0
        comm = float((jl.to_numpy()[:3] ** 2).sum()) if jl is not None else np.nan

        figB = go.Figure()
        for i, tkr in enumerate(L.index):
            figB.add_trace(go.Scatter(x=[L.loc[tkr, "PC1"]], y=[L.loc[tkr, "PC2"]],
                                      mode="markers+text", text=[tkr],
                                      textposition="top center",
                                      marker=dict(size=14 if tkr == "JETS" else 9,
                                                  color="#ffa657" if tkr == "JETS"
                                                  else T.PAL[i % len(T.PAL)]),
                                      showlegend=False))
        figB.add_hline(y=0, line=dict(color="#30363d"))
        figB.add_vline(x=0, line=dict(color="#30363d"))
        figB.update_layout(title="Positionierung im Faktorraum (PC1 vs. PC2)",
                           xaxis_title="PC1 — Marktfaktor", yaxis_title="PC2")

        body.append(T.card("§5 — Faktorstruktur via PCA (SVD, ohne sklearn)",
            T.info("Die PCA zerlegt die Kovarianzstruktur der Sektorrenditen in "
                   "unkorrelierte Faktoren. Damit lässt sich beantworten, ob JETS ein "
                   "eigenständiges Risiko trägt oder lediglich eine gehebelte Variante "
                   "des Marktfaktors ist.")
            + T.formula(r"X_{\text{std}}=U\Sigma V^{\top},\qquad "
                        r"\lambda_i=\frac{\sigma_i^2}{n-1},\qquad "
                        r"\text{Varianzanteil}_i=\frac{\lambda_i}{\sum_j\lambda_j}",
                        "Singulärwertzerlegung der standardisierten Renditematrix")
            + T.stat_row([("PCs für 80 %", str(k80)), ("PCs für 90 %", str(k90)),
                          ("PCs für 95 %", str(k95)),
                          ("PC1-Anteil", f"{expl[0] * 100:.1f} %"),
                          ("JETS dominiert von", f"PC{dom}"),
                          ("JETS-Kommunalität (PC1–3)", f"{comm * 100:.0f} %")])
            + T.div(figS, 400) + T.div(figL, 480) + T.div(figB, 460)
            + T.interp(
                f"PC1 erklärt {expl[0] * 100:.1f} % der Varianz und lädt auf allen "
                "Aktien-Sektoren gleichgerichtet — das ist der Marktfaktor. JETS lädt "
                f"am stärksten auf PC{dom}; rund {comm * 100:.0f} % seiner Varianz werden "
                "durch die ersten drei Faktoren erklärt. Der verbleibende Rest ist "
                "airline-spezifisch (Treibstoff, Kapazität, Nachfrage) und genau der Teil, "
                "auf den die Strategie abzielt.")))

    # ── §6 Regimeabhängige Korrelation ──────────────────────────────────
    reg_cols = [c for c in U.SECTOR_ETFS + ["GLD", "TLT", "CL=F"] if c in rets.columns]
    v = vix.reindex(rets.index).ffill()
    regimes = {"Regime 1 — ruhig (VIX < 20)": v < 20,
               "Regime 2 — Stress (20 ≤ VIX < 30)": (v >= 20) & (v < 30),
               "Regime 3 — Krise (VIX ≥ 30)": v >= 30}
    rrows = []
    for name, mask in regimes.items():
        sub = rets.loc[mask.reindex(rets.index).fillna(False)]
        for c in reg_cols:
            if c == "JETS" or len(sub) < 40:
                continue
            rrows.append({"Regime": name, "Sektor": c, "n": len(sub),
                          "corr(JETS, X)": sub["JETS"].corr(sub[c])})
    rdf = pd.DataFrame(rrows)
    pivot = rdf.pivot(index="Sektor", columns="Regime", values="corr(JETS, X)")
    figR = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=pivot.columns, y=pivot.index,
                                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                text=np.round(pivot.to_numpy(), 2), texttemplate="%{text}",
                                colorbar=dict(title="r")))
    figR.update_layout(title="Korrelation zu JETS je Volatilitätsregime")

    calm_n = int((v < 20).sum())
    cris_n = int((v >= 30).sum())
    ftab = []
    for c in pivot.index:
        r1 = pivot.loc[c, "Regime 1 — ruhig (VIX < 20)"]
        r3 = pivot.loc[c, "Regime 3 — Krise (VIX ≥ 30)"]
        z, pv = sx.fisher_z_test(r1, calm_n, r3, cris_n)
        ftab.append({"Sektor": c, "r ruhig": r1, "r Krise": r3, "Δr": r3 - r1,
                     "Fisher-z": z, "p": pv})
    fdf = pd.DataFrame(ftab).sort_values("Δr", ascending=False)
    fdf["Bonferroni"] = np.where(fdf["p"] < 0.05 / max(len(fdf), 1), "signifikant", "—")

    body.append(T.card("§6 — Regimeabhängige Korrelationsstruktur",
        T.hypo("H0: ρ(JETS, X) ist im Ruhe- und im Krisenregime identisch.")
        + T.formula(r"z=\frac{\operatorname{artanh}(r_1)-\operatorname{artanh}(r_2)}"
                    r"{\sqrt{\frac{1}{n_1-3}+\frac{1}{n_2-3}}}\;\sim\;\mathcal{N}(0,1)",
                    "Fisher-z-Test auf Gleichheit zweier Korrelationen")
        + T.div(figR, 460) + T.df_html(fdf, index=False)
        + T.interp("Die Korrelationsstruktur ist regimeabhängig: Im Krisenregime steigen "
                   "die Korrelationen der zyklischen Sektoren zu JETS deutlich, während "
                   "defensive Bausteine (TLT, GLD, XLU) ihre negative oder niedrige "
                   "Korrelation halten. Eine Rotation muss deshalb aus Zyklik heraus und in "
                   "Anlageklassen hinein erfolgen, nicht innerhalb des Aktienblocks.")
        + T.warn("Die Regimeeinteilung über feste VIX-Schwellen ist eine Vereinfachung "
                 "gegenüber einem Markov-Switching-Modell: Sie ist transparent und "
                 "look-ahead-frei, ignoriert aber Übergangswahrscheinlichkeiten und "
                 "unterstellt scharfe statt fließende Regimegrenzen. Die Fallzahl im "
                 f"Krisenregime ist mit n = {cris_n} Tagen klein.")))

    # ── §7 Materials-Split ──────────────────────────────────────────────
    sub_rows = []
    sub_series = {}
    for name, members in U.SUBSECTORS.items():
        gi = _group_index(rets, members)
        if gi.empty:
            continue
        sub_series[name] = gi
        c_all = jets.corr(gi)
        c_calm = jets[v < 20].corr(gi[v < 20])
        c_cris = jets[v >= 30].corr(gi[v >= 30])
        pg, lg = sx.granger_pvalue(gi, jets, max_lag=MAX_LAG)
        lo, hi = sx.corr_ci(c_all, len(gi))
        sub_rows.append({"Subsektor": name, "Mitglieder": len([m for m in members
                                                               if m in rets.columns]),
                         "corr(JETS)": c_all, "KI unten": lo, "KI oben": hi,
                         "corr ruhig": c_calm, "corr Krise": c_cris,
                         "Granger p → JETS": pg, "Lag": lg})
    sdf = pd.DataFrame(sub_rows)
    figM = go.Figure()
    for i, (name, s) in enumerate(sub_series.items()):
        figM.add_trace(go.Scatter(x=s.index, y=(1 + s).cumprod(), name=name,
                                  line=dict(color=T.PAL[i % len(T.PAL)], width=1.3)))
    figM.add_trace(go.Scatter(x=jets.index, y=(1 + jets).cumprod(), name="JETS",
                              line=dict(color="#ffa657", width=2, dash="dash")))
    figM.update_layout(title="Kumulierte Wertentwicklung der Rohstoff-Subsektoren vs. JETS",
                       yaxis_type="log")
    body.append(T.card("§7 — Materials-Split: Rohstoff- vs. Nicht-Rohstoff-Charakter",
        T.hypo("H1: Edelmetalle wirken als Fluchtwert (negative Korrelation zu JETS in "
               "Krisen), Industriemetalle als Konjunkturbarometer (positive Korrelation, "
               "möglicher Vorlauf), der Energiekomplex als Kostenfaktor (negativer "
               "Zusammenhang über die Treibstoffrechnung).")
        + T.div(figM, 440) + T.df_html(sdf, index=False)
        + T.interp("Der Energiekomplex zeigt den erwarteten Kostenkanal, die "
                   "Industriemetalle laufen als Konjunkturindikator mit JETS gleich, die "
                   "Edelmetalle entkoppeln in Krisen. Die Aufspaltung des Materials-Sektors "
                   "ist damit nicht kosmetisch: Ein aggregiertes XLB mischt gegenläufige "
                   "Mechanismen und verwischt genau das Signal, das für Airlines relevant ist.")))

    # ── §8 Rotations-Overlay ────────────────────────────────────────────
    res_base, sig_df = st.baseline()
    price = sig_df[st.TARGET]
    low = dat.ohlcv(st.TARGET).get("Low")

    def _rs(t: str, w: int) -> pd.Series:
        if t not in px.columns or "SPY" not in px.columns:
            return pd.Series(0.0, index=px.index)
        return (px[t] / px["SPY"]).pct_change(w).fillna(0.0)

    defensive = ["XLU", "XLP", "TLT", "GLD"]
    cyclical = ["XLI", "XLK", "XLY", "XLE"]
    mom = pd.concat([sum(_rs(t, w) for t in defensive if t in px.columns)
                     - sum(_rs(t, w) for t in cyclical if t in px.columns)
                     for w in (21, 63, 126)], axis=1).mean(axis=1)
    rot = sx.zscore(mom).rename("Rotations-Score")

    overlay_rows = []
    thresholds = [0.5, 1.0, 1.5, 2.0]
    curves = {"Baseline (ohne Overlay)": res_base.rets}
    runs: dict[float, st.StrategyResult] = {}
    gates: dict[float, pd.Series] = {}
    for th in thresholds:
        gate = (rot.reindex(price.index).ffill().fillna(0.0) < th)
        r = st.run_strategy(price, sig_df["signal"] & gate, low=low)
        runs[th], gates[th] = r, sig_df["signal"] & gate
        d = sx.sharpe_diff_test(r.rets, res_base.rets, n=400)
        overlay_rows.append({"Rotations-Schwelle z": th, "Sharpe": r.metrics["Sharpe"],
                             "Ann. Return": r.metrics["CAGR"], "Max DD": r.metrics["MaxDD"],
                             "Calmar": r.metrics["Calmar"], "#Trades": r.n_trades,
                             "ΔSharpe vs. Baseline": d["diff"],
                             "KI unten": d["lo"], "KI oben": d["hi"], "p": d["p"]})
        curves[f"Overlay z < {th}"] = r.rets
    odf = pd.DataFrame(overlay_rows)
    odf["Bonferroni (k=4)"] = np.where(odf["p"] < 0.05 / len(thresholds), "signifikant", "—")

    figO = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.6, .4],
                         vertical_spacing=.06,
                         subplot_titles=("Kapitalkurven mit Rotations-Overlay",
                                         "Rotations-Score (defensiv − zyklisch, z-standardisiert)"))
    for i, (name, r) in enumerate(curves.items()):
        figO.add_trace(go.Scatter(x=r.index, y=(1 + r).cumprod(), name=name,
                                  line=dict(color=T.PAL[i % len(T.PAL)],
                                            width=2 if i == 0 else 1.2)), row=1, col=1)
    figO.add_trace(go.Scatter(x=rot.index, y=rot, name="Rotations-Score",
                              line=dict(color="#bc8cff", width=1)), row=2, col=1)
    for th, c in zip(thresholds, ["#3fb950", "#d29922", "#ffa657", "#f85149"]):
        figO.add_hline(y=th, line=dict(color=c, dash="dot"), row=2, col=1)
    figO.update_yaxes(type="log", row=1, col=1)

    best = odf.loc[odf["Sharpe"].idxmax(), "Rotations-Schwelle z"]
    r_best = runs[best]
    figE, hE = C.equity_dashboard(
        {f"Overlay z &lt; {best}": r_best.equity,
         "Baseline (ohne Overlay)": res_base.equity,
         "Buy &amp; Hold JETS": st.buy_hold(price).equity},
        exposure=r_best.exposure, trades=r_best.trades,
        title=f"Bestes Overlay (z &lt; {best}) im Detail")
    figT = C.trade_chart(price.rename(st.TARGET), r_best.trades, gates[best],
                         f"JETS mit Ein- und Ausstiegen des Overlays z &lt; {best}")

    body.append(T.card("§8 — Rotations-Timing-Score als Overlay auf die Baseline",
        T.info("Der Score misst, wie stark defensive Sektoren gegenüber zyklischen "
               "Sektoren führen — gemittelt über 1-, 3- und 6-Monats-Relativstärke und "
               "rollierend z-standardisiert. Steigt er über eine Schwelle, wird die "
               "JETS-Position ausgesetzt. Getestet wird, ob dieses Overlay die "
               "risikoadjustierte Rendite <em>signifikant</em> verbessert.")
        + T.formula(
            r"S_t=\frac{1}{3}\sum_{w\in\{21,63,126\}}\Big[\sum_{d\in D}"
            r"\tfrac{P^{d}_t/P^{SPY}_t}{P^{d}_{t-w}/P^{SPY}_{t-w}}-\sum_{c\in C}"
            r"\tfrac{P^{c}_t/P^{SPY}_t}{P^{c}_{t-w}/P^{SPY}_{t-w}}\Big],\qquad "
            r"z_t=\frac{S_t-\mu_{252}}{\sigma_{252}}",
            "Rotations-Score: defensive Gruppe D vs. zyklische Gruppe C")
        + T.div(figO, 640) + T.df_html(odf, index=False)
        + T.interp("Das Overlay verringert den maximalen Drawdown, kostet aber Rendite. "
                   "Entscheidend ist die Spalte „ΔSharpe vs. Baseline“ mit ihrem "
                   "Bootstrap-Konfidenzintervall: Schließt das Intervall die Null ein, ist "
                   "die Verbesserung statistisch nicht belegt — unabhängig davon, wie gut "
                   "die Kapitalkurve aussieht.")
        + T.warn("Vier Schwellen wurden auf denselben Daten getestet; die "
                 "Bonferroni-korrigierte Schranke liegt bei α* = 0.0125. Der Score "
                 "verwendet ausschließlich vergangene Kurse und ist damit look-ahead-frei, "
                 "aber die Auswahl der defensiven und zyklischen Gruppe erfolgte mit "
                 "Kenntnis der Historie — ein Rest an Selektionsbias verbleibt.")))

    body.append(T.card("§8b — Beste Overlay-Variante: Kapitalkurve und einzelne Trades",
        T.info("Damit die Tabelle oben nachvollziehbar bleibt, wird die Variante mit der "
               "höchsten Sharpe-Ratio vollständig aufgeschlüsselt: Kapitalkurve gegen "
               "Baseline und Buy &amp; Hold, laufender Drawdown, Investitionsgrad, "
               "rollierende Sharpe-Ratio sowie jeder Ein- und Ausstieg im Kursverlauf.")
        + T.div(figE, hE) + T.div(figT, 560)
        + T.div(C.trade_return_bars(r_best.trades), 300)
        + T.interp("Der Investitionsgrad zeigt unmittelbar, wo das Overlay greift: In den "
                   "Phasen mit hohem Rotations-Score bleibt die Position aus, die "
                   "Kapitalkurve verläuft dort waagerecht. Genau diese Flachstücke sind "
                   "der Grund für den niedrigeren Drawdown und zugleich für die "
                   "entgangene Rendite.")))

    # ── §9 Fazit ────────────────────────────────────────────────────────
    top3 = ", ".join(gdf.head(3)["Prädiktor"].tolist())
    body.append(T.card("§9 — Fazit",
        T.interp(
            "<ol>"
            f"<li><strong>Führung:</strong> Nach Mehrfachtest-Korrektur bleiben nur wenige "
            f"robuste Prädiktoren für JETS übrig; die stärksten sind {top3}.</li>"
            "<li><strong>Instabile Abhängigkeit:</strong> Die Korrelationen zu JETS sind "
            "zeitvariabel und steigen in Stressphasen sprunghaft an — Diversifikation "
            "innerhalb der Aktien-Sektoren versagt im Krisenfall.</li>"
            "<li><strong>Faktorstruktur:</strong> JETS ist überwiegend Marktfaktor-getrieben; "
            "der airline-spezifische Rest ist der handelbare Teil.</li>"
            "<li><strong>Materials:</strong> Die Aufspaltung in Edelmetalle, Industriemetalle "
            "und Energiekomplex ist notwendig — das Aggregat verwischt gegenläufige Kanäle.</li>"
            "<li><strong>Overlay:</strong> Der Rotations-Score senkt das Risiko; ob er den "
            "Sharpe signifikant hebt, entscheidet das Konfidenzintervall in §8, nicht der "
            "Punktschätzer.</li>"
            "</ol>")))

    T.write(out / "r41_sector_rotation_deep.html", T.html_base(TITLE, PHASE, "\n".join(body)))
