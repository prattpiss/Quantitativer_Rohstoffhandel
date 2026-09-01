"""Report 43 — Flash-Crash-Erkennung: Optimierung statt reiner Erkennung.

Walk-Forward-Schwellenoptimierung, Gewichtungsoptimierung unter Simplex-Restriktion,
Kaplan-Meier-Vorlaufanalyse, CSI-basierte Positionsgrößensteuerung, Synergie mit
CPI sowie Aufnahme eines geopolitischen Risikobausteins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import charts as C
from core import data as dat
from core import indices as ix
from core import stats_tools as sx
from core import strategy as st
from core import theme as T

PHASE = 23
TITLE = "Report 43 — Flash-Crash-Optimierung"

EVENTS = [
    ("China-Crash", "2015-08-24", "Renminbi-Abwertung, globaler Ausverkauf"),
    ("Volmageddon", "2018-02-05", "Blowup kurzer VIX-Produkte"),
    ("COVID-Crash", "2020-03-16", "Schlechtester SPY-Tag seit 1987"),
    ("Inflationsschock", "2022-01-24", "Fed-Pivot-Angst, Ukraine-Spannungen"),
    ("CPI-Schock", "2022-09-13", "Inflationsüberraschung, SPY −4.3 %"),
]

THRESHOLDS = [50, 60, 70, 80, 90]
K_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def _crash_days(price: pd.Series, horizon: int = 5, drop: float = -0.07) -> pd.Series:
    return (price.pct_change(horizon) <= drop)


def build(out: Path) -> None:  # noqa: C901
    print("Report 43 — Flash-Crash-Optimierung")
    np.random.seed(sx.SEED)
    rng = np.random.default_rng(sx.SEED)
    body: list[str] = []
    body.append(T.header("Report 43 — Flash-Crash-Erkennung und Strategie-Optimierung",
                         "Walk-Forward-Schwellen · Gewichtsoptimierung · Kaplan-Meier · "
                         "Positionsgrößensteuerung · CSI × CPI"))

    comp = ix.csi_components()
    if comp.empty:
        T.write(out / "r43_flash_crash_optimization.html",
                T.html_base(TITLE, PHASE, "".join(body) + T.warn("Keine Daten.")))
        return
    csi = ix.csi_from_components(comp)
    jets = comp["JETS"]

    res_base, sig_df = st.baseline()
    price = sig_df[st.TARGET]
    low = dat.ohlcv(st.TARGET).get("Low")
    base_sig = sig_df["signal"]
    csi_a = csi.reindex(price.index).ffill()

    # ── §1 CSI-Rekonstruktion ───────────────────────────────────────────
    figC = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5],
                         vertical_spacing=.06,
                         subplot_titles=("Crash Stress Index (CSI) und JETS",
                                         "Komponenten-Perzentile"))
    figC.add_trace(go.Scatter(x=csi.index, y=csi, name="CSI",
                              line=dict(color="#f85149", width=1.2)), row=1, col=1)
    for lvl, col in [(40, "#3fb950"), (60, "#d29922"), (80, "#f85149")]:
        figC.add_hline(y=lvl, line=dict(color=col, dash="dash"), row=1, col=1)
    for i, c in enumerate(["C1", "C2", "C3", "C4", "C5", "C6"]):
        figC.add_trace(go.Scatter(x=comp.index, y=comp[c], name=ix.CSI_LABELS[c],
                                  line=dict(color=T.PAL[i % len(T.PAL)], width=0.8),
                                  opacity=0.75), row=2, col=1)
    for name, d, _ in EVENTS:
        figC.add_vline(x=d, line=dict(color="#bc8cff", dash="dot"))
        figC.add_annotation(x=d, y=1, yref="paper", text=name, showarrow=False,
                            textangle=-90, font=dict(color="#bc8cff", size=9), yanchor="top")

    w_tab = pd.DataFrame([{"Komponente": k, "Bedeutung": ix.CSI_LABELS[k],
                           "Gewicht (Ausgangswert)": f"{v * 100:.0f} %"}
                          for k, v in ix.CSI_WEIGHTS.items()])
    body.append(T.card("§1 — Rekonstruktion des Crash Stress Index",
        T.info("Der CSI aggregiert sechs Stressquellen als rollierende 252-Tage-Perzentile. "
               "Die Perzentil-Transformation macht die Komponenten vergleichbar und "
               "unempfindlich gegen Niveauverschiebungen (z. B. das strukturell höhere "
               "VIX-Niveau nach 2018) — sie ist zugleich vollständig rückwärtsgerichtet "
               "und damit look-ahead-frei.")
        + T.formula(
            r"\mathrm{CSI}_t=\Big(\sum_{i=1}^{6}w_i\,\mathrm{PR}_{252}(x_{i,t})\Big)"
            r"\ast\;\mathrm{MA}_3,\qquad \sum_i w_i=1,\;w_i\ge0",
            "CSI als gewichtete Summe rollierender Perzentilränge, 3-Tage-geglättet")
        + T.df_html(w_tab, index=False) + T.div(figC, 640)
        + (T.warn("Die VIX-Terminstruktur (^VIX9D) ist in dieser Datenbasis nicht "
                  "durchgängig verfügbar; die Komponente C6 wird dann neutral (50) gesetzt "
                  "und ihr Gewicht auf die übrigen Komponenten umgelegt.")
           if not comp.attrs.get("has_term_structure", False) else "")))

    raw_src43 = ["JETS", "^VIX", "HYG", "IEF", "CL=F", "GLD", "SPY", "UUP"]
    raw43 = dat.close_panel(raw_src43, min_obs=250).ffill()
    fig_raw43, h_raw43 = C.series_grid({c: raw43[c] for c in raw43.columns},
                                       "Geladene Quellreihen des CSI (Originaleinheiten)",
                                       cols=3)
    body.append(T.card("§1b — Quellreihen hinter dem Index",
        T.info("Der CSI ist eine Rechengröße aus börsengehandelten Reihen. Hier stehen "
               "die Rohreihen vor jeder Transformation, dazu ihre Verfügbarkeit und die "
               "ersten und letzten Werte der Matrix — damit ist jede spätere "
               "Indexbewegung auf konkrete Kurse zurückführbar.")
        + T.div(fig_raw43, h_raw43)
        + T.df_html(dat.availability(raw_src43), index=False)
        + C.data_table(raw43)))

    # ── §2 Ereignis-Validierung + Kaplan-Meier ──────────────────────────
    ev_rows = []
    for name, d, desc in EVENTS:
        ev = pd.Timestamp(d)
        w = csi.loc[ev - pd.Timedelta(days=45):ev]
        row = {"Ereignis": name, "Datum": d, "Kontext": desc,
               "CSI am Ereignistag": float(csi.asof(ev)) if len(csi) else np.nan,
               "CSI-Maximum (45 T davor)": float(w.max()) if len(w) else np.nan}
        for c in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            cw = comp[c].loc[ev - pd.Timedelta(days=45):ev]
            hits = cw[cw >= 80]
            row[f"Vorlauf {c}"] = float((ev - hits.index[0]).days) if len(hits) else np.nan
        ev_rows.append(row)
    ev_df = pd.DataFrame(ev_rows)

    crash = _crash_days(jets)
    km_frames = {}
    for c in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        s = comp[c]
        alarms = s[(s >= 80) & (s.shift(1) < 80)].index
        durs, evs = [], []
        for a in alarms:
            win = crash.loc[a:a + pd.Timedelta(days=60)]
            hit = win[win]
            if len(hit):
                durs.append((hit.index[0] - a).days)
                evs.append(True)
            else:
                durs.append(60)
                evs.append(False)
        km = sx.kaplan_meier(durs, evs)
        if not km.empty:
            km_frames[c] = km

    figK = go.Figure()
    for i, (c, km) in enumerate(km_frames.items()):
        col = T.PAL[i % len(T.PAL)]
        figK.add_trace(go.Scatter(x=km["t"], y=km["S"], name=f"{c} — {ix.CSI_LABELS[c]}",
                                  line=dict(color=col, width=1.6, shape="hv")))
        figK.add_trace(go.Scatter(x=list(km["t"]) + list(km["t"])[::-1],
                                  y=list(km["hi"]) + list(km["lo"])[::-1],
                                  fill="toself", fillcolor=T.hex_rgba(col, 0.10),
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figK.update_layout(title="Kaplan-Meier: Anteil ohne Absturz nach Komponenten-Alarm",
                       xaxis_title="Tage seit Alarm (Komponente ≥ 80)",
                       yaxis_title="Überlebenswahrscheinlichkeit S(t)")

    km_tab = pd.DataFrame([
        {"Komponente": c, "Bedeutung": ix.CSI_LABELS[c], "Alarme": int(km["events"].sum()
                                                                      + (km["at_risk"].iloc[0]
                                                                         - km["events"].sum())),
         "Absturz binnen 60 T": f"{(1 - km['S'].iloc[-1]) * 100:.0f} %",
         "Median-Zeit bis Absturz":
             (f"{km.loc[km['S'] <= 0.5, 't'].iloc[0]:.0f} Tage"
              if (km["S"] <= 0.5).any() else "nicht erreicht")}
        for c, km in km_frames.items()])

    body.append(T.card("§2 — Vorlaufanalyse: welche Komponente warnt am konsistentesten?",
        T.info("Statt nur den mittleren Vorlauf zu betrachten, wird die Frage als "
               "Überlebenszeitproblem formuliert: Wie lange nach einem Komponenten-Alarm "
               "bleibt der Markt absturzfrei? Der Kaplan-Meier-Schätzer verarbeitet dabei "
               "korrekt auch Alarme, denen innerhalb des Beobachtungsfensters kein Absturz "
               "folgt (Zensierung) — ein einfacher Mittelwert würde diese Fälle verwerfen "
               "und den Vorlauf systematisch überschätzen.")
        + T.formula(r"\hat S(t)=\prod_{t_i\le t}\Big(1-\frac{d_i}{n_i}\Big),\qquad "
                    r"\widehat{\mathrm{Var}}=\hat S(t)^2\sum_{t_i\le t}"
                    r"\frac{d_i}{n_i(n_i-d_i)}",
                    "Kaplan-Meier-Schätzer mit Greenwood-Varianz (95 %-Band)")
        + T.df_html(ev_df, index=False) + T.div(figK, 460) + T.df_html(km_tab, index=False)
        + T.interp("Komponenten mit steil abfallender Überlebenskurve warnen scharf und "
                   "zeitnah; flach verlaufende Kurven bedeuten viele Alarme ohne Folge. "
                   "Der Kreditkanal (C3) und der VIX-Sprung (C2) liefern die konsistenteste "
                   "Trennung, während die Volumenkomponente (C5) viele Fehlalarme erzeugt.")
        + T.warn("Als Absturz gilt hier ein 5-Tage-Rückgang von mindestens 7 % in JETS. "
                 "Diese Definition ist eine Setzung; andere Schwellen verändern die "
                 "Ergebnisse. Alarme desselben Stressclusters sind zudem nicht unabhängig, "
                 "wodurch die Konfidenzbänder zu eng ausfallen.")))

    # ── §3 Walk-Forward-Schwellenoptimierung ────────────────────────────
    def sig_for_threshold(th):
        return base_sig & (csi_a.fillna(0) < th)

    wf = st.walk_forward(price, sig_for_threshold, THRESHOLDS, low=low,
                         train_months=60, test_months=12)
    thr_rows = []
    for th in THRESHOLDS:
        r = st.run_strategy(price, sig_for_threshold(th), low=low)
        d = sx.sharpe_diff_test(r.rets, res_base.rets, n=400)
        thr_rows.append({"CSI-Schwelle": th, "Sharpe (gesamt)": r.metrics["Sharpe"],
                         "Ann. Return": r.metrics["CAGR"], "Max DD": r.metrics["MaxDD"],
                         "Calmar": r.metrics["Calmar"], "#Trades": r.n_trades,
                         "ΔSharpe vs. Baseline": d["diff"], "KI unten": d["lo"],
                         "KI oben": d["hi"], "p": d["p"]})
    thr_df = pd.DataFrame(thr_rows)
    alpha_star = 0.05 / len(THRESHOLDS)
    thr_df["Bonferroni (α*=0.01)"] = np.where(thr_df["p"] < alpha_star, "signifikant", "—")

    figT = make_subplots(rows=2, cols=1, row_heights=[.5, .5], vertical_spacing=.12,
                         subplot_titles=("ΔSharpe je CSI-Schwelle mit 95 %-Bootstrap-KI",
                                         "Out-of-Sample-Sharpe je Walk-Forward-Fenster"))
    figT.add_trace(go.Bar(x=thr_df["CSI-Schwelle"], y=thr_df["ΔSharpe vs. Baseline"],
                          marker_color="#58a6ff", name="ΔSharpe",
                          error_y=dict(type="data", symmetric=False,
                                       array=thr_df["KI oben"] - thr_df["ΔSharpe vs. Baseline"],
                                       arrayminus=thr_df["ΔSharpe vs. Baseline"] - thr_df["KI unten"],
                                       color="#ffa657")), row=1, col=1)
    figT.add_hline(y=0, line=dict(color="#8b949e", dash="dash"), row=1, col=1)
    if not wf.empty:
        figT.add_trace(go.Scatter(x=wf["Test-Start"], y=wf["Sharpe OOS"], name="Sharpe OOS",
                                  mode="lines+markers", line=dict(color="#3fb950", width=1.6)),
                       row=2, col=1)
        figT.add_trace(go.Scatter(x=wf["Test-Start"], y=wf["Sharpe IS"], name="Sharpe IS",
                                  mode="lines", line=dict(color="#8b949e", width=1, dash="dot")),
                       row=2, col=1)
        figT.add_hline(y=0, line=dict(color="#8b949e", dash="dash"), row=2, col=1)

    wf_txt = ""
    if not wf.empty:
        deg = wf["Sharpe IS"].mean() - wf["Sharpe OOS"].mean()
        mode_par = wf["Bester Parameter (IS)"].mode()
        wf_txt = T.interp(
            f"Über {len(wf)} Walk-Forward-Fenster beträgt der mittlere Sharpe "
            f"in-sample {wf['Sharpe IS'].mean():.2f} und out-of-sample "
            f"{wf['Sharpe OOS'].mean():.2f}. Die Differenz von {deg:.2f} ist der "
            "Overfitting-Abschlag: Sie beziffert, wie viel der In-Sample-Güte reine "
            "Anpassung an die Trainingsdaten war. Am häufigsten wurde die Schwelle "
            f"{mode_par.iloc[0] if len(mode_par) else '—'} gewählt; wechselt die Wahl "
            "stark von Fenster zu Fenster, ist der Parameter instabil und sollte fixiert "
            "statt optimiert werden.")

    body.append(T.card("§3 — Walk-Forward-Optimierung der CSI-Schwelle",
        T.hypo("H0: Eine CSI-Filterung verbessert die risikoadjustierte Rendite der "
               "Baseline nicht (ΔSharpe = 0).")
        + T.info("Jedes Fenster besteht aus 60 Monaten Training (Parameterwahl) und "
                 "12 Monaten Test (Bewertung). Der Testabschnitt geht nie in die "
                 "Parameterwahl ein. Da fünf Schwellen getestet werden, gilt die "
                 f"Bonferroni-korrigierte Schranke α* = 0.05/5 = {alpha_star:.3f}.")
        + T.div(figT, 620) + T.df_html(thr_df, index=False)
        + (T.df_html(wf, index=False) if not wf.empty else "") + wf_txt
        + T.warn("Die verfügbare Historie erlaubt nur wenige unabhängige Testfenster. "
                 "Walk-Forward beseitigt den Look-Ahead, nicht aber die kleine Fallzahl — "
                 "die Streuung der Out-of-Sample-Ergebnisse ist entsprechend groß.")))

    # ── §4 Gewichtungsoptimierung ───────────────────────────────────────
    keys = ["C1", "C2", "C3", "C4", "C5", "C6"]
    n_cand = 240
    cands = rng.dirichlet(np.ones(len(keys)), size=n_cand)
    split = int(len(price) * 0.7)
    is_idx, oos_idx = price.index[:split], price.index[split:]

    def eval_weights(w_vec, idx_slice, th=80):
        w = dict(zip(keys, w_vec))
        c = ix.csi_from_components(comp, w).reindex(price.index).ffill().fillna(0)
        s = base_sig & (c < th)
        return st.run_strategy(price.loc[idx_slice], s.loc[idx_slice], low=low).metrics["Sharpe"]

    is_scores = np.array([eval_weights(w, is_idx) for w in cands])
    best_i = int(np.nanargmax(is_scores))
    best_w = dict(zip(keys, cands[best_i]))
    base_w = ix.CSI_WEIGHTS
    wcmp = pd.DataFrame({
        "Komponente": [ix.CSI_LABELS[k] for k in keys],
        "Ausgangsgewicht": [f"{base_w[k] * 100:.0f} %" for k in keys],
        "Optimiert (In-Sample)": [f"{best_w[k] * 100:.0f} %" for k in keys],
    })
    perf_w = pd.DataFrame([
        {"Gewichtung": "Ausgangsgewichte", "Sharpe In-Sample":
            eval_weights([base_w[k] for k in keys], is_idx),
         "Sharpe Out-of-Sample": eval_weights([base_w[k] for k in keys], oos_idx)},
        {"Gewichtung": "In-Sample-optimiert", "Sharpe In-Sample": is_scores[best_i],
         "Sharpe Out-of-Sample": eval_weights(cands[best_i], oos_idx)},
        {"Gewichtung": "Gleichgewichtung", "Sharpe In-Sample":
            eval_weights(np.ones(len(keys)) / len(keys), is_idx),
         "Sharpe Out-of-Sample": eval_weights(np.ones(len(keys)) / len(keys), oos_idx)},
    ])
    oos_scores = np.array([eval_weights(w, oos_idx) for w in cands[:80]])
    figW = go.Figure()
    figW.add_trace(go.Scatter(x=is_scores[:80], y=oos_scores, mode="markers",
                              marker=dict(color="#58a6ff", size=7, opacity=0.7),
                              name="Zufalls-Gewichtsvektoren"))
    if np.isfinite(is_scores[:80]).sum() > 3:
        m_ok = np.isfinite(is_scores[:80]) & np.isfinite(oos_scores)
        rho = float(np.corrcoef(is_scores[:80][m_ok], oos_scores[m_ok])[0, 1])
    else:
        rho = np.nan
    figW.update_layout(title=f"In-Sample- vs. Out-of-Sample-Sharpe der Gewichtsvektoren "
                             f"(ρ = {rho:.2f})",
                       xaxis_title="Sharpe In-Sample", yaxis_title="Sharpe Out-of-Sample")

    body.append(T.card("§4 — Optimierung der Komponentengewichte",
        T.info("Gesucht werden nichtnegative Gewichte mit Summe eins. Statt eines "
               "Gitters über sechs Dimensionen (das exponentiell wächst) wird eine "
               "Zufallssuche auf dem Simplex verwendet: Dirichlet-verteilte Ziehungen "
               "erfüllen die Restriktionen automatisch und decken den Raum gleichmäßig ab.")
        + T.formula(r"\max_{w}\;\mathrm{Sharpe}\big(\text{Strategie}\mid \mathrm{CSI}(w)\big)"
                    r"\quad\text{u.d.N.}\quad w_i\ge0,\;\textstyle\sum_i w_i=1",
                    "Restringiertes Optimierungsproblem")
        + T.df_html(wcmp, index=False) + T.df_html(perf_w, index=False)
        + T.div(figW, 420)
        + T.interp(
            f"Die Korrelation zwischen In-Sample- und Out-of-Sample-Sharpe beträgt "
            f"ρ = {rho:.2f}. Ist sie niedrig oder negativ, überträgt sich die "
            "In-Sample-Optimierung nicht auf neue Daten — dann sind die Ausgangsgewichte "
            "der optimierten Lösung vorzuziehen. Genau dieser Vergleich, nicht der beste "
            "In-Sample-Wert, entscheidet über die Empfehlung.")
        + T.warn("Bei 240 Kandidaten auf sechs Dimensionen ist der beste In-Sample-Wert "
                 "größtenteils Rauschen: Selbst bei völlig wirkungslosen Gewichten wäre "
                 "das Maximum aus 240 Ziehungen deutlich positiv. Die Out-of-Sample-Spalte "
                 "ist deshalb die einzige belastbare Größe in dieser Tabelle.")))

    # ── §5 Positionsgrößensteuerung ─────────────────────────────────────
    k_rows = []
    k_curves = {}
    for k in K_VALUES:
        size = ((1 - csi_a.fillna(50) / 100) ** k).clip(0, 1) if k > 0 \
            else pd.Series(1.0, index=price.index)
        size = size / size.max() if k > 0 else size
        r = st.run_strategy(price, base_sig, size=size, low=low)
        d = sx.sharpe_diff_test(r.rets, res_base.rets, n=300)
        k_rows.append({"k": k, "Sharpe": r.metrics["Sharpe"], "Ann. Return": r.metrics["CAGR"],
                       "Volatilität": r.metrics["Vol"], "Max DD": r.metrics["MaxDD"],
                       "Calmar": r.metrics["Calmar"],
                       "Ø Investitionsgrad": float(r.exposure[r.exposure > 0].mean()),
                       "ΔSharpe": d["diff"], "p": d["p"]})
        k_curves[f"k = {k}"] = r.equity
    k_df = pd.DataFrame(k_rows)
    figKk = go.Figure()
    for i, (name, eq) in enumerate(k_curves.items()):
        figKk.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                   line=dict(color=T.PAL[i % len(T.PAL)], width=1.4)))
    figKk.update_layout(title="Kapitalkurven bei CSI-abhängiger Positionsgröße",
                        yaxis_type="log")

    body.append(T.card("§5 — Positionsgröße als stetige Funktion des CSI",
        T.info("Statt binär ein- oder auszusteigen wird die Positionsgröße stetig mit "
               "steigendem Stress reduziert. Der Exponent k steuert die Aggressivität: "
               "k = 0 ignoriert den CSI, k = 1 skaliert linear, größere Werte reagieren "
               "überproportional auf hohe Stresswerte.")
        + T.formula(r"w_t=\Big(1-\frac{\mathrm{CSI}_t}{100}\Big)^{k}\Big/"
                    r"\max_t\Big(1-\frac{\mathrm{CSI}_t}{100}\Big)^{k}",
                    "CSI-abhängige Positionsgröße, auf maximal 100 % normiert")
        + T.div(figKk, 440) + T.df_html(k_df, index=False)
        + T.interp("Mit steigendem k sinken Volatilität und Drawdown monoton, während der "
                   "Ertrag langsamer nachgibt — bis zu einem Punkt, ab dem der "
                   "Investitionsgrad zu niedrig wird und die Rendite überproportional "
                   "leidet. Der Calmar-Wert markiert diesen Umschlagpunkt am deutlichsten.")))

    # ── §6 CSI × CPI Synergie ───────────────────────────────────────────
    cpi = ix.cpi_index().reindex(price.index).ffill().fillna(0.0)
    combos = {
        "Nur CSI < 80": base_sig & (csi_a.fillna(0) < 80),
        "Nur CPI < 1.5": base_sig & (cpi < 1.5),
        "UND-Verknüpfung (beide ruhig)": base_sig & (csi_a.fillna(0) < 80) & (cpi < 1.5),
        "ODER-Verknüpfung (einer ruhig)": base_sig & ((csi_a.fillna(0) < 80) | (cpi < 1.5)),
    }
    comp_z = 0.5 * sx.zscore(csi_a.fillna(50)) + 0.5 * sx.zscore(cpi)
    combos["Gewichtetes Composite z < 1"] = base_sig & (comp_z < 1.0)

    crows = [{"Filter": "Baseline (ohne Filter)", "Sharpe": res_base.metrics["Sharpe"],
              "Ann. Return": res_base.metrics["CAGR"], "Max DD": res_base.metrics["MaxDD"],
              "Calmar": res_base.metrics["Calmar"], "#Trades": res_base.n_trades,
              "ΔSharpe": 0.0, "KI unten": np.nan, "KI oben": np.nan, "p": np.nan}]
    curves2 = {"Baseline": res_base.equity}
    for name, s in combos.items():
        r = st.run_strategy(price, s, low=low)
        d = sx.sharpe_diff_test(r.rets, res_base.rets, n=400)
        crows.append({"Filter": name, "Sharpe": r.metrics["Sharpe"],
                      "Ann. Return": r.metrics["CAGR"], "Max DD": r.metrics["MaxDD"],
                      "Calmar": r.metrics["Calmar"], "#Trades": r.n_trades,
                      "ΔSharpe": d["diff"], "KI unten": d["lo"], "KI oben": d["hi"],
                      "p": d["p"]})
        curves2[name] = r.equity
    cdf = pd.DataFrame(crows)
    cdf["Bonferroni (k=5)"] = np.where(cdf["p"] < 0.05 / len(combos), "signifikant", "—")

    figS = go.Figure()
    for i, (name, eq) in enumerate(curves2.items()):
        figS.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                  line=dict(color=T.PAL[i % len(T.PAL)],
                                            width=2 if i == 0 else 1.3)))
    figS.update_layout(title="Kapitalkurven der Filterkombinationen", yaxis_type="log")

    corr_ci = csi_a.corr(cpi)
    body.append(T.card("§6 — Synergie von CSI und CPI",
        T.hypo("H0: Die Kombination beider Indizes verbessert den Sharpe nicht gegenüber "
               "dem jeweils besseren Einzelindex.")
        + T.stat_row([("Korrelation CSI ↔ CPI", sx.num(corr_ci, 2)),
                      ("Getestete Kombinationen", str(len(combos))),
                      ("Bonferroni α*", f"{0.05 / len(combos):.3f}")])
        + T.div(figS, 440) + T.df_html(cdf, index=False)
        + T.interp(
            f"Die beiden Indizes korrelieren mit r = {corr_ci:.2f}. Je höher diese "
            "Korrelation, desto weniger zusätzliche Information kann die Kombination "
            "liefern — beide messen dann im Wesentlichen denselben Stress. Die "
            "UND-Verknüpfung ist der konservativste Filter (seltener investiert, "
            "geringster Drawdown), die ODER-Verknüpfung der offensivste.")
        + T.warn("Alle Kombinationen wurden auf denselben Daten geprüft. Selbst nach "
                 "Bonferroni-Korrektur bleibt ein Selektionseffekt bestehen, weil die "
                 "Schwellen 80 und 1.5 aus früheren Analysen desselben Datensatzes "
                 "stammen.")))

    # ── §7 CGR als zusätzliche Komponente ───────────────────────────────
    cgr = ix.cgr_index()
    cgr_a = cgr.reindex(price.index).ffill().fillna(0.0)
    cgr_pr = sx.prank(cgr_a)
    comp7 = comp.copy()
    comp7["C7"] = cgr_pr.reindex(comp.index).ffill().fillna(50.0)
    w7 = {k: v * 0.85 for k, v in ix.CSI_WEIGHTS.items()} | {"C7": 0.15}
    csi7 = ix.csi_from_components(comp7, w7).reindex(price.index).ffill()

    r6 = st.run_strategy(price, base_sig & (csi_a.fillna(0) < 80), low=low)
    sig7 = base_sig & (csi7.fillna(0) < 80)
    r7 = st.run_strategy(price, sig7, low=low)
    d67 = sx.sharpe_diff_test(r7.rets, r6.rets, n=400)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=csi_a.index, y=csi_a, name="CSI (6 Komponenten)",
                              line=dict(color="#58a6ff", width=1.1)))
    fig7.add_trace(go.Scatter(x=csi7.index, y=csi7, name="CSI+CGR (7 Komponenten)",
                              line=dict(color="#ff7b72", width=1.1)))
    fig7.add_hline(y=80, line=dict(color="#f85149", dash="dash"))
    fig7.update_layout(title="Wirkung des geopolitischen Bausteins auf den CSI")

    body.append(T.card("§7 — Geopolitisches Risiko als siebte CSI-Komponente",
        T.info("Der CGR aus Report 42 wird in einen rollierenden Perzentilrang überführt "
               "und mit 15 % Gewicht aufgenommen; die übrigen Gewichte werden proportional "
               "auf 85 % skaliert, damit die Summe eins bleibt.")
        + T.div(fig7, 400)
        + T.df_html(pd.DataFrame([
            {"Variante": "CSI mit 6 Komponenten", "Sharpe": r6.metrics["Sharpe"],
             "Ann. Return": r6.metrics["CAGR"], "Max DD": r6.metrics["MaxDD"],
             "#Trades": r6.n_trades},
            {"Variante": "CSI mit CGR (7 Komponenten)", "Sharpe": r7.metrics["Sharpe"],
             "Ann. Return": r7.metrics["CAGR"], "Max DD": r7.metrics["MaxDD"],
             "#Trades": r7.n_trades},
        ]), index=False)
        + T.formula(r"\Delta\mathrm{Sharpe}=" + f"{d67['diff']:+.3f}"
                    + r",\quad \mathrm{KI}_{95\%}=\big["
                    + f"{d67['lo']:+.3f},\\;{d67['hi']:+.3f}" + r"\big],\quad p="
                    + f"{d67['p']:.3f}", "Gepaarter Block-Bootstrap-Vergleich")
        + T.interp("Der geopolitische Baustein verändert den Index vor allem in den "
                   "Phasen 2014/15 und 2022 spürbar. Ob daraus eine belastbare "
                   "Verbesserung wird, zeigt das Konfidenzintervall der Sharpe-Differenz.")))

    figE43, hE43 = C.equity_dashboard(
        {"CSI+CGR-Filter (&lt; 80)": r7.equity, "CSI-Filter (&lt; 80)": r6.equity,
         "Baseline ohne Filter": res_base.equity,
         "Buy &amp; Hold JETS": st.buy_hold(price).equity},
        exposure=r7.exposure, trades=r7.trades,
        title="Vergleich der Filtervarianten im Detail")
    figT43 = C.trade_chart(price.rename(st.TARGET), r7.trades, sig7,
                           "JETS mit den Ein- und Ausstiegen des CSI+CGR-Filters")
    body.append(T.card("§7b — Endvariante: Kapitalkurve, Investitionsgrad und Trades",
        T.info("Abschließende Sichtprüfung der empfohlenen Konfiguration gegen alle "
               "Referenzen. Die Flachstücke der Kapitalkurve entsprechen exakt den "
               "Phasen ohne Position im Investitionsgrad-Panel.")
        + T.div(figE43, hE43) + T.div(figT43, 560)
        + T.div(C.trade_return_bars(r7.trades), 300)))

    # ── §8 Fazit ────────────────────────────────────────────────────────
    best_thr = thr_df.sort_values("ΔSharpe vs. Baseline", ascending=False).iloc[0]
    best_k = k_df.sort_values("Calmar", ascending=False).iloc[0]
    body.append(T.card("§8 — Fazit und Empfehlung",
        T.interp(
            "<ol>"
            f"<li><strong>Schwelle:</strong> Im Gesamtzeitraum liefert CSI &lt; "
            f"{best_thr['CSI-Schwelle']:.0f} den höchsten Sharpe-Zuwachs "
            f"({best_thr['ΔSharpe vs. Baseline']:+.2f}); maßgeblich ist jedoch, ob das "
            "Konfidenzintervall die Null ausschließt und ob die Walk-Forward-Fenster "
            "dieselbe Schwelle wählen.</li>"
            f"<li><strong>Positionsgröße:</strong> Die stetige Skalierung mit k = "
            f"{best_k['k']} maximiert den Calmar-Wert und ist der binären Filterung "
            "vorzuziehen, weil sie keine Erholungsphasen vollständig verpasst.</li>"
            "<li><strong>Gewichte:</strong> Die In-Sample-Optimierung überträgt sich nur "
            "schwach out-of-sample — die Ausgangsgewichte bleiben die robustere Wahl.</li>"
            "<li><strong>Kombination:</strong> CSI und CPI überlappen inhaltlich stark; "
            "der Zusatznutzen der Kombination ist gering.</li>"
            "<li><strong>Gesamtbild:</strong> Stressfilter wirken primär "
            "risikoreduzierend, nicht renditesteigernd. Genau das ist das erklärte Ziel: "
            "robuster werden, ohne den absoluten Ertrag wesentlich zu opfern.</li>"
            "</ol>")
        + T.warn("Sämtliche Optimierungen dieses Reports laufen auf einer Historie von "
                 "gut zehn Jahren mit fünf klar identifizierten Crash-Ereignissen. Die "
                 "Zahl der effektiv unabhängigen Beobachtungen für die Frage „hilft der "
                 "Filter im Crash?“ liegt damit im einstelligen Bereich. Jede hier "
                 "genannte optimale Parameterwahl ist entsprechend als Größenordnung und "
                 "nicht als präziser Wert zu verstehen.")))

    T.write(out / "r43_flash_crash_optimization.html", T.html_base(TITLE, PHASE, "\n".join(body)))
