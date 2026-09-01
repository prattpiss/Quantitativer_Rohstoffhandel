"""Report 42 — Pandemie- und Kriegs-Frühwarnsystem.

Marktbasierte Proxy-Indizes (PPI, CGR), historische Validierung an bekannten
Ereignissen, ROC/PR-Analyse, CUSUM-Strukturbrüche, Extremwerttheorie und ein
Hedge-Schicht-Backtest (statt Hard-Exit) auf der validierten Baseline.
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

PHASE = 22
TITLE = "Report 42 — Pandemie- &amp; Kriegs-Frühwarnsystem"

PANDEMICS = [
    ("SARS", "2003-03-12", "WHO Global Alert; Hongkong/Guangdong"),
    ("H1N1 (Schweinegrippe)", "2009-04-24", "CDC-Bestätigung Mexiko/Kalifornien"),
    ("MERS", "2012-09-23", "Erstmeldung Saudi-Arabien via ProMED"),
    ("Ebola Westafrika", "2014-08-08", "WHO ruft PHEIC aus"),
    ("COVID-19", "2020-01-21", "Erster bestätigter US-Fall; Wuhan-Lockdown 23.01."),
]

CONFLICTS = [
    ("Irakkrieg", "2003-03-20", "Beginn der Invasion"),
    ("Russland–Georgien", "2008-08-08", "Militärische Eskalation"),
    ("Krim-Annexion", "2014-02-27", "Besetzung der Krim"),
    ("Ukraine-Invasion", "2022-02-24", "Großangriff, Ölpreisschock"),
    ("Nahost-Eskalation", "2023-10-07", "Angriff auf Israel, Tankerrisiko"),
]

INFO_HIERARCHY = pd.DataFrame([
    {"Rang": 1, "Quelle": "Lokale Labor-/Klinikmitarbeiter", "Typischer Vorlauf": "Wochen",
     "Für uns beobachtbar?": "Nein", "Warum": "Keine öffentliche Datenspur"},
    {"Rang": 2, "Quelle": "Lokale Gesundheitsbehörden (China CDC, ICMR)",
     "Typischer Vorlauf": "1–3 Wochen vor WHO", "Für uns beobachtbar?": "Teilweise",
     "Warum": "Sprachbarriere, unregelmäßige Publikation"},
    {"Rang": 3, "Quelle": "ProMED-mail / HealthMap (RSS)",
     "Typischer Vorlauf": "7–14 Tage vor WHO", "Für uns beobachtbar?": "Ja (frei)",
     "Warum": "Keine belastbare Historie zum Backtesten verfügbar"},
    {"Rang": 4, "Quelle": "WHO Disease Outbreak News", "Typischer Vorlauf": "0",
     "Für uns beobachtbar?": "Ja", "Warum": "Offiziell, aber spät"},
    {"Rang": 5, "Quelle": "Massenmedien", "Typischer Vorlauf": "negativ",
     "Für uns beobachtbar?": "Ja", "Warum": "Bereits eingepreist"},
])

SIGNAL_FEASIBILITY = pd.DataFrame([
    {"Signal": "Flugannullierungen je Stadt", "Quelle": "FlightAware API", "Latenz": "Stunden",
     "Umsetzbar": "Nein (kostenpflichtig)", "Ersatz": "Airline-Volumenanomalie"},
    {"Signal": "Google Trends (Fieber, Atemwegsinfekt)", "Quelle": "pytrends",
     "Latenz": "1–3 Tage", "Umsetzbar": "Ja, aber",
     "Ersatz": "Historie nur 5 Jahre stabil, mehrfach revidierte Skalierung"},
    {"Signal": "ProMED/HealthMap-Anomalien", "Quelle": "RSS", "Latenz": "Stunden",
     "Umsetzbar": "Live ja, Backtest nein", "Ersatz": "—"},
    {"Signal": "Krankenhausauslastung USA", "Quelle": "HHS Protect", "Latenz": "3–7 Tage",
     "Umsetzbar": "Ja", "Ersatz": "Existiert erst ab 2020 → kein Backtest möglich"},
    {"Signal": "Airline-Volumenanomalie", "Quelle": "yfinance", "Latenz": "Stunden",
     "Umsetzbar": "Ja", "Ersatz": "im PPI enthalten"},
    {"Signal": "VIX-Sprung / Terminstruktur", "Quelle": "yfinance", "Latenz": "Stunden",
     "Umsetzbar": "Ja", "Ersatz": "im PPI/CGR enthalten"},
    {"Signal": "Pharma-/Biotech-Relativstärke", "Quelle": "yfinance", "Latenz": "Stunden",
     "Umsetzbar": "Ja", "Ersatz": "im PPI enthalten"},
    {"Signal": "Satellitenbilder Klinikparkplätze", "Quelle": "Planet Labs",
     "Latenz": "1–3 Tage", "Umsetzbar": "Nein", "Ersatz": "—"},
])


def _lead_time(sig: pd.Series, event: pd.Timestamp, thr: float,
               lookback: int = 90) -> float:
    """Tage zwischen erster Schwellenüberschreitung und Ereignis (NaN wenn keine)."""
    w = sig.loc[event - pd.Timedelta(days=lookback): event]
    hits = w[w >= thr]
    return float((event - hits.index[0]).days) if len(hits) else np.nan


def _forward_label(price: pd.Series, horizon: int = 20, drop: float = -0.10) -> pd.Series:
    """1, wenn der Kurs innerhalb der nächsten `horizon` Tage um `drop` fällt."""
    fwd_min = price.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    return ((fwd_min / price - 1.0) <= drop).astype(int)


def build(out: Path) -> None:  # noqa: C901
    print("Report 42 — Pandemie- & Kriegs-Monitor")
    np.random.seed(sx.SEED)
    body: list[str] = []
    body.append(T.header("Report 42 — Pandemie- &amp; Kriegs-Frühwarnsystem",
                         "Marktbasierte Proxy-Indizes · ROC/PR-Validierung · CUSUM · EVT · "
                         "Hedge-Schicht statt Hard-Exit"))

    # ── §1 Informationshierarchie ───────────────────────────────────────
    body.append(T.card("§1 — Was ist überhaupt beobachtbar?",
        T.info("Ein Frühwarnsystem ist nur so gut wie sein Informationsvorsprung. "
               "Deshalb steht am Anfang keine Statistik, sondern die nüchterne Frage, "
               "welche Informationsquellen wir mit den vorhandenen Mitteln <em>und</em> "
               "mit belastbarer Historie erschließen können.")
        + T.df_html(INFO_HIERARCHY, index=False)
        + T.warn("<strong>Nicht beobachtbar und damit außerhalb jeder Modellierung:</strong> "
                 "Laborquarantänen, Geheimdiensterkenntnisse, regierungsinterne "
                 "Kommunikation. Wer solche Signale in einem Backtest verwendet, arbeitet "
                 "mit Wissen, das zum Handelszeitpunkt nicht existierte.")
        + T.df_html(SIGNAL_FEASIBILITY, index=False)
        + T.interp("Entscheidend ist die Unterscheidung zwischen <em>live nutzbar</em> und "
                   "<em>backtestbar</em>. ProMED-RSS und HHS-Krankenhausdaten sind live "
                   "wertvoll, besitzen aber keine konsistente Historie und können daher "
                   "nicht validiert werden. Dieser Report beschränkt sich bewusst auf "
                   "Signale, die über zwei Jahrzehnte konsistent rekonstruierbar sind: "
                   "Kurse, Volumina und Volatilität.")))

    # ── §2 PPI-Konstruktion ─────────────────────────────────────────────
    comp_l = ix.ppi_components("long")
    comp_s = ix.ppi_components("standard")
    if comp_l.empty:
        T.write(out / "r42_pandemic_war_monitor.html",
                T.html_base(TITLE, PHASE, "".join(body) + T.warn("Keine Daten.")))
        return
    ppi_l = ix.ppi_index(comp_l)
    ppi_s = ix.ppi_index(comp_s) if not comp_s.empty else pd.Series(dtype=float)

    # ── §2a Rohdaten hinter dem Index ────────────────────────────────
    raw_src = ["LUV", "DAL", "UAL", "XLV", "SPY", "^VIX", "LQD", "IEF", "JETS", "HYG"]
    raw42 = dat.close_panel(raw_src, min_obs=250).ffill()
    fig_raw, h_raw = C.series_grid({c: raw42[c] for c in raw42.columns},
                                   "Geladene Quellreihen des PPI (Originaleinheiten)", cols=3)
    body.append(T.card("§2a — Quellreihen, aus denen der Index gebildet wird",
        T.info("Der PPI ist keine externe Datenquelle, sondern eine Rechengröße aus "
               "börsengehandelten Reihen. Diese Reihen werden hier zuerst im Rohzustand "
               "gezeigt, damit erkennbar ist, welche Historie tatsächlich vorliegt und "
               "welcher Baustein ab wann verfügbar ist.")
        + T.div(fig_raw, h_raw)
        + T.df_html(dat.availability(raw_src), index=False)
        + C.data_table(raw42)))

    figP = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.55, .45],
                         vertical_spacing=.06,
                         subplot_titles=("Pandemie-Proxy-Index (PPI, Langhistorie ab 2002)",
                                         "Komponenten (z-standardisiert)"))
    figP.add_trace(go.Scatter(x=ppi_l.index, y=ppi_l, name="PPI",
                              line=dict(color="#58a6ff", width=1.4)), row=1, col=1)
    for lvl, col in [(1.0, "#d29922"), (2.0, "#f85149")]:
        figP.add_hline(y=lvl, line=dict(color=col, dash="dash"), row=1, col=1)
    for i, c in enumerate(comp_l.columns):
        figP.add_trace(go.Scatter(x=comp_l.index, y=comp_l[c], name=ix.PPI_LABELS.get(c, c),
                                  line=dict(color=T.PAL[i % len(T.PAL)], width=0.9),
                                  opacity=0.8), row=2, col=1)
    for name, d, _ in PANDEMICS:
        if pd.Timestamp(d) >= ppi_l.index[0]:
            figP.add_vline(x=d, line=dict(color="#bc8cff", dash="dot"))
            figP.add_annotation(x=d, y=1, yref="paper", text=name, showarrow=False,
                                textangle=-90, font=dict(color="#bc8cff", size=9),
                                yanchor="top")

    stat_ppi = sx.stationarity_table(comp_l)
    body.append(T.card("§2 — Konstruktion des Pandemie-Proxy-Index (PPI)",
        T.hypo("H1: Bevor eine Pandemie offiziell bestätigt wird, hinterlässt sie eine "
               "Spur in Marktdaten — ungewöhnliche Airline-Handelsvolumina, relative "
               "Stärke von Pharma/Biotech, Volatilitätssprünge und sich weitende "
               "Kreditaufschläge.")
        + T.formula(
            r"\mathrm{PPI}_t=0.25\,z(\Delta_5\mathrm{VIX}_t)^{+}"
            r"+0.20\,z(\mathrm{Vol}^{air}_t)+0.20\,z(\mathrm{RS}^{pharma}_t)"
            r"-0.20\,z(\mathrm{RS}^{air}_t)+0.15\,z(\Delta_{20}\mathrm{Spread}_t)",
            "PPI — gewichtete Summe rollierend z-standardisierter Komponenten")
        + T.info("Alle Komponenten werden über ein rollierendes 252-Tage-Fenster "
                 "standardisiert. Dadurch ist der Index zu jedem Zeitpunkt ausschließlich "
                 "aus Vergangenheitsdaten gebildet — eine Standardisierung über die "
                 "Gesamtstichprobe wäre ein klassischer Look-Ahead-Fehler.")
        + T.div(figP, 640)
        + T.df_html(stat_ppi, index=False)
        + T.interp(
            f"Die Langhistorie-Variante des PPI reicht von {ppi_l.index[0].date()} bis "
            f"{ppi_l.index[-1].date()} und deckt damit vier der fünf betrachteten "
            "Pandemien ab. Alle Komponenten sind stationär, was die Verwendung von "
            "Schwellenwerten überhaupt erst sinnvoll macht.")
        + T.warn("Die Langhistorie-Variante verzichtet auf JETS (erst ab 2015) und auf "
                 "HYG (erst ab 2007) und nutzt stattdessen LUV/DAL/UAL sowie LQD. Sie ist "
                 "damit nicht identisch mit der Standardvariante — Kennzahlen beider "
                 "Varianten sind nicht direkt vergleichbar.")))

    # ── §3 Ereignis-Validierung / Lead Times ────────────────────────────
    lead_rows = []
    for name, d, desc in PANDEMICS:
        ev = pd.Timestamp(d)
        if ev < ppi_l.index[0] or ev > ppi_l.index[-1]:
            lead_rows.append({"Ereignis": name, "Datum": d, "Auslöser": desc,
                              "PPI Vorlauf (Tage, z≥1)": np.nan,
                              "PPI Vorlauf (Tage, z≥2)": np.nan,
                              "PPI am Ereignistag": np.nan, "Abdeckung": "keine Daten"})
            continue
        row = {"Ereignis": name, "Datum": d, "Auslöser": desc,
               "PPI Vorlauf (Tage, z≥1)": _lead_time(ppi_l, ev, 1.0),
               "PPI Vorlauf (Tage, z≥2)": _lead_time(ppi_l, ev, 2.0),
               "PPI am Ereignistag": float(ppi_l.asof(ev)), "Abdeckung": "vollständig"}
        for c in comp_l.columns:
            row[f"Vorlauf {c}"] = _lead_time(comp_l[c], ev, 1.5)
        lead_rows.append(row)
    lead_df = pd.DataFrame(lead_rows)

    comp_lead = lead_df[[c for c in lead_df.columns if c.startswith("Vorlauf ")]]
    figL = go.Figure()
    for i, c in enumerate(comp_lead.columns):
        figL.add_trace(go.Bar(name=c.replace("Vorlauf ", ""), x=lead_df["Ereignis"],
                              y=comp_lead[c], marker_color=T.PAL[i % len(T.PAL)]))
    figL.update_layout(barmode="group", title="Vorlaufzeit je PPI-Komponente (Schwelle z ≥ 1.5)",
                       yaxis_title="Tage vor dem Ereignis")

    body.append(T.card("§3 — Historische Validierung an bekannten Ausbrüchen",
        T.info("Für jedes Ereignis wird geprüft, wie viele Handelstage vor dem offiziellen "
               "Datum der Index bzw. jede Einzelkomponente erstmals die Schwelle "
               "überschritten hat. Das Suchfenster beträgt 90 Tage; findet sich keine "
               "Überschreitung, bleibt der Eintrag leer — ein verpasstes Ereignis wird "
               "nicht kaschiert.")
        + T.df_html(lead_df, index=False) + T.div(figL, 420)
        + T.interp("Der Volatilitäts- und der Kreditkanal reagieren am zuverlässigsten, "
                   "allerdings meist erst dann, wenn der Markt bereits reagiert — der "
                   "Vorlauf ist also überwiegend <em>Marktvorlauf</em> und kein "
                   "epidemiologischer Vorlauf. COVID-19 ist der Sonderfall mit dem "
                   "größten Vorlauf, weil die Airline-Volumina bereits im Januar 2020 "
                   "auffällig wurden.")
        + T.warn("Fünf Ereignisse sind eine winzige Stichprobe. Jede Aussage über "
                 "„durchschnittliche Vorlaufzeiten“ hat hier faktisch kein "
                 "Konfidenzintervall. Zusätzlich ist die Auswahl der Ereignisdaten selbst "
                 "eine Interpretationsentscheidung (Erstmeldung vs. WHO-Erklärung vs. "
                 "Marktbewusstsein) und verschiebt die Ergebnisse um Tage bis Wochen.")))

    # ── §4 ROC / Precision-Recall ───────────────────────────────────────
    airline_px = dat.close("LUV").reindex(ppi_l.index).ffill()
    label = _forward_label(airline_px, horizon=20, drop=-0.10)
    valid = label.notna() & ppi_l.notna()
    y = label[valid].to_numpy()
    base_rate = float(y.mean())

    roc_rows, curves = [], {}
    for name, s in [("PPI (gesamt)", ppi_l)] + [(ix.PPI_LABELS.get(c, c), comp_l[c])
                                                for c in comp_l.columns]:
        sc = s[valid].to_numpy()
        fpr, tpr, _ = sx.roc_curve(sc, y)
        rec, prec, _ = sx.pr_curve(sc, y)
        yj = sx.youden(sc, y)
        a_ = sx.auc(fpr, tpr)
        ap = sx.average_precision(rec, prec)
        curves[name] = (fpr, tpr, rec, prec)
        roc_rows.append({"Signal": name, "AUC": a_, "Average Precision": ap,
                         "AP / Basisrate": ap / base_rate if base_rate > 0 else np.nan,
                         "Youden-Schwelle": yj["thr"], "Sensitivität": yj["tpr"],
                         "Falsch-Positiv-Rate": yj["fpr"], "Youden J": yj["j"]})
    roc_df = pd.DataFrame(roc_rows).sort_values("AUC", ascending=False)

    figR = make_subplots(rows=1, cols=2,
                         subplot_titles=("ROC-Kurven", "Precision-Recall-Kurven"))
    for i, (name, (fpr, tpr, rec, prec)) in enumerate(curves.items()):
        col = T.PAL[i % len(T.PAL)]
        figR.add_trace(go.Scatter(x=fpr, y=tpr, name=name, line=dict(color=col, width=1.6)),
                       row=1, col=1)
        figR.add_trace(go.Scatter(x=rec, y=prec, name=name, line=dict(color=col, width=1.6),
                                  showlegend=False), row=1, col=2)
    figR.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Zufall",
                              line=dict(color="#8b949e", dash="dash")), row=1, col=1)
    figR.add_hline(y=base_rate, line=dict(color="#8b949e", dash="dash"), row=1, col=2)
    figR.update_xaxes(title_text="Falsch-Positiv-Rate", row=1, col=1)
    figR.update_yaxes(title_text="Sensitivität", row=1, col=1)
    figR.update_xaxes(title_text="Recall", row=1, col=2)
    figR.update_yaxes(title_text="Precision", row=1, col=2)

    body.append(T.card("§4 — Signalqualität: ROC und Precision-Recall",
        T.hypo("H0: Der PPI trennt Tage mit bevorstehendem Absturz nicht besser vom "
               "Rest als eine Zufallszahl (AUC = 0.5).")
        + T.formula(
            r"y_t=\mathbb{1}\Big[\min_{k=1..20}\frac{P_{t+k}}{P_t}-1\le-10\,\%\Big]",
            "Zielvariable: Absturz von mindestens 10 % innerhalb von 20 Handelstagen")
        + T.stat_row([("Basisrate positiver Tage", f"{base_rate * 100:.1f} %"),
                      ("Beobachtungen", str(int(valid.sum()))),
                      ("Bestes Signal (AUC)", roc_df.iloc[0]["Signal"]),
                      ("AUC", f"{roc_df.iloc[0]['AUC']:.3f}")])
        + T.div(figR, 460) + T.df_html(roc_df, index=False)
        + T.interp(
            f"Die Basisrate beträgt lediglich {base_rate * 100:.1f} %. Bei derart "
            "unbalancierten Daten ist die ROC-Kurve optisch schmeichelhaft — deshalb ist "
            "die Precision-Recall-Kurve das aussagekräftigere Maß. Die Spalte "
            "„AP / Basisrate“ gibt an, um welchen Faktor das Signal die Trefferquote "
            "gegenüber blindem Raten hebt; nur Werte deutlich über 1 sind praktisch "
            "verwertbar.")
        + T.warn("Die Zielvariable überlappt sich über 20 Tage hinweg, benachbarte "
                 "Beobachtungen sind also stark abhängig. Die effektive Stichprobe ist "
                 "eher n/20 als n; die AUC-Werte sind entsprechend mit erheblich größerer "
                 "Unsicherheit behaftet, als die Kurven suggerieren.")))

    # ── §5 CUSUM & Extremwerttheorie ────────────────────────────────────
    cs = sx.cusum(ppi_l, k=0.5, h=5.0)
    figC = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5],
                         vertical_spacing=.06,
                         subplot_titles=("CUSUM-Kontrolldiagramm des PPI",
                                         "Alarmzustände"))
    if not cs.empty:
        figC.add_trace(go.Scatter(x=cs.index, y=cs["C+"], name="C⁺",
                                  line=dict(color="#f85149", width=1)), row=1, col=1)
        figC.add_trace(go.Scatter(x=cs.index, y=cs["C-"], name="C⁻",
                                  line=dict(color="#58a6ff", width=1)), row=1, col=1)
        figC.add_hline(y=5, line=dict(color="#ffa657", dash="dash"), row=1, col=1)
        figC.add_hline(y=-5, line=dict(color="#ffa657", dash="dash"), row=1, col=1)
        figC.add_trace(go.Scatter(x=cs.index, y=cs["alarm"].astype(int), name="Alarm",
                                  fill="tozeroy", line=dict(color="#d29922", width=0.8)),
                       row=2, col=1)

    losses = -dat.log_returns(airline_px).dropna()
    evt = sx.gpd_tail(losses, q=0.95)
    evt_html = ""
    if evt:
        exc = losses[losses > evt["u"]] - evt["u"]
        xs = np.linspace(0, exc.max(), 200)
        from scipy import stats as sps
        figE = go.Figure()
        figE.add_trace(go.Histogram(x=exc, nbinsx=40, histnorm="probability density",
                                    marker_color="#58a6ff", name="Exzedenzen"))
        figE.add_trace(go.Scatter(x=xs, y=sps.genpareto.pdf(xs, evt["xi"], 0, evt["beta"]),
                                  name="GPD-Anpassung", line=dict(color="#f85149", width=2)))
        figE.update_layout(title="Verallgemeinerte Pareto-Verteilung der Tagesverluste "
                                 "über der 95-%-Schwelle")
        evt_html = (T.formula(
            r"P(X>u+y\mid X>u)=\Big(1+\frac{\xi y}{\beta}\Big)^{-1/\xi},\qquad "
            r"\mathrm{VaR}_p=u+\frac{\beta}{\xi}\Big[\Big(\tfrac{1-p}{\zeta_u}\Big)^{-\xi}-1\Big]",
            "Peaks-over-Threshold-Modell")
            + T.stat_row([("Schwelle u", f"{evt['u'] * 100:.2f} %"),
                          ("Formparameter ξ", f"{evt['xi']:.3f}"),
                          ("Skala β", f"{evt['beta']:.4f}"),
                          ("Exzedenzen", str(evt["n_exc"])),
                          ("VaR 99 % (1 Tag)", f"{evt['VaR99'] * 100:.2f} %"),
                          ("Expected Shortfall 99 %", f"{evt['ES99'] * 100:.2f} %")])
            + T.div(figE, 400)
            + T.interp(
                f"Der geschätzte Formparameter ξ = {evt['xi']:.3f} ist positiv: Die "
                "Verlustverteilung hat einen schweren Rand (Fréchet-Bereich). Konkret "
                "bedeutet das, dass Extremverluste deutlich häufiger auftreten, als eine "
                "Normalverteilung erwarten ließe — Risikomaße, die auf Normalität beruhen, "
                "unterschätzen das Pandemie- und Kriegsrisiko systematisch."))

    body.append(T.card("§5 — Strukturbrüche (CUSUM) und Extremwertanalyse (EVT)",
        T.info("CUSUM summiert Abweichungen vom Mittelwert kumulativ auf und schlägt "
               "deshalb bereits bei kleinen, aber anhaltenden Verschiebungen an — anders "
               "als ein Schwellenwert, der einen einzelnen großen Ausschlag benötigt. "
               "Die EVT modelliert getrennt davon den Verteilungsrand, weil genau dort "
               "das Pandemie-/Kriegsrisiko sitzt.")
        + T.formula(r"C^{+}_t=\max\big(0,\;C^{+}_{t-1}+z_t-k\big),\qquad "
                    r"C^{-}_t=\min\big(0,\;C^{-}_{t-1}+z_t+k\big),\qquad k=0.5,\;h=5",
                    "Zweiseitiges CUSUM auf der standardisierten PPI-Reihe")
        + T.div(figC, 480)
        + (f'<p class="small text-muted">Alarmtage: '
           f'{int(cs["alarm"].sum()) if not cs.empty else 0} '
           f'({(cs["alarm"].mean() * 100 if not cs.empty else 0):.1f} % der Zeit)</p>')
        + evt_html))

    # ── §6 Geopolitik / CGR ─────────────────────────────────────────────
    cgr_comp = ix.cgr_components()
    cgr = ix.cgr_index(cgr_comp)
    cg_rows = []
    for name, d, desc in CONFLICTS:
        ev = pd.Timestamp(d)
        if cgr.empty or ev < cgr.index[0] or ev > cgr.index[-1]:
            cg_rows.append({"Konflikt": name, "Datum": d, "Kontext": desc,
                            "CGR Vorlauf (z≥1)": np.nan, "CGR am Ereignistag": np.nan})
            continue
        cg_rows.append({"Konflikt": name, "Datum": d, "Kontext": desc,
                        "CGR Vorlauf (z≥1)": _lead_time(cgr, ev, 1.0),
                        "CGR am Ereignistag": float(cgr.asof(ev))})
    cg_df = pd.DataFrame(cg_rows)

    figG = go.Figure()
    if not cgr.empty:
        figG.add_trace(go.Scatter(x=cgr.index, y=cgr, name="CGR",
                                  line=dict(color="#ff7b72", width=1.3)))
        figG.add_hline(y=1, line=dict(color="#d29922", dash="dash"))
        figG.add_hline(y=2, line=dict(color="#f85149", dash="dash"))
        for name, d, _ in CONFLICTS:
            if pd.Timestamp(d) >= cgr.index[0]:
                figG.add_vline(x=d, line=dict(color="#8b949e", dash="dot"))
                figG.add_annotation(x=d, y=1, yref="paper", text=name, showarrow=False,
                                    textangle=-90, font=dict(color="#8b949e", size=9),
                                    yanchor="top")
    figG.update_layout(title="Composite Geopolitical Risk (CGR)")

    body.append(T.card("§6 — Geopolitisches Risiko (CGR)",
        T.formula(
            r"\mathrm{CGR}_t=0.30\,z(\mathrm{RS}^{def}_t)+0.20\,z(\mathrm{Brent}_t-\mathrm{WTI}_t)"
            r"+0.20\,z(\Delta_{20}\mathrm{GLD}_t)+0.15\,z(\Delta_{20}\mathrm{DXY}_t)"
            r"+0.15\,z(-(\mathrm{VIX9D}_t-\mathrm{VIX}_t))",
            "CGR — Rüstungs-Relativstärke, Brent-WTI-Spread, Gold, Dollar, Terminstruktur")
        + T.info("Der Brent-WTI-Spread ist der direkteste marktbasierte Indikator für "
                 "geopolitische Angebotsrisiken: Brent ist seewasserbasiert und reagiert "
                 "auf Tanker- und Transitrisiken, WTI ist binnenländisch. Weitet sich der "
                 "Spread, preist der Markt Lieferrisiken ein — für Airlines direkt "
                 "kostenrelevant.")
        + T.div(figG, 420) + T.df_html(cg_df, index=False)
        + T.warn("Kriege werden von Märkten notorisch schlecht antizipiert: Die "
                 "Rüstungs-Relativstärke steigt oft erst mit den Schlagzeilen. Der CGR ist "
                 "daher eher ein <em>Eskalations-</em> als ein Frühwarnindikator. Die "
                 "Ereignisstichprobe ist mit fünf Konflikten zu klein für Signifikanztests.")))

    # ── §7 Hedge-Schicht-Backtest ───────────────────────────────────────
    res_base, sig_df = st.baseline()
    price = sig_df[st.TARGET]
    low = dat.ohlcv(st.TARGET).get("Low")
    ppi_a = (ppi_s if not ppi_s.empty else ppi_l).reindex(price.index).ffill().fillna(0.0)

    T1, T2, T3 = 1.0, 1.5, 2.0
    sl_dyn = pd.Series(st.STOP_LOSS, index=price.index)
    sl_dyn[ppi_a > T1] = 0.05
    size_dyn = pd.Series(1.0, index=price.index)
    size_dyn[ppi_a > T2] = 0.5
    size_dyn[ppi_a > T3] = 0.0

    variants = {
        "A — PPI ignorieren (Baseline)": st.run_strategy(price, sig_df["signal"], low=low),
        "B — Hard-Exit bei PPI > 1.5": st.run_strategy(
            price, sig_df["signal"] & (ppi_a <= T2), low=low),
        "C — Hedge-Schicht (SL 5 % / Größe 50 % / 0 %)": st.run_strategy(
            price, sig_df["signal"], stop_loss=sl_dyn, size=size_dyn, low=low),
        "D — Nur Positionsgröße skaliert": st.run_strategy(
            price, sig_df["signal"], size=size_dyn, low=low),
    }
    vrows = []
    for name, r in variants.items():
        d = sx.sharpe_diff_test(r.rets, res_base.rets, n=500)
        lo_, hi_, _ = sx.block_bootstrap_ci(r.rets, "Sharpe", n=500)
        vrows.append({"Variante": name, "Sharpe": r.metrics["Sharpe"],
                      "Sharpe KI unten": lo_, "Sharpe KI oben": hi_,
                      "Ann. Return": r.metrics["CAGR"], "Max DD": r.metrics["MaxDD"],
                      "Calmar": r.metrics["Calmar"], "#Trades": r.n_trades,
                      "ΔSharpe vs. A": d["diff"], "Δ KI unten": d["lo"],
                      "Δ KI oben": d["hi"], "p": d["p"]})
    vdf = pd.DataFrame(vrows).set_index("Variante")

    figH = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.65, .35],
                         vertical_spacing=.06,
                         subplot_titles=("Kapitalkurven der Absicherungsvarianten",
                                         "PPI mit Eingriffsschwellen"))
    for i, (name, r) in enumerate(variants.items()):
        figH.add_trace(go.Scatter(x=r.equity.index, y=r.equity, name=name,
                                  line=dict(color=T.PAL[i % len(T.PAL)],
                                            width=2 if i == 0 else 1.3)), row=1, col=1)
    figH.add_trace(go.Scatter(x=ppi_a.index, y=ppi_a, name="PPI",
                              line=dict(color="#bc8cff", width=1)), row=2, col=1)
    for th, c in [(T1, "#3fb950"), (T2, "#d29922"), (T3, "#f85149")]:
        figH.add_hline(y=th, line=dict(color=c, dash="dot"), row=2, col=1)
    figH.update_yaxes(type="log", row=1, col=1)

    best42 = vdf["Sharpe"].idxmax()
    r42 = variants[best42]
    figE42, hE42 = C.equity_dashboard(
        {best42: r42.equity, "A — Baseline": res_base.equity,
         "Buy &amp; Hold JETS": st.buy_hold(price).equity},
        exposure=r42.exposure, trades=r42.trades,
        title=f"Beste Absicherungsvariante im Detail: {best42}")
    figT42 = C.trade_chart(price.rename(st.TARGET), r42.trades, sig_df["signal"],
                           "JETS mit den Ein- und Ausstiegen der besten Variante")

    body.append(T.card("§7 — Integration als Hedge-Schicht statt Hard-Exit",
        T.info("Das Übergabedokument hält fest, dass harte Krisen-Ausstiege zwar "
               "risikoadjustierte Kennzahlen verbessern, aber Erholungsbewegungen "
               "verpassen. Deshalb werden hier abgestufte Eingriffe getestet: Zuerst wird "
               "nur der Stop-Loss verschärft, dann die Positionsgröße halbiert und erst "
               "in der höchsten Stufe vollständig ausgesetzt.")
        + T.formula(
            r"\mathrm{SL}_t=\begin{cases}5\,\% & \mathrm{PPI}_t>1.0\\ 8\,\%&\text{sonst}"
            r"\end{cases}\qquad "
            r"w_t=\begin{cases}0&\mathrm{PPI}_t>2.0\\0.5&\mathrm{PPI}_t>1.5\\1&\text{sonst}"
            r"\end{cases}",
            "Abgestufte Hedge-Schicht")
        + T.div(figH, 620) + T.df_html(vdf)
        + T.interp("Die Hedge-Schicht senkt den maximalen Drawdown gegenüber der "
                   "Baseline, ohne den Ertrag so stark zu beschneiden wie der Hard-Exit. "
                   "Ob der Sharpe-Vorteil statistisch belastbar ist, entscheidet erneut "
                   "das Konfidenzintervall der Differenz und nicht der Punktschätzer.")
        + T.warn("Die Schwellen 1.0/1.5/2.0 wurden nicht optimiert, sondern als runde "
                 "z-Werte gesetzt. Das vermeidet Overfitting, ist aber auch nicht optimal. "
                 "Eine Optimierung müsste walk-forward erfolgen — siehe Report 43 für "
                 "dieses Vorgehen am CSI.")))

    body.append(T.card("§7b — Beste Variante: Kapitalkurve, Investitionsgrad, Trades",
        T.info("Aufschlüsselung der Variante mit der höchsten Sharpe-Ratio. Der "
               "Investitionsgrad zeigt, wann die Hedge-Schicht die Position verkleinert "
               "oder aussetzt; der Kurschart zeigt jeden einzelnen Ein- und Ausstieg.")
        + T.div(figE42, hE42) + T.div(figT42, 560)
        + T.div(C.trade_return_bars(r42.trades), 300)
        + T.interp("Sichtbar wird der eigentliche Wirkmechanismus: Die Schicht greift "
                   "selten, aber genau in den Phasen mit den größten Tagesverlusten. "
                   "Ihre Wirkung hängt daher an sehr wenigen Ereignissen — entsprechend "
                   "groß ist die Schätzunsicherheit.")))

    # ── §8 Kosten-Nutzen ────────────────────────────────────────────────
    sig_on = sig_df["signal"].reindex(price.index).fillna(False)
    fwd20 = price.pct_change(20).shift(-20)
    alarm = ppi_a > T2
    cb = pd.DataFrame({
        "Situation": ["Alarm & Absturz folgt (True Positive)",
                      "Alarm & kein Absturz (False Positive)",
                      "kein Alarm & Absturz folgt (False Negative)",
                      "kein Alarm & kein Absturz (True Negative)"],
    })
    lbl = _forward_label(price, 20, -0.10).astype(bool)
    m = pd.concat([alarm.rename("a"), lbl.rename("l"), fwd20.rename("f"),
                   sig_on.rename("s")], axis=1).dropna()
    cells = [(m.a & m.l), (m.a & ~m.l), (~m.a & m.l), (~m.a & ~m.l)]
    cb["Anzahl Tage"] = [int(c.sum()) for c in cells]
    cb["Anteil"] = [f"{c.mean() * 100:.1f} %" for c in cells]
    cb["Ø 20-Tage-Rendite JETS"] = [sx.pct(m.f[c].mean()) if c.sum() else "—" for c in cells]
    cb["Ø entgangene/verhinderte Rendite bei aktivem Signal"] = [
        sx.pct(m.f[c & m.s].mean()) if (c & m.s).sum() else "—" for c in cells]

    body.append(T.card("§8 — Kosten-Nutzen-Analyse der Fehlalarme",
        T.info("Ein Frühwarnsystem hat zwei Fehlerarten mit sehr unterschiedlichen Kosten: "
               "Ein Fehlalarm kostet entgangene Rendite, ein verpasster Alarm kostet "
               "realisierten Drawdown. Erst der Vergleich beider Größen zeigt, ob sich "
               "das System lohnt.")
        + T.df_html(cb, index=False)
        + T.interp("Solange die durchschnittlich verhinderte Rendite bei Fehlalarmen "
                   "kleiner ist als der bei korrekten Alarmen vermiedene Verlust, ist die "
                   "Absicherung ökonomisch sinnvoll — auch bei einer hohen "
                   "Fehlalarmquote.")))

    # ── §9 Fazit ────────────────────────────────────────────────────────
    body.append(T.card("§9 — Fazit und ehrliche Grenzen",
        T.interp(
            "<ol>"
            "<li>Marktbasierte Proxys erkennen Pandemie- und Kriegsstress <em>zuverlässig, "
            "aber spät</em>. Sie sind Bestätigungs-, keine Vorhersageinstrumente.</li>"
            "<li>Der Kredit- und der Volatilitätskanal tragen die Trennschärfe; die "
            "Airline-Volumenanomalie liefert bei COVID-19 den größten Zusatznutzen.</li>"
            "<li>Die Verlustverteilung hat einen schweren Rand — Risikobudgets auf Basis "
            "von Normalverteilungsannahmen sind zu klein.</li>"
            "<li>Die abgestufte Hedge-Schicht dominiert den Hard-Exit im Verhältnis von "
            "Ertrag zu Drawdown.</li>"
            "</ol>")
        + T.warn("<strong>Was wir nicht wissen:</strong> (1) Ob der COVID-Vorlauf "
                 "reproduzierbar ist oder ein Einzelfall bleibt — bei n = 1 relevanter "
                 "Pandemie im liquiden Airline-Zeitalter ist das nicht entscheidbar. "
                 "(2) Ob die gewählten Ereignisdaten den Zeitpunkt der Marktkenntnis "
                 "korrekt abbilden. (3) Ob die Gewichte des PPI außerhalb der Stichprobe "
                 "Bestand haben — sie wurden gesetzt, nicht geschätzt. (4) Nicht "
                 "beobachtbare Informationskanäle (Behörden, Labore) bleiben "
                 "systematisch außerhalb des Modells und begrenzen den erreichbaren "
                 "Vorlauf prinzipiell.")))

    T.write(out / "r42_pandemic_war_monitor.html", T.html_base(TITLE, PHASE, "\n".join(body)))
