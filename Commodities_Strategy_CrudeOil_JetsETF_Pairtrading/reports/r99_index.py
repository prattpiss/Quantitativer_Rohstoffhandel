"""Report 99 — Dashboard-Index des Strategy Labs."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from core import stats_tools as sx
from core import strategy as st
from core import theme as T

PHASE = 24

CARDS = [
    {"file": "r00_baseline_reproduction.html", "nr": "00",
     "title": "Baseline-Reproduktion &amp; Diagnose", "colour": "#58a6ff",
     "desc": "Wörtliche Umsetzung der dokumentierten JETS-Strategie, Abgleich mit den "
             "Sollkennzahlen, Ursachenanalyse der Abweichung und Festlegung der "
             "validierten Arbeits-Baseline.",
     "tags": ["ADF/KPSS", "Block-Bootstrap", "Parameter-Sensitivität", "Lag-Korrelation"]},
    {"file": "r41_sector_rotation_deep.html", "nr": "41",
     "title": "Sektor-Rotation Deep Dive", "colour": "#7ee787",
     "desc": "Welche Sektoren führen JETS, wie verändert sich die Korrelationsstruktur in "
             "Krisen, und lohnt sich ein Rotations-Overlay?",
     "tags": ["Granger + Bonferroni/BH", "DCC-Approximation", "PCA (SVD)",
              "Regime-Korrelation", "Materials-Split"]},
    {"file": "r42_pandemic_war_monitor.html", "nr": "42",
     "title": "Pandemie- &amp; Kriegs-Frühwarnsystem", "colour": "#ffa657",
     "desc": "Marktbasierte Proxy-Indizes für Pandemie- und Konfliktrisiko, validiert an "
             "historischen Ereignissen, integriert als abgestufte Hedge-Schicht.",
     "tags": ["ROC / Precision-Recall", "Youden-Index", "CUSUM", "Extremwerttheorie",
              "Kosten-Nutzen"]},
    {"file": "r43_flash_crash_optimization.html", "nr": "43",
     "title": "Flash-Crash-Optimierung", "colour": "#ff7b72",
     "desc": "Walk-Forward-Optimierung der CSI-Schwelle und -Gewichte, stetige "
             "Positionsgrößensteuerung sowie Synergieprüfung von CSI, CPI und "
             "geopolitischem Risiko.",
     "tags": ["Walk-Forward", "Simplex-Zufallssuche", "Kaplan-Meier",
              "Positionsgrößen-Skalierung"]},
    {"file": "r44_structures_grid.html", "nr": "44",
     "title": "Strukturvarianten, Kombinationen &amp; Parameter-Grid", "colour": "#d2a8ff",
     "desc": "Korrelationsabhängigkeit, Long/Short, Pair und Spread, die 16 "
             "Signalkombinationen samt adaptiver Umschaltung sowie ein vollständiges "
             "Kreuzprodukt aus Stop-Logik, Sizing und Transaktionskosten — je getrennt "
             "für IS, OOS und IS+OOS.",
     "tags": ["IS / OOS / IS+OOS", "1 152 Simulationen", "Pair &amp; Spread",
              "Adaptive Umschaltung", "Kostenkurve", "Data-Snooping-Schranke"]},
]

METHOD_NOTES = """
<ul class="small">
  <li><strong>Stationarität zuerst:</strong> Vor jedem linearen Modell werden ADF und KPSS
      berichtet — mit Teststatistik, p-Wert, kritischem Wert und Schlussfolgerung.</li>
  <li><strong>Mehrfachtests:</strong> Jede Testfamilie wird Bonferroni- bzw. FDR-korrigiert;
      die korrigierte Schranke steht jeweils im Abschnitt.</li>
  <li><strong>Konfidenzintervalle:</strong> Performance-Kennzahlen erhalten stationäre
      Block-Bootstrap-Intervalle (N = 1000, Blocklänge 21 Tage); Vergleiche zweier
      Strategien nutzen einen gepaarten Bootstrap auf der Sharpe-Differenz.</li>
  <li><strong>Look-Ahead-Freiheit:</strong> Alle Standardisierungen und Perzentilränge sind
      rollierend; Signale wirken frühestens auf die Rendite des Folgetages.</li>
  <li><strong>Reproduzierbarkeit:</strong> Fester Zufallskeim (42), lokaler Datencache unter
      <code>data_cache/</code>, Neuaufbau über <code>python build_all.py</code>.</li>
</ul>
"""


def build(out: Path) -> None:
    print("Report 99 — Dashboard")
    res, sig = st.baseline()
    period = (f"{sig.index[0].date()} – {sig.index[-1].date()}" if len(sig) else "—")

    cards = []
    for c in CARDS:
        tags = "".join(
            f'<span class="badge rounded-pill me-1 mb-1" '
            f'style="background:{c["colour"]}22;color:{c["colour"]};'
            f'border:1px solid {c["colour"]}55;font-weight:500;">{t}</span>'
            for t in c["tags"])
        cards.append(f"""
<div class="col-md-6">
  <a href="{c['file']}" style="text-decoration:none;">
    <div class="card h-100" style="border-left:4px solid {c['colour']};">
      <div class="card-body">
        <div class="d-flex justify-content-between align-items-start mb-2">
          <h5 style="color:{c['colour']};margin:0;">{c['title']}</h5>
          <span class="badge" style="background:{c['colour']}33;color:{c['colour']};">
            Report {c['nr']}</span>
        </div>
        <p class="small" style="color:#8b949e;">{c['desc']}</p>
        <div>{tags}</div>
      </div>
    </div>
  </a>
</div>""")

    body = (
        T.header("Strategy Lab — Commodity &amp; Airline Research",
                 "Sektor-Rotation · Pandemie- und Kriegsrisiko · Flash-Crash-Optimierung "
                 f"&nbsp;|&nbsp; Stand: {date.today().isoformat()}")
        + T.stat_row([
            ("Untersuchungszeitraum", period),
            ("Handelstage", str(len(sig))),
            ("Baseline Sharpe", sx.num(res.metrics["Sharpe"], 2)),
            ("Ann. Return", sx.pct(res.metrics["CAGR"])),
            ("Max Drawdown", sx.pct(res.metrics["MaxDD"])),
            ("Trades", str(res.n_trades)),
        ])
        + T.info("Alle Reports bauen auf derselben validierten Baseline auf: Long JETS, "
                 "wenn das 20-Tage-Mittel des Energie-Basket-Returns negativ ist und der "
                 "VIX unter 25 liegt, abgesichert mit einem 8-%-Stop-Loss. Die Herleitung "
                 "und die Abweichung zur ursprünglichen Dokumentation stehen in Report 00.")
        + f'<div class="row g-4 mb-4">{"".join(cards)}</div>'
        + T.card("Methodische Leitplanken", METHOD_NOTES)
        + T.warn("Die gemeinsame Datengrundlage beginnt mit der JETS-Auflegung im April "
                 "2015. Alle Aussagen beruhen auf gut zehn Jahren Historie mit einer "
                 "einstelligen Zahl echter Krisenereignisse — Punktschätzer sind daher "
                 "durchgängig mit breiten Konfidenzintervallen zu lesen."))

    T.write(out / "index.html", T.html_base("Dashboard", PHASE, body))
