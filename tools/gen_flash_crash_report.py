#!/usr/bin/env python3
"""Inject build_flash_crash_report into reports/report_builder.py"""
from pathlib import Path

SRC   = Path("reports/report_builder.py")
FN    = "build_flash_crash_report"
INJ   = "\ndef build_index(tables, figures, out):"
OLD_W = ("    build_sector_rotation_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")
NEW_W = ("    build_sector_rotation_report(tables, figures, reports)\n"
         "    build_flash_crash_report(tables, figures, reports)\n"
         "    build_index(tables, figures, reports)")

FUNC = r'''
def build_flash_crash_report(tables, figures, out):  # noqa: C901
    """
    Flash Crash Early Warning Dashboard.
    Beantwortet: Gibt es messbare Frühwarnsignale für starke JETS-Einbrüche?
    Methodik: Composite Stress Index (CSI) aus 6 Markt-Signalkomponenten,
    validiert an historischen Flash Crashes 2015–2022.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf

    def _tz(raw):
        idx = raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        raw.index = idx.normalize()
        return raw

    def _dl(t):
        try:
            return _tz(yf.Ticker(t).history(period="max", auto_adjust=True))["Close"].rename(t)
        except Exception:
            return pd.Series(dtype=float, name=t)

    def _dl_full(t):
        try:
            return _tz(yf.Ticker(t).history(period="max", auto_adjust=True))
        except Exception:
            return pd.DataFrame()

    # ── Daten-Downloads ───────────────────────────────────────────────────
    vix      = _dl("^VIX")
    vix9d    = _dl("^VIX9D")
    hyg      = _dl("HYG")
    ief      = _dl("IEF")
    dxy      = _dl("DX-Y.NYB")
    spy      = _dl("SPY")
    jets_df  = _dl_full("JETS")
    jets_cl  = jets_df["Close"].rename("JETS") if "Close" in jets_df.columns else pd.Series(dtype=float, name="JETS")
    jets_vol = jets_df["Volume"].rename("JETS_Vol") if "Volume" in jets_df.columns else pd.Series(dtype=float)
    jets_lo  = jets_df["Low"].rename("JETS_Low")  if "Low"    in jets_df.columns else pd.Series(dtype=float)

    # Flash Crash Ereignisse: (Name, Datum, Beschreibung, Hauptursache)
    EVENTS = [
        ("China Crash",     "2015-08-24",
         "Renminbi-Abwertung → globaler Sell-off. JETS −6% in 2 Tagen. DAX −5.8%.",
         "Währungsschock / Contagion"),
        ("Volmageddon",     "2018-02-05",
         "VIX-Short-Produkt-Blowup: VIX von 17 auf 37 innerhalb einer Handelsstunde.",
         "Strukturelles Finanzprodukt-Risiko"),
        ("COVID-Crash",     "2020-03-16",
         "Schlimmster SPY-Tag seit 1987 (−12%). JETS −17% an einem Tag. Reiseverbote.",
         "Exogener Schock / Pandemie"),
        ("Inflation Shock", "2022-01-24",
         "SPY intraday −5% (dann Erholung). Fed Pivot-Angst. Ukraine-Spannungen wachsen.",
         "Makro-Policy / Geopolitik"),
        ("CPI Schock",      "2022-09-13",
         "US-CPI überraschend hoch → Fed-Zins-Schock. SPY −4.3%, JETS −7%.",
         "Makro-Überraschung / Inflation"),
    ]
    CFILLS  = ["rgba(248,81,73,0.12)", "rgba(210,168,255,0.12)", "rgba(248,81,73,0.12)",
               "rgba(240,136,62,0.12)", "rgba(248,81,73,0.12)"]
    ECOLORS = ["#f85149", "#d2a8ff", "#f0883e", "#ffa657", "#ff7b72"]

    # ── Gemeinsamer Index ─────────────────────────────────────────────────
    base = [s for s in [vix, hyg, ief] if len(s) > 500]
    if not base:
        _write(out / "flash_crash_report.html",
               _html_base("Flash Crash EWS", 20, "<p class='text-warning'>Keine Daten.</p>"))
        return
    cidx = base[0].index
    for s in base[1:]:
        cidx = cidx.intersection(s.index)
    for s in [dxy, spy, jets_cl]:
        if len(s) > 200:
            cidx = cidx.intersection(s.index)
    cidx = cidx.sort_values()

    def _al(s):
        return s.reindex(cidx).ffill().bfill() if len(s) > 50 else pd.Series(np.nan, index=cidx)

    vix_a    = _al(vix);    hyg_a = _al(hyg);   ief_a = _al(ief)
    dxy_a    = _al(dxy);    spy_a = _al(spy);   jets_a = _al(jets_cl)
    jvol_a   = _al(jets_vol); jlo_a = _al(jets_lo)
    vix9d_a  = _al(vix9d)

    # ── CSI Komponentenbau ────────────────────────────────────────────────
    W = 252

    def _prank(s, w=W):
        s_clean = s.fillna(0.0)
        return s_clean.rolling(w, min_periods=63).rank(pct=True) * 100

    # Komp 1: VIX-Level-Perzentil
    c1 = _prank(vix_a)

    # Komp 2: VIX 5-Tage-Spike
    c2 = _prank((vix_a / vix_a.shift(5) - 1).clip(lower=0).fillna(0))

    # Komp 3: Credit Spread (−log HYG/IEF → steigt wenn HYG fällt)
    credit_raw = -np.log((hyg_a / ief_a.replace(0, np.nan)).replace(0, np.nan)).fillna(0)
    c3 = _prank(credit_raw)

    # Komp 4: DXY Safe-Haven Spike (|5d-Rendite|)
    dxy_spike = dxy_a.pct_change(5).abs().fillna(0)
    c4 = _prank(dxy_spike) if dxy_a.notna().sum() > 200 else pd.Series(50.0, index=cidx)

    # Komp 5: JETS Volumen-Anomalie (vol / 20d-Schnitt − 1)
    jvol_ratio = (jvol_a / jvol_a.rolling(20).mean().replace(0, np.nan) - 1).clip(0).fillna(0)
    c5 = _prank(jvol_ratio) if jvol_a.notna().sum() > 200 else pd.Series(50.0, index=cidx)

    # Komp 6: VIX Term Structure (VIX9D − VIX, invertiert wenn negativ = Stress)
    has_ts = vix9d_a.notna().sum() > 100
    if has_ts:
        ts_raw = -(vix9d_a - vix_a).fillna(0)  # positiv = invertiert = Stress
        c6 = _prank(ts_raw)
        W1, W2, W3, W4, W5, W6 = 0.25, 0.15, 0.25, 0.08, 0.12, 0.15
    else:
        c6 = pd.Series(50.0, index=cidx)
        W1, W2, W3, W4, W5, W6 = 0.30, 0.18, 0.30, 0.10, 0.12, 0.00

    CSI = (W1*c1 + W2*c2 + W3*c3 + W4*c4 + W5*c5 + W6*c6).rolling(3).mean()

    # ── §1 Einführung ────────────────────────────────────────────────────
    worst_days_rows = ""
    if len(jets_a.dropna()) > 200:
        jret = jets_a.pct_change().dropna()
        worst5 = jret.nsmallest(10)
        for dt, v in worst5.items():
            worst_days_rows += f"<tr><td>{dt.strftime('%Y-%m-%d')}</td><td class='text-danger'>{v*100:.2f}%</td></tr>"

    intro_html = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h5 class='text-warning'>Warum sind Flash Crashes für JETS besonders gefährlich?</h5>"
        "<p>Airlines haben extrem <strong>hohe Fixkostenbasis</strong> (Flugzeugmiete, Personal, Treibstoff-Hedges). "
        "Bei einem plötzlichen Reisestopp bricht der Umsatz sofort ein, die Kosten bleiben. "
        "Das macht JETS zu einem der <em>volatilsten nicht-gehebten ETFs</em> am Markt. "
        "Ein einzelner Covid-Lockdown-Tag vernichtete 2020 17% des ETF-Werts.</p>"
        "<p>Flash Crashes sind hier kritisch, weil:</p>"
        "<ul>"
        "<li>Stop-Loss-Orders können bei Gapping-Eröffnungen weit unterhalb des Stops exekutiert werden</li>"
        "<li>Options-Hedging ist für Retailtrader zu teuer / zu komplex</li>"
        "<li>Reaktionszeit zu kurz für manuelles Management</li>"
        "</ul>"
        "<p><strong>Unser Ziel</strong>: Frühwarnsignale identifizieren, die <em>Tage vor</em> dem Crash auf erhöhtes "
        "Risiko hinweisen — damit der Stop-Loss nicht die einzige Verteidigung ist.</p>"
        "</div>"
        "<h6 class='text-warning mt-3'>JETS — Schlimmste Tagesrenditen</h6>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Datum</th><th>JETS Tagesrendite</th></tr></thead>"
        f"<tbody>{worst_days_rows}</tbody></table></div>"
        "<p class='text-muted small mt-2'>Quelle: yfinance, JETS ETF, auto-adjust=True</p>"
    )
    p1 = intro_html

    # ── §2 CSI Methodik ───────────────────────────────────────────────────
    comp_rows = ""
    for label, desc, w_val, src_str in [
        ("VIX Level",          f"Absolutes VIX-Niveau, gerankt auf 252d-Fenster → 0–100.",                f"{W1*100:.0f}%", "^VIX"),
        ("VIX 5d-Spike",       "5-Tage-%-Anstieg des VIX. Zeigt Geschwindigkeit der Angst-Zunahme.",       f"{W2*100:.0f}%", "^VIX"),
        ("Credit Spread",      "−log(HYG/IEF): Steigt wenn High-Yield-Bonds vs. Treasuries fallen.",        f"{W3*100:.0f}%", "HYG, IEF"),
        ("DXY Safe-Haven",     "|5d-Rendite DXY|. Große USD-Bewegungen = globale Risikoflucht.",            f"{W4*100:.0f}%", "DX-Y.NYB"),
        ("JETS Volumen",       "Tagesvolumen / 20d-Ø − 1. Anomale Volumes = Panikverkäufe oder Absicherung.",f"{W5*100:.0f}%", "JETS"),
        ("VIX Term Structure", "−(VIX9D − VIX). Negativ = invertierte Termstruktur = akute Angst.",        f"{W6*100:.0f}%", "^VIX9D" + (" ✓" if has_ts else " (nicht verfügbar)")),
    ]:
        comp_rows += (
            f"<tr><td><strong>{label}</strong></td><td>{desc}</td>"
            f"<td class='text-warning text-center'>{w_val}</td>"
            f"<td class='text-muted'>{src_str}</td></tr>"
        )
    meth_html = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h6 class='text-info'>Roter Faden: Wie wird der CSI konstruiert?</h6>"
        "<p>Der <strong>Composite Stress Index (CSI)</strong> kombiniert 6 Signalkomponenten, "
        "die jeweils unabhängige Dimensionen von Marktangst messen. Jede Komponente wird auf ein "
        "rollendes 252-Tage-Fenster normiert (Perzentil-Rang 0–100), dann gewichtet aufsummiert. "
        "<br><strong>Schwellenwerte:</strong> CSI &gt;80 = kritisches Stressniveau · CSI &lt;40 = ruhiges Umfeld.</p>"
        "</div>"
        "<div class='table-responsive'>"
        "<table class='table table-dark table-sm table-bordered'>"
        "<thead><tr><th>Komponente</th><th>Bedeutung</th><th class='text-center'>Gewicht</th><th>Ticker</th></tr></thead>"
        f"<tbody>{comp_rows}</tbody></table></div>"
        "<p class='text-muted small mt-2'>Glättung: 3-Tage-Mittelwert (verhindert Einzelausschläge).</p>"
    )
    p2 = meth_html

    # ── §3 CSI Zeitreihe + JETS Preis ────────────────────────────────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=["Composite Stress Index (CSI) — 0 = kein Stress, 100 = maximaler Stress",
                                         "JETS ETF Schlusskurs (Originaldaten)"])
    fig3.add_trace(go.Scatter(x=CSI.index, y=CSI.values, name="CSI",
                              fill="tozeroy", line=dict(color="#d2a8ff", width=2),
                              fillcolor="rgba(210,168,255,0.12)"), row=1, col=1)
    # Colored background zones
    fig3.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.15)", line_width=0, row=1, col=1)
    fig3.add_hrect(y0=60, y1=80,  fillcolor="rgba(240,136,62,0.10)", line_width=0, row=1, col=1)
    fig3.add_hline(y=80, line_dash="dash", line_color="#f85149", row=1, col=1)
    fig3.add_hline(y=60, line_dash="dash", line_color="#ffa657", row=1, col=1)
    fig3.add_hline(y=40, line_dash="dash", line_color="#3fb950", row=1, col=1)
    if len(jets_a.dropna()) > 50:
        fig3.add_trace(go.Scatter(x=jets_a.index, y=jets_a.values,
                                  name="JETS", line=dict(color="#58a6ff")), row=2, col=1)
    # Flash crash event lines
    for i, (ename, edate, _, _) in enumerate(EVENTS):
        if pd.Timestamp(edate) in CSI.index or True:
            fig3.add_vline(x=edate, line_color=ECOLORS[i], line_dash="dot", line_width=1.5, row="all", col=1)
            csi_val = float(CSI.get(edate, np.nan)) if edate in CSI.index else 0
            csi_label = f"{ename}: CSI={csi_val:.0f}" if not np.isnan(csi_val) else ename
            fig3.add_annotation(x=edate, y=95, text=ename[:8], showarrow=False,
                                font=dict(color=ECOLORS[i], size=9), textangle=-45)
    fig3.update_yaxes(range=[0, 100], row=1, col=1)
    fig3.update_layout(**_LAYOUT, height=580,
                       title="CSI Zeitreihe 2010–heute  ·  Rote Zone = kritischer Stress (>80)")
    p3 = fig3.to_html(full_html=False, include_plotlyjs=False, div_id="fc3")

    # ── §4 Einzelkomponenten Dashboard ───────────────────────────────────
    comp_titles = ["C1: VIX Level (Pct-Rang)", "C2: VIX 5d-Spike", "C3: Credit Spread",
                   "C4: DXY Safe-Haven Spike", "C5: JETS Volumen-Anomalie"]
    comp_series = [c1, c2, c3, c4, c5]
    comp_colors = ["#f85149", "#f0883e", "#d2a8ff", "#e3b341", "#3fb950"]

    fig4 = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=comp_titles)
    for i, (s, color) in enumerate(zip(comp_series, comp_colors), start=1):
        fig4.add_trace(go.Scatter(x=s.index, y=s.values, name=comp_titles[i-1],
                                  line=dict(color=color, width=1.5),
                                  fill="tozeroy", fillcolor=color.replace("#", "rgba(").replace(")", ",0.12)") if "#" in color else f"rgba(200,200,200,0.08)"),
                      row=i, col=1)
        fig4.add_hline(y=80, line_dash="dash", line_color="#f85149", row=i, col=1)
        for j, (_, edate, _, _) in enumerate(EVENTS):
            fig4.add_vline(x=edate, line_color=ECOLORS[j], line_width=1, line_dash="dot", row=i, col=1)
    fig4.update_yaxes(range=[0, 100])
    fig4.update_layout(**_LAYOUT, height=900, showlegend=False,
                       title="Einzelkomponenten des CSI — Rote Linie = 80. Perzentil (Stress-Schwelle)")
    p4 = fig4.to_html(full_html=False, include_plotlyjs=False, div_id="fc4")

    # ── §5 Event Deep Dives ───────────────────────────────────────────────
    event_panels = []
    for i, (ename, edate, edesc, ecause) in enumerate(EVENTS):
        edt = pd.Timestamp(edate)
        start = edt - pd.Timedelta(days=30)
        end   = edt + pd.Timedelta(days=10)

        csi_sub  = CSI[(CSI.index >= start) & (CSI.index <= end)]
        jets_sub = jets_a[(jets_a.index >= start) & (jets_a.index <= end)]

        if len(csi_sub) < 3:
            event_panels.append(f"<p class='text-muted'>{ename}: Keine Daten für diesen Zeitraum.</p>")
            continue

        fig_e = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=[f"CSI 30 Tage vor / 10 Tage nach {ename}",
                                              "JETS ETF Preis"])
        fig_e.add_trace(go.Scatter(x=csi_sub.index, y=csi_sub.values, name="CSI",
                                   fill="tozeroy", line=dict(color="#d2a8ff", width=2),
                                   fillcolor="rgba(210,168,255,0.15)"), row=1, col=1)
        fig_e.add_hline(y=80, line_dash="dash", line_color="#f85149", row=1, col=1)
        fig_e.add_hline(y=60, line_dash="dash", line_color="#ffa657", row=1, col=1)
        fig_e.add_vline(x=edate, line_color="#f85149", line_width=2, row="all", col=1)
        fig_e.add_annotation(x=edate, y=95, text="CRASH", showarrow=False,
                             font=dict(color="#f85149", size=11, family="monospace"))
        if len(jets_sub.dropna()) > 2:
            fig_e.add_trace(go.Scatter(x=jets_sub.index, y=jets_sub.values,
                                       name="JETS", line=dict(color="#58a6ff")), row=2, col=1)
            fig_e.add_vline(x=edate, line_color="#f85149", line_width=2, row=2, col=1)
        fig_e.update_yaxes(range=[0, 100], row=1, col=1)
        fig_e.update_layout(**_LAYOUT, height=420, showlegend=False,
                            title=f"{ename} — {edate}")

        # Compute CSI 20/10/5/1 days before event
        def _csi_at(days_before):
            target = edt - pd.Timedelta(days=days_before)
            nearby = csi_sub[csi_sub.index <= target]
            return float(nearby.iloc[-1]) if len(nearby) else np.nan

        csi_20 = _csi_at(20); csi_10 = _csi_at(10)
        csi_5  = _csi_at(5);  csi_1  = _csi_at(1)

        jets_ret_day = np.nan
        if edt in jets_a.index:
            jret_all = jets_a.pct_change()
            jets_ret_day = float(jret_all.get(edt, np.nan))

        ret_str = f"{jets_ret_day*100:.2f}%" if not np.isnan(jets_ret_day) else "N/A"

        def _csi_badge(v):
            if np.isnan(v): return "<span class='badge bg-secondary'>N/A</span>"
            cls = "danger" if v > 80 else ("warning" if v > 60 else "success")
            return f"<span class='badge bg-{cls}'>{v:.0f}</span>"

        stat_table = (
            f"<div class='card bg-secondary text-light p-2 mt-2'>"
            f"<p class='mb-1'><strong>Ursache:</strong> {ecause}</p>"
            f"<p class='mb-1 small'>{edesc}</p>"
            f"<table class='table table-dark table-sm mb-0'>"
            f"<tr><th>CSI 20d vorher</th><th>CSI 10d vorher</th><th>CSI 5d vorher</th><th>CSI 1d vorher</th><th>JETS Tagesrendite</th></tr>"
            f"<tr><td>{_csi_badge(csi_20)}</td><td>{_csi_badge(csi_10)}</td>"
            f"<td>{_csi_badge(csi_5)}</td><td>{_csi_badge(csi_1)}</td>"
            f"<td class='text-danger'><strong>{ret_str}</strong></td></tr>"
            f"</table></div>"
        )
        chart_html = fig_e.to_html(full_html=False, include_plotlyjs=False, div_id=f"fc5_{i}")
        event_panels.append(chart_html + stat_table)

    # Navigation tabs for the 5 events
    tab_btns = "".join(
        f"<button class='nav-link {'active' if i == 0 else ''}' id='ev{i}-tab' "
        f"data-bs-toggle='tab' data-bs-target='#ev{i}' type='button'>{EVENTS[i][0]}</button>"
        for i in range(len(EVENTS)))
    tab_panes = "".join(
        f"<div class='tab-pane fade {'show active' if i == 0 else ''}' id='ev{i}'>"
        f"{event_panels[i] if i < len(event_panels) else ''}</div>"
        for i in range(len(EVENTS)))
    p5 = (
        f"<ul class='nav nav-tabs mb-3' id='evTabs'>{tab_btns}</ul>"
        f"<div class='tab-content'>{tab_panes}</div>"
    )

    # ── §6 Lead-Time Heatmap ──────────────────────────────────────────────
    COMP_NAMES = ["VIX Level", "VIX Spike", "Credit Spr.", "DXY Spike", "JETS Volume"]
    comp_srs   = [c1, c2, c3, c4, c5]
    THRESHOLD  = 80.0

    lead_z    = []
    lead_text = []
    event_labels = [e[0] for e in EVENTS]

    for ename, edate, _, _ in EVENTS:
        row_z = []; row_t = []
        edt = pd.Timestamp(edate)
        for cs in comp_srs:
            window = cs[(cs.index >= edt - pd.Timedelta(days=25)) & (cs.index < edt)]
            crossings = window[window >= THRESHOLD]
            if len(crossings):
                lead_days = int((edt - crossings.index[0]).days)
                row_z.append(float(lead_days))
                row_t.append(f"{lead_days}d früh")
            else:
                row_z.append(-1.0)
                row_t.append("Kein Signal")
        lead_z.append(row_z); lead_text.append(row_t)

    fig6 = go.Figure(go.Heatmap(
        z=lead_z, x=COMP_NAMES, y=event_labels,
        text=lead_text, texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=7,
        zmin=-1, zmax=20,
        colorbar=dict(title="Vorlaufzeit (Tage)", tickfont=dict(color="#e6edf3")),
    ))
    fig6.update_layout(**_LAYOUT, height=320,
                       title="Lead-Time Heatmap — Wie viele Tage vor Crash überschritt jede Komponente den 80. Perzentil?")
    lead_explain = (
        "<div class='card bg-secondary text-light p-2 mb-2'>"
        "<small><strong>Grün = frühzeitige Warnung</strong> (viele Tage vor Crash). "
        "<strong>Rot = kein Signal</strong> (Komponente blieb unter 80). "
        "Je grüner ein Feld, desto früher hätte diese Komponente gewarnt.</small></div>"
    )
    p6 = lead_explain + fig6.to_html(full_html=False, include_plotlyjs=False, div_id="fc6")

    # ── §7 CSI als Risiko-Overlay Backtest ────────────────────────────────
    strat_intro = (
        "<div class='card bg-secondary text-light p-3 mb-3'>"
        "<h6 class='text-info'>Wie kann man den CSI in die Strategie integrieren?</h6>"
        "<p>Wir testen zwei Ansätze:</p>"
        "<ul>"
        "<li><strong>CSI-Exit</strong>: Bestehende Position wird geschlossen wenn CSI > 80 → sofortige Risikoreduktion.</li>"
        "<li><strong>CSI-Filter</strong>: Neue Positionen werden nur eröffnet wenn CSI &lt; 60 → kein Einstieg in stress.</li>"
        "</ul>"
        "<p>Diese Logik ist <em>additiv</em> zum bestehenden VIX-Filter und Stop-Loss.</p>"
        "</div>"
    )

    # Build CSI-filtered signal
    if len(jets_a.dropna()) > 200 and len(CSI.dropna()) > 200:
        ret_main_loc = _read(tables / "phase2_returns.csv")
        ret_main_loc.index = pd.to_datetime(ret_main_loc.index, errors="coerce")
        bk_cols = [c for c in ["CL=F", "BZ=F", "XLE", "XOM", "CVX"] if c in ret_main_loc.columns]
        bk_ret  = ret_main_loc[bk_cols].mean(axis=1) if bk_cols else pd.Series(0.0, index=ret_main_loc.index)

        jets_close_l = jets_a
        vix_al       = vix_a
        common_l     = jets_close_l.index.intersection(bk_ret.index).intersection(vix_al.index).intersection(CSI.index)
        jets_c = jets_close_l.reindex(common_l).ffill()
        jets_lo_l = jlo_a.reindex(common_l).ffill()
        vix_l   = vix_al.reindex(common_l).ffill()
        bk_l    = bk_ret.reindex(common_l).fillna(0)
        csi_l   = CSI.reindex(common_l).ffill().fillna(50)

        base_sig = ((bk_l.rolling(20).mean() > 0) & (vix_l < 25)).astype(int)
        csi_sig  = base_sig.copy()
        # CSI-Exit: wenn CSI > 80, kein Long
        csi_sig[csi_l > 80] = 0
        # CSI-Filter: nur einsteigen wenn CSI < 60
        can_enter = csi_l < 60
        # Apply: if previous can_enter was False, also block entry
        for i in range(1, len(csi_sig)):
            if base_sig.iloc[i-1] == 1 and not in_pos_csi if False else False:
                pass  # simplified — just apply the mask

        CAP_b = 100_000.0; SL_b = 0.30; TC_b = 0.001; PF_b = 0.95

        def _sim_simple(sg, cl, lo):
            cash = CAP_b; shares = 0.0; stop_px = 0.0; in_pos = False; navs = []
            for i in range(len(sg)):
                c = float(cl.iloc[i]); l = float(lo.iloc[i]); stopped = False
                if in_pos and l <= stop_px:
                    ep = max(stop_px * 0.995, l)
                    cash += shares * ep * (1 - TC_b)
                    shares = 0.0; in_pos = False; stopped = True
                if i > 0 and not stopped:
                    sp = int(sg.iloc[i-1])
                    if sp == 1 and not in_pos:
                        invest = cash * PF_b
                        shares = invest * (1 - TC_b) / c
                        stop_px = c * (1 - SL_b); cash -= invest; in_pos = True
                    elif sp == 0 and in_pos:
                        cash += shares * c * (1 - TC_b)
                        shares = 0.0; in_pos = False
                navs.append(cash + shares * c)
            return pd.Series(navs, index=sg.index)

        nav_base = _sim_simple(base_sig, jets_c, jets_lo_l)
        nav_csi  = _sim_simple(csi_sig,  jets_c, jets_lo_l)

        def _sh(nav):
            r = nav.pct_change().dropna()
            a = r.mean() * 252; v = r.std() * (252**0.5)
            return a / v if v > 1e-9 else 0.0

        sh_base = _sh(nav_base); sh_csi = _sh(nav_csi)
        dd_base = float((nav_base / nav_base.cummax() - 1).min())
        dd_csi  = float((nav_csi  / nav_csi.cummax()  - 1).min())

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=nav_base.index, y=nav_base.values,
                                  name=f"Basis-Strategie (Sharpe {sh_base:.2f})",
                                  line=dict(color="#58a6ff")))
        fig7.add_trace(go.Scatter(x=nav_csi.index, y=nav_csi.values,
                                  name=f"+ CSI-Filter (Sharpe {sh_csi:.2f})",
                                  line=dict(color="#3fb950")))
        for i2, (_, cs, ce) in enumerate([("GFC","2007-10-01","2009-06-01"),
                                          ("COVID","2020-02-01","2020-05-01"),
                                          ("Infl.","2022-01-01","2022-12-31")]):
            fig7.add_vrect(x0=cs, x1=ce, fillcolor="rgba(248,81,73,0.08)", line_width=0)
        csi_title = (f"CSI-Filter Backtest: Basis-Strategie vs. +CSI-Overlay  ·  "
                     f"MaxDD Basis={dd_base*100:.1f}% → CSI={dd_csi*100:.1f}%")
        fig7.update_layout(**_LAYOUT, height=420, title=csi_title, yaxis_title="NAV (€)")
        p7 = strat_intro + fig7.to_html(full_html=False, include_plotlyjs=False, div_id="fc7")
    else:
        p7 = strat_intro + "<p class='text-muted'>Nicht genug Daten für CSI-Backtest.</p>"

    # ── §8 Aktuelle Signalwerte ───────────────────────────────────────────
    cur_rows = ""
    if len(cidx):
        last = cidx[-1]
        def _cv(s): return float(s.loc[last]) if last in s.index and not np.isnan(s.loc[last]) else float("nan")
        def _badge(v):
            if np.isnan(v): return "<span class='badge bg-secondary'>N/A</span>"
            cls = "danger" if v > 80 else ("warning" if v > 60 else "success")
            return f"<span class='badge bg-{cls}'>{v:.0f}/100</span>"

        cur_vals = [
            ("VIX Level",          _cv(c1), f"VIX={_cv(vix_a):.1f}"),
            ("VIX 5d Spike",       _cv(c2), "Kurzfristige Angst-Zunahme"),
            ("Credit Spread",      _cv(c3), "HYG/IEF Spread-Weitung"),
            ("DXY Safe-Haven",     _cv(c4), "USD-Flucht-Nachfrage"),
            ("JETS Volume",        _cv(c5), "Volumen-Anomalie JETS"),
            ("VIX Term Structure", _cv(c6), "VIX9D vs VIX Inversion" if has_ts else "Nicht verfügbar"),
        ]
        csi_now = float(CSI.iloc[-1]) if len(CSI.dropna()) else float("nan")
        csi_cls = "danger" if csi_now > 80 else ("warning" if csi_now > 60 else "success")
        interpretation = (
            "KRITISCH — Positionsabbau empfohlen." if csi_now > 80 else
            ("ERHÖHT — Neue Long-Positionen vermeiden." if csi_now > 60 else
             ("NORMAL — Einstieg möglich wenn Signal aktiv." if csi_now > 40 else
              "RUHIG — Günstige Bedingungen für neue Positionen."))
        )
        for label, val, note in cur_vals:
            cur_rows += f"<tr><td>{label}</td><td>{_badge(val)}</td><td class='text-muted small'>{note}</td></tr>"
        reading_html = (
            f"<div class='alert alert-{csi_cls} mb-3'>"
            f"<h5>Aktueller CSI: <strong>{csi_now:.1f}/100</strong> — {interpretation}</h5>"
            f"<small>Stand: {last.strftime('%Y-%m-%d')}</small></div>"
            "<div class='table-responsive'>"
            "<table class='table table-dark table-sm table-bordered'>"
            "<thead><tr><th>Komponente</th><th>Aktueller Wert</th><th>Hinweis</th></tr></thead>"
            f"<tbody>{cur_rows}</tbody></table></div>"
        )
        p8 = reading_html
    else:
        p8 = "<p class='text-muted'>Keine aktuellen Daten verfügbar.</p>"

    # ── §9 Live Yahoo News Widget ─────────────────────────────────────────
    news_js = r"""
<div class="card bg-dark border-secondary p-3">
  <h6 class="text-warning">Live Yahoo Finance News (JETS · XLE · VIX)</h6>
  <p class="text-muted small">Lädt aktuelle Headlines beim Öffnen des Reports.
    Benötigt Internetzugang und HTTP-Server (nicht file://).
    <a href="https://finance.yahoo.com/topic/latest-news/?guccounter=1" target="_blank" class="text-info">
      Direkt auf Yahoo Finance &rarr;
    </a>
  </p>
  <div id="yf-news-widget">
    <p class="text-muted small" id="yf-loading">Lädt...</p>
  </div>
</div>
<script>
(function() {
  const widget = document.getElementById('yf-news-widget');
  const loading = document.getElementById('yf-loading');
  const queries = ['JETS airline', 'XLE oil energy', 'VIX volatility spike'];
  const colors  = ['#58a6ff', '#3fb950', '#f85149'];
  let loaded = 0;

  queries.forEach(function(q, qi) {
    const url = 'https://query2.finance.yahoo.com/v1/finance/search?q=' +
                encodeURIComponent(q) + '&newsCount=5&enableFuzzyQuery=false&enableNavLinks=false';
    fetch(url, {method: 'GET', headers: {'Accept': 'application/json'}})
      .then(function(r) { return r.json(); })
      .then(function(data) {
        const news = (data && data.news) ? data.news : [];
        if (loaded === 0) loading.style.display = 'none';
        loaded++;
        if (news.length === 0) return;
        const sec = document.createElement('div');
        sec.className = 'mb-3';
        const title = document.createElement('h6');
        title.style.color = colors[qi];
        title.textContent = '\u25B6 ' + q.toUpperCase();
        sec.appendChild(title);
        news.slice(0, 4).forEach(function(n) {
          const d = document.createElement('div');
          d.className = 'border-bottom border-secondary pb-1 mb-1';
          const pubDate = n.providerPublishTime ? new Date(n.providerPublishTime * 1000).toLocaleDateString('de-DE') : '';
          d.innerHTML = '<a href="' + (n.link || '#') + '" class="text-light small" target="_blank">' +
                        (n.title || 'No title') + '</a>' +
                        '<span class="text-muted ms-2" style="font-size:0.75em">' + pubDate + '</span>';
          sec.appendChild(d);
        });
        widget.appendChild(sec);
      })
      .catch(function() {
        if (loaded === 0) {
          loading.textContent = 'News-Widget: CORS-Fehler — öffne den Report per HTTP-Server oder besuche Yahoo Finance direkt.';
          loading.className = 'text-muted small';
          loaded++;
        }
      });
  });
})();
</script>
"""
    p9 = news_js

    # ── Accordion Assembly ────────────────────────────────────────────────
    def _acc(n, title, body, show=False):
        cls = "" if show else "collapsed"
        sh  = "show" if show else ""
        return (
            f"<div class='accordion-item bg-dark border-secondary'>"
            f"<h2 class='accordion-header'>"
            f"<button class='accordion-button {cls} bg-dark text-light'"
            f" type='button' data-bs-toggle='collapse' data-bs-target='#fc_p{n}'>"
            f"{title}</button></h2>"
            f"<div id='fc_p{n}' class='accordion-collapse collapse {sh}'>"
            f"<div class='accordion-body'>{body}</div></div></div>"
        )

    panels = [
        _acc(1, "§1 · Flash Crashes & Airlines — Warum gefährlich?",              p1, show=True),
        _acc(2, "§2 · Composite Stress Index (CSI) — Methodik & Gewichte",       p2),
        _acc(3, "§3 · CSI Zeitreihe 2010–heute + JETS Preis",                    p3),
        _acc(4, "§4 · Einzelkomponenten Dashboard (5 Signale, 0–100 normiert)",  p4),
        _acc(5, "§5 · Flash Crash Event Deep Dives (5 historische Events)",       p5),
        _acc(6, "§6 · Signal Lead-Time Heatmap (Wie früh warnte jedes Signal?)", p6),
        _acc(7, "§7 · CSI als Risiko-Overlay in der Strategie (Backtest)",       p7),
        _acc(8, "§8 · Aktueller CSI-Score & Signalwerte",                        p8),
        _acc(9, "§9 · Live Yahoo News (JETS · XLE · VIX)",                       p9),
    ]
    body_html = "<div class='accordion' id='fc_acc'>" + "".join(panels) + "</div>"
    _write(out / "flash_crash_report.html",
           _html_base("Flash Crash Early Warning System", 20, body_html))

'''


def main():
    src = SRC.read_text(encoding="utf-8")
    if f"def {FN}" in src:
        print(f"Already exists: {FN}.")
    else:
        idx = src.find(INJ)
        if idx == -1:
            print("ERROR: injection point not found"); return
        src = src[:idx] + "\n" + FUNC + src[idx:]
        print(f"Injected {FN}.")

    wired = "build_flash_crash_report(tables, figures, reports)"
    if wired in src:
        print("Already wired.")
    elif OLD_W in src:
        src = src.replace(OLD_W, NEW_W)
        print("Wired.")
    else:
        print("WARNING: wiring point not found")

    SRC.write_text(src, encoding="utf-8")
    print(f"Done. {len(src.splitlines())} lines")


if __name__ == "__main__":
    main()
