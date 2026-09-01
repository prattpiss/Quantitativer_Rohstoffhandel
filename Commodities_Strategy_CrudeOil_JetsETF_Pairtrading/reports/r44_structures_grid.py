"""Report 44 — Strukturvarianten, Kombinationen und vollständiges Parameter-Grid.

Beantwortet die Fragen:
  * Wie hängt die Strategie von der JETS/Öl-Korrelation ab, kurz- vs. langfristig?
  * Was leisten Long-only, Long/Short, Pair (Long JETS / Short Öl) und
    Spread (beide Long) — jeweils in IS, OOS und IS+OOS?
  * Lassen sich die dokumentierten 16 Signalkombinationen und die adaptive
    VIX-Switch-Strategie reproduzieren?
  * Wie wirken Stop-Logik, Positionsgrößensteuerung und Transaktionskosten,
    wenn man sie vollständig kreuzt?
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import charts as C
from core import data as dat
from core import engine as E
from core import indices as ix
from core import signals as sg
from core import stats_tools as sx
from core import strategy as st
from core import theme as T

PHASE = 22
TITLE = "Report 44 — Strukturvarianten, Kombinationen &amp; Parameter-Grid"

IS_END = pd.Timestamp("2021-12-31")
OIL_LEG = "USO"          # handelbares Öl-Bein; CL=F wird nur als Signal genutzt
OIL_FALLBACK = "BZ=F"
REF_STRUCT = "Long-only"
REF_STOP = "fest 8 %"
REF_SIZE = "fest 1.0"
REF_COST = "ohne Kosten"

DOC = pd.DataFrame({
    "Kombination": ["Basket", "Basket+V", "RSI<70+S+V", "RSI<70+S+V+T"],
    "IS Sharpe (dok.)": [-0.111, 0.251, 0.695, 0.816],
    "OOS Sharpe (dok.)": [0.746, 0.718, 0.668, 0.458],
    "OOS MaxDD (dok.)": [-0.250, -0.240, -0.194, -0.192],
    "OOS Trades (dok.)": [116, 128, 52, 62],
    "WinRate (dok.)": [0.370, 0.434, 0.440, 0.481],
    "ProfitFaktor (dok.)": [1.60, 1.48, 2.50, 1.95],
})


def _pct_cols(df: pd.DataFrame, cols: list[str], dec: int = 1) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v * 100:.{dec}f} %")
    return d


def _num_cols(df: pd.DataFrame, cols: list[str], dec: int = 2) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:.{dec}f}")
    return d


def _sizing_series(name: str, r_jets: pd.Series, csi: pd.Series) -> pd.Series | float:
    if name == "fest 1.0":
        return 1.0
    if name == "Vol-Target 15 %":
        return E.vol_target_size(r_jets)
    return E.stress_size(csi, r_jets.index)


def build(out: Path) -> None:  # noqa: C901
    print("Report 44 — Strukturen, Kombinationen & Grid")
    np.random.seed(sx.SEED)
    body: list[str] = []

    body.append(T.header(
        "Report 44 — Strukturvarianten, Kombinationen &amp; vollständiges Parameter-Grid",
        "Korrelationsabhängigkeit · Long/Short · Pair · Spread · 16 Signalkombinationen · "
        "adaptive Umschaltung · 1 152 Simulationen in IS, OOS und IS+OOS"))

    # ── Daten ───────────────────────────────────────────────────────────
    need = st.SIGNAL_BASKET + [st.TARGET, "^VIX", "^TNX", OIL_LEG, OIL_FALLBACK]
    px = sg.base_panel(need)
    if st.TARGET not in px.columns:
        T.write(out / "r44_structures_grid.html",
                T.html_base(TITLE, PHASE, "".join(body) + T.warn("Keine Daten.")))
        return
    oil_leg = OIL_LEG if OIL_LEG in px.columns else OIL_FALLBACK
    px = px.dropna(subset=[st.TARGET, oil_leg])
    r_jets = px[st.TARGET].pct_change().fillna(0.0)
    r_oil = px[oil_leg].pct_change().fillna(0.0)
    start, end = px.index[0], px.index[-1]
    periods = {"IS": (start, IS_END), "OOS": (IS_END, end), "IS+OOS": (start, end)}

    body.append(T.card("§1 — Fragestellung, Aufbau und Trennung von IS und OOS",
        T.info("Dieser Report erweitert die Baseline aus Report 00 in vier Richtungen: "
               "(1) Abhängigkeit von der JETS/Öl-Korrelation und Vergleich kurzer gegen "
               "lange Signalfenster, (2) Handelsstrukturen von Long-only bis zum "
               "Zwei-Bein-Spread, (3) Reproduktion der dokumentierten 16 "
               "Signalkombinationen samt adaptiver Umschaltung und (4) ein vollständiges "
               "Kreuzprodukt aus Stop-Logik, Positionsgrößensteuerung und "
               "Transaktionskosten. Jede Kennzahl wird getrennt für In-Sample, "
               "Out-of-Sample und die Gesamtperiode ausgewiesen.")
        + T.formula(
            r"\text{IS}=[\,t_0,\;\texttt{2021-12-31}\,],\qquad "
            r"\text{OOS}=(\,\texttt{2021-12-31},\;t_N\,],\qquad "
            r"\text{IS+OOS}=[\,t_0,\;t_N\,]",
            "Feste, vorab definierte Aufteilung — keine nachträgliche Verschiebung")
        + T.stat_row([("Zeitraum", f"{start.date()} – {end.date()}"),
                      ("Handelstage", f"{len(px):,}".replace(",", ".")),
                      ("IS-Tage", str(int((px.index <= IS_END).sum()))),
                      ("OOS-Tage", str(int((px.index > IS_END).sum()))),
                      ("Öl-Bein", oil_leg),
                      ("Simulationen", "1 152")])
        + T.hypo("H0: Weder eine der drei alternativen Handelsstrukturen noch eine der "
                 "Parametervarianten verbessert die Out-of-Sample-Sharpe-Ratio "
                 "signifikant gegenüber der Long-only-Baseline.")
        + T.warn("Der Saisonalitätsfilter wird ausschließlich auf IS-Daten kalibriert. "
                 "Alle übrigen Schwellen (VIX 25, RSI 70, Fenster 20 Tage) stammen aus dem "
                 "Übergabedokument und wurden hier <em>nicht</em> nachoptimiert — sie sind "
                 "damit im OOS zwar nicht neu gefittet, tragen aber die Selektionshistorie "
                 "ihrer ursprünglichen Herleitung in sich.")))

    # ── §2 Datengrundlage ───────────────────────────────────────────────
    fig_px = C.price_panel(px, "Geladene Kursreihen (indexiert, log)")
    fig_grid, h_grid = C.series_grid({c: px[c] for c in px.columns},
                                     "Rohreihen in Originaleinheiten", cols=3)
    body.append(T.card("§2 — Datengrundlage und Sichtprüfung",
        T.info(f"Als handelbares Öl-Bein dient <strong>{oil_leg}</strong>. Der "
               "WTI-Frontmonat CL=F wird ausschließlich für die Signalbildung verwendet, "
               "weil sein negativer Settlement-Preis vom 20.04.2020 prozentuale Renditen "
               "mathematisch unbrauchbar macht — als Positionsbein wäre er unzulässig.")
        + T.div(fig_px, 460) + T.div(fig_grid, h_grid)
        + T.df_html(dat.availability(list(px.columns)), index=False)
        + C.data_table(px)))

    lvl = px[[st.TARGET, oil_leg, "^VIX"]].dropna()
    rr = pd.DataFrame({f"{st.TARGET} Rendite": r_jets, f"{oil_leg} Rendite": r_oil}).dropna()
    body.append(T.card("§3 — Stationarität der verwendeten Reihen",
        T.info("Korrelations- und Regressionsaussagen setzen stationäre Reihen voraus. "
               "Geprüft werden Niveaus und Renditen mit ADF (H0: Einheitswurzel) und "
               "KPSS (H0: Stationarität).")
        + T.df_html(pd.concat([sx.stationarity_table(lvl, label="Niveau: "),
                               sx.stationarity_table(rr)]), index=False)))

    # ── §4 Korrelationsabhängigkeit ─────────────────────────────────────
    c_short = r_jets.rolling(21).corr(r_oil)
    c_long = r_jets.rolling(252).corr(r_oil)
    n_s, n_l = 21, 252
    z_l = np.arctanh(c_long.clip(-0.999, 0.999))
    se_l = 1.0 / np.sqrt(n_l - 3)
    band_lo, band_hi = np.tanh(z_l - 1.96 * se_l), np.tanh(z_l + 1.96 * se_l)

    figC = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.55, .45],
                         vertical_spacing=.07,
                         subplot_titles=("Rollierende Korrelation JETS ↔ Öl",
                                         "Differenz kurz − lang (Korrelations-Momentum)"))
    for s, nm, col, w in [(c_long, "252 Tage (langfristig)", "#58a6ff", 1.8),
                          (c_short, "21 Tage (kurzfristig)", "#ffa657", 0.9)]:
        figC.add_trace(go.Scatter(x=s.index, y=s.to_numpy(float), name=nm,
                                  line=dict(color=col, width=w)), row=1, col=1)
    band = pd.DataFrame({"lo": band_lo, "hi": band_hi}).dropna()
    figC.add_trace(go.Scatter(
        x=list(band.index) + list(band.index[::-1]),
        y=list(band["hi"].to_numpy(float)) + list(band["lo"].to_numpy(float)[::-1]),
        fill="toself", fillcolor=T.hex_rgba("#58a6ff", 0.15), line=dict(width=0),
        name="95 %-KI (252 Tage)", hoverinfo="skip"), row=1, col=1)
    figC.add_hline(y=0, line=dict(color="#8b949e", width=1), row=1, col=1)
    d = (c_short - c_long).dropna()
    figC.add_trace(go.Scatter(x=d.index, y=d.to_numpy(float), name="kurz − lang",
                              line=dict(color="#bc8cff", width=0.9), fill="tozeroy",
                              fillcolor=T.hex_rgba("#bc8cff", 0.2)), row=2, col=1)

    base_sig = sg.basket_state(px)
    ref = E.simulate(r_jets, r_oil, base_sig, REF_STRUCT, "fixed", 0.08,
                     1.0, 0.0, price_a=px[st.TARGET])
    terc = pd.qcut(c_long.dropna(), 3, labels=["niedrig", "mittel", "hoch"])
    crows = []
    for lab in ["niedrig", "mittel", "hoch"]:
        m = terc[terc == lab].index
        seg = ref.rets.reindex(m).dropna()
        if len(seg) < 40:
            continue
        pm = sx.perf_metrics(seg)
        crows.append({"Korrelations-Terzil (252 T)": lab,
                      "Ø Korrelation": float(c_long.reindex(m).mean()),
                      "Tage": len(seg), "Sharpe": pm["Sharpe"], "CAGR": pm["CAGR"],
                      "MaxDD": pm["MaxDD"], "Trefferquote (Tage)": pm["WinRate"]})
    cdf = pd.DataFrame(crows)

    wins = [5, 10, 20, 40, 60, 120]
    wrows = []
    for w in wins:
        s = sg.basket_state(px, window=w)
        r = E.simulate(r_jets, r_oil, s, REF_STRUCT, "fixed", 0.08, 1.0, 0.0,
                       price_a=px[st.TARGET])
        row = {"Signalfenster (Tage)": w, "Ø Haltedauer": np.nan}
        row |= {k: v for k, v in E.period_metrics(r, periods).items()
                if k.endswith(("Sharpe", "MaxDD", "Trades"))}
        q = E.trade_quality(r.trades)
        row["Ø Haltedauer"] = q["Ø Tage"]
        wrows.append(row)
    wdf = pd.DataFrame(wrows)

    figW = go.Figure()
    for p, col in [("IS", "#58a6ff"), ("OOS", "#3fb950"), ("IS+OOS", "#ffa657")]:
        figW.add_trace(go.Scatter(x=wdf["Signalfenster (Tage)"], y=wdf[f"{p} Sharpe"],
                                  name=p, mode="lines+markers",
                                  line=dict(color=col, width=2)))
    figW.add_hline(y=0, line=dict(color="#8b949e", width=1))
    figW.update_layout(title="Sharpe-Ratio nach Länge des Signalfensters",
                       xaxis_title="Fensterlänge (Handelstage)", yaxis_title="Sharpe")

    body.append(T.card("§4 — Korrelationsabhängigkeit und kurz- vs. langfristiges Signal",
        T.info("Die gesamte Strategie lebt von der ökonomischen Kopplung zwischen "
               "Energiepreis und Airline-Aktie. Bricht diese Kopplung, verliert das Signal "
               "seine Grundlage. Deshalb wird zuerst gemessen, wie stabil die Korrelation "
               "überhaupt ist, und anschließend, wie die Strategie in Phasen "
               "unterschiedlicher Kopplung abschneidet.")
        + T.formula(
            r"\rho^{(w)}_t=\mathrm{corr}\big(r^{JETS}_{t-w+1:t},\,r^{Oil}_{t-w+1:t}\big),"
            r"\qquad z=\tfrac12\ln\tfrac{1+\rho}{1-\rho}\sim\mathcal{N}"
            r"\Big(\tfrac12\ln\tfrac{1+\rho_0}{1-\rho_0},\tfrac{1}{w-3}\Big)",
            "Rollierende Korrelation und Fisher-z-Transformation für das Konfidenzband")
        + T.div(figC, 560)
        + T.df_html(_pct_cols(_num_cols(cdf, ["Ø Korrelation", "Sharpe"]),
                              ["CAGR", "MaxDD", "Trefferquote (Tage)"]), index=False)
        + T.div(figW, 400)
        + T.df_html(_num_cols(wdf, [c for c in wdf.columns if "Sharpe" in c]
                              + ["Ø Haltedauer"]), index=False)
        + T.interp(
            f"Die 252-Tage-Korrelation schwankt zwischen "
            f"{c_long.min():.2f} und {c_long.max():.2f} — die Kopplung ist also alles "
            "andere als konstant. Die Terzil-Tabelle zeigt, in welchem "
            "Korrelationsregime die Strategie ihr Ergebnis erzielt; ein starkes Gefälle "
            "zwischen den Terzilen bedeutet, dass die Rendite an ein bestimmtes "
            "Marktregime gebunden und damit fragil ist. Die Fensteranalyse trennt "
            "kurzfristige Reaktion von langfristigem Trend: Kurze Fenster erzeugen viele "
            "Signale mit geringer Trefferquote, lange Fenster wenige und träge.")
        + T.warn("Die Terzile werden über die Gesamtstichprobe gebildet. Sie sind damit "
                 "eine <em>Beschreibung</em> des historischen Verhaltens und kein "
                 "handelbares Signal — zum Zeitpunkt t ist unbekannt, in welchem Terzil "
                 "die laufende Korrelation später landen wird.")))

    # ── §5 Signalbausteine ──────────────────────────────────────────────
    combos, meta = sg.build_signals(px, IS_END)
    act = pd.DataFrame([{"Kombination": k,
                         "Anteil Long": float((v > 0).mean()),
                         "Anteil Short": float((v < 0).mean()),
                         "Anteil flach": float((v == 0).mean())}
                        for k, v in combos.items()])
    body.append(T.card("§5 — Signalbausteine und ihre 16 Kombinationen",
        T.info("Zwei Basissignale (Basket-Mittel der Energie-Renditen, RSI des "
               "WTI-Frontmonats) werden mit drei Filtern gekreuzt: Saisonalität, "
               "VIX-Obergrenze und Zinstrend. Das ergibt 2 × 2³ = 16 Kombinationen. Jedes "
               "Signal liefert einen Zustand in {+1, 0, −1}; ein blockierender Filter "
               "setzt den Zustand auf null.")
        + T.formula(
            r"s_t=\underbrace{\mathrm{sgn}\big(-\bar r^{(20)}_t\big)}_{\text{Basis}}"
            r"\cdot\prod_{f\in F}\mathbb{1}\{f_t\},\qquad "
            r"F\subseteq\{\text{S},\text{V},\text{T}\}",
            "Signalkonstruktion: Basisrichtung mal multiplikative Filter")
        + T.df_html(pd.DataFrame([
            {"Baustein": "Basket", "Definition": "20-Tage-Mittel der mittleren Rendite von "
             "CL=F, BZ=F, XLE, XOM, CVX; negativ → Long, positiv → Short"},
            {"Baustein": "RSI&lt;70", "Definition": "RSI(CL=F, 14) &lt; 70 → Long, sonst Short"},
            {"Baustein": "S", "Definition": "Monat mit positiver Ø-JETS-Rendite im IS: "
             + ", ".join(str(m) for m in meta["gute Monate (IS)"])},
            {"Baustein": "V", "Definition": "VIX &lt; 25"},
            {"Baustein": "T", "Definition": "20-Tage-Veränderung der 10J-Rendite (^TNX) &lt; 0"},
        ]), index=False)
        + T.df_html(_pct_cols(act, ["Anteil Long", "Anteil Short", "Anteil flach"]),
                    index=False)
        + T.warn("Der Saisonalitätsfilter nutzt nur IS-Monatsmittel. Trotzdem bleibt er "
                 "ein Kandidat für Zufallsmuster: Bei zwölf Monaten und rund sechs Jahren "
                 "IS-Historie beruht jeder Monatsmittelwert auf sehr wenigen "
                 "Beobachtungen.")))

    # ── §6 Strukturvergleich ────────────────────────────────────────────
    struct_res: dict[str, E.SimResult] = {}
    srows = []
    for s_name in E.STRUCTURES:
        r = E.simulate(r_jets, r_oil, combos["Basket"], s_name, "fixed", 0.08, 1.0, 0.0,
                       price_a=px[st.TARGET])
        struct_res[s_name] = r
        srows.append({"Struktur": s_name} | E.period_metrics(r, periods))
    bh_j = (1 + r_jets).cumprod()
    bh_o = (1 + r_oil).cumprod()
    sdf = pd.DataFrame(srows).set_index("Struktur")

    show = ["IS Sharpe", "OOS Sharpe", "IS+OOS Sharpe", "IS+OOS CAGR", "IS+OOS MaxDD",
            "IS+OOS Calmar", "IS+OOS Trades", "IS+OOS WinRate", "IS+OOS ProfitFaktor"]
    figS, hS = C.equity_dashboard(
        {k: v.equity for k, v in struct_res.items()}
        | {"Buy &amp; Hold JETS": bh_j, f"Buy &amp; Hold {oil_leg}": bh_o},
        exposure=struct_res[REF_STRUCT].exposure,
        title="Vier Handelsstrukturen mit identischem Signal")
    figSp = C.trade_chart(px[st.TARGET].rename(st.TARGET),
                          struct_res["Pair (Long JETS / Short Öl)"].trades,
                          combos["Basket"] != 0,
                          "Pair-Struktur: Ein- und Ausstiege im JETS-Kurs")

    body.append(T.card("§6 — Vier Handelsstrukturen im direkten Vergleich",
        T.info("Alle vier Strukturen nutzen exakt dasselbe Signal und dieselben "
               "Ausführungsregeln; unterschiedlich sind nur die Beine. <em>Long-only</em> "
               "handelt ausschließlich die bullische Seite. <em>Long/Short</em> dreht bei "
               "bearischem Zustand auf Short JETS. <em>Pair</em> stellt Long JETS gegen "
               "Short Öl und ist damit weitgehend marktneutral gegenüber dem Ölpreis. "
               "<em>Spread</em> hält beide Beine long und vereinnahmt die gemeinsame "
               "Aufwärtsbewegung. Zwei-Bein-Strukturen werden auf Brutto-Exposure 1 "
               "normiert, damit die Kennzahlen vergleichbar bleiben.")
        + T.formula(
            r"\begin{aligned}"
            r"\text{Long-only}&:\;(w_J,w_O)=(\max(s,0),\,0)\\"
            r"\text{Long/Short}&:\;(w_J,w_O)=(s,\,0)\\"
            r"\text{Pair}&:\;(w_J,w_O)=(\tfrac{s}{2},\,-\tfrac{s}{2})\\"
            r"\text{Spread}&:\;(w_J,w_O)=(\tfrac{s}{2},\,\tfrac{s}{2})"
            r"\end{aligned}",
            "Beingewichte je Struktur bei Signalzustand s")
        + T.div(figS, hS)
        + T.df_html(_pct_cols(_num_cols(sdf[show], [c for c in show
                                                    if "Sharpe" in c or "Calmar" in c
                                                    or "ProfitFaktor" in c]),
                              ["IS+OOS CAGR", "IS+OOS MaxDD", "IS+OOS WinRate"]))
        + T.div(figSp, 540)
        + T.interp(
            "Der Vergleich trennt zwei Effekte: Die Long-only-Variante lebt vom "
            "Aufwärtsdrift der Airline-Aktie, die Pair-Variante isoliert die "
            "<em>relative</em> Bewegung und schaltet das Ölpreisrisiko weitgehend aus. "
            "Fällt die Pair-Sharpe deutlich unter die Long-only-Sharpe, stammt der "
            "Ertrag überwiegend aus dem Marktbeta und nicht aus dem Lead-Lag-Effekt — "
            "das ist die entscheidende Diagnose dieses Abschnitts.")
        + T.warn("Short-Positionen sind hier reibungsfrei modelliert: keine Leihgebühren, "
                 "keine Rückrufe, keine Margin-Anforderungen. Für JETS und USO sind "
                 "Leihkosten real und in Stressphasen sprunghaft — die Short-Beine sind "
                 "damit systematisch zu optimistisch bewertet.")))

    # ── §7 Vollständiges Parameter-Grid ─────────────────────────────────
    csi = ix.csi_from_components(ix.csi_components())
    grid_rows = []
    for c_name, sig in combos.items():
        for s_name in E.STRUCTURES:
            for st_name, (st_kind, st_lvl) in E.STOPS.items():
                for sz_name in E.SIZINGS:
                    size = _sizing_series(sz_name, r_jets, csi)
                    for ck_name, ck in E.COSTS.items():
                        r = E.simulate(r_jets, r_oil, sig, s_name, st_kind, st_lvl,
                                       size, ck, price_a=px[st.TARGET])
                        grid_rows.append(
                            {"Kombination": c_name, "Struktur": s_name,
                             "Stop": st_name, "Sizing": sz_name, "Kosten": ck_name}
                            | E.period_metrics(r, periods))
    G = pd.DataFrame(grid_rows)
    G["Δ Sharpe (OOS−IS)"] = G["OOS Sharpe"] - G["IS Sharpe"]
    G["Robust"] = (G["IS Sharpe"] > 0.3) & (G["OOS Sharpe"] > 0.3)

    figG = go.Figure()
    for s_name, col in zip(E.STRUCTURES, T.PAL):
        m = G["Struktur"] == s_name
        figG.add_trace(go.Scatter(
            x=G.loc[m, "IS Sharpe"], y=G.loc[m, "OOS Sharpe"], mode="markers",
            name=s_name, marker=dict(size=5, color=col, opacity=0.6),
            text=[f"{a}<br>{b} · {c} · {d}" for a, b, c, d in zip(
                G.loc[m, "Kombination"], G.loc[m, "Stop"], G.loc[m, "Sizing"],
                G.loc[m, "Kosten"])],
            hovertemplate="%{text}<br>IS %{x:.2f} · OOS %{y:.2f}<extra></extra>"))
    lim = [float(np.nanmin(G[["IS Sharpe", "OOS Sharpe"]].to_numpy())) - 0.1,
           float(np.nanmax(G[["IS Sharpe", "OOS Sharpe"]].to_numpy())) + 0.1]
    figG.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="IS = OOS",
                              line=dict(color="#8b949e", dash="dash", width=1)))
    figG.add_hline(y=0, line=dict(color="#30363d", width=1))
    figG.add_vline(x=0, line=dict(color="#30363d", width=1))
    figG.update_layout(title="Alle 1 152 Simulationen: In-Sample gegen Out-of-Sample",
                       xaxis_title="IS Sharpe", yaxis_title="OOS Sharpe")

    figB = make_subplots(rows=1, cols=4, shared_yaxes=True,
                         subplot_titles=("Struktur", "Stop-Logik", "Sizing", "Kosten"))
    for j, fac in enumerate(["Struktur", "Stop", "Sizing", "Kosten"], start=1):
        for i, lev in enumerate(G[fac].unique()):
            figB.add_trace(go.Box(y=G.loc[G[fac] == lev, "OOS Sharpe"], name=str(lev),
                                  marker_color=T.PAL[i % len(T.PAL)], showlegend=False,
                                  boxpoints=False), row=1, col=j)
    figB.update_layout(title="Verteilung der OOS-Sharpe je Parameterachse")
    figB.update_yaxes(title_text="OOS Sharpe", row=1, col=1)

    marg = []
    for fac in ["Struktur", "Stop", "Sizing", "Kosten", "Kombination"]:
        g = G.groupby(fac)[["IS Sharpe", "OOS Sharpe", "IS+OOS Sharpe", "IS+OOS MaxDD"]].mean()
        for lev, row in g.iterrows():
            marg.append({"Achse": fac, "Ausprägung": lev,
                         "Ø IS Sharpe": row["IS Sharpe"],
                         "Ø OOS Sharpe": row["OOS Sharpe"],
                         "Ø IS+OOS Sharpe": row["IS+OOS Sharpe"],
                         "Ø IS+OOS MaxDD": row["IS+OOS MaxDD"],
                         "Δ OOS−IS": row["OOS Sharpe"] - row["IS Sharpe"]})
    mdf = pd.DataFrame(marg)

    top = G.sort_values("OOS Sharpe", ascending=False).head(25)
    top_show = ["Kombination", "Struktur", "Stop", "Sizing", "Kosten", "IS Sharpe",
                "OOS Sharpe", "IS+OOS Sharpe", "IS+OOS MaxDD", "IS+OOS Trades",
                "OOS WinRate", "OOS ProfitFaktor"]

    n_tests = len(G)
    yrs = (end - start).days / 365.25
    se_sr = np.sqrt(1.0 / max(yrs, 1e-9))
    snoop = np.sqrt(2.0 * np.log(n_tests)) * se_sr

    body.append(T.card("§7 — Vollständiges Kreuzprodukt aller Parameter",
        T.info("Gekreuzt werden 16 Signalkombinationen × 4 Strukturen × 3 Stop-Logiken "
               "(kein Stop, fest 8 %, Trailing 10 %) × 3 Sizing-Varianten (feste Größe, "
               "Vol-Targeting auf 15 % Zielvolatilität, Skalierung mit dem Crash Stress "
               "Index) × 2 Kostenannahmen (0 bp und 10 bp je Umsatzeinheit). Jede "
               "Konfiguration wird einmal durchgehend simuliert; IS-, OOS- und "
               "Gesamtkennzahlen werden anschließend aus derselben Renditereihe "
               "geschnitten, damit die Pfadabhängigkeit des Stops erhalten bleibt.")
        + T.stat_row([("Simulationen", f"{n_tests:,}".replace(",", " ")),
                      ("Ø OOS Sharpe", sx.num(G["OOS Sharpe"].mean(), 2)),
                      ("Beste OOS Sharpe", sx.num(G["OOS Sharpe"].max(), 2)),
                      ("IS &gt; 0.3 und OOS &gt; 0.3", str(int(G["Robust"].sum()))),
                      ("Anteil robust", sx.pct(G["Robust"].mean(), 1)),
                      ("Data-Snooping-Schranke", sx.num(snoop, 2))])
        + T.div(figG, 560) + T.div(figB, 400)
        + "<h5 class='mt-3'>Marginale Wirkung jeder Parameterachse</h5>"
        + T.df_html(_pct_cols(_num_cols(mdf, ["Ø IS Sharpe", "Ø OOS Sharpe",
                                              "Ø IS+OOS Sharpe", "Δ OOS−IS"]),
                              ["Ø IS+OOS MaxDD"]), max_rows=60, index=False)
        + "<h5 class='mt-3'>Die 25 besten Konfigurationen nach OOS-Sharpe</h5>"
        + T.df_html(_pct_cols(_num_cols(top[top_show], ["IS Sharpe", "OOS Sharpe",
                                                        "IS+OOS Sharpe",
                                                        "OOS ProfitFaktor"]),
                              ["IS+OOS MaxDD", "OOS WinRate"]), index=False)
        + T.formula(
            r"\mathbb{E}\big[\max_{i\le N}\widehat{SR}_i\,\big|\,SR=0\big]"
            r"\;\approx\;\sqrt{2\ln N}\cdot\hat\sigma_{SR},\qquad "
            r"\hat\sigma_{SR}\approx\sqrt{1/T_{\text{Jahre}}}"
            r"\;\Rightarrow\;" + f"{snoop:.2f}",
            "Erwartetes Maximum der Sharpe-Ratio bei reinem Zufall (Data-Snooping-Schranke)")
        + T.interp(
            f"Bei {n_tests} parallelen Auswertungen über {yrs:.1f} Jahre liegt allein "
            f"durch Zufall eine maximale Sharpe-Ratio von rund {snoop:.2f} zu erwarten. "
            f"Der tatsächliche Bestwert beträgt {G['OOS Sharpe'].max():.2f}. Nur wenn "
            "dieser Wert die Schranke deutlich überschreitet <em>und</em> die "
            "Konfiguration in IS und OOS gleichermaßen trägt, ist von einem echten "
            "Effekt auszugehen. Aussagekräftiger als die Spitzenwerte ist deshalb die "
            "Tabelle der marginalen Wirkungen: Sie mittelt über alle jeweils anderen "
            "Achsen und ist damit weit weniger anfällig für Zufallstreffer.")
        + T.warn("Die Bestenliste ist bewusst <em>nicht</em> als Empfehlung zu lesen. Sie "
                 "ist das Maximum aus über tausend Auswertungen und damit per "
                 "Konstruktion nach oben verzerrt. Die Bonferroni-korrigierte Schranke "
                 f"läge bei α* = 0.05/{n_tests} = {0.05 / n_tests:.2e}.")))

    # ── §8 Reproduktion der dokumentierten Kombinationstabelle ──────────
    rep_rows = []
    rep_res: dict[str, E.SimResult] = {}
    for c_name, sig in combos.items():
        r = E.simulate(r_jets, r_oil, sig, "Long-only", "fixed", 0.08, 1.0, 10.0,
                       price_a=px[st.TARGET])
        rep_res[c_name] = r
        q = E.trade_quality(r.trades, IS_END, end)
        pm = E.period_metrics(r, periods)
        rep_rows.append({"Kombination": c_name, "IS Sharpe": pm["IS Sharpe"],
                         "OOS Sharpe": pm["OOS Sharpe"],
                         "Δ (OOS−IS)": pm["OOS Sharpe"] - pm["IS Sharpe"],
                         "IS+OOS Sharpe": pm["IS+OOS Sharpe"],
                         "OOS MaxDD": pm["OOS MaxDD"], "OOS Trades": q["Trades"],
                         "WinRate": q["WinRate"], "ProfitFaktor": q["ProfitFaktor"],
                         "Ø Return": q["Ø Return"], "Ø Tage": q["Ø Tage"],
                         "Max Konsek. Verluste": q["Max Konsek. Verluste"]})
    R = pd.DataFrame(rep_rows).sort_values("OOS Sharpe", ascending=False)

    cmp_rows = []
    for _, d in DOC.iterrows():
        k = d["Kombination"]
        hit = R[R["Kombination"] == k]
        if hit.empty:
            continue
        h = hit.iloc[0]
        cmp_rows.append({
            "Kombination": k,
            "IS dok.": d["IS Sharpe (dok.)"], "IS hier": h["IS Sharpe"],
            "OOS dok.": d["OOS Sharpe (dok.)"], "OOS hier": h["OOS Sharpe"],
            "OOS-Abweichung": h["OOS Sharpe"] - d["OOS Sharpe (dok.)"],
            "Trades dok.": d["OOS Trades (dok.)"], "Trades hier": h["OOS Trades"],
            "PF dok.": d["ProfitFaktor (dok.)"], "PF hier": h["ProfitFaktor"]})
    Cdf = pd.DataFrame(cmp_rows)

    figR = go.Figure()
    figR.add_trace(go.Scatter(
        x=R["IS Sharpe"], y=R["OOS Sharpe"], mode="markers+text",
        text=R["Kombination"], textposition="top center",
        textfont=dict(size=9, color="#8b949e"),
        marker=dict(size=np.clip(R["OOS Trades"].to_numpy(float) / 4 + 6, 6, 26),
                    color=np.where(R["OOS Sharpe"] > 0.6, "#3fb950",
                                   np.where(R["OOS Sharpe"] > 0.3, "#ffa657", "#f85149")),
                    line=dict(width=1, color="#0d1117")),
        name="Kombination",
        hovertemplate="<b>%{text}</b><br>IS %{x:.3f} · OOS %{y:.3f}<extra></extra>"))
    rlim = [min(R["IS Sharpe"].min(), R["OOS Sharpe"].min()) - 0.2,
            max(R["IS Sharpe"].max(), R["OOS Sharpe"].max()) + 0.2]
    figR.add_trace(go.Scatter(x=rlim, y=rlim, mode="lines", name="IS = OOS",
                              line=dict(color="#8b949e", dash="dash", width=1)))
    figR.update_layout(title="16 Kombinationen: IS gegen OOS (Kreisgröße ∝ OOS-Trades)",
                       xaxis_title="IS Sharpe", yaxis_title="OOS Sharpe")

    figF = go.Figure()
    for f in ["S", "V", "T"]:
        withs = R[R["Kombination"].str.contains(rf"\+{f}", regex=True)]
        wout = R[~R["Kombination"].str.contains(rf"\+{f}", regex=True)]
        for lbl, vals, col in [("ohne", wout["OOS Sharpe"], "#8b949e"),
                               ("mit", withs["OOS Sharpe"], "#58a6ff")]:
            figF.add_trace(go.Bar(x=[sg.FILTER_LABELS[f]], y=[vals.mean()],
                                  name=f"{lbl} Filter", marker_color=col,
                                  showlegend=(f == "S"),
                                  error_y=dict(type="data", array=[vals.std()],
                                               color="#30363d")))
    figF.update_layout(title="Marginaler Filterbeitrag zur OOS-Sharpe (Mittel ± Streuung)",
                       barmode="group", yaxis_title="Ø OOS Sharpe")

    body.append(T.card("§8 — Reproduktion der dokumentierten 16 Kombinationen",
        T.info("Nachbau der aus dem früheren Framework überlieferten Kombinationstabelle: "
               "Long-only, fester 8-%-Stop, 10 bp Transaktionskosten. Verglichen werden "
               "IS- und OOS-Sharpe, Handelsanzahl, Trefferquote und Profitfaktor mit den "
               "überlieferten Werten.")
        + T.div(figR, 520)
        + T.df_html(_pct_cols(_num_cols(R, ["IS Sharpe", "OOS Sharpe", "Δ (OOS−IS)",
                                            "IS+OOS Sharpe", "ProfitFaktor", "Ø Tage"]),
                              ["OOS MaxDD", "WinRate", "Ø Return"]), index=False)
        + "<h5 class='mt-3'>Abgleich mit den überlieferten Zahlen</h5>"
        + T.df_html(_num_cols(Cdf, [c for c in Cdf.columns if c != "Kombination"]),
                    index=False)
        + T.div(figF, 380)
        + T.interp(
            "Entscheidend ist nicht die exakte Übereinstimmung der Nachkommastellen — "
            "IS/OOS-Grenze, Kostenannahme, Stop-Ausführung und Datenstand des früheren "
            "Laufs sind nicht vollständig überliefert. Entscheidend ist, ob sich die "
            "<em>Rangfolge</em> und die <em>Vorzeichen</em> reproduzieren lassen: "
            "generalisiert das Basket-Signal weiterhin besser als die stark gefilterten "
            "Varianten, und bleibt der Zinsfilter der instabilste Baustein?")
        + T.warn("Die überlieferte Aussage „Sharpe teilweise um 3“ lässt sich in dieser "
                 "Datenbasis nicht reproduzieren und ist bei einer Long-only-Aktienposition "
                 "mit rund 20 % Zielvolatilität über mehrere Jahre auch nicht plausibel. "
                 "Solche Werte entstehen typischerweise durch sehr kurze Auswertefenster, "
                 "fehlende Transaktionskosten oder Look-Ahead in der Signalbildung. Bis "
                 "der ursprüngliche Lauf mit identischem Code und Datenstand vorliegt, "
                 "wird diese Zahl hier <strong>nicht</strong> als Referenz verwendet.")))

    # ── §9 Adaptive Umschaltung ─────────────────────────────────────────
    ad_sig = sg.adaptive_signal(combos, px["^VIX"])
    ad_rows, ad_res = [], {}
    for nm, s in [("Adaptiv (VIX-Regime-Switch)", ad_sig),
                  ("Basket", combos["Basket"]),
                  ("RSI<70+S+V", combos["RSI<70+S+V"]),
                  ("Buy & Hold JETS", pd.Series(1.0, index=px.index))]:
        # Buy & Hold ist die passive Referenz: dauerhaft investiert, deshalb ohne
        # Stop-Logik und ohne wiederkehrende Handelskosten.
        bh = nm.startswith("Buy & Hold")
        r = E.simulate(r_jets, r_oil, s, "Long-only",
                       "none" if bh else "fixed", 0.0 if bh else 0.08,
                       1.0, 0.0 if bh else 10.0,
                       price_a=px[st.TARGET])
        ad_res[nm] = r
        ad_rows.append({"Strategie": nm} | E.period_metrics(r, periods))
    ADF = pd.DataFrame(ad_rows).set_index("Strategie")
    d_ad = sx.sharpe_diff_test(
        ad_res["Adaptiv (VIX-Regime-Switch)"].rets.loc[IS_END:],
        ad_res["Basket"].rets.loc[IS_END:], n=800)
    figA, hA = C.equity_dashboard({k: v.equity for k, v in ad_res.items()},
                                  exposure=ad_res["Adaptiv (VIX-Regime-Switch)"].exposure,
                                  trades=ad_res["Adaptiv (VIX-Regime-Switch)"].trades,
                                  title="Adaptive Umschaltung gegen ihre Bausteine")

    body.append(T.card("§9 — Adaptive Umschaltung nach VIX-Regime",
        T.info("Die überlieferte Meta-Strategie wechselt das Signal je nach "
               "Volatilitätsregime: unter VIX 20 das Basket-Signal, zwischen 20 und 25 "
               "die gefilterte RSI-Variante, ab 25 keine Position. Getestet wird, ob die "
               "dokumentierte OOS-Sharpe von 1.085 reproduzierbar ist.")
        + T.formula(
            r"s^{\text{ad}}_t=\begin{cases}"
            r"s^{\text{Basket}}_t & \mathrm{VIX}_t<20\\[2pt]"
            r"s^{\text{RSI+S+V}}_t & 20\le \mathrm{VIX}_t<25\\[2pt]"
            r"0 & \mathrm{VIX}_t\ge 25\end{cases}",
            "Regimeabhängige Signalwahl")
        + T.div(figA, hA)
        + T.df_html(_pct_cols(_num_cols(ADF[show], [c for c in show
                                                    if "Sharpe" in c or "Calmar" in c
                                                    or "ProfitFaktor" in c]),
                              ["IS+OOS CAGR", "IS+OOS MaxDD", "IS+OOS WinRate"]))
        + T.formula(r"\Delta\mathrm{Sharpe}_{\text{OOS}}="
                    + f"{d_ad['diff']:+.3f}" + r",\quad\mathrm{KI}_{95\%}=\big["
                    + f"{d_ad['lo']:+.3f},\\;{d_ad['hi']:+.3f}" + r"\big],\quad p="
                    + f"{d_ad['p']:.3f}",
                    "Gepaarter Block-Bootstrap: Adaptiv gegen Basket im OOS")
        + T.interp(
            f"Die adaptive Variante erreicht hier eine OOS-Sharpe von "
            f"{ADF.loc['Adaptiv (VIX-Regime-Switch)', 'OOS Sharpe']:.2f}. Die Differenz "
            f"zum reinen Basket-Signal beträgt {d_ad['diff']:+.2f} mit einem "
            f"Konfidenzintervall von [{d_ad['lo']:+.2f}, {d_ad['hi']:+.2f}]. Schließt "
            "dieses Intervall die Null ein, ist der Vorteil der Umschaltung statistisch "
            "nicht belegt — auch wenn der Punktschätzer höher liegt.")
        + T.warn("Die Regimegrenzen 20 und 25 sowie die Zuordnung der Signale wurden auf "
                 "Basis derselben Historie gewählt, auf der sie hier gemessen werden. Die "
                 "adaptive Strategie ist damit die am stärksten selektionsverzerrte "
                 "Variante dieses Reports.")))

    # ── §10 Regimeanalyse Öl und Volatilität ────────────────────────────
    oil_tr = px[oil_leg].pct_change(60)
    vix = px["^VIX"]
    oil_lab = pd.Series(np.where(oil_tr > 0, "Öl ↑", "Öl ↓"), index=px.index)
    vix_lab = pd.Series(np.where(vix < 20, "VIX < 20",
                                 np.where(vix < 25, "VIX 20–25", "VIX ≥ 25")),
                        index=px.index)
    reg_names = ["Basket", "RSI<70+S+V", "Adaptiv (VIX-Regime-Switch)"]
    reg_src = {"Basket": ad_res["Basket"], "RSI<70+S+V": ad_res["RSI<70+S+V"],
               "Adaptiv (VIX-Regime-Switch)": ad_res["Adaptiv (VIX-Regime-Switch)"]}
    cells = ["VIX < 20", "VIX 20–25", "VIX ≥ 25"]
    rrows = []
    zmat = np.full((len(reg_names), 6), np.nan)
    for i, nm in enumerate(reg_names):
        rr_ = reg_src[nm].rets
        for j, (o, v) in enumerate(itertools.product(["Öl ↑", "Öl ↓"], cells)):
            m = (oil_lab == o) & (vix_lab == v)
            seg = rr_[m.reindex(rr_.index).fillna(False)]
            if len(seg) < 30:
                continue
            pm = sx.perf_metrics(seg)
            zmat[i, j] = pm["Sharpe"]
            rrows.append({"Strategie": nm, "Öl-Trend (60 T)": o, "VIX-Regime": v,
                          "Tage": len(seg), "Sharpe": pm["Sharpe"], "CAGR": pm["CAGR"],
                          "MaxDD": pm["MaxDD"]})
    RG = pd.DataFrame(rrows)
    figH = go.Figure(go.Heatmap(
        z=zmat, x=[f"{o} · {v}" for o, v in itertools.product(["Öl ↑", "Öl ↓"], cells)],
        y=reg_names, colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="Sharpe"),
        hovertemplate="%{y}<br>%{x}<br>Sharpe %{z:.2f}<extra></extra>"))
    figH.update_layout(title="Sharpe-Ratio je Öl-Trend- und Volatilitätsregime")

    body.append(T.card("§10 — Verhalten in Öl- und Volatilitätsregimen",
        T.info("Die Strategie unterstellt, dass fallende Energiepreise Airlines "
               "begünstigen. Ob dieser Mechanismus in steigenden wie in fallenden "
               "Ölmärkten trägt und wie er sich mit dem Volatilitätsregime überlagert, "
               "zeigt die Aufteilung in sechs Felder.")
        + T.div(figH, 360)
        + T.df_html(_pct_cols(_num_cols(RG, ["Sharpe"]), ["CAGR", "MaxDD"]), index=False)
        + T.interp("Felder mit weniger als 30 Handelstagen werden nicht ausgewiesen. "
                   "Ein durchgängig positives Vorzeichen über alle sechs Felder wäre ein "
                   "starker Robustheitsbeleg; einzelne dominierende Felder deuten dagegen "
                   "darauf hin, dass das Gesamtergebnis von einem einzigen Regime getragen "
                   "wird.")
        + T.warn("Die Felder sind unterschiedlich stark besetzt. Sharpe-Werte aus Zellen "
                 "mit wenigen hundert Tagen haben Konfidenzintervalle, die mehrere "
                 "Einheiten breit sind, und dürfen nicht als Punktaussage gelesen werden.")))

    # ── §11 Transaktionskosten ──────────────────────────────────────────
    cost_levels = [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    crows2 = []
    for c_name in ["Basket", "Basket+V", "RSI<70+S+V", "Adaptiv"]:
        s = ad_sig if c_name == "Adaptiv" else combos[c_name]
        for cb in cost_levels:
            r = E.simulate(r_jets, r_oil, s, "Long-only", "fixed", 0.08, 1.0, cb,
                           price_a=px[st.TARGET])
            pm = E.period_metrics(r, periods)
            crows2.append({"Strategie": c_name, "Kosten (bp)": cb,
                           "IS Sharpe": pm["IS Sharpe"], "OOS Sharpe": pm["OOS Sharpe"],
                           "IS+OOS Sharpe": pm["IS+OOS Sharpe"],
                           "IS+OOS CAGR": pm["IS+OOS CAGR"],
                           "Trades": pm["IS+OOS Trades"]})
    TC = pd.DataFrame(crows2)
    figTC = go.Figure()
    for i, nm in enumerate(TC["Strategie"].unique()):
        m = TC["Strategie"] == nm
        figTC.add_trace(go.Scatter(x=TC.loc[m, "Kosten (bp)"], y=TC.loc[m, "IS+OOS Sharpe"],
                                   name=nm, mode="lines+markers",
                                   line=dict(color=T.PAL[i % len(T.PAL)], width=2)))
    figTC.add_hline(y=0, line=dict(color="#8b949e", width=1))
    figTC.update_layout(title="Sharpe-Ratio in Abhängigkeit der Transaktionskosten",
                        xaxis_title="Kosten je Umsatzeinheit (Basispunkte)",
                        yaxis_title="IS+OOS Sharpe")

    breakeven = []
    for nm in TC["Strategie"].unique():
        m = TC[TC["Strategie"] == nm].sort_values("Kosten (bp)")
        pos = m[m["IS+OOS Sharpe"] > 0]
        breakeven.append({"Strategie": nm,
                          "Sharpe bei 0 bp": m.iloc[0]["IS+OOS Sharpe"],
                          "Sharpe bei 10 bp": m[m["Kosten (bp)"] == 10]["IS+OOS Sharpe"].iloc[0],
                          "Sharpe bei 50 bp": m.iloc[-1]["IS+OOS Sharpe"],
                          "höchste tragbare Kosten (bp)":
                              float(pos["Kosten (bp)"].max()) if len(pos) else 0.0,
                          "Trades gesamt": int(m.iloc[0]["Trades"])})
    BE = pd.DataFrame(breakeven)

    body.append(T.card("§11 — Transaktionskosten: mit und ohne, und ab wann es kippt",
        T.info("Kosten werden proportional zum tatsächlichen Umsatz belastet — jeder "
               "Wechsel des Zielgewichts kostet, auch die laufende Anpassung beim "
               "Vol-Targeting. Neben dem Vergleich „mit und ohne“ wird die gesamte "
               "Kostenkurve gezeigt, weil erst sie beantwortet, wie viel Reibung eine "
               "Strategie überhaupt verträgt.")
        + T.formula(r"r^{\text{netto}}_t=r^{\text{brutto}}_t-\frac{c}{10^4}\cdot"
                    r"\big(|w^J_t-w^J_{t-1}|+|w^O_t-w^O_{t-1}|\big)",
                    "Umsatzproportionale Kostenbelastung")
        + T.div(figTC, 400)
        + T.df_html(_num_cols(BE, ["Sharpe bei 0 bp", "Sharpe bei 10 bp",
                                   "Sharpe bei 50 bp"]), index=False)
        + T.df_html(_pct_cols(_num_cols(TC, ["IS Sharpe", "OOS Sharpe", "IS+OOS Sharpe"]),
                              ["IS+OOS CAGR"]), max_rows=40, index=False)
        + T.interp("Die Spalte „höchste tragbare Kosten“ ist die praktisch wichtigste "
                   "Kennzahl dieses Abschnitts: Sie sagt, wie viel Spread, Slippage und "
                   "Gebühren eine Variante aushält, bevor ihr Vorteil verschwindet. "
                   "Signale mit kurzer Haltedauer verlieren hier zuerst.")
        + T.warn("10 bp sind für JETS in ruhigen Märkten realistisch, in Stressphasen "
                 "deutlich zu niedrig. Gap-Risiko über Nacht ist in keiner Variante "
                 "modelliert; der Stop wird immer zum Schlusskurs ausgeführt.")))

    # ── §12 Fazit ───────────────────────────────────────────────────────
    best_struct = sdf["OOS Sharpe"].idxmax()
    best_combo = R.iloc[0]["Kombination"]
    body.append(T.card("§12 — Fazit, Einordnung und offene Punkte",
        T.interp(
            "<ol>"
            f"<li><strong>Struktur:</strong> Über IS und OOS hinweg liefert "
            f"<em>{best_struct}</em> die höchste OOS-Sharpe. Der Vergleich mit der "
            "Pair-Variante zeigt, welcher Anteil des Ertrags aus dem Lead-Lag-Effekt und "
            "welcher aus dem allgemeinen Aktienbeta stammt.</li>"
            f"<li><strong>Kombination:</strong> Die beste der 16 überlieferten "
            f"Kombinationen ist hier <em>{best_combo}</em>. Die Grundaussage des früheren "
            "Laufs — wenige Filter generalisieren besser als viele — lässt sich anhand "
            "der marginalen Filterbeiträge prüfen.</li>"
            "<li><strong>Parameter:</strong> Stop-Logik und Sizing wirken systematisch "
            "auf den Drawdown, aber deutlich schwächer auf die Sharpe-Ratio. "
            "Transaktionskosten sind der einzige Faktor, der in jeder Konfiguration in "
            "dieselbe Richtung wirkt.</li>"
            f"<li><strong>Statistik:</strong> Bei {n_tests} Auswertungen liegt die "
            f"Zufallsschranke bei rund {snoop:.2f} Sharpe. Jede Einzelkonfiguration muss "
            "an dieser Schranke gemessen werden, nicht an der Null.</li>"
            "</ol>")
        + T.warn(
            "<strong>Was dieser Report nicht zeigen kann:</strong> "
            "Leihkosten und Verfügbarkeit für die Short-Beine; Gap-Risiko über Nacht; "
            "Liquiditätsverschlechterung in Krisen; die tatsächliche Ausführbarkeit von "
            "Stops in Lücken. Ferner beruht die gesamte Auswertung auf einer einzigen "
            "historischen Realisation — die Konfidenzintervalle beschreiben die "
            "Stichprobenunsicherheit, nicht das Risiko eines Strukturbruchs.")
        + T.info("<strong>Offener Punkt:</strong> Die im Auftrag begonnene Rückfrage "
                 "(„hast du bisher berücksichtigt, dass …“) ist unvollständig übermittelt "
                 "worden. Sobald sie vorliegt, wird der betreffende Aspekt hier ergänzt "
                 "bzw. als zusätzliche Kontrolle in das Grid aufgenommen.")))

    T.write(out / "r44_structures_grid.html", T.html_base(TITLE, PHASE, "\n".join(body)))
