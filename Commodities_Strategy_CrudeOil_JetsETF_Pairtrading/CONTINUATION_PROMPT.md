# Continuation Prompt — Commodity & Airline Research Framework
**Stand: September 2026 | Übergabe an neuen Chat**

---

## 0. Kontext & Ziel dieses Prompts

Du arbeitest an einem laufenden quantitativen Forschungsprojekt zu Commodities und Airline-Aktien.
Dieser Prompt übergibt den vollständigen Stand inkl. Strategie-Ergebnisse, Methodik-Anforderungen und drei konkrete Folgeuntersuchungen.
Das Projekt läuft in Python (3.13), Plotly Dash, yfinance. Alle Reports werden als `.html` in `outputs/reports/` geschrieben.

---

## 1. Das Framework — Überblick

### Technischer Stack
- **Python 3.13**, venv unter `.venv`
- **Plotly 2.27.0** (CDN, `include_plotlyjs=False`), Dark-Theme: `paper_bgcolor="#161b22"`, `plot_bgcolor="#0d1117"`, Text `#e6edf3`
- **Bootstrap 5.3.0** (Accordion + Nav-Tabs, `data-bs-*` Attribute)
- **Kein sklearn** — PCA via `np.linalg.svd()`, Rolling-Sharpe vectorisiert
- **yfinance** nur Einzel-Ticker (`yf.Ticker(t).history(...)`), kein Batch-Download
- **`_read(file)`** gibt immer String-Index zurück → immer `pd.to_datetime(index, errors="coerce")` danach
- **yfinance Timezone**: immer normalisieren: `idx.tz_convert("UTC").tz_localize(None).normalize()`
- **`_html_base(title, phase, body)`** — 3 Argumente, `phase=20` für neue Reports

### Alle vorhandenen Reports (39 + Index)
| Nr | Datei | Inhalt |
|----|-------|--------|
| 1–10 | phase01–phase10 | Datenladen, EDA, Stationarität, Korrelation, Lead-Lag, Events, Kointegration, GARCH/Regime |
| 11 | pca_deep_dive | PCA-Strategie |
| 12–27 | phase11–timeseries_viewer | Bootstrap, Hypothesen, Netzwerk, Backtesting, Insights, Predictive |
| 28 | mega_strategies | Alle Strategien kombiniert |
| 29 | strategy_stress_test | Stress-Testing |
| 30 | airline_oil_report | Airline/Öl-Beziehung (29 Sektionen) |
| 31 | seasonality_report | Saisonalität |
| 32 | alpha_ideas_report | 10 Alpha-Ideen |
| 33 | combination_holdperiod_report | Haltedauer-Kombination |
| 34 | leverage_crisis_report | Leverage & Krisen |
| 35 | portfolio_simulation_report | Portfolio-Simulation |
| 36 | combination_deepdive_report | Kombinations-Deep-Dive |
| 37 | crisis_vs_nocrisis_report | Krisen vs. Nicht-Krisen (v2) |
| 38 | crisis_predictivity_report | Krisenvorhersage-Index (CPI) |
| 39 | sector_rotation_report | Sektorrotation (L1/L2 Trigger) |
| 40 | flash_crash_report | Flash-Crash-Frühwarnsystem (CSI) |
| — | index.html | Dashboard |

---

## 2. Kern-Strategien — Parameter & Ergebnisse

### 2.1 Haupt-Strategie: JETS Long mit Öl-Signal

**Universe**: JETS ETF (Airline-Basket)
**Signal-Quellen**: CL=F (WTI Crude), BZ=F (Brent), XLE, XOM, CVX

**Einstiegs-Signal (Basket)**:
- 20-Tage Rolling Mean der Renditen von (CL=F, BZ=F, XLE, XOM, CVX) > 0 **UND**
- VIX < 25
→ Long JETS

**Exit-Bedingungen**:
1. Signal dreht negativ (Signal-Exit)
2. Stop-Loss: JETS fällt > 8% unter Einstieg

**Simulations-Parameter** (`_strat_exec`):
- SL = 0.08 (8%)
- Hold-Period: signalbasiert, kein festes Maximum
- Positions-Größe: 1.0 (vollständig investiert, kein Hebel in Baseline)

**Performance-Metriken (historisch, ca. 2010–2026)**:
| Metrik | Full-Sim | Crisis-Excluded |
|--------|----------|-----------------|
| Ann. Return | ~12–16% | ~10–14% |
| Volatilität | ~18–22% | ~14–18% |
| Sharpe | ~0.65–0.85 | ~0.70–0.90 |
| Sortino | ~0.90–1.20 | ~0.95–1.30 |
| Calmar | ~0.50–0.70 | ~0.55–0.75 |
| Max. Drawdown | ~35–45% | ~20–30% |
| Win-Rate | ~55–65% | ~58–68% |
| #Trades | ~60–90 | ~50–75 |

*(Alle Werte sind approximiert aus dem crisis_vs_nocrisis Report v2. Für exakte Zahlen: `outputs/reports/crisis_vs_nocrisis_report.html` §7-Tabelle aufrufen)*

**Wichtige Beobachtung**: Crisis-Exclusion verbessert zwar die risikoadjustierten Kennzahlen deutlich, der absolute Gewinn-Vorteil ist jedoch gering — weil in Krisen gelegentlich auch starke Rebounds kommen, die man dann verpasst. Das deutet auf eine **Hedge-Schicht** als besseren Ansatz hin als Hard-Exits.

### 2.2 CPI — Composite Predictivity Index (crisis_predictivity_report)

**Komponenten** (Z-Score normiert):
```
CPI = 0.30 × VIX_z + 0.25 × CreditSpread_z + 0.20 × (−YieldCurve_z) + 0.15 × Gold_z + 0.10 × Defense_z
```
- **VIX**: ^VIX
- **Credit Spread**: -log(HYG/IEF) (High-Yield vs Treasury)
- **Yield Curve**: ^TNX - ^IRX (10Y - 3M)
- **Gold**: GLD 20d Return
- **Defense**: ITA 20d Return relativ zu SPY

**Lead-Time**: CPI hat empirisch 5–15 Tage Vorlauf vor größeren JETS-Drawdowns.
**Schwellen**: CPI > 2.0 → kritisch, 1.0–2.0 → erhöht, < 1.0 → normal

### 2.3 CSI — Crash Stress Index (flash_crash_report)

**Komponenten** (0–100 Rolling 252d Percentile Rank):
```
C1 (30%):  VIX-Level Percentile
C2 (18%):  VIX 5d-Spike Percentile
C3 (30%):  Credit Spread = -log(HYG/IEF) Percentile
C4 (10%):  |DXY 5d Return| Percentile (Safe-Haven-Nachfrage)
C5 (12%):  JETS Volume-Ratio Percentile (vol/20d_avg - 1, clipped at 0)
C6 (15%):  -(VIX9D - VIX) Percentile (Terminstruktur-Inversion, wenn verfügbar)
```
Glättung: 3-Tage Rolling Mean.
Schwellen: > 80 = kritisch, 60–80 = erhöht, 40–60 = normal, < 40 = ruhig.

**Historisch validierte Flash-Crash-Events**:
| Event | Datum | Auslöser |
|-------|-------|----------|
| China Crash | 2015-08-24 | Renminbi-Abwertung |
| Volmageddon | 2018-02-05 | VIX Short Squeeze |
| COVID Crash | 2020-03-16 | Schlimmster Tag seit 1987 |
| Inflation Shock | 2022-01-24 | Fed Pivot Fears |
| CPI Schock | 2022-09-13 | CPI-Überraschung, SPY -4.3% |

**CSI als Trading-Overlay**: Signal = Base-Signal AND CSI < 80 → geringere Drawdowns in Crashphasen.

### 2.4 Sektor-Rotations-Trigger (sector_rotation_report)

**L1-Trigger**: VIX > 20 UND SPY 20d Return < -5%
**L2-Trigger**: VIX > 25 UND XLU outperformt XLK UND GLD 20d Return > 3%

Rotation in: GLD, XLU (Utilities), TLT (Bonds)
Rotation aus: JETS, XLK (Tech), XLI (Industrials)

---

## 3. Methodische Anforderungen — unveränderlich

**Diese Anforderungen gelten für alle Folge-Untersuchungen:**

### 3.1 Stationarität zuerst
- **Vor jedem linearen Modell** (OLS, ARIMA, Korrelation) Augmented Dickey-Fuller (ADF) oder KPSS-Test
- Preisreihen sind I(1) → differenzieren oder Log-Renditen verwenden
- Bei Kointegration: VECM statt VAR
- **Immer angeben**: Teststatistik, p-Wert, kritische Werte, Schlussfolgerung

### 3.2 Statistische Korrektheit
- **Multiple Testing**: Bei n Tests Bonferroni-Korrektur (`α* = 0.05/n`) oder FDR-Kontrolle (Benjamini-Hochberg)
- **Overfitting-Kontrolle**: Walk-Forward-Validation, Out-of-Sample Split (min. 20%)
- **Rollendes Fenster**: Alle Parameter müssen auf Stabilität über Zeit getestet werden (rolling coefficient plots)
- **Konfidenzintervalle** immer angeben — Punktschätzer ohne CI sind wertlos
- **Bootstrap-CIs** für Performance-Metriken (N=1000 Resamples)

### 3.3 Explainability & Nachvollziehbarkeit
- Jeder Report-Abschnitt erklärt zuerst **Was** und **Warum** (Hypothese) bevor Ergebnisse gezeigt werden
- **Unsicherheiten explizit benennen**: Was wissen wir nicht? Welche Annahmen werden gemacht?
- Mathematische Formeln als LaTeX/Text wo sinnvoll
- Keine Black-Box-Ergebnisse ohne Herleitung

### 3.4 Reproduzierbarkeit
- **Zu Beginn jedes neuen Chats/Reports**: Erst JETS und CL=F Strategie mit den obigen Parametern reproduzieren, dann neue Analyse aufbauen
- Random Seeds setzen: `np.random.seed(42)`
- Daten-Zeitraum explizit benennen

---

## 4. Folge-Untersuchung 1: Sektor-Rotations-Analyse (Hochstatistisch)

### Ziel
Statistisch fundierte Analyse welche Sektoren als Frühindikatoren für JETS dienen, wie sich die Korrelationsstruktur in Krisen ändert, und ob systematische Rotation aus Sektoren raus/rein profitable Signale liefert.

### Ticker-Universum (nach Market-Cap-Konvergenz)

**Logik**: Nehme alle Aktien bis zur Grenze wo `market_cap(Aktie_i) / Σ market_cap(1..i) < 1%` oder Aktie < $500M Market Cap. Unterhalb dieser Schwelle ist der marginale Erklärungsbeitrag für Sektordynamik vernachlässigbar.

#### Airlines (JETS-Komponenten + Wettbewerber):
```
DAL, UAL, AAL, LUV, JBLU, ALGT, SAVE, HA, ULCC, RYAAY (Ryanair ADR),
ICAIRY (IAG ADR), AFRAF (Air France ADR), DLAKY (Lufthansa ADR),
AZUL, GOL, CPA (Copa Holdings), VLRS (Viva Aerobus)
```

#### Energie/Öl & Gas (XLE-Komponenten + relevante):
```
XOM, CVX, COP, EOG, MPC, PSX, VLO, HES, DVN, FANG, OXY, SLB, HAL, BKR,
MRO, APA, CTRA, PDCE, SM, RRC, AR, EQT, CHK, CRC, MTDR
```

#### Materials — Commodities-Teil (besonders relevant!):
```
FCX (Kupfer), NEM, GOLD (Barrick), AEM, AGI, WPM, KGC, PAAS,
VALE, RIO (ADR), BHP (ADR), AA (Aluminium), X (US Steel), NUE, STLD,
CLF, MP (Seltene Erden), SCCO (Southern Copper), CENX, CSTM
```
ETFs: GLD, SLV, GDX, GDXJ, COPX, SIL, REMX, PALL, PPLT

#### Materials — Non-Commodities (Chemie, Spezialwerkstoffe):
```
LIN, APD, ECL, SHW, PPG, DD, DOW, EMN, ALB, AMCR, AVY, IP, PKG,
CF (Dünger), MOS, NTR, FMC, IFF, RPM, H.B. Fuller (FUL)
```
ETFs: XLB (gesamt), VAW

#### Aerospace & Defense (da Airline-Lieferkette relevant):
```
BA, LMT, RTX, NOC, GD, HII, L3T, LDOS, CACI, BWXT, TDG, HEICO,
TransDigm (TDG), Curtiss-Wright (CW), Moog (MOG.A), DRS, AXON (neu)
```
ETFs: ITA, XAR, PPA

#### Alle 11 GICS-Sektor-ETFs:
```
XLK (Tech), XLV (Health), XLF (Finance), XLU (Utilities), XLP (Staples),
XLY (Discretionary), XLI (Industrials), XLE (Energy), XLB (Materials),
XLRE (Real Estate), XLC (Communication)
```

### Statistische Analysen (alle hochstatistisch):

1. **Granger-Kausalitäts-Test** (AIC-optimierte Lag-Auswahl, Bonferroni-korrigiert):
   - Welche Sektoren Granger-kausal für JETS Renditen?
   - Matrix aller Paare: N×N mit p-Wert-Heatmap

2. **Rolling DCC-GARCH** (Dynamic Conditional Correlation):
   - Zeitvariable Korrelationen zwischen JETS und jedem Sektor
   - Identifikation: Wann brechen Korrelationen zusammen/explodieren?
   - Besonders: Korrelations-Regime in Krisen vs. Normal

3. **PCA auf Sektoren** (SVD, kein sklearn):
   - Wie viele Faktoren erklären 80/90/95% der Varianz?
   - Faktor-Loadings: Welche Sektoren dominieren welchen Faktor?
   - JETS Position im Faktorraum — in welchem Faktor lebt JETS?

4. **Regime-abhängige Korrelation** (Markov-Switching oder VIX-Schwelle):
   - Regime 1: VIX < 20 (normal)
   - Regime 2: 20 ≤ VIX < 30 (stress)
   - Regime 3: VIX ≥ 30 (crisis)
   - Korrelationsmatrix separat für jedes Regime, Fisher-Z-Test auf Gleichheit

5. **Rotation-Timing-Score**:
   - Momentum-Signal: 1m/3m/6m relative Stärke vs. SPY
   - Mean-Reversion-Signal: Abweichung von 52-Wochen rolling Beta-bereinigt
   - Kombination: Welcher Mix maximiert Sharpe(JETS-Signal | Rotation-Overlay)?

6. **Materials-Split**: Rohstoff-Subsektoren separat:
   - Precious Metals (GLD, SLV, GDX) — Fluchtinvestment oder Inflationsschutz?
   - Base Metals (COPX, FCX, VALE) — Konjunkturbarometer
   - Energy Materials vs. Airline-Kosten

---

## 5. Folge-Untersuchung 2: Pandemie-Erkennungs-System (Hochstatistisch)

### Kontextuelles Wissen (Alternative Data — Hedge-Fund-Ansatz)

**Informationshierarchie** (wer erfährt es zuerst):

| Rang | Quelle | Vorlaufzeit |
|------|--------|-------------|
| 1 | Lokale Labormitarbeiter/Ärzte | Nicht observierbar |
| 2 | Lokale Gesundheitsbehörden (China CDC, ICMR) | Wochen vor WHO |
| 3 | **ProMED-mail** (`promedmail.org`, RSS) | **7–14 Tage vor WHO** |
| 4 | WHO Disease Outbreak News | Offiziell, langsam |
| 5 | Mainstream Media | Wochen bis Monate danach |

**Beobachtbare Frühwarnsignale:**

| Signal | Quelle | Latenz | Realistisch mit yfinance? |
|--------|--------|--------|--------------------------|
| Flugannullierungen in spezifischen Städten | FlightAware API | Stunden | ❌ (Paid) → Proxy via Airline-Volumen |
| Google Trends: "Fieber", "Atemwegsinfektion" (lokale Sprache) | Google Trends API | 1–3 Tage | ✅ (pytrends) |
| ProMED/HealthMap Anomalien | RSS-Feed Scraping | Stunden | ✅ (requests + feedparser) |
| ClinicalTrials.gov Notfallstudien | API | Tage | ✅ (REST API) |
| bioRxiv/medRxiv Preprint-Wellen | API | Tage | ✅ |
| Amazon-Suchanfragen Masken/Desinfektionsmittel | Keepa/JungleScout | 1–3 Tage | ❌ (Paid) → Proxy |
| Krankenhausauslastung USA | HHS Protect API | 3–7 Tage | ✅ (kostenlos) |
| Satellitenbilder Krankenhaus-Parkplätze | Planet Labs / Sentinel-2 | 1–3 Tage | ❌ (Bildverarbeitung) |
| Energieverbrauchsanomalien | Netzbetreiber | Wochen | ❌ |
| JETS/DAL/UAL Volumen-Anomalien | yfinance | Stunden | ✅ |
| VIX-Terminstruktur (VIX9D vs VIX) | yfinance | Stunden | ✅ |
| Pharma-Aktien-Anomalien (PFE, MRNA, BNTX, JNJ) | yfinance | Stunden | ✅ |
| Biotech-ETF Anomalien (IBB, XBI) | yfinance | Stunden | ✅ |
| Defense-ETF Anomalien (ITA) | yfinance | Stunden | ✅ |

**Was NICHT observierbar ist:**
- Laborquarantänen in geschlossenen Einrichtungen
- Geheimdiensterkenntnisse
- Regierungsinterne Kommunikation

### Realistisch baubare Proxy-Signale (nur mit yfinance + freien APIs)

**Pandemie Proxy Index (PPI)**:
```
PPI = w1 × VIX_spike + w2 × JETS_vol_anomaly + w3 × Pharma_momentum
    + w4 × (−Airline_options_skew) + w5 × HYG_spread_widening
```

Erweiterung mit pytrends (Google Trends, kostenlos):
```
PPI_extended = PPI + w6 × GoogleTrends("pneumonia" OR "flu symptoms", geo="CN,IN,BR")
```

### Statistische Analysen

1. **Historische Validierung** (Backtesting an bekannten Events):
   - SARS 2003, H1N1 2009, MERS 2012, Ebola 2014, COVID-19 2020
   - Für jedes Event: Wie viele Tage Vorlauf hat jeder Proxy-Indikator?
   - **Receiver Operating Characteristic (ROC)**: AUC für jeden Indikator
   - **Optimal Threshold**: Youden-Index (max Sensitivität + Spezifität - 1)

2. **Signal-Qualitäts-Test**:
   - **Precision-Recall-Kurven** (wichtiger als ROC bei unbalancierten Daten!)
   - False Positive Rate quantifizieren: Wie oft schlägt Alarm ohne Event?
   - **Cost-Benefit-Analyse**: Kosten eines False Positive (ausgestiegene Position × verpasste Rendite) vs. True Positive (verhindeter Drawdown)

3. **Anomalie-Detektion** (statistisch korrekt):
   - Alle Signale: ADF-Test → stationär? Wenn nicht → differentieren
   - **CUSUM-Test** (Cumulative Sum Control Chart) für strukturelle Brüche
   - **Isolation Forest** via numpy (ohne sklearn): basierend auf Subsample-Verfahren
   - **Extreme-Value-Theory (EVT)**: Generalized Pareto Distribution für Tail-Events

4. **Integration in Strategie (nicht Hard-Exit sondern Hedge-Schicht)**:
   - Bei PPI > Schwelle_1: Stop-Loss von 8% auf 5% verschärfen
   - Bei PPI > Schwelle_2: Positionsgröße auf 50% reduzieren
   - Bei PPI > Schwelle_3: Rotation in GLD/XLU/TLT (keine Komplettliquidation)
   - Backtest: Hedge-Schicht vs. Hard-Exit vs. Ignore → Sharpe-Vergleich mit Bootstrap-CIs

5. **Datenverfügbarkeit-Analyse** (eigener Report-Abschnitt!):
   - Welche Signale haben ausreichend Geschichte (>10 Jahre)?
   - Welche sind Survivorship-Bias-frei?
   - Welche haben Look-Ahead-Bias Risiko?

---

## 6. Folge-Untersuchung 3: Flash-Crash-Erkennungs-Optimierung

### Bestehende Basis (flash_crash_report.html)

Der bestehende CSI (Crash Stress Index) ist bereits implementiert. Die Folge-Untersuchung soll die **Strategie-Optimierung** in den Vordergrund stellen, nicht nur die Erkennung.

### Erweiterte Statistische Analysen

1. **Threshold-Optimierung** (statistisch):
   - Grid-Search über CSI-Schwelle (50, 60, 70, 80, 90)
   - Für jede Schwelle: Walk-Forward-Backtest (60-Monats-Train, 12-Monats-Test)
   - Metrik: Sharpe-Differenz (CSI-gefilterte Strategie - Base-Strategie)
   - **Wichtig**: Bonferroni-Korrektur über alle getesteten Schwellen!
   - Ergebnis: Optimale Schwelle mit CI, Stabilitäts-Plot über Zeit

2. **Komponenten-Gewichtungs-Optimierung**:
   - Aktuelle Gewichte: [30%, 18%, 30%, 10%, 12%, 15%]
   - Optimierung via Walk-Forward Grid-Search (oder Bayesian Optimization ohne sklearn)
   - Constraint: Gewichte ≥ 0 und Summe = 100%
   - Out-of-Sample Validierung zwingend!

3. **Lead-Time-Analyse** (über bestehenden Report hinaus):
   - Für jede Komponente: Wie viele Tage vor dem Crash überschreitet sie 80?
   - **Kaplan-Meier-Schätzer** für "Time-to-Crash nach Signal"
   - **Cox Proportional Hazard** (implementierbar ohne scipy.stats durch Log-Likelihood)
   - Ergebnis: Welche Komponente hat konsistentesten Vorlauf?

4. **Positionsgrößen-Anpassung basierend auf CSI**:
   ```
   position_size = base_size × (1 - CSI/100)^k
   ```
   k-Parameter optimieren (k=0: ignoriere CSI, k=1: linear, k=2: quadratisch)
   Backtest für k ∈ {0.5, 1.0, 1.5, 2.0, 2.5}

5. **Kombination CSI + CPI** (Synergie-Analyse):
   - Beide Indizes gemeinsam als Filter
   - AND-Logik vs. OR-Logik vs. gewichtetes Composite
   - Signifikanztest: Verbessert die Kombination den Sharpe gegenüber jedem Einzelindex?

6. **Geopolitisches Risiko als zusätzliche CSI-Komponente**:
   - Composite Geopolitical Risk (CGR) aus Marktdaten:
     ```
     CGR = w1 × LMT_momentum + w2 × (Brent-WTI Spread) + w3 × GLD_z + w4 × Rubel/USD_z + w5 × VIX_term_structure
     ```
   - Ticker: LMT, RTX, NOC (Rüstung), BZ=F (Brent), CL=F (WTI), GLD, RUBUSD=X
   - Validierung an: Golfkrieg 1990, 9/11 2001, Iraq 2003, Ukraine 2022
   - GDELT (frei, 15-Min-Updates): `https://api.gdeltproject.org/api/v2/summary/summary`

---

## 7. Alternative Data — Weiterführende Quellen (für beide Pandemie/Geo)

### Kostenlos & sofort nutzbar

**GDELT Project** (Geopolitisches Ereignis-Tracking):
```python
import requests
url = "https://api.gdeltproject.org/api/v2/summary/summary?QUERY=airline+oil+OPEC&MODE=timelinevolume&TIMESPAN=180d&FORMAT=json"
data = requests.get(url).json()
# Gibt Ereignis-Volumen der Keywords über Zeit zurück
```

**ProMED RSS** (Epidemiologie-Frühwarnung):
```python
import feedparser
feed = feedparser.parse("https://promedmail.org/feed/")
# Anzahl Einträge pro Woche, Keyword-Anomalien → Pandemie-Signal
```

**Google Trends via pytrends**:
```python
from pytrends.request import TrendReq
pt = TrendReq(hl='en-US', tz=360)
pt.build_payload(['pneumonia', 'flu symptoms'], geo='CN', timeframe='today 5-y')
trends = pt.interest_over_time()
```

**HHS Protect (US-Krankenhausauslastung)**:
```python
url = "https://healthdata.gov/resource/g62h-syeh.json?$limit=1000&$order=date DESC"
```

**ClinicalTrials.gov API**:
```python
url = "https://clinicaltrials.gov/api/query/study_fields?expr=AREA[ConditionSearch]+unknown+pathogen&fields=NCTId,StartDate,Condition&min_rnk=1&max_rnk=50&fmt=json"
```

### Bezahlt / aufwendiger (Hinweis für spätere Phasen):
- **Planet Labs** / Sentinel-2: Satellitenbilder für Krankenhaus-Parkplatzanalyse
- **Panjiva**: Handelsdaten für Medizinprodukt-Exporte
- **Jungle Scout / Keepa**: Amazon-Suchanfragen

---

## 8. Ablauf-Empfehlung für den nächsten Chat

### Schritt 1: Reproduktion (PFLICHT zu Beginn!)
```python
# 1. JETS + CL=F Strategie mit obigen Parametern reproduzieren
# Signal: basket(CL=F, BZ=F, XLE, XOM, CVX) 20d mean > 0 AND VIX < 25
# SL = 8%, Long-Only JETS
# Ausgabe: Performance-Tabelle mit den obigen Benchmark-Werten vergleichen
# Wenn Abweichung > 10% → Daten-Problem untersuchen, dann weitermachen
```

### Schritt 2: Sektor-Rotation (Report 41)
Neuer Report `sector_rotation_deep_report.html`, hochstatistisch:
- Granger-Test, DCC-GARCH (approximiert), PCA, Regime-Korrelation
- Alle Ticker aus Abschnitt 4 (nach Konvergenz-Logik)
- Materials gesondert: Commodities-Teil vs. Non-Commodities

### Schritt 3: Pandemie-Monitor (Report 42)
Neuer Report `pandemic_war_monitor_report.html`:
- Historische Validierung an bekannten Events
- ROC/AUC, Precision-Recall, CUSUM
- Hedge-Schicht Backtest (nicht Hard-Exit)

### Schritt 4: Flash-Crash-Optimierung (Update Report 40)
Update `flash_crash_report.html` oder neuer `flash_crash_optimization_report.html`:
- CSI Threshold-Optimierung (Walk-Forward)
- Positionsgrößen-Anpassung
- CSI + CPI Kombination
- Geopolitischer Risiko-Indikator

---

## 9. Bekannte Bugs & Lösungen (wichtig für Coding!)

```python
# BUG: add_vline mit annotation_text crasht → annotation_text entfernen, add_annotation separat
fig.add_vline(x=date_str)  # ← kein annotation_text
fig.add_annotation(x=date_str, text="Event", ...)  # ← separat

# BUG: Rolling Sharpe via .apply(lambda) ist O(n²) → vectorisiert:
sharpe = (r.rolling(w).mean() / r.rolling(w).std().replace(0, np.nan) * 252**0.5).fillna(0)

# BUG: Hex-zu-RGBA Konversion → Helper-Funktion:
def _hex_rgba(hx, a=0.12):
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

# BUG: pd.DataFrame(records).T mit mixed Types bricht Spalten-Dtypes
# → Skalare und Series getrennt speichern

# WICHTIG: yfinance Timezone immer normalisieren:
idx = df.index
if hasattr(idx, 'tz') and idx.tz is not None:
    df.index = idx.tz_convert("UTC").tz_localize(None).normalize()

# WICHTIG: Nach _read() immer:
df.index = pd.to_datetime(df.index, errors="coerce")
```

---

## 10. Erwartetes End-Ergebnis

Nach Abschluss aller drei Folge-Untersuchungen:
- **42 Reports** + `index.html` im Dashboard
- Sektor-Rotations-Analyse mit statistisch fundierter Validierung
- Pandemie-Monitor mit historischer Backtesting-Evidenz und klarer Unsicherheitsquantifizierung
- Flash-Crash-Optimierung mit Walk-Forward-validierten Parametern
- Alle drei integriert in die JETS-Strategie als **Hedge-Schichten** (nicht Hard-Exits)
- **Kernaussage**: Die Strategie wird robuster und risikoärmer, ohne den absoluten Return signifikant zu opfern

---

*Erstellt: September 2026 | Projekt: Commodity & Airline Research Framework*
*Workspace: `c:\Users\Labor\Desktop\Neuer Ordner\Commodities`*
*Python: 3.13 | Plotly: 2.27.0 | Bootstrap: 5.3.0*
