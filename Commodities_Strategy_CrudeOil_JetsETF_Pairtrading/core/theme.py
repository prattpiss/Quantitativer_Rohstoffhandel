"""HTML/Plotly-Primitiven — identisches Dark-Theme wie das Haupt-Framework."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"
MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
# Muss exakt zur installierten plotly.py passen - sonst rendern Achsen und Traces falsch.
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"

PHASE_COLOURS = {
    1: "#1f6feb", 2: "#3fb950", 3: "#d29922", 4: "#f78166", 5: "#bc8cff",
    20: "#58a6ff", 21: "#7ee787", 22: "#ffa657", 23: "#ff7b72", 24: "#d2a8ff",
}

LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="'Segoe UI',Arial,sans-serif", size=12),
    margin=dict(l=60, r=20, t=50, b=60),
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
)
AXIS = dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d")

PAL = ["#58a6ff", "#3fb950", "#d29922", "#f78166", "#bc8cff", "#39d353",
       "#ff7b72", "#ffa657", "#7ee787", "#e3b341", "#a5d6ff", "#ff9fef"]


def hex_rgba(hx: str, a: float = 0.12) -> str:
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def html_base(title: str, phase: int, body: str) -> str:
    col = PHASE_COLOURS.get(phase, "#58a6ff")
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <link href="{BOOTSTRAP_CSS}" rel="stylesheet"/>
  <script src="{PLOTLY_CDN}"></script>
  <script>
    MathJax = {{
      tex: {{ inlineMath:[['$','$'],['\\\\(','\\\\)']] }},
      options: {{ skipHtmlTags:['script','noscript','style','textarea','pre'] }}
    }};
  </script>
  <script async src="{MATHJAX_CDN}"></script>
  <style>
    body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',Arial,sans-serif; }}
    mjx-container, mjx-container * {{ color:#e6edf3 !important; fill:#e6edf3 !important; }}
    mjx-container svg path {{ fill:#e6edf3 !important; }}
    .ph-header {{ background:{col}22; border-left:5px solid {col};
                  padding:1.5rem 2rem; border-radius:8px; margin-bottom:2rem; }}
    .ph-header h1 {{ color:{col}; font-size:1.8rem; margin:0; }}
    .ph-header .sub {{ color:#8b949e; font-size:.95rem; margin-top:.4rem; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; }}
    .card-header {{ background:{col}18; border-bottom:1px solid #30363d;
                    color:{col}; font-weight:600; font-size:.95rem; }}
    .card-body {{ padding:1rem 1.25rem; }}
    table {{ color:#e6edf3 !important; font-size:.78rem; }}
    .table-dark {{ --bs-table-bg:#161b22; --bs-table-striped-bg:#1c2128;
                   --bs-table-hover-bg:#21262d; }}
    .formula-box {{ background:#1c2128; border:1px solid #30363d; border-radius:6px;
                    padding:1rem 1.5rem; margin:.5rem 0 1rem;
                    font-size:1.05rem; color:#e6edf3 !important; }}
    .formula-box * {{ color:#e6edf3 !important; }}
    .interp-box {{ background:#0e3a1a; border-left:4px solid #3fb950;
                   padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                   color:#e6edf3; margin-bottom:1rem; }}
    .warn-box  {{ background:#3a1e0e; border-left:4px solid #d29922;
                  padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                  color:#e6edf3; margin-bottom:1rem; }}
    .info-box  {{ background:#0e2a3a; border-left:4px solid #58a6ff;
                  padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                  color:#e6edf3; margin-bottom:1rem; }}
    .hypo-box  {{ background:#241a3a; border-left:4px solid #bc8cff;
                  padding:.8rem 1.2rem; border-radius:0 6px 6px 0;
                  color:#e6edf3; margin-bottom:1rem; }}
    .stat-card {{ background:#1c2128; border:1px solid #30363d; border-radius:8px;
                  padding:1rem; text-align:center; height:100%; }}
    .stat-card .val {{ font-size:1.5rem; font-weight:700; color:{col}; }}
    .stat-card .lbl {{ font-size:.75rem; color:#8b949e; }}
    .badge-ph {{ background:{col}33; color:{col}; border:1px solid {col}55;
                 padding:.25rem .7rem; border-radius:12px; font-size:.8rem; }}
    .table-responsive {{ max-height:420px; overflow-y:auto; }}
    h2,h3,h4 {{ color:#e6edf3; }}
    a {{ color:{col}; }} a:hover {{ color:{col}cc; }}
    .breadcrumb-item.active {{ color:#8b949e; }}
    .slbl {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
             color:#8b949e; margin-bottom:.4rem; }}
    .accordion-item {{ background:#161b22; border:1px solid #30363d; }}
    .accordion-button {{ background:#1c2128; color:#e6edf3; font-weight:600; }}
    .accordion-button:not(.collapsed) {{ background:{col}22; color:{col}; }}
    .accordion-button::after {{ filter:invert(1); }}
    .accordion-body {{ background:#161b22; color:#e6edf3; }}
    .nav-tabs .nav-link {{ color:#8b949e; }}
    .nav-tabs .nav-link.active {{ background:#161b22; color:{col};
                                  border-color:#30363d #30363d #161b22; }}
  </style>
</head>
<body>
<nav class="navbar" style="background:#010409;border-bottom:1px solid #30363d;padding:.7rem 1.5rem;">
  <a class="navbar-brand fw-bold" style="color:#58a6ff;" href="index.html">
    &#9992; Commodity &amp; Airline Research — Strategy Lab
  </a>
  <span class="badge-ph">Phase {phase}</span>
</nav>
<div class="container-xl py-4">
  <nav aria-label="breadcrumb" class="mb-3">
    <ol class="breadcrumb">
      <li class="breadcrumb-item"><a href="index.html">Dashboard</a></li>
      <li class="breadcrumb-item active">{title}</li>
    </ol>
  </nav>
  {body}
</div>
<script src="{BOOTSTRAP_JS}"></script>
</body></html>"""


def header(title: str, subtitle: str) -> str:
    return f'<div class="ph-header"><h1>{title}</h1><div class="sub">{subtitle}</div></div>'


def card(head: str, body_html: str) -> str:
    return (f'<div class="card mb-4"><div class="card-header">{head}</div>'
            f'<div class="card-body">{body_html}</div></div>')


def formula(latex: str, label: str = "") -> str:
    lbl = f'<div class="slbl">{label}</div>' if label else ""
    return f'<div class="formula-box">{lbl}$${latex}$$</div>'


def interp(text: str) -> str:
    return f'<div class="interp-box"><strong>&#128270; Interpretation:</strong> {text}</div>'


def warn(text: str) -> str:
    return f'<div class="warn-box"><strong>&#9888; Unsicherheit / Vorbehalt:</strong> {text}</div>'


def info(text: str) -> str:
    return f'<div class="info-box"><strong>&#8505; Was &amp; Warum:</strong> {text}</div>'


def hypo(text: str) -> str:
    return f'<div class="hypo-box"><strong>&#129518; Hypothese:</strong> {text}</div>'


def stat_row(stats: list[tuple[str, str]]) -> str:
    cols = "".join(
        f'<div class="col"><div class="stat-card">'
        f'<div class="val">{v}</div><div class="lbl">{lbl}</div></div></div>'
        for lbl, v in stats
    )
    return f'<div class="row g-3 mb-4">{cols}</div>'


def df_html(df: pd.DataFrame | None, max_rows: int = 400, index: bool = True) -> str:
    if df is None or len(df) == 0:
        return '<p class="text-muted small">Keine Daten.</p>'
    d = df.head(max_rows).copy()
    num = d.select_dtypes(include=[np.number]).columns
    for c in num:
        d[c] = d[c].round(4)
    return ('<div class="table-responsive">'
            + d.to_html(classes="table table-dark table-striped table-sm table-hover",
                        border=0, index=index, escape=False)
            + '</div>')


def div(fig: go.Figure, height: int = 420) -> str:
    fig.update_layout(height=height, **LAYOUT)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": True, "responsive": True})


def chart_card(head: str, fig: go.Figure, height: int = 420,
               interp_text: str = "", formula_tex: str = "", flabel: str = "") -> str:
    parts = []
    if formula_tex:
        parts.append(formula(formula_tex, flabel))
    parts.append(div(fig, height))
    if interp_text:
        parts.append(interp(interp_text))
    return card(head, "\n".join(parts))


def write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"  [OK] {path.name}  ({len(html) / 1024:.0f} KB)")
