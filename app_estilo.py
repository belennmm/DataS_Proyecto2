"""
app_estilo.py  -  identidad visual de la app (estética tipo Reddit, paleta poco saturada),
con modo claro y modo oscuro.

Paleta (validada con el validador de paletas categóricas: banda de luminosidad OKLCH,
croma mínimo, separación para daltonismo protan/deutan y contraste >= 3:1, tanto sobre
el fondo claro #FFFFFF como sobre el oscuro #181C1F):
  - Par de clases: naranja Reddit suavizado (viola) vs azul periwinkle (no viola).
    Son complementarios cálido/frío, por lo que se distinguen también con daltonismo
    (se evitó el par rojo/verde).
  - Un color fijo por modelo, igual en todas las gráficas y en ambos modos
    (el color sigue a la entidad).
  - Escala secuencial de un solo tono (periwinkle) para magnitudes y divergente
    periwinkle <-> gris <-> naranja para correlaciones; en modo oscuro la escala
    secuencial invierte su ancla (lo oscuro es "poco").
"""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit import config as st_config

from paleta import CLASS_COLORS, MODEL_COLORS, ORANGE, PERI, model_color  # noqa: F401

CLASS_ORDER = ["No viola", "Viola"]
MODEL_SHORT = {"SBERT + Reg. Logística": "SBERT", "Random Forest": "RF", "SVM": "SVM",
               "Regresión Logística": "Reg. Log.", "XGBoost": "XGBoost", "Naive Bayes": "NB"}
FONT = "IBM Plex Sans, Noto Sans, Segoe UI, sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}

# ---------- tokens por modo ----------
MODOS = {
    "claro": dict(
        BG="#F3F5F8", SURFACE="#FFFFFF", SURFACE_2="#F8F9FA", BORDER="#E3E7EC", BORDER_HOVER="#C9D1D8",
        GRID="#EEF1F4", AXIS="#D5DADF", INK="#1C1C1C", INK2="#4A5A63", MUTED="#7C8A92",
        ORANGE_SOFT="#FDEBE4", ORANGE_TEXT="#A8431F", ORANGE_BORDER="#F5C4AF",
        PERI_SOFT="#E8EEFD", PERI_TEXT="#3A55A8", PERI_BORDER="#C7D4FA",
        INSIGHT_BG="#FFF7F3", INSIGHT_BORDER="#F7D7C9",
        LINE="#C9D1D8", ERR="#B7C1C7", NEUTRAL_MARK="#576F76", LEGEND_BG="rgba(255,255,255,0.85)",
        SEQ_SCALE=["#F4F7FE", "#D3DEFB", "#A9BDF6", "#7F9CF0", "#5A78D6", "#3A55A8"],
        DIV_SCALE=[[0.0, "#3A55A8"], [0.25, "#9DB3F4"], [0.5, "#EEEEEE"], [0.75, "#F3A687"], [1.0, "#C9512C"]],
    ),
    "oscuro": dict(
        BG="#0E1113", SURFACE="#181C1F", SURFACE_2="#1F2427", BORDER="#2A3236", BORDER_HOVER="#3D474D",
        GRID="#252B2F", AXIS="#3A4348", INK="#E8EBED", INK2="#B7C0C5", MUTED="#8A979E",
        ORANGE_SOFT="#3A2219", ORANGE_TEXT="#F4A07E", ORANGE_BORDER="#5C3324",
        PERI_SOFT="#1E2740", PERI_TEXT="#A9BDF6", PERI_BORDER="#2F3E66",
        INSIGHT_BG="#241A15", INSIGHT_BORDER="#4A2F23",
        LINE="#4A555B", ERR="#5E6A70", NEUTRAL_MARK="#9AAAB2", LEGEND_BG="rgba(24,28,31,0.85)",
        SEQ_SCALE=["#1C2230", "#22304F", "#2B3F75", "#3A55A8", "#4A69C9", "#5A78D6"],   # tope oscuro: texto claro legible
        DIV_SCALE=[[0.0, "#6C8EEF"], [0.25, "#3A4E86"], [0.5, "#30373B"], [0.75, "#8A4A33"], [1.0, "#E8663D"]],
    ),
}
# tema nativo de Streamlit (inputs, tablas, botones) para cada modo
TEMA_STREAMLIT = {
    "claro": {"theme.base": "light", "theme.primaryColor": ORANGE, "theme.backgroundColor": "#F3F5F8",
              "theme.secondaryBackgroundColor": "#EEF1F4", "theme.textColor": "#1C1C1C"},
    "oscuro": {"theme.base": "dark", "theme.primaryColor": ORANGE, "theme.backgroundColor": "#0E1113",
               "theme.secondaryBackgroundColor": "#22282C", "theme.textColor": "#E8EBED"},
}

MODO = "claro"
globals().update(MODOS[MODO])


def set_modo(oscuro):
    """Activa los tokens del modo y devuelve True si hubo que cambiar el tema nativo de
    Streamlit (en ese caso la app debe hacer st.rerun() para que el navegador lo reciba)."""
    global MODO
    MODO = "oscuro" if oscuro else "claro"
    globals().update(MODOS[MODO])
    pio.templates.default = f"reddit_{MODO}"
    cambio = False
    for k, v in TEMA_STREAMLIT[MODO].items():
        if st_config.get_option(k) != v:
            st_config.set_option(k, v)
            cambio = True
    return cambio


# ---------- plantillas de Plotly ----------
def _template(t):
    tpl = go.layout.Template()
    axis = dict(gridcolor=t["GRID"], linecolor=t["AXIS"], zeroline=False, ticks="",
                title=dict(font=dict(size=12, color=t["MUTED"])))
    tpl.layout = go.Layout(
        font=dict(family=FONT, size=13, color=t["INK2"]),
        title=dict(font=dict(size=15, color=t["INK"]), x=0, xanchor="left"),
        paper_bgcolor=t["SURFACE"], plot_bgcolor=t["SURFACE"],
        colorway=list(MODEL_COLORS.values()),
        margin=dict(l=10, r=10, t=40, b=10), xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="",
                    font=dict(size=12, color=t["INK2"]), itemclick="toggle",
                    itemdoubleclick="toggleothers"),
        hoverlabel=dict(bgcolor=t["SURFACE"], bordercolor=t["BORDER"],
                        font=dict(family=FONT, color=t["INK"], size=12)),
        bargap=0.28, bargroupgap=0.08,
    )
    return tpl


for _m, _t in MODOS.items():
    pio.templates[f"reddit_{_m}"] = _template(_t)
pio.templates.default = "reddit_claro"


def estilo_fig(fig, height=380, legend=True):
    fig.update_layout(height=height, showlegend=legend, legend_title_text="", template=f"reddit_{MODO}",
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=dict(l=10, r=20, t=50, b=10),
                      font_color=INK2)
    for upd in (fig.update_xaxes, fig.update_yaxes):
        upd(automargin=True, gridcolor=GRID, linecolor=AXIS, zeroline=False, title_standoff=10,
            tickfont=dict(color=INK2), title_font_color=MUTED)
    fig.update_annotations(font_color=INK2)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


# ---------- CSS ----------
def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: {FONT} !important;
}}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"] {{ display:none; }}

/* barra superior tipo Reddit */
.rd-topbar {{ display:flex; align-items:center; gap:14px; background:{SURFACE};
  border:1px solid {BORDER}; border-radius:18px; padding:14px 20px; }}
.rd-logo {{ width:44px; height:44px; border-radius:50%; background:{ORANGE}; display:flex;
  align-items:center; justify-content:center; flex-shrink:0; }}
.rd-title {{ font-size:22px; font-weight:700; color:{INK}; line-height:1.1; }}
.rd-sub {{ font-size:13px; color:{MUTED}; }}
.rd-topbar .rd-chips {{ margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; }}
div[class*="st-key-modo"] button {{ height:72px; border-radius:18px; background:{SURFACE};
  border:1px solid {BORDER}; color:{INK}; }}
div[class*="st-key-modo"] button:hover {{ border-color:{ORANGE}; color:{ORANGE}; }}

/* chips / flairs */
.rd-flair {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px;
  font-weight:600; background:{GRID}; color:{INK2}; margin-right:6px; }}
.rd-flair.orange {{ background:{ORANGE_SOFT}; color:{ORANGE_TEXT}; }}
.rd-flair.peri {{ background:{PERI_SOFT}; color:{PERI_TEXT}; }}
.rd-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;
  vertical-align:middle; }}

/* tarjetas (contenedores con key que empieza por 'card' o 'filtros') */
div[class*="st-key-card"] {{ background:{SURFACE}; border:1px solid {BORDER};
  border-radius:16px; padding:18px 20px 12px 20px; }}
div[class*="st-key-filtros"] {{ background:{SURFACE}; border:1px solid {BORDER};
  border-radius:16px; padding:12px 18px 4px 18px; }}

/* post tipo Reddit */
.rd-post {{ display:flex; background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px;
  overflow:hidden; margin-bottom:12px; transition:border-color .15s; }}
.rd-post:hover {{ border-color:{BORDER_HOVER}; }}
.rd-vote {{ width:78px; background:{SURFACE_2}; display:flex; flex-direction:column; align-items:center;
  justify-content:center; padding:10px 4px; gap:2px; flex-shrink:0; }}
.rd-vote .up {{ color:{ORANGE}; font-size:16px; line-height:1; }}
.rd-vote .down {{ color:{PERI}; font-size:16px; line-height:1; }}
.rd-vote .num {{ font-size:17px; font-weight:700; color:{INK}; }}
.rd-body {{ padding:12px 16px; }}
.rd-meta {{ font-size:12px; color:{MUTED}; margin-bottom:4px; }}
.rd-meta b {{ color:{INK2}; }}
.rd-ptitle {{ font-size:16px; font-weight:600; color:{INK}; margin:2px 0 6px 0; }}
.rd-ptext {{ font-size:14px; color:{INK2}; line-height:1.5; }}
.rd-ptext b, .rd-ptext i {{ color:{INK}; }}

/* stat tiles */
.rd-stat {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:14px 16px; }}
.rd-stat .lab {{ font-size:12px; color:{MUTED}; font-weight:500; }}
.rd-stat .val {{ font-size:28px; font-weight:700; color:{INK}; line-height:1.2; }}
.rd-stat .note {{ font-size:12px; color:{INK2}; }}

/* hallazgo */
.rd-insight {{ background:{INSIGHT_BG}; border:1px solid {INSIGHT_BORDER}; border-radius:12px;
  padding:10px 14px; font-size:14px; color:{INK2}; margin:4px 0 10px 0; }}
.rd-insight b {{ color:{INK}; }}
.rd-insight .tag {{ display:inline-block; font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; color:{ORANGE_TEXT}; background:{ORANGE_SOFT}; border-radius:999px;
  padding:1px 8px; margin-right:8px; }}

/* veredicto */
.rd-verdict {{ border-radius:16px; padding:16px 18px; border:1px solid {BORDER}; }}
.rd-verdict.viola {{ background:{ORANGE_SOFT}; border-color:{ORANGE_BORDER}; }}
.rd-verdict.noviola {{ background:{PERI_SOFT}; border-color:{PERI_BORDER}; }}
.rd-verdict .big {{ font-size:22px; font-weight:700; color:{INK}; }}
.rd-verdict .small {{ font-size:14px; color:{INK2}; }}

.rd-section {{ font-size:18px; font-weight:700; color:{INK}; margin:2px 0 2px 0; }}
.rd-section-sub {{ font-size:13px; color:{MUTED}; margin-bottom:8px; }}

/* pestañas en forma de píldora */
.stTabs [role="tablist"] {{ gap:6px; background:{SURFACE}; border:1px solid {BORDER};
  border-radius:999px; padding:5px; width:fit-content; margin-bottom:10px; }}
.stTabs [data-testid="stTab"] {{ border-radius:999px; padding:7px 18px; }}
.stTabs [data-testid="stTab"] p {{ font-weight:600; font-size:15px; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background:{ORANGE_SOFT}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] p {{ color:{ORANGE_TEXT}; }}
.stTabs .react-aria-SelectionIndicator {{ display:none; }}

/* botones redondeados */
.stButton button, .stDownloadButton button {{ border-radius:999px; font-weight:600; }}
div[data-testid="stExpander"] details {{ border-radius:12px; border-color:{BORDER}; background:{SURFACE}; }}
</style>
""", unsafe_allow_html=True)


# ---------- componentes HTML ----------
LOGO_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
<path d="M12 2.5 4.5 5.5v6c0 4.6 3.1 8.6 7.5 10 4.4-1.4 7.5-5.4 7.5-10v-6L12 2.5Z" fill="white"/>
<path d="m8.6 12.2 2.3 2.3 4.6-4.8" stroke="#E8663D" stroke-width="2.2" stroke-linecap="round"
stroke-linejoin="round"/></svg>"""


def esc(t):
    return html.escape(str(t))


def topbar(titulo, sub, chips):
    chips_html = "".join(f'<span class="rd-flair {c}">{esc(t)}</span>' for t, c in chips)
    st.markdown(f"""
<div class="rd-topbar">
  <div class="rd-logo">{LOGO_SVG}</div>
  <div><div class="rd-title">{esc(titulo)}</div><div class="rd-sub">{esc(sub)}</div></div>
  <div class="rd-chips">{chips_html}</div>
</div>""", unsafe_allow_html=True)


def stat(label, value, note=""):
    st.markdown(f'<div class="rd-stat"><div class="lab">{esc(label)}</div>'
                f'<div class="val">{esc(value)}</div><div class="note">{esc(note)}</div></div>',
                unsafe_allow_html=True)


def post(votos, flair, flair_color, titulo, texto, meta="u/equipo_proyecto2"):
    st.markdown(f"""
<div class="rd-post">
  <div class="rd-vote"><span class="up">▲</span><span class="num">{esc(votos)}</span><span class="down">▼</span></div>
  <div class="rd-body">
    <div class="rd-meta"><span class="rd-flair {flair_color}">{esc(flair)}</span> publicado por <b>{esc(meta)}</b></div>
    <div class="rd-ptitle">{esc(titulo)}</div>
    <div class="rd-ptext">{texto}</div>
  </div>
</div>""", unsafe_allow_html=True)


def hallazgo(texto_html):
    st.markdown(f'<div class="rd-insight"><span class="tag">Hallazgo</span>{texto_html}</div>',
                unsafe_allow_html=True)


def seccion(titulo, sub=""):
    st.markdown(f'<div class="rd-section">{esc(titulo)}</div>'
                + (f'<div class="rd-section-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)
