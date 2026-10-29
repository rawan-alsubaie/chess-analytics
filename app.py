"""
WHITE MOVES FIRST — Lichess chess analytics (Streamlit)
Monochrome editorial design · built on the EDA notebook
"Are players using the White pieces more likely to win?"

Run:  streamlit run app.py
Data: put games.csv (Kaggle: datasnaek/chess) or clean_data.csv next to this file,
      or upload it on the start screen.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

import chess_data as cd

try:
    import chess
    import chess.svg
    HAS_CHESS = True
except ImportError:  # board replay degrades gracefully
    HAS_CHESS = False

APP_DIR = Path(__file__).parent
BRAND = "WHITE MOVES FIRST"

# ---------------------------------------------------------------------------
# Page + monochrome palette
# ---------------------------------------------------------------------------
st.set_page_config(page_title="White Moves First · Chess Analytics", page_icon="♚",
                   layout="wide", initial_sidebar_state="expanded")

INK = "#0E0E0E"        # primary black
ACCENT = INK
MID = "#55524C"
SOFT = "#B5B2AA"
PAPER = "#ECEAE4"
C_WHITE = "#F6F5F1"     # White pieces → white fill with black outline
C_BLACK = INK
C_DRAW = "#9A978F"
LINE_WHITE = "#9A978F"  # when "White" must be drawn as a line on white paper
RESULT_COLORS = {"white": C_WHITE, "black": C_BLACK, "draw": C_DRAW}
STATUS_COLORS = {"mate": INK, "resign": "#55524C", "outoftime": "#B5B2AA", "draw": "#9A978F"}
SKILL_COLORS = {cd.SKILL_ORDER[0]: "#D6D3CB", cd.SKILL_ORDER[1]: "#8F8B83", cd.SKILL_ORDER[2]: INK}
HR_COLORS = {"Higher-rated won": INK, "Draw": "#9A978F", "Lower-rated won": "#D6D3CB"}
MONO_SCALE = [[0, "#F6F5F1"], [1, INK]]
DIVERGE = [[0, "#F6F5F1"], [.5, "#B5B2AA"], [1, INK]]

pio.templates["mono"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[INK, "#8F8B83", "#B5B2AA", "#55524C", "#9A978F"],
        xaxis=dict(gridcolor="#D6D3CB", zerolinecolor="#B5B2AA", linecolor=INK, ticks="outside",
                   tickcolor=INK, showline=True),
        yaxis=dict(gridcolor="#D6D3CB", zerolinecolor="#B5B2AA", linecolor=INK, showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.01, yanchor="bottom", x=0, font=dict(size=12)),
        margin=dict(l=10, r=10, t=96, b=10),
        hoverlabel=dict(bgcolor=INK, font_color=PAPER, bordercolor=INK,
                        font_family="Inter, Helvetica, Arial, sans-serif"),
        title=dict(font=dict(family="Anton, Impact, sans-serif", size=20, color=INK), x=0, xanchor="left",
                   y=.98, yanchor="top", yref="container"),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10)),
    ),
    data=dict(
        bar=[go.Bar(marker_line_color=INK, marker_line_width=1.2)],
        histogram=[go.Histogram(marker_line_color=INK, marker_line_width=.6)],
        pie=[go.Pie(marker_line_color=INK, marker_line_width=1.5)],
        funnel=[go.Funnel(marker_line_color=INK, marker_line_width=1.2)],
    ),
)
pio.templates.default = "plotly_white+mono"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, input, textarea, button {
  font-family: 'Inter', Helvetica, Arial, sans-serif;
}
.stApp { background: #ECEAE4; }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1380px; }
h1, h2, h3, h4, h5 { font-family: 'Anton', Impact, sans-serif !important; font-weight: 400 !important;
  text-transform: uppercase; letter-spacing: .01em; color: #0E0E0E; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- editorial header ---------- */
.ed-top { display:flex; align-items:center; gap:18px; margin: 6px 0 22px; }
.ed-top .rule { flex:1; height:1.5px; background:#0E0E0E; }
.dots { width:26px; height:26px; flex:none;
  background-image: radial-gradient(#0E0E0E 1.6px, transparent 1.7px); background-size: 6.5px 6.5px; }
.ed-title { font-family:'Anton', Impact, sans-serif; text-transform:uppercase; color:#0E0E0E;
  font-size: clamp(2.6rem, 6.2vw, 5.4rem); line-height: .92; letter-spacing: .005em; margin: 0; }
.ed-row { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap: 18px 40px; }
.ed-tag { display:inline-block; background:#0E0E0E; color:#ECEAE4; padding: 9px 16px; margin-top: 18px;
  font-weight:700; letter-spacing:.08em; font-size:.86rem; text-transform:uppercase; }
.ed-sub { max-width: 430px; color:#0E0E0E; font-size:.98rem; line-height:1.5; margin: 0 0 4px; }
.ed-bottom { display:flex; align-items:center; gap:14px; margin: 26px 0 8px; }
.ed-bottom .rule { flex:1; height:1px; background:#0E0E0E; }
.lines { width:44px; height:12px; flex:none;
  background: repeating-linear-gradient(#0E0E0E 0 1.5px, transparent 1.5px 4px); }
.pixel { width:22px; height:30px; flex:none; background:
  linear-gradient(#0E0E0E,#0E0E0E) 0 0/11px 10px no-repeat,
  linear-gradient(#0E0E0E,#0E0E0E) 11px 10px/11px 10px no-repeat,
  linear-gradient(#0E0E0E,#0E0E0E) 0 20px/11px 10px no-repeat; }
.pixel.inv { background:
  linear-gradient(#ECEAE4,#ECEAE4) 0 0/11px 10px no-repeat,
  linear-gradient(#ECEAE4,#ECEAE4) 11px 10px/11px 10px no-repeat,
  linear-gradient(#ECEAE4,#ECEAE4) 0 20px/11px 10px no-repeat; }

/* ---------- KPI tiles ---------- */
.kpi { border:1.5px solid #0E0E0E; padding: 14px 16px 12px; height:100%; min-height: 124px; background:#F6F5F1;
  display:flex; flex-direction:column; justify-content:flex-start; }
.kpi .lbl { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .lbl { font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#0E0E0E;
  display:flex; justify-content:space-between; align-items:center; }
.kpi .val { font-family:'Anton', Impact, sans-serif; font-size: 2.1rem; line-height:1.15; color:#0E0E0E; }
.kpi .sub { font-size:.78rem; color:#55524C; }
.kpi.accent { background:#0E0E0E; }
.kpi.accent .lbl, .kpi.accent .val { color:#ECEAE4; }
.kpi.accent .sub { color:#B5B2AA; }
.sq { display:inline-flex; gap:3px; }
.sq i { width:8px; height:8px; background:#0E0E0E; display:inline-block; }
.sq i.o { background:transparent; outline:1.4px solid #0E0E0E; outline-offset:-1.4px; }
.kpi.accent .sq i { background:#F6F5F1; } .kpi.accent .sq i.o { background:transparent; outline-color:#ECEAE4; }

/* ---------- insights / verdict ---------- */
.insight { border-left: 4px solid #0E0E0E; padding: 10px 14px; margin-bottom: 10px; background:#F6F5F1;
  color:#0E0E0E; }
.verdict { padding: 18px 20px; margin: 4px 0 14px; font-size: 1rem; border:1.5px solid #0E0E0E; }
.verdict b.big { font-family:'Anton', Impact, sans-serif; font-weight:400; font-size:1.6rem;
  text-transform:uppercase; display:block; margin-bottom:6px; letter-spacing:.01em; }
.verdict.yes { background:#0E0E0E; color:#ECEAE4; }
.verdict.no  { background:#F6F5F1; color:#0E0E0E; }
.section { font-family:'Anton', Impact, sans-serif; text-transform:uppercase; font-size:1.7rem;
  margin: 26px 0 10px; display:flex; align-items:center; gap:14px; }
.section::after { content:""; flex:1; height:1.5px; background:#0E0E0E; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  border-radius: 0 !important; border: 1.5px solid #0E0E0E !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: .07em; font-size: .85rem !important; min-height: 46px;
  background:#F6F5F1; color:#0E0E0E !important; box-shadow:none !important; }
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {
  background:#0E0E0E !important; color:#ECEAE4 !important; }
.stButton > button[kind="primary"] { background:#0E0E0E !important; color:#ECEAE4 !important; }
.stButton > button[kind="primary"]:hover { background:#333333 !important; }

/* ---------- inputs ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput div[data-baseweb="input"] {
  border-radius: 0 !important; border-color:#0E0E0E !important; }
span[data-baseweb="tag"] { border-radius:0 !important; background:#0E0E0E !important; }
span[data-baseweb="tag"] span { color:#ECEAE4 !important; }
.stDataFrame { border:1.5px solid #0E0E0E; }
div[data-testid="stExpander"] details { border-radius:0; border:1.5px solid #0E0E0E; }
.stProgress > div > div > div > div { background:#0E0E0E; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background:#0E0E0E; border-right: 0; }
section[data-testid="stSidebar"] * { color:#ECEAE4; }
section[data-testid="stSidebar"] .side-logo { display:flex; align-items:center; gap:10px; margin: 4px 0 2px; }
section[data-testid="stSidebar"] .side-logo .k { font-size: 2.1rem; line-height:1; }
section[data-testid="stSidebar"] .side-logo .t { font-weight:800; letter-spacing:.14em; font-size:.82rem; line-height:1.15; }
section[data-testid="stSidebar"] .side-h { font-family:'Anton', Impact, sans-serif; font-size:1.25rem;
  letter-spacing:.03em; margin: 8px 0 4px; }
section[data-testid="stSidebar"] hr { border-color:#333333; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 7px 0; border-bottom:1px solid #333333;
  font-weight:600; letter-spacing:.06em; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size:.84rem; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background:#1C1C1C; border-color:#444444 !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] input { color:#ECEAE4 !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] { background:#ECEAE4 !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] span,
section[data-testid="stSidebar"] span[data-baseweb="tag"] svg { color:#0E0E0E !important; fill:#0E0E0E !important; }
section[data-testid="stSidebar"] .stButton > button { background:#0E0E0E; color:#ECEAE4 !important; border-color:#ECEAE4 !important; }
section[data-testid="stSidebar"] .stButton > button:hover { background:#ECEAE4 !important; color:#0E0E0E !important; }
section[data-testid="stSidebar"] .stProgress > div > div > div > div { background:#ECEAE4; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#1C1C1C; border-radius:0; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#8F8B83 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { color:#8F8B83; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#ECEAE4; }
section[data-testid="stSidebar"] .stButton > button { border-color:#444444 !important; }

/* ---------- board + moves ---------- */
.board-wrap svg { width:100%; height:auto; border:2px solid #0E0E0E; }
.movelist { font-family: ui-monospace, Menlo, Consolas, monospace; font-size:.9rem; line-height:1.9;
  max-height: 400px; overflow-y:auto; padding: 12px 16px; border:1.5px solid #0E0E0E; color:#0E0E0E; }
.movelist .cur { background:#0E0E0E; color:#ECEAE4; padding:0 5px; font-weight:700; }
.movelist .num { color:#9A978F; }

/* ---------- intro screens ---------- */
.st-key-splash { background:#0E0E0E; padding: 30px 36px 0 !important; color:#ECEAE4; overflow:hidden;
}
.st-key-splash .stButton > button { background:#F6F5F1 !important; color:#0E0E0E !important; border-color:#ECEAE4 !important;
  justify-content: space-between; padding: 0 22px; min-height: 58px; font-size: .95rem !important; }
.st-key-splash .stButton > button:hover { background:#DCD9D1 !important; }
.sp-logo { display:flex; align-items:center; gap:12px; color:#ECEAE4; }
.sp-logo .k { font-size: 2.6rem; line-height: 1; }
.sp-logo .t { font-weight: 800; letter-spacing: .16em; font-size: 1rem; line-height: 1.15; }
.sp-head { display:flex; justify-content:space-between; align-items:center; }
.sp-head .dots { background-image: radial-gradient(#ECEAE4 1.6px, transparent 1.7px); }
.sp-chess { font-family:'Anton', Impact, sans-serif; color:#ECEAE4; text-align:center; line-height:.86;
  font-size: clamp(6rem, 21vw, 19rem); margin: 18px 0 0; letter-spacing:.01em; }
.sp-grid { display:grid; grid-template-columns: 1fr 1fr; align-items:end; gap: 10px; }
.sp-motto { border-left: 1.5px solid #ECEAE4; padding: 6px 0 6px 22px; margin-bottom: 70px; }
.sp-motto .m { font-size: clamp(1.3rem, 2.4vw, 2rem); font-weight:500; line-height:1.18; letter-spacing:.02em; }
.sp-motto .s { margin-top: 18px; font-size: .98rem; color:#D6D3CB; line-height:1.45; }
.sp-king, .podium .p2 { background: linear-gradient(115deg, #D8D8D8 0%, #6A6A6A 22%, #0E0E0E 48%, #050505 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-stroke: 1px rgba(255,255,255,.35); }
.sp-king { font-size: clamp(11rem, 26vw, 24rem); line-height: .8; text-align:center;
  filter: drop-shadow(0 0 28px rgba(255,255,255,.22)); position: relative; z-index: 2; margin-bottom: -40px; }
.floor { height: 140px; margin: 0 -36px; position:relative;
  background:
    linear-gradient(180deg, #0E0E0E 0%, rgba(14,14,14,0) 45%),
    repeating-conic-gradient(#ECEAE4 0 25%, #333333 0 50%) 0 0/120px 120px;
  transform: perspective(380px) rotateX(48deg); transform-origin: top; opacity:.9; }
.sp-cta-space { height: 18px; }

.st-key-welcome { padding: 6px 4px 10px !important; }
.st-key-welcome .stButton > button { min-height: 56px; }
.podium { position: relative; height: 420px; overflow:hidden; }
.podium .tri { position:absolute; right:0; top:0; width:58%; height:78%;
  background: linear-gradient(to bottom left, #0E0E0E 50%, transparent 50.2%); }
.podium .blk { position:absolute; bottom:0; }
.podium .b1 { left: 10%; width: 26%; height: 90px; background: linear-gradient(90deg,#D6D3CB,#ECEAE4); border:1px solid #B5B2AA; }
.podium .b2 { left: 36%; width: 30%; height: 160px; background: linear-gradient(90deg,#0E0E0E 55%,#2C2C2C); }
.podium .b3 { left: 66%; width: 30%; height: 120px; background:#0E0E0E; }
.podium .pc { position:absolute; line-height:.8; text-align:center; }
.podium .p1 { left: 10%; width:26%; bottom: 88px; font-size: 8rem; color:#ECEAE4; -webkit-text-stroke: 2px #9A978F; }
.podium .p2 { left: 36%; width:30%; bottom: 158px; font-size: 12.5rem; z-index:3;
  filter: drop-shadow(0 0 18px rgba(255,255,255,.55)); }
.podium .p3 { left: 66%; width:30%; bottom: 118px; font-size: 8rem; color:#ECEAE4; -webkit-text-stroke: 2px #9A978F; z-index:2; }

.exp-card { display:flex; align-items:center; gap: 26px; padding: 18px 26px; border: 1.5px solid #0E0E0E;
  background:#F6F5F1; color:#0E0E0E; min-height: 150px; }
.exp-card.dark { background:#0E0E0E; color:#ECEAE4; }
.exp-card .g { font-size: 6.4rem; line-height: .9; width: 92px; text-align:center; flex:none; }
.exp-card.dark .g { color:#ECEAE4; }
.exp-card .h { font-family:'Anton', Impact, sans-serif; font-size: 1.9rem; text-transform:uppercase;
  display:flex; align-items:center; gap: 16px; line-height: 1.1; }
.exp-card .d { font-size:.95rem; margin-top: 6px; line-height:1.4; }
.exp-card .st { display:flex; gap: 22px; margin-top: 10px; font-size:.78rem; letter-spacing:.06em;
  text-transform:uppercase; font-weight:700; }
.exp-card .st span b { font-family:'Anton', Impact, sans-serif; font-weight:400; font-size:1.15rem;
  letter-spacing:0; margin-right:4px; }
.exp-card.dark .sq i { background:#F6F5F1; } .exp-card.dark .sq i.o { background:transparent; outline-color:#ECEAE4; }
[class*="st-key-pick_"] .stButton > button { min-height: 150px; font-size: 1.8rem !important; border-left: 0 !important; }
.st-key-pick_1 .stButton > button { background:#0E0E0E !important; color:#ECEAE4 !important; border-color:#0E0E0E !important; }
[class*="st-key-pick_"] { gap: 0 !important; }
/* ---------- insights summary ---------- */
.ins-card { border:1.5px solid #0E0E0E; background:#F6F5F1; height:100%; }
.ins-head { display:flex; align-items:center; gap:12px; background:#0E0E0E; color:#ECEAE4; padding: 12px 16px; }
.ins-num { font-family:'Anton', Impact, sans-serif; font-size:1.5rem; color:#8F8B83; line-height:1; }
.ins-title { font-family:'Anton', Impact, sans-serif; font-size:1.35rem; text-transform:uppercase; line-height:1.05; flex:1; }
.ins-icon { font-size:1.6rem; line-height:1; }
.ins-list { list-style:none; margin:0; padding: 4px 16px 8px; }
.ins-list li { display:flex; gap:12px; padding: 12px 0; border-bottom: 1px solid #D6D3CB; color:#0E0E0E;
  font-size:.92rem; line-height:1.5; }
.ins-list li:last-child { border-bottom:0; }
.ins-list .ix { font-family:'Anton', Impact, sans-serif; color:#9A978F; font-size:1.05rem; min-width: 22px; line-height:1.45; }
.ins-list b { font-weight:800; }

/* ---------- play CTA ---------- */
.st-key-cta { background:#0E0E0E; color:#ECEAE4; padding: 34px 38px 6px !important; margin-top: 8px; }
.st-key-cta .stButton > button { background:#F6F5F1 !important; color:#0E0E0E !important; border-color:#ECEAE4 !important;
  min-height: 58px; font-size: .95rem !important; }
.st-key-cta .stButton > button:hover { background:#DCD9D1 !important; }
.cta-grid { display:grid; grid-template-columns: 1.6fr 1fr; align-items:center; gap: 20px; }
.cta-k { font-size: 3rem; line-height: 1; color:#ECEAE4; }
.cta-t { font-family:'Anton', Impact, sans-serif; text-transform:uppercase; color:#ECEAE4;
  font-size: clamp(3rem, 7vw, 6rem); line-height:.92; margin-top: 10px; }
.cta-s { color:#ECEAE4; font-size: 1.2rem; font-weight: 500; margin-top: 18px; }
.cta-d { color:#B5B2AA; font-size: .95rem; margin-top: 10px; max-width: 560px; line-height: 1.5; }
.cta-d b { color:#ECEAE4; }
.cta-pieces { font-size: clamp(9rem, 20vw, 17rem); line-height:.85; text-align:center;
  background: linear-gradient(115deg, #D8D8D8 0%, #6A6A6A 25%, #0E0E0E 55%, #050505 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-stroke: 1px rgba(255,255,255,.35); filter: drop-shadow(0 0 26px rgba(255,255,255,.2)); }
@media (max-width: 760px) { .cta-grid { grid-template-columns: 1fr; } .cta-pieces { display:none; } }

.stButton > button p { font-size: inherit !important; font-weight: inherit !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def show(fig, **kw):
    """Plotly chart that works across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def squares(n_full, total=3):
    return '<span class="sq">' + "".join("<i></i>" if i < n_full else '<i class="o"></i>'
                                         for i in range(total)) + "</span>"


def kpi(col, label, value, sub="", accent=False, level=None):
    sq = squares(level) if level else ""
    col.markdown(f'<div class="kpi"><div class="lbl">{label}{sq}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub or "&nbsp;"}</div></div>',
                 unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def fmt_p(p):
    if pd.isna(p):
        return "—"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def need(df, n=30):
    if len(df) < n:
        st.warning(f"Only {len(df)} games match the current filters — widen them to see this analysis.")
        return False
    return True


def hero(title, subtitle, tag=None):
    """Editorial page header: rule + dot grid, giant condensed title, black tag."""
    # Drop a leading emoji from legacy titles so the condensed type stays clean
    first, _, rest = title.partition(" ")
    if rest and not any(ch.isalnum() for ch in first):
        title = rest
    t = f'<div class="ed-tag">{tag}</div>' if tag else ""
    st.markdown(f"""
      <div class="ed-top"><div class="rule"></div><div class="dots"></div></div>
      <div class="ed-row">
        <div><h1 class="ed-title">{title}</h1>{t}</div>
        <p class="ed-sub">{subtitle}</p>
      </div>
      <div class="ed-bottom"><div class="lines"></div><div class="rule"></div><div class="pixel"></div></div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning the games…")
def load(source) -> tuple[pd.DataFrame, list[dict]]:
    raw = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    return cd.clean(raw)


@st.cache_data(show_spinner=False)
def players_long(df: pd.DataFrame) -> pd.DataFrame:
    return cd.player_table(df)


@st.cache_data(show_spinner=False)
def logistic(df: pd.DataFrame):
    return cd.fit_logistic(df)


ss = st.session_state
ss.setdefault("stage", "splash")
ss.setdefault("skills", list(cd.SKILL_ORDER))


def go_to(stage):
    ss.stage = stage


def pick_level(levels):
    ss.skills = list(levels)
    ss.stage = "app"


local = next((APP_DIR / f for f in ("games.csv", "clean_data.csv") if (APP_DIR / f).exists()), None)
source = ss.get("upload_bytes") or (str(local) if local else None)
data = clean_log = None
if source is not None:
    try:
        data, clean_log = load(source)
    except ValueError as e:
        st.error(f"This file doesn't look like the Lichess dataset. {e}")
        ss.pop("upload_bytes", None)
        source = None


def uploader(where):
    f = where.file_uploader("Dataset — games.csv or clean_data.csv", type="csv", key=f"up_{ss.stage}",
                            help="Kaggle: datasnaek/chess. Or place the file next to app.py.")
    if f is not None and ss.get("upload_name") != f.name + str(f.size):
        ss.upload_bytes = f.getvalue()
        ss.upload_name = f.name + str(f.size)
        st.rerun()


HIDE_SIDEBAR = """<style>section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stExpandSidebarButton"] {display:none !important;}
.block-container {max-width: 1180px;}
.stApp {background: #ECEAE4 !important;}</style>"""

# ===========================================================================
# INTRO 1 — Splash
# ===========================================================================
if ss.stage == "splash":
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    with st.container(key="splash"):
        st.markdown(f"""
          <div class="sp-head">
            <div class="sp-logo"><span class="k">♚</span><span class="t">WHITE MOVES<br>FIRST</span></div>
            <div class="dots"></div>
          </div>
          <div class="sp-chess">CHESS</div>
          <div class="sp-grid">
            <div class="sp-motto">
              <div class="m">LOGIC.<br>FOCUS.<br>VICTORY.</div>
              <div class="s">Every move shapes the result.<br>20,000 Lichess games, analysed.</div>
              <div style="margin-top:34px"><div class="pixel inv"></div></div>
            </div>
            <div class="sp-king">♚</div>
          </div>
          <div class="floor"></div>
        """, unsafe_allow_html=True)
        c1, _ = st.columns([1, 1.25])
        c1.button("Start your journey  →", on_click=go_to, args=("welcome",), width="stretch", key="start")
        st.markdown('<div class="sp-cta-space"></div>', unsafe_allow_html=True)
    st.stop()

# ===========================================================================
# INTRO 2 — Welcome
# ===========================================================================
if ss.stage == "welcome":
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    with st.container(key="welcome"):
        st.markdown('<div class="ed-top"><div class="rule"></div><div class="dots"></div></div>',
                    unsafe_allow_html=True)
        left, right = st.columns([1, 1.05], gap="large")
        with left:
            n_games = f"{len(data):,}" if data is not None else "20,000"
            st.markdown(f"""
              <h1 class="ed-title">Welcome to<br>white moves<br>first</h1>
              <div class="ed-tag">Explore • Analyse • Predict</div>
              <p class="ed-sub" style="margin-top:22px">Sharpen your read of the game. {n_games} real Lichess
              games answer one question: <b>does moving first actually win?</b> Ratings, openings, endings,
              players — and a replay board for every game.</p>
              <div class="ed-bottom" style="margin-top:40px"><div class="lines"></div></div>
            """, unsafe_allow_html=True)
            if data is None:
                st.info("No dataset found next to app.py — upload it here to continue.")
                uploader(st)
            b1, b2 = st.columns([3.2, 1])
            b1.button("Next", type="primary", on_click=go_to, args=("choose",), width="stretch",
                      disabled=data is None, key="next")
            b2.button("→", on_click=go_to, args=("choose",), width="stretch", disabled=data is None,
                      key="next_arrow")
            st.button("←  Back", on_click=go_to, args=("splash",), key="back_splash")
        with right:
            st.markdown("""
              <div class="podium">
                <div class="tri"></div>
                <div class="blk b1"></div><div class="blk b2"></div><div class="blk b3"></div>
                <div class="pc p1">♙</div><div class="pc p2">♚</div><div class="pc p3">♙</div>
              </div>""", unsafe_allow_html=True)
    st.stop()

if data is None:  # safety: never reach the dashboard without data
    ss.stage = "welcome"
    st.rerun()

# ===========================================================================
# INTRO 3 — Choose your experience (sets the skill filter)
# ===========================================================================
if ss.stage == "choose":
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown("""
      <div class="ed-top"><div class="rule"></div><div class="dots"></div></div>
      <div class="ed-row">
        <h1 class="ed-title">Choose your<br>experience</h1>
        <p class="ed-sub">Select the level you want to explore. Every chart, test and model in the
        dashboard will follow your choice — you can change it any time from the sidebar.</p>
      </div><div style="height:26px"></div>""", unsafe_allow_html=True)

    cards = [
        ("Beginner", "♟", "Rated up to 1400. New to chess? See how the game is won at the start of the journey.", 1),
        ("Intermediate", "♞", "Rated 1400 – 1790. You know the basics — level up with the heart of the data.", 2),
        ("Professional", "♛", "Rated above 1790. For advanced players seeking the real challenge.", 3),
    ]
    for i, ((name, glyph, desc, lvl), skill) in enumerate(zip(cards, cd.SKILL_ORDER)):
        sub = data[data["skill_level"] == skill]
        vc = sub["winner"].value_counts(normalize=True)
        c1, c2 = st.columns([6, 1], gap="small")
        c1.markdown(f"""
          <div class="exp-card {'dark' if i == 1 else ''}">
            <div class="g">{glyph}</div>
            <div>
              <div class="h">{name} {squares(lvl)}</div>
              <div class="d">{desc}</div>
              <div class="st"><span><b>{len(sub):,}</b>games</span>
                <span><b>{pct(vc.get('white', 0))}</b>White wins</span>
                <span><b>{sub['turns'].median():.0f}</b>median turns</span></div>
            </div>
          </div>""", unsafe_allow_html=True)
        with c2.container(key=f"pick_{i}"):
            st.button("→", key=f"pick_btn_{i}", on_click=pick_level, args=([skill],), width="stretch")
        st.write("")

    st.markdown('<div class="ed-bottom"><div class="pixel"></div><div class="rule"></div></div>',
                unsafe_allow_html=True)
    s1, _, s2 = st.columns([1, 4, 1])
    s1.button("←  Back", on_click=go_to, args=("welcome",), width="stretch", key="back_welcome")
    s2.button("Skip  →", on_click=pick_level, args=(cd.SKILL_ORDER,), width="stretch", key="skip")
    st.stop()

# ---------------------------------------------------------------------------
# Dashboard — sidebar: navigation + filters
# ---------------------------------------------------------------------------

# Dashboard-only palette: soft warm grey instead of pure white (easier on the eyes).
# The three intro screens keep pure white via HIDE_SIDEBAR.
DASH_CSS = """<style>
.stApp { background: #ECEAE4 !important; }
.kpi, .verdict.no, .exp-card { background: #F6F5F1 !important; }
.insight { background: #F6F5F1 !important; }
.stButton > button, .stDownloadButton > button, .stLinkButton > a { background: #F6F5F1; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #F6F5F1 !important; }
.movelist { background: #F6F5F1; }
div[data-testid="stExpander"] details { background: #F6F5F1; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1.5px solid #0E0E0E; }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; background: transparent; border-radius: 0; }
.stTabs [data-baseweb="tab"] p { font-weight: 700; text-transform: uppercase; letter-spacing: .07em; font-size: .85rem; }
.stTabs [aria-selected="true"] { background: #0E0E0E; }
.stTabs [aria-selected="true"] p { color: #ECEAE4 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
</style>"""
st.markdown(DASH_CSS, unsafe_allow_html=True)

PAGES = {
    "01   Overview": "overview",
    "02   White advantage": "white",
    "03   Openings & strategy": "openings",
    "04   Game analysis": "analysis",
    "05   Play against AI": "play",
}

with st.sidebar:
    st.markdown('<div class="side-logo"><span class="k">♚</span><span class="t">WHITE MOVES<br>FIRST</span></div>',
                unsafe_allow_html=True)
    st.divider()
    page = PAGES[st.radio("Navigate", list(PAGES), label_visibility="collapsed")]
    st.divider()
    st.markdown('<div class="side-h">Filters</div>', unsafe_allow_html=True)

    skills = st.multiselect("Skill level", cd.SKILL_ORDER, key="skills")
    tgroups = st.multiselect("Time control", cd.TIME_ORDER, default=cd.TIME_ORDER)
    rated_opt = st.radio("Game type", ["All", "Rated", "Casual"], horizontal=True)
    lo, hi = int(np.floor(data["avg_rating"].min() / 10) * 10), int(np.ceil(data["avg_rating"].max() / 10) * 10)
    r_range = st.slider("Average rating", lo, hi, (lo, hi), step=10)
    statuses = st.multiselect("Ending", cd.STATUS_ORDER, default=cd.STATUS_ORDER)
    fam_all = data["opening_family"].value_counts().index.tolist()
    fams = st.multiselect("Opening family (optional)", fam_all, default=[],
                          placeholder="All openings")

mask = (data["skill_level"].isin(skills) & data["time_group"].isin(tgroups)
        & data["avg_rating"].between(*r_range) & data["victory_status"].isin(statuses))
if rated_opt != "All":
    mask &= data["rated"] == (rated_opt == "Rated")
if fams:
    mask &= data["opening_family"].isin(fams)
df = data[mask]

with st.sidebar:
    share = len(df) / len(data)
    st.progress(share, text=f"**{len(df):,}** of {len(data):,} games ({share:.0%})")
    st.button("↺  Change experience", on_click=go_to, args=("choose",), width="stretch", key="back_choose")
    with st.expander("Dataset"):
        uploader(st)
    st.caption("Built from the EDA notebook · Data: Lichess via Kaggle")

if df.empty:
    st.warning("No games match these filters.")
    st.stop()



# ===========================================================================
# Page sections (each is a function so pages can combine them / use tabs)
# ===========================================================================
def render_overview():
    vc = df["winner"].value_counts(normalize=True)
    w_, b_ = vc.get("white", 0), vc.get("black", 0)
    sk_r = pd.crosstab(df["skill_level"], df["winner"], normalize="index")
    sk_gap = ((sk_r.get("white", 0) - sk_r.get("black", 0)) if len(sk_r) else pd.Series(dtype=float))
    answer = ("Yes, but only slightly" if w_ > b_ else "No") + f": {pct(w_)} vs {pct(b_)}."
    if len(sk_gap) > 1:
        lvl = str(sk_gap.idxmax()).split(" ")[0].lower()
        answer += f" The edge is biggest for {lvl if lvl == 'intermediate' else lvl + 's'}."
    hero("♟️ Do White pieces win more?", answer, f"{len(df):,} games in view")

    c = st.columns(6)
    kpi(c[0], "Games", f"{len(df):,}", f"{pd.concat([df['white_id'], df['black_id']]).nunique():,} players")
    kpi(c[1], "White wins", pct(w_), f"{(w_ - b_) * 100:+.1f} pts vs Black")
    kpi(c[2], "Black wins", pct(b_), "moves second")
    d_ = vc.get("draw", 0)
    kpi(c[3], "Draws", pct(d_), f"1 in {round(1 / d_):,} games" if d_ else "none")
    kpi(c[4], "Avg rating", f"{df['avg_rating'].mean():,.0f}", f"median {df['avg_rating'].median():,.0f}")
    kpi(c[5], "Avg length", f"{df['turns'].mean():.0f} turns", f"median {df['turns'].median():.0f}")
    st.write("")

    left, right = st.columns([1, 1.25])
    with left:
        res = df["winner"].value_counts().reindex(cd.RESULT_ORDER).fillna(0)
        fig = go.Figure(go.Pie(labels=[s.title() for s in res.index], values=res.values, hole=.62,
                               marker=dict(colors=[RESULT_COLORS[s] for s in res.index],
                                           line=dict(color=INK, width=2)),
                               sort=False, textinfo="percent", textfont_size=14))
        fig.update_layout(title="Game results", showlegend=True, height=380,
                          annotations=[dict(text=f"<b>{len(df):,}</b><br>games", showarrow=False,
                                            font=dict(size=22, color=INK, family="Anton, Impact, sans-serif"))])
        show(fig)
    with right:
        sk = (pd.crosstab(df["skill_level"], df["winner"], normalize="index")
              .reindex(index=[s for s in cd.SKILL_ORDER if s in df["skill_level"].unique()],
                       columns=cd.RESULT_ORDER).fillna(0) * 100)
        n_by = df["skill_level"].value_counts()
        xl = [f"{str(i).split(' ')[0]}<br><span style='font-size:11px;color:#55524C'>n = {n_by.get(i, 0):,} · "
              f"gap {sk.loc[i, 'white'] - sk.loc[i, 'black']:+.1f}</span>" for i in sk.index]
        fig = go.Figure()
        for r in cd.RESULT_ORDER:
            fig.add_bar(name=r.title(), x=xl, y=sk[r], marker_color=RESULT_COLORS[r],
                        text=sk[r].map("{:.1f}%".format), textposition="outside")
        fig.update_layout(title="Result share by skill level", barmode="group", height=380,
                          yaxis_title="% of games", yaxis_range=[0, max(60, sk.values.max() + 8)])
        show(fig)

    a, b = st.columns([1.25, 1])
    with a:
        # Notebook: White win rate in each rating-difference category (rating_diff_cat)
        wr = (df.groupby("rating_diff_cat", observed=True)["white_win"].agg(["mean", "size"])
              .reindex(cd.DIFF_ORDER).dropna())
        fig = go.Figure(go.Bar(
            x=wr.index.astype(str).str.replace(" (", "<br>(", regex=False), y=wr["mean"] * 100,
            marker_color=[C_BLACK, "#55524C", "#9A978F", "#D6D3CB", "#F6F5F1", C_WHITE][:len(wr)],
            text=[f"{v:.1f}%" for v in wr["mean"] * 100], textposition="outside",
            customdata=wr["size"], hovertemplate="%{x}<br>White wins %{y:.1f}%<br>games %{customdata:,}<extra></extra>"))
        fig.add_hline(y=50, line_dash="dash", line_color=MID, annotation_text="50%")
        fig.update_layout(title="White win rate by rating difference", height=380,
                          yaxis_title="White win %", yaxis_range=[0, 100], xaxis_tickfont_size=11)
        show(fig)
    with b:
        stt = df["victory_status"].value_counts().reindex(cd.STATUS_ORDER).dropna()
        fig = go.Figure(go.Bar(x=stt.values, y=stt.index, orientation="h",
                               marker_color=[STATUS_COLORS[s] for s in stt.index],
                               text=[f"{v:,}  ({v / len(df):.0%})" for v in stt.values],
                               textposition="auto"))
        fig.update_layout(title="How games end", height=380, yaxis=dict(autorange="reversed"))
        show(fig)

    render_insights()


def render_insights():
    section("Insights summary")
    groups = cd.insights(df)
    meta = {"Game results": ("01", "\u2654\uFE0E"), "Game duration & endings": ("02", "\u23F1\uFE0E"),
            "Openings": ("03", "\u265E\uFE0E")}
    cols = st.columns(3)
    for col, (name, items) in zip(cols, groups.items()):
        num, icon = meta[name]
        lis = "".join(f'<li><span class="ix">{i + 1:02d}</span><span>{t}</span></li>' for i, t in enumerate(items))
        if not items:
            lis = '<li><span class="ix">—</span><span>Not enough games in this selection.</span></li>'
        col.markdown(f'<div class="ins-card"><div class="ins-head"><span class="ins-num">{num}</span>'
                     f'<span class="ins-title">{name}</span><span class="ins-icon">{icon}</span></div>'
                     f'<ul class="ins-list">{lis}</ul></div>', unsafe_allow_html=True)
    st.caption("Every sentence is recalculated from the games in view — with no filters it matches the "
               "notebook's Insights Summary.")


def render_cleaning():
    log = pd.DataFrame(clean_log)
    a, b = st.columns([1, 1.2])
    with a:
        fig = go.Figure(go.Funnel(y=log["step"], x=log["rows"], textinfo="value+percent initial",
                                  marker=dict(color=[INK, "#55524C", "#8F8B83", "#B5B2AA"])))
        fig.update_layout(title="Cleaning funnel", height=360)
        show(fig)
    with b:
        st.markdown("##### Cleaning log")
        table(log, hide_index=True)
        insight("Draws by timeout (winner = draw, ending = outoftime) are kept: on Lichess a flag is a "
                "draw when the opponent has no mating material.")
        insight("Lichess's own Bullet/Blitz/Rapid/Classical labels put almost every game in Rapid, so the "
                "app uses a simpler Short / Medium / Long split on base time.")


def render_white():
    hero("⚖️ Is there a first-move advantage?",
         "Rating gaps hide the colour effect, so we compare only evenly matched players and run a "
         "two-sided binomial test on decisive games (draws excluded).", "Hypothesis test")

    c1, c2 = st.columns([1, 2.2])
    with c1:
        thr = st.slider("Max rating gap counted as “even”", 10, 300, 50, step=5)
        alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
    t = cd.white_advantage_test(df, thr)
    with c2:
        if pd.isna(t["white_share"]):
            st.warning("No decisive games in this window.")
        else:
            yes = t["p_value"] < alpha
            st.markdown(
                f'<div class="verdict {"yes" if yes else "no"}">'
                f'<b class="big">{"White has a significant edge" if yes else "No significant White edge"}</b>'
                f'In {t["white_wins"] + t["black_wins"]:,} decisive games with |gap| ≤ {thr}, White won '
                f'<b>{pct(t["white_share"])}</b> (95% CI {pct(t["ci_low"])} – {pct(t["ci_high"])}). '
                f'p = <b>{fmt_p(t["p_value"])}</b> vs α = {alpha}.</div>', unsafe_allow_html=True)

    c = st.columns(5)
    kpi(c[0], "Even games", f"{t['games']:,}")
    kpi(c[1], "White wins", f"{t['white_wins']:,}")
    kpi(c[2], "Black wins", f"{t['black_wins']:,}")
    kpi(c[3], "Draws (excluded)", f"{t['draws']:,}")
    kpi(c[4], "White share", pct(t["white_share"]), "of decisive games", accent=True)
    st.write("")

    even = df[(df["rating_diff"].abs() <= thr) & df["winner"].isin(["white", "black"])]
    if not need(even):
        return
    a, b = st.columns(2)
    with a:
        sk = (even.groupby("skill_level", observed=True)["winner"].value_counts(normalize=True)
              .unstack().reindex(columns=["white", "black"]) * 100)
        fig = go.Figure()
        for r in ["white", "black"]:
            fig.add_bar(name=r.title(), x=sk.index.astype(str), y=sk[r], marker_color=RESULT_COLORS[r],
                        text=sk[r].map("{:.1f}%".format), textposition="outside")
        fig.add_hline(y=50, line_dash="dash", line_color=SOFT)
        fig.update_layout(title=f"White vs Black win rate by skill (even games, gap ≤ {thr})",
                          barmode="group", yaxis_range=[0, 65], yaxis_title="% of decisive games", height=420)
        show(fig)
    with b:
        g = (even.groupby(["skill_level", "time_group"], observed=True)["white_win"]
             .agg(["mean", "size"]).reset_index())
        g = g[g["size"] >= 20]
        if len(g):
            heat = g.pivot(index="skill_level", columns="time_group", values="mean") * 100
            size = g.pivot(index="skill_level", columns="time_group", values="size")
            fig = go.Figure(go.Heatmap(
                z=heat.values, x=heat.columns.astype(str), y=heat.index.astype(str),
                zmid=50, colorscale=DIVERGE,
                text=[[f"{v:.1f}%<br>n={int(n)}" if not pd.isna(v) else "" for v, n in zip(r, s)]
                      for r, s in zip(heat.values, size.fillna(0).values)],
                texttemplate="%{text}", colorbar=dict(title="White %")))
            fig.update_layout(title="White share of decisive even games: skill × time", height=420)
            show(fig)

    with st.expander("📋 Robustness table (notebook thresholds 25 / 50 / 100)"):
        rt = pd.DataFrame([cd.white_advantage_test(df, x) for x in (25, 50, 100, thr)]).drop_duplicates("threshold")
        rt["white_share"] = rt["white_share"].map(pct)
        rt["p_value"] = rt["p_value"].map(fmt_p)
        table(rt.drop(columns=["ci_low", "ci_high"]), hide_index=True)


def render_rating_effect():
    bw = st.slider("Bin width (rating points)", 25, 200, 50, step=25)
    d = df[df["rating_diff"].abs() <= 800].copy()
    d["bin"] = (d["rating_diff"] / bw).round() * bw
    g = d.groupby("bin").agg(n=("id", "size"), score=("white_score", "mean"),
                             win=("white_win", "mean")).reset_index()
    g = g[g["n"] >= 15]
    xs = np.linspace(-800, 800, 200)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=[cd.elo_expected(x, 0) * 100 for x in xs], name="Elo expected score",
                    line=dict(color=SOFT, dash="dash"))
    fig.add_scatter(x=g["bin"], y=g["score"] * 100, mode="markers+lines", name="Observed score",
                    marker=dict(size=np.clip(np.sqrt(g["n"]) / 1.6, 5, 22), color=ACCENT),
                    line=dict(color=ACCENT, width=1),
                    customdata=g["n"], hovertemplate="gap %{x:+}<br>score %{y:.1f}%<br>n=%{customdata}<extra></extra>")
    fig.add_scatter(x=g["bin"], y=g["win"] * 100, mode="lines", name="White win %",
                    line=dict(color=LINE_WHITE, width=1.5, dash="dot"))
    fig.update_layout(title="White's score vs rating gap (White − Black)", height=430,
                      xaxis_title="Rating difference", yaxis_title="%", yaxis_range=[0, 100])
    show(fig)


def render_openings():
    hero("📖 Which openings lead to a win?",
         "1,476 opening names grouped into families: how each one splits between White, Black "
         "and draws, and which levels play it.", "Opening explorer")

    c1, c2, c3 = st.columns(3)
    top_n = c1.slider("Openings shown", 5, 30, 12)
    min_g = c2.slider("Minimum games", 10, 500, 100, step=10)
    level = c3.radio("Group by", ["Family", "Full name"], horizontal=True)
    col = "opening_family" if level == "Family" else "opening_name"
    ost = cd.opening_stats(df, col=col, top_n=top_n, min_games=min_g)
    if ost.empty:
        st.warning("No openings meet the minimum — lower it.")
        return

    a, b = st.columns(2)
    with a:
        o = ost.sort_values("white_rate")
        fig = go.Figure()
        for r, k in [("white", "white_rate"), ("draw", "draw_rate"), ("black", "black_rate")]:
            fig.add_bar(y=o[col], x=o[k] * 100, name=r.title(), orientation="h",
                        marker_color=RESULT_COLORS[r],
                        text=(o[k] * 100).map("{:.1f}%".format) if r == "white" else None,
                        textposition="inside", insidetextanchor="end",
                        customdata=o["games"],
                        hovertemplate="%{y}<br>" + r.title() + " %{x:.1f}%<br>games %{customdata:,}<extra></extra>")
        fig.add_vline(x=50, line_dash="dash", line_color=MID)
        fig.update_layout(barmode="stack", title="Result split by opening (sorted by White win %)",
                          xaxis_title="% of games", height=max(460, 30 * len(o) + 140))
        show(fig)
    with b:
        top_fams = df["opening_family"].value_counts().head(10).index
        tg = df[df["opening_family"].isin(top_fams)]
        sh = (pd.crosstab(tg["opening_family"], tg["skill_level"], normalize="columns") * 100)
        sh = sh.reindex(index=top_fams[::-1], columns=[s for s in cd.SKILL_ORDER if s in sh.columns])
        fig = go.Figure()
        for s_ in sh.columns:
            fig.add_bar(y=sh.index, x=sh[s_], name=s_, orientation="h", marker_color=SKILL_COLORS[s_])
        fig.update_layout(barmode="group", title="Popularity by skill level (share within level)",
                          xaxis_title="% of games in level", height=max(460, 30 * len(o) + 140))
        show(fig)

    with st.expander("📋 Full opening table"):
        out = ost[[col, "games", "white_rate", "draw_rate", "black_rate", "avg_rating", "avg_turns"]].copy()
        for k in ("white_rate", "draw_rate", "black_rate"):
            out[k] = (out[k] * 100).round(1)
        table(out.round(1), hide_index=True)


def render_ratings_upsets():
    hr = df[df["higher_rated_result"] != "Equal rating"]
    hrv = hr["higher_rated_result"].value_counts(normalize=True)
    c = st.columns(4)
    kpi(c[0], "Higher-rated wins", pct(hrv.get("Higher-rated won", 0)), "stronger player wins")
    kpi(c[1], "Upsets", pct(hrv.get("Lower-rated won", 0)), "lower-rated player wins")
    kpi(c[2], "Rating correlation", f"{df['white_rating'].corr(df['black_rating']):.2f}", "White vs Black rating")
    kpi(c[3], "Corr(gap, White win)", f"{df['rating_diff'].corr(df['white_win']):.2f}", "rating gap vs result")
    st.write("")

    rates = (pd.crosstab(hr["gap_group"], hr["higher_rated_result"], normalize="index")
             .reindex(columns=list(HR_COLORS)).fillna(0) * 100)
    fig = go.Figure()
    for col in HR_COLORS:
        fig.add_bar(name=col, x=rates.index.astype(str), y=rates[col], marker_color=HR_COLORS[col],
                    text=rates[col].map("{:.0f}%".format), textposition="inside")
    fig.add_hline(y=50, line_dash="dash", line_color=MID)
    fig.update_layout(barmode="stack", title="Result by rating gap (higher-rated player's view)",
                      xaxis_title="Rating gap", yaxis_title="% of games", height=430)
    show(fig)

    a, b = st.columns([1.2, 1])
    with a:
        samp = df.sample(min(len(df), 6000), random_state=0)
        fig = px.scatter(samp, x="white_rating", y="black_rating", color="winner",
                         color_discrete_map=RESULT_COLORS, category_orders={"winner": cd.RESULT_ORDER},
                         opacity=.45, hover_data=["white_id", "black_id", "turns", "opening_family"])
        fig.update_traces(marker_size=4)
        lims = [df[["white_rating", "black_rating"]].min().min(), df[["white_rating", "black_rating"]].max().max()]
        fig.add_scatter(x=lims, y=lims, mode="lines", line=dict(color=MID, dash="dash"), name="Equal rating")
        fig.update_layout(title=f"White vs Black rating (sample of {len(samp):,})", height=480)
        show(fig)
    with b:
        num = ["turns", "white_rating", "black_rating", "rating_diff", "avg_rating", "white_win", "est_time_sec"]
        cm = df[num].corr().round(2)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#55524C"], [.5, "#F6F5F1"], [1, INK]],
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Correlation heatmap", height=480, coloraxis_showscale=False)
        show(fig)


def render_length():
    c = st.columns(4)
    kpi(c[0], "Median turns", f"{df['turns'].median():.0f}")
    kpi(c[1], "Mean turns", f"{df['turns'].mean():.1f}")
    kpi(c[2], "Marathons (>150)", f"{(df['turns'] > 150).sum():,}")
    kpi(c[3], "Longest game", f"{df['turns'].max()} turns", accent=True)
    st.write("")

    a, b = st.columns([1.3, 1])
    with a:
        split = st.radio("Colour histogram by", ["None", "Ending", "Result", "Skill level"], horizontal=True)
        colmap = {"Ending": ("victory_status", STATUS_COLORS, cd.STATUS_ORDER),
                  "Result": ("winner", RESULT_COLORS, cd.RESULT_ORDER),
                  "Skill level": ("skill_level", SKILL_COLORS, cd.SKILL_ORDER)}
        if split == "None":
            fig = px.histogram(df, x="turns", nbins=70, color_discrete_sequence=[ACCENT])
        else:
            k, cm, order = colmap[split]
            fig = px.histogram(df.assign(**{k: df[k].astype(str)}), x="turns", color=k, nbins=70,
                               color_discrete_map=cm, category_orders={k: order}, barmode="overlay",
                               opacity=.6)
        fig.add_vline(x=df["turns"].mean(), line_dash="dash", line_color=MID,
                      annotation_text=f"mean {df['turns'].mean():.0f}")
        fig.add_vline(x=df["turns"].median(), line_color=MID,
                      annotation_text=f"median {df['turns'].median():.0f}", annotation_position="bottom right")
        fig.update_layout(title="Distribution of game length", height=430, xaxis_title="Turns")
        show(fig)
    with b:
        fig = go.Figure()
        for s in cd.STATUS_ORDER:
            v = df.loc[df["victory_status"] == s, "turns"]
            if len(v):
                fig.add_box(y=v, name=s, marker_color=STATUS_COLORS[s], boxmean=True)
        fig.update_layout(title="Turns by ending", height=470, showlegend=False, yaxis_title="Turns")
        show(fig)

    a, b = st.columns(2)
    with a:
        sr = (pd.crosstab(df["skill_level"], df["victory_status"], normalize="index") * 100)
        sr = sr.reindex(columns=[s for s in cd.STATUS_ORDER if s in sr.columns])
        fig = go.Figure()
        for s in sr.columns:
            fig.add_bar(name=s, x=sr.index.astype(str), y=sr[s], marker_color=STATUS_COLORS[s],
                        text=sr[s].map("{:.1f}%".format), textposition="inside")
        fig.update_layout(barmode="stack", title="How games end by skill level", yaxis_title="%", height=420)
        show(fig)
    with b:
        fig = go.Figure()
        for s in cd.SKILL_ORDER:
            v = df.loc[df["skill_level"] == s, "turns"]
            if len(v):
                fig.add_violin(y=v, name=s.split(" ")[0], line_color=INK, fillcolor=SKILL_COLORS[s], opacity=.9, box_visible=True,
                               meanline_visible=True, points=False)
        fig.update_layout(title="Turns by skill level", height=420, showlegend=False, yaxis_title="Turns")
        show(fig)



@st.cache_data(show_spinner=False)
def play_html() -> str:
    page = (APP_DIR / "play.html").read_text(encoding="utf-8")
    engine = (APP_DIR / "chess_engine.js").read_text(encoding="utf-8")
    return page.replace("/*__ENGINE__*/", engine)


def start_play():
    ss.playing = True


def stop_play():
    ss.playing = False


def render_play():
    if not ss.get("playing"):
        vc = df["winner"].value_counts(normalize=True)
        with st.container(key="cta"):
            st.markdown(f"""
              <div class="cta-grid">
                <div>
                  <div class="cta-k">\u265F\uFE0E</div>
                  <div class="cta-t">Challenge<br>the machine</div>
                  <div class="cta-s">You've studied the game. Now it's your turn to play.</div>
                  <div class="cta-d">In {len(df):,} games White won <b>{pct(vc.get('white', 0))}</b> and Black
                    <b>{pct(vc.get('black', 0))}</b> — pick your side and test the first-move advantage yourself.</div>
                </div>
                <div class="cta-pieces">\u265E\uFE0E</div>
              </div>""", unsafe_allow_html=True)
            c1, _ = st.columns([1, 1.6])
            c1.button("Play now  →", on_click=start_play, width="stretch", key="play_now")
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        return

    top_l, top_r = st.columns([5, 1])
    top_l.markdown('<div class="ed-top" style="margin:0"><div class="rule"></div><div class="dots"></div></div>'
                   '<h1 class="ed-title" style="font-size:clamp(2rem,4vw,3.2rem);margin-top:14px">'
                   'Play against the machine</h1>', unsafe_allow_html=True)
    top_r.button("←  Back", on_click=stop_play, width="stretch", key="play_back")
    st.caption("Click a piece to see its legal moves, then click a highlighted square. "
               "Pawns promote to a queen automatically.")
    components.html(play_html(), height=760, scrolling=True)


# ===========================================================================
# Router — 6 pages
# ===========================================================================
if page == "overview":
    render_overview()
    section("Data & cleaning")
    render_cleaning()

elif page == "white":
    render_white()
    section("Effect of the rating difference")
    render_rating_effect()

elif page == "openings":
    render_openings()

elif page == "analysis":
    hero("Game analysis",
         "Who wins when ratings differ, how often the underdog pulls off an upset, and how long games "
         "last before they end in mate, resignation, timeout or a draw.", "Ratings • Upsets • Length • Endings")
    t1, t2 = st.tabs(["Ratings & upsets", "Length & endings"])
    with t1:
        render_ratings_upsets()
    with t2:
        render_length()

elif page == "play":
    render_play()
