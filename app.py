# app.py
import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from wordcloud import WordCloud

from backend.news_service import run_full_pipeline, load_news_data
from backend.nlp_service import run_nlp_pipeline
from backend.trend_service import run_milestone3
from backend.google_auth import get_google_auth_url, get_user_info, check_admin_password

REDIRECT_URI = "http://localhost:8501"

st.set_page_config(
    page_title="NewsPulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');

/* SF Pro stack — matches macOS exactly */
:root {
    --sf: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
    --mono: "SF Mono", "Fira Mono", "Cascadia Code", ui-monospace, monospace;

    --bg:          #0A0A0F;
    --bg-layer:    #111118;
    --bg-card:     #16161F;
    --bg-raised:   #1C1C27;

    --acid-lime:   #C8FF00;
    --acid-green:  #00FF88;
    --acid-yellow: #FFE500;
    --acid-orange: #FF6B00;
    --acid-pink:   #FF2D78;
    --acid-cyan:   #00E5FF;

    --text-primary: #F0F0F5;
    --text-secondary: #8888AA;
    --text-dim:     #44445A;

    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;
}

/* ── BASE ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg) !important;
    font-family: var(--sf) !important;
    color: var(--text-primary);
}

/* grain overlay */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.35;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-layer) !important;
    border-right: 1px solid rgba(200,255,0,0.1) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(200,255,0,0.2) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    width: 100% !important;
    margin-bottom: 0.4rem !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(200,255,0,0.1) !important;
    border-color: var(--acid-lime) !important;
    color: var(--acid-lime) !important;
    box-shadow: 0 0 16px rgba(200,255,0,0.15) !important;
}

/* ── HEADER ── */
.acid-header {
    background: var(--bg-layer);
    border-bottom: 1px solid rgba(200,255,0,0.12);
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2.5rem -1rem;
    position: relative;
    overflow: hidden;
}
.acid-header::before {
    content: 'NEWSPULSE';
    position: absolute;
    font-family: var(--sf);
    font-size: 9rem;
    font-weight: 700;
    color: rgba(200,255,0,0.025);
    top: -1rem; left: -1rem;
    letter-spacing: -4px;
    white-space: nowrap;
    pointer-events: none;
    user-select: none;
}
.acid-header::after {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(200,255,0,0.08) 0%, transparent 70%);
    top: -100px; right: -50px;
    border-radius: 50%;
    pointer-events: none;
}
.acid-inner { position: relative; z-index: 1; }
.acid-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(200,255,0,0.1);
    border: 1px solid rgba(200,255,0,0.3);
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--acid-lime);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.acid-tag::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--acid-lime);
    border-radius: 50%;
    animation: pulse-dot 2s ease infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.acid-title {
    font-family: var(--sf);
    font-size: 3rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.04em; line-height: 1;
    margin: 0 0 0.5rem;
}
.acid-title em {
    font-style: normal;
    color: var(--acid-lime);
    -webkit-text-stroke: 0px;
}
.acid-sub {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── SECTION HEADERS ── */
.acid-section {
    font-family: var(--sf);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--acid-lime);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-variant-numeric: tabular-nums;
    margin: 2.2rem 0 1rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.acid-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(200,255,0,0.3), transparent);
}

/* ── CARDS ── */
.acid-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.acid-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(200,255,0,0.4), transparent);
}
.acid-card:hover {
    border-color: rgba(200,255,0,0.2);
    box-shadow: 0 0 30px rgba(200,255,0,0.05), 0 8px 32px rgba(0,0,0,0.3);
}

/* ── METRIC GRID ── */
.acid-metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin-bottom: 2rem;
}
.acid-metric {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.2rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.2s,
                border-color 0.2s;
}
.acid-metric:hover {
    transform: translateY(-3px);
    border-color: rgba(200,255,0,0.3);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px rgba(200,255,0,0.06);
}
.acid-metric-glow {
    position: absolute;
    width: 80px; height: 80px;
    border-radius: 50%;
    filter: blur(30px);
    top: -20px; right: -10px;
    opacity: 0.4;
}
.acid-metric .val {
    font-family: var(--sf);
    font-size: 2.4rem; font-weight: 700;
    letter-spacing: -0.04em; line-height: 1;
    position: relative; z-index: 1;
}
.acid-metric .lbl {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
    position: relative; z-index: 1;
}

/* ── ARTICLE CARDS ── */
.acid-article {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.7rem;
    transition: transform 0.2s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.2s,
                border-color 0.2s;
    position: relative; overflow: hidden;
}
.acid-article::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: transparent;
    transition: background 0.2s;
    border-radius: 0 3px 3px 0;
}
.acid-article:hover {
    transform: translateX(4px);
    border-color: rgba(200,255,0,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), -4px 0 20px rgba(200,255,0,0.05);
}
.acid-article:hover::before {
    background: var(--acid-lime);
}
.acid-title-text {
    font-family: var(--sf);
    font-size: 0.93rem; font-weight: 500;
    color: var(--text-primary);
    line-height: 1.45; margin-bottom: 0.45rem;
    letter-spacing: -0.01em;
}
.acid-meta {
    font-family: var(--mono);
    font-size: 0.65rem; color: var(--text-secondary);
    display: flex; gap: 1.2rem; flex-wrap: wrap;
    margin-bottom: 0.45rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.acid-desc {
    font-size: 0.82rem; color: var(--text-secondary);
    line-height: 1.55; margin-bottom: 0.55rem;
    font-family: var(--sf);
}
.acid-link a {
    font-family: var(--mono);
    font-size: 0.67rem; color: var(--acid-lime);
    text-decoration: none; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
    transition: opacity 0.15s;
}
.acid-link a:hover { opacity: 0.7; text-decoration: underline; }

/* ── BADGES ── */
.acid-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-family: var(--mono);
    font-size: 0.6rem; font-weight: 500;
    padding: 0.12rem 0.55rem;
    border-radius: 100px;
    text-transform: uppercase; letter-spacing: 0.06em;
    border: 1px solid;
}
.acid-badge-pos { color: var(--acid-green);  border-color: rgba(0,255,136,0.3);  background: rgba(0,255,136,0.08);  }
.acid-badge-neg { color: var(--acid-pink);   border-color: rgba(255,45,120,0.3);  background: rgba(255,45,120,0.08);  }
.acid-badge-neu { color: var(--acid-yellow); border-color: rgba(255,229,0,0.3);   background: rgba(255,229,0,0.08);   }

/* ── KEYWORD PILLS ── */
.acid-pill {
    display: inline-flex; align-items: center;
    background: var(--bg-raised);
    border: 1px solid rgba(200,255,0,0.2);
    border-radius: 100px;
    padding: 0.3rem 0.9rem; margin: 0.2rem;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--acid-lime);
    text-transform: uppercase; letter-spacing: 0.05em;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
    cursor: default;
}
.acid-pill:hover {
    background: rgba(200,255,0,0.12);
    border-color: var(--acid-lime);
    box-shadow: 0 0 16px rgba(200,255,0,0.2);
    transform: translateY(-2px);
}
.acid-pill-count {
    color: var(--text-dim);
    margin-left: 0.3rem;
    font-size: 0.58rem;
}

/* ── TOPIC BADGE ── */
.acid-topic-tag {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 100px;
    padding: 0.18rem 0.65rem;
    font-family: var(--mono);
    font-size: 0.6rem; color: var(--acid-cyan);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.acid-topic-row {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.acid-topic-row:last-child { border-bottom: none; }
.acid-topic-kw {
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-family: var(--sf);
    line-height: 1.5;
}

/* ── ADMIN BADGE ── */
.acid-admin-badge {
    display: inline-flex; align-items: center;
    background: rgba(200,255,0,0.12);
    border: 1px solid rgba(200,255,0,0.3);
    border-radius: 100px;
    padding: 0.1rem 0.5rem;
    font-family: var(--mono);
    font-size: 0.58rem; color: var(--acid-lime);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-left: 0.4rem; vertical-align: middle;
}

/* ── PROGRESS ── */
@keyframes acid-flow {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
.acid-progress-wrap {
    background: var(--bg-card);
    border: 1px solid rgba(200,255,0,0.15);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
}
.acid-progress-label {
    font-family: var(--mono);
    font-size: 0.67rem; color: var(--acid-lime);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    display: flex; justify-content: space-between;
}
.acid-progress-track {
    background: var(--bg-raised);
    border-radius: 100px;
    height: 6px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.acid-progress-fill {
    height: 100%;
    background: linear-gradient(90deg,
        var(--acid-lime), var(--acid-green), var(--acid-cyan),
        var(--acid-lime), var(--acid-green));
    background-size: 200% 100%;
    animation: acid-flow 1.5s linear infinite;
    border-radius: 100px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 0 12px rgba(200,255,0,0.5);
}
.acid-steps {
    display: flex; gap: 0.5rem; margin-top: 0.9rem; flex-wrap: wrap;
}
.acid-step {
    font-family: var(--mono);
    font-size: 0.62rem;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    border: 1px solid;
    text-transform: uppercase; letter-spacing: 0.05em;
    transition: all 0.2s;
}
.acid-step-done    { border-color: rgba(0,255,136,0.4);  color: var(--acid-green);  background: rgba(0,255,136,0.08); }
.acid-step-active  { border-color: var(--acid-lime);      color: var(--bg);          background: var(--acid-lime); font-weight: 600; }
.acid-step-pending { border-color: rgba(255,255,255,0.1); color: var(--text-dim);    background: transparent; }

/* ── SKELETON ── */
@keyframes acid-shimmer {
    0%   { background-position: -700px 0; }
    100% { background-position:  700px 0; }
}
.acid-skeleton {
    background: linear-gradient(90deg,
        var(--bg-card) 25%,
        var(--bg-raised) 50%,
        var(--bg-card) 75%);
    background-size: 700px 100%;
    animation: acid-shimmer 1.6s infinite linear;
    border-radius: var(--radius-lg);
    border: 1px solid rgba(255,255,255,0.04);
}
.acid-skel-card { height: 108px; margin-bottom: 0.7rem; }

/* ── EMPTY STATE ── */
.acid-empty {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius-xl);
    padding: 4rem 2rem; text-align: center; margin: 2rem 0;
}
.acid-empty-icon { font-size: 3rem; display: block; margin-bottom: 1rem; }
.acid-empty-title {
    font-family: var(--sf);
    font-size: 1.4rem; font-weight: 600;
    letter-spacing: -0.03em; color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.acid-empty-sub {
    font-family: var(--mono);
    font-size: 0.68rem; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── ADMIN CTRL CARDS ── */
.acid-ctrl {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius-lg);
    padding: 1.4rem;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.acid-ctrl::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(200,255,0,0.3), transparent);
}
.acid-ctrl:hover {
    transform: translateY(-2px);
    border-color: rgba(200,255,0,0.2);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.acid-ctrl-title {
    font-family: var(--sf);
    font-size: 0.95rem; font-weight: 600;
    letter-spacing: -0.02em; color: var(--text-primary);
    margin-bottom: 0.35rem;
}
.acid-ctrl-desc {
    font-family: var(--mono);
    font-size: 0.63rem; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 1rem;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
[data-testid="stButton"] > button {
    background: rgba(200,255,0,0.1) !important;
    color: var(--acid-lime) !important;
    border: 1px solid rgba(200,255,0,0.3) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 0 20px rgba(200,255,0,0.06) !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(200,255,0,0.18) !important;
    border-color: var(--acid-lime) !important;
    box-shadow: 0 0 28px rgba(200,255,0,0.2) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stTextInput"] input {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(200,255,0,0.4) !important;
    box-shadow: 0 0 0 3px rgba(200,255,0,0.08) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
}
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: none !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--sf) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    color: var(--text-primary) !important;
}
[data-testid="stNumberInput"] input {
    background: var(--bg-raised) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
    color: var(--text-primary) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--radius-lg) !important;
}
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--sf) !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] label {
    color: var(--text-secondary) !important;
    font-family: var(--sf) !important;
    font-size: 0.82rem !important;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 2rem 0 !important; }

/* ── SIDEBAR INPUT ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB ACID DARK ──
matplotlib.rcParams.update({
    "figure.facecolor": "#16161F",
    "axes.facecolor":   "#1C1C27",
    "axes.edgecolor":   "none",
    "axes.labelcolor":  "#8888AA",
    "xtick.color":      "#8888AA",
    "ytick.color":      "#8888AA",
    "text.color":       "#F0F0F5",
    "grid.color":       "#1C1C27",
    "grid.linestyle":   "--",
    "grid.alpha":       1.0,
    "font.family":      "DejaVu Sans",
})

# ── SESSION STATE ──
for key, default in [
    ("user", None), ("is_admin", False), ("page", "user"),
    ("nlp_result", None), ("trend_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── HELPERS ──
def show_progress(steps, current_step, label=""):
    pct = int((current_step / len(steps)) * 100)
    steps_html = ""
    for i, s in enumerate(steps):
        if i < current_step:
            cls, icon = "acid-step-done", "✓"
        elif i == current_step:
            cls, icon = "acid-step-active", "▶"
        else:
            cls, icon = "acid-step-pending", "○"
        steps_html += f'<span class="acid-step {cls}">{icon} {s}</span>'
    st.markdown(f"""
    <div class="acid-progress-wrap">
        <div class="acid-progress-label">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div class="acid-progress-track">
            <div class="acid-progress-fill" style="width:{pct}%"></div>
        </div>
        <div class="acid-steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)

def show_skeletons(n=5):
    for _ in range(n):
        st.markdown('<div class="acid-skel-card acid-skeleton"></div>', unsafe_allow_html=True)

def header(tag, title_main, title_accent, subtitle):
    st.markdown(f"""
    <div class="acid-header">
        <div class="acid-inner">
            <div class="acid-tag">{tag}</div>
            <div class="acid-title">{title_main}<em>{title_accent}</em></div>
            <div class="acid-sub">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="acid-section">{label}</div>', unsafe_allow_html=True)

# ── GOOGLE AUTH ──
query_params = st.query_params
if "code" in query_params and st.session_state.user is None:
    code = query_params.get("code")
    if isinstance(code, list): code = code[0]
    ph = st.empty()
    with ph.container():
        show_progress(["Connect","Verify","Authorize"], 1, "AUTHENTICATING")
    user_info = get_user_info(code, REDIRECT_URI)
    if user_info and "email" in user_info:
        st.session_state.user = user_info["email"]
        ph.empty()
        st.query_params.clear()
        st.rerun()
    else:
        ph.empty()
        st.error("Google login failed.")
        st.query_params.clear()
        st.stop()

# ── LOGIN ──
if not st.session_state.user:
    header("Live · AI-Powered · Real-Time", "News", "Pulse", "Global Trend Analyzer · Real-Time Sentiment · ML Classification")
    _, cc, _ = st.columns([2, 2, 2])
    auth_url  = get_google_auth_url(REDIRECT_URI)
    with cc:
        st.markdown(f"""
        <div style="text-align:center; margin-top:3rem;">
            <a href="{auth_url}" target="_self" style="text-decoration:none;">
            <button style="
                padding:14px 32px;
                font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Helvetica Neue',sans-serif;
                font-size:0.88rem; font-weight:500; letter-spacing:-0.01em;
                background:rgba(200,255,0,0.1); color:#C8FF00;
                border:1px solid rgba(200,255,0,0.4); border-radius:14px;
                box-shadow:0 0 28px rgba(200,255,0,0.15);
                cursor:pointer; transition:all 0.2s;
            ">⬥ &nbsp;Continue with Google</button></a>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── SIDEBAR ──
admin_badge = '<span class="acid-admin-badge">admin</span>' if st.session_state.is_admin else ""
st.sidebar.markdown(f"""
<div style="background:rgba(200,255,0,0.04); border:1px solid rgba(200,255,0,0.12);
            border-radius:14px; padding:0.9rem 1rem; margin-bottom:1.2rem;">
    <div style="font-family:var(--mono,'SF Mono',monospace); font-size:0.58rem;
                color:#888; text-transform:uppercase; letter-spacing:0.1em;">
        Signed in
    </div>
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;
                font-size:0.85rem; font-weight:500; color:#F0F0F5;
                margin-top:0.3rem; word-break:break-all; letter-spacing:-0.01em;">
        {st.session_state.user}{admin_badge}
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-family:'SF Mono','Fira Mono',monospace; font-size:0.6rem;
            color:#C8FF00; text-transform:uppercase; letter-spacing:0.1em;
            margin-bottom:0.6rem; opacity:0.8;">
    ◈ Navigate
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("◎  News Feed"):
    st.session_state.page = "user"; st.rerun()
if st.session_state.is_admin:
    if st.sidebar.button("◈  Admin Dashboard"):
        st.session_state.page = "admin"; st.rerun()
else:
    st.sidebar.markdown("""
    <div style="font-family:'SF Mono',monospace; font-size:0.6rem;
                color:#C8FF00; text-transform:uppercase; letter-spacing:0.1em;
                margin:1rem 0 0.6rem; opacity:0.8;">◈ Admin</div>
    """, unsafe_allow_html=True)
    admin_pw = st.sidebar.text_input("Password", type="password", key="admin_pw_input")
    if st.sidebar.button("⊕  Unlock Admin"):
        if check_admin_password(admin_pw):
            st.session_state.is_admin = True
            st.session_state.page = "admin"
            st.sidebar.success("✓ Access granted")
            st.rerun()
        else:
            st.sidebar.error("✗ Incorrect password")

st.sidebar.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
if st.sidebar.button("⊗  Sign Out"):
    for k in ["user","is_admin","page","nlp_result","trend_result"]:
        st.session_state[k] = {"page":"user","is_admin":False}.get(k)
    st.rerun()


# ══════════════════════════════
# USER PAGE
# ══════════════════════════════
if st.session_state.page == "user":
    header("Live Feed · AI Analysis", "News", "Pulse", "Real-Time Sentiment Analysis · ML-Powered Trend Detection")

    df = load_news_data(cleaned=True)

    if df is None or df.empty:
        st.markdown("""
        <div class="acid-empty">
            <span class="acid-empty-icon">📡</span>
            <div class="acid-empty-title">No Articles Yet</div>
            <div class="acid-empty-sub">Ask the admin to fetch the latest news</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    for col in ["URL","Date","Topic","Source","Description","sentiment_label"]:
        if col not in df.columns: df[col] = ""
    df["sentiment_label"] = df["sentiment_label"].fillna("Neutral")
    df["URL"]   = df["URL"].fillna("")
    df["Date"]  = df["Date"].fillna("")
    df["Topic"] = df["Topic"].fillna("general")

    # metrics
    sent_df = df.copy()
    for sp in ["data/milestone3_news.csv","data/processed_news.csv"]:
        if os.path.exists(sp):
            try:
                tmp = pd.read_csv(sp)
                if "sentiment_label" in tmp.columns and not tmp.empty:
                    sent_df = tmp; break
            except: pass
    sent_df["sentiment_label"] = sent_df["sentiment_label"].fillna("Neutral")

    total   = len(df)
    sources = df["Source"].nunique()
    pos     = (sent_df["sentiment_label"] == "Positive").sum()
    neg     = (sent_df["sentiment_label"] == "Negative").sum()

    section("◎ Overview")
    st.markdown(f"""
    <div class="acid-metric-grid">
        <div class="acid-metric">
            <div class="acid-metric-glow" style="background:var(--acid-lime);"></div>
            <div class="val" style="color:var(--acid-lime);">{total}</div>
            <div class="lbl">Total Articles</div>
        </div>
        <div class="acid-metric">
            <div class="acid-metric-glow" style="background:var(--acid-cyan);"></div>
            <div class="val" style="color:var(--acid-cyan);">{sources}</div>
            <div class="lbl">News Sources</div>
        </div>
        <div class="acid-metric">
            <div class="acid-metric-glow" style="background:var(--acid-green);"></div>
            <div class="val" style="color:var(--acid-green);">{pos}</div>
            <div class="lbl">Positive</div>
        </div>
        <div class="acid-metric">
            <div class="acid-metric-glow" style="background:var(--acid-pink);"></div>
            <div class="val" style="color:var(--acid-pink);">{neg}</div>
            <div class="lbl">Negative</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # keywords
    if os.path.exists("data/processed_news.csv"):
        df_p = pd.read_csv("data/processed_news.csv")
        if "processed_text" in df_p.columns:
            section("◎ Trending Keywords")
            words = " ".join(df_p["processed_text"].astype(str)).split()
            kw_html = "".join([
                f'<span class="acid-pill">{w}<span class="acid-pill-count">({c})</span></span>'
                for w, c in Counter(words).most_common(15)
            ])
            st.markdown(f"<div style='margin-bottom:1.5rem;'>{kw_html}</div>", unsafe_allow_html=True)

    # sentiment charts
    section("◎ Sentiment Overview")
    c1, c2 = st.columns(2)

    sent_counts = sent_df["sentiment_label"].value_counts()
    s_labels = sent_counts.index.tolist()
    s_values = sent_counts.values.tolist()
    acid_colors = {"Positive":"#00FF88","Negative":"#FF2D78","Neutral":"#FFE500"}
    pie_colors  = [acid_colors.get(l,"#8888AA") for l in s_labels]

    with c1:
        st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
        fig_p, ax_p = plt.subplots(figsize=(4,3))
        wedges, texts, at = ax_p.pie(
            s_values, labels=s_labels, autopct="%1.1f%%",
            colors=pie_colors, startangle=140,
            wedgeprops={"edgecolor":"#0A0A0F","linewidth":2}
        )
        for t in texts+at:
            t.set_color("#F0F0F5"); t.set_fontsize(9)
        fig_p.patch.set_facecolor("#16161F")
        st.pyplot(fig_p)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
        fig_b, ax_b = plt.subplots(figsize=(4,3))
        bars = ax_b.bar(s_labels, s_values,
                        color=[acid_colors.get(l,"#8888AA") for l in s_labels],
                        edgecolor="#0A0A0F", linewidth=1.5, width=0.5)
        ax_b.yaxis.grid(True, alpha=0.3)
        ax_b.set_axisbelow(True)
        for bar in bars:
            ax_b.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
                      str(int(bar.get_height())),
                      ha='center', va='bottom', fontsize=9, color="#F0F0F5", fontweight='bold')
        fig_b.tight_layout()
        st.pyplot(fig_b)
        st.markdown("</div>", unsafe_allow_html=True)

    # article feed
    section("◎ Article Feed")
    f1, f2, f3 = st.columns([2,2,1])
    with f1: search_q = st.text_input("Search", placeholder="Search articles...")
    with f2:
        topics_av = ["All"] + sorted(df["Topic"].dropna().unique().tolist())
        topic_f   = st.selectbox("Topic", topics_av)
    with f3:
        sent_f = st.selectbox("Sentiment", ["All","Positive","Negative","Neutral"])

    fdf = df.copy()
    if search_q:
        mask = (fdf["Title"].str.contains(search_q,case=False,na=False) |
                fdf["Description"].str.contains(search_q,case=False,na=False))
        fdf = fdf[mask]
    if topic_f != "All": fdf = fdf[fdf["Topic"]==topic_f]
    if sent_f  != "All": fdf = fdf[fdf["sentiment_label"]==sent_f]

    st.markdown(
        f'<div style="font-family:var(--mono,monospace);font-size:0.65rem;'
        f'color:#888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem;">'
        f'◎ {len(fdf)} results</div>',
        unsafe_allow_html=True
    )

    if "articles_loaded" not in st.session_state:
        ph = st.empty()
        with ph.container(): show_skeletons(5)
        time.sleep(0.4)
        ph.empty()
        st.session_state["articles_loaded"] = True

    PAGE_SZ     = 20
    total_pages = max(1, (len(fdf)-1)//PAGE_SZ + 1)
    page_num    = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    page_df     = fdf.iloc[(page_num-1)*PAGE_SZ : page_num*PAGE_SZ]

    if page_df.empty:
        st.markdown("""
        <div class="acid-empty">
            <span class="acid-empty-icon">◎</span>
            <div class="acid-empty-title">No Results</div>
            <div class="acid-empty-sub">Try adjusting your search or filters</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for _, row in page_df.iterrows():
            title  = str(row.get("Title","No title"))
            desc   = str(row.get("Description",""))
            source = str(row.get("Source","Unknown"))
            date   = str(row.get("Date",""))
            url    = str(row.get("URL",""))
            topic  = str(row.get("Topic","")).upper()
            sent   = str(row.get("sentiment_label","Neutral"))
            badge  = {"Positive":"acid-badge-pos","Negative":"acid-badge-neg"}.get(sent,"acid-badge-neu")
            link   = f'<div class="acid-link"><a href="{url}" target="_blank">↗ Read Article</a></div>' if url and url!="nan" else ""
            desc_h = f'<div class="acid-desc">{desc[:180]}{"..." if len(desc)>180 else ""}</div>' if desc and desc not in("nan","") else ""
            st.markdown(f"""
            <div class="acid-article">
                <div class="acid-title-text">{title}</div>
                <div class="acid-meta">
                    <span>{source}</span><span>{date}</span><span>{topic}</span>
                    <span class="acid-badge {badge}">{sent}</span>
                </div>
                {desc_h}{link}
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:var(--mono,monospace);text-align:center;'
        f'color:#888;font-size:0.65rem;margin-top:1rem;text-transform:uppercase;letter-spacing:0.08em;">'
        f'Page {page_num} / {total_pages}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════
# ADMIN PAGE
# ══════════════════════════════
elif st.session_state.page == "admin" and st.session_state.is_admin:
    header("Admin · Restricted", "Pulse", "Admin", "System Control · Analytics · Pipeline Management")

    section("◈ System Controls")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='acid-ctrl'>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-title'>◎ Fetch News</div>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-desc'>Pull articles from NewsAPI</div>", unsafe_allow_html=True)
        query = st.text_input("Topic", "technology", key="admin_query")
        if st.button("▶ FETCH"):
            ph = st.empty()
            steps = ["Clear","Fetch","Clean","Save"]
            with ph.container(): show_progress(steps, 0, "FETCHING")
            time.sleep(0.4)
            with ph.container(): show_progress(steps, 1, "FETCHING")
            result = run_full_pipeline(query)
            with ph.container(): show_progress(steps, 4, "FETCHING")
            time.sleep(0.3); ph.empty()
            st.success(f"✓ {result['fetched']} articles fetched")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='acid-ctrl'>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-title'>◎ NLP Pipeline</div>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-desc'>Tokenize · TF-IDF · Sentiment · LDA</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ RUN NLP"):
            ph = st.empty()
            steps = ["Load","Tokenize","TF-IDF","Sentiment","Save"]
            with ph.container(): show_progress(steps, 0, "NLP")
            time.sleep(0.4)
            with ph.container(): show_progress(steps, 2, "NLP")
            nlp_r = run_nlp_pipeline()
            with ph.container(): show_progress(steps, 5, "NLP")
            time.sleep(0.3); ph.empty()
            if "error" in nlp_r:
                st.error(nlp_r["error"])
            else:
                ml_info = f"ML: {nlp_r.get('ml_accuracy')}%" if nlp_r.get("ml_trained") else "VADER"
                st.success(f"✓ {nlp_r['records_processed']} records · {ml_info}")
                st.session_state.nlp_result = nlp_r
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='acid-ctrl'>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-title'>◎ Trend Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='acid-ctrl-desc'>Freq · TF-IDF · Model Eval</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ RUN TRENDS"):
            ph = st.empty()
            steps = ["Load","Dedupe","TF-IDF","Model","Save"]
            with ph.container(): show_progress(steps, 0, "TRENDS")
            time.sleep(0.4)
            with ph.container(): show_progress(steps, 2, "TRENDS")
            tr = run_milestone3()
            with ph.container(): show_progress(steps, 5, "TRENDS")
            time.sleep(0.3); ph.empty()
            if "error" in tr:
                st.error(tr["error"])
            else:
                st.success(f"✓ {tr['final_records']} clean records")
                st.session_state.trend_result = tr
        st.markdown("</div>", unsafe_allow_html=True)

    # NLP Results
    if st.session_state.nlp_result:
        nr = st.session_state.nlp_result
        st.markdown("<hr>", unsafe_allow_html=True)
        section("◈ NLP Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Records Processed", nr.get("records_processed",0))
        if nr.get("ml_trained"):
            m2.metric("ML Accuracy", f"{nr.get('ml_accuracy')}%")
            m3.metric("Model", "Logistic Regression")
        else:
            m2.metric("Sentiment Method", "VADER")
            m3.metric("ML Training", "Insufficient data")

        wc_col, sent_col = st.columns(2)
        with wc_col:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ Word Cloud")
            if nr.get("top_keywords"):
                wc = WordCloud(
                    width=700, height=350,
                    background_color="#16161F",
                    colormap="summer", max_words=80,
                    prefer_horizontal=0.85
                ).generate(" ".join(nr["top_keywords"]))
                fig_wc, ax_wc = plt.subplots(figsize=(6,3))
                ax_wc.imshow(wc); ax_wc.axis("off")
                fig_wc.patch.set_facecolor("#16161F")
                st.pyplot(fig_wc)
            st.markdown("</div>", unsafe_allow_html=True)

        with sent_col:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ Sentiment Split")
            sd = nr.get("sentiment_distribution", {})
            if sd:
                sl = list(sd.keys()); sv = list(sd.values())
                sc = [{"Positive":"#00FF88","Negative":"#FF2D78"}.get(l,"#FFE500") for l in sl]
                fig_s, ax_s = plt.subplots(figsize=(5,3.5))
                ax_s.pie(sv, labels=sl, autopct="%1.1f%%", colors=sc, startangle=140,
                         wedgeprops={"edgecolor":"#0A0A0F","linewidth":2},
                         textprops={"color":"#F0F0F5","fontsize":10})
                fig_s.patch.set_facecolor("#16161F")
                st.pyplot(fig_s)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
        section("◎ Detected Topics")
        topic_labels = ["Economy","Technology","Politics","Health","Sports"]
        for i, t in enumerate(nr.get("topics",[]), 1):
            lbl = topic_labels[i-1] if i <= len(topic_labels) else f"Topic {i}"
            st.markdown(f"""
            <div class="acid-topic-row">
                <span class="acid-topic-tag">{lbl}</span>
                <span class="acid-topic-kw">{", ".join(t)}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Trend Results
    if st.session_state.trend_result:
        tr = st.session_state.trend_result
        st.markdown("<hr>", unsafe_allow_html=True)
        section("◈ Trend Results")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Clean Records",      tr["final_records"])
        t2.metric("Duplicates Removed", tr["duplicates_removed"])
        t3.metric("ML Accuracy",        f"{tr['model_accuracy']}%")
        t4.metric("Baseline",           f"{tr['baseline_accuracy']}%")

        fc, tc = st.columns(2)
        with fc:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ Top Frequency Words")
            fdf2 = pd.DataFrame(tr["top_frequency_words"], columns=["Word","Frequency"])
            fig_f, ax_f = plt.subplots(figsize=(5,3.5))
            ax_f.barh(fdf2["Word"][::-1], fdf2["Frequency"][::-1],
                      color="#C8FF00", edgecolor="none", height=0.6)
            ax_f.xaxis.grid(True, alpha=0.3); ax_f.set_axisbelow(True)
            fig_f.tight_layout(); st.pyplot(fig_f)
            st.markdown("</div>", unsafe_allow_html=True)

        with tc:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ TF-IDF Keywords")
            for w in tr["top_tfidf_words"]:
                st.markdown(f'<span class="acid-pill">{w}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Analytics Dashboard
    m3path = os.path.join("data","milestone3_news.csv")
    df_adm = None
    if os.path.exists(m3path):
        try: df_adm = pd.read_csv(m3path)
        except: pass

    if df_adm is not None and not df_adm.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        section("◈ Analytics Dashboard")
        for col in ["processed_text","sentiment_label","URL","Date","Source","Title","Topic"]:
            if col not in df_adm.columns: df_adm[col] = ""

        words    = " ".join(df_adm["processed_text"].astype(str)).split()
        top_w    = Counter(words).most_common(10)
        kws      = [x[0] for x in top_w]
        cnts     = [x[1] for x in top_w]
        sent_c   = df_adm["sentiment_label"].value_counts()
        sl2      = sent_c.index.tolist()
        sv2      = sent_c.values.tolist()
        pc2      = [{"Positive":"#00FF88","Negative":"#FF2D78"}.get(l,"#FFE500") for l in sl2]

        kc, sc2 = st.columns(2)
        with kc:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ Top 10 Keywords")
            acid_bar_c = ["#C8FF00","#00FF88","#00E5FF","#FFE500","#FF6B00",
                          "#FF2D78","#C8FF00","#00FF88","#00E5FF","#FFE500"]
            fig_k, ax_k = plt.subplots(figsize=(6,3.8))
            ax_k.bar(kws, cnts, color=acid_bar_c[:len(kws)], edgecolor="none", width=0.6)
            ax_k.set_xticks(range(len(kws)))
            ax_k.set_xticklabels(kws, rotation=40, ha="right", fontsize=9)
            ax_k.yaxis.grid(True, alpha=0.3); ax_k.set_axisbelow(True)
            fig_k.tight_layout(); st.pyplot(fig_k)
            st.markdown("</div>", unsafe_allow_html=True)

        with sc2:
            st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
            section("◎ Sentiment Breakdown")
            p2, s2 = st.columns([1.2,1])
            with p2:
                fig_p3, ax_p3 = plt.subplots(figsize=(4,3.2))
                ax_p3.pie(sv2, labels=sl2, autopct="%1.1f%%", colors=pc2, startangle=140,
                          wedgeprops={"edgecolor":"#0A0A0F","linewidth":2},
                          textprops={"color":"#F0F0F5","fontsize":9})
                fig_p3.patch.set_facecolor("#16161F"); st.pyplot(fig_p3)
            with s2:
                total_a = len(df_adm)
                st.markdown(f"""
                <div style="padding-top:0.5rem;">
                    <div style="font-family:var(--mono,'SF Mono',monospace);font-size:0.6rem;
                                color:#8888AA;text-transform:uppercase;letter-spacing:0.08em;">Total</div>
                    <div style="font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;
                                font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:#F0F0F5;">
                        {total_a}
                    </div>
                """, unsafe_allow_html=True)
                for lbl, val in zip(sl2, sv2):
                    pct = round(val/total_a*100, 1)
                    c   = {"Positive":"#00FF88","Negative":"#FF2D78"}.get(lbl,"#FFE500")
                    st.markdown(f"""
                    <div style="margin-top:0.5rem;font-family:var(--mono,'SF Mono',monospace);font-size:0.7rem;">
                        <span style="color:{c};font-weight:600;">{lbl}</span>
                        <span style="color:#8888AA;"> {pct}%</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
        section("◎ System Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Articles",     len(df_adm))
        s2.metric("Top Keyword",        kws[0] if kws else "—")
        s3.metric("Dominant Sentiment", sent_c.idxmax())
        s4.metric("Unique Sources",     df_adm["Source"].nunique())
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='acid-card'>", unsafe_allow_html=True)
        section("◎ Article Table")
        dc = [c for c in ["Title","Source","Date","Topic","sentiment_label","URL"] if c in df_adm.columns]
        st.dataframe(df_adm[dc].head(50), use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="acid-empty">
            <span class="acid-empty-icon">◈</span>
            <div class="acid-empty-title">No Analytics Yet</div>
            <div class="acid-empty-sub">Run Fetch → NLP → Trend Analysis to populate</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "admin" and not st.session_state.is_admin:
    st.error("Admin access required.")
    st.session_state.page = "user"
    st.rerun()