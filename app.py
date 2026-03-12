import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from wordcloud import WordCloud

from backend.news_service import run_full_pipeline, load_news_data
from backend.nlp_service import run_nlp_pipeline
from backend.trend_service import run_milestone3
from backend.google_auth import get_google_auth_url, get_user_info

# =========================
# CONFIG
# =========================
REDIRECT_URI = "http://localhost:8501"

st.set_page_config(
    page_title="NewsPulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL DARK THEME CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ---- Root colors ---- */
:root {
    --bg-primary:    #0d1b2a;
    --bg-card:       #132237;
    --bg-card2:      #1a2f4a;
    --accent-blue:   #1a73e8;
    --accent-green:  #34a853;
    --accent-red:    #ea4335;
    --accent-yellow: #fbbc04;
    --accent-teal:   #17becf;
    --text-primary:  #e8f0fe;
    --text-muted:    #8aa3be;
    --border:        rgba(255,255,255,0.07);
}

/* ---- Global background ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: #0a1520 !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ---- Header banner ---- */
.header-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #132237 50%, #0d1b2a 100%);
    border-bottom: 2px solid rgba(26,115,232,0.4);
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    text-align: center;
}
.header-banner h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #e8f0fe;
    margin: 0;
    letter-spacing: 2px;
}
.header-banner h1 span {
    color: var(--accent-blue);
}
.header-banner p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0.3rem 0 0;
    letter-spacing: 1px;
    font-weight: 300;
}

/* ---- Section headers ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1.2rem;
    letter-spacing: 1px;
}
.section-header .icon { font-size: 1.2rem; }

/* ---- Cards / panels ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ---- Metric tiles ---- */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-tile {
    flex: 1;
    min-width: 130px;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-tile .lbl {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    opacity: 0.85;
}
.tile-blue   { background: var(--accent-blue);   color: #fff; }
.tile-green  { background: var(--accent-green);  color: #fff; }
.tile-red    { background: var(--accent-red);    color: #fff; }
.tile-grey   { background: #3c4d5c;              color: #fff; }
.tile-teal   { background: var(--accent-teal);   color: #fff; }
.tile-yellow { background: var(--accent-yellow); color: #1a1a1a; }

/* ---- Topic badges ---- */
.topic-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
}
.topic-row:last-child { border-bottom: none; }
.topic-badge {
    background: var(--bg-card2);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 5px;
    padding: 0.2rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.5px;
    white-space: nowrap;
    color: var(--text-primary);
    min-width: 80px;
    text-align: center;
}
.topic-keywords { font-size: 0.82rem; color: var(--text-muted); }

/* ---- Sentiment tags ---- */
.tag-pos   { color: var(--accent-green)  !important; font-weight: 600; }
.tag-neg   { color: var(--accent-red)    !important; font-weight: 600; }
.tag-neu   { color: var(--accent-yellow) !important; font-weight: 600; }

/* ---- Table styling ---- */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
thead th { background: var(--bg-card2) !important; color: var(--text-primary) !important; }
tbody tr { background: var(--bg-card) !important; }
tbody tr:nth-child(even) { background: var(--bg-card2) !important; }

/* ---- Streamlit widget overrides ---- */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1.8rem !important; }

/* ---- Buttons ---- */
[data-testid="stButton"] > button {
    background: var(--accent-blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ---- Spinner / info / success ---- */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ---- Divider ---- */
hr { border-color: var(--border) !important; }

/* ---- Sidebar text input ---- */
[data-testid="stTextInput"] input {
    background: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    border-radius: 6px !important;
}

/* ---- Charts background ---- */
.stPlotlyChart, .element-container { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)

# =========================
# MATPLOTLIB DARK THEME
# =========================
matplotlib.rcParams.update({
    "figure.facecolor":  "#132237",
    "axes.facecolor":    "#132237",
    "axes.edgecolor":    "#2a4060",
    "axes.labelcolor":   "#8aa3be",
    "xtick.color":       "#8aa3be",
    "ytick.color":       "#8aa3be",
    "text.color":        "#e8f0fe",
    "grid.color":        "#1e3048",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})

# =========================
# SESSION
# =========================
if "user" not in st.session_state:
    st.session_state.user = None

# =========================
# GOOGLE LOGIN HANDLER
# =========================
query_params = st.query_params

if "code" in query_params and st.session_state.user is None:
    code = query_params.get("code")
    if isinstance(code, list):
        code = code[0]
    user_info = get_user_info(code, REDIRECT_URI)
    if user_info and "email" in user_info:
        st.session_state.user = user_info["email"]
    else:
        st.error("Google login failed")
        st.stop()
    st.query_params.clear()
    st.rerun()

# =========================
# LOGIN PAGE
# =========================
if not st.session_state.user:

    st.markdown("""
    <div class="header-banner">
        <h1>📰 News<span>Pulse</span></h1>
        <p>Global News Trend Analyzer · Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    auth_url = get_google_auth_url(REDIRECT_URI)

    col_c, col_btn, col_r = st.columns([3, 2, 3])
    with col_btn:
        st.markdown(f"""
        <div style="text-align:center; margin-top:3rem;">
            <a href="{auth_url}" style="text-decoration:none;">
            <button style="padding:12px 28px; font-size:16px; font-weight:600;
                           background:#1a73e8; color:white; border:none;
                           border-radius:8px; cursor:pointer; letter-spacing:0.5px;
                           font-family:'Inter',sans-serif;">
                🔐 &nbsp;Login with Google
            </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown(f"""
<div style="background:#132237; border-radius:8px; padding:0.8rem 1rem;
            border:1px solid rgba(255,255,255,0.08); margin-bottom:1rem;">
    <div style="font-size:0.7rem; color:#8aa3be; text-transform:uppercase; letter-spacing:1px;">Logged in as</div>
    <div style="font-size:0.9rem; font-weight:600; color:#e8f0fe; margin-top:0.2rem;">
        {st.session_state.user}
    </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown("<div class='section-header'><span class='icon'>⚙️</span> Controls</div>", unsafe_allow_html=True)

query = st.sidebar.text_input("🔍 Search Topic", "technology")

if st.sidebar.button("📡 Fetch Latest News"):
    with st.spinner("Fetching news..."):
        result = run_full_pipeline(query)
    st.sidebar.success(f"✅ Fetched {result['stats']['total_articles']} articles")

# =========================
# HEADER BANNER
# =========================
st.markdown("""
<div class="header-banner">
    <h1>📊 News<span>Pulse</span> Dashboard</h1>
    <p>AI-Powered Global News Trend Analyzer</p>
</div>
""", unsafe_allow_html=True)

# =========================
# MILESTONE 1 — DATA OVERVIEW
# =========================
df = load_news_data(cleaned=True)

if df is not None and not df.empty:

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(by="Date", ascending=False)

    total_articles = len(df)
    unique_sources = df["Source"].nunique()

    st.markdown("""
    <div class='section-header'>
        <span class='icon'>📰</span> News Summary
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile tile-blue">
            <div class="val">{total_articles}</div>
            <div class="lbl">Total Articles</div>
        </div>
        <div class="metric-tile tile-teal">
            <div class="val">{unique_sources}</div>
            <div class="lbl">Unique Sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span class='icon'>📊</span> Articles by Source</div>", unsafe_allow_html=True)
        src_counts = df["Source"].value_counts()
        fig_src, ax_src = plt.subplots(figsize=(5, 3.5))
        bars = ax_src.bar(src_counts.index[:8], src_counts.values[:8],
                          color=["#1a73e8","#34a853","#ea4335","#fbbc04",
                                 "#17becf","#9c27b0","#ff5722","#607d8b"])
        ax_src.set_xticklabels(src_counts.index[:8], rotation=40, ha="right", fontsize=8)
        ax_src.yaxis.grid(True)
        ax_src.set_axisbelow(True)
        fig_src.tight_layout()
        st.pyplot(fig_src)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_table:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span class='icon'>🗂️</span> Latest News Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:2.5rem; color:#8aa3be;">
        <div style="font-size:2.5rem;">📡</div>
        <div style="margin-top:0.5rem; font-size:1rem;">Use the sidebar to fetch news and load the dataset.</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MILESTONE 2 — NLP
# =========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='section-header'>
    <span class='icon'>🧠</span> NLP Analysis
</div>
""", unsafe_allow_html=True)

if st.button("▶ Run NLP Analysis"):

    with st.spinner("Running NLP pipeline..."):
        nlp_result = run_nlp_pipeline()

    if "error" in nlp_result:
        st.error(nlp_result["error"])
    else:
        st.success(f"✅ Processed {nlp_result['records_processed']} articles")

        col_wc, col_sent = st.columns(2)

        with col_wc:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><span class='icon'>☁️</span> Keyword Word Cloud</div>", unsafe_allow_html=True)
            keywords_text = " ".join(nlp_result["top_keywords"])
            wordcloud = WordCloud(
                width=700, height=350,
                background_color="#132237",
                colormap="Blues",
                max_words=80
            ).generate(keywords_text)
            fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
            ax_wc.imshow(wordcloud)
            ax_wc.axis("off")
            fig_wc.patch.set_facecolor("#132237")
            st.pyplot(fig_wc)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_sent:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><span class='icon'>😊</span> Sentiment Distribution</div>", unsafe_allow_html=True)
            sentiment = nlp_result["sentiment_distribution"]
            labels = list(sentiment.keys())
            values = list(sentiment.values())
            colors = ["#34a853", "#ea4335", "#fbbc04"]
            fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
            wedges, texts, autotexts = ax_pie.pie(
                values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=140,
                textprops={"color": "#e8f0fe", "fontsize": 11}
            )
            for at in autotexts:
                at.set_fontweight("bold")
            fig_pie.patch.set_facecolor("#132237")
            st.pyplot(fig_pie)
            st.markdown("</div>", unsafe_allow_html=True)

        # Topics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span class='icon'>🏷️</span> Detected Topics</div>", unsafe_allow_html=True)
        topic_colors = ["Economy","Technology","Politics","Health","Sports"]
        for i, topic in enumerate(nlp_result["topics"], 1):
            label = topic_colors[i-1] if i <= len(topic_colors) else f"Topic {i}"
            kws = ", ".join(topic)
            st.markdown(f"""
            <div class="topic-row">
                <span class="topic-badge">{label}</span>
                <span class="topic-keywords">{kws}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MILESTONE 3 — TRENDS
# =========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='section-header'>
    <span class='icon'>📈</span> Trend Detection & Evaluation
</div>
""", unsafe_allow_html=True)

if st.button("▶ Run Trend Analysis"):

    with st.spinner("Running trend analysis..."):
        result = run_milestone3()

    if "error" in result:
        st.error(result["error"])
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Clean Records", result["final_records"])
        col_m2.metric("Duplicates Removed", result["duplicates_removed"])
        col_m3.metric("Model Accuracy", f"{result['model_accuracy']}%")
        col_m4.metric("Baseline Accuracy", f"{result['baseline_accuracy']}%")

        col_freq, col_tfidf = st.columns(2)

        with col_freq:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><span class='icon'>🔎</span> Top Frequency Words</div>", unsafe_allow_html=True)
            freq_df = pd.DataFrame(result["top_frequency_words"], columns=["Word", "Frequency"])
            fig_freq, ax_freq = plt.subplots(figsize=(5, 3.5))
            ax_freq.barh(freq_df["Word"][::-1], freq_df["Frequency"][::-1], color="#1a73e8")
            ax_freq.xaxis.grid(True)
            ax_freq.set_axisbelow(True)
            fig_freq.tight_layout()
            st.pyplot(fig_freq)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_tfidf:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><span class='icon'>🔑</span> Top TF-IDF Keywords</div>", unsafe_allow_html=True)
            for w in result["top_tfidf_words"]:
                st.markdown(f"""
                <div style="display:inline-block; background:#1a2f4a; border:1px solid rgba(26,115,232,0.3);
                            border-radius:5px; padding:0.25rem 0.7rem; margin:0.25rem; font-size:0.85rem;
                            color:#e8f0fe; font-family:'Rajdhani',sans-serif; font-weight:600;">
                    {w}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MILESTONE 4 — ADMIN DASHBOARD
# =========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='section-header'>
    <span class='icon'>📊</span> Admin Dashboard
</div>
""", unsafe_allow_html=True)

try:
    df_admin = pd.read_csv("data/milestone3_news.csv")
except Exception:
    df_admin = None

if df_admin is not None:

    # Trending keywords bar chart
    words = " ".join(df_admin["processed_text"].astype(str)).split()
    word_count = Counter(words)
    top_words = word_count.most_common(10)
    keywords = [w[0] for w in top_words]
    counts   = [w[1] for w in top_words]

    sentiment_counts = df_admin["sentiment_label"].value_counts()
    s_labels = sentiment_counts.index.tolist()
    s_values = sentiment_counts.values.tolist()
    sent_colors = {"Positive": "#34a853", "Negative": "#ea4335", "Neutral": "#fbbc04"}
    pie_colors = [sent_colors.get(l, "#8aa3be") for l in s_labels]

    col_kw, col_sa = st.columns(2)

    with col_kw:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span class='icon'>🔥</span> Live Global Trends – Top 10 Keywords</div>", unsafe_allow_html=True)
        bar_colors = ["#1a73e8","#34a853","#ea4335","#fbbc04",
                      "#17becf","#9c27b0","#ff7043","#607d8b",
                      "#00acc1","#8d6e63"]
        fig_kw, ax_kw = plt.subplots(figsize=(6, 3.8))
        ax_kw.bar(keywords, counts, color=bar_colors[:len(keywords)])
        ax_kw.set_xticklabels(keywords, rotation=40, ha="right", fontsize=9)
        ax_kw.yaxis.grid(True)
        ax_kw.set_axisbelow(True)
        fig_kw.tight_layout()
        st.pyplot(fig_kw)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sa:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span class='icon'>😊</span> Global News Sentiment Analysis</div>", unsafe_allow_html=True)

        pie_col, stats_col = st.columns([1.2, 1])

        with pie_col:
            fig_pie2, ax_pie2 = plt.subplots(figsize=(4, 3.2))
            wedges, texts, autotexts = ax_pie2.pie(
                s_values, labels=s_labels, autopct="%1.1f%%",
                colors=pie_colors, startangle=140,
                textprops={"color": "#e8f0fe", "fontsize": 10}
            )
            for at in autotexts:
                at.set_fontweight("bold")
            fig_pie2.patch.set_facecolor("#132237")
            st.pyplot(fig_pie2)

        with stats_col:
            total_a = len(df_admin)
            st.markdown(f"""
            <div style="padding-top:0.5rem; font-size:0.85rem; line-height:2;">
                <div style="color:#8aa3be;">Total Articles</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700;">{total_a}</div>
            """, unsafe_allow_html=True)
            for lbl, val in zip(s_labels, s_values):
                pct = round(val / total_a * 100, 1)
                col_cls = {"Positive":"tag-pos","Negative":"tag-neg"}.get(lbl,"tag-neu")
                st.markdown(f"""
                <div style="margin-top:0.3rem;">
                    <span class="{col_cls}">{lbl}:</span>
                    <span style="color:#e8f0fe;"> {pct}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # System summary
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><span class='icon'>🖥️</span> System Summary</div>", unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Total Articles Analyzed", len(df_admin))
    s_col2.metric("Top Trending Keyword", keywords[0] if keywords else "—")
    s_col3.metric("Most Common Sentiment", sentiment_counts.idxmax())
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:2rem; color:#8aa3be;">
        <div style="font-size:2rem;">⚠️</div>
        <div style="margin-top:0.5rem;">Run Trend Analysis first to load admin dashboard data.</div>
    </div>
    """, unsafe_allow_html=True)