# backend/news_service.py
import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY      = os.getenv("NEWS_API_KEY")
DATA_DIR     = "data"
RAW_FILE     = os.path.join(DATA_DIR, "news_raw.csv")
CLEANED_FILE = os.path.join(DATA_DIR, "news_data_cleaned.csv")

os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_TOPICS = [
    "technology", "politics", "business",
    "health", "science", "sports",
    "entertainment", "world"
]

# ==============================
# Fetch single topic
# ==============================
def fetch_news(query="technology", total_articles=100):
    if not API_KEY:
        raise ValueError("NEWS_API_KEY not set in .env")

    news_list = []
    page      = 1
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    while len(news_list) < total_articles:
        params = {
            "q":        query,
            "from":     from_date,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": 100,
            "page":     page,
            "apiKey":   API_KEY,
        }
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for '{query}': {e}")
            break

        if data.get("status") != "ok":
            print(f"[ERROR] API error for '{query}': {data.get('message')}")
            break

        articles = data.get("articles", [])
        if not articles:
            break

        for article in articles:
            title = article.get("title", "")
            if not title or title == "[Removed]":
                continue
            news_list.append({
                "Title":       title,
                "Description": article.get("description", ""),
                "Source":      article.get("source", {}).get("name", "Unknown"),
                "Date":        article.get("publishedAt", ""),
                "URL":         article.get("url", ""),
                "Topic":       query,
            })
            if len(news_list) >= total_articles:
                break
        page += 1

    return news_list


# ==============================
# Fetch 500+ across topics
# ==============================
def fetch_bulk_news(topics=None, target=500):
    if topics is None:
        topics = DEFAULT_TOPICS

    all_articles = []
    seen_titles  = set()

    print(f"[INFO] Fetching up to {target} articles across {len(topics)} topics...")

    for topic in topics:
        if len(all_articles) >= target:
            break

        remaining = target - len(all_articles)
        per_topic = min(100, remaining)

        print(f"[INFO] Topic: '{topic}' — fetching up to {per_topic}")
        articles = fetch_news(query=topic, total_articles=per_topic)

        added = 0
        for a in articles:
            if a["Title"] not in seen_titles:
                seen_titles.add(a["Title"])
                all_articles.append(a)
                added += 1

        print(f"[INFO] +{added} unique (total: {len(all_articles)})")

    print(f"[INFO] Final: {len(all_articles)} unique articles")
    return all_articles[:target]


# ==============================
# Save raw CSV
# ==============================
def save_news_to_csv(news_list):
    if not news_list:
        print("[WARN] No articles to save.")
        return 0
    df = pd.DataFrame(news_list)
    df.to_csv(RAW_FILE, index=False)
    print(f"[INFO] Saved {len(df)} raw articles → {RAW_FILE}")
    return len(df)


# ==============================
# Clean data
# ==============================
def clean_news_data():
    if not os.path.exists(RAW_FILE):
        print("[WARN] Raw file not found.")
        return None

    df = pd.read_csv(RAW_FILE)
    before = len(df)

    df = df.dropna(subset=["Title"])
    df = df.drop_duplicates(subset=["Title", "Source"])

    df["Title"] = df["Title"].apply(
        lambda t: re.sub(r"\s+", " ", str(t)).strip()
    )
    df["Description"] = df["Description"].fillna("").apply(
        lambda t: re.sub(r"\s+", " ", str(t)).strip()
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(CLEANED_FILE, index=False)
    after = len(df)
    print(f"[INFO] Cleaned: {before} → {after} ({before - after} removed)")
    return {"before": before, "after": after, "removed": before - after}


# ==============================
# Load data — always prefer freshest large file
# ==============================
def load_news_data(cleaned=True):
    paths = [
        os.path.join(DATA_DIR, "news_data_cleaned.csv"),   # ← freshest after fetch
        os.path.join(DATA_DIR, "news_raw.csv"),
        os.path.join(DATA_DIR, "processed_news.csv"),
        os.path.join(DATA_DIR, "milestone3_news.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


# ==============================
# Full pipeline
# ==============================
def run_full_pipeline(query="technology"):
    print(f"[INFO] Pipeline started: '{query}'")

    if query.strip().lower() in ("all", "technology", ""):
        news = fetch_bulk_news(topics=DEFAULT_TOPICS, target=500)
    else:
        # specific topic — fetch up to 100 (free plan limit)
        news = fetch_news(query=query, total_articles=100)

    saved_count   = save_news_to_csv(news)
    cleaning_info = clean_news_data()

    return {
        "fetched": len(news),
        "saved":   saved_count,
        "stats":   {"total_articles": saved_count, "cleaning": cleaning_info},
    }