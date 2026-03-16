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

            pub_date = article.get("publishedAt", "")
            try:
                pub_date = datetime.strptime(
                    pub_date, "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%d %b %Y, %H:%M")
            except Exception:
                pass

            news_list.append({
                "Title":       title,
                "Description": article.get("description", "") or "",
                "Source":      article.get("source", {}).get("name", "Unknown"),
                "Date":        pub_date,
                "URL":         article.get("url", "") or "",
                "Topic":       query,
            })
            if len(news_list) >= total_articles:
                break
        page += 1

    return news_list


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
        articles  = fetch_news(query=topic, total_articles=per_topic)

        for a in articles:
            if a["Title"] not in seen_titles:
                seen_titles.add(a["Title"])
                all_articles.append(a)

    return all_articles[:target]


def save_news_to_csv(news_list):
    if not news_list:
        return 0
    df = pd.DataFrame(news_list)
    df.to_csv(RAW_FILE, index=False)
    return len(df)


def clean_news_data():
    if not os.path.exists(RAW_FILE):
        return None

    df = pd.read_csv(RAW_FILE)
    before = len(df)

    df = df.dropna(subset=["Title"])
    df = df.drop_duplicates(subset=["Title", "Source"])
    df["Title"]       = df["Title"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())
    df["Description"] = df["Description"].fillna("").apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())
    df["URL"]         = df["URL"].fillna("")
    df["Date"]        = df["Date"].fillna("")
    df["Topic"]       = df["Topic"].fillna("general")

    df.to_csv(CLEANED_FILE, index=False)
    after = len(df)
    return {"before": before, "after": after, "removed": before - after}


def load_news_data(cleaned=True):
    # Priority: files with sentiment_label first
    priority_paths = [
        os.path.join(DATA_DIR, "milestone3_news.csv"),
        os.path.join(DATA_DIR, "processed_news.csv"),
    ]
    fallback_paths = [
        os.path.join(DATA_DIR, "news_data_cleaned.csv"),
        os.path.join(DATA_DIR, "news_raw.csv"),
    ]
    for p in priority_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if not df.empty and "sentiment_label" in df.columns:
                    return df
            except Exception:
                continue
    for p in fallback_paths + priority_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                continue
    return None


def clear_old_data():
    """Delete all stale CSVs before a fresh fetch."""
    stale_files = [
        "news_raw.csv",
        "news_data_cleaned.csv",
        "news_cleaned_text.csv",
        "processed_news.csv",
        "news_with_sentiment.csv",
        "milestone3_news.csv",
    ]
    for f in stale_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Deleted stale file: {path}")


def run_full_pipeline(query="technology"):
    # ✅ Always clear old data first to prevent stale sentiment/NLP issues
    clear_old_data()

    # Single query, max 100 articles (free NewsAPI plan limit)
    news = fetch_news(query=query, total_articles=100)

    saved_count   = save_news_to_csv(news)
    cleaning_info = clean_news_data()

    return {
        "fetched": len(news),
        "saved":   saved_count,
        "stats":   {"total_articles": saved_count, "cleaning": cleaning_info},
    }