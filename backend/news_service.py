import requests
import pandas as pd
import re
import os
from datetime import datetime, timedelta

# ==============================
# Configuration
# ==============================
API_KEY = "8f699a3ccdb149abb366f107cb93ba24"   # <-- put your NewsAPI key here

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "news_data.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "news_data_cleaned.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# ==============================
# Fetch Recent News (Day 3–4)
# ==============================
def fetch_news(query="technology", total_articles=60):
    news_list = []
    page = 1
    page_size = 20

    # Last 24 hours
    from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    while len(news_list) < total_articles:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"from={from_date}&"
            f"sortBy=publishedAt&"
            f"language=en&"
            f"pageSize={page_size}&"
            f"page={page}&"
            f"apiKey={API_KEY}"
        )

        response = requests.get(url)
        data = response.json()

        if data.get("status") != "ok" or not data.get("articles"):
            break

        for article in data["articles"]:
            news_list.append({
                "Title": article.get("title"),
                "Description": article.get("description"),
                "Source": article["source"]["name"],
                "Date": article.get("publishedAt")
            })

            if len(news_list) >= total_articles:
                break

        page += 1

    return news_list


# ==============================
# Save to CSV (Day 6)
# ==============================
def save_news_to_csv(news_list):
    df = pd.DataFrame(news_list)
    df.to_csv(RAW_FILE, index=False)
    return len(df)


# ==============================
# Load Data (Day 7)
# ==============================
def load_news_data(cleaned=False):
    file_path = CLEAN_FILE if cleaned else RAW_FILE

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None


# ==============================
# Clean Data (Day 8)
# ==============================
def clean_news_data():
    df = load_news_data(cleaned=False)

    if df is None:
        return None

    before_rows = len(df)

    # Remove empty titles
    df = df.dropna(subset=["Title"])

    # Remove duplicates
    df = df.drop_duplicates(subset=["Title", "Source"])

    # Clean text
    def clean_text(text):
        text = str(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df["Title"] = df["Title"].apply(clean_text)

    df.to_csv(CLEAN_FILE, index=False)

    after_rows = len(df)

    return {"before": before_rows, "after": after_rows}


# ==============================
# Analysis (Day 9)
# ==============================
def get_dataset_stats():
    df = load_news_data(cleaned=True)

    if df is None:
        return None

    return {
        "total_articles": len(df),
        "unique_sources": df["Source"].nunique(),
        "sources": df["Source"].unique().tolist()
    }


# ==============================
# Full Pipeline (Day 3–9)
# ==============================
def run_full_pipeline(query="technology"):
    news = fetch_news(query, total_articles=60)
    saved_count = save_news_to_csv(news)
    cleaning_info = clean_news_data()
    stats = get_dataset_stats()

    return {
        "fetched": len(news),
        "saved": saved_count,
        "cleaning": cleaning_info,
        "stats": stats
    }