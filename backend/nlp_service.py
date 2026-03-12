# backend/nlp_service.py
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

DATA_DIR           = "data"
INPUT_FILE         = os.path.join(DATA_DIR, "news_data_cleaned.csv")
CLEANED_TEXT_FILE  = os.path.join(DATA_DIR, "news_cleaned_text.csv")
PROCESSED_FILE     = os.path.join(DATA_DIR, "processed_news.csv")
SENTIMENT_FILE     = os.path.join(DATA_DIR, "news_with_sentiment.csv")

nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)

STOPWORDS = set(stopwords.words("english"))

# ==============================
# Text preprocessing
# ==============================
def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        t for t in word_tokenize(text)
        if t.isalpha() and t not in STOPWORDS and len(t) > 2
    ]
    return " ".join(tokens)


# ==============================
# Step 1 — cleaned_text file
# ==============================
def create_cleaned_text_file():
    # Try to find the best available source file
    source_candidates = [
        os.path.join(DATA_DIR, "news_data_cleaned.csv"),
        os.path.join(DATA_DIR, "news_raw.csv"),
    ]
    df = None
    for path in source_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                break

    if df is None or df.empty:
        raise FileNotFoundError("No source news file found. Please fetch news first.")

    title_col = "Title"       if "Title"       in df.columns else df.columns[0]
    desc_col  = "Description" if "Description" in df.columns else ""

    df["full_text"] = df[title_col].fillna("")
    if desc_col:
        df["full_text"] = df["full_text"] + " " + df[desc_col].fillna("")

    df["cleaned_text"] = df["full_text"].astype(str).apply(preprocess_text)
    df = df[df["cleaned_text"].str.strip().str.len() > 0]
    df.to_csv(CLEANED_TEXT_FILE, index=False)
    print(f"[INFO] Cleaned text saved: {len(df)} rows → {CLEANED_TEXT_FILE}")
    return CLEANED_TEXT_FILE


# ==============================
# Step 2 — processed_news file
# ==============================
def create_processed_dataset():
    if not os.path.exists(CLEANED_TEXT_FILE):
        create_cleaned_text_file()
    df = pd.read_csv(CLEANED_TEXT_FILE)
    df["processed_text"] = df["cleaned_text"]
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"[INFO] Processed dataset saved: {len(df)} rows → {PROCESSED_FILE}")
    return PROCESSED_FILE


# ==============================
# TF-IDF keywords
# ==============================
def extract_top_keywords(df, top_n=50):
    texts = df["processed_text"].fillna("").tolist()
    texts = [t for t in texts if len(t.strip()) > 5]
    if not texts:
        return []
    vec = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X   = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    mean_scores   = X.mean(axis=0).A1
    ranked_idx    = mean_scores.argsort()[::-1]
    return [feature_names[i] for i in ranked_idx[:top_n]]


# ==============================
# LDA topics
# ==============================
def compute_topics(df, n_topics=4, n_words=8):
    texts = df["processed_text"].fillna("").tolist()
    texts = [t for t in texts if len(t.strip()) > 5]
    if not texts:
        return [[] for _ in range(n_topics)]
    vec = TfidfVectorizer(max_features=500)
    X   = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vec.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        terms = [feature_names[i] for i in comp.argsort()[-n_words:][::-1]]
        topics.append(terms)
    return topics


# ==============================
# Sentiment
# ==============================
def add_sentiment(df):
    def get_label(text):
        score = TextBlob(str(text)).sentiment.polarity
        if score > 0.05:
            return "Positive", score
        elif score < -0.05:
            return "Negative", score
        else:
            return "Neutral", score

    results = df["processed_text"].fillna("").apply(get_label)
    df["sentiment_label"] = results.apply(lambda x: x[0])
    df["sentiment_score"] = results.apply(lambda x: x[1])
    df.to_csv(SENTIMENT_FILE, index=False)
    return df


# ==============================
# Full NLP pipeline
# ==============================
def run_nlp_pipeline():
    try:
        processed_path = create_processed_dataset()
        df = pd.read_csv(processed_path)

        if df.empty or "processed_text" not in df.columns:
            return {"error": "Processed dataset is empty or missing 'processed_text' column."}

        top_keywords = extract_top_keywords(df, top_n=50)
        topics       = compute_topics(df, n_topics=4)
        df           = add_sentiment(df)

        sentiment_distribution = df["sentiment_label"].value_counts().to_dict()
        records_processed      = len(df)

        # save back with sentiment columns
        df.to_csv(processed_path, index=False)

        print(f"[INFO] NLP complete: {records_processed} records processed.")
        return {
            "records_processed":      records_processed,
            "top_keywords":           top_keywords,
            "topics":                 topics,
            "sentiment_distribution": sentiment_distribution,
        }
    except Exception as e:
        print(f"[ERROR] NLP pipeline failed: {e}")
        return {"error": str(e)}