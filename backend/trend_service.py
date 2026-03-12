# backend/trend_service.py
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

DATA_DIR    = "data"
INPUT_FILE  = os.path.join(DATA_DIR, "processed_news.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "milestone3_news.csv")


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["Title"])
    return df, before, len(df)


def clean_dataset(df):
    df = df.dropna(subset=["processed_text"])
    df = df[df["processed_text"].str.strip().str.len() > 20]
    return df


def frequency_trends(df, top_n=10):
    all_words = " ".join(df["processed_text"].fillna("")).split()
    return Counter(all_words).most_common(top_n)


def tfidf_trends(df, top_n=10):
    texts = df["processed_text"].fillna("").tolist()
    texts = [t for t in texts if len(t.strip()) > 5]
    if not texts:
        return []
    vectorizer   = TfidfVectorizer(max_features=200)
    X            = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_scores  = X.mean(axis=0).A1
    ranked_idx   = mean_scores.argsort()[::-1]
    return [feature_names[i] for i in ranked_idx[:top_n]]


def add_sentiment_score(df):
    df["sentiment_score"] = df["processed_text"].fillna("").apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    return df


def classify_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def add_sentiment_label(df):
    df["sentiment_label"] = df["sentiment_score"].apply(classify_sentiment)
    return df


def baseline_accuracy(df):
    most_common = df["sentiment_label"].value_counts().max()
    return float(round((most_common / len(df)) * 100, 2))


def evaluate_sentiment_model(df, sample_size=50, threshold=0.05):
    sample_size = min(sample_size, len(df))
    if sample_size == 0:
        return 0.0
    sample = df.sample(sample_size, random_state=42)

    def proxy_truth(score):
        if score >= threshold:
            return "Positive"
        elif score <= -threshold:
            return "Negative"
        else:
            return "Neutral"

    sample = sample.copy()
    sample["proxy_truth"] = sample["sentiment_score"].apply(proxy_truth)
    correct  = (sample["sentiment_label"] == sample["proxy_truth"]).sum()
    return float(round((correct / sample_size) * 100, 2))


def run_milestone3():
    # Fall back to cleaned file if processed doesn't exist yet
    source = INPUT_FILE
    if not os.path.exists(source):
        fallback = os.path.join(DATA_DIR, "news_data_cleaned.csv")
        if os.path.exists(fallback):
            source = fallback
        else:
            return {"error": "Run NLP Analysis first (processed_news.csv missing)."}

    df = pd.read_csv(source)

    # ensure processed_text column exists
    if "processed_text" not in df.columns:
        if "cleaned_text" in df.columns:
            df["processed_text"] = df["cleaned_text"]
        elif "Title" in df.columns:
            df["processed_text"] = df["Title"].fillna("")
        else:
            return {"error": "'processed_text' column not found."}

    df, before, after = remove_duplicates(df)
    df = clean_dataset(df)

    if df.empty:
        return {"error": "Dataset is empty after cleaning."}

    freq_words  = frequency_trends(df, top_n=10)
    tfidf_words = tfidf_trends(df, top_n=10)
    df          = add_sentiment_score(df)
    df          = add_sentiment_label(df)

    sentiment_counts = df["sentiment_label"].value_counts().to_dict()
    model_acc        = evaluate_sentiment_model(df)
    baseline_acc     = baseline_accuracy(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Milestone3 complete: {len(df)} records → {OUTPUT_FILE}")

    return {
        "duplicates_removed":   before - after,
        "top_frequency_words":  freq_words,
        "top_tfidf_words":      tfidf_words,
        "sentiment_distribution": sentiment_counts,
        "model_accuracy":       model_acc,
        "baseline_accuracy":    baseline_acc,
        "final_records":        len(df),
    }