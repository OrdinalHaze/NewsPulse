import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

DATA_DIR = "data"

# Input dataset from Milestone 2
INPUT_FILE = os.path.join(DATA_DIR, "processed_news.csv")

# Output dataset for Milestone 3
OUTPUT_FILE = os.path.join(DATA_DIR, "milestone3_news.csv")


# =====================================
# Remove Duplicate Articles
# =====================================
def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(subset=["Title"])

    after = len(df)

    return df, before, after


# =====================================
# Noise Reduction
# =====================================
def clean_dataset(df):

    df = df.dropna()

    df = df[df["processed_text"].str.len() > 20]

    return df


# =====================================
# Frequency-Based Trend Detection
# =====================================
def frequency_trends(df, top_n=5):

    all_words = " ".join(df["processed_text"]).split()

    word_freq = Counter(all_words)

    return word_freq.most_common(top_n)


# =====================================
# TF-IDF Trend Detection
# =====================================
def tfidf_trends(df, top_n=10):

    vectorizer = TfidfVectorizer(max_features=100)

    X = vectorizer.fit_transform(df["processed_text"])

    feature_names = vectorizer.get_feature_names_out()

    mean_scores = X.mean(axis=0).A1

    ranked_indices = mean_scores.argsort()[::-1]

    top_words = [feature_names[i] for i in ranked_indices[:top_n]]

    return top_words


# =====================================
# Sentiment Score
# =====================================
def add_sentiment_score(df):

    df["sentiment_score"] = df["processed_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return df


# =====================================
# Sentiment Label (Threshold Logic)
# =====================================
def classify_sentiment(score):

    if score > 0:
        return "Positive"

    elif score < 0:
        return "Negative"

    else:
        return "Neutral"


def add_sentiment_label(df):

    df["sentiment_label"] = df["sentiment_score"].apply(classify_sentiment)

    return df


# =====================================
# Baseline Accuracy
# =====================================
def baseline_accuracy(df):

    most_common = df["sentiment_label"].value_counts().max()

    total = len(df)

    baseline = (most_common / total) * 100

    return float(round(baseline, 2))


# =====================================
# Model Accuracy (Proxy Evaluation)
# =====================================
def evaluate_sentiment_model(df, sample_size=20, threshold=0.2):

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

    sample["proxy_truth"] = sample["sentiment_score"].apply(proxy_truth)

    correct = (sample["sentiment_label"] == sample["proxy_truth"]).sum()

    accuracy = (correct / sample_size) * 100

    return float(round(accuracy, 2))


# =====================================
# Main Milestone 3 Pipeline
# =====================================
def run_milestone3():

    if not os.path.exists(INPUT_FILE):

        return {"error": "Run Milestone 2 first (processed_news.csv missing)."}

    df = pd.read_csv(INPUT_FILE)

    # Remove duplicates
    df, before, after = remove_duplicates(df)

    # Clean dataset
    df = clean_dataset(df)

    # Trend detection
    freq_trends = frequency_trends(df)

    tfidf_words = tfidf_trends(df)

    # Sentiment analysis
    df = add_sentiment_score(df)

    df = add_sentiment_label(df)

    sentiment_counts = df["sentiment_label"].value_counts().to_dict()

    # Evaluation metrics
    model_acc = evaluate_sentiment_model(df)

    baseline_acc = baseline_accuracy(df)

    # Save dataset
    df.to_csv(OUTPUT_FILE, index=False)

    return {
        "duplicates_removed": before - after,
        "top_frequency_words": freq_trends,
        "top_tfidf_words": tfidf_words,
        "sentiment_distribution": sentiment_counts,
        "model_accuracy": model_acc,
        "baseline_accuracy": baseline_acc,
        "final_records": len(df)
    }