import os
import re
import pandas as pd
import numpy as np

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from textblob import TextBlob

# =====================================
# NLTK Setup (download only if missing)
# =====================================
def download_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",   # Required for Python 3.14+
        "stopwords": "corpora/stopwords"
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

stop_words = set(stopwords.words("english"))

# =====================================
# Paths
# =====================================
DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "news_data_cleaned.csv")
CLEAN_TEXT_FILE = os.path.join(DATA_DIR, "news_cleaned_text.csv")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_news.csv")
FINAL_FILE = os.path.join(DATA_DIR, "news_with_sentiment.csv")

# =====================================
# Step 1 – Basic Text Cleaning
# =====================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)        # remove HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove numbers & punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_clean_text_dataset():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Run Milestone 1 first: news_data_cleaned.csv not found")

    df = pd.read_csv(INPUT_FILE)

    # Combine title and description
    df["full_text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    df["cleaned_text"] = df["full_text"].apply(clean_text)

    df.to_csv(CLEAN_TEXT_FILE, index=False)
    return len(df)


# =====================================
# Step 2 – Tokenization + Stopwords
# =====================================
def preprocess_text(text):
    tokens = word_tokenize(text)

    filtered = [
        word for word in tokens
        if word.isalpha()
        and word not in stop_words
        and len(word) > 2
    ]

    return " ".join(filtered)


def create_processed_dataset():
    df = pd.read_csv(CLEAN_TEXT_FILE)

    df["processed_text"] = df["cleaned_text"].apply(preprocess_text)

    df.to_csv(PROCESSED_FILE, index=False)
    return len(df)


# =====================================
# Step 3 – TF-IDF Keywords
# =====================================
def extract_top_keywords(top_n=10):
    df = pd.read_csv(PROCESSED_FILE)

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    top_indices = mean_scores.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in top_indices]

    return keywords


# =====================================
# Step 4 – Topic Modeling (LDA)
# =====================================
def generate_topics(n_topics=3):
    df = pd.read_csv(PROCESSED_FILE)

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(df["processed_text"])

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic in lda.components_:
        top_indices = topic.argsort()[-10:][::-1]
        topic_words = [feature_names[i] for i in top_indices]
        topics.append(topic_words)

    return topics


# =====================================
# Step 5 – Sentiment Analysis
# =====================================
def add_sentiment():
    df = pd.read_csv(PROCESSED_FILE)

    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity

        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment"] = df["processed_text"].apply(get_sentiment)

    df.to_csv(FINAL_FILE, index=False)

    return df["sentiment"].value_counts().to_dict()


# =====================================
# Full NLP Pipeline
# =====================================
def run_nlp_pipeline():
    if not os.path.exists(INPUT_FILE):
        return {
            "error": "Milestone 1 data not found. Run news pipeline first."
        }

    count_clean = create_clean_text_dataset()
    count_processed = create_processed_dataset()

    keywords = extract_top_keywords()
    topics = generate_topics()
    sentiment_stats = add_sentiment()

    return {
        "records_processed": count_processed,
        "top_keywords": keywords,
        "topics": topics,
        "sentiment_distribution": sentiment_stats
    }