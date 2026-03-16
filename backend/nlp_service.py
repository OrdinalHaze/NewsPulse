# backend/nlp_service.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

DATA_DIR          = "data"
INPUT_FILE        = os.path.join(DATA_DIR, "news_data_cleaned.csv")
CLEANED_TEXT_FILE = os.path.join(DATA_DIR, "news_cleaned_text.csv")
PROCESSED_FILE    = os.path.join(DATA_DIR, "processed_news.csv")

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
vader     = SentimentIntensityAnalyzer()


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
# VADER sentiment (better than TextBlob for news)
# ==============================
def vader_sentiment(text: str) -> str:
    scores = vader.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def vader_score(text: str) -> float:
    return vader.polarity_scores(str(text))["compound"]


# ==============================
# ML sentiment using Logistic Regression on TF-IDF
# ==============================
def train_ml_sentiment(df):
    """
    Train a Logistic Regression classifier on VADER labels.
    Returns (model, vectorizer, accuracy).
    """
    texts  = df["processed_text"].fillna("").tolist()
    labels = df["vader_label"].tolist()

    # Need at least 3 samples per class
    label_counts = pd.Series(labels).value_counts()
    valid_labels = label_counts[label_counts >= 3].index.tolist()
    mask   = [l in valid_labels for l in labels]
    texts  = [t for t, m in zip(texts, mask) if m]
    labels = [l for l, m in zip(labels, mask) if m]

    if len(set(labels)) < 2 or len(texts) < 10:
        print("[WARN] Not enough data for ML training, using VADER only.")
        return None, None, None

    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X   = vec.fit_transform(texts)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    preds    = clf.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds) * 100, 2)
    print(f"[INFO] ML Sentiment Accuracy: {accuracy}%")

    return clf, vec, accuracy


# ==============================
# Build cleaned text file
# ==============================
def create_cleaned_text_file():
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

    # ✅ preserve URL, Date, Topic columns
    for col in ["URL", "Date", "Topic", "Source"]:
        if col not in df.columns:
            df[col] = ""

    df.to_csv(CLEANED_TEXT_FILE, index=False)
    return CLEANED_TEXT_FILE


# ==============================
# Build processed dataset
# ==============================
def create_processed_dataset():
    if not os.path.exists(CLEANED_TEXT_FILE):
        create_cleaned_text_file()
    df = pd.read_csv(CLEANED_TEXT_FILE)
    df["processed_text"] = df["cleaned_text"]

    # ✅ ensure URL, Date, Topic preserved
    for col in ["URL", "Date", "Topic", "Source"]:
        if col not in df.columns:
            df[col] = ""

    df.to_csv(PROCESSED_FILE, index=False)
    return PROCESSED_FILE


# ==============================
# TF-IDF keywords
# ==============================
def extract_top_keywords(df, top_n=50):
    texts = df["processed_text"].fillna("").tolist()
    texts = [t for t in texts if len(t.strip()) > 5]
    if not texts:
        return []
    vec  = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X    = vec.fit_transform(texts)
    feat = vec.get_feature_names_out()
    mean = X.mean(axis=0).A1
    idx  = mean.argsort()[::-1]
    return [feat[i] for i in idx[:top_n]]


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
    feat   = vec.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        terms = [feat[i] for i in comp.argsort()[-n_words:][::-1]]
        topics.append(terms)
    return topics


# ==============================
# Full NLP pipeline
# ==============================
def run_nlp_pipeline():
    try:
        processed_path = create_processed_dataset()
        df = pd.read_csv(processed_path)

        if df.empty or "processed_text" not in df.columns:
            return {"error": "Processed dataset is empty."}

        # Step 1: VADER sentiment labels
        df["vader_label"] = df["processed_text"].fillna("").apply(vader_sentiment)
        df["vader_score"] = df["processed_text"].fillna("").apply(vader_score)

        # Step 2: Train ML model on VADER labels
        clf, vec, ml_accuracy = train_ml_sentiment(df)

        # Step 3: Apply ML labels if model trained successfully
        if clf is not None:
            X_all = vec.transform(df["processed_text"].fillna(""))
            df["sentiment_label"] = clf.predict(X_all)
            df["ml_trained"] = True
        else:
            df["sentiment_label"] = df["vader_label"]
            df["ml_trained"] = False
            ml_accuracy = None

        # Step 4: keywords + topics
        top_keywords = extract_top_keywords(df, top_n=50)
        topics       = compute_topics(df, n_topics=4)

        sentiment_distribution = df["sentiment_label"].value_counts().to_dict()

        # ✅ save with all columns including URL, Date, Topic
        df.to_csv(processed_path, index=False)

        return {
            "records_processed":      len(df),
            "top_keywords":           top_keywords,
            "topics":                 topics,
            "sentiment_distribution": sentiment_distribution,
            "ml_accuracy":            ml_accuracy,
            "ml_trained":             clf is not None,
        }

    except Exception as e:
        print(f"[ERROR] NLP pipeline: {e}")
        return {"error": str(e)}