import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import time
from googlesearch import search

# Load saved models
model = joblib.load('model/xgb_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
scaler = joblib.load('model/scaler.pkl')

# Clean text function (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Sentiment extraction using TextBlob
def get_sentiment_features(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# 🔍 Related Article Search
def find_related_articles(query, max_results=5):
    related_articles = []
    try:
        for url in search(query, num_results=max_results):  # FIXED: num_results replaces 'stop'
            related_articles.append(url)
            time.sleep(1)  # prevent rate-limiting
    except Exception as e:
        related_articles.append(f"Error searching for articles: {str(e)}")
    return related_articles

# 🎯 Main Predict Function
def predict_news(text):
    cleaned = clean_text(text)
    polarity, subjectivity = get_sentiment_features(cleaned)

    tfidf_features = vectorizer.transform([cleaned])
    numerical_scaled = scaler.transform([[polarity, subjectivity]])

    # Ensure dimensions match what model expects
    expected_shape = model.n_features_in_
    if tfidf_features.shape[1] < (expected_shape - 2):
        padding = csr_matrix((1, (expected_shape - 2) - tfidf_features.shape[1]))
        tfidf_features = hstack((tfidf_features, padding))
    elif tfidf_features.shape[1] > (expected_shape - 2):
        tfidf_features = tfidf_features[:, : (expected_shape - 2)]

    combined_features = hstack((tfidf_features, numerical_scaled))

    prediction = model.predict(combined_features)[0]
    confidence = model.predict_proba(combined_features)[0][1]

    result = "Real News" if prediction == 1 else "Fake News"
    confidence_score = f"{confidence * 100:.2f}%"
    related = find_related_articles(text) if result == "Real News" else []

    return {
        "result": result,
        "confidence": confidence_score,
        "related_articles": related
    }
