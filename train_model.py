import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
from scipy.sparse import hstack

# Load the Dataset
real_df = pd.read_csv('real_news.csv')
fake_df = pd.read_csv('fake_news.csv')

real_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([real_df, fake_df], ignore_index=True)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply cleaning
df['clean_title'] = df['title'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)
X_text = vectorizer.fit_transform(df['clean_title'])

# Numerical features
X_numerical = df[['polarity', 'subjectivity']].values
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Combine features
X_combined = hstack((X_text, X_numerical_scaled))
y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    eval_metric='logloss',
    use_label_encoder=False
)

# Train
xgb_model.fit(X_train, y_train)

# Evaluate (Optional)
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model, vectorizer, scaler
joblib.dump(xgb_model, 'model/xgb_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
