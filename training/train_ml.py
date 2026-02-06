import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def train_model():
    df = pd.read_csv("data/processed/cleaned_reviews.csv")

    X = df['clean_review']
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model & vectorizer
    joblib.dump(model, "models/fake_review_ml.pkl")
    joblib.dump(vectorizer, "models/tfidf.pkl")

    print("✅ Model training complete!")
    print("📁 Saved files:")
    print(" - models/fake_review_ml.pkl")
    print(" - models/tfidf.pkl")

if __name__ == "__main__":
    train_model()
