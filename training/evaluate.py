import joblib
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/cleaned_reviews.csv")

X = df['clean_review']
y = df['label_encoded']

vectorizer = joblib.load("models/tfidf.pkl")
model = joblib.load("models/fake_review_ml.pkl")

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

print(classification_report(y, y_pred, target_names=["Genuine", "Fake"]))
