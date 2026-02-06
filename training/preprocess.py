import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Rename column
    df.rename(columns={'text_': 'review_text'}, inplace=True)

    # Clean text
    df['clean_review'] = df['review_text'].apply(clean_text)

    # Encode labels
    df['label_encoded'] = df['label'].map({'CG': 0, 'OR': 1})

    # ❗ DROP EMPTY / INVALID ROWS (THIS FIXES ERROR)
    df = df.dropna(subset=['clean_review', 'label_encoded'])
    df = df[df['clean_review'].str.strip() != ""]

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed data
    df[['clean_review', 'label_encoded']].to_csv(output_path, index=False)

    print("✅ Preprocessing complete!")
    print(f"📁 Saved file at: {output_path}")
    print(f"📊 Final dataset size: {len(df)} rows")

if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/fake_reviews.csv",
        output_path="data/processed/cleaned_reviews.csv"
    )
