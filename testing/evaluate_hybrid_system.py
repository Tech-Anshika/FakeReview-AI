import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from behavior.behavior_engine import calculate_behavior_score

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", device)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "models/distilbert_fake_review"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(
    "data/raw/final_labeled_fake_reviews.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

df = df[
    ["text", "label", "rating", "verified_purchase", "helpful_vote", "user_review_burst"]
].dropna()

df["label"] = df["label"].astype(int)

print("📊 Samples:", len(df))

# -----------------------------
# Hybrid scoring
# -----------------------------
predictions = []
true_labels = df["label"].tolist()

for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Hybrid Evaluation"):

    # TEXT SCORE
    inputs = tokenizer(
        row["text"],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        text_risk = probs[0][1].item() * 100  # fake probability %

    # BEHAVIOR SCORE
    behavior_score, reasons = calculate_behavior_score(row)

    # ✅ HYBRID DECISION (TEXT FIRST, BEHAVIOR CONFIRM)
    if text_risk >= 40 and behavior_score >= 20:
        predictions.append(1)   # Fake
    else:
        predictions.append(0)   # Genuine

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(true_labels, predictions)

print("\n✅ HYBRID SYSTEM RESULTS")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions, target_names=["Genuine", "Fake"]))
