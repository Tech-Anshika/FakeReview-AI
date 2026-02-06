import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", device)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/distilbert_fake_review"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------------
# Load Amazon LABELED dataset
# -----------------------------
df = pd.read_csv(
    "data/raw/final_labeled_fake_reviews.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

print("📄 Columns found:", df.columns.tolist())

# -----------------------------
# Select required columns
# -----------------------------
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

print("📊 Total test samples:", len(df))
print(df.head())

# -----------------------------
# Inference
# -----------------------------
predictions = []
true_labels = df["label"].tolist()

with torch.no_grad():
    for text in tqdm(df["text"], desc="🔍 Testing Amazon reviews"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\n✅ AMAZON DATASET – FINAL EVALUATION RESULTS")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\n📊 Detailed Classification Report:\n")
print(classification_report(true_labels, predictions, target_names=["Genuine", "Fake"]))
