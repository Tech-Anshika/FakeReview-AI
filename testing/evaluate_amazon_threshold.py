import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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

df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

print("📊 Total test samples:", len(df))

# -----------------------------
# Threshold-based inference
# -----------------------------
THRESHOLD = 0.4   # ⭐ KEY CHANGE

predictions = []
true_labels = df["label"].tolist()

with torch.no_grad():
    for text in tqdm(df["text"], desc="🔍 Threshold Testing"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

        fake_prob = probs[0][1].item()

        pred = 1 if fake_prob >= THRESHOLD else 0
        predictions.append(pred)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\n✅ AMAZON DATASET – THRESHOLD EVALUATION RESULTS")
print(f"Threshold : {THRESHOLD}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

print("\n📊 Classification Report:")
print(classification_report(true_labels, predictions, target_names=["Genuine", "Fake"]))
