import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# Load trained model & tokenizer
MODEL_PATH = "models/distilbert_fake_review"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("🔥 Device in use:", device)

# Load Dataset-2
df = pd.read_csv("data/raw/dataset2_reviews.csv")

# Rename if needed
if "review_text" not in df.columns:
    raise ValueError("❌ 'review_text' column not found in dataset")

reviews = df["review_text"].astype(str).tolist()

predictions = []
confidences = []

# Inference loop
for text in tqdm(reviews, desc="🔍 Testing reviews"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)

    fake_prob = probs[0][1].item()
    label = "Fake" if fake_prob >= 0.5 else "Genuine"

    predictions.append(label)
    confidences.append(round(fake_prob, 4))

# Save results
df["prediction"] = predictions
df["fake_confidence"] = confidences

output_path = "data/processed/dataset2_predictions.csv"
df.to_csv(output_path, index=False)

print("✅ Testing complete")
print(f"📁 Predictions saved at: {output_path}")
