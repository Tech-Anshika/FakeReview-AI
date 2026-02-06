import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from api.schema import ReviewInput, PredictionOutput
from behavior.behavior_engine import calculate_behavior_score

# -----------------------
# App
# -----------------------
app = FastAPI(
    title="Fake Review Detection API",
    description="Hybrid AI (Text + Behavior) Fraud Detection",
    version="1.0"
)

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model
# -----------------------
MODEL_PATH = "models/distilbert_fake_review"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------
# Health Check
# -----------------------
@app.get("/")
def health():
    return {"status": "API running 🚀"}

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(review: ReviewInput):

    # -------- TEXT SCORE --------
    inputs = tokenizer(
        review.text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    text_risk = probs[0][1].item() * 100  # fake %

    # -------- BEHAVIOR SCORE --------
    behavior_score, reasons = calculate_behavior_score(review.dict())

    # -------- FINAL DECISION --------
    if text_risk >= 45 and behavior_score >= 25:
        prediction = "Fake"
        confidence = max(text_risk, behavior_score)
    else:
        prediction = "Genuine"
        confidence = 100 - text_risk

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "text_risk": round(text_risk, 2),
        "behavior_score": behavior_score
    }
