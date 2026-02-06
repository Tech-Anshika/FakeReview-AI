import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from transformers import DistilBertTokenizerFast
from huggingface_hub import hf_hub_download

from api.schema import ReviewInput, PredictionOutput
from behavior.behavior_engine import calculate_behavior_score

# -----------------------
# App
# -----------------------
app = FastAPI(
    title="Fake Review Detection API",
    description="Hybrid AI (Text + Behavior) Fraud Detection [ONNX Optimized]",
    version="1.1"
)

# -----------------------
# Load Model (ONNX)
# -----------------------
REPO_ID = "Anshikaaaaaaaa/distilbert_fake_review"
FILENAME = "model.onnx"

print("Downloading/Loading ONNX model...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
tokenizer = DistilBertTokenizerFast.from_pretrained(REPO_ID)

# Create ONNX Runtime Session
ort_session = ort.InferenceSession(model_path)

# -----------------------
# Health Check
# -----------------------
@app.get("/")
def health():
    return {"status": "API running (ONNX Optimized) 🚀"}

# -----------------------
# Prediction Endpoint
# -----------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.post("/predict", response_model=PredictionOutput)
def predict(review: ReviewInput):

    # -------- TEXT SCORE --------
    inputs = tokenizer(
        review.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="numpy"
    )

    # Prepare inputs for ONNX
    # Note: Types must typically be int64 for indices
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Run Inference
    logits = ort_session.run(None, ort_inputs)[0]
    
    # Softmax
    probs = softmax(logits[0])
    text_risk = float(probs[1] * 100)  # fake %

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
