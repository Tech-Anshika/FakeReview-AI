import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
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
    try:
        # SAFE DEFAULTS & LOGGING (Prevent crash on None types)
        text_content = review.text or ""
        
        # -------- TEXT SCORE (Softmax ONNX) --------
        # Handle empty text to avoid model crash
        if not text_content.strip():
            text_risk = 0.0
        else:
            inputs = tokenizer(
                text_content,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="np"
            )

            # Prepare inputs for ONNX
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }

            # Run Inference
            logits = ort_session.run(None, ort_inputs)[0]
            
            # Softmax
            probs = softmax(logits[0])
            text_risk = float(probs[1] * 100)

        # -------- BEHAVIOR SCORE --------
        # Ensure behavior scoring doesn't crash on missing optional fields
        review_dict = review.dict()
        # safe defaults for behavior calc might be handled inside, but good to be sure.
        # Pydantic dict() includes None for missing fields. 
        behavior_score, reasons = calculate_behavior_score(review_dict)

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

    except Exception as e:
        print(f"🔥 PREDICTION ERROR: {str(e)}")
        # Return a 500 with clear message instead of crashing
        raise HTTPException(
            status_code=500,
            detail=f"Internal Processing Error: {str(e)}"
        )
