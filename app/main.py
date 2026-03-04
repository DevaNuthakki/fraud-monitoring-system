import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.json")
XTRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Feature columns (exact order used in training)
if os.path.exists(XTRAIN_PATH):
    feature_cols = pd.read_csv(XTRAIN_PATH, nrows=0).columns.tolist()
else:
    # Fallback (Time + V1..V28 + Amount)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

N_FEATURES = len(feature_cols)

# Threshold cache (auto-refresh if threshold.json changes)
# NOTE: your file is: {"fraud_threshold": 0.31}
_threshold_value = 0.5
_threshold_mtime = None

def get_threshold() -> float:
    global _threshold_value, _threshold_mtime

    if not os.path.exists(THRESHOLD_PATH):
        return _threshold_value

    mtime = os.path.getmtime(THRESHOLD_PATH)
    if _threshold_mtime is None or mtime != _threshold_mtime:
        with open(THRESHOLD_PATH, "r") as f:
            cfg = json.load(f)

        # accept either key, but prefer fraud_threshold
        if "fraud_threshold" in cfg:
            _threshold_value = float(cfg["fraud_threshold"])
        else:
            _threshold_value = float(cfg.get("threshold", 0.5))

        _threshold_mtime = mtime

    return _threshold_value

# FastAPI app
app = FastAPI(title="Fraud Monitoring API", version="1.0")

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Exactly {N_FEATURES} features in training order: {feature_cols}",
    )

class PredictResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": N_FEATURES,
        "threshold": get_threshold(),
        "feature_cols_preview": feature_cols[:5],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {N_FEATURES} features, got {len(req.features)}"
        )

    # Make a 1-row DataFrame with correct feature names (removes sklearn warning)
    X_df = pd.DataFrame([req.features], columns=feature_cols)

    proba = float(model.predict_proba(X_df)[0][1])
    threshold = get_threshold()
    pred = 1 if proba >= threshold else 0

    return PredictResponse(
        fraud_probability=proba,
        threshold=threshold,
        prediction=pred,
    )