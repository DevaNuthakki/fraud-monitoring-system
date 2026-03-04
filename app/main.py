import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.json")

# Use processed data header to get the exact feature order used in training
XTRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

# Optional model metadata (versioning)
MODEL_META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# =========================
# Feature columns (training order)
# =========================
if os.path.exists(XTRAIN_PATH):
    feature_cols = pd.read_csv(XTRAIN_PATH, nrows=0).columns.tolist()
else:
    # Fallback (should match your processed X order)
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

N_FEATURES = len(feature_cols)

# =========================
# Threshold cache (auto-refresh if file changes)
# IMPORTANT: your threshold.json key is "fraud_threshold"
# =========================
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

        # ✅ handle both keys: "fraud_threshold" (your file) and "threshold" (fallback)
        if "fraud_threshold" in cfg:
            _threshold_value = float(cfg["fraud_threshold"])
        else:
            _threshold_value = float(cfg.get("threshold", 0.5))

        _threshold_mtime = mtime

    return _threshold_value


# =========================
# Model metadata / version (optional)
# =========================
def get_model_meta() -> dict:
    """
    Priority:
    1) model/model_meta.json (if exists)
    2) ENV MODEL_VERSION (if set)
    3) fallback defaults
    """
    meta = {
        "model_version": os.getenv("MODEL_VERSION", "unknown"),
        "model_file": os.path.basename(MODEL_PATH),
        "threshold_file": os.path.basename(THRESHOLD_PATH),
    }

    if os.path.exists(MODEL_META_PATH):
        try:
            with open(MODEL_META_PATH, "r") as f:
                file_meta = json.load(f)
            if isinstance(file_meta, dict):
                meta.update(file_meta)
        except Exception:
            # keep defaults if metadata is malformed
            pass

    return meta


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Fraud Monitoring API", version="1.0")

# =========================
# In-memory stats aggregator
# (good for local/demo; for Cloud Run you'd move to Redis/DB)
# =========================
STATS = {
    "total_predictions": 0,
    "fraud_predictions": 0,
    "sum_probability": 0.0,
    "last_proba": None,
    "last_prediction": None,
    "last_latency_ms": None,      # latency for last /predict
    "sum_latency_ms": 0.0,        # sum of /predict latencies
}


def reset_stats():
    STATS["total_predictions"] = 0
    STATS["fraud_predictions"] = 0
    STATS["sum_probability"] = 0.0
    STATS["last_proba"] = None
    STATS["last_prediction"] = None
    STATS["last_latency_ms"] = None
    STATS["sum_latency_ms"] = 0.0


# =========================
# Request latency tracking (middleware)
# Adds header: X-Process-Time-ms
# =========================
@app.middleware("http")
async def add_latency_headers(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


# =========================
# Schemas
# =========================
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=N_FEATURES,
        max_items=N_FEATURES,
        description=f"Exactly {N_FEATURES} numeric features in the same order as training: {feature_cols}",
    )


class PredictResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: int  # 0 or 1


class StatsResponse(BaseModel):
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    avg_probability: float
    threshold: float
    avg_latency_ms: float
    last_latency_ms: Optional[float]
    last_probability: Optional[float]
    last_prediction: Optional[int]


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_features": N_FEATURES,
        "threshold": get_threshold(),
        "feature_cols_preview": feature_cols[:5],
    }


@app.get("/model-info")
def model_info():
    meta = get_model_meta()
    meta.update(
        {
            "n_features": N_FEATURES,
            "threshold": get_threshold(),
        }
    )
    return meta


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {N_FEATURES} features, got {len(req.features)}",
        )

    threshold = get_threshold()

    # Create DataFrame with correct column names (fixes sklearn warning)
    X_df = pd.DataFrame([req.features], columns=feature_cols)

    # Track prediction latency separately (model inference only)
    start = time.perf_counter()
    proba = float(model.predict_proba(X_df)[0][1])
    infer_ms = (time.perf_counter() - start) * 1000.0

    pred = 1 if proba >= threshold else 0

    # Update in-memory stats
    STATS["total_predictions"] += 1
    STATS["fraud_predictions"] += int(pred == 1)
    STATS["sum_probability"] += proba
    STATS["last_proba"] = proba
    STATS["last_prediction"] = pred
    STATS["last_latency_ms"] = infer_ms
    STATS["sum_latency_ms"] += infer_ms

    return PredictResponse(
        fraud_probability=proba,
        threshold=threshold,
        prediction=pred,
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    total = STATS["total_predictions"]
    fraud = STATS["fraud_predictions"]
    threshold = get_threshold()

    if total == 0:
        return StatsResponse(
            total_predictions=0,
            fraud_predictions=0,
            fraud_rate=0.0,
            avg_probability=0.0,
            threshold=threshold,
            avg_latency_ms=0.0,
            last_latency_ms=None,
            last_probability=None,
            last_prediction=None,
        )

    avg_prob = STATS["sum_probability"] / total
    fraud_rate = fraud / total
    avg_latency = STATS["sum_latency_ms"] / total

    return StatsResponse(
        total_predictions=total,
        fraud_predictions=fraud,
        fraud_rate=float(fraud_rate),
        avg_probability=float(avg_prob),
        threshold=float(threshold),
        avg_latency_ms=float(avg_latency),
        last_latency_ms=STATS["last_latency_ms"],
        last_probability=STATS["last_proba"],
        last_prediction=STATS["last_prediction"],
    )


@app.post("/reset-stats")
def reset_stats_endpoint():
    reset_stats()
    return {"status": "ok", "message": "Stats reset"}


# =========================
# Simple monitoring dashboard (HTML)
# =========================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    total = STATS["total_predictions"]
    fraud = STATS["fraud_predictions"]
    threshold = get_threshold()

    avg_prob = (STATS["sum_probability"] / total) if total > 0 else 0.0
    fraud_rate = (fraud / total) if total > 0 else 0.0
    avg_latency = (STATS["sum_latency_ms"] / total) if total > 0 else 0.0

    last_proba = STATS["last_proba"]
    last_pred = STATS["last_prediction"]
    last_latency = STATS["last_latency_ms"]

    model_meta = get_model_meta()

    html = f"""
    <html>
      <head>
        <title>Fraud Monitoring Dashboard</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
          .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
          .big {{ font-size: 24px; font-weight: bold; }}
          .muted {{ color: #666; }}
          code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }}
        </style>
      </head>
      <body>
        <h1>Fraud Monitoring Dashboard</h1>
        <p class="muted">Refresh the page after making predictions.</p>

        <div class="card">
          <div class="big">Model</div>
          <p>Version: <code>{model_meta.get("model_version", "unknown")}</code></p>
          <p>Model file: <code>{model_meta.get("model_file")}</code></p>
          <p>Threshold: <code>{threshold}</code></p>
        </div>

        <div class="grid">
          <div class="card">
            <div class="big">{total}</div>
            <div class="muted">Total Predictions</div>
          </div>

          <div class="card">
            <div class="big">{fraud}</div>
            <div class="muted">Fraud Predictions</div>
          </div>

          <div class="card">
            <div class="big">{fraud_rate:.4f}</div>
            <div class="muted">Fraud Rate</div>
          </div>

          <div class="card">
            <div class="big">{avg_prob:.4f}</div>
            <div class="muted">Avg Fraud Probability</div>
          </div>

          <div class="card">
            <div class="big">{avg_latency:.2f} ms</div>
            <div class="muted">Avg Inference Latency</div>
          </div>

          <div class="card">
            <div class="big">{(last_latency if last_latency is not None else 0):.2f} ms</div>
            <div class="muted">Last Inference Latency</div>
          </div>
        </div>

        <div class="card">
          <div class="big">Last Prediction</div>
          <p>Probability: <code>{last_proba if last_proba is not None else "N/A"}</code></p>
          <p>Prediction: <code>{last_pred if last_pred is not None else "N/A"}</code></p>
        </div>

        <div class="card">
          <div class="big">Useful Links</div>
          <ul>
            <li><a href="/docs">Swagger Docs</a></li>
            <li><a href="/stats">Stats (JSON)</a></li>
            <li><a href="/model-info">Model Info (JSON)</a></li>
          </ul>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)