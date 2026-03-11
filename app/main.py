import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "model", "calibrator.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "threshold.json")

# Use processed data header to get the exact feature order used in training
XTRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

# Optional model metadata (versioning)
MODEL_META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

# Prediction log file
PREDICTIONS_LOG_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")

# Drift monitoring files
BASELINE_PATH = os.path.join(BASE_DIR, "data", "baseline_reference.csv")
DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "data", "drift_report.json")

# =========================
# Load serving artifact
# Prefer calibrator if it exists, else fall back to base model
# =========================
serving_model = None
serving_model_file = None

if os.path.exists(CALIBRATOR_PATH):
    serving_model = joblib.load(CALIBRATOR_PATH)
    serving_model_file = os.path.basename(CALIBRATOR_PATH)
elif os.path.exists(MODEL_PATH):
    serving_model = joblib.load(MODEL_PATH)
    serving_model_file = os.path.basename(MODEL_PATH)
else:
    raise FileNotFoundError(
        f"No serving model found. Checked:\n- {CALIBRATOR_PATH}\n- {MODEL_PATH}"
    )

# =========================
# Feature columns (training order)
# =========================
if os.path.exists(XTRAIN_PATH):
    feature_cols = pd.read_csv(XTRAIN_PATH, nrows=0).columns.tolist()
else:
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

N_FEATURES = len(feature_cols)

# =========================
# Threshold cache (auto-refresh if file changes)
# IMPORTANT: threshold.json key is "fraud_threshold"
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
    meta = {
        "model_version": os.getenv("MODEL_VERSION", "unknown"),
        "model_file": serving_model_file,
        "threshold_file": os.path.basename(THRESHOLD_PATH),
    }

    if os.path.exists(MODEL_META_PATH):
        try:
            with open(MODEL_META_PATH, "r") as f:
                file_meta = json.load(f)
            if isinstance(file_meta, dict):
                meta.update(file_meta)
        except Exception:
            pass

    return meta


# =========================
# Prediction logging
# =========================
def log_prediction(features: List[float], predicted_probability: float, predicted_class: int):
    os.makedirs(os.path.dirname(PREDICTIONS_LOG_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_probability": round(float(predicted_probability), 6),
        "predicted_class": int(predicted_class),
    }

    # Add feature values with actual feature names
    for col_name, value in zip(feature_cols, features):
        log_entry[f"input_{col_name}"] = value

    log_df = pd.DataFrame([log_entry])

    if os.path.exists(PREDICTIONS_LOG_PATH):
        log_df.to_csv(PREDICTIONS_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTIONS_LOG_PATH, mode="w", header=True, index=False)


# =========================
# Drift monitoring
# =========================
def compute_feature_drift_score(baseline_df: pd.DataFrame, live_df: pd.DataFrame) -> float:
    """
    Compute feature drift using standardized mean difference:
    abs(live_mean - baseline_mean) / baseline_std
    """
    scores = []

    for col in feature_cols:
        if col in baseline_df.columns and col in live_df.columns:
            baseline_mean = float(baseline_df[col].mean())
            live_mean = float(live_df[col].mean())
            baseline_std = float(baseline_df[col].std())

            score = abs(live_mean - baseline_mean) / (baseline_std + 1e-6)
            scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def compute_prediction_drift_score(
    baseline_pred_probs: pd.Series,
    live_pred_probs: pd.Series
) -> float:
    """
    Compute prediction drift using standardized difference in prediction means.
    """
    if baseline_pred_probs.empty or live_pred_probs.empty:
        return 0.0

    baseline_mean = float(baseline_pred_probs.mean())
    live_mean = float(live_pred_probs.mean())
    baseline_std = float(baseline_pred_probs.std())

    return float(abs(live_mean - baseline_mean) / (baseline_std + 1e-6))


def generate_drift_report() -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {
            "status": "error",
            "message": f"Baseline file not found at {BASELINE_PATH}"
        }

    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return {
            "status": "error",
            "message": f"Predictions log not found at {PREDICTIONS_LOG_PATH}"
        }

    baseline_df = pd.read_csv(BASELINE_PATH)
    live_log_df = pd.read_csv(PREDICTIONS_LOG_PATH)

    if baseline_df.empty:
        return {
            "status": "error",
            "message": "Baseline dataset is empty"
        }

    if live_log_df.empty:
        return {
            "status": "error",
            "message": "No live predictions available yet"
        }

    # Rebuild live feature dataframe from logged input columns
    live_feature_data = {}
    for col in feature_cols:
        logged_col = f"input_{col}"
        if logged_col in live_log_df.columns:
            live_feature_data[col] = live_log_df[logged_col]

    live_feature_df = pd.DataFrame(live_feature_data)

    if live_feature_df.empty:
        return {
            "status": "error",
            "message": "Live feature data could not be reconstructed from prediction logs"
        }

    # Ensure correct column order
    baseline_features = baseline_df[feature_cols]
    live_feature_df = live_feature_df[feature_cols]

    # Baseline prediction probabilities
    baseline_pred_probs = pd.Series(
        serving_model.predict_proba(baseline_features)[:, 1]
    )

    # Live prediction probabilities from logs
    if "predicted_probability" not in live_log_df.columns:
        return {
            "status": "error",
            "message": "predicted_probability column not found in predictions log"
        }

    live_pred_probs = pd.Series(live_log_df["predicted_probability"])

    feature_drift = compute_feature_drift_score(baseline_features, live_feature_df)
    prediction_drift = compute_prediction_drift_score(baseline_pred_probs, live_pred_probs)
    overall_drift = (feature_drift + prediction_drift) / 2.0

    report = {
        "status": "ok",
        "baseline_rows": int(len(baseline_df)),
        "live_rows": int(len(live_log_df)),
        "feature_drift_score": round(float(feature_drift), 6),
        "prediction_drift_score": round(float(prediction_drift), 6),
        "overall_drift_score": round(float(overall_drift), 6),
        "drift_flag": bool(overall_drift > 1.0),
        "drift_method": {
            "feature_drift": "standardized_mean_difference",
            "prediction_drift": "standardized_prediction_mean_difference"
        }
    }

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Fraud Monitoring API", version="1.1")

# =========================
# In-memory stats aggregator
# =========================
STATS = {
    "total_predictions": 0,
    "fraud_predictions": 0,
    "sum_probability": 0.0,
    "last_proba": None,
    "last_prediction": None,
    "last_latency_ms": None,
    "sum_latency_ms": 0.0,
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
# Request latency tracking
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
    prediction: int


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
        "serving_model_file": serving_model_file,
    }


@app.get("/model-info")
def model_info():
    meta = get_model_meta()
    meta.update(
        {
            "n_features": N_FEATURES,
            "threshold": get_threshold(),
            "serving_model_file": serving_model_file,
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

    X_df = pd.DataFrame([req.features], columns=feature_cols)

    start = time.perf_counter()
    proba = float(serving_model.predict_proba(X_df)[0][1])
    infer_ms = (time.perf_counter() - start) * 1000.0

    pred = 1 if proba >= threshold else 0

    # Log prediction to CSV
    log_prediction(req.features, proba, pred)

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


@app.get("/drift")
def drift():
    return generate_drift_report()


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
    drift_data = generate_drift_report()

    drift_status = drift_data.get("status", "unknown")
    feature_drift = drift_data.get("feature_drift_score", "N/A")
    prediction_drift = drift_data.get("prediction_drift_score", "N/A")
    overall_drift = drift_data.get("overall_drift_score", "N/A")
    drift_flag = drift_data.get("drift_flag", "N/A")

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
          <p>Serving artifact: <code>{serving_model_file}</code></p>
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
          <div class="big">Drift Monitoring</div>
          <p>Status: <code>{drift_status}</code></p>
          <p>Feature Drift Score: <code>{feature_drift}</code></p>
          <p>Prediction Drift Score: <code>{prediction_drift}</code></p>
          <p>Overall Drift Score: <code>{overall_drift}</code></p>
          <p>Drift Flag: <code>{drift_flag}</code></p>
        </div>

        <div class="card">
          <div class="big">Useful Links</div>
          <ul>
            <li><a href="/docs">Swagger Docs</a></li>
            <li><a href="/stats">Stats (JSON)</a></li>
            <li><a href="/model-info">Model Info (JSON)</a></li>
            <li><a href="/drift">Drift Report (JSON)</a></li>
          </ul>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
