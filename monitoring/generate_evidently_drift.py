import os
import json
import uuid
import joblib
import pandas as pd
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_PATH = os.path.join(BASE_DIR, "data", "baseline_reference.csv")
PREDICTIONS_LOG_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")
XTRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

CALIBRATOR_PATH = os.path.join(BASE_DIR, "model", "calibrator.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EVIDENTLY_HTML_PATH = os.path.join(REPORTS_DIR, "evidently_drift_report.html")
DRIFT_SUMMARY_PATH = os.path.join(REPORTS_DIR, "drift_summary.json")
ALERTS_PATH = os.path.join(REPORTS_DIR, "alerts.json")

os.makedirs(REPORTS_DIR, exist_ok=True)


# =========================
# Alert helpers
# =========================
def load_existing_alerts():
    if os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH, "r") as f:
            return json.load(f)
    return []


def save_alerts(alerts):
    with open(ALERTS_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


def get_alert_severity(drift_share: float) -> str:
    if drift_share >= 0.8:
        return "high"
    elif drift_share >= 0.5:
        return "medium"
    elif drift_share >= 0.2:
        return "low"
    return "none"


def create_alert(metric: str, value: float, threshold: float, severity: str, message: str):
    return {
        "id": str(uuid.uuid4()),
        "status": "open",
        "severity": severity,
        "metric": metric,
        "value": round(float(value), 6),
        "threshold": round(float(threshold), 6),
        "timestamp": datetime.utcnow().isoformat(),
        "message": message
    }


# =========================
# Load model
# =========================
if os.path.exists(CALIBRATOR_PATH):
    serving_model = joblib.load(CALIBRATOR_PATH)
elif os.path.exists(MODEL_PATH):
    serving_model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError("No model found")


# =========================
# Feature columns
# =========================
feature_cols = pd.read_csv(XTRAIN_PATH, nrows=0).columns.tolist()


# =========================
# Load data
# =========================
baseline_df = pd.read_csv(BASELINE_PATH)
live_log_df = pd.read_csv(PREDICTIONS_LOG_PATH)

if baseline_df.empty:
    raise ValueError("Baseline dataset is empty")

if live_log_df.empty:
    raise ValueError("Predictions log is empty")


# =========================
# Build current/live feature dataframe from logs
# =========================
live_feature_data = {}
for col in feature_cols:
    log_col = f"input_{col}"
    if log_col not in live_log_df.columns:
        raise ValueError(f"Missing logged column: {log_col}")
    live_feature_data[col] = live_log_df[log_col]

current_df = pd.DataFrame(live_feature_data)


# =========================
# Add prediction probability to both datasets
# so Evidently also checks prediction drift
# =========================
baseline_df = baseline_df[feature_cols].copy()
current_df = current_df[feature_cols].copy()

baseline_df["predicted_probability"] = serving_model.predict_proba(baseline_df)[:, 1]
current_df["predicted_probability"] = live_log_df["predicted_probability"].astype(float)


# =========================
# Evidently report
# =========================
report = Report([
    DataDriftPreset()
])

my_eval = report.run(
    current_data=current_df,
    reference_data=baseline_df
)

my_eval.save_html(EVIDENTLY_HTML_PATH)

report_dict = my_eval.dict()


# =========================
# Compute simple drift score
# =========================

number_of_columns = len(current_df.columns)

drifted_columns = []

for col in current_df.columns:

    ref_mean = baseline_df[col].mean()
    cur_mean = current_df[col].mean()

    if abs(ref_mean - cur_mean) > 0.01:
        drifted_columns.append(col)

number_of_drifted_columns = len(drifted_columns)

share_of_drifted_columns = number_of_drifted_columns / number_of_columns

dataset_drift = share_of_drifted_columns > 0.2


# =========================
# Create alerts automatically
# =========================
alerts = load_existing_alerts()

severity = get_alert_severity(share_of_drifted_columns)

if severity != "none":
    alert = create_alert(
        metric="evidently_drift_share",
        value=share_of_drifted_columns,
        threshold=0.2,
        severity=severity,
        message=(
            f"Evidently detected dataset drift. "
            f"{number_of_drifted_columns} out of {number_of_columns} columns drifted "
            f"({share_of_drifted_columns:.3f} drift share)."
        )
    )
    alerts.append(alert)

save_alerts(alerts)


# =========================
# Save drift summary
# =========================
summary = {
    "status": "ok",
    "baseline_rows": int(len(baseline_df)),
    "current_rows": int(len(current_df)),
    "columns_checked": list(current_df.columns),
    "number_of_columns": number_of_columns,
    "number_of_drifted_columns": number_of_drifted_columns,
    "share_of_drifted_columns": share_of_drifted_columns,
    "dataset_drift": dataset_drift,
    "report_path": EVIDENTLY_HTML_PATH,
    "alerts_path": ALERTS_PATH
}

with open(DRIFT_SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved:", EVIDENTLY_HTML_PATH)
print("Saved:", DRIFT_SUMMARY_PATH)
print("Saved:", ALERTS_PATH)