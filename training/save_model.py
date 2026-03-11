import os
import json
import time
import pandas as pd
from joblib import dump

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ----------------------------
# Config
# ----------------------------
THRESHOLD = 0.70
DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("model")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "calibrator.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def load_processed():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("Loading processed data...")
    X_train_full, y_train_full, X_test, y_test = load_processed()
    print(f"Train full: {X_train_full.shape}  Test: {X_test.shape}")

    neg = (y_train_full == 0).sum()
    pos = (y_train_full == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print("\nTraining calibrated XGBoost model...")
    start = time.time()

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )
    calibrator.fit(X_train_full, y_train_full)

    train_time = time.time() - start
    print(f"Calibrated model training time: {train_time:.2f} sec")

    print(f"\nEvaluating calibrated model with threshold = {THRESHOLD}")
    y_prob = calibrator.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob, THRESHOLD)

    print("\n=== Final Calibrated Model Metrics (Test) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save calibrator as main serving artifact
    dump(calibrator, CALIBRATOR_PATH)

    # Optional: save threshold
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"fraud_threshold": THRESHOLD}, f, indent=2)

    # Metadata
    with open(MODEL_META_PATH, "w") as f:
        json.dump(
            {
                "model_version": "xgboost-calibrated-v1",
                "model_type": "XGBClassifier + CalibratedClassifierCV",
                "calibration_method": "sigmoid",
                "training_time_sec": round(train_time, 2),
                "threshold": THRESHOLD,
                "feature_count": int(X_train_full.shape[1]),
                "feature_names": X_train_full.columns.tolist(),
            },
            f,
            indent=2,
        )

    print("\nSaved files:")
    print(" -", CALIBRATOR_PATH)
    print(" -", THRESHOLD_PATH)
    print(" -", MODEL_META_PATH)


if __name__ == "__main__":
    main()