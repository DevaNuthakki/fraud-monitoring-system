import os
import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

# Paths
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print("\n==============================")
    print(f"Training: {name}")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Probabilities for AUC metrics
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    # Default threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Training time: {train_time:.2f} seconds")
    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)
    print("Confusion Matrix [TN FP FN TP]:", tn, fp, fn, tp)

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "training_time_sec": train_time,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

def main():
    # Load processed data
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")

    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel().astype(int)
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel().astype(int)

    print("✅ Data Loaded")
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # Ensure reports folder exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Class imbalance ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.4f}")

    # Models
    models = [
        (
            "Logistic Regression (balanced)",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear"
            ),
        ),
        (
            "Random Forest (300 trees, balanced)",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            ),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                random_state=42
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
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
            ),
        ),
    ]

    results = []
    for name, model in models:
        row = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(row)

    # Leaderboard
    df = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)

    out_csv = f"{REPORTS_DIR}/model_comparison.csv"
    df.to_csv(out_csv, index=False)

    print("\n==============================")
    print("MODEL COMPARISON (sorted by PR-AUC)")
    print("==============================")
    print(df[["model", "roc_auc", "pr_auc", "training_time_sec", "fp", "fn", "tp"]].to_string(index=False))
    print(f"\n✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()