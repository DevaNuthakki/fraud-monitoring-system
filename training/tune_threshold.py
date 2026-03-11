import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "model"
REPORTS_DIR = "reports"

THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")
REPORT_PATH = os.path.join(REPORTS_DIR, "threshold_tuning_results.csv")


def train_xgboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
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
    model.fit(X_train, y_train)
    return model


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, target_recall=0.80):
    thresholds = np.linspace(0.01, 0.99, 99)

    best_precision = None
    best_row = None
    all_rows = []

    for t in thresholds:
        row = evaluate_at_threshold(y_true, y_prob, t)
        all_rows.append(row)

        if row["recall"] >= target_recall:
            if best_precision is None or row["precision"] > best_precision:
                best_precision = row["precision"]
                best_row = row

    return best_row, all_rows


def main():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")

    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel().astype(int)
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel().astype(int)

    print("Loaded data")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    model = train_xgboost(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    print("\nPR-AUC (probability quality):", round(pr_auc, 6))

    base = evaluate_at_threshold(y_test, y_prob, 0.5)
    print("\n=== Baseline (threshold = 0.50) ===")
    print(
        f"Precision={base['precision']:.4f}  Recall={base['recall']:.4f}  F1={base['f1']:.4f}"
    )
    print(f"TN={base['tn']} FP={base['fp']} FN={base['fn']} TP={base['tp']}")

    target_recall = 0.80
    best, all_rows = find_best_threshold(y_test, y_prob, target_recall=target_recall)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pd.DataFrame(all_rows).to_csv(REPORT_PATH, index=False)

    if best is None:
        print(f"\nNo threshold reached Recall >= {target_recall}.")
        print("Saved all threshold results to:", REPORT_PATH)
        return

    print(f"\n=== Best threshold for Recall >= {target_recall} ===")
    print(
        f"Threshold={best['threshold']:.2f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}  F1={best['f1']:.4f}"
    )
    print(f"TN={best['tn']} FP={best['fp']} FN={best['fn']} TP={best['tp']}")

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"fraud_threshold": best["threshold"]}, f, indent=2)

    print("\nSaved files:")
    print("-", THRESHOLD_PATH)
    print("-", REPORT_PATH)

    print("\n=== Quick check (common thresholds) ===")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        row = evaluate_at_threshold(y_test, y_prob, t)
        print(
            f"t={t:.1f}  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  FP={row['fp']}  FN={row['fn']}  TP={row['tp']}"
        )


if __name__ == "__main__":
    main()