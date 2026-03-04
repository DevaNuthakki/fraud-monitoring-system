import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROCESSED_DIR = "../data/processed"


def train_random_forest(X_train, y_train):
    # Same idea as your best model, but keep it stable
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
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
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_threshold(y_true, y_prob, target_recall=0.90):
    thresholds = np.linspace(0.01, 0.99, 99)

    best = None
    best_row = None

    for t in thresholds:
        row = evaluate_at_threshold(y_true, y_prob, t)

        # Must meet target recall
        if row["recall"] >= target_recall:
            # Choose the one with highest precision
            if best is None or row["precision"] > best:
                best = row["precision"]
                best_row = row

    return best_row


def main():
    # Load processed data
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")

    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel().astype(int)
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel().astype(int)

    print("✅ Loaded data")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Train model
    model = train_random_forest(X_train, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    print("\nPR-AUC (probability quality):", round(pr_auc, 6))

    # Baseline at threshold 0.5
    base = evaluate_at_threshold(y_test, y_prob, 0.5)
    print("\n=== Baseline (threshold = 0.50) ===")
    print(
        f"Precision={base['precision']:.4f}  Recall={base['recall']:.4f}  F1={base['f1']:.4f}"
    )
    print(f"TN={base['tn']} FP={base['fp']} FN={base['fn']} TP={base['tp']}")

    # Find best threshold for target recall
    target_recall = 0.80
    best = find_best_threshold(y_test, y_prob, target_recall=target_recall)

    if best is None:
        print(f"\n❌ No threshold reached Recall >= {target_recall}.")
        print("Try 0.85 or 0.80 instead.")
        return

    print(f"\n=== Best threshold for Recall >= {target_recall} ===")
    print(
        f"Threshold={best['threshold']:.2f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}  F1={best['f1']:.4f}"
    )
    print(f"TN={best['tn']} FP={best['fp']} FN={best['fn']} TP={best['tp']}")

    # Also show a small table for common thresholds
    print("\n=== Quick check (common thresholds) ===")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        row = evaluate_at_threshold(y_test, y_prob, t)
        print(
            f"t={t:.1f}  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  FP={row['fp']}  FN={row['fn']}  TP={row['tp']}"
        )


if __name__ == "__main__":
    main()