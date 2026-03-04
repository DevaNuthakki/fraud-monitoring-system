import os
import json
import time
import pandas as pd
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
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
THRESHOLD = 0.31  # from your tune_threshold.py result
DATA_DIR = os.path.join("..", "data", "processed")
MODEL_DIR = os.path.join("..", "model")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")


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
    print("✅ Loading processed data...")
    X_train, y_train, X_test, y_test = load_processed()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # Train the final model (same idea as your best model from compare_models)
    print("\n✅ Training final RandomForest model...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # helps because fraud is rare
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start
    print(f"Training time: {train_time:.2f} sec")

    # Evaluate on test set using the chosen threshold
    print("\n✅ Evaluating with threshold =", THRESHOLD)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob, THRESHOLD)

    print("\n=== Final Model Metrics (Test) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save model + threshold
    os.makedirs(MODEL_DIR, exist_ok=True)

    dump(model, MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"fraud_threshold": THRESHOLD}, f, indent=2)

    print("\n✅ Saved files:")
    print(" -", MODEL_PATH)
    print(" -", THRESHOLD_PATH)


if __name__ == "__main__":
    main()