import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/creditcard.csv"
OUT_DIR = "data/processed"
TEST_SIZE = 0.20
RANDOM_STATE = 42

def main():
    # Load
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # Basic checks
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # Separate features/label
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # Train/test split with stratification (VERY important for fraud)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nSplit done:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    print("\nClass distribution:")
    print("Train fraud %:", (y_train.mean() * 100))
    print("Test  fraud %:", (y_test.mean() * 100))

    # Scale ONLY Amount (V1-V28 are already scaled-ish because PCA)
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

    # Save scaler params (so we can reproduce later)
    with open(f"{OUT_DIR}/amount_scaler.txt", "w") as f:
        f.write(f"mean={scaler.mean_[0]}\n")
        f.write(f"scale={scaler.scale_[0]}\n")

    print("\nSaved processed files to:", OUT_DIR)

if __name__ == "__main__":
    main()