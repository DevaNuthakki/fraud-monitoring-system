import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["Class"].value_counts())

fraud_percentage = df["Class"].mean() * 100
print(f"\nFraud percentage: {fraud_percentage:.4f}%")