
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

for name in ["train", "val", "test"]:
    path = DATA_DIR / f"{name}.parquet"

    print("\n" + "=" * 55)
    print(f"{name.upper()} DATASET")
    print("=" * 55)

    # Read only metadata columns first
    df = pd.read_parquet(
        path,
        columns=["node_id", "t0_timestamp"]
    )

    print("Number of rows:", len(df))
    print("Number of unique nodes:", df["node_id"].nunique())
    print("First timestamp:", df["t0_timestamp"].min())
    print("Last timestamp:", df["t0_timestamp"].max())
    print("Missing node IDs:", df["node_id"].isna().sum())
    print("Missing timestamps:", df["t0_timestamp"].isna().sum())

    print("\nFirst 5 records:")
    print(df.head())

    del df

# Inspect the schema without loading all feature data
train = pd.read_parquet(
    DATA_DIR / "train.parquet",
    engine="pyarrow"
)

print("\n" + "=" * 55)
print("FEATURE INSPECTION")
print("=" * 55)

print("Training shape:", train.shape)
print("\nAll columns:")
print(train.columns.tolist())

print("\nMissing values in each column:")
print(train.isna().sum().to_string())

print("\nDuplicate rows:", train.duplicated().sum())
