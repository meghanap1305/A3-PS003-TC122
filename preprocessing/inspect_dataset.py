
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

for filename in ["train.parquet", "val.parquet", "test.parquet"]:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    if not file_path.exists():
        print(f"File not found: {file_path.resolve()}")
        continue

    df = pd.read_parquet(file_path)

    print("Shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist()[:30])

    print("\nData types:")
    print(df.dtypes.head(15))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:", int(df.isna().sum().sum()))

    print("\nIndex type:", type(df.index).__name__)
    print("Index sample:", df.index[:5].tolist())
