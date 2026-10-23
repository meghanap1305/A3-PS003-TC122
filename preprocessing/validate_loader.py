
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from data_loader import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    fit_input_scaler,
    iter_batches,
)

DATA_DIR = Path("data/raw")

print("Input feature columns:", len(INPUT_COLUMNS))
print("Target feature columns:", len(TARGET_COLUMNS))

print("\nInput columns:")
print(INPUT_COLUMNS)

print("\nTarget columns:")
print(TARGET_COLUMNS)

# Confirm that all required columns exist in every split.
required = set(
    ["node_id", "t0_timestamp"]
    + INPUT_COLUMNS
    + TARGET_COLUMNS
)

for split in ["train", "val", "test"]:
    path = DATA_DIR / f"{split}.parquet"
    parquet = pq.ParquetFile(path)

    available = set(parquet.schema_arrow.names)
    missing = required - available

    if missing:
        raise ValueError(f"{split}: missing columns: {missing}")

    print(f"\n{split}: all required columns exist")

# Reuse training-only normalization statistics.
mean, std = fit_input_scaler(DATA_DIR / "train.parquet")

print("\nTraining means:", mean)
print("Training standard deviations:", std)

# Inspect the first batch of each split.
for split in ["train", "val", "test"]:
    X, y, metadata = next(iter_batches(split, mean, std))

    assert X.shape[1:] == (12, 2)
    assert y.shape[1:] == (12, 2)
    assert len(X) == len(y) == len(metadata)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    print(
        f"\n{split.upper()} passed: "
        f"X={X.shape}, y={y.shape}"
    )

print("\nAll initial validation checks passed.")
