
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/processed")

BATCH_SIZE = 4096
INPUT_STEPS = 12
PREDICTION_STEPS = 12
NUM_FEATURES = 2

SPLITS = ["train", "val", "test"]

# Input: previous 11 steps and the current step.
INPUT_COLUMNS = [
    f"x_t{i:+d}_d{d}"
    for i in range(-11, 1)
    for d in range(NUM_FEATURES)
]

# Target: the next 12 steps.
TARGET_COLUMNS = [
    f"y_t+{i}_d{d}"
    for i in range(1, PREDICTION_STEPS + 1)
    for d in range(NUM_FEATURES)
]

METADATA_COLUMNS = ["node_id", "t0_timestamp"]


# --------------------------------------------------
# VALIDATE FILES AND COLUMNS
# --------------------------------------------------

def validate_files():
    """Check that all processed split files exist and have required columns."""

    required_columns = (
        METADATA_COLUMNS + INPUT_COLUMNS + TARGET_COLUMNS
    )

    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run create_clean_splits.py first."
            )

        parquet = pq.ParquetFile(path)
        available_columns = set(parquet.schema_arrow.names)

        missing = [
            column for column in required_columns
            if column not in available_columns
        ]

        if missing:
            raise ValueError(
                f"{split}.parquet is missing columns: {missing}"
            )

        print(
            f"{split.upper()}: "
            f"{parquet.metadata.num_rows:,} rows; schema OK"
        )


# --------------------------------------------------
# RESHAPE INPUTS AND TARGETS
# --------------------------------------------------

def reshape_windows(df, columns, steps):
    """
    Convert flattened columns into:
        (batch_size, steps, num_features)
    """

    values = df[columns].to_numpy(dtype=np.float32)

    expected_columns = steps * NUM_FEATURES

    if values.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, "
            f"got {values.shape[1]}"
        )

    return values.reshape(
        len(values), steps, NUM_FEATURES
    )


# --------------------------------------------------
# FIT SCALER USING TRAINING INPUTS ONLY
# --------------------------------------------------

def fit_input_scaler(train_path=None):
    """
    Calculate per-feature mean and standard deviation
    using training input columns only.

    Batch processing avoids loading the entire dataset
    into memory at once.
    """

    if train_path is None:
        train_path = DATA_DIR / "train.parquet"

    parquet = pq.ParquetFile(train_path)

    feature_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    feature_sum_sq = np.zeros(NUM_FEATURES, dtype=np.float64)
    total_values = 0

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=INPUT_COLUMNS
    ):
        df = batch.to_pandas()
        X = reshape_windows(df, INPUT_COLUMNS, INPUT_STEPS)

        # Count all observed input values for each feature.
        flattened = X.reshape(-1, NUM_FEATURES).astype(np.float64)

        if not np.isfinite(flattened).all():
            raise ValueError(
                "Non-finite value found in training inputs."
            )

        feature_sum += flattened.sum(axis=0)
        feature_sum_sq += np.square(flattened).sum(axis=0)
        total_values += flattened.shape[0]

    if total_values == 0:
        raise ValueError("No training input values were found.")

    mean = feature_sum / total_values

    variance = (
        feature_sum_sq / total_values
        - np.square(mean)
    )

    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    # Prevent division by zero for constant features.
    std[std < 1e-8] = 1.0

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    print("\nTraining-input scaler:")
    print("Mean:", mean)
    print("Std: ", std)

    return mean, std


# --------------------------------------------------
# ITERATE THROUGH BATCHES
# --------------------------------------------------

def iter_batches(
    split_name,
    mean,
    std,
    batch_size=BATCH_SIZE
):
    """
    Yield batches as:
        X:        (batch, 12, 2)
        y:        (batch, 12, 2)
        metadata: (batch, 2)

    Both X and y are normalized using statistics fitted
    on training inputs only.
    """

    if split_name not in SPLITS:
        raise ValueError(
            f"Invalid split '{split_name}'. "
            f"Choose one of {SPLITS}."
        )

    path = DATA_DIR / f"{split_name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    columns = (
        METADATA_COLUMNS + INPUT_COLUMNS + TARGET_COLUMNS
    )

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns
    ):
        df = batch.to_pandas()

        X = reshape_windows(
            df, INPUT_COLUMNS, INPUT_STEPS
        )

        y = reshape_windows(
            df, TARGET_COLUMNS, PREDICTION_STEPS
        )

        # Normalize inputs and targets using the same
        # per-feature training-input statistics.
        X = (X - mean) / std
        y = (y - mean) / std

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite normalized input in {split_name}."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                f"Non-finite normalized target in {split_name}."
            )

        metadata = df[METADATA_COLUMNS].to_numpy()

        yield (
            X.astype(np.float32, copy=False),
            y.astype(np.float32, copy=False),
            metadata
        )


# --------------------------------------------------
# MAIN: QUICK LOADER TEST
# --------------------------------------------------

if __name__ == "__main__":
    validate_files()

    mean, std = fit_input_scaler()

    for split in SPLITS:
        print(f"\nTesting {split.upper()} loader...")

        batches = iter_batches(split, mean, std)
        X, y, metadata = next(batches)

        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Metadata shape:", metadata.shape)
        print("X finite:", np.isfinite(X).all())
        print("y finite:", np.isfinite(y).all())
        print("Example metadata:")
        print(metadata[:3])

    print("\nLoader test completed successfully.")
