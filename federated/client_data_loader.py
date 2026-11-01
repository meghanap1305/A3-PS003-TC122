"""Streaming loader for one simulated client's Parquet partition."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from preprocessing.data_loader import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    METADATA_COLUMNS,
    INPUT_STEPS,
    PREDICTION_STEPS,
    NUM_FEATURES,
    reshape_windows,
)


def iter_client_batches(
    client_path,
    mean,
    std,
    batch_size=4096,
    shuffle=False,
    seed=42,
):
    """
    Yield normalized client batches.

    Returns:
        X: (batch_size, 12, 2)
        y: (batch_size, 12, 2)
        metadata: (batch_size, 2)
    """

    path = Path(client_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Client partition not found: {path}"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    mean = np.asarray(
        mean, dtype=np.float32
    ).reshape(1, 1, NUM_FEATURES)

    std = np.asarray(
        std, dtype=np.float32
    ).reshape(1, 1, NUM_FEATURES)

    if not np.isfinite(mean).all():
        raise ValueError("mean must contain finite values")

    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(
            "std must contain finite positive values"
        )

    parquet = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)

    columns = (
        METADATA_COLUMNS
        + INPUT_COLUMNS
        + TARGET_COLUMNS
    )

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        df = batch.to_pandas()

        X = reshape_windows(
            df, INPUT_COLUMNS, INPUT_STEPS
        )

        y = reshape_windows(
            df, TARGET_COLUMNS, PREDICTION_STEPS
        )

        # Use the shared scaler fitted on training inputs.
        X = ((X - mean) / std).astype(
            np.float32, copy=False
        )

        y = ((y - mean) / std).astype(
            np.float32, copy=False
        )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite normalized inputs found in {path}"
            )

        if not np.isfinite(y).all():
            raise ValueError(
                f"Non-finite normalized targets found in {path}"
            )

        metadata = df[METADATA_COLUMNS].to_numpy()

        if shuffle and len(X) > 1:
            order = rng.permutation(len(X))
            X = X[order]
            y = y[order]
            metadata = metadata[order]

        yield X, y, metadata