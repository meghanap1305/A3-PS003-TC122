
import numpy as np
from pathlib import Path

from data_loader import (
    DATA_DIR,
    fit_input_scaler,
    iter_batches,
)

def evaluate_baseline(split, mean, std):
    total_abs_error = np.zeros(2, dtype=np.float64)
    total_sq_error = np.zeros(2, dtype=np.float64)
    total_count = 0

    for X, y, metadata in iter_batches(split, mean, std):
        # Last observed value for each feature.
        # Shape: (batch_size, 1, 2)
        last_value = X[:, -1:, :]

        # Predict the same value for all 12 future steps.
        # Shape: (batch_size, 12, 2)
        prediction = np.repeat(last_value, 12, axis=1)

        # Convert predictions and targets back to original units.
        prediction = prediction * std.reshape(1, 1, 2)
        prediction = prediction + mean.reshape(1, 1, 2)

        actual = y * std.reshape(1, 1, 2)
        actual = actual + mean.reshape(1, 1, 2)

        errors = prediction - actual

        total_abs_error += np.abs(errors).sum(axis=(0, 1))
        total_sq_error += (errors ** 2).sum(axis=(0, 1))
        total_count += errors.shape[0] * errors.shape[1]

    mae = total_abs_error / total_count
    rmse = np.sqrt(total_sq_error / total_count)

    return mae, rmse


if __name__ == "__main__":
    train_path = DATA_DIR / "train.parquet"
    mean, std = fit_input_scaler(train_path)

    for split in ["val", "test"]:
        print(f"\n{split.upper()} RESULTS")

        mae, rmse = evaluate_baseline(split, mean, std)

        print(f"d0 MAE:  {mae[0]:.4f}")
        print(f"d0 RMSE: {rmse[0]:.4f}")
        print(f"d1 MAE:  {mae[1]:.4f}")
        print(f"d1 RMSE: {rmse[1]:.4f}")
