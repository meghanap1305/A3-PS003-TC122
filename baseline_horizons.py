
import numpy as np
from data_loader import DATA_DIR, fit_input_scaler, iter_batches


def evaluate_horizons(split, mean, std):
    abs_errors = np.zeros((12, 2), dtype=np.float64)
    sq_errors = np.zeros((12, 2), dtype=np.float64)
    count = 0

    for X, y, metadata in iter_batches(split, mean, std):
        # Persistence forecast: repeat the latest observation.
        pred = np.repeat(X[:, -1:, :], 12, axis=1)

        # Convert normalized values back to original units.
        pred = pred * std[None, None, :] + mean[None, None, :]
        actual = y * std[None, None, :] + mean[None, None, :]

        error = pred - actual

        abs_errors += np.abs(error).sum(axis=0)
        sq_errors += (error ** 2).sum(axis=0)
        count += len(X)

    mae = abs_errors / count
    rmse = np.sqrt(sq_errors / count)

    print(f"\n{split.upper()} — ERROR BY HORIZON")
    print("Horizon | d0 MAE | d0 RMSE | d1 MAE | d1 RMSE")

    for h in range(12):
        print(
            f"{h + 1:>7} | "
            f"{mae[h, 0]:>7.4f} | "
            f"{rmse[h, 0]:>8.4f} | "
            f"{mae[h, 1]:>7.4f} | "
            f"{rmse[h, 1]:>8.4f}"
        )


if __name__ == "__main__":
    mean, std = fit_input_scaler(DATA_DIR / "train.parquet")

    evaluate_horizons("val", mean, std)
