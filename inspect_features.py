
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path("data/raw")

columns = [
    "x_t-11_d0", "x_t-11_d1",
    "x_t+0_d0", "x_t+0_d1",
    "y_t+1_d0", "y_t+1_d1",
    "y_t+12_d0", "y_t+12_d1",
]

for split in ["train", "val", "test"]:
    path = DATA_DIR / f"{split}.parquet"
    parquet_file = pq.ParquetFile(path)

    print(f"\n{'=' * 55}")
    print(f"{split.upper()} FEATURE STATISTICS")
    print(f"{'=' * 55}")

    count = 0
    total = np.zeros(len(columns), dtype=np.float64)
    total_sq = np.zeros(len(columns), dtype=np.float64)
    minimum = np.full(len(columns), np.inf)
    maximum = np.full(len(columns), -np.inf)

    for batch in parquet_file.iter_batches(
        batch_size=100_000,
        columns=columns
    ):
        values = batch.to_pandas().to_numpy(
            dtype=np.float64
        )

        count += len(values)
        total += values.sum(axis=0)
        total_sq += (values * values).sum(axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0)
    std = np.sqrt(variance)

    print("Rows scanned:", count)
    print(
        f"{'Column':<16} {'Min':>12} {'Max':>12} "
        f"{'Mean':>12} {'Std':>12}"
    )

    for i, col in enumerate(columns):
        print(
            f"{col:<16} {minimum[i]:>12.4f} "
            f"{maximum[i]:>12.4f} {mean[i]:>12.4f} "
            f"{std[i]:>12.4f}"
        )
