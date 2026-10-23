
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

DATA_DIR = Path("data/raw")
HORIZON_STEPS = 12
INTERVAL_MINUTES = 5

def get_boundaries(split):
    path = DATA_DIR / f"{split}.parquet"
    parquet = pq.ParquetFile(path)

    first = None
    last = None

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["t0_timestamp"]
    ):
        values = batch.column(0).to_pylist()

        if not values:
            continue

        if first is None:
            first = pd.Timestamp(values[0])

        last = pd.Timestamp(values[-1])

    return first, last


train_start, train_end = get_boundaries("train")
val_start, val_end = get_boundaries("val")
test_start, test_end = get_boundaries("test")

print("Split boundaries:")
print("Train:", train_start, "to", train_end)
print("Validation:", val_start, "to", val_end)
print("Test:", test_start, "to", test_end)

horizon = pd.Timedelta(
    minutes=HORIZON_STEPS * INTERVAL_MINUTES
)

print("\nPotential target-window overlap:")

for name, end, next_start in [
    ("Train -> Validation", train_end, val_start),
    ("Validation -> Test", val_end, test_start),
]:
    final_target = end + horizon

    print(f"\n{name}")
    print("Last anchor:", end)
    print("Approximate final target:", final_target)
    print("Next split begins:", next_start)

    if final_target >= next_start:
        print("Target windows may cross the next split boundary.")
    else:
        print("No overlap detected by this boundary check.")
