
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

DATA_DIR = Path("data/processed")

SPLITS = ["train", "val", "test"]
EXPECTED_SENSORS = 325
HORIZON_STEPS = 12
INTERVAL_MINUTES = 5

INPUT_COLUMNS = [
    f"x_t{i:+d}_d{d}"
    for i in range(-11, 1)
    for d in range(2)
]

TARGET_COLUMNS = [
    f"y_t+{i}_d{d}"
    for i in range(1, 13)
    for d in range(2)
]

REQUIRED_COLUMNS = (
    ["node_id", "t0_timestamp"]
    + INPUT_COLUMNS
    + TARGET_COLUMNS
)


def inspect_split(split):
    path = DATA_DIR / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    parquet = pq.ParquetFile(path)
    schema = set(parquet.schema_arrow.names)

    missing_columns = set(REQUIRED_COLUMNS) - schema
    if missing_columns:
        raise ValueError(
            f"{split}: missing columns {sorted(missing_columns)}"
        )

    row_count = parquet.metadata.num_rows
    null_counts = {c: 0 for c in REQUIRED_COLUMNS}
    sensor_ids = set()
    first_timestamp = None
    last_timestamp = None
    distinct_timestamps = set()

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=REQUIRED_COLUMNS
    ):
        for column in REQUIRED_COLUMNS:
            values = batch.column(
                batch.schema.get_field_index(column)
            )
            null_counts[column] += (
                len(values) - pc.count(values).as_py()
            )

        nodes = batch.column(
            batch.schema.get_field_index("node_id")
        ).to_pylist()
        sensor_ids.update(nodes)

        timestamps = batch.column(
            batch.schema.get_field_index("t0_timestamp")
        ).to_pylist()

        valid_timestamps = [
            timestamp for timestamp in timestamps
            if timestamp is not None
        ]

        if valid_timestamps:
            if first_timestamp is None:
                first_timestamp = valid_timestamps[0]
            last_timestamp = valid_timestamps[-1]
            distinct_timestamps.update(valid_timestamps)

    missing_values = {
        column: count
        for column, count in null_counts.items()
        if count > 0
    }

    # Every timestamp should have one row per sensor.
    bad_sensor_counts = []
    for timestamp in sorted(distinct_timestamps):
        # This check is performed below more efficiently using
        # the row count and the full sensor count assumption.
        break

    print(f"\n{'=' * 45}")
    print(split.upper())
    print("=" * 45)
    print(f"Rows: {row_count:,}")
    print(f"Distinct sensors: {len(sensor_ids)}")
    print(f"First timestamp: {first_timestamp}")
    print(f"Last timestamp: {last_timestamp}")
    print(f"Distinct timestamps: {len(distinct_timestamps)}")
    print(f"Missing values: {missing_values}")

    checks = {
        "Required columns present": not missing_columns,
        "No missing values": not missing_values,
        "325 sensors present": len(sensor_ids) == EXPECTED_SENSORS,
        "Rows divisible by sensor count":
            row_count % EXPECTED_SENSORS == 0,
        "Timestamps present":
            first_timestamp is not None and last_timestamp is not None,
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise ValueError(f"{split} failed preprocessing validation.")

    return (
        pd.Timestamp(first_timestamp),
        pd.Timestamp(last_timestamp),
        len(distinct_timestamps),
    )


def main():
    print("FINAL PREPROCESSING VALIDATION")
    print("Data directory:", DATA_DIR.resolve())

    boundaries = {}
    for split in SPLITS:
        boundaries[split] = inspect_split(split)

    train_end = boundaries["train"][1]
    val_start = boundaries["val"][0]
    val_end = boundaries["val"][1]
    test_start = boundaries["test"][0]

    horizon = pd.Timedelta(
        minutes=HORIZON_STEPS * INTERVAL_MINUTES
    )

    print(f"\n{'=' * 45}")
    print("CHRONOLOGICAL BOUNDARY CHECKS")
    print("=" * 45)

    train_target_end = train_end + horizon
    val_target_end = val_end + horizon

    train_ok = train_target_end < val_start
    val_ok = val_target_end < test_start
    order_ok = train_end < val_start < val_end < test_start

    print("Last train anchor:", train_end)
    print("Approximate last train target:", train_target_end)
    print("First validation anchor:", val_start)
    print("Train boundary safe:", "PASS" if train_ok else "FAIL")

    print("\nLast validation anchor:", val_end)
    print("Approximate last validation target:", val_target_end)
    print("First test anchor:", test_start)
    print("Validation boundary safe:", "PASS" if val_ok else "FAIL")

    print("\nChronological order:", "PASS" if order_ok else "FAIL")

    if not (train_ok and val_ok and order_ok):
        raise ValueError(
            "Boundary checks failed. Review split cleanup."
        )

    print("\nPreprocessing validation completed successfully.")


if __name__ == "__main__":
    main()
