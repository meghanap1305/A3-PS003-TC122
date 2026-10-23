
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

DATA_DIR = Path("data/raw")
SPLITS = ["train", "val", "test"]
EXPECTED_INTERVAL = 5.0


def check_split(split_name):
    path = DATA_DIR / f"{split_name}.parquet"

    print("\n" + "=" * 50)
    print(split_name.upper())
    print("=" * 50)

    parquet = pq.ParquetFile(path)

    first_timestamp = None
    last_timestamp = None
    previous_timestamp = None

    intervals = set()
    unusual_gaps = []
    recent_timestamps = []

    # Read the timestamp column in batches to avoid loading
    # the entire dataset into memory.
    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["t0_timestamp"]
    ):
        values = batch.column(0).to_pylist()

        for value in values:
            if value is None:
                continue

            if first_timestamp is None:
                first_timestamp = value

            # Each timestamp appears once for every node.
            # Compare only when the timestamp changes.
            if value != previous_timestamp:

                if previous_timestamp is not None:
                    previous = pd.Timestamp(previous_timestamp)
                    current = pd.Timestamp(value)

                    minutes = (
                        current - previous
                    ).total_seconds() / 60

                    intervals.add(minutes)

                    if minutes != EXPECTED_INTERVAL:
                        unusual_gaps.append(
                            (previous_timestamp, value, minutes)
                        )

                recent_timestamps.append(value)

                if len(recent_timestamps) > 5:
                    recent_timestamps.pop(0)

                previous_timestamp = value

            last_timestamp = value

    print("First timestamp:", first_timestamp)
    print("Last timestamp:", last_timestamp)

    print(
        "Observed timestamp intervals in minutes:",
        sorted(intervals)
    )

    print("Last 5 distinct timestamps:", recent_timestamps)

    print("\nUnusual gaps:")

    if unusual_gaps:
        for start, end, minutes in unusual_gaps:
            print(
                f"  {start} -> {end} "
                f"({minutes} minutes)"
            )
    else:
        print("  No unusual gaps found.")

    print("Total unusual gaps:", len(unusual_gaps))


def main():
    for split in SPLITS:
        check_split(split)


if __name__ == "__main__":
    main()
