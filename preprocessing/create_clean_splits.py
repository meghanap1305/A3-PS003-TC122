
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

HORIZON_STEPS = 12
INTERVAL_MINUTES = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_boundary(split_name):
    """Get the first and last timestamps of a split."""
    path = DATA_DIR / f"{split_name}.parquet"
    parquet = pq.ParquetFile(path)

    first = None
    last = None

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["t0_timestamp"]
    ):
        values = batch.column(0).to_pylist()

        if values:
            if first is None:
                first = values[0]
            last = values[-1]

    return first, last


def clean_split(split_name, next_split_start=None):
    input_path = DATA_DIR / f"{split_name}.parquet"
    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    print(f"\nProcessing {split_name.upper()}...")

    # Test data has no following split, so preserve it unchanged.
    if next_split_start is None:
        pq.write_table(
            pq.read_table(input_path),
            output_path,
            compression="snappy"
        )
        print("Test split copied unchanged.")
        print("Saved to:", output_path)
        return

    cutoff = (
        pd.Timestamp(next_split_start)
        - pd.Timedelta(minutes=HORIZON_STEPS * INTERVAL_MINUTES)
    )

    # Your timestamps are ISO-format strings, so compare strings.
    cutoff_string = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    print("Next split begins:", next_split_start)
    print("Cutoff timestamp:", cutoff_string)

    parquet = pq.ParquetFile(input_path)
    writer = None
    kept_rows = 0
    removed_rows = 0

    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            timestamps = batch.column(
                batch.schema.get_field_index("t0_timestamp")
            )

            # Keep anchors strictly earlier than the cutoff.
            mask = pc.less(
                timestamps,
                pa.scalar(cutoff_string, type=timestamps.type)
            )

            filtered_batch = batch.filter(mask)

            kept_rows += filtered_batch.num_rows
            removed_rows += batch.num_rows - filtered_batch.num_rows

            if filtered_batch.num_rows > 0:
                table = pa.Table.from_batches([filtered_batch])

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="snappy"
                    )

                writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    print("Rows kept:", kept_rows)
    print("Rows removed:", removed_rows)
    print("Saved to:", output_path)


def main():
    val_start, _ = get_boundary("val")
    test_start, _ = get_boundary("test")

    clean_split("train", val_start)
    clean_split("val", test_start)
    clean_split("test")


if __name__ == "__main__":
    main()
