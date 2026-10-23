
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc

DATA_DIR = Path("data/processed")

for split in ["train", "val", "test"]:
    path = DATA_DIR / f"{split}.parquet"
    parquet = pq.ParquetFile(path)

    missing_counts = {
        column: 0 for column in parquet.schema_arrow.names
    }

    for batch in parquet.iter_batches(batch_size=100_000):
        for column in missing_counts:
            values = batch.column(
                batch.schema.get_field_index(column)
            )
            missing_counts[column] += (
                len(values) - pc.count(values).as_py()
            )

    missing_columns = {
        column: count
        for column, count in missing_counts.items()
        if count > 0
    }

    print(f"\n{split.upper()}")
    print("Rows:", f"{parquet.metadata.num_rows:,}")
    print("Columns with missing values:", missing_columns)
    print("Missing-value check:", 
          "PASS" if not missing_columns else "NEEDS ATTENTION")
