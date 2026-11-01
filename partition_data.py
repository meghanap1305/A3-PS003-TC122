"""Create reproducible IID or sensor-based non-IID client Parquet partitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("data/processed/train.parquet")
CLIENT_IDS = ("client_A", "client_B", "client_C")


def create_partitions(
    input_path,
    output_dir,
    num_clients=3,
    mode="iid",
    seed=42,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training Parquet file not found: {input_path}"
        )

    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")

    if mode not in {"iid", "non_iid"}:
        raise ValueError("mode must be 'iid' or 'non_iid'")

    parquet = pq.ParquetFile(input_path)
    schema = parquet.schema_arrow

    if "node_id" not in schema.names:
        raise ValueError(
            "Input training Parquet must contain a 'node_id' column"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    client_ids = [
        CLIENT_IDS[i] if i < len(CLIENT_IDS) else f"client_{i + 1}"
        for i in range(num_clients)
    ]

    paths = [
        output_dir / f"{client_id}_train.parquet"
        for client_id in client_ids
    ]

    writers = [
        pq.ParquetWriter(str(path), schema, compression="snappy")
        for path in paths
    ]

    counts = [0] * num_clients
    sensors = [set() for _ in range(num_clients)]
    rng = np.random.default_rng(seed)
    sensor_to_client = None

    try:
        if mode == "non_iid":
            # Find the sensor IDs without loading all training rows.
            unique_sensors = set()

            for batch in parquet.iter_batches(
                batch_size=65536,
                columns=["node_id"],
            ):
                unique_sensors.update(batch.column(0).to_pylist())

            ordered_sensors = sorted(
                unique_sensors,
                key=lambda value: (str(type(value)), str(value)),
            )

            if len(ordered_sensors) < num_clients:
                raise ValueError(
                    f"Only {len(ordered_sensors)} sensors found for "
                    f"{num_clients} clients"
                )

            # Assign each entire sensor group to exactly one client.
            sensor_to_client = {
                sensor: (i * num_clients) // len(ordered_sensors)
                for i, sensor in enumerate(ordered_sensors)
            }

        # Process the source file in manageable batches.
        for batch in parquet.iter_batches(batch_size=65536):
            table = pa.Table.from_batches([batch])

            node_column_index = batch.schema.get_field_index("node_id")
            node_values = np.asarray(
                batch.column(node_column_index).to_numpy(
                    zero_copy_only=False
                )
            )

            if mode == "iid":
                # Randomly distribute examples using the fixed seed.
                assignments = rng.integers(
                    0,
                    num_clients,
                    size=len(node_values),
                )
            else:
                # Keep all examples from a sensor together.
                assignments = np.fromiter(
                    (sensor_to_client[value] for value in node_values),
                    dtype=np.int64,
                    count=len(node_values),
                )

            for client_idx in range(num_clients):
                row_indices = np.flatnonzero(
                    assignments == client_idx
                )

                if row_indices.size == 0:
                    continue

                client_table = table.take(
                    pa.array(row_indices, type=pa.int64())
                )

                writers[client_idx].write_table(client_table)

                counts[client_idx] += len(row_indices)
                sensors[client_idx].update(
                    node_values[row_indices].tolist()
                )

    finally:
        for writer in writers:
            writer.close()

    if any(count == 0 for count in counts):
        raise RuntimeError(
            "One or more client partitions are empty: "
            f"{dict(zip(client_ids, counts))}"
        )

    manifest = {
        "mode": mode,
        "seed": int(seed),
        "num_clients": num_clients,
        "source_train_file": str(input_path),
        "total_examples": int(sum(counts)),
        "clients": [
            {
                "client_id": client_id,
                "path": path.as_posix(),
                "num_examples": int(count),
                "sensor_ids": sorted(
                    list(sensor_ids),
                    key=lambda value: (str(type(value)), str(value)),
                ),
            }
            for client_id, path, count, sensor_ids in zip(
                client_ids, paths, counts, sensors
            )
        ],
        "partition_note": (
            "IID assigns each example randomly with a fixed seed; "
            "counts are approximately balanced."
            if mode == "iid"
            else
            "Non-IID assigns whole node_id sensor groups to clients; "
            "sensor assignments are disjoint."
        ),
    }

    manifest_path = output_dir / "partition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Created {mode} partitions in {output_dir}")

    for client in manifest["clients"]:
        print(
            f"  {client['client_id']}: "
            f"{client['num_examples']:,} examples, "
            f"{len(client['sensor_ids'])} sensors"
        )

    print(f"Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the processed training Parquet file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to data/federated/<mode>",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--mode",
        choices=["iid", "non_iid"],
        default="iid",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output)
        if args.output
        else Path("data/federated") / args.mode
    )

    create_partitions(
        input_path=args.input,
        output_dir=output_dir,
        num_clients=args.clients,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()