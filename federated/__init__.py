"""Tests for IID and non-IID client partitioning."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.data_loader import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
)
from partition_data import create_partitions


def make_train_file(path):
    """Create a small synthetic training Parquet file."""

    num_examples = 30

    data = {
        "node_id": np.repeat(np.arange(6), 5),
        "t0_timestamp": [
            f"timestamp_{i}" for i in range(num_examples)
        ],
    }

    rng = np.random.default_rng(5)

    for column in INPUT_COLUMNS + TARGET_COLUMNS:
        data[column] = rng.normal(
            size=num_examples
        ).astype(np.float32)

    df = pd.DataFrame(data)

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    pq.write_table(table, path)


def test_iid_partitions_cover_all_examples(tmp_path):
    """IID partitions should contain all input examples."""

    source_path = tmp_path / "train.parquet"
    output_dir = tmp_path / "iid"

    make_train_file(source_path)

    manifest = create_partitions(
        input_path=source_path,
        output_dir=output_dir,
        num_clients=3,
        mode="iid",
        seed=42,
    )

    assert manifest["mode"] == "iid"
    assert manifest["total_examples"] == 30

    total_client_examples = sum(
        client["num_examples"]
        for client in manifest["clients"]
    )

    assert total_client_examples == 30

    for client in manifest["clients"]:
        client_file = output_dir / (
            f"{client['client_id']}_train.parquet"
        )

        assert client_file.is_file()

    manifest_file = output_dir / "partition_manifest.json"

    assert manifest_file.is_file()

    saved_manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    assert saved_manifest["mode"] == "iid"


def test_non_iid_sensor_groups_are_disjoint(tmp_path):
    """A sensor should belong to only one non-IID client."""

    source_path = tmp_path / "train.parquet"
    output_dir = tmp_path / "non_iid"

    make_train_file(source_path)

    manifest = create_partitions(
        input_path=source_path,
        output_dir=output_dir,
        num_clients=3,
        mode="non_iid",
        seed=42,
    )

    assert manifest["mode"] == "non_iid"

    sensor_groups = [
        set(client["sensor_ids"])
        for client in manifest["clients"]
    ]

    for i in range(len(sensor_groups)):
        for j in range(i + 1, len(sensor_groups)):
            assert sensor_groups[i].isdisjoint(
                sensor_groups[j]
            )