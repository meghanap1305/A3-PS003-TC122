"""Tests for client data loading and local model training."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from preprocessing.data_loader import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
)
from src.model import TrafficMLP
from federated.client_data_loader import iter_client_batches
from federated.client import train_client


def make_tiny_partition(path, num_examples=5):
    """Create a tiny synthetic client dataset."""

    data = {
        "node_id": [1] * num_examples,
        "t0_timestamp": [
            f"timestamp_{i}" for i in range(num_examples)
        ],
    }

    rng = np.random.default_rng(3)

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


def test_client_loader_shapes(tmp_path):
    """The loader should return the expected tensor dimensions."""

    client_path = tmp_path / "client.parquet"

    make_tiny_partition(client_path)

    batches = iter_client_batches(
        client_path=client_path,
        mean=[0.0, 0.0],
        std=[1.0, 1.0],
        batch_size=3,
    )

    X, y, metadata = next(batches)

    assert X.shape == (3, 12, 2)
    assert y.shape == (3, 12, 2)
    assert metadata.shape == (3, 2)

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_train_client_returns_updated_weights(tmp_path):
    """Local training should update weights and return metrics."""

    client_path = tmp_path / "client.parquet"

    make_tiny_partition(
        client_path,
        num_examples=5,
    )

    model = TrafficMLP()

    initial_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    updated_state, num_examples, metrics = train_client(
        global_state_dict=initial_state,
        client_path=client_path,
        mean=[0.0, 0.0],
        std=[1.0, 1.0],
        local_epochs=1,
        learning_rate=0.001,
        batch_size=2,
        device="cpu",
    )

    assert num_examples == 5

    assert "train_loss" in metrics
    assert np.isfinite(metrics["train_loss"])

    assert updated_state.keys() == initial_state.keys()

    weights_changed = any(
        not torch.equal(initial_state[key], updated_state[key])
        for key in initial_state
        if initial_state[key].is_floating_point()
    )

    assert weights_changed