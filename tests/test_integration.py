"""End-to-end test: runner + the REAL client code + REAL validation, on tiny synthetic data.

No big data needed: client partitions and a validation split are generated in tmp_path,
and preprocessing.data_loader.DATA_DIR is redirected there.
"""
import csv
import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import preprocessing.data_loader as dl
from federated.federated_train import FedConfig, load_client_specs, run_federated
from src.model import TrafficMLP

SIZES = {"client_A": 40, "client_B": 24, "client_C": 16}


def make_parquet(path, num_examples, seed):
    rng = np.random.default_rng(seed)
    data = {
        "node_id": rng.integers(0, 6, size=num_examples),
        "t0_timestamp": [f"2017-01-01T00:{i % 60:02d}:00" for i in range(num_examples)],
    }
    for col in dl.INPUT_COLUMNS + dl.TARGET_COLUMNS:
        data[col] = rng.normal(size=num_examples).astype(np.float32)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False), path)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    fed_dir = tmp_path / "data" / "federated" / "iid"
    fed_dir.mkdir(parents=True)
    for i, (cid, n) in enumerate(SIZES.items()):
        make_parquet(fed_dir / f"{cid}_train.parquet", n, seed=i)

    manifest = {
        "mode": "iid",
        "seed": 42,
        "total_examples": sum(SIZES.values()),
        "clients": [
            {"client_id": cid, "num_examples": n, "sensor_ids": []} for cid, n in SIZES.items()
        ],
    }
    (fed_dir / "partition_manifest.json").write_text(json.dumps(manifest))

    processed = tmp_path / "processed"
    processed.mkdir()
    make_parquet(processed / "val.parquet", 30, seed=99)
    monkeypatch.setattr(dl, "DATA_DIR", processed)   # evaluation reads val from here
    return tmp_path, fed_dir


def test_manifest_loading_matches_partition_owner_schema(workspace):
    _, fed_dir = workspace
    specs = load_client_specs(fed_dir / "partition_manifest.json")
    assert [s.client_id for s in specs] == list(SIZES)
    assert [s.expected_examples for s in specs] == list(SIZES.values())
    assert all(s.path.endswith(f"{s.client_id}_train.parquet") for s in specs)


def test_missing_manifest_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_client_specs(tmp_path / "nope.json")


@pytest.mark.parametrize("local_epochs", [1, 2])
def test_two_rounds_with_real_client_and_real_validation(workspace, local_epochs):
    tmp_path, fed_dir = workspace
    cfg = FedConfig(
        mode="iid",
        manifest_path=str(fed_dir / "partition_manifest.json"),
        output_dir=str(tmp_path / "results"),
        rounds=2,
        local_epochs=local_epochs,
        batch_size=8,
        seed=42,
    )
    result = run_federated(cfg, scaler=([0.0, 0.0], [1.0, 1.0]))

    rows = list(csv.DictReader(open(tmp_path / "results" / "round_metrics.csv")))
    assert len(rows) == 2
    for r in rows:
        assert int(r["participating_clients"]) == 3
        # unique examples, regardless of local_epochs
        assert int(r["total_client_examples"]) == sum(SIZES.values())
        for field in ("val_mae_norm", "val_rmse_norm", "val_mae_orig_d0", "val_rmse_orig_d1",
                      "local_loss_sample_weighted_mean"):
            assert math.isfinite(float(r[field]))

    ckpt = torch.load(tmp_path / "results" / "best_model.pt", map_location="cpu")
    TrafficMLP().load_state_dict(ckpt["model_state_dict"])
    assert result["best_round"] in (1, 2)


def test_runs_are_reproducible_with_same_seed(workspace):
    tmp_path, fed_dir = workspace

    def go(name):
        cfg = FedConfig(mode="iid", manifest_path=str(fed_dir / "partition_manifest.json"),
                        output_dir=str(tmp_path / name), rounds=2, batch_size=8, seed=42)
        return run_federated(cfg, scaler=([0.0, 0.0], [1.0, 1.0]))["model"].state_dict()

    a, b = go("run_a"), go("run_b")
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6)
