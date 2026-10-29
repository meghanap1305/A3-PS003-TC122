import copy
import csv
import json

import pytest
import torch

from federated.evaluation import evaluate_global
from federated.federated_train import ClientSpec, FedConfig, run_federated
from src.model import TrafficMLP

MEAN, STD = [60.0, 0.3], [10.0, 0.2]
COUNTS = {"A": 100, "B": 200, "C": 300}


def _make_cfg(tmp_path, rounds=2):
    return FedConfig(mode="iid", manifest_path="unused", output_dir=str(tmp_path / "out"),
                     rounds=rounds, seed=42)


def _clients(counts=COUNTS):
    return [ClientSpec(client_id=k, path=f"fake_{k}.parquet") for k in counts]


class FakeClient:
    """Mock train_client. Client k adds +delta_k to every weight and MUTATES its input
    in place (to prove the runner isolates clients from each other)."""

    def __init__(self, counts=COUNTS):
        self.deltas = {f"fake_{k}.parquet": float(i + 1) for i, k in enumerate(counts)}
        self.counts = {f"fake_{k}.parquet": n for k, n in counts.items()}
        self.calls = []   # (round-ish call index, path, copy of incoming state)

    def __call__(self, global_state_dict, client_path, mean, std, local_epochs,
                 learning_rate, batch_size, device):
        self.calls.append((client_path, copy.deepcopy(global_state_dict)))
        out = {k: v + self.deltas[client_path] for k, v in global_state_dict.items()}
        for v in global_state_dict.values():     # nasty: mutate the input
            v.add_(1000.0)
        return out, self.counts[client_path], {"train_loss": self.deltas[client_path]}


class FakeVal:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, mean, std, device, batch_size, max_batches):
        self.calls += 1
        return {"mae_norm": 1.0 / self.calls, "rmse_norm": 2.0 / self.calls,
                "mae_orig_d0": 1.0, "rmse_orig_d0": 1.0, "mae_orig_d1": 1.0, "rmse_orig_d1": 1.0}


def _run(tmp_path, rounds=2, counts=COUNTS, fake=None):
    fake = fake or FakeClient(counts)
    val = FakeVal()
    res = run_federated(_make_cfg(tmp_path, rounds), clients=_clients(counts),
                        train_client_fn=fake, validation_fn=val, scaler=(MEAN, STD))
    return res, fake, val


def test_two_rounds_complete_and_validation_logged_once_per_round(tmp_path):
    res, _, val = _run(tmp_path)
    assert len(res["history"]) == 2
    assert val.calls == 2
    rows = list(csv.DictReader(open(tmp_path / "out" / "round_metrics.csv")))
    assert [r["round"] for r in rows] == ["1", "2"]
    assert rows[0]["participating_clients"] == "3"
    assert rows[0]["total_client_examples"] == "600"


def test_all_clients_get_identical_start_weights_each_round(tmp_path):
    _, fake, _ = _run(tmp_path)
    assert len(fake.calls) == 6
    for r in range(2):
        round_calls = fake.calls[3 * r: 3 * r + 3]
        ref = round_calls[0][1]
        for _, st in round_calls[1:]:
            for k in ref:
                assert torch.equal(st[k], ref[k])


def test_round2_start_equals_round1_aggregate_and_state_changes(tmp_path):
    torch.manual_seed(0)
    res, fake, _ = _run(tmp_path, rounds=1)
    start = fake.calls[0][1]
    final = res["model"].state_dict()
    expected_shift = (100 * 1 + 200 * 2 + 300 * 3) / 600       # 2.3333
    for k in start:
        assert not torch.equal(start[k], final[k])
        assert torch.allclose(final[k], start[k] + expected_shift, atol=1e-4)


def test_zero_example_client_is_skipped(tmp_path):
    counts = {"A": 100, "B": 0, "C": 300}
    res, _, _ = _run(tmp_path, counts=counts)
    assert res["history"][0]["participating_clients"] == 2
    assert res["history"][0]["total_client_examples"] == 400


def test_all_clients_empty_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError):
        _run(tmp_path, counts={"A": 0, "B": 0})


def test_checkpoints_config_and_seed_recorded(tmp_path):
    _run(tmp_path)
    out = tmp_path / "out"
    for name in ("best_model.pt", "final_model.pt", "config.json", "round_metrics.csv"):
        assert (out / name).exists()
    ckpt = torch.load(out / "best_model.pt", map_location="cpu")
    TrafficMLP().load_state_dict(ckpt["model_state_dict"])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["seed"] == 42 and cfg["rounds"] == 2 and cfg["scaler_mean"] == MEAN
    rows = list(csv.DictReader(open(out / "round_metrics.csv")))
    assert all(r["seed"] == "42" and r["mode"] == "iid" for r in rows)
    assert [r["is_best"] for r in rows] == ["1", "1"]          # FakeVal improves each round


def test_loss_labels_weighted_vs_simple(tmp_path):
    res, _, _ = _run(tmp_path, rounds=1)
    row = res["history"][0]
    assert row["local_loss_simple_mean"] == pytest.approx(2.0)
    assert row["local_loss_sample_weighted_mean"] == pytest.approx((100 * 1 + 200 * 2 + 300 * 3) / 600)


def test_test_split_is_blocked_during_training():
    with pytest.raises(PermissionError):
        evaluate_global(TrafficMLP(), MEAN, STD, split="test")
    with pytest.raises(ValueError):
        evaluate_global(TrafficMLP(), MEAN, STD, split="train")
