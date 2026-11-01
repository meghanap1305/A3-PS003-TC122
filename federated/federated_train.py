"""Federated training runner (server side): FedAvg over simulated clients.

Run from the repository root:

    # 2-round smoke test (IID, then non-IID)
    python -m federated.federated_train --mode iid --smoke
    python -m federated.federated_train --mode non_iid --smoke

    # first full run
    python -m federated.federated_train --mode iid --rounds 5
    python -m federated.federated_train --mode non_iid --rounds 5

Per round (same starting weights for EVERY client, aggregate once at the end):
    broadcast -> each client trains locally -> FedAvg -> validate -> log -> checkpoint
"""
import argparse
import copy
import csv
import json
import logging
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .fedavg import fedavg

log = logging.getLogger("federated")

CSV_FIELDS = [
    "mode",
    "seed",
    "round",
    "participating_clients",
    "total_client_examples",
    "local_loss_sample_weighted_mean",   # sum(n_k * loss_k) / sum(n_k)
    "local_loss_simple_mean",            # plain mean of loss_k over clients
    "val_mae_norm",                      # same definition as src/train.py
    "val_rmse_norm",
    "val_mae_orig_d0", "val_rmse_orig_d0",   # same definition as baseline.py
    "val_mae_orig_d1", "val_rmse_orig_d1",
    "duration_sec",
    "est_comm_bytes",                    # ESTIMATE: 2 * model_bytes * clients
    "is_best",
]


# ----------------------------------------------------------------------------
# Config / helpers
# ----------------------------------------------------------------------------

@dataclass
class FedConfig:
    mode: str = "iid"                         # "iid" or "non_iid"
    manifest_path: str = ""
    output_dir: str = ""
    rounds: int = 2
    local_epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 4096
    seed: int = 42
    device: str = "cpu"
    max_val_batches: Optional[int] = None     # smoke-test only; None = full val split


@dataclass
class ClientSpec:
    client_id: str
    path: str
    expected_examples: Optional[int] = None   # from the manifest, if present


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_client_specs(manifest_path):
    """Read client ids / paths / counts from partition_manifest.json.

    Matches the partition owner's manifest: top-level "mode", "total_examples" and a
    "clients" list whose entries have "client_id", "num_examples" and "sensor_ids".
    If an entry has no explicit path ("path" / "file_path" / "file"), the file is
    assumed to be <manifest_dir>/<client_id>_train.parquet (the naming used by
    create_partitions).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Partition manifest not found: {manifest_path}. "
            "Run partition_data.create_partitions first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    clients = manifest.get("clients")
    if isinstance(clients, dict):
        clients = [{"client_id": k, **v} for k, v in clients.items()]
    if not clients:
        raise ValueError(f"No 'clients' entry found in manifest {manifest_path}.")

    specs = []
    for i, c in enumerate(clients):
        client_id = str(c.get("client_id", c.get("id", i)))
        raw = c.get("path") or c.get("file_path") or c.get("file")
        if raw is not None:
            candidates = [Path(raw), manifest_path.parent / raw,
                          manifest_path.parent / Path(raw).name]
        else:
            candidates = [manifest_path.parent / f"{client_id}_train.parquet"]
        resolved = next((p for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Client file for '{client_id}' not found. Tried: {[str(p) for p in candidates]}"
            )
        expected = c.get("num_examples", c.get("example_count", c.get("n_examples")))
        specs.append(
            ClientSpec(
                client_id=client_id,
                path=str(resolved),
                expected_examples=None if expected is None else int(expected),
            )
        )
    return specs


def _clone_state(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _state_bytes(state_dict):
    return sum(v.numel() * v.element_size() for v in state_dict.values())


def _save_checkpoint(path, model, round_num, val, mean, std, cfg):
    torch.save(
        {
            "round": round_num,
            "model_state_dict": _clone_state(model.state_dict()),
            "validation_mae": val["mae_norm"],
            "validation_rmse": val["rmse_norm"],
            # plain lists (not numpy arrays) so torch.load(weights_only=True) works
            "mean": [float(x) for x in np.asarray(mean).ravel()],
            "std": [float(x) for x in np.asarray(std).ravel()],
            "mode": cfg.mode,
            "seed": cfg.seed,
        },
        path,
    )


def _default_train_client():
    from .client import train_client   # owned by the client-side teammate
    return train_client


def _default_validation_fn(split_guard="val"):
    from .evaluation import evaluate_global

    def validate(model, mean, std, device, batch_size, max_batches):
        # The split is hard-coded: the runner can never touch test data.
        return evaluate_global(
            model, mean, std, device=device, batch_size=batch_size,
            split=split_guard, max_batches=max_batches, allow_test=False,
        )
    return validate


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def run_federated(
    cfg,
    clients=None,
    train_client_fn=None,
    validation_fn=None,
    scaler=None,
    model_factory=None,
):
    """Run federated training. The keyword hooks exist so tests can inject mocks.

    validation_fn(model, mean, std, device, batch_size, max_batches) -> dict with
        at least "mae_norm" and "rmse_norm" (optionally *_orig_d0/d1).
    train_client_fn: the agreed client interface (see contract section 6).
    """
    if cfg.rounds < 1:
        raise ValueError("rounds must be >= 1")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    if clients is None:
        clients = load_client_specs(cfg.manifest_path)
    if not clients:
        raise ValueError("No clients to train.")

    if scaler is None:
        from preprocessing.data_loader import fit_input_scaler
        scaler = fit_input_scaler()          # TRAIN inputs only; shared by everyone
    mean, std = scaler

    if model_factory is None:
        from src.model import TrafficMLP
        model_factory = TrafficMLP
    if train_client_fn is None:
        train_client_fn = _default_train_client()
    if validation_fn is None:
        validation_fn = _default_validation_fn()

    device = torch.device(cfg.device)
    global_model = model_factory().to(device)
    model_bytes = _state_bytes(global_model.state_dict())

    config_record = {
        **asdict(cfg),
        "clients": [asdict(c) for c in clients],
        "scaler_mean": [float(x) for x in np.asarray(mean).ravel()],
        "scaler_std": [float(x) for x in np.asarray(std).ravel()],
        "model_bytes": model_bytes,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "comm_bytes_note": "Estimate only: 2 * model_bytes * participating_clients per round "
                           "(1 download + 1 upload per client); no real network transfer.",
        "loss_note": "local_loss_* is the mean per-example TRAINING loss reported by clients "
                     "(normalized scale, MSE).",
    }
    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    csv_path = out_dir / "round_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    best_mae = float("inf")
    best_round = None
    history = []
    run_start = time.perf_counter()

    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()

        # 1-2. This round's broadcast state (CPU copy of the current global model).
        broadcast = _clone_state(global_model.state_dict())

        states, counts, losses, used_ids = [], [], [], []
        for idx, spec in enumerate(clients):
            set_seed(cfg.seed + rnd * 1000 + idx)     # reproducible per-client shuffling

            # 3. EVERY client gets its own deep copy of the SAME broadcast state,
            #    so no client can ever see another client's updated weights.
            state, n, metrics = train_client_fn(
                global_state_dict=copy.deepcopy(broadcast),
                client_path=spec.path,
                mean=mean,
                std=std,
                local_epochs=cfg.local_epochs,
                learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )

            n = int(n)
            # The client reports examples *processed across local epochs*; the contract
            # wants unique examples. Every client is scaled by the same factor, so the
            # FedAvg weights are unaffected, but normalize here so counts/logs are right.
            if (cfg.local_epochs > 1 and spec.expected_examples is not None
                    and n == spec.expected_examples * cfg.local_epochs):
                n = spec.expected_examples
            if n <= 0:
                log.warning("Round %d: client %s reported 0 examples -> SKIPPED.", rnd, spec.client_id)
                continue
            if spec.expected_examples is not None and spec.expected_examples != n:
                log.warning(
                    "Round %d: client %s trained on %d examples but manifest says %d.",
                    rnd, spec.client_id, n, spec.expected_examples,
                )

            states.append(_clone_state(state))
            counts.append(n)
            used_ids.append(spec.client_id)
            metrics = metrics or {}
            loss = metrics.get("train_loss", metrics.get("loss"))
            losses.append(float("nan") if loss is None else float(loss))

        if not states:
            raise RuntimeError(f"Round {rnd}: no client produced an update.")

        # 6-7. Aggregate once, after ALL clients have returned.
        aggregated = fedavg(states, counts)
        global_model.load_state_dict(aggregated)

        # 8. Validation only (test split is unreachable from here).
        val = validation_fn(global_model, mean, std, cfg.device, cfg.batch_size, cfg.max_val_batches)

        total_n = int(sum(counts))
        weighted_loss = float(sum(n * l for n, l in zip(counts, losses)) / total_n)
        simple_loss = float(np.mean(losses))

        is_best = val["mae_norm"] < best_mae
        if is_best:
            best_mae, best_round = val["mae_norm"], rnd
            _save_checkpoint(out_dir / "best_model.pt", global_model, rnd, val, mean, std, cfg)

        # "final_model.pt" is refreshed every round, so a crash still leaves the latest state.
        _save_checkpoint(out_dir / "final_model.pt", global_model, rnd, val, mean, std, cfg)

        row = {
            "mode": cfg.mode,
            "seed": cfg.seed,
            "round": rnd,
            "participating_clients": len(states),
            "total_client_examples": total_n,
            "local_loss_sample_weighted_mean": weighted_loss,
            "local_loss_simple_mean": simple_loss,
            "val_mae_norm": val["mae_norm"],
            "val_rmse_norm": val["rmse_norm"],
            "val_mae_orig_d0": val.get("mae_orig_d0", ""),
            "val_rmse_orig_d0": val.get("rmse_orig_d0", ""),
            "val_mae_orig_d1": val.get("mae_orig_d1", ""),
            "val_rmse_orig_d1": val.get("rmse_orig_d1", ""),
            "duration_sec": round(time.perf_counter() - t0, 3),
            "est_comm_bytes": 2 * model_bytes * len(states),
            "is_best": int(is_best),
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        history.append(row)

        log.info(
            "[%s] round %d/%d | clients=%d n=%d | local loss=%.5f | val MAE=%.5f RMSE=%.5f%s | %.1fs",
            cfg.mode, rnd, cfg.rounds, len(states), total_n, weighted_loss,
            val["mae_norm"], val["rmse_norm"], "  *best*" if is_best else "", row["duration_sec"],
        )

    config_record.update(
        completed_rounds=cfg.rounds,
        best_round=best_round,
        best_val_mae_norm=best_mae,
        total_duration_sec=round(time.perf_counter() - run_start, 2),
    )
    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    return {"history": history, "best_round": best_round, "best_val_mae_norm": best_mae,
            "output_dir": str(out_dir), "model": global_model}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Federated TrafficMLP training (FedAvg).")
    p.add_argument("--mode", choices=["iid", "non_iid"], required=True)
    p.add_argument("--manifest", default=None,
                   help="default: data/federated/<mode>/partition_manifest.json")
    p.add_argument("--output-dir", default=None, help="default: results/federated/<mode>")
    p.add_argument("--rounds", type=int, default=None, help="default 2 (5 for the first full run)")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="limit validation batches (smoke tests only)")
    p.add_argument("--smoke", action="store_true",
                   help="2 rounds and 5 validation batches unless overridden")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args(argv)

    rounds = a.rounds if a.rounds is not None else 2
    max_val = a.max_val_batches
    if a.smoke:
        rounds = a.rounds if a.rounds is not None else 2
        max_val = max_val if max_val is not None else 5

    cfg = FedConfig(
        mode=a.mode,
        manifest_path=a.manifest or f"data/federated/{a.mode}/partition_manifest.json",
        output_dir=a.output_dir or (
            f"results/federated/{a.mode}" + ("_smoke" if a.smoke else "")
        ),
        rounds=rounds,
        local_epochs=a.local_epochs,
        learning_rate=a.lr,
        batch_size=a.batch_size,
        seed=a.seed,
        device=a.device,
        max_val_batches=max_val,
    )
    result = run_federated(cfg)
    log.info("Done. Best round: %s (val MAE %.5f). Outputs in %s",
             result["best_round"], result["best_val_mae_norm"], result["output_dir"])


if __name__ == "__main__":
    main()
