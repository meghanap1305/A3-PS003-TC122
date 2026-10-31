"""
AIORI live traffic-learning demo server.

Run from the repository root:
    python -m demo.server

The server serves demo/static/index.html and exposes:
    GET /api/state   - live training metrics/config/partition metadata
    GET /api/replay  - predictions from the current checkpoints on val.parquet

It reads the real project artifacts; it does not generate model predictions.
The visualization replays the validation split only. It never reads test.parquet.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
FED_MODES = ("iid", "non_iid")
VAL_COLUMNS = ("val_mae", "val_mae_norm", "validation_mae")
CHECKPOINT_NAMES = ("best_model.pt", "final_model.pt", "best_model.pth")

ARGS = None
LOCK = threading.RLock()
REPLAY = {"version": 0, "modes": {}, "status": {}, "key": None, "busy": False}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {}
            for key, value in row.items():
                try:
                    clean[key] = float(value)
                except (TypeError, ValueError):
                    clean[key] = value
            rows.append(clean)
        return rows


def checkpoint_of(directory):
    for name in CHECKPOINT_NAMES:
        path = Path(directory) / name
        if path.is_file():
            return path
    return None


def central_checkpoint():
    candidates = [
        Path(ARGS.central_checkpoint),
        Path("models/best_model.pth"),
        Path("models/best_model.pt"),
        Path(ARGS.central_dir) / "best_model.pt",
        Path(ARGS.central_dir) / "best_model.pth",
    ]
    return next((p for p in candidates if p.is_file()), None)


def central_history():
    candidates = [
        Path("models/training_history.csv"),
        Path(ARGS.central_dir) / "training_history.csv",
        Path(ARGS.central_dir) / "epochs.csv",
    ]
    for path in candidates:
        rows = read_csv(path)
        if rows:
            col = next((c for c in VAL_COLUMNS if c in rows[0]), None)
            if col is None:
                col = next((c for c in rows[0]
                            if "val" in c.lower() and "mae" in c.lower()), None)
            if col:
                # Read MAE and RMSE from the same epoch-history file so both
                # charts use the same validation epochs and metric scale.
                rmse_col = next((c for c in ("val_rmse", "val_rmse_norm", "validation_rmse")
                                 if c in rows[0]), None)
                if rmse_col is None:
                    rmse_col = next((c for c in rows[0]
                                     if "val" in c.lower() and "rmse" in c.lower()), None)
                epoch_col = next((c for c in rows[0]
                                  if c.lower() in ("epoch", "round")), None)
                return [
                    {"epoch": int(r.get(epoch_col, i + 1)) if epoch_col else i + 1,
                     "val_mae": float(r[col]),
                     "val_rmse": (float(r[rmse_col]) if rmse_col and
                                  isinstance(r.get(rmse_col), (int, float)) else None)}
                    for i, r in enumerate(rows)
                    if isinstance(r.get(col), (int, float))
                ], path
    return [], None


def fresh(path, seconds=None):
    if not path or not Path(path).is_file():
        return False
    return time.time() - Path(path).stat().st_mtime < (
        ARGS.live_window if seconds is None else seconds
    )


def build_state():
    results = Path(ARGS.results)
    federated = Path(ARGS.federated)
    modes = {}

    for mode in FED_MODES:
        mode_dir = results / mode
        csv_path = mode_dir / "round_metrics.csv"
        cfg = read_json(mode_dir / "config.json")
        manifest = read_json(federated / mode / "partition_manifest.json")
        clients = []
        for client in manifest.get("clients", []):
            clients.append({
                "client_id": client.get("client_id"),
                "num_examples": client.get("num_examples"),
                "n_sensors": len(client.get("sensor_ids", [])),
            })
        if manifest:
            manifest["clients"] = clients
        rounds = read_csv(csv_path)
        modes[mode] = {
            "rounds": rounds,
            "config": cfg,
            "manifest": manifest or None,
            "running": fresh(csv_path) and (
                not cfg.get("rounds") or len(rounds) < int(cfg["rounds"])
            ),
        }

    epochs, epoch_path = central_history()
    train_file = Path(ARGS.processed) / "train.parquet"
    ckpt = central_checkpoint()
    central = {
        "epochs": epochs,
        "running": bool(epoch_path and fresh(epoch_path) and
                        (not epochs or len(epochs) < int(ARGS.central_epochs))),
        "data_bytes": train_file.stat().st_size if train_file.is_file() else None,
        "checkpoint": str(ckpt) if ckpt else None,
    }

    final_metrics = {
        "iid": read_json(results / "iid" / "best_model_test_metrics.json") or None,
        "non_iid": read_json(results / "non_iid" / "best_model_test_metrics.json") or None,
        "classical": read_json(Path("models/best_model_test_metrics.json")) or
                     read_json(Path(ARGS.central_dir) / "best_model_test_metrics.json") or None,
    }
    with LOCK:
        return {
            "final_test": final_metrics,
            "modes": modes,
            "classical": central,
            "replay_version": REPLAY["version"],
            "replay_status": dict(REPLAY["status"]),
            "server_time": time.time(),
        }



def load_predictor(checkpoint_path):
    import torch
    from src.model import TrafficMLP

    # Load the trained model checkpoint.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

    model = TrafficMLP()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if "mean" in checkpoint and "std" in checkpoint:
        mean = np.asarray(
            checkpoint["mean"], dtype=np.float32
        ).reshape(1, 1, -1)

        std = np.asarray(
            checkpoint["std"], dtype=np.float32
        ).reshape(1, 1, -1)
    else:
        from preprocessing.data_loader import fit_input_scaler

        mean_raw, std_raw = fit_input_scaler()

        mean = np.asarray(
            mean_raw, dtype=np.float32
        ).reshape(1, 1, -1)

        std = np.asarray(
            std_raw, dtype=np.float32
        ).reshape(1, 1, -1)

    def predict(x_original):
        x_norm = (x_original - mean) / std

        with torch.no_grad():
            output = model(
                torch.as_tensor(x_norm, dtype=torch.float32)
            ).cpu().numpy()

        return output * std + mean

    return predict, checkpoint


def make_sensor_groups(val_path, manifest, clients_count=3, sensors_per_client=0):
    """non-IID uses the real disjoint sensor ownership from its manifest.
    IID uses balanced validation-sensor buckets for visualization only: IID
    training partitions are random examples, not fixed sensor ownership.
    """
    import pyarrow.parquet as pq

    all_sensors = sorted(
        set(pq.read_table(val_path, columns=["node_id"]).column(0).to_pylist()),
        key=lambda v: (str(type(v)), str(v)),
    )
    count = clients_count
    limit = sensors_per_client or max(1, int(np.ceil(len(all_sensors) / count)))
    even = [all_sensors[i::count][:limit] for i in range(count)]

    clients = (manifest or {}).get("clients", [])
    owned = [set(c.get("sensor_ids", [])) for c in clients[:count]]
    disjoint = len(owned) == count and all(
        not (owned[i] & owned[j])
        for i in range(count) for j in range(i)
    )
    if (manifest or {}).get("mode") == "non_iid" and disjoint:
        available = set(all_sensors)
        groups = [
            sorted([sensor for sensor in group if sensor in available],
                   key=lambda v: (str(type(v)), str(v)))[:limit]
            for group in owned
        ]
        ids = [c.get("client_id", f"client_{i+1}")
               for i, c in enumerate(clients[:count])]
        return groups, ids, "Actual sensor ownership from the non-IID partition manifest."

    ids = [c.get("client_id", f"client_{i+1}")
           for i, c in enumerate(clients[:count])]
    if len(ids) != count:
        ids = [f"client_{i+1}" for i in range(count)]
    return even, ids, (
        "Display buckets only: IID training assigns individual examples randomly, "
        "so these replay sensor buckets are not fixed client ownership."
    )


def get_sensor_windows(val_path, sensor_ids, cache):
    key = tuple(sensor_ids)
    if key in cache:
        return cache[key]

    import pyarrow.parquet as pq
    from preprocessing.data_loader import (
        INPUT_COLUMNS, METADATA_COLUMNS, TARGET_COLUMNS
    )

    columns = METADATA_COLUMNS + INPUT_COLUMNS + TARGET_COLUMNS
    table = pq.read_table(
        val_path,
        columns=columns,
        filters=[("node_id", "in", list(sensor_ids))],
    )
    frame = table.to_pandas()
    frame = frame.sort_values(["node_id", "t0_timestamp"])
    per_sensor = [
        group.iloc[ARGS.start:ARGS.start + ARGS.points].copy()
        for _, group in frame.groupby("node_id", sort=False)
    ]
    cache[key] = per_sensor
    return per_sensor


def replay_checkpoint(checkpoint_path, groups, client_ids, val_path, cache, mode):
    from preprocessing.data_loader import (
        INPUT_COLUMNS, INPUT_STEPS, PREDICTION_STEPS,
        TARGET_COLUMNS, reshape_windows,
    )

    predict, checkpoint = load_predictor(checkpoint_path)
    clients_out = []

    for client_id, sensor_ids in zip(client_ids, groups):
        series = []
        for frame in get_sensor_windows(val_path, sensor_ids, cache):
            if frame.empty:
                continue
            x = reshape_windows(frame, INPUT_COLUMNS, INPUT_STEPS)
            y = reshape_windows(frame, TARGET_COLUMNS, PREDICTION_STEPS)
            prediction = predict(x)
            nodes = frame["node_id"].tolist()
            timestamps = frame["t0_timestamp"].astype(str).tolist()

            # One record per validation anchor:
            # [timestamp, sensor_id, current_d0, current_d1,
            #  12 * predicted feature pairs, 12 * actual feature pairs].
            # This preserves the complete 5..60 minute forecast horizon.
            series.append([
                [
                    timestamps[i], nodes[i],
                    *np.round(x[i, -1], 4).tolist(),
                    *np.round(prediction[i], 4).reshape(-1).tolist(),
                    *np.round(y[i], 4).reshape(-1).tolist(),
                ]
                for i in range(len(frame))
            ])

        points = [
            sequence[i]
            for i in range(max((len(s) for s in series), default=0))
            for sequence in series if i < len(sequence)
        ]
        if points:
            predicted = np.asarray([p[4:28] for p in points], dtype=np.float32).reshape(-1, 12, 2)
            actual = np.asarray([p[28:52] for p in points], dtype=np.float32).reshape(-1, 12, 2)
            mae_by_horizon = np.abs(predicted - actual).mean(axis=0).round(4).tolist()
        else:
            mae_by_horizon = np.zeros((12, 2), dtype=np.float32).tolist()
        clients_out.append({
            "client_id": client_id,
            "points": points,
            "mae": mae_by_horizon[-1],
            "mae_by_horizon": mae_by_horizon,
            "sensor_count": len(sensor_ids),
            "sensor_ids": sensor_ids,
        })

    return clients_out, checkpoint


def rebuild_replay():
    val_path = Path(ARGS.processed) / "val.parquet"
    if not val_path.is_file():
        with LOCK:
            REPLAY["status"] = {"all": f"waiting for {val_path}"}
        return

    fed_root = Path(ARGS.federated)
    result_root = Path(ARGS.results)
    checkpoints = {
        "iid": checkpoint_of(result_root / "iid"),
        "non_iid": checkpoint_of(result_root / "non_iid"),
        "classical": central_checkpoint(),
    }
    key = tuple(
        (mode, str(path), path.stat().st_mtime if path else None)
        for mode, path in checkpoints.items()
    )
    with LOCK:
        if key == REPLAY["key"] or REPLAY["busy"]:
            return
        REPLAY["busy"] = True

    modes, status, cache = {}, {}, {}
    iid_groups = iid_ids = None
    try:
        for mode, checkpoint in checkpoints.items():
            if checkpoint is None:
                status[mode] = "waiting for model checkpoint"
                continue
            try:
                manifest = (
                    read_json(fed_root / mode / "partition_manifest.json")
                    if mode in FED_MODES else None
                )
                groups, ids, note = make_sensor_groups(
                    val_path, manifest,
                    clients_count=ARGS.clients,
                    sensors_per_client=ARGS.sensors_per_client,
                )
                if mode == "iid":
                    iid_groups, iid_ids = groups, ids
                elif mode == "classical" and iid_groups is not None:
                    groups, ids = iid_groups, [
                        f"client_{i+1}" for i in range(len(iid_groups))
                    ]

                clients, ckpt = replay_checkpoint(
                    checkpoint, groups, ids, val_path, cache, mode
                )
                modes[mode] = clients
                step = ckpt.get("round", ckpt.get("epoch"))
                status[mode] = (
                    f"checkpoint round {step}" if mode != "classical" and step
                    else f"checkpoint epoch {step}" if step
                    else "checkpoint loaded"
                )
                status[mode + "_note"] = note
            except Exception as exc:
                status[mode] = f"error: {type(exc).__name__}: {exc}"
    finally:
        with LOCK:
            REPLAY.update(
                modes=modes,
                status=status,
                key=key,
                version=REPLAY["version"] + 1,
                busy=False,
            )


def replay_watcher():
    while True:
        try:
            rebuild_replay()
        except Exception as exc:
            print("Replay refresh error:", repr(exc), flush=True)
        time.sleep(ARGS.watch_every)


def clean_json(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    return value


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(clean_json(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            return self.send_json(build_state())
        if path == "/api/replay":
            with LOCK:
                snapshot = {
                    "version": REPLAY["version"],
                    "modes": REPLAY["modes"],
                    "status": REPLAY["status"],
                }
            return self.send_json(snapshot)
        if path == "/api/health":
            return self.send_json({"ok": True, "server_time": time.time()})
        if path in ("/", "/index.html"):
            html = (Path(__file__).parent / "static" / "index.html")
            if not html.is_file():
                return self.send_json({"error": "static/index.html not found"}, 404)
            data = html.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return self.send_json({"error": "not found"}, 404)

    def log_message(self, *_args):
        pass


def main():
    global ARGS
    parser = argparse.ArgumentParser(description="AIORI live traffic-learning visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--results", default="results/federated")
    parser.add_argument("--federated", default="data/federated")
    parser.add_argument("--central-dir", default="results/centralized")
    parser.add_argument("--central-checkpoint", default="models/best_model.pth")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--sensors-per-client", type=int, default=0)
    parser.add_argument("--points", type=int, default=288)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--central-epochs", type=int, default=5)
    parser.add_argument("--live-window", type=float, default=120)
    parser.add_argument("--watch-every", type=float, default=5)
    ARGS = parser.parse_args()

    threading.Thread(target=replay_watcher, daemon=True).start()
    print(f"AIORI live demo: http://{ARGS.host}:{ARGS.port}", flush=True)
    print("Reading real checkpoints and validation data. Ctrl+C to stop.", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
