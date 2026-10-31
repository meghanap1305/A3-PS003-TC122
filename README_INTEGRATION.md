# AIORI Live Traffic Learning Demo — integration overlay

This overlay adds a live browser dashboard to the existing A3-PS003-TC122 project.
It does not replace the model, data loader, client training, FedAvg, partitioning,
or evaluation implementation.

## What the dashboard reads

- `data/processed/val.parquet`: validation-only sensor replay.
- `data/processed/train.parquet`: file-size estimate for centralized raw-data movement.
- `data/federated/iid/partition_manifest.json`
- `data/federated/non_iid/partition_manifest.json`
- `results/federated/iid/{config.json,round_metrics.csv,best_model.pt|final_model.pt}`
- `results/federated/non_iid/{config.json,round_metrics.csv,best_model.pt|final_model.pt}`
- `models/best_model.pth`: centralized `TrafficMLP` checkpoint.
- `models/training_history.csv`: per-epoch centralized validation history (written by the patched `src/train.py`).

The animation uses the model checkpoint to predict on `val.parquet`. It shows the last
forecast horizon (+60 minutes) and the corresponding actual validation target. The
selected feature `d0` or `d1` controls the displayed feature. It does not assume that
either feature is a physical speed measurement: car speed/congestion is a visual mapping
of the feature's relative value.

## Install overlay

Copy these files into the root of your existing repository, preserving folders:

- `demo/server.py`
- `demo/__init__.py`
- `demo/static/index.html`
- `src/train.py` (patched version with epoch-history CSV logging)

The patched `src/train.py` is your existing centralized trainer plus a small CSV logger.
All other project files remain unchanged.

## Run

Run commands from the repository root, in separate terminals.

1. Ensure processed splits exist:
   `data/processed/train.parquet`, `val.parquet`, `test.parquet`

2. Create both federated partitions:
   `python partition_data.py --mode iid`
   `python partition_data.py --mode non_iid`

3. Train the centralized model:
   `python -m src.train`

4. Train federated IID:
   `python -m federated.federated_train --mode iid --rounds 5`

5. Train federated non-IID:
   `python -m federated.federated_train --mode non_iid --rounds 5`

   You can run the two federated commands in separate terminals if desired.
   The dashboard updates as the CSVs and checkpoints are written. A checkpoint is
   needed before that model's road replay can appear.

6. Start the dashboard:
   `python -m demo.server`

7. Open:
   `http://127.0.0.1:8000`

If the project already has trained checkpoints, you can start only the server to
replay them. The frontend polls the state API every 2 seconds; checkpoint replay
refreshes when checkpoint modification times change.

## Notes on interpretation

- The three panels compare centralized training, federated IID, and federated non-IID.
  IID and non-IID use the same `TrafficMLP` architecture and differ in data partitioning.
- IID training randomly assigns examples to clients. To show three distinguishable
  traffic streams, the validation replay uses balanced sensor buckets as a *visual
  grouping only*. Those buckets are not the actual IID partition ownership.
- non-IID replay uses the sensor ownership recorded in the partition manifest.
- The +60-minute value is the 12th forecast step (12 × 5 minutes).
- Validation MAE curves are used for training progress. Test metrics are displayed
  only if final evaluation JSON files already exist; the replay never reads test data.
- Communication volume is explicitly an estimate. Federated bytes use
  `est_comm_bytes` from the training log (upload + download model size per client
  per round). Centralized bytes use the training Parquet file size as a simple
  one-time raw-data transfer proxy. They are not a network packet capture.
- The project uses simulated clients in one Python process; animated packets are
  explanatory visualization, not actual network traffic.
- Federated learning keeps raw training partitions at clients in this simulation,
  but this alone is not a formal privacy guarantee. No secure aggregation,
  differential privacy, or encrypted transport is implemented here.
