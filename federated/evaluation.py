"""Evaluation of a global TrafficMLP.

Reports metrics on TWO scales so every comparison in the report is valid:

* ``*_norm``  : normalized scale, pooled over all target elements. This is
                exactly what ``src/train.py::evaluate`` computes (centralized MLP).
* ``*_orig_dK``: original units, per feature K. This is what
                ``preprocessing/baseline.py`` computes (persistence baseline).

Test-split access is blocked unless ``allow_test=True``. The runner never sets
it; only the final CLI below does (with --confirm-final).

Final test evaluation (after settings are frozen):

    python -m federated.evaluation \
        --checkpoint results/federated/iid/best_model.pt --confirm-final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def evaluate_global(
    model,
    mean,
    std,
    device="cpu",
    batch_size=None,
    split="val",
    max_batches=None,
    allow_test=False,
):
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test' (never 'train' here).")
    if split == "test" and not allow_test:
        raise PermissionError(
            "Test split is reserved for the final evaluation. "
            "Pass allow_test=True only after settings are frozen."
        )

    # Imported lazily so unit tests that inject fakes don't need pyarrow.
    from preprocessing.data_loader import BATCH_SIZE, iter_batches

    if batch_size is None:
        batch_size = BATCH_SIZE

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32, device=device).view(1, 1, -1)
    std_t = torch.as_tensor(np.asarray(std), dtype=torch.float32, device=device).view(1, 1, -1)
    num_features = mean_t.shape[-1]

    abs_norm = torch.zeros(num_features, dtype=torch.float64)
    sq_norm = torch.zeros(num_features, dtype=torch.float64)
    abs_orig = torch.zeros(num_features, dtype=torch.float64)
    sq_orig = torch.zeros(num_features, dtype=torch.float64)
    num_examples = 0
    num_batches = 0
    steps = None

    with torch.no_grad():
        for X, y, _metadata in iter_batches(split, mean, std, batch_size=batch_size):
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)

            err = model(X) - y                      # normalized scale
            err_orig = err * std_t                  # original units (shift cancels)

            abs_norm += err.abs().sum(dim=(0, 1)).double().cpu()
            sq_norm += (err ** 2).sum(dim=(0, 1)).double().cpu()
            abs_orig += err_orig.abs().sum(dim=(0, 1)).double().cpu()
            sq_orig += (err_orig ** 2).sum(dim=(0, 1)).double().cpu()

            num_examples += X.shape[0]
            steps = y.shape[1]
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    if num_examples == 0:
        raise ValueError(f"No examples found in the '{split}' split.")

    per_feature_count = num_examples * steps            # elements per feature
    total_count = per_feature_count * num_features

    result = {
        "split": split,
        "num_examples": int(num_examples),
        "num_batches": int(num_batches),
        "truncated": max_batches is not None and num_batches >= max_batches,
        # Same definition as src/train.py::evaluate (pooled, normalized).
        "mae_norm": float(abs_norm.sum() / total_count),
        "rmse_norm": float(torch.sqrt(sq_norm.sum() / total_count)),
    }
    for k in range(num_features):
        result[f"mae_norm_d{k}"] = float(abs_norm[k] / per_feature_count)
        result[f"rmse_norm_d{k}"] = float(torch.sqrt(sq_norm[k] / per_feature_count))
        # Same definition as preprocessing/baseline.py (original units).
        result[f"mae_orig_d{k}"] = float(abs_orig[k] / per_feature_count)
        result[f"rmse_orig_d{k}"] = float(torch.sqrt(sq_orig[k] / per_feature_count))
    return result


def _final_test_cli():
    parser = argparse.ArgumentParser(description="Final TEST evaluation of a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required. Confirms settings/checkpoint selection are frozen.",
    )
    args = parser.parse_args()

    if not args.confirm_final:
        raise SystemExit("Refusing to touch the test split without --confirm-final.")

    from src.model import TrafficMLP

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TrafficMLP()
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_global(
        model,
        ckpt["mean"],
        ckpt["std"],
        device=args.device,
        batch_size=args.batch_size,
        split="test",
        allow_test=True,
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["checkpoint_round"] = ckpt.get("round")

    out_path = Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_test_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Saved to:", out_path)


if __name__ == "__main__":
    _final_test_cli()
