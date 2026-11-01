import random
import csv
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from .model import TrafficMLP

from preprocessing.data_loader import (
    fit_input_scaler,
    iter_batches,
    BATCH_SIZE,
)


# -----------------------------
# Configuration
# -----------------------------

EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.pth"

SEED = 42


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(model, mean, std, device):
    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_values = 0

    with torch.no_grad():
        for X, y, metadata in iter_batches(
            "val",
            mean,
            std,
            batch_size=BATCH_SIZE,
        ):
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)

            predictions = model(X)

            squared_error = (predictions - y) ** 2
            absolute_error = torch.abs(predictions - y)

            total_squared_error += squared_error.sum().item()
            total_absolute_error += absolute_error.sum().item()
            total_values += y.numel()

    mse = total_squared_error / total_values
    mae = total_absolute_error / total_values
    rmse = np.sqrt(mse)

    return mae, rmse


# -----------------------------
# Training
# -----------------------------

def train():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading training scaler...")

    # Fit normalization statistics using training inputs only.
    mean, std = fit_input_scaler()

    model = TrafficMLP().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    history_path = MODEL_DIR / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history_writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_mse", "val_mae", "val_rmse"])
        history_writer.writeheader()

    print("\nStarting training...")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_samples = 0

        # Read training data in batches.
        for X, y, metadata in iter_batches(
            "train",
            mean,
            std,
            batch_size=BATCH_SIZE,
        ):
            X = torch.as_tensor(
                X,
                dtype=torch.float32,
                device=device,
            )

            y = torch.as_tensor(
                y,
                dtype=torch.float32,
                device=device,
            )

            # 1. Clear gradients from the previous batch.
            optimizer.zero_grad()

            # 2. Predict the next 12 time steps.
            predictions = model(X)

            # 3. Calculate prediction error.
            loss = criterion(predictions, y)

            # 4. Calculate gradients.
            loss.backward()

            # 5. Update model parameters.
            optimizer.step()

            # Accumulate loss weighted by batch size.
            batch_samples = X.shape[0]

            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        train_loss = total_loss / total_samples

        # Evaluate on validation data after each epoch.
        val_mae, val_rmse = evaluate(
            model,
            mean,
            std,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train MSE: {train_loss:.6f} "
            f"| Validation MAE: {val_mae:.6f} "
            f"| Validation RMSE: {val_rmse:.6f}"
        )
        # The live demo reads this file to draw the centralized epoch curve.
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            history_writer = csv.DictWriter(
                history_file,
                fieldnames=["epoch", "train_mse", "val_mae", "val_rmse"],
            )
            history_writer.writerow({
                "epoch": epoch + 1,
                "train_mse": train_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            })

        # Save the model whenever validation MAE improves.
        if val_mae < best_val_mae:
            best_val_mae = val_mae

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_mae": val_mae,
                    "validation_rmse": val_rmse,
                    "mean": [float(x) for x in np.asarray(mean).ravel()],
                    "std": [float(x) for x in np.asarray(std).ravel()],
                },
                MODEL_PATH,
            )

            print("  Best model saved!")

    print("\nTraining completed!")
    print(f"Best validation MAE: {best_val_mae:.6f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()