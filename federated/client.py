"""Local training step for one simulated federated client."""

import numpy as np
import torch
from torch import nn

from src.model import TrafficMLP
from federated.client_data_loader import iter_client_batches


def train_client(
    global_state_dict,
    client_path,
    mean,
    std,
    local_epochs=1,
    learning_rate=0.001,
    batch_size=4096,
    device="cpu",
):
    """
    Train one client using the received global model weights.

    Args:
        global_state_dict: Current global model state from the server.
        client_path: Path to this client's training Parquet file.
        mean: Shared feature means fitted on training inputs.
        std: Shared feature standard deviations.
        local_epochs: Number of local training epochs.
        learning_rate: Adam learning rate.
        batch_size: Number of examples per batch.
        device: "cpu" or an available supported device.

    Returns:
        updated_state_dict: Locally trained model weights.
        num_examples: Number of examples processed across local epochs.
        metrics: Dictionary containing mean training loss.
    """

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = torch.device(device)

    # Create the existing model architecture.
    model = TrafficMLP().to(device)

    # Every client starts with the global weights supplied by the server.
    model.load_state_dict(global_state_dict, strict=True)

    # Use the optimizer and loss agreed in the project contract.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_examples = 0

    model.train()

    for epoch in range(local_epochs):
        # Use a repeatable but different shuffle order for each epoch.
        for X_np, y_np, _metadata in iter_client_batches(
            client_path=client_path,
            mean=mean,
            std=std,
            batch_size=batch_size,
            shuffle=True,
            seed=42 + epoch,
        ):
            X = torch.as_tensor(
                X_np,
                dtype=torch.float32,
                device=device,
            )

            y = torch.as_tensor(
                y_np,
                dtype=torch.float32,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            predictions = model(X)
            loss = criterion(predictions, y)

            loss.backward()
            optimizer.step()

            batch_examples = int(X.shape[0])

            # Weight batch means by the number of examples.
            total_loss += (
                float(loss.detach().item()) * batch_examples
            )

            total_examples += batch_examples

    if total_examples == 0:
        raise ValueError(
            f"Client partition contains no examples: {client_path}"
        )

    average_loss = total_loss / total_examples

    # Return independent CPU copies of the updated model parameters.
    updated_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    metrics = {
        "train_loss": average_loss,
        "loss": average_loss,
    }

    return updated_state_dict, total_examples, metrics