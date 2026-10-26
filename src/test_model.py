import torch

from model import TrafficMLP


def main():
    model = TrafficMLP()

    # Simulate one batch from your data loader.
    X = torch.randn(32, 12, 2)

    predictions = model(X)

    print("Input shape:", X.shape)
    print("Output shape:", predictions.shape)
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    assert predictions.shape == (32, 12, 2)

    print("Model test passed!")


if __name__ == "__main__":
    main()