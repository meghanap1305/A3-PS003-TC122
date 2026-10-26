import torch
import torch.nn as nn


class TrafficMLP(nn.Module):
    """
    Compact traffic forecasting model.

    Input:
        (batch_size, 12, 2)

    Output:
        (batch_size, 12, 2)
    """

    def __init__(
        self,
        input_steps=12,
        num_features=2,
        prediction_steps=12
    ):
        super().__init__()

        self.input_steps = input_steps
        self.num_features = num_features
        self.prediction_steps = prediction_steps

        input_size = input_steps * num_features
        output_size = prediction_steps * num_features

        self.network = nn.Sequential(
            nn.Linear(input_size, 128), #first arg input size == weights and 128 (2nd) == bias
            #y=wx+b 
            nn.ReLU(), #rectified linear unit max(0,x)
            #relu introuces non linearity 
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # (batch, 12, 2) -> (batch, 24)
        x = x.reshape(batch_size, -1)

        # (batch, 24) -> (batch, 24)
        x = self.network(x)

        # (batch, 24) -> (batch, 12, 2)
        x = x.reshape(
            batch_size,
            self.prediction_steps,
            self.num_features
        )

        return x