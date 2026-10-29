"""Shared 16-channel CNN/XAI-CNN for 100- or 20-cycle capacity matrices."""
from __future__ import annotations

import torch
from torch import nn
from typing import Literal

Activation = Literal["relu", "leaky_relu"]


class CNN(nn.Module):
    """ReLU comparator or the LeakyReLU(0.3) XAI-CNN."""

    def __init__(self, activation: Activation = "relu", input_shape: tuple[int, int] = (100, 100)) -> None:
        """Construct a scalar predictor; adapt only the full-spatial projection.

        ``input_shape`` gives cycle and voltage dimensions. The 100 x 100
        network is checkpoint-compatible; MIX-20 uses a 1 x 10 projection.
        """
        super().__init__()
        if activation not in ("relu", "leaky_relu"):
            raise ValueError("activation must be relu or leaky_relu")
        if input_shape not in ((100, 100), (20, 100)):
            raise ValueError("input_shape must be (100,100) or (20,100)")
        self.activation = activation
        self.input_shape = input_shape
        act = nn.ReLU if activation == "relu" else lambda: nn.LeakyReLU(0.3)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3), act(), nn.AvgPool2d(3),
            nn.Conv2d(16, 16, 3), act(), nn.AvgPool2d(3),
        )
        projection_shape = tuple(((size - 2) // 3 - 2) // 3 for size in input_shape)
        self.proj = nn.Conv2d(16, 16, projection_shape)
        self.act = act()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map float32 (N, 1, H, W) matrices to (N,) log/Z-score RUL values."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, *self.input_shape):
            raise ValueError(f"Expected NCHW input (N,1,{self.input_shape[0]},{self.input_shape[1]})")
        return self.linear(self.flatten(self.act(self.proj(self.cnn(x))))).view(-1)


def make_model(activation: Activation = "relu", seed: int = 0,
               input_shape: tuple[int, int] = (100, 100)) -> CNN:
    """Seed PyTorch and construct a CNN with default parameter initialization."""
    torch.manual_seed(seed)
    return CNN(activation, input_shape)
