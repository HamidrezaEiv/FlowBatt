"""Supervised SOH Transformer with its own capacity-to-trajectory encoder.

Attention blocks reuse the DiT-derived implementation in models/dit.py.
See its upstream attribution and the repository-root LICENSE (CC BY-NC 4.0).
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from flowbatt.models.dit import get_1d_sincos_pos_embed, DiTBlock, FinalLayer

class CNNEmbedder(nn.Module):
    """
    Encodes capacity matrices into a sequence-sized vector, with optional dropout.
    """
    def __init__(self, input_shape: tuple[int, int, int], hidden_size: int,
                 output_size: int, dropout_prob: float, **block_kwargs: object) -> None:
        """Configure convolutional features and the linear trajectory projection."""
        super().__init__()

        in_channels, height, width = input_shape
        self.kernel_size = (3, 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, self.kernel_size),
            nn.LeakyReLU(negative_slope=0.3),
            nn.AvgPool2d(self.kernel_size),
            nn.Conv2d(hidden_size, hidden_size, self.kernel_size),
            nn.LeakyReLU(negative_slope=0.3),
            nn.AvgPool2d(self.kernel_size)
        )
        H, W = self.output_shape(height, width)

        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.proj = nn.Conv2d(hidden_size, 1, (1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(H*W, output_size)

        self.dropout_prob = dropout_prob
    
    def output_shape(self, H: int, W: int) -> tuple[int, int]:
        """Return spatial dimensions after two valid convolutions and poolings."""
        # conv1 output
        H, W = H - self.kernel_size[0] + 1, W - self.kernel_size[1] + 1
        # pool1 output
        H = int((H - self.kernel_size[0]) / self.kernel_size[0] + 1)
        W = int((W - self.kernel_size[1]) / self.kernel_size[1] + 1)
        # conv2 output
        H, W = H - self.kernel_size[0] + 1, W - self.kernel_size[1] + 1
        # pool2 output
        H = int((H - self.kernel_size[0]) / self.kernel_size[0] + 1)
        W = int((W - self.kernel_size[1]) / self.kernel_size[1] + 1)

        return H, W

    def token_drop(self, labels: Tensor, force_drop_ids: Tensor | None = None) -> Tensor:
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return labels * (1.0 - drop_ids.unsqueeze(1).float())  # Zero out dropped labels

    def forward(self, x: Tensor, train: bool, force_drop_ids: Tensor | None = None) -> Tensor:
        """Encode capacity matrices (B, C, H, W) into conditioning vectors (B, D)."""
        x = self.cnn(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.linear(x)

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            x = self.token_drop(x, force_drop_ids)

        return x

class TrF(nn.Module):
    """Predict normalized full SOH trajectories directly from capacity matrices."""
    def __init__(
        self,
        input_dim: int = 256,
        input_shape: tuple[int, int, int] = (1, 100, 100),
        num_channels: int = 16,
        num_blocks: int = 2,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
    ) -> None:
        """Set sequence length, encoder shape, hidden width, and attention depth."""
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, num_channels), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(num_channels, input_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.y_embedder = CNNEmbedder(input_shape, num_channels, input_dim, class_dropout_prob)
        self.blocks = nn.ModuleList([DiTBlock(num_channels, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])
        
        self.final = FinalLayer(num_channels, 1, 1)
        self.x_to_t = nn.Linear(input_dim, num_channels)

    def forward(self, c: Tensor) -> Tensor:
        """Map capacity matrices (B, H, W) or (B, 1, H, W) to SOH (B, T)."""
        if len(c.shape) == 3:
            c = c.unsqueeze(1) # (B, C, H, W)
        x = self.y_embedder(c, self.training)    # (N, D)
        h = x.unsqueeze(1).transpose(1, 2) + self.pos_embed # (B, T, C)
        t = self.x_to_t(x)

        for transformer in self.blocks:
            h = transformer(h, t)

        out = self.final(h, t).squeeze(-1)
        return out 


