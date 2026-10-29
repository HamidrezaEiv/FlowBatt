"""Conditional diffusion Transformer for one-dimensional SOH trajectories.

DiT blocks are adapted from Facebook Research DiT (CC BY-NC 4.0):
https://github.com/facebookresearch/DiT/blob/main/models.py
Copyright (c) Meta Platforms, Inc. and affiliates.
The sinusoidal timestep embedding derives from OpenAI GLIDE (MIT):
https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
Copyright (c) 2021 OpenAI.
Full upstream notices and license texts are in the repository-root LICENSE.
Adaptations include battery conditioning, one-dimensional trajectories, GELU
modulation, observed-prefix masking, and zero-initialized residual gates.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply channel-wise affine modulation to tokens of shape (B, T, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 32) -> None:
        """Map sinusoidal features to the hidden width used by the DiT blocks."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """Embed batch timesteps (B,) as hidden vectors (B, D)."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class CNNEmbedder(nn.Module):
    """
    Embeds early-cycle conditions and randomly drops them for guidance training.
    """
    def __init__(self, input_shape: tuple[int, int, int], hidden_size: int,
                 dropout_prob: float, **block_kwargs: object) -> None:
        """Configure an encoder for capacity matrices (channels, cycles, voltage)."""
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

        self.proj = nn.Conv2d(hidden_size, hidden_size, (H, W))
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.flatten = nn.Flatten()

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

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            x = self.token_drop(x, force_drop_ids)

        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer normalization and residual gates.

    Residual gates start at zero, so a new block initially acts as the identity.
    Other parameters retain their PyTorch initialization.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 **block_kwargs: object) -> None:
        """Configure self-attention, MLP, and six conditioning modulation vectors."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Chunk order: shift, scale, gate for attention, then for the MLP.
        modulation = self.adaLN_modulation[-1]
        for start in (2 * hidden_size, 5 * hidden_size):
            nn.init.zeros_(modulation.weight[start:start + hidden_size])
            nn.init.zeros_(modulation.bias[start:start + hidden_size])

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Update tokens (B, T, D) conditioned on global embeddings (B, D)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        """Create the conditioned projection from hidden tokens to output values."""
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Project tokens (B, T, D) using global conditioning vectors (B, D)."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """Predict a flow vector or diffusion noise for normalized SOH sequences.

    Inputs are trajectories (B, T), timesteps (B,), and capacity matrices
    (B, H, W) or (B, 1, H, W). The observed prefix is masked in the output.
    """
    def __init__(
        self,
        input_dim: int = 256,
        input_shape: tuple[int, int, int] = (1, 100, 100),
        num_channels: int = 16,
        num_blocks: int = 4,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        mask_size: int = 10
    ) -> None:
        """Configure model dimensions and the number of observed SOH points."""
        super().__init__()

        self.t_embedder = TimestepEmbedder(num_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, num_channels), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(num_channels, input_dim)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.y_embedder = CNNEmbedder(input_shape, num_channels, class_dropout_prob)

        self.blocks = nn.ModuleList([DiTBlock(num_channels, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])

        self.final = FinalLayer(num_channels, 1, 1)
        self.first = nn.Conv1d(1, num_channels, kernel_size=1, padding="same")

        self.mask = torch.ones(input_dim).unsqueeze(0)
        self.mask[:, :mask_size] = 0.0  # Keep the supplied observed SOH prefix unchanged.
    
    def forward(self, x: Tensor, t: Tensor, c: Tensor,
                force_drop_ids: Tensor | None = None) -> Tensor:
        """Return a masked prediction with the same (B, T) shape as x."""
        h = x.unsqueeze(1)  # (B, 1, T)
        if len(c.shape) == 3:
            c = c.unsqueeze(1) # (B, C, H, W)
        h = self.first(h).transpose(1, 2) + self.pos_embed # (B, T, C)
        t = self.t_embedder(t)
        c = self.y_embedder(c, self.training, force_drop_ids=force_drop_ids)    # (N, D)
        t = t + c

        for transformer in self.blocks:
            h = transformer(h, t)

        out = self.final(h, t).squeeze(-1)
        return out * self.mask.to(x.device)  # Apply mask to the output

    def forward_with_guidance(self, x: Tensor, t: Tensor, c: Tensor,
                              guidance_scale: float) -> Tensor:
        """Combine conditional and unconditional predictions; scale 1 is conditional."""
        guidance_scale = float(guidance_scale)
        if guidance_scale == 1.0:
            return self.forward(x, t, c)

        batch_size = x.shape[0]
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([c, c], dim=0)
        force_drop_ids = torch.cat(
            [
                torch.zeros(batch_size, device=x.device, dtype=torch.long),
                torch.ones(batch_size, device=x.device, dtype=torch.long),
            ],
            dim=0,
        )

        pred_cond, pred_uncond = self.forward(
            x_in, t_in, c_in, force_drop_ids=force_drop_ids
        ).chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    
class InitializerNet(nn.Module):
    """Insert observed normalized SOH values into a noise trajectory prefix."""
    def __init__(self, input_dim: int = 256, mask_size: int = 10) -> None:
        """Mask the first mask_size entries in trajectories of length input_dim."""
        super().__init__()
        self.mask = torch.ones(input_dim).unsqueeze(0)
        self.mask[:, :mask_size] = 0  
        self.mask_size = mask_size
        
    def forward(self, z: Tensor, x: Tensor, c: Tensor) -> Tensor:
        """Combine noise (B, T) with observed SOH (B, mask_size); c is unused."""
        x = F.pad(x, (0, z.shape[-1] - x.shape[-1]), 'constant', 0.0)  # Pad x to match output length
        return z * self.mask.to(z.device) + x 
    
def get_1d_sincos_pos_embed(embed_dim: int, input_dim: int) -> NDArray[np.float64]:
    """Return sinusoidal positional features with shape (input_dim, embed_dim)."""
    pos = np.arange(input_dim, dtype=np.float32)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
