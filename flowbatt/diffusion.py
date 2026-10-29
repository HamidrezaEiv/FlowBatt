"""One denoising-diffusion optimizer step."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from flowbatt.models.dit import DiT, InitializerNet
from diffusers import DDPMScheduler

def train_step(
    model: DiT,
    initializer: InitializerNet,
    batch: Mapping[str, Tensor],
    noise_scheduler: DDPMScheduler,
    optim: Optimizer,
    lr_scheduler: LRScheduler,
    device: str | torch.device,
) -> float:
    """Optimize mean squared error for one batch and advance the learning rate.

    batch contains normalized label trajectories (B, T) and capacity matrices
    (B, H, W). Returns a scalar loss; gradients and optimizer state are updated.
    """
    optim.zero_grad() 
    x_1 = batch['label'].to(device)
    c = batch['feature'].to(device) 
    z = torch.randn_like(x_1).to(device)

    x_0 = initializer(z, x_1[:, :initializer.mask_size], c)

    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_1.shape[0],), device=device).long()

    # Use the initialized noise with the observed SOH prefix, as in training.
    x_n = noise_scheduler.add_noise(x_1, x_0, t)

    pred = model(x_n, t, c)
    loss = torch.pow(pred - x_0, 2).mean() 

    loss.backward()
    optim.step()
    lr_scheduler.step()
    
    return loss.item()


