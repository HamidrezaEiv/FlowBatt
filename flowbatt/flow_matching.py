"""One conditional flow-matching optimizer step."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from flowbatt.models.dit import DiT, InitializerNet
from flow_matching.path import AffineProbPath

def train_step(
    model: DiT,
    initializer: InitializerNet,
    batch: Mapping[str, Tensor],
    path: AffineProbPath,
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

    t = torch.rand(x_1.shape[0]).to(device) 

    # Couple Gaussian noise and data through the prescribed interpolation path.
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # Match the interpolant's target velocity rather than its position.
    pred = model(path_sample.x_t, path_sample.t, c)
    loss = torch.pow(pred - path_sample.dx_t, 2).mean() 

    loss.backward()
    optim.step()
    lr_scheduler.step()
    
    return loss.item()


