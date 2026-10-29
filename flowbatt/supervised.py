"""One supervised SOH trajectory optimizer step."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from flowbatt.models.transformer import TrF

def train_step(
    model: TrF,
    batch: Mapping[str, Tensor],
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

    pred = model(c)
    loss = torch.pow(pred - x_1, 2).mean() 

    loss.backward()
    optim.step()
    lr_scheduler.step()
    
    return loss.item()


