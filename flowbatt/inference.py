"""Construct architectures and strictly load pretrained checkpoint state dictionaries."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from diffusers import DDPMScheduler

from flowbatt.config import ModelConfig, load_config
from flowbatt.data import DataBundle
from flowbatt.models.dit import DiT, InitializerNet
from flowbatt.models.transformer import TrF
from flowbatt.paths import checkpoint_path
from flowbatt.sampling import GeneratedTrajectories

def validate_flowbatt_guidance(config: ModelConfig) -> None:
    """Reject guidance when a checkpoint was trained without condition dropout."""
    if config.guidance_scale != 1.0 and config.class_dropout_prob <= 0.0:
        raise ValueError(
            "guidance_scale can differ from 1.0 only for FlowBatt checkpoints "
            "trained with class_dropout_prob > 0."
        )


def load_flowbatt(
    input_shape: tuple[int, int, int],
    mask_size: int,
    device: str | torch.device,
    battery_dataset: str,
    class_dropout_prob: float | None = None,
    guidance_scale: float | None = None,
    checkpoint: str | Path | None = None,
) -> tuple[DiT, InitializerNet, Callable[..., GeneratedTrajectories], ModelConfig]:
    """Load FlowBatt weights strictly and return the model, initializer, sampler, config."""
    from flowbatt.sampling import generate_flowbatt

    config_kwargs = {}
    if class_dropout_prob is not None:
        config_kwargs["class_dropout_prob"] = class_dropout_prob
    if guidance_scale is not None:
        config_kwargs["guidance_scale"] = guidance_scale
    config = load_config("flowbatt", **config_kwargs)
    validate_flowbatt_guidance(config)

    model = DiT(
        input_dim=config.sequence_length,
        input_shape=input_shape,
        num_blocks=config.num_blocks,
        num_channels=config.channels,
        class_dropout_prob=config.class_dropout_prob,
        mask_size=mask_size,
    ).to(device)

    initializer = InitializerNet(
        input_dim=config.sequence_length, mask_size=mask_size
    ).to(device)

    ckpt = config.checkpoint_path(battery_dataset) if checkpoint is None else checkpoint
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()

    return model, initializer, generate_flowbatt, config


def load_diffbatt(
    input_shape: tuple[int, int, int], mask_size: int, device: str | torch.device,
    battery_dataset: str,
    checkpoint: str | Path | None = None,
) -> tuple[DiT, InitializerNet, Callable[..., GeneratedTrajectories], ModelConfig, DDPMScheduler]:
    """Strictly load bundled or supplied DiffBatt weights and a 1,000-step scheduler."""
    
    from flowbatt.sampling import generate_diffbatt

    config = load_config("diffbatt")
    model = DiT(
        input_dim=config.sequence_length,
        input_shape=input_shape,
        num_blocks=config.num_blocks,
        num_channels=config.channels,
        class_dropout_prob=config.class_dropout_prob,
        mask_size=mask_size,
    ).to(device)

    initializer = InitializerNet(
        input_dim=config.sequence_length, mask_size=mask_size
    ).to(device)

    ckpt = checkpoint_path("diffbatt", battery_dataset) if checkpoint is None else checkpoint
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return model, initializer, generate_diffbatt, config, scheduler


def load_trf(
    input_shape: tuple[int, int, int], device: str | torch.device, battery_dataset: str,
) -> tuple[TrF, Callable[[DataBundle, TrF, str | torch.device], tuple[Tensor, Tensor]], ModelConfig]:
    """Build Transformer and return its sampler/config; the caller loads each seed's weights."""
    from flowbatt.sampling import generate_transformer

    config = load_config("transformer")
    model = TrF(
        input_dim=config.sequence_length,
        input_shape=input_shape,
        num_blocks=config.num_blocks,
        num_channels=config.channels,
        class_dropout_prob=config.class_dropout_prob,
    ).to(device)

    return model, generate_transformer, config


