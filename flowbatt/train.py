"""Train FlowBatt, DiffBatt, or Transformer using only the training split."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from dataclasses import asdict

from flowbatt.config import load_config

import numpy as np
import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from flowbatt.data import get_dataset
from flowbatt.models.dit import DiT, InitializerNet
from flowbatt.flow_matching import train_step as flow_step
from flowbatt.diffusion import train_step as diffusion_step
from flowbatt.models.transformer import TrF
from flowbatt.supervised import train_step as transformer_step
from flowbatt.paths import WORKSPACES_DIR, DATASETS, dataset_name


def main() -> None:
    """Parse training options, optimize for all epochs, and save the final state only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flowbatt", "diffbatt", "transformer"), default="flowbatt")
    parser.add_argument("--dataset", type=dataset_name, choices=DATASETS, default="MATR-1")
    parser.add_argument("--epochs", type=int, default=None, help="Override the model's published default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--conditioning-dropout", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    overrides = {"seed": args.seed, "class_dropout_prob": args.conditioning_dropout}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    config = load_config(args.model, **overrides)
    settings = asdict(config)
    if config.num_epochs < 1 or not 0 <= args.conditioning_dropout <= 1:
        parser.error("epochs must be positive and conditioning dropout must be in [0,1].")
    output = args.output or (WORKSPACES_DIR / "training" / args.model / args.dataset /
                             f"{args.model}_seed_{args.seed}.pth")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}. Choose a new --output.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dataset = get_dataset(args.dataset)
    loader = torch.utils.data.DataLoader(dataset.train_data,
        batch_size=max(1, len(dataset.train_data) // config.n_batches), shuffle=True)
    model_kwargs = dict(input_dim=config.sequence_length,
        input_shape=(1,) + tuple(dataset.train_data.feature.shape[1:]),
        num_channels=config.channels, num_blocks=config.num_blocks,
        class_dropout_prob=config.class_dropout_prob)
    mask_size = 2 if args.dataset == "MIX-20" else 10
    if args.model == "transformer":
        model = TrF(**model_kwargs).to(device)
        initializer = None
    else:
        model = DiT(**model_kwargs, mask_size=mask_size).to(device)
        initializer = InitializerNet(config.sequence_length, mask_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    schedule = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=config.lr_warmup_steps, num_training_steps=len(loader) * config.num_epochs)
    process = (AffineProbPath(scheduler=CondOTScheduler()) if args.model == "flowbatt"
               else DDPMScheduler(num_train_timesteps=1000))
    history = []
    started = time.perf_counter()
    for epoch in range(config.num_epochs):
        model.train()
        losses = []
        for batch in loader:
            if args.model == "transformer":
                value = transformer_step(model, batch, optimizer, schedule, device)
            else:
                step = flow_step if args.model == "flowbatt" else diffusion_step
                value = step(model, initializer, batch, process, optimizer, schedule, device)
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}.")
            losses.append(value)
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        if epoch == 0 or (epoch + 1) % config.print_every == 0 or epoch + 1 == config.num_epochs:
            print(f"epoch {epoch + 1}/{config.num_epochs} | train loss {mean_loss:.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output)
    metadata = {"model": args.model, "dataset": args.dataset, "device": str(device),
        "config": settings, "batch_size": loader.batch_size, "training_seconds": seconds,
        "loss": history, "test_data_used_for_training_or_selection": False,
        "checkpoint_selection": "final epoch", "torch_version": torch.__version__}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output} ({seconds:.1f} s)")


if __name__ == "__main__":
    main()
