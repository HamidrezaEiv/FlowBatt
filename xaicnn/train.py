"""Train CNN or XAI-CNN on stored RUL labels and save the final epoch."""
from __future__ import annotations

import argparse
import platform
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import (DATASETS, REPOSITORY, WORKSPACES, dataset_name, dataset_path,
                  load_training_data, load_training_transform, validate_output,
                  validate_split, write_json)
from models import Activation, CNN, make_model

OPTIMIZER = {"name": "torch.optim.AdamW", "lr": .001, "betas": [.9, .999],
             "eps": 1e-8, "weight_decay": .01, "amsgrad": False}


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy model parameters to independent CPU tensors for checkpoint saving."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    """Construct AdamW with the fixed CNN training hyperparameters."""
    return torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999),
                             eps=1e-8, weight_decay=.01, amsgrad=False)


def fit_model(
    model: CNN,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    *,
    epochs: int,
    device: str = "cpu",
    log_every: int = 100,
) -> tuple[dict[str, torch.Tensor], list[dict[str, int | float]]]:
    """Fit all training cells and return the final state and training losses.

    ``x`` has shape (N, 1, H, 100); ``y`` contains dimensionless log/Z-score
    targets of shape (N,). There is one full-batch update per epoch, with no
    validation or test monitoring. The command defaults to 1000 epochs.
    """
    validate_split(x, y)
    if epochs < 1 or log_every < 1:
        raise ValueError("Positive epochs and log interval are required")
    loader = DataLoader(TensorDataset(torch.as_tensor(x), torch.as_tensor(y)),
                        batch_size=len(x), shuffle=True, num_workers=0)
    model.to(device)
    optimizer = make_optimizer(model)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (model(features.to(device)) - targets.to(device)).square().mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        value = float(loss.detach().cpu())
        curve.append({"epoch": epoch, "train_mse_before_update": value})
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch={epoch}/{epochs} train_mse={value:.8g}", flush=True)
    return cpu_state(model), curve


def resolve_device(requested: str) -> str:
    """Select an available accelerator, preferring CUDA then MPS for auto."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return requested


def build_parser() -> argparse.ArgumentParser:
    """Create the train-only CNN/XAI-CNN command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "xai-cnn"), default="cnn",
                        help="CNN uses ReLU; XAI-CNN uses LeakyReLU(0.3)")
    parser.add_argument("--dataset", type=dataset_name, choices=DATASETS, default="MATR-1")
    parser.add_argument("--output", type=str,
                        help="New directory under workspaces/; relative paths start at xaicnn/")
    seed_args = parser.add_mutually_exclusive_group()
    seed_args.add_argument("--seed", type=int, help="Train one initialization seed")
    seed_args.add_argument("--seeds", help="Comma-separated seeds (default: 0,1,2,3,4,5,6,7,8,9)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--check-data", action="store_true", help="Validate training data without fitting or writing")
    return parser


def main() -> None:
    """Train the requested seeds and record only final-epoch model checkpoints."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        seed_list = args.seeds if args.seeds is not None else "0,1,2,3,4,5,6,7,8,9"
        seeds = [args.seed] if args.seed is not None else [int(value) for value in seed_list.split(",")]
        if (not seeds or len(seeds) != len(set(seeds))
                or any(seed < 0 or seed >= 2**32 for seed in seeds)):
            raise ValueError("Seeds must be distinct nonnegative 32-bit integers")
        if min(args.epochs, args.threads, args.log_every) < 1:
            raise ValueError("Epochs, threads, and log interval must be positive")
        path = dataset_path(args.dataset)
        x, y = load_training_data(path)
        transform = load_training_transform(path)
        if args.check_data:
            print(f"Verified {args.dataset}: {len(x)} training matrices {x.shape}, "
                  "rul_train_scaled targets and stored RUL normalization; no fitting or outputs.")
            return
        family = args.model.replace("-", "_")
        output = validate_output(args.output or WORKSPACES / f"{family}_{args.dataset}")
        device = resolve_device(args.device)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    activation: Activation = "relu" if args.model == "cnn" else "leaky_relu"
    input_shape = (int(x.shape[2]), int(x.shape[3]))
    protocol: dict[str, Any] = {
        "experiment": "scalar_cnn_16ch", "model": args.model, "dataset": args.dataset,
        "target_protocol": "dataset_rul", "target_keys": ["rul_train_scaled", "rul_test_legacy"],
        "activation": activation, "channels": 16,
        "input_shape": list(x.shape[1:]), "seeds": seeds, "epochs": args.epochs,
        "fit_count": len(x), "fit_indices": list(range(len(x))), "batch_size": len(x),
        "updates_per_epoch": 1, "shuffle": True, "optimizer": OPTIMIZER,
        "learning_rate_schedule": "constant", "loss": "MSE on log/Z-score RUL",
        "initialization": "PyTorch defaults; manual_seed before model construction",
        "checkpoint_policy": "final_epoch_only", "test_evaluation_during_training": False,
        "data_path": str(path.relative_to(REPOSITORY)),
        "label_transform": {"mean": float(transform["label_mean"]),
                            "std": float(transform["label_std"]),
                            "log_base": float(transform["label_log_base"])},
        "device": device, "threads": args.threads,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    write_json(output / "protocol.json", protocol)
    entries = []
    for seed in seeds:
        model = make_model(activation, seed, input_shape)
        run = output / f"{family}_seed_{seed:03d}"
        run.mkdir()
        print(f"Training {args.model} on {args.dataset}: seed={seed}, "
              f"all {len(x)} cells, {args.epochs} epochs", flush=True)
        started = time.perf_counter()
        final, curve = fit_model(model, x, y, epochs=args.epochs, device=device, log_every=args.log_every)
        torch.save(final, run / "final.pt")
        entry = {"seed": seed, "directory": run.name, "checkpoint": f"{run.name}/final.pt",
                 "epochs": args.epochs, "checkpoint_policy": "final_epoch_only",
                 "elapsed_seconds": time.perf_counter() - started}
        write_json(run / "training.json", {**entry, "curve": curve})
        entries.append(entry)
    write_json(output / "training_complete.json", {
        "status": "complete", "experiment": protocol["experiment"], "seeds": seeds,
        "model": args.model, "dataset": args.dataset, "target_protocol": "dataset_rul",
        "epochs": args.epochs, "checkpoint_policy": "final_epoch_only",
        "test_evaluation_during_training": False, "runs": entries,
    })
    print(f"Finished: {len(entries)} final checkpoints in {output}. No test evaluation was run.")


if __name__ == "__main__":
    main()
