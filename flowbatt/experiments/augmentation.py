"""RF-only MATR-1 synthesis guidance sweep from the main paper.

Defaults use five synthetic curves per training cell.
See docs/experiments.md for the protocol and evaluation commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import platform
import random
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor

from flowbatt.data import get_dataset
from flowbatt.metrics import synthetic_soh_metrics
from flowbatt.paths import WORKSPACES_DIR, dataset_name as normalize_dataset_name
from flowbatt.plotting import plot_synthetic_soh_by_scale
from flowbatt.postprocess import soh_percent_to_rul, soh_to_percent

if TYPE_CHECKING:
    from flowbatt.config import ModelConfig
    from flowbatt.data import DataBundle
    from flowbatt.models.dit import DiT, InitializerNet

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators for one run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_torch_device(requested_device: str) -> torch.device:
    """Resolve an explicit device or prefer CUDA, then MPS, then CPU."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_float_list(value: str | float | int) -> list[float]:
    """Parse a comma-separated scalar option into floating-point values."""
    if isinstance(value, (float, int)):
        return [float(value)]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_max_features(value: str | None) -> str | int | float | None:
    """Interpret a scikit-learn max_features option as a string, number, or None."""
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"sqrt", "log2"}:
        return value.lower()
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def format_scale(scale: float) -> str:
    """Make a filename-safe label for a classifier-free guidance scale."""
    return str(scale).replace(".", "p")


def parse_cycle_index(value: str | int) -> int | Literal["full"]:
    """Resolve a zero-based cycle index or the last/full input aliases."""
    if isinstance(value, int):
        return value

    value = str(value).strip().lower()
    if value in {"full", "all"}:
        return "full"
    if value == "last":
        return -1

    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--cycle-index must be an integer, 'last', 'full', or 'all'"
        ) from exc


def is_full_q_input(cycle_index: str | int) -> bool:
    """Return whether the RF input should flatten the complete capacity matrix."""
    return cycle_index == "full"


def eol_to_percent(eol_soh: float) -> float:
    """Convert a fractional or percentage SOH threshold into percentage points."""
    return eol_soh * 100.0 if eol_soh <= 1.0 else eol_soh


def select_cycle_features(features: torch.Tensor, cycle_index: str | int) -> np.ndarray:
    """Return (cells, features) RF inputs from (cells, cycles, voltage) tensors."""
    cycle_index = parse_cycle_index(cycle_index)
    features = features.detach().cpu()
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape (N, cycles, Q), got {features.shape}")
    if is_full_q_input(cycle_index):
        return features.reshape(features.shape[0], -1).numpy()
    if not -features.shape[1] <= cycle_index < features.shape[1]:
        raise ValueError(
            f"Cycle index {cycle_index} is outside available cycle range "
            f"[0, {features.shape[1] - 1}]"
        )
    return features[:, cycle_index, :].numpy()


def inverse_rul(dataset: DataBundle, values: torch.Tensor) -> torch.Tensor:
    """Convert transformed scalar RUL values to physical cycle counts."""
    values = values.detach().cpu()
    if dataset.rul_transformation is not None:
        values = dataset.rul_transformation.inverse_transform(values)
    return values.float()


def transform_rul(dataset: DataBundle, values: torch.Tensor) -> torch.Tensor:
    """Map physical scalar RUL cycle counts to the dataset's fitted target space."""
    values = values.detach().cpu()
    if dataset.rul_transformation is not None:
        values = dataset.rul_transformation.transform(values)
    return values.float()


def rul_to_target_space(dataset: DataBundle, real_rul: np.ndarray | torch.Tensor, target_space: str) -> torch.Tensor:
    """Convert physical RUL targets to the selected real or transformed space."""
    real_rul = torch.as_tensor(real_rul, dtype=torch.float32)
    if target_space == "real":
        return real_rul.float()
    if target_space == "transformed":
        return transform_rul(dataset, real_rul)
    raise ValueError(f"Unknown target space: {target_space}")


def get_train_rul_targets(dataset: DataBundle, target_space: str="transformed") -> torch.Tensor:
    """Return scalar RUL labels for training cells in the requested target space."""
    train_rul = inverse_rul(dataset, dataset.train_rul.label)
    return rul_to_target_space(dataset, train_rul, target_space)


def get_transformed_train_rul(dataset: DataBundle) -> torch.Tensor:
    """Return training RUL labels after the dataset's fitted transformation."""
    return get_train_rul_targets(dataset, target_space="transformed")


def get_target_space(args: argparse.Namespace) -> str:
    """Read the target-space option, defaulting to transformed labels."""
    return getattr(args, "target_space", "transformed")


def describe_rul(name: str, values: np.ndarray | torch.Tensor) -> None:
    """Log the mean, deviation, and range of physical RUL cycle counts."""
    values = np.asarray(values, dtype=np.float32)
    logging.info(
        "%s RUL mean %.1f | std %.1f | min %.0f | max %.0f",
        name,
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
    )


def evaluate_rul(dataset: DataBundle, prediction: torch.Tensor, target_space: str="transformed") -> tuple[float, float]:
    """Return test RUL RMSE in cycles and MAPE in percent."""
    target = inverse_rul(dataset, dataset.test_rul.label)
    prediction = prediction.detach().cpu().float()
    if target_space == "transformed":
        prediction = inverse_rul(dataset, prediction)
    elif target_space != "real":
        raise ValueError(f"Unknown target space: {target_space}")

    rul_rmse = torch.sqrt(torch.mean((target - prediction) ** 2)).item()
    rul_mape = torch.mean(torch.abs((target - prediction) / target)).item() * 100.0
    return rul_rmse, rul_mape


def validate_flowbatt_guidance(config: ModelConfig, guidance_scales: Sequence[float]) -> None:
    """Reject nonunit guidance when the checkpoint has no unconditional training."""
    guided_scales = [scale for scale in guidance_scales if scale != 1.0]
    if guided_scales and config.class_dropout_prob <= 0.0:
        raise ValueError(
            "Guidance scales other than 1.0 require a FlowBatt checkpoint "
            "trained with class_dropout_prob > 0."
        )


def load_flowbatt(
    dataset: DataBundle,
    dataset_name: str,
    device: torch.device,
    checkpoint: str | Path | None=None,
    class_dropout_prob: float=0.0,
) -> tuple[DiT, InitializerNet, ModelConfig, str | Path]:
    """Construct and strictly load the FlowBatt checkpoint used for synthesis."""
    from flowbatt.config import load_config
    from flowbatt.models.dit import DiT, InitializerNet

    config = load_config('flowbatt', class_dropout_prob=class_dropout_prob)
    input_shape = (1,) + tuple(dataset.train_data.feature.shape[1:])
    mask_size = 2 if normalize_dataset_name(dataset_name) == "MIX-20" else 10

    flow_dtype = torch.float32
    model = DiT(
        input_dim=config.sequence_length,
        input_shape=input_shape,
        num_blocks=config.num_blocks,
        num_channels=config.channels,
        class_dropout_prob=config.class_dropout_prob,
        mask_size=mask_size,
    ).to(device=device, dtype=flow_dtype)
    initializer = InitializerNet(
        input_dim=config.sequence_length, mask_size=mask_size
    ).to(device=device, dtype=flow_dtype)

    if checkpoint is None:
        checkpoint = config.checkpoint_path(dataset_name)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"FlowBatt checkpoint not found: {checkpoint}. "
            "Train it first with python -m flowbatt.train or pass --flowbatt-checkpoint."
        )

    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model = model.to(device=device, dtype=flow_dtype)
    initializer = initializer.to(device=device, dtype=flow_dtype)
    model.eval()
    initializer.eval()
    return model, initializer, config, checkpoint


@torch.no_grad()
def generate_flowbatt_soh_train(
    dataset: DataBundle,
    source_indices: np.ndarray,
    condition_source: torch.Tensor,
    model: DiT,
    initializer: InitializerNet,
    config: ModelConfig,
    device: torch.device,
    guidance_scale: float,
    batch_size: int,
    num_sample_steps: int,
    ode_method: str,
    ode_step_size: float | None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate transformed SOH from training-cell capacity matrices and observed prefixes."""
    from flow_matching.solver import ODESolver
    from flowbatt.sampling import WrappedFlowBattModel
    from torchdiffeq import odeint

    if ode_method == "dopri5":
        ode_step_size = None

    x_train = dataset.train_data.label
    flow_dtype = torch.float32

    model = model.to(device=device, dtype=flow_dtype)
    initializer = initializer.to(device=device, dtype=flow_dtype)
    wrapped_vf = WrappedFlowBattModel(model, guidance_scale=guidance_scale).to(device=device)
    solver = ODESolver(velocity_model=wrapped_vf)
    time_grid = torch.linspace(
        0,
        1,
        num_sample_steps,
        device=device,
        dtype=flow_dtype,
    )

    generated = []
    for start in range(0, len(source_indices), batch_size):
        end = min(start + batch_size, len(source_indices))
        batch_indices = torch.as_tensor(source_indices[start:end], device=device)
        c_batch = condition_source[start:end].to(device=device, dtype=flow_dtype)
        x_ref = x_train[batch_indices.cpu()].to(device=device, dtype=flow_dtype)

        x_init = torch.randn(
            (len(batch_indices), config.sequence_length),
            dtype=flow_dtype,
            device=device,
        )
        x_init = initializer(x_init, x_ref[:, : initializer.mask_size], c_batch)
        x_init = x_init.to(device=device, dtype=flow_dtype)

        if device.type == "mps":
            def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                """Evaluate the guided vector field for the current training-condition batch."""
                return wrapped_vf(x=x, t=t, model_extras=c_batch)

            ode_options = {"dtype": flow_dtype}
            if ode_step_size is not None:
                ode_options["step_size"] = ode_step_size
            x_pred = odeint(
                ode_func,
                x_init,
                time_grid,
                method=ode_method,
                options=ode_options,
                atol=1e-5,
                rtol=1e-5,
            )[-1]
        else:
            x_pred = solver.sample(
                x_init=x_init,
                step_size=ode_step_size,
                method=ode_method,
                time_grid=time_grid,
                return_intermediates=False,
                model_extras=c_batch,
            )
        if not torch.isfinite(x_pred).all():
            raise RuntimeError("FlowBatt generated non-finite SOH values.")
        generated.append(x_pred.cpu())

    return torch.cat(generated, dim=0), source_indices


def soh_to_rul(dataset: DataBundle, soh_transformed: torch.Tensor, eol_soh: float) -> np.ndarray:
    """Extract one-based EOL cycle counts from transformed SOH trajectories."""
    soh_percent = soh_to_percent(dataset, soh_transformed)
    return soh_percent_to_rul(soh_percent, eol_to_percent(eol_soh))


def gaussian_balanced_indices(candidate_rul: np.ndarray, real_rul: np.ndarray, n_keep: int, n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """Select synthetic labels to approximate a Gaussian training-RUL histogram."""
    if n_keep <= 0 or len(candidate_rul) == 0:
        return np.array([], dtype=np.int64)

    n_keep = min(n_keep, len(candidate_rul))
    real_rul = np.asarray(real_rul, dtype=np.float32)
    candidate_rul = np.asarray(candidate_rul, dtype=np.float32)

    mean = float(real_rul.mean())
    std = max(float(real_rul.std()), 1e-6)
    lower = float(min(real_rul.min(), candidate_rul.min()))
    upper = float(max(real_rul.max(), candidate_rul.max()))
    if lower == upper:
        return rng.choice(len(candidate_rul), size=n_keep, replace=False)

    edges = np.linspace(lower, upper, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = np.exp(-0.5 * ((centers - mean) / std) ** 2)
    probs = probs / probs.sum()

    target_total = len(real_rul) + n_keep
    desired_final = np.rint(probs * target_total).astype(int)
    real_counts, _ = np.histogram(real_rul, bins=edges)
    desired_synthetic = np.maximum(desired_final - real_counts, 0)

    selected = []
    candidate_bins = np.digitize(candidate_rul, edges[1:-1], right=False)
    for bin_id, desired_count in enumerate(desired_synthetic):
        if desired_count <= 0:
            continue
        candidates = np.flatnonzero(candidate_bins == bin_id)
        if len(candidates) == 0:
            continue
        take = min(desired_count, len(candidates), n_keep - len(selected))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
        if len(selected) >= n_keep:
            break

    if len(selected) < n_keep:
        selected_set = set(selected)
        remaining = np.array(
            [i for i in range(len(candidate_rul)) if i not in selected_set],
            dtype=np.int64,
        )
        if len(remaining) > 0:
            weights = np.exp(-0.5 * ((candidate_rul[remaining] - mean) / std) ** 2)
            weights = weights / weights.sum()
            fill = min(n_keep - len(selected), len(remaining))
            selected.extend(
                rng.choice(remaining, size=fill, replace=False, p=weights).tolist()
            )

    return np.asarray(selected[:n_keep], dtype=np.int64)


def sample_source_indices(real_rul: np.ndarray, synthetic_per_sample: int, source_sampling: str, source_rul_power: float, rng: np.random.Generator) -> np.ndarray:
    """Choose training cells uniformly or with lifetime-weighted replacement."""
    real_rul = np.asarray(real_rul, dtype=np.float32)
    total_samples = synthetic_per_sample * len(real_rul)
    if total_samples <= 0:
        return np.array([], dtype=np.int64)

    if source_sampling == "uniform":
        return np.repeat(np.arange(len(real_rul)), synthetic_per_sample)

    weights = np.maximum(real_rul, 1e-6) ** source_rul_power
    weights = weights / weights.sum()
    return rng.choice(
        len(real_rul),
        size=total_samples,
        replace=True,
        p=weights,
    ).astype(np.int64)


def select_conditions(condition_base: torch.Tensor, source_indices: np.ndarray) -> torch.Tensor:
    """Copy selected training capacity matrices without mutating the source tensor."""
    condition_base = condition_base.detach().cpu().numpy()
    return torch.from_numpy(condition_base[source_indices].copy()).float()


def make_noisy_conditions(condition_base: torch.Tensor, source_indices: np.ndarray, noise_std: float, rng: np.random.Generator) -> torch.Tensor:
    """Copy selected matrices and add noise scaled by training-only pointwise deviations."""
    condition_base = condition_base.detach().cpu().numpy()
    condition_syn = condition_base[source_indices].copy()
    condition_syn = add_feature_noise(condition_syn, condition_base, noise_std, rng)
    return torch.from_numpy(condition_syn).float()


def add_feature_noise(x_syn: np.ndarray, x_base: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb synthetic inputs using pointwise standard deviations from real training cells."""
    if noise_std <= 0:
        return x_syn
    feature_std = np.maximum(x_base.std(axis=0, keepdims=True), 1e-6)
    noise = rng.normal(loc=0.0, scale=noise_std, size=x_syn.shape)
    return x_syn + noise * feature_std


def build_augmented_training_data(
    dataset: DataBundle,
    x_train: np.ndarray,
    y_train: np.ndarray,
    real_rul: np.ndarray,
    flowbatt_model: DiT,
    flowbatt_initializer: InitializerNet,
    flowbatt_config: ModelConfig,
    flow_device: torch.device,
    args: argparse.Namespace,
    guidance_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic trajectories and combine their RF inputs, labels, and sample weights."""
    rng = np.random.default_rng(args.seed + int(round(guidance_scale * 1000)))
    source_indices = sample_source_indices(
        real_rul=real_rul,
        synthetic_per_sample=args.synthetic_per_sample,
        source_sampling=args.source_sampling,
        source_rul_power=args.source_rul_power,
        rng=rng,
    )

    logging.info(
        "Generating FlowBatt synthetic SOH | guidance_scale %.3f | %d candidates",
        guidance_scale,
        len(source_indices),
    )
    logging.info(
        "Source sampling %s | RUL power %.2f",
        args.source_sampling,
        args.source_rul_power,
    )
    describe_rul("Synthetic source cells", real_rul[source_indices])
    clean_conditions = select_conditions(
        condition_base=dataset.train_data.feature,
        source_indices=source_indices,
    )
    generated_soh, source_indices = generate_flowbatt_soh_train(
        dataset=dataset,
        source_indices=source_indices,
        condition_source=clean_conditions,
        model=flowbatt_model,
        initializer=flowbatt_initializer,
        config=flowbatt_config,
        device=flow_device,
        guidance_scale=guidance_scale,
        batch_size=args.flow_batch_size,
        num_sample_steps=args.num_sample_steps,
        ode_method=args.ode_method,
        ode_step_size=args.ode_step_size,
    )

    candidate_rul = soh_to_rul(dataset, generated_soh, args.synthetic_eol_soh)
    candidate_soh_percent = soh_to_percent(dataset, generated_soh)
    valid = np.isfinite(candidate_rul) & (candidate_rul > args.min_synthetic_rul)
    candidate_rul = candidate_rul[valid]
    candidate_soh_percent = candidate_soh_percent[valid]
    source_indices = source_indices[valid]

    if len(candidate_rul) == 0:
        raise RuntimeError("No valid synthetic RUL labels were generated.")

    n_keep = args.max_synthetic_samples or len(candidate_rul)
    if args.gaussian_balance:
        selected = gaussian_balanced_indices(
            candidate_rul=candidate_rul,
            real_rul=real_rul,
            n_keep=n_keep,
            n_bins=args.rul_bins,
            rng=rng,
        )
    else:
        n_keep = min(n_keep, len(candidate_rul))
        selected = rng.choice(len(candidate_rul), size=n_keep, replace=False)

    selected_rul = candidate_rul[selected]
    selected_soh_percent = candidate_soh_percent[selected]
    selected_sources = source_indices[selected]
    y_syn = rul_to_target_space(
        dataset,
        selected_rul,
        get_target_space(args),
    ).numpy()
    noisy_rf_conditions = make_noisy_conditions(
        condition_base=dataset.train_data.feature,
        source_indices=selected_sources,
        noise_std=args.synthetic_noise_std,
        rng=rng,
    )
    x_syn = select_cycle_features(noisy_rf_conditions, args.cycle_index)

    sample_weight = np.concatenate(
        [
            np.ones(len(y_train), dtype=np.float32),
            np.full(len(y_syn), args.synthetic_weight, dtype=np.float32),
        ]
    )

    describe_rul("Synthetic selected", selected_rul)
    describe_rul("Augmented train", np.concatenate([real_rul, selected_rul]))
    logging.info(
        "Synthetic input method noisy RF Q only | noise std %.3f | synthetic weight %.3f",
        args.synthetic_noise_std,
        args.synthetic_weight,
    )

    return (
        np.concatenate([x_train, x_syn], axis=0),
        np.concatenate([y_train, y_syn], axis=0),
        sample_weight,
        {
            "synthetic_rul": selected_rul,
            "synthetic_soh_percent": selected_soh_percent,
            "source_indices": selected_sources,
        },
    )


def train_and_evaluate_rf(dataset: DataBundle, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, args: argparse.Namespace, run_name: str, sample_weight: np.ndarray | None=None) -> dict[str, float]:
    """Fit an RF on supplied training arrays and save predictions on the fixed test split."""
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=True,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    model.fit(x_train, y_train, sample_weight=sample_weight)
    prediction = torch.from_numpy(model.predict(x_test)).float()

    rul_rmse, rul_mape = evaluate_rul(
        dataset,
        prediction,
        target_space=get_target_space(args),
    )
    logging.info("%s | RUL RMSE %.0f | RUL MAPE %.0f", run_name, rul_rmse, rul_mape)

    prediction_cycles = (inverse_rul(dataset, prediction)
                         if get_target_space(args) == "transformed" else prediction)
    np.savez_compressed(
        Path(args.results_dir) / f"rf_predictions_{run_name}.npz",
        reference=inverse_rul(dataset, dataset.test_rul.label).numpy(),
        prediction=prediction_cycles.numpy(),
    )

    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(
            args.model_dir,
            f"rf_cycle_{args.cycle_index}_{run_name}_{args.dataset}_seed_{args.seed}.pkl",
        )
        if os.path.exists(model_path):
            raise FileExistsError(f"Refusing to overwrite {model_path}")
        with open(model_path, "wb") as fout:
            pickle.dump(model, fout)
        logging.info("Saved model to %s", model_path)

    return {
        "rul_rmse": rul_rmse,
        "rul_mape": rul_mape,
    }


def write_metrics_csv(path: str | Path, baseline_metrics: dict[str, float], scale_metrics: dict[float, dict[str, float | int]], scales: Sequence[float], baseline_label: str="RF") -> None:
    """Write synthesis-quality and RUL metrics for the baseline and guidance sweep."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    rows = [
        ("FID", "fid"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("RUL_RMSE", "rul_rmse"),
        ("RUL_MAPE", "rul_mape"),
        ("N_synthetic", "n_synthetic"),
    ]
    header = ["metric", baseline_label] + [f"w={scale}" for scale in scales]

    with open(path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for label, key in rows:
            row = [label]
            baseline_value = baseline_metrics.get(key, "")
            row.append(baseline_value)
            for scale in scales:
                value = scale_metrics.get(scale, {}).get(key, "")
                row.append(value)
            writer.writerow(row)


def main(args: argparse.Namespace) -> None:
    if args.synthetic_per_sample < 1 or args.flow_batch_size < 1 or args.n_estimators < 1:
        raise ValueError("Sample counts, batch size, and tree count must be positive.")
    if args.num_sample_steps < 2:
        raise ValueError("At least two ODE output times are required.")
    if not np.isfinite(args.synthetic_noise_std) or args.synthetic_noise_std < 0:
        raise ValueError("Synthetic noise standard deviation must be finite and nonnegative.")
    args.max_features = parse_max_features(args.max_features)
    args.cycle_index = parse_cycle_index(args.cycle_index)
    guidance_scales = parse_float_list(args.guidance_scales) if args.augment_flowbatt else []
    if args.augment_flowbatt and (not guidance_scales or not np.isfinite(guidance_scales).all()
                                 or len(set(guidance_scales)) != len(guidance_scales)):
        raise ValueError("Guidance scales must be nonempty, finite and unique.")
    output = Path(args.results_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Results already exist in {output}. Choose a new --results-dir.")
    metrics_csv = args.metrics_csv or str(output / f"rf_flowbatt_metrics_{args.dataset}.csv")
    synthetic_figure = args.synthetic_figure or str(output / f"synthetic_soh_{args.dataset}.pdf")
    run_names = ["real"] + [f"w_{format_scale(scale)}" for scale in guidance_scales]
    planned = [output / "protocol.json", Path(metrics_csv)]
    planned.extend(output / f"rf_predictions_{name}.npz" for name in run_names)
    if args.augment_flowbatt:
        planned.append(Path(synthetic_figure))
        planned.extend(output / f"synthetic_{name}.npz" for name in run_names[1:])
    if args.save_model:
        planned.extend(Path(args.model_dir) / f"rf_cycle_{args.cycle_index}_{name}_{args.dataset}_seed_{args.seed}.pkl"
                       for name in run_names)
    planned = [path.resolve() for path in planned]
    if len(set(planned)) != len(planned):
        raise ValueError("Output paths must be distinct, including protocol, predictions and metrics.")
    for path in planned:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    set_seed(args.seed)

    logging.info("Using sklearn RandomForestRegressor on CPU")
    logging.info("Loading dataset: %s", args.dataset)
    dataset = get_dataset(args.dataset)

    target_space = get_target_space(args)
    x_train = select_cycle_features(dataset.train_rul.feature, args.cycle_index)
    y_train = get_train_rul_targets(dataset, target_space=target_space).numpy()
    x_test = select_cycle_features(dataset.test_rul.feature, args.cycle_index)
    real_rul = inverse_rul(dataset, dataset.train_rul.label).numpy()

    if is_full_q_input(args.cycle_index):
        logging.info("Using full Q sequence as RF input")
    else:
        logging.info("Using cycle index %d as RF input", args.cycle_index)
    logging.info("Training RF on %s RUL labels", target_space)
    logging.info(
        "Train features %s | Test features %s | Train RUL targets %s",
        x_train.shape,
        x_test.shape,
        y_train.shape,
    )
    describe_rul("Real train", real_rul)
    real_soh_percent = soh_to_percent(dataset, dataset.train_data.label)

    os.makedirs(args.results_dir, exist_ok=True)
    protocol = {
        "status": "started",
        "arguments": vars(args).copy(),
        "train_cells": len(x_train),
        "test_cells": len(x_test),
        "candidate_count_per_guidance": len(x_train) * args.synthetic_per_sample,
        "rf_bootstrap": True,
        "real_targets": "cached scalar RUL labels, inverse-transformed to cycles",
        "synthetic_targets": "first SOH <= threshold after 10x linear interpolation; no crossing uses horizon",
        "conditioning": "clean training capacity matrix and observed training SOH prefix",
        "input_augmentation": "Gaussian noise added only to synthetic RF inputs after generation",
        "split_usage": "training cells only for generation, noise statistics and PCA; fixed test split for scoring",
        "provenance": "new RF synthesis run with explicit sample count and seed",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "scikit_learn": sklearn.__version__},
    }
    def save_protocol() -> None:
        """Persist the current experiment status and declared settings."""
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    save_protocol()
    if not args.augment_flowbatt:
        baseline_metrics = train_and_evaluate_rf(
            dataset,
            x_train,
            y_train,
            x_test,
            args,
            run_name="real",
        )
        write_metrics_csv(metrics_csv, baseline_metrics, {}, [])
        logging.info("Saved metrics CSV to %s", metrics_csv)
        protocol["status"] = "complete"
        save_protocol()
        return

    flow_device = get_torch_device(args.flow_device)
    flowbatt_model, flowbatt_initializer, flowbatt_config, checkpoint = load_flowbatt(
        dataset=dataset,
        dataset_name=args.dataset,
        device=flow_device,
        checkpoint=args.flowbatt_checkpoint,
        class_dropout_prob=args.flow_class_dropout_prob,
    )
    validate_flowbatt_guidance(flowbatt_config, guidance_scales)
    protocol["flow_device"] = str(flow_device)
    protocol["checkpoint"] = str(checkpoint)
    protocol["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    protocol["selected_synthetic_counts"] = {}
    save_protocol()
    logging.info("Loaded FlowBatt from %s on %s", checkpoint, flow_device)
    logging.info(
        "FlowBatt class_dropout_prob %.3f | guidance scales %s",
        flowbatt_config.class_dropout_prob,
        ", ".join(str(scale) for scale in guidance_scales),
    )

    baseline_metrics = train_and_evaluate_rf(
        dataset,
        x_train,
        y_train,
        x_test,
        args,
        run_name="real",
    )
    scale_metrics = {}
    synthetic_soh_by_scale = {}

    for guidance_scale in guidance_scales:
        x_aug, y_aug, sample_weight, synthetic_info = build_augmented_training_data(
            dataset=dataset,
            x_train=x_train,
            y_train=y_train,
            real_rul=real_rul,
            flowbatt_model=flowbatt_model,
            flowbatt_initializer=flowbatt_initializer,
            flowbatt_config=flowbatt_config,
            flow_device=flow_device,
            args=args,
            guidance_scale=guidance_scale,
        )
        logging.info(
            "Augmented train features %s | targets %s | weights %s",
            x_aug.shape,
            y_aug.shape,
            sample_weight.shape,
        )
        run_name = f"w_{format_scale(guidance_scale)}"
        rf_metrics = train_and_evaluate_rf(
            dataset,
            x_aug,
            y_aug,
            x_test,
            args,
            run_name=run_name,
            sample_weight=sample_weight,
        )
        syn_soh = synthetic_info["synthetic_soh_percent"]
        syn_metrics = synthetic_soh_metrics(
            real_soh_percent / 100.0,
            syn_soh / 100.0,
            n_components=args.metrics_pca_components,
            n_neighbors=args.metrics_neighbors,
        )
        scale_metrics[guidance_scale] = {
            **syn_metrics,
            **rf_metrics,
            "n_synthetic": len(syn_soh),
        }
        synthetic_soh_by_scale[guidance_scale] = syn_soh
        np.savez_compressed(
            output / f"synthetic_w_{format_scale(guidance_scale)}.npz",
            soh_percent=syn_soh,
            rul_cycles=synthetic_info["synthetic_rul"],
            source_train_indices=synthetic_info["source_indices"],
            rf_features=x_aug[len(x_train):],
        )
        protocol["selected_synthetic_counts"][str(guidance_scale)] = len(syn_soh)
        save_protocol()

    write_metrics_csv(metrics_csv, baseline_metrics, scale_metrics, guidance_scales)
    logging.info("Saved metrics CSV to %s", metrics_csv)

    cycles = np.arange(real_soh_percent.shape[1]) + 1
    Path(synthetic_figure).parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_synthetic_soh_by_scale(
        samples_by_scale=synthetic_soh_by_scale,
        scales=guidance_scales,
        cycles=cycles,
        save_path=synthetic_figure,
        max_curves=args.max_plot_curves,
    )
    logging.info("Saved synthetic SOH figure to %s", synthetic_figure)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    protocol["status"] = "complete"
    save_protocol()


def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest RUL predictor on the FlowBatt dataset split"
    )
    parser.add_argument(
        "--dataset",
        type=normalize_dataset_name,
        default="MATR-1",
        choices=["MATR-1"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--target-space",
        type=str,
        default="real",
        choices=["real", "transformed"],
        help="Train RF on real RUL cycle labels or transformed RUL labels.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="squared_error",
        choices=["squared_error", "absolute_error", "friedman_mse", "poisson"],
        help="RandomForestRegressor split criterion.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--cycle-index",
        type=parse_cycle_index,
        default='full',
        help="Zero-based cycle index, 'last', or 'full'/'all' to flatten the full Q matrix.",
    )
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-dir", type=str, default=str(WORKSPACES_DIR / "trained_models_RF"))
    parser.add_argument("--results-dir", type=str, default=str(WORKSPACES_DIR / "rf_synthesis"))
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--synthetic-figure", type=str, default=None)
    parser.add_argument("--metrics-pca-components", type=float, default=0.99)
    parser.add_argument("--metrics-neighbors", type=int, default=3)
    parser.add_argument("--max-plot-curves", type=int, default=1000)
    parser.add_argument("--augment-flowbatt", action="store_true")
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="0.5,1.0,2.0,3.0,4.0",
        help="Comma-separated FlowBatt guidance scales. One RF experiment is run per scale.",
    )
    parser.add_argument("--flowbatt-checkpoint", type=str, default=None)
    parser.add_argument(
        "--flow-class-dropout-prob",
        type=float,
        default=0.2,
        help="FlowBatt condition dropout probability used by the augmentation checkpoint.",
    )
    parser.add_argument(
        "--flow-device",
        type=str,
        default="auto",
        help="Device for FlowBatt sampling: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--synthetic-per-sample", type=int, default=5,
                        help="Synthetic curves per training cell (default: 5).")
    parser.add_argument(
        "--source-sampling",
        type=str,
        default="uniform",
        choices=["high-rul", "uniform"],
        help="How to choose training conditions for FlowBatt synthetic generation.",
    )
    parser.add_argument(
        "--source-rul-power",
        type=float,
        default=1.0,
        help="Exponent for high-RUL source sampling; larger values oversample long-RUL cells more.",
    )
    parser.add_argument("--max-synthetic-samples", type=int, default=None)
    parser.add_argument(
        "--synthetic-noise-std",
        type=float,
        default=0.1,
        help="Gaussian noise multiplier applied to synthetic Q inputs before RF training.",
    )
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-eol-soh",
        type=float,
        default=0.8,
        help="EOL SOH used to convert generated SOH curves into RUL labels.",
    )
    parser.add_argument("--min-synthetic-rul", type=float, default=100.0)
    parser.add_argument("--gaussian-balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rul-bins", type=int, default=100)
    parser.add_argument("--flow-batch-size", type=int, default=128)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    parser.add_argument("--ode-method", type=str, default="dopri5")
    parser.add_argument("--ode-step-size", type=float, default=0.001)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
