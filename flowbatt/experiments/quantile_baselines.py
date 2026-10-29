"""Train native-final-cycle QRF and full-matrix QMLP trajectory baselines."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flowbatt.evaluation_metrics import (
    DATASETS,
    IntervalMetrics,
    Scalar,
    eol_percent,
    interval_metrics_from_bounds,
    load_physical_arrays,
    set_seed,
    write_csv,
    write_json,
)
from flowbatt.experiments.full_matrix_baselines import (
    CapsuleTrajectoryMLP,
    LeafQuantileForest,
    quantile_loss,
)
from flowbatt.experiments.baseline_preprocessing import BaselineNormalization as Normalization
from flowbatt.paths import ROOT, WORKSPACES_DIR

DEFAULT_RESULTS = WORKSPACES_DIR
DEFAULT_CONFIG = ROOT / "configs" / "matched_quantile.json"


def load_matched_configuration(
    dataset_name: str,
    models: list[str] | tuple[str, ...] = ("qmlp",),
    *,
    config: Path = DEFAULT_CONFIG,
    baseline_root: Path | None = None,
) -> dict[str, dict[str, Scalar]]:
    """Load QMLP settings only; QRF never reuses full-matrix RF tuning settings."""
    if "qmlp" not in models:
        return {}
    if baseline_root is None:
        with config.open() as stream:
            settings = json.load(stream)
        if settings.get("schema_version") != 1:
            raise ValueError(f"Unsupported matched-quantile configuration: {config}")
        return {"qmlp": dict(settings["datasets"][dataset_name]["qmlp"])}

    path = baseline_root / dataset_name / "mlp_tuning.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing selected MLP training settings: {path}. Run "
            "python -m flowbatt.experiments.full_matrix_baselines --models mlp "
            "first, or omit --baseline-root to use configs/matched_quantile.json."
        )
    with path.open() as stream:
        tuning = json.load(stream)
    return {"qmlp": {**tuning["best"], "epochs": int(tuning["selected_epoch"])}}


def resolve_device(requested: str) -> torch.device:
    """Choose a supported PyTorch device, rejecting unavailable requested accelerators."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def train_matched_qmlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    normalization: Normalization | None = None,
) -> tuple[CapsuleTrajectoryMLP, Normalization]:
    """Train ordered quantile heads for a fixed epoch count on the full training split."""
    if epochs < 1:
        raise ValueError("QMLP epochs must be positive.")
    set_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)

    normalization = normalization if normalization is not None else Normalization(x_train, y_train)
    train_x = torch.from_numpy(normalization.transform_x(x_train))
    train_y = torch.from_numpy(normalization.transform_y(y_train))
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(64, len(train_x)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = CapsuleTrajectoryMLP(
        n_cycles=x_train.shape[1],
        n_voltage=x_train.shape[2],
        trajectory_length=y_train.shape[1],
        hidden=hidden,
        dropout=dropout,
        quantile=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = quantile_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model, normalization


@torch.no_grad()
def predict_matched_qmlp(
    model: CapsuleTrajectoryMLP,
    normalization: Normalization,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return fractional-SOH predictions shaped (batteries, 3, trajectory)."""
    model.eval()
    test_x = torch.from_numpy(normalization.transform_x(x_test)).to(device)
    prediction = model(test_x).cpu().numpy()
    return np.stack(
        [
            normalization.inverse_y(prediction[:, quantile_idx])
            for quantile_idx in range(3)
        ],
        axis=1,
    )


def metric_row(
    dataset_name: str,
    method: str,
    seed: int,
    metrics: IntervalMetrics,
    runtime: float,
    **extra: Scalar,
) -> dict[str, Scalar]:
    """Build one scalar results row from interval metrics and training settings."""
    return {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "coverage": metrics["coverage"],
        "width": metrics["width"],
        "trajectory_soh_rmse_percent": metrics["trajectory_soh_rmse_percent"],
        "reference_horizon_coverage": metrics["reference_horizon_coverage"],
        "reference_horizon_width": metrics["reference_horizon_width"],
        "rul_rmse": metrics["rul_rmse"],
        "train_eval_seconds": runtime,
        **extra,
    }


def run_dataset(
    args: argparse.Namespace, dataset_name: str, device: torch.device
) -> list[dict[str, Scalar]]:
    """Fit the requested models on one dataset and write test predictions and metrics."""
    arrays = load_physical_arrays(dataset_name)
    x_train = arrays["x_train"]
    x_test = arrays["x_test"]
    y_train = arrays["y_train"]
    y_test = arrays["y_test"]
    if (x_train.ndim != 3 or x_test.ndim != 3
            or x_train.shape[1:] != x_test.shape[1:]
            or x_train.shape[1] < 1 or x_train.shape[-1] != 100):
        raise ValueError("Expected matching native capacity matrices with 100 voltage points.")
    threshold = eol_percent(dataset_name)
    selected = load_matched_configuration(
        dataset_name, args.models, config=args.config, baseline_root=args.baseline_root
    )
    rf_params = {"max_features": 1.0, "min_samples_leaf": 1, "max_depth": None}
    mlp_params = dict(selected.get("qmlp", {}))
    selected_epoch = mlp_params.pop("epochs", None)
    if "qmlp" in args.models and (not isinstance(selected_epoch, int) or selected_epoch < 1):
        raise ValueError("Configured QMLP epochs must be a positive integer.")

    model_protocol = {}
    if "qrf" in args.models:
        qrf_x_train = x_train[:, -1, :]
        qrf_x_test = x_test[:, -1, :]
        qrf_normalization = Normalization(qrf_x_train, y_train)
        qrf_train = qrf_normalization.transform_x(qrf_x_train)
        qrf_test = qrf_normalization.transform_x(qrf_x_test)
        qrf_y_train = qrf_normalization.transform_y(y_train)
        model_protocol["qrf"] = {
            "input": "final observed early-life cycle; native 100-point capacity difference curve",
            "input_shape": list(qrf_x_train.shape[1:]),
            "voltage_resampling": "none",
            "normalization": qrf_normalization.metadata(),
            **rf_params, "n_estimators": args.rf_trees,
            "source": "Fixed final-cycle forest settings; no full-matrix RF tuning settings reused",
            "device": "cpu",
        }
    if "qmlp" in args.models:
        qmlp_normalization = Normalization(x_train, y_train)
        model_protocol["qmlp"] = {
            "input": "complete early-cycle capacity matrix; native 100 voltage points",
            "input_shape": list(x_train.shape[1:]),
            "voltage_resampling": "none",
            "normalization": qmlp_normalization.metadata(),
            **mlp_params, "epochs": selected_epoch, "batch_size": min(64, len(x_train)),
            "source": str(args.baseline_root / dataset_name / "mlp_tuning.json"
                          if args.baseline_root is not None else args.config),
            "settings_policy": "Frozen selected deterministic MLP hyperparameters and epochs; no retuning here",
            "device": str(device),
        }

    dataset_dir = args.output_root / "quantile_baselines" / dataset_name
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite runs in {dataset_dir}.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_dir / "protocol.json",
        {
            "output": "256-point SOH trajectory",
            "prediction_units": "physical fractional SOH before saving/scoring",
            "trajectory_metric": "mean per-battery trajectory RMSE in SOH percentage points",
            "evaluation_interpolation": "256 to 2560 cycles; linear; align_corners=False",
            "evaluation_horizon": "full padded horizon; reference and prediction clipped from below at EOL",
            "eol_percent": threshold,
            "train_cells": len(x_train),
            "test_cells": len(x_test),
            "interval": [0.05, 0.95],
            **model_protocol,
            "quantile_changes": {
                "rf": "leaf-weighted 0.05/0.50/0.95 quantiles",
                "mlp": "three ordered quantile heads trained with pinball loss",
            },
            "seeds": args.seeds,
        },
    )

    rows: list[dict[str, Scalar]] = []
    for seed in range(args.seeds):
        if "qrf" in args.models:
            print(f"{dataset_name}: seed {seed + 1}/{args.seeds} QRF", flush=True)
            started = time.perf_counter()
            qrf = LeafQuantileForest(
                **rf_params,
                n_estimators=args.rf_trees,
                random_state=seed,
                n_jobs=-1,
            ).fit(qrf_train, qrf_y_train)
            quantiles, weights = qrf.predict_quantiles(qrf_test)
            quantiles = qrf_normalization.inverse_y(quantiles)
            lower, center, upper = (
                quantiles[:, 0],
                quantiles[:, 1],
                quantiles[:, 2],
            )
            metrics = interval_metrics_from_bounds(
                y_test, lower, center, upper, threshold
            )
            runtime = time.perf_counter() - started
            np.savez_compressed(
                dataset_dir / f"qrf_seed_{seed}.npz",
                reference=y_test,
                lower=lower,
                center=center,
                upper=upper,
                weights=weights,
            )
            rows.append(
                metric_row(
                    dataset_name,
                    "Quantile RF (native final-cycle input)",
                    seed,
                    metrics,
                    runtime,
                    n_estimators=args.rf_trees,
                    **rf_params,
                )
            )
    
        if "qmlp" in args.models:
            print(f"{dataset_name}: seed {seed + 1}/{args.seeds} QMLP", flush=True)
            started = time.perf_counter()
            model, normalization = train_matched_qmlp(
                x_train,
                y_train,
                **mlp_params,
                epochs=selected_epoch,
                seed=seed,
                device=device,
                normalization=qmlp_normalization,
            )
            prediction = predict_matched_qmlp(
                model, normalization, x_test, device
            )
            lower, center, upper = (
                prediction[:, 0],
                prediction[:, 1],
                prediction[:, 2],
            )
            metrics = interval_metrics_from_bounds(
                y_test, lower, center, upper, threshold
            )
            runtime = time.perf_counter() - started
            torch.save(
                {
                    "state_dict": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.state_dict().items()
                    },
                    "normalization": normalization.state_dict(),
                    "preprocessing_id": normalization.preprocessing_id,
                    "input_shape": list(x_train.shape[1:]),
                    "trajectory_length": y_train.shape[1],
                    "quantile": True,
                    "params": mlp_params,
                    "epochs": selected_epoch,
                },
                dataset_dir / f"qmlp_seed_{seed}.pth",
            )
            np.savez_compressed(
                dataset_dir / f"qmlp_seed_{seed}.npz",
                reference=y_test,
                lower=lower,
                center=center,
                upper=upper,
            )
            rows.append(
                metric_row(
                    dataset_name,
                    "Quantile MLP (full-matrix matched settings)",
                    seed,
                    metrics,
                    runtime,
                    epochs=selected_epoch,
                    **mlp_params,
                )
            )
    
    write_csv(dataset_dir / "metrics_by_seed.csv", rows)
    write_json(dataset_dir / "metrics.json", rows)
    return rows


def summarize(rows: list[dict[str, Scalar]]) -> list[dict[str, Scalar]]:
    """Aggregate per-run scalar metrics into dataset/model means and population deviations."""
    summary = []
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset_name, method in groups:
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset_name and row["method"] == method
        ]
        output: dict[str, Scalar] = {
            "dataset": dataset_name,
            "method": method,
            "n_seeds": len(selected),
        }
        for metric in (
            "trajectory_soh_rmse_percent",
            "coverage",
            "width",
            "reference_horizon_coverage",
            "reference_horizon_width",
            "rul_rmse",
        ):
            values = [row[metric] for row in selected]
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_std"] = float(np.std(values))
        summary.append(output)
    return summary


def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=DATASETS
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--models", nargs="+", choices=("qrf", "qmlp"), default=["qrf", "qmlp"],
                        help="Native final-cycle QRF and full-matrix QMLP (both by default).")
    parser.add_argument("--rf-trees", type=int, default=100,
                        help="QRF trees (default: 100); all other forest settings are fixed.")
    parser.add_argument("--device", default="auto")
    settings = parser.add_mutually_exclusive_group()
    settings.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Frozen QMLP hyperparameters/epochs; defaults to configs/matched_quantile.json. Does not configure QRF.",
    )
    settings.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="QMLP-only override: read <dataset>/mlp_tuning.json from a deterministic baseline run; does not configure QRF.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.rf_trees < 1:
        parser.error("--seeds and --rf-trees must be positive integers.")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device) if "qmlp" in args.models else torch.device("cpu")
    print(f"QRF device: cpu; QMLP device: {device if 'qmlp' in args.models else 'not requested'}", flush=True)
    all_rows: list[dict[str, Scalar]] = []
    for dataset_name in args.datasets:
        all_rows.extend(run_dataset(args, dataset_name, device))
    result_dir = args.output_root / "quantile_baselines"
    write_csv(result_dir / "metrics_by_seed.csv", all_rows)
    write_csv(result_dir / "summary.csv", summarize(all_rows))


if __name__ == "__main__":
    main()
