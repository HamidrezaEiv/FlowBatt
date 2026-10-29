"""Train RF and separable MLP trajectory baselines on complete capacity matrices."""
from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from flowbatt.evaluation_metrics import (
    DATASETS,
    Scalar,
    eol_percent,
    load_physical_arrays,
    set_seed,
    trajectory_rmse_percent,
    weighted_quantile,
    write_csv,
    write_json,
)
from flowbatt.experiments.baseline_preprocessing import BaselineNormalization as Normalization
from flowbatt.paths import WORKSPACES_DIR

DEFAULT_RESULTS = WORKSPACES_DIR


class LeafQuantileForest:
    """Quantile Regression Forest using training-observation leaf weights."""

    def __init__(self, **forest_kwargs: Scalar) -> None:
        """Construct the scikit-learn forest with the supplied scalar hyperparameters."""
        self.forest = RandomForestRegressor(**forest_kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LeafQuantileForest":
        """Fit on (batteries, features) inputs and (batteries, trajectory) targets."""
        self.x_train = np.asarray(x, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.float32)
        self.forest.fit(self.x_train, self.y_train)
        self.train_leaves = self.forest.apply(self.x_train)
        return self

    def predict_weights(self, x: np.ndarray) -> np.ndarray:
        """Return float32 training-observation weights for each test battery."""
        x = np.asarray(x, dtype=np.float32)
        test_leaves = self.forest.apply(x)
        weights = np.zeros((len(x), len(self.x_train)), dtype=np.float64)
        n_trees = self.train_leaves.shape[1]
        for tree_idx in range(n_trees):
            train_leaf = self.train_leaves[:, tree_idx]
            for test_idx, leaf in enumerate(test_leaves[:, tree_idx]):
                members = np.flatnonzero(train_leaf == leaf)
                weights[test_idx, members] += 1.0 / (n_trees * len(members))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return weights.astype(np.float32)

    def predict_quantiles(
        self,
        x: np.ndarray,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return leaf-weighted quantiles ordered by test battery and trajectory point."""
        weights = self.predict_weights(x)
        predictions = np.stack(
            [weighted_quantile(self.y_train, weight, quantiles) for weight in weights],
            axis=0,
        )
        # (test, quantile, trajectory)
        return predictions, weights


def rf_grid() -> list[dict[str, Scalar]]:
    """Return the training-only random-forest hyperparameter search space."""
    return [
        {
            "max_features": max_features,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
        }
        for max_features, min_samples_leaf, max_depth in itertools.product(
            ("sqrt", 0.3), (1, 2), (None, 20)
        )
    ]


def tune_rf(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, Scalar], list[dict[str, Scalar]]]:
    """Select forest settings by training-only three-fold mean per-battery SOH RMSE."""
    folds = KFold(n_splits=3, shuffle=True, random_state=seed)
    rows: list[dict[str, Scalar]] = []
    best_score = float("inf")
    best_params: dict[str, Scalar] | None = None
    for params in rf_grid():
        scores = []
        for train_idx, valid_idx in folds.split(x):
            normalization = Normalization(x[train_idx], y[train_idx])
            train_x = normalization.transform_x(x[train_idx]).reshape(len(train_idx), -1)
            valid_x = normalization.transform_x(x[valid_idx]).reshape(len(valid_idx), -1)
            train_y = normalization.transform_y(y[train_idx])
            kwargs = {
                **params,
                "n_estimators": n_estimators,
                "random_state": seed,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**kwargs)
            model.fit(train_x, train_y)
            prediction = normalization.inverse_y(model.predict(valid_x))
            score = trajectory_rmse_percent(y[valid_idx], prediction, threshold)
            scores.append(score)
        mean_score = float(np.mean(scores))
        row = {**params, "trajectory_soh_rmse_percent": mean_score}
        rows.append(row)
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    assert best_params is not None
    return best_params, rows


class CapsuleTrajectoryMLP(nn.Module):
    """Separable voltage/cycle MLP with trajectory or ordered-quantile output heads."""

    def __init__(
        self,
        n_cycles: int,
        n_voltage: int,
        trajectory_length: int,
        hidden: int,
        dropout: float,
        quantile: bool,
    ) -> None:
        """Create separable voltage/cycle projections and the requested output heads."""
        super().__init__()
        self.quantile = quantile
        self.voltage_projection = nn.Sequential(
            nn.Linear(n_voltage, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        self.cycle_projection = nn.Sequential(
            nn.Linear(n_cycles, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.center_head = nn.Linear(hidden, trajectory_length)
        if quantile:
            self.lower_gap_head = nn.Linear(hidden, trajectory_length)
            self.upper_gap_head = nn.Linear(hidden, trajectory_length)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Map (batch, cycles, voltage) inputs to trajectories or (batch, 3, points) quantiles."""
        cycle_values = self.voltage_projection(feature).squeeze(-1)
        embedding = self.cycle_projection(cycle_values)
        center = self.center_head(embedding)
        if not self.quantile:
            return center
        lower = center - F.softplus(self.lower_gap_head(embedding))
        upper = center + F.softplus(self.upper_gap_head(embedding))
        return torch.stack((lower, center, upper), dim=1)


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum mean pinball losses for ordered 5th, 50th, and 95th percentile heads."""
    losses = []
    for index, quantile in enumerate((0.05, 0.5, 0.95)):
        error = target - prediction[:, index]
        losses.append(torch.maximum(quantile * error, (quantile - 1.0) * error).mean())
    return sum(losses)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    *,
    hidden: int,
    dropout: float,
    lr: float,
    seed: int,
    threshold: float,
    max_epochs: int,
    patience: int,
    fixed_epochs: int | None = None,
) -> tuple[CapsuleTrajectoryMLP, Normalization, int, float]:
    """Fit with standardized MSE; select epochs using physical validation SOH RMSE."""
    if fixed_epochs is None and (x_valid is None or y_valid is None):
        raise ValueError("Validation arrays are required unless fixed_epochs is provided.")
    set_seed(seed)
    norm = Normalization(x_train, y_train)
    train_x = torch.from_numpy(norm.transform_x(x_train))
    train_y = torch.from_numpy(norm.transform_y(y_train))
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(64, len(train_x)),
        shuffle=True,
        generator=loader_generator,
    )
    model = CapsuleTrajectoryMLP(
        n_cycles=x_train.shape[1],
        n_voltage=x_train.shape[2],
        trajectory_length=y_train.shape[1],
        hidden=hidden,
        dropout=dropout,
        quantile=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    epochs = fixed_epochs or max_epochs

    valid_x_t = None if x_valid is None else torch.from_numpy(norm.transform_x(x_valid))

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            loss.backward()
            optimizer.step()

        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            prediction = norm.inverse_y(model(valid_x_t).cpu().numpy())
            valid_loss = trajectory_rmse_percent(y_valid, prediction, threshold)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= 50 and stale >= patience:
            break

    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
        best_loss = float("nan")
    return model, norm, best_epoch, best_loss


@torch.no_grad()
def predict_mlp(
    model: CapsuleTrajectoryMLP,
    norm: Normalization,
    x: np.ndarray,
) -> np.ndarray:
    """Return inverse-normalized SOH predictions with one trajectory per battery."""
    model.eval()
    prediction = model(torch.from_numpy(norm.transform_x(x))).cpu().numpy()
    if model.quantile:
        return np.stack(
            [norm.inverse_y(prediction[:, quantile_idx]) for quantile_idx in range(3)],
            axis=1,
        )
    return norm.inverse_y(prediction)


def tune_mlp(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Scalar], int, list[dict[str, Scalar]]]:
    """Select MLP settings and epoch count using an internal training/validation split."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    n_valid = max(4, int(round(0.2 * len(x))))
    valid_idx = indices[:n_valid]
    train_idx = indices[n_valid:]
    candidates = [
        {"hidden": hidden, "dropout": 0.05, "lr": lr}
        for hidden, lr in itertools.product((32, 64), (1e-3, 3e-4))
    ]
    rows = []
    best: tuple[float, dict[str, Scalar], int] | None = None
    for candidate_idx, candidate in enumerate(candidates):
        _, _, best_epoch, best_loss = train_mlp(
            x[train_idx],
            y[train_idx],
            x[valid_idx],
            y[valid_idx],
            **candidate,
            seed=seed + candidate_idx,
            threshold=threshold,
            max_epochs=max_epochs,
            patience=patience,
        )
        rows.append({
            **candidate,
            "best_epoch": best_epoch,
            "validation_trajectory_soh_rmse_percent": best_loss,
        })
        if best is None or best_loss < best[0]:
            best = (best_loss, candidate, best_epoch)
    assert best is not None
    return best[1], best[2], rows


def run_dataset(args: argparse.Namespace, dataset_name: str) -> list[dict[str, Scalar]]:
    """Fit the requested models on one dataset and write test predictions and metrics."""
    arrays = load_physical_arrays(dataset_name)
    x_train = arrays["x_train"]
    x_test = arrays["x_test"]
    y_train = arrays["y_train"]
    y_test = arrays["y_test"]
    if (x_train.ndim != 3 or x_test.ndim != 3
            or x_train.shape[-1] != 100 or x_test.shape[1:] != x_train.shape[1:]):
        raise ValueError("Expected matching capacity matrices shaped (batteries, cycles, 100).")
    normalization = Normalization(x_train, y_train)
    x_train_flat = normalization.transform_x(x_train).reshape(len(x_train), -1)
    x_test_flat = normalization.transform_x(x_test).reshape(len(x_test), -1)
    scaled_y_train = normalization.transform_y(y_train)
    threshold = eol_percent(dataset_name)
    dataset_dir = args.output_root / "baselines" / dataset_name
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory to preserve existing runs: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_dir / "protocol.json",
        {
            **normalization.metadata(),
            "dataset": dataset_name,
            "input": "complete cached early-cycle capacity matrix; no voltage resampling",
            "input_shape": list(x_train.shape[1:]),
            "train_cells": len(x_train),
            "test_cells": len(x_test),
            "trajectory_soh_rmse_percent": "mean per-battery RMSE after interpolation and EOL clipping",
            "evaluation_interpolation": "256 to 2560 cycles; linear; align_corners=False",
            "evaluation_horizon": "full padded horizon; reference and prediction clipped from below at EOL",
            "eol_percent": threshold,
            "prediction_units": "physical fractional SOH",
            "selection": "training-only validation; no test-based tuning or checkpoint selection",
            "seeds": args.seeds,
            "rf_trees": args.rf_trees,
        },
    )
    rows: list[dict[str, Scalar]] = []

    if "rf" in args.models:
        rf_params, rf_cv = tune_rf(
            x_train,
            y_train,
            threshold,
            seed=0,
            n_estimators=args.rf_cv_trees,
        )
        write_json(
            dataset_dir / "rf_tuning.json",
            {"rf_best": rf_params, "rows": rf_cv},
        )

        for seed in range(args.seeds):
            started = time.perf_counter()
            rf = RandomForestRegressor(
                **rf_params,
                n_estimators=args.rf_trees,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(x_train_flat, scaled_y_train)
            prediction = normalization.inverse_y(rf.predict(x_test_flat))
            runtime = time.perf_counter() - started
            np.savez_compressed(
                dataset_dir / f"rf_seed_{seed}.npz",
                reference=y_test,
                prediction=prediction,
            )
            rows.append({
                "dataset": dataset_name,
                "method": "RF",
                "seed": seed,
                "trajectory_soh_rmse_percent": trajectory_rmse_percent(y_test, prediction, threshold),
                "train_eval_seconds": runtime,
            })

    if "mlp" in args.models:
        params, selected_epoch, tuning_rows = tune_mlp(
            x_train,
            y_train,
            threshold=threshold,
            seed=0,
            max_epochs=args.mlp_max_epochs,
            patience=args.mlp_patience,
        )
        write_json(
            dataset_dir / "mlp_tuning.json",
            {
                "best": params,
                "selected_epoch": selected_epoch,
                "rows": tuning_rows,
            },
        )
        for seed in range(args.seeds):
            started = time.perf_counter()
            model, norm, _, _ = train_mlp(
                x_train,
                y_train,
                None,
                None,
                **params,
                seed=seed,
                threshold=threshold,
                max_epochs=selected_epoch,
                patience=args.mlp_patience,
                fixed_epochs=selected_epoch,
            )
            prediction = predict_mlp(model, norm, x_test)
            runtime = time.perf_counter() - started
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalization": norm.state_dict(),
                    "preprocessing_id": norm.preprocessing_id,
                    "input_shape": list(x_train.shape[1:]),
                    "trajectory_length": int(y_train.shape[1]),
                    "quantile": False,
                    "params": params,
                    "epochs": selected_epoch,
                },
                dataset_dir / f"mlp_seed_{seed}.pth",
            )
            np.savez_compressed(
                dataset_dir / f"mlp_seed_{seed}.npz",
                reference=y_test,
                prediction=prediction,
            )
            rows.append({
                "dataset": dataset_name,
                "method": "MLP",
                "seed": seed,
                "trajectory_soh_rmse_percent": trajectory_rmse_percent(y_test, prediction, threshold),
                "train_eval_seconds": runtime,
                "epochs": selected_epoch,
            })

    write_csv(dataset_dir / "metrics_by_seed.csv", rows)
    write_json(dataset_dir / "metrics.json", rows)
    return rows


def summarize(rows: list[dict[str, Scalar]]) -> list[dict[str, Scalar]]:
    """Aggregate per-run scalar metrics into dataset/model means and population deviations."""
    summary = []
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset_name, method in groups:
        selected = [
            row for row in rows if row["dataset"] == dataset_name and row["method"] == method
        ]
        output: dict[str, Scalar] = {
            "dataset": dataset_name,
            "method": method,
            "n_seeds": len(selected),
        }
        values = [row["trajectory_soh_rmse_percent"] for row in selected]
        output["trajectory_soh_rmse_percent_mean"] = float(np.mean(values))
        output["trajectory_soh_rmse_percent_std"] = float(np.std(values))
        summary.append(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=["rf", "mlp"],
                        choices=("rf", "mlp"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-cv-trees", type=int, default=150)
    parser.add_argument("--mlp-max-epochs", type=int, default=800)
    parser.add_argument("--mlp-patience", type=int, default=80)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    for name in ("seeds", "rf_trees", "rf_cv_trees", "mlp_max_epochs", "mlp_patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Scalar]] = []
    for dataset_name in args.datasets:
        print(f"Running baselines for {dataset_name}", flush=True)
        all_rows.extend(run_dataset(args, dataset_name))
    write_csv(args.output_root / "baselines" / "metrics_by_seed.csv", all_rows)
    write_csv(args.output_root / "baselines" / "summary.csv", summarize(all_rows))
    write_json(args.output_root / "baselines" / "metrics.json", all_rows)
    write_json(args.output_root / "baselines" / "summary.json", summarize(all_rows))


if __name__ == "__main__":
    main()
