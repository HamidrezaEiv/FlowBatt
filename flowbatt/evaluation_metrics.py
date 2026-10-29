"""Shared SOH interpolation, interval metrics, reproducible seeds, and result writers."""
from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from flowbatt.data import DataBundle

Scalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = Union[Scalar, list["JsonValue"], dict[str, "JsonValue"]]
IntervalMetrics: TypeAlias = dict[str, float | np.ndarray]

DATASETS = ("MATR-1", "MATR-2", "HUST", "MIX-20", "MIX-100")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators for one run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eol_percent(dataset_name: str) -> float:
    """Return the benchmark end-of-life threshold in SOH percentage points."""
    return 90.0 if dataset_name == "MIX-20" else 80.0


def inverse_soh(dataset: DataBundle, values: torch.Tensor) -> np.ndarray:
    """Convert transformed (batteries, trajectory) labels to fractional SOH."""
    values = values.detach().cpu()
    if dataset.label_transformation is not None:
        values = dataset.label_transformation.inverse_transform(values)
    return values.float().numpy()


def load_physical_arrays(dataset_name: str) -> dict[str, np.ndarray]:
    """Load capacity matrices and inverse-transformed SOH arrays for both splits."""
    # Public data exports resolve relative to the installed repository root.
    from flowbatt.data import get_dataset

    dataset = get_dataset(dataset_name)
    return {
        "x_train": dataset.train_data.feature.detach().cpu().float().numpy(),
        "x_test": dataset.test_data.feature.detach().cpu().float().numpy(),
        "y_train": inverse_soh(dataset, dataset.train_data.label),
        "y_test": inverse_soh(dataset, dataset.test_data.label),
    }


def upsample_soh_percent(values: np.ndarray, scale_factor: int = 10) -> np.ndarray:
    """Interpolate the last axis to physical cycles and convert fractional SOH to percent."""
    values_t = torch.as_tensor(values, dtype=torch.float32)
    original_shape = values_t.shape
    if values_t.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {original_shape}")
    values_t = values_t.reshape(-1, 1, original_shape[-1])
    upsampled = F.interpolate(
        values_t,
        scale_factor=scale_factor,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return (upsampled.reshape(*original_shape[:-1], -1).numpy() * 100.0)


def first_crossing(values_percent: np.ndarray, threshold: float) -> np.ndarray:
    """Return one-based first threshold crossings, using the horizon when none exists."""
    values = np.asarray(values_percent)
    if values.ndim != 2:
        raise ValueError(f"Expected (batteries, cycles), got {values.shape}")
    below = values <= threshold
    has_crossing = below.any(axis=1)
    first = np.argmax(below, axis=1)
    first = np.where(has_crossing, first, values.shape[1] - 1)
    return first.astype(np.float64) + 1.0


def _mean_battery_rmse_percent(
    reference_percent: np.ndarray, prediction_percent: np.ndarray, threshold: float,
) -> float:
    """Average per-battery RMSEs over the EOL-clipped, padded physical-cycle horizon."""
    reference = np.asarray(reference_percent, dtype=np.float64)
    prediction = np.asarray(prediction_percent, dtype=np.float64)
    if reference.ndim != 2 or prediction.shape != reference.shape or not reference.size:
        raise ValueError("Expected matching nonempty (batteries, cycles) arrays.")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("SOH arrays must be finite.")
    error = np.maximum(prediction, threshold) - np.maximum(reference, threshold)
    return float(np.sqrt(np.mean(error ** 2, axis=1)).mean())


def trajectory_rmse_percent(
    reference: np.ndarray, prediction: np.ndarray, threshold: float,
) -> float:
    """Return the paper's mean per-battery trajectory RMSE in SOH percentage points.

    Inputs are fractional SOH arrays shaped (batteries, trajectory points).
    Interpolate to physical cycles, clip both arrays at the task EOL threshold,
    compute each battery's RMSE over the padded horizon, then average batteries.
    """
    return _mean_battery_rmse_percent(
        upsample_soh_percent(reference), upsample_soh_percent(prediction), threshold,
    )


def reference_valid_mask(reference_percent: np.ndarray, floor: float = 70.0) -> np.ndarray:
    """Include each reference cycle up to and including its first floor crossing."""
    reference = np.asarray(reference_percent)
    mask = np.zeros_like(reference, dtype=bool)
    crossings = first_crossing(reference, floor).astype(int) - 1
    for battery_idx, crossing in enumerate(crossings):
        mask[battery_idx, : crossing + 1] = True
    return mask


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> np.ndarray:
    """Compute weighted empirical quantiles for (observations, targets) values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected values (observations, targets), got {values.shape}")
    if weights.shape != (values.shape[0],):
        raise ValueError(f"Expected weights {(values.shape[0],)}, got {weights.shape}")
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Weights must have a positive finite sum.")
    weights = weights / weight_sum
    output = np.empty((len(quantiles), values.shape[1]), dtype=np.float64)
    for target_idx in range(values.shape[1]):
        order = np.argsort(values[:, target_idx], kind="stable")
        sorted_values = values[order, target_idx]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        cumulative[-1] = 1.0
        for quantile_idx, quantile in enumerate(quantiles):
            location = min(np.searchsorted(cumulative, quantile, side="left"), len(order) - 1)
            output[quantile_idx, target_idx] = sorted_values[location]
    return output.astype(np.float32)


def _interval_metrics(
    reference_percent: np.ndarray,
    lower_percent: np.ndarray,
    center_percent: np.ndarray,
    upper_percent: np.ndarray,
    threshold: float,
) -> IntervalMetrics:
    """Measure clipped full-horizon coverage, width, and point errors in SOH percent."""
    reference = np.asarray(reference_percent, dtype=np.float64)
    lower = np.asarray(lower_percent, dtype=np.float64)
    center = np.asarray(center_percent, dtype=np.float64)
    upper = np.asarray(upper_percent, dtype=np.float64)
    expected = reference.shape
    for name, value in (("lower", lower), ("center", center), ("upper", upper)):
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected}")

    # Prediction-interval protocol: clamp at the task EOL and evaluate the
    # complete padded 2,560-cycle arrays.
    reference_clipped = np.maximum(reference, threshold)
    lower_clipped = np.maximum(lower, threshold)
    upper_clipped = np.maximum(upper, threshold)
    covered = (reference_clipped >= lower_clipped) & (reference_clipped <= upper_clipped)
    width = upper_clipped - lower_clipped

    # Also measure intervals over the unclipped reference trajectory to 70% SOH.
    valid_mask = reference_valid_mask(reference, floor=70.0)
    covered_reference = (reference >= lower) & (reference <= upper)
    width_reference = upper - lower

    rul_ref = first_crossing(reference, threshold)
    rul_center = first_crossing(center, threshold)

    return {
        "coverage": float(covered.mean()),
        "width": float(width.mean()),
        "trajectory_soh_rmse_percent": _mean_battery_rmse_percent(reference, center, threshold),
        "reference_horizon_coverage": float(covered_reference[valid_mask].mean()),
        "reference_horizon_width": float(width_reference[valid_mask].mean()),
        "rul_rmse": float(np.sqrt(np.mean((rul_center - rul_ref) ** 2))),
        "per_battery_coverage": covered.mean(axis=1).astype(np.float32),
        "coverage_by_cycle": covered.mean(axis=0).astype(np.float32),
        "width_by_cycle": width.mean(axis=0).astype(np.float32),
    }


def interval_metrics_from_bounds(
    reference_soh: np.ndarray,
    lower_soh: np.ndarray,
    center_soh: np.ndarray,
    upper_soh: np.ndarray,
    threshold: float,
) -> IntervalMetrics:
    """Interpolate fractional-SOH bounds and evaluate their coverage and width."""
    return _interval_metrics(
        upsample_soh_percent(reference_soh),
        upsample_soh_percent(lower_soh),
        upsample_soh_percent(center_soh),
        upsample_soh_percent(upper_soh),
        threshold,
    )


def interval_metrics_from_samples(
    reference_soh: np.ndarray,
    prediction_samples: np.ndarray,
    threshold: float,
    interval: float = 0.90,
) -> IntervalMetrics:
    """Evaluate fractional-SOH draws shaped (draws, batteries, trajectory)."""
    samples = np.asarray(prediction_samples)
    if samples.ndim != 3:
        raise ValueError(f"Expected (samples, batteries, trajectory), got {samples.shape}")
    alpha = 1.0 - interval
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    center = np.median(samples, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    metrics = interval_metrics_from_bounds(reference_soh, lower, center, upper, threshold)

    samples_percent = upsample_soh_percent(samples)
    rul_samples = np.stack(
        [first_crossing(samples_percent[sample_idx], threshold) for sample_idx in range(len(samples))]
    )
    rul_ref = first_crossing(upsample_soh_percent(reference_soh), threshold)
    rul_lower = np.quantile(rul_samples, alpha / 2.0, axis=0)
    rul_upper = np.quantile(rul_samples, 1.0 - alpha / 2.0, axis=0)
    metrics.update(
        {
            "rul_coverage": float(((rul_ref >= rul_lower) & (rul_ref <= rul_upper)).mean()),
            "rul_width": float(np.mean(rul_upper - rul_lower)),
        }
    )
    return metrics


def serialize_metrics(metrics: IntervalMetrics) -> dict[str, JsonValue]:
    """Convert NumPy metric values into JSON-compatible scalars and lists."""
    serialized: dict[str, JsonValue] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def write_json(path: Path, payload: JsonValue) -> None:
    """Create parent directories and write an indented JSON result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Scalar]]) -> None:
    """Write nonempty result rows using the union of their column names."""
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
