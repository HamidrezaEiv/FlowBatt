"""Training-only Z-score preprocessing shared by RF and MLP trajectory baselines.

Capacity features retain the native voltage grid. Statistics are fitted across
training batteries independently for each input feature and SOH target position.
Predictions are inverse-transformed to fractional SOH before saving or scoring.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


class BaselineNormalization:
    """Per-feature input and per-position target Z-scores fitted on training cells."""

    preprocessing_id = "native-capacity_feature-zscore_soh-position-zscore_v1"
    input_std_floor = 1e-5
    target_std_floor = 1e-3

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit population statistics across aligned, finite training arrays."""
        x = np.asarray(x_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.float32)
        if x.ndim < 2 or y.ndim != 2 or not len(x) or len(x) != len(y):
            raise ValueError("Expected aligned nonempty inputs and 2-D SOH targets.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training arrays must be finite.")
        self.x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.x_std = np.maximum(x.std(axis=0, keepdims=True), self.input_std_floor).astype(np.float32)
        self.y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
        self.y_std = np.maximum(y.std(axis=0, keepdims=True), self.target_std_floor).astype(np.float32)
        self.n_train = len(x)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        """Standardize capacity features without refitting on new batteries."""
        values = np.asarray(x, dtype=np.float32)
        if values.shape[1:] != self.x_mean.shape[1:]:
            raise ValueError("Capacity feature shape does not match the fitted scaler.")
        return ((values - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Standardize each SOH position using training-cell statistics."""
        values = np.asarray(y, dtype=np.float32)
        if values.shape[-1] != self.y_mean.shape[-1]:
            raise ValueError("Trajectory length does not match the fitted scaler.")
        return ((values - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, prediction: np.ndarray) -> np.ndarray:
        """Restore fractional SOH, including arrays with a quantile dimension."""
        values = np.asarray(prediction, dtype=np.float32)
        if values.shape[-1] != self.y_mean.shape[-1]:
            raise ValueError("Trajectory length does not match the fitted scaler.")
        return (values * self.y_std + self.y_mean).astype(np.float32)

    def metadata(self) -> dict[str, object]:
        """Describe the fitted preprocessing without embedding large arrays in logs."""
        return {
            "preprocessing_id": self.preprocessing_id,
            "input_scaling": "training_only_per_feature_zscore",
            "target_scaling": "training_only_per_position_zscore",
            "input_std_floor": self.input_std_floor,
            "target_std_floor": self.target_std_floor,
            "std_ddof": 0,
            "statistics_axis": "training batteries (axis 0)",
            "input_shape": list(self.x_mean.shape[1:]),
            "trajectory_length": self.y_mean.shape[-1],
            "n_train": self.n_train,
        }

    def state_dict(self) -> dict[str, object]:
        """Save fitted statistics as tensors for safe PyTorch checkpoint loading."""
        return {
            **self.metadata(),
            **{name: torch.from_numpy(getattr(self, name).copy())
               for name in ("x_mean", "x_std", "y_mean", "y_std")},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> BaselineNormalization:
        """Restore this protocol and reject incompatible or malformed statistics."""
        if (state.get("preprocessing_id") != cls.preprocessing_id
                or state.get("input_scaling") != "training_only_per_feature_zscore"
                or state.get("target_scaling") != "training_only_per_position_zscore"
                or state.get("input_std_floor") != cls.input_std_floor
                or state.get("target_std_floor") != cls.target_std_floor
                or state.get("std_ddof") != 0):
            raise ValueError("Incompatible baseline preprocessing metadata.")
        restored = cls.__new__(cls)
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            value = state[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float32).copy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite normalization statistics.")
            setattr(restored, name, array)
        restored.n_train = int(state["n_train"])
        if (restored.x_mean.shape != (1, *state["input_shape"])
                or restored.x_std.shape != restored.x_mean.shape
                or restored.y_mean.shape != (1, state["trajectory_length"])
                or restored.y_std.shape != restored.y_mean.shape
                or np.any(restored.x_std < cls.input_std_floor)
                or np.any(restored.y_std < cls.target_std_floor)
                or restored.n_train < 1):
            raise ValueError("Invalid normalization statistic shapes or deviations.")
        return restored
