"""Load public battery datasets with the benchmark preprocessing transforms.

Transform arithmetic is adapted from BatLiNet (MIT), copyright (c) 2024 Xiaofan.
See the repository-root LICENSE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from flowbatt.paths import DATASETS_DIR, dataset_name


class MinMaxTransform:
    """Apply the stored affine SOH normalization between physical units and [-1, 1]."""
    def __init__(self, minimum: NDArray[np.floating], maximum: NDArray[np.floating]) -> None:
        """Copy the training-fitted minimum and maximum into CPU tensors."""
        self.min = torch.from_numpy(minimum.copy())
        self.max = torch.from_numpy(maximum.copy())

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Map physical values to the normalization space used during training."""
        return (data - self.min) / (self.max - self.min + 1e-8) * 2 - 1

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Recover physical values from their normalized representation."""
        scaled = (data + 1) / 2
        return scaled * (self.max - self.min + 1e-8) + self.min

    def to(self, device: str | torch.device) -> MinMaxTransform:
        """Move stored transform tensors to device and return this transform."""
        self.min, self.max = self.min.to(device), self.max.to(device)
        return self


class LegacyRULTransform:
    """Historical log + z-score transform, not a definition of paper RUL."""
    def __init__(self, mean: NDArray[np.floating], std: NDArray[np.floating],
                 base: float | NDArray[np.floating] = math.e) -> None:
        """Store training-fitted log-scale location, scale, and logarithm base."""
        self.mean = torch.from_numpy(mean.copy())
        self.std = torch.from_numpy(std.copy())
        self.base = float(base)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Map physical values to the normalization space used during training."""
        logged = torch.log(data) if self.base == math.e else torch.log(data) / math.log(self.base)
        return (logged - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Recover physical values from their normalized representation."""
        logged = data * self.std + self.mean
        return torch.exp(logged) if self.base == math.e else torch.pow(self.base, logged)

    def to(self, device: str | torch.device) -> LegacyRULTransform:
        """Move stored transform tensors to device and return this transform."""
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


@dataclass
class DataSplit(Dataset[dict[str, torch.Tensor]]):
    """Aligned capacity features (B, H, W) and scaled labels for one cell split."""
    feature: torch.Tensor
    label: torch.Tensor

    def __len__(self) -> int:
        """Return the number of batteries in the split."""
        return len(self.label)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one battery's feature matrix and scalar or trajectory label."""
        return {"feature": self.feature[index], "label": self.label[index]}

    def to(self, device: str | torch.device) -> DataSplit:
        """Move features and labels to device in place."""
        self.feature, self.label = self.feature.to(device), self.label.to(device)
        return self


@dataclass
class DataBundle:
    """Train/test tensors and training-fitted SOH/RUL transforms.

    Capacity matrices have shape (B, 100, 100), or (B, 20, 100) for MIX-20.
    SOH labels have shape (B, 256); physical SOH is a capacity fraction.
    Legacy scalar RUL labels remain separate from SOH-derived EOL cycles.
    """
    train_data: DataSplit
    test_data: DataSplit
    train_rul: DataSplit
    test_rul: DataSplit
    label_transformation: MinMaxTransform
    rul_transformation: LegacyRULTransform
    feature_transformation: None = None
    device: str = "cpu"

    def to(self, device: str | torch.device) -> DataBundle:
        """Move all splits and transform parameters to the same device."""
        for item in (self.train_data, self.test_data, self.train_rul, self.test_rul,
                     self.label_transformation, self.rul_transformation):
            item.to(device)
        self.device = str(device)
        return self


def get_dataset(name: str, *, datasets_dir: str | Path | None = None) -> DataBundle:
    """Load a benchmark NPZ without pickle and validate its SOH transform roundtrip."""
    path = (DATASETS_DIR if datasets_dir is None else Path(datasets_dir)) / f"{dataset_name(name)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}. See datasets/README.md.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    def tensor(key: str) -> torch.Tensor:
        """Wrap one copied archive array without changing its dtype or values."""
        return torch.from_numpy(arrays[key])
    label_transform = MinMaxTransform(arrays["label_min"], arrays["label_max"])
    rul_transform = LegacyRULTransform(arrays["rul_zscore_mean"], arrays["rul_zscore_std"], arrays["rul_log_base"])
    for split in ("train", "test"):
        physical = label_transform.inverse_transform(tensor(f"y_{split}_scaled")).numpy()
        if not np.array_equal(physical, arrays[f"y_{split}_physical"]):
            raise ValueError(f"SOH transform parity failed for {path.name}, {split}.")
    return DataBundle(
        train_data=DataSplit(tensor("x_train"), tensor("y_train_scaled")),
        test_data=DataSplit(tensor("x_test"), tensor("y_test_scaled")),
        train_rul=DataSplit(tensor("x_train"), tensor("rul_train_scaled")),
        test_rul=DataSplit(tensor("x_test"), tensor("rul_test_scaled")),
        label_transformation=label_transform,
        rul_transformation=rul_transform,
    )
