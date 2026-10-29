"""Load stored scalar RUL labels from the five shared benchmark datasets.

Capacity matrices have shape (cells, cycles, voltage points). The loader adds
the single channel axis expected by PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
DATASET = REPOSITORY / "datasets/MATR-1.npz"
WORKSPACES = ROOT / "workspaces"
DATASETS = ("MATR-1", "MATR-2", "HUST", "MIX-20", "MIX-100")


def dataset_name(value: str) -> str:
    """Return a canonical benchmark name, accepting case-insensitive names."""
    name = value.upper()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {value!r}; choose from {DATASETS}")
    return name


def dataset_path(name: str) -> Path:
    """Locate the shared numeric archive for a supported benchmark."""
    return REPOSITORY / "datasets" / f"{dataset_name(name)}.npz"


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object describing a training run or evaluation."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write a new JSON record, rejecting overwrites and nonfinite numbers."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def contained(path: str | Path, root: str | Path = ROOT) -> Path:
    """Resolve a path and require it to remain strictly inside ``root``."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path must be strictly inside {root}")
    return path


def resolve_package_path(path: str | Path) -> Path:
    """Interpret relative CLI paths from xaicnn/, independent of shell cwd."""
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_output(path: str | Path) -> Path:
    """Require an empty or new output directory inside package workspaces."""
    # Check the resolved parent as well: workspaces itself cannot be a symlink
    # redirecting generated outputs outside this package.
    contained(WORKSPACES)
    output = contained(resolve_package_path(path), WORKSPACES)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    return output


def load_training_data(path: str | Path = DATASET) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return NCHW matrices and stored log/Z-score RUL targets for training only.

    Shapes are (N, 1, H, 100) and (N,), with H=20 for MIX-20 and H=100
    otherwise. No test arrays are read, and no labels are derived from SOH.
    """
    with np.load(path, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x_train"][:, None], dtype=np.float32)
        y = np.ascontiguousarray(archive["rul_train_scaled"], dtype=np.float32)
    validate_split(x, y)
    return x, y


def validate_split(x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
    """Require aligned, finite NCHW capacity matrices and scalar RUL labels."""
    if (x.ndim != 4 or tuple(x.shape[1:]) not in ((1, 100, 100), (1, 20, 100))
            or len(x) == 0 or y.shape != (len(x),)):
        raise ValueError("Expected nonempty (N,1,H,100) matrices and (N,) targets; H must be 20 or 100")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite features or targets")


def validate_transform(data: dict[str, NDArray[Any]]) -> None:
    """Validate scalar log/Z-score parameters for the inverse RUL transform."""
    for key in ("label_mean", "label_std", "label_log_base"):
        if np.asarray(data[key]).shape != () or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid target normalization: {key}")
    base = float(data["label_log_base"])
    if float(data["label_std"]) <= 0 or base <= 0 or base == 1:
        raise ValueError("RUL standard deviation must be positive and log base positive and not one")


def load_training_transform(path: str | Path = DATASET) -> dict[str, NDArray[Any]]:
    """Read stored scalar-RUL normalization without accessing any test arrays."""
    with np.load(path, allow_pickle=False) as archive:
        data = {"label_mean": archive["rul_zscore_mean"].reshape(()).copy(),
                "label_std": archive["rul_zscore_std"].reshape(()).copy(),
                "label_log_base": archive["rul_log_base"].reshape(()).copy()}
    validate_transform(data)
    return data


def load_checkpoint_transform(path: str | Path = DATASET) -> dict[str, NDArray[Any]]:
    """Read the original MATR-1 weights' training transform, not metric targets.

    Released weights predict historical standardized log lifetimes. Their
    predictions must retain that inverse transform even when reference labels
    come from the regular stored RUL arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = {"label_mean": archive["xai_rul_log_mean"].reshape(()).copy(),
                "label_std": archive["xai_rul_log_std"].reshape(()).copy(),
                "label_log_base": np.asarray(np.e, dtype=np.float64)}
    validate_transform(data)
    return data


def load_evaluation_data(path: str | Path = DATASET) -> dict[str, NDArray[Any]]:
    """Load both splits and the stored train-fitted RUL normalization.

    Feature arrays have NCHW shape (N, 1, H, 100), with H=20 or H=100.
    Physical RUL labels are cycles; scaled labels are dimensionless. Row indices
    identify positions in each split, not original battery identifiers.
    """
    data = load_training_transform(path)
    with np.load(path, allow_pickle=False) as archive:
        for split in ("train", "test"):
            data[f"x_{split}"] = archive[f"x_{split}"][:, None].copy()
            data[f"y_{split}_scaled"] = archive[f"rul_{split}_scaled"].copy()
            data[f"y_{split}_physical"] = archive[f"rul_{split}_legacy"].copy()
            data[f"row_indices_{split}"] = np.arange(len(data[f"x_{split}"]), dtype=np.int64)
    for split in ("train", "test"):
        x = data[f"x_{split}"]
        for kind in ("scaled", "physical"):
            y = data[f"y_{split}_{kind}"]
            validate_split(x, y)
        if np.any(data[f"y_{split}_physical"] <= 0):
            raise ValueError(f"Invalid {split} RUL labels")
    if data["x_train"].shape[1:] != data["x_test"].shape[1:]:
        raise ValueError("Training and test capacity matrix shapes must match")
    return data
