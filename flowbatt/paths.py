"""Repository-relative paths shared by training and evaluation."""

from pathlib import Path
import os

ROOT = Path(os.environ.get("FLOWBATT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATASETS_DIR = ROOT / "datasets"
CHECKPOINTS_DIR = ROOT / "checkpoints"
WORKSPACES_DIR = ROOT / "workspaces"
DATASETS = ("MATR-1", "MATR-2", "HUST", "MIX-20", "MIX-100")
_ALIASES = dict(zip(("matr_1_Q", "matr_2_Q", "hust_Q", "mix_20_Q", "mix_100_Q"), DATASETS))


def dataset_name(value: str) -> str:
    """Resolve supported aliases to a canonical benchmark name."""
    name = _ALIASES.get(value, value.upper())
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {value!r}. Choose from {DATASETS}.")
    return name


def checkpoint_path(model: str, dataset: str, seed: int = 0, synthesis: bool = False) -> Path:
    """Locate weights by model, benchmark, and optional Transformer seed or synthesis mode."""
    name = dataset_name(dataset)
    model = {"trf": "transformer", "FlowBatt": "flowbatt", "DiffBatt": "diffbatt",
             "Transformer": "transformer"}.get(model, model)
    if model == "transformer":
        return CHECKPOINTS_DIR / model / name / f"seed_{seed}.pth"
    if model == "flowbatt_synthesis" or (model == "flowbatt" and synthesis):
        return CHECKPOINTS_DIR / "flowbatt_synthesis" / f"{name}.pth"
    if model not in ("flowbatt", "diffbatt"):
        raise ValueError(f"Unknown model {model!r}")
    return CHECKPOINTS_DIR / model / f"{name}.pth"
