"""Shared training and inference settings loaded from configs/*.json."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from flowbatt.paths import ROOT, checkpoint_path

ModelName = Literal["flowbatt", "diffbatt", "transformer"]


@dataclass
class ModelConfig:
    """Typed settings shared by training, checkpoint loading, and sampling.

    Model-specific architecture/training values come from the named JSON file.
    Sampling defaults specify ten ODE output times.
    """

    model: ModelName
    sequence_length: int
    channels: int
    num_blocks: int
    num_epochs: int
    learning_rate: float
    lr_warmup_steps: int
    n_batches: int
    print_every: int
    seed: int = 0
    class_dropout_prob: float = 0.0
    guidance_scale: float = 1.0
    num_sample_steps: int = 10
    ode_step_size: float = 0.001

    def checkpoint_path(self, dataset_name: str = "MATR-1") -> str:
        """Locate bundled weights, using the synthesis FlowBatt variant if needed."""
        if self.model == "flowbatt" and self.class_dropout_prob not in (0.0, 0.2):
            raise ValueError("Bundled checkpoints use conditioning dropout 0.0 or 0.2.")
        return str(checkpoint_path(
            self.model, dataset_name, seed=self.seed,
            synthesis=self.model == "flowbatt" and self.class_dropout_prob == 0.2,
        ))


def load_config(
    model: ModelName,
    *,
    config_dir: str | Path | None = None,
    **overrides: int | float,
) -> ModelConfig:
    """Load one model's JSON settings and apply explicit numeric overrides.

    The same entry point is used for training and inference to avoid separate
    copies of architecture settings. Overrides support CLI options such as seed,
    conditioning dropout, epoch count, and guidance scale.
    """
    if model not in ("flowbatt", "diffbatt", "transformer"):
        raise ValueError(f"Unknown model {model!r}.")
    directory = ROOT / "configs" if config_dir is None else Path(config_dir)
    with (directory / f"{model}.json").open(encoding="utf-8") as handle:
        settings = json.load(handle)
    settings.update(overrides)
    return ModelConfig(model=model, **settings)
