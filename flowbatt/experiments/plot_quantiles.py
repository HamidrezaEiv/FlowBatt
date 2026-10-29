"""Plot MIX-100 reference trajectories, predicted medians, and quantile intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from flowbatt import plotting  # noqa: F401  # Applies the repository's plotting style.
from flowbatt.paths import WORKSPACES_DIR
from flowbatt.postprocess import PostProcess

RESULTS_ROOT = WORKSPACES_DIR
DEFAULT_OUTPUT = WORKSPACES_DIR / "figures"
MODEL_ARTIFACTS = {
    "qrf": (
        Path("quantile_baselines") / "MIX-100",
        "qrf_seed_{seed}.npz",
        "samples_MIX-100_qrf.pdf",
    ),
    "qmlp": (
        Path("quantile_baselines") / "MIX-100",
        "qmlp_seed_{seed}.npz",
        "samples_MIX-100_qmlp.pdf",
    ),
}
DEFAULT_MODELS = ("qrf", "qmlp")


def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MIX-100 Quantile RF and Quantile MLP predictions for every test cell."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help="Root containing freshly generated baseline predictions (default: workspaces/).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=MODEL_ARTIFACTS,
    )
    return parser.parse_args(argv)


def require_predictions(results_root: Path, models: list[str] | tuple[str, ...], seed: int) -> None:
    """Check for required predictions before creating any output directories."""
    missing = []
    for model in models:
        directory, pattern, _ = MODEL_ARTIFACTS[model]
        path = results_root / directory / pattern.format(seed=seed)
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing quantile predictions:\n" + "\n".join(str(path) for path in missing) +
            "\nGenerate the selected methods first: "
            "python -m flowbatt.experiments.quantile_baselines "
            "--datasets MIX-100 --models qrf qmlp. Use the same --output-root as this "
            "--results-root and enough --seeds to include the selected zero-based --seed."
        )


def main() -> None:
    args = parse_args()
    require_predictions(args.results_root, args.models, args.seed)

    post_process = PostProcess(eol=80)
    for model in args.models:
        results_dir, artifact_pattern, filename = MODEL_ARTIFACTS[model]
        results_dir = args.results_root / results_dir
        artifact = np.load(
            results_dir / artifact_pattern.format(seed=args.seed)
        )
        reference = torch.from_numpy(artifact["reference"])
        reference_percent, center_percent = post_process.post_process(
            reference,
            torch.from_numpy(artifact["center"]),
        )
        _, lower_percent = post_process.post_process(
            reference,
            torch.from_numpy(artifact["lower"]),
        )
        _, upper_percent = post_process.post_process(
            reference,
            torch.from_numpy(artifact["upper"]),
        )
        fig, _ = post_process.plot_sample(
            reference_percent,
            center_percent,
            lower=lower_percent,
            upper=upper_percent,
        )
        output_path = args.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", transparent=True)
        print(output_path, flush=True)


if __name__ == "__main__":
    main()
