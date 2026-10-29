"""Recompute metrics from generated workspace predictions without refitting models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flowbatt.evaluation_metrics import (
    DATASETS, eol_percent, interval_metrics_from_bounds,
    interval_metrics_from_samples, trajectory_rmse_percent, write_csv,
)
from flowbatt.paths import WORKSPACES_DIR


def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", "--benchmark-root", dest="results_root", type=Path,
                        default=WORKSPACES_DIR,
                        help="Generated result root (default: workspaces/); --benchmark-root is a compatibility alias.")
    parser.add_argument("--output", type=Path, default=WORKSPACES_DIR / "metrics_summary.csv")
    return parser.parse_args(argv)


def required_prediction_files(results_root: Path) -> list[Path]:
    """List the fresh prediction files required for the five-dataset summary."""
    paths = []
    for dataset in DATASETS:
        for directory, prefixes in (
            ("baselines", ("rf", "mlp")),
            ("quantile_baselines", ("qrf", "qmlp")),
        ):
            for prefix in prefixes:
                paths.extend(results_root / directory / dataset / f"{prefix}_seed_{seed}.npz"
                             for seed in range(10))
        paths.extend(results_root / "generative_uq" / dataset / f"{method}_10_samples.npz"
                     for method in ("flowbatt", "diffbatt"))
    return paths


def require_predictions(results_root: Path) -> None:
    """Check for required predictions before creating any output directories."""
    missing = [path for path in required_prediction_files(results_root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction files under {results_root}; first missing: {missing[0]}. "
            "Generate all five datasets with full_matrix_baselines --models rf mlp --seeds 10, "
            "quantile_baselines --models qrf qmlp --seeds 10, and "
            "uncertainty --num-samples 10. Each entry point runs with "
            "python -m flowbatt.experiments.<name>; use the same --output-root as this --results-root."
        )


def main() -> None:
    args = parse_args()
    require_predictions(args.results_root)
    rows = []
    for dataset in DATASETS:
        threshold = eol_percent(dataset)
        for prefix in ("rf", "mlp"):
            scores = []
            for seed in range(10):
                with np.load(args.results_root / "baselines" / dataset / f"{prefix}_seed_{seed}.npz") as data:
                    scores.append(trajectory_rmse_percent(data["reference"], data["prediction"], threshold))
            rows.append(dict(dataset=dataset, method=f"Full-matrix {prefix.upper()}",
                             metric="trajectory_soh_rmse_percent", mean=float(np.mean(scores)),
                             std_population=float(np.std(scores)), count=len(scores)))
        for directory, prefix, method in (
            ("quantile_baselines", "qrf", "Final-curve Quantile RF"),
            ("quantile_baselines", "qmlp", "Full-matrix Quantile MLP"),
        ):
            metrics = []
            for seed in range(10):
                with np.load(args.results_root / directory / dataset / f"{prefix}_seed_{seed}.npz") as data:
                    metrics.append(interval_metrics_from_bounds(data["reference"], data["lower"],
                                   data["center"], data["upper"], threshold))
            for key in ("trajectory_soh_rmse_percent", "coverage", "width"):
                values = [item[key] for item in metrics]
                rows.append(dict(dataset=dataset, method=method, metric=key,
                                 mean=float(np.mean(values)), std_population=float(np.std(values)), count=10))
        for method in ("flowbatt", "diffbatt"):
            with np.load(args.results_root / "generative_uq" / dataset / f"{method}_10_samples.npz") as data:
                metrics = interval_metrics_from_samples(data["reference"], data["samples"], threshold, interval=0.90)
                scores = [trajectory_rmse_percent(data["reference"], sample, threshold)
                          for sample in data["samples"]]
            rows.append(dict(dataset=dataset, method=method, metric="trajectory_soh_rmse_percent",
                             mean=float(np.mean(scores)), std_population=float(np.std(scores)),
                             count=len(scores)))
            for key in ("coverage", "width"):
                rows.append(dict(dataset=dataset, method=method, metric=key,
                                 mean=metrics[key], std_population="", count=10))
    write_csv(args.output, rows)
    print(f"Recomputed {len(rows)} metric rows without fitting models: {args.output}")


if __name__ == "__main__":
    main()
