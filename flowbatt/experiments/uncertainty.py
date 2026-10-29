"""Evaluate empirical prediction intervals from pretrained generative trajectories."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from flowbatt.data import get_dataset
from flowbatt.evaluation_metrics import (
    DATASETS,
    IntervalMetrics,
    Scalar,
    eol_percent,
    interval_metrics_from_samples,
    inverse_soh,
    set_seed,
    write_csv,
)
from flowbatt.inference import load_diffbatt, load_flowbatt
from flowbatt.paths import WORKSPACES_DIR
from flowbatt.sampling import sample_diffbatt, sample_flowbatt

DEFAULT_RESULTS = WORKSPACES_DIR


def serializable_metrics(metrics: IntervalMetrics) -> dict[str, float]:
    """Select scalar statistics from interval metrics for CSV output."""
    return {
        key: value
        for key, value in metrics.items()
        if np.isscalar(value)
    }


def run_one(
    dataset_name: str,
    method: str,
    *,
    num_samples: int,
    sample_chunk: int,
    inference_steps: int,
    seed: int,
    device: torch.device,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Scalar]:
    """Generate or reload one model/dataset sample set and compute interval statistics."""
    output_dir = output_root / "generative_uq" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_{num_samples}_samples.npz"

    if output_path.exists() and not overwrite:
        artifact = np.load(output_path)
        reference = artifact["reference"]
        samples = artifact["samples"]
        runtime = float(artifact["runtime_seconds"])
        function_evaluations = (
            int(artifact["function_evaluations"])
            if "function_evaluations" in artifact.files
            else ""
        )
    else:
        dataset = get_dataset(dataset_name)
        reference = inverse_soh(dataset, dataset.test_data.label)
        input_shape = (1,) + tuple(dataset.train_data.feature.shape[1:])
        mask_size = 2 if dataset_name == "MIX-20" else 10
        set_seed(seed)
        started = time.perf_counter()
        if method == "flowbatt":
            model, initializer, _, config = load_flowbatt(
                input_shape, mask_size, device, dataset_name
            )
            samples, function_evaluations = sample_flowbatt(
                dataset,
                model,
                initializer,
                config,
                device,
                num_samples,
                sample_chunk,
            )
        elif method == "diffbatt":
            model, initializer, _, config, scheduler = load_diffbatt(
                input_shape, mask_size, device, dataset_name
            )
            samples = sample_diffbatt(
                dataset,
                model,
                initializer,
                scheduler,
                config,
                device,
                num_samples,
                sample_chunk,
                inference_steps,
            )
            function_evaluations = inference_steps * int(
                np.ceil(num_samples / sample_chunk)
            )
        else:
            raise ValueError(method)
        runtime = time.perf_counter() - started
        np.savez_compressed(
            output_path,
            reference=reference,
            samples=samples,
            runtime_seconds=np.float64(runtime),
            seed=np.int64(seed),
            inference_steps=np.int64(inference_steps),
            function_evaluations=np.int64(function_evaluations),
        )

    metrics = interval_metrics_from_samples(
        reference,
        samples,
        eol_percent(dataset_name),
        interval=0.90,
    )
    return {
        "dataset": dataset_name,
        "method": "FlowBatt" if method == "flowbatt" else "DiffBatt",
        "num_samples": len(samples),
        "nominal_coverage": 0.90,
        **serializable_metrics(metrics),
        "runtime_seconds": runtime,
        "seconds_per_sample": runtime / len(samples),
        "diffusion_inference_steps": inference_steps if method == "diffbatt" else "",
        "function_evaluations": function_evaluations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument(
        "--models", nargs="+", default=["flowbatt", "diffbatt"], choices=("flowbatt", "diffbatt")
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-chunk", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    rows = []
    for dataset_name in args.datasets:
        for method in args.models:
            print(f"Evaluating {method} on {dataset_name}", flush=True)
            row = run_one(
                dataset_name,
                method,
                num_samples=args.num_samples,
                sample_chunk=args.sample_chunk,
                inference_steps=args.diffusion_steps,
                seed=args.seed,
                device=device,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
            rows.append(row)
            write_csv(args.output_root / "generative_uq" / "summary.csv", rows)


if __name__ == "__main__":
    main()
