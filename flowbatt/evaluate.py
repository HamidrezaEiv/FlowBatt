"""Generate test trajectories and save point and predictive-interval metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from flowbatt.config import ModelName
from flowbatt.data import get_dataset
from flowbatt.paths import DATASETS, WORKSPACES_DIR, checkpoint_path, dataset_name
from flowbatt.inference import load_flowbatt, load_diffbatt, load_trf
from flowbatt.evaluation_metrics import (
    set_seed, inverse_soh, interval_metrics_from_samples, serialize_metrics,
    upsample_soh_percent, first_crossing,
)
from flowbatt.sampling import sample_flowbatt, sample_diffbatt


PAPER_CYCLE_INDICES = (100, 200, 300)


def cyclewise_rmse_percent(
    reference_percent: NDArray[np.floating], predicted_percent: NDArray[np.floating],
    *, eol_percent: float | None = 80.0,
) -> dict[str, NDArray[np.floating]]:
    """RMSE across batteries at paper cycle indices, separately for each draw.

    Inputs are already interpolated to physical-cycle resolution and expressed
    in SOH percent. Cycle numbers are one-based. Both arrays are clipped from
    below at ``eol_percent`` (default 80); pass 90 for MIX-20 or None to disable
    clipping. The input arrays are not modified.
    """
    reference = np.asarray(reference_percent)
    predicted = np.asarray(predicted_percent)
    if reference.ndim != 2 or predicted.ndim != 3 or predicted.shape[1:] != reference.shape:
        raise ValueError("Expected reference (batteries, cycles) and predictions (draws, batteries, cycles).")
    if reference.shape[1] < max(PAPER_CYCLE_INDICES):
        raise ValueError("Cycle-wise evaluation requires at least 300 physical-cycle values.")
    if eol_percent is not None:
        reference = np.maximum(reference, eol_percent)
        predicted = np.maximum(predicted, eol_percent)
    return {
        str(cycle): np.sqrt(np.mean(
            (predicted[:, :, cycle - 1] - reference[:, cycle - 1]) ** 2, axis=1
        ))
        for cycle in PAPER_CYCLE_INDICES
    }


def summarize_draws(values: NDArray[np.floating]) -> dict[str, float | list[float]]:
    """Summarize per-draw errors with their mean and population standard deviation."""
    return {"mean": float(np.mean(values)), "std_population": float(np.std(values)),
            "per_draw": values.tolist()}


def resolve_checkpoints(
    model: ModelName,
    dataset: str,
    checkpoints: list[Path] | None,
    num_samples: int | None,
    conditioning_dropout: float = 0.0,
) -> tuple[list[Path], int]:
    """Select weights and draw count, preserving explicit Transformer path order.

    Generative models sample from one checkpoint. Explicit Transformer weights
    each contribute one prediction; otherwise the first bundled seeds are used.
    """
    if num_samples is not None and num_samples < 1:
        raise ValueError("num-samples must be positive.")
    count = 10 if num_samples is None else num_samples
    if checkpoints:
        paths = [path.expanduser().resolve() for path in checkpoints]
        if model == "transformer":
            if len(set(paths)) != len(paths):
                raise ValueError("Supply each Transformer checkpoint only once.")
            if num_samples is not None and num_samples != len(paths):
                raise ValueError("For custom Transformer weights, num-samples must match the number of checkpoints.")
            count = len(paths)
        elif len(paths) != 1:
            raise ValueError("FlowBatt and DiffBatt accept one checkpoint; use num-samples for generative draws.")
    elif model == "transformer":
        if count > 10:
            raise ValueError("Ten bundled Transformer seeds are available; select at most ten or supply custom checkpoints.")
        paths = [checkpoint_path(model, dataset, seed) for seed in range(count)]
    else:
        paths = [checkpoint_path(model, dataset,
                                 synthesis=model == "flowbatt" and conditioning_dropout == 0.2)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return paths, count


@torch.inference_mode()
def main() -> None:
    """Generate predictions, evaluate the paper's metrics, and save new output files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flowbatt", "diffbatt", "transformer"), default="flowbatt")
    parser.add_argument("--dataset", type=dataset_name, choices=DATASETS, default="MATR-1")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Generative draws or bundled Transformer seeds (default: 10); "
                             "inferred from custom Transformer checkpoints when omitted.")
    parser.add_argument("--sample-chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--conditioning-dropout", type=float, choices=(0.0, 0.2), default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--cyclewise-clipping", choices=("eol", "none"), default="eol",
                        help="Clip reference and predicted SOH from below for cycle-wise RMSE: "
                             "eol (default) uses 80%%, or 90%% for MIX-20; none disables clipping. "
                             "Other metrics and saved predictions are unchanged.")
    parser.add_argument("--checkpoint", type=Path, action="append", default=None,
                        help="Use custom weights instead of bundled weights; repeat for multiple Transformer seeds.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_chunk < 1:
        parser.error("sample-chunk must be positive.")
    if args.model != "flowbatt" and (args.conditioning_dropout != 0 or args.guidance_scale != 1):
        parser.error("Conditioning dropout and guidance overrides apply to FlowBatt only.")
    try:
        checkpoints, args.num_samples = resolve_checkpoints(
            args.model, args.dataset, args.checkpoint, args.num_samples, args.conditioning_dropout)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    output = args.output or WORKSPACES_DIR / "evaluation" / args.dataset / args.model
    if (output / "samples.npz").exists() or (output / "metrics.json").exists():
        raise FileExistsError(f"Results already exist in {output}. Choose a new --output.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dataset = get_dataset(args.dataset)
    reference = inverse_soh(dataset, dataset.test_data.label)
    input_shape = (1,) + tuple(dataset.train_data.feature.shape[1:])
    mask_size = 2 if args.dataset == "MIX-20" else 10
    set_seed(args.seed)
    if args.model == "flowbatt":
        model, initializer, generate, config = load_flowbatt(input_shape, mask_size, device,
            args.dataset, class_dropout_prob=args.conditioning_dropout,
            guidance_scale=args.guidance_scale, checkpoint=checkpoints[0])
        if args.guidance_scale == 1 and device.type != "mps":
            samples, _ = sample_flowbatt(dataset, model, initializer, config, device,
                                        args.num_samples, args.sample_chunk)
        else:
            samples = np.stack([generate(dataset, model, initializer, device, config)[1].numpy()
                                for _ in range(args.num_samples)])
    elif args.model == "diffbatt":
        model, initializer, _, config, scheduler = load_diffbatt(
            input_shape, mask_size, device, args.dataset, checkpoint=checkpoints[0])
        samples = sample_diffbatt(dataset, model, initializer, scheduler, config, device,
                                 args.num_samples, args.sample_chunk, 1000)
    else:
        model, _, _ = load_trf(input_shape, device, args.dataset)
        draws = []
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            draws.append(inverse_soh(dataset, model(dataset.test_data.feature.to(device))))
        samples = np.stack(draws)
    if samples.shape != (args.num_samples,) + reference.shape or not np.isfinite(samples).all():
        raise ValueError("Invalid prediction shape or non-finite predictions.")
    threshold = 90.0 if args.dataset == "MIX-20" else 80.0
    ref = upsample_soh_percent(reference)
    predicted = upsample_soh_percent(samples)
    clipped_error = np.maximum(predicted, threshold) - np.maximum(ref, threshold)[None]
    per_draw_soh = np.sqrt(np.mean(clipped_error**2, axis=-1)).mean(axis=-1)
    rul_ref = first_crossing(ref, threshold)
    rul_predictions = np.stack([first_crossing(draw, threshold) for draw in predicted])
    per_draw_rul = np.sqrt(np.mean((rul_predictions - rul_ref)**2, axis=-1))
    per_draw_mape = np.mean(np.abs((rul_predictions - rul_ref) / rul_ref), axis=-1) * 100
    metrics = {"dataset": args.dataset, "model": args.model, "num_samples": args.num_samples,
        "checkpoints": [str(path.resolve()) for path in checkpoints],
        "seed": args.seed, "sample_chunk": args.sample_chunk, "device": str(device),
        "variability": "training seeds" if args.model == "transformer" else "generative draws from one checkpoint",
        "trajectory_soh_rmse_mean_over_batteries_percent": summarize_draws(per_draw_soh),
        "rul_rmse_cycles": summarize_draws(per_draw_rul), "rul_mape_percent": summarize_draws(per_draw_mape),
        "cyclewise_clipping": args.cyclewise_clipping,
        "cyclewise_soh_rmse_percent": {cycle: summarize_draws(values)
                                        for cycle, values in cyclewise_rmse_percent(
                                            ref, predicted,
                                            eol_percent=threshold if args.cyclewise_clipping == "eol" else None,
                                        ).items()},
        "eol_percent": threshold, "guidance_scale": args.guidance_scale,
        "conditioning_dropout": args.conditioning_dropout}
    if args.model != "transformer" and args.num_samples > 1:
        metrics["raw_empirical_90pct_prediction_intervals"] = serialize_metrics(
            interval_metrics_from_samples(reference, samples, threshold))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "samples.npz", reference=reference, samples=samples)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"SOH RMSE {per_draw_soh.mean():.3f} ± {per_draw_soh.std():.3f}%")
    print(f"RUL RMSE {per_draw_rul.mean():.2f} ± {per_draw_rul.std():.2f} cycles")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
