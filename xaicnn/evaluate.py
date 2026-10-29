"""Evaluate a final-epoch CNN/XAI-CNN run against regular dataset RUL labels."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from data import (DATASETS, REPOSITORY, ROOT, WORKSPACES, contained, dataset_name,
                  dataset_path, load_checkpoint_transform, load_evaluation_data,
                  load_training_data, load_training_transform, read_json,
                  resolve_package_path, validate_output, write_json)
from models import CNN, make_model
from train import OPTIMIZER

CheckpointRecord = dict[str, Any]
EvaluationData = dict[str, NDArray[Any]]
RunRecords = tuple[dict[str, Any], list[CheckpointRecord], dict[str, NDArray[Any]]]
LabelTransform = dict[str, float]


def label_transform(data: EvaluationData) -> LabelTransform:
    """Convert loader normalization scalars to the saved protocol representation."""
    return {name: float(data[f"label_{name}"]) for name in ("mean", "std", "log_base")}


def validate_transform(value: Any, expected: LabelTransform) -> LabelTransform:
    """Require the checkpoint's saved transform to match the selected dataset."""
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Missing or invalid saved label_transform")
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not np.isfinite(item) for item in value.values()):
        raise ValueError("Invalid saved label_transform values")
    transform = {key: float(value[key]) for key in expected}
    if transform != expected:
        raise ValueError("Saved label_transform does not match the dataset's regular RUL transform")
    return transform


def original_checkpoints() -> RunRecords:
    """List released MATR-1 weights with their original prediction transform."""
    records = [{"seed": seed, "checkpoint": f"checkpoints/xai_cnn/seed_{seed:03d}.pth"}
               for seed in range(10)]
    metadata = {
        "kind": "released_xai_cnn", "model": "xai-cnn", "dataset": "MATR-1",
        "activation": "leaky_relu", "input_shape": [1, 100, 100],
        "target_protocol": "dataset_rul", "checkpoint_target_protocol": "historical_xai_rul",
        "label_transform": label_transform(load_checkpoint_transform(dataset_path("MATR-1"))),
        "prediction_transform_source": "Historical xai_rul_log_* statistics used by the released checkpoints",
        "reproduction_caveat": (
            "Released checkpoints were trained against historical xai_rul labels. "
            "Their predictions retain that historical inverse transform, but evaluation "
            "references use regular rul_* labels. These metrics do not reproduce the paper table."
        ),
    }
    return metadata, records, {}


def audit_run(run: str | Path) -> RunRecords:
    """Require a completed, train-only run using the regular dataset RUL labels.

    Training records establish the final-epoch policy. Model state compatibility
    is checked separately by strict loading, without binding runs to source hashes.
    """
    run = contained(resolve_package_path(run), WORKSPACES)
    contained(WORKSPACES)
    protocol = read_json(contained(run / "protocol.json", run))
    complete = read_json(contained(run / "training_complete.json", run))
    if protocol.get("target_protocol") != "dataset_rul":
        raise ValueError("This run uses historical or unspecified targets; retrain with the regular dataset rul_* labels")
    dataset = protocol.get("dataset")
    if not isinstance(dataset, str) or dataset_name(dataset) != dataset:
        raise ValueError("Training protocol must identify a canonical dataset")
    path = dataset_path(dataset)
    x, _ = load_training_data(path)
    count = len(x)
    model = protocol.get("model")
    if model not in ("cnn", "xai-cnn"):
        raise ValueError("Training protocol model must be cnn or xai-cnn")
    activation = "relu" if model == "cnn" else "leaky_relu"
    epochs = protocol.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Training protocol requires a positive integer epoch count")
    transform = validate_transform(protocol.get("label_transform"),
                                   label_transform(load_training_transform(path)))
    expected = {"experiment": "scalar_cnn_16ch", "activation": activation,
                "target_keys": ["rul_train_scaled", "rul_test_legacy"],
                "channels": 16, "input_shape": list(x.shape[1:]),
                "fit_count": count, "fit_indices": list(range(count)), "batch_size": count,
                "updates_per_epoch": 1, "shuffle": True, "optimizer": OPTIMIZER,
                "learning_rate_schedule": "constant", "checkpoint_policy": "final_epoch_only",
                "test_evaluation_during_training": False,
                "data_path": str(path.relative_to(REPOSITORY))}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Unexpected training setting: {key}")
    if complete.get("status") != "complete":
        raise ValueError("A successful training completion record is required")
    for key in ("experiment", "model", "dataset", "target_protocol", "seeds", "epochs",
                "checkpoint_policy", "test_evaluation_during_training"):
        if complete.get(key) != protocol.get(key):
            raise ValueError(f"Completion/protocol mismatch: {key}")
    seeds = protocol.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**32 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Invalid training seeds")
    entries = complete.get("runs")
    if (not isinstance(entries, list)
            or any(not isinstance(entry, dict) or not isinstance(entry.get("seed"), int) for entry in entries)):
        raise ValueError("Invalid checkpoint records")
    if sorted(entry["seed"] for entry in entries) != sorted(seeds):
        raise ValueError("Missing or duplicate checkpoint records")
    records: list[CheckpointRecord] = []
    curves: dict[str, NDArray[Any]] = {}
    prefix = "cnn" if model == "cnn" else "xai_cnn"
    for entry in sorted(entries, key=lambda value: value["seed"]):
        key = f"{prefix}_seed_{entry['seed']:03d}"
        if (entry["directory"] != key or entry["checkpoint"] != f"{key}/final.pt"
                or entry["epochs"] != epochs or entry["checkpoint_policy"] != "final_epoch_only"):
            raise ValueError(f"Invalid final-epoch checkpoint entry: {key}")
        training = read_json(contained(run / key / "training.json", run))
        if any(training.get(k) != value for k, value in entry.items()):
            raise ValueError(f"Training/completion mismatch: {key}")
        curve = training["curve"]
        losses = np.array([point["train_mse_before_update"] for point in curve])
        if ([point["epoch"] for point in curve] != list(range(1, epochs + 1))
                or not np.isfinite(losses).all() or np.any(losses < 0)):
            raise ValueError(f"Invalid training curve: {key}")
        curves[key] = losses
        final = contained(run / entry["checkpoint"], run)
        if not final.is_file():
            raise FileNotFoundError(f"Missing final checkpoint: {final}")
        records.append({"seed": entry["seed"], "checkpoint": str(final.relative_to(ROOT))})
    inventory = set(run.glob("cnn_seed_*/final.pt")) | set(run.glob("xai_cnn_seed_*/final.pt"))
    if inventory != {ROOT / entry["checkpoint"] for entry in records}:
        raise ValueError("Checkpoint inventory differs from completion record")
    return {"kind": f"trained_{prefix}", "model": model, "dataset": dataset,
            "activation": activation, "input_shape": list(x.shape[1:]), "epochs": epochs,
            "target_protocol": "dataset_rul", "checkpoint_target_protocol": "dataset_rul",
            "label_transform": transform, "prediction_transform_source": "Saved training protocol; validated against dataset",
            "checkpoint_policy": "final_epoch_only",
            "test_evaluation_during_training": False,
            "reproduction_caveat": protocol.get("reproduction_caveat", "New training uses regular dataset RUL labels.")}, records, curves


def metrics(prediction: NDArray[Any], reference: NDArray[Any]) -> dict[str, float]:
    """Return RMSE/MAE in cycles and MAPE in percent for paired RUL arrays."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Invalid metric inputs")
    return {"rmse_cycles": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae_cycles": float(np.mean(np.abs(p - y))),
            "mape_percent": float(100 * np.mean(np.abs(p - y) / y))}


def physical(scaled: torch.Tensor, transform: LabelTransform) -> NDArray[np.float32]:
    """Invert the checkpoint's own log/Z-score transform, returning RUL cycles."""
    logged = scaled * transform["std"] + transform["mean"]
    # Retain the original float32 exp calculation for the released checkpoints.
    result = (torch.exp(logged) if transform["log_base"] == float(np.e)
              else torch.pow(transform["log_base"], logged))
    return result.detach().cpu().numpy()


def gradcam(model: CNN, features: torch.Tensor) -> tuple[torch.Tensor, EvaluationData]:
    """Compute second-convolution Grad-CAM, globally normalized over the batch.

    ``features`` has shape (N, 1, H, 100), with H equal to 100 or 20. Predictions
    have shape (N,), and CAMs have shape (N, 30, 30) or (N, 4, 30), respectively.
    Per-cell gradient maxima have shape (N,).
    The differentiated output is the sum of dimensionless scaled RUL predictions.
    """
    captured = {}
    handles = [model.cnn[layer].register_forward_hook(
        lambda module, args, output, layer=layer: captured.__setitem__(layer, output))
        for layer in (3, 4)]
    try:
        x = features.detach().clone().requires_grad_(True)
        prediction = model(x)
        pre_grad, post_grad, input_grad = torch.autograd.grad(
            prediction.sum(), (captured[3], captured[4], x))
        signed = (pre_grad.mean((2, 3), keepdim=True) * captured[3]).sum(1)
        positive = signed.relu()
        normalized = (positive - positive.min()) / (positive.max() - positive.min() + 1e-8)
        result = {"signed_lowres": signed.detach().cpu().numpy(), "cam": normalized.detach().cpu().numpy()}
        for name, gradient in (("preactivation", pre_grad), ("postactivation", post_grad), ("input", input_grad)):
            result[f"{name}_gradient_abs_max"] = gradient.detach().abs().flatten(1).amax(1).cpu().numpy()
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite Grad-CAM output")
        return prediction.detach(), result
    finally:
        for handle in handles:
            handle.remove()


def plot_heatmaps(cams: NDArray[Any], path: Path, plt: ModuleType) -> None:
    """Save an oriented grid of (N, H, W) CAMs with the shared [0, 1] scale."""
    samples = np.flip(cams.transpose(0, 2, 1), axis=1)
    side = int(np.ceil(np.sqrt(len(samples))))
    fig, axes = plt.subplots(side, side, figsize=(5, 5), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index < len(samples):
            ax.imshow(samples[index], cmap="jet", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        if index >= len(samples):
            ax.set_axis_off()
    fig.subplots_adjust(wspace=.12, hspace=.12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(
    results: list[CheckpointRecord], arrays: EvaluationData, output: Path,
    family: str, plt: ModuleType, curves: dict[str, NDArray[Any]],
) -> None:
    """Plot RUL predictions, descriptive seed comparisons, and training losses."""
    columns = min(5, len(results))
    rows = int(np.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    y = arrays["reference_test_physical"]
    for ax, record in zip(axes.flat, results):
        prediction = arrays[f"seed_{record['seed']:03d}_test_physical"]
        bounds = [0, max(float(y.max()), float(prediction.max())) * 1.05]
        ax.scatter(y, prediction, s=10, color="teal")
        ax.plot(bounds, bounds, "k--", lw=.8)
        ax.set(xlim=bounds, ylim=bounds, xlabel="Reference RUL (cycles)", ylabel="Predicted RUL (cycles)",
               title=f"Seed {record['seed']}: RMSE {record['test']['rmse_cycles']:.2f}")
    for ax in list(axes.flat)[len(results):]:
        ax.set_axis_off()
    fig.suptitle(f"{family}: all checkpoints, {len(y)} test cells")
    fig.tight_layout()
    fig.savefig(output / "figures/rul_all_seeds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    ranked = sorted(results, key=lambda r: (r["test"]["rmse_cycles"], r["seed"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, label, record in zip(axes, ("Lowest", "Highest"), (ranked[0], ranked[-1])):
        ax.imshow(plt.imread(output / record["figure"]))
        ax.set_axis_off()
        ax.set_title(f"{label} test RMSE: seed {record['seed']}\n{record['test']['rmse_cycles']:.2f} cycles")
    fig.suptitle("Post-hoc seed comparison only; no checkpoint selection")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "figures/heatmaps_best_worst.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, losses in curves.items():
            ax.semilogy(np.arange(1, len(losses) + 1), losses, label=name)
        ax.set(xlabel="Epoch", ylabel="Scaled training MSE before update", title="Training losses; no test monitoring")
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "figures/training_curves.png", dpi=160)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Create the multi-dataset evaluation CLI without loading any arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed CNN/XAI-CNN run; relative paths start at xaicnn/")
    parser.add_argument("--dataset", type=dataset_name, choices=DATASETS,
                        help="Defaults to the run's dataset, or MATR-1 for released weights")
    parser.add_argument("--output", type=Path, help="New output directory under workspaces/; relative paths start at xaicnn/")
    return parser


def main() -> None:
    """Evaluate all supplied seeds and save metrics, arrays, and Grad-CAM plots."""
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.run:
            run = resolve_package_path(args.run)
            metadata, entries, curves = audit_run(run)
            if args.dataset is not None and args.dataset != metadata["dataset"]:
                raise ValueError("--dataset must match the dataset recorded in --run")
            default_output = run / "evaluation"
        else:
            if args.dataset is not None and args.dataset != "MATR-1":
                raise ValueError("Released weights support only MATR-1; train this dataset and evaluate with --run")
            metadata, entries, curves = original_checkpoints()
            default_output = WORKSPACES / "xai_cnn_evaluation"
        path = dataset_path(metadata["dataset"])
        data = load_evaluation_data(path)
        if list(data["x_train"].shape[1:]) != metadata["input_shape"]:
            raise ValueError("Checkpoint input shape does not match the evaluation dataset")
        output = validate_output(args.output or default_output)
        for entry in entries:
            checkpoint = contained(ROOT / entry["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    family = f"{'CNN' if metadata['model'] == 'cnn' else 'XAI-CNN'} / {metadata['dataset']}"
    # Default plotting caches stay local; respect explicit cache-directory overrides.
    cache = contained(ROOT / ".cache")
    os.environ.setdefault("MPLCONFIGDIR", str(contained(cache / "matplotlib")))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir()
    arrays: EvaluationData = {}
    cams: EvaluationData = {}
    results: list[CheckpointRecord] = []
    for split in ("train", "test"):
        arrays[f"reference_{split}_physical"] = data[f"y_{split}_physical"]
        arrays[f"reference_{split}_scaled"] = data[f"y_{split}_scaled"]
        arrays[f"row_indices_{split}"] = data[f"row_indices_{split}"]
    for entry in entries:
        model = make_model(metadata["activation"], entry["seed"], tuple(metadata["input_shape"][1:])).eval()
        model.load_state_dict(torch.load(contained(ROOT / entry["checkpoint"]), map_location="cpu", weights_only=True), strict=True)
        record = dict(entry)
        key = f"seed_{entry['seed']:03d}"
        for split in ("train", "test"):
            features = torch.from_numpy(data[f"x_{split}"])
            if split == "test":
                scaled, cam = gradcam(model, features)
                for name, value in cam.items():
                    cams[f"{key}_{name}"] = value
                record["gradient_diagnostics"] = {
                    name: np.flatnonzero(value == 0).tolist()
                    for name, value in cam.items() if name.endswith("gradient_abs_max")}
                record["gradient_diagnostics"]["zero_positive_cam_cells"] = np.flatnonzero(cam["cam"].max((1, 2)) == 0).tolist()
                record["figure"] = f"figures/heatmaps_{key}.png"
                plot_heatmaps(cam["cam"], output / record["figure"], plt)
            else:
                with torch.no_grad():
                    scaled = model(features)
            prediction = physical(scaled, metadata["label_transform"])
            arrays[f"{key}_{split}_scaled"] = scaled.numpy()
            arrays[f"{key}_{split}_physical"] = prediction
            record[split] = metrics(prediction, data[f"y_{split}_physical"])
        results.append(record)
        print(f"seed={entry['seed']} test_RMSE={record['test']['rmse_cycles']:.6f} test_MAPE={record['test']['mape_percent']:.6f}", flush=True)
    aggregate: dict[str, Any] = {"n_seeds": len(results), "std_ddof": 0}
    for split in ("train", "test"):
        aggregate[split] = {}
        for metric in ("rmse_cycles", "mae_cycles", "mape_percent"):
            values = np.array([record[split][metric] for record in results])
            aggregate[split][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    plot_comparisons(results, arrays, output, family, plt, curves)
    np.savez_compressed(output / "predictions.npz", **arrays)
    np.savez_compressed(output / "gradcam.npz", **cams)
    counts = {split: len(data[f"x_{split}"]) for split in ("train", "test")}
    summary = {
        "status": "complete", "model_label": family, "evaluation_device": "cpu", **metadata,
        "train_cells": counts["train"], "test_cells": counts["test"], "aggregate": aggregate, "runs": results,
        "reference_target_keys": ["rul_train_legacy", "rul_test_legacy"],
        "reference_label_transform": label_transform(data),
        "metric_convention": "Float32 inverse of the checkpoint's saved log/Z-score transform and log base; regular dataset physical RUL references; float64 metric accumulation; population SD over all seeds",
        "gradcam_convention": f"cnn[3] preactivation; scaled output sum; spatial mean of gradients; weighted channel sum; ReLU; global min-max over all {counts['test']} test maps with epsilon 1e-8; common plot range [0,1]",
        "gradient_diagnostic_convention": "Zero-based test-cell indices with exactly zero max absolute gradient. A zero positive CAM alone does not establish zero gradients.",
        "caveats": ["All provided seeds enter the aggregate. Best/worst figures are descriptive, not checkpoint selection.",
                    metadata["reproduction_caveat"],
                    "Grad-CAM follows the historical calculation, but figures are not claimed to reproduce original paper image pixels."],
        "data_path": str(path.relative_to(REPOSITORY)),
        "sample_index_convention": f"Zero-based row indices within each split of {path.relative_to(REPOSITORY)}; not original battery identifiers",
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(output / "summary.json", summary)
    print(f"Test RMSE: {aggregate['test']['rmse_cycles']['mean']:.6f} +/- {aggregate['test']['rmse_cycles']['std']:.6f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
