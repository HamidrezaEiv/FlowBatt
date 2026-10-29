"""Plots for SOH trajectories, capacity matrices, and RUL predictions.

Plotting utilities adapted from BatteryML (MIT).
Copyright (c) Microsoft Corporation. See the repository-root LICENSE.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
import numpy as np
import os
from flowbatt.paths import WORKSPACES_DIR

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

USE_TEX = os.environ.get("FLOWBATT_USETEX", "0") == "1"
plt.rc('text', usetex=USE_TEX)
plt.rc('font', size=SMALL_SIZE,family='serif')          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE,linewidth=1)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_rul(
    ref: NDArray[np.floating], pred: NDArray[np.floating],
    std: NDArray[np.floating] | None = None, name: str = "res", save: bool = False,
) -> None:
    """Plot reference/predicted RUL in cycles, optionally colored by uncertainty."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.grid()
    c = ax.scatter(ref, pred, s=5, c=std, cmap='cool', zorder=3)
    ax.plot(ref, ref, c='tab:pink', lw=0.5, alpha=1, zorder=2)
    ax.set_xlabel('Reference')
    ax.set_ylabel('Prediction')
    ax.set_xlim(ref.min()//500 * 500, ref.max()//500 * 500 + 500)
    ax.set_ylim(ref.min()//500 * 500, ref.max()//500 * 500 + 500)
    ax.tick_params(direction="in",top=True,right=True,which='both')
    ax.set_aspect('equal', adjustable='box')    
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)
    cb = fig.colorbar(c, ax=ax, format='%d', shrink=0.8)
    cb.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    cb.ax.set_title(r'$\sigma_{RUL}$')
    if save:
        output = WORKSPACES_DIR / "figures" / f"rul_{name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight', transparent=True)
    

def plot_samples(samples: NDArray[np.floating], name: str = "samples") -> None:
    """Display capacity matrices (B, H, W), accepting an extra singleton channel."""
    if len(samples.shape) == 4:
        samples = np.squeeze(samples, axis=1)
    samples = np.flip(np.transpose(samples, (0, 2, 1)), axis=1)
    n = int(np.ceil(np.sqrt(samples.shape[0])))
    v = np.linspace(2.0, 3.6, num=samples.shape[1], endpoint=True)
    cycle = np.linspace(1.0, 100.0, num=samples.shape[2], endpoint=True)
    x, y = np.meshgrid(cycle, v)
    plt.set_cmap('jet')        
    fig, ax = plt.subplots(n, n, figsize=(n/2, n/2), sharey=True)
    for i in range(samples.shape[0]):
        axx = ax.flatten()[i]

        axx.pcolormesh(x, y, samples[i])
        axx.set_aspect('auto')
        axx.set_yticks([])
        axx.set_xticks([])
        
    for i in range(n**2 - samples.shape[0]):
        axx = ax.flatten()[-i-1]
        axx.set_axis_off()
    


def plot_synthetic_soh_by_scale(
    samples_by_scale: Mapping[float, NDArray[np.floating]],
    scales: Sequence[float], cycles: NDArray | None = None,
    save_path: str | Path | None = None, max_curves: int = 1000,
) -> tuple[Figure, NDArray]:
    """Plot the RF experiment's synthetic training curves by guidance scale."""
    if cycles is None:
        first = next(iter(samples_by_scale.values()))
        cycles = np.arange(first.shape[1]) + 1
    fig, axes = plt.subplots(1, len(scales), figsize=(1.65 * len(scales), 1.25),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scale in zip(axes, scales):
        samples = samples_by_scale[scale]
        if len(samples) > max_curves:
            indices = np.linspace(0, len(samples) - 1, max_curves).astype(int)
            samples = samples[indices]
        ax.plot(cycles, samples.T, lw=0.45, c='tab:blue', alpha=0.85)
        ax.set_title(f'$w={scale}$')
        ax.set_xlabel('Cycle')
        ax.grid()
        ax.tick_params(direction="in", top=True, right=True, which='both')
        ax.set_xlim(0, min(3000, cycles[-1]))
    axes[0].set_ylabel('SOH (\\%)' if USE_TEX else 'SOH (%)')
    axes[0].set_ylim(80, 105)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_soh_samples(
    ref: NDArray[np.floating], pred: NDArray[np.floating], *,
    eol: float = 80.0, cycles: NDArray | None = None,
    lower: NDArray[np.floating] | None = None,
    upper: NDArray[np.floating] | None = None,
) -> tuple[Figure, NDArray]:
    """Plot per-battery SOH percentages, hiding curves beyond their EOL crossing.

    Optional interval bounds use the same (batteries, physical cycles) shape.
    """
    if cycles is None:
        cycles = np.arange(2560) + 1
    n = int(np.ceil(np.sqrt(ref.shape[0])))
    ref = ref.copy()
    pred = pred.copy()
    if (lower is None) != (upper is None):
        raise ValueError("Both lower and upper bounds must be provided together.")
    if lower is not None:
        lower = lower.copy()
        upper = upper.copy()

    ref[ref <= eol] = np.nan
    pred[pred <= eol] = np.nan
    
    fig, ax = plt.subplots(n, n, figsize=(n/2, n/2), sharey=True)
    for i in range(ref.shape[0]):
        axx = ax.flatten()[i]

        if lower is not None:
            axx.fill_between(
                cycles,
                lower[i],
                upper[i],
                color='lightgray',
                label='90\\% prediction interval' if USE_TEX else '90% prediction interval',
            )
        axx.plot(cycles, ref[i], c='cyan', lw=1.5, label='Reference')
        axx.plot(cycles, pred[i], c='magenta', lw=1.5, ls = '--', label='Prediction')

        axx.set_ylim(eol, 115)
        axx.set_xlim(0, cycles[max(np.argmin(ref[i]>eol), np.argmin(pred[i]>eol))] // 500 * 500 + 500)
        
        axx.set_xticks([])
        axx.grid()
        
    for i in range(n**2 - ref.shape[0]):
        axx = ax.flatten()[-i-1]
        axx.set_axis_off()
    
    for axx in ax[:1, 0]:
        axx.set_ylabel('SOH (\\%)' if USE_TEX else 'SOH (%)', fontsize=8)
        
    for axx in ax[-1, :1]:
        axx.set_xlabel('Cycle', fontsize=8)

    return fig, ax
