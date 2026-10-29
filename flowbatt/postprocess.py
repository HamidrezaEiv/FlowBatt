"""Interpolate SOH trajectories and derive cycle-based degradation metrics."""
from __future__ import annotations

import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from flowbatt.data import DataBundle


def inverse_transform_soh(dataset: DataBundle, soh: torch.Tensor) -> torch.Tensor:
    """Convert normalized SOH (B, T) to CPU float32 capacity fractions."""
    soh = soh.detach().cpu()
    if dataset.label_transformation is not None:
        soh = dataset.label_transformation.inverse_transform(soh)
    return soh.float()


def upsample_soh(soh: torch.Tensor, scale_factor: int = 10) -> torch.Tensor:
    """Linearly upsample SOH to physical cycles using half-pixel interpolation."""
    return F.interpolate(
        soh.unsqueeze(1),
        scale_factor=scale_factor,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def soh_to_percent(dataset: DataBundle, soh: torch.Tensor,
                   scale_factor: int = 10) -> NDArray[np.floating]:
    """Inverse-transform and interpolate normalized SOH to physical-cycle percentages."""
    return (upsample_soh(inverse_transform_soh(dataset, soh), scale_factor) * 100.0).numpy()


def soh_percent_to_rul(soh_percent: NDArray[np.floating],
                       eol_percent: float = 80.0) -> NDArray[np.float32]:
    """Return first one-based EOL crossing cycles, or the last cycle if uncrossed."""
    cycles = np.arange(soh_percent.shape[1]) + 1
    below_eol = soh_percent <= eol_percent
    rul = np.full(soh_percent.shape[0], cycles[-1], dtype=np.float32)
    for i, row in enumerate(below_eol):
        crossing = np.flatnonzero(row)
        if len(crossing) > 0:
            rul[i] = float(cycles[crossing[0]])
    return rul

class PostProcess:
    """Evaluate SOH and RUL on the fixed 2,560-cycle benchmark horizon."""
    def __init__(self, eol: float = 80) -> None:
        """Set the EOL threshold in SOH percentage points."""
        super().__init__()
        self.cycles = np.arange(0, 2560) + 1
        self.eol = eol
        

    def post_process(self, ref: torch.Tensor, pred: torch.Tensor) -> tuple[NDArray, NDArray]:
        """Interpolate reference/prediction fractions (B, 256) to cycle-wise percentages."""
        ref = F.interpolate(ref.unsqueeze(1), scale_factor=(10), mode='linear').squeeze(1) * 100.0
        pred = F.interpolate(pred.unsqueeze(1), scale_factor=(10), mode='linear').squeeze(1) * 100.0

        ref = ref.cpu().numpy()
        pred = pred.cpu().numpy()
        return ref, pred
    
    def soh_rmse(self, ref: NDArray[np.floating], pred: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute per-battery SOH RMSE after clipping both curves at the EOL level."""
        refc = ref.copy()
        refc[refc <= self.eol] = self.eol
        predc = pred.copy()
        predc[predc <= self.eol] = self.eol
        
        error = refc - predc
        soh_rmse = np.sqrt(np.mean(error**2, axis = 1))
        return soh_rmse
    
    def eval_soh(self, refs: NDArray[np.floating], preds: NDArray[np.floating]) -> np.floating:
        """Average per-battery SOH RMSE, in percentage points."""
        soh_rmse = self.soh_rmse(refs, preds)
        return soh_rmse.mean()
    
    def get_rul(self, data: NDArray[np.floating]) -> NDArray[np.integer]:
        """Extract first EOL cycle, using the final horizon cycle if not crossed."""
        below = data <= self.eol
        rul_index = np.argmax(below, axis=1)
        rul_index = np.where(below.any(axis=1), rul_index, -1)
        rul = self.cycles[rul_index]
        return rul
        
    def eval_rul(self, refs: NDArray[np.floating], preds: NDArray[np.floating]) -> tuple[np.floating, np.floating, NDArray, NDArray]:
        """Return RUL RMSE (cycles), MAPE (%), and reference/prediction cycle arrays."""
        rul_ref = self.get_rul(refs)
        rul_pred = self.get_rul(preds)
        
        rul_rmse = np.sqrt(np.mean((rul_ref - rul_pred)**2))
        rul_mape = np.mean(np.abs((rul_ref - rul_pred) / rul_ref)) * 100

        return rul_rmse, rul_mape, rul_ref, rul_pred
    
    def plot_sample(
        self, ref: NDArray[np.floating], pred: NDArray[np.floating],
        lower: NDArray[np.floating] | None = None,
        upper: NDArray[np.floating] | None = None,
    ) -> tuple["Figure", NDArray]:
        """Plot cycle-wise percentages; lower and upper are optional interval bounds."""
        from flowbatt.plotting import plot_soh_samples

        return plot_soh_samples(ref, pred, eol=self.eol, cycles=self.cycles,
                                lower=lower, upper=upper)

