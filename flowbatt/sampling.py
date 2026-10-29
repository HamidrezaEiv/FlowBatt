"""SOH trajectory samplers, cached conditioning, and classifier-free guidance."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn
from diffusers import DDPMScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from tqdm.auto import tqdm

from flowbatt.config import ModelConfig
from flowbatt.data import DataBundle
from flowbatt.evaluation_metrics import inverse_soh
from flowbatt.models.dit import DiT, InitializerNet
from flowbatt.models.transformer import TrF

GeneratedTrajectories = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, NDArray[np.floating]]

def _expand_timestep(t: float | Tensor, batch_size: int, device: torch.device) -> Tensor:
    """Broadcast a scalar or singleton timestep to one value per battery."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    t = t.to(device)
    if t.ndim == 0:
        return t.expand(batch_size)
    if t.shape[0] == 1:
        return t.expand(batch_size)
    if t.shape[0] == batch_size:
        return t
    raise ValueError(f"Expected scalar or batch timestep, got shape {tuple(t.shape)}")


class WrappedFlowBattModel(ModelWrapper):
    """Adapt the conditional DiT to the flow-matching solver interface."""
    def __init__(self, model: DiT, guidance_scale: float = 1.0) -> None:
        """Store a vector-field model and its default classifier-free guidance scale."""
        super().__init__(model)
        self.guidance_scale = guidance_scale

    def forward(self, x: Tensor, t: float | Tensor,
                model_extras: Tensor | dict[str, Tensor | float] | None = None) -> Tensor:
        """Evaluate a vector field with capacity data supplied in model_extras."""
        if isinstance(model_extras, dict):
            c = model_extras["c"]
            guidance_scale = model_extras.get("guidance_scale", self.guidance_scale)
        else:
            c = model_extras
            guidance_scale = self.guidance_scale

        t = _expand_timestep(t, x.shape[0], x.device)
        return self.model.forward_with_guidance(x, t, c, guidance_scale)

def repeated_test_tensors(dataset: DataBundle, count: int,
                          device: torch.device) -> tuple[Tensor, int]:
    """Repeat test SOH labels draw-major and return the number of test batteries."""
    reference = dataset.test_data.label.to(device=device, dtype=torch.float32)
    return reference.repeat(count, 1), reference.shape[0]


def encode_condition(model: DiT, condition: Tensor) -> Tensor:
    """Encode capacity matrices once, without conditioning dropout."""
    if condition.ndim == 3:
        condition = condition.unsqueeze(1)
    return model.y_embedder(condition, train=False)


def forward_with_encoded_condition(
    model: DiT,
    x: torch.Tensor,
    timestep: torch.Tensor,
    condition_embedding: torch.Tensor,
) -> torch.Tensor:
    """Evaluate DiT with cached (B, D) capacity embeddings instead of raw matrices."""
    if timestep.ndim == 0:
        timestep = timestep.expand(x.shape[0])
    elif timestep.shape[0] == 1:
        timestep = timestep.expand(x.shape[0])
    hidden = model.first(x.unsqueeze(1)).transpose(1, 2) + model.pos_embed
    conditioning = model.t_embedder(timestep) + condition_embedding
    for block in model.blocks:
        hidden = block(hidden, conditioning)
    output = model.final(hidden, conditioning).squeeze(-1)
    return output * model.mask.to(x.device)


class CachedConditionFlowModel(nn.Module):
    """Reuse condition embeddings across ODE evaluations and count vector-field calls."""
    def __init__(self, model: DiT) -> None:
        """Wrap a DiT and initialize its ODE function-evaluation counter."""
        super().__init__()
        self.model = model
        self.function_evaluations = 0

    def forward(self, x: Tensor, t: Tensor,
                model_extras: Tensor | None = None) -> Tensor:
        """Evaluate one ODE state using the condition embedding in model_extras."""
        self.function_evaluations += 1
        return forward_with_encoded_condition(
            self.model, x, t, model_extras
        )


@torch.inference_mode()
def sample_flowbatt(
    dataset: DataBundle,
    model: DiT,
    initializer: InitializerNet,
    config: ModelConfig,
    device: torch.device,
    num_samples: int,
    sample_chunk: int,
) -> tuple[NDArray[np.floating], int]:
    """Draw (samples, batteries, T) SOH fractions and return the ODE call count."""
    model.eval()
    initializer.eval()
    if config.guidance_scale != 1.0:
        raise ValueError("The cached-condition evaluator requires guidance_scale=1.")
    wrapped = CachedConditionFlowModel(model).to(device)
    solver = ODESolver(velocity_model=wrapped)
    time_grid = torch.linspace(
        0, 1, config.num_sample_steps, device=device, dtype=torch.float32
    )
    base_condition = dataset.test_data.feature.to(device=device, dtype=torch.float32)
    base_condition_embedding = encode_condition(model, base_condition)
    output = []
    for start in range(0, num_samples, sample_chunk):
        count = min(sample_chunk, num_samples - start)
        reference, n_test = repeated_test_tensors(dataset, count, device)
        condition_embedding = base_condition_embedding.repeat(count, 1)
        initial = torch.randn(
            (count * n_test, config.sequence_length),
            dtype=torch.float32,
            device=device,
        )
        initial = initializer(
            initial, reference[:, : initializer.mask_size], condition_embedding
        )
        solution = solver.sample(
            time_grid=time_grid,
            x_init=initial,
            method="dopri5",
            step_size=None,
            return_intermediates=True,
            model_extras=condition_embedding,
        )
        final = solution[-1].reshape(count, n_test, config.sequence_length)
        for sample in final:
            output.append(inverse_soh(dataset, sample))
    return np.stack(output), wrapped.function_evaluations


@torch.inference_mode()
def sample_diffbatt(
    dataset: DataBundle,
    model: DiT,
    initializer: InitializerNet,
    scheduler: DDPMScheduler,
    config: ModelConfig,
    device: torch.device,
    num_samples: int,
    sample_chunk: int,
    inference_steps: int,
) -> NDArray[np.floating]:
    """Draw (samples, batteries, T) SOH fractions with cached DDPM conditioning."""
    model.eval()
    initializer.eval()
    scheduler.set_timesteps(inference_steps, device=device)
    base_condition = dataset.test_data.feature.to(device=device, dtype=torch.float32)
    base_condition_embedding = encode_condition(model, base_condition)
    output = []
    for start in range(0, num_samples, sample_chunk):
        count = min(sample_chunk, num_samples - start)
        reference, n_test = repeated_test_tensors(dataset, count, device)
        condition_embedding = base_condition_embedding.repeat(count, 1)
        sample = torch.randn(
            (count * n_test, config.sequence_length),
            dtype=torch.float32,
            device=device,
        )
        sample = initializer(
            sample, reference[:, : initializer.mask_size], condition_embedding
        )
        for timestep in scheduler.timesteps:
            time_batch = timestep.expand(sample.shape[0])
            predicted_noise = forward_with_encoded_condition(
                model, sample, time_batch, condition_embedding
            )
            sample = scheduler.step(
                model_output=predicted_noise,
                timestep=timestep,
                sample=sample,
            ).prev_sample
            sample = initializer(
                sample, reference[:, : initializer.mask_size], condition_embedding
            )
        sample = sample.reshape(count, n_test, config.sequence_length)
        for draw in sample:
            output.append(inverse_soh(dataset, draw))
    return np.stack(output)


@torch.inference_mode()
def generate_flowbatt(
    dataset: DataBundle,
    model: DiT,
    initializer: InitializerNet,
    device: str | torch.device,
    config: ModelConfig,
    num_samples: int = 1,
    return_records: bool = False,
    guidance_scale: float | None = None,
) -> GeneratedTrajectories:
    """Generate one trajectory per battery; use flowbatt.evaluate for many draws."""
    if num_samples != 1:
        raise ValueError("This low-level helper returns one draw. Use flowbatt.evaluate for multiple draws.")
    model.eval()
    device = torch.device(device)
    guidance_scale = config.guidance_scale if guidance_scale is None else guidance_scale
    wrapped_vf = WrappedFlowBattModel(model, guidance_scale=guidance_scale).to(device)

    x_test = dataset.test_data.label.to(device=device, dtype=torch.float32)
    c_test = dataset.test_data.feature.to(device=device, dtype=torch.float32)
    batch_size = x_test.shape[0]

    num_sample_steps = getattr(config, "num_sample_steps", 10)
    T = torch.linspace(0, 1, num_sample_steps, device=device, dtype=torch.float32)  # sample times
    x_init = torch.randn((batch_size, config.sequence_length), dtype=torch.float32, device=device)
    x_init = initializer(x_init, x_test[:, :initializer.mask_size], c_test)

    if device.type == "mps":
        from torchdiffeq import odeint

        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            """Evaluate the vector field for the float32 MPS ODE solver."""
            return wrapped_vf(x=x, t=t, model_extras=c_test)

        ode_options = {"dtype": torch.float32}
        sol = odeint(
            ode_func,
            x_init,
            T,
            method="dopri5",
            options=ode_options,
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method='dopri5',
            step_size=None,
            return_intermediates=True,
            model_extras=c_test,
        )

    x_test = dataset.label_transformation.inverse_transform(dataset.test_data.label.cpu())
    x_pred = dataset.label_transformation.inverse_transform(sol.cpu()[-1])

    records = []
    for rec in sol.cpu():
        records.append(dataset.label_transformation.inverse_transform(rec))
    if return_records:
        return x_test, x_pred, np.array(records)
    else:
        return x_test, x_pred

@torch.inference_mode()
def generate_diffbatt(
    dataset: DataBundle, model: DiT, initializer: InitializerNet,
    noise_scheduler: DDPMScheduler, device: str | torch.device,
    config: ModelConfig, num_samples: int = 1, return_records: bool = False,
) -> GeneratedTrajectories:
    """One complete 1,000-step reverse-diffusion draw per battery."""
    if num_samples != 1:
        raise ValueError("This low-level helper returns one draw. Use flowbatt.evaluate for multiple draws.")
    noise_scheduler.set_timesteps(1000, device=device)
    model.eval()
    x_test = dataset.test_data.label.to(device)
    c_test = dataset.test_data.feature.to(device)
    batch_size = x_test.shape[0]

    x = torch.randn((batch_size, config.sequence_length), dtype=torch.float32, device=device)
    x = initializer(x, x_test[:, :initializer.mask_size].to(device=device), c_test)
    records = []
    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps):
            tb = t.repeat(x.shape[0]).to(device)
            pred = model(x, tb, c_test)
            x = noise_scheduler.step(model_output=pred, timestep=t, sample=x).prev_sample
            x = initializer(x, x_test[:, :initializer.mask_size].to(device=device), c_test)
            if t % 100 == 0:
                records.append(dataset.label_transformation.inverse_transform(x.cpu()))

    x_test = dataset.label_transformation.inverse_transform(dataset.test_data.label.cpu())
    x_pred = dataset.label_transformation.inverse_transform(x.cpu())
    if return_records:
        return x_test, x_pred, np.array(records)
    else:
        return x_test, x_pred
@torch.inference_mode()
def generate_transformer(dataset: DataBundle, model: TrF,
                         device: str | torch.device) -> tuple[Tensor, Tensor]:
    """Predict one deterministic SOH trajectory per battery, in physical fractions."""
    model.eval()
    x_test = dataset.test_data.label.to(device)
    c_test = dataset.test_data.feature.to(device)
    pred = model(c_test)

    x_test = dataset.label_transformation.inverse_transform(dataset.test_data.label.cpu())
    x_pred = dataset.label_transformation.inverse_transform(pred.cpu())
    return x_test, x_pred
