from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class FullLaplaceConfig:
    num_samples: int
    scale: float
    prior_precision: float
    damping: float = 1e-5
    hessian_batches: int | None = None
    num_train_examples: int | None = None
    max_parameters: int = 2000


@dataclass(frozen=True)
class FullLaplaceFactors:
    parameter_names: tuple[str, ...]
    parameter_shapes: dict[str, tuple[int, ...]]
    parameter_slices: dict[str, tuple[int, int]]
    base_state: dict[str, torch.Tensor]
    mean: torch.Tensor
    precision_cholesky: torch.Tensor
    parameter_count: int
    examples_seen: int
    hessian_scale: float


def _trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not parameters:
        raise ValueError("full Laplace requires at least one trainable parameter")
    return parameters


def estimate_full_laplace_factors(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: FullLaplaceConfig,
) -> FullLaplaceFactors:
    if config.prior_precision < 0.0:
        raise ValueError("Full Laplace prior_precision must be non-negative")
    if config.damping <= 0.0:
        raise ValueError("Full Laplace damping must be positive")
    if config.hessian_batches is not None and config.hessian_batches <= 0:
        raise ValueError("Full Laplace hessian_batches must be positive or None")

    parameters = _trainable_parameters(model)
    parameter_names = tuple(parameters)
    parameter_shapes = {
        name: tuple(parameter.shape) for name, parameter in parameters.items()
    }
    parameter_slices: dict[str, tuple[int, int]] = {}
    means = []
    offset = 0
    for name, parameter in parameters.items():
        flat = parameter.detach().reshape(-1).to(dtype=torch.float32)
        means.append(flat)
        next_offset = offset + int(flat.numel())
        parameter_slices[name] = (offset, next_offset)
        offset = next_offset
    parameter_count = offset
    if parameter_count > config.max_parameters:
        raise ValueError(
            f"full Laplace has {parameter_count} parameters, above max_parameters="
            f"{config.max_parameters}"
        )

    model.to(device)
    model.eval()
    base_state = {
        key: value.detach().to(device=device).clone()
        for key, value in model.state_dict().items()
    }
    mean = torch.cat(means).to(dtype=torch.float32)

    def logits_for_vector(vector: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        call_state = dict(base_state)
        for name in parameter_names:
            start, end = parameter_slices[name]
            call_state[name] = vector[start:end].reshape(parameter_shapes[name]).to(
                dtype=base_state[name].dtype
            )
        logits = functional_call(model, call_state, (x_single.unsqueeze(0),))
        return logits.squeeze(0)

    jacobian_one = jacrev(logits_for_vector, argnums=0)
    hessian = torch.zeros(
        parameter_count,
        parameter_count,
        device=device,
        dtype=torch.float64,
    )
    examples_seen = 0
    vector = mean.to(device=device)

    for batch_idx, (x, _y) in enumerate(train_loader):
        if config.hessian_batches is not None and batch_idx >= config.hessian_batches:
            break
        x = x.to(device)
        jacobian = vmap(jacobian_one, in_dims=(None, 0))(vector, x)
        with torch.no_grad():
            call_state = dict(base_state)
            for name in parameter_names:
                start, end = parameter_slices[name]
                call_state[name] = vector[start:end].reshape(parameter_shapes[name]).to(
                    dtype=base_state[name].dtype
                )
            logits = functional_call(model, call_state, (x,)).detach()
            probs = torch.softmax(logits.to(dtype=torch.float64), dim=1)
            covariance = torch.diag_embed(probs) - probs.unsqueeze(2) * probs.unsqueeze(1)
            jacobian64 = jacobian.reshape(x.shape[0], logits.shape[1], parameter_count).to(
                dtype=torch.float64
            )
            weighted = torch.einsum("ncd,ndp->ncp", covariance, jacobian64)
            hessian.add_(torch.einsum("ncp,ncq->pq", jacobian64, weighted))
            examples_seen += int(x.shape[0])
        del jacobian

    if examples_seen == 0:
        raise RuntimeError("no examples available for full Laplace Hessian estimate")

    hessian_scale = 1.0
    if config.num_train_examples is not None and examples_seen < config.num_train_examples:
        hessian_scale = float(config.num_train_examples / examples_seen)
        hessian.mul_(hessian_scale)

    precision = hessian.cpu()
    del hessian
    if device.type == "cuda":
        torch.cuda.empty_cache()
    precision.diagonal().add_(config.prior_precision + config.damping)
    jitter = config.damping
    for _ in range(6):
        try:
            chol = torch.linalg.cholesky(precision)
            break
        except RuntimeError:
            precision.diagonal().add_(jitter)
            jitter *= 10.0
    else:
        chol = torch.linalg.cholesky(precision)

    return FullLaplaceFactors(
        parameter_names=parameter_names,
        parameter_shapes=parameter_shapes,
        parameter_slices=parameter_slices,
        base_state=state_to_cpu(model),
        mean=mean.cpu().to(dtype=torch.float64),
        precision_cholesky=chol,
        parameter_count=parameter_count,
        examples_seen=examples_seen,
        hessian_scale=hessian_scale,
    )


def sample_full_laplace_from_factors(
    factors: FullLaplaceFactors,
    config: FullLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.num_samples <= 0:
        raise ValueError("Full Laplace num_samples must be positive")
    if config.scale <= 0.0:
        raise ValueError("Full Laplace scale must be positive")

    samples: list[dict[str, torch.Tensor]] = []
    chol_t = factors.precision_cholesky.t()
    for _ in range(config.num_samples):
        noise = torch.randn(factors.parameter_count, dtype=torch.float64)
        delta = torch.linalg.solve_triangular(
            chol_t,
            noise.reshape(-1, 1),
            upper=True,
        ).reshape(-1)
        vector = factors.mean + delta * (config.scale**0.5)
        sample = {name: value.detach().clone() for name, value in factors.base_state.items()}
        for name in factors.parameter_names:
            start, end = factors.parameter_slices[name]
            reference = factors.base_state[name]
            sample[name] = vector[start:end].reshape(factors.parameter_shapes[name]).to(
                reference.dtype
            )
        samples.append(sample)
    return samples


def collect_full_laplace_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: FullLaplaceConfig,
) -> tuple[list[dict[str, torch.Tensor]], FullLaplaceFactors]:
    factors = estimate_full_laplace_factors(model, train_loader, device, config)
    return sample_full_laplace_from_factors(factors, config), factors
