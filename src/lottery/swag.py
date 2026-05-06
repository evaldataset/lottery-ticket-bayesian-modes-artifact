from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class SWAGConfig:
    epochs: int
    lr: float
    weight_decay: float
    collection_start_epoch: int
    sample_every_epochs: int
    max_snapshots: int
    num_samples: int
    scale: float = 1.0
    diagonal_scale: float = 1.0
    low_rank_scale: float = 1.0


@dataclass(frozen=True)
class SWAGResult:
    samples: list[dict[str, torch.Tensor]]
    snapshot_count: int
    parameter_count: int


@dataclass(frozen=True)
class SWAGPosterior:
    names: list[str]
    mean: torch.Tensor
    variance: torch.Tensor
    deviations: torch.Tensor
    base_state: dict[str, torch.Tensor]
    snapshot_count: int
    parameter_count: int


def _parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def _flatten_state(state: Mapping[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([state[name].detach().cpu().float().reshape(-1) for name in names])


def _state_from_vector(
    vector: torch.Tensor,
    base_state: Mapping[str, torch.Tensor],
    names: list[str],
) -> dict[str, torch.Tensor]:
    out = {key: value.detach().cpu().clone() for key, value in base_state.items()}
    offset = 0
    for name in names:
        reference = base_state[name]
        size = reference.numel()
        out[name] = vector[offset : offset + size].reshape(reference.shape).to(reference.dtype)
        offset += size
    return out


def fit_swag_posterior(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: SWAGConfig,
) -> SWAGPosterior:
    if config.epochs <= 0:
        raise ValueError("SWAG epochs must be positive")
    if config.sample_every_epochs <= 0:
        raise ValueError("SWAG sample_every_epochs must be positive")

    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    snapshots: list[dict[str, torch.Tensor]] = []

    for epoch in range(1, config.epochs + 1):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        should_collect = (
            epoch >= config.collection_start_epoch
            and (epoch - config.collection_start_epoch) % config.sample_every_epochs == 0
        )
        if should_collect:
            snapshots.append(state_to_cpu(model))
            if config.max_snapshots > 0 and len(snapshots) > config.max_snapshots:
                snapshots.pop(0)

    if not snapshots:
        raise ValueError("SWAG did not collect any snapshots; lower collection_start_epoch")

    names = _parameter_names(model)
    stacked = torch.stack([_flatten_state(snapshot, names) for snapshot in snapshots], dim=0)
    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0, unbiased=False).clamp_min(0.0)
    deviations = stacked - mean
    base_state = snapshots[-1]
    parameter_count = int(mean.numel())

    return SWAGPosterior(
        names=names,
        mean=mean,
        variance=variance,
        deviations=deviations,
        base_state=base_state,
        snapshot_count=len(snapshots),
        parameter_count=parameter_count,
    )


def sample_swag_posterior(
    posterior: SWAGPosterior,
    config: SWAGConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.num_samples <= 0:
        raise ValueError("SWAG num_samples must be positive")
    if config.scale < 0:
        raise ValueError("SWAG scale must be non-negative")
    if config.diagonal_scale < 0:
        raise ValueError("SWAG diagonal_scale must be non-negative")
    if config.low_rank_scale < 0:
        raise ValueError("SWAG low_rank_scale must be non-negative")

    samples: list[dict[str, torch.Tensor]] = []
    low_rank_available = posterior.snapshot_count > 1
    for _ in range(config.num_samples):
        vector = posterior.mean.clone()
        if config.diagonal_scale > 0:
            diag_multiplier = config.scale * config.diagonal_scale
            if low_rank_available and config.low_rank_scale > 0:
                diag_multiplier *= 0.5
            vector = (
                vector
                + posterior.variance.sqrt()
                * torch.randn_like(posterior.mean)
                * (diag_multiplier**0.5)
            )

        if low_rank_available and config.low_rank_scale > 0:
            z = torch.randn(posterior.snapshot_count)
            low_rank = posterior.deviations.t().matmul(z) / (
                (posterior.snapshot_count - 1) ** 0.5
            )
            vector = vector + low_rank * (0.5 * config.scale * config.low_rank_scale) ** 0.5

        samples.append(_state_from_vector(vector, posterior.base_state, posterior.names))

    return samples


def collect_swag_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: SWAGConfig,
) -> SWAGResult:
    posterior = fit_swag_posterior(model, train_loader, device, config)
    samples = sample_swag_posterior(posterior, config)

    return SWAGResult(
        samples=samples,
        snapshot_count=posterior.snapshot_count,
        parameter_count=posterior.parameter_count,
    )
