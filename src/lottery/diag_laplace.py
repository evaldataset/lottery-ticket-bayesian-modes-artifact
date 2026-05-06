from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class DiagonalLaplaceConfig:
    num_samples: int
    scale: float
    prior_precision: float
    fisher_batches: int
    variance_floor: float = 1e-12
    num_train_examples: int = 1


def estimate_minibatch_fisher_diag(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: DiagonalLaplaceConfig,
) -> dict[str, torch.Tensor]:
    if config.fisher_batches <= 0:
        raise ValueError("Diagonal Laplace fisher_batches must be positive")

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    batches_seen = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        # grad(sum loss) / sqrt(batch) squared is a cheap mini-batch diagonal
        # Fisher proxy that keeps the scale closer to a per-example estimate.
        loss = criterion(model(x), y) / (y.numel() ** 0.5)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name].add_(param.grad.detach().pow(2))
        batches_seen += 1
        if batches_seen >= config.fisher_batches:
            break

    if batches_seen == 0:
        raise RuntimeError("no batches available for diagonal Laplace Fisher estimate")
    return {name: value.div(batches_seen).detach().cpu() for name, value in fisher.items()}


def collect_diag_laplace_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: DiagonalLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.num_samples <= 0:
        raise ValueError("Diagonal Laplace num_samples must be positive")
    if config.scale <= 0.0:
        raise ValueError("Diagonal Laplace scale must be positive")
    if config.prior_precision < 0.0:
        raise ValueError("Diagonal Laplace prior_precision must be non-negative")
    if config.variance_floor <= 0.0:
        raise ValueError("Diagonal Laplace variance_floor must be positive")
    if config.num_train_examples <= 0:
        raise ValueError("Diagonal Laplace num_train_examples must be positive")

    base_state = state_to_cpu(model)
    fisher = estimate_minibatch_fisher_diag(model, train_loader, device, config)
    samples: list[dict[str, torch.Tensor]] = []

    for _ in range(config.num_samples):
        sample = {name: value.detach().clone() for name, value in base_state.items()}
        for name, base_value in base_state.items():
            if name not in fisher:
                continue
            precision = (
                config.prior_precision
                + config.num_train_examples * fisher[name].to(base_value.device)
            )
            variance = (config.scale / precision.clamp_min(config.variance_floor)).clamp_min(
                config.variance_floor
            )
            sample[name] = base_value + torch.randn_like(base_value) * variance.sqrt()
        samples.append(sample)

    return samples
