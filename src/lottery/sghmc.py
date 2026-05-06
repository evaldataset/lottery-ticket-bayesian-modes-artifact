from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class SGHMCConfig:
    steps: int
    lr: float
    momentum_decay: float
    temperature: float
    prior_precision: float
    burn_in: int
    sample_every: int
    num_train_examples: int
    likelihood_scale: str = "dataset"


def collect_sghmc_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: SGHMCConfig,
) -> list[dict[str, torch.Tensor]]:
    if not 0.0 <= config.momentum_decay < 1.0:
        raise ValueError("SGHMC momentum_decay must be in [0, 1)")
    if config.lr <= 0.0:
        raise ValueError("SGHMC lr must be positive")
    if config.temperature <= 0.0:
        raise ValueError("SGHMC temperature must be positive")

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loader_iter = iter(train_loader)
    samples: list[dict[str, torch.Tensor]] = []
    velocities = {
        id(param): torch.zeros_like(param)
        for param in model.parameters()
        if param.requires_grad
    }
    noise_std = (config.lr * config.temperature * (1.0 - config.momentum_decay**2)) ** 0.5

    for step in range(config.steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        nll_sum = criterion(model(x), y)
        if config.likelihood_scale == "dataset":
            loss = nll_sum * (config.num_train_examples / y.numel())
        elif config.likelihood_scale == "mean":
            loss = nll_sum / y.numel()
        else:
            raise ValueError(f"Unsupported likelihood_scale: {config.likelihood_scale}")
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                posterior_grad = param.grad + config.prior_precision * param
                velocity = velocities[id(param)]
                velocity.mul_(config.momentum_decay)
                velocity.add_(posterior_grad, alpha=-0.5 * config.lr)
                velocity.add_(torch.randn_like(param), alpha=noise_std)
                param.add_(velocity)

        if step >= config.burn_in and (step - config.burn_in) % config.sample_every == 0:
            samples.append(state_to_cpu(model))

    return samples
