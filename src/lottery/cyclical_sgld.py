from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class CyclicalSGLDConfig:
    steps: int
    lr: float
    lr_min_ratio: float
    cycle_length: int
    temperature: float
    prior_precision: float
    burn_in: int
    sample_every: int
    num_train_examples: int
    likelihood_scale: str = "dataset"
    sample_phase_start: float = 0.0


def cyclic_lr(step: int, config: CyclicalSGLDConfig) -> tuple[float, float]:
    if config.cycle_length <= 1:
        return config.lr, 1.0
    cycle_step = step % config.cycle_length
    phase = cycle_step / float(config.cycle_length - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    min_lr = config.lr * config.lr_min_ratio
    return min_lr + (config.lr - min_lr) * cosine, phase


def collect_cyclical_sgld_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: CyclicalSGLDConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.lr <= 0.0:
        raise ValueError("Cyclical SGLD lr must be positive")
    if not 0.0 <= config.lr_min_ratio <= 1.0:
        raise ValueError("Cyclical SGLD lr_min_ratio must be in [0, 1]")
    if config.cycle_length <= 0:
        raise ValueError("Cyclical SGLD cycle_length must be positive")
    if config.temperature <= 0.0:
        raise ValueError("Cyclical SGLD temperature must be positive")
    if not 0.0 <= config.sample_phase_start <= 1.0:
        raise ValueError("Cyclical SGLD sample_phase_start must be in [0, 1]")

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loader_iter = iter(train_loader)
    samples: list[dict[str, torch.Tensor]] = []

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

        local_lr, phase = cyclic_lr(step, config)
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                posterior_grad = param.grad + config.prior_precision * param
                noise_std = (local_lr * config.temperature) ** 0.5
                param.add_(posterior_grad, alpha=-0.5 * local_lr)
                param.add_(torch.randn_like(param), alpha=noise_std)

        should_sample = (
            step >= config.burn_in
            and (step - config.burn_in) % config.sample_every == 0
            and phase >= config.sample_phase_start
        )
        if should_sample:
            samples.append(state_to_cpu(model))

    return samples
