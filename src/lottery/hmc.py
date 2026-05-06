from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class HMCConfig:
    steps: int
    step_size: float
    leapfrog_steps: int
    prior_precision: float
    burn_in: int
    sample_every: int


@dataclass(frozen=True)
class HMCResult:
    samples: list[dict[str, torch.Tensor]]
    accept_rate: float
    energies: list[float]


def _potential_and_grad(
    model: nn.Module,
    train_loader: DataLoader,
    position: torch.Tensor,
    device: torch.device,
    prior_precision: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vector_to_parameters(position, model.parameters())
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    nll = torch.zeros((), device=device)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        nll = nll + criterion(model(x), y)
    prior = 0.5 * prior_precision * torch.dot(position, position)
    potential = nll + prior
    potential.backward()
    likelihood_grad = parameters_to_vector(
        [
            param.grad if param.grad is not None else torch.zeros_like(param)
            for param in model.parameters()
        ]
    ).detach()
    grad = likelihood_grad + prior_precision * position.detach()
    return potential.detach(), grad


def collect_hmc_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: HMCConfig,
) -> HMCResult:
    model.to(device)
    model.train()
    position = parameters_to_vector(model.parameters()).detach().to(device)
    potential, grad = _potential_and_grad(
        model, train_loader, position, device, config.prior_precision
    )
    samples: list[dict[str, torch.Tensor]] = []
    energies: list[float] = []
    accepted = 0

    for step in range(config.steps):
        current_position = position.detach().clone()
        current_potential = potential.detach().clone()
        current_grad = grad.detach().clone()
        momentum = torch.randn_like(position)
        current_hamiltonian = current_potential + 0.5 * torch.dot(momentum, momentum)

        proposal_position = current_position.detach().clone()
        proposal_momentum = momentum.detach().clone()
        proposal_grad = current_grad

        proposal_momentum = proposal_momentum - 0.5 * config.step_size * proposal_grad
        for leapfrog_idx in range(config.leapfrog_steps):
            proposal_position = proposal_position + config.step_size * proposal_momentum
            proposal_potential, proposal_grad = _potential_and_grad(
                model,
                train_loader,
                proposal_position,
                device,
                config.prior_precision,
            )
            if leapfrog_idx != config.leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - config.step_size * proposal_grad
        proposal_momentum = proposal_momentum - 0.5 * config.step_size * proposal_grad
        proposal_momentum = -proposal_momentum
        proposal_hamiltonian = proposal_potential + 0.5 * torch.dot(
            proposal_momentum, proposal_momentum
        )

        log_accept_prob = current_hamiltonian - proposal_hamiltonian
        if torch.log(torch.rand((), device=device)) < log_accept_prob:
            position = proposal_position.detach()
            potential = proposal_potential.detach()
            grad = proposal_grad.detach()
            accepted += 1
        else:
            position = current_position.detach()
            potential = current_potential.detach()
            grad = current_grad.detach()
            vector_to_parameters(position, model.parameters())

        energies.append(float(potential.item()))
        if step >= config.burn_in and (step - config.burn_in) % config.sample_every == 0:
            vector_to_parameters(position, model.parameters())
            samples.append(state_to_cpu(model))

    return HMCResult(
        samples=samples,
        accept_rate=accepted / max(1, config.steps),
        energies=energies,
    )
