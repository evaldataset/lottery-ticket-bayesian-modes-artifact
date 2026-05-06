from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class SubspaceHMCConfig:
    steps: int
    step_size: float
    leapfrog_steps: int
    prior_precision: float
    burn_in: int
    sample_every: int
    subspace_dim: int
    direction_seed: int
    direction_scale: float = 1.0
    batchnorm_mode: str = "eval"


@dataclass(frozen=True)
class SubspaceHMCResult:
    samples: list[dict[str, torch.Tensor]]
    accept_rate: float
    energies: list[float]
    coordinate_norms: list[float]
    parameter_distances: list[float]


def random_orthonormal_directions(
    num_parameters: int,
    subspace_dim: int,
    device: torch.device,
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive")
    if subspace_dim > num_parameters:
        raise ValueError("subspace_dim cannot exceed parameter count")
    if scale <= 0:
        raise ValueError("direction scale must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw = torch.randn(
        num_parameters,
        subspace_dim,
        generator=generator,
        device=device,
    )
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.mul(scale)


def orthonormalize_directions(
    raw_directions: torch.Tensor,
    subspace_dim: int,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive")
    if scale <= 0:
        raise ValueError("direction scale must be positive")
    if raw_directions.ndim != 2:
        raise ValueError("raw_directions must have shape [parameters, directions]")
    if raw_directions.shape[1] < subspace_dim:
        raise ValueError("raw_directions has fewer columns than subspace_dim")

    raw = raw_directions[:, :subspace_dim].detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(raw).all():
        raise ValueError("raw_directions contains non-finite values")
    gram = raw.t().matmul(raw)
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues.min().item()) <= 1e-12:
        raise ValueError("raw_directions are rank deficient for requested subspace_dim")
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.mul(scale)


def _set_batchnorm_mode(model: nn.Module, mode: str) -> None:
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(f"unsupported batchnorm_mode: {mode}")


def _potential_and_grad(
    model: nn.Module,
    train_loader: DataLoader,
    base_position: torch.Tensor,
    directions: torch.Tensor,
    coordinates: torch.Tensor,
    device: torch.device,
    prior_precision: float,
    batchnorm_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position = base_position + directions.mv(coordinates)
    vector_to_parameters(position, model.parameters())
    _set_batchnorm_mode(model, batchnorm_mode)
    model.zero_grad(set_to_none=True)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    nll_value = torch.zeros((), device=device)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        loss = criterion(model(x), y)
        nll_value = nll_value + loss.detach()
        loss.backward()

    likelihood_grad = parameters_to_vector(
        [
            param.grad if param.grad is not None else torch.zeros_like(param)
            for param in model.parameters()
        ]
    ).detach()
    full_grad = likelihood_grad + prior_precision * position.detach()
    grad = directions.t().mv(full_grad)
    prior = 0.5 * prior_precision * torch.dot(position.detach(), position.detach())
    potential = nll_value.detach() + prior
    return potential, grad.detach(), position.detach()


def collect_subspace_hmc_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: SubspaceHMCConfig,
    directions: torch.Tensor | None = None,
) -> SubspaceHMCResult:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.step_size <= 0:
        raise ValueError("step_size must be positive")
    if config.leapfrog_steps <= 0:
        raise ValueError("leapfrog_steps must be positive")
    if config.prior_precision < 0:
        raise ValueError("prior_precision must be non-negative")
    if config.burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if config.sample_every <= 0:
        raise ValueError("sample_every must be positive")

    model.to(device)
    base_position = parameters_to_vector(model.parameters()).detach().to(device)
    if directions is None:
        directions = random_orthonormal_directions(
            base_position.numel(),
            config.subspace_dim,
            device=device,
            seed=config.direction_seed,
            scale=config.direction_scale,
        )
    else:
        if directions.shape[0] != base_position.numel():
            raise ValueError("directions parameter count does not match model parameters")
        directions = orthonormalize_directions(
            directions,
            config.subspace_dim,
            device=device,
            scale=config.direction_scale,
        )
    coordinates = torch.zeros(config.subspace_dim, device=device)
    potential, grad, position = _potential_and_grad(
        model,
        train_loader,
        base_position,
        directions,
        coordinates,
        device,
        config.prior_precision,
        config.batchnorm_mode,
    )

    samples: list[dict[str, torch.Tensor]] = []
    energies: list[float] = []
    coordinate_norms: list[float] = []
    parameter_distances: list[float] = []
    accepted = 0

    for step in range(config.steps):
        current_coordinates = coordinates.detach().clone()
        current_potential = potential.detach().clone()
        current_grad = grad.detach().clone()
        momentum = torch.randn_like(coordinates)
        current_hamiltonian = current_potential + 0.5 * torch.dot(momentum, momentum)

        proposal_coordinates = current_coordinates.detach().clone()
        proposal_momentum = momentum.detach().clone()
        proposal_grad = current_grad

        proposal_momentum = proposal_momentum - 0.5 * config.step_size * proposal_grad
        for leapfrog_idx in range(config.leapfrog_steps):
            proposal_coordinates = (
                proposal_coordinates + config.step_size * proposal_momentum
            )
            proposal_potential, proposal_grad, proposal_position = _potential_and_grad(
                model,
                train_loader,
                base_position,
                directions,
                proposal_coordinates,
                device,
                config.prior_precision,
                config.batchnorm_mode,
            )
            if leapfrog_idx != config.leapfrog_steps - 1:
                proposal_momentum = (
                    proposal_momentum - config.step_size * proposal_grad
                )
        proposal_momentum = proposal_momentum - 0.5 * config.step_size * proposal_grad
        proposal_momentum = -proposal_momentum
        proposal_hamiltonian = proposal_potential + 0.5 * torch.dot(
            proposal_momentum, proposal_momentum
        )

        log_accept_prob = current_hamiltonian - proposal_hamiltonian
        if torch.log(torch.rand((), device=device)) < log_accept_prob:
            coordinates = proposal_coordinates.detach()
            potential = proposal_potential.detach()
            grad = proposal_grad.detach()
            position = proposal_position.detach()
            accepted += 1
        else:
            coordinates = current_coordinates.detach()
            potential = current_potential.detach()
            grad = current_grad.detach()
            position = base_position + directions.mv(coordinates)
            vector_to_parameters(position, model.parameters())

        energies.append(float(potential.item()))
        coordinate_norms.append(float(coordinates.norm().item()))
        parameter_distances.append(float((position - base_position).norm().item()))
        if step >= config.burn_in and (step - config.burn_in) % config.sample_every == 0:
            vector_to_parameters(position, model.parameters())
            samples.append(state_to_cpu(model))

    return SubspaceHMCResult(
        samples=samples,
        accept_rate=accepted / max(1, config.steps),
        energies=energies,
        coordinate_norms=coordinate_norms,
        parameter_distances=parameter_distances,
    )
