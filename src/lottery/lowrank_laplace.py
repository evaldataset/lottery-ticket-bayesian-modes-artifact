from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from lottery.train import load_trainable_state, state_to_cpu


@dataclass(frozen=True)
class LowRankLaplaceConfig:
    num_samples: int
    scale: float
    prior_precision: float
    fisher_batches: int
    hessian_batches: int
    rank: int
    power_iterations: int = 1
    oversample: int = 4
    damping: float = 1e-6
    variance_floor: float = 1e-12
    eigenvalue_floor: float = 1e-12
    num_train_examples: int = 1
    batchnorm_mode: str = "eval"


@dataclass(frozen=True)
class LowRankLaplaceFactors:
    diag_fisher: torch.Tensor
    directions: torch.Tensor
    hessian_eigenvalues: torch.Tensor
    fisher_examples_seen: int
    hessian_examples_seen: int
    parameter_count: int

    @property
    def positive_rank(self) -> int:
        return int((self.hessian_eigenvalues > 0.0).sum().item())


def _trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _flatten_tensors(tensors: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _set_batchnorm_mode(model: nn.Module, mode: str) -> None:
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(f"unsupported batchnorm_mode: {mode}")


def _validate_config(config: LowRankLaplaceConfig) -> None:
    if config.num_samples <= 0:
        raise ValueError("LowRank Laplace num_samples must be positive")
    if config.scale <= 0.0:
        raise ValueError("LowRank Laplace scale must be positive")
    if config.prior_precision < 0.0:
        raise ValueError("LowRank Laplace prior_precision must be non-negative")
    if config.fisher_batches <= 0:
        raise ValueError("LowRank Laplace fisher_batches must be positive")
    if config.hessian_batches <= 0:
        raise ValueError("LowRank Laplace hessian_batches must be positive")
    if config.rank <= 0:
        raise ValueError("LowRank Laplace rank must be positive")
    if config.power_iterations < 0:
        raise ValueError("LowRank Laplace power_iterations must be non-negative")
    if config.oversample < 0:
        raise ValueError("LowRank Laplace oversample must be non-negative")
    if config.damping < 0.0:
        raise ValueError("LowRank Laplace damping must be non-negative")
    if config.variance_floor <= 0.0:
        raise ValueError("LowRank Laplace variance_floor must be positive")
    if config.eigenvalue_floor <= 0.0:
        raise ValueError("LowRank Laplace eigenvalue_floor must be positive")
    if config.num_train_examples <= 0:
        raise ValueError("LowRank Laplace num_train_examples must be positive")


def estimate_empirical_fisher_diag_vector(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: LowRankLaplaceConfig,
) -> tuple[torch.Tensor, int]:
    model.to(device)
    _set_batchnorm_mode(model, config.batchnorm_mode)
    params = _trainable_parameters(model)
    diag = torch.zeros(
        sum(param.numel() for param in params),
        device=device,
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    batches_seen = 0
    examples_seen = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y) / (y.numel() ** 0.5)
        loss.backward()
        grads = [
            param.grad if param.grad is not None else torch.zeros_like(param)
            for param in params
        ]
        diag.add_(_flatten_tensors(grads).detach().pow(2))
        batches_seen += 1
        examples_seen += int(y.numel())
        if batches_seen >= config.fisher_batches:
            break
    if batches_seen == 0:
        raise RuntimeError("no batches available for LowRank Laplace Fisher estimate")
    return diag.div(float(batches_seen)).cpu(), examples_seen


def hessian_vector_product(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    vector: torch.Tensor,
    max_batches: int,
    batchnorm_mode: str,
) -> tuple[torch.Tensor, int]:
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    params = _trainable_parameters(model)
    parameter_count = sum(param.numel() for param in params)
    if vector.numel() != parameter_count:
        raise ValueError("Hessian-vector product has mismatched parameter count")

    vector = vector.detach().to(device=device, dtype=torch.float32)
    result = torch.zeros_like(vector)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    examples_seen = 0
    batches_seen = 0
    for x, y in train_loader:
        _set_batchnorm_mode(model, batchnorm_mode)
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = _flatten_tensors(grads)
        grad_dot_vector = torch.dot(flat_grad, vector)
        hvp = torch.autograd.grad(grad_dot_vector, params)
        result.add_(_flatten_tensors(hvp).detach())
        examples_seen += int(y.numel())
        batches_seen += 1
        if batches_seen >= max_batches:
            break
    if examples_seen == 0:
        raise RuntimeError("no batches available for Hessian-vector products")
    return result.div(float(examples_seen)).cpu(), examples_seen


def estimate_lowrank_hessian(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: LowRankLaplaceConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    params = _trainable_parameters(model)
    parameter_count = sum(param.numel() for param in params)
    probe_dim = min(parameter_count, config.rank + config.oversample)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        parameter_count,
        probe_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    q, _ = torch.linalg.qr(q, mode="reduced")
    examples_seen = 0

    for _ in range(config.power_iterations):
        hq_columns = []
        for column in range(q.shape[1]):
            hvp, seen = hessian_vector_product(
                model,
                train_loader,
                device,
                q[:, column],
                max_batches=config.hessian_batches,
                batchnorm_mode=config.batchnorm_mode,
            )
            hq_columns.append(hvp.to(device))
            examples_seen = max(examples_seen, seen)
        hq = torch.stack(hq_columns, dim=1)
        q, _ = torch.linalg.qr(hq, mode="reduced")

    hq_columns = []
    for column in range(q.shape[1]):
        hvp, seen = hessian_vector_product(
            model,
            train_loader,
            device,
            q[:, column],
            max_batches=config.hessian_batches,
            batchnorm_mode=config.batchnorm_mode,
        )
        hq_columns.append(hvp.to(device))
        examples_seen = max(examples_seen, seen)
    hq = torch.stack(hq_columns, dim=1)
    projected = q.t().matmul(hq)
    projected = 0.5 * (projected + projected.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(projected.double())
    order = torch.argsort(eigenvalues, descending=True)
    keep = order[: config.rank]
    directions = q.double().matmul(eigenvectors[:, keep]).float()
    directions, _ = torch.linalg.qr(directions, mode="reduced")
    return directions.cpu(), eigenvalues[keep].float().cpu(), examples_seen


def estimate_lowrank_laplace_factors(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: LowRankLaplaceConfig,
    seed: int,
) -> LowRankLaplaceFactors:
    _validate_config(config)
    model.to(device)
    parameter_count = parameters_to_vector(_trainable_parameters(model)).numel()
    diag_fisher, fisher_examples_seen = estimate_empirical_fisher_diag_vector(
        model,
        train_loader,
        device,
        config,
    )
    directions, eigenvalues, hessian_examples_seen = estimate_lowrank_hessian(
        model,
        train_loader,
        device,
        config,
        seed=seed,
    )
    if directions.shape[0] != parameter_count:
        raise RuntimeError("LowRank Laplace directions have wrong parameter count")
    return LowRankLaplaceFactors(
        diag_fisher=diag_fisher,
        directions=directions,
        hessian_eigenvalues=eigenvalues,
        fisher_examples_seen=fisher_examples_seen,
        hessian_examples_seen=hessian_examples_seen,
        parameter_count=parameter_count,
    )


def sample_lowrank_laplace_from_factors(
    model: nn.Module,
    factors: LowRankLaplaceFactors,
    device: torch.device,
    config: LowRankLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    _validate_config(config)
    model.to(device)
    base_state = state_to_cpu(model)
    params = _trainable_parameters(model)
    base_vector = parameters_to_vector(params).detach().cpu().float()
    if base_vector.numel() != factors.parameter_count:
        raise ValueError("LowRank Laplace factors do not match model parameter count")

    diag_precision = (
        config.prior_precision
        + config.damping
        + config.num_train_examples * factors.diag_fisher.float()
    ).clamp_min(config.variance_floor)
    inv_sqrt_diag = diag_precision.rsqrt()
    lowrank_precision = (
        config.num_train_examples * factors.hessian_eigenvalues.float().clamp_min(0.0)
    )
    active = lowrank_precision > config.eigenvalue_floor
    q_correction: torch.Tensor | None = None
    rho_active: torch.Tensor | None = None
    if bool(active.any()):
        weighted = (
            factors.directions[:, active].float()
            * lowrank_precision[active].sqrt().unsqueeze(0)
        )
        whitened = inv_sqrt_diag.unsqueeze(1) * weighted
        gram = whitened.t().matmul(whitened)
        gram = 0.5 * (gram + gram.t())
        rho, rotation = torch.linalg.eigh(gram.double())
        rho_mask = rho > config.eigenvalue_floor
        if bool(rho_mask.any()):
            q_correction = (
                whitened.double().matmul(rotation[:, rho_mask])
                / rho[rho_mask].sqrt().unsqueeze(0)
            ).float()
            rho_active = rho[rho_mask].float()

    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(config.num_samples):
        z = torch.randn_like(base_vector)
        whitened_noise = z
        if q_correction is not None and rho_active is not None:
            coeff = q_correction.t().matmul(z)
            shrink = 1.0 - (1.0 + rho_active).rsqrt()
            whitened_noise = z - q_correction.matmul(shrink * coeff)
        sample_vector = base_vector + (config.scale**0.5) * inv_sqrt_diag * whitened_noise
        vector_to_parameters(sample_vector.to(device), params)
        samples.append(state_to_cpu(model))

    load_trainable_state(model, base_state)
    return samples


def collect_lowrank_laplace_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: LowRankLaplaceConfig,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    factors = estimate_lowrank_laplace_factors(
        model,
        train_loader,
        device,
        config,
        seed=seed,
    )
    return sample_lowrank_laplace_from_factors(model, factors, device, config)
