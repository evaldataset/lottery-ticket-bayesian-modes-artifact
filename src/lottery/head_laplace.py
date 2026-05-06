from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class HeadLaplaceConfig:
    num_samples: int
    scale: float
    prior_precision: float
    damping: float = 1e-5
    hessian_batches: int | None = None
    num_train_examples: int | None = None
    max_parameters: int = 5000
    head_name: str | None = None


@dataclass(frozen=True)
class HeadLaplaceFactors:
    head_name: str
    weight_name: str
    bias_name: str | None
    base_state: dict[str, torch.Tensor]
    mean: torch.Tensor
    precision_cholesky: torch.Tensor
    in_features: int
    out_features: int
    has_bias: bool
    parameter_count: int
    examples_seen: int
    hessian_scale: float


def _last_linear_name(model: nn.Module) -> str:
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_names:
        raise ValueError("model has no nn.Linear module for head Laplace")
    return linear_names[-1]


def _named_linear(model: nn.Module, head_name: str | None) -> tuple[str, nn.Linear]:
    resolved = _last_linear_name(model) if head_name is None else head_name
    modules = dict(model.named_modules())
    if resolved not in modules:
        raise ValueError(f"head module {resolved!r} does not exist")
    module = modules[resolved]
    if not isinstance(module, nn.Linear):
        raise ValueError(f"head module {resolved!r} is not nn.Linear")
    return resolved, module


def estimate_head_laplace_factors(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: HeadLaplaceConfig,
) -> HeadLaplaceFactors:
    if config.prior_precision < 0.0:
        raise ValueError("Head Laplace prior_precision must be non-negative")
    if config.damping <= 0.0:
        raise ValueError("Head Laplace damping must be positive")
    if config.hessian_batches is not None and config.hessian_batches <= 0:
        raise ValueError("Head Laplace hessian_batches must be positive or None")

    head_name, head = _named_linear(model, config.head_name)
    has_bias = head.bias is not None
    in_features = int(head.in_features)
    out_features = int(head.out_features)
    augmented_features = in_features + int(has_bias)
    parameter_count = out_features * augmented_features
    if parameter_count > config.max_parameters:
        raise ValueError(
            f"head Laplace has {parameter_count} parameters, above max_parameters="
            f"{config.max_parameters}"
        )

    model.to(device)
    model.eval()
    head.to(device)
    hessian = torch.zeros(
        out_features,
        augmented_features,
        out_features,
        augmented_features,
        device=device,
        dtype=torch.float64,
    )
    captured_features: list[torch.Tensor] = []

    def capture_features(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_features.append(inputs[0].detach())

    handle = head.register_forward_pre_hook(capture_features)
    examples_seen = 0
    try:
        with torch.no_grad():
            for batch_idx, (x, _y) in enumerate(train_loader):
                if config.hessian_batches is not None and batch_idx >= config.hessian_batches:
                    break
                captured_features.clear()
                logits = model(x.to(device)).detach()
                if not captured_features:
                    raise RuntimeError("head feature hook did not capture any features")
                features = captured_features[-1].detach().to(dtype=torch.float64)
                if features.ndim != 2 or features.shape[1] != in_features:
                    raise ValueError(
                        "head Laplace expects a 2D final linear input with shape "
                        f"(batch, {in_features}), got {tuple(features.shape)}"
                    )
                if has_bias:
                    ones = torch.ones(
                        features.shape[0],
                        1,
                        device=device,
                        dtype=torch.float64,
                    )
                    features = torch.cat([features, ones], dim=1)
                probs = torch.softmax(logits.to(dtype=torch.float64), dim=1)
                covariance = torch.diag_embed(probs) - probs.unsqueeze(2) * probs.unsqueeze(1)
                hessian.add_(torch.einsum("nab,ni,nj->aibj", covariance, features, features))
                examples_seen += int(features.shape[0])
    finally:
        handle.remove()

    if examples_seen == 0:
        raise RuntimeError("no examples available for head Laplace Hessian estimate")

    hessian_scale = 1.0
    if config.num_train_examples is not None and examples_seen < config.num_train_examples:
        hessian_scale = float(config.num_train_examples / examples_seen)
        hessian.mul_(hessian_scale)

    precision = hessian.reshape(parameter_count, parameter_count).cpu()
    eye = torch.eye(parameter_count, dtype=precision.dtype)
    precision = precision + (config.prior_precision + config.damping) * eye
    jitter = config.damping
    for _ in range(6):
        try:
            chol = torch.linalg.cholesky(precision)
            break
        except RuntimeError:
            precision = precision + jitter * eye
            jitter *= 10.0
    else:
        chol = torch.linalg.cholesky(precision)

    weight = head.weight.detach().cpu().to(dtype=torch.float64)
    if has_bias:
        bias = head.bias.detach().cpu().to(dtype=torch.float64).reshape(out_features, 1)
        mean_matrix = torch.cat([weight, bias], dim=1)
    else:
        mean_matrix = weight

    return HeadLaplaceFactors(
        head_name=head_name,
        weight_name=f"{head_name}.weight",
        bias_name=f"{head_name}.bias" if has_bias else None,
        base_state=state_to_cpu(model),
        mean=mean_matrix.reshape(-1),
        precision_cholesky=chol,
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        parameter_count=parameter_count,
        examples_seen=examples_seen,
        hessian_scale=hessian_scale,
    )


def sample_head_laplace_from_factors(
    factors: HeadLaplaceFactors,
    config: HeadLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.num_samples <= 0:
        raise ValueError("Head Laplace num_samples must be positive")
    if config.scale <= 0.0:
        raise ValueError("Head Laplace scale must be positive")

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
        matrix = vector.reshape(factors.out_features, factors.in_features + int(factors.has_bias))
        sample = {name: value.detach().clone() for name, value in factors.base_state.items()}
        reference_weight = factors.base_state[factors.weight_name]
        sample[factors.weight_name] = matrix[:, : factors.in_features].to(reference_weight.dtype)
        if factors.has_bias and factors.bias_name is not None:
            reference_bias = factors.base_state[factors.bias_name]
            sample[factors.bias_name] = matrix[:, factors.in_features].to(reference_bias.dtype)
        samples.append(sample)
    return samples


def collect_head_laplace_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: HeadLaplaceConfig,
) -> tuple[list[dict[str, torch.Tensor]], HeadLaplaceFactors]:
    factors = estimate_head_laplace_factors(model, train_loader, device, config)
    return sample_head_laplace_from_factors(factors, config), factors
