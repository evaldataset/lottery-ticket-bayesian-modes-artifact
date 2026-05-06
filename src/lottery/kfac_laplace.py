from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lottery.train import state_to_cpu


@dataclass(frozen=True)
class KFACLaplaceConfig:
    num_samples: int
    scale: float
    prior_precision: float
    fisher_batches: int
    damping: float = 1e-3
    factor_sample_rows: int = 8192
    num_train_examples: int = 1


@dataclass
class _LayerFactors:
    a_cov: torch.Tensor
    g_cov: torch.Tensor
    batches: int = 0


def _module_param_name(module_name: str, suffix: str) -> str:
    return f"{module_name}.{suffix}" if module_name else suffix


def _supported_layers(model: nn.Module) -> dict[str, nn.Module]:
    layers: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if isinstance(module, nn.Conv2d) and module.groups != 1:
                continue
            layers[name] = module
    return layers


def _linear_rows(value: torch.Tensor) -> torch.Tensor:
    return value.detach().reshape(-1, value.shape[-1]).float()


def _conv_input_rows(module: nn.Conv2d, value: torch.Tensor) -> torch.Tensor:
    patches = F.unfold(
        value.detach(),
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    return patches.transpose(1, 2).reshape(-1, patches.shape[1]).float()


def _conv_grad_rows(value: torch.Tensor) -> torch.Tensor:
    return value.detach().permute(0, 2, 3, 1).reshape(-1, value.shape[1]).float()


def _sample_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
    if max_rows <= 0 or rows.shape[0] <= max_rows:
        return rows
    idx = torch.randperm(rows.shape[0], device=rows.device)[:max_rows]
    return rows.index_select(0, idx)


def _covariance(rows: torch.Tensor) -> torch.Tensor:
    if rows.shape[0] == 0:
        raise RuntimeError("cannot build KFAC factor from zero rows")
    return rows.t().matmul(rows).div(float(rows.shape[0])).cpu()


def _inv_sqrt(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.t())
    eigvals, eigvecs = torch.linalg.eigh(matrix.double())
    inv_sqrt = eigvals.clamp_min(floor).rsqrt()
    return (eigvecs * inv_sqrt.unsqueeze(0)).matmul(eigvecs.t()).float()


def estimate_kfac_factors(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: KFACLaplaceConfig,
) -> dict[str, _LayerFactors]:
    if config.fisher_batches <= 0:
        raise ValueError("KFAC Laplace fisher_batches must be positive")
    if config.factor_sample_rows < 0:
        raise ValueError("KFAC Laplace factor_sample_rows must be non-negative")

    layers = _supported_layers(model)
    if not layers:
        raise ValueError("KFAC Laplace found no Linear or Conv2d layers")

    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    saved_inputs: dict[str, torch.Tensor] = {}
    factors: dict[str, _LayerFactors] = {}
    hooks = []

    def forward_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor):
            saved_inputs[name] = inputs[0].detach()

        return hook

    def backward_hook(name: str):
        def hook(module: nn.Module, _grad_input, grad_output):
            if name not in saved_inputs or grad_output[0] is None:
                return
            x = saved_inputs[name]
            g = grad_output[0].detach()
            if isinstance(module, nn.Linear):
                a_rows = _linear_rows(x)
                g_rows = _linear_rows(g)
            elif isinstance(module, nn.Conv2d):
                a_rows = _conv_input_rows(module, x)
                g_rows = _conv_grad_rows(g)
            else:
                return
            max_rows = config.factor_sample_rows
            if max_rows > 0 and a_rows.shape[0] > max_rows:
                idx = torch.randperm(a_rows.shape[0], device=a_rows.device)[:max_rows]
                a_rows = a_rows.index_select(0, idx)
                g_rows = g_rows.index_select(0, idx)
            else:
                a_rows = _sample_rows(a_rows, max_rows)
                g_rows = _sample_rows(g_rows, max_rows)
            a_cov = _covariance(a_rows)
            g_cov = _covariance(g_rows)
            if name not in factors:
                factors[name] = _LayerFactors(a_cov=a_cov, g_cov=g_cov, batches=1)
            else:
                factors[name].a_cov.add_(a_cov)
                factors[name].g_cov.add_(g_cov)
                factors[name].batches += 1

        return hook

    for name, module in layers.items():
        hooks.append(module.register_forward_hook(forward_hook(name)))
        hooks.append(module.register_full_backward_hook(backward_hook(name)))

    try:
        batches_seen = 0
        for x, y in train_loader:
            x = x.to(device).detach().requires_grad_(True)
            y = y.to(device)
            model.zero_grad(set_to_none=True)
            saved_inputs.clear()
            # Match the diagonal Laplace scaling: the squared gradient of
            # sum-loss/sqrt(batch) is a cheap mini-batch empirical-Fisher proxy.
            loss = criterion(model(x), y) / (y.numel() ** 0.5)
            loss.backward()
            batches_seen += 1
            if batches_seen >= config.fisher_batches:
                break
    finally:
        for hook in hooks:
            hook.remove()

    if not factors:
        raise RuntimeError("KFAC Laplace did not collect any layer factors")
    for factor in factors.values():
        factor.a_cov.div_(factor.batches)
        factor.g_cov.div_(factor.batches)
    return factors


def sample_kfac_laplace_from_factors(
    model: nn.Module,
    factors: dict[str, _LayerFactors],
    config: KFACLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.num_samples <= 0:
        raise ValueError("KFAC Laplace num_samples must be positive")
    if config.scale <= 0.0:
        raise ValueError("KFAC Laplace scale must be positive")
    if config.prior_precision < 0.0:
        raise ValueError("KFAC Laplace prior_precision must be non-negative")
    if config.damping <= 0.0:
        raise ValueError("KFAC Laplace damping must be positive")
    if config.num_train_examples <= 0:
        raise ValueError("KFAC Laplace num_train_examples must be positive")

    base_state = state_to_cpu(model)
    layer_modules = _supported_layers(model)
    inv_sqrts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    prior_per_example = config.prior_precision / float(config.num_train_examples)
    regularizer = config.damping + prior_per_example**0.5
    for name, factor in factors.items():
        a_eye = torch.eye(factor.a_cov.shape[0], dtype=factor.a_cov.dtype)
        g_eye = torch.eye(factor.g_cov.shape[0], dtype=factor.g_cov.dtype)
        inv_sqrts[name] = (
            _inv_sqrt(factor.a_cov + regularizer * a_eye, config.damping),
            _inv_sqrt(factor.g_cov + regularizer * g_eye, config.damping),
        )

    samples: list[dict[str, torch.Tensor]] = []
    noise_scale = (config.scale / float(config.num_train_examples)) ** 0.5
    for _ in range(config.num_samples):
        sample = {name: value.detach().clone() for name, value in base_state.items()}
        for module_name, module in layer_modules.items():
            param_name = _module_param_name(module_name, "weight")
            if param_name not in sample or module_name not in inv_sqrts:
                continue
            a_inv_sqrt, g_inv_sqrt = inv_sqrts[module_name]
            weight = sample[param_name]
            matrix_shape = (weight.shape[0], int(weight.numel() / weight.shape[0]))
            z = torch.randn(matrix_shape, dtype=torch.float32)
            noise = g_inv_sqrt.matmul(z).matmul(a_inv_sqrt) * noise_scale
            sample[param_name] = weight + noise.reshape_as(weight).to(weight.dtype)
        samples.append(sample)
    return samples


def collect_kfac_laplace_samples(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: KFACLaplaceConfig,
) -> list[dict[str, torch.Tensor]]:
    factors = estimate_kfac_factors(model, train_loader, device, config)
    return sample_kfac_laplace_from_factors(model, factors, config)
