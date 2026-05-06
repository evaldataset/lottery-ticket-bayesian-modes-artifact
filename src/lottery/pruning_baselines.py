from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.func import functional_call

from lottery.masks import Mask, global_score_mask
from lottery.models import weight_parameter_names
from lottery.train import load_trainable_state


def snip_mask(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    sparsity: float,
    max_batches: int = 1,
) -> Mask:
    load_trainable_state(model, initial_state)
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    names = weight_parameter_names(model)

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        loss = criterion(model(x), y) / max_batches
        loss.backward()

    scores = {}
    for name, param in model.named_parameters():
        if name in names:
            if param.grad is None:
                scores[name] = torch.zeros_like(param.detach()).cpu()
            else:
                scores[name] = (param.detach() * param.grad.detach()).abs().cpu()
    return global_score_mask(scores, names, sparsity=sparsity, largest=True)


def synflow_mask(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    input_shape: tuple[int, ...],
    device: torch.device,
    sparsity: float,
) -> Mask:
    load_trainable_state(model, initial_state)
    model.to(device)
    model.eval()
    names = weight_parameter_names(model)
    signs = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            signs[name] = torch.sign(param)
            param.abs_()

    model.zero_grad(set_to_none=True)
    x = torch.ones((1, *input_shape), device=device)
    torch.sum(model(x)).backward()

    scores = {}
    for name, param in model.named_parameters():
        if name in names:
            if param.grad is None:
                scores[name] = torch.zeros_like(param.detach()).cpu()
            else:
                scores[name] = (param.detach() * param.grad.detach()).abs().cpu()

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.mul_(signs[name])
    return global_score_mask(scores, names, sparsity=sparsity, largest=True)


def _freeze_low_scores_to_sparsity(
    score_parameters: Mapping[str, torch.Tensor],
    active_masks: dict[str, torch.Tensor],
    names: list[str],
    target_sparsity: float,
) -> None:
    flat_scores = torch.cat(
        [
            score_parameters[name].detach().flatten()[active_masks[name].flatten()]
            for name in names
        ]
    )
    if flat_scores.numel() == 0:
        return
    total = sum(active_masks[name].numel() for name in names)
    target_keep = max(1, int(round(total * (1.0 - target_sparsity))))
    current_keep = sum(int(active_masks[name].sum().item()) for name in names)
    remove_count = current_keep - target_keep
    if remove_count <= 0:
        return

    threshold = torch.topk(flat_scores, remove_count, largest=False).values.max()
    remaining_to_remove = remove_count
    for name in names:
        active = active_masks[name]
        candidates = active & (score_parameters[name].detach() <= threshold)
        candidate_count = int(candidates.sum().item())
        if candidate_count <= remaining_to_remove:
            active_masks[name] = active & ~candidates
            remaining_to_remove -= candidate_count
            continue
        flat_candidate = candidates.flatten().nonzero(as_tuple=False).flatten()
        selected = flat_candidate[:remaining_to_remove]
        flat_active = active.flatten().clone()
        flat_active[selected] = False
        active_masks[name] = flat_active.reshape_as(active)
        break


def gem_miner_mask(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    sparsity: float,
    epochs: int,
    lr: float,
    regularization: float = 0.0,
    freeze_period: int = 1,
    max_batches_per_epoch: int | None = None,
) -> Mask:
    """Train Gem-Miner-style STE scores on frozen initialization weights."""

    if epochs <= 0:
        raise ValueError("Gem-Miner epochs must be positive")
    if freeze_period <= 0:
        raise ValueError("Gem-Miner freeze period must be positive")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")

    load_trainable_state(model, initial_state)
    model.to(device)
    model.train()
    names = weight_parameter_names(model)
    base_state = {
        key: value.detach().to(device=device).clone()
        for key, value in model.state_dict().items()
    }
    score_parameters = {
        name: nn.Parameter(torch.rand_like(base_state[name], dtype=torch.float32))
        for name in names
    }
    active_masks = {
        name: torch.ones_like(base_state[name], dtype=torch.bool)
        for name in names
    }
    optimizer = torch.optim.SGD(score_parameters.values(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_weights = sum(active_masks[name].numel() for name in names)
    c = torch.log(torch.tensor(1.0 / (1.0 - sparsity))).item() / epochs

    for epoch in range(1, epochs + 1):
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            x = x.to(device)
            y = y.to(device)
            call_state = dict(base_state)
            for name in names:
                rounded = (score_parameters[name] >= 0.5).to(dtype=base_state[name].dtype)
                ste_mask = rounded.detach() + score_parameters[name] - score_parameters[name].detach()
                call_state[name] = base_state[name] * active_masks[name].to(
                    dtype=base_state[name].dtype
                ) * ste_mask
            logits = functional_call(model, call_state, (x,))
            loss = criterion(logits, y)
            if regularization > 0:
                score_l2 = sum(param.square().sum() for param in score_parameters.values())
                loss = loss + regularization * score_l2 / total_weights
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for param in score_parameters.values():
                    param.clamp_(0.0, 1.0)

        if epoch % freeze_period == 0 or epoch == epochs:
            target_sparsity = min(sparsity, 1.0 - torch.exp(torch.tensor(-c * epoch)).item())
            _freeze_low_scores_to_sparsity(
                score_parameters,
                active_masks,
                names,
                target_sparsity=target_sparsity,
            )

    scores = {
        name: (
            score_parameters[name].detach().cpu()
            * active_masks[name].detach().cpu().to(dtype=torch.float32)
        )
        for name in names
    }
    return global_score_mask(scores, names, sparsity=sparsity, largest=True)


def variational_pruning_mask(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    sparsity: float,
    epochs: int,
    lr: float,
    kl_weight: float = 1e-4,
    sparsity_weight: float = 10.0,
    entropy_weight: float = 1e-3,
    temperature_start: float = 2.0,
    temperature_end: float = 0.2,
    samples_per_batch: int = 1,
    max_batches_per_epoch: int | None = None,
) -> Mask:
    """Optimize Bernoulli mask probabilities on frozen initialization weights."""

    if epochs <= 0:
        raise ValueError("variational pruning epochs must be positive")
    if lr <= 0:
        raise ValueError("variational pruning lr must be positive")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("Concrete temperatures must be positive")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive")

    load_trainable_state(model, initial_state)
    model.to(device)
    model.train()
    names = weight_parameter_names(model)
    base_state = {
        key: value.detach().to(device=device).clone()
        for key, value in model.state_dict().items()
    }
    keep_prior = 1.0 - sparsity
    keep_prior = min(max(keep_prior, 1e-6), 1.0 - 1e-6)
    init_logit = torch.logit(torch.tensor(keep_prior, dtype=torch.float32)).item()
    mask_logits = {
        name: nn.Parameter(torch.full_like(base_state[name], init_logit, dtype=torch.float32))
        for name in names
    }
    optimizer = torch.optim.Adam(mask_logits.values(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    prior = torch.tensor(keep_prior, device=device)

    def temperature(epoch: int) -> float:
        if epochs == 1:
            return temperature_end
        ratio = (epoch - 1) / (epochs - 1)
        return temperature_start * ((temperature_end / temperature_start) ** ratio)

    for epoch in range(1, epochs + 1):
        temp = temperature(epoch)
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            x = x.to(device)
            y = y.to(device)
            nll = torch.tensor(0.0, device=device)
            for _ in range(samples_per_batch):
                call_state = dict(base_state)
                for name in names:
                    probs = torch.sigmoid(mask_logits[name])
                    noise = torch.rand_like(probs).clamp_(1e-6, 1.0 - 1e-6)
                    logistic = torch.log(noise) - torch.log1p(-noise)
                    soft_mask = torch.sigmoid((mask_logits[name] + logistic) / temp)
                    call_state[name] = base_state[name] * soft_mask.to(
                        dtype=base_state[name].dtype
                    )
                nll = nll + criterion(functional_call(model, call_state, (x,)), y)
            nll = nll / samples_per_batch

            probs = torch.cat(
                [torch.sigmoid(mask_logits[name]).flatten() for name in names]
            )
            probs = probs.clamp(1e-6, 1.0 - 1e-6)
            kl = (
                probs * (torch.log(probs) - torch.log(prior))
                + (1.0 - probs)
                * (torch.log1p(-probs) - torch.log1p(-prior))
            ).mean()
            expected_keep = probs.mean()
            entropy = -(
                probs * torch.log(probs) + (1.0 - probs) * torch.log1p(-probs)
            ).mean()
            loss = (
                nll
                + kl_weight * kl
                + sparsity_weight * (expected_keep - keep_prior) ** 2
                + entropy_weight * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probability_scores = {
        name: torch.sigmoid(mask_logits[name]).detach().cpu()
        for name in names
    }
    return global_score_mask(probability_scores, names, sparsity=sparsity, largest=True)


def hard_concrete_mask(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    sparsity: float,
    epochs: int,
    lr: float,
    l0_weight: float = 1e-4,
    sparsity_weight: float = 10.0,
    temperature_start: float = 2.0,
    temperature_end: float = 0.67,
    stretch_low: float = -0.1,
    stretch_high: float = 1.1,
    samples_per_batch: int = 1,
    max_batches_per_epoch: int | None = None,
) -> Mask:
    """Optimize hard-concrete L0 gates on frozen initialization weights."""

    if epochs <= 0:
        raise ValueError("hard-concrete epochs must be positive")
    if lr <= 0:
        raise ValueError("hard-concrete lr must be positive")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("hard-concrete temperatures must be positive")
    if not stretch_low < 0.0 < 1.0 < stretch_high:
        raise ValueError("hard-concrete stretch must satisfy low < 0 < 1 < high")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive")

    load_trainable_state(model, initial_state)
    model.to(device)
    model.train()
    names = weight_parameter_names(model)
    base_state = {
        key: value.detach().to(device=device).clone()
        for key, value in model.state_dict().items()
    }
    keep_prior = min(max(1.0 - sparsity, 1e-6), 1.0 - 1e-6)
    stretch_ratio = torch.tensor(-stretch_low / stretch_high, device=device)
    init_log_alpha = (
        torch.logit(torch.tensor(keep_prior, dtype=torch.float32, device=device))
        + temperature_start * torch.log(stretch_ratio)
    ).item()
    log_alpha = {
        name: nn.Parameter(torch.full_like(base_state[name], init_log_alpha, dtype=torch.float32))
        for name in names
    }
    optimizer = torch.optim.Adam(log_alpha.values(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def temperature(epoch: int) -> float:
        if epochs == 1:
            return temperature_end
        ratio = (epoch - 1) / (epochs - 1)
        return temperature_start * ((temperature_end / temperature_start) ** ratio)

    def sample_gate(alpha: torch.Tensor, temp: float) -> torch.Tensor:
        noise = torch.rand_like(alpha).clamp_(1e-6, 1.0 - 1e-6)
        logistic = torch.log(noise) - torch.log1p(-noise)
        soft = torch.sigmoid((alpha + logistic) / temp)
        stretched = soft * (stretch_high - stretch_low) + stretch_low
        return stretched.clamp(0.0, 1.0)

    for epoch in range(1, epochs + 1):
        temp = temperature(epoch)
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            x = x.to(device)
            y = y.to(device)
            nll = torch.tensor(0.0, device=device)
            for _ in range(samples_per_batch):
                call_state = dict(base_state)
                for name in names:
                    gate = sample_gate(log_alpha[name], temp)
                    call_state[name] = base_state[name] * gate.to(dtype=base_state[name].dtype)
                nll = nll + criterion(functional_call(model, call_state, (x,)), y)
            nll = nll / samples_per_batch

            expected_nonzero = torch.cat(
                [
                    torch.sigmoid(
                        log_alpha[name] - temp * torch.log(stretch_ratio)
                    ).flatten()
                    for name in names
                ]
            )
            expected_keep = expected_nonzero.mean()
            loss = (
                nll
                + l0_weight * expected_keep
                + sparsity_weight * (expected_keep - keep_prior) ** 2
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    gate_scores = {}
    for name in names:
        score = (
            torch.sigmoid(log_alpha[name]) * (stretch_high - stretch_low) + stretch_low
        ).clamp(0.0, 1.0)
        score = score.detach().cpu()
        tie_break = torch.linspace(
            0.0,
            1e-7,
            score.numel(),
            dtype=score.dtype,
        ).reshape_as(score)
        gate_scores[name] = score + tie_break
    return global_score_mask(gate_scores, names, sparsity=sparsity, largest=True)
