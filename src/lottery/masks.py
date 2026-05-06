from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


Mask = dict[str, torch.Tensor]


def dense_mask(model: nn.Module) -> Mask:
    return {
        name: torch.ones_like(param, dtype=torch.bool)
        for name, param in model.named_parameters()
        if param.ndim > 1
    }


def clone_mask(mask: Mapping[str, torch.Tensor]) -> Mask:
    return {name: value.detach().clone() for name, value in mask.items()}


def apply_mask_(model: nn.Module, mask: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name].to(device=param.device, dtype=param.dtype))


def mask_gradients_(model: nn.Module, mask: Mapping[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in mask and param.grad is not None:
            param.grad.mul_(mask[name].to(device=param.device, dtype=param.grad.dtype))


def global_magnitude_mask_from_state(
    state_dict: Mapping[str, torch.Tensor],
    names: list[str],
    sparsity: float,
) -> Mask:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    weights = [state_dict[name].detach().abs().flatten().cpu() for name in names]
    flat = torch.cat(weights)
    keep = max(1, int(round(flat.numel() * (1.0 - sparsity))))
    if keep >= flat.numel():
        threshold = flat.min() - 1
    else:
        threshold = torch.topk(flat, keep, largest=True).values.min()
    return {
        name: (state_dict[name].detach().abs().cpu() >= threshold)
        for name in names
    }


def global_score_mask(
    scores: Mapping[str, torch.Tensor],
    names: list[str],
    sparsity: float,
    largest: bool = True,
) -> Mask:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    flat = torch.cat([scores[name].detach().flatten().cpu() for name in names])
    keep = max(1, int(round(flat.numel() * (1.0 - sparsity))))
    if keep >= flat.numel():
        threshold = flat.min() - 1 if largest else flat.max() + 1
    else:
        threshold = torch.topk(flat, keep, largest=largest).values[-1]
    if largest:
        return {name: (scores[name].detach().cpu() >= threshold) for name in names}
    return {name: (scores[name].detach().cpu() <= threshold) for name in names}


def combine_masks(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> Mask:
    return {name: left[name].bool() & right[name].bool() for name in left}


def mask_sparsity(mask: Mapping[str, torch.Tensor]) -> float:
    total = sum(value.numel() for value in mask.values())
    kept = sum(value.bool().sum().item() for value in mask.values())
    return 1.0 - kept / total


def support_jaccard(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    intersections = 0
    unions = 0
    for name in left:
        a = left[name].bool().flatten().cpu()
        b = right[name].bool().flatten().cpu()
        intersections += (a & b).sum().item()
        unions += (a | b).sum().item()
    return float(intersections / unions) if unions else 1.0


def hamming_similarity(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    equal = 0
    total = 0
    for name in left:
        a = left[name].bool().flatten().cpu()
        b = right[name].bool().flatten().cpu()
        equal += (a == b).sum().item()
        total += a.numel()
    return float(equal / total)


def random_mask_like(reference: Mapping[str, torch.Tensor], sparsity: float, seed: int) -> Mask:
    generator = torch.Generator()
    generator.manual_seed(seed)
    names = list(reference)
    sizes = [reference[name].numel() for name in names]
    total = sum(sizes)
    keep = max(1, int(round(total * (1.0 - sparsity))))
    scores = torch.rand(total, generator=generator)
    threshold = torch.topk(scores, keep, largest=True).values.min()
    flat_mask = scores >= threshold
    out: Mask = {}
    offset = 0
    for name, size in zip(names, sizes):
        out[name] = flat_mask[offset : offset + size].reshape(reference[name].shape)
        offset += size
    return out
