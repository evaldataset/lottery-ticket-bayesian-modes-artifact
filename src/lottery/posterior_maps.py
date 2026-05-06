from __future__ import annotations

from collections.abc import Mapping

import torch

from lottery.masks import Mask, global_score_mask


def posterior_score_masks(
    samples: list[Mapping[str, torch.Tensor]],
    names: list[str],
    sparsity: float,
    eps: float = 1e-8,
) -> dict[str, Mask]:
    if not samples:
        return {}

    stacked = {
        name: torch.stack([sample[name].detach().cpu() for sample in samples], dim=0)
        for name in names
    }
    mean = {name: values.mean(dim=0) for name, values in stacked.items()}
    variance = {name: values.var(dim=0, unbiased=False) for name, values in stacked.items()}
    std = {name: variance[name].sqrt() for name in names}
    rms = {name: (values.square().mean(dim=0) + eps).sqrt() for name, values in stacked.items()}
    snr = {name: mean[name].abs() / (std[name] + eps) for name in names}

    return {
        "posterior_mean_abs": global_score_mask(
            {name: mean[name].abs() for name in names}, names, sparsity, largest=True
        ),
        "posterior_rms": global_score_mask(rms, names, sparsity, largest=True),
        "posterior_snr": global_score_mask(snr, names, sparsity, largest=True),
        "posterior_high_variance": global_score_mask(variance, names, sparsity, largest=True),
        "posterior_low_variance": global_score_mask(variance, names, sparsity, largest=False),
    }

