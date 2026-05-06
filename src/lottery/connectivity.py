from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from lottery.train import evaluate, load_trainable_state


def interpolate_states(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, left_value in left.items():
        right_value = right[key]
        if torch.is_floating_point(left_value):
            out[key] = (1.0 - alpha) * left_value + alpha * right_value
        else:
            out[key] = left_value.clone()
    return out


def linear_path_losses(
    model_factory,
    left_state: Mapping[str, torch.Tensor],
    right_state: Mapping[str, torch.Tensor],
    data_loader: DataLoader,
    device: torch.device,
    points: int = 11,
) -> list[float]:
    if points < 2:
        raise ValueError("points must be at least 2")
    losses = []
    for alpha in np.linspace(0.0, 1.0, points):
        model = model_factory()
        load_trainable_state(model, interpolate_states(left_state, right_state, float(alpha)))
        losses.append(evaluate(model, data_loader, device)["loss"])
    return losses


def linear_barrier(losses: list[float]) -> float:
    if len(losses) < 2:
        raise ValueError("losses must contain at least two endpoints")
    endpoint = max(losses[0], losses[-1])
    return float(max(losses) - endpoint)

