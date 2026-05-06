from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lottery.masks import Mask, apply_mask_, combine_masks, dense_mask, global_magnitude_mask_from_state, mask_sparsity
from lottery.models import weight_parameter_names
from lottery.train import evaluate, load_trainable_state, state_to_cpu, train_model


@dataclass(frozen=True)
class IMPResult:
    mask: Mask
    final_state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    history: list[dict[str, float]]


def iterative_magnitude_pruning(
    model_factory,
    initial_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    rounds: int,
    prune_fraction_per_round: float,
    epochs_per_round: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str = "constant",
    rewind_state: dict[str, torch.Tensor] | None = None,
    final_epochs: int | None = None,
) -> IMPResult:
    if not 0.0 < prune_fraction_per_round < 1.0:
        raise ValueError("prune_fraction_per_round must be in (0, 1)")
    model = model_factory()
    names = weight_parameter_names(model)
    current_mask = dense_mask(model)
    history: list[dict[str, float]] = []
    train_state = initial_state if rewind_state is None else rewind_state

    for round_idx in range(rounds):
        model = model_factory()
        load_trainable_state(model, train_state)
        apply_mask_(model, current_mask)
        train_model(
            model,
            train_loader,
            device,
            epochs=epochs_per_round,
            lr=lr,
            weight_decay=weight_decay,
            mask=current_mask,
            lr_schedule=lr_schedule,
        )
        trained_state = state_to_cpu(model)
        current_sparsity = mask_sparsity(current_mask)
        target_sparsity = 1.0 - (1.0 - current_sparsity) * (1.0 - prune_fraction_per_round)
        round_mask = global_magnitude_mask_from_state(trained_state, names, target_sparsity)
        current_mask = combine_masks(current_mask, round_mask)
        metrics = evaluate(model, test_loader, device)
        metrics.update(
            {
                "round": float(round_idx + 1),
                "sparsity": mask_sparsity(current_mask),
            }
        )
        history.append(metrics)

    final_model = model_factory()
    load_trainable_state(final_model, train_state)
    apply_mask_(final_model, current_mask)
    train_model(
        final_model,
        train_loader,
        device,
        epochs=epochs_per_round if final_epochs is None else final_epochs,
        lr=lr,
        weight_decay=weight_decay,
        mask=current_mask,
        lr_schedule=lr_schedule,
    )
    final_metrics = evaluate(final_model, test_loader, device)
    final_metrics["sparsity"] = mask_sparsity(current_mask)
    return IMPResult(
        mask=current_mask,
        final_state=state_to_cpu(final_model),
        metrics=final_metrics,
        history=history,
    )
