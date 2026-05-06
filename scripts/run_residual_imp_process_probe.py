#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lottery.data import DatasetBundle, load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.diag_laplace import DiagonalLaplaceConfig, collect_diag_laplace_samples
from lottery.masks import (
    Mask,
    apply_mask_,
    combine_masks,
    dense_mask,
    global_magnitude_mask_from_state,
    global_score_mask,
    mask_sparsity,
    support_jaccard,
)
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu, train_model
from run_residual_predictor_mask_probe import (
    flatten_mask,
    flatten_scores,
    mask_fingerprint,
    mask_from_swaps,
    select_indices,
    train_fixed_mask,
    train_one_epoch,
    trajectory_score_tensors,
)


@dataclass(frozen=True)
class IMPProcessTrace:
    mask: Mask
    final_state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    history: list[dict[str, float]]
    round_masks: list[Mask]
    round_states: list[dict[str, torch.Tensor]]


@dataclass(frozen=True)
class SwapPlan:
    remove_idx: torch.Tensor
    add_idx: torch.Tensor
    swap_count: int
    base_only_count: int
    imp_only_count: int
    candidate_count: int


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_source_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--dataset",
        choices=["digits", "mnist", "fashion-mnist", "cifar10", "fake-cifar10"],
        default="cifar10",
    )
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="resnet20")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trajectory-epochs", default="0,1,2,5,10,20,30")
    parser.add_argument("--rewind-epochs", type=int, default=1)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--mask-train-epochs", type=int, default=None)
    parser.add_argument("--imp-rounds", type=int, default=5)
    parser.add_argument("--process-rounds", default="1,3,5")
    parser.add_argument(
        "--round-variants",
        default="survivor,final-imp",
        help=(
            "Comma-separated subset of survivor,survivor-random,survivor-low,"
            "final-imp,final-imp-dense-score,final-imp-base-score,"
            "final-imp-round-residualized-score,"
            "final-imp-round-posterior-residualized-score,"
            "final-imp-round-learned-subspace-residualized-score,"
            "final-imp-round-excluded-oracle,"
            "final-imp-round-excluded-layer-oracle "
            "(parameter-tensor matched),"
            "final-imp-round-excluded-tensor-score-oracle "
            "(parameter-tensor and round-score decile matched)."
        ),
    )
    parser.add_argument("--random-trials", type=int, default=1)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--base-sources", default="epoch_30,traj_rms_abs,epoch_10")
    parser.add_argument("--alphas", default="0.5")
    parser.add_argument("--posterior-projection-laplace-samples", type=int, default=10)
    parser.add_argument("--posterior-projection-laplace-scale", type=float, default=1e-3)
    parser.add_argument("--posterior-projection-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--posterior-projection-laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--posterior-projection-laplace-variance-floor", type=float, default=1e-12)
    parser.add_argument("--learned-subspace-rank", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_imp_process_probe"))
    return parser.parse_args()


def load_bundle(args: argparse.Namespace, seed: int) -> DatasetBundle:
    if args.dataset == "digits":
        return load_digits_bundle(args.batch_size, 1024, seed)
    if args.dataset == "fake-cifar10":
        return load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            seed,
            train_size=args.train_subset or 2048,
            test_size=args.test_subset or 512,
        )
    return load_torchvision_bundle(
        args.dataset,
        args.batch_size,
        1024,
        seed,
        flatten=args.model == "mlp",
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        augment=args.augment,
    )


def make_model_factory(args: argparse.Namespace, bundle: DatasetBundle):
    def model_factory() -> torch.nn.Module:
        if args.model == "mlp":
            return MLP(
                input_dim=bundle.input_dim,
                num_classes=bundle.num_classes,
                hidden_dim=args.hidden_dim,
                depth=args.depth,
            )
        if args.model == "tiny-cnn":
            return TinyCNN(
                input_shape=bundle.input_shape,
                num_classes=bundle.num_classes,
                width=args.cnn_width,
            )
        if args.model == "resnet20":
            return ResNetCIFAR(
                input_shape=bundle.input_shape,
                num_classes=bundle.num_classes,
                blocks_per_stage=3,
                width=args.resnet_width,
            )
        raise ValueError(f"Unsupported model: {args.model}")

    return model_factory


def iterative_magnitude_pruning_process_trace(
    *,
    model_factory,
    initial_state: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    rounds: int,
    prune_fraction_per_round: float,
    epochs_per_round: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str,
    rewind_state: dict[str, torch.Tensor] | None,
    final_epochs: int | None,
) -> IMPProcessTrace:
    model = model_factory()
    names = weight_parameter_names(model)
    current_mask = dense_mask(model)
    train_state = initial_state if rewind_state is None else rewind_state
    history: list[dict[str, float]] = []
    round_masks: list[Mask] = []
    round_states: list[dict[str, torch.Tensor]] = []

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
        round_states.append(trained_state)
        current_sparsity = mask_sparsity(current_mask)
        target_sparsity = 1.0 - (1.0 - current_sparsity) * (1.0 - prune_fraction_per_round)
        round_mask = global_magnitude_mask_from_state(trained_state, names, target_sparsity)
        current_mask = combine_masks(current_mask, round_mask)
        round_masks.append({name: current_mask[name].detach().clone() for name in names})
        metrics = evaluate(model, test_loader, device)
        metrics.update({"round": float(round_idx + 1), "sparsity": mask_sparsity(current_mask)})
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
    return IMPProcessTrace(
        mask=current_mask,
        final_state=state_to_cpu(final_model),
        metrics=final_metrics,
        history=history,
        round_masks=round_masks,
        round_states=round_states,
    )


def residual_counts(base: Mask, imp: Mask, names: list[str], alpha: float) -> tuple[int, int, int]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swappable = min(int(base_only.sum().item()), int(imp_only.sum().item()))
    return int(round(alpha * swappable)), int(base_only.sum().item()), int(imp_only.sum().item())


def make_swap_plan(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    add_scores: torch.Tensor,
    candidate_pool: torch.Tensor,
    alpha: float,
    add_order: str = "top",
    random_seed: int | None = None,
) -> SwapPlan:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    swap_count, base_only_count, imp_only_count = residual_counts(base, imp, names, alpha)
    candidate_pool = candidate_pool & ~base_flat
    swap_count = min(swap_count, base_only_count, int(candidate_pool.sum().item()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    if add_order == "top":
        add_idx = select_indices(add_scores, candidate_pool, swap_count, largest=True)
    elif add_order == "low":
        add_idx = select_indices(add_scores, candidate_pool, swap_count, largest=False)
    elif add_order == "random":
        candidates = torch.nonzero(candidate_pool, as_tuple=False).flatten()
        if swap_count > 0:
            generator = torch.Generator(device="cpu")
            if random_seed is not None:
                generator.manual_seed(random_seed)
            order = torch.randperm(candidates.numel(), generator=generator)
            add_idx = candidates[order[:swap_count]]
        else:
            add_idx = candidates[:0]
    else:
        raise ValueError(f"unknown add_order: {add_order}")
    return SwapPlan(
        remove_idx=remove_idx,
        add_idx=add_idx,
        swap_count=swap_count,
        base_only_count=base_only_count,
        imp_only_count=imp_only_count,
        candidate_count=int(candidate_pool.sum().item()),
    )


def random_indices(
    pool: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    candidates = torch.nonzero(pool, as_tuple=False).flatten()
    if count <= 0:
        return candidates[:0]
    if candidates.numel() <= count:
        return candidates
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


def flat_parameter_ids(reference: Mask, names: list[str]) -> torch.Tensor:
    chunks = []
    for group_id, name in enumerate(names):
        chunks.append(torch.full((reference[name].numel(),), group_id, dtype=torch.long))
    return torch.cat(chunks)


def flat_parameter_score_bins(
    flat_scores: torch.Tensor,
    reference: Mask,
    names: list[str],
    *,
    bins: int = 10,
) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("bins must be positive")
    chunks = []
    offset = 0
    for name in names:
        size = reference[name].numel()
        scores = flat_scores[offset : offset + size].detach().float().cpu()
        offset += size
        if size <= 1:
            chunks.append(torch.zeros(size, dtype=torch.long))
            continue
        order = torch.argsort(scores, stable=True)
        ranks = torch.empty(size, dtype=torch.long)
        ranks[order] = torch.arange(size, dtype=torch.long)
        chunks.append(torch.div(ranks * bins, size, rounding_mode="floor").clamp(max=bins - 1))
    return torch.cat(chunks)


def residualize_scores(
    target_scores: torch.Tensor,
    feature_scores: list[torch.Tensor],
    pool: torch.Tensor,
    *,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Remove the linear magnitude-feature subspace from a score vector."""
    residual = torch.zeros_like(target_scores.detach().float().cpu())
    idx = torch.nonzero(pool, as_tuple=False).flatten()
    if idx.numel() == 0 or not feature_scores:
        return residual
    y = target_scores.detach().float().cpu()[idx]
    y_centered = y - y.mean()
    features = []
    for score in feature_scores:
        column = score.detach().float().cpu()[idx]
        std = column.std(unbiased=False)
        if float(std.item()) <= 1e-12:
            features.append(torch.zeros_like(column))
        else:
            features.append((column - column.mean()) / std)
    x = torch.stack(features, dim=1)
    if x.numel() == 0:
        residual[idx] = y_centered
        return residual
    gram = x.t().matmul(x)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    beta = torch.linalg.solve(gram + ridge * eye, x.t().matmul(y_centered))
    residual[idx] = y_centered - x.matmul(beta)
    return residual


def learned_subspace_component_scores(
    feature_scores: list[torch.Tensor],
    pool: torch.Tensor,
    *,
    rank: int,
) -> list[torch.Tensor]:
    """Return top PCA component scores learned inside a candidate pool."""
    if rank <= 0 or not feature_scores:
        return []
    idx = torch.nonzero(pool, as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    standardized_columns = []
    for score in feature_scores:
        column = score.detach().float().cpu()[idx]
        std = column.std(unbiased=False)
        if float(std.item()) <= 1e-12:
            continue
        standardized_columns.append((column - column.mean()) / std)
    if not standardized_columns:
        return []
    x = torch.stack(standardized_columns, dim=1)
    cov = x.t().matmul(x) / max(1, x.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    keep = order[: min(rank, order.numel())]
    components = x.matmul(evecs[:, keep])
    out = []
    template = feature_scores[0].detach().float().cpu()
    for component_idx in range(components.shape[1]):
        full = torch.zeros_like(template)
        full[idx] = components[:, component_idx]
        out.append(full)
    return out


def posterior_score_flats(
    *,
    model_factory,
    dense_state: dict[str, torch.Tensor],
    dense_score_tensors: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    train_size: int,
    names: list[str],
    device: torch.device,
    samples: int,
    scale: float,
    prior_precision: float,
    fisher_batches: int,
    variance_floor: float,
) -> dict[str, torch.Tensor]:
    if samples <= 0:
        raise ValueError("posterior projection Laplace samples must be positive")
    model = model_factory().to(device)
    load_trainable_state(model, dense_state)
    config = DiagonalLaplaceConfig(
        num_samples=samples,
        scale=scale,
        prior_precision=prior_precision,
        fisher_batches=fisher_batches,
        variance_floor=variance_floor,
        num_train_examples=train_size,
    )
    posterior_samples = collect_diag_laplace_samples(model, train_loader, device, config)
    score_tensors: dict[str, dict[str, torch.Tensor]] = {
        "diag_laplace_rms": {},
        "diag_laplace_std": {},
        "diag_laplace_rms_minus_dense_abs": {},
    }
    eps = 1e-12
    for name in names:
        values = torch.stack(
            [sample[name].detach().float().cpu() for sample in posterior_samples],
            dim=0,
        )
        rms = (values.square().mean(dim=0) + eps).sqrt()
        mean = values.mean(dim=0)
        std = ((values - mean).square().mean(dim=0) + eps).sqrt()
        dense_abs = dense_score_tensors[name].detach().float().cpu().abs()
        score_tensors["diag_laplace_rms"][name] = rms
        score_tensors["diag_laplace_std"][name] = std
        score_tensors["diag_laplace_rms_minus_dense_abs"][name] = rms - dense_abs
    return {
        score_name: flatten_scores(tensors, names)
        for score_name, tensors in score_tensors.items()
    }


def make_oracle_overlap_matched_random_plan(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    final_imp_only_pool: torch.Tensor,
    oracle_add_idx: torch.Tensor,
    target_oracle_hits: int,
    alpha: float,
    random_seed: int,
) -> SwapPlan:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    swap_count, base_only_count, imp_only_count = residual_counts(base, imp, names, alpha)
    candidate_pool = final_imp_only_pool & ~base_flat
    swap_count = min(swap_count, base_only_count, int(candidate_pool.sum().item()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)

    oracle_pool = torch.zeros_like(candidate_pool, dtype=torch.bool)
    oracle_pool[oracle_add_idx] = True
    oracle_pool &= candidate_pool
    non_oracle_pool = candidate_pool & ~oracle_pool

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    oracle_count = min(target_oracle_hits, swap_count, int(oracle_pool.sum().item()))
    oracle_idx = random_indices(oracle_pool, oracle_count, generator)
    used = torch.zeros_like(candidate_pool, dtype=torch.bool)
    used[oracle_idx] = True

    non_oracle_count = swap_count - int(oracle_idx.numel())
    non_oracle_idx = random_indices(non_oracle_pool & ~used, non_oracle_count, generator)
    used[non_oracle_idx] = True

    remaining = swap_count - int(oracle_idx.numel()) - int(non_oracle_idx.numel())
    if remaining > 0:
        fill_idx = random_indices(candidate_pool & ~used, remaining, generator)
        add_idx = torch.cat([oracle_idx, non_oracle_idx, fill_idx])
    else:
        add_idx = torch.cat([oracle_idx, non_oracle_idx])

    return SwapPlan(
        remove_idx=remove_idx,
        add_idx=add_idx,
        swap_count=swap_count,
        base_only_count=base_only_count,
        imp_only_count=imp_only_count,
        candidate_count=int(candidate_pool.sum().item()),
    )


def make_group_matched_round_excluded_oracle_plan(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    final_imp_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    final_imp_only_pool: torch.Tensor,
    reference_add_idx: torch.Tensor,
    group_ids: torch.Tensor,
    alpha: float,
) -> SwapPlan:
    base_flat = flatten_mask(base, names)
    base_only = base_flat & ~flatten_mask(imp, names)
    swap_count, base_only_count, imp_only_count = residual_counts(base, imp, names, alpha)
    candidate_pool = final_imp_only_pool & ~base_flat
    swap_count = min(swap_count, base_only_count, int(candidate_pool.sum().item()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)

    selected_pool = torch.zeros_like(candidate_pool, dtype=torch.bool)
    selected_pool[reference_add_idx] = True
    selected_pool &= candidate_pool
    used = torch.zeros_like(candidate_pool, dtype=torch.bool)
    add_parts: list[torch.Tensor] = []

    for group_id in torch.unique(group_ids[reference_add_idx], sorted=True).tolist():
        group_mask = group_ids == int(group_id)
        selected_in_group = selected_pool & group_mask
        target_count = int(selected_in_group.sum().item())
        if target_count == 0:
            continue
        replacement_pool = candidate_pool & group_mask & ~selected_pool & ~used
        replacement_idx = select_indices(
            final_imp_scores,
            replacement_pool,
            target_count,
            largest=True,
        )
        if replacement_idx.numel():
            add_parts.append(replacement_idx)
            used[replacement_idx] = True
        missing = target_count - int(replacement_idx.numel())
        if missing > 0:
            # If a tensor does not have enough non-selected final-IMP residual
            # coordinates, fill from the least round-preferred selected ones.
            fill_idx = select_indices(
                reference_scores,
                selected_in_group & ~used,
                missing,
                largest=False,
            )
            if fill_idx.numel():
                add_parts.append(fill_idx)
                used[fill_idx] = True

    if add_parts:
        add_idx = torch.cat(add_parts)
    else:
        add_idx = reference_add_idx[:0]
    remaining = swap_count - int(add_idx.numel())
    if remaining > 0:
        fill_idx = select_indices(
            final_imp_scores,
            candidate_pool & ~used,
            remaining,
            largest=True,
        )
        if fill_idx.numel():
            add_idx = torch.cat([add_idx, fill_idx])
    return SwapPlan(
        remove_idx=remove_idx,
        add_idx=add_idx[:swap_count],
        swap_count=swap_count,
        base_only_count=base_only_count,
        imp_only_count=imp_only_count,
        candidate_count=int(candidate_pool.sum().item()),
    )


def variant_meta(
    *,
    variant: str,
    process_round: int | None,
    plan: SwapPlan,
    imp_flat: torch.Tensor,
    oracle_add_idx: torch.Tensor,
    control_trial: int | None = None,
) -> dict[str, int | float | str | None]:
    oracle = torch.zeros_like(imp_flat, dtype=torch.bool)
    oracle[oracle_add_idx] = True
    hits = int(imp_flat[plan.add_idx].sum().item())
    oracle_hits = int(oracle[plan.add_idx].sum().item())
    return {
        "variant": variant,
        "process_round": process_round,
        "control_trial": control_trial,
        "swap_count": plan.swap_count,
        "base_only_count": plan.base_only_count,
        "imp_only_count": plan.imp_only_count,
        "candidate_count": plan.candidate_count,
        "added_final_imp_hits": hits,
        "added_final_imp_precision": hits / plan.swap_count if plan.swap_count else None,
        "added_oracle_overlap_hits": oracle_hits,
        "added_oracle_overlap_precision": oracle_hits / plan.swap_count if plan.swap_count else None,
    }


def run_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    trajectory_epochs: list[int],
    process_rounds: list[int],
    round_variants: list[str],
    base_sources: list[str],
    alphas: list[float],
    run_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args, seed)
    model_factory = make_model_factory(args, bundle)
    initial_model = model_factory()
    initial_state = state_to_cpu(initial_model)
    trajectory_model = model_factory().to(device)
    load_trainable_state(trajectory_model, initial_state)
    optimizer = torch.optim.SGD(
        trajectory_model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "constant":
        scheduler = None
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        raise ValueError(f"Unsupported lr_schedule: {args.lr_schedule}")

    states: dict[int, dict[str, torch.Tensor]] = {0: initial_state}
    checkpoint_metrics: dict[int, dict[str, float]] = {
        0: evaluate(trajectory_model, bundle.test_loader, device)
    }
    train_history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(trajectory_model, bundle.train_loader, device, optimizer)
        if scheduler is not None:
            scheduler.step()
        if epoch in trajectory_epochs:
            test_metrics = evaluate(trajectory_model, bundle.test_loader, device)
            checkpoint_metrics[epoch] = test_metrics
            states[epoch] = state_to_cpu(trajectory_model)
            row = {
                "seed": seed,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
            print(json.dumps(row), flush=True)
            train_history.append(row)

    rewind_state = states[args.rewind_epochs] if args.rewind_epochs > 0 else None
    dense_state = states[args.epochs]
    train_state = initial_state if rewind_state is None else rewind_state
    imp_epochs = args.epochs if args.imp_epochs is None else args.imp_epochs
    mask_train_epochs = imp_epochs if args.mask_train_epochs is None else args.mask_train_epochs
    imp = iterative_magnitude_pruning_process_trace(
        model_factory=model_factory,
        initial_state=initial_state,
        train_loader=bundle.train_loader,
        test_loader=bundle.test_loader,
        device=device,
        rounds=args.imp_rounds,
        prune_fraction_per_round=args.prune_fraction,
        epochs_per_round=imp_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        rewind_state=rewind_state,
        final_epochs=args.imp_final_epochs,
    )

    names = weight_parameter_names(model_factory())
    checkpoint_scores = {
        f"epoch_{epoch}": {name: states[epoch][name].detach().abs().cpu() for name in names}
        for epoch in trajectory_epochs
    }
    trajectory_scores = trajectory_score_tensors(
        states=states,
        names=names,
        trajectory_epochs=trajectory_epochs,
        rewind_epochs=args.rewind_epochs,
    )
    source_scores = {**checkpoint_scores, **trajectory_scores}
    source_masks = {
        source: global_score_mask(score, names, sparsity=imp.metrics["sparsity"], largest=True)
        for source, score in source_scores.items()
    }
    unknown_sources = sorted(set(base_sources) - set(source_masks))
    if unknown_sources:
        raise ValueError(f"unknown base sources {unknown_sources}; expected one of {sorted(source_masks)}")

    dense_mask = global_magnitude_mask_from_state(dense_state, names, imp.metrics["sparsity"])
    rewind_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else None
    )
    imp_flat = flatten_mask(imp.mask, names)
    final_imp_scores = flatten_scores(
        {name: imp.final_state[name].detach().abs().cpu() for name in names},
        names,
    )
    dense_score_tensors = {name: dense_state[name].detach().abs().cpu() for name in names}
    dense_scores = flatten_scores(
        dense_score_tensors,
        names,
    )
    needs_posterior_projection = (
        "final-imp-round-posterior-residualized-score" in round_variants
    )
    posterior_projection_scores: dict[str, torch.Tensor] = {}
    if needs_posterior_projection:
        posterior_projection_scores = posterior_score_flats(
            model_factory=model_factory,
            dense_state=dense_state,
            dense_score_tensors=dense_score_tensors,
            train_loader=bundle.train_loader,
            train_size=bundle.train_size,
            names=names,
            device=device,
            samples=args.posterior_projection_laplace_samples,
            scale=args.posterior_projection_laplace_scale,
            prior_precision=args.posterior_projection_laplace_prior_precision,
            fisher_batches=args.posterior_projection_laplace_fisher_batches,
            variance_floor=args.posterior_projection_laplace_variance_floor,
        )
    round_masks_flat = [flatten_mask(mask, names) for mask in imp.round_masks]
    round_scores = [
        flatten_scores({name: state[name].detach().abs().cpu() for name in names}, names)
        for state in imp.round_states
    ]
    trajectory_feature_flats = [
        flatten_scores(source_scores[source], names)
        for source in sorted(source_scores)
    ]
    group_ids = flat_parameter_ids(imp.mask, names)
    rows = []
    train_cache: dict[str, dict[str, float]] = {}

    def evaluate_candidate(
        *,
        base_source: str,
        alpha: float,
        mask: Mask,
        meta: dict[str, int | float | str | None],
    ) -> None:
        fingerprint = mask_fingerprint(mask, names)
        cache_hit = fingerprint in train_cache
        if not cache_hit:
            train_cache[fingerprint] = train_fixed_mask(
                model_factory=model_factory,
                train_state=train_state,
                mask=mask,
                train_loader=bundle.train_loader,
                test_loader=bundle.test_loader,
                device=device,
                epochs=mask_train_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_schedule=args.lr_schedule,
            )
        metrics = train_cache[fingerprint]
        base_mask = source_masks[base_source]
        out: dict[str, Any] = {
            "seed": seed,
            "base_source": base_source,
            "alpha": alpha,
            "mask_train_epochs": mask_train_epochs,
            "trained_loss": metrics["loss"],
            "trained_accuracy": metrics["accuracy"],
            "mask_sparsity": metrics["sparsity"],
            "mask_to_base_jaccard": support_jaccard(mask, base_mask),
            "mask_to_imp_jaccard": support_jaccard(mask, imp.mask),
            "mask_to_dense_final_magnitude_jaccard": support_jaccard(mask, dense_mask),
            "mask_to_rewind_magnitude_jaccard": (
                support_jaccard(mask, rewind_mask) if rewind_mask is not None else None
            ),
            "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
            "imp_accuracy": imp.metrics["accuracy"],
            "accuracy_minus_imp": metrics["accuracy"] - imp.metrics["accuracy"],
            "accuracy_minus_dense": metrics["accuracy"] - checkpoint_metrics[args.epochs]["accuracy"],
            "cache_hit": cache_hit,
        }
        out.update(meta)
        rows.append(out)
        print(json.dumps(out), flush=True)

    for base_source in base_sources:
        base_mask = source_masks[base_source]
        base_flat = flatten_mask(base_mask, names)
        base_scores = flatten_scores(source_scores[base_source], names)
        _, base_only_count, imp_only_count = residual_counts(base_mask, imp.mask, names, 0.0)
        evaluate_candidate(
            base_source=base_source,
            alpha=0.0,
            mask=base_mask,
            meta={
                "variant": "base",
                "process_round": None,
                "control_trial": None,
                "swap_count": 0,
                "base_only_count": base_only_count,
                "imp_only_count": imp_only_count,
                "candidate_count": None,
                "added_final_imp_hits": 0,
                "added_final_imp_precision": None,
                "added_oracle_overlap_hits": 0,
                "added_oracle_overlap_precision": None,
            },
        )
        final_imp_only_pool = imp_flat & ~base_flat
        for alpha in alphas:
            oracle_plan = make_swap_plan(
                base=base_mask,
                imp=imp.mask,
                names=names,
                base_scores=base_scores,
                add_scores=final_imp_scores,
                candidate_pool=final_imp_only_pool,
                alpha=alpha,
            )
            if oracle_plan.swap_count == 0:
                continue
            oracle_mask = mask_from_swaps(base_mask, names, oracle_plan.remove_idx, oracle_plan.add_idx)
            evaluate_candidate(
                base_source=base_source,
                alpha=alpha,
                mask=oracle_mask,
                meta=variant_meta(
                    variant="final_oracle_residual",
                    process_round=None,
                    plan=oracle_plan,
                    imp_flat=imp_flat,
                    oracle_add_idx=oracle_plan.add_idx,
                    control_trial=None,
                ),
            )
            if "final-imp-dense-score" in round_variants:
                dense_score_plan = make_swap_plan(
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    base_scores=base_scores,
                    add_scores=dense_scores,
                    candidate_pool=final_imp_only_pool,
                    alpha=alpha,
                )
                dense_score_mask = mask_from_swaps(
                    base_mask,
                    names,
                    dense_score_plan.remove_idx,
                    dense_score_plan.add_idx,
                )
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=dense_score_mask,
                    meta=variant_meta(
                        variant="dense_score_final_imp_residual",
                        process_round=None,
                        plan=dense_score_plan,
                        imp_flat=imp_flat,
                        oracle_add_idx=oracle_plan.add_idx,
                        control_trial=None,
                    ),
                )
            if "final-imp-base-score" in round_variants:
                base_score_plan = make_swap_plan(
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    base_scores=base_scores,
                    add_scores=base_scores,
                    candidate_pool=final_imp_only_pool,
                    alpha=alpha,
                )
                base_score_mask = mask_from_swaps(
                    base_mask,
                    names,
                    base_score_plan.remove_idx,
                    base_score_plan.add_idx,
                )
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=base_score_mask,
                    meta=variant_meta(
                        variant="base_score_final_imp_residual",
                        process_round=None,
                        plan=base_score_plan,
                        imp_flat=imp_flat,
                        oracle_add_idx=oracle_plan.add_idx,
                        control_trial=None,
                    ),
                )
            for process_round in process_rounds:
                round_idx = process_round - 1
                round_score = round_scores[round_idx]
                if "survivor" in round_variants:
                    survivor_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=round_masks_flat[round_idx] & ~base_flat,
                        alpha=alpha,
                    )
                    survivor_mask = mask_from_swaps(
                        base_mask,
                        names,
                        survivor_plan.remove_idx,
                        survivor_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=survivor_mask,
                        meta=variant_meta(
                            variant="round_survivor_residual",
                            process_round=process_round,
                            plan=survivor_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "survivor-low" in round_variants:
                    low_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=round_masks_flat[round_idx] & ~base_flat,
                        alpha=alpha,
                        add_order="low",
                    )
                    low_mask = mask_from_swaps(
                        base_mask,
                        names,
                        low_plan.remove_idx,
                        low_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=low_mask,
                        meta=variant_meta(
                            variant="round_survivor_low_residual",
                            process_round=process_round,
                            plan=low_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "survivor-random" in round_variants:
                    for trial in range(args.random_trials):
                        random_plan = make_swap_plan(
                            base=base_mask,
                            imp=imp.mask,
                            names=names,
                            base_scores=base_scores,
                            add_scores=round_score,
                            candidate_pool=round_masks_flat[round_idx] & ~base_flat,
                            alpha=alpha,
                            add_order="random",
                            random_seed=(
                                seed * 1000003
                                + process_round * 10007
                                + trial * 101
                                + int(round(alpha * 1000))
                            ),
                        )
                        random_mask = mask_from_swaps(
                            base_mask,
                            names,
                            random_plan.remove_idx,
                            random_plan.add_idx,
                        )
                        evaluate_candidate(
                            base_source=base_source,
                            alpha=alpha,
                            mask=random_mask,
                            meta=variant_meta(
                                variant="round_survivor_random_residual",
                                process_round=process_round,
                                plan=random_plan,
                                imp_flat=imp_flat,
                                oracle_add_idx=oracle_plan.add_idx,
                                control_trial=trial,
                            ),
                        )
                if "final-imp" in round_variants:
                    final_imp_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    final_imp_mask = mask_from_swaps(
                        base_mask,
                        names,
                        final_imp_plan.remove_idx,
                        final_imp_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=final_imp_mask,
                        meta=variant_meta(
                            variant="round_final_imp_residual",
                            process_round=process_round,
                            plan=final_imp_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-round-residualized-score" in round_variants:
                    residualized_round_score = residualize_scores(
                        round_score,
                        [base_scores, dense_scores, final_imp_scores],
                        final_imp_only_pool & ~base_flat,
                    )
                    residualized_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=residualized_round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    residualized_mask = mask_from_swaps(
                        base_mask,
                        names,
                        residualized_plan.remove_idx,
                        residualized_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=residualized_mask,
                        meta=variant_meta(
                            variant="round_final_imp_residualized_score_residual",
                            process_round=process_round,
                            plan=residualized_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-round-posterior-residualized-score" in round_variants:
                    posterior_residualized_round_score = residualize_scores(
                        round_score,
                        [
                            base_scores,
                            dense_scores,
                            final_imp_scores,
                            posterior_projection_scores["diag_laplace_rms"],
                            posterior_projection_scores["diag_laplace_std"],
                            posterior_projection_scores[
                                "diag_laplace_rms_minus_dense_abs"
                            ],
                        ],
                        final_imp_only_pool & ~base_flat,
                    )
                    posterior_residualized_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=posterior_residualized_round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    posterior_residualized_mask = mask_from_swaps(
                        base_mask,
                        names,
                        posterior_residualized_plan.remove_idx,
                        posterior_residualized_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=posterior_residualized_mask,
                        meta=variant_meta(
                            variant=(
                                "round_final_imp_posterior_residualized_score_residual"
                            ),
                            process_round=process_round,
                            plan=posterior_residualized_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-round-learned-subspace-residualized-score" in round_variants:
                    learned_features = learned_subspace_component_scores(
                        (
                            trajectory_feature_flats
                            + [final_imp_scores]
                            + round_scores[:round_idx]
                        ),
                        final_imp_only_pool & ~base_flat,
                        rank=args.learned_subspace_rank,
                    )
                    learned_residualized_round_score = residualize_scores(
                        round_score,
                        learned_features,
                        final_imp_only_pool & ~base_flat,
                    )
                    learned_residualized_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=learned_residualized_round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    learned_residualized_mask = mask_from_swaps(
                        base_mask,
                        names,
                        learned_residualized_plan.remove_idx,
                        learned_residualized_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=learned_residualized_mask,
                        meta=variant_meta(
                            variant=(
                                "round_final_imp_learned_subspace_residualized_score_residual"
                            ),
                            process_round=process_round,
                            plan=learned_residualized_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-oracle-matched-random" in round_variants:
                    reference_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    oracle = torch.zeros_like(imp_flat, dtype=torch.bool)
                    oracle[oracle_plan.add_idx] = True
                    target_oracle_hits = int(oracle[reference_plan.add_idx].sum().item())
                    for trial in range(args.random_trials):
                        matched_plan = make_oracle_overlap_matched_random_plan(
                            base=base_mask,
                            imp=imp.mask,
                            names=names,
                            base_scores=base_scores,
                            final_imp_only_pool=final_imp_only_pool,
                            oracle_add_idx=oracle_plan.add_idx,
                            target_oracle_hits=target_oracle_hits,
                            alpha=alpha,
                            random_seed=(
                                seed * 1000003
                                + process_round * 10007
                                + trial * 101
                                + int(round(alpha * 1000))
                                + 97
                            ),
                        )
                        matched_mask = mask_from_swaps(
                            base_mask,
                            names,
                            matched_plan.remove_idx,
                            matched_plan.add_idx,
                        )
                        evaluate_candidate(
                            base_source=base_source,
                            alpha=alpha,
                            mask=matched_mask,
                            meta=variant_meta(
                                variant=(
                                    "round_final_imp_oracle_matched_random_residual"
                                ),
                                process_round=process_round,
                                plan=matched_plan,
                                imp_flat=imp_flat,
                                oracle_add_idx=oracle_plan.add_idx,
                                control_trial=trial,
                            ),
                        )
                if "final-imp-round-excluded-oracle" in round_variants:
                    reference_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    excluded_pool = final_imp_only_pool.detach().clone()
                    excluded_pool[reference_plan.add_idx] = False
                    excluded_candidate_count = int((excluded_pool & ~base_flat).sum().item())
                    if excluded_candidate_count < reference_plan.swap_count:
                        selected_pool = torch.zeros_like(final_imp_only_pool, dtype=torch.bool)
                        selected_pool[reference_plan.add_idx] = True
                        fill_count = reference_plan.swap_count - excluded_candidate_count
                        fill_idx = select_indices(
                            round_score,
                            selected_pool,
                            fill_count,
                            largest=False,
                        )
                        excluded_pool[fill_idx] = True
                    excluded_oracle_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=final_imp_scores,
                        candidate_pool=excluded_pool,
                        alpha=alpha,
                    )
                    excluded_oracle_mask = mask_from_swaps(
                        base_mask,
                        names,
                        excluded_oracle_plan.remove_idx,
                        excluded_oracle_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=excluded_oracle_mask,
                        meta=variant_meta(
                            variant="round_excluded_oracle_final_imp_residual",
                            process_round=process_round,
                            plan=excluded_oracle_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-round-excluded-layer-oracle" in round_variants:
                    reference_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    layer_oracle_plan = make_group_matched_round_excluded_oracle_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        final_imp_scores=final_imp_scores,
                        reference_scores=round_score,
                        final_imp_only_pool=final_imp_only_pool,
                        reference_add_idx=reference_plan.add_idx,
                        group_ids=group_ids,
                        alpha=alpha,
                    )
                    layer_oracle_mask = mask_from_swaps(
                        base_mask,
                        names,
                        layer_oracle_plan.remove_idx,
                        layer_oracle_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=layer_oracle_mask,
                        meta=variant_meta(
                            variant="round_excluded_layer_oracle_final_imp_residual",
                            process_round=process_round,
                            plan=layer_oracle_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )
                if "final-imp-round-excluded-tensor-score-oracle" in round_variants:
                    reference_plan = make_swap_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        add_scores=round_score,
                        candidate_pool=final_imp_only_pool,
                        alpha=alpha,
                    )
                    round_score_bins = flat_parameter_score_bins(
                        round_score,
                        imp.mask,
                        names,
                        bins=10,
                    )
                    tensor_score_ids = group_ids * 10 + round_score_bins
                    tensor_score_oracle_plan = make_group_matched_round_excluded_oracle_plan(
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        base_scores=base_scores,
                        final_imp_scores=final_imp_scores,
                        reference_scores=round_score,
                        final_imp_only_pool=final_imp_only_pool,
                        reference_add_idx=reference_plan.add_idx,
                        group_ids=tensor_score_ids,
                        alpha=alpha,
                    )
                    tensor_score_oracle_mask = mask_from_swaps(
                        base_mask,
                        names,
                        tensor_score_oracle_plan.remove_idx,
                        tensor_score_oracle_plan.add_idx,
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        alpha=alpha,
                        mask=tensor_score_oracle_mask,
                        meta=variant_meta(
                            variant="round_excluded_tensor_score_oracle_final_imp_residual",
                            process_round=process_round,
                            plan=tensor_score_oracle_plan,
                            imp_flat=imp_flat,
                            oracle_add_idx=oracle_plan.add_idx,
                            control_trial=None,
                        ),
                    )

    payload = {
        "seed": seed,
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "trajectory_epochs": trajectory_epochs,
            "rewind_epochs": args.rewind_epochs,
            "imp_epochs": imp_epochs,
            "imp_final_epochs": args.imp_final_epochs,
            "mask_train_epochs": mask_train_epochs,
            "imp_rounds": args.imp_rounds,
            "process_rounds": process_rounds,
            "round_variants": round_variants,
            "random_trials": args.random_trials,
            "prune_fraction": args.prune_fraction,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "augment": args.augment,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
            "base_sources": base_sources,
            "alphas": alphas,
            "posterior_projection_laplace_samples": (
                args.posterior_projection_laplace_samples
            ),
            "posterior_projection_laplace_scale": (
                args.posterior_projection_laplace_scale
            ),
            "posterior_projection_laplace_prior_precision": (
                args.posterior_projection_laplace_prior_precision
            ),
            "posterior_projection_laplace_fisher_batches": (
                args.posterior_projection_laplace_fisher_batches
            ),
            "posterior_projection_laplace_variance_floor": (
                args.posterior_projection_laplace_variance_floor
            ),
            "learned_subspace_rank": args.learned_subspace_rank,
        },
        "dense": checkpoint_metrics[args.epochs],
        "imp": imp.metrics,
        "imp_history": imp.history,
        "rows": rows,
        "train_history": train_history,
    }
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    with (seed_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (seed_dir / "residual_imp_process_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "seed": seed,
                "dataset": args.dataset,
                "model": args.model,
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "num_rows": len(rows),
                "unique_trained_masks": len(train_cache),
                "best_candidate": max(rows, key=lambda row: float(row["trained_accuracy"])),
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    alphas = parse_float_list(args.alphas)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    base_sources = parse_source_list(args.base_sources)
    process_rounds = parse_int_list(args.process_rounds)
    if any(round_idx < 1 or round_idx > args.imp_rounds for round_idx in process_rounds):
        raise ValueError("process rounds must be within [1, imp_rounds]")
    round_variants = parse_source_list(args.round_variants)
    valid_variants = {
        "survivor",
        "survivor-random",
        "survivor-low",
        "final-imp",
        "final-imp-dense-score",
        "final-imp-base-score",
        "final-imp-round-residualized-score",
        "final-imp-round-posterior-residualized-score",
        "final-imp-round-learned-subspace-residualized-score",
        "final-imp-oracle-matched-random",
        "final-imp-round-excluded-oracle",
        "final-imp-round-excluded-layer-oracle",
        "final-imp-round-excluded-tensor-score-oracle",
    }
    unknown_variants = sorted(set(round_variants) - valid_variants)
    if unknown_variants:
        raise ValueError(f"unknown round variants: {unknown_variants}")
    if args.random_trials < 1:
        raise ValueError("random_trials must be positive")
    trajectory_epochs = sorted(set(parse_int_list(args.trajectory_epochs) + [0, args.epochs]))
    if args.rewind_epochs not in trajectory_epochs:
        trajectory_epochs = sorted(set(trajectory_epochs + [args.rewind_epochs]))
    if any(epoch < 0 or epoch > args.epochs for epoch in trajectory_epochs):
        raise ValueError("trajectory epochs must be in [0, epochs]")
    if args.rewind_epochs < 0 or args.rewind_epochs > args.epochs:
        raise ValueError("rewind_epochs must be in [0, epochs]")

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    payloads = []
    for seed in seeds:
        payloads.append(
            run_seed(
                args=args,
                seed=seed,
                trajectory_epochs=trajectory_epochs,
                process_rounds=process_rounds,
                round_variants=round_variants,
                base_sources=base_sources,
                alphas=alphas,
                run_dir=run_dir,
            )
        )
    print(
        json.dumps(
            {
                "seeds": seeds,
                "dataset": args.dataset,
                "model": args.model,
                "run_dir": str(run_dir),
                "num_payloads": len(payloads),
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
