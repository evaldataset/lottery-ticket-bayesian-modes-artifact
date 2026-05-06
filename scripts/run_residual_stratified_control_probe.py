#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import Mask, global_magnitude_mask_from_state, global_score_mask, mask_sparsity, support_jaccard
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu
from run_residual_predictor_mask_probe import (
    flatten_mask,
    flatten_scores,
    mask_fingerprint,
    mask_from_swaps,
    percentile_by_parameter,
    select_indices,
    train_fixed_mask,
    train_one_epoch,
    trajectory_score_tensors,
)


@dataclass
class SwapPlan:
    remove_idx: torch.Tensor
    add_idx: torch.Tensor
    swap_count: int
    base_only_count: int
    imp_only_count: int


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
    parser.add_argument("--random-trials", type=int, default=1)
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_stratified_control_probe"))
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


def parameter_ids(names: list[str], reference: Mask) -> torch.Tensor:
    chunks = []
    for idx, name in enumerate(names):
        chunks.append(torch.full((reference[name].numel(),), idx, dtype=torch.long))
    return torch.cat(chunks)


def score_bins(scores: dict[str, torch.Tensor], names: list[str], bins: int) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("score_bins must be positive")
    percentile = percentile_by_parameter(scores, names)
    return torch.clamp((percentile * bins).long(), max=bins - 1)


def sample_indices(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    pool_idx = pool.nonzero(as_tuple=False).flatten()
    if count <= 0:
        return pool_idx[:0]
    if pool_idx.numel() <= count:
        return pool_idx
    perm = torch.randperm(pool_idx.numel(), generator=generator)
    return pool_idx[perm[:count]]


def sample_with_strata(
    *,
    target_idx: torch.Tensor,
    pool: torch.Tensor,
    primary: torch.Tensor,
    secondary: torch.Tensor | None,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int | float | None]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    selected_parts = []
    used = torch.zeros_like(pool, dtype=torch.bool)
    exact_count = 0
    primary_relaxed_count = 0
    global_relaxed_count = 0

    target_primary = primary[target_idx]
    target_secondary = secondary[target_idx] if secondary is not None else None
    if secondary is None:
        keys = Counter(int(value.item()) for value in target_primary)
        for primary_key, count in sorted(keys.items()):
            local_pool = pool & ~used & (primary == primary_key)
            chosen = sample_indices(local_pool, count, generator)
            exact_count += int(chosen.numel())
            selected_parts.append(chosen)
            used[chosen] = True
            missing = count - int(chosen.numel())
            if missing > 0:
                global_pool = pool & ~used
                relaxed = sample_indices(global_pool, missing, generator)
                global_relaxed_count += int(relaxed.numel())
                selected_parts.append(relaxed)
                used[relaxed] = True
    else:
        keys = Counter(
            (int(p.item()), int(s.item()))
            for p, s in zip(target_primary, target_secondary, strict=True)
        )
        for (primary_key, secondary_key), count in sorted(keys.items()):
            exact_pool = pool & ~used & (primary == primary_key) & (secondary == secondary_key)
            chosen = sample_indices(exact_pool, count, generator)
            exact_count += int(chosen.numel())
            selected_parts.append(chosen)
            used[chosen] = True
            missing = count - int(chosen.numel())
            if missing > 0:
                primary_pool = pool & ~used & (primary == primary_key)
                relaxed = sample_indices(primary_pool, missing, generator)
                primary_relaxed_count += int(relaxed.numel())
                selected_parts.append(relaxed)
                used[relaxed] = True
                missing -= int(relaxed.numel())
            if missing > 0:
                global_pool = pool & ~used
                relaxed = sample_indices(global_pool, missing, generator)
                global_relaxed_count += int(relaxed.numel())
                selected_parts.append(relaxed)
                used[relaxed] = True

    selected = (
        torch.cat([part for part in selected_parts if part.numel() > 0])
        if selected_parts
        else target_idx[:0]
    )
    if selected.numel() > target_idx.numel():
        selected = selected[: target_idx.numel()]
    return selected, {
        "stratum_target_count": int(target_idx.numel()),
        "stratum_exact_count": exact_count,
        "stratum_primary_relaxed_count": primary_relaxed_count,
        "stratum_global_relaxed_count": global_relaxed_count,
        "stratum_exact_fraction": exact_count / int(target_idx.numel()) if target_idx.numel() else None,
    }


def residual_swap_count(base: Mask, target: Mask, names: list[str], alpha: float) -> tuple[int, int, int]:
    base_flat = flatten_mask(base, names)
    target_flat = flatten_mask(target, names)
    base_only = base_flat & ~target_flat
    target_only = target_flat & ~base_flat
    swappable = min(int(base_only.sum().item()), int(target_only.sum().item()))
    return int(round(alpha * swappable)), int(base_only.sum().item()), int(target_only.sum().item())


def oracle_swap_plan(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    imp_scores: torch.Tensor,
    alpha: float,
) -> SwapPlan:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count, imp_only_count)
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    add_idx = select_indices(imp_scores, imp_only, swap_count, largest=True)
    return SwapPlan(
        remove_idx=remove_idx,
        add_idx=add_idx,
        swap_count=swap_count,
        base_only_count=base_only_count,
        imp_only_count=imp_only_count,
    )


def variant_meta(
    *,
    variant: str,
    trial: int | None,
    plan: SwapPlan,
    base: Mask,
    imp: Mask,
    names: list[str],
    add_idx: torch.Tensor,
    oracle_add_idx: torch.Tensor,
    extra: dict[str, int | float | None] | None = None,
) -> dict[str, int | float | str | None]:
    imp_flat = flatten_mask(imp, names)
    oracle_added = torch.zeros_like(imp_flat, dtype=torch.bool)
    oracle_added[oracle_add_idx] = True
    hits = int(imp_flat[add_idx].sum().item())
    oracle_overlap = int(oracle_added[add_idx].sum().item())
    out: dict[str, int | float | str | None] = {
        "variant": variant,
        "trial": trial,
        "swap_count": plan.swap_count,
        "base_only_count": plan.base_only_count,
        "imp_only_count": plan.imp_only_count,
        "added_imp_only_hits": hits,
        "added_imp_only_precision": hits / int(add_idx.numel()) if add_idx.numel() else None,
        "added_oracle_overlap_hits": oracle_overlap,
        "added_oracle_overlap_precision": (
            oracle_overlap / int(add_idx.numel()) if add_idx.numel() else None
        ),
        "stratum_target_count": None,
        "stratum_exact_count": None,
        "stratum_primary_relaxed_count": None,
        "stratum_global_relaxed_count": None,
        "stratum_exact_fraction": None,
    }
    if extra:
        out.update(extra)
    return out


def run_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    trajectory_epochs: list[int],
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
    imp = iterative_magnitude_pruning(
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
    imp_scores = flatten_scores({name: imp.final_state[name].detach().abs().cpu() for name in names}, names)
    param_id = parameter_ids(names, imp.mask)
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
        imp_flat = flatten_mask(imp.mask, names)
        nonimp_pool = (~base_flat) & (~imp_flat)
        imp_only_pool = (~base_flat) & imp_flat
        base_score_flat = flatten_scores(source_scores[base_source], names)
        source_bins = score_bins(source_scores[base_source], names, args.score_bins)
        _, base_only_count, imp_only_count = residual_swap_count(base_mask, imp.mask, names, 0.0)
        evaluate_candidate(
            base_source=base_source,
            alpha=0.0,
            mask=base_mask,
            meta={
                "variant": "base",
                "trial": None,
                "swap_count": 0,
                "base_only_count": base_only_count,
                "imp_only_count": imp_only_count,
                "added_imp_only_hits": 0,
                "added_imp_only_precision": None,
                "added_oracle_overlap_hits": 0,
                "added_oracle_overlap_precision": None,
                "stratum_target_count": None,
                "stratum_exact_count": None,
                "stratum_primary_relaxed_count": None,
                "stratum_global_relaxed_count": None,
                "stratum_exact_fraction": None,
            },
        )
        for alpha in alphas:
            plan = oracle_swap_plan(
                base=base_mask,
                imp=imp.mask,
                names=names,
                base_scores=base_score_flat,
                imp_scores=imp_scores,
                alpha=alpha,
            )
            if plan.swap_count == 0:
                continue

            oracle_mask = mask_from_swaps(base_mask, names, plan.remove_idx, plan.add_idx)
            evaluate_candidate(
                base_source=base_source,
                alpha=alpha,
                mask=oracle_mask,
                meta=variant_meta(
                    variant="oracle_top_imp_residual",
                    trial=None,
                    plan=plan,
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    add_idx=plan.add_idx,
                    oracle_add_idx=plan.add_idx,
                ),
            )
            for trial in range(args.random_trials):
                seed_offset = seed * 1000003 + trial * 1009 + int(round(alpha * 1000))

                random_imp_idx = sample_indices(imp_only_pool, plan.swap_count, torch.Generator().manual_seed(seed_offset + 11))
                random_imp_mask = mask_from_swaps(base_mask, names, plan.remove_idx, random_imp_idx)
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=random_imp_mask,
                    meta=variant_meta(
                        variant="random_imp_only_residual",
                        trial=trial,
                        plan=plan,
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        add_idx=random_imp_idx,
                        oracle_add_idx=plan.add_idx,
                    ),
                )

                random_nonimp_idx = sample_indices(
                    nonimp_pool,
                    plan.swap_count,
                    torch.Generator().manual_seed(seed_offset + 23),
                )
                random_nonimp_mask = mask_from_swaps(base_mask, names, plan.remove_idx, random_nonimp_idx)
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=random_nonimp_mask,
                    meta=variant_meta(
                        variant="random_nonimp_global_residual",
                        trial=trial,
                        plan=plan,
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        add_idx=random_nonimp_idx,
                        oracle_add_idx=plan.add_idx,
                    ),
                )

                param_idx, param_meta = sample_with_strata(
                    target_idx=plan.add_idx,
                    pool=nonimp_pool,
                    primary=param_id,
                    secondary=None,
                    seed=seed_offset + 37,
                )
                param_mask = mask_from_swaps(base_mask, names, plan.remove_idx, param_idx)
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=param_mask,
                    meta=variant_meta(
                        variant="random_nonimp_parameter_matched_residual",
                        trial=trial,
                        plan=plan,
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        add_idx=param_idx,
                        oracle_add_idx=plan.add_idx,
                        extra=param_meta,
                    ),
                )

                param_score_idx, param_score_meta = sample_with_strata(
                    target_idx=plan.add_idx,
                    pool=nonimp_pool,
                    primary=param_id,
                    secondary=source_bins,
                    seed=seed_offset + 41,
                )
                param_score_mask = mask_from_swaps(base_mask, names, plan.remove_idx, param_score_idx)
                evaluate_candidate(
                    base_source=base_source,
                    alpha=alpha,
                    mask=param_score_mask,
                    meta=variant_meta(
                        variant="random_nonimp_parameter_score_matched_residual",
                        trial=trial,
                        plan=plan,
                        base=base_mask,
                        imp=imp.mask,
                        names=names,
                        add_idx=param_score_idx,
                        oracle_add_idx=plan.add_idx,
                        extra=param_score_meta,
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
            "random_trials": args.random_trials,
            "score_bins": args.score_bins,
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
    with (seed_dir / "residual_stratified_control_probe.csv").open("w", newline="", encoding="utf-8") as f:
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
    if args.random_trials < 0:
        raise ValueError("random_trials must be non-negative")
    if args.score_bins <= 0:
        raise ValueError("score_bins must be positive")
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    alphas = parse_float_list(args.alphas)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    base_sources = parse_source_list(args.base_sources)
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
