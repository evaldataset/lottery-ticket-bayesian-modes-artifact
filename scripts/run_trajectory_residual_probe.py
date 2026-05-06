#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import (
    Mask,
    apply_mask_,
    global_magnitude_mask_from_state,
    global_score_mask,
    mask_sparsity,
    support_jaccard,
)
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu, train_model


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_source_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--base-sources",
        default="epoch_30,traj_rms_abs,epoch_10",
        help="Comma-separated checkpoint or aggregate trajectory masks to interpolate toward IMP.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.5,1.0",
        help="Fraction of the base-vs-IMP residual support to swap.",
    )
    parser.add_argument(
        "--imp-remove-orders",
        default="low",
        help=(
            "Comma-separated base-only removal orders for IMP-residual additions: "
            "low,random,high. The default preserves the original probe."
        ),
    )
    parser.add_argument("--random-residual-trials", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_residual_probe"))
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return {"loss": total_loss / total, "accuracy": correct / total}


def trajectory_score_tensors(
    states: dict[int, dict[str, torch.Tensor]],
    names: list[str],
    trajectory_epochs: list[int],
    rewind_epochs: int,
) -> dict[str, dict[str, torch.Tensor]]:
    ordered_epochs = sorted(trajectory_epochs)
    state_sequence = [states[epoch] for epoch in ordered_epochs]
    rewind_state = states[rewind_epochs]
    initial_state = states[0]
    post_rewind_sequence = [states[epoch] for epoch in ordered_epochs if epoch >= rewind_epochs]
    if len(post_rewind_sequence) < 2:
        post_rewind_sequence = state_sequence

    scores: dict[str, dict[str, torch.Tensor]] = {
        "traj_mean_abs": {},
        "traj_rms_abs": {},
        "traj_max_abs": {},
        "traj_initial_rms_movement": {},
        "traj_rewind_rms_movement": {},
        "traj_path_length": {},
        "traj_post_rewind_path_length": {},
    }
    for name in names:
        stacked = torch.stack([state[name].detach().cpu() for state in state_sequence])
        scores["traj_mean_abs"][name] = stacked.abs().mean(dim=0)
        scores["traj_rms_abs"][name] = stacked.square().mean(dim=0).sqrt()
        scores["traj_max_abs"][name] = stacked.abs().amax(dim=0)
        scores["traj_initial_rms_movement"][name] = (
            torch.stack(
                [
                    (state[name].detach().cpu() - initial_state[name].detach().cpu()).square()
                    for state in state_sequence
                ]
            )
            .mean(dim=0)
            .sqrt()
        )
        scores["traj_rewind_rms_movement"][name] = (
            torch.stack(
                [
                    (state[name].detach().cpu() - rewind_state[name].detach().cpu()).square()
                    for state in post_rewind_sequence
                ]
            )
            .mean(dim=0)
            .sqrt()
        )
        path_score = torch.zeros_like(stacked[0])
        for prev, curr in zip(state_sequence, state_sequence[1:], strict=False):
            path_score = path_score + (
                curr[name].detach().cpu() - prev[name].detach().cpu()
            ).abs()
        scores["traj_path_length"][name] = path_score
        post_path_score = torch.zeros_like(stacked[0])
        for prev, curr in zip(post_rewind_sequence, post_rewind_sequence[1:], strict=False):
            post_path_score = post_path_score + (
                curr[name].detach().cpu() - prev[name].detach().cpu()
            ).abs()
        scores["traj_post_rewind_path_length"][name] = post_path_score
    return scores


def train_fixed_mask(
    model_factory,
    train_state: dict[str, torch.Tensor],
    mask: Mask,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str,
) -> dict[str, float]:
    model = model_factory()
    load_trainable_state(model, train_state)
    apply_mask_(model, mask)
    train_model(
        model,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        mask=mask,
        lr_schedule=lr_schedule,
    )
    metrics = evaluate(model, test_loader, device)
    metrics["sparsity"] = mask_sparsity(mask)
    return metrics


def flatten_mask(mask: Mask, names: list[str]) -> torch.Tensor:
    return torch.cat([mask[name].bool().flatten().cpu() for name in names])


def flatten_scores(scores: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([scores[name].detach().float().flatten().cpu() for name in names])


def unflatten_mask(flat: torch.Tensor, names: list[str], reference: Mask) -> Mask:
    out: Mask = {}
    offset = 0
    for name in names:
        size = reference[name].numel()
        out[name] = flat[offset : offset + size].reshape(reference[name].shape).clone()
        offset += size
    return out


def select_indices(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    k: int,
    *,
    largest: bool,
) -> torch.Tensor:
    candidate_idx = candidates.nonzero(as_tuple=False).flatten()
    if k <= 0:
        return candidate_idx[:0]
    if candidate_idx.numel() <= k:
        return candidate_idx
    selected = torch.topk(scores[candidate_idx], k, largest=largest).indices
    return candidate_idx[selected]


def select_removal_indices(
    *,
    base_scores: torch.Tensor,
    base_only: torch.Tensor,
    k: int,
    remove_order: str,
    seed: int | None = None,
) -> torch.Tensor:
    if remove_order == "low":
        return select_indices(base_scores, base_only, k, largest=False)
    if remove_order == "high":
        return select_indices(base_scores, base_only, k, largest=True)
    if remove_order == "random":
        candidate_idx = base_only.nonzero(as_tuple=False).flatten()
        if k <= 0:
            return candidate_idx[:0]
        if candidate_idx.numel() <= k:
            return candidate_idx
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        perm = torch.randperm(candidate_idx.numel(), generator=generator)
        return candidate_idx[perm[:k]]
    raise ValueError(f"unknown remove_order: {remove_order}")


def residual_swap_count(base: Mask, target: Mask, names: list[str], alpha: float) -> tuple[int, int, int]:
    base_flat = flatten_mask(base, names)
    target_flat = flatten_mask(target, names)
    base_only = base_flat & ~target_flat
    target_only = target_flat & ~base_flat
    swappable = min(int(base_only.sum().item()), int(target_only.sum().item()))
    return int(round(alpha * swappable)), int(base_only.sum().item()), int(target_only.sum().item())


def imp_residual_mask(
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    imp_scores: torch.Tensor,
    alpha: float,
    remove_order: str = "low",
    remove_seed: int | None = None,
) -> tuple[Mask, dict[str, int | str]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count, imp_only_count)
    remove_idx = select_removal_indices(
        base_scores=base_scores,
        base_only=base_only,
        k=swap_count,
        remove_order=remove_order,
        seed=remove_seed,
    )
    add_idx = select_indices(imp_scores, imp_only, swap_count, largest=True)
    out = base_flat.clone()
    out[remove_idx] = False
    out[add_idx] = True
    return unflatten_mask(out, names, base), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "remove_order": remove_order,
        "add_order": "top_imp",
    }


def random_residual_mask(
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    alpha: float,
    seed: int,
) -> tuple[Mask, dict[str, int | str]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count)
    remove_idx = select_removal_indices(
        base_scores=base_scores,
        base_only=base_only,
        k=swap_count,
        remove_order="low",
    )

    pool = (~base_flat) & (~imp_flat)
    pool_idx = pool.nonzero(as_tuple=False).flatten()
    if pool_idx.numel() < swap_count:
        pool_idx = (~base_flat).nonzero(as_tuple=False).flatten()
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(pool_idx.numel(), generator=generator)
    add_idx = pool_idx[perm[:swap_count]]
    out = base_flat.clone()
    out[remove_idx] = False
    out[add_idx] = True
    return unflatten_mask(out, names, base), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "remove_order": "low",
        "add_order": "random_nonimp",
    }


def mask_fingerprint(mask: Mask, names: list[str]) -> str:
    flat = flatten_mask(mask, names).to(torch.uint8).numpy().tobytes()
    return hashlib.sha1(flat).hexdigest()


def source_label(source: str) -> str:
    if source.startswith("epoch_"):
        return source.replace("epoch_", "Epoch ")
    return source.replace("traj_", "").replace("_", " ")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.random_residual_trials < 0:
        raise ValueError("random_residual_trials must be non-negative")
    alphas = parse_float_list(args.alphas)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    base_sources = parse_source_list(args.base_sources)
    imp_remove_orders = parse_source_list(args.imp_remove_orders)
    valid_remove_orders = {"low", "random", "high"}
    unknown_remove_orders = sorted(set(imp_remove_orders) - valid_remove_orders)
    if unknown_remove_orders:
        raise ValueError(
            f"unknown imp remove orders {unknown_remove_orders}; "
            f"expected subset of {sorted(valid_remove_orders)}"
        )
    trajectory_epochs = sorted(set(parse_int_list(args.trajectory_epochs) + [0, args.epochs]))
    if args.rewind_epochs not in trajectory_epochs:
        trajectory_epochs = sorted(set(trajectory_epochs + [args.rewind_epochs]))
    if any(epoch < 0 or epoch > args.epochs for epoch in trajectory_epochs):
        raise ValueError("trajectory epochs must be in [0, epochs]")
    if args.rewind_epochs < 0 or args.rewind_epochs > args.epochs:
        raise ValueError("rewind_epochs must be in [0, epochs]")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "digits":
        bundle = load_digits_bundle(args.batch_size, 1024, args.seed)
    elif args.dataset == "fake-cifar10":
        bundle = load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            args.seed,
            train_size=args.train_subset or 2048,
            test_size=args.test_subset or 512,
        )
    else:
        bundle = load_torchvision_bundle(
            args.dataset,
            args.batch_size,
            1024,
            args.seed,
            flatten=args.model == "mlp",
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            augment=args.augment,
        )

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
        )
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
    mask_train_epochs = (
        imp_epochs if args.mask_train_epochs is None else args.mask_train_epochs
    )
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
    dense_mask = global_magnitude_mask_from_state(dense_state, names, imp.metrics["sparsity"])
    rewind_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else None
    )
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
    imp_scores = flatten_scores(
        {name: imp.final_state[name].detach().abs().cpu() for name in names},
        names,
    )

    unknown_sources = sorted(set(base_sources) - set(source_masks))
    if unknown_sources:
        raise ValueError(
            f"unknown base sources {unknown_sources}; expected one of {sorted(source_masks)}"
        )

    rows = []
    train_cache: dict[str, dict[str, float]] = {}

    def evaluate_candidate(
        *,
        base_source: str,
        variant: str,
        alpha: float,
        trial: int | None,
        mask: Mask,
        swap_meta: dict[str, int | str],
    ) -> dict[str, float | str | int | None | bool]:
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
        out: dict[str, float | str | int | None | bool] = {
            "base_source": base_source,
            "base_label": source_label(base_source),
            "variant": variant,
            "alpha": alpha,
            "trial": trial,
            "mask_train_epochs": mask_train_epochs,
            "trained_loss": metrics["loss"],
            "trained_accuracy": metrics["accuracy"],
            "mask_sparsity": metrics["sparsity"],
            "swap_count": swap_meta["swap_count"],
            "base_only_count": swap_meta["base_only_count"],
            "imp_only_count": swap_meta["imp_only_count"],
            "remove_order": swap_meta["remove_order"],
            "add_order": swap_meta["add_order"],
            "mask_to_base_jaccard": support_jaccard(mask, base_mask),
            "mask_to_imp_jaccard": support_jaccard(mask, imp.mask),
            "mask_to_dense_final_magnitude_jaccard": support_jaccard(mask, dense_mask),
            "mask_to_rewind_magnitude_jaccard": (
                support_jaccard(mask, rewind_mask) if rewind_mask is not None else None
            ),
            "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
            "imp_accuracy": imp.metrics["accuracy"],
            "accuracy_minus_imp": metrics["accuracy"] - imp.metrics["accuracy"],
            "accuracy_minus_dense": (
                metrics["accuracy"] - checkpoint_metrics[args.epochs]["accuracy"]
            ),
            "cache_hit": cache_hit,
        }
        rows.append(out)
        print(json.dumps(out), flush=True)
        return out

    for base_source in base_sources:
        base_mask = source_masks[base_source]
        base_score_flat = flatten_scores(source_scores[base_source], names)
        for alpha in alphas:
            for remove_order in imp_remove_orders:
                if alpha <= 0.0 and remove_order != "low":
                    continue
                imp_mask, swap_meta = imp_residual_mask(
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    base_scores=base_score_flat,
                    imp_scores=imp_scores,
                    alpha=alpha,
                    remove_order=remove_order,
                    remove_seed=(
                        args.seed * 100000
                        + int(round(alpha * 1000))
                        + {"low": 11, "random": 17, "high": 23}[remove_order]
                    ),
                )
                variant = (
                    "imp_residual"
                    if remove_order == "low"
                    else f"imp_residual_{remove_order}_remove"
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant=variant,
                    alpha=alpha,
                    trial=None,
                    mask=imp_mask,
                    swap_meta=swap_meta,
                )
            if alpha <= 0.0 or args.random_residual_trials == 0:
                continue
            for trial in range(args.random_residual_trials):
                random_mask, random_meta = random_residual_mask(
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    base_scores=base_score_flat,
                    alpha=alpha,
                    seed=args.seed * 100000 + trial * 1000 + int(round(alpha * 1000)),
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="random_residual",
                    alpha=alpha,
                    trial=trial,
                    mask=random_mask,
                    swap_meta=random_meta,
                )

    payload = {
        "seed": args.seed,
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
            "imp_remove_orders": imp_remove_orders,
            "random_residual_trials": args.random_residual_trials,
        },
        "dense": checkpoint_metrics[args.epochs],
        "imp": imp.metrics,
        "imp_history": imp.history,
        "rows": rows,
        "train_history": train_history,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (run_dir / "trajectory_residual_probe.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "seed": args.seed,
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
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
