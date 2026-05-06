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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lottery.data import DatasetBundle, load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import IMPResult, iterative_magnitude_pruning
from lottery.masks import (
    Mask,
    global_magnitude_mask_from_state,
    global_score_mask,
    mask_sparsity,
    support_jaccard,
)
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu
from run_residual_predictor_mask_probe import (
    binary_auc,
    flatten_mask,
    flatten_scores,
    group_feature_matrix,
    imp_residual_mask,
    mask_fingerprint,
    mask_from_swaps,
    percentile_by_parameter,
    random_heldout_residual_mask,
    select_indices,
    topk_binary_metrics,
    train_fixed_mask,
    train_one_epoch,
    trajectory_score_tensors,
)


@dataclass
class SeedArtifacts:
    seed: int
    bundle: DatasetBundle
    initial_state: dict[str, torch.Tensor]
    train_state: dict[str, torch.Tensor]
    dense_state: dict[str, torch.Tensor]
    rewind_state: dict[str, torch.Tensor] | None
    imp: IMPResult
    names: list[str]
    source_scores: dict[str, dict[str, torch.Tensor]]
    source_masks: dict[str, Mask]
    predictor_features: torch.Tensor
    dense_mask: Mask
    rewind_mask: Mask | None
    imp_scores: torch.Tensor
    checkpoint_metrics: dict[int, dict[str, float]]
    train_history: list[dict[str, float]]


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_source_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
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
    parser.add_argument("--predictor-steps", type=int, default=120)
    parser.add_argument("--predictor-batch-size", type=int, default=16384)
    parser.add_argument("--predictor-lr", type=float, default=0.03)
    parser.add_argument("--random-residual-trials", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_cross_seed_transfer_probe"))
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


def build_artifacts(
    args: argparse.Namespace,
    *,
    seed: int,
    trajectory_epochs: list[int],
    base_sources: list[str],
    imp_epochs: int,
    device: torch.device,
) -> SeedArtifacts:
    set_seed(seed)
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
        train_metrics = train_one_epoch(
            trajectory_model,
            bundle.train_loader,
            device,
            optimizer,
        )
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
        raise ValueError(
            f"unknown base sources {unknown_sources}; expected one of {sorted(source_masks)}"
        )
    dense_mask = global_magnitude_mask_from_state(dense_state, names, imp.metrics["sparsity"])
    rewind_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else None
    )
    imp_scores = flatten_scores(
        {name: imp.final_state[name].detach().abs().cpu() for name in names},
        names,
    )
    feature_names = sorted(source_scores)
    rank_features = torch.stack(
        [percentile_by_parameter(source_scores[name], names) for name in feature_names],
        dim=1,
    )
    predictor_features = torch.cat([rank_features, group_feature_matrix(names, imp.mask)], dim=1)
    print(
        json.dumps(
            {
                "seed": seed,
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "sparsity": imp.metrics["sparsity"],
            }
        ),
        flush=True,
    )
    return SeedArtifacts(
        seed=seed,
        bundle=bundle,
        initial_state=initial_state,
        train_state=train_state,
        dense_state=dense_state,
        rewind_state=rewind_state,
        imp=imp,
        names=names,
        source_scores=source_scores,
        source_masks=source_masks,
        predictor_features=predictor_features,
        dense_mask=dense_mask,
        rewind_mask=rewind_mask,
        imp_scores=imp_scores,
        checkpoint_metrics=checkpoint_metrics,
        train_history=train_history,
    )


def fit_cross_seed_scores(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, int | float | None]]:
    train_x = train_features.float()
    train_y = train_labels.float()
    test_x = test_features.float()
    test_y = test_labels.bool()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    model = nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_pos = max(1, int(train_y.sum().item()))
    train_neg = max(1, int(train_y.numel() - train_pos))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_neg / train_pos]))
    draw = torch.Generator()
    draw.manual_seed(seed)
    for _ in range(steps):
        if train_y.numel() > batch_size:
            batch_idx = torch.randint(train_y.numel(), (batch_size,), generator=draw)
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
        else:
            batch_x = train_x
            batch_y = train_y
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x).flatten()
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(test_x).flatten()).cpu()
    return scores, {
        "train_candidate_count": int(train_y.numel()),
        "train_positive_count": int(train_y.sum().item()),
        "candidate_count": int(test_y.numel()),
        "heldout_count": int(test_y.numel()),
        "heldout_positive_count": int(test_y.sum().item()),
        "predictor_auc": binary_auc(scores, test_y),
    }


def cross_seed_residual_mask(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    candidate_idx: torch.Tensor,
    candidate_scores: torch.Tensor,
    predictor_meta: dict[str, int | float | None],
    alpha: float,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    base_only_count = int(base_only.sum().item())
    imp_only_count = int(imp_only.sum().item())
    swap_count = min(int(round(alpha * min(base_only_count, imp_only_count))), int(candidate_idx.numel()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    if swap_count:
        selected_local = torch.topk(candidate_scores, swap_count, largest=True).indices
    else:
        selected_local = torch.empty(0, dtype=torch.long)
    add_idx = candidate_idx[selected_local]
    hits = int(imp_only[add_idx].sum().item())
    topk = topk_binary_metrics(candidate_scores, imp_only[candidate_idx], swap_count)
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "candidate_count": predictor_meta["candidate_count"],
        "heldout_count": predictor_meta["heldout_count"],
        "heldout_positive_count": predictor_meta["heldout_positive_count"],
        "train_candidate_count": predictor_meta["train_candidate_count"],
        "train_positive_count": predictor_meta["train_positive_count"],
        "predictor_auc": predictor_meta["predictor_auc"],
        "predictor_topk_recall": topk["topk_recall"],
        "predictor_topk_precision": topk["topk_precision"],
        "predictor_baseline_precision": topk["baseline_precision"],
        "added_imp_only_hits": hits,
        "added_imp_only_precision": hits / swap_count if swap_count else None,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if len(seeds) < 2:
        raise ValueError("cross-seed transfer requires at least two seeds")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.random_residual_trials < 0:
        raise ValueError("random_residual_trials must be non-negative")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imp_epochs = args.epochs if args.imp_epochs is None else args.imp_epochs
    mask_train_epochs = imp_epochs if args.mask_train_epochs is None else args.mask_train_epochs

    artifacts = [
        build_artifacts(
            args,
            seed=seed,
            trajectory_epochs=trajectory_epochs,
            base_sources=base_sources,
            imp_epochs=imp_epochs,
            device=device,
        )
        for seed in seeds
    ]
    by_seed = {artifact.seed: artifact for artifact in artifacts}
    rows: list[dict[str, Any]] = []
    train_history = [
        row for artifact in artifacts for row in artifact.train_history
    ]

    for target_seed in seeds:
        target = by_seed[target_seed]
        model_factory = make_model_factory(args, target.bundle)
        train_cache: dict[str, dict[str, float]] = {}

        def evaluate_candidate(
            *,
            base_source: str,
            variant: str,
            alpha: float,
            trial: int | None,
            source_seeds: list[int],
            mask: Mask,
            meta: dict[str, int | float | None],
        ) -> None:
            fingerprint = mask_fingerprint(mask, target.names)
            cache_hit = fingerprint in train_cache
            if not cache_hit:
                train_cache[fingerprint] = train_fixed_mask(
                    model_factory=model_factory,
                    train_state=target.train_state,
                    mask=mask,
                    train_loader=target.bundle.train_loader,
                    test_loader=target.bundle.test_loader,
                    device=device,
                    epochs=mask_train_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lr_schedule=args.lr_schedule,
                )
            metrics = train_cache[fingerprint]
            base_mask = target.source_masks[base_source]
            out: dict[str, Any] = {
                "target_seed": target.seed,
                "source_seeds": ",".join(str(seed) for seed in source_seeds),
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trial": trial,
                "mask_train_epochs": mask_train_epochs,
                "trained_loss": metrics["loss"],
                "trained_accuracy": metrics["accuracy"],
                "mask_sparsity": metrics["sparsity"],
                "mask_to_base_jaccard": support_jaccard(mask, base_mask),
                "mask_to_imp_jaccard": support_jaccard(mask, target.imp.mask),
                "mask_to_dense_final_magnitude_jaccard": support_jaccard(mask, target.dense_mask),
                "mask_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, target.rewind_mask)
                    if target.rewind_mask is not None
                    else None
                ),
                "dense_accuracy": target.checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": target.imp.metrics["accuracy"],
                "accuracy_minus_imp": metrics["accuracy"] - target.imp.metrics["accuracy"],
                "accuracy_minus_dense": (
                    metrics["accuracy"] - target.checkpoint_metrics[args.epochs]["accuracy"]
                ),
                "cache_hit": cache_hit,
            }
            out.update(meta)
            rows.append(out)
            print(json.dumps(out), flush=True)

        for base_source in base_sources:
            base_mask = target.source_masks[base_source]
            base_flat = flatten_mask(base_mask, target.names)
            target_imp_flat = flatten_mask(target.imp.mask, target.names)
            base_score_flat = flatten_scores(target.source_scores[base_source], target.names)
            target_candidate_idx = (~base_flat).nonzero(as_tuple=False).flatten()
            target_labels = torch.zeros_like(base_flat, dtype=torch.bool)
            target_labels[(~base_flat) & target_imp_flat] = True
            source_feature_chunks = []
            source_label_chunks = []
            source_seeds = [seed for seed in seeds if seed != target_seed]
            for source_seed in source_seeds:
                source = by_seed[source_seed]
                source_base_flat = flatten_mask(source.source_masks[base_source], source.names)
                source_imp_flat = flatten_mask(source.imp.mask, source.names)
                source_candidate_idx = (~source_base_flat).nonzero(as_tuple=False).flatten()
                source_labels = torch.zeros_like(source_base_flat, dtype=torch.bool)
                source_labels[(~source_base_flat) & source_imp_flat] = True
                source_feature_chunks.append(source.predictor_features[source_candidate_idx])
                source_label_chunks.append(source_labels[source_candidate_idx])
            train_features = torch.cat(source_feature_chunks, dim=0)
            train_labels = torch.cat(source_label_chunks, dim=0)
            candidate_scores, predictor_meta = fit_cross_seed_scores(
                train_features=train_features,
                train_labels=train_labels,
                test_features=target.predictor_features[target_candidate_idx],
                test_labels=target_labels[target_candidate_idx],
                seed=target.seed * 1009 + len(base_source),
                steps=args.predictor_steps,
                batch_size=args.predictor_batch_size,
                lr=args.predictor_lr,
            )
            evaluate_candidate(
                base_source=base_source,
                variant="base",
                alpha=0.0,
                trial=None,
                source_seeds=[],
                mask=base_mask,
                meta={
                    "swap_count": 0,
                    "base_only_count": int((base_flat & ~target_imp_flat).sum().item()),
                    "imp_only_count": int((~base_flat & target_imp_flat).sum().item()),
                    "candidate_count": int(target_candidate_idx.numel()),
                    "heldout_count": int(target_candidate_idx.numel()),
                    "heldout_positive_count": int(target_labels[target_candidate_idx].sum().item()),
                    "train_candidate_count": int(train_labels.numel()),
                    "train_positive_count": int(train_labels.sum().item()),
                    "predictor_auc": predictor_meta["predictor_auc"],
                    "predictor_topk_recall": None,
                    "predictor_topk_precision": None,
                    "predictor_baseline_precision": (
                        float(target_labels[target_candidate_idx].float().mean().item())
                        if target_candidate_idx.numel()
                        else None
                    ),
                    "added_imp_only_hits": 0,
                    "added_imp_only_precision": None,
                },
            )
            for alpha in alphas:
                oracle_mask, oracle_meta = imp_residual_mask(
                    base=base_mask,
                    imp=target.imp.mask,
                    names=target.names,
                    base_scores=base_score_flat,
                    imp_scores=target.imp_scores,
                    alpha=alpha,
                )
                oracle_meta["train_candidate_count"] = None
                oracle_meta["train_positive_count"] = None
                evaluate_candidate(
                    base_source=base_source,
                    variant="oracle_imp_residual",
                    alpha=alpha,
                    trial=None,
                    source_seeds=[],
                    mask=oracle_mask,
                    meta=oracle_meta,
                )
                cross_mask, cross_meta = cross_seed_residual_mask(
                    base=base_mask,
                    imp=target.imp.mask,
                    names=target.names,
                    base_scores=base_score_flat,
                    candidate_idx=target_candidate_idx,
                    candidate_scores=candidate_scores,
                    predictor_meta=predictor_meta,
                    alpha=alpha,
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="cross_seed_predictor_residual",
                    alpha=alpha,
                    trial=None,
                    source_seeds=source_seeds,
                    mask=cross_mask,
                    meta=cross_meta,
                )
                for trial in range(args.random_residual_trials):
                    random_mask, random_meta = random_heldout_residual_mask(
                        base=base_mask,
                        imp=target.imp.mask,
                        names=target.names,
                        base_scores=base_score_flat,
                        heldout_idx=target_candidate_idx,
                        alpha=alpha,
                        seed=target.seed * 100000 + trial * 1000 + int(round(alpha * 1000)),
                    )
                    random_meta["train_candidate_count"] = int(train_labels.numel())
                    random_meta["train_positive_count"] = int(train_labels.sum().item())
                    random_meta["predictor_auc"] = predictor_meta["predictor_auc"]
                    random_meta["predictor_baseline_precision"] = (
                        float(target_labels[target_candidate_idx].float().mean().item())
                        if target_candidate_idx.numel()
                        else None
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        variant="target_random_residual",
                        alpha=alpha,
                        trial=trial,
                        source_seeds=[],
                        mask=random_mask,
                        meta=random_meta,
                    )
        print(
            json.dumps(
                {
                    "target_seed": target_seed,
                    "num_rows_so_far": len(rows),
                    "unique_trained_masks": len(train_cache),
                }
            ),
            flush=True,
        )

    payload = {
        "seeds": seeds,
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
            "predictor_steps": args.predictor_steps,
            "predictor_batch_size": args.predictor_batch_size,
            "predictor_lr": args.predictor_lr,
            "random_residual_trials": args.random_residual_trials,
        },
        "seed_summaries": [
            {
                "seed": artifact.seed,
                "dense": artifact.checkpoint_metrics[args.epochs],
                "imp": artifact.imp.metrics,
            }
            for artifact in artifacts
        ],
        "rows": rows,
        "train_history": train_history,
    }
    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_rows(run_dir / "residual_cross_seed_transfer_probe.csv", rows)
    print(
        json.dumps(
            {
                "seeds": seeds,
                "dataset": args.dataset,
                "model": args.model,
                "num_rows": len(rows),
                "run_dir": str(run_dir),
                "best_candidate": max(rows, key=lambda row: float(row["trained_accuracy"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
