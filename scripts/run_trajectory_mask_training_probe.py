#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

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
    random_mask_like,
    support_jaccard,
)
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.pruning_baselines import (
    gem_miner_mask,
    hard_concrete_mask,
    variational_pruning_mask,
)
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu, train_model


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


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
        "--mask-sources",
        default=(
            "imp,random,epoch_1,epoch_10,epoch_30,"
            "traj_rms_abs,traj_mean_abs,traj_path_length,traj_rewind_rms_movement"
        ),
    )
    parser.add_argument("--random-trials", type=int, default=1)
    parser.add_argument("--gem-miner-epochs", type=int, default=10)
    parser.add_argument("--gem-miner-lr", type=float, default=0.1)
    parser.add_argument("--gem-miner-regularization", type=float, default=0.0)
    parser.add_argument("--gem-miner-freeze-period", type=int, default=1)
    parser.add_argument("--gem-miner-max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--variational-prune-epochs", type=int, default=10)
    parser.add_argument("--variational-prune-lr", type=float, default=0.01)
    parser.add_argument("--variational-prune-kl-weight", type=float, default=1e-4)
    parser.add_argument("--variational-prune-sparsity-weight", type=float, default=10.0)
    parser.add_argument("--variational-prune-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--variational-prune-temperature-start", type=float, default=2.0)
    parser.add_argument("--variational-prune-temperature-end", type=float, default=0.2)
    parser.add_argument("--variational-prune-samples-per-batch", type=int, default=1)
    parser.add_argument("--variational-prune-max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--hard-concrete-epochs", type=int, default=10)
    parser.add_argument("--hard-concrete-lr", type=float, default=0.01)
    parser.add_argument("--hard-concrete-l0-weight", type=float, default=1e-4)
    parser.add_argument("--hard-concrete-sparsity-weight", type=float, default=10.0)
    parser.add_argument("--hard-concrete-temperature-start", type=float, default=2.0)
    parser.add_argument("--hard-concrete-temperature-end", type=float, default=0.67)
    parser.add_argument("--hard-concrete-stretch-low", type=float, default=-0.1)
    parser.add_argument("--hard-concrete-stretch-high", type=float, default=1.1)
    parser.add_argument("--hard-concrete-samples-per-batch", type=int, default=1)
    parser.add_argument("--hard-concrete-max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_mask_training_probe"))
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


def trajectory_score_masks(
    states: dict[int, dict[str, torch.Tensor]],
    names: list[str],
    trajectory_epochs: list[int],
    rewind_epochs: int,
    sparsity: float,
) -> dict[str, Mask]:
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

    return {
        source: global_score_mask(score, names, sparsity=sparsity, largest=True)
        for source, score in scores.items()
    }


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
    metrics.update(calibration_metrics(model, test_loader, device))
    metrics["sparsity"] = mask_sparsity(mask)
    return metrics


@torch.no_grad()
def calibration_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    bins: int = 15,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    probs = []
    labels = []
    for x, y in data_loader:
        logits = model(x.to(device))
        probs.append(F.softmax(logits, dim=1).cpu())
        labels.append(y.cpu().long())
    prob = torch.cat(probs, dim=0)
    label = torch.cat(labels, dim=0)
    confidence, prediction = prob.max(dim=1)
    correct = prediction.eq(label)
    one_hot = F.one_hot(label, num_classes=prob.shape[1]).float()
    brier = ((prob - one_hot) ** 2).sum(dim=1).mean().item()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if int(mask.sum()) == 0:
            continue
        bin_confidence = confidence[mask].mean().item()
        bin_accuracy = correct[mask].float().mean().item()
        ece += (int(mask.sum()) / confidence.numel()) * abs(
            bin_accuracy - bin_confidence
        )
    return {
        "ece": float(ece),
        "brier": float(brier),
    }


def evaluate_fixed_mask(
    model_factory,
    state: dict[str, torch.Tensor],
    mask: Mask,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model = model_factory()
    load_trainable_state(model, state)
    apply_mask_(model, mask)
    metrics = evaluate(model, test_loader, device)
    metrics.update(calibration_metrics(model, test_loader, device))
    metrics["sparsity"] = mask_sparsity(mask)
    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.random_trials < 0:
        raise ValueError("random_trials must be non-negative")
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
    checkpoint_masks = {
        f"epoch_{epoch}": global_magnitude_mask_from_state(
            states[epoch], names, imp.metrics["sparsity"]
        )
        for epoch in trajectory_epochs
    }
    aggregate_masks = trajectory_score_masks(
        states=states,
        names=names,
        trajectory_epochs=trajectory_epochs,
        rewind_epochs=args.rewind_epochs,
        sparsity=imp.metrics["sparsity"],
    )

    candidate_sources = parse_source_list(args.mask_sources)
    rows = []
    for source in candidate_sources:
        source_masks: list[tuple[str, str, Mask]] = []
        if source == "imp":
            source_masks.append(("imp", "imp", imp.mask))
        elif source == "random":
            for trial in range(args.random_trials):
                source_masks.append(
                    (
                        f"random_{trial}",
                        "random",
                        random_mask_like(
                            imp.mask,
                            sparsity=imp.metrics["sparsity"],
                            seed=args.seed * 1000 + trial,
                        ),
                    )
                )
        elif source in checkpoint_masks:
            source_masks.append((source, "checkpoint", checkpoint_masks[source]))
        elif source in aggregate_masks:
            source_masks.append((source, "aggregate", aggregate_masks[source]))
        elif source == "gem_miner":
            set_seed(args.seed + 70_000)
            source_masks.append(
                (
                    "gem_miner",
                    "gem_miner",
                    gem_miner_mask(
                        model_factory(),
                        initial_state,
                        bundle.train_loader,
                        device,
                        imp.metrics["sparsity"],
                        epochs=args.gem_miner_epochs,
                        lr=args.gem_miner_lr,
                        regularization=args.gem_miner_regularization,
                        freeze_period=args.gem_miner_freeze_period,
                        max_batches_per_epoch=args.gem_miner_max_batches_per_epoch,
                    ),
                )
            )
        elif source == "variational_prune":
            set_seed(args.seed + 80_000)
            source_masks.append(
                (
                    "variational_prune",
                    "variational_prune",
                    variational_pruning_mask(
                        model_factory(),
                        initial_state,
                        bundle.train_loader,
                        device,
                        imp.metrics["sparsity"],
                        epochs=args.variational_prune_epochs,
                        lr=args.variational_prune_lr,
                        kl_weight=args.variational_prune_kl_weight,
                        sparsity_weight=args.variational_prune_sparsity_weight,
                        entropy_weight=args.variational_prune_entropy_weight,
                        temperature_start=args.variational_prune_temperature_start,
                        temperature_end=args.variational_prune_temperature_end,
                        samples_per_batch=args.variational_prune_samples_per_batch,
                        max_batches_per_epoch=args.variational_prune_max_batches_per_epoch,
                    ),
                )
            )
        elif source == "hard_concrete":
            set_seed(args.seed + 90_000)
            source_masks.append(
                (
                    "hard_concrete",
                    "hard_concrete",
                    hard_concrete_mask(
                        model_factory(),
                        initial_state,
                        bundle.train_loader,
                        device,
                        imp.metrics["sparsity"],
                        epochs=args.hard_concrete_epochs,
                        lr=args.hard_concrete_lr,
                        l0_weight=args.hard_concrete_l0_weight,
                        sparsity_weight=args.hard_concrete_sparsity_weight,
                        temperature_start=args.hard_concrete_temperature_start,
                        temperature_end=args.hard_concrete_temperature_end,
                        stretch_low=args.hard_concrete_stretch_low,
                        stretch_high=args.hard_concrete_stretch_high,
                        samples_per_batch=args.hard_concrete_samples_per_batch,
                        max_batches_per_epoch=args.hard_concrete_max_batches_per_epoch,
                    ),
                )
            )
        else:
            raise ValueError(
                "unknown mask source "
                f"{source!r}; expected imp, random, gem_miner, "
                f"variational_prune, hard_concrete, {sorted(checkpoint_masks)}, "
                f"or {sorted(aggregate_masks)}"
            )

        for source_name, source_kind, mask in source_masks:
            source_train_state = (
                initial_state
                if source_kind in {"gem_miner", "variational_prune", "hard_concrete"}
                else train_state
            )
            source_train_state_name = (
                "initial"
                if source_kind in {"gem_miner", "variational_prune", "hard_concrete"}
                else "rewind"
            )
            pretrain_metrics = evaluate_fixed_mask(
                model_factory=model_factory,
                state=source_train_state,
                mask=mask,
                test_loader=bundle.test_loader,
                device=device,
            )
            metrics = train_fixed_mask(
                model_factory=model_factory,
                train_state=source_train_state,
                mask=mask,
                train_loader=bundle.train_loader,
                test_loader=bundle.test_loader,
                device=device,
                epochs=mask_train_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_schedule=args.lr_schedule,
            )
            row = {
                "source": source_name,
                "source_kind": source_kind,
                "train_state_source": source_train_state_name,
                "mask_train_epochs": mask_train_epochs,
                "pretrain_loss": pretrain_metrics["loss"],
                "pretrain_accuracy": pretrain_metrics["accuracy"],
                "pretrain_ece": pretrain_metrics["ece"],
                "pretrain_brier": pretrain_metrics["brier"],
                "trained_loss": metrics["loss"],
                "trained_accuracy": metrics["accuracy"],
                "trained_ece": metrics["ece"],
                "trained_brier": metrics["brier"],
                "mask_sparsity": metrics["sparsity"],
                "source_to_imp_jaccard": support_jaccard(mask, imp.mask),
                "source_to_dense_final_magnitude_jaccard": support_jaccard(mask, dense_mask),
                "source_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, rewind_mask) if rewind_mask is not None else None
                ),
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "accuracy_minus_imp": metrics["accuracy"] - imp.metrics["accuracy"],
                "accuracy_minus_dense": (
                    metrics["accuracy"] - checkpoint_metrics[args.epochs]["accuracy"]
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

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
            "mask_sources": candidate_sources,
            "random_trials": args.random_trials,
            "gem_miner_epochs": args.gem_miner_epochs,
            "gem_miner_lr": args.gem_miner_lr,
            "gem_miner_regularization": args.gem_miner_regularization,
            "gem_miner_freeze_period": args.gem_miner_freeze_period,
            "gem_miner_max_batches_per_epoch": args.gem_miner_max_batches_per_epoch,
            "variational_prune_epochs": args.variational_prune_epochs,
            "variational_prune_lr": args.variational_prune_lr,
            "variational_prune_kl_weight": args.variational_prune_kl_weight,
            "variational_prune_sparsity_weight": args.variational_prune_sparsity_weight,
            "variational_prune_entropy_weight": args.variational_prune_entropy_weight,
            "variational_prune_temperature_start": (
                args.variational_prune_temperature_start
            ),
            "variational_prune_temperature_end": args.variational_prune_temperature_end,
            "variational_prune_samples_per_batch": (
                args.variational_prune_samples_per_batch
            ),
            "variational_prune_max_batches_per_epoch": (
                args.variational_prune_max_batches_per_epoch
            ),
            "hard_concrete_epochs": args.hard_concrete_epochs,
            "hard_concrete_lr": args.hard_concrete_lr,
            "hard_concrete_l0_weight": args.hard_concrete_l0_weight,
            "hard_concrete_sparsity_weight": args.hard_concrete_sparsity_weight,
            "hard_concrete_temperature_start": args.hard_concrete_temperature_start,
            "hard_concrete_temperature_end": args.hard_concrete_temperature_end,
            "hard_concrete_stretch_low": args.hard_concrete_stretch_low,
            "hard_concrete_stretch_high": args.hard_concrete_stretch_high,
            "hard_concrete_samples_per_batch": args.hard_concrete_samples_per_batch,
            "hard_concrete_max_batches_per_epoch": (
                args.hard_concrete_max_batches_per_epoch
            ),
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
    with (run_dir / "mask_training_probe.csv").open("w", newline="", encoding="utf-8") as f:
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
                "best_candidate": max(rows, key=lambda row: float(row["trained_accuracy"])),
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
