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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import (
    Mask,
    global_magnitude_mask_from_state,
    global_score_mask,
    mask_sparsity,
    support_jaccard,
)
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import evaluate, load_trainable_state, set_seed, state_to_cpu


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


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
    parser.add_argument("--imp-rounds", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_probe"))
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


def parameter_group(name: str) -> str:
    if name.startswith("conv1"):
        return "stem"
    if name.startswith("layer1"):
        return "stage1"
    if name.startswith("layer2"):
        return "stage2"
    if name.startswith("layer3"):
        return "stage3"
    if name.startswith(("fc", "classifier")):
        return "head"
    if name.startswith("features"):
        return "features"
    if name.startswith("net"):
        return "mlp"
    return "other"


def single_tensor_jaccard(left: torch.Tensor, right: torch.Tensor) -> float | None:
    a = left.bool().flatten().cpu()
    b = right.bool().flatten().cpu()
    union = (a | b).sum().item()
    if union == 0:
        return None
    return float((a & b).sum().item() / union)


def group_support_jaccards(mask: Mask, imp_mask: Mask, names: list[str]) -> list[dict[str, object]]:
    by_group: dict[str, list[str]] = {}
    for name in names:
        by_group.setdefault(parameter_group(name), []).append(name)
    rows: list[dict[str, object]] = []
    for group, group_names in sorted(by_group.items()):
        rows.append(
            {
                "group": group,
                "group_jaccard": support_jaccard(
                    {name: mask[name] for name in group_names},
                    {name: imp_mask[name] for name in group_names},
                ),
                "kept": int(sum(mask[name].bool().sum().item() for name in group_names)),
                "imp_kept": int(sum(imp_mask[name].bool().sum().item() for name in group_names)),
                "total": int(sum(mask[name].numel() for name in group_names)),
            }
        )
    return rows


def layer_support_jaccards(mask: Mask, imp_mask: Mask, names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        rows.append(
            {
                "parameter": name,
                "group": parameter_group(name),
                "layer_jaccard": single_tensor_jaccard(mask[name], imp_mask[name]),
                "kept": int(mask[name].bool().sum().item()),
                "imp_kept": int(imp_mask[name].bool().sum().item()),
                "total": int(mask[name].numel()),
            }
        )
    return rows


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
    scores: dict[str, dict[str, torch.Tensor]] = {
        "traj_mean_abs": {},
        "traj_rms_abs": {},
        "traj_max_abs": {},
        "traj_initial_rms_movement": {},
        "traj_rewind_rms_movement": {},
        "traj_path_length": {},
        "traj_post_rewind_path_length": {},
    }
    post_rewind_sequence = [
        states[epoch] for epoch in ordered_epochs if epoch >= rewind_epochs
    ]
    if len(post_rewind_sequence) < 2:
        post_rewind_sequence = state_sequence

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


def attach_layer_rows(
    *,
    out_group_rows: list[dict[str, object]],
    out_layer_rows: list[dict[str, object]],
    source_kind: str,
    source: str,
    mask: Mask,
    imp_mask: Mask,
    names: list[str],
) -> None:
    for row in group_support_jaccards(mask, imp_mask, names):
        out = {"source_kind": source_kind, "source": source}
        out.update(row)
        out_group_rows.append(out)
    for row in layer_support_jaccards(mask, imp_mask, names):
        out = {"source_kind": source_kind, "source": source}
        out.update(row)
        out_layer_rows.append(out)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    trajectory_epochs = sorted(set(parse_int_list(args.trajectory_epochs) + [0, args.epochs]))
    if any(epoch < 0 or epoch > args.epochs for epoch in trajectory_epochs):
        raise ValueError("trajectory epochs must be in [0, epochs]")
    if args.rewind_epochs < 0 or args.rewind_epochs > args.epochs:
        raise ValueError("rewind_epochs must be in [0, epochs]")
    if args.rewind_epochs not in trajectory_epochs:
        trajectory_epochs.append(args.rewind_epochs)
        trajectory_epochs = sorted(set(trajectory_epochs))

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
    imp_epochs = args.epochs if args.imp_epochs is None else args.imp_epochs
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
    rows = []
    group_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    checkpoint_masks: dict[int, Mask] = {}
    for epoch in trajectory_epochs:
        state = states[epoch]
        mask = global_magnitude_mask_from_state(state, names, imp.metrics["sparsity"])
        checkpoint_masks[epoch] = mask
        attach_layer_rows(
            out_group_rows=group_rows,
            out_layer_rows=layer_rows,
            source_kind="checkpoint",
            source=f"epoch_{epoch}",
            mask=mask,
            imp_mask=imp.mask,
            names=names,
        )
        rows.append(
            {
                "epoch": epoch,
                "checkpoint_accuracy": checkpoint_metrics[epoch]["accuracy"],
                "checkpoint_loss": checkpoint_metrics[epoch]["loss"],
                "trajectory_magnitude_to_imp_jaccard": support_jaccard(mask, imp.mask),
                "trajectory_to_dense_final_magnitude_jaccard": support_jaccard(mask, dense_mask),
                "trajectory_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, rewind_mask) if rewind_mask is not None else None
                ),
                "dense_magnitude_to_imp_jaccard": support_jaccard(dense_mask, imp.mask),
                "rewind_magnitude_to_imp_jaccard": (
                    support_jaccard(rewind_mask, imp.mask) if rewind_mask is not None else None
                ),
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
            }
        )

    best = max(rows, key=lambda row: float(row["trajectory_magnitude_to_imp_jaccard"]))
    aggregate_rows = []
    aggregate_masks = trajectory_score_masks(
        states=states,
        names=names,
        trajectory_epochs=trajectory_epochs,
        rewind_epochs=args.rewind_epochs,
        sparsity=imp.metrics["sparsity"],
    )
    for source, mask in aggregate_masks.items():
        attach_layer_rows(
            out_group_rows=group_rows,
            out_layer_rows=layer_rows,
            source_kind="aggregate",
            source=source,
            mask=mask,
            imp_mask=imp.mask,
            names=names,
        )
        aggregate_rows.append(
            {
                "source": source,
                "trajectory_score_to_imp_jaccard": support_jaccard(mask, imp.mask),
                "trajectory_score_to_dense_final_magnitude_jaccard": support_jaccard(
                    mask, dense_mask
                ),
                "trajectory_score_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, rewind_mask) if rewind_mask is not None else None
                ),
                "trajectory_score_to_best_checkpoint_jaccard": support_jaccard(
                    mask, checkpoint_masks[int(best["epoch"])]
                ),
                "mask_sparsity": mask_sparsity(mask),
                "best_checkpoint_epoch": best["epoch"],
                "best_checkpoint_to_imp_jaccard": best[
                    "trajectory_magnitude_to_imp_jaccard"
                ],
                "dense_magnitude_to_imp_jaccard": support_jaccard(dense_mask, imp.mask),
                "rewind_magnitude_to_imp_jaccard": (
                    support_jaccard(rewind_mask, imp.mask) if rewind_mask is not None else None
                ),
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
            }
        )
    best_aggregate = max(
        aggregate_rows,
        key=lambda row: float(row["trajectory_score_to_imp_jaccard"]),
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
            "imp_rounds": args.imp_rounds,
            "prune_fraction": args.prune_fraction,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "augment": args.augment,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
        },
        "dense": checkpoint_metrics[args.epochs],
        "imp": imp.metrics,
        "imp_history": imp.history,
        "best_trajectory": best,
        "best_aggregate": best_aggregate,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "group_rows": group_rows,
        "layer_rows": layer_rows,
        "train_history": train_history,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (run_dir / "trajectory_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "trajectory_aggregate_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (run_dir / "trajectory_group_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    with (run_dir / "trajectory_layer_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "dataset": args.dataset,
                "model": args.model,
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
                "best_trajectory": best,
                "best_aggregate": best_aggregate,
                "num_group_rows": len(group_rows),
                "num_layer_rows": len(layer_rows),
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
