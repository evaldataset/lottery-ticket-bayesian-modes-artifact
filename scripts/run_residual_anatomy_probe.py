#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
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


@dataclass(frozen=True)
class IMPTrace:
    mask: Mask
    final_state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    history: list[dict[str, float]]
    round_masks: list[Mask]


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
        help="Comma-separated checkpoint or aggregate masks whose IMP residual is analyzed.",
    )
    parser.add_argument("--predictor-steps", type=int, default=120)
    parser.add_argument("--predictor-batch-size", type=int, default=16384)
    parser.add_argument("--predictor-lr", type=float, default=0.03)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_anatomy_probe"))
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


def iterative_magnitude_pruning_trace(
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
) -> IMPTrace:
    model = model_factory()
    names = weight_parameter_names(model)
    current_mask = dense_mask(model)
    train_state = initial_state if rewind_state is None else rewind_state
    history: list[dict[str, float]] = []
    round_masks: list[Mask] = []
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
    return IMPTrace(
        mask=current_mask,
        final_state=state_to_cpu(final_model),
        metrics=final_metrics,
        history=history,
        round_masks=round_masks,
    )


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


def flatten_mask(mask: Mask, names: list[str]) -> torch.Tensor:
    return torch.cat([mask[name].bool().flatten().cpu() for name in names])


def flatten_scores(scores: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([scores[name].detach().float().flatten().cpu() for name in names])


def flatten_group_labels(names: list[str], reference: Mask) -> tuple[list[str], list[str]]:
    labels = []
    for name in names:
        labels.extend([parameter_group(name)] * reference[name].numel())
    return labels, sorted(set(labels))


def percentile_by_parameter(scores: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    chunks = []
    for name in names:
        flat = scores[name].detach().float().flatten().cpu()
        if flat.numel() <= 1:
            chunks.append(torch.full_like(flat, 0.5))
            continue
        order = torch.argsort(flat, stable=True)
        ranks = torch.empty(flat.numel(), dtype=torch.float32)
        ranks[order] = torch.arange(flat.numel(), dtype=torch.float32)
        chunks.append(ranks / float(flat.numel() - 1))
    return torch.cat(chunks)


def pruning_round_vector(round_masks: list[Mask], names: list[str]) -> torch.Tensor:
    if not round_masks:
        raise ValueError("round_masks must not be empty")
    prev = torch.ones_like(flatten_mask(round_masks[0], names), dtype=torch.bool)
    pruned_round = torch.zeros(prev.numel(), dtype=torch.int16)
    for idx, mask in enumerate(round_masks, start=1):
        current = flatten_mask(mask, names)
        dropped = prev & ~current
        pruned_round[dropped] = idx
        prev = current
    pruned_round[prev] = len(round_masks) + 1
    return pruned_round


def mask_mean(scores: torch.Tensor, selector: torch.Tensor) -> float | None:
    if int(selector.sum().item()) == 0:
        return None
    return float(scores[selector].double().mean().item())


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.bool()
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    scores = scores.detach().float().cpu()
    labels_f = labels.detach().double().cpu()
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels_f[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    rank_sum = 0.0
    start = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        end = start + count
        avg_rank = (start + 1 + end) / 2.0
        rank_sum += float(sorted_labels[start:end].sum().item()) * avg_rank
        start = end
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def topk_binary_metrics(scores: torch.Tensor, labels: torch.Tensor, k: int | None = None) -> dict[str, float | int | None]:
    labels = labels.bool()
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    if n_pos == 0 or n_total == 0:
        return {
            "topk": 0,
            "topk_hits": 0,
            "topk_recall": None,
            "topk_precision": None,
            "baseline_precision": None,
            "topk_lift": None,
        }
    topk = n_pos if k is None else min(k, n_total)
    selected = torch.topk(scores.detach().float().cpu(), topk, largest=True).indices
    hits = int(labels.cpu()[selected].sum().item())
    precision = hits / topk
    baseline = n_pos / n_total
    return {
        "topk": topk,
        "topk_hits": hits,
        "topk_recall": hits / n_pos,
        "topk_precision": precision,
        "baseline_precision": baseline,
        "topk_lift": precision / baseline if baseline > 0 else None,
    }


def fit_logistic_predictor(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    feature_names: list[str],
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[dict[str, float | int | None], list[dict[str, float | str]]]:
    y = y.bool().cpu()
    n_total = int(y.numel())
    n_pos = int(y.sum().item())
    n_neg = n_total - n_pos
    if steps <= 0 or n_pos == 0 or n_neg == 0 or n_total < 10:
        return (
            {
                "candidate_count": n_total,
                "positive_count": n_pos,
                "negative_count": n_neg,
                "train_count": 0,
                "test_count": 0,
                "test_auc": None,
                "test_topk_recall": None,
                "test_topk_precision": None,
                "test_baseline_precision": None,
                "test_topk_lift": None,
                "test_loss": None,
            },
            [],
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    test_count = max(1, int(round(0.2 * n_total)))
    test_idx = perm[:test_count]
    train_idx = perm[test_count:]
    train_x = x[train_idx].float()
    test_x = x[test_idx].float()
    train_y = y[train_idx].float()
    test_y = y[test_idx]
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
    draw.manual_seed(seed + 100003)
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
        logits = model(test_x).flatten()
        probs = torch.sigmoid(logits)
        test_loss = torch.nn.functional.binary_cross_entropy(probs, test_y.float()).item()
    topk = topk_binary_metrics(probs, test_y)
    metrics = {
        "candidate_count": n_total,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "train_count": int(train_y.numel()),
        "test_count": int(test_y.numel()),
        "test_auc": binary_auc(probs, test_y),
        "test_topk_recall": topk["topk_recall"],
        "test_topk_precision": topk["topk_precision"],
        "test_baseline_precision": topk["baseline_precision"],
        "test_topk_lift": topk["topk_lift"],
        "test_loss": test_loss,
    }
    coefs = model.weight.detach().flatten().cpu()
    coef_rows = [
        {
            "feature": name,
            "coefficient": float(value.item()),
        }
        for name, value in zip(feature_names, coefs, strict=True)
    ]
    return metrics, coef_rows


def add_group_or_layer_rows(
    *,
    rows: list[dict[str, object]],
    seed: int,
    base_source: str,
    unit_kind: str,
    unit: str,
    selector: torch.Tensor,
    base_flat: torch.Tensor,
    imp_flat: torch.Tensor,
    pruned_round: torch.Tensor,
    total_count: int,
    global_imp_only_count: int,
    global_base_only_count: int,
) -> None:
    base = base_flat & selector
    imp = imp_flat & selector
    shared = base & imp
    imp_only = (~base_flat) & imp_flat & selector
    base_only = base_flat & (~imp_flat) & selector
    union = base | imp
    total = int(selector.sum().item())
    unit_size_share = total / total_count if total_count else 0.0
    imp_only_count = int(imp_only.sum().item())
    base_only_count = int(base_only.sum().item())
    imp_only_share = imp_only_count / global_imp_only_count if global_imp_only_count else None
    base_only_share = base_only_count / global_base_only_count if global_base_only_count else None
    rows.append(
        {
            "seed": seed,
            "base_source": base_source,
            "unit_kind": unit_kind,
            "unit": unit,
            "total": total,
            "base_kept": int(base.sum().item()),
            "imp_kept": int(imp.sum().item()),
            "shared": int(shared.sum().item()),
            "imp_only": imp_only_count,
            "base_only": base_only_count,
            "neither": int((~base_flat & ~imp_flat & selector).sum().item()),
            "union": int(union.sum().item()),
            "jaccard": float(shared.sum().item() / union.sum().item()) if union.sum().item() else 1.0,
            "imp_only_share": imp_only_share,
            "base_only_share": base_only_share,
            "size_share": unit_size_share,
            "imp_only_enrichment": (
                imp_only_share / unit_size_share
                if imp_only_share is not None and unit_size_share > 0
                else None
            ),
            "base_only_enrichment": (
                base_only_share / unit_size_share
                if base_only_share is not None and unit_size_share > 0
                else None
            ),
            "imp_only_density": imp_only_count / total if total else None,
            "base_only_density": base_only_count / total if total else None,
            "base_only_pruned_round_mean": (
                float(pruned_round[base_only].float().mean().item())
                if base_only_count
                else None
            ),
        }
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    trajectory_epochs = sorted(set(parse_int_list(args.trajectory_epochs) + [0, args.epochs]))
    if args.rewind_epochs not in trajectory_epochs:
        trajectory_epochs = sorted(set(trajectory_epochs + [args.rewind_epochs]))
    if any(epoch < 0 or epoch > args.epochs for epoch in trajectory_epochs):
        raise ValueError("trajectory epochs must be in [0, epochs]")
    if args.rewind_epochs < 0 or args.rewind_epochs > args.epochs:
        raise ValueError("rewind_epochs must be in [0, epochs]")
    base_sources = parse_source_list(args.base_sources)

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
    imp = iterative_magnitude_pruning_trace(
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
    dense_mask = global_magnitude_mask_from_state(dense_state, names, imp.metrics["sparsity"])
    source_masks.setdefault(f"epoch_{args.epochs}", dense_mask)
    source_scores.setdefault(
        f"epoch_{args.epochs}",
        {name: dense_state[name].detach().abs().cpu() for name in names},
    )
    unknown_sources = sorted(set(base_sources) - set(source_masks))
    if unknown_sources:
        raise ValueError(
            f"unknown base sources {unknown_sources}; expected one of {sorted(source_masks)}"
        )

    imp_flat = flatten_mask(imp.mask, names)
    pruned_round = pruning_round_vector(imp.round_masks, names)
    total_count = int(imp_flat.numel())
    group_labels, group_names = flatten_group_labels(names, imp.mask)
    group_label_tensor = torch.tensor(
        [group_names.index(label) for label in group_labels],
        dtype=torch.long,
    )
    parameter_slices: list[tuple[str, str, slice]] = []
    offset = 0
    for name in names:
        size = imp.mask[name].numel()
        parameter_slices.append((name, parameter_group(name), slice(offset, offset + size)))
        offset += size

    predictor_feature_scores = {
        feature: percentile_by_parameter(score, names)
        for feature, score in source_scores.items()
    }
    ordered_feature_names = sorted(predictor_feature_scores)
    base_feature_matrix = torch.stack(
        [predictor_feature_scores[name] for name in ordered_feature_names],
        dim=1,
    )
    group_one_hot = torch.nn.functional.one_hot(
        group_label_tensor,
        num_classes=len(group_names),
    ).float()
    predictor_matrix = torch.cat([base_feature_matrix, group_one_hot], dim=1)
    predictor_feature_names = ordered_feature_names + [f"group_{name}" for name in group_names]

    global_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for base_source in base_sources:
        base_mask = source_masks[base_source]
        base_flat = flatten_mask(base_mask, names)
        shared = base_flat & imp_flat
        imp_only = (~base_flat) & imp_flat
        base_only = base_flat & (~imp_flat)
        neither = (~base_flat) & (~imp_flat)
        union = base_flat | imp_flat
        imp_only_count = int(imp_only.sum().item())
        base_only_count = int(base_only.sum().item())
        global_rows.append(
            {
                "seed": args.seed,
                "base_source": base_source,
                "total": total_count,
                "base_kept": int(base_flat.sum().item()),
                "imp_kept": int(imp_flat.sum().item()),
                "shared": int(shared.sum().item()),
                "imp_only": imp_only_count,
                "base_only": base_only_count,
                "neither": int(neither.sum().item()),
                "union": int(union.sum().item()),
                "jaccard": support_jaccard(base_mask, imp.mask),
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
                "base_only_pruned_round_mean": (
                    float(pruned_round[base_only].float().mean().item())
                    if base_only_count
                    else None
                ),
            }
        )
        for group in group_names:
            selector = group_label_tensor == group_names.index(group)
            add_group_or_layer_rows(
                rows=group_rows,
                seed=args.seed,
                base_source=base_source,
                unit_kind="group",
                unit=group,
                selector=selector,
                base_flat=base_flat,
                imp_flat=imp_flat,
                pruned_round=pruned_round,
                total_count=total_count,
                global_imp_only_count=imp_only_count,
                global_base_only_count=base_only_count,
            )
        for parameter, _group, slc in parameter_slices:
            selector = torch.zeros(total_count, dtype=torch.bool)
            selector[slc] = True
            add_group_or_layer_rows(
                rows=layer_rows,
                seed=args.seed,
                base_source=base_source,
                unit_kind="parameter",
                unit=parameter,
                selector=selector,
                base_flat=base_flat,
                imp_flat=imp_flat,
                pruned_round=pruned_round,
                total_count=total_count,
                global_imp_only_count=imp_only_count,
                global_base_only_count=base_only_count,
            )
        for group in ["all", *group_names]:
            selector = torch.ones(total_count, dtype=torch.bool) if group == "all" else (
                group_label_tensor == group_names.index(group)
            )
            group_base_only = base_only & selector
            denom = int(group_base_only.sum().item())
            for round_idx in range(1, args.imp_rounds + 1):
                count = int((group_base_only & (pruned_round == round_idx)).sum().item())
                round_rows.append(
                    {
                        "seed": args.seed,
                        "base_source": base_source,
                        "group": group,
                        "pruned_round": round_idx,
                        "base_only_pruned": count,
                        "base_only_total": denom,
                        "fraction": count / denom if denom else None,
                    }
                )

        candidate = ~base_flat
        labels = imp_only[candidate]
        for feature_name in ordered_feature_names:
            feature_scores = predictor_feature_scores[feature_name]
            candidate_scores = feature_scores[candidate]
            topk = topk_binary_metrics(candidate_scores, labels)
            score_rows.append(
                {
                    "seed": args.seed,
                    "base_source": base_source,
                    "feature": feature_name,
                    "candidate_count": int(labels.numel()),
                    "positive_count": int(labels.sum().item()),
                    "auc_imp_only_vs_nonbase": binary_auc(candidate_scores, labels),
                    "imp_only_mean": mask_mean(feature_scores, imp_only),
                    "base_only_mean": mask_mean(feature_scores, base_only),
                    "shared_mean": mask_mean(feature_scores, shared),
                    "neither_mean": mask_mean(feature_scores, neither),
                    **topk,
                }
            )

        predictor_metrics, coefs = fit_logistic_predictor(
            x=predictor_matrix[candidate],
            y=labels,
            feature_names=predictor_feature_names,
            seed=args.seed * 1009 + len(base_source),
            steps=args.predictor_steps,
            batch_size=args.predictor_batch_size,
            lr=args.predictor_lr,
        )
        predictor_row = {
            "seed": args.seed,
            "base_source": base_source,
            "feature_set": "trajectory_rank_plus_group_logistic",
            "steps": args.predictor_steps,
            **predictor_metrics,
        }
        predictor_rows.append(predictor_row)
        for row in coefs:
            out = {
                "seed": args.seed,
                "base_source": base_source,
                "feature_set": "trajectory_rank_plus_group_logistic",
            }
            out.update(row)
            coefficient_rows.append(out)
        print(json.dumps(predictor_row), flush=True)

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
            "base_sources": base_sources,
            "predictor_steps": args.predictor_steps,
            "predictor_batch_size": args.predictor_batch_size,
            "predictor_lr": args.predictor_lr,
        },
        "dense": checkpoint_metrics[args.epochs],
        "imp": imp.metrics,
        "imp_history": imp.history,
        "global_rows": global_rows,
        "group_rows": group_rows,
        "layer_rows": layer_rows,
        "round_rows": round_rows,
        "score_rows": score_rows,
        "predictor_rows": predictor_rows,
        "coefficient_rows": coefficient_rows,
        "train_history": train_history,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_rows(run_dir / "residual_anatomy_global.csv", global_rows)
    write_rows(run_dir / "residual_anatomy_group.csv", group_rows)
    write_rows(run_dir / "residual_anatomy_layer.csv", layer_rows)
    write_rows(run_dir / "residual_pruning_round.csv", round_rows)
    write_rows(run_dir / "residual_score_probe.csv", score_rows)
    write_rows(run_dir / "residual_predictor_probe.csv", predictor_rows)
    write_rows(run_dir / "residual_predictor_coefficients.csv", coefficient_rows)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "dataset": args.dataset,
                "model": args.model,
                "dense_accuracy": checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "num_global_rows": len(global_rows),
                "num_group_rows": len(group_rows),
                "num_layer_rows": len(layer_rows),
                "num_score_rows": len(score_rows),
                "num_predictor_rows": len(predictor_rows),
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
