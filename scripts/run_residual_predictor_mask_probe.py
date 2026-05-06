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
    parser.add_argument("--base-sources", default="epoch_30,traj_rms_abs,epoch_10")
    parser.add_argument("--alphas", default="0.5")
    parser.add_argument("--predictor-steps", type=int, default=120)
    parser.add_argument("--predictor-batch-size", type=int, default=16384)
    parser.add_argument("--predictor-lr", type=float, default=0.03)
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    parser.add_argument("--random-residual-trials", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_predictor_mask_probe"))
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


def residual_swap_count(base: Mask, target: Mask, names: list[str], alpha: float) -> tuple[int, int, int]:
    base_flat = flatten_mask(base, names)
    target_flat = flatten_mask(target, names)
    base_only = base_flat & ~target_flat
    target_only = target_flat & ~base_flat
    swappable = min(int(base_only.sum().item()), int(target_only.sum().item()))
    return int(round(alpha * swappable)), int(base_only.sum().item()), int(target_only.sum().item())


def mask_from_swaps(
    base: Mask,
    names: list[str],
    remove_idx: torch.Tensor,
    add_idx: torch.Tensor,
) -> Mask:
    out = flatten_mask(base, names)
    out[remove_idx] = False
    out[add_idx] = True
    return unflatten_mask(out, names, base)


def imp_residual_mask(
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    imp_scores: torch.Tensor,
    alpha: float,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count, imp_only_count)
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    add_idx = select_indices(imp_scores, imp_only, swap_count, largest=True)
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "candidate_count": None,
        "heldout_count": None,
        "heldout_positive_count": None,
        "predictor_auc": None,
        "predictor_topk_recall": None,
        "predictor_topk_precision": None,
        "predictor_baseline_precision": None,
        "added_imp_only_hits": swap_count,
        "added_imp_only_precision": 1.0 if swap_count else None,
    }


def random_heldout_residual_mask(
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    heldout_idx: torch.Tensor,
    alpha: float,
    seed: int,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count, int(heldout_idx.numel()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(heldout_idx.numel(), generator=generator)
    add_idx = heldout_idx[perm[:swap_count]]
    hits = int(imp_only[add_idx].sum().item())
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "candidate_count": None,
        "heldout_count": int(heldout_idx.numel()),
        "heldout_positive_count": int(imp_only[heldout_idx].sum().item()),
        "predictor_auc": None,
        "predictor_topk_recall": None,
        "predictor_topk_precision": None,
        "predictor_baseline_precision": None,
        "added_imp_only_hits": hits,
        "added_imp_only_precision": hits / swap_count if swap_count else None,
    }


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


def group_feature_matrix(names: list[str], reference: Mask) -> torch.Tensor:
    labels = []
    groups = []
    for name in names:
        group = parameter_group(name)
        groups.append(group)
        labels.extend([group] * reference[name].numel())
    ordered = sorted(set(groups))
    encoded = torch.tensor([ordered.index(label) for label in labels], dtype=torch.long)
    return torch.nn.functional.one_hot(encoded, num_classes=len(ordered)).float()


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


def topk_binary_metrics(scores: torch.Tensor, labels: torch.Tensor, k: int) -> dict[str, float | int | None]:
    labels = labels.bool()
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    if n_pos == 0 or n_total == 0 or k <= 0:
        return {
            "topk": 0,
            "topk_hits": 0,
            "topk_recall": None,
            "topk_precision": None,
            "baseline_precision": None,
        }
    topk = min(k, n_total)
    selected = torch.topk(scores.detach().float().cpu(), topk, largest=True).indices
    hits = int(labels.cpu()[selected].sum().item())
    return {
        "topk": topk,
        "topk_hits": hits,
        "topk_recall": hits / n_pos,
        "topk_precision": hits / topk,
        "baseline_precision": n_pos / n_total,
    }


def fit_predictor_scores(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    candidate_idx: torch.Tensor,
    heldout_fraction: float,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | None]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(candidate_idx.numel(), generator=generator)
    heldout_count = max(1, int(round(candidate_idx.numel() * heldout_fraction)))
    heldout_idx = candidate_idx[perm[:heldout_count]]
    train_idx = candidate_idx[perm[heldout_count:]]

    train_x = features[train_idx].float()
    train_y = labels[train_idx].float()
    heldout_x = features[heldout_idx].float()
    heldout_y = labels[heldout_idx].bool()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    heldout_x = (heldout_x - mean) / std
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
        scores = torch.sigmoid(model(heldout_x).flatten()).cpu()
    metrics = {
        "candidate_count": int(candidate_idx.numel()),
        "heldout_count": int(heldout_idx.numel()),
        "heldout_positive_count": int(heldout_y.sum().item()),
        "predictor_auc": binary_auc(scores, heldout_y),
    }
    return heldout_idx, scores, metrics


def predictor_residual_mask(
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    heldout_idx: torch.Tensor,
    heldout_scores: torch.Tensor,
    predictor_meta: dict[str, int | float | None],
    alpha: float,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    swap_count = min(swap_count, base_only_count, int(heldout_idx.numel()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    selected_local = torch.topk(heldout_scores, swap_count, largest=True).indices if swap_count else torch.empty(0, dtype=torch.long)
    add_idx = heldout_idx[selected_local]
    hits = int(imp_only[add_idx].sum().item())
    topk = topk_binary_metrics(heldout_scores, imp_only[heldout_idx], swap_count)
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "candidate_count": predictor_meta["candidate_count"],
        "heldout_count": predictor_meta["heldout_count"],
        "heldout_positive_count": predictor_meta["heldout_positive_count"],
        "predictor_auc": predictor_meta["predictor_auc"],
        "predictor_topk_recall": topk["topk_recall"],
        "predictor_topk_precision": topk["topk_precision"],
        "predictor_baseline_precision": topk["baseline_precision"],
        "added_imp_only_hits": hits,
        "added_imp_only_precision": hits / swap_count if swap_count else None,
    }


def mask_fingerprint(mask: Mask, names: list[str]) -> str:
    flat = flatten_mask(mask, names).to(torch.uint8).numpy().tobytes()
    return hashlib.sha1(flat).hexdigest()


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
    imp_flat = flatten_mask(imp.mask, names)

    rows = []
    train_cache: dict[str, dict[str, float]] = {}

    def evaluate_candidate(
        *,
        base_source: str,
        variant: str,
        alpha: float,
        trial: int | None,
        mask: Mask,
        meta: dict[str, int | float | None],
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
        out: dict[str, object] = {
            "base_source": base_source,
            "variant": variant,
            "alpha": alpha,
            "trial": trial,
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
            "accuracy_minus_dense": (
                metrics["accuracy"] - checkpoint_metrics[args.epochs]["accuracy"]
            ),
            "cache_hit": cache_hit,
        }
        out.update(meta)
        rows.append(out)
        print(json.dumps(out), flush=True)

    for base_source in base_sources:
        base_mask = source_masks[base_source]
        base_flat = flatten_mask(base_mask, names)
        base_score_flat = flatten_scores(source_scores[base_source], names)
        candidate_idx = (~base_flat).nonzero(as_tuple=False).flatten()
        labels = torch.zeros_like(base_flat, dtype=torch.bool)
        labels[(~base_flat) & imp_flat] = True
        heldout_idx, heldout_scores, predictor_meta = fit_predictor_scores(
            features=predictor_features,
            labels=labels,
            candidate_idx=candidate_idx,
            heldout_fraction=args.heldout_fraction,
            seed=args.seed * 1009 + len(base_source),
            steps=args.predictor_steps,
            batch_size=args.predictor_batch_size,
            lr=args.predictor_lr,
        )
        evaluate_candidate(
            base_source=base_source,
            variant="base",
            alpha=0.0,
            trial=None,
            mask=base_mask,
            meta={
                "swap_count": 0,
                "base_only_count": int((base_flat & ~imp_flat).sum().item()),
                "imp_only_count": int((~base_flat & imp_flat).sum().item()),
                "candidate_count": int(candidate_idx.numel()),
                "heldout_count": int(heldout_idx.numel()),
                "heldout_positive_count": int(labels[heldout_idx].sum().item()),
                "predictor_auc": predictor_meta["predictor_auc"],
                "predictor_topk_recall": None,
                "predictor_topk_precision": None,
                "predictor_baseline_precision": (
                    float(labels[heldout_idx].float().mean().item())
                    if heldout_idx.numel()
                    else None
                ),
                "added_imp_only_hits": 0,
                "added_imp_only_precision": None,
            },
        )
        for alpha in alphas:
            oracle_mask, oracle_meta = imp_residual_mask(
                base=base_mask,
                imp=imp.mask,
                names=names,
                base_scores=base_score_flat,
                imp_scores=imp_scores,
                alpha=alpha,
            )
            evaluate_candidate(
                base_source=base_source,
                variant="oracle_imp_residual",
                alpha=alpha,
                trial=None,
                mask=oracle_mask,
                meta=oracle_meta,
            )
            pred_mask, pred_meta = predictor_residual_mask(
                base=base_mask,
                imp=imp.mask,
                names=names,
                base_scores=base_score_flat,
                heldout_idx=heldout_idx,
                heldout_scores=heldout_scores,
                predictor_meta=predictor_meta,
                alpha=alpha,
            )
            evaluate_candidate(
                base_source=base_source,
                variant="heldout_predictor_residual",
                alpha=alpha,
                trial=None,
                mask=pred_mask,
                meta=pred_meta,
            )
            for trial in range(args.random_residual_trials):
                random_mask, random_meta = random_heldout_residual_mask(
                    base=base_mask,
                    imp=imp.mask,
                    names=names,
                    base_scores=base_score_flat,
                    heldout_idx=heldout_idx,
                    alpha=alpha,
                    seed=args.seed * 100000 + trial * 1000 + int(round(alpha * 1000)),
                )
                random_meta["predictor_auc"] = predictor_meta["predictor_auc"]
                random_meta["predictor_baseline_precision"] = (
                    float(labels[heldout_idx].float().mean().item())
                    if heldout_idx.numel()
                    else None
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="heldout_random_residual",
                    alpha=alpha,
                    trial=trial,
                    mask=random_mask,
                    meta=random_meta,
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
            "heldout_fraction": args.heldout_fraction,
            "predictor_steps": args.predictor_steps,
            "predictor_batch_size": args.predictor_batch_size,
            "predictor_lr": args.predictor_lr,
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
    with (run_dir / "residual_predictor_mask_probe.csv").open(
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
