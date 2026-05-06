#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.data import (  # noqa: E402
    DatasetBundle,
    load_digits_bundle,
    load_fake_cifar10_bundle,
    load_torchvision_bundle,
)
from lottery.imp import iterative_magnitude_pruning  # noqa: E402
from lottery.masks import apply_mask_, random_mask_like  # noqa: E402
from lottery.models import MLP, ResNetCIFAR, TinyCNN  # noqa: E402
from lottery.pruning_baselines import (  # noqa: E402
    gem_miner_mask,
    hard_concrete_mask,
    variational_pruning_mask,
)
from lottery.swag import SWAGConfig, collect_swag_samples  # noqa: E402
from lottery.train import (  # noqa: E402
    evaluate,
    load_trainable_state,
    set_seed,
    state_to_cpu,
    train_model,
)


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
    parser.add_argument(
        "--ood-dataset",
        choices=["cifar100", "svhn", "gaussian-noise", "fake-cifar10"],
        default="cifar100",
    )
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="resnet20")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=1)
    parser.add_argument("--imp-rounds", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--ood-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--swag-epochs", type=int, default=5)
    parser.add_argument("--swag-lr", type=float, default=0.01)
    parser.add_argument("--swag-weight-decay", type=float, default=None)
    parser.add_argument("--swag-collection-start-epoch", type=int, default=1)
    parser.add_argument("--swag-sample-every-epochs", type=int, default=1)
    parser.add_argument("--swag-max-snapshots", type=int, default=20)
    parser.add_argument("--swag-scale", type=float, default=1.0)
    parser.add_argument("--swag-diagonal-scale", type=float, default=1.0)
    parser.add_argument("--swag-low-rank-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--learned-mask-sources", default="")
    parser.add_argument("--learned-random-trials", type=int, default=1)
    parser.add_argument("--mask-train-epochs", type=int, default=None)
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/calibration_ood_probe"),
    )
    return parser.parse_args()


def load_bundle(args: argparse.Namespace) -> DatasetBundle:
    if args.dataset == "digits":
        return load_digits_bundle(args.batch_size, 1024, args.seed)
    if args.dataset == "fake-cifar10":
        return load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            args.seed,
            train_size=args.train_subset or 2048,
            test_size=args.test_subset or 512,
        )
    return load_torchvision_bundle(
        args.dataset,
        args.batch_size,
        1024,
        args.seed,
        flatten=args.model == "mlp",
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        augment=args.augment,
    )


def load_ood_loader(args: argparse.Namespace, input_shape: tuple[int, ...]) -> DataLoader:
    if args.ood_dataset == "fake-cifar10":
        return load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            args.seed + 91_000,
            train_size=args.train_subset or 2048,
            test_size=args.ood_subset or args.test_subset or 512,
        ).test_loader

    if args.ood_dataset == "gaussian-noise":
        count = args.ood_subset or args.test_subset or 512
        generator = torch.Generator().manual_seed(args.seed + 92_000)
        x = torch.randn((count, *input_shape), generator=generator)
        y = torch.zeros(count, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=False)

    if len(input_shape) != 3 or input_shape[0] != 3:
        raise ValueError("CIFAR100/SVHN OOD loaders require 3x32x32 image inputs")

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for CIFAR100/SVHN OOD loaders") from exc

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.ood_dataset == "cifar100":
        dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    elif args.ood_dataset == "svhn":
        dataset = datasets.SVHN(root="data", split="test", download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported OOD dataset: {args.ood_dataset}")

    if args.ood_subset is not None:
        dataset = Subset(dataset, list(range(min(args.ood_subset, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def make_model_factory(
    args: argparse.Namespace,
    bundle: DatasetBundle,
) -> Any:
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


@torch.no_grad()
def probabilities_and_labels(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()
    probs = []
    labels = []
    for x, y in data_loader:
        logits = model(x.to(device))
        probs.append(F.softmax(logits, dim=1).cpu())
        labels.append(y.cpu().long())
    return torch.cat(probs, dim=0), torch.cat(labels, dim=0)


def ensemble_probabilities(
    model_factory: Any,
    states: list[dict[str, torch.Tensor]],
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prob_sum: torch.Tensor | None = None
    labels_ref: torch.Tensor | None = None
    for state in states:
        model = model_factory()
        load_trainable_state(model, state)
        probs, labels = probabilities_and_labels(model, data_loader, device)
        prob_sum = probs if prob_sum is None else prob_sum + probs
        if labels_ref is None:
            labels_ref = labels
    if prob_sum is None or labels_ref is None:
        raise ValueError("cannot evaluate an empty ensemble")
    return prob_sum / len(states), labels_ref


def ece_score(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    bins: int,
) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = confidences.numel()
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if int(mask.sum()) == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc = correct[mask].float().mean().item()
        ece += (int(mask.sum()) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def id_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    bins: int,
) -> dict[str, float]:
    eps = 1e-12
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)
    one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
    nll = -torch.log(probs[torch.arange(labels.numel()), labels].clamp_min(eps)).mean()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(nll.item()),
        "brier": float(brier.item()),
        "ece": ece_score(confidences, correct, bins),
        "confidence": float(confidences.mean().item()),
    }


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = float(np.quantile(id_scores, 0.05))
    return float(np.mean(ood_scores >= threshold))


def ood_metrics(
    id_probs: torch.Tensor,
    ood_probs: torch.Tensor,
) -> dict[str, float]:
    id_msp = id_probs.max(dim=1).values.numpy()
    ood_msp = ood_probs.max(dim=1).values.numpy()
    id_neg_entropy = (-entropy(id_probs)).numpy()
    ood_neg_entropy = (-entropy(ood_probs)).numpy()
    labels = np.concatenate([np.ones_like(id_msp), np.zeros_like(ood_msp)])

    def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        return float(roc_auc_score(labels, np.concatenate([id_scores, ood_scores])))

    def aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        return float(average_precision_score(labels, np.concatenate([id_scores, ood_scores])))

    return {
        "msp_auroc": auroc(id_msp, ood_msp),
        "msp_aupr": aupr(id_msp, ood_msp),
        "msp_fpr95": fpr_at_95_tpr(id_msp, ood_msp),
        "entropy_auroc": auroc(id_neg_entropy, ood_neg_entropy),
        "entropy_aupr": aupr(id_neg_entropy, ood_neg_entropy),
        "entropy_fpr95": fpr_at_95_tpr(id_neg_entropy, ood_neg_entropy),
        "id_msp": float(id_msp.mean()),
        "ood_msp": float(ood_msp.mean()),
        "id_entropy": float(entropy(id_probs).mean().item()),
        "ood_entropy": float(entropy(ood_probs).mean().item()),
    }


def evaluate_prob_source(
    id_probs: torch.Tensor,
    id_labels: torch.Tensor,
    ood_probs: torch.Tensor,
    bins: int,
) -> dict[str, dict[str, float]]:
    return {
        "id": id_metrics(id_probs, id_labels, bins),
        "ood": ood_metrics(id_probs, ood_probs),
    }


def evaluate_state_source(
    model_factory: Any,
    state: dict[str, torch.Tensor],
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: torch.device,
    bins: int,
) -> dict[str, dict[str, float]]:
    model = model_factory()
    load_trainable_state(model, state)
    id_probs, id_labels = probabilities_and_labels(model, id_loader, device)
    ood_probs, _ = probabilities_and_labels(model, ood_loader, device)
    return evaluate_prob_source(id_probs, id_labels, ood_probs, bins)


def train_fixed_mask_state(
    model_factory: Any,
    train_state: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str,
) -> dict[str, torch.Tensor]:
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
    return state_to_cpu(model)


def main() -> None:
    args = parse_args()
    if args.imp_epochs is not None and args.imp_epochs <= 0:
        raise ValueError("--imp-epochs must be positive")
    if args.imp_final_epochs is not None and args.imp_final_epochs <= 0:
        raise ValueError("--imp-final-epochs must be positive")
    if args.mask_train_epochs is not None and args.mask_train_epochs <= 0:
        raise ValueError("--mask-train-epochs must be positive")
    if args.learned_random_trials < 0:
        raise ValueError("--learned-random-trials must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args)
    ood_loader = load_ood_loader(args, bundle.input_shape)
    model_factory = make_model_factory(args, bundle)

    initial_model = model_factory()
    initial_state = state_to_cpu(initial_model)

    rewind_state = None
    rewind_metrics = None
    if args.rewind_epochs > 0:
        rewind_model = model_factory()
        load_trainable_state(rewind_model, initial_state)
        train_model(
            rewind_model,
            bundle.train_loader,
            device,
            epochs=args.rewind_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
            lr_schedule_epochs=args.epochs,
        )
        rewind_metrics = evaluate(rewind_model, bundle.test_loader, device)
        rewind_state = state_to_cpu(rewind_model)

    dense_model = model_factory()
    load_trainable_state(dense_model, initial_state)
    train_model(
        dense_model,
        bundle.train_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
    )
    dense_state = state_to_cpu(dense_model)
    dense_metrics = evaluate(dense_model, bundle.test_loader, device)

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

    swag_config = SWAGConfig(
        epochs=args.swag_epochs,
        lr=args.swag_lr,
        weight_decay=args.weight_decay if args.swag_weight_decay is None else args.swag_weight_decay,
        collection_start_epoch=args.swag_collection_start_epoch,
        sample_every_epochs=args.swag_sample_every_epochs,
        max_snapshots=args.swag_max_snapshots,
        num_samples=args.samples,
        scale=args.swag_scale,
        diagonal_scale=args.swag_diagonal_scale,
        low_rank_scale=args.swag_low_rank_scale,
    )
    set_seed(args.seed + 20_000)
    swag_model = model_factory()
    load_trainable_state(swag_model, dense_state)
    swag_result = collect_swag_samples(
        swag_model,
        bundle.train_loader,
        device,
        swag_config,
    )
    samples = swag_result.samples[: args.samples]
    if not samples:
        raise RuntimeError("SWAG produced no samples")

    id_probs, id_labels = ensemble_probabilities(
        model_factory,
        samples,
        bundle.test_loader,
        device,
    )
    ood_probs, _ = ensemble_probabilities(model_factory, samples, ood_loader, device)

    member_id_metrics = []
    member_ood_metrics = []
    for sample in samples:
        source = evaluate_state_source(
            model_factory,
            sample,
            bundle.test_loader,
            ood_loader,
            device,
            args.ece_bins,
        )
        member_id_metrics.append(source["id"])
        member_ood_metrics.append(source["ood"])

    def mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        keys = sorted(metrics[0])
        return {key: float(np.mean([row[key] for row in metrics])) for key in keys}

    sources = {
        "dense": evaluate_state_source(
            model_factory,
            dense_state,
            bundle.test_loader,
            ood_loader,
            device,
            args.ece_bins,
        ),
        "imp": evaluate_state_source(
            model_factory,
            imp.final_state,
            bundle.test_loader,
            ood_loader,
            device,
            args.ece_bins,
        ),
        "swag_ensemble": evaluate_prob_source(id_probs, id_labels, ood_probs, args.ece_bins),
        "swag_member_mean": {
            "id": mean_metrics(member_id_metrics),
            "ood": mean_metrics(member_ood_metrics),
        },
    }
    learned_masks: dict[str, dict[str, float | str]] = {}
    learned_mask_sources = parse_source_list(args.learned_mask_sources)
    mask_train_epochs = args.mask_train_epochs or imp_epochs
    if learned_mask_sources:
        for source in learned_mask_sources:
            masks = []
            if source == "random":
                for trial in range(args.learned_random_trials):
                    masks.append(
                        (
                            f"learned_random_{trial}",
                            random_mask_like(
                                imp.mask,
                                sparsity=imp.metrics["sparsity"],
                                seed=args.seed * 10_000 + trial,
                            ),
                        )
                    )
            elif source == "gem_miner":
                set_seed(args.seed + 70_000)
                masks.append(
                    (
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
                masks.append(
                    (
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
                            max_batches_per_epoch=(
                                args.variational_prune_max_batches_per_epoch
                            ),
                        ),
                    )
                )
            elif source == "hard_concrete":
                set_seed(args.seed + 90_000)
                masks.append(
                    (
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
                            max_batches_per_epoch=(
                                args.hard_concrete_max_batches_per_epoch
                            ),
                        ),
                    )
                )
            else:
                raise ValueError(
                    "unknown learned mask source "
                    f"{source!r}; expected random, gem_miner, "
                    "variational_prune, or hard_concrete"
                )

            for source_name, mask in masks:
                learned_state = train_fixed_mask_state(
                    model_factory=model_factory,
                    train_state=initial_state,
                    mask=mask,
                    train_loader=bundle.train_loader,
                    device=device,
                    epochs=mask_train_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lr_schedule=args.lr_schedule,
                )
                sources[source_name] = evaluate_state_source(
                    model_factory,
                    learned_state,
                    bundle.test_loader,
                    ood_loader,
                    device,
                    args.ece_bins,
                )
                learned_masks[source_name] = {
                    "source": source_name,
                    "train_state": "initial",
                    "mask_train_epochs": float(mask_train_epochs),
                    "mask_sparsity": float(imp.metrics["sparsity"]),
                }

    metrics = {
        "seed": args.seed,
        "dataset": args.dataset,
        "ood_dataset": args.ood_dataset,
        "model": args.model,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "imp_epochs": imp_epochs,
            "imp_final_epochs": (
                imp_epochs if args.imp_final_epochs is None else args.imp_final_epochs
            ),
            "rewind_epochs": args.rewind_epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "augment": args.augment,
            "train_size": bundle.train_size,
            "test_size": bundle.test_size,
            "learned_mask_sources": learned_mask_sources,
            "learned_random_trials": args.learned_random_trials,
            "mask_train_epochs": mask_train_epochs,
            "gem_miner_epochs": args.gem_miner_epochs,
            "gem_miner_lr": args.gem_miner_lr,
            "gem_miner_regularization": args.gem_miner_regularization,
            "gem_miner_freeze_period": args.gem_miner_freeze_period,
            "gem_miner_max_batches_per_epoch": args.gem_miner_max_batches_per_epoch,
            "variational_prune_epochs": args.variational_prune_epochs,
            "variational_prune_lr": args.variational_prune_lr,
            "variational_prune_kl_weight": args.variational_prune_kl_weight,
            "variational_prune_sparsity_weight": (
                args.variational_prune_sparsity_weight
            ),
            "variational_prune_entropy_weight": args.variational_prune_entropy_weight,
            "variational_prune_temperature_start": (
                args.variational_prune_temperature_start
            ),
            "variational_prune_temperature_end": (
                args.variational_prune_temperature_end
            ),
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
        "rewind": rewind_metrics,
        "dense": dense_metrics,
        "imp": imp.metrics,
        "swag": {
            "epochs": swag_config.epochs,
            "lr": swag_config.lr,
            "weight_decay": swag_config.weight_decay,
            "collection_start_epoch": swag_config.collection_start_epoch,
            "sample_every_epochs": swag_config.sample_every_epochs,
            "max_snapshots": swag_config.max_snapshots,
            "scale": swag_config.scale,
            "diagonal_scale": swag_config.diagonal_scale,
            "low_rank_scale": swag_config.low_rank_scale,
            "num_samples": len(samples),
            "snapshot_count": swag_result.snapshot_count,
            "parameter_count": swag_result.parameter_count,
        },
        "learned_masks": learned_masks,
        "sources": sources,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
