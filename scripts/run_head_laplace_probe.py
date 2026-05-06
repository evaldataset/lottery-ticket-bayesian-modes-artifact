#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.analysis import overlap_rows, summarize_overlaps
from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.head_laplace import (
    HeadLaplaceConfig,
    estimate_head_laplace_factors,
    sample_head_laplace_from_factors,
)
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import global_magnitude_mask_from_state, support_jaccard
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.train import (
    evaluate,
    load_trainable_state,
    predictions,
    set_seed,
    state_to_cpu,
    train_model,
)


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--imp-rounds", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--head-name", default=None)
    parser.add_argument("--head-laplace-scales", default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--head-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--head-laplace-damping", type=float, default=1e-5)
    parser.add_argument("--head-laplace-hessian-batches", type=int, default=None)
    parser.add_argument("--head-laplace-max-parameters", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/head_laplace_probe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    dense_metrics = evaluate(dense_model, bundle.test_loader, device)
    dense_state = state_to_cpu(dense_model)
    dense_pred = predictions(dense_model, bundle.test_loader, device)

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
    imp_model = model_factory()
    load_trainable_state(imp_model, imp.final_state)
    imp_pred = predictions(imp_model, bundle.test_loader, device)

    scales = parse_float_list(args.head_laplace_scales)
    if not scales:
        raise ValueError("at least one head Laplace scale is required")
    set_seed(args.seed + 5500)
    factor_model = model_factory()
    load_trainable_state(factor_model, dense_state)
    factors = estimate_head_laplace_factors(
        factor_model,
        bundle.train_loader,
        device,
        HeadLaplaceConfig(
            num_samples=args.samples,
            scale=scales[0],
            prior_precision=args.head_laplace_prior_precision,
            damping=args.head_laplace_damping,
            hessian_batches=args.head_laplace_hessian_batches,
            num_train_examples=bundle.train_size,
            max_parameters=args.head_laplace_max_parameters,
            head_name=args.head_name,
        ),
    )

    names = weight_parameter_names(model_factory())
    dense_global_mask = global_magnitude_mask_from_state(
        dense_state,
        names,
        imp.metrics["sparsity"],
    )
    initial_global_mask = global_magnitude_mask_from_state(
        initial_state,
        names,
        imp.metrics["sparsity"],
    )
    rewind_global_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else None
    )
    if factors.weight_name not in imp.mask:
        raise ValueError(f"IMP mask does not contain head weight {factors.weight_name!r}")
    head_names = [factors.weight_name]
    imp_head_mask = {factors.weight_name: imp.mask[factors.weight_name]}
    head_keep = float(imp.mask[factors.weight_name].float().mean().item())
    head_sparsity = 1.0 - head_keep
    dense_head_mask = global_magnitude_mask_from_state(dense_state, head_names, head_sparsity)
    initial_head_mask = global_magnitude_mask_from_state(initial_state, head_names, head_sparsity)
    rewind_head_mask = (
        global_magnitude_mask_from_state(rewind_state, head_names, head_sparsity)
        if rewind_state is not None
        else None
    )

    rows = []
    for config_idx, scale in enumerate(scales):
        set_seed(args.seed + 6000 + config_idx)
        config = HeadLaplaceConfig(
            num_samples=args.samples,
            scale=scale,
            prior_precision=args.head_laplace_prior_precision,
            damping=args.head_laplace_damping,
            hessian_batches=args.head_laplace_hessian_batches,
            num_train_examples=bundle.train_size,
            max_parameters=args.head_laplace_max_parameters,
            head_name=args.head_name,
        )
        samples = sample_head_laplace_from_factors(factors, config)
        head_masks = [
            global_magnitude_mask_from_state(sample, head_names, head_sparsity)
            for sample in samples
        ]
        global_masks = [
            global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
            for sample in samples
        ]
        head_overlap = summarize_overlaps(
            overlap_rows(
                head_masks,
                imp_head_mask,
                sparsity=head_sparsity,
                random_trials=args.random_trials,
                seed=args.seed + 21_000 + config_idx,
            )
        )
        global_overlap = summarize_overlaps(
            overlap_rows(
                global_masks,
                imp.mask,
                sparsity=imp.metrics["sparsity"],
                random_trials=args.random_trials,
                seed=args.seed + 31_000 + config_idx,
            )
        )
        head_posterior_chain = [
            support_jaccard(mask, dense_head_mask) for mask in head_masks
        ]
        global_posterior_chain = [
            support_jaccard(mask, dense_global_mask) for mask in global_masks
        ]
        sample_accuracies = []
        sample_dense_agreements = []
        sample_imp_agreements = []
        for sample in samples:
            sample_model = model_factory()
            load_trainable_state(sample_model, sample)
            sample_metrics = evaluate(sample_model, bundle.test_loader, device)
            sample_pred = predictions(sample_model, bundle.test_loader, device)
            sample_accuracies.append(sample_metrics["accuracy"])
            sample_dense_agreements.append((sample_pred == dense_pred).float().mean().item())
            sample_imp_agreements.append((sample_pred == imp_pred).float().mean().item())

        rows.append(
            {
                "head_laplace_scale": scale,
                "head_name": factors.head_name,
                "head_weight_name": factors.weight_name,
                "head_parameter_count": factors.parameter_count,
                "head_examples_seen": factors.examples_seen,
                "head_hessian_scale": factors.hessian_scale,
                "num_samples": len(samples),
                "dense_accuracy": dense_metrics["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity_global": imp.metrics["sparsity"],
                "imp_sparsity_head": head_sparsity,
                "head_posterior_jaccard_mean": head_overlap["posterior_jaccard_mean"],
                "head_random_jaccard_mean": head_overlap["random_jaccard_mean"],
                "head_posterior_minus_random_jaccard": head_overlap[
                    "posterior_minus_random_jaccard"
                ],
                "head_chain_start_magnitude_to_imp_jaccard": support_jaccard(
                    dense_head_mask,
                    imp_head_mask,
                ),
                "head_posterior_minus_chain_start_jaccard": (
                    head_overlap["posterior_jaccard_mean"]
                    - support_jaccard(dense_head_mask, imp_head_mask)
                ),
                "head_posterior_to_chain_start_magnitude_jaccard_mean": float(
                    np.mean(head_posterior_chain)
                ),
                "head_initial_magnitude_to_imp_jaccard": support_jaccard(
                    initial_head_mask,
                    imp_head_mask,
                ),
                "head_rewind_magnitude_to_imp_jaccard": (
                    support_jaccard(rewind_head_mask, imp_head_mask)
                    if rewind_head_mask is not None
                    else None
                ),
                "global_posterior_jaccard_mean": global_overlap["posterior_jaccard_mean"],
                "global_random_jaccard_mean": global_overlap["random_jaccard_mean"],
                "global_posterior_minus_random_jaccard": global_overlap[
                    "posterior_minus_random_jaccard"
                ],
                "global_chain_start_magnitude_to_imp_jaccard": support_jaccard(
                    dense_global_mask,
                    imp.mask,
                ),
                "global_posterior_minus_chain_start_jaccard": (
                    global_overlap["posterior_jaccard_mean"]
                    - support_jaccard(dense_global_mask, imp.mask)
                ),
                "global_posterior_to_chain_start_magnitude_jaccard_mean": float(
                    np.mean(global_posterior_chain)
                ),
                "global_initial_magnitude_to_imp_jaccard": support_jaccard(
                    initial_global_mask,
                    imp.mask,
                ),
                "global_rewind_magnitude_to_imp_jaccard": (
                    support_jaccard(rewind_global_mask, imp.mask)
                    if rewind_global_mask is not None
                    else None
                ),
                "sample_accuracy_mean": float(np.mean(sample_accuracies)),
                "sample_accuracy_std": sample_std(sample_accuracies),
                "sample_to_dense_prediction_agreement_mean": float(
                    np.mean(sample_dense_agreements)
                ),
                "sample_to_imp_prediction_agreement_mean": float(np.mean(sample_imp_agreements)),
            }
        )

    payload = {
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "imp_epochs": imp_epochs,
            "imp_final_epochs": args.imp_final_epochs,
            "rewind_epochs": args.rewind_epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "augment": args.augment,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
        },
        "head_laplace": {
            "head_name": factors.head_name,
            "weight_name": factors.weight_name,
            "bias_name": factors.bias_name,
            "parameter_count": factors.parameter_count,
            "examples_seen": factors.examples_seen,
            "hessian_scale": factors.hessian_scale,
            "prior_precision": args.head_laplace_prior_precision,
            "damping": args.head_laplace_damping,
            "hessian_batches": args.head_laplace_hessian_batches,
        },
        "rewind": rewind_metrics,
        "dense": dense_metrics,
        "imp": imp.metrics,
        "imp_history": imp.history,
        "rows": rows,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (run_dir / "head_laplace_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
