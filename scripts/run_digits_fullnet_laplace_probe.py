#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.data import load_digits_bundle, load_fake_cifar10_bundle
from lottery.full_laplace import (
    FullLaplaceConfig,
    estimate_full_laplace_factors,
    sample_full_laplace_from_factors,
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


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "fake-cifar10"], default="digits")
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cnn-width", type=int, default=2)
    parser.add_argument("--resnet-width", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--imp-rounds", type=int, default=2)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--full-laplace-scales", default="1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--full-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--full-laplace-damping", type=float, default=1e-5)
    parser.add_argument("--full-laplace-hessian-batches", type=int, default=None)
    parser.add_argument("--full-laplace-max-parameters", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/digits_fullnet_laplace_probe"))
    return parser.parse_args()


def evaluate_state(
    model_factory,
    state: dict[str, torch.Tensor],
    test_loader,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    model = model_factory()
    load_trainable_state(model, state)
    return evaluate(model, test_loader, device), predictions(model, test_loader, device)


def prediction_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left == right).float().mean().item())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "digits":
        bundle = load_digits_bundle(args.batch_size, args.test_batch_size, args.seed)
    else:
        bundle = load_fake_cifar10_bundle(
            args.batch_size,
            args.test_batch_size,
            args.seed,
            train_size=args.train_size,
            test_size=args.test_size,
        )

    def model_factory() -> torch.nn.Module:
        if args.model == "mlp":
            if args.dataset != "digits":
                raise ValueError("MLP fullnet probe currently expects flattened digits")
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
        raise ValueError(f"unsupported model: {args.model}")

    initial_model = model_factory()
    initial_state = state_to_cpu(initial_model)
    rewind_state = None
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
    dense_pred = predictions(dense_model, bundle.test_loader, device)
    dense_state = state_to_cpu(dense_model)

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
    imp_metrics, imp_pred = evaluate_state(model_factory, imp.final_state, bundle.test_loader, device)

    scales = parse_float_list(args.full_laplace_scales)
    if not scales:
        raise ValueError("at least one full Laplace scale is required")

    set_seed(args.seed + 7000)
    factor_model = model_factory()
    load_trainable_state(factor_model, dense_state)
    factors = estimate_full_laplace_factors(
        factor_model,
        bundle.train_loader,
        device,
        FullLaplaceConfig(
            num_samples=args.samples,
            scale=scales[0],
            prior_precision=args.full_laplace_prior_precision,
            damping=args.full_laplace_damping,
            hessian_batches=args.full_laplace_hessian_batches,
            num_train_examples=bundle.train_size,
            max_parameters=args.full_laplace_max_parameters,
        ),
    )

    names = weight_parameter_names(model_factory())
    dense_mask = global_magnitude_mask_from_state(dense_state, names, imp.metrics["sparsity"])
    initial_mask = global_magnitude_mask_from_state(
        initial_state,
        names,
        imp.metrics["sparsity"],
    )
    rewind_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for scale_idx, scale in enumerate(scales):
        set_seed(args.seed + 8000 + scale_idx)
        config = FullLaplaceConfig(
            num_samples=args.samples,
            scale=scale,
            prior_precision=args.full_laplace_prior_precision,
            damping=args.full_laplace_damping,
            hessian_batches=args.full_laplace_hessian_batches,
            num_train_examples=bundle.train_size,
            max_parameters=args.full_laplace_max_parameters,
        )
        samples = sample_full_laplace_from_factors(factors, config)
        posterior_masks = [
            global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
            for sample in samples
        ]
        posterior_to_imp = [support_jaccard(mask, imp.mask) for mask in posterior_masks]
        posterior_to_dense = [support_jaccard(mask, dense_mask) for mask in posterior_masks]
        sample_accuracies = []
        sample_dense_agreements = []
        sample_imp_agreements = []
        for sample in samples:
            metrics, pred = evaluate_state(model_factory, sample, bundle.test_loader, device)
            sample_accuracies.append(metrics["accuracy"])
            sample_dense_agreements.append(prediction_agreement(pred, dense_pred))
            sample_imp_agreements.append(prediction_agreement(pred, imp_pred))

        rows.append(
            {
                "full_laplace_scale": scale,
                "dense_accuracy": dense_metrics["accuracy"],
                "imp_accuracy": imp_metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
                "parameter_count": factors.parameter_count,
                "weight_parameter_count": int(
                    sum(dense_state[name].numel() for name in names)
                ),
                "examples_seen": factors.examples_seen,
                "hessian_scale": factors.hessian_scale,
                "posterior_jaccard_mean": mean(posterior_to_imp),
                "posterior_jaccard_std": std(posterior_to_imp),
                "chain_start_magnitude_to_imp_jaccard": support_jaccard(
                    dense_mask,
                    imp.mask,
                ),
                "posterior_minus_chain_start_jaccard": mean(posterior_to_imp)
                - support_jaccard(dense_mask, imp.mask),
                "posterior_to_chain_start_magnitude_jaccard_mean": mean(
                    posterior_to_dense
                ),
                "posterior_to_chain_start_magnitude_jaccard_std": std(
                    posterior_to_dense
                ),
                "initial_magnitude_to_imp_jaccard": support_jaccard(initial_mask, imp.mask),
                "rewind_magnitude_to_imp_jaccard": (
                    support_jaccard(rewind_mask, imp.mask)
                    if rewind_mask is not None
                    else float("nan")
                ),
                "sample_accuracy_mean": mean(sample_accuracies),
                "sample_accuracy_std": std(sample_accuracies),
                "sample_to_dense_prediction_agreement_mean": mean(sample_dense_agreements),
                "sample_to_imp_prediction_agreement_mean": mean(sample_imp_agreements),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "config": vars(args) | {"out_dir": str(args.out_dir)},
                "dense": dense_metrics,
                "imp": imp_metrics,
                "full_laplace": {
                    "parameter_count": factors.parameter_count,
                    "parameter_names": list(factors.parameter_names),
                    "examples_seen": factors.examples_seen,
                    "hessian_scale": factors.hessian_scale,
                    "precision_cholesky_shape": list(factors.precision_cholesky.shape),
                },
                "rows": rows,
                "imp_history": imp.history,
            },
            f,
            indent=2,
        )

    with (run_dir / "fullnet_laplace_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"run_dir": str(run_dir), "rows": len(rows)}))


if __name__ == "__main__":
    main()
