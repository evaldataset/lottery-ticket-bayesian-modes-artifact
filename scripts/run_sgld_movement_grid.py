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

from lottery.analysis import cluster_matrix, cluster_states, overlap_rows, summarize_overlaps
from lottery.cyclical_sgld import CyclicalSGLDConfig, collect_cyclical_sgld_samples
from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.diag_laplace import DiagonalLaplaceConfig, collect_diag_laplace_samples
from lottery.imp import iterative_magnitude_pruning
from lottery.kfac_laplace import (
    KFACLaplaceConfig,
    estimate_kfac_factors,
    sample_kfac_laplace_from_factors,
)
from lottery.lowrank_laplace import (
    LowRankLaplaceConfig,
    estimate_lowrank_laplace_factors,
    sample_lowrank_laplace_from_factors,
)
from lottery.masks import global_magnitude_mask_from_state, support_jaccard
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.pruning_baselines import snip_mask, synflow_mask
from lottery.sgld import SGLDConfig, collect_sgld_samples
from lottery.sghmc import SGHMCConfig, collect_sghmc_samples
from lottery.swag import SWAGConfig, fit_swag_posterior, sample_swag_posterior
from lottery.train import (
    evaluate,
    load_trainable_state,
    logits_matrix,
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
    parser.add_argument(
        "--posterior-sampler",
        choices=[
            "sgld",
            "sghmc",
            "cyclical-sgld",
            "diag-laplace",
            "kfac-laplace",
            "lowrank-laplace",
            "swag",
        ],
        default="sgld",
    )
    parser.add_argument("--sgld-lrs", default="1e-10,3e-10,1e-9,3e-9,1e-8")
    parser.add_argument("--laplace-scales", default="1e-10,1e-8,1e-6")
    parser.add_argument("--laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--laplace-variance-floor", type=float, default=1e-12)
    parser.add_argument("--kfac-laplace-scales", default="1e-2,1e0,1e2")
    parser.add_argument("--kfac-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--kfac-laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--kfac-laplace-damping", type=float, default=1e-3)
    parser.add_argument("--kfac-laplace-factor-rows", type=int, default=8192)
    parser.add_argument("--lowrank-laplace-scales", default="1e-4,1e-3,1e-2")
    parser.add_argument("--lowrank-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--lowrank-laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--lowrank-laplace-hessian-batches", type=int, default=2)
    parser.add_argument("--lowrank-laplace-rank", type=int, default=16)
    parser.add_argument("--lowrank-laplace-power-iterations", type=int, default=1)
    parser.add_argument("--lowrank-laplace-oversample", type=int, default=4)
    parser.add_argument("--lowrank-laplace-damping", type=float, default=1e-6)
    parser.add_argument("--lowrank-laplace-variance-floor", type=float, default=1e-12)
    parser.add_argument("--lowrank-laplace-eigenvalue-floor", type=float, default=1e-12)
    parser.add_argument(
        "--lowrank-laplace-batchnorm-mode",
        choices=["eval", "train"],
        default="eval",
    )
    parser.add_argument("--swag-scales", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--swag-epochs", type=int, default=5)
    parser.add_argument("--swag-lr", type=float, default=0.01)
    parser.add_argument("--swag-weight-decay", type=float, default=None)
    parser.add_argument("--swag-collection-start-epoch", type=int, default=1)
    parser.add_argument("--swag-sample-every-epochs", type=int, default=1)
    parser.add_argument("--swag-max-snapshots", type=int, default=20)
    parser.add_argument("--swag-diagonal-scale", type=float, default=1.0)
    parser.add_argument("--swag-low-rank-scale", type=float, default=1.0)
    parser.add_argument("--sgld-temperature", type=float, default=1.0)
    parser.add_argument("--sgld-prior-precision", type=float, default=1e-4)
    parser.add_argument("--sgld-likelihood-scale", choices=["dataset", "mean"], default="dataset")
    parser.add_argument("--sgld-steps", type=int, default=200)
    parser.add_argument("--sgld-burn-in", type=int, default=50)
    parser.add_argument("--sgld-sample-every", type=int, default=10)
    parser.add_argument("--sghmc-momentum-decay", type=float, default=0.9)
    parser.add_argument("--csgld-lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--csgld-cycle-length", type=int, default=50)
    parser.add_argument("--csgld-sample-phase-start", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--snip-batches", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/sgld_movement_grid"))
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

    names = weight_parameter_names(model_factory())
    dense_magnitude_mask = global_magnitude_mask_from_state(
        dense_state,
        names,
        imp.metrics["sparsity"],
    )
    initial_magnitude_mask = global_magnitude_mask_from_state(
        initial_state,
        names,
        imp.metrics["sparsity"],
    )
    rewind_magnitude = (
        support_jaccard(
            global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"]),
            imp.mask,
        )
        if rewind_state is not None
        else None
    )
    snip = snip_mask(
        model_factory(),
        initial_state,
        bundle.train_loader,
        device,
        imp.metrics["sparsity"],
        max_batches=args.snip_batches,
    )
    synflow = synflow_mask(
        model_factory(),
        initial_state,
        bundle.input_shape,
        device,
        imp.metrics["sparsity"],
    )
    chain_start_jaccard = support_jaccard(dense_magnitude_mask, imp.mask)

    rows = []
    if args.posterior_sampler == "diag-laplace":
        sampler_values = parse_float_list(args.laplace_scales)
    elif args.posterior_sampler == "kfac-laplace":
        sampler_values = parse_float_list(args.kfac_laplace_scales)
    elif args.posterior_sampler == "lowrank-laplace":
        sampler_values = parse_float_list(args.lowrank_laplace_scales)
    elif args.posterior_sampler == "swag":
        sampler_values = parse_float_list(args.swag_scales)
    else:
        sampler_values = parse_float_list(args.sgld_lrs)
    shared_kfac_factors = None
    if args.posterior_sampler == "kfac-laplace":
        set_seed(args.seed + 4500)
        kfac_factor_model = model_factory()
        load_trainable_state(kfac_factor_model, dense_state)
        kfac_factor_config = KFACLaplaceConfig(
            num_samples=args.samples,
            scale=sampler_values[0],
            prior_precision=args.kfac_laplace_prior_precision,
            fisher_batches=args.kfac_laplace_fisher_batches,
            damping=args.kfac_laplace_damping,
            factor_sample_rows=args.kfac_laplace_factor_rows,
            num_train_examples=bundle.train_size,
        )
        shared_kfac_factors = estimate_kfac_factors(
            kfac_factor_model,
            bundle.train_loader,
            device,
            kfac_factor_config,
        )
    shared_lowrank_factors = None
    if args.posterior_sampler == "lowrank-laplace":
        set_seed(args.seed + 4500)
        lowrank_factor_model = model_factory()
        load_trainable_state(lowrank_factor_model, dense_state)
        lowrank_factor_config = LowRankLaplaceConfig(
            num_samples=args.samples,
            scale=sampler_values[0],
            prior_precision=args.lowrank_laplace_prior_precision,
            fisher_batches=args.lowrank_laplace_fisher_batches,
            hessian_batches=args.lowrank_laplace_hessian_batches,
            rank=args.lowrank_laplace_rank,
            power_iterations=args.lowrank_laplace_power_iterations,
            oversample=args.lowrank_laplace_oversample,
            damping=args.lowrank_laplace_damping,
            variance_floor=args.lowrank_laplace_variance_floor,
            eigenvalue_floor=args.lowrank_laplace_eigenvalue_floor,
            num_train_examples=bundle.train_size,
            batchnorm_mode=args.lowrank_laplace_batchnorm_mode,
        )
        shared_lowrank_factors = estimate_lowrank_laplace_factors(
            lowrank_factor_model,
            bundle.train_loader,
            device,
            lowrank_factor_config,
            seed=args.seed + 4600,
        )
    shared_swag_posterior = None
    if args.posterior_sampler == "swag":
        set_seed(args.seed + 4500)
        swag_factor_model = model_factory()
        load_trainable_state(swag_factor_model, dense_state)
        swag_fit_config = SWAGConfig(
            epochs=args.swag_epochs,
            lr=args.swag_lr,
            weight_decay=(
                args.weight_decay if args.swag_weight_decay is None else args.swag_weight_decay
            ),
            collection_start_epoch=args.swag_collection_start_epoch,
            sample_every_epochs=args.swag_sample_every_epochs,
            max_snapshots=args.swag_max_snapshots,
            num_samples=args.samples,
            scale=1.0,
            diagonal_scale=args.swag_diagonal_scale,
            low_rank_scale=args.swag_low_rank_scale,
        )
        shared_swag_posterior = fit_swag_posterior(
            swag_factor_model,
            bundle.train_loader,
            device,
            swag_fit_config,
        )
    for config_idx, sampler_value in enumerate(sampler_values):
        set_seed(args.seed + 5000 + config_idx)
        posterior_model = model_factory()
        load_trainable_state(posterior_model, dense_state)
        steps = max(args.sgld_steps, args.sgld_burn_in + args.samples * args.sgld_sample_every)
        swag_snapshot_count = None
        swag_parameter_count = None
        lowrank_positive_rank = None
        lowrank_hessian_eigen_max = None
        lowrank_hessian_eigen_min = None
        lowrank_diag_fisher_mean = None
        lowrank_diag_fisher_max = None
        lowrank_fisher_examples_seen = None
        lowrank_hessian_examples_seen = None
        lowrank_parameter_count = None
        if args.posterior_sampler == "sgld":
            sgld_config = SGLDConfig(
                steps=steps,
                lr=sampler_value,
                temperature=args.sgld_temperature,
                prior_precision=args.sgld_prior_precision,
                burn_in=args.sgld_burn_in,
                sample_every=args.sgld_sample_every,
                num_train_examples=bundle.train_size,
                likelihood_scale=args.sgld_likelihood_scale,
            )
            samples = collect_sgld_samples(
                posterior_model,
                bundle.train_loader,
                device,
                sgld_config,
            )[: args.samples]
        elif args.posterior_sampler == "sghmc":
            sghmc_config = SGHMCConfig(
                steps=steps,
                lr=sampler_value,
                momentum_decay=args.sghmc_momentum_decay,
                temperature=args.sgld_temperature,
                prior_precision=args.sgld_prior_precision,
                burn_in=args.sgld_burn_in,
                sample_every=args.sgld_sample_every,
                num_train_examples=bundle.train_size,
                likelihood_scale=args.sgld_likelihood_scale,
            )
            samples = collect_sghmc_samples(
                posterior_model,
                bundle.train_loader,
                device,
                sghmc_config,
            )[: args.samples]
        elif args.posterior_sampler == "cyclical-sgld":
            csgld_config = CyclicalSGLDConfig(
                steps=steps,
                lr=sampler_value,
                lr_min_ratio=args.csgld_lr_min_ratio,
                cycle_length=args.csgld_cycle_length,
                temperature=args.sgld_temperature,
                prior_precision=args.sgld_prior_precision,
                burn_in=args.sgld_burn_in,
                sample_every=args.sgld_sample_every,
                num_train_examples=bundle.train_size,
                likelihood_scale=args.sgld_likelihood_scale,
                sample_phase_start=args.csgld_sample_phase_start,
            )
            samples = collect_cyclical_sgld_samples(
                posterior_model,
                bundle.train_loader,
                device,
                csgld_config,
            )[: args.samples]
        elif args.posterior_sampler == "diag-laplace":
            laplace_config = DiagonalLaplaceConfig(
                num_samples=args.samples,
                scale=sampler_value,
                prior_precision=args.laplace_prior_precision,
                fisher_batches=args.laplace_fisher_batches,
                variance_floor=args.laplace_variance_floor,
                num_train_examples=bundle.train_size,
            )
            samples = collect_diag_laplace_samples(
                posterior_model,
                bundle.train_loader,
                device,
                laplace_config,
            )
        elif args.posterior_sampler == "kfac-laplace":
            kfac_config = KFACLaplaceConfig(
                num_samples=args.samples,
                scale=sampler_value,
                prior_precision=args.kfac_laplace_prior_precision,
                fisher_batches=args.kfac_laplace_fisher_batches,
                damping=args.kfac_laplace_damping,
                factor_sample_rows=args.kfac_laplace_factor_rows,
                num_train_examples=bundle.train_size,
            )
            if shared_kfac_factors is None:
                raise RuntimeError("missing shared KFAC factors")
            samples = sample_kfac_laplace_from_factors(
                posterior_model,
                shared_kfac_factors,
                kfac_config,
            )
        elif args.posterior_sampler == "lowrank-laplace":
            lowrank_config = LowRankLaplaceConfig(
                num_samples=args.samples,
                scale=sampler_value,
                prior_precision=args.lowrank_laplace_prior_precision,
                fisher_batches=args.lowrank_laplace_fisher_batches,
                hessian_batches=args.lowrank_laplace_hessian_batches,
                rank=args.lowrank_laplace_rank,
                power_iterations=args.lowrank_laplace_power_iterations,
                oversample=args.lowrank_laplace_oversample,
                damping=args.lowrank_laplace_damping,
                variance_floor=args.lowrank_laplace_variance_floor,
                eigenvalue_floor=args.lowrank_laplace_eigenvalue_floor,
                num_train_examples=bundle.train_size,
                batchnorm_mode=args.lowrank_laplace_batchnorm_mode,
            )
            if shared_lowrank_factors is None:
                raise RuntimeError("missing shared LowRank Laplace factors")
            samples = sample_lowrank_laplace_from_factors(
                posterior_model,
                shared_lowrank_factors,
                device,
                lowrank_config,
            )
            lowrank_positive_rank = shared_lowrank_factors.positive_rank
            lowrank_hessian_eigen_max = float(
                shared_lowrank_factors.hessian_eigenvalues.max().item()
            )
            lowrank_hessian_eigen_min = float(
                shared_lowrank_factors.hessian_eigenvalues.min().item()
            )
            lowrank_diag_fisher_mean = float(
                shared_lowrank_factors.diag_fisher.mean().item()
            )
            lowrank_diag_fisher_max = float(
                shared_lowrank_factors.diag_fisher.max().item()
            )
            lowrank_fisher_examples_seen = shared_lowrank_factors.fisher_examples_seen
            lowrank_hessian_examples_seen = shared_lowrank_factors.hessian_examples_seen
            lowrank_parameter_count = shared_lowrank_factors.parameter_count
        elif args.posterior_sampler == "swag":
            swag_config = SWAGConfig(
                epochs=args.swag_epochs,
                lr=args.swag_lr,
                weight_decay=(
                    args.weight_decay
                    if args.swag_weight_decay is None
                    else args.swag_weight_decay
                ),
                collection_start_epoch=args.swag_collection_start_epoch,
                sample_every_epochs=args.swag_sample_every_epochs,
                max_snapshots=args.swag_max_snapshots,
                num_samples=args.samples,
                scale=sampler_value,
                diagonal_scale=args.swag_diagonal_scale,
                low_rank_scale=args.swag_low_rank_scale,
            )
            if shared_swag_posterior is None:
                raise RuntimeError("missing shared SWAG posterior")
            samples = sample_swag_posterior(shared_swag_posterior, swag_config)[
                : args.samples
            ]
            swag_snapshot_count = shared_swag_posterior.snapshot_count
            swag_parameter_count = shared_swag_posterior.parameter_count
        if not samples:
            raise RuntimeError(
                "posterior sampler produced no samples; lower sample phase start "
                "or increase steps/cycles"
            )
        posterior_masks = [
            global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
            for sample in samples
        ]
        overlap_summary = summarize_overlaps(
            overlap_rows(
                posterior_masks,
                imp.mask,
                sparsity=imp.metrics["sparsity"],
                random_trials=args.random_trials,
                seed=args.seed + 20_000 + config_idx,
            )
        )
        posterior_chain = [
            support_jaccard(mask, dense_magnitude_mask) for mask in posterior_masks
        ]
        sample_accuracies = []
        sample_dense_agreements = []
        sample_imp_agreements = []
        sample_logit_features = []
        for sample in samples:
            sample_model = model_factory()
            load_trainable_state(sample_model, sample)
            sample_metrics = evaluate(sample_model, bundle.test_loader, device)
            sample_pred = predictions(sample_model, bundle.test_loader, device)
            sample_logits = logits_matrix(sample_model, bundle.test_loader, device)
            sample_accuracies.append(sample_metrics["accuracy"])
            sample_dense_agreements.append((sample_pred == dense_pred).float().mean().item())
            sample_imp_agreements.append((sample_pred == imp_pred).float().mean().item())
            sample_logit_features.append(sample_logits.flatten().numpy())
        function_clustering = cluster_matrix(np.stack(sample_logit_features, axis=0))
        state_clustering = cluster_states(samples, names)

        rows.append(
            {
                "sgld_lr": sampler_value,
                "posterior_sampler": args.posterior_sampler,
                "temperature": args.sgld_temperature,
                "sghmc_momentum_decay": (
                    args.sghmc_momentum_decay
                    if args.posterior_sampler == "sghmc"
                    else None
                ),
                "csgld_lr_min_ratio": (
                    args.csgld_lr_min_ratio
                    if args.posterior_sampler == "cyclical-sgld"
                    else None
                ),
                "csgld_cycle_length": (
                    args.csgld_cycle_length
                    if args.posterior_sampler == "cyclical-sgld"
                    else None
                ),
                "csgld_sample_phase_start": (
                    args.csgld_sample_phase_start
                    if args.posterior_sampler == "cyclical-sgld"
                    else None
                ),
                "laplace_scale": (
                    sampler_value if args.posterior_sampler == "diag-laplace" else None
                ),
                "laplace_prior_precision": (
                    args.laplace_prior_precision
                    if args.posterior_sampler == "diag-laplace"
                    else None
                ),
                "laplace_fisher_batches": (
                    args.laplace_fisher_batches
                    if args.posterior_sampler == "diag-laplace"
                    else None
                ),
                "kfac_laplace_scale": (
                    sampler_value if args.posterior_sampler == "kfac-laplace" else None
                ),
                "kfac_laplace_prior_precision": (
                    args.kfac_laplace_prior_precision
                    if args.posterior_sampler == "kfac-laplace"
                    else None
                ),
                "kfac_laplace_fisher_batches": (
                    args.kfac_laplace_fisher_batches
                    if args.posterior_sampler == "kfac-laplace"
                    else None
                ),
                "kfac_laplace_damping": (
                    args.kfac_laplace_damping
                    if args.posterior_sampler == "kfac-laplace"
                    else None
                ),
                "kfac_laplace_factor_rows": (
                    args.kfac_laplace_factor_rows
                    if args.posterior_sampler == "kfac-laplace"
                    else None
                ),
                "lowrank_laplace_scale": (
                    sampler_value if args.posterior_sampler == "lowrank-laplace" else None
                ),
                "lowrank_laplace_prior_precision": (
                    args.lowrank_laplace_prior_precision
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_fisher_batches": (
                    args.lowrank_laplace_fisher_batches
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_hessian_batches": (
                    args.lowrank_laplace_hessian_batches
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_rank": (
                    args.lowrank_laplace_rank
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_power_iterations": (
                    args.lowrank_laplace_power_iterations
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_oversample": (
                    args.lowrank_laplace_oversample
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_damping": (
                    args.lowrank_laplace_damping
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_batchnorm_mode": (
                    args.lowrank_laplace_batchnorm_mode
                    if args.posterior_sampler == "lowrank-laplace"
                    else None
                ),
                "lowrank_laplace_positive_rank": lowrank_positive_rank,
                "lowrank_laplace_hessian_eigen_max": lowrank_hessian_eigen_max,
                "lowrank_laplace_hessian_eigen_min": lowrank_hessian_eigen_min,
                "lowrank_laplace_diag_fisher_mean": lowrank_diag_fisher_mean,
                "lowrank_laplace_diag_fisher_max": lowrank_diag_fisher_max,
                "lowrank_laplace_fisher_examples_seen": lowrank_fisher_examples_seen,
                "lowrank_laplace_hessian_examples_seen": lowrank_hessian_examples_seen,
                "lowrank_laplace_parameter_count": lowrank_parameter_count,
                "swag_scale": (
                    sampler_value if args.posterior_sampler == "swag" else None
                ),
                "swag_epochs": (
                    args.swag_epochs if args.posterior_sampler == "swag" else None
                ),
                "swag_lr": (
                    args.swag_lr if args.posterior_sampler == "swag" else None
                ),
                "swag_weight_decay": (
                    args.weight_decay
                    if args.posterior_sampler == "swag"
                    and args.swag_weight_decay is None
                    else (
                        args.swag_weight_decay
                        if args.posterior_sampler == "swag"
                        else None
                    )
                ),
                "swag_collection_start_epoch": (
                    args.swag_collection_start_epoch
                    if args.posterior_sampler == "swag"
                    else None
                ),
                "swag_sample_every_epochs": (
                    args.swag_sample_every_epochs
                    if args.posterior_sampler == "swag"
                    else None
                ),
                "swag_max_snapshots": (
                    args.swag_max_snapshots if args.posterior_sampler == "swag" else None
                ),
                "swag_diagonal_scale": (
                    args.swag_diagonal_scale
                    if args.posterior_sampler == "swag"
                    else None
                ),
                "swag_low_rank_scale": (
                    args.swag_low_rank_scale
                    if args.posterior_sampler == "swag"
                    else None
                ),
                "swag_snapshot_count": swag_snapshot_count,
                "swag_parameter_count": swag_parameter_count,
                "num_samples": len(samples),
                "dense_accuracy": dense_metrics["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
                "posterior_jaccard_mean": overlap_summary["posterior_jaccard_mean"],
                "random_jaccard_mean": overlap_summary["random_jaccard_mean"],
                "posterior_minus_random_jaccard": overlap_summary[
                    "posterior_minus_random_jaccard"
                ],
                "chain_start_magnitude_to_imp_jaccard": chain_start_jaccard,
                "posterior_minus_chain_start_jaccard": (
                    overlap_summary["posterior_jaccard_mean"] - chain_start_jaccard
                ),
                "posterior_to_chain_start_magnitude_jaccard_mean": float(
                    np.mean(posterior_chain)
                ),
                "dense_magnitude_to_imp_jaccard": support_jaccard(
                    dense_magnitude_mask,
                    imp.mask,
                ),
                "initial_magnitude_to_imp_jaccard": support_jaccard(
                    initial_magnitude_mask,
                    imp.mask,
                ),
                "rewind_magnitude_to_imp_jaccard": rewind_magnitude,
                "snip_to_imp_jaccard": support_jaccard(snip, imp.mask),
                "synflow_to_imp_jaccard": support_jaccard(synflow, imp.mask),
                "sample_accuracy_mean": float(np.mean(sample_accuracies)),
                "sample_accuracy_std": sample_std(sample_accuracies),
                "sample_to_dense_prediction_agreement_mean": float(
                    np.mean(sample_dense_agreements)
                ),
                "sample_to_imp_prediction_agreement_mean": float(
                    np.mean(sample_imp_agreements)
                ),
                "state_num_clusters": float(state_clustering["num_clusters"]),
                "function_num_clusters": float(function_clustering["num_clusters"]),
            }
        )

    payload = {
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "posterior_sampler": args.posterior_sampler,
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
    with (run_dir / "movement_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
