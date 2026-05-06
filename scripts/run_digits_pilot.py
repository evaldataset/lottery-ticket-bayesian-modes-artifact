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
from lottery.connectivity import linear_barrier, linear_path_losses
from lottery.cyclical_sgld import CyclicalSGLDConfig, collect_cyclical_sgld_samples
from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import global_magnitude_mask_from_state, support_jaccard
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.posterior_maps import posterior_score_masks
from lottery.pruning_baselines import snip_mask, synflow_mask
from lottery.sgld import SGLDConfig, collect_sgld_samples
from lottery.sghmc import SGHMCConfig, collect_sghmc_samples
from lottery.swag import SWAGConfig, collect_swag_samples
from lottery.train import (
    evaluate,
    load_trainable_state,
    logits_matrix,
    predictions,
    set_seed,
    state_to_cpu,
    train_model,
)


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        choices=["digits", "mnist", "fashion-mnist", "cifar10", "fake-cifar10"],
        default="digits",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--imp-rounds", type=int, default=4)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--sgld-steps", type=int, default=400)
    parser.add_argument("--sgld-lr", type=float, default=1e-7)
    parser.add_argument("--sgld-temperature", type=float, default=1.0)
    parser.add_argument("--sgld-prior-precision", type=float, default=1e-4)
    parser.add_argument("--sgld-likelihood-scale", choices=["dataset", "mean"], default="dataset")
    parser.add_argument("--sgld-burn-in", type=int, default=100)
    parser.add_argument("--sgld-sample-every", type=int, default=10)
    parser.add_argument("--sgld-chains", type=int, default=1)
    parser.add_argument(
        "--sgld-chain-init",
        choices=["dense", "independent-dense"],
        default="dense",
    )
    parser.add_argument(
        "--posterior-sampler",
        choices=["sgld", "sghmc", "cyclical-sgld", "swag"],
        default="sgld",
    )
    parser.add_argument("--sghmc-lr", type=float, default=None)
    parser.add_argument("--sghmc-momentum-decay", type=float, default=0.9)
    parser.add_argument("--sghmc-temperature", type=float, default=None)
    parser.add_argument("--sghmc-prior-precision", type=float, default=None)
    parser.add_argument("--csgld-lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--csgld-cycle-length", type=int, default=50)
    parser.add_argument("--csgld-sample-phase-start", type=float, default=0.0)
    parser.add_argument("--swag-epochs", type=int, default=5)
    parser.add_argument("--swag-lr", type=float, default=0.01)
    parser.add_argument("--swag-weight-decay", type=float, default=None)
    parser.add_argument("--swag-collection-start-epoch", type=int, default=1)
    parser.add_argument("--swag-sample-every-epochs", type=int, default=1)
    parser.add_argument("--swag-max-snapshots", type=int, default=20)
    parser.add_argument("--swag-scale", type=float, default=1.0)
    parser.add_argument("--swag-diagonal-scale", type=float, default=1.0)
    parser.add_argument("--swag-low-rank-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--snip-batches", type=int, default=1)
    parser.add_argument("--barrier-samples", type=int, default=10)
    parser.add_argument("--barrier-points", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = Path("runs") / f"{args.dataset}_pilot"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.imp_epochs is not None and args.imp_epochs <= 0:
        raise ValueError("--imp-epochs must be positive")
    if args.imp_final_epochs is not None and args.imp_final_epochs <= 0:
        raise ValueError("--imp-final-epochs must be positive")
    if args.rewind_epochs < 0:
        raise ValueError("--rewind-epochs must be non-negative")
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

    sgld_config = None
    sghmc_config = None
    csgld_config = None
    swag_config = None
    if args.posterior_sampler == "sgld":
        sgld_config = SGLDConfig(
            steps=max(args.sgld_steps, args.sgld_burn_in + args.samples * args.sgld_sample_every),
            lr=args.sgld_lr,
            temperature=args.sgld_temperature,
            prior_precision=args.sgld_prior_precision,
            burn_in=args.sgld_burn_in,
            sample_every=args.sgld_sample_every,
            num_train_examples=bundle.train_size,
            likelihood_scale=args.sgld_likelihood_scale,
        )
    elif args.posterior_sampler == "sghmc":
        sghmc_config = SGHMCConfig(
            steps=max(args.sgld_steps, args.sgld_burn_in + args.samples * args.sgld_sample_every),
            lr=args.sgld_lr if args.sghmc_lr is None else args.sghmc_lr,
            momentum_decay=args.sghmc_momentum_decay,
            temperature=(
                args.sgld_temperature
                if args.sghmc_temperature is None
                else args.sghmc_temperature
            ),
            prior_precision=(
                args.sgld_prior_precision
                if args.sghmc_prior_precision is None
                else args.sghmc_prior_precision
            ),
            burn_in=args.sgld_burn_in,
            sample_every=args.sgld_sample_every,
            num_train_examples=bundle.train_size,
            likelihood_scale=args.sgld_likelihood_scale,
        )
    elif args.posterior_sampler == "cyclical-sgld":
        csgld_config = CyclicalSGLDConfig(
            steps=max(args.sgld_steps, args.sgld_burn_in + args.samples * args.sgld_sample_every),
            lr=args.sgld_lr,
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
    else:
        swag_config = SWAGConfig(
            epochs=args.swag_epochs,
            lr=args.swag_lr,
            weight_decay=(
                args.weight_decay if args.swag_weight_decay is None else args.swag_weight_decay
            ),
            collection_start_epoch=args.swag_collection_start_epoch,
            sample_every_epochs=args.swag_sample_every_epochs,
            max_snapshots=args.swag_max_snapshots,
            num_samples=args.samples,
            scale=args.swag_scale,
            diagonal_scale=args.swag_diagonal_scale,
            low_rank_scale=args.swag_low_rank_scale,
        )
    samples = []
    chain_start_states = []
    chain_start_metrics = []
    chain_ids = []
    swag_snapshot_counts = []
    swag_parameter_counts = []
    samples_per_chain = max(1, args.samples)
    for chain_idx in range(args.sgld_chains):
        if args.sgld_chain_init == "independent-dense":
            set_seed(args.seed + 1000 + chain_idx)
            chain_dense_model = model_factory()
            train_model(
                chain_dense_model,
                bundle.train_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_schedule=args.lr_schedule,
            )
            chain_start_state = state_to_cpu(chain_dense_model)
            chain_start_metric = evaluate(chain_dense_model, bundle.test_loader, device)
        else:
            chain_start_state = dense_state
            chain_start_metric = dense_metrics

        set_seed(args.seed + 2000 + chain_idx)
        posterior_model = model_factory()
        load_trainable_state(posterior_model, chain_start_state)
        if args.posterior_sampler == "sgld":
            assert sgld_config is not None
            chain_samples = collect_sgld_samples(
                posterior_model,
                bundle.train_loader,
                device,
                sgld_config,
            )[:samples_per_chain]
        elif args.posterior_sampler == "sghmc":
            assert sghmc_config is not None
            chain_samples = collect_sghmc_samples(
                posterior_model,
                bundle.train_loader,
                device,
                sghmc_config,
            )[:samples_per_chain]
        elif args.posterior_sampler == "cyclical-sgld":
            assert csgld_config is not None
            chain_samples = collect_cyclical_sgld_samples(
                posterior_model,
                bundle.train_loader,
                device,
                csgld_config,
            )[:samples_per_chain]
        else:
            assert swag_config is not None
            swag_result = collect_swag_samples(
                posterior_model,
                bundle.train_loader,
                device,
                swag_config,
            )
            chain_samples = swag_result.samples[:samples_per_chain]
            swag_snapshot_counts.append(swag_result.snapshot_count)
            swag_parameter_counts.append(swag_result.parameter_count)
        samples.extend(chain_samples)
        chain_ids.extend([chain_idx] * len(chain_samples))
        chain_start_states.append(chain_start_state)
        chain_start_metrics.append(chain_start_metric)
    if not samples:
        raise RuntimeError(
            "posterior sampler produced no samples; lower sample phase start "
            "or increase steps/cycles"
        )
    names = weight_parameter_names(model_factory())

    posterior_masks = [
        global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
        for sample in samples
    ]
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
    rewind_magnitude_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
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
    chain_start_masks = [
        global_magnitude_mask_from_state(state, names, imp.metrics["sparsity"])
        for state in chain_start_states
    ]
    rows = overlap_rows(
        posterior_masks,
        imp.mask,
        sparsity=imp.metrics["sparsity"],
        random_trials=args.random_trials,
        seed=args.seed + 10_000,
    )
    overlap_summary = summarize_overlaps(rows)
    posterior_dense_jaccards = [
        support_jaccard(mask, dense_magnitude_mask) for mask in posterior_masks
    ]
    posterior_chain_start_jaccards = [
        support_jaccard(mask, chain_start_masks[chain_ids[idx]])
        for idx, mask in enumerate(posterior_masks)
    ]
    chain_start_to_imp_jaccards = [
        support_jaccard(mask, imp.mask) for mask in chain_start_masks
    ]
    aggregate_posterior_masks = posterior_score_masks(samples, names, imp.metrics["sparsity"])
    aggregate_posterior_map_jaccards = {
        map_name: support_jaccard(mask, imp.mask)
        for map_name, mask in aggregate_posterior_masks.items()
    }
    chainwise_posterior_map_jaccards = {}
    for map_name in aggregate_posterior_masks:
        values = []
        for chain_idx in range(args.sgld_chains):
            chain_samples = [
                sample for sample, sample_chain in zip(samples, chain_ids) if sample_chain == chain_idx
            ]
            chain_masks = posterior_score_masks(chain_samples, names, imp.metrics["sparsity"])
            if map_name in chain_masks:
                values.append(support_jaccard(chain_masks[map_name], imp.mask))
        chainwise_posterior_map_jaccards[f"{map_name}_chainwise_mean"] = (
            float(np.mean(values)) if values else 0.0
        )
    cluster_summary = cluster_states(samples, names)

    imp_model = model_factory()
    load_trainable_state(imp_model, imp.final_state)
    dense_pred = predictions(dense_model, bundle.test_loader, device)
    imp_pred = predictions(imp_model, bundle.test_loader, device)
    dense_imp_agreement = (dense_pred == imp_pred).float().mean().item()

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
    dense_imp_losses = linear_path_losses(
        model_factory,
        dense_state,
        imp.final_state,
        bundle.test_loader,
        device,
        points=args.barrier_points,
    )
    dense_sample_barriers = []
    imp_sample_barriers = []
    for sample in samples[: args.barrier_samples]:
        dense_sample_barriers.append(
            linear_barrier(
                linear_path_losses(
                    model_factory,
                    dense_state,
                    sample,
                    bundle.test_loader,
                    device,
                    points=args.barrier_points,
                )
            )
        )
        imp_sample_barriers.append(
            linear_barrier(
                linear_path_losses(
                    model_factory,
                    imp.final_state,
                    sample,
                    bundle.test_loader,
                    device,
                    points=args.barrier_points,
                )
            )
        )

    posterior_metrics = {
        "sampler": args.posterior_sampler,
        "num_samples": len(samples),
        "chains": args.sgld_chains,
        "chain_init": args.sgld_chain_init,
        "chain_start_accuracy_mean": float(
            np.mean([metric["accuracy"] for metric in chain_start_metrics])
        ),
        "sample_accuracy_mean": float(np.mean(sample_accuracies)),
        "sample_accuracy_std": sample_std(sample_accuracies),
        "sample_to_dense_prediction_agreement_mean": float(np.mean(sample_dense_agreements)),
        "sample_to_imp_prediction_agreement_mean": float(np.mean(sample_imp_agreements)),
    }
    if args.posterior_sampler == "sgld":
        assert sgld_config is not None
        posterior_metrics.update(
            {
                "steps": sgld_config.steps,
                "lr": sgld_config.lr,
                "temperature": sgld_config.temperature,
                "prior_precision": sgld_config.prior_precision,
                "likelihood_scale": sgld_config.likelihood_scale,
                "num_train_examples": sgld_config.num_train_examples,
            }
        )
    elif args.posterior_sampler == "sghmc":
        assert sghmc_config is not None
        posterior_metrics.update(
            {
                "steps": sghmc_config.steps,
                "lr": sghmc_config.lr,
                "momentum_decay": sghmc_config.momentum_decay,
                "temperature": sghmc_config.temperature,
                "prior_precision": sghmc_config.prior_precision,
                "likelihood_scale": sghmc_config.likelihood_scale,
                "num_train_examples": sghmc_config.num_train_examples,
            }
        )
    elif args.posterior_sampler == "cyclical-sgld":
        assert csgld_config is not None
        posterior_metrics.update(
            {
                "steps": csgld_config.steps,
                "lr": csgld_config.lr,
                "lr_min_ratio": csgld_config.lr_min_ratio,
                "cycle_length": csgld_config.cycle_length,
                "sample_phase_start": csgld_config.sample_phase_start,
                "temperature": csgld_config.temperature,
                "prior_precision": csgld_config.prior_precision,
                "likelihood_scale": csgld_config.likelihood_scale,
                "num_train_examples": csgld_config.num_train_examples,
            }
        )
    else:
        assert swag_config is not None
        posterior_metrics.update(
            {
                "epochs": swag_config.epochs,
                "lr": swag_config.lr,
                "weight_decay": swag_config.weight_decay,
                "collection_start_epoch": swag_config.collection_start_epoch,
                "sample_every_epochs": swag_config.sample_every_epochs,
                "max_snapshots": swag_config.max_snapshots,
                "scale": swag_config.scale,
                "diagonal_scale": swag_config.diagonal_scale,
                "low_rank_scale": swag_config.low_rank_scale,
                "snapshot_count_mean": float(np.mean(swag_snapshot_counts)),
                "parameter_count": int(swag_parameter_counts[0]) if swag_parameter_counts else 0,
            }
        )

    metrics = {
        "seed": args.seed,
        "dataset": args.dataset,
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
        },
        "rewind": rewind_metrics,
        "dense": dense_metrics,
        "imp": imp.metrics,
        "imp_history": imp.history,
        "posterior_sampler": args.posterior_sampler,
        "posterior": posterior_metrics,
        "sgld": posterior_metrics,
        "posterior_mask_overlap": overlap_summary,
        "controls": {
            "dense_magnitude_to_imp_jaccard": support_jaccard(dense_magnitude_mask, imp.mask),
            "initial_magnitude_to_imp_jaccard": support_jaccard(initial_magnitude_mask, imp.mask),
            "rewind_magnitude_to_imp_jaccard": (
                support_jaccard(rewind_magnitude_mask, imp.mask)
                if rewind_magnitude_mask is not None
                else None
            ),
            "snip_to_imp_jaccard": support_jaccard(snip, imp.mask),
            "synflow_to_imp_jaccard": support_jaccard(synflow, imp.mask),
            "posterior_to_dense_magnitude_jaccard_mean": (
                sum(posterior_dense_jaccards) / len(posterior_dense_jaccards)
                if posterior_dense_jaccards
                else 0.0
            ),
            "posterior_to_chain_start_magnitude_jaccard_mean": (
                sum(posterior_chain_start_jaccards) / len(posterior_chain_start_jaccards)
                if posterior_chain_start_jaccards
                else 0.0
            ),
            "chain_start_magnitude_to_imp_jaccard_mean": (
                sum(chain_start_to_imp_jaccards) / len(chain_start_to_imp_jaccards)
                if chain_start_to_imp_jaccards
                else 0.0
            ),
        },
        "posterior_map_jaccards": {
            **aggregate_posterior_map_jaccards,
            **chainwise_posterior_map_jaccards,
        },
        "state_clustering": cluster_summary,
        "function_clustering": function_clustering,
        "linear_connectivity": {
            "dense_imp_barrier": linear_barrier(dense_imp_losses),
            "dense_imp_losses": dense_imp_losses,
            "dense_sample_barrier_mean": float(np.mean(dense_sample_barriers)),
            "imp_sample_barrier_mean": float(np.mean(imp_sample_barriers)),
            "barrier_samples": min(args.barrier_samples, len(samples)),
            "barrier_points": args.barrier_points,
        },
        "dense_imp_prediction_agreement": dense_imp_agreement,
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (run_dir / "mask_overlaps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "index", "jaccard", "hamming"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
