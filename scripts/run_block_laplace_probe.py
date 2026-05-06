#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.analysis import overlap_rows, summarize_overlaps
from lottery.block_laplace import (
    BlockLaplaceConfig,
    JointBlockLaplaceConfig,
    estimate_block_laplace_factors,
    estimate_joint_block_laplace_factors,
    sample_block_laplace_from_factors,
    sample_joint_block_laplace_from_factors,
)
from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
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


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_joint_block_groups(text: str) -> list[list[str]]:
    groups = [parse_string_list(part) for part in text.split(";")]
    return [group for group in groups if group]


def flatten_groups(groups: list[list[str]]) -> list[str]:
    return [name for group in groups for name in group]


def greedy_joint_groups_under_max(
    parameter_names: list[str],
    parameter_map: dict[str, torch.nn.Parameter],
    max_parameters: int,
) -> list[list[str]]:
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_count = 0
    for name in parameter_names:
        parameter_count = int(parameter_map[name].numel())
        if parameter_count > max_parameters:
            continue
        if current_group and current_count + parameter_count > max_parameters:
            groups.append(current_group)
            current_group = []
            current_count = 0
        current_group.append(name)
        current_count += parameter_count
    if current_group:
        groups.append(current_group)
    return groups


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
    parser.add_argument("--block-name", default="layer1.0.conv1.weight")
    parser.add_argument(
        "--block-names",
        default=None,
        help="Comma-separated block names. Overrides --block-name when provided.",
    )
    parser.add_argument(
        "--joint-block-names",
        default=None,
        help=(
            "Comma-separated block names sampled as one joint full-covariance group. "
            "Overrides --block-name/--block-names when provided."
        ),
    )
    parser.add_argument(
        "--auto-blocks-under-max",
        action="store_true",
        help=(
            "Select all weight tensors whose parameter count is at most "
            "--block-laplace-max-parameters. Overrides manual block names unless "
            "--joint-block-names is set."
        ),
    )
    parser.add_argument(
        "--joint-block-groups",
        default=None,
        help=(
            "Semicolon-separated joint groups, with comma-separated parameter "
            "names inside each group. Groups are sampled independently as one "
            "block-diagonal posterior, while each group uses exact cross-tensor "
            "covariance."
        ),
    )
    parser.add_argument(
        "--auto-joint-groups-under-max",
        action="store_true",
        help=(
            "Select all weight tensors whose parameter count is at most "
            "--block-laplace-max-parameters, greedily pack consecutive tensors "
            "into joint groups under that limit, and sample the groups "
            "independently in one combined network sample."
        ),
    )
    parser.add_argument(
        "--stream-joint-groups",
        action="store_true",
        help=(
            "For joint block groups, estimate one exact group factor at a time, "
            "merge sampled tensors into combined samples, and release the factor "
            "before estimating the next group. This lowers peak host memory for "
            "large block-diagonal joint rows."
        ),
    )
    parser.add_argument(
        "--independent-block-diagonal",
        action="store_true",
        help=(
            "Estimate an exact full-covariance Laplace factor for each requested "
            "tensor, then sample all requested tensors independently in one "
            "combined block-diagonal full-network sample."
        ),
    )
    parser.add_argument("--block-laplace-scales", default="1e-4,1e-3,1e-2")
    parser.add_argument("--block-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--block-laplace-damping", type=float, default=1e-5)
    parser.add_argument("--block-laplace-hessian-batches", type=int, default=2)
    parser.add_argument("--block-laplace-max-parameters", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/block_laplace_probe"))
    return parser.parse_args()


def sample_independent_block_diagonal_laplace(
    factors,
    *,
    scale: float,
    num_samples: int,
) -> list[dict[str, torch.Tensor]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not factors:
        raise ValueError("at least one block factor is required")

    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(num_samples):
        sample = {
            name: value.detach().clone()
            for name, value in factors[0].base_state.items()
        }
        for factor in factors:
            chol_t = factor.precision_cholesky.t()
            noise = torch.randn(factor.parameter_count, dtype=torch.float64)
            delta = torch.linalg.solve_triangular(
                chol_t,
                noise.reshape(-1, 1),
                upper=True,
            ).reshape(-1)
            vector = factor.mean + delta * (scale**0.5)
            reference = factor.base_state[factor.parameter_name]
            sample[factor.parameter_name] = vector.reshape(factor.parameter_shape).to(
                dtype=reference.dtype
            )
        samples.append(sample)
    return samples


def sample_independent_joint_block_diagonal_laplace(
    factors,
    *,
    scale: float,
    num_samples: int,
) -> list[dict[str, torch.Tensor]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not factors:
        raise ValueError("at least one joint block factor is required")

    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(num_samples):
        sample = {
            name: value.detach().clone()
            for name, value in factors[0].base_state.items()
        }
        for factor in factors:
            chol_t = factor.precision_cholesky.t()
            noise = torch.randn(factor.parameter_count, dtype=torch.float64)
            delta = torch.linalg.solve_triangular(
                chol_t,
                noise.reshape(-1, 1),
                upper=True,
            ).reshape(-1)
            vector = factor.mean + delta * (scale**0.5)
            for name in factor.parameter_names:
                start, end = factor.parameter_slices[name]
                reference = factor.base_state[name]
                sample[name] = vector[start:end].reshape(
                    factor.parameter_shapes[name]
                ).to(dtype=reference.dtype)
        samples.append(sample)
    return samples


def make_base_samples(
    base_state: dict[str, torch.Tensor],
    num_samples: int,
) -> list[dict[str, torch.Tensor]]:
    return [
        {name: value.detach().clone() for name, value in base_state.items()}
        for _ in range(num_samples)
    ]


def cleanup_laplace_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_bundle(args: argparse.Namespace):
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args)

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

    scales = parse_float_list(args.block_laplace_scales)
    if not scales:
        raise ValueError("at least one block Laplace scale is required")
    joint_mode = args.joint_block_names is not None
    joint_group_mode = (
        args.joint_block_groups is not None or args.auto_joint_groups_under_max
    )
    if joint_mode and args.independent_block_diagonal:
        raise ValueError("--joint-block-names cannot be combined with --independent-block-diagonal")
    if joint_mode and joint_group_mode:
        raise ValueError("--joint-block-names cannot be combined with joint block groups")
    if joint_group_mode and args.independent_block_diagonal:
        raise ValueError(
            "joint block groups cannot be combined with --independent-block-diagonal"
        )
    if args.stream_joint_groups and not joint_group_mode:
        raise ValueError("--stream-joint-groups requires joint block groups")
    if args.auto_blocks_under_max and args.auto_joint_groups_under_max:
        raise ValueError("--auto-blocks-under-max cannot be combined with joint auto groups")
    parameter_map = dict(model_factory().named_parameters())
    if joint_mode:
        block_groups = [parse_string_list(args.joint_block_names)]
    elif args.joint_block_groups is not None:
        block_groups = parse_joint_block_groups(args.joint_block_groups)
    elif args.auto_joint_groups_under_max:
        block_groups = greedy_joint_groups_under_max(
            weight_parameter_names(model_factory()),
            parameter_map,
            args.block_laplace_max_parameters,
        )
    elif args.auto_blocks_under_max:
        requested_block_names = [
            name
            for name in weight_parameter_names(model_factory())
            if int(parameter_map[name].numel()) <= args.block_laplace_max_parameters
        ]
        block_groups = (
            [requested_block_names]
            if args.independent_block_diagonal
            else [[name] for name in requested_block_names]
        )
    else:
        requested_block_names = (
            parse_string_list(args.block_names) if args.block_names else [args.block_name]
        )
        block_groups = (
            [requested_block_names]
            if args.independent_block_diagonal
            else [[name] for name in requested_block_names]
        )
    requested_block_names = flatten_groups(block_groups)
    if not requested_block_names:
        raise ValueError("at least one block name is required")
    duplicate_blocks = sorted(
        {name for name in requested_block_names if requested_block_names.count(name) > 1}
    )
    if duplicate_blocks:
        raise ValueError(f"duplicate block name(s): {duplicate_blocks}")
    unknown_blocks = [name for name in requested_block_names if name not in parameter_map]
    if unknown_blocks:
        raise ValueError(f"unknown block name(s): {unknown_blocks}")
    oversize_groups = [
        group
        for group in block_groups
        if sum(int(parameter_map[name].numel()) for name in group)
        > args.block_laplace_max_parameters
    ]
    if oversize_groups and (joint_mode or joint_group_mode):
        raise ValueError(
            "joint block group exceeds --block-laplace-max-parameters: "
            f"{oversize_groups[0]}"
        )
    missing_blocks = [name for name in requested_block_names if name not in imp.mask]
    if missing_blocks:
        raise ValueError(f"IMP mask does not contain block(s): {missing_blocks}")

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

    combined_joint_group_mode = joint_group_mode
    rows = []
    evaluation_groups = [requested_block_names] if combined_joint_group_mode else block_groups
    for block_idx, block_names in enumerate(evaluation_groups):
        if combined_joint_group_mode:
            block_label = (
                f"jointdiag:{len(block_groups)}groups<={args.block_laplace_max_parameters}"
            )
        elif args.independent_block_diagonal:
            block_label = (
                f"blockdiag:{len(block_names)}blocks<={args.block_laplace_max_parameters}"
            )
        else:
            block_label = (
                "joint:" + "+".join(name.removesuffix(".weight") for name in block_names)
                if len(block_names) > 1
                else block_names[0]
            )
        imp_block_mask = {name: imp.mask[name] for name in block_names}
        block_total = sum(mask.numel() for mask in imp_block_mask.values())
        block_kept = sum(mask.float().sum().item() for mask in imp_block_mask.values())
        block_keep = float(block_kept / block_total)
        block_sparsity = 1.0 - block_keep
        dense_block_mask = global_magnitude_mask_from_state(
            dense_state,
            block_names,
            block_sparsity,
        )
        initial_block_mask = global_magnitude_mask_from_state(
            initial_state,
            block_names,
            block_sparsity,
        )
        rewind_block_mask = (
            global_magnitude_mask_from_state(rewind_state, block_names, block_sparsity)
            if rewind_state is not None
            else None
        )

        set_seed(args.seed + 7500 + block_idx)
        factor_model = model_factory()
        load_trainable_state(factor_model, dense_state)
        streamed_samples_by_scale = None
        streamed_parameter_count = None
        streamed_examples_seen = None
        streamed_hessian_scale = None
        if combined_joint_group_mode and args.stream_joint_groups:
            base_state = state_to_cpu(factor_model)
            streamed_samples_by_scale = {
                scale: make_base_samples(base_state, args.samples) for scale in scales
            }
            factor_parameter_counts = []
            factor_examples_seen = []
            factor_hessian_scales = []
            for factor_idx, group_names in enumerate(block_groups):
                set_seed(args.seed + 7500 + block_idx * 1000 + factor_idx)
                factor = estimate_joint_block_laplace_factors(
                    factor_model,
                    bundle.train_loader,
                    device,
                    JointBlockLaplaceConfig(
                        parameter_names=tuple(group_names),
                        num_samples=args.samples,
                        scale=scales[0],
                        prior_precision=args.block_laplace_prior_precision,
                        damping=args.block_laplace_damping,
                        hessian_batches=args.block_laplace_hessian_batches,
                        num_train_examples=bundle.train_size,
                        max_parameters=args.block_laplace_max_parameters,
                    ),
                )
                factor_parameter_counts.append(factor.parameter_count)
                factor_examples_seen.append(factor.examples_seen)
                factor_hessian_scales.append(factor.hessian_scale)
                for config_idx, scale in enumerate(scales):
                    set_seed(args.seed + 8000 + 1000 * block_idx + 100 * config_idx + factor_idx)
                    config = JointBlockLaplaceConfig(
                        parameter_names=tuple(group_names),
                        num_samples=args.samples,
                        scale=scale,
                        prior_precision=args.block_laplace_prior_precision,
                        damping=args.block_laplace_damping,
                        hessian_batches=args.block_laplace_hessian_batches,
                        num_train_examples=bundle.train_size,
                        max_parameters=args.block_laplace_max_parameters,
                    )
                    group_samples = sample_joint_block_laplace_from_factors(factor, config)
                    combined_samples = streamed_samples_by_scale[scale]
                    for sample_idx, group_sample in enumerate(group_samples):
                        for name in group_names:
                            combined_samples[sample_idx][name] = (
                                group_sample[name].detach().clone()
                            )
                    del group_samples
                del factor
                cleanup_laplace_memory()
            streamed_parameter_count = sum(factor_parameter_counts)
            streamed_examples_seen = min(factor_examples_seen)
            streamed_hessian_scale = float(np.mean(factor_hessian_scales))
        elif combined_joint_group_mode:
            factors = []
            for factor_idx, group_names in enumerate(block_groups):
                set_seed(args.seed + 7500 + block_idx * 1000 + factor_idx)
                factors.append(
                    estimate_joint_block_laplace_factors(
                        factor_model,
                        bundle.train_loader,
                        device,
                        JointBlockLaplaceConfig(
                            parameter_names=tuple(group_names),
                            num_samples=args.samples,
                            scale=scales[0],
                            prior_precision=args.block_laplace_prior_precision,
                            damping=args.block_laplace_damping,
                            hessian_batches=args.block_laplace_hessian_batches,
                            num_train_examples=bundle.train_size,
                            max_parameters=args.block_laplace_max_parameters,
                        ),
                    )
                )
        elif args.independent_block_diagonal:
            factors = []
            for factor_idx, block_name in enumerate(block_names):
                set_seed(args.seed + 7500 + block_idx * 1000 + factor_idx)
                factors.append(
                    estimate_block_laplace_factors(
                        factor_model,
                        bundle.train_loader,
                        device,
                        BlockLaplaceConfig(
                            parameter_name=block_name,
                            num_samples=args.samples,
                            scale=scales[0],
                            prior_precision=args.block_laplace_prior_precision,
                            damping=args.block_laplace_damping,
                            hessian_batches=args.block_laplace_hessian_batches,
                            num_train_examples=bundle.train_size,
                            max_parameters=args.block_laplace_max_parameters,
                        ),
                    )
                )
        elif len(block_names) == 1:
            factors = estimate_block_laplace_factors(
                factor_model,
                bundle.train_loader,
                device,
                BlockLaplaceConfig(
                    parameter_name=block_names[0],
                    num_samples=args.samples,
                    scale=scales[0],
                    prior_precision=args.block_laplace_prior_precision,
                    damping=args.block_laplace_damping,
                    hessian_batches=args.block_laplace_hessian_batches,
                    num_train_examples=bundle.train_size,
                    max_parameters=args.block_laplace_max_parameters,
                ),
            )
        else:
            factors = estimate_joint_block_laplace_factors(
                factor_model,
                bundle.train_loader,
                device,
                JointBlockLaplaceConfig(
                    parameter_names=tuple(block_names),
                    num_samples=args.samples,
                    scale=scales[0],
                    prior_precision=args.block_laplace_prior_precision,
                    damping=args.block_laplace_damping,
                    hessian_batches=args.block_laplace_hessian_batches,
                    num_train_examples=bundle.train_size,
                    max_parameters=args.block_laplace_max_parameters,
                ),
            )

        for config_idx, scale in enumerate(scales):
            set_seed(args.seed + 8000 + 1000 * block_idx + config_idx)
            if combined_joint_group_mode and args.stream_joint_groups:
                if (
                    streamed_samples_by_scale is None
                    or streamed_parameter_count is None
                    or streamed_examples_seen is None
                    or streamed_hessian_scale is None
                ):
                    raise RuntimeError("streamed joint-group samples were not initialized")
                samples = streamed_samples_by_scale[scale]
                parameter_count = streamed_parameter_count
                examples_seen = streamed_examples_seen
                hessian_scale = streamed_hessian_scale
            elif combined_joint_group_mode:
                samples = sample_independent_joint_block_diagonal_laplace(
                    factors,
                    scale=scale,
                    num_samples=args.samples,
                )
                parameter_count = sum(factor.parameter_count for factor in factors)
                examples_seen = min(factor.examples_seen for factor in factors)
                hessian_scale = float(np.mean([factor.hessian_scale for factor in factors]))
            elif args.independent_block_diagonal:
                samples = sample_independent_block_diagonal_laplace(
                    factors,
                    scale=scale,
                    num_samples=args.samples,
                )
                parameter_count = sum(factor.parameter_count for factor in factors)
                examples_seen = min(factor.examples_seen for factor in factors)
                hessian_scale = float(np.mean([factor.hessian_scale for factor in factors]))
            elif len(block_names) == 1:
                config = BlockLaplaceConfig(
                    parameter_name=block_names[0],
                    num_samples=args.samples,
                    scale=scale,
                    prior_precision=args.block_laplace_prior_precision,
                    damping=args.block_laplace_damping,
                    hessian_batches=args.block_laplace_hessian_batches,
                    num_train_examples=bundle.train_size,
                    max_parameters=args.block_laplace_max_parameters,
                )
                samples = sample_block_laplace_from_factors(factors, config)
                parameter_count = factors.parameter_count
                examples_seen = factors.examples_seen
                hessian_scale = factors.hessian_scale
            else:
                config = JointBlockLaplaceConfig(
                    parameter_names=tuple(block_names),
                    num_samples=args.samples,
                    scale=scale,
                    prior_precision=args.block_laplace_prior_precision,
                    damping=args.block_laplace_damping,
                    hessian_batches=args.block_laplace_hessian_batches,
                    num_train_examples=bundle.train_size,
                    max_parameters=args.block_laplace_max_parameters,
                )
                samples = sample_joint_block_laplace_from_factors(factors, config)
                parameter_count = factors.parameter_count
                examples_seen = factors.examples_seen
                hessian_scale = factors.hessian_scale
            block_masks = [
                global_magnitude_mask_from_state(sample, block_names, block_sparsity)
                for sample in samples
            ]
            global_masks = [
                global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
                for sample in samples
            ]
            block_overlap = summarize_overlaps(
                overlap_rows(
                    block_masks,
                    imp_block_mask,
                    sparsity=block_sparsity,
                    random_trials=args.random_trials,
                    seed=args.seed + 41_000 + 1000 * block_idx + config_idx,
                )
            )
            global_overlap = summarize_overlaps(
                overlap_rows(
                    global_masks,
                    imp.mask,
                    sparsity=imp.metrics["sparsity"],
                    random_trials=args.random_trials,
                    seed=args.seed + 51_000 + 1000 * block_idx + config_idx,
                )
            )
            block_posterior_chain = [
                support_jaccard(mask, dense_block_mask) for mask in block_masks
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
                    "block_laplace_scale": scale,
                    "block_name": block_label,
                    "block_parameter_count": parameter_count,
                    "block_count": len(block_names),
                    "block_group_count": len(block_groups) if combined_joint_group_mode else 1,
                    "block_parameter_fraction": float(parameter_count / sum(
                        dense_state[name].numel() for name in names
                    )),
                    "block_examples_seen": examples_seen,
                    "block_hessian_scale": hessian_scale,
                    "num_samples": len(samples),
                    "dense_accuracy": dense_metrics["accuracy"],
                    "imp_accuracy": imp.metrics["accuracy"],
                    "imp_sparsity_global": imp.metrics["sparsity"],
                    "imp_sparsity_block": block_sparsity,
                    "block_posterior_jaccard_mean": block_overlap["posterior_jaccard_mean"],
                    "block_random_jaccard_mean": block_overlap["random_jaccard_mean"],
                    "block_posterior_minus_random_jaccard": block_overlap[
                        "posterior_minus_random_jaccard"
                    ],
                    "block_chain_start_magnitude_to_imp_jaccard": support_jaccard(
                        dense_block_mask,
                        imp_block_mask,
                    ),
                    "block_posterior_minus_chain_start_jaccard": (
                        block_overlap["posterior_jaccard_mean"]
                        - support_jaccard(dense_block_mask, imp_block_mask)
                    ),
                    "block_posterior_to_chain_start_magnitude_jaccard_mean": float(
                        np.mean(block_posterior_chain)
                    ),
                    "block_initial_magnitude_to_imp_jaccard": support_jaccard(
                        initial_block_mask,
                        imp_block_mask,
                    ),
                    "block_rewind_magnitude_to_imp_jaccard": (
                        support_jaccard(rewind_block_mask, imp_block_mask)
                        if rewind_block_mask is not None
                        else None
                    ),
                    "global_posterior_jaccard_mean": global_overlap[
                        "posterior_jaccard_mean"
                    ],
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
                    "sample_to_imp_prediction_agreement_mean": float(
                        np.mean(sample_imp_agreements)
                    ),
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
        "block_laplace": {
            "block_names": requested_block_names,
            "block_groups": block_groups,
            "joint_mode": joint_mode,
            "joint_block_diagonal": combined_joint_group_mode,
            "independent_block_diagonal": args.independent_block_diagonal,
            "auto_blocks_under_max": args.auto_blocks_under_max,
            "auto_joint_groups_under_max": args.auto_joint_groups_under_max,
            "stream_joint_groups": args.stream_joint_groups,
            "scales": scales,
            "prior_precision": args.block_laplace_prior_precision,
            "damping": args.block_laplace_damping,
            "hessian_batches": args.block_laplace_hessian_batches,
            "max_parameters": args.block_laplace_max_parameters,
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
    with (run_dir / "block_laplace_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
