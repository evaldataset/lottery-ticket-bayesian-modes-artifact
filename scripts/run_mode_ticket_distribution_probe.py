#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.block_laplace import (
    JointBlockLaplaceConfig,
    estimate_joint_block_laplace_factors,
    sample_joint_block_laplace_from_factors,
)
from lottery.cyclical_sgld import CyclicalSGLDConfig, collect_cyclical_sgld_samples
from lottery.data import load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import iterative_magnitude_pruning
from lottery.lowrank_laplace import LowRankLaplaceConfig, collect_lowrank_laplace_samples
from lottery.masks import Mask, global_magnitude_mask_from_state, mask_sparsity
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.sghmc import SGHMCConfig, collect_sghmc_samples
from lottery.sgld import SGLDConfig, collect_sgld_samples
from lottery.swag import SWAGConfig, collect_swag_samples
from lottery.train import (
    evaluate,
    load_trainable_state,
    logits_matrix,
    set_seed,
    state_to_cpu,
)


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=parse_int_list("0,1,2,3,4"),
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        choices=["digits", "mnist", "fashion-mnist", "cifar10", "fake-cifar10"],
        default="digits",
    )
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--imp-rounds", type=int, default=3)
    parser.add_argument("--prune-fraction", type=float, default=0.3)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument(
        "--posterior-sampler",
        choices=[
            "sgld",
            "sghmc",
            "cyclical-sgld",
            "lowrank-laplace",
            "jointdiag-laplace",
            "swag",
        ],
        default="sgld",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sgld-steps", type=int, default=120)
    parser.add_argument("--sgld-lr", type=float, default=1e-7)
    parser.add_argument("--sgld-temperature", type=float, default=1.0)
    parser.add_argument("--sgld-prior-precision", type=float, default=1e-4)
    parser.add_argument("--sgld-likelihood-scale", choices=["dataset", "mean"], default="dataset")
    parser.add_argument("--sgld-burn-in", type=int, default=20)
    parser.add_argument("--sgld-sample-every", type=int, default=10)
    parser.add_argument(
        "--posterior-chains",
        "--sgld-chains",
        dest="posterior_chains",
        type=int,
        default=1,
        help="Number of posterior chains to collect per seed.",
    )
    parser.add_argument(
        "--posterior-chain-init",
        "--sgld-chain-init",
        dest="posterior_chain_init",
        choices=["dense", "independent-dense"],
        default="dense",
        help=(
            "Use the seed's dense model for every posterior chain, or train an "
            "independent dense start for each chain."
        ),
    )
    parser.add_argument("--sghmc-lr", type=float, default=None)
    parser.add_argument("--sghmc-momentum-decay", type=float, default=0.9)
    parser.add_argument("--sghmc-temperature", type=float, default=None)
    parser.add_argument("--sghmc-prior-precision", type=float, default=None)
    parser.add_argument("--csgld-lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--csgld-cycle-length", type=int, default=50)
    parser.add_argument("--csgld-sample-phase-start", type=float, default=0.0)
    parser.add_argument("--lowrank-laplace-scale", type=float, default=1e-2)
    parser.add_argument("--lowrank-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--lowrank-laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--lowrank-laplace-hessian-batches", type=int, default=2)
    parser.add_argument("--lowrank-laplace-rank", type=int, default=128)
    parser.add_argument("--lowrank-laplace-power-iterations", type=int, default=1)
    parser.add_argument("--lowrank-laplace-oversample", type=int, default=32)
    parser.add_argument("--lowrank-laplace-damping", type=float, default=1e-6)
    parser.add_argument("--lowrank-laplace-variance-floor", type=float, default=1e-12)
    parser.add_argument("--lowrank-laplace-eigenvalue-floor", type=float, default=1e-12)
    parser.add_argument(
        "--lowrank-laplace-batchnorm-mode",
        choices=["eval", "train"],
        default="eval",
    )
    parser.add_argument("--jointdiag-laplace-scale", type=float, default=1e-6)
    parser.add_argument("--jointdiag-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--jointdiag-laplace-damping", type=float, default=1e-5)
    parser.add_argument("--jointdiag-laplace-hessian-batches", type=int, default=1)
    parser.add_argument("--jointdiag-laplace-max-parameters", type=int, default=40000)
    parser.add_argument("--swag-epochs", type=int, default=5)
    parser.add_argument("--swag-lr", type=float, default=0.01)
    parser.add_argument("--swag-weight-decay", type=float, default=None)
    parser.add_argument("--swag-collection-start-epoch", type=int, default=1)
    parser.add_argument("--swag-sample-every-epochs", type=int, default=1)
    parser.add_argument("--swag-max-snapshots", type=int, default=20)
    parser.add_argument("--swag-scale", type=float, default=1.0)
    parser.add_argument("--swag-diagonal-scale", type=float, default=1.0)
    parser.add_argument("--swag-low-rank-scale", type=float, default=1.0)
    parser.add_argument("--cluster-pca-dim", type=int, default=20)
    parser.add_argument("--sliced-projections", type=int, default=128)
    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument(
        "--alignment-method",
        choices=["none", "activation", "weight"],
        default="none",
        help=(
            "Optionally add channel-aligned ResNet mask comparisons against the "
            "first seed dense model coordinate frame. Activation alignment uses "
            "held-out activation correlations; weight alignment uses "
            "incoming/outgoing weight correlations."
        ),
    )
    parser.add_argument("--alignment-batches", type=int, default=20)
    parser.add_argument(
        "--save-mask-artifacts",
        action="store_true",
        help=(
            "Write flattened raw mask matrices and record ids to "
            "mask_artifacts.npz for downstream permutation/graph audits."
        ),
    )
    parser.add_argument(
        "--save-state-artifacts",
        action="store_true",
        help=(
            "Also write flattened trainable state matrices to mask_artifacts.npz. "
            "This can be large on full CIFAR runs and implies --save-mask-artifacts."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def make_bundle(args: argparse.Namespace):
    if args.dataset == "digits":
        return load_digits_bundle(args.batch_size, 1024, args.data_seed)
    if args.dataset == "fake-cifar10":
        return load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            args.data_seed,
            train_size=args.train_subset or 512,
            test_size=args.test_subset or 128,
        )
    return load_torchvision_bundle(
        args.dataset,
        args.batch_size,
        1024,
        args.data_seed,
        flatten=args.model == "mlp",
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        augment=args.augment,
    )


def make_model_factory(args: argparse.Namespace, bundle):
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


def flatten_state(state: dict[str, torch.Tensor], names: list[str]) -> np.ndarray:
    return np.concatenate(
        [state[name].detach().cpu().reshape(-1).numpy() for name in names],
        axis=0,
    )


def flatten_mask(mask: Mask, names: list[str]) -> np.ndarray:
    return np.concatenate(
        [mask[name].detach().cpu().bool().reshape(-1).numpy() for name in names],
        axis=0,
    )


def parameter_sizes(names: list[str], state_or_mask: dict[str, torch.Tensor]) -> list[int]:
    return [int(state_or_mask[name].numel()) for name in names]


def parameter_shapes(
    names: list[str],
    state_or_mask: dict[str, torch.Tensor],
) -> dict[str, list[int]]:
    return {name: [int(dim) for dim in state_or_mask[name].shape] for name in names}


def parameter_offsets(sizes: list[int]) -> list[int]:
    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    return offsets


def mask_matrix(masks: list[Mask], names: list[str]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack([flatten_mask(mask, names).astype(np.uint8) for mask in masks], axis=0)


def state_matrix(states: list[np.ndarray]) -> np.ndarray:
    if not states:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([state.astype(np.float32, copy=False) for state in states], axis=0)


def layer_sparsity_vector(mask: Mask, names: list[str]) -> np.ndarray:
    values = []
    for name in names:
        keep = mask[name].detach().cpu().bool().float().mean().item()
        values.append(1.0 - keep)
    return np.asarray(values, dtype=np.float64)


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


def resnet_block_input_key(layer_idx: int, block_idx: int) -> str:
    if block_idx > 0:
        return f"layer{layer_idx}.{block_idx - 1}.out"
    if layer_idx == 1:
        return "stem"
    return f"layer{layer_idx - 1}.2.out"


def resnet_weight_axes(name: str) -> tuple[str | None, str | None]:
    if name == "conv1.weight":
        return "stem", None
    if name == "fc.weight":
        return None, "layer3.2.out"
    parts = name.split(".")
    if len(parts) == 4 and parts[0].startswith("layer") and parts[2] in {"conv1", "conv2"}:
        layer_idx = int(parts[0].removeprefix("layer"))
        block_idx = int(parts[1])
        if parts[2] == "conv1":
            return (
                f"layer{layer_idx}.{block_idx}.conv1",
                resnet_block_input_key(layer_idx, block_idx),
            )
        return (
            f"layer{layer_idx}.{block_idx}.out",
            f"layer{layer_idx}.{block_idx}.conv1",
        )
    if len(parts) == 5 and parts[0].startswith("layer") and parts[2] == "shortcut":
        layer_idx = int(parts[0].removeprefix("layer"))
        block_idx = int(parts[1])
        return (
            f"layer{layer_idx}.{block_idx}.out",
            resnet_block_input_key(layer_idx, block_idx),
        )
    raise ValueError(f"unsupported ResNet weight name for alignment: {name}")


def collect_resnet_activation_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    if max_batches <= 0:
        raise ValueError("alignment_batches must be positive")

    features: dict[str, list[torch.Tensor]] = {}
    handles = []

    def add_feature(key: str, value: torch.Tensor) -> None:
        pooled = value.detach().float()
        if pooled.ndim == 4:
            pooled = pooled.mean(dim=(2, 3))
        if pooled.ndim != 2:
            raise ValueError(f"expected 2D/4D activation for {key}, got {tuple(value.shape)}")
        features.setdefault(key, []).append(pooled.cpu())

    def hook(key: str, relu: bool):
        def _hook(_module, _inputs, output):
            add_feature(key, F.relu(output) if relu else output)

        return _hook

    handles.append(model.bn1.register_forward_hook(hook("stem", relu=True)))
    for layer_idx in [1, 2, 3]:
        layer = getattr(model, f"layer{layer_idx}")
        for block_idx, block in enumerate(layer):
            handles.append(
                block.bn1.register_forward_hook(
                    hook(f"layer{layer_idx}.{block_idx}.conv1", relu=True)
                )
            )
            handles.append(
                block.register_forward_hook(
                    hook(f"layer{layer_idx}.{block_idx}.out", relu=False)
                )
            )

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            model(x.to(device))

    for handle in handles:
        handle.remove()
    return {key: torch.cat(values, dim=0) for key, values in features.items()}


def activation_channel_alignment(
    source_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    maps: dict[str, torch.Tensor] = {}
    correlations = []
    for key, source_value in sorted(source_features.items()):
        target_value = target_features[key]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"alignment feature shape mismatch for {key}: "
                f"{tuple(source_value.shape)} vs {tuple(target_value.shape)}"
            )
        source_norm = source_value - source_value.mean(dim=0, keepdim=True)
        target_norm = target_value - target_value.mean(dim=0, keepdim=True)
        source_norm = source_norm / source_norm.norm(dim=0, keepdim=True).clamp_min(1e-12)
        target_norm = target_norm / target_norm.norm(dim=0, keepdim=True).clamp_min(1e-12)
        corr = source_norm.t().matmul(target_norm)
        source_idx, target_idx = linear_sum_assignment((-corr).numpy())
        perm = torch.empty(source_value.shape[1], dtype=torch.long)
        perm[torch.as_tensor(source_idx, dtype=torch.long)] = torch.as_tensor(
            target_idx,
            dtype=torch.long,
        )
        maps[key] = perm
        selected = corr[source_idx, target_idx]
        correlations.extend(float(value) for value in selected)
    if not correlations:
        raise ValueError("activation alignment produced no channel maps")
    corr_tensor = torch.tensor(correlations, dtype=torch.float32)
    return maps, {
        "alignment_map_count": len(maps),
        "alignment_channel_count": int(corr_tensor.numel()),
        "alignment_mean_corr": float(corr_tensor.mean().item()),
        "alignment_min_corr": float(corr_tensor.min().item()),
    }


def _weight_input_channel_view(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.t().reshape(value.shape[1], -1)
    if value.ndim == 4:
        return value.permute(1, 0, 2, 3).reshape(value.shape[1], -1)
    raise ValueError(f"unsupported weight tensor shape for input alignment: {tuple(value.shape)}")


def collect_resnet_weight_features(
    state: dict[str, torch.Tensor],
    names: list[str],
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {}
    for name in names:
        value = state[name].detach().cpu().float()
        out_key, in_key = resnet_weight_axes(name)
        if out_key is not None:
            chunks.setdefault(out_key, []).append(value.reshape(value.shape[0], -1))
        if in_key is not None:
            chunks.setdefault(in_key, []).append(_weight_input_channel_view(value))
    features: dict[str, torch.Tensor] = {}
    for key, values in chunks.items():
        channel_count = values[0].shape[0]
        if any(value.shape[0] != channel_count for value in values):
            raise ValueError(f"weight alignment feature channel mismatch for {key}")
        # `activation_channel_alignment` expects observations x channels.
        features[key] = torch.cat(values, dim=1).t().contiguous()
    if not features:
        raise ValueError("weight alignment produced no channel features")
    return features


def collect_alignment_features(
    *,
    method: str,
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    names: list[str],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    if method == "activation":
        return collect_resnet_activation_features(model, loader, device, max_batches)
    if method == "weight":
        return collect_resnet_weight_features(state, names)
    raise ValueError(f"unsupported alignment method: {method}")


def _align_weight_tensor(
    value: torch.Tensor,
    name: str,
    channel_maps: dict[str, torch.Tensor],
) -> torch.Tensor:
    out_key, in_key = resnet_weight_axes(name)
    aligned = value.detach().cpu()
    if out_key is not None:
        target = torch.zeros_like(aligned)
        target.index_copy_(0, channel_maps[out_key], aligned)
        aligned = target
    if in_key is not None:
        target = torch.zeros_like(aligned)
        target.index_copy_(1, channel_maps[in_key], aligned)
        aligned = target
    return aligned


def aligned_weight_mask(
    mask: Mask,
    names: list[str],
    channel_maps: dict[str, torch.Tensor],
) -> Mask:
    return {
        name: _align_weight_tensor(mask[name].bool(), name, channel_maps).bool()
        for name in names
    }


def aligned_weight_state(
    state: dict[str, torch.Tensor],
    names: list[str],
    channel_maps: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: _align_weight_tensor(state[name], name, channel_maps) for name in names}


def empirical_overlap(
    left: np.ndarray,
    right: np.ndarray,
    bins: int,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    if left.size == 0 or right.size == 0:
        return math.nan
    left_hist, _ = np.histogram(left, bins=bins, range=value_range)
    right_hist, _ = np.histogram(right, bins=bins, range=value_range)
    if left_hist.sum() == 0 or right_hist.sum() == 0:
        return math.nan
    left_prob = left_hist.astype(np.float64) / left_hist.sum()
    right_prob = right_hist.astype(np.float64) / right_hist.sum()
    return float(np.minimum(left_prob, right_prob).sum())


def ks_values(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.size == 0 or right.size == 0:
        return {"ks_statistic": math.nan, "ks_pvalue": math.nan}
    result = ks_2samp(left, right, method="auto")
    return {
        "ks_statistic": float(result.statistic),
        "ks_pvalue": float(result.pvalue),
    }


def pairwise_hamming(flat_masks: list[np.ndarray]) -> np.ndarray:
    values = []
    for i in range(len(flat_masks)):
        for j in range(i + 1, len(flat_masks)):
            values.append(float(np.mean(flat_masks[i] != flat_masks[j])))
    return np.asarray(values, dtype=np.float64)


def cross_hamming(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    values = []
    for left_mask in left:
        for right_mask in right:
            values.append(float(np.mean(left_mask != right_mask)))
    return np.asarray(values, dtype=np.float64)


def median_bandwidth(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 1.0
    diffs = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 1e-12]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_mmd(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.size == 0 or right.size == 0:
        return {"mmd_rbf": math.nan, "mmd_bandwidth": math.nan}
    combined = np.concatenate([left, right], axis=0)
    bandwidth = median_bandwidth(combined)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diff * diff, axis=-1))

    k_xx = kernel(left, left).mean()
    k_yy = kernel(right, right).mean()
    k_xy = kernel(left, right).mean()
    return {
        "mmd_rbf": float(max(0.0, k_xx + k_yy - 2.0 * k_xy)),
        "mmd_bandwidth": bandwidth,
    }


def sliced_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    projections: int,
    seed: int,
) -> float:
    if left.size == 0 or right.size == 0:
        return math.nan
    if left.shape[1] == 1:
        return float(wasserstein_distance(left[:, 0], right[:, 0]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(projections):
        direction = rng.normal(size=left.shape[1])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction = direction / norm
        values.append(
            wasserstein_distance(left @ direction, right @ direction)
        )
    return float(np.mean(values)) if values else math.nan


def layer_sparsity_comparison(
    left: np.ndarray,
    right: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    per_layer = []
    for idx, name in enumerate(names):
        item = ks_values(left[:, idx], right[:, idx])
        item.update(
            {
                "layer": name,
                "left_mean": float(left[:, idx].mean()),
                "right_mean": float(right[:, idx].mean()),
            }
        )
        per_layer.append(item)
    aggregate = ks_values(left.reshape(-1), right.reshape(-1))
    pvalues = [row["ks_pvalue"] for row in per_layer if not math.isnan(row["ks_pvalue"])]
    stats = [row["ks_statistic"] for row in per_layer if not math.isnan(row["ks_statistic"])]
    aggregate.update(
        {
            "per_layer_min_ks_pvalue": float(min(pvalues)) if pvalues else math.nan,
            "per_layer_median_ks_pvalue": float(np.median(pvalues)) if pvalues else math.nan,
            "per_layer_max_ks_statistic": float(max(stats)) if stats else math.nan,
            "left_mean_layer_sparsity": float(left.mean()),
            "right_mean_layer_sparsity": float(right.mean()),
            "per_layer": per_layer,
        }
    )
    return aggregate


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    left_norm = np.linalg.norm(left.T @ left, ord="fro")
    right_norm = np.linalg.norm(right.T @ right, ord="fro")
    denominator = left_norm * right_norm
    if denominator <= 1e-12:
        return math.nan
    return float(numerator / denominator)


def final_linear_module(model: nn.Module) -> nn.Linear:
    final: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            final = module
    if final is None:
        raise ValueError(f"{type(model).__name__} has no Linear classifier head")
    return final


@torch.no_grad()
def feature_matrix(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.to(device)
    model.eval()
    head = final_linear_module(model)
    chunks: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not inputs:
            raise RuntimeError("classifier feature hook received no inputs")
        features = inputs[0].detach()
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        chunks.append(features.cpu())

    handle = head.register_forward_pre_hook(capture)
    try:
        for x, _ in data_loader:
            model(x.to(device))
    finally:
        handle.remove()
    if not chunks:
        raise RuntimeError("classifier feature hook did not capture activations")
    return torch.cat(chunks, dim=0)


def cka_matrix(left_logits: list[np.ndarray], right_logits: list[np.ndarray]) -> np.ndarray:
    rows = []
    for left in left_logits:
        rows.append([linear_cka(left, right) for right in right_logits])
    return np.asarray(rows, dtype=np.float64)


def hungarian_cost_from_cka(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {"hungarian_cost": math.nan, "hungarian_mean_cka": math.nan}
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    cost = 1.0 - safe_matrix
    row_idx, col_idx = linear_sum_assignment(cost)
    assigned = safe_matrix[row_idx, col_idx]
    return {
        "hungarian_cost": float(np.mean(1.0 - assigned)),
        "hungarian_mean_cka": float(np.mean(assigned)),
    }


def cluster_representatives(
    matrix: np.ndarray,
    pca_dim: int,
) -> dict[str, Any]:
    if matrix.shape[0] < 3:
        labels = np.arange(matrix.shape[0], dtype=int)
        counts = np.bincount(labels)
        probabilities = counts.astype(np.float64) / max(1, labels.shape[0])
        positive_probabilities = probabilities[probabilities > 0.0]
        entropy = float(
            -(positive_probabilities * np.log(positive_probabilities)).sum()
        )
        if abs(entropy) < 1e-12:
            entropy = 0.0
        max_entropy = math.log(max(1, labels.shape[0]))
        return {
            "labels": list(range(matrix.shape[0])),
            "representative_indices": list(range(matrix.shape[0])),
            "num_clusters": int(matrix.shape[0]),
            "largest_cluster_fraction": 1.0 / max(1, matrix.shape[0]),
            "cluster_counts": [int(value) for value in counts.tolist()],
            "entropy_nats": entropy,
            "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
            "effective_cluster_count": float(math.exp(entropy)),
        }
    n_components = min(pca_dim, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    bandwidth = estimate_bandwidth(reduced, quantile=0.3, n_samples=matrix.shape[0])
    if bandwidth <= 1e-12:
        labels = np.zeros(matrix.shape[0], dtype=int)
        center = reduced.mean(axis=0)
        reps = [int(np.argmin(np.sum((reduced - center) ** 2, axis=1)))]
    else:
        clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(reduced)
        labels = clusterer.labels_
        reps = []
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label)
            center = clusterer.cluster_centers_[label]
            distances = np.sum((reduced[indices] - center) ** 2, axis=1)
            reps.append(int(indices[int(np.argmin(distances))]))
    counts = np.bincount(labels)
    probabilities = counts.astype(np.float64) / max(1, labels.shape[0])
    positive_probabilities = probabilities[probabilities > 0.0]
    entropy = float(-(positive_probabilities * np.log(positive_probabilities)).sum())
    if abs(entropy) < 1e-12:
        entropy = 0.0
    max_entropy = math.log(max(1, labels.shape[0]))
    return {
        "labels": [int(value) for value in labels.tolist()],
        "representative_indices": reps,
        "num_clusters": int(len(set(labels.tolist()))),
        "largest_cluster_fraction": float(counts.max() / labels.shape[0]),
        "cluster_counts": [int(value) for value in counts.tolist()],
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "effective_cluster_count": float(math.exp(entropy)),
    }


def compare_sets(
    label: str,
    left_masks: list[Mask],
    right_masks: list[Mask],
    left_logits: list[np.ndarray],
    right_logits: list[np.ndarray],
    left_features: list[np.ndarray],
    right_features: list[np.ndarray],
    names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    left_flat = [flatten_mask(mask, names) for mask in left_masks]
    right_flat = [flatten_mask(mask, names) for mask in right_masks]
    left_layer = np.stack([layer_sparsity_vector(mask, names) for mask in left_masks], axis=0)
    right_layer = np.stack([layer_sparsity_vector(mask, names) for mask in right_masks], axis=0)
    left_pairwise = pairwise_hamming(left_flat)
    right_pairwise = pairwise_hamming(right_flat)
    cross = cross_hamming(left_flat, right_flat)
    cka = cka_matrix(left_logits, right_logits)
    activation_cka = cka_matrix(left_features, right_features)
    hungarian = hungarian_cost_from_cka(cka)
    activation_hungarian = hungarian_cost_from_cka(activation_cka)
    layer = layer_sparsity_comparison(left_layer, right_layer, names)
    distance_ks = ks_values(left_pairwise, right_pairwise)
    distance_overlap = empirical_overlap(
        left_pairwise,
        right_pairwise,
        bins=args.histogram_bins,
        value_range=(0.0, 1.0),
    )
    nearest_right = []
    for left_mask in left_flat:
        nearest_right.append(min(float(np.mean(left_mask != right_mask)) for right_mask in right_flat))
    nearest_left = []
    for right_mask in right_flat:
        nearest_left.append(min(float(np.mean(left_mask != right_mask)) for left_mask in left_flat))
    result = {
        "label": label,
        "left_count": len(left_masks),
        "right_count": len(right_masks),
        "layer_sparsity": {
            **layer,
            **rbf_mmd(left_layer, right_layer),
            "sliced_wasserstein": sliced_wasserstein(
                left_layer,
                right_layer,
                projections=args.sliced_projections,
                seed=args.data_seed,
            ),
        },
        "mask_hamming_distribution": {
            **distance_ks,
            "overlap": distance_overlap,
            "left_pairwise_mean": (
                float(left_pairwise.mean()) if left_pairwise.size else math.nan
            ),
            "right_pairwise_mean": (
                float(right_pairwise.mean()) if right_pairwise.size else math.nan
            ),
            "cross_mean": float(cross.mean()) if cross.size else math.nan,
            "cross_median": float(np.median(cross)) if cross.size else math.nan,
            "nearest_ticket_distance_mean": float(np.mean(nearest_right)),
            "nearest_mode_distance_mean": float(np.mean(nearest_left)),
        },
        "logit_space_cka": {
            "cross_mean": float(np.nanmean(cka)),
            "cross_median": float(np.nanmedian(cka)),
            "best_per_left_mean": float(np.nanmean(np.nanmax(cka, axis=1))),
            "best_per_right_mean": float(np.nanmean(np.nanmax(cka, axis=0))),
            **hungarian,
        },
        "activation_space_cka": {
            "cross_mean": float(np.nanmean(activation_cka)),
            "cross_median": float(np.nanmedian(activation_cka)),
            "best_per_left_mean": float(np.nanmean(np.nanmax(activation_cka, axis=1))),
            "best_per_right_mean": float(np.nanmean(np.nanmax(activation_cka, axis=0))),
            **activation_hungarian,
        },
    }
    result["proposal_thresholds"] = {
        "layer_sparsity_ks_pvalue_gt_0p1": bool(
            result["layer_sparsity"]["ks_pvalue"] > 0.1
        ),
        "mask_hamming_overlap_gt_0p7": bool(
            result["mask_hamming_distribution"]["overlap"] > 0.7
        ),
        "logit_cka_hungarian_mean_gt_0p85": bool(
            result["logit_space_cka"]["hungarian_mean_cka"] > 0.85
        ),
        "hungarian_cost_lt_0p3": bool(
            result["logit_space_cka"]["hungarian_cost"] < 0.3
        ),
        "activation_cka_hungarian_mean_gt_0p85": bool(
            result["activation_space_cka"]["hungarian_mean_cka"] > 0.85
        ),
        "activation_hungarian_cost_lt_0p3": bool(
            result["activation_space_cka"]["hungarian_cost"] < 0.3
        ),
    }
    return result


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


def train_dense_with_rewind_state(
    model_factory,
    initial_state: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, float], list[dict[str, float]]]:
    if args.rewind_epochs < 0 or args.rewind_epochs > args.epochs:
        raise ValueError("rewind_epochs must be in [0, epochs]")
    model = model_factory().to(device)
    load_trainable_state(model, initial_state)
    optimizer = torch.optim.SGD(
        model.parameters(),
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

    rewind_state = initial_state if args.rewind_epochs == 0 else None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, device, optimizer)
        if scheduler is not None:
            scheduler.step()
        if epoch == args.rewind_epochs:
            rewind_state = state_to_cpu(model)
        if epoch in {1, args.rewind_epochs, args.epochs}:
            test_metrics = evaluate(model, test_loader, device)
            row = {
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    if rewind_state is None:
        raise RuntimeError("failed to capture rewind state")
    dense_state = state_to_cpu(model)
    dense_metrics = evaluate(model, test_loader, device)
    return dense_state, (None if args.rewind_epochs == 0 else rewind_state), dense_metrics, history


def sampler_config(args: argparse.Namespace, train_size: int):
    steps = max(args.sgld_steps, args.sgld_burn_in + args.samples * args.sgld_sample_every)
    if args.posterior_sampler == "sgld":
        return SGLDConfig(
            steps=steps,
            lr=args.sgld_lr,
            temperature=args.sgld_temperature,
            prior_precision=args.sgld_prior_precision,
            burn_in=args.sgld_burn_in,
            sample_every=args.sgld_sample_every,
            num_train_examples=train_size,
            likelihood_scale=args.sgld_likelihood_scale,
        )
    if args.posterior_sampler == "sghmc":
        return SGHMCConfig(
            steps=steps,
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
            num_train_examples=train_size,
            likelihood_scale=args.sgld_likelihood_scale,
        )
    if args.posterior_sampler == "cyclical-sgld":
        return CyclicalSGLDConfig(
            steps=steps,
            lr=args.sgld_lr,
            lr_min_ratio=args.csgld_lr_min_ratio,
            cycle_length=args.csgld_cycle_length,
            temperature=args.sgld_temperature,
            prior_precision=args.sgld_prior_precision,
            burn_in=args.sgld_burn_in,
            sample_every=args.sgld_sample_every,
            num_train_examples=train_size,
            likelihood_scale=args.sgld_likelihood_scale,
            sample_phase_start=args.csgld_sample_phase_start,
        )
    if args.posterior_sampler == "lowrank-laplace":
        return LowRankLaplaceConfig(
            num_samples=args.samples,
            scale=args.lowrank_laplace_scale,
            prior_precision=args.lowrank_laplace_prior_precision,
            fisher_batches=args.lowrank_laplace_fisher_batches,
            hessian_batches=args.lowrank_laplace_hessian_batches,
            rank=args.lowrank_laplace_rank,
            power_iterations=args.lowrank_laplace_power_iterations,
            oversample=args.lowrank_laplace_oversample,
            damping=args.lowrank_laplace_damping,
            variance_floor=args.lowrank_laplace_variance_floor,
            eigenvalue_floor=args.lowrank_laplace_eigenvalue_floor,
            num_train_examples=train_size,
            batchnorm_mode=args.lowrank_laplace_batchnorm_mode,
        )
    if args.posterior_sampler == "jointdiag-laplace":
        return {
            "num_samples": args.samples,
            "scale": args.jointdiag_laplace_scale,
            "prior_precision": args.jointdiag_laplace_prior_precision,
            "damping": args.jointdiag_laplace_damping,
            "hessian_batches": args.jointdiag_laplace_hessian_batches,
            "max_parameters": args.jointdiag_laplace_max_parameters,
            "num_train_examples": train_size,
        }
    return SWAGConfig(
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


def sampler_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.posterior_sampler == "sgld":
        return {
            "steps": args.sgld_steps,
            "lr": args.sgld_lr,
            "temperature": args.sgld_temperature,
            "prior_precision": args.sgld_prior_precision,
            "likelihood_scale": args.sgld_likelihood_scale,
            "burn_in": args.sgld_burn_in,
            "sample_every": args.sgld_sample_every,
        }
    if args.posterior_sampler == "sghmc":
        return {
            "steps": args.sgld_steps,
            "lr": args.sgld_lr if args.sghmc_lr is None else args.sghmc_lr,
            "momentum_decay": args.sghmc_momentum_decay,
            "temperature": (
                args.sgld_temperature
                if args.sghmc_temperature is None
                else args.sghmc_temperature
            ),
            "prior_precision": (
                args.sgld_prior_precision
                if args.sghmc_prior_precision is None
                else args.sghmc_prior_precision
            ),
            "likelihood_scale": args.sgld_likelihood_scale,
            "burn_in": args.sgld_burn_in,
            "sample_every": args.sgld_sample_every,
        }
    if args.posterior_sampler == "cyclical-sgld":
        return {
            "steps": args.sgld_steps,
            "lr": args.sgld_lr,
            "lr_min_ratio": args.csgld_lr_min_ratio,
            "cycle_length": args.csgld_cycle_length,
            "temperature": args.sgld_temperature,
            "prior_precision": args.sgld_prior_precision,
            "likelihood_scale": args.sgld_likelihood_scale,
            "burn_in": args.sgld_burn_in,
            "sample_every": args.sgld_sample_every,
            "sample_phase_start": args.csgld_sample_phase_start,
        }
    if args.posterior_sampler == "lowrank-laplace":
        return {
            "scale": args.lowrank_laplace_scale,
            "prior_precision": args.lowrank_laplace_prior_precision,
            "fisher_batches": args.lowrank_laplace_fisher_batches,
            "hessian_batches": args.lowrank_laplace_hessian_batches,
            "rank": args.lowrank_laplace_rank,
            "power_iterations": args.lowrank_laplace_power_iterations,
            "oversample": args.lowrank_laplace_oversample,
            "damping": args.lowrank_laplace_damping,
            "variance_floor": args.lowrank_laplace_variance_floor,
            "eigenvalue_floor": args.lowrank_laplace_eigenvalue_floor,
            "batchnorm_mode": args.lowrank_laplace_batchnorm_mode,
        }
    if args.posterior_sampler == "jointdiag-laplace":
        return {
            "scale": args.jointdiag_laplace_scale,
            "prior_precision": args.jointdiag_laplace_prior_precision,
            "damping": args.jointdiag_laplace_damping,
            "hessian_batches": args.jointdiag_laplace_hessian_batches,
            "max_parameters": args.jointdiag_laplace_max_parameters,
            "stream_joint_groups": True,
        }
    return {
        "epochs": args.swag_epochs,
        "lr": args.swag_lr,
        "weight_decay": (
            args.weight_decay if args.swag_weight_decay is None else args.swag_weight_decay
        ),
        "collection_start_epoch": args.swag_collection_start_epoch,
        "sample_every_epochs": args.swag_sample_every_epochs,
        "max_snapshots": args.swag_max_snapshots,
        "scale": args.swag_scale,
        "diagonal_scale": args.swag_diagonal_scale,
        "low_rank_scale": args.swag_low_rank_scale,
    }


def collect_samples(
    model,
    train_loader,
    device,
    config,
    args: argparse.Namespace,
    sample_seed: int,
):
    if args.posterior_sampler == "sgld":
        return collect_sgld_samples(model, train_loader, device, config)[: args.samples]
    if args.posterior_sampler == "sghmc":
        return collect_sghmc_samples(model, train_loader, device, config)[: args.samples]
    if args.posterior_sampler == "cyclical-sgld":
        return collect_cyclical_sgld_samples(model, train_loader, device, config)[: args.samples]
    if args.posterior_sampler == "lowrank-laplace":
        return collect_lowrank_laplace_samples(
            model,
            train_loader,
            device,
            config,
            seed=sample_seed,
        )[: args.samples]
    if args.posterior_sampler == "jointdiag-laplace":
        parameter_map = dict(model.named_parameters())
        block_groups = greedy_joint_groups_under_max(
            weight_parameter_names(model),
            parameter_map,
            int(config["max_parameters"]),
        )
        if not block_groups:
            raise ValueError("jointdiag-laplace found no weight tensors under max_parameters")
        base_state = state_to_cpu(model)
        samples = make_base_samples(base_state, args.samples)
        for group_idx, group_names in enumerate(block_groups):
            set_seed(sample_seed + 1000 * group_idx)
            factor = estimate_joint_block_laplace_factors(
                model,
                train_loader,
                device,
                JointBlockLaplaceConfig(
                    parameter_names=tuple(group_names),
                    num_samples=args.samples,
                    scale=float(config["scale"]),
                    prior_precision=float(config["prior_precision"]),
                    damping=float(config["damping"]),
                    hessian_batches=int(config["hessian_batches"]),
                    num_train_examples=int(config["num_train_examples"]),
                    max_parameters=int(config["max_parameters"]),
                ),
            )
            group_samples = sample_joint_block_laplace_from_factors(
                factor,
                JointBlockLaplaceConfig(
                    parameter_names=tuple(group_names),
                    num_samples=args.samples,
                    scale=float(config["scale"]),
                    prior_precision=float(config["prior_precision"]),
                    damping=float(config["damping"]),
                    hessian_batches=int(config["hessian_batches"]),
                    num_train_examples=int(config["num_train_examples"]),
                    max_parameters=int(config["max_parameters"]),
                ),
            )
            for sample_idx, group_sample in enumerate(group_samples):
                for name in group_names:
                    samples[sample_idx][name] = group_sample[name].detach().clone()
            del group_samples
            del factor
            cleanup_laplace_memory()
        return samples
    return collect_swag_samples(model, train_loader, device, config).samples[: args.samples]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_mask_artifacts(
    path: Path,
    *,
    names: list[str],
    reference_tensors: dict[str, torch.Tensor],
    mask_collections: list[tuple[str, list[dict[str, Any]], list[Mask]]],
    state_collections: list[tuple[str, list[dict[str, Any]], list[np.ndarray]]],
    metadata: dict[str, Any],
    save_states: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = parameter_sizes(names, reference_tensors)
    shapes = parameter_shapes(names, reference_tensors)
    payload: dict[str, Any] = {
        "artifact_schema_version": np.asarray([1], dtype=np.int64),
        "parameter_names": np.asarray(names, dtype=np.str_),
        "parameter_sizes": np.asarray(sizes, dtype=np.int64),
        "parameter_offsets": np.asarray(parameter_offsets(sizes), dtype=np.int64),
        "parameter_shapes_json": np.asarray(json.dumps(shapes, sort_keys=True), dtype=np.str_),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    }
    collection_summaries = []
    for collection_name, records, masks in mask_collections:
        ids = [str(record["id"]) for record in records]
        if len(ids) != len(masks):
            raise RuntimeError(f"mask artifact id/mask count mismatch: {collection_name}")
        payload[f"ids__{collection_name}"] = np.asarray(ids, dtype=np.str_)
        payload[f"masks__{collection_name}"] = mask_matrix(masks, names)
        collection_summaries.append(
            {
                "name": collection_name,
                "ids": len(ids),
                "masks": len(masks),
                "states": 0,
            }
        )
    if save_states:
        for collection_name, records, states in state_collections:
            ids = [str(record["id"]) for record in records]
            if len(ids) != len(states):
                raise RuntimeError(f"state artifact id/state count mismatch: {collection_name}")
            payload[f"state_ids__{collection_name}"] = np.asarray(ids, dtype=np.str_)
            payload[f"states__{collection_name}"] = state_matrix(states)
            for summary in collection_summaries:
                if summary["name"] == collection_name:
                    summary["states"] = len(states)
                    break
    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "schema_version": 1,
        "save_states": bool(save_states),
        "parameter_count": int(sum(sizes)),
        "parameter_shapes": shapes,
        "collections": collection_summaries,
    }


def main() -> None:
    args = parse_args()
    if args.save_state_artifacts:
        args.save_mask_artifacts = True
    if args.alignment_method != "none" and args.model != "resnet20":
        raise ValueError("channel alignment is currently implemented for resnet20 only")
    if args.alignment_batches <= 0:
        raise ValueError("alignment_batches must be positive")
    if args.posterior_chains <= 0:
        raise ValueError("posterior_chains must be positive")
    if args.out_dir is None:
        args.out_dir = Path("runs") / f"{args.dataset}_{args.model}_mode_ticket_distribution"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_bundle = make_bundle(args)
    reference_factory = make_model_factory(args, reference_bundle)
    names = weight_parameter_names(reference_factory())
    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "running",
            "config": {
                "dataset": args.dataset,
                "model": args.model,
                "seeds": args.seeds,
                "data_seed": args.data_seed,
                "epochs": args.epochs,
                "rewind_epochs": args.rewind_epochs,
                "imp_rounds": args.imp_rounds,
                "prune_fraction": args.prune_fraction,
                "posterior_sampler": args.posterior_sampler,
                "posterior_sampler_config": sampler_metadata(args),
                "samples_per_chain": args.samples,
                "posterior_chains": args.posterior_chains,
                "posterior_chain_init": args.posterior_chain_init,
                "samples_per_seed": args.samples * args.posterior_chains,
                "alignment_method": args.alignment_method,
                "alignment_batches": args.alignment_batches,
                "save_mask_artifacts": args.save_mask_artifacts,
                "save_state_artifacts": args.save_state_artifacts,
                "device": str(device),
            },
        },
    )

    ticket_masks: list[Mask] = []
    aligned_ticket_masks: list[Mask] = []
    ticket_states: list[np.ndarray] = []
    aligned_ticket_states: list[np.ndarray] = []
    ticket_logits: list[np.ndarray] = []
    ticket_features: list[np.ndarray] = []
    chain_start_masks: list[Mask] = []
    aligned_chain_start_masks: list[Mask] = []
    chain_start_logits: list[np.ndarray] = []
    chain_start_features: list[np.ndarray] = []
    chain_start_states: list[np.ndarray] = []
    aligned_chain_start_states: list[np.ndarray] = []
    posterior_masks: list[Mask] = []
    aligned_posterior_masks: list[Mask] = []
    posterior_logits: list[np.ndarray] = []
    posterior_features: list[np.ndarray] = []
    posterior_states: list[np.ndarray] = []
    aligned_posterior_states: list[np.ndarray] = []
    posterior_records: list[dict[str, Any]] = []
    chain_start_records: list[dict[str, Any]] = []
    ticket_records: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    alignment_target_features: dict[str, torch.Tensor] | None = None
    alignment_target_seed: int | None = None
    alignment_records: list[dict[str, Any]] = []
    alignment_prefix = f"{args.alignment_method}_aligned"

    for seed in args.seeds:
        bundle = make_bundle(args)
        model_factory = make_model_factory(args, bundle)
        set_seed(seed)
        initial_model = model_factory()
        initial_state = state_to_cpu(initial_model)
        dense_state, rewind_state, dense_metrics, train_history = (
            train_dense_with_rewind_state(
                model_factory,
                initial_state,
                bundle.train_loader,
                bundle.test_loader,
                device,
                args,
            )
        )
        if args.alignment_method != "none" and alignment_target_features is None:
            alignment_target_model = model_factory()
            load_trainable_state(alignment_target_model, dense_state)
            alignment_target_features = collect_alignment_features(
                method=args.alignment_method,
                state=dense_state,
                model=alignment_target_model,
                names=names,
                loader=bundle.test_loader,
                device=device,
                max_batches=args.alignment_batches,
            )
            alignment_target_seed = seed
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
            final_epochs=args.imp_final_epochs,
            rewind_state=rewind_state,
        )
        ticket_model = model_factory()
        load_trainable_state(ticket_model, imp.final_state)
        ticket_logits.append(logits_matrix(ticket_model, bundle.test_loader, device).numpy())
        ticket_features.append(feature_matrix(ticket_model, bundle.test_loader, device).numpy())
        ticket_masks.append(imp.mask)
        ticket_states.append(flatten_state(imp.final_state, names))
        if args.alignment_method != "none":
            assert alignment_target_features is not None
            source_features = collect_alignment_features(
                method=args.alignment_method,
                state=imp.final_state,
                model=ticket_model,
                names=names,
                loader=bundle.test_loader,
                device=device,
                max_batches=args.alignment_batches,
            )
            channel_maps, alignment_stats = activation_channel_alignment(
                source_features,
                alignment_target_features,
            )
            aligned_ticket_masks.append(aligned_weight_mask(imp.mask, names, channel_maps))
            aligned_ticket_states.append(
                flatten_state(aligned_weight_state(imp.final_state, names, channel_maps), names)
            )
            alignment_records.append(
                {
                    "source": "ticket",
                    "id": f"ticket_seed_{seed}",
                    "seed": seed,
                    **alignment_stats,
                }
            )
        ticket_records.append(
            {
                "id": f"ticket_seed_{seed}",
                "seed": seed,
                "accuracy": imp.metrics["accuracy"],
                "loss": imp.metrics["loss"],
                "sparsity": imp.metrics["sparsity"],
            }
        )

        sample_accuracies = []
        sample_chain_start_hamming = []
        seed_chain_records = []
        config = sampler_config(args, bundle.train_size)
        for chain_idx in range(args.posterior_chains):
            if args.posterior_chain_init == "independent-dense":
                set_seed(seed + 10_000 + chain_idx)
                chain_initial_model = model_factory()
                chain_initial_state = state_to_cpu(chain_initial_model)
                chain_start_state, _chain_rewind_state, chain_start_metrics, chain_history = (
                    train_dense_with_rewind_state(
                        model_factory,
                        chain_initial_state,
                        bundle.train_loader,
                        bundle.test_loader,
                        device,
                        args,
                    )
                )
            else:
                chain_start_state = dense_state
                chain_start_metrics = dense_metrics
                chain_history = train_history if chain_idx == 0 else []

            chain_start_id = f"chain_start_seed_{seed}_chain_{chain_idx}"
            chain_start_model = model_factory()
            load_trainable_state(chain_start_model, chain_start_state)
            chain_start_mask = global_magnitude_mask_from_state(
                chain_start_state,
                names,
                imp.metrics["sparsity"],
            )
            chain_start_flat = flatten_mask(chain_start_mask, names)
            chain_start_masks.append(chain_start_mask)
            chain_start_states.append(flatten_state(chain_start_state, names))
            chain_start_logits.append(
                logits_matrix(chain_start_model, bundle.test_loader, device).numpy()
            )
            chain_start_features.append(
                feature_matrix(chain_start_model, bundle.test_loader, device).numpy()
            )
            if args.alignment_method != "none":
                assert alignment_target_features is not None
                source_features = collect_alignment_features(
                    method=args.alignment_method,
                    state=chain_start_state,
                    model=chain_start_model,
                    names=names,
                    loader=bundle.test_loader,
                    device=device,
                    max_batches=args.alignment_batches,
                )
                channel_maps, alignment_stats = activation_channel_alignment(
                    source_features,
                    alignment_target_features,
                )
                aligned_chain_start_masks.append(
                    aligned_weight_mask(chain_start_mask, names, channel_maps)
                )
                aligned_chain_start_states.append(
                    flatten_state(
                        aligned_weight_state(chain_start_state, names, channel_maps),
                        names,
                    )
                )
                alignment_records.append(
                    {
                        "source": "chain_start",
                        "id": chain_start_id,
                        "seed": seed,
                        "chain": chain_idx,
                        **alignment_stats,
                    }
                )
            chain_record = {
                "id": chain_start_id,
                "seed": seed,
                "chain": chain_idx,
                "chain_init": args.posterior_chain_init,
                "accuracy": chain_start_metrics["accuracy"],
                "loss": chain_start_metrics["loss"],
                "sparsity": mask_sparsity(chain_start_mask),
                "train_history": chain_history,
            }
            chain_start_records.append(chain_record)
            seed_chain_records.append(chain_record)

            set_seed(seed + 20_000 + chain_idx)
            posterior_model = model_factory()
            load_trainable_state(posterior_model, chain_start_state)
            samples = collect_samples(
                posterior_model,
                bundle.train_loader,
                device,
                config,
                args,
                sample_seed=seed + 20_000 + chain_idx,
            )
            for sample_idx, sample in enumerate(samples):
                sample_mask = global_magnitude_mask_from_state(
                    sample,
                    names,
                    imp.metrics["sparsity"],
                )
                posterior_masks.append(sample_mask)
                posterior_states.append(flatten_state(sample, names))
                sample_model = model_factory()
                load_trainable_state(sample_model, sample)
                sample_metrics = evaluate(sample_model, bundle.test_loader, device)
                sample_accuracies.append(sample_metrics["accuracy"])
                sample_logits = logits_matrix(
                    sample_model,
                    bundle.test_loader,
                    device,
                ).numpy()
                posterior_logits.append(sample_logits)
                posterior_features.append(
                    feature_matrix(sample_model, bundle.test_loader, device).numpy()
                )
                sample_chain_hamming = float(
                    np.mean(flatten_mask(sample_mask, names) != chain_start_flat)
                )
                sample_chain_start_hamming.append(sample_chain_hamming)
                if args.alignment_method != "none":
                    assert alignment_target_features is not None
                    source_features = collect_alignment_features(
                        method=args.alignment_method,
                        state=sample,
                        model=sample_model,
                        names=names,
                        loader=bundle.test_loader,
                        device=device,
                        max_batches=args.alignment_batches,
                    )
                    channel_maps, alignment_stats = activation_channel_alignment(
                        source_features,
                        alignment_target_features,
                    )
                    aligned_posterior_masks.append(
                        aligned_weight_mask(sample_mask, names, channel_maps)
                    )
                    aligned_posterior_states.append(
                        flatten_state(
                            aligned_weight_state(sample, names, channel_maps),
                            names,
                        )
                    )
                    alignment_records.append(
                        {
                            "source": "posterior_sample",
                            "id": (
                                f"posterior_seed_{seed}_chain_{chain_idx}_"
                                f"sample_{sample_idx}"
                            ),
                            "seed": seed,
                            "chain": chain_idx,
                            "sample": sample_idx,
                            **alignment_stats,
                        }
                    )
                posterior_records.append(
                    {
                        "id": (
                            f"posterior_seed_{seed}_chain_{chain_idx}_"
                            f"sample_{sample_idx}"
                        ),
                        "seed": seed,
                        "chain": chain_idx,
                        "sample": sample_idx,
                        "chain_start_id": chain_start_id,
                        "chain_start_hamming": sample_chain_hamming,
                        "accuracy": sample_metrics["accuracy"],
                        "loss": sample_metrics["loss"],
                        "sparsity": mask_sparsity(sample_mask),
                    }
                )
        seed_summaries.append(
            {
                "seed": seed,
                "dense_accuracy": dense_metrics["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "imp_sparsity": imp.metrics["sparsity"],
                "posterior_chain_count": args.posterior_chains,
                "posterior_chain_init": args.posterior_chain_init,
                "posterior_sample_count": len(sample_accuracies),
                "chain_start_accuracy_mean": (
                    float(np.mean([row["accuracy"] for row in seed_chain_records]))
                    if seed_chain_records
                    else math.nan
                ),
                "posterior_to_chain_start_hamming_mean": (
                    float(np.mean(sample_chain_start_hamming))
                    if sample_chain_start_hamming
                    else math.nan
                ),
                "posterior_sample_accuracy_mean": (
                    float(np.mean(sample_accuracies)) if sample_accuracies else math.nan
                ),
                "chain_starts": seed_chain_records,
                "train_history": train_history,
            }
        )
        write_json(
            run_dir / "partial_seed_summaries.json",
            {
                "status": "running",
                "completed_seed_count": len(seed_summaries),
                "total_seed_count": len(args.seeds),
                "seed_summaries": seed_summaries,
            },
        )

    if not posterior_masks:
        raise RuntimeError("posterior sampler produced no samples")
    chain_start_matrix = np.stack(chain_start_states, axis=0)
    chain_start_clusters = cluster_representatives(
        chain_start_matrix,
        args.cluster_pca_dim,
    )
    state_matrix = np.stack(posterior_states, axis=0)
    clusters = cluster_representatives(state_matrix, args.cluster_pca_dim)
    mode_indices = clusters["representative_indices"]
    mode_masks = [posterior_masks[idx] for idx in mode_indices]
    mode_states = [posterior_states[idx] for idx in mode_indices]
    mode_logits = [posterior_logits[idx] for idx in mode_indices]
    mode_features = [posterior_features[idx] for idx in mode_indices]
    mode_records = [
        {
            **posterior_records[idx],
            "id": f"posterior_mode_{mode_idx}_sample_{idx}",
            "mode_index": mode_idx,
            "cluster_label": clusters["labels"][idx],
        }
        for mode_idx, idx in enumerate(mode_indices)
    ]
    comparison_entries: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (
            compare_sets(
                "chain_start_magnitude_vs_tickets",
                chain_start_masks,
                ticket_masks,
                chain_start_logits,
                ticket_logits,
                chain_start_features,
                ticket_features,
                names,
                args,
            ),
            chain_start_clusters,
        ),
        (
            compare_sets(
                "posterior_samples_vs_tickets",
                posterior_masks,
                ticket_masks,
                posterior_logits,
                ticket_logits,
                posterior_features,
                ticket_features,
                names,
                args,
            ),
            clusters,
        ),
        (
            compare_sets(
                "posterior_modes_vs_tickets",
                mode_masks,
                ticket_masks,
                mode_logits,
                ticket_logits,
                mode_features,
                ticket_features,
                names,
                args,
            ),
            clusters,
        ),
    ]
    aligned_clusters = None
    aligned_chain_start_clusters = None
    aligned_mode_indices: list[int] = []
    aligned_mode_masks: list[Mask] = []
    aligned_mode_states: list[np.ndarray] = []
    aligned_mode_records: list[dict[str, Any]] = []
    if args.alignment_method != "none":
        if len(aligned_posterior_masks) != len(posterior_masks):
            raise RuntimeError("alignment did not produce one mask per posterior sample")
        if len(aligned_ticket_masks) != len(ticket_masks):
            raise RuntimeError("alignment did not produce one mask per ticket")
        if len(aligned_chain_start_masks) != len(chain_start_masks):
            raise RuntimeError("alignment did not produce one mask per chain start")
        if len(aligned_chain_start_states) != len(chain_start_states):
            raise RuntimeError("alignment did not produce one state per chain start")
        if len(aligned_posterior_states) != len(posterior_states):
            raise RuntimeError("alignment did not produce one state per posterior sample")
        aligned_chain_start_matrix = np.stack(aligned_chain_start_states, axis=0)
        aligned_chain_start_clusters = cluster_representatives(
            aligned_chain_start_matrix,
            args.cluster_pca_dim,
        )
        aligned_state_matrix = np.stack(aligned_posterior_states, axis=0)
        aligned_clusters = cluster_representatives(aligned_state_matrix, args.cluster_pca_dim)
        aligned_mode_indices = aligned_clusters["representative_indices"]
        aligned_mode_masks = [aligned_posterior_masks[idx] for idx in aligned_mode_indices]
        aligned_mode_states = [aligned_posterior_states[idx] for idx in aligned_mode_indices]
        aligned_mode_logits = [posterior_logits[idx] for idx in aligned_mode_indices]
        aligned_mode_features = [posterior_features[idx] for idx in aligned_mode_indices]
        aligned_mode_records = [
            {
                **posterior_records[idx],
                "id": f"{alignment_prefix}_posterior_mode_{mode_idx}_sample_{idx}",
                "mode_index": mode_idx,
                "cluster_label": aligned_clusters["labels"][idx],
            }
            for mode_idx, idx in enumerate(aligned_mode_indices)
        ]
        comparison_entries.extend(
            [
                (
                    compare_sets(
                        f"{alignment_prefix}_chain_start_magnitude_vs_tickets",
                        aligned_chain_start_masks,
                        aligned_ticket_masks,
                        chain_start_logits,
                        ticket_logits,
                        chain_start_features,
                        ticket_features,
                        names,
                        args,
                    ),
                    aligned_chain_start_clusters,
                ),
                (
                    compare_sets(
                        f"{alignment_prefix}_posterior_samples_vs_tickets",
                        aligned_posterior_masks,
                        aligned_ticket_masks,
                        posterior_logits,
                        ticket_logits,
                        posterior_features,
                        ticket_features,
                        names,
                        args,
                    ),
                    aligned_clusters,
                ),
                (
                    compare_sets(
                        f"{alignment_prefix}_posterior_modes_vs_tickets",
                        aligned_mode_masks,
                        aligned_ticket_masks,
                        aligned_mode_logits,
                        ticket_logits,
                        aligned_mode_features,
                        ticket_features,
                        names,
                        args,
                    ),
                    aligned_clusters,
                ),
            ]
        )
    comparisons = [comparison for comparison, _cluster_info in comparison_entries]
    layer_rows = []
    for source, records, masks in [
        ("chain_start", chain_start_records, chain_start_masks),
        ("posterior_sample", posterior_records, posterior_masks),
        ("posterior_mode", mode_records, mode_masks),
        ("ticket", ticket_records, ticket_masks),
    ]:
        for record, mask in zip(records, masks):
            row = {"source": source, "id": record["id"]}
            for name, value in zip(names, layer_sparsity_vector(mask, names)):
                row[f"sparsity::{name}"] = value
            layer_rows.append(row)
    if args.alignment_method != "none":
        for source, records, masks in [
            (f"{alignment_prefix}_chain_start", chain_start_records, aligned_chain_start_masks),
            (f"{alignment_prefix}_posterior_sample", posterior_records, aligned_posterior_masks),
            (f"{alignment_prefix}_posterior_mode", aligned_mode_records, aligned_mode_masks),
            (f"{alignment_prefix}_ticket", ticket_records, aligned_ticket_masks),
        ]:
            for record, mask in zip(records, masks):
                row = {"source": source, "id": record["id"]}
                for name, value in zip(names, layer_sparsity_vector(mask, names)):
                    row[f"sparsity::{name}"] = value
                layer_rows.append(row)
    cka_rows = []
    mode_ticket_cka = cka_matrix(mode_logits, ticket_logits)
    mode_ticket_activation_cka = cka_matrix(mode_features, ticket_features)
    for i, mode_record in enumerate(mode_records):
        for j, ticket_record in enumerate(ticket_records):
            cka_rows.append(
                {
                    "mode_id": mode_record["id"],
                    "ticket_id": ticket_record["id"],
                    "logit_cka": mode_ticket_cka[i, j],
                    "cost": 1.0 - mode_ticket_cka[i, j],
                    "activation_cka": mode_ticket_activation_cka[i, j],
                    "activation_cost": 1.0 - mode_ticket_activation_cka[i, j],
                }
            )

    mask_artifact_summary = None
    if args.save_mask_artifacts:
        mask_collections = [
            ("chain_start", chain_start_records, chain_start_masks),
            ("posterior_sample", posterior_records, posterior_masks),
            ("posterior_mode", mode_records, mode_masks),
            ("ticket", ticket_records, ticket_masks),
        ]
        state_collections = [
            ("chain_start", chain_start_records, chain_start_states),
            ("posterior_sample", posterior_records, posterior_states),
            ("posterior_mode", mode_records, mode_states),
            ("ticket", ticket_records, ticket_states),
        ]
        if args.alignment_method != "none":
            mask_collections.extend(
                [
                    (f"{alignment_prefix}_chain_start", chain_start_records, aligned_chain_start_masks),
                    (f"{alignment_prefix}_posterior_sample", posterior_records, aligned_posterior_masks),
                    (f"{alignment_prefix}_posterior_mode", aligned_mode_records, aligned_mode_masks),
                    (f"{alignment_prefix}_ticket", ticket_records, aligned_ticket_masks),
                ]
            )
            state_collections.extend(
                [
                    (f"{alignment_prefix}_chain_start", chain_start_records, aligned_chain_start_states),
                    (f"{alignment_prefix}_posterior_sample", posterior_records, aligned_posterior_states),
                    (f"{alignment_prefix}_posterior_mode", aligned_mode_records, aligned_mode_states),
                    (f"{alignment_prefix}_ticket", ticket_records, aligned_ticket_states),
                ]
            )
        mask_artifact_summary = save_mask_artifacts(
            run_dir / "mask_artifacts.npz",
            names=names,
            reference_tensors=ticket_masks[0],
            mask_collections=mask_collections,
            state_collections=state_collections,
            metadata={
                "dataset": args.dataset,
                "model": args.model,
                "input_shape": list(reference_bundle.input_shape),
                "num_classes": reference_bundle.num_classes,
                "hidden_dim": args.hidden_dim,
                "cnn_width": args.cnn_width,
                "resnet_width": args.resnet_width,
                "depth": args.depth,
                "blocks_per_stage": 3 if args.model == "resnet20" else None,
                "data_seed": args.data_seed,
                "train_subset": args.train_subset,
                "test_subset": args.test_subset,
                "seeds": args.seeds,
                "posterior_sampler": args.posterior_sampler,
                "posterior_chains": args.posterior_chains,
                "posterior_chain_init": args.posterior_chain_init,
                "alignment_method": args.alignment_method,
                "mask_encoding": "uint8_keep_mask_flattened_by_parameter_names",
                "state_encoding": "float32_flattened_by_parameter_names",
            },
            save_states=args.save_state_artifacts,
        )

    metrics = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "seeds": args.seeds,
            "data_seed": args.data_seed,
            "epochs": args.epochs,
            "rewind_epochs": args.rewind_epochs,
            "imp_rounds": args.imp_rounds,
            "prune_fraction": args.prune_fraction,
            "posterior_sampler": args.posterior_sampler,
            "posterior_sampler_config": sampler_metadata(args),
            "samples_per_chain": args.samples,
            "posterior_chains": args.posterior_chains,
            "posterior_chain_init": args.posterior_chain_init,
            "samples_per_seed": args.samples * args.posterior_chains,
            "alignment_method": args.alignment_method,
            "alignment_batches": args.alignment_batches,
            "save_mask_artifacts": args.save_mask_artifacts,
            "save_state_artifacts": args.save_state_artifacts,
            "device": str(device),
        },
        "seed_summaries": seed_summaries,
        "layer_names": names,
        "chain_start_clustering": chain_start_clusters,
        "posterior_clustering": clusters,
        "aligned_chain_start_clustering": aligned_chain_start_clusters,
        "aligned_posterior_clustering": aligned_clusters,
        "posterior_chain_diagnostics": {
            "chain_start_count": len(chain_start_records),
            "posterior_sample_count": len(posterior_records),
            "posterior_to_chain_start_hamming_mean": (
                float(np.mean([row["chain_start_hamming"] for row in posterior_records]))
                if posterior_records
                else math.nan
            ),
            "posterior_to_chain_start_hamming_median": (
                float(np.median([row["chain_start_hamming"] for row in posterior_records]))
                if posterior_records
                else math.nan
            ),
            "posterior_to_chain_start_hamming_min": (
                float(np.min([row["chain_start_hamming"] for row in posterior_records]))
                if posterior_records
                else math.nan
            ),
            "posterior_to_chain_start_hamming_max": (
                float(np.max([row["chain_start_hamming"] for row in posterior_records]))
                if posterior_records
                else math.nan
            ),
            "chain_start_accuracy_mean": (
                float(np.mean([row["accuracy"] for row in chain_start_records]))
                if chain_start_records
                else math.nan
            ),
            "posterior_sample_accuracy_mean": (
                float(np.mean([row["accuracy"] for row in posterior_records]))
                if posterior_records
                else math.nan
            ),
        },
        "alignment": {
            "method": args.alignment_method,
            "target_seed": alignment_target_seed,
            "target": "first_seed_dense_model" if args.alignment_method != "none" else None,
            "records": alignment_records,
        },
        "mask_artifacts": mask_artifact_summary,
        "comparisons": comparisons,
        "caveats": [
            "Posterior modes are mean-shift representatives in raw parameter PCA space.",
            (
                "Chain-start rows use magnitude masks from the dense state that "
                "initialized each posterior chain."
            ),
            (
                "Activation-aligned comparisons cluster and compare masks after mapping "
                "ResNet channels to the first seed dense model by activation-correlation "
                "Hungarian matching."
                if args.alignment_method == "activation"
                else (
                    "Weight-aligned comparisons cluster and compare masks after mapping "
                    "ResNet channels to the first seed dense model by incoming/outgoing "
                    "weight-correlation Hungarian matching."
                    if args.alignment_method == "weight"
                    else "No channel permutation alignment is applied here."
                )
            ),
            "Activation comparison uses final hidden-feature linear CKA on the held-out test set.",
            "Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.",
        ],
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "complete",
            "config": metrics["config"],
            "metrics_path": str(run_dir / "metrics.json"),
            "mask_artifacts": mask_artifact_summary,
        },
    )
    write_json(
        run_dir / "partial_seed_summaries.json",
        {
            "status": "complete",
            "completed_seed_count": len(seed_summaries),
            "total_seed_count": len(args.seeds),
            "seed_summaries": seed_summaries,
        },
    )
    write_csv(run_dir / "layer_sparsity_vectors.csv", layer_rows)
    write_csv(run_dir / "mode_ticket_cka.csv", cka_rows)
    summary_rows = []
    for comparison, cluster_info in comparison_entries:
        summary_rows.append(
            {
                "comparison": comparison["label"],
                "posterior_num_clusters": cluster_info["num_clusters"],
                "posterior_largest_cluster_fraction": cluster_info[
                    "largest_cluster_fraction"
                ],
                "posterior_cluster_entropy_nats": cluster_info["entropy_nats"],
                "posterior_cluster_entropy_normalized": cluster_info["normalized_entropy"],
                "posterior_effective_cluster_count": cluster_info[
                    "effective_cluster_count"
                ],
                "left_count": comparison["left_count"],
                "right_count": comparison["right_count"],
                "layer_ks_pvalue": comparison["layer_sparsity"]["ks_pvalue"],
                "layer_mmd_rbf": comparison["layer_sparsity"]["mmd_rbf"],
                "layer_sliced_wasserstein": comparison["layer_sparsity"][
                    "sliced_wasserstein"
                ],
                "hamming_overlap": comparison["mask_hamming_distribution"]["overlap"],
                "hamming_cross_mean": comparison["mask_hamming_distribution"]["cross_mean"],
                "logit_cka_hungarian_mean": comparison["logit_space_cka"][
                    "hungarian_mean_cka"
                ],
                "hungarian_cost": comparison["logit_space_cka"]["hungarian_cost"],
                "activation_cka_hungarian_mean": comparison["activation_space_cka"][
                    "hungarian_mean_cka"
                ],
                "activation_hungarian_cost": comparison["activation_space_cka"][
                    "hungarian_cost"
                ],
            }
        )
    write_csv(run_dir / "mode_ticket_distribution_summary.csv", summary_rows)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
