#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lottery.masks import Mask, mask_sparsity, support_jaccard
from lottery.train import load_trainable_state
from run_residual_cross_seed_transfer_probe import (
    build_artifacts,
    make_model_factory,
    parse_float_list,
    parse_int_list,
    parse_source_list,
)
from run_residual_predictor_mask_probe import (
    flatten_mask,
    flatten_scores,
    imp_residual_mask,
    mask_fingerprint,
    mask_from_swaps,
    random_heldout_residual_mask,
    residual_swap_count,
    select_indices,
    train_fixed_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
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
    parser.add_argument("--random-residual-trials", type=int, default=1)
    parser.add_argument(
        "--alignment-method",
        choices=["none", "activation"],
        default="none",
        help="Optionally add activation-channel-aligned source-vote residual variants.",
    )
    parser.add_argument("--alignment-batches", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_direct_transfer_probe"))
    return parser.parse_args()


def source_oracle_add_indices(
    *,
    artifact: Any,
    base_source: str,
    alpha: float,
) -> torch.Tensor:
    base = artifact.source_masks[base_source]
    base_flat = flatten_mask(base, artifact.names)
    imp_flat = flatten_mask(artifact.imp.mask, artifact.names)
    imp_only = imp_flat & ~base_flat
    _, base_only_count, imp_only_count = residual_swap_count(
        base,
        artifact.imp.mask,
        artifact.names,
        alpha,
    )
    swap_count = min(int(round(alpha * min(base_only_count, imp_only_count))), imp_only_count)
    return select_indices(artifact.imp_scores, imp_only, swap_count, largest=True)


def source_vote_scores(
    *,
    sources: list[Any],
    base_source: str,
    alpha: float,
    total_params: int,
) -> torch.Tensor:
    votes = torch.zeros(total_params, dtype=torch.float32)
    for source in sources:
        add_idx = source_oracle_add_indices(
            artifact=source,
            base_source=base_source,
            alpha=alpha,
        )
        votes[add_idx] += 1.0
    return votes


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


def flat_indices_to_mask(
    indices: torch.Tensor,
    template: Mask,
    names: list[str],
) -> Mask:
    out: Mask = {}
    offset = 0
    for name in names:
        numel = template[name].numel()
        local = torch.zeros(numel, dtype=torch.bool)
        keep = indices[(indices >= offset) & (indices < offset + numel)] - offset
        if keep.numel():
            local[keep.long()] = True
        out[name] = local.reshape(template[name].shape)
        offset += numel
    return out


def aligned_weight_mask(mask: Mask, names: list[str], channel_maps: dict[str, torch.Tensor]) -> Mask:
    aligned: Mask = {}
    for name in names:
        out_key, in_key = resnet_weight_axes(name)
        value = mask[name].bool().cpu()
        if out_key is not None:
            target = torch.zeros_like(value)
            target.index_copy_(0, channel_maps[out_key], value)
            value = target
        if in_key is not None:
            target = torch.zeros_like(value)
            target.index_copy_(1, channel_maps[in_key], value)
            value = target
        aligned[name] = value
    return aligned


def aligned_source_vote_scores(
    *,
    target: Any,
    sources: list[Any],
    model_factory,
    base_source: str,
    alpha: float,
    total_params: int,
    device: torch.device,
    alignment_batches: int,
    target_feature_cache: dict[int, dict[str, torch.Tensor]],
    source_feature_cache: dict[tuple[int, int], dict[str, torch.Tensor]],
    alignment_cache: dict[tuple[int, int], tuple[dict[str, torch.Tensor], dict[str, float | int]]],
) -> tuple[torch.Tensor, dict[str, float | int]]:
    votes = torch.zeros(total_params, dtype=torch.float32)
    pair_stats: list[dict[str, float | int]] = []

    if target.seed not in target_feature_cache:
        target_model = model_factory().to(device)
        load_trainable_state(target_model, target.dense_state)
        target_feature_cache[target.seed] = collect_resnet_activation_features(
            target_model,
            target.bundle.test_loader,
            device,
            alignment_batches,
        )

    for source in sources:
        cache_key = (target.seed, source.seed)
        if cache_key not in alignment_cache:
            if cache_key not in source_feature_cache:
                source_model = model_factory().to(device)
                load_trainable_state(source_model, source.dense_state)
                source_feature_cache[cache_key] = collect_resnet_activation_features(
                    source_model,
                    target.bundle.test_loader,
                    device,
                    alignment_batches,
                )
            alignment_cache[cache_key] = activation_channel_alignment(
                source_feature_cache[cache_key],
                target_feature_cache[target.seed],
            )
        channel_maps, stats = alignment_cache[cache_key]
        pair_stats.append(stats)
        add_idx = source_oracle_add_indices(
            artifact=source,
            base_source=base_source,
            alpha=alpha,
        )
        source_add_mask = flat_indices_to_mask(add_idx, source.imp.mask, source.names)
        aligned_add_mask = aligned_weight_mask(source_add_mask, source.names, channel_maps)
        votes += flatten_mask(aligned_add_mask, source.names).float()

    mean_corr = torch.tensor(
        [float(stats["alignment_mean_corr"]) for stats in pair_stats],
        dtype=torch.float32,
    )
    min_corr = torch.tensor(
        [float(stats["alignment_min_corr"]) for stats in pair_stats],
        dtype=torch.float32,
    )
    return votes, {
        "alignment_pair_count": len(pair_stats),
        "alignment_map_count": int(pair_stats[0]["alignment_map_count"]) if pair_stats else 0,
        "alignment_channel_count": (
            int(pair_stats[0]["alignment_channel_count"]) if pair_stats else 0
        ),
        "alignment_mean_corr": float(mean_corr.mean().item()) if pair_stats else None,
        "alignment_min_corr": float(min_corr.min().item()) if pair_stats else None,
    }


def direct_transfer_mask(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    votes: torch.Tensor,
    alpha: float,
    seed: int,
    random_within_votes: bool,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swap_count, base_only_count, imp_only_count = residual_swap_count(base, imp, names, alpha)
    candidate_mask = ~base_flat
    swap_count = min(swap_count, base_only_count, int(candidate_mask.sum().item()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    if random_within_votes:
        pool = candidate_mask & (votes > 0)
        pool_idx = pool.nonzero(as_tuple=False).flatten()
        if pool_idx.numel() < swap_count:
            pool_idx = candidate_mask.nonzero(as_tuple=False).flatten()
        perm = torch.randperm(pool_idx.numel(), generator=generator)
        add_idx = pool_idx[perm[:swap_count]]
    else:
        candidate_idx = candidate_mask.nonzero(as_tuple=False).flatten()
        jitter = torch.rand(candidate_idx.numel(), generator=generator) * 1e-6
        score = votes[candidate_idx] + jitter
        selected_local = torch.topk(score, swap_count, largest=True).indices
        add_idx = candidate_idx[selected_local]

    hits = int(imp_only[add_idx].sum().item())
    selected_votes = votes[add_idx] if add_idx.numel() else torch.empty(0)
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": base_only_count,
        "imp_only_count": imp_only_count,
        "candidate_count": int(candidate_mask.sum().item()),
        "source_vote_positive_count": int((candidate_mask & (votes > 0)).sum().item()),
        "source_vote_max": float(votes[candidate_mask].max().item()) if candidate_mask.any() else None,
        "selected_source_vote_mean": (
            float(selected_votes.mean().item()) if selected_votes.numel() else None
        ),
        "selected_source_vote_positive_fraction": (
            float((selected_votes > 0).float().mean().item()) if selected_votes.numel() else None
        ),
        "added_imp_only_hits": hits,
        "added_imp_only_precision": hits / swap_count if swap_count else None,
    }


def fill_direct_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for key in [
        "candidate_count",
        "source_vote_positive_count",
        "source_vote_max",
        "selected_source_vote_mean",
        "selected_source_vote_positive_fraction",
        "alignment_pair_count",
        "alignment_map_count",
        "alignment_channel_count",
        "alignment_mean_corr",
        "alignment_min_corr",
    ]:
        out.setdefault(key, None)
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if len(seeds) < 2:
        raise ValueError("direct transfer requires at least two seeds")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.random_residual_trials < 0:
        raise ValueError("random_residual_trials must be non-negative")
    if args.alignment_method == "activation" and args.model != "resnet20":
        raise ValueError("activation alignment is currently implemented for resnet20 only")
    if args.alignment_batches <= 0:
        raise ValueError("alignment_batches must be positive")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imp_epochs = args.epochs if args.imp_epochs is None else args.imp_epochs
    mask_train_epochs = imp_epochs if args.mask_train_epochs is None else args.mask_train_epochs
    artifacts = [
        build_artifacts(
            args,
            seed=seed,
            trajectory_epochs=trajectory_epochs,
            base_sources=base_sources,
            imp_epochs=imp_epochs,
            device=device,
        )
        for seed in seeds
    ]
    by_seed = {artifact.seed: artifact for artifact in artifacts}
    rows: list[dict[str, Any]] = []
    train_history = [row for artifact in artifacts for row in artifact.train_history]
    target_feature_cache: dict[int, dict[str, torch.Tensor]] = {}
    source_feature_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    alignment_cache: dict[
        tuple[int, int],
        tuple[dict[str, torch.Tensor], dict[str, float | int]],
    ] = {}

    for target_seed in seeds:
        target = by_seed[target_seed]
        model_factory = make_model_factory(args, target.bundle)
        train_cache: dict[str, dict[str, float]] = {}

        def evaluate_candidate(
            *,
            base_source: str,
            variant: str,
            alpha: float,
            trial: int | None,
            source_seeds: list[int],
            mask: Mask,
            meta: dict[str, Any],
        ) -> None:
            fingerprint = mask_fingerprint(mask, target.names)
            cache_hit = fingerprint in train_cache
            if not cache_hit:
                train_cache[fingerprint] = train_fixed_mask(
                    model_factory=model_factory,
                    train_state=target.train_state,
                    mask=mask,
                    train_loader=target.bundle.train_loader,
                    test_loader=target.bundle.test_loader,
                    device=device,
                    epochs=mask_train_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lr_schedule=args.lr_schedule,
                )
            metrics = train_cache[fingerprint]
            base_mask = target.source_masks[base_source]
            out: dict[str, Any] = {
                "target_seed": target.seed,
                "source_seeds": ",".join(str(seed) for seed in source_seeds),
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trial": trial,
                "mask_train_epochs": mask_train_epochs,
                "trained_loss": metrics["loss"],
                "trained_accuracy": metrics["accuracy"],
                "mask_sparsity": metrics["sparsity"],
                "mask_to_base_jaccard": support_jaccard(mask, base_mask),
                "mask_to_imp_jaccard": support_jaccard(mask, target.imp.mask),
                "mask_to_dense_final_magnitude_jaccard": support_jaccard(mask, target.dense_mask),
                "mask_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, target.rewind_mask)
                    if target.rewind_mask is not None
                    else None
                ),
                "dense_accuracy": target.checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": target.imp.metrics["accuracy"],
                "accuracy_minus_imp": metrics["accuracy"] - target.imp.metrics["accuracy"],
                "accuracy_minus_dense": (
                    metrics["accuracy"] - target.checkpoint_metrics[args.epochs]["accuracy"]
                ),
                "cache_hit": cache_hit,
            }
            out.update(fill_direct_meta(meta))
            rows.append(out)
            print(json.dumps(out), flush=True)

        for base_source in base_sources:
            base_mask = target.source_masks[base_source]
            base_flat = flatten_mask(base_mask, target.names)
            target_imp_flat = flatten_mask(target.imp.mask, target.names)
            base_score_flat = flatten_scores(target.source_scores[base_source], target.names)
            heldout_idx = (~base_flat).nonzero(as_tuple=False).flatten()
            source_seeds = [seed for seed in seeds if seed != target_seed]
            source_artifacts = [by_seed[seed] for seed in source_seeds]

            evaluate_candidate(
                base_source=base_source,
                variant="base",
                alpha=0.0,
                trial=None,
                source_seeds=[],
                mask=base_mask,
                meta={
                    "swap_count": 0,
                    "base_only_count": int((base_flat & ~target_imp_flat).sum().item()),
                    "imp_only_count": int((~base_flat & target_imp_flat).sum().item()),
                    "candidate_count": int(heldout_idx.numel()),
                    "added_imp_only_hits": 0,
                    "added_imp_only_precision": None,
                },
            )

            for alpha in alphas:
                oracle_mask, oracle_meta = imp_residual_mask(
                    base=base_mask,
                    imp=target.imp.mask,
                    names=target.names,
                    base_scores=base_score_flat,
                    imp_scores=target.imp_scores,
                    alpha=alpha,
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="target_oracle_residual",
                    alpha=alpha,
                    trial=None,
                    source_seeds=[],
                    mask=oracle_mask,
                    meta=oracle_meta,
                )

                votes = source_vote_scores(
                    sources=source_artifacts,
                    base_source=base_source,
                    alpha=alpha,
                    total_params=base_flat.numel(),
                )
                direct_mask, direct_meta = direct_transfer_mask(
                    base=base_mask,
                    imp=target.imp.mask,
                    names=target.names,
                    base_scores=base_score_flat,
                    votes=votes,
                    alpha=alpha,
                    seed=target.seed * 100000 + int(round(alpha * 1000)) + 17,
                    random_within_votes=False,
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="source_vote_residual",
                    alpha=alpha,
                    trial=None,
                    source_seeds=source_seeds,
                    mask=direct_mask,
                    meta=direct_meta,
                )

                vote_random_mask, vote_random_meta = direct_transfer_mask(
                    base=base_mask,
                    imp=target.imp.mask,
                    names=target.names,
                    base_scores=base_score_flat,
                    votes=votes,
                    alpha=alpha,
                    seed=target.seed * 100000 + int(round(alpha * 1000)) + 29,
                    random_within_votes=True,
                )
                evaluate_candidate(
                    base_source=base_source,
                    variant="source_vote_random_residual",
                    alpha=alpha,
                    trial=None,
                    source_seeds=source_seeds,
                    mask=vote_random_mask,
                    meta=vote_random_meta,
                )

                if args.alignment_method == "activation":
                    aligned_votes, alignment_meta = aligned_source_vote_scores(
                        target=target,
                        sources=source_artifacts,
                        model_factory=model_factory,
                        base_source=base_source,
                        alpha=alpha,
                        total_params=base_flat.numel(),
                        device=device,
                        alignment_batches=args.alignment_batches,
                        target_feature_cache=target_feature_cache,
                        source_feature_cache=source_feature_cache,
                        alignment_cache=alignment_cache,
                    )
                    aligned_mask, aligned_meta = direct_transfer_mask(
                        base=base_mask,
                        imp=target.imp.mask,
                        names=target.names,
                        base_scores=base_score_flat,
                        votes=aligned_votes,
                        alpha=alpha,
                        seed=target.seed * 100000 + int(round(alpha * 1000)) + 41,
                        random_within_votes=False,
                    )
                    aligned_meta.update(alignment_meta)
                    evaluate_candidate(
                        base_source=base_source,
                        variant="aligned_source_vote_residual",
                        alpha=alpha,
                        trial=None,
                        source_seeds=source_seeds,
                        mask=aligned_mask,
                        meta=aligned_meta,
                    )

                    aligned_random_mask, aligned_random_meta = direct_transfer_mask(
                        base=base_mask,
                        imp=target.imp.mask,
                        names=target.names,
                        base_scores=base_score_flat,
                        votes=aligned_votes,
                        alpha=alpha,
                        seed=target.seed * 100000 + int(round(alpha * 1000)) + 43,
                        random_within_votes=True,
                    )
                    aligned_random_meta.update(alignment_meta)
                    evaluate_candidate(
                        base_source=base_source,
                        variant="aligned_source_vote_random_residual",
                        alpha=alpha,
                        trial=None,
                        source_seeds=source_seeds,
                        mask=aligned_random_mask,
                        meta=aligned_random_meta,
                    )

                for trial in range(args.random_residual_trials):
                    random_mask, random_meta = random_heldout_residual_mask(
                        base=base_mask,
                        imp=target.imp.mask,
                        names=target.names,
                        base_scores=base_score_flat,
                        heldout_idx=heldout_idx,
                        alpha=alpha,
                        seed=target.seed * 100000 + trial * 1000 + int(round(alpha * 1000)),
                    )
                    evaluate_candidate(
                        base_source=base_source,
                        variant="target_random_residual",
                        alpha=alpha,
                        trial=trial,
                        source_seeds=[],
                        mask=random_mask,
                        meta=random_meta,
                    )
        print(
            json.dumps(
                {
                    "target_seed": target_seed,
                    "num_rows_so_far": len(rows),
                    "unique_trained_masks": len(train_cache),
                }
            ),
            flush=True,
        )

    payload = {
        "seeds": seeds,
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
            "random_residual_trials": args.random_residual_trials,
            "alignment_method": args.alignment_method,
            "alignment_batches": args.alignment_batches,
        },
        "seed_summaries": [
            {
                "seed": artifact.seed,
                "dense": artifact.checkpoint_metrics[args.epochs],
                "imp": artifact.imp.metrics,
            }
            for artifact in artifacts
        ],
        "rows": rows,
        "train_history": train_history,
    }
    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_rows(run_dir / "residual_direct_transfer_probe.csv", rows)
    print(
        json.dumps(
            {
                "seeds": seeds,
                "dataset": args.dataset,
                "model": args.model,
                "num_rows": len(rows),
                "run_dir": str(run_dir),
                "best_candidate": max(rows, key=lambda row: float(row["trained_accuracy"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
