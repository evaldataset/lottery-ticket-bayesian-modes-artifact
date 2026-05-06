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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lottery.diag_laplace import DiagonalLaplaceConfig, collect_diag_laplace_samples
from lottery.masks import Mask, support_jaccard
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
    mask_from_swaps,
    mask_fingerprint,
    random_heldout_residual_mask,
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
    parser.add_argument(
        "--base-kinds",
        default="trajectory,imp_overlap_random",
        help="Comma-separated subset of trajectory,imp_overlap_random.",
    )
    parser.add_argument(
        "--residual-variants",
        default="oracle,random-residual",
        help=(
            "Comma-separated subset of oracle,random-residual,random-imp,low-imp,"
            "posterior-imp,dense-imp,posterior-excess-imp,posterior-std-imp. "
            "The default reproduces the original compatibility probe."
        ),
    )
    parser.add_argument("--random-residual-trials", type=int, default=1)
    parser.add_argument("--posterior-laplace-samples", type=int, default=10)
    parser.add_argument("--posterior-laplace-scale", type=float, default=1e-3)
    parser.add_argument("--posterior-laplace-prior-precision", type=float, default=1e-2)
    parser.add_argument("--posterior-laplace-fisher-batches", type=int, default=20)
    parser.add_argument("--posterior-laplace-variance-floor", type=float, default=1e-12)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/residual_base_compatibility_probe"))
    return parser.parse_args()


def matched_imp_overlap_random_base(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    seed: int,
) -> Mask:
    generator = torch.Generator()
    generator.manual_seed(seed)
    out: Mask = {}
    for name in names:
        base_flat = base[name].bool().flatten().cpu()
        imp_flat = imp[name].bool().flatten().cpu()
        keep_flat = torch.zeros_like(base_flat)

        imp_pool = imp_flat.nonzero(as_tuple=False).flatten()
        nonimp_pool = (~imp_flat).nonzero(as_tuple=False).flatten()
        imp_keep = int((base_flat & imp_flat).sum().item())
        nonimp_keep = int((base_flat & ~imp_flat).sum().item())

        if imp_keep:
            perm = torch.randperm(imp_pool.numel(), generator=generator)
            keep_flat[imp_pool[perm[:imp_keep]]] = True
        if nonimp_keep:
            perm = torch.randperm(nonimp_pool.numel(), generator=generator)
            keep_flat[nonimp_pool[perm[:nonimp_keep]]] = True
        out[name] = keep_flat.reshape(base[name].shape).clone()
    return out


def random_indices(candidates: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    candidate_idx = candidates.nonzero(as_tuple=False).flatten()
    if k <= 0:
        return candidate_idx[:0]
    if candidate_idx.numel() <= k:
        return candidate_idx
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(candidate_idx.numel(), generator=generator)
    return candidate_idx[perm[:k]]


def imp_only_residual_mask(
    *,
    base: Mask,
    imp: Mask,
    names: list[str],
    base_scores: torch.Tensor,
    imp_scores: torch.Tensor,
    add_scores: torch.Tensor | None = None,
    alpha: float,
    add_order: str,
    seed: int,
    oracle_add_idx: torch.Tensor,
) -> tuple[Mask, dict[str, int | float | None]]:
    base_flat = flatten_mask(base, names)
    imp_flat = flatten_mask(imp, names)
    base_only = base_flat & ~imp_flat
    imp_only = imp_flat & ~base_flat
    swappable = min(int(base_only.sum().item()), int(imp_only.sum().item()))
    swap_count = int(round(alpha * swappable))
    swap_count = min(swap_count, int(base_only.sum().item()), int(imp_only.sum().item()))
    remove_idx = select_indices(base_scores, base_only, swap_count, largest=False)
    if add_order == "random_imp":
        add_idx = random_indices(imp_only, swap_count, seed)
    elif add_order == "low_imp":
        add_idx = select_indices(imp_scores, imp_only, swap_count, largest=False)
    elif add_order == "score_imp":
        if add_scores is None:
            raise ValueError("score_imp add_order requires add_scores")
        add_idx = select_indices(add_scores, imp_only, swap_count, largest=True)
    else:
        raise ValueError(f"unknown add_order: {add_order}")
    oracle = torch.zeros_like(imp_only)
    oracle[oracle_add_idx] = True
    overlap = int(oracle[add_idx].sum().item())
    return mask_from_swaps(base, names, remove_idx, add_idx), {
        "swap_count": swap_count,
        "base_only_count": int(base_only.sum().item()),
        "imp_only_count": int(imp_only.sum().item()),
        "candidate_count": int(imp_only.sum().item()),
        "heldout_count": int(imp_only.sum().item()),
        "heldout_positive_count": int(imp_only.sum().item()),
        "added_imp_only_hits": int(imp_only[add_idx].sum().item()),
        "added_imp_only_precision": 1.0 if swap_count else None,
        "added_oracle_overlap_hits": overlap,
        "added_oracle_overlap_precision": overlap / swap_count if swap_count else None,
    }


def posterior_score_tensors(
    samples: list[dict[str, torch.Tensor]],
    names: list[str],
    dense_scores: dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, dict[str, torch.Tensor]]:
    if not samples:
        raise ValueError("posterior_score_tensors requires at least one sample")
    rms_scores: dict[str, torch.Tensor] = {}
    std_scores: dict[str, torch.Tensor] = {}
    excess_scores: dict[str, torch.Tensor] = {}
    for name in names:
        values = torch.stack(
            [sample[name].detach().float().cpu() for sample in samples],
            dim=0,
        )
        rms = (values.square().mean(dim=0) + eps).sqrt()
        mean = values.mean(dim=0)
        std = ((values - mean).square().mean(dim=0) + eps).sqrt()
        dense_abs = dense_scores[name].detach().float().cpu().abs()
        rms_scores[name] = rms
        std_scores[name] = std
        excess_scores[name] = rms - dense_abs
    return {
        "diag_laplace_rms": rms_scores,
        "diag_laplace_std": std_scores,
        "diag_laplace_rms_minus_dense_abs": excess_scores,
    }


def collect_diag_laplace_score_flats(
    *,
    model_factory,
    dense_state: dict[str, torch.Tensor],
    dense_scores: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    train_size: int,
    names: list[str],
    device: torch.device,
    samples: int,
    scale: float,
    prior_precision: float,
    fisher_batches: int,
    variance_floor: float,
) -> torch.Tensor:
    model = model_factory().to(device)
    load_trainable_state(model, dense_state)
    config = DiagonalLaplaceConfig(
        num_samples=samples,
        scale=scale,
        prior_precision=prior_precision,
        fisher_batches=fisher_batches,
        variance_floor=variance_floor,
        num_train_examples=train_size,
    )
    posterior_samples = collect_diag_laplace_samples(model, train_loader, device, config)
    return {
        score_name: flatten_scores(score_tensors, names)
        for score_name, score_tensors in posterior_score_tensors(
            posterior_samples,
            names,
            dense_scores,
        ).items()
    }


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
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.random_residual_trials < 0:
        raise ValueError("random_residual_trials must be non-negative")
    alphas = parse_float_list(args.alphas)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    base_sources = parse_source_list(args.base_sources)
    base_kinds = parse_source_list(args.base_kinds)
    valid_base_kinds = {"trajectory", "imp_overlap_random"}
    unknown_base_kinds = sorted(set(base_kinds) - valid_base_kinds)
    if unknown_base_kinds:
        raise ValueError(
            f"unknown base kinds {unknown_base_kinds}; expected subset of {sorted(valid_base_kinds)}"
        )
    residual_variants = parse_source_list(args.residual_variants)
    valid_residual_variants = {
        "oracle",
        "random-residual",
        "random-imp",
        "low-imp",
        "posterior-imp",
        "dense-imp",
        "posterior-excess-imp",
        "posterior-std-imp",
    }
    unknown_residual_variants = sorted(set(residual_variants) - valid_residual_variants)
    if unknown_residual_variants:
        raise ValueError(
            f"unknown residual variants {unknown_residual_variants}; "
            f"expected subset of {sorted(valid_residual_variants)}"
        )
    posterior_residual_variants = {
        "posterior-imp",
        "posterior-excess-imp",
        "posterior-std-imp",
    }
    if posterior_residual_variants & set(residual_variants):
        if args.posterior_laplace_samples <= 0:
            raise ValueError("posterior_laplace_samples must be positive")
        if args.posterior_laplace_scale <= 0.0:
            raise ValueError("posterior_laplace_scale must be positive")
        if args.posterior_laplace_prior_precision < 0.0:
            raise ValueError("posterior_laplace_prior_precision must be non-negative")
        if args.posterior_laplace_fisher_batches <= 0:
            raise ValueError("posterior_laplace_fisher_batches must be positive")
        if args.posterior_laplace_variance_floor <= 0.0:
            raise ValueError("posterior_laplace_variance_floor must be positive")
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

    rows: list[dict[str, Any]] = []
    train_history = [row for artifact in artifacts for row in artifact.train_history]

    for artifact in artifacts:
        model_factory = make_model_factory(args, artifact.bundle)
        train_cache: dict[str, dict[str, float]] = {}
        dense_score_flat = flatten_scores(
            artifact.source_scores[f"epoch_{args.epochs}"],
            artifact.names,
        )
        posterior_score_flats: dict[str, torch.Tensor] = {}
        if posterior_residual_variants & set(residual_variants):
            posterior_score_flats = collect_diag_laplace_score_flats(
                model_factory=model_factory,
                dense_state=artifact.dense_state,
                dense_scores=artifact.source_scores[f"epoch_{args.epochs}"],
                train_loader=artifact.bundle.train_loader,
                train_size=artifact.bundle.train_size,
                names=artifact.names,
                device=device,
                samples=args.posterior_laplace_samples,
                scale=args.posterior_laplace_scale,
                prior_precision=args.posterior_laplace_prior_precision,
                fisher_batches=args.posterior_laplace_fisher_batches,
                variance_floor=args.posterior_laplace_variance_floor,
            )
            print(
                json.dumps(
                    {
                        "seed": artifact.seed,
                        "posterior_scores": sorted(posterior_score_flats),
                        "posterior_laplace_samples": args.posterior_laplace_samples,
                        "posterior_laplace_scale": args.posterior_laplace_scale,
                        "posterior_laplace_fisher_batches": (
                            args.posterior_laplace_fisher_batches
                        ),
                    }
                ),
                flush=True,
            )

        def evaluate_candidate(
            *,
            base_source: str,
            base_kind: str,
            variant: str,
            alpha: float,
            trial: int | None,
            reference_base: Mask,
            effective_base: Mask,
            mask: Mask,
            meta: dict[str, Any],
        ) -> None:
            fingerprint = mask_fingerprint(mask, artifact.names)
            cache_hit = fingerprint in train_cache
            if not cache_hit:
                train_cache[fingerprint] = train_fixed_mask(
                    model_factory=model_factory,
                    train_state=artifact.train_state,
                    mask=mask,
                    train_loader=artifact.bundle.train_loader,
                    test_loader=artifact.bundle.test_loader,
                    device=device,
                    epochs=mask_train_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lr_schedule=args.lr_schedule,
                )
            metrics = train_cache[fingerprint]
            out: dict[str, Any] = {
                "seed": artifact.seed,
                "base_source": base_source,
                "base_kind": base_kind,
                "variant": variant,
                "alpha": alpha,
                "trial": trial,
                "mask_train_epochs": mask_train_epochs,
                "trained_loss": metrics["loss"],
                "trained_accuracy": metrics["accuracy"],
                "mask_sparsity": metrics["sparsity"],
                "base_to_imp_jaccard": support_jaccard(effective_base, artifact.imp.mask),
                "base_to_reference_jaccard": support_jaccard(effective_base, reference_base),
                "mask_to_effective_base_jaccard": support_jaccard(mask, effective_base),
                "mask_to_reference_base_jaccard": support_jaccard(mask, reference_base),
                "mask_to_imp_jaccard": support_jaccard(mask, artifact.imp.mask),
                "mask_to_dense_final_magnitude_jaccard": support_jaccard(mask, artifact.dense_mask),
                "mask_to_rewind_magnitude_jaccard": (
                    support_jaccard(mask, artifact.rewind_mask)
                    if artifact.rewind_mask is not None
                    else None
                ),
                "dense_accuracy": artifact.checkpoint_metrics[args.epochs]["accuracy"],
                "imp_accuracy": artifact.imp.metrics["accuracy"],
                "accuracy_minus_imp": metrics["accuracy"] - artifact.imp.metrics["accuracy"],
                "accuracy_minus_dense": (
                    metrics["accuracy"] - artifact.checkpoint_metrics[args.epochs]["accuracy"]
                ),
                "cache_hit": cache_hit,
            }
            out.update(meta)
            rows.append(out)
            print(json.dumps(out), flush=True)

        for base_source in base_sources:
            reference_base = artifact.source_masks[base_source]
            reference_flat = flatten_mask(reference_base, artifact.names)
            imp_flat = flatten_mask(artifact.imp.mask, artifact.names)
            base_score_flat = flatten_scores(artifact.source_scores[base_source], artifact.names)
            matched_base = matched_imp_overlap_random_base(
                base=reference_base,
                imp=artifact.imp.mask,
                names=artifact.names,
                seed=artifact.seed * 100003 + len(base_source),
            )
            all_bases = {
                "trajectory": reference_base,
                "imp_overlap_random": matched_base,
            }
            for base_kind in base_kinds:
                effective_base = all_bases[base_kind]
                base_flat = flatten_mask(effective_base, artifact.names)
                heldout_idx = (~base_flat).nonzero(as_tuple=False).flatten()
                evaluate_candidate(
                    base_source=base_source,
                    base_kind=base_kind,
                    variant="base",
                    alpha=0.0,
                    trial=None,
                    reference_base=reference_base,
                    effective_base=effective_base,
                    mask=effective_base,
                    meta={
                        "swap_count": 0,
                        "base_only_count": int((base_flat & ~imp_flat).sum().item()),
                        "imp_only_count": int((~base_flat & imp_flat).sum().item()),
                        "candidate_count": int(heldout_idx.numel()),
                        "heldout_count": int(heldout_idx.numel()),
                        "heldout_positive_count": int(imp_flat[heldout_idx].sum().item()),
                        "added_imp_only_hits": 0,
                        "added_imp_only_precision": None,
                    },
                )
                for alpha in alphas:
                    oracle_mask, oracle_meta_raw = imp_residual_mask(
                        base=effective_base,
                        imp=artifact.imp.mask,
                        names=artifact.names,
                        base_scores=base_score_flat,
                        imp_scores=artifact.imp_scores,
                        alpha=alpha,
                    )
                    oracle_meta = dict(oracle_meta_raw)
                    oracle_meta["added_oracle_overlap_hits"] = oracle_meta["swap_count"]
                    oracle_meta["added_oracle_overlap_precision"] = (
                        1.0 if oracle_meta["swap_count"] else None
                    )
                    base_eff_flat = flatten_mask(effective_base, artifact.names)
                    imp_eff_flat = flatten_mask(artifact.imp.mask, artifact.names)
                    oracle_add_mask = (
                        flatten_mask(oracle_mask, artifact.names)
                        & ~base_eff_flat
                        & imp_eff_flat
                    )
                    oracle_add_idx = oracle_add_mask.nonzero(as_tuple=False).flatten()
                    if "oracle" in residual_variants:
                        evaluate_candidate(
                            base_source=base_source,
                            base_kind=base_kind,
                            variant="oracle_imp_residual",
                            alpha=alpha,
                            trial=None,
                            reference_base=reference_base,
                            effective_base=effective_base,
                            mask=oracle_mask,
                            meta=oracle_meta,
                        )
                    if "low-imp" in residual_variants:
                        low_imp_mask, low_imp_meta = imp_only_residual_mask(
                            base=effective_base,
                            imp=artifact.imp.mask,
                            names=artifact.names,
                            base_scores=base_score_flat,
                            imp_scores=artifact.imp_scores,
                            alpha=alpha,
                            add_order="low_imp",
                            seed=artifact.seed * 100000 + int(round(alpha * 1000)) + 37,
                            oracle_add_idx=oracle_add_idx,
                        )
                        evaluate_candidate(
                            base_source=base_source,
                            base_kind=base_kind,
                            variant="low_imp_only_residual",
                            alpha=alpha,
                            trial=None,
                            reference_base=reference_base,
                            effective_base=effective_base,
                            mask=low_imp_mask,
                            meta=low_imp_meta,
                        )
                    score_ordered_variants: list[
                        tuple[str, str, torch.Tensor, str, int, dict[str, Any]]
                    ] = []
                    if "dense-imp" in residual_variants:
                        score_ordered_variants.append(
                            (
                                "dense-imp",
                                "dense_imp_only_residual",
                                dense_score_flat,
                                "dense_final_abs",
                                59,
                                {},
                            )
                        )
                    posterior_score_specs = [
                        (
                            "posterior-imp",
                            "posterior_imp_only_residual",
                            "diag_laplace_rms",
                            71,
                        ),
                        (
                            "posterior-excess-imp",
                            "posterior_excess_imp_only_residual",
                            "diag_laplace_rms_minus_dense_abs",
                            73,
                        ),
                        (
                            "posterior-std-imp",
                            "posterior_std_imp_only_residual",
                            "diag_laplace_std",
                            79,
                        ),
                    ]
                    for (
                        requested_variant,
                        output_variant,
                        score_name,
                        seed_offset,
                    ) in posterior_score_specs:
                        if requested_variant not in residual_variants:
                            continue
                        if score_name not in posterior_score_flats:
                            raise RuntimeError(f"missing posterior score {score_name}")
                        score_ordered_variants.append(
                            (
                                requested_variant,
                                output_variant,
                                posterior_score_flats[score_name],
                                score_name,
                                seed_offset,
                                {
                                    "posterior_score": score_name,
                                    "posterior_laplace_samples": (
                                        args.posterior_laplace_samples
                                    ),
                                    "posterior_laplace_scale": (
                                        args.posterior_laplace_scale
                                    ),
                                    "posterior_laplace_prior_precision": (
                                        args.posterior_laplace_prior_precision
                                    ),
                                    "posterior_laplace_fisher_batches": (
                                        args.posterior_laplace_fisher_batches
                                    ),
                                    "posterior_laplace_variance_floor": (
                                        args.posterior_laplace_variance_floor
                                    ),
                                },
                            )
                        )
                    for (
                        _requested_variant,
                        output_variant,
                        add_score_flat,
                        add_score_name,
                        seed_offset,
                        extra_meta,
                    ) in score_ordered_variants:
                        score_imp_mask, score_imp_meta = imp_only_residual_mask(
                            base=effective_base,
                            imp=artifact.imp.mask,
                            names=artifact.names,
                            base_scores=base_score_flat,
                            imp_scores=artifact.imp_scores,
                            add_scores=add_score_flat,
                            alpha=alpha,
                            add_order="score_imp",
                            seed=(
                                artifact.seed * 100000
                                + int(round(alpha * 1000))
                                + seed_offset
                            ),
                            oracle_add_idx=oracle_add_idx,
                        )
                        score_imp_meta.update({"add_score": add_score_name})
                        score_imp_meta.update(extra_meta)
                        evaluate_candidate(
                            base_source=base_source,
                            base_kind=base_kind,
                            variant=output_variant,
                            alpha=alpha,
                            trial=None,
                            reference_base=reference_base,
                            effective_base=effective_base,
                            mask=score_imp_mask,
                            meta=score_imp_meta,
                        )
                    for trial in range(args.random_residual_trials):
                        if "random-imp" in residual_variants:
                            random_imp_mask, random_imp_meta = imp_only_residual_mask(
                                base=effective_base,
                                imp=artifact.imp.mask,
                                names=artifact.names,
                                base_scores=base_score_flat,
                                imp_scores=artifact.imp_scores,
                                alpha=alpha,
                                add_order="random_imp",
                                seed=artifact.seed * 100000 + trial * 1000 + int(round(alpha * 1000)) + 19,
                                oracle_add_idx=oracle_add_idx,
                            )
                            evaluate_candidate(
                                base_source=base_source,
                                base_kind=base_kind,
                                variant="random_imp_only_residual",
                                alpha=alpha,
                                trial=trial,
                                reference_base=reference_base,
                                effective_base=effective_base,
                                mask=random_imp_mask,
                                meta=random_imp_meta,
                            )
                        if "random-residual" in residual_variants:
                            random_mask, random_meta = random_heldout_residual_mask(
                                base=effective_base,
                                imp=artifact.imp.mask,
                                names=artifact.names,
                                base_scores=base_score_flat,
                                heldout_idx=heldout_idx,
                                alpha=alpha,
                                seed=artifact.seed * 100000 + trial * 1000 + int(round(alpha * 1000)),
                            )
                            evaluate_candidate(
                                base_source=base_source,
                                base_kind=base_kind,
                                variant="random_residual",
                                alpha=alpha,
                                trial=trial,
                                reference_base=reference_base,
                                effective_base=effective_base,
                                mask=random_mask,
                                meta=random_meta,
                            )
        print(
            json.dumps(
                {
                    "seed": artifact.seed,
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
            "base_kinds": base_kinds,
            "alphas": alphas,
            "residual_variants": residual_variants,
            "random_residual_trials": args.random_residual_trials,
            "posterior_laplace_samples": args.posterior_laplace_samples,
            "posterior_laplace_scale": args.posterior_laplace_scale,
            "posterior_laplace_prior_precision": args.posterior_laplace_prior_precision,
            "posterior_laplace_fisher_batches": args.posterior_laplace_fisher_batches,
            "posterior_laplace_variance_floor": args.posterior_laplace_variance_floor,
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
    write_rows(run_dir / "residual_base_compatibility_probe.csv", rows)
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
