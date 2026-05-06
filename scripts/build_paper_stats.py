#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
T_CRIT_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "paper_stats.md")
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "paper" / "tables" / "statistical_summary.tex",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs" / "paper_stats.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_mode_distribution_equivalence_rows() -> list[dict[str, Any]]:
    path = ROOT / "runs" / "mode_distribution_equivalence_audit_summary.csv"
    if not path.exists():
        return []
    numeric_keys = {
        "runs",
        "n_pairs",
        "n_posterior_values",
        "n_baseline_values",
        "posterior_mean",
        "posterior_std",
        "baseline_mean",
        "baseline_std",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "positive",
        "negative",
        "zero",
        "paired_win_rate",
        "all_pair_win_rate",
        "ks_stat",
        "ks_pvalue",
        "wasserstein",
        "mmd_rbf",
        "raw_ks_stat",
        "raw_ks_pvalue",
        "raw_wasserstein",
        "raw_mmd_rbf",
        "raw_pair_win_rate",
        "mean_post_chain",
        "mean_sample_accuracy",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = dict(row)
            for key in numeric_keys:
                if key in out:
                    out[key] = float(out[key])
            rows.append(out)
    return rows


def read_direct_mode_ticket_distribution_rows() -> list[dict[str, Any]]:
    paths = [
        (
            "Digits MLP",
            ROOT / "runs" / "digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv",
        ),
        (
            "CIFAR subset ResNet",
            ROOT
            / "runs"
            / "cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5_summary.csv",
        ),
        (
            "CIFAR subset act. pilot",
            ROOT
            / "runs"
            / "cifar10_subset4096_mode_ticket_distribution_activation_pilot_summary.csv",
        ),
        (
            "CIFAR full ResNet",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full aligned",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full weight-aligned",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full cSGLD multi-chain",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full cSGLD independent",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full LowRank128Lap",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv",
        ),
        (
            "CIFAR full JointDiagLap270k",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv",
        ),
    ]
    numeric_keys = {
        "left_count",
        "right_count",
        "posterior_num_clusters",
        "posterior_largest_cluster_fraction",
        "posterior_cluster_entropy_nats",
        "posterior_cluster_entropy_normalized",
        "posterior_effective_cluster_count",
        "layer_ks_pvalue",
        "layer_mmd_rbf",
        "layer_sliced_wasserstein",
        "hamming_overlap",
        "hamming_cross_mean",
        "logit_cka_hungarian_mean",
        "hungarian_cost",
        "activation_cka_hungarian_mean",
        "activation_hungarian_cost",
    }
    bool_keys = {
        "passes_layer_ks",
        "passes_hamming_overlap",
        "passes_logit_cka",
        "passes_hungarian_cost",
        "passes_activation_cka",
        "passes_activation_hungarian_cost",
    }
    rows: list[dict[str, Any]] = []
    for setting, path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if setting == "CIFAR full aligned" and not row.get(
                    "comparison", ""
                ).startswith("activation_aligned_"):
                    continue
                if setting == "CIFAR full weight-aligned" and not row.get(
                    "comparison", ""
                ).startswith("weight_aligned_"):
                    continue
                out: dict[str, Any] = dict(row)
                out["setting"] = setting
                for key in numeric_keys:
                    if key in out:
                        value = str(out[key]).strip()
                        out[key] = float(value) if value else math.nan
                for key in bool_keys:
                    if key in out and str(out[key]).strip():
                        out[key] = str(out[key]).lower() == "true"
                available_bool_keys = [
                    key for key in bool_keys if isinstance(out.get(key), bool)
                ]
                out["threshold_pass_count"] = sum(
                    1 for key in available_bool_keys if out.get(key)
                )
                out["threshold_pass_total"] = len(available_bool_keys)
                rows.append(out)
    return rows


def direct_mode_ticket_comparison_label(comparison: str) -> str:
    if comparison.startswith("activation_aligned_"):
        comparison = "aligned_" + comparison.removeprefix("activation_aligned_")
    if comparison.startswith("weight_aligned_"):
        comparison = "weight-aligned_" + comparison.removeprefix("weight_aligned_")
    return comparison.replace("_", " ")


def ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, math.nan, math.nan
    tcrit = T_CRIT_95.get(len(arr), 1.96)
    half = float(tcrit * arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, mean - half, mean + half


def summarize_values(label: str, values: list[float]) -> dict[str, Any]:
    mean, low, high = ci95(values)
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    zeros = len(values) - positives - negatives
    return {
        "label": label,
        "n": len(values),
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive": positives,
        "negative": negatives,
        "zero": zeros,
    }


def summarize_optional_values(label: str, values: list[float | None]) -> dict[str, Any]:
    clean = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    if not clean:
        return {
            "label": label,
            "n": 0,
            "mean": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "positive": 0,
            "negative": 0,
            "zero": 0,
        }
    return summarize_values(label, clean)


def gate1_rows(dataset: str, pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs").glob(pattern)):
        payload = read_json(path)
        cfg = path.parent.name.replace(f"{dataset}_gate1_full_", "")
        posterior = payload["posterior_mask_overlap"]["posterior_jaccard_mean"]
        random = payload["posterior_mask_overlap"]["random_jaccard_mean"]
        chain = payload["controls"]["chain_start_magnitude_to_imp_jaccard_mean"]
        dense = payload["controls"]["dense_magnitude_to_imp_jaccard"]
        rows.append(
            {
                "dataset": dataset,
                "config": cfg,
                "seed": payload["seed"],
                "posterior_minus_random": posterior - random,
                "posterior_minus_chain": posterior - chain,
                "dense_minus_posterior": dense - posterior,
                "post_chain": payload["controls"][
                    "posterior_to_chain_start_magnitude_jaccard_mean"
                ],
            }
        )
    return rows


def movement_rows(label: str, root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "sampler": label,
                    "seed": payload["seed"],
                    "scale": row["sgld_lr"],
                    "posterior_minus_chain": (
                        row["posterior_jaccard_mean"]
                        - row["chain_start_magnitude_to_imp_jaccard"]
                    ),
                    "rewind_minus_posterior": (
                        row["rewind_magnitude_to_imp_jaccard"]
                        - row["posterior_jaccard_mean"]
                    ),
                    "post_chain": row[
                        "posterior_to_chain_start_magnitude_jaccard_mean"
                    ],
                    "sample_accuracy": row["sample_accuracy_mean"],
                }
            )
    return rows


def head_laplace_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "sampler": "HeadLap",
                    "seed": payload["seed"],
                    "scale": row["head_laplace_scale"],
                    "posterior_minus_chain": (
                        row["head_posterior_jaccard_mean"]
                        - row["head_chain_start_magnitude_to_imp_jaccard"]
                    ),
                    "rewind_minus_posterior": (
                        row["head_rewind_magnitude_to_imp_jaccard"]
                        - row["head_posterior_jaccard_mean"]
                    ),
                    "post_chain": row[
                        "head_posterior_to_chain_start_magnitude_jaccard_mean"
                    ],
                    "sample_accuracy": row["sample_accuracy_mean"],
                }
            )
    return rows


def block_laplace_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            block_name = row["block_name"]
            rows.append(
                {
                    "sampler": (
                        "JointLap"
                        if str(block_name).startswith("joint:")
                        else (
                            "JointDiagLap"
                            if str(block_name).startswith("jointdiag:")
                            else (
                                "BlockDiagLap"
                                if str(block_name).startswith("blockdiag:")
                                else "BlockLap"
                            )
                        )
                    ),
                    "seed": payload["seed"],
                    "block": block_name,
                    "scale": row["block_laplace_scale"],
                    "parameter_count": row["block_parameter_count"],
                    "block_posterior_minus_chain": (
                        row["block_posterior_jaccard_mean"]
                        - row["block_chain_start_magnitude_to_imp_jaccard"]
                    ),
                    "block_rewind_minus_posterior": (
                        row["block_rewind_magnitude_to_imp_jaccard"]
                        - row["block_posterior_jaccard_mean"]
                    ),
                    "block_post_chain": row[
                        "block_posterior_to_chain_start_magnitude_jaccard_mean"
                    ],
                    "global_posterior_minus_chain": (
                        row["global_posterior_jaccard_mean"]
                        - row["global_chain_start_magnitude_to_imp_jaccard"]
                    ),
                    "global_rewind_minus_posterior": (
                        row["global_rewind_magnitude_to_imp_jaccard"]
                        - row["global_posterior_jaccard_mean"]
                    ),
                    "global_post_chain": row[
                        "global_posterior_to_chain_start_magnitude_jaccard_mean"
                    ],
                    "sample_accuracy": row["sample_accuracy_mean"],
                }
            )
    return rows


def subspace_hmc_rows(root_name: str, label: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            posterior = row["posterior_jaccard_mean"]
            chain = row["chain_start_magnitude_to_imp_jaccard"]
            rewind = row["rewind_magnitude_to_imp_jaccard"]
            rows.append(
                {
                    "sampler": label,
                    "seed": payload["seed"],
                    "step": row["hmc_step_size"],
                    "posterior_minus_chain": posterior - chain,
                    "rewind_minus_posterior": rewind - posterior,
                    "post_chain": row[
                        "posterior_to_chain_start_magnitude_jaccard_mean"
                    ],
                    "sample_accuracy": row["sample_accuracy_mean"],
                    "accept_rate": row["hmc_accept_rate"],
                    "parameter_distance": row["hmc_parameter_distance_mean"],
                }
            )
    return rows


def calibration_ood_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for source, metrics in payload["sources"].items():
            rows.append(
                {
                    "seed": payload["seed"],
                    "source": source,
                    "id_accuracy": metrics["id"]["accuracy"],
                    "id_nll": metrics["id"]["nll"],
                    "id_brier": metrics["id"]["brier"],
                    "id_ece": metrics["id"]["ece"],
                    "ood_msp_auroc": metrics["ood"]["msp_auroc"],
                    "ood_msp_fpr95": metrics["ood"]["msp_fpr95"],
                    "ood_entropy_auroc": metrics["ood"]["entropy_auroc"],
                    "ood_entropy_fpr95": metrics["ood"]["entropy_fpr95"],
                }
            )
    return rows


def trajectory_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "epoch": row["epoch"],
                    "checkpoint_accuracy": row["checkpoint_accuracy"],
                    "trajectory_magnitude_to_imp": row[
                        "trajectory_magnitude_to_imp_jaccard"
                    ],
                    "trajectory_to_dense_final_magnitude": row[
                        "trajectory_to_dense_final_magnitude_jaccard"
                    ],
                    "trajectory_to_rewind_magnitude": row[
                        "trajectory_to_rewind_magnitude_jaccard"
                    ],
                    "dense_magnitude_to_imp": row["dense_magnitude_to_imp_jaccard"],
                    "rewind_magnitude_to_imp": row["rewind_magnitude_to_imp_jaccard"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def trajectory_aggregate_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("aggregate_rows", []):
            rows.append(
                {
                    "seed": payload["seed"],
                    "source": row["source"],
                    "score_to_imp": row["trajectory_score_to_imp_jaccard"],
                    "score_to_dense": row[
                        "trajectory_score_to_dense_final_magnitude_jaccard"
                    ],
                    "score_to_rewind": row[
                        "trajectory_score_to_rewind_magnitude_jaccard"
                    ],
                    "score_to_best_checkpoint": row[
                        "trajectory_score_to_best_checkpoint_jaccard"
                    ],
                    "score_minus_best_checkpoint": (
                        row["trajectory_score_to_imp_jaccard"]
                        - row["best_checkpoint_to_imp_jaccard"]
                    ),
                    "score_minus_dense": (
                        row["trajectory_score_to_imp_jaccard"]
                        - row["dense_magnitude_to_imp_jaccard"]
                    ),
                    "score_minus_rewind": (
                        row["trajectory_score_to_imp_jaccard"]
                        - row["rewind_magnitude_to_imp_jaccard"]
                    ),
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def trajectory_mask_training_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "source_kind": row["source_kind"],
                    "source": row["source"],
                    "trained_accuracy": row["trained_accuracy"],
                    "trained_ece": row.get("trained_ece"),
                    "trained_brier": row.get("trained_brier"),
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "source_to_imp": row["source_to_imp_jaccard"],
                    "source_to_dense": row[
                        "source_to_dense_final_magnitude_jaccard"
                    ],
                    "source_to_rewind": row["source_to_rewind_magnitude_jaccard"],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def summarize_mask_training_rows(
    rows: list[dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    summary = []
    by_mask_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mask_source[(row["source_kind"], row["source"])].append(row)
    for (source_kind, source), grouped_rows in sorted(by_mask_source.items()):
        summary.append(
            {
                "source_kind": source_kind,
                "source": source,
                "trained_accuracy": summarize_values(
                    f"{label} {source}: accuracy",
                    [row["trained_accuracy"] for row in grouped_rows],
                ),
                "trained_ece": summarize_optional_values(
                    f"{label} {source}: ECE",
                    [row.get("trained_ece") for row in grouped_rows],
                ),
                "trained_brier": summarize_optional_values(
                    f"{label} {source}: Brier",
                    [row.get("trained_brier") for row in grouped_rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"{label} {source}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in grouped_rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"{label} {source}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in grouped_rows],
                ),
                "source_to_imp": summarize_values(
                    f"{label} {source}: support - IMP",
                    [row["source_to_imp"] for row in grouped_rows],
                ),
                "source_to_dense": summarize_values(
                    f"{label} {source}: support - dense",
                    [row["source_to_dense"] for row in grouped_rows],
                ),
                "source_to_rewind": summarize_optional_values(
                    f"{label} {source}: support - rewind",
                    [row.get("source_to_rewind") for row in grouped_rows],
                ),
                "dense_accuracy": summarize_values(
                    f"{label} {source}: dense accuracy",
                    [row["dense_accuracy"] for row in grouped_rows],
                ),
                "imp_accuracy": summarize_values(
                    f"{label} {source}: IMP accuracy",
                    [row["imp_accuracy"] for row in grouped_rows],
                ),
            }
        )
    return summary


def trajectory_residual_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_anatomy_global_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("global_rows", []):
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "jaccard": row["jaccard"],
                    "imp_only": row["imp_only"],
                    "base_only": row["base_only"],
                    "base_only_pruned_round_mean": row[
                        "base_only_pruned_round_mean"
                    ],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_anatomy_group_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("group_rows", []):
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "unit": row["unit"],
                    "imp_only_share": row["imp_only_share"],
                    "base_only_share": row["base_only_share"],
                    "imp_only_enrichment": row["imp_only_enrichment"],
                    "base_only_enrichment": row["base_only_enrichment"],
                    "base_only_pruned_round_mean": row[
                        "base_only_pruned_round_mean"
                    ],
                }
            )
    return rows


def residual_anatomy_score_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("score_rows", []):
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "feature": row["feature"],
                    "auc": row["auc_imp_only_vs_nonbase"],
                    "topk_recall": row["topk_recall"],
                    "topk_lift": row["topk_lift"],
                }
            )
    return rows


def residual_anatomy_predictor_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("predictor_rows", []):
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "test_auc": row["test_auc"],
                    "test_topk_recall": row["test_topk_recall"],
                    "test_topk_precision": row["test_topk_precision"],
                    "test_baseline_precision": row["test_baseline_precision"],
                    "test_topk_lift": row["test_topk_lift"],
                }
            )
    return rows


def residual_predictor_mask_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "predictor_auc": row["predictor_auc"],
                    "predictor_topk_recall": row["predictor_topk_recall"],
                    "predictor_topk_precision": row["predictor_topk_precision"],
                    "predictor_baseline_precision": row[
                        "predictor_baseline_precision"
                    ],
                    "added_imp_only_precision": row["added_imp_only_precision"],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_cross_seed_transfer_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "target_seed": row["target_seed"],
                    "source_seeds": row["source_seeds"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "predictor_auc": row["predictor_auc"],
                    "predictor_topk_recall": row["predictor_topk_recall"],
                    "predictor_topk_precision": row["predictor_topk_precision"],
                    "predictor_baseline_precision": row[
                        "predictor_baseline_precision"
                    ],
                    "added_imp_only_precision": row["added_imp_only_precision"],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_direct_transfer_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("**/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "target_seed": row["target_seed"],
                    "source_seeds": row["source_seeds"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "candidate_count": row["candidate_count"],
                    "source_vote_positive_count": row[
                        "source_vote_positive_count"
                    ],
                    "source_vote_max": row["source_vote_max"],
                    "selected_source_vote_mean": row[
                        "selected_source_vote_mean"
                    ],
                    "selected_source_vote_positive_fraction": row[
                        "selected_source_vote_positive_fraction"
                    ],
                    "alignment_mean_corr": row.get("alignment_mean_corr"),
                    "alignment_min_corr": row.get("alignment_min_corr"),
                    "added_imp_only_precision": row[
                        "added_imp_only_precision"
                    ],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_base_compatibility_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("**/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "base_source": row["base_source"],
                    "base_kind": row["base_kind"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "base_to_imp": row["base_to_imp_jaccard"],
                    "base_to_reference": row["base_to_reference_jaccard"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_effective_base": row[
                        "mask_to_effective_base_jaccard"
                    ],
                    "mask_to_reference_base": row[
                        "mask_to_reference_base_jaccard"
                    ],
                    "swap_count": row["swap_count"],
                    "candidate_count": row["candidate_count"],
                    "heldout_count": row["heldout_count"],
                    "heldout_positive_count": row["heldout_positive_count"],
                    "added_imp_only_precision": row[
                        "added_imp_only_precision"
                    ],
                    "added_oracle_overlap_precision": row.get(
                        "added_oracle_overlap_precision"
                    ),
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def summarize_residual_base_rows(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    summary = []
    by_key: dict[tuple[str, str, str, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        by_key[
            (
                row["base_source"],
                row["base_kind"],
                row["variant"],
                float(row["alpha"]),
            )
        ].append(row)
    for (base_source, base_kind, variant, alpha), grouped_rows in sorted(
        by_key.items()
    ):
        summary.append(
            {
                "base_source": base_source,
                "base_kind": base_kind,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in grouped_rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in grouped_rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in grouped_rows],
                ),
                "base_to_imp": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: base - IMP",
                    [row["base_to_imp"] for row in grouped_rows],
                ),
                "base_to_reference": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: base - reference",
                    [row["base_to_reference"] for row in grouped_rows],
                ),
                "mask_to_imp": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in grouped_rows],
                ),
                "mask_to_effective_base": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: support - effective base",
                    [row["mask_to_effective_base"] for row in grouped_rows],
                ),
                "mask_to_reference_base": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: support - reference base",
                    [row["mask_to_reference_base"] for row in grouped_rows],
                ),
                "swap_count": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in grouped_rows],
                ),
                "candidate_count": summarize_optional_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: candidates",
                    [row["candidate_count"] for row in grouped_rows],
                ),
                "heldout_count": summarize_optional_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: heldout",
                    [row["heldout_count"] for row in grouped_rows],
                ),
                "heldout_positive_count": summarize_optional_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: heldout positives",
                    [row["heldout_positive_count"] for row in grouped_rows],
                ),
                "added_imp_only_precision": summarize_optional_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: added precision",
                    [row["added_imp_only_precision"] for row in grouped_rows],
                ),
                "added_oracle_overlap_precision": summarize_optional_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: oracle overlap precision",
                    [row["added_oracle_overlap_precision"] for row in grouped_rows],
                ),
                "dense_accuracy": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in grouped_rows],
                ),
                "imp_accuracy": summarize_values(
                    f"{label} {base_source} {base_kind} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in grouped_rows],
                ),
            }
        )
    return summary


def residual_stratified_control_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("**/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "added_imp_only_precision": row[
                        "added_imp_only_precision"
                    ],
                    "added_oracle_overlap_precision": row[
                        "added_oracle_overlap_precision"
                    ],
                    "stratum_exact_fraction": row["stratum_exact_fraction"],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def residual_imp_process_rows(root_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "runs" / root_name).glob("**/metrics.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            rows.append(
                {
                    "seed": payload["seed"],
                    "base_source": row["base_source"],
                    "variant": row["variant"],
                    "process_round": row["process_round"],
                    "alpha": row["alpha"],
                    "trained_accuracy": row["trained_accuracy"],
                    "accuracy_minus_imp": row["accuracy_minus_imp"],
                    "accuracy_minus_dense": row["accuracy_minus_dense"],
                    "mask_to_imp": row["mask_to_imp_jaccard"],
                    "mask_to_base": row["mask_to_base_jaccard"],
                    "swap_count": row["swap_count"],
                    "candidate_count": row["candidate_count"],
                    "added_final_imp_precision": row[
                        "added_final_imp_precision"
                    ],
                    "added_oracle_overlap_precision": row[
                        "added_oracle_overlap_precision"
                    ],
                    "dense_accuracy": row["dense_accuracy"],
                    "imp_accuracy": row["imp_accuracy"],
                }
            )
    return rows


def summarize_imp_process_rows(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    summary = []
    by_key: dict[tuple[str, str, int | None, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        process_round = (
            None if row["process_round"] is None else int(row["process_round"])
        )
        by_key[
            (row["base_source"], row["variant"], process_round, float(row["alpha"]))
        ].append(row)
    for (base_source, variant, process_round, alpha), grouped_rows in sorted(
        by_key.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
        ),
    ):
        summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "process_round": process_round,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in grouped_rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in grouped_rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in grouped_rows],
                ),
                "mask_to_imp": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in grouped_rows],
                ),
                "mask_to_base": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: support - base",
                    [row["mask_to_base"] for row in grouped_rows],
                ),
                "swap_count": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: swaps",
                    [row["swap_count"] for row in grouped_rows],
                ),
                "candidate_count": summarize_optional_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: candidates",
                    [row["candidate_count"] for row in grouped_rows],
                ),
                "added_final_imp_precision": summarize_optional_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: final IMP precision",
                    [row["added_final_imp_precision"] for row in grouped_rows],
                ),
                "added_oracle_overlap_precision": summarize_optional_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: oracle overlap",
                    [row["added_oracle_overlap_precision"] for row in grouped_rows],
                ),
                "dense_accuracy": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in grouped_rows],
                ),
                "imp_accuracy": summarize_values(
                    f"{label} {base_source} {variant} {process_round} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_oracle_matched_pairs(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, int, float, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        if row["process_round"] is None:
            continue
        process_round = int(row["process_round"])
        key = (row["base_source"], process_round, float(row["alpha"]), row["seed"])
        if row["variant"] == "round_final_imp_residual":
            pairs[key]["score"] = row
        elif row["variant"] == "round_final_imp_oracle_matched_random_residual":
            pairs[key]["matched_random"] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (base_source, process_round, alpha, _seed), pair in pairs.items():
        score = pair.get("score")
        matched = pair.get("matched_random")
        if score is None or matched is None:
            continue
        by_group[(base_source, process_round, alpha)].append(
            {
                "score_accuracy": score["trained_accuracy"],
                "matched_accuracy": matched["trained_accuracy"],
                "accuracy_delta": (
                    score["trained_accuracy"] - matched["trained_accuracy"]
                ),
                "oracle_overlap": score["added_oracle_overlap_precision"],
            }
        )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "score_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: score accuracy",
                    [row["score_accuracy"] for row in grouped_rows],
                ),
                "matched_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: matched accuracy",
                    [row["matched_accuracy"] for row in grouped_rows],
                ),
                "accuracy_delta": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: score - matched",
                    [row["accuracy_delta"] for row in grouped_rows],
                ),
                "oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: oracle overlap",
                    [row["oracle_overlap"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_score_source_pairs(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    by_seed_base: dict[tuple[int, str, float], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = (row["seed"], row["base_source"], float(row["alpha"]))
        if row["variant"] == "dense_score_final_imp_residual":
            by_seed_base[key]["dense_score"] = row
        elif row["variant"] == "base_score_final_imp_residual":
            by_seed_base[key]["base_score"] = row
        elif row["variant"] == "round_final_imp_residual":
            process_round = int(row["process_round"])
            by_seed_base[key].setdefault("rounds", {})[process_round] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (_seed, base_source, alpha), bundle in by_seed_base.items():
        dense_score = bundle.get("dense_score")
        base_score = bundle.get("base_score")
        if dense_score is None or base_score is None:
            continue
        for process_round, round_row in bundle.get("rounds", {}).items():
            by_group[(base_source, process_round, alpha)].append(
                {
                    "round_accuracy": round_row["trained_accuracy"],
                    "dense_score_accuracy": dense_score["trained_accuracy"],
                    "base_score_accuracy": base_score["trained_accuracy"],
                    "round_minus_dense_score": (
                        round_row["trained_accuracy"]
                        - dense_score["trained_accuracy"]
                    ),
                    "round_minus_base_score": (
                        round_row["trained_accuracy"]
                        - base_score["trained_accuracy"]
                    ),
                    "oracle_overlap": round_row["added_oracle_overlap_precision"],
                }
            )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "round_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round accuracy",
                    [row["round_accuracy"] for row in grouped_rows],
                ),
                "dense_score_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: dense-score accuracy",
                    [row["dense_score_accuracy"] for row in grouped_rows],
                ),
                "base_score_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: base-score accuracy",
                    [row["base_score_accuracy"] for row in grouped_rows],
                ),
                "round_minus_dense_score": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - dense-score",
                    [row["round_minus_dense_score"] for row in grouped_rows],
                ),
                "round_minus_base_score": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - base-score",
                    [row["round_minus_base_score"] for row in grouped_rows],
                ),
                "oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: oracle overlap",
                    [row["oracle_overlap"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_round_exclusion_pairs(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, int, float], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["process_round"] is None:
            continue
        key = (
            row["seed"],
            row["base_source"],
            int(row["process_round"]),
            float(row["alpha"]),
        )
        if row["variant"] == "round_final_imp_residual":
            pairs[key]["round"] = row
        elif row["variant"] == "round_excluded_oracle_final_imp_residual":
            pairs[key]["excluded"] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (_seed, base_source, process_round, alpha), pair in pairs.items():
        round_row = pair.get("round")
        excluded = pair.get("excluded")
        if round_row is None or excluded is None:
            continue
        by_group[(base_source, process_round, alpha)].append(
            {
                "round_accuracy": round_row["trained_accuracy"],
                "excluded_accuracy": excluded["trained_accuracy"],
                "accuracy_delta": (
                    round_row["trained_accuracy"]
                    - excluded["trained_accuracy"]
                ),
                "round_oracle_overlap": round_row[
                    "added_oracle_overlap_precision"
                ],
                "excluded_oracle_overlap": excluded[
                    "added_oracle_overlap_precision"
                ],
            }
        )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "round_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round accuracy",
                    [row["round_accuracy"] for row in grouped_rows],
                ),
                "excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded accuracy",
                    [row["excluded_accuracy"] for row in grouped_rows],
                ),
                "accuracy_delta": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - excluded",
                    [row["accuracy_delta"] for row in grouped_rows],
                ),
                "round_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round oracle overlap",
                    [row["round_oracle_overlap"] for row in grouped_rows],
                ),
                "excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded oracle overlap",
                    [row["excluded_oracle_overlap"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_layer_exclusion_pairs(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, int, float], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["process_round"] is None:
            continue
        key = (
            row["seed"],
            row["base_source"],
            int(row["process_round"]),
            float(row["alpha"]),
        )
        if row["variant"] == "round_final_imp_residual":
            pairs[key]["round"] = row
        elif row["variant"] == "round_excluded_oracle_final_imp_residual":
            pairs[key]["excluded"] = row
        elif row["variant"] == "round_excluded_layer_oracle_final_imp_residual":
            pairs[key]["layer_excluded"] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (_seed, base_source, process_round, alpha), pair in pairs.items():
        round_row = pair.get("round")
        excluded = pair.get("excluded")
        layer_excluded = pair.get("layer_excluded")
        if round_row is None or excluded is None or layer_excluded is None:
            continue
        by_group[(base_source, process_round, alpha)].append(
            {
                "round_accuracy": round_row["trained_accuracy"],
                "excluded_accuracy": excluded["trained_accuracy"],
                "layer_excluded_accuracy": layer_excluded["trained_accuracy"],
                "round_minus_excluded": (
                    round_row["trained_accuracy"] - excluded["trained_accuracy"]
                ),
                "round_minus_layer_excluded": (
                    round_row["trained_accuracy"]
                    - layer_excluded["trained_accuracy"]
                ),
                "round_oracle_overlap": round_row[
                    "added_oracle_overlap_precision"
                ],
                "excluded_oracle_overlap": excluded[
                    "added_oracle_overlap_precision"
                ],
                "layer_excluded_oracle_overlap": layer_excluded[
                    "added_oracle_overlap_precision"
                ],
            }
        )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "round_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round accuracy",
                    [row["round_accuracy"] for row in grouped_rows],
                ),
                "excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded accuracy",
                    [row["excluded_accuracy"] for row in grouped_rows],
                ),
                "layer_excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-matched excluded accuracy",
                    [row["layer_excluded_accuracy"] for row in grouped_rows],
                ),
                "round_minus_excluded": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - excluded",
                    [row["round_minus_excluded"] for row in grouped_rows],
                ),
                "round_minus_layer_excluded": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - tensor-matched excluded",
                    [row["round_minus_layer_excluded"] for row in grouped_rows],
                ),
                "round_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round oracle overlap",
                    [row["round_oracle_overlap"] for row in grouped_rows],
                ),
                "excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded oracle overlap",
                    [row["excluded_oracle_overlap"] for row in grouped_rows],
                ),
                "layer_excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-matched excluded oracle overlap",
                    [row["layer_excluded_oracle_overlap"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_tensor_score_exclusion_pairs(
    rows: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, int, float], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["process_round"] is None:
            continue
        key = (
            row["seed"],
            row["base_source"],
            int(row["process_round"]),
            float(row["alpha"]),
        )
        if row["variant"] == "round_final_imp_residual":
            pairs[key]["round"] = row
        elif row["variant"] == "round_excluded_oracle_final_imp_residual":
            pairs[key]["excluded"] = row
        elif row["variant"] == "round_excluded_layer_oracle_final_imp_residual":
            pairs[key]["layer_excluded"] = row
        elif row["variant"] == "round_excluded_tensor_score_oracle_final_imp_residual":
            pairs[key]["tensor_score_excluded"] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (_seed, base_source, process_round, alpha), pair in pairs.items():
        round_row = pair.get("round")
        excluded = pair.get("excluded")
        layer_excluded = pair.get("layer_excluded")
        tensor_score_excluded = pair.get("tensor_score_excluded")
        if (
            round_row is None
            or excluded is None
            or layer_excluded is None
            or tensor_score_excluded is None
        ):
            continue
        by_group[(base_source, process_round, alpha)].append(
            {
                "round_accuracy": round_row["trained_accuracy"],
                "excluded_accuracy": excluded["trained_accuracy"],
                "layer_excluded_accuracy": layer_excluded["trained_accuracy"],
                "tensor_score_excluded_accuracy": tensor_score_excluded[
                    "trained_accuracy"
                ],
                "round_minus_excluded": (
                    round_row["trained_accuracy"] - excluded["trained_accuracy"]
                ),
                "round_minus_layer_excluded": (
                    round_row["trained_accuracy"]
                    - layer_excluded["trained_accuracy"]
                ),
                "round_minus_tensor_score_excluded": (
                    round_row["trained_accuracy"]
                    - tensor_score_excluded["trained_accuracy"]
                ),
                "round_oracle_overlap": round_row[
                    "added_oracle_overlap_precision"
                ],
                "excluded_oracle_overlap": excluded[
                    "added_oracle_overlap_precision"
                ],
                "layer_excluded_oracle_overlap": layer_excluded[
                    "added_oracle_overlap_precision"
                ],
                "tensor_score_excluded_oracle_overlap": tensor_score_excluded[
                    "added_oracle_overlap_precision"
                ],
            }
        )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "round_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round accuracy",
                    [row["round_accuracy"] for row in grouped_rows],
                ),
                "excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded accuracy",
                    [row["excluded_accuracy"] for row in grouped_rows],
                ),
                "layer_excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-matched excluded accuracy",
                    [row["layer_excluded_accuracy"] for row in grouped_rows],
                ),
                "tensor_score_excluded_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-score-matched excluded accuracy",
                    [row["tensor_score_excluded_accuracy"] for row in grouped_rows],
                ),
                "round_minus_excluded": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - excluded",
                    [row["round_minus_excluded"] for row in grouped_rows],
                ),
                "round_minus_layer_excluded": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - tensor-matched excluded",
                    [row["round_minus_layer_excluded"] for row in grouped_rows],
                ),
                "round_minus_tensor_score_excluded": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - tensor-score-matched excluded",
                    [row["round_minus_tensor_score_excluded"] for row in grouped_rows],
                ),
                "round_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round oracle overlap",
                    [row["round_oracle_overlap"] for row in grouped_rows],
                ),
                "excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: excluded oracle overlap",
                    [row["excluded_oracle_overlap"] for row in grouped_rows],
                ),
                "layer_excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-matched excluded oracle overlap",
                    [row["layer_excluded_oracle_overlap"] for row in grouped_rows],
                ),
                "tensor_score_excluded_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: tensor-score-matched excluded oracle overlap",
                    [row["tensor_score_excluded_oracle_overlap"] for row in grouped_rows],
                ),
            }
        )
    return summary


def summarize_imp_process_projection_pairs(
    rows: list[dict[str, Any]],
    label: str,
    *,
    residualized_variant: str = "round_final_imp_residualized_score_residual",
    residualized_label: str = "residualized",
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, int, float], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key_base = (row["seed"], row["base_source"], float(row["alpha"]))
        if row["variant"] == "dense_score_final_imp_residual":
            pairs[(key_base[0], key_base[1], 5, key_base[2])]["dense_score"] = row
        elif row["variant"] == "base_score_final_imp_residual":
            pairs[(key_base[0], key_base[1], 5, key_base[2])]["base_score"] = row
        elif row["process_round"] is not None:
            key = (
                row["seed"],
                row["base_source"],
                int(row["process_round"]),
                float(row["alpha"]),
            )
            if row["variant"] == "round_final_imp_residual":
                pairs[key]["round"] = row
            elif row["variant"] == residualized_variant:
                pairs[key]["residualized"] = row

    by_group: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for (_seed, base_source, process_round, alpha), pair in pairs.items():
        round_row = pair.get("round")
        residualized = pair.get("residualized")
        if round_row is None or residualized is None:
            continue
        dense_score = pair.get("dense_score")
        base_score = pair.get("base_score")
        by_group[(base_source, process_round, alpha)].append(
            {
                "round_accuracy": round_row["trained_accuracy"],
                "residualized_accuracy": residualized["trained_accuracy"],
                "dense_score_accuracy": (
                    dense_score["trained_accuracy"] if dense_score is not None else math.nan
                ),
                "base_score_accuracy": (
                    base_score["trained_accuracy"] if base_score is not None else math.nan
                ),
                "round_minus_residualized": (
                    round_row["trained_accuracy"] - residualized["trained_accuracy"]
                ),
                "round_oracle_overlap": round_row[
                    "added_oracle_overlap_precision"
                ],
                "residualized_oracle_overlap": residualized[
                    "added_oracle_overlap_precision"
                ],
                "round_minus_residualized_oracle": (
                    round_row["added_oracle_overlap_precision"]
                    - residualized["added_oracle_overlap_precision"]
                ),
            }
        )

    summary = []
    for (base_source, process_round, alpha), grouped_rows in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "base_source": base_source,
                "process_round": process_round,
                "alpha": alpha,
                "round_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round accuracy",
                    [row["round_accuracy"] for row in grouped_rows],
                ),
                "residualized_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: {residualized_label} accuracy",
                    [row["residualized_accuracy"] for row in grouped_rows],
                ),
                "dense_score_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: dense-score accuracy",
                    [row["dense_score_accuracy"] for row in grouped_rows],
                ),
                "base_score_accuracy": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: base-score accuracy",
                    [row["base_score_accuracy"] for row in grouped_rows],
                ),
                "round_minus_residualized": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - {residualized_label}",
                    [row["round_minus_residualized"] for row in grouped_rows],
                ),
                "round_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round oracle overlap",
                    [row["round_oracle_overlap"] for row in grouped_rows],
                ),
                "residualized_oracle_overlap": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: {residualized_label} oracle overlap",
                    [row["residualized_oracle_overlap"] for row in grouped_rows],
                ),
                "round_minus_residualized_oracle": summarize_values(
                    f"{label} {base_source} {process_round} {alpha}: round - {residualized_label} oracle",
                    [row["round_minus_residualized_oracle"] for row in grouped_rows],
                ),
            }
        )
    return summary


def fmt(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if value == 0:
        return "0.0000"
    if abs(value) < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def fmt_ci(row: dict[str, Any]) -> str:
    if row.get("n") == 0:
        return "n/a"
    return f"{fmt(row['mean'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"


def selected_mode_equivalence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        ("MNIST Gate1", "r5 p0.30", "global", "posterior-chain"),
        ("Fashion Gate1", "r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SGLD-3chain", "30ep rewind r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SWAG", "30ep rewind r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SGLD movement", "lr=1.0e-06", "global", "posterior-chain"),
        ("CIFAR SGHMC movement", "lr=1.0e-07", "global", "posterior-chain"),
        ("CIFAR cSGLD movement", "lr=1.0e-06", "global", "posterior-chain"),
        ("CIFAR SWAG20 movement", "scale=1.6e+01", "global", "posterior-chain"),
        ("CIFAR DiagLap movement", "scale=1.0e-03", "global", "posterior-chain"),
        ("CIFAR KFACLap movement", "scale=1.0e-04", "global", "posterior-chain"),
        ("CIFAR LowRankLap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR LowRank32Lap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR HeadLap", "scale=1.0e-03", "head", "posterior-chain"),
        ("CIFAR BlockLap", "joint-group, scale=1.0e-04", "block", "posterior-chain"),
        ("CIFAR RandSubHMC", "step=3.0e-03", "global", "posterior-chain"),
        ("CIFAR TrajSubHMC", "step=1.0e-03", "global", "posterior-chain"),
        ("CIFAR Hess16SubHMC", "step=3.0e-04", "global", "posterior-chain"),
        ("CIFAR Hess32SubHMC", "step=3.0e-04", "global", "posterior-chain"),
    }
    return [
        row
        for row in rows
        if (
            row["family"],
            row["config"],
            row["scope"],
            row["comparison"],
        )
        in keep
    ]


def mode_equivalence_label(row: dict[str, Any]) -> str:
    labels = {
        ("MNIST Gate1", "r5 p0.30", "global"): "MNIST",
        ("Fashion Gate1", "r5 p0.30", "global"): "Fashion",
        ("CIFAR SGLD-3chain", "30ep rewind r5 p0.30", "global"): "SGLD-3",
        ("CIFAR SWAG", "30ep rewind r5 p0.30", "global"): "SWAG",
        ("CIFAR SGLD movement", "lr=1.0e-06", "global"): "SGLD move",
        ("CIFAR SGHMC movement", "lr=1.0e-07", "global"): "SGHMC move",
        ("CIFAR cSGLD movement", "lr=1.0e-06", "global"): "cSGLD move",
        ("CIFAR SWAG20 movement", "scale=1.6e+01", "global"): "SWAG20",
        ("CIFAR DiagLap movement", "scale=1.0e-03", "global"): "DiagLap",
        ("CIFAR KFACLap movement", "scale=1.0e-04", "global"): "KFACLap",
        ("CIFAR LowRankLap movement", "scale=1.0e-02", "global"): "LowRankLap",
        ("CIFAR LowRank32Lap movement", "scale=1.0e-02", "global"): "LowRank32Lap",
        ("CIFAR HeadLap", "scale=1.0e-03", "head"): "HeadLap",
        ("CIFAR BlockLap", "joint-group, scale=1.0e-04", "block"): "JointLap",
        ("CIFAR RandSubHMC", "step=3.0e-03", "global"): "RandHMC",
        ("CIFAR TrajSubHMC", "step=1.0e-03", "global"): "TrajHMC",
        ("CIFAR Hess16SubHMC", "step=3.0e-04", "global"): "Hess16HMC",
        ("CIFAR Hess32SubHMC", "step=3.0e-04", "global"): "Hess32HMC",
    }
    return labels.get(
        (row["family"], row["config"], row["scope"]),
        str(row["family"]).replace("_", " "),
    )


def tex_scale(value: float) -> str:
    exponent = int(math.floor(math.log10(value))) if value > 0 else 0
    if math.isclose(value, 10.0**exponent, rel_tol=1e-9, abs_tol=0.0):
        return f"10^{{{exponent}}}"
    mantissa = value / (10.0**exponent)
    if math.isclose(mantissa, round(mantissa), rel_tol=1e-9, abs_tol=1e-9):
        mantissa_text = f"{round(mantissa):.0f}"
    else:
        mantissa_text = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return f"{mantissa_text}\\times 10^{{{exponent}}}"


def source_label(source: str) -> str:
    label = source.replace("traj_", "").replace("_", " ")
    if label.startswith("epoch "):
        return f"Epoch {label.removeprefix('epoch ')}"
    if label == "imp":
        return "IMP"
    if label == "gem miner":
        return "Gem-Miner"
    if label == "variational prune":
        return "Var. prune"
    if label == "hard concrete":
        return "Hard concrete"
    return label


def variant_label(variant: str) -> str:
    labels = {
        "imp_residual": "top IMP",
        "imp_residual_random_remove": "top IMP, random rm",
        "imp_residual_high_remove": "top IMP, high rm",
        "random_residual": "random non-IMP",
        "oracle_top_imp_residual": "oracle top IMP",
        "random_imp_only_residual": "random IMP-only",
        "random_nonimp_global_residual": "global non-IMP",
        "random_nonimp_parameter_matched_residual": "parameter non-IMP",
        "random_nonimp_parameter_score_matched_residual": "param+score non-IMP",
        "final_oracle_residual": "final oracle",
        "round_survivor_residual": "round survivor",
        "round_survivor_random_residual": "round survivor random",
        "round_survivor_low_residual": "round survivor low",
        "round_final_imp_residual": "round final-IMP",
        "round_final_imp_residualized_score_residual": (
            "round residualized score"
        ),
        "round_final_imp_posterior_residualized_score_residual": (
            "round posterior-residualized score"
        ),
        "round_final_imp_learned_subspace_residualized_score_residual": (
            "round learned-subspace residualized score"
        ),
        "round_final_imp_oracle_matched_random_residual": (
            "round final-IMP oracle-match random"
        ),
        "dense_score_final_imp_residual": "dense-score final-IMP",
        "base_score_final_imp_residual": "base-score final-IMP",
        "round_excluded_oracle_final_imp_residual": (
            "round-excluded oracle final-IMP"
        ),
        "round_excluded_layer_oracle_final_imp_residual": (
            "round-excluded tensor-matched oracle final-IMP"
        ),
        "round_excluded_tensor_score_oracle_final_imp_residual": (
            "round-excluded tensor+score-matched oracle final-IMP"
        ),
        "target_oracle_residual": "target oracle",
        "source_vote_residual": "source vote",
        "source_vote_random_residual": "source-vote random",
        "target_random_residual": "target random",
        "oracle_imp_residual": "oracle residual",
        "posterior_imp_only_residual": "posterior IMP-only",
        "dense_imp_only_residual": "dense IMP-only",
        "posterior_excess_imp_only_residual": "posterior excess IMP-only",
        "posterior_std_imp_only_residual": "posterior std IMP-only",
        "low_imp_only_residual": "low IMP-only",
    }
    return labels.get(variant, variant.replace("_", " "))


def calibration_source_label(source: str) -> str:
    labels = {
        "dense": "Dense",
        "gem_miner": "Gem-Miner",
        "imp": "IMP",
        "learned_random_0": "Learned random",
        "swag_ensemble": "SWAG ens.",
        "swag_member_mean": "SWAG member",
        "variational_prune": "Var. prune",
        "hard_concrete": "Hard concrete",
    }
    return labels.get(source, source.replace("_", " "))


def fmt_alpha(value: float) -> str:
    return f"{value:g}"


def fmt_round(value: Any) -> str:
    if value is None:
        return ""
    return f"{int(value)}"


def fmt_k(value: float) -> str:
    return f"{value / 1000.0:.1f}k"


def build_stats() -> dict[str, Any]:
    mnist = gate1_rows("mnist", "mnist_gate1_full_*/*/metrics.json")
    fashion = gate1_rows("fashion", "fashion_gate1_full_*/*/metrics.json")

    gate1_summary = []
    for dataset, rows in [("MNIST", mnist), ("Fashion-MNIST", fashion)]:
        gate1_summary.extend(
            [
                summarize_values(
                    f"{dataset}: posterior - random",
                    [row["posterior_minus_random"] for row in rows],
                ),
                summarize_values(
                    f"{dataset}: posterior - chain-start",
                    [row["posterior_minus_chain"] for row in rows],
                ),
                summarize_values(
                    f"{dataset}: dense magnitude - posterior",
                    [row["dense_minus_posterior"] for row in rows],
                ),
                summarize_values(
                    f"{dataset}: posterior-to-chain support overlap",
                    [row["post_chain"] for row in rows],
                ),
            ]
        )

    movement_specs = [
        (
            "SGLD",
            "cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3",
        ),
        (
            "SGHMC",
            "cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3",
        ),
        (
            "cSGLD",
            "cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3",
        ),
        (
            "SWAG20",
            "cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3",
        ),
        (
            "DiagLap",
            "cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3",
        ),
        (
            "KFACLap",
            "cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3",
        ),
        (
            "LowRankLap",
            "cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3",
        ),
        (
            "LowRank32Lap",
            "cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3",
        ),
        (
            "LowRank64Lap",
            "cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3",
        ),
        (
            "LowRank128Lap",
            "cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3",
        ),
    ]
    movement = []
    for label, root in movement_specs:
        movement.extend(movement_rows(label, root))

    movement_summary = []
    by_sampler_scale: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in movement:
        by_sampler_scale[(row["sampler"], float(row["scale"]))].append(row)
    for (sampler, scale), rows in sorted(by_sampler_scale.items()):
        if scale <= 1e-10:
            continue
        movement_summary.append(
            {
                "sampler": sampler,
                "scale": scale,
                "posterior_minus_chain": summarize_values(
                    f"{sampler} {scale:g}: posterior - chain-start",
                    [row["posterior_minus_chain"] for row in rows],
                ),
                "rewind_minus_posterior": summarize_values(
                    f"{sampler} {scale:g}: rewind - posterior",
                    [row["rewind_minus_posterior"] for row in rows],
                ),
                "post_chain": summarize_values(
                    f"{sampler} {scale:g}: post-chain",
                    [row["post_chain"] for row in rows],
                ),
                "sample_accuracy": summarize_values(
                    f"{sampler} {scale:g}: sample accuracy",
                    [row["sample_accuracy"] for row in rows],
                ),
            }
        )

    head_laplace = head_laplace_rows(
        "cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3"
    )
    head_laplace_summary = []
    by_head_scale: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in head_laplace:
        by_head_scale[float(row["scale"])].append(row)
    for scale, rows in sorted(by_head_scale.items()):
        head_laplace_summary.append(
            {
                "sampler": "HeadLap",
                "scale": scale,
                "posterior_minus_chain": summarize_values(
                    f"HeadLap {scale:g}: head posterior - head chain-start",
                    [row["posterior_minus_chain"] for row in rows],
                ),
                "rewind_minus_posterior": summarize_values(
                    f"HeadLap {scale:g}: head rewind - head posterior",
                    [row["rewind_minus_posterior"] for row in rows],
                ),
                "post_chain": summarize_values(
                    f"HeadLap {scale:g}: head post-chain",
                    [row["post_chain"] for row in rows],
                ),
                "sample_accuracy": summarize_values(
                    f"HeadLap {scale:g}: sample accuracy",
                    [row["sample_accuracy"] for row in rows],
                ),
            }
        )

    block_laplace = []
    for root_name in [
        "cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3",
    ]:
        block_laplace.extend(block_laplace_rows(root_name))
    block_laplace_summary = []
    by_block_scale: dict[tuple[str, str, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in block_laplace:
        by_block_scale[(row["sampler"], row["block"], float(row["scale"]))].append(row)
    for (sampler, block, scale), rows in sorted(by_block_scale.items()):
        block_laplace_summary.append(
            {
                "sampler": sampler,
                "block": block,
                "scale": scale,
                "parameter_count": summarize_values(
                    f"{sampler} {block} {scale:g}: parameter count",
                    [row["parameter_count"] for row in rows],
                ),
                "block_posterior_minus_chain": summarize_values(
                    f"{sampler} {block} {scale:g}: block posterior - block chain-start",
                    [row["block_posterior_minus_chain"] for row in rows],
                ),
                "block_rewind_minus_posterior": summarize_values(
                    f"{sampler} {block} {scale:g}: block rewind - block posterior",
                    [row["block_rewind_minus_posterior"] for row in rows],
                ),
                "block_post_chain": summarize_values(
                    f"{sampler} {block} {scale:g}: block post-chain",
                    [row["block_post_chain"] for row in rows],
                ),
                "global_posterior_minus_chain": summarize_values(
                    f"{sampler} {block} {scale:g}: global posterior - global chain-start",
                    [row["global_posterior_minus_chain"] for row in rows],
                ),
                "global_rewind_minus_posterior": summarize_values(
                    f"{sampler} {block} {scale:g}: global rewind - global posterior",
                    [row["global_rewind_minus_posterior"] for row in rows],
                ),
                "global_post_chain": summarize_values(
                    f"{sampler} {block} {scale:g}: global post-chain",
                    [row["global_post_chain"] for row in rows],
                ),
                "sample_accuracy": summarize_values(
                    f"{sampler} {block} {scale:g}: sample accuracy",
                    [row["sample_accuracy"] for row in rows],
                ),
            }
        )

    subspace_hmc_specs = [
        (
            "RandSubHMC",
            "cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3",
        ),
        (
            "TrajSubHMC",
            "cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3",
        ),
        (
            "HessSubHMC",
            "cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3",
        ),
        (
            "Hess16SubHMC",
            "cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3",
        ),
        (
            "Hess32SubHMC",
            "cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3",
        ),
    ]
    subspace_hmc = []
    for label, root in subspace_hmc_specs:
        subspace_hmc.extend(subspace_hmc_rows(root, label))
    subspace_hmc_summary = []
    by_hmc_step: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in subspace_hmc:
        by_hmc_step[(row["sampler"], float(row["step"]))].append(row)
    for (sampler, step), rows in sorted(by_hmc_step.items()):
        subspace_hmc_summary.append(
            {
                "sampler": sampler,
                "step": step,
                "posterior_minus_chain": summarize_values(
                    f"{sampler} {step:g}: posterior - chain-start",
                    [row["posterior_minus_chain"] for row in rows],
                ),
                "rewind_minus_posterior": summarize_values(
                    f"{sampler} {step:g}: rewind - posterior",
                    [row["rewind_minus_posterior"] for row in rows],
                ),
                "post_chain": summarize_values(
                    f"{sampler} {step:g}: post-chain",
                    [row["post_chain"] for row in rows],
                ),
                "sample_accuracy": summarize_values(
                    f"{sampler} {step:g}: sample accuracy",
                    [row["sample_accuracy"] for row in rows],
                ),
                "accept_rate": summarize_values(
                    f"{sampler} {step:g}: accept rate",
                    [row["accept_rate"] for row in rows],
                ),
                "parameter_distance": summarize_values(
                    f"{sampler} {step:g}: parameter distance",
                    [row["parameter_distance"] for row in rows],
                ),
            }
        )

    calibration_ood = calibration_ood_rows(
        "cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3"
    )
    calibration_learned_ood = [
        row
        for row in calibration_ood_rows(
            "cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3"
        )
        if row["source"]
        in {"learned_random_0", "gem_miner", "variational_prune", "hard_concrete"}
    ]
    calibration_ood.extend(calibration_learned_ood)
    calibration_ood_summary = []
    by_calibration_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_ood:
        by_calibration_source[row["source"]].append(row)
    calibration_order = [
        "dense",
        "imp",
        "swag_ensemble",
        "swag_member_mean",
        "learned_random_0",
        "gem_miner",
        "variational_prune",
        "hard_concrete",
    ]
    ordered_sources = [
        source for source in calibration_order if source in by_calibration_source
    ]
    ordered_sources.extend(
        sorted(source for source in by_calibration_source if source not in ordered_sources)
    )
    for source in ordered_sources:
        rows = by_calibration_source[source]
        calibration_ood_summary.append(
            {
                "source": source,
                "id_accuracy": summarize_values(
                    f"{source}: ID accuracy",
                    [row["id_accuracy"] for row in rows],
                ),
                "id_nll": summarize_values(
                    f"{source}: ID NLL",
                    [row["id_nll"] for row in rows],
                ),
                "id_brier": summarize_values(
                    f"{source}: ID Brier",
                    [row["id_brier"] for row in rows],
                ),
                "id_ece": summarize_values(
                    f"{source}: ID ECE",
                    [row["id_ece"] for row in rows],
                ),
                "ood_msp_auroc": summarize_values(
                    f"{source}: CIFAR100 MSP AUROC",
                    [row["ood_msp_auroc"] for row in rows],
                ),
                "ood_msp_fpr95": summarize_values(
                    f"{source}: CIFAR100 MSP FPR95",
                    [row["ood_msp_fpr95"] for row in rows],
                ),
                "ood_entropy_auroc": summarize_values(
                    f"{source}: CIFAR100 entropy AUROC",
                    [row["ood_entropy_auroc"] for row in rows],
                ),
                "ood_entropy_fpr95": summarize_values(
                    f"{source}: CIFAR100 entropy FPR95",
                    [row["ood_entropy_fpr95"] for row in rows],
                ),
            }
        )

    trajectory_root = "cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2"
    trajectory = trajectory_rows(trajectory_root)
    trajectory_summary = []
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory:
        by_epoch[int(row["epoch"])].append(row)
    for epoch, rows in sorted(by_epoch.items()):
        trajectory_summary.append(
            {
                "epoch": epoch,
                "checkpoint_accuracy": summarize_values(
                    f"Trajectory epoch {epoch}: checkpoint accuracy",
                    [row["checkpoint_accuracy"] for row in rows],
                ),
                "trajectory_magnitude_to_imp": summarize_values(
                    f"Trajectory epoch {epoch}: trajectory - IMP",
                    [row["trajectory_magnitude_to_imp"] for row in rows],
                ),
                "trajectory_to_dense_final_magnitude": summarize_values(
                    f"Trajectory epoch {epoch}: trajectory - dense final",
                    [row["trajectory_to_dense_final_magnitude"] for row in rows],
                ),
                "trajectory_to_rewind_magnitude": summarize_values(
                    f"Trajectory epoch {epoch}: trajectory - rewind",
                    [row["trajectory_to_rewind_magnitude"] for row in rows],
                ),
                "dense_magnitude_to_imp": summarize_values(
                    f"Trajectory epoch {epoch}: dense - IMP",
                    [row["dense_magnitude_to_imp"] for row in rows],
                ),
                "rewind_magnitude_to_imp": summarize_values(
                    f"Trajectory epoch {epoch}: rewind - IMP",
                    [row["rewind_magnitude_to_imp"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Trajectory epoch {epoch}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    trajectory_aggregate = trajectory_aggregate_rows(trajectory_root)
    trajectory_aggregate_summary = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_aggregate:
        by_source[row["source"]].append(row)
    for source, rows in sorted(by_source.items()):
        trajectory_aggregate_summary.append(
            {
                "source": source,
                "score_to_imp": summarize_values(
                    f"Trajectory aggregate {source}: score - IMP",
                    [row["score_to_imp"] for row in rows],
                ),
                "score_to_dense": summarize_values(
                    f"Trajectory aggregate {source}: score - dense",
                    [row["score_to_dense"] for row in rows],
                ),
                "score_to_rewind": summarize_values(
                    f"Trajectory aggregate {source}: score - rewind",
                    [row["score_to_rewind"] for row in rows],
                ),
                "score_to_best_checkpoint": summarize_values(
                    f"Trajectory aggregate {source}: score - best checkpoint",
                    [row["score_to_best_checkpoint"] for row in rows],
                ),
                "score_minus_best_checkpoint": summarize_values(
                    f"Trajectory aggregate {source}: score - best checkpoint IMP",
                    [row["score_minus_best_checkpoint"] for row in rows],
                ),
                "score_minus_dense": summarize_values(
                    f"Trajectory aggregate {source}: score - dense IMP",
                    [row["score_minus_dense"] for row in rows],
                ),
                "score_minus_rewind": summarize_values(
                    f"Trajectory aggregate {source}: score - rewind IMP",
                    [row["score_minus_rewind"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Trajectory aggregate {source}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    trajectory_mask_training = trajectory_mask_training_rows(
        "cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3"
    )
    trajectory_mask_training.extend(
        trajectory_mask_training_rows(
            "cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3"
        )
    )
    trajectory_mask_training.extend(
        trajectory_mask_training_rows(
            "cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3"
        )
    )
    trajectory_mask_training.extend(
        trajectory_mask_training_rows(
            "cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3"
        )
    )
    trajectory_mask_training_summary = summarize_mask_training_rows(
        trajectory_mask_training,
        "Trajectory mask training",
    )
    variational_pruning_summary = summarize_mask_training_rows(
        trajectory_mask_training_rows("digits_mlp_variational_prune_calib_r5"),
        "Digits variational pruning",
    )

    trajectory_residual = trajectory_residual_rows(
        "cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3"
    )
    trajectory_residual_summary = []
    by_residual: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_residual:
        by_residual[(row["base_source"], row["variant"], float(row["alpha"]))].append(row)
    for (base_source, variant, alpha), rows in sorted(by_residual.items()):
        trajectory_residual_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Trajectory residual {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    trajectory_residual_removal_controls = trajectory_residual_rows(
        "cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3"
    )
    trajectory_residual_removal_controls_summary = []
    by_residual_removal: dict[
        tuple[str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in trajectory_residual_removal_controls:
        by_residual_removal[
            (row["base_source"], row["variant"], float(row["alpha"]))
        ].append(row)
    for (base_source, variant, alpha), rows in sorted(by_residual_removal.items()):
        trajectory_residual_removal_controls_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Trajectory residual removal {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    anatomy_root = "cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3"
    anatomy_global = residual_anatomy_global_rows(anatomy_root)
    anatomy_global_summary = []
    by_anatomy_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anatomy_global:
        by_anatomy_base[row["base_source"]].append(row)
    for base_source, rows in sorted(by_anatomy_base.items()):
        anatomy_global_summary.append(
            {
                "base_source": base_source,
                "jaccard": summarize_values(
                    f"Residual anatomy {base_source}: support - IMP",
                    [row["jaccard"] for row in rows],
                ),
                "imp_only": summarize_values(
                    f"Residual anatomy {base_source}: IMP-only count",
                    [row["imp_only"] for row in rows],
                ),
                "base_only": summarize_values(
                    f"Residual anatomy {base_source}: base-only count",
                    [row["base_only"] for row in rows],
                ),
                "base_only_pruned_round_mean": summarize_values(
                    f"Residual anatomy {base_source}: base-only prune round",
                    [row["base_only_pruned_round_mean"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Residual anatomy {base_source}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Residual anatomy {base_source}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    anatomy_groups = residual_anatomy_group_rows(anatomy_root)
    anatomy_group_summary = []
    by_anatomy_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anatomy_groups:
        by_anatomy_group[(row["base_source"], row["unit"])].append(row)
    for (base_source, unit), rows in sorted(by_anatomy_group.items()):
        anatomy_group_summary.append(
            {
                "base_source": base_source,
                "unit": unit,
                "imp_only_share": summarize_values(
                    f"Residual anatomy {base_source} {unit}: IMP-only share",
                    [row["imp_only_share"] for row in rows],
                ),
                "base_only_share": summarize_values(
                    f"Residual anatomy {base_source} {unit}: base-only share",
                    [row["base_only_share"] for row in rows],
                ),
                "imp_only_enrichment": summarize_values(
                    f"Residual anatomy {base_source} {unit}: IMP-only enrichment",
                    [row["imp_only_enrichment"] for row in rows],
                ),
                "base_only_pruned_round_mean": summarize_values(
                    f"Residual anatomy {base_source} {unit}: base-only prune round",
                    [row["base_only_pruned_round_mean"] for row in rows],
                ),
            }
        )

    anatomy_scores = residual_anatomy_score_rows(anatomy_root)
    anatomy_score_summary = []
    by_anatomy_score: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anatomy_scores:
        by_anatomy_score[(row["base_source"], row["feature"])].append(row)
    for (base_source, feature), rows in sorted(by_anatomy_score.items()):
        anatomy_score_summary.append(
            {
                "base_source": base_source,
                "feature": feature,
                "auc": summarize_values(
                    f"Residual anatomy {base_source} {feature}: AUC",
                    [row["auc"] for row in rows],
                ),
                "topk_recall": summarize_values(
                    f"Residual anatomy {base_source} {feature}: top-k recall",
                    [row["topk_recall"] for row in rows],
                ),
                "topk_lift": summarize_values(
                    f"Residual anatomy {base_source} {feature}: top-k lift",
                    [row["topk_lift"] for row in rows],
                ),
            }
        )

    anatomy_predictors = residual_anatomy_predictor_rows(anatomy_root)
    anatomy_predictor_summary = []
    by_anatomy_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anatomy_predictors:
        by_anatomy_predictor[row["base_source"]].append(row)
    for base_source, rows in sorted(by_anatomy_predictor.items()):
        anatomy_predictor_summary.append(
            {
                "base_source": base_source,
                "test_auc": summarize_values(
                    f"Residual anatomy {base_source}: learned AUC",
                    [row["test_auc"] for row in rows],
                ),
                "test_topk_recall": summarize_values(
                    f"Residual anatomy {base_source}: learned top-k recall",
                    [row["test_topk_recall"] for row in rows],
                ),
                "test_baseline_precision": summarize_values(
                    f"Residual anatomy {base_source}: baseline precision",
                    [row["test_baseline_precision"] for row in rows],
                ),
                "test_topk_lift": summarize_values(
                    f"Residual anatomy {base_source}: learned top-k lift",
                    [row["test_topk_lift"] for row in rows],
                ),
            }
        )

    predictor_mask = residual_predictor_mask_rows(
        "cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3"
    )
    predictor_mask_summary = []
    by_predictor_mask: dict[
        tuple[str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in predictor_mask:
        by_predictor_mask[
            (row["base_source"], row["variant"], float(row["alpha"]))
        ].append(row)
    for (base_source, variant, alpha), rows in sorted(by_predictor_mask.items()):
        predictor_mask_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "predictor_auc": summarize_optional_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: predictor AUC",
                    [row["predictor_auc"] for row in rows],
                ),
                "predictor_topk_recall": summarize_optional_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: predictor top-k recall",
                    [row["predictor_topk_recall"] for row in rows],
                ),
                "predictor_topk_precision": summarize_optional_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: predictor top-k precision",
                    [row["predictor_topk_precision"] for row in rows],
                ),
                "predictor_baseline_precision": summarize_optional_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: baseline precision",
                    [row["predictor_baseline_precision"] for row in rows],
                ),
                "added_imp_only_precision": summarize_optional_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: added precision",
                    [row["added_imp_only_precision"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Residual predictor mask {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    cross_seed_transfer = residual_cross_seed_transfer_rows(
        "cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3"
    )
    cross_seed_transfer_summary = []
    by_cross_seed: dict[
        tuple[str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in cross_seed_transfer:
        by_cross_seed[
            (row["base_source"], row["variant"], float(row["alpha"]))
        ].append(row)
    for (base_source, variant, alpha), rows in sorted(by_cross_seed.items()):
        cross_seed_transfer_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "predictor_auc": summarize_optional_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: predictor AUC",
                    [row["predictor_auc"] for row in rows],
                ),
                "predictor_topk_recall": summarize_optional_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: predictor top-k recall",
                    [row["predictor_topk_recall"] for row in rows],
                ),
                "predictor_topk_precision": summarize_optional_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: predictor top-k precision",
                    [row["predictor_topk_precision"] for row in rows],
                ),
                "predictor_baseline_precision": summarize_optional_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: baseline precision",
                    [row["predictor_baseline_precision"] for row in rows],
                ),
                "added_imp_only_precision": summarize_optional_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: added precision",
                    [row["added_imp_only_precision"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Residual cross-seed transfer {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    direct_transfer = residual_direct_transfer_rows(
        "cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3"
    )
    direct_transfer_summary = []
    by_direct_transfer: dict[
        tuple[str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in direct_transfer:
        by_direct_transfer[
            (row["base_source"], row["variant"], float(row["alpha"]))
        ].append(row)
    for (base_source, variant, alpha), rows in sorted(by_direct_transfer.items()):
        direct_transfer_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "candidate_count": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: candidates",
                    [row["candidate_count"] for row in rows],
                ),
                "source_vote_positive_count": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: vote positives",
                    [row["source_vote_positive_count"] for row in rows],
                ),
                "source_vote_max": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: vote max",
                    [row["source_vote_max"] for row in rows],
                ),
                "selected_source_vote_mean": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: selected vote mean",
                    [row["selected_source_vote_mean"] for row in rows],
                ),
                "selected_source_vote_positive_fraction": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: selected vote positive fraction",
                    [row["selected_source_vote_positive_fraction"] for row in rows],
                ),
                "added_imp_only_precision": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: added precision",
                    [row["added_imp_only_precision"] for row in rows],
                ),
                "alignment_mean_corr": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: alignment mean corr",
                    [row["alignment_mean_corr"] for row in rows],
                ),
                "alignment_min_corr": summarize_optional_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: alignment min corr",
                    [row["alignment_min_corr"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Residual direct transfer {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    base_compatibility = residual_base_compatibility_rows(
        "cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3"
    )
    base_compatibility_summary = summarize_residual_base_rows(
        base_compatibility, "Residual base compatibility"
    )
    base_ordering = residual_base_compatibility_rows(
        "cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3"
    )
    base_ordering_summary = summarize_residual_base_rows(
        base_ordering, "Residual base ordering"
    )

    stratified_controls = residual_stratified_control_rows(
        "cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3"
    )
    stratified_controls_summary = []
    by_stratified: dict[
        tuple[str, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in stratified_controls:
        by_stratified[
            (row["base_source"], row["variant"], float(row["alpha"]))
        ].append(row)
    for (base_source, variant, alpha), rows in sorted(by_stratified.items()):
        stratified_controls_summary.append(
            {
                "base_source": base_source,
                "variant": variant,
                "alpha": alpha,
                "trained_accuracy": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: accuracy",
                    [row["trained_accuracy"] for row in rows],
                ),
                "accuracy_minus_imp": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: accuracy - IMP",
                    [row["accuracy_minus_imp"] for row in rows],
                ),
                "accuracy_minus_dense": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: accuracy - dense",
                    [row["accuracy_minus_dense"] for row in rows],
                ),
                "mask_to_imp": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: support - IMP",
                    [row["mask_to_imp"] for row in rows],
                ),
                "mask_to_base": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: support - base",
                    [row["mask_to_base"] for row in rows],
                ),
                "swap_count": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: swaps",
                    [row["swap_count"] for row in rows],
                ),
                "added_imp_only_precision": summarize_optional_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: added precision",
                    [row["added_imp_only_precision"] for row in rows],
                ),
                "added_oracle_overlap_precision": summarize_optional_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: oracle overlap",
                    [row["added_oracle_overlap_precision"] for row in rows],
                ),
                "stratum_exact_fraction": summarize_optional_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: exact strata",
                    [row["stratum_exact_fraction"] for row in rows],
                ),
                "dense_accuracy": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: dense accuracy",
                    [row["dense_accuracy"] for row in rows],
                ),
                "imp_accuracy": summarize_values(
                    f"Residual stratified control {base_source} {variant} {alpha}: IMP accuracy",
                    [row["imp_accuracy"] for row in rows],
                ),
            }
        )

    imp_process = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3"
    )
    imp_process_summary = summarize_imp_process_rows(
        imp_process, "Residual IMP process"
    )
    imp_process_controls = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3"
    )
    imp_process_controls_summary = summarize_imp_process_rows(
        imp_process_controls, "Residual IMP process controls"
    )
    imp_process_oracle_matched = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3"
    )
    imp_process_oracle_matched_summary = summarize_imp_process_rows(
        imp_process_oracle_matched, "Residual IMP process oracle-matched controls"
    )
    imp_process_oracle_matched_pairs = (
        summarize_imp_process_oracle_matched_pairs(
            imp_process_oracle_matched,
            "Residual IMP process oracle-matched pairs",
        )
    )
    imp_process_score_source = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3"
    )
    imp_process_score_source_summary = summarize_imp_process_rows(
        imp_process_score_source,
        "Residual IMP process score-source controls",
    )
    imp_process_score_source_pairs = summarize_imp_process_score_source_pairs(
        imp_process_score_source,
        "Residual IMP process score-source pairs",
    )
    imp_process_round_exclusion = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3"
    )
    imp_process_round_exclusion_summary = summarize_imp_process_rows(
        imp_process_round_exclusion,
        "Residual IMP process round-exclusion controls",
    )
    imp_process_round_exclusion_pairs = (
        summarize_imp_process_round_exclusion_pairs(
            imp_process_round_exclusion,
            "Residual IMP process round-exclusion pairs",
        )
    )
    imp_process_layer_exclusion = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3"
    )
    imp_process_layer_exclusion_summary = summarize_imp_process_rows(
        imp_process_layer_exclusion,
        "Residual IMP process tensor-matched round-exclusion controls",
    )
    imp_process_layer_exclusion_pairs = (
        summarize_imp_process_layer_exclusion_pairs(
            imp_process_layer_exclusion,
            "Residual IMP process tensor-matched round-exclusion pairs",
        )
    )
    imp_process_tensor_score_exclusion = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3"
    )
    imp_process_tensor_score_exclusion_summary = summarize_imp_process_rows(
        imp_process_tensor_score_exclusion,
        "Residual IMP process tensor+score-matched round-exclusion controls",
    )
    imp_process_tensor_score_exclusion_pairs = (
        summarize_imp_process_tensor_score_exclusion_pairs(
            imp_process_tensor_score_exclusion,
            "Residual IMP process tensor+score-matched round-exclusion pairs",
        )
    )
    imp_process_projection = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3"
    )
    imp_process_projection_summary = summarize_imp_process_rows(
        imp_process_projection,
        "Residual IMP process projection controls",
    )
    imp_process_projection_pairs = summarize_imp_process_projection_pairs(
        imp_process_projection,
        "Residual IMP process projection pairs",
    )
    imp_process_posterior_projection = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3"
    )
    imp_process_posterior_projection_summary = summarize_imp_process_rows(
        imp_process_posterior_projection,
        "Residual IMP process posterior projection controls",
    )
    imp_process_posterior_projection_pairs = summarize_imp_process_projection_pairs(
        imp_process_posterior_projection,
        "Residual IMP process posterior projection pairs",
        residualized_variant=(
            "round_final_imp_posterior_residualized_score_residual"
        ),
        residualized_label="posterior-residualized",
    )
    imp_process_learned_subspace = residual_imp_process_rows(
        "cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3"
    )
    imp_process_learned_subspace_summary = summarize_imp_process_rows(
        imp_process_learned_subspace,
        "Residual IMP process learned-subspace controls",
    )
    imp_process_learned_subspace_pairs = summarize_imp_process_projection_pairs(
        imp_process_learned_subspace,
        "Residual IMP process learned-subspace pairs",
        residualized_variant=(
            "round_final_imp_learned_subspace_residualized_score_residual"
        ),
        residualized_label="learned-subspace",
    )
    mode_distribution_equivalence = read_mode_distribution_equivalence_rows()
    direct_mode_ticket_distribution = read_direct_mode_ticket_distribution_rows()

    return {
        "gate1": gate1_summary,
        "movement": movement_summary,
        "head_laplace": head_laplace_summary,
        "block_laplace": block_laplace_summary,
        "subspace_hmc": subspace_hmc_summary,
        "mode_distribution_equivalence": mode_distribution_equivalence,
        "direct_mode_ticket_distribution": direct_mode_ticket_distribution,
        "calibration_ood": calibration_ood_summary,
        "trajectory": trajectory_summary,
        "trajectory_aggregate": trajectory_aggregate_summary,
        "trajectory_mask_training": trajectory_mask_training_summary,
        "variational_pruning": variational_pruning_summary,
        "trajectory_residual": trajectory_residual_summary,
        "trajectory_residual_removal_controls": (
            trajectory_residual_removal_controls_summary
        ),
        "residual_anatomy_global": anatomy_global_summary,
        "residual_anatomy_group": anatomy_group_summary,
        "residual_anatomy_score": anatomy_score_summary,
        "residual_anatomy_predictor": anatomy_predictor_summary,
        "residual_predictor_mask": predictor_mask_summary,
        "residual_cross_seed_transfer": cross_seed_transfer_summary,
        "residual_direct_transfer": direct_transfer_summary,
        "residual_base_compatibility": base_compatibility_summary,
        "residual_base_ordering": base_ordering_summary,
        "residual_stratified_controls": stratified_controls_summary,
        "residual_imp_process": imp_process_summary,
        "residual_imp_process_controls": imp_process_controls_summary,
        "residual_imp_process_oracle_matched": (
            imp_process_oracle_matched_summary
        ),
        "residual_imp_process_oracle_matched_pairs": (
            imp_process_oracle_matched_pairs
        ),
        "residual_imp_process_score_source": (
            imp_process_score_source_summary
        ),
        "residual_imp_process_score_source_pairs": (
            imp_process_score_source_pairs
        ),
        "residual_imp_process_round_exclusion": (
            imp_process_round_exclusion_summary
        ),
        "residual_imp_process_round_exclusion_pairs": (
            imp_process_round_exclusion_pairs
        ),
        "residual_imp_process_layer_exclusion": (
            imp_process_layer_exclusion_summary
        ),
        "residual_imp_process_layer_exclusion_pairs": (
            imp_process_layer_exclusion_pairs
        ),
        "residual_imp_process_tensor_score_exclusion": (
            imp_process_tensor_score_exclusion_summary
        ),
        "residual_imp_process_tensor_score_exclusion_pairs": (
            imp_process_tensor_score_exclusion_pairs
        ),
        "residual_imp_process_projection": imp_process_projection_summary,
        "residual_imp_process_projection_pairs": imp_process_projection_pairs,
        "residual_imp_process_posterior_projection": (
            imp_process_posterior_projection_summary
        ),
        "residual_imp_process_posterior_projection_pairs": (
            imp_process_posterior_projection_pairs
        ),
        "residual_imp_process_learned_subspace": (
            imp_process_learned_subspace_summary
        ),
        "residual_imp_process_learned_subspace_pairs": (
            imp_process_learned_subspace_pairs
        ),
    }


def write_markdown(stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paper Statistical Audit",
        "",
        "All intervals are seed/config-level 95% t intervals. With five seeds,",
        "the sign counts are often more informative than the normality assumption.",
        "",
        "## Gate1 Controls",
        "",
        "| Claim | n | Mean [95% CI] | + | - | 0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["gate1"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {fmt_ci(row)} | "
            f"{row['positive']} | {row['negative']} | {row['zero']} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Movement Diagnostics",
            "",
            "| Sampler | Scale | Post-Chain | Posterior-Chain [95% CI] | "
            "Rewind-Posterior [95% CI] | Sample Acc. |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["movement"]:
        lines.append(
            f"| {row['sampler']} | {row['scale']:.1e} | "
            f"{fmt(row['post_chain']['mean'])} | "
            f"{fmt_ci(row['posterior_minus_chain'])} | "
            f"{fmt_ci(row['rewind_minus_posterior'])} | "
            f"{fmt(row['sample_accuracy']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Exact Head Laplace Probe",
            "",
            "These rows are exact full-covariance Laplace probes for the final",
            "linear classification head only, so head-level support is the primary",
            "quantity and global support is auxiliary.",
            "",
            "| Sampler | Scale | Head Post-Chain | Head Posterior-Chain [95% CI] | "
            "Head Rewind-Posterior [95% CI] | Sample Acc. |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["head_laplace"]:
        lines.append(
            f"| {row['sampler']} | {row['scale']:.1e} | "
            f"{fmt(row['post_chain']['mean'])} | "
            f"{fmt_ci(row['posterior_minus_chain'])} | "
            f"{fmt_ci(row['rewind_minus_posterior'])} | "
            f"{fmt(row['sample_accuracy']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Block Laplace Probe",
            "",
            "These rows are full-covariance Laplace/softmax-GGN probes for",
            "selected weight tensors, tensor groups, independent tensor-block",
            "diagonal subsets, or independent joint-group subsets while the",
            "rest of the network is frozen. They are stronger than diagonal or",
            "Kronecker-factor covariance for those selected parameters, but not",
            "a full-network full-covariance posterior.",
            "",
            "| Sampler | Block | Scale | Params | Block Post-Chain | "
            "Block Posterior-Chain [95% CI] | Block Rewind-Posterior [95% CI] | "
            "Global Posterior-Chain [95% CI] | Sample Acc. |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["block_laplace"]:
        lines.append(
            f"| {row['sampler']} | {row['block']} | {row['scale']:.1e} | "
            f"{fmt(row['parameter_count']['mean'])} | "
            f"{fmt(row['block_post_chain']['mean'])} | "
            f"{fmt_ci(row['block_posterior_minus_chain'])} | "
            f"{fmt_ci(row['block_rewind_minus_posterior'])} | "
            f"{fmt_ci(row['global_posterior_minus_chain'])} | "
            f"{fmt(row['sample_accuracy']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Subspace HMC Probe",
            "",
            "These rows run full-network HMC in a low-dimensional random",
            "or structured orthonormal subspace around the dense CIFAR",
            "checkpoint. They are higher fidelity than Gaussian mask",
            "perturbations, but not an exact full-network full-covariance",
            "posterior.",
            "",
            "| Sampler | Step | Post-Chain | Posterior-Chain [95% CI] | "
            "Rewind-Posterior [95% CI] | Sample Acc. | Accept | Param Dist. |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["subspace_hmc"]:
        lines.append(
            f"| {row['sampler']} | {row['step']:.1e} | "
            f"{fmt(row['post_chain']['mean'])} | "
            f"{fmt_ci(row['posterior_minus_chain'])} | "
            f"{fmt_ci(row['rewind_minus_posterior'])} | "
            f"{fmt(row['sample_accuracy']['mean'])} | "
            f"{fmt(row['accept_rate']['mean'])} | "
            f"{fmt(row['parameter_distance']['mean'])} |"
        )
    mode_rows = selected_mode_equivalence_rows(
        stats.get("mode_distribution_equivalence", [])
    )
    lines.extend(
        [
            "",
            "## Mode/Ticket Distribution Equivalence Audit",
            "",
            "This table turns the proposal's distributional equivalence criterion",
            "into a matched support-overlap audit. A posterior-mode rescue would",
            "require positive Posterior-Chain deltas; practical ties indicate",
            "that posterior supports behave like local chain-start magnitude",
            "supports.",
            "",
            "| Row | Scope | n | Posterior | Chain | Posterior-Chain | Wins | KS p | Wasserstein | Post-chain | Verdict |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in mode_rows:
        lines.append(
            f"| {mode_equivalence_label(row)} | {row['scope']} | "
            f"{int(row['n_pairs'])} | {fmt(row['posterior_mean'])} | "
            f"{fmt(row['baseline_mean'])} | {fmt(row['delta_mean'])} | "
            f"{fmt(row['paired_win_rate'])} | {fmt(row['ks_pvalue'])} | "
            f"{fmt(row['wasserstein'])} | {fmt(row['mean_post_chain'])} | "
            f"{row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Direct Proposal-Level Mode/Ticket Distribution Probe",
            "",
            "These rows evaluate the literal proposal metrics: layer-sparsity",
            "KS, RBF MMD, sliced Wasserstein distance, pairwise mask-Hamming",
            "distribution overlap, logit-space CKA, final-hidden-activation CKA,",
            "Hungarian matching cost, and raw parameter-PCA basin entropy.",
            "Posterior modes are mean-shift representatives in PCA-reduced",
            "parameter space.",
            "",
            "| Setting | Comparison | Modes | H norm | Eff. modes | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | "
            "Logit CKA | Act. CKA | Hungarian cost | Act. H-cost | Passes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats.get("direct_mode_ticket_distribution", []):
        lines.append(
            f"| {row.get('setting', '')} | "
            f"{direct_mode_ticket_comparison_label(row['comparison'])} | "
            f"{int(row.get('posterior_num_clusters', 0))} | "
            f"{fmt(row.get('posterior_cluster_entropy_normalized', math.nan))} | "
            f"{fmt(row.get('posterior_effective_cluster_count', math.nan))} | "
            f"{int(row['left_count'])} | "
            f"{int(row['right_count'])} | {fmt(row['layer_ks_pvalue'])} | "
            f"{fmt(row['layer_mmd_rbf'])} | "
            f"{fmt(row['layer_sliced_wasserstein'])} | "
            f"{fmt(row['hamming_overlap'])} | "
            f"{fmt(row['logit_cka_hungarian_mean'])} | "
            f"{fmt(row.get('activation_cka_hungarian_mean', math.nan))} | "
            f"{fmt(row['hungarian_cost'])} | "
            f"{fmt(row.get('activation_hungarian_cost', math.nan))} | "
            f"{int(row['threshold_pass_count'])}/{int(row.get('threshold_pass_total', 4))} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Calibration/OOD Probe",
            "",
            "These rows evaluate CIFAR-10 calibration and CIFAR-100 OOD",
            "detection for the dense model, the IMP ticket, SWAG predictives,",
            "and learned-mask controls. Lower NLL/ECE/Brier/FPR95 and higher",
            "AUROC are better. Dense/IMP/SWAG rows use the canonical SWAG run;",
            "learned-mask rows use the matched learned-mask calibration run.",
            "",
            "| Source | Acc. | NLL | ECE | Brier | MSP AUROC | MSP FPR95 | Ent. AUROC | Ent. FPR95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["calibration_ood"]:
        lines.append(
            f"| {calibration_source_label(row['source'])} | "
            f"{fmt(row['id_accuracy']['mean'])} | "
            f"{fmt(row['id_nll']['mean'])} | "
            f"{fmt(row['id_ece']['mean'])} | "
            f"{fmt(row['id_brier']['mean'])} | "
            f"{fmt(row['ood_msp_auroc']['mean'])} | "
            f"{fmt(row['ood_msp_fpr95']['mean'])} | "
            f"{fmt(row['ood_entropy_auroc']['mean'])} | "
            f"{fmt(row['ood_entropy_fpr95']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Dense Trajectory Probe",
            "",
            "| Epoch | Checkpoint Acc. | Trajectory-IMP [95% CI] | "
            "Dense-IMP | Rewind-IMP | IMP Acc. |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["trajectory"]:
        lines.append(
            f"| {row['epoch']} | {fmt(row['checkpoint_accuracy']['mean'])} | "
            f"{fmt_ci(row['trajectory_magnitude_to_imp'])} | "
            f"{fmt(row['dense_magnitude_to_imp']['mean'])} | "
            f"{fmt(row['rewind_magnitude_to_imp']['mean'])} | "
            f"{fmt(row['imp_accuracy']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Aggregate Trajectory Score Masks",
            "",
            "| Source | Score-IMP [95% CI] | Score-Best Ckpt | Score-Dense | "
            "Score-Rewind | Score-BestCkpt IMP Gap [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["trajectory_aggregate"]:
        source = source_label(row["source"])
        lines.append(
            f"| {source} | {fmt_ci(row['score_to_imp'])} | "
            f"{fmt(row['score_to_best_checkpoint']['mean'])} | "
            f"{fmt(row['score_to_dense']['mean'])} | "
            f"{fmt(row['score_to_rewind']['mean'])} | "
            f"{fmt_ci(row['score_minus_best_checkpoint'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Trajectory Mask Training Probe",
            "",
            "| Source | Kind | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Acc-Dense [95% CI] | Support-IMP |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["trajectory_mask_training"]:
        lines.append(
            f"| {source_label(row['source'])} | {row['source_kind']} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt_ci(row['accuracy_minus_dense'])} | "
            f"{fmt(row['source_to_imp']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Digits Variational Pruning Probe",
            "",
            "This is the first direct implementation of the proposal's",
            "Bernoulli-mask variational pruning baseline. It is a small-model",
            "sanity check, not CIFAR submission evidence.",
            "",
            "| Source | Kind | Accuracy [95% CI] | ECE | Brier | "
            "Acc-IMP [95% CI] | Acc-Dense [95% CI] | Support-IMP | "
            "Support-Dense |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["variational_pruning"]:
        lines.append(
            f"| {source_label(row['source'])} | {source_label(row['source_kind'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt(row['trained_ece']['mean'])} | "
            f"{fmt(row['trained_brier']['mean'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt_ci(row['accuracy_minus_dense'])} | "
            f"{fmt(row['source_to_imp']['mean'])} | "
            f"{fmt(row['source_to_dense']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Trajectory Residual Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | Support-Base |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["trajectory_residual"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt(row['mask_to_base']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Removal Control Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | Support-Base |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["trajectory_residual_removal_controls"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt(row['mask_to_base']['mean'])} |"
        )
    predictor_by_source = {
        row["base_source"]: row for row in stats["residual_anatomy_predictor"]
    }
    lines.extend(
        [
            "",
            "## CIFAR Residual Anatomy Probe",
            "",
            "| Base | Jaccard-IMP [95% CI] | IMP-only | Base-only Prune Round | "
            "Pred. AUC | Pred. Top-k Recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_anatomy_global"]:
        pred = predictor_by_source[row["base_source"]]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_ci(row['jaccard'])} | "
            f"{fmt_k(row['imp_only']['mean'])} | "
            f"{fmt_ci(row['base_only_pruned_round_mean'])} | "
            f"{fmt_ci(pred['test_auc'])} | "
            f"{fmt_ci(pred['test_topk_recall'])} |"
        )
    lines.extend(
        [
            "",
            "| Base | Group | IMP-only Share | IMP-only Enrichment | Base-only Share |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_anatomy_group"]:
        if row["unit"] not in {"stem", "stage1", "stage2", "stage3", "head"}:
            continue
        lines.append(
            f"| {source_label(row['base_source'])} | {row['unit']} | "
            f"{fmt_ci(row['imp_only_share'])} | "
            f"{fmt_ci(row['imp_only_enrichment'])} | "
            f"{fmt_ci(row['base_only_share'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Predictor Mask Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | Pred. AUC | Top-k Recall | Added Precision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_predictor_mask"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['predictor_auc'])} | "
            f"{fmt_ci(row['predictor_topk_recall'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Cross-Seed Transfer Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | Pred. AUC | Top-k Recall | Added Precision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_cross_seed_transfer"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['predictor_auc'])} | "
            f"{fmt_ci(row['predictor_topk_recall'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Direct Cross-Seed Transfer Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | Added Precision | Vote+ Frac. | Vote Mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_direct_transfer"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} | "
            f"{fmt_ci(row['selected_source_vote_positive_fraction'])} | "
            f"{fmt_ci(row['selected_source_vote_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Base Compatibility Probe",
            "",
            "| Base | Base Kind | Variant | Alpha | Accuracy [95% CI] | "
            "Acc-IMP [95% CI] | Base-IMP | Support-IMP | Added Precision | "
            "Oracle Overlap | Base-Ref |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_base_compatibility"]:
        base_kind = row["base_kind"].replace("_", " ")
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{base_kind} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['base_to_imp']['mean'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} | "
            f"{fmt_ci(row['added_oracle_overlap_precision'])} | "
            f"{fmt(row['base_to_reference']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Base Ordering Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | "
            "Acc-IMP [95% CI] | Support-IMP | Added Precision | "
            "Oracle Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_base_ordering"]:
        if row["base_kind"] != "imp_overlap_random":
            continue
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} | "
            f"{fmt_ci(row['added_oracle_overlap_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual Stratified Control Probe",
            "",
            "| Base | Variant | Alpha | Accuracy [95% CI] | Acc-IMP [95% CI] | "
            "Support-IMP | IMP Precision | Oracle Overlap | Exact Strata |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_stratified_controls"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_imp_only_precision'])} | "
            f"{fmt_ci(row['added_oracle_overlap_precision'])} | "
            f"{fmt_ci(row['stratum_exact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Probe",
            "",
            "| Base | Variant | Round | Alpha | Accuracy [95% CI] | "
            "Acc-IMP [95% CI] | Support-IMP | Final-IMP Precision | "
            "Oracle Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | "
            f"{fmt_round(row['process_round'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_final_imp_precision'])} | "
            f"{fmt_ci(row['added_oracle_overlap_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Control Probe",
            "",
            "| Base | Variant | Round | Alpha | Accuracy [95% CI] | "
            "Acc-IMP [95% CI] | Support-IMP | Final-IMP Precision | "
            "Oracle Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_controls"]:
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{variant_label(row['variant'])} | "
            f"{fmt_round(row['process_round'])} | {fmt_alpha(row['alpha'])} | "
            f"{fmt_ci(row['trained_accuracy'])} | "
            f"{fmt_ci(row['accuracy_minus_imp'])} | "
            f"{fmt(row['mask_to_imp']['mean'])} | "
            f"{fmt_ci(row['added_final_imp_precision'])} | "
            f"{fmt_ci(row['added_oracle_overlap_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Oracle-Matched Control",
            "",
            "| Base | Round | Score Acc. [95% CI] | Matched Acc. [95% CI] | "
            "Score-Matched Delta [95% CI] | Wins | Oracle Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_oracle_matched_pairs"]:
        delta = row["accuracy_delta"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['score_accuracy'])} | "
            f"{fmt_ci(row['matched_accuracy'])} | "
            f"{fmt_ci(delta)} | "
            f"{delta['positive']}/{delta['n']} | "
            f"{fmt_ci(row['oracle_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Score-Source Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Dense-Score Acc. [95% CI] | "
            "Base-Score Acc. [95% CI] | Round-Dense Delta [95% CI] | "
            "Wins | Round-Base Delta [95% CI] | Wins | Oracle Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_score_source_pairs"]:
        dense_delta = row["round_minus_dense_score"]
        base_delta = row["round_minus_base_score"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['dense_score_accuracy'])} | "
            f"{fmt_ci(row['base_score_accuracy'])} | "
            f"{fmt_ci(dense_delta)} | "
            f"{dense_delta['positive']}/{dense_delta['n']} | "
            f"{fmt_ci(base_delta)} | "
            f"{base_delta['positive']}/{base_delta['n']} | "
            f"{fmt_ci(row['oracle_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Round-Exclusion Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Excluded Acc. [95% CI] | "
            "Round-Excluded Delta [95% CI] | Wins | Round Oracle Overlap | "
            "Excluded Oracle Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_round_exclusion_pairs"]:
        delta = row["accuracy_delta"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['excluded_accuracy'])} | "
            f"{fmt_ci(delta)} | "
            f"{delta['positive']}/{delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['excluded_oracle_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Tensor-Matched Round-Exclusion Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Excluded Acc. [95% CI] | "
            "Tensor-Matched Excl. Acc. [95% CI] | Round-Excl. Delta [95% CI] | "
            "Wins | Round-Tensor Delta [95% CI] | Wins | Round Oracle | "
            "Tensor-Matched Oracle |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_layer_exclusion_pairs"]:
        excluded_delta = row["round_minus_excluded"]
        layer_delta = row["round_minus_layer_excluded"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['excluded_accuracy'])} | "
            f"{fmt_ci(row['layer_excluded_accuracy'])} | "
            f"{fmt_ci(excluded_delta)} | "
            f"{excluded_delta['positive']}/{excluded_delta['n']} | "
            f"{fmt_ci(layer_delta)} | "
            f"{layer_delta['positive']}/{layer_delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['layer_excluded_oracle_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Tensor+Score-Matched Round-Exclusion Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Tensor Excl. Acc. [95% CI] | "
            "Tensor+Score Excl. Acc. [95% CI] | Round-Tensor Delta [95% CI] | "
            "Wins | Round-Tensor+Score Delta [95% CI] | Wins | Round Oracle | "
            "Tensor+Score Oracle |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_tensor_score_exclusion_pairs"]:
        layer_delta = row["round_minus_layer_excluded"]
        tensor_score_delta = row["round_minus_tensor_score_excluded"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['layer_excluded_accuracy'])} | "
            f"{fmt_ci(row['tensor_score_excluded_accuracy'])} | "
            f"{fmt_ci(layer_delta)} | "
            f"{layer_delta['positive']}/{layer_delta['n']} | "
            f"{fmt_ci(tensor_score_delta)} | "
            f"{tensor_score_delta['positive']}/{tensor_score_delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['tensor_score_excluded_oracle_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Projection Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Residualized Acc. [95% CI] | "
            "Dense-Score Acc. [95% CI] | Base-Score Acc. [95% CI] | "
            "Round-Residualized Delta [95% CI] | Wins | Round Oracle | "
            "Residualized Oracle | Oracle Delta [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_projection_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['residualized_accuracy'])} | "
            f"{fmt_ci(row['dense_score_accuracy'])} | "
            f"{fmt_ci(row['base_score_accuracy'])} | "
            f"{fmt_ci(delta)} | "
            f"{delta['positive']}/{delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['residualized_oracle_overlap'])} | "
            f"{fmt_ci(oracle_delta)} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Posterior-Projection Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Posterior-Residualized Acc. [95% CI] | "
            "Dense-Score Acc. [95% CI] | Base-Score Acc. [95% CI] | "
            "Round-Posterior-Residualized Delta [95% CI] | Wins | Round Oracle | "
            "Posterior-Residualized Oracle | Oracle Delta [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_posterior_projection_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['residualized_accuracy'])} | "
            f"{fmt_ci(row['dense_score_accuracy'])} | "
            f"{fmt_ci(row['base_score_accuracy'])} | "
            f"{fmt_ci(delta)} | "
            f"{delta['positive']}/{delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['residualized_oracle_overlap'])} | "
            f"{fmt_ci(oracle_delta)} |"
        )
    lines.extend(
        [
            "",
            "## CIFAR Residual IMP Process Learned-Subspace Control",
            "",
            "| Base | Round | Round Acc. [95% CI] | Learned-Subspace Acc. [95% CI] | "
            "Dense-Score Acc. [95% CI] | Base-Score Acc. [95% CI] | "
            "Round-Learned-Subspace Delta [95% CI] | Wins | Round Oracle | "
            "Learned-Subspace Oracle | Oracle Delta [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stats["residual_imp_process_learned_subspace_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"| {source_label(row['base_source'])} | "
            f"{fmt_round(row['process_round'])} | "
            f"{fmt_ci(row['round_accuracy'])} | "
            f"{fmt_ci(row['residualized_accuracy'])} | "
            f"{fmt_ci(row['dense_score_accuracy'])} | "
            f"{fmt_ci(row['base_score_accuracy'])} | "
            f"{fmt_ci(delta)} | "
            f"{delta['positive']}/{delta['n']} | "
            f"{fmt_ci(row['round_oracle_overlap'])} | "
            f"{fmt_ci(row['residualized_oracle_overlap'])} | "
            f"{fmt_ci(oracle_delta)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "Gate1 would require posterior-chain gaps to become positive under",
            "movement while rewind/dense controls stop dominating. The current",
            "movement rows instead show that lower Post-Chain generally coincides",
            "with non-positive Posterior-Chain gaps and positive Rewind-Posterior",
            "gaps.",
            "",
            "This file is generated by `scripts/build_paper_stats.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(stats: dict[str, Any], path: Path) -> None:
    selected = []
    keep = {
        ("SGLD", 1e-6),
        ("SGHMC", 1e-7),
        ("cSGLD", 1e-6),
        ("SWAG20", 16.0),
        ("SWAG20", 64.0),
        ("DiagLap", 1e-3),
        ("KFACLap", 1e-3),
        ("KFACLap", 1e-2),
        ("LowRankLap", 1e-3),
        ("LowRankLap", 1e-2),
        ("LowRank32Lap", 1e-3),
        ("LowRank32Lap", 1e-2),
    }
    for row in stats["movement"]:
        if (row["sampler"], row["scale"]) in keep:
            selected.append(row)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Sampler & Scale & Post-chain & Post.-Chain & Rewind-Post. & Sample Acc. \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['sampler']} & ${tex_scale(row['scale'])}$ & "
            f"{fmt(row['post_chain']['mean'])} & "
            f"{fmt(row['posterior_minus_chain']['mean'])} & "
            f"{fmt(row['rewind_minus_posterior']['mean'])} & "
            f"{fmt(row['sample_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Paired five-seed CIFAR movement statistics. "
            "Post.-Chain is posterior-to-IMP minus chain-start-to-IMP; "
            "Rewind-Post. is rewind-magnitude-to-IMP minus posterior-to-IMP.}",
            "\\label{tab:cifar-movement-stats}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Sampler & Scale & Head post-chain & Head post.-chain & "
            "Head rewind-post. & Sample Acc. \\\\",
            "\\midrule",
        ]
    )
    head_keep = {1e-3, 1e-2, 1.0}
    for row in stats["head_laplace"]:
        if row["scale"] not in head_keep:
            continue
        lines.append(
            f"{row['sampler']} & ${tex_scale(row['scale'])}$ & "
            f"{fmt(row['post_chain']['mean'])} & "
            f"{fmt(row['posterior_minus_chain']['mean'])} & "
            f"{fmt(row['rewind_minus_posterior']['mean'])} & "
            f"{fmt(row['sample_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Exact full-covariance Laplace probe for the final CIFAR",
            "classification head only. Head post.-chain is head posterior-to-IMP",
            "minus head chain-start-to-IMP; head rewind-post. is head",
            "rewind-magnitude-to-IMP minus head posterior-to-IMP.}",
            "\\label{tab:cifar-head-laplace-stats}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "Sampler & Block & Params & B post-chain & B post.-chain & "
            "B rewind-post. & G post.-chain & Acc. \\\\",
            "\\midrule",
        ]
    )
    for row in stats["block_laplace"]:
        raw_block_label = str(row["block"])
        if raw_block_label.startswith("joint:"):
            block_label = "joint group"
        elif raw_block_label.startswith("jointdiag:"):
            if "40000" in raw_block_label:
                block_label = "joint-group diag 40k"
            elif "20000" in raw_block_label:
                block_label = "joint-group diag 20k"
            else:
                block_label = "joint-group diag 10k"
        elif raw_block_label.startswith("blockdiag:"):
            block_label = (
                "16-block diag"
                if "16blocks" in raw_block_label
                else "11-block diag"
            )
        else:
            block_label = raw_block_label.removesuffix(".weight").replace("_", r"\_")
        lines.append(
            f"{row['sampler']} & {block_label} & "
            f"{row['parameter_count']['mean']:.0f} & "
            f"{fmt(row['block_post_chain']['mean'])} & "
            f"{fmt(row['block_posterior_minus_chain']['mean'])} & "
            f"{fmt(row['block_rewind_minus_posterior']['mean'])} & "
            f"{fmt(row['global_posterior_minus_chain']['mean'])} & "
            f"{fmt(row['sample_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Full-covariance block Laplace probes for selected CIFAR",
            "ResNet-20 weight tensors, tensor groups, or a tensor-block",
            "diagonal subset with the rest of the network frozen.",
            "Block post.-chain is block posterior-to-IMP minus block",
            "chain-start-to-IMP; global post.-chain reports the induced global",
            "support change.}",
            "\\label{tab:cifar-block-laplace-stats}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "Sampler & Step & PostCh. & Post.-Chain & Rwd.-Post. & "
            "Acc. & Accept & ParamDist. \\\\",
            "\\midrule",
        ]
    )
    for row in stats["subspace_hmc"]:
        lines.append(
            f"{row['sampler']} & ${tex_scale(row['step'])}$ & "
            f"{fmt(row['post_chain']['mean'])} & "
            f"{fmt(row['posterior_minus_chain']['mean'])} & "
            f"{fmt(row['rewind_minus_posterior']['mean'])} & "
            f"{fmt(row['sample_accuracy']['mean'])} & "
            f"{fmt(row['accept_rate']['mean'])} & "
            f"{fmt(row['parameter_distance']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Full-network low-dimensional subspace HMC probes around",
            "the dense CIFAR checkpoint. RandSubHMC and TrajSubHMC use",
            "random and dense-trajectory subspaces; Hessian rows use randomized",
            "top-Hessian subspaces.}",
            "\\label{tab:cifar-subspace-hmc-stats}",
            "\\end{table}",
            "",
        ]
    )
    mode_rows = selected_mode_equivalence_rows(
        stats.get("mode_distribution_equivalence", [])
    )
    lines.extend(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Row & Post. & Chain & Delta & Wins & Post-chain & W-dist. \\\\",
            "\\midrule",
        ]
    )
    for row in mode_rows:
        lines.append(
            f"{mode_equivalence_label(row)} & "
            f"{fmt(row['posterior_mean'])} & "
            f"{fmt(row['baseline_mean'])} & "
            f"{fmt(row['delta_mean'])} & "
            f"{fmt(row['paired_win_rate'])} & "
            f"{fmt(row['mean_post_chain'])} & "
            f"{fmt(row['wasserstein'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Mode/ticket distribution-equivalence audit for selected",
            "posterior families. Delta is posterior-to-IMP minus matched",
            "chain-start-to-IMP. Wins is the fraction of seed-level paired",
            "posterior-chain deltas above zero; W-dist. is the two-sample",
            "Wasserstein distance over grouped seed means. A positive posterior",
            "mode account would require positive deltas rather than practical",
            "ties to chain-start magnitude supports.}",
            "\\label{tab:mode-ticket-equivalence-audit}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{1.5pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{llrrrrrrrrrrr}",
            "\\toprule",
            "Setting & Comparison & Modes & $H_n$ & $n_L$ & $n_T$ & KS p & Ham. & Logit & Act. & H-cost & A-cost & Pass \\\\",
            "\\midrule",
        ]
    )
    for row in stats.get("direct_mode_ticket_distribution", []):
        lines.append(
            f"{row.get('setting', '')} & "
            f"{direct_mode_ticket_comparison_label(row['comparison'])} & "
            f"{int(row.get('posterior_num_clusters', 0))} & "
            f"{fmt(row.get('posterior_cluster_entropy_normalized', math.nan))} & "
            f"{int(row['left_count'])} & "
            f"{int(row['right_count'])} & "
            f"{fmt(row['layer_ks_pvalue'])} & "
            f"{fmt(row['hamming_overlap'])} & "
            f"{fmt(row['logit_cka_hungarian_mean'])} & "
            f"{fmt(row.get('activation_cka_hungarian_mean', math.nan))} & "
            f"{fmt(row['hungarian_cost'])} & "
            f"{fmt(row.get('activation_hungarian_cost', math.nan))} & "
            f"{int(row['threshold_pass_count'])}/{int(row.get('threshold_pass_total', 4))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Direct proposal-level mode/ticket distribution probe on",
            "digits MLP plus CIFAR-10 ResNet-20 subset and full-data settings. Posterior sample masks",
            "and mean-shift posterior-mode representatives are compared with",
            "IMP tickets using the proposal's layer-sparsity KS, mask-Hamming",
            "overlap, logit-space CKA, final-hidden-activation CKA, and",
            "Hungarian-cost thresholds; Modes and $H_n$ report mean-shift",
            "cluster count and normalized raw-parameter basin entropy.}",
            "\\label{tab:direct-mode-ticket-distribution}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Source & Acc. & NLL & ECE & MSP AUROC & Ent. AUROC & MSP FPR95 \\\\",
            "\\midrule",
        ]
    )
    calibration_keep = {
        "dense",
        "imp",
        "swag_ensemble",
        "learned_random_0",
        "gem_miner",
        "variational_prune",
        "hard_concrete",
    }
    for row in stats["calibration_ood"]:
        if row["source"] not in calibration_keep:
            continue
        lines.append(
            f"{calibration_source_label(row['source'])} & "
            f"{fmt(row['id_accuracy']['mean'])} & "
            f"{fmt(row['id_nll']['mean'])} & "
            f"{fmt(row['id_ece']['mean'])} & "
            f"{fmt(row['ood_msp_auroc']['mean'])} & "
            f"{fmt(row['ood_entropy_auroc']['mean'])} & "
            f"{fmt(row['ood_msp_fpr95']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR-10 calibration and CIFAR-100 OOD",
            "diagnostics. SWAG ens. averages ten SWAG predictive samples per",
            "seed. Learned random, Gem-Miner, and Var. prune rows are trained",
            "from initialization at the IMP sparsity. MSP and entropy AUROC use",
            "maximum softmax probability and negative predictive entropy as",
            "in-distribution scores; lower FPR95 is better.}",
            "\\label{tab:cifar-calibration-ood-stats}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Source & Acc. & Support-IMP & Support-Dense & Support-Rewind \\\\",
            "\\midrule",
        ]
    )
    trajectory_keep = {1, 5, 10, 20, 30}
    for row in stats["trajectory"]:
        if row["epoch"] not in trajectory_keep:
            continue
        lines.append(
            f"Epoch {row['epoch']} & "
            f"{fmt(row['checkpoint_accuracy']['mean'])} & "
            f"{fmt(row['trajectory_magnitude_to_imp']['mean'])} & "
            f"{fmt(row['trajectory_to_dense_final_magnitude']['mean'])} & "
            f"{fmt(row['trajectory_to_rewind_magnitude']['mean'])} \\\\"
        )
    aggregate_keep = {"traj_rms_abs", "traj_mean_abs", "traj_max_abs"}
    for row in stats["trajectory_aggregate"]:
        if row["source"] not in aggregate_keep:
            continue
        source = source_label(row["source"])
        lines.append(
            f"{source} & -- & "
            f"{fmt(row['score_to_imp']['mean'])} & "
            f"{fmt(row['score_to_dense']['mean'])} & "
            f"{fmt(row['score_to_rewind']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR dense-trajectory controls. "
            "The epoch-1 state is the IMP rewind point. Checkpoint rows use "
            "magnitude supports at that epoch; aggregate rows use trajectory "
            "score masks across checkpoints.}",
            "\\label{tab:cifar-trajectory-stats}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Source & Kind & Accuracy & Acc.-IMP & Support-IMP \\\\",
            "\\midrule",
        ]
    )
    mask_training_by_source = {
        (row["source_kind"], row["source"]): row
        for row in stats["trajectory_mask_training"]
    }
    mask_training_keep = [
        ("imp", "imp"),
        ("random", "random_0"),
        ("checkpoint", "epoch_1"),
        ("checkpoint", "epoch_10"),
        ("checkpoint", "epoch_30"),
        ("aggregate", "traj_rms_abs"),
        ("aggregate", "traj_mean_abs"),
        ("aggregate", "traj_path_length"),
        ("aggregate", "traj_rewind_rms_movement"),
        ("gem_miner", "gem_miner"),
        ("variational_prune", "variational_prune"),
        ("hard_concrete", "hard_concrete"),
    ]
    for key in mask_training_keep:
        row = mask_training_by_source.get(key)
        if row is None:
            continue
        lines.append(
            f"{source_label(row['source'])} & {source_label(row['source_kind'])} & "
            f"{fmt(row['trained_accuracy']['mean'])} & "
            f"{fmt(row['accuracy_minus_imp']['mean'])} & "
            f"{fmt(row['source_to_imp']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR trajectory-mask retraining controls. "
            "All masks are trained for 30 epochs from the same epoch-1 rewind "
            "state at the IMP sparsity. Acc.-IMP is candidate accuracy minus "
            "the final IMP accuracy from the corresponding seed.}",
            "\\label{tab:cifar-trajectory-mask-training}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "Source & Kind & Acc. & ECE & Brier & Acc.-IMP & Supp.-IMP & Supp.-Dense \\\\",
            "\\midrule",
        ]
    )
    variational_by_source = {
        (row["source_kind"], row["source"]): row
        for row in stats["variational_pruning"]
    }
    variational_keep = [
        ("imp", "imp"),
        ("random", "random_0"),
        ("gem_miner", "gem_miner"),
        ("variational_prune", "variational_prune"),
    ]
    for key in variational_keep:
        row = variational_by_source.get(key)
        if row is None:
            continue
        lines.append(
            f"{source_label(row['source'])} & {source_label(row['source_kind'])} & "
            f"{fmt(row['trained_accuracy']['mean'])} & "
            f"{fmt(row['trained_ece']['mean'])} & "
            f"{fmt(row['trained_brier']['mean'])} & "
            f"{fmt(row['accuracy_minus_imp']['mean'])} & "
            f"{fmt(row['source_to_imp']['mean'])} & "
            f"{fmt(row['source_to_dense']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed digits MLP variational pruning sanity check. "
            "Var. prune optimizes Bernoulli mask probabilities on frozen "
            "initialization weights with Concrete samples, KL, sparsity, and "
            "entropy penalties before retraining the selected hard mask.}",
            "\\label{tab:digits-variational-pruning}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Base & Variant & Alpha & Accuracy & Acc.-IMP \\\\",
            "\\midrule",
        ]
    )
    residual_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["trajectory_residual"]
    }
    residual_keep = [
        ("epoch_30", "imp_residual", 0.0),
        ("epoch_30", "imp_residual", 0.5),
        ("epoch_30", "random_residual", 0.5),
        ("epoch_30", "imp_residual", 1.0),
        ("traj_rms_abs", "imp_residual", 0.0),
        ("traj_rms_abs", "imp_residual", 0.5),
        ("traj_rms_abs", "random_residual", 0.5),
        ("epoch_10", "imp_residual", 0.0),
        ("epoch_10", "imp_residual", 0.5),
        ("epoch_10", "random_residual", 0.5),
    ]
    for key in residual_keep:
        row = residual_by_key.get(key)
        if row is None:
            continue
        lines.append(
            f"{source_label(row['base_source'])} & {variant_label(row['variant'])} & "
            f"{fmt_alpha(row['alpha'])} & "
            f"{fmt(row['trained_accuracy']['mean'])} & "
            f"{fmt(row['accuracy_minus_imp']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR trajectory residual controls. "
            "Starting from a base trajectory mask, an IMP-residual row swaps "
            "the indicated fraction of base-only support for IMP-only support; "
            "a random-residual row swaps the same number of positions for "
            "non-IMP random support.}",
            "\\label{tab:cifar-trajectory-residual}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Base & Base & Low rm & Rand. rm & High rm & Rand. non \\\\",
            "\\midrule",
        ]
    )
    removal_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["trajectory_residual_removal_controls"]
    }
    removal_keep = ["epoch_30", "traj_rms_abs", "epoch_10"]
    for source in removal_keep:
        base = removal_by_key.get((source, "imp_residual", 0.0))
        low = removal_by_key.get((source, "imp_residual", 0.5))
        random_rm = removal_by_key.get(
            (source, "imp_residual_random_remove", 0.5)
        )
        high_rm = removal_by_key.get((source, "imp_residual_high_remove", 0.5))
        random_non = removal_by_key.get((source, "random_residual", 0.5))
        if (
            base is None
            or low is None
            or random_rm is None
            or high_rm is None
            or random_non is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(low['trained_accuracy']['mean'])} & "
            f"{fmt(random_rm['trained_accuracy']['mean'])} & "
            f"{fmt(high_rm['trained_accuracy']['mean'])} & "
            f"{fmt(random_non['trained_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual removal-order controls. "
            "Low/Rand./High rm add the same top IMP-only residual support "
            "and vary which base-only weights are removed by low, random, "
            "or high base score. Rand. non removes low base-score weights "
            "but adds same-size non-IMP random support.}",
            "\\label{tab:cifar-residual-removal-controls}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Base & Support-IMP & IMP-only(k) & Prune rnd. & Pred. AUC & Pred. recall \\\\",
            "\\midrule",
        ]
    )
    predictor_by_source = {
        row["base_source"]: row for row in stats["residual_anatomy_predictor"]
    }
    anatomy_keep = ["epoch_30", "traj_rms_abs", "epoch_10"]
    anatomy_by_source = {
        row["base_source"]: row for row in stats["residual_anatomy_global"]
    }
    for source in anatomy_keep:
        row = anatomy_by_source.get(source)
        pred = predictor_by_source.get(source)
        if row is None or pred is None:
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(row['jaccard']['mean'])} & "
            f"{fmt_k(row['imp_only']['mean'])} & "
            f"{fmt(row['base_only_pruned_round_mean']['mean'])} & "
            f"{fmt(pred['test_auc']['mean'])} & "
            f"{fmt(pred['test_topk_recall']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual anatomy. IMP-only(k) is the "
            "number of IMP-kept weights missing from the base support. Prune "
            "rnd. is the mean IMP pruning round for base-only weights. The "
            "predictor is a held-out logistic model over dense-trajectory "
            "rank features plus stage indicators, trained to recover IMP-only "
            "residual support among non-base weights.}",
            "\\label{tab:cifar-residual-anatomy}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Base & Base & Oracle & Pred. & Random & Pred. prec. & Rand. prec. \\\\",
            "\\midrule",
        ]
    )
    predictor_mask_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["residual_predictor_mask"]
    }
    for source in anatomy_keep:
        base = predictor_mask_by_key.get((source, "base", 0.0))
        oracle = predictor_mask_by_key.get((source, "oracle_imp_residual", 0.5))
        pred = predictor_mask_by_key.get(
            (source, "heldout_predictor_residual", 0.5)
        )
        random = predictor_mask_by_key.get(
            (source, "heldout_random_residual", 0.5)
        )
        if base is None or oracle is None or pred is None or random is None:
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(pred['trained_accuracy']['mean'])} & "
            f"{fmt(random['trained_accuracy']['mean'])} & "
            f"{fmt(pred['added_imp_only_precision']['mean'])} & "
            f"{fmt(random['added_imp_only_precision']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR functional residual-predictor masks. "
            "For each base, the held-out predictor swaps half of base-only "
            "support for non-base weights with highest predicted IMP-only "
            "probability. Oracle uses true IMP-only residual support, and "
            "random uses same-size held-out non-IMP-random support. Prec. is "
            "the IMP-only precision of added weights.}",
            "\\label{tab:cifar-residual-predictor-mask}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Base & Base & Oracle & Cross-seed & Random & Cross prec. & Rand. prec. \\\\",
            "\\midrule",
        ]
    )
    cross_seed_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["residual_cross_seed_transfer"]
    }
    for source in anatomy_keep:
        base = cross_seed_by_key.get((source, "base", 0.0))
        oracle = cross_seed_by_key.get((source, "oracle_imp_residual", 0.5))
        pred = cross_seed_by_key.get(
            (source, "cross_seed_predictor_residual", 0.5)
        )
        random = cross_seed_by_key.get((source, "target_random_residual", 0.5))
        if base is None or oracle is None or pred is None or random is None:
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(pred['trained_accuracy']['mean'])} & "
            f"{fmt(random['trained_accuracy']['mean'])} & "
            f"{fmt(pred['added_imp_only_precision']['mean'])} & "
            f"{fmt(random['added_imp_only_precision']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR cross-seed residual-transfer masks. "
            "For each held-out target seed, the predictor is trained on "
            "non-base candidate weights from the other four seeds, then swaps "
            "half of target base-only support for target non-base weights with "
            "highest predicted IMP-only probability. Oracle and random use the "
            "target seed only.}",
            "\\label{tab:cifar-residual-cross-seed-transfer}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Base & Base & Oracle & Src vote & Aln src & Vote rand. & Aln rand. & Tgt rand. & Src prec. \\\\",
            "\\midrule",
        ]
    )
    direct_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["residual_direct_transfer"]
    }
    for source in anatomy_keep:
        base = direct_by_key.get((source, "base", 0.0))
        oracle = direct_by_key.get((source, "target_oracle_residual", 0.5))
        source_vote = direct_by_key.get((source, "source_vote_residual", 0.5))
        aligned_source_vote = direct_by_key.get(
            (source, "aligned_source_vote_residual", 0.5)
        )
        vote_random = direct_by_key.get(
            (source, "source_vote_random_residual", 0.5)
        )
        aligned_vote_random = direct_by_key.get(
            (source, "aligned_source_vote_random_residual", 0.5)
        )
        target_random = direct_by_key.get(
            (source, "target_random_residual", 0.5)
        )
        if (
            base is None
            or oracle is None
            or source_vote is None
            or aligned_source_vote is None
            or vote_random is None
            or aligned_vote_random is None
            or target_random is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(source_vote['trained_accuracy']['mean'])} & "
            f"{fmt(aligned_source_vote['trained_accuracy']['mean'])} & "
            f"{fmt(vote_random['trained_accuracy']['mean'])} & "
            f"{fmt(aligned_vote_random['trained_accuracy']['mean'])} & "
            f"{fmt(target_random['trained_accuracy']['mean'])} & "
            f"{fmt(source_vote['added_imp_only_precision']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR direct cross-seed residual-support "
            "transfer. Source-vote variants add target non-base coordinates "
            "selected by votes from the other seeds' oracle residual supports. "
            "Aln src and Aln rand. first map source channels to target "
            "channels by activation-correlation Hungarian matching. Tgt rand. "
            "is a target-seed random non-base residual control. Src prec. is "
            "target IMP-only precision among source-vote additions.}",
            "\\label{tab:cifar-residual-direct-transfer}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Base & T base & T oracle & T rand. & M base & M oracle & M rand. \\\\",
            "\\midrule",
        ]
    )
    compatibility_by_key = {
        (
            row["base_source"],
            row["base_kind"],
            row["variant"],
            float(row["alpha"]),
        ): row
        for row in stats["residual_base_compatibility"]
    }
    for source in anatomy_keep:
        traj_base = compatibility_by_key.get(
            (source, "trajectory", "base", 0.0)
        )
        traj_oracle = compatibility_by_key.get(
            (source, "trajectory", "oracle_imp_residual", 0.5)
        )
        traj_random = compatibility_by_key.get(
            (source, "trajectory", "random_residual", 0.5)
        )
        matched_base = compatibility_by_key.get(
            (source, "imp_overlap_random", "base", 0.0)
        )
        matched_oracle = compatibility_by_key.get(
            (source, "imp_overlap_random", "oracle_imp_residual", 0.5)
        )
        matched_random = compatibility_by_key.get(
            (source, "imp_overlap_random", "random_residual", 0.5)
        )
        if (
            traj_base is None
            or traj_oracle is None
            or traj_random is None
            or matched_base is None
            or matched_oracle is None
            or matched_random is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(traj_base['trained_accuracy']['mean'])} & "
            f"{fmt(traj_oracle['trained_accuracy']['mean'])} & "
            f"{fmt(traj_random['trained_accuracy']['mean'])} & "
            f"{fmt(matched_base['trained_accuracy']['mean'])} & "
            f"{fmt(matched_oracle['trained_accuracy']['mean'])} & "
            f"{fmt(matched_random['trained_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual base-compatibility controls. "
            "T rows use the actual trajectory base. M rows replace that base "
            "with a per-parameter random base preserving the same IMP/non-IMP "
            "counts and therefore the same base-to-IMP overlap before adding "
            "oracle or random residual support.}",
            "\\label{tab:cifar-residual-base-compatibility}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{1.8pt}",
            "\\begin{tabular}{lrrrrrrrrrr}",
            "\\toprule",
            "Base & M base & Top & Dense & RMS & RMS-d & Std & Rand & Dense ovlp & RMS ovlp & Rand ovlp \\\\",
            "\\midrule",
        ]
    )
    ordering_by_key = {
        (
            row["base_source"],
            row["base_kind"],
            row["variant"],
            float(row["alpha"]),
        ): row
        for row in stats["residual_base_ordering"]
    }
    for source in anatomy_keep:
        base = ordering_by_key.get(
            (source, "imp_overlap_random", "base", 0.0)
        )
        oracle = ordering_by_key.get(
            (source, "imp_overlap_random", "oracle_imp_residual", 0.5)
        )
        posterior_imp = ordering_by_key.get(
            (source, "imp_overlap_random", "posterior_imp_only_residual", 0.5)
        )
        dense_imp = ordering_by_key.get(
            (source, "imp_overlap_random", "dense_imp_only_residual", 0.5)
        )
        posterior_excess_imp = ordering_by_key.get(
            (
                source,
                "imp_overlap_random",
                "posterior_excess_imp_only_residual",
                0.5,
            )
        )
        posterior_std_imp = ordering_by_key.get(
            (
                source,
                "imp_overlap_random",
                "posterior_std_imp_only_residual",
                0.5,
            )
        )
        random_imp = ordering_by_key.get(
            (source, "imp_overlap_random", "random_imp_only_residual", 0.5)
        )
        if (
            base is None
            or oracle is None
            or posterior_imp is None
            or dense_imp is None
            or posterior_excess_imp is None
            or posterior_std_imp is None
            or random_imp is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(dense_imp['trained_accuracy']['mean'])} & "
            f"{fmt(posterior_imp['trained_accuracy']['mean'])} & "
            f"{fmt(posterior_excess_imp['trained_accuracy']['mean'])} & "
            f"{fmt(posterior_std_imp['trained_accuracy']['mean'])} & "
            f"{fmt(random_imp['trained_accuracy']['mean'])} & "
            f"{fmt(dense_imp['added_oracle_overlap_precision']['mean'])} & "
            f"{fmt(posterior_imp['added_oracle_overlap_precision']['mean'])} & "
            f"{fmt(random_imp['added_oracle_overlap_precision']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual posterior-decomposition "
            "controls on IMP-overlap-matched random bases. Top adds the "
            "target oracle IMP-only residual. Dense, RMS, RMS-d, Std, and "
            "Rand add the same number of target IMP-only residual weights "
            "ranked by dense final magnitude, diagonal-Laplace posterior "
            "RMS, posterior RMS minus dense magnitude, posterior standard "
            "deviation, or uniform sampling. Ovlp columns report the "
            "fraction of additions that also belong to the top oracle "
            "additions.}",
            "\\label{tab:cifar-residual-base-ordering}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Base & Base & Oracle & Rand. IMP & Global non & Param+score non \\\\",
            "\\midrule",
        ]
    )
    stratified_by_key = {
        (row["base_source"], row["variant"], float(row["alpha"])): row
        for row in stats["residual_stratified_controls"]
    }
    for source in anatomy_keep:
        base = stratified_by_key.get((source, "base", 0.0))
        oracle = stratified_by_key.get(
            (source, "oracle_top_imp_residual", 0.5)
        )
        random_imp = stratified_by_key.get(
            (source, "random_imp_only_residual", 0.5)
        )
        global_non = stratified_by_key.get(
            (source, "random_nonimp_global_residual", 0.5)
        )
        param_score_non = stratified_by_key.get(
            (
                source,
                "random_nonimp_parameter_score_matched_residual",
                0.5,
            )
        )
        if (
            base is None
            or oracle is None
            or random_imp is None
            or global_non is None
            or param_score_non is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(random_imp['trained_accuracy']['mean'])} & "
            f"{fmt(global_non['trained_accuracy']['mean'])} & "
            f"{fmt(param_score_non['trained_accuracy']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual stratified controls. "
            "All non-base variants remove the same low-base-score weights "
            "as the oracle. Rand. IMP adds random IMP-only residual weights. "
            "Global non adds random non-IMP non-base weights. Param+score "
            "non matches the oracle added weights by parameter tensor and "
            "within-parameter score decile, but excludes IMP-only weights.}",
            "\\label{tab:cifar-residual-stratified-controls}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "Base & Base & Oracle & R1 acc. & R3 acc. & R5 acc. & R1 prec. & R3 prec. \\\\",
            "\\midrule",
        ]
    )
    imp_process_by_key = {
        (
            row["base_source"],
            row["variant"],
            row["process_round"],
            float(row["alpha"]),
        ): row
        for row in stats["residual_imp_process"]
    }
    for source in anatomy_keep:
        base = imp_process_by_key.get((source, "base", None, 0.0))
        oracle = imp_process_by_key.get(
            (source, "final_oracle_residual", None, 0.5)
        )
        round1 = imp_process_by_key.get(
            (source, "round_survivor_residual", 1, 0.5)
        )
        round3 = imp_process_by_key.get(
            (source, "round_survivor_residual", 3, 0.5)
        )
        round5 = imp_process_by_key.get(
            (source, "round_survivor_residual", 5, 0.5)
        )
        if (
            base is None
            or oracle is None
            or round1 is None
            or round3 is None
            or round5 is None
        ):
            continue
        lines.append(
            f"{source_label(source)} & "
            f"{fmt(base['trained_accuracy']['mean'])} & "
            f"{fmt(oracle['trained_accuracy']['mean'])} & "
            f"{fmt(round1['trained_accuracy']['mean'])} & "
            f"{fmt(round3['trained_accuracy']['mean'])} & "
            f"{fmt(round5['trained_accuracy']['mean'])} & "
            f"{fmt(round1['added_final_imp_precision']['mean'])} & "
            f"{fmt(round3['added_final_imp_precision']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual IMP-process probe. R1/R3/R5 "
            "rows add half residual support from IMP round-survivor candidates "
            "ranked by the corresponding round-trained weights. Prec. is the "
            "fraction of added weights that survive in the final IMP mask; R5 "
            "precision is one by construction.}",
            "\\label{tab:cifar-residual-imp-process}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Top acc. & Rand. acc. & Low acc. & "
            "Top final & Rand. final & Low final & "
            "Top oracle & Rand. oracle & Low oracle \\\\",
            "\\midrule",
        ]
    )
    imp_process_control_by_key = {
        (
            row["base_source"],
            row["variant"],
            row["process_round"],
            float(row["alpha"]),
        ): row
        for row in stats["residual_imp_process_controls"]
    }
    for source in anatomy_keep:
        for process_round in [1, 3, 5]:
            top = imp_process_control_by_key.get(
                (source, "round_survivor_residual", process_round, 0.5)
            )
            random = imp_process_control_by_key.get(
                (source, "round_survivor_random_residual", process_round, 0.5)
            )
            low = imp_process_control_by_key.get(
                (source, "round_survivor_low_residual", process_round, 0.5)
            )
            if top is None or random is None or low is None:
                continue
            lines.append(
                f"{source_label(source)} & {process_round} & "
                f"{fmt(top['trained_accuracy']['mean'])} & "
                f"{fmt(random['trained_accuracy']['mean'])} & "
                f"{fmt(low['trained_accuracy']['mean'])} & "
                f"{fmt(top['added_final_imp_precision']['mean'])} & "
                f"{fmt(random['added_final_imp_precision']['mean'])} & "
                f"{fmt(low['added_final_imp_precision']['mean'])} & "
                f"{fmt(top['added_oracle_overlap_precision']['mean'])} & "
                f"{fmt(random['added_oracle_overlap_precision']['mean'])} & "
                f"{fmt(low['added_oracle_overlap_precision']['mean'])} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process ranking controls. "
            "Top uses the highest round-trained scores among round-survivor "
            "candidates; Rand. samples the same round-survivor candidate set; "
            "Low uses the lowest round-trained scores. Final is the fraction "
            "of added weights that survive in the final IMP mask; oracle is "
            "overlap with the final oracle added subset.}",
            "\\label{tab:cifar-residual-imp-process-controls}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Base & Rnd & Score acc. & Match acc. & $\\Delta$ & Wins & Oracle \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_oracle_matched_pairs"]:
        delta = row["accuracy_delta"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['score_accuracy']['mean'])} & "
            f"{fmt(row['matched_accuracy']['mean'])} & "
            f"{fmt(delta['mean'])} & "
            f"{delta['positive']}/{delta['n']} & "
            f"{fmt(row['oracle_overlap']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual IMP-process oracle-overlap-"
            "matched control. Score acc. uses the round-trained score ordering "
            "inside the final-IMP residual candidate set. Match acc. samples "
            "random final-IMP residual additions with the same final-oracle "
            "overlap count. $\\Delta$ is score minus matched random accuracy; "
            "wins are positive paired seed deltas.}",
            "\\label{tab:cifar-residual-imp-process-oracle-matched}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Dense acc. & Base acc. & "
            "$\\Delta_D$ & WinD & $\\Delta_B$ & WinB & Oracle \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_score_source_pairs"]:
        dense_delta = row["round_minus_dense_score"]
        base_delta = row["round_minus_base_score"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['dense_score_accuracy']['mean'])} & "
            f"{fmt(row['base_score_accuracy']['mean'])} & "
            f"{fmt(dense_delta['mean'])} & "
            f"{dense_delta['positive']}/{dense_delta['n']} & "
            f"{fmt(base_delta['mean'])} & "
            f"{base_delta['positive']}/{base_delta['n']} & "
            f"{fmt(row['oracle_overlap']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process score-source "
            "controls. Candidate set is final-IMP residual for all variants; "
            "dense/base controls rank that candidate set by dense-final or "
            "base-source magnitudes. $\\Delta_D$/$\\Delta_B$ are round-score "
            "minus dense/base score accuracy; wins are positive paired seed "
            "deltas.}",
            "\\label{tab:cifar-residual-imp-process-score-source}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Excl. acc. & $\\Delta$ & Wins & "
            "Round oracle & Excl. oracle \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_round_exclusion_pairs"]:
        delta = row["accuracy_delta"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['excluded_accuracy']['mean'])} & "
            f"{fmt(delta['mean'])} & "
            f"{delta['positive']}/{delta['n']} & "
            f"{fmt(row['round_oracle_overlap']['mean'])} & "
            f"{fmt(row['excluded_oracle_overlap']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed CIFAR residual IMP-process round-exclusion "
            "control. Round acc. uses the round-trained score ordering inside "
            "the final-IMP residual candidate set. Excl. acc. removes the "
            "round-selected additions from that candidate set, then chooses "
            "the best remaining final-IMP residual additions by final IMP "
            "magnitude under the same support budget. $\\Delta$ is round minus "
            "excluded accuracy; wins are positive paired seed deltas.}",
            "\\label{tab:cifar-residual-imp-process-round-exclusion}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Excl. acc. & Tensor excl. & "
            "$\\Delta_E$ & WinE & $\\Delta_T$ & WinT & Tensor oracle \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_layer_exclusion_pairs"]:
        excluded_delta = row["round_minus_excluded"]
        layer_delta = row["round_minus_layer_excluded"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['excluded_accuracy']['mean'])} & "
            f"{fmt(row['layer_excluded_accuracy']['mean'])} & "
            f"{fmt(excluded_delta['mean'])} & "
            f"{excluded_delta['positive']}/{excluded_delta['n']} & "
            f"{fmt(layer_delta['mean'])} & "
            f"{layer_delta['positive']}/{layer_delta['n']} & "
            f"{fmt(row['layer_excluded_oracle_overlap']['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process tensor-matched "
            "round-exclusion control at the RMS trajectory base and round 5. "
            "Tensor excl. removes the round-selected additions, then chooses "
            "replacement final-IMP residual additions matched to the removed "
            "coordinates by parameter tensor before applying the final IMP "
            "magnitude score. $\\Delta_E$ is round minus the unrestricted "
            "excluded oracle; $\\Delta_T$ is round minus the tensor-matched "
            "excluded oracle.}",
            "\\label{tab:cifar-residual-imp-process-layer-exclusion}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Tensor excl. & Tensor+score excl. & "
            "$\\Delta_T$ & WinT & $\\Delta_{TS}$ & WinTS \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_tensor_score_exclusion_pairs"]:
        layer_delta = row["round_minus_layer_excluded"]
        tensor_score_delta = row["round_minus_tensor_score_excluded"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['layer_excluded_accuracy']['mean'])} & "
            f"{fmt(row['tensor_score_excluded_accuracy']['mean'])} & "
            f"{fmt(layer_delta['mean'])} & "
            f"{layer_delta['positive']}/{layer_delta['n']} & "
            f"{fmt(tensor_score_delta['mean'])} & "
            f"{tensor_score_delta['positive']}/{tensor_score_delta['n']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process tensor+score-"
            "matched round-exclusion control at the RMS trajectory base and "
            "round 5. Tensor+score excl. removes the round-selected additions, "
            "then chooses replacement final-IMP residual additions matched to "
            "the removed coordinates by parameter tensor and within-tensor "
            "round-score decile before applying final IMP magnitude. "
            "$\\Delta_T$ is round minus the tensor-matched excluded oracle; "
            "$\\Delta_{TS}$ is round minus the tensor+score-matched excluded "
            "oracle.}",
            "\\label{tab:cifar-residual-imp-process-tensor-score-exclusion}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Resid. acc. & Dense acc. & Base acc. & "
            "$\\Delta_R$ & WinR & Oracle drop \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_projection_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['residualized_accuracy']['mean'])} & "
            f"{fmt(row['dense_score_accuracy']['mean'])} & "
            f"{fmt(row['base_score_accuracy']['mean'])} & "
            f"{fmt(delta['mean'])} & "
            f"{delta['positive']}/{delta['n']} & "
            f"{fmt(oracle_delta['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process projection "
            "control at the RMS trajectory base and round 5. Resid. acc. "
            "uses a round-score ordering after linearly projecting out the "
            "base-source, dense-final, and final-IMP magnitude scores inside "
            "the final-IMP residual candidate pool. $\\Delta_R$ is round "
            "minus residualized-score accuracy; Oracle drop is round oracle "
            "overlap minus residualized oracle overlap.}",
            "\\label{tab:cifar-residual-imp-process-projection}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Post-resid. acc. & Dense acc. & Base acc. & "
            "$\\Delta_P$ & WinP & Oracle drop \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_posterior_projection_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['residualized_accuracy']['mean'])} & "
            f"{fmt(row['dense_score_accuracy']['mean'])} & "
            f"{fmt(row['base_score_accuracy']['mean'])} & "
            f"{fmt(delta['mean'])} & "
            f"{delta['positive']}/{delta['n']} & "
            f"{fmt(oracle_delta['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process posterior-projection "
            "control at the RMS trajectory base and round 5. Post-resid. acc. "
            "uses a round-score ordering after linearly projecting out the "
            "base-source, dense-final, final-IMP magnitude, and diagonal-Laplace "
            "posterior RMS/std/excess scores inside the final-IMP residual "
            "candidate pool. $\\Delta_P$ is round minus posterior-residualized "
            "accuracy; Oracle drop is round oracle overlap minus "
            "posterior-residualized oracle overlap.}",
            "\\label{tab:cifar-residual-imp-process-posterior-projection}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Base & Rnd & Round acc. & Subspace-resid. acc. & Dense acc. & Base acc. & "
            "$\\Delta_S$ & WinS & Oracle drop \\\\",
            "\\midrule",
        ]
    )
    for row in stats["residual_imp_process_learned_subspace_pairs"]:
        delta = row["round_minus_residualized"]
        oracle_delta = row["round_minus_residualized_oracle"]
        lines.append(
            f"{source_label(row['base_source'])} & "
            f"{fmt_round(row['process_round'])} & "
            f"{fmt(row['round_accuracy']['mean'])} & "
            f"{fmt(row['residualized_accuracy']['mean'])} & "
            f"{fmt(row['dense_score_accuracy']['mean'])} & "
            f"{fmt(row['base_score_accuracy']['mean'])} & "
            f"{fmt(delta['mean'])} & "
            f"{delta['positive']}/{delta['n']} & "
            f"{fmt(oracle_delta['mean'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Five-seed CIFAR residual IMP-process learned-subspace "
            "control at the RMS trajectory base and round 5. Subspace-resid. "
            "acc. uses a round-score ordering after projecting out the top "
            "rank-8 PCA component scores learned from dense trajectory, "
            "final-IMP magnitude, and earlier IMP-round scores inside the "
            "final-IMP residual candidate pool. $\\Delta_S$ is round minus "
            "learned-subspace-residualized accuracy; Oracle drop is round "
            "oracle overlap minus learned-subspace-residualized oracle "
            "overlap.}",
            "\\label{tab:cifar-residual-imp-process-learned-subspace}",
            "\\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stats = build_stats()
    write_markdown(stats, args.out_md)
    write_tex(stats, args.out_tex)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_tex}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
