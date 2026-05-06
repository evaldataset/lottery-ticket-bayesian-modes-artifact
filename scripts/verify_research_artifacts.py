#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-package-mode",
        action="store_true",
        help=(
            "Verify an extracted public release package. This mode skips checks "
            "for the outer release tarball and archive-smoke sidecars, which "
            "cannot be included inside the tarball without self-reference."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str, min_size: int = 1) -> None:
    full_path = ROOT / path
    if not full_path.exists():
        fail(f"missing required file: {path}")
    if full_path.stat().st_size < min_size:
        fail(f"required file is too small: {path}")


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def rows_matching(rows: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    matched = []
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            matched.append(row)
    return matched


def require_stats_sections(stats: dict[str, Any]) -> None:
    required_sections = [
        "gate1",
        "movement",
        "head_laplace",
        "block_laplace",
        "subspace_hmc",
        "mode_distribution_equivalence",
        "direct_mode_ticket_distribution",
        "calibration_ood",
        "trajectory_mask_training",
        "variational_pruning",
        "trajectory_residual",
        "residual_anatomy_global",
        "residual_predictor_mask",
        "residual_cross_seed_transfer",
        "residual_direct_transfer",
        "residual_base_compatibility",
        "residual_base_ordering",
        "residual_stratified_controls",
        "residual_imp_process",
        "residual_imp_process_controls",
        "residual_imp_process_oracle_matched",
        "residual_imp_process_score_source",
        "residual_imp_process_round_exclusion",
        "residual_imp_process_layer_exclusion",
        "residual_imp_process_layer_exclusion_pairs",
        "residual_imp_process_tensor_score_exclusion",
        "residual_imp_process_tensor_score_exclusion_pairs",
        "residual_imp_process_projection",
        "residual_imp_process_projection_pairs",
        "residual_imp_process_posterior_projection",
        "residual_imp_process_posterior_projection_pairs",
        "residual_imp_process_learned_subspace",
        "residual_imp_process_learned_subspace_pairs",
    ]
    for section in required_sections:
        if section not in stats:
            fail(f"paper_stats.json missing section: {section}")
        if not stats[section]:
            fail(f"paper_stats.json section is empty: {section}")


def require_gate1(stats: dict[str, Any]) -> None:
    labels = {row["label"] for row in stats["gate1"]}
    required_labels = {
        "MNIST: posterior - random",
        "MNIST: posterior - chain-start",
        "MNIST: dense magnitude - posterior",
        "Fashion-MNIST: posterior - random",
        "Fashion-MNIST: posterior - chain-start",
        "Fashion-MNIST: dense magnitude - posterior",
    }
    missing = sorted(required_labels - labels)
    if missing:
        fail(f"Gate1 summary missing labels: {missing}")
    for row in stats["gate1"]:
        if row["label"].endswith("posterior - chain-start"):
            if abs(float(row["mean"])) > 0.005:
                fail(f"posterior-chain Gate1 delta is not near control: {row}")
        if row["label"].endswith("dense magnitude - posterior"):
            if float(row["mean"]) <= 0.0:
                fail(f"dense magnitude should dominate posterior masks: {row}")


def require_movement(stats: dict[str, Any]) -> None:
    rows = stats["movement"]
    lowrank_expectations = {
        "LowRankLap": (
            16,
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3",
        ),
        "LowRank32Lap": (
            32,
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3",
        ),
        "LowRank64Lap": (
            64,
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3",
        ),
        "LowRank128Lap": (
            128,
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3",
        ),
    }
    for sampler in lowrank_expectations:
        lowrank_rows = [
            row
            for row in rows
            if row.get("sampler") == sampler and abs(float(row.get("scale")) - 1e-2) < 1e-12
        ]
        if not lowrank_rows:
            fail(f"{sampler} movement row at scale 1e-2 is missing")
        row = lowrank_rows[0]
        if int(row["posterior_minus_chain"]["n"]) != 5:
            fail(f"{sampler} movement row must be five-seed")
        if float(row["posterior_minus_chain"]["mean"]) >= 0.0:
            fail(f"{sampler} should not beat chain-start support")
        if float(row["post_chain"]["mean"]) >= 0.80:
            fail(f"{sampler} scale 1e-2 should move away from chain-start support")
        if float(row["sample_accuracy"]["mean"]) <= 0.87:
            fail(f"{sampler} scale 1e-2 sample accuracy is unexpectedly low")
    for sampler, (expected_rank, root) in lowrank_expectations.items():
        metrics = sorted(root.glob("*/metrics.json"))
        if len(metrics) != 5:
            fail(f"{sampler} raw metrics should contain five seeds, found {len(metrics)}")
        for path in metrics:
            payload = load_json(path)
            for row in payload.get("rows", []):
                if int(row.get("lowrank_laplace_positive_rank", 0)) != expected_rank:
                    fail(
                        f"{sampler} did not retain {expected_rank} positive directions in {path}"
                    )


def require_block_laplace(stats: dict[str, Any]) -> None:
    rows = rows_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:11blocks<=5000",
        scale=0.0001,
    )
    if not rows:
        fail("BlockDiagLap selected CIFAR row is missing")
    row = rows[0]
    if int(row["block_posterior_minus_chain"]["n"]) != 5:
        fail("BlockDiagLap row must be five-seed")
    if float(row["parameter_count"]["mean"]) < 22_000:
        fail("BlockDiagLap row should cover at least 22k parameters")
    if float(row["block_posterior_minus_chain"]["mean"]) >= 0.0:
        fail("BlockDiagLap block posterior should not beat chain-start support")
    if float(row["global_posterior_minus_chain"]["mean"]) >= 0.01:
        fail("BlockDiagLap global posterior gain should remain small")
    if float(row["global_rewind_minus_posterior"]["mean"]) <= 0.02:
        fail("BlockDiagLap rewind support should remain closer than posterior")
    if float(row["sample_accuracy"]["mean"]) <= 0.87:
        fail("BlockDiagLap samples should preserve useful accuracy")
    root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3"
    )
    metrics = sorted(root.glob("*/metrics.json"))
    if len(metrics) != 5:
        fail(f"BlockDiagLap raw metrics should contain five seeds, found {len(metrics)}")
    for path in metrics:
        payload = load_json(path)
        config = payload.get("block_laplace", {})
        if not config.get("independent_block_diagonal"):
            fail(f"BlockDiagLap raw metrics missing independent blockdiag flag: {path}")
        if len(config.get("block_names", [])) != 11:
            fail(f"BlockDiagLap raw metrics should contain 11 block names: {path}")
    wider_rows = rows_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:16blocks<=10000",
        scale=1e-05,
    )
    if not wider_rows:
        fail("BlockDiagLap max10k CIFAR row is missing")
    row = wider_rows[0]
    if int(row["block_posterior_minus_chain"]["n"]) != 5:
        fail("BlockDiagLap max10k row must be five-seed")
    if float(row["parameter_count"]["mean"]) < 68_000:
        fail("BlockDiagLap max10k row should cover at least 68k parameters")
    if float(row["block_posterior_minus_chain"]["mean"]) >= 0.0:
        fail("BlockDiagLap max10k block posterior should not beat chain-start support")
    if float(row["global_posterior_minus_chain"]["mean"]) >= 0.005:
        fail("BlockDiagLap max10k global posterior gain should remain tiny")
    if float(row["global_rewind_minus_posterior"]["mean"]) <= 0.025:
        fail("BlockDiagLap max10k rewind support should remain closer than posterior")
    if float(row["global_post_chain"]["mean"]) >= 0.80:
        fail("BlockDiagLap max10k samples should move from chain-start support")
    if float(row["sample_accuracy"]["mean"]) <= 0.875:
        fail("BlockDiagLap max10k samples should preserve useful accuracy")
    wider_root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3"
    )
    metrics = sorted(wider_root.glob("*/metrics.json"))
    if len(metrics) != 5:
        fail(f"BlockDiagLap max10k raw metrics should contain five seeds, found {len(metrics)}")
    for path in metrics:
        payload = load_json(path)
        config = payload.get("block_laplace", {})
        if not config.get("independent_block_diagonal"):
            fail(f"BlockDiagLap max10k raw metrics missing independent blockdiag flag: {path}")
        if int(config.get("max_parameters", 0)) != 10_000:
            fail(f"BlockDiagLap max10k raw metrics should use max_parameters=10000: {path}")
        if len(config.get("block_names", [])) != 16:
            fail(f"BlockDiagLap max10k raw metrics should contain 16 block names: {path}")
    jointdiag_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=10000",
        scale=1e-05,
    )
    if not jointdiag_rows:
        fail("JointDiagLap max10k CIFAR row is missing")
    row = jointdiag_rows[0]
    if int(row["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max10k row must be five-seed")
    if float(row["parameter_count"]["mean"]) < 68_000:
        fail("JointDiagLap max10k row should cover at least 68k parameters")
    if float(row["block_posterior_minus_chain"]["mean"]) >= 0.0:
        fail("JointDiagLap max10k block posterior should not beat chain-start support")
    if float(row["global_posterior_minus_chain"]["mean"]) >= 0.005:
        fail("JointDiagLap max10k global posterior gain should remain tiny")
    if float(row["global_rewind_minus_posterior"]["mean"]) <= 0.025:
        fail("JointDiagLap max10k rewind support should remain closer than posterior")
    if float(row["global_post_chain"]["mean"]) >= 0.80:
        fail("JointDiagLap max10k samples should move from chain-start support")
    if float(row["sample_accuracy"]["mean"]) <= 0.875:
        fail("JointDiagLap max10k samples should preserve useful accuracy")
    jointdiag_root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3"
    )
    metrics = sorted(jointdiag_root.glob("*/metrics.json"))
    if len(metrics) != 5:
        fail(f"JointDiagLap max10k raw metrics should contain five seeds, found {len(metrics)}")
    for path in metrics:
        payload = load_json(path)
        config = payload.get("block_laplace", {})
        if not config.get("joint_block_diagonal"):
            fail(f"JointDiagLap max10k raw metrics missing joint blockdiag flag: {path}")
        if int(config.get("max_parameters", 0)) != 10_000:
            fail(f"JointDiagLap max10k raw metrics should use max_parameters=10000: {path}")
        if len(config.get("block_names", [])) != 16:
            fail(f"JointDiagLap max10k raw metrics should contain 16 block names: {path}")
        if len(config.get("block_groups", [])) != 8:
            fail(f"JointDiagLap max10k raw metrics should contain eight joint groups: {path}")
    jointdiag20_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:6groups<=20000",
        scale=3e-06,
    )
    if not jointdiag20_rows:
        fail("JointDiagLap max20k CIFAR row is missing")
    row = jointdiag20_rows[0]
    if int(row["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max20k row must be five-seed")
    if float(row["parameter_count"]["mean"]) < 86_000:
        fail("JointDiagLap max20k row should cover at least 86k parameters")
    if float(row["block_posterior_minus_chain"]["mean"]) >= 0.0:
        fail("JointDiagLap max20k block posterior should not beat chain-start support")
    if float(row["global_posterior_minus_chain"]["mean"]) >= 0.005:
        fail("JointDiagLap max20k global posterior gain should remain tiny")
    if float(row["global_rewind_minus_posterior"]["mean"]) <= 0.025:
        fail("JointDiagLap max20k rewind support should remain closer than posterior")
    if float(row["global_post_chain"]["mean"]) >= 0.85:
        fail("JointDiagLap max20k samples should move from chain-start support")
    if float(row["sample_accuracy"]["mean"]) <= 0.875:
        fail("JointDiagLap max20k samples should preserve useful accuracy")
    jointdiag20_root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3"
    )
    metrics = sorted(jointdiag20_root.glob("*/metrics.json"))
    if len(metrics) != 5:
        fail(f"JointDiagLap max20k raw metrics should contain five seeds, found {len(metrics)}")
    for path in metrics:
        payload = load_json(path)
        config = payload.get("block_laplace", {})
        if not config.get("joint_block_diagonal"):
            fail(f"JointDiagLap max20k raw metrics missing joint blockdiag flag: {path}")
        if int(config.get("max_parameters", 0)) != 20_000:
            fail(f"JointDiagLap max20k raw metrics should use max_parameters=20000: {path}")
        if len(config.get("block_names", [])) != 17:
            fail(f"JointDiagLap max20k raw metrics should contain 17 block names: {path}")
        if len(config.get("block_groups", [])) != 6:
            fail(f"JointDiagLap max20k raw metrics should contain six joint groups: {path}")
    jointdiag40_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=40000",
        scale=1e-06,
    )
    if not jointdiag40_rows:
        fail("JointDiagLap max40k CIFAR row is missing")
    row = jointdiag40_rows[0]
    if int(row["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max40k row must be five-seed")
    if float(row["parameter_count"]["mean"]) < 270_000:
        fail("JointDiagLap max40k row should cover the full weight vector")
    if float(row["block_posterior_minus_chain"]["mean"]) >= 0.0:
        fail("JointDiagLap max40k block posterior should not beat chain-start support")
    if float(row["global_posterior_minus_chain"]["mean"]) >= 0.001:
        fail("JointDiagLap max40k global posterior gain should remain non-positive or tiny")
    if float(row["global_rewind_minus_posterior"]["mean"]) <= 0.03:
        fail("JointDiagLap max40k rewind support should remain closer than posterior")
    if float(row["global_post_chain"]["mean"]) >= 0.80:
        fail("JointDiagLap max40k samples should move from chain-start support")
    if float(row["sample_accuracy"]["mean"]) <= 0.875:
        fail("JointDiagLap max40k samples should preserve useful accuracy")
    jointdiag40_root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3"
    )
    metrics = sorted(jointdiag40_root.glob("*/metrics.json"))
    if len(metrics) != 5:
        fail(f"JointDiagLap max40k raw metrics should contain five seeds, found {len(metrics)}")
    for path in metrics:
        payload = load_json(path)
        config = payload.get("block_laplace", {})
        if not config.get("joint_block_diagonal"):
            fail(f"JointDiagLap max40k raw metrics missing joint blockdiag flag: {path}")
        if not config.get("stream_joint_groups"):
            fail(f"JointDiagLap max40k raw metrics should stream joint groups: {path}")
        if int(config.get("max_parameters", 0)) != 40_000:
            fail(f"JointDiagLap max40k raw metrics should use max_parameters=40000: {path}")
        if len(config.get("block_names", [])) != 22:
            fail(f"JointDiagLap max40k raw metrics should contain 22 block names: {path}")
        if len(config.get("block_groups", [])) != 8:
            fail(f"JointDiagLap max40k raw metrics should contain eight joint groups: {path}")


def require_subspace_hmc(stats: dict[str, Any]) -> None:
    rows = stats["subspace_hmc"]
    hess32_rows = [
        row
        for row in rows
        if row.get("sampler") == "Hess32SubHMC"
        and abs(float(row.get("step")) - 3e-4) < 1e-12
    ]
    if not hess32_rows:
        fail("Hess32SubHMC selected subspace-HMC row is missing")
    row = hess32_rows[0]
    for key in [
        "posterior_minus_chain",
        "rewind_minus_posterior",
        "post_chain",
        "sample_accuracy",
        "accept_rate",
        "parameter_distance",
    ]:
        if int(row[key]["n"]) != 5:
            fail(f"Hess32SubHMC {key} summary must be five-seed")
    if abs(float(row["posterior_minus_chain"]["mean"])) >= 0.001:
        fail("Hess32SubHMC should remain practically tied to chain-start support")
    if float(row["rewind_minus_posterior"]["mean"]) <= 0.02:
        fail("Hess32SubHMC rewind control should remain closer to IMP")
    if float(row["post_chain"]["mean"]) <= 0.99:
        fail("Hess32SubHMC post-chain overlap should remain near chain-start")
    if float(row["sample_accuracy"]["mean"]) <= 0.88:
        fail("Hess32SubHMC sample accuracy is unexpectedly low")
    if float(row["accept_rate"]["mean"]) <= 0.80:
        fail("Hess32SubHMC accept rate is unexpectedly low")
    if float(row["parameter_distance"]["mean"]) <= 0.005:
        fail("Hess32SubHMC parameter movement is unexpectedly small")


def require_mode_distribution_audit(stats: dict[str, Any]) -> None:
    rows = stats["mode_distribution_equivalence"]
    if len(rows) < 150:
        fail(f"mode distribution audit has too few grouped rows: {len(rows)}")
    random_rows = [row for row in rows if row["comparison"] == "posterior-random"]
    chain_rows = [row for row in rows if row["comparison"] == "posterior-chain"]
    if len(random_rows) < 40 or len(chain_rows) < 40:
        fail("mode distribution audit is missing random or chain controls")
    random_wins = [
        row
        for row in random_rows
        if str(row.get("verdict", "")) == "posterior separates from random"
    ]
    chain_wins = [row for row in chain_rows if float(row["delta_mean"]) > 0.005]
    if len(random_wins) < 40:
        fail(f"posterior-vs-random support evidence too weak: {len(random_wins)}")
    if chain_wins:
        fail(f"posterior beats chain-start by >0.005 in audit rows: {chain_wins[:3]}")


def require_direct_mode_ticket(stats: dict[str, Any]) -> None:
    rows = stats["direct_mode_ticket_distribution"]
    required_settings = {
        "Digits MLP",
        "CIFAR full ResNet",
        "CIFAR full aligned",
        "CIFAR full weight-aligned",
        "CIFAR full cSGLD multi-chain",
        "CIFAR full cSGLD independent",
        "CIFAR full LowRank128Lap",
        "CIFAR full JointDiagLap270k",
    }
    settings = {row.get("setting") for row in rows}
    missing = sorted(required_settings - settings)
    if missing:
        fail(f"direct mode/ticket table missing settings: {missing}")
    for setting in ["Digits MLP", "CIFAR full ResNet"]:
        row = rows_matching(
            rows,
            setting=setting,
            comparison="posterior_samples_vs_tickets",
        )[0]
        if int(float(row["posterior_num_clusters"])) != 1:
            fail(f"{setting} should collapse to one mean-shift cluster")
        if float(row["posterior_effective_cluster_count"]) != 1.0:
            fail(f"{setting} should have effective cluster count 1")
        if float(row["hamming_overlap"]) >= 0.70:
            fail(f"{setting} should fail Hamming-overlap threshold")
        if float(row["logit_cka_hungarian_mean"]) <= 0.85:
            fail(f"{setting} should pass logit CKA threshold")
        if setting == "CIFAR full ResNet":
            if float(row["layer_ks_pvalue"]) >= 0.001:
                fail("full CIFAR direct row should strongly fail layer KS")
            if float(row["activation_cka_hungarian_mean"]) <= 0.85:
                fail("full CIFAR direct row should pass final-hidden activation CKA")
    full_mode_row = rows_matching(
        rows,
        setting="CIFAR full ResNet",
        comparison="posterior_modes_vs_tickets",
    )[0]
    if int(float(full_mode_row["left_count"])) != 1:
        fail("full CIFAR mode row should have one representative")
    if float(full_mode_row["layer_ks_pvalue"]) >= 0.10:
        fail("full CIFAR mode row should fail the layer KS threshold")
    aligned_sample_rows = rows_matching(
        rows,
        setting="CIFAR full aligned",
        comparison="activation_aligned_posterior_samples_vs_tickets",
    )
    if not aligned_sample_rows:
        fail("aligned full CIFAR direct row is missing")
    row = aligned_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("aligned full CIFAR direct row should collapse to one cluster")
    if int(float(row["left_count"])) != 50 or int(float(row["right_count"])) != 5:
        fail("aligned full CIFAR direct row should compare 50 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("aligned full CIFAR direct row should strongly fail layer KS")
    if float(row["hamming_overlap"]) >= 0.70:
        fail("aligned full CIFAR direct row should fail Hamming-overlap threshold")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("aligned full CIFAR direct row should pass final-hidden activation CKA")
    weight_aligned_sample_rows = rows_matching(
        rows,
        setting="CIFAR full weight-aligned",
        comparison="weight_aligned_posterior_samples_vs_tickets",
    )
    if not weight_aligned_sample_rows:
        fail("weight-aligned full CIFAR direct row is missing")
    row = weight_aligned_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("weight-aligned full CIFAR direct row should collapse to one cluster")
    if int(float(row["left_count"])) != 50 or int(float(row["right_count"])) != 5:
        fail("weight-aligned full CIFAR direct row should compare 50 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("weight-aligned full CIFAR direct row should strongly fail layer KS")
    if float(row["hamming_overlap"]) >= 0.70:
        fail("weight-aligned full CIFAR direct row should fail Hamming-overlap threshold")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("weight-aligned full CIFAR direct row should pass final-hidden activation CKA")
    csgld_sample_rows = rows_matching(
        rows,
        setting="CIFAR full cSGLD multi-chain",
        comparison="posterior_samples_vs_tickets",
    )
    if not csgld_sample_rows:
        fail("multi-chain cSGLD full CIFAR direct row is missing")
    row = csgld_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("multi-chain cSGLD full CIFAR direct row should collapse to one cluster")
    if int(float(row["left_count"])) != 75 or int(float(row["right_count"])) != 5:
        fail("multi-chain cSGLD full CIFAR direct row should compare 75 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("multi-chain cSGLD full CIFAR direct row should strongly fail layer KS")
    if float(row["hamming_overlap"]) >= 0.70:
        fail("multi-chain cSGLD full CIFAR direct row should fail Hamming-overlap threshold")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("multi-chain cSGLD full CIFAR direct row should pass final-hidden activation CKA")
    csgld_chain_rows = rows_matching(
        rows,
        setting="CIFAR full cSGLD multi-chain",
        comparison="chain_start_magnitude_vs_tickets",
    )
    if not csgld_chain_rows:
        fail("multi-chain cSGLD full CIFAR chain-start control row is missing")
    if int(float(csgld_chain_rows[0]["left_count"])) != 15:
        fail("multi-chain cSGLD full CIFAR should record 15 chain-start masks")
    csgld_independent_sample_rows = rows_matching(
        rows,
        setting="CIFAR full cSGLD independent",
        comparison="posterior_samples_vs_tickets",
    )
    if not csgld_independent_sample_rows:
        fail("independent-start cSGLD full CIFAR direct row is missing")
    row = csgld_independent_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("independent-start cSGLD full CIFAR direct row should collapse to one cluster")
    if int(float(row["left_count"])) != 75 or int(float(row["right_count"])) != 5:
        fail("independent-start cSGLD full CIFAR direct row should compare 75 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("independent-start cSGLD full CIFAR direct row should strongly fail layer KS")
    if float(row["hamming_overlap"]) >= 0.70:
        fail("independent-start cSGLD full CIFAR direct row should fail Hamming-overlap threshold")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("independent-start cSGLD full CIFAR direct row should pass final-hidden activation CKA")
    csgld_independent_chain_rows = rows_matching(
        rows,
        setting="CIFAR full cSGLD independent",
        comparison="chain_start_magnitude_vs_tickets",
    )
    if not csgld_independent_chain_rows:
        fail("independent-start cSGLD full CIFAR chain-start control row is missing")
    if int(float(csgld_independent_chain_rows[0]["left_count"])) != 15:
        fail("independent-start cSGLD full CIFAR should record 15 chain-start masks")
    lowrank_sample_rows = rows_matching(
        rows,
        setting="CIFAR full LowRank128Lap",
        comparison="posterior_samples_vs_tickets",
    )
    if not lowrank_sample_rows:
        fail("LowRank128Lap full CIFAR direct row is missing")
    row = lowrank_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("LowRank128Lap full CIFAR direct row should collapse to one cluster")
    if float(row["posterior_effective_cluster_count"]) != 1.0:
        fail("LowRank128Lap full CIFAR direct row should have effective cluster count 1")
    if int(float(row["left_count"])) != 50 or int(float(row["right_count"])) != 5:
        fail("LowRank128Lap full CIFAR direct row should compare 50 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("LowRank128Lap full CIFAR direct row should still strongly fail layer KS")
    if float(row["hamming_overlap"]) <= 0.70:
        fail("LowRank128Lap full CIFAR direct row should pass Hamming-overlap threshold")
    if float(row["logit_cka_hungarian_mean"]) <= 0.85:
        fail("LowRank128Lap full CIFAR direct row should pass logit CKA")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("LowRank128Lap full CIFAR direct row should pass final-hidden activation CKA")
    jointdiag_sample_rows = rows_matching(
        rows,
        setting="CIFAR full JointDiagLap270k",
        comparison="posterior_samples_vs_tickets",
    )
    if not jointdiag_sample_rows:
        fail("JointDiagLap270k full CIFAR direct row is missing")
    row = jointdiag_sample_rows[0]
    if int(float(row["posterior_num_clusters"])) != 1:
        fail("JointDiagLap270k full CIFAR direct row should collapse to one cluster")
    if float(row["posterior_effective_cluster_count"]) != 1.0:
        fail("JointDiagLap270k full CIFAR direct row should have effective cluster count 1")
    if int(float(row["left_count"])) != 25 or int(float(row["right_count"])) != 5:
        fail("JointDiagLap270k full CIFAR direct row should compare 25 samples to 5 tickets")
    if float(row["layer_ks_pvalue"]) >= 0.001:
        fail("JointDiagLap270k full CIFAR direct row should strongly fail layer KS")
    if float(row["hamming_overlap"]) >= 0.70:
        fail("JointDiagLap270k full CIFAR direct row should fail Hamming-overlap threshold")
    if float(row["logit_cka_hungarian_mean"]) <= 0.85:
        fail("JointDiagLap270k full CIFAR direct row should pass logit CKA")
    if float(row["activation_cka_hungarian_mean"]) <= 0.85:
        fail("JointDiagLap270k full CIFAR direct row should pass final-hidden activation CKA")
    metrics_path = ROOT / str(row["run"]) / "metrics.json"
    if not metrics_path.exists():
        fail(f"JointDiagLap270k raw metrics are missing: {metrics_path}")
    payload = load_json(metrics_path)
    config = payload.get("config", {})
    if config.get("posterior_sampler") != "jointdiag-laplace":
        fail("JointDiagLap270k direct metrics should use jointdiag-laplace sampler")
    sampler_config = config.get("posterior_sampler_config", {})
    if not sampler_config.get("stream_joint_groups"):
        fail("JointDiagLap270k direct metrics should stream joint groups")
    if int(sampler_config.get("max_parameters", 0)) != 40_000:
        fail("JointDiagLap270k direct metrics should use max_parameters=40000")
    if abs(float(sampler_config.get("scale", 0.0)) - 1e-6) > 1e-12:
        fail("JointDiagLap270k direct metrics should use scale=1e-6")
    diagnostics = payload.get("posterior_chain_diagnostics", {})
    if int(diagnostics.get("posterior_sample_count", 0)) != 25:
        fail("JointDiagLap270k direct metrics should contain 25 posterior samples")
    if float(diagnostics.get("posterior_sample_accuracy_mean", 0.0)) <= 0.875:
        fail("JointDiagLap270k direct samples should preserve useful accuracy")
    hamming = float(diagnostics.get("posterior_to_chain_start_hamming_mean", math.nan))
    if not (0.045 <= hamming <= 0.055):
        fail("JointDiagLap270k direct posterior movement should remain around 0.05 Hamming")


def require_mode_ticket_alignment_artifact_audit() -> None:
    audit_path = ROOT / "runs" / "mode_ticket_alignment_artifact_audit.json"
    doc_path = ROOT / "docs" / "mode_ticket_alignment_artifact_audit.md"
    payload = load_json(audit_path)
    overall = payload.get("overall", {})
    if int(overall.get("run_count", 0)) < 7:
        fail("mode/ticket alignment artifact audit should cover seven full-data CIFAR runs")
    if int(overall.get("aligned_run_count", 0)) != 2:
        fail("mode/ticket alignment artifact audit should cover activation and weight alignment")
    if overall.get("aligned_rows_all_fail_layer_ks") is not True:
        fail("aligned direct rows should fail the layer-KS criterion")
    if overall.get("aligned_rows_all_fail_hamming_overlap") is not True:
        fail("aligned direct rows should fail the Hamming-overlap criterion")
    if overall.get("any_direct_equivalence_pass") is not False:
        fail("no audited direct run should pass full mode/ticket equivalence")
    if overall.get("direct_rows_all_collapse_to_one_basin") is not True:
        fail("audited direct runs should collapse posterior samples to one basin")
    if overall.get("raw_mask_artifacts_present") is not False:
        fail("current direct-run release should not claim saved raw mask artifacts")
    if int(overall.get("raw_mask_or_state_file_count", -1)) != 0:
        fail("current direct-run release should have zero raw mask/state files")
    if overall.get("posthoc_exhaustive_permutation_supported") is not False:
        fail("post-hoc exhaustive permutation support should be explicitly false")
    limitation = str(overall.get("posthoc_limitation_statement", ""))
    if "post-hoc exhaustive graph/permutation realignment is not supported" not in limitation:
        fail("mode/ticket alignment artifact audit missing limitation statement")

    runs = {
        str(run.get("label")): run
        for run in payload.get("runs", [])
        if isinstance(run, dict)
    }
    for label in [
        "CIFAR activation-aligned SGLD",
        "CIFAR weight-aligned SGLD",
        "CIFAR LowRank128 Laplace direct",
        "CIFAR JointDiagLap270k direct",
    ]:
        if label not in runs:
            fail(f"mode/ticket alignment artifact audit missing run: {label}")
    for label in ["CIFAR activation-aligned SGLD", "CIFAR weight-aligned SGLD"]:
        run = runs[label]
        row = run.get("direct_sample_row", {})
        if row.get("passes_layer_ks") is not False:
            fail(f"{label} should fail layer-KS after alignment")
        if row.get("passes_hamming_overlap") is not False:
            fail(f"{label} should fail Hamming-overlap after alignment")
        if float(row.get("activation_cka_hungarian_mean", 0.0)) <= 0.85:
            fail(f"{label} should retain high activation CKA")
        alignment = run.get("alignment", {})
        if int(alignment.get("target_frame_count", 0)) != 1:
            fail(f"{label} should record one seed-0 target frame")
    jointdiag = runs["CIFAR JointDiagLap270k direct"].get("direct_sample_row", {})
    if jointdiag.get("passes_layer_ks") is not False:
        fail("JointDiagLap270k audited direct row should fail layer-KS")
    if jointdiag.get("passes_hamming_overlap") is not False:
        fail("JointDiagLap270k audited direct row should fail Hamming-overlap")
    lowrank = runs["CIFAR LowRank128 Laplace direct"].get("direct_sample_row", {})
    if lowrank.get("passes_layer_ks") is not False:
        fail("LowRank128 audited direct row should still fail layer-KS")
    if lowrank.get("direct_equivalence_pass") is not False:
        fail("LowRank128 audited direct row should not pass full direct equivalence")

    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "post-hoc exhaustive graph/permutation realignment is not supported",
        "raw posterior/ticket mask or state tensors",
        "CIFAR activation-aligned SGLD",
        "CIFAR weight-aligned SGLD",
        "Raw mask/state artifacts present: `False`",
    ]:
        if phrase not in text:
            fail(f"mode/ticket alignment artifact audit doc missing phrase: {phrase}")


def require_mode_ticket_mask_artifact_smoke() -> None:
    root = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_smoke"
    metrics_paths = sorted(root.glob("*/metrics.json"))
    if not metrics_paths:
        fail("mode/ticket mask-artifact smoke metrics are missing")
    metrics_path = metrics_paths[-1]
    payload = load_json(metrics_path)
    config = payload.get("config", {})
    if config.get("dataset") != "fake-cifar10" or config.get("model") != "resnet20":
        fail("mode/ticket mask-artifact smoke should be fake-CIFAR ResNet-20")
    if config.get("save_mask_artifacts") is not True:
        fail("mode/ticket mask-artifact smoke did not enable mask saving")
    if config.get("save_state_artifacts") is not True:
        fail("mode/ticket mask-artifact smoke did not enable state saving")
    artifact = payload.get("mask_artifacts", {})
    artifact_path = ROOT / str(artifact.get("path", ""))
    if not artifact_path.exists():
        fail(f"mode/ticket mask artifact is missing: {artifact_path}")
    if int(artifact.get("parameter_count", 0)) != 4350:
        fail("mode/ticket mask-artifact smoke should cover 4,350 parameters")
    expected_collections = {
        "chain_start",
        "posterior_sample",
        "posterior_mode",
        "ticket",
        "activation_aligned_chain_start",
        "activation_aligned_posterior_sample",
        "activation_aligned_posterior_mode",
        "activation_aligned_ticket",
    }
    collections = {
        str(row.get("name")): row
        for row in artifact.get("collections", [])
        if isinstance(row, dict)
    }
    missing = sorted(expected_collections - set(collections))
    if missing:
        fail(f"mode/ticket mask artifact missing collections: {missing}")
    for name in expected_collections:
        row = collections[name]
        if int(row.get("masks", 0)) != 2 or int(row.get("ids", 0)) != 2:
            fail(f"mode/ticket mask artifact collection should have two masks: {name}")
        if int(row.get("states", 0)) != 2:
            fail(f"mode/ticket mask artifact collection should have two states: {name}")

    with np.load(artifact_path, allow_pickle=False) as data:
        if int(data["artifact_schema_version"][0]) != 1:
            fail("mode/ticket mask artifact schema should be version 1")
        parameter_count = int(data["parameter_sizes"].sum())
        if parameter_count != 4350:
            fail("mode/ticket mask artifact parameter sizes should sum to 4,350")
        if int(data["parameter_names"].shape[0]) != 22:
            fail("mode/ticket mask artifact should store 22 parameter names")
        if "parameter_shapes_json" not in data.files:
            fail("mode/ticket mask artifact missing parameter_shapes_json")
        shapes = json.loads(str(data["parameter_shapes_json"]))
        if shapes.get("conv1.weight") != [2, 3, 3, 3]:
            fail("mode/ticket mask artifact has wrong conv1 shape metadata")
        if shapes.get("fc.weight") != [10, 8]:
            fail("mode/ticket mask artifact has wrong fc shape metadata")
        for key in [
            "masks__posterior_sample",
            "masks__ticket",
            "masks__activation_aligned_posterior_sample",
            "masks__activation_aligned_ticket",
        ]:
            if key not in data.files:
                fail(f"mode/ticket mask artifact missing key: {key}")
            value = data[key]
            if value.shape != (2, parameter_count):
                fail(f"mode/ticket mask artifact has wrong shape for {key}: {value.shape}")
            if value.dtype != np.uint8:
                fail(f"mode/ticket mask artifact masks must be uint8 for {key}")
            if not np.isin(value, [0, 1]).all():
                fail(f"mode/ticket mask artifact masks must be binary for {key}")
        for key in [
            "states__posterior_sample",
            "states__ticket",
            "states__activation_aligned_posterior_sample",
            "states__activation_aligned_ticket",
        ]:
            if key not in data.files:
                fail(f"mode/ticket state artifact missing key: {key}")
            value = data[key]
            if value.shape != (2, parameter_count):
                fail(f"mode/ticket state artifact has wrong shape for {key}: {value.shape}")
            if value.dtype != np.float32:
                fail(f"mode/ticket state artifact states must be float32 for {key}")
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("mask_encoding") != "uint8_keep_mask_flattened_by_parameter_names":
            fail("mode/ticket mask artifact metadata missing mask encoding")
        if metadata.get("state_encoding") != "float32_flattened_by_parameter_names":
            fail("mode/ticket mask artifact metadata missing state encoding")
        if int(metadata.get("resnet_width", 0)) != 2:
            fail("mode/ticket mask artifact metadata missing resnet_width=2")
        if metadata.get("input_shape") != [3, 32, 32]:
            fail("mode/ticket mask artifact metadata missing input shape")

    doc_text = (
        ROOT / "docs" / "fake_cifar10_mode_ticket_mask_artifact_smoke.md"
    ).read_text(encoding="utf-8")
    for phrase in [
        "Mask Artifact Check",
        "--save-mask-artifacts --save-state-artifacts",
        "parameter_shapes_json",
        "masks__posterior_sample",
        "states__activation_aligned_posterior_sample",
    ]:
        if phrase not in doc_text:
            fail(f"mode/ticket mask artifact smoke doc missing phrase: {phrase}")


def require_mask_artifact_posthoc_audit() -> None:
    audit_path = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json"
    doc_path = ROOT / "docs" / "fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md"
    payload = load_json(audit_path)
    overall = payload.get("overall", {})
    if int(overall.get("schema_version", 0)) != 1:
        fail("mask artifact post-hoc audit should use schema version 1")
    if overall.get("dataset") != "fake-cifar10" or overall.get("model") != "resnet20":
        fail("mask artifact post-hoc audit should target fake-CIFAR ResNet-20")
    if int(overall.get("parameter_count", 0)) != 4350:
        fail("mask artifact post-hoc audit should cover 4,350 parameters")
    if int(overall.get("parameter_name_count", 0)) != 22:
        fail("mask artifact post-hoc audit should cover 22 parameter tensors")
    if int(overall.get("mask_collection_count", 0)) != 8:
        fail("mask artifact post-hoc audit should cover eight mask collections")
    if int(overall.get("state_collection_count", 0)) != 8:
        fail("mask artifact post-hoc audit should cover eight state collections")
    if overall.get("parameter_shapes_present") is not True:
        fail("mask artifact post-hoc audit should have parameter shapes")
    if int(overall.get("resnet_channel_key_count", 0)) != 19:
        fail("mask artifact post-hoc audit should expose 19 ResNet channel keys")
    if overall.get("required_collections_present") is not True:
        fail("mask artifact post-hoc audit missing required collections")
    if overall.get("record_level_posthoc_matching_supported") is not True:
        fail("mask artifact post-hoc audit should support record-level matching")
    if overall.get("local_channel_permutation_matching_supported") is not True:
        fail("mask artifact post-hoc audit should support local channel matching")
    if overall.get("exhaustive_graph_channel_permutation_supported") is not False:
        fail("mask artifact post-hoc audit must not claim exhaustive channel permutation support")
    if "full-data rerun" not in str(overall.get("limitation_statement", "")):
        fail("mask artifact post-hoc audit missing full-data rerun limitation")

    artifact_path = ROOT / str(payload.get("artifact_path", ""))
    if not artifact_path.exists():
        fail(f"mask artifact post-hoc audit references missing artifact: {artifact_path}")

    collections = {
        str(row.get("name")): row
        for row in payload.get("collections", [])
        if isinstance(row, dict)
    }
    for name in [
        "posterior_sample",
        "ticket",
        "activation_aligned_posterior_sample",
        "activation_aligned_ticket",
    ]:
        row = collections.get(name)
        if row is None:
            fail(f"mask artifact post-hoc audit missing collection: {name}")
        if int(row.get("mask_count", 0)) != 2 or int(row.get("state_count", 0)) != 2:
            fail(f"mask artifact post-hoc audit collection should have two records: {name}")
        if abs(float(row.get("keep_fraction_mean", 0.0)) - 0.7) > 1e-6:
            fail(f"mask artifact post-hoc audit collection keep fraction changed: {name}")

    comparisons = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in payload.get("comparisons", [])
        if isinstance(row, dict)
    }
    required_comparisons = [
        ("posterior_sample", "ticket"),
        ("activation_aligned_posterior_sample", "activation_aligned_ticket"),
        ("posterior_sample", "activation_aligned_posterior_sample"),
        ("ticket", "activation_aligned_ticket"),
    ]
    for key in required_comparisons:
        row = comparisons.get(key)
        if row is None:
            fail(f"mask artifact post-hoc audit missing comparison: {key}")
        natural = row.get("natural_hamming", {}).get("mean")
        optimal = row.get("optimal_hamming", {}).get("mean")
        if not finite(natural) or not finite(optimal):
            fail(f"mask artifact post-hoc audit comparison has non-finite hamming: {key}")
        if float(optimal) - float(natural) > 1e-12:
            fail(f"mask artifact post-hoc audit optimal hamming worsens direct pairing: {key}")
        if int(row.get("same_index_pair_count", 0)) != 2:
            fail(f"mask artifact post-hoc audit should compare two same-index pairs: {key}")
        if int(row.get("optimal_pair_count", 0)) != 2:
            fail(f"mask artifact post-hoc audit should compare two optimal pairs: {key}")
        if row.get("state_distance") is None:
            fail(f"mask artifact post-hoc audit should include state distance: {key}")
        channel = row.get("channel_permutation")
        if not isinstance(channel, dict):
            fail(f"mask artifact post-hoc audit should include channel permutation: {key}")
        if int(channel.get("channel_key_count", 0)) != 19:
            fail(f"mask artifact post-hoc audit channel key count changed: {key}")
        channel_hamming = channel.get("channel_optimal_hamming", {}).get("mean")
        if not finite(channel_hamming):
            fail(f"mask artifact post-hoc audit channel hamming is non-finite: {key}")

    doc_text = doc_path.read_text(encoding="utf-8")
    for phrase in [
        "Mask Artifact Post-hoc Matching Audit",
        "Parameter shapes present: `True`",
        "Record-level post-hoc matching supported: `True`",
        "Local channel-permutation matching supported: `True`",
        "Exhaustive graph/channel permutation supported: `False`",
        "Local Channel Permutation",
        "not claim-level CIFAR evidence",
        "full-data permutation rerun remains open",
    ]:
        if phrase not in doc_text:
            fail(f"mask artifact post-hoc audit doc missing phrase: {phrase}")


def require_mode_ticket_artifact_storage_budget() -> None:
    payload = load_json(ROOT / "runs" / "mode_ticket_artifact_storage_budget.json")
    doc_path = ROOT / "docs" / "mode_ticket_artifact_storage_budget.md"
    if payload.get("model") != "resnet20" or payload.get("dataset") != "cifar10":
        fail("mode/ticket artifact storage budget should target CIFAR ResNet-20")
    if int(payload.get("resnet_width", 0)) != 16:
        fail("mode/ticket artifact storage budget should use ResNet width 16")
    if int(payload.get("parameter_count", 0)) != 270896:
        fail("mode/ticket artifact storage budget should cover 270,896 weight parameters")
    fake = payload.get("fake_cifar_fixture", {})
    fake_path = ROOT / str(fake.get("path", ""))
    if not fake_path.exists():
        fail("mode/ticket artifact storage budget references missing fake fixture")
    if int(fake.get("bytes", 0)) <= 200_000:
        fail("mode/ticket artifact storage budget fake fixture is unexpectedly small")
    rows = {
        str(row.get("name")): row
        for row in payload.get("scenarios", [])
        if isinstance(row, dict)
    }
    required = {
        "sgld_activation_aligned_save_states": (220, 220, 297_985_600),
        "csgld_independent_multichain_save_states": (170, 170, 230_261_600),
        "jointdiag_laplace_save_states": (60, 60, 81_268_800),
    }
    for name, (mask_records, state_records, total_bytes) in required.items():
        row = rows.get(name)
        if row is None:
            fail(f"mode/ticket artifact storage budget missing scenario: {name}")
        if int(row.get("mask_record_count_upper_bound", 0)) != mask_records:
            fail(f"mode/ticket artifact storage budget mask records changed: {name}")
        if int(row.get("state_record_count_upper_bound", 0)) != state_records:
            fail(f"mode/ticket artifact storage budget state records changed: {name}")
        if int(row.get("total_bytes_uncompressed", 0)) != total_bytes:
            fail(f"mode/ticket artifact storage budget total bytes changed: {name}")
    recommended = payload.get("recommended_next_rerun", {})
    if recommended.get("scenario") != "sgld_activation_aligned_save_states":
        fail("mode/ticket artifact storage budget recommended wrong scenario")
    command = str(recommended.get("command", ""))
    for phrase in [
        "--save-mask-artifacts --save-state-artifacts",
        "--alignment-method activation",
        "activation_aligned_saved_artifacts_r5_p0p3",
    ]:
        if phrase not in command:
            fail(f"mode/ticket artifact storage budget command missing phrase: {phrase}")
    if float(recommended.get("estimated_total_mib_uncompressed", 0.0)) < 280.0:
        fail("mode/ticket artifact storage budget recommended MiB too small")
    doc_text = doc_path.read_text(encoding="utf-8")
    for phrase in [
        "Mode/Ticket Artifact Storage Budget",
        "Weight parameter count: `270896`",
        "sgld_activation_aligned_save_states",
        "284.18 MiB",
        "--save-mask-artifacts --save-state-artifacts",
    ]:
        if phrase not in doc_text:
            fail(f"mode/ticket artifact storage budget doc missing phrase: {phrase}")


def require_full_data_saved_artifact_posthoc_audit() -> None:
    run_root = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3"
    )
    run_dir = run_root / "20260506_230706"
    metrics_path = run_dir / "metrics.json"
    mask_path = run_dir / "mask_artifacts.npz"
    summary_csv_path = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv"
    )
    summary_doc_path = (
        ROOT
        / "docs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md"
    )
    posthoc_json_path = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json"
    )
    posthoc_doc_path = (
        ROOT
        / "docs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md"
    )
    for path, min_size in [
        (metrics_path, 50_000),
        (mask_path, 100_000_000),
        (summary_csv_path, 1_000),
        (summary_doc_path, 1_000),
        (posthoc_json_path, 10_000),
        (posthoc_doc_path, 1_000),
    ]:
        if not path.exists() or path.stat().st_size < min_size:
            fail(f"full-data saved-artifact output missing or too small: {path}")

    metrics = load_json(metrics_path)
    config = metrics.get("config", {})
    if config.get("dataset") != "cifar10" or config.get("model") != "resnet20":
        fail("full-data saved-artifact run should target CIFAR-10 ResNet-20")
    if config.get("seeds") != [0, 1, 2, 3, 4]:
        fail("full-data saved-artifact run should use seeds 0..4")
    if int(config.get("epochs", 0)) != 30 or int(config.get("imp_rounds", 0)) != 5:
        fail("full-data saved-artifact run should use the long30/r5 protocol")
    if config.get("alignment_method") != "activation":
        fail("full-data saved-artifact run should use activation alignment")
    if config.get("save_mask_artifacts") is not True:
        fail("full-data saved-artifact run should save masks")
    if config.get("save_state_artifacts") is not True:
        fail("full-data saved-artifact run should save states")
    artifact = metrics.get("mask_artifacts", {})
    if int(artifact.get("parameter_count", 0)) != 270896:
        fail("full-data mask artifact should cover 270,896 parameters")
    if artifact.get("save_states") is not True:
        fail("full-data mask artifact should include states")
    collection_counts = {
        str(row.get("name")): (int(row.get("masks", 0)), int(row.get("states", 0)))
        for row in artifact.get("collections", [])
        if isinstance(row, dict)
    }
    expected_counts = {
        "posterior_sample": (50, 50),
        "ticket": (5, 5),
        "activation_aligned_posterior_sample": (50, 50),
        "activation_aligned_ticket": (5, 5),
        "posterior_mode": (1, 1),
        "activation_aligned_posterior_mode": (1, 1),
    }
    for name, expected in expected_counts.items():
        if collection_counts.get(name) != expected:
            fail(f"full-data mask artifact collection count changed: {name}")

    with summary_csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6:
        fail("full-data saved-artifact summary should contain six comparison rows")
    by_comparison = {row["comparison"]: row for row in rows}
    for name in [
        "posterior_samples_vs_tickets",
        "activation_aligned_posterior_samples_vs_tickets",
    ]:
        row = by_comparison.get(name)
        if row is None:
            fail(f"full-data saved-artifact summary missing row: {name}")
        if int(float(row["left_count"])) != 50 or int(float(row["right_count"])) != 5:
            fail(f"full-data saved-artifact row count changed: {name}")
        if float(row["layer_ks_pvalue"]) >= 1e-6:
            fail(f"full-data saved-artifact row should reject layer KS: {name}")
        if float(row["hamming_overlap"]) != 0.0:
            fail(f"full-data saved-artifact row should have zero Hamming overlap: {name}")
        if float(row["logit_cka_hungarian_mean"]) < 0.93:
            fail(f"full-data saved-artifact row CKA unexpectedly low: {name}")
        if str(row["passes_layer_ks"]) != "False" or str(row["passes_hamming_overlap"]) != "False":
            fail(f"full-data saved-artifact row should fail mask-distribution thresholds: {name}")

    posthoc = load_json(posthoc_json_path)
    overall = posthoc.get("overall", {})
    if overall.get("dataset") != "cifar10" or overall.get("model") != "resnet20":
        fail("full-data posthoc audit should target CIFAR ResNet-20")
    if int(overall.get("parameter_count", 0)) != 270896:
        fail("full-data posthoc audit should cover 270,896 parameters")
    if overall.get("record_level_posthoc_matching_supported") is not True:
        fail("full-data posthoc audit should support record-level matching")
    if overall.get("local_channel_permutation_matching_supported") is not False:
        fail("full-data posthoc audit should not claim local channel matching when capped")
    if int(overall.get("channel_permutation_skipped_count", 0)) != 7:
        fail("full-data posthoc audit should mark capped channel searches")
    if int(overall.get("max_channel_pair_count", 0)) != 1:
        fail("full-data posthoc audit should record the channel-pair cap")
    comparisons = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in posthoc.get("comparisons", [])
        if isinstance(row, dict)
    }
    key = ("posterior_sample", "ticket")
    row = comparisons.get(key)
    if row is None:
        fail("full-data posthoc audit missing posterior_sample vs ticket")
    natural = float(row["natural_hamming"]["mean"])
    optimal = float(row["optimal_hamming"]["mean"])
    if not (0.20 < optimal < natural < 0.26):
        fail("full-data posthoc posterior-ticket hamming changed")
    skipped = posthoc.get("skipped_comparisons", [])
    if len(skipped) != 1 or int(skipped[0].get("pair_count", 0)) != 2500:
        fail("full-data posthoc should skip the 50x50 raw-vs-aligned comparison")

    doc_text = posthoc_doc_path.read_text(encoding="utf-8")
    for phrase in [
        "full-data CIFAR saved-artifact evidence",
        "full-data saved-artifact rerun is complete",
        "Local channel-permutation matching was not run",
        "record-level evidence plus a saved artifact",
    ]:
        if phrase not in doc_text:
            fail(f"full-data posthoc doc missing phrase: {phrase}")
    summary_text = summary_doc_path.read_text(encoding="utf-8")
    for phrase in [
        "Posterior-to-chain-start Hamming mean: 0.0511",
        "posterior_samples_vs_tickets",
        "activation_aligned_posterior_samples_vs_tickets",
    ]:
        if phrase not in summary_text:
            fail(f"full-data saved-artifact summary doc missing phrase: {phrase}")


def require_full_data_global_channel_permutation_audit() -> None:
    json_path = (
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json"
    )
    doc_path = (
        ROOT
        / "docs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md"
    )
    if not json_path.exists() or json_path.stat().st_size < 20_000:
        fail("full-data global channel audit JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 1_000:
        fail("full-data global channel audit doc missing or too small")
    payload = load_json(json_path)
    overall = payload.get("overall", {})
    if overall.get("dataset") != "cifar10" or overall.get("model") != "resnet20":
        fail("full-data global channel audit should target CIFAR ResNet-20")
    if int(overall.get("parameter_count", 0)) != 270896:
        fail("full-data global channel audit should cover 270,896 parameters")
    if int(overall.get("resnet_channel_key_count", 0)) != 19:
        fail("full-data global channel audit should cover 19 channel keys")
    if overall.get("objective") != "mask":
        fail("full-data global channel audit should optimize mask objective")
    if int(overall.get("max_iters", 0)) != 6:
        fail("full-data global channel audit iteration budget changed")
    if int(overall.get("comparison_count", 0)) != 6:
        fail("full-data global channel audit should include six comparisons")
    if overall.get("global_channel_coordinate_descent_supported") is not True:
        fail("full-data global channel audit should support coordinate descent")
    if overall.get("exhaustive_graph_channel_permutation_supported") is not False:
        fail("full-data global channel audit must not claim exhaustive search")

    rows = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in payload.get("comparisons", [])
        if isinstance(row, dict)
    }
    required = [
        ("posterior_sample", "ticket"),
        ("activation_aligned_posterior_sample", "activation_aligned_ticket"),
        ("posterior_mode", "ticket"),
        ("activation_aligned_chain_start", "activation_aligned_ticket"),
    ]
    for key in required:
        if key not in rows:
            fail(f"full-data global channel audit missing comparison: {key}")
    raw = rows[("posterior_sample", "ticket")]
    raw_hamming = float(raw["raw_record_hamming"]["mean"])
    global_hamming = float(raw["global_channel_hamming"]["mean"])
    global_overlap = float(raw["global_channel_support_overlap_min"]["mean"])
    if not (0.20 < global_hamming < raw_hamming < 0.22):
        fail("raw posterior/ticket global channel Hamming changed")
    if global_overlap >= 0.40:
        fail("raw posterior/ticket global channel overlap unexpectedly high")
    aligned = rows[
        ("activation_aligned_posterior_sample", "activation_aligned_ticket")
    ]
    aligned_global = float(aligned["global_channel_hamming"]["mean"])
    aligned_improvement = float(aligned["hamming_improvement"]["mean"])
    if not (0.20 < aligned_global < 0.22):
        fail("aligned posterior/ticket global channel Hamming changed")
    if aligned_improvement <= 0.03:
        fail("aligned posterior/ticket channel optimization should reduce frame mismatch")
    mode = rows[("posterior_mode", "ticket")]
    if float(mode["global_channel_hamming"]["mean"]) <= 0.20:
        fail("posterior mode global channel Hamming should remain non-ticket-like")

    doc_text = doc_path.read_text(encoding="utf-8")
    for phrase in [
        "Full-data Global Channel Permutation Audit",
        "block-coordinate descent with exact per-key Hungarian updates",
        "Global-channel Hamming",
        "0.2105",
        "channel relabeling does not rescue",
        "not an exhaustive graph-isomorphism proof",
    ]:
        if phrase not in doc_text:
            fail(f"full-data global channel audit doc missing phrase: {phrase}")


def require_exhaustive_channel_permutation_feasibility_audit() -> None:
    json_path = ROOT / "runs" / "resnet_channel_permutation_exhaustive_feasibility_audit.json"
    doc_path = ROOT / "docs" / "resnet_channel_permutation_exhaustive_feasibility_audit.md"
    if not json_path.exists() or json_path.stat().st_size < 5_000:
        fail("exhaustive channel permutation feasibility JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 1_000:
        fail("exhaustive channel permutation feasibility doc missing or too small")
    payload = load_json(json_path)
    overall = payload.get("overall", {})
    if overall.get("stage1_exact_enumeration_supported") is not True:
        fail("exhaustive feasibility audit should support stage-1 exact enumeration")
    if int(overall.get("stage1_parameter_count", 0)) != 270:
        fail("exhaustive feasibility audit stage-1 parameter count changed")
    if int(overall.get("stage1_exact_permutation_count", 0)) != 128:
        fail("exhaustive feasibility audit should enumerate 128 stage-1 assignments")
    if overall.get("stage1_coordinate_descent_all_exact") is not True:
        fail("stage-1 coordinate-descent audit should match exact enumeration")
    if overall.get("full_exhaustive_channel_permutation_supported") is not False:
        fail("exhaustive feasibility audit should not claim full-data exhaustive support")
    full_log10 = float(overall.get("full_log10_permutation_count", 0.0))
    if not (800.0 < full_log10 < 900.0):
        fail("full-data channel permutation search-space estimate changed")
    search_spaces = payload.get("search_spaces", {})
    stage1 = search_spaces.get("fake_stage1_subgraph", {})
    full = search_spaces.get("full_cifar_artifact", {})
    if int(stage1.get("channel_key_count", 0)) != 7:
        fail("stage-1 exact subgraph should contain seven channel keys")
    if int(full.get("channel_key_count", 0)) != 19:
        fail("full CIFAR search-space estimate should contain 19 channel keys")
    if full.get("channel_count_histogram") != {"16": 7, "32": 6, "64": 6}:
        fail("full CIFAR channel-count histogram changed")
    rows = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in payload.get("exact_stage1_comparisons", [])
        if isinstance(row, dict)
    }
    ticket = rows.get(("ticket", "activation_aligned_ticket"))
    chain = rows.get(("chain_start", "activation_aligned_chain_start"))
    posterior = rows.get(("posterior_sample", "ticket"))
    aligned = rows.get(
        ("activation_aligned_posterior_sample", "activation_aligned_ticket")
    )
    if ticket is None or chain is None or posterior is None or aligned is None:
        fail("exhaustive feasibility audit missing expected stage-1 comparisons")
    for row in [ticket, chain, posterior, aligned]:
        if row.get("coordinate_descent_matches_exact") is not True:
            fail("coordinate descent should match exact enumeration on stage-1 audit")
    if float(ticket["exact_global_hamming"]["mean"]) != 0.0:
        fail("ticket raw-vs-aligned stage-1 exact Hamming should be zero")
    if float(chain["exact_global_hamming"]["mean"]) != 0.0:
        fail("chain-start raw-vs-aligned stage-1 exact Hamming should be zero")
    if float(posterior["exact_global_hamming"]["mean"]) >= 0.02:
        fail("fake stage-1 posterior-ticket exact Hamming unexpectedly high")
    if float(aligned["exact_improvement"]["mean"]) <= 0.04:
        fail("fake stage-1 aligned posterior/ticket exact audit should remove frame mismatch")
    doc_text = doc_path.read_text(encoding="utf-8")
    for phrase in [
        "Exhaustive Channel Permutation Feasibility Audit",
        "fake stage-1 subgraph",
        "Stage-1 exact assignments: `128`",
        "Coordinate descent matches exact enumeration: `True`",
        "full CIFAR artifact",
        "10^840.4",
        "infeasible and unimplemented",
    ]:
        if phrase not in doc_text:
            fail(f"exhaustive feasibility audit doc missing phrase: {phrase}")


def require_calibration_and_learned_masks(stats: dict[str, Any]) -> None:
    sources = {row["source"] for row in stats["calibration_ood"]}
    required_sources = {
        "dense",
        "imp",
        "swag_ensemble",
        "learned_random_0",
        "gem_miner",
        "variational_prune",
    }
    missing = sorted(required_sources - sources)
    if missing:
        fail(f"calibration/OOD summary missing sources: {missing}")
    for row in stats["calibration_ood"]:
        if row["source"] in required_sources:
            if int(row["id_accuracy"]["n"]) != 5:
                fail(f"calibration/OOD row is not five-seed: {row['source']}")
            if not finite(row["ood_msp_auroc"]["mean"]):
                fail(f"calibration/OOD row has non-finite OOD AUROC: {row['source']}")
    variational = rows_matching(stats["variational_pruning"], source="variational_prune")
    if not variational:
        fail("digits variational pruning summary missing variational_prune row")
    if float(variational[0]["accuracy_minus_imp"]["mean"]) >= 0.0:
        fail("digits variational pruning should not beat IMP in current evidence")
    learned_support = stats["trajectory_mask_training"]
    for source in ["gem_miner", "variational_prune", "hard_concrete"]:
        rows = rows_matching(learned_support, source=source)
        if not rows:
            fail(f"CIFAR learned-mask support row missing: {source}")
        row = rows[0]
        if int(row["trained_accuracy"]["n"]) != 5:
            fail(f"CIFAR learned-mask support row is not five-seed: {source}")
        if float(row["accuracy_minus_imp"]["mean"]) >= 0.0:
            fail(f"CIFAR learned-mask row should remain below IMP: {source}")
        if float(row["source_to_imp"]["mean"]) >= 0.10:
            fail(f"CIFAR learned-mask support should stay random-scale: {source}")
    hard = rows_matching(learned_support, source="hard_concrete")[0]
    if float(hard["trained_accuracy"]["mean"]) >= 0.50:
        fail("CIFAR hard-concrete full-data row should not be competitive")


def require_residual_process(stats: dict[str, Any]) -> None:
    process_rows = stats["residual_imp_process_round_exclusion"]
    final_rows = [
        row
        for row in process_rows
        if row.get("variant") == "round_excluded_oracle_final_imp_residual"
    ]
    if not final_rows:
        fail("round-exclusion process intervention rows are missing")
    if not all(int(row["trained_accuracy"]["n"]) == 5 for row in final_rows):
        fail("round-exclusion process rows must be five-seed summaries")
    layer_rows = stats["residual_imp_process_layer_exclusion_pairs"]
    layer_rms_r5 = [
        row
        for row in layer_rows
        if row.get("base_source") == "traj_rms_abs"
        and int(row.get("process_round")) == 5
        and abs(float(row.get("alpha")) - 0.5) < 1e-12
    ]
    if not layer_rms_r5:
        fail("tensor-matched round-exclusion RMS round-5 pair row is missing")
    row = layer_rms_r5[0]
    if int(row["round_minus_layer_excluded"]["n"]) != 5:
        fail("tensor-matched round-exclusion row must be five-seed")
    if float(row["round_minus_layer_excluded"]["mean"]) <= 0.0:
        fail("round-selected process residual should beat tensor-matched replacement")
    if int(row["round_minus_layer_excluded"]["positive"]) < 4:
        fail("tensor-matched round-exclusion paired wins are unexpectedly weak")
    tensor_score_rows = stats["residual_imp_process_tensor_score_exclusion_pairs"]
    tensor_score_rms_r5 = [
        row
        for row in tensor_score_rows
        if row.get("base_source") == "traj_rms_abs"
        and int(row.get("process_round")) == 5
        and abs(float(row.get("alpha")) - 0.5) < 1e-12
    ]
    if not tensor_score_rms_r5:
        fail("tensor+score-matched round-exclusion RMS round-5 pair row is missing")
    row = tensor_score_rms_r5[0]
    if int(row["round_minus_tensor_score_excluded"]["n"]) != 5:
        fail("tensor+score-matched round-exclusion row must be five-seed")
    if float(row["round_minus_tensor_score_excluded"]["mean"]) <= 0.0:
        fail("round-selected process residual should beat tensor+score-matched replacement")
    if int(row["round_minus_tensor_score_excluded"]["positive"]) < 4:
        fail("tensor+score-matched round-exclusion paired wins are unexpectedly weak")
    if (
        float(row["tensor_score_excluded_oracle_overlap"]["mean"])
        <= float(row["layer_excluded_oracle_overlap"]["mean"])
    ):
        fail("tensor+score replacement should improve oracle overlap over tensor-only replacement")
    projection_rows = stats["residual_imp_process_projection_pairs"]
    projection_rms_r5 = [
        row
        for row in projection_rows
        if row.get("base_source") == "traj_rms_abs"
        and int(row.get("process_round")) == 5
        and abs(float(row.get("alpha")) - 0.5) < 1e-12
    ]
    if not projection_rms_r5:
        fail("residualized-score projection RMS round-5 pair row is missing")
    row = projection_rms_r5[0]
    if int(row["round_minus_residualized"]["n"]) != 5:
        fail("residualized-score projection row must be five-seed")
    if float(row["round_minus_residualized"]["mean"]) <= 0.0:
        fail("round-selected process score should beat residualized score")
    if int(row["round_minus_residualized"]["positive"]) < 4:
        fail("residualized-score projection paired wins are unexpectedly weak")
    if float(row["round_minus_residualized_oracle"]["mean"]) <= 0.10:
        fail("residualized projection should materially lower oracle overlap")
    posterior_projection_rows = stats[
        "residual_imp_process_posterior_projection_pairs"
    ]
    posterior_projection_rms_r5 = [
        row
        for row in posterior_projection_rows
        if row.get("base_source") == "traj_rms_abs"
        and int(row.get("process_round")) == 5
        and abs(float(row.get("alpha")) - 0.5) < 1e-12
    ]
    if not posterior_projection_rms_r5:
        fail("posterior-residualized projection RMS round-5 pair row is missing")
    row = posterior_projection_rms_r5[0]
    if int(row["round_minus_residualized"]["n"]) != 5:
        fail("posterior-residualized projection row must be five-seed")
    if float(row["round_minus_residualized"]["mean"]) <= 0.0:
        fail("round-selected process score should beat posterior-residualized score")
    if int(row["round_minus_residualized"]["positive"]) != 5:
        fail("posterior-residualized projection should have 5/5 paired accuracy wins")
    if float(row["round_minus_residualized_oracle"]["mean"]) <= 0.18:
        fail("posterior-residualized projection should materially lower oracle overlap")
    learned_subspace_rows = stats[
        "residual_imp_process_learned_subspace_pairs"
    ]
    learned_subspace_rms_r5 = [
        row
        for row in learned_subspace_rows
        if row.get("base_source") == "traj_rms_abs"
        and int(row.get("process_round")) == 5
        and abs(float(row.get("alpha")) - 0.5) < 1e-12
    ]
    if not learned_subspace_rms_r5:
        fail("learned-subspace residualized projection RMS round-5 pair row is missing")
    row = learned_subspace_rms_r5[0]
    if int(row["round_minus_residualized"]["n"]) != 5:
        fail("learned-subspace residualized projection row must be five-seed")
    if float(row["round_minus_residualized"]["mean"]) <= 0.0:
        fail("round-selected process score should beat learned-subspace residualized score")
    if int(row["round_minus_residualized"]["positive"]) != 5:
        fail("learned-subspace projection should have 5/5 paired accuracy wins")
    if float(row["round_minus_residualized_oracle"]["mean"]) <= 0.18:
        fail("learned-subspace projection should materially lower oracle overlap")


def require_text_evidence() -> None:
    main_tex = " ".join(
        (ROOT / "paper" / "main.tex").read_text(encoding="utf-8").split()
    )
    submission_audit = " ".join(
        (ROOT / "docs" / "submission_readiness_audit.md").read_text(encoding="utf-8").split()
    )
    consistency_targets = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/next_experiment_protocol.md": (
            ROOT / "docs" / "next_experiment_protocol.md"
        ).read_text(encoding="utf-8"),
        "docs/negative_result_paper_plan.md": (
            ROOT / "docs" / "negative_result_paper_plan.md"
        ).read_text(encoding="utf-8"),
        "docs/experiment_log.md": (ROOT / "docs" / "experiment_log.md").read_text(
            encoding="utf-8"
        ),
    }
    expected_phrases = [
        "posterior supports beat uniform random masks in 58 of",
        "raw-parameter basin entropy is 0.0 nats",
        "full-data CIFAR-10 ResNet-20 strengthens",
        "activation-channel alignment are insufficient",
        "weight-correlation channel alignment",
        "block-coordinate ResNet channel audit",
        "exact stage-1 enumeration",
        "75 cyclical-SGLD posterior samples",
        "rank-16, rank-32, rank-64, and rank-128 top-Hessian",
        "selected and tensor-block-diagonal full-covariance Laplace",
        "22,064-parameter block-diagonal",
        "68,144-parameter block-diagonal",
        "68,144-parameter joint-group",
        "86,576-parameter joint-group",
        "270,896-parameter joint-group",
        "full-network exact/full-covariance Laplace",
        "exact dense full-network Laplace sanity",
        "linear connectivity audit",
        "orthogonal landscape diagnostics",
        "553.1 GiB",
        "tensor matched excluded oracle",
        "tensor+score matched excluded oracle",
        "residualized round-score projection",
        "posterior-residualized row reaches",
        "diagonal-Laplace posterior score subspace",
        "learned-subspace residualized row reaches",
        "learned trajectory/process subspace",
        "Representative five-seed CIFAR-10 ResNet-20 epoch-1 rewind movement",
        "Generated Evidence Tables",
        "support-equivalence claim",
        "Scope of the claim",
    ]
    for phrase in expected_phrases:
        if phrase not in main_tex:
            fail(f"paper/main.tex missing expected evidence phrase: {phrase}")
    expected_audit_phrases = [
        "posterior supports beat random masks in 58/59 groups",
        "0/59 groups",
        "55/57 groups",
        "20-snapshot full-network SWAG movement row",
        "block-diagonal",
        "tensor-matched round-exclusion",
        "tensor+score-matched round-exclusion",
        "residualized-score projection",
        "posterior-residualized",
        "learned-subspace residualized",
        "scoped support-equivalence claim",
        "hard-concrete L0 gate baseline",
        "post-hoc exhaustive graph/permutation realignment",
        "structured global channel audit",
        "exact stage-1 enumeration",
        "exact dense full-network Laplace sanity",
        "linear connectivity barrier audit",
    ]
    for phrase in expected_audit_phrases:
        if phrase not in submission_audit:
            fail(f"submission readiness audit missing expected evidence phrase: {phrase}")
    stale_audit_phrases = [
        "41/42",
        "38/40",
        "posterior beats random in 44/45",
        "posterior supports beat random in 44/45",
        "44/45 grouped support-overlap comparisons",
        "0/45 comparisons",
        "33/45",
        "12/45",
        "41/43 grouped comparisons",
        "48/49",
        "0/49",
        "45/47",
        "52/53",
        "0/53",
        "49/51",
        "52 of 53",
        "49 of 51",
        "53/54",
        "0/54",
        "50/52",
        "54/58",
        "57/58",
        "0/58",
        "54/56",
        "53 of 54",
        "50 of 52",
    ]
    for phrase in stale_audit_phrases:
        if phrase in submission_audit:
            fail(f"submission readiness audit contains stale count: {phrase}")
    for doc_path, text in consistency_targets.items():
        for phrase in stale_audit_phrases:
            if phrase in text:
                fail(f"{doc_path} contains stale mode-distribution count: {phrase}")


def require_bibliography() -> None:
    main_tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    refs = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    cited_keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]*)\}", main_tex):
        cited_keys.update(
            key.strip()
            for key in match.group(1).split(",")
            if key.strip()
        )
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", refs, flags=re.MULTILINE))
    missing = sorted(cited_keys - bib_keys)
    if missing:
        fail(f"paper/main.tex cites keys missing from paper/refs.bib: {missing}")
    expected_phrases = [
        "@inproceedings{paul2023unmasking",
        "Paul, Mansheej and Chen, Feng and Larsen, Brett W.",
        "@inproceedings{sakamoto2022pacbayes",
        "Sakamoto, Keitaro and Sato, Issei",
        "Analyzing Lottery Ticket Hypothesis from PAC-Bayesian Theory Perspective",
        "@misc{kuhn2026bayesian",
        "Kuhn, Nicholas and Weyrauch, Arvid",
        "2602.18825",
    ]
    for phrase in expected_phrases:
        if phrase not in refs:
            fail(f"paper/refs.bib missing expected citation metadata: {phrase}")


def tex_sci(value: float) -> str:
    if value == 0.0:
        return "0.0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.1f}{{\\times}}10^{{{exponent}}}"


def require_paper_numeric_claims(stats: dict[str, Any]) -> None:
    main_tex = " ".join(
        (ROOT / "paper" / "main.tex").read_text(encoding="utf-8").split()
    )

    def require_phrase(phrase: str) -> None:
        if phrase not in main_tex:
            fail(f"paper/main.tex missing generated numeric claim: {phrase}")

    random_rows = [
        row
        for row in stats["mode_distribution_equivalence"]
        if row.get("comparison") == "posterior-random"
    ]
    random_wins = sum(
        str(row.get("verdict")) == "posterior separates from random"
        for row in random_rows
    )
    require_phrase(
        "posterior supports beat uniform random masks in "
        f"{random_wins} of {len(random_rows)} grouped comparisons"
    )

    chain_rows = [
        row
        for row in stats["mode_distribution_equivalence"]
        if row.get("comparison") == "posterior-chain"
    ]
    chain_tied = sum(
        str(row.get("verdict")) == "practically tied to control"
        for row in chain_rows
    )
    chain_control = sum(
        str(row.get("verdict")) == "control closer to ticket"
        for row in chain_rows
    )
    chain_mixed = sum(str(row.get("verdict")) == "mixed" for row in chain_rows)
    mixed_text = "one" if chain_mixed == 1 else str(chain_mixed)
    require_phrase(
        f"none of the {len(chain_rows)} posterior-versus-chain-start comparisons"
    )
    require_phrase(
        f"{chain_tied} are practically tied to the control, "
        f"{chain_control} favor chain-start magnitude, and {mixed_text} is mixed"
    )

    rewind_rows = [
        row
        for row in stats["mode_distribution_equivalence"]
        if row.get("comparison") == "posterior-rewind"
    ]
    rewind_control = sum(
        str(row.get("verdict")) == "control closer to ticket"
        for row in rewind_rows
    )
    require_phrase(
        f"support in {rewind_control} of {len(rewind_rows)} grouped comparisons"
    )

    block_rows = rows_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:11blocks<=5000",
        scale=0.0001,
    )
    if not block_rows:
        fail("BlockDiagLap row unavailable for paper numeric claim checks")
    block = block_rows[0]
    parameter_count = int(round(float(block["parameter_count"]["mean"])))
    sample_accuracy = float(block["sample_accuracy"]["mean"])
    post_chain = float(block["global_post_chain"]["mean"])
    block_delta = abs(float(block["block_posterior_minus_chain"]["mean"]))
    global_delta = float(block["global_posterior_minus_chain"]["mean"])
    rewind_delta = float(block["global_rewind_minus_posterior"]["mean"])
    require_phrase(f"covering {parameter_count:,} trainable weights")
    require_phrase(f"sample accuracy remains {100.0 * sample_accuracy:.1f}\\%")
    require_phrase(f"falls to {post_chain:.4f}")
    require_phrase(f"by {block_delta:.4f} on average")
    require_phrase(f"posterior-minus-chain gain is only {global_delta:.4f}")
    require_phrase(f"by {rewind_delta:.4f}")
    require_phrase(f"this {parameter_count:,}-parameter block-diagonal")

    wider_block_rows = rows_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:16blocks<=10000",
        scale=1e-05,
    )
    if not wider_block_rows:
        fail("BlockDiagLap max10k row unavailable for paper numeric claim checks")
    wider_block = wider_block_rows[0]
    wider_count = int(round(float(wider_block["parameter_count"]["mean"])))
    wider_sample_accuracy = float(wider_block["sample_accuracy"]["mean"])
    wider_post_chain = float(wider_block["global_post_chain"]["mean"])
    wider_block_delta = float(wider_block["block_posterior_minus_chain"]["mean"])
    wider_global_delta = float(wider_block["global_posterior_minus_chain"]["mean"])
    wider_rewind_delta = float(wider_block["global_rewind_minus_posterior"]["mean"])
    require_phrase(f"covering {wider_count:,} trainable weights")
    require_phrase(f"sample accuracy remains {100.0 * wider_sample_accuracy:.1f}\\%")
    require_phrase(f"falls further to {wider_post_chain:.4f}")
    require_phrase(f"remains negative at {wider_block_delta:.4f}")
    require_phrase(f"global posterior-minus-chain gain shrinks to {wider_global_delta:.4f}")
    require_phrase(f"rewind remains closer by {wider_rewind_delta:.4f}")
    require_phrase(f"this {wider_count:,}-parameter block-diagonal")

    jointdiag_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=10000",
        scale=1e-05,
    )
    if not jointdiag_rows:
        fail("JointDiagLap max10k row unavailable for paper numeric claim checks")
    jointdiag = jointdiag_rows[0]
    jointdiag_count = int(round(float(jointdiag["parameter_count"]["mean"])))
    jointdiag_sample_accuracy = float(jointdiag["sample_accuracy"]["mean"])
    jointdiag_post_chain = float(jointdiag["global_post_chain"]["mean"])
    jointdiag_block_delta = float(jointdiag["block_posterior_minus_chain"]["mean"])
    jointdiag_global_delta = float(jointdiag["global_posterior_minus_chain"]["mean"])
    jointdiag_rewind_delta = float(jointdiag["global_rewind_minus_posterior"]["mean"])
    require_phrase(f"keeps the same {jointdiag_count:,} trainable weights")
    require_phrase(f"sample accuracy remains {100.0 * jointdiag_sample_accuracy:.1f}\\%")
    require_phrase(f"global posterior-to-chain-start support falls to {jointdiag_post_chain:.4f}")
    require_phrase(f"block posterior-minus-chain remains {jointdiag_block_delta:.4f}")
    require_phrase(f"global posterior-minus-chain is {jointdiag_global_delta:.4f}")
    require_phrase(f"rewind remains closer by {jointdiag_rewind_delta:.4f}")
    require_phrase(f"this {jointdiag_count:,}-parameter joint-group")

    jointdiag20_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:6groups<=20000",
        scale=3e-06,
    )
    if not jointdiag20_rows:
        fail("JointDiagLap max20k row unavailable for paper numeric claim checks")
    jointdiag20 = jointdiag20_rows[0]
    jointdiag20_count = int(round(float(jointdiag20["parameter_count"]["mean"])))
    jointdiag20_sample_accuracy = float(jointdiag20["sample_accuracy"]["mean"])
    jointdiag20_post_chain = float(jointdiag20["global_post_chain"]["mean"])
    jointdiag20_block_delta = float(jointdiag20["block_posterior_minus_chain"]["mean"])
    jointdiag20_global_delta = float(jointdiag20["global_posterior_minus_chain"]["mean"])
    jointdiag20_rewind_delta = float(jointdiag20["global_rewind_minus_posterior"]["mean"])
    require_phrase(f"covering {jointdiag20_count:,} trainable weights")
    require_phrase(f"sample accuracy remains {100.0 * jointdiag20_sample_accuracy:.1f}\\%")
    require_phrase(
        f"global posterior-to-chain-start support is {jointdiag20_post_chain:.4f}"
    )
    require_phrase(
        f"block posterior-minus-chain stays negative at {jointdiag20_block_delta:.4f}"
    )
    require_phrase(f"global posterior-minus-chain stays {jointdiag20_global_delta:.4f}")
    require_phrase(f"rewind remains closer by {jointdiag20_rewind_delta:.4f}")
    require_phrase(f"this {jointdiag20_count:,}-parameter joint-group")

    jointdiag40_rows = rows_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=40000",
        scale=1e-06,
    )
    if not jointdiag40_rows:
        fail("JointDiagLap max40k row unavailable for paper numeric claim checks")
    jointdiag40 = jointdiag40_rows[0]
    jointdiag40_count = int(round(float(jointdiag40["parameter_count"]["mean"])))
    jointdiag40_sample_accuracy = float(jointdiag40["sample_accuracy"]["mean"])
    jointdiag40_post_chain = float(jointdiag40["global_post_chain"]["mean"])
    jointdiag40_block_delta = float(jointdiag40["block_posterior_minus_chain"]["mean"])
    jointdiag40_global_delta = float(jointdiag40["global_posterior_minus_chain"]["mean"])
    jointdiag40_rewind_delta = float(jointdiag40["global_rewind_minus_posterior"]["mean"])
    require_phrase(f"covers all {jointdiag40_count:,} weight parameters")
    require_phrase(f"sample accuracy remains {100.0 * jointdiag40_sample_accuracy:.1f}\\%")
    require_phrase(
        f"global posterior-to-chain-start support is {jointdiag40_post_chain:.4f}"
    )
    require_phrase(
        f"block posterior-minus-chain remains negative at {jointdiag40_block_delta:.4f}"
    )
    require_phrase(f"global posterior-minus-chain is {jointdiag40_global_delta:.4f}")
    require_phrase(f"rewind remains closer by {jointdiag40_rewind_delta:.4f}")
    require_phrase(f"this {jointdiag40_count:,}-parameter joint-group")

    direct_rows = rows_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full LowRank128Lap",
        comparison="posterior_samples_vs_tickets",
    )
    if not direct_rows:
        fail("LowRank128Lap direct mode/ticket row unavailable for paper checks")
    direct = direct_rows[0]
    sample_count = int(round(float(direct["left_count"])))
    require_phrase(
        f"{sample_count} posterior samples now pass the Hamming-overlap "
        f"threshold ({float(direct['hamming_overlap']):.3f})"
    )
    require_phrase(
        "logit/final-hidden activation CKA "
        f"({float(direct['logit_cka_hungarian_mean']):.3f}/"
        f"{float(direct['activation_cka_hungarian_mean']):.3f})"
    )
    require_phrase(
        "layer-sparsity KS threshold "
        f"($p={tex_sci(float(direct['layer_ks_pvalue']))}$)"
    )
    require_phrase(
        f"all {sample_count} samples still collapse to one parameter-PCA basin"
    )

    jointdiag_direct_rows = rows_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full JointDiagLap270k",
        comparison="posterior_samples_vs_tickets",
    )
    if not jointdiag_direct_rows:
        fail("JointDiagLap270k direct mode/ticket row unavailable for paper checks")
    jointdiag_direct = jointdiag_direct_rows[0]
    jointdiag_sample_count = int(round(float(jointdiag_direct["left_count"])))
    jointdiag_metrics_path = ROOT / str(jointdiag_direct["run"]) / "metrics.json"
    if not jointdiag_metrics_path.exists():
        fail("JointDiagLap270k direct metrics unavailable for paper checks")
    jointdiag_payload = load_json(jointdiag_metrics_path)
    jointdiag_diagnostics = jointdiag_payload.get("posterior_chain_diagnostics", {})
    jointdiag_sample_accuracy = float(
        jointdiag_diagnostics["posterior_sample_accuracy_mean"]
    )
    jointdiag_hamming = float(
        jointdiag_diagnostics["posterior_to_chain_start_hamming_mean"]
    )
    require_phrase(
        "A streamed exact joint-group direct probe over the same all "
        f"{jointdiag40_count:,} weight parameters"
    )
    require_phrase(f"{jointdiag_sample_count} joint-group posterior samples")
    require_phrase(f"sample accuracy remains {100.0 * jointdiag_sample_accuracy:.1f}\\%")
    require_phrase(f"posterior-to-chain-start Hamming is {jointdiag_hamming:.4f}")
    require_phrase(
        "layer-sparsity KS is "
        f"$p={tex_sci(float(jointdiag_direct['layer_ks_pvalue']))}$"
    )
    require_phrase(
        f"Hamming overlap is {float(jointdiag_direct['hamming_overlap']):.3f}"
    )
    require_phrase(
        "logit/final-hidden activation CKA remains "
        f"{float(jointdiag_direct['logit_cka_hungarian_mean']):.3f}/"
        f"{float(jointdiag_direct['activation_cka_hungarian_mean']):.3f}"
    )
    require_phrase(
        f"All {jointdiag_sample_count} joint-group samples still collapse to one "
        "parameter-PCA basin"
    )

    posterior_projection = rows_matching(
        stats["residual_imp_process_posterior_projection_pairs"],
        base_source="traj_rms_abs",
        process_round=5,
        alpha=0.5,
    )
    if not posterior_projection:
        fail("posterior-residualized projection row unavailable for paper checks")
    projection = posterior_projection[0]
    round_accuracy = float(projection["round_accuracy"]["mean"])
    residualized_accuracy = float(projection["residualized_accuracy"]["mean"])
    delta = projection["round_minus_residualized"]
    oracle_delta = projection["round_minus_residualized_oracle"]
    round_oracle = float(projection["round_oracle_overlap"]["mean"])
    residualized_oracle = float(projection["residualized_oracle_overlap"]["mean"])
    require_phrase(f"{100.0 * round_accuracy:.2f}\\% accuracy")
    require_phrase(f"posterior-residualized row reaches {100.0 * residualized_accuracy:.2f}\\%")
    require_phrase(
        f"{100.0 * float(delta['mean']):.2f} percentage points, "
        f"{int(delta['positive'])}/{int(delta['n'])} positive seeds"
    )
    require_phrase(
        f"95\\% CI $[{100.0 * float(delta['ci95_low']):.2f},"
        f"{100.0 * float(delta['ci95_high']):.2f}]$"
    )
    require_phrase(
        f"overlap drops from {round_oracle:.4f} to {residualized_oracle:.4f}"
    )
    require_phrase(
        f"a {float(oracle_delta['mean']):.4f} paired drop with "
        f"{int(oracle_delta['positive'])}/{int(oracle_delta['n'])} positive seeds"
    )

    learned_subspace = rows_matching(
        stats["residual_imp_process_learned_subspace_pairs"],
        base_source="traj_rms_abs",
        process_round=5,
        alpha=0.5,
    )
    if not learned_subspace:
        fail("learned-subspace residualized projection row unavailable for paper checks")
    learned = learned_subspace[0]
    learned_round_accuracy = float(learned["round_accuracy"]["mean"])
    learned_accuracy = float(learned["residualized_accuracy"]["mean"])
    learned_delta = learned["round_minus_residualized"]
    learned_oracle_delta = learned["round_minus_residualized_oracle"]
    learned_round_oracle = float(learned["round_oracle_overlap"]["mean"])
    learned_oracle = float(learned["residualized_oracle_overlap"]["mean"])
    require_phrase(f"{100.0 * learned_round_accuracy:.2f}\\% accuracy")
    require_phrase(
        f"learned-subspace residualized row reaches {100.0 * learned_accuracy:.2f}\\%"
    )
    require_phrase(
        f"{100.0 * float(learned_delta['mean']):.2f} percentage points, "
        f"{int(learned_delta['positive'])}/{int(learned_delta['n'])} positive seeds"
    )
    require_phrase(
        f"95\\% CI $[{100.0 * float(learned_delta['ci95_low']):.2f},"
        f"{100.0 * float(learned_delta['ci95_high']):.2f}]$"
    )
    require_phrase(
        f"overlap falls from {learned_round_oracle:.4f} to {learned_oracle:.4f}"
    )
    require_phrase(
        f"a {float(learned_oracle_delta['mean']):.4f} learned-subspace paired drop"
    )


def require_environment_lock() -> None:
    lock = load_json(ROOT / "docs" / "environment_lock.json")
    expected_packages = {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "scikit-learn": "1.4.1.post1",
        "matplotlib": "3.6.3",
    }
    packages = lock.get("packages", {})
    for name, version in expected_packages.items():
        if packages.get(name) != version:
            fail(f"environment lock missing {name}=={version}")
    lock_text = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    for name, version in expected_packages.items():
        if f"{name}=={version}" not in lock_text:
            fail(f"requirements-lock.txt missing {name}=={version}")
    expected_gpu_packages = {
        name: version
        for name, version in expected_packages.items()
        if name != "matplotlib"
    }
    gpu_lock_text = (ROOT / "requirements-gpu-lock.txt").read_text(encoding="utf-8")
    for name, version in expected_gpu_packages.items():
        if f"{name}=={version}" not in gpu_lock_text:
            fail(f"requirements-gpu-lock.txt missing {name}=={version}")
    if "matplotlib==" in gpu_lock_text:
        fail("requirements-gpu-lock.txt should exclude plotting-only matplotlib")


def require_full_covariance_feasibility() -> None:
    payload = load_json(
        ROOT / "runs" / "cifar10_resnet20_full_covariance_feasibility.json"
    )
    all_trainable = payload.get("all_trainable", {})
    weight_only = payload.get("weight_only", {})
    weight_blocks = payload.get("weight_tensor_block_diagonal", {})
    largest = payload.get("largest_parameter_tensors", [])
    if int(all_trainable.get("parameter_count", 0)) < 270_000:
        fail("full-covariance feasibility audit has too few trainable parameters")
    if float(all_trainable.get("dense_precision_float64_gib", 0.0)) < 500.0:
        fail("full-covariance audit underestimates dense matrix memory")
    if float(all_trainable.get("precision_plus_cholesky_float64_gib", 0.0)) < 1000.0:
        fail("full-covariance audit underestimates Cholesky memory")
    if float(weight_only.get("cholesky_flops", 0.0)) < 1e15:
        fail("full-covariance audit underestimates full-network Cholesky cost")
    if float(weight_blocks.get("precision_plus_cholesky_float64_gib", 0.0)) < 100.0:
        fail("full-covariance audit underestimates tensor-block memory")
    if not largest:
        fail("full-covariance audit has no largest tensor rows")
    top = largest[0]
    if int(top.get("parameter_count", 0)) != 36_864:
        fail("full-covariance audit top tensor should have 36,864 parameters")
    if float(top.get("dense_precision_float64_gib", 0.0)) < 10.0:
        fail("full-covariance audit top tensor memory is too small")
    text = (
        ROOT / "docs" / "cifar10_resnet20_full_covariance_feasibility.md"
    ).read_text(encoding="utf-8")
    for phrase in [
        "not a runnable exact",
        "553.1",
        "1,106.3",
        "full-network rank-16/rank-32/rank-64/rank-128 Hessian-plus-diagonal Laplace",
    ]:
        if phrase not in text:
            fail(f"full-covariance feasibility doc missing phrase: {phrase}")


def require_digits_fullnet_laplace_probe() -> None:
    csv_path = ROOT / "runs" / "digits_fullnet_laplace_tiny_r2_p0p3_summary.csv"
    doc_path = ROOT / "docs" / "digits_fullnet_laplace_tiny_r2_p0p3.md"
    if not csv_path.exists() or csv_path.stat().st_size < 500:
        fail("tiny full-network dense Laplace summary CSV missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 500:
        fail("tiny full-network dense Laplace summary doc missing or too small")

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 4:
        fail("tiny full-network dense Laplace summary should contain four scales")
    scales = {round(float(row["full_laplace_scale"]), 8) for row in rows}
    expected_scales = {round(value, 8) for value in [1e-5, 1e-4, 1e-3, 1e-2]}
    if scales != expected_scales:
        fail(f"tiny full-network dense Laplace scales changed: {scales}")
    selected = [
        row
        for row in rows
        if abs(float(row["full_laplace_scale"]) - 1e-3) < 1e-12
    ]
    if len(selected) != 1:
        fail("tiny full-network dense Laplace scale 1e-3 row missing")
    row = selected[0]
    if int(float(row["num_runs"])) != 5:
        fail("tiny full-network dense Laplace row must be five-seed")
    if int(round(float(row["parameter_count"]))) != 310:
        fail("tiny full-network dense Laplace row should cover 310 trainable parameters")
    if float(row["examples_seen"]) < 1400:
        fail("tiny full-network dense Laplace Hessian should see the full digits train set")
    if float(row["dense_accuracy"]) <= 0.80:
        fail("tiny full-network dense Laplace dense model accuracy unexpectedly low")
    if float(row["imp_accuracy"]) <= 0.84:
        fail("tiny full-network dense Laplace IMP accuracy unexpectedly low")
    if float(row["sample_accuracy_mean"]) <= 0.83:
        fail("tiny full-network dense Laplace samples should remain accurate")
    if float(row["posterior_minus_chain_start_jaccard"]) >= -0.05:
        fail("tiny full-network dense Laplace posterior should remain below chain-start support")
    if float(row["posterior_to_chain_start_magnitude_jaccard_mean"]) >= 0.85:
        fail("tiny full-network dense Laplace scale 1e-3 samples should move from chain-start")
    if float(row["chain_start_magnitude_to_imp_jaccard"]) <= float(
        row["posterior_jaccard_mean"]
    ):
        fail("tiny full-network dense Laplace posterior should not beat chain-start support")

    metrics_paths = sorted(
        (ROOT / "runs" / "digits_fullnet_laplace_tiny_r2_p0p3").glob("*/metrics.json")
    )
    if len(metrics_paths) != 5:
        fail(
            "tiny full-network dense Laplace raw metrics should contain five seeds, "
            f"found {len(metrics_paths)}"
        )
    seeds = set()
    for path in metrics_paths:
        payload = load_json(path)
        seeds.add(int(payload.get("seed", -1)))
        factors = payload.get("full_laplace", {})
        if int(factors.get("parameter_count", 0)) != 310:
            fail(f"tiny full-network dense Laplace raw metrics parameter count changed: {path}")
        if factors.get("precision_cholesky_shape") != [310, 310]:
            fail(f"tiny full-network dense Laplace raw metrics Cholesky shape changed: {path}")
        if len(payload.get("rows", [])) != 4:
            fail(f"tiny full-network dense Laplace raw metrics should contain four scale rows: {path}")
    if seeds != {0, 1, 2, 3, 4}:
        fail(f"tiny full-network dense Laplace raw metrics seeds changed: {sorted(seeds)}")

    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Full-network Dense Laplace Probe Summary",
        "exact dense full-network softmax-GGN/Laplace",
        "310.0000",
        "0.7545",
        "not CIFAR-scale evidence",
    ]:
        if phrase not in text:
            fail(f"tiny full-network dense Laplace doc missing phrase: {phrase}")


def require_fake_resnet_fullnet_laplace_smoke() -> None:
    csv_path = ROOT / "runs" / "fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv"
    doc_path = ROOT / "docs" / "fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md"
    if not csv_path.exists() or csv_path.stat().st_size < 500:
        fail("fake-CIFAR ResNet full-network dense Laplace summary CSV missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 500:
        fail("fake-CIFAR ResNet full-network dense Laplace summary doc missing or too small")

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2:
        fail("fake-CIFAR ResNet full-network dense Laplace summary should contain two scales")
    scales = {round(float(row["full_laplace_scale"]), 8) for row in rows}
    expected_scales = {round(value, 8) for value in [1e-5, 1e-3]}
    if scales != expected_scales:
        fail(f"fake-CIFAR ResNet full-network dense Laplace scales changed: {scales}")
    selected = [
        row
        for row in rows
        if abs(float(row["full_laplace_scale"]) - 1e-3) < 1e-12
    ]
    if len(selected) != 1:
        fail("fake-CIFAR ResNet full-network dense Laplace scale 1e-3 row missing")
    row = selected[0]
    if int(float(row["num_runs"])) != 5:
        fail("fake-CIFAR ResNet full-network dense Laplace row must be five-seed")
    if int(round(float(row["parameter_count"]))) != 1229:
        fail("fake-CIFAR ResNet full-network dense Laplace row should cover 1,229 trainable parameters")
    if int(round(float(row["weight_parameter_count"]))) != 1121:
        fail("fake-CIFAR ResNet full-network dense Laplace row should cover 1,121 weight parameters")
    if int(round(float(row["examples_seen"]))) != 16:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should use one Hessian batch")
    if float(row["posterior_to_chain_start_magnitude_jaccard_mean"]) >= 0.60:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should move from chain-start support")
    if float(row["posterior_minus_chain_start_jaccard"]) >= -0.30:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should remain below chain-start support")

    metrics_paths = sorted(
        (ROOT / "runs" / "fake_cifar10_resnet20_w1_fullnet_laplace_smoke").glob(
            "*/metrics.json"
        )
    )
    if len(metrics_paths) != 5:
        fail(
            "fake-CIFAR ResNet full-network dense Laplace raw metrics should contain "
            f"five seeds, found {len(metrics_paths)}"
        )
    seeds = set()
    for path in metrics_paths:
        payload = load_json(path)
        seeds.add(int(payload.get("seed", -1)))
        config = payload.get("config", {})
        if config.get("dataset") != "fake-cifar10" or config.get("model") != "resnet20":
            fail(f"fake-CIFAR ResNet full-network dense Laplace raw config changed: {path}")
        if int(config.get("resnet_width", 0)) != 1:
            fail(f"fake-CIFAR ResNet full-network dense Laplace should use width 1: {path}")
        factors = payload.get("full_laplace", {})
        if int(factors.get("parameter_count", 0)) != 1229:
            fail(f"fake-CIFAR ResNet full-network dense Laplace parameter count changed: {path}")
        if factors.get("precision_cholesky_shape") != [1229, 1229]:
            fail(f"fake-CIFAR ResNet full-network dense Laplace Cholesky shape changed: {path}")
        if len(payload.get("rows", [])) != 2:
            fail(f"fake-CIFAR ResNet full-network dense Laplace raw metrics should contain two scale rows: {path}")
    if seeds != {0, 1, 2, 3, 4}:
        fail(f"fake-CIFAR ResNet full-network dense Laplace raw seeds changed: {sorted(seeds)}")

    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "fake-CIFAR ResNet-20 width-1",
        "convolutional/residual/BatchNorm code-path smoke test",
        "1229.0000",
        "0.5392",
        "not real CIFAR evidence",
    ]:
        if phrase not in text:
            fail(f"fake-CIFAR ResNet full-network dense Laplace doc missing phrase: {phrase}")


def require_linear_connectivity_barrier_audit() -> None:
    csv_path = ROOT / "runs" / "linear_connectivity_barrier_audit.csv"
    json_path = ROOT / "runs" / "linear_connectivity_barrier_audit.json"
    doc_path = ROOT / "docs" / "linear_connectivity_barrier_audit.md"
    for path, label in [
        (csv_path, "CSV"),
        (json_path, "JSON"),
        (doc_path, "doc"),
    ]:
        if not path.exists() or path.stat().st_size < 500:
            fail(f"linear connectivity barrier audit {label} missing or too small")

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6:
        fail("linear connectivity barrier audit should contain six rows")
    by_label = {row["label"]: row for row in rows}
    required_labels = {
        "MNIST Gate1 SGLD r5",
        "Fashion-MNIST Gate1 SGLD r5",
        "CIFAR-10 ResNet-20 long SGLD r5",
        "CIFAR-10 ResNet-20 long SWAG r5",
        "CIFAR-10 ResNet-20 3-chain SGLD r5",
        "CIFAR-10 ResNet-20 short SWAG r5",
    }
    missing = sorted(required_labels - set(by_label))
    if missing:
        fail(f"linear connectivity barrier audit missing rows: {missing}")
    for row in rows:
        if int(row["num_runs"]) != 5:
            fail(f"linear connectivity audit row should be five-seed: {row['label']}")
        if float(row["posterior_minus_chain_start_jaccard_mean"]) > 0.001:
            fail(f"posterior should not beat chain-start in connectivity audit: {row['label']}")
    mnist = by_label["MNIST Gate1 SGLD r5"]
    fashion = by_label["Fashion-MNIST Gate1 SGLD r5"]
    cifar_sgld = by_label["CIFAR-10 ResNet-20 long SGLD r5"]
    cifar_swag = by_label["CIFAR-10 ResNet-20 long SWAG r5"]
    if float(mnist["dense_imp_barrier_mean"]) >= 0.01:
        fail("MNIST dense-IMP linear barrier should stay near zero")
    if float(fashion["dense_imp_barrier_mean"]) >= 0.05:
        fail("Fashion-MNIST dense-IMP linear barrier should stay near zero")
    if float(cifar_sgld["dense_imp_barrier_mean"]) <= 2.0:
        fail("CIFAR long SGLD linear barrier should be large")
    if float(cifar_swag["dense_imp_barrier_mean"]) <= 2.0:
        fail("CIFAR long SWAG linear barrier should be large")

    payload = load_json(json_path)
    checks = payload.get("interpretation_checks", {})
    for key in [
        "all_rows_five_seed",
        "mnist_dense_imp_barrier_near_zero",
        "fashion_dense_imp_barrier_near_zero",
        "cifar_dense_imp_barriers_large",
        "posterior_never_beats_chain_start",
    ]:
        if checks.get(key) is not True:
            fail(f"linear connectivity barrier audit check failed: {key}")

    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Linear Connectivity Barrier Audit",
        "MNIST Gate1 SGLD r5",
        "0.0026",
        "3.0827",
        "orthogonal landscape diagnostics",
        "not evidence of posterior-ticket equivalence",
    ]:
        if phrase not in text:
            fail(f"linear connectivity barrier audit doc missing phrase: {phrase}")


def require_container_lock() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    gpu_dockerfile = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
    container_doc = (ROOT / "docs" / "container_lock.md").read_text(
        encoding="utf-8"
    )
    gpu_container_doc = (ROOT / "docs" / "gpu_training_container.md").read_text(
        encoding="utf-8"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ci_workflow = (ROOT / ".github" / "workflows" / "check.yml").read_text(
        encoding="utf-8"
    )
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    ci_requirements = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    gpu_requirements = (ROOT / "requirements-gpu-lock.txt").read_text(encoding="utf-8")
    expected_digest = (
        "python:3.12.3-slim-bookworm@sha256:"
        "fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84"
    )
    expected_gpu_digest = (
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:"
        "5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f"
    )
    for phrase in [
        expected_digest,
        "requirements-ci.txt",
        "git",
        "poppler-utils",
        "texlive-latex-base",
        "texlive-bibtex-extra",
        "texlive-latex-extra",
        "ripgrep",
        "make source-repository-check PYTHON=python",
        "make ci-check paper-check PYTHON=python",
        "Full artifact payload absent",
    ]:
        if phrase not in dockerfile:
            fail(f"Dockerfile missing container lock phrase: {phrase}")
    for phrase in [
        expected_digest,
        "numpy==1.26.4",
        "git",
        "poppler-utils",
        "make container-build",
        "make container-check",
        "make source-repository-check PYTHON=python",
        "CPU-only artifact-verification container",
        "Dockerfile.gpu",
    ]:
        if phrase not in container_doc:
            fail(f"container lock doc missing phrase: {phrase}")
    for phrase in [
        expected_gpu_digest,
        "requirements-gpu-lock.txt",
        'CMD ["python", "scripts/check_gpu_training_environment.py"]',
    ]:
        if phrase not in gpu_dockerfile:
            fail(f"Dockerfile.gpu missing GPU container phrase: {phrase}")
    for phrase in [
        expected_gpu_digest,
        "make gpu-container-build",
        "make gpu-container-env-check",
        "requirements-gpu-lock.txt",
        "torch==2.11.0",
        "Torch CUDA `13.0`",
    ]:
        if phrase not in gpu_container_doc:
            fail(f"GPU training container doc missing phrase: {phrase}")
    for phrase in [
        "container-build",
        "container-check",
        "CONTAINER_IMAGE",
        "gpu-container-build",
        "gpu-container-env-check",
        "GPU_CONTAINER_IMAGE",
        "scripts/run_gpu_container_env_check.py",
    ]:
        if phrase not in makefile:
            fail(f"Makefile missing container target phrase: {phrase}")
    for phrase in [
        "sudo apt-get update",
        "git",
        "make",
        "poppler-utils",
        "ripgrep",
        "texlive-bibtex-extra",
        "texlive-fonts-recommended",
        "texlive-latex-extra",
        "texlive-latex-base",
        "texlive-latex-recommended",
        "make source-repository-check PYTHON=python",
        "Full artifact payload absent",
        "make ci-check paper-check PYTHON=python",
    ]:
        if phrase not in ci_workflow:
            fail(f"CI workflow missing paper-build phrase: {phrase}")
    for phrase in [
        "paper/main_submission.aux",
        "paper/main_submission.bbl",
        "paper/main_submission.blg",
        "paper/main_submission.log",
        "paper/main_submission.out",
        "paper/neurips_submission.aux",
        "paper/neurips_submission.bbl",
        "paper/neurips_submission.blg",
        "paper/neurips_submission.log",
        "paper/neurips_submission.out",
    ]:
        if phrase not in dockerignore:
            fail(f".dockerignore missing submission build artifact: {phrase}")
    if "numpy==1.26.4" not in ci_requirements:
        fail("requirements-ci.txt missing numpy==1.26.4")
    for phrase in [
        "numpy==1.26.4",
        "scikit-learn==1.4.1.post1",
        "scipy==1.11.4",
        "torch==2.11.0",
        "torchvision==0.26.0",
    ]:
        if phrase not in gpu_requirements:
            fail(f"requirements-gpu-lock.txt missing GPU package phrase: {phrase}")
    if "matplotlib==" in gpu_requirements:
        fail("requirements-gpu-lock.txt should not install matplotlib")


def require_claim_ledger() -> None:
    text = (ROOT / "docs" / "paper_claim_ledger.md").read_text(encoding="utf-8")
    expected_phrases = [
        "Posterior support is not random",
        "58/59 groups",
        "0/59 groups",
        "55/57 groups",
        "Full-data CIFAR direct mode/ticket probes",
        "rank-16, rank-32, rank-64, and rank-128 Hessian-plus-diagonal Laplace",
        "tiny exact dense full-network Laplace",
        "fullnet scale=0.0010",
        "fullnet params=310",
        "fullnet post-chain=0.8084",
        "fullnet posterior-chain=-0.1050",
        "convolutional ResNet smoke",
        "fake-resnet fullnet scale=0.0010",
        "params=1229",
        "post-chain=0.5398",
        "Linear connectivity barriers are orthogonal",
        "MNIST dense-IMP barrier=0.0026",
        "CIFAR long SGLD/SWAG dense-IMP barriers=3.0827/3.7402",
        "68,144-parameter exact block-diagonal",
        "68,144-parameter exact joint-group",
        "86,576-parameter exact joint-group",
        "270,896-parameter exact joint-group",
        "270,896-parameter direct joint-group",
        "post-hoc exhaustive graph/permutation realignment",
        "raw mask/state files=0",
        "mask-artifact smoke parameters=4350",
        "record-level posthoc matching=True",
        "local channel matching=True",
        "full-data saved-artifact budget=284.18 MiB",
        "full-data record-level posthoc=True",
        "full-data channel skipped=7",
        "full-data local channel=False",
        "global-channel hamming raw/aligned=0.2105/0.2113",
        "global-channel support overlap=0.3738",
        "exact stage1 assignments=128",
        "stage1 coordinate exact=True",
        "full channel assignments log10=840.4",
        "tensor+score replacement acc",
        "residualized-score acc",
        "posterior-residualized acc",
        "learned-subspace residualized acc",
        "oracle-overlap drop",
        "Open limitation, bounded",
    ]
    for phrase in expected_phrases:
        if phrase not in text:
            fail(f"paper claim ledger missing phrase: {phrase}")
    pass_count = text.count("| Pass |")
    if pass_count < 7:
        fail(f"paper claim ledger has too few passing claim rows: {pass_count}")


def require_reviewer_objection_matrix() -> None:
    json_path = ROOT / "runs" / "reviewer_objection_matrix.json"
    doc_path = ROOT / "docs" / "reviewer_objection_matrix.md"
    if not json_path.exists() or json_path.stat().st_size < 1000:
        fail("reviewer objection matrix JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 1000:
        fail("reviewer objection matrix doc missing or too small")
    payload = load_json(json_path)
    rows = payload.get("rows", [])
    if len(rows) != 9:
        fail("reviewer objection matrix should contain nine rows")
    objections = {str(row.get("objection")) for row in rows}
    expected_objections = {
        "Posterior masks only need to beat random masks.",
        "The posterior sampler is too weak or stuck in one chain.",
        "Function-space agreement is enough even if masks differ.",
        "A channel permutation or re-basin step would align posterior masks.",
        "The covariance posterior is too diagonal, local, or head-only.",
        "Linear mode connectivity explains the ticket/posterior relation.",
        "Learned Bayesian or variational masks could recover tickets.",
        "A simpler trajectory or process subspace explains the IMP residual.",
        "The artifact is not yet a submit-ready package.",
    }
    missing = sorted(expected_objections - objections)
    if missing:
        fail(f"reviewer objection matrix missing objections: {missing}")
    summary = payload.get("summary", {})
    if int(summary.get("closed_count", 0)) < 5:
        fail("reviewer objection matrix should close at least five objections")
    if int(summary.get("bounded_open_count", 0)) != 2:
        fail("reviewer objection matrix should have two bounded-open limitations")
    if int(summary.get("open_packaging_count", 0)) != 1:
        fail("reviewer objection matrix should record one packaging limitation")
    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Reviewer Objection Matrix",
        "posterior>random 58/59 groups",
        "dense-start cSGLD samples=75",
        "global-channel posterior/ticket Hamming=0.2105",
        "dense CIFAR matrix=553.1 GiB",
        "MNIST/Fashion dense-IMP barriers=0.0026/0.0395",
        "variational/hard-concrete support=0.0907/0.0922",
        "Open packaging limitation",
        "not a claim that the project is submission-ready",
    ]:
        if phrase not in text:
            fail(f"reviewer objection matrix missing phrase: {phrase}")


def require_paper_submission_shape_audit() -> None:
    json_path = ROOT / "runs" / "paper_submission_shape_audit.json"
    doc_path = ROOT / "docs" / "paper_submission_shape_audit.md"
    if not json_path.exists() or json_path.stat().st_size < 1000:
        fail("paper submission shape audit JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 1000:
        fail("paper submission shape audit doc missing or too small")
    payload = load_json(json_path)
    if payload.get("paper_tex") != "paper/main.tex":
        fail("paper submission shape audit should target paper/main.tex")
    if int(payload.get("section_count", 0)) < 7:
        fail("paper submission shape audit section count unexpectedly low")
    if int(payload.get("main_body_lines_before_appendix", 0)) > 850:
        fail("paper submission shape audit should keep the main body within target")
    result_sections = [
        section
        for section in payload.get("sections", [])
        if section.get("title") == "Current Results"
    ]
    if not result_sections:
        fail("paper submission shape audit missing Current Results section")
    if int(result_sections[0].get("lines", 0)) > 450:
        fail("paper submission shape audit should keep Current Results within target")
    if int(payload.get("reviewer_objection_rows", 0)) != 9:
        fail("paper submission shape audit should consume the nine-row reviewer matrix")
    if payload.get("submission_shape_ready") is not True:
        fail("paper submission shape audit should mark the condensed draft ready")
    risk_flags = set(payload.get("risk_flags", []))
    if risk_flags:
        fail(f"paper submission shape audit should have no blocking risk flags: {sorted(risk_flags)}")
    checks = payload.get("objection_checks", {})
    for key in [
        "random_control",
        "sampler_movement",
        "function_vs_mask",
        "alignment_permutation",
        "covariance_fidelity",
        "linear_connectivity",
        "learned_masks",
        "process_mechanism",
    ]:
        if checks.get(key) is not True:
            fail(f"paper submission shape audit missing objection coverage: {key}")
    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Paper Submission Shape Audit",
        "Current status: ready",
        "Current Results",
        "PDF pages are total compiled pages",
        "Completed Condensation",
        "Main-body lines are within the 850-line target",
        "Current Results is within the 450-line target",
    ]:
        if phrase not in text:
            fail(f"paper submission shape audit missing phrase: {phrase}")


def require_submission_pdf_shape_audit() -> None:
    json_path = ROOT / "runs" / "submission_pdf_shape_audit.json"
    doc_path = ROOT / "docs" / "submission_pdf_shape_audit.md"
    if not json_path.exists() or json_path.stat().st_size < 250:
        fail("submission PDF shape audit JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 500:
        fail("submission PDF shape audit doc missing or too small")
    payload = load_json(json_path)
    if payload.get("submission_pdf") != "paper/main_submission.pdf":
        fail("submission PDF shape audit should target paper/main_submission.pdf")
    if payload.get("submission_pdf_ready") is not True:
        fail("submission PDF shape audit should mark the main-only PDF ready")
    if payload.get("risk_flags"):
        fail(f"submission PDF shape audit has risk flags: {payload['risk_flags']}")
    if int(payload.get("pdf_page_count", 0)) > int(payload.get("max_pages", 10)):
        fail("submission PDF exceeds page budget")
    if int(payload.get("size_bytes", 0)) < 100_000:
        fail("submission PDF audit reports an unexpectedly small PDF")
    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Submission PDF Shape Audit",
        "Current status: ready",
        "main-only submission PDF",
        "paper/main_submission.pdf",
        "Risk Flags",
        "none",
    ]:
        if phrase not in text:
            fail(f"submission PDF shape audit missing phrase: {phrase}")


def require_venue_submission_compliance_audit() -> None:
    json_path = ROOT / "runs" / "venue_submission_compliance_audit.json"
    doc_path = ROOT / "docs" / "venue_submission_compliance_audit.md"
    if not json_path.exists() or json_path.stat().st_size < 500:
        fail("venue submission compliance audit JSON missing or too small")
    if not doc_path.exists() or doc_path.stat().st_size < 500:
        fail("venue submission compliance audit doc missing or too small")
    payload = load_json(json_path)
    if payload.get("paper_tex") != "paper/main.tex":
        fail("venue submission audit should target paper/main.tex")
    if payload.get("submission_pdf") != "paper/neurips_submission.pdf":
        fail("venue submission audit should target paper/neurips_submission.pdf")
    if payload.get("venue_profile") != "neurips_2026_main":
        fail("venue submission audit should use the NeurIPS 2026 profile")
    if payload.get("target_venue") != "NeurIPS 2026 Main Track":
        fail("venue submission audit should target NeurIPS 2026 Main Track")
    if payload.get("content_packet_ready") is not True:
        fail(f"venue content packet should be ready: {payload.get('content_risk_flags')}")
    if payload.get("venue_binding_ready") is not True:
        fail(
            f"NeurIPS style/checklist binding should be ready: "
            f"{payload.get('venue_binding_risk_flags')}"
        )
    if payload.get("checklist_release_ready") is not True:
        fail(
            f"venue submission audit should have resolved checklist release risks: "
            f"{payload.get('checklist_release_risk_flags')}"
        )
    if payload.get("release_packaging_ready") is not True:
        fail("venue submission audit should accept local GPU-container validation for submission packaging")
    if payload.get("venue_submission_ready") is not True:
        fail("venue submission audit should mark the venue submission packet ready")
    if payload.get("document_class") != "article":
        fail("venue submission audit should record the current article class")
    if payload.get("known_venue_style_detected") is not True:
        fail("venue submission audit should detect the official NeurIPS style")
    if payload.get("official_style_file_present") is not True:
        fail("venue submission audit should confirm paper/neurips_2026.sty")
    if payload.get("neurips_style_source_wired") is not True:
        fail("venue submission audit should confirm NeurIPS source wiring")
    if payload.get("neurips_checklist_present") is not True:
        fail("venue submission audit should confirm the NeurIPS checklist")
    if payload.get("neurips_checklist_no_todos") is not True:
        fail("venue submission audit should confirm no checklist TODOs")
    if payload.get("anonymous_author") is not True:
        fail("venue submission audit should confirm anonymous author")
    if int(payload.get("abstract_word_count", 9999)) > int(
        payload.get("max_abstract_words", 250)
    ):
        fail("venue submission audit abstract exceeds the local word budget")
    content_pages = payload.get("content_pages_before_references")
    if content_pages is None:
        fail("venue submission audit should locate References in the NeurIPS PDF")
    if int(content_pages) > int(payload.get("max_pages_before_references", 9)):
        fail("venue submission audit NeurIPS main content exceeds page budget")
    if payload.get("reference_start_page") is None:
        fail("venue submission audit should record the References start page")
    if payload.get("appendix_after_references_wired") is not True:
        fail("venue submission audit should keep appendix/checklist after references")
    binding_flags = set(payload.get("venue_binding_risk_flags", []))
    if binding_flags:
        fail(f"venue submission audit has binding flags: {sorted(binding_flags)}")
    metadata_docs = payload.get("release_metadata_docs", {})
    for key in [
        "compute_resource_accounting",
        "asset_license_inventory",
        "new_asset_inventory",
    ]:
        status = metadata_docs.get(key)
        if not isinstance(status, dict):
            fail(f"venue submission audit missing release metadata doc status: {key}")
        if status.get("exists") is not True:
            fail(f"release metadata doc should exist: {key}")
        if status.get("required_terms_present") is not True:
            fail(f"release metadata doc missing required terms: {key}: {status}")
    checklist_flags = set(payload.get("checklist_release_risk_flags", []))
    if checklist_flags:
        fail(f"unexpected venue checklist release flags: {sorted(checklist_flags)}")
    packaging_flags = set(payload.get("release_packaging_risk_flags", []))
    if packaging_flags:
        fail(f"unexpected venue release packaging flags: {sorted(packaging_flags)}")
    packaging_warning_flags = set(payload.get("release_packaging_warning_flags", []))
    if packaging_warning_flags != {"external_gpu_container_run_not_observed"}:
        fail(f"unexpected venue release packaging warning flags: {sorted(packaging_warning_flags)}")
    local_gpu = payload.get("local_gpu_validation", {})
    if not isinstance(local_gpu, dict) or local_gpu.get("local_gpu_container_ready") is not True:
        fail(f"venue submission audit missing ready local GPU validation: {local_gpu}")
    text = " ".join(doc_path.read_text(encoding="utf-8").split())
    for phrase in [
        "Venue Submission Compliance Audit",
        "Current content-packet status: ready",
        "Current venue-binding status: ready",
        "Current checklist-release status: ready",
        "Current release-packaging status: ready",
        "Current venue-submission status: ready",
        "paper/neurips_submission.pdf",
        "runs/external_validation_readiness_audit.json",
        "runs/local_gpu_container_validation.json",
        "Release Metadata Docs",
        "docs/compute_resource_accounting.md",
        "docs/asset_license_inventory.md",
        "docs/new_asset_inventory.md",
        "Main-content pages before references",
        "external_gpu_container_run_not_observed",
    ]:
        if phrase not in text:
            fail(f"venue submission compliance audit missing phrase: {phrase}")


def require_release_metadata_docs() -> None:
    doc_terms = {
        "docs/compute_resource_accounting.md": [
            "Compute Resource Accounting",
            "NVIDIA GeForce RTX 5090",
            "Torch CUDA 13.0",
            "553.1 GiB",
            "1,106.3 GiB",
            "wall-clock",
            "mask_artifacts.npz",
        ],
        "docs/asset_license_inventory.md": [
            "Asset License Inventory",
            "MIT License",
            "LICENSE",
            "Anonymous Authors",
            "MNIST",
            "Fashion-MNIST",
            "CIFAR-10",
            "CIFAR-100",
            "raw benchmark datasets",
            "third-party",
        ],
        "docs/new_asset_inventory.md": [
            "New Asset Inventory",
            "public_release_manifest",
            "mask_artifacts.npz",
            "paper/neurips_submission.pdf",
            "data/",
            "Public Release Status",
        ],
    }
    for rel, phrases in doc_terms.items():
        path = ROOT / rel
        if not path.exists() or path.stat().st_size < 1000:
            fail(f"release metadata doc missing or too small: {rel}")
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase not in text:
                fail(f"release metadata doc {rel} missing phrase: {phrase}")


def require_release_manifest() -> None:
    manifest = load_json(ROOT / "runs" / "public_release_manifest.json")
    if manifest.get("root") != ".":
        fail("public release manifest root should be anonymized as '.'")
    manifest_text = json.dumps(manifest, sort_keys=True)
    forbidden_manifest_terms = [
        "/home/",
        "/Users/",
        "\\\\Users\\\\",
        "/Projects/" + "Lottery",
        "suan" + "lab",
        "MyUbuntu" + "5090",
    ]
    for term in forbidden_manifest_terms:
        if term in manifest_text:
            fail(f"public release manifest contains local identity/path term: {term}")
    if int(manifest.get("file_count", 0)) < 150:
        fail("public release manifest has too few files")
    files = {
        str(entry["path"]): entry
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and "path" in entry
    }
    if any(path.startswith("data/") for path in files):
        fail("public release manifest should not include raw data/ caches")
    if "docs/external_validation_receipts.json" in files:
        fail("public release manifest should exclude the mutable external receipt registry")
    required_paths = [
        ".dockerignore",
        ".github/workflows/check.yml",
        ".gitignore",
        "Dockerfile",
        "Dockerfile.gpu",
        "LICENSE",
        "Makefile",
        "README.md",
        "requirements-ci.txt",
        "requirements-gpu-lock.txt",
        "requirements-lock.txt",
        "docs/container_lock.md",
        "docs/gpu_training_container.md",
        "docs/local_gpu_container_validation.md",
        "docs/compute_resource_accounting.md",
        "docs/asset_license_inventory.md",
        "docs/new_asset_inventory.md",
        "docs/cifar10_resnet20_full_covariance_feasibility.md",
        "docs/digits_fullnet_laplace_tiny_r2_p0p3.md",
        "docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md",
        "docs/linear_connectivity_barrier_audit.md",
        "docs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md",
        "docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md",
        "docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md",
        "docs/mode_ticket_artifact_storage_budget.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md",
        "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md",
        "docs/resnet_channel_permutation_exhaustive_feasibility_audit.md",
        "docs/cifar10_subset_hard_concrete_mask_training_smoke.md",
        "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3.md",
        "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md",
        "docs/environment_lock.json",
        "docs/mode_ticket_alignment_artifact_audit.md",
        "docs/paper_claim_ledger.md",
        "docs/paper_submission_shape_audit.md",
        "docs/submission_pdf_shape_audit.md",
        "docs/venue_submission_compliance_audit.md",
        "docs/reviewer_objection_matrix.md",
        "docs/reproducibility_manifest.md",
        "docs/submission_readiness_audit.md",
        "docs/thread_goal_completion_audit.md",
        "runs/mode_ticket_alignment_artifact_audit.json",
        "runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json",
        "runs/mode_ticket_artifact_storage_budget.json",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json",
        "runs/resnet_channel_permutation_exhaustive_feasibility_audit.json",
        "runs/paper_submission_shape_audit.json",
        "runs/submission_pdf_shape_audit.json",
        "runs/venue_submission_compliance_audit.json",
        "runs/local_gpu_container_validation.json",
        "runs/reviewer_objection_matrix.json",
        "runs/paper_stats.json",
        "paper/main.tex",
        "paper/refs.bib",
        "paper/main.pdf",
        "paper/main_submission.pdf",
        "paper/neurips_2026.sty",
        "paper/neurips_checklist.tex",
        "paper/neurips_submission.pdf",
        "paper/tables/statistical_summary.tex",
        "scripts/audit_mode_ticket_alignment_artifacts.py",
        "scripts/audit_mask_artifact_posthoc_matching.py",
        "scripts/audit_full_data_channel_permutation_matching.py",
        "scripts/audit_exhaustive_channel_permutation_feasibility.py",
        "scripts/audit_mode_ticket_artifact_storage_budget.py",
        "scripts/run_digits_fullnet_laplace_probe.py",
        "scripts/summarize_fullnet_laplace_probe.py",
        "scripts/audit_linear_connectivity_barriers.py",
        "src/lottery/full_laplace.py",
        "scripts/build_paper_claim_ledger.py",
        "scripts/audit_paper_submission_shape.py",
        "scripts/audit_submission_pdf_shape.py",
        "scripts/audit_venue_submission_compliance.py",
        "scripts/build_reviewer_objection_matrix.py",
        "scripts/audit_external_validation_readiness.py",
        "scripts/build_external_validation_receipt_template.py",
        "scripts/update_external_validation_receipts.py",
        "scripts/build_external_validation_runbook.py",
        "scripts/build_submission_handoff.py",
        "scripts/stage_public_repository_snapshot.py",
        "scripts/smoke_public_repository_snapshot.py",
        "scripts/verify_source_repository_snapshot.py",
        "scripts/audit_release_anonymization.py",
        "scripts/build_public_release_archive.py",
        "scripts/smoke_public_release_archive.py",
        "scripts/check_gpu_training_environment.py",
        "scripts/run_gpu_container_env_check.py",
        "scripts/build_local_gpu_container_validation.py",
        "runs/cifar10_resnet20_full_covariance_feasibility.json",
        "runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv",
        "runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv",
        "runs/linear_connectivity_barrier_audit.csv",
        "runs/linear_connectivity_barrier_audit.json",
        "runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/metrics.json",
        "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz",
        "runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv",
        "runs/cifar10_subset_hard_concrete_mask_training_smoke_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv",
    ]
    missing = [path for path in required_paths if path not in files]
    if missing:
        fail(f"public release manifest missing required paths: {missing}")
    mask_artifact_paths = [
        path
        for path in files
        if path.startswith("runs/fake_cifar10_mode_ticket_mask_artifact_smoke/")
        and path.endswith("/mask_artifacts.npz")
    ]
    if not mask_artifact_paths:
        fail("public release manifest missing mask_artifacts.npz smoke fixture")
    for rel in required_paths:
        expected = str(files[rel].get("sha256", ""))
        actual = sha256(ROOT / rel)
        if actual != expected:
            fail(f"public release manifest hash mismatch for {rel}")
    for rel in mask_artifact_paths:
        expected = str(files[rel].get("sha256", ""))
        actual = sha256(ROOT / rel)
        if actual != expected:
            fail(f"public release manifest hash mismatch for {rel}")
    release_text = (ROOT / "docs" / "public_release_manifest.md").read_text(
        encoding="utf-8"
    )
    for phrase in ["Total files:", "Required Artifacts", "make paper-check", "make paper-neurips-check"]:
        if phrase not in release_text:
            fail(f"public release manifest markdown missing phrase: {phrase}")


def require_release_anonymization_audit() -> None:
    payload = load_json(ROOT / "runs" / "release_anonymization_audit.json")
    if payload.get("manifest") != "runs/public_release_manifest.json":
        fail("release anonymization audit points at the wrong manifest")
    if payload.get("manifest_root") != ".":
        fail("release anonymization audit should confirm manifest root '.'")
    if payload.get("release_anonymization_ready") is not True:
        fail(
            "release anonymization audit should be ready: "
            f"{payload.get('risk_flags')}"
        )
    if payload.get("risk_flags") != []:
        fail(f"release anonymization audit has risk flags: {payload.get('risk_flags')}")
    if int(payload.get("finding_count", -1)) != 0:
        fail("release anonymization audit should have zero findings")
    if payload.get("findings") not in ([], None):
        fail("release anonymization audit findings should be empty")
    if payload.get("forbidden_manifest_paths") not in ([], None):
        fail("release anonymization audit should not find forbidden manifest paths")
    if payload.get("missing_paths") not in ([], None):
        fail("release anonymization audit should not find missing manifest paths")
    if int(payload.get("manifest_file_count", 0)) < 150:
        fail("release anonymization audit saw too few manifest files")
    if int(payload.get("scanned_text_files", 0)) < 100:
        fail("release anonymization audit scanned too few text files")

    doc_text = (ROOT / "docs" / "release_anonymization_audit.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "Release Anonymization Audit",
        "Current status: ready.",
        "runs/public_release_manifest.json",
        "Risk Flags",
        "- none",
        "This file is generated by `scripts/audit_release_anonymization.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"release anonymization audit markdown missing phrase: {phrase}")


def require_public_release_archive() -> None:
    payload = load_json(ROOT / "runs" / "public_release_archive_audit.json")
    if payload.get("archive_ready") is not True:
        fail(f"public release archive should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"public release archive has risk flags: {payload.get('risk_flags')}")
    if payload.get("manifest_root") != ".":
        fail("public release archive audit should confirm manifest root '.'")
    if payload.get("package_root") != "lottery_artifact_public_release":
        fail("public release archive package root changed unexpectedly")
    if int(payload.get("manifest_file_count", 0)) < 150:
        fail("public release archive audit saw too few manifest files")
    if int(payload.get("release_metadata_count", 0)) != 4:
        fail("public release archive should include four release metadata sidecars")
    if int(payload.get("expected_member_count", 0)) != int(
        payload.get("actual_member_count", -1)
    ):
        fail("public release archive member count mismatch")
    if int(payload.get("archive_bytes", 0)) < 100_000_000:
        fail("public release archive is unexpectedly small")
    for key in [
        "missing_source_paths",
        "missing_members",
        "extra_members",
        "unsafe_members",
        "non_file_members",
        "member_size_mismatches",
    ]:
        if payload.get(key) not in ([], None):
            fail(f"public release archive audit has {key}: {payload.get(key)}")
    expected_metadata = {
        "docs/public_release_manifest.md",
        "runs/public_release_manifest.json",
        "docs/release_anonymization_audit.md",
        "runs/release_anonymization_audit.json",
    }
    if set(payload.get("release_metadata_paths", [])) != expected_metadata:
        fail("public release archive metadata sidecars changed unexpectedly")

    archive = ROOT / str(payload.get("archive", ""))
    sidecar = ROOT / str(payload.get("sha256_sidecar", ""))
    if not archive.exists():
        fail(f"public release archive missing: {archive}")
    if not sidecar.exists():
        fail(f"public release archive sha256 sidecar missing: {sidecar}")
    digest = sha256(archive)
    if digest != payload.get("archive_sha256"):
        fail("public release archive sha256 does not match audit JSON")
    sidecar_text = sidecar.read_text(encoding="utf-8").strip()
    if not sidecar_text.startswith(f"{digest}  "):
        fail("public release archive sha256 sidecar does not match archive")

    doc_text = (ROOT / "docs" / "public_release_archive_audit.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "Public Release Archive Audit",
        "Current status: ready.",
        "dist/lottery_artifact_public_release_2026-05-06.tar.gz",
        "Release metadata sidecars",
        "Risk Flags",
        "- none",
        "This file is generated by `scripts/build_public_release_archive.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"public release archive audit markdown missing phrase: {phrase}")


def require_public_release_archive_smoke() -> None:
    payload = load_json(ROOT / "runs" / "public_release_archive_smoke.json")
    if payload.get("release_archive_smoke_ready") is not True:
        fail(f"public release archive smoke should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"public release archive smoke has risk flags: {payload.get('risk_flags')}")
    if payload.get("package_root") != "lottery_artifact_public_release":
        fail("public release archive smoke package root changed unexpectedly")
    if payload.get("manifest_root") != ".":
        fail("public release archive smoke should confirm manifest root '.'")
    if int(payload.get("manifest_file_count", 0)) < 150:
        fail("public release archive smoke saw too few manifest files")
    if int(payload.get("expected_file_count", 0)) != int(
        payload.get("actual_file_count", -1)
    ):
        fail("public release archive smoke extracted file count mismatch")
    if int(payload.get("checked_hash_count", 0)) < 150:
        fail("public release archive smoke checked too few manifest hashes")
    verifier = payload.get("verifier", {})
    if not isinstance(verifier, dict) or verifier.get("returncode") != 0:
        fail(f"public release extracted-package verifier failed: {verifier}")
    if "verified research artifacts" not in str(verifier.get("stdout_tail", "")):
        fail("public release extracted-package verifier output missing success marker")
    for key in ["metadata_missing", "missing_files", "extra_files", "hash_mismatches"]:
        if payload.get(key) not in ([], None):
            fail(f"public release archive smoke has {key}: {payload.get(key)}")

    doc_text = (ROOT / "docs" / "public_release_archive_smoke.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "Public Release Archive Smoke Test",
        "Current status: ready.",
        "runs the artifact verifier in release-package mode",
        "Risk Flags",
        "- none",
        "verified research artifacts",
        "This file is generated by `scripts/smoke_public_release_archive.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"public release archive smoke markdown missing phrase: {phrase}")


def require_public_repository_snapshot_audit() -> None:
    payload = load_json(ROOT / "runs" / "public_repository_snapshot_audit.json")
    if payload.get("public_repository_snapshot_ready") is not True:
        fail(f"public repository snapshot should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"public repository snapshot has risk flags: {payload.get('risk_flags')}")
    if payload.get("marker") != ".lottery_public_repository_snapshot":
        fail("public repository snapshot marker changed unexpectedly")
    if int(payload.get("source_file_count", 0)) < 100:
        fail("public repository snapshot has too few source files")
    if int(payload.get("tracked_file_count", 0)) != int(payload.get("source_file_count", -1)) + 1:
        fail("public repository snapshot tracked count should include source files plus marker")
    if int(payload.get("max_file_bytes", 0)) != 100_000_000:
        fail("public repository snapshot max file limit changed unexpectedly")
    if payload.get("oversized_files") not in ([], None):
        fail(f"public repository snapshot has oversized files: {payload.get('oversized_files')}")
    if payload.get("text_findings") not in ([], None):
        fail(f"public repository snapshot has text findings: {payload.get('text_findings')}")
    git = payload.get("git", {})
    if not isinstance(git, dict) or git.get("git_ready") is not True:
        fail(f"public repository snapshot git state is not ready: {git}")
    if git.get("git_clean") is not True:
        fail("public repository snapshot git state should be clean")
    commit = str(git.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail(f"public repository snapshot commit is not a git SHA1: {commit}")
    stage_dir = ROOT / str(payload.get("stage_dir", ""))
    if not (stage_dir / ".git").is_dir():
        fail(f"public repository snapshot stage dir missing .git: {stage_dir}")
    if not (stage_dir / ".lottery_public_repository_snapshot").is_file():
        fail(f"public repository snapshot marker missing: {stage_dir}")
    doc_text = (ROOT / "docs" / "public_repository_snapshot_audit.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "Public Repository Snapshot Audit",
        "Current status: ready.",
        "source-only anonymous git repository",
        "mutable external receipt registry",
        "run-artifact package remains the separate public",
        "Git commit",
        "Risk Flags",
        "- none",
        "This file is generated by `scripts/stage_public_repository_snapshot.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"public repository snapshot audit markdown missing phrase: {phrase}")


def require_public_repository_snapshot_smoke() -> None:
    payload = load_json(ROOT / "runs" / "public_repository_snapshot_smoke.json")
    if payload.get("source_repository_smoke_ready") is not True:
        fail(f"public repository snapshot smoke should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"public repository snapshot smoke has risk flags: {payload.get('risk_flags')}")
    check = payload.get("check", {})
    if not isinstance(check, dict) or check.get("returncode") != 0:
        fail(f"public repository snapshot smoke check failed: {check}")
    if check.get("success_marker_seen") is not True:
        fail("public repository snapshot smoke missing source verification marker")
    git = payload.get("git", {})
    if not isinstance(git, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", ""))):
        fail(f"public repository snapshot smoke missing git commit: {git}")
    doc_text = (ROOT / "docs" / "public_repository_snapshot_smoke.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "Public Repository Snapshot Smoke Test",
        "Current status: ready.",
        "make source-repository-check",
        "source_repository_snapshot_verified",
        "Risk Flags",
        "- none",
        "This file is generated by `scripts/smoke_public_repository_snapshot.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"public repository snapshot smoke markdown missing phrase: {phrase}")


def require_external_validation_readiness_audit() -> None:
    payload = load_json(ROOT / "runs" / "external_validation_readiness_audit.json")
    local = payload.get("local", {})
    if not isinstance(local, dict):
        fail("external validation audit missing local gate payload")
    if local.get("local_artifact_release_ready") is not True:
        fail(f"external validation audit local release gates are not ready: {local}")
    gates = local.get("gates", {})
    for key in [
        "release_anonymization",
        "public_release_archive",
        "public_release_archive_smoke",
        "archive_sha256_sidecar",
    ]:
        if not isinstance(gates, dict) or gates.get(key) is not True:
            fail(f"external validation audit local gate not ready: {key}")
    required_receipts = set(payload.get("required_receipts", []))
    expected_receipts = {
        "public_release_upload",
        "public_repository",
        "external_ci",
        "external_gpu_container",
    }
    if required_receipts != expected_receipts:
        fail(f"external validation audit required receipts changed: {required_receipts}")
    receipt_rows = payload.get("receipt_statuses", [])
    if not isinstance(receipt_rows, list) or len(receipt_rows) != len(expected_receipts):
        fail("external validation audit has incomplete receipt rows")
    row_keys = {str(row.get("key")) for row in receipt_rows if isinstance(row, dict)}
    if row_keys != expected_receipts:
        fail(f"external validation audit receipt rows changed: {row_keys}")
    repository_snapshot = payload.get("public_repository_snapshot", {})
    if not isinstance(repository_snapshot, dict):
        fail("external validation audit missing public repository snapshot status")
    if repository_snapshot.get("public_repository_snapshot_ready") is not True:
        fail(f"external validation audit should see a ready repository snapshot: {repository_snapshot}")
    if payload.get("local_clean_repository_ready") is not True:
        fail("external validation audit should mark local clean repository staging ready")
    if payload.get("clean_repository_ready") is not True:
        fail("external validation audit should mark clean repository status ready locally")
    risk_flags = set(payload.get("risk_flags", []))
    open_receipt_flags: set[str] = set()
    for row in receipt_rows:
        if not isinstance(row, dict):
            fail(f"external validation audit receipt row is not an object: {row}")
        flag = str(row.get("risk_flag", "")).strip()
        if row.get("ready") is True:
            if flag:
                fail(f"ready external receipt still carries a risk flag: {row}")
        else:
            if not flag:
                fail(f"open external receipt missing risk flag: {row}")
            open_receipt_flags.add(flag)
    if payload.get("external_validation_ready") is True:
        external_flags = {
            "public_release_upload_not_verified",
            "public_repository_state_not_verified",
            "external_ci_run_not_observed",
            "external_gpu_container_run_not_observed",
        }
        if risk_flags & external_flags:
            fail("external validation audit is ready but still has external risk flags")
        if open_receipt_flags:
            fail(f"external validation audit is ready but has open receipt flags: {open_receipt_flags}")
        if payload.get("top_conference_release_ready") is not True:
            fail("external validation audit ready state should imply top-conference release ready")
    else:
        if not open_receipt_flags:
            fail("external validation audit is not ready but no open receipt flags were found")
        missing_flags = sorted(open_receipt_flags.difference(risk_flags))
        if missing_flags:
            fail(f"external validation audit missing open receipt flags: {missing_flags}")
        if payload.get("top_conference_release_ready") is True:
            fail("external validation audit top-conference status should not be ready while receipts are missing")
    doc_text = (ROOT / "docs" / "external_validation_readiness_audit.md").read_text(
        encoding="utf-8"
    )
    required_phrases = [
        "External Validation Readiness Audit",
        "Current local artifact-release status: ready.",
        "Current external-validation status:",
        "Current top-conference release status:",
        "Public Repository Snapshot",
        "runs/public_repository_snapshot_audit.json",
        "docs/external_validation_receipts.json",
        "URL checks enabled:",
        "Strict Gate",
        "python scripts/audit_external_validation_readiness.py --strict",
        "This file is generated by `scripts/audit_external_validation_readiness.py`.",
    ]
    if payload.get("external_validation_ready") is not True:
        required_phrases.extend(sorted(open_receipt_flags))
    for phrase in required_phrases:
        if phrase not in doc_text:
            fail(f"external validation readiness audit markdown missing phrase: {phrase}")


def require_external_validation_receipt_template() -> None:
    payload = load_json(ROOT / "runs" / "external_validation_receipt_template.json")
    if payload.get("external_validation_receipt_template_ready") is not True:
        fail(f"external validation receipt template should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"external validation receipt template has risk flags: {payload.get('risk_flags')}")
    archive = load_json(ROOT / "runs" / "public_release_archive_audit.json")
    snapshot = load_json(ROOT / "runs" / "public_repository_snapshot_audit.json")
    facts = payload.get("local_facts", {})
    if facts.get("archive_sha256") != archive.get("archive_sha256"):
        fail("external validation receipt template archive SHA256 does not match archive audit")
    source_commit = snapshot.get("git", {}).get("commit")
    if facts.get("source_repository_commit") != source_commit:
        fail("external validation receipt template source commit does not match snapshot audit")
    template = payload.get("receipt_template", {})
    receipts = template.get("receipts", {}) if isinstance(template, dict) else {}
    expected = {
        "public_release_upload",
        "public_repository",
        "external_ci",
        "external_gpu_container",
    }
    if set(receipts) != expected:
        fail(f"external validation receipt template receipts changed: {set(receipts)}")
    if receipts["public_release_upload"].get("artifact_sha256") != archive.get("archive_sha256"):
        fail("external validation receipt template has stale archive SHA256")
    for key in ["public_repository", "external_ci", "external_gpu_container"]:
        if receipts[key].get("commit") != source_commit:
            fail(f"external validation receipt template has stale source commit for {key}")
    for key, row in receipts.items():
        if row.get("status") != "pending":
            fail(f"external validation receipt template should not mark observed evidence: {key}")
    doc_text = (ROOT / "docs" / "external_validation_receipt_template.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "External Validation Receipt Template",
        "Template status: ready.",
        "Archive SHA256",
        "Source snapshot commit",
        "Prefilled Fields",
        "Manual Fields",
        "Receipt Update Helper",
        "scripts/update_external_validation_receipts.py --require-all",
        "Final Validation",
        "runs/external_validation_receipt_template.json",
        "scripts/build_external_validation_receipt_template.py",
    ]:
        if phrase not in doc_text:
            fail(f"external validation receipt template markdown missing phrase: {phrase}")


def require_external_validation_runbook() -> None:
    payload = load_json(ROOT / "runs" / "external_validation_runbook.json")
    if payload.get("external_validation_runbook_ready") is not True:
        fail(f"external validation runbook should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"external validation runbook has risk flags: {payload.get('risk_flags')}")
    archive = load_json(ROOT / "runs" / "public_release_archive_audit.json")
    snapshot = load_json(ROOT / "runs" / "public_repository_snapshot_audit.json")
    facts = payload.get("local_facts", {})
    if facts.get("archive_sha256") != archive.get("archive_sha256"):
        fail("external validation runbook archive SHA256 does not match archive audit")
    if facts.get("source_repository_commit") != snapshot.get("git", {}).get("commit"):
        fail("external validation runbook source commit does not match snapshot audit")
    required = set(payload.get("required_external_receipts", []))
    expected = {
        "public_release_upload",
        "public_repository",
        "external_ci",
        "external_gpu_container",
    }
    if required != expected:
        fail(f"external validation runbook required receipts changed: {required}")
    commands = payload.get("commands", {})
    facts = payload.get("local_facts", {})
    if facts.get("receipt_template") != "docs/external_validation_receipt_template.md":
        fail("external validation runbook missing receipt template fact")
    for section in [
        "local_preflight",
        "archive_upload",
        "source_repository_publish",
        "external_ci",
        "external_gpu_container",
        "receipt_registry_update",
        "final_gate",
    ]:
        if not isinstance(commands, dict) or not commands.get(section):
            fail(f"external validation runbook missing commands section: {section}")
    doc_text = (ROOT / "docs" / "external_validation_runbook.md").read_text(
        encoding="utf-8"
    )
    for phrase in [
        "External Validation Runbook",
        "Runbook status: ready.",
        "Archive SHA256",
        "Source snapshot commit",
        "Required Receipts",
        "Local Preflight",
        "Archive Upload",
        "Source Repository Publish",
        "External CI",
        "External GPU Container",
        "Receipt Registry Update",
        "Final Gate",
        "docs/external_validation_receipts.json",
        "docs/external_validation_receipt_template.md",
        "make check",
        "make container-check",
        "scripts/build_external_validation_receipt_template.py",
        "scripts/update_external_validation_receipts.py --require-all",
        "git push -u origin main",
        "make gpu-container-env-check",
        "scripts/audit_external_validation_readiness.py --strict",
        "This file is generated by `scripts/build_external_validation_runbook.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"external validation runbook markdown missing phrase: {phrase}")


def require_submission_handoff() -> None:
    payload = load_json(ROOT / "runs" / "submission_handoff.json")
    if payload.get("submission_handoff_ready") is not True:
        fail(f"submission handoff should be ready: {payload.get('risk_flags')}")
    if payload.get("risk_flags") != []:
        fail(f"submission handoff has risk flags: {payload.get('risk_flags')}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        fail("submission handoff missing metadata payload")
    if "Winning Tickets Are Not Posterior Modes" not in str(metadata.get("title", "")):
        fail("submission handoff title does not match paper title")
    abstract_words = int(metadata.get("abstract_words", 0))
    if abstract_words <= 0 or abstract_words > 250:
        fail(f"submission handoff abstract word count invalid: {abstract_words}")
    archive = load_json(ROOT / "runs" / "public_release_archive_audit.json")
    snapshot = load_json(ROOT / "runs" / "public_repository_snapshot_audit.json")
    supplement = payload.get("supplement_files", {})
    if supplement.get("artifact_archive_sha256") != archive.get("archive_sha256"):
        fail("submission handoff archive SHA256 does not match archive audit")
    if supplement.get("source_repository_snapshot_commit") != snapshot.get("git", {}).get("commit"):
        fail("submission handoff source commit does not match snapshot audit")
    if supplement.get("external_receipt_template") != "docs/external_validation_receipt_template.md":
        fail("submission handoff missing external receipt template reference")
    commands = payload.get("check_commands", [])
    for command in [
        "make check",
        "make paper-neurips-check",
        "make container-check",
        "make external-validation-readiness",
    ]:
        if command not in commands:
            fail(f"submission handoff missing local check command: {command}")
    doc_text = (ROOT / "docs" / "submission_handoff.md").read_text(encoding="utf-8")
    for phrase in [
        "Submission Handoff",
        "Handoff status: ready.",
        "Submission Metadata",
        "Winning Tickets Are Not Posterior Modes",
        "Venue Metrics",
        "Files To Upload Or Reference",
        "paper/neurips_submission.pdf",
        "dist/lottery_artifact_public_release_2026-05-06.tar.gz",
        "docs/external_validation_receipt_template.md",
        "scripts/update_external_validation_receipts.py",
        "Local Checks",
        "Final External Gate",
        "Release Blockers",
        "This file is generated by `scripts/build_submission_handoff.py`.",
    ]:
        if phrase not in doc_text:
            fail(f"submission handoff markdown missing phrase: {phrase}")


def main(*, release_package_mode: bool = False) -> None:
    required_files = [
        ("proposal_A3_lottery_ticket_bayesian_modes.md", 1000),
        ("README.md", 1000),
        ("Makefile", 100),
        (".dockerignore", 20),
        (".gitignore", 20),
        ("Dockerfile", 500),
        ("Dockerfile.gpu", 500),
        ("LICENSE", 500),
        ("requirements.txt", 20),
        ("requirements-ci.txt", 20),
        ("requirements-gpu-lock.txt", 50),
        ("requirements-lock.txt", 50),
        ("docs/container_lock.md", 1000),
        ("docs/gpu_training_container.md", 1000),
        ("docs/local_gpu_container_validation.md", 500),
        ("docs/compute_resource_accounting.md", 1000),
        ("docs/asset_license_inventory.md", 1000),
        ("docs/new_asset_inventory.md", 1000),
        ("docs/cifar10_resnet20_full_covariance_feasibility.md", 1000),
        ("docs/digits_fullnet_laplace_tiny_r2_p0p3.md", 500),
        ("docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md", 500),
        ("docs/linear_connectivity_barrier_audit.md", 500),
        (
            "docs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3.md",
            800,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md",
            500,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md",
            1000,
        ),
        ("docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md", 1000),
        ("docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md", 1000),
        ("docs/mode_ticket_artifact_storage_budget.md", 1000),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md",
            1000,
        ),
        ("docs/resnet_channel_permutation_exhaustive_feasibility_audit.md", 1000),
        ("docs/cifar10_subset_hard_concrete_mask_training_smoke.md", 500),
        ("docs/environment_snapshot.md", 500),
        ("docs/environment_lock.json", 200),
        ("docs/mode_ticket_alignment_artifact_audit.md", 1000),
        ("docs/reproducibility_manifest.md", 1000),
        ("docs/public_release_manifest.md", 1000),
        ("docs/release_anonymization_audit.md", 500),
        ("docs/paper_claim_ledger.md", 1000),
        ("docs/paper_submission_shape_audit.md", 1000),
        ("docs/submission_pdf_shape_audit.md", 500),
        ("docs/venue_submission_compliance_audit.md", 500),
        ("docs/reviewer_objection_matrix.md", 1000),
        ("docs/submission_readiness_audit.md", 1000),
        ("docs/thread_goal_completion_audit.md", 1000),
        ("docs/paper_stats.md", 1000),
        ("runs/mode_ticket_alignment_artifact_audit.json", 1000),
        ("runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json", 1000),
        ("runs/mode_ticket_artifact_storage_budget.json", 1000),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json",
            10000,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json",
            20000,
        ),
        ("runs/resnet_channel_permutation_exhaustive_feasibility_audit.json", 5000),
        ("runs/paper_submission_shape_audit.json", 1000),
        ("runs/submission_pdf_shape_audit.json", 250),
        ("runs/venue_submission_compliance_audit.json", 500),
        ("runs/local_gpu_container_validation.json", 500),
        ("runs/reviewer_objection_matrix.json", 1000),
        ("runs/paper_stats.json", 1000),
        ("runs/cifar10_resnet20_full_covariance_feasibility.json", 1000),
        ("runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv", 500),
        ("runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv", 500),
        ("runs/linear_connectivity_barrier_audit.csv", 500),
        ("runs/linear_connectivity_barrier_audit.json", 500),
        (
            "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv",
            100,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv",
            1000,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/metrics.json",
            50000,
        ),
        (
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz",
            100_000_000,
        ),
        ("runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv", 100),
        ("runs/cifar10_subset_hard_concrete_mask_training_smoke_summary.csv", 100),
        ("runs/public_release_manifest.json", 1000),
        ("runs/release_anonymization_audit.json", 300),
        ("scripts/audit_release_anonymization.py", 1000),
        ("scripts/audit_external_validation_readiness.py", 1000),
        ("scripts/build_external_validation_receipt_template.py", 1000),
        ("scripts/update_external_validation_receipts.py", 1000),
        ("scripts/build_external_validation_runbook.py", 1000),
        ("scripts/build_submission_handoff.py", 1000),
        ("scripts/stage_public_repository_snapshot.py", 1000),
        ("scripts/smoke_public_repository_snapshot.py", 1000),
        ("scripts/verify_source_repository_snapshot.py", 1000),
        ("scripts/build_public_release_archive.py", 1000),
        ("scripts/smoke_public_release_archive.py", 1000),
        (
            "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3.md",
            1000,
        ),
        (
            "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md",
            1000,
        ),
        ("paper/main.tex", 1000),
        ("paper/refs.bib", 1000),
        ("paper/main.pdf", 200_000),
        ("paper/main_submission.pdf", 100_000),
        ("paper/neurips_2026.sty", 1000),
        ("paper/neurips_checklist.tex", 1000),
        ("paper/neurips_submission.pdf", 100_000),
        ("paper/tables/statistical_summary.tex", 1000),
    ]
    if not release_package_mode:
        required_files.extend(
            [
                ("docs/public_release_archive_audit.md", 500),
                ("docs/public_release_archive_smoke.md", 500),
                ("runs/public_release_archive_audit.json", 300),
                ("runs/public_release_archive_smoke.json", 300),
                ("docs/public_repository_snapshot_audit.md", 500),
                ("runs/public_repository_snapshot_audit.json", 500),
                ("docs/public_repository_snapshot_smoke.md", 500),
                ("runs/public_repository_snapshot_smoke.json", 500),
                ("docs/external_validation_receipts.json", 500),
                ("docs/external_validation_readiness_audit.md", 500),
                ("runs/external_validation_readiness_audit.json", 500),
                ("docs/external_validation_receipt_template.md", 500),
                ("runs/external_validation_receipt_template.json", 500),
                ("docs/external_validation_runbook.md", 500),
                ("runs/external_validation_runbook.json", 500),
                ("docs/submission_handoff.md", 500),
                ("runs/submission_handoff.json", 500),
                (
                    "dist/lottery_artifact_public_release_2026-05-06.tar.gz",
                    100_000_000,
                ),
                (
                    "dist/lottery_artifact_public_release_2026-05-06.tar.gz.sha256",
                    50,
                ),
            ]
        )
    for path, min_size in required_files:
        require_file(path, min_size)
    stats = load_json(ROOT / "runs" / "paper_stats.json")
    require_stats_sections(stats)
    require_gate1(stats)
    require_movement(stats)
    require_block_laplace(stats)
    require_subspace_hmc(stats)
    require_mode_distribution_audit(stats)
    require_direct_mode_ticket(stats)
    require_mode_ticket_alignment_artifact_audit()
    require_mode_ticket_mask_artifact_smoke()
    require_mask_artifact_posthoc_audit()
    require_mode_ticket_artifact_storage_budget()
    require_full_data_saved_artifact_posthoc_audit()
    require_full_data_global_channel_permutation_audit()
    require_exhaustive_channel_permutation_feasibility_audit()
    require_digits_fullnet_laplace_probe()
    require_fake_resnet_fullnet_laplace_smoke()
    require_linear_connectivity_barrier_audit()
    require_calibration_and_learned_masks(stats)
    require_residual_process(stats)
    require_text_evidence()
    require_bibliography()
    require_paper_numeric_claims(stats)
    require_environment_lock()
    require_full_covariance_feasibility()
    require_container_lock()
    require_claim_ledger()
    require_reviewer_objection_matrix()
    require_paper_submission_shape_audit()
    require_submission_pdf_shape_audit()
    require_venue_submission_compliance_audit()
    require_release_metadata_docs()
    require_release_manifest()
    require_release_anonymization_audit()
    if not release_package_mode:
        require_public_release_archive()
        require_public_release_archive_smoke()
        require_public_repository_snapshot_audit()
        require_public_repository_snapshot_smoke()
        require_external_validation_readiness_audit()
        require_external_validation_receipt_template()
        require_external_validation_runbook()
        require_submission_handoff()
    print("verified research artifacts: core paper evidence and generated stats are present")


if __name__ == "__main__":
    try:
        main(release_package_mode=parse_args().release_package_mode)
    except AssertionError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
