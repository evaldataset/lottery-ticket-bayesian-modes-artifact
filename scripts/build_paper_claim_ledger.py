#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fail(message: str) -> None:
    raise AssertionError(message)


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if not finite(value):
        return "n/a"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) < 0.001 and number != 0.0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def summary_mean(stats: dict[str, Any], label: str) -> dict[str, Any]:
    for row in stats["gate1"]:
        if row.get("label") == label:
            return row
    fail(f"missing Gate1 summary: {label}")


def first_matching(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    fail(f"missing row with criteria: {criteria}")


def metric(summary: dict[str, Any]) -> float:
    value = summary.get("mean")
    if not finite(value):
        fail(f"non-finite metric mean: {summary}")
    return float(value)


def audit_row(audit: dict[str, Any], label: str) -> dict[str, Any]:
    for row in audit.get("rows", []):
        if row.get("label") == label:
            return row
    fail(f"missing audit row: {label}")


def audit_metric(audit: dict[str, Any], label: str, metric_name: str) -> float:
    row = audit_row(audit, label)
    metrics = row.get("metrics", {})
    if metric_name not in metrics:
        fail(f"missing audit metric {metric_name}: {label}")
    return metric(metrics[metric_name])


def add_claim(
    claims: list[dict[str, str]],
    claim: str,
    evidence: str,
    numbers: str,
    rule: str,
    status: str = "Pass",
) -> None:
    claims.append(
        {
            "claim": claim,
            "evidence": evidence,
            "numbers": numbers,
            "rule": rule,
            "status": status,
        }
    )


def build_claims(
    stats: dict[str, Any],
    feasibility: dict[str, Any],
    alignment_audit: dict[str, Any],
    mask_artifact_smoke: dict[str, Any],
    posthoc_audit: dict[str, Any],
    storage_budget: dict[str, Any],
    full_posthoc_audit: dict[str, Any],
    global_channel_audit: dict[str, Any],
    exhaustive_channel_audit: dict[str, Any],
    digits_fullnet_laplace: list[dict[str, str]],
    fake_resnet_fullnet_laplace: list[dict[str, str]],
    linear_connectivity_audit: dict[str, Any],
    posterior_covariance_audit: dict[str, Any],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []

    mnist_random = summary_mean(stats, "MNIST: posterior - random")
    mnist_chain = summary_mean(stats, "MNIST: posterior - chain-start")
    mnist_dense = summary_mean(stats, "MNIST: dense magnitude - posterior")
    fashion_random = summary_mean(stats, "Fashion-MNIST: posterior - random")
    fashion_chain = summary_mean(stats, "Fashion-MNIST: posterior - chain-start")
    fashion_dense = summary_mean(
        stats, "Fashion-MNIST: dense magnitude - posterior"
    )
    if int(mnist_random["n"]) != 20 or int(fashion_random["n"]) != 20:
        fail("Gate1 sweeps must aggregate 20 rows per dataset")
    if metric(mnist_random) <= 0.0 or metric(fashion_random) <= 0.0:
        fail("posterior masks should beat random in Gate1 sweeps")
    if abs(metric(mnist_chain)) >= 0.005 or abs(metric(fashion_chain)) >= 0.005:
        fail("posterior-chain-start Gate1 gap should stay near zero")
    if metric(mnist_dense) <= 0.0 or metric(fashion_dense) <= 0.0:
        fail("dense magnitude should dominate posterior Gate1 masks")
    add_claim(
        claims,
        "Posterior support is not random, but the random gap is explained by local magnitude controls.",
        "`runs/paper_stats.json` section `gate1`",
        (
            f"MNIST posterior-random {fmt(mnist_random['mean'])}; "
            f"MNIST posterior-chain {fmt(mnist_chain['mean'])}; "
            f"MNIST dense-posterior {fmt(mnist_dense['mean'])}. "
            f"Fashion posterior-random {fmt(fashion_random['mean'])}; "
            f"Fashion posterior-chain {fmt(fashion_chain['mean'])}; "
            f"Fashion dense-posterior {fmt(fashion_dense['mean'])}."
        ),
        "20 grouped rows per dataset; posterior-random > 0; abs(posterior-chain) < 0.005; dense-posterior > 0.",
    )

    mode_rows = stats["mode_distribution_equivalence"]
    random_rows = [row for row in mode_rows if row.get("comparison") == "posterior-random"]
    random_wins = [
        row
        for row in random_rows
        if row.get("verdict") == "posterior separates from random"
    ]
    chain_rows = [row for row in mode_rows if row.get("comparison") == "posterior-chain"]
    chain_wins = [row for row in chain_rows if float(row["delta_mean"]) > 0.005]
    rewind_rows = [row for row in mode_rows if row.get("comparison") == "posterior-rewind"]
    rewind_beats = [row for row in rewind_rows if float(row["delta_mean"]) < -0.005]
    if len(random_rows) != 59 or len(random_wins) != 58:
        fail("mode-distribution random comparison counts changed")
    if len(chain_rows) != 59 or chain_wins:
        fail("posterior should not beat chain-start by >0.005 in mode audit")
    if len(rewind_rows) != 57 or len(rewind_beats) != 55:
        fail("rewind comparison counts changed")
    add_claim(
        claims,
        "The proposal-level support-distribution audit rejects posterior-vs-ticket equivalence beyond controls.",
        "`runs/mode_distribution_equivalence_audit_summary.csv` via `runs/paper_stats.json`",
        (
            f"posterior beats random in {len(random_wins)}/{len(random_rows)} groups; "
            f"posterior beats chain-start by >0.005 in {len(chain_wins)}/{len(chain_rows)} groups; "
            f"rewind beats posterior by >0.005 in {len(rewind_beats)}/{len(rewind_rows)} groups."
        ),
        "Exact grouped counts must remain 58/59, 0/59, and 55/57.",
    )

    direct_rows = stats["direct_mode_ticket_distribution"]
    full = first_matching(
        direct_rows,
        setting="CIFAR full ResNet",
        comparison="posterior_samples_vs_tickets",
    )
    aligned = first_matching(
        direct_rows,
        setting="CIFAR full aligned",
        comparison="activation_aligned_posterior_samples_vs_tickets",
    )
    weight_aligned = first_matching(
        direct_rows,
        setting="CIFAR full weight-aligned",
        comparison="weight_aligned_posterior_samples_vs_tickets",
    )
    csgld = first_matching(
        direct_rows,
        setting="CIFAR full cSGLD multi-chain",
        comparison="posterior_samples_vs_tickets",
    )
    csgld_independent = first_matching(
        direct_rows,
        setting="CIFAR full cSGLD independent",
        comparison="posterior_samples_vs_tickets",
    )
    lowrank_direct = first_matching(
        direct_rows,
        setting="CIFAR full LowRank128Lap",
        comparison="posterior_samples_vs_tickets",
    )
    jointdiag_direct = first_matching(
        direct_rows,
        setting="CIFAR full JointDiagLap270k",
        comparison="posterior_samples_vs_tickets",
    )
    for row in [full, aligned, weight_aligned, csgld, csgld_independent]:
        if int(float(row["posterior_num_clusters"])) != 1:
            fail(f"direct row should collapse to one posterior cluster: {row['setting']}")
        if float(row["layer_ks_pvalue"]) >= 0.001:
            fail(f"direct row should fail layer KS strongly: {row['setting']}")
        if float(row["hamming_overlap"]) >= 0.70:
            fail(f"direct row should fail hamming threshold: {row['setting']}")
        if float(row["logit_cka_hungarian_mean"]) <= 0.85:
            fail(f"direct row should keep high logit CKA: {row['setting']}")
    if int(float(lowrank_direct["posterior_num_clusters"])) != 1:
        fail("LowRank128Lap direct row should collapse to one posterior cluster")
    if int(float(lowrank_direct["left_count"])) != 50:
        fail("LowRank128Lap direct row must contain 50 posterior samples")
    if float(lowrank_direct["layer_ks_pvalue"]) >= 0.001:
        fail("LowRank128Lap direct row should still fail layer KS strongly")
    if float(lowrank_direct["hamming_overlap"]) <= 0.70:
        fail("LowRank128Lap direct row should pass the Hamming-overlap threshold")
    if float(lowrank_direct["logit_cka_hungarian_mean"]) <= 0.85:
        fail("LowRank128Lap direct row should keep high logit CKA")
    if float(lowrank_direct["activation_cka_hungarian_mean"]) <= 0.85:
        fail("LowRank128Lap direct row should keep high activation CKA")
    if int(float(jointdiag_direct["posterior_num_clusters"])) != 1:
        fail("JointDiagLap270k direct row should collapse to one posterior cluster")
    if int(float(jointdiag_direct["left_count"])) != 25:
        fail("JointDiagLap270k direct row must contain 25 posterior samples")
    if float(jointdiag_direct["layer_ks_pvalue"]) >= 0.001:
        fail("JointDiagLap270k direct row should strongly fail layer KS")
    if float(jointdiag_direct["hamming_overlap"]) >= 0.70:
        fail("JointDiagLap270k direct row should fail the Hamming-overlap threshold")
    if float(jointdiag_direct["logit_cka_hungarian_mean"]) <= 0.85:
        fail("JointDiagLap270k direct row should keep high logit CKA")
    if float(jointdiag_direct["activation_cka_hungarian_mean"]) <= 0.85:
        fail("JointDiagLap270k direct row should keep high activation CKA")
    if int(float(csgld["left_count"])) != 75:
        fail("multi-chain cSGLD direct row must contain 75 posterior samples")
    if int(float(csgld_independent["left_count"])) != 75:
        fail("independent-start cSGLD direct row must contain 75 posterior samples")
    add_claim(
        claims,
        "Full-data CIFAR direct mode/ticket probes do not satisfy full mask-distribution equivalence even when function-space CKA passes.",
        "`runs/*mode_ticket_distribution*_summary.csv` via `runs/paper_stats.json`",
        (
            f"Unaligned p={fmt(full['layer_ks_pvalue'])}, hamming={fmt(full['hamming_overlap'])}, "
            f"logit CKA={fmt(full['logit_cka_hungarian_mean'])}; "
            f"aligned p={fmt(aligned['layer_ks_pvalue'])}, hamming={fmt(aligned['hamming_overlap'])}; "
            f"weight-aligned p={fmt(weight_aligned['layer_ks_pvalue'])}, "
            f"hamming={fmt(weight_aligned['hamming_overlap'])}; "
            f"multi-chain samples={int(float(csgld['left_count']))}, p={fmt(csgld['layer_ks_pvalue'])}, "
            f"hamming={fmt(csgld['hamming_overlap'])}; "
            f"independent-start samples={int(float(csgld_independent['left_count']))}, "
            f"p={fmt(csgld_independent['layer_ks_pvalue'])}, "
            f"hamming={fmt(csgld_independent['hamming_overlap'])}; "
            f"LowRank128Lap p={fmt(lowrank_direct['layer_ks_pvalue'])}, "
            f"hamming={fmt(lowrank_direct['hamming_overlap'])}, "
            f"logit/activation CKA={fmt(lowrank_direct['logit_cka_hungarian_mean'])}/"
            f"{fmt(lowrank_direct['activation_cka_hungarian_mean'])}; "
            f"JointDiagLap270k samples={int(float(jointdiag_direct['left_count']))}, "
            f"p={fmt(jointdiag_direct['layer_ks_pvalue'])}, "
            f"hamming={fmt(jointdiag_direct['hamming_overlap'])}, "
            f"logit/activation CKA={fmt(jointdiag_direct['logit_cka_hungarian_mean'])}/"
            f"{fmt(jointdiag_direct['activation_cka_hungarian_mean'])}."
        ),
        "SGLD/aligned/dense-start and independent-start cyclical-SGLD direct CIFAR sample rows must have one posterior cluster, layer KS p < 0.001, hamming overlap < 0.70, and logit CKA > 0.85; LowRank128Lap must still have one posterior cluster and layer KS p < 0.001 while passing Hamming and CKA thresholds; JointDiagLap270k must contain 25 samples, one posterior cluster, layer KS p < 0.001, hamming overlap < 0.70, and high CKA.",
    )
    add_claim(
        claims,
        "A 270,896-parameter direct joint-group Laplace probe closes the direct full-weight posterior gap without rescuing mask-distribution equivalence.",
        "`runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"{int(float(jointdiag_direct['left_count']))} samples versus "
            f"{int(float(jointdiag_direct['right_count']))} tickets; "
            f"clusters={int(float(jointdiag_direct['posterior_num_clusters']))}; "
            f"layer KS p={fmt(jointdiag_direct['layer_ks_pvalue'])}; "
            f"hamming={fmt(jointdiag_direct['hamming_overlap'])}; "
            f"logit/activation CKA={fmt(jointdiag_direct['logit_cka_hungarian_mean'])}/"
            f"{fmt(jointdiag_direct['activation_cka_hungarian_mean'])}."
        ),
        "The direct exact joint-group row must use 25 posterior samples, fail layer KS and Hamming-overlap thresholds, keep high CKA, and collapse to one parameter-PCA basin.",
    )

    alignment_overall = alignment_audit.get("overall", {})
    if int(alignment_overall.get("run_count", 0)) != 7:
        fail("alignment artifact audit must cover seven full-data direct runs")
    if alignment_overall.get("aligned_rows_all_fail_layer_ks") is not True:
        fail("alignment artifact audit should record aligned layer-KS failures")
    if alignment_overall.get("aligned_rows_all_fail_hamming_overlap") is not True:
        fail("alignment artifact audit should record aligned Hamming failures")
    if alignment_overall.get("any_direct_equivalence_pass") is not False:
        fail("alignment artifact audit should reject all audited direct equivalence rows")
    if alignment_overall.get("raw_mask_artifacts_present") is not False:
        fail("alignment artifact audit should record no raw mask/state artifacts")
    if alignment_overall.get("posthoc_exhaustive_permutation_supported") is not False:
        fail("alignment artifact audit should bound post-hoc exhaustive permutation support")
    smoke_artifact = mask_artifact_smoke.get("mask_artifacts", {})
    if int(smoke_artifact.get("parameter_count", 0)) != 4350:
        fail("mask-artifact smoke should cover 4,350 fake-CIFAR parameters")
    if smoke_artifact.get("save_states") is not True:
        fail("mask-artifact smoke should include state matrices")
    if len(smoke_artifact.get("collections", [])) != 8:
        fail("mask-artifact smoke should include eight raw/aligned collections")
    posthoc_overall = posthoc_audit.get("overall", {})
    if posthoc_overall.get("record_level_posthoc_matching_supported") is not True:
        fail("mask-artifact post-hoc audit should support record-level matching")
    if posthoc_overall.get("local_channel_permutation_matching_supported") is not True:
        fail("mask-artifact post-hoc audit should support local channel matching")
    if posthoc_overall.get("exhaustive_graph_channel_permutation_supported") is not False:
        fail("mask-artifact post-hoc audit should leave exhaustive permutation open")
    if int(posthoc_overall.get("parameter_count", 0)) != 4350:
        fail("mask-artifact post-hoc audit should cover 4,350 fake-CIFAR parameters")
    if int(posthoc_overall.get("resnet_channel_key_count", 0)) != 19:
        fail("mask-artifact post-hoc audit should expose 19 ResNet channel keys")
    posthoc_comparisons = posthoc_audit.get("comparisons", [])
    if len(posthoc_comparisons) != 8:
        fail("mask-artifact post-hoc audit should include eight comparisons")
    if int(storage_budget.get("parameter_count", 0)) != 270896:
        fail("mask-artifact storage budget should cover 270,896 full-data weights")
    budget_recommended = storage_budget.get("recommended_next_rerun", {})
    if budget_recommended.get("scenario") != "sgld_activation_aligned_save_states":
        fail("mask-artifact storage budget should recommend activation-aligned SGLD")
    budget_mib = float(budget_recommended.get("estimated_total_mib_uncompressed", 0.0))
    if budget_mib < 280.0:
        fail("mask-artifact storage budget should estimate a nontrivial full-data footprint")
    full_posthoc_overall = full_posthoc_audit.get("overall", {})
    if full_posthoc_overall.get("dataset") != "cifar10":
        fail("full-data post-hoc audit should target CIFAR-10")
    if int(full_posthoc_overall.get("parameter_count", 0)) != 270896:
        fail("full-data post-hoc audit should cover 270,896 parameters")
    if full_posthoc_overall.get("record_level_posthoc_matching_supported") is not True:
        fail("full-data post-hoc audit should support record-level matching")
    if full_posthoc_overall.get("local_channel_permutation_matching_supported") is not False:
        fail("full-data post-hoc audit should not claim capped local channel matching")
    if int(full_posthoc_overall.get("channel_permutation_skipped_count", 0)) != 7:
        fail("full-data post-hoc audit should record skipped channel comparisons")
    full_posthoc_comparisons = full_posthoc_audit.get("comparisons", [])
    if len(full_posthoc_comparisons) != 7:
        fail("full-data post-hoc audit should include seven retained comparisons")
    global_channel_overall = global_channel_audit.get("overall", {})
    if global_channel_overall.get("dataset") != "cifar10":
        fail("global channel audit should target CIFAR-10")
    if int(global_channel_overall.get("parameter_count", 0)) != 270896:
        fail("global channel audit should cover 270,896 parameters")
    if global_channel_overall.get("global_channel_coordinate_descent_supported") is not True:
        fail("global channel audit should support coordinate descent")
    if global_channel_overall.get("exhaustive_graph_channel_permutation_supported") is not False:
        fail("global channel audit should not claim exhaustive search")
    global_rows = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in global_channel_audit.get("comparisons", [])
        if isinstance(row, dict)
    }
    global_raw = global_rows.get(("posterior_sample", "ticket"))
    global_aligned = global_rows.get(
        ("activation_aligned_posterior_sample", "activation_aligned_ticket")
    )
    if global_raw is None or global_aligned is None:
        fail("global channel audit missing posterior/ticket comparisons")
    global_raw_hamming = float(global_raw["global_channel_hamming"]["mean"])
    global_aligned_hamming = float(global_aligned["global_channel_hamming"]["mean"])
    global_raw_overlap = float(global_raw["global_channel_support_overlap_min"]["mean"])
    if global_raw_hamming <= 0.20 or global_aligned_hamming <= 0.20:
        fail("global channel audit should remain far from ticket agreement")
    exhaustive_overall = exhaustive_channel_audit.get("overall", {})
    if exhaustive_overall.get("stage1_exact_enumeration_supported") is not True:
        fail("exhaustive channel audit should support exact stage-1 enumeration")
    if int(exhaustive_overall.get("stage1_exact_permutation_count", 0)) != 128:
        fail("exhaustive channel audit should enumerate 128 stage-1 assignments")
    if exhaustive_overall.get("stage1_coordinate_descent_all_exact") is not True:
        fail("coordinate channel audit should match exact stage-1 enumeration")
    if exhaustive_overall.get("full_exhaustive_channel_permutation_supported") is not False:
        fail("exhaustive channel audit should not claim full-data exhaustive support")
    full_log10_assignments = float(
        exhaustive_overall.get("full_log10_permutation_count", 0.0)
    )
    if full_log10_assignments <= 800.0:
        fail("full-data channel search space should be astronomically large")
    exact_stage1_rows = {
        (str(row.get("left")), str(row.get("right"))): row
        for row in exhaustive_channel_audit.get("exact_stage1_comparisons", [])
        if isinstance(row, dict)
    }
    ticket_frame = exact_stage1_rows.get(("ticket", "activation_aligned_ticket"))
    if ticket_frame is None:
        fail("exhaustive channel audit missing ticket raw-vs-aligned comparison")
    if float(ticket_frame["exact_global_hamming"]["mean"]) != 0.0:
        fail("exact stage-1 ticket frame comparison should align perfectly")
    add_claim(
        claims,
        "First-order channel alignment is negative; the full-data saved-artifact rerun now supports record-level post-hoc matching, while post-hoc exhaustive graph/permutation realignment remains open.",
        "`runs/mode_ticket_alignment_artifact_audit.json`; `runs/fake_cifar10_mode_ticket_mask_artifact_smoke/*/mask_artifacts.npz`; `runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json`; `runs/mode_ticket_artifact_storage_budget.json`; `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz`; `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json`; `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json`; `runs/resnet_channel_permutation_exhaustive_feasibility_audit.json`",
        (
            f"audited direct runs={int(alignment_overall['run_count'])}; "
            "activation/weight aligned rows both fail layer-KS and Hamming; "
            f"raw mask/state files={int(alignment_overall['raw_mask_or_state_file_count'])}; "
            "post-hoc exhaustive permutation supported=False; "
            f"mask-artifact smoke parameters={int(smoke_artifact['parameter_count'])}; "
            f"collections={len(smoke_artifact['collections'])}; save_states=True; "
            f"posthoc comparisons={len(posthoc_comparisons)}; "
            "record-level posthoc matching=True; local channel matching=True; "
            "channel keys=19; exhaustive graph/channel permutation=False; "
            f"full-data saved-artifact budget={budget_mib:.2f} MiB; "
            f"full-data posthoc comparisons={len(full_posthoc_comparisons)}; "
            "full-data record-level posthoc=True; "
            f"full-data channel skipped={int(full_posthoc_overall['channel_permutation_skipped_count'])}; "
            "full-data local channel=False; "
            f"global-channel hamming raw/aligned={global_raw_hamming:.4f}/"
            f"{global_aligned_hamming:.4f}; "
            f"global-channel support overlap={global_raw_overlap:.4f}; "
            f"exact stage1 assignments={int(exhaustive_overall['stage1_exact_permutation_count'])}; "
            f"stage1 coordinate exact={exhaustive_overall['stage1_coordinate_descent_all_exact']}; "
            f"full channel assignments log10={full_log10_assignments:.1f}."
        ),
        "Seven full-data direct rows and the saved-artifact rerun both fail the proposal mask-distribution thresholds; fake-CIFAR validates local and exact small-subgraph channel post-hoc code, the full-data artifact validates record-level matching, and the block-coordinate global channel audit still leaves posterior/ticket Hamming near 0.21 rather than ticket-like agreement. Exhaustive full-data graph isomorphism remains infeasible and unimplemented.",
        status="Open limitation, bounded",
    )

    lowrank = None
    lowrank32 = None
    lowrank64 = None
    lowrank128 = None
    for row in stats["movement"]:
        if row.get("sampler") == "LowRankLap" and abs(float(row.get("scale")) - 0.01) < 1e-12:
            lowrank = row
        if row.get("sampler") == "LowRank32Lap" and abs(float(row.get("scale")) - 0.01) < 1e-12:
            lowrank32 = row
        if row.get("sampler") == "LowRank64Lap" and abs(float(row.get("scale")) - 0.01) < 1e-12:
            lowrank64 = row
        if row.get("sampler") == "LowRank128Lap" and abs(float(row.get("scale")) - 0.01) < 1e-12:
            lowrank128 = row
    if lowrank is None:
        fail("missing LowRankLap scale 0.01 movement row")
    if lowrank32 is None:
        fail("missing LowRank32Lap scale 0.01 movement row")
    if lowrank64 is None:
        fail("missing LowRank64Lap scale 0.01 movement row")
    if lowrank128 is None:
        fail("missing LowRank128Lap scale 0.01 movement row")
    if int(lowrank["posterior_minus_chain"]["n"]) != 5:
        fail("LowRankLap movement row must be five-seed")
    if int(lowrank32["posterior_minus_chain"]["n"]) != 5:
        fail("LowRank32Lap movement row must be five-seed")
    if int(lowrank64["posterior_minus_chain"]["n"]) != 5:
        fail("LowRank64Lap movement row must be five-seed")
    if int(lowrank128["posterior_minus_chain"]["n"]) != 5:
        fail("LowRank128Lap movement row must be five-seed")
    if metric(lowrank["posterior_minus_chain"]) >= 0.0:
        fail("LowRankLap movement row should not beat chain-start support")
    if metric(lowrank32["posterior_minus_chain"]) >= 0.0:
        fail("LowRank32Lap movement row should not beat chain-start support")
    if metric(lowrank64["posterior_minus_chain"]) >= 0.0:
        fail("LowRank64Lap movement row should not beat chain-start support")
    if metric(lowrank128["posterior_minus_chain"]) >= 0.0:
        fail("LowRank128Lap movement row should not beat chain-start support")
    add_claim(
        claims,
        "Full-network rank-16, rank-32, rank-64, and rank-128 Hessian-plus-diagonal Laplace movement rows move support but do not improve IMP alignment.",
        "`runs/cifar10_resnet20_long30_rewind1_lowrank*_laplace_movement*_summary.csv` via `runs/paper_stats.json`",
        (
            f"rank-16 posterior-chain delta {fmt(lowrank['posterior_minus_chain']['mean'])}, "
            f"post-chain overlap {fmt(lowrank['post_chain']['mean'])}, "
            f"sample accuracy {fmt(lowrank['sample_accuracy']['mean'])}; "
            f"rank-32 posterior-chain delta {fmt(lowrank32['posterior_minus_chain']['mean'])}, "
            f"post-chain overlap {fmt(lowrank32['post_chain']['mean'])}, "
            f"sample accuracy {fmt(lowrank32['sample_accuracy']['mean'])}; "
            f"rank-64 posterior-chain delta {fmt(lowrank64['posterior_minus_chain']['mean'])}, "
            f"post-chain overlap {fmt(lowrank64['post_chain']['mean'])}, "
            f"sample accuracy {fmt(lowrank64['sample_accuracy']['mean'])}; "
            f"rank-128 posterior-chain delta {fmt(lowrank128['posterior_minus_chain']['mean'])}, "
            f"post-chain overlap {fmt(lowrank128['post_chain']['mean'])}, "
            f"sample accuracy {fmt(lowrank128['sample_accuracy']['mean'])}."
        ),
        "Five seeds per rank; posterior-chain delta < 0 while support has moved from the chain start.",
    )

    fullnet_scale_row = None
    for row in digits_fullnet_laplace:
        if abs(float(row.get("full_laplace_scale", "nan")) - 1e-3) < 1e-12:
            fullnet_scale_row = row
            break
    if fullnet_scale_row is None:
        fail("missing tiny full-network dense Laplace scale 1e-3 row")
    if int(float(fullnet_scale_row["num_runs"])) != 5:
        fail("tiny full-network dense Laplace row must be five-seed")
    if int(round(float(fullnet_scale_row["parameter_count"]))) != 310:
        fail("tiny full-network dense Laplace row should cover 310 parameters")
    if float(fullnet_scale_row["examples_seen"]) < 1400:
        fail("tiny full-network dense Laplace Hessian should see the full train set")
    if float(fullnet_scale_row["sample_accuracy_mean"]) <= 0.83:
        fail("tiny full-network dense Laplace samples should remain accurate")
    if float(fullnet_scale_row["posterior_to_chain_start_magnitude_jaccard_mean"]) >= 0.85:
        fail("tiny full-network dense Laplace scale 1e-3 samples should move from chain-start support")
    if float(fullnet_scale_row["posterior_minus_chain_start_jaccard"]) >= -0.05:
        fail("tiny full-network dense Laplace scale 1e-3 should not rescue support equivalence")
    if float(fullnet_scale_row["chain_start_magnitude_to_imp_jaccard"]) <= float(
        fullnet_scale_row["posterior_jaccard_mean"]
    ):
        fail("tiny full-network dense Laplace posterior should remain below chain-start support")
    add_claim(
        claims,
        "A tiny exact dense full-network Laplace sanity check does not rescue support-equivalence when posterior samples actually move.",
        "`runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv`; `docs/digits_fullnet_laplace_tiny_r2_p0p3.md`",
        (
            f"5 seeds; fullnet scale={fmt(fullnet_scale_row['full_laplace_scale'])}; "
            f"fullnet params={int(round(float(fullnet_scale_row['parameter_count'])))}; "
            f"posterior={fmt(fullnet_scale_row['posterior_jaccard_mean'])}; "
            f"chain={fmt(fullnet_scale_row['chain_start_magnitude_to_imp_jaccard'])}; "
            f"fullnet posterior-chain={fmt(fullnet_scale_row['posterior_minus_chain_start_jaccard'])}; "
            f"fullnet post-chain={fmt(fullnet_scale_row['posterior_to_chain_start_magnitude_jaccard_mean'])}; "
            f"sample acc={fmt(fullnet_scale_row['sample_accuracy_mean'])}."
        ),
        "Five seeds; 310 trainable parameters; sample accuracy > 0.83; posterior-to-chain-start < 0.85; posterior-chain gap < -0.05; chain-start support remains closer to IMP.",
    )

    fake_resnet_row = None
    for row in fake_resnet_fullnet_laplace:
        if abs(float(row.get("full_laplace_scale", "nan")) - 1e-3) < 1e-12:
            fake_resnet_row = row
            break
    if fake_resnet_row is None:
        fail("missing fake-CIFAR ResNet full-network dense Laplace scale 1e-3 row")
    if int(float(fake_resnet_row["num_runs"])) != 5:
        fail("fake-CIFAR ResNet full-network dense Laplace row must be five-seed")
    if int(round(float(fake_resnet_row["parameter_count"]))) != 1229:
        fail("fake-CIFAR ResNet full-network dense Laplace row should cover 1,229 parameters")
    if int(round(float(fake_resnet_row["weight_parameter_count"]))) != 1121:
        fail("fake-CIFAR ResNet full-network dense Laplace row should cover 1,121 weight parameters")
    if int(round(float(fake_resnet_row["examples_seen"]))) != 16:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should use one Hessian batch")
    if float(fake_resnet_row["posterior_to_chain_start_magnitude_jaccard_mean"]) >= 0.60:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should move support at scale 1e-3")
    if float(fake_resnet_row["posterior_minus_chain_start_jaccard"]) >= -0.30:
        fail("fake-CIFAR ResNet full-network dense Laplace smoke should remain clearly below chain-start support")
    add_claim(
        claims,
        "The exact dense full-network Laplace code path also covers a convolutional ResNet smoke setting.",
        "`runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv`; `docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md`",
        (
            f"5 seeds; fake-resnet fullnet scale={fmt(fake_resnet_row['full_laplace_scale'])}; "
            f"params={int(round(float(fake_resnet_row['parameter_count'])))}; "
            f"weight params={int(round(float(fake_resnet_row['weight_parameter_count'])))}; "
            f"examples seen={int(round(float(fake_resnet_row['examples_seen'])))}; "
            f"posterior={fmt(fake_resnet_row['posterior_jaccard_mean'])}; "
            f"chain={fmt(fake_resnet_row['chain_start_magnitude_to_imp_jaccard'])}; "
            f"posterior-chain={fmt(fake_resnet_row['posterior_minus_chain_start_jaccard'])}; "
            f"post-chain={fmt(fake_resnet_row['posterior_to_chain_start_magnitude_jaccard_mean'])}."
        ),
        "Five seeds; fake-CIFAR ResNet-20 width-1; 1,229 trainable parameters; dense Cholesky is full-network; marked as code-path smoke, not real CIFAR evidence.",
    )

    linear_checks = linear_connectivity_audit.get("interpretation_checks", {})
    if len(linear_connectivity_audit.get("rows", [])) != 6:
        fail("linear connectivity barrier audit must contain six rows")
    if linear_checks.get("all_rows_five_seed") is not True:
        fail("linear connectivity barrier audit rows must all be five-seed")
    if linear_checks.get("posterior_never_beats_chain_start") is not True:
        fail("linear connectivity audit should not let posterior beat chain-start")
    if linear_checks.get("cifar_dense_imp_barriers_large") is not True:
        fail("CIFAR linear barriers should be large in the connectivity audit")
    mnist_barrier = audit_metric(
        linear_connectivity_audit, "MNIST Gate1 SGLD r5", "dense_imp_barrier"
    )
    fashion_barrier = audit_metric(
        linear_connectivity_audit,
        "Fashion-MNIST Gate1 SGLD r5",
        "dense_imp_barrier",
    )
    cifar_sgld_barrier = audit_metric(
        linear_connectivity_audit,
        "CIFAR-10 ResNet-20 long SGLD r5",
        "dense_imp_barrier",
    )
    cifar_swag_barrier = audit_metric(
        linear_connectivity_audit,
        "CIFAR-10 ResNet-20 long SWAG r5",
        "dense_imp_barrier",
    )
    cifar_sgld_gap = audit_metric(
        linear_connectivity_audit,
        "CIFAR-10 ResNet-20 long SGLD r5",
        "posterior_minus_chain_start_jaccard",
    )
    mnist_gap = audit_metric(
        linear_connectivity_audit,
        "MNIST Gate1 SGLD r5",
        "posterior_minus_chain_start_jaccard",
    )
    if mnist_barrier >= 0.01:
        fail("MNIST dense-IMP linear barrier should stay near zero")
    if fashion_barrier >= 0.05:
        fail("Fashion-MNIST dense-IMP linear barrier should stay near zero")
    if cifar_sgld_barrier <= 2.0 or cifar_swag_barrier <= 2.0:
        fail("CIFAR long-run linear barriers should remain large")
    add_claim(
        claims,
        "Linear connectivity barriers are orthogonal landscape diagnostics and do not rescue posterior-ticket support equivalence.",
        "`runs/linear_connectivity_barrier_audit.csv`; `runs/linear_connectivity_barrier_audit.json`; `docs/linear_connectivity_barrier_audit.md`",
        (
            f"6 five-seed rows; MNIST dense-IMP barrier={fmt(mnist_barrier)}; "
            f"Fashion dense-IMP barrier={fmt(fashion_barrier)}; "
            f"CIFAR long SGLD/SWAG dense-IMP barriers={fmt(cifar_sgld_barrier)}/"
            f"{fmt(cifar_swag_barrier)}; "
            f"MNIST posterior-chain={fmt(mnist_gap)}; "
            f"CIFAR long SGLD posterior-chain={fmt(cifar_sgld_gap)}."
        ),
        "Near-zero MNIST/Fashion dense-to-IMP barriers and large CIFAR barriers both coexist with posterior-chain support gaps <= 0.001, so linear barriers are not support-equivalence evidence.",
    )

    blockdiag = first_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:11blocks<=5000",
        scale=0.0001,
    )
    if int(blockdiag["block_posterior_minus_chain"]["n"]) != 5:
        fail("BlockDiagLap row must be five-seed")
    if metric(blockdiag["parameter_count"]) < 22_000:
        fail("BlockDiagLap row should cover at least 22k parameters")
    if metric(blockdiag["block_posterior_minus_chain"]) >= 0.0:
        fail("BlockDiagLap selected-block posterior should not beat chain-start")
    if metric(blockdiag["global_posterior_minus_chain"]) >= 0.01:
        fail("BlockDiagLap global posterior gain should remain small")
    if metric(blockdiag["global_rewind_minus_posterior"]) <= 0.02:
        fail("BlockDiagLap rewind support should remain clearly closer than posterior")
    if metric(blockdiag["sample_accuracy"]) <= 0.87:
        fail("BlockDiagLap samples should preserve useful accuracy")
    add_claim(
        claims,
        "A wider exact block-diagonal full-covariance Laplace probe still does not make support movement ticket-directed.",
        "`runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"11 tensors, {int(round(float(blockdiag['parameter_count']['mean']))):,} parameters; "
            f"block post-chain {fmt(blockdiag['block_post_chain']['mean'])}; "
            f"block posterior-chain {fmt(blockdiag['block_posterior_minus_chain']['mean'])}; "
            f"global posterior-chain {fmt(blockdiag['global_posterior_minus_chain']['mean'])}; "
            f"global rewind-posterior {fmt(blockdiag['global_rewind_minus_posterior']['mean'])}; "
            f"sample accuracy {fmt(blockdiag['sample_accuracy']['mean'])}."
        ),
        "Five seeds; selected-block posterior-chain < 0; global posterior-chain < 0.01; global rewind-posterior > 0.02; sample accuracy > 0.87.",
    )

    blockdiag_max10k = first_matching(
        stats["block_laplace"],
        sampler="BlockDiagLap",
        block="blockdiag:16blocks<=10000",
        scale=1e-05,
    )
    if int(blockdiag_max10k["block_posterior_minus_chain"]["n"]) != 5:
        fail("BlockDiagLap max10k row must be five-seed")
    if metric(blockdiag_max10k["parameter_count"]) < 68_000:
        fail("BlockDiagLap max10k row should cover at least 68k parameters")
    if metric(blockdiag_max10k["block_posterior_minus_chain"]) >= 0.0:
        fail("BlockDiagLap max10k selected-block posterior should not beat chain-start")
    if metric(blockdiag_max10k["global_posterior_minus_chain"]) >= 0.005:
        fail("BlockDiagLap max10k global posterior gain should remain tiny")
    if metric(blockdiag_max10k["global_rewind_minus_posterior"]) <= 0.025:
        fail("BlockDiagLap max10k rewind support should remain clearly closer than posterior")
    if metric(blockdiag_max10k["global_post_chain"]) >= 0.80:
        fail("BlockDiagLap max10k should move support away from chain-start")
    if metric(blockdiag_max10k["sample_accuracy"]) <= 0.875:
        fail("BlockDiagLap max10k samples should preserve useful accuracy")
    add_claim(
        claims,
        "An even wider 68,144-parameter exact block-diagonal Laplace row moves support but remains non-ticket-directed.",
        "`runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"16 tensors, {int(round(float(blockdiag_max10k['parameter_count']['mean']))):,} parameters; "
            f"global post-chain {fmt(blockdiag_max10k['global_post_chain']['mean'])}; "
            f"block posterior-chain {fmt(blockdiag_max10k['block_posterior_minus_chain']['mean'])}; "
            f"global posterior-chain {fmt(blockdiag_max10k['global_posterior_minus_chain']['mean'])}; "
            f"global rewind-posterior {fmt(blockdiag_max10k['global_rewind_minus_posterior']['mean'])}; "
            f"sample accuracy {fmt(blockdiag_max10k['sample_accuracy']['mean'])}."
        ),
        "Five seeds; 68k+ exact block parameters; block posterior-chain < 0; global posterior-chain < 0.005; global rewind-posterior > 0.025; global post-chain < 0.80; sample accuracy > 0.875.",
    )

    jointdiag_max10k = first_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=10000",
        scale=1e-05,
    )
    if int(jointdiag_max10k["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max10k row must be five-seed")
    if metric(jointdiag_max10k["parameter_count"]) < 68_000:
        fail("JointDiagLap max10k row should cover at least 68k parameters")
    if metric(jointdiag_max10k["block_posterior_minus_chain"]) >= 0.0:
        fail("JointDiagLap max10k selected-block posterior should not beat chain-start")
    if metric(jointdiag_max10k["global_posterior_minus_chain"]) >= 0.005:
        fail("JointDiagLap max10k global posterior gain should remain tiny")
    if metric(jointdiag_max10k["global_rewind_minus_posterior"]) <= 0.025:
        fail("JointDiagLap max10k rewind support should remain clearly closer than posterior")
    if metric(jointdiag_max10k["global_post_chain"]) >= 0.80:
        fail("JointDiagLap max10k should move support away from chain-start")
    if metric(jointdiag_max10k["sample_accuracy"]) <= 0.875:
        fail("JointDiagLap max10k samples should preserve useful accuracy")
    add_claim(
        claims,
        "A 68,144-parameter exact joint-group Laplace row adds cross-tensor covariance without rescuing ticket-directed support.",
        "`runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"8 joint groups over {int(round(float(jointdiag_max10k['parameter_count']['mean']))):,} parameters; "
            f"global post-chain {fmt(jointdiag_max10k['global_post_chain']['mean'])}; "
            f"block posterior-chain {fmt(jointdiag_max10k['block_posterior_minus_chain']['mean'])}; "
            f"global posterior-chain {fmt(jointdiag_max10k['global_posterior_minus_chain']['mean'])}; "
            f"global rewind-posterior {fmt(jointdiag_max10k['global_rewind_minus_posterior']['mean'])}; "
            f"sample accuracy {fmt(jointdiag_max10k['sample_accuracy']['mean'])}."
        ),
        "Five seeds; 68k+ exact joint-group parameters; block posterior-chain < 0; global posterior-chain < 0.005; global rewind-posterior > 0.025; global post-chain < 0.80; sample accuracy > 0.875.",
    )

    jointdiag_max20k = first_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:6groups<=20000",
        scale=3e-06,
    )
    if int(jointdiag_max20k["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max20k row must be five-seed")
    if metric(jointdiag_max20k["parameter_count"]) < 86_000:
        fail("JointDiagLap max20k row should cover at least 86k parameters")
    if metric(jointdiag_max20k["block_posterior_minus_chain"]) >= 0.0:
        fail("JointDiagLap max20k selected-block posterior should not beat chain-start")
    if metric(jointdiag_max20k["global_posterior_minus_chain"]) >= 0.005:
        fail("JointDiagLap max20k global posterior gain should remain tiny")
    if metric(jointdiag_max20k["global_rewind_minus_posterior"]) <= 0.025:
        fail("JointDiagLap max20k rewind support should remain clearly closer than posterior")
    if metric(jointdiag_max20k["global_post_chain"]) >= 0.85:
        fail("JointDiagLap max20k should move support away from chain-start")
    if metric(jointdiag_max20k["sample_accuracy"]) <= 0.875:
        fail("JointDiagLap max20k samples should preserve useful accuracy")
    add_claim(
        claims,
        "An 86,576-parameter exact joint-group Laplace row adds the first stage-3 convolution block without rescuing ticket-directed support.",
        "`runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"6 joint groups over {int(round(float(jointdiag_max20k['parameter_count']['mean']))):,} parameters; "
            f"global post-chain {fmt(jointdiag_max20k['global_post_chain']['mean'])}; "
            f"block posterior-chain {fmt(jointdiag_max20k['block_posterior_minus_chain']['mean'])}; "
            f"global posterior-chain {fmt(jointdiag_max20k['global_posterior_minus_chain']['mean'])}; "
            f"global rewind-posterior {fmt(jointdiag_max20k['global_rewind_minus_posterior']['mean'])}; "
            f"sample accuracy {fmt(jointdiag_max20k['sample_accuracy']['mean'])}."
        ),
        "Five seeds; 86k+ exact joint-group parameters; block posterior-chain < 0; global posterior-chain < 0.005; global rewind-posterior > 0.025; global post-chain < 0.85; sample accuracy > 0.875.",
    )

    jointdiag_max40k = first_matching(
        stats["block_laplace"],
        sampler="JointDiagLap",
        block="jointdiag:8groups<=40000",
        scale=1e-06,
    )
    if int(jointdiag_max40k["block_posterior_minus_chain"]["n"]) != 5:
        fail("JointDiagLap max40k row must be five-seed")
    if metric(jointdiag_max40k["parameter_count"]) < 270_000:
        fail("JointDiagLap max40k row should cover the full weight vector")
    if metric(jointdiag_max40k["block_posterior_minus_chain"]) >= 0.0:
        fail("JointDiagLap max40k selected-block posterior should not beat chain-start")
    if metric(jointdiag_max40k["global_posterior_minus_chain"]) >= 0.001:
        fail("JointDiagLap max40k global posterior gain should remain non-positive or tiny")
    if metric(jointdiag_max40k["global_rewind_minus_posterior"]) <= 0.03:
        fail("JointDiagLap max40k rewind support should remain clearly closer than posterior")
    if metric(jointdiag_max40k["global_post_chain"]) >= 0.80:
        fail("JointDiagLap max40k should move support away from chain-start")
    if metric(jointdiag_max40k["sample_accuracy"]) <= 0.875:
        fail("JointDiagLap max40k samples should preserve useful accuracy")
    add_claim(
        claims,
        "A 270,896-parameter exact joint-group Laplace row covers the full weight vector without rescuing ticket-directed support.",
        "`runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"8 streamed joint groups over {int(round(float(jointdiag_max40k['parameter_count']['mean']))):,} parameters; "
            f"global post-chain {fmt(jointdiag_max40k['global_post_chain']['mean'])}; "
            f"block posterior-chain {fmt(jointdiag_max40k['block_posterior_minus_chain']['mean'])}; "
            f"global posterior-chain {fmt(jointdiag_max40k['global_posterior_minus_chain']['mean'])}; "
            f"global rewind-posterior {fmt(jointdiag_max40k['global_rewind_minus_posterior']['mean'])}; "
            f"sample accuracy {fmt(jointdiag_max40k['sample_accuracy']['mean'])}."
        ),
        "Five seeds; 270k+ exact joint-group parameters; streamed joint groups; block posterior-chain < 0; global posterior-chain < 0.001; global rewind-posterior > 0.03; global post-chain < 0.80; sample accuracy > 0.875.",
    )

    covariance_checks = posterior_covariance_audit.get("interpretation_checks", {})
    if covariance_checks.get("posterior_covariance_robustness_ready") is not True:
        fail("posterior covariance robustness audit should be ready")
    if int(covariance_checks.get("movement_row_count", 0)) != 9:
        fail("posterior covariance robustness audit should contain nine movement rows")
    if int(covariance_checks.get("direct_row_count", 0)) != 2:
        fail("posterior covariance robustness audit should contain two direct rows")
    lowrank128_direct = audit_row(
        posterior_covariance_audit, "LowRank128 direct samples"
    )
    jointdiag270_direct = audit_row(
        posterior_covariance_audit, "JointDiag270k direct samples"
    )
    if lowrank128_direct.get("direct_equivalence_passed") is not False:
        fail("LowRank128 covariance audit direct row should fail equivalence")
    if jointdiag270_direct.get("direct_equivalence_passed") is not False:
        fail("JointDiag270k covariance audit direct row should fail equivalence")
    add_claim(
        claims,
        "The covariance-fidelity spectrum does not show a trend toward posterior-ticket support equivalence.",
        "`runs/posterior_covariance_robustness_audit.json`; `docs/posterior_covariance_robustness_audit.md`",
        (
            f"movement/direct rows={int(covariance_checks['movement_row_count'])}/"
            f"{int(covariance_checks['direct_row_count'])}; "
            f"max movement posterior-chain={fmt(covariance_checks['max_movement_posterior_minus_chain_start_jaccard'])}; "
            f"min exact rewind-posterior={fmt(covariance_checks['min_exact_rewind_minus_posterior_jaccard'])}; "
            f"min sample acc={fmt(covariance_checks['min_movement_sample_accuracy'])}; "
            f"LowRank128 direct p={fmt(lowrank128_direct['layer_ks_pvalue'])}; "
            f"JointDiag270k direct p={fmt(jointdiag270_direct['layer_ks_pvalue'])}, "
            f"hamming={fmt(jointdiag270_direct['hamming_overlap'])}."
        ),
        "Nine five-seed movement rows preserve accuracy, no movement row beats chain-start by 0.005 Jaccard, exact rows keep rewind closer than posterior, and both direct rows fail proposal-level equivalence.",
    )

    imp = first_matching(stats["calibration_ood"], source="imp")
    swag = first_matching(stats["calibration_ood"], source="swag_ensemble")
    var = first_matching(stats["calibration_ood"], source="variational_prune")
    if metric(swag["id_accuracy"]) >= metric(imp["id_accuracy"]):
        fail("SWAG ensemble should not beat IMP accuracy in current evidence")
    if metric(var["id_accuracy"]) >= metric(imp["id_accuracy"]):
        fail("variational prune should not beat IMP accuracy in current evidence")
    if metric(swag["ood_msp_auroc"]) >= metric(imp["ood_msp_auroc"]):
        fail("SWAG ensemble should not beat IMP OOD AUROC in current evidence")
    if metric(var["ood_msp_auroc"]) >= metric(imp["ood_msp_auroc"]):
        fail("variational prune should not beat IMP OOD AUROC in current evidence")
    add_claim(
        claims,
        "Calibration improvements from posterior or learned-mask rows do not rescue ticket-support performance.",
        "`runs/cifar10_resnet20_long30_rewind1_calibration_ood_*_summary.csv` via `runs/paper_stats.json`",
        (
            f"IMP acc/AUROC {fmt(imp['id_accuracy']['mean'])}/{fmt(imp['ood_msp_auroc']['mean'])}; "
            f"SWAG acc/AUROC {fmt(swag['id_accuracy']['mean'])}/{fmt(swag['ood_msp_auroc']['mean'])}; "
            f"variational acc/AUROC {fmt(var['id_accuracy']['mean'])}/{fmt(var['ood_msp_auroc']['mean'])}."
        ),
        "Five seeds; SWAG and variational-prune rows remain below IMP on accuracy and CIFAR-100 MSP AUROC.",
    )

    gem = first_matching(stats["trajectory_mask_training"], source="gem_miner")
    var_support = first_matching(
        stats["trajectory_mask_training"], source="variational_prune"
    )
    hard = first_matching(stats["trajectory_mask_training"], source="hard_concrete")
    for row in [gem, var_support, hard]:
        if int(row["trained_accuracy"]["n"]) != 5:
            fail(f"learned-mask support row must be five-seed: {row['source']}")
        if metric(row["accuracy_minus_imp"]) >= 0.0:
            fail(f"learned-mask row should stay below IMP accuracy: {row['source']}")
        if metric(row["source_to_imp"]) >= 0.10:
            fail(f"learned-mask support should remain random-scale: {row['source']}")
    if metric(hard["trained_accuracy"]) >= 0.50:
        fail("hard-concrete full-data row should not appear competitive")
    add_claim(
        claims,
        "Learned-mask baselines do not recover CIFAR IMP support at full-data scale.",
        "`runs/cifar10_resnet20_long30_rewind1_*_selected_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"Gem-Miner acc/support {fmt(gem['trained_accuracy']['mean'])}/"
            f"{fmt(gem['source_to_imp']['mean'])}; "
            f"variational acc/support {fmt(var_support['trained_accuracy']['mean'])}/"
            f"{fmt(var_support['source_to_imp']['mean'])}; "
            f"hard-concrete acc/support {fmt(hard['trained_accuracy']['mean'])}/"
            f"{fmt(hard['source_to_imp']['mean'])}."
        ),
        "Five seeds per learned-mask source; each accuracy is below IMP and each support-to-IMP Jaccard stays below 0.10.",
    )

    process = stats["residual_imp_process_tensor_score_exclusion_pairs"][0]
    delta = process["round_minus_tensor_score_excluded"]
    overlap = process["tensor_score_excluded_oracle_overlap"]
    tensor_overlap = process["layer_excluded_oracle_overlap"]
    if int(delta["n"]) != 5 or int(delta["positive"]) != 5:
        fail("tensor+score process row must be 5/5 positive")
    if metric(delta) <= 0.0:
        fail("tensor+score replacement should lose to process-selected row")
    if metric(overlap) <= metric(tensor_overlap):
        fail("tensor+score replacement should improve overlap over tensor-only")
    add_claim(
        claims,
        "IMP-process-selected final-IMP residual coordinates remain functional after tensor and score-bin matching controls.",
        "`runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"round acc {fmt(process['round_accuracy']['mean'])}; "
            f"tensor+score replacement acc {fmt(process['tensor_score_excluded_accuracy']['mean'])}; "
            f"delta {fmt(delta['mean'])} with {int(delta['positive'])}/{int(delta['n'])} wins; "
            f"oracle overlap {fmt(overlap['mean'])}."
        ),
        "Five seeds; round-minus-tensor+score delta > 0; paired wins >= 4/5; tensor+score overlap > tensor-only overlap.",
    )

    projection = stats["residual_imp_process_projection_pairs"][0]
    projection_delta = projection["round_minus_residualized"]
    projection_oracle_delta = projection["round_minus_residualized_oracle"]
    if int(projection_delta["n"]) != 5 or int(projection_delta["positive"]) != 5:
        fail("residualized-score projection row must be 5/5 positive")
    if metric(projection_delta) <= 0.0:
        fail("round score should beat residualized process score")
    if metric(projection_oracle_delta) <= 0.10:
        fail("residualized process score should materially reduce oracle overlap")
    add_claim(
        claims,
        "The useful IMP round-score ordering is tied to the trajectory/final-magnitude subspace.",
        "`runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"round acc {fmt(projection['round_accuracy']['mean'])}; "
            f"residualized-score acc {fmt(projection['residualized_accuracy']['mean'])}; "
            f"delta {fmt(projection_delta['mean'])} with "
            f"{int(projection_delta['positive'])}/{int(projection_delta['n'])} wins; "
            f"oracle-overlap drop {fmt(projection_oracle_delta['mean'])}."
        ),
        "Five seeds; round-minus-residualized delta > 0; paired wins >= 4/5; oracle-overlap drop > 0.10.",
    )

    posterior_projection = stats["residual_imp_process_posterior_projection_pairs"][0]
    posterior_projection_delta = posterior_projection["round_minus_residualized"]
    posterior_projection_oracle_delta = posterior_projection[
        "round_minus_residualized_oracle"
    ]
    if (
        int(posterior_projection_delta["n"]) != 5
        or int(posterior_projection_delta["positive"]) != 5
    ):
        fail("posterior-residualized projection row must be 5/5 positive")
    if metric(posterior_projection_delta) <= 0.0:
        fail("round score should beat posterior-residualized process score")
    if metric(posterior_projection_oracle_delta) <= 0.18:
        fail("posterior-residualized score should materially reduce oracle overlap")
    add_claim(
        claims,
        "The IMP round-score ordering is not explained away by the current diagonal-Laplace posterior score subspace.",
        "`runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"round acc {fmt(posterior_projection['round_accuracy']['mean'])}; "
            f"posterior-residualized acc {fmt(posterior_projection['residualized_accuracy']['mean'])}; "
            f"delta {fmt(posterior_projection_delta['mean'])} with "
            f"{int(posterior_projection_delta['positive'])}/{int(posterior_projection_delta['n'])} wins; "
            f"oracle-overlap drop {fmt(posterior_projection_oracle_delta['mean'])}."
        ),
        "Five seeds; round-minus-posterior-residualized delta > 0; paired wins 5/5; oracle-overlap drop > 0.18.",
    )

    learned_subspace = stats["residual_imp_process_learned_subspace_pairs"][0]
    learned_delta = learned_subspace["round_minus_residualized"]
    learned_oracle_delta = learned_subspace[
        "round_minus_residualized_oracle"
    ]
    if int(learned_delta["n"]) != 5 or int(learned_delta["positive"]) != 5:
        fail("learned-subspace residualized projection row must be 5/5 positive")
    if metric(learned_delta) <= 0.0:
        fail("round score should beat learned-subspace residualized process score")
    if metric(learned_oracle_delta) <= 0.18:
        fail("learned-subspace residualized score should materially reduce oracle overlap")
    add_claim(
        claims,
        "A learned trajectory/process subspace does not replace the IMP round-selected residual coordinates.",
        "`runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv` via `runs/paper_stats.json`",
        (
            f"round acc {fmt(learned_subspace['round_accuracy']['mean'])}; "
            f"learned-subspace residualized acc {fmt(learned_subspace['residualized_accuracy']['mean'])}; "
            f"delta {fmt(learned_delta['mean'])} with "
            f"{int(learned_delta['positive'])}/{int(learned_delta['n'])} wins; "
            f"oracle-overlap drop {fmt(learned_oracle_delta['mean'])}."
        ),
        "Five seeds; round-minus-learned-subspace delta > 0; paired wins 5/5; oracle-overlap drop > 0.18.",
    )

    full_cov = feasibility["all_trainable"]
    blocks = feasibility["weight_tensor_block_diagonal"]
    if float(full_cov["dense_precision_float64_gib"]) < 500.0:
        fail("full covariance memory estimate is too small")
    if float(full_cov["precision_plus_cholesky_float64_gib"]) < 1000.0:
        fail("full covariance Cholesky memory estimate is too small")
    add_claim(
        claims,
        "Literal full-network dense full-covariance CIFAR Laplace is outside the single-workstation artifact budget.",
        "`runs/cifar10_resnet20_full_covariance_feasibility.json`",
        (
            f"all-trainable parameters {int(full_cov['parameter_count']):,}; "
            f"one dense float64 matrix {fmt(full_cov['dense_precision_float64_gib'], 1)} GiB; "
            f"matrix plus Cholesky {fmt(full_cov['precision_plus_cholesky_float64_gib'], 1)} GiB; "
            f"tensor-block matrix plus Cholesky {fmt(blocks['precision_plus_cholesky_float64_gib'], 1)} GiB."
        ),
        "Memory estimates must stay above 500 GiB for one dense matrix and 1000 GiB with Cholesky resident.",
        status="Open limitation, bounded",
    )

    return claims


def write_markdown(path: Path, claims: list[dict[str, str]]) -> None:
    lines = [
        "# Paper Claim Ledger",
        "",
        "This ledger is generated from current artifacts by",
        "`scripts/build_paper_claim_ledger.py`. It is a reviewer-facing map from",
        "the manuscript's central claims to concrete run artifacts, key numbers,",
        "and verification rules. It does not mark the project complete; it makes",
        "the remaining limitations explicit.",
        "",
        "| Claim | Evidence | Key numbers | Verification rule | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in claims:
        lines.append(
            "| {claim} | {evidence} | {numbers} | {rule} | {status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Remaining Gaps",
            "",
            "- Exact or near-exact full-network full-covariance CIFAR posterior evidence",
            "  remains the main posterior-side limitation; the feasibility audit now",
            "  bounds the literal dense option and the current covariance-fidelity",
            "  trend audit spans low-rank, exact block/joint, and direct 270k rows.",
            "- Broader learned-mask distribution and permutation-aware variants are",
            "  still useful reviewer-facing robustness work; the alignment",
            "  artifact audit now bounds the current post-hoc permutation gap.",
            "- Broader process/subspace causal interventions remain optional hardening",
            "  beyond the current tensor+score-matched round-exclusion,",
            "  residualized/posterior-residualized projection, and learned-subspace",
            "  controls.",
            "- Public archive upload, external CI/GPU-container run status, and",
            "  clean public repository state remain packaging requirements before",
            "  submission.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stats = load_json(ROOT / "runs" / "paper_stats.json")
    feasibility = load_json(
        ROOT / "runs" / "cifar10_resnet20_full_covariance_feasibility.json"
    )
    alignment_audit = load_json(ROOT / "runs" / "mode_ticket_alignment_artifact_audit.json")
    mask_smoke_paths = sorted(
        (ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_smoke").glob(
            "*/metrics.json"
        )
    )
    if not mask_smoke_paths:
        fail("missing fake-CIFAR mode/ticket mask artifact smoke metrics")
    mask_artifact_smoke = load_json(mask_smoke_paths[-1])
    posthoc_audit = load_json(
        ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json"
    )
    storage_budget = load_json(ROOT / "runs" / "mode_ticket_artifact_storage_budget.json")
    full_posthoc_audit = load_json(
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json"
    )
    global_channel_audit = load_json(
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json"
    )
    exhaustive_channel_audit = load_json(
        ROOT / "runs" / "resnet_channel_permutation_exhaustive_feasibility_audit.json"
    )
    digits_fullnet_laplace = load_csv(
        ROOT / "runs" / "digits_fullnet_laplace_tiny_r2_p0p3_summary.csv"
    )
    fake_resnet_fullnet_laplace = load_csv(
        ROOT / "runs" / "fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv"
    )
    linear_connectivity_audit = load_json(
        ROOT / "runs" / "linear_connectivity_barrier_audit.json"
    )
    posterior_covariance_audit = load_json(
        ROOT / "runs" / "posterior_covariance_robustness_audit.json"
    )
    claims = build_claims(
        stats,
        feasibility,
        alignment_audit,
        mask_artifact_smoke,
        posthoc_audit,
        storage_budget,
        full_posthoc_audit,
        global_channel_audit,
        exhaustive_channel_audit,
        digits_fullnet_laplace,
        fake_resnet_fullnet_laplace,
        linear_connectivity_audit,
        posterior_covariance_audit,
    )
    out_path = ROOT / "docs" / "paper_claim_ledger.md"
    write_markdown(out_path, claims)
    print(json.dumps({"claims": len(claims), "out_md": str(out_path)}, indent=None))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"claim ledger verification failed: {exc}")
        raise SystemExit(1)
