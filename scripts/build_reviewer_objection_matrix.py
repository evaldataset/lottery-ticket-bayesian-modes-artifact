#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "reviewer_objection_matrix.json"
DEFAULT_OUT_MD = ROOT / "docs" / "reviewer_objection_matrix.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def first_matching(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    fail(f"missing row with criteria: {criteria}")


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, {}).get("mean")
    if not finite(value):
        fail(f"missing finite metric {key}: {row}")
    return float(value)


def audit_metric(audit: dict[str, Any], label: str, metric_name: str) -> float:
    for row in audit.get("rows", []):
        if row.get("label") == label:
            value = row.get("metrics", {}).get(metric_name, {}).get("mean")
            if not finite(value):
                fail(f"missing audit metric {metric_name}: {label}")
            return float(value)
    fail(f"missing audit row: {label}")


def global_channel_row(audit: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    for row in audit.get("comparisons", []):
        if row.get("left") == left and row.get("right") == right:
            return row
    fail(f"missing global channel row: {left} vs {right}")


def add_objection(
    rows: list[dict[str, Any]],
    *,
    objection: str,
    reviewer_risk: str,
    current_answer: str,
    key_numbers: str,
    evidence: list[str],
    status: str,
    remaining_gap: str,
) -> None:
    rows.append(
        {
            "objection": objection,
            "reviewer_risk": reviewer_risk,
            "current_answer": current_answer,
            "key_numbers": key_numbers,
            "evidence": evidence,
            "status": status,
            "remaining_gap": remaining_gap,
        }
    )


def build_rows(
    stats: dict[str, Any],
    feasibility: dict[str, Any],
    covariance_audit: dict[str, Any],
    linear_audit: dict[str, Any],
    global_channel_audit: dict[str, Any],
    exhaustive_channel_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    mode_rows = stats["mode_distribution_equivalence"]
    random_rows = [row for row in mode_rows if row.get("comparison") == "posterior-random"]
    random_wins = [
        row for row in random_rows if row.get("verdict") == "posterior separates from random"
    ]
    chain_rows = [row for row in mode_rows if row.get("comparison") == "posterior-chain"]
    chain_wins = [row for row in chain_rows if float(row["delta_mean"]) > 0.005]
    rewind_rows = [row for row in mode_rows if row.get("comparison") == "posterior-rewind"]
    rewind_beats = [row for row in rewind_rows if float(row["delta_mean"]) < -0.005]
    if (len(random_wins), len(random_rows), len(chain_wins), len(chain_rows)) != (
        58,
        59,
        0,
        59,
    ):
        fail("mode-distribution objection counts changed")
    if (len(rewind_beats), len(rewind_rows)) != (55, 57):
        fail("rewind objection counts changed")
    add_objection(
        rows,
        objection="Posterior masks only need to beat random masks.",
        reviewer_risk="The positive posterior signal could be overstated if random is the only baseline.",
        current_answer=(
            "Posterior supports usually beat random, but this signal disappears "
            "against chain-start and rewind magnitude controls."
        ),
        key_numbers=(
            f"posterior>random {len(random_wins)}/{len(random_rows)} groups; "
            f"posterior>chain by >0.005 {len(chain_wins)}/{len(chain_rows)}; "
            f"rewind>posterior by >0.005 {len(rewind_beats)}/{len(rewind_rows)}."
        ),
        evidence=[
            "runs/paper_stats.json: mode_distribution_equivalence",
            "docs/paper_claim_ledger.md",
        ],
        status="Closed for current artifacts",
        remaining_gap="None for the support-equivalence claim; this remains a negative result.",
    )

    full = first_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full ResNet",
        comparison="posterior_samples_vs_tickets",
    )
    csgld = first_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full cSGLD multi-chain",
        comparison="posterior_samples_vs_tickets",
    )
    csgld_ind = first_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full cSGLD independent",
        comparison="posterior_samples_vs_tickets",
    )
    for row in [full, csgld, csgld_ind]:
        if int(float(row["posterior_num_clusters"])) != 1:
            fail("direct sampler row should collapse to one posterior cluster")
        if float(row["layer_ks_pvalue"]) >= 0.001:
            fail("direct sampler row should fail layer KS")
    add_objection(
        rows,
        objection="The posterior sampler is too weak or stuck in one chain.",
        reviewer_risk="A better-mixed sampler might reveal ticket-like posterior supports.",
        current_answer=(
            "Dense-start and independent-start multi-chain cyclical SGLD "
            "increase posterior sample count and chain diversity without "
            "recovering mask-distribution equivalence."
        ),
        key_numbers=(
            f"full SGLD p={fmt(full['layer_ks_pvalue'])}, hamming={fmt(full['hamming_overlap'])}; "
            f"dense-start cSGLD samples={int(float(csgld['left_count']))}, "
            f"p={fmt(csgld['layer_ks_pvalue'])}, hamming={fmt(csgld['hamming_overlap'])}; "
            f"independent-start cSGLD samples={int(float(csgld_ind['left_count']))}, "
            f"p={fmt(csgld_ind['layer_ks_pvalue'])}, hamming={fmt(csgld_ind['hamming_overlap'])}."
        ),
        evidence=[
            "runs/paper_stats.json: direct_mode_ticket_distribution",
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv",
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv",
        ],
        status="Closed for SGLD/cyclical-SGLD family",
        remaining_gap="Still open for a qualitatively different exact full-network CIFAR posterior.",
    )

    jointdiag_direct = first_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full JointDiagLap270k",
        comparison="posterior_samples_vs_tickets",
    )
    lowrank_direct = first_matching(
        stats["direct_mode_ticket_distribution"],
        setting="CIFAR full LowRank128Lap",
        comparison="posterior_samples_vs_tickets",
    )
    if float(jointdiag_direct["logit_cka_hungarian_mean"]) <= 0.85:
        fail("jointdiag direct CKA should remain high")
    add_objection(
        rows,
        objection="Function-space agreement is enough even if masks differ.",
        reviewer_risk="High CKA could make support mismatch look like a parameterization artifact.",
        current_answer=(
            "Function-space CKA is high, but the proposal requires mask-distribution "
            "equivalence; layer sparsity, Hamming overlap, and basin counts fail."
        ),
        key_numbers=(
            f"full SGLD logit/activation CKA={fmt(full['logit_cka_hungarian_mean'])}/"
            f"{fmt(full['activation_cka_hungarian_mean'])}, hamming={fmt(full['hamming_overlap'])}; "
            f"JointDiag270k CKA={fmt(jointdiag_direct['logit_cka_hungarian_mean'])}/"
            f"{fmt(jointdiag_direct['activation_cka_hungarian_mean'])}, "
            f"p={fmt(jointdiag_direct['layer_ks_pvalue'])}; "
            f"LowRank128 hamming={fmt(lowrank_direct['hamming_overlap'])} but "
            f"p={fmt(lowrank_direct['layer_ks_pvalue'])}."
        ),
        evidence=[
            "runs/paper_stats.json: direct_mode_ticket_distribution",
            "paper/main.tex: direct proposal-level CIFAR mode/ticket paragraphs",
        ],
        status="Closed for current direct metrics",
        remaining_gap="A paper revision should keep this distinction explicit in the main text.",
    )

    raw_global = global_channel_row(global_channel_audit, "posterior_sample", "ticket")
    raw_hamming = float(raw_global["global_channel_hamming"]["mean"])
    raw_overlap = float(raw_global["global_channel_support_overlap_min"]["mean"])
    exact_overall = exhaustive_channel_audit.get("overall", {})
    log10_assignments = float(exact_overall.get("full_log10_permutation_count", 0.0))
    if raw_hamming <= 0.20 or raw_overlap >= 0.50:
        fail("global channel audit should remain far from ticket-like support")
    if log10_assignments <= 800.0:
        fail("full channel search space should remain infeasible")
    add_objection(
        rows,
        objection="A channel permutation or re-basin step would align posterior masks.",
        reviewer_risk="Naive parameter coordinates may compare different neuron orderings.",
        current_answer=(
            "Activation/weight alignment, record-level matching, and a global "
            "block-coordinate channel audit all remain negative; exact full-data "
            "graph isomorphism is quantified as infeasible."
        ),
        key_numbers=(
            f"global-channel posterior/ticket Hamming={fmt(raw_hamming)}; "
            f"support overlap={fmt(raw_overlap)}; exact stage-1 assignments=128; "
            f"full search log10 assignments={fmt(log10_assignments, 1)}."
        ),
        evidence=[
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json",
            "runs/resnet_channel_permutation_exhaustive_feasibility_audit.json",
            "docs/mode_ticket_alignment_artifact_audit.md",
        ],
        status="Bounded open limitation",
        remaining_gap="Exhaustive full-data graph isomorphism remains unimplemented and infeasible.",
    )

    full_cov_gib = float(feasibility["all_trainable"]["dense_precision_float64_gib"])
    full_cov_chol_gib = float(
        feasibility["all_trainable"]["precision_plus_cholesky_float64_gib"]
    )
    if full_cov_gib <= 500.0 or full_cov_chol_gib <= 1000.0:
        fail("full covariance feasibility budget changed")
    covariance_checks = covariance_audit.get("interpretation_checks", {})
    if covariance_checks.get("posterior_covariance_robustness_ready") is not True:
        fail("posterior covariance robustness audit should be ready")
    if int(covariance_checks.get("movement_row_count", 0)) != 9:
        fail("posterior covariance audit should contain nine movement rows")
    if int(covariance_checks.get("direct_row_count", 0)) != 2:
        fail("posterior covariance audit should contain two direct rows")
    add_objection(
        rows,
        objection="The covariance posterior is too diagonal, local, or head-only.",
        reviewer_risk="A richer posterior covariance could be the missing ticket signal.",
        current_answer=(
            "The covariance-fidelity trend audit spans full-network rank-16/32/64/128 "
            "Hessian-plus-diagonal rows, exact tensor-block and joint-group "
            "rows up to the 270,896-weight vector, direct LowRank128 and "
            "JointDiag270k distribution rows, and exact dense small-model "
            "code-path checks; literal dense CIFAR covariance is outside the "
            "workstation budget."
        ),
        key_numbers=(
            f"movement/direct covariance rows={int(covariance_checks['movement_row_count'])}/"
            f"{int(covariance_checks['direct_row_count'])}; "
            f"max movement posterior-chain={fmt(covariance_checks['max_movement_posterior_minus_chain_start_jaccard'])}; "
            f"min exact rewind-posterior={fmt(covariance_checks['min_exact_rewind_minus_posterior_jaccard'])}; "
            f"JointDiag270k samples={int(float(jointdiag_direct['left_count']))}, "
            f"p={fmt(jointdiag_direct['layer_ks_pvalue'])}, hamming={fmt(jointdiag_direct['hamming_overlap'])}; "
            f"dense CIFAR matrix={fmt(full_cov_gib, 1)} GiB; matrix+Cholesky={fmt(full_cov_chol_gib, 1)} GiB."
        ),
        evidence=[
            "runs/posterior_covariance_robustness_audit.json",
            "docs/posterior_covariance_robustness_audit.md",
            "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv",
            "runs/cifar10_resnet20_full_covariance_feasibility.json",
            "docs/digits_fullnet_laplace_tiny_r2_p0p3.md",
        ],
        status="Bounded open limitation",
        remaining_gap="Exact dense full-network CIFAR posterior evidence is still absent.",
    )

    mnist_barrier = audit_metric(linear_audit, "MNIST Gate1 SGLD r5", "dense_imp_barrier")
    fashion_barrier = audit_metric(
        linear_audit, "Fashion-MNIST Gate1 SGLD r5", "dense_imp_barrier"
    )
    cifar_sgld_barrier = audit_metric(
        linear_audit, "CIFAR-10 ResNet-20 long SGLD r5", "dense_imp_barrier"
    )
    cifar_swag_barrier = audit_metric(
        linear_audit, "CIFAR-10 ResNet-20 long SWAG r5", "dense_imp_barrier"
    )
    if mnist_barrier >= 0.01 or fashion_barrier >= 0.05:
        fail("small-data dense-IMP barriers should be near zero")
    if cifar_sgld_barrier <= 2.0 or cifar_swag_barrier <= 2.0:
        fail("CIFAR linear barriers should be large")
    add_objection(
        rows,
        objection="Linear mode connectivity explains the ticket/posterior relation.",
        reviewer_risk="Low-loss paths might make posterior modes and tickets equivalent after all.",
        current_answer=(
            "Barrier size and support equivalence are empirically orthogonal in "
            "the current artifacts."
        ),
        key_numbers=(
            f"MNIST/Fashion dense-IMP barriers={fmt(mnist_barrier)}/{fmt(fashion_barrier)}; "
            f"CIFAR long SGLD/SWAG barriers={fmt(cifar_sgld_barrier)}/{fmt(cifar_swag_barrier)}; "
            "posterior-chain gaps stay <=0.001."
        ),
        evidence=[
            "runs/linear_connectivity_barrier_audit.json",
            "docs/linear_connectivity_barrier_audit.md",
        ],
        status="Closed for current barrier probes",
        remaining_gap="Nonlinear re-basin paths remain conceptually distinct from the tested linear barriers.",
    )

    cal_imp = first_matching(stats["calibration_ood"], source="imp")
    cal_swag = first_matching(stats["calibration_ood"], source="swag_ensemble")
    cal_var = first_matching(stats["calibration_ood"], source="variational_prune")
    support_var = first_matching(
        stats["trajectory_mask_training"],
        source_kind="variational_prune",
        source="variational_prune",
    )
    support_hard = first_matching(
        stats["trajectory_mask_training"],
        source_kind="hard_concrete",
        source="hard_concrete",
    )
    if metric(cal_swag, "id_accuracy") >= metric(cal_imp, "id_accuracy"):
        fail("SWAG calibration row should not beat IMP accuracy")
    if metric(cal_var, "ood_msp_auroc") >= metric(cal_imp, "ood_msp_auroc"):
        fail("variational calibration row should not beat IMP OOD AUROC")
    if metric(support_var, "source_to_imp") >= 0.10 or metric(support_hard, "source_to_imp") >= 0.10:
        fail("learned masks should remain random-scale in support")
    add_objection(
        rows,
        objection="Learned Bayesian or variational masks could recover tickets.",
        reviewer_risk="Support failures might be specific to sample-magnitude maps.",
        current_answer=(
            "Gem-Miner-style, Bernoulli/Concrete variational, and hard-concrete "
            "mask sources are below IMP in current CIFAR support/accuracy and "
            "do not rescue calibration/OOD tradeoffs."
        ),
        key_numbers=(
            f"IMP acc/AUROC={fmt(metric(cal_imp, 'id_accuracy'))}/{fmt(metric(cal_imp, 'ood_msp_auroc'))}; "
            f"SWAG acc/AUROC={fmt(metric(cal_swag, 'id_accuracy'))}/{fmt(metric(cal_swag, 'ood_msp_auroc'))}; "
            f"variational acc/AUROC={fmt(metric(cal_var, 'id_accuracy'))}/{fmt(metric(cal_var, 'ood_msp_auroc'))}; "
            f"variational/hard-concrete support={fmt(metric(support_var, 'source_to_imp'))}/"
            f"{fmt(metric(support_hard, 'source_to_imp'))}."
        ),
        evidence=[
            "runs/paper_stats.json: calibration_ood",
            "runs/paper_stats.json: trajectory_mask_training",
            "docs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3.md",
            "docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md",
        ],
        status="Closed for implemented learned-mask baselines",
        remaining_gap="Broader permutation-aware learned-mask distributions remain optional hardening.",
    )

    learned_round = first_matching(
        stats["residual_imp_process_learned_subspace"],
        variant="round_final_imp_residual",
        process_round=5,
    )
    learned_resid = first_matching(
        stats["residual_imp_process_learned_subspace"],
        variant="round_final_imp_learned_subspace_residualized_score_residual",
        process_round=5,
    )
    delta_learned = metric(learned_round, "trained_accuracy") - metric(
        learned_resid, "trained_accuracy"
    )
    tensor_round = first_matching(
        stats["residual_imp_process_tensor_score_exclusion"],
        variant="round_final_imp_residual",
        process_round=5,
    )
    tensor_repl = first_matching(
        stats["residual_imp_process_tensor_score_exclusion"],
        variant="round_excluded_tensor_score_oracle_final_imp_residual",
        process_round=5,
    )
    delta_tensor = metric(tensor_round, "trained_accuracy") - metric(
        tensor_repl, "trained_accuracy"
    )
    if delta_learned <= 0.0 or delta_tensor <= 0.0:
        fail("process residual controls should favor round-selected masks")
    add_objection(
        rows,
        objection="A simpler trajectory or process subspace explains the IMP residual.",
        reviewer_risk="The alternative mechanism could be an artifact of coarse controls.",
        current_answer=(
            "Tensor+score matching and learned-subspace residualization reduce "
            "but do not replace the process-selected IMP residual coordinates."
        ),
        key_numbers=(
            f"tensor+score round/replacement acc={fmt(metric(tensor_round, 'trained_accuracy'))}/"
            f"{fmt(metric(tensor_repl, 'trained_accuracy'))}, delta={fmt(delta_tensor)}; "
            f"learned-subspace round/residualized acc={fmt(metric(learned_round, 'trained_accuracy'))}/"
            f"{fmt(metric(learned_resid, 'trained_accuracy'))}, delta={fmt(delta_learned)}."
        ),
        evidence=[
            "runs/paper_stats.json: residual_imp_process_tensor_score_exclusion",
            "runs/paper_stats.json: residual_imp_process_learned_subspace",
            "docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md",
        ],
        status="Partially closed; useful positive mechanism",
        remaining_gap="Additional causal process interventions would strengthen the positive mechanism.",
    )

    add_objection(
        rows,
        objection="The artifact is venue-submission ready, but strict external GPU hardening is pending.",
        reviewer_risk="Even strong evidence can fail review if reproducibility and paper packaging are weak.",
        current_answer=(
            "Local checks, main-only submission PDF, clean artifact verification "
            "after `make clean`, the rebuilt CPU container path, and the local "
            "GPU-container validation receipt pass. The venue "
            "compliance audit now binds the draft to the official NeurIPS 2026 "
            "style/checklist and marks the main content plus venue binding "
            "ready. Local compute-resource, asset-license, and new-asset "
            "metadata plus a top-level MIT license are verifier-checked, and "
            "the anonymous-review release tarball is built, uploaded as a "
            "public GitHub release, archive-audited, and extraction-smoked. "
            "The source-only anonymous repository is pushed publicly, and the "
            "external CI receipt is observed green. Strict external-validation "
            "readiness remains open only because the independent external "
            "CUDA-host GPU-container receipt is still pending."
        ),
        key_numbers="latest local `make check`, clean verifier, rebuilt `make container-check`, and `make gpu-container-env-check` passed; main-only PDF is 9 pages; NeurIPS audit enforces a 9-page main-content budget before references; abstract is within the 250-word budget; release archive audit, extraction smoke, public-repository snapshot, public release URL, public source URL, and external CI URL are observed; strict external-validation gate fails only with `external_gpu_container_run_not_observed`.",
        evidence=[
            "docs/thread_goal_completion_audit.md",
            "docs/submission_pdf_shape_audit.md",
            "docs/venue_submission_compliance_audit.md",
            "docs/local_gpu_container_validation.md",
            "docs/external_validation_receipts.json",
            "docs/external_validation_runbook.md",
            "docs/submission_handoff.md",
            "docs/public_repository_snapshot_audit.md",
            "docs/external_validation_readiness_audit.md",
            "docs/public_release_manifest.md",
            "docs/release_anonymization_audit.md",
            "docs/public_release_archive_audit.md",
            "docs/public_release_archive_smoke.md",
            "docs/compute_resource_accounting.md",
            "docs/asset_license_inventory.md",
            "docs/new_asset_inventory.md",
            "LICENSE",
            "runs/public_release_manifest.json",
            "runs/public_repository_snapshot_audit.json",
            "runs/external_validation_readiness_audit.json",
            "runs/public_release_archive_audit.json",
            "runs/public_release_archive_smoke.json",
            "Dockerfile",
            "Dockerfile.gpu",
        ],
        status="Open external hardening limitation",
        remaining_gap="Only the independent external CUDA-host GPU-container receipt remains for strict external-validation readiness.",
    )

    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "closed_count": sum(1 for row in rows if str(row["status"]).startswith("Closed")),
            "bounded_open_count": sum("Bounded" in str(row["status"]) for row in rows),
            "open_packaging_count": sum(
                str(row["status"]) == "Open packaging limitation" for row in rows
            ),
            "external_hardening_count": sum(
                str(row["status"]) == "Open external hardening limitation"
                for row in rows
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Reviewer Objection Matrix",
        "",
        "This generated matrix maps likely reviewer objections to the current",
        "artifact-backed answer, key numbers, and remaining gap. It is not a",
        "claim that every robustness gap is closed or that strict external",
        "hardening is complete; it is a compact review risk register for the",
        "current negative-result paper.",
        "",
        "| Objection | Current answer | Key numbers | Evidence | Status | Remaining gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = "<br>".join(f"`{item}`" for item in row["evidence"])
        row_text = dict(row)
        row_text["evidence"] = evidence
        lines.append(
            "| {objection} | {current_answer} | {key_numbers} | {evidence} | {status} | {remaining_gap} |".format(
                **row_text,
            )
        )
    lines.extend(
        [
            "",
            "## Use In The Paper",
            "",
            "The main text should keep the first five objections visible: random",
            "controls, sampler movement, direct function-vs-mask mismatch,",
            "alignment/permutation robustness, and posterior covariance fidelity.",
            "The remaining rows are appendix, limitation, or release-package",
            "guidance.",
            "",
            "This file is generated by `scripts/build_reviewer_objection_matrix.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(
        stats=load_json(ROOT / "runs" / "paper_stats.json"),
        feasibility=load_json(ROOT / "runs" / "cifar10_resnet20_full_covariance_feasibility.json"),
        covariance_audit=load_json(ROOT / "runs" / "posterior_covariance_robustness_audit.json"),
        linear_audit=load_json(ROOT / "runs" / "linear_connectivity_barrier_audit.json"),
        global_channel_audit=load_json(
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json"
        ),
        exhaustive_channel_audit=load_json(
            ROOT / "runs" / "resnet_channel_permutation_exhaustive_feasibility_audit.json"
        ),
    )
    write_json(rows, args.out_json)
    write_markdown(rows, args.out_md)
    print(json.dumps({"rows": len(rows), "out_json": str(args.out_json), "out_md": str(args.out_md)}))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"reviewer objection matrix failed: {exc}")
        raise SystemExit(1)
