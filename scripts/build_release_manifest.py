#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "public_release_manifest.json"
DEFAULT_OUT_MD = ROOT / "docs" / "public_release_manifest.md"

ROOT_FILES = [
    ".dockerignore",
    ".github/workflows/check.yml",
    ".gitignore",
    "Dockerfile",
    "Dockerfile.gpu",
    "LICENSE",
    "Makefile",
    "README.md",
    "proposal_A3_lottery_ticket_bayesian_modes.md",
    "requirements.txt",
    "requirements-ci.txt",
    "requirements-gpu-lock.txt",
    "requirements-lock.txt",
]

GLOB_PATTERNS = [
    "src/lottery/*.py",
    "scripts/*.py",
    "docs/*.md",
    "docs/*.json",
    "paper/*.tex",
    "paper/*.sty",
    "paper/*.bib",
    "paper/*.md",
    "paper/*.pdf",
    "paper/tables/*.tex",
    "paper/figures/*.pdf",
    "paper/figures/*.png",
    "runs/**/*.csv",
    "runs/**/*.json",
    "runs/**/*mask_artifacts.npz",
]

EXCLUDED = {
    # Mutable post-release evidence. This registry records public URLs, the
    # source snapshot commit, and the archive SHA after publication; including
    # it in the immutable release manifest would create a self-reference.
    "docs/external_validation_receipts.json",
    "docs/external_validation_readiness_audit.md",
    "docs/external_validation_receipt_template.md",
    "docs/external_validation_runbook.md",
    "docs/submission_handoff.md",
    "docs/public_repository_snapshot_audit.md",
    "docs/public_repository_snapshot_smoke.md",
    "docs/public_release_manifest.md",
    "docs/public_release_archive_audit.md",
    "docs/public_release_archive_smoke.md",
    "docs/release_anonymization_audit.md",
    "runs/external_validation_readiness_audit.json",
    "runs/external_validation_receipt_template.json",
    "runs/external_validation_runbook.json",
    "runs/submission_handoff.json",
    "runs/public_repository_snapshot_audit.json",
    "runs/public_repository_snapshot_smoke.json",
    "runs/public_release_manifest.json",
    "runs/public_release_archive_audit.json",
    "runs/public_release_archive_smoke.json",
    "runs/release_anonymization_audit.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-06")
    return parser.parse_args()


def release_category(path: str) -> str:
    if path.startswith("src/"):
        return "source"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith("docs/"):
        return "docs"
    if path.startswith("paper/figures/"):
        return "paper-figures"
    if path.startswith("paper/tables/"):
        return "paper-tables"
    if path.startswith("paper/"):
        return "paper"
    if path.startswith("runs/"):
        return "run-artifacts"
    if path.startswith(".github/"):
        return "ci"
    return "root"


def iter_files() -> list[Path]:
    paths: set[Path] = set()
    for text in ROOT_FILES:
        path = ROOT / text
        if path.exists():
            paths.add(path)
    for pattern in GLOB_PATTERNS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    clean = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED:
            continue
        if "__pycache__" in path.parts:
            continue
        clean.append(path)
    return sorted(clean, key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_entries(paths: Iterable[Path]) -> list[dict[str, object]]:
    entries = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": rel,
                "category": release_category(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return entries


def summarize(entries: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for entry in entries:
        bucket = buckets[str(entry["category"])]
        bucket["files"] += 1
        bucket["bytes"] += int(entry["bytes"])
    return dict(sorted(buckets.items()))


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def write_markdown(
    path: Path,
    *,
    release_date: str,
    entries: list[dict[str, object]],
    summary: dict[str, dict[str, int]],
    out_json: Path,
) -> None:
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    lines = [
        "# Public Release Manifest",
        "",
        f"Date: {release_date}",
        "",
        "This manifest defines the minimum local package needed to inspect the",
        "paper, regenerate generated statistics from included run artifacts,",
        "and verify the core claims. The full per-file SHA256 inventory is in",
        f"`{out_json.relative_to(ROOT).as_posix()}`.",
        "",
        "`docs/external_validation_receipts.json` is intentionally excluded:",
        "it is a mutable post-release registry for public URLs, source commits,",
        "CI runs, and GPU logs.",
        "",
        f"Total files: {len(entries)}",
        f"Total bytes: {fmt_bytes(total_bytes)}",
        "",
        "## Category Summary",
        "",
        "| Category | Files | Bytes |",
        "| --- | ---: | ---: |",
    ]
    for category, row in summary.items():
        lines.append(f"| {category} | {row['files']} | {fmt_bytes(row['bytes'])} |")

    required = [
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
        "docs/posterior_covariance_robustness_audit.md",
        "docs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3.md",
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
        "scripts/audit_posterior_covariance_robustness.py",
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
        "runs/posterior_covariance_robustness_audit.csv",
        "runs/posterior_covariance_robustness_audit.json",
        "runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv",
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
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv",
        "runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv",
    ]
    by_path = {str(entry["path"]): entry for entry in entries}
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "",
            "| Path | Bytes | SHA256 |",
            "| --- | ---: | --- |",
        ]
    )
    for rel in required:
        entry = by_path.get(rel)
        if entry is None:
            lines.append(f"| {rel} | missing | missing |")
        else:
            digest = str(entry["sha256"])
            lines.append(f"| {rel} | {entry['bytes']} | `{digest}` |")

    lines.extend(
        [
            "",
            "## Verification",
            "",
            "```bash",
            "make check",
            "make paper-check",
            "make paper-neurips-check",
            "```",
            "",
            "This file is generated by `scripts/build_release_manifest.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    entries = build_entries(iter_files())
    summary = summarize(entries)
    payload = {
        "date": args.date,
        "root": ".",
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "summary": summary,
        "files": entries,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(
        args.out_md,
        release_date=args.date,
        entries=entries,
        summary=summary,
        out_json=args.out_json,
    )
    print(
        json.dumps(
            {
                "files": len(entries),
                "total_bytes": payload["total_bytes"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
