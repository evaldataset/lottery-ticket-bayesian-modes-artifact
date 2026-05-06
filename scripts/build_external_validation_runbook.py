#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_AUDIT = ROOT / "runs" / "public_release_archive_audit.json"
DEFAULT_SNAPSHOT_AUDIT = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_SNAPSHOT_SMOKE = ROOT / "runs" / "public_repository_snapshot_smoke.json"
DEFAULT_READINESS_AUDIT = ROOT / "runs" / "external_validation_readiness_audit.json"
DEFAULT_RECEIPTS = ROOT / "docs" / "external_validation_receipts.json"
DEFAULT_RECEIPT_TEMPLATE = ROOT / "runs" / "external_validation_receipt_template.json"
DEFAULT_RECEIPT_TEMPLATE_MD = ROOT / "docs" / "external_validation_receipt_template.md"
DEFAULT_OUT_JSON = ROOT / "runs" / "external_validation_runbook.json"
DEFAULT_OUT_MD = ROOT / "docs" / "external_validation_runbook.md"

CPU_CONTAINER_IMAGE = "lottery-artifact:2026-05-06"
GPU_CONTAINER_IMAGE = "lottery-training-gpu:2026-05-06"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_ARCHIVE_AUDIT)
    parser.add_argument("--snapshot-audit", type=Path, default=DEFAULT_SNAPSHOT_AUDIT)
    parser.add_argument("--snapshot-smoke", type=Path, default=DEFAULT_SNAPSHOT_SMOKE)
    parser.add_argument("--readiness-audit", type=Path, default=DEFAULT_READINESS_AUDIT)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--receipt-template", type=Path, default=DEFAULT_RECEIPT_TEMPLATE)
    parser.add_argument("--receipt-template-md", type=Path, default=DEFAULT_RECEIPT_TEMPLATE_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_json_or_none(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def receipt_rows(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    rows = readiness.get("receipt_statuses", [])
    return rows if isinstance(rows, list) else []


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    archive = load_json_or_none(args.archive_audit)
    snapshot = load_json_or_none(args.snapshot_audit)
    smoke = load_json_or_none(args.snapshot_smoke)
    readiness = load_json_or_none(args.readiness_audit)
    receipts = load_json_or_none(args.receipts)
    receipt_template = load_json_or_none(args.receipt_template)

    risk_flags: list[str] = []
    if not isinstance(archive, dict) or archive.get("archive_ready") is not True:
        risk_flags.append("public_release_archive_audit_not_ready")
        archive = archive if isinstance(archive, dict) else {}
    if not isinstance(snapshot, dict) or snapshot.get("public_repository_snapshot_ready") is not True:
        risk_flags.append("public_repository_snapshot_audit_not_ready")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
    if not isinstance(smoke, dict) or smoke.get("source_repository_smoke_ready") is not True:
        risk_flags.append("public_repository_snapshot_smoke_not_ready")
        smoke = smoke if isinstance(smoke, dict) else {}
    if not isinstance(readiness, dict):
        risk_flags.append("external_validation_readiness_audit_missing")
        readiness = {}
    if not isinstance(receipts, dict):
        risk_flags.append("external_validation_receipts_missing")
        receipts = {}
    if (
        not isinstance(receipt_template, dict)
        or receipt_template.get("external_validation_receipt_template_ready") is not True
    ):
        risk_flags.append("external_validation_receipt_template_not_ready")
        receipt_template = receipt_template if isinstance(receipt_template, dict) else {}

    source_commit = str(snapshot.get("git", {}).get("commit", ""))
    smoke_commit = str(smoke.get("git", {}).get("commit", ""))
    if source_commit and smoke_commit and source_commit != smoke_commit:
        risk_flags.append("public_repository_snapshot_commit_mismatch")

    rows = receipt_rows(readiness)
    required_receipts = sorted(str(row.get("key", "")) for row in rows if isinstance(row, dict))
    blocking_receipts = [
        {
            "key": str(row.get("key", "")),
            "missing": row.get("missing", []),
            "invalid": row.get("invalid", []),
            "risk_flag": row.get("risk_flag", ""),
        }
        for row in rows
        if isinstance(row, dict) and row.get("ready") is not True
    ]

    commands = {
        "local_preflight": [
            "make check",
            "make clean && .venv/bin/python scripts/verify_research_artifacts.py",
            "make container-build",
            "make container-check",
            ".venv/bin/python scripts/audit_external_validation_readiness.py",
            ".venv/bin/python scripts/build_external_validation_receipt_template.py",
        ],
        "archive_upload": [
            "sha256sum dist/lottery_artifact_public_release_2026-05-06.tar.gz",
            "cat dist/lottery_artifact_public_release_2026-05-06.tar.gz.sha256",
        ],
        "source_repository_publish": [
            "cd dist/lottery_public_repository_snapshot",
            "git status --porcelain",
            "git rev-parse HEAD",
            "git remote add origin <anonymous-source-repository-url>",
            "git push -u origin main",
        ],
        "external_ci": [
            "# Open the public repository Actions page for the pushed commit.",
            "# Confirm the check workflow passed for the exact source snapshot commit.",
        ],
        "external_gpu_container": [
            "make gpu-container-build",
            "docker image inspect lottery-training-gpu:2026-05-06 --format '{{.Id}}'",
            "make gpu-container-env-check",
            (
                "python scripts/build_external_gpu_container_receipt.py "
                f"--expected-commit {source_commit}"
            ),
            "# If validating from an extracted archive without .git, add:",
            f"#   --observed-commit {source_commit}",
            "cat runs/external_gpu_container_receipt.json",
            "# Upload runs/external_gpu_container_receipt.json or",
            "# docs/external_gpu_container_receipt.md and use that public URL below.",
        ],
        "receipt_registry_update": [
            ".venv/bin/python scripts/update_external_validation_receipts.py --require-all \\",
            "  --public-release-url <public-archive-url> \\",
            "  --public-repository-url <anonymous-source-repository-url> \\",
            "  --public-repository-clean-tree-evidence '<clean-tree-evidence>' \\",
            "  --external-ci-url <public-ci-run-url> --external-ci-passed \\",
            "  --external-gpu-url <external-gpu-log-url> \\",
            "  --external-gpu-image-digest <sha256-image-id-or-digest> --external-gpu-passed",
            "# Re-run with --write after reviewing the printed candidate JSON.",
        ],
        "final_gate": [
            ".venv/bin/python scripts/audit_external_validation_readiness.py --strict",
            "# Strict mode rejects placeholder URLs and probes external URL reachability.",
            ".venv/bin/python scripts/verify_research_artifacts.py",
        ],
    }

    return {
        "external_validation_runbook_ready": not risk_flags,
        "risk_flags": risk_flags,
        "local_facts": {
            "archive": str(archive.get("archive", "")),
            "archive_sha256": str(archive.get("archive_sha256", "")),
            "source_repository_snapshot": str(snapshot.get("stage_dir", "")),
            "source_repository_commit": source_commit,
            "source_repository_smoke_ready": smoke.get("source_repository_smoke_ready") is True,
            "receipt_registry": relpath(args.receipts),
            "receipt_template": relpath(args.receipt_template_md),
            "cpu_container_image": CPU_CONTAINER_IMAGE,
            "gpu_container_image": GPU_CONTAINER_IMAGE,
        },
        "required_external_receipts": required_receipts,
        "blocking_receipts": blocking_receipts,
        "top_conference_release_ready": readiness.get("top_conference_release_ready") is True,
        "commands": commands,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    facts = payload["local_facts"]
    status = "ready" if payload["external_validation_runbook_ready"] else "not ready"
    lines = [
        "# External Validation Runbook",
        "",
        "This generated runbook turns the current local artifact state into the",
        "four external receipts required before claiming top-conference release",
        "readiness.",
        "",
        f"Runbook status: {status}.",
        f"Top-conference release status: {payload['top_conference_release_ready']}.",
        "",
        "## Current Local Facts",
        "",
        "| Fact | Value |",
        "| --- | --- |",
        f"| Archive | `{facts['archive']}` |",
        f"| Archive SHA256 | `{facts['archive_sha256']}` |",
        f"| Source snapshot | `{facts['source_repository_snapshot']}` |",
        f"| Source snapshot commit | `{facts['source_repository_commit']}` |",
        f"| Source snapshot smoke ready | {facts['source_repository_smoke_ready']} |",
        f"| Receipt registry | `{facts['receipt_registry']}` |",
        f"| Receipt fill template | `{facts['receipt_template']}` |",
        f"| CPU artifact image | `{facts['cpu_container_image']}` |",
        f"| GPU training image | `{facts['gpu_container_image']}` |",
        "",
        "## Required Receipts",
        "",
        "| Receipt | Missing | Invalid | Risk flag |",
        "| --- | --- | --- | --- |",
    ]
    blocking_by_key = {row["key"]: row for row in payload["blocking_receipts"]}
    for key in payload["required_external_receipts"]:
        row = blocking_by_key.get(key)
        if row is None:
            lines.append(f"| {key} | none | none | none |")
        else:
            missing = ", ".join(str(item) for item in row.get("missing", [])) or "none"
            invalid = ", ".join(str(item) for item in row.get("invalid", [])) or "none"
            lines.append(f"| {key} | {missing} | {invalid} | {row.get('risk_flag', '')} |")
    if not payload["required_external_receipts"]:
        lines.append("| none | no readiness audit rows | none | external_validation_readiness_audit_missing |")

    command_sections = [
        ("Local Preflight", "local_preflight"),
        ("Archive Upload", "archive_upload"),
        ("Source Repository Publish", "source_repository_publish"),
        ("External CI", "external_ci"),
        ("External GPU Container", "external_gpu_container"),
        ("Receipt Registry Update", "receipt_registry_update"),
        ("Final Gate", "final_gate"),
    ]
    for title, key in command_sections:
        lines.extend(["", f"## {title}", "", "```bash"])
        lines.extend(str(command) for command in payload["commands"][key])
        lines.append("```")
    lines.extend(
        [
            "",
            "## Receipt Update Checklist",
            "",
            "1. Set each completed receipt in `docs/external_validation_receipts.json`",
            "   to `status: observed`; use `docs/external_validation_receipt_template.md`",
            "   for the current archive SHA256 and source commit.",
            "2. Copy the archive URL and the exact archive SHA256 into",
            "   `public_release_upload`.",
            "3. Copy the anonymous repository URL, source snapshot commit, and clean",
            "   tree evidence into `public_repository`.",
            "4. Copy the public CI run URL, commit, and `passed: true` into",
            "   `external_ci`.",
            "5. Copy the external GPU log URL, commit, image digest or image ID, and",
            "   `passed: true` into `external_gpu_container`.",
            "6. Re-run the final gate commands above.",
            "",
            "This file is generated by `scripts/build_external_validation_runbook.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "external_validation_runbook_ready": payload[
                    "external_validation_runbook_ready"
                ],
                "risk_flags": payload["risk_flags"],
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["external_validation_runbook_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
