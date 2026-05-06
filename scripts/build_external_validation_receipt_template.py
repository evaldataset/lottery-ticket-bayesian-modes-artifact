#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_AUDIT = ROOT / "runs" / "public_release_archive_audit.json"
DEFAULT_SNAPSHOT_AUDIT = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_READINESS_AUDIT = ROOT / "runs" / "external_validation_readiness_audit.json"
DEFAULT_RECEIPTS = ROOT / "docs" / "external_validation_receipts.json"
DEFAULT_OUT_JSON = ROOT / "runs" / "external_validation_receipt_template.json"
DEFAULT_OUT_MD = ROOT / "docs" / "external_validation_receipt_template.md"

EXPECTED_RECEIPTS = {
    "public_release_upload",
    "public_repository",
    "external_ci",
    "external_gpu_container",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_ARCHIVE_AUDIT)
    parser.add_argument("--snapshot-audit", type=Path, default=DEFAULT_SNAPSHOT_AUDIT)
    parser.add_argument("--readiness-audit", type=Path, default=DEFAULT_READINESS_AUDIT)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
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


def build_receipt_template(
    *,
    archive_sha256: str,
    source_commit: str,
    archive_path: str,
    source_snapshot_path: str,
    schema_version: int,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "purpose": (
            "Fill this template only after the corresponding public upload or "
            "external validation has actually been observed. Immutable local "
            "fields are prefilled from the current archive and source snapshot."
        ),
        "receipts": {
            "public_release_upload": {
                "status": "pending",
                "url": "",
                "artifact_sha256": archive_sha256,
                "notes": (
                    "After uploading "
                    f"{archive_path}, set status to observed and add the public "
                    "archive URL. Keep artifact_sha256 unchanged unless the "
                    "archive is rebuilt."
                ),
            },
            "public_repository": {
                "status": "pending",
                "url": "",
                "commit": source_commit,
                "clean_tree_evidence": "",
                "notes": (
                    "After publishing "
                    f"{source_snapshot_path}, set status to observed, add the "
                    "anonymous repository URL, and paste clean-tree evidence "
                    "from git status --porcelain for this commit."
                ),
            },
            "external_ci": {
                "status": "pending",
                "url": "",
                "commit": source_commit,
                "passed": False,
                "notes": (
                    "After the public CI run passes for this exact commit, set "
                    "status to observed, add the run URL, and set passed to true."
                ),
            },
            "external_gpu_container": {
                "status": "pending",
                "url": "",
                "commit": source_commit,
                "image_digest": "",
                "passed": False,
                "notes": (
                    "After an external CUDA host validates the GPU container for "
                    "this exact commit, set status to observed, add the log URL, "
                    "paste the sha256 image ID or digest, and set passed to true."
                ),
            },
        },
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    archive = load_json_or_none(args.archive_audit)
    snapshot = load_json_or_none(args.snapshot_audit)
    readiness = load_json_or_none(args.readiness_audit)
    receipts = load_json_or_none(args.receipts)

    risk_flags: list[str] = []
    if not isinstance(archive, dict) or archive.get("archive_ready") is not True:
        risk_flags.append("public_release_archive_audit_not_ready")
        archive = archive if isinstance(archive, dict) else {}
    if not isinstance(snapshot, dict) or snapshot.get("public_repository_snapshot_ready") is not True:
        risk_flags.append("public_repository_snapshot_audit_not_ready")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
    if not isinstance(readiness, dict):
        risk_flags.append("external_validation_readiness_audit_missing")
        readiness = {}
    if not isinstance(receipts, dict):
        risk_flags.append("external_validation_receipts_missing")
        receipts = {}

    archive_sha256 = str(archive.get("archive_sha256", ""))
    archive_path = str(archive.get("archive", ""))
    source_commit = str(snapshot.get("git", {}).get("commit", ""))
    source_snapshot_path = str(snapshot.get("stage_dir", ""))
    required_receipts = set(str(item) for item in readiness.get("required_receipts", []))
    if required_receipts != EXPECTED_RECEIPTS:
        risk_flags.append("external_validation_required_receipts_changed")
    if not archive_sha256:
        risk_flags.append("archive_sha256_missing")
    if not source_commit:
        risk_flags.append("source_repository_commit_missing")

    schema_version = receipts.get("schema_version", 1)
    if not isinstance(schema_version, int):
        risk_flags.append("external_validation_receipt_schema_version_invalid")
        schema_version = 1

    template = build_receipt_template(
        archive_sha256=archive_sha256,
        source_commit=source_commit,
        archive_path=archive_path,
        source_snapshot_path=source_snapshot_path,
        schema_version=schema_version,
    )
    prefilled_fields = [
        {
            "receipt": "public_release_upload",
            "field": "artifact_sha256",
            "value": archive_sha256,
        },
        {"receipt": "public_repository", "field": "commit", "value": source_commit},
        {"receipt": "external_ci", "field": "commit", "value": source_commit},
        {"receipt": "external_gpu_container", "field": "commit", "value": source_commit},
    ]
    manual_fields = [
        {"receipt": "public_release_upload", "fields": ["status", "url"]},
        {"receipt": "public_repository", "fields": ["status", "url", "clean_tree_evidence"]},
        {"receipt": "external_ci", "fields": ["status", "url", "passed"]},
        {
            "receipt": "external_gpu_container",
            "fields": ["status", "url", "image_digest", "passed"],
        },
    ]

    return {
        "external_validation_receipt_template_ready": not risk_flags,
        "risk_flags": risk_flags,
        "inputs": {
            "archive_audit": relpath(args.archive_audit),
            "snapshot_audit": relpath(args.snapshot_audit),
            "readiness_audit": relpath(args.readiness_audit),
            "receipt_registry": relpath(args.receipts),
        },
        "local_facts": {
            "archive": archive_path,
            "archive_sha256": archive_sha256,
            "source_repository_snapshot": source_snapshot_path,
            "source_repository_commit": source_commit,
        },
        "required_external_receipts": sorted(EXPECTED_RECEIPTS),
        "prefilled_fields": prefilled_fields,
        "manual_fields": manual_fields,
        "receipt_template": template,
        "final_validation_commands": [
            ".venv/bin/python scripts/audit_external_validation_readiness.py --strict",
            ".venv/bin/python scripts/verify_research_artifacts.py",
        ],
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    facts = payload["local_facts"]
    status = "ready" if payload["external_validation_receipt_template_ready"] else "not ready"
    lines = [
        "# External Validation Receipt Template",
        "",
        "This generated template pre-fills the immutable local values needed for",
        "`docs/external_validation_receipts.json`. It is not itself an external",
        "receipt and should not be treated as observed evidence.",
        "",
        f"Template status: {status}.",
        "",
        "## Current Local Values",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Archive | `{facts['archive']}` |",
        f"| Archive SHA256 | `{facts['archive_sha256']}` |",
        f"| Source snapshot | `{facts['source_repository_snapshot']}` |",
        f"| Source snapshot commit | `{facts['source_repository_commit']}` |",
        "",
        "## Prefilled Fields",
        "",
        "| Receipt | Field | Value |",
        "| --- | --- | --- |",
    ]
    for row in payload["prefilled_fields"]:
        lines.append(f"| {row['receipt']} | {row['field']} | `{row['value']}` |")
    lines.extend(["", "## Manual Fields", "", "| Receipt | Fields |", "| --- | --- |"])
    for row in payload["manual_fields"]:
        fields = ", ".join(f"`{field}`" for field in row["fields"])
        lines.append(f"| {row['receipt']} | {fields} |")
    lines.extend(
        [
            "",
            "## Receipt Update Helper",
            "",
            "```bash",
            ".venv/bin/python scripts/update_external_validation_receipts.py --require-all \\",
            "  --public-release-url <public-archive-url> \\",
            "  --public-repository-url <anonymous-source-repository-url> \\",
            "  --public-repository-clean-tree-evidence '<clean-tree-evidence>' \\",
            "  --external-ci-url <public-ci-run-url> --external-ci-passed \\",
            "  --external-gpu-url <external-gpu-log-url> \\",
            "  --external-gpu-image-digest <sha256-image-id-or-digest> --external-gpu-passed",
            "# Re-run with --write after reviewing the printed candidate JSON.",
            "```",
            "",
            "## Final Validation",
            "",
            "```bash",
        ]
    )
    lines.extend(payload["final_validation_commands"])
    lines.extend(
        [
            "```",
            "",
            "## Risk Flags",
            "",
        ]
    )
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "The machine-readable template is written to",
            "`runs/external_validation_receipt_template.json`.",
            "",
            "This file is generated by `scripts/build_external_validation_receipt_template.py`.",
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
                "external_validation_receipt_template_ready": payload[
                    "external_validation_receipt_template_ready"
                ],
                "risk_flags": payload["risk_flags"],
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["external_validation_receipt_template_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
