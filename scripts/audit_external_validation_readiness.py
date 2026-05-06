#!/usr/bin/env python
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = ROOT / "docs" / "external_validation_receipts.json"
DEFAULT_REPOSITORY_SNAPSHOT_AUDIT = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_REPOSITORY_SNAPSHOT_SMOKE = ROOT / "runs" / "public_repository_snapshot_smoke.json"
DEFAULT_OUT_JSON = ROOT / "runs" / "external_validation_readiness_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "external_validation_readiness_audit.md"
DEFAULT_ARCHIVE = ROOT / "dist" / "lottery_artifact_public_release_2026-05-06.tar.gz"
DEFAULT_ARCHIVE_SHA256 = DEFAULT_ARCHIVE.with_suffix(DEFAULT_ARCHIVE.suffix + ".sha256")

REQUIRED_RECEIPTS = {
    "public_release_upload": {
        "required_fields": ["url", "artifact_sha256"],
        "required_true": [],
        "risk_flag": "public_release_upload_not_verified",
    },
    "public_repository": {
        "required_fields": ["url", "commit", "clean_tree_evidence"],
        "required_true": [],
        "risk_flag": "public_repository_state_not_verified",
    },
    "external_ci": {
        "required_fields": ["url", "commit"],
        "required_true": ["passed"],
        "risk_flag": "external_ci_run_not_observed",
    },
    "external_gpu_container": {
        "required_fields": ["url", "commit", "image_digest"],
        "required_true": ["passed"],
        "risk_flag": "external_gpu_container_run_not_observed",
    },
}

EMPTY_MARKERS = {"", "pending", "todo", "tbd", "none", "n/a", "na", "null"}
URL_PATTERN = re.compile(r"^(https?://|doi:|10\.)", re.IGNORECASE)
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
HEX40_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
IMAGE_DIGEST_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
PLACEHOLDER_URL_TOKENS = {
    "<",
    ">",
    "example.com",
    "example.org",
    "example.net",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "your-",
    "placeholder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument(
        "--repository-snapshot-audit",
        type=Path,
        default=DEFAULT_REPOSITORY_SNAPSHOT_AUDIT,
    )
    parser.add_argument(
        "--repository-snapshot-smoke",
        type=Path,
        default=DEFAULT_REPOSITORY_SNAPSHOT_SMOKE,
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--archive-sha256", type=Path, default=DEFAULT_ARCHIVE_SHA256)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Probe external receipt URLs with HTTP requests. --strict implies this.",
    )
    parser.add_argument("--url-timeout", type=float, default=10.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless all local and external submission-readiness receipts are verified.",
    )
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


def meaningful(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text.lower() not in EMPTY_MARKERS


def url_like(value: Any) -> bool:
    return meaningful(value) and bool(URL_PATTERN.search(str(value).strip()))


def normalized_url(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("doi:"):
        doi = text.split(":", 1)[1].strip()
        return f"https://doi.org/{doi}"
    if text.startswith("10."):
        return f"https://doi.org/{text}"
    return text


def url_has_placeholder(value: Any) -> bool:
    text = str(value).strip().lower()
    if any(token in text for token in PLACEHOLDER_URL_TOKENS):
        return True
    parsed = urllib.parse.urlparse(normalized_url(value))
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def probe_url(value: Any, *, timeout: float) -> dict[str, Any]:
    url = normalized_url(value)
    result = {
        "url": str(value).strip(),
        "normalized_url": url,
        "checked": True,
        "ok": False,
        "status": None,
        "final_url": "",
        "error": "",
    }
    headers = {"User-Agent": "lottery-artifact-external-validation/1.0"}
    for method in ["HEAD", "GET"]:
        request = urllib.request.Request(url, headers=headers, method=method)
        if method == "GET":
            request.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result["status"] = int(response.status)
                result["final_url"] = response.geturl()
                result["ok"] = 200 <= int(response.status) < 400
                result["error"] = ""
                return result
        except urllib.error.HTTPError as exc:
            result["status"] = int(exc.code)
            result["final_url"] = exc.geturl()
            result["error"] = f"http_error:{exc.code}"
            if method == "HEAD" and exc.code in {403, 405, 501}:
                continue
            result["ok"] = 200 <= int(exc.code) < 400
            return result
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            result["error"] = type(exc).__name__
            if method == "HEAD":
                continue
            return result
    return result


def read_sha256_sidecar(path: Path) -> str:
    if not path.exists():
        return ""
    first = path.read_text(encoding="utf-8").strip().split()
    return first[0] if first else ""


def git_status() -> dict[str, Any]:
    status = {
        "repository_detected": False,
        "commit": "",
        "clean": False,
        "porcelain_count": None,
        "risk_flags": [],
    }
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        status["risk_flags"].append("git_not_installed")
        return status
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        status["risk_flags"].append("local_git_repository_not_detected")
        return status
    status["repository_detected"] = True
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if commit.returncode == 0:
        status["commit"] = commit.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if dirty.returncode == 0:
        lines = [line for line in dirty.stdout.splitlines() if line.strip()]
        status["porcelain_count"] = len(lines)
        status["clean"] = len(lines) == 0
        if lines:
            status["risk_flags"].append("local_git_worktree_dirty")
    else:
        status["risk_flags"].append("local_git_status_unavailable")
    if not status["commit"]:
        status["risk_flags"].append("local_git_commit_unavailable")
    return status


def local_gate_status(archive: Path, archive_sha256: Path) -> dict[str, Any]:
    anonymization = load_json_or_none(ROOT / "runs" / "release_anonymization_audit.json")
    archive_audit = load_json_or_none(ROOT / "runs" / "public_release_archive_audit.json")
    smoke = load_json_or_none(ROOT / "runs" / "public_release_archive_smoke.json")
    sha_sidecar = read_sha256_sidecar(archive_sha256)

    gates = {
        "release_anonymization": bool(
            anonymization
            and anonymization.get("release_anonymization_ready") is True
            and anonymization.get("risk_flags") == []
        ),
        "public_release_archive": bool(
            archive.exists()
            and archive.stat().st_size > 100_000_000
            and archive_audit
            and archive_audit.get("archive_ready") is True
            and archive_audit.get("risk_flags") == []
        ),
        "public_release_archive_smoke": bool(
            smoke
            and smoke.get("release_archive_smoke_ready") is True
            and smoke.get("risk_flags") == []
            and smoke.get("verifier", {}).get("returncode") == 0
        ),
        "archive_sha256_sidecar": bool(HEX64_PATTERN.match(sha_sidecar)),
    }
    archive_sha = ""
    if isinstance(archive_audit, dict):
        archive_sha = str(archive_audit.get("archive_sha256", ""))
    if archive_sha and sha_sidecar and archive_sha != sha_sidecar:
        gates["archive_sha256_sidecar"] = False

    risk_flags = []
    if not gates["release_anonymization"]:
        risk_flags.append("release_anonymization_not_ready")
    if not gates["public_release_archive"]:
        risk_flags.append("public_release_archive_not_ready")
    if not gates["public_release_archive_smoke"]:
        risk_flags.append("public_release_archive_smoke_not_ready")
    if not gates["archive_sha256_sidecar"]:
        risk_flags.append("archive_sha256_sidecar_not_ready")

    return {
        "archive": relpath(archive),
        "archive_sha256_sidecar": relpath(archive_sha256),
        "archive_sha256": archive_sha or sha_sidecar,
        "gates": gates,
        "risk_flags": risk_flags,
        "local_artifact_release_ready": not risk_flags,
    }


def receipt_status(
    key: str,
    receipt: dict[str, Any],
    *,
    local_archive_sha256: str,
    local_repository_commit: str,
    check_urls: bool,
    url_timeout: float,
) -> dict[str, Any]:
    spec = REQUIRED_RECEIPTS[key]
    missing = []
    invalid = []
    status = str(receipt.get("status", "pending")).strip().lower()
    if status != "observed":
        missing.append("status=observed")
    for field in spec["required_fields"]:
        if not meaningful(receipt.get(field)):
            missing.append(field)
    for field in spec["required_true"]:
        if receipt.get(field) is not True:
            missing.append(f"{field}=true")
    url_check: dict[str, Any] | None = None
    if "url" in spec["required_fields"] and meaningful(receipt.get("url")):
        if not url_like(receipt.get("url")):
            invalid.append("url")
        elif url_has_placeholder(receipt.get("url")):
            invalid.append("url_placeholder")
        elif check_urls:
            url_check = probe_url(receipt.get("url"), timeout=url_timeout)
            if not url_check["ok"]:
                invalid.append("url_unreachable")
    if key == "public_release_upload":
        artifact_sha = str(receipt.get("artifact_sha256", "")).strip()
        if meaningful(artifact_sha) and not HEX64_PATTERN.match(artifact_sha):
            invalid.append("artifact_sha256_format")
        if local_archive_sha256 and artifact_sha and artifact_sha != local_archive_sha256:
            invalid.append("artifact_sha256_mismatch")
    if "commit" in spec["required_fields"] and meaningful(receipt.get("commit")):
        commit = str(receipt.get("commit", "")).strip()
        if not HEX40_PATTERN.match(commit):
            invalid.append("commit_format")
        elif local_repository_commit and commit != local_repository_commit:
            invalid.append("commit_mismatch")
    if key == "external_gpu_container" and meaningful(receipt.get("image_digest")):
        image_digest = str(receipt.get("image_digest", "")).strip()
        if not IMAGE_DIGEST_PATTERN.match(image_digest):
            invalid.append("image_digest_format")

    ready = not missing and not invalid
    return {
        "key": key,
        "status": status,
        "ready": ready,
        "missing": missing,
        "invalid": invalid,
        "url_check": url_check,
        "risk_flag": "" if ready else str(spec["risk_flag"]),
    }


def repository_snapshot_status(path: Path, smoke_path: Path) -> dict[str, Any]:
    payload = load_json_or_none(path)
    smoke = load_json_or_none(smoke_path)
    if not isinstance(payload, dict):
        return {
            "path": relpath(path),
            "smoke_path": relpath(smoke_path),
            "present": False,
            "smoke_present": isinstance(smoke, dict),
            "public_repository_snapshot_ready": False,
            "source_repository_smoke_ready": bool(
                isinstance(smoke, dict) and smoke.get("source_repository_smoke_ready") is True
            ),
            "commit": "",
            "stage_dir": "",
            "risk_flags": ["public_repository_snapshot_audit_missing"],
        }
    smoke_ready = isinstance(smoke, dict) and smoke.get("source_repository_smoke_ready") is True
    risk_flags = list(payload.get("risk_flags", []))
    if not smoke_ready:
        risk_flags.append("public_repository_snapshot_smoke_not_ready")
    return {
        "path": relpath(path),
        "smoke_path": relpath(smoke_path),
        "present": True,
        "smoke_present": isinstance(smoke, dict),
        "public_repository_snapshot_ready": payload.get("public_repository_snapshot_ready") is True,
        "source_repository_smoke_ready": smoke_ready,
        "commit": str(payload.get("git", {}).get("commit", "")),
        "stage_dir": str(payload.get("stage_dir", "")),
        "source_file_count": payload.get("source_file_count"),
        "tracked_file_count": payload.get("tracked_file_count"),
        "risk_flags": risk_flags,
    }


def build_audit(
    receipts_path: Path,
    repository_snapshot_audit_path: Path,
    repository_snapshot_smoke_path: Path,
    archive: Path,
    archive_sha256: Path,
    *,
    check_urls: bool,
    url_timeout: float,
) -> dict[str, Any]:
    local = local_gate_status(archive, archive_sha256)
    git = git_status()
    repository_snapshot = repository_snapshot_status(
        repository_snapshot_audit_path,
        repository_snapshot_smoke_path,
    )
    receipts_payload = load_json(receipts_path) if receipts_path.exists() else {"receipts": {}}
    receipts = receipts_payload.get("receipts", {})
    if not isinstance(receipts, dict):
        receipts = {}

    receipt_rows = []
    risk_flags = list(local["risk_flags"])
    local_repository_commit = str(repository_snapshot.get("commit", ""))
    for key in REQUIRED_RECEIPTS:
        row = receipt_status(
            key,
            receipts.get(key, {}) if isinstance(receipts.get(key, {}), dict) else {},
            local_archive_sha256=str(local.get("archive_sha256", "")),
            local_repository_commit=local_repository_commit,
            check_urls=check_urls,
            url_timeout=url_timeout,
        )
        receipt_rows.append(row)
        if row["risk_flag"]:
            risk_flags.append(row["risk_flag"])
    repository_receipt_ready = any(
        row["key"] == "public_repository" and row["ready"] for row in receipt_rows
    )
    local_clean_repository_ready = bool(
        (
            repository_snapshot.get("public_repository_snapshot_ready")
            and repository_snapshot.get("source_repository_smoke_ready")
        )
        or (git.get("repository_detected") is True and git.get("clean") is True)
    )
    clean_repository_ready = bool(repository_receipt_ready or local_clean_repository_ready)
    for flag in repository_snapshot.get("risk_flags", []):
        if flag not in risk_flags:
            risk_flags.append(flag)
    if not local_clean_repository_ready:
        for flag in git.get("risk_flags", []):
            if flag not in risk_flags:
                risk_flags.append(flag)

    external_validation_ready = all(row["ready"] for row in receipt_rows)
    top_conference_release_ready = bool(
        local["local_artifact_release_ready"]
        and external_validation_ready
        and clean_repository_ready
    )

    blocking_next_steps = []
    if not local["local_artifact_release_ready"]:
        blocking_next_steps.append("Repair local release archive, anonymization, or extraction-smoke gates.")
    if not local_clean_repository_ready:
        blocking_next_steps.append("Create a clean source-only anonymous repository snapshot.")
    for row in receipt_rows:
        if row["ready"]:
            continue
        if row["key"] == "public_release_upload":
            blocking_next_steps.append("Upload the anonymous tarball and record its URL plus matching SHA256.")
        elif row["key"] == "public_repository":
            blocking_next_steps.append("Publish the clean anonymous repository and record URL, commit, and clean-tree evidence.")
        elif row["key"] == "external_ci":
            blocking_next_steps.append("Run public CI for the anonymous repository and record the passing run URL.")
        elif row["key"] == "external_gpu_container":
            blocking_next_steps.append("Run the GPU container on an external CUDA host and record the passing log URL.")

    return {
        "receipts": relpath(receipts_path),
        "receipt_schema_version": receipts_payload.get("schema_version"),
        "required_receipts": sorted(REQUIRED_RECEIPTS),
        "local": local,
        "git": git,
        "public_repository_snapshot": repository_snapshot,
        "receipt_statuses": receipt_rows,
        "url_checks_enabled": check_urls,
        "external_validation_ready": external_validation_ready,
        "local_clean_repository_ready": local_clean_repository_ready,
        "clean_repository_ready": clean_repository_ready,
        "top_conference_release_ready": top_conference_release_ready,
        "risk_flags": risk_flags,
        "blocking_next_steps": blocking_next_steps,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    local_status = "ready" if payload["local"]["local_artifact_release_ready"] else "not ready"
    external_status = "ready" if payload["external_validation_ready"] else "not ready"
    repository_status = "ready" if payload["clean_repository_ready"] else "not ready"
    top_status = "ready" if payload["top_conference_release_ready"] else "not ready"
    lines = [
        "# External Validation Readiness Audit",
        "",
        "This generated audit separates local artifact readiness from external",
        "submission-readiness receipts that cannot be created inside this workspace.",
        "",
        f"Current local artifact-release status: {local_status}.",
        f"Current external-validation status: {external_status}.",
        f"Current clean-repository status: {repository_status}.",
        f"Current top-conference release status: {top_status}.",
        "",
        "## Local Gates",
        "",
        "| Gate | Ready |",
        "| --- | ---: |",
    ]
    for key, value in payload["local"]["gates"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            f"| archive | `{payload['local']['archive']}` |",
            f"| archive_sha256 | `{payload['local']['archive_sha256']}` |",
            "",
            "## Public Repository Snapshot",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Snapshot audit | `{payload['public_repository_snapshot']['path']}` |",
            f"| Snapshot smoke | `{payload['public_repository_snapshot']['smoke_path']}` |",
            f"| Present | {payload['public_repository_snapshot']['present']} |",
            f"| Smoke present | {payload['public_repository_snapshot']['smoke_present']} |",
            f"| Ready | {payload['public_repository_snapshot']['public_repository_snapshot_ready']} |",
            f"| Smoke ready | {payload['public_repository_snapshot']['source_repository_smoke_ready']} |",
            f"| Stage directory | `{payload['public_repository_snapshot']['stage_dir']}` |",
            f"| Commit | `{payload['public_repository_snapshot']['commit']}` |",
            f"| Source files | {payload['public_repository_snapshot'].get('source_file_count')} |",
            f"| Tracked files | {payload['public_repository_snapshot'].get('tracked_file_count')} |",
            "",
            "## External Receipts",
            "",
            f"Receipt registry: `{payload['receipts']}`",
            f"URL checks enabled: {payload['url_checks_enabled']}",
            "",
            "| Receipt | Ready | Missing | Invalid | URL check | Risk flag |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in payload["receipt_statuses"]:
        missing = ", ".join(row["missing"]) if row["missing"] else "none"
        invalid = ", ".join(row["invalid"]) if row["invalid"] else "none"
        check = row.get("url_check")
        if check:
            url_check = (
                f"status={check.get('status')}; ok={check.get('ok')}; "
                f"error={check.get('error') or 'none'}"
            )
        else:
            url_check = "not checked"
        risk = row["risk_flag"] or "none"
        lines.append(
            f"| {row['key']} | {row['ready']} | {missing} | {invalid} | {url_check} | {risk} |"
        )
    lines.extend(
        [
            "",
            "## Git State",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Repository detected | {payload['git']['repository_detected']} |",
            f"| Commit | `{payload['git']['commit']}` |",
            f"| Clean | {payload['git']['clean']} |",
            f"| Porcelain count | {payload['git']['porcelain_count']} |",
            "",
            "## Risk Flags",
            "",
        ]
    )
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    if payload["blocking_next_steps"]:
        lines.extend(["", "## Blocking Next Steps", ""])
        lines.extend(f"- {step}" for step in payload["blocking_next_steps"])
    lines.extend(
        [
            "",
            "## Strict Gate",
            "",
            "Run this command only after external receipts have been filled:",
            "",
            "```bash",
            "python scripts/audit_external_validation_readiness.py --strict",
            "# Use --check-urls without --strict to inspect URL reachability before all receipts are complete.",
            "```",
            "",
            "This file is generated by `scripts/audit_external_validation_readiness.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_audit(
        args.receipts,
        args.repository_snapshot_audit,
        args.repository_snapshot_smoke,
        args.archive,
        args.archive_sha256,
        check_urls=bool(args.check_urls or args.strict),
        url_timeout=args.url_timeout,
    )
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "local_artifact_release_ready": payload["local"]["local_artifact_release_ready"],
                "external_validation_ready": payload["external_validation_ready"],
                "clean_repository_ready": payload["clean_repository_ready"],
                "top_conference_release_ready": payload["top_conference_release_ready"],
                "risk_flags": payload["risk_flags"],
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if args.strict and not payload["top_conference_release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
