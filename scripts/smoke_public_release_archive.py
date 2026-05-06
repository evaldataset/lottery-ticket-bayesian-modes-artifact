#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "dist" / "lottery_artifact_public_release_2026-05-06.tar.gz"
DEFAULT_OUT_JSON = ROOT / "runs" / "public_release_archive_smoke.json"
DEFAULT_OUT_MD = ROOT / "docs" / "public_release_archive_smoke.md"
PACKAGE_ROOT = "lottery_artifact_public_release"

RELEASE_METADATA_PATHS = [
    "docs/public_release_manifest.md",
    "runs/public_release_manifest.json",
    "docs/release_anonymization_audit.md",
    "runs/release_anonymization_audit.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--verifier-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(tar: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], list[str]]:
    safe: list[tarfile.TarInfo] = []
    unsafe: list[str] = []
    for member in tar.getmembers():
        if (
            member.name.startswith("/")
            or member.name.startswith("../")
            or "/../" in member.name
            or not member.name.startswith(f"{PACKAGE_ROOT}/")
            or not member.isfile()
        ):
            unsafe.append(member.name)
        else:
            safe.append(member)
    return safe, unsafe


def manifest_paths(manifest: dict[str, Any]) -> list[str]:
    return sorted(
        str(entry["path"])
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )


def run_package_verifier(package_root: Path, timeout_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_research_artifacts.py",
            "--release-package-mode",
        ],
        cwd=package_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def build_smoke(archive: Path, timeout_seconds: int) -> dict[str, Any]:
    risk_flags: list[str] = []
    if not archive.exists():
        return {
            "release_archive_smoke_ready": False,
            "archive": relpath(archive),
            "archive_sha256": "",
            "risk_flags": ["release_archive_missing"],
        }

    archive_digest = sha256(archive)
    with tempfile.TemporaryDirectory(prefix="lottery_release_smoke_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, mode="r:gz") as tar:
            members, unsafe = safe_members(tar)
            if unsafe:
                risk_flags.append("release_archive_has_unsafe_members")
            tar.extractall(tmp_path, members=members)

        package_root = tmp_path / PACKAGE_ROOT
        if not package_root.is_dir():
            risk_flags.append("release_archive_package_root_missing")

        manifest_path = package_root / "runs" / "public_release_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        paths = manifest_paths(manifest)
        expected_files = sorted(set(paths + RELEASE_METADATA_PATHS))
        actual_files = sorted(
            path.relative_to(package_root).as_posix()
            for path in package_root.rglob("*")
            if path.is_file()
        )
        missing_files = sorted(set(expected_files) - set(actual_files))
        extra_files = sorted(set(actual_files) - set(expected_files))
        if missing_files or extra_files:
            risk_flags.append("release_archive_extracted_file_set_mismatch")

        hash_mismatches: list[str] = []
        by_path = {
            str(entry["path"]): str(entry.get("sha256", ""))
            for entry in manifest.get("files", [])
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        for rel, expected_digest in by_path.items():
            full_path = package_root / rel
            if not full_path.exists():
                continue
            if sha256(full_path) != expected_digest:
                hash_mismatches.append(rel)
        if hash_mismatches:
            risk_flags.append("release_archive_manifest_hash_mismatch")

        metadata_missing = [
            rel for rel in RELEASE_METADATA_PATHS if not (package_root / rel).is_file()
        ]
        if metadata_missing:
            risk_flags.append("release_archive_metadata_sidecars_missing")

        verifier = run_package_verifier(package_root, timeout_seconds)
        if verifier["returncode"] != 0:
            risk_flags.append("release_package_verifier_failed")

    return {
        "release_archive_smoke_ready": not risk_flags,
        "archive": relpath(archive),
        "archive_sha256": archive_digest,
        "package_root": PACKAGE_ROOT,
        "manifest_root": manifest.get("root"),
        "manifest_file_count": len(paths),
        "expected_file_count": len(expected_files),
        "actual_file_count": len(actual_files),
        "release_metadata_paths": RELEASE_METADATA_PATHS,
        "metadata_missing": metadata_missing,
        "missing_files": missing_files[:50],
        "extra_files": extra_files[:50],
        "hash_mismatches": hash_mismatches[:50],
        "checked_hash_count": len(by_path) - len(hash_mismatches),
        "verifier": verifier,
        "risk_flags": risk_flags,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    status = "ready" if payload["release_archive_smoke_ready"] else "not ready"
    verifier = payload.get("verifier", {})
    lines = [
        "# Public Release Archive Smoke Test",
        "",
        "This generated smoke test extracts the local anonymous-review release",
        "tarball, verifies manifest hashes inside the extracted package, and",
        "runs the artifact verifier in release-package mode.",
        "",
        f"Current status: {status}.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Archive | `{payload.get('archive')}` |",
        f"| Archive SHA256 | `{payload.get('archive_sha256')}` |",
        f"| Manifest files | {payload.get('manifest_file_count')} |",
        f"| Expected extracted files | {payload.get('expected_file_count')} |",
        f"| Actual extracted files | {payload.get('actual_file_count')} |",
        f"| Checked manifest hashes | {payload.get('checked_hash_count')} |",
        f"| Verifier return code | {verifier.get('returncode')} |",
        "",
        "## Risk Flags",
        "",
    ]
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Verifier Output", "", "```text"])
    lines.append(str(verifier.get("stdout_tail", "")).strip())
    if verifier.get("stderr_tail"):
        lines.append("")
        lines.append(str(verifier["stderr_tail"]).strip())
    lines.extend(
        [
            "```",
            "",
            "This file is generated by `scripts/smoke_public_release_archive.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_smoke(args.archive, args.verifier_timeout_seconds)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "release_archive_smoke_ready": payload["release_archive_smoke_ready"],
                "risk_flags": payload["risk_flags"],
                "archive": payload.get("archive"),
                "archive_sha256": payload.get("archive_sha256"),
                "checked_hash_count": payload.get("checked_hash_count"),
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["release_archive_smoke_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
