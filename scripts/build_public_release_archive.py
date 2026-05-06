#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs" / "public_release_manifest.json"
DEFAULT_ANONYMIZATION_AUDIT = ROOT / "runs" / "release_anonymization_audit.json"
DEFAULT_ARCHIVE = ROOT / "dist" / "lottery_artifact_public_release_2026-05-06.tar.gz"
DEFAULT_OUT_JSON = ROOT / "runs" / "public_release_archive_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "public_release_archive_audit.md"
PACKAGE_ROOT = "lottery_artifact_public_release"

RELEASE_METADATA_PATHS = [
    "docs/public_release_manifest.md",
    "runs/public_release_manifest.json",
    "docs/release_anonymization_audit.md",
    "runs/release_anonymization_audit.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--anonymization-audit",
        type=Path,
        default=DEFAULT_ANONYMIZATION_AUDIT,
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_paths(manifest: dict[str, Any]) -> list[str]:
    paths = [
        str(entry["path"])
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]
    return sorted(paths)


def expected_paths(manifest: dict[str, Any]) -> list[str]:
    paths = set(manifest_paths(manifest))
    paths.update(RELEASE_METADATA_PATHS)
    return sorted(paths)


def safe_member_name(path: str) -> str:
    return f"{PACKAGE_ROOT}/{path}"


def add_file(tar: tarfile.TarFile, source: Path, member_name: str) -> None:
    info = tar.gettarinfo(str(source), arcname=member_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with source.open("rb") as f:
        tar.addfile(info, f)


def build_archive(paths: list[str], archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for rel in paths:
                    add_file(tar, ROOT / rel, safe_member_name(rel))


def audit_archive(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    anonymization_audit: dict[str, Any],
    archive: Path,
    paths: list[str],
) -> dict[str, Any]:
    risk_flags: list[str] = []
    missing_source_paths = [path for path in paths if not (ROOT / path).is_file()]
    if missing_source_paths:
        risk_flags.append("release_archive_source_paths_missing")

    if manifest.get("root") != ".":
        risk_flags.append("release_manifest_root_not_anonymized")
    if anonymization_audit.get("release_anonymization_ready") is not True:
        risk_flags.append("release_anonymization_audit_not_ready")
    if anonymization_audit.get("risk_flags") != []:
        risk_flags.append("release_anonymization_audit_has_flags")

    archive_exists = archive.exists() and archive.is_file()
    archive_bytes = archive.stat().st_size if archive_exists else 0
    archive_sha256 = sha256(archive) if archive_exists else ""
    if not archive_exists:
        risk_flags.append("release_archive_missing")
    elif archive_bytes < 100_000_000:
        risk_flags.append("release_archive_unexpectedly_small")

    expected_members = sorted(safe_member_name(path) for path in paths)
    actual_members: list[str] = []
    unsafe_members: list[str] = []
    non_file_members: list[str] = []
    member_size_mismatches: list[dict[str, object]] = []
    if archive_exists:
        try:
            with tarfile.open(archive, mode="r:gz") as tar:
                members = tar.getmembers()
                actual_members = sorted(member.name for member in members)
                for member in members:
                    if (
                        member.name.startswith("/")
                        or member.name.startswith("../")
                        or "/../" in member.name
                        or not member.name.startswith(f"{PACKAGE_ROOT}/")
                    ):
                        unsafe_members.append(member.name)
                    if not member.isfile():
                        non_file_members.append(member.name)
                    rel = member.name.removeprefix(f"{PACKAGE_ROOT}/")
                    source = ROOT / rel
                    if source.exists() and source.is_file() and source.stat().st_size != member.size:
                        member_size_mismatches.append(
                            {
                                "member": member.name,
                                "archive_bytes": member.size,
                                "source_bytes": source.stat().st_size,
                            }
                        )
        except tarfile.TarError:
            risk_flags.append("release_archive_not_readable_as_tar_gz")

    missing_members = sorted(set(expected_members) - set(actual_members))
    extra_members = sorted(set(actual_members) - set(expected_members))
    if missing_members or extra_members:
        risk_flags.append("release_archive_member_set_mismatch")
    if unsafe_members:
        risk_flags.append("release_archive_has_unsafe_member_paths")
    if non_file_members:
        risk_flags.append("release_archive_has_non_file_members")
    if member_size_mismatches:
        risk_flags.append("release_archive_member_size_mismatch")

    archive_ready = not risk_flags
    return {
        "archive_ready": archive_ready,
        "archive": relpath(archive),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "sha256_sidecar": relpath(Path(str(archive) + ".sha256")),
        "package_root": PACKAGE_ROOT,
        "manifest": relpath(manifest_path),
        "manifest_root": manifest.get("root"),
        "manifest_file_count": len(manifest_paths(manifest)),
        "release_metadata_paths": RELEASE_METADATA_PATHS,
        "release_metadata_count": len(RELEASE_METADATA_PATHS),
        "expected_member_count": len(expected_members),
        "actual_member_count": len(actual_members),
        "missing_source_paths": missing_source_paths[:50],
        "missing_members": missing_members[:50],
        "extra_members": extra_members[:50],
        "unsafe_members": unsafe_members[:50],
        "non_file_members": non_file_members[:50],
        "member_size_mismatches": member_size_mismatches[:50],
        "risk_flags": risk_flags,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    status = "ready" if payload["archive_ready"] else "not ready"
    lines = [
        "# Public Release Archive Audit",
        "",
        "This generated audit checks the local anonymous-review release tarball",
        "against the public release manifest plus release metadata sidecars.",
        "",
        f"Current status: {status}.",
        "",
        "## Archive",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Archive | `{payload['archive']}` |",
        f"| SHA256 | `{payload['archive_sha256']}` |",
        f"| SHA256 sidecar | `{payload['sha256_sidecar']}` |",
        f"| Bytes | {payload['archive_bytes']} |",
        f"| Package root | `{payload['package_root']}` |",
        f"| Manifest files | {payload['manifest_file_count']} |",
        f"| Release metadata sidecars | {payload['release_metadata_count']} |",
        f"| Expected members | {payload['expected_member_count']} |",
        f"| Actual members | {payload['actual_member_count']} |",
        "",
        "## Included Release Metadata",
        "",
    ]
    lines.extend(f"- `{path}`" for path in payload["release_metadata_paths"])
    lines.extend(["", "## Risk Flags", ""])
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Member Mismatches", ""])
    mismatch_sections = [
        ("Missing source paths", "missing_source_paths"),
        ("Missing archive members", "missing_members"),
        ("Extra archive members", "extra_members"),
        ("Unsafe member paths", "unsafe_members"),
        ("Non-file members", "non_file_members"),
        ("Member size mismatches", "member_size_mismatches"),
    ]
    wrote_any = False
    for title, key in mismatch_sections:
        values = payload[key]
        if values:
            wrote_any = True
            lines.extend([f"### {title}", ""])
            lines.extend(f"- `{value}`" for value in values)
            lines.append("")
    if not wrote_any:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This file is generated by `scripts/build_public_release_archive.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    anonymization_audit = load_json(args.anonymization_audit)
    paths = expected_paths(manifest)
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"release archive source paths missing: {missing[:10]}")
    if anonymization_audit.get("release_anonymization_ready") is not True:
        raise SystemExit("release anonymization audit is not ready")

    build_archive(paths, args.archive)
    digest = sha256(args.archive)
    sidecar = Path(str(args.archive) + ".sha256")
    sidecar.write_text(f"{digest}  {args.archive.name}\n", encoding="utf-8")
    payload = audit_archive(
        manifest_path=args.manifest,
        manifest=manifest,
        anonymization_audit=anonymization_audit,
        archive=args.archive,
        paths=paths,
    )
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "archive_ready": payload["archive_ready"],
                "risk_flags": payload["risk_flags"],
                "archive": payload["archive"],
                "archive_sha256": payload["archive_sha256"],
                "archive_bytes": payload["archive_bytes"],
                "member_count": payload["actual_member_count"],
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["archive_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
