#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs" / "public_release_manifest.json"
DEFAULT_OUT_JSON = ROOT / "runs" / "release_anonymization_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "release_anonymization_audit.md"

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".csv",
    ".dockerignore",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}

EXTRA_SCAN_PATHS = [
    "docs/public_release_manifest.md",
    "runs/public_release_manifest.json",
    "docs/release_anonymization_audit.md",
    "runs/release_anonymization_audit.json",
]

FORBIDDEN_PATH_PREFIXES = [
    ".git/",
    ".venv/",
    "__pycache__/",
    "data/",
    "discarded_runs/",
]

LOCAL_USERNAME = "suan" + "lab"
LOCAL_HOSTNAME = "MyUbuntu" + "5090"
LOCAL_PROJECT_PATH = "/Projects/" + "Lottery"

FORBIDDEN_PATTERNS = [
    ("home_directory", re.compile(r"/home/[A-Za-z0-9_.-]+")),
    ("users_directory", re.compile(r"/Users/[A-Za-z0-9_.-]+")),
    ("windows_user_directory", re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.-]+")),
    ("local_username", re.compile(rf"\b{re.escape(LOCAL_USERNAME)}\b", re.IGNORECASE)),
    ("local_hostname", re.compile(rf"\b{re.escape(LOCAL_HOSTNAME)}\b", re.IGNORECASE)),
    ("absolute_project_path", re.compile(rf"{re.escape(LOCAL_PROJECT_PATH)}\b")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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


def is_text_path(path: Path) -> bool:
    return path.name in {"Dockerfile", "Dockerfile.gpu", "Makefile"} or path.suffix in TEXT_SUFFIXES


def read_text(path: Path) -> str | None:
    if not is_text_path(path):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def scan_text(path: Path, text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for name, pattern in FORBIDDEN_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        line_numbers = sorted({text.count("\n", 0, match.start()) + 1 for match in matches})
        findings.append(
            {
                "path": relpath(path),
                "pattern": name,
                "count": len(matches),
                "lines": line_numbers[:10],
            }
        )
    return findings


def build_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    entries = [
        entry
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]
    file_paths = [str(entry["path"]) for entry in entries]
    extra_paths = [path for path in EXTRA_SCAN_PATHS if (ROOT / path).exists()]
    scan_paths = sorted(set(file_paths + extra_paths))

    risk_flags: list[str] = []
    manifest_root = manifest.get("root")
    if manifest_root not in {None, "", "."}:
        risk_flags.append("manifest_contains_absolute_root")

    forbidden_manifest_paths = [
        path
        for path in file_paths
        if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES)
    ]
    if forbidden_manifest_paths:
        risk_flags.append("manifest_contains_forbidden_local_paths")

    missing_paths = [path for path in scan_paths if not (ROOT / path).exists()]
    if missing_paths:
        risk_flags.append("manifest_references_missing_files")

    findings: list[dict[str, object]] = []
    scanned_text_files = 0
    skipped_binary_files = 0
    for rel in scan_paths:
        path = ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            skipped_binary_files += 1
            continue
        scanned_text_files += 1
        findings.extend(scan_text(path, text))
    if findings:
        risk_flags.append("release_text_contains_local_identity_or_absolute_paths")

    release_anonymization_ready = not risk_flags
    return {
        "manifest": relpath(manifest_path),
        "manifest_root": manifest_root,
        "manifest_file_count": len(file_paths),
        "scanned_paths": len(scan_paths),
        "scanned_text_files": scanned_text_files,
        "skipped_binary_files": skipped_binary_files,
        "extra_scan_paths": extra_paths,
        "forbidden_manifest_paths": forbidden_manifest_paths[:50],
        "missing_paths": missing_paths[:50],
        "findings": findings[:200],
        "finding_count": len(findings),
        "risk_flags": risk_flags,
        "release_anonymization_ready": release_anonymization_ready,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    status = "ready" if payload["release_anonymization_ready"] else "not ready"
    lines = [
        "# Release Anonymization Audit",
        "",
        "This generated audit checks the public release manifest and included",
        "text artifacts for local user names, host names, and absolute local",
        "workspace paths that could break anonymous review.",
        "",
        f"Current status: {status}.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Manifest | `{payload['manifest']}` |",
        f"| Manifest root | `{payload['manifest_root']}` |",
        f"| Manifest files | {payload['manifest_file_count']} |",
        f"| Scanned paths | {payload['scanned_paths']} |",
        f"| Scanned text files | {payload['scanned_text_files']} |",
        f"| Skipped binary files | {payload['skipped_binary_files']} |",
        f"| Finding count | {payload['finding_count']} |",
        "",
        "## Risk Flags",
        "",
    ]
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        lines.append("| Path | Pattern | Count | Lines |")
        lines.append("| --- | --- | ---: | --- |")
        for finding in payload["findings"]:
            lines.append(
                f"| `{finding['path']}` | {finding['pattern']} | "
                f"{finding['count']} | {finding['lines']} |"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This file is generated by `scripts/audit_release_anonymization.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_audit(args.manifest)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "release_anonymization_ready": payload["release_anonymization_ready"],
                "risk_flags": payload["risk_flags"],
                "finding_count": payload["finding_count"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )
    if not payload["release_anonymization_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
