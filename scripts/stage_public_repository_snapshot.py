#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = ROOT / "dist" / "lottery_public_repository_snapshot"
DEFAULT_OUT_JSON = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "public_repository_snapshot_audit.md"
MARKER = ".lottery_public_repository_snapshot"
GITHUB_SOFT_FILE_LIMIT_BYTES = 100_000_000
FIXED_COMMIT_DATE = "2026-05-06T00:00:00+0000"

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
    "runs/*.csv",
    "runs/*.json",
]

EXCLUDED = {
    # Mutable post-release evidence. The public source snapshot commit is one
    # of the values recorded in this registry, so the registry cannot be part
    # of the commit whose hash it records.
    "docs/external_validation_receipts.json",
    "docs/external_validation_readiness_audit.md",
    "docs/external_validation_receipt_template.md",
    "docs/external_validation_runbook.md",
    "docs/external_gpu_container_receipt.md",
    "docs/public_repository_snapshot_audit.md",
    "docs/public_repository_snapshot_smoke.md",
    "docs/submission_handoff.md",
    "runs/external_validation_readiness_audit.json",
    "runs/external_validation_receipt_template.json",
    "runs/external_validation_runbook.json",
    "runs/external_gpu_container_receipt.json",
    "runs/public_repository_snapshot_audit.json",
    "runs/public_repository_snapshot_smoke.json",
    "runs/submission_handoff.json",
}

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
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--max-file-bytes", type=int, default=GITHUB_SOFT_FILE_LIMIT_BYTES)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_git(args: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    return {
        "args": ["git", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def iter_source_files() -> list[Path]:
    paths: set[Path] = set()
    for rel in ROOT_FILES:
        path = ROOT / rel
        if path.is_file():
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


def prepare_stage_dir(path: Path) -> None:
    marker = path / MARKER
    if path.exists():
        if not marker.exists():
            raise RuntimeError(f"refusing to replace unmarked stage directory: {relpath(path)}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    marker.write_text("generated public repository snapshot\n", encoding="utf-8")


def copy_files(files: Iterable[Path], stage_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source in files:
        rel = source.relative_to(ROOT)
        target = stage_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "path": rel.as_posix(),
                "bytes": source.stat().st_size,
            }
        )
    return rows


def is_text_path(path: Path) -> bool:
    return path.name in {"Dockerfile", "Dockerfile.gpu", "Makefile"} or path.suffix in TEXT_SUFFIXES


def scan_text_files(stage_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row in rows:
        rel = str(row["path"])
        path = stage_dir / rel
        if not is_text_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in FORBIDDEN_PATTERNS:
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            line_numbers = sorted({text.count("\n", 0, match.start()) + 1 for match in matches})
            findings.append(
                {
                    "path": rel,
                    "pattern": name,
                    "count": len(matches),
                    "lines": line_numbers[:10],
                }
            )
    return findings


def stage_git_repository(stage_dir: Path) -> dict[str, Any]:
    git_steps: list[dict[str, Any]] = []
    init = run_git(["init", "--initial-branch=main"], stage_dir)
    if init["returncode"] != 0:
        init = run_git(["init"], stage_dir)
        git_steps.append(init)
        if init["returncode"] == 0:
            git_steps.append(run_git(["branch", "-M", "main"], stage_dir))
    else:
        git_steps.append(init)
    for args in [
        ["config", "user.name", "Anonymous Authors"],
        ["config", "user.email", "anonymous@example.com"],
        ["add", "-A"],
    ]:
        git_steps.append(run_git(args, stage_dir))
    env = {
        **dict(os.environ),
        "GIT_AUTHOR_DATE": FIXED_COMMIT_DATE,
        "GIT_COMMITTER_DATE": FIXED_COMMIT_DATE,
    }
    git_steps.append(
        run_git(["commit", "-m", "Anonymous artifact repository snapshot"], stage_dir, env=env)
    )
    status = run_git(["status", "--porcelain"], stage_dir)
    commit = run_git(["rev-parse", "HEAD"], stage_dir)
    files = run_git(["ls-files"], stage_dir)
    count = run_git(["count-objects", "-vH"], stage_dir)
    git_steps.extend([status, commit, files, count])
    tracked_files = [line for line in files["stdout"].splitlines() if line.strip()]
    status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    return {
        "steps": [
            {
                "args": step["args"],
                "returncode": step["returncode"],
                "stderr_tail": step["stderr"][-500:],
            }
            for step in git_steps
        ],
        "commit": commit["stdout"] if commit["returncode"] == 0 else "",
        "status_porcelain": status_lines,
        "tracked_file_count": len(tracked_files),
        "count_objects": count["stdout"],
        "git_clean": status["returncode"] == 0 and not status_lines,
        "git_ready": all(step["returncode"] == 0 for step in git_steps) and not status_lines,
    }


def build_audit(stage_dir: Path, max_file_bytes: int) -> dict[str, Any]:
    files = iter_source_files()
    prepare_stage_dir(stage_dir)
    rows = copy_files(files, stage_dir)
    text_findings = scan_text_files(stage_dir, rows)
    oversized_files = [
        row for row in rows if int(row["bytes"]) > max_file_bytes
    ]
    missing_expected = [
        rel for rel in ROOT_FILES if not (ROOT / rel).is_file()
    ]
    git = stage_git_repository(stage_dir)
    risk_flags: list[str] = []
    if missing_expected:
        risk_flags.append("repository_snapshot_missing_root_files")
    if oversized_files:
        risk_flags.append("repository_snapshot_contains_large_files")
    if text_findings:
        risk_flags.append("repository_snapshot_text_contains_local_identity_or_paths")
    if not git["git_ready"]:
        risk_flags.append("repository_snapshot_git_commit_not_clean")
    if git["tracked_file_count"] != len(rows) + 1:
        risk_flags.append("repository_snapshot_tracked_file_count_mismatch")
    return {
        "stage_dir": relpath(stage_dir),
        "marker": MARKER,
        "source_file_count": len(rows),
        "tracked_file_count": git["tracked_file_count"],
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "max_file_bytes": max_file_bytes,
        "largest_files": sorted(rows, key=lambda row: int(row["bytes"]), reverse=True)[:20],
        "oversized_files": oversized_files[:20],
        "missing_expected": missing_expected,
        "text_findings": text_findings[:100],
        "text_finding_count": len(text_findings),
        "git": git,
        "risk_flags": risk_flags,
        "public_repository_snapshot_ready": not risk_flags,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    status = "ready" if payload["public_repository_snapshot_ready"] else "not ready"
    lines = [
        "# Public Repository Snapshot Audit",
        "",
        "This generated audit creates a source-only anonymous git repository",
        "snapshot under `dist/` for public-repository upload preparation. It",
        "excludes raw dataset caches, the release tarball, and nested large run",
        "artifacts, plus the mutable external receipt registry; the full",
        "run-artifact package remains the separate public release archive.",
        "",
        f"Current status: {status}.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Stage directory | `{payload['stage_dir']}` |",
        f"| Source files | {payload['source_file_count']} |",
        f"| Tracked files | {payload['tracked_file_count']} |",
        f"| Total bytes | {fmt_bytes(int(payload['total_bytes']))} |",
        f"| Max public git file bytes | {payload['max_file_bytes']} |",
        f"| Git commit | `{payload['git']['commit']}` |",
        f"| Git clean | {payload['git']['git_clean']} |",
        "",
        "## Largest Files",
        "",
        "| Path | Bytes |",
        "| --- | ---: |",
    ]
    for row in payload["largest_files"]:
        lines.append(f"| `{row['path']}` | {row['bytes']} |")
    lines.extend(["", "## Risk Flags", ""])
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Oversized Files", ""])
    if payload["oversized_files"]:
        for row in payload["oversized_files"]:
            lines.append(f"- `{row['path']}` ({row['bytes']} bytes)")
    else:
        lines.append("- none")
    lines.extend(["", "## Text Findings", ""])
    if payload["text_findings"]:
        lines.append("| Path | Pattern | Count | Lines |")
        lines.append("| --- | --- | ---: | --- |")
        for finding in payload["text_findings"]:
            lines.append(
                f"| `{finding['path']}` | {finding['pattern']} | "
                f"{finding['count']} | {finding['lines']} |"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Upload Use",
            "",
            "The staged directory is a clean local git repository. Add an anonymous",
            "remote and push it only after confirming the repository URL and branch",
            "match the external-validation receipt registry.",
            "",
            "This file is generated by `scripts/stage_public_repository_snapshot.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        payload = build_audit(args.stage_dir, args.max_file_bytes)
    except FileNotFoundError as exc:
        if exc.filename == "git":
            payload = {
                "stage_dir": relpath(args.stage_dir),
                "marker": MARKER,
                "source_file_count": 0,
                "tracked_file_count": 0,
                "total_bytes": 0,
                "max_file_bytes": args.max_file_bytes,
                "largest_files": [],
                "oversized_files": [],
                "missing_expected": [],
                "text_findings": [],
                "text_finding_count": 0,
                "git": {"commit": "", "git_clean": False, "git_ready": False},
                "risk_flags": ["git_not_installed"],
                "public_repository_snapshot_ready": False,
            }
        else:
            raise
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "public_repository_snapshot_ready": payload["public_repository_snapshot_ready"],
                "risk_flags": payload["risk_flags"],
                "commit": payload["git"].get("commit", ""),
                "source_file_count": payload["source_file_count"],
                "tracked_file_count": payload["tracked_file_count"],
                "stage_dir": relpath(args.stage_dir),
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["public_repository_snapshot_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
