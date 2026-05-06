#!/usr/bin/env python
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_FILE_BYTES = 100_000_000

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

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "data",
    "dist",
}

REQUIRED_FILES = [
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
    "src/lottery/analysis.py",
    "scripts/verify_source_repository_snapshot.py",
    "scripts/verify_research_artifacts.py",
    "scripts/stage_public_repository_snapshot.py",
    "scripts/build_external_validation_receipt_template.py",
    "scripts/update_external_validation_receipts.py",
    "scripts/build_external_validation_runbook.py",
    "scripts/build_submission_handoff.py",
    "scripts/run_gpu_container_env_check.py",
    "scripts/build_paper_stats.py",
    "docs/public_release_manifest.md",
    "docs/release_anonymization_audit.md",
    "docs/submission_readiness_audit.md",
    "docs/reproducibility_manifest.md",
    "runs/paper_stats.json",
    "runs/public_release_manifest.json",
    "paper/main.tex",
    "paper/refs.bib",
    "paper/main.pdf",
    "paper/main_submission.pdf",
    "paper/neurips_submission.pdf",
    "paper/figures/gate1_controls.pdf",
    "paper/figures/cifar_movement.pdf",
    "paper/figures/cifar_trajectory.pdf",
    "paper/tables/statistical_summary.tex",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> list[Path]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts
        if any(part in EXCLUDED_DIRS for part in parts):
            continue
        if parts and parts[0] == "runs" and len(parts) > 2:
            continue
        paths.append(path)
    return sorted(paths, key=rel)


def is_text(path: Path) -> bool:
    return path.name in {"Dockerfile", "Dockerfile.gpu", "Makefile"} or path.suffix in TEXT_SUFFIXES


def pdf_pages(path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        return None
    completed = subprocess.run(
        [pdfinfo, str(path)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def scan_forbidden(paths: list[Path]) -> list[dict[str, Any]]:
    findings = []
    for path in paths:
        if not is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in FORBIDDEN_PATTERNS:
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            lines = sorted({text.count("\n", 0, match.start()) + 1 for match in matches})
            findings.append(
                {
                    "path": rel(path),
                    "pattern": name,
                    "count": len(matches),
                    "lines": lines[:10],
                }
            )
    return findings


def main() -> None:
    files = iter_files()
    by_rel = {rel(path): path for path in files}
    missing = [path for path in REQUIRED_FILES if path not in by_rel]
    if missing:
        fail(f"source repository snapshot missing required files: {missing}")

    oversized = [
        {"path": rel(path), "bytes": path.stat().st_size}
        for path in files
        if path.stat().st_size > MAX_SOURCE_FILE_BYTES
    ]
    if oversized:
        fail(f"source repository snapshot has oversized files: {oversized}")

    forbidden = scan_forbidden(files)
    if forbidden:
        fail(f"source repository snapshot has local identity/path findings: {forbidden[:10]}")

    workflow = (ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
    for phrase in [
        "make source-repository-check PYTHON=python",
        "Full artifact payload absent",
        "make ci-check paper-check PYTHON=python",
    ]:
        if phrase not in workflow:
            fail(f"source repository workflow missing phrase: {phrase}")

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for phrase in [
        "source-repository-check",
        "paper-existing-check",
        "verify_source_repository_snapshot.py",
    ]:
        if phrase not in makefile:
            fail(f"Makefile missing source repository phrase: {phrase}")

    pages = {
        "paper/main_submission.pdf": pdf_pages(ROOT / "paper" / "main_submission.pdf"),
        "paper/neurips_submission.pdf": pdf_pages(ROOT / "paper" / "neurips_submission.pdf"),
    }
    if pages["paper/main_submission.pdf"] != 9:
        fail(f"main-only source PDF should be 9 pages: {pages}")
    if pages["paper/neurips_submission.pdf"] is None or pages["paper/neurips_submission.pdf"] < 9:
        fail(f"NeurIPS source PDF page count unavailable or too small: {pages}")

    print(
        json.dumps(
            {
                "source_repository_snapshot_verified": True,
                "checked_files": len(files),
                "max_file_bytes": MAX_SOURCE_FILE_BYTES,
                "pdf_pages": pages,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"source repository verification failed: {exc}")
        raise SystemExit(1)
