#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = ROOT / "dist" / "lottery_public_repository_snapshot"
DEFAULT_OUT_JSON = ROOT / "runs" / "public_repository_snapshot_smoke.json"
DEFAULT_OUT_MD = ROOT / "docs" / "public_repository_snapshot_smoke.md"
SUCCESS_MARKER = "source_repository_snapshot_verified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sanitize_output(text: str) -> str:
    return text.replace(str(ROOT), "<workspace>").replace(
        "/tmp/lottery_public_repo_smoke_", "/tmp/<lottery_public_repo_smoke>_"
    )


def tail_with_marker(text: str, marker: str, tail_bytes: int = 4000) -> tuple[str, bool]:
    marker_seen = marker in text
    tail = sanitize_output(text[-tail_bytes:])
    if marker_seen and marker not in tail:
        prefix = f"{marker}: present before truncated output\n...\n"
        tail = prefix + tail[-max(0, tail_bytes - len(prefix)) :]
    return tail, marker_seen


def run_check(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["make", "source-repository-check", f"PYTHON={sys.executable}"],
        cwd=path,
        check=False,
        text=True,
        capture_output=True,
    )
    stdout_tail, success_marker_seen = tail_with_marker(completed.stdout, SUCCESS_MARKER)
    stderr_tail, _ = tail_with_marker(completed.stderr, SUCCESS_MARKER)
    return {
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stderr_bytes": len(completed.stderr.encode("utf-8")),
        "success_marker_seen": success_marker_seen,
    }


def git_status(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=False,
        text=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "status_returncode": status.returncode,
        "porcelain": [line for line in status.stdout.splitlines() if line.strip()],
        "commit": commit.stdout.strip() if commit.returncode == 0 else "",
    }


def build_smoke(stage_dir: Path) -> dict[str, Any]:
    risk_flags: list[str] = []
    if not stage_dir.is_dir():
        risk_flags.append("public_repository_snapshot_stage_dir_missing")
        return {
            "stage_dir": relpath(stage_dir),
            "source_repository_smoke_ready": False,
            "risk_flags": risk_flags,
            "check": {
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "success_marker_seen": False,
            },
            "git": {"porcelain": [], "commit": ""},
            "post_check_git": {"porcelain": [], "commit": ""},
        }
    with tempfile.TemporaryDirectory(prefix="lottery_public_repo_smoke_") as tmp:
        tmp_path = Path(tmp) / "snapshot"
        shutil.copytree(stage_dir, tmp_path, symlinks=False)
        git = git_status(tmp_path)
        check = run_check(tmp_path)
        post_check_git = git_status(tmp_path)
    if check["returncode"] != 0:
        risk_flags.append("source_repository_check_failed")
    if not check["success_marker_seen"]:
        risk_flags.append("source_repository_check_success_marker_missing")
    if git["porcelain"]:
        risk_flags.append("source_repository_snapshot_not_clean_before_check")
    return {
        "stage_dir": relpath(stage_dir),
        "source_repository_smoke_ready": not risk_flags,
        "risk_flags": risk_flags,
        "check": check,
        "git": git,
        "post_check_git": post_check_git,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    status = "ready" if payload["source_repository_smoke_ready"] else "not ready"
    lines = [
        "# Public Repository Snapshot Smoke Test",
        "",
        "This generated smoke test copies the staged source-only repository to a",
        "temporary directory and runs `make source-repository-check` there. It",
        "models the public GitHub Actions source-repository path without the large",
        "full artifact payload.",
        "",
        f"Current status: {status}.",
        f"Success marker: `{SUCCESS_MARKER}`.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Stage directory | `{payload['stage_dir']}` |",
        f"| Check return code | {payload['check']['returncode']} |",
        f"| Success marker seen | {payload['check']['success_marker_seen']} |",
        f"| Snapshot commit | `{payload['git']['commit']}` |",
        f"| Pre-check clean paths | {len(payload['git']['porcelain'])} |",
        f"| Post-check modified paths | {len(payload['post_check_git']['porcelain'])} |",
        "",
        "## Risk Flags",
        "",
    ]
    if payload["risk_flags"]:
        lines.extend(f"- {flag}" for flag in payload["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Check Output Tail",
            "",
            "```text",
            str(payload["check"]["stdout_tail"])[-2000:],
            "```",
            "",
            "This file is generated by `scripts/smoke_public_repository_snapshot.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_smoke(args.stage_dir)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "source_repository_smoke_ready": payload["source_repository_smoke_ready"],
                "risk_flags": payload["risk_flags"],
                "stage_dir": relpath(args.stage_dir),
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if not payload["source_repository_smoke_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
