#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_local_gpu_container_validation import image_metadata, run_gpu_check


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "lottery-training-gpu:2026-05-06"
DEFAULT_TEMPLATE = ROOT / "runs" / "external_validation_receipt_template.json"
DEFAULT_SNAPSHOT_AUDIT = ROOT / "runs" / "public_repository_snapshot_audit.json"
DEFAULT_OUT_JSON = ROOT / "runs" / "external_gpu_container_receipt.json"
DEFAULT_OUT_MD = ROOT / "docs" / "external_gpu_container_receipt.md"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CUDA Docker validation on an independent GPU host and "
            "write a machine-readable receipt suitable for upload as external "
            "GPU-container evidence."
        )
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--expected-commit",
        default="",
        help=(
            "Expected public source snapshot commit. Defaults to the current "
            "external-validation receipt template, then the snapshot audit."
        ),
    )
    parser.add_argument(
        "--observed-commit",
        default="",
        help=(
            "Observed source commit when running from an extracted archive "
            "rather than a git checkout. Public-repository validation should "
            "prefer the automatic git rev-parse value."
        ),
    )
    parser.add_argument(
        "--evidence-url",
        default="",
        help=(
            "Optional public URL where this receipt/log is uploaded. When set, "
            "the output includes a ready-to-run receipt-registry update command."
        ),
    )
    parser.add_argument(
        "--require-evidence-url",
        action="store_true",
        help="Fail unless --evidence-url is provided.",
    )
    parser.add_argument(
        "--host-label",
        default="external-cuda-host",
        help="Non-identifying label for the external CUDA host.",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--snapshot-audit", type=Path, default=DEFAULT_SNAPSHOT_AUDIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_or_none(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_text(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def json_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    result = run_text(command, timeout=timeout)
    payload: Any | None = None
    if result["returncode"] == 0 and result["stdout"]:
        try:
            payload = json.loads(str(result["stdout"]))
        except json.JSONDecodeError:
            payload = None
    return {**result, "json": payload}


def expected_commit(args: argparse.Namespace) -> str:
    if args.expected_commit.strip():
        return args.expected_commit.strip()
    template = load_json_or_none(args.template)
    if isinstance(template, dict):
        commit = (
            template.get("local_facts", {})
            .get("source_repository_commit", "")
        )
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    snapshot = load_json_or_none(args.snapshot_audit)
    if isinstance(snapshot, dict):
        commit = snapshot.get("git", {}).get("commit", "")
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    return ""


def current_git_commit() -> str:
    result = run_text(["git", "rev-parse", "HEAD"])
    if result["returncode"] != 0:
        return ""
    return str(result["stdout"]).strip()


def observed_commit(args: argparse.Namespace) -> str:
    if args.observed_commit.strip():
        return args.observed_commit.strip()
    return current_git_commit()


def nvidia_smi_host_summary() -> dict[str, Any]:
    query = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    rows: list[dict[str, str]] = []
    if query["returncode"] == 0:
        for line in str(query["stdout"]).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                rows.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_total_mib": parts[2],
                    }
                )
            elif line.strip():
                rows.append({"raw": line.strip()})
    return {
        "returncode": query["returncode"],
        "gpus": rows,
        "stderr_first_lines": str(query["stderr"]).splitlines()[:8],
    }


def host_metadata(host_label: str) -> dict[str, Any]:
    docker = json_command(["docker", "version", "--format", "{{json .}}"])
    info = json_command(["docker", "info", "--format", "{{json .}}"])
    return {
        "host_label": host_label,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "docker_version": docker,
        "docker_info": info,
        "nvidia_smi": nvidia_smi_host_summary(),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    expected = expected_commit(args)
    observed = observed_commit(args)
    observed_commit_inferred = False
    if not observed and expected:
        observed = expected
        observed_commit_inferred = True
    image = image_metadata(args.image)
    gpu_check = run_gpu_check(args.image)
    check = gpu_check.get("stdout_json") if isinstance(gpu_check.get("stdout_json"), dict) else {}
    cuda_probe = check.get("cuda_probe", {}) if isinstance(check, dict) else {}

    risk_flags: list[str] = []
    warning_flags: list[str] = []

    if not HEX40_RE.fullmatch(expected):
        risk_flags.append("expected_source_commit_missing")
    if not HEX40_RE.fullmatch(observed):
        risk_flags.append("observed_source_commit_missing")
    if expected and observed and expected != observed:
        risk_flags.append("source_commit_mismatch")
    if image.get("present") is not True:
        risk_flags.append("gpu_image_not_present")
    if not str(image.get("id", "")).startswith("sha256:"):
        risk_flags.append("gpu_image_id_missing")
    if gpu_check.get("returncode") != 0:
        risk_flags.append("gpu_container_env_check_failed")
    if check.get("status") != "ok":
        risk_flags.append("gpu_container_env_json_missing")
    if check.get("cuda_available") is not True:
        risk_flags.append("cuda_not_available")
    if check.get("torch_cuda_version") != "13.0":
        risk_flags.append("torch_cuda_version_mismatch")
    if float(cuda_probe.get("matmul_sum", 0.0) or 0.0) != 4096.0:
        risk_flags.append("cuda_matmul_probe_failed")

    evidence_url = args.evidence_url.strip()
    if not evidence_url:
        warning_flags.append("evidence_url_not_supplied")
        if args.require_evidence_url:
            risk_flags.append("evidence_url_required")
    if observed_commit_inferred:
        warning_flags.append("observed_commit_inferred_from_expected")

    host_validation_ready = not any(
        flag
        for flag in risk_flags
        if flag != "evidence_url_required"
    )
    registry_update_ready = host_validation_ready and bool(evidence_url)
    image_id = str(image.get("id", ""))
    receipt_candidate = {
        "status": "observed" if registry_update_ready else "pending",
        "url": evidence_url,
        "commit": observed,
        "image_digest": image_id,
        "passed": bool(host_validation_ready),
        "notes": (
            "External CUDA-host GPU container receipt generated by "
            "scripts/build_external_gpu_container_receipt.py."
        ),
    }

    return {
        "schema_version": 1,
        "purpose": (
            "Independent CUDA-host receipt for the GPU training Docker image. "
            "Upload this JSON or its Markdown companion and use the uploaded "
            "URL as the external_gpu_container receipt URL."
        ),
        "external_gpu_container_host_validation_ready": host_validation_ready,
        "receipt_registry_update_ready": registry_update_ready,
        "risk_flags": risk_flags,
        "warning_flags": warning_flags,
        "expected_source_commit": expected,
        "observed_source_commit": observed,
        "evidence_url": evidence_url,
        "image": image,
        "gpu_check": gpu_check,
        "host": host_metadata(args.host_label),
        "receipt_candidate": {"external_gpu_container": receipt_candidate},
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_command(payload: dict[str, Any]) -> list[str]:
    candidate = payload["receipt_candidate"]["external_gpu_container"]
    url = candidate["url"] or "<uploaded-external-gpu-receipt-url>"
    digest = candidate["image_digest"] or "<sha256-image-id-or-digest>"
    return [
        ".venv/bin/python scripts/update_external_validation_receipts.py --write \\",
        f"  --external-gpu-url {url} \\",
        f"  --external-gpu-image-digest {digest} --external-gpu-passed",
    ]


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    image = payload["image"]
    check = payload["gpu_check"].get("stdout_json") or {}
    cuda_probe = check.get("cuda_probe", {}) if isinstance(check, dict) else {}
    host_status = (
        "ready"
        if payload["external_gpu_container_host_validation_ready"]
        else "not ready"
    )
    registry_status = (
        "ready" if payload["receipt_registry_update_ready"] else "not ready"
    )
    lines = [
        "# External GPU Container Receipt",
        "",
        "This generated receipt records an independent CUDA-host validation of",
        "the GPU training Docker image. Upload this JSON or Markdown file and",
        "use the uploaded URL in `docs/external_validation_receipts.json`.",
        "",
        f"Host validation status: {host_status}.",
        f"Receipt-registry update status: {registry_status}.",
        "",
        "## Source",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Expected commit | `{payload['expected_source_commit']}` |",
        f"| Observed commit | `{payload['observed_source_commit']}` |",
        f"| Evidence URL | `{payload['evidence_url'] or '<upload this receipt first>'}` |",
        "",
        "## Image",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Image | `{image.get('image', '')}` |",
        f"| Image ID | `{image.get('id', '')}` |",
        f"| Size bytes | {image.get('size_bytes', 0)} |",
        "",
        "## Runtime Check",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Command mode | `{payload['gpu_check'].get('mode', '')}` |",
        f"| Fallback used | {payload['gpu_check'].get('fallback_used')} |",
        f"| Return code | {payload['gpu_check'].get('returncode')} |",
        f"| Package lock | `{check.get('package_lock', '')}` |",
        f"| Torch CUDA version | `{check.get('torch_cuda_version', '')}` |",
        f"| CUDA available | {check.get('cuda_available')} |",
        f"| CUDA device | `{cuda_probe.get('device_name', '')}` |",
        f"| CUDA device count | {cuda_probe.get('device_count', '')} |",
        f"| CUDA matmul sum | {cuda_probe.get('matmul_sum', '')} |",
        "",
        "## Host Metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Host label | `{payload['host'].get('host_label', '')}` |",
        f"| Timestamp UTC | `{payload['host'].get('timestamp_utc', '')}` |",
        f"| Platform | `{payload['host'].get('platform', {})}` |",
        f"| NVIDIA GPUs | `{payload['host'].get('nvidia_smi', {}).get('gpus', [])}` |",
        "",
        "## Receipt Update Command",
        "",
        "```bash",
        *update_command(payload),
        "```",
        "",
        "## Risk Flags",
        "",
    ]
    lines.extend(f"- {flag}" for flag in payload["risk_flags"] or ["none"])
    lines.extend(["", "## Warning Flags", ""])
    lines.extend(f"- {flag}" for flag in payload["warning_flags"] or ["none"])
    lines.extend(
        [
            "",
            "This file is generated by `scripts/build_external_gpu_container_receipt.py`.",
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
                "external_gpu_container_host_validation_ready": payload[
                    "external_gpu_container_host_validation_ready"
                ],
                "receipt_registry_update_ready": payload[
                    "receipt_registry_update_ready"
                ],
                "risk_flags": payload["risk_flags"],
                "warning_flags": payload["warning_flags"],
                "image_id": payload["image"].get("id"),
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )
    if payload["risk_flags"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
