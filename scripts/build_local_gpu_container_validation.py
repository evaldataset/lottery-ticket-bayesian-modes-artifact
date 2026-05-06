#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from run_gpu_container_env_check import (
    can_use_manual_driver_mount,
    manual_driver_mount_command,
    run_capture,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "lottery-training-gpu:2026-05-06"
DEFAULT_OUT_JSON = ROOT / "runs" / "local_gpu_container_validation.json"
DEFAULT_OUT_MD = ROOT / "docs" / "local_gpu_container_validation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def image_metadata(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return {
            "image": image,
            "present": False,
            "id": "",
            "size_bytes": 0,
            "error": result.stderr.strip(),
        }
    payload = json.loads(result.stdout)
    return {
        "image": image,
        "present": True,
        "id": str(payload.get("Id", "")),
        "size_bytes": int(payload.get("Size", 0)),
        "repo_tags": payload.get("RepoTags", []),
        "repo_digests": payload.get("RepoDigests", []),
    }


def extract_json_payload(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    if start < 0:
        return None
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError:
        return None


def run_gpu_check(image: str) -> dict[str, Any]:
    primary_command = ["docker", "run", "--rm", "--gpus", "all", image]
    primary = run_capture(primary_command)
    mode = "docker_gpus_all"
    result = primary
    command = primary_command
    fallback_used = False
    fallback_reason = ""

    if primary.returncode != 0 and can_use_manual_driver_mount(primary.stderr):
        fallback_used = True
        fallback_reason = "nvidia_container_toolkit_runtime_unavailable"
        mode = "manual_driver_mount"
        command = manual_driver_mount_command(image)
        result = run_capture(command)

    payload = extract_json_payload(result.stdout)
    return {
        "command": command,
        "mode": mode,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "returncode": result.returncode,
        "stdout_json": payload,
        "stderr_first_lines": result.stderr.strip().splitlines()[:8],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    image = image_metadata(args.image)
    gpu_check = run_gpu_check(args.image)
    check = gpu_check.get("stdout_json") if isinstance(gpu_check.get("stdout_json"), dict) else {}
    cuda_probe = check.get("cuda_probe", {}) if isinstance(check, dict) else {}
    ready = bool(
        image.get("present") is True
        and str(image.get("id", "")).startswith("sha256:")
        and gpu_check.get("returncode") == 0
        and check.get("status") == "ok"
        and check.get("cuda_available") is True
        and check.get("torch_cuda_version") == "13.0"
        and float(cuda_probe.get("matmul_sum", 0.0)) == 4096.0
    )
    risk_flags: list[str] = []
    if not ready:
        risk_flags.append("local_gpu_container_validation_failed")
    return {
        "schema_version": 1,
        "purpose": (
            "Local CUDA Docker validation receipt for the GPU training image. "
            "This is not an independent external receipt."
        ),
        "image": image,
        "gpu_check": gpu_check,
        "local_gpu_container_ready": ready,
        "risk_flags": risk_flags,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    image = payload["image"]
    check = payload["gpu_check"].get("stdout_json") or {}
    cuda_probe = check.get("cuda_probe", {}) if isinstance(check, dict) else {}
    status = "ready" if payload["local_gpu_container_ready"] else "not ready"
    lines = [
        "# Local GPU Container Validation",
        "",
        "This generated receipt records the local CUDA Docker validation for the",
        "GPU training image. It is useful submission evidence, but it is not an",
        "independent external GPU-host receipt.",
        "",
        f"Validation status: {status}.",
        "",
        "## Image",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Image | `{image['image']}` |",
        f"| Image ID | `{image['id']}` |",
        f"| Size bytes | {image['size_bytes']} |",
        "",
        "## Runtime Check",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Command mode | `{payload['gpu_check']['mode']}` |",
        f"| Fallback used | {payload['gpu_check']['fallback_used']} |",
        f"| Return code | {payload['gpu_check']['returncode']} |",
        f"| Package lock | `{check.get('package_lock', '')}` |",
        f"| Torch CUDA version | `{check.get('torch_cuda_version', '')}` |",
        f"| CUDA available | {check.get('cuda_available')} |",
        f"| CUDA device | `{cuda_probe.get('device_name', '')}` |",
        f"| CUDA device count | {cuda_probe.get('device_count', '')} |",
        f"| CUDA matmul sum | {cuda_probe.get('matmul_sum', '')} |",
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
            "This file is generated by `scripts/build_local_gpu_container_validation.py`.",
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
                "local_gpu_container_ready": payload["local_gpu_container_ready"],
                "risk_flags": payload["risk_flags"],
                "image_id": payload["image"].get("id"),
                "out_json": relpath(args.out_json),
                "out_md": relpath(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
