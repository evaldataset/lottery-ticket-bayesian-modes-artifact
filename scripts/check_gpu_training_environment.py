#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_LOCK = ROOT / "requirements-gpu-lock.txt"
REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^#\s]+)$")


def fail(message: str) -> None:
    raise AssertionError(message)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise AssertionError(f"missing package: {name}") from exc


def load_lock() -> dict[str, Any]:
    with (ROOT / "docs" / "environment_lock.json").open(encoding="utf-8") as f:
        return json.load(f)


def parse_requirement_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = REQUIREMENT_RE.match(line)
        if match is None:
            fail(f"unsupported requirement in {path.name}:{lineno}: {raw_line}")
        name, version = match.groups()
        packages[name] = version
    if not packages:
        fail(f"package lock is empty: {path}")
    return packages


def nvidia_smi_summary() -> dict[str, str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return {"raw": line}
    return {"name": parts[0], "driver_version": parts[1], "memory_total_mib": parts[2]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Only check package versions and skip the CUDA runtime assertion.",
    )
    parser.add_argument(
        "--package-lock",
        type=Path,
        default=DEFAULT_PACKAGE_LOCK,
        help="Requirement lock to validate inside the GPU training environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = load_lock()
    expected_packages = parse_requirement_lock(args.package_lock)
    locked_packages = lock["packages"]
    lock_mismatches = {
        name: {"expected_in_environment_lock": version, "observed": locked_packages.get(name)}
        for name, version in expected_packages.items()
        if locked_packages.get(name) != version
    }
    if lock_mismatches:
        fail(f"GPU package lock does not match docs/environment_lock.json: {lock_mismatches}")
    for name in ["torch", "torchvision"]:
        if name not in expected_packages:
            fail(f"GPU package lock missing required package: {name}")
    observed_packages = {
        name: package_version(name)
        for name in sorted(expected_packages)
    }
    mismatches = {
        name: {"expected": expected_packages[name], "observed": observed_packages[name]}
        for name in expected_packages
        if observed_packages[name] != expected_packages[name]
    }
    if mismatches:
        fail(f"package version mismatches: {mismatches}")

    import torch
    import torchvision

    if observed_packages["torch"] != torch.__version__.split("+")[0]:
        fail("imported torch version does not match package metadata")
    if observed_packages["torchvision"] != torchvision.__version__.split("+")[0]:
        fail("imported torchvision version does not match package metadata")

    expected_cuda = str(lock["torch"].get("cuda_version"))
    observed_cuda = str(torch.version.cuda)
    cuda_available = bool(torch.cuda.is_available())
    if observed_cuda != expected_cuda:
        fail(f"torch CUDA version mismatch: expected {expected_cuda}, observed {observed_cuda}")
    if not args.allow_no_cuda and not cuda_available:
        fail("CUDA is not available to torch; run with a CUDA host and --gpus all")

    cuda_probe: dict[str, Any] | None = None
    if cuda_available:
        device = torch.device("cuda")
        tensor = torch.ones((16, 16), device=device)
        value = float((tensor @ tensor).sum().detach().cpu().item())
        torch.cuda.synchronize()
        cuda_probe = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "matmul_sum": value,
        }
        if value != 4096.0:
            fail(f"unexpected CUDA matmul check value: {value}")

    print(
        json.dumps(
            {
                "status": "ok",
                "package_lock": args.package_lock.relative_to(ROOT).as_posix()
                if args.package_lock.is_absolute()
                and args.package_lock.resolve().is_relative_to(ROOT)
                else args.package_lock.as_posix(),
                "packages": observed_packages,
                "skipped_environment_lock_packages": sorted(
                    set(locked_packages) - set(expected_packages)
                ),
                "torch_cuda_version": observed_cuda,
                "cuda_available": cuda_available,
                "cuda_probe": cuda_probe,
                "nvidia_smi": nvidia_smi_summary(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"gpu training environment check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
