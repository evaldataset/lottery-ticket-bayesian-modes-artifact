#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CUDA_LIB = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
NVML_LIB = Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def write_completed(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def nvidia_devices() -> list[Path]:
    return sorted(path for path in Path("/dev").glob("nvidia*") if path.exists())


def can_use_manual_driver_mount(stderr: str) -> bool:
    if "could not select device driver" not in stderr and "capabilities: [[gpu]]" not in stderr:
        return False
    if not nvidia_devices():
        return False
    return CUDA_LIB.exists() and NVML_LIB.exists()


def manual_driver_mount_command(image: str) -> list[str]:
    command = ["docker", "run", "--rm"]
    for device in nvidia_devices():
        command.extend(["--device", device.as_posix()])
    command.extend(
        [
            "-v",
            f"{CUDA_LIB.resolve()}:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro",
            "-v",
            f"{NVML_LIB.resolve()}:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro",
            image,
        ]
    )
    return command


def main() -> None:
    args = parse_args()
    primary = ["docker", "run", "--rm", "--gpus", "all", args.image]
    result = run_capture(primary)
    if result.returncode == 0:
        write_completed(result)
        return

    if not can_use_manual_driver_mount(result.stderr):
        write_completed(result)
        raise SystemExit(result.returncode)

    print(
        "NVIDIA Container Toolkit runtime was not available; "
        "falling back to explicit /dev/nvidia* and driver-library mounts.",
        file=sys.stderr,
    )
    fallback_env = os.environ.copy()
    fallback = subprocess.run(manual_driver_mount_command(args.image), check=False, env=fallback_env)
    raise SystemExit(fallback.returncode)


if __name__ == "__main__":
    main()
