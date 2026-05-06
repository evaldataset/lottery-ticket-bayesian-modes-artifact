#!/usr/bin/env python
from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "docs" / "environment_lock.json"


def fail(message: str) -> None:
    raise AssertionError(message)


def load_lock() -> dict[str, Any]:
    with LOCK_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def command_head(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        fail(f"required executable is missing: {command}")
    proc = subprocess.run(
        [path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout.splitlines()[0].strip()


def check_python(lock: dict[str, Any]) -> None:
    expected = lock["python"]
    current = sys.version_info
    for field in ("major", "minor", "micro"):
        if getattr(current, field) != int(expected[field]):
            fail(
                "Python version mismatch: "
                f"expected {expected['major']}.{expected['minor']}.{expected['micro']}, "
                f"got {current.major}.{current.minor}.{current.micro}"
            )
    expected_platform = lock.get("platform")
    if expected_platform and platform.platform() != expected_platform:
        fail(
            "platform mismatch: "
            f"expected {expected_platform}, got {platform.platform()}"
        )


def check_packages(lock: dict[str, Any]) -> None:
    for name, expected in sorted(lock["packages"].items()):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise AssertionError(f"locked package is missing: {name}") from exc
        if actual != expected:
            fail(f"package mismatch for {name}: expected {expected}, got {actual}")


def check_torch(lock: dict[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise AssertionError("torch import failed") from exc
    expected = lock["torch"]
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available != bool(expected["cuda_available"]):
        fail(
            "torch CUDA availability mismatch: "
            f"expected {expected['cuda_available']}, got {cuda_available}"
        )
    cuda_version = str(torch.version.cuda)
    if cuda_version != str(expected["cuda_version"]):
        fail(
            "torch CUDA version mismatch: "
            f"expected {expected['cuda_version']}, got {cuda_version}"
        )


def check_executables(lock: dict[str, Any]) -> None:
    executables = lock.get("executables", {})
    pdflatex_prefix = executables.get("pdflatex_prefix")
    if pdflatex_prefix:
        actual = command_head("pdflatex")
        if not actual.startswith(pdflatex_prefix):
            fail(f"pdflatex mismatch: expected prefix {pdflatex_prefix}, got {actual}")
    bibtex_prefix = executables.get("bibtex_prefix")
    if bibtex_prefix:
        actual = command_head("bibtex")
        if not actual.startswith(bibtex_prefix):
            fail(f"bibtex mismatch: expected prefix {bibtex_prefix}, got {actual}")


def main() -> None:
    lock = load_lock()
    check_python(lock)
    check_packages(lock)
    check_torch(lock)
    check_executables(lock)
    print("environment lock matches current runtime")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"environment lock check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
