#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.models import ResNetCIFAR, weight_parameter_names


BYTES_PER_GIB = 1024**3


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--input-shape must have form C,H,W")
    return tuple(parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--blocks-per-stage", type=int, default=3)
    parser.add_argument("--input-shape", default="3,32,32")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs" / "cifar10_resnet20_full_covariance_feasibility.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "cifar10_resnet20_full_covariance_feasibility.md",
    )
    return parser.parse_args()


def matrix_gib(n: int, *, dtype_bytes: int = 8) -> float:
    return float(n * n * dtype_bytes / BYTES_PER_GIB)


def packed_symmetric_gib(n: int, *, dtype_bytes: int = 8) -> float:
    return float((n * (n + 1) // 2) * dtype_bytes / BYTES_PER_GIB)


def cholesky_flops(n: int) -> float:
    return float(n**3 / 3.0)


def covariance_stats(n: int) -> dict[str, float | int]:
    return {
        "parameter_count": n,
        "dense_matrix_elements": n * n,
        "dense_precision_float64_gib": matrix_gib(n, dtype_bytes=8),
        "precision_plus_cholesky_float64_gib": 2.0 * matrix_gib(n, dtype_bytes=8),
        "packed_symmetric_float64_gib": packed_symmetric_gib(n, dtype_bytes=8),
        "dense_precision_float32_gib": matrix_gib(n, dtype_bytes=4),
        "cholesky_flops": cholesky_flops(n),
    }


def block_diagonal_stats(counts: list[int]) -> dict[str, float | int]:
    matrix_elements = sum(count * count for count in counts)
    packed_elements = sum(count * (count + 1) // 2 for count in counts)
    flops = sum(cholesky_flops(count) for count in counts)
    return {
        "block_count": len(counts),
        "parameter_count": sum(counts),
        "dense_matrix_elements": matrix_elements,
        "dense_precision_float64_gib": float(matrix_elements * 8 / BYTES_PER_GIB),
        "precision_plus_cholesky_float64_gib": float(2 * matrix_elements * 8 / BYTES_PER_GIB),
        "packed_symmetric_float64_gib": float(packed_elements * 8 / BYTES_PER_GIB),
        "dense_precision_float32_gib": float(matrix_elements * 4 / BYTES_PER_GIB),
        "cholesky_flops": float(flops),
    }


def fmt_gib(value: float) -> str:
    if value >= 1000.0:
        return f"{value:,.1f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_sci(value: float) -> str:
    return f"{value:.2e}"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    model = ResNetCIFAR(
        input_shape=parse_shape(args.input_shape),
        num_classes=args.num_classes,
        blocks_per_stage=args.blocks_per_stage,
        width=args.resnet_width,
    )
    all_params = [
        {
            "name": name,
            "shape": list(param.shape),
            "parameter_count": int(param.numel()),
            "is_weight_mask_parameter": param.ndim > 1,
        }
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    weight_names = set(weight_parameter_names(model))
    weight_params = [row for row in all_params if row["name"] in weight_names]
    all_counts = [int(row["parameter_count"]) for row in all_params]
    weight_counts = [int(row["parameter_count"]) for row in weight_params]
    largest_params = sorted(
        all_params,
        key=lambda row: int(row["parameter_count"]),
        reverse=True,
    )[:12]
    for row in largest_params:
        row["dense_precision_float64_gib"] = matrix_gib(int(row["parameter_count"]))
        row["precision_plus_cholesky_float64_gib"] = 2.0 * matrix_gib(
            int(row["parameter_count"])
        )
        row["cholesky_flops"] = cholesky_flops(int(row["parameter_count"]))
    return {
        "model": "cifar10_resnet20",
        "resnet_width": args.resnet_width,
        "blocks_per_stage": args.blocks_per_stage,
        "input_shape": list(parse_shape(args.input_shape)),
        "num_classes": args.num_classes,
        "all_trainable": covariance_stats(sum(all_counts)),
        "weight_only": covariance_stats(sum(weight_counts)),
        "all_trainable_tensor_block_diagonal": block_diagonal_stats(all_counts),
        "weight_tensor_block_diagonal": block_diagonal_stats(weight_counts),
        "largest_parameter_tensors": largest_params,
        "interpretation": {
            "decision": (
                "not a runnable exact full-covariance CIFAR posterior under the "
                "single-workstation budget"
            ),
            "reason": (
                "the dense all-trainable covariance needs hundreds of GiB before "
                "workspace and an O(N^3) Cholesky; even tensor-block exact "
                "covariance is dominated by 36,864-parameter convolution blocks"
            ),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = [
        ("All trainable parameters", payload["all_trainable"]),
        ("Weight tensors only", payload["weight_only"]),
        (
            "All trainable tensor-block diagonal",
            payload["all_trainable_tensor_block_diagonal"],
        ),
        ("Weight tensor-block diagonal", payload["weight_tensor_block_diagonal"]),
    ]
    lines = [
        "# CIFAR ResNet-20 Full-Covariance Feasibility Audit",
        "",
        "This audit quantifies the exact dense covariance cost for the CIFAR-10",
        "ResNet-20 model used in the paper. It is not a runnable exact",
        "full-covariance CIFAR posterior experiment; it is a resource bound that",
        "explains why the paper uses full-network low-rank-plus-diagonal and",
        "selected-block full-covariance posterior checks instead.",
        "",
        "## Dense Covariance Costs",
        "",
        "| Scope | Params | Matrix GiB | Matrix+Chol GiB | Packed Sym. GiB | Cholesky flops |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows:
        lines.append(
            "| "
            f"{label} | "
            f"{int(row['parameter_count']):,} | "
            f"{fmt_gib(float(row['dense_precision_float64_gib']))} | "
            f"{fmt_gib(float(row['precision_plus_cholesky_float64_gib']))} | "
            f"{fmt_gib(float(row['packed_symmetric_float64_gib']))} | "
            f"{fmt_sci(float(row['cholesky_flops']))} |"
        )
    lines.extend(
        [
            "",
            "## Largest Parameter Tensors",
            "",
            "| Tensor | Shape | Params | Matrix GiB | Matrix+Chol GiB | Cholesky flops |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["largest_parameter_tensors"]:
        shape = "x".join(str(part) for part in row["shape"])
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"`{shape}` | "
            f"{int(row['parameter_count']):,} | "
            f"{fmt_gib(float(row['dense_precision_float64_gib']))} | "
            f"{fmt_gib(float(row['precision_plus_cholesky_float64_gib']))} | "
            f"{fmt_sci(float(row['cholesky_flops']))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Exact full-network full covariance over all trainable parameters would "
            f"require {fmt_gib(float(payload['all_trainable']['dense_precision_float64_gib']))} "
            "GiB for a single dense float64 precision/covariance matrix and "
            f"{fmt_gib(float(payload['all_trainable']['precision_plus_cholesky_float64_gib']))} "
            "GiB if the matrix and its Cholesky factor are both resident. The "
            "nominal Cholesky cost is "
            f"{fmt_sci(float(payload['all_trainable']['cholesky_flops']))} flops.",
            "",
            "Even an exact tensor-block-diagonal approximation across the weight",
            "tensors is dominated by the 36,864-parameter stage-3 convolution",
            "blocks, each requiring about 10 GiB for one dense float64 matrix and",
            "about 20 GiB with a Cholesky factor. This makes exhaustive five-seed",
            "exact tensor-block covariance a poor fit for the single-workstation",
            "budget.",
            "",
            "The existing posterior evidence therefore uses the tractable parts of",
            "this spectrum: exact full-covariance final-head and selected-block",
            "checks, joint selected-block covariance, exact 22k- and 68k-parameter",
            "tensor-block-diagonal covariance rows, exact 68k-, 86k-, and",
            "270,896-parameter streamed joint-group covariance rows over all",
            "weight tensors, full-network SWAG, full-network rank-16/rank-32/rank-64/rank-128 Hessian-plus-diagonal Laplace,",
            "and low-dimensional full-network HMC subspaces.",
            "",
            "This file is generated by `scripts/audit_full_covariance_feasibility.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "all_trainable_params": payload["all_trainable"]["parameter_count"],
                "all_trainable_matrix_gib": payload["all_trainable"][
                    "dense_precision_float64_gib"
                ],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
