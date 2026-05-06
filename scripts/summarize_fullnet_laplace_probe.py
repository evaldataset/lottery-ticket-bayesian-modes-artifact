#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "full_laplace_scale",
    "num_runs",
    "dense_accuracy",
    "imp_accuracy",
    "imp_sparsity",
    "parameter_count",
    "weight_parameter_count",
    "examples_seen",
    "posterior_jaccard_mean",
    "chain_start_magnitude_to_imp_jaccard",
    "posterior_minus_chain_start_jaccard",
    "posterior_to_chain_start_magnitude_jaccard_mean",
    "initial_magnitude_to_imp_jaccard",
    "rewind_magnitude_to_imp_jaccard",
    "sample_accuracy_mean",
    "sample_to_dense_prediction_agreement_mean",
    "sample_to_imp_prediction_agreement_mean",
]

T975_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_payloads(run_root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open(encoding="utf-8") as f:
            payloads.append(json.load(f))
    if not payloads:
        raise SystemExit(f"no full-network Laplace metrics found under {run_root}")
    return payloads


def load_rows(payloads: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for payload in payloads:
        seed = float(payload["seed"])
        for row in payload["rows"]:
            out = {"seed": seed}
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    out[key] = float(value)
            rows.append(out)
    return rows


def metadata(payloads: list[dict[str, Any]]) -> dict[str, str]:
    configs = [
        payload.get("config", {})
        for payload in payloads
        if isinstance(payload.get("config", {}), dict)
    ]
    datasets = sorted({str(config.get("dataset", "digits")) for config in configs})
    models = sorted({str(config.get("model", "mlp")) for config in configs})
    if len(datasets) == 1 and len(models) == 1:
        dataset = datasets[0]
        model = models[0]
        if dataset == "digits" and model == "mlp":
            subject = "tiny digits MLP"
            scope = "a small-model sanity check for the dense full-covariance code path, not CIFAR-scale evidence"
        elif dataset == "fake-cifar10" and model == "resnet20":
            widths = sorted({str(config.get("resnet_width", "unknown")) for config in configs})
            width = widths[0] if len(widths) == 1 else "mixed"
            subject = f"fake-CIFAR ResNet-20 width-{width}"
            scope = (
                "a convolutional/residual/BatchNorm code-path smoke test for exact "
                "dense full-network covariance, not real CIFAR evidence"
            )
        else:
            subject = f"{dataset} {model}"
            scope = "a small-model sanity check for the exact dense full-covariance code path"
    else:
        subject = "mixed full-network probe"
        scope = "a mixed-run sanity check for the exact dense full-covariance code path"
    return {"subject": subject, "scope": scope}


def valid(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    values = valid(values)
    return float(np.mean(values)) if values else float("nan")


def std(values: list[float]) -> float:
    values = valid(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    values = valid(values)
    if not values:
        return float("nan"), float("nan")
    center = float(np.mean(values))
    if len(values) < 2:
        return center, center
    df = len(values) - 1
    t_critical = T975_CRITICAL.get(df, 1.96)
    half_width = float(t_critical * np.std(values, ddof=1) / len(values) ** 0.5)
    return center - half_width, center + half_width


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_scale: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_scale[row["full_laplace_scale"]].append(row)
    out_rows = []
    for scale in sorted(by_scale):
        group = by_scale[scale]
        out: dict[str, float] = {
            "full_laplace_scale": scale,
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"full_laplace_scale", "num_runs"}:
                continue
            values = [row.get(field, float("nan")) for row in group]
            low, high = ci95(values)
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
            out[f"{field}_ci95_low"] = low
            out[f"{field}_ci95_high"] = high
        out_rows.append(out)
    return out_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if abs(value) < 1e-4 and value != 0.0:
            return f"{value:.1e}"
        if abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(
    rows: list[dict[str, float]],
    path: Path,
    metadata: dict[str, str],
) -> None:
    headers = [
        "Scale",
        "Runs",
        "Params",
        "Dense Acc",
        "IMP Acc",
        "Posterior",
        "Chain",
        "Post-Chain Gap",
        "Post-Chain",
        "Initial",
        "Sample Acc",
    ]
    keys = [
        "full_laplace_scale",
        "num_runs",
        "parameter_count",
        "dense_accuracy",
        "imp_accuracy",
        "posterior_jaccard_mean",
        "chain_start_magnitude_to_imp_jaccard",
        "posterior_minus_chain_start_jaccard",
        "posterior_to_chain_start_magnitude_jaccard_mean",
        "initial_magnitude_to_imp_jaccard",
        "sample_accuracy_mean",
    ]
    lines = [
        "# Full-network Dense Laplace Probe Summary",
        "",
        "This is an exact dense full-network softmax-GGN/Laplace probe for",
        f"{metadata['subject']}. It is {metadata['scope']}.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in keys) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "The posterior row samples one dense Gaussian over every trainable",
            "parameter of the model. A support-equivalence rescue would require",
            "posterior support to beat the dense chain-start magnitude support",
            "against IMP while samples remain accurate. If the post-chain gap is",
            "non-positive or tiny despite exact dense covariance, the small-model",
            "dense-covariance sanity check is also negative.",
            "",
            "This file is generated by `scripts/summarize_fullnet_laplace_probe.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payloads = load_payloads(args.run_root)
    rows = summarize(load_rows(payloads))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for field in FIELDS:
        fieldnames.append(field)
        if field not in {"full_laplace_scale", "num_runs"}:
            fieldnames.extend(
                [
                    f"{field}_std",
                    f"{field}_ci95_low",
                    f"{field}_ci95_high",
                ]
            )
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out_md, metadata(payloads))
    print(json.dumps({"rows": len(rows), "out_csv": str(args.out_csv), "out_md": str(args.out_md)}))


if __name__ == "__main__":
    main()
