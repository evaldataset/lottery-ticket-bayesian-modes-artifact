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
    "block_name",
    "block_laplace_scale",
    "num_runs",
    "block_parameter_count",
    "block_examples_seen",
    "dense_accuracy",
    "imp_accuracy",
    "block_posterior_jaccard_mean",
    "block_random_jaccard_mean",
    "block_chain_start_magnitude_to_imp_jaccard",
    "block_posterior_minus_chain_start_jaccard",
    "block_posterior_to_chain_start_magnitude_jaccard_mean",
    "block_initial_magnitude_to_imp_jaccard",
    "block_rewind_magnitude_to_imp_jaccard",
    "global_posterior_jaccard_mean",
    "global_random_jaccard_mean",
    "global_chain_start_magnitude_to_imp_jaccard",
    "global_posterior_minus_chain_start_jaccard",
    "global_posterior_to_chain_start_magnitude_jaccard_mean",
    "global_initial_magnitude_to_imp_jaccard",
    "global_rewind_magnitude_to_imp_jaccard",
    "sample_accuracy_mean",
    "sample_to_dense_prediction_agreement_mean",
    "sample_to_imp_prediction_agreement_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, float | str]]:
    rows = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        seed = float(payload["seed"])
        for row in payload["rows"]:
            out: dict[str, float | str] = {"seed": seed}
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    out[key] = float(value)
                elif value is None:
                    out[key] = float("nan")
                else:
                    out[key] = str(value)
            rows.append(out)
    if not rows:
        raise SystemExit(f"no block Laplace metrics found under {run_root}")
    return rows


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, float], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["block_name"]), float(row["block_laplace_scale"]))].append(row)

    summary: list[dict[str, float | str]] = []
    for (block_name, scale), group in sorted(grouped.items()):
        out: dict[str, float | str] = {
            "block_name": block_name,
            "block_laplace_scale": scale,
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"block_name", "block_laplace_scale", "num_runs"}:
                continue
            values = [
                float(row.get(field, float("nan")))
                for row in group
                if not isinstance(row.get(field, float("nan")), str)
            ]
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
        summary.append(out)
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, float | str]], path: Path) -> None:
    headers = [
        "Block",
        "Scale",
        "Runs",
        "Params",
        "Dense Acc",
        "IMP Acc",
        "Block Post.",
        "Block Chain",
        "Block Post-Chain",
        "Global Post.",
        "Global Chain",
        "Sample Acc.",
    ]
    keys = [
        "block_name",
        "block_laplace_scale",
        "num_runs",
        "block_parameter_count",
        "dense_accuracy",
        "imp_accuracy",
        "block_posterior_jaccard_mean",
        "block_chain_start_magnitude_to_imp_jaccard",
        "block_posterior_to_chain_start_magnitude_jaccard_mean",
        "global_posterior_jaccard_mean",
        "global_chain_start_magnitude_to_imp_jaccard",
        "sample_accuracy_mean",
    ]
    lines = [
        "# Block Laplace Probe Summary",
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
            "This is a full-covariance Laplace/softmax-GGN probe for selected",
            "weight tensors, joint tensor groups, or independent tensor-block",
            "diagonal subsets while the rest of the network is frozen. Single",
            "block rows use exact covariance for one tensor; joint rows include",
            "cross-tensor covariance inside the selected group; jointdiag rows",
            "estimate exact covariance inside several tensor groups and sample",
            "the groups independently; blockdiag rows estimate exact per-tensor",
            "covariance factors and sample those tensors independently in one",
            "combined network sample. These rows",
            "are stronger than diagonal or Kronecker-factor covariance for the",
            "covered parameters, but they are not dense full-network",
            "full-covariance posteriors.",
            "",
            "This file is generated by `scripts/summarize_block_laplace_probe.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = summarize(load_rows(args.run_root))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for field in FIELDS:
        fieldnames.append(field)
        if field not in {"block_name", "block_laplace_scale", "num_runs"}:
            fieldnames.append(f"{field}_std")
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out_md)
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
