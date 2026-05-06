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
    "subspace_dim",
    "hmc_step_size",
    "num_runs",
    "dense_accuracy",
    "imp_accuracy",
    "posterior_jaccard_mean",
    "random_jaccard_mean",
    "chain_start_magnitude_to_imp_jaccard",
    "posterior_minus_chain_start_jaccard",
    "posterior_to_chain_start_magnitude_jaccard_mean",
    "dense_magnitude_to_imp_jaccard",
    "initial_magnitude_to_imp_jaccard",
    "rewind_magnitude_to_imp_jaccard",
    "snip_to_imp_jaccard",
    "synflow_to_imp_jaccard",
    "posterior_map_mean_to_imp_jaccard",
    "posterior_map_rms_to_imp_jaccard",
    "sample_accuracy_mean",
    "sample_to_dense_prediction_agreement_mean",
    "sample_to_imp_prediction_agreement_mean",
    "hmc_accept_rate",
    "hmc_coordinate_norm_mean",
    "hmc_parameter_distance_mean",
    "state_num_clusters",
    "function_num_clusters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, float]]:
    rows = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        seed = float(payload["seed"])
        for row in payload["rows"]:
            out = {"seed": seed}
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    out[key] = float(value)
            rows.append(out)
    if not rows:
        raise SystemExit(f"no subspace HMC metrics found under {run_root}")
    return rows


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_key: dict[tuple[int, float], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_key[(int(row["subspace_dim"]), row["hmc_step_size"])].append(row)

    summary_rows = []
    for (subspace_dim, step_size), group in sorted(by_key.items()):
        out: dict[str, float] = {
            "subspace_dim": float(subspace_dim),
            "hmc_step_size": step_size,
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"subspace_dim", "hmc_step_size", "num_runs"}:
                continue
            values = [row.get(field, float("nan")) for row in group]
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
        summary_rows.append(out)
    return summary_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, float]], path: Path) -> None:
    headers = [
        "Dim",
        "Step",
        "Runs",
        "Dense Acc",
        "IMP Acc",
        "Posterior",
        "Random",
        "Chain Start",
        "Post-Chain Start",
        "Post-Chain",
        "Rewind Mag",
        "Sample Acc",
        "Accept",
        "Coord Norm",
        "Param Dist",
        "State Clusters",
        "Function Clusters",
    ]
    keys = [
        "subspace_dim",
        "hmc_step_size",
        "num_runs",
        "dense_accuracy",
        "imp_accuracy",
        "posterior_jaccard_mean",
        "random_jaccard_mean",
        "chain_start_magnitude_to_imp_jaccard",
        "posterior_minus_chain_start_jaccard",
        "posterior_to_chain_start_magnitude_jaccard_mean",
        "rewind_magnitude_to_imp_jaccard",
        "sample_accuracy_mean",
        "hmc_accept_rate",
        "hmc_coordinate_norm_mean",
        "hmc_parameter_distance_mean",
        "state_num_clusters",
        "function_num_clusters",
    ]
    lines = [
        "# Subspace HMC Probe Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in keys) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "A posterior-mode rescue would require Post-Chain to fall while",
            "Posterior rises above Chain Start. If full-network subspace HMC",
            "moves in parameter space but Posterior stays at or below Chain Start,",
            "the sampled subspace posterior is not ticket-directed.",
            "",
            "This file is generated by `scripts/summarize_subspace_hmc_probe.py`.",
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
        if field not in {"subspace_dim", "hmc_step_size", "num_runs"}:
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
