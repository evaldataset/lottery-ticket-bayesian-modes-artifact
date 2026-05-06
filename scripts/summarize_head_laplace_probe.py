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
from scipy.stats import t


FIELDS = [
    "head_laplace_scale",
    "num_runs",
    "dense_accuracy",
    "imp_accuracy",
    "imp_sparsity_global",
    "imp_sparsity_head",
    "head_posterior_jaccard_mean",
    "head_random_jaccard_mean",
    "head_chain_start_magnitude_to_imp_jaccard",
    "head_posterior_minus_chain_start_jaccard",
    "head_posterior_to_chain_start_magnitude_jaccard_mean",
    "head_initial_magnitude_to_imp_jaccard",
    "head_rewind_magnitude_to_imp_jaccard",
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
        raise SystemExit(f"no head Laplace metrics found under {run_root}")
    return rows


def valid_values(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def mean(values: list[float]) -> float:
    valid = valid_values(values)
    return float(np.mean(valid)) if valid else float("nan")


def std(values: list[float]) -> float:
    valid = valid_values(values)
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    valid = valid_values(values)
    if not valid:
        return float("nan"), float("nan")
    center = float(np.mean(valid))
    if len(valid) < 2:
        return center, center
    half_width = float(t.ppf(0.975, len(valid) - 1) * np.std(valid, ddof=1) / len(valid) ** 0.5)
    return center - half_width, center + half_width


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_scale: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_scale[row["head_laplace_scale"]].append(row)

    summary_rows = []
    for scale in sorted(by_scale):
        group = by_scale[scale]
        out: dict[str, float] = {
            "head_laplace_scale": scale,
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"head_laplace_scale", "num_runs"}:
                continue
            values = [row.get(field, float("nan")) for row in group]
            low, high = ci95(values)
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
            out[f"{field}_ci95_low"] = low
            out[f"{field}_ci95_high"] = high
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
        "Scale",
        "Runs",
        "Dense Acc",
        "IMP Acc",
        "Head Sparsity",
        "Head Posterior",
        "Head Random",
        "Head Chain",
        "Head Post-Chain Gap",
        "Head Post-Chain",
        "Head Initial",
        "Head Rewind",
        "Global Posterior",
        "Global Chain",
        "Global Post-Chain",
        "Sample Acc",
    ]
    keys = [
        "head_laplace_scale",
        "num_runs",
        "dense_accuracy",
        "imp_accuracy",
        "imp_sparsity_head",
        "head_posterior_jaccard_mean",
        "head_random_jaccard_mean",
        "head_chain_start_magnitude_to_imp_jaccard",
        "head_posterior_minus_chain_start_jaccard",
        "head_posterior_to_chain_start_magnitude_jaccard_mean",
        "head_initial_magnitude_to_imp_jaccard",
        "head_rewind_magnitude_to_imp_jaccard",
        "global_posterior_jaccard_mean",
        "global_chain_start_magnitude_to_imp_jaccard",
        "global_posterior_to_chain_start_magnitude_jaccard_mean",
        "sample_accuracy_mean",
    ]
    lines = [
        "# Head Laplace Probe Summary",
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
            "This is an exact full-covariance Laplace probe for the final linear",
            "classification head only. A posterior-mode rescue would require head",
            "posterior support to exceed the head chain-start magnitude control",
            "while support moves away from that control. If the head post-chain gap",
            "is non-positive as Head Post-Chain decreases, head posterior movement",
            "is not ticket-directed.",
            "",
            "Global rows are auxiliary because only the final head is sampled.",
            "",
            "This file is generated by `scripts/summarize_head_laplace_probe.py`.",
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
        if field not in {"head_laplace_scale", "num_runs"}:
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
    write_markdown(rows, args.out_md)
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
