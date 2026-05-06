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
    "id_accuracy",
    "id_nll",
    "id_brier",
    "id_ece",
    "id_confidence",
    "ood_msp_auroc",
    "ood_msp_aupr",
    "ood_msp_fpr95",
    "ood_entropy_auroc",
    "ood_entropy_aupr",
    "ood_entropy_fpr95",
    "ood_id_msp",
    "ood_ood_msp",
    "ood_id_entropy",
    "ood_ood_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.glob("*/metrics.json")):
        payload = read_json(path)
        for source, metrics in payload["sources"].items():
            row = {
                "seed": int(payload["seed"]),
                "source": source,
                "dataset": payload["dataset"],
                "ood_dataset": payload["ood_dataset"],
            }
            for key, value in metrics["id"].items():
                row[f"id_{key}"] = float(value)
            for key, value in metrics["ood"].items():
                row[f"ood_{key}"] = float(value)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no calibration/OOD metrics found under {run_root}")
    return rows


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    preferred_order = ["dense", "imp", "swag_member_mean", "swag_ensemble"]
    sources = [source for source in preferred_order if source in by_source]
    sources.extend(sorted(source for source in by_source if source not in sources))

    summary_rows = []
    for source in sources:
        group = by_source[source]
        out: dict[str, Any] = {
            "source": source,
            "num_runs": len({row["seed"] for row in group}),
            "dataset": group[0]["dataset"],
            "ood_dataset": group[0]["ood_dataset"],
        }
        for field in FIELDS:
            values = [float(row.get(field, math.nan)) for row in group]
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
        summary_rows.append(out)

    dense = next((row for row in summary_rows if row["source"] == "dense"), None)
    if dense is not None:
        for row in summary_rows:
            row["delta_id_nll_vs_dense"] = row["id_nll"] - dense["id_nll"]
            row["delta_id_ece_vs_dense"] = row["id_ece"] - dense["id_ece"]
            row["delta_ood_msp_auroc_vs_dense"] = (
                row["ood_msp_auroc"] - dense["ood_msp_auroc"]
            )
            row["delta_ood_entropy_auroc_vs_dense"] = (
                row["ood_entropy_auroc"] - dense["ood_entropy_auroc"]
            )
    return summary_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "Source",
        "Runs",
        "ID Acc",
        "ID NLL",
        "ID ECE",
        "ID Brier",
        "MSP AUROC",
        "MSP FPR95",
        "Entropy AUROC",
        "Entropy FPR95",
        "dNLL",
        "dECE",
        "dMSP AUROC",
    ]
    keys = [
        "source",
        "num_runs",
        "id_accuracy",
        "id_nll",
        "id_ece",
        "id_brier",
        "ood_msp_auroc",
        "ood_msp_fpr95",
        "ood_entropy_auroc",
        "ood_entropy_fpr95",
        "delta_id_nll_vs_dense",
        "delta_id_ece_vs_dense",
        "delta_ood_msp_auroc_vs_dense",
    ]
    lines = [
        "# Calibration/OOD Probe Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "ID metrics are computed on the in-distribution test set. OOD metrics",
            "treat higher maximum-softmax probability or lower entropy as the",
            "in-distribution score. Lower ID NLL/ECE and higher OOD AUROC are",
            "better. Delta columns are relative to the dense model mean.",
            "",
            "This file is generated by `scripts/summarize_calibration_ood_probe.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = summarize(load_rows(args.run_root))
    fieldnames = [
        "source",
        "num_runs",
        "dataset",
        "ood_dataset",
    ]
    for field in FIELDS:
        fieldnames.append(field)
        fieldnames.append(f"{field}_std")
    fieldnames.extend(
        [
            "delta_id_nll_vs_dense",
            "delta_id_ece_vs_dense",
            "delta_ood_msp_auroc_vs_dense",
            "delta_ood_entropy_auroc_vs_dense",
        ]
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
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
