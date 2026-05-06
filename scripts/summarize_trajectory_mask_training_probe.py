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
    "source_kind",
    "source",
    "num_runs",
    "trained_accuracy",
    "trained_ece",
    "trained_brier",
    "accuracy_minus_imp",
    "accuracy_minus_dense",
    "source_to_imp_jaccard",
    "source_to_dense_final_magnitude_jaccard",
    "source_to_rewind_magnitude_jaccard",
    "mask_sparsity",
    "dense_accuracy",
    "imp_accuracy",
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
                if value is None:
                    out[key] = float("nan")
                elif isinstance(value, (int, float)):
                    out[key] = float(value)
                else:
                    out[key] = str(value)
            rows.append(out)
    if not rows:
        raise SystemExit(f"no trajectory mask training probe metrics found under {run_root}")
    return rows


def valid(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


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
    half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / len(values) ** 0.5)
    return center - half, center + half


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_kind"]), str(row["source"]))].append(row)

    summary: list[dict[str, float | str]] = []
    for (source_kind, source), group in sorted(grouped.items()):
        out: dict[str, float | str] = {
            "source_kind": source_kind,
            "source": source,
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"source_kind", "source", "num_runs"}:
                continue
            values = [
                float(row.get(field, float("nan")))
                for row in group
                if not isinstance(row.get(field, float("nan")), str)
            ]
            low, high = ci95(values)
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
            out[f"{field}_ci95_low"] = low
            out[f"{field}_ci95_high"] = high
        summary.append(out)
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if value.is_integer() and abs(value) < 10000:
            return str(int(value))
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


def source_label(value: Any) -> str:
    text = str(value)
    if text == "gem_miner":
        return "Gem-Miner"
    if text == "variational_prune":
        return "Variational prune"
    if text == "hard_concrete":
        return "Hard concrete"
    if text.startswith("epoch_"):
        return text.replace("epoch_", "Epoch ")
    return text.replace("traj_", "").replace("_", " ")


def write_markdown(rows: list[dict[str, float | str]], path: Path) -> None:
    headers = [
        "Source",
        "Kind",
        "Runs",
        "Acc.",
        "ECE",
        "Brier",
        "Acc-IMP",
        "Acc-Dense",
        "Jaccard-IMP",
        "Jaccard-Dense",
        "Jaccard-Rewind",
    ]
    keys = [
        "source",
        "source_kind",
        "num_runs",
        "trained_accuracy",
        "trained_ece",
        "trained_brier",
        "accuracy_minus_imp",
        "accuracy_minus_dense",
        "source_to_imp_jaccard",
        "source_to_dense_final_magnitude_jaccard",
        "source_to_rewind_magnitude_jaccard",
    ]
    lines = [
        "# Trajectory Mask Training Probe Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |",
    ]
    for row in rows:
        values = [source_label(row["source"]), str(row["source_kind"])] + [
            fmt(row[key]) for key in keys[2:]
        ]
        lines.append("| " + " | ".join(values) + " |")

    best = max(rows, key=lambda row: float(row["trained_accuracy"]))
    best_non_imp = max(
        [row for row in rows if row["source_kind"] != "imp"],
        key=lambda row: float(row["trained_accuracy"]),
    )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            f"The best non-IMP mask is {source_label(best_non_imp['source'])} "
            f"with accuracy {fmt(best_non_imp['trained_accuracy'])} and "
            f"Acc-IMP {fmt(best_non_imp['accuracy_minus_imp'])}.",
            f"The overall best row is {source_label(best['source'])} "
            f"with accuracy {fmt(best['trained_accuracy'])}.",
            "",
            "This file is generated by `scripts/summarize_trajectory_mask_training_probe.py`.",
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
        if field not in {"source_kind", "source", "num_runs"}:
            fieldnames.extend([f"{field}_std", f"{field}_ci95_low", f"{field}_ci95_high"])
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
