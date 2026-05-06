#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "seed",
    "sample_accuracy_mean",
    "posterior_jaccard_mean",
    "random_jaccard_mean",
    "chain_start_magnitude_to_imp_jaccard_mean",
    "posterior_to_chain_start_magnitude_jaccard_mean",
    "dense_magnitude_to_imp_jaccard",
    "state_num_clusters",
    "function_num_clusters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="runs/mnist_sgld_rescue_*/*/metrics.json")
    parser.add_argument("--out-csv", type=Path, default=Path("runs/mnist_sgld_rescue_summary.csv"))
    return parser.parse_args()


def load_row(path: Path) -> dict[str, float | str]:
    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "run": path.parts[1],
        "seed": float(metrics["seed"]),
        "sample_accuracy_mean": float(metrics["sgld"]["sample_accuracy_mean"]),
        "posterior_jaccard_mean": float(
            metrics["posterior_mask_overlap"]["posterior_jaccard_mean"]
        ),
        "random_jaccard_mean": float(metrics["posterior_mask_overlap"]["random_jaccard_mean"]),
        "chain_start_magnitude_to_imp_jaccard_mean": float(
            metrics["controls"]["chain_start_magnitude_to_imp_jaccard_mean"]
        ),
        "posterior_to_chain_start_magnitude_jaccard_mean": float(
            metrics["controls"]["posterior_to_chain_start_magnitude_jaccard_mean"]
        ),
        "dense_magnitude_to_imp_jaccard": float(
            metrics["controls"]["dense_magnitude_to_imp_jaccard"]
        ),
        "state_num_clusters": float(metrics["state_clustering"]["num_clusters"]),
        "function_num_clusters": float(metrics["function_clustering"]["num_clusters"]),
    }


def main() -> None:
    args = parse_args()
    rows = [load_row(path) for path in sorted(Path().glob(args.pattern))]
    if not rows:
        raise SystemExit(f"No metrics found for pattern: {args.pattern}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()

