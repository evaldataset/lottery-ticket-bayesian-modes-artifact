#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = [
    "seed",
    "dense_accuracy",
    "imp_accuracy",
    "imp_sparsity",
    "posterior_jaccard_mean",
    "random_jaccard_mean",
    "posterior_minus_random_jaccard",
    "jaccard_cohens_d",
    "mannwhitney_greater_pvalue",
    "dense_magnitude_to_imp_jaccard",
    "initial_magnitude_to_imp_jaccard",
    "chain_start_magnitude_to_imp_jaccard_mean",
    "posterior_to_chain_start_magnitude_jaccard_mean",
    "state_num_clusters",
    "function_num_clusters",
    "hmc_accept_rate",
    "hmc_sample_accuracy_mean",
    "hmc_sample_to_dense_prediction_agreement_mean",
    "hmc_sample_to_imp_prediction_agreement_mean",
    "hmc_step_size",
    "hmc_leapfrog_steps",
    "hmc_energy_first",
    "hmc_energy_last",
]


def optional_float(mapping: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = mapping.get(key)
    return default if value is None else float(value)


def row_from_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    overlap = metrics["hmc_mask_overlap"]
    controls = metrics["controls"]
    hmc = metrics["hmc"]
    state = metrics.get("state_clustering", {})
    function = metrics.get("function_clustering", {})
    dense_mag = float(controls["dense_magnitude_to_imp_jaccard"])
    return {
        "seed": float(metrics["seed"]),
        "dense_accuracy": float(metrics["dense"]["accuracy"]),
        "imp_accuracy": float(metrics["imp"]["accuracy"]),
        "imp_sparsity": float(metrics["imp"]["sparsity"]),
        "posterior_jaccard_mean": float(overlap["posterior_jaccard_mean"]),
        "random_jaccard_mean": float(overlap["random_jaccard_mean"]),
        "posterior_minus_random_jaccard": float(overlap["posterior_minus_random_jaccard"]),
        "jaccard_cohens_d": float(overlap["jaccard_cohens_d"]),
        "mannwhitney_greater_pvalue": float(overlap["mannwhitney_greater_pvalue"]),
        "dense_magnitude_to_imp_jaccard": dense_mag,
        "initial_magnitude_to_imp_jaccard": float(controls["initial_magnitude_to_imp_jaccard"]),
        "chain_start_magnitude_to_imp_jaccard_mean": optional_float(
            controls,
            "chain_start_magnitude_to_imp_jaccard_mean",
            default=dense_mag,
        ),
        "posterior_to_chain_start_magnitude_jaccard_mean": float(
            controls["hmc_to_dense_magnitude_jaccard_mean"]
        ),
        "state_num_clusters": optional_float(state, "num_clusters"),
        "function_num_clusters": optional_float(function, "num_clusters"),
        "hmc_accept_rate": float(hmc["accept_rate"]),
        "hmc_sample_accuracy_mean": optional_float(hmc, "sample_accuracy_mean"),
        "hmc_sample_to_dense_prediction_agreement_mean": optional_float(
            hmc,
            "sample_to_dense_prediction_agreement_mean",
        ),
        "hmc_sample_to_imp_prediction_agreement_mean": optional_float(
            hmc,
            "sample_to_imp_prediction_agreement_mean",
        ),
        "hmc_step_size": float(hmc["step_size"]),
        "hmc_leapfrog_steps": float(hmc["leapfrog_steps"]),
        "hmc_energy_first": optional_float(hmc, "energy_first"),
        "hmc_energy_last": optional_float(hmc, "energy_last"),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        values = np.array([row[field] for row in rows], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            out[field] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
            continue
        out[field] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("*/metrics.json"))
    if not paths:
        raise SystemExit(f"no metrics.json files under {args.run_root}")
    rows = [row_from_metrics(path) for path in paths]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "num_runs": len(rows),
        "runs": rows,
        "summary": summarize(rows),
    }
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
