#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
    "rewind_magnitude_to_imp_jaccard",
    "snip_to_imp_jaccard",
    "synflow_to_imp_jaccard",
    "posterior_to_dense_magnitude_jaccard_mean",
    "posterior_to_chain_start_magnitude_jaccard_mean",
    "chain_start_magnitude_to_imp_jaccard_mean",
    "state_num_clusters",
    "function_num_clusters",
    "sgld_sample_accuracy_mean",
    "sgld_sample_to_dense_prediction_agreement_mean",
    "sgld_sample_to_imp_prediction_agreement_mean",
    "sgld_chains",
    "sgld_chain_start_accuracy_mean",
    "posterior_mean_abs_jaccard",
    "posterior_rms_jaccard",
    "posterior_snr_jaccard",
    "posterior_high_variance_jaccard",
    "posterior_low_variance_jaccard",
    "posterior_mean_abs_chainwise_jaccard",
    "posterior_rms_chainwise_jaccard",
    "posterior_snr_chainwise_jaccard",
    "posterior_high_variance_chainwise_jaccard",
    "posterior_low_variance_chainwise_jaccard",
    "dense_imp_linear_barrier",
    "dense_sample_linear_barrier_mean",
    "imp_sample_linear_barrier_mean",
]


def optional_float(mapping: dict, key: str, default: float = float("nan")) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/digits_pilot"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/digits_pilot_summary.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/digits_pilot_summary.json"))
    return parser.parse_args()


def load_row(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    overlap = metrics["posterior_mask_overlap"]
    controls = metrics["controls"]
    state_clusters = metrics["state_clustering"]
    function_clusters = metrics["function_clustering"]
    sgld = metrics.get("posterior", metrics["sgld"])
    posterior_maps = metrics.get("posterior_map_jaccards", {})
    connectivity = metrics["linear_connectivity"]
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
        "dense_magnitude_to_imp_jaccard": float(controls["dense_magnitude_to_imp_jaccard"]),
        "initial_magnitude_to_imp_jaccard": float(controls["initial_magnitude_to_imp_jaccard"]),
        "rewind_magnitude_to_imp_jaccard": optional_float(
            controls, "rewind_magnitude_to_imp_jaccard"
        ),
        "snip_to_imp_jaccard": optional_float(controls, "snip_to_imp_jaccard"),
        "synflow_to_imp_jaccard": optional_float(controls, "synflow_to_imp_jaccard"),
        "posterior_to_dense_magnitude_jaccard_mean": float(
            controls["posterior_to_dense_magnitude_jaccard_mean"]
        ),
        "posterior_to_chain_start_magnitude_jaccard_mean": optional_float(
            controls, "posterior_to_chain_start_magnitude_jaccard_mean"
        ),
        "chain_start_magnitude_to_imp_jaccard_mean": optional_float(
            controls, "chain_start_magnitude_to_imp_jaccard_mean"
        ),
        "state_num_clusters": float(state_clusters["num_clusters"]),
        "function_num_clusters": float(function_clusters["num_clusters"]),
        "sgld_sample_accuracy_mean": float(sgld["sample_accuracy_mean"]),
        "sgld_sample_to_dense_prediction_agreement_mean": float(
            sgld["sample_to_dense_prediction_agreement_mean"]
        ),
        "sgld_sample_to_imp_prediction_agreement_mean": float(
            sgld["sample_to_imp_prediction_agreement_mean"]
        ),
        "sgld_chains": optional_float(sgld, "chains", default=1.0),
        "sgld_chain_start_accuracy_mean": optional_float(sgld, "chain_start_accuracy_mean"),
        "posterior_mean_abs_jaccard": optional_float(posterior_maps, "posterior_mean_abs"),
        "posterior_rms_jaccard": optional_float(posterior_maps, "posterior_rms"),
        "posterior_snr_jaccard": optional_float(posterior_maps, "posterior_snr"),
        "posterior_high_variance_jaccard": optional_float(
            posterior_maps, "posterior_high_variance"
        ),
        "posterior_low_variance_jaccard": optional_float(posterior_maps, "posterior_low_variance"),
        "posterior_mean_abs_chainwise_jaccard": optional_float(
            posterior_maps, "posterior_mean_abs_chainwise_mean"
        ),
        "posterior_rms_chainwise_jaccard": optional_float(
            posterior_maps, "posterior_rms_chainwise_mean"
        ),
        "posterior_snr_chainwise_jaccard": optional_float(
            posterior_maps, "posterior_snr_chainwise_mean"
        ),
        "posterior_high_variance_chainwise_jaccard": optional_float(
            posterior_maps, "posterior_high_variance_chainwise_mean"
        ),
        "posterior_low_variance_chainwise_jaccard": optional_float(
            posterior_maps, "posterior_low_variance_chainwise_mean"
        ),
        "dense_imp_linear_barrier": float(connectivity["dense_imp_barrier"]),
        "dense_sample_linear_barrier_mean": float(connectivity["dense_sample_barrier_mean"]),
        "imp_sample_linear_barrier_mean": float(connectivity["imp_sample_barrier_mean"]),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            summary[field] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
            continue
        summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("*/metrics.json"))
    if not paths:
        raise SystemExit(f"No metrics.json files found under {args.run_root}")
    rows = []
    skipped = []
    for path in paths:
        try:
            rows.append(load_row(path))
        except KeyError as exc:
            skipped.append({"path": str(path), "reason": f"missing key {exc}"})
    if not rows:
        raise SystemExit(f"No compatible metrics.json files found under {args.run_root}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "num_runs": len(rows),
        "skipped": skipped,
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
