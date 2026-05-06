#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--min-random-gap", type=float, default=0.02)
    parser.add_argument("--min-chain-start-gap", type=float, default=0.02)
    parser.add_argument("--max-posterior-chain-start-overlap", type=float, default=0.90)
    parser.add_argument("--max-dense-minus-posterior", type=float, default=0.05)
    return parser.parse_args()


def mean(summary: dict, field: str) -> float:
    return float(summary["summary"][field]["mean"])


def optional_mean(summary: dict, field: str) -> float | None:
    if field not in summary["summary"]:
        return None
    return float(summary["summary"][field]["mean"])


def main() -> None:
    args = parse_args()
    with args.summary_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    posterior = mean(payload, "posterior_jaccard_mean")
    random = mean(payload, "random_jaccard_mean")
    chain_start = mean(payload, "chain_start_magnitude_to_imp_jaccard_mean")
    posterior_to_chain_start = mean(payload, "posterior_to_chain_start_magnitude_jaccard_mean")
    dense = mean(payload, "dense_magnitude_to_imp_jaccard")
    state_clusters = mean(payload, "state_num_clusters")
    function_clusters = mean(payload, "function_num_clusters")
    dense_imp_barrier = optional_mean(payload, "dense_imp_linear_barrier")

    checks = [
        {
            "name": "posterior_beats_random",
            "value": posterior - random,
            "threshold": args.min_random_gap,
            "passed": posterior - random >= args.min_random_gap,
        },
        {
            "name": "posterior_exceeds_chain_start",
            "value": posterior - chain_start,
            "threshold": args.min_chain_start_gap,
            "passed": posterior - chain_start >= args.min_chain_start_gap,
        },
        {
            "name": "posterior_moves_support_from_chain_start",
            "value": posterior_to_chain_start,
            "threshold": args.max_posterior_chain_start_overlap,
            "passed": posterior_to_chain_start <= args.max_posterior_chain_start_overlap,
        },
        {
            "name": "dense_magnitude_does_not_dominate",
            "value": dense - posterior,
            "threshold": args.max_dense_minus_posterior,
            "passed": dense - posterior <= args.max_dense_minus_posterior,
        },
        {
            "name": "records_function_and_state_clusters",
            "value": min(state_clusters, function_clusters),
            "threshold": 1.0,
            "passed": state_clusters >= 1.0 and function_clusters >= 1.0,
        },
    ]
    if dense_imp_barrier is not None:
        checks.append(
            {
                "name": "records_dense_imp_connectivity",
                "value": dense_imp_barrier,
                "threshold": 0.0,
                "passed": dense_imp_barrier >= 0.0,
            },
        )

    result = {
        "summary_json": str(args.summary_json),
        "num_runs": payload["num_runs"],
        "metrics": {
            "posterior_jaccard_mean": posterior,
            "random_jaccard_mean": random,
            "chain_start_magnitude_to_imp_jaccard_mean": chain_start,
            "posterior_to_chain_start_magnitude_jaccard_mean": posterior_to_chain_start,
            "dense_magnitude_to_imp_jaccard": dense,
            "state_num_clusters": state_clusters,
            "function_num_clusters": function_clusters,
            "dense_imp_linear_barrier": dense_imp_barrier,
        },
        "checks": checks,
        "gate1_passed": all(check["passed"] for check in checks),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
