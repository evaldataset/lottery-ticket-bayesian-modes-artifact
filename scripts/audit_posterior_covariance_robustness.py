#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = ROOT / "runs" / "posterior_covariance_robustness_audit.csv"
DEFAULT_OUT_JSON = ROOT / "runs" / "posterior_covariance_robustness_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "posterior_covariance_robustness_audit.md"

LOWRANK_SOURCES = [
    {
        "label": "LowRank16 Laplace",
        "rank": 16,
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv",
    },
    {
        "label": "LowRank32 Laplace",
        "rank": 32,
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv",
    },
    {
        "label": "LowRank64 Laplace",
        "rank": 64,
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv",
    },
    {
        "label": "LowRank128 Laplace",
        "rank": 128,
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv",
    },
]

EXACT_COVARIANCE_SOURCES = [
    {
        "label": "BlockDiag22k Laplace",
        "family": "exact tensor-block covariance",
        "coverage": "11 tensor blocks",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv",
        "scale": 1e-4,
    },
    {
        "label": "BlockDiag68k Laplace",
        "family": "exact tensor-block covariance",
        "coverage": "16 tensor blocks",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
        "scale": 1e-5,
    },
    {
        "label": "JointDiag68k Laplace",
        "family": "exact joint-group covariance",
        "coverage": "8 joint groups",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv",
        "scale": 1e-5,
    },
    {
        "label": "JointDiag86k Laplace",
        "family": "exact joint-group covariance",
        "coverage": "6 joint groups",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv",
        "scale": 3e-6,
    },
    {
        "label": "JointDiag270k Laplace",
        "family": "exact joint-group covariance",
        "coverage": "8 streamed joint groups over the full weight vector",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv",
        "scale": 1e-6,
    },
]

DIRECT_SOURCES = [
    {
        "label": "LowRank128 direct samples",
        "family": "low-rank Hessian-plus-diagonal",
        "coverage": "full-network rank-128 posterior samples",
        "rank": 128,
        "parameter_count_key": "all_trainable",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv",
    },
    {
        "label": "JointDiag270k direct samples",
        "family": "exact joint-group covariance",
        "coverage": "direct 270,896-weight joint-group posterior samples",
        "rank": None,
        "parameter_count_key": "weight_only",
        "source_csv": "runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv",
    },
]

CSV_FIELDS = [
    "row_kind",
    "label",
    "covariance_family",
    "coverage",
    "source_csv",
    "parameter_count",
    "rank",
    "scale",
    "num_runs",
    "posterior_jaccard",
    "chain_start_jaccard",
    "posterior_minus_chain_start_jaccard",
    "posterior_to_chain_start_jaccard",
    "rewind_minus_posterior_jaccard",
    "sample_accuracy",
    "selected_posterior_minus_chain_start_jaccard",
    "selected_posterior_to_chain_start_jaccard",
    "examples_seen",
    "posterior_sampler",
    "comparison",
    "posterior_num_clusters",
    "left_count",
    "right_count",
    "layer_ks_pvalue",
    "hamming_overlap",
    "logit_cka_hungarian_mean",
    "activation_cka_hungarian_mean",
    "passes_layer_ks",
    "passes_hamming_overlap",
    "passes_logit_cka",
    "passes_activation_cka",
    "direct_equivalence_passed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing source CSV: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty source CSV: {path}")
    return rows


def get_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def get_bool(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise SystemExit(f"missing boolean field {key}: {row}")


def require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise SystemExit(f"non-finite value for {label}")
    return value


def first_float_match(rows: list[dict[str, str]], key: str, expected: float) -> dict[str, str]:
    for row in rows:
        if abs(get_float(row, key) - expected) <= max(1e-12, abs(expected) * 1e-9):
            return row
    raise SystemExit(f"missing row where {key}={expected}")


def first_text_match(rows: list[dict[str, str]], key: str, expected: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == expected:
            return row
    raise SystemExit(f"missing row where {key}={expected}")


def build_lowrank_row(spec: dict[str, Any], feasibility: dict[str, Any]) -> dict[str, Any]:
    source_csv = str(spec["source_csv"])
    source = load_csv(ROOT / source_csv)
    row = first_float_match(source, "sgld_lr", 0.01)
    posterior = get_float(row, "posterior_jaccard_mean")
    rewind = get_float(row, "rewind_magnitude_to_imp_jaccard")
    out = {
        "row_kind": "movement",
        "label": spec["label"],
        "covariance_family": "low-rank Hessian-plus-diagonal",
        "coverage": "full-network covariance-fidelity spectrum",
        "source_csv": source_csv,
        "parameter_count": int(feasibility["all_trainable"]["parameter_count"]),
        "rank": int(spec["rank"]),
        "scale": get_float(row, "sgld_lr"),
        "num_runs": int(round(get_float(row, "num_runs"))),
        "posterior_jaccard": posterior,
        "chain_start_jaccard": get_float(row, "chain_start_magnitude_to_imp_jaccard"),
        "posterior_minus_chain_start_jaccard": get_float(
            row, "posterior_minus_chain_start_jaccard"
        ),
        "posterior_to_chain_start_jaccard": get_float(
            row, "posterior_to_chain_start_magnitude_jaccard_mean"
        ),
        "rewind_minus_posterior_jaccard": rewind - posterior,
        "sample_accuracy": get_float(row, "sample_accuracy_mean"),
    }
    for key in [
        "posterior_jaccard",
        "chain_start_jaccard",
        "posterior_minus_chain_start_jaccard",
        "posterior_to_chain_start_jaccard",
        "rewind_minus_posterior_jaccard",
        "sample_accuracy",
    ]:
        require_finite(float(out[key]), f"{out['label']} {key}")
    return out


def build_exact_covariance_row(spec: dict[str, Any]) -> dict[str, Any]:
    source_csv = str(spec["source_csv"])
    source = load_csv(ROOT / source_csv)
    row = first_float_match(source, "block_laplace_scale", float(spec["scale"]))
    global_posterior = get_float(row, "global_posterior_jaccard_mean")
    global_rewind = get_float(row, "global_rewind_magnitude_to_imp_jaccard")
    out = {
        "row_kind": "movement",
        "label": spec["label"],
        "covariance_family": spec["family"],
        "coverage": spec["coverage"],
        "source_csv": source_csv,
        "parameter_count": int(round(get_float(row, "block_parameter_count"))),
        "rank": None,
        "scale": get_float(row, "block_laplace_scale"),
        "num_runs": int(round(get_float(row, "num_runs"))),
        "posterior_jaccard": global_posterior,
        "chain_start_jaccard": get_float(row, "global_chain_start_magnitude_to_imp_jaccard"),
        "posterior_minus_chain_start_jaccard": get_float(
            row, "global_posterior_minus_chain_start_jaccard"
        ),
        "posterior_to_chain_start_jaccard": get_float(
            row, "global_posterior_to_chain_start_magnitude_jaccard_mean"
        ),
        "rewind_minus_posterior_jaccard": global_rewind - global_posterior,
        "sample_accuracy": get_float(row, "sample_accuracy_mean"),
        "selected_posterior_minus_chain_start_jaccard": get_float(
            row, "block_posterior_minus_chain_start_jaccard"
        ),
        "selected_posterior_to_chain_start_jaccard": get_float(
            row, "block_posterior_to_chain_start_magnitude_jaccard_mean"
        ),
        "examples_seen": int(round(get_float(row, "block_examples_seen"))),
    }
    for key in [
        "posterior_jaccard",
        "chain_start_jaccard",
        "posterior_minus_chain_start_jaccard",
        "posterior_to_chain_start_jaccard",
        "rewind_minus_posterior_jaccard",
        "sample_accuracy",
        "selected_posterior_minus_chain_start_jaccard",
        "selected_posterior_to_chain_start_jaccard",
    ]:
        require_finite(float(out[key]), f"{out['label']} {key}")
    return out


def build_direct_row(spec: dict[str, Any], feasibility: dict[str, Any]) -> dict[str, Any]:
    source_csv = str(spec["source_csv"])
    source = load_csv(ROOT / source_csv)
    row = first_text_match(source, "comparison", "posterior_samples_vs_tickets")
    passes_layer_ks = get_bool(row, "passes_layer_ks")
    passes_hamming = get_bool(row, "passes_hamming_overlap")
    passes_logit = get_bool(row, "passes_logit_cka")
    passes_activation = get_bool(row, "passes_activation_cka")
    parameter_source = str(spec["parameter_count_key"])
    direct_equivalence_passed = (
        passes_layer_ks and passes_hamming and passes_logit and passes_activation
    )
    out = {
        "row_kind": "direct",
        "label": spec["label"],
        "covariance_family": spec["family"],
        "coverage": spec["coverage"],
        "source_csv": source_csv,
        "parameter_count": int(feasibility[parameter_source]["parameter_count"]),
        "rank": spec["rank"],
        "posterior_sampler": row["posterior_sampler"],
        "comparison": row["comparison"],
        "posterior_num_clusters": int(round(get_float(row, "posterior_num_clusters"))),
        "left_count": int(round(get_float(row, "left_count"))),
        "right_count": int(round(get_float(row, "right_count"))),
        "layer_ks_pvalue": get_float(row, "layer_ks_pvalue"),
        "hamming_overlap": get_float(row, "hamming_overlap"),
        "logit_cka_hungarian_mean": get_float(row, "logit_cka_hungarian_mean"),
        "activation_cka_hungarian_mean": get_float(row, "activation_cka_hungarian_mean"),
        "passes_layer_ks": passes_layer_ks,
        "passes_hamming_overlap": passes_hamming,
        "passes_logit_cka": passes_logit,
        "passes_activation_cka": passes_activation,
        "direct_equivalence_passed": direct_equivalence_passed,
    }
    for key in [
        "layer_ks_pvalue",
        "hamming_overlap",
        "logit_cka_hungarian_mean",
        "activation_cka_hungarian_mean",
    ]:
        require_finite(float(out[key]), f"{out['label']} {key}")
    return out


def build_payload() -> dict[str, Any]:
    feasibility = load_json(ROOT / "runs" / "cifar10_resnet20_full_covariance_feasibility.json")
    rows: list[dict[str, Any]] = []
    rows.extend(build_lowrank_row(spec, feasibility) for spec in LOWRANK_SOURCES)
    rows.extend(build_exact_covariance_row(spec) for spec in EXACT_COVARIANCE_SOURCES)
    rows.extend(build_direct_row(spec, feasibility) for spec in DIRECT_SOURCES)

    movement_rows = [row for row in rows if row["row_kind"] == "movement"]
    direct_rows = [row for row in rows if row["row_kind"] == "direct"]
    exact_rows = [
        row
        for row in movement_rows
        if str(row["covariance_family"]).startswith("exact")
    ]
    max_movement_gain = max(
        float(row["posterior_minus_chain_start_jaccard"]) for row in movement_rows
    )
    min_exact_rewind_minus_posterior = min(
        float(row["rewind_minus_posterior_jaccard"]) for row in exact_rows
    )
    min_movement_accuracy = min(float(row["sample_accuracy"]) for row in movement_rows)
    dense = feasibility["all_trainable"]
    dense_ready = (
        float(dense["dense_precision_float64_gib"]) > 500.0
        and float(dense["precision_plus_cholesky_float64_gib"]) > 1000.0
    )
    checks = {
        "movement_row_count": len(movement_rows),
        "direct_row_count": len(direct_rows),
        "all_movement_rows_five_seed": all(int(row["num_runs"]) == 5 for row in movement_rows),
        "all_movement_rows_preserve_accuracy": min_movement_accuracy >= 0.875,
        "no_movement_row_beats_chain_start_by_0p005": max_movement_gain <= 0.005,
        "exact_rows_rewind_remains_closer": min_exact_rewind_minus_posterior > 0.025,
        "direct_rows_one_cluster": all(
            int(row["posterior_num_clusters"]) == 1 for row in direct_rows
        ),
        "direct_rows_fail_equivalence": all(
            not bool(row["direct_equivalence_passed"]) for row in direct_rows
        ),
        "dense_full_covariance_infeasible": dense_ready,
        "max_movement_posterior_minus_chain_start_jaccard": max_movement_gain,
        "min_exact_rewind_minus_posterior_jaccard": min_exact_rewind_minus_posterior,
        "min_movement_sample_accuracy": min_movement_accuracy,
    }
    checks["posterior_covariance_robustness_ready"] = all(
        bool(checks[key])
        for key in [
            "all_movement_rows_five_seed",
            "all_movement_rows_preserve_accuracy",
            "no_movement_row_beats_chain_start_by_0p005",
            "exact_rows_rewind_remains_closer",
            "direct_rows_one_cluster",
            "direct_rows_fail_equivalence",
            "dense_full_covariance_infeasible",
        ]
    )
    return {
        "source_note": (
            "This audit aggregates existing artifacts; it does not create new posterior "
            "samples or close the exact dense CIFAR posterior limitation."
        ),
        "rows": rows,
        "interpretation_checks": checks,
        "dense_feasibility": {
            "parameter_count": int(dense["parameter_count"]),
            "dense_precision_float64_gib": float(dense["dense_precision_float64_gib"]),
            "precision_plus_cholesky_float64_gib": float(
                dense["precision_plus_cholesky_float64_gib"]
            ),
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) < 0.001 and number != 0.0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in CSV_FIELDS})


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    movement_rows = [row for row in rows if row["row_kind"] == "movement"]
    direct_rows = [row for row in rows if row["row_kind"] == "direct"]
    dense = payload["dense_feasibility"]
    checks = payload["interpretation_checks"]
    lines = [
        "# Posterior Covariance Robustness Audit",
        "",
        "This generated audit summarizes the covariance-fidelity spectrum already",
        "present in the artifact. It is a trend audit over existing movement and",
        "direct mode-ticket rows, not a new posterior sampler run and not a claim",
        "that exact dense CIFAR full-covariance posterior evidence is available.",
        "",
        "## Dense CIFAR Bound",
        "",
        (
            f"All-trainable CIFAR ResNet-20 parameters: {dense['parameter_count']:,}. "
            f"One float64 dense precision matrix is {dense['dense_precision_float64_gib']:.1f} GiB; "
            "keeping the precision matrix plus Cholesky factor resident is "
            f"{dense['precision_plus_cholesky_float64_gib']:.1f} GiB."
        ),
        "",
        "## Movement Rows",
        "",
        "| Row | Coverage | Params/rank | Posterior-chain | Post-chain | Rewind-posterior | Sample acc | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in movement_rows:
        rank = f"r={row['rank']}" if row.get("rank") is not None else ""
        params = f"{int(row['parameter_count']):,}"
        params_rank = f"{params} {rank}".strip()
        lines.append(
            "| {label} | {coverage} | {params_rank} | {gap} | {post_chain} | {rewind_gap} | {acc} | `{source}` |".format(
                label=row["label"],
                coverage=row["coverage"],
                params_rank=params_rank,
                gap=fmt(row["posterior_minus_chain_start_jaccard"]),
                post_chain=fmt(row["posterior_to_chain_start_jaccard"]),
                rewind_gap=fmt(row["rewind_minus_posterior_jaccard"]),
                acc=fmt(row["sample_accuracy"]),
                source=row["source_csv"],
            )
        )
    lines.extend(
        [
            "",
            "## Direct Distribution Rows",
            "",
            "| Row | Samples/tickets | Clusters | Layer KS p | Hamming overlap | Logit CKA | Activation CKA | Direct equivalence | Source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in direct_rows:
        lines.append(
            "| {label} | {left}/{right} | {clusters} | {pvalue} | {hamming} | {logit} | {activation} | {equiv} | `{source}` |".format(
                label=row["label"],
                left=int(row["left_count"]),
                right=int(row["right_count"]),
                clusters=int(row["posterior_num_clusters"]),
                pvalue=fmt(row["layer_ks_pvalue"]),
                hamming=fmt(row["hamming_overlap"]),
                logit=fmt(row["logit_cka_hungarian_mean"]),
                activation=fmt(row["activation_cka_hungarian_mean"]),
                equiv="pass" if row["direct_equivalence_passed"] else "fail",
                source=row["source_csv"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Checks",
            "",
            f"- Movement rows: {checks['movement_row_count']}; direct rows: {checks['direct_row_count']}.",
            (
                "- All movement rows preserve sample accuracy above 0.875: "
                f"{fmt(checks['all_movement_rows_preserve_accuracy'])} "
                f"(minimum {fmt(checks['min_movement_sample_accuracy'])})."
            ),
            (
                "- No movement row beats chain-start by 0.005 Jaccard: "
                f"{fmt(checks['no_movement_row_beats_chain_start_by_0p005'])} "
                f"(maximum posterior-chain {fmt(checks['max_movement_posterior_minus_chain_start_jaccard'])})."
            ),
            (
                "- Exact block/joint rows keep rewind support closer than posterior support: "
                f"{fmt(checks['exact_rows_rewind_remains_closer'])} "
                f"(minimum rewind-posterior {fmt(checks['min_exact_rewind_minus_posterior_jaccard'])})."
            ),
            (
                "- Direct covariance rows fail proposal-level equivalence while retaining one posterior cluster: "
                f"{fmt(checks['direct_rows_fail_equivalence'])}."
            ),
            (
                "- Posterior covariance robustness ready: "
                f"{fmt(checks['posterior_covariance_robustness_ready'])}."
            ),
            "",
            "Exact dense full-network CIFAR covariance remains a bounded open limitation;",
            "this audit strengthens the trend evidence across feasible covariance",
            "families and parameter coverage.",
            "",
            "This file is generated by `scripts/audit_posterior_covariance_robustness.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload()
    if payload["interpretation_checks"]["posterior_covariance_robustness_ready"] is not True:
        raise SystemExit("posterior covariance robustness checks did not pass")
    write_csv(payload["rows"], args.out_csv)
    write_json(payload, args.out_json)
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                "rows": len(payload["rows"]),
                "out_csv": str(args.out_csv),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
