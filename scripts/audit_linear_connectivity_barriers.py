#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = ROOT / "runs" / "linear_connectivity_barrier_audit.csv"
DEFAULT_OUT_JSON = ROOT / "runs" / "linear_connectivity_barrier_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "linear_connectivity_barrier_audit.md"

T975_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class SourceSpec:
    label: str
    dataset: str
    model: str
    sampler: str
    source_csv: str


@dataclass(frozen=True)
class MetricSpec:
    out_key: str
    label: str
    getter: Callable[[dict[str, str]], float]


DEFAULT_SOURCES = [
    SourceSpec(
        label="MNIST Gate1 SGLD r5",
        dataset="MNIST",
        model="MLP",
        sampler="SGLD",
        source_csv="runs/mnist_gate1_full_r5_p0p3_summary.csv",
    ),
    SourceSpec(
        label="Fashion-MNIST Gate1 SGLD r5",
        dataset="Fashion-MNIST",
        model="MLP",
        sampler="SGLD",
        source_csv="runs/fashion_gate1_full_r5_p0p3_summary.csv",
    ),
    SourceSpec(
        label="CIFAR-10 ResNet-20 long SGLD r5",
        dataset="CIFAR-10",
        model="ResNet-20",
        sampler="SGLD",
        source_csv="runs/cifar10_resnet20_long30_rewind1_r5_p0p3_summary.csv",
    ),
    SourceSpec(
        label="CIFAR-10 ResNet-20 long SWAG r5",
        dataset="CIFAR-10",
        model="ResNet-20",
        sampler="SWAG",
        source_csv="runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_summary.csv",
    ),
    SourceSpec(
        label="CIFAR-10 ResNet-20 3-chain SGLD r5",
        dataset="CIFAR-10",
        model="ResNet-20",
        sampler="3-chain SGLD",
        source_csv="runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_summary.csv",
    ),
    SourceSpec(
        label="CIFAR-10 ResNet-20 short SWAG r5",
        dataset="CIFAR-10",
        model="ResNet-20",
        sampler="short SWAG",
        source_csv="runs/cifar10_resnet20_swag_short_r5_p0p3_summary.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def get_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    values = finite(values)
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    values = finite(values)
    if len(values) < 2:
        return 0.0 if values else float("nan")
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ci95(values: list[float]) -> tuple[float, float]:
    values = finite(values)
    if not values:
        return float("nan"), float("nan")
    center = mean(values)
    if len(values) < 2:
        return center, center
    t_critical = T975_CRITICAL.get(len(values) - 1, 1.96)
    half_width = t_critical * std(values) / math.sqrt(len(values))
    return center - half_width, center + half_width


def metric_from_field(out_key: str, label: str, field: str) -> MetricSpec:
    return MetricSpec(out_key, label, lambda row, field=field: get_float(row, field))


METRICS = [
    metric_from_field("dense_imp_barrier", "Dense-IMP barrier", "dense_imp_linear_barrier"),
    metric_from_field(
        "dense_sample_barrier",
        "Dense-sample barrier",
        "dense_sample_linear_barrier_mean",
    ),
    metric_from_field(
        "imp_sample_barrier",
        "IMP-sample barrier",
        "imp_sample_linear_barrier_mean",
    ),
    metric_from_field("posterior_jaccard", "Posterior-IMP Jaccard", "posterior_jaccard_mean"),
    metric_from_field(
        "chain_start_jaccard",
        "Chain-start magnitude-IMP Jaccard",
        "chain_start_magnitude_to_imp_jaccard_mean",
    ),
    MetricSpec(
        "posterior_minus_chain_start_jaccard",
        "Posterior minus chain-start Jaccard",
        lambda row: get_float(row, "posterior_jaccard_mean")
        - get_float(row, "chain_start_magnitude_to_imp_jaccard_mean"),
    ),
    metric_from_field(
        "posterior_to_chain_start_jaccard",
        "Posterior to chain-start Jaccard",
        "posterior_to_chain_start_magnitude_jaccard_mean",
    ),
    metric_from_field(
        "dense_magnitude_jaccard",
        "Dense magnitude-IMP Jaccard",
        "dense_magnitude_to_imp_jaccard",
    ),
    metric_from_field("random_jaccard", "Random-IMP Jaccard", "random_jaccard_mean"),
    metric_from_field(
        "posterior_minus_random_jaccard",
        "Posterior minus random Jaccard",
        "posterior_minus_random_jaccard",
    ),
    metric_from_field("state_num_clusters", "State clusters", "state_num_clusters"),
    metric_from_field("function_num_clusters", "Function clusters", "function_num_clusters"),
    metric_from_field("sample_accuracy", "Posterior sample accuracy", "sgld_sample_accuracy_mean"),
    metric_from_field("dense_accuracy", "Dense accuracy", "dense_accuracy"),
    metric_from_field("imp_accuracy", "IMP accuracy", "imp_accuracy"),
    metric_from_field("sampler_chains", "Sampler chains", "sgld_chains"),
]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing source CSV: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty source CSV: {path}")
    return rows


def summarize_metric(rows: list[dict[str, str]], metric: MetricSpec) -> dict[str, float]:
    values = [metric.getter(row) for row in rows]
    low, high = ci95(values)
    clean = finite(values)
    return {
        "n": float(len(clean)),
        "mean": mean(values),
        "std": std(values),
        "ci95_low": low,
        "ci95_high": high,
    }


def summarize_source(spec: SourceSpec) -> dict[str, Any]:
    rows = load_csv(ROOT / spec.source_csv)
    summary: dict[str, Any] = {
        "label": spec.label,
        "dataset": spec.dataset,
        "model": spec.model,
        "sampler": spec.sampler,
        "source_csv": spec.source_csv,
        "num_runs": len(rows),
        "metrics": {},
    }
    for metric in METRICS:
        stats = summarize_metric(rows, metric)
        if stats["n"] != float(len(rows)):
            raise SystemExit(
                f"{spec.source_csv} missing metric {metric.out_key}: "
                f"{int(stats['n'])}/{len(rows)} finite values"
            )
        summary["metrics"][metric.out_key] = stats
    return summary


def flatten_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "label": summary["label"],
        "dataset": summary["dataset"],
        "model": summary["model"],
        "sampler": summary["sampler"],
        "source_csv": summary["source_csv"],
        "num_runs": summary["num_runs"],
    }
    for metric in METRICS:
        stats = summary["metrics"][metric.out_key]
        row[f"{metric.out_key}_mean"] = stats["mean"]
        row[f"{metric.out_key}_std"] = stats["std"]
        row[f"{metric.out_key}_ci95_low"] = stats["ci95_low"]
        row[f"{metric.out_key}_ci95_high"] = stats["ci95_high"]
    return row


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        if abs(number) < 1e-4 and number != 0.0:
            return f"{number:.1e}"
        if abs(number) >= 1000:
            return f"{number:,.1f}"
        return f"{number:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "label",
        "dataset",
        "model",
        "sampler",
        "source_csv",
        "num_runs",
    ]
    for metric in METRICS:
        for suffix in ["mean", "std", "ci95_low", "ci95_high"]:
            fieldnames.append(f"{metric.out_key}_{suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summaries: list[dict[str, Any]], path: Path) -> None:
    by_label = {row["label"]: row for row in summaries}
    payload = {
        "rows": summaries,
        "interpretation_checks": {
            "all_rows_five_seed": all(row["num_runs"] == 5 for row in summaries),
            "mnist_dense_imp_barrier_near_zero": by_label[
                "MNIST Gate1 SGLD r5"
            ]["metrics"]["dense_imp_barrier"]["mean"]
            < 0.01,
            "fashion_dense_imp_barrier_near_zero": by_label[
                "Fashion-MNIST Gate1 SGLD r5"
            ]["metrics"]["dense_imp_barrier"]["mean"]
            < 0.05,
            "cifar_dense_imp_barriers_large": all(
                row["metrics"]["dense_imp_barrier"]["mean"] > 2.0
                for row in summaries
                if row["dataset"] == "CIFAR-10"
            ),
            "posterior_never_beats_chain_start": all(
                row["metrics"]["posterior_minus_chain_start_jaccard"]["mean"] <= 0.001
                for row in summaries
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    rows = [flatten_row(row) for row in summaries]
    lines = [
        "# Linear Connectivity Barrier Audit",
        "",
        "This audit reuses existing five-seed summary CSVs to put linear",
        "loss-barrier probes next to posterior-support overlap controls. It",
        "asks whether linear mode-connectivity barriers rescue the posterior",
        "ticket-support equivalence claim.",
        "",
        "| Setting | Runs | Dense-IMP Barrier | Dense-Sample Barrier | IMP-Sample Barrier | Posterior | Chain Start | Posterior-Chain | Post-Chain | Sample Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["num_runs"]),
                    fmt(row["dense_imp_barrier_mean"]),
                    fmt(row["dense_sample_barrier_mean"]),
                    fmt(row["imp_sample_barrier_mean"]),
                    fmt(row["posterior_jaccard_mean"]),
                    fmt(row["chain_start_jaccard_mean"]),
                    fmt(row["posterior_minus_chain_start_jaccard_mean"]),
                    fmt(row["posterior_to_chain_start_jaccard_mean"]),
                    fmt(row["sample_accuracy_mean"]),
                ]
            )
            + " |"
        )

    mnist = rows[0]
    fashion = rows[1]
    cifar_sgld = rows[2]
    cifar_swag = rows[3]
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- MNIST and Fashion-MNIST have nearly connected dense-to-IMP paths "
            f"({fmt(mnist['dense_imp_barrier_mean'])} and "
            f"{fmt(fashion['dense_imp_barrier_mean'])}), but posterior samples "
            "still do not beat the chain-start magnitude support.",
            "- CIFAR-10 ResNet-20 rows have large linear barriers across dense, "
            f"IMP, and posterior samples (long SGLD dense-IMP "
            f"{fmt(cifar_sgld['dense_imp_barrier_mean'])}; long SWAG dense-IMP "
            f"{fmt(cifar_swag['dense_imp_barrier_mean'])}), yet posterior "
            "support remains tied to the chain-start control.",
            "- Linear connectivity barriers are orthogonal landscape diagnostics, "
            "not evidence of posterior-ticket equivalence.",
            "",
            "This file is generated by `scripts/audit_linear_connectivity_barriers.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = [summarize_source(spec) for spec in DEFAULT_SOURCES]
    write_csv([flatten_row(row) for row in summaries], args.out_csv)
    write_json(summaries, args.out_json)
    write_markdown(summaries, args.out_md)
    print(
        json.dumps(
            {
                "rows": len(summaries),
                "out_csv": str(args.out_csv),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
