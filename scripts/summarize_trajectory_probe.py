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
    "epoch",
    "num_runs",
    "checkpoint_accuracy",
    "trajectory_magnitude_to_imp_jaccard",
    "trajectory_to_dense_final_magnitude_jaccard",
    "trajectory_to_rewind_magnitude_jaccard",
    "dense_magnitude_to_imp_jaccard",
    "rewind_magnitude_to_imp_jaccard",
    "imp_accuracy",
    "imp_sparsity",
]

AGGREGATE_FIELDS = [
    "source",
    "num_runs",
    "trajectory_score_to_imp_jaccard",
    "trajectory_score_to_dense_final_magnitude_jaccard",
    "trajectory_score_to_rewind_magnitude_jaccard",
    "trajectory_score_to_best_checkpoint_jaccard",
    "mask_sparsity",
    "best_checkpoint_to_imp_jaccard",
    "dense_magnitude_to_imp_jaccard",
    "rewind_magnitude_to_imp_jaccard",
    "imp_accuracy",
    "imp_sparsity",
]

GROUP_FIELDS = [
    "source_kind",
    "source",
    "group",
    "num_runs",
    "group_jaccard",
    "kept",
    "imp_kept",
    "total",
]

LAYER_FIELDS = [
    "source_kind",
    "source",
    "parameter",
    "group",
    "num_runs",
    "layer_jaccard",
    "kept",
    "imp_kept",
    "total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-aggregate-csv", type=Path, default=None)
    parser.add_argument("--out-group-csv", type=Path, default=None)
    parser.add_argument("--out-layer-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def derived_summary_path(path: Path, label: str) -> Path:
    stem = path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    return path.with_name(f"{stem}_{label}_summary{path.suffix}")


def load_payloads(run_root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open(encoding="utf-8") as f:
            payloads.append(json.load(f))
    if not payloads:
        raise SystemExit(f"no trajectory probe metrics found under {run_root}")
    return payloads


def typed_row(seed: float, row: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {"seed": seed}
    for key, value in row.items():
        if value is None:
            out[key] = float("nan")
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def load_rows(payloads: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    rows = []
    for payload in payloads:
        seed = float(payload["seed"])
        for row in payload["rows"]:
            rows.append(typed_row(seed, row))
    if not rows:
        raise SystemExit("trajectory probe payloads have no checkpoint rows")
    return rows


def load_optional_rows(
    payloads: list[dict[str, Any]],
    payload_key: str,
) -> list[dict[str, float | str]]:
    rows = []
    for payload in payloads:
        seed = float(payload["seed"])
        for row in payload.get(payload_key, []):
            rows.append(typed_row(seed, row))
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


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float]]:
    by_epoch: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(row)

    summary = []
    for epoch in sorted(by_epoch):
        group = by_epoch[epoch]
        out: dict[str, float] = {
            "epoch": float(epoch),
            "num_runs": float(len({row["seed"] for row in group})),
        }
        for field in FIELDS:
            if field in {"epoch", "num_runs"}:
                continue
            values = [row.get(field, float("nan")) for row in group]
            low, high = ci95(values)
            out[field] = mean(values)
            out[f"{field}_std"] = std(values)
            out[f"{field}_ci95_low"] = low
            out[f"{field}_ci95_high"] = high
        summary.append(out)
    return summary


def summarize_keyed(
    rows: list[dict[str, float | str]],
    key_fields: list[str],
    fields: list[str],
) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, ...], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in key_fields)].append(row)

    summary: list[dict[str, float | str]] = []
    for key, group in sorted(grouped.items()):
        out: dict[str, float | str] = {
            field: value for field, value in zip(key_fields, key, strict=True)
        }
        out["num_runs"] = float(len({row["seed"] for row in group}))
        for field in fields:
            if field in set(key_fields) | {"num_runs"}:
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
    if text.startswith("epoch_"):
        return text.replace("epoch_", "Epoch ")
    return text.replace("traj_", "").replace("_", " ")


def write_markdown(
    rows: list[dict[str, float]],
    aggregate_rows: list[dict[str, float | str]],
    group_rows: list[dict[str, float | str]],
    path: Path,
) -> None:
    headers = [
        "Epoch",
        "Runs",
        "Checkpoint Acc",
        "Traj-IMP",
        "Traj-Dense",
        "Traj-Rewind",
        "Dense-IMP",
        "Rewind-IMP",
        "IMP Acc",
    ]
    keys = [
        "epoch",
        "num_runs",
        "checkpoint_accuracy",
        "trajectory_magnitude_to_imp_jaccard",
        "trajectory_to_dense_final_magnitude_jaccard",
        "trajectory_to_rewind_magnitude_jaccard",
        "dense_magnitude_to_imp_jaccard",
        "rewind_magnitude_to_imp_jaccard",
        "imp_accuracy",
    ]
    lines = [
        "# Trajectory Probe Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in keys) + " |")

    best = max(rows, key=lambda row: row["trajectory_magnitude_to_imp_jaccard"])
    best_aggregate = (
        max(aggregate_rows, key=lambda row: float(row["trajectory_score_to_imp_jaccard"]))
        if aggregate_rows
        else None
    )
    if aggregate_rows:
        lines.extend(
            [
                "",
                "## Aggregate Trajectory Score Masks",
                "",
                "| Source | Runs | Score-IMP | Score-Dense | Score-Rewind | Score-Best Ckpt | Sparsity |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        aggregate_keys = [
            "source",
            "num_runs",
            "trajectory_score_to_imp_jaccard",
            "trajectory_score_to_dense_final_magnitude_jaccard",
            "trajectory_score_to_rewind_magnitude_jaccard",
            "trajectory_score_to_best_checkpoint_jaccard",
            "mask_sparsity",
        ]
        for row in aggregate_rows:
            values = [source_label(row["source"])] + [
                fmt(row[key]) for key in aggregate_keys[1:]
            ]
            lines.append("| " + " | ".join(values) + " |")

    if group_rows:
        interesting = [("checkpoint", f"epoch_{int(best['epoch'])}")]
        if best_aggregate is not None:
            interesting.append(("aggregate", str(best_aggregate["source"])))
        selected_groups = [
            row
            for row in group_rows
            if (row["source_kind"], row["source"]) in interesting
        ]
        lines.extend(
            [
                "",
                "## Stage-Level Overlap",
                "",
                "| Source | Stage | Stage-IMP | Kept | IMP Kept |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in selected_groups:
            lines.append(
                "| "
                + " | ".join(
                    [
                        source_label(row["source"]),
                        str(row["group"]),
                        fmt(row["group_jaccard"]),
                        fmt(row["kept"]),
                        fmt(row["imp_kept"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            f"The best dense-trajectory magnitude support is epoch {int(best['epoch'])} "
            f"with Traj-IMP {fmt(best['trajectory_magnitude_to_imp_jaccard'])}.",
            (
                f"The best aggregate trajectory score mask is "
                f"{source_label(best_aggregate['source'])} with Score-IMP "
                f"{fmt(best_aggregate['trajectory_score_to_imp_jaccard'])}."
                if best_aggregate is not None
                else ""
            ),
            "A trajectory-subspace account is strengthened when early or rewind",
            "trajectory masks exceed posterior-induced supports under the same IMP",
            "mask and sparsity.",
            "",
            "This file is generated by `scripts/summarize_trajectory_probe.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, float | str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    key_fields = [field for field in fields if field in rows[0] and isinstance(rows[0][field], str)]
    for field in fields:
        fieldnames.append(field)
        if field not in set(key_fields) | {"epoch", "num_runs"}:
            fieldnames.extend([f"{field}_std", f"{field}_ci95_low", f"{field}_ci95_high"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payloads = load_payloads(args.run_root)
    rows = summarize(load_rows(payloads))
    aggregate_rows = summarize_keyed(
        load_optional_rows(payloads, "aggregate_rows"),
        key_fields=["source"],
        fields=AGGREGATE_FIELDS,
    )
    group_rows = summarize_keyed(
        load_optional_rows(payloads, "group_rows"),
        key_fields=["source_kind", "source", "group"],
        fields=GROUP_FIELDS,
    )
    layer_rows = summarize_keyed(
        load_optional_rows(payloads, "layer_rows"),
        key_fields=["source_kind", "source", "parameter", "group"],
        fields=LAYER_FIELDS,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for field in FIELDS:
        fieldnames.append(field)
        if field not in {"epoch", "num_runs"}:
            fieldnames.extend([f"{field}_std", f"{field}_ci95_low", f"{field}_ci95_high"])
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if aggregate_rows:
        write_csv(
            args.out_aggregate_csv or derived_summary_path(args.out_csv, "aggregate"),
            aggregate_rows,
            AGGREGATE_FIELDS,
        )
    if group_rows:
        write_csv(
            args.out_group_csv or derived_summary_path(args.out_csv, "group"),
            group_rows,
            GROUP_FIELDS,
        )
    if layer_rows:
        write_csv(
            args.out_layer_csv or derived_summary_path(args.out_csv, "layer"),
            layer_rows,
            LAYER_FIELDS,
        )
    write_markdown(rows, aggregate_rows, group_rows, args.out_md)
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out_csv}")
    if aggregate_rows:
        print(f"wrote {args.out_aggregate_csv or derived_summary_path(args.out_csv, 'aggregate')}")
    if group_rows:
        print(f"wrote {args.out_group_csv or derived_summary_path(args.out_csv, 'group')}")
    if layer_rows:
        print(f"wrote {args.out_layer_csv or derived_summary_path(args.out_csv, 'layer')}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
