#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


VALUE_KEYS = [
    "trained_accuracy",
    "accuracy_minus_imp",
    "accuracy_minus_dense",
    "base_to_imp_jaccard",
    "base_to_reference_jaccard",
    "mask_to_imp_jaccard",
    "mask_to_effective_base_jaccard",
    "mask_to_reference_base_jaccard",
    "swap_count",
    "candidate_count",
    "heldout_count",
    "heldout_positive_count",
    "added_imp_only_hits",
    "added_imp_only_precision",
    "added_oracle_overlap_hits",
    "added_oracle_overlap_precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload["rows"]:
            out = dict(row)
            out["run_path"] = str(path)
            rows.append(out)
    if not rows:
        raise SystemExit(f"no residual base compatibility metrics found under {run_root}")
    return rows


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def stats(values: Iterable[object]) -> dict[str, float | int | None]:
    clean = [to_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    if len(clean) == 1:
        return {"n": 1, "mean": clean[0], "std": 0.0, "ci95": 0.0}
    sd = stdev(clean)
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": sd,
        "ci95": 1.96 * sd / math.sqrt(len(clean)),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(row["base_source"], row["base_kind"], row["variant"], row["alpha"])].append(row)
    out = []
    for (base_source, base_kind, variant, alpha), group in sorted(buckets.items()):
        summary: dict[str, object] = {
            "base_source": base_source,
            "base_kind": base_kind,
            "variant": variant,
            "alpha": alpha,
            "seeds": len({row["seed"] for row in group}),
            "rows": len(group),
        }
        for key in VALUE_KEYS:
            value_stats = stats(row.get(key) for row in group)
            summary[f"{key}_mean"] = value_stats["mean"]
            summary[f"{key}_std"] = value_stats["std"]
            summary[f"{key}_ci95"] = value_stats["ci95"]
            summary[f"{key}_n"] = value_stats["n"]
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) >= 1000:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def fmt_ci(row: dict[str, object], key: str, digits: int = 4) -> str:
    center = to_float(row.get(f"{key}_mean"))
    ci = to_float(row.get(f"{key}_ci95"))
    if center is None:
        return ""
    if ci is None:
        return fmt(center, digits)
    return f"{fmt(center, digits)} [{fmt(center - ci, digits)}, {fmt(center + ci, digits)}]"


def label(text: object) -> str:
    labels = {
        "epoch_30": "Epoch 30",
        "epoch_10": "Epoch 10",
        "traj_rms_abs": "rms abs",
        "trajectory": "trajectory",
        "imp_overlap_random": "IMP-overlap random",
        "oracle_imp_residual": "oracle residual",
        "posterior_imp_only_residual": "posterior IMP-only",
        "dense_imp_only_residual": "dense IMP-only",
        "posterior_excess_imp_only_residual": "posterior excess IMP-only",
        "posterior_std_imp_only_residual": "posterior std IMP-only",
        "random_imp_only_residual": "random IMP-only",
        "low_imp_only_residual": "low IMP-only",
        "random_residual": "random residual",
    }
    return labels.get(str(text), str(text).replace("_", " "))


def build_markdown(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Residual Base Compatibility Probe Summary",
        "",
        "| Base | Base Kind | Variant | Alpha | Seeds | Acc. | Acc-IMP | Base-IMP | Support-IMP | Added Precision | Oracle Overlap | Base-Ref |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{label(row['base_source'])} | "
            f"{label(row['base_kind'])} | "
            f"{label(row['variant'])} | "
            f"{fmt(row['alpha'])} | "
            f"{row['seeds']} | "
            f"{fmt_ci(row, 'trained_accuracy')} | "
            f"{fmt_ci(row, 'accuracy_minus_imp')} | "
            f"{fmt_ci(row, 'base_to_imp_jaccard')} | "
            f"{fmt_ci(row, 'mask_to_imp_jaccard')} | "
            f"{fmt_ci(row, 'added_imp_only_precision')} | "
            f"{fmt_ci(row, 'added_oracle_overlap_precision')} | "
            f"{fmt_ci(row, 'base_to_reference_jaccard')} |"
        )
    lines.append("")
    by_base: dict[object, dict[tuple[object, object], dict[str, object]]] = defaultdict(dict)
    for row in summary:
        by_base[row["base_source"]][(row["base_kind"], row["variant"])] = row
    lines.append("Interpretation:")
    lines.append("")
    for base, variants in sorted(by_base.items()):
        traj_base = variants.get(("trajectory", "base"))
        traj_oracle = variants.get(("trajectory", "oracle_imp_residual"))
        random_base = variants.get(("imp_overlap_random", "base"))
        random_oracle = variants.get(("imp_overlap_random", "oracle_imp_residual"))
        posterior_imp = variants.get(("imp_overlap_random", "posterior_imp_only_residual"))
        dense_imp = variants.get(("imp_overlap_random", "dense_imp_only_residual"))
        posterior_excess_imp = variants.get(
            ("imp_overlap_random", "posterior_excess_imp_only_residual")
        )
        posterior_std_imp = variants.get(
            ("imp_overlap_random", "posterior_std_imp_only_residual")
        )
        random_imp = variants.get(("imp_overlap_random", "random_imp_only_residual"))
        low_imp = variants.get(("imp_overlap_random", "low_imp_only_residual"))
        if random_base is None or random_oracle is None:
            continue
        message = "- " f"{label(base)}: "
        if traj_base is not None and traj_oracle is not None:
            message += (
                "trajectory base/oracle Acc "
                f"{fmt_ci(traj_base, 'trained_accuracy')}/"
                f"{fmt_ci(traj_oracle, 'trained_accuracy')}; "
            )
        message += (
            "IMP-overlap-random base/oracle Acc "
            f"{fmt_ci(random_base, 'trained_accuracy')}/"
            f"{fmt_ci(random_oracle, 'trained_accuracy')}. "
            "IMP-overlap-random Base-IMP "
            f"{fmt_ci(random_base, 'base_to_imp_jaccard')}."
        )
        if random_imp is not None:
            message += (
                " Random IMP-only Acc "
                f"{fmt_ci(random_imp, 'trained_accuracy')}"
            )
        if posterior_imp is not None:
            message += (
                "; posterior IMP-only Acc "
                f"{fmt_ci(posterior_imp, 'trained_accuracy')}"
                " with oracle-overlap "
                f"{fmt_ci(posterior_imp, 'added_oracle_overlap_precision')}"
            )
        if dense_imp is not None:
            message += (
                "; dense IMP-only Acc "
                f"{fmt_ci(dense_imp, 'trained_accuracy')}"
                " with oracle-overlap "
                f"{fmt_ci(dense_imp, 'added_oracle_overlap_precision')}"
            )
        if posterior_excess_imp is not None:
            message += (
                "; posterior excess IMP-only Acc "
                f"{fmt_ci(posterior_excess_imp, 'trained_accuracy')}"
                " with oracle-overlap "
                f"{fmt_ci(posterior_excess_imp, 'added_oracle_overlap_precision')}"
            )
        if posterior_std_imp is not None:
            message += (
                "; posterior std IMP-only Acc "
                f"{fmt_ci(posterior_std_imp, 'trained_accuracy')}"
                " with oracle-overlap "
                f"{fmt_ci(posterior_std_imp, 'added_oracle_overlap_precision')}"
            )
        if low_imp is not None:
            message += (
                "; low IMP-only Acc "
                f"{fmt_ci(low_imp, 'trained_accuracy')}."
            )
        lines.append(message)
    lines.append("")
    lines.append(
        "This file is generated by `scripts/summarize_residual_base_compatibility_probe.py`."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_root)
    summary = summarize(rows)
    write_csv(args.out_csv, summary)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_markdown(summary) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "groups": len(summary),
                "out_csv": str(args.out_csv),
                "out_md": str(args.out_md),
            }
        )
    )


if __name__ == "__main__":
    main()
