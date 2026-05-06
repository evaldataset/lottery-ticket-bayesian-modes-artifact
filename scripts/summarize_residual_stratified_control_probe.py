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
    "mask_to_imp_jaccard",
    "mask_to_base_jaccard",
    "swap_count",
    "added_imp_only_hits",
    "added_imp_only_precision",
    "added_oracle_overlap_hits",
    "added_oracle_overlap_precision",
    "stratum_exact_fraction",
    "stratum_primary_relaxed_count",
    "stratum_global_relaxed_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_rows(run_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_root.glob("**/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload["rows"]:
            out = dict(row)
            out["seed"] = payload["seed"]
            out["run_path"] = str(path)
            rows.append(out)
    if not rows:
        raise SystemExit(f"no residual stratified control metrics found under {run_root}")
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
    buckets: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(row["base_source"], row["variant"], row["alpha"])].append(row)
    out = []
    for (base_source, variant, alpha), group in sorted(buckets.items()):
        summary: dict[str, object] = {
            "base_source": base_source,
            "variant": variant,
            "alpha": alpha,
            "runs": len({row["seed"] for row in group}),
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
        "oracle_top_imp_residual": "oracle top IMP",
        "random_imp_only_residual": "random IMP-only",
        "random_nonimp_global_residual": "global non-IMP",
        "random_nonimp_parameter_matched_residual": "parameter non-IMP",
        "random_nonimp_parameter_score_matched_residual": "parameter+score non-IMP",
    }
    return labels.get(str(text), str(text).replace("_", " "))


def build_markdown(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Residual Stratified Control Probe Summary",
        "",
        "| Base | Variant | Alpha | Runs | Acc. | Acc-IMP | Support-IMP | IMP precision | Oracle overlap | Exact strata |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{label(row['base_source'])} | "
            f"{label(row['variant'])} | "
            f"{fmt(row['alpha'])} | "
            f"{row['runs']} | "
            f"{fmt_ci(row, 'trained_accuracy')} | "
            f"{fmt_ci(row, 'accuracy_minus_imp')} | "
            f"{fmt_ci(row, 'mask_to_imp_jaccard')} | "
            f"{fmt_ci(row, 'added_imp_only_precision')} | "
            f"{fmt_ci(row, 'added_oracle_overlap_precision')} | "
            f"{fmt_ci(row, 'stratum_exact_fraction')} |"
        )
    lines.append("")
    by_base: dict[object, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary:
        if str(row["alpha"]) == "0.0":
            by_base[row["base_source"]]["base"] = row
        else:
            by_base[row["base_source"]][row["variant"]] = row
    lines.append("Interpretation:")
    lines.append("")
    for base, variants in sorted(by_base.items()):
        base_row = variants.get("base")
        oracle = variants.get("oracle_top_imp_residual")
        imp_random = variants.get("random_imp_only_residual")
        global_random = variants.get("random_nonimp_global_residual")
        param_random = variants.get("random_nonimp_parameter_matched_residual")
        param_score_random = variants.get("random_nonimp_parameter_score_matched_residual")
        if oracle is None:
            continue
        parts = [f"- {label(base)}:"]
        if base_row is not None:
            parts.append(f"base Acc {fmt_ci(base_row, 'trained_accuracy')};")
        parts.append(f"oracle Acc {fmt_ci(oracle, 'trained_accuracy')};")
        if imp_random is not None:
            parts.append(f"random IMP-only Acc {fmt_ci(imp_random, 'trained_accuracy')};")
        if global_random is not None:
            parts.append(f"global non-IMP Acc {fmt_ci(global_random, 'trained_accuracy')};")
        if param_random is not None:
            parts.append(f"parameter non-IMP Acc {fmt_ci(param_random, 'trained_accuracy')};")
        if param_score_random is not None:
            parts.append(
                "parameter+score non-IMP Acc "
                f"{fmt_ci(param_score_random, 'trained_accuracy')} "
                f"(exact strata {fmt_ci(param_score_random, 'stratum_exact_fraction')})."
            )
        lines.append(" ".join(parts))
    lines.append("")
    lines.append("This file is generated by `scripts/summarize_residual_stratified_control_probe.py`.")
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
