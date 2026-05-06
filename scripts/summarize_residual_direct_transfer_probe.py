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
    "candidate_count",
    "source_vote_positive_count",
    "source_vote_max",
    "selected_source_vote_mean",
    "selected_source_vote_positive_fraction",
    "alignment_mean_corr",
    "alignment_min_corr",
    "added_imp_only_hits",
    "added_imp_only_precision",
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
        raise SystemExit(f"no residual direct transfer metrics found under {run_root}")
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
            "targets": len({row["target_seed"] for row in group}),
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
        "target_oracle_residual": "target oracle",
        "source_vote_residual": "source vote",
        "source_vote_random_residual": "source-vote random",
        "aligned_source_vote_residual": "aligned source vote",
        "aligned_source_vote_random_residual": "aligned-vote random",
        "target_random_residual": "target random",
    }
    return labels.get(str(text), str(text).replace("_", " "))


def build_markdown(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Residual Direct Cross-Seed Transfer Probe Summary",
        "",
        "| Base | Variant | Alpha | Targets | Acc. | Acc-IMP | Support-IMP | Added Precision | Vote+ Frac. | Vote Mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{label(row['base_source'])} | "
            f"{label(row['variant'])} | "
            f"{fmt(row['alpha'])} | "
            f"{row['targets']} | "
            f"{fmt_ci(row, 'trained_accuracy')} | "
            f"{fmt_ci(row, 'accuracy_minus_imp')} | "
            f"{fmt_ci(row, 'mask_to_imp_jaccard')} | "
            f"{fmt_ci(row, 'added_imp_only_precision')} | "
            f"{fmt_ci(row, 'selected_source_vote_positive_fraction')} | "
            f"{fmt_ci(row, 'selected_source_vote_mean')} |"
        )
    lines.append("")
    by_base: dict[object, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary:
        by_base[row["base_source"]][row["variant"]] = row
    lines.append("Interpretation:")
    lines.append("")
    for base, variants in sorted(by_base.items()):
        oracle = variants.get("target_oracle_residual")
        vote = variants.get("source_vote_residual")
        vote_random = variants.get("source_vote_random_residual")
        aligned_vote = variants.get("aligned_source_vote_residual")
        aligned_vote_random = variants.get("aligned_source_vote_random_residual")
        target_random = variants.get("target_random_residual")
        if oracle is None or vote is None or target_random is None:
            continue
        parts = [
            f"- {label(base)}:",
            f"source-vote Acc {fmt_ci(vote, 'trained_accuracy')},",
        ]
        if vote_random is not None:
            parts.append(f"source-vote-random Acc {fmt_ci(vote_random, 'trained_accuracy')},")
        if aligned_vote is not None:
            parts.append(f"aligned-source-vote Acc {fmt_ci(aligned_vote, 'trained_accuracy')},")
            parts.append(
                f"aligned precision {fmt_ci(aligned_vote, 'added_imp_only_precision')},"
            )
        if aligned_vote_random is not None:
            parts.append(
                f"aligned-vote-random Acc {fmt_ci(aligned_vote_random, 'trained_accuracy')},"
            )
        parts.append(f"target-random Acc {fmt_ci(target_random, 'trained_accuracy')},")
        parts.append(f"target-oracle Acc {fmt_ci(oracle, 'trained_accuracy')};")
        parts.append(
            f"source-vote added precision {fmt_ci(vote, 'added_imp_only_precision')}."
        )
        lines.append(" ".join(parts))
    lines.append("")
    lines.append("This file is generated by `scripts/summarize_residual_direct_transfer_probe.py`.")
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
