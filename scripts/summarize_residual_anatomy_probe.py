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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_payloads(run_root: Path) -> list[dict]:
    payloads = []
    for path in sorted(run_root.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = str(path)
        payloads.append(payload)
    if not payloads:
        raise SystemExit(f"no residual anatomy metrics found under {run_root}")
    return payloads


def collect(payloads: Iterable[dict], key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        rows.extend(payload.get(key, []))
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


def stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
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


def aggregate(
    rows: list[dict[str, object]],
    group_keys: list[str],
    value_keys: list[str],
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in group_keys)].append(row)
    out = []
    for key, group in sorted(buckets.items()):
        row = {name: value for name, value in zip(group_keys, key, strict=True)}
        row["runs"] = len({int(g["seed"]) for g in group if "seed" in g})
        row["rows"] = len(group)
        for value_key in value_keys:
            summary = stats([to_float(g.get(value_key)) for g in group])
            row[f"{value_key}_mean"] = summary["mean"]
            row[f"{value_key}_std"] = summary["std"]
            row[f"{value_key}_ci95"] = summary["ci95"]
            row[f"{value_key}_n"] = summary["n"]
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def best_rows(
    rows: list[dict[str, object]],
    group_key: str,
    score_key: str,
    *,
    exclude_features: set[str] | None = None,
) -> list[dict[str, object]]:
    best: dict[object, dict[str, object]] = {}
    for row in rows:
        if exclude_features and str(row.get("feature")) in exclude_features:
            continue
        score = to_float(row.get(score_key))
        if score is None:
            continue
        group = row[group_key]
        previous = best.get(group)
        if previous is None or score > to_float(previous.get(score_key)):
            best[group] = row
    return [best[key] for key in sorted(best)]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def build_markdown(
    *,
    payloads: list[dict],
    global_summary: list[dict[str, object]],
    group_summary: list[dict[str, object]],
    round_summary: list[dict[str, object]],
    score_summary: list[dict[str, object]],
    predictor_summary: list[dict[str, object]],
) -> str:
    lines = ["# Residual Anatomy Probe Summary", ""]
    lines.append(f"Runs: {len(payloads)}")
    first = payloads[0]
    training = first.get("training", {})
    lines.append(
        "Setting: "
        f"{first.get('dataset')} {first.get('model')}, "
        f"epochs={training.get('epochs')}, "
        f"rewind={training.get('rewind_epochs')}, "
        f"IMP rounds={training.get('imp_rounds')}, "
        f"prune_fraction={training.get('prune_fraction')}."
    )
    lines.append("")

    lines.append("## Global Residual Size")
    lines.append(
        markdown_table(
            [
                "Base",
                "Runs",
                "Jaccard-IMP",
                "IMP-only",
                "Base-only",
                "Base-only prune round",
            ],
            [
                [
                    str(row["base_source"]),
                    str(row["runs"]),
                    fmt_ci(row, "jaccard"),
                    fmt_ci(row, "imp_only", 1),
                    fmt_ci(row, "base_only", 1),
                    fmt_ci(row, "base_only_pruned_round_mean"),
                ]
                for row in global_summary
            ],
        )
    )
    lines.append("")

    lines.append("## Group Concentration")
    group_rows = [
        row
        for row in group_summary
        if row["unit"] in {"stem", "stage1", "stage2", "stage3", "head", "features", "mlp"}
    ]
    lines.append(
        markdown_table(
            [
                "Base",
                "Group",
                "IMP-only share",
                "IMP-only enrichment",
                "Base-only share",
                "Base-only prune round",
            ],
            [
                [
                    str(row["base_source"]),
                    str(row["unit"]),
                    fmt_ci(row, "imp_only_share"),
                    fmt_ci(row, "imp_only_enrichment"),
                    fmt_ci(row, "base_only_share"),
                    fmt_ci(row, "base_only_pruned_round_mean"),
                ]
                for row in group_rows
            ],
        )
    )
    lines.append("")

    lines.append("## IMP Pruning Round For Base-only Weights")
    all_round_rows = [row for row in round_summary if row["group"] == "all"]
    lines.append(
        markdown_table(
            ["Base", "Round", "Fraction", "Count"],
            [
                [
                    str(row["base_source"]),
                    str(row["pruned_round"]),
                    fmt_ci(row, "fraction"),
                    fmt_ci(row, "base_only_pruned", 1),
                ]
                for row in all_round_rows
            ],
        )
    )
    lines.append("")

    lines.append("## Best Univariate Residual Predictors")
    best_auc = best_rows(
        score_summary,
        "base_source",
        "auc_imp_only_vs_nonbase_mean",
        exclude_features=set(),
    )
    best_lift = best_rows(score_summary, "base_source", "topk_lift_mean", exclude_features=set())
    best_by_base = {row["base_source"]: {"auc": row} for row in best_auc}
    for row in best_lift:
        best_by_base.setdefault(row["base_source"], {})["lift"] = row
    lines.append(
        markdown_table(
            ["Base", "Best AUC feature", "AUC", "Best lift feature", "Top-k recall", "Lift"],
            [
                [
                    str(base),
                    str(bundle.get("auc", {}).get("feature", "")),
                    fmt_ci(bundle.get("auc", {}), "auc_imp_only_vs_nonbase"),
                    str(bundle.get("lift", {}).get("feature", "")),
                    fmt_ci(bundle.get("lift", {}), "topk_recall"),
                    fmt_ci(bundle.get("lift", {}), "topk_lift"),
                ]
                for base, bundle in sorted(best_by_base.items())
            ],
        )
    )
    lines.append("")

    lines.append("## Learned Residual Predictor")
    lines.append(
        markdown_table(
            ["Base", "Runs", "AUC", "Top-k recall", "Precision", "Baseline", "Lift"],
            [
                [
                    str(row["base_source"]),
                    str(row["runs"]),
                    fmt_ci(row, "test_auc"),
                    fmt_ci(row, "test_topk_recall"),
                    fmt_ci(row, "test_topk_precision"),
                    fmt_ci(row, "test_baseline_precision"),
                    fmt_ci(row, "test_topk_lift"),
                ]
                for row in predictor_summary
            ],
        )
    )
    lines.append("")

    focus = next((row for row in group_summary if row["base_source"] == "traj_rms_abs"), None)
    if focus is not None:
        top_group = max(
            [row for row in group_summary if row["base_source"] == "traj_rms_abs"],
            key=lambda row: to_float(row.get("imp_only_enrichment_mean")) or -1.0,
        )
        lines.append("Interpretation:")
        lines.append("")
        lines.append(
            "- For the `traj_rms_abs` base, the most enriched IMP-only residual "
            f"group is `{top_group['unit']}` with enrichment "
            f"{fmt_ci(top_group, 'imp_only_enrichment')}."
        )
        pred = next(
            (row for row in predictor_summary if row["base_source"] == "traj_rms_abs"),
            None,
        )
        if pred is not None:
            lines.append(
                "- A rank-feature logistic residual predictor reaches "
                f"AUC {fmt_ci(pred, 'test_auc')} and top-k recall "
                f"{fmt_ci(pred, 'test_topk_recall')} for `traj_rms_abs`, "
                f"versus baseline precision {fmt_ci(pred, 'test_baseline_precision')}."
            )
    lines.append("")
    lines.append(
        "This file is generated by `scripts/summarize_residual_anatomy_probe.py`."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payloads = load_payloads(args.run_root)
    global_rows = collect(payloads, "global_rows")
    group_rows = collect(payloads, "group_rows")
    layer_rows = collect(payloads, "layer_rows")
    round_rows = collect(payloads, "round_rows")
    score_rows = collect(payloads, "score_rows")
    predictor_rows = collect(payloads, "predictor_rows")
    coefficient_rows = collect(payloads, "coefficient_rows")

    global_summary = aggregate(
        global_rows,
        ["base_source"],
        [
            "jaccard",
            "imp_only",
            "base_only",
            "base_only_pruned_round_mean",
            "dense_accuracy",
            "imp_accuracy",
        ],
    )
    group_summary = aggregate(
        group_rows,
        ["base_source", "unit"],
        [
            "imp_only_share",
            "base_only_share",
            "imp_only_enrichment",
            "base_only_enrichment",
            "imp_only_density",
            "base_only_density",
            "base_only_pruned_round_mean",
        ],
    )
    layer_summary = aggregate(
        layer_rows,
        ["base_source", "unit"],
        [
            "imp_only_share",
            "base_only_share",
            "imp_only_enrichment",
            "base_only_enrichment",
            "imp_only_density",
            "base_only_density",
            "base_only_pruned_round_mean",
        ],
    )
    round_summary = aggregate(
        round_rows,
        ["base_source", "group", "pruned_round"],
        ["fraction", "base_only_pruned"],
    )
    score_summary = aggregate(
        score_rows,
        ["base_source", "feature"],
        [
            "auc_imp_only_vs_nonbase",
            "topk_recall",
            "topk_precision",
            "baseline_precision",
            "topk_lift",
            "imp_only_mean",
            "base_only_mean",
            "shared_mean",
            "neither_mean",
        ],
    )
    predictor_summary = aggregate(
        predictor_rows,
        ["base_source"],
        [
            "test_auc",
            "test_topk_recall",
            "test_topk_precision",
            "test_baseline_precision",
            "test_topk_lift",
            "test_loss",
        ],
    )
    coefficient_summary = aggregate(
        coefficient_rows,
        ["base_source", "feature"],
        ["coefficient"],
    )

    prefix = args.out_prefix
    write_csv(prefix.with_name(prefix.name + "_global.csv"), global_summary)
    write_csv(prefix.with_name(prefix.name + "_group.csv"), group_summary)
    write_csv(prefix.with_name(prefix.name + "_layer.csv"), layer_summary)
    write_csv(prefix.with_name(prefix.name + "_round.csv"), round_summary)
    write_csv(prefix.with_name(prefix.name + "_score.csv"), score_summary)
    write_csv(prefix.with_name(prefix.name + "_predictor.csv"), predictor_summary)
    write_csv(prefix.with_name(prefix.name + "_coefficients.csv"), coefficient_summary)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_markdown(
            payloads=payloads,
            global_summary=global_summary,
            group_summary=group_summary,
            round_summary=round_summary,
            score_summary=score_summary,
            predictor_summary=predictor_summary,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(payloads), "out_prefix": str(prefix), "out_md": str(args.out_md)}))


if __name__ == "__main__":
    main()
