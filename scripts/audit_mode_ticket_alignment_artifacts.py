#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "mode_ticket_alignment_artifact_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "mode_ticket_alignment_artifact_audit.md"

RUN_ROOTS = [
    (
        "CIFAR SGLD direct",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3",
    ),
    (
        "CIFAR activation-aligned SGLD",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3",
    ),
    (
        "CIFAR weight-aligned SGLD",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3",
    ),
    (
        "CIFAR dense-start cSGLD multi-chain",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3",
    ),
    (
        "CIFAR independent-start cSGLD multi-chain",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3",
    ),
    (
        "CIFAR LowRank128 Laplace direct",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3",
    ),
    (
        "CIFAR JointDiagLap270k direct",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3",
    ),
]

RAW_MASK_SUFFIXES = {
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-06")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def latest_metrics_path(run_root: Path) -> Path | None:
    metrics = sorted(run_root.glob("*/metrics.json"))
    if not metrics:
        return None
    return metrics[-1]


def summary_path_for(run_root: Path) -> Path:
    return run_root.with_name(f"{run_root.name}_summary.csv")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_direct_sample_row(rows: list[dict[str, str]], method: str) -> dict[str, str] | None:
    preferred_by_method = {
        "activation": "activation_aligned_posterior_samples_vs_tickets",
        "weight": "weight_aligned_posterior_samples_vs_tickets",
    }
    preferred = preferred_by_method.get(method, "posterior_samples_vs_tickets")
    for row in rows:
        if row.get("comparison") == preferred:
            return row
    for row in rows:
        if row.get("comparison") == "posterior_samples_vs_tickets":
            return row
    return None


def summarize_alignment_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source", "unknown"))].append(record)
    source_summaries = {}
    for source, source_records in sorted(by_source.items()):
        mean_corr = [
            value
            for value in (to_float(record.get("alignment_mean_corr")) for record in source_records)
            if value is not None
        ]
        min_corr = [
            value
            for value in (to_float(record.get("alignment_min_corr")) for record in source_records)
            if value is not None
        ]
        seeds = sorted(
            {
                int(float(record["seed"]))
                for record in source_records
                if to_float(record.get("seed")) is not None
            }
        )
        source_summaries[source] = {
            "record_count": len(source_records),
            "seed_count": len(seeds),
            "seeds": seeds,
            "mean_corr_mean": statistics.fmean(mean_corr) if mean_corr else None,
            "mean_corr_min": min(mean_corr) if mean_corr else None,
            "mean_corr_max": max(mean_corr) if mean_corr else None,
            "min_corr_mean": statistics.fmean(min_corr) if min_corr else None,
            "min_corr_min": min(min_corr) if min_corr else None,
        }
    return {
        "record_count": len(records),
        "source_counts": dict(sorted(Counter(str(r.get("source", "unknown")) for r in records).items())),
        "by_source": source_summaries,
    }


def file_inventory(run_root: Path) -> dict[str, Any]:
    files = sorted(path for path in run_root.rglob("*") if path.is_file())
    suffix_counts = Counter(path.suffix or "<none>" for path in files)
    raw_files = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.suffix in RAW_MASK_SUFFIXES
    ]
    return {
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "raw_mask_or_state_files": raw_files,
        "raw_mask_or_state_file_count": len(raw_files),
    }


def row_metric(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    return to_float(row.get(key))


def row_bool(row: dict[str, str] | None, key: str) -> bool | None:
    if row is None:
        return None
    return to_bool(row.get(key))


def compare_row_payload(row: dict[str, str] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    passes_layer_ks = row_bool(row, "passes_layer_ks")
    passes_hamming = row_bool(row, "passes_hamming_overlap")
    passes_logit_cka = row_bool(row, "passes_logit_cka")
    passes_activation_cka = row_bool(row, "passes_activation_cka")
    cluster_count = row_metric(row, "posterior_num_clusters")
    ticket_count = row_metric(row, "right_count")
    mask_distribution_pass = bool(passes_layer_ks and passes_hamming)
    basin_count_matches_tickets = (
        cluster_count is not None
        and ticket_count is not None
        and int(round(cluster_count)) == int(round(ticket_count))
    )
    return {
        "comparison": row.get("comparison"),
        "left_count": row_metric(row, "left_count"),
        "right_count": ticket_count,
        "posterior_num_clusters": cluster_count,
        "posterior_effective_cluster_count": row_metric(
            row, "posterior_effective_cluster_count"
        ),
        "layer_ks_pvalue": row_metric(row, "layer_ks_pvalue"),
        "hamming_overlap": row_metric(row, "hamming_overlap"),
        "hamming_cross_mean": row_metric(row, "hamming_cross_mean"),
        "logit_cka_hungarian_mean": row_metric(row, "logit_cka_hungarian_mean"),
        "activation_cka_hungarian_mean": row_metric(
            row, "activation_cka_hungarian_mean"
        ),
        "passes_layer_ks": passes_layer_ks,
        "passes_hamming_overlap": passes_hamming,
        "passes_logit_cka": passes_logit_cka,
        "passes_activation_cka": passes_activation_cka,
        "mask_distribution_pass": mask_distribution_pass,
        "basin_count_matches_tickets": basin_count_matches_tickets,
        "direct_equivalence_pass": bool(
            mask_distribution_pass and basin_count_matches_tickets
        ),
    }


def build_run_payload(label: str, run_root: Path) -> dict[str, Any]:
    metrics_path = latest_metrics_path(run_root)
    summary_path = summary_path_for(run_root)
    inventory = file_inventory(run_root) if run_root.exists() else {
        "file_count": 0,
        "suffix_counts": {},
        "raw_mask_or_state_files": [],
        "raw_mask_or_state_file_count": 0,
    }
    if metrics_path is None:
        return {
            "label": label,
            "run_root": run_root.relative_to(ROOT).as_posix(),
            "exists": run_root.exists(),
            "metrics_path": None,
            "summary_path": summary_path.relative_to(ROOT).as_posix(),
            "summary_exists": summary_path.exists(),
            "file_inventory": inventory,
            "missing": True,
        }

    payload = load_json(metrics_path)
    config = payload.get("config", {})
    alignment = payload.get("alignment", {})
    records = alignment.get("records", [])
    rows = read_summary_rows(summary_path)
    alignment_method = str(alignment.get("method") or config.get("alignment_method") or "none")
    sample_row = select_direct_sample_row(rows, alignment_method)
    seeds = config.get("seeds", [])
    target_seed = alignment.get("target_seed")
    target_frame_count = 1 if target_seed is not None else 0
    target_frame_fraction = (
        target_frame_count / len(seeds)
        if isinstance(seeds, list) and seeds
        else None
    )
    return {
        "label": label,
        "run_root": run_root.relative_to(ROOT).as_posix(),
        "exists": True,
        "run_dir": metrics_path.parent.relative_to(ROOT).as_posix(),
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "summary_exists": summary_path.exists(),
        "dataset": config.get("dataset"),
        "model": config.get("model"),
        "posterior_sampler": config.get("posterior_sampler"),
        "seeds": seeds,
        "samples_per_seed": config.get("samples_per_seed"),
        "posterior_chains": config.get("posterior_chains"),
        "alignment": {
            "method": alignment_method,
            "target": alignment.get("target"),
            "target_seed": target_seed,
            "target_frame_count": target_frame_count,
            "target_frame_fraction": target_frame_fraction,
            **summarize_alignment_records(records),
        },
        "direct_sample_row": compare_row_payload(sample_row),
        "file_inventory": inventory,
        "missing": False,
    }


def sci(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0.0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def fixed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def pass_fail(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "pass" if value else "fail"


def build_overall(runs: list[dict[str, Any]]) -> dict[str, Any]:
    present = [run for run in runs if not run.get("missing")]
    direct_rows = [
        row
        for row in (run.get("direct_sample_row") for run in present)
        if isinstance(row, dict)
    ]
    aligned = [
        run
        for run in present
        if run.get("alignment", {}).get("method") in {"activation", "weight"}
    ]
    aligned_rows = [
        row
        for row in (run.get("direct_sample_row") for run in aligned)
        if isinstance(row, dict)
    ]
    raw_count = sum(
        int(run.get("file_inventory", {}).get("raw_mask_or_state_file_count", 0))
        for run in present
    )
    posthoc_supported = raw_count > 0
    return {
        "run_count": len(present),
        "direct_sample_row_count": len(direct_rows),
        "aligned_run_count": len(aligned),
        "aligned_sample_row_count": len(aligned_rows),
        "aligned_rows_all_fail_layer_ks": all(
            row.get("passes_layer_ks") is False for row in aligned_rows
        )
        if aligned_rows
        else False,
        "aligned_rows_all_fail_hamming_overlap": all(
            row.get("passes_hamming_overlap") is False for row in aligned_rows
        )
        if aligned_rows
        else False,
        "any_direct_equivalence_pass": any(
            bool(row.get("direct_equivalence_pass")) for row in direct_rows
        ),
        "direct_rows_all_collapse_to_one_basin": all(
            int(round(float(row.get("posterior_num_clusters", math.nan)))) == 1
            for row in direct_rows
            if row.get("posterior_num_clusters") is not None
        ),
        "direct_rows_failing_layer_ks": sum(
            row.get("passes_layer_ks") is False for row in direct_rows
        ),
        "direct_rows_failing_hamming_overlap": sum(
            row.get("passes_hamming_overlap") is False for row in direct_rows
        ),
        "raw_mask_artifacts_present": raw_count > 0,
        "raw_mask_or_state_file_count": raw_count,
        "posthoc_exhaustive_permutation_supported": posthoc_supported,
        "posthoc_limitation_statement": (
            "post-hoc exhaustive graph/permutation realignment is not supported "
            "by the current direct-run artifacts"
        ),
        "future_closure_requires": [
            "saving raw posterior, chain-start, and ticket masks or states",
            "rerunning direct probes with alternative target frames",
            "adding an exhaustive graph/permutation matching objective over saved masks",
        ],
    }


def write_markdown(path: Path, *, date: str, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    if overall["posthoc_exhaustive_permutation_supported"]:
        posthoc_sentence = (
            "post-hoc exhaustive graph/permutation realignment has raw "
            "mask/state artifacts available in the audited run roots."
        )
    else:
        posthoc_sentence = (
            "post-hoc exhaustive graph/permutation realignment is not supported by "
            "the current direct-run artifacts."
        )
    lines = [
        "# Mode/Ticket Alignment Artifact Audit",
        "",
        f"Date: {date}",
        "",
        "This audit reuses the existing full-data CIFAR direct mode/ticket run",
        "artifacts. It checks whether the current activation-channel and",
        "weight-correlation alignment rows rescue the proposal-level metrics,",
        "and whether the saved artifacts are sufficient for stronger post-hoc",
        "graph/permutation realignment.",
        "",
        "Conclusion: first-order channel alignment remains negative.",
        posthoc_sentence,
        "The existing release contains summary CSV/JSON, layer-sparsity vectors,",
        "and CKA tables; current full-data direct CIFAR roots have no saved raw",
        "posterior/ticket mask or state tensors unless rerun with explicit mask",
        "artifact saving enabled.",
        "",
        "## Direct Sample Rows",
        "",
        "| Run | Align | Sampler | Samples | Target seed | Layer KS p | Hamming overlap | Clusters | Layer KS | Hamming | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in payload["runs"]:
        row = run.get("direct_sample_row")
        if not isinstance(row, dict):
            continue
        alignment = run.get("alignment", {})
        verdict = "reject equivalence"
        if row.get("direct_equivalence_pass"):
            verdict = "passes direct checks"
        target = alignment.get("target_seed")
        target_text = "n/a" if target is None else str(target)
        lines.append(
            "| "
            f"{run['label']} | "
            f"{alignment.get('method', 'none')} | "
            f"{run.get('posterior_sampler', 'n/a')} | "
            f"{int(row.get('left_count') or 0)} | "
            f"{target_text} | "
            f"{sci(row.get('layer_ks_pvalue'))} | "
            f"{fixed(row.get('hamming_overlap'))} | "
            f"{fixed(row.get('posterior_num_clusters'), 1)} | "
            f"{pass_fail(row.get('passes_layer_ks'))} | "
            f"{pass_fail(row.get('passes_hamming_overlap'))} | "
            f"{verdict} |"
        )

    lines.extend(
        [
            "",
            "## Alignment Quality",
            "",
            "| Run | Source | Records | Seeds | Mean corr | Minimum corr | Target frame coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in payload["runs"]:
        alignment = run.get("alignment", {})
        if alignment.get("method") not in {"activation", "weight"}:
            continue
        coverage = alignment.get("target_frame_fraction")
        by_source = alignment.get("by_source", {})
        for source, summary in by_source.items():
            lines.append(
                "| "
                f"{run['label']} | "
                f"{source} | "
                f"{summary.get('record_count', 0)} | "
                f"{summary.get('seed_count', 0)} | "
                f"{fixed(summary.get('mean_corr_mean'))} | "
                f"{fixed(summary.get('min_corr_min'))} | "
                f"{fixed(coverage)} |"
            )

    lines.extend(
        [
            "",
            "## Artifact Inventory",
            "",
            "| Run | Files | Suffixes | Raw mask/state files | Post-hoc exhaustive permutation |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )
    for run in payload["runs"]:
        inventory = run.get("file_inventory", {})
        suffixes = ", ".join(
            f"{suffix}:{count}"
            for suffix, count in inventory.get("suffix_counts", {}).items()
        )
        raw_count = int(inventory.get("raw_mask_or_state_file_count", 0))
        supported = "supported" if raw_count else "not supported"
        lines.append(
            "| "
            f"{run['label']} | "
            f"{inventory.get('file_count', 0)} | "
            f"{suffixes or 'none'} | "
            f"{raw_count} | "
            f"{supported} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Aligned sample rows all fail layer KS: `{overall['aligned_rows_all_fail_layer_ks']}`.",
            f"- Aligned sample rows all fail Hamming overlap: `{overall['aligned_rows_all_fail_hamming_overlap']}`.",
            f"- Any direct row passes full direct equivalence: `{overall['any_direct_equivalence_pass']}`.",
            f"- Direct rows collapsing to one posterior basin: `{overall['direct_rows_all_collapse_to_one_basin']}`.",
            f"- Raw mask/state artifacts present: `{overall['raw_mask_artifacts_present']}`.",
            "",
            "The activation and weight-correlation rows are useful first-order",
            "permutation checks because they map ResNet channels into a common",
            "seed-0 frame before evaluating proposal-level mask metrics. They do",
            "not exhaust all graph-isomorphism or target-frame choices. Closing",
            "that remaining reviewer-facing gap requires rerunning direct probes",
            "with `--save-mask-artifacts`, optional `--save-state-artifacts`,",
            "and alternative target frames. The saved artifacts can then be",
            "checked with `scripts/audit_mask_artifact_posthoc_matching.py`,",
            "which currently validates record-level and local channel matching",
            "on the fake-CIFAR fixture but not exhaustive full-data graph/channel",
            "permutation.",
            "",
            "This file is generated by",
            "`scripts/audit_mode_ticket_alignment_artifacts.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = [build_run_payload(label, run_root) for label, run_root in RUN_ROOTS]
    payload = {
        "date": args.date,
        "runs": runs,
        "overall": build_overall(runs),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, date=args.date, payload=payload)
    print(
        json.dumps(
            {
                "out_json": args.out_json.relative_to(ROOT).as_posix(),
                "out_md": args.out_md.relative_to(ROOT).as_posix(),
                "runs": payload["overall"]["run_count"],
                "posthoc_exhaustive_permutation_supported": payload["overall"][
                    "posthoc_exhaustive_permutation_supported"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
