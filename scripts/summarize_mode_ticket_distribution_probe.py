#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/digits_mlp_mode_ticket_distribution"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/digits_mlp_mode_ticket_distribution.md"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("runs/digits_mlp_mode_ticket_distribution_summary.csv"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def latest_metrics(run_root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(run_root.glob("*/metrics.json"))
    if not paths:
        raise SystemExit(f"No metrics.json files found under {run_root}")
    path = paths[-1]
    return path, read_json(path)


def clustering_entropy(clustering: dict[str, Any]) -> dict[str, Any]:
    present_keys = {
        "entropy_nats",
        "normalized_entropy",
        "effective_cluster_count",
    }
    if present_keys.issubset(clustering):
        cluster_counts = clustering.get("cluster_counts")
        if cluster_counts is None:
            labels = clustering.get("labels", [])
            cluster_counts = sorted(
                [labels.count(label) for label in set(labels)], reverse=True
            )
        return {
            "entropy_nats": clustering["entropy_nats"],
            "normalized_entropy": clustering["normalized_entropy"],
            "effective_cluster_count": clustering["effective_cluster_count"],
            "cluster_counts": cluster_counts,
        }
    labels = clustering.get("labels", [])
    if not labels:
        return {
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "effective_cluster_count": 0.0,
            "cluster_counts": [],
        }
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    probabilities = [count / len(labels) for count in counts.values()]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    if abs(entropy) < 1e-12:
        entropy = 0.0
    max_entropy = math.log(len(labels)) if len(labels) > 1 else 0.0
    return {
        "entropy_nats": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0.0 else 0.0,
        "effective_cluster_count": math.exp(entropy),
        "cluster_counts": sorted(counts.values(), reverse=True),
    }


def clustering_for_comparison(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    label = comparison.get("label", "")
    alignment_method = metrics.get("config", {}).get("alignment_method", "none")
    alignment_prefix = f"{alignment_method}_aligned_"
    if (
        label.startswith(f"{alignment_prefix}chain_start")
        and metrics.get("aligned_chain_start_clustering") is not None
    ):
        return metrics["aligned_chain_start_clustering"]
    if label.startswith("chain_start") and metrics.get("chain_start_clustering") is not None:
        return metrics["chain_start_clustering"]
    if (
        label.startswith(alignment_prefix)
        and metrics.get("aligned_posterior_clustering") is not None
    ):
        return metrics["aligned_posterior_clustering"]
    return metrics["posterior_clustering"]


def comparison_rows(metrics: dict[str, Any], run_path: Path) -> list[dict[str, Any]]:
    rows = []
    config = metrics["config"]
    for comparison in metrics["comparisons"]:
        clustering = clustering_for_comparison(metrics, comparison)
        cluster_stats = clustering_entropy(clustering)
        layer = comparison["layer_sparsity"]
        hamming = comparison["mask_hamming_distribution"]
        cka = comparison["logit_space_cka"]
        activation_cka = comparison.get("activation_space_cka", {})
        thresholds = comparison["proposal_thresholds"]
        rows.append(
            {
                "run": str(run_path.parent),
                "dataset": config["dataset"],
                "model": config["model"],
                "posterior_sampler": config["posterior_sampler"],
                "comparison": comparison["label"],
                "posterior_num_clusters": clustering["num_clusters"],
                "posterior_largest_cluster_fraction": clustering[
                    "largest_cluster_fraction"
                ],
                "posterior_cluster_entropy_nats": cluster_stats["entropy_nats"],
                "posterior_cluster_entropy_normalized": cluster_stats[
                    "normalized_entropy"
                ],
                "posterior_effective_cluster_count": cluster_stats[
                    "effective_cluster_count"
                ],
                "left_count": comparison["left_count"],
                "right_count": comparison["right_count"],
                "layer_ks_pvalue": layer["ks_pvalue"],
                "layer_mmd_rbf": layer["mmd_rbf"],
                "layer_sliced_wasserstein": layer["sliced_wasserstein"],
                "hamming_overlap": hamming["overlap"],
                "hamming_cross_mean": hamming["cross_mean"],
                "logit_cka_hungarian_mean": cka["hungarian_mean_cka"],
                "hungarian_cost": cka["hungarian_cost"],
                "activation_cka_hungarian_mean": activation_cka.get(
                    "hungarian_mean_cka"
                ),
                "activation_hungarian_cost": activation_cka.get("hungarian_cost"),
                "passes_layer_ks": thresholds["layer_sparsity_ks_pvalue_gt_0p1"],
                "passes_hamming_overlap": thresholds["mask_hamming_overlap_gt_0p7"],
                "passes_logit_cka": thresholds["logit_cka_hungarian_mean_gt_0p85"],
                "passes_hungarian_cost": thresholds["hungarian_cost_lt_0p3"],
                "passes_activation_cka": thresholds.get(
                    "activation_cka_hungarian_mean_gt_0p85"
                ),
                "passes_activation_hungarian_cost": thresholds.get(
                    "activation_hungarian_cost_lt_0p3"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, run_path: Path, metrics: dict[str, Any]) -> None:
    config = metrics["config"]
    clustering = metrics["posterior_clustering"]
    cluster_stats = clustering_entropy(clustering)
    chain_start_clustering = metrics.get("chain_start_clustering")
    chain_start_cluster_stats = (
        clustering_entropy(chain_start_clustering)
        if chain_start_clustering is not None
        else None
    )
    aligned_clustering = metrics.get("aligned_posterior_clustering")
    aligned_cluster_stats = (
        clustering_entropy(aligned_clustering) if aligned_clustering is not None else None
    )
    diagnostics = metrics.get("posterior_chain_diagnostics", {})
    samples_per_chain = config.get("samples_per_chain", config.get("samples_per_seed"))
    posterior_chains = config.get("posterior_chains", 1)
    posterior_chain_init = config.get("posterior_chain_init", "dense")
    alignment_method = config.get("alignment_method", "none")
    alignment_label = {
        "activation": "Activation-aligned",
        "weight": "Weight-aligned",
    }.get(alignment_method, "Aligned")
    has_activation_cka = any(
        "activation_space_cka" in comparison
        for comparison in metrics.get("comparisons", [])
    )
    function_space_line = (
        "as linear CKA over held-out logits and final hidden activations."
        if has_activation_cka
        else "as linear CKA over held-out logits."
    )
    cka_threshold_line = (
        "- Passing the logit/activation CKA and Hungarian thresholds alone is"
        if has_activation_cka
        else "- Passing the logit CKA and Hungarian thresholds alone is"
    )
    lines = [
        "# Mode/Ticket Distribution Probe",
        "",
        "This is a direct check of the proposal-level equivalence criteria.",
        "Posterior samples are converted to magnitude masks at the",
        "matched IMP sparsity; posterior modes are mean-shift representatives",
        "in PCA-reduced parameter space. Function-space similarity is measured",
        function_space_line,
        "",
        f"- Run: `{run_path.parent}`",
        f"- Dataset/model: `{config['dataset']}` / `{config['model']}`",
        f"- Seeds: `{config['seeds']}`; data seed `{config['data_seed']}`",
        f"- Posterior sampler: `{config['posterior_sampler']}` with "
        f"{samples_per_chain} samples per chain, {posterior_chains} chain(s) per seed "
        f"from `{posterior_chain_init}` starts",
        f"- Posterior clusters: {clustering['num_clusters']} "
        f"(largest fraction {fmt(clustering['largest_cluster_fraction'])})",
        f"- Posterior basin entropy: {fmt(cluster_stats['entropy_nats'])} nats; "
        f"normalized {fmt(cluster_stats['normalized_entropy'])}; "
        f"effective clusters {fmt(cluster_stats['effective_cluster_count'])}",
    ]
    if chain_start_clustering is not None and chain_start_cluster_stats is not None:
        lines.extend(
            [
                f"- Chain-start clusters: {chain_start_clustering['num_clusters']} "
                f"(largest fraction "
                f"{fmt(chain_start_clustering['largest_cluster_fraction'])})",
                f"- Posterior-to-chain-start Hamming mean: "
                f"{fmt(diagnostics.get('posterior_to_chain_start_hamming_mean'))}; "
                f"sample accuracy mean "
                f"{fmt(diagnostics.get('posterior_sample_accuracy_mean'))}; "
                f"chain-start accuracy mean "
                f"{fmt(diagnostics.get('chain_start_accuracy_mean'))}",
            ]
        )
    if aligned_clustering is not None and aligned_cluster_stats is not None:
        lines.extend(
            [
                f"- {alignment_label} posterior clusters: "
                f"{aligned_clustering['num_clusters']} "
                f"(largest fraction {fmt(aligned_clustering['largest_cluster_fraction'])})",
                f"- {alignment_label} basin entropy: "
                f"{fmt(aligned_cluster_stats['entropy_nats'])} nats; "
                f"normalized {fmt(aligned_cluster_stats['normalized_entropy'])}; "
                f"effective clusters {fmt(aligned_cluster_stats['effective_cluster_count'])}",
            ]
        )
    lines.extend(
        [
            "",
            "| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | "
            "Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in metrics["comparisons"]:
        layer = comparison["layer_sparsity"]
        hamming = comparison["mask_hamming_distribution"]
        cka = comparison["logit_space_cka"]
        activation_cka = comparison.get("activation_space_cka", {})
        thresholds = comparison["proposal_thresholds"]
        pass_count = sum(1 for value in thresholds.values() if value)
        lines.append(
            f"| {comparison['label']} | {comparison['left_count']} | "
            f"{comparison['right_count']} | {fmt(layer['ks_pvalue'])} | "
            f"{fmt(layer['mmd_rbf'])} | {fmt(layer['sliced_wasserstein'])} | "
            f"{fmt(hamming['overlap'])} | {fmt(cka['hungarian_mean_cka'])} | "
            f"{fmt(activation_cka.get('hungarian_mean_cka'))} | "
            f"{fmt(cka['hungarian_cost'])} | "
            f"{fmt(activation_cka.get('hungarian_cost'))} | "
            f"{pass_count}/{len(thresholds)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `posterior_samples_vs_tickets` tests the raw posterior sample-induced",
            "  mask distribution against IMP tickets.",
            "- `posterior_modes_vs_tickets` first collapses posterior samples to",
            "  mean-shift mode representatives, then compares those representatives",
            "  with IMP tickets.",
            "- `*_aligned_*` rows first map ResNet masks into the first",
            "  seed dense-model channel frame using the configured channel-alignment",
            "  method, then repeat the same distribution checks.",
            cka_threshold_line,
            "  not enough for H1; the proposal also requires mask-distribution agreement. Low",
            "  Hamming-overlap or low KS support therefore counts against the",
            "  strong one-to-one mode/ticket equivalence claim.",
            "",
            "Caveats:",
        ]
    )
    for caveat in metrics.get("caveats", []):
        if caveat == "No activation-channel permutation alignment or basin entropy is applied here.":
            caveat = "No activation-channel permutation alignment is applied here."
        elif caveat == "No channel permutation alignment or basin entropy is applied here.":
            caveat = "No channel permutation alignment is applied here."
        else:
            caveat = caveat.replace(" or basin entropy", "")
        if (
            not has_activation_cka
            and caveat.startswith("Activation comparison uses logit-space")
        ):
            caveat = caveat.replace(
                "Activation comparison", "Function-space comparison", 1
            )
        lines.append(f"- {caveat}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_path, metrics = latest_metrics(args.run_root)
    rows = comparison_rows(metrics, run_path)
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, run_path, metrics)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
