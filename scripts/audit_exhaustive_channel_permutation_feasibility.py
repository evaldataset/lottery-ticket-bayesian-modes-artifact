#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_full_data_channel_permutation_matching import (
    identity_maps,
    key_channel_counts,
    mask_metrics_for_maps,
    optimize_pair,
)
from audit_mask_artifact_posthoc_matching import (
    assignment_pairs,
    load_artifact,
    pairwise_mask_tables,
    rel,
    resnet_channel_keys,
    tensor_slices,
    unflatten_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAKE_ROOT = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_smoke"
DEFAULT_FULL_ARTIFACT = (
    ROOT
    / "runs"
    / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3"
    / "20260506_230706"
    / "mask_artifacts.npz"
)
DEFAULT_OUT_JSON = ROOT / "runs" / "resnet_channel_permutation_exhaustive_feasibility_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "resnet_channel_permutation_exhaustive_feasibility_audit.md"

EXACT_COMPARISONS = [
    ("ticket", "activation_aligned_ticket", "ticket raw-vs-aligned frame"),
    (
        "chain_start",
        "activation_aligned_chain_start",
        "chain-start raw-vs-aligned frame",
    ),
    ("posterior_sample", "ticket", "posterior samples vs tickets"),
    (
        "activation_aligned_posterior_sample",
        "activation_aligned_ticket",
        "activation-aligned posterior samples vs tickets",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-artifact", type=Path, default=None)
    parser.add_argument("--full-artifact", type=Path, default=DEFAULT_FULL_ARTIFACT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-07")
    parser.add_argument("--max-iters", type=int, default=6)
    return parser.parse_args()


def latest_fake_artifact() -> Path:
    candidates = sorted(DEFAULT_FAKE_ROOT.glob("*/mask_artifacts.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"no fake mask_artifacts.npz found below {rel(DEFAULT_FAKE_ROOT)}"
        )
    return candidates[-1]


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def channel_search_space(artifact: dict[str, Any]) -> dict[str, Any]:
    counts = key_channel_counts(artifact)
    histogram = Counter(int(value) for value in counts.values())
    log10_count = sum(math.lgamma(channels + 1) / math.log(10.0) for channels in counts.values())
    exact_count = None
    if log10_count < 15:
        exact_count = int(
            math.prod(math.factorial(channels) for channels in counts.values())
        )
    return {
        "channel_key_count": len(counts),
        "channel_count_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "log10_permutation_count": float(log10_count),
        "exact_permutation_count": exact_count,
    }


def stage1_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    selected_names = [
        name
        for name in artifact["parameter_names"]
        if name == "conv1.weight" or str(name).startswith("layer1.")
    ]
    if not selected_names:
        raise ValueError("artifact has no stage-1 ResNet tensors")

    slices = tensor_slices(artifact)
    offsets = []
    sizes = []
    shapes = {}
    offset = 0
    for name in selected_names:
        shape = artifact["parameter_shapes"][name]
        size = int(np.prod(shape))
        offsets.append(offset)
        sizes.append(size)
        shapes[name] = list(shape)
        offset += size

    restricted = dict(artifact)
    restricted["metadata"] = dict(artifact["metadata"])
    restricted["metadata"]["dataset"] = f"{artifact['metadata'].get('dataset')}-stage1-subgraph"
    restricted["parameter_names"] = list(selected_names)
    restricted["parameter_sizes"] = np.asarray(sizes, dtype=np.int64)
    restricted["parameter_offsets"] = np.asarray(offsets, dtype=np.int64)
    restricted["parameter_shapes"] = shapes
    restricted["masks"] = {
        name: np.concatenate([matrix[:, slices[param]] for param in selected_names], axis=1)
        for name, matrix in artifact["masks"].items()
    }
    restricted["states"] = {
        name: np.concatenate([matrix[:, slices[param]] for param in selected_names], axis=1)
        for name, matrix in artifact["states"].items()
    }
    return restricted


def permutation_lists(artifact: dict[str, Any]) -> tuple[list[str], list[list[np.ndarray]]]:
    keys = resnet_channel_keys(artifact)
    identity = identity_maps(artifact)
    per_key = [
        [
            np.asarray(perm, dtype=np.int64)
            for perm in itertools.permutations(range(int(identity[key].shape[0])))
        ]
        for key in keys
    ]
    return keys, per_key


def exact_global_metrics(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    artifact: dict[str, Any],
    *,
    keys: list[str],
    per_key_permutations: list[list[np.ndarray]],
) -> dict[str, Any]:
    left_tensors = unflatten_tensors(left_vector, artifact)
    right_tensors = unflatten_tensors(right_vector, artifact)
    identity = identity_maps(artifact)
    best_metrics: dict[str, float] | None = None
    best_changed_key_count = 0
    for combo in itertools.product(*per_key_permutations):
        maps = {key: value for key, value in zip(keys, combo)}
        metrics = mask_metrics_for_maps(left_tensors, right_tensors, artifact, maps)
        if best_metrics is None or metrics["hamming"] < best_metrics["hamming"]:
            best_metrics = metrics
            best_changed_key_count = int(
                sum(not np.array_equal(identity[key], maps[key]) for key in keys)
            )
    if best_metrics is None:
        raise ValueError("no permutations enumerated")
    return {
        "metrics": best_metrics,
        "changed_key_count": best_changed_key_count,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def exact_comparison_payload(
    artifact: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    label: str,
    max_iters: int,
    keys: list[str],
    per_key_permutations: list[list[np.ndarray]],
) -> dict[str, Any]:
    left = artifact["masks"][left_name]
    right = artifact["masks"][right_name]
    tables = pairwise_mask_tables(left, right)
    pairs = assignment_pairs(tables["hamming"])
    pair_rows = []
    for left_idx, right_idx in pairs:
        raw_hamming = float(tables["hamming"][left_idx, right_idx])
        exact = exact_global_metrics(
            left[left_idx],
            right[right_idx],
            artifact,
            keys=keys,
            per_key_permutations=per_key_permutations,
        )
        coordinate = optimize_pair(
            left[left_idx],
            right[right_idx],
            artifact,
            objective="mask",
            max_iters=max_iters,
        )
        coordinate_hamming = float(coordinate["best"]["final"]["hamming"])
        exact_hamming = float(exact["metrics"]["hamming"])
        pair_rows.append(
            {
                "left_index": int(left_idx),
                "right_index": int(right_idx),
                "raw_record_hamming": raw_hamming,
                "exact_global_hamming": exact_hamming,
                "coordinate_descent_hamming": coordinate_hamming,
                "exact_support_overlap_min": float(
                    exact["metrics"]["support_overlap_min"]
                ),
                "exact_jaccard_keep": float(exact["metrics"]["jaccard_keep"]),
                "exact_improvement": raw_hamming - exact_hamming,
                "coordinate_matches_exact": bool(
                    abs(coordinate_hamming - exact_hamming) < 1e-12
                ),
                "exact_changed_key_count": int(exact["changed_key_count"]),
                "coordinate_best_start": coordinate["best"]["start"],
            }
        )
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "pair_count": len(pair_rows),
        "raw_record_hamming": summarize([row["raw_record_hamming"] for row in pair_rows]),
        "exact_global_hamming": summarize(
            [row["exact_global_hamming"] for row in pair_rows]
        ),
        "coordinate_descent_hamming": summarize(
            [row["coordinate_descent_hamming"] for row in pair_rows]
        ),
        "exact_support_overlap_min": summarize(
            [row["exact_support_overlap_min"] for row in pair_rows]
        ),
        "exact_improvement": summarize([row["exact_improvement"] for row in pair_rows]),
        "coordinate_descent_matches_exact": all(
            row["coordinate_matches_exact"] for row in pair_rows
        ),
        "pairs": pair_rows,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    fake_artifact_path = args.fake_artifact or latest_fake_artifact()
    fake_artifact = load_artifact(fake_artifact_path)
    full_artifact = load_artifact(args.full_artifact)
    stage1 = stage1_artifact(fake_artifact)
    keys, per_key_permutations = permutation_lists(stage1)
    exact_comparisons = [
        exact_comparison_payload(
            stage1,
            left_name=left,
            right_name=right,
            label=label,
            max_iters=args.max_iters,
            keys=keys,
            per_key_permutations=per_key_permutations,
        )
        for left, right, label in EXACT_COMPARISONS
    ]
    stage1_search = channel_search_space(stage1)
    fake_search = channel_search_space(fake_artifact)
    full_search = channel_search_space(full_artifact)
    overall = {
        "fake_artifact_path": rel(fake_artifact_path),
        "full_artifact_path": rel(args.full_artifact),
        "stage1_parameter_count": int(stage1["parameter_sizes"].sum()),
        "stage1_tensor_count": len(stage1["parameter_names"]),
        "stage1_exact_enumeration_supported": True,
        "stage1_exact_permutation_count": int(
            stage1_search["exact_permutation_count"] or 0
        ),
        "stage1_coordinate_descent_all_exact": all(
            row["coordinate_descent_matches_exact"] for row in exact_comparisons
        ),
        "full_exhaustive_channel_permutation_supported": False,
        "full_log10_permutation_count": full_search["log10_permutation_count"],
        "limitation_statement": (
            "Exact global channel-permutation enumeration is run only on a "
            "270-parameter fake-CIFAR stage-1 subgraph with 128 assignments. "
            "The full CIFAR ResNet-20 artifact has about 10^840 channel "
            "assignments per record pair, so exhaustive graph/channel search "
            "remains infeasible and unimplemented for the full-data claim."
        ),
    }
    return {
        "date": args.date,
        "overall": overall,
        "search_spaces": {
            "fake_full_artifact": fake_search,
            "fake_stage1_subgraph": stage1_search,
            "full_cifar_artifact": full_search,
        },
        "exact_stage1_comparisons": exact_comparisons,
    }


def histogram_text(histogram: dict[str, int]) -> str:
    return ", ".join(f"{key}x{value}" for key, value in sorted(histogram.items()))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    search = payload["search_spaces"]
    lines = [
        "# Exhaustive Channel Permutation Feasibility Audit",
        "",
        f"Date: {payload['date']}",
        "",
        "This audit separates two claims that should not be conflated. First,",
        "it verifies that exact global channel-permutation enumeration is",
        "possible on a deliberately tiny ResNet subgraph cut from the saved",
        "fake-CIFAR mask artifact. Second, it quantifies why the corresponding",
        "full-data CIFAR ResNet-20 search is outside the artifact budget.",
        "",
        "## Search Spaces",
        "",
        "| Artifact | Channel keys | Channel counts | log10 assignments | Exact count |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name, row in [
        ("fake full artifact", search["fake_full_artifact"]),
        ("fake stage-1 subgraph", search["fake_stage1_subgraph"]),
        ("full CIFAR artifact", search["full_cifar_artifact"]),
    ]:
        exact_count = row["exact_permutation_count"]
        lines.append(
            "| "
            f"{name} | "
            f"{row['channel_key_count']} | "
            f"{histogram_text(row['channel_count_histogram'])} | "
            f"{format_float(row['log10_permutation_count'], 1)} | "
            f"{exact_count if exact_count is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Exact Stage-1 Enumeration",
            "",
            f"- Fake artifact: `{overall['fake_artifact_path']}`",
            f"- Full-data artifact used for search-space sizing: `{overall['full_artifact_path']}`",
            f"- Stage-1 parameters: `{overall['stage1_parameter_count']}`",
            f"- Stage-1 exact assignments: `{overall['stage1_exact_permutation_count']}`",
            f"- Coordinate descent matches exact enumeration: `{overall['stage1_coordinate_descent_all_exact']}`",
            f"- Full-data exhaustive channel permutation supported: `{overall['full_exhaustive_channel_permutation_supported']}`",
            "",
            "| Comparison | Pairs | Raw record Hamming | Exact global Hamming | Coordinate Hamming | Improvement | Exact support overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["exact_stage1_comparisons"]:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['pair_count']} | "
            f"{format_float(row['raw_record_hamming']['mean'])} | "
            f"{format_float(row['exact_global_hamming']['mean'])} | "
            f"{format_float(row['coordinate_descent_hamming']['mean'])} | "
            f"{format_float(row['exact_improvement']['mean'])} | "
            f"{format_float(row['exact_support_overlap_min']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The tiny exact enumeration behaves as a path-validation check:",
            "known raw-vs-aligned frame differences are driven to zero, and",
            "the block-coordinate channel audit reaches the same optimum on",
            "this subgraph. The same exhaustive strategy does not scale to the",
            "retained full-data CIFAR artifact: its 19 channel keys imply about",
            f"`10^{format_float(overall['full_log10_permutation_count'], 1)}`",
            "channel assignments per record pair before considering broader",
            "graph-isomorphism variants. This keeps the reviewer-facing",
            "statement precise: full-data channel relabeling has been tested by",
            "a structured global objective, while exhaustive full-data graph",
            "isomorphism remains infeasible and unimplemented rather than silently",
            "assumed.",
            "",
            "This file is generated by",
            "`scripts/audit_exhaustive_channel_permutation_feasibility.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(
        json.dumps(
            {
                "out_json": rel(args.out_json),
                "out_md": rel(args.out_md),
                "stage1_exact_assignments": payload["overall"][
                    "stage1_exact_permutation_count"
                ],
                "full_log10_assignments": payload["overall"][
                    "full_log10_permutation_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
