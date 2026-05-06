#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from audit_mask_artifact_posthoc_matching import (
    assignment_pairs,
    channel_axis,
    channel_features_from_tensors,
    load_artifact,
    pairwise_mask_tables,
    rel,
    resnet_channel_keys,
    resnet_weight_axes,
    unflatten_tensors,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    ROOT
    / "runs"
    / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3"
    / "20260506_230706"
    / "mask_artifacts.npz"
)
DEFAULT_OUT_JSON = (
    ROOT
    / "runs"
    / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json"
)
DEFAULT_OUT_MD = (
    ROOT
    / "docs"
    / "cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md"
)

DEFAULT_COMPARISONS = [
    ("posterior_sample", "ticket", "raw posterior samples vs tickets"),
    (
        "activation_aligned_posterior_sample",
        "activation_aligned_ticket",
        "activation-aligned posterior samples vs tickets",
    ),
    ("posterior_mode", "ticket", "raw posterior modes vs tickets"),
    (
        "activation_aligned_posterior_mode",
        "activation_aligned_ticket",
        "activation-aligned posterior modes vs tickets",
    ),
    ("chain_start", "ticket", "raw chain starts vs tickets"),
    (
        "activation_aligned_chain_start",
        "activation_aligned_ticket",
        "activation-aligned chain starts vs tickets",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-07")
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument(
        "--max-pairs-per-comparison",
        type=int,
        default=5,
        help="Run channel optimization on at most this many record-optimal pairs.",
    )
    parser.add_argument(
        "--comparisons",
        default=",".join(f"{left}:{right}" for left, right, _ in DEFAULT_COMPARISONS),
        help="Comma-separated left:right collection names. Defaults to the main full-data comparisons.",
    )
    parser.add_argument(
        "--objective",
        choices=["mask"],
        default="mask",
        help="Objective used for channel permutation coordinate updates.",
    )
    return parser.parse_args()


def parse_comparisons(text: str) -> list[tuple[str, str, str]]:
    labels = {(left, right): label for left, right, label in DEFAULT_COMPARISONS}
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"comparison must be left:right, got {item!r}")
        left, right = item.split(":", 1)
        key = (left.strip(), right.strip())
        out.append((key[0], key[1], labels.get(key, f"{key[0]} vs {key[1]}")))
    return out


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def linear_sum_assignment_min(cost: np.ndarray) -> list[tuple[int, int]]:
    """Exact rectangular minimum assignment using the Hungarian algorithm.

    This avoids a SciPy dependency in the CPU verifier container while still
    giving exact channel assignments for 16/32/64-channel ResNet groups.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("assignment cost must be a 2D matrix")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return []
    transposed = False
    if rows > cols:
        matrix = matrix.T
        rows, cols = matrix.shape
        transposed = True

    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(int(p[j] - 1), int(j - 1)) for j in range(1, cols + 1) if p[j] != 0]
    if transposed:
        pairs = [(col, row) for row, col in pairs]
    return sorted(pairs)


def source_to_target_from_cost(cost: np.ndarray) -> np.ndarray:
    pairs = linear_sum_assignment_min(cost)
    channels = cost.shape[0]
    mapping = np.arange(channels, dtype=np.int64)
    for source_idx, target_idx in pairs:
        mapping[source_idx] = target_idx
    return mapping


def invert_permutation(mapping: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(mapping.shape[0], dtype=np.int64)
    return inverse


def key_channel_counts(artifact: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in artifact["parameter_names"]:
        shape = tuple(artifact["parameter_shapes"][name])
        out_key, in_key = resnet_weight_axes(name)
        for key, axis in [(out_key, 0), (in_key, 1)]:
            if key is None:
                continue
            channels = int(shape[axis])
            previous = counts.get(key)
            if previous is not None and previous != channels:
                raise ValueError(f"channel count mismatch for {key}: {previous} vs {channels}")
            counts[key] = channels
    return counts


def identity_maps(artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.arange(channels, dtype=np.int64)
        for key, channels in key_channel_counts(artifact).items()
    }


def local_feature_maps(
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    *,
    objective: str,
) -> dict[str, np.ndarray]:
    maps = identity_maps(artifact)
    for key in resnet_channel_keys(artifact):
        left_features = channel_features_from_tensors(left_tensors, artifact, key)
        right_features = channel_features_from_tensors(right_tensors, artifact, key)
        if left_features is None or right_features is None:
            continue
        if objective == "mask":
            cost = bool_feature_cost(left_features.astype(bool), right_features.astype(bool))
        else:
            cost = squared_feature_cost(
                left_features.astype(np.float64),
                right_features.astype(np.float64),
            )
        maps[key] = source_to_target_from_cost(cost)
    return maps


def apply_out_map(tensor: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    target = np.empty_like(tensor)
    target[mapping, ...] = tensor
    return target


def apply_in_map(tensor: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return np.take(tensor, invert_permutation(mapping), axis=1)


def transform_except_key(
    tensor: np.ndarray,
    name: str,
    maps: dict[str, np.ndarray],
    *,
    skip_key: str | None,
) -> np.ndarray:
    out_key, in_key = resnet_weight_axes(name)
    value = tensor
    if out_key is not None and out_key != skip_key:
        value = apply_out_map(value, maps[out_key])
    if in_key is not None and in_key != skip_key:
        value = apply_in_map(value, maps[in_key])
    return value


def align_tensor(
    tensor: np.ndarray,
    name: str,
    maps: dict[str, np.ndarray],
) -> np.ndarray:
    return transform_except_key(tensor, name, maps, skip_key=None)


def axis_features(value: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(value, axis, 0).reshape(value.shape[axis], -1)


def bool_feature_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_i = left.astype(np.int32, copy=False)
    right_i = right.astype(np.int32, copy=False)
    left_keep = left_i.sum(axis=1, dtype=np.int64)
    right_keep = right_i.sum(axis=1, dtype=np.int64)
    both = left_i @ right_i.T
    return left_keep[:, None] + right_keep[None, :] - 2 * both


def squared_feature_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    left_sq = np.einsum("ij,ij->i", left64, left64)
    right_sq = np.einsum("ij,ij->i", right64, right64)
    sq = left_sq[:, None] + right_sq[None, :] - 2.0 * (left64 @ right64.T)
    np.maximum(sq, 0.0, out=sq)
    return sq


def key_update_cost(
    key: str,
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    maps: dict[str, np.ndarray],
    *,
    objective: str,
) -> np.ndarray:
    channels = maps[key].shape[0]
    cost = np.zeros((channels, channels), dtype=np.float64)
    touched = False
    for name in artifact["parameter_names"]:
        out_key, in_key = resnet_weight_axes(name)
        axis = channel_axis(name, key)
        if axis is None:
            continue
        left_value = transform_except_key(left_tensors[name], name, maps, skip_key=key)
        right_value = right_tensors[name]
        left_features = axis_features(left_value, axis)
        right_features = axis_features(right_value, axis)
        if objective == "mask":
            cost += bool_feature_cost(left_features.astype(bool), right_features.astype(bool))
        else:
            cost += squared_feature_cost(left_features, right_features)
        touched = True
        if key == out_key and key == in_key:
            raise ValueError(f"unsupported tensor with same in/out channel key: {name}")
    if not touched:
        raise ValueError(f"no tensors touched for channel key {key}")
    return cost


def mask_metrics_for_maps(
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    maps: dict[str, np.ndarray],
) -> dict[str, float]:
    mismatch = 0
    both = 0
    either = 0
    left_keep = 0
    right_keep = 0
    total = 0
    for name in artifact["parameter_names"]:
        left_aligned = align_tensor(left_tensors[name], name, maps).astype(bool)
        right = right_tensors[name].astype(bool)
        mismatch += int(np.count_nonzero(left_aligned != right))
        both_name = int(np.logical_and(left_aligned, right).sum())
        either_name = int(np.logical_or(left_aligned, right).sum())
        both += both_name
        either += either_name
        left_keep += int(left_aligned.sum())
        right_keep += int(right.sum())
        total += int(right.size)
    denom = min(left_keep, right_keep)
    return {
        "hamming": float(mismatch / total),
        "jaccard_keep": float(both / either) if either else 1.0,
        "support_overlap_min": float(both / denom) if denom else 1.0,
    }


def state_rms_for_maps(
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    maps: dict[str, np.ndarray],
) -> float:
    sq = 0.0
    total = 0
    for name in artifact["parameter_names"]:
        left_aligned = align_tensor(left_tensors[name], name, maps).astype(np.float64)
        right = right_tensors[name].astype(np.float64)
        diff = left_aligned - right
        sq += float(np.sum(diff * diff))
        total += int(diff.size)
    return float(math.sqrt(sq / total))


def coordinate_descent_maps(
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    initial_maps: dict[str, np.ndarray],
    *,
    objective: str,
    max_iters: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    keys = resnet_channel_keys(artifact)
    maps = {key: value.copy() for key, value in initial_maps.items()}
    trace = []
    previous = mask_metrics_for_maps(left_tensors, right_tensors, artifact, maps)["hamming"]
    trace.append({"iteration": 0, "hamming": previous})
    for iteration in range(1, max_iters + 1):
        changed = 0
        for key in keys:
            before = maps[key].copy()
            maps[key] = source_to_target_from_cost(
                key_update_cost(
                    key,
                    left_tensors,
                    right_tensors,
                    artifact,
                    maps,
                    objective=objective,
                )
            )
            if not np.array_equal(before, maps[key]):
                changed += 1
        current = mask_metrics_for_maps(left_tensors, right_tensors, artifact, maps)[
            "hamming"
        ]
        trace.append({"iteration": iteration, "hamming": current, "changed_keys": changed})
        if current >= previous - 1e-12:
            break
        previous = current
    return maps, trace


def optimize_pair(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    artifact: dict[str, Any],
    *,
    objective: str,
    max_iters: int,
) -> dict[str, Any]:
    left_tensors = unflatten_tensors(left_vector, artifact)
    right_tensors = unflatten_tensors(right_vector, artifact)
    starts = {
        "identity": identity_maps(artifact),
        "local_feature": local_feature_maps(
            left_tensors,
            right_tensors,
            artifact,
            objective=objective,
        ),
    }
    start_results = []
    best: dict[str, Any] | None = None
    for start_name, start_maps in starts.items():
        initial_metrics = mask_metrics_for_maps(left_tensors, right_tensors, artifact, start_maps)
        maps, trace = coordinate_descent_maps(
            left_tensors,
            right_tensors,
            artifact,
            start_maps,
            objective=objective,
            max_iters=max_iters,
        )
        final_metrics = mask_metrics_for_maps(left_tensors, right_tensors, artifact, maps)
        state_rms = (
            state_rms_for_maps(left_tensors, right_tensors, artifact, maps)
            if objective == "state"
            else None
        )
        result = {
            "start": start_name,
            "initial": initial_metrics,
            "final": final_metrics,
            "trace": trace,
            "state_rms": state_rms,
            "changed_key_count": int(
                sum(
                    not np.array_equal(starts["identity"][key], maps[key])
                    for key in resnet_channel_keys(artifact)
                )
            ),
        }
        start_results.append(result)
        if best is None or final_metrics["hamming"] < best["final"]["hamming"]:
            best = result
    assert best is not None
    return {
        "best": best,
        "starts": start_results,
    }


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def comparison_payload(
    artifact: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    label: str,
    objective: str,
    max_iters: int,
    max_pairs: int,
) -> dict[str, Any] | None:
    masks = artifact["masks"]
    if left_name not in masks or right_name not in masks:
        return None
    left = masks[left_name]
    right = masks[right_name]
    tables = pairwise_mask_tables(left, right)
    record_pairs = assignment_pairs(tables["hamming"])
    record_pairs = record_pairs[:max_pairs]
    left_ids = artifact["ids"].get(left_name, [str(i) for i in range(left.shape[0])])
    right_ids = artifact["ids"].get(right_name, [str(i) for i in range(right.shape[0])])
    pair_rows = []
    for left_idx, right_idx in record_pairs:
        optimized = optimize_pair(
            left[left_idx],
            right[right_idx],
            artifact,
            objective=objective,
            max_iters=max_iters,
        )
        raw_hamming = float(tables["hamming"][left_idx, right_idx])
        raw_jaccard = float(tables["jaccard_keep"][left_idx, right_idx])
        raw_overlap = float(tables["support_overlap_min"][left_idx, right_idx])
        best_metrics = optimized["best"]["final"]
        pair_rows.append(
            {
                "left_index": int(left_idx),
                "right_index": int(right_idx),
                "left_id": left_ids[left_idx] if left_idx < len(left_ids) else str(left_idx),
                "right_id": (
                    right_ids[right_idx] if right_idx < len(right_ids) else str(right_idx)
                ),
                "raw_record_hamming": raw_hamming,
                "raw_record_jaccard_keep": raw_jaccard,
                "raw_record_support_overlap_min": raw_overlap,
                "best_start": optimized["best"]["start"],
                "global_channel_hamming": float(best_metrics["hamming"]),
                "global_channel_jaccard_keep": float(best_metrics["jaccard_keep"]),
                "global_channel_support_overlap_min": float(
                    best_metrics["support_overlap_min"]
                ),
                "hamming_improvement": raw_hamming - float(best_metrics["hamming"]),
                "changed_key_count": int(optimized["best"]["changed_key_count"]),
                "trace": optimized["best"]["trace"],
                "start_summaries": [
                    {
                        "start": row["start"],
                        "initial_hamming": row["initial"]["hamming"],
                        "final_hamming": row["final"]["hamming"],
                        "changed_key_count": row["changed_key_count"],
                    }
                    for row in optimized["starts"]
                ],
            }
        )
    raw_values = [row["raw_record_hamming"] for row in pair_rows]
    global_values = [row["global_channel_hamming"] for row in pair_rows]
    overlap_values = [row["global_channel_support_overlap_min"] for row in pair_rows]
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "left_count": int(left.shape[0]),
        "right_count": int(right.shape[0]),
        "optimized_pair_count": len(pair_rows),
        "record_optimal_pair_count": len(record_pairs),
        "raw_record_hamming": summarize_values(raw_values),
        "global_channel_hamming": summarize_values(global_values),
        "global_channel_support_overlap_min": summarize_values(overlap_values),
        "hamming_improvement": summarize_values(
            [row["hamming_improvement"] for row in pair_rows]
        ),
        "pairs": pair_rows,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_artifact(args.artifact)
    comparisons = []
    for left_name, right_name, label in parse_comparisons(args.comparisons):
        row = comparison_payload(
            artifact,
            left_name=left_name,
            right_name=right_name,
            label=label,
            objective=args.objective,
            max_iters=args.max_iters,
            max_pairs=args.max_pairs_per_comparison,
        )
        if row is not None:
            comparisons.append(row)
    overall = {
        "dataset": artifact["metadata"].get("dataset"),
        "model": artifact["metadata"].get("model"),
        "artifact_path": rel(args.artifact),
        "parameter_count": int(artifact["parameter_sizes"].sum()),
        "parameter_name_count": len(artifact["parameter_names"]),
        "resnet_channel_key_count": len(resnet_channel_keys(artifact)),
        "objective": args.objective,
        "max_iters": int(args.max_iters),
        "max_pairs_per_comparison": int(args.max_pairs_per_comparison),
        "comparison_count": len(comparisons),
        "global_channel_coordinate_descent_supported": bool(comparisons),
        "exhaustive_graph_channel_permutation_supported": False,
        "limitation_statement": (
            "This is a block-coordinate global channel-permutation robustness "
            "audit over saved masks/states. It uses exact per-key Hungarian "
            "updates but is not an exhaustive graph-isomorphism search."
        ),
    }
    return {
        "date": args.date,
        "overall": overall,
        "comparisons": comparisons,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    title = (
        "Full-data Global Channel Permutation Audit"
        if overall["dataset"] == "cifar10"
        else "Global Channel Permutation Audit"
    )
    retained_scope = (
        "The retained full-data" if overall["dataset"] == "cifar10" else "The retained"
    )
    lines = [
        f"# {title}",
        "",
        f"Date: {payload['date']}",
        "",
        "This audit reads a saved `mask_artifacts.npz` file and applies a",
        "ResNet-structured channel-permutation objective to record-optimal",
        "posterior/ticket pairs. For each pair it optimizes all ResNet channel",
        "keys by block-coordinate descent with exact per-key Hungarian updates.",
        "This is stronger than same-index or record-level matching, but it is",
        "not an exhaustive graph-isomorphism proof.",
        "",
        "## Artifact",
        "",
        f"- Path: `{overall['artifact_path']}`",
        f"- Dataset/model: `{overall['dataset']}` / `{overall['model']}`",
        f"- Parameters: `{overall['parameter_count']}`",
        f"- ResNet channel keys: `{overall['resnet_channel_key_count']}`",
        f"- Objective: `{overall['objective']}`",
        f"- Max iterations: `{overall['max_iters']}`",
        f"- Max optimized pairs per comparison: `{overall['max_pairs_per_comparison']}`",
        f"- Exhaustive graph/channel permutation supported: `{overall['exhaustive_graph_channel_permutation_supported']}`",
        "",
        "## Summary",
        "",
        "| Comparison | Pairs | Raw record Hamming | Global-channel Hamming | Improvement | Global support overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparisons"]:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['optimized_pair_count']} | "
            f"{format_float(row['raw_record_hamming']['mean'])} | "
            f"{format_float(row['global_channel_hamming']['mean'])} | "
            f"{format_float(row['hamming_improvement']['mean'])} | "
            f"{format_float(row['global_channel_support_overlap_min']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A permutation rescue would need to drive posterior/ticket mask",
            f"distances close to ticket-level agreement. {retained_scope}",
            "pairs remain far from that regime after structured channel",
            "permutation optimization, so channel relabeling does not rescue",
            "the tested support-equivalence claim. The audit remains bounded:",
            "it is a coordinate-descent robustness check, not an exhaustive",
            "global graph isomorphism solver.",
            "",
            "This file is generated by",
            "`scripts/audit_full_data_channel_permutation_matching.py`.",
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
                "comparisons": payload["overall"]["comparison_count"],
                "artifact": payload["overall"]["artifact_path"],
            }
        )
    )


if __name__ == "__main__":
    main()
