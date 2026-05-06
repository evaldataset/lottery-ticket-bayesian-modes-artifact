#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json"
DEFAULT_OUT_MD = ROOT / "docs" / "fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md"
DEFAULT_ARTIFACT_ROOT = ROOT / "runs" / "fake_cifar10_mode_ticket_mask_artifact_smoke"

DEFAULT_COMPARISONS = [
    ("posterior_sample", "ticket", "raw posterior samples vs tickets"),
    ("posterior_mode", "ticket", "raw posterior modes vs tickets"),
    ("chain_start", "ticket", "raw chain starts vs tickets"),
    (
        "activation_aligned_posterior_sample",
        "activation_aligned_ticket",
        "activation-aligned posterior samples vs tickets",
    ),
    (
        "activation_aligned_posterior_mode",
        "activation_aligned_ticket",
        "activation-aligned posterior modes vs tickets",
    ),
    (
        "activation_aligned_chain_start",
        "activation_aligned_ticket",
        "activation-aligned chain starts vs tickets",
    ),
    (
        "posterior_sample",
        "activation_aligned_posterior_sample",
        "posterior sample raw-vs-aligned delta",
    ),
    ("ticket", "activation_aligned_ticket", "ticket raw-vs-aligned delta"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--date", default="2026-05-06")
    parser.add_argument(
        "--max-pair-count",
        type=int,
        default=1024,
        help=(
            "Skip full comparison tables when left_count * right_count exceeds "
            "this value. Use 0 for no cap."
        ),
    )
    parser.add_argument(
        "--max-channel-pair-count",
        type=int,
        default=512,
        help=(
            "Skip local channel-permutation matching when left_count * "
            "right_count exceeds this value. Use 0 for no cap."
        ),
    )
    return parser.parse_args()


def latest_default_artifact() -> Path:
    candidates = sorted(DEFAULT_ARTIFACT_ROOT.glob("*/mask_artifacts.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"no mask_artifacts.npz found below {DEFAULT_ARTIFACT_ROOT.relative_to(ROOT)}"
        )
    return candidates[-1]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_artifact(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if "parameter_shapes_json" in data.files:
            parameter_shapes = {
                str(name): [int(dim) for dim in shape]
                for name, shape in json.loads(str(data["parameter_shapes_json"])).items()
            }
        else:
            parameter_shapes = {
                str(name): [int(size)]
                for name, size in zip(data["parameter_names"], data["parameter_sizes"])
            }
        masks: dict[str, np.ndarray] = {}
        states: dict[str, np.ndarray] = {}
        ids: dict[str, list[str]] = {}
        state_ids: dict[str, list[str]] = {}
        for key in data.files:
            if key.startswith("masks__"):
                masks[key.removeprefix("masks__")] = data[key].astype(bool)
            elif key.startswith("states__"):
                states[key.removeprefix("states__")] = data[key].astype(np.float32)
            elif key.startswith("ids__"):
                ids[key.removeprefix("ids__")] = [str(value) for value in data[key]]
            elif key.startswith("state_ids__"):
                state_ids[key.removeprefix("state_ids__")] = [
                    str(value) for value in data[key]
                ]
        return {
            "path": path,
            "schema_version": int(data["artifact_schema_version"][0]),
            "parameter_names": [str(value) for value in data["parameter_names"]],
            "parameter_sizes": data["parameter_sizes"].astype(np.int64),
            "parameter_offsets": data["parameter_offsets"].astype(np.int64),
            "parameter_shapes": parameter_shapes,
            "has_parameter_shapes": "parameter_shapes_json" in data.files,
            "metadata": metadata,
            "masks": masks,
            "states": states,
            "ids": ids,
            "state_ids": state_ids,
        }


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def pairwise_mask_tables(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("mask collections must be 2D matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("mask collections must have the same parameter width")
    left_int = left.astype(np.int32, copy=False)
    right_int = right.astype(np.int32, copy=False)
    left_keep = left_int.sum(axis=1, dtype=np.int64)
    right_keep = right_int.sum(axis=1, dtype=np.int64)
    both = (left_int @ right_int.T).astype(np.float64)
    either = left_keep[:, None] + right_keep[None, :] - both
    denom = np.minimum(left_keep[:, None], right_keep[None, :])
    hamming = (left_keep[:, None] + right_keep[None, :] - 2.0 * both) / float(left.shape[1])
    jaccard = np.divide(
        both,
        either,
        out=np.ones_like(both, dtype=np.float64),
        where=either != 0,
    )
    overlap_min = np.divide(
        both,
        denom,
        out=np.ones_like(both, dtype=np.float64),
        where=denom != 0,
    )
    return {
        "hamming": hamming,
        "jaccard_keep": jaccard,
        "support_overlap_min": overlap_min,
    }


def pairwise_state_rms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("state collections must be 2D matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("state collections must have the same parameter width")
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    left_sq = np.einsum("ij,ij->i", left64, left64)
    right_sq = np.einsum("ij,ij->i", right64, right64)
    sq = left_sq[:, None] + right_sq[None, :] - 2.0 * (left64 @ right64.T)
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq / float(left.shape[1]))


def index_pairs(size_left: int, size_right: int) -> list[tuple[int, int]]:
    return [(idx, idx) for idx in range(min(size_left, size_right))]


def values_for_pairs(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> list[float]:
    return [float(matrix[i, j]) for i, j in pairs]


def greedy_assignment_pairs(cost: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat_order = np.argsort(cost, axis=None)
    for flat_idx in flat_order:
        row, col = np.unravel_index(int(flat_idx), cost.shape)
        if int(row) in used_rows or int(col) in used_cols:
            continue
        pairs.append((int(row), int(col)))
        used_rows.add(int(row))
        used_cols.add(int(col))
        if len(pairs) == min(cost.shape):
            break
    return sorted(pairs)


def assignment_pairs(cost: np.ndarray, *, exact_limit: int = 16) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment without optional SciPy dependency.

    The DP is exact when the smaller side has at most exact_limit records, which
    covers the current smoke fixture and the planned posterior-vs-five-ticket
    full-data reruns. Larger square posterior-vs-posterior comparisons fall back
    to a deterministic greedy assignment.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return []
    if min(rows, cols) > exact_limit:
        return greedy_assignment_pairs(cost)
    if rows > cols:
        return sorted((col, row) for row, col in assignment_pairs(cost.T, exact_limit=exact_limit))

    target_mask = (1 << rows) - 1
    dp: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for col in range(cols):
        next_dp = {mask: (value, pairs.copy()) for mask, (value, pairs) in dp.items()}
        for mask, (value, pairs) in dp.items():
            for row in range(rows):
                bit = 1 << row
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_value = value + float(cost[row, col])
                current = next_dp.get(next_mask)
                if current is None or next_value < current[0]:
                    next_dp[next_mask] = (next_value, pairs + [(row, col)])
        dp = next_dp
    return sorted(dp[target_mask][1])


def resnet_block_input_key(layer_idx: int, block_idx: int) -> str:
    if block_idx > 0:
        return f"layer{layer_idx}.{block_idx - 1}.out"
    if layer_idx == 1:
        return "stem"
    return f"layer{layer_idx - 1}.2.out"


def resnet_weight_axes(name: str) -> tuple[str | None, str | None]:
    if name == "conv1.weight":
        return "stem", None
    if name == "fc.weight":
        return None, "layer3.2.out"
    parts = name.split(".")
    if len(parts) == 4 and parts[0].startswith("layer") and parts[2] in {"conv1", "conv2"}:
        layer_idx = int(parts[0].removeprefix("layer"))
        block_idx = int(parts[1])
        if parts[2] == "conv1":
            return (
                f"layer{layer_idx}.{block_idx}.conv1",
                resnet_block_input_key(layer_idx, block_idx),
            )
        return (
            f"layer{layer_idx}.{block_idx}.out",
            f"layer{layer_idx}.{block_idx}.conv1",
        )
    if len(parts) == 5 and parts[0].startswith("layer") and parts[2] == "shortcut":
        layer_idx = int(parts[0].removeprefix("layer"))
        block_idx = int(parts[1])
        return (
            f"layer{layer_idx}.{block_idx}.out",
            resnet_block_input_key(layer_idx, block_idx),
        )
    raise ValueError(f"unsupported ResNet weight name for channel matching: {name}")


def tensor_slices(artifact: dict[str, Any]) -> dict[str, slice]:
    offsets = artifact["parameter_offsets"]
    sizes = artifact["parameter_sizes"]
    names = artifact["parameter_names"]
    return {
        name: slice(int(offset), int(offset + size))
        for name, offset, size in zip(names, offsets, sizes)
    }


def unflatten_tensors(vector: np.ndarray, artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    slices = tensor_slices(artifact)
    shapes = artifact["parameter_shapes"]
    tensors = {}
    for name in artifact["parameter_names"]:
        tensors[name] = vector[slices[name]].reshape(tuple(shapes[name]))
    return tensors


def resnet_channel_keys(artifact: dict[str, Any]) -> list[str]:
    if artifact["metadata"].get("model") != "resnet20" or not artifact["has_parameter_shapes"]:
        return []
    keys: set[str] = set()
    for name in artifact["parameter_names"]:
        try:
            out_key, in_key = resnet_weight_axes(name)
        except ValueError:
            return []
        if out_key is not None:
            keys.add(out_key)
        if in_key is not None:
            keys.add(in_key)
    return sorted(keys)


def channel_axis(name: str, key: str) -> int | None:
    out_key, in_key = resnet_weight_axes(name)
    if key == out_key:
        return 0
    if key == in_key:
        return 1
    return None


def channel_features(
    vector: np.ndarray,
    artifact: dict[str, Any],
    key: str,
) -> np.ndarray | None:
    return channel_features_from_tensors(unflatten_tensors(vector, artifact), artifact, key)


def channel_features_from_tensors(
    tensors: dict[str, np.ndarray],
    artifact: dict[str, Any],
    key: str,
) -> np.ndarray | None:
    pieces = []
    expected_channels = None
    for name in artifact["parameter_names"]:
        axis = channel_axis(name, key)
        if axis is None:
            continue
        value = tensors[name]
        if value.ndim <= axis:
            raise ValueError(f"cannot take channel axis {axis} for {name}")
        channels = int(value.shape[axis])
        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise ValueError(f"channel feature mismatch for {key}: {channels} vs {expected_channels}")
        pieces.append(np.moveaxis(value, axis, 0).reshape(channels, -1))
    if not pieces:
        return None
    return np.concatenate(pieces, axis=1)


def pairwise_channel_cost(left_features: np.ndarray, right_features: np.ndarray) -> np.ndarray:
    if left_features.dtype == np.bool_ and right_features.dtype == np.bool_:
        return np.mean(
            left_features[:, None, :] != right_features[None, :, :],
            axis=2,
            dtype=np.float64,
        )
    diff = right_features[None, :, :].astype(np.float64) - left_features[:, None, :].astype(
        np.float64
    )
    return np.sqrt(np.mean(diff * diff, axis=2))


def source_to_target_from_pairs(
    pairs: list[tuple[int, int]],
    channels: int,
) -> np.ndarray:
    mapping = np.arange(channels, dtype=np.int64)
    for source_idx, target_idx in pairs:
        mapping[source_idx] = target_idx
    return mapping


def invert_permutation(source_to_target: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(source_to_target)
    inverse[source_to_target] = np.arange(source_to_target.shape[0], dtype=np.int64)
    return inverse


def channel_maps_for_pair(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    artifact: dict[str, Any],
) -> dict[str, np.ndarray]:
    maps = {}
    left_tensors = unflatten_tensors(left_vector, artifact)
    right_tensors = unflatten_tensors(right_vector, artifact)
    for key in resnet_channel_keys(artifact):
        left_features = channel_features_from_tensors(left_tensors, artifact, key)
        right_features = channel_features_from_tensors(right_tensors, artifact, key)
        if left_features is None or right_features is None:
            continue
        if left_features.shape != right_features.shape:
            raise ValueError(
                f"channel feature shape mismatch for {key}: "
                f"{left_features.shape} vs {right_features.shape}"
            )
        cost = pairwise_channel_cost(left_features.astype(bool), right_features.astype(bool))
        pairs = assignment_pairs(cost)
        maps[key] = source_to_target_from_pairs(pairs, left_features.shape[0])
    return maps


def align_tensor_by_channel_maps(
    tensor: np.ndarray,
    name: str,
    channel_maps: dict[str, np.ndarray],
) -> np.ndarray:
    out_key, in_key = resnet_weight_axes(name)
    aligned = tensor.copy()
    if out_key is not None and out_key in channel_maps:
        source_to_target = channel_maps[out_key]
        target = np.empty_like(aligned)
        target[source_to_target, ...] = aligned
        aligned = target
    if in_key is not None and in_key in channel_maps:
        target_to_source = invert_permutation(channel_maps[in_key])
        aligned = np.take(aligned, target_to_source, axis=1)
    return aligned


def align_vector_by_channel_maps(
    vector: np.ndarray,
    artifact: dict[str, Any],
    channel_maps: dict[str, np.ndarray],
) -> np.ndarray:
    tensors = unflatten_tensors(vector, artifact)
    aligned_parts = [
        align_tensor_by_channel_maps(tensors[name], name, channel_maps).reshape(-1)
        for name in artifact["parameter_names"]
    ]
    return np.concatenate(aligned_parts, axis=0)


def channel_permuted_hamming(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    artifact: dict[str, Any],
) -> float:
    channel_maps = channel_maps_for_pair(left_vector, right_vector, artifact)
    if not channel_maps:
        return float("nan")
    aligned = align_vector_by_channel_maps(left_vector, artifact, channel_maps).astype(bool)
    return float(np.mean(aligned != right_vector.astype(bool)))


def pairwise_channel_permuted_hamming(
    left: np.ndarray,
    right: np.ndarray,
    artifact: dict[str, Any],
) -> np.ndarray | None:
    if not resnet_channel_keys(artifact):
        return None
    out = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    for i in range(left.shape[0]):
        for j in range(right.shape[0]):
            out[i, j] = channel_permuted_hamming(left[i], right[j], artifact)
    return out


def ids_for_pairs(
    pairs: list[tuple[int, int]],
    *,
    left_ids: list[str],
    right_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "left_index": i,
            "right_index": j,
            "left_id": left_ids[i] if i < len(left_ids) else str(i),
            "right_id": right_ids[j] if j < len(right_ids) else str(j),
        }
        for i, j in pairs
    ]


def collection_summary(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    masks = artifact["masks"][name]
    states = artifact["states"].get(name)
    keep_counts = masks.sum(axis=1).astype(np.int64)
    return {
        "name": name,
        "mask_count": int(masks.shape[0]),
        "parameter_count": int(masks.shape[1]),
        "keep_count_mean": float(np.mean(keep_counts)),
        "keep_fraction_mean": float(np.mean(keep_counts) / masks.shape[1]),
        "state_count": int(states.shape[0]) if states is not None else 0,
    }


def comparison_payload(
    left_name: str,
    right_name: str,
    label: str,
    artifact: dict[str, Any],
    *,
    max_channel_pair_count: int | None,
) -> dict[str, Any] | None:
    masks = artifact["masks"]
    if left_name not in masks or right_name not in masks:
        return None
    left = masks[left_name]
    right = masks[right_name]
    tables = pairwise_mask_tables(left, right)
    natural = index_pairs(left.shape[0], right.shape[0])
    optimal = assignment_pairs(tables["hamming"])
    left_ids = artifact["ids"].get(left_name, [str(i) for i in range(left.shape[0])])
    right_ids = artifact["ids"].get(right_name, [str(i) for i in range(right.shape[0])])
    natural_hamming = values_for_pairs(tables["hamming"], natural)
    optimal_hamming = values_for_pairs(tables["hamming"], optimal)
    natural_jaccard = values_for_pairs(tables["jaccard_keep"], natural)
    optimal_jaccard = values_for_pairs(tables["jaccard_keep"], optimal)
    natural_overlap = values_for_pairs(tables["support_overlap_min"], natural)
    optimal_overlap = values_for_pairs(tables["support_overlap_min"], optimal)

    state_payload = None
    states = artifact["states"]
    if left_name in states and right_name in states:
        state_rms = pairwise_state_rms(states[left_name], states[right_name])
        state_payload = {
            "natural_state_rms": summarize_values(values_for_pairs(state_rms, natural)),
            "optimal_state_rms": summarize_values(values_for_pairs(state_rms, optimal)),
        }
    channel_payload = None
    pair_count = int(left.shape[0] * right.shape[0])
    channel_hamming = None
    channel_skipped = (
        max_channel_pair_count is not None and pair_count > max_channel_pair_count
    )
    if not channel_skipped:
        channel_hamming = pairwise_channel_permuted_hamming(left, right, artifact)
    if channel_hamming is not None:
        channel_optimal = assignment_pairs(channel_hamming)
        channel_payload = {
            "channel_key_count": len(resnet_channel_keys(artifact)),
            "natural_hamming": summarize_values(values_for_pairs(channel_hamming, natural)),
            "record_optimal_hamming": summarize_values(values_for_pairs(channel_hamming, optimal)),
            "channel_optimal_hamming": summarize_values(
                values_for_pairs(channel_hamming, channel_optimal)
            ),
            "channel_optimal_pair_count": len(channel_optimal),
            "channel_optimal_pairs": ids_for_pairs(
                channel_optimal,
                left_ids=left_ids,
                right_ids=right_ids,
            ),
        }

    natural_mean = finite_mean(natural_hamming)
    optimal_mean = finite_mean(optimal_hamming)
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "left_count": int(left.shape[0]),
        "right_count": int(right.shape[0]),
        "same_index_pair_count": len(natural),
        "optimal_pair_count": len(optimal),
        "hungarian_pair_count": len(optimal),
        "natural_hamming": summarize_values(natural_hamming),
        "optimal_hamming": summarize_values(optimal_hamming),
        "hamming_improvement": (
            float(natural_mean - optimal_mean)
            if natural_mean is not None and optimal_mean is not None
            else None
        ),
        "natural_jaccard_keep": summarize_values(natural_jaccard),
        "optimal_jaccard_keep": summarize_values(optimal_jaccard),
        "natural_support_overlap_min": summarize_values(natural_overlap),
        "optimal_support_overlap_min": summarize_values(optimal_overlap),
        "natural_pairs": ids_for_pairs(natural, left_ids=left_ids, right_ids=right_ids),
        "optimal_pairs": ids_for_pairs(optimal, left_ids=left_ids, right_ids=right_ids),
        "state_distance": state_payload,
        "channel_permutation": channel_payload,
        "channel_permutation_skipped": bool(channel_skipped),
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_markdown(path: Path, *, date: str, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    if overall["dataset"] == "fake-cifar10":
        evidence_scope = [
            "This is a path-validation fixture, not claim-level CIFAR evidence.",
            "It does not implement exhaustive ResNet graph-isomorphism or channel",
            "permutation search; the full-data permutation rerun remains open.",
        ]
    else:
        evidence_scope = [
            "This is full-data CIFAR saved-artifact evidence for record-level",
            "post-hoc matching. It does not implement exhaustive ResNet",
            "graph-isomorphism or full local channel-permutation search.",
            "The full-data saved-artifact rerun is complete, but global graph",
            "or channel-permutation matching remains a separate open analysis.",
        ]
    lines = [
        "# Mask Artifact Post-hoc Matching Audit",
        "",
        f"Date: {date}",
        "",
        "This audit consumes a saved `mask_artifacts.npz` file and verifies that",
        "downstream post-hoc matching can be computed without rerunning training.",
        "It performs same-index and minimum-cost record matching across raw and",
        "activation-aligned mask collections, plus optional state RMS distances",
        "when saved state matrices are present.",
        "",
        *evidence_scope,
        "",
        "## Artifact",
        "",
        f"- Path: `{payload['artifact_path']}`",
        f"- Schema version: `{overall['schema_version']}`",
        f"- Dataset/model: `{overall['dataset']}` / `{overall['model']}`",
        f"- Parameters: `{overall['parameter_count']}`",
        f"- Parameter shapes present: `{overall['parameter_shapes_present']}`",
        f"- Mask collections: `{overall['mask_collection_count']}`",
        f"- State collections: `{overall['state_collection_count']}`",
        f"- ResNet channel keys: `{overall['resnet_channel_key_count']}`",
        f"- Record-level post-hoc matching supported: `{overall['record_level_posthoc_matching_supported']}`",
        f"- Local channel-permutation matching supported: `{overall['local_channel_permutation_matching_supported']}`",
        f"- Exhaustive graph/channel permutation supported: `{overall['exhaustive_graph_channel_permutation_supported']}`",
        "",
        "## Collections",
        "",
        "| Collection | Masks | States | Keep frac. |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["collections"]:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['mask_count']} | "
            f"{row['state_count']} | "
            f"{format_float(row['keep_fraction_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Comparison | Same-index Hamming | Optimal Hamming | Improvement | Same-index IoU | Optimal IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["comparisons"]:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{format_float(row['natural_hamming']['mean'])} | "
            f"{format_float(row['optimal_hamming']['mean'])} | "
            f"{format_float(row['hamming_improvement'])} | "
            f"{format_float(row['natural_jaccard_keep']['mean'])} | "
            f"{format_float(row['optimal_jaccard_keep']['mean'])} |"
        )

    if overall["local_channel_permutation_matching_supported"]:
        lines.extend(
            [
                "",
                "## Local Channel Permutation",
                "",
                "| Comparison | Record-optimal Hamming | Channel-permuted Hamming | Improvement |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in payload["comparisons"]:
            channel = row.get("channel_permutation")
            if not isinstance(channel, dict):
                continue
            record_hamming = row["optimal_hamming"]["mean"]
            channel_hamming = channel["channel_optimal_hamming"]["mean"]
            improvement = (
                None
                if record_hamming is None or channel_hamming is None
                else float(record_hamming) - float(channel_hamming)
            )
            lines.append(
                "| "
                f"{row['label']} | "
                f"{format_float(record_hamming)} | "
                f"{format_float(channel_hamming)} | "
                f"{format_float(improvement)} |"
            )

    if payload.get("skipped_comparisons"):
        lines.extend(
            [
                "",
                "## Skipped High-Cost Comparisons",
                "",
                "| Comparison | Left | Right | Pair count | Reason |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in payload["skipped_comparisons"]:
            lines.append(
                "| "
                f"{row['label']} | "
                f"{row['left_count']} | "
                f"{row['right_count']} | "
                f"{row['pair_count']} | "
                f"{row['reason']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The saved artifact is sufficient for record-level post-hoc matching:",
            "the script reads raw masks, aligned masks, record ids, parameter",
            "metadata, and optional state matrices from one `.npz` fixture. This",
            "closes the software-path gap between saving artifacts and analyzing",
            "them.",
        ]
    )
    if overall["local_channel_permutation_matching_supported"]:
        lines.extend(
            [
                "When parameter shapes are present, it also runs a local",
                "ResNet channel-permutation matching objective over saved masks.",
                "That local objective is not a proof of global graph isomorphism.",
            ]
        )
    else:
        lines.extend(
            [
                "Local channel-permutation matching was not run for the retained",
                "comparisons in this invocation because the configured pair-count",
                "cap skipped those expensive searches. The fake-CIFAR audit keeps",
                "the local-channel software path covered; this full-data audit is",
                "record-level evidence plus a saved artifact for later channel",
                "or graph-permutation analysis.",
            ]
        )
    lines.extend(
        [
            "The exhaustive graph/channel permutation objective over full-data",
            "masks or states remains a separate analysis.",
            "",
            "This file is generated by",
            "`scripts/audit_mask_artifact_posthoc_matching.py`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_skip_payload(
    *,
    left_name: str,
    right_name: str,
    label: str,
    artifact: dict[str, Any],
    max_pair_count: int | None,
) -> dict[str, Any] | None:
    masks = artifact["masks"]
    if left_name not in masks or right_name not in masks:
        return None
    left_count = int(masks[left_name].shape[0])
    right_count = int(masks[right_name].shape[0])
    pair_count = int(left_count * right_count)
    if max_pair_count is None or pair_count <= max_pair_count:
        return None
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "left_count": left_count,
        "right_count": right_count,
        "pair_count": pair_count,
        "reason": f"pair count exceeds max_pair_count={max_pair_count}",
    }


def build_payload(
    artifact_path: Path,
    *,
    date: str,
    max_pair_count: int | None = 1024,
    max_channel_pair_count: int | None = 512,
) -> dict[str, Any]:
    artifact = load_artifact(artifact_path)
    channel_key_count = len(resnet_channel_keys(artifact))
    collections = [
        collection_summary(name, artifact)
        for name in sorted(artifact["masks"])
    ]
    comparisons = []
    skipped_comparisons = []
    for left, right, label in DEFAULT_COMPARISONS:
        skip_payload = comparison_skip_payload(
            left_name=left,
            right_name=right,
            label=label,
            artifact=artifact,
            max_pair_count=max_pair_count,
        )
        if skip_payload is not None:
            skipped_comparisons.append(skip_payload)
            continue
        comparison = comparison_payload(
            left,
            right,
            label,
            artifact,
            max_channel_pair_count=max_channel_pair_count,
        )
        if comparison is not None:
            comparisons.append(comparison)
    required_collections = {
        "posterior_sample",
        "ticket",
        "activation_aligned_posterior_sample",
        "activation_aligned_ticket",
    }
    metadata = artifact["metadata"]
    channel_comparison_count = sum(
        1 for row in comparisons if row.get("channel_permutation") is not None
    )
    channel_skipped_count = sum(
        1 for row in comparisons if row.get("channel_permutation_skipped") is True
    )
    if metadata.get("dataset") == "fake-cifar10":
        limitation_statement = (
            "record-level and local channel matching are supported when shapes "
            "are saved, but this fake fixture does not replace the full-data "
            "rerun or an exhaustive graph/channel permutation objective"
        )
    else:
        limitation_statement = (
            "record-level matching is supported when shapes and masks are saved; "
            "local channel matching is bounded by max_channel_pair_count, and "
            "exhaustive graph/channel permutation search still requires a "
            "global permutation objective"
        )
    return {
        "date": date,
        "artifact_path": rel(artifact_path),
        "metadata": metadata,
        "collections": collections,
        "comparisons": comparisons,
        "skipped_comparisons": skipped_comparisons,
        "overall": {
            "schema_version": artifact["schema_version"],
            "dataset": metadata.get("dataset"),
            "model": metadata.get("model"),
            "parameter_count": int(artifact["parameter_sizes"].sum()),
            "parameter_name_count": len(artifact["parameter_names"]),
            "parameter_shapes_present": bool(artifact["has_parameter_shapes"]),
            "mask_collection_count": len(artifact["masks"]),
            "state_collection_count": len(artifact["states"]),
            "resnet_channel_key_count": channel_key_count,
            "required_collections_present": required_collections.issubset(
                set(artifact["masks"])
            ),
            "record_level_posthoc_matching_supported": len(comparisons) >= 4,
            "local_channel_permutation_matching_supported": channel_key_count > 0
            and any(row.get("channel_permutation") is not None for row in comparisons),
            "channel_permutation_comparison_count": channel_comparison_count,
            "channel_permutation_skipped_count": channel_skipped_count,
            "max_pair_count": max_pair_count,
            "max_channel_pair_count": max_channel_pair_count,
            "exhaustive_graph_channel_permutation_supported": False,
            "limitation_statement": limitation_statement,
        },
    }


def main() -> None:
    args = parse_args()
    artifact_path = args.artifact or latest_default_artifact()
    max_pair_count = None if args.max_pair_count <= 0 else int(args.max_pair_count)
    max_channel_pair_count = (
        None if args.max_channel_pair_count <= 0 else int(args.max_channel_pair_count)
    )
    payload = build_payload(
        artifact_path,
        date=args.date,
        max_pair_count=max_pair_count,
        max_channel_pair_count=max_channel_pair_count,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, date=args.date, payload=payload)
    print(
        json.dumps(
            {
                "artifact": payload["artifact_path"],
                "out_json": rel(args.out_json),
                "out_md": rel(args.out_md),
                "comparisons": len(payload["comparisons"]),
                "record_level_posthoc_matching_supported": payload["overall"][
                    "record_level_posthoc_matching_supported"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
