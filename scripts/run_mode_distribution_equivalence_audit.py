#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Observation:
    family: str
    config: str
    scope: str
    seed: int
    run_path: str
    posterior: float
    random: float | None = None
    chain: float | None = None
    dense: float | None = None
    rewind: float | None = None
    post_chain: float | None = None
    sample_accuracy: float | None = None
    raw_posterior: tuple[float, ...] = field(default_factory=tuple)
    raw_random: tuple[float, ...] = field(default_factory=tuple)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "runs" / "mode_distribution_equivalence_audit_summary.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs" / "mode_distribution_equivalence_audit.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "mode_distribution_equivalence_audit.md",
    )
    parser.add_argument(
        "--equivalence-band",
        type=float,
        default=0.005,
        help="Jaccard band used to flag posterior/control practical equivalence.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    center = mean(values)
    if len(values) == 1:
        return center, center, center
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def scalar_mmd_rbf(x_values: list[float], y_values: list[float]) -> float:
    if not x_values or not y_values:
        return math.nan
    x = np.asarray(x_values, dtype=np.float64)[:, None]
    y = np.asarray(y_values, dtype=np.float64)[:, None]
    pooled = np.concatenate([x, y], axis=0)
    dists = np.abs(pooled - pooled.T)
    nonzero = dists[dists > 0]
    bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    if bandwidth <= 0:
        bandwidth = 1.0
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    k_xx = np.exp(-gamma * (x - x.T) ** 2).mean()
    k_yy = np.exp(-gamma * (y - y.T) ** 2).mean()
    k_xy = np.exp(-gamma * (x - y.T) ** 2).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def distribution_stats(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    if len(x_values) < 2 or len(y_values) < 2:
        return {
            "ks_stat": math.nan,
            "ks_pvalue": math.nan,
            "wasserstein": math.nan,
            "mmd_rbf": math.nan,
            "all_pair_win_rate": math.nan,
        }
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    ks = ks_2samp(x, y, alternative="two-sided", method="asymp")
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(x, y)),
        "mmd_rbf": scalar_mmd_rbf(x_values, y_values),
        "all_pair_win_rate": float((x[:, None] > y[None, :]).mean()),
    }


def raw_overlap_values(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    csv_path = path.with_name("mask_overlaps.csv")
    if not csv_path.exists():
        return (), ()
    posterior: list[float] = []
    random: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = to_float(row.get("jaccard"))
            if value is None:
                continue
            source = str(row.get("source"))
            if source == "posterior":
                posterior.append(value)
            elif source == "random":
                random.append(value)
    return tuple(posterior), tuple(random)


def top_level_observations(root_name: str, family: str, config: str) -> list[Observation]:
    root = ROOT / "runs" / root_name
    rows: list[Observation] = []
    for path in sorted(root.glob("*/metrics.json")):
        payload = read_json(path)
        overlap = payload["posterior_mask_overlap"]
        controls = payload.get("controls", {})
        raw_posterior, raw_random = raw_overlap_values(path)
        rows.append(
            Observation(
                family=family,
                config=config,
                scope="global",
                seed=int(payload["seed"]),
                run_path=relpath(path),
                posterior=float(overlap["posterior_jaccard_mean"]),
                random=to_float(overlap.get("random_jaccard_mean")),
                chain=to_float(controls.get("chain_start_magnitude_to_imp_jaccard_mean")),
                dense=to_float(controls.get("dense_magnitude_to_imp_jaccard")),
                rewind=to_float(controls.get("rewind_magnitude_to_imp_jaccard")),
                post_chain=to_float(
                    controls.get("posterior_to_chain_start_magnitude_jaccard_mean")
                ),
                sample_accuracy=to_float(
                    payload.get("posterior", {}).get("sample_accuracy_mean")
                ),
                raw_posterior=raw_posterior,
                raw_random=raw_random,
            )
        )
    return rows


def metric_key(row: dict[str, Any], key: str) -> float | None:
    return to_float(row.get(key))


def row_observations(
    root_name: str,
    family: str,
    config_fn: Callable[[dict[str, Any]], str],
    posterior_key: str = "posterior_jaccard_mean",
    random_key: str = "random_jaccard_mean",
    chain_key: str = "chain_start_magnitude_to_imp_jaccard",
    dense_key: str = "dense_magnitude_to_imp_jaccard",
    rewind_key: str = "rewind_magnitude_to_imp_jaccard",
    post_chain_key: str = "posterior_to_chain_start_magnitude_jaccard_mean",
    scope: str = "global",
) -> list[Observation]:
    root = ROOT / "runs" / root_name
    observations: list[Observation] = []
    for path in sorted(root.glob("*/metrics.json")):
        payload = read_json(path)
        for row in payload.get("rows", []):
            posterior = metric_key(row, posterior_key)
            if posterior is None:
                continue
            observations.append(
                Observation(
                    family=family,
                    config=config_fn(row),
                    scope=scope,
                    seed=int(payload["seed"]),
                    run_path=relpath(path),
                    posterior=posterior,
                    random=metric_key(row, random_key),
                    chain=metric_key(row, chain_key),
                    dense=metric_key(row, dense_key),
                    rewind=metric_key(row, rewind_key),
                    post_chain=metric_key(row, post_chain_key),
                    sample_accuracy=metric_key(row, "sample_accuracy_mean"),
                )
            )
    return observations


def laplace_config(row: dict[str, Any], key: str) -> str:
    return f"scale={float(row[key]):.1e}"


def hmc_config(row: dict[str, Any]) -> str:
    return f"step={float(row['hmc_step_size']):.1e}"


def block_config(row: dict[str, Any]) -> str:
    block = str(row["block_name"])
    if block.startswith("joint:"):
        block = "joint-group"
    else:
        block = block.removesuffix(".weight")
    return f"{block}, scale={float(row['block_laplace_scale']):.1e}"


def collect_observations() -> list[Observation]:
    observations: list[Observation] = []
    observations.extend(
        top_level_observations(
            "mnist_gate1_full_r5_p0p3",
            "MNIST Gate1",
            "r5 p0.30",
        )
    )
    observations.extend(
        top_level_observations(
            "fashion_gate1_full_r5_p0p3",
            "Fashion Gate1",
            "r5 p0.30",
        )
    )
    observations.extend(
        top_level_observations(
            "cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3",
            "CIFAR SGLD-3chain",
            "30ep rewind r5 p0.30",
        )
    )
    observations.extend(
        top_level_observations(
            "cifar10_resnet20_long30_rewind1_swag_r5_p0p3",
            "CIFAR SWAG",
            "30ep rewind r5 p0.30",
        )
    )

    movement_specs = [
        (
            "cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3",
            "CIFAR SGLD movement",
            lambda row: f"lr={float(row['sgld_lr']):.1e}",
        ),
        (
            "cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3",
            "CIFAR SGHMC movement",
            lambda row: f"lr={float(row['sgld_lr']):.1e}",
        ),
        (
            "cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3",
            "CIFAR cSGLD movement",
            lambda row: f"lr={float(row['sgld_lr']):.1e}",
        ),
        (
            "cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3",
            "CIFAR SWAG20 movement",
            lambda row: f"scale={float(row['sgld_lr']):.1e}",
        ),
        (
            "cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3",
            "CIFAR DiagLap movement",
            lambda row: laplace_config(row, "laplace_scale"),
        ),
        (
            "cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3",
            "CIFAR KFACLap movement",
            lambda row: laplace_config(row, "kfac_laplace_scale"),
        ),
        (
            "cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3",
            "CIFAR LowRankLap movement",
            lambda row: laplace_config(row, "lowrank_laplace_scale"),
        ),
        (
            "cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3",
            "CIFAR LowRank32Lap movement",
            lambda row: laplace_config(row, "lowrank_laplace_scale"),
        ),
        (
            "cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3",
            "CIFAR LowRank64Lap movement",
            lambda row: laplace_config(row, "lowrank_laplace_scale"),
        ),
        (
            "cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3",
            "CIFAR LowRank128Lap movement",
            lambda row: laplace_config(row, "lowrank_laplace_scale"),
        ),
    ]
    for root_name, family, config_fn in movement_specs:
        observations.extend(row_observations(root_name, family, config_fn))

    observations.extend(
        row_observations(
            "cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3",
            "CIFAR HeadLap",
            lambda row: laplace_config(row, "head_laplace_scale"),
            posterior_key="head_posterior_jaccard_mean",
            random_key="head_random_jaccard_mean",
            chain_key="head_chain_start_magnitude_to_imp_jaccard",
            dense_key="",
            rewind_key="head_rewind_magnitude_to_imp_jaccard",
            post_chain_key="head_posterior_to_chain_start_magnitude_jaccard_mean",
            scope="head",
        )
    )
    observations.extend(
        row_observations(
            "cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3",
            "CIFAR HeadLap",
            lambda row: laplace_config(row, "head_laplace_scale"),
            posterior_key="global_posterior_jaccard_mean",
            random_key="global_random_jaccard_mean",
            chain_key="global_chain_start_magnitude_to_imp_jaccard",
            dense_key="",
            rewind_key="global_rewind_magnitude_to_imp_jaccard",
            post_chain_key="global_posterior_to_chain_start_magnitude_jaccard_mean",
            scope="global",
        )
    )

    for root_name in [
        "cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3",
        "cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3",
    ]:
        observations.extend(
            row_observations(
                root_name,
                "CIFAR BlockLap",
                block_config,
                posterior_key="block_posterior_jaccard_mean",
                random_key="block_random_jaccard_mean",
                chain_key="block_chain_start_magnitude_to_imp_jaccard",
                dense_key="",
                rewind_key="block_rewind_magnitude_to_imp_jaccard",
                post_chain_key="block_posterior_to_chain_start_magnitude_jaccard_mean",
                scope="block",
            )
        )
        observations.extend(
            row_observations(
                root_name,
                "CIFAR BlockLap",
                block_config,
                posterior_key="global_posterior_jaccard_mean",
                random_key="global_random_jaccard_mean",
                chain_key="global_chain_start_magnitude_to_imp_jaccard",
                dense_key="",
                rewind_key="global_rewind_magnitude_to_imp_jaccard",
                post_chain_key="global_posterior_to_chain_start_magnitude_jaccard_mean",
                scope="global",
            )
        )

    for root_name, family in [
        (
            "cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3",
            "CIFAR RandSubHMC",
        ),
        (
            "cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3",
            "CIFAR TrajSubHMC",
        ),
        (
            "cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3",
            "CIFAR HessSubHMC",
        ),
        (
            "cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3",
            "CIFAR Hess16SubHMC",
        ),
        (
            "cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3",
            "CIFAR Hess32SubHMC",
        ),
    ]:
        observations.extend(row_observations(root_name, family, hmc_config))

    return observations


def verdict(comparison: str, delta: float, win_rate: float, band: float) -> str:
    if comparison == "posterior-random":
        return "posterior separates from random" if delta > 0 else "no random separation"
    if abs(delta) <= band:
        return "practically tied to control"
    if delta > band and win_rate >= 0.8:
        return "posterior closer to ticket"
    if delta < -band and win_rate <= 0.2:
        return "control closer to ticket"
    return "mixed"


def summarize_bucket(
    family: str,
    config: str,
    scope: str,
    observations: list[Observation],
    band: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    posterior = [obs.posterior for obs in observations]
    raw_posterior = [
        value for obs in observations for value in obs.raw_posterior
    ]
    raw_random = [value for obs in observations for value in obs.raw_random]
    for baseline_name in ["random", "chain", "dense", "rewind"]:
        pairs = [
            (obs.posterior, getattr(obs, baseline_name))
            for obs in observations
            if getattr(obs, baseline_name) is not None
        ]
        if not pairs:
            continue
        baseline = [float(value) for _, value in pairs]
        paired_posterior = [float(value) for value, _ in pairs]
        deltas = [left - right for left, right in zip(paired_posterior, baseline)]
        delta_mean, delta_low, delta_high = ci95(deltas)
        dist = distribution_stats(paired_posterior, baseline)
        raw_dist = (
            distribution_stats(raw_posterior, raw_random)
            if baseline_name == "random" and raw_posterior and raw_random
            else {
                "ks_stat": math.nan,
                "ks_pvalue": math.nan,
                "wasserstein": math.nan,
                "mmd_rbf": math.nan,
                "all_pair_win_rate": math.nan,
            }
        )
        pair_win = sum(delta > 0 for delta in deltas) / len(deltas)
        rows.append(
            {
                "family": family,
                "config": config,
                "scope": scope,
                "comparison": f"posterior-{baseline_name}",
                "runs": len({obs.seed for obs in observations}),
                "n_pairs": len(pairs),
                "n_posterior_values": len(posterior),
                "n_baseline_values": len(baseline),
                "posterior_mean": mean(paired_posterior),
                "posterior_std": stdev(paired_posterior) if len(paired_posterior) > 1 else 0.0,
                "baseline_mean": mean(baseline),
                "baseline_std": stdev(baseline) if len(baseline) > 1 else 0.0,
                "delta_mean": delta_mean,
                "delta_ci95_low": delta_low,
                "delta_ci95_high": delta_high,
                "positive": sum(delta > 0 for delta in deltas),
                "negative": sum(delta < 0 for delta in deltas),
                "zero": sum(delta == 0 for delta in deltas),
                "paired_win_rate": pair_win,
                "all_pair_win_rate": dist["all_pair_win_rate"],
                "ks_stat": dist["ks_stat"],
                "ks_pvalue": dist["ks_pvalue"],
                "wasserstein": dist["wasserstein"],
                "mmd_rbf": dist["mmd_rbf"],
                "raw_ks_stat": raw_dist["ks_stat"],
                "raw_ks_pvalue": raw_dist["ks_pvalue"],
                "raw_wasserstein": raw_dist["wasserstein"],
                "raw_mmd_rbf": raw_dist["mmd_rbf"],
                "raw_pair_win_rate": raw_dist["all_pair_win_rate"],
                "mean_post_chain": mean(
                    [obs.post_chain for obs in observations if obs.post_chain is not None]
                )
                if any(obs.post_chain is not None for obs in observations)
                else math.nan,
                "mean_sample_accuracy": mean(
                    [
                        obs.sample_accuracy
                        for obs in observations
                        if obs.sample_accuracy is not None
                    ]
                )
                if any(obs.sample_accuracy is not None for obs in observations)
                else math.nan,
                "verdict": verdict(
                    f"posterior-{baseline_name}",
                    delta_mean,
                    pair_win,
                    band,
                ),
            }
        )
    return rows


def summarize(observations: list[Observation], band: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.family, obs.config, obs.scope)].append(obs)
    rows: list[dict[str, Any]] = []
    for (family, config, scope), group in sorted(grouped.items()):
        rows.extend(summarize_bucket(family, config, scope, group, band))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        ("MNIST Gate1", "r5 p0.30", "global", "posterior-chain"),
        ("Fashion Gate1", "r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SGLD-3chain", "30ep rewind r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SWAG", "30ep rewind r5 p0.30", "global", "posterior-chain"),
        ("CIFAR SGLD movement", "lr=1.0e-06", "global", "posterior-chain"),
        ("CIFAR SGHMC movement", "lr=1.0e-07", "global", "posterior-chain"),
        ("CIFAR cSGLD movement", "lr=1.0e-06", "global", "posterior-chain"),
        ("CIFAR SWAG20 movement", "scale=1.6e+01", "global", "posterior-chain"),
        ("CIFAR DiagLap movement", "scale=1.0e-03", "global", "posterior-chain"),
        ("CIFAR KFACLap movement", "scale=1.0e-04", "global", "posterior-chain"),
        ("CIFAR LowRankLap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR LowRank32Lap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR LowRank64Lap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR LowRank128Lap movement", "scale=1.0e-02", "global", "posterior-chain"),
        ("CIFAR HeadLap", "scale=1.0e-03", "head", "posterior-chain"),
        ("CIFAR BlockLap", "joint-group, scale=1.0e-04", "block", "posterior-chain"),
        ("CIFAR RandSubHMC", "step=3.0e-03", "global", "posterior-chain"),
        ("CIFAR TrajSubHMC", "step=1.0e-03", "global", "posterior-chain"),
        ("CIFAR Hess16SubHMC", "step=3.0e-04", "global", "posterior-chain"),
        ("CIFAR Hess32SubHMC", "step=3.0e-04", "global", "posterior-chain"),
    }
    return [
        row
        for row in rows
        if (
            row["family"],
            row["config"],
            row["scope"],
            row["comparison"],
        )
        in keep
    ]


def build_markdown(rows: list[dict[str, Any]], band: float) -> str:
    critical = selected_rows(rows)
    chain_rows = [row for row in rows if row["comparison"] == "posterior-chain"]
    rewind_rows = [row for row in rows if row["comparison"] == "posterior-rewind"]
    random_rows = [row for row in rows if row["comparison"] == "posterior-random"]

    chain_positive = sum(row["delta_mean"] > band for row in chain_rows)
    chain_tied = sum(abs(row["delta_mean"]) <= band for row in chain_rows)
    chain_negative = sum(row["delta_mean"] < -band for row in chain_rows)
    rewind_negative = sum(row["delta_mean"] < -band for row in rewind_rows)
    random_positive = sum(row["delta_mean"] > 0 for row in random_rows)

    lines = [
        "# Mode/Ticket Distribution Equivalence Audit",
        "",
        "This audit reuses existing posterior artifacts and treats each row as a",
        "distributional support-overlap test against the matching IMP ticket.",
        "A posterior-mode rescue would require posterior-to-IMP overlap to exceed",
        "matched chain-start, dense, or rewind magnitude controls, not merely",
        "uniform random masks.",
        "",
        "Summary:",
        "",
        f"- Posterior beats random in {random_positive}/{len(random_rows)} grouped comparisons.",
        f"- Posterior beats chain-start by more than {band:.3f} Jaccard in {chain_positive}/{len(chain_rows)} grouped comparisons.",
        f"- Posterior is practically tied to chain-start in {chain_tied}/{len(chain_rows)} grouped comparisons.",
        f"- Posterior is below chain-start by more than {band:.3f} Jaccard in {chain_negative}/{len(chain_rows)} grouped comparisons.",
        f"- Rewind magnitude beats posterior by more than {band:.3f} Jaccard in {rewind_negative}/{len(rewind_rows)} grouped comparisons.",
        "",
        "## Critical Posterior-vs-Chain Comparisons",
        "",
        "| Family | Config | Scope | n | Posterior | Chain | Delta | Wins | KS p | W-dist | Post-chain | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in critical:
        lines.append(
            f"| {row['family']} | {row['config']} | {row['scope']} | "
            f"{row['n_pairs']} | {fmt_float(row['posterior_mean'])} | "
            f"{fmt_float(row['baseline_mean'])} | {fmt_float(row['delta_mean'])} | "
            f"{fmt_float(row['paired_win_rate'])} | {fmt_float(row['ks_pvalue'])} | "
            f"{fmt_float(row['wasserstein'])} | {fmt_float(row['mean_post_chain'])} | "
            f"{row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## All Grouped Comparisons",
            "",
            "| Family | Config | Scope | Comparison | n | Posterior | Baseline | Delta | 95% CI | Wins | KS p | W-dist | MMD | Verdict |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        ci = (
            f"[{fmt_float(row['delta_ci95_low'])}, "
            f"{fmt_float(row['delta_ci95_high'])}]"
        )
        lines.append(
            f"| {row['family']} | {row['config']} | {row['scope']} | "
            f"{row['comparison']} | {row['n_pairs']} | "
            f"{fmt_float(row['posterior_mean'])} | "
            f"{fmt_float(row['baseline_mean'])} | "
            f"{fmt_float(row['delta_mean'])} | {ci} | "
            f"{fmt_float(row['paired_win_rate'])} | "
            f"{fmt_float(row['ks_pvalue'])} | "
            f"{fmt_float(row['wasserstein'])} | "
            f"{fmt_float(row['mmd_rbf'])} | {row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "Generated by `scripts/run_mode_distribution_equivalence_audit.py`.",
        ]
    )
    return "\n".join(lines)


def observation_json(observations: list[Observation]) -> list[dict[str, Any]]:
    return [
        {
            "family": obs.family,
            "config": obs.config,
            "scope": obs.scope,
            "seed": obs.seed,
            "run_path": obs.run_path,
            "posterior": obs.posterior,
            "random": obs.random,
            "chain": obs.chain,
            "dense": obs.dense,
            "rewind": obs.rewind,
            "post_chain": obs.post_chain,
            "sample_accuracy": obs.sample_accuracy,
            "raw_posterior_n": len(obs.raw_posterior),
            "raw_random_n": len(obs.raw_random),
        }
        for obs in observations
    ]


def main() -> None:
    args = parse_args()
    observations = collect_observations()
    if not observations:
        raise SystemExit("no mode/ticket audit observations found")
    rows = summarize(observations, args.equivalence_band)
    if not rows:
        raise SystemExit("no mode/ticket audit summaries produced")

    write_csv(args.out_csv, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "equivalence_band": args.equivalence_band,
                "observations": observation_json(observations),
                "summary": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_markdown(rows, args.equivalence_band) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "observations": len(observations),
                "summary_rows": len(rows),
                "out_csv": str(args.out_csv),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
