#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, float | str]] = []
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_gate1_controls(out_dir: Path) -> None:
    datasets = [
        ("MNIST", ROOT / "runs" / "mnist_gate1_full_sweep.csv"),
        ("Fashion-MNIST", ROOT / "runs" / "fashion_gate1_full_sweep.csv"),
    ]
    controls = [
        ("Random", "random", "#b7b7b7"),
        ("Posterior", "posterior", "#4c78a8"),
        ("Chain start", "chain_start", "#f58518"),
        ("Dense mag.", "dense_magnitude", "#54a24b"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (title, path) in zip(axes, datasets, strict=True):
        rows = load_csv(path)
        xs = list(range(len(rows)))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        for offset, (label, key, color) in zip(offsets, controls, strict=True):
            ax.bar(
                [x + offset for x in xs],
                [float(row[key]) for row in rows],
                width=width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{float(row['sparsity']):.2f}" for row in rows])
        ax.set_xlabel("IMP sparsity")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Jaccard to IMP mask")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "gate1_controls.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "gate1_controls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_cifar_movement(out_dir: Path) -> None:
    sources = [
        (
            "SGLD",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv",
            "#4c78a8",
            "o",
        ),
        (
            "SGHMC",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv",
            "#f58518",
            "s",
        ),
        (
            "cSGLD",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv",
            "#e45756",
            "^",
        ),
        (
            "SWAG20",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv",
            "#b279a2",
            "X",
        ),
        (
            "DiagLap",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv",
            "#72b7b2",
            "D",
        ),
        (
            "KFACLap",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv",
            "#54a24b",
            "P",
        ),
        (
            "LowRankLap",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv",
            "#9d755d",
            "*",
        ),
        (
            "LowRank32Lap",
            ROOT
            / "runs"
            / "cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv",
            "#ff9da6",
            "v",
        ),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    all_chain = []
    all_rewind = []
    for label, path, color, marker in sources:
        rows = load_csv(path)
        xs = [float(row["posterior_to_chain_start_magnitude_jaccard_mean"]) for row in rows]
        ys = [float(row["posterior_jaccard_mean"]) for row in rows]
        ax.plot(xs, ys, marker=marker, markersize=4, linewidth=1.5, label=label, color=color)
        all_chain.extend(float(row["chain_start_magnitude_to_imp_jaccard"]) for row in rows)
        all_rewind.extend(float(row["rewind_magnitude_to_imp_jaccard"]) for row in rows)

    chain = sum(all_chain) / len(all_chain)
    rewind = sum(all_rewind) / len(all_rewind)
    ax.axhline(chain, color="#666666", linewidth=1.0, linestyle="--", label="Chain-start mean")
    ax.axhline(rewind, color="#222222", linewidth=1.0, linestyle=":", label="Rewind mean")
    ax.set_xlabel("Jaccard to chain-start support")
    ax.set_ylabel("Jaccard to IMP mask")
    ax.set_xlim(0.32, 1.02)
    ax.set_ylim(0.118, 0.184)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / "cifar_movement.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cifar_movement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_cifar_trajectory(out_dir: Path) -> None:
    trajectory_rows = load_csv(
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_summary.csv"
    )
    aggregate_rows = load_csv(
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_aggregate_summary.csv"
    )
    movement_paths = [
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv",
        ROOT
        / "runs"
        / "cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv",
    ]
    posterior_values = []
    for path in movement_paths:
        posterior_values.extend(float(row["posterior_jaccard_mean"]) for row in load_csv(path))
    posterior_max = max(posterior_values)

    xs = [int(float(row["epoch"])) for row in trajectory_rows]
    ys = [float(row["trajectory_magnitude_to_imp_jaccard"]) for row in trajectory_rows]
    ci_low = [
        float(row["trajectory_magnitude_to_imp_jaccard_ci95_low"])
        for row in trajectory_rows
    ]
    ci_high = [
        float(row["trajectory_magnitude_to_imp_jaccard_ci95_high"])
        for row in trajectory_rows
    ]
    dense = float(trajectory_rows[-1]["dense_magnitude_to_imp_jaccard"])
    rewind = float(trajectory_rows[1]["rewind_magnitude_to_imp_jaccard"])
    best_aggregate = max(
        aggregate_rows,
        key=lambda row: float(row["trajectory_score_to_imp_jaccard"]),
    )
    aggregate_value = float(best_aggregate["trajectory_score_to_imp_jaccard"])

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, color="#4c78a8", label="Trajectory")
    ax.fill_between(xs, ci_low, ci_high, color="#4c78a8", alpha=0.16, linewidth=0)
    ax.axhline(
        aggregate_value,
        color="#b279a2",
        linewidth=1.0,
        linestyle=(0, (4, 1.5)),
        label="Best aggregate",
    )
    ax.axhline(dense, color="#54a24b", linewidth=1.0, linestyle="--", label="Dense final")
    ax.axhline(rewind, color="#222222", linewidth=1.0, linestyle=":", label="Epoch-1 rewind")
    ax.axhline(
        posterior_max,
        color="#e45756",
        linewidth=1.0,
        linestyle="-.",
        label="Best posterior",
    )
    ax.set_xlabel("Dense trajectory epoch")
    ax.set_ylabel("Jaccard to IMP mask")
    ax.set_xticks(xs)
    ax.set_ylim(0.13, 0.262)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "cifar_trajectory.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cifar_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    build_gate1_controls(args.out_dir)
    build_cifar_movement(args.out_dir)
    build_cifar_trajectory(args.out_dir)
    print(f"wrote {args.out_dir / 'gate1_controls.pdf'}")
    print(f"wrote {args.out_dir / 'cifar_movement.pdf'}")
    print(f"wrote {args.out_dir / 'cifar_trajectory.pdf'}")


if __name__ == "__main__":
    main()
