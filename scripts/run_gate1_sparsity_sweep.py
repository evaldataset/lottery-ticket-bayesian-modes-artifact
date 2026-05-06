#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_configs(text: str) -> list[tuple[int, float]]:
    configs = []
    for item in text.split(","):
        if not item:
            continue
        rounds_text, prune_text = item.split(":")
        configs.append((int(rounds_text), float(prune_text)))
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="mlp")
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Prefix for run directories and summaries. Defaults to "
            "'<dataset>_gate1'. Use this to keep smoke and full sweeps separate."
        ),
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--configs",
        default="2:0.30,3:0.30,5:0.30,8:0.30",
        help="Comma list of imp_rounds:prune_fraction.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--sgld-chains", type=int, default=3)
    parser.add_argument("--sgld-lr", type=float, default=1e-8)
    parser.add_argument("--sgld-steps", type=int, default=600)
    parser.add_argument("--sgld-burn-in", type=int, default=200)
    parser.add_argument("--sgld-sample-every", type=int, default=20)
    parser.add_argument(
        "--posterior-sampler",
        choices=["sgld", "sghmc", "cyclical-sgld", "swag"],
        default="sgld",
    )
    parser.add_argument("--sghmc-lr", type=float, default=None)
    parser.add_argument("--sghmc-momentum-decay", type=float, default=0.9)
    parser.add_argument("--sghmc-temperature", type=float, default=None)
    parser.add_argument("--sghmc-prior-precision", type=float, default=None)
    parser.add_argument("--csgld-lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--csgld-cycle-length", type=int, default=50)
    parser.add_argument("--csgld-sample-phase-start", type=float, default=0.0)
    parser.add_argument("--swag-epochs", type=int, default=5)
    parser.add_argument("--swag-lr", type=float, default=0.01)
    parser.add_argument("--swag-weight-decay", type=float, default=None)
    parser.add_argument("--swag-collection-start-epoch", type=int, default=1)
    parser.add_argument("--swag-sample-every-epochs", type=int, default=1)
    parser.add_argument("--swag-max-snapshots", type=int, default=20)
    parser.add_argument("--swag-scale", type=float, default=1.0)
    parser.add_argument("--swag-diagonal-scale", type=float, default=1.0)
    parser.add_argument("--swag-low-rank-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--barrier-samples", type=int, default=20)
    parser.add_argument("--barrier-points", type=int, default=11)
    parser.add_argument("--skip-existing-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_prefix = args.run_prefix or f"{args.dataset}_gate1"
    for rounds, prune_fraction in parse_configs(args.configs):
        tag = f"r{rounds}_p{str(prune_fraction).replace('.', 'p')}"
        out_dir = Path("runs") / f"{run_prefix}_{tag}"
        summary_prefix = Path("runs") / f"{run_prefix}_{tag}"
        cmd = [
            sys.executable,
            "scripts/run_gate1_sweep.py",
            "--dataset",
            args.dataset,
            "--model",
            args.model,
            "--seeds",
            args.seeds,
            "--epochs",
            str(args.epochs),
            "--rewind-epochs",
            str(args.rewind_epochs),
            "--imp-rounds",
            str(rounds),
            "--prune-fraction",
            str(prune_fraction),
            "--hidden-dim",
            str(args.hidden_dim),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lr-schedule",
            args.lr_schedule,
            "--sgld-chains",
            str(args.sgld_chains),
            "--sgld-lr",
            str(args.sgld_lr),
            "--sgld-steps",
            str(args.sgld_steps),
            "--sgld-burn-in",
            str(args.sgld_burn_in),
            "--sgld-sample-every",
            str(args.sgld_sample_every),
            "--posterior-sampler",
            args.posterior_sampler,
            "--sghmc-momentum-decay",
            str(args.sghmc_momentum_decay),
            "--csgld-lr-min-ratio",
            str(args.csgld_lr_min_ratio),
            "--csgld-cycle-length",
            str(args.csgld_cycle_length),
            "--csgld-sample-phase-start",
            str(args.csgld_sample_phase_start),
            "--swag-epochs",
            str(args.swag_epochs),
            "--swag-lr",
            str(args.swag_lr),
            "--swag-collection-start-epoch",
            str(args.swag_collection_start_epoch),
            "--swag-sample-every-epochs",
            str(args.swag_sample_every_epochs),
            "--swag-max-snapshots",
            str(args.swag_max_snapshots),
            "--swag-scale",
            str(args.swag_scale),
            "--swag-diagonal-scale",
            str(args.swag_diagonal_scale),
            "--swag-low-rank-scale",
            str(args.swag_low_rank_scale),
            "--samples",
            str(args.samples),
            "--random-trials",
            str(args.random_trials),
            "--barrier-samples",
            str(args.barrier_samples),
            "--barrier-points",
            str(args.barrier_points),
            "--out-dir",
            str(out_dir),
            "--summary-prefix",
            str(summary_prefix),
        ]
        if args.imp_epochs is not None:
            cmd.extend(["--imp-epochs", str(args.imp_epochs)])
        if args.imp_final_epochs is not None:
            cmd.extend(["--imp-final-epochs", str(args.imp_final_epochs)])
        if args.sghmc_lr is not None:
            cmd.extend(["--sghmc-lr", str(args.sghmc_lr)])
        if args.sghmc_temperature is not None:
            cmd.extend(["--sghmc-temperature", str(args.sghmc_temperature)])
        if args.sghmc_prior_precision is not None:
            cmd.extend(["--sghmc-prior-precision", str(args.sghmc_prior_precision)])
        if args.swag_weight_decay is not None:
            cmd.extend(["--swag-weight-decay", str(args.swag_weight_decay)])
        if args.augment:
            cmd.append("--augment")
        if args.skip_existing_seeds:
            cmd.append("--skip-existing-seeds")
        if args.dry_run:
            cmd.append("--dry-run")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
