#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imp-rounds", type=int, default=4)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--sgld-chains", type=int, default=3)
    parser.add_argument("--sgld-lr", type=float, default=1e-8)
    parser.add_argument("--sgld-temperature", type=float, default=1.0)
    parser.add_argument("--sgld-steps", type=int, default=200)
    parser.add_argument("--sgld-burn-in", type=int, default=50)
    parser.add_argument("--sgld-sample-every", type=int, default=10)
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
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--barrier-samples", type=int, default=6)
    parser.add_argument("--barrier-points", type=int, default=7)
    parser.add_argument("--snip-batches", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summary-prefix", type=Path, default=None)
    parser.add_argument("--skip-existing-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_completed(out_dir: Path, seed: int) -> bool:
    for path in out_dir.glob("*/metrics.json"):
        try:
            with path.open(encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if int(metrics.get("seed", -1)) == seed:
            return True
    return False


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    out_dir = args.out_dir or Path("runs") / f"{args.dataset}_gate1_sweep"
    summary_prefix = args.summary_prefix or Path("runs") / f"{args.dataset}_gate1_sweep"

    for seed in seeds:
        if args.skip_existing_seeds and seed_completed(out_dir, seed):
            print(f"skip seed {seed}: existing metrics found in {out_dir}", flush=True)
            continue
        cmd = [
            sys.executable,
            "scripts/run_digits_pilot.py",
            "--dataset",
            args.dataset,
            "--model",
            args.model,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--imp-rounds",
            str(args.imp_rounds),
            "--prune-fraction",
            str(args.prune_fraction),
            "--rewind-epochs",
            str(args.rewind_epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lr-schedule",
            args.lr_schedule,
            "--weight-decay",
            str(args.weight_decay),
            "--sgld-chains",
            str(args.sgld_chains),
            "--sgld-chain-init",
            "independent-dense",
            "--posterior-sampler",
            args.posterior_sampler,
            "--sgld-likelihood-scale",
            "dataset",
            "--sgld-lr",
            str(args.sgld_lr),
            "--sgld-temperature",
            str(args.sgld_temperature),
            "--sgld-steps",
            str(args.sgld_steps),
            "--sgld-burn-in",
            str(args.sgld_burn_in),
            "--sgld-sample-every",
            str(args.sgld_sample_every),
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
            "--snip-batches",
            str(args.snip_batches),
            "--barrier-samples",
            str(args.barrier_samples),
            "--barrier-points",
            str(args.barrier_points),
            "--out-dir",
            str(out_dir),
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
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    summary_cmd = [
        sys.executable,
        "scripts/summarize_digits_runs.py",
        "--run-root",
        str(out_dir),
        "--out-csv",
        f"{summary_prefix}_summary.csv",
        "--out-json",
        f"{summary_prefix}_summary.json",
    ]
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_gate1.py",
        f"{summary_prefix}_summary.json",
        "--out-json",
        f"{summary_prefix}_gate1_eval.json",
    ]
    print(" ".join(summary_cmd), flush=True)
    print(" ".join(eval_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(summary_cmd, check=True)
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
