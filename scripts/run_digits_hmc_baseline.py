#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.analysis import cluster_matrix, cluster_states, overlap_rows, summarize_overlaps
from lottery.data import load_digits_bundle
from lottery.hmc import HMCConfig, collect_hmc_samples
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import global_magnitude_mask_from_state, support_jaccard
from lottery.models import MLP, weight_parameter_names
from lottery.posterior_maps import posterior_score_masks
from lottery.train import (
    evaluate,
    load_trainable_state,
    logits_matrix,
    predictions,
    set_seed,
    state_to_cpu,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imp-rounds", type=int, default=2)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--hmc-steps", type=int, default=40)
    parser.add_argument("--hmc-step-size", type=float, default=2e-4)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=5)
    parser.add_argument("--hmc-prior-precision", type=float, default=1e-4)
    parser.add_argument("--hmc-burn-in", type=int, default=10)
    parser.add_argument("--hmc-sample-every", type=int, default=3)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/digits_hmc_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_digits_bundle(args.batch_size, 1024, args.seed)

    def model_factory() -> MLP:
        return MLP(
            input_dim=bundle.input_dim,
            num_classes=bundle.num_classes,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )

    initial_model = model_factory()
    initial_state = state_to_cpu(initial_model)

    dense_model = model_factory()
    load_trainable_state(dense_model, initial_state)
    train_model(
        dense_model,
        bundle.train_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    dense_metrics = evaluate(dense_model, bundle.test_loader, device)
    dense_pred = predictions(dense_model, bundle.test_loader, device)
    dense_state = state_to_cpu(dense_model)

    imp = iterative_magnitude_pruning(
        model_factory=model_factory,
        initial_state=initial_state,
        train_loader=bundle.train_loader,
        test_loader=bundle.test_loader,
        device=device,
        rounds=args.imp_rounds,
        prune_fraction_per_round=args.prune_fraction,
        epochs_per_round=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    imp_model = model_factory()
    load_trainable_state(imp_model, imp.final_state)
    imp_pred = predictions(imp_model, bundle.test_loader, device)

    hmc_model = model_factory()
    load_trainable_state(hmc_model, dense_state)
    hmc = collect_hmc_samples(
        hmc_model,
        bundle.train_loader,
        device,
        HMCConfig(
            steps=args.hmc_steps,
            step_size=args.hmc_step_size,
            leapfrog_steps=args.hmc_leapfrog_steps,
            prior_precision=args.hmc_prior_precision,
            burn_in=args.hmc_burn_in,
            sample_every=args.hmc_sample_every,
        ),
    )
    samples = hmc.samples
    names = weight_parameter_names(model_factory())
    posterior_masks = [
        global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
        for sample in samples
    ]
    dense_magnitude_mask = global_magnitude_mask_from_state(
        dense_state,
        names,
        imp.metrics["sparsity"],
    )
    initial_magnitude_mask = global_magnitude_mask_from_state(
        initial_state,
        names,
        imp.metrics["sparsity"],
    )
    rows = overlap_rows(
        posterior_masks,
        imp.mask,
        sparsity=imp.metrics["sparsity"],
        random_trials=args.random_trials,
        seed=args.seed + 20_000,
    )
    posterior_maps = posterior_score_masks(samples, names, imp.metrics["sparsity"])
    posterior_map_jaccards = {
        map_name: support_jaccard(mask, imp.mask) for map_name, mask in posterior_maps.items()
    }
    posterior_dense_jaccards = [
        support_jaccard(mask, dense_magnitude_mask) for mask in posterior_masks
    ]
    sample_accuracies = []
    sample_dense_agreements = []
    sample_imp_agreements = []
    sample_logit_features = []
    for sample in samples:
        sample_model = model_factory()
        load_trainable_state(sample_model, sample)
        sample_metrics = evaluate(sample_model, bundle.test_loader, device)
        sample_pred = predictions(sample_model, bundle.test_loader, device)
        sample_logits = logits_matrix(sample_model, bundle.test_loader, device)
        sample_accuracies.append(sample_metrics["accuracy"])
        sample_dense_agreements.append((sample_pred == dense_pred).float().mean().item())
        sample_imp_agreements.append((sample_pred == imp_pred).float().mean().item())
        sample_logit_features.append(sample_logits.flatten().numpy())
    function_clustering = (
        cluster_matrix(np.stack(sample_logit_features, axis=0))
        if sample_logit_features
        else {"num_clusters": 0.0, "noise_fraction": 0.0}
    )
    metrics = {
        "seed": args.seed,
        "dataset": "digits",
        "model": "mlp",
        "dense": dense_metrics,
        "imp": imp.metrics,
        "hmc": {
            "num_samples": len(samples),
            "accept_rate": hmc.accept_rate,
            "energy_first": hmc.energies[0] if hmc.energies else None,
            "energy_last": hmc.energies[-1] if hmc.energies else None,
            "step_size": args.hmc_step_size,
            "leapfrog_steps": args.hmc_leapfrog_steps,
            "sample_accuracy_mean": float(np.mean(sample_accuracies)) if sample_accuracies else 0.0,
            "sample_accuracy_std": float(np.std(sample_accuracies, ddof=1))
            if len(sample_accuracies) > 1
            else 0.0,
            "sample_to_dense_prediction_agreement_mean": float(np.mean(sample_dense_agreements))
            if sample_dense_agreements
            else 0.0,
            "sample_to_imp_prediction_agreement_mean": float(np.mean(sample_imp_agreements))
            if sample_imp_agreements
            else 0.0,
        },
        "hmc_mask_overlap": summarize_overlaps(rows),
        "controls": {
            "dense_magnitude_to_imp_jaccard": support_jaccard(dense_magnitude_mask, imp.mask),
            "initial_magnitude_to_imp_jaccard": support_jaccard(initial_magnitude_mask, imp.mask),
            "hmc_to_dense_magnitude_jaccard_mean": float(np.mean(posterior_dense_jaccards)),
            "chain_start_magnitude_to_imp_jaccard_mean": support_jaccard(
                dense_magnitude_mask,
                imp.mask,
            ),
        },
        "posterior_map_jaccards": posterior_map_jaccards,
        "state_clustering": cluster_states(samples, names),
        "function_clustering": function_clustering,
        "dense_imp_prediction_agreement": (dense_pred == imp_pred).float().mean().item(),
    }

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
