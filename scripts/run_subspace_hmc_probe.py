#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lottery.analysis import cluster_matrix, cluster_states, overlap_rows, summarize_overlaps
from lottery.data import DatasetBundle, load_digits_bundle, load_fake_cifar10_bundle, load_torchvision_bundle
from lottery.imp import iterative_magnitude_pruning
from lottery.masks import global_magnitude_mask_from_state, support_jaccard
from lottery.models import MLP, ResNetCIFAR, TinyCNN, weight_parameter_names
from lottery.posterior_maps import posterior_score_masks
from lottery.pruning_baselines import snip_mask, synflow_mask
from lottery.subspace_hmc import SubspaceHMCConfig, collect_subspace_hmc_samples
from lottery.train import (
    evaluate,
    load_trainable_state,
    logits_matrix,
    predictions,
    set_seed,
    state_to_cpu,
    train_model,
)


def parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def sample_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def finite_or_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        choices=["digits", "mnist", "fashion-mnist", "cifar10", "fake-cifar10"],
        default="cifar10",
    )
    parser.add_argument("--model", choices=["mlp", "tiny-cnn", "resnet20"], default="resnet20")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--resnet-width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imp-epochs", type=int, default=None)
    parser.add_argument("--imp-final-epochs", type=int, default=None)
    parser.add_argument("--rewind-epochs", type=int, default=0)
    parser.add_argument("--imp-rounds", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--posterior-augment",
        action="store_true",
        help="Use stochastic training augmentation inside the HMC potential.",
    )
    parser.add_argument("--subspace-dims", default="8")
    parser.add_argument("--hmc-step-sizes", default="1e-4,3e-4,1e-3")
    parser.add_argument("--hmc-steps", type=int, default=40)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=3)
    parser.add_argument("--hmc-prior-precision", type=float, default=1e-4)
    parser.add_argument("--hmc-burn-in", type=int, default=10)
    parser.add_argument("--hmc-sample-every", type=int, default=5)
    parser.add_argument("--hmc-direction-scale", type=float, default=1.0)
    parser.add_argument("--hmc-batchnorm-mode", choices=["eval", "train"], default="eval")
    parser.add_argument(
        "--hmc-basis",
        choices=["random", "trajectory", "hessian"],
        default="random",
    )
    parser.add_argument("--hmc-trajectory-epochs", default="0,1,2,5,10,20,30")
    parser.add_argument("--hessian-batches", type=int, default=5)
    parser.add_argument("--hessian-power-iterations", type=int, default=1)
    parser.add_argument("--hessian-oversample", type=int, default=2)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--snip-batches", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/subspace_hmc_probe"))
    return parser.parse_args()


def load_bundle(args: argparse.Namespace, augment: bool) -> DatasetBundle:
    if args.dataset == "digits":
        return load_digits_bundle(args.batch_size, 1024, args.seed)
    if args.dataset == "fake-cifar10":
        return load_fake_cifar10_bundle(
            args.batch_size,
            1024,
            args.seed,
            train_size=args.train_subset or 2048,
            test_size=args.test_subset or 512,
        )
    return load_torchvision_bundle(
        args.dataset,
        args.batch_size,
        1024,
        args.seed,
        flatten=args.model == "mlp",
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        augment=augment,
    )


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def train_dense_with_checkpoints(
    model: torch.nn.Module,
    initial_state: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    checkpoint_epochs: list[int],
    lr: float,
    weight_decay: float,
    lr_schedule: str,
) -> tuple[
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, float]],
]:
    load_trainable_state(model, initial_state)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    if lr_schedule == "constant":
        scheduler = None
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")

    wanted = sorted(set(checkpoint_epochs + [0, epochs]))
    states: dict[int, dict[str, torch.Tensor]] = {}
    metrics: dict[int, dict[str, float]] = {}
    if 0 in wanted:
        states[0] = state_to_cpu(model)
        metrics[0] = evaluate(model, test_loader, device)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, device, optimizer)
        if scheduler is not None:
            scheduler.step()
        if epoch in wanted:
            metrics[epoch] = evaluate(model, test_loader, device)
            states[epoch] = state_to_cpu(model)
    return states, metrics


def flatten_trainable_state(
    state: dict[str, torch.Tensor],
    names: list[str],
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [state[name].detach().to(device=device, dtype=torch.float32).reshape(-1) for name in names]
    )


def flatten_tensors(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def trajectory_basis_directions(
    states: dict[int, dict[str, torch.Tensor]],
    dense_state: dict[str, torch.Tensor],
    parameter_names: list[str],
    epochs: list[int],
    device: torch.device,
) -> torch.Tensor:
    base = flatten_trainable_state(dense_state, parameter_names, device)
    columns = []
    for epoch in epochs:
        if epoch not in states:
            continue
        direction = flatten_trainable_state(states[epoch], parameter_names, device) - base
        if float(direction.norm().item()) > 0.0:
            columns.append(direction)
    if not columns:
        raise ValueError("trajectory basis has no nonzero checkpoint directions")
    return torch.stack(columns, dim=1)


def set_curvature_batchnorm_mode(model: torch.nn.Module, mode: str) -> None:
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(f"unsupported batchnorm mode: {mode}")


def hessian_vector_product(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    vector: torch.Tensor,
    max_batches: int,
    batchnorm_mode: str,
) -> torch.Tensor:
    if max_batches <= 0:
        raise ValueError("hessian-batches must be positive")
    params = [param for param in model.parameters() if param.requires_grad]
    if vector.numel() != sum(param.numel() for param in params):
        raise ValueError("Hessian-vector product has mismatched parameter count")

    vector = vector.detach().to(device=device, dtype=torch.float32)
    result = torch.zeros_like(vector)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    examples_seen = 0
    batches_seen = 0
    for x, y in train_loader:
        set_curvature_batchnorm_mode(model, batchnorm_mode)
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = flatten_tensors(grads)
        grad_dot_vector = torch.dot(flat_grad, vector)
        hvp = torch.autograd.grad(grad_dot_vector, params)
        result.add_(flatten_tensors(hvp).detach())
        examples_seen += int(y.numel())
        batches_seen += 1
        if batches_seen >= max_batches:
            break
    if examples_seen == 0:
        raise RuntimeError("no batches available for Hessian-vector products")
    return result.div(float(examples_seen))


def hessian_basis_directions(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    subspace_dim: int,
    max_batches: int,
    power_iterations: int,
    oversample: int,
    seed: int,
    batchnorm_mode: str,
) -> torch.Tensor:
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive")
    if power_iterations < 0:
        raise ValueError("hessian-power-iterations must be non-negative")
    if oversample < 0:
        raise ValueError("hessian-oversample must be non-negative")

    model.to(device)
    parameter_count = flatten_tensors(
        tuple(param.detach() for param in model.parameters() if param.requires_grad)
    ).numel()
    probe_dim = min(parameter_count, subspace_dim + oversample)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        parameter_count,
        probe_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    q, _ = torch.linalg.qr(q, mode="reduced")

    for _ in range(power_iterations):
        hq = torch.stack(
            [
                hessian_vector_product(
                    model,
                    train_loader,
                    device,
                    q[:, column],
                    max_batches=max_batches,
                    batchnorm_mode=batchnorm_mode,
                )
                for column in range(q.shape[1])
            ],
            dim=1,
        )
        q, _ = torch.linalg.qr(hq, mode="reduced")

    hq = torch.stack(
        [
            hessian_vector_product(
                model,
                train_loader,
                device,
                q[:, column],
                max_batches=max_batches,
                batchnorm_mode=batchnorm_mode,
            )
            for column in range(q.shape[1])
        ],
        dim=1,
    )
    projected = q.t().matmul(hq)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (projected + projected.t()))
    order = torch.argsort(eigenvalues, descending=True)
    directions = q.matmul(eigenvectors[:, order[:subspace_dim]])
    directions, _ = torch.linalg.qr(directions, mode="reduced")
    return directions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bundle = load_bundle(args, augment=args.augment)
    posterior_bundle = (
        train_bundle
        if args.posterior_augment == args.augment
        else load_bundle(args, augment=args.posterior_augment)
    )

    def model_factory() -> torch.nn.Module:
        if args.model == "mlp":
            return MLP(
                input_dim=train_bundle.input_dim,
                num_classes=train_bundle.num_classes,
                hidden_dim=args.hidden_dim,
                depth=args.depth,
            )
        if args.model == "tiny-cnn":
            return TinyCNN(
                input_shape=train_bundle.input_shape,
                num_classes=train_bundle.num_classes,
                width=args.cnn_width,
            )
        if args.model == "resnet20":
            return ResNetCIFAR(
                input_shape=train_bundle.input_shape,
                num_classes=train_bundle.num_classes,
                blocks_per_stage=3,
                width=args.resnet_width,
            )
        raise ValueError(f"Unsupported model: {args.model}")

    initial_model = model_factory()
    initial_state = state_to_cpu(initial_model)
    trajectory_states: dict[int, dict[str, torch.Tensor]] = {}
    trajectory_epochs = sorted(
        set(parse_int_list(args.hmc_trajectory_epochs) + [0, args.epochs])
    )
    if args.rewind_epochs > 0:
        trajectory_epochs = sorted(set(trajectory_epochs + [args.rewind_epochs]))
    if any(epoch < 0 or epoch > args.epochs for epoch in trajectory_epochs):
        raise ValueError("hmc trajectory epochs must be in [0, epochs]")

    rewind_state = None
    rewind_metrics = None
    dense_model = model_factory()
    if args.hmc_basis == "trajectory":
        trajectory_states, checkpoint_metrics = train_dense_with_checkpoints(
            dense_model,
            initial_state,
            train_bundle.train_loader,
            train_bundle.test_loader,
            device,
            epochs=args.epochs,
            checkpoint_epochs=trajectory_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
        )
        dense_state = trajectory_states[args.epochs]
        dense_metrics = checkpoint_metrics[args.epochs]
        if args.rewind_epochs > 0:
            rewind_state = trajectory_states[args.rewind_epochs]
            rewind_metrics = checkpoint_metrics[args.rewind_epochs]
    else:
        if args.rewind_epochs > 0:
            rewind_model = model_factory()
            load_trainable_state(rewind_model, initial_state)
            train_model(
                rewind_model,
                train_bundle.train_loader,
                device,
                epochs=args.rewind_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_schedule=args.lr_schedule,
                lr_schedule_epochs=args.epochs,
            )
            rewind_metrics = evaluate(rewind_model, train_bundle.test_loader, device)
            rewind_state = state_to_cpu(rewind_model)

        load_trainable_state(dense_model, initial_state)
        train_model(
            dense_model,
            train_bundle.train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
        )
        dense_metrics = evaluate(dense_model, train_bundle.test_loader, device)
        dense_state = state_to_cpu(dense_model)
    dense_pred = predictions(dense_model, train_bundle.test_loader, device)

    imp_epochs = args.epochs if args.imp_epochs is None else args.imp_epochs
    imp = iterative_magnitude_pruning(
        model_factory=model_factory,
        initial_state=initial_state,
        train_loader=train_bundle.train_loader,
        test_loader=train_bundle.test_loader,
        device=device,
        rounds=args.imp_rounds,
        prune_fraction_per_round=args.prune_fraction,
        epochs_per_round=imp_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        rewind_state=rewind_state,
        final_epochs=args.imp_final_epochs,
    )
    imp_model = model_factory()
    load_trainable_state(imp_model, imp.final_state)
    imp_pred = predictions(imp_model, train_bundle.test_loader, device)

    names = weight_parameter_names(model_factory())
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
    rewind_magnitude_mask = (
        global_magnitude_mask_from_state(rewind_state, names, imp.metrics["sparsity"])
        if rewind_state is not None
        else initial_magnitude_mask
    )
    snip = snip_mask(
        model_factory(),
        initial_state,
        train_bundle.train_loader,
        device,
        sparsity=imp.metrics["sparsity"],
        max_batches=args.snip_batches,
    )
    synflow = synflow_mask(
        model_factory(),
        initial_state,
        train_bundle.input_shape,
        device,
        sparsity=imp.metrics["sparsity"],
    )

    rows = []
    subspace_dims = parse_int_list(args.subspace_dims)
    step_sizes = parse_float_list(args.hmc_step_sizes)
    basis_directions = None
    if args.hmc_basis == "trajectory":
        parameter_names = [name for name, _ in model_factory().named_parameters()]
        basis_epochs = [epoch for epoch in trajectory_epochs if epoch != args.epochs]
        basis_directions = trajectory_basis_directions(
            trajectory_states,
            dense_state,
            parameter_names,
            basis_epochs,
            device,
        )
    hessian_direction_cache: dict[int, torch.Tensor] = {}
    for subspace_dim in subspace_dims:
        active_directions = basis_directions
        if args.hmc_basis == "hessian":
            if subspace_dim not in hessian_direction_cache:
                hessian_model = model_factory()
                load_trainable_state(hessian_model, dense_state)
                hessian_direction_cache[subspace_dim] = hessian_basis_directions(
                    hessian_model,
                    posterior_bundle.train_loader,
                    device,
                    subspace_dim=subspace_dim,
                    max_batches=args.hessian_batches,
                    power_iterations=args.hessian_power_iterations,
                    oversample=args.hessian_oversample,
                    seed=args.seed + 303_000 + subspace_dim,
                    batchnorm_mode=args.hmc_batchnorm_mode,
                )
            active_directions = hessian_direction_cache[subspace_dim]
        for step_size in step_sizes:
            hmc_model = model_factory()
            load_trainable_state(hmc_model, dense_state)
            result = collect_subspace_hmc_samples(
                hmc_model,
                posterior_bundle.train_loader,
                device,
                SubspaceHMCConfig(
                    steps=args.hmc_steps,
                    step_size=step_size,
                    leapfrog_steps=args.hmc_leapfrog_steps,
                    prior_precision=args.hmc_prior_precision,
                    burn_in=args.hmc_burn_in,
                    sample_every=args.hmc_sample_every,
                    subspace_dim=subspace_dim,
                    direction_seed=args.seed + 101_000 + subspace_dim,
                    direction_scale=args.hmc_direction_scale,
                    batchnorm_mode=args.hmc_batchnorm_mode,
                ),
                directions=active_directions,
            )
            samples = result.samples
            posterior_masks = [
                global_magnitude_mask_from_state(sample, names, imp.metrics["sparsity"])
                for sample in samples
            ]
            overlap = summarize_overlaps(
                overlap_rows(
                    posterior_masks,
                    imp.mask,
                    sparsity=imp.metrics["sparsity"],
                    random_trials=args.random_trials,
                    seed=args.seed + 202_000 + int(step_size * 1e12) + subspace_dim,
                )
            )
            posterior_chain = [
                support_jaccard(mask, dense_magnitude_mask) for mask in posterior_masks
            ]
            posterior_maps = posterior_score_masks(samples, names, imp.metrics["sparsity"])
            sample_accuracies = []
            sample_dense_agreements = []
            sample_imp_agreements = []
            sample_logit_features = []
            for sample in samples:
                sample_model = model_factory()
                load_trainable_state(sample_model, sample)
                sample_metrics = evaluate(sample_model, train_bundle.test_loader, device)
                sample_pred = predictions(sample_model, train_bundle.test_loader, device)
                sample_logits = logits_matrix(sample_model, train_bundle.test_loader, device)
                sample_accuracies.append(sample_metrics["accuracy"])
                sample_dense_agreements.append(
                    (sample_pred == dense_pred).float().mean().item()
                )
                sample_imp_agreements.append(
                    (sample_pred == imp_pred).float().mean().item()
                )
                sample_logit_features.append(sample_logits.flatten().numpy())
            state_clustering = cluster_states(samples, names)
            function_clustering = (
                cluster_matrix(np.stack(sample_logit_features, axis=0))
                if sample_logit_features
                else {"num_clusters": 0.0, "largest_cluster_fraction": 0.0}
            )
            row = {
                "hmc_basis": args.hmc_basis,
                "subspace_dim": subspace_dim,
                "hmc_step_size": step_size,
                "hmc_direction_scale": args.hmc_direction_scale,
                "hmc_basis_vectors": (
                    int(active_directions.shape[1]) if active_directions is not None else None
                ),
                "hessian_batches": (
                    args.hessian_batches if args.hmc_basis == "hessian" else None
                ),
                "hessian_power_iterations": (
                    args.hessian_power_iterations if args.hmc_basis == "hessian" else None
                ),
                "hessian_oversample": (
                    args.hessian_oversample if args.hmc_basis == "hessian" else None
                ),
                "hmc_accept_rate": result.accept_rate,
                "hmc_energy_first": result.energies[0] if result.energies else None,
                "hmc_energy_last": result.energies[-1] if result.energies else None,
                "hmc_coordinate_norm_mean": sample_mean(result.coordinate_norms),
                "hmc_coordinate_norm_std": sample_std(result.coordinate_norms),
                "hmc_parameter_distance_mean": sample_mean(result.parameter_distances),
                "hmc_parameter_distance_std": sample_std(result.parameter_distances),
                "num_samples": len(samples),
                "dense_accuracy": dense_metrics["accuracy"],
                "imp_accuracy": imp.metrics["accuracy"],
                "rewind_accuracy": (
                    rewind_metrics["accuracy"] if rewind_metrics is not None else None
                ),
                "sparsity": imp.metrics["sparsity"],
                "posterior_jaccard_mean": overlap["posterior_jaccard_mean"],
                "posterior_jaccard_std": overlap["posterior_jaccard_std"],
                "random_jaccard_mean": overlap["random_jaccard_mean"],
                "posterior_minus_random_jaccard": overlap[
                    "posterior_minus_random_jaccard"
                ],
                "chain_start_magnitude_to_imp_jaccard": support_jaccard(
                    dense_magnitude_mask,
                    imp.mask,
                ),
                "posterior_minus_chain_start_jaccard": (
                    overlap["posterior_jaccard_mean"]
                    - support_jaccard(dense_magnitude_mask, imp.mask)
                ),
                "posterior_to_chain_start_magnitude_jaccard_mean": sample_mean(
                    posterior_chain
                ),
                "posterior_to_chain_start_magnitude_jaccard_std": sample_std(
                    posterior_chain
                ),
                "dense_magnitude_to_imp_jaccard": support_jaccard(
                    dense_magnitude_mask,
                    imp.mask,
                ),
                "initial_magnitude_to_imp_jaccard": support_jaccard(
                    initial_magnitude_mask,
                    imp.mask,
                ),
                "rewind_magnitude_to_imp_jaccard": support_jaccard(
                    rewind_magnitude_mask,
                    imp.mask,
                ),
                "snip_to_imp_jaccard": support_jaccard(snip, imp.mask),
                "synflow_to_imp_jaccard": support_jaccard(synflow, imp.mask),
                "posterior_map_mean_to_imp_jaccard": support_jaccard(
                    posterior_maps["posterior_mean_abs"],
                    imp.mask,
                ),
                "posterior_map_rms_to_imp_jaccard": support_jaccard(
                    posterior_maps["posterior_rms"],
                    imp.mask,
                ),
                "sample_accuracy_mean": finite_or_none(sample_mean(sample_accuracies)),
                "sample_accuracy_std": sample_std(sample_accuracies),
                "sample_to_dense_prediction_agreement_mean": finite_or_none(
                    sample_mean(sample_dense_agreements)
                ),
                "sample_to_imp_prediction_agreement_mean": finite_or_none(
                    sample_mean(sample_imp_agreements)
                ),
                "state_num_clusters": state_clustering["num_clusters"],
                "function_num_clusters": function_clustering["num_clusters"],
            }
            rows.append(row)
            print(json.dumps({"seed": args.seed, **row}))

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "train_size": train_bundle.train_size,
        "test_size": train_bundle.test_size,
        "posterior_augment": args.posterior_augment,
        "hmc_batchnorm_mode": args.hmc_batchnorm_mode,
        "hmc_basis": args.hmc_basis,
        "hmc_trajectory_epochs": trajectory_epochs,
        "hessian": {
            "batches": args.hessian_batches,
            "power_iterations": args.hessian_power_iterations,
            "oversample": args.hessian_oversample,
        },
        "hmc": {
            "steps": args.hmc_steps,
            "leapfrog_steps": args.hmc_leapfrog_steps,
            "prior_precision": args.hmc_prior_precision,
            "burn_in": args.hmc_burn_in,
            "sample_every": args.hmc_sample_every,
        },
        "dense": dense_metrics,
        "imp": imp.metrics,
        "rows": rows,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if rows:
        with (run_dir / "subspace_hmc_probe.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "dataset": args.dataset,
                "model": args.model,
                "num_rows": len(rows),
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
