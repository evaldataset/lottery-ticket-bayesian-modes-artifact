# Lottery Ticket Bayesian Modes

This repository turns `proposal_A3_lottery_ticket_bayesian_modes.md` into a
reproducible research project.

Core question:

> Do posterior basins or modes induce sparse supports that statistically align
> with iterative magnitude pruning winning tickets?

The first milestone is intentionally small: a full pipeline on `sklearn` digits
that runs without external dataset downloads. MNIST/CIFAR experiments should
reuse the same abstractions once `torchvision` or an equivalent data loader is
available.

## Current Entry Point

Environment:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

The current local environment used for generated results is recorded in
`docs/environment_snapshot.md`; the project-critical package and runtime lock
is recorded in `docs/environment_lock.json` and `requirements-lock.txt`.
Project-authored assets in the local release package use the top-level MIT
`LICENSE`; third-party dataset, style, and dependency terms are inventoried in
`docs/asset_license_inventory.md`.
The public release file inventory and hashes are recorded in
`docs/public_release_manifest.md` and `runs/public_release_manifest.json`.
The anonymous-review release audit is generated in
`docs/release_anonymization_audit.md`; it scans the manifest package for local
user names, host names, and absolute workstation paths.
The local anonymous-review release tarball is generated as
`dist/lottery_artifact_public_release_2026-05-06.tar.gz`, with archive-level
checks in `docs/public_release_archive_audit.md` and extraction-package smoke
checks in `docs/public_release_archive_smoke.md`.
External submission-readiness receipts are tracked in
`docs/external_validation_receipts.json`, and the generated readiness audit is
written to `docs/external_validation_readiness_audit.md`. The exact command
sequence for collecting those receipts is generated in
`docs/external_validation_runbook.md`; `docs/external_validation_receipt_template.md`
pre-fills the current archive SHA and source commit so only externally observed
URLs, clean-tree evidence, pass flags, and GPU image digest need manual entry.
Use `scripts/update_external_validation_receipts.py` to validate those fields
and write the receipt registry after the external evidence exists.
Submission-form metadata and upload handoff fields are generated in
`docs/submission_handoff.md`. The strict
external gate is intentionally separate from `make check` because it requires
public upload, public repository, external CI, and external GPU-host evidence;
it rejects placeholder URLs and probes external URL reachability.
A source-only anonymous repository snapshot is staged under `dist/` by
`scripts/stage_public_repository_snapshot.py` and audited in
`docs/public_repository_snapshot_audit.md`; it excludes the large run-artifact
payload that belongs in the separate release archive.
The paper's reviewer-facing claim-to-artifact map is generated in
`docs/paper_claim_ledger.md`.
The reviewer-objection risk register is generated in
`docs/reviewer_objection_matrix.md`.
The manuscript length/shape audit is generated in
`docs/paper_submission_shape_audit.md`.
The venue-facing main-only PDF audit is generated in
`docs/submission_pdf_shape_audit.md`.
The NeurIPS 2026 venue-binding compliance audit is generated in
`docs/venue_submission_compliance_audit.md`; it checks
`paper/neurips_submission.pdf`, `paper/neurips_2026.sty`, and
`paper/neurips_checklist.tex`.
Checklist-facing release metadata is documented in
`docs/compute_resource_accounting.md`,
`docs/asset_license_inventory.md`, and `docs/new_asset_inventory.md`; the venue
audit now verifies those documents while still keeping external public
code/data upload readiness open.
The CPU artifact-verification container is defined by `Dockerfile` and
documented in `docs/container_lock.md`.
The optional CUDA training container is defined by `Dockerfile.gpu` and
documented in `docs/gpu_training_container.md`.

Fast artifact verification:

```bash
make check
make paper-check
make paper-submission-check
make paper-neurips-check
make venue-submission-audit
make release-anonymization-audit
make release-archive
make release-archive-smoke
make public-repository-snapshot
make external-validation-receipt-template
make external-validation-readiness
```

`make check` includes `scripts/check_environment_lock.py`, so a package,
Python, CUDA, or TeX mismatch fails before paper statistics are trusted.
It also regenerates the public release manifest and fails if the anonymization
audit finds local identity or absolute-path leakage in release text artifacts.
It builds and audits the local public-release tarball before the final artifact
verifier runs, then extracts that tarball and runs the artifact verifier in
release-package mode.
It also records the external-validation receipt state without failing the local
gate and stages a clean source-only public repository snapshot; use
`make external-validation-readiness` for the current runbook and
`make external-validation-strict` after external receipts are filled.
It also regenerates the linear connectivity barrier audit in
`docs/linear_connectivity_barrier_audit.md`, which shows that near-zero
MNIST/Fashion dense-to-IMP barriers and large CIFAR barriers both fail to rescue
posterior-ticket support equivalence once chain-start controls are included.

Containerized artifact verification:

```bash
make container-build
make container-check
```

The container is CPU-only and first runs
`make source-repository-check PYTHON=python`; when the full artifact payload is
present it then runs `make ci-check paper-check PYTHON=python`. `ci-check`
verifies the rebuildable artifact gates without requiring the rebuilt in-container
archive bytes to match the public-release receipt; that receipt-bound check is
kept in the local `make check` release path. `paper-check` builds the
appendix-inclusive reproducibility PDF, the main-only article PDF, and the
official-style NeurIPS 2026 PDF.
full CIFAR training remains tied to the local CUDA environment lock.
The GitHub Actions workflow installs the same lightweight TeX/poppler/ripgrep system
dependencies, runs the source-repository check, and conditionally runs the full
artifact/paper gate when the large artifact payload is present.

CUDA training-container setup:

```bash
make gpu-container-build
make gpu-container-env-check
make local-gpu-container-validation
make external-gpu-container-receipt
```

`make gpu-container-env-check` requires a CUDA host with Docker's NVIDIA
runtime and validates the pinned `requirements-gpu-lock.txt` Torch/CUDA
training stack against `docs/environment_lock.json` inside the image. The
target first uses Docker's `--gpus all` path and then falls back to explicit
NVIDIA device and driver-library mounts when the host GPU is present but the
NVIDIA Container Toolkit is not registered with Docker.
`make local-gpu-container-validation` records the local pass in
`docs/local_gpu_container_validation.md`; a separate external GPU-host receipt
is tracked as optional independent hardening. On an independent CUDA host,
`make external-gpu-container-receipt` writes
`runs/external_gpu_container_receipt.json` and
`docs/external_gpu_container_receipt.md`; upload one of those files and use the
public URL plus the recorded image ID in the external receipt registry.

```bash
python scripts/run_digits_pilot.py --epochs 30 --sgld-steps 600 --samples 60
python scripts/summarize_digits_runs.py
```

Multi-chain corrected-SGLD smoke run:

```bash
python scripts/run_digits_pilot.py --dataset digits --sgld-chains 2 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr 1e-7
```

Gate evaluation:

```bash
python scripts/evaluate_gate1.py runs/mnist_gate1_quick_summary.json
```

SGLD movement sweep summary:

```bash
python scripts/summarize_rescue_sweep.py --pattern 'runs/mnist_sgld_rescue_*/*/metrics.json'
```

Full Gate1 sparsity sweep:

```bash
.venv/bin/python scripts/run_gate1_sparsity_sweep.py --dataset mnist --run-prefix mnist_gate1_full --seeds 0,1,2,3,4 --configs 2:0.30,3:0.30,5:0.30,8:0.30 --epochs 5
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/mnist_gate1_full_r2_p0p3_summary.json --summary runs/mnist_gate1_full_r3_p0p3_summary.json --summary runs/mnist_gate1_full_r5_p0p3_summary.json --summary runs/mnist_gate1_full_r8_p0p3_summary.json --out-csv runs/mnist_gate1_full_sweep.csv --out-md docs/mnist_gate1_full_sweep.md
```

SWAG posterior baseline:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset mnist --posterior-sampler swag --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 5 --prune-fraction 0.30 --samples 20 --swag-epochs 5 --out-dir runs/mnist_swag_r5_p0p3 --summary-prefix runs/mnist_swag_r5_p0p3
.venv/bin/python scripts/run_gate1_sweep.py --dataset fashion-mnist --posterior-sampler swag --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 5 --prune-fraction 0.30 --samples 20 --swag-epochs 5 --out-dir runs/fashion_swag_r5_p0p3 --summary-prefix runs/fashion_swag_r5_p0p3
```

Small-model full-batch HMC baseline:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_hmc_baseline.py --seed "$seed" --hidden-dim 8 --depth 2 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --hmc-steps 160 --hmc-step-size 1e-2 --hmc-leapfrog-steps 4 --hmc-burn-in 60 --hmc-sample-every 10 --random-trials 100 --out-dir runs/digits_hmc_long_eps1e-2_r2_p0p3; done
.venv/bin/python scripts/summarize_hmc_runs.py --run-root runs/digits_hmc_long_eps1e-2_r2_p0p3 --out-csv runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.csv --out-json runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json
.venv/bin/python scripts/evaluate_gate1.py runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json --out-json runs/digits_hmc_long_eps1e-2_r2_p0p3_gate1_eval.json
```

Image model smoke:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset fake-cifar10 --model resnet20 --resnet-width 4 --epochs 1 --imp-rounds 1
```

Real CIFAR-10 subset smoke:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --resnet-width 4 --train-subset 512 --test-subset 256 --epochs 1 --imp-rounds 1
```

CIFAR-10 full-data training sanity check:

```bash
.venv/bin/python scripts/run_cifar_baseline.py --seed 0 --epochs 10 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_baseline
```

CIFAR-10 full-data Gate1 pilot:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale dataset --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_gate1_pilot_r2_p0p3
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_gate1_short_r2_p0p3_summary.json --summary runs/cifar10_resnet20_gate1_short_r5_p0p3_summary.json --summary runs/cifar10_resnet20_gate1_short_r8_p0p3_summary.json --out-csv runs/cifar10_resnet20_gate1_short_sweep.csv --out-md docs/cifar10_resnet20_gate1_short_sweep.md
```

CIFAR-10 full-data SWAG short control:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --sgld-chains 1 --swag-epochs 5 --swag-lr 0.01 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_swag_short_r5_p0p3 --summary-prefix runs/cifar10_resnet20_swag_short_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 full-data SGLD multi-chain short control:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 3 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3 --summary-prefix runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 full-data SGLD movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,3e-10,1e-9,3e-9,1e-8,3e-8 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_r5_p0p3
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-7,3e-7,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_highlr_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3.md
```

CIFAR-10 full-data SGHMC movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sghmc_movement_short_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3.md
```

CIFAR-10 30-epoch r5 pilot:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 30-epoch gradual-pruning pilot:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --imp-rounds 8 --prune-fraction 0.20 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_r8_p0p2 --summary-prefix runs/cifar10_resnet20_long30_r8_p0p2 --skip-existing-seeds
```

CIFAR-10 30-epoch epoch-1 rewind pilot:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 30-epoch epoch-1 rewind SWAG control:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --sgld-chains 1 --swag-epochs 5 --swag-lr 0.01 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 30-epoch epoch-1 rewind calibration/OOD probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --samples 10 --out-dir runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3; done
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3.md
```

CIFAR-10 subset learned-mask calibration/OOD smoke, for path validation only:

```bash
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset gaussian-noise --model resnet20 --resnet-width 4 --seed 0 --epochs 2 --imp-rounds 2 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --ood-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --samples 1 --swag-epochs 1 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 1 --learned-mask-sources random,gem_miner,variational_prune,hard_concrete --learned-random-trials 1 --mask-train-epochs 1 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --variational-prune-epochs 1 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 1 --variational-prune-samples-per-batch 1 --hard-concrete-epochs 1 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 1 --hard-concrete-samples-per-batch 1 --out-dir runs/cifar10_subset_calibration_ood_learned_masks_smoke
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_subset_calibration_ood_learned_masks_smoke --out-csv runs/cifar10_subset_calibration_ood_learned_masks_smoke_summary.csv --out-md docs/cifar10_subset_calibration_ood_learned_masks_smoke.md
```

CIFAR-10 large-subset seed-0 learned-mask calibration/OOD pilot:

```bash
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --resnet-width 8 --seed 0 --epochs 5 --imp-rounds 2 --prune-fraction 0.30 --batch-size 128 --train-subset 4096 --test-subset 2000 --ood-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 3 --swag-epochs 2 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 3 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 3 --gem-miner-epochs 3 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 10 --variational-prune-epochs 3 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 10 --variational-prune-samples-per-batch 1 --out-dir runs/cifar10_subset4096_calibration_ood_learned_masks_pilot
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_subset4096_calibration_ood_learned_masks_pilot --out-csv runs/cifar10_subset4096_calibration_ood_learned_masks_pilot_summary.csv --out-md docs/cifar10_subset4096_calibration_ood_learned_masks_pilot.md
```

CIFAR-10 full-data learned-mask calibration/OOD row:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --samples 10 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 30 --gem-miner-epochs 10 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 20 --variational-prune-samples-per-batch 1 --out-dir runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3; done
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3.md
```

Current result: learned-random, Gem-Miner-style, and variational-prune hard
masks reduce ECE to 0.0270, 0.0283, and 0.0255, but their accuracies drop to
0.8449, 0.8418, and 0.8301 and their MSP OOD AUROCs drop to 0.7897, 0.7853,
and 0.7754. IMP remains better on accuracy, NLL, Brier, and OOD AUROC.

CIFAR-10 30-epoch epoch-1 rewind SGLD multi-chain control:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 3 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3 --skip-existing-seeds
```

CIFAR-10 30-epoch epoch-1 rewind SGLD movement diagnostic:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind SGHMC movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7,3e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sghmc_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7,3e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind cyclical SGLD movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler cyclical-sgld --sgld-lrs 1e-10,1e-6,3e-6,1e-5 --csgld-lr-min-ratio 0.01 --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --sgld-steps 400 --sgld-burn-in 100 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_csgld_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler cyclical-sgld --sgld-lrs 1e-10,1e-6,3e-6,1e-5 --csgld-lr-min-ratio 0.01 --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --sgld-steps 400 --sgld-burn-in 100 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind full-network SWAG movement diagnostic:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --swag-scales 1.0,16.0,64.0 --swag-epochs 20 --swag-lr 0.001 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind diagonal Laplace movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 1e-10,1e-8,1e-6,3e-6,1e-5,3e-5,1e-4 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_tune_r5_p0p3
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 3e-4,1e-3,3e-3,1e-2,3e-2 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_highscale_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 1e-10,1e-3,3e-3,1e-2 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind KFAC-style Laplace movement diagnostic:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler kfac-laplace --kfac-laplace-scales 1e-10,1e-8,1e-6,1e-4,1e-2 --kfac-laplace-prior-precision 1e-2 --kfac-laplace-fisher-batches 5 --kfac-laplace-damping 1e-2 --kfac-laplace-factor-rows 2048 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler kfac-laplace --kfac-laplace-scales 1e-10,1e-4,1e-3,1e-2 --kfac-laplace-prior-precision 1e-2 --kfac-laplace-fisher-batches 10 --kfac-laplace-damping 1e-2 --kfac-laplace-factor-rows 4096 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind full-network low-rank Hessian-plus-diagonal
Laplace movement diagnostic:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 16 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 4 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 32 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 8 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 64 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 16 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-2 --lowrank-laplace-rank 128 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 32 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3.md
```

Current result: the rank-16, rank-32, rank-64, and rank-128 Hessian-plus-diagonal rows move
full-network support away from the chain start, but not toward IMP. At scale
`1e-2`, rank-16 posterior-to-chain-start overlap falls to `0.7359`, sample
accuracy is `0.8784`, and posterior-to-IMP overlap is `0.1351`, below
chain-start magnitude `0.1456` and epoch-1 rewind magnitude `0.1777`. The
rank-32 row is nearly identical: posterior-to-chain-start overlap `0.7402`,
sample accuracy `0.8789`, and posterior-to-IMP `0.1358`, below chain-start
`0.1457` and rewind magnitude `0.1795`. The rank-64 row retains all 64
positive Hessian directions across five seeds and remains negative:
posterior-to-chain-start `0.7397`, sample accuracy `0.8816`, and
posterior-to-IMP `0.1339`, below chain-start `0.1433` and rewind magnitude
`0.1766`. The selected rank-128 row retains all 128 positive Hessian
directions and stays negative: posterior-to-chain-start `0.7358`, sample
accuracy `0.8813`, and posterior-to-IMP `0.1351`, below chain-start `0.1453`
and rewind magnitude `0.1780`.

Full-covariance feasibility audit for the same CIFAR-10 ResNet-20 model:

```bash
.venv/bin/python scripts/audit_full_covariance_feasibility.py
```

Current result: exact dense full covariance over all trainable parameters would
need `553.1` GiB for one float64 matrix and `1,106.3` GiB with a Cholesky
factor resident. A tensor-block-diagonal exact covariance over weight tensors
still needs `113.5` GiB for matrix plus Cholesky, dominated by five
`36,864`-parameter stage-3 convolution tensors. This quantifies why the paper
uses exact head/selected-block covariance plus full-network low-rank/SWAG/HMC
approximations rather than an exact full-network CIFAR covariance run.

Tiny digits exact dense full-network Laplace sanity probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_fullnet_laplace_probe.py --seed "$seed" --hidden-dim 4 --depth 2 --epochs 20 --imp-rounds 2 --prune-fraction 0.3 --samples 10 --full-laplace-scales 1e-5,1e-4,1e-3,1e-2 --out-dir runs/digits_fullnet_laplace_tiny_r2_p0p3; done
.venv/bin/python scripts/summarize_fullnet_laplace_probe.py --run-root runs/digits_fullnet_laplace_tiny_r2_p0p3 --out-csv runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv --out-md docs/digits_fullnet_laplace_tiny_r2_p0p3.md
```

Current result: the tiny MLP exact dense full-network covariance covers all
`310` trainable parameters. At scale `1e-3`, samples remain accurate
(`0.8450`) and move from the dense chain-start support (`0.8084` post-chain),
but posterior-to-IMP support is `0.7545` versus `0.8596` for the chain-start
magnitude control, a `-0.1050` gap. This validates the exact dense code path
in a small setting; it is not CIFAR-scale evidence.

Tiny fake-CIFAR ResNet exact dense full-network Laplace smoke:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_fullnet_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 1 --seed "$seed" --epochs 1 --imp-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --prune-fraction 0.3 --batch-size 16 --test-batch-size 32 --train-size 32 --test-size 32 --samples 2 --full-laplace-scales 1e-5,1e-3 --full-laplace-hessian-batches 1 --full-laplace-max-parameters 2000 --out-dir runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke; done
.venv/bin/python scripts/summarize_fullnet_laplace_probe.py --run-root runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke --out-csv runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv --out-md docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md
```

Current result: the smoke covers every trainable parameter of a width-1
ResNet-20 (`1,229` parameters, dense Cholesky shape `1229 x 1229`) and checks
the convolutional/residual/BatchNorm path. At scale `1e-3`, support moves from
the chain start (`post-chain=0.5398`) and remains below the chain-start control
(`posterior-chain=-0.4372`). This is code-path validation only, not real CIFAR
evidence.

CIFAR-10 30-epoch epoch-1 rewind exact final-head Laplace probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_head_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --head-laplace-scales 1e-6,1e-3,1e-2,1e0 --head-laplace-prior-precision 1e-2 --head-laplace-damping 1e-5 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_head_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind full-covariance block Laplace probe:

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 256 --test-subset 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --block-name conv1.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --samples 2 --random-trials 5 --out-dir runs/fake_cifar10_block_laplace_smoke
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/fake_cifar10_block_laplace_smoke --out-csv runs/fake_cifar10_block_laplace_smoke_summary.csv --out-md docs/fake_cifar10_block_laplace_smoke.md
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer1.0.conv1.weight --block-laplace-scales 1e-4,1e-3,1e-2 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 3000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer1.0.conv1.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 3000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3.md
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-names conv1.weight,layer1.0.conv1.weight,layer1.2.conv2.weight,layer2.0.conv1.weight,layer2.0.shortcut.0.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 5000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer3.0.shortcut.0.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 5000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3.md
.venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --joint-block-names conv1.weight,layer2.0.shortcut.0.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 2000 --samples 1 --random-trials 2 --out-dir runs/fake_cifar10_joint_block_laplace_smoke
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --joint-block-names conv1.weight,layer1.0.conv1.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-4,3e-4,1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 6000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --joint-block-names conv1.weight,layer1.0.conv1.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-4 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 6000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3.md
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-blocks-under-max --independent-block-diagonal --random-trials 4 --out-dir runs/fake_cifar10_blockdiag_laplace_smoke
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/fake_cifar10_blockdiag_laplace_smoke --out-csv runs/fake_cifar10_blockdiag_laplace_smoke_summary.csv --out-md docs/fake_cifar10_blockdiag_laplace_smoke.md
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-4,3e-4,1e-3 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 5000 --auto-blocks-under-max --independent-block-diagonal --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 5000 --auto-blocks-under-max --independent-block-diagonal --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 4 --out-dir runs/fake_cifar10_blockdiag_laplace_max10k_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6,3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_tune_small_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3.md
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-joint-groups-under-max --random-trials 4 --out-dir runs/fake_cifar10_jointdiag_laplace_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6,3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-joint-groups-under-max --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-joint-groups-under-max --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 20000 --auto-joint-groups-under-max --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 3e-6 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 20000 --auto-joint-groups-under-max --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3.md
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-joint-groups-under-max --stream-joint-groups --random-trials 4 --out-dir runs/fake_cifar10_jointdiag_laplace_stream_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 3 --block-laplace-scales 1e-6,3e-6,1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 40000 --auto-joint-groups-under-max --stream-joint-groups --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 40000 --auto-joint-groups-under-max --stream-joint-groups --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3.md
```

Current block-diagonal result: the independent exact block-covariance row covers
11 tensors and 22,064 parameters. It preserves useful sample accuracy
(`0.8810`) and moves away from the dense chain-start support
(`global post-chain=0.8287`), but selected-block posterior-to-IMP is below the
selected-block chain-start support by `-0.0114`; the global posterior-chain
gain is only `+0.0036`, and rewind support remains much closer to IMP
(`rewind-posterior=0.0292`). This narrows the selected-block/full-network
covariance gap without rescuing ticket-directed support movement.

The wider max-10k block-diagonal row covers 16 tensors and 68,144 parameters.
It preserves sample accuracy (`0.8802`) and moves farther from chain-start
support (`global post-chain=0.7400`), but block posterior-chain remains
negative (`-0.0050`), global posterior-chain is only `+0.0010`, and rewind
support remains closer by `0.0319`.

The max-10k joint-group diagonal row keeps the same 16 tensors and 68,144
parameters, but greedily packs them into 8 exact joint groups to include
cross-tensor covariance inside each group. It preserves sample accuracy
(`0.8811`) and moves farther from chain-start support (`global
post-chain=0.7148`), but block posterior-chain remains negative (`-0.0050`),
global posterior-chain is only `+0.0015`, and rewind support remains closer by
`0.0311`.

The max-20k joint-group row adds `layer3.0.conv1.weight`, covering 17 tensors
and 86,576 parameters in 6 exact covariance groups. It preserves sample
accuracy (`0.8828`) and keeps support away from the chain start (`global
post-chain=0.7863`), but block posterior-chain remains negative (`-0.0023`),
global posterior-chain is only `+0.0006`, and rewind support remains closer by
`0.0317`.

The streamed max-40k joint-group row covers all 22 ResNet-20 weight tensors and
270,896 weight parameters in 8 exact covariance groups. It preserves sample
accuracy (`0.8824`) and keeps support away from the chain start (`global
post-chain=0.7389`), but block/global posterior-chain remains negative
(`-0.0019`) and rewind support remains closer by `0.0362`.

CIFAR-10 30-epoch epoch-1 rewind low-dimensional full-network subspace HMC probes:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 8 --hmc-direction-scale 10 --hmc-step-sizes 3e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --hmc-basis trajectory --hmc-trajectory-epochs 0,1,2,5,10,20,30 --subspace-dims 6 --hmc-direction-scale 10 --hmc-step-sizes 3e-4,1e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --hmc-basis hessian --subspace-dims 4 --hmc-direction-scale 1 --hmc-step-sizes 3e-4 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 5 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 2 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3.md
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 16 --hmc-basis hessian --hmc-step-sizes 1e-4,3e-4,1e-3 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3.md
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 32 --hmc-basis hessian --hmc-step-sizes 1e-4,3e-4,1e-3 --hmc-steps 8 --hmc-leapfrog-steps 2 --hmc-burn-in 2 --hmc-sample-every 2 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 8 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 16 --hmc-basis hessian --hmc-step-sizes 3e-4 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 32 --hmc-basis hessian --hmc-step-sizes 3e-4 --hmc-steps 8 --hmc-leapfrog-steps 2 --hmc-burn-in 2 --hmc-sample-every 2 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 8 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind matched dense-trajectory probe, including
aggregate trajectory score masks and stage/layer overlap diagnostics:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2; done
.venv/bin/python scripts/summarize_trajectory_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2.md
```

CIFAR-10 30-epoch epoch-1 rewind trajectory mask retraining probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-sources imp,random,epoch_1,epoch_10,epoch_30,traj_rms_abs,traj_mean_abs,traj_path_length,traj_rewind_rms_movement --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3.md
```

Gem-Miner-style mask source smoke:

```bash
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --out-dir runs/cifar10_subset_gem_miner_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_subset_gem_miner_mask_training_smoke --out-csv runs/cifar10_subset_gem_miner_mask_training_smoke_summary.csv --out-md docs/cifar10_subset_gem_miner_mask_training_smoke.md
```

Hard-concrete L0 mask source smoke:

```bash
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --resnet-width 4 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 2 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --mask-train-epochs 1 --mask-sources hard_concrete --random-trials 0 --hard-concrete-epochs 1 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 1 --hard-concrete-samples-per-batch 1 --out-dir runs/cifar10_subset_hard_concrete_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_subset_hard_concrete_mask_training_smoke --out-csv runs/cifar10_subset_hard_concrete_mask_training_smoke_summary.csv --out-md docs/cifar10_subset_hard_concrete_mask_training_smoke.md
```

Current smoke result: the hard-concrete path produces a sparse mask at 0.5099
sparsity with 0.3216 Jaccard overlap to IMP on the short CIFAR subset path.
This validates the stronger learned-mask implementation; it is not yet a
five-seed full-data evidence row.

Gem-Miner-style one-seed full-data pilot:

```bash
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 5 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3.md
```

Gem-Miner-style five-seed full-data selected row:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 5 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3.md
```

CIFAR-10 five-seed variational-pruning support row:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources variational_prune --random-trials 0 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-kl-weight 1e-4 --variational-prune-sparsity-weight 10 --variational-prune-entropy-weight 1e-3 --variational-prune-temperature-start 2.0 --variational-prune-temperature-end 0.2 --variational-prune-samples-per-batch 1 --variational-prune-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3.md
```

Current result: CIFAR variational pruning reaches 0.8306 accuracy, 0.0907
Jaccard overlap to IMP, and sits 0.0669 below IMP accuracy. Its support is
random-scale and below the Gem-Miner-style accuracy row, so it does not rescue
the proposal on CIFAR.

CIFAR-10 five-seed hard-concrete support row:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources hard_concrete --random-trials 0 --hard-concrete-epochs 10 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 20 --hard-concrete-samples-per-batch 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md
```

Current result: hard-concrete retrains to 0.2766 accuracy at the matched IMP
sparsity, sits 0.6204 below IMP accuracy, and has 0.0922 Jaccard overlap to
IMP, i.e. random-scale support rather than a ticket-like learned mask.

Digits variational-pruning sanity check:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset digits --model mlp --hidden-dim 128 --depth 3 --seed "$seed" --epochs 10 --trajectory-epochs 0,10 --rewind-epochs 0 --imp-rounds 3 --prune-fraction 0.30 --batch-size 128 --lr 0.05 --lr-schedule constant --weight-decay 1e-4 --mask-train-epochs 10 --mask-sources imp,random,gem_miner,variational_prune --random-trials 1 --gem-miner-epochs 10 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-kl-weight 1e-4 --variational-prune-sparsity-weight 10 --variational-prune-entropy-weight 1e-3 --variational-prune-temperature-start 2.0 --variational-prune-temperature-end 0.2 --variational-prune-samples-per-batch 1 --variational-prune-max-batches-per-epoch 20 --out-dir runs/digits_mlp_variational_prune_calib_r5; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/digits_mlp_variational_prune_calib_r5 --out-csv runs/digits_mlp_variational_prune_calib_r5_summary.csv --out-md docs/digits_mlp_variational_prune_calib_r5.md
```

Current result: variational pruning reaches 0.9633 accuracy on digits, above
random masks at 0.9489 and Gem-Miner-style masks at 0.9550, but below IMP at
0.9711 and dense training at 0.9739. Its ECE is 0.0250 versus IMP's 0.0211,
so the first direct H3 row does not support calibration superiority.

CIFAR-10 30-epoch epoch-1 rewind trajectory residual-swap probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_residual_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0,0.5,1.0 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual removal-order controls:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_residual_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0,0.5 --imp-remove-orders low,random,high --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual-anatomy probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_residual_anatomy_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --predictor-steps 120 --predictor-batch-size 16384 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3; done
.venv/bin/python scripts/summarize_residual_anatomy_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3 --out-prefix runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary --out-md docs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual predictor-mask probe:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_residual_predictor_mask_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --predictor-steps 120 --predictor-batch-size 16384 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3; done
.venv/bin/python scripts/summarize_residual_predictor_mask_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual cross-seed transfer probe:

```bash
.venv/bin/python scripts/run_residual_cross_seed_transfer_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --predictor-steps 120 --predictor-batch-size 16384 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3
.venv/bin/python scripts/summarize_residual_cross_seed_transfer_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind activation-aligned direct cross-seed
residual-support transfer:

```bash
.venv/bin/python scripts/run_residual_direct_transfer_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-residual-trials 1 --alignment-method activation --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3
.venv/bin/python scripts/summarize_residual_direct_transfer_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual base-compatibility probe:

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual posterior-decomposition probe:

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,random-imp,dense-imp,posterior-imp,posterior-excess-imp,posterior-std-imp --random-residual-trials 1 --posterior-laplace-samples 10 --posterior-laplace-scale 1e-3 --posterior-laplace-prior-precision 1e-2 --posterior-laplace-fisher-batches 20 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual stratified-control probe:

```bash
.venv/bin/python scripts/run_residual_stratified_control_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-trials 1 --score-bins 10 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3
.venv/bin/python scripts/summarize_residual_stratified_control_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual IMP-process probe:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants survivor,final-imp --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3.md
```

CIFAR-10 30-epoch epoch-1 rewind residual IMP-process ranking controls:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants survivor,survivor-random,survivor-low --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3.md
```

CIFAR-10 residual IMP-process oracle-overlap-matched controls:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 1 --imp-rounds 1 --process-rounds 1 --round-variants final-imp,final-imp-oracle-matched-random --base-sources epoch_1 --alphas 0.5 --mask-train-epochs 1 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_oracle_matched_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_oracle_matched_smoke --out-csv runs/fake_cifar10_residual_imp_process_oracle_matched_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_oracle_matched_smoke.md
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --process-rounds 1 --round-variants final-imp,final-imp-oracle-matched-random --base-sources epoch_2 --alphas 0.5 --mask-train-epochs 1 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --random-trials 1 --out-dir runs/cifar10_subset_residual_imp_process_oracle_matched_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_subset_residual_imp_process_oracle_matched_smoke --out-csv runs/cifar10_subset_residual_imp_process_oracle_matched_smoke_summary.csv --out-md docs/cifar10_subset_residual_imp_process_oracle_matched_smoke.md
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-oracle-matched-random --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3.md
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-oracle-matched-random --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3.md
```

The 5-seed oracle-overlap-matched row gives a small but positive paired signal:
round-score final-IMP residuals beat matched random final-IMP residuals in
35/45 base/round/seed comparisons, with mean accuracy delta `+0.0020`.

CIFAR-10 residual IMP-process score-source controls:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3.md
```

The 5-seed score-source row fixes the final-IMP residual candidate set and
support budget, but ranks additions by dense-final or base-source magnitude
instead of round-trained IMP weights. Round-trained scores beat dense-score
controls in 37/45 comparisons with mean delta `+0.0026`, and beat base-score
controls in 39/45 comparisons with mean delta `+0.0028`.

CIFAR-10 residual IMP-process round-exclusion controls:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3.md
```

The 5-seed round-exclusion row removes the round-selected final-IMP residual
additions from the candidate set, then chooses the best remaining final-IMP
residual additions by final IMP magnitude under the same support budget.
Round-selected masks beat this replacement in 44/45 comparisons, with mean
delta `+0.0061`.

CIFAR-10 residual IMP-process tensor-matched round-exclusion control:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3
for seed in 1 2 3 4; do .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3; done
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3.md
```

The tensor-matched 5-seed row focuses on the strongest RMS trajectory
round-5 setting. After removing the round-selected final-IMP residual
coordinates, the replacement is matched by parameter tensor and then selected
by final IMP magnitude. Round-selected masks reach `0.8855` accuracy versus
`0.8764` for the tensor-matched replacement, with mean paired delta `+0.0091`
and `5/5` positive seed deltas.

CIFAR-10 residual IMP-process tensor+score-matched round-exclusion control:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle,final-imp-round-excluded-tensor-score-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md
```

The tensor+score-matched 5-seed row makes the replacement more competitive by
matching each removed process-selected coordinate by parameter tensor and
within-tensor round-score decile before applying final IMP magnitude.
Round-selected masks reach `0.8878` accuracy versus `0.8837` for the
tensor+score-matched replacement, with mean paired delta `+0.0041` and `5/5`
positive seed deltas. The replacement's final-oracle overlap rises to `0.6440`
from `0.4167` for tensor-only matching, but it still does not replace the
process-selected row.

Residualized round-score projection control:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-residualized-score,final-imp-dense-score,final-imp-base-score --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3.md
```

This projection removes the linear subspace spanned by base-source,
dense-final, and final-IMP magnitude scores inside the final-IMP residual
candidate pool before ranking by the remaining round-score residual.
Round-selected masks reach `0.8852` accuracy versus `0.8811` for the
residualized-score control, with mean paired delta `+0.0041` and `5/5`
positive seed deltas. Final-oracle overlap drops from `0.6684` to `0.4854`,
localizing the useful round ordering to an interaction with the
trajectory/final-magnitude subspace.

Posterior-residualized round-score projection control:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 0 --imp-rounds 1 --process-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --mask-train-epochs 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --base-sources epoch_1 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-posterior-residualized-score --posterior-projection-laplace-samples 2 --posterior-projection-laplace-fisher-batches 1 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_posterior_projection_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_posterior_projection_smoke --out-csv runs/fake_cifar10_residual_imp_process_posterior_projection_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_posterior_projection_smoke.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --mask-train-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-posterior-residualized-score --posterior-projection-laplace-samples 10 --posterior-projection-laplace-scale 1e-3 --posterior-projection-laplace-prior-precision 1e-2 --posterior-projection-laplace-fisher-batches 20 --posterior-projection-laplace-variance-floor 1e-12 --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3.md
```

This projection additionally removes diagonal-Laplace posterior RMS,
posterior standard deviation, and posterior RMS-minus-dense scores before
ranking by the remaining round-score residual. Round-selected masks reach
`0.8847` accuracy versus `0.8825` for the posterior-residualized control, with
`5/5` positive seed deltas. The coordinate signal is sharper than the accuracy
gap: final-oracle overlap drops from `0.6773` to `0.4850`.

Learned-subspace residualized round-score projection control:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 0 --imp-rounds 1 --process-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --mask-train-epochs 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --base-sources epoch_1 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-learned-subspace-residualized-score --learned-subspace-rank 2 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_learned_subspace_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_learned_subspace_smoke --out-csv runs/fake_cifar10_residual_imp_process_learned_subspace_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_learned_subspace_smoke.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --mask-train-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-learned-subspace-residualized-score --learned-subspace-rank 8 --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md
```

This projection learns a rank-8 PCA subspace from dense trajectory scores,
final-IMP magnitude, and earlier IMP-round scores inside the final-IMP
residual candidate pool, then ranks by the remaining round-score residual.
Round-selected masks reach `0.8869` accuracy versus `0.8821` for the
learned-subspace residualized control, with mean paired delta `+0.0048` and
`5/5` positive seed deltas. Final-oracle overlap drops from `0.6807` to
`0.4917`, so a learned trajectory/process subspace does not replace the exact
IMP-selected residual coordinates.

Paper figures:

```bash
.venv/bin/python scripts/build_paper_figures.py
.venv/bin/python scripts/run_mode_distribution_equivalence_audit.py
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset digits --model mlp --hidden-dim 128 --depth 3 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 3 --prune-fraction 0.3 --samples 10 --sgld-steps 120 --sgld-burn-in 20 --sgld-sample-every 10 --batch-size 128 --lr 0.05 --weight-decay 1e-4 --out-dir runs/digits_mlp_mode_ticket_distribution_sgld_r5
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/digits_mlp_mode_ticket_distribution_sgld_r5 --out-md docs/digits_mlp_mode_ticket_distribution_sgld_r5.md --out-csv runs/digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 8 --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 2 --prune-fraction 0.3 --samples 5 --sgld-steps 80 --sgld-burn-in 20 --sgld-sample-every 12 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 4096 --test-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 10 --sliced-projections 64 --out-dir runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5 --out-md docs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5.md --out-csv runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 8 --seeds 0,1,2 --epochs 5 --rewind-epochs 1 --imp-rounds 2 --prune-fraction 0.3 --samples 5 --sgld-steps 80 --sgld-burn-in 20 --sgld-sample-every 12 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 4096 --test-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 10 --sliced-projections 64 --out-dir runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot --out-md docs/cifar10_subset4096_mode_ticket_distribution_activation_pilot.md --out-csv runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --alignment-method activation --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --alignment-method weight --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler cyclical-sgld --samples 5 --posterior-chains 3 --posterior-chain-init dense --sgld-steps 400 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler cyclical-sgld --samples 5 --posterior-chains 3 --posterior-chain-init independent-dense --sgld-steps 400 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler lowrank-laplace --samples 10 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --lowrank-laplace-scale 1e-2 --lowrank-laplace-rank 128 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 32 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler jointdiag-laplace --samples 5 --jointdiag-laplace-scale 1e-6 --jointdiag-laplace-prior-precision 1e-2 --jointdiag-laplace-damping 1e-5 --jointdiag-laplace-hessian-batches 1 --jointdiag-laplace-max-parameters 40000 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv
.venv/bin/python scripts/audit_mode_ticket_alignment_artifacts.py
.venv/bin/python scripts/build_paper_stats.py
```

The mode/ticket distribution audit reuses existing posterior artifacts and
writes `docs/mode_distribution_equivalence_audit.md`,
`runs/mode_distribution_equivalence_audit_summary.csv`, and
`runs/mode_distribution_equivalence_audit.json`. Current result: posterior
supports beat random masks in 58/59 grouped comparisons, but beat the matched
chain-start support by more than 0.005 Jaccard in 0/59 comparisons. Rewind
magnitude beats posterior by more than 0.005 Jaccard in 55/57 comparisons.

The direct proposal-level distribution probe writes
`docs/digits_mlp_mode_ticket_distribution_sgld_r5.md`,
`docs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5.md`,
`docs/cifar10_subset4096_mode_ticket_distribution_activation_pilot.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md`,
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md`,
and their matched summary CSVs. Current result:
50 posterior sample masks versus 5 IMP tickets fail the proposal's layer KS
threshold (`p=0.0788`) and Hamming-overlap threshold (`0.6314 < 0.70`) while
passing logit-CKA/Hungarian thresholds. Mean-shift clustering collapses the
posterior samples to one mode representative versus five tickets, with basin
entropy `0.0000` and effective cluster count `1.0000`, so this is direct
small-model evidence against the strong 1:1 mode/ticket claim.
The full-data CIFAR ResNet-20 row is stronger: raw posterior sample masks fail
layer KS (`p=5.3e-09`) and Hamming overlap (`0.0033 < 0.70`), and mean-shift
again collapses all 50 posterior samples to one basin versus five IMP tickets
(`H=0`, effective clusters `1`). Logit and final-hidden activation CKA still
pass, so functional similarity alone does not rescue mask-distribution or
basin-count equivalence. A smaller three-seed activation-CKA CIFAR subset pilot
is kept as a robustness note: its collapsed mode comparison passes, but its raw
sample masks still fail KS/Hamming thresholds. The alignment robustness check
also has a negative row: activation alignment maps ResNet
channels into the first dense seed's activation frame and still leaves one
posterior basin with layer KS `p=2.3e-09`, Hamming overlap `0.0000`, logit CKA
`0.9373`, and activation CKA `0.9168`. Weight-correlation channel alignment
uses incoming/outgoing ResNet weight features and is also negative: one
posterior basin, layer KS `p=1.2e-08`, Hamming overlap `0.1290`, and
logit/activation CKA `0.9336`/`0.9131`. A stronger full-data multi-chain
cyclical-SGLD direct probe also stays negative: 75 posterior samples from
three dense-start chains per seed move away from chain-start supports
(`posterior-to-chain-start` Hamming mean `0.0443`) while retaining sample
accuracy `0.8760`, but still collapse to one basin and fail layer KS
(`p=3.3e-08`) plus Hamming overlap (`0.2461 < 0.70`) despite high logit and
activation CKA (`0.9327`/`0.9144`).
An independent-start multi-chain cyclical-SGLD direct probe is also negative:
75 posterior samples from 15 independently trained dense starts still collapse
to one basin, move only `0.0439` Hamming from their chain starts, and fail
layer KS (`p=9.3e-10`) plus Hamming overlap (`0.0000 < 0.70`) despite high
logit/activation CKA (`0.9269`/`0.9051`).
A full-network rank-128 Hessian-plus-diagonal Laplace direct probe is a partial
rescue but not full proposal equivalence: 50 posterior samples collapse to one
basin, pass Hamming overlap (`0.8163`) and logit/activation CKA
(`0.9319`/`0.9096`), but still strongly fail layer-sparsity KS
(`p=2.0e-06`). The single posterior mode representative passes layer KS
(`p=0.1388`) but remains one mode versus five tickets, so the basin-count
claim remains unsupported.
A streamed 270,896-parameter joint-group Laplace direct probe closes the
full-weight exact grouped-covariance direct gap and is sharper: 25 posterior
samples keep `0.8835` sample accuracy and move `0.0503` Hamming from their
chain starts, but still collapse to one basin and fail layer KS (`p=1.1e-08`)
plus Hamming overlap (`0.0000 < 0.70`) while logit/activation CKA remain high
(`0.9373`/`0.9199`).
`scripts/audit_mode_ticket_alignment_artifacts.py` records the current
alignment/permutation artifact boundary in
`docs/mode_ticket_alignment_artifact_audit.md` and
`runs/mode_ticket_alignment_artifact_audit.json`: activation and
weight-correlation alignment rows both fail layer-KS/Hamming, all seven
full-data direct rows collapse to one basin, and post-hoc exhaustive
graph/permutation realignment is not supported by the current direct-run
artifacts because no raw posterior/ticket mask or state tensors were saved.
The direct probe now writes `--save-mask-artifacts` and, when parameter basin
reconstruction is needed, `--save-state-artifacts`. The fake-CIFAR
path-validation fixture at
`docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md` verifies the resulting
`mask_artifacts.npz` schema, and
`docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md` verifies that
the saved `.npz` can drive record-level post-hoc minimum-cost matching plus a
local ResNet channel-permutation objective. The activation-aligned full-data
SGLD saved-artifact rerun is now complete at
`runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706`;
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md`
verifies record-level post-hoc matching on the saved CIFAR masks/states.
`docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md`
then applies a ResNet-structured block-coordinate channel-permutation objective
to the record-optimal full-data pairs; posterior/ticket Hamming remains about
`0.2105` rather than ticket-like agreement. The audit is not an exhaustive
graph-isomorphism proof, but it makes the remaining permutation gap much
narrower. `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md`
adds an exact stage-1 enumeration check on a 270-parameter fake-CIFAR ResNet
subgraph: all `128` channel assignments are enumerated, the coordinate solver
matches the exact optimum, and known raw-vs-aligned frame differences go to
zero. The same audit sizes the full CIFAR artifact at about `10^840.4` channel
assignments per record pair, so exhaustive full-data graph isomorphism remains
an explicit infeasibility limitation. `docs/mode_ticket_artifact_storage_budget.md`
records that the reference run was within the estimated `284.18 MiB`
uncompressed budget.

MNIST/Fashion-MNIST/CIFAR-10 require `torchvision`. The system Python is
externally managed, so use the project venv:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install torchvision
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --epochs 1 --imp-rounds 1
```

The script writes a timestamped result directory under `runs/digits_pilot/` with:

- `metrics.json`: dense, IMP, posterior sampler, overlap, and clustering metrics
- `mask_overlaps.csv`: posterior-mask and random-mask overlap comparisons
- `runs/digits_pilot_summary.json`: aggregate summary across compatible runs
- `runs/mnist_pilot_summary.json`: aggregate summary for MNIST runs when
  `--dataset mnist` is used

## Research Gates

Gate 1 passes only if posterior-induced masks overlap IMP masks more than
same-sparsity random masks with a meaningful effect size.

Gate 2 passes only if the signal survives functional checks: prediction
agreement, representation similarity, and loss-barrier or connectivity tests.

Gate 3 passes only if the CIFAR-10 ResNet-20 version preserves the Gate 1/2
pattern across seeds and sparsity levels.
