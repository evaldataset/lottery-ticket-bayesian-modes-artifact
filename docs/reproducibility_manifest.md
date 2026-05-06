# Reproducibility Manifest

Date: 2026-05-06

This manifest lists the artifacts that currently support the paper draft and
the commands that regenerate or verify them. `docs/environment_snapshot.md`
records the local runtime snapshot, while `docs/environment_lock.json` and
`requirements-lock.txt` pin the project-critical Python, CUDA-visible PyTorch,
and TeX versions checked by `scripts/check_environment_lock.py`.
`Dockerfile` and `docs/container_lock.md` provide a separate CPU-only
artifact-verification container for generated statistics, release-manifest
checks, release-anonymization checks, public-release archive checks, and paper
compilation. The release archive smoke test also extracts the tarball and runs
the verifier in release-package mode. `docs/external_validation_receipts.json`
is the human-maintained registry for external upload, public-repository, CI, and
GPU-container receipts. `docs/external_validation_receipt_template.md`
pre-fills the current archive SHA and source commit for that registry;
`scripts/update_external_validation_receipts.py` validates the externally
observed URLs/evidence before writing the registry;
`scripts/audit_external_validation_readiness.py` generates the local readiness
sidecar and exposes a strict pre-upload gate that rejects placeholder URLs and
probes external URL reachability.
`scripts/stage_public_repository_snapshot.py` creates a source-only anonymous
git snapshot under `dist/` so the public repository can exclude large run
artifacts while the full artifact archive remains separately uploadable.
`Dockerfile.gpu`,
`docs/gpu_training_container.md`, and
`scripts/check_gpu_training_environment.py` define a CUDA training-container
path for rerunning CIFAR experiments on a GPU host. That image installs
`requirements-gpu-lock.txt`, a checked subset of `requirements-lock.txt` that
omits plotting-only packages.

## Standard Verification

```bash
make check
make paper-check
make paper-submission-check
make paper-neurips-check
make venue-submission-audit
make public-repository-snapshot
make external-validation-receipt-template
make external-validation-readiness
```

`make check` compiles all Python source files under `src/lottery/` and
`scripts/`, checks `docs/environment_lock.json`, regenerates
`runs/paper_stats.json`, regenerates `docs/paper_claim_ledger.md`, regenerates
the public release SHA256 inventory, runs the release-anonymization audit,
builds the local public-release tarball, and runs
`scripts/verify_research_artifacts.py`. The verifier checks required artifact
presence, selected raw metrics, generated statistical claims, and key numeric
claims in `paper/main.tex` against `runs/paper_stats.json`, and it fails if
manifest-included text artifacts expose local user names, host names, or
absolute workstation paths. It also checks that
`dist/lottery_artifact_public_release_2026-05-06.tar.gz` matches the manifest
member set plus release metadata sidecars, and that the extracted package can
pass the artifact verifier without the outer archive self-checks.
It also regenerates the external-validation readiness audit and receipt-fill
template, which keep the local artifact-release gate and local clean repository
snapshot separate from still-missing external receipts.
`make paper-check` rebuilds `paper/main.pdf` and fails on unresolved LaTeX
warnings after the final pass. It also runs the main-only submission build via
`paper-submission-check`, producing `paper/main_submission.pdf`, and the
official-style NeurIPS 2026 build via `paper-neurips-check`, producing
`paper/neurips_submission.pdf`.
`make venue-submission-audit` checks the NeurIPS 2026 content page budget before
References, official style binding, and checklist state while explicitly
flagging that public code/data upload, public repository state, and external
CI/GPU validation remain unverified. The audit also verifies the local
compute-resource, asset-license, and new-asset metadata docs.
The GitHub Actions workflow installs `make`, `poppler-utils`, `ripgrep`, and the
same lightweight TeX package set used by the artifact-verification container, then runs
`make ci-check paper-check PYTHON=python`, so generated artifact checks and both
paper PDF builds are part of the configured CI gate.

Containerized artifact verification is available with:

```bash
make container-build
make container-check
```

The container path runs `make ci-check paper-check PYTHON=python`. It does not
rerun CUDA training or posterior sampling experiments.

GPU training-container setup is available with:

```bash
make gpu-container-build
make gpu-container-env-check
```

The GPU path validates pinned package versions from `requirements-gpu-lock.txt`
against `docs/environment_lock.json`, Torch CUDA `13.0`, CUDA availability, and
a small CUDA tensor operation. It is intentionally outside the default CPU
artifact-verification path. `make gpu-container-env-check` calls
`scripts/run_gpu_container_env_check.py`, which tries Docker's standard
`--gpus all` path first and then falls back to explicit NVIDIA device and
driver-library mounts when the host lacks NVIDIA Container Toolkit integration.

## Generated Paper Artifacts

| Artifact | Source command |
| --- | --- |
| `Dockerfile` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` |
| `Dockerfile.gpu` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` |
| `LICENSE` | manually selected MIT License for the anonymous-review release; checked by `.venv/bin/python scripts/verify_research_artifacts.py` |
| `docs/container_lock.md` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` |
| `docs/gpu_training_container.md` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` |
| `docs/compute_resource_accounting.md` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` and `scripts/audit_venue_submission_compliance.py` |
| `docs/asset_license_inventory.md` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` and `scripts/audit_venue_submission_compliance.py` |
| `docs/new_asset_inventory.md` | manually updated; checked by `.venv/bin/python scripts/verify_research_artifacts.py` and `scripts/audit_venue_submission_compliance.py` |
| `docs/external_validation_receipts.json` | manually filled after public upload/external validation; checked by `.venv/bin/python scripts/audit_external_validation_readiness.py` |
| `docs/external_validation_receipt_template.md` | `.venv/bin/python scripts/build_external_validation_receipt_template.py` |
| `scripts/update_external_validation_receipts.py` | manual receipt-update helper; validates against `runs/external_validation_receipt_template.json` |
| `docs/environment_lock.json` | manually updated from `docs/environment_snapshot.md`; checked by `.venv/bin/python scripts/check_environment_lock.py` |
| `scripts/check_gpu_training_environment.py` | manually updated; run by `make gpu-env-check` or `make gpu-container-env-check` |
| `scripts/run_gpu_container_env_check.py` | manually updated; run by `make gpu-container-env-check` |
| `requirements-lock.txt` | manually updated from the project-critical packages in the active runtime |
| `requirements-gpu-lock.txt` | manually updated from the CUDA training subset in the active runtime; checked by `.venv/bin/python scripts/check_gpu_training_environment.py --allow-no-cuda` |
| `runs/public_release_manifest.json` | `.venv/bin/python scripts/build_release_manifest.py` |
| `docs/public_release_manifest.md` | `.venv/bin/python scripts/build_release_manifest.py` |
| `runs/release_anonymization_audit.json` | `.venv/bin/python scripts/audit_release_anonymization.py` |
| `docs/release_anonymization_audit.md` | `.venv/bin/python scripts/audit_release_anonymization.py` |
| `dist/lottery_artifact_public_release_2026-05-06.tar.gz` | `.venv/bin/python scripts/build_public_release_archive.py` |
| `dist/lottery_artifact_public_release_2026-05-06.tar.gz.sha256` | `.venv/bin/python scripts/build_public_release_archive.py` |
| `runs/public_release_archive_audit.json` | `.venv/bin/python scripts/build_public_release_archive.py` |
| `docs/public_release_archive_audit.md` | `.venv/bin/python scripts/build_public_release_archive.py` |
| `runs/public_release_archive_smoke.json` | `.venv/bin/python scripts/smoke_public_release_archive.py` |
| `docs/public_release_archive_smoke.md` | `.venv/bin/python scripts/smoke_public_release_archive.py` |
| `runs/public_repository_snapshot_audit.json` | `.venv/bin/python scripts/stage_public_repository_snapshot.py` |
| `docs/public_repository_snapshot_audit.md` | `.venv/bin/python scripts/stage_public_repository_snapshot.py` |
| `runs/external_validation_readiness_audit.json` | `.venv/bin/python scripts/audit_external_validation_readiness.py` |
| `docs/external_validation_readiness_audit.md` | `.venv/bin/python scripts/audit_external_validation_readiness.py` |
| `runs/cifar10_resnet20_full_covariance_feasibility.json` | `.venv/bin/python scripts/audit_full_covariance_feasibility.py` |
| `docs/cifar10_resnet20_full_covariance_feasibility.md` | `.venv/bin/python scripts/audit_full_covariance_feasibility.py` |
| `runs/mode_ticket_alignment_artifact_audit.json` | `.venv/bin/python scripts/audit_mode_ticket_alignment_artifacts.py` |
| `docs/mode_ticket_alignment_artifact_audit.md` | `.venv/bin/python scripts/audit_mode_ticket_alignment_artifacts.py` |
| `docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md` | `.venv/bin/python scripts/run_mode_ticket_distribution_probe.py ... --save-mask-artifacts --save-state-artifacts`; summarized by `scripts/summarize_mode_ticket_distribution_probe.py` |
| `runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv` | `.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_mask_artifact_smoke ...` |
| `runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json` | `.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py` |
| `docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md` | `.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py` |
| `runs/mode_ticket_artifact_storage_budget.json` | `.venv/bin/python scripts/audit_mode_ticket_artifact_storage_budget.py` |
| `docs/mode_ticket_artifact_storage_budget.md` | `.venv/bin/python scripts/audit_mode_ticket_artifact_storage_budget.py` |
| `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz` | `.venv/bin/python scripts/run_mode_ticket_distribution_probe.py ... --save-mask-artifacts --save-state-artifacts` |
| `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv` | `.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3 ...` |
| `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json` | `.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --max-channel-pair-count 1` |
| `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md` | `.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --max-channel-pair-count 1` |
| `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json` | `.venv/bin/python scripts/audit_full_data_channel_permutation_matching.py` |
| `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md` | `.venv/bin/python scripts/audit_full_data_channel_permutation_matching.py` |
| `runs/resnet_channel_permutation_exhaustive_feasibility_audit.json` | `.venv/bin/python scripts/audit_exhaustive_channel_permutation_feasibility.py` |
| `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md` | `.venv/bin/python scripts/audit_exhaustive_channel_permutation_feasibility.py` |
| `runs/paper_stats.json` | `.venv/bin/python scripts/build_paper_stats.py` |
| `docs/paper_stats.md` | `.venv/bin/python scripts/build_paper_stats.py` |
| `paper/tables/statistical_summary.tex` | `.venv/bin/python scripts/build_paper_stats.py` |
| `runs/linear_connectivity_barrier_audit.csv` | `.venv/bin/python scripts/audit_linear_connectivity_barriers.py` |
| `runs/linear_connectivity_barrier_audit.json` | `.venv/bin/python scripts/audit_linear_connectivity_barriers.py` |
| `docs/linear_connectivity_barrier_audit.md` | `.venv/bin/python scripts/audit_linear_connectivity_barriers.py` |
| `docs/paper_claim_ledger.md` | `.venv/bin/python scripts/build_paper_claim_ledger.py` |
| `runs/reviewer_objection_matrix.json` | `.venv/bin/python scripts/build_reviewer_objection_matrix.py` |
| `docs/reviewer_objection_matrix.md` | `.venv/bin/python scripts/build_reviewer_objection_matrix.py` |
| `runs/paper_submission_shape_audit.json` | `.venv/bin/python scripts/audit_paper_submission_shape.py` |
| `docs/paper_submission_shape_audit.md` | `.venv/bin/python scripts/audit_paper_submission_shape.py` |
| `runs/submission_pdf_shape_audit.json` | `.venv/bin/python scripts/audit_submission_pdf_shape.py` |
| `docs/submission_pdf_shape_audit.md` | `.venv/bin/python scripts/audit_submission_pdf_shape.py` |
| `runs/venue_submission_compliance_audit.json` | `.venv/bin/python scripts/audit_venue_submission_compliance.py` |
| `docs/venue_submission_compliance_audit.md` | `.venv/bin/python scripts/audit_venue_submission_compliance.py` |
| `paper/neurips_2026.sty` | official NeurIPS 2026 style copied into `paper/` |
| `paper/neurips_checklist.tex` | manually maintained NeurIPS checklist; checked by `.venv/bin/python scripts/audit_venue_submission_compliance.py` |
| `paper/figures/gate1_controls.*` | `.venv/bin/python scripts/build_paper_figures.py` |
| `paper/figures/cifar_movement.*` | `.venv/bin/python scripts/build_paper_figures.py` |
| `paper/figures/cifar_trajectory.*` | `.venv/bin/python scripts/build_paper_figures.py` |
| `paper/main.pdf` | `make paper-check` |
| `paper/main_submission.pdf` | `make paper-submission-check` |
| `paper/neurips_submission.pdf` | `make paper-neurips-check` |

## Core Evidence Groups

| Evidence group | Required summary artifacts |
| --- | --- |
| MNIST/Fashion Gate1 | `runs/mnist_gate1_full_r*_p0p3_summary.csv`, `runs/fashion_gate1_full_r*_p0p3_summary.csv` |
| CIFAR posterior movement | `runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv` |
| Full-covariance feasibility | `runs/cifar10_resnet20_full_covariance_feasibility.json`, `docs/cifar10_resnet20_full_covariance_feasibility.md` |
| High-fidelity posterior checks | `runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv`, `docs/digits_fullnet_laplace_tiny_r2_p0p3.md`, `runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv`, `docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md`, `runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3_summary.csv` |
| Linear connectivity barrier audit | `runs/linear_connectivity_barrier_audit.csv`, `runs/linear_connectivity_barrier_audit.json`, `docs/linear_connectivity_barrier_audit.md` |
| Reviewer objection matrix | `runs/reviewer_objection_matrix.json`, `docs/reviewer_objection_matrix.md` |
| Paper submission shape audit | `runs/paper_submission_shape_audit.json`, `docs/paper_submission_shape_audit.md` |
| Main-only submission PDF audit | `runs/submission_pdf_shape_audit.json`, `docs/submission_pdf_shape_audit.md`, `paper/main_submission.pdf` |
| Venue compliance audit | `runs/venue_submission_compliance_audit.json`, `docs/venue_submission_compliance_audit.md`, `paper/neurips_submission.pdf`, `paper/neurips_2026.sty`, `paper/neurips_checklist.tex` |
| Direct mode/ticket probes | `runs/digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv`, `runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5_summary.csv`, `runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv` |
| Channel-aligned mode/ticket probes | `runs/cifar10_subset_alignment_mode_ticket_smoke_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv`, `runs/mode_ticket_alignment_artifact_audit.json`, `docs/mode_ticket_alignment_artifact_audit.md` |
| Mask-artifact path validation | `runs/fake_cifar10_mode_ticket_mask_artifact_smoke/*/mask_artifacts.npz`, `runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv`, `docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md`, `runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json`, `docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md`, `runs/mode_ticket_artifact_storage_budget.json`, `docs/mode_ticket_artifact_storage_budget.md`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json`, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md`, `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json`, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md`, `runs/resnet_channel_permutation_exhaustive_feasibility_audit.json`, `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md` |
| Calibration/OOD and learned masks | `runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3_summary.csv`, `runs/digits_mlp_variational_prune_calib_r5_summary.csv` |
| IMP-process and residual controls | `runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv`, `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv` |

## Current Open Reproducibility Gaps

- The artifact-verification container now pins the linux/amd64 Python base
  image digest and TeX/poppler package set. The GitHub Actions workflow now installs
  the same lightweight TeX/poppler/ripgrep stack and runs `make ci-check paper-check
  PYTHON=python`; a separate CUDA training-container definition pins the
  linux/amd64 CUDA 13.0.1/cuDNN base digest and the project-critical Python
  stack. `make public-repository-snapshot` now creates a clean local
  source-only git snapshot under `dist/`; `make external-validation-readiness`
  records the remaining receipt gaps, and `make external-validation-strict` is
  expected to fail until an externally observed CI run, GPU-host run, public
  archive upload, and public repository receipt are recorded.
- The root project directory is still not a git repository, but the generated
  source-only repository snapshot provides a clean local commit for upload
  preparation.
- The full-data activation-aligned, weight-correlation-aligned, multi-chain
  cyclical-SGLD, rank-128 low-rank Laplace, and 270,896-parameter streamed
  joint-group Laplace CIFAR mode/ticket runs are integrated. The alignment
  artifact audit records that old direct-run artifacts lacked raw
  posterior/ticket mask or state tensors. The direct probe now has
  `--save-mask-artifacts` and `--save-state-artifacts`, with a fake-CIFAR
  fixture checking the `.npz` schema and parameter shapes plus a post-hoc
  matching audit that reads the fixture and computes record-level minimum-cost
  and local channel-permutation mask/state comparisons. The activation-aligned
  full-data saved-artifact rerun is now integrated and its post-hoc audit
  verifies record-level matching over saved CIFAR masks/states; broader
  posterior-chain graph/channel permutation variants remain open analyses over
  that saved artifact. The structured global channel audit already checks a
  block-coordinate channel-relabeling rescue and keeps posterior/ticket Hamming
  near `0.21`. The exact stage-1 enumeration feasibility audit validates all
  `128` channel assignments on a tiny saved-artifact subgraph and sizes the
  full CIFAR channel search at about `10^840.4` assignments per record pair,
  leaving exhaustive full-data graph isomorphism explicit and unimplemented.
- The tiny exact dense full-network Laplace sanity row is integrated as a
  small-model covariance-code validation artifact; it does not remove the
  remaining CIFAR dense-covariance limitation.
- A fake-CIFAR width-1 ResNet-20 exact dense full-network Laplace smoke is
  integrated to validate the convolutional/residual/BatchNorm code path. It is
  explicitly not real CIFAR evidence.
- The linear connectivity barrier audit is integrated as a generated artifact;
  it is not an open rescue path because near-zero MNIST/Fashion barriers and
  large CIFAR barriers both coexist with posterior support tied to chain-start
  controls.
