# Compute Resource Accounting

Date: 2026-05-07

This document collects the compute-resource information needed to reproduce
the current paper artifacts. It is not a profiler log: historical
`metrics.json` files do not contain runtime or energy fields. The purpose is
to give a reviewer enough concrete hardware, software, memory, and command
context to plan a rerun, while making the remaining wall-clock limitation
explicit.

## Environments

| Resource path | Device class | Evidence | Role |
| --- | --- | --- | --- |
| CPU artifact-verification container | CPU only | `Dockerfile`, `docs/container_lock.md`, `.github/workflows/check.yml` | Regenerates paper statistics from included `runs/` artifacts, verifies release manifests, compiles all paper PDFs, and runs NeurIPS PDF text/page checks. |
| CUDA training environment | 1 NVIDIA CUDA GPU | `docs/environment_snapshot.md`, `docs/environment_lock.json`, `requirements-lock.txt` | Full CIFAR-10/100, MNIST, Fashion-MNIST, posterior-sampling, pruning, alignment, calibration, and residual-process experiments. |
| CUDA training container | 1 NVIDIA CUDA GPU with NVIDIA container runtime | `Dockerfile.gpu`, `requirements-gpu-lock.txt`, `docs/gpu_training_container.md` | Recreates the locked CUDA package subset for training and posterior runs on a GPU host. |
| CPU smoke and small-data paths | CPU acceptable | `README.md`, `runs/*digits*`, `runs/fake_cifar10*` | Digits, fake-CIFAR, and paper-verifier checks that do not require full CIFAR throughput. |

The local training workstation snapshot records Python 3.12.3, Linux
6.17.0, an NVIDIA GeForce RTX 5090, Torch CUDA 13.0, `torch==2.11.0`,
`torchvision==0.26.0`, `numpy==1.26.4`, `scipy==1.11.4`,
`scikit-learn==1.4.1.post1`, and `matplotlib==3.6.3`.

## Reproduction Classes

| Experiment class | Typical command evidence | Compute needed | Resource notes |
| --- | --- | --- | --- |
| Artifact verification and paper build | `make check`, `make paper-check`, `make paper-neurips-check`, `make container-check` | CPU-only container | Uses included run summaries and PDF tooling. No external dataset download or GPU is required. |
| Digits and fake-CIFAR smoke tests | `README.md` commands with `--dataset digits` or `--dataset fake-cifar10` | CPU acceptable; GPU optional | Small synthetic or packaged-data checks for dense full-network, mask-artifact, and residual-process code paths. |
| MNIST/Fashion-MNIST Gate-1 and barrier rows | `README.md` Gate-1 commands, `runs/mnist*`, `runs/fashion*` | CPU possible; GPU recommended for repeated five-seed sweeps | Main rows use five seeds and short training schedules. |
| Full CIFAR-10 ResNet-20 training and pruning | `README.md` CIFAR commands, especially `--epochs 30 --batch-size 512 --seeds 0,1,2,3,4` | 1 CUDA GPU recommended | Main evidence rows use ResNet-20 width 16, cosine schedule, augmentation, rewind epoch 1, five IMP rounds, and five seeds unless marked as smoke or pilot. |
| Full CIFAR posterior sampling | SGLD, cSGLD, SWAG, low-rank Laplace, block/joint diagonal Laplace, and HMC commands in `README.md` | 1 CUDA GPU recommended | Posterior families vary in sample count, Hessian/Fisher batches, rank, and covariance block size. The exact command lines are preserved in `README.md` and run-specific docs. |
| OOD and calibration rows | `scripts/run_calibration_ood_probe.py`, CIFAR-100 OOD commands in `README.md` | 1 CUDA GPU recommended | Uses CIFAR-10 in-distribution and CIFAR-100 or synthetic OOD evaluation. |
| Dense full-covariance feasibility audit | `scripts/audit_full_covariance_feasibility.py` | CPU-only audit; infeasible dense posterior | The audit computes memory/flop bounds without forming the dense matrix. |

## Memory Bounds

The dense full-network CIFAR-10 ResNet-20 covariance row is deliberately not a
runnable experiment in the artifact package. The generated feasibility audit
reports:

| Scope | Parameters | Dense matrix | Matrix plus Cholesky |
| --- | ---: | ---: | ---: |
| All trainable parameters | 272,474 | 553.1 GiB | 1,106.3 GiB |
| Weight tensors only | 270,896 | 546.8 GiB | 1,093.5 GiB |
| All trainable tensor-block diagonal | 272,474 | 56.8 GiB | 113.5 GiB |

This is why the paper uses exact dense small-model checks, selected-block
full-covariance checks, streamed joint-group covariance, low-rank
Hessian-plus-diagonal Laplace, SWAG, and low-dimensional HMC subspaces rather
than a literal dense full-network CIFAR covariance.

## Storage

Raw benchmark data caches under `data/` are intentionally excluded from the
release manifest and container context. The manifest packages generated source,
paper, docs, run summaries, and selected generated mask artifacts instead.

The largest currently packaged generated artifact is:

| Asset | Bytes | Role |
| --- | ---: | --- |
| `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz` | 118,946,673 | Saved activation-aligned full-data mode/ticket mask/state artifact. |

`docs/mode_ticket_artifact_storage_budget.md` estimates that a full-data
saved-artifact rerun for the strongest activation-aligned SGLD configuration
requires about 284.18 MiB uncompressed before compression.

## Remaining Limitation

The current metrics do not contain exact per-experiment wall-clock durations.
Future reruns should add `start_time`, `end_time`, `wall_clock_seconds`, GPU
model, driver version, peak allocated GPU memory, and peak host RSS to each
`metrics.json`. Until then, this document, the locked environments, and the
exact command lines are the compute-resource accounting used by the NeurIPS
checklist.
