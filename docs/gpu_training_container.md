# GPU Training Container

Date: 2026-05-06

This repository includes a separate CUDA-oriented container definition for
rerunning CIFAR training and posterior-sampling commands. It is intentionally
not the default CI container: the default `Dockerfile` remains a small CPU-only
artifact verifier for regenerated statistics and the paper build.

## Image

Base image:

```text
nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f
```

The digest above is the linux/amd64 platform manifest digest observed via
`docker manifest inspect --verbose` on 2026-05-06. The matching tag was chosen
because the locked local training environment reports `torch==2.11.0`,
`torchvision==0.26.0`, and Torch CUDA `13.0` in
`docs/environment_lock.json`.

The container installs:

- Ubuntu 24.04 Python 3
- `make`
- `ripgrep`
- the pinned project-critical Python stack from `requirements-lock.txt`

It does not install TeX. Paper compilation stays in the CPU artifact container.

## Commands

Build the training image:

```bash
make gpu-container-build
```

Check the image on a CUDA host:

```bash
make gpu-container-env-check
```

The default command inside the image is:

```bash
python scripts/check_gpu_training_environment.py
```

That check validates pinned package versions, Torch CUDA version, CUDA
availability, and a small CUDA matrix multiply. For metadata-only inspection on
a non-GPU host, run:

```bash
python scripts/check_gpu_training_environment.py --allow-no-cuda
```

## Scope

Covered:

- pinned CUDA base image for linux/amd64
- pinned Python numerical/deep-learning stack from `requirements-lock.txt`
- runtime CUDA availability check when launched with `--gpus all`
- smoke-level CUDA tensor execution

Not covered:

- full CIFAR experiment reruns inside the image
- data download and storage policy for `data/`
- externally observed green GPU-container CI status
- multi-GPU distributed training

The image is a reproducibility hardening artifact, not proof that every
expensive training command has been rerun in Docker.
