# Container Lock

Date: 2026-05-06

This repository includes a CPU-only artifact-verification container. It is
intended to rebuild generated paper statistics from included run artifacts,
verify the public release manifest, and compile the manuscript. It is not the
GPU training environment used for the CIFAR experiments.

A separate CUDA training-container definition is documented in
`docs/gpu_training_container.md` and implemented by `Dockerfile.gpu`.

## Image

Base image:

```text
python:3.12.3-slim-bookworm@sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84
```

The digest above is the linux/amd64 manifest digest observed from Docker Hub
on 2026-05-06. The image installs only the OS packages required for paper and
artifact verification:

- `make`
- `git`
- `poppler-utils`
- `ripgrep`
- `texlive-bibtex-extra`
- `texlive-latex-extra`
- `texlive-fonts-recommended`
- `texlive-latex-base`
- `texlive-latex-recommended`

Python dependencies are installed from `requirements-ci.txt`, which pins
`numpy==1.26.4` for generated-stat and artifact-verifier compatibility. This
keeps the container small and aligned with `.github/workflows/check.yml`; the
full local CUDA/PyTorch runtime remains locked separately in
`requirements-lock.txt` and `docs/environment_lock.json`. The optional GPU
training image installs the narrower `requirements-gpu-lock.txt` subset so it
does not build plotting-only packages that are irrelevant to CUDA training.

## Commands

Build the artifact image:

```bash
make container-build
```

Run artifact verification and the paper build inside the image:

```bash
make container-check
```

Build the optional CUDA training image:

```bash
make gpu-container-build
```

Check the optional CUDA training image on a GPU host:

```bash
make gpu-container-env-check
make local-gpu-container-validation
make external-gpu-container-receipt
```

The default container command first verifies the source-only public repository
path, then runs the full artifact verifier when the large artifact payload is
present. The container check intentionally treats `ci-check` as a portable
rebuildability gate; exact matching to the public release URL, archive SHA, and
external CI receipt remains part of the local `make check` release path because
the in-container TeX/PDF toolchain can produce different archive bytes.

```bash
make source-repository-check PYTHON=python
make ci-check paper-check PYTHON=python  # only when full artifact payload is present
```

## Scope

Covered:

- Python 3.12.3 CPU artifact path
- generated statistics from included `runs/` artifacts
- public release SHA256 inventory
- source-only public repository snapshot staging
- source-only public repository CI smoke path
- LaTeX/BibTeX paper compilation
- PDF page/text inspection for submission compliance audits
- lightweight verifier and paper-build gate used by CI

Not covered:

- CUDA driver/runtime behavior
- full CIFAR training and posterior sampling runs
- externally observed green CI status
- external public git repository upload state

The optional GPU container covers the CUDA runtime/package-lock path but is
not part of the default artifact-verification CI because it requires an NVIDIA
runtime and a much larger image. `scripts/run_gpu_container_env_check.py`
keeps the standard `--gpus all` path while allowing an explicit driver-library
mount fallback on CUDA hosts where Docker is not configured with the NVIDIA
Container Toolkit.
`docs/local_gpu_container_validation.md` records the local GPU-container pass;
an independent external GPU-host receipt remains a stricter optional hardening
item in `docs/external_validation_receipts.json`. On that external host,
`scripts/build_external_gpu_container_receipt.py` writes the uploadable
JSON/Markdown receipt with commit, image ID, CUDA device metadata, Docker/NVIDIA
metadata, and the receipt-registry update command.

Those remain tracked in `docs/thread_goal_completion_audit.md` and
`docs/submission_readiness_audit.md`.
