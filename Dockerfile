# Artifact-verification container for the Lottery/Bayesian modes paper.
#
# This image is intentionally CPU-only. It is meant to rebuild generated
# statistics from committed run artifacts and compile the paper, not to rerun
# the CUDA training experiments.
FROM python:3.12.3-slim-bookworm@sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        make \
        poppler-utils \
        ripgrep \
        texlive-bibtex-extra \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-ci.txt ./
RUN python -m pip install --no-cache-dir -r requirements-ci.txt

COPY . .

CMD ["sh", "-c", "make source-repository-check PYTHON=python && if [ -f runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz ]; then make ci-check paper-check PYTHON=python; else echo 'Full artifact payload absent; source-repository check completed.'; fi"]
