# Paper Draft

This directory contains the current working draft for the negative-result
version of the project.

The draft is intentionally honest about evidence status:

- Full MNIST and Fashion-MNIST Gate1 sparsity sweeps are included.
- CIFAR-10 ResNet-20 short and long epoch-1 rewind diagnostics are included,
  with SGLD, SGHMC, cyclical SGLD, SWAG, multi-chain, diagonal Laplace, and
  KFAC-style Laplace controls plus rank-16/rank-32 full-network low-rank
  Hessian-plus-diagonal Laplace movement diagnostics, an exact full-covariance
  final-head Laplace probe, full-covariance selected-block and joint-block
  Laplace probes, random, trajectory-informed, and Hessian-informed
  full-network low-dimensional subspace HMC probes, a tiny exact dense
  full-network Laplace sanity row, CIFAR-100 OOD/calibration
  diagnostics, direct
  mode/ticket distribution probes including unaligned and activation-aligned
  five-seed full-data CIFAR-10 ResNet-20 rows plus a 75-sample multi-chain
  cyclical-SGLD full-data direct probe, and a matched dense-trajectory support
  probe with aggregate trajectory-score, layer/stage overlap
  diagnostics, and fixed-mask retraining plus residual-swap, residual-anatomy,
  residual predictor-mask, cross-seed residual-transfer, activation-aligned
  direct cross-seed residual-support transfer, residual base-compatibility,
  residual posterior-decomposition,
  residual stratified-control, residual removal-order, IMP-process,
  IMP-process ranking-control, oracle-overlap, score-source, and
  round-exclusion residual probes.
- Small-model HMC is included as a higher-fidelity small-network check.
- A tiny exact dense full-network Laplace sanity row validates the all-parameter
  dense covariance path on a 310-parameter digits MLP and remains negative.
- A fake-CIFAR ResNet-20 width-1 exact dense full-network Laplace smoke
  validates the convolutional/residual/BatchNorm path; it is not real CIFAR
  evidence.
- The generated linear connectivity barrier audit shows that near-zero
  MNIST/Fashion barriers and large CIFAR barriers both fail to make posterior
  support beat chain-start controls.
- `docs/reviewer_objection_matrix.md` is now the compact risk register for
  paper editing: it maps nine likely reviewer objections to artifact-backed
  answers and remaining gaps.
- `docs/paper_submission_shape_audit.md` currently marks the condensed draft as
  shape-ready by the local main-text gate; the remaining readiness gaps are
  external validation and bounded robustness limitations.
- `paper/main_submission.pdf` is the main-only venue-facing PDF built from the
  same source with appendix/generated evidence tables excluded; its page budget
  is checked in `docs/submission_pdf_shape_audit.md`.
- `paper/neurips_submission.pdf` is the NeurIPS 2026 official-style PDF built
  with `paper/neurips_2026.sty` and `paper/neurips_checklist.tex`.
  `docs/venue_submission_compliance_audit.md` marks the content packet and
  venue binding ready for NeurIPS 2026, while keeping final submission upload
  readiness open until public code/data upload, external CI/GPU validation, and
  public repository state are verified. Local compute-resource, asset-license,
  and new-asset metadata are now documented in `docs/compute_resource_accounting.md`,
  `docs/asset_license_inventory.md`, and `docs/new_asset_inventory.md`.
- `.github/workflows/check.yml` now installs the lightweight TeX/poppler/ripgrep
  stack and runs `make ci-check paper-check PYTHON=python`, so all three paper
  PDFs are in the configured CI gate.
- Full-network exact/full-covariance CIFAR posterior checks beyond multi-chain
  cyclical SGLD, low-rank Hessian-plus-diagonal Laplace,
  selected-block/joint-block Laplace, and low-dimensional subspace-HMC probes
  remain open before submission; broader posterior-chain permutation variants
  are secondary robustness work, though mask/state saving and a fake-CIFAR
  post-hoc matching audit with local channel matching are now wired. The
  current release also includes a full-data block-coordinate channel audit and
  an exact stage-1 enumeration feasibility audit that validates a tiny
  saved-artifact subgraph while sizing the full CIFAR channel search at about
  `10^840.4` assignments per record pair.

Regenerate paper figures:

```bash
.venv/bin/python scripts/build_paper_figures.py
.venv/bin/python scripts/build_paper_stats.py
.venv/bin/python scripts/build_paper_claim_ledger.py
.venv/bin/python scripts/verify_research_artifacts.py
```

Equivalent top-level targets:

```bash
make figures
make paper-check
make paper-submission-check
make paper-neurips-check
make verify
```

Build, if a LaTeX toolchain is installed:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Build the main-only submission PDF:

```bash
cd paper
pdflatex -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
bibtex main_submission
pdflatex -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
pdflatex -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
```

Build the NeurIPS 2026 official-style PDF:

```bash
cd paper
pdflatex -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
bibtex neurips_submission
pdflatex -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
pdflatex -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
```
