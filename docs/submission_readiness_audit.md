# Submission Readiness Audit

Objective:

> Produce research and paper-quality evidence suitable for a top conference on
> the relationship between lottery tickets and Bayesian posterior modes.

Current status: not ready for submission.

## Deliverable Checklist

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Precise top-level hypothesis | `docs/research_roadmap.md` reframes exact mode equivalence as a support-equivalence claim: posterior-induced sparse supports must explain IMP tickets beyond dense, rewind, trajectory, and pruning-process controls. | Substantial; scoped negative claim is now explicit |
| Novelty positioning | `docs/literature_matrix.md` identifies direct Bayesian LTH and LMC competitors, and `paper/main.tex` now distinguishes the support-equivalence test from general Bayesian pruning, PAC-Bayesian, posterior-approximation, and mode-connectivity claims. `paper/refs.bib` now includes the ICLR 2023 mask-subspace competitor and corrected PAC-Bayesian/Bayesian-LTH metadata, and `scripts/verify_research_artifacts.py` checks cited-key coverage plus key citation metadata. | Substantial; final venue editing still open |
| Reproducible codebase | `src/lottery/`, `scripts/`, `requirements.txt`, `requirements-ci.txt`, `requirements-lock.txt`, `.github/workflows/check.yml`, `Dockerfile`, `Dockerfile.gpu`, `.dockerignore`, `.gitignore`, `Makefile`, `LICENSE`, `docs/environment_snapshot.md`, `docs/environment_lock.json`, `docs/container_lock.md`, `docs/gpu_training_container.md`, `docs/compute_resource_accounting.md`, `docs/asset_license_inventory.md`, `docs/new_asset_inventory.md`, `docs/paper_claim_ledger.md`, `docs/reviewer_objection_matrix.md`, `docs/paper_submission_shape_audit.md`, `docs/submission_pdf_shape_audit.md`, `docs/venue_submission_compliance_audit.md`, `docs/submission_handoff.md`, `docs/public_release_manifest.md`, `docs/release_anonymization_audit.md`, `docs/public_release_archive_audit.md`, `docs/public_release_archive_smoke.md`, `docs/public_repository_snapshot_audit.md`, `docs/external_validation_readiness_audit.md`, `docs/external_validation_runbook.md`, `runs/public_release_manifest.json`, `runs/release_anonymization_audit.json`, `runs/public_release_archive_audit.json`, `runs/public_release_archive_smoke.json`, `runs/public_repository_snapshot_audit.json`, `runs/external_validation_readiness_audit.json`, `runs/external_validation_runbook.json`, `runs/submission_handoff.json`, `scripts/check_environment_lock.py`, `scripts/check_gpu_training_environment.py`, `scripts/build_paper_claim_ledger.py`, `scripts/build_reviewer_objection_matrix.py`, `scripts/audit_paper_submission_shape.py`, `scripts/audit_submission_pdf_shape.py`, `scripts/audit_venue_submission_compliance.py`, `scripts/build_release_manifest.py`, `scripts/audit_release_anonymization.py`, `scripts/build_public_release_archive.py`, `scripts/smoke_public_release_archive.py`, `scripts/stage_public_repository_snapshot.py`, `scripts/audit_external_validation_readiness.py`, `scripts/build_external_validation_runbook.py`, `scripts/build_submission_handoff.py`, and `scripts/verify_research_artifacts.py` implement runnable experiments, generated summaries, standard check/build targets, a checked project-critical environment lock, release metadata docs, a project MIT license, a CI workflow that installs git/TeX/poppler/ripgrep and runs `make ci-check paper-check PYTHON=python`, a CPU artifact-verification container, a GPU training-container definition, a claim-to-artifact ledger, a reviewer-objection risk register, manuscript/PDF/NeurIPS-venue-compliance audits, a submission handoff, a public release SHA256 inventory, a release-anonymization gate, a local public-release tarball gate, an extracted-package smoke gate, a clean source-only public-repository snapshot gate, an external-validation readiness gate, and an external-validation runbook over the exact public archive/source/CI/GPU receipts. | Substantial; no externally observed green CI/GPU-container run, public release upload, or public source-repository receipt |
| Baseline IMP | `src/lottery/imp.py` implements iterative magnitude pruning with rewinding. | Smoke-validated |
| Posterior sampler | `src/lottery/sgld.py`, `src/lottery/sghmc.py`, `src/lottery/cyclical_sgld.py`, `src/lottery/diag_laplace.py`, `src/lottery/kfac_laplace.py`, `src/lottery/lowrank_laplace.py`, `src/lottery/subspace_hmc.py`, and `src/lottery/swag.py` implement dataset-scaled SGLD, SGHMC, cyclical SGLD, diagonal/KFAC-style Laplace, full-network low-rank Hessian-plus-diagonal Laplace, low-dimensional subspace HMC, and SWAG controls. | Partial |
| Mode/mask comparison | `src/lottery/analysis.py` computes posterior-vs-random mask overlap, CIs, and tests; `scripts/run_mode_distribution_equivalence_audit.py` aggregates existing artifacts into KS/Wasserstein/MMD support-overlap comparisons against random, chain-start, dense, and rewind controls; `scripts/run_mode_ticket_distribution_probe.py` directly evaluates layer-sparsity KS, MMD, sliced Wasserstein, mask-Hamming overlap, logit CKA, optional final-hidden activation CKA, Hungarian cost, and raw parameter-PCA basin entropy between posterior sample/mode masks and IMP tickets on a small digits MLP, CIFAR-10 ResNet subset pilots, a five-seed full-data CIFAR-10 ResNet-20 row, activation- and weight-correlation-aligned full-data CIFAR rows, dense-start and independent-start 75-sample multi-chain cyclical-SGLD full-data CIFAR rows, a rank-128 low-rank Laplace full-data CIFAR row, and a 270,896-parameter streamed joint-group Laplace full-data CIFAR row with chain-start diagnostics. `scripts/audit_mode_ticket_alignment_artifacts.py` verifies the alignment rows stay negative and records that old direct-run artifacts lack raw masks/states for exhaustive permutation analysis. The direct probe now has `--save-mask-artifacts`/`--save-state-artifacts`, with a fake-CIFAR `.npz` schema/shape fixture, a local-channel post-hoc matching audit on that fixture, a full-data activation-aligned saved-artifact rerun with record-level post-hoc matching over saved CIFAR masks/states, a block-coordinate global channel audit that keeps posterior/ticket Hamming near `0.21`, and an exact stage-1 enumeration feasibility audit that checks all `128` fake-subgraph assignments while sizing the full CIFAR channel search at about `10^840.4` assignments per record pair. | Substantial negative proposal-metric evidence plus saved full-data artifact and structured/exact-small channel-permutation robustness; exhaustive full-data graph isomorphism remains infeasible and open |
| Function-space checks | `scripts/run_digits_pilot.py` records logits clustering and prediction agreement. | Smoke-validated |
| Connectivity checks | `src/lottery/connectivity.py` records linear loss barriers, and `scripts/audit_linear_connectivity_barriers.py` now aggregates six existing five-seed Gate1/CIFAR rows into `docs/linear_connectivity_barrier_audit.md`. MNIST/Fashion dense-to-IMP barriers are near zero while CIFAR barriers are large, yet posterior support never beats the chain-start control. | Audited negative support evidence; linear connectivity barrier audit treats barriers as orthogonal diagnostics, not support-equivalence evidence |
| Trajectory controls | Initial, rewind, dense, matched dense-trajectory checkpoint, aggregate trajectory-score, layer/stage overlap, fixed-mask retraining, residual-swap, residual-anatomy, functional residual-predictor mask, cross-seed residual-transfer, activation-aligned direct cross-seed residual-support transfer, residual base-compatibility, residual posterior-decomposition, residual stratified controls, residual removal-order controls, IMP-process, IMP-process ranking-control, oracle-overlap-matched process-control, score-source process-control, round-exclusion process-intervention, tensor-matched round-exclusion, tensor+score-matched round-exclusion, residualized-score projection, posterior-residualized projection, and learned-subspace residualized projection probes now exist; trajectory masks dominate posterior supports, IMP-only residual swaps recover much of the remaining functional gap, and dense-trajectory/process probes show that the functional residual is gradually constructed by IMP rather than explained by a simple coordinate rule, directly transferable seed-invariant support, activation-channel permutation, exact trajectory-base identity, final IMP membership alone, posterior RMS ordering beyond dense final magnitude, posterior uncertainty, layer/tensor/score-bin structure, low-score-removal artifact, oracle-overlap amount alone, dense/base magnitude ranking alone, replaceability of process-selected final-IMP residual coordinates by the best remaining final-IMP-magnitude coordinates even with parameter-tensor and within-tensor score-decile matching, a standalone process-score component orthogonal to base/dense/final-IMP magnitude, the current diagonal-Laplace posterior score subspace, or a learned trajectory/process subspace. | Partial |
| Repeated seeds | Five compatible digits seeds are summarized in `runs/digits_pilot_summary.json`. | Smoke-only |
| MNIST/Fashion-MNIST experiments | MNIST and Fashion-MNIST 5-seed, 4-sparsity Gate1 sweeps both fail at every sparsity. | Solid negative evidence |
| CIFAR-10 ResNet-20 experiments | Full-data short grid, SWAG/multi-chain controls, SGLD/SGHMC/cyclical-SGLD/full-network-SWAG/diagonal-Laplace/KFAC-style-Laplace/low-rank-Hessian-plus-diagonal-Laplace movement diagnostics, exact final-head Laplace, selected-block, joint-block, 22k/68k-parameter block-diagonal full-covariance Laplace, and 68k/86k/270k-parameter joint-group full-covariance Laplace including all weight tensors, random/trajectory/Hessian low-dimensional subspace HMC, distribution-equivalence audit, direct raw/aligned/multi-chain mode-ticket probes, CIFAR-100 OOD/calibration diagnostics, and 30-epoch epoch-1 rewind pilots exist; all support the negative Gate1 conclusion. | Substantial negative evidence, still not final submission-grade |
| Multi-chain posterior sampling | Multiple-chain independent-start controls exist on MNIST and CIFAR; CIFAR short and long-budget rows both fail Gate1 despite separated chains. | Partial |
| HMC or higher-fidelity posterior baseline | Small-model full-batch HMC now has a 5-seed tuned digits baseline with high sample accuracy and real support movement; an exact dense full-network Laplace sanity row on a tiny digits MLP samples all `310` trainable parameters in one covariance and is also negative (`0.7545` posterior support versus `0.8596` chain-start at scale `1e-3`). A fake-CIFAR ResNet-20 width-1 exact dense full-network Laplace smoke validates the convolutional/residual/BatchNorm code path over all `1,229` trainable parameters, but is explicitly not real CIFAR evidence. CIFAR diagonal, KFAC-style, and low-rank Hessian-plus-diagonal Laplace movement diagnostics, exact full-covariance final-head Laplace, seed-0 multi-block, two five-seed selected-block, one five-seed joint-block, one five-seed 22,064-parameter tensor-block-diagonal full-covariance Laplace probe, one five-seed 68,144-parameter tensor-block-diagonal full-covariance Laplace probe, and five-seed 68,144-, 86,576-, plus 270,896-parameter streamed joint-group full-covariance Laplace probes with within-group cross-tensor covariance over all weight tensors, plus five-seed full-network random/trajectory/top-Hessian low-dimensional subspace-HMC probes exist. Dense-start and independent-start 75-sample full-data multi-chain cyclical-SGLD direct probes, a 20-snapshot full-network SWAG movement row, rank-16/rank-32/rank-64/rank-128 full-network low-rank Laplace movement rows, a rank-128 low-rank Laplace direct row, and a 270,896-parameter streamed joint-group Laplace direct row now show moving posterior samples still fail direct/support equivalence. The 68k block-diagonal exact covariance row preserves `0.8802` sample accuracy and moves from chain-start support (`global post-chain=0.7400`), but block posterior-chain is `-0.0050`, global posterior-chain is only `+0.0010`, and rewind remains closer by `0.0319`; the 68k joint-group row preserves `0.8811` accuracy and moves farther (`global post-chain=0.7148`), but block posterior-chain is `-0.0050`, global posterior-chain is only `+0.0015`, and rewind remains closer by `0.0311`; the 86k joint-group row preserves `0.8828` accuracy and remains moved from chain-start support (`global post-chain=0.7863`), but block posterior-chain is `-0.0023`, global posterior-chain is only `+0.0006`, and rewind remains closer by `0.0317`; the 270k full-weight movement row preserves `0.8824` accuracy and remains moved from chain-start support (`global post-chain=0.7389`), but block/global posterior-chain is `-0.0019` and rewind remains closer by `0.0362`; the matching 270k direct row preserves sample accuracy `0.8835`, moves `0.0503` Hamming from chain starts, but fails layer KS (`p=1.1e-08`) and Hamming overlap (`0.0000`). `docs/cifar10_resnet20_full_covariance_feasibility.md` quantifies why literal dense exact CIFAR covariance is outside the single-workstation budget: `553.1` GiB for one all-trainable float64 matrix and `1,106.3` GiB with Cholesky resident. Exact all-parameter dense full-covariance CIFAR posterior evidence remains missing. | Partial but stronger |
| SNIP/SynFlow/Gem-Miner controls | SNIP and SynFlow implemented and used in posterior-control tables. A Gem-Miner-style STE score-training mask source is implemented in `src/lottery/pruning_baselines.py`, wired into `scripts/run_trajectory_mask_training_probe.py`, smoke-tested on fake-CIFAR plus a real CIFAR subset, and run as a five-seed full-data CIFAR baseline. The selected row is random-like in support and far below IMP, so it does not rescue score-training masks in this setting. | Initial baseline evidence |
| Variational / hard-concrete pruning baselines | Proposal-style Bernoulli/Concrete mask optimization and a hard-concrete L0 gate baseline are implemented in `src/lottery/pruning_baselines.py` and wired into the trajectory support and calibration/OOD probes. Fake-CIFAR and real CIFAR subset smokes pass for hard-concrete. The five-seed digits row records variational accuracy, ECE, Brier, and support; it beats random and Gem-Miner-style masks in accuracy but remains below IMP and does not improve ECE over IMP. `run_calibration_ood_probe.py` evaluates variational-prune, Gem-Miner-style, and learned-random hard masks after fixed-mask retraining in a five-seed full-data CIFAR-10/CIFAR-100 row. Matched five-seed CIFAR support rows now show variational pruning and hard-concrete gates are random-scale in support and below IMP on accuracy. | Multi-seed CIFAR negative evidence exists |
| Permutation/alignment handling | Activation-channel Hungarian alignment is implemented for the direct cross-seed residual-support transfer probe and the full-data direct mode/ticket probe; weight-correlation Hungarian alignment is implemented for the full-data direct mode/ticket probe; aligned source-vote masks remain base/random-like and below target oracle residuals, and aligned posterior mode/ticket masks still fail layer KS/Hamming thresholds. `docs/mode_ticket_alignment_artifact_audit.md` adds a checked target-frame/artifact-bound audit: seven full-data direct rows reject equivalence, activation/weight aligned rows both fail layer-KS/Hamming, and old direct-run artifacts do not contain raw masks/states. `docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md` validates the new `.npz` mask/state artifact path, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md` verifies record-level full-data post-hoc matching from saved CIFAR artifacts, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md` applies a structured global channel-permutation objective without rescuing posterior/ticket support, and `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md` validates exact stage-1 enumeration on a tiny saved-artifact subgraph while quantifying the full-data search space. | Substantial first-order evidence plus full-data saved-artifact record and channel matching; exact small-subgraph path validated, exhaustive full-data graph isomorphism remains infeasible/open |
| Calibration/OOD evaluation | `scripts/run_calibration_ood_probe.py` and `scripts/summarize_calibration_ood_probe.py` run five-seed CIFAR-10 calibration plus CIFAR-100 OOD diagnostics for dense, IMP, SWAG predictive controls, learned random masks, Gem-Miner-style masks, and variational-prune masks. SWAG and learned masks improve ECE in some rows but hurt accuracy, NLL, Brier, and OOD AUROC, so uncertainty behavior does not rescue the posterior-support account. | Solid calibration/OOD negative evidence under current CIFAR setting |
| Paper draft | Compilable working draft exists in `paper/main.tex`; appendix-inclusive `paper/main.pdf`, main-only `paper/main_submission.pdf`, and official-style NeurIPS `paper/neurips_submission.pdf` all build. The abstract is within the 250-word budget, the NeurIPS main content is 8 pages before References against the 9-page budget, the main generated tables are appendix-scoped, the representative CIFAR movement table is compact, and the introduction/discussion/limitations state the support-equivalence claim, scope of the claim, and tested-posterior limitation explicitly. `docs/venue_submission_compliance_audit.md` marks the content packet and NeurIPS 2026 style/checklist binding ready while keeping final upload readiness open because public code/data upload, external CI/GPU validation, and public repository state remain unverified. `docs/submission_handoff.md` records the exact upload files, title/abstract metadata, archive SHA256, source snapshot commit, local check commands, and final strict gate. | Substantial; final external receipts and venue polishing still open |

## Current Empirical Finding

The digits pilot shows a stable gap between SGLD-induced masks and random masks,
but dense magnitude masks explain IMP masks much better. Linear barriers are
near zero, suggesting the small setting may be dominated by one connected
low-loss region rather than meaningful posterior mode structure.

After corrected dataset-scaled multi-chain SGLD, posterior masks are almost
identical to chain-start magnitude masks in digits and MNIST smoke runs. This is
a strong warning against claiming posterior-mode discovery without stronger
sampling evidence.

The MNIST and Fashion-MNIST 5-seed, 4-sparsity Gate1 sweeps failed the
automated evaluator at every tested sparsity. The sweeps are summarized in
`docs/mnist_gate1_full_sweep.md` and `docs/fashion_gate1_full_sweep.md`.

- Posterior masks beat random masks.
- Posterior masks do not beat chain-start magnitude masks.
- Posterior masks remain highly overlapping with chain-start supports.
- Dense magnitude masks dominate posterior-induced masks.

This is currently evidence for the negative-result framing, not the original
positive equivalence claim.

The explicit mode/ticket distribution-equivalence audit strengthens this
negative framing at the proposal-metric level. Across the current grouped
KS/Wasserstein/MMD support-overlap comparisons from existing posterior
artifacts, posterior supports beat random masks in 58/59 groups, but beat the
matched chain-start support by more than 0.005 Jaccard in 0/59 groups.
Forty-three groups are practically tied to chain-start, 15 favor chain-start,
and one is mixed; rewind magnitude beats posterior by more than 0.005 Jaccard
in 55/57 groups.

The linear connectivity barrier audit separates landscape connectivity from
support equivalence. MNIST and Fashion-MNIST have nearly connected dense-to-IMP
paths (`0.0026` and `0.0395` mean barrier), whereas CIFAR-10 ResNet-20 long
SGLD/SWAG rows have large dense-to-IMP barriers (`3.0827` and `3.7402`). In
both regimes, posterior support remains tied to or below the chain-start
magnitude control, so linear barriers are orthogonal diagnostics rather than
evidence for posterior-ticket equivalence.

`docs/reviewer_objection_matrix.md` now compresses the evidence into nine
likely reviewer objections. Five are closed for current artifacts, two are
bounded open limitations, one is a partially closed positive-mechanism row, and
one is the remaining packaging limitation. This is useful for paper editing,
but it does not change the current status: the project is still not ready for
submission until the open posterior/permutation/package gaps are resolved or
explicitly accepted.

`docs/paper_submission_shape_audit.md` now makes the manuscript-shape gate
explicit. After the condensation pass, the current draft is shape-ready by this
local gate: the main body before generated tables is under the 850-line target
and `Current Results` is under the 450-line target while reviewer-objection
coverage remains visible. The project status is still not submission-ready
because the remaining blockers are external packaging/CI/GPU validation and the
explicitly bounded posterior/permutation robustness gaps, not main-text length.

`docs/submission_pdf_shape_audit.md` now checks the venue-facing main-only PDF.
`paper/main_submission.pdf` is built from the same source with appendix/generated
evidence tables excluded, and it is currently 9 pages against a 10-page local
budget. This separates the submission PDF from the appendix-inclusive
reproducibility PDF, `paper/main.pdf`.

`docs/venue_submission_compliance_audit.md` now checks the NeurIPS 2026 binding.
The content packet and venue binding are locally ready: anonymous author,
238-word abstract, official `paper/neurips_2026.sty`, checklist source,
figures, bibliography, appendix/checklist after References, and 8 main-content
pages before References against the 9-page NeurIPS budget. The same audit keeps
venue submission not ready because the public code/data release, external
CI/GPU validation, and public repository upload remain unverified. Local
compute-resource, asset-license, and new-asset metadata are now documented and
checked by the audit.
The release manifest is also guarded by
`docs/release_anonymization_audit.md`, which scans manifest-included text for
local usernames, hostnames, and absolute workstation paths before anonymous
review packaging is trusted.
`docs/public_release_archive_audit.md` now checks the local tarball
`dist/lottery_artifact_public_release_2026-05-06.tar.gz` against the manifest
member set plus release metadata sidecars.
`docs/public_release_archive_smoke.md` now extracts that tarball and confirms
the package passes `scripts/verify_research_artifacts.py --release-package-mode`.
`docs/external_validation_receipt_template.md` now pre-fills the current archive
SHA and source commit for receipt entry, while
`scripts/update_external_validation_receipts.py` validates externally observed
URLs/evidence before writing the receipt registry. `docs/external_validation_runbook.md`
converts these local facts into the four required public receipts: archive
upload, source repository, external CI, and external GPU-container validation.
`docs/submission_handoff.md` collects the corresponding submission UI and
supplement fields. These are local handoff artifacts; none replaces the external
receipts.

The direct proposal-level digits MLP probe adds literal distribution metrics
instead of only support-overlap summaries. With five IMP tickets and 50 SGLD
posterior sample masks, raw posterior sample masks fail the layer-sparsity KS
threshold (`p=0.0788`) and the mask-Hamming distribution-overlap threshold
(`0.6314 < 0.70`), while logit-space CKA and Hungarian matching pass. Mean-shift
parameter-space clustering collapses the posterior samples to one representative
mode versus five tickets, with basin entropy `0.0000` and effective cluster
count `1.0000`, which is inconsistent with the strong 1:1 mode/ticket claim.
This closes the small-model KS/MMD/Wasserstein/CKA/Hungarian gap, but does not
replace a full CIFAR posterior-mode pipeline.

The direct proposal metrics now also cover CIFAR-10 ResNet-20. With five seeds,
4096 training examples, width 8, and 25 SGLD posterior sample masks, the subset
row fails layer-sparsity KS (`p=1.10e-05`) and mask-Hamming overlap
(`0.6000 < 0.70`) while logit CKA/Hungarian matching pass (`0.8878`/`0.1122`).
Mean-shift clustering again collapses the posterior samples to one
representative mode versus five tickets, again with basin entropy `0.0000`.
A smaller three-seed CIFAR subset activation-CKA pilot finds three equal-size
parameter clusters and passing collapsed-mode metrics, but raw posterior sample
masks still fail layer KS and Hamming overlap.

The full-data CIFAR-10 ResNet-20 direct row is sharper. With 50 SGLD posterior
sample masks and five IMP tickets, raw posterior masks fail layer KS
(`p=5.3e-09`) and Hamming overlap (`0.0033 < 0.70`) while passing logit CKA
(`0.9369`), final-hidden activation CKA (`0.9172`), and both Hungarian-cost
thresholds. Mean-shift again collapses all posterior samples to one basin
(`H=0`, effective clusters `1`) versus five IMP tickets. This closes the
unaligned full-data direct-proposal gap in the negative direction.

The activation-channel-aligned full-data row does not rescue the proposal.
After mapping ResNet masks into the first seed dense-model channel frame,
aligned posterior samples still collapse to one basin and fail layer KS
(`p=2.3e-09`) plus Hamming overlap (`0.0000 < 0.70`), while logit CKA
(`0.9373`) and activation CKA (`0.9168`) still pass. The single aligned
representative also fails layer KS (`p=0.0413`).

The weight-correlation-aligned full-data row is also negative. Using
incoming/outgoing ResNet weight features for Hungarian channel matching,
aligned posterior samples still collapse to one basin and fail layer KS
(`p=1.2e-08`) plus Hamming overlap (`0.1290 < 0.70`), while logit CKA
(`0.9336`) and activation CKA (`0.9131`) still pass.

The alignment artifact audit makes the remaining permutation limitation
explicit. It re-reads seven full-data direct mode/ticket runs, verifies that
both aligned rows fail layer-KS/Hamming and that no audited direct row passes
full equivalence, and records that post-hoc exhaustive graph/permutation
realignment is not supported by the current direct-run artifacts because raw
posterior/ticket mask or state tensors were not saved. The direct probe now
supports `--save-mask-artifacts` and `--save-state-artifacts`, and the
fake-CIFAR mask-artifact smoke validates the resulting `.npz` schema and
parameter shapes. A post-hoc matching audit now reads that fixture and computes
record-level minimum-cost plus local channel-permutation mask comparisons; the
storage-budget audit estimates the reference full-data saved-artifact rerun at
about 284 MiB uncompressed. That activation-aligned full-data rerun is now
complete, its post-hoc audit verifies record-level matching over saved CIFAR
masks/states, and the structured global channel audit keeps posterior/ticket
Hamming around `0.21`. Exhaustive graph isomorphism over those saved artifacts
remains a separate robustness item.

The full-data multi-chain cyclical-SGLD direct row reduces the sampler-mixing
objection. It collects 75 posterior samples from three dense-start chains per
seed. The samples move from chain-start masks (`posterior-to-chain-start`
Hamming mean `0.0443`) and keep mean sample accuracy `0.8760`, but still
collapse to one parameter-PCA basin and fail layer KS (`p=3.3e-08`) plus
Hamming-distribution overlap (`0.2461 < 0.70`). Logit and activation CKA still
pass (`0.9327` and `0.9144`), so stronger posterior movement does not rescue
the proposal's mask-distribution equivalence claim.

The full-weight direct posterior rows close the strongest single-workstation
Gaussian objection. The rank-128 low-rank Laplace row improves Hamming overlap
to `0.8163`, but still collapses to one basin and fails layer KS (`p=2.0e-06`).
The streamed 270,896-parameter joint-group Laplace row uses exact covariance
groups over all ResNet-20 weight tensors. It preserves sample accuracy
(`0.8835`) and moves from chain starts (`posterior-to-chain-start` Hamming
`0.0503`), but it also collapses to one basin and fails layer KS (`p=1.1e-08`)
plus Hamming overlap (`0.0000 < 0.70`) while logit/activation CKA remain high
(`0.9373`/`0.9199`).

A small SGLD movement sweep found that higher temperature moves supports away
from chain starts, but does not move them toward IMP masks. This further favors
the negative-result direction.

A posterior-to-mask map smoke test found that mean, RMS, SNR, and variance maps
also stay near the chain-start control and far below dense magnitude alignment.

SNIP and SynFlow controls are now available. A smoke run shows they sit below
dense magnitude but above random, making them useful comparators for the
negative-result narrative. A Gem-Miner-style score-training control is now
implemented and smoke-tested. A five-seed full-data CIFAR selected row with 5
score-training epochs and 20 batches per epoch produces masks at 0.8319
sparsity whose support is random-like relative to IMP (Jaccard 0.0917
[0.0913, 0.0921]) and whose 30-epoch retraining accuracy is 0.8471
[0.8434, 0.8508], far below matched IMP accuracy 0.8970. This is useful
negative baseline evidence, though still a Gem-Miner-style implementation
rather than an exhaustive reproduction of every Gem-Miner training recipe.

The proposal-style variational pruning baseline is now directly implemented,
and a stronger hard-concrete L0 mask source is now implemented and tested
through support and calibration/OOD code paths. The real CIFAR subset support
smoke produces a sparse hard-concrete mask at 0.5099 sparsity with 0.3216
Jaccard overlap to IMP, validating the path; the full-data five-seed row is
decisively negative, retraining to 0.2766 accuracy with 0.0922 Jaccard overlap
to IMP.
On a five-seed digits MLP row at 0.657 sparsity, variational pruning retrains
to 0.9633 accuracy, above random masks at 0.9489 and Gem-Miner-style masks at
0.9550, but below IMP at 0.9711 and dense at 0.9739. Its ECE is 0.0250 versus
IMP's 0.0211, and Brier is 0.0532 versus IMP's 0.0410, so this small-model H3
row does not support the proposed calibration advantage. The learned-mask
calibration/OOD path now has a full-data five-seed CIFAR-10/CIFAR-100 row.
Learned-random, Gem-Miner-style, and variational-prune masks lower ECE to
0.0270, 0.0283, and 0.0255, respectively, but reach only 0.8449, 0.8418, and
0.8301 accuracy, with MSP OOD AUROC 0.7897, 0.7853, and 0.7754. IMP remains
better on accuracy, NLL, Brier, and OOD AUROC, so the CIFAR-scale H3
calibration/OOD row also fails as a rescue. The matched CIFAR support rows are
even more direct: variational pruning reaches 0.8306 accuracy and only 0.0907
Jaccard overlap to IMP, while hard-concrete retrains to 0.2766 accuracy with
0.0922 Jaccard overlap to IMP, essentially the random/Gem-Miner support scale.

Small-model HMC is now stronger than the first conservative sweep. A 5-seed
tuned full-batch HMC baseline on digits r2 p0.30 reaches 0.9201 mean HMC sample
accuracy, accept rate 0.95, state clusters 3.0, and function clusters 2.4. HMC
support moves away from dense/chain-start magnitude (`post-chain = 0.4095`) and
beats random (`0.4136` vs. `0.3254`), but it still fails Gate1 because
dense/chain-start magnitude is much closer to IMP (`0.8736`). This materially
reduces the "SGLD artifact" concern, but it is still a small-model posterior
baseline rather than a CIFAR-scale high-fidelity posterior.

The proposal's variational-pruning baseline now has first direct coverage.
On a five-seed digits MLP row, Bernoulli/Concrete variational pruning reaches
0.9633 accuracy, above random masks at 0.9489 and Gem-Miner-style score masks
at 0.9550, but below IMP at 0.9711 and dense training at 0.9739. Its support
overlap with IMP is 0.3822, higher than random at 0.2151 but far from the IMP
mask. Its ECE/Brier are 0.0250/0.0532 versus IMP's 0.0211/0.0410, so this
small-model row does not support the proposed calibration advantage. This is
useful H3 evidence, not a CIFAR-scale rescue.

A selected 5-seed CIFAR short SGLD movement diagnostic shows the same movement
failure at image scale. Increasing SGLD LR moves support away from the dense
chain start (`post-chain = 0.6932` at `1e-6` and `0.5440` at `3e-6`), but
posterior-to-IMP overlap decreases from 0.1730 to 0.1683 and 0.1602
respectively. This is repeated-seed diagnostic evidence that movement is not
ticket-directed, though it is still a short-budget CIFAR posterior baseline.

A selected 5-seed CIFAR short SGHMC movement diagnostic adds a momentum-based
posterior dynamics control. SGHMC moves supports away from chain starts while
keeping usable sample accuracy (`post-chain = 0.7733`, sample accuracy 0.8178
at `3e-8`; `post-chain = 0.6329`, sample accuracy 0.8107 at `1e-7`), but
posterior-to-IMP overlap falls from 0.1702 to 0.1682 and 0.1637. It records 6.0
state clusters, yet the support-level result still fails the posterior-mode
rescue condition.

First SWAG baselines are now available on MNIST and Fashion-MNIST r5 p0.30
across 5 seeds. Both fail Gate1 in the same way as SGLD: posterior masks beat
random but do not beat chain-start magnitude, remain highly overlapping with
chain-start supports, and are dominated by dense magnitude. This reduces the
"SGLD artifact" concern, but it is still not a full stronger-posterior sweep
because it covers one representative sparsity per dataset.

Fashion-MNIST now reproduces the MNIST Gate1 failure in a full 5-seed,
4-sparsity sweep. At the highest tested Fashion-MNIST sparsity, IMP accuracy
drops below dense accuracy, so that row is best interpreted as a high-sparsity
stress case; it still fails Gate1 in the same way.

The full seed/sparsity sweep runner is scripted in
`scripts/run_gate1_sparsity_sweep.py`, now with a `--run-prefix` option so smoke
and full sweeps do not mix.

The CIFAR-10 data blocker is resolved for local runs: the canonical CIFAR-10
Python tarball is present under `data/`. The previous subset smoke was
near-chance, but the full-data training path now works: a 10-epoch ResNet-20
baseline with crop/flip augmentation and cosine LR reached 0.8302 test accuracy.
A full-data short Gate1 grid now exists. It failed Gate1 in all rows with
meaningful dense/IMP accuracy; the representative r5 p0.30 SGLD row is now 5
seeds, while r2/r8 remain 3 seeds. This is a major step beyond the subset
smoke, but it is still not submission-grade CIFAR evidence because it uses a
short 10-epoch budget, one SGLD chain, and 10 posterior samples per seed. A
matching 5-seed CIFAR r5 p0.30 SWAG short control also fails Gate1: posterior
overlap is 0.1302, chain-start magnitude overlap is 0.1304, and
posterior-to-chain-start overlap is 0.9097. A 5-seed r5 p0.30 SGLD
multi-chain short control with 3 independent dense starts per seed also fails:
posterior overlap is 0.1291, chain-start magnitude overlap is 0.1291,
posterior-to-chain-start overlap is 0.9963, and both state and function
cluster counts are 3.0. One-seed 30-epoch initialization-rewind pilots reached
about 0.884 dense accuracy but only about 0.858 IMP accuracy at r5-level
sparsity, so simply training longer or pruning more gradually was not enough.
An epoch-1 rewind pilot fixes that IMP issue across 5 seeds: dense accuracy is
0.8859 and IMP accuracy is 0.8980 at 0.8319 sparsity. It still fails Gate1:
posterior overlap is 0.1342, chain-start magnitude overlap is 0.1342,
posterior-to-chain-start overlap is 0.9969, and epoch-1 rewind magnitude
overlap with IMP is higher at 0.1783. A matched 5-seed long-budget SWAG control
also fails: posterior overlap is 0.1361, chain-start magnitude overlap is
0.1361, posterior-to-chain-start overlap drops to 0.9265, and epoch-1 rewind
magnitude overlap remains higher at 0.1786.
A 5-seed long-budget SGLD multi-chain control also fails despite recording
separated chains: posterior overlap is 0.1368, chain-start magnitude overlap is
0.1368, posterior-to-chain-start overlap is 0.9969, state clusters average 3.0,
function clusters average 3.2, and epoch-1 rewind magnitude overlap remains
higher at 0.1800.
A 5-seed long-budget SGLD movement diagnostic closes the corresponding
"too-little SGLD movement" objection for the epoch-1 rewind setting. Dense
accuracy is 0.8845 and IMP accuracy is 0.8993. Raising SGLD LR from `1e-10` to
`1e-6` reduces posterior-to-chain-start overlap from 0.9969 to 0.7362 while
sample accuracy remains 0.8753, but posterior-to-IMP overlap drops from 0.1441
to 0.1425. At `3e-6`, post-chain falls to 0.5928, sample accuracy falls to
0.8593, and posterior-to-IMP drops further to 0.1381. Movement again is not
ticket-directed, while epoch-1 rewind magnitude remains closer to IMP at
0.1784.
A matched 5-seed long-budget SGHMC movement diagnostic gives the same result
with momentum dynamics. Dense accuracy is 0.8883 and IMP accuracy is 0.8970.
Raising SGHMC LR from `1e-10` to `1e-7` reduces posterior-to-chain-start
overlap from 0.9876 to 0.6796 while sample accuracy remains 0.8752, but
posterior-to-IMP overlap drops from 0.1456 to 0.1419. At `3e-7`, post-chain
falls to 0.5214 and posterior-to-IMP drops further to 0.1360. SGHMC records
6.0 state clusters and up to 3.2 function clusters, but the support still fails
to beat chain-start magnitude 0.1457 or epoch-1 rewind magnitude 0.1777.
A 5-seed long-budget cyclical SGLD movement diagnostic adds a stronger
exploration baseline. With 400 posterior steps, 50-step cycles, and second-half
cycle sampling, the `1e-6` setting reduces posterior-to-chain-start overlap to
0.7046 while keeping sample accuracy 0.8782, but posterior-to-IMP drops to
0.1422 versus chain-start magnitude 0.1454 and epoch-1 rewind magnitude 0.1789.
At `3e-6` and `1e-5`, post-chain falls to 0.5533 and 0.3700, while
posterior-to-IMP falls further to 0.1371 and 0.1260.

A full-network SWAG movement diagnostic with 20 snapshots and learning rate
`1e-3` gives the same pattern. At scale `1.0`, posterior-to-chain-start
overlap is 0.9778, sample accuracy is 0.8813, and posterior-to-IMP is 0.1455,
tied with chain-start magnitude. At scales `16` and `64`, post-chain falls to
0.9528 and 0.9086, but posterior-to-IMP remains 0.1454 and 0.1453 while sample
accuracy falls to 0.8636 and 0.8041. Rewind magnitude remains closer to IMP at
0.1782.

A 5-seed long-budget diagonal Laplace movement diagnostic gives the same
answer with a local curvature-weighted Gaussian approximation. This is a
mini-batch diagonal empirical-Fisher approximation rather than exact Hessian,
KFAC, or full-covariance Laplace. Dense accuracy is 0.8849 and IMP accuracy is
0.8980. At scale `1e-3`, posterior-to-chain-start overlap falls to 0.8826 and
sample accuracy remains 0.8799, but posterior-to-IMP falls from 0.1469 to
0.1447. At scale `3e-3`, post-chain falls to 0.7803 and posterior-to-IMP falls
to 0.1400. At scale `1e-2`, post-chain falls to 0.5961 and posterior-to-IMP
falls further to 0.1278. The diagonal Laplace row therefore reduces the
"SGLD-family artifact" concern, but it does not replace a richer structured or
exact CIFAR posterior baseline.

A 5-seed long-budget KFAC-style Laplace movement diagnostic now fills that next
CIFAR curvature baseline. It is still approximate: it uses mini-batch
empirical-Fisher Kronecker factors for Linear and Conv2d weights, not exact
Hessian or full-covariance Laplace. Dense accuracy is 0.8860 and IMP accuracy
is 0.8956. At scale `1e-3`, posterior-to-chain-start overlap falls to 0.8016
and sample accuracy remains 0.8839, but posterior-to-IMP falls from 0.1456 to
0.1441. At scale `1e-2`, post-chain falls to 0.4859 and sample accuracy remains
0.8695, but posterior-to-IMP falls to 0.1303. This reduces the
"diagonal-only Laplace" concern, though a full-network exact/full-covariance
CIFAR posterior remains missing.

A 5-seed full-network low-rank Hessian-plus-diagonal Laplace movement
diagnostic narrows the covariance objection further. It combines a
20-minibatch diagonal empirical-Fisher precision with randomized rank-16,
rank-32, rank-64, and rank-128 top-Hessian precision over all trainable parameters; all
seeds retain
the requested positive Hessian directions. At scale `1e-3`, ranks 16--64 are
practically tied to chain-start support. At scale `1e-2`, rank-16
posterior-to-chain-start overlap falls to 0.7359, rank-32 falls to 0.7402, and
rank-64 falls to 0.7397; the selected rank-128 row falls to 0.7358. But
posterior-to-IMP drops to 0.1351, 0.1358, 0.1339, and 0.1351 while sample
accuracy remains 0.8784, 0.8789, 0.8816, and 0.8813. Rewind magnitude remains
much closer to IMP at 0.1777, 0.1795, 0.1766, and 0.1780.

A 5-seed long-budget exact final-head Laplace probe now adds a limited
full-covariance CIFAR check. The feature extractor is frozen and the exact
softmax-cross-entropy Hessian is computed for the 650-parameter final linear
head. Because only the head is sampled, the primary diagnostic is head-level
support. Dense accuracy is 0.8856 and IMP accuracy is 0.8957. At scale `1e-3`,
head posterior-to-chain-start overlap falls to 0.7773 while sample accuracy
remains 0.8856, but head posterior-minus-chain-start is negative on average
(-0.0085). At scale `1e-2`, head post-chain falls to 0.6912 and the
posterior-minus-chain-start CI is fully negative (-0.0284 [-0.0510, -0.0058]).
At scale `1`, head post-chain is 0.6769, sample accuracy is 0.8835, and the
posterior-minus-chain-start CI is also fully negative (-0.0352
[-0.0582, -0.0122]). The head rewind magnitude support is closer to IMP at
0.7191. This is not full-network posterior evidence, but it materially reduces
the concern that covariance within the classifier head rescues the
posterior-mode interpretation.

Full-covariance block Laplace probes extend the exact-covariance check beyond
the final classifier head. For `layer1.0.conv1.weight`, a 2304-parameter
softmax-GGN/Laplace covariance at scale `1e-3` keeps sample accuracy at 0.8961
and moves block support away from the dense chain start (block post-chain
0.2236), but block posterior-to-IMP is 0.1959 versus block chain-start 0.2034
and block rewind magnitude 0.2423. A seed-0 scan over seven small and
medium-sized tensors found no broad block-level rescue. Its only mildly positive
candidate, `layer3.0.shortcut.0.weight`, also fails after five seeds: block
posterior-to-IMP is 0.2402 versus block chain-start 0.2411, block post-chain is
0.3626, sample accuracy is 0.8905, and block rewind magnitude is 0.3050. These
rows are stronger than diagonal or KFAC covariance for the selected tensors. A
joint four-tensor row adds cross-tensor covariance over 5424 parameters
(`conv1.weight`, `layer1.0.conv1.weight`, `layer3.0.shortcut.0.weight`, and
`fc.weight`): sample accuracy is 0.8922 and group post-chain is 0.5088, but
group posterior-to-IMP is 0.3294 versus group chain-start 0.3501 and group
rewind magnitude 0.3637. Wider exact covariance rows sample many tensors
simultaneously. The 68k block-diagonal row covers 16 tensors, preserves sample
accuracy 0.8802, and moves from chain-start support (global post-chain 0.7400),
but block posterior-chain is -0.0050 and global posterior-chain is only
+0.0010. The 68k joint-group row packs those tensors into 8 covariance groups,
preserves sample accuracy 0.8811, and moves farther (global post-chain 0.7148),
but block posterior-chain is -0.0050 and global posterior-chain is only
+0.0015. The 86,576-parameter joint-group row adds the first stage-3
convolution block in 6 groups; it preserves sample accuracy 0.8828 and has
global post-chain 0.7863, but block posterior-chain is -0.0023 and global
posterior-chain is only +0.0006. The streamed 270,896-parameter joint-group row
covers all 22 ResNet-20 weight tensors in 8 groups; it preserves sample
accuracy 0.8824 and has global post-chain 0.7389, but block/global
posterior-chain is -0.0019 and rewind remains closer by 0.0362. These remain
structured exact covariance approximations rather than a dense all-parameter
full-covariance posterior.

5-seed long-budget full-network subspace HMC probes add tractable CIFAR-scale
HMC checks. The random-basis probe samples an 8-dimensional random orthonormal
subspace around the dense ResNet-20 checkpoint with direction scale 10,
full-data deterministic HMC potential, frozen batchnorm statistics, step size
`3e-3`, and 20 HMC steps. Dense accuracy is 0.8866, IMP accuracy is 0.8965, HMC
accept rate is 0.7400, sample accuracy is 0.8863, and mean parameter distance
from the dense checkpoint is 0.3672. The support result still fails Gate1:
posterior-to-IMP is 0.1440 versus chain-start magnitude 0.1440. The
trajectory-basis probe samples the 6-dimensional subspace spanned by dense
trajectory checkpoint directions. It starts from a stronger dense trajectory
magnitude support, but again fails the rescue test: at HMC step size `1e-3`,
accept rate is 0.6900, sample accuracy is 0.8847, post-chain is 0.9915, and
posterior-to-IMP is 0.2290 versus chain-start magnitude 0.2292. A top-Hessian
subspace variant samples a 4-dimensional randomized Hessian eigenspace around
the dense checkpoint. It has accept rate 0.8600 and sample accuracy 0.8865, but
the posterior support remains almost identical to the chain-start support:
post-chain is 0.9999 and posterior-to-IMP exceeds chain-start by only
0.000002. A 16-dimensional top-Hessian variant widens the curvature-informed
subspace across 5 seeds. It accepts at 0.8833, keeps sample accuracy at
0.8881, and increases mean parameter distance to 0.00949, but post-chain is
still 0.9994 and posterior-to-IMP is 0.14680 versus chain-start 0.14682. A
five-seed 32-dimensional top-Hessian selected row also fails to produce
ticket-directed movement: at step `3e-4`, accept rate is 0.9500, sample
accuracy is 0.8872, parameter distance is 0.0104, post-chain is 0.9993, and
posterior-to-IMP is 0.14614 versus chain-start 0.14611. These probes reduce the
concern that only Gaussian local posterior
approximations were tested at CIFAR scale, but they remain low-dimensional
subspace checks rather than exact full-network full-covariance posterior
evidence.

A 5-seed CIFAR-10 calibration and CIFAR-100 OOD probe now fills the first
uncertainty-evaluation gap. In the long-budget epoch-1 rewind setting, dense
accuracy is 0.8866, NLL is 0.3536, ECE is 0.0353, and maximum-softmax OOD
AUROC is 0.8230. IMP improves accuracy to 0.8953, NLL to 0.3387, and OOD
AUROC to 0.8306, but has slightly worse ECE at 0.0393. A ten-sample SWAG
predictive ensemble improves ECE to 0.0285 but lowers accuracy to 0.8688,
worsens NLL to 0.4018, and lowers OOD AUROC to 0.8050. This suggests that
posterior predictives can change uncertainty behavior without yielding a
ticket-support explanation.

A 5-seed matched dense-trajectory probe now directly tests the negative
paper's alternative explanation. It uses a single dense ResNet-20 trajectory,
takes the epoch-1 checkpoint as the IMP rewind state, and compares magnitude
supports from epochs 0, 1, 2, 5, 10, 20, and 30 to the final IMP mask. Dense
accuracy is 0.8841 and IMP accuracy is 0.8963. The epoch-1 rewind support has
Jaccard 0.1782 to IMP; the trajectory then rises to 0.2197 at epoch 5, peaks
at 0.2342 at epoch 10, and remains 0.2312 at epoch 30. Aggregating the same
trajectory by RMS absolute magnitude raises overlap to 0.2400, while movement
and path-length score masks remain much weaker. These supports are far closer
to IMP than the CIFAR posterior movement supports, which peak around 0.147.
This materially strengthens the trajectory/magnitude-subspace account and
narrows it to persistent trajectory magnitude rather than movement alone.

A 5-seed trajectory mask retraining probe adds a functional check. All fixed
masks are trained for 30 epochs from the same epoch-1 rewind state at the IMP
sparsity. The IMP retrain reaches 0.8983 accuracy, the final dense-magnitude
mask reaches 0.8826, RMS/mean trajectory-magnitude masks reach about 0.874,
epoch-10 reaches 0.8730, path/movement/epoch-1 masks are 0.854--0.857, and a
random mask reaches 0.8422. This means trajectory-magnitude masks are genuinely
trainable and far better than random or pure movement masks, but they do not
recover the IMP advantage. The remaining trajectory-side gap is causal: explain
what IMP refines beyond the dense trajectory magnitude subspace.

A 5-seed residual-swap probe directly tests that refinement. Starting from
trajectory masks, swapping half of base-only support for IMP-only support
recovers substantial accuracy: final dense mask 0.8797 -> 0.8882, RMS
trajectory mask 0.8733 -> 0.8851, and epoch-10 mask 0.8712 -> 0.8855. The
same-size non-IMP random residual controls stay low at 0.8780, 0.8705, and
0.8704. This is current causal evidence that the IMP residual support carries
specific functional information rather than merely changing mask distance.

A 5-seed residual-anatomy probe then characterizes that support. The final
dense, RMS trajectory, and epoch-10 bases each miss about 27.8k--28.4k
IMP-kept weights, with support-to-IMP Jaccard 0.2323, 0.2412, and 0.2359.
Base-only weights are removed throughout IMP, with mean pruning round
2.90--2.97 rather than a final-round-only correction. The residual is only
mildly stage-structured: for RMS trajectory, stage 2 is enriched 1.1348x, while
stage 3 contains 74.4% of IMP-only residual but is near its size share. A
held-out logistic predictor from dense-trajectory rank features plus stage
indicators reaches only AUC 0.6165--0.6206 and top-k recall 0.2087--0.2206.
Thus dense trajectory statistics explain useful base masks but do not
reconstruct the IMP-only residual.

A functional residual-predictor mask probe closes the immediate predictor
question. The held-out predictor raises added IMP-only precision above random
controls (0.1834--0.1866 versus 0.1233--0.1253), but the generated masks do not
recover the oracle residual accuracy gain. Final dense reaches 0.8793 with
predictor residual, 0.8805 with random residual, and 0.8892 with oracle
residual; RMS trajectory reaches 0.8744, 0.8744, and 0.8866 respectively; and
epoch 10 reaches 0.8728, 0.8724, and 0.8890. This indicates that marginally
predicting residual coordinates is insufficient to reconstruct the functional
IMP-only residual subnetwork.

A cross-seed residual-transfer probe closes the immediate transfer question.
Training on four source seeds raises target-seed added IMP-only precision above
random controls (0.2238--0.2413 versus 0.1246--0.1264), but again does not
recover oracle residual accuracy. Final dense reaches 0.8776 with cross-seed
residual, 0.8781 with random residual, and 0.8905 with oracle residual; RMS
trajectory reaches 0.8731, 0.8725, and 0.8878 respectively; and epoch 10
reaches 0.8745, 0.8726, and 0.8890. The residual signal is therefore
seed-transferable at the coordinate level, but still not sufficient to
instantiate the functional IMP-only residual support.

A direct cross-seed residual-support transfer probe closes a stricter
seed-invariant support question. Source-vote additions from other seeds' oracle
residual coordinates are slightly enriched for target IMP-only weights
(0.143--0.152 added precision versus 0.123--0.125 target-random precision), but
they do not transfer the functional residual gain. Final dense base/source-vote/
source-vote-random/target-random/oracle accuracies are 0.8808, 0.8779, 0.8794,
0.8788, and 0.8877; RMS trajectory gives 0.8738, 0.8715, 0.8730, 0.8729, and
0.8866; epoch 10 gives 0.8709, 0.8688, 0.8727, 0.8714, and 0.8866. The
functional residual is therefore not a directly transferable coordinate set
shared across seeds.

A residual base-compatibility probe refines the trajectory-subspace account.
Matched random bases preserving the same per-parameter IMP/non-IMP counts and
the same base-to-IMP overlap are weak alone: final dense, RMS trajectory, and
epoch-10 matched-base accuracies are 0.8641, 0.8605, and 0.8607, compared with
trajectory-base accuracies 0.8827, 0.8743, and 0.8744. But top oracle IMP-only
residual additions recover 0.8926, 0.8910, and 0.8942, matching or exceeding
trajectory-oracle accuracies 0.8893, 0.8892, and 0.8892. Matched random
residual additions remain weak at 0.8649, 0.8628, and 0.8636. Exact
trajectory-base identity is therefore not necessary once target IMP overlap
and top residual identity are fixed.

A residual posterior-decomposition probe separates final IMP membership from
dense magnitude and posterior-uncertainty ordering within the IMP-only residual
candidate pool. With the same matched random bases, top oracle IMP-only
additions reach 0.8915, 0.8928, and 0.8911 for final dense, RMS trajectory, and
epoch 10. Uniformly random IMP-only additions reach 0.8783, 0.8795, and 0.8791
with about 0.50 oracle-overlap precision. Diagonal-Laplace posterior-RMS-ranked
IMP-only additions reach 0.8852, 0.8834, and 0.8829 with 0.551--0.556
oracle-overlap precision, but dense final magnitude nearly matches this at
0.8821, 0.8812, and 0.8827 with 0.553--0.557 overlap. Posterior
RMS-minus-dense and posterior standard deviation fall to 0.8745--0.8770 and
0.8710--0.8717 with 0.478--0.479 and 0.446--0.450 overlap. Final IMP
membership is therefore useful but not sufficient; posterior RMS is mostly a
dense-magnitude signal rather than a posterior-uncertainty signal.

A residual stratified-control probe closes the coarse-structure question. All
generated controls remove the same low-base-score weights as the oracle. Random
IMP-only additions recover part of the gap but stay below oracle: final dense
0.8818 versus 0.8872, RMS trajectory 0.8794 versus 0.8858, and epoch 10
0.8764 versus 0.8854. Non-IMP additions matched to the oracle by parameter
tensor and within-parameter score decile match more than 99.9% of oracle strata
but remain near weak base controls: 0.8758, 0.8716, and 0.8685. The residual
gain is therefore not explained by coarse allocation or score-bin structure.

A residual removal-order control closes the low-score-removal artifact
question. Holding the top-IMP additions fixed, low/random/high base-only
removals reach 0.8881/0.8883/0.8906 for final dense, 0.8874/0.8896/0.8914 for
RMS trajectory, and 0.8862/0.8922/0.8920 for epoch 10. Same-size non-IMP
random additions under low removal remain much weaker at 0.8779, 0.8709, and
0.8701. The residual gain is therefore driven by the added IMP-only weights,
not by preferentially removing weak base weights.

An IMP-process residual probe now closes the first process-timing question.
Round-survivor additions increasingly concentrate final IMP residual support:
final-IMP precision is about 0.43--0.44 at round 1, about 0.75 at round 3, and
1.0 at round 5 across the final dense, RMS trajectory, and epoch-10 bases.
Accuracy also moves toward the final oracle residual as rounds progress: final
dense base/oracle/round-1/round-3/round-5 accuracies are 0.8793, 0.8898,
0.8792, 0.8829, and 0.8867; RMS trajectory gives 0.8736, 0.8881, 0.8771,
0.8832, and 0.8857; epoch 10 gives 0.8724, 0.8884, 0.8772, 0.8826, and
0.8861. The round-5 process masks still trail the final oracle and overlap only
about 0.67 of the oracle added subset, so the evidence supports gradual
process-specific construction rather than an independent dense-trajectory
coordinate rule.

An IMP-process ranking-control probe further separates survivor-set membership
from score ordering. Top-score round survivors beat random and low-score round
survivors across bases. For RMS trajectory, top/random/low accuracies are
0.8788/0.8733/0.8738 at round 1, 0.8830/0.8759/0.8726 at round 3, and
0.8857/0.8803/0.8759 at round 5. At round 5 all variants are final-IMP
residual by construction, but top-score additions overlap about 0.68 of the
oracle added subset versus about 0.50 for random and 0.32 for low-score
additions. This strengthens the claim that the IMP process contributes an
ordering, not just a final residual membership set.

An oracle-overlap-matched process control now tests the strongest remaining
confound in that ranking result. It samples random final-IMP residual additions
with the same final-oracle overlap count as the round-score-selected
additions. Across the 5-seed CIFAR row, round-score final-IMP residuals beat
the matched-random control in 35 of 45 paired base/round/seed comparisons,
with mean accuracy delta `+0.0020`. This is a small effect, but it means the
process ranking signal is not explained solely by final-IMP membership or by
oracle-overlap amount.

A score-source process control fixes the final-IMP residual candidate set and
support budget, but replaces the round-trained score ordering with dense-final
or base-source magnitudes. Across 45 paired comparisons, round-trained scores
beat dense-score controls in 37/45 cases with mean delta `+0.0026`, and beat
base-score controls in 39/45 cases with mean delta `+0.0028`. The effect is
again small, but it further separates IMP-process ordering from dense/base
magnitude ranking.

A round-exclusion process intervention now fixes the final-IMP residual
candidate set and support budget, removes the round-selected final-IMP
additions, and chooses the best remaining final-IMP residual additions by final
IMP magnitude. Across 45 paired comparisons, round-selected masks beat this
replacement in 44 cases, with mean accuracy delta `+0.0061`. This is the
strongest current process-side evidence because the replacement is allowed to
use an oracle-like final IMP magnitude score after the process-selected
coordinates are ablated.

A tensor-matched round-exclusion row now focuses on the strongest RMS
trajectory round-5 process row. After removing the round-selected final-IMP
residual coordinates, the replacement is matched by parameter tensor and then
chosen by final IMP magnitude. Across five seeds, round-selected masks reach
`0.8855` accuracy versus `0.8764` for the tensor-matched replacement, with
mean paired delta `+0.0091` and `5/5` positive deltas. This closes the simpler
layer/tensor-composition objection for that strongest process row.

A stricter tensor+score-matched round-exclusion row then matches replacement
coordinates by parameter tensor and within-tensor round-score decile before
applying final IMP magnitude. This makes the replacement much more competitive:
final-oracle overlap rises from `0.4167` for tensor-only matching to `0.6440`.
Even so, round-selected masks reach `0.8878` accuracy versus `0.8837` for the
tensor+score-matched replacement, with mean paired delta `+0.0041` and `5/5`
positive deltas. This closes the stronger score-bin-composition objection for
the strongest process row.

A residualized-score projection row then tests whether the useful round-score
ordering survives after removing a simple magnitude subspace. Inside the same
RMS trajectory round-5 final-IMP residual candidate pool, the round score is
linearly residualized against the base-source, dense-final, and final-IMP
magnitude scores before ranking. The residualized-score mask reaches `0.8811`
accuracy versus `0.8852` for the original round score, with mean paired delta
`+0.0041` and `5/5` positive deltas for round over residualized. Final-oracle
overlap drops from `0.6684` to `0.4854`. This does not make the process signal
larger; it localizes the useful ordering to the trajectory/final-magnitude
subspace rather than a standalone orthogonal residual score.

A posterior-residualized projection row then removes diagonal-Laplace posterior
RMS, posterior standard deviation, and posterior RMS-minus-dense scores in
addition to the base/dense/final-IMP magnitude subspace. In the same RMS
trajectory round-5 setting, the round-selected row reaches `0.8847` accuracy
versus `0.8825` for the posterior-residualized score, with `5/5` positive
seed deltas and mean paired delta `+0.0023`. The accuracy CI touches zero, but
the coordinate result is sharp: final-oracle overlap drops from `0.6773` to
`0.4850`, a `+0.1923` overlap delta with `5/5` positive seeds. This reduces
the posterior-projection objection without claiming the posterior-residualized
mask is functionally useless.

A learned-subspace residualized projection row then asks whether a low-rank
process model can replace the exact round-selected coordinates. It learns the
top rank-8 PCA component scores from dense trajectory scores, final-IMP
magnitude, and earlier IMP-round scores inside the final-IMP residual
candidate pool, then residualizes the current round score against those
components. The round-selected row reaches `0.8869` accuracy versus `0.8821`
for the learned-subspace residualized score, with mean paired delta `+0.0048`
and `5/5` positive seed deltas. Final-oracle overlap drops from `0.6807` to
`0.4917`, a `+0.1890` overlap delta with `5/5` positive seeds. This closes the
learned trajectory/process subspace replacement objection under the current
rank-8 control.

A working negative-result draft now exists under `paper/` and compiles to
`paper/main.pdf`, `paper/main_submission.pdf`, and the official-style
`paper/neurips_submission.pdf`. The large generated evidence tables are
appendix-scoped, the main CIFAR movement table has been shortened to
representative rows with the full generated movement table kept in the appendix,
and the manuscript now states a scoped support-equivalence claim rather than a
theorem about all Bayesian neural-network posteriors. It is not upload-ready
because the strongest remaining objections are not fully closed: public release
upload, external CI/GPU validation, clean public repository state, plus the
bounded full-network exact dense covariance/permutation robustness limitations.
`docs/external_validation_receipts.json` now makes the four external release
blockers explicit fill-in receipts, while
`docs/external_validation_receipt_template.md`,
`docs/external_validation_readiness_audit.md` and
`runs/external_validation_readiness_audit.json` separate the ready local archive
plus source-only repository snapshot from the still-missing external evidence.
`make external-validation-strict`
should remain failing until those receipts are real.
The current full-data
evidence includes long-budget multi-chain, cyclical-SGLD, diagonal-Laplace,
KFAC-style Laplace, exact final-head Laplace, random/trajectory/top-Hessian
low-dimensional subspace-HMC, selected-block, joint-block, 22k/68k-parameter
block-diagonal full-covariance Laplace, and 68k/86k/270k-parameter joint-group
full-covariance Laplace, CIFAR-100
OOD/calibration diagnostics, and unaligned plus
activation-aligned plus weight-correlation-aligned plus multi-chain
cyclical-SGLD direct full-data mode/ticket probes plus the activation-aligned
full-data saved-artifact rerun. The alignment artifact audit bounds the old
artifact gap, the saved-artifact post-hoc audit closes the record-level
matching part of that gap, and the structured global channel audit shows that
channel relabeling does not rescue the saved CIFAR supports; exhaustive graph
isomorphism remains open.

## Next Gate

The next gate is not "scale up immediately." It is:

> After activation alignment and multi-chain posterior movement checks, do
> posterior-induced supports explain IMP masks beyond dense magnitude,
> initialization magnitude, and pruning-process controls?

If no, the paper should pivot toward a negative result: winning tickets are
trajectory/subspace objects rather than posterior-mode objects.
