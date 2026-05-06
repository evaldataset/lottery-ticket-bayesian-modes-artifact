# Thread Goal Completion Audit

Date: 2026-05-07

Objective under audit:

> top conf에 제출할 수준의 논문과 연구 성과가 나올 수 있도록 이 연구
> 주제에 대한 모든 것을 진행해

Source proposal:

- `proposal_A3_lottery_ticket_bayesian_modes.md`

Conclusion:

The objective is not complete. The project has a strong negative-result draft
and substantial CIFAR-10 evidence, including unaligned, activation-aligned,
weight-correlation-aligned, multi-chain cyclical-SGLD full-data direct
mode/ticket probes, an independent-start multi-chain cyclical-SGLD full-data
direct probe, a rank-128 low-rank Laplace full-data direct probe, a
270,896-parameter streamed joint-group Laplace full-data direct probe, a
22,064-parameter exact tensor-block-diagonal Laplace row, a 68,144-parameter
exact tensor-block-diagonal Laplace row, a 68,144-parameter exact joint-group
Laplace row with within-group cross-tensor covariance, an 86,576-parameter
exact joint-group Laplace row that adds a stage-3 convolution block, a streamed
270,896-parameter exact joint-group Laplace row that covers all ResNet-20
weight tensors, and a 20-snapshot full-network SWAG movement row plus
rank-16/rank-32/rank-64/rank-128 full-network low-rank Hessian-plus-diagonal
Laplace movement rows, plus a tiny exact dense full-network Laplace sanity row
over all 310 trainable parameters of a digits MLP and a fake-CIFAR ResNet-20
width-1 exact dense full-network smoke over 1,229 parameters, plus a six-row
linear connectivity barrier audit showing that barriers do not rescue
posterior-ticket support equivalence, and a reviewer-objection matrix that
maps nine likely review risks to artifact-backed answers, plus a paper
submission-shape audit that now marks the condensed current draft shape-ready
under the local main-text gate, but it is still below top-conference submission
standard because the proposal's strongest
posterior-mode claims still have open robustness gaps: exact/full-covariance
full-network CIFAR posterior evidence, broader learned-mask
distribution/permutation variants beyond the current support rows, and broader
process-intervention robustness for the IMP residual. A new alignment artifact
audit now bounds the current permutation evidence: first-order activation and
weight-correlation alignment rows are negative, but post-hoc exhaustive
graph/permutation realignment is not supported by the current direct-run
artifacts because raw posterior/ticket mask or state tensors were not saved.
The direct probe now has explicit mask/state artifact saving flags, a
fake-CIFAR schema fixture with parameter shapes, and a post-hoc matching audit
over that fixture including local channel-permutation matching. The
activation-aligned full-data saved-artifact rerun has now been performed and
its audit verifies record-level post-hoc matching over saved CIFAR masks and
states. A structured global channel audit over those saved artifacts keeps
posterior/ticket Hamming near `0.21`, so simple channel relabeling does not
rescue support equivalence. An exact stage-1 enumeration feasibility audit now
checks all `128` channel assignments on a 270-parameter fake-CIFAR subgraph and
shows the coordinate solver matches the exact optimum there; the full CIFAR
artifact is sized at about `10^840.4` channel assignments per record pair. The
remaining gap is therefore exhaustive full-data graph isomorphism, not
rerunning, storage, or the basic channel-matching software path.

## Concrete Success Criteria

The broad objective is treated as these deliverables:

| Criterion | Required evidence for completion | Current artifact evidence | Status |
| --- | --- | --- | --- |
| Proposal read and reframed | The original positive equivalence proposal must be translated into falsifiable empirical gates and a paper plan. | `docs/research_roadmap.md`, `docs/negative_result_paper_plan.md`, `docs/submission_readiness_audit.md`, `paper/main.tex` | Substantial; the negative support-equivalence claim and its scope are now explicit, but the full original proposal still has robustness gaps. |
| Reproducible codebase | Implement IMP, posterior samplers, mask comparisons, controls, summaries, and paper generation in runnable scripts. | `src/lottery/`, `scripts/`, `README.md`, `requirements.txt`, `requirements-ci.txt`, `requirements-gpu-lock.txt`, `requirements-lock.txt`, `Dockerfile`, `Dockerfile.gpu`, `.dockerignore`, `Makefile`, `LICENSE`, `.github/workflows/check.yml`, `docs/environment_snapshot.md`, `docs/environment_lock.json`, `docs/container_lock.md`, `docs/gpu_training_container.md`, `docs/compute_resource_accounting.md`, `docs/asset_license_inventory.md`, `docs/new_asset_inventory.md`, `docs/external_validation_receipts.json`, `docs/external_validation_receipt_template.md`, `docs/external_validation_runbook.md`, `docs/submission_handoff.md`, `docs/paper_claim_ledger.md`, `docs/reviewer_objection_matrix.md`, `docs/paper_submission_shape_audit.md`, `docs/submission_pdf_shape_audit.md`, `docs/venue_submission_compliance_audit.md`, `docs/local_gpu_container_validation.md`, `docs/public_release_manifest.md`, `docs/release_anonymization_audit.md`, `docs/public_release_archive_audit.md`, `docs/public_release_archive_smoke.md`, `docs/public_repository_snapshot_audit.md`, `docs/external_validation_readiness_audit.md`; latest `py_compile` passed, `scripts/check_environment_lock.py` checks the active runtime, `scripts/check_gpu_training_environment.py` checks the CUDA/package-lock path, `.github/workflows/check.yml` installs git/TeX/poppler/ripgrep and runs `make ci-check paper-check PYTHON=python`, `scripts/build_paper_claim_ledger.py` maps central claims to checked artifact numbers, `scripts/build_reviewer_objection_matrix.py` maps likely review risks to artifact-backed answers, `scripts/audit_paper_submission_shape.py` checks manuscript shape, `scripts/audit_submission_pdf_shape.py` checks the main-only submission PDF, `scripts/audit_venue_submission_compliance.py` checks the NeurIPS 2026 official style/checklist binding, checklist metadata docs, local GPU-container receipt, and release-packaging risks, `scripts/build_release_manifest.py` records the public package SHA256 inventory, `scripts/audit_release_anonymization.py` scans release text for local identity/path leakage, `scripts/build_public_release_archive.py` builds the local anonymous-review tarball, `scripts/smoke_public_release_archive.py` extracts and verifies the release package, `scripts/stage_public_repository_snapshot.py` stages a clean source-only anonymous git snapshot, `scripts/audit_external_validation_readiness.py` records the external upload/repository/CI/GPU receipt state, `scripts/build_external_validation_receipt_template.py` pre-fills immutable receipt fields from the current archive and source snapshot, `scripts/update_external_validation_receipts.py` validates externally observed receipt values before writing the registry, `scripts/build_external_validation_runbook.py` generates the public upload/repository/CI/GPU receipt runbook, `scripts/build_submission_handoff.py` generates the submission UI and supplement handoff, and `scripts/verify_research_artifacts.py` checks the core generated evidence, container locks, release metadata docs, project license, CI paper-build gate, local GPU-container validation, claim ledger, reviewer matrix, paper-shape audit, submission-PDF audit, venue-compliance audit, release manifest, release-anonymization audit, release-archive audit, release-archive smoke, public-repository snapshot, external-validation readiness audit, external-validation receipt template, external-validation runbook, and submission handoff. | Submission packaging is ready: public release, public source repository, and public CI receipts are observed; local CPU and GPU container checks pass. Strict independent external GPU-host validation is still pending. |
| Baseline LTH/IMP evidence | Standard IMP with rewinding across small datasets and CIFAR, repeated seeds. | MNIST/Fashion Gate1 sweeps; CIFAR ResNet-20 long epoch-1 rewind rows in `runs/` and `docs/`. | Substantial. |
| Bayesian posterior samplers | SGLD, SGHMC, cyclical SGLD, SWAG, Laplace, and HMC variants must test whether posterior support aligns with IMP better than controls. | `src/lottery/sgld.py`, `sghmc.py`, `cyclical_sgld.py`, `swag.py`, `diag_laplace.py`, `kfac_laplace.py`, `lowrank_laplace.py`, `full_laplace.py`, `head_laplace.py`, `block_laplace.py`, `subspace_hmc.py`; generated summaries in `docs/paper_stats.md`, `docs/digits_fullnet_laplace_tiny_r2_p0p3.md`, and `docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md`. | Substantial; tiny exact dense full-network Laplace and fake-CIFAR ResNet exact dense code-path smoke are validated, and CIFAR full-weight exact joint-group evidence exists, but dense all-parameter exact/full-covariance CIFAR posterior evidence remains missing. |
| Mode/ticket equivalence tests from proposal | KS/MMD/Wasserstein/mask-distance/activation-space comparisons between mode and ticket distributions. | `scripts/run_mode_distribution_equivalence_audit.py` performs KS/Wasserstein/MMD support-overlap comparisons over existing posterior artifacts; posterior beats random in 58/59 groups but never beats matched chain-start by more than 0.005 Jaccard. Rewind magnitude beats posterior in 55/57 grouped comparisons. `scripts/run_mode_ticket_distribution_probe.py` now adds direct probes with layer-sparsity KS, MMD, sliced Wasserstein, mask-Hamming overlap, logit CKA, optional final-hidden activation CKA, Hungarian cost, raw parameter-PCA basin entropy, chain-start rows, and posterior-to-chain-start Hamming diagnostics. The digits row fails the layer KS/Hamming thresholds and has one posterior mode representative versus five tickets, with entropy 0.0 and effective clusters 1.0. A five-seed CIFAR-10 ResNet subset row repeats this pattern. The five-seed full-data CIFAR-10 ResNet-20 row is sharper: layer KS p=5.3e-09, Hamming overlap 0.0033, logit CKA 0.9369, activation CKA 0.9172, one posterior basin, and entropy 0.0. The activation-channel-aligned full-data row still has one basin and fails layer KS p=2.3e-09 plus Hamming overlap 0.0000. The weight-correlation-aligned full-data row also has one basin and fails layer KS p=1.2e-08 plus Hamming overlap 0.1290. A 75-sample dense-start multi-chain cyclical-SGLD full-data row moves from chain starts (Hamming 0.0443) but still has one basin and fails layer KS p=3.3e-08 plus Hamming overlap 0.2461. A 75-sample independent-start cyclical-SGLD row also collapses 15 chain starts and 75 posterior samples to one basin, moves only 0.0439 Hamming from chain starts, and fails layer KS p=9.3e-10 plus Hamming overlap 0.0000. The rank-128 low-rank Laplace direct row is a partial rescue: 50 samples pass Hamming overlap 0.8163 and logit/activation CKA 0.9319/0.9096, but still collapse to one basin and fail layer KS p=2.0e-06. The 270,896-parameter streamed joint-group Laplace direct row compares 25 samples to five tickets, keeps sample accuracy 0.8835 and posterior-to-chain-start Hamming 0.0503, but still has one basin, layer KS p=1.1e-08, and Hamming overlap 0.0000 while CKA remains high. | Substantial and negative for full direct proposal equivalence; exact dense full-covariance posterior baselines remain the main gap. |
| Connectivity/barrier evidence | Check whether linear mode connectivity can explain away posterior/ticket mismatch. | `scripts/audit_linear_connectivity_barriers.py` aggregates six five-seed rows into `docs/linear_connectivity_barrier_audit.md`: MNIST/Fashion dense-to-IMP barriers are near zero (`0.0026`/`0.0395`), CIFAR long SGLD/SWAG barriers are large (`3.0827`/`3.7402`), and posterior support remains tied to chain-start controls in both cases. | Covered as negative support evidence; linear barriers are orthogonal landscape diagnostics, not support-equivalence evidence. |
| Permutation/alignment handling | Chain/model alignment or evidence that alignment does not rescue support conclusions. | Activation-channel Hungarian alignment exists for direct cross-seed residual transfer and for the full-data direct mode/ticket probe. Weight-correlation Hungarian alignment now also exists for the full-data mode/ticket probe. Source-vote/aligned controls fail, and both aligned mode/ticket rows still fail mask-distribution thresholds. `docs/mode_ticket_alignment_artifact_audit.md` verifies seven full-data direct rows reject equivalence, the aligned rows both fail layer-KS/Hamming, and old direct-run artifacts lacked raw posterior/ticket masks or states. `scripts/run_mode_ticket_distribution_probe.py` now supports `--save-mask-artifacts` and `--save-state-artifacts`; `docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md` verifies the `.npz` schema plus parameter shapes on fake-CIFAR, `docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md` verifies record-level post-hoc minimum-cost matching plus local channel-permutation matching over that saved fixture, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md` verifies record-level post-hoc matching over the full-data saved CIFAR masks/states, `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md` applies block-coordinate global channel matching without rescuing posterior/ticket supports, and `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md` validates exact small-subgraph enumeration while quantifying the full-data search space. | Substantial first-order alignment evidence plus full-data saved-artifact record/channel matching; exact small-subgraph path validated, but exhaustive full-data graph-isomorphism remains infeasible/open. |
| Variational pruning / learned mask baseline | Implement and compare mode-seeking/learned mask algorithms against IMP, including accuracy and calibration/OOD. | SNIP/SynFlow, Gem-Miner-style STE score-training, a proposal-style Bernoulli/Concrete variational pruning baseline, and a hard-concrete L0 gate baseline exist. The hard-concrete path is wired into support and calibration/OOD probes and smoke-tested on fake-CIFAR plus a real CIFAR subset. The five-seed digits variational row compares accuracy, ECE, Brier, and support: variational pruning beats random/Gem-Miner in accuracy but remains below IMP and does not improve ECE over IMP. A five-seed full-data CIFAR-10/CIFAR-100 learned-mask calibration/OOD row shows learned-random, Gem-Miner-style, and variational-prune masks lower ECE but remain below IMP on accuracy, NLL, Brier, and OOD AUROC. Matched five-seed CIFAR support rows give variational-prune accuracy/support 0.8306/0.0907 and hard-concrete accuracy/support 0.2766/0.0922, random-scale rather than ticket-like. | Strong negative evidence for current learned-mask baselines; broader permutation-aware variants remain optional. |
| Calibration/OOD evaluation | Compare posterior/mask methods on calibration and OOD, not only accuracy/support. | `run_calibration_ood_probe.py`; CIFAR-10/CIFAR-100 dense/IMP/SWAG results; optional learned-random, Gem-Miner-style, and variational-prune hard-mask sources now have a five-seed full-data CIFAR row. SWAG and learned masks can improve ECE, but they hurt accuracy, NLL, Brier, and OOD AUROC relative to IMP. | Solid negative evidence under the current CIFAR setting. |
| Strong CIFAR posterior baseline | At least one credible full-network high-fidelity CIFAR posterior comparison beyond selected blocks and low-dimensional subspaces. | Exact final-head Laplace, selected-block, joint-block, 22,064-parameter tensor-block-diagonal full-covariance Laplace, 68,144-parameter tensor-block-diagonal full-covariance Laplace, 68,144-parameter joint-group full-covariance Laplace, 86,576-parameter joint-group full-covariance Laplace, and a streamed 270,896-parameter joint-group full-covariance Laplace row over all weight tensors, random/trajectory/Hessian subspace HMC, a full-network 20-snapshot SWAG movement probe, rank-16/rank-32/rank-64/rank-128 full-network low-rank Hessian-plus-diagonal Laplace movement probes, a tiny exact dense full-network Laplace sanity row over all 310 trainable digits-MLP parameters, a fake-CIFAR ResNet exact dense smoke over all 1,229 width-1 trainable parameters, dense-start and independent-start 75-sample full-data multi-chain cyclical-SGLD direct probes, a rank-128 low-rank Laplace direct probe, a streamed 270,896-parameter joint-group Laplace direct probe, and `docs/cifar10_resnet20_full_covariance_feasibility.md` quantifying the dense exact covariance cost exist. The tiny exact dense row keeps sample accuracy `0.8450`, moves from chain-start support (`post-chain=0.8084`), but has posterior support `0.7545` versus chain-start `0.8596`; the fake-CIFAR ResNet smoke validates a dense `1229 x 1229` Cholesky over convolutional/residual/BatchNorm parameters but is not real CIFAR evidence; the 68k block-diagonal row keeps sample accuracy `0.8802`, has global post-chain `0.7400`, but block posterior-chain is `-0.0050` and global posterior-chain is only `+0.0010` while rewind remains closer by `0.0319`; the 68k joint-group row keeps sample accuracy `0.8811`, has global post-chain `0.7148`, but block posterior-chain is `-0.0050` and global posterior-chain is only `+0.0015` while rewind remains closer by `0.0311`; the 86k joint-group row keeps sample accuracy `0.8828`, has global post-chain `0.7863`, but block posterior-chain is `-0.0023` and global posterior-chain is only `+0.0006` while rewind remains closer by `0.0317`; the 270k full-weight movement row keeps sample accuracy `0.8824`, has global post-chain `0.7389`, but block/global posterior-chain is `-0.0019` while rewind remains closer by `0.0362`; the matching 270k direct row keeps sample accuracy `0.8835` but fails layer KS p=1.1e-08 and Hamming overlap 0.0000. | Partially addressed; exact all-parameter dense full-covariance CIFAR posterior evidence remains missing, but literal dense exact covariance is now documented as outside the single-workstation budget. |
| Mechanistic alternative after negative result | If posterior-mode equivalence fails, identify what does explain IMP beyond posterior support. | Dense-trajectory, residual-swap, residual anatomy, predictor, cross-seed transfer, base-compatibility, posterior decomposition, stratified controls, removal-order, IMP-process, ranking, oracle-overlap-matched process, score-source process, round-exclusion process-intervention, tensor-matched round-exclusion, tensor+score-matched round-exclusion, residualized round-score projection, posterior-residualized round-score projection, and learned-subspace residualized projection controls. | Strong but still incomplete; next gap is high-fidelity posterior evidence plus optional broader process robustness. |
| Paper draft | A coherent, compilable manuscript with tables/figures and honest limitations. | `paper/main.tex`, `paper/refs.bib`, `paper/neurips_2026.sty`, `paper/neurips_checklist.tex`, `paper/tables/statistical_summary.tex`, appendix-inclusive `paper/main.pdf`, main-only `paper/main_submission.pdf`, official-style `paper/neurips_submission.pdf`, `docs/paper_submission_shape_audit.md`, `docs/submission_pdf_shape_audit.md`, `docs/venue_submission_compliance_audit.md`, `docs/submission_handoff.md`, `docs/release_anonymization_audit.md`, `docs/public_release_archive_audit.md`; latest build passed. The abstract is under the 250-word budget, NeurIPS main content is 8 pages before References against the 9-page budget, the paper distinguishes the support-equivalence claim from general Bayesian pruning and scopes the limitation to tested posterior families, and the verifier checks cited-key coverage plus corrected key citation metadata. The submission handoff records the exact upload files, abstract/title metadata, current release archive SHA256, source snapshot commit, local check commands, and remaining strict external GPU hardening blocker. | Substantial condensed NeurIPS-bound draft; content/PDF/style/anonymized-release/archive/source-snapshot/handoff/public release/public source/public CI gates are ready, while independent external GPU-host validation remains open. |
| Generated paper statistics | Tables and reported CIs must be generated from run artifacts, not hand-edited. | `scripts/build_paper_stats.py`, `scripts/build_paper_claim_ledger.py`, `runs/paper_stats.json`, `docs/paper_stats.md`, `docs/paper_claim_ledger.md`, `paper/tables/statistical_summary.tex`; `scripts/verify_research_artifacts.py` asserts key sections, rows, claim-ledger phrases, and numerical claims are present. | Good. |
| Verification | Build/test commands must cover the changed artifacts and paper. | Latest checks: `py_compile`, `make paper-check`, `make paper-submission-check`, `make check`, clean-state `scripts/verify_research_artifacts.py`, rebuilt `make container-build && make container-check`, `make gpu-container-build && make gpu-container-env-check`, and `scripts/build_local_gpu_container_validation.py`; the configured GitHub Actions workflow runs `make ci-check paper-check PYTHON=python`, the public GitHub Actions run is observed green, local PDFs remain built, and the release manifest/anonymization/archive/extracted-package smoke, public-repository snapshot, external-validation readiness, external-validation runbook, and submission-handoff audits are regenerated by `make check`. | Good local/container/public-CI coverage; only the independent external GPU-host receipt is not observed. |

## Prompt-To-Artifact Checklist

| Explicit item | Artifact or evidence inspected | Coverage judgment |
| --- | --- | --- |
| `proposal_A3_lottery_ticket_bayesian_modes.md` | Read the proposal. It requires LTH/posterior-mode equivalence tests, SGLD/cyclical SGLD/HMC, mode identification, IMP distributions, variational pruning, calibration/OOD, and top venue readiness. | Current project covers many negative-result variants but not every original positive-proposal test. |
| "top conf" | `paper/main.tex`, `paper/main.pdf`, `paper/main_submission.pdf`, `paper/neurips_submission.pdf`, `paper/neurips_2026.sty`, `paper/neurips_checklist.tex`, `docs/submission_readiness_audit.md`, `docs/negative_result_paper_plan.md`, `docs/paper_submission_shape_audit.md`, `docs/submission_pdf_shape_audit.md`, `docs/venue_submission_compliance_audit.md`, `docs/submission_handoff.md`, `docs/local_gpu_container_validation.md`, `docs/release_anonymization_audit.md`, `docs/public_release_archive_audit.md`, `docs/public_repository_snapshot_audit.md`, `docs/external_validation_readiness_audit.md`, `docs/external_validation_receipt_template.md`, `docs/external_validation_runbook.md`. | Draft exists and local shape/main-only/NeurIPS PDF/content-packet/style/anonymization/archive/repository-snapshot/handoff/template/runbook audits are ready; public archive, public source repository, and external CI receipts are observed. Remaining gaps are independent external GPU-host hardening and bounded scientific robustness limitations. |
| "논문" | `paper/main.tex`, `paper/refs.bib`, `paper/tables/statistical_summary.tex`, `paper/figures/`. | Compilable working draft, not final paper. |
| "연구 성과" | `runs/*_summary.csv`, `docs/*selected*.md`, `docs/paper_stats.md`, `docs/experiment_log.md`. | Substantial empirical record with repeated-seed negative evidence. |
| "Bayesian posterior modes" | SGLD/SGHMC/cyclical SGLD/SWAG/Laplace/HMC code, summaries, and the mode/ticket distribution-equivalence support audit. | Strong negative posterior-support evidence with KS/Wasserstein/MMD support comparisons; unaligned, activation-aligned, weight-correlation-aligned, multi-chain cyclical-SGLD, full-network SWAG movement, rank-16/rank-32/rank-64/rank-128 full-network low-rank Hessian-plus-diagonal Laplace, a tiny exact dense full-network Laplace sanity row, 22k/68k exact tensor-block-diagonal Laplace, and 68k/86k/270k exact joint-group Laplace movement/direct CIFAR rows now exist, but exact all-parameter dense full-covariance CIFAR posterior coverage remains incomplete. |
| "Lottery ticket" | `src/lottery/imp.py`, Gate1 and CIFAR IMP summaries. | Covered for IMP-style tickets; broader ticket algorithms are only partially covered. |
| "mode identification" from proposal | `scripts/run_mode_ticket_distribution_probe.py` uses PCA plus mean-shift representatives and raw/aligned parameter-PCA basin entropy for the direct distribution probes; other posterior scripts retain state/logit clustering. First-order activation-channel and weight-correlation alignment now exist for full-data CIFAR, but exhaustive posterior-chain permutation alignment is still absent. | Partial. |
| "variational pruning" from proposal | `variational_pruning_mask` and `hard_concrete_mask` in `src/lottery/pruning_baselines.py`; five-seed digits accuracy/ECE/Brier run in `runs/digits_mlp_variational_prune_calib_r5_summary.csv` and `docs/digits_mlp_variational_prune_calib_r5.md`; learned-mask OOD smokes in `docs/fake_cifar10_calibration_ood_learned_masks_smoke.md` and `docs/cifar10_subset_calibration_ood_learned_masks_smoke.md`; seed-0 larger-subset pilot in `docs/cifar10_subset4096_calibration_ood_learned_masks_pilot.md`; five-seed full-data CIFAR learned-mask OOD row in `docs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3.md`; five-seed CIFAR support rows in `docs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3.md` and `docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md`. | Explicit implementation plus CIFAR calibration/OOD and support evidence now exist; broader mask-distribution/permutation claims remain incomplete. |
| Build command | Latest LaTeX build completed after NeurIPS 2026 style/checklist binding; `paper/main.pdf`, `paper/main_submission.pdf`, and `paper/neurips_submission.pdf` exist, and `make paper-check` covers all three. | Covered for current draft. |
| Test command | `make check`, clean-state `scripts/verify_research_artifacts.py`, rebuilt `make container-build && make container-check`, `make gpu-container-build && make gpu-container-env-check`, and `scripts/build_local_gpu_container_validation.py` passed after the latest changes; the configured CI command is `make ci-check paper-check PYTHON=python`, the public CI run is observed green, and `make check` includes the release-anonymization, release-archive, extracted-package smoke, public-repository snapshot, external-validation readiness, external-validation runbook, and submission-handoff audits. | Good local/container/public-CI coverage; independent external GPU-host validation is still unobserved. |
| PR state | Workspace has no `.git` repository. | Not applicable; no PR/commit state can be inspected. |

## Current Strongest Evidence

- MNIST and Fashion-MNIST Gate1 sweeps fail the posterior-mode support rescue
  criterion across sparsities.
- CIFAR-10 ResNet-20 long-budget epoch-1 rewind rows show IMP accuracy exceeds
  dense accuracy, but posterior masks do not beat chain-start/rewind controls.
- SGLD, SGHMC, cyclical SGLD, canonical SWAG, 20-snapshot full-network SWAG,
  diagonal Laplace, KFAC-style Laplace, rank-16/rank-32/rank-64/rank-128 low-rank Hessian-plus-diagonal
  Laplace, exact final-head Laplace,
  selected-block full-covariance Laplace, joint-block full-covariance Laplace,
  22,064- and 68,144-parameter tensor-block-diagonal full-covariance Laplace,
  68,144-, 86,576-, and 270,896-parameter joint-group full-covariance Laplace rows,
  a tiny exact dense full-network Laplace sanity row, and low-dimensional
  subspace HMC all support the negative conclusion under
  their respective approximation limits. A five-seed
  32-dimensional top-Hessian HMC selected row also stays at chain-start support.
- The full-network SWAG movement probe scales posterior spread over all
  trainable parameters, but moving from chain-start overlap 0.9778 to 0.9086
  does not improve posterior-to-IMP beyond chain-start magnitude.
- The full-network low-rank Hessian-plus-diagonal Laplace movement probes
  add correlated rank-16, rank-32, rank-64, and rank-128 curvature Gaussians over all
  trainable parameters; moving to chain-start overlap 0.7359, 0.7402, and
  0.7397 lowers posterior-to-IMP from the chain-start values 0.1456, 0.1457,
  and 0.1433 to 0.1351, 0.1358, and 0.1339.
- The direct multi-chain cyclical-SGLD full-data probe collects 75 posterior
  samples that move from chain starts while preserving accuracy, but the
  proposal-level distribution metrics still fail.
- The latest block-diagonal Laplace row estimates exact covariance blocks for
  11 tensors and 22,064 selected parameters; samples move from chain starts
  while keeping `0.8810` accuracy, but block posterior-minus-chain is negative
  (`-0.0114`) and the global gain is only `+0.0036` with rewind still closer.
- The wider exact block-diagonal Laplace row estimates exact covariance blocks
  for 16 tensors and 68,144 selected parameters; samples move farther from
  chain starts (`global post-chain=0.7400`) while keeping `0.8802` accuracy,
  but block posterior-minus-chain remains negative (`-0.0050`) and the global
  gain is only `+0.0010` with rewind still closer by `+0.0319`.
- The joint-group Laplace row packs the same 16 tensors and 68,144 parameters
  into 8 exact covariance groups, adding within-group cross-tensor covariance;
  samples move farther from chain starts (`global post-chain=0.7148`) while
  keeping `0.8811` accuracy, but block posterior-minus-chain remains negative
  (`-0.0050`) and the global gain is only `+0.0015` with rewind still closer
  by `+0.0311`.
- The max-20k joint-group Laplace row adds the first stage-3 convolution block,
  covering 17 tensors and 86,576 parameters in 6 exact covariance groups;
  samples keep `0.8828` accuracy and remain moved from chain starts (`global
  post-chain=0.7863`), but block posterior-minus-chain remains negative
  (`-0.0023`) and the global gain is only `+0.0006` with rewind still closer
  by `+0.0317`.
- The streamed max-40k joint-group Laplace row covers all 22 ResNet-20 weight
  tensors and 270,896 weight parameters in 8 exact covariance groups; samples
  keep `0.8824` accuracy and remain moved from chain starts (`global
  post-chain=0.7389`), but block/global posterior-minus-chain is `-0.0019`
  with rewind still closer by `+0.0362`.
- The tiny exact dense full-network Laplace row samples all `310` trainable
  digits MLP parameters in one covariance; at scale `1e-3`, samples keep
  `0.8450` accuracy and move from chain-start support (`post-chain=0.8084`),
  but posterior-to-IMP support is `0.7545` versus chain-start support
  `0.8596`.
- The fake-CIFAR ResNet exact dense full-network smoke samples all `1,229`
  trainable width-1 ResNet-20 parameters in one covariance and verifies a dense
  `1229 x 1229` Cholesky over convolutional/residual/BatchNorm parameters. It
  is code-path validation, not real CIFAR evidence.
- The linear connectivity barrier audit rules out a simple LMC rescue:
  MNIST/Fashion dense-to-IMP barriers are near zero while CIFAR long-run
  barriers are large, but posterior support remains tied to chain-start
  controls in both regimes.
- The matching streamed max-40k joint-group direct mode/ticket row compares 25
  posterior samples against five IMP tickets; samples keep `0.8835` accuracy
  and move `0.0503` Hamming from chain starts, but all samples collapse to one
  basin and fail layer KS (`p=1.1e-08`) plus Hamming overlap (`0.0000`).
- Trajectory and IMP-process controls provide an alternative mechanism:
  IMP-only residual support is functional and gradually constructed by the IMP
  process, not explained by posterior RMS, dense magnitude alone, simple
  coordinate transfer, layer/tensor strata, or removal order.

## Missing Or Weak Requirements

1. Full-network exact or near-exact full-covariance CIFAR posterior evidence.
   The multi-chain cyclical-SGLD, full-network SWAG, and full-network
   low-rank Hessian-plus-diagonal Laplace objections are reduced through rank
   128, a 22,064-parameter exact tensor-block-diagonal covariance row, a
   68,144-parameter max-10k exact tensor-block-diagonal row, and
   68,144-, 86,576-, and 270,896-parameter joint-group rows now cover wider
   subsets, within-group cross-tensor covariance, the entire weight vector, and
   a direct proposal-metric full-weight grouped row. The exact dense
   all-parameter software path is now validated on a 310-parameter digits MLP
   and a 1,229-parameter fake-CIFAR ResNet smoke, but exact dense
   all-parameter CIFAR full-network posterior coverage is still absent.
   The feasibility audit now bounds the literal dense option at `553.1` GiB
   for one all-trainable float64 covariance/precision matrix and `1,106.3` GiB
   with Cholesky resident, so the remaining practical route is still stronger
   structured covariance/subspace evidence rather than exact dense CIFAR
   covariance.
2. Broader learned-mask distribution/permutation variants. The explicit
   Bernoulli/Concrete algorithm now has five-seed digits evidence, a five-seed
   full-data CIFAR calibration/OOD row, and a five-seed CIFAR support row; a
   hard-concrete L0 variant also has a five-seed CIFAR support row. The full
   original proposal's mask-distribution/permutation claims are still not
   exhaustively tested. The current artifact audit makes the limitation
   explicit: old direct-run artifacts lacked raw masks/states, but the new
   activation-aligned full-data saved-artifact rerun now supports record-level
   matching over CIFAR masks/states. The direct probe has the needed save
   flags, a fake-CIFAR schema fixture with parameter shapes, and a record-level
   plus local channel post-hoc matching audit over that fixture. The new
   structured global channel audit over saved CIFAR masks keeps
   posterior/ticket Hamming near `0.21`. The exact stage-1 enumeration audit
   validates all `128` assignments on a tiny subgraph and sizes the full CIFAR
   channel search at about `10^840.4` assignments per pair; the remaining
   permutation row is therefore an infeasible exhaustive full-data
   graph-isomorphism objective over saved artifacts.
3. Broader constrained process/subspace causal controls for the IMP residual.
   Oracle-overlap-matched, score-source, round-exclusion, tensor-matched
   round-exclusion, tensor+score-matched round-exclusion, residualized
   round-score projection, posterior-residualized projection, and
   learned-subspace residualized projection process-control rows are now
   implemented, smoke-tested, and run at 5 seeds. The learned-subspace row
   projects out rank-8 dense trajectory/final-IMP/earlier-round PCA components;
   round-selected masks still beat it by `+0.0048` accuracy with `5/5`
   positive seeds, and oracle overlap drops from `0.6807` to `0.4917`.
   Additional constrained process interventions remain possible hardening work.
4. Reproducibility hardening: `requirements.txt`, `requirements-ci.txt`,
   `requirements-gpu-lock.txt`, `requirements-lock.txt`, `Dockerfile`, `Dockerfile.gpu`, `.dockerignore`,
   `Makefile`, `LICENSE`, `.github/workflows/check.yml`, `docs/environment_snapshot.md`,
   `docs/environment_lock.json`, `docs/container_lock.md`,
   `docs/gpu_training_container.md`, `docs/public_release_manifest.md`,
   `docs/release_anonymization_audit.md`,
   `docs/public_release_archive_audit.md`,
   `docs/public_release_archive_smoke.md`,
   `docs/external_validation_receipts.json`,
   `docs/external_validation_receipt_template.md`,
   `docs/external_validation_runbook.md`, `docs/submission_handoff.md`,
   `docs/public_repository_snapshot_audit.md`,
   `docs/external_validation_readiness_audit.md`,
   `scripts/check_environment_lock.py`,
   `scripts/check_gpu_training_environment.py`,
   `scripts/build_release_manifest.py`,
   `scripts/audit_release_anonymization.py`,
   `scripts/build_public_release_archive.py`,
   `scripts/smoke_public_release_archive.py`,
   `scripts/stage_public_repository_snapshot.py`,
   `scripts/audit_external_validation_readiness.py`,
   `scripts/build_external_validation_receipt_template.py`,
   `scripts/update_external_validation_receipts.py`,
   `scripts/build_external_validation_runbook.py`,
   `scripts/build_submission_handoff.py`, and
   `scripts/verify_research_artifacts.py` now cover the core
   command/artifact/environment/container/GPU-container definition/release
   manifest/anonymization/archive/extracted-package/source-repository/external-receipt template/checks, and the configured
   CI workflow now runs both artifact verification and paper PDF compilation.
   The runbook and handoff bind the local archive/source facts to the exact
   public receipts needed before claiming release readiness. An externally
   public release/source/CI receipts are now observed, and local GPU-container
   validation is recorded. The independent external GPU-container validation
   receipt is still missing.
5. Final paper polish: the generated evidence tables are appendix-scoped, the
   main CIFAR movement table is compact, and the thesis, limitation wording,
   related-work positioning, and key citation metadata now state a scoped
   support-equivalence claim. Remaining paper work is venue-level editing
   rather than basic claim framing.

## Completion Decision

Do not mark the thread goal complete. The venue-submission packet is now ready
under the local audit, and the public archive, public source repository, and
external CI receipts are observed. The next external milestone is narrower:
run the GPU container on an independent CUDA host, record the passing URL/image
digest in `docs/external_validation_receipts.json`, then rerun the strict
external gate. If that is blocked, the remaining productive research work is a
qualitatively different posterior/permutation robustness check beyond the
current full-weight joint-group direct row, followed by paper and audit
integration.
