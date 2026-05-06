PYTHON ?= .venv/bin/python
CONTAINER_IMAGE ?= lottery-artifact:2026-05-06
GPU_CONTAINER_IMAGE ?= lottery-training-gpu:2026-05-06

.PHONY: check ci-check source-repository-check stats claim-ledger mode-ticket-alignment-audit mask-artifact-posthoc-audit full-mask-artifact-posthoc-audit full-channel-permutation-audit exhaustive-channel-feasibility-audit digits-fullnet-laplace-summary fake-resnet-fullnet-laplace-summary linear-connectivity-audit reviewer-objection-matrix paper-shape-audit submission-pdf-audit venue-submission-audit mode-ticket-artifact-budget release-anonymization-audit release-archive release-archive-smoke public-repository-snapshot external-validation-receipt-template external-validation-readiness external-validation-strict verify env-check gpu-env-check local-gpu-container-validation release-manifest figures paper paper-submission paper-neurips paper-check paper-submission-check paper-neurips-check paper-existing paper-existing-submission paper-existing-neurips paper-existing-check paper-existing-submission-check paper-existing-neurips-check container-build container-check gpu-container-build gpu-container-env-check clean

check:
	$(PYTHON) -m py_compile src/lottery/*.py scripts/*.py
	$(PYTHON) scripts/check_environment_lock.py
	$(PYTHON) scripts/audit_full_covariance_feasibility.py
	$(PYTHON) scripts/build_paper_stats.py
	$(PYTHON) scripts/audit_mode_ticket_alignment_artifacts.py
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --out-json runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md --max-channel-pair-count 1
	$(PYTHON) scripts/audit_full_data_channel_permutation_matching.py
	$(PYTHON) scripts/audit_exhaustive_channel_permutation_feasibility.py
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/digits_fullnet_laplace_tiny_r2_p0p3 --out-csv runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv --out-md docs/digits_fullnet_laplace_tiny_r2_p0p3.md
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke --out-csv runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv --out-md docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md
	$(PYTHON) scripts/audit_linear_connectivity_barriers.py
	$(PYTHON) scripts/build_reviewer_objection_matrix.py
	$(PYTHON) scripts/audit_paper_submission_shape.py
	$(PYTHON) scripts/audit_submission_pdf_shape.py
	$(PYTHON) scripts/audit_mode_ticket_artifact_storage_budget.py
	$(PYTHON) scripts/build_paper_claim_ledger.py
	$(PYTHON) scripts/audit_external_validation_readiness.py
	$(PYTHON) scripts/audit_venue_submission_compliance.py
	$(PYTHON) scripts/build_submission_handoff.py
	$(PYTHON) scripts/build_release_manifest.py
	$(PYTHON) scripts/audit_release_anonymization.py
	$(PYTHON) scripts/build_public_release_archive.py
	$(PYTHON) scripts/smoke_public_release_archive.py
	$(PYTHON) scripts/stage_public_repository_snapshot.py
	$(PYTHON) scripts/smoke_public_repository_snapshot.py
	$(PYTHON) scripts/audit_external_validation_readiness.py
	$(PYTHON) scripts/build_external_validation_receipt_template.py
	$(PYTHON) scripts/build_external_validation_runbook.py
	$(PYTHON) scripts/audit_venue_submission_compliance.py
	$(PYTHON) scripts/build_submission_handoff.py
	$(PYTHON) scripts/verify_research_artifacts.py

ci-check:
	$(PYTHON) -m py_compile src/lottery/*.py scripts/*.py
	$(PYTHON) scripts/build_paper_stats.py
	$(PYTHON) scripts/audit_mode_ticket_alignment_artifacts.py
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --out-json runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md --max-channel-pair-count 1
	$(PYTHON) scripts/audit_full_data_channel_permutation_matching.py
	$(PYTHON) scripts/audit_exhaustive_channel_permutation_feasibility.py
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/digits_fullnet_laplace_tiny_r2_p0p3 --out-csv runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv --out-md docs/digits_fullnet_laplace_tiny_r2_p0p3.md
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke --out-csv runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv --out-md docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md
	$(PYTHON) scripts/audit_linear_connectivity_barriers.py
	$(PYTHON) scripts/build_reviewer_objection_matrix.py
	$(PYTHON) scripts/audit_paper_submission_shape.py
	$(PYTHON) scripts/audit_submission_pdf_shape.py
	$(PYTHON) scripts/audit_mode_ticket_artifact_storage_budget.py
	$(PYTHON) scripts/build_paper_claim_ledger.py
	$(PYTHON) scripts/audit_external_validation_readiness.py
	$(PYTHON) scripts/audit_venue_submission_compliance.py
	$(PYTHON) scripts/build_submission_handoff.py
	$(PYTHON) scripts/build_release_manifest.py
	$(PYTHON) scripts/audit_release_anonymization.py
	$(PYTHON) scripts/build_public_release_archive.py
	$(PYTHON) scripts/smoke_public_release_archive.py
	$(PYTHON) scripts/stage_public_repository_snapshot.py
	$(PYTHON) scripts/smoke_public_repository_snapshot.py
	$(PYTHON) scripts/verify_research_artifacts.py --release-package-mode

source-repository-check:
	$(PYTHON) -m py_compile src/lottery/*.py scripts/*.py
	$(PYTHON) scripts/verify_source_repository_snapshot.py
	$(MAKE) paper-existing-check PYTHON=$(PYTHON)

stats:
	$(PYTHON) scripts/build_paper_stats.py

claim-ledger: stats mode-ticket-alignment-audit mask-artifact-posthoc-audit full-mask-artifact-posthoc-audit full-channel-permutation-audit exhaustive-channel-feasibility-audit digits-fullnet-laplace-summary fake-resnet-fullnet-laplace-summary linear-connectivity-audit mode-ticket-artifact-budget
	$(PYTHON) scripts/build_paper_claim_ledger.py

mode-ticket-alignment-audit: stats
	$(PYTHON) scripts/audit_mode_ticket_alignment_artifacts.py

mask-artifact-posthoc-audit:
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py

full-mask-artifact-posthoc-audit:
	$(PYTHON) scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --out-json runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md --max-channel-pair-count 1

full-channel-permutation-audit:
	$(PYTHON) scripts/audit_full_data_channel_permutation_matching.py

exhaustive-channel-feasibility-audit:
	$(PYTHON) scripts/audit_exhaustive_channel_permutation_feasibility.py

digits-fullnet-laplace-summary:
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/digits_fullnet_laplace_tiny_r2_p0p3 --out-csv runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv --out-md docs/digits_fullnet_laplace_tiny_r2_p0p3.md

fake-resnet-fullnet-laplace-summary:
	$(PYTHON) scripts/summarize_fullnet_laplace_probe.py --run-root runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke --out-csv runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv --out-md docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md

linear-connectivity-audit:
	$(PYTHON) scripts/audit_linear_connectivity_barriers.py

reviewer-objection-matrix: stats full-channel-permutation-audit exhaustive-channel-feasibility-audit linear-connectivity-audit
	$(PYTHON) scripts/audit_full_covariance_feasibility.py
	$(PYTHON) scripts/build_reviewer_objection_matrix.py

paper-shape-audit: reviewer-objection-matrix
	$(PYTHON) scripts/audit_paper_submission_shape.py

submission-pdf-audit:
	$(PYTHON) scripts/audit_submission_pdf_shape.py

venue-submission-audit:
	$(PYTHON) scripts/audit_venue_submission_compliance.py

mode-ticket-artifact-budget:
	$(PYTHON) scripts/audit_mode_ticket_artifact_storage_budget.py

env-check:
	$(PYTHON) scripts/check_environment_lock.py

gpu-env-check:
	$(PYTHON) scripts/check_gpu_training_environment.py

local-gpu-container-validation:
	$(PYTHON) scripts/build_local_gpu_container_validation.py

release-manifest:
	$(PYTHON) scripts/build_release_manifest.py

release-anonymization-audit: release-manifest
	$(PYTHON) scripts/audit_release_anonymization.py

release-archive: release-anonymization-audit
	$(PYTHON) scripts/build_public_release_archive.py

release-archive-smoke: release-archive
	$(PYTHON) scripts/smoke_public_release_archive.py

public-repository-snapshot: release-archive-smoke
	$(PYTHON) scripts/stage_public_repository_snapshot.py
	$(PYTHON) scripts/smoke_public_repository_snapshot.py

external-validation-receipt-template: public-repository-snapshot
	$(PYTHON) scripts/audit_external_validation_readiness.py
	$(PYTHON) scripts/build_external_validation_receipt_template.py

external-validation-readiness: external-validation-receipt-template
	$(PYTHON) scripts/build_external_validation_runbook.py
	$(PYTHON) scripts/build_submission_handoff.py

external-validation-strict: public-repository-snapshot
	$(PYTHON) scripts/audit_external_validation_readiness.py --strict

verify: stats claim-ledger mode-ticket-alignment-audit mask-artifact-posthoc-audit reviewer-objection-matrix paper-shape-audit submission-pdf-audit venue-submission-audit mode-ticket-artifact-budget external-validation-readiness
	$(PYTHON) scripts/verify_research_artifacts.py

figures:
	$(PYTHON) scripts/build_paper_figures.py

paper: stats
	cd paper && pdflatex -interaction=nonstopmode main
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main
	cd paper && pdflatex -interaction=nonstopmode main

paper-submission: stats
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
	cd paper && bibtex main_submission
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'

paper-neurips: stats
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
	cd paper && bibtex neurips_submission
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'

paper-check: paper paper-submission-check paper-neurips-check
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Underfull|Rerun|Missing \\$$|Extra \\}" main.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

paper-submission-check: paper-submission
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Underfull|Rerun|Missing \\$$|Extra \\}" main_submission.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

paper-neurips-check: paper-neurips
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Rerun|Missing \\$$|Extra \\}" neurips_submission.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

paper-existing:
	cd paper && pdflatex -interaction=nonstopmode main
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main
	cd paper && pdflatex -interaction=nonstopmode main

paper-existing-submission:
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
	cd paper && bibtex main_submission
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'
	cd paper && pdflatex -interaction=nonstopmode -jobname=main_submission '\def\LOTTERYMAINONLY{1}\input{main.tex}'

paper-existing-neurips:
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
	cd paper && bibtex neurips_submission
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'
	cd paper && pdflatex -interaction=nonstopmode -jobname=neurips_submission '\def\LOTTERYNEURIPS{1}\input{main.tex}'

paper-existing-check: paper-existing paper-existing-submission-check paper-existing-neurips-check
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Underfull|Rerun|Missing \\$$|Extra \\}" main.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

paper-existing-submission-check: paper-existing-submission
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Underfull|Rerun|Missing \\$$|Extra \\}" main_submission.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

paper-existing-neurips-check: paper-existing-neurips
	@cd paper && if rg -n "Warning|Undefined|undefined|Overfull|Rerun|Missing \\$$|Extra \\}" neurips_submission.log | rg -v "Package: rerunfilecheck"; then exit 1; fi

container-build:
	docker build -t $(CONTAINER_IMAGE) .

container-check:
	docker run --rm $(CONTAINER_IMAGE)

gpu-container-build:
	docker build -f Dockerfile.gpu -t $(GPU_CONTAINER_IMAGE) .

gpu-container-env-check:
	$(PYTHON) scripts/run_gpu_container_env_check.py --image $(GPU_CONTAINER_IMAGE)

clean:
	rm -rf src/lottery/__pycache__ scripts/__pycache__
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out
	rm -f paper/main_submission.aux paper/main_submission.bbl paper/main_submission.blg paper/main_submission.log paper/main_submission.out
	rm -f paper/neurips_submission.aux paper/neurips_submission.bbl paper/neurips_submission.blg paper/neurips_submission.log paper/neurips_submission.out
