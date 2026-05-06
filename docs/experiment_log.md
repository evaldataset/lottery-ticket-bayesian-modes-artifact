# Experiment Log

## 2026-05-03: Digits Pilot

Command:

```bash
python scripts/run_digits_pilot.py --epochs 12 --imp-rounds 4 --sgld-steps 500 --sgld-burn-in 100 --sgld-sample-every 10 --samples 40 --random-trials 200
```

Output:

- `runs/digits_pilot/20260503_101130/metrics.json`
- `runs/digits_pilot/20260503_101130/mask_overlaps.csv`

Key numbers:

| Metric | Value |
| --- | ---: |
| Dense accuracy | 0.9778 |
| IMP accuracy | 0.9694 |
| IMP sparsity | 0.6836 |
| Posterior mask to IMP Jaccard | 0.3491 |
| Random mask to IMP Jaccard | 0.1877 |
| Posterior minus random Jaccard, 95% CI | [0.1468, 0.1760] |
| Mann-Whitney p-value | 9.46e-24 |
| Dense magnitude mask to IMP Jaccard | 0.7227 |
| Initial magnitude mask to IMP Jaccard | 0.4915 |
| SGLD clusters | 1 |

Interpretation:

The smoke pipeline works, and posterior-induced masks beat same-sparsity random
masks. This is not yet evidence for the main thesis because the dense magnitude
control is much stronger than the SGLD posterior-mask signal. The next
experiment must separate posterior geometry from ordinary magnitude structure.

Next actions:

1. Add repeated seeds and aggregate confidence intervals.
2. Add a control that samples SGLD from multiple independently trained dense
   seeds, not only one dense checkpoint.
3. Add loss-barrier and prediction-disagreement checks.
4. Move to MNIST only after the controls are stable.

## 2026-05-03: Digits Five-Seed Aggregate

Commands:

```bash
for seed in 1 2 3 4; do python scripts/run_digits_pilot.py --seed "$seed" --epochs 12 --imp-rounds 4 --sgld-steps 500 --sgld-burn-in 100 --sgld-sample-every 10 --samples 40 --random-trials 200 > "/tmp/digits_pilot_seed_${seed}.log"; done
python scripts/summarize_digits_runs.py
```

Output:

- `runs/digits_pilot_summary.csv`
- `runs/digits_pilot_summary.json`

Aggregate over compatible seeds 0-4:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Dense accuracy | 0.9761 | 0.0054 |
| IMP accuracy | 0.9728 | 0.0060 |
| IMP sparsity | 0.6836 | 0.0000 |
| Posterior mask to IMP Jaccard | 0.3505 | 0.0038 |
| Random mask to IMP Jaccard | 0.1880 | 0.0002 |
| Posterior minus random Jaccard | 0.1625 | 0.0038 |
| Dense magnitude mask to IMP Jaccard | 0.7189 | 0.0100 |
| Initial magnitude mask to IMP Jaccard | 0.4953 | 0.0055 |
| SGLD parameter-space clusters | 1.0000 | 0.0000 |
| SGLD function-space clusters | 2.0000 | 0.0000 |
| SGLD sample accuracy | 0.9344 | 0.0072 |
| SGLD sample to dense prediction agreement | 0.9456 | 0.0083 |
| SGLD sample to IMP prediction agreement | 0.9441 | 0.0062 |
| Dense-IMP linear loss barrier | 2.81e-05 | 5.97e-05 |
| Dense-sample linear loss barrier | 0.0000 | 0.0000 |
| IMP-sample linear loss barrier | 0.0000 | 0.0000 |

Interpretation:

The random-control gap is stable across five seeds, but the dense-magnitude
control remains much stronger. The added function-space check shows two
logit-space clusters despite one parameter-space cluster, so parameter-space
mode counting is already fragile even in the smoke setting. Linear barriers are
effectively zero, supporting the competing explanation that these small-model
objects live in one connected low-loss region. The current result is therefore a
useful pipeline validation, not a claim-level finding. The next decisive
experiment is a multi-chain setup initialized from independently trained dense
checkpoints and then MNIST-scale replication.

## 2026-05-03: MNIST Loader and Smoke Run

Environment:

- System Python blocks direct `pip install` because it is externally managed.
- Created `.venv --system-site-packages`.
- Installed `torchvision 0.26.0` and `torch 2.11.0+cu130` inside `.venv`.
- System Python remains on `torch 2.10.0+cu128` without `torchvision`.

Command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --epochs 1 --imp-rounds 1 --hidden-dim 64 --batch-size 512 --lr 0.05 --sgld-steps 20 --sgld-burn-in 5 --sgld-sample-every 2 --samples 5 --random-trials 10 --barrier-samples 2 --barrier-points 5
.venv/bin/python scripts/summarize_digits_runs.py --run-root runs/mnist_pilot --out-csv runs/mnist_pilot_summary.csv --out-json runs/mnist_pilot_summary.json
```

Output:

- `runs/mnist_pilot/20260503_101846/metrics.json`
- `runs/mnist_pilot_summary.json`

Key numbers:

| Metric | Value |
| --- | ---: |
| Dense accuracy | 0.9370 |
| IMP accuracy | 0.9385 |
| IMP sparsity | 0.2500 |
| Posterior mask to IMP Jaccard | 0.7150 |
| Random mask to IMP Jaccard | 0.6001 |
| Dense magnitude mask to IMP Jaccard | 0.8949 |
| Initial magnitude mask to IMP Jaccard | 0.7595 |
| SGLD sample accuracy | 0.9331 |
| State/function clusters | 1 / 1 |
| Linear barriers | 0.0000 |

Interpretation:

MNIST data loading and the pipeline work, but this is only a one-epoch smoke run
at 25% sparsity. The dense and initial magnitude controls again dominate the
posterior-mask signal. A serious MNIST run needs deeper IMP, more SGLD samples,
and repeated seeds before any scientific claim.

## 2026-05-03: Corrected SGLD and Multi-Chain Smoke

Code changes:

- `DatasetBundle` now records train/test set sizes.
- `SGLDConfig` now records `num_train_examples` and `likelihood_scale`.
- SGLD supports `dataset` scaling: minibatch NLL sum times `N / batch_size`.
- `scripts/run_digits_pilot.py` supports multiple SGLD chains and
  `--sgld-chain-init independent-dense`.
- Summaries now include chain-start magnitude controls.

Digits command:

```bash
python scripts/run_digits_pilot.py --dataset digits --seed 0 --epochs 2 --imp-rounds 2 --sgld-chains 2 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr 1e-7 --sgld-steps 30 --sgld-burn-in 10 --sgld-sample-every 4 --samples 5 --random-trials 20 --barrier-samples 4 --barrier-points 5 --out-dir runs/digits_multichain_smoke
```

MNIST command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --seed 0 --epochs 1 --imp-rounds 1 --hidden-dim 64 --batch-size 512 --lr 0.05 --sgld-chains 2 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr 1e-8 --sgld-steps 20 --sgld-burn-in 5 --sgld-sample-every 5 --samples 3 --random-trials 20 --barrier-samples 4 --barrier-points 5 --out-dir runs/mnist_multichain_smoke
```

Key numbers:

| Dataset | Posterior Jaccard | Random Jaccard | Chain-start to IMP | Posterior to chain-start | Dense magnitude to IMP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Digits | 0.4011 | 0.3927 | 0.4013 | 0.9808 | 0.9239 |
| MNIST | 0.6052 | 0.5998 | 0.6050 | 0.9916 | 0.8949 |

Interpretation:

After using dataset-scaled SGLD and independent dense chains, the posterior mask
advantage over random becomes very small. More importantly, posterior masks are
almost identical to the chain-start magnitude masks. This strongly suggests
that naive SGLD around trained dense checkpoints is not yet discovering new
posterior support structure; it is mostly preserving the starting basin's
magnitude support. The next experiment must either use stronger posterior
movement or accept a negative-result pivot.

## 2026-05-03: MNIST Gate1 Quick Run

Command:

```bash
for seed in 0 1 2; do .venv/bin/python scripts/run_digits_pilot.py --dataset mnist --seed "$seed" --epochs 3 --imp-rounds 4 --prune-fraction 0.30 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 3 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr 1e-8 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 6 --barrier-points 7 --out-dir runs/mnist_gate1_quick > "/tmp/mnist_gate1_quick_seed_${seed}.log"; done
.venv/bin/python scripts/summarize_digits_runs.py --run-root runs/mnist_gate1_quick --out-csv runs/mnist_gate1_quick_summary.csv --out-json runs/mnist_gate1_quick_summary.json
.venv/bin/python scripts/evaluate_gate1.py runs/mnist_gate1_quick_summary.json --out-json runs/mnist_gate1_quick_gate1_eval.json
```

Output:

- `runs/mnist_gate1_quick_summary.json`
- `runs/mnist_gate1_quick_gate1_eval.json`

Aggregate over seeds 0-2:

| Metric | Mean |
| --- | ---: |
| Dense accuracy | 0.9575 |
| IMP accuracy | 0.9659 |
| IMP sparsity | 0.7599 |
| Posterior mask to IMP Jaccard | 0.2387 |
| Random mask to IMP Jaccard | 0.1365 |
| Chain-start magnitude to IMP Jaccard | 0.2388 |
| Posterior to chain-start Jaccard | 0.9453 |
| Dense magnitude to IMP Jaccard | 0.6605 |
| State/function clusters | 3 / 3 |
| Dense-sample linear barrier | 0.8018 |
| IMP-sample linear barrier | 0.8731 |

Gate1 evaluator:

- `posterior_beats_random`: pass.
- `posterior_exceeds_chain_start`: fail.
- `posterior_moves_support_from_chain_start`: fail.
- `dense_magnitude_does_not_dominate`: fail.
- Overall: fail.

Interpretation:

This quick MNIST Gate1 run is strong evidence against the current positive
claim. Independent dense starts create distinct parameter/function clusters and
high linear barriers, but SGLD samples stay close to each chain's starting
magnitude support. IMP is far better predicted by ordinary dense magnitude than
by posterior-induced masks from these chains. The current best paper direction
is shifting toward a negative result unless stronger samplers, cyclical
temperature schedules, or alternative posterior-to-mask maps change this.

## 2026-05-03: SGLD Movement Rescue Sweep

Command:

```bash
for temp in 1 10 50; do for lr in 1e-8 3e-8; do name="temp_${temp}_lr_${lr}"; .venv/bin/python scripts/run_digits_pilot.py --dataset mnist --seed 0 --epochs 2 --imp-rounds 3 --prune-fraction 0.30 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 2 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr "$lr" --sgld-temperature "$temp" --sgld-steps 150 --sgld-burn-in 50 --sgld-sample-every 10 --samples 8 --random-trials 80 --barrier-samples 4 --barrier-points 5 --out-dir "runs/mnist_sgld_rescue_${name}" > "/tmp/mnist_sgld_rescue_${name}.log"; done; done
.venv/bin/python scripts/summarize_rescue_sweep.py --pattern 'runs/mnist_sgld_rescue_*/*/metrics.json' --out-csv runs/mnist_sgld_rescue_summary.csv
```

Output:

- `runs/mnist_sgld_rescue_summary.csv`

Key numbers:

| Setting | SGLD acc | Posterior | Chain-start | Posterior to chain-start | Dense magnitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| temp 1, lr 1e-8 | 0.9461 | 0.2622 | 0.2624 | 0.9542 | 0.7697 |
| temp 1, lr 3e-8 | 0.9470 | 0.2624 | 0.2624 | 0.9225 | 0.7697 |
| temp 10, lr 1e-8 | 0.9459 | 0.2625 | 0.2624 | 0.8642 | 0.7697 |
| temp 10, lr 3e-8 | 0.9461 | 0.2623 | 0.2624 | 0.7815 | 0.7697 |
| temp 50, lr 1e-8 | 0.9453 | 0.2618 | 0.2624 | 0.7336 | 0.7697 |
| temp 50, lr 3e-8 | 0.9440 | 0.2592 | 0.2624 | 0.6132 | 0.7697 |

Interpretation:

Increasing SGLD temperature and step size moves supports away from the chain
start, while preserving reasonable accuracy in this short run. But movement
does not increase alignment with IMP; the posterior-to-IMP Jaccard stays around
the chain-start value or decreases. This weakens the rescue hypothesis that
more SGLD motion will reveal posterior supports aligned with IMP.

## 2026-05-03: Posterior-to-Mask Map Smoke

Code changes:

- Added `src/lottery/posterior_maps.py`.
- Added posterior mean-absolute, RMS, SNR, high-variance, and low-variance masks.
- `scripts/run_digits_pilot.py` now records aggregate and chainwise posterior
  map Jaccards.

Command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --seed 0 --epochs 1 --imp-rounds 1 --hidden-dim 64 --batch-size 1024 --lr 0.05 --sgld-chains 2 --sgld-chain-init independent-dense --sgld-likelihood-scale dataset --sgld-lr 1e-8 --sgld-temperature 10 --sgld-steps 30 --sgld-burn-in 10 --sgld-sample-every 5 --samples 4 --random-trials 20 --barrier-samples 2 --barrier-points 5 --out-dir runs/mnist_posterior_maps_smoke
```

Output:

- `runs/mnist_posterior_maps_smoke/20260503_103632/metrics.json`

Key numbers:

| Map/control | Jaccard to IMP |
| --- | ---: |
| Posterior sample magnitude mean | 0.6050 |
| Posterior mean abs | 0.6057 |
| Posterior RMS | 0.6057 |
| Posterior SNR | 0.5989 |
| Posterior high variance | 0.6062 |
| Posterior low variance | 0.5836 |
| Chain-start magnitude mean | 0.6055 |
| Dense magnitude | 0.9426 |

Interpretation:

Alternative posterior-to-mask maps do not rescue the positive claim in this
smoke setting. The maps stay near the chain-start control and far below dense
magnitude alignment with IMP.

## 2026-05-03: SNIP/SynFlow Control Smoke

Code changes:

- Added `src/lottery/pruning_baselines.py`.
- `scripts/run_digits_pilot.py` now records `snip_to_imp_jaccard` and
  `synflow_to_imp_jaccard`.

Command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --seed 0 --epochs 1 --imp-rounds 1 --hidden-dim 64 --batch-size 1024 --lr 0.05 --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale dataset --sgld-lr 1e-8 --sgld-steps 20 --sgld-burn-in 5 --sgld-sample-every 5 --samples 3 --random-trials 10 --snip-batches 2 --barrier-samples 2 --barrier-points 5 --out-dir runs/mnist_pruning_controls_smoke
```

Output:

- `runs/mnist_pruning_controls_smoke/20260503_103832/metrics.json`

Key numbers:

| Control | Jaccard to IMP |
| --- | ---: |
| Posterior sample magnitude | 0.9417 |
| Dense magnitude | 0.9426 |
| Chain-start magnitude | 0.9426 |
| Initial magnitude | 0.7697 |
| SynFlow | 0.7684 |
| SNIP | 0.7012 |
| Random | 0.5999 |

Interpretation:

In the single-chain dense-start setting, posterior sample magnitude is almost
identical to dense and chain-start magnitude. SNIP and SynFlow provide useful
non-Bayesian controls for the eventual negative-result paper.

## 2026-05-03: Image Model and ResNet-20 Smoke

Code changes:

- `DatasetBundle` now records `input_shape`.
- `load_torchvision_bundle` can keep image tensors instead of flattening.
- Added `TinyCNN` and CIFAR-style `ResNetCIFAR`/ResNet-20 in `src/lottery/models.py`.
- `scripts/run_digits_pilot.py` supports `--model mlp|tiny-cnn|resnet20`.
- `synflow_mask` now handles non-flat image input shapes.
- Added `fake-cifar10` for code-path validation when real CIFAR-10 download is
  unavailable.

MNIST TinyCNN smoke:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset mnist --model tiny-cnn --seed 0 --epochs 1 --imp-rounds 1 --batch-size 2048 --lr 0.03 --cnn-width 8 --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale mean --sgld-lr 1e-6 --sgld-steps 10 --sgld-burn-in 2 --sgld-sample-every 4 --samples 2 --random-trials 5 --snip-batches 1 --barrier-samples 1 --barrier-points 3 --out-dir runs/mnist_tinycnn_smoke
```

Real CIFAR-10 attempt:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --resnet-width 4 --seed 0 --epochs 1 --imp-rounds 1 --batch-size 2048 --lr 0.03 --weight-decay 5e-4 --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale mean --sgld-lr 1e-6 --sgld-steps 8 --sgld-burn-in 2 --sgld-sample-every 3 --samples 2 --random-trials 5 --snip-batches 1 --barrier-samples 1 --barrier-points 3 --out-dir runs/cifar10_resnet20_smoke
```

Result:

- Real CIFAR-10 download failed with `HTTP Error 503: Service Unavailable`.
- This is a data availability blocker, not a model/runner compile failure.

Fake-CIFAR ResNet-20 code-path smoke:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset fake-cifar10 --model resnet20 --resnet-width 4 --seed 0 --epochs 1 --imp-rounds 1 --batch-size 512 --lr 0.03 --weight-decay 5e-4 --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale mean --sgld-lr 1e-6 --sgld-steps 8 --sgld-burn-in 2 --sgld-sample-every 3 --samples 2 --random-trials 5 --snip-batches 1 --barrier-samples 1 --barrier-points 3 --out-dir runs/fake_cifar10_resnet20_smoke
.venv/bin/python scripts/summarize_digits_runs.py --run-root runs/fake_cifar10_resnet20_smoke --out-csv runs/fake_cifar10_resnet20_smoke_summary.csv --out-json runs/fake_cifar10_resnet20_smoke_summary.json
```

Output:

- `runs/fake_cifar10_resnet20_smoke/20260503_104210/metrics.json`
- `runs/fake_cifar10_resnet20_smoke_summary.json`

Interpretation:

The ResNet-20 code path works end to end, including IMP, SGLD, posterior maps,
SNIP, SynFlow, clustering, and connectivity. This is not scientific evidence
about CIFAR-10; it only verifies that the implementation is ready once the
dataset can be downloaded.

## 2026-05-03: Small-Model HMC Baseline

Code changes:

- Added `src/lottery/hmc.py`, a full-batch HMC sampler for small models.
- Added `scripts/run_digits_hmc_baseline.py`.
- Fixed HMC prior gradient so the Gaussian prior contributes to leapfrog
  dynamics, not only to the acceptance energy.

Smoke command:

```bash
python scripts/run_digits_hmc_baseline.py --seed 0 --hidden-dim 6 --depth 2 --epochs 5 --imp-rounds 1 --prune-fraction 0.30 --hmc-steps 12 --hmc-step-size 1e-5 --hmc-leapfrog-steps 3 --hmc-burn-in 3 --hmc-sample-every 2 --random-trials 20
```

Sweep command:

```bash
for eps in 2e-5 5e-5 1e-4 2e-4; do python scripts/run_digits_hmc_baseline.py --seed 0 --hidden-dim 8 --depth 2 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --hmc-steps 20 --hmc-step-size "$eps" --hmc-leapfrog-steps 4 --hmc-burn-in 5 --hmc-sample-every 3 --random-trials 30 --out-dir "runs/digits_hmc_eps_${eps}" > "/tmp/digits_hmc_eps_${eps}.log"; done
```

Output:

- `runs/digits_hmc_baseline/20260503_104416/metrics.json`
- `runs/digits_hmc_priorfix_smoke/20260503_104528/metrics.json`
- `runs/digits_hmc_sweep_summary.csv`

Sweep summary:

| HMC step size | Accept | HMC mask to IMP | Random | Dense magnitude | HMC to dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2e-5 | 1.0000 | 0.8893 | 0.3278 | 0.8893 | 1.0000 |
| 5e-5 | 1.0000 | 0.8893 | 0.3278 | 0.8893 | 0.9986 |
| 1e-4 | 1.0000 | 0.8868 | 0.3278 | 0.8893 | 0.9890 |
| 2e-4 | 1.0000 | 0.8843 | 0.3278 | 0.8893 | 0.9741 |

Interpretation:

The HMC path is now implemented, but this is still a small and conservative
baseline. In the current sweep, HMC masks beat random but remain almost identical
to dense magnitude masks, matching the SGLD story. This reduces, but does not
eliminate, the concern that the negative result is purely an SGLD artifact. This
section is superseded by the tuned 5-seed HMC baseline later in this log.

## 2026-05-03: Fashion-MNIST Gate1 Quick Replication

Code changes:

- Added `scripts/run_gate1_sweep.py` to run repeated Gate1 experiments and then
  call summarization/evaluation automatically.
- Added `scripts/build_results_table.py` to generate paper-facing result
  snapshots from summary JSON files.

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset fashion-mnist --seeds 0,1 --epochs 2 --imp-rounds 3 --prune-fraction 0.30 --hidden-dim 128 --batch-size 2048 --lr 0.05 --sgld-chains 2 --sgld-lr 1e-8 --sgld-steps 120 --sgld-burn-in 40 --sgld-sample-every 10 --samples 8 --random-trials 80 --barrier-samples 4 --barrier-points 5 --out-dir runs/fashion_gate1_quick --summary-prefix runs/fashion_gate1_quick
.venv/bin/python scripts/build_results_table.py --entry MNIST:runs/mnist_gate1_quick_summary.json:runs/mnist_gate1_quick_gate1_eval.json --entry Fashion-MNIST:runs/fashion_gate1_quick_summary.json:runs/fashion_gate1_quick_gate1_eval.json --out-csv runs/results_snapshot.csv --out-md docs/results_snapshot.md
```

Output:

- `runs/fashion_gate1_quick_summary.json`
- `runs/fashion_gate1_quick_gate1_eval.json`
- `docs/results_snapshot.md`

Fashion-MNIST aggregate over seeds 0-1:

| Metric | Mean |
| --- | ---: |
| Dense accuracy | 0.8091 |
| IMP accuracy | 0.8286 |
| IMP sparsity | 0.6570 |
| Posterior mask to IMP Jaccard | 0.2577 |
| Random mask to IMP Jaccard | 0.2068 |
| Chain-start magnitude to IMP Jaccard | 0.2577 |
| Posterior to chain-start Jaccard | 0.9581 |
| Dense magnitude to IMP Jaccard | 0.7101 |
| SNIP to IMP Jaccard | 0.3595 |
| SynFlow to IMP Jaccard | 0.6046 |
| Gate1 | fail |

Interpretation:

Fashion-MNIST reproduces the MNIST pattern. Posterior masks beat random, but do
not exceed chain-start magnitude, stay very close to chain-start supports, and
are dominated by dense magnitude. SynFlow is a stronger non-Bayesian control
than posterior masks in this quick run.

## 2026-05-03: Full Sweep Orchestrator

Code changes:

- Added `scripts/run_gate1_sparsity_sweep.py`.

Dry-run command:

```bash
.venv/bin/python scripts/run_gate1_sparsity_sweep.py --dataset mnist --seeds 0 --configs 2:0.30,3:0.30 --epochs 1 --sgld-steps 10 --samples 2 --dry-run
```

Interpretation:

The full seed/sparsity sweep is now scripted but not yet executed. This is the
next required long-running experiment for submission-grade evidence.

Actual smoke command:

```bash
.venv/bin/python scripts/run_gate1_sparsity_sweep.py --dataset mnist --seeds 0 --configs 1:0.30,2:0.30 --epochs 1 --hidden-dim 64 --batch-size 2048 --sgld-chains 1 --sgld-steps 20 --sgld-burn-in 5 --sgld-sample-every 5 --samples 3 --random-trials 10 --barrier-samples 1 --barrier-points 3
```

Output:

- `runs/mnist_gate1_r1_p0p3_summary.json`
- `runs/mnist_gate1_r1_p0p3_gate1_eval.json`
- `runs/mnist_gate1_r2_p0p3_summary.json`
- `runs/mnist_gate1_r2_p0p3_gate1_eval.json`

Smoke result:

| Config | Sparsity | Posterior | Random | Chain-start | Dense magnitude | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| r1 p0.30 | 0.3000 | 0.5459 | 0.5387 | 0.5459 | 0.9703 | fail |
| r2 p0.30 | 0.5100 | 0.3497 | 0.3243 | 0.3498 | 0.8596 | fail |

Interpretation:

The sparsity-sweep wrapper works end to end. This smoke is not submission-grade,
but it shows the same failure mode at two sparsities: posterior masks are close
to chain-start magnitude and are dominated by dense magnitude.

Resume check:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset mnist --seeds 0 --epochs 1 --imp-rounds 1 --hidden-dim 64 --batch-size 2048 --sgld-chains 1 --sgld-steps 20 --sgld-burn-in 5 --sgld-sample-every 5 --samples 3 --random-trials 10 --barrier-samples 1 --barrier-points 3 --out-dir runs/mnist_gate1_r1_p0p3 --summary-prefix runs/mnist_gate1_r1_p0p3 --skip-existing-seeds
```

Result:

- Existing seed 0 was skipped.
- Summary and Gate1 evaluation were regenerated.

This confirms long sweeps can be resumed without rerunning completed seeds.

## 2026-05-03: Working Paper Draft

Files:

- `paper/main.tex`
- `paper/refs.bib`
- `paper/README.md`
- `paper/main.pdf`

Build command:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Result:

- PDF build succeeded.
- Citation keys were checked before build; no missing cite keys.

Interpretation:

The draft is a working negative-result manuscript with abstract, introduction,
related work, operational test, current results, discussion, and limitations.
It explicitly marks the current evidence as quick and not submission-grade.

## 2026-05-03: Real CIFAR-10 ResNet-20 Subset Smoke

Data note:

- The default torchvision CIFAR-10 URL returned `HTTP Error 503` during the
  first real-data attempt.
- A CIFAR-10 Python tarball was downloaded from the Zenodo mirror and placed at
  `data/cifar-10-python.tar.gz`.
- The tarball MD5 is `c58f30108f718f92721af3b95e74349a`, matching the canonical
  CIFAR-10 Python archive checksum.

Code changes:

- `load_torchvision_bundle` supports `train_subset` and `test_subset`.
- `fake-cifar10` accepts the same subset controls, so image-model smoke runs can
  remain cheap and deterministic.

Command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --resnet-width 4 --train-subset 512 --test-subset 256 --seed 0 --epochs 1 --imp-rounds 1 --batch-size 128 --lr 0.03 --weight-decay 5e-4 --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale mean --sgld-lr 1e-6 --sgld-steps 6 --sgld-burn-in 2 --sgld-sample-every 2 --samples 2 --random-trials 5 --snip-batches 1 --barrier-samples 1 --barrier-points 3 --out-dir runs/cifar10_resnet20_subset_smoke
.venv/bin/python scripts/summarize_digits_runs.py --run-root runs/cifar10_resnet20_subset_smoke --out-csv runs/cifar10_resnet20_subset_smoke_summary.csv --out-json runs/cifar10_resnet20_subset_smoke_summary.json
```

Output:

- `runs/cifar10_resnet20_subset_smoke/20260503_105959/metrics.json`
- `runs/cifar10_resnet20_subset_smoke_summary.json`

Subset-smoke numbers:

| Metric | Value |
| --- | ---: |
| Train/test subset | 512 / 256 |
| Dense accuracy | 0.0781 |
| IMP accuracy | 0.0742 |
| IMP sparsity | 0.2500 |
| Posterior mask to IMP Jaccard | 0.9742 |
| Random mask to IMP Jaccard | 0.6005 |
| Dense magnitude mask to IMP Jaccard | 0.9901 |
| Initial magnitude mask to IMP Jaccard | 0.9876 |
| Posterior to chain-start Jaccard | 0.9760 |
| SNIP to IMP Jaccard | 0.7615 |
| SynFlow to IMP Jaccard | 0.9142 |

Interpretation:

The real CIFAR-10 ResNet-20 path now runs end to end on an actual CIFAR-10
archive, including IMP, posterior maps, SNIP, SynFlow, clustering, and
connectivity. This is still not scientific evidence: the subset is tiny, the
training budget is one epoch, and accuracy is around chance. The value of this
run is implementation risk reduction. The next CIFAR milestone is a real
training run, not more subset-smoke interpretation.

## 2026-05-03: MNIST Full Gate1 Sparsity Sweep

Code changes:

- `scripts/run_gate1_sparsity_sweep.py` now supports `--run-prefix`, so smoke
  runs and full sweeps do not share result directories.
- Added `scripts/build_sparsity_sweep_table.py`.

Command:

```bash
.venv/bin/python scripts/run_gate1_sparsity_sweep.py --dataset mnist --run-prefix mnist_gate1_full --seeds 0,1,2,3,4 --configs 2:0.30,3:0.30,5:0.30,8:0.30 --epochs 5 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 3 --sgld-lr 1e-8 --sgld-steps 600 --sgld-burn-in 200 --sgld-sample-every 20 --samples 20 --random-trials 200 --barrier-samples 20 --barrier-points 11 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/mnist_gate1_full_r2_p0p3_summary.json --summary runs/mnist_gate1_full_r3_p0p3_summary.json --summary runs/mnist_gate1_full_r5_p0p3_summary.json --summary runs/mnist_gate1_full_r8_p0p3_summary.json --out-csv runs/mnist_gate1_full_sweep.csv --out-md docs/mnist_gate1_full_sweep.md
```

Output:

- `runs/mnist_gate1_full_r2_p0p3_summary.json`
- `runs/mnist_gate1_full_r3_p0p3_summary.json`
- `runs/mnist_gate1_full_r5_p0p3_summary.json`
- `runs/mnist_gate1_full_r8_p0p3_summary.json`
- `docs/mnist_gate1_full_sweep.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r2 p0.30 | 5 | 0.5100 | 0.3539 | 0.3246 | 0.3537 | 0.9303 | 0.8259 | fail |
| r3 p0.30 | 5 | 0.6570 | 0.2669 | 0.2071 | 0.2670 | 0.9115 | 0.7258 | fail |
| r5 p0.30 | 5 | 0.8319 | 0.2373 | 0.0918 | 0.2385 | 0.9298 | 0.5924 | fail |
| r8 p0.30 | 5 | 0.9424 | 0.1275 | 0.0298 | 0.1275 | 0.9423 | 0.4028 | fail |

Interpretation:

MNIST now has the planned 5-seed, 4-sparsity Gate1 sweep. Posterior masks beat
random at every sparsity, but they do not exceed chain-start magnitude, remain
highly overlapping with chain-start supports, and are dominated by dense
magnitude masks. This is solid evidence for the negative-result framing on
MNIST. The next required dataset-level replication is the Fashion-MNIST full
sweep.

## 2026-05-03: Fashion-MNIST Full Gate1 Sparsity Sweep

Command:

```bash
.venv/bin/python scripts/run_gate1_sparsity_sweep.py --dataset fashion-mnist --run-prefix fashion_gate1_full --seeds 0,1,2,3,4 --configs 2:0.30,3:0.30,5:0.30,8:0.30 --epochs 5 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 3 --sgld-lr 1e-8 --sgld-steps 600 --sgld-burn-in 200 --sgld-sample-every 20 --samples 20 --random-trials 200 --barrier-samples 20 --barrier-points 11 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/fashion_gate1_full_r2_p0p3_summary.json --summary runs/fashion_gate1_full_r3_p0p3_summary.json --summary runs/fashion_gate1_full_r5_p0p3_summary.json --summary runs/fashion_gate1_full_r8_p0p3_summary.json --out-csv runs/fashion_gate1_full_sweep.csv --out-md docs/fashion_gate1_full_sweep.md
```

Output:

- `runs/fashion_gate1_full_r2_p0p3_summary.json`
- `runs/fashion_gate1_full_r3_p0p3_summary.json`
- `runs/fashion_gate1_full_r5_p0p3_summary.json`
- `runs/fashion_gate1_full_r8_p0p3_summary.json`
- `docs/fashion_gate1_full_sweep.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r2 p0.30 | 5 | 0.5100 | 0.3516 | 0.3246 | 0.3516 | 0.9314 | 0.7385 | fail |
| r3 p0.30 | 5 | 0.6570 | 0.2595 | 0.2070 | 0.2594 | 0.9160 | 0.6378 | fail |
| r5 p0.30 | 5 | 0.8319 | 0.2114 | 0.0917 | 0.2122 | 0.9212 | 0.5255 | fail |
| r8 p0.30 | 5 | 0.9424 | 0.1300 | 0.0297 | 0.1300 | 0.9408 | 0.3813 | fail |

Interpretation:

Fashion-MNIST reproduces the MNIST full-sweep failure. Posterior masks beat
random at every sparsity but match chain-start magnitude and are dominated by
dense magnitude. The highest sparsity row has lower IMP accuracy than dense
accuracy, so it should be described as a high-sparsity stress case rather than a
clean winning-ticket regime. The overall result still strengthens the negative
claim because the same control failure appears across both datasets.

## 2026-05-03: MNIST SWAG Posterior Baseline at r5 p0.30

Code changes:

- Added `src/lottery/swag.py`, a state-dict SWAG sampler with diagonal plus
  low-rank covariance sampling.
- `scripts/run_digits_pilot.py`, `scripts/run_gate1_sweep.py`, and
  `scripts/run_gate1_sparsity_sweep.py` now support
  `--posterior-sampler swag`.
- `scripts/summarize_digits_runs.py` now reads sampler-neutral
  `metrics["posterior"]` while preserving the legacy `sgld_*` summary columns.

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset mnist --model mlp --posterior-sampler swag --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 5 --prune-fraction 0.30 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 3 --samples 20 --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --random-trials 200 --barrier-samples 20 --barrier-points 11 --out-dir runs/mnist_swag_r5_p0p3 --summary-prefix runs/mnist_swag_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/mnist_swag_r5_p0p3_summary.json --out-csv runs/mnist_swag_r5_p0p3_table.csv --out-md docs/mnist_swag_r5_p0p3.md
```

Output:

- `runs/mnist_swag_r5_p0p3_summary.json`
- `runs/mnist_swag_r5_p0p3_gate1_eval.json`
- `docs/mnist_swag_r5_p0p3.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 | 5 | 0.8319 | 0.2383 | 0.0918 | 0.2385 | 0.9581 | 0.5924 | fail |

Interpretation:

SWAG does not rescue the posterior-mode explanation at the representative MNIST
r5 sparsity. It improves sample accuracy over the independent dense starts
(`0.9711` vs. `0.9676`) and still beats random support overlap, but its masks
match chain-start magnitude (`posterior - chain-start = -0.0002`), remain highly
overlapping with chain-start supports (`0.9581`), and are dominated by dense
magnitude (`dense - posterior = 0.3541`). This reduces the concern that the
negative result is only an SGLD artifact, but it is not yet a complete stronger
posterior baseline because only one MNIST sparsity has been run.

## 2026-05-03: Fashion-MNIST SWAG Posterior Baseline at r5 p0.30

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset fashion-mnist --model mlp --posterior-sampler swag --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 5 --prune-fraction 0.30 --hidden-dim 128 --batch-size 1024 --lr 0.05 --sgld-chains 3 --samples 20 --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --random-trials 200 --barrier-samples 20 --barrier-points 11 --out-dir runs/fashion_swag_r5_p0p3 --summary-prefix runs/fashion_swag_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/fashion_swag_r5_p0p3_summary.json --out-csv runs/fashion_swag_r5_p0p3_table.csv --out-md docs/fashion_swag_r5_p0p3.md
```

Output:

- `runs/fashion_swag_r5_p0p3_summary.json`
- `runs/fashion_swag_r5_p0p3_gate1_eval.json`
- `docs/fashion_swag_r5_p0p3.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 | 5 | 0.8319 | 0.2117 | 0.0917 | 0.2122 | 0.9480 | 0.5255 | fail |

Interpretation:

Fashion-MNIST replicates the MNIST SWAG result at the same sparsity. SWAG sample
accuracy again improves over the independent dense starts (`0.8725` vs.
`0.8640`), but support alignment remains a chain-start effect: posterior masks
do not beat chain-start magnitude (`posterior - chain-start = -0.0005`), stay
close to chain-start supports (`0.9480`), and are far below dense magnitude
(`dense - posterior = 0.3138`). The SWAG evidence now covers two datasets at
the representative r5 sparsity, but not yet the full sparsity grid.

## 2026-05-03: CIFAR-10 Full-Data ResNet-20 Training Baseline

Code changes:

- `src/lottery/data.py` now supports CIFAR crop/flip augmentation through
  `augment=True`.
- `src/lottery/train.py` and `src/lottery/imp.py` now support
  `lr_schedule="cosine"` while preserving the old constant-LR default.
- `scripts/run_digits_pilot.py`, `scripts/run_gate1_sweep.py`, and
  `scripts/run_gate1_sparsity_sweep.py` expose `--lr-schedule` and `--augment`.
- Added `scripts/run_cifar_baseline.py` for dense-only CIFAR training sanity
  checks before running expensive IMP/posterior sweeps.

Command:

```bash
.venv/bin/python scripts/run_cifar_baseline.py --seed 0 --epochs 10 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_baseline
```

Output:

- `runs/cifar10_resnet20_baseline/20260503_135923/metrics.json`

Result:

The 10-epoch full-data ResNet-20 baseline reached 0.8302 CIFAR-10 test
accuracy. This clears the previous CIFAR blocker where the only real-data run
was a tiny near-chance subset smoke.

## 2026-05-03: CIFAR-10 Full-Data ResNet-20 Gate1 Pilot

Command:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale dataset --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --snip-batches 1 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_gate1_pilot_r2_p0p3
.venv/bin/python scripts/summarize_digits_runs.py --run-root runs/cifar10_resnet20_gate1_pilot_r2_p0p3 --out-csv runs/cifar10_resnet20_gate1_pilot_r2_p0p3_summary.csv --out-json runs/cifar10_resnet20_gate1_pilot_r2_p0p3_summary.json
.venv/bin/python scripts/evaluate_gate1.py runs/cifar10_resnet20_gate1_pilot_r2_p0p3_summary.json --out-json runs/cifar10_resnet20_gate1_pilot_r2_p0p3_gate1_eval.json
```

Output:

- `runs/cifar10_resnet20_gate1_pilot_r2_p0p3/20260503_140118/metrics.json`
- `runs/cifar10_resnet20_gate1_pilot_r2_p0p3_summary.json`
- `runs/cifar10_resnet20_gate1_pilot_r2_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_gate1_pilot_r2_p0p3.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r2 p0.30 | 1 | 0.5100 | 0.4062 | 0.3246 | 0.4061 | 0.9968 | 0.4061 | fail |

Interpretation:

This is the first non-chance full-data CIFAR Gate1 result: dense accuracy is
0.8254 and IMP accuracy is 0.8500. The posterior mask beats random, but it is
indistinguishable from the dense/chain-start magnitude support and has
posterior-to-chain-start overlap of 0.9968. This is aligned with the negative
story, but it is only one seed, one sparsity, one dense-start SGLD chain, and a
short 10-epoch training budget. It should be treated as a pilot, not paper
evidence.

## 2026-05-03: CIFAR-10 ResNet-20 3-Seed Short Gate1 Grid

Command shape:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_gate1_short_r2_p0p3 --summary-prefix runs/cifar10_resnet20_gate1_short_r2_p0p3 --skip-existing-seeds
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_gate1_short_r5_p0p3 --summary-prefix runs/cifar10_resnet20_gate1_short_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2 --epochs 10 --imp-rounds 8 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_gate1_short_r8_p0p3 --summary-prefix runs/cifar10_resnet20_gate1_short_r8_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_gate1_short_r2_p0p3_summary.json --summary runs/cifar10_resnet20_gate1_short_r5_p0p3_summary.json --summary runs/cifar10_resnet20_gate1_short_r8_p0p3_summary.json --out-csv runs/cifar10_resnet20_gate1_short_sweep.csv --out-md docs/cifar10_resnet20_gate1_short_sweep.md
```

Output:

- `runs/cifar10_resnet20_gate1_short_r2_p0p3_summary.json`
- `runs/cifar10_resnet20_gate1_short_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_gate1_short_r8_p0p3_summary.json`
- `docs/cifar10_resnet20_gate1_short_sweep.md`

Result:

| Config | Runs | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r2 p0.30 | 3 | 0.5100 | 0.3446 | 0.3245 | 0.3445 | 0.9967 | 0.4122 | fail |
| r5 p0.30 | 5 | 0.8319 | 0.1297 | 0.0917 | 0.1297 | 0.9963 | 0.1704 | fail |
| r8 p0.30 | 3 | 0.9424 | 0.0786 | 0.0296 | 0.0785 | 0.9969 | 0.1064 | fail |

Interpretation:

The short CIFAR grid now gives non-chance image-model evidence in the same
direction as MNIST/Fashion. IMP accuracy remains strong (`0.8491`, `0.8634`,
`0.8534`) across sparsities. Posterior masks beat random in all rows, but the
posterior-chain-start gap is effectively zero and posterior-to-chain-start
overlap is about `0.997` throughout. This is a substantial improvement over the
old CIFAR subset smoke. It is still not final submission-grade evidence because
it uses 10 training epochs, one posterior chain, and only 10 posterior samples
per seed. The r5 p0.30 row was later extended from 3 to 5 seeds; r2/r8 remain
3-seed short rows.

## 2026-05-03: CIFAR-10 ResNet-20 SWAG Short Control

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --sgld-chains 1 --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_swag_short_r5_p0p3 --summary-prefix runs/cifar10_resnet20_swag_short_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_swag_short_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_swag_short_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_swag_short_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_swag_short_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_swag_short_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_swag_short_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 SWAG | 5 | 0.8255 | 0.8631 | 0.8319 | 0.1302 | 0.0917 | 0.1304 | 0.9097 | 0.1703 | fail |

Interpretation:

SWAG moves supports farther from the chain start than the very-low-step-size
short SGLD sampler (`0.9097` vs. `0.9963` posterior-to-chain-start overlap),
but it does not rescue Gate1. The SWAG posterior mask beats random, yet it does
not exceed the independent dense chain-start magnitude mask and remains highly
overlapping with the chain-start support. This weakens the objection that the
CIFAR short-grid failure is only an SGLD artifact.

## 2026-05-03: CIFAR-10 ResNet-20 SGLD Multi-Chain Short Control

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 3 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3 --summary-prefix runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_sgld_multichain_short_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_sgld_multichain_short_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | State Clusters | Function Clusters | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 SGLD-3chain | 5 | 0.8215 | 0.8621 | 0.8319 | 0.1291 | 0.0917 | 0.1291 | 0.9963 | 0.1689 | 3.0 | 3.0 | fail |

Interpretation:

This directly tests the single-chain objection. The run uses three independent
dense starts per seed and collects 10 SGLD samples per chain. State and
function clustering correctly recover three clusters, but posterior support
still matches each chain-start magnitude mask. The posterior-random gap remains
positive, yet the posterior-chain-start gap is effectively zero.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch r5 Pilot

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_long30_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_long30_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_long30_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_long30_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_long30_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 30ep | 1 | 0.8836 | 0.8584 | 0.8319 | 0.1381 | 0.0916 | 0.1381 | 0.9972 | 0.1424 | fail |

Interpretation:

This is a pilot, not paper evidence. Longer dense training works, but the r5
IMP schedule underperforms dense at this budget, so long-training CIFAR needs a
tuned IMP schedule before scale-up. The Gate1 failure pattern still appears:
posterior support beats random but matches chain-start magnitude and remains
almost identical to the chain-start support.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch Gradual-Pruning Pilot

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --imp-rounds 8 --prune-fraction 0.20 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_r8_p0p2 --summary-prefix runs/cifar10_resnet20_long30_r8_p0p2 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_long30_r8_p0p2_summary.json --out-csv runs/cifar10_resnet20_long30_r8_p0p2_table.csv --out-md docs/cifar10_resnet20_long30_r8_p0p2.md
```

Output:

- `runs/cifar10_resnet20_long30_r8_p0p2_summary.json`
- `runs/cifar10_resnet20_long30_r8_p0p2_gate1_eval.json`
- `docs/cifar10_resnet20_long30_r8_p0p2.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8 p0.20 30ep | 1 | 0.8846 | 0.8581 | 0.8322 | 0.1284 | 0.0915 | 0.1286 | 0.9968 | 0.1339 | fail |

Interpretation:

This tests whether the 30-epoch r5 underperformance is due to pruning too
aggressively per round. At nearly the same sparsity, 8 rounds with p0.20 still
underperforms dense and still fails Gate1. The long-budget CIFAR schedule
therefore needs a different tuning axis, not just smaller pruning increments.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch Epoch-1 Rewind Pilot

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_long30_rewind1_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_long30_rewind1_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_long30_rewind1_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_long30_rewind1_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_long30_rewind1_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Rewind Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 30ep epoch-1 rewind | 5 | 0.8859 | 0.8980 | 0.8319 | 0.1342 | 0.0918 | 0.1342 | 0.9969 | 0.1472 | 0.1783 | fail |

Interpretation:

This is the first long-budget CIFAR pilot where IMP is consistently competitive
with dense: epoch-1 rewind raises the r5 sparse model from the one-seed
initialization-rewind baseline's 0.8584 to a 5-seed mean of 0.8980, above the
5-seed dense mean of 0.8859. The Gate1 failure remains unchanged. Posterior
support beats random, but it is indistinguishable from chain-start magnitude
(`posterior - chain-start = 0.0000`) and stays almost identical to the
chain-start support (`0.9969`). The epoch-1 rewind magnitude support is also
closer to IMP (`0.1783`) than the posterior support is (`0.1342`), which
reinforces the trajectory/rewinding explanation rather than the posterior-mode
claim.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch Epoch-1 Rewind SWAG Control

Command:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --sgld-chains 1 --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Rewind Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 30ep epoch-1 rewind SWAG | 5 | 0.8859 | 0.8969 | 0.8319 | 0.1361 | 0.0918 | 0.1361 | 0.9265 | 0.1463 | 0.1786 | fail |

Interpretation:

This is the long-budget counterpart to the short CIFAR SWAG control. SWAG moves
posterior supports farther from the chain-start support than the very-low-step
SGLD run (`0.9265` vs. `0.9969` Post-Chain), but it still fails the key test:
posterior-to-IMP overlap does not exceed chain-start magnitude overlap. The
SWAG samples also have lower mean accuracy (`0.8573`) than the chain-start
dense model (`0.8832`), so this setting is useful as a posterior-movement
control rather than as a high-quality predictive posterior.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch Epoch-1 Rewind SGLD Multi-Chain Control

Commands:

```bash
.venv/bin/python scripts/run_gate1_sweep.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 3 --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5 --out-dir runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3 --summary-prefix runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3 --skip-existing-seeds
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_summary.json --out-csv runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_table.csv --out-md docs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3.md
```

The run was executed in two batches (`--seeds 0,1,2`, then `--seeds 3,4`) with
`--skip-existing-seeds`; the command above is the equivalent reproducible
5-seed invocation.

Output:

- `runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_summary.json`
- `runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_gate1_eval.json`
- `docs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Rewind Mag | State Clusters | Function Clusters | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 30ep epoch-1 rewind SGLD-3chain | 5 | 0.8860 | 0.8990 | 0.8319 | 0.1368 | 0.0918 | 0.1368 | 0.9969 | 0.1460 | 0.1800 | 3.0 | 3.2 | fail |

Interpretation:

This is the long-budget counterpart to the short CIFAR SGLD multi-chain
control. The three independent dense starts remain separated: state and
function cluster counts average 3.0 and 3.2. That removes the simplest concern
that the single-chain posterior sampler was only probing one local basin. The
support result still does not improve. Posterior masks beat random, but they
match chain-start magnitude (`0.1368` vs. `0.1368`) and stay almost identical
to the chain-start support (`0.9969`). The epoch-1 rewind magnitude control is
closer to IMP (`0.1800`) than either posterior support is, so the long-budget
multi-chain result strengthens the negative trajectory/rewinding interpretation.

## 2026-05-03: CIFAR-10 ResNet-20 30-Epoch Epoch-1 Rewind SGLD Movement Diagnostic

Commands:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3.md`

Result:

| SGLD LR | Runs | Dense Acc | IMP Acc | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Sample Acc | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 5 | 0.8845 | 0.8993 | 0.1441 | 0.0918 | 0.1441 | 0.9969 | 0.1441 | 0.8842 | matches chain start |
| 1e-6 | 5 | 0.8845 | 0.8993 | 0.1425 | 0.0918 | 0.1441 | 0.7362 | 0.1441 | 0.8753 | moves, IMP overlap drops |
| 3e-6 | 5 | 0.8845 | 0.8993 | 0.1381 | 0.0918 | 0.1441 | 0.5928 | 0.1441 | 0.8593 | moves strongly, sample quality drops |

Interpretation:

This is the long-budget counterpart to the short CIFAR movement diagnostic.
Epoch-1 rewinding gives a strong ticket (`0.8993` IMP accuracy vs. `0.8845`
dense accuracy), so the failure is not due to a weak sparse model. Increasing
SGLD step size moves supports away from the dense chain start, but it does not
move them toward the IMP ticket. At `1e-6`, post-chain overlap drops to
`0.7362` while posterior-to-IMP overlap drops from `0.1441` to `0.1425`; at
`3e-6`, post-chain drops to `0.5928` while posterior-to-IMP drops to `0.1381`
and sample accuracy falls to `0.8593`. Rewind magnitude remains closer to IMP
at `0.1784` than any posterior support, reinforcing the trajectory account.

## 2026-05-03: Tuned 5-Seed Full-Batch HMC Digits Baseline

Code changes:

- `scripts/run_digits_hmc_baseline.py` now records HMC sample accuracy,
  sample-to-dense and sample-to-IMP prediction agreement, function clustering,
  and dense magnitude as the HMC chain-start support control.
- Added `scripts/summarize_hmc_runs.py` to aggregate HMC runs into the same
  summary schema used by Gate1 and the paper-facing result tables.
- `scripts/evaluate_gate1.py` now treats dense-IMP connectivity as optional,
  so HMC summaries can be evaluated on the support Gate1 checks without a
  separate linear-connectivity run.
- `scripts/build_results_table.py` now treats SNIP and SynFlow as optional for
  posterior-only baselines such as HMC.

Commands:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_hmc_baseline.py --seed "$seed" --hidden-dim 8 --depth 2 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --hmc-steps 160 --hmc-step-size 1e-2 --hmc-leapfrog-steps 4 --hmc-burn-in 60 --hmc-sample-every 10 --random-trials 100 --out-dir runs/digits_hmc_long_eps1e-2_r2_p0p3; done
.venv/bin/python scripts/summarize_hmc_runs.py --run-root runs/digits_hmc_long_eps1e-2_r2_p0p3 --out-csv runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.csv --out-json runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json
.venv/bin/python scripts/evaluate_gate1.py runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json --out-json runs/digits_hmc_long_eps1e-2_r2_p0p3_gate1_eval.json
.venv/bin/python scripts/build_sparsity_sweep_table.py --summary runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json --out-csv runs/digits_hmc_long_eps1e-2_r2_p0p3_table.csv --out-md docs/digits_hmc_long_eps1e-2_r2_p0p3.md
```

Output:

- `runs/digits_hmc_long_eps1e-2_r2_p0p3_summary.json`
- `runs/digits_hmc_long_eps1e-2_r2_p0p3_gate1_eval.json`
- `docs/digits_hmc_long_eps1e-2_r2_p0p3.md`

Result:

| Config | Runs | Dense Acc | IMP Acc | Sparsity | HMC Posterior | Random | Chain Start | Post-Chain | Dense Mag | HMC Acc | Accept | State Clusters | Function Clusters | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| digits r2 p0.30 HMC | 5 | 0.7544 | 0.7739 | 0.5101 | 0.4136 | 0.3254 | 0.8736 | 0.4095 | 0.8736 | 0.9201 | 0.9500 | 3.0 | 2.4 | fail |

Interpretation:

This is stronger than the earlier conservative HMC smoke. Full-batch HMC moves
substantially away from the dense chain start while maintaining high sample
accuracy. That removes the objection that HMC failed only because it was stuck
at the chain start. The support result still fails Gate1: HMC masks beat random,
but they are far below the dense/chain-start magnitude control. This reinforces
the negative claim in a high-fidelity small-model setting; the remaining
posterior-baseline gap is CIFAR-scale high-fidelity sampling, not small-model
HMC.

## 2026-05-03: CIFAR-10 ResNet-20 Short SGLD Movement Diagnostics

Code change:

- Added `scripts/run_sgld_movement_grid.py`, which trains dense and IMP once
  and then evaluates multiple SGLD step sizes from the same dense checkpoint.
  This avoids retraining dense/IMP for every posterior movement setting.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,3e-10,1e-9,3e-9,1e-8,3e-8 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_r5_p0p3
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-7,3e-7,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_highlr_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-lrs 1e-10,1e-6,3e-6 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_sgld_movement_short_r5_p0p3/20260503_184103/metrics.json`
- `runs/cifar10_resnet20_sgld_movement_short_highlr_r5_p0p3/20260503_184406/metrics.json`
- `docs/cifar10_resnet20_sgld_movement_short_r5_p0p3.md`
- `runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3.md`

Five-seed selected result:

| SGLD LR | Posterior | Random | Chain Start | Post-Chain | Sample Acc | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1730 | 0.0918 | 0.1729 | 0.9963 | 0.8225 | matches chain start |
| 1e-6 | 0.1683 | 0.0917 | 0.1729 | 0.6932 | 0.8169 | moves, IMP overlap drops |
| 3e-6 | 0.1602 | 0.0917 | 0.1729 | 0.5440 | 0.7954 | moves strongly, sample quality drops |

One-seed tuning grid:

| SGLD LR | Posterior | Random | Chain Start | Post-Chain | Sample Acc | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1701 | 0.0917 | 0.1701 | 0.9965 | 0.8243 | matches chain start |
| 3e-8 | 0.1701 | 0.0916 | 0.1701 | 0.9383 | 0.8241 | moves slightly, no IMP gain |
| 1e-7 | 0.1670 | 0.0918 | 0.1679 | 0.8880 | 0.8253 | moves, IMP overlap drops |
| 3e-7 | 0.1659 | 0.0918 | 0.1679 | 0.8150 | 0.8239 | moves, IMP overlap drops |
| 1e-6 | 0.1619 | 0.0918 | 0.1679 | 0.6949 | 0.8188 | moves strongly, IMP overlap drops |
| 3e-6 | 0.1547 | 0.0918 | 0.1679 | 0.5437 | 0.7975 | moves strongly, sample quality drops |

Interpretation:

The movement objection now has a repeated-seed CIFAR-scale diagnostic.
Increasing SGLD step size moves support away from the dense chain start while
preserving usable sample accuracy at `1e-6`. However, movement does not increase
posterior-to-IMP support overlap; it decreases it from `0.1730` at `1e-10` to
`0.1683` at `1e-6` and `0.1602` at `3e-6`. This matches the tuned HMC result
and the MNIST movement sweep: posterior support can move, but movement is not
ticket-directed.

## 2026-05-03: CIFAR-10 ResNet-20 Short SGHMC Movement Diagnostic

Code changes:

- Added `src/lottery/sghmc.py`, a momentum-based SGHMC sampler with dataset
  likelihood scaling, prior precision, temperature, burn-in, and sampling
  interval controls.
- `scripts/run_digits_pilot.py`, `scripts/run_gate1_sweep.py`, and
  `scripts/run_gate1_sparsity_sweep.py` now accept
  `--posterior-sampler sghmc`.
- `scripts/run_sgld_movement_grid.py` now accepts `--posterior-sampler sghmc`
  so dense/IMP training can be reused across SGHMC LR settings.
- `scripts/summarize_digits_runs.py` now handles optional `None` controls.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sghmc_movement_short_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 10 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_sghmc_movement_short_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_sghmc_movement_short_r5_p0p3.md`
- `runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3.md`

Five-seed selected result:

| SGHMC LR | Posterior | Random | Chain Start | Post-Chain | Sample Acc | State Clusters | Function Clusters | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1702 | 0.0917 | 0.1701 | 0.9848 | 0.8204 | 6.0 | 2.0 | near chain start |
| 3e-8 | 0.1682 | 0.0917 | 0.1701 | 0.7733 | 0.8178 | 6.0 | 2.8 | moves, IMP overlap drops |
| 1e-7 | 0.1637 | 0.0917 | 0.1701 | 0.6329 | 0.8107 | 6.0 | 1.8 | moves strongly, IMP overlap drops |

Interpretation:

SGHMC gives a momentum-based posterior dynamics check beyond SGLD and SWAG.
It moves support farther from the dense chain start than the short SGLD
low-step setting and records more parameter-space clusters (`6.0`), but the
support still moves away from IMP rather than toward it. At `3e-8`, sample
accuracy remains close to dense (`0.8178` vs. `0.8183`) while post-chain falls
to `0.7733`; posterior-to-IMP overlap drops below chain-start magnitude
(`0.1682` vs. `0.1701`). At `1e-7`, movement is stronger but posterior-to-IMP
drops further to `0.1637`. This reduces the concern that the negative CIFAR
short result is specific to overdamped SGLD.

## 2026-05-03: CIFAR-10 ResNet-20 Long Epoch-1 Rewind SGHMC Movement Diagnostic

Purpose:

Match the strongest CIFAR epoch-1 rewind row with a momentum posterior sampler.
The diagnostic tests whether SGHMC movement away from the dense chain start
becomes ticket-directed in the long-budget setting where IMP exceeds dense
accuracy.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7,3e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sghmc_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler sghmc --sgld-lrs 1e-10,3e-8,1e-7,3e-7 --sghmc-momentum-decay 0.9 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_sghmc_movement_tune_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3.md`

Five-seed selected result:

| SGHMC LR | Posterior | Random | Chain Start | Post-Chain | Rewind Mag | Sample Acc | State Clusters | Function Clusters | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1456 | 0.0918 | 0.1457 | 0.9876 | 0.1777 | 0.8881 | 6.0 | 1.6 | near chain start |
| 3e-8 | 0.1445 | 0.0917 | 0.1457 | 0.8060 | 0.1777 | 0.8839 | 6.0 | 2.8 | moves, IMP overlap drops |
| 1e-7 | 0.1419 | 0.0918 | 0.1457 | 0.6796 | 0.1777 | 0.8752 | 6.0 | 3.2 | moves strongly, IMP overlap drops |
| 3e-7 | 0.1360 | 0.0918 | 0.1457 | 0.5214 | 0.1777 | 0.8537 | 6.0 | 1.8 | strongest movement, IMP overlap drops further |

Interpretation:

SGHMC closes the corresponding momentum-sampler objection in the long
epoch-1 rewind setting. Dense accuracy is `0.8883` and IMP accuracy is
`0.8970`. Raising SGHMC LR moves the support away from the dense chain start
while keeping sample accuracy usable at `3e-8` and `1e-7`, but posterior-to-IMP
overlap decreases instead of improving. At `1e-7`, post-chain falls to `0.6796`
and sample accuracy remains `0.8752`, yet posterior-to-IMP is only `0.1419`
versus chain-start magnitude `0.1457` and rewind magnitude `0.1777`. At
`3e-7`, support movement is stronger (`post-chain = 0.5214`) but posterior
alignment falls to `0.1360`. This is a stronger negative row than the short
SGHMC diagnostic because it uses the epoch-1 rewind long-budget IMP setting.

## 2026-05-03: CIFAR-10 ResNet-20 Long Epoch-1 Rewind Cyclical SGLD Movement Diagnostic

Purpose:

Add a cyclical SGLD posterior sampler to test whether repeated high-LR
exploration followed by low-LR sampling can discover IMP-like supports in the
strongest CIFAR epoch-1 rewind setting.

Code changes:

- Added `src/lottery/cyclical_sgld.py`, with cosine cyclic SGLD learning rates,
  dataset likelihood scaling, configurable cycle length, LR floor, and optional
  low-phase sample filtering.
- `scripts/run_sgld_movement_grid.py`, `scripts/run_digits_pilot.py`,
  `scripts/run_gate1_sweep.py`, and `scripts/run_gate1_sparsity_sweep.py` now
  accept `--posterior-sampler cyclical-sgld`.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler cyclical-sgld --sgld-lrs 1e-10,1e-6,3e-6,1e-5 --csgld-lr-min-ratio 0.01 --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --sgld-steps 400 --sgld-burn-in 100 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_csgld_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler cyclical-sgld --sgld-lrs 1e-10,1e-6,3e-6,1e-5 --csgld-lr-min-ratio 0.01 --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --sgld-steps 400 --sgld-burn-in 100 --sgld-sample-every 10 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3.md
```

Five-seed selected result:

| cSGLD max LR | Posterior | Random | Chain Start | Post-Chain | Rewind Mag | Sample Acc | State Clusters | Function Clusters | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1454 | 0.0918 | 0.1454 | 0.9963 | 0.1789 | 0.8855 | 3.0 | 2.0 | near chain start |
| 1e-6 | 0.1422 | 0.0918 | 0.1454 | 0.7046 | 0.1789 | 0.8782 | 3.0 | 1.2 | moves, IMP overlap drops |
| 3e-6 | 0.1371 | 0.0918 | 0.1454 | 0.5533 | 0.1789 | 0.8686 | 3.0 | 1.0 | moves strongly, IMP overlap drops |
| 1e-5 | 0.1260 | 0.0918 | 0.1454 | 0.3700 | 0.1789 | 0.8434 | 2.8 | 1.0 | strongest movement, sample accuracy degrades |

Interpretation:

Cyclical SGLD gives a stronger exploration baseline than fixed-step SGLD in
the sense that support movement can be large while samples remain usable at
`1e-6` and `3e-6`. It still fails the posterior-mode rescue condition. At
`1e-6`, post-chain falls to `0.7046` and sample accuracy remains `0.8782`, but
posterior-to-IMP falls to `0.1422`, below chain-start magnitude `0.1454` and
well below epoch-1 rewind magnitude `0.1789`. Stronger cycling moves farther
from the chain start but decreases IMP alignment.

## 2026-05-03: CIFAR-10 ResNet-20 Long Epoch-1 Rewind Diagonal Laplace Movement Diagnostic

Purpose:

Add a local Gaussian posterior approximation around the dense checkpoint to
test whether a curvature-weighted sampler changes the long-budget CIFAR
movement conclusion. This is a mini-batch diagonal empirical-Fisher Laplace
approximation, not an exact Hessian, KFAC, or full-covariance Laplace baseline.

Code changes:

- Added `src/lottery/diag_laplace.py`, with mini-batch diagonal Fisher
  estimation and diagonal Gaussian sampling around the chain-start state.
- `scripts/run_sgld_movement_grid.py` now accepts
  `--posterior-sampler diag-laplace`, `--laplace-scales`,
  `--laplace-prior-precision`, `--laplace-fisher-batches`, and
  `--laplace-variance-floor`.
- `scripts/summarize_sgld_movement_grid.py` now labels the shared first column
  as `Sampler LR/Scale`, because Laplace uses the compatibility field as a
  posterior scale rather than an SGLD learning rate.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 1e-10,1e-8,1e-6,3e-6,1e-5,3e-5,1e-4 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_tune_r5_p0p3
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 3e-4,1e-3,3e-3,1e-2,3e-2 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_highscale_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler diag-laplace --laplace-scales 1e-10,1e-3,3e-3,1e-2 --laplace-prior-precision 1e-2 --laplace-fisher-batches 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_tune_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_highscale_tune_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3.md`

Five-seed selected result:

| Laplace scale | Posterior | Random | Chain Start | Post-Chain | Rewind Mag | Sample Acc | State Clusters | Function Clusters | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1469 | 0.0918 | 0.1469 | 0.9999 | 0.1787 | 0.8849 | 1.0 | 1.0 | near chain start |
| 1e-3 | 0.1447 | 0.0918 | 0.1469 | 0.8826 | 0.1787 | 0.8799 | 1.0 | 1.0 | moves, IMP overlap drops |
| 3e-3 | 0.1400 | 0.0918 | 0.1469 | 0.7803 | 0.1787 | 0.8707 | 1.0 | 1.0 | moves strongly, IMP overlap drops |
| 1e-2 | 0.1278 | 0.0918 | 0.1469 | 0.5961 | 0.1787 | 0.8274 | 1.0 | 1.0 | strongest movement, sample accuracy degrades |

Interpretation:

The diagonal Laplace approximation gives the same support-level answer as SGLD,
SGHMC, SWAG, and cyclical SGLD. Dense accuracy is `0.8849` and IMP accuracy is
`0.8980`. At scale `1e-3`, support moves away from the chain start
(`post-chain = 0.8826`) while sample accuracy remains usable (`0.8799`), but
posterior-to-IMP falls from `0.1469` to `0.1447`. At scale `3e-3`, movement is
larger (`post-chain = 0.7803`) and sample accuracy remains `0.8707`, but
posterior-to-IMP falls to `0.1400`. At scale `1e-2`, post-chain falls to
`0.5961` and posterior-to-IMP drops further to `0.1278`. Thus, a
curvature-weighted local Gaussian approximation does not make posterior support
ticket-directed; epoch-1 rewind magnitude remains closer to IMP at `0.1787`.

## 2026-05-03: CIFAR-10 ResNet-20 Long Epoch-1 Rewind KFAC-Style Laplace Movement Diagnostic

Purpose:

Add a Kronecker-factored curvature approximation around the dense checkpoint.
This is still an approximate local Gaussian posterior: it uses mini-batch
empirical-Fisher KFAC factors for Linear and Conv2d weights, not exact Hessian
sampling or full-covariance Laplace. It is nevertheless a stronger CIFAR-scale
curvature baseline than diagonal Laplace.

Code changes:

- Added `src/lottery/kfac_laplace.py`, with Linear/Conv2d KFAC factor
  estimation, factor row subsampling, damping, and reusable factor sampling
  across scale grids.
- `scripts/run_sgld_movement_grid.py` now accepts
  `--posterior-sampler kfac-laplace`, `--kfac-laplace-scales`,
  `--kfac-laplace-prior-precision`, `--kfac-laplace-fisher-batches`,
  `--kfac-laplace-damping`, and `--kfac-laplace-factor-rows`.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler kfac-laplace --kfac-laplace-scales 1e-10,1e-8,1e-6,1e-4,1e-2 --kfac-laplace-prior-precision 1e-2 --kfac-laplace-fisher-batches 5 --kfac-laplace-damping 1e-2 --kfac-laplace-factor-rows 2048 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_tune_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler kfac-laplace --kfac-laplace-scales 1e-10,1e-4,1e-3,1e-2 --kfac-laplace-prior-precision 1e-2 --kfac-laplace-fisher-batches 10 --kfac-laplace-damping 1e-2 --kfac-laplace-factor-rows 4096 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_tune_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3.md`

Five-seed selected result:

| KFAC scale | Posterior | Random | Chain Start | Post-Chain | Rewind Mag | Sample Acc | State Clusters | Function Clusters | Gate1 Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.1456 | 0.0917 | 0.1456 | 0.9999 | 0.1775 | 0.8861 | 1.0 | 1.0 | near chain start |
| 1e-4 | 0.1456 | 0.0917 | 0.1456 | 0.9334 | 0.1775 | 0.8854 | 1.0 | 1.0 | weak movement, no IMP gain |
| 1e-3 | 0.1441 | 0.0917 | 0.1456 | 0.8016 | 0.1775 | 0.8839 | 1.0 | 1.0 | moves, IMP overlap drops |
| 1e-2 | 0.1303 | 0.0917 | 0.1456 | 0.4859 | 0.1775 | 0.8695 | 1.0 | 1.0 | strong movement, IMP overlap drops |

Interpretation:

The KFAC-style Laplace approximation strengthens the curvature-baseline story
and again fails the posterior-mode rescue condition. Dense accuracy is
`0.8860` and IMP accuracy is `0.8956`. At scale `1e-4`, post-chain falls to
`0.9334` with sample accuracy `0.8854`, but posterior-to-IMP is unchanged
within noise (`0.1456`) and does not beat chain-start magnitude (`0.1456`). At
scale `1e-3`, post-chain falls to `0.8016` with sample accuracy `0.8839`, but
posterior-to-IMP falls to `0.1441`. At scale `1e-2`, support moves strongly
(`post-chain = 0.4859`) and sample accuracy remains usable but lower
(`0.8695`); posterior-to-IMP drops to `0.1303`. The epoch-1 rewind magnitude
control remains much closer to IMP at `0.1775`.

## 2026-05-03: Paper Figure Artifact Generation

Purpose:

Convert the strongest current summary tables into reviewer-facing paper
figures.

Code changes:

- Added `scripts/build_paper_figures.py`.

Command:

```bash
.venv/bin/python scripts/build_paper_figures.py
```

Output:

- `paper/figures/gate1_controls.pdf`
- `paper/figures/gate1_controls.png`
- `paper/figures/cifar_movement.pdf`
- `paper/figures/cifar_movement.png`

Interpretation:

The Gate1 figure shows that MNIST/Fashion-MNIST posterior masks beat random
but track chain-start magnitude and stay far below dense magnitude. The CIFAR
movement figure shows that SGLD, SGHMC, cyclical SGLD, diagonal Laplace, and
KFAC-style Laplace all move away from chain-start support without increasing
posterior-to-IMP overlap.

## 2026-05-03: Paper Statistical Audit Artifact Generation

Purpose:

Add reviewer-facing paired gap summaries for the main claims.

Code changes:

- Added `scripts/build_paper_stats.py`.

Command:

```bash
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `docs/paper_stats.md`
- `paper/tables/statistical_summary.tex`
- `runs/paper_stats.json`

Interpretation:

MNIST and Fashion-MNIST posterior masks beat random in all 20 seed/config
cells for each dataset, while posterior-minus-chain-start stays essentially
zero and dense-minus-posterior is positive in all cells. CIFAR movement
diagnostics show negative posterior-minus-chain-start gaps and positive
rewind-minus-posterior gaps once posterior support moves meaningfully away from
the chain start.

## 2026-05-04: CIFAR Exact Final-Head Full-Covariance Laplace Probe

Purpose:

Add a limited but exact full-covariance Laplace posterior check at CIFAR scale.
The probe freezes the ResNet-20 feature extractor, computes the exact
softmax-cross-entropy Hessian for the 650-parameter final linear classifier
head, samples the corresponding full-covariance Gaussian, and compares
head-level supports to the IMP head mask.

Code changes:

- Added `src/lottery/head_laplace.py`.
- Added `scripts/run_head_laplace_probe.py`.
- Added `scripts/summarize_head_laplace_probe.py`.
- Extended `scripts/build_paper_stats.py` with a head-Laplace section.

Commands:

```bash
.venv/bin/python scripts/run_head_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --head-laplace-scales 1e-4,1e-3 --head-laplace-hessian-batches 2 --samples 3 --random-trials 10 --out-dir runs/fake_cifar10_head_laplace_smoke
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_head_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --head-laplace-scales 1e-6,1e-3,1e-2,1e0 --head-laplace-prior-precision 1e-2 --head-laplace-damping 1e-5 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_head_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/fake_cifar10_head_laplace_smoke/20260503_235109`
- `runs/cifar10_resnet20_long30_rewind1_head_laplace_tune_r5_p0p3/20260503_235808`
- `runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3.md`

Result:

Dense accuracy is `0.8856` and IMP accuracy is `0.8957`. Head sparsity is
`0.1947`. At scale `1e-3`, head posterior-to-chain-start support overlap falls
to `0.7773` while sample accuracy remains `0.8856`, but head
posterior-minus-chain-start is negative on average (`-0.0085`). At scale
`1e-2`, the head support moves further (`post-chain = 0.6912`) and the
posterior-minus-chain-start CI is fully negative (`-0.0284`
`[-0.0510, -0.0058]`). At scale `1`, head post-chain is `0.6769`, sample
accuracy remains usable at `0.8835`, and the posterior-minus-chain-start CI is
again fully negative (`-0.0352` `[-0.0582, -0.0122]`). The head rewind
magnitude support remains closer to IMP at `0.7191`.

Interpretation:

This is not a full-network posterior baseline, but it is an exact
full-covariance local Gaussian over the CIFAR final head. Within that
well-defined subproblem, posterior support movement is again not
ticket-directed.

## 2026-05-04: CIFAR Matched Dense-Trajectory Support Probe

Purpose:

Strengthen the trajectory/magnitude-subspace explanation by measuring dense
training trajectory supports directly. Unlike earlier controls that trained a
separate one-epoch rewind clone, this probe uses a single dense trajectory,
takes its epoch-1 checkpoint as the IMP rewind state, and compares magnitude
supports from epochs 0, 1, 2, 5, 10, 20, and 30 to the final IMP mask.

Code changes:

- Added `scripts/run_trajectory_probe.py`.
- Added `scripts/summarize_trajectory_probe.py`.
- Extended `scripts/build_paper_figures.py` with `cifar_trajectory.pdf`.
- Extended `scripts/build_paper_stats.py` with a dense-trajectory section.

Commands:

```bash
.venv/bin/python scripts/run_trajectory_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --out-dir runs/fake_cifar10_trajectory_probe_smoke
.venv/bin/python scripts/summarize_trajectory_probe.py --run-root runs/fake_cifar10_trajectory_probe_smoke --out-csv runs/fake_cifar10_trajectory_probe_smoke_summary.csv --out-md docs/fake_cifar10_trajectory_probe_smoke.md
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_trajectory_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_trajectory_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
.venv/bin/python scripts/build_paper_figures.py
```

Output:

- `runs/fake_cifar10_trajectory_probe_smoke/20260504_003944`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3.md`
- `paper/figures/cifar_trajectory.pdf`
- `paper/figures/cifar_trajectory.png`

Result:

Dense accuracy is `0.8857` and IMP accuracy is `0.8943`. The epoch-1 rewind
support has Jaccard `0.1789` to the final IMP mask. The matched dense
trajectory becomes substantially closer: epoch 5 reaches `0.2206`, epoch 10
peaks at `0.2347` with 95% CI `[0.2245, 0.2449]`, epoch 20 is `0.2326`, and
the final dense support is `0.2313`. These values are well above the global
posterior supports in the CIFAR movement diagnostics, which peak around
`0.147`.

Interpretation:

This is currently the strongest direct support for the negative paper's
alternative explanation: IMP tickets are tied to the training/rewinding
trajectory and its magnitude subspace, not to local posterior supports around
the dense endpoint.

## 2026-05-04: CIFAR Aggregate and Layerwise Trajectory Controls

Purpose:

Refine the matched dense-trajectory result by testing whether the ticket
support is explained by trajectory movement, trajectory path length, or the
magnitude subspace that persists along the trajectory.

Code changes:

- Extended `scripts/run_trajectory_probe.py` with aggregate trajectory score
  masks: mean absolute weight, RMS absolute weight, max absolute weight,
  movement from initialization, movement from rewind, full path length, and
  post-rewind path length.
- Added stage-level and per-parameter support overlap rows to the trajectory
  probe output.
- Extended `scripts/summarize_trajectory_probe.py` to write aggregate, group,
  and layer summary CSVs.
- Updated `scripts/build_paper_stats.py` and `scripts/build_paper_figures.py`
  to use the aggregate trajectory result.

Commands:

```bash
.venv/bin/python scripts/run_trajectory_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --out-dir runs/fake_cifar10_trajectory_probe_smoke_v2
.venv/bin/python scripts/summarize_trajectory_probe.py --run-root runs/fake_cifar10_trajectory_probe_smoke_v2 --out-csv runs/fake_cifar10_trajectory_probe_smoke_v2_summary.csv --out-md docs/fake_cifar10_trajectory_probe_smoke_v2.md
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_trajectory_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2
done
```

```bash
.venv/bin/python scripts/summarize_trajectory_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_summary.csv`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_aggregate_summary.csv`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_group_summary.csv`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_layer_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2.md`

Result:

Dense accuracy is `0.8841` and IMP accuracy is `0.8963`. The best checkpoint
magnitude support is epoch 10 with Jaccard `0.2342` to IMP. The final dense
support is `0.2312`, and the epoch-1 rewind support is `0.1782`. The best
aggregate trajectory score is RMS absolute magnitude with Jaccard `0.2400`
and 95% CI `[0.2282, 0.2519]`; mean absolute magnitude is close at `0.2388`.
Movement-only scores are much weaker: initialization RMS movement is `0.1756`,
rewind RMS movement is `0.1607`, full path length is `0.1765`, and post-rewind
path length is `0.1679`.

Stage-level summary for the best aggregate RMS absolute score shows high
overlap in the head (`0.8128`) and stem (`0.7223`), moderate overlap in
stage1 (`0.3819`) and stage2 (`0.2758`), and lower overlap in stage3
(`0.1719`). This mirrors the global-support result: the aggregate support is
not just moving weights, but a persistent trajectory magnitude subspace.

Interpretation:

The stronger claim should be phrased as trajectory-magnitude support rather
than trajectory movement alone. The ticket support aligns with weights that
remain large along the matched training trajectory, and this remains far above
all current global posterior-induced supports, which peak around `0.147`.

## 2026-05-04: CIFAR Trajectory Mask Training Functional Probe

Purpose:

Test whether trajectory-derived supports are only geometrically closer to IMP,
or whether they also reproduce IMP after fixed-mask retraining from the same
epoch-1 rewind state.

Code changes:

- Added `scripts/run_trajectory_mask_training_probe.py`.
- Added `scripts/summarize_trajectory_mask_training_probe.py`.
- Extended `scripts/build_paper_stats.py` with a trajectory mask retraining
  section and a LaTeX table for the paper draft.

Smoke commands:

```bash
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --mask-sources imp,random,epoch_1,epoch_2,traj_rms_abs,traj_path_length --random-trials 1 --out-dir runs/fake_cifar10_trajectory_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/fake_cifar10_trajectory_mask_training_smoke --out-csv runs/fake_cifar10_trajectory_mask_training_smoke_summary.csv --out-md docs/fake_cifar10_trajectory_mask_training_smoke.md
```

Full CIFAR commands:

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-sources imp,random,epoch_1,epoch_10,epoch_30,traj_rms_abs,traj_mean_abs,traj_path_length,traj_rewind_rms_movement --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

All fixed masks are trained for 30 epochs from the same epoch-1 rewind state at
the IMP sparsity. The IMP mask retrain reaches `0.8983` accuracy. The best
non-IMP mask is final dense magnitude at `0.8826`, with Acc-IMP `-0.0151`.
RMS and mean trajectory-magnitude masks reach `0.8739` and `0.8743`, epoch 10
reaches `0.8730`, path length and rewind movement reach `0.8549` and
`0.8572`, epoch 1 reaches `0.8540`, and random reaches `0.8422`.

Interpretation:

Trajectory-magnitude masks are not just overlap artifacts: they are trainable
and clearly stronger than random or pure movement/path masks. They still do not
recover the IMP advantage. The paper should state the positive mechanism as
IMP refining a persistent dense trajectory magnitude subspace, not as trajectory
magnitude alone fully explaining winning tickets.

## 2026-05-04: CIFAR Trajectory Residual Swap Probe

Purpose:

Test what IMP adds beyond trajectory-magnitude masks by swapping base-only
support for IMP-only residual support, with same-size non-IMP random residual
swaps as a control.

Code changes:

- Added `scripts/run_trajectory_residual_probe.py`.
- Added `scripts/summarize_trajectory_residual_probe.py`.
- Extended `scripts/build_paper_stats.py` with residual-swap statistics and a
  LaTeX table for the paper draft.

Commands:

```bash
.venv/bin/python scripts/run_trajectory_residual_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0,0.5,1 --random-residual-trials 1 --out-dir runs/fake_cifar10_trajectory_residual_smoke
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/fake_cifar10_trajectory_residual_smoke --out-csv runs/fake_cifar10_trajectory_residual_smoke_summary.csv --out-md docs/fake_cifar10_trajectory_residual_smoke.md
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_trajectory_residual_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0,0.5,1.0 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

All masks are retrained for 30 epochs from the same epoch-1 rewind state. The
final dense mask starts at `0.8797`; swapping half of its base-only support for
IMP-only support raises accuracy to `0.8882`, while the same-size non-IMP
random residual reaches only `0.8780`. The RMS trajectory mask starts at
`0.8733`; half IMP residual raises it to `0.8851`, while random residual is
`0.8705`. The epoch-10 mask starts at `0.8712`; half IMP residual raises it to
`0.8855`, while random residual is `0.8704`. The full IMP endpoint retrains to
`0.8966`, close to the corresponding IMP run accuracy `0.8972`.

Interpretation:

This is the strongest current causal evidence for the refined trajectory
account. Trajectory-magnitude masks define useful base supports, but the
IMP-only residual support carries real functional information. Random support
replacement at the same distance does not recover the gap and often damages
accuracy. The next step is to characterize the structure of that IMP residual
support by layer/stage, pruning round, and weight trajectory statistics.

## 2026-05-04: CIFAR Residual Anatomy Probe

Purpose:

Characterize the IMP-only residual support exposed by the residual-swap probe:
where it appears by stage/layer, when the competing base-only weights are
removed during IMP, and how well dense-trajectory statistics can predict the
missing IMP-only weights.

Code changes:

- Added `scripts/run_residual_anatomy_probe.py`.
- Added `scripts/summarize_residual_anatomy_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper statistics table with
  residual-anatomy statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_anatomy_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --base-sources epoch_2,traj_rms_abs --predictor-steps 5 --predictor-batch-size 2048 --out-dir runs/fake_cifar10_residual_anatomy_smoke
.venv/bin/python scripts/summarize_residual_anatomy_probe.py --run-root runs/fake_cifar10_residual_anatomy_smoke --out-prefix runs/fake_cifar10_residual_anatomy_smoke_summary --out-md docs/fake_cifar10_residual_anatomy_smoke.md
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_residual_anatomy_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --predictor-steps 120 --predictor-batch-size 16384 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_residual_anatomy_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3 --out-prefix runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary --out-md docs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_global.csv`
- `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_group.csv`
- `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_round.csv`
- `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_score.csv`
- `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_predictor.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

For the final dense, RMS trajectory, and epoch-10 bases, support-to-IMP
Jaccard remains low at `0.2323`, `0.2412`, and `0.2359`; each base misses
about `27.8k--28.4k` IMP-kept weights. Base-only weights are pruned throughout
IMP, not only at the final round: mean pruning round is `2.90--2.97`.

The IMP-only residual is only mildly stage-structured. For the RMS trajectory
base, stage 2 is enriched at `1.1348x`, while stage 3 contains `0.7442` of the
IMP-only residual but is near its size share (`0.9844x` enrichment). A held-out
logistic predictor using dense-trajectory rank features plus stage indicators
only weakly predicts the IMP-only residual: AUC is `0.6165--0.6206`, top-k
recall is `0.2087--0.2206`, versus a baseline positive rate around
`0.123--0.126`.

Interpretation:

Dense trajectory statistics explain why trajectory masks are useful, but they
do not reconstruct the IMP-only residual. Together with the residual-swap
probe, this suggests that the remaining IMP advantage is process-specific:
IMP refines the dense trajectory magnitude subspace in a way that is only
weakly visible from the dense trajectory alone.

## 2026-05-04: CIFAR Residual Predictor Mask Probe

Purpose:

Test whether the weak held-out residual predictor from the residual-anatomy
probe can generate a functional mask. For each base, the probe trains a
logistic residual predictor on non-base candidate weights, holds out a separate
candidate set, replaces half of the base-only support with the highest-scoring
held-out candidates, and retrains the resulting mask from the same epoch-1
rewind state.

Code changes:

- Added `scripts/run_residual_predictor_mask_probe.py`.
- Added `scripts/summarize_residual_predictor_mask_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper statistics table with
  functional residual-predictor mask statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_predictor_mask_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --predictor-steps 5 --predictor-batch-size 2048 --random-residual-trials 1 --out-dir runs/fake_cifar10_residual_predictor_mask_smoke
.venv/bin/python scripts/summarize_residual_predictor_mask_probe.py --run-root runs/fake_cifar10_residual_predictor_mask_smoke --out-csv runs/fake_cifar10_residual_predictor_mask_smoke_summary.csv --out-md docs/fake_cifar10_residual_predictor_mask_smoke.md
```

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_residual_predictor_mask_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --predictor-steps 120 --predictor-batch-size 16384 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3
done
```

```bash
.venv/bin/python scripts/summarize_residual_predictor_mask_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

The predictor consistently raises held-out added IMP-only precision above the
random residual control: `0.1866` vs. `0.1253` for the final dense base,
`0.1834` vs. `0.1233` for the RMS trajectory base, and `0.1843` vs. `0.1239`
for the epoch-10 base. However, this added precision does not recover the
oracle residual accuracy gain. For final dense, the predictor mask retrains to
`0.8793`, random residual to `0.8805`, and oracle residual to `0.8892`. For RMS
trajectory, predictor and random both retrain to about `0.8744`, while oracle
reaches `0.8866`. For epoch 10, predictor and random reach `0.8728` and
`0.8724`, while oracle reaches `0.8890`.

Interpretation:

The residual-anatomy predictor captures some marginal coordinate-level signal,
but that signal is not enough to reconstruct a functional residual subnetwork.
Together with the residual-swap probe, this sharpens the mechanism: IMP-only
residual support is functionally specific, and its value is not explained by
dense-trajectory rank features plus stage indicators alone.

## 2026-05-04 - CIFAR Residual Cross-Seed Transfer Probe

Purpose:

Test whether the residual predictor's coordinate-level signal transfers across
seeds. For each held-out target seed, the probe trains a logistic residual
predictor on non-base candidate weights from the other four seeds, applies it
to target-seed candidates, replaces half of target base-only support with the
highest-scoring target non-base weights, and retrains from the target epoch-1
rewind state.

Code changes:

- Added `scripts/run_residual_cross_seed_transfer_probe.py`.
- Added `scripts/summarize_residual_cross_seed_transfer_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper statistics table with
  cross-seed residual-transfer statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_cross_seed_transfer_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --predictor-steps 5 --predictor-batch-size 2048 --random-residual-trials 1 --out-dir runs/fake_cifar10_residual_cross_seed_transfer_smoke
.venv/bin/python scripts/summarize_residual_cross_seed_transfer_probe.py --run-root runs/fake_cifar10_residual_cross_seed_transfer_smoke --out-csv runs/fake_cifar10_residual_cross_seed_transfer_smoke_summary.csv --out-md docs/fake_cifar10_residual_cross_seed_transfer_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_cross_seed_transfer_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --predictor-steps 120 --predictor-batch-size 16384 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_cross_seed_transfer_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

The cross-seed predictor consistently transfers coordinate-level signal:
added IMP-only precision rises from random levels of `0.1246--0.1264` to
`0.2238--0.2413`. However, this transfer does not reconstruct the functional
oracle residual. For final dense, cross-seed residual retrains to `0.8776`,
random residual to `0.8781`, and oracle residual to `0.8905`. For RMS
trajectory, cross-seed and random reach `0.8731` and `0.8725`, while oracle
reaches `0.8878`. For epoch 10, cross-seed and random reach `0.8745` and
`0.8726`, while oracle reaches `0.8890`.

Interpretation:

Residual coordinate signal is real and seed-transferable, but it is not enough
to instantiate the functional residual subnetwork. This strengthens the current
mechanism: IMP refines a dense-trajectory magnitude subspace using
process-specific, combinatorial residual support rather than an independent
coordinate-ranking rule.

## 2026-05-04 - CIFAR Residual Stratified Control Probe

Purpose:

Test whether the oracle IMP residual gain is explained by coarse residual
structure rather than exact IMP coordinates. The probe removes the same
low-base-score weights as the oracle and compares oracle top-IMP additions
against random IMP-only additions, global non-IMP additions, parameter-matched
non-IMP additions, and parameter-plus-score-decile-matched non-IMP additions.

Code changes:

- Added `scripts/run_residual_stratified_control_probe.py`.
- Added `scripts/summarize_residual_stratified_control_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper statistics table with
  residual stratified-control statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_stratified_control_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --random-trials 1 --score-bins 5 --out-dir runs/fake_cifar10_residual_stratified_control_smoke
.venv/bin/python scripts/summarize_residual_stratified_control_probe.py --run-root runs/fake_cifar10_residual_stratified_control_smoke --out-csv runs/fake_cifar10_residual_stratified_control_smoke_summary.csv --out-md docs/fake_cifar10_residual_stratified_control_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_stratified_control_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-trials 1 --score-bins 10 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_stratified_control_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Oracle top-IMP residual masks still outperform all generated controls. For the
final dense base, base/oracle/random-IMP/parameter-score-non-IMP accuracies are
`0.8803`, `0.8872`, `0.8818`, and `0.8758`. For RMS trajectory they are
`0.8708`, `0.8858`, `0.8794`, and `0.8716`. For epoch 10 they are `0.8678`,
`0.8854`, `0.8764`, and `0.8685`. The parameter-score non-IMP control matches
`>0.999` of the oracle's parameter tensor and within-parameter score-decile
strata but does not recover the oracle gain.

Interpretation:

The residual is not explained by layer/tensor allocation or score-bin
structure. Random IMP-only additions recover more function than non-IMP
controls, so IMP membership itself carries signal, but the remaining oracle
gap shows that the specific high-IMP residual subset selected by IMP is also
functionally important.

## 2026-05-04 - CIFAR Residual IMP Process Probe

Purpose:

Test when the IMP-only residual support becomes usable. The probe starts from
the same trajectory base masks, removes the lowest base-score base-only weights
as before, then adds half residual support using candidates from IMP process
rounds 1, 3, and 5 ranked by the weights trained at that round. Two candidate
variants are evaluated: round survivors and final-IMP residual candidates
ranked by round-trained weights.

Code changes:

- Added `scripts/run_residual_imp_process_probe.py`.
- Added `scripts/summarize_residual_imp_process_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper statistics table with
  residual IMP-process statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 2 --process-rounds 1,2 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --round-variants survivor,final-imp --out-dir runs/fake_cifar10_residual_imp_process_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_smoke --out-csv runs/fake_cifar10_residual_imp_process_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants survivor,final-imp --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Round-survivor additions increasingly concentrate final IMP residual support.
For final dense, round-1/round-3/round-5 final-IMP precision is
`0.4431`/`0.7523`/`1.0000`, and accuracy is
`0.8792`/`0.8829`/`0.8867` against base `0.8793` and oracle `0.8898`. For RMS
trajectory, precision is `0.4308`/`0.7459`/`1.0000`, and accuracy is
`0.8771`/`0.8832`/`0.8857` against base `0.8736` and oracle `0.8881`. For epoch
10, precision is `0.4395`/`0.7536`/`1.0000`, and accuracy is
`0.8772`/`0.8826`/`0.8861` against base `0.8724` and oracle `0.8884`.

Interpretation:

The residual signal is progressively constructed by the IMP process. Early
round survivors already carry above-random final-IMP residual signal; middle
round survivors are much more concentrated; final-round survivors are the final
IMP residual by construction. The process masks still trail the final oracle
residual and overlap only about `0.67` of the oracle added subset at round 5,
so the result supports process-specific residual construction rather than a
standalone dense-trajectory coordinate rule.

## 2026-05-04 - CIFAR Residual IMP Process Ranking Controls

Purpose:

Separate round-survivor membership from round-trained score ordering. The
previous IMP-process probe showed that round-survivor additions increasingly
concentrate final IMP residual support. This control asks whether that is only
because the candidate set is closer to final IMP, or whether the ordering
within a round survivor set is also functionally meaningful.

Code changes:

- Extended `scripts/run_residual_imp_process_probe.py` with
  `survivor-random` and `survivor-low` round variants plus `--random-trials`.
- Extended `scripts/summarize_residual_imp_process_probe.py` labels for the
  new control variants.
- Extended `scripts/build_paper_stats.py`, `docs/paper_stats.md`, and
  `paper/tables/statistical_summary.tex` with residual IMP-process ranking
  control statistics.

Commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 2 --process-rounds 1,2 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --round-variants survivor,survivor-random,survivor-low,final-imp --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_controls_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_controls_smoke --out-csv runs/fake_cifar10_residual_imp_process_controls_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_controls_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants survivor,survivor-random,survivor-low --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Top round-survivor additions outperform random and low-score survivor
additions. For the RMS trajectory base, top/random/low accuracies are
`0.8788`/`0.8733`/`0.8738` at round 1, `0.8830`/`0.8759`/`0.8726` at round 3,
and `0.8857`/`0.8803`/`0.8759` at round 5. The final dense and epoch-10 bases
show the same ordering. At rounds 1 and 3, top-score additions have much higher
final-IMP precision than random or low-score additions; for RMS trajectory the
values are `0.4268`/`0.7496` for top, `0.1871`/`0.4294` for random, and
`0.0808`/`0.1885` for low. At round 5 all variants are final-IMP residual by
construction, but top-score additions overlap far more of the final oracle
added subset than random or low additions: about `0.68` versus `0.50` and
`0.32` across bases.

Interpretation:

Final IMP membership explains part of the residual signal, but it is not the
whole explanation. The score ordering created within IMP rounds adds functional
structure and selects a subset closer to the final oracle residual. This
strengthens the process-specific residual account: IMP is not simply exposing
a dense-trajectory coordinate ranking or a final-support membership set.

## 2026-05-04 - CIFAR Residual Removal-Order Controls

Purpose:

Test whether the residual-swap gain is an artifact of removing low-score
base-only weights. The control holds the added top-IMP residual support fixed
and varies which base-only weights are removed: low, random, or high base
score. The same run keeps the non-IMP random residual control under low
removal.

Code changes:

- Extended `scripts/run_trajectory_residual_probe.py` with
  `--imp-remove-orders low,random,high`.
- Added removal metadata to trajectory residual rows: `remove_order` and
  `add_order`.
- Extended `scripts/build_paper_stats.py`, `docs/paper_stats.md`, and
  `paper/tables/statistical_summary.tex` with residual removal-order controls.

Commands:

```bash
.venv/bin/python scripts/run_trajectory_residual_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0,0.5 --imp-remove-orders low,random,high --random-residual-trials 1 --out-dir runs/fake_cifar10_trajectory_residual_removal_controls_smoke
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/fake_cifar10_trajectory_residual_removal_controls_smoke --out-csv runs/fake_cifar10_trajectory_residual_removal_controls_smoke_summary.csv --out-md docs/fake_cifar10_trajectory_residual_removal_controls_smoke.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_residual_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0,0.5 --imp-remove-orders low,random,high --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3; done
```

```bash
.venv/bin/python scripts/summarize_trajectory_residual_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Top-IMP residual additions remain strong regardless of whether low, random, or
high base-only weights are removed. For the final dense base, low/random/high
removal accuracies are `0.8881`/`0.8883`/`0.8906`, while non-IMP random
additions reach `0.8779`. For RMS trajectory they are
`0.8874`/`0.8896`/`0.8914` versus `0.8709`; for epoch 10 they are
`0.8862`/`0.8922`/`0.8920` versus `0.8701`.

Interpretation:

The residual gain is not explained by preferentially removing weak base-mask
weights. The identity of the added IMP-only residual weights is the dominant
factor, strengthening the claim that IMP constructs or selects a functional
residual support beyond dense-trajectory magnitude.

## 2026-05-04 - CIFAR Direct Cross-Seed Residual-Support Transfer

Purpose:

Test a stricter seed-transfer hypothesis than the cross-seed predictor probe:
if the IMP-only residual is a seed-invariant coordinate set, then oracle
residual coordinates from the other seeds should directly transfer to the target
seed. The probe adds target non-base weights using vote counts over source
seeds' oracle residual supports, with matched source-vote random, target
random, and target oracle residual controls.

Code changes:

- Added `scripts/run_residual_direct_transfer_probe.py`.
- Added `scripts/summarize_residual_direct_transfer_probe.py`.
- Extended `scripts/build_paper_stats.py`, `docs/paper_stats.md`, and
  `paper/tables/statistical_summary.tex` with the direct support-transfer
  table.

Commands:

```bash
.venv/bin/python scripts/run_residual_direct_transfer_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --random-residual-trials 1 --out-dir runs/fake_cifar10_residual_direct_transfer_smoke
.venv/bin/python scripts/summarize_residual_direct_transfer_probe.py --run-root runs/fake_cifar10_residual_direct_transfer_smoke --out-csv runs/fake_cifar10_residual_direct_transfer_smoke_summary.csv --out-md docs/fake_cifar10_residual_direct_transfer_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_direct_transfer_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_direct_transfer_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_direct_transfer_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Source-vote additions from other seeds are slightly enriched for target IMP-only
residual weights, but the enrichment is much weaker than oracle and does not
produce a functional gain. Added precision is `0.1517` for final dense,
`0.1433` for RMS trajectory, and `0.1482` for epoch 10, compared with
target-random precision `0.1249`, `0.1229`, and `0.1250`. Accuracy remains at
base/random-like levels: final dense base/source-vote/source-vote-random/
target-random/target-oracle accuracies are `0.8808`/`0.8779`/`0.8794`/`0.8788`/
`0.8877`; RMS trajectory gives `0.8738`/`0.8715`/`0.8730`/`0.8729`/`0.8866`;
epoch 10 gives `0.8709`/`0.8688`/`0.8727`/`0.8714`/`0.8866`.

Interpretation:

The functional IMP-only residual is not a directly transferable coordinate set
shared across seeds. Other-seed oracle residual supports contain a small amount
of target residual signal, but they behave like random/source-vote-random masks
after retraining and remain far below the target oracle residual. This closes a
stronger seed-invariance alternative and supports the process-specific or
combinatorial residual-support interpretation.

## 2026-05-04 - CIFAR Residual Base-Compatibility Probe

Purpose:

Test whether the oracle IMP-only residual only works when added to the exact
dense-trajectory base. For each trajectory base, the probe constructs a
per-parameter random base that preserves the same number of IMP and non-IMP
weights, and therefore the same base-to-IMP overlap, then compares base,
oracle-residual, and random-residual retraining.

Code changes:

- Added `scripts/run_residual_base_compatibility_probe.py`.
- Added `scripts/summarize_residual_base_compatibility_probe.py`.
- Extended `scripts/build_paper_stats.py`, `docs/paper_stats.md`, and
  `paper/tables/statistical_summary.tex` with the base-compatibility table.

Commands:

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --random-residual-trials 1 --out-dir runs/fake_cifar10_residual_base_compatibility_smoke
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/fake_cifar10_residual_base_compatibility_smoke --out-csv runs/fake_cifar10_residual_base_compatibility_smoke_summary.csv --out-md docs/fake_cifar10_residual_base_compatibility_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

The result reverses the simple base-compatibility hypothesis. IMP-overlap
matched random bases are weak alone: final dense, RMS trajectory, and epoch-10
matched-base accuracies are `0.8641`, `0.8605`, and `0.8607`, below the
trajectory-base accuracies `0.8827`, `0.8743`, and `0.8744`. But adding the top
oracle IMP-only residual recovers `0.8926`, `0.8910`, and `0.8942`, matching
or exceeding trajectory-oracle accuracies `0.8893`, `0.8892`, and `0.8892`.
Matched random residual additions stay weak at `0.8649`, `0.8628`, and
`0.8636`.

Interpretation:

Exact dense-trajectory base identity is not necessary once target IMP overlap
and the top IMP-only residual subset are fixed. Dense trajectory magnitude is
still a useful way to find a trainable support, but this control shows that the
strongest causal residual signal is the particular high-IMP residual subset and
its overlap structure, not the original trajectory base coordinates by
themselves.

## 2026-05-04 - CIFAR Residual Base-Ordering Probe

Purpose:

Separate final IMP membership from within-IMP residual ordering. The probe
keeps the IMP-overlap-matched random base from the base-compatibility control,
then adds the same number of target IMP-only residual weights using the top
oracle ordering, a uniformly random IMP-only ordering, or the lowest
trajectory-score IMP-only ordering, plus a random non-base residual baseline.

Code changes:

- Extended `scripts/run_residual_base_compatibility_probe.py` with
  `--base-kinds` and `--residual-variants`.
- Added `random_imp_only_residual` and `low_imp_only_residual` variants with
  oracle-overlap accounting.
- Extended `scripts/summarize_residual_base_compatibility_probe.py` and
  `scripts/build_paper_stats.py` to report oracle overlap and the
  base-ordering table.

Commands:

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,random-imp,low-imp,random-residual --random-residual-trials 1 --out-dir runs/fake_cifar10_residual_base_ordering_smoke
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/fake_cifar10_residual_base_ordering_smoke --out-csv runs/fake_cifar10_residual_base_ordering_smoke_summary.csv --out-md docs/fake_cifar10_residual_base_ordering_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,random-imp,low-imp,random-residual --random-residual-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3
```

```bash
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_base_ordering_r5_p0p3.md`
- `paper/tables/statistical_summary.tex`
- `docs/paper_stats.md`
- `runs/paper_stats.json`

Result:

Across final dense, RMS trajectory, and epoch-10 bases, matched bases reach
`0.8613`, `0.8601`, and `0.8600`. Top oracle IMP-only residual additions reach
`0.8904`, `0.8910`, and `0.8903`. Uniformly random IMP-only residual additions
reach `0.8785`, `0.8791`, and `0.8787` with about `0.50` oracle-overlap
precision. Low IMP-only residual additions reach `0.8683`, `0.8677`, and
`0.8694` with near-zero oracle overlap. Random non-base residual additions stay
near base at `0.8624`, `0.8618`, and `0.8651`.

Interpretation:

Final IMP membership is useful but not sufficient. Once the matched random base
and final IMP-only candidate pool are fixed, selecting the top target
IMP-residual coordinates still gives a large additional gain over random or
low-ranked IMP-only additions. The result supports a residual-ranking
mechanism rather than a final-membership-only explanation.

## 2026-05-04 - CIFAR Low-Dimensional Full-Network Subspace HMC Probe

Purpose:

Add a tractable CIFAR-scale HMC posterior baseline beyond SGLD-family dynamics
and Gaussian Laplace/SWAG perturbations. The probe samples a random
low-dimensional orthonormal subspace around the dense ResNet-20 checkpoint with
full-data HMC, then induces magnitude masks from the sampled full-network
states.

Code changes:

- Added `src/lottery/subspace_hmc.py` with random orthonormal subspace HMC,
  full-data deterministic potential evaluation, batchnorm-mode control, and
  parameter-distance diagnostics.
- Added `scripts/run_subspace_hmc_probe.py` for dense/IMP preparation,
  SNIP/SynFlow controls, posterior mask overlap, posterior map, sample accuracy,
  prediction agreement, and state/function clustering.
- Added `scripts/summarize_subspace_hmc_probe.py`.
- Extended `scripts/build_paper_stats.py` and the paper/docs to include the
  selected subspace-HMC row.

Commands:

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --subspace-dims 2 --hmc-step-sizes 1e-3 --hmc-steps 4 --hmc-leapfrog-steps 2 --hmc-burn-in 1 --hmc-sample-every 1 --random-trials 5 --snip-batches 1 --out-dir runs/fake_cifar10_subspace_hmc_smoke
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/fake_cifar10_subspace_hmc_smoke --out-csv runs/fake_cifar10_subspace_hmc_smoke_summary.csv --out-md docs/fake_cifar10_subspace_hmc_smoke.md
```

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 8 --hmc-direction-scale 10 --hmc-step-sizes 1e-4,3e-4,1e-3,3e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 8 --hmc-direction-scale 10 --hmc-step-sizes 3e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/fake_cifar10_subspace_hmc_smoke_summary.csv`
- `docs/fake_cifar10_subspace_hmc_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_subspace_hmc_scale10_tune_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3.md`

Result:

The selected five-seed row uses subspace dimension 8, direction scale 10, and
HMC step size `3e-3`. Dense accuracy is `0.8866`, IMP accuracy is `0.8965`,
accept rate is `0.7400`, sample accuracy is `0.8863`, and mean parameter
distance from the dense checkpoint is `0.3672`. The posterior support remains
non-ticket-directed: posterior-to-IMP is `0.1440`, chain-start magnitude is
`0.1440`, posterior-minus-chain-start is `-2.2e-05`, posterior-to-chain-start
support overlap is `0.9766`, and epoch-1 rewind magnitude is closer to IMP at
`0.1779`.

Interpretation:

This is not exact full-network full-covariance posterior evidence, but it does
reduce the concern that CIFAR baselines only used stochastic-gradient samplers
or Gaussian perturbations. A moving full-network subspace HMC chain preserves
dense function and accuracy, yet its magnitude support does not move toward
IMP.

## 2026-05-05 - CIFAR Trajectory-Informed Subspace HMC Probe

Purpose:

Test a stronger version of the subspace-HMC objection: perhaps the useful local
posterior directions are not random but aligned with the same dense trajectory
directions that already explain much of the IMP support. This probe samples the
orthonormal subspace spanned by dense checkpoint directions to the final dense
checkpoint.

Code changes:

- Extended `src/lottery/subspace_hmc.py` so HMC can use caller-provided
  subspace directions after rank checking and QR orthonormalization.
- Extended `scripts/run_subspace_hmc_probe.py` with `--hmc-basis trajectory`
  and `--hmc-trajectory-epochs`.
- Extended `scripts/build_paper_stats.py` to report `RandSubHMC` and
  `TrajSubHMC` separately.

Commands:

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --hmc-basis trajectory --hmc-trajectory-epochs 0,1,2 --subspace-dims 2 --hmc-direction-scale 10 --hmc-step-sizes 1e-3 --hmc-steps 4 --hmc-leapfrog-steps 2 --hmc-burn-in 1 --hmc-sample-every 1 --random-trials 5 --snip-batches 1 --out-dir runs/fake_cifar10_trajectory_subspace_hmc_smoke
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/fake_cifar10_trajectory_subspace_hmc_smoke --out-csv runs/fake_cifar10_trajectory_subspace_hmc_smoke_summary.csv --out-md docs/fake_cifar10_trajectory_subspace_hmc_smoke.md
```

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --hmc-basis trajectory --hmc-trajectory-epochs 0,1,2,5,10,20,30 --subspace-dims 6 --hmc-direction-scale 10 --hmc-step-sizes 1e-4,3e-4,1e-3,3e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --hmc-basis trajectory --hmc-trajectory-epochs 0,1,2,5,10,20,30 --subspace-dims 6 --hmc-direction-scale 10 --hmc-step-sizes 3e-4,1e-3 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 4 --hmc-sample-every 4 --hmc-prior-precision 1e-4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/fake_cifar10_trajectory_subspace_hmc_smoke_summary.csv`
- `docs/fake_cifar10_trajectory_subspace_hmc_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_tune_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3.md`

Result:

The selected five-seed trajectory-subspace row uses subspace dimension 6,
direction scale 10, and HMC step sizes `3e-4` and `1e-3`. At `1e-3`, dense
accuracy is `0.8849`, IMP accuracy is `0.8961`, accept rate is `0.6900`, sample
accuracy is `0.8847`, and mean parameter distance from the dense checkpoint is
`0.1589`. The support starts from a strong dense-trajectory chain-start control
(`0.2292`), but HMC does not improve it: posterior-to-IMP is `0.2290`,
posterior-minus-chain-start is `-0.00019`, and post-chain is `0.9915`.

Interpretation:

Trajectory-informed HMC closes a stronger posterior rescue path than random
subspace HMC. The dense trajectory directions themselves contain much more IMP
support signal, but HMC posterior movement in that subspace remains a
chain-start magnitude proxy rather than becoming more ticket-directed.

## 2026-05-05 - CIFAR Residual Posterior-Ordering Probe

Purpose:

Test whether posterior scores can rank useful residual additions after giving
the method the oracle IMP-only residual candidate pool. This separates the
question "does the posterior identify final IMP membership?" from the stronger
question "does the posterior rank the functionally important IMP-only residual
coordinates?"

Code changes:

- Extended `scripts/run_residual_base_compatibility_probe.py` with
  `posterior-imp`, which ranks target IMP-only residual additions by
  diagonal-Laplace posterior RMS from the dense model.
- Added posterior-Laplace controls:
  `--posterior-laplace-samples`, `--posterior-laplace-scale`,
  `--posterior-laplace-prior-precision`,
  `--posterior-laplace-fisher-batches`, and
  `--posterior-laplace-variance-floor`.
- Updated `scripts/summarize_residual_base_compatibility_probe.py` and
  `scripts/build_paper_stats.py` to report the posterior IMP-only variant.

Commands:

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-epochs 1 --imp-final-epochs 1 --mask-train-epochs 1 --imp-rounds 2 --prune-fraction 0.30 --batch-size 32 --train-subset 128 --test-subset 64 --lr 0.02 --lr-schedule cosine --weight-decay 0.0 --base-sources epoch_2,traj_rms_abs,epoch_1 --base-kinds imp_overlap_random --alphas 0.5 --residual-variants oracle,posterior-imp,random-imp,low-imp,random-residual --random-residual-trials 1 --posterior-laplace-samples 2 --posterior-laplace-scale 1e-3 --posterior-laplace-fisher-batches 1 --out-dir runs/fake_cifar10_residual_posterior_ordering_smoke
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/fake_cifar10_residual_posterior_ordering_smoke --out-csv runs/fake_cifar10_residual_posterior_ordering_smoke_summary.csv --out-md docs/fake_cifar10_residual_posterior_ordering_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,posterior-imp,random-imp,low-imp,random-residual --random-residual-trials 1 --posterior-laplace-samples 10 --posterior-laplace-scale 1e-3 --posterior-laplace-prior-precision 1e-2 --posterior-laplace-fisher-batches 20 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_residual_posterior_ordering_smoke_summary.csv`
- `docs/fake_cifar10_residual_posterior_ordering_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_posterior_ordering_r5_p0p3.md`

Result:

Across final dense, RMS trajectory, and epoch-10 matched random bases,
diagonal-Laplace posterior-RMS-ranked IMP-only residual additions reach
`0.8835`, `0.8845`, and `0.8829` accuracy. This is consistently above
uniformly random IMP-only additions (`0.8779`, `0.8764`, `0.8778`) and has
oracle-overlap precision `0.548--0.554` rather than about `0.50`. However it
remains below top oracle IMP-only additions (`0.8903`, `0.8897`, `0.8908`) and
below IMP.

Interpretation:

Posterior RMS is not completely uninformative once the final IMP-only residual
candidate pool is granted, but it captures only a weak residual-ordering signal.
This supports a refined negative result: posterior approximations can inherit
dense/trajectory magnitude structure and slightly enrich the useful residual
subset, but they do not reproduce the process-specific top IMP residual
ranking that closes the functional gap.

## 2026-05-05 - CIFAR Residual Posterior-Decomposition Probe

Purpose:

Separate whether the posterior-RMS residual-ordering signal is genuinely
posterior-specific or mostly dense final magnitude inherited by the local
Laplace samples.

Implementation:

- Extended `scripts/run_residual_base_compatibility_probe.py` with score-ranked
  IMP-only residual variants:
  `dense-imp`, `posterior-excess-imp`, and `posterior-std-imp`.
- The diagonal-Laplace pass now emits posterior RMS, posterior standard
  deviation, and posterior RMS minus dense magnitude from one shared sample set.
- Updated the residual summarizer, paper stats labels, README commands, and
  paper narrative to report the decomposition.

Commands:

```bash
.venv/bin/python -m py_compile scripts/run_residual_base_compatibility_probe.py scripts/summarize_residual_base_compatibility_probe.py scripts/build_paper_stats.py
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,random-imp,dense-imp,posterior-imp,posterior-excess-imp,posterior-std-imp --random-residual-trials 1 --posterior-laplace-samples 3 --posterior-laplace-scale 1e-3 --posterior-laplace-prior-precision 1e-2 --posterior-laplace-fisher-batches 1 --out-dir runs/fake_cifar10_residual_posterior_decomposition_smoke
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/fake_cifar10_residual_posterior_decomposition_smoke --out-csv runs/fake_cifar10_residual_posterior_decomposition_smoke_summary.csv --out-md docs/fake_cifar10_residual_posterior_decomposition_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_base_compatibility_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --base-kinds imp_overlap_random --residual-variants oracle,random-imp,dense-imp,posterior-imp,posterior-excess-imp,posterior-std-imp --random-residual-trials 1 --posterior-laplace-samples 10 --posterior-laplace-scale 1e-3 --posterior-laplace-prior-precision 1e-2 --posterior-laplace-fisher-batches 20 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3
.venv/bin/python scripts/summarize_residual_base_compatibility_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_residual_posterior_decomposition_smoke_summary.csv`
- `docs/fake_cifar10_residual_posterior_decomposition_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3.md`

Result:

Across final dense, RMS trajectory, and epoch-10 matched random bases, top
oracle IMP-only residual additions reach `0.8915`, `0.8928`, and `0.8911`.
Dense-final-magnitude-ranked IMP-only additions reach `0.8821`, `0.8812`, and
`0.8827` with oracle-overlap precision `0.553--0.557`. Diagonal-Laplace
posterior-RMS-ranked additions closely match this at `0.8852`, `0.8834`, and
`0.8829` with `0.551--0.556` overlap. Uniform random IMP-only additions reach
`0.8783`, `0.8795`, and `0.8791` with about `0.50` overlap. Posterior
RMS-minus-dense and posterior standard deviation are weaker: `0.8745--0.8770`
and `0.8710--0.8717`, with `0.478--0.479` and `0.446--0.450` oracle-overlap
precision.

Interpretation:

The earlier posterior-RMS residual-ordering signal is not a clean posterior
uncertainty effect. It is largely matched by dense final magnitude, while
posterior movement/uncertainty scores fail to recover the oracle residual
ranking. This strengthens the negative result: posterior approximations can
inherit dense magnitude structure inside the final IMP candidate pool, but they
do not explain the process-specific functional residual ordering built by IMP.

## 2026-05-05 - CIFAR Activation-Aligned Direct Residual Transfer

Purpose:

Test whether direct cross-seed residual-support transfer fails merely because
independent ResNet seeds learn permuted channels. The new control maps each
source seed into the target seed's channel coordinates using activation
correlation matching before voting over source oracle residual supports.

Implementation:

- Extended `scripts/run_residual_direct_transfer_probe.py` with
  `--alignment-method activation` and `--alignment-batches`.
- The ResNet-20 alignment captures stem, block-input, and block-output
  activation features, solves source-to-target channel matchings by Hungarian
  assignment, then maps source conv/fc mask axes into target coordinates before
  source-vote aggregation.
- Added aligned variants to the direct-transfer summary:
  `aligned_source_vote_residual` and
  `aligned_source_vote_random_residual`.

Commands:

```bash
.venv/bin/python -m py_compile scripts/run_residual_direct_transfer_probe.py scripts/summarize_residual_direct_transfer_probe.py
.venv/bin/python scripts/run_residual_direct_transfer_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0,1 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --base-sources epoch_2,traj_rms_abs --alphas 0.5 --random-residual-trials 1 --alignment-method activation --alignment-batches 1 --out-dir runs/fake_cifar10_residual_aligned_direct_transfer_smoke
.venv/bin/python scripts/summarize_residual_direct_transfer_probe.py --run-root runs/fake_cifar10_residual_aligned_direct_transfer_smoke --out-csv runs/fake_cifar10_residual_aligned_direct_transfer_smoke_summary.csv --out-md docs/fake_cifar10_residual_aligned_direct_transfer_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_direct_transfer_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --random-residual-trials 1 --alignment-method activation --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3
.venv/bin/python scripts/summarize_residual_direct_transfer_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_residual_aligned_direct_transfer_smoke_summary.csv`
- `docs/fake_cifar10_residual_aligned_direct_transfer_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3.md`

Result:

Activation matching has mean correlation about `0.593` across aligned rows, so
the matching path is nontrivial. However aligned source-vote masks still fail
to recover the target oracle residual. For final dense, source-vote,
aligned-source-vote, aligned-random, target-random, and target-oracle
accuracies are `0.8769`, `0.8797`, `0.8809`, `0.8779`, and `0.8886`. For RMS
trajectory they are `0.8739`, `0.8727`, `0.8743`, `0.8728`, and `0.8872`. For
epoch 10 they are `0.8725`, `0.8710`, `0.8740`, `0.8719`, and `0.8890`.
Aligned source-vote added target IMP-only precision remains only
`0.1469--0.1509`, close to the unaligned source-vote precision and far below
oracle precision.

Interpretation:

The direct cross-seed residual-transfer failure is not explained by a simple
channel-permutation mismatch. Activation alignment sometimes moves accuracy
slightly, but aligned random controls move similarly or more. The useful
residual support remains target/process-specific rather than a seed-invariant
coordinate set recoverable by channel alignment.

## 2026-05-05 - CIFAR Hessian-Subspace HMC Probe

Purpose:

Test whether the negative CIFAR posterior-support result is an artifact of
using random or trajectory-selected low-dimensional HMC subspaces. This probe
constructs a full-network subspace from randomized top-Hessian directions at
the dense checkpoint, then runs the existing deterministic full-data HMC
sampler inside that curvature-informed subspace.

Implementation:

- Extended `scripts/run_subspace_hmc_probe.py` with `--hmc-basis hessian`.
- Added randomized Hessian subspace iteration using Hessian-vector products,
  controlled by `--hessian-batches`, `--hessian-power-iterations`, and
  `--hessian-oversample`.
- The HMC sampler, support masks, posterior-map controls, clustering, SNIP, and
  SynFlow diagnostics are reused from the existing subspace-HMC path.

Commands:

```bash
.venv/bin/python -m py_compile scripts/run_subspace_hmc_probe.py
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --subspace-dims 2 --hmc-basis hessian --hmc-trajectory-epochs 0,1 --hmc-step-sizes 1e-4 --hmc-steps 4 --hmc-leapfrog-steps 1 --hmc-burn-in 1 --hmc-sample-every 1 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hessian-batches 1 --hessian-power-iterations 0 --hessian-oversample 1 --random-trials 5 --snip-batches 1 --out-dir runs/fake_cifar10_hessian_subspace_hmc_smoke
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/fake_cifar10_hessian_subspace_hmc_smoke --out-csv runs/fake_cifar10_hessian_subspace_hmc_smoke_summary.csv --out-md docs/fake_cifar10_hessian_subspace_hmc_smoke.md
```

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 4 --hmc-basis hessian --hmc-step-sizes 1e-3,3e-3,1e-2 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 2 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 4 --hmc-basis hessian --hmc-step-sizes 1e-5,3e-5,1e-4,3e-4 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 2 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_smallstep_seed0_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 4 --hmc-basis hessian --hmc-step-sizes 3e-4 --hmc-steps 20 --hmc-leapfrog-steps 2 --hmc-burn-in 5 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 2 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_hessian_subspace_hmc_smoke_summary.csv`
- `docs/fake_cifar10_hessian_subspace_hmc_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_smallstep_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_tune_smallstep_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3.md`

Result:

The large-step seed-0 tune rejected all proposals at `1e-3`, `3e-3`, and
`1e-2`. The small-step tune showed that `3e-4` gives nonzero movement and
acceptable HMC behavior, but still leaves support essentially unchanged. The
5-seed selected run at `3e-4` has accept rate `0.86`, sample accuracy `0.8865`,
and mean parameter distance `0.0022`. Posterior-to-IMP is `0.14713`, matching
chain-start magnitude at `0.14713`; posterior-minus-chain is only `0.000002`,
and posterior-to-chain-start support overlap is `0.99989`. Rewind magnitude is
substantially closer to IMP at `0.1810`.

Interpretation:

Top-Hessian full-network HMC is stable and function-preserving in this setting,
but its local posterior support is even more tightly locked to the dense
chain-start support than the random-subspace HMC control. This reduces the
concern that random or trajectory subspace selection caused the negative HMC
result. It still does not replace an exact full-network full-covariance CIFAR
posterior, but it is a stronger curvature-informed subspace posterior control.

## 2026-05-05 - CIFAR Calibration/OOD Probe

Purpose:

Test whether a posterior-predictive uncertainty view rescues the negative
support result. The probe compares dense, IMP, SWAG member, and SWAG ensemble
predictives on CIFAR-10 calibration and CIFAR-100 OOD detection in the same
30-epoch epoch-1 rewind r5 p0.30 setting.

Implementation:

- Added `scripts/run_calibration_ood_probe.py`.
- Added `scripts/summarize_calibration_ood_probe.py`.
- The probe reports ID accuracy, NLL, Brier, ECE, maximum-softmax OOD AUROC,
  entropy OOD AUROC, and FPR95.

Commands:

```bash
.venv/bin/python -m py_compile scripts/run_calibration_ood_probe.py scripts/summarize_calibration_ood_probe.py
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset fake-cifar10 --ood-dataset gaussian-noise --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --ood-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --swag-epochs 1 --swag-lr 0.005 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 2 --samples 2 --out-dir runs/fake_cifar10_calibration_ood_smoke
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/fake_cifar10_calibration_ood_smoke --out-csv runs/fake_cifar10_calibration_ood_smoke_summary.csv --out-md docs/fake_cifar10_calibration_ood_smoke.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --swag-scale 1.0 --swag-diagonal-scale 1.0 --swag-low-rank-scale 1.0 --samples 10 --out-dir runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3; done
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_calibration_ood_smoke_summary.csv`
- `docs/fake_cifar10_calibration_ood_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3.md`

Result:

Across five seeds, dense reaches accuracy `0.8866`, NLL `0.3536`, ECE
`0.0353`, and MSP OOD AUROC `0.8230`. IMP reaches accuracy `0.8953`, NLL
`0.3387`, ECE `0.0393`, and MSP OOD AUROC `0.8306`. The 10-sample SWAG
ensemble reaches accuracy `0.8688`, NLL `0.4018`, ECE `0.0285`, and MSP OOD
AUROC `0.8050`.

Interpretation:

SWAG predictive averaging improves ECE, but it worsens accuracy, NLL, and OOD
AUROC. This supports a separation statement: posterior uncertainty behavior is
not sufficient to explain winning-ticket support alignment.

## 2026-05-05 - Gem-Miner-Style Mask Source Smoke

Purpose:

Add an implementation path for the remaining Gem-Miner-style mask-subspace
control. The implementation follows the paper's central mechanism: learn score
variables over fixed initialization weights with a straight-through rounded
mask, then iteratively freeze low-score coordinates until the target sparsity
is reached.

Implementation:

- Added `gem_miner_mask` to `src/lottery/pruning_baselines.py`.
- Wired `gem_miner` into `scripts/run_trajectory_mask_training_probe.py` as a
  mask source.
- Gem-Miner masks train from the initial state, while the existing trajectory
  masks keep using the configured rewind state.

Commands:

```bash
.venv/bin/python -m py_compile src/lottery/pruning_baselines.py scripts/run_trajectory_mask_training_probe.py
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --mask-sources gem_miner,random --random-trials 1 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-regularization 0 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --out-dir runs/fake_cifar10_gem_miner_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/fake_cifar10_gem_miner_mask_training_smoke --out-csv runs/fake_cifar10_gem_miner_mask_training_smoke_summary.csv --out-md docs/fake_cifar10_gem_miner_mask_training_smoke.md
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-regularization 0 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --out-dir runs/cifar10_subset_gem_miner_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_subset_gem_miner_mask_training_smoke --out-csv runs/cifar10_subset_gem_miner_mask_training_smoke_summary.csv --out-md docs/cifar10_subset_gem_miner_mask_training_smoke.md
```

Output:

- `runs/fake_cifar10_gem_miner_mask_training_smoke_summary.csv`
- `docs/fake_cifar10_gem_miner_mask_training_smoke.md`
- `runs/cifar10_subset_gem_miner_mask_training_smoke_summary.csv`
- `docs/cifar10_subset_gem_miner_mask_training_smoke.md`

Result:

Both fake-CIFAR and real CIFAR-subset smoke paths complete. The CIFAR-subset
smoke reaches target sparsity `0.3000` and produces a trainable Gem-Miner mask
source. This is only a plumbing validation, not scientific evidence.

Interpretation:

Gem-Miner is no longer unimplemented in the codebase. A submission-grade use
requires full-data multi-seed CIFAR evidence; the selected five-seed row below
provides the first version of that evidence for this Gem-Miner-style
implementation.

## 2026-05-05 - CIFAR Gem-Miner-Style Full-Data Seed0 Pilot

Purpose:

Check whether the newly implemented Gem-Miner-style mask source has any
immediate full-data CIFAR signal before investing in a multi-seed sweep.

Command:

```bash
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 5 --gem-miner-lr 0.1 --gem-miner-regularization 0 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_gem_miner_tune_seed0_r5_p0p3.md`

Result:

The dense model reaches `0.8843`, IMP reaches `0.8946`, and the Gem-Miner-style
mask retrained from initialization reaches `0.8428` at the matched `0.8319`
sparsity. The Gem-Miner-style mask has support Jaccard `0.0909` to IMP,
`0.0917` to the dense final magnitude mask, and `0.0920` to the rewind
magnitude mask.

Interpretation:

This seed0/tuned-budget pilot is strongly negative and essentially random-like
in support, but it is not by itself a full Gem-Miner baseline. The selected
five-seed row below tests the same setting across seeds.

## 2026-05-05 - CIFAR 16D Hessian-Subspace HMC Tune

Purpose:

Test whether the 4-dimensional top-Hessian HMC control was too narrow. This
tune widens the curvature-informed full-network subspace to 16 dimensions on
seed 0 and sweeps HMC step sizes.

Command:

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 16 --hmc-basis hessian --hmc-step-sizes 1e-4,3e-4,1e-3 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3.md`

Result:

At step `1e-4`, accept rate is `1.0`, sample accuracy is `0.8892`, parameter
distance is `0.00325`, and posterior-minus-chain-start is `0.000024`. At step
`3e-4`, accept rate is `0.75`, sample accuracy is `0.8893`, parameter distance
is `0.00795`, post-chain is `0.9996`, and posterior-minus-chain-start is
`0.0`. At step `1e-3`, all proposals reject.

Interpretation:

Widening the top-Hessian subspace from 4D to 16D increases parameter movement
relative to the selected 4D probe, but the sparse support remains locked to the
dense chain-start magnitude mask. This is still a one-seed tune, not a selected
five-seed row.

## 2026-05-05 - CIFAR 16D Hessian-Subspace HMC Selected Row

Purpose:

Promote the seed0 16D top-Hessian tune to a selected five-seed CIFAR row at the
best tuned step size.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 16 --hmc-basis hessian --hmc-step-sizes 3e-4 --hmc-steps 12 --hmc-leapfrog-steps 2 --hmc-burn-in 3 --hmc-sample-every 3 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 4 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3.md`

Result:

Across five seeds, dense accuracy is `0.8873`, IMP accuracy is `0.8974`, HMC
accept rate is `0.8833`, and sample accuracy is `0.8881`. The sampled support
does not improve over the chain-start magnitude support: posterior-to-IMP is
`0.14680`, chain-start-to-IMP is `0.14682`, posterior-minus-chain-start is
`-0.000016`, and posterior-to-chain-start remains `0.9994`. Mean parameter
distance from the dense checkpoint is `0.00949`.

Interpretation:

The 16D curvature-informed HMC row keeps dense-level predictive accuracy and
accepts proposals, but its sparse support remains effectively identical to the
dense chain-start magnitude support. This strengthens the subspace-HMC negative
evidence, while still not replacing an exact full-network full-covariance
CIFAR posterior baseline.

## 2026-05-05 - CIFAR Gem-Miner-Style Five-Seed Selected Row

Purpose:

Promote the one-seed Gem-Miner-style score-training pilot to a five-seed
full-data CIFAR baseline.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources gem_miner --random-trials 0 --gem-miner-epochs 5 --gem-miner-lr 0.1 --gem-miner-regularization 0 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_gem_miner_selected_r5_p0p3.md`

Result:

Across five seeds, dense accuracy is `0.8846`, IMP accuracy is `0.8970`, and
the Gem-Miner-style mask retrained from initialization reaches `0.8471`
accuracy at `0.8319` sparsity. It is below IMP by `0.0499` and below dense by
`0.0376`. Its support overlap with IMP is `0.0917` with 95% CI
`[0.0913, 0.0921]`, essentially random-scale for this sparsity.

Interpretation:

This selected row closes the immediate multi-seed Gem-Miner-style baseline gap.
It is negative for the current score-training implementation, but should be
described as Gem-Miner-style evidence rather than a complete reproduction of
every possible Gem-Miner recipe.

## 2026-05-05 - CIFAR Full-Covariance Block Laplace Probe

Purpose:

Add a higher-fidelity covariance check beyond the final classifier head. The
probe freezes the rest of ResNet-20 and computes a full-covariance
softmax-GGN/Laplace approximation for the 2304-parameter
`layer1.0.conv1.weight` tensor.

Code changes:

- Added `src/lottery/block_laplace.py`.
- Added `scripts/run_block_laplace_probe.py`.
- Added `scripts/summarize_block_laplace_probe.py`.
- Extended `scripts/build_paper_stats.py` with a block-Laplace section.

Commands:

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 256 --test-subset 128 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --block-name conv1.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --samples 2 --random-trials 5 --out-dir runs/fake_cifar10_block_laplace_smoke
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/fake_cifar10_block_laplace_smoke --out-csv runs/fake_cifar10_block_laplace_smoke_summary.csv --out-md docs/fake_cifar10_block_laplace_smoke.md
```

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer1.0.conv1.weight --block-laplace-scales 1e-4,1e-3,1e-2 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 3000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer1.0.conv1.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 3000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3.md
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/fake_cifar10_block_laplace_smoke_summary.csv`
- `docs/fake_cifar10_block_laplace_smoke.md`
- `runs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_block_laplace_tune_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3.md`

Result:

The seed-0 scale sweep chose `1e-3`: it moves the selected block support
substantially (`block post-chain = 0.2092`) while preserving sample accuracy
(`0.8933`). Across five seeds at this scale, dense accuracy is `0.8970`, IMP
accuracy is `0.9029`, and block-Laplace sample accuracy is `0.8961`. The block
posterior-to-IMP support is `0.1959`, below the block chain-start support
`0.2034`; posterior-minus-chain-start is `-0.0075` with CI
`[-0.0194, 0.0044]`. The block rewind-magnitude support remains higher at
`0.2423`. The induced global posterior-to-IMP support is `0.1329`, close to
global chain-start `0.1315`, and global posterior-to-chain-start remains
`0.9557`.

Interpretation:

This reduces the head-only covariance objection because the selected tensor uses
a true full-covariance local Gaussian. It still is not a full-network
full-covariance posterior, and it does not reverse the main support-level
negative result.

## 2026-05-05 - CIFAR Block Laplace Scan and Layer3 Shortcut Selected Row

Purpose:

Check whether the first selected block-Laplace result depends on the particular
`layer1.0.conv1.weight` tensor. The scan evaluates seven small or medium
ResNet-20 weight tensors after one shared dense/IMP training run, then promotes
the only mildly positive scan candidate to a five-seed selected row.

Code change:

- Extended `scripts/run_block_laplace_probe.py` with `--block-names`, so one
  dense/IMP training run can evaluate multiple frozen-rest full-covariance
  block Laplace probes.

Commands:

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --block-names conv1.weight,layer2.0.shortcut.0.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --samples 1 --random-trials 2 --out-dir runs/fake_cifar10_block_laplace_multiblock_smoke
```

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-names conv1.weight,layer1.0.conv1.weight,layer1.2.conv2.weight,layer2.0.conv1.weight,layer2.0.shortcut.0.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 5000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --block-name layer3.0.shortcut.0.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 5000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_block_laplace_multiblock_smoke/20260505_130710`
- `runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3.md`

Result:

The seed-0 scan does not show a broad selected-block rescue. Most block
posterior-minus-chain gaps are negative or near zero. The only mildly positive
block is `layer3.0.shortcut.0.weight`, with block posterior-to-IMP `0.2503`
versus chain-start `0.2388`, but rewind magnitude is still larger at `0.3146`.
The five-seed selected row removes this apparent positive signal: block
posterior-to-IMP is `0.2402`, block chain-start is `0.2411`,
posterior-minus-chain is `-0.0010`, block post-chain is `0.3626`, and sample
accuracy is `0.8905`. Rewind magnitude remains much closer to IMP at `0.3050`.
The induced global support barely changes: global posterior-to-IMP is `0.1333`
versus global chain-start `0.1324`.

Interpretation:

The layer3 shortcut row is useful because it shows large block-level posterior
movement under a full-covariance local Gaussian, but the movement is not
ticket-directed. The first seed-0 scan positive was not stable across seeds.
This strengthens the block-Laplace negative evidence while preserving the
limitation that these are selected-block, not full-network, covariance probes.

## 2026-05-05 - CIFAR Joint Multi-Block Laplace Probe

Purpose:

Reduce the single-tensor and no-cross-covariance objections to the selected
block-Laplace evidence. The probe samples one joint full-covariance
softmax-GGN/Laplace Gaussian over four selected tensors while freezing the rest
of ResNet-20.

Code changes:

- Extended `src/lottery/block_laplace.py` with joint multi-tensor factor
  estimation and sampling.
- Extended `scripts/run_block_laplace_probe.py` with `--joint-block-names`.
- Kept the existing single-block and multi-block scan paths unchanged.

Commands:

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --joint-block-names conv1.weight,layer2.0.shortcut.0.weight --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 2000 --samples 1 --random-trials 2 --out-dir runs/fake_cifar10_joint_block_laplace_smoke
```

```bash
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --joint-block-names conv1.weight,layer1.0.conv1.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-4,3e-4,1e-3 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 6000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3.md
```

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 256 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --joint-block-names conv1.weight,layer1.0.conv1.weight,layer3.0.shortcut.0.weight,fc.weight --block-laplace-scales 1e-4 --block-laplace-hessian-batches 2 --block-laplace-max-parameters 6000 --samples 5 --random-trials 50 --out-dir runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_joint_block_laplace_smoke/20260505_135626`
- `runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3.md`
- `runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3.md`

Result:

The seed-0 scale sweep chose `1e-4`: it preserves sample accuracy (`0.8952`)
and moves selected-group support substantially (group post-chain `0.5132`).
Across five seeds at this scale, dense accuracy is `0.8941`, IMP accuracy is
`0.9037`, and joint-Laplace sample accuracy is `0.8922`. The selected group
has 5424 parameters. Group posterior-to-IMP is `0.3294`, below group
chain-start `0.3501`; posterior-minus-chain is `-0.0206`, and every seed is
negative. Group rewind magnitude is higher at `0.3637`. The induced global
posterior-to-IMP is `0.1319` versus global chain-start `0.1309` and global
rewind `0.1509`.

Interpretation:

This row is stronger than the single-tensor selected-block probes because it
includes cross-tensor covariance among `conv1.weight`,
`layer1.0.conv1.weight`, `layer3.0.shortcut.0.weight`, and `fc.weight`. It
still is not a full-network covariance posterior, and the selected-group
movement is not ticket-directed.

## 2026-05-05 - CIFAR Hessian32 Subspace HMC Tune

Purpose:

Check whether the 16-dimensional top-Hessian HMC result is only negative
because the curvature-informed full-network subspace is too narrow. This tune
widens the Hessian basis to 32 dimensions on seed 0.

Command:

```bash
.venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 32 --hmc-basis hessian --hmc-step-sizes 1e-4,3e-4,1e-3 --hmc-steps 8 --hmc-leapfrog-steps 2 --hmc-burn-in 2 --hmc-sample-every 2 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 8 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3/20260505_145638`
- `runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3.md`

Result:

Dense accuracy is `0.8875` and IMP accuracy is `0.8945`. At step `1e-4`,
accept rate is `1.0000`, sample accuracy is `0.8873`, parameter distance is
`0.0031`, post-chain is `0.9998`, and posterior-to-IMP is `0.1435` versus
chain-start `0.1435`. At step `3e-4`, accept rate is `0.8750`, sample accuracy
is `0.8886`, parameter distance is `0.0113`, post-chain is `0.9994`, and
posterior-to-IMP again matches chain-start up to numerical precision. At step
`1e-3`, accept rate is `0.0000`.

Interpretation:

This does not merit a 5-seed selected row by itself. Widening the top-Hessian
subspace from 16 to 32 dimensions increases the curvature-subspace coverage but
does not create ticket-directed support movement. The larger unresolved gap is
still full-network high-fidelity posterior evidence outside low-dimensional
subspaces.

## 2026-05-05 - Residual IMP-Process Oracle-Overlap-Matched Control Smoke

Purpose:

Add a stricter process-causality control. Existing process-ranking rows show
that round-trained final-IMP additions have partial overlap with the final
oracle residual and improve accuracy. This new control samples random
final-IMP residual additions with the same oracle-overlap count as the
round-score-selected additions. It separates round-score ordering from the
amount of final-oracle residual overlap.

Code changes:

- Extended `scripts/run_residual_imp_process_probe.py` with
  `final-imp-oracle-matched-random`.
- Extended `scripts/summarize_residual_imp_process_probe.py` with a readable
  label for the new variant.

Smoke commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 1 --imp-rounds 1 --process-rounds 1 --round-variants final-imp,final-imp-oracle-matched-random --base-sources epoch_1 --alphas 0.5 --mask-train-epochs 1 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_oracle_matched_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_oracle_matched_smoke --out-csv runs/fake_cifar10_residual_imp_process_oracle_matched_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_oracle_matched_smoke.md
```

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 1 --process-rounds 1 --round-variants final-imp,final-imp-oracle-matched-random --base-sources epoch_2 --alphas 0.5 --mask-train-epochs 1 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --random-trials 1 --out-dir runs/cifar10_subset_residual_imp_process_oracle_matched_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_subset_residual_imp_process_oracle_matched_smoke --out-csv runs/cifar10_subset_residual_imp_process_oracle_matched_smoke_summary.csv --out-md docs/cifar10_subset_residual_imp_process_oracle_matched_smoke.md
```

Output:

- `runs/fake_cifar10_residual_imp_process_oracle_matched_smoke/20260505_150128`
- `docs/fake_cifar10_residual_imp_process_oracle_matched_smoke.md`
- `runs/cifar10_subset_residual_imp_process_oracle_matched_smoke/20260505_150154`
- `docs/cifar10_subset_residual_imp_process_oracle_matched_smoke.md`

Smoke result:

Both smokes confirm the invariant. The new variant has final-IMP precision
`1.0000` and exactly matches the reference round final-IMP oracle-overlap
precision (`0.5641` on fake-CIFAR, `0.5725` on the real CIFAR subset).

Selected full-data command to run next:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-oracle-matched-random --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3.md
```

Seed-0 pilot command:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-oracle-matched-random --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3.md
```

Seed-0 pilot output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3/20260505_150440`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3.md`

Seed-0 pilot result:

The invariant holds for all nine base/round pairs: final-IMP precision is
`1.0000`, and each matched-random row has the same oracle-overlap precision as
its round-score reference. Round-score final-IMP residuals beat the
oracle-overlap-matched random control in 7 of 9 pairs, with mean accuracy delta
`+0.0024`. The signal is not uniform: `epoch_30` round 1 and `traj_rms_abs`
round 3 favor matched random by `0.0020`. This should be expanded to the
5-seed selected row before using it as paper evidence.

## 2026-05-05 - Residual IMP-Process Oracle-Overlap-Matched Control, 5 Seeds

Purpose:

Promote the seed-0 pilot to a selected five-seed CIFAR row. This tests whether
round-trained score ordering inside the final-IMP residual candidate set still
beats a random final-IMP residual when the random control is forced to have the
same final-oracle overlap count as the score-selected additions.

Command:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-oracle-matched-random --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3/20260505_150440`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3/20260505_153421`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3.md`

Result:

Across 45 paired base/round/seed comparisons, round-score final-IMP residuals
beat the oracle-overlap-matched random control in 35 cases. The paired mean
accuracy delta is `+0.0020`. Group means are positive for eight of nine
base/round groups; the exception is `epoch_30` round 1 at `-0.0002`. The
strongest group mean is `traj_rms_abs` round 1 at `+0.0038`, followed by
`epoch_10` round 3 at `+0.0036`.

Interpretation:

This strengthens the process-specific residual story: the top round-trained
ordering is not explained solely by final-IMP membership or by the amount of
final-oracle overlap. The effect is still small and mixed at the seed/round
level, so it should be presented as a mechanistic control rather than as a
large standalone accuracy result.

## 2026-05-05 - Residual IMP-Process Score-Source Control, 5 Seeds

Purpose:

Test whether the remaining round-trained final-IMP ranking signal is just dense
final magnitude or base-source magnitude ordering inside the same final-IMP
residual candidate set. The support budget and candidate pool are fixed; only
the score source changes.

Command:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3/20260505_173401`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3/20260505_175823`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3.md`

Result:

Across 45 paired base/round/seed comparisons, round-trained final-IMP scores
beat dense-final-score ranking in 37 cases with mean accuracy delta `+0.0026`.
They beat base-source-score ranking in 39 cases with mean accuracy delta
`+0.0028`. Eight of nine base/round group means are positive against each
score-source control; the exception is `traj_rms_abs` round 1. Round-5 rows
are consistently positive across bases.

Interpretation:

This strengthens the process-specific ordering story beyond final-IMP
membership, final-oracle overlap, dense-final magnitude, and base-source
magnitude. The effect remains small, so it should be framed as a constrained
mechanistic control rather than as a standalone performance result.

## 2026-05-05 - Residual IMP-Process Round-Exclusion Control, 5 Seeds

Purpose:

Move from correlational score-source controls to a direct intervention on the
process-selected coordinates. For each base and process round, first select the
round-score final-IMP residual additions. Then remove those additions from the
final-IMP residual candidate set and choose the best remaining final-IMP
residual additions by final IMP magnitude under the same support budget.

Command:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 1,3,5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources epoch_30,traj_rms_abs,epoch_10 --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3/20260505_194457`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3/20260505_201224`
- `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3.md`

Result:

Across 45 paired base/round/seed comparisons, round-selected final-IMP
residual masks beat the round-excluded oracle replacement in 44 cases. The
mean paired accuracy delta is `+0.0061`. All nine base/round group means are
positive. The only seed-level reversal is `epoch_30` round 1 for seed 3.
Round-selected rows have oracle-overlap precision about `0.56/0.60/0.67` at
rounds 1/3/5, while the excluded replacement drops to about
`0.44/0.40/0.32`.

Interpretation:

This is the strongest current process-side evidence: even after fixing the
final-IMP residual pool and support budget, and allowing the replacement to use
final IMP magnitude as an oracle score, removing the process-selected
coordinates sharply degrades the constructed mask. The result supports a
functionally important process-selected subset inside the final IMP residual
pool, not just final-IMP membership or score-source correlation.

## 2026-05-05 - Direct Mode/Ticket Distribution Probe, Digits MLP

Purpose:

Evaluate the literal proposal-level distribution criteria rather than only
support-overlap summaries: layer-sparsity KS, RBF MMD, sliced Wasserstein,
pairwise mask-Hamming distribution overlap, logit-space CKA, and Hungarian
matching cost between posterior sample/mode masks and IMP tickets.

Command:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset digits --model mlp --hidden-dim 128 --depth 3 --seeds 0,1,2,3,4 --epochs 10 --imp-rounds 3 --prune-fraction 0.3 --samples 10 --sgld-steps 120 --sgld-burn-in 20 --sgld-sample-every 10 --batch-size 128 --lr 0.05 --weight-decay 1e-4 --out-dir runs/digits_mlp_mode_ticket_distribution_sgld_r5
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/digits_mlp_mode_ticket_distribution_sgld_r5 --out-md docs/digits_mlp_mode_ticket_distribution_sgld_r5.md --out-csv runs/digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv
```

Output:

- `runs/digits_mlp_mode_ticket_distribution_sgld_r5/20260505_221553`
- `runs/digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv`
- `docs/digits_mlp_mode_ticket_distribution_sgld_r5.md`

Result:

The probe collects 50 SGLD posterior sample masks and five IMP tickets at the
matched IMP sparsity. Mean-shift clustering in parameter PCA space collapses
the posterior samples to one representative mode, with basin entropy `0.0000`
and effective cluster count `1.0000`. Raw posterior sample masks
fail two of the proposal thresholds: layer-sparsity KS has `p=0.0788`, and
pairwise mask-Hamming distribution overlap is `0.6314`, below the proposed
`0.70` criterion. Logit-space CKA and Hungarian matching pass
(`CKA=0.9819`, `cost=0.0181`), so functional similarity alone does not imply
mask-distribution equivalence.

Interpretation:

This directly closes the small-model KS/MMD/Wasserstein/CKA/Hungarian gap in
the proposal audit. The result is negative for the strong 1:1 mode/ticket
claim: posterior modes do not match ticket diversity, and raw posterior mask
distributions fail the mask-level thresholds even though logits are highly
similar.

## 2026-05-05 - Mode/Ticket Distribution-Equivalence Audit

Purpose:

Make the original proposal's distributional equivalence test explicit using
existing posterior artifacts. The audit compares posterior-to-IMP support
overlap distributions against random masks and matched chain-start, dense, and
rewind magnitude controls with paired deltas plus KS, Wasserstein, and RBF-MMD
statistics.

Command:

```bash
.venv/bin/python scripts/run_mode_distribution_equivalence_audit.py
.venv/bin/python scripts/build_paper_stats.py
```

Output:

- `runs/mode_distribution_equivalence_audit_summary.csv`
- `runs/mode_distribution_equivalence_audit.json`
- `docs/mode_distribution_equivalence_audit.md`
- Integrated table `tab:mode-ticket-equivalence-audit` in
  `paper/tables/statistical_summary.tex`

Result:

After later full-network SWAG20, Hessian-plus-diagonal Laplace, random-subspace
HMC, trajectory-subspace HMC, rank-32/rank-64 low-rank Laplace, and five-seed Hessian-32
subspace-HMC integration, the
audit aggregates the current posterior artifacts into grouped comparisons.
Posterior supports beat random masks in 58/59 grouped comparisons. Against the
matched chain-start magnitude support, posterior wins by more than 0.005
Jaccard in 0/59 comparisons; 43/59 are practically tied to chain-start,
15/59 favor chain-start by more than 0.005, and one row is mixed. Rewind
magnitude beats posterior by more than 0.005 Jaccard in 55/57 grouped
comparisons.

Interpretation:

This directly closes the proposal's mask-distribution equivalence gap at the
support-overlap level for existing artifacts. The posterior-not-random signal is
real, but it is not a ticket-distribution equivalence signal: it is explained by
local chain-start or trajectory magnitude controls.

## 2026-05-05 - Variational Pruning Baseline, Digits 5 Seeds

Purpose:

Implement the proposal's explicit Bernoulli-mask variational pruning baseline.
The new baseline optimizes mask logits on frozen initialization weights using
Concrete samples, expected NLL, Bernoulli KL, sparsity penalty, and entropy
penalty, then selects a hard mask at the target sparsity and retrains it.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset digits --model mlp --hidden-dim 128 --depth 3 --seed "$seed" --epochs 10 --trajectory-epochs 0,10 --rewind-epochs 0 --imp-rounds 3 --prune-fraction 0.30 --batch-size 128 --lr 0.05 --lr-schedule constant --weight-decay 1e-4 --mask-train-epochs 10 --mask-sources imp,random,gem_miner,variational_prune --random-trials 1 --gem-miner-epochs 10 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-kl-weight 1e-4 --variational-prune-sparsity-weight 10 --variational-prune-entropy-weight 1e-3 --variational-prune-temperature-start 2.0 --variational-prune-temperature-end 0.2 --variational-prune-samples-per-batch 1 --variational-prune-max-batches-per-epoch 20 --out-dir runs/digits_mlp_variational_prune_calib_r5; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/digits_mlp_variational_prune_calib_r5 --out-csv runs/digits_mlp_variational_prune_calib_r5_summary.csv --out-md docs/digits_mlp_variational_prune_calib_r5.md
```

Output:

- `runs/digits_mlp_variational_prune_calib_r5/`
- `runs/digits_mlp_variational_prune_calib_r5_summary.csv`
- `docs/digits_mlp_variational_prune_calib_r5.md`
- `src/lottery/pruning_baselines.py`

Result:

Dense reaches 0.9739 accuracy, IMP reaches 0.9711, variational pruning reaches
0.9633, random masks reach 0.9489, and the Gem-Miner-style score baseline
reaches 0.9550. Variational-prune support is more IMP-like than random
(0.3822 vs. 0.2151 Jaccard), but still far from the IMP mask. Variational
pruning has ECE 0.0250 and Brier 0.0532; IMP has ECE 0.0211 and Brier 0.0410.

Interpretation:

This gives the project first direct H3 coverage. The variational objective is
useful relative to random and the existing score-training baseline on a small
model, but it does not match IMP and does not show the calibration advantage
predicted by the original positive proposal. CIFAR-scale variational pruning
and OOD diagnostics remain open.

## 2026-05-05 - Learned-Mask Calibration/OOD Smoke

Purpose:

Wire the learned mask sources into the existing calibration/OOD evaluator so H3
can be tested with the same ID calibration and OOD metrics as dense, IMP, and
SWAG. This is path validation, not a paper result.

Implementation:

- Added `--learned-mask-sources` to `scripts/run_calibration_ood_probe.py` with
  `random`, `gem_miner`, and `variational_prune` sources.
- Added fixed-mask retraining from the initial state before evaluating learned
  masks on ID and OOD loaders.
- Recorded learned-mask metadata in `metrics.json`.

Commands:

```bash
.venv/bin/python -m py_compile scripts/run_calibration_ood_probe.py scripts/summarize_calibration_ood_probe.py
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset fake-cifar10 --ood-dataset gaussian-noise --model resnet20 --resnet-width 4 --seed 0 --epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --ood-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 1 --swag-epochs 1 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 1 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 1 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --variational-prune-epochs 1 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 1 --variational-prune-samples-per-batch 1 --out-dir runs/fake_cifar10_calibration_ood_learned_masks_smoke
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/fake_cifar10_calibration_ood_learned_masks_smoke --out-csv runs/fake_cifar10_calibration_ood_learned_masks_smoke_summary.csv --out-md docs/fake_cifar10_calibration_ood_learned_masks_smoke.md
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset gaussian-noise --model resnet20 --resnet-width 4 --seed 0 --epochs 2 --imp-rounds 1 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --ood-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --samples 1 --swag-epochs 1 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 1 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 1 --gem-miner-epochs 1 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 1 --variational-prune-epochs 1 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 1 --variational-prune-samples-per-batch 1 --out-dir runs/cifar10_subset_calibration_ood_learned_masks_smoke
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_subset_calibration_ood_learned_masks_smoke --out-csv runs/cifar10_subset_calibration_ood_learned_masks_smoke_summary.csv --out-md docs/cifar10_subset_calibration_ood_learned_masks_smoke.md
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --resnet-width 8 --seed 0 --epochs 5 --imp-rounds 2 --prune-fraction 0.30 --batch-size 128 --train-subset 4096 --test-subset 2000 --ood-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 3 --swag-epochs 2 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 3 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 3 --gem-miner-epochs 3 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 10 --variational-prune-epochs 3 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 10 --variational-prune-samples-per-batch 1 --out-dir runs/cifar10_subset4096_calibration_ood_learned_masks_pilot
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_subset4096_calibration_ood_learned_masks_pilot --out-csv runs/cifar10_subset4096_calibration_ood_learned_masks_pilot_summary.csv --out-md docs/cifar10_subset4096_calibration_ood_learned_masks_pilot.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_calibration_ood_probe.py --dataset cifar10 --ood-dataset cifar100 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --swag-epochs 5 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --samples 10 --learned-mask-sources random,gem_miner,variational_prune --learned-random-trials 1 --mask-train-epochs 30 --gem-miner-epochs 10 --gem-miner-lr 0.1 --gem-miner-freeze-period 1 --gem-miner-max-batches-per-epoch 20 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-max-batches-per-epoch 20 --variational-prune-samples-per-batch 1 --out-dir runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3; done
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3.md
```

Output:

- `runs/fake_cifar10_calibration_ood_learned_masks_smoke/20260505_223354/metrics.json`
- `runs/fake_cifar10_calibration_ood_learned_masks_smoke_summary.csv`
- `docs/fake_cifar10_calibration_ood_learned_masks_smoke.md`
- `runs/cifar10_subset_calibration_ood_learned_masks_smoke/20260505_223548/metrics.json`
- `runs/cifar10_subset_calibration_ood_learned_masks_smoke_summary.csv`
- `docs/cifar10_subset_calibration_ood_learned_masks_smoke.md`
- `runs/cifar10_subset4096_calibration_ood_learned_masks_pilot/20260505_223941/metrics.json`
- `runs/cifar10_subset4096_calibration_ood_learned_masks_pilot_summary.csv`
- `docs/cifar10_subset4096_calibration_ood_learned_masks_pilot.md`
- `runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_calibration_ood_learned_masks_r5_p0p3.md`

Result:

Both fake-CIFAR and real CIFAR-subset smoke paths complete and include dense,
IMP, SWAG ensemble/member, learned-random, Gem-Miner-style, and
variational-prune sources. In the real subset smoke, accuracies are near chance
because the run uses only 512 train examples and two epochs: dense `0.0938`,
IMP `0.0664`, learned-random `0.0938`, Gem-Miner-style `0.0898`, and
variational-prune `0.0781`. The learned-source metrics are therefore not
interpretable as H3 evidence.

The larger seed-0 pilot uses 4096 CIFAR-10 train examples, 2000 CIFAR-10 test
examples, and 2000 CIFAR-100 OOD examples. It gives a non-chance sanity signal:
IMP reaches accuracy `0.5355`, NLL `1.2984`, ECE `0.0306`, and MSP OOD AUROC
`0.6198`. Dense reaches accuracy `0.3965` and MSP OOD AUROC `0.5783`.
Learned-random, Gem-Miner-style, and variational-prune masks reach accuracies
`0.2905`, `0.3255`, and `0.2535`; their MSP OOD AUROCs are `0.5736`, `0.5267`,
and `0.5651`.

The full-data five-seed 30-epoch epoch-1 rewind CIFAR-10/CIFAR-100 learned-mask
row completes. Learned-random, Gem-Miner-style, and variational-prune masks
reach accuracies `0.8449`, `0.8418`, and `0.8301`, NLLs `0.4632`, `0.4694`,
and `0.5008`, Brier scores `0.2246`, `0.2280`, and `0.2436`, ECE `0.0270`,
`0.0283`, and `0.0255`, and MSP OOD AUROCs `0.7897`, `0.7853`, and `0.7754`.
In the canonical dense/IMP/SWAG calibration row, IMP reaches accuracy `0.8953`,
NLL `0.3387`, Brier `0.1583`, ECE `0.0393`, and MSP OOD AUROC `0.8306`.

Interpretation:

The full CIFAR learned-mask calibration/OOD row does not rescue H3. Learned
masks can reduce ECE, especially variational pruning, but they lose accuracy,
NLL, Brier, and OOD AUROC relative to IMP. This supports the paper's separation
claim: calibration improvements are not sufficient evidence for ticket-support
or posterior-mode equivalence.

## 2026-05-06 - CIFAR Variational-Pruning Support Row

Purpose:

Close the remaining CIFAR-scale learned-mask support gap for the proposal's
explicit Bernoulli/Concrete variational-pruning baseline. The prior
calibration/OOD row showed a calibration tradeoff, but did not measure support
overlap against the IMP ticket.

Commands:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources variational_prune --random-trials 0 --variational-prune-epochs 10 --variational-prune-lr 0.01 --variational-prune-kl-weight 1e-4 --variational-prune-sparsity-weight 10 --variational-prune-entropy-weight 1e-3 --variational-prune-temperature-start 2.0 --variational-prune-temperature-end 0.2 --variational-prune-samples-per-batch 1 --variational-prune-max-batches-per-epoch 20 --out-dir runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3/`
- `runs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_variational_prune_selected_r5_p0p3.md`

Result:

Across five full-data seeds at 0.832 sparsity, variational pruning reaches
`0.8306` retrained accuracy, `0.0284` ECE, and `0.2433` Brier. It is `0.0669`
below IMP accuracy and `0.0533` below dense accuracy. Its support overlap to
IMP is `0.0907` Jaccard, essentially random-scale and slightly below the
random-mask row (`0.0922`) and Gem-Miner-style row (`0.0917`) in the matched
CIFAR trajectory-mask table.

Interpretation:

The CIFAR support row closes the main learned-mask support gap for H3. The
proposal-style variational objective does not learn a ticket-like support on
this CIFAR setting; it has lower ECE but worse accuracy, Brier, OOD AUROC, and
IMP support overlap than the IMP ticket.

## 2026-05-06 - CIFAR Subset Direct Mode/Ticket Distribution Probe

Purpose:

Extend the literal proposal-level KS/MMD/Wasserstein/Hamming/CKA test beyond
digits MLP to a CNN/CIFAR setting. This is a width-8 ResNet-20 subset pilot,
not a full CIFAR posterior-mode pipeline, but it checks whether the direct
metrics fail in the same way outside the small tabular-image model.

Commands:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 8 --seeds 0,1,2,3,4 --epochs 5 --imp-rounds 2 --prune-fraction 0.3 --samples 5 --sgld-steps 80 --sgld-burn-in 20 --sgld-sample-every 12 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 4096 --test-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 10 --sliced-projections 64 --out-dir runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5 --out-md docs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5.md --out-csv runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5_summary.csv
```

Output:

- `runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5/20260506_001038`
- `runs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5_summary.csv`
- `docs/cifar10_subset4096_resnet20_w8_mode_ticket_distribution_sgld_r5.md`

Result:

The probe compares 25 SGLD posterior sample masks with five IMP tickets at
0.510 sparsity. IMP accuracies are nontrivial for the subset (`0.4590` to
`0.4875`) and above dense in every seed. Raw posterior sample masks fail two of
the four proposal thresholds: layer-sparsity KS has `p=1.10e-05`, and
pairwise mask-Hamming distribution overlap is `0.6000`, below the proposed
`0.70` criterion. Logit-space CKA and Hungarian matching pass (`CKA=0.8878`,
`cost=0.1122`). Mean-shift clustering again collapses the posterior samples to
one representative mode versus five tickets, with basin entropy `0.0000` and
effective cluster count `1.0000`.

Interpretation:

The CIFAR subset row reproduces the digits pattern: function-space matching can
look good while mask-distribution criteria fail. This strengthens the
proposal-level negative result beyond MLPs, while still leaving full-data
CIFAR permutation-aligned mode clustering as open work.

## 2026-05-06 - CIFAR Subset Activation-CKA Mode/Ticket Pilot

Purpose:

Check whether adding final-hidden activation CKA changes the direct
mode/ticket conclusion on the CIFAR subset.

Commands:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 8 --seeds 0,1,2 --epochs 5 --rewind-epochs 1 --imp-rounds 2 --prune-fraction 0.3 --samples 5 --sgld-steps 80 --sgld-burn-in 20 --sgld-sample-every 12 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 4096 --test-subset 2000 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 10 --sliced-projections 64 --out-dir runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot --out-md docs/cifar10_subset4096_mode_ticket_distribution_activation_pilot.md --out-csv runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot_summary.csv
```

Output:

- `runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot/20260506_001418`
- `runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot_summary.csv`
- `docs/cifar10_subset4096_mode_ticket_distribution_activation_pilot.md`

Result:

The activation pilot compares 15 SGLD posterior sample masks with three IMP
tickets. Raw posterior sample masks still fail layer KS (`p=0.0330`) and
Hamming overlap (`0.6571 < 0.70`) while logit and final-hidden activation CKA
pass (`0.9022` and `0.8726`). Mean-shift clustering yields three equal-size
clusters, basin entropy `1.0986` nats, normalized entropy `0.4057`, and
effective cluster count `3.0000`; the collapsed mode/ticket comparison passes
all current thresholds. This is useful as a robustness pilot, but it is
three-seed subset evidence and does not close the full-data,
permutation-aligned CIFAR mode-clustering gap.

## 2026-05-06 - Reproducibility and Paper Polish Pass

Purpose:

Harden the project for repeated paper generation while the full-data CIFAR
mode/ticket probe runs.

Changes:

- Added `requirements.txt` with the core Python dependencies.
- Added `Makefile` targets: `check`, `stats`, `verify`, `figures`, `paper`,
  `paper-check`, and `clean`.
- Added `scripts/verify_research_artifacts.py`, which checks that
  `runs/paper_stats.json` contains the core Gate1, mode-distribution,
  direct mode/ticket, calibration/OOD, learned-mask, and residual-process
  evidence rows used by the manuscript.
- Added `docs/environment_snapshot.md` with the local Python, CUDA, package,
  GPU, and LaTeX versions used for the current build.
- Shortened the paper abstract from a long experiment inventory to a concise
  statement of the hypothesis, negative evidence, mechanistic alternative,
  and open CIFAR posterior limitation.
- Added an explicit contribution paragraph in the introduction and rewrote the
  limitations section around the remaining direct full-data mode-clustering,
  high-fidelity posterior, learned-mask, process-causality, and reproducibility
  gaps.
- Moved the large generated evidence tables into an appendix section so the
  main narrative is not interrupted by the full table dump.
- Updated `scripts/run_mode_ticket_distribution_probe.py` so future long
  mode/ticket runs create the run directory at launch, write
  `run_metadata.json`, and update `partial_seed_summaries.json` after each
  seed.

Verification:

```bash
make check
make paper-check
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 1 --sgld-steps 2 --sgld-burn-in 0 --sgld-sample-every 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --out-dir /tmp/lottery_mode_ticket_partial_smoke_status
```

Result:

`make check` passed. `make paper-check` rebuilt `paper/main.pdf` and enforced
the LaTeX log scan, ignoring only the `rerunfilecheck` package-loading line.
The CPU smoke created `run_metadata.json`, a complete-status
`partial_seed_summaries.json`, `metrics.json`, and entropy-bearing
`mode_ticket_distribution_summary.csv`.

## 2026-05-06 - Full-Data CIFAR Direct Mode/Ticket Activation Probe

Purpose:

Close the remaining direct proposal-metric gap on the canonical full-data
CIFAR-10 ResNet-20 epoch-1 rewind setting. This extends the direct
KS/MMD/Wasserstein/Hamming/logit-CKA test with final-hidden activation CKA and
raw parameter-PCA basin entropy.

Commands:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3/20260506_004811`
- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3.md`

Result:

Across five full-data seeds, dense accuracy is `0.8859` mean, IMP accuracy is
`0.8977` mean, and SGLD posterior sample accuracy is `0.8785` mean. Mean-shift
clustering collapses all 50 posterior samples to one raw parameter-PCA basin
(`H_n=0.0000`, effective clusters `1.0000`) versus five IMP tickets. Raw
posterior sample masks fail the proposal's mask-distribution thresholds:
layer-sparsity KS has `p=5.34e-09`, and mask-Hamming overlap is `0.0033`.
The single mode representative also fails layer KS (`p=0.0329`), with Hamming
overlap undefined because only one representative exists. In contrast,
logit-space CKA and final-hidden activation CKA pass for both comparisons:
sample/ticket CKA is `0.9369` logit and `0.9172` activation; mode/ticket CKA is
`0.9383` logit and `0.9179` activation.

Interpretation:

The full-data row closes the direct CIFAR mode/ticket distribution gap in the
negative direction. High logit and activation similarity does not imply
mask-distribution agreement or a one-to-one basin/ticket count. The smaller
three-seed activation subset pilot remains useful as a robustness warning: its
collapsed mode representatives pass the current thresholds, but the full-data
row does not reproduce that positive-looking subset artifact.

## 2026-05-06 - Activation-Aligned Mode/Ticket Probe Path

Purpose:

Close the remaining permutation/channel-alignment objection for the direct
mode/ticket probe. The new path keeps the raw comparisons and adds
`activation_aligned_*` comparisons that map every ResNet mask into the first
seed dense-model channel frame by activation-correlation Hungarian matching.
Aligned posterior samples are also clustered in the aligned parameter-PCA
space, so basin entropy can be compared before and after channel alignment.

Implementation:

- Added `--alignment-method {none,activation}` and `--alignment-batches` to
  `scripts/run_mode_ticket_distribution_probe.py`.
- Reused the ResNet activation feature keys and weight-axis mapping already
  used by the activation-aligned residual transfer probe.
- Added aligned posterior/ticket masks, aligned posterior state vectors,
  aligned posterior clustering, alignment metadata, and
  `activation_aligned_posterior_samples_vs_tickets` /
  `activation_aligned_posterior_modes_vs_tickets` comparison rows.
- Updated `scripts/summarize_mode_ticket_distribution_probe.py` so aligned rows
  report the aligned cluster count and entropy.
- Added the aligned full-data summary path to `scripts/build_paper_stats.py`.

Smoke commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0,1 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 1 --sgld-steps 2 --sgld-burn-in 0 --sgld-sample-every 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --alignment-method activation --alignment-batches 1 --out-dir /tmp/lottery_mode_ticket_alignment_smoke
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 4 --seeds 0,1 --epochs 2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 2 --sgld-steps 6 --sgld-burn-in 0 --sgld-sample-every 2 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 4 --sliced-projections 16 --alignment-method activation --alignment-batches 1 --out-dir runs/cifar10_subset_alignment_mode_ticket_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset_alignment_mode_ticket_smoke --out-md docs/cifar10_subset_alignment_mode_ticket_smoke.md --out-csv runs/cifar10_subset_alignment_mode_ticket_smoke_summary.csv
```

Smoke result:

Both smokes completed. The real CIFAR subset smoke writes
`docs/cifar10_subset_alignment_mode_ticket_smoke.md` and
`runs/cifar10_subset_alignment_mode_ticket_smoke_summary.csv`. It produces both
raw and `activation_aligned_*` rows; raw and aligned posterior clustering both
collapse to one basin in this tiny setting. The smoke is only a path check, not
evidence for the paper claim.

Full-data command:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --alignment-method activation --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3
```

Full-data result:

- Run directory:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3/20260506_005822`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3.md`

The five-seed full-data activation-aligned row preserves the negative result.
Both raw and aligned posterior samples collapse to one posterior basin
(`H_n=0`, effective clusters `1.0`) versus five IMP tickets. Aligned posterior
sample masks fail layer KS (`p=2.32e-09`) and Hamming overlap (`0.0000 < 0.70`)
while passing logit CKA (`0.9373`), final-hidden activation CKA (`0.9168`), and
both Hungarian-cost thresholds. The single aligned mode representative also
fails layer KS (`p=0.0413`). Thus the original direct mode/ticket failure is
not explained by a simple ResNet channel-permutation mismatch.

## 2026-05-06 - Multi-Chain Cyclical-SGLD Direct Mode/Ticket Probe

Purpose:

Reduce the remaining "weak posterior sampler" objection for the full-data
CIFAR direct mode/ticket probe. The run adds three dense-start cyclical-SGLD
chains per seed, records chain-start magnitude controls, and writes
posterior-to-chain-start Hamming diagnostics alongside the proposal metrics.

Implementation:

- Added `--posterior-chains` / `--posterior-chain-init` aliases to
  `scripts/run_mode_ticket_distribution_probe.py`.
- Added `chain_start_magnitude_vs_tickets` comparison rows and
  `posterior_chain_diagnostics` to `metrics.json`.
- Updated `scripts/summarize_mode_ticket_distribution_probe.py` so chain-start
  rows use chain-start clustering and reports the posterior-to-chain-start
  Hamming diagnostic.

Smoke commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 1 --posterior-chains 2 --posterior-chain-init dense --sgld-steps 2 --sgld-burn-in 0 --sgld-sample-every 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --alignment-method activation --alignment-batches 1 --out-dir /tmp/lottery_mode_ticket_multichain_smoke
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 4 --seeds 0,1 --epochs 2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --posterior-sampler cyclical-sgld --samples 2 --posterior-chains 2 --posterior-chain-init dense --sgld-steps 12 --sgld-burn-in 0 --sgld-sample-every 2 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 6 --csgld-sample-phase-start 0.5 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 4 --sliced-projections 16 --out-dir runs/cifar10_subset_multichain_csgld_mode_ticket_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset_multichain_csgld_mode_ticket_smoke --out-md docs/cifar10_subset_multichain_csgld_mode_ticket_smoke.md --out-csv runs/cifar10_subset_multichain_csgld_mode_ticket_smoke_summary.csv
```

Full-data command:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler cyclical-sgld --samples 5 --posterior-chains 3 --posterior-chain-init dense --sgld-steps 400 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv
```

Full-data result:

- Run directory:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3/20260506_014145`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3.md`

The multi-chain run collects 75 posterior samples from three dense-start
cyclical-SGLD chains per seed. Samples keep mean accuracy `0.8760` while moving
from the chain-start masks (`posterior-to-chain-start` Hamming mean `0.0443`,
median `0.0456`). Even with this movement, the direct H1 metrics fail:
posterior samples collapse to one parameter-PCA basin, layer KS is
`p=3.33e-08`, Hamming-distribution overlap is `0.2461 < 0.70`, and the single
representative fails layer KS (`p=0.0413`). Logit and final-hidden activation
CKA remain high (`0.9327` and `0.9144`), so the result reinforces the
separation between functional similarity and mask-distribution equivalence.

## 2026-05-06 - Full-Network SWAG Movement Diagnostic

Purpose:

Add a broader full-network low-rank-plus-diagonal Gaussian posterior baseline to
the CIFAR movement diagnostics. This closes a gap between the old five-epoch
canonical SWAG support row and the movement-grid rows used for SGLD-family and
Laplace samplers.

Implementation:

- Split `src/lottery/swag.py` into `fit_swag_posterior` and
  `sample_swag_posterior`, while keeping `collect_swag_samples` compatible.
- Added `--posterior-sampler swag` and SWAG scale/fit controls to
  `scripts/run_sgld_movement_grid.py`.
- Updated `scripts/build_paper_stats.py`,
  `scripts/run_mode_distribution_equivalence_audit.py`, and
  `scripts/build_paper_figures.py` to include the selected SWAG20 movement row.

Smoke commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --posterior-sampler swag --swag-scales 0.5,1.0 --swag-epochs 1 --swag-lr 0.005 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 2 --samples 2 --random-trials 5 --out-dir runs/fake_cifar10_swag_movement_shared_smoke
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/fake_cifar10_swag_movement_shared_smoke --out-csv runs/fake_cifar10_swag_movement_shared_smoke_summary.csv --out-md docs/fake_cifar10_swag_movement_shared_smoke.md
```

Tuning:

Seed-0 20-snapshot SWAG with `swag_lr=0.01` moved support but damaged sample
accuracy too much (`0.7469` already at scale `0.25`). Lowering to
`swag_lr=0.001` preserved sample accuracy: scale `16` had post-chain `0.9561`,
posterior-chain `-0.00025`, and sample accuracy `0.8751`.

Selected command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler swag --swag-scales 1.0,16.0,64.0 --swag-epochs 20 --swag-lr 0.001 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 20 --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3.md
```

Selected result:

- Run root:
  `runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_swag_movement_selected20_lr1e3_r5_p0p3.md`

At scale `16`, posterior-to-chain-start overlap drops to `0.9528`, sample
accuracy is `0.8636`, and posterior-chain is `-7.0e-05` with a 95% CI spanning
zero. At scale `64`, post-chain drops further to `0.9086`, sample accuracy is
`0.8041`, and posterior-chain remains negative (`-0.00018`). Rewind magnitude
beats posterior by about `0.0328--0.0329` Jaccard. The full-network SWAG row
therefore behaves like a local chain-start magnitude perturbation, not a
ticket-directed posterior mode.

## 2026-05-06 - Low-Rank Hessian-Plus-Diagonal Laplace Movement Diagnostic

Purpose:

Add a full-network curvature-informed Gaussian posterior baseline with explicit
low-rank covariance over the trainable parameter vector. This narrows the gap
between diagonal/KFAC Laplace controls and exact full-network covariance.

Implementation:

- Added `src/lottery/lowrank_laplace.py`.
- Added `--posterior-sampler lowrank-laplace` and low-rank Laplace controls to
  `scripts/run_sgld_movement_grid.py`.
- The sampler estimates a diagonal empirical-Fisher precision over 20
  minibatches, adds a randomized rank-16 top-Hessian precision estimated from
  two minibatches, and samples the full-vector Gaussian using a Woodbury
  low-rank correction.

Smoke commands:

```bash
.venv/bin/python -m py_compile src/lottery/lowrank_laplace.py scripts/run_sgld_movement_grid.py scripts/summarize_sgld_movement_grid.py
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3 --lowrank-laplace-rank 2 --lowrank-laplace-oversample 1 --lowrank-laplace-power-iterations 0 --lowrank-laplace-fisher-batches 1 --lowrank-laplace-hessian-batches 1 --samples 2 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --random-trials 4 --out-dir /tmp/lottery_lowrank_laplace_smoke
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root /tmp/lottery_lowrank_laplace_smoke --out-csv /tmp/lottery_lowrank_laplace_smoke_summary.csv --out-md /tmp/lottery_lowrank_laplace_smoke.md
```

Selected command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 16 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 4 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3.md
```

Selected result:

- Run root:
  `runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3.md`

All five seeds retained 16 positive top-Hessian directions. At scale `1e-3`,
posterior-to-chain-start overlap is `0.9295`, sample accuracy is `0.8848`,
and posterior-to-IMP is `0.1447`, below chain-start support `0.1456`. At scale
`1e-2`, posterior-to-chain-start overlap drops to `0.7359`, but
posterior-to-IMP drops to `0.1351` and sample accuracy to `0.8784`. Rewind
magnitude remains higher at `0.1777`. The new full-network correlated Gaussian
therefore does not rescue posterior-mode/ticket support equivalence.

## 2026-05-06 - Tensor-Matched Residual IMP-Process Round-Exclusion Control

Purpose:

Test whether the strong round-exclusion result is explained by coarse parameter
tensor composition. The control removes the process-selected round-5 final-IMP
residual additions, then selects replacement final-IMP residual additions
matched by parameter tensor and ranked by final IMP magnitude.

Implementation:

- Extended `scripts/build_paper_stats.py` with
  `residual_imp_process_layer_exclusion` and paired tensor-matched
  round-exclusion summaries.
- Extended `scripts/verify_research_artifacts.py` so the RMS trajectory
  round-5 tensor-matched row must be five-seed and must favor the
  round-selected process residual.

Commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3
for seed in 1 2 3 4; do .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds "$seed" --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3; done
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3.md
```

Artifacts:

- Run root:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3.md`

Result:

For the RMS trajectory base at process round 5, round-selected final-IMP
residual masks reach `0.8855` accuracy. The unrestricted round-excluded
replacement reaches `0.8753`, and the tensor-matched round-excluded
replacement reaches `0.8764`. Round-selected masks beat the unrestricted
excluded replacement by `+0.0103` and the tensor-matched replacement by
`+0.0091`, with `5/5` positive paired seed deltas in both comparisons.
The tensor-matched replacement also has much lower final-oracle overlap
(`0.4153`) than the process-selected row (`0.6768`). This strengthens the
process-specific residual interpretation beyond parameter-tensor composition.

## 2026-05-06 - Tensor+Score-Matched Residual IMP-Process Round-Exclusion Control

Purpose:

Test whether the strongest tensor-matched process result is still explained by
within-tensor round-score composition. The control removes the process-selected
round-5 final-IMP residual additions, then selects replacement final-IMP
residual additions matched by parameter tensor and within-tensor round-score
decile before applying final IMP magnitude.

Implementation:

- Extended `scripts/build_paper_stats.py` with
  `residual_imp_process_tensor_score_exclusion` and paired tensor+score
  round-exclusion summaries.
- Extended `scripts/verify_research_artifacts.py` so the RMS trajectory
  round-5 tensor+score-matched row must be five-seed, must favor the
  round-selected process residual, and must improve final-oracle overlap over
  tensor-only matching.

Commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-excluded-oracle,final-imp-round-excluded-layer-oracle,final-imp-round-excluded-tensor-score-oracle --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md
```

Artifacts:

- Run root:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3`
- Summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv`
- Report:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md`

Result:

For the RMS trajectory base at process round 5, round-selected final-IMP
residual masks reach `0.8878` accuracy. The unrestricted round-excluded
replacement reaches `0.8781`, the tensor-matched replacement reaches `0.8784`,
and the tensor+score-matched replacement reaches `0.8837`. The tensor+score
replacement narrows the gap and raises final-oracle overlap to `0.6440` versus
`0.4167` for tensor-only matching, but round-selected masks still beat it by
`+0.0041` accuracy with `5/5` positive paired seed deltas. This closes the
stronger tensor-plus-score-bin composition objection for the strongest process
row.

## 2026-05-06 - Project-Critical Environment Lock

Purpose:

Reduce the remaining reproducibility gap by turning the environment snapshot
into a checked lock for the package/runtime versions that affect generated
statistics and the paper build.

Implementation:

- Added `requirements-lock.txt` with pinned project-critical Python packages.
- Added `docs/environment_lock.json` with Python, platform, PyTorch CUDA, TeX,
  and package versions.
- Added `scripts/check_environment_lock.py`.
- Wired `make env-check` and `make check` to run the lock check before trusting
  generated paper statistics.
- Extended `scripts/verify_research_artifacts.py` to require the lock artifacts
  and their core package pins.

Result:

`make check` now fails early if the active runtime differs from the locked
Python/package/CUDA/TeX environment. At that point this did not replace CI or
the later artifact container, but it removed the previous unverified-Python-lock
gap for local artifact regeneration.

## 2026-05-06 - Public Release Manifest and CI Check Split

Purpose:

Define the minimum public package for this negative-result paper and avoid
mixing the local CUDA/TeX environment lock with lightweight CI checks.

Implementation:

- Added `scripts/build_release_manifest.py`.
- Added `docs/public_release_manifest.md` and
  `runs/public_release_manifest.json`.
- Added `make release-manifest`.
- Split `make check` and `make ci-check`: local `check` verifies the
  environment lock, while CI uses `ci-check` so it can verify generated
  statistics and release inventory without requiring the local GPU/TeX stack.
- Updated `.github/workflows/check.yml` to run `make ci-check PYTHON=python`.
- Extended `scripts/verify_research_artifacts.py` to require the release
  manifest and verify SHA256 hashes for required release artifacts.

Result:

The release manifest currently tracks `1436` files and `13.8 MiB` of code,
docs, paper files, and JSON/CSV run artifacts. Required artifacts such as
`paper/main.pdf`, `runs/paper_stats.json`, environment locks, CI config, and
the core audits are SHA256-checked by the verifier. A real external green CI
run still requires pushing this directory to a repository, which is not
possible to certify inside the current non-git workspace.

## 2026-05-06 - CPU Artifact Container Lock

Purpose:

Close the missing container/OS-lock gap for artifact verification without
pretending to lock the GPU training stack.

Implementation:

- Added `Dockerfile` pinned to the linux/amd64
  `python:3.12.3-slim-bookworm` manifest digest.
- Added `.dockerignore` to exclude the local virtualenv, raw datasets, and
  transient LaTeX/cache outputs.
- Added `docs/container_lock.md` documenting the base digest, TeX packages,
  scope, and limitations.
- Added `make container-build` and `make container-check`.
- Extended the public release manifest and verifier to require the container
  lock artifacts.

Result:

The repository now has a CPU-only container path for `make ci-check paper-check
PYTHON=python`. This verifies paper statistics, the release manifest, and the
paper build from included artifacts. It does not replace the local CUDA/PyTorch
environment lock for full CIFAR training or provide an externally observed CI
run.

## 2026-05-06 - CIFAR Full-Covariance Feasibility Audit

Purpose:

Quantify the cost of the literal exact full-network CIFAR posterior baseline so
the paper's remaining posterior limitation is stated as a resource-bounded gap
rather than a vague omission.

Implementation:

- Added `scripts/audit_full_covariance_feasibility.py`.
- Generated `runs/cifar10_resnet20_full_covariance_feasibility.json`.
- Generated `docs/cifar10_resnet20_full_covariance_feasibility.md`.
- Wired local `make check` to regenerate the audit.
- Extended the verifier and public release manifest to require the audit
  artifacts.

Result:

For the CIFAR-10 ResNet-20 used in the paper, exact dense covariance over all
trainable parameters has 272,474 dimensions and needs `553.1` GiB for one
float64 matrix, or `1,106.3` GiB if the matrix and Cholesky factor are both
resident. A tensor-block-diagonal exact covariance over weight tensors still
needs `113.5` GiB with Cholesky, dominated by 36,864-parameter stage-3
convolution tensors. This does not close the empirical full-covariance gap, but
it justifies why the submission relies on exact head/selected-block covariance
plus full-network low-rank/SWAG/HMC approximations.

## 2026-05-06 - Paper Claim Ledger

Purpose:

Harden the paper for review by mapping each central manuscript claim to the
exact artifact and numerical rule that supports it.

Implementation:

- Added `scripts/build_paper_claim_ledger.py`.
- Generated `docs/paper_claim_ledger.md`.
- Added `make claim-ledger` and wired the ledger into `make check`,
  `make ci-check`, `make verify`, `scripts/verify_research_artifacts.py`, and
  `scripts/build_release_manifest.py`.
- Updated the reproducibility manifest and README to point to the ledger.

Result:

The ledger now records nine claim rows: Gate1 magnitude-control failure,
proposal-level support-distribution audit counts, full-data direct CIFAR
mode/ticket failures including activation- and weight-aligned rows,
rank-16/rank-32/rank-64/rank-128 full-network low-rank Laplace movement, calibration/OOD
non-rescue, learned-mask support non-rescue, tensor+score IMP-process
round-exclusion, residualized-score projection, and the bounded
full-covariance feasibility limitation. Eight rows must pass concrete numeric
rules, while the exact full-covariance item is explicitly marked as an open but
bounded limitation.

## 2026-05-06 - CIFAR Rank-32 Low-Rank Laplace Movement

Purpose:

Harden the posterior-side rebuttal by doubling the full-network low-rank
Hessian-plus-diagonal Laplace rank from 16 to 32 under the same five-seed
CIFAR-10 ResNet-20 epoch-1 rewind protocol.

Commands:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 32 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 8 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3.md
.venv/bin/python scripts/run_mode_distribution_equivalence_audit.py
.venv/bin/python scripts/build_paper_stats.py
.venv/bin/python scripts/build_paper_claim_ledger.py
```

Result:

All five seeds retained 32 positive Hessian directions. At scale `1e-2`,
posterior-to-chain-start overlap falls to `0.7402`, but posterior-to-IMP is
`0.1358`, below chain-start support `0.1457` and rewind magnitude `0.1795`;
sample accuracy is `0.8789`.

## 2026-05-06 - Five-Seed Hessian-32 Subspace HMC Selected Probe

Command:
```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_subspace_hmc_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --subspace-dims 32 --hmc-basis hessian --hmc-step-sizes 3e-4 --hmc-steps 8 --hmc-leapfrog-steps 2 --hmc-burn-in 2 --hmc-sample-every 2 --hmc-prior-precision 1e-4 --hmc-direction-scale 1.0 --hmc-batchnorm-mode eval --hessian-batches 2 --hessian-power-iterations 1 --hessian-oversample 8 --random-trials 50 --snip-batches 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_subspace_hmc_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3.md
```

Interpretation:
The 32-dimensional top-Hessian selected row accepts at `0.9500`, keeps sample
accuracy at `0.8872`, and moves parameters by `0.0104`, but support remains
tied to the chain-start magnitude control: posterior-to-IMP is `0.14614`
versus chain-start `0.14611`, with post-chain overlap `0.9993`. Rewind
magnitude remains closer at `0.17876`. The updated distribution-equivalence
audit now reports posterior-random `58/59`, posterior-chain `0/59`, and
rewind-over-posterior `55/57`, preserving the negative support-equivalence
conclusion.

## 2026-05-06 - Residual IMP-Process Round-Score Projection Control

Purpose:

Test whether the useful round-trained ordering inside the final-IMP residual
candidate pool survives after removing the linear magnitude subspace spanned by
the RMS trajectory base score, dense-final magnitude, and final-IMP magnitude.

Commands:

```bash
.venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-round-residualized-score,final-imp-dense-score,final-imp-base-score --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3.md
```

Result:

At the RMS trajectory base, round 5, and alpha 0.5, the original round-score
final-IMP residual mask reaches `0.8852` accuracy. The residualized-score mask,
after projecting out base/dense/final-IMP magnitude scores inside the same
candidate pool, reaches `0.8811`. The paired round-minus-residualized delta is
`+0.0041` with `5/5` positive seed deltas. Final-oracle overlap drops from
`0.6684` to `0.4854`. This localizes the useful process ordering to an
interaction with trajectory/final magnitude structure rather than a standalone
orthogonal residual score.

## 2026-05-06 - Hard-Concrete Learned-Mask Path

Purpose:

Close part of the learned-mask robustness gap by adding a stronger
hard-concrete L0 gate baseline alongside the existing Gem-Miner-style STE and
Bernoulli/Concrete variational-pruning mask sources.

Commands:

```bash
.venv/bin/python -m py_compile src/lottery/pruning_baselines.py scripts/run_trajectory_mask_training_probe.py scripts/run_calibration_ood_probe.py scripts/summarize_trajectory_mask_training_probe.py
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 2 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --mask-train-epochs 1 --mask-sources hard_concrete --random-trials 0 --hard-concrete-epochs 1 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 1 --hard-concrete-samples-per-batch 1 --out-dir runs/fake_cifar10_hard_concrete_mask_training_smoke_v3
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/fake_cifar10_hard_concrete_mask_training_smoke_v3 --out-csv runs/fake_cifar10_hard_concrete_mask_training_smoke_v3_summary.csv --out-md docs/fake_cifar10_hard_concrete_mask_training_smoke.md
.venv/bin/python scripts/run_calibration_ood_probe.py --dataset fake-cifar10 --ood-dataset gaussian-noise --model resnet20 --resnet-width 2 --seed 0 --epochs 2 --imp-rounds 2 --prune-fraction 0.30 --batch-size 64 --train-subset 128 --test-subset 64 --ood-subset 64 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 1 --swag-epochs 1 --swag-lr 0.01 --swag-collection-start-epoch 1 --swag-sample-every-epochs 1 --swag-max-snapshots 1 --learned-mask-sources hard_concrete --mask-train-epochs 1 --hard-concrete-epochs 1 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 1 --hard-concrete-samples-per-batch 1 --out-dir runs/fake_cifar10_calibration_ood_hard_concrete_smoke
.venv/bin/python scripts/summarize_calibration_ood_probe.py --run-root runs/fake_cifar10_calibration_ood_hard_concrete_smoke --out-csv runs/fake_cifar10_calibration_ood_hard_concrete_smoke_summary.csv --out-md docs/fake_cifar10_calibration_ood_hard_concrete_smoke.md
.venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --resnet-width 4 --seed 0 --epochs 2 --trajectory-epochs 0,1,2 --rewind-epochs 1 --imp-rounds 2 --prune-fraction 0.30 --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --augment --mask-train-epochs 1 --mask-sources hard_concrete --random-trials 0 --hard-concrete-epochs 1 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 1 --hard-concrete-samples-per-batch 1 --out-dir runs/cifar10_subset_hard_concrete_mask_training_smoke
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_subset_hard_concrete_mask_training_smoke --out-csv runs/cifar10_subset_hard_concrete_mask_training_smoke_summary.csv --out-md docs/cifar10_subset_hard_concrete_mask_training_smoke.md
```

Result:

The hard-concrete path now works in both support and calibration/OOD probes.
The real CIFAR subset smoke produces a sparse hard-concrete mask at `0.5099`
sparsity with `0.3216` Jaccard overlap to IMP and `0.0781` one-epoch
fixed-mask accuracy. This is path validation only; the five-seed full-data
CIFAR row remained an open learned-mask robustness item until the selected row
below.

## 2026-05-06 - Five-Seed Hard-Concrete Learned-Mask Support Row

Purpose:

Turn the hard-concrete L0 learned-mask path from smoke-only validation into a
matched full-data CIFAR support row against IMP.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_trajectory_mask_training_probe.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --trajectory-epochs 0,1,30 --rewind-epochs 1 --imp-epochs 30 --imp-final-epochs 30 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --mask-train-epochs 30 --mask-sources hard_concrete --random-trials 0 --hard-concrete-epochs 10 --hard-concrete-lr 0.01 --hard-concrete-max-batches-per-epoch 20 --hard-concrete-samples-per-batch 1 --out-dir runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_trajectory_mask_training_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md
```

Provenance note:

An accidental duplicate seed-0 launch was excluded from the canonical selected
run root before summarization and moved to `discarded_runs/` so the release
manifest does not mix exploratory duplicate output into the evidence package.

Result:

The selected root contains exactly five canonical seeds. Hard-concrete masks
retrain to `0.2766` accuracy with 95% CI `[0.2743, 0.2788]`, `0.6204` below
matched IMP, and only `0.0922` Jaccard overlap to IMP support. This closes the
hard-concrete support-row gap as negative evidence rather than a rescue.

## 2026-05-06 - Weight-Correlation Aligned Mode/Ticket Probe

Purpose:

Close the next channel-permutation robustness objection after activation
alignment by aligning ResNet channels with incoming/outgoing weight-correlation
features before recomputing the direct mode/ticket distribution metrics.

Implementation:

- Extended `scripts/run_mode_ticket_distribution_probe.py` with
  `--alignment-method weight`.
- Reused the existing Hungarian correlation matching path, but collected
  per-channel features from incoming and outgoing ResNet convolution weights.
- Updated `scripts/summarize_mode_ticket_distribution_probe.py` so aligned
  rows are named from the configured alignment method, e.g.
  `weight_aligned_posterior_samples_vs_tickets`.

Smoke commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0,1 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 1 --sgld-steps 2 --sgld-burn-in 0 --sgld-sample-every 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --alignment-method weight --alignment-batches 1 --out-dir runs/fake_cifar10_mode_ticket_weight_alignment_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_weight_alignment_smoke --out-md docs/fake_cifar10_mode_ticket_weight_alignment_smoke.md --out-csv runs/fake_cifar10_mode_ticket_weight_alignment_smoke_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 4 --seeds 0,1 --epochs 2 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 2 --sgld-steps 6 --sgld-burn-in 0 --sgld-sample-every 2 --sgld-lr 1e-6 --sgld-likelihood-scale mean --batch-size 128 --train-subset 512 --test-subset 256 --lr 0.03 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 4 --sliced-projections 16 --alignment-method weight --alignment-batches 1 --out-dir runs/cifar10_subset_weight_alignment_mode_ticket_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_subset_weight_alignment_mode_ticket_smoke --out-md docs/cifar10_subset_weight_alignment_mode_ticket_smoke.md --out-csv runs/cifar10_subset_weight_alignment_mode_ticket_smoke_summary.csv
```

Full-data selected command:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --alignment-method weight --alignment-batches 10 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv
```

Result:

The full-data weight-aligned row is negative. It keeps one posterior basin
against five IMP tickets; `weight_aligned_posterior_samples_vs_tickets` fails
layer KS with `p=1.2056e-08` and Hamming overlap `0.1290`, while logit CKA
and final-hidden activation CKA remain high at `0.9336` and `0.9131`.
Weight-correlation channel alignment therefore does not rescue the direct
mask-distribution or basin-count claim.

## 2026-05-06 - CIFAR Rank-64 Low-Rank Laplace Movement

Purpose:

Harden the full-network posterior-side rebuttal by increasing the
Hessian-plus-diagonal Laplace rank from 32 to 64 under the same five-seed
CIFAR-10 ResNet-20 epoch-1 rewind protocol.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 64 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 16 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 64 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 16 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3.md
```

Result:

All five seeds retain 64 positive Hessian directions. At scale `1e-2`,
posterior-to-chain-start overlap falls to `0.7397`, but posterior-to-IMP is
`0.1339`, below chain-start support `0.1433` and rewind magnitude `0.1766`;
sample accuracy is `0.8816`. This strengthens the rank-16/rank-32 conclusion:
larger full-network low-rank curvature movement still is not ticket-directed.

## 2026-05-06 - CIFAR Rank-128 Low-Rank Laplace Movement

Purpose:

Push the full-network Hessian-plus-diagonal Laplace robustness check beyond
rank 64 while staying within the single-workstation budget. This tests whether
doubling the retained positive Hessian directions changes the negative
rank-16/rank-32/rank-64 pattern.

Commands:

```bash
.venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed 0 --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-4,1e-3,3e-3,1e-2 --lowrank-laplace-rank 128 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 32 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_sgld_movement_grid.py --dataset cifar10 --model resnet20 --seed "$seed" --epochs 30 --imp-epochs 30 --imp-final-epochs 30 --rewind-epochs 1 --imp-rounds 5 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler lowrank-laplace --lowrank-laplace-scales 1e-2 --lowrank-laplace-rank 128 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 32 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --samples 10 --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_sgld_movement_grid.py --run-root runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3.md
```

Result:

All five selected seeds retain 128 positive Hessian directions. At scale
`1e-2`, posterior-to-chain-start overlap falls to `0.7358`, but
posterior-to-IMP is `0.1351`, below chain-start support `0.1453` and rewind
magnitude `0.1780`; sample accuracy is `0.8813`. The mode-distribution audit
now reports posterior-random `58/59`, posterior-chain `0/59`, and
rewind-over-posterior `55/57`. The rank-128 row therefore further narrows the
low-rank covariance objection without changing the conclusion.

## 2026-05-06 - CIFAR Rank-128 Low-Rank Laplace Direct Mode/Ticket Probe

Purpose:

Test the rank-128 Hessian-plus-diagonal Laplace posterior with the proposal's
direct distributional metrics rather than only movement Jaccard. This asks
whether the Hessian-informed posterior can satisfy layer-sparsity KS,
Hamming-distribution overlap, logit/final-hidden activation CKA, and
parameter-PCA basin-count criteria against five IMP tickets.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --posterior-sampler lowrank-laplace --samples 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --lowrank-laplace-rank 2 --lowrank-laplace-oversample 1 --lowrank-laplace-fisher-batches 1 --lowrank-laplace-hessian-batches 1 --cluster-pca-dim 2 --sliced-projections 4 --out-dir runs/fake_cifar10_mode_ticket_lowrank_laplace_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_lowrank_laplace_smoke --out-md docs/fake_cifar10_mode_ticket_lowrank_laplace_smoke.md --out-csv runs/fake_cifar10_mode_ticket_lowrank_laplace_smoke_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler lowrank-laplace --samples 10 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --lowrank-laplace-scale 1e-2 --lowrank-laplace-rank 128 --lowrank-laplace-power-iterations 1 --lowrank-laplace-oversample 32 --lowrank-laplace-fisher-batches 20 --lowrank-laplace-hessian-batches 2 --lowrank-laplace-prior-precision 1e-2 --lowrank-laplace-damping 1e-6 --lowrank-laplace-batchnorm-mode eval --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv
```

Result:

The full-data rank-128 low-rank Laplace direct row is a partial rescue, not a
positive equivalence result. `posterior_samples_vs_tickets` compares 50
posterior samples to five IMP tickets, collapses to one posterior basin, and
passes Hamming overlap `0.8163` plus logit/final-hidden activation CKA
`0.9319`/`0.9096`. It still strongly fails raw sample layer-sparsity KS with
`p=1.9820e-06`. The collapsed mode representative passes layer KS
(`p=0.1388`) but remains one representative versus five IMP tickets, so the
proposal's basin-count equivalence remains unsupported.

## 2026-05-06 - CIFAR Independent-Start Multi-Chain cSGLD Direct Probe

Purpose:

Test whether the negative direct mode/ticket result is an artifact of starting
all posterior chains from the same trained dense checkpoint within each seed.
This run trains three independent dense chain starts per seed, samples
cyclical-SGLD from each, and compares 75 posterior samples against the five IMP
tickets with the same proposal-level direct metrics.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --posterior-sampler cyclical-sgld --samples 1 --posterior-chains 2 --posterior-chain-init independent-dense --sgld-steps 4 --sgld-burn-in 0 --sgld-sample-every 1 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 2 --csgld-sample-phase-start 0.5 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --out-dir runs/fake_cifar10_mode_ticket_csgld_independent_multichain_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_csgld_independent_multichain_smoke --out-md docs/fake_cifar10_mode_ticket_csgld_independent_multichain_smoke.md --out-csv runs/fake_cifar10_mode_ticket_csgld_independent_multichain_smoke_summary.csv
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --posterior-sampler cyclical-sgld --samples 5 --posterior-chains 3 --posterior-chain-init independent-dense --sgld-steps 400 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --sgld-likelihood-scale mean --csgld-cycle-length 50 --csgld-sample-phase-start 0.5 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --cluster-pca-dim 20 --sliced-projections 128 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3 --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv
```

Result:

The independent-start row is negative. It compares 75 posterior samples from
15 independently trained dense starts against five IMP tickets. Chain starts
and posterior samples both collapse to one parameter-PCA basin; posterior
samples move only `0.0439` Hamming from their own chain starts and have mean
accuracy `0.8763` versus chain-start mean `0.8845`. The
`posterior_samples_vs_tickets` row fails layer KS with `p=9.3025e-10` and
Hamming overlap `0.0000`, while logit/final-hidden activation CKA remain high
at `0.9269`/`0.9051`.

## 2026-05-06 - CIFAR 22k-Parameter Block-Diagonal Laplace Probe

Purpose:

Narrow the remaining exact-covariance posterior gap between selected/joint
block Laplace probes and infeasible dense full-network covariance. This probe
estimates exact full-covariance softmax-GGN/Laplace blocks for all weight
tensors with at most 5000 parameters, then samples those 11 tensors
independently and simultaneously in one block-diagonal full-network sample.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-3 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-blocks-under-max --independent-block-diagonal --random-trials 4 --out-dir runs/fake_cifar10_blockdiag_laplace_smoke
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/fake_cifar10_blockdiag_laplace_smoke --out-csv runs/fake_cifar10_blockdiag_laplace_smoke_summary.csv --out-md docs/fake_cifar10_blockdiag_laplace_smoke.md
.venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-4,3e-4,1e-3 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 5000 --auto-blocks-under-max --independent-block-diagonal --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_tune_seed0_r5_p0p3.md
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 5000 --auto-blocks-under-max --independent-block-diagonal --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md
```

Result:

The selected row covers 11 tensors and 22,064 parameters, or 8.1% of the
ResNet-20 weight vector. At scale `1e-4`, it preserves sample accuracy
(`0.8810`) and moves support away from the dense chain start (`global
post-chain=0.8287`). It still does not make support movement ticket-directed:
selected-block posterior-minus-chain is `-0.0114`, global posterior-minus-chain
is only `+0.0036`, and global rewind-minus-posterior is `+0.0292`. This is a
stronger exact-covariance subset than the earlier selected/joint block rows,
but it remains negative.

## 2026-05-06 - CIFAR Residual IMP-Process Posterior-Projection Control

Purpose:

Test whether the useful RMS trajectory round-5 IMP-process residual ordering is
explained by a diagonal-Laplace posterior score subspace. The new control
residualizes the round-trained score against base-source, dense-final,
final-IMP magnitude, diagonal-Laplace posterior RMS, posterior standard
deviation, and posterior RMS-minus-dense scores inside the final-IMP residual
candidate pool.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 0 --imp-rounds 1 --process-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --mask-train-epochs 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --base-sources epoch_1 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-posterior-residualized-score --posterior-projection-laplace-samples 2 --posterior-projection-laplace-fisher-batches 1 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_posterior_projection_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_posterior_projection_smoke --out-csv runs/fake_cifar10_residual_imp_process_posterior_projection_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_posterior_projection_smoke.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --mask-train-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-posterior-residualized-score --posterior-projection-laplace-samples 10 --posterior-projection-laplace-scale 1e-3 --posterior-projection-laplace-prior-precision 1e-2 --posterior-projection-laplace-fisher-batches 20 --posterior-projection-laplace-variance-floor 1e-12 --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3.md
```

Result:

The five-seed row is negative for the posterior-projection rescue. The original
round final-IMP residual row reaches `0.8847` accuracy and `0.6773`
final-oracle overlap. The posterior-residualized row reaches `0.8825` accuracy
and `0.4850` final-oracle overlap. The paired accuracy delta is small
(`+0.0023`, `5/5` positive seeds, CI crossing zero), but the oracle-overlap
drop is sharp at `+0.1923` with `5/5` positive seeds. The process-selected
coordinate ordering is therefore not explained away by the current
diagonal-Laplace posterior score subspace.

## 2026-05-06 - CIFAR Residual IMP-Process Learned-Subspace Control

Purpose:

Test whether a learned low-dimensional trajectory/process subspace can replace
the exact RMS trajectory round-5 IMP-process residual coordinates. The control
builds rank-8 PCA component scores inside the final-IMP residual candidate pool
from dense trajectory score flats, final-IMP magnitude, and earlier IMP-round
scores, then residualizes the current round score against those components.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --trajectory-epochs 0,1 --rewind-epochs 0 --imp-rounds 1 --process-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --mask-train-epochs 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --base-sources epoch_1 --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-learned-subspace-residualized-score --learned-subspace-rank 2 --random-trials 1 --out-dir runs/fake_cifar10_residual_imp_process_learned_subspace_smoke
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/fake_cifar10_residual_imp_process_learned_subspace_smoke --out-csv runs/fake_cifar10_residual_imp_process_learned_subspace_smoke_summary.csv --out-md docs/fake_cifar10_residual_imp_process_learned_subspace_smoke.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_residual_imp_process_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --trajectory-epochs 0,1,2,5,10,20,30 --rewind-epochs 1 --imp-rounds 5 --process-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --mask-train-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --base-sources traj_rms_abs --alphas 0.5 --round-variants final-imp,final-imp-dense-score,final-imp-base-score,final-imp-round-learned-subspace-residualized-score --learned-subspace-rank 8 --random-trials 1 --out-dir runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3
.venv/bin/python scripts/summarize_residual_imp_process_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md
```

Result:

The five-seed learned-subspace control is negative for the subspace-replacement
rescue. The original round final-IMP residual row reaches `0.8869` accuracy
and `0.6807` final-oracle overlap. The learned-subspace residualized row
reaches `0.8821` accuracy and `0.4917` final-oracle overlap. The paired
accuracy delta is `+0.0048` with `5/5` positive seeds, and the 95% CI is
`[+0.0029,+0.0067]`. The oracle-overlap drop is `+0.1890` with `5/5`
positive seeds. A learned trajectory/process subspace therefore captures broad
score directions but does not replace the exact process-selected residual
coordinates in this control.

## 2026-05-06 - CIFAR 68k-Parameter Block-Diagonal Laplace Probe

Purpose:

Widen the exact tensor-block-diagonal covariance axis beyond the 22,064
parameter row. The max-10k setting includes all weight tensors with at most
10,000 parameters, adding the stage-2 convolution blocks and covering 16 tensors
and 68,144 trainable weights.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 4 --out-dir runs/fake_cifar10_blockdiag_laplace_max10k_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6,3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_tune_small_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-blocks-under-max --independent-block-diagonal --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3.md
```

Result:

The five-seed max-10k row preserves sample accuracy (`0.8802`) and moves from
the dense chain-start support (`global post-chain=0.7400`) more than the
earlier 22k row. It still does not make support movement ticket-directed:
block posterior-chain remains negative (`-0.0050`), global posterior-chain is
only `+0.0010`, and global rewind-minus-posterior remains `+0.0319`. This
reduces the selected-block/full-network covariance gap further, while leaving
literal dense or cross-tensor full-network covariance as the remaining
posterior-side limitation.

## 2026-05-06 - CIFAR 68k-Parameter Joint-Group Laplace Probe

Purpose:

Relax the independent-tensor assumption in the max-10k exact covariance row.
The probe keeps the same 16 tensors and 68,144 trainable weights, but greedily
packs them into 8 joint groups under 10,000 parameters each. Each group uses an
exact full-covariance softmax-GGN factor with cross-tensor covariance inside the
group, and the groups are sampled independently in one combined network sample.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-joint-groups-under-max --random-trials 4 --out-dir runs/fake_cifar10_jointdiag_laplace_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6,3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-joint-groups-under-max --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 10000 --auto-joint-groups-under-max --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3.md
```

Result:

The five-seed joint-group row preserves sample accuracy (`0.8811`) and moves
from the dense chain-start support (`global post-chain=0.7148`) more than the
independent tensor-block-diagonal max-10k row. It still does not make support
movement ticket-directed: block posterior-chain remains negative (`-0.0050`),
global posterior-chain is only `+0.0015`, and global rewind-minus-posterior
remains `+0.0311`. This reduces the simple per-tensor-covariance objection
while leaving literal dense full-network covariance as the remaining posterior
side limitation.

## 2026-05-06 - Citation Metadata and Related-Work Hardening

Purpose:

Close a paper-polish gap in the related-work and bibliography layer. The local
literature matrix identified Paul et al.'s ICLR 2023 mask-subspace explanation
as a direct competitor, but the paper did not cite it explicitly. The
Bayesian-LTH and PAC-Bayesian LTH references also had placeholder or shortened
metadata.

Commands:

```bash
perl -ne 'while(/\\cite[a-zA-Z*]*\{([^}]*)\}/g){print "$1\n"}' paper/main.tex | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sort -u
perl -ne 'print "$1\n" if /^@\w+\{([^,]+),/' paper/refs.bib | sort -u
comm -23 <(perl -ne 'while(/\\cite[a-zA-Z*]*\{([^}]*)\}/g){print "$1\n"}' paper/main.tex | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sort -u) <(perl -ne 'print "$1\n" if /^@\w+\{([^,]+),/' paper/refs.bib | sort -u)
make paper-check
rg -n "undefined|Warning|Citation|Reference|Overfull|Underfull" paper/main.log || true
.venv/bin/python scripts/verify_research_artifacts.py
```

Result:

`paper/main.tex` now cites Paul et al. 2023 as the direct mask-subspace
competitor in the related-work section. `paper/refs.bib` now includes
`paul2023unmasking`, corrects Sakamoto and Sato's PAC-Bayesian LTH title and
author metadata, and expands the 2026 Bayesian LTH arXiv entry to the full
author list and arXiv ID. `scripts/verify_research_artifacts.py` now checks
that every citation key in `paper/main.tex` exists in `paper/refs.bib` and that
the key competitor citation metadata is present. The final LaTeX log has no
warning lines after the full `pdflatex/bibtex/pdflatex/pdflatex` sequence.

## 2026-05-06 - CIFAR 86k-Parameter Joint-Group Laplace Probe

Purpose:

Push the exact joint-group covariance row beyond max-10k coverage by adding the
first stage-3 convolution block. The max-20k setting covers 17 tensors and
86,576 trainable weights packed into 6 exact covariance groups, keeping each
group under 20,000 parameters.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 3e-6,1e-5,3e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 20000 --auto-joint-groups-under-max --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 10 --block-laplace-scales 3e-6 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 20000 --auto-joint-groups-under-max --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3.md
```

Result:

The seed-0 scale scan selected `3e-6`: it preserved sample accuracy while
moving support substantially from the chain start. The five-seed selected row
preserves sample accuracy (`0.8828`) and keeps support moved from the dense
chain-start control (`global post-chain=0.7863`). It still does not make
support movement ticket-directed: block posterior-chain remains negative
(`-0.0023`), global posterior-chain is only `+0.0006`, and global
rewind-minus-posterior remains `+0.0317`. This extends the exact local
covariance probe into stage-3 while preserving the same negative conclusion.

## 2026-05-06 - CIFAR 270k-Parameter Streamed Joint-Group Laplace Probe

Purpose:

Remove the practical weight-tensor coverage objection to the block-Laplace
evidence. The max-40k setting greedily packs all 22 ResNet-20 weight tensors
into 8 exact joint covariance groups, covering all 270,896 weight parameters.
`--stream-joint-groups` estimates and samples one exact group at a time, so the
run does not need to keep all large Cholesky factors resident simultaneously.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_block_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seed 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --samples 2 --block-laplace-scales 1e-4 --block-laplace-hessian-batches 1 --block-laplace-max-parameters 512 --auto-joint-groups-under-max --stream-joint-groups --random-trials 4 --out-dir runs/fake_cifar10_jointdiag_laplace_stream_smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed 0 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 3 --block-laplace-scales 1e-6,3e-6,1e-5 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 40000 --auto-joint-groups-under-max --stream-joint-groups --random-trials 20 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_tune_seed0_r5_p0p3
for seed in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_block_laplace_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seed "$seed" --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --samples 5 --block-laplace-scales 1e-6 --block-laplace-hessian-batches 1 --block-laplace-prior-precision 1e-2 --block-laplace-damping 1e-5 --block-laplace-max-parameters 40000 --auto-joint-groups-under-max --stream-joint-groups --random-trials 100 --out-dir runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3; done
.venv/bin/python scripts/summarize_block_laplace_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3.md
```

Result:

The seed-0 scale scan selected `1e-6`: higher scales lowered sample accuracy
without improving ticket-directed support. The five-seed selected row preserves
sample accuracy (`0.8824`) and moves support from the dense chain-start control
(`global post-chain=0.7389`). It still does not make support movement
ticket-directed: block and global posterior-chain are both `-0.0019`, and
global rewind-minus-posterior remains `+0.0362`. The largest groups repeatedly
used about 31.7 GiB of GPU memory on the RTX 5090 workstation, confirming that
streaming makes this row feasible while full dense covariance remains outside
the local budget.

## 2026-05-06 - CIFAR Full-Weight Joint-Group Laplace Direct Mode/Ticket Probe

Purpose:

Feed the same streamed 270,896-parameter exact joint-group Laplace posterior
into the proposal's literal direct mode/ticket distribution probe. The movement
row already showed non-ticket-directed support movement; this run checks the
stronger KS/Hamming/CKA/basin-count criteria directly.

Commands:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.30 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --posterior-sampler jointdiag-laplace --samples 2 --jointdiag-laplace-scale 1e-4 --jointdiag-laplace-hessian-batches 1 --jointdiag-laplace-max-parameters 512 --out-dir runs/fake_cifar10_mode_ticket_jointdiag_laplace_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_jointdiag_laplace_smoke --out-csv runs/fake_cifar10_mode_ticket_jointdiag_laplace_smoke_summary.csv --out-md docs/fake_cifar10_mode_ticket_jointdiag_laplace_smoke.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --posterior-sampler jointdiag-laplace --samples 5 --jointdiag-laplace-scale 1e-6 --jointdiag-laplace-prior-precision 1e-2 --jointdiag-laplace-damping 1e-5 --jointdiag-laplace-hessian-batches 1 --jointdiag-laplace-max-parameters 40000 --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md
```

Result:

The direct full-weight grouped row is negative. It compares 25 exact
joint-group posterior samples against five IMP tickets. Samples preserve useful
accuracy (`0.8835`) and move from their chain starts
(`posterior-to-chain-start` Hamming `0.0503`), but they still collapse to one
parameter-PCA basin. The raw sample masks fail layer-sparsity KS (`p=1.1e-08`)
and Hamming overlap (`0.0000 < 0.70`) while logit/final-hidden activation CKA
remain high (`0.9373`/`0.9199`). Thus full-weight exact grouped covariance
does not rescue the proposal-level mode/ticket equivalence criterion.

## 2026-05-06 - Mode/Ticket Alignment Artifact Audit

Purpose:

Make the remaining permutation gap explicit without rerunning CIFAR training.
The audit reuses the existing full-data direct mode/ticket artifacts and checks
whether the saved release files can support stronger post-hoc
graph/permutation realignment.

Command:

```bash
.venv/bin/python scripts/audit_mode_ticket_alignment_artifacts.py
```

Output:

- `docs/mode_ticket_alignment_artifact_audit.md`
- `runs/mode_ticket_alignment_artifact_audit.json`

Result:

The audit covers seven full-data CIFAR direct rows: unaligned SGLD,
activation-aligned SGLD, weight-aligned SGLD, dense-start and
independent-start multi-chain cyclical-SGLD, LowRank128 Laplace, and
JointDiagLap270k. Every row collapses posterior samples to one basin and fails
full direct equivalence. Both aligned rows fail layer-KS and Hamming-overlap
after mapping into the seed-0 frame. The artifact scan finds zero raw
posterior/ticket mask or state tensor files, so post-hoc exhaustive
graph/permutation realignment is not supported by the current direct-run
artifacts; a stronger permutation row requires a rerun with saved masks/states.

## 2026-05-06 - Direct Probe Mask Artifact Saving Smoke

Purpose:

Add the missing artifact path needed for future exhaustive permutation work.
The previous full-data direct runs saved summary CSV/JSON but not raw masks or
states. This smoke validates that direct probes can now write a compact `.npz`
fixture containing flattened masks, IDs, parameter order, and optional
flattened trainable states.

Command:

```bash
.venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 2 --seeds 0,1 --epochs 1 --rewind-epochs 1 --imp-rounds 1 --prune-fraction 0.3 --samples 1 --sgld-steps 2 --sgld-burn-in 0 --sgld-sample-every 1 --batch-size 32 --train-subset 64 --test-subset 32 --lr 0.01 --lr-schedule cosine --weight-decay 1e-4 --cluster-pca-dim 2 --sliced-projections 4 --alignment-method activation --alignment-batches 1 --save-mask-artifacts --save-state-artifacts --out-dir runs/fake_cifar10_mode_ticket_mask_artifact_smoke
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/fake_cifar10_mode_ticket_mask_artifact_smoke --out-md docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md --out-csv runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv
```

Output:

- `runs/fake_cifar10_mode_ticket_mask_artifact_smoke/20260506_225421/mask_artifacts.npz`
- `docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md`
- `runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv`

Result:

The fixture stores schema version 1, 22 parameter names, 4,350 flattened
parameters, `parameter_shapes_json` for ResNet graph/channel reconstruction,
eight raw/aligned mask collections, and matching state collections.
Posterior/ticket mask arrays are `uint8` with shape `(2, 4350)` and state
arrays are `float32` with shape `(2, 4350)`. At the time of this smoke it
validated the artifact path before the full-data saved-artifact rerun; by
itself it is not claim-level CIFAR evidence.

## 2026-05-06 - Mask Artifact Post-hoc Matching Audit

Purpose:

Close the software-path gap after adding `mask_artifacts.npz`. The previous
smoke verified that raw masks, states, and parameter shapes can be saved; this
audit verifies that a downstream script can read the saved fixture and compute
same-index plus minimum-cost record-level post-hoc comparisons and local
channel-permutation comparisons without rerunning training.

Command:

```bash
.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py
```

Output:

- `docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md`
- `runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json`

Result:

The audit reads
`runs/fake_cifar10_mode_ticket_mask_artifact_smoke/20260506_225421/mask_artifacts.npz`,
covers eight raw/aligned mask collections, eight state collections, and 19
ResNet channel keys, and computes eight collection comparisons. Raw posterior
samples versus tickets have same-index and optimal-assignment Hamming `0.0030`;
activation-aligned posterior samples versus aligned tickets have Hamming
`0.1333`, which the local channel-permutation objective reduces to `0.0030`
on this path-validation fixture. This confirms record-level and local channel
post-hoc matching are wired, but it explicitly does not claim exhaustive
graph/channel permutation support or full-data CIFAR evidence.

## 2026-05-06 - Full-data Mask Artifact Storage Budget

Purpose:

Quantify whether the next claim-level full-data saved-artifact rerun is blocked
by storage. The audit is dependency-light and computes ResNet-20 weight counts
analytically so it also runs inside the minimal artifact-verification
container.

Command:

```bash
.venv/bin/python scripts/audit_mode_ticket_artifact_storage_budget.py
```

Output:

- `docs/mode_ticket_artifact_storage_budget.md`
- `runs/mode_ticket_artifact_storage_budget.json`

Result:

The full-data CIFAR-10 ResNet-20 weight-mask parameter count is `270,896`.
The recommended activation-aligned SGLD rerun with `--save-mask-artifacts` and
`--save-state-artifacts` has an upper-bound footprint of `220` mask records and
`220` state records, or `284.18 MiB` uncompressed before `.npz` compression.
The independent-start cyclical-SGLD saved-state variant is estimated at
`219.59 MiB`, LowRank128 at `142.09 MiB`, and streamed JointDiagLap at
`77.50 MiB`. Storage is therefore not the blocker for the full-data
permutation artifact rerun; GPU time is.

## 2026-05-06 - Full-data Activation-aligned Saved-artifact Rerun

Purpose:

Close the claim-level saved-artifact gap for the strongest first-order
channel-aligned SGLD mode/ticket row by rerunning full-data CIFAR-10
ResNet-20 with raw masks, aligned masks, and trainable state matrices saved.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_mode_ticket_distribution_probe.py --dataset cifar10 --model resnet20 --resnet-width 16 --seeds 0,1,2,3,4 --epochs 30 --rewind-epochs 1 --imp-rounds 5 --imp-epochs 30 --imp-final-epochs 30 --prune-fraction 0.30 --samples 10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --sgld-lr 1e-6 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --alignment-method activation --alignment-batches 10 --save-mask-artifacts --save-state-artifacts --out-dir runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3
.venv/bin/python scripts/summarize_mode_ticket_distribution_probe.py --run-root runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3 --out-csv runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz`
- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/metrics.json`
- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv`
- `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md`

Result:

All five seeds completed. Dense accuracies were `0.8875`, `0.8874`, `0.8927`,
`0.8837`, and `0.8837`; IMP ticket accuracies were `0.9013`, `0.8966`,
`0.9001`, `0.9009`, and `0.8954`. The saved run stores eight raw/aligned mask
collections and eight state collections over `270,896` parameters. The direct
proposal metrics remain negative: posterior samples versus tickets have layer
KS `p=1.82e-09`, Hamming overlap `0.0000`, one posterior basin, logit CKA
`0.9349`, and activation CKA `0.9161`; activation-aligned posterior samples
show the same threshold pattern.

## 2026-05-06 - Full-data Saved-artifact Post-hoc Matching Audit

Purpose:

Verify that the full-data saved CIFAR artifact can drive downstream
record-level post-hoc matching without rerunning training.

Command:

```bash
.venv/bin/python scripts/audit_mask_artifact_posthoc_matching.py --artifact runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz --out-json runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json --out-md docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md --max-channel-pair-count 1
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json`
- `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md`

Result:

The audit covers eight mask collections, eight state collections, and 19 ResNet
channel keys for the full-data `270,896`-parameter artifact. Record-level
post-hoc matching is supported. Raw posterior samples versus tickets improve
from same-index Hamming `0.2469` to optimal-record Hamming `0.2113`, and
activation-aligned posterior samples versus aligned tickets improve from
`0.2466` to `0.2440`. Local channel-permutation search was intentionally
capped and skipped for the retained full-data comparisons; the fake-CIFAR audit
continues to cover that software path. Exhaustive graph/channel permutation
over full-data masks or states remains a separate analysis.

## 2026-05-07 - Full-data Structured Global Channel Audit

Purpose:

Test the strongest remaining channel-relabeling rescue on the saved full-data
CIFAR artifact without rerunning training. The audit optimizes all ResNet
channel keys jointly by block-coordinate descent, using exact per-key Hungarian
updates on record-optimal posterior/ticket pairs.

Command:

```bash
.venv/bin/python scripts/audit_full_data_channel_permutation_matching.py --max-iters 6 --max-pairs-per-comparison 5
```

Output:

- `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json`
- `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md`

Result:

The audit covers six comparisons over the full-data `270,896`-parameter saved
artifact. Raw posterior samples versus tickets improve only from record-optimal
Hamming `0.2113` to global-channel Hamming `0.2105`, with support overlap
`0.3738`. Activation-aligned posterior samples improve from `0.2440` to
`0.2113`, showing that the optimizer removes much of the aligned-frame mismatch
but still does not approach ticket-like support agreement. Posterior modes
remain similar (`0.2080` to `0.2075`). This is not an exhaustive graph
isomorphism proof, but it rules out the simple structured channel-relabeling
rescue on the saved full-data masks.

## 2026-05-07 - Exhaustive Channel-Permutation Feasibility Audit

Purpose:

Make the remaining graph-isomorphism limitation precise. The full-data global
channel audit is a structured coordinate objective, not an exhaustive search,
so this audit checks the exact enumeration path on a tiny saved-artifact
subgraph and quantifies the full-data search space.

Command:

```bash
.venv/bin/python scripts/audit_exhaustive_channel_permutation_feasibility.py
```

Output:

- `runs/resnet_channel_permutation_exhaustive_feasibility_audit.json`
- `docs/resnet_channel_permutation_exhaustive_feasibility_audit.md`

Result:

The fake-CIFAR stage-1 ResNet subgraph contains `270` parameters and seven
two-channel keys, so all `128` global channel assignments can be enumerated.
Exact stage-1 enumeration drives ticket raw-vs-aligned and chain-start
raw-vs-aligned Hamming to `0.0000`, and the block-coordinate solver matches
the exact optimum on every audited pair. The full-data CIFAR ResNet-20 saved
artifact has 19 channel keys with widths 16, 32, and 64, giving about
`10^840.4` channel assignments per record pair. Thus exhaustive full-data
graph/channel isomorphism remains infeasible and unimplemented, while the
small exact path validates that the audit machinery catches known frame
equivalences when enumeration is possible.

## 2026-05-07 - Tiny Exact Dense Full-network Laplace Probe

Purpose:

Validate the all-parameter dense covariance code path in a setting where exact
full-network softmax-GGN/Laplace sampling is runnable, without treating it as
CIFAR-scale evidence.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_fullnet_laplace_probe.py --seed "$seed" --hidden-dim 4 --depth 2 --epochs 20 --imp-rounds 2 --prune-fraction 0.3 --samples 10 --full-laplace-scales 1e-5,1e-4,1e-3,1e-2 --out-dir runs/digits_fullnet_laplace_tiny_r2_p0p3; done
.venv/bin/python scripts/summarize_fullnet_laplace_probe.py --run-root runs/digits_fullnet_laplace_tiny_r2_p0p3 --out-csv runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv --out-md docs/digits_fullnet_laplace_tiny_r2_p0p3.md
```

Output:

- `runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv`
- `docs/digits_fullnet_laplace_tiny_r2_p0p3.md`

Result:

The probe samples one exact dense Gaussian over all `310` trainable parameters
of a tiny digits MLP. At scale `1e-3`, samples keep `0.8450` accuracy and move
away from the dense chain-start support (`post-chain=0.8084`), but
posterior-to-IMP support is `0.7545` versus `0.8596` for the chain-start
magnitude control, giving a `-0.1050` posterior-chain gap. The exact dense
full-network Laplace sanity row is therefore negative and keeps the remaining
CIFAR dense-covariance limitation scoped rather than untested.

## 2026-05-07 - Fake-CIFAR ResNet Exact Dense Full-network Laplace Smoke

Purpose:

Check that the exact dense full-network softmax-GGN/Laplace path also works
for convolutional, residual, and BatchNorm parameters, not only for MLP
parameters. This is code-path validation, not real CIFAR evidence.

Command:

```bash
for seed in 0 1 2 3 4; do .venv/bin/python scripts/run_digits_fullnet_laplace_probe.py --dataset fake-cifar10 --model resnet20 --resnet-width 1 --seed "$seed" --epochs 1 --imp-rounds 1 --imp-epochs 1 --imp-final-epochs 1 --prune-fraction 0.3 --batch-size 16 --test-batch-size 32 --train-size 32 --test-size 32 --samples 2 --full-laplace-scales 1e-5,1e-3 --full-laplace-hessian-batches 1 --full-laplace-max-parameters 2000 --out-dir runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke; done
.venv/bin/python scripts/summarize_fullnet_laplace_probe.py --run-root runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke --out-csv runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv --out-md docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md
```

Output:

- `runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv`
- `docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md`

Result:

The fake-CIFAR width-1 ResNet-20 smoke covers all `1,229` trainable parameters
and all `1,121` weight parameters with a dense `1229 x 1229` Cholesky factor.
At scale `1e-3`, support moves from the dense chain start
(`post-chain=0.5398`) and remains below the chain-start support control
(`posterior-chain=-0.4372`). Because labels and inputs are fake, the accuracy
numbers are not research evidence; the artifact only validates the exact dense
convolutional/residual/BatchNorm code path.

## 2026-05-07 - Linear Connectivity Barrier Audit

Purpose:

Turn the existing linear loss-barrier columns into a reviewer-facing audit
that separates landscape connectivity from the posterior-ticket support claim.

Command:

```bash
.venv/bin/python scripts/audit_linear_connectivity_barriers.py
```

Output:

- `runs/linear_connectivity_barrier_audit.csv`
- `runs/linear_connectivity_barrier_audit.json`
- `docs/linear_connectivity_barrier_audit.md`

Result:

The audit aggregates six existing five-seed rows: MNIST Gate1, Fashion-MNIST
Gate1, CIFAR-10 ResNet-20 long SGLD, long SWAG, three-chain SGLD, and short
SWAG. MNIST/Fashion dense-to-IMP barriers are near zero (`0.0026` and
`0.0395`), but posterior support remains slightly below chain-start support.
CIFAR long SGLD/SWAG dense-to-IMP barriers are large (`3.0827` and `3.7402`),
yet posterior support is still tied to chain-start support. The conclusion is
that linear connectivity barriers are orthogonal landscape diagnostics, not
evidence of posterior-ticket equivalence.

## 2026-05-07 - Reviewer Objection Matrix

Purpose:

Compress the current evidence into a reviewer-facing risk register rather than
another long narrative audit.

Command:

```bash
.venv/bin/python scripts/build_reviewer_objection_matrix.py
```

Output:

- `runs/reviewer_objection_matrix.json`
- `docs/reviewer_objection_matrix.md`

Result:

The generated matrix contains nine likely objections. Five are closed for the
current artifacts, two are bounded open limitations, one is a partially closed
positive-mechanism row, and one remains an open packaging limitation. The most
important rows for a top-conference paper are random-control, sampler-movement,
function-vs-mask, alignment/permutation, and covariance-fidelity objections.
The matrix makes clear that this is not yet a submit-ready package: exact dense
CIFAR posterior evidence, exhaustive full-data graph isomorphism, external
CI/GPU validation, and clean repository state remain open or bounded gaps.

## 2026-05-07 - Paper Submission Shape Audit

Purpose:

Make the manuscript-editing blocker machine-checkable. The artifact verifier
can pass while the draft is still too long or too diffuse for a top-conference
submission.

Command:

```bash
.venv/bin/python scripts/audit_paper_submission_shape.py
```

Output:

- `runs/paper_submission_shape_audit.json`
- `docs/paper_submission_shape_audit.md`

Result:

The initial audit marked the paper shape as not ready. The main body before the
generated evidence tables was over the target line budget, and the `Current
Results` section dominated the draft. All major reviewer-objection families
were visible in the main text, so the next editing step was not to add new
evidence: it was to reorganize the paper around the objection matrix and move
most residual/process variants plus code-path smokes to appendix or
reproducibility text.

## 2026-05-07 - Paper Main-Text Condensation Pass

Purpose:

Convert the evidence-rich working draft into a submission-shaped main text
without dropping the reviewer-objection coverage or generated numeric claim
checks.

Command:

```bash
.venv/bin/python scripts/audit_paper_submission_shape.py
```

Output:

- `paper/main.tex`
- `runs/paper_submission_shape_audit.json`
- `docs/paper_submission_shape_audit.md`

Result:

The manuscript main body is now condensed from the prior 1,225-line state to
500 lines before the appendix/generated tables, and `Current Results` is now
260 lines. The shape audit marks the current draft ready by the local main-text
gate, with no blocking risk flags and all reviewer-objection families still
visible in the main text. The total compiled PDF page count is tracked only as
visibility because the same PDF includes appendix/generated evidence tables.

## 2026-05-07 - Main-only Submission PDF Gate

Purpose:

Separate the venue-facing main paper from the appendix-inclusive
reproducibility PDF. This keeps `paper/main.pdf` as the full evidence package
while producing a page-budgeted main-only PDF from the same source.

Command:

```bash
make paper-submission-check
.venv/bin/python scripts/audit_submission_pdf_shape.py
```

Output:

- `paper/main_submission.pdf`
- `runs/submission_pdf_shape_audit.json`
- `docs/submission_pdf_shape_audit.md`

Result:

`paper/main_submission.pdf` builds from `paper/main.tex` with
`LOTTERYMAINONLY` set, excluding the appendix/generated evidence tables and
rewriting the appendix-table pointer to a supplemental-text reference. The
submission PDF is currently 9 pages against a 10-page local budget, and its
LaTeX log passes the same unresolved-warning/overfull check as `paper/main.pdf`.

## 2026-05-07 - CI Paper-build Gate

Purpose:

Close the remaining configured-CI packaging gap where GitHub Actions verified
generated artifacts but did not compile the appendix-inclusive and main-only
paper PDFs.

Implementation:

- Updated `.github/workflows/check.yml` to install `make`, `poppler-utils`,
  `ripgrep`, and the
  lightweight TeX package set used by the artifact-verification container.
- Changed the workflow command from `make ci-check PYTHON=python` to
  `make ci-check paper-check PYTHON=python`.
- Extended `scripts/verify_research_artifacts.py` to require the CI paper-build
  command and TeX/poppler/ripgrep dependencies.
- Added the `paper/main_submission.*` transient LaTeX outputs to
  `.dockerignore`.

Result:

The configured GitHub Actions gate now covers generated statistics, release
manifest checks, PDF text/page inspection, the appendix-inclusive
`paper/main.pdf`, and the main-only `paper/main_submission.pdf`. This still
needs an externally observed green run after the project is pushed to a real
repository.

## 2026-05-07 - Venue Compliance Audit and Abstract Trim

Purpose:

Separate local content readiness from true venue binding. The main-only PDF was
within the page budget, but the source still used a generic `article`/`geometry`
layout rather than a concrete top-conference style.

Implementation:

- Shortened the paper abstract from 268 words to 238 words while preserving the
  negative support-equivalence claim and explicit CIFAR covariance limitation.
- Added `scripts/audit_venue_submission_compliance.py`.
- Added `runs/venue_submission_compliance_audit.json` and
  `docs/venue_submission_compliance_audit.md`.
- Wired the audit into `make check`, `make ci-check`, `make verify`, the public
  release manifest, and `scripts/verify_research_artifacts.py`.
- Updated submission-readiness and thread-goal audit docs to distinguish
  content-packet readiness from missing venue/style binding.

Result:

The audit marks the content packet ready under a generic ML top-conference
proxy: anonymous author, 238-word abstract, 9-page main-only PDF, figures,
bibliography, and conditional appendix exclusion. It deliberately keeps
venue-submission readiness false because no concrete target venue/year is
selected and no official conference LaTeX style is applied yet.

## 2026-05-07 - NeurIPS 2026 Style Binding

Purpose:

Move the paper from a generic top-conference proxy into a concrete NeurIPS 2026
submission shape while keeping final upload readiness honest.

Implementation:

- Added the official `paper/neurips_2026.sty` style file and
  `paper/neurips_checklist.tex`.
- Added `LOTTERYNEURIPS` source wiring in `paper/main.tex`, with NeurIPS
  natbib options, bibliography before appendix, and checklist inclusion after
  the appendix.
- Added `make paper-neurips-check`, producing `paper/neurips_submission.pdf`.
- Updated `scripts/audit_venue_submission_compliance.py` to target NeurIPS 2026,
  count main-content pages before the References heading, and separate content,
  venue binding, checklist-release, and external-packaging readiness.
- Added NeurIPS PDF/style/checklist artifacts to the release manifest and
  verifier.
- Added `poppler-utils` to the CPU container and CI system packages because the
  NeurIPS audit uses `pdftotext` to locate the References heading.
- Added `texlive-latex-extra` to the CPU container and CI system packages
  because the official NeurIPS 2026 style imports `environ.sty`.

Result:

`make paper-check` now builds `paper/main.pdf`, `paper/main_submission.pdf`, and
`paper/neurips_submission.pdf`. The NeurIPS audit reports 8 main-content pages
before References against the 9-page budget, official style/checklist binding
ready, and final venue-submission readiness still false because public
code/data release metadata, compute and asset metadata, external CI/GPU
validation, and public repository upload remain unverified.

## 2026-05-07 - Checklist Release Metadata Inventory

Purpose:

Close the local NeurIPS checklist metadata gaps that can be resolved without
externally publishing the repository.

Implementation:

- Added `docs/compute_resource_accounting.md` for CPU artifact-verifier,
  CUDA training, dense-covariance memory, storage, and wall-clock limitations.
- Added `docs/asset_license_inventory.md` for dataset, NeurIPS style, and
  dependency source/license notes while keeping raw benchmark data outside the
  release manifest.
- Added `docs/new_asset_inventory.md` for generated source, paper, figure,
  table, run-summary, saved-mask, container, and excluded-cache assets.
- Updated `paper/neurips_checklist.tex` so compute resources, existing assets,
  and new assets point to those inventories.
- Updated `scripts/audit_venue_submission_compliance.py` and
  `scripts/verify_research_artifacts.py` so the checklist metadata docs are
  verifier-checked and only unresolved public code/data upload remains as a
  local checklist-release risk.

Result:

The venue compliance audit now reports checklist release risk
`public_code_or_data_not_yet_open` only. Final venue submission is still not
ready because external CI/GPU validation and public repository upload remain
unverified.

## 2026-05-07 - Release Anonymization Audit

Purpose:

Prevent anonymous-review release artifacts from leaking local workstation
identity through generated manifests, audit JSON, or documentation paths.

Implementation:

- Added `scripts/audit_release_anonymization.py`.
- Changed the public release manifest root from an absolute local path to `.`.
- Changed the mode-distribution equivalence audit to write relative run paths.
- Redacted the local hostname from `docs/environment_snapshot.md`.
- Added `make release-anonymization-audit` and wired the audit into `make check`,
  `make ci-check`, `make verify`, and `scripts/verify_research_artifacts.py`.
- Updated release/reproducibility docs so the anonymization gate is part of the
  local artifact-verification story.

Result:

The release package now has a local gate that scans manifest-included text plus
the manifest itself for local usernames, hostnames, and absolute workstation
paths. Public upload and external CI/GPU validation remain separate unresolved
steps.

## 2026-05-07 - Public Release Archive Gate

Purpose:

Turn the anonymized manifest package into a concrete local tarball that can be
uploaded as supplementary material or mirrored to an anonymous public
repository.

Implementation:

- Added `scripts/build_public_release_archive.py`.
- Added `make release-archive` and wired it into `make check`, `make ci-check`,
  `make verify`, and `scripts/verify_research_artifacts.py`.
- The archive builder creates
  `dist/lottery_artifact_public_release_2026-05-06.tar.gz` and a `.sha256`
  sidecar from the release manifest plus manifest/anonymization metadata.
- The generated `docs/public_release_archive_audit.md` and
  `runs/public_release_archive_audit.json` verify member counts, member sizes,
  safe relative tar paths, archive SHA256, and metadata sidecar inclusion.
- Added `dist/` to `.dockerignore` and `.gitignore` so generated release
  tarballs do not inflate the container build context or repository state.

Result:

The project now has a local anonymous-review release archive gate. Public
upload, externally observed CI, and external GPU-container validation remain
separate unresolved steps.

## 2026-05-07 - Project License For Release Package

Purpose:

Remove the local public-release staging gap caused by the missing top-level
project license.

Implementation:

- Added a top-level `LICENSE` file using the MIT License with the anonymous
  review holder string "Anonymous Authors".
- Updated `docs/asset_license_inventory.md` to distinguish the project license
  from third-party dataset, style, and dependency terms.
- Updated `docs/new_asset_inventory.md`, README, reproducibility docs, and
  submission-readiness docs so the local tarball/license state is explicit.
- Extended `scripts/build_release_manifest.py`,
  `scripts/audit_venue_submission_compliance.py`, and
  `scripts/verify_research_artifacts.py` to require the license in the release
  package and release metadata checks.

Result:

The local anonymous-review release archive now includes a top-level project
license. Public upload, externally observed CI, and external GPU-container
validation remain separate unresolved steps.

## 2026-05-07 - Extracted Release Package Smoke

Purpose:

Check that the tarball a reviewer receives is not only member/hash correct, but
also self-verifiable after extraction.

Implementation:

- Added `scripts/smoke_public_release_archive.py`.
- Added `--release-package-mode` to `scripts/verify_research_artifacts.py` so
  extracted packages can skip the outer tarball/archive-audit self-reference.
- Added `make release-archive-smoke` and wired the smoke into `make check`,
  `make ci-check`, and `make verify`.
- Excluded `docs/public_release_archive_smoke.md` and
  `runs/public_release_archive_smoke.json` from the release manifest to avoid
  archive self-reference.

Result:

The release smoke extracts
`dist/lottery_artifact_public_release_2026-05-06.tar.gz`, verifies manifest
hashes for included files, checks release metadata sidecars, and runs
`scripts/verify_research_artifacts.py --release-package-mode` inside the
extracted package.
