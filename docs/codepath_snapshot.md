# Code Path Snapshot

This file records implementation checks that should not be treated as paper
evidence.

| Check | Data | Model | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Fake-CIFAR smoke | Synthetic CIFAR-shaped tensors | ResNet-20 width 4 | Pass | Verifies image-model plumbing without external data. |
| Real CIFAR subset smoke | CIFAR-10, 512 train / 256 test | ResNet-20 width 4 | Pass | Verifies real-data loading and the ResNet-20 IMP/SGLD/control pipeline. Not scientific evidence because accuracy is near chance. |
| Real CIFAR full-data dense baseline | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Verifies that augmentation plus cosine LR trains above chance; 10 epochs reached 83.0% test accuracy. |
| Real CIFAR full-data Gate1 short grid | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Short-training Gate1 grid with meaningful dense/IMP accuracy; r5 is now 5 seeds, r2/r8 remain 3 seeds. Not final submission evidence because posterior budget and training budget are still small. |
| Real CIFAR full-data SWAG short control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed r5 p0.30 SWAG posterior control; fails Gate1 by matching chain-start magnitude. |
| Gem-Miner-style mask source smoke/baseline | Fake-CIFAR, CIFAR-10 subset, and full CIFAR-10 | ResNet-20 width 16 | Pass | STE score-training mask source is wired into the fixed-mask retraining probe; the five-seed full-data selected row is random-like in support and far below IMP. |
| Variational pruning mask source | Fake-CIFAR, CIFAR-10 subset/full, and digits | ResNet-20 width 4/16 / MLP | Pass | Proposal-style Bernoulli/Concrete variational mask source is wired into the fixed-mask retraining probe; the five-seed digits sanity check records accuracy/ECE/Brier and beats random/Gem-Miner-style masks in accuracy but remains below IMP, while the five-seed full CIFAR support row is random-scale in IMP overlap and far below IMP accuracy. |
| Learned-mask calibration/OOD smoke/pilot | Fake-CIFAR, CIFAR-10 subset with Gaussian-noise OOD, and CIFAR-10 4096-sample seed-0 pilot with CIFAR-100 OOD | ResNet-20 width 4/8 | Pass | `run_calibration_ood_probe.py` now evaluates optional random, Gem-Miner-style, and variational-prune hard masks after fixed-mask retraining; subset smoke validates the path, and the large-subset pilot shows learned masks below IMP, but neither is submission evidence. |
| Real CIFAR full-data learned-mask calibration/OOD row | CIFAR-10 ID, CIFAR-100 OOD | ResNet-20 width 16 | Pass | Five-seed row for learned-random, Gem-Miner-style, and variational-prune hard masks. Learned masks improve ECE relative to dense/IMP but are below IMP on accuracy, NLL, Brier, and OOD AUROC. |
| Real CIFAR full-data SGLD multi-chain short control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed r5 p0.30 control with 3 independent dense starts per seed; fails Gate1 even though state/function clusters separate by chain. |
| Real CIFAR full-data SGLD movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed selected r5 p0.30 diagnostic, with one-seed tuning grids; larger SGLD steps move support away from chain start but decrease IMP overlap. |
| Real CIFAR full-data SGHMC movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed selected r5 p0.30 momentum-sampler diagnostic; SGHMC moves support and creates more state clusters, but IMP overlap decreases. |
| Real CIFAR full-data 30-epoch pilots | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Longer-training pilots; epoch-1 rewind fixes r5 IMP accuracy, but single-chain SGLD, SWAG, and 3-chain SGLD controls still fail because posterior support remains a chain-start magnitude proxy. |
| Real CIFAR full-data 30-epoch SGHMC movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind momentum-sampler diagnostic; SGHMC moves support substantially but posterior-to-IMP decreases. |
| Real CIFAR full-data 30-epoch cyclical SGLD movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind exploration diagnostic with cyclic high/low LR; movement increases while posterior-to-IMP decreases. |
| Real CIFAR full-data 30-epoch full-network SWAG movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind 20-snapshot low-rank-plus-diagonal SWAG diagnostic; support moves from chain-start magnitude but remains tied to or below the chain-start IMP overlap. |
| Real CIFAR full-data 30-epoch diagonal Laplace movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind mini-batch diagonal-Fisher Laplace diagnostic; local Gaussian movement increases while posterior-to-IMP decreases. |
| Real CIFAR full-data 30-epoch KFAC-style Laplace movement diagnostic | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind Kronecker-factored empirical-Fisher Laplace diagnostic; structured local Gaussian movement increases while posterior-to-IMP decreases. |
| Real CIFAR full-data 30-epoch exact final-head Laplace probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed exact full-covariance posterior for the frozen final classifier head; head support movement does not improve IMP overlap. |
| Real CIFAR full-data 30-epoch block Laplace probes | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Seed-0 seven-tensor scan plus five-seed full-covariance softmax-GGN/Laplace probes for two single tensors, one joint four-tensor group, and one 22,064-parameter 11-tensor block-diagonal row; selected support moves but does not improve IMP overlap over chain-start or rewind controls. |
| Real CIFAR full-data 30-epoch subspace HMC probes | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed full-network low-dimensional random, trajectory-informed, and top-Hessian HMC probes; samples keep dense accuracy while posterior support stays at or below the matching chain-start IMP overlap. |
| Mode/ticket distribution-equivalence audit | Existing MNIST/Fashion/CIFAR posterior artifacts | Multiple posterior families | Pass | Aggregates posterior observations into KS/Wasserstein/MMD support-overlap comparisons; posterior beats random in 58/59 groups but never beats matched chain-start by more than 0.005 Jaccard. |
| Real CIFAR full-data 30-epoch calibration/OOD probe | CIFAR-10 ID, CIFAR-100 OOD | ResNet-20 width 16 | Pass | Five-seed dense/IMP/SWAG predictive calibration and OOD diagnostics; SWAG improves ECE but worsens accuracy, NLL, and OOD AUROC. |
| Real CIFAR full-data 30-epoch matched dense-trajectory probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed epoch-1 rewind probe using the same dense trajectory for rewind; checkpoint and aggregate trajectory-magnitude supports dominate posterior supports. |
| Real CIFAR full-data 30-epoch trajectory mask retraining probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed fixed-mask retraining from the same epoch-1 rewind state; trajectory-magnitude masks train above random but below IMP. |
| Real CIFAR full-data 30-epoch trajectory residual swap probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed residual-swap control; IMP-only residual support recovers much of the trajectory-mask gap, while non-IMP random residual support does not. |
| Real CIFAR full-data 30-epoch residual anatomy probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed residual decomposition by stage, IMP pruning round, dense-trajectory score predictability, and held-out logistic residual prediction. |
| Real CIFAR full-data 30-epoch residual predictor mask probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed functional mask generation from held-out residual predictors; predictor improves coordinate precision over random but does not recover oracle residual accuracy. |
| Real CIFAR full-data 30-epoch residual cross-seed transfer probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed leave-one-seed-out residual transfer; cross-seed predictors improve coordinate precision over random but do not recover oracle residual accuracy. |
| Real CIFAR full-data 30-epoch activation-aligned residual direct cross-seed support transfer probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed direct source-vote residual support transfer with activation-channel Hungarian alignment controls; other-seed oracle residual coordinates are slightly enriched for target IMP-only weights, but aligned and unaligned source-vote masks remain base/random-like and below the target oracle residual. |
| Real CIFAR full-data 30-epoch residual base-compatibility probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed matched-base control; IMP-overlap-matched random bases are weak alone, but top IMP-only oracle residual additions recover trajectory-oracle accuracy, while random residual additions stay weak. |
| Real CIFAR full-data 30-epoch residual posterior-decomposition probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed matched-base decomposition control; posterior-RMS-ranked IMP-only residual additions are closely matched by dense-final-magnitude ranking, while posterior RMS-minus-dense and posterior-std rankings fail. |
| Real CIFAR full-data 30-epoch residual stratified control probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed stratified residual controls; random IMP-only support recovers part of the oracle gain, while parameter/score-matched non-IMP support does not. |
| Real CIFAR full-data 30-epoch residual removal-order controls | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed removal-order controls; top IMP-only additions remain effective under low, random, or high base-only removals, while non-IMP additions remain weak. |
| Real CIFAR full-data 30-epoch residual IMP-process probe | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed replay-style process control; round-survivor additions increasingly concentrate final IMP residual support and improve accuracy as IMP rounds progress. |
| Real CIFAR full-data 30-epoch residual IMP-process ranking controls | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed process-ranking controls; top round-survivor additions outperform random and low-score survivor additions, separating round membership from round-trained score ordering. |
| Real CIFAR full-data 30-epoch residual IMP-process oracle-overlap-matched controls | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed final-IMP oracle-overlap-matched random controls; round-score final-IMP residuals beat matched random in 35/45 paired comparisons with mean accuracy delta +0.0020, so score ordering is not explained solely by final-IMP membership or oracle-overlap amount. |
| Real CIFAR full-data 30-epoch residual IMP-process score-source controls | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed final-IMP score-source controls; round-trained scores beat dense-score controls in 37/45 and base-score controls in 39/45 paired comparisons, so the process ordering is not just dense/base magnitude ranking. |
| Real CIFAR full-data 30-epoch residual IMP-process round-exclusion controls | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed process intervention; after removing round-selected final-IMP residual additions, even the best remaining final-IMP-magnitude replacements lose to the round-selected masks in 44/45 paired comparisons with mean delta +0.0061. |
| Real CIFAR full-data 30-epoch residual IMP-process tensor-matched round-exclusion control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed RMS trajectory round-5 process intervention; after removing round-selected final-IMP residual additions, parameter-tensor-matched final-IMP-magnitude replacements lose to round-selected masks in 5/5 seeds with mean delta +0.0091. |
| Real CIFAR full-data 30-epoch residual IMP-process tensor+score round-exclusion control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed RMS trajectory round-5 process intervention; tensor+score-matched replacements narrow the gap but still lose to round-selected masks in 5/5 seeds with mean delta +0.0041. |
| Real CIFAR full-data 30-epoch residual IMP-process residualized projection control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed RMS trajectory round-5 projection control; removing base/dense/final-IMP magnitude from the round score lowers accuracy by +0.0041 for the original score and drops oracle overlap from 0.6684 to 0.4854. |
| Real CIFAR full-data 30-epoch residual IMP-process posterior projection control | CIFAR-10 full train/test | ResNet-20 width 16 | Pass | Five-seed RMS trajectory round-5 posterior-projection control; additionally removing diagonal-Laplace posterior RMS/std/excess features leaves the original round score higher in 5/5 seeds and drops oracle overlap from 0.6773 to 0.4850. |

Current CIFAR subset smoke:

- Run: `runs/cifar10_resnet20_subset_smoke/20260503_105959/metrics.json`
- Summary: `runs/cifar10_resnet20_subset_smoke_summary.json`
- Dense accuracy: 0.0781
- IMP accuracy: 0.0742
- Posterior-to-IMP Jaccard: 0.9742
- Dense-magnitude-to-IMP Jaccard: 0.9901
- Posterior-to-chain-start Jaccard: 0.9760

The high Jaccard values here mostly reflect a tiny, undertrained run where
dense, initial, and chain-start magnitude supports are almost identical. The
next valid CIFAR milestone is full-data training across seeds and sparsities.

Current CIFAR full-data baseline:

- Run: `runs/cifar10_resnet20_baseline/20260503_135923/metrics.json`
- Dense-only setting: ResNet-20 width 16, 10 epochs, batch 512, LR 0.1,
  cosine schedule, weight decay 5e-4, CIFAR crop/flip augmentation.
- Final test accuracy: 0.8302.

Current CIFAR full-data Gate1 short grid:

- Summaries:
  - `runs/cifar10_resnet20_gate1_short_r2_p0p3_summary.json`
  - `runs/cifar10_resnet20_gate1_short_r5_p0p3_summary.json`
  - `runs/cifar10_resnet20_gate1_short_r8_p0p3_summary.json`
- Table: `docs/cifar10_resnet20_gate1_short_sweep.md`

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r2 p0.30 | 3 | 0.8239 | 0.8491 | 0.5100 | 0.3446 | 0.3245 | 0.3445 | 0.9967 | 0.4122 | fail |
| r5 p0.30 | 5 | 0.8250 | 0.8634 | 0.8319 | 0.1297 | 0.0917 | 0.1297 | 0.9963 | 0.1704 | fail |
| r8 p0.30 | 3 | 0.8292 | 0.8534 | 0.9424 | 0.0786 | 0.0296 | 0.0785 | 0.9969 | 0.1064 | fail |

This first full-data short grid reproduces the same control failure in a
non-chance CIFAR setting. The next valid CIFAR milestone is a submission-grade
budget: more seeds, longer training, independent dense chains, more posterior
samples, and higher-fidelity posterior controls.

Current CIFAR full-data SWAG short control:

- Summary: `runs/cifar10_resnet20_swag_short_r5_p0p3_summary.json`
- Eval: `runs/cifar10_resnet20_swag_short_r5_p0p3_gate1_eval.json`
- Table: `docs/cifar10_resnet20_swag_short_r5_p0p3.md`
- Setting: r5 p0.30, 5 seeds, 10 training epochs, 1 independent dense start,
  5 SWAG collection epochs, 10 posterior samples per seed.

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 SWAG | 5 | 0.8255 | 0.8631 | 0.8319 | 0.1302 | 0.0917 | 0.1304 | 0.9097 | 0.1703 | fail |

SWAG moves supports farther from the chain start than the very-low-step-size
short SGLD sampler, but it does not rescue the posterior-mode claim. The SWAG
posterior mask overlap does not exceed the independent dense chain-start
magnitude mask, and the posterior support remains highly overlapping with the
chain-start support.

Current CIFAR full-data SGLD multi-chain short control:

- Summary: `runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_summary.json`
- Eval: `runs/cifar10_resnet20_sgld_multichain_short_r5_p0p3_gate1_eval.json`
- Table: `docs/cifar10_resnet20_sgld_multichain_short_r5_p0p3.md`
- Setting: r5 p0.30, 5 seeds, 10 training epochs, 3 independent dense starts
  per seed, 10 SGLD samples per chain.

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 SGLD-3chain | 5 | 0.8215 | 0.8621 | 0.8319 | 0.1291 | 0.0917 | 0.1291 | 0.9963 | 0.1689 | fail |

This control records three state clusters and three function clusters per seed,
confirming that the chains are separated. The support-level posterior signal
still matches each chain-start magnitude mask, so the single-chain objection
does not rescue the posterior-mode claim in this short CIFAR setting.

Current CIFAR full-data SGLD movement diagnostic:

- One-seed full LR tuning grids:
  - `runs/cifar10_resnet20_sgld_movement_short_r5_p0p3/20260503_184103/metrics.json`
  - `runs/cifar10_resnet20_sgld_movement_short_highlr_r5_p0p3/20260503_184406/metrics.json`
- Five-seed selected-grid summary:
  `runs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3_summary.csv`
- Table: `docs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3.md`
- Setting: r5 p0.30, 5 seeds, 10 training epochs, one dense-start SGLD chain,
  selected SGLD LRs 1e-10, 1e-6, and 3e-6.

| SGLD LR | Runs | Dense Acc | IMP Acc | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Sample Acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-10 | 5 | 0.8218 | 0.8647 | 0.1730 | 0.0918 | 0.1729 | 0.9963 | 0.1729 | 0.8225 |
| 1e-6 | 5 | 0.8218 | 0.8647 | 0.1683 | 0.0917 | 0.1729 | 0.6932 | 0.1729 | 0.8169 |
| 3e-6 | 5 | 0.8218 | 0.8647 | 0.1602 | 0.0917 | 0.1729 | 0.5440 | 0.1729 | 0.7954 |

This selected grid confirms that SGLD can move support away from the dense
chain start in CIFAR, but the movement reduces IMP alignment rather than
improving it. The result is a movement diagnostic, not a higher-fidelity
posterior baseline.

Current CIFAR full-data SGHMC movement diagnostic:

- Implementation: `src/lottery/sghmc.py`
- One-seed full LR tuning grid:
  `runs/cifar10_resnet20_sghmc_movement_short_r5_p0p3_summary.csv`
- Five-seed selected-grid summary:
  `runs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3_summary.csv`
- Table: `docs/cifar10_resnet20_sghmc_movement_short_selected_r5_p0p3.md`
- Setting: r5 p0.30, 5 seeds, 10 training epochs, one dense-start SGHMC chain,
  momentum decay 0.9, selected SGHMC LRs 1e-10, 3e-8, and 1e-7.

| SGHMC LR | Runs | Dense Acc | IMP Acc | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Sample Acc | State Clusters | Function Clusters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-10 | 5 | 0.8183 | 0.8629 | 0.1702 | 0.0917 | 0.1701 | 0.9848 | 0.1701 | 0.8204 | 6.0 | 2.0 |
| 3e-8 | 5 | 0.8183 | 0.8629 | 0.1682 | 0.0917 | 0.1701 | 0.7733 | 0.1701 | 0.8178 | 6.0 | 2.8 |
| 1e-7 | 5 | 0.8183 | 0.8629 | 0.1637 | 0.0917 | 0.1701 | 0.6329 | 0.1701 | 0.8107 | 6.0 | 1.8 |

SGHMC is a useful sampler-family control because it moves supports and records
more state clusters than the short SGLD movement row. The support result still
fails the rescue condition: movement decreases posterior-to-IMP overlap instead
of increasing it.

Current CIFAR full-data 30-epoch pilots:

- Summary: `runs/cifar10_resnet20_long30_r5_p0p3_summary.json`
- Eval: `runs/cifar10_resnet20_long30_r5_p0p3_gate1_eval.json`
- Table: `docs/cifar10_resnet20_long30_r5_p0p3.md`
- Setting: r5 p0.30, 1 seed, 30 training epochs, 1 independent dense start,
  10 SGLD samples.
- Gradual-pruning summary: `runs/cifar10_resnet20_long30_r8_p0p2_summary.json`
- Gradual-pruning eval: `runs/cifar10_resnet20_long30_r8_p0p2_gate1_eval.json`
- Gradual-pruning table: `docs/cifar10_resnet20_long30_r8_p0p2.md`
- Gradual-pruning setting: r8 p0.20, 1 seed, 30 training epochs, about the same
  r5-level sparsity, 1 independent dense start, 10 SGLD samples.
- Rewind summary: `runs/cifar10_resnet20_long30_rewind1_r5_p0p3_summary.json`
- Rewind eval: `runs/cifar10_resnet20_long30_rewind1_r5_p0p3_gate1_eval.json`
- Rewind table: `docs/cifar10_resnet20_long30_rewind1_r5_p0p3.md`
- Rewind setting: r5 p0.30, 5 seeds, 30 training epochs, IMP rewound to epoch 1,
  1 independent dense start, 10 SGLD samples.
- Rewind SWAG summary:
  `runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_summary.json`
- Rewind SWAG eval:
  `runs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3_gate1_eval.json`
- Rewind SWAG table: `docs/cifar10_resnet20_long30_rewind1_swag_r5_p0p3.md`
- Rewind SWAG setting: r5 p0.30, 5 seeds, 30 training epochs, IMP rewound to
  epoch 1, 1 independent dense start, 5 SWAG collection epochs, 10 samples.
- Rewind calibration/OOD summary:
  `runs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3_summary.csv`
- Rewind calibration/OOD table:
  `docs/cifar10_resnet20_long30_rewind1_calibration_ood_swag_r5_p0p3.md`
- Rewind calibration/OOD setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, CIFAR-10 ID, CIFAR-100 OOD, dense/IMP/SWAG predictive
  sources, 5 SWAG collection epochs, 10 SWAG samples.
- Rewind SGLD-3chain summary:
  `runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_summary.json`
- Rewind SGLD-3chain eval:
  `runs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3_gate1_eval.json`
- Rewind SGLD-3chain table:
  `docs/cifar10_resnet20_long30_rewind1_sgld3_r5_p0p3.md`
- Rewind SGLD-3chain setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, 3 independent dense starts per seed, 10 SGLD samples per
  chain.
- Rewind SGLD movement summary:
  `runs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3_summary.csv`
- Rewind SGLD movement table:
  `docs/cifar10_resnet20_long30_rewind1_sgld_movement_selected_r5_p0p3.md`
- Rewind SGLD movement setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, one dense-start SGLD chain, selected SGLD LRs 1e-10,
  1e-6, and 3e-6.
- Rewind SGHMC movement summary:
  `runs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3_summary.csv`
- Rewind SGHMC movement table:
  `docs/cifar10_resnet20_long30_rewind1_sghmc_movement_selected_r5_p0p3.md`
- Rewind SGHMC movement setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, one dense-start SGHMC chain, momentum decay 0.9, selected
  SGHMC LRs 1e-10, 3e-8, 1e-7, and 3e-7.
- Rewind cyclical SGLD movement summary:
  `runs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3_summary.csv`
- Rewind cyclical SGLD movement table:
  `docs/cifar10_resnet20_long30_rewind1_csgld_movement_selected_r5_p0p3.md`
- Rewind cyclical SGLD movement setting: r5 p0.30, 5 seeds, 30 training
  epochs, IMP rewound to epoch 1, one dense-start cyclical SGLD chain, max LRs
  1e-10, 1e-6, 3e-6, and 1e-5, 50-step cycles, second-half cycle sampling.
- Rewind diagonal Laplace movement summary:
  `runs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3_summary.csv`
- Rewind diagonal Laplace movement table:
  `docs/cifar10_resnet20_long30_rewind1_diag_laplace_movement_selected_r5_p0p3.md`
- Rewind diagonal Laplace movement setting: r5 p0.30, 5 seeds, 30 training
  epochs, IMP rewound to epoch 1, one dense-start mini-batch diagonal-Fisher
  Laplace approximation, scales 1e-10, 1e-3, 3e-3, and 1e-2. This is not exact
  Hessian, KFAC, or full-covariance Laplace evidence.
- Rewind KFAC-style Laplace movement summary:
  `runs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3_summary.csv`
- Rewind KFAC-style Laplace movement table:
  `docs/cifar10_resnet20_long30_rewind1_kfac_laplace_movement_selected_r5_p0p3.md`
- Rewind KFAC-style Laplace movement setting: r5 p0.30, 5 seeds, 30 training
  epochs, IMP rewound to epoch 1, one dense-start KFAC-style empirical-Fisher
  Laplace approximation, scales 1e-10, 1e-4, 1e-3, and 1e-2. This is not exact
  Hessian or full-covariance Laplace evidence.
- Rewind exact final-head Laplace summary:
  `runs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3_summary.csv`
- Rewind exact final-head Laplace table:
  `docs/cifar10_resnet20_long30_rewind1_head_laplace_selected_r5_p0p3.md`
- Rewind exact final-head Laplace setting: r5 p0.30, 5 seeds, 30 training
  epochs, IMP rewound to epoch 1, frozen ResNet-20 feature extractor, exact
  full-covariance softmax Hessian for the 650-parameter final linear head,
  scales 1e-6, 1e-3, 1e-2, and 1. This is exact for the final head only, not a
  full-network posterior.
- Rewind block Laplace summary:
  `runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3_summary.csv`
- Rewind block Laplace table:
  `docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer1_0_conv1_r5_p0p3.md`
- Rewind block Laplace scan summary:
  `runs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3_summary.csv`
- Rewind block Laplace scan table:
  `docs/cifar10_resnet20_long30_rewind1_block_laplace_scan_seed0_r5_p0p3.md`
- Rewind layer3 shortcut block Laplace summary:
  `runs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3_summary.csv`
- Rewind layer3 shortcut block Laplace table:
  `docs/cifar10_resnet20_long30_rewind1_block_laplace_selected_layer3_0_shortcut_r5_p0p3.md`
- Rewind joint block Laplace tune summary:
  `runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3_summary.csv`
- Rewind joint block Laplace tune table:
  `docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_tune_seed0_r5_p0p3.md`
- Rewind joint block Laplace selected summary:
  `runs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3_summary.csv`
- Rewind joint block Laplace selected table:
  `docs/cifar10_resnet20_long30_rewind1_joint_block_laplace_selected_conv1_l1c1_l3shortcut_fc_r5_p0p3.md`
- Rewind block-diagonal Laplace selected summary:
  `runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv`
- Rewind block-diagonal Laplace selected table:
  `docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md`
- Rewind block Laplace setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, batch size 256, frozen rest of network, full-covariance
  softmax-GGN/Laplace covariance for the 2304-parameter
  `layer1.0.conv1.weight` tensor, 2 Hessian batches scaled to dataset size,
  scale 1e-3, and 5 posterior samples per seed. This is exact full covariance
  for the selected block only, not a full-network posterior.
- Rewind block Laplace scan setting: seed 0, same training setup, seven
  selected tensors up to 4608 parameters at scale 1e-3. The only mildly positive
  scan candidate, `layer3.0.shortcut.0.weight`, was promoted to a five-seed
  selected row; it keeps sample accuracy at 0.8905 and moves block support
  (post-chain 0.3626), but posterior-to-IMP is 0.2402 versus chain-start 0.2411
  and rewind 0.3050.
- Rewind joint block Laplace setting: r5 p0.30, 5 seeds, 30 training epochs,
  IMP rewound to epoch 1, batch size 256, frozen rest of network,
  full-covariance softmax-GGN/Laplace covariance over the joint 5424-parameter
  group `conv1.weight`, `layer1.0.conv1.weight`,
  `layer3.0.shortcut.0.weight`, and `fc.weight`; 2 Hessian batches scaled to
  dataset size, scale 1e-4, and 5 posterior samples per seed. This includes
  cross-tensor covariance inside the selected group, but it is not a
  full-network posterior.
- Rewind block-diagonal Laplace setting: r5 p0.30, 5 seeds, 30 training epochs,
  IMP rewound to epoch 1, batch size 512, 11 automatically selected weight
  tensors up to 5000 parameters each, 22,064 selected parameters total, exact
  full-covariance softmax-GGN/Laplace covariance per tensor, independent
  tensor-block sampling, 1 Hessian batch scaled to dataset size, scale 1e-4,
  and 10 posterior samples per seed. This is a wider exact covariance subset
  than the selected/joint rows, but still not dense full-network covariance.
- Rewind subspace HMC summary:
  `runs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3_summary.csv`
- Rewind subspace HMC table:
  `docs/cifar10_resnet20_long30_rewind1_subspace_hmc_selected_scale10_step3e3_r5_p0p3.md`
- Rewind subspace HMC setting: r5 p0.30, 5 seeds, 30 training epochs, IMP
  rewound to epoch 1, full-network 8-dimensional random orthonormal subspace
  around the dense checkpoint, direction scale 10, HMC step size 3e-3, 20 HMC
  steps, 2 leapfrog steps, 4 burn-in steps, 4-step thinning, frozen batchnorm
  statistics, and full-data deterministic train-loader potential. This is a
  full-network low-dimensional HMC check, not exact full-covariance posterior
  evidence.
- Rewind trajectory-subspace HMC summary:
  `runs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3_summary.csv`
- Rewind trajectory-subspace HMC table:
  `docs/cifar10_resnet20_long30_rewind1_trajectory_subspace_hmc_selected_r5_p0p3.md`
- Rewind trajectory-subspace HMC setting: r5 p0.30, 5 seeds, 30 training
  epochs, IMP rewound to epoch 1, full-network 6-dimensional orthonormal
  subspace spanned by dense checkpoint directions from epochs 0, 1, 2, 5, 10,
  and 20 to the epoch-30 dense checkpoint, direction scale 10, HMC step sizes
  3e-4 and 1e-3, 20 HMC steps, 2 leapfrog steps, 4 burn-in steps, 4-step
  thinning, frozen batchnorm statistics, and full-data deterministic
  train-loader potential.
- Rewind Hessian-subspace HMC summary:
  `runs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3_summary.csv`
- Rewind Hessian-subspace HMC table:
  `docs/cifar10_resnet20_long30_rewind1_hessian_subspace_hmc_selected_r5_p0p3.md`
- Rewind Hessian-subspace HMC setting: r5 p0.30, 5 seeds, 30 training epochs,
  IMP rewound to epoch 1, full-network 4-dimensional randomized top-Hessian
  subspace around the dense checkpoint, 2 Hessian-vector-product batches, 1
  subspace iteration, HMC step size 3e-4, 20 HMC steps, 2 leapfrog steps,
  5 burn-in steps, 3-step thinning, frozen batchnorm statistics, and
  full-data deterministic train-loader potential.
- Rewind Hessian16-subspace HMC tune:
  `runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- Rewind Hessian16-subspace HMC tune table:
  `docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_tune_seed0_r5_p0p3.md`
- Rewind Hessian32-subspace HMC tune:
  `runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3_summary.csv`
- Rewind Hessian32-subspace HMC tune table:
  `docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_tune_seed0_r5_p0p3.md`
- Rewind Hessian16-subspace HMC selected summary:
  `runs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3_summary.csv`
- Rewind Hessian16-subspace HMC selected table:
  `docs/cifar10_resnet20_long30_rewind1_hessian16_subspace_hmc_selected_r5_p0p3.md`
- Rewind matched dense-trajectory summary:
  `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_summary.csv`
- Rewind matched dense-trajectory aggregate/layer summaries:
  `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_aggregate_summary.csv`,
  `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_group_summary.csv`,
  and
  `runs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2_layer_summary.csv`
- Rewind matched dense-trajectory table:
  `docs/cifar10_resnet20_long30_rewind1_trajectory_probe_r5_p0p3_v2.md`
- Rewind matched dense-trajectory figure:
  `paper/figures/cifar_trajectory.pdf`
- Rewind matched dense-trajectory setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to the epoch-1 checkpoint from the same dense
  trajectory, checkpoint supports at epochs 0, 1, 2, 5, 10, 20, and 30, plus
  aggregate trajectory score masks and stage/layer overlap diagnostics.
- Rewind trajectory mask retraining summary:
  `runs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3_summary.csv`
- Rewind trajectory mask retraining table:
  `docs/cifar10_resnet20_long30_rewind1_trajectory_mask_training_r5_p0p3.md`
- Rewind trajectory mask retraining setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, then fixed candidate masks trained
  for 30 epochs from the same epoch-1 rewind state.
- Rewind trajectory residual-swap summary:
  `runs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3_summary.csv`
- Rewind trajectory residual-swap table:
  `docs/cifar10_resnet20_long30_rewind1_trajectory_residual_r5_p0p3.md`
- Rewind trajectory residual-swap setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, alpha values 0, 0.5, and 1.0, and one
  same-size non-IMP random residual control per nonzero alpha.
- Rewind residual-anatomy summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3_summary_global.csv`
- Rewind residual-anatomy table:
  `docs/cifar10_resnet20_long30_rewind1_residual_anatomy_r5_p0p3.md`
- Rewind residual-anatomy setting: r5 p0.30, 5 seeds, 30 dense trajectory
  epochs, IMP rewound to epoch 1, base masks `epoch_30`, `traj_rms_abs`, and
  `epoch_10`, residual decomposition by stage/layer and IMP pruning round, and
  held-out logistic prediction from dense-trajectory rank features plus stage
  indicators.
- Rewind residual-predictor mask summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3_summary.csv`
- Rewind residual-predictor mask table:
  `docs/cifar10_resnet20_long30_rewind1_residual_predictor_mask_r5_p0p3.md`
- Rewind residual-predictor mask setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, held-out logistic residual prediction, half
  residual replacement, oracle residual control, and same-size held-out random
  residual control.
- Rewind residual cross-seed transfer summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3_summary.csv`
- Rewind residual cross-seed transfer table:
  `docs/cifar10_resnet20_long30_rewind1_residual_cross_seed_transfer_r5_p0p3.md`
- Rewind residual cross-seed transfer setting: r5 p0.30, 5 target seeds, 30
  dense trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, leave-one-seed-out logistic residual
  prediction from the other four seeds, half residual replacement, oracle
  residual control, and same-size target random residual control.
- Rewind residual activation-aligned direct cross-seed support transfer summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3_summary.csv`
- Rewind residual activation-aligned direct cross-seed support transfer table:
  `docs/cifar10_resnet20_long30_rewind1_residual_aligned_direct_transfer_r5_p0p3.md`
- Rewind residual activation-aligned direct cross-seed support transfer setting:
  r5 p0.30,
  5 target seeds, 30 dense trajectory epochs, IMP rewound to epoch 1, base
  masks `epoch_30`, `traj_rms_abs`, and `epoch_10`, source-vote additions from
  the other four seeds' oracle residual coordinates, matched source-vote random
  and target-random controls, activation-channel Hungarian source-to-target
  alignment controls, and target oracle residual control.
- Rewind residual base-compatibility summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3_summary.csv`
- Rewind residual base-compatibility table:
  `docs/cifar10_resnet20_long30_rewind1_residual_base_compatibility_r5_p0p3.md`
- Rewind residual base-compatibility setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, trajectory bases and per-parameter random
  bases preserving the same IMP/non-IMP counts, half residual replacement,
  oracle IMP-only additions, and same-size random residual additions.
- Rewind residual posterior-decomposition summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3_summary.csv`
- Rewind residual posterior-decomposition table:
  `docs/cifar10_resnet20_long30_rewind1_residual_posterior_decomposition_r5_p0p3.md`
- Rewind residual posterior-decomposition setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, only IMP-overlap-matched random bases,
  half residual replacement, top oracle IMP-only additions,
  dense-final-magnitude-ranked IMP-only additions, diagonal-Laplace
  posterior-RMS-ranked IMP-only additions, posterior RMS-minus-dense additions,
  posterior-std additions, and uniformly random IMP-only additions.
- Rewind residual stratified-control summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3_summary.csv`
- Rewind residual stratified-control table:
  `docs/cifar10_resnet20_long30_rewind1_residual_stratified_controls_r5_p0p3.md`
- Rewind residual stratified-control setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, half residual replacement, shared oracle
  removals, random IMP-only additions, global non-IMP additions, and
  parameter/score-decile-matched non-IMP additions.
- Rewind residual removal-order control summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3_summary.csv`
- Rewind residual removal-order control table:
  `docs/cifar10_resnet20_long30_rewind1_residual_removal_controls_r5_p0p3.md`
- Rewind residual removal-order control setting: r5 p0.30, 5 seeds, 30 dense
  trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, half residual replacement, fixed top-IMP
  additions, low/random/high base-only removal orders, and same-size non-IMP
  random additions under the low-removal baseline.
- Rewind residual IMP-process summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3_summary.csv`
- Rewind residual IMP-process table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_r5_p0p3.md`
- Rewind residual IMP-process setting: r5 p0.30, 5 seeds, 30 dense trajectory
  epochs, IMP rewound to epoch 1, base masks `epoch_30`, `traj_rms_abs`, and
  `epoch_10`, half residual replacement, IMP process rounds 1, 3, and 5, and
  round-survivor plus final-IMP candidate variants ranked by round-trained
  weights.
- Rewind residual IMP-process ranking-control summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3_summary.csv`
- Rewind residual IMP-process ranking-control table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_controls_r5_p0p3.md`
- Rewind residual IMP-process ranking-control setting: r5 p0.30, 5 seeds,
  30 dense trajectory epochs, IMP rewound to epoch 1, base masks `epoch_30`,
  `traj_rms_abs`, and `epoch_10`, half residual replacement, IMP process
  rounds 1, 3, and 5, and top, random, and low-score round-survivor additions.
- Residual IMP-process oracle-overlap-matched smoke summary:
  `runs/fake_cifar10_residual_imp_process_oracle_matched_smoke_summary.csv`
- Residual IMP-process oracle-overlap-matched smoke table:
  `docs/fake_cifar10_residual_imp_process_oracle_matched_smoke.md`
- CIFAR subset residual IMP-process oracle-overlap-matched smoke summary:
  `runs/cifar10_subset_residual_imp_process_oracle_matched_smoke_summary.csv`
- CIFAR subset residual IMP-process oracle-overlap-matched smoke table:
  `docs/cifar10_subset_residual_imp_process_oracle_matched_smoke.md`
- Residual IMP-process oracle-overlap-matched smoke setting: `fake-cifar10`
  and CIFAR-10 subset checks for the `final-imp-oracle-matched-random`
  variant. The new variant exactly matches the round final-IMP oracle-overlap
  precision while preserving final-IMP precision 1.0 in both smokes.
- Rewind residual IMP-process oracle-overlap-matched seed0 pilot summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3_summary.csv`
- Rewind residual IMP-process oracle-overlap-matched seed0 pilot table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_seed0_pilot_r5_p0p3.md`
- Rewind residual IMP-process oracle-overlap-matched seed0 pilot setting:
  r5 p0.30, seed 0, 30 dense trajectory epochs, IMP rewound to epoch 1, base
  masks `epoch_30`, `traj_rms_abs`, and `epoch_10`, half residual replacement,
  process rounds 1, 3, and 5, round final-IMP additions versus random final-IMP
  additions with matched oracle-overlap counts. Round-score additions beat the
  matched-random control in 7 of 9 pairs, with mean accuracy delta +0.0024, but
  two pairs reverse; this is a pilot, not paper evidence.
- Rewind residual IMP-process oracle-overlap-matched 5-seed summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3_summary.csv`
- Rewind residual IMP-process oracle-overlap-matched 5-seed table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_oracle_matched_r5_p0p3.md`
- Rewind residual IMP-process oracle-overlap-matched 5-seed setting:
  r5 p0.30, seeds 0--4, 30 dense trajectory epochs, IMP rewound to epoch 1,
  base masks `epoch_30`, `traj_rms_abs`, and `epoch_10`, half residual
  replacement, process rounds 1, 3, and 5. Round-score final-IMP additions beat
  oracle-overlap-matched random final-IMP additions in 35/45 paired comparisons,
  with mean accuracy delta +0.0020.
- Rewind residual IMP-process score-source smoke summary:
  `runs/fake_cifar10_residual_imp_process_score_source_smoke_summary.csv`
- Rewind residual IMP-process score-source smoke table:
  `docs/fake_cifar10_residual_imp_process_score_source_smoke.md`
- CIFAR subset residual IMP-process score-source smoke summary:
  `runs/cifar10_subset_residual_imp_process_score_source_smoke_summary.csv`
- CIFAR subset residual IMP-process score-source smoke table:
  `docs/cifar10_subset_residual_imp_process_score_source_smoke.md`
- Rewind residual IMP-process score-source 5-seed summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3_summary.csv`
- Rewind residual IMP-process score-source 5-seed table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_score_source_r5_p0p3.md`
- Rewind residual IMP-process score-source 5-seed setting:
  r5 p0.30, seeds 0--4, 30 dense trajectory epochs, IMP rewound to epoch 1,
  base masks `epoch_30`, `traj_rms_abs`, and `epoch_10`, half residual
  replacement, process rounds 1, 3, and 5. The candidate pool is final-IMP
  residual for all variants; dense/base controls rerank it by dense-final or
  base-source magnitude. Round-trained scores beat dense-score controls in
  37/45 paired comparisons and base-score controls in 39/45, with mean deltas
  +0.0026 and +0.0028.
- Rewind residual IMP-process round-exclusion smoke summary:
  `runs/cifar10_subset_residual_imp_process_round_exclusion_smoke_v2_summary.csv`
- Rewind residual IMP-process round-exclusion smoke table:
  `docs/cifar10_subset_residual_imp_process_round_exclusion_smoke_v2.md`
- Rewind residual IMP-process round-exclusion 5-seed summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3_summary.csv`
- Rewind residual IMP-process round-exclusion 5-seed table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_round_exclusion_r5_p0p3.md`
- Rewind residual IMP-process round-exclusion 5-seed setting:
  r5 p0.30, seeds 0--4, 30 dense trajectory epochs, IMP rewound to epoch 1,
  base masks `epoch_30`, `traj_rms_abs`, and `epoch_10`, half residual
  replacement, process rounds 1, 3, and 5. The control removes the
  round-selected final-IMP residual additions from the candidate set, then
  chooses the best remaining final-IMP residual additions by final IMP
  magnitude under the same support budget. Round-selected masks beat this
  replacement in 44/45 paired comparisons, with mean accuracy delta +0.0061.
- Rewind residual IMP-process tensor-matched round-exclusion 5-seed summary:
  `runs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3_summary.csv`
- Rewind residual IMP-process tensor-matched round-exclusion 5-seed table:
  `docs/cifar10_resnet20_long30_rewind1_residual_imp_process_layer_exclusion_r5_p0p3.md`
- Rewind residual IMP-process tensor-matched round-exclusion 5-seed setting:
  r5 p0.30, seeds 0--4, 30 dense trajectory epochs, IMP rewound to epoch 1,
  RMS trajectory base mask, half residual replacement, process round 5. The
  control removes the round-selected final-IMP residual additions, then chooses
  replacements matched by parameter tensor and ranked by final IMP magnitude.
  Round-selected masks beat the tensor-matched replacement in 5/5 paired
  comparisons, with mean accuracy delta +0.0091.

| Config | Runs | Dense Acc | IMP Acc | Sparsity | Posterior | Random | Chain Start | Post-Chain | Dense Mag | Rewind Mag | Gate1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r5 p0.30 30ep init rewind | 1 | 0.8836 | 0.8584 | 0.8319 | 0.1381 | 0.0916 | 0.1381 | 0.9972 | 0.1424 | n/a | fail |
| r8 p0.20 30ep init rewind | 1 | 0.8846 | 0.8581 | 0.8322 | 0.1284 | 0.0915 | 0.1286 | 0.9968 | 0.1339 | n/a | fail |
| r5 p0.30 30ep epoch-1 rewind SGLD | 5 | 0.8859 | 0.8980 | 0.8319 | 0.1342 | 0.0918 | 0.1342 | 0.9969 | 0.1472 | 0.1783 | fail |
| r5 p0.30 30ep epoch-1 rewind SWAG | 5 | 0.8859 | 0.8969 | 0.8319 | 0.1361 | 0.0918 | 0.1361 | 0.9265 | 0.1463 | 0.1786 | fail |
| r5 p0.30 30ep epoch-1 rewind SGLD-3chain | 5 | 0.8860 | 0.8990 | 0.8319 | 0.1368 | 0.0918 | 0.1368 | 0.9969 | 0.1460 | 0.1800 | fail |
| r5 p0.30 30ep epoch-1 rewind SGLD movement 1e-6 | 5 | 0.8845 | 0.8993 | 0.8319 | 0.1425 | 0.0918 | 0.1441 | 0.7362 | 0.1441 | 0.1784 | fail |
| r5 p0.30 30ep epoch-1 rewind SGLD movement 3e-6 | 5 | 0.8845 | 0.8993 | 0.8319 | 0.1381 | 0.0918 | 0.1441 | 0.5928 | 0.1441 | 0.1784 | fail |
| r5 p0.30 30ep epoch-1 rewind SGHMC movement 1e-7 | 5 | 0.8883 | 0.8970 | 0.8319 | 0.1419 | 0.0918 | 0.1457 | 0.6796 | 0.1457 | 0.1777 | fail |
| r5 p0.30 30ep epoch-1 rewind SGHMC movement 3e-7 | 5 | 0.8883 | 0.8970 | 0.8319 | 0.1360 | 0.0918 | 0.1457 | 0.5214 | 0.1457 | 0.1777 | fail |
| r5 p0.30 30ep epoch-1 rewind cSGLD movement 1e-6 | 5 | 0.8852 | 0.8960 | 0.8319 | 0.1422 | 0.0918 | 0.1454 | 0.7046 | 0.1454 | 0.1789 | fail |
| r5 p0.30 30ep epoch-1 rewind cSGLD movement 3e-6 | 5 | 0.8852 | 0.8960 | 0.8319 | 0.1371 | 0.0918 | 0.1454 | 0.5533 | 0.1454 | 0.1789 | fail |
| r5 p0.30 30ep epoch-1 rewind cSGLD movement 1e-5 | 5 | 0.8852 | 0.8960 | 0.8319 | 0.1260 | 0.0918 | 0.1454 | 0.3700 | 0.1454 | 0.1789 | fail |
| r5 p0.30 30ep epoch-1 rewind SWAG20 movement scale 16 | 5 | 0.8859 | 0.8973 | 0.8319 | 0.1454 | 0.0917 | 0.1455 | 0.9528 | 0.1455 | 0.1782 | fail |
| r5 p0.30 30ep epoch-1 rewind SWAG20 movement scale 64 | 5 | 0.8859 | 0.8973 | 0.8319 | 0.1453 | 0.0917 | 0.1455 | 0.9086 | 0.1455 | 0.1782 | fail |
| r5 p0.30 30ep epoch-1 rewind diag Laplace scale 1e-3 | 5 | 0.8849 | 0.8980 | 0.8319 | 0.1447 | 0.0918 | 0.1469 | 0.8826 | 0.1469 | 0.1787 | fail |
| r5 p0.30 30ep epoch-1 rewind diag Laplace scale 3e-3 | 5 | 0.8849 | 0.8980 | 0.8319 | 0.1400 | 0.0918 | 0.1469 | 0.7803 | 0.1469 | 0.1787 | fail |
| r5 p0.30 30ep epoch-1 rewind diag Laplace scale 1e-2 | 5 | 0.8849 | 0.8980 | 0.8319 | 0.1278 | 0.0918 | 0.1469 | 0.5961 | 0.1469 | 0.1787 | fail |
| r5 p0.30 30ep epoch-1 rewind KFAC Laplace scale 1e-4 | 5 | 0.8860 | 0.8956 | 0.8319 | 0.1456 | 0.0917 | 0.1456 | 0.9334 | 0.1456 | 0.1775 | fail |
| r5 p0.30 30ep epoch-1 rewind KFAC Laplace scale 1e-3 | 5 | 0.8860 | 0.8956 | 0.8319 | 0.1441 | 0.0917 | 0.1456 | 0.8016 | 0.1456 | 0.1775 | fail |
| r5 p0.30 30ep epoch-1 rewind KFAC Laplace scale 1e-2 | 5 | 0.8860 | 0.8956 | 0.8319 | 0.1303 | 0.0917 | 0.1456 | 0.4859 | 0.1456 | 0.1775 | fail |
| r5 p0.30 30ep epoch-1 rewind block Laplace layer1.0.conv1 scale 1e-3 | 5 | 0.8970 | 0.9029 | 0.8319 | 0.1329 | 0.0917 | 0.1315 | 0.9557 | 0.1315 | 0.1504 | fail |
| r5 p0.30 30ep epoch-1 rewind block Laplace layer3.0.shortcut scale 1e-3 | 5 | 0.8960 | 0.8996 | 0.8319 | 0.1333 | 0.0918 | 0.1324 | 0.9740 | 0.1324 | 0.1513 | fail |
| r5 p0.30 30ep epoch-1 rewind joint block Laplace conv1+l1c1+l3shortcut+fc scale 1e-4 | 5 | 0.8941 | 0.9037 | 0.8319 | 0.1319 | 0.0918 | 0.1309 | 0.9459 | 0.1309 | 0.1509 | fail |
| r5 p0.30 30ep epoch-1 rewind blockdiag Laplace 11 tensors scale 1e-4 | 5 | 0.8859 | 0.8973 | 0.8319 | 0.1504 | 0.0918 | 0.1468 | 0.8287 | 0.1468 | 0.1797 | fail |
| r5 p0.30 30ep epoch-1 rewind random-subspace HMC step 3e-3 | 5 | 0.8866 | 0.8965 | 0.8319 | 0.1440 | 0.0918 | 0.1440 | 0.9766 | 0.1440 | 0.1779 | fail |
| r5 p0.30 30ep epoch-1 rewind trajectory-subspace HMC step 1e-3 | 5 | 0.8849 | 0.8961 | 0.8319 | 0.2290 | 0.0917 | 0.2292 | 0.9915 | 0.2292 | 0.1792 | fail |
| r5 p0.30 30ep epoch-1 rewind Hessian-subspace HMC step 3e-4 | 5 | 0.8867 | 0.8945 | 0.8319 | 0.1471 | 0.0916 | 0.1471 | 0.9999 | 0.1471 | 0.1810 | fail |
| r5 p0.30 30ep epoch-1 rewind Hessian16-subspace HMC step 3e-4 | 5 | 0.8873 | 0.8974 | 0.8319 | 0.1468 | 0.0917 | 0.1468 | 0.9994 | 0.1468 | 0.1799 | fail |

These are still pilots: the initialization-rewind and gradual-pruning long rows
are one seed each, while the epoch-1 rewind SGLD, SWAG, and SGLD-3chain rows
are five seeds. They show that longer dense
training is viable and that epoch-1 rewinding is the right long-budget IMP axis
to tune: it improves r5 IMP from the one-seed initialization-rewind value of
0.8584 to a 5-seed mean of 0.8980, above the 5-seed dense mean of 0.8859. That
does not rescue the posterior-mode claim. Very-low-step SGLD barely moves from
the chain-start support. SWAG moves the support more, reducing Post-Chain to
0.9265, but still does not exceed the chain-start magnitude mask. The 3-chain
SGLD row records 3.0 state clusters and 3.2 function clusters, yet the posterior
support remains almost identical to each chain-start support and the epoch-1
rewind magnitude control is still closer to IMP. The selected movement row
shows that even when SGLD moves substantially from the chain start, the
posterior support moves away from IMP rather than toward it. The matched SGHMC
movement row gives the same conclusion with momentum dynamics: at LR 1e-7,
post-chain is 0.6796 and sample accuracy is 0.8752, but posterior-to-IMP is
0.1419 versus chain-start magnitude 0.1457; at LR 3e-7, post-chain falls to
0.5214 while posterior-to-IMP falls further to 0.1360.
The cyclical SGLD row gives the same answer for a stronger exploration schedule:
at max LR 1e-6, post-chain is 0.7046 and sample accuracy is 0.8782, but
posterior-to-IMP is 0.1422 versus chain-start magnitude 0.1454; increasing the
max LR to 3e-6 or 1e-5 moves farther from the chain start and reduces
posterior-to-IMP further.
The diagonal Laplace row gives the same answer for a local Gaussian
curvature-weighted approximation: at scale 1e-3, post-chain is 0.8826 and
sample accuracy is 0.8799, but posterior-to-IMP is 0.1447 versus chain-start
magnitude 0.1469; at scales 3e-3 and 1e-2, post-chain falls to 0.7803 and
0.5961 while posterior-to-IMP falls further to 0.1400 and 0.1278.
The KFAC-style Laplace row gives the same answer with structured curvature
factors: at scale 1e-3, post-chain is 0.8016 and sample accuracy is 0.8839,
but posterior-to-IMP is 0.1441 versus chain-start magnitude 0.1456; at scale
1e-2, post-chain falls to 0.4859 while posterior-to-IMP falls to 0.1303.
The full-covariance block Laplace rows extend the covariance check beyond the
final head. For the 2304-parameter `layer1.0.conv1.weight` tensor, scale 1e-3
keeps sample accuracy at 0.8961 and moves block support away from the dense
chain start (block post-chain 0.2236), but block posterior-to-IMP is 0.1959
versus block chain-start 0.2034 and block rewind 0.2423. A seed-0 scan over
seven small and medium tensors found no broad rescue; the promoted
`layer3.0.shortcut.0.weight` row has even larger block movement (post-chain
0.3626) but posterior-to-IMP is 0.2402 versus chain-start 0.2411 and rewind
0.3050. The induced global support changes only slightly: global
posterior-to-IMP is 0.1333 versus global chain-start 0.1324 and global rewind
0.1513 for the layer3 shortcut row. The joint four-tensor row adds
cross-tensor covariance over 5424 selected parameters and keeps sample accuracy
at 0.8922 with group post-chain 0.5088, but group posterior-to-IMP is 0.3294
versus group chain-start 0.3501 and group rewind 0.3637. Its induced global
posterior-to-IMP is 0.1319 versus global chain-start 0.1309 and global rewind
0.1509. The 11-tensor block-diagonal row covers 22,064 parameters and moves
global support more strongly (global post-chain 0.8287) while keeping sample
accuracy at 0.8810, but selected-block posterior-minus-chain is -0.0114,
global posterior-minus-chain is only +0.0036, and global rewind remains closer
by +0.0292.
The subspace HMC rows give full-network low-dimensional HMC checks. In a random
8D subspace, accept rate is 0.7400 and sample accuracy is 0.8863, but
posterior-to-IMP is 0.1440 versus chain-start magnitude 0.1440 and rewind
magnitude 0.1779. In the 6D dense-trajectory checkpoint subspace, the
chain-start support is much stronger at 0.2292, but HMC still does not improve
it: at step 1e-3, accept rate is 0.6900, sample accuracy is 0.8847, post-chain
is 0.9915, and posterior-to-IMP is 0.2290. In a 4D top-Hessian subspace,
accept rate is 0.8600 and sample accuracy is 0.8865, but post-chain is 0.9999
and posterior-to-IMP is 0.1471, matching chain-start magnitude.
A 16D top-Hessian selected row increases local movement without changing the
support conclusion: at step 3e-4, accept rate is 0.8833, sample accuracy is
0.8881, parameter distance is 0.00949, post-chain is 0.9994, and
posterior-to-IMP is 0.14680 versus chain-start 0.14682. A five-seed 32D
top-Hessian selected row makes the same point in a wider curvature subspace: at
step 3e-4, accept rate is 0.9500, sample accuracy is 0.8872, parameter
distance is 0.0104, post-chain is 0.9993, and posterior-to-IMP is 0.14614
versus chain-start 0.14611.
The calibration/OOD row fills a separate uncertainty diagnostic gap rather than
a Gate1 support test. On CIFAR-10 ID and CIFAR-100 OOD, dense reaches 0.8866
accuracy, NLL 0.3536, ECE 0.0353, and MSP AUROC 0.8230. IMP reaches 0.8953
accuracy, NLL 0.3387, ECE 0.0393, and MSP AUROC 0.8306. A 10-sample SWAG
predictive ensemble improves ECE to 0.0285, but lowers accuracy to 0.8688,
worsens NLL to 0.4018, and lowers MSP AUROC to 0.8050. Posterior predictive
uncertainty therefore does not rescue the posterior-support interpretation.
The selected full-data Gem-Miner-style row now covers five seeds. With 5
score-training epochs, 20 score batches per epoch, and 30-epoch retraining from
initialization, the mask reaches 0.8471 accuracy at 0.8319 sparsity, versus
0.8846 dense and 0.8970 IMP. Its support overlap with IMP is 0.0917
[0.0913, 0.0921], essentially random-scale for this sparsity. This is useful
negative baseline evidence for the current Gem-Miner-style implementation, but
not an exhaustive reproduction of every Gem-Miner training recipe.
The matched dense-trajectory probe is the strongest current trajectory-side
control: epoch-10 magnitude support reaches 0.2342 to IMP, the final dense
support reaches 0.2312, and the RMS-absolute aggregate trajectory score reaches
0.2400. These are far above the CIFAR posterior movement supports that peak
around 0.147. Movement-only and path-length trajectory scores are much weaker,
so the current mechanism is persistent trajectory magnitude rather than
movement alone. The fixed-mask retraining probe adds functional nuance: IMP
retraining reaches 0.8983, final dense magnitude reaches 0.8826, RMS/mean
trajectory magnitude reaches about 0.874, path/movement/epoch-1 masks reach
0.854--0.857, and random reaches 0.8422. Trajectory magnitude is useful but
does not fully reproduce IMP. The residual-swap probe shows what the missing
piece contains: half IMP-only residual swaps raise final dense masks from
0.8797 to 0.8882, RMS trajectory masks from 0.8733 to 0.8851, and epoch-10
masks from 0.8712 to 0.8855, while same-size non-IMP random residual swaps
stay at 0.8780, 0.8705, and 0.8704.
The residual-anatomy probe shows that the same bases miss about 27.8k--28.4k
IMP-kept weights, base-only weights are pruned throughout IMP with mean pruning
round 2.90--2.97, and dense-trajectory rank features only weakly predict the
IMP-only residual (AUC 0.6165--0.6206, top-k recall 0.2087--0.2206). This
coordinate signal is insufficient functionally: the residual-predictor mask
probe raises added IMP-only precision to 0.1834--0.1866 from random
0.1233--0.1253, but predictor masks do not recover the oracle residual accuracy
gain. The cross-seed transfer probe raises target-seed added precision to
0.2238--0.2413 from random 0.1246--0.1264, but still fails to recover oracle
residual accuracy. The direct cross-seed support-transfer probe gives an even
stricter negative result: source-vote additions from other seeds' oracle
residual coordinates are only mildly enriched for target IMP-only residual
weights (0.143--0.152 added precision versus 0.123--0.125 target random) and
train at base/random-like accuracy rather than target-oracle accuracy. The
base-compatibility probe adds a useful correction: exact trajectory-base
identity is not necessary once target IMP overlap and top IMP-only residual
additions are fixed. Matched random bases with the same per-parameter IMP and
non-IMP counts are weak alone (0.8605--0.8641), but top oracle residual
additions recover 0.8910--0.8942, matching or exceeding trajectory-oracle
accuracy (0.8892--0.8893), while matched random residual additions remain weak
(0.8628--0.8649). The posterior-decomposition probe fixes the
IMP-overlap-matched random base and final IMP-only membership: top oracle
IMP-only additions reach 0.8911--0.8928, diagonal-Laplace posterior-RMS-ranked
IMP-only additions reach 0.8829--0.8852, and dense-final-magnitude-ranked
IMP-only additions closely match them at 0.8812--0.8827. Uniformly random
IMP-only additions reach 0.8783--0.8795, while posterior RMS-minus-dense and
posterior-std additions fall to 0.8745--0.8770 and 0.8710--0.8717. This shows
that final IMP membership is useful but not sufficient, and that the posterior
RMS signal is largely dense magnitude rather than posterior uncertainty. The
stratified-control probe
shows that random IMP-only
additions recover part of the oracle gain, but non-IMP additions matched by
parameter tensor and score decile remain near base controls despite matching
more than 99.9% of oracle strata. This keeps the refined mechanism narrow: IMP
improves a useful dense-trajectory magnitude subspace, but the strongest
residual signal is a support-combinatorial IMP subset rather than exact
trajectory-base identity, coarse allocation, or score-bin structure. The
removal-order controls add that the oracle residual gain is not
an artifact of removing low base-score weights: top-IMP additions stay high
under low/random/high removals (final dense 0.8881/0.8883/0.8906, RMS
0.8874/0.8896/0.8914, epoch-10 0.8862/0.8922/0.8920), while non-IMP random
additions remain near 0.8779/0.8709/0.8701. The IMP-process probe adds that
this residual support is
progressively constructed across pruning rounds: round-survivor final-IMP
precision rises from about 0.43--0.44 at round 1 to about 0.75 at round 3 and
1.0 at round 5, while accuracy moves toward the final oracle residual. The
IMP-process ranking controls further separate membership from ordering:
top-score round survivors beat random and low-score round survivors in
accuracy, final-IMP precision at rounds 1/3, and oracle-overlap at round 5.
Oracle-overlap-matched and score-source controls further show that the small
round-trained ordering signal is not explained solely by final-IMP membership,
final-oracle overlap amount, dense-final magnitude, or base-source magnitude.
The round-exclusion intervention strengthens this from ranking correlation to
an ablation-style result: removing the process-selected final-IMP residual
coordinates and replacing them with the best remaining final-IMP-magnitude
coordinates loses in 44/45 paired comparisons.

Current paper figure artifacts:

- Builder: `scripts/build_paper_figures.py`
- Gate1 controls: `paper/figures/gate1_controls.pdf` and
  `paper/figures/gate1_controls.png`
- CIFAR movement: `paper/figures/cifar_movement.pdf` and
  `paper/figures/cifar_movement.png`
- CIFAR trajectory: `paper/figures/cifar_trajectory.pdf` and
  `paper/figures/cifar_trajectory.png`

Current paper statistical artifacts:

- Builder: `scripts/build_paper_stats.py`
- Mode/ticket audit builder: `scripts/run_mode_distribution_equivalence_audit.py`
- Mode/ticket audit Markdown: `docs/mode_distribution_equivalence_audit.md`
- Mode/ticket audit CSV/JSON:
  `runs/mode_distribution_equivalence_audit_summary.csv`,
  `runs/mode_distribution_equivalence_audit.json`
- Markdown audit: `docs/paper_stats.md`
- LaTeX table: `paper/tables/statistical_summary.tex`
- JSON payload: `runs/paper_stats.json`

Direct proposal-level mode/ticket distribution artifacts:

- Runner: `scripts/run_mode_ticket_distribution_probe.py`
- Summarizer: `scripts/summarize_mode_ticket_distribution_probe.py`
- Selected run:
  `runs/digits_mlp_mode_ticket_distribution_sgld_r5/20260505_221553`
- Summary:
  `docs/digits_mlp_mode_ticket_distribution_sgld_r5.md`
- CSV:
  `runs/digits_mlp_mode_ticket_distribution_sgld_r5_summary.csv`
- Full-data CIFAR run:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3/20260506_004811`
- Full-data CIFAR summary:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3.md`
- Full-data CIFAR CSV:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_r5_p0p3_summary.csv`
- Activation-aligned full-data CIFAR run:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3/20260506_005822`
- Activation-aligned full-data CIFAR summary:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3.md`
- Activation-aligned full-data CIFAR CSV:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3_summary.csv`
- Weight-correlation-aligned full-data CIFAR run:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3/20260506_091445`
- Weight-correlation-aligned full-data CIFAR summary:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md`
- Weight-correlation-aligned full-data CIFAR CSV:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv`
- Independent-start multi-chain cyclical-SGLD full-data CIFAR run:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3/20260506_125221`
- Independent-start multi-chain cyclical-SGLD full-data CIFAR summary:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md`
- Independent-start multi-chain cyclical-SGLD full-data CIFAR CSV:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv`
- Rank-128 low-rank Laplace full-data CIFAR run:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3/20260506_120015`
- Rank-128 low-rank Laplace full-data CIFAR summary:
  `docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md`
- Rank-128 low-rank Laplace full-data CIFAR CSV:
  `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv`
- Paper table:
  `tab:direct-mode-ticket-distribution`
- Result: 50 posterior sample masks versus five IMP tickets fail the
  layer-sparsity KS and Hamming-overlap thresholds on both digits and full
  CIFAR. The full CIFAR row also collapses all posterior samples to one
  raw-parameter basin versus five tickets, while logit-space and final-hidden
  activation CKA/Hungarian matching pass. Mapping ResNet masks into the first
  seed dense-model activation-channel frame preserves the failure: aligned
  posterior samples still form one basin and fail KS/Hamming thresholds.
  A second incoming/outgoing weight-correlation channel frame also preserves
  the failure with one posterior basin, layer KS `p=1.2e-08`, and Hamming
  overlap `0.1290`. The independent-start multi-chain cyclical-SGLD row
  collects 75 samples from 15 independently trained dense starts; chain starts
  and posterior samples still collapse to one basin, posterior-to-chain-start
  Hamming is `0.0439`, and samples fail layer KS `p=9.3e-10` plus Hamming
  overlap `0.0000`. The rank-128 low-rank Laplace direct row partially rescues
  pairwise mask geometry, with Hamming overlap `0.8163` and logit/activation
  CKA `0.9319`/`0.9096`, but still has one posterior basin and fails raw
  sample layer KS with `p=2.0e-06`.
