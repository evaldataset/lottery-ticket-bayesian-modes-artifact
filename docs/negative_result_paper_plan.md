# Negative-Result Paper Plan

## Working Title

Winning Tickets Are Not Posterior Modes: Evidence from Posterior Support and
Magnitude Controls

## Core Claim

The original positive claim was:

> IMP winning tickets correspond to Bayesian posterior modes or mode-induced
> sparse supports.

The emerging evidence supports a sharper negative claim:

> In small networks, MNIST/Fashion-MNIST sweeps, and current CIFAR-10 ResNet-20
> pilots, posterior-induced sparse supports can beat random masks, but this
> apparent signal is explained by chain-start and dense magnitude controls. An
> explicit KS/Wasserstein/MMD support-distribution audit now makes this failure
> direct: posterior supports beat random in 58/59 grouped comparisons, but beat
> the matched chain-start support by more than 0.005 Jaccard in 0/59 groups;
> rewind magnitude beats posterior by more than 0.005 Jaccard in 55/57 groups. IMP
> masks are better understood as trajectory/magnitude subspace objects than as
> posterior-mode supports. Fixed trajectory-magnitude masks are trainable and
> much better than random, but they still underperform IMP, so IMP appears to
> refine rather than merely copy the dense trajectory magnitude subspace. A
> residual-swap probe shows that IMP-only residual support recovers much of the
> missing accuracy, while same-size non-IMP random residual swaps do not.
> Residual-anatomy controls show that dense trajectory statistics only weakly
> predict this IMP-only residual, and functional predictor-mask controls show
> that this weak coordinate-level signal does not recover the oracle residual
> accuracy gain. Cross-seed and direct support-transfer controls show that even
> seed-transferable residual coordinate signal does not instantiate the
> functional residual. Base-compatibility controls show that exact
> trajectory-base identity is not necessary once target IMP overlap and the top
> IMP-only residual subset are fixed. Base-ordering controls show that final
> IMP membership is useful but not sufficient without the top IMP-only residual
> ordering. Stratified, removal-order, and IMP-process controls further show
> that the residual gain is not explained by coarse layer/tensor/score-bin
> structure or by preferentially removing weak base weights; IMP gradually
> constructs the useful residual support across pruning rounds, and the
> round-trained ordering within survivor/final-IMP residual sets carries a
> small but measurable functional signal beyond final-oracle overlap and
> dense/base magnitude ranking. A round-exclusion intervention further shows
> that the process-selected final-IMP residual coordinates are not easily
> replaceable by the best remaining final-IMP-magnitude coordinates, even when
> the replacement is matched by parameter tensor and within-tensor round-score
> decile in the strongest RMS trajectory round-5 row. A residualized round-score
> projection control then shows that removing the base/dense/final-IMP magnitude
> subspace drops the round-ordering gain, localizing the useful signal to an
> interaction between IMP process state and trajectory/final magnitude. A
> posterior-residualized projection control further removes diagonal-Laplace
> posterior RMS/std/excess scores; the round-selected row still has `5/5`
> positive accuracy deltas and a much higher oracle-overlap signal. A
> learned-subspace residualized projection control then removes rank-8 PCA
> directions learned from dense trajectory, final-IMP magnitude, and earlier
> IMP-round scores; the round-selected row still wins by `+0.0048` accuracy
> with `5/5` positive seeds and oracle overlap drops from `0.6807` to
> `0.4917`.

## Why This Can Be Top-Conference Worthy

The contribution is not "SGLD failed." The contribution is a controlled
separation of three explanations that are often conflated:

1. Posterior structure.
2. Dense magnitude structure.
3. Rewinding/trajectory subspace structure.

If the experiments hold at MNIST/Fashion-MNIST and CIFAR-10 scale, the paper
would directly constrain Bayesian interpretations of LTH and clarify what kind
of object a winning ticket is not.

## Paper Structure

1. Introduction
   - LTH is often interpreted as evidence for hidden structure in the training
     distribution or posterior.
   - We ask whether posterior samples/modes induce the same sparse supports as
     IMP.
   - We find that the posterior signal disappears under stronger controls.

2. Related Work
   - LTH and IMP.
   - Linear mode connectivity and model re-basin work.
   - Bayesian neural networks, SGLD/HMC/SWAG.
   - PAC-Bayesian and Bayesian lottery-ticket papers.

3. Operational Test
   - Define posterior-to-mask maps: sample magnitude, posterior mean, RMS, SNR,
     high variance, low variance.
   - Define controls: random same-sparsity, initialization magnitude, dense
     magnitude, chain-start magnitude.
   - Define function checks: prediction agreement, logits clustering, linear
     path barriers.

4. Results
   - RQ1: Do posterior masks beat random? Yes, often.
   - RQ2: Do posterior masks beat chain-start magnitude? No in current evidence.
   - RQ3: Do posterior maps beat dense magnitude? No in current evidence.
   - RQ4: Does stronger SGLD movement rescue the claim? No in current sweep:
     supports move away from chain start but not toward IMP.
   - RQ5: Are independent chains meaningfully separated? Yes, function/parameter
     clusters and high barriers can appear, but this does not align them with
     IMP. The six-row linear connectivity barrier audit now makes the same
     point across MNIST/Fashion and CIFAR: barriers vary from near-zero to
     large without making posterior support beat chain-start controls.
   - RQ6: Do trajectory-magnitude supports functionally reproduce IMP after
     retraining? No. They train well above random, but remain below IMP.
   - RQ7: Is the missing residual support functionally specific? Yes in the
     current CIFAR row: IMP-only residual swaps recover much of the gap, while
     random residual swaps do not.
   - RQ8: Can dense-trajectory residual predictors generate functional masks?
     Not in the current CIFAR row. They improve held-out IMP-only coordinate
     precision over random but do not recover oracle residual accuracy.
   - RQ9: When does the residual support appear in the IMP process? It is
     progressively constructed: round-survivor additions concentrate final IMP
     residual weights and improve accuracy as IMP rounds progress. A stricter
     oracle-overlap-matched final-IMP control and score-source controls leave
     a small positive paired score-ordering effect beyond oracle-overlap and
     dense/base magnitude. A round-exclusion intervention gives a stronger
     ablation-style result: removing the round-selected final-IMP residual
     additions and replacing them with the best remaining final-IMP-magnitude
     additions loses in 44/45 paired comparisons, though the process masks
     still trail the final oracle residual. A tensor-matched version at the
     strongest RMS trajectory round-5 row also loses: round-selected masks
     reach `0.8855` accuracy versus `0.8764` for the tensor-matched
     replacement, with `5/5` positive paired deltas. A still stricter
     tensor+score-matched rerun makes the replacement more competitive
     (`0.8837`) but still trails the round-selected row (`0.8878`) by
     `+0.0041` with `5/5` positive paired deltas. A residualized round-score
     projection control removes the base/dense/final-IMP magnitude subspace
     and lowers accuracy from `0.8852` to `0.8811`, again with `5/5` paired
     deltas for the original round score; the process signal is therefore
     coupled to trajectory/final magnitude rather than isolated in an
     orthogonal residual score. A posterior-residualized projection control
     also removes diagonal-Laplace posterior RMS/std/excess scores and yields
     `0.8825` accuracy versus `0.8847` for the original round score; the
     paired accuracy delta is smaller (`+0.0023`) but `5/5` positive, and
     oracle overlap drops from `0.6773` to `0.4850`. A learned-subspace
     residualized control removes rank-8 dense trajectory/final-IMP/earlier
     IMP-round PCA directions and still trails the original round score:
     `0.8821` versus `0.8869`, paired delta `+0.0048`, `5/5` positive seeds,
     and oracle-overlap drop `0.6807` to `0.4917`.
   - RQ10: Is the residual-swap gain just an artifact of removing weak base
     weights? No in the current CIFAR row. Top-IMP additions stay high when
     removing low, random, or high base-only weights, while non-IMP additions
     remain weak.

5. Discussion
   - The evidence supports a trajectory/magnitude subspace account.
   - Posterior support is not useless, but it is insufficient as an explanation
     of IMP.
   - The explicit support-distribution audit rejects the proposal's mask-level
     mode/ticket equivalence criterion under the tested posterior families.
   - The direct digits MLP distribution probe reaches the same conclusion using
     the proposal's literal KS/MMD/Wasserstein/Hamming/CKA/Hungarian metrics:
     mask-level thresholds fail and posterior mode count does not match ticket
     diversity, even though logit-space matching passes.
   - A small-model Bernoulli/Concrete variational-pruning baseline improves over
     random and Gem-Miner-style masks but remains below IMP and does not improve
     ECE/Brier over IMP; the matched full-data CIFAR support row is
     random-scale in IMP overlap and 6.7 accuracy points below IMP.
   - A stronger hard-concrete L0 learned-mask source now has a five-seed
     full-data CIFAR support row; it retrains to 0.2766 accuracy, 62.0 points
     below IMP, with 0.0922 Jaccard overlap to IMP.
   - The result narrows the space of Bayesian LTH interpretations.

6. Limitations
   - SGLD is approximate and may not fully mix.
   - Tuned full-batch HMC covers a small-model posterior check, and random,
     trajectory-informed, plus top-Hessian low-dimensional full-network
     subspace HMC probes now cover tractable CIFAR-scale HMC checks. SWAG and
     low-rank Hessian-plus-diagonal Laplace provide full-vector
     low-rank-plus-diagonal Gaussian checks, 22,064- plus
     68,144-parameter tensor-block-diagonal Laplace rows, and 68,144- plus
     86,576- plus 270,896-parameter streamed joint-group Laplace rows cover
     wider exact-covariance subsets and the full weight vector with
     within-group cross-tensor covariance, but exact dense all-parameter
     full-network/full-covariance posterior evidence is still missing.
   - The direct proposal-metric distribution probe now covers full-data
     CIFAR-10 ResNet-20 rows before and after activation-channel and
     weight-correlation alignment, dense-start and independent-start
     multi-chain cyclical-SGLD rows, a rank-128 low-rank Laplace direct row,
     and a 270,896-parameter streamed joint-group Laplace direct row. The
     low-rank Laplace row passes Hamming overlap but still fails layer KS and
     basin-count equivalence; the exact joint-group direct row fails layer KS
     (`p=1.1e-08`), Hamming overlap (`0.0000 < 0.70`), and basin-count
     equivalence while preserving logit/activation CKA (`0.937`/`0.920`).
     The alignment artifact audit verifies that post-hoc exhaustive
     graph/permutation realignment is not supported by the current direct-run
     artifacts because raw posterior/ticket masks or states were not saved.
     The direct probe now has `--save-mask-artifacts` and
     `--save-state-artifacts`, validated by a fake-CIFAR `.npz` schema/shape
     smoke and a record-level plus local channel post-hoc matching audit over
     that saved fixture. The activation-aligned full-data saved-artifact rerun
     is now complete and verifies record-level post-hoc matching over saved
     CIFAR masks/states. A structured global channel audit over that artifact
     keeps posterior/ticket Hamming near `0.21`, so simple channel relabeling
     does not rescue the support claim. An exact stage-1 enumeration feasibility
     audit now validates the exhaustive code path on a 270-parameter
     fake-CIFAR ResNet subgraph by checking all `128` channel assignments and
     matching the block-coordinate optimum; it also sizes the full CIFAR
     channel search at about `10^840.4` assignments per record pair. Broader
     graph-level permutation pipelines can now be treated as an explicitly
     infeasible/full-data robustness limitation rather than an unexamined
     artifact gap.
   - CIFAR-10 ResNet-20 still lacks a literal dense all-parameter
     full-covariance posterior, but full-weight exact joint-group movement and
     direct proposal-metric rows now bound the feasible single-workstation
     covariance objection.

## Required Experiments Before Submission

Minimum:

- MNIST 5 seeds at approximately 50%, 70%, 90%, 95% sparsity. Completed:
  `docs/mnist_gate1_full_sweep.md`.
- Fashion-MNIST 5 seeds at the same sparsities. Completed:
  `docs/fashion_gate1_full_sweep.md`.
- Exact or higher-fidelity posterior on a small MLP. Completed first stronger
  version: a tuned 5-seed full-batch HMC digits row reaches 0.9201 HMC sample
  accuracy, accept rate 0.95, posterior 0.4136 vs. random 0.3254, post-chain
  0.4095, dense/chain-start 0.8736, and fails Gate1.
- First SWAG comparisons exist on MNIST and Fashion-MNIST r5 p0.30 across
  5 seeds and fail Gate1; they should be extended across sparsities before
  being treated as submission-grade.
- SNIP and SynFlow controls.
- Gem-Miner-style score-training mask source is implemented and smoke-tested.
  The selected five-seed full-data CIFAR row is negative: 0.8471 retrain
  accuracy and 0.0917 support overlap to IMP versus IMP 0.8970. This is a
  useful score-training baseline, though not an exhaustive reproduction of
  every Gem-Miner training recipe.
- Hard-concrete L0 mask source is implemented and smoke-tested through support
  and calibration/OOD paths; the real CIFAR subset smoke reaches 0.3216 support
  overlap to IMP at 0.5099 sparsity. A full-data five-seed row remains open.
- Posterior map comparisons: sample magnitude, mean abs, RMS, SNR, variance.
- Gate1 evaluator table.
- Current evidence now includes MNIST and Fashion-MNIST 5-seed, 4-sparsity
  Gate1 failures.

Strong:

- CIFAR-10 ResNet-20, at least 3 seeds and 3 sparsities.
- The ResNet-20 code path is smoke-tested on fake-CIFAR and on a real CIFAR-10
  subset. Full-data CIFAR-10 training is now viable: a 10-epoch dense baseline
  reached 0.8302 test accuracy, and the short r2/r5/r8 Gate1 grid reached
  0.8239--0.8292 dense / 0.8491--0.8634 IMP accuracy and failed Gate1 in every
  row. The representative r5 SGLD row is now 5 seeds. A 5-seed r5 p0.30 CIFAR
  SWAG short control also failed Gate1 with posterior 0.1302 vs. chain-start
  magnitude 0.1304 and post-chain overlap 0.9097. A 5-seed r5 p0.30 SGLD
  multi-chain short control with 3 independent dense starts per seed also failed
  Gate1 with posterior 0.1291 vs. chain-start magnitude 0.1291 and post-chain
  overlap 0.9963, while recording 3.0 state and function clusters. A selected
  5-seed CIFAR short SGLD movement diagnostic moved support substantially from
  the dense chain start (`post-chain = 0.6932` at `1e-6` and `0.5440` at
  `3e-6`), but posterior-to-IMP overlap fell from 0.1730 to 0.1683 and 0.1602
  rather than improving. A selected 5-seed CIFAR short SGHMC movement diagnostic
  adds a momentum sampler: post-chain falls to 0.7733 at `3e-8` and 0.6329 at
  `1e-7`, but posterior-to-IMP falls from 0.1702 to 0.1682 and 0.1637. A
  one-seed
  30-epoch r5 pilot reached 0.8836 dense accuracy but only 0.8584 IMP accuracy;
  it also failed Gate1 with posterior 0.1381 vs. chain-start magnitude 0.1381
  and post-chain overlap 0.9972. A more gradual 30-epoch r8 p0.20 pilot at
  nearly the same sparsity also underperformed dense (0.8581 IMP vs. 0.8846
  dense) and failed Gate1 with posterior 0.1284 vs. chain-start magnitude
  0.1286 and post-chain overlap 0.9968. An epoch-1 rewind r5 pilot fixes the
  long-budget IMP issue across 5 seeds (0.8980 IMP vs. 0.8859 dense), but still
  fails Gate1 with posterior 0.1342 vs. chain-start magnitude 0.1342 and
  post-chain overlap 0.9969; epoch-1 rewind magnitude is closer to IMP at
  0.1783. A matched 5-seed long-budget SWAG control also fails: posterior
  0.1361 vs. chain-start magnitude 0.1361, post-chain overlap 0.9265, and
  epoch-1 rewind magnitude 0.1786. A 5-seed long-budget SGLD multi-chain
  control records 3.0 state clusters and 3.2 function clusters, but still fails
  Gate1 with posterior 0.1368 vs. chain-start magnitude 0.1368, post-chain
  overlap 0.9969, and epoch-1 rewind magnitude 0.1800. A 5-seed long-budget
  SGLD movement diagnostic also fails the rescue test: post-chain drops to
  0.7362 at `1e-6` and 0.5928 at `3e-6`, but posterior-to-IMP drops from
  0.1441 to 0.1425 and 0.1381. A matched 5-seed long-budget SGHMC movement
  diagnostic gives the same answer with momentum dynamics: post-chain drops to
  0.6796 at `1e-7` and 0.5214 at `3e-7`, but posterior-to-IMP drops from
  0.1456 to 0.1419 and 0.1360. A cyclical SGLD long-budget diagnostic also
  fails: post-chain drops to 0.7046 at `1e-6`, 0.5533 at `3e-6`, and 0.3700 at
  `1e-5`, while posterior-to-IMP drops from 0.1454 to 0.1422, 0.1371, and
  0.1260. A 20-snapshot full-network SWAG movement diagnostic gives the same
  answer with a low-rank-plus-diagonal Gaussian over all trainable parameters:
  post-chain drops to 0.9528 at scale `16` and 0.9086 at scale `64`, but
  posterior-to-IMP remains 0.1454 and 0.1453 versus chain-start 0.1455 and
  rewind magnitude 0.1782. A diagonal Laplace long-budget diagnostic gives the
  same answer with a mini-batch diagonal empirical-Fisher local Gaussian approximation:
  post-chain drops to 0.8826 at scale `1e-3`, 0.7803 at `3e-3`, and 0.5961 at
  `1e-2`, while posterior-to-IMP drops from 0.1469 to 0.1447, 0.1400, and
  0.1278. A KFAC-style Laplace long-budget diagnostic gives the same answer
  with structured empirical-Fisher factors: post-chain drops to 0.9334 at
  scale `1e-4`, 0.8016 at `1e-3`, and 0.4859 at `1e-2`, while
  posterior-to-IMP stays at or below chain-start magnitude (0.1456, 0.1441, and
  0.1303 vs. chain-start 0.1456). Full-network low-rank
  Hessian-plus-diagonal Laplace diagnostics add explicit correlated rank-16,
  rank-32, rank-64, and rank-128 curvature over all trainable parameters. At scale
  `1e-2`, post-chain drops to 0.7359, 0.7402, and 0.7397, but posterior-to-IMP
  drops from chain-start 0.1456/0.1457/0.1433 to 0.1351/0.1358/0.1339. A
  final-head exact
  full-covariance Laplace
  probe now supplies a limited higher-fidelity check: head post-chain drops to
  0.6912 at scale `1e-2` and 0.6769 at scale `1`, but head
  posterior-minus-chain-start is negative with 95% CIs below zero. A
  2304-parameter full-covariance block Laplace probe extends the check beyond
  the final head: at scale `1e-3`, block post-chain is 0.2236 and sample
  accuracy is 0.8961, but block posterior-to-IMP is 0.1959 versus block
  chain-start 0.2034 and block rewind 0.2423. A seed-0 seven-tensor block scan
  and a five-seed selected `layer3.0.shortcut.0.weight` row give the same
  answer: block post-chain is 0.3626, but block posterior-to-IMP is 0.2402
  versus block chain-start 0.2411 and block rewind 0.3050. A joint
  four-tensor full-covariance row over 5424 parameters adds cross-tensor
  covariance: group post-chain is 0.5088 and sample accuracy is 0.8922, but
  group posterior-to-IMP is 0.3294 versus group chain-start 0.3501 and group
  rewind 0.3637. A wider exact block-diagonal full-covariance row over
  11 tensors and 22,064 parameters samples the selected tensors
  simultaneously: global post-chain is 0.8287 and sample accuracy is 0.8810,
  but selected-block posterior-minus-chain is `-0.0114`, global
  posterior-minus-chain is only `+0.0036`, and global rewind-posterior remains
  `+0.0292`. The max-10k extension covers 16 tensors and 68,144 parameters:
  global post-chain falls to 0.7400 and sample accuracy is 0.8802, but block
  posterior-minus-chain remains `-0.0050`, global posterior-minus-chain is
  only `+0.0010`, and global rewind-posterior remains `+0.0319`. The pattern
  persists when the same 16 tensors are packed into 8 exact joint groups with
  within-group cross-tensor covariance: global post-chain falls to 0.7148 and
  sample accuracy is 0.8811, but block posterior-minus-chain is `-0.0050`,
  global posterior-minus-chain is only `+0.0015`, and global
  rewind-posterior remains `+0.0311`. The max-20k joint-group row adds the
  first stage-3 convolution block, covering 17 tensors and 86,576 parameters:
  global post-chain is 0.7863 and sample accuracy is 0.8828, but block
  posterior-minus-chain remains `-0.0023`, global posterior-minus-chain is
  only `+0.0006`, and global rewind-posterior remains `+0.0317`. These still
  must be paired with a full-network exact
  full-covariance posterior before making a broad image-model claim, but they
  reduce the head-only, single-tensor, and narrow selected-group covariance
  objections.
  5-seed low-dimensional full-network subspace HMC probes supply tractable
  CIFAR-scale HMC checks around the dense checkpoint. In an 8-dimensional
  random subspace, HMC accept rate is 0.7400, sample accuracy is 0.8863, mean
  parameter distance is 0.3672, and posterior-to-IMP is 0.1440 versus
  chain-start magnitude 0.1440. In the 6-dimensional dense-trajectory
  checkpoint subspace, the chain-start magnitude support is much stronger
  (0.2292), but HMC still does not improve it: at step `1e-3`, accept rate is
  0.6900, sample accuracy is 0.8847, post-chain is 0.9915, and posterior-to-IMP
  is 0.2290 versus chain-start 0.2292. This reduces the
  Gaussian-approximation and random-subspace concerns. A 4-dimensional
  top-Hessian subspace HMC probe adds a curvature-informed version: accept rate
  is 0.8600, sample accuracy is 0.8865, post-chain is 0.9999, and
  posterior-to-IMP is 0.1471 versus chain-start 0.1471. A 16-dimensional
  top-Hessian variant keeps the same conclusion across five seeds: accept rate
  is 0.8833, sample accuracy is 0.8881, parameter distance is 0.00949,
  post-chain is 0.9994, and posterior-to-IMP is 0.14680 versus chain-start
  0.14682. A five-seed 32-dimensional top-Hessian selected row also stays at the
  chain-start support: at step `3e-4`, accept rate is 0.9500, sample accuracy
  is 0.8872, parameter distance is 0.0104, post-chain is 0.9993, and
  posterior-to-IMP is 0.14614 versus chain-start 0.14611. These remain
  low-dimensional HMC checks rather
  than exact full-covariance full-network posterior evidence.
  A 5-seed CIFAR-10/CIFAR-100 calibration/OOD probe adds an uncertainty
  diagnostic. Dense has accuracy 0.8866, NLL 0.3536, ECE 0.0353, and MSP OOD
  AUROC 0.8230. IMP improves accuracy and NLL to 0.8953 and 0.3387, while a
  10-sample SWAG predictive ensemble improves ECE to 0.0285 but worsens
  accuracy, NLL, and OOD AUROC (0.8688, 0.4018, 0.8050). The matched
  learned-mask rows show the same tradeoff: learned-random, Gem-Miner-style,
  and variational-prune masks lower ECE to 0.0270, 0.0283, and 0.0255, but
  reach only 0.8449, 0.8418, and 0.8301 accuracy and lower MSP OOD AUROC
  (0.7897, 0.7853, 0.7754). This supports the narrative only as a separation
  result: posterior uncertainty and learned mask calibration are not the same
  object as ticket-support alignment.
  A 5-seed trajectory mask retraining probe adds the functional comparator:
  the IMP mask retrains to 0.8983, final dense magnitude to 0.8826, RMS/mean
  trajectory magnitude to about 0.874, path/movement/epoch-1 masks to
  0.854--0.857, and random to 0.8422. This strengthens the trajectory account
  over posterior/random baselines, but it also shows that the dense trajectory
  magnitude subspace is not by itself the full IMP mechanism.
  A 5-seed residual-swap probe tests the missing component directly. Swapping
  half of base-only support for IMP-only support raises final dense masks from
  0.8797 to 0.8882, RMS trajectory masks from 0.8733 to 0.8851, and epoch-10
  masks from 0.8712 to 0.8855. Same-size non-IMP random residual swaps stay at
  0.8780, 0.8705, and 0.8704. This is stronger causal evidence that the IMP
  residual is specific rather than a generic support-distance effect.
  A 5-seed residual-anatomy probe characterizes the same residual. The final
  dense, RMS trajectory, and epoch-10 bases each miss about 27.8k--28.4k
  IMP-kept weights; base-only weights are pruned throughout IMP with mean
  pruning round 2.90--2.97. Dense-trajectory rank features plus stage
  indicators weakly predict the residual (AUC 0.6165--0.6206, top-k recall
  0.2087--0.2206), so dense trajectory statistics do not reconstruct the
  process-specific IMP residual.
  Functional residual-predictor and cross-seed transfer probes sharpen the same
  point. Coordinate precision improves above random, including across seeds
  (cross-seed added precision 0.2238--0.2413 versus random 0.1246--0.1264),
  but generated masks do not recover oracle residual accuracy.
  A direct cross-seed residual-support transfer probe gives a stricter negative
  check: source-vote additions from other seeds' oracle residual coordinates
  are only mildly enriched for target IMP-only residual weights (0.143--0.152
  versus 0.123--0.125 target random) and train at base/random-like accuracy
  rather than target-oracle accuracy.
  A residual base-compatibility probe refines the subspace interpretation:
  matched random bases preserving the same per-parameter IMP/non-IMP counts are
  weak alone (0.8605--0.8641), but top oracle IMP-only residual additions
  recover 0.8910--0.8942, matching or exceeding trajectory-oracle accuracies
  0.8892--0.8893; matched random residual additions remain weak
  (0.8628--0.8649).
  A residual posterior-decomposition probe fixes the matched random base and
  compares top, dense-ranked, posterior-ranked, posterior-uncertainty-ranked,
  and random IMP-only additions: top oracle IMP-only additions reach
  0.8911--0.8928, diagonal-Laplace posterior-RMS-ranked IMP-only additions
  reach 0.8829--0.8852 with 0.551--0.556 oracle-overlap precision, and dense
  final magnitude nearly matches them at 0.8812--0.8827 with 0.553--0.557
  overlap. Random IMP-only additions reach 0.8783--0.8795 with about 0.50
  overlap, while posterior RMS-minus-dense and posterior standard deviation
  fall to 0.8745--0.8770 and 0.8710--0.8717 with 0.478--0.479 and
  0.446--0.450 overlap. Thus final IMP membership helps, but the apparent
  posterior-RMS residual-ordering signal is mostly dense magnitude, not
  posterior uncertainty.
  A residual stratified-control probe closes the coarse-structure alternative:
  random IMP-only additions recover part of the oracle gain, but non-IMP
  additions matched by parameter tensor and score decile stay near base
  controls despite matching more than 99.9% of oracle strata.
  A residual removal-order control closes the low-score-removal alternative:
  holding the top-IMP additions fixed, low/random/high removal accuracies are
  0.8881/0.8883/0.8906 for final dense, 0.8874/0.8896/0.8914 for RMS
  trajectory, and 0.8862/0.8922/0.8920 for epoch 10, while non-IMP random
  additions remain at 0.8779/0.8709/0.8701.
  A residual IMP-process probe closes the first process-timing alternative:
  round-survivor final-IMP precision rises from about 0.43--0.44 at round 1 to
  about 0.75 at round 3 and 1.0 at round 5, and retrained accuracy moves toward
  the final oracle residual as the process advances.
  A residual IMP-process ranking-control probe separates survivor-set
  membership from score ordering: top round-survivor additions beat random and
  low-score survivor additions in accuracy, final-IMP precision at rounds 1/3,
  and oracle overlap at round 5.
- Broader HMC, CIFAR full-network full-covariance Laplace, or another
  higher-fidelity posterior comparison beyond the current
  selected/joint/block-diagonal/joint-group Laplace, exact dense
  full-network Laplace sanity row on a 310-parameter digits MLP, fake-CIFAR
  ResNet exact dense code-path smoke, and random, trajectory, and top-Hessian
  CIFAR subspace HMC checks.
- Linear mode connectivity is now covered by
  `docs/linear_connectivity_barrier_audit.md`: MNIST/Fashion dense-to-IMP
  barriers are near zero (`0.0026`/`0.0395`), CIFAR long SGLD/SWAG barriers
  are large (`3.0827`/`3.7402`), and posterior support remains tied to
  chain-start controls in both regimes.
- Calibration/OOD can be included as a short supporting diagnostic, not as a
  central table, because it separates SWAG uncertainty behavior from support
  alignment.

## New Trajectory-Control Result

A matched dense-trajectory probe now supplies direct evidence for the
trajectory/magnitude-subspace account. The probe uses one dense CIFAR ResNet-20
trajectory per seed and takes its epoch-1 checkpoint as the IMP rewind state.
Across five seeds, IMP accuracy is 0.8963 and dense accuracy is 0.8841.
Magnitude support overlap with the final IMP mask rises from 0.1782 at epoch 1
to 0.2197 at epoch 5, peaks at 0.2342 at epoch 10, and remains 0.2312 at the
final dense checkpoint. Aggregating the same trajectory by RMS absolute
magnitude reaches 0.2400; movement-only and path-length scores remain much
weaker. These controls dominate the global posterior movement overlaps, which
peak around 0.147. This should become the main positive mechanism in the
negative paper: posterior supports are not random, but the training
trajectory's persistent magnitude subspace carries substantially more
ticket-specific support information. A functional retraining probe adds the
important qualifier: trajectory-magnitude masks train well above random but
below IMP, with the final dense support the strongest non-IMP fixed mask. The
paper should therefore claim that IMP refines a trajectory-magnitude subspace,
not that trajectory magnitude alone exactly explains winning tickets. The
residual-swap probe strengthens this refinement claim: IMP-only residual
support recovers much of the remaining gap and non-IMP random residual support
does not. The residual-anatomy probe adds that the missing IMP-only support is
only mildly stage-structured and only weakly predictable from dense trajectory
rank features.
The functional residual-predictor and cross-seed transfer probes add that
residual coordinates are marginally predictable, and even seed-transferable,
but that coordinate signal is not enough to generate the functional IMP
residual support. The activation-aligned direct cross-seed support-transfer
probe adds that even other seeds' oracle residual coordinates do not directly
transfer the functional residual: source-vote additions improve target IMP-only
precision slightly above random but remain base/random-like in accuracy and far
below the target oracle residual, and activation-channel permutation alignment
does not rescue them. The residual base-compatibility probe adds that exact
trajectory-base identity is not required once target IMP overlap and the top
IMP-only residual subset are fixed: matched random bases are weak alone but
recover trajectory-oracle accuracy after top oracle residual additions. The
residual posterior-decomposition probe adds that final IMP membership is not
sufficient: top IMP-only additions beat dense-ranked, posterior-RMS-ranked,
posterior-uncertainty-ranked, and random additions even with the same matched
base and candidate pool. Posterior-RMS-ranked additions are better than random
but are closely matched by dense final magnitude, while posterior uncertainty
rankings are weak. The
residual stratified-control probe adds that the oracle gain is not explained by
coarse layer/tensor allocation or
within-parameter score-bin structure. The residual removal-order control adds
that the oracle
gain is not a low-score-removal artifact: keeping the top-IMP additions fixed
and removing low, random, or high base-only weights preserves the gain, while
non-IMP random additions remain weak. The residual IMP-process probe adds that
the useful
residual is progressively constructed across pruning rounds: round-survivor
final-IMP precision rises from about 0.43--0.44 at round 1 to about 0.75 at
round 3 and 1.0 at round 5, and accuracy moves toward the final oracle
residual. The ranking-control variant adds that top round-survivor additions
beat random and low-score survivor additions, so the IMP process contributes
ordering within its survivor sets, not only final residual membership. The
missing piece should be described as process-specific or combinatorial residual
support, not merely a dense-trajectory scalar ranking.

## Figure Plan

Figure 1:

- Schematic: posterior masks beat random but collapse to chain-start support.

Figure 2:

- MNIST/Fashion-MNIST Jaccard bar chart across controls:
  random, posterior sample, chain-start, initialization, dense magnitude.

Figure 3:

- SGLD movement sweep: posterior-to-chain-start decreases with temperature, but
  posterior-to-IMP does not increase.

Figure 4:

- Function-space clusters plus the generated linear connectivity barrier audit
  for independent and long-budget chains.

Figure 5:

- CIFAR-10 ResNet-20 replication or failure-mode comparison.

## Reviewer-Facing Risk Register

`docs/reviewer_objection_matrix.md` maps nine likely reviewer objections to
artifact-backed answers, key numbers, status, and remaining gaps. The main
paper should keep the random-control, sampler-movement, function-vs-mask,
alignment/permutation, and covariance objections visible.
`docs/paper_submission_shape_audit.md` tracks the current editing constraint:
after the condensation pass, the manuscript is shape-ready by the local
main-text gate, with `Current Results` under the target line budget while the
main reviewer-objection families remain visible.

## Decision Rule

If HMC or CIFAR-10 contradicts the current pattern, return to the positive or
partial-equivalence claim. Otherwise, write the negative paper.
