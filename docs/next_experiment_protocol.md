# Next Experiment Protocol

## Gate 1: MNIST Posterior Support Beyond Magnitude

Purpose:

Test whether posterior-induced sparse supports explain IMP masks beyond dense
magnitude, initialization magnitude, and chain-start magnitude controls.

## Required Settings

Dataset:

- MNIST first, Fashion-MNIST second.

Model:

- MLP, hidden dim 128, depth 3.

Seeds:

- Minimum: 5 seeds.
- Submission-grade: 10 seeds.

Sparsity:

- Use IMP rounds/prune fraction combinations that cover approximately 50%, 70%,
  90%, and 95% sparsity.

Posterior sampling:

- Use corrected SGLD with `--sgld-likelihood-scale dataset`.
- Start with `--sgld-lr 1e-8` on MNIST and tune only if sample accuracy collapses
  or samples are indistinguishable from chain starts.
- Use at least 3 independent dense chains per seed.

Primary command template:

```bash
.venv/bin/python scripts/run_digits_pilot.py \
  --dataset mnist \
  --epochs 5 \
  --imp-rounds 6 \
  --prune-fraction 0.30 \
  --hidden-dim 128 \
  --batch-size 512 \
  --lr 0.05 \
  --sgld-chains 3 \
  --sgld-chain-init independent-dense \
  --sgld-likelihood-scale dataset \
  --sgld-lr 1e-8 \
  --sgld-steps 600 \
  --sgld-burn-in 200 \
  --sgld-sample-every 20 \
  --samples 20 \
  --random-trials 200 \
  --barrier-samples 20 \
  --barrier-points 11 \
  --out-dir runs/mnist_gate1
```

## Pass Criteria

The posterior explanation only survives Gate 1 if all hold:

1. `posterior_jaccard_mean` exceeds `random_jaccard_mean` with repeated-seed
   confidence intervals.
2. `posterior_jaccard_mean` is not merely equal to
   `chain_start_magnitude_to_imp_jaccard_mean`.
3. `posterior_to_chain_start_magnitude_jaccard_mean` is meaningfully below 1,
   showing SGLD moved the support.
4. Dense/initial magnitude controls do not dominate the posterior signal across
   all sparsities.
5. Function-space and linear-connectivity checks do not collapse the result into
   a single connected low-loss basin explanation.

The generated linear connectivity barrier audit now shows that this condition
does not rescue the posterior explanation: MNIST/Fashion dense-to-IMP paths can
be nearly connected and CIFAR paths can have large barriers, but posterior
support remains tied to chain-start controls in both cases.

## Failure Interpretation

If Gate 1 fails, the paper should pivot:

> Winning tickets are better explained as trajectory/magnitude subspace objects
> than as Bayesian posterior-mode supports.

The current mode/ticket distribution-equivalence audit makes this pivot
stronger: using existing posterior artifacts, posterior supports beat random in
58/59 grouped support-overlap comparisons, but beat matched chain-start support
by more than 0.005 Jaccard in 0/59 comparisons; rewind magnitude beats posterior
by more than 0.005 Jaccard in 55/57 comparisons. Future posterior experiments
therefore need to target genuinely higher-fidelity full-network posterior
coverage or activation-space mode tests, not another local support-overlap
variant.

The explicit variational-pruning code path is now present and tested on small
models, CIFAR support overlap, and CIFAR-scale calibration/OOD. A stronger
hard-concrete L0 gate path is also implemented and smoke-tested on fake-CIFAR
plus a real CIFAR subset for both support and calibration/OOD probes. The
digits 5-seed row is useful H3 coverage but not decisive: variational pruning
beats random and Gem-Miner-style masks in accuracy, yet stays below IMP and
does not improve ECE/Brier over IMP. The full-data CIFAR support rows are
negative: variational pruning reaches 0.8306 accuracy, 0.0907 IMP Jaccard, and
sits 0.0669 below IMP, while hard-concrete reaches 0.2766 accuracy, 0.0922 IMP
Jaccard, and sits 0.6204 below IMP. The five-seed full-data
CIFAR-10/CIFAR-100 learned-mask row also closes the OOD diagnostic gap:
learned-random, Gem-Miner-style, and variational-prune masks lower ECE, but all
lose accuracy, NLL, Brier, and OOD AUROC relative to IMP. There is no obvious
H3 rescue from learned mask selection. The next high-value work should return
to full-network posterior fidelity or mode/activation-distribution tests.

## Current MNIST Full-Sweep Result

The MNIST full sweep in `docs/mnist_gate1_full_sweep.md` failed Gate 1 across
five seeds and four sparsities. The failure was not due to random controls:
posterior-induced masks beat random masks at every sparsity. Failure came from
stronger controls:

- posterior masks did not exceed chain-start magnitude masks;
- posterior-to-chain-start support overlap stayed between 0.9115 and 0.9423;
- dense magnitude masks were much closer to IMP masks than posterior masks.

Before scaling to CIFAR-10, the only positive-claim rescue paths worth testing
are stronger posterior movement and alternative posterior-to-mask maps.

The first movement sweep tested higher SGLD temperature and step size. It moved
supports away from chain starts but did not improve posterior-to-IMP alignment.
The remaining positive rescue path is therefore alternative posterior-to-mask
maps, not merely more SGLD noise.

A posterior map smoke test has now tried mean, RMS, SNR, high variance, and low
variance maps. None approached dense magnitude alignment. Unless a stronger
posterior sampler changes this pattern, the next paper-building step is the
negative-result version.

First SWAG baselines on MNIST and Fashion-MNIST r5 p0.30 across five seeds also
failed Gate1. SWAG masks beat random but match chain-start magnitude
(`posterior - chain-start = -0.0002` on MNIST and `-0.0005` on Fashion-MNIST),
remain close to chain starts (`0.9581` and `0.9480` support overlap), and are
dominated by dense magnitude (`dense - posterior = 0.3541` and `0.3138`). This
reduces the SGLD-artifact concern, but it is not a full posterior-baseline sweep
yet.

A tuned 5-seed full-batch HMC baseline on a small digits MLP now provides a
higher-fidelity posterior stress test. It reaches 0.9201 mean HMC sample
accuracy with 0.95 acceptance and moves supports away from the dense chain start
(`post-chain = 0.4095`), but still fails Gate1: HMC posterior overlap is 0.4136
against random 0.3254 and dense/chain-start magnitude 0.8736. This is stronger
small-model evidence against the SGLD-artifact explanation.

A direct proposal-level digits MLP probe now tests the literal
KS/MMD/Wasserstein/Hamming/CKA/Hungarian criteria. With five IMP tickets and 50
SGLD posterior sample masks, raw posterior sample masks fail the layer-sparsity
KS threshold (`p=0.0788`) and Hamming-overlap threshold (`0.6314 < 0.70`) while
passing logit-CKA/Hungarian thresholds. Mean-shift parameter-space clustering
finds one posterior mode representative versus five tickets.

The full-data CIFAR-10 ResNet-20 direct row now gives the same answer with
activation-space metrics: 50 SGLD posterior sample masks fail layer KS
(`p=5.3e-09`) and Hamming overlap (`0.0033 < 0.70`), mean-shift collapses to
one basin versus five IMP tickets, and the single representative still fails
layer KS (`p=0.0329`). Logit and final-hidden activation CKA pass, so the
remaining positive-claim rescue path is not another unaligned direct
distribution metric. A first full-data activation-channel alignment row now
maps posterior and ticket masks into the first seed dense-model channel frame;
it still has one posterior basin and fails layer KS (`p=2.3e-09`) plus Hamming
overlap (`0.0000 < 0.70`) while logit/activation CKA pass. A second full-data
weight-correlation alignment row, using incoming/outgoing ResNet weight
features, also has one posterior basin and fails layer KS (`p=1.2e-08`) plus
Hamming overlap (`0.1290 < 0.70`) while logit/activation CKA pass. At that
point, the remaining positive-claim rescue path was a stronger full-network
CIFAR posterior baseline, an exhaustive graph/permutation pipeline, or an
explicit decision to abandon the strong H1 claim.

The stronger full-network direct baseline is now also negative. The streamed
270,896-parameter joint-group Laplace direct row compares 25 posterior samples
against five IMP tickets using exact covariance groups over all ResNet-20
weight tensors. It preserves sample accuracy (`0.8835`) and moves from
chain-start masks (`posterior-to-chain-start` Hamming `0.0503`), but all
samples still collapse to one parameter-PCA basin and fail layer KS
(`p=1.1e-08`) plus Hamming overlap (`0.0000 < 0.70`) while logit/activation
CKA remain high (`0.9373`/`0.9199`). The remaining H1 rescue path is therefore
narrower: an exhaustive graph/permutation pipeline or a posterior family
qualitatively different from the local Gaussian/SGLD/SWAG/HMC rows tested here.
The exact dense full-network covariance software path is now checked in two
small settings: a 310-parameter digits MLP and a fake-CIFAR ResNet-20 width-1
smoke with a dense `1229 x 1229` Cholesky over convolutional/residual/BatchNorm
parameters. These rows are sanity checks rather than real CIFAR evidence.
The alignment artifact audit now fixes the current boundary: seven full-data
direct rows reject equivalence, activation/weight aligned rows both fail
layer-KS/Hamming, and post-hoc exhaustive graph/permutation realignment is not
supported by the current direct-run artifacts because raw masks/states were not
saved. The direct probe now has `--save-mask-artifacts` and
`--save-state-artifacts`, with a fake-CIFAR `.npz` schema/shape smoke and a
record-level plus local channel post-hoc matching audit over that saved
fixture. The activation-aligned full-data saved-artifact rerun is now complete
and verifies record-level post-hoc matching over saved CIFAR masks/states. A
structured global channel objective over that saved artifact keeps
posterior/ticket Hamming near `0.21`, and an exact stage-1 enumeration audit
validates all `128` assignments on a tiny saved-artifact subgraph while sizing
the full CIFAR channel search at about `10^840.4` assignments per pair. A true
exhaustive permutation gate would therefore need a qualitatively different
graph solver over the saved full-data artifact, not another storage-budget
rerun.

A selected 5-seed CIFAR-10 ResNet-20 short SGLD movement diagnostic now reaches
the same qualitative conclusion at image scale. Raising SGLD LR from `1e-10` to
`1e-6` lowers posterior-to-chain-start overlap from `0.9963` to `0.6932` while
keeping sample accuracy usable (`0.8169`), but posterior-to-IMP overlap falls
from `0.1730` to `0.1683` and remains below chain-start magnitude. At `3e-6`,
movement is stronger (`post-chain = 0.5440`) but sample accuracy drops to
`0.7954` and IMP alignment drops further to `0.1602`.

Fashion-MNIST has now also failed the same 5-seed, 4-sparsity Gate1 sweep. The
next required work is either extending stronger posterior baselines across
sparsities and datasets or running higher-fidelity CIFAR-10 posterior checks.

## CIFAR-10 Follow-Up

The CIFAR-10 download blocker is resolved locally: `data/cifar-10-python.tar.gz`
is present and passed checksum validation. A one-epoch ResNet-20 subset smoke
works, but it is not evidence because dense and IMP accuracy are near chance.

The full-data training path is now viable. A 10-epoch ResNet-20 width-16
baseline with CIFAR crop/flip augmentation, LR 0.1, cosine schedule, and weight
decay 5e-4 reached 0.8302 test accuracy. A short r2/r5/r8 Gate1 grid now
exists. It failed Gate1 in every row because posterior overlap matched
chain-start magnitude and posterior-to-chain-start overlap stayed near 0.997.
The representative r5 SGLD row is now 5 seeds. A matching 5-seed r5 p0.30 SWAG
short control also exists. It failed Gate1 with posterior overlap 0.1302,
chain-start magnitude 0.1304, and posterior-to-chain-start overlap 0.9097. A
5-seed r5 p0.30 SGLD multi-chain short control with 3 independent dense starts
per seed also fails Gate1: posterior 0.1291, chain-start magnitude 0.1291,
posterior-to-chain-start 0.9963, state clusters 3.0, and function clusters 3.0.
A selected 5-seed SGLD movement diagnostic confirms that increasing SGLD LR
moves supports away from the dense chain start but not toward IMP: posterior
0.1730 at `1e-10`, 0.1683 at `1e-6`, and 0.1602 at `3e-6`, while post-chain
falls from 0.9963 to 0.6932 and 0.5440.
A selected 5-seed SGHMC movement diagnostic gives the same answer with
momentum dynamics: posterior 0.1702 at `1e-10`, 0.1682 at `3e-8`, and 0.1637
at `1e-7`, while post-chain falls from 0.9848 to 0.7733 and 0.6329. SGHMC
records 6.0 state clusters, but the support does not become more IMP-like.
A one-seed 30-epoch r5 pilot reached 0.8836 dense accuracy but only 0.8584 IMP
accuracy, so the longer-training pruning schedule needs tuning before scaling.
It still failed Gate1 with posterior 0.1381, chain-start magnitude 0.1381, and
posterior-to-chain-start 0.9972. A one-seed 30-epoch r8 p0.20 gradual-pruning
pilot matched the r5-level sparsity at 0.8322 but still produced only 0.8581
IMP accuracy against 0.8846 dense accuracy and failed Gate1 with posterior
0.1284, chain-start magnitude 0.1286, and posterior-to-chain-start 0.9968. A
5-seed 30-epoch r5 p0.30 epoch-1 rewind pilot fixes the long-budget IMP
underperformance: dense accuracy is 0.8859 and IMP accuracy is 0.8980. It
still fails Gate1 with posterior 0.1342, chain-start magnitude 0.1342,
posterior-to-chain-start 0.9969, dense magnitude 0.1472, and epoch-1 rewind
magnitude 0.1783. A matched 5-seed long-budget SWAG control also fails Gate1:
posterior 0.1361, chain-start magnitude 0.1361, posterior-to-chain-start
0.9265, dense magnitude 0.1463, and epoch-1 rewind magnitude 0.1786. SWAG moves
the support farther from the chain start than SGLD, but not toward IMP. A
5-seed long-budget SGLD multi-chain control also fails Gate1 with posterior
0.1368, chain-start magnitude 0.1368, posterior-to-chain-start 0.9969, state
clusters 3.0, function clusters 3.2, dense magnitude 0.1460, and epoch-1 rewind
magnitude 0.1800. Thus, even separated long-budget posterior chains do not
yield supports beyond local chain-start magnitude. A matched 5-seed long-budget
SGLD movement diagnostic confirms that this is not just insufficient movement:
posterior-to-chain-start falls from 0.9969 at `1e-10` to 0.7362 at `1e-6` and
0.5928 at `3e-6`, but posterior-to-IMP falls from 0.1441 to 0.1425 and 0.1381.
The `1e-6` setting preserves usable sample accuracy at 0.8753; `3e-6` degrades
to 0.8593. A matched 5-seed long-budget SGHMC movement diagnostic gives the
same answer with momentum dynamics in the epoch-1 rewind setting.
Posterior-to-chain-start falls from 0.9876 at `1e-10` to 0.8060 at `3e-8`,
0.6796 at `1e-7`, and 0.5214 at `3e-7`, while posterior-to-IMP falls from
0.1456 to 0.1445, 0.1419, and 0.1360. The `1e-7` setting keeps usable sample
accuracy at 0.8752 and records 6.0 state clusters plus 3.2 function clusters,
but it still does not beat chain-start magnitude 0.1457 or epoch-1 rewind
magnitude 0.1777.
A 5-seed long-budget cyclical SGLD movement diagnostic adds a stronger
exploration baseline with 400 posterior steps, 50-step cycles, and samples from
the second half of each cycle. It also fails the rescue test. Posterior-to-chain
falls from 0.9963 at `1e-10` to 0.7046 at `1e-6`, 0.5533 at `3e-6`, and
0.3700 at `1e-5`, while posterior-to-IMP falls from 0.1454 to 0.1422, 0.1371,
and 0.1260. The `1e-6` setting keeps usable sample accuracy at 0.8782, but it
does not beat chain-start magnitude 0.1454 or epoch-1 rewind magnitude 0.1789.
A 5-seed 20-snapshot full-network SWAG movement diagnostic adds a
low-rank-plus-diagonal Gaussian check over all trainable parameters. It also
fails the rescue test. Posterior-to-chain falls to 0.9528 at scale `16` and
0.9086 at scale `64`, but posterior-to-IMP stays at 0.1454 and 0.1453 versus
chain-start magnitude 0.1455 and epoch-1 rewind magnitude 0.1782.
A 5-seed long-budget diagonal Laplace movement diagnostic gives the same answer
with a mini-batch diagonal empirical-Fisher local Gaussian approximation.
Posterior-to-chain falls from 0.9999 at scale `1e-10` to 0.8826 at `1e-3`,
0.7803 at `3e-3`, and 0.5961 at `1e-2`, while posterior-to-IMP falls from
0.1469 to 0.1447, 0.1400, and 0.1278. The `1e-3` and `3e-3` settings keep
usable sample accuracy at 0.8799 and 0.8707, but neither beats chain-start
magnitude 0.1469 or epoch-1 rewind magnitude 0.1787. This is not an exact or
KFAC Laplace baseline, but it reduces the concern that the negative movement
result is specific to SGLD-family samplers.
A 5-seed long-budget KFAC-style Laplace movement diagnostic now gives the same
answer with a structured Kronecker-factored empirical-Fisher approximation.
Posterior-to-chain falls from 0.9999 at scale `1e-10` to 0.9334 at `1e-4`,
0.8016 at `1e-3`, and 0.4859 at `1e-2`, while posterior-to-IMP moves from
0.1456 to 0.1456, 0.1441, and 0.1303. The `1e-3` setting keeps usable sample
accuracy at 0.8839, but it does not beat chain-start magnitude 0.1456 or
epoch-1 rewind magnitude 0.1775. This reduces the diagonal-only Laplace
concern, but it is still not exact Hessian or full-covariance posterior
sampling.
A 5-seed exact full-covariance Laplace probe on the final CIFAR classifier head
also fails at head-support level. The 650-parameter head Hessian is exact under
the frozen feature extractor. Head posterior-to-chain falls from 0.9917 at
scale `1e-6` to 0.7773 at `1e-3`, 0.6912 at `1e-2`, and 0.6769 at `1`, while
head posterior-to-IMP falls from 0.7067 to 0.6983, 0.6784, and 0.6716. The
head chain-start magnitude support is 0.7068 and the head rewind magnitude
support is 0.7191. This is not a full-network posterior, but it closes the
limited final-head full-covariance objection.
Full-covariance block Laplace probes extend this beyond the final head. The
`layer1.0.conv1.weight` selected row has 2304 parameters; at scale `1e-3`,
sample accuracy is 0.8961 and block post-chain is 0.2236, but
posterior-to-IMP is 0.1959 versus block chain-start 0.2034 and block rewind
0.2423. A seed-0 seven-tensor block scan found only one mildly positive
candidate; the selected 5-seed `layer3.0.shortcut.0.weight` row has block
post-chain 0.3626, but posterior-to-IMP is 0.2402 versus chain-start 0.2411 and
rewind 0.3050. These are stronger than diagonal or KFAC covariance for selected
tensors. A joint four-tensor row adds cross-block covariance over 5424 selected
parameters: sample accuracy is 0.8922 and group post-chain is 0.5088, but
group posterior-to-IMP is 0.3294 versus group chain-start 0.3501 and group
rewind 0.3637. A wider independent tensor-block-diagonal row estimates exact
full-covariance blocks for 11 tensors and 22,064 parameters, then samples all
11 tensors simultaneously. It keeps sample accuracy at 0.8810 and has global
post-chain 0.8287, but selected-block posterior-chain is -0.0114, global
posterior-chain is only +0.0036, and global rewind remains closer by +0.0292.
The max-10k extension estimates exact blocks for 16 tensors and 68,144
parameters. It keeps sample accuracy at 0.8802 and moves farther from
chain-start support (global post-chain 0.7400), but block posterior-chain is
-0.0050, global posterior-chain is only +0.0010, and global rewind remains
closer by +0.0319.
A max-10k joint-group extension keeps the same 16 tensors and 68,144
parameters but packs them into 8 exact covariance groups, adding cross-tensor
covariance inside each group. It keeps sample accuracy at 0.8811 and moves
farther from chain-start support (global post-chain 0.7148), but block
posterior-chain is -0.0050, global posterior-chain is only +0.0015, and global
rewind remains closer by +0.0311.
A max-20k joint-group extension adds `layer3.0.conv1.weight`, covering 17
tensors and 86,576 parameters in 6 exact covariance groups. It keeps sample
accuracy at 0.8828 and moves from chain-start support (global post-chain
0.7863), but block posterior-chain is -0.0023, global posterior-chain is only
+0.0006, and global rewind remains closer by +0.0317.
A streamed max-40k joint-group extension covers all 22 ResNet-20 weight tensors
and 270,896 weight parameters in 8 exact covariance groups. It keeps sample
accuracy at 0.8824 and moves from chain-start support (global post-chain
0.7389), but block/global posterior-chain is -0.0019 and global rewind remains
closer by +0.0362. These remain structured exact covariance checks rather than
dense all-parameter full-network full-covariance posterior evidence.
A direct version of the same full-weight joint-group posterior now feeds the
samples into the literal mode/ticket distribution probe. It gives the same
negative answer at proposal level: 25 samples, one basin, layer KS `p=1.1e-08`,
Hamming overlap `0.0000`, logit CKA `0.9373`, activation CKA `0.9199`, and
sample accuracy `0.8835`.
5-seed full-network low-dimensional subspace HMC probes add tractable
CIFAR-scale HMC checks around the dense checkpoint. With an 8-dimensional
random orthonormal subspace, direction scale 10, HMC step size `3e-3`, and
frozen batchnorm statistics, accept rate is 0.7400, sample accuracy is 0.8863,
and mean parameter distance is 0.3672. The support result still fails Gate1:
posterior-to-IMP is 0.1440 versus chain-start magnitude 0.1440. A
trajectory-informed variant uses the 6-dimensional subspace spanned by dense
trajectory checkpoint directions. It starts from much stronger dense trajectory
magnitude support, but still fails the rescue test: at HMC step size `1e-3`,
accept rate is 0.6900, sample accuracy is 0.8847, post-chain is 0.9915, and
posterior-to-IMP is 0.2290 versus chain-start magnitude 0.2292. A top-Hessian
subspace variant uses a 4-dimensional randomized Hessian eigenspace around the
dense checkpoint. It accepts at 0.8600 and keeps sample accuracy at 0.8865, but
post-chain remains 0.9999 and posterior-to-IMP is 0.14713 versus chain-start
0.14713. This reduces the Gaussian-approximation, random-subspace, and
trajectory-subspace concerns but remains a low-dimensional subspace check, not
exact full-network full-covariance posterior evidence. A 5-seed
16-dimensional top-Hessian selected row keeps the same answer: at step `3e-4`,
accept rate is 0.8833, sample accuracy is 0.8881, parameter distance is
0.00949, post-chain is 0.9994, and posterior-to-IMP is 0.14680 versus
chain-start 0.14682. A five-seed 32-dimensional top-Hessian selected row widens
this curvature subspace and still does not move support toward IMP: at step
`3e-4`, accept rate is 0.9500, sample accuracy is 0.8872, parameter distance is
0.0104, post-chain is 0.9993, and posterior-to-IMP is 0.14614 versus
chain-start 0.14611.
A 5-seed CIFAR-10 calibration plus CIFAR-100 OOD probe now tests whether a
posterior predictive uncertainty view rescues the claim. It does not. Dense
has accuracy 0.8866, NLL 0.3536, ECE 0.0353, and maximum-softmax OOD AUROC
0.8230. IMP improves accuracy and NLL to 0.8953 and 0.3387, with OOD AUROC
0.8306 but slightly worse ECE 0.0393. A ten-sample SWAG ensemble improves ECE
to 0.0285, but lowers accuracy to 0.8688, worsens NLL to 0.4018, and lowers
OOD AUROC to 0.8050. The uncertainty behavior is therefore separable from
ticket-support alignment rather than a rescue of posterior-mode support.
A 5-seed matched dense-trajectory support probe now strengthens the
trajectory-control side. Taking the epoch-1 checkpoint from the same dense
trajectory used for the control, IMP reaches 0.8963 and dense reaches 0.8841.
Trajectory magnitude support overlap with IMP rises from 0.1782 at epoch 1 to
0.2197 at epoch 5, peaks at 0.2342 at epoch 10, and stays at 0.2312 at epoch
30. Aggregating the trajectory by RMS absolute magnitude reaches 0.2400, while
movement-only and path-length score masks remain much weaker. These values
dominate the global posterior movement overlaps, which peak around 0.147, and
refine the positive explanation toward persistent trajectory magnitude rather
than movement alone.

A 5-seed trajectory mask retraining probe separates support overlap from
trainability. Training fixed masks for 30 epochs from the same epoch-1 rewind
state gives 0.8983 accuracy for the IMP mask, 0.8826 for the final dense
magnitude mask, about 0.874 for RMS/mean trajectory-magnitude masks, 0.8730
for the epoch-10 mask, 0.854--0.857 for path/movement/epoch-1 masks, and
0.8422 for a random mask. Thus trajectory-magnitude masks are useful and much
better than random, but they do not reproduce IMP. The next trajectory-side
question is no longer whether dense trajectory magnitude carries signal; it is
what IMP adds beyond that subspace.

A 5-seed residual-swap probe now gives the first answer. Replacing half of the
base-only support with IMP-only support raises final dense masks from 0.8797 to
0.8882, RMS trajectory masks from 0.8733 to 0.8851, and epoch-10 masks from
0.8712 to 0.8855. Same-size non-IMP random residual swaps do not recover the
gap: 0.8780, 0.8705, and 0.8704 respectively. Thus the next trajectory-side
question is the structure of the IMP residual support, not whether the residual
is functionally meaningful.

A 5-seed residual-anatomy probe now gives the first structural readout. The
final dense, RMS trajectory, and epoch-10 bases each miss about 27.8k--28.4k
IMP-kept weights. Base-only weights are pruned throughout IMP, with mean pruning
round 2.90--2.97. The residual is only mildly stage-structured: RMS trajectory
has stage-2 enrichment 1.1348x, while stage 3 holds 74.4% of IMP-only residual
but is near its size share. Dense-trajectory rank features plus stage indicators
predict the residual only weakly, with held-out AUC 0.6165--0.6206 and top-k
recall 0.2087--0.2206. Thus the next trajectory-side question is now causal
residual generation, not basic residual characterization.

A 5-seed functional residual-predictor mask probe now tests the immediate
generation question. The held-out predictor raises added IMP-only precision
from random-control levels of 0.1233--0.1253 to 0.1834--0.1866, but this does
not translate into the oracle residual accuracy gain. Predictor and random
residual masks are essentially tied for RMS trajectory (0.8744 vs. 0.8744), and
the predictor underperforms random for final dense (0.8793 vs. 0.8805), while
oracle residual reaches 0.8866--0.8892. Thus coordinate-level predictability is
not enough.

A 5-seed cross-seed residual-transfer probe now shows that even transferable
coordinate signal is insufficient. Training the residual predictor on four
source seeds raises target-seed added IMP-only precision from random
0.1246--0.1264 to 0.2238--0.2413. But generated masks still do not recover the
oracle residual accuracy: final dense reaches 0.8776 with cross-seed residual,
0.8781 with random residual, and 0.8905 with oracle residual; RMS trajectory
reaches 0.8731, 0.8725, and 0.8878; epoch 10 reaches 0.8745, 0.8726, and
0.8890. The next trajectory-side question is therefore a stronger causal model
of the residual support, not whether a simple rank-feature predictor transfers.

A 5-seed activation-aligned direct cross-seed residual-support transfer probe
now tests a stricter seed-invariance alternative. Instead of training a
predictor, it adds target non-base weights using votes over the other seeds'
oracle residual coordinates; an activation-aligned variant first maps source
ResNet channels to target channels using held-out activation correlations.
Source-vote additions are slightly enriched for target IMP-only residual
weights: added precision is 0.1513 for final dense, 0.1451 for RMS trajectory,
and 0.1491 for epoch 10, versus target-random precision 0.1251, 0.1231, and
0.1265. But accuracy remains base/random-like and far below target oracle.
Final dense source-vote/aligned-source-vote/aligned-random/target-random/oracle
accuracies are 0.8769/0.8797/0.8809/0.8779/0.8886; RMS trajectory gives
0.8739/0.8727/0.8743/0.8728/0.8872; epoch 10 gives
0.8725/0.8710/0.8740/0.8719/0.8890. Thus the residual support is not a
directly transferable seed-invariant coordinate set, and the failure is not
rescued by a simple activation-channel permutation correction.

A 5-seed residual base-compatibility probe now tests whether the target oracle
residual only works with the exact trajectory base. It replaces each trajectory
base with a per-parameter random base preserving the same IMP/non-IMP counts
and therefore the same base-to-IMP overlap. These matched bases are weak alone:
final dense, RMS trajectory, and epoch-10 matched-base accuracies are 0.8641,
0.8605, and 0.8607, versus trajectory bases at 0.8827, 0.8743, and 0.8744.
But top oracle IMP-only residual additions recover 0.8926, 0.8910, and 0.8942,
matching or exceeding trajectory-oracle accuracies 0.8893, 0.8892, and 0.8892.
Matched random residual additions remain weak at 0.8649, 0.8628, and 0.8636.
Thus exact trajectory-base identity is not necessary once target IMP overlap
and the top IMP-only residual subset are fixed.

A 5-seed residual posterior-decomposition probe now separates final IMP
membership, dense magnitude ordering, and posterior-uncertainty ordering under
the same IMP-overlap-matched random bases. Top oracle IMP-only residual
additions reach 0.8915, 0.8928, and 0.8911 for final dense, RMS trajectory, and
epoch 10. Uniformly random IMP-only additions reach 0.8783, 0.8795, and 0.8791
with about 0.50 oracle-overlap precision. Dense-final-magnitude-ranked
IMP-only additions reach 0.8821, 0.8812, and 0.8827 with 0.553--0.557
oracle-overlap precision, closely matching diagonal-Laplace posterior-RMS
ranked additions at 0.8852, 0.8834, and 0.8829 with 0.551--0.556 overlap.
By contrast, posterior RMS-minus-dense reaches only 0.8745--0.8770 with
0.478--0.479 overlap, and posterior standard deviation reaches 0.8710--0.8717
with 0.446--0.450 overlap. Thus final IMP membership carries signal, but the
posterior-RMS signal is largely explainable by dense final magnitude rather
than posterior uncertainty.

A 5-seed residual stratified-control probe now tests a stronger causal
alternative: maybe the oracle residual gain is just layer/tensor allocation or
score-bin structure. All generated controls remove the same low-base-score
weights as the oracle. Random IMP-only additions recover part of the gap but
remain below oracle: final dense 0.8818 vs. 0.8872, RMS trajectory 0.8794 vs.
0.8858, and epoch 10 0.8764 vs. 0.8854. Non-IMP additions matched to oracle
parameter tensor and within-parameter score decile match more than 99.9% of
oracle strata but stay near the weak base controls: 0.8758, 0.8716, and
0.8685. Thus IMP membership carries functional signal, but layer/tensor/score
structure alone does not explain the oracle residual gain.

A 5-seed residual removal-order control now tests whether the oracle residual
gain is only caused by removing low-score base-only weights. Holding the
top-IMP additions fixed, low/random/high removal accuracies are
0.8881/0.8883/0.8906 for final dense, 0.8874/0.8896/0.8914 for RMS trajectory,
and 0.8862/0.8922/0.8920 for epoch 10. Same-size non-IMP random additions
under low removal remain much weaker: 0.8779, 0.8709, and 0.8701. Thus the
residual gain is driven by the added IMP-only weights, not by the removal
order artifact.

A 5-seed residual IMP-process probe now tests when the residual ranking is
constructed. Round-survivor additions from IMP process rounds 1, 3, and 5
increasingly concentrate final IMP residual support: final-IMP precision rises
from about 0.43--0.44 at round 1 to about 0.75 at round 3 and 1.0 at round 5.
Retrained accuracy moves in the same direction. For final dense, base/oracle/
round-1/round-3/round-5 accuracies are 0.8793/0.8898/0.8792/0.8829/0.8867; for
RMS trajectory they are 0.8736/0.8881/0.8771/0.8832/0.8857; for epoch 10 they
are 0.8724/0.8884/0.8772/0.8826/0.8861. The round-5 process masks still trail
the final oracle residual and overlap only about 0.67 of the oracle added
subset, so the next question is constrained process/subspace causality rather
than another simple coordinate generator.

A 5-seed residual IMP-process ranking-control probe now separates survivor-set
membership from score ordering inside each IMP round. Top-score round survivors
beat random and low-score round survivors across bases. For the RMS trajectory
base, top/random/low accuracies are 0.8788/0.8733/0.8738 at round 1,
0.8830/0.8759/0.8726 at round 3, and 0.8857/0.8803/0.8759 at round 5. At
rounds 1 and 3, top-score additions have much higher final-IMP precision than
random or low-score additions; at round 5, where all variants are final-IMP
residual by construction, top-score additions still have much higher oracle
overlap, about 0.68 versus 0.50 and 0.32. Thus final IMP membership carries
signal, but round-trained score ordering adds functional structure.

The final-IMP oracle-overlap-matched process control is now a selected 5-seed
row. It samples random final-IMP residual additions with the same
oracle-overlap count as the round-score-selected final-IMP additions. The
invariant holds in fake-CIFAR, real-CIFAR-subset, seed-0 full-data, and the
5-seed full-data row. Across 45 paired base/round/seed comparisons,
round-score final-IMP residuals beat the matched-random control in 35 cases,
with mean accuracy delta `+0.0020`. Eight of nine base/round group means are
positive, but the effect is small and mixed at the seed level. This supports a
process-specific ranking effect beyond final-IMP membership and final-oracle
overlap, while leaving constrained process/subspace causality as the next
mechanistic gate.

A score-source process control now fixes the final-IMP residual candidate set
and support budget, then ranks additions by dense-final or base-source
magnitude instead of round-trained IMP weights. Across 45 paired comparisons,
round-trained scores beat dense-score controls in 37/45 cases with mean
accuracy delta `+0.0026`, and beat base-score controls in 39/45 cases with
mean delta `+0.0028`. Round-5 rows are consistently positive, while
`traj_rms_abs` round 1 is the weak exception. This further separates
process-specific ordering from dense/base magnitude, but still leaves stronger
intervention-style process/subspace causality as the next mechanistic gate.

A round-exclusion process intervention now directly tests that gate. It fixes
the final-IMP residual candidate set and support budget, removes the
round-score-selected final-IMP additions, and then chooses the best remaining
final-IMP residual additions by final IMP magnitude. Across 45 paired
comparisons, round-selected masks beat this round-excluded oracle replacement
in 44 cases, with mean accuracy delta `+0.0061`. All nine base/round group
means are positive, with one seed-level reversal at `epoch_30` round 1. This is
stronger evidence that the process-selected subset is functionally important,
not merely correlated with final-IMP membership, final-oracle overlap, or
dense/base magnitude.

A tensor-matched round-exclusion control now closes a simpler composition
confound for the strongest RMS trajectory round-5 row. After removing the
round-selected final-IMP residual additions, the replacement preserves the
removed additions' parameter-tensor counts and then uses final IMP magnitude.
Round-selected masks reach `0.8855` accuracy versus `0.8764` for the
tensor-matched replacement, with paired delta `+0.0091` and `5/5` positive
seed deltas. The process signal is therefore not explained by tensor-level
allocation inside the final-IMP residual pool.

A tensor+score-matched round-exclusion control further narrows this confound.
It preserves both parameter-tensor counts and within-tensor round-score deciles
before using final IMP magnitude. The replacement rises to `0.8837`, but the
round-selected row remains higher at `0.8878`, with paired delta `+0.0041` and
`5/5` positive seed deltas. This suggests round-score strata explain part of
the process advantage, but not all of it.

The MNIST/Fashion-MNIST sweep table is now stable, and CIFAR has a short-grid
replication. Use subset flags only for debugging or resume checks.

Recommended next CIFAR command shape for a stronger row:

```bash
.venv/bin/python scripts/run_digits_pilot.py --dataset cifar10 --model resnet20 --epochs 10 --imp-rounds 2 --prune-fraction 0.30 --batch-size 512 --lr 0.1 --lr-schedule cosine --weight-decay 5e-4 --augment --sgld-chains 1 --sgld-chain-init dense --sgld-likelihood-scale dataset --sgld-lr 1e-10 --sgld-steps 200 --sgld-burn-in 50 --sgld-sample-every 10 --samples 10 --random-trials 100 --barrier-samples 5 --barrier-points 5
```

For submission-grade CIFAR evidence, increase the budget beyond the current
short grid: at least 5 seeds, longer training, independent dense starts with
multiple posterior chains, more posterior samples, and a full-network exact,
full-covariance, or otherwise higher-fidelity posterior comparison on at least
one CIFAR sparsity. The current evidence now covers full-weight exact
joint-group covariance in both movement and direct proposal-metric forms, so
the remaining covariance gap is literal dense all-parameter full covariance or
a qualitatively different posterior family.
The next long-budget CIFAR row should continue from epoch-1 rewinding rather
than initialization rewinding and add a stronger posterior check beyond the
current local Gaussian, selected/joint/block-diagonal/full-weight joint-group
full-covariance, and
low-dimensional subspace approximations. Simply changing from 5 rounds at p0.30
to 8 rounds at p0.20 was not enough. The calibration/OOD probe is now a
supporting uncertainty diagnostic rather than the main remaining
posterior-evidence gap, and its learned-mask path now has full five-seed CIFAR
negative evidence.
For the negative paper, the next trajectory-side checks should be causal
process controls rather than coordinate generators: projection of posterior
samples onto IMP-overlap/residual-defined subspaces, constrained retraining
inside those subspaces, or ablations that remove/permute process-specific IMP
decisions within matched support budgets. The current evidence already shows that
IMP-only residual support is functionally special relative to random residual
support, is not functionally reconstructed from dense trajectory features,
transfers only as a marginal coordinate signal, does not require exact
trajectory-base identity once IMP overlap and top residual identity are fixed,
is not explained by layer/tensor/score-bin structure, and is progressively
constructed across IMP rounds with a meaningful round-trained score ordering.
A residualized round-score projection control now removes the linear subspace
spanned by the base-source, dense-final, and final-IMP magnitude scores inside
the final-IMP residual candidate pool. At the RMS trajectory round-5 setting,
round-score masks reach 0.8852 accuracy versus 0.8811 for the residualized
score, with paired delta +0.0041 and 5/5 positive seeds; oracle overlap drops
from 0.6684 to 0.4854. This suggests the useful process ordering is not a
standalone orthogonal residual score, but an interaction with the
trajectory/final-magnitude subspace.
A posterior-residualized projection control now extends that check by also
projecting out diagonal-Laplace posterior RMS, posterior standard deviation,
and posterior RMS-minus-dense scores. In the same RMS trajectory round-5
setting, round-score masks reach 0.8847 accuracy versus 0.8825 for the
posterior-residualized score, with paired delta +0.0023 and 5/5 positive
seeds; oracle overlap drops from 0.6773 to 0.4850.
A learned-subspace residualized projection control now covers the next
trajectory-side objection: it learns rank-8 PCA component scores from dense
trajectory, final-IMP magnitude, and earlier IMP-round scores inside the
candidate pool. Round-score masks reach 0.8869 accuracy versus 0.8821 for the
learned-subspace residualized score, with paired delta +0.0048 and 5/5
positive seeds; oracle overlap drops from 0.6807 to 0.4917. The remaining
trajectory-side hardening path is therefore broader process-intervention
design, not another simple posterior-score or low-rank learned-subspace
projection.
