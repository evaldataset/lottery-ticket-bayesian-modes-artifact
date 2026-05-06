# Research Roadmap

Source proposal: `proposal_A3_lottery_ticket_bayesian_modes.md`.

## Thesis State

The original proposal's strongest claim was:

> Lottery tickets are in one-to-one correspondence with Bayesian posterior modes.

The current evidence does not support that claim. A better submission-grade
thesis is now:

> Winning tickets are sparse supports constructed by the pruning trajectory and
> compatible with local posterior/functional structure, but they are not
> posterior modes under the tested posterior approximations.

This is now a stronger negative-result paper: full-data activation-channel and
weight-correlation alignment checks, full-data dense-start and
independent-start multi-chain cyclical-SGLD direct probes, a rank-128 low-rank
Laplace direct probe, and a 270,896-parameter streamed joint-group Laplace
direct probe preserve the negative result. The linear connectivity audit now
separates loss-barrier geometry from support equivalence: near-zero
MNIST/Fashion dense-to-IMP barriers and large CIFAR barriers both leave
posterior support tied to chain-start controls. A generated reviewer objection
matrix now maps the current evidence to nine likely reviewer risks, keeping the
remaining exact dense-CIFAR posterior, exhaustive graph-isomorphism, and
packaging gaps explicit. A paper submission-shape audit now marks the condensed
current draft shape-ready under the local main-text gate while preserving
reviewer-objection coverage. A separate main-only submission PDF audit now
checks `paper/main_submission.pdf`, which excludes appendix/generated evidence
tables from the venue-facing build.

## Current Evidence

The project already has five main evidence blocks:

1. Gate-style support tests on MNIST, Fashion-MNIST, and CIFAR-10 show that
   posterior-induced masks beat random masks but do not beat matched chain-start
   or rewind magnitude controls.
2. Posterior approximations including SGLD, SGHMC, cyclical SGLD, canonical
   SWAG, 20-snapshot full-network SWAG movement, diagonal/KFAC-style Laplace,
   rank-16/rank-32/rank-64/rank-128 low-rank Hessian-plus-diagonal Laplace,
   a tiny exact dense full-network Laplace sanity row over all 310 digits MLP
   parameters, a fake-CIFAR ResNet exact dense smoke over all 1,229 width-1
   trainable parameters,
   exact head/block plus 22k- and 68k-parameter tensor-block-diagonal Laplace,
   68k-, 86k-, and streamed 270k-parameter joint-group Laplace rows with
   within-group cross-tensor covariance over all weight tensors, and
   low-dimensional subspace HMC repeatedly fail to move support toward IMP
   tickets.
3. Direct proposal-metric probes now cover digits MLP, CIFAR subset pilots, and
   five-seed full-data CIFAR-10 ResNet-20. The unaligned full-data row fails
   layer KS (`p=5.3e-09`) and Hamming overlap (`0.0033 < 0.70`) while passing
   logit CKA, activation CKA, and Hungarian-cost thresholds; posterior samples
   collapse to one basin with entropy `0.0` versus five IMP tickets. The
   activation-channel-aligned full-data rerun preserves the failure: aligned
   sample masks have layer KS `p=2.3e-09`, Hamming overlap `0.0000`, logit CKA
   `0.9373`, activation CKA `0.9168`, and one aligned posterior basin. The
   weight-correlation-aligned full-data rerun also preserves the failure:
   sample masks have layer KS `p=1.2e-08`, Hamming overlap `0.1290`,
   logit CKA `0.9336`, activation CKA `0.9131`, and one posterior basin.
   A stronger multi-chain cyclical-SGLD full-data rerun collects 75 posterior
   samples from three dense-start chains per seed; the samples move from
   chain-start support (`posterior-to-chain-start` Hamming mean `0.0443`) and
   keep sample accuracy `0.8760`, but still collapse to one basin and fail
   layer KS (`p=3.3e-08`) plus Hamming overlap (`0.2461 < 0.70`).
   An independent-start multi-chain rerun collects 75 posterior samples from
   15 independently trained dense starts; both chain starts and posterior
   samples still collapse to one basin, posterior-to-chain-start Hamming is
   `0.0439`, and samples fail layer KS (`p=9.3e-10`) plus Hamming overlap
   (`0.0000 < 0.70`) while preserving high logit/activation CKA.
   The rank-128 low-rank Laplace direct row partially rescues Hamming overlap
   (`0.8163`) but still fails layer KS (`p=2.0e-06`) and basin-count
   equivalence. The streamed 270,896-parameter joint-group Laplace direct row
   is sharper: 25 samples preserve accuracy (`0.8835`) and move from
   chain-start support (`0.0503` Hamming), but they still form one basin and
   fail layer KS (`p=1.1e-08`) plus Hamming overlap (`0.0000 < 0.70`) while
   preserving high logit/activation CKA (`0.9373`/`0.9199`).
4. Linear connectivity barriers are now audited across six existing five-seed
   Gate1/CIFAR rows. MNIST and Fashion-MNIST dense-to-IMP linear barriers are
   nearly zero (`0.0026` and `0.0395`), whereas CIFAR-10 ResNet-20 long
   SGLD/SWAG dense-to-IMP barriers are large (`3.0827` and `3.7402`). In both
   regimes, posterior support remains tied to or below the chain-start
   magnitude control, so barrier evidence is an orthogonal landscape diagnostic
   rather than support-equivalence evidence.
5. Process and residual controls show that the remaining IMP advantage is
   gradually constructed by the IMP process rather than explained by posterior
   RMS, final dense magnitude, direct coordinate transfer, layer strata,
   low-score-removal artifacts, simple learned-mask baselines, or
   tensor-matched replacement of the process-selected final-IMP residual
   coordinates. In the strongest RMS trajectory round-5 process row,
   round-selected masks reach `0.8855` accuracy versus `0.8764` for the
   tensor-matched excluded replacement, with `5/5` positive seed deltas. A
   stricter tensor+score-matched replacement raises the replacement accuracy to
   `0.8837` and final-oracle overlap to `0.6440`, but the round-selected row
   still reaches `0.8878` with mean paired delta `+0.0041` and `5/5` positive
   seed deltas. A residualized round-score projection control removes the
   linear base/dense/final-IMP magnitude subspace inside the final-IMP residual
   candidate pool; round-selected masks reach `0.8852` versus `0.8811` for the
   residualized score, again with paired delta `+0.0041` and `5/5` positive
   seed deltas, while oracle overlap drops from `0.6684` to `0.4854`. A
   posterior-projection variant additionally removes diagonal-Laplace
   posterior RMS, posterior standard deviation, and posterior RMS-minus-dense
   features from the round score; the round row still reaches `0.8847` versus
   `0.8825` for the posterior-residualized row with `5/5` positive accuracy
   deltas, and oracle overlap drops from `0.6773` to `0.4850`. A learned
   rank-8 trajectory/process subspace control also fails to replace the exact
   process-selected coordinates: round-selected masks reach `0.8869` versus
   `0.8821` for the learned-subspace residualized row, with paired delta
   `+0.0048`, `5/5` positive seeds, and oracle overlap dropping from `0.6807`
   to `0.4917`.

## Remaining Submission Gaps

The plan should focus on closing objections, not expanding the study broadly.

1. Stronger CIFAR posterior baseline.
   Dense-start and independent-start multi-chain cyclical-SGLD direct probes
   close the most immediate sampler-mixing and chain-start diversity
   objections. The 20-snapshot full-network SWAG row and the
   rank-16/rank-32/rank-64/rank-128 Hessian-plus-diagonal Laplace rows now add full-vector
   low-rank-plus-diagonal Gaussian checks over all trainable parameters, and
   the rank-128 low-rank Laplace plus streamed 270,896-parameter joint-group
   Laplace direct probes test stronger posteriors with the proposal-level
   KS/Hamming/CKA metrics. A 22,064-parameter exact
   tensor-block-diagonal Laplace row now samples 11 tensors simultaneously:
   it preserves `0.8810` sample accuracy and moves from chain-start support
   (`global post-chain=0.8287`), but its selected-block posterior-chain delta
   is `-0.0114` and the global gain is only `+0.0036`, with rewind support
   still closer by `0.0292`. A wider 16-tensor row covers 68,144 parameters,
   preserves `0.8802` sample accuracy, and moves farther from chain-start
   support (`global post-chain=0.7400`), but block posterior-chain remains
   negative (`-0.0050`), global posterior-chain is only `+0.0010`, and rewind
   remains closer by `0.0319`. A matching 68,144-parameter joint-group row
   packs the tensors into 8 exact covariance groups with cross-tensor
   covariance inside each group; it preserves `0.8811` sample accuracy and
   moves farther still (`global post-chain=0.7148`), but block
   posterior-chain remains negative (`-0.0050`), global posterior-chain is
   only `+0.0015`, and rewind remains closer by `0.0311`. Raising the
   joint-group budget to 20,000 parameters adds a stage-3 convolution block,
   covering 17 tensors and 86,576 parameters in 6 exact covariance groups; it
   preserves `0.8828` sample accuracy and has `global post-chain=0.7863`, but
   block posterior-chain remains negative (`-0.0023`), global
   posterior-chain is only `+0.0006`, and rewind remains closer by `0.0317`.
   The streamed max-40k joint-group row covers all 270,896 ResNet-20 weight
   parameters in 8 exact covariance groups; its movement summary preserves
   `0.8824` sample accuracy but has block/global posterior-chain `-0.0019`,
   and its direct proposal-metric row has one basin, layer KS `p=1.1e-08`, and
   Hamming overlap `0.0000`.
   A tiny exact dense full-network Laplace sanity row now validates the dense
   all-parameter covariance code path on a 310-parameter digits MLP; at scale
   `1e-3`, samples keep `0.8450` accuracy and move from chain-start support
   (`post-chain=0.8084`) but posterior support is `0.7545` versus `0.8596`
   for chain-start magnitude. A fake-CIFAR ResNet-20 width-1 smoke additionally
   validates a dense `1229 x 1229` Cholesky over convolutional/residual/BatchNorm
   parameters; it is not real CIFAR evidence. The remaining high-end posterior
   objection is narrower: exact or near-exact full-network full-covariance
   CIFAR posterior evidence is still missing.
   A feasibility audit now quantifies the exact dense option: all-trainable
   CIFAR ResNet-20 covariance requires `553.1` GiB for one float64 matrix and
   `1,106.3` GiB with Cholesky resident, so the practical next posterior
   hardening path is stronger low-rank/subspace coverage rather than literal
   dense full covariance.

2. Alignment/permutation variants.
   The activation-channel and weight-correlation aligned full-data checks are
   negative, so this is no longer the primary blocker. The artifact audit in
   `docs/mode_ticket_alignment_artifact_audit.md` now records the boundary:
   seven full-data direct rows reject equivalence, the activation and
   weight-correlation aligned rows both fail layer-KS/Hamming, and post-hoc
   exhaustive graph/permutation realignment is not supported by the current
   direct-run artifacts because raw posterior/ticket masks or states were not
   saved. The direct probe now supports `--save-mask-artifacts` and
   `--save-state-artifacts`, and a fake-CIFAR fixture validates the `.npz`
   schema, parameter shapes, record-level post-hoc minimum-cost matching, and
   local channel-permutation matching. The activation-aligned full-data
   saved-artifact rerun is now complete and verifies record-level post-hoc
   matching over saved CIFAR masks/states. A structured global channel audit
   over that saved artifact keeps posterior/ticket Hamming near `0.21`, so
   channel relabeling does not rescue support equivalence. An exact stage-1
   enumeration audit now cuts a 270-parameter fake-CIFAR ResNet subgraph from
   the saved artifact path, enumerates all `128` channel assignments, and
   verifies that the block-coordinate solver matches the exact optimum on that
   subgraph. The same audit sizes the full CIFAR channel search at about
   `10^840.4` assignments per record pair, so a reviewer-facing cleanup can
   treat exhaustive full-data graph isomorphism as an explicit infeasibility
   limitation unless a radically different graph solver is introduced.

3. Learned-mask distribution variants.
   Keep this bounded. The stronger hard-concrete L0 mask source is now
   implemented, wired into support and calibration/OOD probes, smoke-tested on
   fake-CIFAR plus a real CIFAR subset, and run in the same five-seed full-data
   CIFAR protocol. The selected row is strongly negative: 0.2766 accuracy and
   0.0922 support-to-IMP Jaccard. Do not turn this into a separate pruning
   benchmark.

4. Process causality cleanup.
   The tensor, tensor+score, residualized-score, posterior-residualized, and
   learned-subspace projection controls now cover the strongest RMS trajectory
   round-5 process row. Remaining process-causality work is optional hardening
   rather than the main blocker.

5. Paper hardening.
   Tighten the contribution claim, keep large generated tables in the appendix,
   keep only decision-relevant figures and compact summary tables in the main
   text, verify every numeric claim from generated artifacts, and prepare a
   reproducibility manifest. The generated statistics are now appendix-scoped,
   the main CIFAR movement table is compact, the paper now states the
   support-equivalence claim and scope of the claim explicitly, and the
   verifier checks the appendix/table and claim-scope structure. The
   project-critical environment lock, CI workflow, claim-to-artifact ledger,
   public release manifest, local release archive, source-only repository
   snapshot, CPU artifact-verification container, external-validation runbook,
   and submission handoff now exist; remaining hardening is public archive
   upload, clean public repository evidence, externally observed green CI, and
   external GPU/CUDA container validation.

## Decision Gates

Negative paper is ready to finalize if:

- stronger CIFAR posterior baselines either stay near chain-start/rewind
  controls or move without improving the direct mode/ticket distribution match;
- broader alignment/permutation variants do not alter the direct full-data
  mode/ticket conclusion;
- learned-mask variants remain below IMP support or require stochastic
  posterior prediction to improve calibration;
- process controls continue to localize the residual to IMP dynamics.

Pivot to a partial-equivalence paper only if a stronger posterior baseline or a
  broader alignment/permutation variant makes posterior mode masks match IMP masks
on both function-space and mask-distribution metrics across repeated seeds.

## Four-Week Execution Plan

Week 1:

- Implement activation-channel and weight-correlation alignment for the
  full-data CIFAR mode/ticket probe. Done.
- Run fake-CIFAR and real CIFAR subset smokes to confirm the metric path. Done.

Week 2:

- Run the five-seed aligned full-data CIFAR comparison. Done.
- Integrate the result into `docs/paper_stats.md`, `paper/main.tex`, and the
  audits. Done.

Week 3:

- Run one stronger posterior baseline at CIFAR scale. Prefer the option that
  covers all trainable layers while staying within the single-GPU budget. Done
  for multi-chain cyclical-SGLD, 20-snapshot full-network SWAG movement, and
  rank-16/rank-32/rank-64/rank-128 Hessian-plus-diagonal Laplace movement, plus
  a 22,064-parameter exact tensor-block-diagonal Laplace row, a
  68,144-parameter tensor-block-diagonal row, and 68,144-, 86,576-, and
  270,896-parameter joint-group movement/direct rows with within-group
  cross-tensor covariance;
  exact dense all-parameter full-covariance full-network posterior remains an
  optional submission-hardening item.

Week 4:

- Run the tensor-matched round-exclusion process-control row for the strongest
  RMS trajectory round-5 setting. Done; the round-selected residual beats the
  parameter-tensor-matched replacement by `+0.0091` accuracy with `5/5`
  positive seed deltas. A stricter tensor+score-matched replacement is also
  done; it narrows the gap but still loses by `+0.0041` accuracy with `5/5`
  positive seed deltas.

- Run one bounded learned-mask or process-causality cleanup experiment. Done
  for the tensor+score-matched process-causality cleanup, residualized,
  posterior-residualized, and learned-subspace round-score projection
  controls; broader process interventions remain optional hardening work.
- Freeze the main paper claim, rebuild figures/tables, run `make check` and
  `make paper-check`, then do a submission-readiness pass. Done for the current
  negative-result framing: the manuscript now names the support-equivalence
  claim, distinguishes it from general Bayesian pruning claims, and scopes the
  limitations around tested posterior families rather than an all-posterior
  theorem.
