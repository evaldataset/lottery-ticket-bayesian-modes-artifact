# Literature Matrix

This file tracks how the project should position itself for a top-conference
submission. It is deliberately action-oriented rather than exhaustive.

## Directly Adjacent Work

| Work | What it establishes | Consequence for this project |
| --- | --- | --- |
| Frankle & Carbin, 2019, LTH | IMP finds sparse trainable subnetworks that need the original initialization. | Baseline method and terminology. |
| Frankle et al., 2020, Linear Mode Connectivity and LTH | Rewinding and stability connect tickets to loss landscape modes. | Our posterior-mode claim must be compared against LMC, not presented in isolation. |
| Paul et al., 2023, Unmasking the LTH | IMP masks identify axial subspaces intersecting linearly connected matching modes. | Primary competing explanation: masks may encode landscape subspaces rather than Bayesian posterior modes. |
| Sakamoto & Sato, 2022, PAC-Bayesian LTH | Winning tickets can be analyzed with spike-and-slab PAC-Bayes bounds and may sit in sharper minima. | Use as theory baseline; avoid claiming Bayesian framing is absent. |
| Kuhn et al., 2026, Bayesian Lottery Ticket Hypothesis | LTH-style tickets appear in Bayesian neural networks; magnitude and std matter for pruning. | Novelty must shift to posterior basin support alignment, not "BNNs have lottery tickets." |
| Marsh et al., 2026 withdrawn, Bayes Always Wins the Lottery in Monte Carlo | HMC plus LTH masks may reduce initialization dependence. | Treat as related but not authoritative because the submission is withdrawn. |

## Positioning Gap

The open gap is not whether LTH, Bayesian pruning, or posterior sampling exist.
The gap is whether deterministic IMP masks can be predicted or explained by
sparse supports induced by posterior basins after controlling for:

- ordinary dense magnitude pruning,
- original initialization magnitude,
- same-sparsity random masks,
- function-space similarity,
- rewinding and trajectory dependence,
- linear connectivity and Hessian/flatness effects.

## Top-Conference Standard

A convincing paper needs all of the following:

1. A precise operational map from posterior sample/mode to sparse mask.
2. A strong non-Bayesian control showing the result is not just dense magnitude.
3. Repeated seeds and sparsity sweeps.
4. Function-space evidence in addition to mask overlap.
5. A clearly viable negative-result framing if the posterior explanation fails.

