# Mode/Ticket Distribution Probe

This is a direct, small-model check of the proposal-level equivalence
criteria. Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/cifar10_subset4096_mode_ticket_distribution_activation_pilot/20260506_001418`
- Dataset/model: `cifar10` / `resnet20`
- Seeds: `[0, 1, 2]`; data seed `0`
- Posterior sampler: `sgld` with 5 samples per seed
- Posterior clusters: 3 (largest fraction 0.3333)
- Posterior basin entropy: 1.0986 nats; normalized 0.4057; effective clusters 3.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 15 | 3 | 0.0330 | 0.5756 | 0.0202 | 0.6571 | 0.9022 | 0.8726 | 0.0978 | 0.1274 | 4/6 |
| posterior_modes_vs_tickets | 3 | 3 | 0.2264 | 0.5625 | 0.0210 | 1.0000 | 0.9014 | 0.8704 | 0.0986 | 0.1296 | 6/6 |

Interpretation:

- `posterior_samples_vs_tickets` tests the raw posterior sample-induced
  mask distribution against IMP tickets.
- `posterior_modes_vs_tickets` first collapses posterior samples to
  mean-shift mode representatives, then compares those representatives
  with IMP tickets.
- Passing the logit/activation CKA and Hungarian thresholds alone is
  not enough for H1; the proposal also requires mask-distribution agreement. Low
  Hamming-overlap or low KS support therefore counts against the
  strong one-to-one mode/ticket equivalence claim.

Caveats:
- Posterior modes are mean-shift representatives in raw parameter PCA space.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- No activation-channel permutation alignment is applied here.
