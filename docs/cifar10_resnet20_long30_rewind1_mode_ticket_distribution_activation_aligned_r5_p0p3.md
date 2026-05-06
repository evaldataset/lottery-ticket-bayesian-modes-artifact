# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_r5_p0p3/20260506_005822`
- Dataset/model: `cifar10` / `resnet20`
- Seeds: `[0, 1, 2, 3, 4]`; data seed `0`
- Posterior sampler: `sgld` with 10 samples per seed
- Posterior clusters: 1 (largest fraction 1.0000)
- Posterior basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000
- Activation-aligned posterior clusters: 1 (largest fraction 1.0000)
- Activation-aligned basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 50 | 5 | 0.0000 | 1.3337 | 0.0644 | 0.0000 | 0.9373 | 0.9168 | 0.0627 | 0.0832 | 4/6 |
| posterior_modes_vs_tickets | 1 | 5 | 0.0413 | 1.7463 | 0.0793 | nan | 0.9363 | 0.9160 | 0.0637 | 0.0840 | 4/6 |
| activation_aligned_posterior_samples_vs_tickets | 50 | 5 | 0.0000 | 1.3337 | 0.0644 | 0.0000 | 0.9373 | 0.9168 | 0.0627 | 0.0832 | 4/6 |
| activation_aligned_posterior_modes_vs_tickets | 1 | 5 | 0.0413 | 1.7463 | 0.0793 | nan | 0.9363 | 0.9160 | 0.0637 | 0.0840 | 4/6 |

Interpretation:

- `posterior_samples_vs_tickets` tests the raw posterior sample-induced
  mask distribution against IMP tickets.
- `posterior_modes_vs_tickets` first collapses posterior samples to
  mean-shift mode representatives, then compares those representatives
  with IMP tickets.
- `activation_aligned_*` rows first map ResNet masks into the first
  seed dense-model channel frame using activation-correlation Hungarian
  matching, then repeat the same distribution checks.
- Passing the logit/activation CKA and Hungarian thresholds alone is
  not enough for H1; the proposal also requires mask-distribution agreement. Low
  Hamming-overlap or low KS support therefore counts against the
  strong one-to-one mode/ticket equivalence claim.

Caveats:
- Posterior modes are mean-shift representatives in raw parameter PCA space.
- Activation-aligned comparisons cluster and compare masks after mapping ResNet channels to the first seed dense model by activation-correlation Hungarian matching.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.
