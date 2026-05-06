# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_alignment_smoke/20260506_005755`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `sgld` with 2 samples per seed
- Posterior clusters: 1 (largest fraction 1.0000)
- Posterior basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 4 | 2 | 1.0000 | 0.0056 | 0.0056 | 0.6667 | 0.8857 | 0.8935 | 0.1143 | 0.1065 | 5/6 |
| posterior_modes_vs_tickets | 1 | 2 | 0.9898 | 0.2141 | 0.0418 | nan | 0.8417 | 0.8585 | 0.1583 | 0.1415 | 4/6 |
| activation_aligned_posterior_samples_vs_tickets | 4 | 2 | 1.0000 | 0.0056 | 0.0056 | 0.6667 | 0.8857 | 0.8935 | 0.1143 | 0.1065 | 5/6 |
| activation_aligned_posterior_modes_vs_tickets | 1 | 2 | 0.9898 | 0.2141 | 0.0418 | nan | 0.8417 | 0.8585 | 0.1583 | 0.1415 | 4/6 |

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
- Activation-aligned comparisons cluster and compare masks after mapping ResNet channels to the first seed dense model by activation-correlation Hungarian matching.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.
