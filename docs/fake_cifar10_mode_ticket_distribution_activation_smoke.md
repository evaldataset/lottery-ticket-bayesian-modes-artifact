# Mode/Ticket Distribution Probe

This is a direct, small-model check of the proposal-level equivalence
criteria. Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_distribution_activation_smoke/20260506_001323`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `sgld` with 2 samples per seed
- Posterior clusters: 1 (largest fraction 1.0000)

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 4 | 2 | 1.0000 | 0.0039 | 0.0026 | 0.6667 | 0.9110 | 0.9354 | 0.0890 | 0.0646 | 5/6 |
| posterior_modes_vs_tickets | 1 | 2 | 1.0000 | 0.2194 | 0.0181 | nan | 0.9204 | 0.9376 | 0.0796 | 0.0624 | 5/6 |

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
- No activation-channel permutation alignment or basin entropy is applied here.
