# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/cifar10_subset_multichain_csgld_mode_ticket_smoke/20260506_014101`
- Dataset/model: `cifar10` / `resnet20`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `cyclical-sgld` with 2 samples per chain, 2 chain(s) per seed from `dense` starts
- Posterior clusters: 6 (largest fraction 0.2500)
- Posterior basin entropy: 1.7329 nats; normalized 0.8333; effective clusters 5.6569
- Chain-start clusters: 1 (largest fraction 1.0000)
- Posterior-to-chain-start Hamming mean: 0.0179; sample accuracy mean 0.0811; chain-start accuracy mean 0.0898

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 4 | 2 | 1.0000 | 0.0432 | 0.0078 | 0.6667 | 0.5983 | 0.5921 | 0.4017 | 0.4079 | 1/6 |
| posterior_samples_vs_tickets | 8 | 2 | 1.0000 | 0.0503 | 0.0090 | 0.5714 | 0.5857 | 0.6125 | 0.4143 | 0.3875 | 1/6 |
| posterior_modes_vs_tickets | 6 | 2 | 1.0000 | 0.0445 | 0.0084 | 0.6000 | 0.5857 | 0.6125 | 0.4143 | 0.3875 | 1/6 |

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
- Chain-start rows use magnitude masks from the dense state that initialized each posterior chain.
- No activation-channel permutation alignment is applied here.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.
