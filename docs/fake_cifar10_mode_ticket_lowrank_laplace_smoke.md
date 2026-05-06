# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_lowrank_laplace_smoke/20260506_115941`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0]`; data seed `0`
- Posterior sampler: `lowrank-laplace` with 1 samples per chain, 1 chain(s) per seed from `dense` starts
- Posterior clusters: 1 (largest fraction 1.0000)
- Posterior basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000
- Chain-start clusters: 1 (largest fraction 1.0000)
- Posterior-to-chain-start Hamming mean: 0.4041; sample accuracy mean 0.2188; chain-start accuracy mean 0.1250

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 1 | 1 | 1.0000 | 0.7869 | 0.0054 | nan | 0.5983 | 0.6661 | 0.4017 | 0.3339 | 1/6 |
| posterior_samples_vs_tickets | 1 | 1 | 0.0002 | 0.7869 | 0.0979 | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0/6 |
| posterior_modes_vs_tickets | 1 | 1 | 0.0002 | 0.7869 | 0.0979 | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0/6 |

Interpretation:

- `posterior_samples_vs_tickets` tests the raw posterior sample-induced
  mask distribution against IMP tickets.
- `posterior_modes_vs_tickets` first collapses posterior samples to
  mean-shift mode representatives, then compares those representatives
  with IMP tickets.
- `*_aligned_*` rows first map ResNet masks into the first
  seed dense-model channel frame using the configured channel-alignment
  method, then repeat the same distribution checks.
- Passing the logit/activation CKA and Hungarian thresholds alone is
  not enough for H1; the proposal also requires mask-distribution agreement. Low
  Hamming-overlap or low KS support therefore counts against the
  strong one-to-one mode/ticket equivalence claim.

Caveats:
- Posterior modes are mean-shift representatives in raw parameter PCA space.
- Chain-start rows use magnitude masks from the dense state that initialized each posterior chain.
- No channel permutation alignment is applied here.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.
