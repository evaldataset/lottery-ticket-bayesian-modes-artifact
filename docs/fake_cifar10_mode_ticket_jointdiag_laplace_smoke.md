# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_jointdiag_laplace_smoke/20260506_211101`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0]`; data seed `0`
- Posterior sampler: `jointdiag-laplace` with 2 samples per chain, 1 chain(s) per seed from `dense` starts
- Posterior clusters: 2 (largest fraction 0.5000)
- Posterior basin entropy: 0.6931 nats; normalized 1.0000; effective clusters 2.0000
- Chain-start clusters: 1 (largest fraction 1.0000)
- Posterior-to-chain-start Hamming mean: 0.0943; sample accuracy mean 0.1250; chain-start accuracy mean 0.1250

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 1 | 1 | 1.0000 | 0.7869 | 0.0070 | nan | 0.5983 | 0.6661 | 0.4017 | 0.3339 | 1/6 |
| posterior_samples_vs_tickets | 2 | 1 | 0.9997 | 0.8028 | 0.0553 | nan | 0.2044 | 0.2392 | 0.7956 | 0.7608 | 1/6 |
| posterior_modes_vs_tickets | 2 | 1 | 0.9997 | 0.8028 | 0.0553 | nan | 0.2044 | 0.2392 | 0.7956 | 0.7608 | 1/6 |

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
