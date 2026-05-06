# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_multichain_r5_p0p3/20260506_014145`
- Dataset/model: `cifar10` / `resnet20`
- Seeds: `[0, 1, 2, 3, 4]`; data seed `0`
- Posterior sampler: `cyclical-sgld` with 5 samples per chain, 3 chain(s) per seed from `dense` starts
- Posterior clusters: 1 (largest fraction 1.0000)
- Posterior basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000
- Chain-start clusters: 1 (largest fraction 1.0000)
- Posterior-to-chain-start Hamming mean: 0.0443; sample accuracy mean 0.8760; chain-start accuracy mean 0.8841

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 15 | 5 | 0.0000 | 1.4465 | 0.0645 | 0.0857 | 0.9363 | 0.9196 | 0.0637 | 0.0804 | 4/6 |
| posterior_samples_vs_tickets | 75 | 5 | 0.0000 | 1.4263 | 0.0592 | 0.2461 | 0.9327 | 0.9144 | 0.0673 | 0.0856 | 4/6 |
| posterior_modes_vs_tickets | 1 | 5 | 0.0413 | 1.7497 | 0.0739 | nan | 0.9288 | 0.9115 | 0.0712 | 0.0885 | 4/6 |

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
