# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_weight_alignment_smoke/20260506_091346`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `sgld` with 1 samples per chain, 1 chain(s) per seed from `dense` starts
- Posterior clusters: 2 (largest fraction 0.5000)
- Posterior basin entropy: 0.6931 nats; normalized 1.0000; effective clusters 2.0000
- Chain-start clusters: 2 (largest fraction 0.5000)
- Posterior-to-chain-start Hamming mean: 0.0018; sample accuracy mean 0.1094; chain-start accuracy mean 0.1094
- Weight-aligned posterior clusters: 2 (largest fraction 0.5000)
- Weight-aligned basin entropy: 0.6931 nats; normalized 1.0000; effective clusters 2.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 2 | 2 | 1.0000 | 0.0053 | 0.0039 | 1.0000 | 0.7217 | 0.7354 | 0.2783 | 0.2646 | 4/6 |
| posterior_samples_vs_tickets | 2 | 2 | 1.0000 | 0.0049 | 0.0045 | 1.0000 | 0.8846 | 0.8925 | 0.1154 | 0.1075 | 6/6 |
| posterior_modes_vs_tickets | 2 | 2 | 1.0000 | 0.0049 | 0.0045 | 1.0000 | 0.8846 | 0.8925 | 0.1154 | 0.1075 | 6/6 |
| weight_aligned_chain_start_magnitude_vs_tickets | 2 | 2 | 1.0000 | 0.0053 | 0.0039 | 1.0000 | 0.7217 | 0.7354 | 0.2783 | 0.2646 | 4/6 |
| weight_aligned_posterior_samples_vs_tickets | 2 | 2 | 1.0000 | 0.0049 | 0.0045 | 1.0000 | 0.8846 | 0.8925 | 0.1154 | 0.1075 | 6/6 |
| weight_aligned_posterior_modes_vs_tickets | 2 | 2 | 1.0000 | 0.0049 | 0.0045 | 1.0000 | 0.8846 | 0.8925 | 0.1154 | 0.1075 | 6/6 |

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
- Weight-aligned comparisons cluster and compare masks after mapping ResNet channels to the first seed dense model by incoming/outgoing weight-correlation Hungarian matching.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.
