# Mode/Ticket Distribution Probe

This is a direct check of the proposal-level equivalence criteria.
Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits and final hidden activations.

- Run: `runs/fake_cifar10_mode_ticket_mask_artifact_smoke/20260506_225421`
- Dataset/model: `fake-cifar10` / `resnet20`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `sgld` with 1 samples per chain, 1 chain(s) per seed from `dense` starts
- Posterior clusters: 2 (largest fraction 0.5000)
- Posterior basin entropy: 0.6931 nats; normalized 1.0000; effective clusters 2.0000
- Chain-start clusters: 2 (largest fraction 0.5000)
- Posterior-to-chain-start Hamming mean: 0.0018; sample accuracy mean 0.1094; chain-start accuracy mean 0.1094
- Activation-aligned posterior clusters: 2 (largest fraction 0.5000)
- Activation-aligned basin entropy: 0.6931 nats; normalized 1.0000; effective clusters 2.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_start_magnitude_vs_tickets | 2 | 2 | 1.0000 | 0.0053 | 0.0039 | 1.0000 | 0.7212 | 0.7349 | 0.2788 | 0.2651 | 4/6 |
| posterior_samples_vs_tickets | 2 | 2 | 1.0000 | 0.0064 | 0.0054 | 1.0000 | 0.8854 | 0.8932 | 0.1146 | 0.1068 | 6/6 |
| posterior_modes_vs_tickets | 2 | 2 | 1.0000 | 0.0064 | 0.0054 | 1.0000 | 0.8854 | 0.8932 | 0.1146 | 0.1068 | 6/6 |
| activation_aligned_chain_start_magnitude_vs_tickets | 2 | 2 | 1.0000 | 0.0053 | 0.0039 | 0.0000 | 0.7212 | 0.7349 | 0.2788 | 0.2651 | 3/6 |
| activation_aligned_posterior_samples_vs_tickets | 2 | 2 | 1.0000 | 0.0064 | 0.0054 | 1.0000 | 0.8854 | 0.8932 | 0.1146 | 0.1068 | 6/6 |
| activation_aligned_posterior_modes_vs_tickets | 2 | 2 | 1.0000 | 0.0064 | 0.0054 | 1.0000 | 0.8854 | 0.8932 | 0.1146 | 0.1068 | 6/6 |

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
- Activation-aligned comparisons cluster and compare masks after mapping ResNet channels to the first seed dense model by activation-correlation Hungarian matching.
- Activation comparison uses final hidden-feature linear CKA on the held-out test set.
- Basin entropy is computed over the parameter-PCA mean-shift clusters used by each row.

## Mask Artifact Check

This smoke was run with `--save-mask-artifacts --save-state-artifacts`.
It writes `mask_artifacts.npz` in the run directory with schema version 1,
22 parameter names, 4,350 flattened parameters, `parameter_shapes_json`
for ResNet graph/channel reconstruction, raw and activation-aligned mask
matrices encoded as `uint8`, and optional flattened trainable state matrices
encoded as `float32`.

Required keys include `masks__posterior_sample`,
`masks__activation_aligned_posterior_sample`, `states__posterior_sample`,
and `states__activation_aligned_posterior_sample`. This artifact is a
path-validation fixture for future exhaustive graph/permutation reruns; it is
not a claim-level CIFAR result. The companion post-hoc matching audit in
`docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md` verifies that
the saved fixture can drive record-level and local channel-permutation
mask/state comparisons.
