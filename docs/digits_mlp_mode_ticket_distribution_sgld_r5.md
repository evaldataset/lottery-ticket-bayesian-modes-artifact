# Mode/Ticket Distribution Probe

This is a direct, small-model check of the proposal-level equivalence
criteria. Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits.

- Run: `runs/digits_mlp_mode_ticket_distribution_sgld_r5/20260505_221553`
- Dataset/model: `digits` / `mlp`
- Seeds: `[0, 1, 2, 3, 4]`; data seed `0`
- Posterior sampler: `sgld` with 10 samples per seed
- Posterior clusters: 1 (largest fraction 1.0000)
- Posterior basin entropy: 0.0000 nats; normalized 0.0000; effective clusters 1.0000

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA | Act. CKA | H-cost | Act. H-cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 50 | 5 | 0.0788 | 1.0349 | 0.0192 | 0.6314 | 0.9819 | nan | 0.0181 | nan | 2/4 |
| posterior_modes_vs_tickets | 1 | 5 | 0.9216 | 0.7144 | 0.0123 | nan | 0.9821 | nan | 0.0179 | nan | 3/4 |

Interpretation:

- `posterior_samples_vs_tickets` tests the raw posterior sample-induced
  mask distribution against IMP tickets.
- `posterior_modes_vs_tickets` first collapses posterior samples to
  mean-shift mode representatives, then compares those representatives
  with IMP tickets.
- Passing the logit CKA and Hungarian thresholds alone is
  not enough for H1; the proposal also requires mask-distribution agreement. Low
  Hamming-overlap or low KS support therefore counts against the
  strong one-to-one mode/ticket equivalence claim.

Caveats:
- Posterior modes are mean-shift representatives in raw parameter PCA space.
- Function-space comparison uses logit-space linear CKA on the held-out test set.
- No activation-channel permutation alignment is applied here.
