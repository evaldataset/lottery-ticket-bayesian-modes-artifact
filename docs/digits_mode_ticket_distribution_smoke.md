# Mode/Ticket Distribution Probe

This is a direct, small-model check of the proposal-level equivalence
criteria. Posterior samples are converted to magnitude masks at the
matched IMP sparsity; posterior modes are mean-shift representatives
in PCA-reduced parameter space. Function-space similarity is measured
as linear CKA over held-out logits.

- Run: `runs/digits_mode_ticket_distribution_smoke/20260505_221521`
- Dataset/model: `digits` / `mlp`
- Seeds: `[0, 1]`; data seed `0`
- Posterior sampler: `sgld` with 2 samples per seed
- Posterior clusters: 1 (largest fraction 1.0000)

| Comparison | n left | n tickets | KS p | MMD | SW2 | Hamming overlap | Logit CKA match | Hungarian cost | Threshold pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior_samples_vs_tickets | 4 | 2 | 0.9980 | 0.1967 | 0.0010 | 0.6667 | 0.9746 | 0.0254 | 3/4 |
| posterior_modes_vs_tickets | 1 | 2 | 1.0000 | 0.0000 | 0.0000 | nan | 0.9699 | 0.0301 | 3/4 |

Interpretation:

- `posterior_samples_vs_tickets` tests the raw posterior sample-induced
  mask distribution against IMP tickets.
- `posterior_modes_vs_tickets` first collapses posterior samples to
  mean-shift mode representatives, then compares those representatives
  with IMP tickets.
- Passing the logit-CKA and Hungarian thresholds alone is not enough for
  H1; the proposal also requires mask-distribution agreement. Low
  Hamming-overlap or low KS support therefore counts against the
  strong one-to-one mode/ticket equivalence claim.

Caveats:
- Posterior modes are mean-shift representatives in raw parameter PCA space.
- Activation comparison uses logit-space linear CKA on the held-out test set.
- No activation-channel permutation alignment or basin entropy is applied here.
