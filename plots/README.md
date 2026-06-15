# plots — legacy exploratory figures

Early, exploratory plots from the first-pass mechanism study on small models
(Qwen-1.5B and Qwen-7B), produced by the `e1/` and `e2/` harness. They are kept for
provenance only and are **not** used in the paper.

| file | what it showed |
|------|----------------|
| `P1_*_minor_attn.png`, `P1_*_semantic_attn.png` | early attention-pattern probes (minor vs. semantic tokens) |
| `P2P3_*_semantic.png` | follow-up semantic-attention plots |
| `P4_*.png` | fourth exploratory probe |
| `frontier_qwen7b.png` | early cost/correctness frontier on Qwen-7B |
| `horizon_qwen7b.png` | early long-trajectory check on Qwen-7B |

The figures that appear in the paper are built by `paper/figs/make_*.py` (vector PDFs in
`paper/figs/`); standalone PNG renders live in [`../figures/`](../figures/).
