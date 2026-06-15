# figures — standalone PNG renders

Standalone PNG versions of the main results, rendered from the records in
[`../results/`](../results/). These are convenience images (e.g. for slides or quick
viewing).

The **canonical, paper-quality figures are the vector PDFs** built by
`paper/figs/make_*.py` and embedded in [`../paper/main.pdf`](../paper/main.pdf). Early
exploratory plots from the small-model runs live in [`../plots/`](../plots/).

| file | result |
|------|--------|
| `fig_memoization_map.png` | where the conclusion is written/read (the core mechanism) |
| `fig_dose_response.png` | recovery vs. how much downstream is recomputed |
| `fig_baseline_frontier.png`, `fig_selective_recompute.png`, `fig_ksweep.png` | the editing cost/correctness frontier and the `field+selective@K` sweep |
| `fig_composable_scaling.png` | transplant TTFT speedup vs. skill length |
| `fig_keystone.png` | editing inside a transplanted skill (edit + compose together) |
| `fig_online_load.png` | online serving throughput / TTFT under load |
| `fig_architecture.png`, `fig_d1_generalization.png` | attention-variant reach and generalization |
