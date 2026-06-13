# Programmable KV Cache

### Models Take Notes at Prefill: KV Cache Can Be Editable and Composable

> 🔎 **Interactive companion (recommended starting point):**
> **https://01.me/research/programmable-kv/**
> — every figure is driven by the released result records; walk the mechanism, the
> circuit, and every experiment interactively.
>
> 📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) (32 pp., preprint — under review)
> · 🧑‍🏫 **Gentle intro:** [`EXPLAINER.md`](EXPLAINER.md)

---

## The one-paragraph version

When an LLM agent reuses a cached prefill, changing a single token inside the reused
region — a timestamp, a user id, an order's status — normally invalidates the entire
downstream cache. You might hope to surgically refresh just that field's key/value
vectors and keep the rest. **It doesn't work**, and *why* it fails is the discovery:
at prefill the transformer has already computed the **field-conditioned conclusion**
and written it onto downstream aggregator/delimiter tokens; at decode the decision
reads those *notes*, not the field. We establish this causally (the field's own KV
drives **under 1%** of the decision), resolve it to a **component-level circuit**
("distributed write, concentrated read"), and replicate it across four model families.

That reframing — the KV cache as a *notebook of memoized conclusions* — makes the cache
a first-class object you can **program**:

- **Editable.** Amend the notes with a one-line salient **erratum** instead of
  recomputing. Matches the hoist-to-end oracle with no prompt surgery, append-only so it
  stays cache-aligned (online serving: **98.5% vs 1% prefix-cache hit-rate**, up to
  **14.5×** throughput, **53–398×** lower p90 TTFT).
- **Composable.** Precompile a reusable **skill** once and RoPE-reposition its cached KV
  into any context — behaviorally indistinguishable from full recompute (logit cosine
  **0.90–0.999**) at **O(L)** instead of O(L²) time-to-first-token (**13.9×** at 32k).

A *keystone* experiment — editing a field **inside** a transplanted skill — shows the two
operations act on one substrate; a unified edit+compose agent stays decision-identical to
full recompute across thirteen models. The substrate is any per-token attention KV cache:
validated across scale, quantization, MoE, and multimodal image caches, with small
adapters for MLA, interleaved M-RoPE, and sliding-window attention.

## Repository layout

| Path | What it is |
|------|------------|
| [`paper/`](paper/) | LaTeX source (`main.tex`), figures, and the built `main.pdf` |
| [`site/`](site/) | The interactive companion website (Vite + React) — see [`site/README.md`](site/README.md) |
| [`e1/`](e1/), [`e2/`](e2/) | Mechanism harness: blast-radius capture, gated-decision scenarios, cache machinery |
| [`esys/`](esys/) | Main experiment system: deep-mechanism controls, the component circuit, the editing frontier, composable transplant, weight-editing comparison, and online serving |
| [`editkv/`](editkv/) | Core editable-KV module (`patchkv_cache`: exact field refresh + optional recency recompute) |
| [`mem/`](mem/) | User-memory application (E1–E5, LoCoMo external validity, cross-referential test) |
| [`results/`](results/) | Result records (JSON) and raw run logs — the source of every number on the site |
| [`figures/`](figures/), [`plots/`](plots/) | Generated figures |

### Background notes

[`EXPLAINER.md`](EXPLAINER.md) (no background assumed) · [`MECHANISM.md`](MECHANISM.md)
(the mechanistic account) · `FINDINGS_*.md` (per-phase result logs) · [`PAPER.md`](PAPER.md)
(extended write-up).

## Reproduce

```bash
# the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

# the figures (run with the repo root as cwd)
python paper/figs/make_figures.py
python paper/figs/make_circuit_figure.py

# the interactive site
cd site
python3 data/build_data.py      # rebuild curated data from results/ (asserts 22 numbers vs the paper)
npm install && npm run build     # -> site/dist/ (static; host anywhere)
```

Experiments were run on a single RTX PRO 6000 (Blackwell, 96 GB); model checkpoints are
the official HuggingFace releases listed in the paper's appendix. The `esys/` and `mem/`
drivers each take a `--model` flag.

## Status & attribution

**Bojie Li** · Pine AI · preprint, under review. Code and interactive companion:
<https://github.com/bojieli/programmable-kv>. A reproducibility statement and the full
model list are in the paper's appendix.

```bibtex
@article{li2026programmablekv,
  title  = {Models Take Notes at Prefill: KV Cache Can Be Editable and Composable},
  author = {Li, Bojie},
  note   = {Preprint. Under review.},
  year   = {2026},
  url    = {https://github.com/bojieli/programmable-kv}
}
```
