# Programmable KV Cache

### Models Take Notes at Prefill: KV Cache Can Be Editable and Composable

> 🔎 **Interactive companion (recommended starting point):**
> **https://01.me/research/programmable-kv/**
> — every figure is driven by the released result records; walk the mechanism, the
> circuit, and every experiment interactively.
>
> 📄 **Paper:** [arXiv:2606.17107](https://arxiv.org/abs/2606.17107) (33 pp.)
> · 🧑‍🏫 **Gentle intro:** [`docs/EXPLAINER.md`](docs/EXPLAINER.md)

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
operations act on the same notes; a unified edit+compose agent stays decision-identical to
full recompute across thirteen models. The approach applies to any per-token attention KV
cache: validated across scale, quantization, MoE, and multimodal image caches, with small
adapters for MLA, interleaved M-RoPE, and sliding-window attention. The longer-term vision
is a KV cache that is **programmable by design** — models trained to expose composable,
editable notes rather than relying on the mechanism arising for free.

## Repository layout

| Path | What it is |
|------|------------|
| [`paper/`](paper/) | LaTeX source (`main.tex`), figures, and the built `main.pdf` |
| [`site/`](site/) | The interactive companion website (Vite + React) — see [`site/README.md`](site/README.md) |
| [`e1/`](e1/), [`e2/`](e2/) | Mechanism harness: blast-radius capture, gated-decision scenarios, cache machinery (each has a `README.md`) |
| [`esys/`](esys/) | Main experiment system: deep-mechanism controls, the component circuit, the editing frontier, composable transplant, weight-editing comparison, and online serving (see [`esys/README.md`](esys/README.md)) |
| [`editkv/`](editkv/) | Core editable-KV module (`EditableContext`: in-place edit + erratum, with a per-edit diagnostic) — see [`editkv/README.md`](editkv/README.md) |
| [`mem/`](mem/) | User-memory application (E1–E5, LoCoMo external validity, cross-referential test) — see [`mem/README.md`](mem/README.md) |
| [`results/`](results/) | Result records (JSON) — the source of every number in the paper and on the site; see [`results/README.md`](results/README.md) for the filename→experiment legend |
| [`figures/`](figures/), [`plots/`](plots/) | Generated figures: standalone paper-style renders (`figures/`) and legacy exploratory plots from the early Qwen-1.5B/7B runs (`plots/`) |
| [`docs/`](docs/) | Background notes: gentle intro + the mechanistic account |
| `requirements.txt`, [`LICENSE`](LICENSE) | Python dependencies; Apache-2.0 license |

### Background notes

[`docs/EXPLAINER.md`](docs/EXPLAINER.md) (no background assumed) ·
[`docs/MECHANISM.md`](docs/MECHANISM.md) (the mechanistic account). The paper
([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)) is the canonical write-up.

## Reproduce

```bash
# 0. dependencies (Python 3.9+); see requirements.txt for optional vllm / tau2-bench
pip install -r requirements.txt
pip install -e editkv             # the standalone editable-KV module

# the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

# the figures (run with the repo root as cwd)
python paper/figs/make_figures.py
python paper/figs/make_circuit_figure.py
python paper/figs/make_appendix_figures.py
python paper/figs/make_horizon_figure.py
python mem/make_figs.py            # user-memory (E1–E5, LoCoMo) figures

# reproduce experiments from scratch (records land in results/; see each dir's README)
python esys/mech_suite.py --model Qwen/Qwen3-8B    # mechanism probes  (esys/README.md)
python e2/run_recovery.py --model Llama-3.1-8B      # editing recovery  (e2/README.md)

# the interactive site
cd site
python3 data/build_data.py      # rebuild curated data from results/ (asserts 22 numbers vs the paper)
npm install && npm run build     # -> site/dist/ (static; host anywhere)
```

Experiments were run on a single RTX PRO 6000 (Blackwell, 96 GB); model checkpoints are
the official HuggingFace releases listed in the paper's appendix. The `esys/` and `mem/`
drivers each take a `--model` flag.

## Status & attribution

**Bojie Li** · Pine AI · arXiv preprint [2606.17107](https://arxiv.org/abs/2606.17107).
Code and interactive companion: <https://github.com/19PINE-AI/programmable-kv>. A
reproducibility statement and the full model list are in the paper's appendix. Released
under **Apache-2.0** (see [`LICENSE`](LICENSE)).

```bibtex
@article{li2026programmablekv,
  title         = {Models Take Notes at Prefill: KV Cache Can Be Editable and Composable},
  author        = {Li, Bojie},
  year          = {2026},
  eprint        = {2606.17107},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2606.17107},
  url           = {https://arxiv.org/abs/2606.17107}
}
```
