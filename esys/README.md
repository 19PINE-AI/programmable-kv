# esys — the main experiment system

The rigorous, multi-model harness behind most of the paper: deep-mechanism controls, the
component-level circuit, the editing frontier, composable transplant, the weight-editing
comparison, and the online serving benchmark. Every driver takes a `--model` flag and
writes JSON records into `results/`.

Run any script from the **repo root**, e.g.:

```bash
python esys/mech_suite.py --model Qwen/Qwen3-8B
python esys/cost_frontier.py --model Llama-3.1-8B
python esys/weight_editing_compare.py --model Llama-3.1-8B
```

## Scripts by topic (prefix → purpose)

| group | files | what it covers |
|-------|-------|----------------|
| **Mechanism** | `mech_*.py`, `mechanism.py`, `mech_suite.py` | locality, suffix-concentration, dose-response, reasoning re-read, oracle controls (§3) |
| **Deep controls** | `mechd_*.py` | dissociation, timing, specificity, note-injection — decodability vs. causation (§3, app. deep-mech) |
| **Circuit** | `circ_*.py`, `circuit_common.py` | read/write heads, conclusion direction, SAE feature, attention-vs-MLP, causal scrubbing (app. circuit) |
| **Editing** | `selective_*.py`, `frontier.py`, `cost_frontier.py`, `erratum_*.py`, `diagnostic_eval.py`, `why_erratum.py` | the erratum / `field+selective@K` / in-place edits and the cost/correctness frontier (§4) |
| **Composing** | `composable_*.py`, `compose_edit.py`, `transplant_mech.py`, `mla_*.py`, `arch_erratum*.py` | RoPE-reposition + splice, the keystone edit-inside-transplant, MLA / vision adapters (§5–6, §8) |
| **Weight-editing baseline** | `rome.py`, `weight_editing_compare.py` | faithful ROME + LoRA comparison for mutable per-request state (§4) |
| **Serving / latency** | `vllm_*.py`, `serving_bench.py`, `latency_bench.py`, `compile_*bench.py` | online vLLM benchmark (APC hit-rate, p90 TTFT, throughput), TTFT scaling (§9) |
| **Agentic** | `tau2_*.py`, `taubench_thinking.py` | tau2-bench retail environment (needs `tau2-bench` installed) |
| **Figures** | `make_figures.py`, `make_composable_figures.py`, `make_ksweep_figure.py`, `make_scorecards.py` | regenerate esys-owned figures from `results/` |

`*_common.py` modules are shared machinery (imported, not run directly). Reproducibility
note: experiments ran on a single RTX PRO 6000 (Blackwell, 96 GB) with the official
HuggingFace checkpoints listed in the paper's appendix.
