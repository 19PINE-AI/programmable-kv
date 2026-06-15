# results — released result records

Every number in the paper and on the [interactive site](https://01.me/research/programmable-kv/)
is computed from the JSON records in this directory (and `../mem/results/`). Figures and the site
read these files directly — nothing is hand-entered.

## Naming

Files are `<experiment>_<model>[_<variant>].json`. The `<experiment>` prefix tells you which
study produced it; the driver scripts live in `../esys/`, `../e1/`, `../e2/` (see those READMEs).

| prefix | experiment | driver(s) |
|--------|-----------|-----------|
| `mech_*`, `mech_suite_*`, `mech_causal_*`, `mech_oracle_*`, `mech_reasoning_*` | the core mechanism probes (§3) | `esys/mech_*.py` |
| `mechd_*` | deep-mechanism controls — dissociation / timing / specificity / injection (§3, app.) | `esys/mechd_*.py` |
| `circ_*` | component-level circuit — heads / direction / SAE / scrubbing (app.) | `esys/circ_*.py` |
| `selective_*`, `ksweep_*`, `cost_frontier_*`, `frontier_*`, `erratum_*`, `why_erratum_*`, `diagnostic_*` | editing: the erratum, `field+selective@K`, the cost/correctness frontier (§4) | `esys/selective_*.py`, `esys/frontier.py`, … |
| `composable_*`, `compose_edit_*`, `transplant_*`, `mla_*`, `arch_erratum_*` | composing: reposition + splice, edit-inside-transplant, adapters (§5–6, §8) | `esys/composable_*.py`, … |
| `editkv_unified_*`, `agent_*`, `editkv_agent_*` | the unified edit+compose agent (§6) | `esys/editkv_unified.py`, `esys/editkv_agent*.py` |
| `editkv_horizon_*`, `horizon_*` | long-trajectory no-compounding test (app.) | `esys/editkv_horizon.py` |
| `weight_edit_compare_*` | KV editing vs. ROME / LoRA weight editing (§4) | `esys/weight_editing_compare.py` |
| `vllm_*`, `serving_*`, `latency_*` | online serving + latency benchmarks (§9) | `esys/vllm_*.py`, … |
| `tau2_*`, `taubench_*` | the τ²-bench agentic environment (§9) | `esys/tau2_*.py` |
| `composable_vision_*`, `vision_*` | multimodal image-KV transplant (§8) | `esys/composable_vision.py`, `esys/vision_ttft.py` |
| `e1_*`, `e2*_*`, `recovery_*`, `selection_*` | the early exploratory harness (Qwen-1.5B/7B) | `e1/`, `e2/` |

`raw_qwen1p5b/` and `raw_qwen7b/` hold raw outputs from those early exploratory runs.
`selective_Ksweep_*_par.json` are the canonical inputs to the `field+selective@K` figure
(`esys/make_ksweep_figure.py`). User-memory records (E1–E5, LoCoMo) live separately in
`../mem/results/` as `.jsonl`.
