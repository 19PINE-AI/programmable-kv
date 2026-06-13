# Findings — comprehensive online vLLM serving + weight-editing comparison

*Autonomous run, 2026-06-12. One RTX PRO 6000 (96 GB). Harnesses:
`esys/vllm_serving_online.py`, `esys/rome.py`, `esys/weight_editing_compare.py`.
Results: `results/vllm_online_qwen3_8b.json`, `results/weight_edit_compare_llama31_8b.json`.*

These address two reviewer asks: (1) make the vLLM systems result a *real online-serving*
benchmark rather than an offline-batch microbenchmark; (2) empirically compare KV editing
against **weight editing** (ROME / LoRA fine-tune) on the model-editing task.

---

## A. Comprehensive online vLLM serving (replaces the offline microbenchmark)

**Method.** vLLM **V1** engine via `AsyncLLMEngine`, **CUDA graphs ON** (not enforce_eager),
**continuous batching**, **automatic prefix caching (APC) ON**, and **Poisson request arrivals**
at controlled offered rates — the standard vLLM serving methodology. Workload: a shared
**8,066-token** agent policy (real τ²-bench retail policy + neutral padding) with one mutable
field; 96 requests/arm; 64 output tokens each. Two arms:
- **baseline** — new field value written EARLY (in-prefix) → mutating a prefix token changes the
  APC block hash and invalidates everything downstream → full re-prefill every request.
- **erratum** — field keeps OLD value (prefix byte-identical) + short appended `[STATE UPDATE]`
  suffix → long policy prefix is an APC hit; only the suffix + decode is new work.

We measure TTFT p50/p90/p99, end-to-end latency, throughput (req/s, output tok/s), and the
engine's **real APC hit-rate** (scraped from the Prometheus `vllm:gpu_prefix_cache_{hits,queries}`
counters, per arm).

**Results** (Qwen3-8B, prompt ≈ 8k tokens):

| offered rate (req/s) | TTFT p90 baseline | TTFT p90 erratum | TTFT speedup | throughput speedup | APC hit base / err |
|---|---|---|---|---|---|
| 2  | 22,530 ms | 86 ms  | 263× | 1.58× | 1.0% / 98.5% |
| 4  | 35,059 ms | 90 ms  | 388× | 2.68× | 1.0% / 98.5% |
| 8  | 45,864 ms | 115 ms | 398× | 5.07× | 1.0% / 98.5% |
| 16 | 51,054 ms | 273 ms | 187× | 7.93× | 1.0% / 98.5% |
| ∞ (unthrottled / max throughput) | 55,356 ms | 1,043 ms | 53× | **14.53×** | 1.0% / 98.5% |

**Takeaways.**
- **Throughput speedup grows with offered load** (1.58×→14.53×): baseline is prefill/compute-
  bound and **saturates at ≈1.5 req/s**, while the erratum is cache-bound and scales with load.
- **TTFT** under load: baseline queues catastrophically (22–55 s p90 — it cannot keep up), erratum
  stays 86 ms–1.0 s. Speedups 53–398×.
- **The APC hit-rate (1.0% vs 98.5%) directly measures the mechanism**: the append-only erratum
  keeps the long prefix cache-aligned; the in-prefix edit destroys reuse. This is the engine's own
  metric, not our instrumentation.
- This is a genuine online-serving result (continuous batching, CUDA graphs, Poisson load,
  percentile latencies), superseding the prior offline-batch "16×" headline.

---

## B. KV editing vs WEIGHT editing (ROME, LoRA) — model-editing comparison

**Question (reviewer).** To make the model act on a changed field, why not edit the *weights*
(ROME/MEMIT) or fine-tune, instead of the KV cache?

**Setup.** Paper's gated task: "cancel order ONLY IF order_status == pending; else deny." The world
changed pending→shipped (correct decision flips cancel→deny) but the cached context still shows the
OLD value. Each method tries to make the model decide **deny**. Llama-3.1-8B-Instruct.

**ROME is implemented faithfully** (`esys/rome.py`: covariance C=E[kk^T] at a mid MLP layer,
optimized v*, closed-form rank-one update of `down_proj`) and **validated on the canonical factual
edit** before use — "The Eiffel Tower is located in the city of" → *Paris* ⇒ *Rome* after the edit,
locality intact. So the baseline is not crippled.

**Results** (`weight_edit_compare_llama31_8b.json`):

| method | flips decision to deny? | edit latency | cross-request contamination | collateral (unrelated decisions) |
|---|---|---|---|---|
| **KV erratum** | ✓ (gap +2.6) | **114 ms** | **0** (per-sequence) | **0** |
| KV in-place | ✗ (−7.4)* | 71 ms | 0 | 0 |
| **ROME** (rank-one) | ✓ (+19.6) | 5.6 s (+11.6 s one-time covariance) | **1.0** | **0.5** |
| **LoRA fine-tune** | ✓ (+22.1) | 3.1 s | **1.0** | **0.5** |

\*in-place failing without reasoning is consistent with the paper's editing law (direct-decode
Llama; under CoT in-place recovers — §editable).

**Takeaways (the argument).** Even when the weight edits **succeed** at their own target (we made
them succeed — fair baseline):
- **Globality kills per-request isolation.** A weight edit is shared across all requests, so all 8
  other orders that are genuinely still pending were **wrongly flipped to deny (contamination = 1.0)**.
  The same model instance cannot hold status=shipped for user A and status=pending for user B. KV
  editing lives in a per-sequence cache → contamination = 0 by construction.
- **Collateral.** Half (0.5) of a 10-item battery of *unrelated* gated decisions changed their answer
  after the weight edit (the ROME/fine-tune specificity tradeoff). KV editing touches no weights → 0.
- **Latency / volatility.** 3–6 s per edit (+ one-time 11.6 s covariance for ROME) vs **114 ms** for an
  append-only erratum that also composes with APC (§A). State mutates every turn; re-editing weights
  every turn multiplies both cost and collateral.

**Conclusion.** Weight editing (ROME/MEMIT/fine-tune) targets *durable, global facts*; mutable
*per-request, per-turn context state* is the wrong job for it. KV editing is the right substrate:
per-sequence, instant, zero collateral, serving-composable. This is a structural result, independent
of how well the weight edit optimizes its target.
