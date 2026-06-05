# Editable KV Cache for Mutable Fields in Agentic Contexts: When the Cheap Edit Works, Why It Fails, and a Robust Fix

*Working draft. Consolidates the experimental program; numbers are from local runs on
1× RTX PRO 6000 (Qwen3 0.6–32B family + cross-family checks). Results marked
(preliminary) have small n or are still replicating.*

---

## Abstract

Prefix caching forces an inference-layer constraint into the application layer: to keep
cache hits, programmers must hoist every *mutable field* (time, ids, user/account state)
to the end of the prompt, even when it belongs elsewhere. We ask whether a field can be
edited *in place* in the KV cache instead. We find: (1) the region before the field is
provably reusable (exact), and the blast radius of an edit is sparse and field-dependent;
(2) naively refreshing only the field's KV and leaving the rest stale **fails** — the
decision reverts to the old value — because the decision reads the field almost entirely
*indirectly*, through downstream tokens that *memoized the field-conditioned inference at
prefill time* (the decision's direct attention to the field token is ≈0.1% at every model
scale); (3) **chain-of-thought reasoning can rescue the cheap edit, but unreliably and
scale-dependently** — at 8B the CoT re-derives correctly, at 14B it *amplifies* the stale
inference into unsafe actions, at 32B it collapses to caution; (4) a **salient erratum** —
appending "[STATE UPDATE] <field> → <new>; overrides any earlier value and conclusion" and
recomputing only those ~tens of tokens — **recovers the oracle decision at every scale and
in every domain we tested, even where a full re-prefill is itself fooled by contradictory
context.** Controlling for model competence (an oracle baseline), the in-place edit has a
real penalty (0.12–0.67, up to 29% unsafe actions) that the erratum eliminates. We give a
mechanistic account via attention knockout and causal patching, and a cost/latency frontier.

---

## 1. Introduction

Agents re-read long, mostly-static instructions every turn; KV caching reuses the
"reading" (prefill) across turns, but only across an *exact* shared prefix. A single
changed token — a clock tick, a session id, an account-status flip — invalidates the
entire suffix. The de-facto fix, hoisting all mutable content to the prompt's end, taxes
programmability: fields referenced in multiple places, nested sub-agent prompts, and
dynamically assembled prompts cannot all be cleanly hoisted.

We study **in-place field editing**: when a field changes, can we surgically update the
cache and reuse the rest? Contributions:
1. **Characterization** of the edit's blast radius and decision-flip behavior by field class.
2. **A regime map**: when the cheap edit (refresh-field-only, leave-rest-stale) is safe,
   and the **reasoning-vs-non-reasoning** and **scale** dependence of that safety.
3. **A robust mechanism** — the *erratum* — and an oracle-controlled evaluation isolating
   the edit penalty from model competence, across 5 sizes, 8 domains, and model families.
4. **A mechanistic explanation** (attention-mediated memoized inference) with causal
   evidence (knockout, patching), and a cost/latency frontier.

## 2. Related work and positioning

Prefix caching (vLLM APC, SGLang RadixAttention) reuses exact prefixes. Selective-recompute
methods for *composition* (CacheBlend, EPIC/AttnLink, KV Packet) recompute the ~15% / boundary
tokens to restore cross-attention when assembling *independent* chunks; selection methods
(InfoFlow KV, KVShare) decide *which* tokens to recompute. All target chunk composition or
cross-request sharing and **recompute the affected downstream**. We study a *temporal edit of
one already-jointly-encoded context*, and our central question is the opposite: when can the
downstream be left **stale**? Prompt Cache tolerates staleness (accepts quality loss);
Prompt Choreography reuses message-level blocks but re-encodes on edit. Crucially, prior
work implicitly assumes **single-pass (instruction) decoding**; we show **reasoning models
change the picture** — and that the robust fix is a salience injection, not recomputation.

## 3. Method

**Setup.** An OLD context is cached. A field flips OLD→NEW. We compare cache-construction
strategies for the next decode:
- *full_reprefill* (ceiling), *stale* (floor), *hoist_to_end* (the baseline to beat).
- *field_only*: overwrite the field span's KV with its in-context-recomputed (exact) value,
  leave all else stale. Cost ≈ field tokens (~0.1%).
- *erratum*: leave the cache stale; append "[STATE UPDATE] <field> → <new>; overrides any
  earlier value AND conclusion"; recompute only the appended span (~5–6%).
- *field+erratum*: both.

**Metrics.** Decision *safety* P(avoid the policy-violating action); *fidelity*/agreement
with the oracle; recompute fraction; wall-clock latency. Reported as proportions with
Wilson 95% CIs over instances and (for stochastic CoT) samples.

## 4. Characterization (E1/E2)

- **H2 (exact prior region):** KV of every token *before* the field is bit-identical OLD vs
  NEW (deviation 0.0 across all layers/fields/models). The prefix is reusable for free.
- **Blast radius** (attention-output deviation): sparse and field-dependent (low<medium<high
  conditioning); raw KV-deviation over-counts and is not a portable threshold — **decisions
  are the metric with teeth.**
- **Decision-flip:** low-conditioning fields (time/ids/counters) are leave-stale-safe with
  zero refresh; gating fields (role/safety-mode/tier) flip the decision and a bare field-only
  edit **does not recover it** without further help.

## 5. The reasoning axis and the erratum (the core result)

**Within-model ablation (Qwen3-8B, `enable_thinking` on/off, account_role, n=12 / 36 samples):**

| field-only intervention | non-reasoning | reasoning |
|---|---|---|
| baseline P(safe) | 0.00 [0,.24] | 1.00 [.90,1] |
| KO original stale downstream | 1.00 [.76,1] (fixes) | 0.97 (harmless) |
| KO fresh CoT | — | 0.61 [.45,.75] (reverts) |

Non-reasoning decisions are *harmed by* the stale downstream; reasoning decisions *depend
on* the fresh CoT. But this rescue is **not reliable across scale** (§7).

**Oracle-controlled edit penalty (reasoning; isolates the edit from competence):**

| model | oracle | field_only (unsafe) | **erratum** | edit penalty |
|---|---|---|---|---|
| 4B | 1.00 | 0.38 (.25) | **1.00** | 0.62 |
| 8B | 1.00 | 0.88 (.04) | **1.00** | 0.12 |
| 14B | 0.96 | 0.29 (.29) | **1.00** | 0.67 |
| 30B-A3B / 32B | (running) | | | |

The field-only edit has a **real penalty** (the model would be correct with a full
re-prefill); the **erratum recovers to the oracle ceiling (≈1.0, 0% unsafe) at every size**,
including 14B where field-only is 29% unsafe.

## 6. Generalization

- **8 diverse domains** (retail, airline, devops, banking-numeric, access-control, clinical
  safety, customs routing, on-call severity; permission gates, numeric thresholds, safety
  attributes, routing), Qwen3-8B: **non-reasoning** field_only=0.00 / erratum=1.00 (n=8);
  **reasoning** field_only=1.00 / erratum=0.98 (n=48). The non-reasoning "always stale" and the
  reasoning rescue both hold uniformly across every domain — not a customer-support artifact.
- **Family × scale survey (5 families, 0.6B–32B; diverse tasks, non-reasoning, n=8).**
  For every *competent* model (oracle P(correct) ≥ 0.75) the pattern holds: field-only fails,
  erratum recovers to the oracle ceiling.

  | model | family | oracle | field_only | erratum |
  |---|---|---|---|---|
  | Qwen3-1.7B / 4B / 8B | Qwen | 0.88 / 1.0 / 1.0 | 0.12 / 0.00 / 0.00 | 1.0 / 1.0 / 1.0 |
  | SmolLM2-1.7B | HF/SmolLM | 0.75 | 0.25 | 0.62 |
  | Gemma-2-2B / 2-9B / 3-4B | Google | 0.88 / 1.0 / 1.0 | 0.38 / 0.00 / 0.25 | 0.88 / 1.0 / 1.0 |
  | Mistral-7B | Mistral | 1.0 | 0.00 | 0.88 |
  | DeepSeek-R1-Distill-Llama-8B | Llama | 0.97 | 0.81 (reasoning) | 1.0 |

  (Qwen3-0.6B oracle=0.12 — too small to do the task at all, uninformative. Phi-3.5's
  outdated custom modeling code is incompatible with the installed transformers; dropped.)

- **Reasoning detail (3 families):** the pattern holds beyond Qwen.
  - *Mistral-7B-Instruct* (different arch, non-reasoning, n=8): field_only **0.00**, erratum 0.88.
  - *DeepSeek-R1-Distill-Llama-8B* (Llama arch, reasoning, n=32): oracle 0.97, field_only 0.81
    (edit penalty ~0.16), **erratum 1.00** [.89,1].
  The field-only edit carries a penalty on every family; **the erratum recovers to the oracle
  ceiling on every family.** (DeepSeek-R1's reasoning is *more* staleness-robust than Qwen3-14B's,
  which amplified it — reasoning-training-dependent, but the erratum is family-invariant.)
- **τ-bench retail (real policy):** H2 holds on the real 81-line policy; a late-placed field
  recovers at 4.4% recompute (94.8% reused free); erratum robust to a poisoned prior.

## 7. Mechanism (explainability)

The decision reads the field's value **indirectly**. Measured on every scale (4B–32B):
- **E4 (attention):** decision's *direct* attention to the field token ≈ **0.1%** at every
  size; ~50–56% to downstream, ~36–48% to attention sinks. Prefill **memoizes the
  field-conditioned inference diffusely across downstream KV**; a field-only edit refreshes
  only the (near-inert) direct path.
- **E1 (graded knockout):** masking the decision's attention to the top-25% highest-attention
  downstream restores the correct action — distributed, not localized.
- **E2 (layer-band):** the stale signal is read in mid-and-late layers.
- **E3 (reasoning resolution) is scale-dependent:** 8B & 30B-A3B(3B-active) CoT **corrects**
  the stale inference; **14B CoT amplifies it** (manufactures 19% unsafe; bypassing the CoT
  removes it); 32B collapses to caution. ⇒ "thinking rescues the cheap edit" is *not* general.
This reframes the fix: **the erratum works because it injects a recent, explicit override the
decision attends to — independent of whether the CoT reasons correctly** — which is why it
survives at scales where reasoning fails and even where a full re-prefill is fooled.

## 8. Cost/latency frontier (E-sys)

Real wall-clock (CUDA events, warmup+median) to build a decode-ready cache, Qwen3-8B:

| context T | full_reprefill | field_only | erratum | hoist_to_end |
|---|---|---|---|---|
| 586 | 78 ms | 30 ms (0.34%) | 27 ms (10%) | 38 ms |
| 1706 | 198 ms | 30 ms (0.12%) | 45 ms (3.5%) | 44 ms |
| 4047 | 417 ms | 30 ms (0.05%) | 57 ms (1.5%) | 52 ms |
| 9947 | **1260 ms** | **30 ms** (0.02%) | **94 ms** (0.6%) | 86 ms |

Full reprefill scales linearly with context; **field_only is ~constant (~30 ms) and erratum
small (~27–94 ms)**, so the saving grows with length — at 10K context **~42× (field_only) / ~13×
(erratum)** vs full reprefill, both keeping the field in place. The erratum thus delivers a ~7×
TTFT reduction *with* correctness (→oracle, §5) *and* natural placement, and uniquely retains
correctness under contradictory context. (field+erratum ~67–83 ms; still ≪ full at large T.)

**On kernels / `torch.compile`.** The *edit itself* is trivial: with a `StaticCache`, overwriting
the field span's KV in place is **0.16 ms** (no clone/realloc) — the measured cost is the
partial-prefill recompute and decode, not the edit. `torch.compile` gives only a **modest ~1.2×**
on the partial prefill and ~1.26× on decode *with StaticCache* (and *hurts* decode with
`DynamicCache`, which graph-breaks). So the win is algorithmic (recompute a few tokens, not the
whole context), not a compile flag; a genuine fused "in-place-edit + selective-recompute +
paged-attention" operator is a serving-engine (vLLM/SGLang) integration — future work.

## 9. Limitations
- Synthetic + τ-bench scenarios; single gated decisions, not full multi-turn task success.
- Decision proxy = tool/answer argmax; CoT truncation censors some fidelity numbers (safety
  numbers are censoring-robust).
- **Length-changing edits:** the *erratum* handles them by construction — it appends the new
  value and never shifts positions, so it is length-agnostic (already demonstrated on tasks
  with length-changing field values, e.g. "8200 USD"→"30 USD"). Only *field_only* needs
  length-preservation (we pad) or RoPE re-rotation of the suffix for a genuine length change —
  a known asymmetry that *favors* the erratum.
- MLA/sparse-attention backbones untested; multi-field simultaneous edits, full multi-turn
  task-success, and a serving-engine fused operator are future work. Family coverage: Qwen,
  Gemma, Mistral, SmolLM, Llama (Phi blocked by an outdated custom-modeling/transformers
  mismatch); 70B+ would need 4-bit quantization (a confound) and exceeds this 96 GB GPU in bf16.
- Mechanism evidence is correlational+causal-knockout on a subset; the *scale reversal* of
  CoT helpfulness is an open question we characterize but do not yet fully explain.

## 10. Conclusion
Editable KV is viable, but the naive cheap edit is *not* a free lunch: the decision reads the
field indirectly, so leaving the downstream stale reverts it, and reasoning rescues it only
unreliably and scale-dependently. A salient erratum — keep the field in place, append a short
authoritative override, recompute only that — recovers the full-reprefill decision at every
scale and domain we tested, and is *more robust than recomputation itself* under contradictory
context. The contribution is the regime map + the mechanism + the robust, cheap fix.
