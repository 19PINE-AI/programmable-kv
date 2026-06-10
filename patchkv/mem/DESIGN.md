# Editable & Composable User Memory in the KV Cache

*Design document for the memory extension of "Models Take Notes at Prefill."*
*Author runs: autonomous overnight session, 2026-06-08.*

---

## 1. Framing

LLM agents carry a **user-memory** document — a dynamically-summarized Markdown profile
of facts about the user (preferences, constraints, history). It is large (thousands to
tens of thousands of tokens), reused across every turn, and **mutated on the fly** by
tool calls as the agent learns new facts. Where it lives in the prompt is a dilemma:

- **Memory at the front** `[sys][MEM][trajectory][decode]`. The trajectory attends to
  memory, so by the paper's mechanism the trajectory has **memoized memory-conditioned
  conclusions downstream**. When memory changes, an in-place edit is *ignored* (the stale
  notes win) and you must reprefill the whole downstream — expensive, and it happens
  every time memory changes.
- **Memory at the end** `[sys][trajectory][MEM][decode]`. Memory's KV depends on the
  entire trajectory, so it must re-attend (re-prefill) every turn as the trajectory grows
  — expensive per *turn*.

Neither is good. This document develops a third option that falls directly out of the
paper's two capabilities:

> **Precompile the memory chunk once (in isolation), place it late, cut the
> memory→trajectory cross-attention, and recompile only when memory changes.**

Concretely the serving layout is

```
[system][past-trajectory][MEMORY][current-user-turn → decode/CoT]
```

- Memory is computed **in isolation** (it attends only to the system prompt), so its KV
  is independent of the live trajectory (the *composable* axis: precompute + RoPE-reposition + splice).
- Memory floats just before the current turn so the **decode and this turn's CoT attend
  to fresh memory** (personalization preserved); it is re-rotated each turn (O(L_mem), cheap).
- When a tool call mutates memory, we **edit** it (recompile the chunk, or append a
  salient erratum, or selective recompute) — the *editable* axis.

Two facts make late placement attractive *for editing specifically*: (a) nothing
downstream of memory can have memoized stale conclusions about it (causal ordering), so
the editing-staleness problem of front placement disappears; (b) memory is semantically
**static w.r.t. the current turn** (a description of the user, updated out-of-band by the
summarizer), so cutting memory→trajectory attention is arguably the *correct* semantics,
not just an approximation.

**The non-obvious risk (load-bearing hypothesis).** Front placement is expensive to edit
*because* the prefill pre-digests memory's facts onto downstream aggregator tokens — and
that pre-digestion is also what makes multi-fact reasoning easy at decode. Late placement
is cheap precisely because it skips pre-digestion, so the decode token (+ its CoT) must
integrate over *raw* memory KV. We therefore expect late placement to lose quality on
tasks requiring **multi-fact integration over long memory**, with the gap growing in
integration depth and memory length, and **chain-of-thought to recover it** (the
generated reasoning tokens become the aggregators). This is the central scientific claim
and **E1** tests it.

---

## 2. Related work (surveyed 2026-06-08)

The space is active as of mid-2026. The closest concurrent work and how we differ:

- **MemArt — "KVCache-Centric Memory for LLM Agents"** (OpenReview, under review). The
  nearest neighbor: stores conversational turns as **reusable KV-cache blocks**, splices
  them **position-independently** via a **decoupled position-encoding mechanism**,
  retrieves relevant blocks by latent attention; 91–135× prefill-token reduction on
  LoCoMo, accuracy near full-context. **Our deltas:** (i) **editing** — MemArt
  retrieves/splices static turns; it has no in-place memory *update/erratum* mechanism;
  (ii) **mechanism** — we give the causal "memoized conclusions / why boundary recompute"
  account that explains *why* decoupled position encoding works and predicts when it
  fails (the pre-digestion limit, E1); (iii) **metric** — MemArt scores QA accuracy; we
  score **decision-governance** (does memory still govern the agent's *action*).
- **EPIC** (ICML 2025, already cited): position-independent caching with LegoLink boundary
  recompute. Our seam-repair is its boundary recompute; we add the memory-edit axis.
- **"Agent Memory Below the Prompt"** (arXiv 2603.04428): persistent Q4 KV cache for
  multi-agent edge inference — quantization + persistence, not editing or placement.
- **SideQuest** (2602.22603), **Adaptive Context Management** (2511.03728): context
  *eviction*/compaction — orthogonal (changes which tokens are present).
- **XKV** (2412.05896): personalized KV-cache *compression*, not reuse/edit.
- **LoCoMo**: de-facto long-conversation memory QA benchmark; we use it for external
  validity (and to be comparable to MemArt).

**Novelty boundary.** "Memory can be KV-cached and spliced" is owned by MemArt+EPIC.
Our contribution is **editable + mechanistically-explained + decision-governed** memory:
the placement law (E1), the edit-method law on the memory substrate (E3), the granularity
frontier (E4), and a working end-to-end agent with statistically-verified
decision-equivalence (E5), all backed by the paper's causal mechanism (E2 keystone).

---

## 3. Experiment design

Conditions vary along three axes: **placement** (early `[sys][MEM][traj]` / late
`[sys][traj][MEM]`), **edit-method** (`stale`, `in_place`, `erratum`, `recompile_chunk`,
`selective@K`, `full_recompute`=oracle), **granularity** (memory as `S` sub-blocks). All
comparisons are **paired** (same item under every condition).

### 3.1 Statistical-rigor protocol (applies to all experiments)

1. **Non-independence.** Decisions are nested decision ⊂ trajectory ⊂ persona, crossed
   with model. Primary inference is **cluster bootstrap (10⁴ resamples) resampling at the
   persona level**, never iid over decisions. For moderator models (placement × n_facts
   …) we use **GEE logistic regression with exchangeable correlation, clustered on
   persona** (cluster-robust SEs) — more stable than GLMM for our cell sizes.
2. **Equivalence, not "n.s."** "Indistinguishable from recompute" is an *equivalence*
   claim. We use **TOST / CI-inclusion**: the 95% CI of the paired difference must lie
   within `[−δ, +δ]`. The margin δ is set **empirically** to `2 ×` the oracle-vs-oracle
   test–retest disagreement (the model's own decision-noise floor), pre-registered below.
3. **Power / sample size.** For an equivalence margin δ = 0.03 on paired proportions
   (true diff ≈ 0, 90% power, α = 0.05): `n ≈ (z_α+z_β)² σ_d² / δ²`. With discordance
   ≈0.05 this gives **≈ 475 paired decisions per (model × condition) cell**; we target
   **≥ 480**. Superiority (detect ≥10pt drop) needs ~150–200, comfortably covered.
4. **Multiplicity.** **Benjamini–Hochberg FDR at q = 0.05** within each experiment's test
   family; **Holm** for the small set of headline equivalence claims.
5. **Determinism.** Decisions read at temperature 0 (argmax of the two decision tokens).
   For CoT, k = 4 sampled chains/seed, decision = majority; report dispersion.
6. **Inclusion threshold.** A (model, task-family) cell enters primary analysis only if
   **oracle decision accuracy ≥ 0.80** (pre-registered; avoids floor effects). Excluded
   cells are reported separately.
7. **Confound controls.** When comparing placements we **hold total sequence length and
   absolute positions fixed** and only move the memory block; we include a
   **full-recompute reference at each placement** so placement effects are separated from
   transplant/RoPE-extrapolation effects.
8. **Negative control.** A memory fact *irrelevant* to the decision must show **no**
   placement/edit effect — a built-in false-positive check.
9. **Pre-registration.** Hypotheses, margins, endpoints, exclusions fixed in
   `PREREG.md` before the confirmatory runs; effect sizes (risk difference, odds ratio,
   Cohen's h) reported with every p-value.

### 3.2 Data

- **Controlled (causal).** Synthetic personas, each with a structured Markdown memory of
  `M` facts. A *gated* decision is determined by `n_facts ∈ {1,2,4,8}` specific memory
  facts (integration depth). Memory length `L_mem ∈ {~1k, ~4k, ~16k, ~32k}` tokens (padded
  with realistic but decision-irrelevant profile facts). This permits clean manipulation of
  depth, length, edits, and a negative control (irrelevant fact).
- **Naturalistic (external).** **LoCoMo** long-conversation QA for external validity and
  MemArt-comparability (accuracy parity *plus* decision-governance variant).

### 3.3 The five experiments

**E1 — Placement × pre-digestion (load-bearing).**
H1: late precompiled memory degrades multi-fact decision accuracy vs early/full-recompute,
gap grows with n_facts and L_mem. H1b: CoT closes the gap.
Design: within-item placement(2) × reasoning(2) × n_facts(4) × L_mem(4); reference =
full-recompute at each placement. Test: GEE logistic with
`placement * n_facts * log(L_mem) * reasoning`; key terms = `placement:n_facts` and its
three-way with reasoning; FDR over interaction terms; predicted-probability curves with
cluster-bootstrap CIs. ≥6 models if GPU allows, spanning scale.

**E2 — Transplant faithfulness / equivalence (compose axis + keystone).**
H2: precompiled+RoPE-repositioned memory is decision- and logit-equivalent to full
recompute. Conditions: oracle; transplant (no repair); transplant+seam@b (b∈{0,1,2,4,8});
isolation- vs context-precompiled. Tests: **TOST** on decision-agreement (margin δ from
the noise floor) and on logit fidelity (cosine ≥ 0.98); seam dose-response (minimal b).
Keystone: re-run the paper's locality/knockout probe with memory as the chunk.

**E3 — Editing memory mid-session (novelty vs MemArt).**
H3: on a mid-trajectory fact flip, `recompile_chunk` and `erratum` preserve decisions;
`in_place` fails; `selective@K` is model-dependent. DV: decision recovery (fraction of
oracle flip) + binary correctness. Tests: GEE for binary; paired bootstrap / Wilcoxon for
recovery; contrasts `in_place < erratum` (superiority) and `erratum ≈ recompile` (TOST);
scale law `recovery ~ method × log(params)`. Report recompute fraction (cost) per method.

**E4 — Edit granularity / sub-chunking.**
H4: sub-chunking into S blocks cuts localized-edit cost ∝ 1/S but degrades decisions when
required facts cross block boundaries (intra-memory attention cut). IV: S∈{1,2,4,8,16} ×
cross-block(within/split). Test: GEE `agreement ~ S × cross_block`; cost model; report
cost–agreement **Pareto frontier** with bootstrap CIs and the knee.

**E5 — End-to-end systems + working app.**
H5: over realistic multi-turn sessions with memory edit-rate r, precompiled-editable
memory reduces cumulative TTFT vs front-recompute and end-recompute, crossover
characterized analytically and empirically — **conditional on decision-equivalence
holding**. A real agent loop (`app.py`) implements the layout, longest-prefix reuse,
per-turn re-rotation, and tool-driven memory edits; we measure cumulative TTFT, decision
agreement vs reprefill-every-turn, across edit-rates and memory lengths.

---

## 4. GPU budget note

This session shares one RTX PRO 6000 (96 GB) with a live training job (~35 GB) and a vLLM
server (~40 GB); ~21 GB is free. We therefore use models ≤ 4B comfortably (Qwen3-0.6B/1.7B/4B
as the primary scale ladder, same family for a clean scale law), and attempt 8B and a
second family (Gemma-2 for sliding-window, Llama/Mistral) opportunistically with short
sequences. Memory length sweeps at 16k/32k are run on the smaller models. All constraints
and any reduced cells are reported honestly in the findings.
