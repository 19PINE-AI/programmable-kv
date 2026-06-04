# Research Plan — Editable KV Cache: Correctness-Preserving In-Place Updates for Mutable Fields in Agentic Contexts

> Working title / system name: **PatchKV** (placeholder)
> Status: pre-implementation plan. **No code until E1 clears go/no-go.**
> Last updated: 2026-06-04

---

## 0. One-paragraph thesis

Prefix caching forces an inference-layer constraint into the application-layer programming model: to keep cache hits, programmers must hoist every mutable field (current time, user/session state, permissions) to the end of the prompt, even when those fields belong semantically elsewhere. We argue this contortion is **unnecessary for the common case**. Because (a) the load-bearing cross-attention is *generation→context*, which is always recomputed live, (b) any rule placed *before* an edited field is *causally independent* of it and exactly reusable, and (c) an in-place field edit perturbs an already-jointly-correct cache far less than RAG-style composition, we hypothesize that **updating only the changed field's KV and leaving downstream KV stale preserves both task quality and agent decisions** for a characterizable class of fields. We (1) characterize the "blast radius" of field edits in long agentic trajectories, (2) provide a correctness-preserving in-place update mechanism that exploits the causally-exact region and refreshes only the sparse residual, and (3) expose it as a programmability abstraction that decouples prompt structure from cache layout.

---

## 1. Problem & motivation

### 1.1 The leaky abstraction
Prefix caching only reuses KV when the prefix matches *exactly*. A single changed token in a 5–10K-token system prompt invalidates everything after it. Consequence: the standard "context-engineering" discipline is to push all mutable content to the suffix. This is a tax nobody chose — it is imposed by the cache mechanism, not preferred by programmers.

### 1.2 The pain is real and current (motivation evidence)
- **Claude Code (early-Mar-2026, Unsloth finding):** prepends a changing attribution header (session id / turn counter / timestamp) to every message, invalidating the prefix cache *every turn* → every turn becomes a full prefill.
- **open-webui issue #24402:** "KV Cache invalidation with dynamic variables in system prompt."
- **"Don't Break the Cache" (arXiv 2601.06007):** documents prompt-caching fragility for long-horizon agentic tasks.

These show the problem is felt in production by the most sophisticated agent systems.

### 1.3 Why "just hoist to the end" is insufficient (the motivation we will defend)
Hoisting works mechanically but complicates agentic programming: fields referenced in multiple places, deeply nested / composed sub-agent prompts, and dynamically assembled prompts cannot all be cleanly hoisted. The contribution is to **let programmers place mutable fields naturally** and have the system maintain cache validity. (Note: we do **not** claim a capability/safety gap from late placement — see §6 dropped framings.)

### 1.4 Research question
> When a small field changes inside an otherwise-static, already-cached agentic context, can we update only that field's KV (plus a sparse, cheaply-identified residual) — leaving the rest stale — and preserve the agent's decisions and task quality at a fraction of re-prefill cost? For which field classes does this hold, and where does it break?

---

## 2. Key insights / hypotheses

- **H1 (Causal asymmetry).** The cross-attention that determines the output is *generation→context*, computed live at decode regardless of cache state. The only direction ever lost by reuse is *context→later-content*, which is largely redundant because integration recurs at the point of use.
- **H2 (Causally-exact region).** Any rule/token positioned *before* the edited field never attended to it (causal mask) → its KV is independent of the field → **exactly reusable, zero recompute, provably correct.**
- **H3 (Edit ≪ composition).** An in-place field edit perturbs an already-jointly-correct cache. The downstream KV deviation is far smaller and sparser than RAG composition (which assembles independently-encoded chunks). Therefore the residual refresh set is ≪ CacheBlend's ~15%, possibly ≈0 for low-conditioning fields.
- **H4 (Central, falsifiable).** For a characterizable class of "low-conditioning" fields (time, ids, session counters, many status flags), **leave-downstream-stale preserves agent decisions and task success** within tolerance. For "high-conditioning" fields that reshape interpretation of many interacting rules, it does not, and the residual must be refreshed (or re-prefilled).
- **The contract.** The achievable floor on refresh is the *downstream mutual information* of the field — proxied by the per-token KV deviation profile of the suffix when the field flips. This is measurable *before* deploying any mechanism.

**Risk note:** H4 runs against the prior-art consensus. Every selective-recompute method (CacheBlend, InfoFlow KV, KVShare-DHD) *recomputes* the affected downstream tokens; none leaves them stale for edits. Demonstrating leave-stale is both our novelty and our principal risk.

---

## 3. Claimed contributions

1. **Characterization** — the first study of the blast radius (KV deviation) *and* decision-flip behavior of in-place field edits in long agentic trajectories, broken down by field class. Establishes the contract for when leave-stale is safe.
2. **Mechanism** — a correctness-preserving in-place update that (i) reuses the causally-exact prior region for free, (ii) refreshes only a sparse residual selected by deviation, and (iii) ideally avoids the mandatory full layer-1 probe that CacheBlend/EPIC/InfoFlow require, by exploiting the *known* edit location.
3. **Abstraction** — a programming model where mutable fields are declared in place and the runtime maintains cache validity, decoupling prompt structure from cache layout.
4. **Systems** — integration into a serving engine (vLLM/SGLang) with a cost/quality frontier vs. the real baselines.

---

## 4. Related work & positioning

### 4.1 The three adjacent clusters and our delta

| Cluster | Representative work (arXiv) | Their setting | Our difference |
|---|---|---|---|
| **Composition** | CacheBlend (2405.16444), EPIC/AttnLink (2410.15332), KV Packet (2604.13226), Prompt Cache (2311.04934), Prompt Choreography (2512.23049) | Assemble *independent* chunks into a new context | We *edit an already-jointly-correct* cache → tiny perturbation + **causally-exact prior region (H2)** that composition lacks |
| **Selection** | InfoFlow KV (2603.05353), KVShare-DHD (2503.16525), CacheBlend | *Which* tokens to recompute (deviation / info-flow) | We treat selection as **known machinery**; our claim is **leave-stale (H4)**, not recompute-the-affected |
| **Sharing** | KVShare (2503.16525), KVCOMM (2510.12872), TokenDance (2604.03143), KVFlow (2507.07400) | Reuse *across requests / agents* | We reuse within **one context over time**, across a field mutation |

### 4.2 Explicit per-paper deltas (for the related-work section)
- **CacheBlend** — non-prefix selective recompute (~15%) for RAG fusion; requires a full layer-1 pass; *composition*, not edit. We are the edit regime with a free exact region and (target) no mandatory full-layer probe.
- **EPIC/AttnLink** — position-independent composition, recomputes ~20 tokens/boundary to kill chunk-start attention sinks; *explicitly does not support in-place edits, mutable fields, or mid-conversation change*. Even SOTA PIC regenerates a whole chunk on edit.
- **KV Packet** — trained Header/Trailer adapters let immutable document caches concatenate recompute-free; *composition of frozen docs*, no mutable fields, requires training.
- **Prompt Cache** — parameterized modules with variable slots, but **tolerates** downstream staleness (accepts quality loss); never updates downstream to reflect the variable. We characterize *when* staleness is harmless and refresh when it is not.
- **Prompt Choreography** — compositional reuse / reordering of pre-encoded components; **explicitly identifies that small edits cause cache misses and does not solve it** (re-encodes the whole component). Cite as acknowledged-but-unsolved.
- **InfoFlow KV** — information-flow-aware recompute selection; *static/append-only RAG*; explicitly not in-place edits / mutable fields / agentic. Closest selector to ours, treated as prior machinery.
- **KVShare** (verified — *Multi-Tenant KV Cache Reuse*, Yang et al.) — DHD deviation-based selective recompute across *different, semantically-similar concurrent prompts*; **recomputes the differing tokens and their downstream dependents — does not leave stale**; uses no "in-place"/"mutable" framing (up to 9.39× TTFT). **Overlap resolved:** different setting (multi-tenant cross-request vs single-context temporal edit), opposite mechanism (recompute downstream vs leave-stale), no causal-exact region, no programmability abstraction.

### 4.3 The single claim no one else makes
> *Leave-downstream-stale, correctness-preserving, for an in-place field edit, training-free, in agentic contexts.* KV Packet leaves stale but only via training for composition; all selective-recompute work refreshes the affected downstream. This intersection is unoccupied — and the field's default assumption is the opposite, which is the experiment's whole point.

### 4.4 Must-cite motivation/context
"Don't Break the Cache" (2601.06007), Lost in the Middle (2307.03172), Premise Order Matters (2402.08939), recency/position-bias literature, and the production reports (Claude Code header finding; open-webui #24402). Security-adjacent: "KV Cache Manipulation Attack" (2511.12752) shows KV editing is feasible (different goal).

### 4.5 Related-work prose (paper-ready draft)

**Prefix and modular context caching.** Production serving engines reuse KV state only across exact shared prefixes (vLLM's automatic prefix caching, SGLang's RadixAttention), so any change to an early token invalidates the entire suffix. Prompt Cache [2311.04934] generalizes reuse to modular, repeatable segments via a markup schema with parameterized slots, but it *tolerates* the resulting cross-segment staleness rather than correcting it, accepting a quality loss. EPIC [2410.15332] makes reuse position-independent through its AttnLink algorithm, recomputing only a small number of boundary tokens (~20 per chunk) to neutralize the attention sink each cached chunk carries at its start; KV Packet [2604.13226] removes even that boundary recompute using trained Header/Trailer adapters. All of these assume immutable, pre-cached chunks that are *composed* at serving time; none supports in-place mutation of a field inside an otherwise-static context, and EPIC explicitly regenerates an entire chunk when its content changes.

**Selective recomputation for non-prefix reuse.** CacheBlend [2405.16444] reuses concatenated chunk caches and recomputes the ~15% of tokens with the highest KV deviation to restore the missing cross-attention; InfoFlow KV [2603.05353] refines which tokens to recompute via an information-flow criterion; KVShare [2503.16525] applies a dual-stage high-deviation rule to multi-tenant similar prompts. These methods share two assumptions we depart from. First, they target chunk *composition* or cross-request *sharing*, not temporal *edits* of a single, already-jointly-encoded context — so they lack the causally-exact prior region that an in-place edit exposes (every token before the edit is provably independent of it). Second, they restore correctness by *recomputing* the affected downstream tokens. Our central hypothesis is the opposite: that for a small in-place field edit, the downstream KV can be left stale while preserving the agent's decisions, because the load-bearing cross-attention (generation→context) is recomputed live at decode regardless of cache state.

**Cross-request and multi-agent KV sharing.** SemShareKV [2509.24832] reuses KV across semantically similar prompts via token-level LSH matching; Prompt Choreography [2512.23049] maintains a shared KV cache that lets multi-agent calls reuse previously-encoded messages in reordered subsets; KVCOMM [2510.12872], TokenDance [2604.03143], and KVFlow [2507.07400] communicate or share KV across agents. These operate *across* requests or agents and are orthogonal to mutation of a field *within* one evolving context. Notably, both Prompt Choreography and SemShareKV observe that small edits to cached content force cache misses, but neither resolves in-place field mutation — they operate at whole-component or full-prompt granularity.

**Prompt structure and position sensitivity.** A body of work shows that *where* information sits in context matters: Lost in the Middle [2307.03172] documents a U-shaped utilization curve, and Premise Order Matters [2402.08939] shows reasoning is brittle to premise ordering. However, recency effects and production practice indicate that placing mutable fields *late* is frequently fine; we therefore do **not** claim a capability penalty from late placement. Our motivation is programmability, not capability.

**The caching–programmability tension in agents.** "Don't Break the Cache" [2601.06007] evaluates prompt-caching fragility for long-horizon agentic tasks, and production reports — Claude Code's per-turn changing attribution header, and open-webui issue #24402 on dynamic system-prompt variables — show that mutable fields invalidate the prefix cache every turn. Current practice resolves this only by hoisting all mutable content to the suffix, contorting prompt structure. We instead let programmers place mutable fields where they belong and maintain cache validity by surgical update, closing the gap that prior modular-caching work names but does not solve.

### 4.6 Head-to-head: Prompt Choreography [2512.23049] vs. this work

Prompt Choreography (PC; Bai & Eisner) is the nearest "programmable prompt structure" neighbor and the one most likely to be conflated with this work, so we differentiate explicitly. PC maintains a *global shared KV cache of named messages* and exposes `prefill(message, parents, offsets)` / `decode(...)` so multi-agent workflows can reuse and reorder previously-encoded messages, repositioning them via on-the-fly RoPE re-rotation (rotate key by (j′−j)). Two properties make it orthogonal to ours:

**(1) Granularity and operation.** PC is *message-level and immutable*: a cached message is frozen, and any content change re-encodes the whole message; it composes/reorders independent messages and has no notion of editing a field *inside* a cached span. Ours is *field-level and mutable* within a single context — exactly the operation PC lacks.

**(2) Opposite ends of the staleness spectrum.** PC reuses *independently-encoded* messages as-is, so reused blocks never co-attended; it explicitly names two failure modes — *information blockage* (a later message encoded without attention to an earlier one) and *information leakage* (a message reused where it now "sees" context it shouldn't, e.g., another agent's private thoughts). This yields severe accuracy loss without fine-tuning (e.g., MultiQA 56.4%→0.4%; MADpar 64.6%→52.4%), recovered only via LoRA (~500 examples). Our setting is the *opposite*: an in-place edit perturbs an *already-jointly-encoded* context, so almost all cross-attention is already correct. The causally-exact region (rules before the field) suffers no blockage (it correctly never attended to the field) and no leakage (single context); the only residual is *staleness w.r.t. the changed value*, milder than blockage and bounded by the field's downstream influence.

**Why this strengthens us.** PC's severe composition-time degradation is evidence *for* our thesis: composition is hard precisely because blocks never co-attended, and an in-place edit is easy precisely because they did. Where PC must fine-tune to recover accuracy, we hypothesize *training-free* correctness for low-conditioning fields and characterize the boundary (E1/E2/E-boundary). Both works share the RoPE-rotation and reuse-without-recompute primitives but apply them to opposite regimes — PC to the hardest (independent composition, paid for with fine-tuning), ours to the mildest (in-place edit, exploiting the free causal-exact region).

---

## 5. Experiment plan

> Run **E1 + a slice of E2 first** as the go/no-go. Do not build the mechanism (E-sys) until H4 survives.

### E1 — Blast-radius characterization *(decisive; do first)*
- **Setup.** Real/realistic agentic trajectories (long system prompt with status fields + multi-step tool-call trajectory). For each field, flip its value and recompute the *oracle* full cache.
- **Measure.** Per-layer, per-token KV deviation (cosine / L2 of K and V) between old and new, across rules and trajectory. Separate the **causally-exact region** (before the field) from the stale region.
- **Output.** Distribution of blast radius (fraction of downstream tokens with deviation > τ) by **field class**.
- **Kill criterion.** Deviation uniformly ≈0 → patch trivially, no paper. Uniformly large for all fields → must re-prefill, no surgery. **Need sparse + field-dependent.**

#### E1 — detailed protocol (runnable design; no code yet)

**Objective.** Quantify how a single in-place field edit perturbs downstream KV, by field class, to (i) decide go/no-go (is the blast radius sparse *and* field-dependent?) and (ii) establish the leave-stale safety contract (the τ threshold).

**Independent variables.**
- *Model* (×3): a small, mid, and large RoPE model (e.g., Llama-3.1-8B, Qwen2.5-32B, Qwen2.5-72B) to test the "weaker → more sensitive" axis. MLA / sparse-attention backbones (DeepSeek-style) deferred to a follow-up.
- *Context source* (×2): (a) **τ-bench** (retail/airline) — the policy document is the *rules*, the user/order/DB state is the *status fields*, with recorded tool-call trajectories — the realistic source; (b) a **synthetic controlled** harness where each field→rule dependency is ground-truthed by construction (a clean knob for conditioning strength).
- *Field class*: **low** (timestamp, request-id, session counter, nonce); **medium** (locale, subscription tier, location); **high** (role/permission e.g. banned↔normal, safety-mode, persona).
- *Edit magnitude*: minor (timestamp tick) vs. semantic (different-meaning value). Use **length-preserving** flips first (pad to equal token length) to remove position-shift confounds; study length-changing edits separately with RoPE re-rotation.

**Procedure** (per model × context × field × flip):
1. Build OLD context; full-prefill → `KV_old`.
2. Build NEW context (field flipped); full-prefill → `KV_new` (the *oracle*).
3. Form the *patched* cache: `KV_old` with only the field's tokens replaced by their in-context recomputed KV; everything else = `KV_old` (leave-stale).
4. For every *non-field* downstream token *t* at layer *l*, compute oracle-vs-patched deviation:
   - **Primary — attention-output deviation:** the change in token *t*'s attention output at layer *l* (most faithful to what propagates; follows CacheBlend's "attention deviation"). Use **eager attention** for measurement, since FlashAttention discards weights.
   - **Secondary:** cosine distance and relative L2 ‖Δ‖/‖·‖ on K and on V.
5. Aggregate.

**Metrics & plots.**
- **P1 (headline).** BR(τ) = fraction of downstream tokens with max-over-layers deviation > τ, shown as a *distribution per field class*. Go/no-go reads directly off this.
- **P2.** Deviation vs. position — verify the causally-exact region (positions before the field) is ≈0 (validates H2; nonzero ⇒ harness bug) and characterize decay after the field.
- **P3.** Deviation vs. layer (where the perturbation concentrates).
- **P4.** Minor vs. semantic flip; τ-bench vs. synthetic.

**Linking deviation → quality (the contract).** Calibrate τ two ways: (a) against CacheBlend's published deviation→F1 curve; (b) directly, by feeding the patched cache to generation and measuring decision agreement (the E2 bridge). The largest τ that still preserves decisions is the contract parameter τ*.

**Go / No-Go decision rule.**
- **GO** if low-conditioning fields have BR(τ*) below a small fraction (target <2–5% of downstream tokens) **and** low- vs high-conditioning classes are clearly separated (field-dependence is real).
- **NO-GO** if BR≈0 for all fields (trivial; no research) **or** BR large for all fields (leave-stale impossible).

**Technical gotchas.**
- Instrument per-layer KV (HF `past_key_values` + forward hooks); eager attention for attention-output deviation.
- Measure deviation of *the rest* (rules + trajectory), **not** the field tokens (those change by construction).
- Control token length: length-preserving flips first; otherwise every position after the field shifts and must be RoPE-re-rotated before comparison.
- Sanity check every run: causally-exact-region deviation ≈ 0.
- Cost: the full-prefill oracle per (context, flip) dominates; sample a manageable contexts × fields grid before scaling.

### E2 — Behavioral faithfulness / decision-flip *(the result with teeth)*
- **Setup.** Compare surgical patch (field-only KV updated, rest stale) vs. full-re-prefill oracle, replaying the same trajectory.
- **Measure.** Decision-flip rate at tool-call points (tool-name argmax, key-argument agreement), action-sequence agreement, final task success. CoT-token count to reach the gated decision.
- **Output.** Per-field-class faithfulness. (User's prior: ≈100% agreement for typical fields → confirms H4.)
- **Kill criterion.** If patch diverges from oracle for *all* fields → leave-stale unsafe → collapse to selective recompute (scooped territory). If it matches even for high-conditioning fields → cross-attention truly irrelevant here → weaker story, pivot to pure systems.

### E-prog — Programmability demonstration *(the heart that differentiates from CacheBlend)*
- **Setup.** Real agent programs written two ways: **natural** (mutable fields in place) vs **hoisted** (fields at end).
- **Measure.** (a) task quality/decisions equivalence natural+PatchKV vs hoisted vs full-re-prefill; (b) cache efficiency (hit rate, TTFT, recompute fraction); (c) a programmability-complexity proxy (forced hoists, template nesting depth, multi-site field references, LoC of cache-discipline glue).
- **Goal.** Natural placement + surgical update ≈ full-re-prefill quality at ≈ prefix-caching cost, while removing measurable structural contortion.

### E-boundary — Failure contract *(rigor)*
- **Setup.** Deliberately construct high-conditioning fields and dense interacting-rule prompts where leave-stale breaks.
- **Measure.** Where quality/decisions degrade; whether the selection mechanism detects and refreshes exactly those tokens; the deviation threshold that maps to a given task-tolerance (calibrated against CacheBlend's curve).
- **Goal.** A clean, stated contract: which field classes are leave-stale-safe vs refresh-required.

### E-sys — Mechanism, frontier, integration *(only after H4 holds)*
- **Setup.** Implement in vLLM/SGLang: in-place field KV update + causally-exact reuse + sparse residual refresh; ideally a refresh-set predictor from the *known* edit location (avoid mandatory layer-1 probe).
- **Measure.** Quality/decision agreement vs **recompute fraction** vs **latency**, swept.
- **Baselines.** (1) full re-prefill (ceiling), (2) stale full-reuse (floor), (3) **hoist-to-end + prefix caching (the real baseline to beat)**, (4) CacheBlend/InfoFlow generic selection. Charge prediction overhead honestly.
- **Kill criterion.** If we cannot beat hoist-to-end + prefix caching on the joint *programmability × efficiency × correctness* axes, fall back to a measurement-only paper.

### E-horizon — Compounding error over long trajectories *(honest risk)*
- **Setup.** Apply a patch early, continue the trajectory many steps.
- **Measure.** Does decision-agreement vs oracle decay with steps-after-patch? (Note: CacheBlend measures F1/Rouge, **not** degeneration or compounding — this is our novel axis.)
- **Goal.** Either "stays flat" (strength) or a characterized decay limit (contribution).

---

## 6. Framing decisions (committed)

- **Lead framing:** *programmability* — decouple prompt structure from cache layout; restore natural field placement. Efficiency is the payoff, not the pitch.
- **Dropped — safety/reliability hook** ("caching makes agents miss changes"): user expects the agent will *not* miss the change under a patch; reframed as **behavioral-faithfulness validation** (E2), not a failure hunt.
- **Dropped — capability-tension** ("must place fields early"): falsified by recency + production practice; not used as motivation.
- **Avoid:** pitching the selection mechanism as novel (owned by InfoFlow/KVShare/CacheBlend).

---

## 7. Risks & open decisions

| Risk | Mitigation / decision |
|---|---|
| **H4 fails (leave-stale unsafe)** → collapse to selective-recompute | E1/E2 first; if it fails, pivot to measurement-only or kill |
| **Beaten by hoist-to-end baseline** | E-prog must show real programmability win; otherwise downgrade venue |
| **KVShare language overlap** | ✅ Resolved — verified multi-tenant + recompute-downstream; clearly differentiated (§4.2) |
| **Scoop velocity** (KV Packet Apr'26, InfoFlow Mar'26) | Lead with abstraction+characterization (scoop-resistant), move fast on E1/E2 |
| **Compounding error** | E-horizon; report honestly as limit or strength |

**Open decisions to resolve:** (a) which agent benchmark(s) — τ-bench / WebArena / AgentBench / SWE-agent / ToolBench / synthetic-controlled; (b) robust "decision-flip" definition for free-form tool arguments; (c) field taxonomy (low- vs high-conditioning) operationalization; (d) deviation metric + threshold calibration.

---

## 8. Venue & sequencing

- **Venue:** MLSys / LLM-serving track primary; ACL/EMNLP/NeurIPS D&B viable if the characterization (E1/E2/E-horizon) is surprising enough to stand alone.
- **Sequence:** (1) E1 + E2 slice → go/no-go; (2) read KVShare, draft related-work; (3) E-prog + E-boundary; (4) E-sys + E-horizon; (5) write.

---

## Appendix — paper list (arXiv)

**Verified (ID + title + authors confirmed on arXiv):**
- **KVShare** 2503.16525 — *An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse*, Yang, Zhang, Huang, Wang, Tang, Li, Liu, Zhang.
- **KV Packet** 2604.13226 — *Recomputation-Free Context-Independent KV Caching for LLMs*, Chen, Zhang, Yin, Zhuo, Li, Schlichtmann.
- **InfoFlow KV** 2603.05353 — *Information-Flow-Aware KV Recomputation for Long Context*, Teng, Zhang, Zheng, Zhuo, Zhou, Wang.
- **Prompt Choreography** 2512.23049 — *Accelerating Language Model Workflows with Prompt Choreography*, TJ Bai, Jason Eisner. (2.0–6.2× TTFT)
- **SemShareKV** 2509.24832 — *Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching*, Zhao, Mastorakis.
- **Don't Break the Cache** 2601.06007 — *An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks*, Lumer, Nizar, Jangiti, Frank, Gulati, Phadate, Subbiah. (41–80% cost, 13–31% TTFT)

**High-confidence, established (not re-verified this pass):**
- CacheBlend 2405.16444 · EPIC 2410.15332 (algorithm: AttnLink) · Prompt Cache 2311.04934 · Lost in the Middle 2307.03172 · Premise Order Matters 2402.08939

**From search results — VERIFY ID/title before citing:**
- KVCOMM 2510.12872 · TokenDance 2604.03143 · KVFlow 2507.07400 · CacheTTL 2511.02230 · Strata 2508.18572 · LMCache 2510.09665 · KV Cache Manipulation Attack 2511.12752

**Production / non-arXiv:** Claude Code per-turn attribution-header finding (Unsloth, ~Mar 2026); open-webui issue #24402 (dynamic system-prompt variables invalidate KV cache).
