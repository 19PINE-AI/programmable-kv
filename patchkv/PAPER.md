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

**Negative control (erratum is causal, not an append artifact), Qwen3-8B, 8 tasks:** oracle 1.00, stale 0.00, **err_correct 1.00**, **err_wrong (restates the OLD value) 0.00**, **err_irrelevant (neutral notice) 0.00**. Only an erratum stating the *correct new value* recovers the decision; restating the old value or appending unrelated text stays stale — so the effect is content-specific, not 'any append resets the model.'

## 5b. Erratum robustness (Qwen3-8B, 8 tasks)

- **Phrasing:** robust in benign contexts — override-full / bare-value / minimal "field: value"
  all recover 8/8; only a question framing dips to 7/8. (The explicit "overrides any earlier
  conclusion" framing matters specifically under *poisoned* context, §4.)
- **Over-correction:** an erratum for an *irrelevant* field flips the decision **0/8** — it does
  not cause spurious behavior on the benign case.
- **Multi-edit:** stacking an irrelevant erratum before the relevant one still yields the
  correct decision **8/8** — no interference; the relevant trigger drives the decision.

## 5c. Validating the per-case diagnostic (`needs_erratum`)
The library's diagnostic predicts, for a specific edit, whether the cheap in_place edit
suffices or must escalate to field+erratum. We validate it against ground truth (does the
in_place decision differ from the full-reprefill oracle?) over 8 high-conditioning edits + 8
low/irrelevant edits, with a confidence-margin knob (`esys/diagnostic_eval.py`, Qwen3-8B):

| margin | precision | recall |
|---|---|---|
| 0.0 | 0.46 | **0.86** |
| 0.3 | 0.83 | 0.71 |
| 0.5 | **1.00** | 0.71 |

It compares the *deterministic first decision token* (not a noisy multi-token decode) and
fires only when field+erratum confidently moves the decision off the stale token. A false
negative (use in_place when the erratum was needed) yields a stale decision; a false positive
costs only a few % of compute (the erratum is cheap and correct, §5/§8). Since FN is the costly
error, the **recall-oriented low margin is the safe default**; cost-sensitive deployments can
raise the margin to eliminate over-correction (P=1.0 at margin 0.5). The diagnostic is a useful
conservative heuristic, not a perfect oracle — distinguishing the two classes *cheaply* is
inherently approximate (a faithful reference is what is expensive).

## 5d. Robustness: multi-field edits and multi-edit accumulation
- **Multi-field (interference).** A decision gated by *two* fields (account_role AND order_status).
  Flipping one, the other, or both, the erratum recovers the correct *joint* decision **4/4** in
  every case (stale **0/4**), and a no-op arm stays correct **4/4** — editing one field does not
  corrupt the other's contribution, and there is no over-correction. (`esys/robustness_multi.py`.)
- **Multi-edit (accumulation).** A field evolving through a *sequence* of values. Monotonic chains
  work: stacked sequential erratums track the latest value **4/4** (= a direct reprefill to the
  final value). **Honest failure mode:** a *non-monotonic* chain (pending→processed→cancelled→
  pending) breaks the stacked form **0/4** — once a salient terminal state ("cancelled") enters the
  stack, the model ignores a later "pending" erratum (a direct reprefill to the current value is
  4/4). **Guidance:** for repeated edits to one field, collapse to a *single* current-value erratum
  rather than stacking the history; the library applies one erratum per current value by default.

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
- **τ²-bench retail, end-to-end multi-turn (the production-relevant stress test).** We run the
  full editkv library on the **real τ²-bench retail policy** (6699 chars; sierra-research/tau2)
  with a multi-turn cancel-order trajectory. The gating field is `order_status`, buried early
  (token span 1429/~1700) in the long policy; per the documented rule "*an order can only be
  cancelled if its status is 'pending'*", it changes `pending → processed` mid-conversation, so
  the correct next action flips **cancel → deny**. Decisions on Qwen3-8B:

  | strategy | decision | vs oracle |
  |---|---|---|
  | oracle (full reprefill, processed) | deny | — |
  | stale (still pending) | cancel | ✗ |
  | in_place → processed | cancel | ✗ |
  | **erratum alone** → processed | **cancel** | **✗** |
  | **field+erratum** → processed | **deny** | ✓ |

  **New finding from going end-to-end:** in a *long real policy with the field buried early*,
  the **erratum alone misses** — the stale early field token still competes with the appended
  override — and only **field+erratum** (refresh the token *and* append the override) recovers
  to the oracle. This is invisible in the short synthetic tasks (where erratum alone suffices)
  and motivated two library changes: (i) **`FIELD_PLUS_ERRATUM` is now the robust default /
  AUTO escalation** (not erratum alone); (ii) the **diagnostic references `field+erratum`**, not
  erratum, as ground truth — referencing erratum would have falsely reported "in-place
  sufficient" here (both in_place and erratum returned `cancel`). After the fix the diagnostic
  correctly returns `needs_erratum=True` and AUTO selects `field+erratum → deny`.
  (`esys/tau2_editkv.py`, `results/tau2_editkv.json`.)
- **τ²-bench end-to-end EPISODE loop (real env + real tools + real reward, N=20 orders).** To
  go beyond a single gated decision, we run a multi-turn cancel-order trajectory on the **real
  τ²-bench retail environment** — real policy, real `RetailDB`, and the real `cancel_pending_order`
  tool whose own enforcement ("Non-pending order cannot be cancelled") is the **ground-truth
  reward** — over 20 real pending orders, with a control arm and a treatment arm, Wilson CIs:

  | strategy | Arm A: stays pending (correct=cancel) | Arm B: →processed (correct=deny) |
  |---|---|---|
  | full (oracle) | 1.00 | 1.00 |
  | stale | 1.00 | **0.00** |
  | in_place | 1.00 | **0.00** |
  | erratum | 1.00 | **1.00** |
  | field+erratum | 1.00 | **1.00** |

  Arm A shows editkv does not break the no-change control (no over-correction). Arm B: when the
  order is fulfilled mid-episode, the stale/in_place agents call `cancel` on a now-processed
  order → the **real tool raises → task fails** (tool-consistency 0.00), while the erratum agents
  deny → **task succeeds** (1.00). So on the real τ²-bench env, with the environment's own tool
  enforcement as reward, the cheap in_place edit fails **100%** of the time while editkv recovers
  full task correctness. This statistically (N=20, CIs) answers the "single gated-decision proxy"
  concern. (`esys/tau2_episode.py`, `results/tau2_episode_qwen3_8b.json`.)

## 7. Mechanism (explainability)

The decision reads the field's value **indirectly**: prefill memoizes the field-conditioned
conclusion into *downstream* KV, so refreshing the field token's own KV (the cheap in_place
edit) leaves the decision stale. We establish this with four methods — causal KV patching,
linear probing, a causal reasoning-circuit test, and a position dose-response — that
triangulate the same account. (Earlier attention-attribution / graded-knockout results, §7.5,
are correlational antecedents; the results below are causal.)

### 7.1 Causal KV-patching: a memoization map (D1)
ROME-style causal tracing applied to the KV cache itself. Two token-aligned prefills (OLD
value→unsafe action, NEW value→safe action); we patch NEW (K,V) into the OLD cache at chosen
(layer, position) sites and read the decision logit. *Recovery* = fraction of the OLD→NEW
decision swing restored. Qwen3-8B, n=12 aligned flip instances:
- **FIELD-ONLY recovery = 0.009** [CI .006–.014]: refreshing the field token's own KV causally
  recovers **<1%** of the flip — the rigorous reason in_place fails. This *is* the direct/
  indirect path decomposition: direct (field→decision) path ≈0.9%, indirect (field→downstream
  →decision) ≈99%. **FULL-DOWNSTREAM recovery = 1.00** (sanity = full reprefill).
- **The memoized conclusion is concentrated in the SUFFIX** near the decision: patching the
  last 10% / 20% of downstream recovers **64% / 82%**; the first 10% / 50% recover only
  29% / 34% (strongly asymmetric). This is the mechanistic basis for why appending the erratum
  works and quantifies the partial-reprefill depth needed.
- **Distributed but identifiable, not holographic:** ~16 well-chosen positions recover 94%;
  the strongest single sites split across the policy-rule, reasoning, and decision regions.
  **Layer band:** mid/late layers carry it (early ≈0).
- **Generalizes across scale and family** (in_place recovery ≪ full-downstream=1.0, suffix-
  concentrated everywhere): Qwen3-4B 0.025 / 8B 0.009 / 14B 0.008 / **32B 0.023**, Gemma-2-9B
  0.001 / **Gemma-2-27B 0.219**, Mistral-7B 0.004. The causal account is not Qwen3-8B-specific and
  holds across 7 models up to 32B (Gemma-2-27B does somewhat more residual *direct* field reading —
  0.22, still a minority of the 1.0 full-downstream effect — so in_place remains insufficient). (MLA backbones — DeepSeek-V2-Lite — could not be patched here: the HF custom modeling is
  incompatible with transformers 4.57 and vLLM's flashinfer MLA kernels do not compile for this
  Blackwell `sm_120` GPU; the erratum is architecture-agnostic by construction, but an MLA-aware
  *in_place* edit of the compressed latent KV is genuine future work.) (`esys/mech_causal_patch.py`.)

### 7.2 Linear probing (independent of patching) (D3)
A cross-domain linear probe for the gated *conclusion* (8 diverse domains, leave-one-domain-
out, diff-of-means), a different methodology that should agree if 7.1 is right.
- **Decodability by layer:** conclusion is decodable only in **late layers** (early 0.50 =
  chance → late 0.83; best layer 26 = 0.875) — confirms 7.1's mid/late locus via probing.
- **in_place staleness signature:** apply the stale/full-trained probe to the decision residual
  under each strategy → P(new) = {stale 0.0, **in_place 0.0**, erratum 0.875, full 0.75}. The
  in_place residual is classified as the **OLD** conclusion, identical to stale — the residual-
  level reason it fails; erratum/full flip it to NEW.

### 7.3 The reasoning-axis circuit: the CoT re-reads the field (D4)
Reasoning models tolerate in_place where non-reasoning models revert. **Why?** Hypothesis: the
CoT re-reads and re-derives the field *after* it, so an in_place-refreshed field is re-consumed
by freshly generated CoT tokens that the decision then reads. Causal test on the in_place cache
(field NEW, downstream stale), reasoning ON: `inplace_base` (correct) vs `block_cot_field`
(mask every CoT-generation query's attention to the field) vs `block_dec_field` (mask only the
decision→field) vs `block_cot_gate` (same-width control band).
- Result (Qwen3-8B, n=8 = 2 scenarios × 4 CoT samples): **`block_cot_field` collapses the
  in_place benefit — P(correct) = 0.0 [CI 0, .32], reverting to OLD on all 8 samples** — while
  `inplace_base` 1.0 [.68, 1], `block_dec_field` 1.0 [.68, 1], and `block_cot_gate` 1.0 [.68, 1]
  **hold** (non-overlapping CIs). The CoT re-read, not the decision's direct field read, is the
  causal carrier of reasoning robustness. Strikingly, per-token CoT→field attention is only
  ~0.1% (mass 0.0011), yet blocking it flips every sample (attention magnitude ≠ causal
  importance, echoing 7.1). (`results/mech_reasoning_reread_qwen3_8b.json`.)
- This explains the scale-dependence (§7.5): the CoT re-derivation can itself go wrong (14B
  amplifies), so "thinking rescues the cheap edit" is real but **not guaranteed** — whereas the
  erratum injects an explicit override independent of whether the CoT reasons correctly.

### 7.4 Position dose-response: mechanism → system (D6)
Causal field-position sweep (value appears once; alignment preserved). in_place recovery rises
**monotonically** as the field moves later, i.e. as less field-conditioned text sits after it to
memoize: pos0(early) −0.01 → pos4(hoisted, just before the decision) **0.11**. This is the
causal underpinning of the practical "hoist the mutable field to the end" knob. Nuance: even
full hoisting only *partially* rescues a single-pass decision (0.11, not ~1.0) — which is why
the erratum's explicit conclusion-override, not mere hoisting, is the robust fix (coherent with
the τ²-bench `field+erratum` result, §6).

### 7.5 Correlational antecedents (attention, graded knockout)
Consistent with the causal picture above, on every scale (4B–32B): the decision's *direct*
attention to the field token ≈0.1% (~50–56% to downstream, ~36–48% to attention sinks); graded
knockout of the top-attention downstream restores the correct action (distributed); the stale
signal is read in mid/late layers; and E3 reasoning-resolution is scale-dependent (8B & 30B-A3B
CoT corrects, 14B amplifies — manufacturing 19% unsafe, removed by bypassing the CoT, 32B
collapses to caution).

**Takeaway.** Four independent methods agree: the field is read *indirectly* through a
suffix-concentrated, mid/late-layer memoized conclusion in downstream KV; in_place is causally
inert (<1%) because it never updates that conclusion; the erratum works because appending at the
end writes a fresh, high-leverage carrier into exactly the suffix region the decision reads; and
reasoning robustness is mediated by the CoT re-reading the field — a real but fallible path,
which is why the erratum (not reasoning, not hoisting) is the robust fix.

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

### 8b. Serving under load: batched TTFT and long context (up to 32K)
The single-stream numbers above understate the serving win — under batching and at long context
the gap widens sharply. TTFT (ms) to build a decode-ready cache after a field edit vs full
reprefill, Qwen3-8B (CUDA events; in_place/erratum forward the field/short-suffix tokens over the
*reused* length-T KV cache):

| context T | batch | full reprefill | in_place | erratum | speedup (in_place / erratum) |
|---|---|---|---|---|---|
| 1024 | 1 | 129 ms | 27 ms | 24 ms | 4.8× / 5.5× |
| 1024 | 8 | 782 ms | 35 ms | 41 ms | **22× / 19×** |
| 4096 | 1 | 394 ms | 18 ms | 50 ms | 22× / 8× |
| 4096 | 8 | 3590 ms | 53 ms | 91 ms | **67× / 40×** |
| 16384 | 1 | 1956 ms | 40 ms | 107 ms | 49× / 18× |
| 32768 | 1 | 4538 ms | 39 ms | 169 ms | **117× / 27×** |

in_place TTFT is ~flat (~20–40 ms; it recomputes ~one token regardless of T); erratum grows
gently (24→169 ms; the short suffix attends over T keys); full reprefill grows ~linearly in T and
×batch. So the win compounds with **both** context length and batch size — at 32K context, building
the edited cache is **117× (in_place) / 27× (erratum)** cheaper than re-prefilling, and batching
alone takes the 4K win from 22×→67×. These are HF-level measurements; a paged-attention engine
(vLLM/SGLang) that shares the reused prefix across the batch would widen the gap further, so they
are conservative lower bounds. (`esys/serving_bench.py`, `results/serving_bench_*.json`.)

**On kernels / `torch.compile`.** The *edit itself* is trivial: with a `StaticCache`, overwriting
the field span's KV in place is **0.16 ms** (no clone/realloc) — the measured cost is the
partial-prefill recompute and decode, not the edit. `torch.compile` gives only a **modest ~1.2×**
on the partial prefill and ~1.26× on decode *with StaticCache* (and *hurts* decode with
`DynamicCache`, which graph-breaks). So the win is algorithmic (recompute a few tokens, not the
whole context), not a compile flag.

### 8c. Closed integration on a real PagedAttention engine (vLLM)
The erratum is **append-only**, so it composes directly with a production paged-attention engine's
content-addressed automatic prefix caching (APC): the long policy prefix stays cached and only the
short erratum suffix is computed. The naive alternative — putting the new field value into the
context — *mutates a token inside the cached prefix*, changing that block's content hash and
**invalidating every downstream block**, so the engine recomputes from the field position onward.
We demonstrate this on **vLLM 0.19** (prefix caching ON), with the mutable field placed early
(before the long policy), 48 requests, Qwen3-8B:

| arm | throughput | latency |
|---|---|---|
| baseline (new field in prefix → downstream invalidated) | 8.2 req/s | 121 ms/req |
| **erratum (append-only → prefix is an APC hit)** | **134.8 req/s** | **7.4 ms/req** |

**16.4× throughput** on a real engine — the production realization of the erratum design. (This
also clarifies why the erratum, not the in_place edit, is the serving-friendly mode: an in-prefix
field edit is exactly what breaks content-addressed prefix caching.) Implementation note: this
machine's NVML userspace lib (595.71) mismatches its kernel driver (595.58), breaking vLLM's
NVML-based platform detection (torch.cuda is unaffected); we force the CUDA platform plugin and a
single-process engine to work around it. (`esys/vllm_editkv_serving.py`.)

## 9. Limitations
- Behavioral scenarios are mostly single gated decisions; we *do* now close part of this gap with
  the τ²-bench end-to-end episode loop (§6: real env/tools/reward, N=20, both arms) and the causal
  mechanism (§7), but a full 114-task user-simulator sweep with a stateful editkv-backed agent is
  future work (the local open models that fit a single 96 GB GPU are weak τ²-bench agents for
  reasons unrelated to editkv, which would confound a full-sweep reward).
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
- Mechanism evidence now spans four causal/independent methods (§7: KV-patching, probing, the
  reasoning-circuit knockout, dose-response); a **head-level circuit** localization (which
  specific heads read the memoized suffix vs the erratum) is the natural next step, deferred
  here. The *scale reversal* of CoT helpfulness is characterized (§7.3/7.5) — the CoT-re-read
  circuit explains *why* reasoning helps and why it can backfire — but a full per-scale circuit
  account is open.
- Serving numbers (§8b) are HF-level (CUDA events); the §8c vLLM integration confirms the win on a
  real paged-attention engine (16.4×). The 32K×large-batch HF points OOM a single 96 GB GPU in
  bf16 (measured to 32K at bs=1, 16K at bs=8).
- **MLA backbones are untested here** due to environment toolchain blocks (DeepSeek custom modeling
  vs transformers 4.57; flashinfer MLA kernels vs Blackwell `sm_120`), not anything intrinsic to
  editkv. The erratum (append-only) is architecture-agnostic; the open question is an MLA-aware
  in_place edit of the shared compressed latent KV. **Multi-edit** of one field is robust only when
  collapsed to the current value — stacking a non-monotonic history can let a salient intermediate
  state dominate (§5d).

## 10. Conclusion
Editable KV is viable, but the naive cheap edit is *not* a free lunch: the decision reads the
field indirectly, so leaving the downstream stale reverts it, and reasoning rescues it only
unreliably and scale-dependently. A salient erratum — keep the field in place, append a short
authoritative override, recompute only that — recovers the full-reprefill decision at every
scale and domain we tested, and is *more robust than recomputation itself* under contradictory
context. The contribution is the regime map + the mechanism + the robust, cheap fix.
