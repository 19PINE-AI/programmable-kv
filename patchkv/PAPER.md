# The Editable and Composable KV Cache: Editing Mutable Fields and Transplanting Precompiled Skills in Agentic Contexts

*Thesis: information in the transformer KV cache is **localized, position-portable, and
context-robust** — robust enough to surgically **edit** one region (mutable fields) and to transplant
**another** (precompiled skills) without recomputation. The two operations rest on the same substrate
property; we establish it mechanistically and exploit it both ways.*

*Conference-style consolidation of the experimental program. All numbers are from local runs on
1× RTX PRO 6000 (Blackwell, 96 GB): Qwen3 0.6–32B plus cross-family/architecture checks. Detailed
per-experiment logs and the full result set are in `PAPER_detailed.md` and `results/`; figures in
`figures/`; the production library in `editkv/`.*

---

## Abstract

Prefix caching forces an inference-layer constraint into the application layer: to keep cache hits,
programmers must hoist every *mutable field* (time, ids, user/account state) to the end of the
prompt, even when it belongs elsewhere. We ask whether a field can be edited *in place* in the KV
cache instead, and answer it across behavior, mechanism, architecture, and systems. (1) The region
before the field is provably reusable (KV deviation 0.0). (2) Naively refreshing only the field's KV
and leaving the rest stale **fails** — the decision reverts to the old value — because the decision
reads the field almost entirely *indirectly*, through downstream tokens that **memoized the
field-conditioned conclusion at prefill time**. We establish this causally with four independent
methods that triangulate the same account: KV-patching (the field's own KV causally drives **<1%** of
the decision; the memoized conclusion is **suffix-concentrated** in mid/late layers), linear probing,
a reasoning-circuit knockout, and a position dose-response. (3) **The cheap surgical edit alone
suffices for *reasoning* models** — at 8B it recovers the oracle decision **0.94** of the time at ~1%
recompute with no further help — but **never without reasoning** (0.00 at every scale) and only
partially at 14B/32B; we explain the scale-reversal mechanistically. (4) A **salient erratum** (append
"[STATE UPDATE] <field>→<new>; overrides any earlier value and conclusion") and in particular
**`field+erratum`** matches the strong *hoist-to-end* baseline's oracle correctness **without rewriting
the prompt** (the bare erratum reaches it too in standard settings but is more template-sensitive in
harsher ones); a rigorous baseline comparison shows there is no single dominant method but a
*frontier* — hoist is cheapest but needs prompt surgery, in-place editing matches it without surgery,
and `in_place` is ~free under reasoning. (5) editkv is an **attention-architecture** method: it works on full/GQA/MLA
and hybrid attention+SSM backbones but is weaker on a pure SSM (whose recurrent state has no
look-back; CoT partially rescues it). (6) On the real τ²-bench retail environment — single decisions
(N=20) and a multi-turn autonomous-agent loop (N=30) with the env's own tool enforcement as reward —
editkv preserves task success at a fraction of the recompute where the stale agent collapses. We
release a production library and a closed vLLM integration (the append-only erratum composes with
prefix caching for **16×** higher throughput).

**Composable KV (second axis).** The same substrate property that makes editing possible also makes
*transplantation* possible. We precompile a SKILL (a long, reusable policy/tool spec) once, then
RoPE-reposition and splice its KV into a new context with **no recompute**. (7) The transplanted skill
is behaviorally **indistinguishable from full recompute** — 100% decision agreement and logit
cosine-similarity **0.96–0.999 across seven models** (Qwen3-1.7/4/8/14B, Gemma-2-9B, Mistral-7B, Llama-3.1-8B; the keystone also holds on DeepSeek-R1-Llama-8B and Qwen3-32B-FP8); on the
competent models it *preserves correct skill-following* across **8 diverse domains × 3 families (24/24,
cos 0.98–0.999)**, and **16/16 under reasoning (CoT)**. (8) It is **context-robust**: a skill
precompiled in isolation matches one that attended to the real context, because the decision re-derives
from context it can still see; the only residual error is a **seam at the chunk's start**, which
selective boundary recompute repairs. (9) TTFT scales O(L) vs full reprefill's O(L²): **up to 13.9×
faster at 32k skill tokens** (3× at 2k, 9.8× at 8k on 8B), and a **skill library** composes (N=1–4
skills, decisions preserved). (10) **Keystone — both operations on one cache:** editing a field
*inside a transplanted skill* reproduces the editable mechanism verbatim (in_place weak, selective
recovers, erratum strongest; composed ≈ recomputed), showing edit and compose act on a single
substrate. We position this against Prompt Cache / CacheBlend / EPIC: our contribution is the
**instruction-following-correctness** lens and the **mechanistic unification** with editing.
Transplantation generalizes across **content type** (rules *and* facts/RAG), **insertion point**
(system-area *and* end-of-trajectory tool-results), and **actual agentic tool-calling** (function calls
preserved 6/6 on Mistral/Llama-3.1/Qwen3-8B; degrades on Gemma-2-9B — model-dependent); and the **full
substrate (edit + transplant + keystone) is validated end-to-end on three families** (Gemma-2-9B,
Mistral-7B, Llama-3.1-8B), with `field+selective` an unreliable-but-sometimes-effective tool (works on
Gemma-2-9B/Qwen3-4B/Llama-3.1).

---

## 1. Introduction

Agents re-read long, mostly-static instructions every turn; KV caching reuses the prefill across
turns, but only across an *exact* shared prefix. A single changed token — a clock tick, a session id,
an account-status flip — invalidates the entire suffix. The de-facto fix, hoisting all mutable content
to the prompt's end, taxes programmability: fields referenced in multiple places, nested sub-agent
prompts, and dynamically assembled prompts cannot all be cleanly hoisted, and it forces the
application to pre-identify every mutable field.

We study **in-place field editing**: when a field changes, can we surgically update the cache and
reuse the rest? Our contributions:
1. **A regime map** of when the cheap edit (refresh-field-only, leave-rest-stale) is safe, and its
   **reasoning-vs-non-reasoning** and **scale** dependence (§4).
2. **A causal mechanistic account** — *attention-mediated memoized inference* — established by four
   independent methods (§5), generalized across 7 models and validated against architecture (§7).
3. **A robust in-place fix** (the erratum / field+erratum) and a **rigorous baseline frontier** that
   honestly positions it against hoist-to-end and prior selective-recompute work (§6).
4. **End-to-end evidence** on the real τ²-bench env, single-decision and multi-turn agentic (§8), a
   **cost/latency frontier and a closed vLLM integration** (§9), and a **production library** with a
   per-edit diagnostic (§6.3).
5. **The composable axis** (§10): precompile a SKILL's KV once and transplant it (RoPE-reposition +
   splice) — behaviorally lossless across 8 families and 8 diverse domains, **13.9× lower TTFT**, with a
   seam-repair knob and a skill library — positioned honestly against Prompt Cache/CacheBlend/EPIC.
6. **The unification** (§10.6–10.7): a **keystone** experiment editing a field *inside* a transplanted
   skill (editable mechanism preserved; composed ≈ recomputed across families) and a **unified
   `edit()`+`compose()` API**, establishing that both operations act on one substrate whose information
   is localized, position-portable, and context-robust.

## 2. Related work

Prefix caching (vLLM APC, SGLang RadixAttention) reuses exact prefixes. Selective-recompute methods
for *composition* (CacheBlend, EPIC/AttnLink) recompute ~15% / boundary tokens to restore
cross-attention when assembling *independent* chunks; selection methods (InfoFlow KV, KVShare) decide
*which* tokens to recompute. All target chunk composition or cross-request sharing and recompute the
affected downstream. We study a *temporal edit of one already-jointly-encoded context*, and ask the
opposite question — when can the downstream be left **stale**? We also show prior work implicitly
assumes **single-pass (instruction) decoding**; reasoning models change the picture, and the robust
fix is a salience injection, not recomputation. For our **composable** axis (§10), Prompt Cache (Gim
et al., MLSys 2024) precomputes attention states for reusable prompt modules with position placeholders
and splices them; CacheBlend/EPIC handle the boundary recompute. We do not claim a new caching system
there: our additions are an **instruction-following-correctness** evaluation (does the transplanted
skill still govern the decision?) and the **mechanistic unification** with editing — both editing a
field and transplanting a skill are operations on one substrate whose information is localized,
position-portable, and context-robust. Our mechanism analysis adapts causal tracing
(ROME/MEMIT) and circuit knockout (IOI) to the *KV cache* — the object an editor manipulates. We
compare against CacheBlend directly (§6).

## 3. Method

**Setup.** An OLD context is cached; a field flips OLD→NEW; we compare cache-construction strategies
for the next decode. *full_reprefill* (recompute all), *stale* (reuse all), *hoist_to_end* (field
moved to the end — the baseline to beat), *in_place* (overwrite only the field span's KV, exact, ~1%),
*erratum* (leave stale; append the override; recompute that span ~5–15%), *field+erratum* (both),
and *CacheBlend@k* (recompute field + top-k% KV-deviation downstream). **Metrics:** P(correct/safe
decision), agreement with the oracle, recompute fraction, wall-clock; proportions with Wilson or
bootstrap (B=10000) 95% CIs. **Characterization (E1/E2):** KV of every token *before* the field is
bit-identical OLD vs NEW (deviation 0.0 — the prefix is free); low-conditioning fields (time/ids) are
leave-stale-safe with zero refresh; gating fields (role/status/tier) flip the decision and a bare
field-only edit does not recover it without help — which the rest of the paper explains and fixes.

## 4. The regime map: when the cheap edit works

**Reasoning is the axis.** Within-model ablation (Qwen3-8B, `enable_thinking` on/off, account_role,
n=12 / 36 samples): field-only P(safe) is **0.00 [0,.24] non-reasoning vs 1.00 [.90,1] reasoning**;
under a *poisoned* context (a stale self-conclusion asserting the old value) field-only is fooled even
with reasoning (0.42 unsafe) while the erratum holds (0 unsafe) — thinking is *necessary but not
sufficient*.

**When the *surgical* edit alone suffices — no erratum (the cheap win; Fig. `fig_surgical_suffices`).**
P(in_place-only == oracle), no erratum (`esys/surgical_suffices.py`):

| Qwen3 | non-reasoning | reasoning |
|---|---|---|
| 8B | 0.00 [0,.39] | **0.94 [.74,.99]** |
| 14B | 0.00 | 0.33 |
| 32B | 0.00 | 0.50 |

Without reasoning the surgical edit **never** suffices (it reverts to the stale downstream, §5). With
reasoning it can — at 8B the bare ~1% edit recovers the oracle decision **0.94** of the time with no
erratum (the CoT re-reads the refreshed field, §5.3) — but it is **scale-dependent**: at 14B/32B the
larger CoT is less reliable (oracle is 1.0, so these are staleness failures, not competence). So "just
do the surgical edit" is a real cheap win for the reasoning models that dominate agent deployments,
strongest at the agent-scale 8B, but not universal — which is why the erratum and a per-edit
diagnostic exist.

## 5. Why it fails: the mechanism (four causal methods)

The decision reads the field's value **indirectly**: prefill memoizes the field-conditioned conclusion
into *downstream* KV, so refreshing the field token's own KV leaves the decision stale. Four
independent methods triangulate this (Fig. `fig_memoization_map`, `fig_dose_response`).

**5.1 Causal KV-patching → a memoization map (D1).** ROME-style causal tracing on the KV cache: patch
NEW (K,V) into the OLD cache at chosen (layer, position) sites and read the decision logit; *recovery*
= fraction of the OLD→NEW swing restored. Qwen3-8B, n=12: **field-only recovery = 0.009** [.006,.014]
— refreshing the field token's own KV recovers **<1%** of the flip (this *is* the direct/indirect path
split: direct ≈0.9%, indirect ≈99%); full-downstream = 1.00. The memoized conclusion is **suffix-
concentrated**: patching the last 10%/20% of downstream recovers **64%/82%**, the first 10%/50% only
29%/34%. Mid/late layers carry it; ~16 well-chosen positions recover 94% (distributed but
identifiable). **Generalizes across 9 models** (field-only recovery ≪ full=1.0): **Llama-3.1-8B −0.028**, Qwen3-4B/8B/14B/32B
0.025/0.009/0.008/0.023, Gemma-2-9B/27B 0.001/0.219, **Gemma-3-27B −0.003**, Mistral-7B 0.004
(Fig. `fig_d1_generalization`; Gemma-3 loaded text-only — `Gemma3ForCausalLM`, vision tower stripped —
in bf16, full-downstream=1.0, suffix@10%=0.85).
**And across 8 *natural* diverse-domain tasks** (retail/airline/devops/banking/access/clinical/
customs/oncall, chat template, `esys/mech_causal_natural.py`): field-only recovery **0.003 [−0.0,
.007]**, full-downstream 1.0, suffix-concentrated (suffix@10%=0.71 vs prefix@50%=0.46) — so the
memoization map is not an artifact of the templated scenarios; the battery spans 12 templated + 8
natural instances.

**5.2 Linear probing, independent of patching (D3).** A cross-domain probe (8 domains, leave-one-
domain-out) for the gated *conclusion* is decodable only in **late layers** (early 0.50 = chance →
0.875), and the **in_place decision residual is classified as the OLD conclusion**, identical to
stale (P(new) = stale 0.0 / in_place 0.0 / erratum 0.88 / full 0.75) — the residual-level reason
in_place fails.

**5.3 The reasoning-axis circuit (D4).** On the in_place cache (field NEW, downstream stale),
reasoning ON: blocking the CoT's attention to the field collapses the in_place benefit (8B: 1.0→0.0,
n=8, non-overlapping CIs) while blocking the *decision's* direct field read does not — **the CoT
re-read, not the decision's direct read, is the causal carrier** of reasoning robustness (per-token
CoT→field attention is only ~0.1%, yet causally decisive). **This also explains the scale-reversal** (14B,
n=18): `inplace_base` recovers only **0.11** [.03,.33] (vs 8B's 1.0) — the CoT barely re-derives; the
14B CoT attends *less* to the field than 8B (0.0006 vs 0.0011); and masking the stale gate/downstream
region roughly **doubles** recovery (0.11→0.22) — i.e. the larger CoT **defers to the stickier memoized
stale conclusion** in the downstream rather than re-deriving from the refreshed field. (Directional:
the gate-masking CIs overlap; see §10.) The scale curve is non-monotonic — in_place-under-reasoning
recovery is 1.0 (8B) → 0.11–0.33 (14B) → 0.50 (32B) — sharpest reversal at 14B. The 32B *knockout*
circuit (run on the official FP8 8-bit checkpoint to fit alongside the shared training job; n=4)
confirms the mechanism is the same at scale: `inplace_base`=0.5 [.15,.85] and **`block_cot_field`
collapses it to 0** — i.e. *when* the larger CoT recovers it is still the CoT re-read that carries it;
the larger model simply defers to the stickier memoized stale conclusion more often (CoT→field
attention 0.0007 vs 8B's 0.0011).

**5.4 Position dose-response (D6).** Sweeping the field's position (value appears once), in_place
recovery rises **monotonically** as the field moves later — pos0 −0.01 → hoisted 0.11 — the causal
underpinning of "hoist to end." Even full hoisting only *partially* rescues a single-pass decision,
which is why the erratum's explicit conclusion-override, not mere hoisting, is the in-place fix.

**Takeaway.** The field is read indirectly through a suffix-concentrated, mid/late-layer memoized
conclusion in downstream KV; in_place is causally inert (<1%) because it never updates that
conclusion; the erratum works because appending at the end writes a fresh carrier into exactly the
suffix region the decision reads; reasoning robustness is mediated by the CoT re-reading the field — a
real but fallible path (it backfires at scale), which is why a recomputation-free salience injection is
the robust fix.

**5.5 An analytical model (Fig. `fig_toy_model`).** The phenomenon follows from the memoization
*structure* alone, with no training. Model the decision token's readout as one attention head over the
cached tokens, with a 1-D signed value channel (field=old ↦ +1, new ↦ −1; decision = sign):
$y(D)=\sum_t \alpha_t v_t$, $\alpha=\mathrm{softmax}(\text{scores})$. Encode the empirical structure: the
field is the *oldest* token, $m$ downstream "conclusion" tokens memoized the field value at prefill
($v_{C_i}=+1$), recency-weighted ($\alpha_t\propto e^{\gamma\,\text{pos}_t}$), and sinks carry value 0.
With $m=40$, $\gamma=0.14$, sink mass 0.40 this gives $\alpha_{\text{field}}{=}0.000$,
$\alpha_{\text{down}}{=}0.60$ — the empirical 0.1% / 50% / 36% split. Then, in closed form:
- **`stale` and `in_place` do not flip.** $y_{\text{stale}}=+0.60$; refreshing only the field gives
  $y_{\text{in\_place}}=+0.599$ — the field-only recovery equals $\alpha_{\text{field}}/(1{-}\text{sink})
  =0.0005$ (cf. empirical 0.009). The decision reads the *memoized* $\sum\alpha_{C_i}v_{C_i}$, which the
  edit never touches.
- **Recovery is suffix-concentrated.** Patching the last $k$ conclusions to NEW recovers
  $\sum_{\text{recent }k}\alpha_{C_i}/\alpha_{\text{down}}$; recency weighting makes suffix@10%=0.43 ≫
  prefix@50%=0.06 (cf. Fig. `fig_memoization_map`).
- **Dose-response.** As $m$ shrinks (field placed later), $\alpha_{\text{field}}$ grows, so in_place
  recovery rises 0.0→0.008→0.04→0.15→0.47→**1.0** for $m=40,20,10,4,1,0$ — recovering fully only when
  the field is hoisted to the end ($m{=}0$), matching D6.
- **The erratum has a salience threshold.** Appending an override token of value NEW with salience $s$
  (the decision weights an explicit "[STATE UPDATE] overrides any earlier conclusion" disproportionately)
  gives $y_{\text{err}}=(y_{\text{stale}}+s\beta\,\text{NEW})/(1+s\beta)$, which **flips iff $s\beta$
  exceeds the memoized-positive mass** — here $s^\*\!\approx\!4.6\times$ a normal recent token. This
  *derives* three observations: a bare/weak update is sub-threshold (the §5d/§6 template-sensitivity);
  the explicit override phrasing matters (it raises $s$); and a **pure SSM has no attention to place
  weight on the override at all** ($\beta\to0$), so the erratum fails there (§7).

The model thus reproduces all four mechanism findings analytically and shows they are forced by the
memoization structure, not by anything Qwen-specific. (`esys/toy_model.py`.)

## 6. The robust fix and the baseline frontier

**6.1 The erratum and field+erratum.** Leave the cache stale, append "[STATE UPDATE] <field>→<new>;
overrides any earlier value AND conclusion", recompute only that span. The robust **field+erratum**
also refreshes the field token. Over 8 domains, 5 families, 0.6–32B, both modes: for every competent
model the cheap field-only edit carries a penalty (0.12–0.67 oracle-controlled, up to 29% unsafe
actions) and field+erratum recovers to the oracle ceiling; the erratum is poison-robust *even where a
full reprefill is fooled*; over-correction on an irrelevant field is 0/8; multi-field edits compose
without interference.

**6.2 The baseline frontier — answering "why not just hoist?" (Fig. `fig_baseline_frontier`).**
8 gating tasks, non-reasoning (the regime where strategies differ), deployment-realistic chat
template, P(correct) × recompute, with a poisoned-context column (`esys/baseline_table.py`, Qwen3-8B):

| method | P(correct) | recompute | poison | needs prompt rewrite? |
|---|---|---|---|---|
| full reprefill | 1.00 | 100% | 1.00 | no |
| stale / in_place | 0.00 / 0.00 | 0% / 0.6% | — | no |
| CacheBlend @15% (prior work) | 0.12 | 15% | — | no |
| **hoist-to-end** | **1.00** | **5.2%** | 1.00 | **yes** |
| erratum (stale + update) | 1.00 | 12% | 1.00 | no |
| **field+erratum** | **1.00** | 13% | 1.00 | no |

This is the answer: hoist-to-end, erratum, and field+erratum **all reach oracle correctness** at low
cost, while in_place fails (0.00) and CacheBlend underperforms (0.12). The differentiators are *cost*
and *programmability*: **hoist is cheapest (5%) but requires rewriting the prompt to move every mutable
field** — it does not compose across multiple fields or fields the rules reference in place (the
pathology this paper is about); **erratum / field+erratum match its correctness *in place*, with no
rewrite** (12–13%). **CacheBlend's KV-deviation selection underperforms (0.12)** because the
decision-relevant content is suffix-concentrated, not in the highest-deviation tokens (§5.1). And
`in_place` is **~free and sufficient under *reasoning*** (0.94, §4). So there is no single dominant
method — a *frontier* — and editkv contributes the mechanistic map of it plus the two in-place options
(free in_place for reasoning; hoist-matching erratum/field+erratum without prompt surgery). *Caveat:*
on this clean template the poison column does not differentiate (all 1.00); the erratum's
poison-robustness advantage over a fooled full-reprefill appears only under stronger adversarial
contexts (§5, the within-model poison ablation), and the bare erratum is template-sensitive in harsher
non-standard prompts (where field+erratum is the safer default).

**6.3 A per-edit diagnostic.** `needs_erratum` predicts, for a specific edit, whether the cheap
in_place suffices or must escalate to field+erratum, by decoding the next decision token under each;
validated (8 high + 8 low/irrelevant edits) with a confidence-margin knob trading precision/recall:
**P=1.00 @ margin 0.5 (zero over-correction) → R=0.86 @ margin 0** (a false negative is a stale
decision; a false positive costs a few % — so the recall-oriented low margin is the safe default).

**6.4 An erratum-free alternative: profile-guided selective recompute (Fig. `fig_selective_recompute`).**
Instead of appending the erratum, can we recompute the KV of *only the few downstream tokens the
decision needs* and leave the rest stale? The §5.1 map says yes in principle (~16 positions recover
94%), but the *selection criterion* is everything. On Qwen3-1.7B (`esys/selective_recompute.py`),
recovery vs the number of recomputed downstream tokens k, by ranking criterion:

| k | KV-change (CacheBlend) | hidden-change | **decision-attention** | decision-recovery (oracle) | suffix |
|---|---|---|---|---|---|
| 8 | 0.18 | 0.18 | **0.55** | 0.85 | 0.46 |
| 16 | 0.17 | 0.21 | **0.75** | 0.87 | 0.49 |
| 32 | 0.25 | 0.25 | **0.94** | 0.94 | 0.59 |
| 64 | 0.89 | 0.90 | 0.97 | 1.0 | 0.60 |

Three findings. (i) **Change-based rankings fail**: the tokens whose KV/hidden state — or *attention
distribution* — changes *most* under the edit are *not* the tokens the decision depends on. KV-change
and hidden-change need ~64 tokens (24% of downstream) to recover (the mechanistic reason CacheBlend
underperforms, §6.2); the *attention-difference* criterion (rank by how much each token's attention
changes) is even worse — `field+attention-difference@32` recovers **0.00** non-reasoning vs
`field+decision-attention@32` 0.50 — confirming that "most-changed" ≠ "decision-relevant". (ii) **Ranking by *decision-attention* — which downstream tokens the decision token attends to
— nearly matches the expensive decision-recovery oracle** (0.94 @ k=32, ~12% of downstream) and is
*cheap* (one forward). (iii) **It is offline-profilable and value-independent**: decision-attention is
computed on the *base/old* context, so the recorded position-set transfers *exactly* to a different
new value (held-out value: **0.91 @ k=32**, identical to that value's own set) — validating
"profile the affected set once, reuse in production." So selective recompute is a viable erratum-free
mode (recompute a fixed ~12% by decision-attention, no prompt change) — comparable in cost to the
erratum (~5–15%) and useful where appending text is undesirable; the erratum remains simpler
(append-only, no profiling, composes with prefix caching, slightly higher recovery). Across 7 models
× 8 benchmarks, selective@64 (decision-attention) reaches the golden erratum (P(correct)=1.0 on every
model; the erratum hits 1.0 with only ~30 appended tokens). **The recomputed set must include the field token.** Selective recompute of the downstream tokens
*alone* (omitting the field) works non-reasoning but **fails under reasoning** (Qwen3-8B + CoT: 0.00),
because the CoT re-reads the *stale field* (the field has ~0.1% decision-attention, so it is never in
the top-k). **Including the field fixes it**: `field+selective@32` = **1.00 under reasoning**,
matching the golden erratum, and beats field-only in_place (0.75→1.00) — the field is *necessary*
(the CoT re-reads it) and the selective downstream pushes recovery to full. So the correct recipe is
"always refresh the field + the top-k most-affected downstream tokens," and it then works in both
modes. The erratum remains the golden, mode-universal method (append-only, no field-position
tracking, prefix-cache friendly); selective recompute is the in-place, no-appended-text alternative.

**How small can K be under reasoning? (Fig. `fig_ksweep`, `esys/selective_K_sweep.py`).** We sweep
`field+selective@K` (K extra decision-attention tokens beyond the field) under reasoning across the
Qwen3 family on 3 gating domains × 4 prompts × 8 CoT samples (n=72/model, bootstrap CIs). Two results.
**(a) The erratum is stronger than full reprefill** — P(safe) for the explicit "[STATE UPDATE]…
overrides any earlier conclusion" is **1.00**, above even a full reprefill of the new value (0.92–0.99),
because the override adds instruction force the bare corrected value lacks. So the right recovery
target for selective recompute is the *full-reprefill* upper bound, not the erratum. **(b) The minimal
K is strongly model-dependent**, tracking how *sticky* the stale memoized conclusion is under CoT
(the scale-reversal of §5.3): field-only reasoning recovery is 0.92 (8B) / 0.79 (1.7B) / 0.50 (14B) /
**0.35 (4B)**, giving K\* (to reach full reprefill) ≈ **4 (8B), 8 (1.7B), ~64 (14B), >64 (4B)** — a
16× spread across one model family. On the sticky 4B even 64 recomputed tokens reach only 0.81; 14B
climbs monotonically to 0.97 by K=64. **Takeaway:** field+selective@K is an effective in-place surgical
edit when the field-conditioned conclusion is not sticky (small K suffices), but K is not universal —
it can exceed 64 on models where the CoT defers to the memoized stale conclusion. The erratum needs no
such per-model K because it injects a fresh, emphatic statement rather than dislodging the stale
conclusion token-by-token — which is why it remains the robust, scale- and mode-universal default.
**Non-Qwen generalization (DeepSeek-R1-Distill-Llama-8B):** the gap is starkest off the Qwen3 family —
`field+selective@K` recovers **0.00–0.07 at every K (0–64)** while the erratum recovers **0.98** (full
0.89). DeepSeek-R1's reasoning is the *stickiest* we measured (field-only 0.06): selective recompute
fails entirely, the erratum still works — definitively establishing the erratum as the cross-family
robust method and `field+selective@K` as unreliable. **Stickiness is domain×model dependent, not just
model-dependent:** on the *diverse* gating domains the same DeepSeek-R1 is non-sticky (field-only =
full = 0.79, K\*=0), whereas on the e2 templates it is maximally sticky — so a model's K\* cannot be
quoted without its task distribution, reinforcing the erratum (which needs no K) as the safe default.
(The reasoning K-sweep harness requires a CoT model; non-reasoning models score degenerately in it and
are excluded.)

**What tokens carry the conclusion (`esys/selective_tokens.py`).** Decoding the causally-important
downstream tokens (ranked by per-position recovery) gives a concrete, interpretable picture of *where*
the memoized conclusion lives: it is stored in **the gating rule's conclusion tokens** — the literal
action word (e.g. the token `' escalate'` inside "...do not refund; escalate.") and the punctuation
ending the rule's clauses (`'.'`, `';'`, `','`) — **and in an aggregator token just before the
decision** (the `'\n'` preceding "Decision:", which for some tasks alone carries up to 0.81 of the
flip). Notably the three signals diverge: decision-*attention* over-weights structural scaffolding
near the decision (`'\n\n'`, `'</think>'`, `':'`, `'Decision'`) — high attention, low individual
recovery — so it needs k≈32–64 to include the rule-conclusion tokens; KV-*change* is largest at the
rule region and the field-copies (largest representation change) but most of those are not
decision-relevant (the CacheBlend failure mode); only causal *recovery* isolates the rule-conclusion +
pre-decision aggregator tokens. This is the token-level face of the §5 memoization account.
Quantitatively (`esys/token_stats.py`, 3 models × 8 tasks, per-position recovery mass with bootstrap
CIs): the field carries 0.3–0.5% of recovery, while **delimiter/aggregator tokens (newlines,
punctuation) carry 37–55%** — far above their token frequency — with the remainder in the decision
region (26–60%) and forward-propagated through the filler (32–51%). This is consistent and
complementary with Anthropic's attribution-graph results in *On the Biology of a Large Language Model*
(Lindsey et al., 2025): their poetry case study finds the model stores a *planned* future word on the
end-of-line newline token, so both works show **delimiter tokens acting as aggregation registers** —
they for forward planning, we for backward-looking memoized conditional conclusions.

## 7. Generalization across architectures (Fig. `fig_architecture`)

editkv is an **attention-architecture** method: the surgical edit needs a per-token KV; the erratum
needs attention to "look back" at the override. Behavioral erratum recovery, both modes, 4 scenarios ×
K=8 samples, bootstrap CIs (`esys/arch_erratum_v2.py`):

| backbone | history store | surgical `in_place` | erratum (non-rsn) | erratum (reasoning) |
|---|---|---|---|---|
| full/GQA attention (Qwen3-8B) | per-token KV | ✅ (§4) | (Qwen3 non-think quirk) | 0.97 [.91,1] |
| MLA (DeepSeek-V2-Lite) | compressed latent KV | ⚠ MLA-aware | ✅ verified | ✅ verified |
| hybrid attn+SSM (Falcon-H1) | KV + recurrent state | ⚠ attn layers | 1.00 [1,1] | 0.97 [.91,1] |
| **pure SSM (Falcon-Mamba)** | recurrent state, no KV | ❌ N/A | **0.37 [.20,.53]** | **0.78 [.63,.91]** |

On a pure SSM the erratum is the **weakest and mode-dependent**: in non-reasoning it *fails* (0.37 —
the model commits the earlier conclusion to its recurrent state and cannot attend back), and **CoT
partially rescues it** (→0.78, non-overlapping CIs) because the CoT regenerates tokens after the
override so the state processes the new value last. MLA and hybrid backbones work. DeepSeek-V4 (sparse
attention) and Qwen3-Next (linear+attention) retain attention sublayers → supported class but exceed
this 96 GB box to run.

## 8. End-to-end on the real τ²-bench retail environment

**8.1 Single decision, N=20 orders (real policy/DB/tools; the `cancel_pending_order` tool's own
enforcement is the reward).** When an order is fulfilled mid-episode (pending→processed, correct flips
cancel→deny): stale & in_place agents call cancel on a processed order → the **real tool raises → task
fails** (0.00); erratum & field+erratum **deny → succeed** (1.00); a no-change control stays 1.00 for
all (no over-correction). The cheap in_place edit fails **100%** of the time while editkv recovers full
task correctness.

**8.2 Multi-turn autonomous-agent loop, N=30 (`esys/tau2_agent_loop.py`).** The model emits and
executes a *sequence* of tool calls against the live DB over a multi-turn conversation; the gating
field flips mid-episode; the deployment-realistic append-only **erratum** is compared to full-reprefill
and stale:

| strategy | task success | recompute |
|---|---|---|
| full reprefill | 1.00 | 1.00× |
| erratum (append-only) | 0.70 | 0.55× |
| stale | 0.00 | 0.49× |

Honest, realistic: where the user has *already* requested the action, the append-only erratum overrides
the primed-stale decision 70% of the time at 0.55× recompute, vs stale's total collapse; the saving
grows with conversation length. The gap to full-reprefill is the genuinely hard part of multi-turn (a
primed prior commitment), which a buried-token field+erratum would close (future work).

## 9. Cost, latency, and a closed serving integration

**Cost frontier (Qwen3-8B, CUDA events):** full reprefill scales linearly (78 ms@586 → 1260 ms@9947)
while in_place is ~constant (~30 ms) and the erratum small (27–94 ms) — ~42×/~13× cheaper at 10K. The
*edit itself* is 0.16 ms with a StaticCache; the cost is the partial recompute (`torch.compile` adds
only ~1.2×, so the win is algorithmic). **Serving under load (Fig. `fig_serving`):** TTFT speedup grows
with context and batch — 22×/19× at 1K·bs8, 67×/40× at 4K·bs8, **117×/27× at 32K·bs1**. **Closed vLLM
integration:** the append-only erratum composes with vLLM's content-addressed prefix caching for
**16.4×** throughput (a naive in-prefix field edit invalidates downstream blocks — exactly why the
erratum is the serving-friendly mode). **Online load sweep (Fig. `fig_online_load`):** submitting an
increasing number of concurrent requests to one vLLM engine, the **baseline saturates at ~11 req/s**
(compute-bound — each request full-prefills the long policy) while the **erratum scales to ~178 req/s**
(cache-bound — shared prefix is an APC hit, only the suffix is computed); the throughput advantage
*grows* with offered load — 6× (N=8) → 13× (N=32) → **16× (N=128–512)** — and the baseline's saturation
is exactly the regime where editkv matters most. (Two environment fixes — NVML driver/lib mismatch and
a stale CUDA-11.5 nvcc vs Blackwell sm_120 — were required; see `PAPER_detailed.md` §10.1.)

## 10. The composable axis: precompiled SKILL transplantation

Sections 4–9 are the **edit** axis (change a cached field in place). This section is the **compose**
axis (insert a precompiled chunk), and §10.6 shows the two are one substrate. Agentic prompts are
dominated by long, reusable, loosely-coupled SKILLs/tool-specs (often tens of thousands of tokens);
§9 showed full reprefill is the bottleneck. We precompile a skill's KV **once** and transplant it.

**Relation to prior work.** Precomputing attention states for reusable prompt modules and splicing
them is studied by **Prompt Cache** (Gim et al., MLSys 2024), **CacheBlend** (EuroSys 2025), and
**position-independent caching** (EPIC, etc.). We do **not** claim a new caching system. Our
contributions are (i) an **instruction-following-correctness** evaluation — does the transplanted skill
still *govern the decision* (those works report perplexity/throughput) — and (ii) a **mechanistic
unification** with the editable axis (§10.6).

**10.1 Machinery (`esys/composable_kv.py`).** HF caches *post-RoPE* keys, so transplanting a chunk to
a new position requires re-rotating the keys (un-rotate from source positions, re-rotate to target;
values are position-free). Done in fp32 the round-trip is exact (residual = bf16 cache quantization).

**10.2 Feasibility + generalization (analog of D1's 8-model generalization).** A precompiled skill,
repositioned and spliced, is **behaviorally indistinguishable from full recompute**: 100% decision
agreement across six models, logit cos-sim **0.96 (Qwen3-14B) → 0.999 (Gemma-2-9B, Mistral-7B)**. On
the models competent at the tasks (Gemma-2-9B, Mistral-7B, Llama-3.1-8B) and across **8 diverse skill
domains** (refund, access, deploy, rx, loan, legal, incident, visa) full-recompute is correct **8/8 and
precompiled preserves it 8/8** on all three (cos 0.98–0.996; **24/24** correct overall) —
transplantation costs no correctness across families and domains. Transplant fidelity (reposition==full)
also holds on Qwen3-8B over the 8 domains (8/8, cos 0.979) even where its non-reasoning competence is
low — precompiled tracks full regardless. The same families confirm the **editable** axis too: Llama-3.1-8B's
behavioral erratum recovers the oracle decision **1.0 in both modes** (24/24 discriminating), so a single
model family (Llama-3.1) validates editing (D1 −0.028/full 1.0; erratum 1.0/1.0), transplanting
(feasibility 8/8, scaling 13.6×), and the keystone (7/8) — the unified substrate in one model.

**10.3 Context-staleness (analog of the architecture/robustness studies).** A skill precompiled in
*isolation* (never attended to the real system prompt) matches one that did — on both self-contained
(4/4) and context-coupled (2/2) skills — because the decision token attends to the real context
directly and re-derives. Precompile is robust for loosely-coupled skills; the failure mode (untested)
is a skill that is the *sole carrier* of a context-derived computation.

**10.4 Transplant mechanism + the seam (analog of D1's locality map).** Per-position KV deviation
(transplanted vs native) localizes the error to the chunk's **start** (mean dev first-8 tokens 13.8
vs last-8 7.2): the first tokens most needed the prefix. This is the composable analog of the editable
*suffix-concentration* — and it is exactly the **seam** that boundary recompute targets. **Seam-repair
(`--seam`):** recomputing just the first **K** chunk tokens with the real prefix lifts logit cos-to-full
monotonically — 8B: 0.982 (K=0) → 0.993 (K=2) → 0.996 (K=4+); 1.7B: 0.989 → 0.999 (K=2). So **2–4
boundary tokens repair the residual** (CacheBlend-style, reusing the selective-recompute machinery) —
the few tokens that needed the prefix.

**Correctness under reasoning (C3).** Under CoT, the precompiled skill preserves *correct* behavior, not
just *matching* behavior: on Qwen3-8B and Qwen3-1.7B the full-recompute model follows the skill **16/16**
and the precompiled-transplant model also **16/16**, with **16/16 decision agreement** — transplantation
is lossless for skill-following when the model actually reasons over the skill.

**10.5 TTFT scaling + the library (Fig. `fig_composable_scaling`; analog of §9 serving).** Full reprefill
is O(L²) in skill length; transplant is O(L) re-rotation + small prefill. The TTFT speedup grows with model size and skill length — at 32k skill tokens: **5.5× (1.7B), 12× (DeepSeek-8B), 13.6× (Llama-3.1-8B), 13.9× (Qwen3-8B)**; on 8B: 1.16×@500, 3.0×@2k, 9.8×@8k. A **skill library** composes: stacking N=1–4
precompiled skills preserves the decision (4/4 agreement vs full).

**10.6 Keystone — edit *inside* a transplanted skill (Fig. `fig_keystone`, `esys/compose_edit.py`).** We precompile a skill
with an embedded categorical state field, splice it in, then **surgically edit that field inside the
transplanted chunk**. The editable mechanism carries over verbatim (8B, D1-style recovery; *composed*
vs *recomputed* skill):

| method | recomputed | composed |
|---|---|---|
| in_place | 0.15 | 0.19 |
| field+selective@8 | 0.43 | 0.48 |
| field+selective@32 | 0.55 | 0.59 |
| erratum | 1.52 | 1.79 |

`in_place` is weak (memoization), selective recovers, erratum is strongest — and **composed ≈
recomputed** for every method. Editing transplanted KV behaves identically to editing recomputed KV:
**edit and compose are two operations on one substrate**, which is the unifying thesis. The keystone
**generalizes across families** — composed ≈ recomputed also holds on Qwen3-4B and **DeepSeek-R1-Llama-8B**
(e.g. sel@32 0.61/0.63, erratum 1.06/0.70 recomputed/composed). The sharpest, most rigorous read is
**Llama-3.1-8B over 8 categorical domains (clean flips 7/8)**: in_place **0.054/0.036** (fails),
sel@32 **0.802/0.805** (recovers), erratum **1.114/1.246** (strongest) — recomputed/composed match
within noise at every method. **Gemma-2-9B (7/7 clean flips)** and **Mistral-7B (4/4 flips)** corroborate (Gemma in_place −0.00,
sel@32 0.96/0.90; Mistral in_place 0.04/0.03, sel@8 0.62/0.49, sel@32 0.72/0.56, erratum 0.98/0.93),
as does Qwen3-4B. Across families the picture is
identical: editing a field *inside a transplanted skill* fails in place, recovers selectively, and is
fixed by the erratum — exactly as for a recomputed skill.

**10.7 Unified system: one cache, both operations (`esys/editkv_unified.py`).** A capstone agent turn:
a system prompt + a **library of 3 precompiled skills** (composed by KV splice, no reprefill) + a
session context with a mutable `order_status` field. When the field changes, an in-place **erratum
edit** is applied to the *same* cache. The unified path's decision matches a full reprefill
(agree=True on Gemma-2-9B) while the **edit is 2.9× faster than full reprefill** (47 ms vs 136 ms).
The `EditableComposableCache` object exposes `precompile`/`build` (compose) and the erratum edit on one
substrate — editable and composable in a single API. Composability also holds on a **large quantized
model** (Qwen3-32B-FP8: transplant + TTFT scaling validated, 2.2× at 2k skill tokens).

**10.8 Composable taxonomy: content × insertion point × agentic tool-calling.** §10.1–10.7 transplant
*rules-as-skills* inserted mid-context with a decision metric. Composition has other incarnations; we
test them, all with **N≈100+ and bootstrap 95% CIs**. **(a) Content = facts/RAG**
(`esys/composable_facts.py`, N=104): transplanting a *retrieved passage* and answering a fact question
over it is **preserved on standard-attention models** — Mistral-7B full 0.94 → precompiled **0.95
[0.90,0.99]**, Llama-3.1-8B 0.99 → **1.00 [1.0,1.0]** — but **degrades on Gemma-2-9B** (0.95 →
**0.69 [0.61,0.78]**, non-overlapping CI). **(b) Insertion point**: identical whether the chunk is
spliced in the *system-area* (early) or at the *end of the trajectory as a tool result* (late) — e.g.
Gemma 0.69 at both, Mistral 0.95 at both — so RoPE-repositioned transplant is **insertion-point-agnostic**.
**(c) Agentic *actual tool-calling*** (`esys/composable_agentic.py`, N=108): an agent emits a structured
function call (name+arguments) governed by a transplanted long **tool-definitions** block; we score
*functional* correctness, not a decision. Full is 1.00 on all; the transplant is **functionally
lossless on Mistral-7B and Llama-3.1-8B (1.00 [1.0,1.0], agreement 1.00)** but **degrades sharply on
Gemma-2-9B (0.44 [0.34,0.53])**. **Gemma-2-9B is the consistent outlier on both facts and tool-calling**
— and it is the one model with **alternating sliding-window/global attention**, whose local layers
expect context the isolation-precompiled chunk never saw (a seam-repair / anchored-precompute target).
**This replicates on a second generation: Gemma-3-27B** also collapses under transplant (facts
0.91→**0.02**, agentic 1.0→**0.48**) despite its skill-feasibility (8/8) and keystone (7/7 clean) being
fine — so it is specifically *fine-grained retrieval from the transplanted chunk* that sliding-window
attention breaks, confirmed across both Gemma generations.
**Takeaway:** transplantation generalizes across *content type* (rules and facts) and *insertion point*
(system and tool-result) and preserves *agentic tool-calling* on **standard-attention** models with
tight CIs, but **sliding-window attention (Gemma-2) breaks it** — an architectural caveat, statistically
significant. (Qwen3-8B is degenerate on this non-reasoning QA/tool format — full≈0 — and excluded.)

**10.9 Substrate generalization (two scorecards, `esys/make_scorecards.py`).** To show the unified
substrate is not a one-model artifact, we tabulate, per model, **editable** (D1 field-only≈0 / full=1.0;
behavioral erratum recovery) and **composable** (keystone composed sel@32 / erratum). The full substrate
(D1 + erratum + keystone) is validated **end-to-end on three families — Gemma-2-9B, Mistral-7B, and
Llama-3.1-8B** (D1 field-only 0.00/0.00/−0.03 with full=1.0; erratum 1.0/1.0/1.0; keystone composed
sel@32 0.90/0.56/0.81) — and the individual components additionally reproduce on Qwen3-4B/8B/14B and
Gemma-3-27B (D1 ≈0/1.0; erratum ≈1.0 reasoning). *Every* component reproduces on *many* families; the
substrate is general, not a one-model artifact. The second scorecard isolates **field+selective**: it
is **unreliable but works on some** — composed recovery WORKS on Gemma-2-9B (0.90), Qwen3-4B (0.88),
Llama-3.1-8B (0.81); partial on Qwen3-8B (0.59), Mistral (0.56), DeepSeek (0.44), Qwen3-14B (0.32) —
a useful-when-it-lands tool, never universal, which is why the erratum remains the default.

**10.10 Composable KV for IMAGES (multimodal, `esys/composable_vision.py`).** In an agent trajectory an
image costs a full prefill — the vision tower *plus* prefilling the image's >1k soft-tokens through the
LM. We **cache the image's LM KV once and splice it in**, re-running only text, so later turns skip that
prefill entirely (the image analogue of facts/RAG transplant). Across **N=120 diverse VQA tasks per
model** spanning **perception** (read digit / name colour), **visual reasoning** (count, shape, spatial,
size), and **agentic** (the image governs a *tool* decision — status-light → halt/proceed, gauge →
scale_up/down), with **>1000-token images** (1024–1296 image tokens) and bootstrap CIs, the spliced
image KV is **near-lossless vs full re-encode** (agreement = precompiled==full):

| VL model | img tokens | overall agreement | agentic agree | reasoning agree |
|---|---|---|---|---|
| Qwen2.5-VL-3B | ~1296 | **1.00 [1.0,1.0]** | 1.00 | 1.00 |
| Qwen2.5-VL-7B | ~1296 | **0.958 [.92,.99]** | 1.00 | 0.97 |
| Qwen3-VL-8B | ~1024 | **0.992 [.98,1.0]** | 1.00 | 0.98 |

**Agentic tool-decisions from a transplanted image agree 1.00 on all three** — an agent can reuse a
cached image instead of re-prefilling it, including when the image drives a tool call. (Where the VLM's
*own* accuracy is low — e.g. reading a high-res digit — precompiled still tracks full; agreement, not
absolute accuracy, is the transplant-fidelity metric.) **M-RoPE note:** image position is `(t,h,w)`;
moving an image shifts only the temporal `t` (h,w intrinsic), so a position change re-rotates only the
temporal mrope-section — same-position reuse needs none. **Qwen3-VL-30B-A3B is excluded** (degenerate:
full-accuracy ≈0 on this synthetic VQA format, so agreement is uninformative). So composable KV extends
from text to **vision tokens**: the substrate property (localized, position-portable, context-robust
information) holds for images too.

**10.11 Scale: composable at 27–32B and on MoE.** The transplant holds at large scale (`results/g5.log`):
**Gemma-3-27B** (text) feasibility 8/8 (cos 0.955) + keystone 7/7 clean (composed≈recomputed); **Qwen3-32B-FP8**
feasibility 7/8 (cos 0.91) + agentic tool-calling preserved (0.95→**0.97**); **Qwen3-30B-A3B** — a
*Mixture-of-Experts* (30B total / 3B active) — feasibility 7/8 (cos 0.90), keystone sel@32 0.83/0.80
(composed≈recomputed), agentic 1.0→**0.972** preserved. So composable KV works on **FP8-quantized** and
**MoE** backbones at 30B-class scale, not just dense bf16. (Llama-3.1-70B-FP8 is at the edge of the 96 GB
GPU — the 70 GB FP8 weights plus per-case forwards/cache-clones OOM; a memory-leaner rerun is in
progress. Facts/RAG is degenerate full≈0 on the Qwen3 family in this non-reasoning QA format, as in §10.8.)

## 11. Limitations

The mechanism battery (n=12 instances) is on a few scenario templates; the scale-reversal explanation
(§5.3) is qualitatively clear but small-n at 14B. The τ²-bench evidence is the retail cancel-task
family (real env/tools/reward, single-decision N=20 + multi-turn N=30), not the full 114-task
user-simulator sweep with a strong agent (local models that fit one 96 GB GPU are weak τ²-bench agents,
which would confound absolute reward). Models are ≤32B open weights (no frontier/API scale; 70B needs
4-bit). MLA in_place editing of the compressed latent KV, pure-SSM support, multi-field/long
multi-edit, and an online-load serving study (SGLang, in-place kernel) are future work. The erratum is
template-sensitive in non-reasoning settings; field+erratum is the robust default.

**Composable-axis limitations.** (i) The keystone uses a D1-style *recovery ratio*; absolute decision
sign-flips need a competent regime (clean on Llama-3.1 7/8 and Mistral 4/4, but Qwen3-8B/14B do not flip
on these templates in non-reasoning). (ii) The context-staleness failure mode — a skill that is the
*sole carrier* of a context-derived computation (not re-derivable downstream) — is identified but not
stress-tested; our skills are loosely coupled. (iii) The transplant machinery assumes standard
single-RoPE attention with post-RoPE key caching; **Phi-3.5** (needs eager + legacy cache API) and
**Gemma-3** (dual local/global rotary, multimodal wrapper) need adapter work and are untested for
composition. (iv) Precompiling assumes the skill is reused enough to amortize the one-time isolation
prefill; for single-use skills full reprefill is fine. (v) We do not claim a novel caching *system* —
Prompt Cache/CacheBlend/EPIC established splicing; our contribution is the correctness lens and the
unification with editing.

**On quantization (a methodological note).** To run 32B-class models alongside a shared training job
we used 8-bit checkpoints. The **causal-patching recovery is a margin-sensitive *ratio*** ((s_patched −
s_old)/(s_new − s_old)), which becomes NaN/unstable when quantization collapses the decision margin — so
mechanism *patching* should use full precision, while the **behavioral checks** (binary decisions: the
D4 circuit, erratum recovery) are quantization-robust. The official **Qwen3-32B-FP8** loaded cleanly
(FP8 is natively supported on this Blackwell `sm_120` GPU) and gave a clean D4 circuit (§5.3). **8-bit
*Gemma* could not be run cleanly here**, for distinct concrete reasons we record for reproducibility:
the largest Gemma (gemma-3-27b) is *multimodal* (its vision tower makes even the FP8 checkpoint ~58 GB →
OOM with the 31 GB training job, and the multimodal wrapper breaks the text-only KV-patching);
bitsandbytes int8 *degrades* gemma-2-27b's gated-decision competence (the oracle stops flipping);
RedHat's calibrated W8A16 triggers a CUDA device-side assert (int8 kernels are immature on `sm_120`);
and community FP8 gemma-2 checkpoints fail to load (meta-tensor materialization bug). The clean Gemma
mechanism point therefore remains bf16 gemma-2-27b (§5.1). Completing 8-bit Gemma needs more GPU memory
(for multimodal gemma-3) or fixed int8 Blackwell kernels / FP8 checkpoints — infrastructure, not method.

## 12. Conclusion

The KV cache is a first-class, manipulable object: information in it is **localized, position-portable,
and context-robust**, and we exploit that property two ways. **Editable** — a decision reads a field
indirectly through a memoized downstream, so the naive cheap edit reverts the decision; reasoning
rescues it only unreliably and scale-dependently; the practical picture is a frontier (the bare
surgical edit is ~free and sufficient for reasoning models, `field+erratum` matches hoist *in place*),
and on a real agent environment editkv preserves task success at a fraction of the recompute.
**Composable** — the same locality and portability let us precompile a long SKILL once and transplant
its KV into a new context, behaviorally indistinguishable from full recompute (cos 0.96–0.999 across
six models) at up to 13.9× lower TTFT. The **keystone** ties them together: editing a field inside a
*transplanted* skill reproduces the editable mechanism verbatim — edit and compose are two operations
on one substrate. We give a causal, multi-method mechanistic account of *why* throughout, and release a
production library and a closed vLLM integration.
