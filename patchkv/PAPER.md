# Editable KV Cache for Mutable Fields in Agentic Contexts: When the Cheap Edit Works, Why It Fails, and a Robust Fix

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

## 2. Related work

Prefix caching (vLLM APC, SGLang RadixAttention) reuses exact prefixes. Selective-recompute methods
for *composition* (CacheBlend, EPIC/AttnLink) recompute ~15% / boundary tokens to restore
cross-attention when assembling *independent* chunks; selection methods (InfoFlow KV, KVShare) decide
*which* tokens to recompute. All target chunk composition or cross-request sharing and recompute the
affected downstream. We study a *temporal edit of one already-jointly-encoded context*, and ask the
opposite question — when can the downstream be left **stale**? We also show prior work implicitly
assumes **single-pass (instruction) decoding**; reasoning models change the picture, and the robust
fix is a salience injection, not recomputation. Our mechanism analysis adapts causal tracing
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
identifiable). **Generalizes across 8 models** (field-only recovery ≪ full=1.0): Qwen3-4B/8B/14B/32B
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

## 10. Limitations

The mechanism battery (n=12 instances) is on a few scenario templates; the scale-reversal explanation
(§5.3) is qualitatively clear but small-n at 14B. The τ²-bench evidence is the retail cancel-task
family (real env/tools/reward, single-decision N=20 + multi-turn N=30), not the full 114-task
user-simulator sweep with a strong agent (local models that fit one 96 GB GPU are weak τ²-bench agents,
which would confound absolute reward). Models are ≤32B open weights (no frontier/API scale; 70B needs
4-bit). MLA in_place editing of the compressed latent KV, pure-SSM support, multi-field/long
multi-edit, and an online-load serving study (SGLang, in-place kernel) are future work. The erratum is
template-sensitive in non-reasoning settings; field+erratum is the robust default.

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

## 11. Conclusion

Editable KV is viable, but the naive cheap edit is not a free lunch: the decision reads the field
indirectly through a memoized downstream, so leaving it stale reverts the decision, and reasoning
rescues it only unreliably and scale-dependently. The practical picture is a frontier, not a winner:
the bare surgical edit is ~free and sufficient for reasoning models; `field+erratum` matches the strong
hoist baseline *in place*, without prompt surgery; and on a real agent environment editkv preserves
task success at a fraction of the recompute where the stale agent collapses. We give a causal,
multi-method mechanistic account of *why*, show it is an attention-architecture method, and release a
production library and a closed vLLM integration.
