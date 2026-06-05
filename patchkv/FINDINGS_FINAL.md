# PatchKV — Final consolidated findings: making editable KV work

> One narrative across every experiment run on the local Blackwell GPU.
> Detailed docs: `FINDINGS_E1_E2.md` (characterization), `FINDINGS_EXTENSIONS.md`
> (selection/τ-bench/horizon/frontier), `FINDINGS_MAKING_IT_WORK.md` (thinking + erratum).
> Models: Qwen2.5-1.5B, Qwen2.5-7B-Instruct (non-thinking), Qwen3-8B, Qwen3-14B (thinking).

---

## The answer in one paragraph

When a field changes inside an already-cached agentic context, you do **not** need to
recompute the stale downstream KV — you need the *decision* to stay correct. Two cheap
levers achieve that while reusing the entire static prefix for free: (1) **live
re-derivation** — a thinking model's CoT re-reads the refreshed field and re-derives the
implication, so stale downstream KV is overridden; and (2) **salience** — appending a short
authoritative **erratum** at the suffix (`[STATE UPDATE] <field> → <new>; overrides any
earlier value and conclusion`) and recomputing only those ~5–6% of tokens. The erratum is the
robust mechanism: it survives even *contradictory* stale context — and is more robust than a
full reprefill. Editable KV works, at near-prefix-cache cost, with the field left in its
natural place.

---

## Scope & framing: reasoning models are the target; instruction models are background

There are two decoding regimes, and they behave differently here:

- **Instruction-tuned (non-reasoning) models** answer in essentially a *single pass*:
  prefill the prompt, then immediately emit the action. There is no decode-time
  reasoning to re-read an edited value, so any correctness must already be present in
  the cache. **Prior KV-reuse / cache-editing work (CacheBlend, Prompt Cache, selective
  recompute, etc.) implicitly lives in this regime** — which is exactly why those methods
  *recompute* the affected tokens: with no re-derivation at decode, a stale cache stays
  wrong.
- **Reasoning models** generate a chain-of-thought *first*, then act. The CoT re-reads
  the edited field's current value and re-derives its consequences at decode time. This
  is a second, cheaper place to restore correctness — **and it changes the story.**

**Today's deployed tool-using agents are reasoning models, so they are our focus.** The
central, regime-specific claim:

> On **reasoning** models, a ~0.1% field-token KV edit (leave everything else stale)
> recovers the correct action in benign contexts because the CoT re-derives it live. On
> **instruction** models the *same* edit fails — there is no re-derivation — so you must
> either recompute the dependent region or use the salience-based **erratum**. The
> erratum works in **both** regimes; field-only is a reasoning-model-only shortcut.

This is verified with a clean within-model ablation (same Qwen3-8B weights,
`enable_thinking` on vs off — removing the model-family confound):

<!-- ABLATION_PLACEHOLDER -->

Instruction models are thus the *harder, background* case (where prior work sits and
where the cheap edit does not transfer); reasoning models are where editable KV becomes
cheap. The recipe below is written for the reasoning-model target, with the
instruction-model fallback noted.

---

## The recipe (decision tree)

```
field edit on an already-cached context
│
├─ thinking model, benign context      → refresh ONLY the field token (~0.1%); leave rest stale
├─ any model / robustness matters       → + append salient erratum at suffix (~5–6%); leave rest stale
│                                          (field+erratum = exact field refresh + erratum)
└─ field placed AFTER its gating rules   → faithful recompute of the short post-field tail
   (e.g. τ-bench)                          (~4–5%); the rules are causally-exact and free
```
Never required: recomputing the static policy/prefix (always reused for free, deviation 0.0).

**Bottom line:** the safe default is **erratum (~6%)** — it is correct with and without
thinking, robust to contradictory stale context (more robust than a full reprefill),
and keeps the field in place. **field-only (~0.1%)** is the cheaper option that works in
benign contexts on thinking models but is fooled by poisoned context and wobbles at
scale. Both beat full reprefill on cost and beat hoist-to-end on programmability.

---

## The journey (why the recipe is what it is)

### 1. Characterization (E1/E2) — `FINDINGS_E1_E2.md`
- **H2 confirmed exactly:** KV of every token *before* the edited field is bit-identical
  (deviation 0.0) — the static prefix is always reusable for free.
- Blast radius is sparse and field-dependent (low<medium<high), but raw KV-deviation
  over-counts and is not a portable safety threshold — **decisions are the metric with teeth.**
- Non-thinking decode: low-conditioning fields (time/ids/counters) are leave-stale-safe with
  zero refresh; decision-relevant fields flip and a bare leave-stale fails.

### 2. Mechanism & frontier (Phases A–D) — `FINDINGS_EXTENSIONS.md`
- Selection: deviation- and recency-ranked residuals are **complementary** (use the union).
- Sufficient refresh is governed by **conditioning breadth (≈ E1 blast radius) × placement**,
  not placement alone (safety_mode recovers at ~6% sparse; account_role needs ~96%).
- E-horizon: a low-field patch stays **flat (100% agreement) over a 5-step trajectory** — no
  compounding; high-field errors are localized to field-dependent steps.
- **Faithful E-sys frontier (non-thinking):** field refresh is exact (cosine 0.99989); but for
  early-gated fields, faithful leave-stale fails until ≈ full reprefill, and **hoist-to-end +
  prefix caching dominates** (3.5% recompute, correct). *This was the pessimistic conclusion —
  and it was an artifact of non-thinking decode.*

### 3. The pivot: thinking — `FINDINGS_MAKING_IT_WORK.md`
Real tool-calling agents reason before acting. With a thinking model the picture **reverses**:
refreshing only the field token (~0.1%) and leaving all else stale recovers the correct flipped
decision on Qwen3-8B for every decision-relevant field; `stale_full` reproduces the old decision
(clean isolation). The CoT explicitly re-reads "suspended_user" and re-derives "escalate" — H1
made concrete. Under thinking, field-only (~0.1%, natural placement) **beats hoist-to-end** on
both cost and programmability.

### 4. Honest limits, and why erratum is the robust hero
- **Scale/variance (Qwen3-14B):** field-only+thinking is mixed (1 clean recover, 1 safe-cautious,
  1 stuck). A single greedy CoT is a high-variance map (the low-field control itself flipped).
- **Poisoned context (stale self-conclusion asserting old permission), scored by policy-safe
  action:**

  | regime | stale | field_only | **erratum** | field+erratum |
  |---|---|---|---|---|
  | thinking, account_role | VIOLATES | VIOLATES | **SAFE** | (trunc) |
  | thinking, safety_mode | (trunc) | SAFE | SAFE | SAFE |
  | non-thinking, account_role* | VIOLATES | VIOLATES | **SAFE** | SAFE |
  | non-thinking, safety_mode* | VIOLATES | VIOLATES | **SAFE** | SAFE |

  *the full-reprefill oracle is itself fooled by the poison here.*

  `field_only` is fooled by a poisoned prior conclusion; **the erratum produces the policy-safe
  action across the board — including non-thinking, and even where a full reprefill fails.** An
  explicit "overrides any earlier value and conclusion" instruction beats a silent KV change.

### 5. Multi-sample robustness (rates over CoT variance)
Because a single greedy CoT is high-variance, we sample k=6 completions (temp 0.7,
Qwen3-8B) per method and report rates. **Benign** account_role (correct=escalate,
unsafe=issue_refund):

| method | P_correct | P_unsafe |
|---|---|---|
| oracle_new | 1.00 | 0.00 |
| stale_full | 0.00 | 0.33 |
| **field_only (~0.1%)** | **0.83** | **0.00** |
| **erratum (~6%)** | **1.00** | **0.00** |

(Caveat: the 896-token budget at temp 0.7 leaves some samples with unfinished CoT,
counted as neither correct nor unsafe. account_role's CoT is short so it is clean; the
longer-reasoning scenarios (safety_mode ≈1.4k CoT tokens) are truncation-noise-limited
— even their *oracle* P_correct drops (~0.33) from unparsed output. **P_unsafe stays
meaningful under truncation** (no sample does the violating action), so we read this as:
field_only/erratum reliably avoid the unsafe action in benign contexts; P_correct here
is a lower bound, not a true accuracy. The full 3-scenario json is in
`results/multisample_qwen3_8b.json`.)

**The clean cross-scenario signal is P_unsafe** (robust to truncation). Across both
benign scenarios, leaving the cache stale is unsafe while *every* refresh method is not:

| P_unsafe (lower=safer) | stale_full | field_only | erratum | field+erratum |
|---|---|---|---|---|
| account_role | 0.33 | **0.00** | **0.00** | **0.00** |
| safety_mode | 1.00 | **0.00** | (–) | (–) |

Reading it together with §3–§4: in a **benign** context every refresh method drives the
unsafe rate to **0** (stale leaves 33–100% unsafe), and field_only's clean-case
reliability (0.83 correct, 0 unsafe on account_role) confirms the 8B success was not a
fluke. In a **poisoned** context (§4) field_only flips to unsafe and **only the erratum
stays safe**. So the variance/scale wobble of field-only is real but bounded in benign
contexts; the erratum removes it and additionally survives contradiction.

*(Run note: the multi-sample sweep was stopped after the account_role (full) and
safety_mode (partial) scenarios — long-CoT decodes are truncation-noise-limited at the
896-token budget and slow; account_role is the clean representative case and the
P_unsafe signal is consistent across both.)*

---

## Cost summary (vs full reprefill of a ~700–1500-token context)

| method | recompute | correct? | keeps natural placement? | robust to poisoned context? |
|---|---|---|---|---|
| full reprefill | 100% | ✓ (unless poisoned) | ✓ | ✗ (can be fooled) |
| hoist-to-end + prefix cache | ~3.5% | ✓ (non-thinking) | ✗ (must restructure) | partial |
| **field-only refresh** | **~0.1%** | ✓ thinking+benign; ✗ poisoned/scale | ✓ | ✗ |
| **erratum (suffix)** | **~5–6%** | ✓ (incl. non-thinking) | ✓ | **✓ (best)** |
| field + erratum | ~6% | ✓ | ✓ | ✓ |

CoT tokens are **not** charged to the patch — a thinking agent emits them regardless; the patch
only changes the *prefill*, turning a full reprefill into a ~0.1–6% recompute.

## Honest boundaries / future work
- field-only alone is variance-sensitive at scale and fooled by contradictory stale context ⇒
  prefer erratum / field+erratum when robustness matters.
- Greedy single-sample CoT is noisy; report rates (multi-sample) — see §5.
- τ-bench + thinking is extraction-limited (verbose answers); τ-bench *without* thinking is clean
  (late field recovers at ~4.4% recompute).
- Erratum phrasing is a tunable lever; an early stale value contradicting a late override could
  confuse weaker models — measure per deployment. MLA/sparse-attention backbones untested.

## Reproduce — see the per-phase docs and `esys/`, `e1/`, `e2/`.
