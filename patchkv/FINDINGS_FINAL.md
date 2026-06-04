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

(Caveat: the 896-token budget at temp 0.7 leaves a few samples with unfinished CoT,
counted as neither correct nor unsafe — this depresses raw P_correct slightly, e.g.
field_only's lone "number" token; it does not affect P_unsafe.)

Reading it together with §3–§4: in a **benign** context both field_only and erratum
drive the unsafe rate to **0** (stale_full leaves 33% unsafe), and field_only's
clean-case reliability (0.83, 0 unsafe) confirms the 8B success was not a fluke. In a
**poisoned** context (§4) field_only flips to unsafe and **only the erratum stays
safe**. So the variance/scale wobble of field-only is real but bounded in benign
contexts; the erratum removes it and additionally survives contradiction.

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
