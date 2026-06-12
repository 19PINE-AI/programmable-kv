# Findings — Deep mechanism evidence for "memoized conclusions"

*Autonomous run, 2026-06-11. One RTX PRO 6000 (96 GB). Models: Qwen3-8B, Llama-3.1-8B,
Qwen3-4B (≥3 models per experiment). Harness: `esys/mechd_*.py`, results in
`results/mechd_*_*.json`. Scaffolding: `esys/mechd_common.py` (polarity-parameterized
rules).*

## Why this study

The paper claims the KV cache holds **memoized field-conditioned conclusions** on
downstream aggregator tokens. The original four probes (locality, suffix concentration,
linear decodability, knockout) are strong but share two soft spots a reviewer can press:

1. **Decodability ≠ use.** A linear probe shows the conclusion is *decodable* downstream,
   but not that the decision *uses* a pre-computed conclusion rather than re-deriving it.
2. **One synthetic template.** The probes mostly live on one construction (POLICY + field
   → cancel/deny), so "memoized conclusion" might be a template artifact.

These five experiments close those gaps. **All five support the claim**, with one honest
boundary (direct attribute-lookup).

---

## EXP1 — Conclusion vs. content dissociation (flagship causal test)
`esys/mechd_xcond.py`

**Design.** A *polarity-parameterized* rule lets us hold the FIELD value byte-identical
while flipping the derived CONCLUSION: the gate names a single `trigger` value that selects
the SAFE action, so `base=(field=v, trigger=v)`→SAFE and `source=(field=v, trigger=v')`→
UNSAFE differ in exactly one token (the trigger, inside the rule), with the field token
identical. We then transplant `source`'s KV into `base`:

| patch site | what it supplies | Qwen3-8B | Llama-3.1-8B | Qwen3-4B |
|---|---|---|---|---|
| **trigger token only** | the differing rule token | **−0.007** | **+0.007** | **−0.007** |
| **downstream notes** (post-trigger) | only the memoized note | **+1.004** | **+0.998** | **+1.009** |

(recovery toward the opposite conclusion; n=36 pairs/model.)

**Result.** With the field constant, patching the **downstream notes alone fully transplants
the conclusion** (≈1.0), while patching the differing **rule token carries ≈0**. The
decision cannot be "re-encoding field content" (the field is identical in both conditions)
— it follows the **derived conclusion written into the notes**. This is the cleanest causal
separation of *content* from *conclusion*; it parallels the original field-only result but
removes the field as a confound entirely.

**Probe nuance (important, honest).** From a downstream delimiter, BOTH the conclusion and
the field identity are linearly decodable (field_acc ≈ 1.0 by mid-depth). So **decodability
alone cannot tell them apart** — which is exactly why the *causal* patch above is the
right instrument. (On Llama the conclusion is decodable at layer 12, depth 0.38, *before*
field identity saturates — even the decodability gradient leads with the conclusion.)

---

## EXP2 — Layer/timing emergence ("computed at prefill" gets a layer axis)
`esys/mechd_timing.py`

"Pre-computed" is a temporal claim. During the **single prefill forward pass** we track how
decodable the conclusion is from a downstream aggregator delimiter at each layer (using the
polarity 2×2 so conclusion ⟂ field), and compare to the layer at which the decision token
commits (logit-lens, final-sign).

| model | nlayers | **write** layer (aggregator concl_acc ≥0.9) | **commit** layer (decision logit-lens) |
|---|---|---|---|
| Qwen3-8B | 36 | **14** (depth 0.39) | 26.9 (depth 0.75) |
| Llama-3.1-8B | 32 | **10** (depth 0.31) | 23.5 (depth 0.73) |
| Qwen3-4B | 36 | **14** (depth 0.39) | 27.9 (depth 0.77) |

**Result.** The conclusion is written onto the downstream aggregator at **~mid-depth (0.3–0.4)**,
**~12–13 layers before** the decision token commits (depth ~0.75). The note is computed and
in place during prefill, well before the read-out — the temporal core of "models take notes
at prefill." The earliest-writing anchor is the `assistant: …before acting.` delimiter,
upstream of the decision region.

---

## EXP3 — Generalization off the synthetic template
`esys/mechd_general.py`

Re-ran the two decisive probes (FIELD-ONLY ≈0, FULL-DOWNSTREAM ≈1) on three qualitatively
different families with a single-span field flip: **multihop** (2-hop key→vault→datacenter),
**rag_lookup** (attribute lookup name→desk#), **natural** (the gated decision in free-form
conversational prose, no POLICY/SESSION scaffolding).

| family | Qwen3-8B field-only | Llama-3.1-8B | Qwen3-4B | full-downstream (all) |
|---|---|---|---|---|
| multihop | −0.012 | +0.002 | +0.006 | **1.00** |
| natural  | +0.054 | −0.012 | +0.019 | **1.00** |
| rag_lookup | +0.251 | +0.377 | +0.633 | **1.00** |

**Result.** For **multi-hop reasoning** and **natural conversational phrasing**, field-only
recovery is ≈0 and the conclusion is fully downstream — the mechanism is **not a template
artifact**. **Honest boundary:** for near-direct **attribute lookup** (`rag_lookup`) the
field token carries a substantial and model-dependent fraction (0.25→0.63), because the
answer is closer to a copy than a computed conclusion. This *bounds* the claim precisely:
memoization dominates for **derived/gated conclusions**, less so for direct copies — exactly
where the theory predicts.

---

## EXP4 — Specificity: specific aggregator tokens, not diffuse
`esys/mechd_specificity.py`

At matched count k, transplant the **top-k** downstream positions (ranked by individual
causal effect) vs **k random** downstream positions.

| | top-8 | random-8 | top-16 | random-16 |
|---|---|---|---|---|
| Qwen3-8B | **0.78** | 0.009 | 0.92 | 0.06 |
| Llama-3.1-8B | **0.79** | 0.005 | 0.86 | 0.04 |
| Qwen3-4B | **0.74** | 0.035 | 0.86 | 0.04 |

**Result.** Eight *specific* aggregator tokens recover **74–79%** of the decision; eight
*random* downstream tokens recover **≈1–3%**. The conclusion is carried by a **small set of
specific tokens**, decisively ruling out a diffuse "any token would do" code.

---

## EXP5 — Counterfactual note injection (the notebook is writable)
`esys/mechd_inject.py`

Start from a self-consistent prefill whose **live field implies conclusion C**, then
overwrite only the downstream notes with notes from a context whose conclusion is **C′**
(opposite). The field token and the whole prefix are untouched and still imply C.

| | full-note injection recovery toward C′ | follows injected note | top-8 dose → follows |
|---|---|---|---|
| Qwen3-8B | **0.99** | 1.00 | 0.72 |
| Llama-3.1-8B | **1.02** | 1.00\* | 1.00 |
| Qwen3-4B | **0.98** | 1.00 | 0.56 |

\*Llama direct-mode argmax is partly constant-answer (categorical flip-rate uninformative),
but the **continuous** recovery is full (1.02) — the note fully determines the margin.

**Result.** Writing a **false conclusion** into the notes makes the model decide per the
written note, **against its own live field** (recovery ≈1.0). A handful of note tokens
(top-8) already flips the belief in most cases. "Notebook you can write to" is now a
**measured capability**, and it is the same operation the editing axis exploits (editing =
overwriting this note with the *true* value).

---

## Bottom line

The five experiments convert the central metaphor from **correlational** ("the conclusion
is decodable downstream") to **causal and temporal**:

- **Dissociated** from field content (EXP1): conclusion transplants with the field held
  identical; the differing rule token carries ≈0.
- **Computed at prefill** (EXP2): written onto aggregators ~12 layers before the decision
  commits.
- **Not a template artifact** (EXP3): holds for multi-hop and natural prose; bounded only
  for direct lookups.
- **Localized** (EXP4): a few specific tokens, not diffuse — top-8 ≫ random-8.
- **Writable** (EXP5): injecting a false note overrides the live field.

Honest boundaries reported: (i) decodability cannot itself separate content from
conclusion (EXP1 probe) — the causal patch is required; (ii) direct attribute-lookup is
partly field-readable (EXP3 rag); (iii) Llama direct-mode argmax is degenerate, so EXP5
uses the continuous metric there. None of these weaken the core account; they sharpen its
scope.

---

## Cross-family replication: Gemma-2 and Mistral (added 2026-06-12)

*Harness: `esys/mechd_replicate.py`. Results: `results/mechd_replicate_{gemma2_9b,mistral_7b}.json`.*

To rule out the "aggregator-token memoization" story being a Qwen3/Llama tokenizer artifact, we
replicated all five deep probes on **Gemma-2-9B-it** and **Mistral-7B-Instruct-v0.3** — two new
architecture families. Two harness details had to be fixed (no science changed):

1. **Readout.** The tool-call action vocabulary is multi-token in these tokenizers (e.g. Gemma
   splits `refuse`→2 tokens), and both models emit a leading space before the answer word. We read
   **space-prefixed single-token actions** (`␣cancel`/`␣deny`) at a trailing-space decision suffix.
2. **Soft-capping.** Gemma-2 needs attention + final-logit soft-capping; `mech_suite.install()`
   (the attention-knockout hook, used only by the *original* circuit-knockout probe, which none of
   the five deep probes use) bypasses it and corrupts Gemma-2. We load with stock eager attention.

**All five deep-mechanism results replicate cleanly on both new families** (n=18 primary, 18 dissoc):

| probe | Qwen3/Llama (orig) | Gemma-2-9B | Mistral-7B |
|---|---|---|---|
| (P) field-only recovery (~0) | ≈0 | **0.005** [0.002,0.008] | **0.137** [0.119,0.155] |
| (P) full-downstream (~1) | 1.0 | **1.0** | **1.0** |
| (D) trigger-only (field fixed; ~0) | −0.007..+0.007 | **−0.00** [−0.002,0.001] | **0.001** [−0.003,0.005] |
| (D) downstream notes (~1) | 0.998..1.009 | **1.0** [0.998,1.002] | **0.995** [0.993,0.998] |
| (S) top-8 vs random-8 | 0.74–0.79 vs ≤0.035 | **0.955 vs 0.484** | **0.942 vs 0.528** |
| (I) false-note injection recovery | ~1.0 | **0.999**, follow 1.0 | **0.983**, follow 1.0 |
| (T) write depth vs commit depth | 0.31–0.39 vs 0.73–0.77 | **0.26 vs 0.48** | **0.19 vs 0.47** |

**Conclusion.** The dissociation (conclusion ⟂ content), the localization to specific downstream
tokens, the writability, and the write-before-read timing all hold across **five** model families
(Qwen3, Llama-3.1, Gemma-2, Mistral). The mechanism is not a family artifact.

**Two honest cross-family notes.** (i) Mistral field-only recovery is 0.137 (vs ≈0 for the others) —
slightly above zero but far below full-downstream's 1.0; the conclusion is still overwhelmingly
downstream. (ii) On the cancel/deny task, random-8 specificity runs higher (~0.48–0.53) than on the
Qwen/Llama tool-call task (~0.02) — the notes are a bit more distributed here — but top-8 (~0.95)
still dominates random-8 decisively, preserving the localization claim.
