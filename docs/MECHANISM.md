# Why field-only editing hedges: a mechanistic account (attention-level)

> **Note.** This is an early exploratory lab note (single model, single scenario, one
> forward pass) kept for context. The **rigorous, replicated account is the paper**
> ([`../paper/main.pdf`](../paper/main.pdf), §3 and the circuit appendix): the same
> mechanism, established causally across four model families with confidence intervals.
> Read this only if you want the original attention-level intuition behind that result.

> Preliminary mechanistic study (Qwen3-8B, account_role, non-thinking single-decode so
> the action is one forward). Causal evidence via KV-patching and attention-knockout.
> `esys/mech_attention.py`, `results/mech_attention_qwen3_8b.json`. n=1 greedy forward,
> one scenario — directional, to be replicated on 30B/32B and more scenarios.

## The framing (revised away from "two registers")

"Registers" was the wrong metaphor — it implies discrete, addressable slots at fixed
positions. The mechanism is a **distributed, attention-mediated dataflow staleness**:

> During **prefill**, every downstream token attends to the field and writes
> *field-conditioned inferences* into its own residual/KV ("given admin, this clause is
> satisfied"). Prefill thus **memoizes the field's implications diffusely across many
> downstream positions**. At decode, the decision token reads the field through two kinds
> of attention path: a **direct** path to the field token (one hop) and an **indirect**
> path to those downstream tokens that already integrated it (the memoized inference). A
> field-only edit refreshes only the direct path; the indirect, memoized path stays stale.
> The decision is the (attention-weighted) superposition — and the **indirect path
> dominates**, so the edit is nearly inert.

## Evidence

**(1) The decision is a 3-way choice, and field-only produces the cautious one.**
benign: stale→`lookup`, **field_only→`lookup`** (the safe-but-unfaithful hedge),
erratum→`escalate`, oracle→`escalate`. poison: stale/field_only/oracle→`issue_refund`
(all fooled), erratum→`escalate`.

**(2) Causal KV-patching: the field is inert; the bulk downstream carries the decision.**
`escalate − issue_refund` logit when patching fresh KV into the stale cache (benign):

| patched span | esc−unsafe logit |
|---|---|
| none (stale) | −14.5 |
| **field only** | **−14.7** (no change) |
| gate only | −15.8 (no change) |
| **all downstream** | **+11.8** (flips to escalate) |

Neither the field nor the gating rule alone moves the decision; only refreshing the
diffuse downstream does. The decisive information is **distributed**, not localized.

**(3) Attention knockout (the causal test).** In the field-only decode, mask the decision
token's attention to selected cached positions:

| knockout | benign | poison |
|---|---|---|
| none (baseline) | lookup | issue_refund |
| → gate span only | lookup (no change) | issue_refund (no change) |
| **→ all stale downstream** | **escalate** (−14.7→+0.97) | **escalate** (−22.6→+0.59) |

Cutting the decision's attention to the **whole** stale downstream flips it to the
**correct** action — in both benign and poison. So: the correct answer *is* recoverable
from prefix + refreshed field alone; the failure is that the decision **over-attends to
the diffuse stale downstream, which out-votes the field.** Knocking out a single span
(the gate) does nothing — confirming the signal is distributed.

**(4) Value-field control.** A non-gating field (current_date feeding a return-window
computation) has **no memoized downstream inference** (nothing concluded from it at
prefill). There, field-only behaves like the oracle: 0.00 stale-answer rate (never the old
value), high new-correct — confirming value fields are the easy regime *because* there is
no indirect path to go stale.

## How this explains the behavior

- **Hedge (safe-but-unfaithful)** = the direct path (refreshed field, "suspended") and the
  dominant indirect path (stale "allowed") conflict; the model retreats to the cautious
  third option. Knockout of the indirect path resolves it to the correct action.
- **Thinking** repairs it by generating fresh downstream tokens (CoT) whose attention
  re-integrates the field correctly; the decision's indirect path now lands on fresh,
  correct tokens (and out-votes the stale ones).
- **Erratum** injects one recent, high-salience downstream token span that the indirect
  path attends to — a fresh correct vote that overrides the stale ones (hence it must
  override "earlier *conclusions*", i.e. the memoized inference, not just restate the value).
- **Poison** plants an explicit stale inference in the text; it is present in the *new*
  context too, so even full reprefill reproduces it (only re-derivation or override wins).
  All-downstream knockout still flips it because masking removes the decision's access to
  that explicit inference.

## Rigorous N>1 suite (Qwen3-8B; the N=1 results above are superseded by these)

`esys/mech_suite.py`. M=12 instances (3 gating scenarios × 4 order-IDs) for the
deterministic forwards; 36 stochastic CoT samples (6 instances × K=6) for reasoning.
Proportions with Wilson 95% CIs. account_role/safety_mode/subscription_tier;
"safe"=the policy-correct action, "unsafe"=the violating one.

**E4 — attention attribution + sinks (where the decision looks).** Mean attention mass
of the decision token: **field 0.001, original-downstream 0.56, sink (pos 0–3) 0.36.**
The direct field edge is ~0.1% — the field's influence reaches the decision almost
entirely through the memoized downstream, with a large attention-sink share.

**E1 — graded knockout (how distributed).** P(safe) when masking the decision's
attention to the top-k% highest-attention downstream positions: 0%→0.00 [0,.24],
10%→0.25 [.09,.53], **25%→1.00 [.76,1]** (saturates). The memoized inference lives in
roughly the **top attention quartile** of downstream — distributed, not one token.

**E2 — layer-band knockout (where in depth).** Masking decision→downstream within layer
thirds: early 0.00 [0,.24], mid 0.25 [.09,.53], late 0.33 [.14,.61], **all 1.00 [.76,1]**.
No single third suffices; the signal is read across **mid-and-late** layers.

**E3 — reasoning resolution (the mode contrast), n=36 samples.**

| intervention (field-only) | NON-reasoning (N=12) | REASONING (n=36) |
|---|---|---|
| baseline P(safe) | **0.00** [0,.24] | **1.00** [.90,1] |
| KO original stale downstream | **1.00** [.76,1] — *fixes it* | 0.97 [.86,1] — *harmless* |
| KO fresh CoT tokens | — | **0.61** [.45,.75] — *reverts* (P_unsafe 0.39) |

This is the crux, and the CIs separate cleanly: **non-reasoning** decisions are *harmed
by* the stale downstream (knocking it out flips 0.00→1.00 safe); **reasoning** decisions
*depend on* the fresh CoT (knocking it out drops 1.00→0.61, CI excludes baseline) and are
*indifferent* to the original stale downstream (0.97, CI overlaps baseline). I.e.
reasoning **relocates** the decision's evidence from the stale memoized downstream onto
the freshly-derived CoT. (KO-CoT does not fully revert to 0 because the field is still
refreshed and some conclusion leaks into unmasked scaffold tokens.)

## Cross-size ladder (4B, 8B, 14B, 30B-A3B MoE, 32B; both modes)

**Scale-INVARIANT (the structural backbone), all 5 sizes:**
- **E4:** the decision's *direct* attention to the field token is **~0.1% at every size**
  (field mass 0.0009–0.0017; downstream ~0.5; sink ~0.36–0.48). The field's influence on
  the decision is overwhelmingly *indirect* (via the memoized downstream) regardless of scale.
- **E1/E2 (non-reasoning):** field-only baseline P(safe)=0.00 at every size; knocking out the
  decision's attention to the stale downstream restores it. Universal.

**Reasoning-rescue is NOT scale-monotone and is model-dependent** (field-only, reasoning):

| model (active) | base safe / unsafe / hedge | KO-CoT safe / unsafe |
|---|---|---|
| 4B  | 0.31 / **0.36** / 0.33 | — |
| 8B  | **1.00** / 0.00 / 0.00 | 0.61 / 0.39 (CoT corrective) |
| 14B | 0.44 / **0.19** / 0.36 | 0.94 / 0.00 (CoT harmful) |
| 30B-A3B (3B act.) | **1.00** / 0.00 / 0.00 | 0.50 / 0.50 (corrective) |
| 32B | 0.12 / 0.00 / **0.88** | 0.06 / 0.19 (over-cautious) |

There is **no clean trend**: 8B and 30B-A3B rescue field-only cleanly; 4B (anchors→unsafe),
14B (CoT manufactures unsafe), and 32B (collapses to caution) do not. ⇒ **"thinking rescues
the cheap field-only edit" is NOT a general property** — it holds for some models and fails,
in distinct ways, for others.

**CONFOUND (important):** `base_safe` conflates the model's *general competence* at the task
with the *cache-edit penalty*. A weak (4B) or cautious (32B) model may give a low safe-rate
even with a full re-prefill. To isolate the edit penalty we need the oracle reasoning
baseline per model (penalty = oracle_safe − fieldonly_safe). That oracle-controlled +
erratum-at-scale study is the next run; until it lands, treat the per-model reasoning
*rescue* numbers as competence-confounded — only the **structural** findings (E4 field≈0.1%;
non-reasoning baseline 0; knockout fixes) are clean cross-scale claims.

## Caveats / next
- n=1 greedy forward, one scenario (account_role), 8B; replicate on 30B-A3B / 32B and
  more scenarios (queued).
- All-downstream knockout is a heavy intervention (removes legitimate info too); the
  informative fact is that it flips to the *correct* action, i.e. prefix+field suffice.
- The poison conclusion-span auto-detection failed (subword boundary); the all-downstream
  knockout sidesteps it, but a span-resolved poison knockout would localize the explicit
  inference. A graded knockout (mask top-k highest-attention downstream positions) would
  measure how distributed the signal is.
- Decision proxy = first-token logit of {escalate, issue_refund, lookup} after forcing
  "tool_call:"; matches the behavioral argmax but should be cross-checked against full
  generation.
