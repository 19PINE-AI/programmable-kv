# Why field-only editing hedges: a mechanistic account (attention-level)

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
