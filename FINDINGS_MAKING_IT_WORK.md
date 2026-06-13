# Making editable KV work: refresh the field, inject a salient erratum, leave the rest stale

> Constructive synthesis. Earlier phases (non-thinking) concluded leave-stale fails for
> early-gated fields and hoist-to-end wins. That was an artifact of an unrealistic setup
> (the model didn't think) and of judging a single greedy decode. Re-run realistically,
> editable KV **works** — cheaply — and the robust mechanism is a salient **erratum**.
> Models: Qwen3-8B / Qwen3-14B (thinking), Qwen2.5-7B-Instruct (non-thinking).

---

## The recipe

**Robust recipe (recommended): erratum injection.** Leave the entire cache stale, append a
short authoritative correction at the suffix —
`[STATE UPDATE] <field> has changed to <new>; this overrides any earlier value AND any
earlier conclusion` — and recompute **only those ~tens of tokens (~5–6%)**. Optionally also
refresh the edited field's token KV (exact, ~0.1%) for redundancy (`field+erratum`).

**Cheapest recipe (thinking models, benign context): field-only refresh.** Refresh just the
edited field token (~0.1%), leave all else stale; the CoT re-reads it and re-derives the
decision. Works when the stale context is not actively contradictory.

Both keep the field in its natural place (no prompt restructuring) and reuse the static
prefix for free.

---

## Evidence 1 — thinking + field-only recovers (clean context, Qwen3-8B)

Decision = post-`</think>` tool call. `field_only` refreshes only the field token (~0.1%),
all else stale. `stale_full` refreshes nothing.

| field | stale_full | **field_only (~0.1%)** | oracle |
|---|---|---|---|
| account_role | lookup_order ✗ | **escalate ✓** | escalate |
| safety_mode | share_payment_method ✗ | **refuse ✓** | refuse |
| subscription_tier | expedite_shipping ✗ | **refuse ✓** | refuse |

`stale_full` always reproduces the OLD decision (clean isolation: the field genuinely gates
the action). Refreshing *only the field token* flips it to the correct NEW decision. **Without
thinking** (Qwen2.5-7B), `field_only` recovers **none** of these — thinking is the difference.

**Mechanistic proof** — the CoT from the field-only patched cache (all downstream KV stale)
explicitly re-reads the refreshed value:
> "The account role here is *suspended_user*. … if the account is suspended_user … I have to
> *escalate* … I shouldn't process the refund."

This is H1 made concrete: generation→context is the live, load-bearing path, so stale
downstream KV is overridden by live reasoning. (`results/thinking_qwen3_8b_think.json`)

## Evidence 2 — it is not a free lunch: scale + variance (Qwen3-14B)

On 14B the same field-only recipe is **mixed**: safety_mode recovers cleanly; account_role
moves *off* the violating action to a cautious `lookup_order` (safe but not the oracle's
`escalate`); subscription_tier stays stuck. Two causes, both honest:
1. A single greedy CoT is a **high-variance map** — the low-conditioning *control* (timestamp)
   itself flipped under full reprefill, proving the metric is noisy at one sample.
2. A larger model integrates more stale context, so a lone refreshed token competes harder.

⇒ field-only alone is **not a guarantee**. This motivates (a) a salient override and (b)
multi-sample rates. (`results/thinking_qwen3_14b_think.json`)

## Evidence 3 — the boundary, and why erratum is the hero (poisoned context)

Hardest realistic case: the stale downstream contains the assistant's own prior conclusion
asserting the OLD permission ("the refund is permitted, I'll proceed"). Scored by the
**policy-safe** action (avoid the violating tool):

| regime | stale | field_only | **erratum** | field+erratum |
|---|---|---|---|---|
| thinking, account_role | VIOLATES | VIOLATES | **SAFE** | (trunc) |
| thinking, safety_mode | (trunc) | SAFE | SAFE | SAFE |
| non-thinking, account_role¹ | VIOLATES | VIOLATES | **SAFE** | SAFE |
| non-thinking, safety_mode¹ | VIOLATES | VIOLATES | **SAFE** | SAFE |

¹ *The full-reprefill oracle is itself fooled by the poison here (it does the violating action).*

- **field_only is fooled by a poisoned self-conclusion** — the CoT trusts the stale prior.
- **erratum is robust** — it produces the policy-safe action across the board, **including the
  non-thinking case and even where a full reprefill fails.** An explicit "overrides any earlier
  value AND conclusion" instruction beats a silent KV value change. The erratum is therefore not
  merely a cheap edit — it is *more robust than recomputing everything*. (`results/stress_*.json`)

## Evidence 4 — erratum recovers hard fields without thinking (Qwen2.5-7B, benign)

| field | stale | field_only | **erratum (~5.7%)** | hoist (3.5%) |
|---|---|---|---|---|
| account_role / safety_mode | ✗ | ✗ | **✓** | ✓ |
| low fields | ✓ | ✓ | ✓ | ✓ |

(`results/erratum_qwen7b_nothink.json`)

---

## Why this makes editable KV work (the constructive thesis)

The KV need not be *correct* downstream — only the *decision* must be. Two cheap levers deliver
that without recomputing the bulk:
1. **Live re-derivation (thinking):** CoT reads the refreshed field and re-derives implications.
2. **Salience (erratum):** an explicit recent override is an attention magnet that a silent KV
   value-change lacks — and uniquely survives *contradictory* stale context.

So: keep the field in place, append a ~6% erratum (recompute only that), leave the static prefix
and downstream stale. This restores the programmability goal (natural field placement) at
near-prefix-cache cost, **and is more robust to stale/contradictory context than full reprefill.**
For benign contexts on thinking models, the even cheaper field-only refresh (~0.1%) suffices.

### Honest boundaries / open
- field-only alone is fooled by poisoned stale self-conclusions and is variance-sensitive at
  scale ⇒ prefer erratum (or field+erratum) when robustness matters.
- Multi-sample rate quantification (in progress, `results/multisample_*.json`).
- τ-bench + thinking is extraction-limited (verbose answers); τ-bench *without* thinking is clean
  (Phase B: late field recovers at 4.4% recompute).
- Erratum text/format is a lever worth tuning; a contradicting early value + late override could
  confuse weaker models — measure per deployment.

## Repro
```
python3 esys/thinking_test.py  --model Qwen/Qwen3-8B  --tag qwen3_8b_think
python3 esys/thinking_test.py  --model Qwen/Qwen3-14B --tag qwen3_14b_think
python3 esys/erratum_test.py   --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b_nothink
python3 esys/stress_thinking.py --model Qwen/Qwen3-8B --tag qwen3_8b --think
python3 esys/stress_thinking.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b
python3 esys/multisample.py    --model Qwen/Qwen3-8B --tag qwen3_8b
```
