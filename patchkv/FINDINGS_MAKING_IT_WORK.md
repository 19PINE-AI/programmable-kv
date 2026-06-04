# Making editable KV work: refresh the field token, leave the rest stale, let the model think

> The earlier phases concluded (for **non-thinking** decoding) that leave-stale fails
> for early-gated fields and hoist-to-end wins. That conclusion was an artifact of an
> unrealistic setup: **the model under test did not think.** Real tool-calling agents
> reason before acting. Re-run with thinking enabled, the result reverses and the
> mechanism works — cheaply. Qwen3-8B (thinking), Qwen2.5-7B-Instruct (non-thinking).

---

## The recipe that works

**Thinking models (modern agents):** refresh **only the edited field's token KV**
(exact, ~0.1% of the context), leave *everything downstream stale*, and let the model
produce its normal chain-of-thought. The CoT re-reads the now-correct field value and
re-derives the implication live, so stale downstream KV is harmless.

**Non-thinking models:** leave the whole cache stale and append a short, salient
**erratum** at the suffix (`[STATE UPDATE] <field> has changed to <new>; this
overrides any earlier value`); recompute only those ~tens of tokens (~5–6%).

Both keep the field in its natural place (no prompt restructuring) and reuse the
entire static prefix for free.

---

## Evidence 1 — thinking makes the cheapest patch correct (Qwen3-8B)

Decision = post-`</think>` tool call. `field_only` = refresh only the field token KV
(~0.1% recompute), all else stale. `stale_full` = refresh nothing.

| field | class | stale_full (refresh 0) | **field_only (~0.1%)** | oracle_new |
|---|---|---|---|---|
| account_role | high | lookup_order ✗ | **escalate ✓** | escalate |
| safety_mode | high | share_payment_method ✗ | **refuse ✓** | refuse |
| subscription_tier | medium | expedite_shipping ✗ | **refuse ✓** | refuse |
| timestamp | low | issue_refund ✓ | issue_refund ✓ | issue_refund |

- **Clean isolation:** `stale_full` always reproduces the OLD decision (the field
  genuinely gates the action; the change is not inferred from elsewhere). Refreshing
  *only the field token* flips it to the correct NEW decision in every case.
- Contrast with **non-thinking** Qwen2.5-7B, where `field_only` recovers **none** of
  these (needs ~full reprefill). Thinking is the difference.

**Mechanistic proof** — the CoT from the field-only patched cache (all downstream KV
stale) explicitly reads the refreshed value:
> "The account role here is *suspended_user*. … the access rule says if the account is
> suspended_user … I have to *escalate* to the trust queue … I shouldn't process the
> refund."

The reasoning re-integrates the field at decode time even though the gating rule's KV
still encodes the old `verified_admin` interpretation. This is exactly H1
(generation→context is the live, load-bearing path) made concrete.

**Cost framing:** the CoT tokens (≈250–1900 here) are *not* charged to the patch — a
thinking agent emits them anyway. On top of normal operation the edit costs ~0.1%
prefill recompute vs a full reprefill, while preserving the decision.

(`results/thinking_qwen3_8b_think.json`, `esys/thinking_test.py`)

## Evidence 2 — erratum recovers without thinking (Qwen2.5-7B)

| field | stale | field_only | **erratum (~5.7%)** | hoist (3.5%) |
|---|---|---|---|---|
| account_role | ✗ | ✗ | **✓** | ✓ |
| safety_mode | ✗ | ✗ | **✓** | ✓ |
| subscription_tier | ✗ | ✗ | (extraction artifact — re-checking) | ✓ |
| timestamp / request_id (low) | ✓ | ✓ | ✓ | ✓ |

The erratum recovers the hard high-conditioning fields that a bare field-token refresh
cannot (without thinking), at ~5.7% recompute — comparable to hoist-to-end but
**without restructuring the prompt**, and it works even when a field appears in
multiple places (un-hoistable). (`results/erratum_qwen7b_nothink.json`,
`esys/erratum_test.py`)

---

## Why this reframes the whole project (constructive, not a refutation)

The KV does not need to be *correct* downstream — only the *decision* must be. Two
cheap levers achieve that without recomputing the bulk:
1. **Live re-derivation (thinking):** the CoT reads the refreshed field and re-derives
   implications; stale downstream KV is overridden by live reasoning.
2. **Salience (erratum):** an explicit recent override instruction acts as an attention
   magnet that refreshed-but-not-salient KV lacks.

So editable KV **works**: keep the field in place, refresh ~0.1% (field token) for
thinking models or append a ~6% erratum for non-thinking models, leave the entire
static prefix and downstream stale. This restores the programmability goal (natural
field placement) at near-prefix-cache cost.

## Open / in-progress validations
- Scale: Qwen3-14B (thinking) — re-run field_only recipe (downloading).
- Realism: τ-bench retail policy + thinking.
- Robustness: multiple simultaneous edits; field buried deep; edit magnitude.
- Fix the subscription_tier erratum tool-extraction artifact (formatting, not mechanism).

## Repro
```
python3 esys/thinking_test.py --model Qwen/Qwen3-8B --tag qwen3_8b_think        # thinking + field_only
python3 esys/erratum_test.py  --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b_nothink   # erratum, non-thinking
python3 esys/erratum_test.py  --model Qwen/Qwen3-8B --tag qwen3_8b_think --think      # erratum + thinking
```
