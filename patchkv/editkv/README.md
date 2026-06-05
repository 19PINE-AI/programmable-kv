# editkv — Editable KV cache for mutable fields in agentic contexts

Change a field inside an **already-cached** prompt without re-prefilling the whole thing.
Two mechanisms, a per-edit diagnostic to choose between them, and baselines — all over any
🤗 Transformers causal LM.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from editkv import EditableContext, Mode
from editkv.diagnostics import needs_erratum

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="cuda", dtype="bfloat16")

ctx = EditableContext(model, tok)
ctx.add_text(POLICY + "\nSESSION\naccount_role: ")
status = ctx.add_field("account_role", "verified_admin")   # declare a mutable field
ctx.add_text(CONVERSATION)
ctx.prefill()                                              # one full prefill, once

# the field flips; decide the next action without re-prefilling:
ctx.generate("account_role", "suspended_user", Mode.ERRATUM, decision_prompt="\nDecision:")
# -> "escalate"   (correct; an in-place-only edit would have stayed "refund")

needs_erratum(ctx, "account_role", "suspended_user").needs_erratum  # -> True
ctx.generate("account_role", "suspended_user", Mode.AUTO, decision_prompt="\nDecision:")  # picks erratum
```

## The two incarnations

| mode | what it does | cost | when it's enough |
|---|---|---|---|
| `Mode.IN_PLACE` | recompute only the changed field's KV (exact — it attends only to the unchanged prefix) and overwrite it; leave the rest stale | ~field tokens (~0.1%) | low-conditioning fields (time/ids/counters); reasoning models in benign contexts |
| `Mode.ERRATUM` | leave the cache stale; append a salient trigger (`[STATE UPDATE] <field> → <new>; overrides any earlier value and conclusion`); recompute only that span | ~tens of tokens (few %) | **robust** — every model family/scale and even under contradictory context; length-agnostic |
| `Mode.FIELD_PLUS_ERRATUM` | both | ~field + trigger | belt-and-suspenders |
| `Mode.AUTO` | run the diagnostic, pick in_place or erratum per-edit | +1 short probe | when you want the cheapest *correct* option automatically |
| `Mode.STALE` / `Mode.FULL_REPREFILL` | baselines (floor / ceiling) | 0 / 100% | evaluation |

**Why in-place alone often fails:** the decision token attends only ~0.1% to the field
directly and ~50% to downstream tokens that *memoized the field's implications at prefill
time*. Refreshing the field's KV doesn't update that memoized downstream, so the decision
reverts to the old value. The erratum injects a recent, explicit override the decision
attends to — which is why it's robust. (See `../MECHANISM.md`, `../PAPER.md`.)

## The diagnostic — do I actually need the erratum?

```python
d = needs_erratum(ctx, "order_status", "delivered", probe="\nDecision:")
d.needs_erratum            # True if in-place would revert to a different decision than the erratum
d.in_place_decision, d.erratum_decision, d.stale_decision
d.logit_drift              # cheap pre-filter: cosine drift stale->in_place at the decode position
d.in_place_available       # False for length-changing edits (use erratum)
```

It decodes the next decision under the in-place edit and under the erratum; if they
disagree, the field conditions the decision through the stale downstream and you need the
erratum. Both caches are cheap, so it's a ~2-short-decode runtime check. `blast_radius()`
is an even cheaper one-forward pre-filter.

## Notes / limitations
- `IN_PLACE` requires the new value to tokenize to the field's length (length-preserving);
  otherwise it raises `LengthChangeError` — use `ERRATUM` (length-agnostic). `AUTO`/`ERRATUM`
  handle length changes transparently.
- Built on HF `DynamicCache`; the per-edit cache is cloned for safety. For production
  throughput, integrate the edit (in-place KV overwrite is ~0.16 ms) + suffix-recompute into a
  paged-attention serving engine (vLLM/SGLang) — the edit itself is trivially cheap; the cost
  is the partial recompute, which is a few percent vs a full reprefill.
- `trigger_template` is configurable on `EditableContext`.

Run the demo: `python -m editkv.example Qwen/Qwen3-8B`
