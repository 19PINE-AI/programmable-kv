# PatchKV — editable KV cache

> **New to this? Start with [`EXPLAINER.md`](EXPLAINER.md)** — the whole project explained
> from scratch, no background assumed.
>
> **Technical summary: [`FINDINGS_FINAL.md`](FINDINGS_FINAL.md)** — the consolidated result.
> **Editable KV works:** keep the field in place → inject a cheap salient suffix *erratum*
> (~6%, recompute only that span; robust even to contradictory stale context) → leave the
> static bulk stale. On thinking models in benign contexts, a ~0.1% field-token refresh
> suffices (CoT re-reads the field). Both beat full reprefill on cost and hoist-to-end on
> programmability. Detailed phase docs: `FINDINGS_E1_E2.md`, `FINDINGS_EXTENSIONS.md`,
> `FINDINGS_MAKING_IT_WORK.md`.

# PatchKV — E1/E2 go/no-go harness

Autonomous implementation of the decisive first experiments from
`../editable-kv-research-plan.md`: blast-radius characterization (E1) and
decision-flip faithfulness (E2 slice). **Outcome: GO.** See `FINDINGS_E1_E2.md`.

## Layout
- `e1/capture.py` — post-RoPE Q/K/V capture (monkeypatches the eager attention global; keeps the correct causal mask).
- `e1/align.py` — length-preserving token alignment of OLD vs NEW (single contiguous field span).
- `e1/contexts.py` — synthetic agentic contexts + field taxonomy (low/medium/high conditioning).
- `e1/deviation.py` — KV deviation + CacheBlend-style attention-output deviation.
- `e1/run_e1.py`, `e1/plot_e1.py` — E1 driver and figures.
- `e2/scenarios.py` — decision-relevant scenarios where one field gates the action (gating rule placed AFTER the field).
- `e2/run_e2.py` — cache machinery (prefill, clone, patched leave-stale cache, greedy decode).
- `e2/run_e2b.py` — decision-flip: oracle_new vs patched (field-only) vs stale_full vs oracle_old.
- `e2/run_e2c.py` — residual-refresh recovery (field + gating-rule / window).
- `e2/run_recovery.py`, `e2/plot_recovery.py` — recent-window refresh contract sweep + figure.
- `results/` — JSON records + raw per-token npz. `plots/` — figures.

## Key result (E1/E2 go/no-go)
Region before an edit is exactly reusable (deviation 0.0). Low-conditioning fields
(time/ids/counters/nonces) are leave-stale-safe with zero downstream refresh.

## Extensions (Phases A–D) — see `FINDINGS_EXTENSIONS.md`
- `e2/run_selection.py` — Phase A: residual selection (deviation ∪ recency beats either alone).
- `e2/taubench_ctx.py`, `e2/run_taubench.py` — Phase B: real τ-bench retail policy + data.
- `e2/run_horizon.py` — Phase C: E-horizon (low field flat 100%; no compounding).
- `esys/mechanism.py`, `esys/frontier.py`, `esys/verify_faithful.py`, `esys/plot_frontier.py`
  — Phase D: faithful update + cost/quality/latency frontier vs hoist-to-end.

**Verdict:** leave-stale is cheap *and* correct only when a field sits AFTER the rules that gate
it (τ-bench late field: 5% recompute, 95% free reuse, recovers). For early-gated fields, faithful
PatchKV loses to hoist-to-end + prefix caching → lead with **characterization**, not a speedup.
NOTE: an early Phase-A run had the capture hook uninstalled (silently wrong ranking); fixed.
