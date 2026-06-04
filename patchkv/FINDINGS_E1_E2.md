# PatchKV — E1 + E2 Go/No-Go Findings

> Status: **GO** (with a refined thesis). Run autonomously per `editable-kv-research-plan.md`.
> Date: 2026-06-04. Hardware: 1× RTX PRO 6000 Blackwell (96 GB), torch 2.10 / transformers 4.57.
> Models: Qwen2.5-1.5B (base) and Qwen2.5-7B-Instruct. Synthetic-controlled contexts.

---

## TL;DR

1. **H2 (causally-exact region) — confirmed, exactly.** Every token positioned *before* the
   edited field has bit-identical KV under old vs. new (max deviation `0.0` across all layers,
   all fields, both models). The region before an edit is provably free to reuse. This is the
   load-bearing structural claim and it holds without caveat.
2. **H4 (leave-stale preserves decisions for low-conditioning fields) — confirmed for the target
   class.** For timestamps, request-ids, session counters, nonces: the agent's decision is
   invariant to the flip, and leave-stale with **zero downstream refresh** reproduces the oracle
   decision. This is exactly the common case the thesis targets.
3. **H4 boundary — confirmed.** For decision-relevant fields (account role, safety mode,
   subscription tier) the correct decision *does* flip, and refreshing only the field's own KV
   **fails** to track it. A **sparse recent-window residual** recovers it; the required size is
   field-dependent (0.9 % → 27 % of downstream tokens), while the bulk static context stays stale.
4. **Refinement to the mechanism (new):** the residual that must be refreshed is a **recent
   window** — the model's own latest tokens that integrated the old value — **not** the distant
   gating rule. Refreshing field + gating-rule (≈2.5 %) did *not* recover; refreshing field +
   last-K recent tokens did. This is a cleaner and cheaper refresh target than scattered
   top-deviation selection.

**Decision: proceed past the E1 go/no-go.** The blast radius is sparse *and* field-dependent
(GO condition met), and the leave-stale-safe class is non-empty and exactly the common case.

---

## E1 — Blast-radius characterization

**Method.** Length-preserving field flips (token-aligned; only the value span differs). Two
forward passes (OLD, NEW) capture per-layer **post-RoPE** Q/K/V via a monkeypatched eager
attention path (keeping the correct additive causal mask — see *Gotcha* below). Metrics per
(layer, token): KV cosine/relative-L2 deviation, and the CacheBlend-style **attention-output
deviation** (hold the query at oracle/NEW; vary only whether downstream K/V is stale). Headline
BR(τ) = fraction of *downstream, non-field* tokens whose max-over-layers attention-output
deviation exceeds τ.

**Sanity (H2).** Causally-exact region (positions before the field) deviation = `0.0` exactly in
every run. Position 0 (BOS) is bit-identical across all layers. This caught a real bug early: an
unknown `_attn_implementation` name suppressed the causal mask (silent bidirectional attention);
the fix is to keep `"eager"` and monkeypatch the module global.

**Headline numbers — mean BR(τ) over downstream tokens, semantic flips:**

| model | class | τ=0.05 | τ=0.10 | τ=0.20 |
|---|---|---|---|---|
| 1.5B | low | 24.3 % | **2.6 %** | 0.2 % |
| 1.5B | medium | 43.9 % | 6.0 % | 0.3 % |
| 1.5B | high | 38.2 % | 9.4 % | 2.7 % |
| 7B | low | 62.4 % | **9.5 %** | 0.9 % |
| 7B | medium | 96.7 % | 30.8 % | 5.9 % |
| 7B | high | 97.8 % | 53.8 % | 13.8 % |

- **Sparse + field-dependent**: at a moderate threshold the blast radius is a small, *class-ordered*
  fraction (low < medium < high), with a 3.6× (1.5B) → 5.6× (7B) high/low separation at τ=0.1.
- **Structure** (`plots/P2P3_*`): deviation is ≈0 before the field, spikes just after it, and
  **decays with distance**; it **grows with layer depth**, peaking in the mid-upper layers.
- **Caveat — threshold/model sensitivity:** at small τ *almost everything* moves a little, and the
  absolute BR is *larger* for 7B than 1.5B. Raw KV-deviation BR therefore **overcounts** and is not
  a model-portable safety threshold on its own. The contract must be calibrated to **decisions**
  (E2), where the picture is much cleaner.

Plots: `plots/P1_{tag}_semantic_attn.png` (BR curves), `plots/P2P3_{tag}_semantic.png`
(position & layer), `plots/P4_{tag}.png` (per-field p95).

---

## E2 — Decision-flip faithfulness (the result with teeth)

**Method.** Build OLD cache; form the leave-stale **patched cache** (= OLD with chosen spans
overwritten by NEW; everything else stale); greedily decode the next tool-call decision. Compare
to **oracle_new** (full new prefill), **oracle_old**, and **stale_full** (refresh nothing). The
patcher is validated: *refresh all downstream* reproduces oracle_new exactly in every case.

**First pass (generic trajectory):** PATCHED == ORACLE_NEW in 100 % of fields — but the decision
*never changed*, so the field was decision-irrelevant. No teeth. Re-ran with **engineered
decision-relevant scenarios** where one field gates the correct action and the gating rule sits
**after** the field (so its KV genuinely goes stale).

**Decision-relevant results (Qwen2.5-7B-Instruct):**

| scenario | class | decision changed? | field-only leave-stale tracks? | min recent-window refresh to recover |
|---|---|---|---|---|
| timestamp | low | no | ✔ (trivially) | **0 %** |
| request_id | low | no | ✔ (trivially) | **0 %** |
| safety_mode | high | yes | ✗ | **0.9 %** |
| subscription_tier | medium | yes | ✗ | **6.9 %** |
| account_role | high | yes | ✗ | **27.4 %** |

- **Low-conditioning fields**: decision invariant; leave-stale safe with no refresh. H4 holds for
  the target class.
- **Decision-relevant fields**: field-only leave-stale collapses to the *old* decision
  (== stale_full == oracle_old). They require a residual refresh — but only a **recent window**;
  the long static policy block (the bulk of the context) stays stale.
- **Recency, not the gating rule**, is what matters: refreshing field + gating-rule (~2.5 %) did
  not recover; refreshing field + the last-K recent tokens did. The needed K is field-dependent
  and bounded (here ≤27 % of downstream).

Plot: `plots/recovery_qwen7b.png`. Data: `results/e2b_qwen7b.json`, `results/e2c_qwen7b.json`,
`results/recovery_qwen7b.json`.

---

## Hypothesis scorecard

| Hyp. | Claim | Verdict |
|---|---|---|
| H1 | load-bearing cross-attention is generation→context, recomputed live | Supported indirectly: refreshing the *recent* window (what the live decode reads) is what recovers decisions. |
| H2 | region before the edit is exactly reusable | **Confirmed exactly** (deviation 0.0). |
| H3 | residual ≪ CacheBlend's ~15 %, possibly ≈0 | **Mixed**: ≈0 (low), ~7 % (medium) — both < 15 %; but account_role needs 27 % > 15 %. Not universally ≪. |
| H4 | leave-stale preserves decisions for a characterizable low-conditioning class | **Confirmed** for low; **boundary confirmed** for high (must refresh a recent residual). |

---

## Go / No-Go (per plan §E1 decision rule)

- NOT trivial (BR ≉ 0 for all) and NOT hopeless (BR ≉ large for all). ✔
- Low-conditioning fields are cheap (0 % residual at the decision level) and clearly separated from
  high (0 % vs 27 %). ✔
- **Verdict: GO.** The leave-stale-safe class is non-empty and is exactly the common case
  (time/ids/counters/nonces). The mechanism contribution sharpens to: *reuse the causally-exact
  prefix for free + refresh the field + refresh a sparse recent window; leave the static bulk
  stale.*

### Recommended thesis refinements before E-sys
1. **Lead metric = decision faithfulness, not raw KV-deviation BR.** Raw BR overcounts and is not
   model-portable; report it as a cheap predictor, calibrate τ\* against decisions.
2. **Reframe the residual as a recent window**, not scattered top-deviation tokens. This is cheaper
   to predict (known location: end of context) and matches the "no mandatory layer-1 probe" goal.
3. **Soften H3**: residual is ≈0 for low-conditioning fields (the pitch) but can exceed CacheBlend's
   15 % for strongly-gating fields. Position the win on the *common case* + the *free exact prefix*.

---

## Limitations / next steps (toward E-prog, E-boundary, E-horizon)
- Synthetic-controlled contexts only; **τ-bench** (retail/airline) not yet wired in.
- Two model sizes; greedy single-decision point. **E-horizon** (compounding over many steps) untested.
- Decision metric = first tool-call line agreement; free-form argument agreement is coarse.
- Recovery sweep used a contiguous recent window; a deviation-ranked residual (tie to E1) is the
  natural next refinement and would tighten the account_role contract.

## Reproduce
```
# E1 blast radius (per model)
python3 e1/run_e1.py  --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b --magnitudes semantic,minor
python3 e1/plot_e1.py qwen7b
# E2 decision-flip + recovery contract
python3 e2/run_e2b.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b --chat
python3 e2/run_recovery.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b
python3 e2/plot_recovery.py qwen7b
```
