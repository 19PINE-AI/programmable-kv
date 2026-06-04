# PatchKV — Extensions (Phases A–D): selection, τ-bench, E-horizon, E-sys

> Follow-on to `FINDINGS_E1_E2.md`. Run autonomously on 1× RTX PRO 6000 Blackwell,
> Qwen2.5-7B-Instruct, transformers 4.57. Date 2026-06-04.
> **Headline: the honest verdict shifts toward a *characterization* paper.** The
> leave-stale mechanism is cheap-and-correct only when a field is placed *after* the
> rules that gate it; otherwise the hoist-to-end + prefix-cache baseline wins.

---

## ⚠️ Correction to Phase A (important)
An early version of the selection sweep ran with the Q/K/V capture hook **not
installed** (the `run_e2.load_model` path skipped `capture.install`), so the
deviation ranking silently degenerated to *field-proximity order* and produced a
wrong conclusion ("recency dominates deviation"). Fixed (`load_model` now always
installs the hook; verified 28/28 layers captured) and **all Phase A numbers below
are post-fix.** Lesson worth keeping: an unpopulated capture fails *silently* as a
plausible-looking ranking.

---

## Phase A — residual selection policy (deviation vs recency vs random)
Rank downstream tokens, refresh field + top-k%, find the minimal k that recovers
the oracle decision. Min recovery fraction (Qwen2.5-7B-Instruct):

| scenario | class | deviation-ranked | recency (suffix) | random |
|---|---|---|---|---|
| account_role | high | **30 %** | 50 % | 50 % |
| safety_mode | high | 5 % | **1 %** | 100 % |
| subscription_tier | medium | 5 % | **5 %** | 100 % |
| timestamp / request_id | low | 0 % | 0 % | 0 % |

**Finding: neither policy dominates; they are complementary.** Deviation-ranking
wins when the decision-critical stale tokens sit near the field (account_role's
early gating rule — a suffix window misses it); recency wins when they are recent.
Both beat random. A practical selector should take the **union** of high-deviation
and recent tokens. (`plots/selection_qwen7b.png`)

---

## Phase B — τ-bench-grounded realistic contexts
Real retail `wiki.md` policy (the rules) + a real order from the DB (status fields
in a tool observation) + the order-status field (`pending`↔`delivered`), which the
policy gates. In real agents the **policy precedes the field** (field arrives late
in an observation), so the policy is in the causally-exact region.

| scenario | exact-region dev | decision flips? | reuse-for-free | recompute to recover |
|---|---|---|---|---|
| order_status (high) | **0.0** | yes (`cancel_pending_order`→`transfer_to_human_agents`) | 94.8 % of context | **4.4 %** |
| current_date (low) | 0.0 | no | 2.2 % (field early) | **0.7 %** |

**Finding: H2 holds on the real 81-line policy (exact region 0.0).** A
high-conditioning field placed *late* (after its gating rules) recovers the correct
flipped decision while recomputing only 4.4 % — the rest is reused for free. A
low-conditioning field early in the prompt is fully leave-stale-safe (0.7 %).
(`results/taubench_qwen7b.json`)

---

## Phase C — E-horizon (compounding over a 5-step trajectory)
Flip the field once, patch, roll 5 sequential gated requests forward without ever
refreshing the stale base. Tool-name agreement vs the full-reprefill oracle:

- **Low field (timestamp): 100 % agreement, flat across all 5 steps — no drift.**
- **High field (account_role): disagreement is localized to exactly the steps where
  the field changed the oracle decision; it does not snowball into later steps.**

Mechanistically (supports H1): each step's decision is driven by the *live* recent
context (the user turn + freshly-decoded tokens), which is computed against the
patched cache, so the fixed initial staleness does not compound as the trajectory
grows. (`plots/horizon_qwen7b.png`)

---

## Phase D — E-sys mechanism + cost/quality/latency frontier
Faithful mechanism (no oracle-copying): **field refresh is exact** (recompute the
field tokens against the identical prefix → cosine 0.99989 vs oracle, gap is bf16);
the residual is recomputed against the stale base. Baselines include the real one
to beat — **hoist-to-end + prefix caching**.

**Decision-relevant, EARLY-gated synthetic fields** (frontier plot):

| method | recompute | latency | recovers? |
|---|---|---|---|
| full_reprefill | 99.9 % | ~128 ms | ✔ (ceiling) |
| **hoist_to_end** | **3.5 %** | **~37 ms** | **✔** |
| patchkv faithful (recency k≤256) | 0–26 % | 20–70 ms | ✗ |
| stale_reuse | 0 % | ~0.3 ms | ✗ |

**Crucial faithful-vs-oracle-copy result:** the Phase-A recovery that worked used
*oracle-copied* residual KV (unrealizable). When the residual is honestly
*recomputed* against the stale base, a recency window **does not** recover an
early-gated field — the recomputed recency tokens still attend to the stale early
gating rule. Recovery then requires recomputing from the gating rule onward
(safety_mode: 93 % recompute), i.e. ≈ full reprefill. So for early-gated fields,
**hoist-to-end strictly dominates faithful PatchKV** on efficiency and correctness.

**Where PatchKV genuinely wins (late-placed field, τ-bench order status):** the
gating rules precede the field (causally exact, reused free), so faithful
field-refresh + recompute of the small post-field tail recovers the correct flipped
decision at **5.1 % recompute, reusing 94.8 % for free** — competitive, and without
restructuring the prompt. (`plots/frontier_qwen7b.png`, `esys/verify_faithful.py`)

---

## Synthesis & recommendation
The four phases converge on a single law:

> **Leave-stale is cheap *and* correct exactly when a field sits after the rules that
> condition on it** (so those rules are causally-exact and reused for free, and only
> the short post-field span needs refresh). When a field precedes its gating rules,
> faithful refresh must redo everything from the rule onward and there is no win over
> full reprefill — and hoist-to-end is both cheaper and correct.

Implications for the thesis:
1. **Lead with characterization, not a systems speedup.** The contribution that
   survives scrutiny is the *contract*: which (field-class × placement) combinations
   are leave-stale-safe, validated at the KV (E1), decision (E2), realistic (τ-bench),
   and horizon levels. The efficiency claim does **not** beat hoist-to-end for the
   hard case — this matches the plan's pre-registered fallback to a measurement paper.
2. **Programmability is the only defensible efficiency-adjacent win:** for
   low-conditioning fields you can leave the cache *fully* stale (0 % recompute, ~0 ms)
   with no decision change *and* no prompt restructuring — hoist-to-end needs the
   restructuring; PatchKV does not. But this win is confined to low-conditioning fields.
3. **Selection should be deviation∪recency**, not either alone (Phase A).
4. **No compounding over the horizon** (Phase C) is a genuine, clean positive worth
   reporting regardless of venue.

## Repro
```
python3 e2/run_selection.py  --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b   && python3 e2/plot_selection.py qwen7b
python3 e2/run_taubench.py    --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b --chat
python3 e2/run_horizon.py     --model Qwen/Qwen2.5-7B-Instruct --tag account_role --field account_role
python3 e2/run_horizon.py     --model Qwen/Qwen2.5-7B-Instruct --tag timestamp    --field timestamp
python3 esys/frontier.py      --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b       && python3 esys/plot_frontier.py qwen7b
python3 esys/verify_faithful.py --model Qwen/Qwen2.5-7B-Instruct
```
