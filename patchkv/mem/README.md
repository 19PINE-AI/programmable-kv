# Editable & Composable User Memory in the KV Cache

Extension of *"Models Take Notes at Prefill"* to the **user-memory** application: a large,
dynamically-mutated Markdown profile that an agent re-reads every turn. We precompile the
memory once, place it late, RoPE-reposition it each turn, and edit it in place — making
user memory both **composable** (precompute + splice) and **editable** (erratum / recompile)
in the KV cache.

See `DESIGN.md` (framing + related-work survey + experiment design), `PREREG.md`
(pre-registered hypotheses/margins), `CALIBRATION.md` (why CoT is the competent regime),
and `FINDINGS_MEMORY.md` (results).

## The application — `app.py`

```python
from app import MemoryAgent
agent = MemoryAgent(model, tok, system_prompt, memory_markdown)   # precompiles memory once
agent.add_turn("User: ...\nAssistant: ...")                       # trajectory grows (prefix-cached)
agent.update_memory(new_markdown, mode="recompile")               # or mode="erratum" (append-only)
out = agent.decide("Should I do X? ...", cot=True)                # out["decision"], out["ttft_ms"]
```

Layout `[system][trajectory][MEMORY][query]`: the system prompt is prefilled once, the
trajectory grows by cached deltas, the memory chunk is precompiled in isolation and
re-rotated to float just before the query each turn (O(L_mem), no re-prefill), and a memory
change is applied by recompiling the chunk (O(L_mem), once) or appending a salient erratum
(O(tokens), composes with prefix caching).

## Code map
- `data.py` — controlled persona/memory generator (gated decisions, integration depth, edits).
- `memkv.py` — core harness: early/late layouts, full-recompute vs precompiled transplant,
  seam-repair, and all edit methods (stale/in_place/erratum/recompile_chunk/selective@K).
- `stats.py` — cluster bootstrap, McNemar-exact, TOST equivalence, GEE-logistic, BH-FDR, power.
- `run_e1.py` placement×pre-digestion · `run_e2.py` transplant equivalence · `run_e3.py`
  editing · `run_e4.py` granularity · `run_e5.py` systems · `app.py` the agent.
- `analyze.py` — all statistics → `results/summary.json`. `make_figs.py` — figures → `figs/`.

## Reproduce
```bash
bash run_e2_all.sh        # faithfulness/equivalence across models
bash run_rest.sh          # E1, E3, E4, E5, then analyze + figures (waits for E2)
bash run_large.sh         # 14B-70B sweep (competence, E2, E3, E5)
bash run_locomo_full.sh   # LoCoMo external validity: all 1,540 Q, full ~20k-token memory (flash)
bash run_retry_robust.sh  # re-run any GPU-contention OOM failures (waits for free memory, backoff)
python analyze.py && python make_figs.py
```
Runs are local on one RTX PRO 6000 (96 GB, **shared/intermittently contended** — see below).
LoCoMo needs `results/locomo10.json` (download: snap-research/locomo `data/locomo10.json`).

**GPU notes (important for large/long-context runs):**
- Long-context (LoCoMo, ≥16k tokens) requires **flash attention 2** (sdpa falls back to an
  O(L²) math backend and OOMs) and **`logits_to_keep=1`** in prefills (the all-position logit
  tensor is 7–15 GB of pure waste — only the KV cache is needed). Both are wired into `run_locomo.py`.
- The node is shared; other jobs grab/free 20–60 GB unpredictably. Use `run_retry_robust.sh`,
  which waits for a free-memory window before each launch and retries on OOM with backoff.
