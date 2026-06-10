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
python analyze.py && python make_figs.py
```
All runs are local on one RTX PRO 6000 (shared); models ≤8B (GPU-budget note in DESIGN.md).
