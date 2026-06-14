# Interactive paper companion

A single-page interactive essay for *Models Take Notes at Prefill: KV Cache Can Be
Editable and Composable* — transformer-circuits-style visualizations of every study in
the paper, driven entirely by the released result records.

## Build & view

```bash
# 1. (optional) regenerate the curated site data from the result records
python3 data/build_data.py        # reads ../results, ../mem/results, ../e1, ../e2
                                  # writes src/data/*.json and prints a 22-row
                                  # assertion table checking extracted values
                                  # against the paper's stated numbers

# 2. build / preview
npm install
npm run build                     # tsc + vite -> dist/ (static, host anywhere)
npm run preview                   # serve dist/ locally
npm run dev                       # dev server with hot reload
```

The generated `src/data/*.json` are checked in, so `npm run build` works without
running the Python step.

## Data fidelity

- Every chart reads from `src/data/*.json`, extracted 1:1 from
  `results/*.json`, `mem/results/`, and the run logs
  (`comp_div_*.log`, parsed).
- Prompts in the explorer are regenerated verbatim by the deterministic harness
  builders (`e1/contexts.py`, `e2/scenarios.py`).
- The handful of values that exist only in the paper text (no released record) live in
  `src/data/constants.json` with an explicit `source` field, and render with a
  "⊙ from paper text" badge.
- Recorded model outputs are shown exactly as stored (tool call, thinking-token count,
  truncated answer head) — never reconstructed.
- `build_data.py` exits non-zero if any extracted headline number drifts from the
  paper's claimed value.

## Attribution

Author: **Bojie Li** (Pine AI). Code and result records:
<https://github.com/19PINE-AI/programmable-kv>. Deployed at
<https://01.me/research/programmable-kv/>.
