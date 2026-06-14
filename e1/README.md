# e1 — blast-radius capture and the gated-decision setup

The earliest mechanism harness. It establishes the core puzzle: changing one field
inside a cached prefill leaves the prefix byte-identical (KV deviation `0.0`) yet a
surgical field-only refresh does **not** flip the decision.

| file | what it does |
|------|--------------|
| `contexts.py`   | builds the gated-decision prompts (policy rule + mutable field) |
| `capture.py`    | prefills a context and captures per-token / per-layer KV |
| `deviation.py`  | measures KV deviation before vs. after a field change |
| `align.py`      | aligns token positions across the stale/edited caches |
| `run_e1.py`     | entrypoint — runs the blast-radius / locality measurement |
| `plot_e1.py`    | renders the E1 figure from the run output |

```bash
# from the repo root
python e1/run_e1.py --model Qwen/Qwen3-8B
python e1/plot_e1.py
```

Records land in `results/`. See `docs/MECHANISM.md` for the conceptual account and
§3 of the paper for the formal probes. The more rigorous, multi-model version of these
probes lives in `esys/` (`mech_*.py`).
