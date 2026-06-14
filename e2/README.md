# e2 — recovery, selective recompute, and long-horizon scenarios

Builds on `e1/`: once we know the conclusion is memoized downstream, e2 measures how
much of the decision each repair recovers, and stress-tests it over long trajectories
and a real agentic benchmark.

| file | what it does |
|------|--------------|
| `scenarios.py`        | the gated-decision scenario library |
| `run_recovery.py` / `plot_recovery.py` | decision recovery vs. how much downstream is recomputed |
| `run_selection.py` / `plot_selection.py` | `field+selective@K`: which K downstream notes to recompute |
| `run_horizon.py`      | long-trajectory no-compounding test (leave-stale + erratum vs. full reprefill) |
| `run_e2.py`, `run_e2b.py`, `run_e2c.py` | end-to-end edit/recover drivers |
| `run_taubench.py`, `taubench_ctx.py` | the tau2-bench retail agentic scenario |

```bash
# from the repo root
python e2/run_recovery.py --model Llama-3.1-8B
python e2/plot_recovery.py
```

Records land in `results/`. See §4 (editing) and the long-horizon appendix of the paper.
