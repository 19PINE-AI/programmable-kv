"""Frontier plot: recompute fraction vs latency, correctness encoded by marker fill."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
MARK = {"full_reprefill": "o", "stale_reuse": "X", "hoist_to_end": "*",
        "patchkv_k0": "s", "patchkv_k128": "^", "patchkv_k256": "D"}
COL = {"full_reprefill": "#444", "stale_reuse": "#d62728", "hoist_to_end": "#2ca02c",
       "patchkv_k0": "#1f77b4", "patchkv_k128": "#1f77b4", "patchkv_k256": "#1f77b4"}


def main(tag):
    recs = json.load(open(os.path.join(RES, f"frontier_{tag}.json")))
    changed = [r for r in recs if r["decision_changed"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for r in changed:
        for name, m in r["methods"].items():
            x = m["recompute_frac"] * 100; y = m["latency_ms"]
            correct = m["agree_oracle"]
            lbl = name if name not in seen else None
            seen.add(name)
            ax.scatter(x, y, marker=MARK.get(name, "o"), s=170 if name == "hoist_to_end" else 90,
                       facecolor=COL.get(name, "#888") if correct else "none",
                       edgecolor=COL.get(name, "#888"), linewidths=1.6, label=lbl, alpha=0.9)
    ax.set_xlabel("recompute fraction (% of full prefill)  — lower is cheaper")
    ax.set_ylabel("update latency (ms)  — lower is faster")
    ax.set_title(f"E-sys frontier on decision-relevant (EARLY-gated) fields — {tag}\n"
                 "filled = recovers oracle decision; hollow = wrong. "
                 "hoist★ wins; faithful PatchKV (□△◇) stays hollow until ~full reprefill.")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    p = os.path.join(PLOTS, f"frontier_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    print(p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b")
