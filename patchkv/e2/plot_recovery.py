"""Plot the E2 recovery contract: recent-window refresh fraction vs decision recovery."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
CLS_COLOR = {"low": "#2c7fb8", "medium": "#d95f0e", "high": "#c51b8a"}


def main(tag):
    recs = json.load(open(os.path.join(RES, f"recovery_{tag}.json")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # left: recovery vs recent-window fraction
    for r in recs:
        xs = [p["frac_down"] * 100 for p in r["sweep"]]
        ys = [1 if p["tracks"] else 0 for p in r["sweep"]]
        axes[0].plot(xs, ys, marker="o", color=CLS_COLOR[r["cls"]], alpha=0.8,
                     label=f"{r['scenario']} [{r['cls']}]")
    axes[0].set_xlabel("recent-window refresh (% of downstream tokens)")
    axes[0].set_ylabel("decision == oracle_new (1=recovered)")
    axes[0].set_title(f"E2 recovery vs sparse recent-window refresh — {tag}")
    axes[0].set_yticks([0, 1]); axes[0].set_xlim(-1, 30)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # right: min recover fraction bar
    names = [r["scenario"] for r in recs]
    mins = [(r["min_recover_frac"] or 0) * 100 for r in recs]
    cols = [CLS_COLOR[r["cls"]] for r in recs]
    axes[1].bar(range(len(names)), mins, color=cols, edgecolor="k")
    for i, r in enumerate(recs):
        lbl = "0% (unchanged)" if not r["decision_changed"] else f"{mins[i]:.1f}%"
        axes[1].text(i, mins[i] + 0.5, lbl, ha="center", fontsize=8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([f"{n}\n[{r['cls']}]" for n, r in zip(names, recs)],
                            rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("min recent-window refresh to recover decision (%)")
    axes[1].set_title("Per-field refresh contract (sparse residual)")
    axes[1].grid(alpha=0.3, axis="y")
    p = os.path.join(PLOTS, f"recovery_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    print(p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b")
