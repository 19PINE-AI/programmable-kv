"""Plot selection-policy recovery curves: deviation vs recency vs random."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
POL_STYLE = {"deviation": ("#c51b8a", "o"), "recency": ("#2c7fb8", "s"),
             "random": ("#999999", "^")}


def main(tag):
    recs = json.load(open(os.path.join(RES, f"selection_{tag}.json")))
    changed = [r for r in recs if r["decision_changed"]]
    n = len(changed)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, r in zip(axes[0], changed):
        for p, (c, m) in POL_STYLE.items():
            xs = [s["frac"] * 100 for s in r["sweep"][p]]
            ys = [1 if s["tracks"] else 0 for s in r["sweep"][p]]
            ax.plot(xs, ys, marker=m, color=c, label=p, alpha=0.85)
        ax.set_title(f"{r['scenario']} [{r['cls']}]", fontsize=10)
        ax.set_xlabel("refresh fraction of downstream (%)")
        ax.set_yticks([0, 1]); ax.set_ylim(-0.1, 1.1); ax.set_xlim(-2, 80)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("decision recovered (1=yes)")
    fig.suptitle(f"Phase A: residual selection policy — {tag} "
                 "(recency recovers cheapest; deviation-ranked is not best for edits)",
                 fontsize=11)
    p = os.path.join(PLOTS, f"selection_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    print(p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b")
