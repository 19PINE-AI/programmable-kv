"""K* figure + table: P_safe vs K (field+selective under reasoning) per model, vs golden erratum.
Reads results/selective_Ksweep_*.json. Run: python esys/make_ksweep_figure.py qwen3_1p7b_par qwen3_4b_par ...
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
tags = sys.argv[1:] or ["qwen3_1p7b_par", "qwen3_4b_par", "qwen3_8b_par", "qwen3_14b_par"]
# order matters: check longer/more-specific keys first ("14b" before "4b", which is a substring)
names = [("1p7b", "Qwen3-1.7B"), ("14b", "Qwen3-14B"), ("4b", "Qwen3-4B"), ("8b", "Qwen3-8B")]


def nm(tag):
    for k, v in names:
        if k in tag:
            return v
    return tag


fig, ax = plt.subplots(figsize=(6.4, 4.0))
colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
print(f"{'model':12s} {'K=0':>6} {'K=4':>6} {'K=8':>6} {'K=16':>6} {'K=32':>6} {'K=64':>6} | {'errat':>6} {'full':>6} {'K*full':>6}")
for i, tag in enumerate(tags):
    p = os.path.join(R, f"selective_Ksweep_{tag}.json")
    if not os.path.exists(p):
        print(f"{nm(tag):12s} (missing)"); continue
    d = json.load(open(p))
    ks = sorted(int(k) for k in d["K_safe"])
    ys = [d["K_safe"][str(k)]["P_safe"] for k in ks]
    los = [d["K_safe"][str(k)]["ci"][0] for k in ks]; his = [d["K_safe"][str(k)]["ci"][1] for k in ks]
    c = colors[i % len(colors)]
    ax.plot(ks, ys, "o-", color=c, label=f"{nm(tag)} (n={d['n']})")
    ax.fill_between(ks, los, his, color=c, alpha=0.12)
    er = d["erratum_P_safe"]; full = d["full_P_safe"]
    ax.axhline(er, color=c, ls=":", lw=1, alpha=0.5)
    ax.axhline(full, color=c, ls="--", lw=1, alpha=0.5)
    # K* against the FULL-REPREFILL reference (the true recomputation upper bound)
    kstar_full = next((k for k in ks if d["K_safe"][str(k)]["P_safe"] >= full - 1e-9), None)
    row = [d["K_safe"][str(k)]["P_safe"] for k in [0, 4, 8, 16, 32, 64]]
    print(f"{nm(tag):12s} " + " ".join(f"{v:6.2f}" for v in row) + f" | {er:6.2f} {full:6.2f} {str(kstar_full):>6}")
ax.set_xscale("symlog", base=2, linthresh=1)
ax.set_xticks([0, 4, 8, 16, 32, 64]); ax.set_xticklabels(["0", "4", "8", "16", "32", "64"])
ax.set_xlabel("K = # downstream tokens recomputed in addition to the field")
ax.set_ylabel("P(safe) after CoT  (reasoning)")
ax.set_ylim(0, 1.05)
ax.set_title("field + selective@K under reasoning vs golden erratum (dotted)\nminimal K to recover reasoning safety")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig_ksweep.png")); plt.close(fig)
print("saved figures/fig_ksweep.png")
