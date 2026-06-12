"""E-horizon figure (fig9_horizon.pdf): no compounding error over a long trajectory.
Run: python paper/figs/make_horizon_figure.py   (cwd = repo patchkv/)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.dirname(__file__)
def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelsize": 8.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8, "axes.edgecolor": "#3a3a3a",
    "lines.linewidth": 1.7, "lines.markersize": 3.2, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.6, "grid.color": "#8a8a8a",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7"}
FAM = [("qwen3_8b", "Qwen3-8B", C["orange"]), ("llama31_8b", "Llama-3.1-8B", C["blue"]),
       ("mistral_7b", "Mistral-7B", C["green"])]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for tag, name, c in FAM:
        d = J(f"editkv_horizon_{tag}_p1.json")
        if not d:
            continue
        pt = d["summary"]["per_turn"]
        ts = [a["t"] for a in pt]
        axes[0].plot(ts, [a["agree"] for a in pt], "o-", color=c, label=name, alpha=0.9)
        axes[1].plot(ts, [a["logit_cos"] for a in pt], "o-", color=c, label=name, alpha=0.9)
    axes[0].set_xlabel("turn (steps after first patch)"); axes[0].set_ylabel("patched=oracle agreement")
    axes[0].set_title("(a) decision agreement vs trajectory length"); axes[0].set_ylim(0.4, 1.04)
    axes[0].legend(loc="lower left", fontsize=6.6)
    axes[1].set_xlabel("turn (steps after first patch)"); axes[1].set_ylabel("decision-logit cosine")
    axes[1].set_title("(b) decision logits stay faithful (no drift)"); axes[1].set_ylim(0.95, 1.002)
    axes[1].legend(loc="lower left", fontsize=6.6)
    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, "fig9_horizon.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
