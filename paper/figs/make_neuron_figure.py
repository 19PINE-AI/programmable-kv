"""Neuron-level figure (fig10_neurons.pdf) from the follow-up records in followup/results/.
Panels: (a) top-k neuron patch sufficiency vs layer-matched random (Llama-3.1-8B, Qwen3-8B), with the
whole-block attention/MLP patch levels; (b) concentration of |AtP| mass over neurons; (c) per-layer share
of |AtP|; (d) sign split: zeroing positive vs negative activations; (e) joint vs summed single ablation
per carrier type; (f) Gao's simulation-substitution test: real vs label-simulated patch.
Run: python paper/figs/make_neuron_figure.py   (cwd = repo root; CPU only)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "followup", "results")
OUT = os.path.dirname(__file__)
J = lambda n: json.load(open(os.path.join(R, n)))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelsize": 8.5, "legend.fontsize": 6.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#3a3a3a", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "lines.linewidth": 1.9, "lines.markersize": 4.5, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.6, "grid.color": "#8a8a8a",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#9a9a9a"}
MODELS = [("llama31_8b", "Llama-3.1-8B", C["blue"]), ("qwen3_8b", "Qwen3-8B", C["orange"])]
KS = [1, 3, 10, 30, 100, 300]


def panel_patch(ax):
    for tag, name, c in MODELS:
        cz = J(f"t1_neurons_{tag}.json")["causal"]
        ax.plot(KS, [cz[f"agg8/suff/k{k}"]["top_mean"] for k in KS], "o-", color=c, label=f"{name}: top-$k$")
        ax.plot(KS, [cz[f"agg8/suff/k{k}"]["rand_mean"] for k in KS], "x:", color=c, alpha=0.6, label=f"{name}: random")
    s = J("t1_split_llama31_8b.json")["summary"]
    ax.axhline(s["patch_all_attn_mean"], color=C["green"], ls="--", lw=1.1)
    ax.text(1, s["patch_all_attn_mean"] + 0.02, f"Llama: all attention outputs {s['patch_all_attn_mean']:.2f}", fontsize=6.4, color=C["green"])
    ax.axhline(s["patch_all_mlp_mean"], color=C["purple"], ls="--", lw=1.1)
    ax.text(1, s["patch_all_mlp_mean"] + 0.02, f"Llama: all MLP outputs {s['patch_all_mlp_mean']:.2f}", fontsize=6.4, color=C["purple"])
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("# MLP neurons patched (top-$k$ by AtP)"); ax.set_ylabel("decision recovery")
    ax.set_title("(a) few neurons do not carry it"); ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=5.6)


def panel_conc(ax):
    for tag, name, c in MODELS:
        a = J(f"t1_neurons_{tag}.json")["concentration"]["atp_train_mean"]
        ks = sorted(int(k) for k in a["abs_frac_topk"])
        ax.plot(ks, [a["abs_frac_topk"][str(k)] for k in ks], "o-", color=c, label=name)
        n50 = a["n_for_abs_frac"]["0.5"]
        ax.plot([n50], [0.5], "D", color=c, markersize=4)
    ax.axhline(0.5, color=C["grey"], ls=":", lw=1)
    ax.text(1.2, 0.53, "50% of |AtP| needs ~28k neurons", fontsize=6.4, color="#555")
    ax.set_xscale("log"); ax.set_ylim(0, 0.62)
    ax.set_xlabel("top-$k$ neurons"); ax.set_ylabel("share of |AtP| mass")
    ax.set_title("(b) attribution is diffuse"); ax.legend(loc="center left", fontsize=6.2)


def panel_layers(ax):
    for tag, name, c in MODELS:
        ls = J(f"t1_neurons_{tag}.json")["concentration"]["layer_abs_share"]
        x = np.arange(len(ls)) / (len(ls) - 1)
        ax.plot(x, ls, "-", color=c, label=f"{name} ({len(ls)} layers)")
    ax.set_xlabel("relative depth"); ax.set_ylabel("share of |AtP| per layer"); ax.set_ylim(0, 0.18)
    ax.set_title("(c) where the MLP write sits"); ax.legend(loc="upper left", fontsize=6.0)


def panel_sign(ax):
    t = J("t7_llama31_8b.json")
    P, S = t["parts"]["ALL"], t["santa"]
    labels = ["all: zero\npositive", "all: zero\nnegative", "all: zero\nboth", "top-8\nneg.-side", "top-8\npos.-side"]
    vals = [P["pos"]["top"], P["neg"]["top"], P["both"]["top"], S["neg_side_8"]["share_lost"], S["pos_side_8"]["share_lost"]]
    rnd = [None, None, None, S["neg_side_8"]["rand"], S["pos_side_8"]["rand"]]
    cols = [C["blue"], C["red"], C["grey"], C["red"], C["blue"]]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.62)
    for i, (v, r) in enumerate(zip(vals, rnd)):
        ax.text(i, v + 0.02, f"{v:.2f}" if v >= 0.1 else f"{v:.3f}", ha="center", fontsize=6.4)
        if r is not None:
            ax.plot([i - 0.31, i + 0.31], [r, r], color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of decision lost")
    ax.set_title("(d) both signs matter, in aggregate")


def panel_additivity(ax):
    a = J("t1b_additivity_ci.json")
    order = [("pos", "posi-\ntions"), ("whead", "write\nheads"), ("neuron", "MLP\nneurons"),
             ("sae", "SAE\nfeat."), ("rhead", "read\nheads")]
    x = np.arange(len(order)); w = 0.38
    j = [a[k]["joint_nec_mean"] for k, _ in order]; s = [a[k]["sum_single_nec_mean"] for k, _ in order]
    ax.bar(x - w / 2, j, w, color=C["blue"], label="remove all jointly")
    ax.bar(x + w / 2, s, w, color=C["sky"], label="sum of one-at-a-time")
    for i, (k, _) in enumerate(order):
        ax.text(i, max(j[i], s[i]) + 0.03, f"{a[k]['singles_capture']:.2f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=6.2)
    ax.set_ylim(0, 1.0); ax.set_ylabel("share of decision lost")
    ax.set_title("(e) joint vs. summed single ablation"); ax.legend(loc="upper center", fontsize=6.0)
    ax.text(0.98, 0.70, "numbers: singles capture\n(sum / joint)", transform=ax.transAxes, ha="right", fontsize=5.8, color="#555")


def panel_gao(ax):
    g = J("gao_llama31_8b.json")
    rows = g["task"]["per_instance"]
    sets = [("top30", "top-30"), ("top100", "top-100"), ("random100", "random-100")]
    x = np.arange(len(sets)); w = 0.38
    real = [np.mean([r[f"{s}/realpatch_recovery"] for r in rows]) for s, _ in sets]
    sim = [np.mean([r[f"{s}/simpatch_recovery"] for r in rows]) for s, _ in sets]
    ax.bar(x - w / 2, real, w, color=C["blue"], label="real activations")
    ax.bar(x + w / 2, sim, w, color=C["orange"], label="simulated from label")
    for i in range(len(sets)):
        ax.text(i - w / 2, real[i] + 0.02, f"{real[i]:.2f}", ha="center", fontsize=6.4)
        ax.text(i + w / 2, sim[i] + 0.02, f"{sim[i]:.2f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in sets])
    ax.set_ylim(0, 0.7); ax.set_ylabel("decision recovery (patch)")
    ax.set_title("(f) legible labels, failed simulation"); ax.legend(loc="upper left", fontsize=6.2)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for f, ax in zip([panel_patch, panel_conc, panel_layers, panel_sign, panel_additivity, panel_gao], axs.flat):
        f(ax)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    p = os.path.join(OUT, "fig10_neurons.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "..", "..", "followup", "fig10_neurons.png"), bbox_inches="tight", dpi=200)
    print("wrote", p)


if __name__ == "__main__":
    main()
