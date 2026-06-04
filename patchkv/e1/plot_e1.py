"""Generate E1 figures from a results json + raw npz dir.

P1: BR(tau) curves by field class (headline).
P2: attention-output deviation vs position (validate causally-exact region ~0).
P3: attention-output deviation vs layer.
P4: per-field summary bars (downstream p95) grouped by class, semantic vs minor.
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(HERE, "..", "plots")
os.makedirs(PLOTS, exist_ok=True)

CLS_COLOR = {"low": "#2c7fb8", "medium": "#d95f0e", "high": "#c51b8a"}
TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def load(tag):
    recs = json.load(open(os.path.join(RES, f"e1_{tag}.json")))
    rawdir = os.path.join(RES, f"raw_{tag}")
    return recs, rawdir


def br_curve_from_raw(raw, metric="attn"):
    arr = raw[metric]                       # [L,T]
    span = raw["field_span"]; s, e = int(span[0]), int(span[1])
    T = int(raw["seq_len"][0])
    down = np.arange(T) >= e
    m = arr.max(0)[down]
    return np.array([(m > t).mean() for t in TAUS])


def fig_P1(tag, recs, rawdir, magnitude="semantic", metric="attn"):
    plt.figure(figsize=(7, 5))
    by_cls = {"low": [], "medium": [], "high": []}
    for r in recs:
        if r["magnitude"] != magnitude:
            continue
        raw = np.load(os.path.join(rawdir, f"{r['field']}_{magnitude}.npz"))
        curve = br_curve_from_raw(raw, metric)
        by_cls[r["cls"]].append((r["field"], curve))
    for cls, items in by_cls.items():
        for field, curve in items:
            plt.plot(TAUS, curve, color=CLS_COLOR[cls], alpha=0.45, lw=1)
        if items:
            mean = np.mean([c for _, c in items], 0)
            plt.plot(TAUS, mean, color=CLS_COLOR[cls], lw=3, label=f"{cls} (mean)")
    plt.xscale("log"); plt.xlabel(r"deviation threshold $\tau$ (attn-output)")
    plt.ylabel(r"BR($\tau$) = frac downstream tokens > $\tau$")
    plt.title(f"P1 Blast radius by field class — {tag}, {magnitude} flip")
    plt.legend(); plt.grid(alpha=0.3)
    p = os.path.join(PLOTS, f"P1_{tag}_{magnitude}_{metric}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    return p


def fig_P2_P3(tag, recs, rawdir, magnitude="semantic"):
    # pick one representative field per class
    reps = {}
    for r in recs:
        if r["magnitude"] == magnitude and r["cls"] not in reps:
            reps[r["cls"]] = r
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for cls, r in reps.items():
        raw = np.load(os.path.join(rawdir, f"{r['field']}_{magnitude}.npz"))
        arr = raw["attn"]; s, e = raw["field_span"]
        # P2: vs position (max over layers)
        vp = arr.max(0)
        axes[0].plot(vp, color=CLS_COLOR[cls], alpha=0.8, lw=0.8,
                     label=f"{cls}: {r['field']}")
        # P3: vs layer (mean over downstream tokens)
        T = arr.shape[1]; down = np.arange(T) >= e
        axes[1].plot(arr[:, down].mean(1), color=CLS_COLOR[cls], lw=2,
                     label=f"{cls}: {r['field']}")
    # mark field location on P2 via axvspan of the rep with smallest s
    r0 = list(reps.values())[0]
    s0, e0 = np.load(os.path.join(rawdir, f"{r0['field']}_{magnitude}.npz"))["field_span"]
    axes[0].axvspan(s0, e0, color="gray", alpha=0.25, label="field (approx)")
    axes[0].set_xlabel("token position"); axes[0].set_ylabel("attn-output dev (max over layers)")
    axes[0].set_title("P2 deviation vs position (causally-exact region before field ≈ 0)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("downstream mean attn-output dev")
    axes[1].set_title("P3 deviation vs layer")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    p = os.path.join(PLOTS, f"P2P3_{tag}_{magnitude}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    return p


def fig_P4(tag, recs):
    fields = sorted({r["field"] for r in recs}, key=lambda f:
                    {"low": 0, "medium": 1, "high": 2}[next(r["cls"] for r in recs if r["field"] == f)])
    x = np.arange(len(fields)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, mag in enumerate(["semantic", "minor"]):
        vals = []
        for f in fields:
            r = next((r for r in recs if r["field"] == f and r["magnitude"] == mag), None)
            vals.append(r["attn_down_p95"] if r else 0)
        ax.bar(x + (i - 0.5) * w, vals, w, label=mag,
               color=["white", "0.6"][i], edgecolor="k")
    for xi, f in zip(x, fields):
        cls = next(r["cls"] for r in recs if r["field"] == f)
        ax.get_xticklabels()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n[{next(r['cls'] for r in recs if r['field']==f)}]"
                        for f in fields], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("downstream attn-output dev p95")
    ax.set_title(f"P4 per-field downstream deviation (p95) — {tag}")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    p = os.path.join(PLOTS, f"P4_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=120); plt.close()
    return p


if __name__ == "__main__":
    tag = sys.argv[1]
    recs, rawdir = load(tag)
    print(fig_P1(tag, recs, rawdir, "semantic"))
    print(fig_P1(tag, recs, rawdir, "minor"))
    print(fig_P2_P3(tag, recs, rawdir, "semantic"))
    print(fig_P4(tag, recs))
