"""T1b additivity with bootstrap CIs (CPU only; reads results/t1b_llama31_8b.json).

For each carrier type, over the 24 SAFE/UNSAFE pairs:
  joint/sum (necessity)  = sum_pairs joint drop / sum_pairs (sum of single-carrier drops)
                           1 = additive, >1 = superadditive (one-at-a-time removal undercounts),
                           <1 = overlap / redundancy
  singles capture        = sum_pairs (sum of single drops) / sum_pairs joint drop
  joint - sum            = mean over pairs of (joint drop - sum of single drops)
  joint/sum (sufficiency)= same ratio for patching SAFE values into the UNSAFE run
95% CIs: percentile bootstrap over pairs (10,000 resamples, seed 0).
Writes results/t1b_additivity_ci.json and prints a markdown table.
"""
import json, os
import numpy as np

OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
NAMES = {"pos": "aggregator positions (16)", "whead": "write heads (15)", "neuron": "MLP neurons (30)",
         "sae": "SAE L14 features (30)", "rhead": "read heads (12)"}


def boot(stat, arrs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(arrs[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(stat(*[a[idx] for a in arrs]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    d = json.load(open(os.path.join(OUT, "t1b_llama31_8b.json")))
    res = {}
    rows = ["| carrier | joint drop | sum of single drops | joint/sum (nec.) | singles capture | joint - sum | joint/sum (suff.) |",
            "|---|---|---|---|---|---|---|"]
    for typ, v in d["types"].items():
        A = v["additivity"]
        jn = np.array([a["joint_nec"] for a in A]); sn = np.array([a["sum_single_nec"] for a in A])
        js = np.array([a["joint_suff"] for a in A]); ss = np.array([a["sum_single_suff"] for a in A])
        r = {
            "n_pairs": len(A),
            "joint_nec_mean": float(jn.mean()), "sum_single_nec_mean": float(sn.mean()),
            "joint_over_sum_nec": float(jn.sum() / sn.sum()), "joint_over_sum_nec_ci": boot(lambda j, s: j.sum() / s.sum(), [jn, sn]),
            "singles_capture": float(sn.sum() / jn.sum()), "singles_capture_ci": boot(lambda j, s: s.sum() / j.sum(), [jn, sn]),
            "joint_minus_sum": float((jn - sn).mean()), "joint_minus_sum_ci": boot(lambda j, s: (j - s).mean(), [jn, sn]),
            "joint_over_sum_suff": float(js.sum() / ss.sum()), "joint_over_sum_suff_ci": boot(lambda j, s: j.sum() / s.sum(), [js, ss]),
        }
        res[typ] = r
        f = lambda x, c: f"{x:.2f} [{c[0]:.2f}, {c[1]:.2f}]"
        rows.append(f"| {NAMES[typ]} | {r['joint_nec_mean']:.3f} | {r['sum_single_nec_mean']:.3f} | "
                    f"{f(r['joint_over_sum_nec'], r['joint_over_sum_nec_ci'])} | {f(r['singles_capture'], r['singles_capture_ci'])} | "
                    f"{r['joint_minus_sum']:+.3f} [{r['joint_minus_sum_ci'][0]:+.3f}, {r['joint_minus_sum_ci'][1]:+.3f}] | "
                    f"{f(r['joint_over_sum_suff'], r['joint_over_sum_suff_ci'])} |")
    json.dump(res, open(os.path.join(OUT, "t1b_additivity_ci.json"), "w"), indent=1)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
