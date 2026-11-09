"""T7: negative-pole test on the Programmable-KV note neurons (Llama-3.1-8B).

At the top-8 aggregator positions, split each neuron's (SwiGLU intermediate) activation by sign
and zero (a) only its positive parts, (b) only its negative parts, (c) both, in BOTH the SAFE and
the UNSAFE prompt. The decision is binary, so the conclusion signal is D = S_safe - S_unsafe
(S = logit[safe action] - logit[unsafe action]); share_lost = 1 - D'/D.

Neuron sets: top-k by |attribution| (train-mean AtP from t1_neurons); controls: random neurons in
the same layer with a matched negative-mass fraction (sign-matched), 3 draws.
'Negative-side' neurons: >60% of their |AtP| comes from aggregator positions where the
neuron's activation is negative. Santa analog: 8 top negative-side vs 20 top positive-side.
"""
import json, os, sys
import numpy as np
import torch

REPO = os.environ.get("PKV_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "esys"))
sys.path.insert(0, os.path.join(REPO, "e1"))
import circuit_common as cc  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_common import build_pair_f  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
TAG = os.environ.get("TAG", "llama31_8b")
MODEL_ID = os.environ.get("MODEL", "unsloth/Llama-3.1-8B-Instruct")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    nl, I = model.config.num_hidden_layers, model.config.intermediate_size
    sd = torch.load(os.path.join(OUT, f"t1_signdata_{TAG}.pt"), weights_only=False)
    absn = [tuple(x) for x in sd["abs_top"]]
    per = sd["per_instance"]

    # ---- capture full intermediates at agg8 for every prompt
    cap = {}
    patch = {}

    def mk(L):
        def hook(mod, a):
            x = a[0]
            if mode[0] == "cap":
                cap[L] = x[0, posT[0]].float().clone()
                return None
            if mode[0] == "patch" and L in patch:
                x = x.clone()
                idx, vals = patch[L]
                x[0, posT[0][:, None], idx[None, :]] = vals.to(x.dtype)
                return (x,)
            return None
        return hook
    mode = [None]; posT = [None]
    hs = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in range(nl)]

    insts = []
    for d in per:
        P = build_pair_f(tok, d["scn"], d["oid"], d["fi"])
        e = {"P": P, "pos": torch.tensor(d["agg8"], device="cuda"), "d": d}
        for w in ("safe", "unsafe"):
            posT[0] = e["pos"]; mode[0] = "cap"
            with torch.no_grad():
                lg = model(input_ids=P[w + "_ids"].cuda(), use_cache=False).logits[0, P["dpos"]].float()
            mode[0] = None
            e["A_" + w] = torch.stack([cap[L] for L in range(nl)])      # [32, 8, I]
            e["S_" + w] = cc.conc_score(lg, P["toi"])
        e["D"] = e["S_safe"] - e["S_unsafe"]
        insts.append(e)
    print("captured", [round(e["D"], 2) for e in insts], flush=True)

    @torch.no_grad()
    def share_lost(neur, part):
        """zero the `part` of each neuron's activation at agg8, in both prompts."""
        vals = []
        byL = {}
        for L, i in neur:
            byL.setdefault(L, []).append(i)
        mass = 0.0
        for e in insts:
            S = {}
            for w in ("safe", "unsafe"):
                A = e["A_" + w]
                patch.clear()
                for L, idxs in byL.items():
                    idx = torch.tensor(idxs, device="cuda")
                    v = A[L][:, idx].clone()
                    if part == "pos":
                        m = v > 0
                    elif part == "neg":
                        m = v < 0
                    else:
                        m = torch.ones_like(v, dtype=torch.bool)
                    mass += float(v[m].abs().sum())
                    v[m] = 0.0
                    patch[L] = (idx, v)
                posT[0] = e["pos"]; mode[0] = "patch"
                P = e["P"]
                lg = model(input_ids=P[w + "_ids"].cuda(), use_cache=False).logits[0, P["dpos"]].float()
                mode[0] = None; patch.clear()
                S[w] = cc.conc_score(lg, P["toi"])
            vals.append(1 - (S["safe"] - S["unsafe"]) / e["D"])
        return float(np.mean(vals)), mass / len(insts), [round(v, 3) for v in vals]

    # ---- negative-mass fraction per neuron (all neurons), from the captured activations
    allA = torch.stack([torch.cat([e["A_safe"], e["A_unsafe"]], 1) for e in insts])   # [n,32,16,I]
    negmass = allA.clamp(max=0).abs().sum((0, 2)); tot = allA.abs().sum((0, 2))
    negfrac = (negmass / (tot + 1e-9)).cpu()                                          # [32, I]

    # ---- attribution split by activation sign (top-2000 |AtP|)
    att_pos = np.zeros(len(absn)); att_neg = np.zeros(len(absn))
    for d in per:
        dA = d["A_safe"] - d["A_unsafe"]
        c = (dA * d["G_unsafe"]).numpy() / d["denom"]                      # [2000, 8]
        sgn = ((d["A_safe"] + d["A_unsafe"]) / 2).numpy()
        att_pos += np.where(sgn > 0, np.abs(c), 0).sum(1)
        att_neg += np.where(sgn < 0, np.abs(c), 0).sum(1)
    neg_share = att_neg / (att_pos + att_neg + 1e-12)
    rng = np.random.default_rng(7)

    def sign_matched_random(neur):
        out = []
        for L, i in neur:
            target = float(negfrac[L, i])
            cand = torch.nonzero((negfrac[L] - target).abs() < 0.05).flatten().tolist()
            cand = [c for c in cand if (L, c) not in set(absn[:300])] or list(range(I))
            out.append((L, int(cand[rng.integers(len(cand))])))
        return out

    res = {"neg_share_of_attribution": {}, "parts": {}, "santa": {}}
    for k in (10, 30, 100, 300, 1000):
        res["neg_share_of_attribution"][k] = {
            "abs_attr_from_negative_activation": float(att_neg[:k].sum() / (att_pos[:k] + att_neg[:k]).sum()),
            "n_negative_side": int((neg_share[:k] > 0.6).sum()), "n_positive_side": int((neg_share[:k] < 0.4).sum())}
    print("attr sign split:", res["neg_share_of_attribution"], flush=True)

    for k in (10, 30, 100, 300):
        neur = absn[:k]
        r = {}
        for part in ("pos", "neg", "both"):
            m, mass, pi = share_lost(neur, part)
            rr = [share_lost(sign_matched_random(neur), part) for _ in range(3)]
            r[part] = {"top": round(m, 4), "top_mass": round(mass, 2), "top_per_inst": pi,
                       "rand": round(float(np.mean([x[0] for x in rr])), 4),
                       "rand_mass": round(float(np.mean([x[1] for x in rr])), 2)}
            print(f"k={k} {part}: top={m:+.3f} (mass {mass:.1f}) rand={r[part]['rand']:+.3f} (mass {r[part]['rand_mass']:.1f})", flush=True)
        res["parts"][k] = r

    # all neurons at agg8: positive parts vs negative parts
    alln = [(L, i) for L in range(nl) for i in range(I)]
    for part in ("pos", "neg", "both"):
        m, mass, pi = share_lost(alln, part)
        res["parts"].setdefault("ALL", {})[part] = {"top": round(m, 4), "mass": round(mass, 1), "per_inst": pi}
        print(f"ALL neurons {part}: {m:+.3f}", flush=True)

    # Santa analog
    negside = [absn[j] for j in range(len(absn)) if neg_share[j] > 0.6]
    posside = [absn[j] for j in range(len(absn)) if neg_share[j] < 0.4]
    for name, neur in (("neg_side_8", negside[:8]), ("pos_side_20", posside[:20]),
                       ("neg_side_20", negside[:20]), ("pos_side_8", posside[:8]),
                       ("top_abs_8", absn[:8]), ("top_abs_20", absn[:20])):
        m, mass, pi = share_lost(neur, "both")
        rr = [share_lost(sign_matched_random(neur), "both")[0] for _ in range(3)]
        res["santa"][name] = {"share_lost": round(m, 4), "rand": round(float(np.mean(rr)), 4), "mass": round(mass, 2),
                              "neurons": [f"L{L}.{i}" for L, i in neur],
                              "neg_share": [round(float(neg_share[absn.index(n)]), 3) for n in neur]}
        print(f"santa {name}: lost={m:+.3f} rand={np.mean(rr):+.3f}", flush=True)
    res["top30_neg_share"] = [{"neuron": f"L{L}.{i}", "neg_share_attr": round(float(neg_share[j]), 3),
                               "neg_mass_frac": round(float(negfrac[L, i]), 3)} for j, (L, i) in enumerate(absn[:30])]
    json.dump(res, open(os.path.join(OUT, f"t7_{TAG}.json"), "w"), indent=1)
    print("T7_DONE", flush=True)


if __name__ == "__main__":
    main()
