"""T1 steps 2-3: MLP-neuron view of the Programmable-KV conclusion note (Llama-3.1-8B).

Task / pairs / aggregator localization reuse the repo's own circuit harness
(esys/circuit_common.py): a conclusion-flip pair whose FIELD is byte-identical and only the
rule trigger differs; aggregators = downstream positions ranked by single-position KV-patch
recovery (find_aggregators).

Neuron = SwiGLU intermediate unit (input of mlp.down_proj), (layer, idx); 32 x 14336.

Attribution (per instance, summed over the top-8 aggregator positions):
  atp[L,i]  = sum_p (a_safe - a_unsafe)[L,p,i] * dS/da |_unsafe     (attribution patching;
              first-order estimate of the recovery from patching that neuron SAFE->UNSAFE)
  axg[L,i]  = sum_p a_safe[L,p,i] * dS/da |_safe                     (activation x gradient)
  S = logit[safe action] - logit[unsafe action] at the decision position; normalized by
  denom = S_safe - S_unsafe so atp is in "recovery" units.
Ranking: mean atp over TRAIN instances (order ids A4471,B8820,C1093); causal tests on all 12
instances and separately on the HELD-OUT order id D5567.

Causal tests at the aggregator positions (all layers):
  sufficiency  : UNSAFE run, set top-k neurons to their SAFE values  -> recovery toward SAFE
  necessity    : SAFE run, set top-k neurons to their UNSAFE values  -> drop toward UNSAFE
  zero / mean  : SAFE run, zero or task-mean-ablate top-k            -> drop
  control      : k random neurons with the SAME layer histogram (5 draws)
  ceilings     : all 458,752 neurons at the aggregator positions (top-8 and top-1)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

REPO = os.environ.get("PKV_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "esys"))
sys.path.insert(0, os.path.join(REPO, "e1"))
import circuit_common as cc  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_common import INSTANCES, build_pair_f  # noqa: E402

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
TRAIN_OIDS = ["A4471", "B8820", "C1093"]
HELD_OID = "D5567"
KS = [1, 3, 10, 30, 100, 300]


# ------------------------------------------------------------------ hooks
class NeuronHooks:
    """pre-hooks on every mlp.down_proj. modes: capture (optionally retain grad) / patch."""

    def __init__(self, model):
        self.layers = cc.decoder_layers(model)
        self.model = model
        self.h = []
        self.mode = None
        self.acts = {}
        self.patch = {}   # L -> list of (pos LongTensor[P], idx LongTensor[N], vals Tensor[P,N])

    def __enter__(self):
        for L, layer in enumerate(self.layers):
            self.h.append(layer.mlp.down_proj.register_forward_pre_hook(self._mk(L)))
        self.h.append(self.model.model.embed_tokens.register_forward_hook(self._emb))
        return self

    def _emb(self, mod, args, out):
        if self.mode == "grad":
            return out.detach().requires_grad_(True)
        return out

    def _mk(self, L):
        def hook(mod, args):
            x = args[0]
            if self.mode in ("capture", "grad"):
                if self.mode == "grad":
                    x.retain_grad()
                    self.acts[L] = x
                else:
                    self.acts[L] = x.detach()
                return None
            if self.mode == "patch" and L in self.patch:
                x = x.clone()
                for pos, idx, vals in self.patch[L]:
                    x[0, pos[:, None], idx[None, :]] = vals.to(x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return hook

    def __exit__(self, *a):
        for x in self.h:
            x.remove()


def fwd_score(model, ids, dpos, toi):
    out = model(input_ids=ids.to("cuda"), use_cache=False)
    lg = out.logits[0, dpos].float()
    return cc.conc_score(lg, toi), lg


# ------------------------------------------------------------------ per-instance data
def instance_data(model, tok, H, scn, oid, fi):
    P = build_pair_f(tok, scn, oid, fi)
    ranked, rec, s_un_c, s_sa_c, denom_c = cc.find_aggregators(model, P, topn=8)
    dpos, toi = P["dpos"], P["toi"]
    agg8 = torch.tensor(sorted(ranked[:8]), device="cuda")
    agg1 = torch.tensor([ranked[0]], device="cuda")
    res = {"P": P, "agg8": agg8, "agg1": agg1, "agg_rec": {int(p): float(rec[p]) for p in ranked[:8]}}
    # SAFE (clean) with grad
    for name, ids in (("safe", P["safe_ids"]), ("unsafe", P["unsafe_ids"])):
        H.mode = "grad"
        H.acts = {}
        with torch.enable_grad():
            o = model(input_ids=ids.to("cuda"), use_cache=False)
            lg = o.logits[0, dpos].float()
            S = lg[toi["safe"]] - lg[toi["unsafe"]]
            S.backward()
        A = torch.stack([H.acts[L][0, agg8].detach().float() for L in range(len(H.layers))])   # [32,8,I]
        G = torch.stack([H.acts[L].grad[0, agg8].float() for L in range(len(H.layers))])
        allA = torch.stack([H.acts[L][0].detach().float().mean(0) for L in range(len(H.layers))])  # task mean [32,I]
        res[name] = {"S": float(S), "A": A, "G": G, "mean": allA}
        H.acts = {}
        del o
        torch.cuda.empty_cache()
    H.mode = None
    denom = res["safe"]["S"] - res["unsafe"]["S"]
    res["denom"] = denom
    dA = res["safe"]["A"] - res["unsafe"]["A"]
    res["atp"] = (dA * res["unsafe"]["G"]).sum(1) / denom              # [32, I]
    res["axg"] = (res["safe"]["A"] * res["safe"]["G"]).sum(1) / denom   # [32, I]
    return res


def set_patch(H, neurons, pos, vals_by_layer):
    """neurons: list of (L, i). vals_by_layer: L -> Tensor[len(pos), n_L] matching order."""
    H.patch = {}
    byL = {}
    for (L, i) in neurons:
        byL.setdefault(L, []).append(i)
    for L, idxs in byL.items():
        idx = torch.tensor(idxs, device="cuda")
        H.patch[L] = [(pos, idx, vals_by_layer(L, idx))]


@torch.no_grad()
def run_patch(model, H, R, neurons, mode, posname="agg8"):
    """Return normalized effect. mode in suff / nec / zero / mean."""
    P = R["P"]
    pos = R[posname]
    sel = R["sel8"] if posname == "agg8" else R["sel1"]   # index into the 8 captured positions
    if mode == "suff":
        base_ids, src = P["unsafe_ids"], R["safe"]["A"]
        fn = lambda L, idx: src[L][sel][:, idx]
    elif mode == "nec":
        base_ids, src = P["safe_ids"], R["unsafe"]["A"]
        fn = lambda L, idx: src[L][sel][:, idx]
    elif mode == "zero":
        base_ids = P["safe_ids"]
        fn = lambda L, idx: torch.zeros(len(pos), len(idx), device="cuda")
    elif mode == "mean":
        base_ids = P["safe_ids"]
        m = (R["safe"]["mean"] + R["unsafe"]["mean"]) / 2
        fn = lambda L, idx: m[L][idx][None, :].expand(len(pos), -1)
    set_patch(H, neurons, pos, fn)
    H.mode = "patch"
    s, _ = fwd_score(model, base_ids, P["dpos"], P["toi"])
    H.mode = None
    H.patch = {}
    su, ss = R["unsafe"]["S"], R["safe"]["S"]
    if mode == "suff":
        return (s - su) / (ss - su)
    return (ss - s) / (ss - su)   # drop toward UNSAFE


def layer_matched_random(neurons, I, rng, exclude):
    out = []
    for (L, _) in neurons:
        while True:
            j = int(rng.integers(I))
            if (L, j) not in exclude:
                out.append((L, j))
                break
    return out


def concentration(score):
    """score [32,I] signed. fraction of total |score| and of net signed score in top-k."""
    a = score.abs().flatten()
    tot = float(a.sum())
    srt = torch.sort(a, descending=True).values
    cum = torch.cumsum(srt, 0)
    fr = {k: float(cum[k - 1] / tot) for k in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]}
    # how many neurons for 50/80/90% of |score|
    need = {q: int(torch.searchsorted(cum, q * tot).item()) + 1 for q in (0.5, 0.8, 0.9)}
    s = score.flatten()
    pos_sorted = torch.sort(s, descending=True).values
    net = float(s.sum())
    netfr = {k: float(pos_sorted[:k].sum() / net) if net != 0 else None for k in [1, 3, 10, 30, 100, 300, 1000, 3000]}
    return {"abs_frac_topk": fr, "n_for_abs_frac": need, "net_sum": net,
            "pos_topk_over_net": netfr, "abs_total": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    args = ap.parse_args()
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nl = model.config.num_hidden_layers
    I = model.config.intermediate_size
    H = NeuronHooks(model).__enter__()

    insts = []
    for scn, oid, fi in INSTANCES:
        R = instance_data(model, tok, H, scn, oid, fi)
        R["scn"], R["oid"], R["fi"] = scn, oid, fi
        pos_list = R["agg8"].tolist()
        R["sel8"] = torch.arange(8, device="cuda")
        R["sel1"] = torch.tensor([pos_list.index(int(R["agg1"][0]))], device="cuda")
        print(f"[{scn}/{oid}/f{fi}] denom={R['denom']:.2f} agg8={pos_list} "
              f"sum_atp={float(R['atp'].sum()):.3f} ({time.time()-t0:.0f}s)", flush=True)
        if abs(R["denom"]) < 0.8:
            print("   non-flipping, skipped", flush=True)
            continue
        # keep GPU memory bounded for 24 instances: move per-instance tensors to CPU, restore on use
        insts.append(R)

    train = [R for R in insts if R["oid"] in TRAIN_OIDS]
    held = [R for R in insts if R["oid"] == HELD_OID]
    atp_tr = torch.stack([R["atp"] for R in train]).mean(0)       # [32,I]
    axg_tr = torch.stack([R["axg"] for R in train]).mean(0)
    atp_all = torch.stack([R["atp"] for R in insts]).mean(0)

    # -------- step 2: concentration + layer histogram
    conc = {"atp_train_mean": concentration(atp_tr), "axg_train_mean": concentration(axg_tr),
            "atp_per_instance_median_n_for_50pct": float(np.median(
                [concentration(R["atp"])["n_for_abs_frac"][0.5] for R in insts])),
            "atp_per_instance_median_n_for_80pct": float(np.median(
                [concentration(R["atp"])["n_for_abs_frac"][0.8] for R in insts]))}
    layer_abs = atp_tr.abs().sum(1)
    layer_net = atp_tr.sum(1)
    conc["layer_abs_share"] = [round(float(x), 4) for x in (layer_abs / layer_abs.sum())]
    conc["layer_net"] = [round(float(x), 4) for x in layer_net]
    conc["atp_sum_mean_over_instances"] = float(np.mean([float(R["atp"].sum()) for R in insts]))

    flat = atp_tr.flatten()
    order = torch.argsort(flat, descending=True)
    top = [(int(j) // I, int(j) % I) for j in order[:max(KS)]]
    top_info = [{"layer": L, "idx": i, "atp_train": round(float(atp_tr[L, i]), 4),
                 "atp_all": round(float(atp_all[L, i]), 4), "axg_train": round(float(axg_tr[L, i]), 4),
                 "atp_by_scn": {s: round(float(np.mean([float(R["atp"][L, i]) for R in insts if R["scn"] == s])), 4)
                                for s in cc.SCNS}}
                for (L, i) in top[:100]]
    # rank agreement atp vs axg
    top_axg = set((int(j) // I, int(j) % I) for j in torch.argsort(axg_tr.flatten(), descending=True)[:100])
    conc["top100_overlap_atp_vs_axg"] = len(set(top[:100]) & top_axg)

    # -------- AtP linearity check: all neurons patched at agg8 vs sum of atp
    lin = []
    for R in insts:
        H.patch = {L: [(R["agg8"], torch.arange(I, device="cuda"), R["safe"]["A"][L])] for L in range(nl)}
        H.mode = "patch"
        with torch.no_grad():
            s, _ = fwd_score(model, R["P"]["unsafe_ids"], R["P"]["dpos"], R["P"]["toi"])
        H.mode = None; H.patch = {}
        actual = (s - R["unsafe"]["S"]) / R["denom"]
        lin.append({"scn": R["scn"], "oid": R["oid"], "fi": R["fi"], "all_neurons_agg8_actual": round(actual, 4),
                    "atp_sum": round(float(R["atp"].sum()), 4)})
    print("linearity:", lin, flush=True)

    # -------- step 3: causal top-k vs random-k
    rng = np.random.default_rng(0)
    results = {}
    for posname in ("agg8", "agg1"):
        for mode in ("suff", "nec", "zero", "mean"):
            for k in KS:
                nk = top[:k]
                excl = set(nk)
                tv, rv, tv_h, rv_h = [], [], [], []
                for R in insts:
                    t = run_patch(model, H, R, nk, mode, posname)
                    rs = [run_patch(model, H, R, layer_matched_random(nk, I, rng, excl), mode, posname)
                          for _ in range(5 if k <= 100 else 3)]
                    tv.append(t); rv.append(float(np.mean(rs)))
                    if R["oid"] == HELD_OID:
                        tv_h.append(t); rv_h.append(float(np.mean(rs)))
                results[f"{posname}/{mode}/k{k}"] = {
                    "top_mean": round(float(np.mean(tv)), 4), "rand_mean": round(float(np.mean(rv)), 4),
                    "top_held": round(float(np.mean(tv_h)), 4) if tv_h else None,
                    "rand_held": round(float(np.mean(rv_h)), 4) if rv_h else None,
                    "top_per_inst": [round(x, 3) for x in tv]}
                print(f"{posname} {mode} k={k}: top={np.mean(tv):+.3f} (held {np.mean(tv_h) if tv_h else float('nan'):+.3f}) "
                      f"rand={np.mean(rv):+.3f} ({time.time()-t0:.0f}s)", flush=True)
        # ceiling: all neurons at these positions
        for mode in ("suff", "nec"):
            vals = []
            for R in insts:
                P = R["P"]; pos = R[posname]; sel = R["sel8"] if posname == "agg8" else R["sel1"]
                src = R["safe"]["A"] if mode == "suff" else R["unsafe"]["A"]
                H.patch = {L: [(pos, torch.arange(I, device="cuda"), src[L][sel])] for L in range(nl)}
                H.mode = "patch"
                with torch.no_grad():
                    s, _ = fwd_score(model, P["unsafe_ids"] if mode == "suff" else P["safe_ids"], P["dpos"], P["toi"])
                H.mode = None; H.patch = {}
                su, ss = R["unsafe"]["S"], R["safe"]["S"]
                vals.append((s - su) / (ss - su) if mode == "suff" else (ss - s) / (ss - su))
            results[f"{posname}/{mode}/ALL_NEURONS"] = {"mean": round(float(np.mean(vals)), 4),
                                                       "per_inst": [round(v, 3) for v in vals]}
            print(f"{posname} {mode} ALL neurons: {np.mean(vals):+.3f}", flush=True)

    # per-instance in-sample top-k sufficiency (upper bound on how few neurons could suffice)
    insample = {}
    for k in [1, 10, 100, 1000]:
        v = []
        for R in insts:
            o = torch.argsort(R["atp"].flatten(), descending=True)[:k]
            nk = [(int(j) // I, int(j) % I) for j in o]
            v.append(run_patch(model, H, R, nk, "suff", "agg8"))
        insample[k] = round(float(np.mean(v)), 4)
        print(f"in-sample per-instance top-{k} suff: {np.mean(v):+.3f}", flush=True)

    out = {"model": args.model, "n_instances": len(insts), "n_train": len(train), "n_held": len(held),
           "instances": [{"scn": R["scn"], "oid": R["oid"], "fi": R["fi"], "denom": round(R["denom"], 3),
                          "agg8": R["agg8"].tolist(), "agg1": int(R["agg1"][0]),
                          "agg_kv_rec": R["agg_rec"]} for R in insts],
           "concentration": conc, "linearity": lin, "causal": results, "insample_suff": insample,
           "top_neurons": top_info, "top300": top}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"t1_neurons_{args.tag}.json"), "w"), indent=1)
    # per-instance raw activations / gradients for the top-2000 |atp| neurons (for T7 sign split)
    absorder = torch.argsort(atp_tr.abs().flatten(), descending=True)[:2000]
    absn = [(int(j) // I, int(j) % I) for j in absorder]
    Ls = torch.tensor([L for L, _ in absn]); Is = torch.tensor([i for _, i in absn])
    per = []
    for R in insts:
        d = {"scn": R["scn"], "oid": R["oid"], "fi": R["fi"], "agg8": R["agg8"].tolist(), "denom": R["denom"],
             "S_safe": R["safe"]["S"], "S_unsafe": R["unsafe"]["S"]}
        for w in ("safe", "unsafe"):
            d[f"A_{w}"] = R[w]["A"][Ls, :, Is].cpu()     # [2000, 8]
            d[f"G_{w}"] = R[w]["G"][Ls, :, Is].cpu()
        per.append(d)
    torch.save({"abs_top": absn, "per_instance": per}, os.path.join(OUT, f"t1_signdata_{args.tag}.pt"))
    torch.save({"atp_tr": atp_tr.cpu(), "axg_tr": axg_tr.cpu(), "atp_all": atp_all.cpu(), "top": top,
                "task_mean": torch.stack([(R["safe"]["mean"] + R["unsafe"]["mean"]) / 2 for R in insts]).mean(0).cpu()},
               os.path.join(OUT, f"t1_atp_{args.tag}.pt"))
    print("T1_NEURONS_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
