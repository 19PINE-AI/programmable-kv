"""T1b: is the note write DISTRIBUTED (each carrier holds a piece) or REDUNDANT (several carriers
each hold most of it)? Same question for the late read heads. Llama-3.1-8B, all 24 polarity pairs.

Carrier types
  pos    : top-16 aggregator positions (KV of the position, all layers; repo's KV-patching harness)
  whead  : top-20 write heads (paper circ_heads ranking), per-head o_proj input at the agg8 positions
  neuron : top-30 MLP neurons from t1_neurons (down_proj input at agg8)
  sae    : top-30 conclusion features of a TopK SAE on the layer-14 residual (repo circ_sae recipe,
           retrained: dict 16384, k 32, 4000 steps), at the top-1 aggregator
  rhead  : top-12 read heads (paper ranking), per-head o_proj input at the decision position
S = logit[safe] - logit[unsafe] at the decision token; clean = SAFE prompt, corrupt = UNSAFE prompt.
  retention(set) on clean run = (S' - S_corrupt)/(S_clean - S_corrupt); ablation 'patch' = carrier set to
  its corrupt-run value, 'mean' = task mean (positions: patch only, since a RoPE'd KV has no clean mean)
 1. ablation-size curve: 20 random k-subsets of the top-n per k
 2. single-carrier sufficiency (patch only it into corrupt) vs necessity (1 - retention, ablate only it)
 3. compensation: after patch-ablating carrier c, change of every other carrier's direct effect
    (attribution patching (x_run - x_corrupt).dS/dx for write carriers; direct logit attribution through
    the final norm for read heads, also over ALL heads in layers >= the ablated head's layer)
    ratio_c = sum_{o != c}(eff_o_after - eff_o_before) / eff_c_before
 additivity: sum of single necessity drops vs joint drop for the full top-n set
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_common import REPO, INSTANCES, build_pair_f, TRAIN_OIDS  # noqa
import circuit_common as cc  # noqa
import circ_sae as cs  # noqa
from mechd_common import POL  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
TAG = os.environ.get("TAG", "llama31_8b")
MODEL_ID = os.environ.get("MODEL", "unsloth/Llama-3.1-8B-Instruct")
SAE_L = 14
NSUB = 20
BATCH = int(os.environ.get("T1B_BATCH", "8"))


def kgrid(n):
    return sorted(set([k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30) if k <= n] + [n]))


class Hooks:
    """Row-specific patches at mlp.down_proj input / o_proj input / decoder-layer output."""

    def __init__(self, model):
        c = model.config
        self.nl, self.nh = c.num_hidden_layers, c.num_attention_heads
        self.hd = c.hidden_size // self.nh
        self.reset()
        for L, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._dn(L))
            layer.self_attn.o_proj.register_forward_pre_hook(self._op(L))
            layer.register_forward_hook(self._res(L))
        model.model.embed_tokens.register_forward_hook(self._emb)

    def reset(self):
        self.neur, self.head, self.inject = {}, {}, {}   # L -> list of (row, pos, idx|h|None, vals)
        self.grad = False
        self.capture = False
        self.cap_o, self.cap_d, self.cap_r = {}, {}, {}

    def _emb(self, m, a, out):
        return out.detach().requires_grad_(True) if self.grad else out

    def _dn(self, L):
        def hook(m, a):
            x = a[0]
            if L in self.neur:
                x = x.clone()
                for row, pos, idx, vals in self.neur[L]:
                    x[row, pos[:, None], idx[None, :]] = vals.to(x.dtype)
            if self.grad:
                x.retain_grad()
            if self.capture:
                self.cap_d[L] = x
            return (x,)
        return hook

    def _op(self, L):
        def hook(m, a):
            x = a[0]
            if L in self.head:
                x = x.clone()
                for row, pos, h, vals in self.head[L]:
                    x[row, pos, h * self.hd:(h + 1) * self.hd] = vals.to(x.dtype)
            if self.grad:
                x.retain_grad()
            if self.capture:
                self.cap_o[L] = x
            return (x,)
        return hook

    def _res(self, L):
        def hook(m, a, out):
            hs = out[0] if isinstance(out, tuple) else out
            if L in self.inject:
                hs = hs.clone()
                for row, pos, _, vec in self.inject[L]:
                    hs[row, pos] = hs[row, pos] + vec.to(hs.dtype)
            if L == SAE_L and self.capture:
                if self.grad:
                    hs.retain_grad()
                self.cap_r[L] = hs
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    H = Hooks(model)
    nl, nh, hd = H.nl, H.nh, H.hd
    W_U = model.lm_head.weight
    W_Uf = W_U.float()                       # fp32 unembedding: decision logits at fp32 resolution (bf16 step ~0.125)
    gfin = model.model.norm.weight.float()
    eps = model.config.rms_norm_eps
    heads_rec = json.load(open(os.path.join(REPO, "results", "circ_heads_llama31_8b.json")))["summary"]
    WH = [tuple(int(x) for x in h["head"].split(".")) for h in heads_rec["write_heads_ranked"][:20]]
    RH = [tuple(int(x) for x in h["head"].split(".")) for h in heads_rec["read_heads_ranked"][:12]]
    nj = json.load(open(os.path.join(OUT, f"t1_neurons_{TAG}.json")))
    NEUR = [tuple(x) for x in nj["top300"][:30]]
    task_mean = torch.load(os.path.join(OUT, f"t1_atp_{TAG}.pt"))["task_mean"].cuda()   # [nl, I]

    insts = {}
    aggf = os.path.join(OUT, f"t1b_aggs_{TAG}.json")
    aggc = json.load(open(aggf)) if os.path.exists(aggf) else {}
    for scn, oid, fi in INSTANCES:
        P = build_pair_f(tok, scn, oid, fi)
        key = f"{scn}|{oid}|{fi}"
        if key not in aggc:
            ranked, rec, *_ = cc.find_aggregators(model, P, topn=16)
            aggc[key] = [int(x) for x in ranked[:16]]
            json.dump(aggc, open(aggf, "w"))
        ranked = aggc[key]
        insts[(scn, oid, fi)] = {"P": P, "agg16": ranked[:16], "agg8": sorted(ranked[:8]), "agg1": ranked[0]}
    print(f"aggregators found ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- SAE (repo recipe) on layer 14
    prompts = []
    for scn in cc.SCNS:
        vA, vB = POL[scn]["values"]
        for oid in TRAIN_OIDS:
            for f in (vA, vB):
                for t in (vA, vB):
                    prompts.append(cs.pid(tok, scn, oid, f, t))
    prompts += cs.general_prompts(tok)
    X = cs.gather_acts(model, tok, SAE_L, prompts)
    scale = float(X.norm(dim=1).mean())
    saef = os.path.join(OUT, f"t1b_sae_L{SAE_L}_{TAG}.pt")
    if os.path.exists(saef):
        sae = cs.TopKSAE(model.config.hidden_size, 16384, 32).cuda()
        sae.load_state_dict(torch.load(saef))
    else:
        torch.manual_seed(0)
        sae = cs.train_sae(X, model.config.hidden_size, 16384, 32, 4000, bs=2048, lr=1e-3, scale=scale)
        torch.save(sae.state_dict(), saef)
    for p in sae.parameters():
        p.requires_grad_(False)
    Sc, Uc = [], []
    for scn in cc.SCNS:
        vA, vB = POL[scn]["values"]
        for oid in TRAIN_OIDS:
            off = insts[(scn, oid, 0)]["P"]["dpos"] - insts[(scn, oid, 0)]["agg1"]
            Sc += [cs.agg_code(model, sae, tok, scn, oid, vA, vA, SAE_L, off, scale),
                   cs.agg_code(model, sae, tok, scn, oid, vB, vB, SAE_L, off, scale)]
            Uc += [cs.agg_code(model, sae, tok, scn, oid, vA, vB, SAE_L, off, scale),
                   cs.agg_code(model, sae, tok, scn, oid, vB, vA, SAE_L, off, scale)]
    Sc, Uc = np.stack(Sc), np.stack(Uc)
    aucs = np.array([cs.auc(Sc[:, f], Uc[:, f]) for f in range(16384)])
    FEATS = [int(f) for f in np.argsort(np.abs(aucs - 0.5))[::-1][:30]]
    with torch.no_grad():
        zmean = torch.cat([sae.encode((X[i:i + 4096] / scale).cuda()) for i in range(0, len(X), 4096)]).mean(0)
    Wdec = sae.W_dec.detach()
    print(f"SAE trained; top feats {FEATS[:5]} AUC {[round(float(aucs[f]), 3) for f in FEATS[:5]]} ({time.time()-t0:.0f}s)", flush=True)

    results = {"carriers": {"whead": [f"{a}.{b}" for a, b in WH], "rhead": [f"{a}.{b}" for a, b in RH],
                            "neuron": [f"L{a}.{b}" for a, b in NEUR], "sae_feats": FEATS,
                            "sae_auc": [round(float(aucs[f]), 3) for f in FEATS]},
               "types": {}}
    rng = np.random.default_rng(0)
    NT = {"pos": 16, "whead": len(WH), "neuron": len(NEUR), "sae": len(FEATS), "rhead": len(RH)}

    for (scn, oid, fi), E in insts.items():
        P = E["P"]; toi = P["toi"]; dpos = P["dpos"]
        agg8 = torch.tensor(E["agg8"], device="cuda"); dposT = torch.tensor([dpos], device="cuda")
        a1 = E["agg1"]
        score = lambda lg: float(lg[toi["safe"]] - lg[toi["unsafe"]])

        def fwd(ids, nrow=1, grad=False):
            x = ids.cuda().expand(nrow, -1).contiguous()
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                o = model.model(input_ids=x, use_cache=False)
                return o.last_hidden_state[:, -1].float() @ W_Uf.T

        def kv_score(base_c, src_c, positions):
            w = cc.clone_cache(base_c, dpos)
            if positions:
                pt = torch.tensor(positions, device="cuda")
                for i in range(nl):
                    w.layers[i].keys[:, :, pt] = src_c.layers[i].keys[:, :, pt]
                    w.layers[i].values[:, :, pt] = src_c.layers[i].values[:, :, pt]
            with torch.no_grad():
                o = model.model(input_ids=torch.tensor([[P["last"]]], device="cuda"), past_key_values=w,
                                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
            return score(o.last_hidden_state[0, -1].float() @ W_Uf.T)

        cap = {}
        for w in ("safe", "unsafe"):
            H.reset(); H.capture = True
            lg = fwd(P[w + "_ids"])[0]
            H.reset()
            S_batched = score(fwd(P[w + "_ids"], nrow=BATCH)[0])   # baseline with the SAME batch numerics as run_many
            H.capture = True
            fwd(P[w + "_ids"])
            cap[w] = {"S": S_batched, "S_b1": score(lg),
                      "o_agg": torch.stack([H.cap_o[L][0, agg8].float() for L in range(nl)]),
                      "o_dpos": torch.stack([H.cap_o[L][0, dpos].float() for L in range(nl)]),
                      "o_mean": torch.stack([H.cap_o[L][0].float().mean(0) for L in range(nl)]),
                      "d_agg": torch.stack([H.cap_d[L][0, agg8].float() for L in range(nl)]),
                      "r1": H.cap_r[SAE_L][0, a1].float().clone()}
            cap[w]["z1"] = sae.encode((cap[w]["r1"] / scale)[None])[0]
        H.reset()
        S_ff = (cap["safe"]["S"], cap["unsafe"]["S"])
        kv = {w: cc.prefill(model, P[w + "_ids"]).past_key_values for w in ("safe", "unsafe")}
        kvS = {w: kv_score(kv[w], kv[w], []) for w in kv}

        def add_patch(typ, subset, method, base, row):
            src = "unsafe" if base == "safe" else "safe"
            if typ == "neuron":
                byL = {}
                for c in subset:
                    byL.setdefault(NEUR[c][0], []).append(NEUR[c][1])
                for L, idxs in byL.items():
                    idx = torch.tensor(idxs, device="cuda")
                    vals = cap[src]["d_agg"][L][:, idx] if method == "patch" else task_mean[L][idx][None].expand(len(agg8), -1)
                    H.neur.setdefault(L, []).append((row, agg8, idx, vals))
            elif typ in ("whead", "rhead"):
                hl = WH if typ == "whead" else RH
                pos = agg8 if typ == "whead" else dposT
                key = "o_agg" if typ == "whead" else "o_dpos"
                for c in subset:
                    L, h = hl[c]
                    sl = slice(h * hd, (h + 1) * hd)
                    if method == "patch":
                        v = cap[src][key][L][..., sl]
                        v = v if v.dim() == 2 else v[None]
                    else:
                        v = cap[base]["o_mean"][L][sl][None].expand(len(pos), -1)
                    H.head.setdefault(L, []).append((row, pos, h, v))
            elif typ == "sae":
                z = cap[base]["z1"]
                delta = torch.zeros_like(cap[base]["r1"])
                for c in subset:
                    f = FEATS[c]
                    tgt = cap[src]["z1"][f] if method == "patch" else zmean[f]
                    delta = delta + (tgt - z[f]) * Wdec[f]
                H.inject.setdefault(SAE_L, []).append((row, a1, None, delta * scale))

        def run_many(typ, subsets, method, base):
            """scores for a list of subsets (batched for full-forward carriers)."""
            out = []
            if typ == "pos":
                src = "unsafe" if base == "safe" else "safe"
                for sub in subsets:
                    pos = [E["agg16"][c] for c in sub]
                    out.append(kv_score(kv[base], kv[src], pos) if pos else kvS[base])
                return out
            for b0 in range(0, len(subsets), BATCH):
                chunk = subsets[b0:b0 + BATCH]
                H.reset()
                for r, sub in enumerate(chunk):
                    add_patch(typ, sub, method, base, r)
                lg = fwd(P[base + "_ids"], nrow=BATCH)          # always BATCH rows (padding rows unpatched)
                H.reset()
                out += [score(lg[r]) for r in range(len(chunk))]
            return out

        def effects(typ, ablate_c, S_x, den, rms):
            n = NT[typ]
            if typ == "pos":
                c = cc.clone_cache(kv["safe"], dpos)
                if ablate_c is not None:
                    pp = E["agg16"][ablate_c]
                    for i in range(nl):
                        c.layers[i].keys[:, :, pp] = kv["unsafe"].layers[i].keys[:, :, pp]
                        c.layers[i].values[:, :, pp] = kv["unsafe"].layers[i].values[:, :, pp]
                leaves = []
                for i in range(nl):
                    c.layers[i].keys.requires_grad_(True); c.layers[i].values.requires_grad_(True)
                    leaves.append((c.layers[i].keys, c.layers[i].values))
                with torch.enable_grad():
                    o = model.model(input_ids=torch.tensor([[P["last"]]], device="cuda"), past_key_values=c,
                                    cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
                    lg = o.last_hidden_state[0, -1].float() @ W_Uf.T
                    (lg[toi["safe"]] - lg[toi["unsafe"]]).backward()
                eff = np.zeros(n)
                for j in range(n):
                    pp = E["agg16"][j]
                    e = 0.0
                    for i, (K, V) in enumerate(leaves):
                        e += float(((K.detach()[:, :, pp] - kv["unsafe"].layers[i].keys[:, :, pp]).float() * K.grad[:, :, pp].float()).sum())
                        e += float(((V.detach()[:, :, pp] - kv["unsafe"].layers[i].values[:, :, pp]).float() * V.grad[:, :, pp].float()).sum())
                    eff[j] = e / den
                del c, leaves, o
                torch.cuda.empty_cache()
                return eff, None
            H.reset()
            if ablate_c is not None:
                add_patch(typ, [ablate_c], "patch", "safe", 0)
            H.capture = True
            H.grad = typ != "rhead"
            lg = fwd(P["safe_ids"], grad=H.grad)[0]
            if H.grad:
                (lg[toi["safe"]] - lg[toi["unsafe"]]).backward()
            eff = np.zeros(n); pool = None
            if typ == "neuron":
                for j, (L, i) in enumerate(NEUR):
                    x = H.cap_d[L]
                    eff[j] = float(((x.detach()[0, agg8, i].float() - cap["unsafe"]["d_agg"][L][:, i]) * x.grad[0, agg8, i].float()).sum()) / den
            elif typ == "whead":
                for j, (L, h) in enumerate(WH):
                    x = H.cap_o[L]; sl = slice(h * hd, (h + 1) * hd)
                    eff[j] = float(((x.detach()[0, agg8, sl].float() - cap["unsafe"]["o_agg"][L][:, sl]) * x.grad[0, agg8, sl].float()).sum()) / den
            elif typ == "sae":
                r = H.cap_r[SAE_L]
                g = r.grad[0, a1].float()
                z = sae.encode((r.detach()[0, a1].float() / scale)[None])[0]
                for j, f in enumerate(FEATS):
                    eff[j] = float((z[f] - cap["unsafe"]["z1"][f]) * scale * (Wdec[f] @ g)) / den
            elif typ == "rhead":
                with torch.no_grad():
                    ld = (W_U[toi["safe"]].float() - W_U[toi["unsafe"]].float()) * gfin
                    pool = {}
                    for L in range(min(L for L, _ in RH), nl):
                        x = H.cap_o[L][0, dpos].float()
                        xc = cap["unsafe"]["o_dpos"][L]
                        Wo = model.model.layers[L].self_attn.o_proj.weight.float()
                        for h in range(nh):
                            sl = slice(h * hd, (h + 1) * hd)
                            pool[(L, h)] = float((Wo[:, sl] @ (x[sl] - xc[sl])) @ ld / rms) / den
                    for j, (L, h) in enumerate(RH):
                        eff[j] = pool[(L, h)]
            H.reset()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return eff, pool

        def resid_rms(typ=None, c=None):
            got = {}
            hh = model.model.norm.register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0][0, -1].float()))
            H.reset()
            if c is not None:
                add_patch(typ, [c], "patch", "safe", 0)
            fwd(P["safe_ids"])
            hh.remove(); H.reset()
            x = got["x"]
            return float(torch.sqrt((x ** 2).mean() + eps))

        for typ in ("pos", "whead", "neuron", "sae", "rhead"):
            S_c, S_x = (kvS["safe"], kvS["unsafe"]) if typ == "pos" else S_ff
            den = S_c - S_x
            ret = lambda s: (s - S_x) / den
            n = NT[typ]
            methods = ("patch",) if typ == "pos" else ("patch", "mean")
            R = results["types"].setdefault(typ, {"curve": {}, "single": [], "compensation": [], "additivity": []})
            for method in methods:
                for k in kgrid(n):
                    subs = [list(range(n))] if k == n else [list(rng.choice(n, k, replace=False)) for _ in range(NSUB)]
                    vals = [ret(s) for s in run_many(typ, subs, method, "safe")]
                    R["curve"].setdefault(method, {}).setdefault(str(k), []).append(float(np.mean(vals)))
            singles_sub = [[c] for c in range(n)]
            suf = [ret(s) for s in run_many(typ, singles_sub, "patch", "unsafe")]
            nec = [1 - ret(s) for s in run_many(typ, singles_sub, "patch", "safe")]
            necm = [None] * n if typ == "pos" else [1 - ret(s) for s in run_many(typ, singles_sub, "mean", "safe")]
            singles = [{"c": c, "suff": suf[c], "nec": nec[c], "nec_mean": necm[c]} for c in range(n)]
            joint_nec = 1 - ret(run_many(typ, [list(range(n))], "patch", "safe")[0])
            joint_suf = ret(run_many(typ, [list(range(n))], "patch", "unsafe")[0])
            R["single"].append({"scn": scn, "oid": oid, "fi": fi, "singles": singles})
            R["additivity"].append({"scn": scn, "oid": oid, "fi": fi, "sum_single_nec": float(sum(nec)),
                                    "joint_nec": joint_nec, "sum_single_suff": float(sum(suf)), "joint_suff": joint_suf})
            rms0 = resid_rms() if typ == "rhead" else None
            base, pool0 = effects(typ, None, S_x, den, rms0)
            comp = []
            for c in range(n):
                rmsc = resid_rms(typ, c) if typ == "rhead" else None
                after, pool1 = effects(typ, c, S_x, den, rmsc)
                d_others = float(np.sum(np.delete(after - base, c)))
                row = {"c": c, "eff_before": float(base[c]), "eff_after_self": float(after[c]),
                       "others_change": d_others, "ratio": d_others / base[c] if abs(base[c]) > 1e-6 else None}
                if typ == "rhead":
                    Lc = RH[c][0]
                    keys = [kk for kk in pool1 if kk != RH[c] and kk[0] >= Lc]
                    dp = float(sum(pool1[kk] - pool0[kk] for kk in keys))
                    row["pool_change_all_heads_ge_layer"] = dp
                    row["pool_ratio"] = dp / base[c] if abs(base[c]) > 1e-6 else None
                comp.append(row)
            R["compensation"].append({"scn": scn, "oid": oid, "fi": fi, "base_eff": [float(x) for x in base], "rows": comp})
            print(f"[{scn}/{oid}/f{fi}] {typ}: joint_nec={joint_nec:+.3f} sum_single_nec={sum(nec):+.3f} "
                  f"joint_suff={joint_suf:+.3f} sum_single_suff={sum(suf):+.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(os.path.join(OUT, f"t1b_{TAG}.json"), "w"), indent=1, default=float)

    summ = {}
    for typ, R in results["types"].items():
        s = {"curve": {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for m, d in R["curve"].items()}}
        allS = [x for inst in R["single"] for x in inst["singles"]]
        suf = np.array([x["suff"] for x in allS]); nec = np.array([x["nec"] for x in allS])
        s["single"] = {"suff_mean": float(suf.mean()), "suff_max": float(suf.max()), "nec_mean": float(nec.mean()),
                       "nec_max": float(nec.max()), "n": len(allS),
                       "n_high_suff_low_nec(suff>0.3,nec<0.1)": int(((suf > 0.3) & (nec < 0.1)).sum()),
                       "median_suff_over_nec_for_suff>0.05": float(np.median((suf / np.maximum(nec, 1e-3))[suf > 0.05])) if (suf > 0.05).any() else None,
                       "corr_suff_nec": float(np.corrcoef(suf, nec)[0, 1]) if suf.std() > 0 and nec.std() > 0 else None}
        A = R["additivity"]
        s["additivity"] = {k: float(np.mean([a[k] for a in A])) for k in ("sum_single_nec", "joint_nec", "sum_single_suff", "joint_suff")}
        rows = [r for inst in R["compensation"] for r in inst["rows"]]
        big = [r for r in rows if r["ratio"] is not None and abs(r["eff_before"]) > 0.02]
        s["compensation"] = {"n": len(big),
                             "mean_ratio": float(np.mean([r["ratio"] for r in big])) if big else None,
                             "median_ratio": float(np.median([r["ratio"] for r in big])) if big else None,
                             "pooled_ratio": float(sum(r["others_change"] for r in big) / sum(r["eff_before"] for r in big)) if big else None,
                             "self_after_over_before": float(sum(r["eff_after_self"] for r in big) / sum(r["eff_before"] for r in big)) if big else None}
        if typ == "rhead":
            pr = [r for r in big if r.get("pool_ratio") is not None]
            s["compensation"]["pool_all_heads_ge_layer_pooled_ratio"] = float(sum(r["pool_change_all_heads_ge_layer"] for r in pr) / sum(r["eff_before"] for r in pr)) if pr else None
        summ[typ] = s
    results["summary"] = summ
    json.dump(results, open(os.path.join(OUT, f"t1b_{TAG}.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    print("T1B_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
