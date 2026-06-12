"""Exp3 - A sparse SAE feature that carries the conclusion note.

We train a TopK sparse autoencoder on the layer-L residual stream (L = the causal conclusion
layer from Exp2) gathered over a diverse set of task prompts (all scenarios x order ids x the
2x2 field/trigger conditions, all token positions). We then ask whether the field-conditioned
CONCLUSION at the aggregator is carried by a small number of monosemantic SAE features:

  1. identify CONCLUSION features: SAE codes whose activation on the aggregator separates
     SAFE vs UNSAFE conclusions (field-controlled), ranked by AUC; report how few there are.
  2. causal sufficiency: in a UNSAFE (corrupt) prefill, clamp the top conclusion feature(s) at
     the aggregator to their SAFE level (inject along the SAE DECODER direction) -> decode the
     decision -> recovery toward SAFE; vs clamping random features (control).
  3. necessity: ablate (zero) the feature in a SAFE prefill -> decision drops toward UNSAFE.
  4. ERRATUM link: a stale-field prompt with an appended salient erratum (the editkv fix) makes
     the SAME conclusion feature fire on the aggregator at its SAFE level, while the stale prompt
     leaves it at UNSAFE level -- tying the interpretable feature to the paper's edit operation.

Feature selection uses train order-ids; the causal tests use a HELD-OUT order id.
Run: python esys/circ_sae.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b --layer 12
"""
import argparse, json, os, sys
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import circuit_common as cc
from mechd_common import POL, build_pol


# ----------------------------- TopK SAE -----------------------------
class TopKSAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.randn(d, m) * (1.0 / d ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.randn(m, d) * (1.0 / m ** 0.5))

    def normalize_dec(self):
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        act = torch.relu(pre)
        topv, topi = act.topk(self.k, dim=-1)
        z = torch.zeros_like(act)
        z.scatter_(-1, topi, topv)
        return z

    def forward(self, x):
        z = self.encode(x)
        xhat = z @ self.W_dec + self.b_dec
        return xhat, z


@torch.no_grad()
def gather_acts(model, tok, layer, prompts, cap_positions=None):
    """Collect layer-`layer` residuals (after block `layer`) at all positions of each prompt."""
    acts = []
    for ids in prompts:
        out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0].float()      # [L, d]
        acts.append(h.cpu())
    return torch.cat(acts, 0)


def train_sae(X, d, m, k, steps, bs, lr, scale):
    sae = TopKSAE(d, m, k).cuda()
    sae.normalize_dec()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    Xg = (X / scale).cuda()
    n = Xg.shape[0]
    g = torch.Generator(device="cuda").manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g, device="cuda")
        x = Xg[idx]
        xhat, z = sae(x)
        loss = ((x - xhat) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        sae.normalize_dec()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                var = ((x - x.mean(0)) ** 2).sum(-1).mean()
                fvu = (loss / var).item()
            print(f"    sae step {step:5d} loss={loss.item():.3f} FVU={fvu:.3f}", flush=True)
    return sae


# ----------------------------- prompts -----------------------------
def pid(tok, scn, oid, field, trig):
    t = build_pol(tok, scn, oid, field, trig, False, True)
    return torch.tensor([tok(t, add_special_tokens=False)["input_ids"]])


GENERAL_TEXT = [
    "The committee reviewed the quarterly budget and approved three of the five proposals.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "She tightened the last bolt, wiped her hands, and stepped back to admire the engine.",
    "In distributed systems, consensus protocols must tolerate node failures and network delays.",
    "The recipe calls for two cups of flour, a pinch of salt, and a tablespoon of honey.",
    "After the storm passed, the harbor was littered with broken masts and tangled rope.",
    "Interest rates rose sharply, and the bond market reacted within minutes of the announcement.",
    "The museum's new wing houses ceramics, textiles, and a small collection of bronze tools.",
    "He debugged the parser for hours before realizing the off-by-one error in the index.",
    "Migratory birds navigate using the earth's magnetic field and the position of the sun.",
    "The contract stipulates a thirty-day notice period and a penalty for early termination.",
    "Volunteers planted four hundred saplings along the eroded bank of the river last spring.",
]


def general_prompts(tok, n_neutral=12):
    out = []
    for txt in GENERAL_TEXT:
        try:
            t = tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=False,
                                        add_generation_prompt=True)
        except Exception:
            t = txt
        out.append(torch.tensor([tok(t, add_special_tokens=False)["input_ids"]]))
    return out


def erratum_prompt(tok, scn, oid, field_stale, trig, correct_field):
    """Stale field with an appended salient erratum that supplies the corrected field late."""
    s = POL[scn]
    base = build_pol(tok, scn, oid, field_stale, trig, False, False)  # no force suffix yet
    err = (f"\n\n[STATE UPDATE] {s['label']} -> {correct_field}; this overrides any earlier "
           f"value and any earlier conclusion. Re-evaluate against the rule.\n")
    # append before the final 'tool_call:' scaffold
    return base + err + "tool_call:"


@torch.no_grad()
def agg_code(model, sae, tok, scn, oid, field, trig, layer, off, scale):
    ids = pid(tok, scn, oid, field, trig)
    dpos = ids.shape[1] - 1; pos = dpos - off
    out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
    x = out.hidden_states[layer + 1][0, pos].float() / scale
    z = sae.encode(x.unsqueeze(0))[0]
    return z.cpu().numpy()


@torch.no_grad()
def feat_act_window(model, sae, ids, feat, layer, scale, window=30):
    """Max activation of `feat` over the last `window` positions before the decision token
    (robust to where the late note lands; used for the erratum prompt whose layout shifts)."""
    out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
    dpos = ids.shape[1] - 1
    lo = max(0, dpos - window)
    X = out.hidden_states[layer + 1][0, lo:dpos].float() / scale
    Z = sae.encode(X)
    return float(Z[:, feat].max())


class ResidInject:
    def __init__(self, model, layer, pos, vec):
        self.layer = cc.decoder_layers(model)[layer]; self.pos = pos; self.vec = vec; self.h = None
    def __enter__(self):
        def hook(mod, args, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] > self.pos:
                hs[0, self.pos] = hs[0, self.pos] + self.vec.to(hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        self.h = self.layer.register_forward_hook(hook); return self
    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def clamp_score(model, tok, sae, scn, oid, field, trig, layer, off, feats, target_acts, scale, toi):
    """Inject sum_f (target_f - z_f) * scale * W_dec[f] at the aggregator; return conc_score."""
    ids = pid(tok, scn, oid, field, trig)
    dpos = ids.shape[1] - 1; pos = dpos - off; last = int(ids[0, dpos])
    out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
    x = out.hidden_states[layer + 1][0, pos].float() / scale
    z = sae.encode(x.unsqueeze(0))[0]
    delta = torch.zeros(sae.b_dec.shape[0], device="cuda")
    for f, tgt in zip(feats, target_acts):
        delta = delta + (tgt - z[f]) * sae.W_dec[f]
    delta = delta * scale
    with ResidInject(model, layer, pos, delta):
        o2 = model(input_ids=ids.to("cuda"), use_cache=True)
    lg = o2.logits[0, dpos].float()
    return cc.conc_score(lg, toi)


def auc(pos_vals, neg_vals):
    """AUC that feature activation > threshold separates pos (SAFE) from neg (UNSAFE)."""
    pos_vals = np.asarray(pos_vals); neg_vals = np.asarray(neg_vals)
    n = 0; c = 0.0
    for p in pos_vals:
        c += (neg_vals < p).sum() + 0.5 * (neg_vals == p).sum(); n += len(neg_vals)
    return c / n if n else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--dict", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    tok, model = cc.load_eager(args.model)
    nh, hd, hidden, nl = cc.cfg_dims(model)
    L = args.layer
    train_oids = cc.OIDS[:3]; test_oid = cc.OIDS[3]

    # 1) gather training activations (diverse: all scns x train oids x 2x2 conditions)
    print("gathering SAE training activations...", flush=True)
    prompts = []
    for scn in cc.SCNS:
        vA, vB = POL[scn]["values"]
        for oid in train_oids:
            for field in (vA, vB):
                for trig in (vA, vB):
                    prompts.append(pid(tok, scn, oid, field, trig))
    prompts += general_prompts(tok)          # add general text for monosemanticity / diversity
    X = gather_acts(model, tok, L, prompts)
    scale = float(X.norm(dim=1).mean())
    print(f"  collected {X.shape[0]} activation vectors (d={X.shape[1]}), scale={scale:.1f}", flush=True)

    # 2) train SAE
    print("training TopK SAE...", flush=True)
    sae = train_sae(X, hidden, args.dict, args.k, args.steps, bs=2048, lr=1e-3, scale=scale)

    # aggregator offsets per (scn,oid) from the pair
    offs = {}
    toi_by_scn = {}
    for scn in cc.SCNS:
        s = POL[scn]
        toi_by_scn[scn] = {"safe": cc.ftok(tok, cc.ACT_TOK[s["safe"]]), "unsafe": cc.ftok(tok, cc.ACT_TOK[s["unsafe"]])}
        for oid in cc.OIDS:
            P = cc.build_pair(tok, scn, oid)
            agg_list, *_ = cc.find_aggregators(model, P, topn=1)
            offs[(scn, oid)] = P["dpos"] - agg_list[0]

    # 3) identify conclusion features on TRAIN: aggregator codes SAFE vs UNSAFE (field-controlled)
    print("identifying conclusion features...", flush=True)
    safe_codes, unsafe_codes = [], []
    for scn in cc.SCNS:
        vA, vB = POL[scn]["values"]
        for oid in train_oids:
            off = offs[(scn, oid)]
            # SAFE: field==trig ; UNSAFE: field!=trig ; both field values -> field controlled
            safe_codes.append(agg_code(model, sae, tok, scn, oid, vA, vA, L, off, scale))
            safe_codes.append(agg_code(model, sae, tok, scn, oid, vB, vB, L, off, scale))
            unsafe_codes.append(agg_code(model, sae, tok, scn, oid, vA, vB, L, off, scale))
            unsafe_codes.append(agg_code(model, sae, tok, scn, oid, vB, vA, L, off, scale))
    S = np.stack(safe_codes); U = np.stack(unsafe_codes)
    aucs = np.array([auc(S[:, f], U[:, f]) for f in range(args.dict)])
    sep = np.abs(aucs - 0.5)
    top_feats = list(np.argsort(sep)[::-1][:30])
    n_active = int(((S > 0).any(0) | (U > 0).any(0)).sum())
    auc_counts = {"|AUC-.5|>0.2": int((sep > 0.2).sum()), ">0.3": int((sep > 0.3).sum()),
                  ">0.4": int((sep > 0.4).sum()), ">0.45": int((sep > 0.45).sum())}
    best_auc = float(aucs[top_feats[0]])
    print(f"  active features on aggregator: {n_active}; AUC-separation counts: {auc_counts}", flush=True)
    print(f"  best single-feature AUC={best_auc:.3f}; top conclusion features (feat, AUC): "
          f"{[(int(f), round(float(aucs[f]),3)) for f in top_feats[:6]]}", flush=True)

    # SAFE/UNSAFE target activation levels for the top features (train means)
    safe_tgt = {int(f): float(S[:, f].mean()) for f in top_feats}
    unsafe_tgt = {int(f): float(U[:, f].mean()) for f in top_feats}

    # 4) causal tests on HELD-OUT oid: clamp the top-K conclusion features JOINTLY (sweep K)
    print("causal clamp on held-out oid...", flush=True)
    KCL = [1, 3, 10, 30]
    suff = {k: [] for k in KCL}; nec = {k: [] for k in KCL}
    suff_ctrl = {k: [] for k in KCL}
    erratum_fires = []
    g = np.random.default_rng(0)
    for scn in cc.SCNS:
        vA, vB = POL[scn]["values"]; off = offs[(scn, test_oid)]; toi = toi_by_scn[scn]
        s_un = clamp_score(model, tok, sae, scn, test_oid, vA, vB, L, off, [], [], scale, toi)  # UNSAFE
        s_sa = clamp_score(model, tok, sae, scn, test_oid, vA, vA, L, off, [], [], scale, toi)  # SAFE
        denom = s_sa - s_un
        if abs(denom) < 0.5:
            continue
        for k in KCL:
            feats = [int(f) for f in top_feats[:k]]
            tgts = [safe_tgt[f] for f in feats]
            s_clamp = clamp_score(model, tok, sae, scn, test_oid, vA, vB, L, off, feats, tgts, scale, toi)
            suff[k].append((s_clamp - s_un) / denom)
            rfeats = [int(x) for x in g.choice(args.dict, k, replace=False)]
            rtg = [safe_tgt[top_feats[0]]] * k
            s_rc = clamp_score(model, tok, sae, scn, test_oid, vA, vB, L, off, rfeats, rtg, scale, toi)
            suff_ctrl[k].append((s_rc - s_un) / denom)
            tgtu = [unsafe_tgt[f] for f in feats]
            s_abl = clamp_score(model, tok, sae, scn, test_oid, vA, vA, L, off, feats, tgtu, scale, toi)
            nec[k].append((s_sa - s_abl) / denom)
        # erratum link: stale field (UNSAFE arrangement) + erratum supplying correct field -> SAFE conclusion;
        # does the top conclusion feature fire at SAFE level on the aggregator of the erratum prompt?
        ferr = int(top_feats[0])
        # stale: field=vB, trig=vA -> conclusion UNSAFE; erratum corrects field to vA -> SAFE.
        stale_ids = pid(tok, scn, test_oid, vB, vA)
        et = erratum_prompt(tok, scn, test_oid, vB, vA, vA)
        eids = torch.tensor([tok(et, add_special_tokens=False)["input_ids"]])
        z_stale = feat_act_window(model, sae, stale_ids, ferr, L, scale)
        z_err = feat_act_window(model, sae, eids, ferr, L, scale)
        erratum_fires.append({"scn": scn, "feat": ferr, "safe_level": round(safe_tgt[ferr], 3),
                              "unsafe_level": round(unsafe_tgt[ferr], 3),
                              "stale_act": round(float(z_stale), 3), "erratum_act": round(float(z_err), 3)})

    def stat(xs):
        return {"mean": round(float(np.mean(xs)), 3), "n": len(xs)} if xs else None
    summary = {
        "model": args.model, "layer": L, "dict": args.dict, "k": args.k,
        "n_active_features": n_active, "auc_counts": auc_counts, "best_single_auc": round(best_auc, 3),
        "top_features_auc": [(int(f), round(float(aucs[f]), 3)) for f in top_feats[:15]],
        "sufficiency_recovery_byK": {k: stat(v) for k, v in suff.items()},
        "sufficiency_control_byK": {k: stat(v) for k, v in suff_ctrl.items()},
        "necessity_drop_byK": {k: stat(v) for k, v in nec.items()},
        "erratum_link": erratum_fires,
    }
    out = {"summary": summary}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"circ_sae_{args.tag}.json"), "w"), indent=2)
    print("\n==== Exp3 SAE CONCLUSION FEATURE (%s, layer %d) ====" % (args.tag, L))
    print(f"dict={args.dict} k={args.k}; active on aggregator={n_active}; AUC counts={auc_counts}; best single AUC={best_auc:.3f}")
    print(f"top conclusion features (feat, AUC): {summary['top_features_auc'][:6]}")
    print("sufficiency: clamp top-K conclusion feats to SAFE in a UNSAFE run -> recovery (vs random-K control):")
    for k in KCL:
        print(f"   K={k:>2}: {summary['sufficiency_recovery_byK'][k]} | control {summary['sufficiency_control_byK'][k]} "
              f"| necessity {summary['necessity_drop_byK'][k]}")
    print("erratum link (does the conclusion feature fire when an erratum corrects a stale field?):")
    for e in erratum_fires:
        print(f"   [{e['scn']}] feat {e['feat']}: SAFE={e['safe_level']} UNSAFE={e['unsafe_level']} "
              f"| stale={e['stale_act']} erratum={e['erratum_act']}")
    print("CIRC_SAE_DONE", flush=True)


if __name__ == "__main__":
    main()
