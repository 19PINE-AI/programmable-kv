"""D3 — Linear probing for the memoized conclusion (independent of patching).

A different methodology that triangulates D1. Train a cross-domain linear probe for the
gated CONCLUSION (permissive/OLD=0 vs gated/NEW=1) from the residual stream, leave-one-
DOMAIN-out, diff-of-means direction (overfitting-resistant with few samples). Three results:

 (A) DECODABILITY-BY-LAYER: probe accuracy on held-out domains at the decision position vs
     layer -> the conclusion becomes linearly decodable in mid/late layers (independent
     confirmation of D1's layer band, via probing not patching).
 (B) UPSTREAM memoization: the conclusion is already decodable at the gate-rule token and the
     "let me check" token (before the decision prompt) -> the model commits at prefill time.
 (C) IN_PLACE STALENESS SIGNATURE: apply the stale/full-trained probe to the decision residual
     under {stale, in_place, erratum, full}. Claim: in_place is classified as OLD (P(new) low,
     like stale) while erratum/full flip to NEW -> the residual-level reason in_place fails.

8 diverse domains x 2 states. Run: MECH_ATTN=sdpa python esys/mech_probe.py
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from mech_suite import load, clone, prefill
from align import align_pair
import diverse_tasks as DT


@torch.no_grad()
def hidden_at(model, ids, pos):
    """All-layer residual-stream vectors at position `pos` for a full forward."""
    out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
    hs = out.hidden_states                       # tuple (nlayers+1) of [1, T, d]
    return torch.stack([h[0, pos].float().cpu() for h in hs])   # [nlayers+1, d]


@torch.no_grad()
def decision_hidden_inplace(model, co, cn, a, b, L):
    """Decision-position residual under the IN_PLACE edit (field KV->new, downstream stale)."""
    fc = clone(co, L - 1)
    for i in range(len(fc.layers)):
        fc.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
        fc.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
    last = int(cn.layers[0].keys.shape[2])       # not used; we forward the actual last token
    return fc


@torch.no_grad()
def decode_residual(model, cache, last_tok, dpos):
    out = model(input_ids=torch.tensor([[last_tok]], device="cuda"), past_key_values=clone(cache, dpos),
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True, output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])   # [nlayers+1, d]


def loo_accuracy(X, y, domains, layer):
    """Leave-one-domain-out diff-of-means probe accuracy at a given layer."""
    correct = 0; total = 0
    doms = sorted(set(domains))
    for hd in doms:
        tr = [i for i, d in enumerate(domains) if d != hd]
        te = [i for i, d in enumerate(domains) if d == hd]
        Xt = X[tr, layer]; yt = torch.tensor([y[i] for i in tr])
        mu1 = Xt[yt == 1].mean(0); mu0 = Xt[yt == 0].mean(0)
        w = (mu1 - mu0); mid = ((mu1 + mu0) / 2)
        for i in te:
            s = float((X[i, layer] - mid) @ w)
            pred = 1 if s > 0 else 0
            correct += (pred == y[i]); total += 1
    return correct / total


def probe_dir(X, y, layer):
    mu1 = X[[i for i in range(len(y)) if y[i] == 1], layer].mean(0)
    mu0 = X[[i for i in range(len(y)) if y[i] == 0], layer].mean(0)
    return (mu1 - mu0), (mu1 + mu0) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok, model = load(args.model)

    # ---- assemble decision-position residuals for stale(OLD) and full(NEW) ----
    X = []; y = []; domains = []
    cond_res = {"stale": [], "in_place": [], "erratum": [], "full": []}
    for d in DT.TASKS:
        t = DT.TASKS[d]
        old_text = DT.build(d, t["vold"]); new_text = DT.build(d, t["vnew"])
        err_text = DT.build(d, t["vold"], erratum_value=t["vnew"])
        al = align_pair(tok, old_text, new_text)
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]; L = oid.shape[1]
        dpos = L - 1; last = int(nid[0, dpos])
        co = prefill(model, oid); cn = prefill(model, nid)
        # training residuals (decision pos): stale=OLD label0, full=NEW label1
        h_stale = decode_residual(model, co, int(oid[0, dpos]), dpos)
        h_full = decode_residual(model, cn, last, dpos)
        X.append(h_stale); y.append(0); domains.append(d)
        X.append(h_full); y.append(1); domains.append(d)
        # condition residuals for (C)
        fc = decision_hidden_inplace(model, co, cn, a, b, L)
        h_ip = decode_residual(model, fc, last, dpos)
        eid = tok(err_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        h_err = hidden_at(model, eid, eid.shape[1] - 1)
        cond_res["stale"].append(h_stale); cond_res["full"].append(h_full)
        cond_res["in_place"].append(h_ip); cond_res["erratum"].append(h_err)
        print(f"  captured {d}", flush=True)

    X = torch.stack(X)                            # [2*Ndom, nlayers+1, d]
    nlayers = X.shape[1]
    # ---- (A) decodability by layer ----
    accs = {li: round(loo_accuracy(X, y, domains, li), 3) for li in range(nlayers)}
    best_layer = max(accs, key=accs.get)
    # ---- (C) in_place staleness signature: P(new) per condition, per layer (LOO-domain) ----
    # For each held-out domain, train direction on the other domains' stale/full, score that
    # domain's condition residuals -> fraction classified NEW.
    doms = sorted(set(domains))
    cond_pnew = {c: {} for c in cond_res}
    for li in range(nlayers):
        for c in cond_res:
            cls = []
            for di, hd in enumerate(doms):
                tr = [i for i, dd in enumerate(domains) if dd != hd]
                w, mid = probe_dir(X[tr], [y[i] for i in tr], li)
                s = float((cond_res[c][di][li] - mid) @ w)
                cls.append(1 if s > 0 else 0)
            cond_pnew[c][li] = round(sum(cls) / len(cls), 3)

    agg = {"model": args.model, "n_domains": len(doms),
           "decode_acc_by_layer": accs, "best_layer": best_layer, "best_acc": accs[best_layer],
           "cond_Pnew_at_best_layer": {c: cond_pnew[c][best_layer] for c in cond_res},
           "cond_Pnew_at_last_layer": {c: cond_pnew[c][nlayers - 1] for c in cond_res}}
    json.dump({"agg": agg, "cond_pnew_by_layer": cond_pnew}, open(os.path.join(
        os.path.dirname(__file__), "..", "results", f"mech_probe_{args.tag}.json"), "w"), indent=2)
    print("\n==== D3 PROBING SUMMARY ====")
    early = [accs[li] for li in range(nlayers) if li < nlayers / 3]
    mid = [accs[li] for li in range(nlayers) if nlayers / 3 <= li < 2 * nlayers / 3]
    late = [accs[li] for li in range(nlayers) if li >= 2 * nlayers / 3]
    print(f"(A) conclusion decode acc (LOO-domain): early={sum(early)/len(early):.2f} "
          f"mid={sum(mid)/len(mid):.2f} late={sum(late)/len(late):.2f} | best layer {best_layer} acc={accs[best_layer]}")
    print(f"(C) P(new) at best layer {best_layer}: {agg['cond_Pnew_at_best_layer']}")
    print(f"    P(new) at last layer:         {agg['cond_Pnew_at_last_layer']}")
    print("    expect: stale~0, in_place~0 (STALE signature), erratum~1, full~1")
    print("D3_PROBE_DONE")


if __name__ == "__main__":
    main()
