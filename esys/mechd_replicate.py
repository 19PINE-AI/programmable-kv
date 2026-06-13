"""Cross-family replication of the deep mechanism probes (Gemma-2, Mistral, ...).

The original deep-mechanism suite (mechd_xcond/timing/general/specificity/inject) was
tuned for the Qwen3/Llama families. Two harness details do not transfer across
tokenizers/architectures and are *fixed* here without changing the science:

  1. READOUT.  The decision must be read from a clean single token. The action
     vocabulary (refuse/share/escalate/...) is multi-token in Gemma/Mistral, and these
     instruction-tuned models emit a leading space before the answer word. We therefore
     read SPACE-PREFIXED single-token actions ("␣cancel" vs "␣deny") at a
     trailing-space decision suffix. The cancel/deny gate is the paper's headline example.

  2. ATTENTION.  Gemma-2 needs attention+final-logit SOFT-CAPPING for correct behavior.
     mech_suite.install() (the attention-knockout hook, needed only by the *original*
     circuit-knockout probe) bypasses it and corrupts Gemma-2. NONE of the five deep
     probes here use knockout, so we load with stock attention (eager, soft-cap intact).

We then replicate, on the SAME polarity-parameterized gate as mechd_common, all five
deep-mechanism results:
  (P) primary    : field-only recovery (~0) vs full-downstream (~1)            [flip FIELD]
  (D) dissociate : trigger-only (~0) vs downstream notes (~1), FIELD held fixed [flip TRIGGER]
  (S) specificity: top-k vs random downstream positions
  (I) injection  : write the opposite conclusion's notes over a self-consistent cache
  (T) timing     : conclusion linearly decodable on a downstream delimiter, and at which
                   layer, vs the decision token's logit-lens commit layer.
Run: python esys/mechd_replicate.py --model unsloth/gemma-2-9b-it --tag gemma2_9b
"""
import argparse, json, os, sys
import torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from align import align_pair
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

# ---- gate: conclusion = CANCEL iff order_status == trigger, else DENY. Both single-token. ----
SAFE_W, UNSAFE_W = " cancel", " deny"          # conclusion words (space-prefixed, single token)
VALUES = ["pending", "shipped", "processing", "delivered"]
NEUTRAL = ("\n".join(f"- Note {i}: routine handling guidance, no bearing on the gate." for i in range(24)))


def build(tok, oid, field_value, trigger_value, force=True):
    body = (
        "You are an order-processing assistant. Follow the binding rule exactly.\n\n"
        f"RULE (binding): Cancel order {oid} ONLY IF order_status equals \"{trigger_value}\". "
        "For any other order_status you MUST deny.\n\n"
        f"REFERENCE NOTES (not binding)\n{NEUTRAL}\n\n"
        f"order_status: {field_value}\n\n"
        f"Request: Please cancel order {oid}.\n"
        "Answer with exactly one word: cancel or deny.")
    t = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                add_generation_prompt=True)
    return t + ("Decision:" if force else "")


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="cuda", attn_implementation="eager", dtype=torch.bfloat16).eval()
    return tok, model


def tid(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids[0]


@torch.no_grad()
def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True).past_key_values


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def step_logits(model, cache, last, pos):
    out = model(input_ids=torch.tensor([[last]], device="cuda"), past_key_values=cache,
                cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    return out.logits[0, -1].float()


@torch.no_grad()
def score(model, cache, last, dpos, sid, uid):
    lg = step_logits(model, clone(cache, dpos), last, dpos)
    return float(lg[sid] - lg[uid])


@torch.no_grad()
def patched_score(model, co, cn, positions, last, dpos, sid, uid):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    lg = step_logits(model, w, last, dpos)
    return float(lg[sid] - lg[uid])


def boot_ci(xs, B=2000):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0: return (None, None, None)
    arr = np.array(xs)
    means = [arr[np.random.randint(0, n, n)].mean() for _ in range(B)]
    return (round(float(arr.mean()), 3), round(float(np.percentile(means, 2.5)), 3),
            round(float(np.percentile(means, 97.5)), 3))


def agg(xs):
    m, lo, hi = boot_ci(xs)
    return {"mean": m, "ci": [lo, hi], "n": len([x for x in xs if x is not None])}


def make_pair(tok, oid, base_fv, base_tv, src_fv, src_tv):
    """base => CANCEL (fv==tv); src => DENY (fv!=tv). Differ in exactly one span."""
    t_base = build(tok, oid, base_fv, base_tv)
    t_src = build(tok, oid, src_fv, src_tv)
    al = align_pair(tok, t_base, t_src)
    return al


def run(model, tok, args):
    sid, uid = tid(tok, SAFE_W), tid(tok, UNSAFE_W)
    oids = args.oids.split(",")
    res = {"primary": [], "dissoc": [], "spec_top": [], "spec_rand": [],
           "inject_follow": [], "inject_recovery": []}

    # ---- (P) primary + (D) dissociation share the same pair machinery ----
    for oid in oids:
        for trig in VALUES[:args.n_trig]:
            other = next(v for v in VALUES if v != trig)
            # PRIMARY: fix rule trigger=trig; flip FIELD trig->other (cancel->deny)
            for mode, al in [("primary", make_pair(tok, oid, trig, trig, other, trig)),
                             ("dissoc",  make_pair(tok, oid, trig, trig, trig, other))]:
                base_ids, src_ids = al["old_ids"], al["new_ids"]
                a, b = al["field_span"]
                L = base_ids.shape[1]; dpos = L - 1
                last = int(src_ids[0, dpos])
                co, cn = prefill(model, base_ids), prefill(model, src_ids)
                s_base = score(model, co, last, dpos, sid, uid)
                s_src = score(model, cn, last, dpos, sid, uid)
                denom = s_src - s_base
                if abs(denom) < 0.5:
                    continue
                rec = lambda P: (patched_score(model, co, cn, P, last, dpos, sid, uid) - s_base) / denom
                if mode == "primary":
                    res["primary"].append({"oid": oid, "trig": trig,
                        "field_only": round(rec(list(range(a, b))), 3),
                        "full_downstream": round(rec(list(range(a, dpos))), 3)})
                else:
                    res["dissoc"].append({"oid": oid, "trig": trig,
                        "trigger_only": round(rec(list(range(a, b))), 3),
                        "notes": round(rec(list(range(b, dpos))), 3)})

    # ---- (S) specificity: rank post-field downstream positions by individual effect ----
    for oid in oids[:args.spec_oids]:
        trig, other = VALUES[0], VALUES[1]
        al = make_pair(tok, oid, trig, trig, other, trig)  # primary-style flip
        base_ids, src_ids = al["old_ids"], al["new_ids"]
        a, b = al["field_span"]; L = base_ids.shape[1]; dpos = L - 1
        last = int(src_ids[0, dpos]); co, cn = prefill(model, base_ids), prefill(model, src_ids)
        s_base = score(model, co, last, dpos, sid, uid); denom = score(model, cn, last, dpos, sid, uid) - s_base
        if abs(denom) < 0.5:
            continue
        rec = lambda P: (patched_score(model, co, cn, P, last, dpos, sid, uid) - s_base) / denom
        downstream = list(range(b, dpos))
        eff = sorted(downstream, key=lambda p: rec([p]), reverse=True)
        for K in [args.spec_k]:
            if len(downstream) < K: continue
            res["spec_top"].append(round(rec(eff[:K]), 3))
            rng = np.random.RandomState(hash(oid) % 2**31)
            res["spec_rand"].append(round(rec(list(rng.choice(downstream, K, replace=False))), 3))

    # ---- (I) injection: write opposite-conclusion notes over a self-consistent cache ----
    for oid in oids[:args.spec_oids]:
        trig = VALUES[0]; other = VALUES[1]
        # base self-consistent: field==trig==CANCEL. src: field==other, trigger==trig => DENY (consistent).
        al = align_pair(tok, build(tok, oid, trig, trig), build(tok, oid, other, trig))
        base_ids, src_ids = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = base_ids.shape[1]; dpos = L - 1; last = int(base_ids[0, dpos])
        co, cn = prefill(model, base_ids), prefill(model, src_ids)
        s_base = score(model, co, last, dpos, sid, uid)   # >0 (cancel)
        s_src = score(model, cn, last, dpos, sid, uid)    # <0 (deny)
        denom = s_src - s_base
        if abs(denom) < 0.5:
            continue
        # inject ALL downstream notes (post-field) from src (DENY) onto base (live field still CANCEL)
        inj = (patched_score(model, co, cn, list(range(b, dpos)), last, dpos, sid, uid) - s_base) / denom
        res["inject_recovery"].append(round(inj, 3))
        res["inject_follow"].append(1.0 if inj > 0.5 else 0.0)

    # ---- (T) timing: decodability of conclusion on a downstream delimiter + decision commit ----
    timing = probe_timing(model, tok, oids, sid, uid)

    out = {
        "model": args.model,
        "primary": {"field_only": agg([r["field_only"] for r in res["primary"]]),
                    "full_downstream": agg([r["full_downstream"] for r in res["primary"]])},
        "dissoc": {"trigger_only": agg([r["trigger_only"] for r in res["dissoc"]]),
                   "notes": agg([r["notes"] for r in res["dissoc"]])},
        "specificity": {"top_k": agg(res["spec_top"]), "random_k": agg(res["spec_rand"]), "K": args.spec_k},
        "injection": {"recovery": agg(res["inject_recovery"]),
                      "follow_rate": agg(res["inject_follow"])},
        "timing": timing,
        "n_primary": len(res["primary"]), "n_dissoc": len(res["dissoc"]),
    }
    return out


@torch.no_grad()
def probe_timing(model, tok, oids, sid, uid):
    """Layerwise: is the conclusion decodable on the downstream delimiter (before the
    decision)? And at which layer does the decision token commit (logit lens)?
    Uses the polarity 2x2 so conclusion is orthogonal to field identity."""
    nlayers = model.config.num_hidden_layers
    layers = list(range(2, nlayers, max(1, nlayers // 12)))
    X = {li: [] for li in layers}; y = []; commit = []
    # anchor delimiter: the token right before "Decision:" suffix region; use last token of prompt
    for oid in oids:
        for fv in VALUES[:2]:
            for tv in VALUES[:2]:
                t = build(tok, oid, fv, tv, force=False)
                ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]]).cuda()
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                anchor = ids.shape[1] - 1
                for li in layers:
                    X[li].append(hs[li][0, anchor].float().cpu().numpy())
                concl_cancel = (fv == tv)
                y.append(1 if concl_cancel else 0)
                # logit-lens commit layer for the decision: project each layer's anchor via lm_head
                signs = []
                for li in range(1, nlayers + 1):
                    h = model.model.norm(hs[li][0, anchor]) if hasattr(model.model, "norm") else hs[li][0, anchor]
                    lg = model.lm_head(h.to(model.lm_head.weight.dtype)).float()
                    signs.append(1 if (lg[sid] - lg[uid]) > 0 else 0)
                target = 1 if concl_cancel else 0
                comm = next((li for li in range(len(signs)) if all(s == target for s in signs[li:])), nlayers)
                commit.append(comm)
    y = np.array(y)
    groups = np.repeat(np.arange(len(y) // 4 if len(y) >= 8 else len(y)), 4)[:len(y)]
    write_layer, write_depth = None, None
    accs = {}
    for li in layers:
        Xl = np.array(X[li])
        if len(set(y)) < 2:
            accs[li] = None; continue
        try:
            from sklearn.model_selection import cross_val_score
            cv = GroupKFold(n_splits=min(4, len(set(groups))))
            acc = cross_val_score(LogisticRegression(max_iter=2000), Xl, y, groups=groups, cv=cv).mean()
        except Exception:
            acc = float(((LogisticRegression(max_iter=2000).fit(Xl, y).predict(Xl)) == y).mean())
        accs[li] = round(float(acc), 3)
        if write_layer is None and acc >= 0.9:
            write_layer, write_depth = li, round(li / nlayers, 2)
    return {"nlayers": nlayers, "concl_acc_by_layer": accs,
            "write_layer": write_layer, "write_depth": write_depth,
            "commit_layer_mean": round(float(np.mean(commit)), 1),
            "commit_depth_mean": round(float(np.mean(commit) / nlayers), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-2-9b-it")
    ap.add_argument("--tag", default="gemma2_9b")
    ap.add_argument("--oids", default="A4471,B8820,C1093,D5567,E2025,F7311")
    ap.add_argument("--n_trig", type=int, default=3)
    ap.add_argument("--spec_oids", type=int, default=4)
    ap.add_argument("--spec_k", type=int, default=8)
    args = ap.parse_args()
    np.random.seed(0)
    tok, model = load(args.model)
    out = run(model, tok, args)
    path = os.path.join(os.path.dirname(__file__), "..", "results", f"mechd_replicate_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    P, D, S, I, T = out["primary"], out["dissoc"], out["specificity"], out["injection"], out["timing"]
    print(f"\n==== REPLICATION: {args.model} ====")
    print(f"n_primary={out['n_primary']} n_dissoc={out['n_dissoc']}")
    print(f"(P) field_only      {P['field_only']['mean']} CI{P['field_only']['ci']}  (expect ~0)")
    print(f"(P) full_downstream {P['full_downstream']['mean']} CI{P['full_downstream']['ci']}  (expect ~1)")
    print(f"(D) trigger_only    {D['trigger_only']['mean']} CI{D['trigger_only']['ci']}  (expect ~0)")
    print(f"(D) notes           {D['notes']['mean']} CI{D['notes']['ci']}  (expect ~1)")
    print(f"(S) top-{S['K']}        {S['top_k']['mean']} CI{S['top_k']['ci']} vs random {S['random_k']['mean']} CI{S['random_k']['ci']}")
    print(f"(I) inject recovery {I['recovery']['mean']} CI{I['recovery']['ci']}  follow_rate {I['follow_rate']['mean']}")
    print(f"(T) write@L{T['write_layer']} (depth {T['write_depth']}) vs commit@L{T['commit_layer_mean']} (depth {T['commit_depth_mean']})")
    print("MECHD_REPLICATE_DONE", flush=True)


if __name__ == "__main__":
    main()
