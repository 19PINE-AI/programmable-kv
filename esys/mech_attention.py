"""Mechanistic test of the 'memoized field-conditioned inference' hypothesis.

Non-thinking decision = ONE decode step after 'tool_call:' -> we read the decision
directly. account_role: safe=escalate, unsafe=issue_refund, cautious=lookup.

Conditions (all are a single decode from a cache of length `pos`):
  stale       : old cache (old field + stale downstream)
  field_only  : old cache, field span KV swapped to NEW (downstream still stale)
  erratum     : old cache + appended suffix erratum
  oracle      : full NEW prefill

Measurements:
  (1) 3-way decision: softmax over {escalate,issue_refund,lookup}, argmax, 3-way entropy.
  (2) causal patching from stale: which fresh span moves esc-vs-unsafe? (field / gate /
      field+gate / all-downstream) -> localize the decisive (memoized-inference) span.
  (3) ATTENTION KNOCKOUT (decisive): in the FIELD-ONLY decode, mask the decision token's
      attention to the gate/conclusion span -> does it flip to escalate? If removing the
      deep stale path makes field-only faithful, the hypothesis holds.
  (4) attention attribution: field-only decode token's attn mass to field vs gate vs
      downstream.
"""
import argparse, json, os, sys, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from align import align_pair, _common_prefix_len
import scenarios as S
import stress_thinking as PZ

_KO = {"on": False, "keys": None}


def install(model):
    import importlib
    mod = importlib.import_module(type(model).__module__)
    orig = mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        if _KO["on"] and _KO["keys"]:
            b, h, q, d = query.shape
            kk = key.shape[2]
            add = torch.zeros(1, 1, q, kk, device=query.device, dtype=query.dtype)
            ks = [k for k in _KO["keys"] if k < kk]
            add[:, :, -1, ks] = torch.finfo(query.dtype).min   # last query -> masked keys
            attention_mask = add if attention_mask is None else attention_mask + add
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
    mod.eager_attention_forward = patched
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    install(model)
    return tok, model


def clone(cache, upto):
    c = DynamicCache()
    for i, l in enumerate(cache.layers):
        c.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return c


@torch.no_grad()
def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True).past_key_values


def first_tok(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def find_span(tok, ids_list, text):
    nd = tok(text, add_special_tokens=False)["input_ids"]
    for L in [len(nd), 20, 12, 8]:
        if L <= 0:
            continue
        sub = nd[:L]
        for i in range(len(ids_list) - L + 1):
            if ids_list[i:i + L] == sub:
                return (i, i + L)
    return None


@torch.no_grad()
def decode(model, tok, cache, last, pos, TOI, keys=None, want_attn=False):
    if keys:
        _KO["on"] = True; _KO["keys"] = keys
    out = model(input_ids=torch.tensor([[last]], device="cuda"),
                past_key_values=clone(cache, pos), cache_position=torch.tensor([pos], device="cuda"),
                use_cache=True, output_attentions=want_attn)
    _KO["on"] = False; _KO["keys"] = None
    lg = out.logits[0, -1].float()
    three = torch.tensor([lg[TOI["escalate"]], lg[TOI["issue_refund"]], lg[TOI["lookup"]]])
    sm = torch.softmax(three, 0)
    ent = float(-(sm * torch.log(sm + 1e-9)).sum())
    argmax = ["escalate", "issue_refund", "lookup"][int(three.argmax())]
    res = {"logits3": {k: round(float(v), 2) for k, v in zip(["escalate", "issue_refund", "lookup"], three)},
           "probs3": {k: round(float(v), 2) for k, v in zip(["escalate", "issue_refund", "lookup"], sm)},
           "argmax": argmax, "entropy3": round(ent, 3), "esc_minus_unsafe": round(float(three[0] - three[1]), 2)}
    if want_attn:
        att = torch.stack([a[0] for a in out.attentions])[:, :, -1, :].mean(1)  # [L, pos+1]
        res["_att"] = att
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok, model = load(args.model)
    TOI = {"escalate": first_tok(tok, "escalate"), "issue_refund": first_tok(tok, "issue"),
           "lookup": first_tok(tok, "lookup")}

    def text(ctx, value, ev=None):
        bfn = PZ.build if ctx == "poison" else (lambda s, v, n, e=None: S.build(s, v, n, erratum_value=e))
        body = bfn("account_role", value, 30, ev)
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + "tool_call:"

    out = {"model": args.model, "results": {}}
    for ctx in ["benign", "poison"]:
        t_old, t_new = text(ctx, "verified_admin"), text(ctx, "suspended_user")
        t_err = text(ctx, "verified_admin", ev="suspended_user")
        oid = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
        nid = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
        eid = torch.tensor([tok(t_err, add_special_tokens=False)["input_ids"]])
        al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; pos = oid.shape[1] - 1
        last_new = nid[0, pos]; nlist = nid[0].tolist()
        gate = find_span(tok, nlist, (PZ.POISON["account_role"]["gate"] if ctx == "poison"
                                      else S.SCENARIOS["account_role"]["gate"])[:80])
        concl = (find_span(tok, nlist, "is permitted for this account") or
                 find_span(tok, nlist, "requested refund is permitted") or
                 find_span(tok, nlist, "permitted")) if ctx == "poison" else None
        co, cn = prefill(model, oid), prefill(model, nid)
        fc = clone(co, pos)                                   # field-only
        for i in range(len(fc.layers)):
            fc.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
            fc.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
        p = _common_prefix_len(oid[0].tolist(), eid[0].tolist()); ue = eid.shape[1] - 1
        ew = clone(co, p)
        model(input_ids=eid[:, p:ue].to("cuda"), past_key_values=ew,
              cache_position=torch.arange(p, ue, device="cuda"), use_cache=True)

        r = {"spans": {"field": [a, b], "gate": gate, "concl": concl}}
        # (1) decisions
        r["decision"] = {
            "stale": decode(model, tok, co, oid[0, pos], pos, TOI),
            "field_only": decode(model, tok, fc, last_new, pos, TOI),
            "erratum": decode(model, tok, ew, eid[0, ue], ue, TOI),
            "oracle": decode(model, tok, cn, last_new, pos, TOI)}
        # (2) patching from stale
        def patch(spans):
            c = clone(co, pos)
            for (s0, s1) in spans:
                for i in range(len(c.layers)):
                    c.layers[i].keys[:, :, s0:s1, :] = cn.layers[i].keys[:, :, s0:s1, :]
                    c.layers[i].values[:, :, s0:s1, :] = cn.layers[i].values[:, :, s0:s1, :]
            return decode(model, tok, c, last_new, pos, TOI)["esc_minus_unsafe"]
        r["patch"] = {"stale": patch([]), "field": patch([(a, b)]),
                      "gate": patch([gate]) if gate else None,
                      "field+gate": patch([(a, b), gate]) if gate else None,
                      "all_downstream": patch([(a, b), (b, pos)])}
        # (3) knockout in the FIELD-ONLY decode: mask decision->{gate/concl} and ->all-downstream
        ks_gate = (list(range(*gate)) if gate else []) + (list(range(*concl)) if concl else [])
        ks_all = list(range(b, pos))   # all stale downstream (field already refreshed)
        kg = decode(model, tok, fc, last_new, pos, TOI, keys=ks_gate) if ks_gate else None
        ka = decode(model, tok, fc, last_new, pos, TOI, keys=ks_all)
        r["knockout"] = {"field_only_baseline_argmax": r["decision"]["field_only"]["argmax"],
                         "field_only_baseline": r["decision"]["field_only"]["esc_minus_unsafe"],
                         "KO_gateconcl": (kg["esc_minus_unsafe"], kg["argmax"]) if kg else None,
                         "KO_all_downstream": (ka["esc_minus_unsafe"], ka["argmax"])}
        # (4) attention attribution in field-only decode
        d = decode(model, tok, fc, last_new, pos, TOI, want_attn=True)
        att = d.pop("_att")
        def mass(s): return round(float(att[:, s[0]:s[1]].sum(1).mean()), 4) if s else None
        r["attn_attr"] = {"field": mass((a, b)), "gate": mass(gate), "concl": mass(concl),
                          "downstream_total": round(float(att[:, b:pos].sum(1).mean()), 4)}
        out["results"][ctx] = r
        D = r["decision"]
        print(f"\n[{ctx}] decisions: stale={D['stale']['argmax']} field_only={D['field_only']['argmax']} "
              f"erratum={D['erratum']['argmax']} oracle={D['oracle']['argmax']}", flush=True)
        print(f"   patch esc-unsafe: stale={r['patch']['stale']} field={r['patch']['field']} "
              f"gate={r['patch']['gate']} all={r['patch']['all_downstream']}", flush=True)
        print(f"   KNOCKOUT field_only base={r['knockout']['field_only_baseline_argmax']}: "
              f"KO_gateconcl={r['knockout']['KO_gateconcl']} KO_all_downstream={r['knockout']['KO_all_downstream']}", flush=True)
        print(f"   attn mass field/gate/concl = {r['attn_attr']['field']}/{r['attn_attr']['gate']}/{r['attn_attr']['concl']}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_attention_{args.tag}.json"), "w"), indent=2)
    print("\nMECH_DONE", flush=True)


if __name__ == "__main__":
    main()
