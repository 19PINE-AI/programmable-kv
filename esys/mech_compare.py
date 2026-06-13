"""Reasoning vs non-reasoning MECHANISM comparison, across a model-size ladder.

Hypothesis (attention-mediated memoized inference):
  - NON-REASONING: the decision is HARMED BY the stale downstream. Field-only hedges;
    masking the decision's attention to the stale downstream FIXES it (prefix+field
    suffice). The field's correct value is present but out-voted by the memoized inference.
  - REASONING: the CoT generates FRESH downstream tokens that re-integrate the field.
    The decision now DEPENDS ON the CoT: masking the decision's attention to the original
    stale downstream barely hurts, but masking its attention to the CoT reverts it.

Per model x context(benign/poison): field-only cache; decode the action; knock out the
decision token's attention to {original-stale-downstream, CoT} and read the 3-way action.
account_role: safe=escalate, unsafe=issue_refund, cautious=lookup.
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from align import align_pair
import scenarios as S
import stress_thinking as PZ

_KO = {"on": False, "keys": None}


def install(model):
    import importlib
    mod = importlib.import_module(type(model).__module__)
    orig = mod.eager_attention_forward

    def patched(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
        if _KO["on"] and _KO["keys"]:
            kk = k.shape[2]
            add = torch.zeros(1, 1, q.shape[2], kk, device=q.device, dtype=q.dtype)
            add[:, :, -1, [x for x in _KO["keys"] if x < kk]] = torch.finfo(q.dtype).min
            attention_mask = add if attention_mask is None else attention_mask + add
        return orig(module, q, k, v, attention_mask, scaling, dropout, **kw)
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


def ftok(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def three_way(lg, TOI):
    t = torch.tensor([lg[TOI["escalate"]], lg[TOI["issue_refund"]], lg[TOI["lookup"]]])
    return ["escalate", "issue_refund", "lookup"][int(t.argmax())], [round(float(x), 2) for x in t]


@torch.no_grad()
def step(model, cache_len_cache, last, pos, keys=None, want_logits=True):
    """Forward one token `last` at position `pos` against cache_len_cache (length pos)."""
    if keys:
        _KO["on"] = True; _KO["keys"] = keys
    out = model(input_ids=torch.tensor([[last]], device="cuda"), past_key_values=cache_len_cache,
                cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    _KO["on"] = False; _KO["keys"] = None
    return out.logits[0, -1].float()


def build_fieldonly_cache(model, tok, oid, nid, a, b, L):
    co, cn = prefill(model, oid), prefill(model, nid)
    fc = clone(co, L)
    for i in range(len(fc.layers)):
        fc.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
        fc.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
    return fc


@torch.no_grad()
def gen_cot(model, tok, fc, prompt_ids, L, max_new=2000):
    """Greedy-generate the CoT from field-only cache until </think>, then FORCE
    'tool_call:' so the decision position is well-defined. Returns
    (decision_query_token, decision_pos, cache, cot_end). Feeding decision_query at
    decision_pos predicts the tool name. CoT span = [L, cot_end)."""
    cache = clone(fc, L - 1)
    cur = int(prompt_ids[0, L - 1]); cur_pos = L - 1
    gen = []; eos = tok.eos_token_id
    for _ in range(max_new):
        lg = step(model, cache, cur, cur_pos)   # cache len -> cur_pos+1
        nx = int(lg.argmax()); gen.append(nx)
        cur = nx; cur_pos += 1
        txt = tok.decode(gen)
        if "</think>" in txt or nx == eos:
            break
    cot_end = cur_pos + 1            # cache will be len cur_pos+1 after feeding `cur`
    # now feed `cur` then force the scaffold "\ntool_call:"
    scaffold = tok("\ntool_call:", add_special_tokens=False)["input_ids"]
    seq = [cur] + scaffold
    pos = cur_pos
    for t in seq[:-1]:
        step(model, cache, t, pos); pos += 1     # append t at pos
    decision_query = seq[-1]
    return decision_query, pos, cache, cot_end    # cache len == pos; feed decision_query at pos


def analyze(model, tok, ctx, TOI, n_neutral=30):
    bfn = PZ.build if ctx == "poison" else (lambda s, v, n, e=None: S.build(s, v, n, erratum_value=e))
    res = {}
    # ---- positions / spans from the thinking prompt ----
    def mk(value, thinking):
        body = bfn("account_role", value, n_neutral)
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking)

    for thinking in [False, True]:
        suffix = "" if thinking else "tool_call:"   # non-reasoning: force the tool-name next
        t_old = mk("verified_admin", thinking) + suffix
        t_new = mk("suspended_user", thinking) + suffix
        oid = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
        nid = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
        al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid.shape[1]
        fc = build_fieldonly_cache(model, tok, oid, nid, a, b, L)

        if not thinking:
            # single-forward decision; KO original downstream [b, L-1)
            base = step(model, clone(fc, L - 1), int(nid[0, L - 1]), L - 1)
            ko = step(model, clone(fc, L - 1), int(nid[0, L - 1]), L - 1, keys=list(range(b, L - 1)))
            res["nonreasoning"] = {
                "baseline": three_way(base, TOI),
                "KO_orig_downstream": three_way(ko, TOI)}
        else:
            dq, dpos, cache, cot_end = gen_cot(model, tok, fc, nid, L)
            base = three_way(step(model, clone(cache, dpos), dq, dpos), TOI)
            ko_orig = three_way(step(model, clone(cache, dpos), dq, dpos, keys=list(range(b, L))), TOI)
            ko_cot = three_way(step(model, clone(cache, dpos), dq, dpos, keys=list(range(L, dpos))), TOI)
            res["reasoning"] = {"decision_pos": dpos, "cot_len": cot_end - L,
                                "baseline": base, "KO_orig_downstream": ko_orig, "KO_cot_all_fresh": ko_cot,
                                "spans": {"orig_downstream": [b, L], "fresh_cot": [L, dpos]}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok, model = load(args.model)
    TOI = {"escalate": ftok(tok, "escalate"), "issue_refund": ftok(tok, "issue"),
           "lookup": ftok(tok, "lookup")}
    out = {"model": args.model, "results": {}}
    for ctx in ["benign", "poison"]:
        out["results"][ctx] = analyze(model, tok, ctx, TOI)
        r = out["results"][ctx]
        nr = r.get("nonreasoning", {}); rr = r.get("reasoning", {})
        def amx(x): return x[0] if isinstance(x, (list, tuple)) else x
        print(f"\n[{ctx}] NONREASONING field-only: base={amx(nr.get('baseline'))} "
              f"KO_orig_downstream={amx(nr.get('KO_orig_downstream'))}", flush=True)
        print(f"[{ctx}] REASONING field-only (cot_len={rr.get('cot_len')}): base={amx(rr.get('baseline'))} "
              f"KO_orig_downstream={amx(rr.get('KO_orig_downstream'))} KO_cot={amx(rr.get('KO_cot'))}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_compare_{args.tag}.json"), "w"), indent=2)
    print("MECH_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
