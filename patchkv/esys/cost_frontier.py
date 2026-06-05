"""E-sys cost/latency frontier (real measurements).

Given an already-cached OLD context, a field changes. Measure the wall-clock to produce
a decode-ready cache (CUDA events, warmup+median) and the recompute-token count for each
method, swept across context length. The OLD prefill is assumed amortized (cached); we
charge only the INCREMENTAL recompute, except full_reprefill which redoes everything.

  full_reprefill : prefill the entire new context           (ceiling)
  hoist_to_end   : recompute field-at-suffix only           (real baseline)
  field_only     : recompute the field tokens (in-context)  (cheapest patch)
  erratum        : recompute the appended erratum suffix      (robust patch)
  field_erratum  : field tokens + erratum suffix
"""
import argparse, json, os, sys, statistics
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import load, clone, prefill, step
from align import align_pair, _common_prefix_len
import scenarios as S


def chat(tok, content):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def timed(fn, trials=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(statistics.median(ts), 2)


@torch.no_grad()
def recompute_span(model, ids, base_cache, start, end):
    c = clone(base_cache, start)
    if end > start:
        model(input_ids=ids[:, start:end].to("cuda"), past_key_values=c,
              cache_position=torch.arange(start, end, device="cuda"), use_cache=True)
    return c


def run(model, tok, n_neutral):
    s = S.SCENARIOS["account_role"]
    t_old = chat(tok, S.build("account_role", s["v_old"], n_neutral))
    t_new = chat(tok, S.build("account_role", s["v_new"], n_neutral))
    t_err = chat(tok, S.build("account_role", s["v_old"], n_neutral, erratum_value=s["v_new"]))
    t_hold = chat(tok, S.build("account_role", s["v_old"], n_neutral, hoist=True))
    t_hnew = chat(tok, S.build("account_role", s["v_new"], n_neutral, hoist=True))
    oid = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    eid = torch.tensor([tok(t_err, add_special_tokens=False)["input_ids"]])
    hno = torch.tensor([tok(t_hold, add_special_tokens=False)["input_ids"]])
    hnn = torch.tensor([tok(t_hnew, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]
    co = prefill(model, oid); T = oid.shape[1]
    p = _common_prefix_len(oid[0].tolist(), eid[0].tolist())          # erratum insertion
    hco = prefill(model, hno); hp = _common_prefix_len(hno[0].tolist(), hnn[0].tolist())  # hoist field pos
    res = {"n_neutral": n_neutral, "T": T, "methods": {}}
    def rec(name, fn, ntok):
        res["methods"][name] = {"latency_ms": timed(fn), "recompute_tokens": int(ntok),
                                "recompute_frac": round(ntok / T, 4)}
    rec("full_reprefill", lambda: prefill(model, nid), T)
    rec("hoist_to_end", lambda: recompute_span(model, hnn, hco, hp, hnn.shape[1] - 1), hnn.shape[1] - 1 - hp)
    rec("field_only", lambda: recompute_span(model, nid, co, a, b), b - a)
    rec("erratum", lambda: recompute_span(model, eid, co, p, eid.shape[1] - 1), eid.shape[1] - 1 - p)
    rec("field_erratum", lambda: (recompute_span(model, nid, co, a, b),
                                  recompute_span(model, eid, co, p, eid.shape[1] - 1)),
        (b - a) + (eid.shape[1] - 1 - p))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--neutrals", default="20,80,200,500")
    args = ap.parse_args()
    tok, model = load(args.model)
    out = {"model": args.model, "sweep": []}
    for nn in [int(x) for x in args.neutrals.split(",")]:
        r = run(model, tok, nn)
        out["sweep"].append(r)
        ms = r["methods"]
        print(f"T={r['T']:5d}: full={ms['full_reprefill']['latency_ms']:7.1f}ms "
              f"hoist={ms['hoist_to_end']['latency_ms']:6.1f}ms({ms['hoist_to_end']['recompute_frac']*100:.1f}%) "
              f"field={ms['field_only']['latency_ms']:6.1f}ms({ms['field_only']['recompute_frac']*100:.2f}%) "
              f"erratum={ms['erratum']['latency_ms']:6.1f}ms({ms['erratum']['recompute_frac']*100:.1f}%) "
              f"field+err={ms['field_erratum']['latency_ms']:6.1f}ms", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"cost_frontier_{args.tag}.json"), "w"), indent=2)
    print("COST_FRONTIER_DONE", flush=True)


if __name__ == "__main__":
    main()
