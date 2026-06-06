"""What ARE the tokens that selective recompute picks? Decode them and their meaning.

For a gating task we compute, per downstream token: the decision's attention to it, its KV-change
under the field edit, and its individual causal decision-recovery (patch that one position). We then
print the top tokens by decision-attention (the ones selective recompute recomputes) with their text,
and render the prompt with the selected positions marked — to see whether they are meaningful (the
gating rule, the conclusion/action words, the decision cue, the field, punctuation, or sinks).
Run: MECH_ATTN=eager python esys/selective_tokens.py --model Qwen/Qwen3-1.7B
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
from mech_suite import load
from transformers.cache_utils import DynamicCache
import diverse_tasks as DT


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def dscore(model, cache, last, pos, tc, ts):
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(cache, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    return float(o.logits[0, -1, tc] - o.logits[0, -1, ts])


@torch.no_grad()
def patch1(model, co, cn, i, dpos, last, tc, ts):
    w = clone(co, dpos)
    for L in range(len(w.layers)):
        w.layers[L].keys[:, :, i, :] = cn.layers[L].keys[:, :, i, :]
        w.layers[L].values[:, :, i, :] = cn.layers[L].values[:, :, i, :]
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=w,
              cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
    return float(o.logits[0, -1, tc] - o.logits[0, -1, ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tasks", default="retail_refund,oncall_route,bank_withdraw")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    tok, model = load(args.model)

    def chat(s):
        try:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)

    for d in args.tasks.split(","):
        t = DT.TASKS[d]
        tc = tok(t["correct"], add_special_tokens=False)["input_ids"][0]
        ts = tok(t["stale"], add_special_tokens=False)["input_ids"][0]
        al = align_pair(tok, chat(DT.build(d, t["vold"])), chat(DT.build(d, t["vnew"])))
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1; last = int(nid[0, dpos])
        co = model(input_ids=oid.to("cuda"), use_cache=True).past_key_values
        cn = model(input_ids=nid.to("cuda"), use_cache=True).past_key_values
        s_old = dscore(model, co, int(oid[0, dpos]), dpos, tc, ts)
        s_new = dscore(model, cn, last, dpos, tc, ts); denom = (s_new - s_old) or 1e-6
        # decision attention (old context)
        o = model(input_ids=torch.tensor([[int(oid[0, dpos])]], device="cuda"), past_key_values=clone(co, dpos),
                  cache_position=torch.tensor([dpos], device="cuda"), use_cache=True, output_attentions=True)
        att = torch.stack([x[0] for x in o.attentions])[:, :, -1, :].mean(1).mean(0)   # [dpos+1]
        # kv-change per downstream position
        kvc = torch.zeros(dpos, device="cuda")
        for lc, ln in zip(co.layers, cn.layers):
            kvc[:dpos] += (ln.keys[0, :, :dpos] - lc.keys[0, :, :dpos]).norm(dim=(0, 2))
        # per-position causal recovery for ALL downstream positions (the ground-truth importance)
        recov = {i: (patch1(model, co, cn, i, dpos, last, tc, ts) - s_old) / denom for i in range(a, dpos)}
        att_order = sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)[:args.topk]
        rec_order = sorted(range(a, dpos), key=lambda i: recov[i], reverse=True)[:args.topk]

        print(f"\n===== {d}: field='{t['field']}' {t['vold']}->{t['vnew']}  (correct='{t['correct']}' vs stale='{t['stale']}') =====")
        print(f"  field span tokens [{a}:{b}] = {[tok.decode([int(x)]) for x in oid[0, a:b]]}")
        print(f"  --- TOP-{args.topk} by DECISION-ATTENTION (what selective recompute picks) ---")
        print(f"  {'pos':>4} {'token':<16} {'attn':>7} {'kv_chg':>7} {'recovery':>9}")
        for i in att_order:
            print(f"  {i:>4} {repr(tok.decode([int(oid[0, i])])):<16} {float(att[i]):>7.4f} {float(kvc[i]):>7.2f} {recov[i]:>9.3f}")
        print(f"  --- TOP-{args.topk} by CAUSAL RECOVERY (the tokens that actually carry the conclusion) ---")
        print(f"  {'pos':>4} {'token':<16} {'recovery':>9} {'attn':>7} {'kv_chg':>7}  context")
        for i in rec_order:
            ctx = tok.decode([int(x) for x in oid[0, max(a, i - 4):i + 2]])
            print(f"  {i:>4} {repr(tok.decode([int(oid[0, i])])):<16} {recov[i]:>9.3f} {float(att[i]):>7.4f} {float(kvc[i]):>7.2f}  ...{ctx!r}")
    print("SELECTIVE_TOKENS_DONE")


if __name__ == "__main__":
    main()
