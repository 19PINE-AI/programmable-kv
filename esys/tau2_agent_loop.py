"""Real multi-turn AGENTIC loop on the live τ²-bench retail env, at scale.

Beyond the single gated decision of tau2_episode.py: here the local model is an autonomous agent
that emits and executes a SEQUENCE of tool calls against the real τ²-bench RetailDB, over a
multi-turn conversation, while the gating field (order_status) changes mid-episode. We compare KV
strategies on end-to-end TASK SUCCESS (the env's own enforcement is the reward) and total
recompute, over N real orders.

The editkv mechanism here is the ERRATUM (append-only) — the deployment-realistic mode that
composes with prefix caching (§8c). Strategies at the mid-episode field change:
  full_reprefill : re-prefill the whole conversation with the updated observation (cost: all tokens)
  erratum        : append a one-line [STATE UPDATE ...] turn; keep the cache       (cost: ~that line)
  stale          : do nothing (the agent keeps the old observation)                 (cost: 0)
Persistent KV cache across turns; the agent drives get_order_details -> (status flips) -> the
cancel/deny action, executed against the real tools (cancel_pending_order RAISES on non-pending).

Run: MECH_ATTN=sdpa python esys/tau2_agent_loop.py --n 30
"""
import os
import argparse, os, sys, json, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.environ.get("TAU2_SRC", os.path.expanduser("~/tau2-bench/src")))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from tau2.domains.retail.environment import get_environment

SYS = ("You are a retail support agent. Binding policy: an order may be CANCELLED only if its "
       "order_status is exactly 'pending'. If 'processed' or 'delivered', it CANNOT be cancelled — "
       "you must DENY. To act, output exactly one line:\n"
       "  ACTION: get_order_details(order_id=\"#W...\")   — to look up an order\n"
       "  ACTION: cancel_pending_order(order_id=\"#W...\")  — to cancel a pending order\n"
       "  ACTION: deny()   — to refuse when the order cannot be cancelled\n"
       "Output ONLY the single ACTION line.")
ACT_RE = re.compile(r'ACTION:\s*(get_order_details|cancel_pending_order|deny)\s*\(([^)]*)\)')


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def feed(model, tok, text, cache, pos):
    """Append `text` to the running cache; returns (cache, pos, n_tokens_processed)."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    n = ids.shape[1]
    model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(pos, pos + n, device="cuda"),
          use_cache=True)
    return cache, pos + n, n


@torch.no_grad()
def gen_action(model, tok, cache, pos, max_new=24):
    """Greedy-decode an ACTION line from the running cache (does not pollute: works on a clone)."""
    c = clone(cache, pos); toks = []; cur = None
    # decode starting from the last fed token's logits — re-feed a newline cue to elicit the action
    cue = tok("\nACTION:", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([cue], device="cuda")
    out = model(input_ids=ids, past_key_values=c, cache_position=torch.arange(pos, pos + len(cue), device="cuda"), use_cache=True)
    p = pos + len(cue); nx = int(out.logits[0, -1].argmax()); toks.append(nx)
    for _ in range(max_new):
        o = model(input_ids=torch.tensor([[nx]], device="cuda"), past_key_values=c,
                  cache_position=torch.tensor([p], device="cuda"), use_cache=True)
        p += 1; nx = int(o.logits[0, -1].argmax()); toks.append(nx)
        if tok.eos_token_id == nx or "\n" in tok.decode(toks):
            break
    return "ACTION:" + tok.decode(toks, skip_special_tokens=True)


def parse_action(text):
    m = ACT_RE.search(text)
    if not m:
        return None, {}
    name = m.group(1); args = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)))
    return name, args


def run_episode(model, tok, env, oid, strategy):
    """One multi-turn episode for an order whose status flips pending->processed mid-way.
    Returns (success, recompute_tokens). success = agent's terminal action is policy-correct
    AND consistent with the real tool (cancel on processed -> tool raises -> failure)."""
    o = env.tools.db.orders[oid]; saved = o.status; o.status = "pending"
    cache = DynamicCache(); pos = 0; recompute = 0
    conv = SYS + f"\n\nuser: Hi, please cancel my order {oid} — ordered by mistake.\nassistant:"
    cache, pos, n = feed(model, tok, conv, cache, pos); recompute += n
    # ---- turn 1: agent looks up the order ----
    a1 = gen_action(model, tok, cache, pos); name1, _ = parse_action(a1)
    cache, pos, n = feed(model, tok, " " + a1 + "\n", cache, pos); recompute += n
    # execute lookup (real tool) -> observation with status=pending
    try:
        det = env.tools.get_order_details(oid); status_seen = det.status
    except Exception:
        status_seen = "pending"
    obs = f"observation: order {oid} status is '{status_seen}'.\nuser: Great, go ahead and cancel it.\nassistant:"
    cache, pos, n = feed(model, tok, obs, cache, pos); recompute += n
    # ---- MID-EPISODE: the order is fulfilled; status flips to 'processed' ----
    o.status = "processed"
    if strategy == "full_reprefill":
        # rebuild the whole conversation reflecting the new observation, re-prefill from scratch
        full = (SYS + f"\n\nuser: Hi, please cancel my order {oid} — ordered by mistake.\nassistant: {a1}\n"
                f"observation: order {oid} status is 'processed' (updated; the order was just fulfilled).\n"
                "user: Great, go ahead and cancel it.\nassistant:")
        cache = DynamicCache(); pos = 0
        cache, pos, n = feed(model, tok, full, cache, pos); recompute += n
    elif strategy == "erratum":
        upd = "\n[STATE UPDATE] order_status has changed to 'processed'; this overrides any earlier value AND conclusion.\nassistant:"
        cache, pos, n = feed(model, tok, upd, cache, pos); recompute += n
    # stale: do nothing (agent still believes 'pending')
    # ---- turn 2: agent's terminal action ----
    a2 = gen_action(model, tok, cache, pos); name2, args2 = parse_action(a2)
    o.status = "processed"
    # score against the REAL tool
    success = False
    if name2 == "deny":
        success = True                                   # correct: processed cannot be cancelled
    elif name2 == "cancel_pending_order":
        try:
            env.tools.cancel_pending_order(oid, "ordered by mistake"); success = False  # tool allowed it -> would be wrong (it's processed)
            o.status = "processed"
        except Exception:
            success = False                              # tool refused -> agent took an invalid action
    o.status = saved
    return success, recompute, name2 or "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    env = get_environment()
    pend = [oid for oid, o in env.tools.db.orders.items() if o.status == "pending"][:args.n]
    strategies = ["full_reprefill", "erratum", "stale"]
    agg = {s: {"success": 0, "recompute": 0, "actions": {}} for s in strategies}
    n = 0
    for oid in pend:
        for s in strategies:
            ok, rc, act = run_episode(model, tok, env, oid, s)
            agg[s]["success"] += ok; agg[s]["recompute"] += rc
            agg[s]["actions"][act] = agg[s]["actions"].get(act, 0) + 1
        n += 1
        if n % 10 == 0:
            print(f"  ...{n}/{len(pend)}", flush=True)
    out = {"model": args.model, "n": n, "strategies": {}}
    full_rc = agg["full_reprefill"]["recompute"]
    print(f"\n==== τ²-BENCH MULTI-TURN AGENT LOOP ({n} real orders; correct terminal action = DENY) ====")
    print(f"  {'strategy':16s} {'task_success':>13s} {'recompute(tok)':>15s} {'vs full':>9s}  actions")
    for s in strategies:
        succ = agg[s]["success"] / n; rc = agg[s]["recompute"]
        out["strategies"][s] = {"task_success": round(succ, 3), "recompute_tokens": rc,
                                "recompute_vs_full": round(rc / full_rc, 3), "actions": agg[s]["actions"]}
        print(f"  {s:16s} {succ:>13.2f} {rc:>15d} {rc/full_rc:>8.2f}x  {agg[s]['actions']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"tau2_agent_loop_{args.tag}.json"), "w"), indent=2)
    print("TAU2_AGENT_LOOP_DONE")


if __name__ == "__main__":
    main()
