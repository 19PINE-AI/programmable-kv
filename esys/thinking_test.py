"""Does live chain-of-thought rescue cheap leave-stale? (the thinking critique)

Real tool-calling agents THINK before acting. Thinking tokens are decoded LIVE
against the patched cache, attending to the *refreshed* field -- exactly the H1
load-bearing path. So a cheap FIELD-ONLY refresh (leave everything else stale) may
recover decisions that, without thinking, needed ~full reprefill.

We compare, with a real thinking model (Qwen3-8B, enable_thinking=True), the final
post-</think> tool call under:
  oracle_new   full new prefill                         (ground truth)
  oracle_old   full old prefill                          (pre-flip)
  field_only   faithful: refresh ONLY the field span, leave all else stale (cheapest)
  stale_full   refresh nothing                           (floor)
Reports tool-name agreement and the number of thinking tokens used.
"""
import argparse, json, os, sys, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture  # noqa
from align import align_pair
from run_e2 import load_model, prefill, clone_cache
import scenarios as S
import mechanism as M

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def build_thinking(tok, scn_key, value, n_neutral, hoist=False):
    ctx = S.build(scn_key, value, n_neutral, hoist=hoist)
    return tok.apply_chat_template([{"role": "user", "content": ctx}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def _extract_tool(answer):
    """Tool name from the post-</think> answer: prefer the one after 'tool_call:'."""
    m = re.search(r"tool_call:\s*([A-Za-z_]\w*)\s*\(", answer)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z_]\w*)\s*\(", answer)
    return m.group(1) if m else ""


@torch.no_grad()
def decode_think(model, tok, cache, last_tok, start, max_new=2048, post_budget=96):
    """Greedy-decode think+answer. After </think> appears, generate a fixed window
    (post_budget tokens) so the tool-call line is fully emitted, then stop."""
    toks = []
    cur = torch.tensor([[last_tok]], device="cuda"); pos = start
    eos = tok.eos_token_id
    countdown = None
    for _ in range(max_new):
        out = model(input_ids=cur, past_key_values=cache,
                    cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
        nxt = int(out.logits[0, -1].argmax()); toks.append(nxt); pos += 1
        if nxt == eos:
            break
        if countdown is not None:
            countdown -= 1
            # also stop early if a full tool-call line + newline already emitted
            if countdown <= 0:
                break
        elif "</think>" in tok.decode(toks):
            countdown = post_budget
        cur = torch.tensor([[nxt]], device="cuda")
    full = tok.decode(toks, skip_special_tokens=False)
    reached = "</think>" in full
    if reached:
        pre, answer = full.split("</think>", 1)
        think_tokens = len(tok(pre, add_special_tokens=False)["input_ids"])
    else:
        answer = full; think_tokens = len(toks)  # never closed -> ran out of budget
    return {"tool": _extract_tool(answer) if reached else "(unfinished)",
            "think_tokens": think_tokens, "reached_answer": reached,
            "answer_head": answer.strip()[:90]}


def run_one(tok, model, scn_key, n_neutral, max_new):
    s = S.SCENARIOS[scn_key]
    ot = build_thinking(tok, scn_key, s["v_old"], n_neutral)
    nt = build_thinking(tok, scn_key, s["v_new"], n_neutral)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]; upto = T - 1
    nid = al["new_ids"]; last = nid[0, upto]
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])

    def dec(cache):
        return decode_think(model, tok, clone_cache(cache, upto), last, upto, max_new)

    oracle_new = dec(cn)
    oracle_old = decode_think(model, tok, clone_cache(co, upto), al["old_ids"][0, upto], upto, max_new)
    # faithful field-only leave-stale (cheapest realizable patch)
    field_cache, nfield = M.patchkv_cache(model, nid, co, (a, b), 0, upto)
    field_only = dec(field_cache)
    stale_full = dec(co)

    o = oracle_new["tool"]
    return {"scenario": scn_key, "cls": s["cls"], "seq_len": T,
            "field_span": [a, b], "field_recompute_frac": nfield / T,
            "oracle_new": oracle_new, "oracle_old": oracle_old,
            "field_only": field_only, "stale_full": stale_full,
            "decision_changed": oracle_old["tool"] != o,
            "field_only_recovers": field_only["tool"] == o,
            "stale_recovers": stale_full["tool"] == o}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b_think")
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--scenarios", default="account_role,safety_mode,subscription_tier,timestamp")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for k in args.scenarios.split(","):
        r = run_one(tok, model, k, args.n_neutral, args.max_new)
        recs.append(r)
        print(f"\n=== {k} [{r['cls']}] changed={int(r['decision_changed'])} "
              f"field_only_recovers={int(r['field_only_recovers'])} "
              f"(field refresh = {r['field_recompute_frac']*100:.1f}% recompute)")
        for c in ["oracle_new", "oracle_old", "field_only", "stale_full"]:
            m = r[c]
            print(f"    {c:11s} tool={m['tool']:24s} think_tok={m['think_tokens']}  '{m['answer_head'][:46]}'")
    json.dump(recs, open(os.path.join(RES, f"thinking_{args.tag}.json"), "w"), indent=2)
    print("\nwrote", os.path.join(RES, f"thinking_{args.tag}.json"))


if __name__ == "__main__":
    main()
