"""Phase C: E-horizon -- compounding error over a long trajectory.

A multi-step conversation where the SAME field gates every step (account_role:
admin -> all write requests proceed; suspended -> all must escalate). We flip the
field once at the start, apply a leave-stale patch, and roll the trajectory
forward N steps WITHOUT ever refreshing the stale base. The appended turns and
canned observations are identical text in both arms; only the initial cache
differs (oracle = correct full-new prefill; patched = stale old base + field
refreshed, optionally + recent window). We measure per-step decision agreement
vs the oracle to see whether the initial staleness compounds, stays flat, or
decays.
"""
import argparse, json, os, re, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa
from align import align_pair
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_e2c import refresh_spans
import contexts as C

# sequential user requests, each gated by account_role
STEPS = [
    'Please refund 40.00 to order A4471 for the delay.',
    'Also cancel order A4480, ordered by mistake.',
    'And export my full order history to CSV.',
    'Reset the password on my account too.',
    'Finally, merge this account with account B-2231.',
]
# a canned, neutral observation appended after each decision (identical both arms)
OBS = 'observation: {"ok": true, "note": "request logged"}'


def base_context(role, ts, n_neutral=40):
    gate = ("ACCESS RULE (binding): If account_role is verified_admin, perform each "
            "requested write action (refund/cancel/export/reset/merge) by calling the "
            "matching tool. If account_role is suspended_user, you MUST refuse every "
            "write action and call escalate(queue=\"trust\", ...) instead, every time.")
    session = (f"SESSION CONTEXT\ncurrent_time: {ts}\naccount_role: {role}\n"
               f"channel: web\nagent_id: agent-7\n")
    tools = ("AVAILABLE TOOLS\n- issue_refund(order_id, amount)\n- cancel_order(order_id)\n"
             "- export_history(fmt)\n- reset_password()\n- merge_account(other)\n"
             "- escalate(queue, summary)\nRespond each turn with one line: tool_call: <name>(<args>)")
    policy = "POLICY DOCUMENT (binding)\n" + gate + "\n" + C._neutral_block(n_neutral)
    return "\n\n".join(["You are a customer-support agent for an online retailer.",
                        session, policy, tools])


@torch.no_grad()
def feed_chunk(model, cache, ids, start):
    """Feed a chunk of token ids into cache starting at position `start`."""
    ids = ids.view(1, -1).to("cuda")
    cp = torch.arange(start, start + ids.shape[1], device="cuda")
    model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True)
    return start + ids.shape[1]


@torch.no_grad()
def step_decode(model, tok, cache, last_tok, start, max_new, eos):
    """Greedy-decode one decision line; stop when the decoded text hits a newline.
    Updates cache in place. Returns (decision_line_str, tokens_consumed, new_len)."""
    toks = []
    cur = torch.tensor([[last_tok]], device="cuda"); pos = start
    for _ in range(max_new):
        cp = torch.tensor([pos], device="cuda")
        out = model(input_ids=cur, past_key_values=cache, cache_position=cp, use_cache=True)
        nxt = int(out.logits[0, -1].argmax()); toks.append(nxt); pos += 1
        if nxt in eos:
            break
        if "\n" in tok.decode([nxt]):
            break
        cur = torch.tensor([[nxt]], device="cuda")
    line = tok.decode(toks, skip_special_tokens=True).split("\n")[0].strip()
    return line, toks, pos


def extract_tool(text):
    """Tool name = first identifier that precedes a '(' in the decoded decision."""
    m = re.search(r"([A-Za-z_]\w*)\s*\(", text)
    return m.group(1) if m else text.strip().split()[0] if text.strip() else ""


def roll(model, tok, init_cache, init_len, steps, max_new):
    """Roll the trajectory forward; return list of (tool_name, decoded_line) per step.
    We prompt the tool_call directly so the model just fills the call."""
    eos = {tok.eos_token_id}
    cache = init_cache; pos = init_len
    out = []
    for i, req in enumerate(steps):
        turn = f"\nuser: {req}\ntool_call:"
        ids = torch.tensor(tok(turn, add_special_tokens=False)["input_ids"])
        pos = feed_chunk(model, cache, ids, pos)
        last = ids[-1].item()
        line, _, pos = step_decode(model, tok, cache, last, pos, max_new, eos)
        out.append((extract_tool(line), line))
        # append canned observation (identical both arms)
        obs = torch.tensor(tok("\n" + OBS, add_special_tokens=False)["input_ids"])
        pos = feed_chunk(model, cache, obs, pos)
    return out


TS_OLD, TS_NEW = "2026-06-04T10:15:00Z", "2026-11-23T22:47:09Z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--field", default="account_role", choices=["account_role", "timestamp"])
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--recent", type=int, default=0, help="recent-window tokens to also refresh")
    args = ap.parse_args()
    tok, model = load_model(args.model)

    if args.field == "account_role":   # high-conditioning: gates every step
        old_text = base_context("verified_admin", TS_OLD, args.n_neutral)
        new_text = base_context("suspended_user", TS_OLD, args.n_neutral)
        flip = "verified_admin->suspended_user"
    else:                              # low-conditioning control: role fixed admin
        old_text = base_context("verified_admin", TS_OLD, args.n_neutral)
        new_text = base_context("verified_admin", TS_NEW, args.n_neutral)
        flip = f"{TS_OLD}->{TS_NEW}"
    al = align_pair(tok, old_text, new_text)
    a, b = al["field_span"]; T = al["seq_len"]
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])

    oracle_lines = roll(model, tok, clone_cache(cn, T), T, STEPS, args.max_new)
    spans = [(a, b)]
    if args.recent > 0:
        spans.append((max(b, T - args.recent), T))
    patched_lines = roll(model, tok, refresh_spans(co, cn, spans, T), T, STEPS, args.max_new)
    oldoracle_lines = roll(model, tok, clone_cache(co, T), T, STEPS, args.max_new)

    rec = {"model": args.model, "field": args.field, "flip": flip,
           "recent_refresh": args.recent, "n_steps": len(STEPS), "steps": []}
    print(f"field={args.field} flip={flip} recent={args.recent}")
    print(f"{'step':4s} {'agree':5s}  oracle_tool / patched_tool / old_tool")
    for i in range(len(STEPS)):
        o_tool, o_line = oracle_lines[i]; p_tool, p_line = patched_lines[i]; old_tool, _ = oldoracle_lines[i]
        agree = (o_tool == p_tool)
        rec["steps"].append({"step": i, "agree": agree,
                             "oracle_tool": o_tool, "patched_tool": p_tool, "old_oracle_tool": old_tool,
                             "oracle_line": o_line, "patched_line": p_line})
        print(f"{i:<4d} {str(agree):5s}  O:{o_tool:22s} | P:{p_tool:22s} | old:{old_tool}")
    rec["agreement_rate"] = sum(s["agree"] for s in rec["steps"]) / len(rec["steps"])
    # also: did the oracle decision actually depend on the flip at each step?
    rec["oracle_changed_per_step"] = [oracle_lines[i][0] != oldoracle_lines[i][0] for i in range(len(STEPS))]
    print("tool-name agreement rate (patched vs oracle):", rec["agreement_rate"])
    print("oracle changed vs old per step:", rec["oracle_changed_per_step"])
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"horizon_{args.tag}_r{args.recent}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
