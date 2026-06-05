"""Robustness gaps: (A) multi-field simultaneous edits + interference, (B) multi-edit
sequential accumulation. Both via the erratum mechanism; deterministic logit forced-choice.

(A) MULTI-FIELD: a decision gated by TWO fields (account_role AND order_status). We flip one,
    the other, or both, and check the erratum recovers the correct joint decision AND that
    editing one field does NOT corrupt the other's contribution (interference control).
(B) MULTI-EDIT: one field changes through a sequence of values (pending->processed->cancelled).
    Stacked sequential erratums must track the LATEST value (no confusion from intermediates),
    and must match a single erratum straight to the final value.
Run: MECH_ATTN=sdpa python esys/robustness_multi.py
"""
import argparse, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLE = "You are a retail support agent."
RULE2 = ("POLICY (binding): Perform the refund ONLY IF account_role is admin AND order_status "
         "is pending. In ALL other cases you MUST deny.")
ERR = "[STATE UPDATE] {f} has changed to {v}; this overrides any earlier value AND any earlier conclusion.\n"
OIDS = ["A4471", "B8820", "C1093", "D5567"]


@torch.no_grad()
def decide(model, tok, text, a="refund", b="deny"):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]]).to("cuda")
    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    ta = tok(a, add_special_tokens=False)["input_ids"][0]
    tb = tok(b, add_special_tokens=False)["input_ids"][0]
    return a if lg[ta] >= lg[tb] else b


def build2(role_v, status_v, oid, updates=None):
    s = (f"{ROLE}\n\nSESSION\norder: {oid}\naccount_role: {role_v}\norder_status: {status_v}\n\n{RULE2}\n\n")
    for f, v in (updates or []):
        s += ERR.format(f=f, v=v)
    s += "\nTASK\nThe user requests a $40 refund. Decide one word: refund or deny.\nDecision:"
    return s


def multi_field(model, tok):
    # ground truth: refund iff (role==admin and status==pending)
    print("=== (A) multi-field (account_role AND order_status gate the refund) ===")
    conds = [  # (role, status, changed-from-old updates needed); OLD = (admin, pending)
        ("flip_role", "suspended", "pending", [("account_role", "suspended")]),
        ("flip_status", "admin", "processed", [("order_status", "processed")]),
        ("flip_both", "suspended", "processed", [("account_role", "suspended"), ("order_status", "processed")]),
        ("flip_none", "admin", "pending", []),
    ]
    tally = {k: {"stale": 0, "erratum": 0, "oracle": 0} for k, *_ in conds}
    for name, role_v, status_v, ups in conds:
        correct = "refund" if (role_v == "admin" and status_v == "pending") else "deny"
        for oid in OIDS:
            oracle = decide(model, tok, build2(role_v, status_v, oid))                     # true values baked in
            stale = decide(model, tok, build2("admin", "pending", oid))                    # old values, no edit
            err = decide(model, tok, build2("admin", "pending", oid, updates=ups))         # old values + erratum(s)
            tally[name]["oracle"] += (oracle == correct)
            tally[name]["stale"] += (stale == correct)
            tally[name]["erratum"] += (err == correct)
        n = len(OIDS)
        print(f"  {name:11s} correct={correct:6s} | oracle={tally[name]['oracle']}/{n} "
              f"stale={tally[name]['stale']}/{n} erratum={tally[name]['erratum']}/{n}", flush=True)
    return {k: {m: f"{v}/{len(OIDS)}" for m, v in d.items()} for k, d in tally.items()}


RULE1 = "POLICY (binding): An order can be cancelled ONLY IF order_status is pending; otherwise deny."


def build_seq(status_v, oid, updates=None):
    s = f"{ROLE}\n\nSESSION\norder: {oid}\norder_status: {status_v}\n\n{RULE1}\n\n"
    for f, v in (updates or []):
        s += ERR.format(f=f, v=v)
    s += "\nTASK\nThe user asks to cancel the order. Decide one word: cancel or deny.\nDecision:"
    return s


def multi_edit(model, tok):
    # order_status evolves pending -> processed -> cancelled; correct (can we cancel?) = cancel iff pending
    print("\n=== (B) multi-edit accumulation (pending -> processed -> cancelled) ===")
    seqs = [
        ("to_processed", ["processed"], "deny"),
        ("to_cancelled", ["processed", "cancelled"], "deny"),
        ("back_to_pending", ["processed", "cancelled", "pending"], "cancel"),  # latest=pending -> cancel
    ]
    res = {}
    for name, chain, correct in seqs:
        ups = [("order_status", v) for v in chain]
        ks, kd = 0, 0
        for oid in OIDS:
            stacked = decide(model, tok, build_seq("pending", oid, updates=ups), a="cancel", b="deny")  # stale base + stacked erratums
            direct = decide(model, tok, build_seq(chain[-1], oid), a="cancel", b="deny")                  # single reprefill to final value
            ks += (stacked == correct); kd += (direct == correct)
        n = len(OIDS)
        res[name] = {"stacked_erratum": f"{ks}/{n}", "direct_oracle": f"{kd}/{n}", "correct": correct,
                     "chain": "pending->" + "->".join(chain)}
        print(f"  {name:16s} latest correct={correct:6s} | stacked_erratum={ks}/{n} direct_oracle={kd}/{n}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    out = {"model": args.model, "multi_field": multi_field(model, tok), "multi_edit": multi_edit(model, tok)}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"robustness_multi_{args.tag}.json"), "w"), indent=2)
    print("ROBUSTNESS_MULTI_DONE")


if __name__ == "__main__":
    main()
