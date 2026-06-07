"""H5 (rigorous) — Unified edit+compose multi-turn agent across >=10 domains, >=100 trajectories, CIs.

Each domain: a long precompiled POLICY + a 3-turn trajectory where a state FIELD is edited (appended
erratum) each turn and the governed tool decision flips. UNIFIED = compose policy once + longest-prefix
incremental reuse + appended edits; FULL = reprefill-every-turn. We measure per-turn decision agreement
(unified==full), correctness, and per-trajectory cumulative-TTFT speedup, over 10 domains x N instances,
with bootstrap 95% CIs. Run: python esys/editkv_agent_rigorous.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""
import argparse, os, sys, json, time, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer
from composable_kv import (load_lm, cache_slice, cache_concat, precompute_chunk, prefill,
                           repositioned_chunk_cache, forward_suffix)

PAD = "\n".join(f"- SOP clause {i}: log the interaction, verify identity, and follow standard procedure." for i in range(220))

# 10 domains. Each: (name, field, rule, [(value, request, act)]) — 3 turns; the decision flips as the field changes.
DOMAINS = [
 ("ORDER", "order_status", "cancel an order ONLY if order_status is pending; for shipped/delivered deny cancellation.",
  [("pending", "cancel order {x}", "cancel"), ("shipped", "cancel order {x} again", "deny"), ("delivered", "cancel order {x} once more", "deny")]),
 ("BANK", "account_status", "permit a withdrawal ONLY if account_status is active; if frozen, deny.",
  [("active", "withdraw from account {x}", "allow"), ("frozen", "withdraw from account {x} again", "deny"), ("active", "withdraw from account {x} now", "allow")]),
 ("ACCESS", "clearance", "grant a CONFIDENTIAL record ONLY if clearance is high; if low, deny.",
  [("high", "read record {x}", "grant"), ("low", "read record {x} again", "deny"), ("high", "read record {x} now", "grant")]),
 ("RX", "allergy_flag", "dispense a drug ONLY if allergy_flag is clear; if flagged, hold for review.",
  [("clear", "dispense drug {x}", "dispense"), ("flagged", "dispense drug {x} again", "hold"), ("clear", "dispense drug {x} now", "dispense")]),
 ("DEPLOY", "ticket", "allow a production deploy ONLY if ticket is approved; if none, block.",
  [("approved", "deploy build {x}", "allow"), ("none", "deploy build {x} again", "block"), ("approved", "deploy build {x} now", "allow")]),
 ("MOD", "post_state", "publish a post ONLY if post_state is approved; if flagged, quarantine.",
  [("approved", "publish post {x}", "publish"), ("flagged", "publish post {x} again", "quarantine"), ("approved", "publish post {x} now", "publish")]),
 ("VISA", "passport", "grant entry ONLY if passport is valid; if expired, refer to inspection.",
  [("valid", "admit traveler {x}", "grant"), ("expired", "admit traveler {x} again", "refer"), ("valid", "admit traveler {x} now", "grant")]),
 ("AUTH", "mfa", "allow login ONLY if mfa is passed; if failed, challenge.",
  [("passed", "log in user {x}", "allow"), ("failed", "log in user {x} again", "challenge"), ("passed", "log in user {x} now", "allow")]),
 ("PAYOUT", "kyc", "release a payout ONLY if kyc is verified; if pending, hold.",
  [("verified", "pay out request {x}", "release"), ("pending", "pay out request {x} again", "hold"), ("verified", "pay out request {x} now", "release")]),
 ("SHIP", "stock", "ship an order ONLY if stock is available; if backordered, defer.",
  [("available", "ship order {x}", "ship"), ("backordered", "ship order {x} again", "defer"), ("available", "ship order {x} now", "ship")]),
]
ACT_OTHER = {"cancel": "deny", "deny": "cancel", "allow": "deny", "grant": "deny", "dispense": "hold",
             "hold": "dispense", "block": "allow", "publish": "quarantine", "quarantine": "publish",
             "refer": "grant", "challenge": "allow", "release": "hold", "defer": "ship", "ship": "defer"}


def policy_text(name, field, rule):
    return f"# SKILL: {name}_POLICY\nRULE: {rule}\nThe relevant state field is {field}.\n{PAD}\nEnd of {name}_POLICY."


def tid(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def chat(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def boot_ci(xs, B=10000, seed=0):
    if not xs:
        return [0.0, 0.0]
    r = random.Random(seed); m = sorted(sum(r.choice(xs) for _ in range(len(xs))) / len(xs) for _ in range(B))
    return [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


@torch.no_grad()
def dlogits(model, cache, last, pos):
    return model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
                 cache_position=torch.tensor([pos], device="cuda"), use_cache=True).logits[0, -1].float()


def lcp(x, y):
    n = min(x.shape[1], y.shape[1]); i = 0
    while i < n and int(x[0, i]) == int(y[0, i]):
        i += 1
    return i


@torch.no_grad()
def run_trajectory(model, tok, dom, xid):
    name, field, rule, turns = dom
    policy = policy_text(name, field, rule)
    sysmsg = f"You are the {name} agent. Apply {name}_POLICY."
    # build per-turn full chat token sequences (convo grows with taken actions + state edits)
    fids = []; convo = f"{sysmsg}\n\n{policy}"; meta = []
    for ti, (val, req, act) in enumerate(turns):
        edit = (f"[STATE UPDATE] {field} is now {val}; overrides any earlier value AND conclusion."
                if ti > 0 else f"Current {field} is {val}.")
        oth = ACT_OTHER[act]
        convo += f"\n\n{edit}\n{req.format(x=xid)} Answer one word — {act} or {oth}.\nDecision:"
        fids.append(tok(chat(tok, convo), add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda"))
        meta.append((act, oth)); convo += " " + act
    # policy span in turn 0
    f0 = chat(tok, f"{sysmsg}\n\n{policy}\n\n" + "x")
    enc = tok(f0, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
    pa = f0.find(policy); pb = pa + len(policy)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= pa < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= pb), len(offs))

    # UNIFIED
    uni = []; t_uni = 0.0; cached = None; cache = None
    chunk = precompute_chunk(model, fids[0][:, a:b])
    for ti, ids in enumerate(fids):
        L = ids.shape[1]; act, oth = meta[ti]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if ti == 0:
            cache = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, chunk, b - a, a))
            if b < L - 1:
                cache = forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values
        else:
            k = lcp(cached, ids); cache = cache_slice(cache, 0, k)
            if k < L - 1:
                cache = forward_suffix(model, cache, ids[:, k:L - 1], k).past_key_values
        lg = dlogits(model, cache, ids[0, L - 1], L - 1)
        torch.cuda.synchronize(); t_uni += (time.perf_counter() - t0) * 1000
        cache = forward_suffix(model, cache, ids[:, L - 1:L], L - 1).past_key_values
        ai = tok(" " + act, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        cache = forward_suffix(model, cache, ai, L).past_key_values
        cached = torch.cat([ids, ai], 1)
        uni.append(("correct" if lg[tid(tok, act)] >= lg[tid(tok, oth)] else "other"))
    # FULL
    full = []; t_full = 0.0
    for ti, ids in enumerate(fids):
        L = ids.shape[1]; act, oth = meta[ti]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fc = prefill(model, ids[:, :L - 1]); lg = dlogits(model, fc, ids[0, L - 1], L - 1)
        torch.cuda.synchronize(); t_full += (time.perf_counter() - t0) * 1000
        full.append(("correct" if lg[tid(tok, act)] >= lg[tid(tok, oth)] else "other"))
    return uni, full, t_uni, t_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--per", type=int, default=10)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    agree, uni_c, full_c, speed = [], [], [], []
    ntraj = 0
    for dom in DOMAINS:
        for j in range(args.per):
            xid = f"{dom[0][:2]}{1000 + j*7}"
            u, f, tu, tf = run_trajectory(model, tok, dom, xid)
            for a_, b_ in zip(u, f):
                agree.append(int(a_ == b_));
            uni_c += [int(x == "correct") for x in u]; full_c += [int(x == "correct") for x in f]
            speed.append(tf / tu); ntraj += 1
        print(f"  {dom[0]:7s} done | running agree~{sum(agree)/len(agree):.3f} speedup~{sum(speed)/len(speed):.2f}x", flush=True)
    out = {"model": args.model, "domains": len(DOMAINS), "trajectories": ntraj, "decisions": len(agree),
           "agreement": round(sum(agree) / len(agree), 3), "agreement_ci": boot_ci(agree),
           "unified_correct": round(sum(uni_c) / len(uni_c), 3), "full_correct": round(sum(full_c) / len(full_c), 3),
           "mean_speedup": round(sum(speed) / len(speed), 2), "speedup_ci": boot_ci([round(s, 2) for s in speed])}
    print(f"\n=== UNIFIED AGENT (rigorous) {args.model}: {len(DOMAINS)} domains x {args.per} = {ntraj} trajectories, {len(agree)} decisions ===")
    print(f"  unified==full agreement = {out['agreement']} CI{out['agreement_ci']}")
    print(f"  unified_correct={out['unified_correct']} full_correct={out['full_correct']}")
    print(f"  cumulative-TTFT speedup = {out['mean_speedup']}x CI{out['speedup_ci']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"agent_rigorous_{tag}.json"), "w"), indent=2)
    print("AGENT_RIGOROUS_DONE")


if __name__ == "__main__":
    main()
