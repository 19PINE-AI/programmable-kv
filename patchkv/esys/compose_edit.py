"""KEYSTONE: compose-then-edit on ONE cache. Unifies the editable and composable axes.

We precompile a SKILL (long policy) whose governing THRESHOLD is an editable field, splice it in
(RoPE-repositioned), then surgically EDIT that threshold *inside the transplanted chunk* and ask
whether the governed decision updates. The editable mechanism should carry over verbatim:
  in_place (patch only the threshold KV)        -> fails (downstream skill+task memoized old threshold)
  field+selective@K (threshold + top-K dec-attn)-> recovers
  erratum (append [STATE UPDATE] after skill)   -> recovers
We report the D1-style recovery ratio for the COMPOSED skill and compare it to the NON-composed
(fully recomputed) skill — if they match, editing works identically on transplanted KV => one substrate.
Run: MECH_ATTN=sdpa python esys/compose_edit.py --model Qwen/Qwen3-1.7B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import (load_lm, prefill, cos_sin, rotate_half, cache_slice, cache_concat,
                           precompute_chunk, repositioned_chunk_cache, forward_suffix)

FILLER = "\n".join(f"- Standard guideline {i+1}: log the interaction and follow SOP for routine matters." for i in range(18))

# Categorical state-gated skills (like the editable benchmark) that flip cleanly when the embedded
# state field is edited (Told -> Tnew). The editable FIELD lives INSIDE the precompiled skill.
INSTANCES = [
 dict(name="account", sys="You are a banking agent.", skill_name="ACCOUNT_POLICY", field="account_status",
      rule="Permit a withdrawal ONLY if account_status is active. If the account_status is frozen, you MUST deny.",
      Told="active", Tnew="frozen", task="The user requests a withdrawal. Answer one word — allow or deny.\nDecision:",
      act_old="allow", act_new="deny"),
 dict(name="order", sys="You are an order agent.", skill_name="ORDER_POLICY", field="order_status",
      rule="Permit cancellation ONLY if order_status is pending. For a shipped order you MUST deny cancellation.",
      Told="pending", Tnew="shipped", task="The user asks to cancel the order. Answer one word — cancel or deny.\nDecision:",
      act_old="cancel", act_new="deny"),
 dict(name="subscription", sys="You are a SaaS assistant.", skill_name="PLAN_POLICY", field="plan",
      rule="Enable premium features ONLY if plan is pro. If the plan is free, you MUST block premium features.",
      Told="pro", Tnew="free", task="The user requests a premium feature. Answer one word — enable or block.\nDecision:",
      act_old="enable", act_new="block"),
 dict(name="door", sys="You are a building access controller.", skill_name="DOOR_POLICY", field="badge",
      rule="Unlock the secure door ONLY if badge is valid. If the badge is revoked, keep the door locked.",
      Told="valid", Tnew="revoked", task="The user swipes to enter. Answer one word — unlock or locked.\nDecision:",
      act_old="unlock", act_new="locked"),
 dict(name="payout", sys="You are a payments agent.", skill_name="PAYOUT_POLICY", field="kyc_status",
      rule="Release a payout ONLY if kyc_status is verified. If kyc_status is pending, you MUST hold the payout.",
      Told="verified", Tnew="pending", task="The user requests a payout. Answer one word — release or hold.\nDecision:",
      act_old="release", act_new="hold"),
 dict(name="post", sys="You are a content moderator.", skill_name="MOD_POLICY", field="post_state",
      rule="Publish a post ONLY if post_state is approved. If post_state is flagged, you MUST quarantine it.",
      Told="approved", Tnew="flagged", task="A post is submitted. Answer one word — publish or quarantine.\nDecision:",
      act_old="publish", act_new="quarantine"),
 dict(name="ship", sys="You are a logistics agent.", skill_name="SHIP_POLICY", field="stock_state",
      rule="Ship an order ONLY if stock_state is in_stock. If stock_state is backordered, you MUST defer shipping.",
      Told="in_stock", Tnew="backordered", task="An order is ready to ship. Answer one word — ship or defer.\nDecision:",
      act_old="ship", act_new="defer"),
 dict(name="login", sys="You are an auth gateway.", skill_name="AUTH_POLICY", field="mfa_state",
      rule="Allow login ONLY if mfa_state is passed. If mfa_state is failed, you MUST challenge the user.",
      Told="passed", Tnew="failed", task="The user attempts to log in. Answer one word — allow or challenge.\nDecision:",
      act_old="allow", act_new="challenge"),
]


def skill_text(inst, val):
    return (f"# SKILL: {inst['skill_name']}\nCURRENT STATE: {inst['field']} = {val}.\nRULE: {inst['rule']}\n"
            f"{FILLER}\nEnd of {inst['skill_name']}.")


def assemble(tok, inst, val, update=False):
    skill = skill_text(inst, val)
    upd = (f"\n[STATE UPDATE] {inst['field']} has changed to {inst['Tnew']}; this overrides any earlier "
           f"value AND any earlier conclusion.\n") if update else ""
    body = f"{inst['sys']}\n\n{skill}\n\n" + upd + inst["task"]
    full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    s_char = full.find(skill); e_char = s_char + len(skill)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= s_char < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= e_char), len(offs))
    return ids, a, b


@torch.no_grad()
def dscore(model, cache, last, pos, tg, td):
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    lg = o.logits[0, -1].float()
    return float(lg[tg] - lg[td])


@torch.no_grad()
def dattn_order(model, cache, last, pos, lo, hi):
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True, output_attentions=True)
    att = torch.stack([x[0, :, -1, :] for x in o.attentions]).mean(1).mean(0)
    return sorted(range(lo, hi), key=lambda i: float(att[i]), reverse=True)


def patch_positions(dst, src, positions):
    for i in range(len(dst.layers)):
        p = torch.tensor(positions, device="cuda")
        dst.layers[i].keys[:, :, p, :] = src.layers[i].keys[:, :, p, :]
        dst.layers[i].values[:, :, p, :] = src.layers[i].values[:, :, p, :]


@torch.no_grad()
def composed_cache(model, ids, a, b, skill_chunk):
    """[sys] prefill + repositioned precomputed skill + [task] forward -> full cache for [0..L-1)."""
    L = ids.shape[1]
    sysc = prefill(model, ids[:, :a])
    rep = repositioned_chunk_cache(model, skill_chunk, b - a, a)
    spliced = cache_concat(sysc, rep)
    out = forward_suffix(model, spliced, ids[:, b:L - 1], b)
    return out.past_key_values, L


@torch.no_grad()
def recomputed_cache(model, ids):
    L = ids.shape[1]
    return cache_slice(prefill(model, ids[:, :L - 1]), 0, L - 1), L


@torch.no_grad()
def one_instance(model, tok, inst, KS):
    tg = tok(inst["act_new"], add_special_tokens=False)["input_ids"][0]   # decision under NEW field
    td = tok(inst["act_old"], add_special_tokens=False)["input_ids"][0]   # decision under OLD field
    ids_A, a, b = assemble(tok, inst, inst["Told"])
    ids_B, _, _ = assemble(tok, inst, inst["Tnew"])
    if ids_A.shape[1] != ids_B.shape[1]:
        return None   # Told/Tnew tokenize to different lengths for this tokenizer; skip (keeps positions aligned)
    L = ids_A.shape[1]; dpos = L - 1; lastA = int(ids_A[0, dpos]); lastB = int(ids_B[0, dpos])
    al = align_pair(tok, skill_text(inst, inst["Told"]), skill_text(inst, inst["Tnew"]))
    fa, fb = al["field_span"]; thr = list(range(a + fa, a + fb))
    skill_A_chunk = precompute_chunk(model, ids_A[:, a:b])
    skill_B_chunk = precompute_chunk(model, ids_B[:, a:b])
    res = {}
    for mode in ["recomputed", "composed"]:
        if mode == "composed":
            cA, _ = composed_cache(model, ids_A, a, b, skill_A_chunk)
            cB, _ = composed_cache(model, ids_B, a, b, skill_B_chunk)
            B_full = cache_concat(prefill(model, ids_B[:, :a]), repositioned_chunk_cache(model, skill_B_chunk, b - a, a))
            srcB = forward_suffix(model, B_full, ids_B[:, b:L - 1], b).past_key_values
        else:
            cA, _ = recomputed_cache(model, ids_A); cB, _ = recomputed_cache(model, ids_B); srcB = cB
        sA = dscore(model, cA, lastA, dpos, tg, td); sB = dscore(model, cB, lastB, dpos, tg, td)
        denom = (sB - sA) or 1e-6
        order = dattn_order(model, cA, lastA, dpos, b, dpos)
        r = {"sA": round(sA, 2), "sB": round(sB, 2), "flip": (sA < 0) != (sB < 0)}
        w = cache_slice(cA, 0, dpos); patch_positions(w, srcB, thr)
        r["in_place"] = (dscore(model, w, lastB, dpos, tg, td) - sA) / denom
        for k in KS:
            w = cache_slice(cA, 0, dpos); patch_positions(w, srcB, thr + order[:k])
            r[f"sel@{k}"] = (dscore(model, w, lastB, dpos, tg, td) - sA) / denom
        ids_E, ae, be = assemble(tok, inst, inst["Told"], update=True); LE = ids_E.shape[1]
        if mode == "composed":
            cE = forward_suffix(model, cache_concat(prefill(model, ids_E[:, :ae]),
                 repositioned_chunk_cache(model, skill_A_chunk, be - ae, ae)), ids_E[:, be:LE - 1], be).past_key_values
        else:
            cE = cache_slice(prefill(model, ids_E[:, :LE - 1]), 0, LE - 1)
        r["erratum"] = (dscore(model, cE, int(ids_E[0, LE - 1]), LE - 1, tg, td) - sA) / denom
        res[mode] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="eager")
    KS = [8, 32]; METHODS = ["in_place", "sel@8", "sel@32", "erratum"]
    agg = {m: {meth: [] for meth in METHODS} for m in ["recomputed", "composed"]}
    flips = 0
    print(f"=== COMPOSE-THEN-EDIT keystone ({args.model}) ===")
    used = 0
    for inst in INSTANCES:
        r = one_instance(model, tok, inst, KS)
        if r is None:
            print(f"  {inst['name']:9s} skipped (token-length mismatch)", flush=True); continue
        used += 1
        flips += int(r["recomputed"]["flip"])
        for m in ["recomputed", "composed"]:
            for meth in METHODS:
                agg[m][meth].append(r[m][meth])
        print(f"  {inst['name']:9s} recomputed: in_place={r['recomputed']['in_place']:+.2f} sel@32={r['recomputed']['sel@32']:+.2f} "
              f"err={r['recomputed']['erratum']:+.2f} | composed: in_place={r['composed']['in_place']:+.2f} "
              f"sel@32={r['composed']['sel@32']:+.2f} err={r['composed']['erratum']:+.2f}", flush=True)
    N = max(used, 1)
    out = {"model": args.model, "n_instances": used, "clean_flips": flips, "agg": {}}
    print(f"\n  MEAN recovery (n={used}, clean flips={flips}/{used}):")
    print(f"  {'method':10s} {'recomputed':>12s} {'composed':>12s}")
    for meth in METHODS:
        rr = sum(agg["recomputed"][meth]) / N; cc = sum(agg["composed"][meth]) / N
        out["agg"][meth] = {"recomputed": round(rr, 3), "composed": round(cc, 3)}
        print(f"  {meth:10s} {rr:>12.3f} {cc:>12.3f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"compose_edit_{tag}.json"), "w"), indent=2)
    print("COMPOSE_EDIT_DONE")


if __name__ == "__main__":
    main()
