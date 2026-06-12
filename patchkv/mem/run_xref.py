"""Cross-referential split-memory test (closes the "untested" caveat in memory.tex E4).

E4 showed splitting *independent* facts across independently-precompiled blocks is decision-
lossless (they integrate at read time). The open question: do genuinely CROSS-REFERENTIAL facts
-- where fact B's meaning depends on fact A (a 2-hop chain) -- break when A and B land in
different blocks (so B's isolated precompute never attended to A)?

Controlled design. One memory contains two linked structures straddling the SAME block boundary:
  cross-ref:  A = "DEFINITION: the governing setting is `gate`."   B = the line giving gate's value
  independent: C, D = two named settings AND-ed by the decision (no chain)
Memory layout: [padL][A][C] | [B][D][padR]  -- the split boundary sits between (A,C) and (B,D),
so BOTH the cross-ref pair (A->B) and the independent pair (C,D) are separated by it. We then ask
two decisions on the SAME caches (only the appended query differs):
  cross-ref decision: proceed iff `gate` is enabled  (requires resolving A->B)
  independent decision: proceed iff C and D both enabled  (two direct lookups)
Conditions: full reprefill (oracle) | transplant split@boundary | transplant colocated (boundary in
padR, so A,B,C,D share one block). Prediction: split hurts the cross-ref decision but not the
independent one; colocated transplant matches full for both.
Run: python mem/run_xref.py --model Qwen/Qwen3-8B --n 60
"""
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "esys"))
from composable_kv import (load_lm, prefill, precompute_chunk, repositioned_chunk_cache,
                           cache_concat, cache_slice, forward_suffix)
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the USER MEMORY exactly."
PADCAT = [("notifications", "marketing_emails"), ("privacy", "data_sharing"), ("billing", "auto_renew"),
          ("content", "beta_features"), ("sharing", "public_profile"), ("automation", "auto_reply"),
          ("comms", "newsletter"), ("data", "cloud_backup"), ("integrations", "calendar_access"),
          ("security", "biometric_unlock"), ("accessibility", "high_contrast"), ("privacy", "personalized_ads")]
GATES = ["record_access", "export_clearance", "admin_override", "vault_unlock"]
INDS = [("security", "two_factor"), ("billing", "saved_cards"), ("privacy", "location_history"),
        ("data", "cross_device")]


def en(b):
    return "enabled" if b else "disabled"


def build_memory(rng, M, gate_val, c_val, d_val):
    gate = rng.choice(GATES)
    ic, ia = rng.choice(INDS); jc, ja = rng.choice([x for x in INDS if x[1] != ia])
    pads = [f"- [{c}] {a}: {en(rng.random() < 0.6)}" for (c, a) in
            [rng.choice(PADCAT) for _ in range(M)]]
    half = len(pads) // 2
    A = f"- DEFINITION: for THIS request the governing setting is `{gate}`."
    B = f"- [access] {gate}: {en(gate_val)}"
    C = f"- [{ic}] {ia}: {en(c_val)}"
    D = f"- [{jc}] {ja}: {en(d_val)}"
    lines = ["# USER MEMORY (account settings)\n"] + pads[:half] + [A, C] + [B, D] + pads[half:] + ["\n# END USER MEMORY\n"]
    mem = "\n".join(lines)
    return mem, gate, ia, ja, A, B, C, D


def find_span(offsets, text, sub, start=0):
    c = text.find(sub, start)
    assert c >= 0, f"marker not found: {sub[:30]}"
    lo = hi = None
    for ti, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if lo is None and e > c:
            lo = ti
        if s < c + len(sub):
            hi = ti + 1
    return lo, hi, c + len(sub)


@torch.no_grad()
def transplant_split(model, ids, mlo, mhi, bnd):
    """Two blocks [mlo,bnd) [bnd,mhi), each precomputed in isolation and repositioned."""
    cache = prefill(model, ids[:, :mlo])
    for lo, hi in [(mlo, bnd), (bnd, mhi)]:
        if hi <= lo:
            continue
        blk = precompute_chunk(model, ids[:, lo:hi])
        cache = cache_concat(cache, repositioned_chunk_cache(model, blk, hi - lo, lo))
    L = ids.shape[1]
    if mhi < L - 1:
        cache = forward_suffix(model, cache, ids[:, mhi:L - 1], mhi).past_key_values
    return cache


@torch.no_grad()
def dec_logit(model, cache, last_tok, pos, toi):
    out = model(input_ids=torch.tensor([[last_tok]], device="cuda"),
                past_key_values=cache_slice(cache, 0, pos), cache_position=torch.tensor([pos], device="cuda"))
    lg = out.logits[0, -1].float()
    return lg, ("yes" if lg[toi[0]] >= lg[toi[1]] else "no")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--M", type=int, default=28)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    yes_t = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_t = tok("no", add_special_tokens=False)["input_ids"][0]
    toi = (yes_t, no_t)

    XREF_Q = ("\n\nTASK: Decide whether to proceed with record access.\n"
              "RULE: Find the DEFINITION line, identify the governing setting it names, look up THAT "
              "setting's value in memory, and proceed ONLY IF it is enabled.\n"
              "Answer with exactly one word — yes or no.\nAnswer:")
    IND_Q_T = ("\n\nTASK: Decide whether to proceed.\n"
               "RULE: Proceed ONLY IF BOTH `{ia}` and `{ja}` are enabled. Look up each value in memory.\n"
               "Answer with exactly one word — yes or no.\nAnswer:")

    rows = []
    rng = random.Random(20260612)
    for k in range(args.n):
        # balanced gold across personas
        gate_val = (k % 2 == 0)
        c_val, d_val = rng.random() < 0.7, rng.random() < 0.7
        mem, gate, ia, ja, A, B, C, D = build_memory(rng, args.M, gate_val, c_val, d_val)
        traj = "\n".join(f"User: chat about topic {i}?\nAssistant: sure, briefly." for i in range(3))

        for kind, query in [("xref", XREF_Q), ("indep", IND_Q_T.format(ia=ia, ja=ja))]:
            body = f"{mem}\n\nCONVERSATION:\n{traj}\n{query}"
            full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                           add_generation_prompt=True)
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
            L = ids.shape[1]; last = int(ids[0, L - 1])
            # memory span
            mlo, _, _ = find_span(offs, full, "# USER MEMORY")
            _, mhi, _ = find_span(offs, full, "# END USER MEMORY")
            # split boundary: between the (A,C) group and (B,D) group  == start of B
            blo, _, _ = find_span(offs, full, B)
            bnd_split = blo
            # colocated boundary: inside padR, after D  -> A,B,C,D all in block1
            _, dhi, dend = find_span(offs, full, D)
            bnd_colo = min(mhi - 1, dhi + 2)
            gold = "yes" if (gate_val if kind == "xref" else (c_val and d_val)) else "no"

            fc = prefill(model, ids[:, :L - 1])
            lg_f, d_full = dec_logit(model, fc, last, L - 1, toi)
            cs = transplant_split(model, ids, mlo, mhi, bnd_split)
            lg_s, d_split = dec_logit(model, cs, last, L - 1, toi)
            cc = transplant_split(model, ids, mlo, mhi, bnd_colo)
            lg_c, d_colo = dec_logit(model, cc, last, L - 1, toi)
            rows.append(dict(k=k, kind=kind, gold=gold, d_full=d_full, d_split=d_split, d_colo=d_colo,
                             full_correct=int(d_full == gold),
                             split_agree=int(d_split == d_full), colo_agree=int(d_colo == d_full),
                             cos_split=float(F.cosine_similarity(lg_f, lg_s, 0)),
                             cos_colo=float(F.cosine_similarity(lg_f, lg_c, 0))))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.n}", flush=True)

    def agg(kind):
        rs = [r for r in rows if r["kind"] == kind]
        n = len(rs)
        return dict(n=n,
                    full_acc=round(sum(r["full_correct"] for r in rs) / n, 3),
                    split_agree=round(sum(r["split_agree"] for r in rs) / n, 3),
                    colo_agree=round(sum(r["colo_agree"] for r in rs) / n, 3),
                    cos_split=round(sum(r["cos_split"] for r in rs) / n, 4),
                    cos_colo=round(sum(r["cos_colo"] for r in rs) / n, 4))
    summary = {"model": args.model, "n": args.n, "M": args.M,
               "xref": agg("xref"), "indep": agg("indep")}
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(
        os.path.dirname(__file__), "results", f"xref_{tag}.jsonl"), "w"), indent=2)
    x, i = summary["xref"], summary["indep"]
    print(f"\n=== Cross-referential split-memory ({args.model}, n={args.n}) ===")
    print(f"{'decision':>10} | full_acc | split=full | colo=full | cos_split | cos_colo")
    for nm, a in [("xref", x), ("indep", i)]:
        print(f"{nm:>10} | {a['full_acc']:>8} | {a['split_agree']:>10} | {a['colo_agree']:>9} | "
              f"{a['cos_split']:>9} | {a['cos_colo']:>8}")
    print(f"\nSPLIT penalty (colo_agree - split_agree): xref {round(x['colo_agree']-x['split_agree'],3)} "
          f"vs independent {round(i['colo_agree']-i['split_agree'],3)}")
    print("XREF_DONE")


if __name__ == "__main__":
    main()
