"""EXP3 - Does the mechanism generalize off the synthetic policy template?

The paper's four probes mostly live on one construction (POLICY + mutable field ->
cancel/deny). Here we re-run the two decisive probes -- FIELD-ONLY recovery (~0) and
FULL-DOWNSTREAM recovery (~1), plus the suffix curve -- on three qualitatively
different task families with a single-span field flip and a binary answer:

  (a) multihop : 2-hop lookup (key -> vault -> datacenter). The answer (datacenter) is
                 not the field (vault); it is derived downstream by matching the
                 directory. Tests memoization on genuine multi-hop reasoning.
  (b) rag_lookup: a directory/table; the question asks for an attribute of the entity
                 named in a mutable field (name -> desk number). Conclusion = a looked-up
                 fact, matched downstream of the field.
  (c) natural  : the SAME gated decision in free-form conversational prose (no POLICY/
                 SESSION scaffolding) -> rules out a template artifact.

If field-only ~ 0 and full-downstream ~ 1 hold here too, the "notes" account is not
specific to the synthetic format.

Run: MECH_ATTN=sdpa python esys/mechd_general.py --model Qwen/Qwen3-8B --tag qwen3_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mech_suite import load, clone, prefill, ftok, step
from align import align_pair


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0: return (0.0, 0.0)
    if n == 1: return (round(xs[0], 3), round(xs[0], 3))
    means = [sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n)) / n for bsi in range(B)]
    means.sort()
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


# ---------- task families: each returns (text_old, text_new, ans_old, ans_new) ----------
_FILLER = ("Operational notes: keep responses concise; cite the relevant record; do not "
           "speculate beyond the provided data; always answer with a single token.\n")


def _ct(tok, body):
    try:
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True)


def t_multihop(tok, oid, swap):
    vault = "bravo" if swap else "alpha"          # single-span field
    body = (f"SECURITY RECORD\nProject {oid}'s signing key is stored in vault {vault}.\n\n"
            "VAULT DIRECTORY (authoritative)\n"
            "Vault alpha is hosted in the Frankfurt datacenter.\n"
            "Vault bravo is hosted in the Tokyo datacenter.\n" + _FILLER +
            f"\nQUESTION\nIn which datacenter is Project {oid}'s signing key physically held? "
            "Reply with exactly one word.\nAnswer:")
    return _ct(tok, body), "Frankfurt", "Tokyo"


def t_rag_lookup(tok, oid, swap):
    name = "Bob" if swap else "Alice"             # single-span field
    body = ("EMPLOYEE DIRECTORY (authoritative)\n"
            "- Alice: role engineer, desk number 12\n"
            "- Bob: role engineer, desk number 7\n"
            "- Carol: role manager, desk number 5\n" + _FILLER +
            f"\nSESSION\nThe on-call engineer for ticket {oid} today is {name}.\n\n"
            "QUESTION\nWhat is the desk number of today's on-call engineer? Reply with the number only.\nAnswer:")
    return _ct(tok, body), "12", "7"


def t_natural(tok, oid, swap):
    role = "suspended" if swap else "admin"       # single-span field
    body = (f"Hey, quick question about order {oid}. My account is currently set to {role}. "
            "I remember the rule you mentioned: if the account is admin you can process a "
            "refund directly, but if the account is suspended you must not refund and have to "
            "hand it to the trust team instead. Given that, can you refund my last order? "
            "Please just tell me the single action you would take, as one word: either "
            "'refund' or 'escalate'.\nAction:")
    return _ct(tok, body), "refund", "escalate"


FAMILIES = {"multihop": t_multihop, "rag_lookup": t_rag_lookup, "natural": t_natural}


@torch.no_grad()
def score(model, cache, last, dpos, oi, ni):
    out = step(model, clone(cache, dpos), last, dpos)
    lg = out.logits[0, -1].float()
    return float(lg[ni] - lg[oi])         # new-answer minus old-answer margin


@torch.no_grad()
def patched(model, co, cn, positions, last, dpos, oi, ni):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    lg = step(model, w, last, dpos).logits[0, -1].float()
    return float(lg[ni] - lg[oi])


def run(model, tok, fam, builder, oid):
    t_old, ans_o, ans_n = builder(tok, oid, False)
    t_new, _, _ = builder(tok, oid, True)
    oi = ftok(tok, ans_o); ni = ftok(tok, ans_n)
    al = align_pair(tok, t_old, t_new)
    oid_ids, nid_ids = al["old_ids"], al["new_ids"]
    a, b = al["field_span"]; L = oid_ids.shape[1]; dpos = L - 1
    last = int(nid_ids[0, dpos])
    co = prefill(model, oid_ids); cn = prefill(model, nid_ids)
    s_old = score(model, co, last, dpos, oi, ni)
    s_new = score(model, cn, last, dpos, oi, ni)
    denom = s_new - s_old
    if abs(denom) < 0.5:
        return None
    def rec(positions):
        return (patched(model, co, cn, positions, last, dpos, oi, ni) - s_old) / denom
    field_only = rec(list(range(a, b)))
    full_down = rec(list(range(a, dpos)))
    ndown = dpos - a
    suffix = {}
    for fr in (0.05, 0.1, 0.25, 0.5, 1.0):
        k = max(1, int(round(fr * ndown)))
        suffix[fr] = rec(list(range(dpos - k, dpos)))
    return {"family": fam, "oid": oid, "L": L, "s_old": round(s_old, 2), "s_new": round(s_new, 2),
            "field_only_recovery": round(field_only, 3), "full_downstream_recovery": round(full_down, 3),
            "suffix": {fr: round(v, 3) for fr, v in suffix.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--oids", default="A4471,B8820,C1093,D5567,E2025,F7311")
    args = ap.parse_args()
    tok, model = load(args.model)
    oids = args.oids.split(",")
    recs = []
    for fam, builder in FAMILIES.items():
        for oid in oids:
            r = run(model, tok, fam, builder, oid)
            if r is None:
                print(f"  [{fam}/{oid}] non-separating, skipped", flush=True); continue
            recs.append(r)
            print(f"  [{fam:10s}/{oid}] field_only={r['field_only_recovery']:+.3f} "
                  f"full_down={r['full_downstream_recovery']:+.3f} "
                  f"suffix@0.1={r['suffix'][0.1]:+.2f} suffix@0.5={r['suffix'][0.5]:+.2f}", flush=True)

    agg = {}
    for fam in FAMILIES:
        fr = [r for r in recs if r["family"] == fam]
        if not fr:
            continue
        fo = [r["field_only_recovery"] for r in fr]; fd = [r["full_downstream_recovery"] for r in fr]
        agg[fam] = {"n": len(fr),
                    "field_only_recovery": {"mean": round(np.mean(fo), 3), "ci": boot_ci(fo)},
                    "full_downstream_recovery": {"mean": round(np.mean(fd), 3), "ci": boot_ci(fd)},
                    "suffix_mean": {fr_: round(np.mean([r["suffix"][fr_] for r in fr]), 3)
                                    for fr_ in (0.05, 0.1, 0.25, 0.5, 1.0)}}
    out = {"model": args.model, "agg": agg, "instances": recs}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mechd_general_{args.tag}.json"), "w"), indent=2)
    print("\n==== EXP3 GENERALIZATION OFF SYNTHETIC TEMPLATE ====")
    for fam, a in agg.items():
        print(f"  [{fam:10s}] n={a['n']:2d}  field_only={a['field_only_recovery']['mean']:+.3f} "
              f"CI{a['field_only_recovery']['ci']}   full_down={a['full_downstream_recovery']['mean']:+.3f} "
              f"CI{a['full_downstream_recovery']['ci']}   suffix={a['suffix_mean']}")
    print("MECHD_GENERAL_DONE", flush=True)


if __name__ == "__main__":
    main()
