"""EXP1 - Conclusion vs. content dissociation (flagship causal test).

Skeptic's null: the downstream "notes" merely re-encode the field's *content*, and
the real reasoning happens at decode. We refute it by dissociating field content
from the derived conclusion using a polarity-parameterized rule (mechd_common):
hold the FIELD value fixed, flip the rule's `trigger`, which inverts the conclusion
(SAFE<->UNSAFE) while every field token is byte-identical.

  base   = (field=v, trigger=v)   -> conclusion SAFE
  source = (field=v, trigger=v')  -> conclusion UNSAFE   (field identical to base!)

The two prompts differ in exactly one contiguous span: the trigger word inside the
rule. We then patch base's cache with source's KV and read the decision:

  (A) TRIGGER-ONLY: patch only the trigger-word span  -> recovery ~ 0 expected
      (parallels field-only: the conclusion is not stored on the rule token)
  (B) NOTES: patch positions strictly AFTER the trigger span -> recovery ~ 1 expected
      The field token is held at base, the rule text is held at base; the ONLY thing
      changing is the memoized downstream content. If the decision follows source,
      the note carries the *derived conclusion*, not field content (field is constant).

We also fit a linear probe (single layer) on a downstream delimiter token to predict
the CONCLUSION vs the FIELD identity, across the orthogonal 2x2. If notes are
conclusions, conclusion-decodability >> field-decodability.

Run: MECH_ATTN=sdpa python esys/mechd_xcond.py --model Qwen/Qwen3-8B --tag qwen3_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mech_suite import load, clone, prefill, ftok, step, TOK_WORDS
from align import align_pair
from mechd_common import POL, build_pol, conclusion_is_safe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0: return (0.0, 0.0)
    if n == 1: return (round(xs[0], 3), round(xs[0], 3))
    means = []
    for bsi in range(B):
        s = sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n))
        means.append(s / n)
    means.sort()
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


@torch.no_grad()
def score(model, cache, last, dpos, toi):
    out = step(model, clone(cache, dpos), last, dpos)
    lg = out.logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


@torch.no_grad()
def patched_score(model, co, cn, positions, last, dpos, toi):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    out = step(model, w, last, dpos)
    lg = out.logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


def run_pair(model, tok, scn, oid, field_value, other_value):
    """base: trigger==field_value (SAFE). source: trigger==other_value (UNSAFE)."""
    s = POL[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[s["safe"]]), "unsafe": ftok(tok, TOK_WORDS[s["unsafe"]])}
    t_base = build_pol(tok, scn, oid, field_value, field_value, False, True)    # SAFE
    t_src = build_pol(tok, scn, oid, field_value, other_value, False, True)     # UNSAFE
    al = align_pair(tok, t_base, t_src)
    base_ids, src_ids = al["old_ids"], al["new_ids"]
    a, b = al["field_span"]                 # the trigger-word span (inside the rule)
    L = base_ids.shape[1]; dpos = L - 1
    last = int(src_ids[0, dpos])
    co = prefill(model, base_ids); cn = prefill(model, src_ids)

    s_base = score(model, co, last, dpos, toi)   # >0 (SAFE)
    s_src = score(model, cn, last, dpos, toi)    # <0 (UNSAFE)
    denom = s_src - s_base
    if abs(denom) < 0.5:                          # require a real, separated flip
        return None
    def rec(positions):
        return (patched_score(model, co, cn, positions, last, dpos, toi) - s_base) / denom

    trigger_only = rec(list(range(a, b)))                 # (A) patch rule trigger token only
    notes = rec(list(range(b, dpos)))                     # (B) patch post-trigger downstream notes
    full_down = rec(list(range(a, dpos)))                 # sanity (trigger + notes)
    return {"scn": scn, "oid": oid, "field": field_value, "L": L,
            "s_base": round(s_base, 3), "s_src": round(s_src, 3),
            "trigger_only_recovery": round(trigger_only, 3),
            "notes_recovery": round(notes, 3),
            "full_downstream_recovery": round(full_down, 3)}


@torch.no_grad()
def probe_features(model, tok, scn, oid, field_value, trigger_value, layers):
    """Hidden state at the decision-anchor (last pre-suffix delimiter) for each layer."""
    t = build_pol(tok, scn, oid, field_value, trigger_value, False, False)   # no forced suffix
    ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]]).to("cuda")
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states            # tuple (nlayers+1) of [1, L, d]
    anchor = ids.shape[1] - 1         # last token (the decision delimiter ':')
    return {li: hs[li][0, anchor].float().cpu().numpy() for li in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--oids", default="A4471,B8820,C1093,D5567,E2025,F7311")
    args = ap.parse_args()
    tok, model = load(args.model)
    oids = args.oids.split(",")

    # ---------- (1) cross-condition causal patching ----------
    recs = []
    for scn in POL:
        vA, vB = POL[scn]["values"]
        for oid in oids:
            for fv, ov in [(vA, vB), (vB, vA)]:
                r = run_pair(model, tok, scn, oid, fv, ov)
                if r is None:
                    continue
                recs.append(r)
                print(f"  [{scn}/{oid} field={fv:>14}] trigger_only={r['trigger_only_recovery']:+.3f} "
                      f"NOTES={r['notes_recovery']:+.3f} full_down={r['full_downstream_recovery']:+.3f}",
                      flush=True)
    trig = [r["trigger_only_recovery"] for r in recs]
    note = [r["notes_recovery"] for r in recs]
    fdn = [r["full_downstream_recovery"] for r in recs]
    patch_agg = {
        "n": len(recs),
        "trigger_only_recovery": {"mean": round(np.mean(trig), 3), "ci": boot_ci(trig)},
        "notes_recovery": {"mean": round(np.mean(note), 3), "ci": boot_ci(note)},
        "full_downstream_recovery": {"mean": round(np.mean(fdn), 3), "ci": boot_ci(fdn)},
    }

    # ---------- (2) probe-target dissociation (single representative layer) ----------
    nl = model.config.num_hidden_layers
    layers = sorted(set([int(nl * f) for f in (0.4, 0.5, 0.6, 0.7, 0.8)]))
    X = {li: [] for li in layers}
    y_concl, y_field, groups = [], [], []
    gi = 0
    for scn in POL:
        vA, vB = POL[scn]["values"]
        for oid in oids:
            for fv in (vA, vB):
                for tv in (vA, vB):          # full 2x2: field x trigger
                    feats = probe_features(model, tok, scn, oid, fv, tv, layers)
                    for li in layers:
                        X[li].append(feats[li])
                    y_concl.append(1 if conclusion_is_safe(fv, tv) else 0)
                    y_field.append(0 if fv == vA else 1)
                    groups.append(gi)        # group by (scn,oid) so train/test split is instance-disjoint
            gi += 1
    y_concl = np.array(y_concl); y_field = np.array(y_field); groups = np.array(groups)

    def cv_acc(Xl, y):
        gkf = GroupKFold(n_splits=min(5, len(set(groups))))
        accs = []
        for tr, te in gkf.split(Xl, y, groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xl[tr], y[tr])
            accs.append(clf.score(Xl[te], y[te]))
        return float(np.mean(accs))

    probe_agg = {}
    for li in layers:
        Xl = np.stack(X[li])
        probe_agg[li] = {"concl_acc": round(cv_acc(Xl, y_concl), 3),
                         "field_acc": round(cv_acc(Xl, y_field), 3),
                         "depth": round(li / nl, 2)}

    out = {"model": args.model, "patch": patch_agg, "patch_instances": recs,
           "probe": probe_agg, "n_probe_samples": len(y_concl)}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mechd_xcond_{args.tag}.json"), "w"), indent=2)

    print("\n==== EXP1 CROSS-CONDITION DISSOCIATION ====")
    print(f"n_patch_pairs={patch_agg['n']}")
    print(f"  TRIGGER-ONLY recovery (rule token; expect ~0): {patch_agg['trigger_only_recovery']['mean']} "
          f"CI{patch_agg['trigger_only_recovery']['ci']}")
    print(f"  NOTES recovery (downstream; expect ~1):        {patch_agg['notes_recovery']['mean']} "
          f"CI{patch_agg['notes_recovery']['ci']}")
    print(f"  FULL-DOWNSTREAM (sanity ~1):                   {patch_agg['full_downstream_recovery']['mean']} "
          f"CI{patch_agg['full_downstream_recovery']['ci']}")
    print(f"PROBE (downstream delimiter; conclusion vs field identity), n={len(y_concl)}:")
    for li in layers:
        p = probe_agg[li]
        print(f"   layer {li:>3} (depth {p['depth']}): conclusion_acc={p['concl_acc']}  field_acc={p['field_acc']}")
    print("MECHD_XCOND_DONE", flush=True)


if __name__ == "__main__":
    main()
