"""EXP2 - Layer / timing emergence of the memoized conclusion.

"Pre-computed" is a temporal claim. Here we give it a layer axis: during the single
PREFILL forward pass, we track how decodable the CONCLUSION is from a DOWNSTREAM
AGGREGATOR token's residual stream at each layer. If the model "computes and writes
down" the conclusion, aggregator-token conclusion-decodability should rise from chance
to ~1.0 across mid layers *at prefill* -- before any decode step -- and be in place by
the layer the decision token commits.

We use the polarity 2x2 (mechd_common) so CONCLUSION is orthogonal to FIELD identity:
a probe predicting the conclusion is not just reading the copied field value.

Per layer we report, on a downstream aggregator delimiter:
  - conclusion_acc : linear-probe accuracy for SAFE/UNSAFE (group-CV, instance-disjoint)
  - field_acc      : linear-probe accuracy for the field value (present early as a copy)
And we compute:
  - write_depth : first layer where aggregator conclusion_acc >= 0.9
  - commit_depth: logit-lens layer where the DECISION token's safe-vs-unsafe margin
                  reaches its final sign and ~saturates
Claim: write_depth <= commit_depth (the note is ready by the time decode reads it).

Run: MECH_ATTN=sdpa python esys/mechd_timing.py --model Qwen/Qwen3-8B --tag qwen3_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mech_suite import load, ftok, TOK_WORDS
from mechd_common import POL, build_pol, conclusion_is_safe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def find_anchor(tok, ids_list, marker):
    """Last token index of the first occurrence of `marker` in the decoded prompt."""
    # decode incrementally is costly; instead find by re-tokenizing marker and substring match
    full = tok.decode(ids_list)
    ci = full.find(marker)
    if ci < 0:
        return None
    # map char position to token index by cumulative decode
    acc = ""
    for i, t in enumerate(ids_list):
        acc = tok.decode(ids_list[:i + 1])
        if len(acc) >= ci + len(marker):
            return i
    return len(ids_list) - 1


@torch.no_grad()
def collect(model, tok, scn, oid, fv, tv, layers, anchors_markers):
    t = build_pol(tok, scn, oid, fv, tv, False, False)
    ids = tok(t, add_special_tokens=False)["input_ids"]
    idt = torch.tensor([ids]).to("cuda")
    out = model(input_ids=idt, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    anchors = {name: find_anchor(tok, ids, mk) for name, mk in anchors_markers.items()}
    anchors["decision"] = len(ids) - 1
    feats = {}
    for name, pos in anchors.items():
        if pos is None:
            continue
        feats[name] = {li: hs[li][0, pos].float().cpu().numpy() for li in layers}
    # logit-lens on the decision token: project each layer through final norm + lm_head
    safe_id = ftok(tok, TOK_WORDS[POL[scn]["safe"]]); unsafe_id = ftok(tok, TOK_WORDS[POL[scn]["unsafe"]])
    norm = model.model.norm; head = model.lm_head
    margins = {}
    for li in layers:
        h = hs[li][0, -1]
        lg = head(norm(h)).float()
        margins[li] = float(lg[safe_id] - lg[unsafe_id])
    return feats, margins, anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--oids", default="A4471,B8820,C1093,D5567,E2025,F7311")
    args = ap.parse_args()
    tok, model = load(args.model)
    oids = args.oids.split(",")
    nl = model.config.num_hidden_layers
    layers = list(range(0, nl + 1, max(1, nl // 24)))      # ~25 layer points incl. embeddings(0)
    # universal downstream-of-rule delimiters present in EVERY scenario prompt
    markers = {"lets_check": "before acting.", "task_hdr": "Next action:"}

    feat_store = {}     # anchor -> layer -> list of vectors
    lab_store = {}      # anchor -> {"concl":[], "field":[], "groups":[]}
    margin_rows = []
    gi = 0
    for scn in POL:
        vA, vB = POL[scn]["values"]
        for oid in oids:
            for fv in (vA, vB):
                for tv in (vA, vB):
                    feats, margins, anchors = collect(model, tok, scn, oid, fv, tv, layers, markers)
                    concl = 1 if conclusion_is_safe(fv, tv) else 0
                    field = 0 if fv == vA else 1
                    for name, perlayer in feats.items():
                        fa = feat_store.setdefault(name, {li: [] for li in layers})
                        la = lab_store.setdefault(name, {"concl": [], "field": [], "groups": []})
                        for li in layers:
                            fa[li].append(perlayer[li])
                        la["concl"].append(concl); la["field"].append(field); la["groups"].append(gi)
                    margin_rows.append({"margins": margins, "final_safe": concl})
            gi += 1

    def cv_acc(Xl, y, groups):
        groups = np.array(groups)
        gkf = GroupKFold(n_splits=min(5, len(set(groups.tolist()))))
        accs = []
        for tr, te in gkf.split(Xl, y, groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xl[tr], y[tr]); accs.append(clf.score(Xl[te], y[te]))
        return float(np.mean(accs))

    # per-anchor layer curves
    curves = {}
    for name, fa in feat_store.items():
        la = lab_store[name]
        yc = np.array(la["concl"]); yf = np.array(la["field"]); gp = la["groups"]
        curves[name] = {}
        for li in layers:
            Xl = np.stack(fa[li])
            curves[name][li] = {"concl_acc": round(cv_acc(Xl, yc, gp), 3),
                                "field_acc": round(cv_acc(Xl, yf, gp), 3),
                                "depth": round(li / nl, 2)}

    # write_depth per aggregator anchor (exclude the decision token itself)
    write_depth = {}
    for name in curves:
        if name == "decision":
            continue
        crossed = [li for li in layers if curves[name][li]["concl_acc"] >= 0.9]
        write_depth[name] = {"layer": (crossed[0] if crossed else None),
                             "depth": (round(crossed[0] / nl, 2) if crossed else None)}

    # commit_depth: logit-lens on decision token. For each instance find first layer whose
    # margin sign matches the final sign AND stays matched through the top; report mean depth.
    commit_layers = []
    for row in margin_rows:
        m = row["margins"]; fs = row["final_safe"]
        sign_final = 1 if fs == 1 else -1
        ll = sorted(layers)
        commit = None
        for k, li in enumerate(ll):
            if np.sign(m[li]) == sign_final and all(np.sign(m[lj]) == sign_final for lj in ll[k:]):
                commit = li; break
        if commit is not None:
            commit_layers.append(commit)
    commit_depth = {"layer_mean": round(float(np.mean(commit_layers)), 1),
                    "depth_mean": round(float(np.mean(commit_layers)) / nl, 2),
                    "n": len(commit_layers)}

    n_samples = len(lab_store["decision"]["concl"])
    out = {"model": args.model, "nlayers": nl, "n_samples": n_samples,
           "curves": {k: {str(li): v for li, v in c.items()} for k, c in curves.items()},
           "write_depth": write_depth, "commit_depth": commit_depth}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mechd_timing_{args.tag}.json"), "w"), indent=2)

    print("\n==== EXP2 TIMING / LAYER EMERGENCE ====")
    print(f"nlayers={nl}  n_samples={n_samples}")
    for name in curves:
        row = " ".join(f"{curves[name][li]['depth']:.2f}:{curves[name][li]['concl_acc']:.2f}"
                       for li in layers if int(round(curves[name][li]['depth'] * 10)) % 2 == 0)
        print(f"  [{name:10s}] conclusion_acc by depth: {row}")
    print("WRITE depth (aggregator conclusion_acc first >=0.9):")
    for name, wd in write_depth.items():
        print(f"   {name:10s}: layer {wd['layer']} (depth {wd['depth']})")
    print(f"COMMIT depth (decision-token logit-lens final sign): layer {commit_depth['layer_mean']} "
          f"(depth {commit_depth['depth_mean']}), n={commit_depth['n']}")
    # the headline contrast
    agg_writes = [wd['layer'] for wd in write_depth.values() if wd['layer'] is not None]
    if agg_writes:
        print(f"HEADLINE: earliest aggregator write_depth={min(agg_writes)} vs commit_depth="
              f"{commit_depth['layer_mean']}  -> note ready {'BEFORE/BY' if min(agg_writes) <= commit_depth['layer_mean'] else 'AFTER'} decode reads it")
    print("MECHD_TIMING_DONE", flush=True)


if __name__ == "__main__":
    main()
