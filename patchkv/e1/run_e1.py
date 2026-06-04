"""E1 driver: blast-radius characterization.

For each (model, field, flip-magnitude):
  1. align OLD/NEW token sequences (length-preserving)
  2. capture per-layer post-RoPE Q/K/V for OLD and NEW forwards
  3. compute per-token KV deviation and attention-output deviation
  4. summarize blast radius BR(tau) over DOWNSTREAM, non-field tokens, by field class
Outputs a JSON record per run plus raw per-token arrays (npz) for plotting.
"""
import argparse, json, os, sys, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import capture
from align import align_pair
from deviation import kv_deviation, attention_output_deviation
import contexts as C
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    capture.install(model)
    model.eval()
    return tok, model


@torch.no_grad()
def forward_capture(model, ids):
    capture.enable_capture(to_cpu=True)
    model(input_ids=ids.to("cuda"), use_cache=True)
    capture.disable_capture()
    return capture.get_capture()


def run_one(tok, model, field_key, magnitude, n_neutral_rules, tau_grid,
            attn_dev=True):
    f = C.FIELDS[field_key]
    old_text = C.build_context(field_key, f["old"], n_neutral_rules)
    new_text = C.build_context(field_key, f[magnitude], n_neutral_rules)
    al = align_pair(tok, old_text, new_text)
    s, e = al["field_span"]
    T = al["seq_len"]

    cap_old = forward_capture(model, al["old_ids"])
    cap_new = forward_capture(model, al["new_ids"])
    nlayers = len(cap_new)

    # per-layer per-token deviations -> stack to [L,T]
    k_cos = np.zeros((nlayers, T)); v_cos = np.zeros((nlayers, T))
    k_rl2 = np.zeros((nlayers, T)); v_rl2 = np.zeros((nlayers, T))
    attn = np.zeros((nlayers, T))
    for li in range(nlayers):
        co, cn = cap_old[li], cap_new[li]
        d = kv_deviation(co["k"][0], cn["k"][0], co["v"][0], cn["v"][0])
        k_cos[li] = d["k_cos"].numpy(); v_cos[li] = d["v_cos"].numpy()
        k_rl2[li] = d["k_rel_l2"].numpy(); v_rl2[li] = d["v_rel_l2"].numpy()
        if attn_dev:
            ad = attention_output_deviation(
                cn["q"][0], cn["k"][0], cn["v"][0], co["k"][0], co["v"][0],
                (s, e), cn["scaling"], device="cuda")
            attn[li] = ad.numpy()

    # masks: downstream non-field tokens are positions > e-1 ; causally-exact = < s
    pos = np.arange(T)
    downstream = pos >= e
    exact = pos < s
    field = (pos >= s) & (pos < e)

    # headline aggregate: max over layers per token
    attn_max = attn.max(0)
    kcos_max = k_cos.max(0); vcos_max = v_cos.max(0)
    kv_max = np.maximum(kcos_max, vcos_max)  # combined K/V cosine blast radius

    def br(arr, mask, tau):
        m = arr[mask]
        if m.size == 0:
            return 0.0
        return float((m > tau).mean())

    rec = {
        "field": field_key, "cls": f["cls"], "magnitude": magnitude,
        "n_cond_rules": len(f["cond_rules"]),
        "seq_len": T, "field_span": [s, e], "field_len": al["field_len"],
        "n_downstream": int(downstream.sum()), "nlayers": nlayers,
        "old_value": f["old"], "new_value": f[magnitude],
        # sanity: causally-exact region must be ~0
        "exact_kv_cos_max": float(kv_max[exact].max()) if exact.any() else 0.0,
        "exact_attn_max": float(attn_max[exact].max()) if exact.any() else 0.0,
        # blast radius over downstream tokens at several tau, for KV-cos and attn
        "br_kvcos": {f"{t:g}": br(kv_max, downstream, t) for t in tau_grid},
        "br_attn": {f"{t:g}": br(attn_max, downstream, t) for t in tau_grid},
        # summary stats of downstream deviation
        "attn_down_mean": float(attn_max[downstream].mean()),
        "attn_down_p50": float(np.median(attn_max[downstream])),
        "attn_down_p95": float(np.percentile(attn_max[downstream], 95)),
        "attn_down_max": float(attn_max[downstream].max()),
        "kvcos_down_mean": float(kv_max[downstream].mean()),
        "kvcos_down_p95": float(np.percentile(kv_max[downstream], 95)),
    }
    raw = dict(attn=attn, k_cos=k_cos, v_cos=v_cos, k_rl2=k_rl2, v_rl2=v_rl2,
               field_span=np.array([s, e]), seq_len=np.array([T]))
    return rec, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short label for output files")
    ap.add_argument("--magnitudes", default="semantic,minor")
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--no_attn", action="store_true")
    ap.add_argument("--fields", default="all")
    args = ap.parse_args()

    tau_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    tok, model = load_model(args.model)
    field_keys = list(C.FIELDS) if args.fields == "all" else args.fields.split(",")
    mags = args.magnitudes.split(",")

    records = []
    rawdir = os.path.join(RESULTS, f"raw_{args.tag}")
    os.makedirs(rawdir, exist_ok=True)
    t0 = time.time()
    for fk in field_keys:
        for mag in mags:
            r, raw = run_one(tok, model, fk, mag, args.n_neutral, tau_grid,
                             attn_dev=not args.no_attn)
            r["model"] = args.model; r["tag"] = args.tag
            records.append(r)
            np.savez_compressed(os.path.join(rawdir, f"{fk}_{mag}.npz"), **raw)
            print(f"[{time.time()-t0:6.1f}s] {fk:18s} {mag:9s} cls={r['cls']:6s} "
                  f"T={r['seq_len']:4d} down={r['n_downstream']:4d} "
                  f"exact_kv={r['exact_kv_cos_max']:.2e} exact_attn={r['exact_attn_max']:.2e} "
                  f"attn>0.05={r['br_attn']['0.05']:.3f} attn_p95={r['attn_down_p95']:.3f}",
                  flush=True)
            gc.collect(); torch.cuda.empty_cache()

    out = os.path.join(RESULTS, f"e1_{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(records, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
