"""E2 slice: decision-flip faithfulness under leave-stale KV patching.

For each (field, flip) we compare the model's generated DECISION under four
KV-cache regimes, all decoded greedily from the same context:

  ORACLE_NEW  : full prefill of the NEW context              (ground truth)
  PATCHED     : OLD cache, field span refreshed to NEW,
                everything downstream LEFT STALE              (our mechanism)
  STALE_FULL  : OLD cache, NOT even the field refreshed       (floor)
  ORACLE_OLD  : full prefill of the OLD context               (pre-flip decision)

Key questions:
  * agree(PATCHED, ORACLE_NEW): does leave-stale reproduce the correct new
    decision?  H4 predicts ~yes for low-conditioning fields.
  * changed = (ORACLE_OLD != ORACLE_NEW): is the field decision-relevant at all?
    When it changed, does PATCHED track the change (vs collapsing to STALE_FULL)?
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
import capture                      # noqa: keep eager+correct masking available
from align import align_pair
import contexts as C
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    capture.install(model)   # safe: only records when capture.enable_capture() is on
    return tok, model


def build_text(field_key, value, n_neutral, tok, use_chat):
    ctx = C.build_context(field_key, value, n_neutral)
    if use_chat:
        msgs = [{"role": "user", "content": ctx}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return ctx


@torch.no_grad()
def prefill(model, ids):
    out = model(input_ids=ids.to("cuda"), use_cache=True)
    return out.past_key_values


def clone_cache(cache, upto):
    c = DynamicCache()
    for i, layer in enumerate(cache.layers):
        c.update(layer.keys[:, :, :upto, :].clone(),
                 layer.values[:, :, :upto, :].clone(), i)
    return c


def patched_cache(cache_old, cache_new, span, upto):
    """old cache truncated to `upto`, with field span overwritten by new."""
    s, e = span
    c = clone_cache(cache_old, upto)
    for i, layer in enumerate(cache_new.layers):
        c.layers[i].keys[:, :, s:e, :] = layer.keys[:, :, s:e, :]
        c.layers[i].values[:, :, s:e, :] = layer.values[:, :, s:e, :]
    return c


@torch.no_grad()
def greedy_decode(model, cache, last_tok, start_pos, max_new=48, eos_ids=None):
    """Decode from a cache of length start_pos, feeding last_tok at start_pos."""
    toks = []
    cur = last_tok.view(1, 1).to("cuda")
    pos = start_pos
    for _ in range(max_new):
        cp = torch.tensor([pos], device="cuda")
        out = model(input_ids=cur, past_key_values=cache, cache_position=cp, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
        toks.append(nxt)
        if eos_ids and nxt in eos_ids:
            break
        cur = torch.tensor([[nxt]], device="cuda")
        pos += 1
    return toks


def first_line(tok, ids):
    txt = tok.decode(ids, skip_special_tokens=True)
    return txt.split("\n")[0].strip()


def run_one(tok, model, field_key, magnitude, n_neutral, use_chat, max_new):
    f = C.FIELDS[field_key]
    old_text = build_text(field_key, f["old"], n_neutral, tok, use_chat)
    new_text = build_text(field_key, f[magnitude], n_neutral, tok, use_chat)
    al = align_pair(tok, old_text, new_text)
    s, e = al["field_span"]; T = al["seq_len"]
    old_ids, new_ids = al["old_ids"], al["new_ids"]

    co = prefill(model, old_ids)
    cn = prefill(model, new_ids)

    eos = {tok.eos_token_id}
    last_new = new_ids[0, T - 1]
    last_old = old_ids[0, T - 1]  # == last_new (suffix identical)

    # ORACLE_NEW
    g_oracle_new = greedy_decode(model, clone_cache(cn, T - 1), last_new, T - 1, max_new, eos)
    # ORACLE_OLD
    g_oracle_old = greedy_decode(model, clone_cache(co, T - 1), last_old, T - 1, max_new, eos)
    # PATCHED (leave-stale)
    g_patched = greedy_decode(model, patched_cache(co, cn, (s, e), T - 1), last_new, T - 1, max_new, eos)
    # STALE_FULL (no field refresh)
    g_stale = greedy_decode(model, clone_cache(co, T - 1), last_new, T - 1, max_new, eos)

    L = {"oracle_new": first_line(tok, g_oracle_new),
         "oracle_old": first_line(tok, g_oracle_old),
         "patched": first_line(tok, g_patched),
         "stale_full": first_line(tok, g_stale)}

    def tok_agree(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        return sum(int(a[i] == b[i]) for i in range(n)) / max(len(a), len(b))

    rec = {
        "field": field_key, "cls": f["cls"], "magnitude": magnitude,
        "n_cond_rules": len(f["cond_rules"]), "seq_len": T, "field_span": [s, e],
        "lines": L,
        "first_tok": {k: (g[0] if g else None) for k, g in
                      dict(oracle_new=g_oracle_new, oracle_old=g_oracle_old,
                           patched=g_patched, stale_full=g_stale).items()},
        "patched_eq_oracle_new": L["patched"] == L["oracle_new"],
        "patched_firsttok_eq_oracle_new": (g_patched[:1] == g_oracle_new[:1]),
        "patched_toptok_agree_oracle_new": tok_agree(g_patched, g_oracle_new),
        "stale_eq_oracle_new": L["stale_full"] == L["oracle_new"],
        "decision_changed": L["oracle_old"] != L["oracle_new"],
        "patched_tracks_change": (L["oracle_old"] != L["oracle_new"]) and (L["patched"] == L["oracle_new"]),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--magnitudes", default="semantic")
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--fields", default="all")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    field_keys = list(C.FIELDS) if args.fields == "all" else args.fields.split(",")
    recs = []
    for fk in field_keys:
        for mag in args.magnitudes.split(","):
            r = run_one(tok, model, fk, mag, args.n_neutral, args.chat, args.max_new)
            r["model"] = args.model; r["tag"] = args.tag
            recs.append(r)
            print(f"{fk:18s} {mag:9s} cls={r['cls']:6s} "
                  f"changed={int(r['decision_changed'])} "
                  f"patch==oracle_new={int(r['patched_eq_oracle_new'])} "
                  f"stale==oracle_new={int(r['stale_eq_oracle_new'])} "
                  f"toptok_agree={r['patched_toptok_agree_oracle_new']:.2f}", flush=True)
            print(f"    oracle_old : {r['lines']['oracle_old'][:80]}")
            print(f"    oracle_new : {r['lines']['oracle_new'][:80]}")
            print(f"    patched    : {r['lines']['patched'][:80]}")
            print(f"    stale_full : {r['lines']['stale_full'][:80]}")
    out = os.path.join(RES, f"e2_{args.tag}.json")
    json.dump(recs, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
