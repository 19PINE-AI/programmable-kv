"""C1 — Transplant mechanism: WHY a precompiled chunk's KV is portable, and WHERE it degrades.

Three probes (composable analog of the editable D1 mechanism study):
 (A) POSITION-PORTABILITY: a chunk computed in isolation, RoPE-repositioned to offset P and spliced
     after a length-P prefix, reproduces the native logits — across a wide range of P. Ablation:
     splicing WITHOUT re-rotation collapses (shows RoPE re-rotation is necessary & sufficient).
 (B) CONTEXT-ROBUSTNESS: native (chunk attended to the real prefix) vs transplanted (isolation):
     logit cos-sim quantifies how much the chunk needed the prefix.
 (C) SEAM LOCATION: per-position KV deviation (transplanted vs native) — locates the tokens that need
     recompute (the 'seam'); ties to selective recompute / boundary repair (#48).
Run: MECH_ATTN=sdpa python esys/transplant_mech.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import (prefill, cos_sin, rotate_half, cache_slice, cache_concat,
                           precompute_chunk, repositioned_chunk_cache, forward_suffix)


def naive_splice(chunk_cache):
    """Splice the isolation KV with NO re-rotation (keys keep position-0 RoPE) — the ablation."""
    d = DynamicCache()
    for i, l in enumerate(chunk_cache.layers):
        d.update(l.keys.clone(), l.values.clone(), i)
    return d


@torch.no_grad()
def logits_after(model, cache, last_tok, pos):
    o = model(input_ids=torch.tensor([[int(last_tok)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    return o.logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    chunk_txt = ("# SKILL: ESCALATION\nRULE: If the customer is a VIP, route to a senior agent and waive "
                 "the first fee; otherwise follow the standard queue and standard fees. Always confirm the "
                 "account before any change.\n" + "\n".join(f"- Note {i}: follow SOP." for i in range(40)))
    probe_txt = "\n\nGiven the ESCALATION skill, the first step is to"
    chunk_ids = tok(chunk_txt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    probe_ids = tok(probe_txt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    N = chunk_ids.shape[1]; chunk_alone = precompute_chunk(model, chunk_ids)

    out = {"model": args.model, "N_chunk": N, "portability": {}, "seam": {}}
    print(f"=== TRANSPLANT MECHANISM ({args.model}), chunk={N} tok ===")
    print(f"  (A) POSITION-PORTABILITY: logit cos-sim vs prefix length P  [reposition vs naive-splice]")
    for P in [0, 64, 256, 1024, 4096]:
        prefix = torch.randint(0, tok.vocab_size, (1, P), device="cuda") if P > 0 else None
        # native: [prefix][chunk] then probe
        if P > 0:
            nat = cache_concat(prefill(model, prefix), DynamicCache()) if False else prefill(model, prefix)
            nat = forward_suffix(model, nat, chunk_ids, P).past_key_values
        else:
            nat = prefill(model, chunk_ids)
        Lnat = P + N
        nat_full = forward_suffix(model, cache_slice(nat, 0, Lnat), probe_ids, Lnat).past_key_values
        nat_logits = logits_after(model, nat_full, probe_ids[0, -1], Lnat + probe_ids.shape[1] - 1)
        # transplanted (RoPE-repositioned)
        base = prefill(model, prefix) if P > 0 else DynamicCache()
        rep = repositioned_chunk_cache(model, chunk_alone, N, P)
        tr = cache_concat(base, rep) if P > 0 else rep
        tr = forward_suffix(model, tr, probe_ids, P + N).past_key_values
        tr_logits = logits_after(model, tr, probe_ids[0, -1], P + N + probe_ids.shape[1] - 1)
        # naive (no re-rotation)
        base2 = prefill(model, prefix) if P > 0 else DynamicCache()
        nv = cache_concat(base2, naive_splice(chunk_alone)) if P > 0 else naive_splice(chunk_alone)
        nv = forward_suffix(model, nv, probe_ids, P + N).past_key_values
        nv_logits = logits_after(model, nv, probe_ids[0, -1], P + N + probe_ids.shape[1] - 1)
        cr = torch.cosine_similarity(nat_logits, tr_logits, 0).item()
        cn = torch.cosine_similarity(nat_logits, nv_logits, 0).item()
        out["portability"][P] = {"reposition_cos": round(cr, 4), "naive_cos": round(cn, 4),
                                 "argmax_agree": bool(nat_logits.argmax() == tr_logits.argmax())}
        print(f"    P={P:>5}: reposition cos={cr:.4f} (argmax-agree={nat_logits.argmax()==tr_logits.argmax()}) | naive cos={cn:.4f}", flush=True)

    # (C) SEAM: per-position KV L2 deviation transplanted(isolation, repositioned to P) vs native, at P=256
    P = 256
    prefix = torch.randint(0, tok.vocab_size, (1, P), device="cuda")
    nat = forward_suffix(model, prefill(model, prefix), chunk_ids, P).past_key_values
    rep = repositioned_chunk_cache(model, chunk_alone, N, P)
    dev = torch.zeros(N)
    for i in range(len(nat.layers)):
        nk = nat.layers[i].keys[:, :, P:P + N, :].float(); rk = rep.layers[i].keys.float()
        dev += (nk - rk).norm(dim=-1).mean(dim=1).squeeze(0).cpu()
    dev = dev / len(nat.layers)
    topk = torch.argsort(dev, descending=True)[:8].tolist()
    out["seam"] = {"mean_dev_first8": round(dev[:8].mean().item(), 3), "mean_dev_last8": round(dev[-8:].mean().item(), 3),
                   "top_dev_positions": topk, "frac_in_first_10pct": round((torch.argsort(dev, descending=True)[:max(1,N//10)] < N//10).float().mean().item(), 2)}
    print(f"  (C) SEAM: mean KV-dev first-8 chunk tok={dev[:8].mean():.3f} vs last-8={dev[-8:].mean():.3f}  "
          f"(higher at the START = tokens that needed the prefix)")
    print(f"      top-deviation chunk positions: {topk}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"transplant_mech_{tag}.json"), "w"), indent=2)
    print("TRANSPLANT_MECH_DONE")


if __name__ == "__main__":
    main()
