"""U2 — Unified substrate: one KV cache exposing BOTH compose(skill) and edit(field), end-to-end.

A realistic agent turn: a system prompt + a LIBRARY of precompiled SKILLs (composed by KV splice, no
reprefill) + a session context with a MUTABLE field. When the field changes, we apply an in-place
EDIT (erratum) on the same cache. We compare the unified path (compose + edit) to a full reprefill on
(a) the decision and (b) wall-clock TTFT. Demonstrates editable AND composable as one system.
Run: python esys/editkv_unified.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import (prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

FILLER = "\n".join(f"- guideline {i}: follow SOP and log the interaction." for i in range(18))
SKILLS = [
 "# SKILL: REFUND\nRULE: refund only if order_status is delivered; else escalate.\n" + FILLER,
 "# SKILL: ACCESS\nRULE: grant a CONFIDENTIAL record only if clearance is L4+; else deny.\n" + FILLER,
 "# SKILL: ESCALATION\nRULE: page a senior agent for any VIP customer dispute.\n" + FILLER,
]
SYS = "You are a customer-operations agent. Apply the loaded SKILLS."


class EditableComposableCache:
    """One KV cache supporting compose(skill) [precompiled splice] and edit(field) [erratum]."""
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.chunks = {}

    def precompile(self, name, text):
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        self.chunks[name] = (precompute_chunk(self.model, ids), ids.shape[1])

    def build(self, prefix_ids, skill_names, suffix_ids):
        """[prefix] + composed precompiled skills + [suffix]; returns (cache, total_len)."""
        cache = prefill(self.model, prefix_ids); pos = prefix_ids.shape[1]
        sep = self.tok("\n\n", add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        for nm in skill_names:
            chunk, n = self.chunks[nm]
            cache = cache_concat(cache, repositioned_chunk_cache(self.model, chunk, n, pos)); pos += n
            cache = forward_suffix(self.model, cache, sep, pos).past_key_values; pos += sep.shape[1]
        cache = forward_suffix(self.model, cache, suffix_ids, pos).past_key_values; pos += suffix_ids.shape[1]
        return cache, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default=None)
    ap.add_argument("--pad", type=int, default=0, help="pad each skill to ~N tokens (realistic long skills)")
    args = ap.parse_args()
    if args.pad:
        global SKILLS
        extra = "\n".join(f"- detailed procedure note {i}: handle the documented edge case per SOP and record the outcome." for i in range(args.pad // 12))
        SKILLS = [s + "\n" + extra for s in SKILLS]
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quant = any(q in args.model.upper() for q in ("FP8", "-INT8", "GPTQ", "AWQ", "W8A", "W4A"))
    kw = dict(device_map="cuda", attn_implementation="sdpa", trust_remote_code=True)
    if not quant:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    tg = tok("escalate", add_special_tokens=False)["input_ids"][0]
    tw = tok("refund", add_special_tokens=False)["input_ids"][0]

    ecc = EditableComposableCache(model, tok)
    for i, s in enumerate(SKILLS):
        ecc.precompile(f"S{i}", s)
    # prefix = chat-templated [sys]; we approximate by templating the whole and locating spans is overkill;
    # instead build prefix/suffix as raw segments around the skills (no chat template for the splice demo).
    prefix_txt = SYS + "\n\n"
    # session context carries the MUTABLE field; task invokes REFUND
    def suffix(order_status):
        return (f"\n\nSESSION: order_status = {order_status}.\nTASK: the customer requests a refund. "
                f"Per the REFUND skill, answer one word — refund or escalate.\nDecision:")
    pre = tok(prefix_txt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda")

    @torch.no_grad()
    def decide(cache, L):
        lg = model(input_ids=tok(suffix("x"), add_special_tokens=False, return_tensors="pt")["input_ids"][:, -1:].to("cuda"),
                   past_key_values=cache_slice(cache, 0, L - 1), cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        return "escalate" if lg[tg] >= lg[tw] else "refund"

    # ---- UNIFIED: compose 3 precompiled skills + build context with order_status=pending ----
    suf_old = tok(suffix("pending"), add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cache, L = ecc.build(pre, ["S0", "S1", "S2"], suf_old)
    torch.cuda.synchronize(); t_compose = (time.perf_counter() - t0) * 1000
    d_old = decide(cache, L)
    # ---- EDIT: order_status pending->delivered via erratum (append on the SAME cache) ----
    err = tok("\n[STATE UPDATE] order_status has changed to delivered; this overrides any earlier value AND conclusion.\nDecision:",
              add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ecache = forward_suffix(model, cache_slice(cache, 0, L - 1), err, L - 1).past_key_values
    torch.cuda.synchronize(); t_edit = (time.perf_counter() - t0) * 1000
    le = (L - 1) + err.shape[1]
    elg = model(input_ids=err[:, -1:], past_key_values=cache_slice(ecache, 0, le - 1), cache_position=torch.tensor([le - 1], device="cuda")).logits[0, -1].float()
    d_edit = "escalate" if elg[tg] >= elg[tw] else "refund"

    # ---- BASELINE: full reprefill of [sys][skills][order_status=delivered][task] ----
    full_txt = prefix_txt + "\n\n".join(SKILLS) + suffix("delivered")
    fids = tok(full_txt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda"); Lf = fids.shape[1]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fc = prefill(model, fids[:, :Lf - 1])
    torch.cuda.synchronize(); t_full = (time.perf_counter() - t0) * 1000
    flg = model(input_ids=fids[:, Lf - 1:Lf], past_key_values=cache_slice(fc, 0, Lf - 1), cache_position=torch.tensor([Lf - 1], device="cuda")).logits[0, -1].float()
    d_full = "escalate" if flg[tg] >= flg[tw] else "refund"

    out = {"model": args.model, "decision_before_edit(pending)": d_old, "decision_after_edit(delivered)": d_edit,
           "decision_full_reprefill(delivered)": d_full, "agree_edit_vs_full": d_edit == d_full,
           "t_compose_ms": round(t_compose, 1), "t_edit_ms": round(t_edit, 1),
           "t_unified_total_ms": round(t_compose + t_edit, 1), "t_full_reprefill_ms": round(t_full, 1),
           "edit_speedup_vs_full": round(t_full / t_edit, 1), "L_tokens": L}
    print(f"=== UNIFIED edit+compose agent turn ({args.model}) ===")
    print(f"  composed 3 precompiled skills + context (order_status=pending): decision={d_old}")
    print(f"  EDIT order_status pending->delivered (erratum on same cache): decision={d_edit}  [{t_edit:.1f}ms]")
    print(f"  full reprefill (delivered): decision={d_full}  [{t_full:.1f}ms]")
    print(f"  agree(edit==full)={d_edit==d_full} | edit is {t_full/t_edit:.1f}x faster than full reprefill | L={L} tok")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"editkv_unified_{tag}.json"), "w"), indent=2)
    print("UNIFIED_DONE")


if __name__ == "__main__":
    main()
