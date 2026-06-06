"""Composable KV for agent SKILLs: precompile a chunk's KV once, re-position (RoPE) + splice it in.

Machinery:
  - reposition(keys, src_pos, tgt_pos): un-rotate cached keys from their source positions and re-rotate
    to target positions (values are position-free). HF caches POST-RoPE keys, so moving a precomputed
    chunk to a new location requires this.
  - precompute_chunk: prefill a chunk in ISOLATION (positions 0..N-1) -> its KV, reusable across requests.
  - splice_and_decide: build [prefix KV][repositioned chunk KV], prefill the suffix, read the decision.

Modes compared on instruction-following (does the SKILL still govern the decision?) + TTFT:
  full      : recompute everything [sys][skill][task]                       (baseline / oracle)
  precompiled: skill KV precomputed in isolation, RoPE-repositioned, spliced (no skill recompute)
Run: python esys/composable_kv.py --model Qwen/Qwen3-1.7B --sanity
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


@torch.no_grad()
def cos_sin(model, positions):
    rot = model.model.rotary_emb
    pos = torch.tensor([positions], device="cuda")
    dummy = torch.zeros(1, 1, 1, dtype=torch.float32, device="cuda")   # fp32 -> precise cos/sin
    cos, sin = rot(dummy, pos)               # [1, N, head_dim]
    return cos[:, None].float(), sin[:, None].float()   # [1, 1, N, head_dim], fp32


@torch.no_grad()
def reposition(model, keys, src_positions, tgt_positions):
    cs, ss = cos_sin(model, src_positions)
    ct, st = cos_sin(model, tgt_positions)
    k = keys.float()
    raw = k * cs - rotate_half(k) * ss                  # un-rotate from source (fp32)
    return (raw * ct + rotate_half(raw) * st).to(keys.dtype)   # re-rotate to target


@torch.no_grad()
def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True).past_key_values


def cache_slice(cache, lo, hi):
    d = DynamicCache()
    for i, l in enumerate(cache.layers):
        d.update(l.keys[:, :, lo:hi, :].clone(), l.values[:, :, lo:hi, :].clone(), i)
    return d


def cache_concat(*caches):
    d = DynamicCache()
    n = len(caches[0].layers)
    for i in range(n):
        ks = torch.cat([c.layers[i].keys for c in caches], dim=2)
        vs = torch.cat([c.layers[i].values for c in caches], dim=2)
        d.update(ks, vs, i)
    return d


@torch.no_grad()
def precompute_chunk(model, chunk_ids):
    """KV of a chunk computed in isolation at positions 0..N-1 (reusable)."""
    return prefill(model, chunk_ids)


@torch.no_grad()
def repositioned_chunk_cache(model, chunk_cache, N, dst_start):
    """Move a chunk precomputed at [0..N-1] to [dst_start..dst_start+N-1] (re-rotate keys).
    cos/sin computed ONCE and shared across layers (the per-request cost is one elementwise pass)."""
    cs, ss = cos_sin(model, list(range(N)))
    ct, st = cos_sin(model, list(range(dst_start, dst_start + N)))
    d = DynamicCache()
    for i, l in enumerate(chunk_cache.layers):
        k = l.keys.float()
        raw = k * cs - rotate_half(k) * ss
        kk = (raw * ct + rotate_half(raw) * st).to(l.keys.dtype)
        d.update(kk, l.values, i)
    return d


@torch.no_grad()
def forward_suffix(model, cache, suffix_ids, start_pos):
    """Prefill suffix_ids on top of `cache` (positions start_pos..)."""
    n = suffix_ids.shape[1]
    out = model(input_ids=suffix_ids.to("cuda"), past_key_values=cache,
                cache_position=torch.arange(start_pos, start_pos + n, device="cuda"), use_cache=True)
    return out


def sanity(model, tok):
    print("=== SANITY: RoPE round-trip + splice reproduces full prefill ===")
    txt = "The quick brown fox jumps over the lazy dog. " * 6
    ids = tok(txt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    L = ids.shape[1]
    full = prefill(model, ids)
    # round-trip: take keys at [10..L), move to [0..), then back to [10..) -> recover
    k0 = full.layers[0].keys[:, :, 10:L, :]
    N = L - 10
    moved = reposition(model, k0, list(range(10, L)), list(range(N)))
    back = reposition(model, moved, list(range(N)), list(range(10, L)))
    err = (back - k0).abs().max().item()
    print(f"  round-trip max abs err: {err:.2e}  ({'OK' if err < 1e-2 else 'FAIL'})")
    # splice: [A][B] full vs [A] + repositioned(B precomputed alone)
    a = L // 2
    A_ids, B_ids = ids[:, :a], ids[:, a:]
    full_logits = model(input_ids=ids.to("cuda")).logits[0, -1].float()
    A_cache = prefill(model, A_ids)
    B_alone = precompute_chunk(model, B_ids)            # B in isolation (no A context)
    B_rep = repositioned_chunk_cache(model, B_alone, B_ids.shape[1], a)
    spliced = cache_concat(cache_slice(A_cache, 0, a), B_rep)
    # decode the last token's logits via a 1-token forward at position L-1
    out = model(input_ids=B_ids[:, -1:].to("cuda"), past_key_values=cache_slice(spliced, 0, L - 1),
                cache_position=torch.tensor([L - 1], device="cuda"))
    spl_logits = out.logits[0, -1].float()
    cos = torch.cosine_similarity(full_logits, spl_logits, dim=0).item()
    agree = (full_logits.argmax() == spl_logits.argmax()).item()
    print(f"  full vs spliced(B-in-isolation): argmax-agree={agree}, logit cos-sim={cos:.3f}")
    print("  (cos<1 expected: B computed without attending to A; tests context-staleness, not a bug)")


# ---- SKILL feasibility: a long self-contained policy SKILL that GOVERNS a decision ----
FILLER_RULES = "\n".join(
    f"- General guideline {i+1}: maintain a professional tone, log the interaction, and follow standard "
    f"operating procedure for routine matters not otherwise specified." for i in range(20))

SKILLS = [
 dict(name="refund", sys="You are a customer-support agent.",
      skill=("# SKILL: REFUND_POLICY\n"
             "You handle refund requests. Core rule:\n"
             "RULE R1: A refund may be issued ONLY if order_status is \"delivered\". For any other status "
             "(pending, shipped, cancelled, returned) you MUST refuse the refund and escalate to a human.\n"
             f"{FILLER_RULES}\n"
             "End of REFUND_POLICY skill."),
      task="Order #7731 has order_status = \"pending\". The customer requests a refund. "
           "Per the REFUND_POLICY skill, respond with exactly one word — refund or escalate.\nDecision:",
      correct="escalate", wrong="refund"),
 dict(name="access", sys="You are an internal data-access assistant.",
      skill=("# SKILL: DATA_ACCESS_CONTROL\n"
             "You gate access to records. Core rule:\n"
             "RULE A1: Grant access to a CONFIDENTIAL record ONLY if the requester clearance is \"L4\" or higher. "
             "For clearance L1, L2, or L3 you MUST deny the request.\n"
             f"{FILLER_RULES}\n"
             "End of DATA_ACCESS_CONTROL skill."),
      task="A requester with clearance = \"L2\" asks to read a CONFIDENTIAL record. "
           "Per the DATA_ACCESS_CONTROL skill, respond with exactly one word — grant or deny.\nDecision:",
      correct="deny", wrong="grant"),
 dict(name="deploy", sys="You are a deployment guard bot.",
      skill=("# SKILL: DEPLOY_GUARD\n"
             "You authorize deployments. Core rule:\n"
             "RULE D1: Deployments to environment \"production\" require an approved change ticket. If "
             "change_ticket is \"none\", you MUST block the deployment to production.\n"
             f"{FILLER_RULES}\n"
             "End of DEPLOY_GUARD skill."),
      task="A deploy to environment = \"production\" has change_ticket = \"none\". "
           "Per the DEPLOY_GUARD skill, respond with exactly one word — allow or block.\nDecision:",
      correct="block", wrong="allow"),
 dict(name="rx", sys="You are a pharmacy verification assistant.",
      skill=("# SKILL: RX_SAFETY\n"
             "You verify prescriptions. Core rule:\n"
             "RULE X1: Do NOT dispense a medication if it appears on the patient's recorded allergy list. "
             "If the drug is on the allergy list, you MUST hold and flag for pharmacist review.\n"
             f"{FILLER_RULES}\n"
             "End of RX_SAFETY skill."),
      task="The requested drug \"amoxicillin\" IS on the patient's allergy list. "
           "Per the RX_SAFETY skill, respond with exactly one word — dispense or hold.\nDecision:",
      correct="hold", wrong="dispense"),
]


def chat_parts(tok, sys_txt, skill_txt, task_txt):
    """Return token ids for [chat-prefix+sys+skill], split so the skill is an isolatable middle chunk.
    We assemble the whole user message then locate the skill span by string offsets."""
    body = f"{sys_txt}\n\n{skill_txt}\n\n{task_txt}"
    full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    s_char = full.find(skill_txt); e_char = s_char + len(skill_txt)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= s_char < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= e_char), len(offs))
    return ids, a, b


@torch.no_grad()
def decision(model, logits, tok, correct, wrong):
    tc = tok(correct, add_special_tokens=False)["input_ids"][0]
    tw = tok(wrong, add_special_tokens=False)["input_ids"][0]
    return "correct" if logits[tc] >= logits[tw] else "wrong"


@torch.no_grad()
def run_experiment(model, tok, trials=5):
    res = {"full": 0, "precompiled": 0, "n": 0}
    tfull, tpre = [], []
    for sk in SKILLS:
        ids, a, b = chat_parts(tok, sk["sys"], sk["skill"], sk["task"])
        L = ids.shape[1]; tc = tok(sk["correct"], add_special_tokens=False)["input_ids"][0]
        tw = tok(sk["wrong"], add_special_tokens=False)["input_ids"][0]
        # FULL recompute
        torch.cuda.synchronize(); t0 = time.perf_counter()
        full_cache = prefill(model, ids[:, :L - 1])
        torch.cuda.synchronize(); tfull.append((time.perf_counter() - t0) * 1000)
        fl = model(input_ids=ids[:, L - 1:L].to("cuda"), past_key_values=cache_slice(full_cache, 0, L - 1),
                   cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        fd = "correct" if fl[tc] >= fl[tw] else "wrong"
        # PRECOMPILED: skill KV precomputed in isolation, repositioned to [a..b), spliced
        skill_ids = ids[:, a:b]; nb = b - a
        skill_alone = precompute_chunk(model, skill_ids)                       # offline, once
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pre_cache = prefill(model, ids[:, :a])                                 # [chat+sys]
        skill_rep = repositioned_chunk_cache(model, skill_alone, nb, a)        # cheap re-rotate
        spliced = cache_concat(pre_cache, skill_rep)
        out = forward_suffix(model, spliced, ids[:, b:L - 1], b)               # only the task suffix
        torch.cuda.synchronize(); tpre.append((time.perf_counter() - t0) * 1000)
        pl = model(input_ids=ids[:, L - 1:L].to("cuda"), past_key_values=cache_slice(out.past_key_values, 0, L - 1),
                   cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        pd = "correct" if pl[tc] >= pl[tw] else "wrong"
        res["full"] += (fd == "correct"); res["precompiled"] += (pd == "correct"); res["n"] += 1
        print(f"  {sk['name']:8s} skill_tok={nb:4d} | full={fd:7s} precompiled={pd:7s} | "
              f"logit-cos={torch.cosine_similarity(fl, pl, 0).item():.3f}", flush=True)
    print(f"\n  FULL correct: {res['full']}/{res['n']} | PRECOMPILED correct: {res['precompiled']}/{res['n']}")
    print(f"  TTFT: full median={sorted(tfull)[len(tfull)//2]:.1f}ms  precompiled median={sorted(tpre)[len(tpre)//2]:.1f}ms")
    return res


@torch.no_grad()
def run_scaling(model, tok, lens=(500, 2000, 8000, 16000, 32000), trials=5):
    """TTFT(full prefill of [sys][skill][task]) vs TTFT(precompiled: prefill [sys]+[task] + reposition skill).
    full attention is O(L^2); the precompiled per-request cost is O(L) re-rotation + small prefill."""
    sys_ids = tok("You are an agent.\n\n", add_special_tokens=False)["input_ids"]
    task_ids = tok("\n\nNow answer the user's question.\nDecision:", add_special_tokens=False)["input_ids"]
    print(f"  {'skill_tok':>9} {'full_ms':>9} {'precomp_ms':>11} {'speedup':>8}")
    out = {}
    for Lskill in lens:
        skill_ids = torch.randint(0, tok.vocab_size, (1, Lskill), device="cuda")
        sysz = torch.tensor([sys_ids], device="cuda"); taskz = torch.tensor([task_ids], device="cuda")
        full_ids = torch.cat([sysz, skill_ids, taskz], dim=1)
        a = sysz.shape[1]; b = a + Lskill
        skill_alone = precompute_chunk(model, skill_ids)            # offline (excluded from timing)
        tf, tp = [], []
        for _ in range(trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model(input_ids=full_ids, use_cache=True)
            torch.cuda.synchronize(); tf.append((time.perf_counter() - t0) * 1000)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pre = prefill(model, sysz)
            rep = repositioned_chunk_cache(model, skill_alone, Lskill, a)
            spliced = cache_concat(pre, rep)
            forward_suffix(model, spliced, taskz, b)
            torch.cuda.synchronize(); tp.append((time.perf_counter() - t0) * 1000)
        f = sorted(tf)[len(tf) // 2]; p = sorted(tp)[len(tp) // 2]
        out[Lskill] = {"full_ms": round(f, 1), "precomp_ms": round(p, 1), "speedup": round(f / p, 2)}
        print(f"  {Lskill:>9} {f:>9.1f} {p:>11.1f} {f/p:>7.2f}x", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    if args.sanity:
        sanity(model, tok)
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    if args.experiment:
        print(f"=== SKILL feasibility ({args.model}) ===")
        run_experiment(model, tok)
    if args.scaling:
        print(f"=== TTFT scaling: full vs precompiled ({args.model}) ===")
        sc = run_scaling(model, tok)
        json.dump({"model": args.model, "scaling": sc},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_scaling_{tag}.json"), "w"), indent=2)
    print("COMPOSABLE_KV_DONE")


if __name__ == "__main__":
    main()
