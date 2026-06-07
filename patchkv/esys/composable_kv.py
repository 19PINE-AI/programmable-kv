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

# Shim legacy cache APIs that some custom modeling files (e.g. DeepSeek-V2 MLA) still call.
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)


def load_lm(name, attn="eager"):
    """Uniform loader: bf16 default; FP8/AWQ/GPTQ quant checkpoints as-is; bnb-4bit for 70B; gemma-3 text-only."""
    from transformers import AutoModelForCausalLM as AM
    up = name.upper()
    if any(k in up for k in ("DEEPSEEK-V2", "DEEPSEEK-CODER-V2", "JAMBA", "ZAMBA")):
        attn = "eager"   # MLA / some hybrids lack an sdpa path in this transformers version
    kw = dict(device_map="cuda", attn_implementation=attn, trust_remote_code=True)
    if "4BIT" in up or "BNB-4" in up:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    elif any(q in up for q in ("FP8", "-INT8", "GPTQ", "AWQ", "W8A", "W4A", "QUANTIZED.W")):
        pass
    elif "GEMMA-3" in up:
        from transformers import Gemma3ForCausalLM
        return Gemma3ForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw).eval()
    else:
        kw["dtype"] = torch.bfloat16
    return AM.from_pretrained(name, **kw).eval()


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
def _as_dyn(pkv):
    """Convert a legacy tuple cache (some custom modeling, e.g. DeepSeek-V2 MLA) to DynamicCache."""
    if pkv is not None and not hasattr(pkv, "layers"):
        return DynamicCache.from_legacy_cache(pkv)
    return pkv


def prefill(model, ids):
    # Pass an explicit DynamicCache so sliding-window models (Gemma-2/3) keep the FULL per-layer KV
    # (the window is enforced by the attention mask, not by truncating the cache) — this makes the
    # uniform reposition/slice/concat ops work for sequences/chunks beyond the window.
    return _as_dyn(model(input_ids=ids.to("cuda"), past_key_values=DynamicCache(),
                         use_cache=True).past_key_values)


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
 dict(name="loan", sys="You are a lending assistant.",
      skill=("# SKILL: LOAN_POLICY\nRULE L1: Approve a loan ONLY if the applicant's debt-to-income ratio "
             "is at most 0.40. If it exceeds 0.40, you MUST decline.\n" + FILLER_RULES + "\nEnd."),
      task="The applicant's debt-to-income ratio is 0.62. Per LOAN_POLICY, one word — approve or decline.\nDecision:",
      correct="decline", wrong="approve"),
 dict(name="legal", sys="You are a contracts assistant.",
      skill=("# SKILL: NDA_POLICY\nRULE N1: Share a CONFIDENTIAL document with a third party ONLY if a "
             "signed NDA is on file. If no NDA is on file, you MUST refuse.\n" + FILLER_RULES + "\nEnd."),
      task="A third party requests a CONFIDENTIAL document; no NDA is on file. One word — share or refuse.\nDecision:",
      correct="refuse", wrong="share"),
 dict(name="incident", sys="You are a security operations bot.",
      skill=("# SKILL: INCIDENT_POLICY\nRULE I1: Auto-isolate a host ONLY if the threat severity is "
             "critical. For low or medium severity, you MUST monitor instead of isolating.\n" + FILLER_RULES + "\nEnd."),
      task="A host shows a low-severity anomaly. Per INCIDENT_POLICY, one word — isolate or monitor.\nDecision:",
      correct="monitor", wrong="isolate"),
 dict(name="visa", sys="You are an immigration desk assistant.",
      skill=("# SKILL: VISA_POLICY\nRULE V1: Grant entry ONLY if the passport is valid for at least 6 "
             "months. If the passport expires sooner, you MUST refer to secondary inspection.\n" + FILLER_RULES + "\nEnd."),
      task="The passport expires in 2 months. Per VISA_POLICY, one word — grant or refer.\nDecision:",
      correct="refer", wrong="grant"),
]


# Coupled skills: the governing threshold lives in the SYS/context, and the skill REFERS to it.
# A precompiled-in-isolation skill never saw the threshold -> should degrade (tests self-containedness).
COUPLED = [
 dict(name="access_cpl",
      sys="You are a data-access assistant. SYSTEM CONFIG: the minimum clearance for CONFIDENTIAL "
          "records is set to \"L4\" for this deployment.",
      skill=("# SKILL: DATA_ACCESS_CONTROL\n"
             "RULE A1: Grant access to a CONFIDENTIAL record ONLY if the requester clearance meets or "
             "exceeds the minimum clearance specified in the SYSTEM CONFIG above. Otherwise deny.\n"
             f"{FILLER_RULES}\nEnd of skill."),
      task="A requester with clearance = \"L2\" asks to read a CONFIDENTIAL record. "
           "Respond with exactly one word — grant or deny.\nDecision:",
      correct="deny", wrong="grant"),
 dict(name="refund_cpl",
      sys="You are a support agent. SYSTEM CONFIG: refunds are allowed ONLY for order_status equal to "
          "the value \"delivered\" in this deployment.",
      skill=("# SKILL: REFUND_POLICY\n"
             "RULE R1: Issue a refund ONLY if the order_status equals the refund-eligible status named "
             "in the SYSTEM CONFIG above; for any other status refuse and escalate.\n"
             f"{FILLER_RULES}\nEnd of skill."),
      task="Order #7731 has order_status = \"pending\". The customer requests a refund. "
           "Respond with exactly one word — refund or escalate.\nDecision:",
      correct="escalate", wrong="refund"),
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
def run_experiment(model, tok, skills=None, label="self-contained"):
    skills = skills or SKILLS
    res = {"full": 0, "precompiled": 0, "agree": 0, "n": 0, "cos": []}
    tfull, tpre = [], []
    print(f"  --- {label} skills ---")
    for sk in skills:
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
        cosv = torch.cosine_similarity(fl, pl, 0).item()
        # NAIVE ablation: splice isolation KV WITHOUT re-rotation (keys keep position-0 RoPE)
        naive = DynamicCache()
        for li, l in enumerate(skill_alone.layers):
            naive.update(l.keys.clone(), l.values.clone(), li)
        nout = forward_suffix(model, cache_concat(pre_cache, naive), ids[:, b:L - 1], b)
        nl = model(input_ids=ids[:, L - 1:L].to("cuda"), past_key_values=cache_slice(nout.past_key_values, 0, L - 1),
                   cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        nd = "correct" if nl[tc] >= nl[tw] else "wrong"; ncos = torch.cosine_similarity(fl, nl, 0).item()
        res["full"] += (fd == "correct"); res["precompiled"] += (pd == "correct")
        res["agree"] += (fd == pd); res["cos"].append(cosv); res["n"] += 1
        res["naive_agree"] = res.get("naive_agree", 0) + (fd == nd); res["naive_cos"] = res.get("naive_cos", []) + [ncos]
        print(f"  {sk['name']:10s} skill_tok={nb:4d} | full={fd:7s} reposition={pd:7s}(cos{cosv:.2f}) "
              f"naive={nd:7s}(cos{ncos:.2f}) | agree={fd==pd}", flush=True)
    mc = sum(res["cos"]) / len(res["cos"]); nmc = sum(res["naive_cos"]) / len(res["naive_cos"])
    print(f"  [{label}] full_correct={res['full']}/{res['n']} precompiled_correct={res['precompiled']}/{res['n']} "
          f"| reposition==full: {res['agree']}/{res['n']} (cos{mc:.3f}) | naive==full: {res['naive_agree']}/{res['n']} (cos{nmc:.3f})")
    if tfull:
        print(f"  TTFT: full median={sorted(tfull)[len(tfull)//2]:.1f}ms  precompiled median={sorted(tpre)[len(tpre)//2]:.1f}ms")
    res["mean_cos"] = round(mc, 3)
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


LIB = [  # a small SKILL LIBRARY; each precompiled once, composed in arbitrary subsets/orders
 ("REFUND", "# SKILL: REFUND\nRULE: refund only if order_status is delivered; else escalate.\n" + FILLER_RULES),
 ("ACCESS", "# SKILL: ACCESS\nRULE: grant a CONFIDENTIAL record only if clearance is L4+; else deny.\n" + FILLER_RULES),
 ("DEPLOY", "# SKILL: DEPLOY\nRULE: block a production deploy if change_ticket is none.\n" + FILLER_RULES),
 ("RX", "# SKILL: RX\nRULE: hold a medication if it is on the patient's allergy list.\n" + FILLER_RULES),
]


@torch.no_grad()
def run_multiskill(model, tok):
    """Compose N precompiled skills (each repositioned to its slot) vs full recompute. The TASK invokes
    one skill (ACCESS). Measure decision agreement + TTFT for N = 1..4 composed skills."""
    sys_txt = "You are an agent with access to the following SKILLS.\n\n"
    task = "\n\nA requester with clearance = L2 asks to read a CONFIDENTIAL record (per ACCESS). One word — grant or deny.\nDecision:"
    tc = tok("deny", add_special_tokens=False)["input_ids"][0]; tw = tok("grant", add_special_tokens=False)["input_ids"][0]
    print(f"  {'#skills':>7} {'full':>8} {'composed':>9} {'agree':>6} {'full_ms':>8} {'comp_ms':>8} {'speedup':>8}")
    out = {}
    for N in range(1, len(LIB) + 1):
        skills = LIB[:N]
        # build full text with explicit skill spans
        parts = [sys_txt]; spans = []
        cur = sys_txt
        for nm, tx in skills:
            cur_start = len(cur); cur += tx + "\n\n"; spans.append((nm, tx))
        body = sys_txt + "\n\n".join(tx for _, tx in skills) + task
        full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]]).to("cuda"); offs = enc["offset_mapping"]; L = ids.shape[1]
        # locate each skill span
        locs = []
        for _, tx in skills:
            sc = full.find(tx); ec = sc + len(tx)
            a = next(i for i, (lo, hi) in enumerate(offs) if lo <= sc < hi)
            b = next((i for i, (lo, hi) in enumerate(offs) if lo >= ec), L)
            locs.append((a, b))
        # FULL
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fc = prefill(model, ids[:, :L - 1])
        torch.cuda.synchronize(); tf = (time.perf_counter() - t0) * 1000
        fl = model(input_ids=ids[:, L - 1:L], past_key_values=cache_slice(fc, 0, L - 1),
                   cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        fd = "deny" if fl[tc] >= fl[tw] else "grant"
        # COMPOSED: each skill precompiled in isolation, repositioned to its slot
        chunks = [precompute_chunk(model, ids[:, a:b]) for (a, b) in locs]   # offline
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cache = prefill(model, ids[:, :locs[0][0]])                          # [sys]
        pos = locs[0][0]
        for (a, b), ch in zip(locs, chunks):
            if a > pos:  # inter-skill text (the "\n\n")
                cache = forward_suffix(model, cache, ids[:, pos:a], pos).past_key_values; pos = a
            cache = cache_concat(cache, repositioned_chunk_cache(model, ch, b - a, a)); pos = b
        cache = forward_suffix(model, cache, ids[:, pos:L - 1], pos).past_key_values
        torch.cuda.synchronize(); tcmp = (time.perf_counter() - t0) * 1000
        cl = model(input_ids=ids[:, L - 1:L], past_key_values=cache_slice(cache, 0, L - 1),
                   cache_position=torch.tensor([L - 1], device="cuda")).logits[0, -1].float()
        cd = "deny" if cl[tc] >= cl[tw] else "grant"
        out[N] = {"full": fd, "composed": cd, "agree": fd == cd, "cos": round(torch.cosine_similarity(fl, cl, 0).item(), 3),
                  "full_ms": round(tf, 1), "comp_ms": round(tcmp, 1), "speedup": round(tf / tcmp, 2)}
        print(f"  {N:>7} {fd:>8} {cd:>9} {str(fd==cd):>6} {tf:>8.1f} {tcmp:>8.1f} {tf/tcmp:>7.2f}x", flush=True)
    return out


@torch.no_grad()
def run_seam(model, tok):
    """SEAM-REPAIR (#48): recompute the first K tokens of a transplanted skill WITH the real prefix
    (they attend to it), keep the rest from isolation. Logit cos to full vs K -> repairs the start-seam."""
    KS = [0, 2, 4, 8, 16, 32]
    agg = {k: [] for k in KS}
    for sk in SKILLS:
        ids, a, b = chat_parts(tok, sk["sys"], sk["skill"], sk["task"])
        L = ids.shape[1]; nb = b - a
        fl = model(input_ids=ids.to("cuda")).logits[0, -1].float()
        alone = precompute_chunk(model, ids[:, a:b])
        for K in KS:
            sysc = prefill(model, ids[:, :a])
            if K > 0:                                  # recompute first K skill tokens with real prefix
                sysc = forward_suffix(model, sysc, ids[:, a:a + K], a).past_key_values
            if K < nb:                                 # rest from isolation, repositioned to a+K
                tail = DynamicCache()
                cs, ss = cos_sin(model, list(range(K, nb))); ct, st = cos_sin(model, list(range(a + K, a + nb)))
                for i, l in enumerate(alone.layers):
                    kk = l.keys[:, :, K:nb, :].float()
                    raw = kk * cs - rotate_half(kk) * ss
                    tail.update((raw * ct + rotate_half(raw) * st).to(l.keys.dtype), l.values[:, :, K:nb, :], i)
                cache = cache_concat(cache_slice(sysc, 0, a + K), tail)
            else:
                cache = cache_slice(sysc, 0, a + nb)
            out = forward_suffix(model, cache, ids[:, b:L], b)
            sl = out.logits[0, -1].float()
            agg[K].append(torch.cosine_similarity(fl, sl, 0).item())
    print(f"  seam-repair: logit cos to full vs #recomputed-start-tokens K")
    for K in KS:
        print(f"    K={K:>3}: cos={sum(agg[K])/len(agg[K]):.4f}", flush=True)
    return {K: round(sum(agg[K]) / len(agg[K]), 4) for K in KS}


@torch.no_grad()
def run_reason(model, tok, Ksamp=4, max_new=300):
    """C3: correctness under REASONING. Generate a CoT from full vs precompiled cache, read the decision."""
    import torch as T
    corr = {"full": 0, "precompiled": 0, "agree": 0, "n": 0}
    for sk in SKILLS:
        ids, a, b = chat_parts(tok, sk["sys"], sk["skill"], sk["task"])
        L = ids.shape[1]; nb = b - a
        full_cache = prefill(model, ids[:, :L])
        alone = precompute_chunk(model, ids[:, a:b])
        pre = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, alone, nb, a))
        pre = forward_suffix(model, pre, ids[:, b:L], b).past_key_values
        for s in range(Ksamp):
            for name, cache in [("full", cache_slice(full_cache, 0, L)), ("precompiled", cache_slice(pre, 0, L))]:
                g = T.Generator(device="cuda"); g.manual_seed(s + 1)
                cur = int(ids[0, L - 1]); pos = L; gen = []
                cache2 = cache_slice(cache, 0, L - 1)
                cur = int(ids[0, L - 1]); pos = L - 1
                eos = tok.eos_token_id
                for _ in range(max_new):
                    o = model(input_ids=T.tensor([[cur]], device="cuda"), past_key_values=cache2,
                              cache_position=T.tensor([pos], device="cuda"), use_cache=True); pos += 1
                    p = T.softmax(o.logits[0, -1].float() / 0.7, -1); cur = int(T.multinomial(p, 1, generator=g)); gen.append(cur)
                    if cur == eos or "</think>" in tok.decode(gen[-12:]):
                        break
                ans = []
                for _ in range(16):
                    o = model(input_ids=T.tensor([[cur]], device="cuda"), past_key_values=cache2,
                              cache_position=T.tensor([pos], device="cuda"), use_cache=True); pos += 1
                    cur = int(o.logits[0, -1].argmax()); ans.append(cur)
                    if cur == eos:
                        break
                txt = tok.decode(ans).lower()
                ci = txt.find(sk["correct"]); wi = txt.find(sk["wrong"])
                dec = "correct" if (ci >= 0 and (wi < 0 or ci < wi)) else ("wrong" if wi >= 0 else "other")
                if name == "full":
                    fdec = dec; corr["full"] += (dec == "correct")
                else:
                    corr["precompiled"] += (dec == "correct"); corr["agree"] += (dec == fdec)
            corr["n"] += 1
        print(f"  {sk['name']}: done", flush=True)
    print(f"  [reasoning] full_correct={corr['full']}/{corr['n']} precompiled_correct={corr['precompiled']}/{corr['n']} "
          f"agree={corr['agree']}/{corr['n']}")
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--seam", action="store_true")
    ap.add_argument("--reason", action="store_true")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--staleness", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    if args.sanity:
        sanity(model, tok)
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    if args.experiment:
        print(f"=== SKILL feasibility ({args.model}) ===")
        run_experiment(model, tok)
    if args.staleness:
        print(f"=== CONTEXT-STALENESS ({args.model}): self-contained vs context-coupled skills ===")
        sc = run_experiment(model, tok, SKILLS, "self-contained")
        cp = run_experiment(model, tok, COUPLED, "context-coupled")
        json.dump({"model": args.model, "self_contained": {k: sc[k] for k in ("full","precompiled","agree","n","mean_cos")},
                   "context_coupled": {k: cp[k] for k in ("full","precompiled","agree","n","mean_cos")}},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_staleness_{args.tag or 'm'}.json"), "w"), indent=2)
    if args.seam:
        print(f"=== SEAM-REPAIR ({args.model}) ===")
        sr = run_seam(model, tok)
        json.dump({"model": args.model, "seam_repair": sr},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_seam_{args.tag or 'm'}.json"), "w"), indent=2)
    if args.reason:
        print(f"=== REASONING CORRECTNESS ({args.model}) ===")
        rc = run_reason(model, tok)
        json.dump({"model": args.model, "reason": rc},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_reason_{args.tag or 'm'}.json"), "w"), indent=2)
    if args.multi:
        print(f"=== MULTI-SKILL LIBRARY composition ({args.model}) ===")
        ms = run_multiskill(model, tok)
        json.dump({"model": args.model, "multiskill": ms},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_multi_{args.tag or 'm'}.json"), "w"), indent=2)
    if args.scaling:
        print(f"=== TTFT scaling: full vs precompiled ({args.model}) ===")
        sc = run_scaling(model, tok)
        json.dump({"model": args.model, "scaling": sc},
                  open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_scaling_{tag}.json"), "w"), indent=2)
    print("COMPOSABLE_KV_DONE")


if __name__ == "__main__":
    main()
