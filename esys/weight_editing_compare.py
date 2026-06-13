"""Empirical comparison: KV editing vs WEIGHT editing (ROME, LoRA fine-tune) for mutable state.

A reviewer asks: to make the model act on a changed field, why not edit the weights (ROME/MEMIT)
or fine-tune, instead of editing the KV cache? This experiment answers it on the paper's own
gated decision task ("cancel order ONLY IF order_status == pending; else deny"). The world has
changed (status pending -> shipped) so the correct decision flips cancel -> deny, but the cached
context still shows the OLD value. Each method tries to make the model decide DENY.

We hold the baselines to a fair bar (each is given a target it CAN achieve) and compare on the
four axes that decide deployability for *per-request, per-turn mutable state*:

  (A) EFFICACY        — does the edit flip THIS order's decision to deny?
  (B) EDIT LATENCY    — wall-clock to perform one edit (the per-turn cost in a live agent).
  (C) ISOLATION       — a weight edit is GLOBAL: other concurrent orders that are genuinely
                        still pending (correct = cancel) are evaluated on the same edited model.
                        Fraction wrongly flipped to deny = cross-request contamination. KV edits
                        live in a per-sequence cache, so contamination is structurally zero.
  (D) COLLATERAL      — fraction of a battery of UNRELATED gated decisions whose argmax changes
                        after the edit (locality / specificity).

Methods: kv_erratum, kv_inplace (editkv); rome (rank-one, esys/rome.py); lora_ft (peft).
Run: python esys/weight_editing_compare.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "editkv"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, AutoModelForCausalLM
import rome
from editkv import EditableContext, Mode

NEUTRAL = "\n".join(f"- Clause {i}: routine, non-binding background guidance for agents." for i in range(20))


def gate_prompt(tok, oid, status, rule_status="pending"):
    body = (
        "You are an order-processing assistant. Follow the binding rule exactly.\n\n"
        f"RULE (binding): Cancel order {oid} ONLY IF order_status equals \"{rule_status}\". "
        "For any other order_status you MUST deny.\n\n"
        f"REFERENCE NOTES (not binding)\n{NEUTRAL}\n\n"
        f"order_status: {status}\n\n"
        f"Request: Please cancel order {oid}.\n"
        "Answer with exactly one word: cancel or deny.")
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                   add_generation_prompt=True) + "Decision:"


# unrelated single-token gated decisions for the collateral battery
COLLATERAL = [
    ("If clearance is \"green\" then allow, otherwise block. clearance: green. Answer allow or block.", " allow", " block"),
    ("If clearance is \"green\" then allow, otherwise block. clearance: red. Answer allow or block.", " block", " allow"),
    ("A user gets a refund ONLY IF tier is \"gold\". tier: gold. Eligible? Answer yes or no.", " yes", " no"),
    ("A user gets a refund ONLY IF tier is \"gold\". tier: bronze. Eligible? Answer yes or no.", " no", " yes"),
    ("Ship overnight ONLY IF plan is \"prime\". plan: prime. Answer ship or hold.", " ship", " hold"),
    ("Ship overnight ONLY IF plan is \"prime\". plan: basic. Answer ship or hold.", " hold", " ship"),
    ("Unlock the door ONLY IF badge is \"valid\". badge: valid. Answer unlock or reject.", " unlock", " reject"),
    ("Unlock the door ONLY IF badge is \"valid\". badge: expired. Answer unlock or reject.", " reject", " unlock"),
    ("Approve the loan ONLY IF score is \"high\". score: high. Answer approve or decline.", " approve", " decline"),
    ("Approve the loan ONLY IF score is \"high\". score: low. Answer approve or decline.", " decline", " approve"),
]


def tid(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def decide(model, tok, prompt, a_id, b_id):
    """Return (argmax_is_a, logit_gap a-b) for a forced-decision prompt."""
    ids = torch.tensor([tok(prompt, add_special_tokens=True)["input_ids"]]).cuda()
    lg = model(ids).logits[0, -1].float()
    return (lg[a_id] > lg[b_id]).item(), float(lg[a_id] - lg[b_id])


@torch.no_grad()
def collateral_signature(model, tok):
    sig = []
    for body, corr, other in COLLATERAL:
        p = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                    add_generation_prompt=True) + "Answer:"
        a, _ = decide(model, tok, p, tid(tok, corr), tid(tok, other))
        sig.append(a)
    return sig


def collateral_drift(model, tok, base_sig):
    new = collateral_signature(model, tok)
    return sum(1 for a, b in zip(base_sig, new) if a != b) / len(base_sig)


def run_lora(model, tok, edit_prompt, deny_id, steps=30, lr=1e-4):
    """Fine-tune LoRA adapters so the stale-context prompt decides deny. Returns latency."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "down_proj"],
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.train()
    ids = torch.tensor([tok(edit_prompt, add_special_tokens=True)["input_ids"]]).cuda()
    tgt = torch.tensor([deny_id]).cuda()
    opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        lg = pm(ids).logits[0, -1].float()
        loss = torch.nn.functional.cross_entropy(lg.unsqueeze(0), tgt)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pm.eval()
    return pm, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--target_oid", default="W2378156")
    ap.add_argument("--other_oids", default="W1100001,W1100002,W1100003,W1100004,W1100005,W1100006,W1100007,W1100008")
    ap.add_argument("--rome_layer", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", dtype=torch.bfloat16).eval()
    L = args.rome_layer if args.rome_layer >= 0 else model.config.num_hidden_layers // 5
    cancel_id, deny_id = tid(tok, " cancel"), tid(tok, " deny")
    others = args.other_oids.split(",")

    # --- the scenario: world changed pending->shipped; cached context still shows pending ---
    stale = gate_prompt(tok, args.target_oid, "pending")    # stale context (correct now = deny)
    base_sig = collateral_signature(model, tok)
    # sanity: clean model on stale context decides cancel (the stale conclusion)
    pre_cancel, pre_gap = decide(model, tok, stale, cancel_id, deny_id)
    # other orders genuinely still pending -> correct = cancel; clean model should say cancel
    def isolation_rate():  # fraction of other (truly-pending) orders WRONGLY flipped to deny
        flips = 0
        for o in others:
            is_cancel, _ = decide(model, tok, gate_prompt(tok, o, "pending"), cancel_id, deny_id)
            flips += (0 if is_cancel else 1)
        return flips / len(others)
    pre_iso = isolation_rate()
    results = {"model": args.model, "rome_layer": L,
               "pre_edit": {"target_decides_cancel": bool(pre_cancel), "gap": round(pre_gap, 2),
                            "other_pending_wrong_deny_rate": round(pre_iso, 3)}}
    rows = {}

    # ---------- KV methods (editkv) ----------
    # Build the chat-templated prompt, then split it around the field value so the
    # EditableContext sees the SAME tokens as `stale` (template included).
    FIELD_RENDER = "order_status: pending"
    templated = gate_prompt(tok, args.target_oid, "pending")   # ends with "Decision:"
    assert FIELD_RENDER in templated
    pre, post = templated.split(FIELD_RENDER, 1)               # post ends with "...deny.\n...Decision:"
    post_body, dp = post.rsplit("Decision:", 1)[0], "Decision:"
    for mname, mode in [("kv_erratum", Mode.ERRATUM), ("kv_inplace", Mode.IN_PLACE)]:
        ctx = EditableContext(model, tok)
        ctx.add_text(pre)
        ctx.add_field("order_status", "pending", label="order_status")
        ctx.add_text(post_body)
        ctx.prefill()
        t0 = time.perf_counter()
        cache, last, pos = ctx.build_cache("order_status", "shipped", mode=mode, decision_prompt=dp)
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([[last]], device="cuda"), past_key_values=cache,
                       cache_position=torch.tensor([pos], device="cuda")).logits[0, -1].float()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        rows[mname] = {"efficacy_deny": bool(lg[deny_id] > lg[cancel_id]), "gap_deny_cancel": round(float(lg[deny_id] - lg[cancel_id]), 2),
                       "latency_ms": round(1000 * dt, 1),
                       "isolation_contamination": 0.0,   # per-sequence cache: other requests untouched (structural)
                       "collateral_drift": 0.0}          # weights unchanged
        print(f"[{mname}] deny={rows[mname]['efficacy_deny']} gap={rows[mname]['gap_deny_cancel']} "
              f"lat={rows[mname]['latency_ms']}ms iso=0 collateral=0", flush=True)

    # ---------- ROME (rank-one weight edit) ----------
    print("[rome] estimating covariance...", flush=True)
    corpus = [f"This is example sentence {i}: people, places, products and ordinary daily events." for i in range(300)]
    t_cov = time.perf_counter()
    C, ntok = rome.estimate_cov(model, tok, L, corpus, max_tokens=20000)
    cov_s = time.perf_counter() - t_cov
    # edit so the STALE-context (status still pending) decision becomes deny for the target order
    k = rome.compute_k_star(model, tok, L, [stale, gate_prompt(tok, args.target_oid, "pending")])
    t0 = time.perf_counter()
    v = rome.compute_v_star(model, tok, L, stale, deny_id)
    orig = rome.apply_rome(model, L, k, v, C)
    torch.cuda.synchronize(); rome_edit_s = time.perf_counter() - t0
    r_cancel, r_gap = decide(model, tok, stale, cancel_id, deny_id)
    rome_iso = isolation_rate()
    rome_collat = collateral_drift(model, tok, base_sig)
    rows["rome"] = {"efficacy_deny": (not bool(r_cancel)), "gap_deny_cancel": round(-r_gap, 2),
                    "latency_ms": round(1000 * rome_edit_s, 1), "cov_estimate_s": round(cov_s, 1),
                    "isolation_contamination": round(rome_iso - pre_iso, 3),
                    "collateral_drift": round(rome_collat, 3)}
    print(f"[rome] deny={rows['rome']['efficacy_deny']} gap={rows['rome']['gap_deny_cancel']} "
          f"lat={rows['rome']['latency_ms']}ms (+cov {cov_s:.1f}s) iso={rows['rome']['isolation_contamination']} "
          f"collateral={rows['rome']['collateral_drift']}", flush=True)
    rome.restore(model, L, orig)

    # ---------- LoRA fine-tune ----------
    print("[lora] fine-tuning...", flush=True)
    try:
        pm, lora_s = run_lora(model, tok, stale, deny_id)
        l_cancel, l_gap = decide(pm, tok, stale, cancel_id, deny_id)
        lora_iso_flips = 0
        for o in others:
            isc, _ = decide(pm, tok, gate_prompt(tok, o, "pending"), cancel_id, deny_id)
            lora_iso_flips += (0 if isc else 1)
        lora_iso = lora_iso_flips / len(others)
        lora_collat = collateral_drift(pm, tok, base_sig)
        rows["lora_ft"] = {"efficacy_deny": (not bool(l_cancel)), "gap_deny_cancel": round(-l_gap, 2),
                           "latency_ms": round(1000 * lora_s, 1),
                           "isolation_contamination": round(lora_iso - pre_iso, 3),
                           "collateral_drift": round(lora_collat, 3)}
        print(f"[lora] deny={rows['lora_ft']['efficacy_deny']} gap={rows['lora_ft']['gap_deny_cancel']} "
              f"lat={rows['lora_ft']['latency_ms']}ms iso={rows['lora_ft']['isolation_contamination']} "
              f"collateral={rows['lora_ft']['collateral_drift']}", flush=True)
        pm = pm.unload() if hasattr(pm, "unload") else pm
    except Exception as e:
        rows["lora_ft"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[lora] failed: {e}", flush=True)

    results["methods"] = rows
    path = os.path.join(os.path.dirname(__file__), "..", "results", f"weight_edit_compare_{args.tag}.json")
    json.dump(results, open(path, "w"), indent=2)
    print("WEIGHT_EDIT_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
