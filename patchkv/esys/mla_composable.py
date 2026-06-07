"""MLA composable transplant adapter (DeepSeek-V2 / Coder-V2, Multi-head Latent Attention) — RIGOROUS.

MLA caches FULL reconstructed K/V but RoPE is applied ONLY to the last qk_rope_head_dim dims of the key
(k_pe, shared across heads); k_nope and value are position-free. So transplant re-rotates ONLY that
sub-vector — the rest is position-portable for free. We evaluate with the SAME rigor as other models:
10 categorical domains x N instances (>=100 decisions) + bootstrap CIs, measuring composed==full decision
agreement and correctness, plus a logit-cos sanity. Run: MECH_ATTN=eager python esys/mla_composable.py
"""
import argparse, os, sys, json, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import load_lm, prefill, cache_slice, cache_concat, forward_suffix

PAD = "\n".join(f"- SOP note {i}: log the interaction and follow standard procedure." for i in range(40))
# 10 domains: (name, field, rule, [(value, act)]) — the policy governs the action given the field value.
DOMAINS = [
 ("ACCOUNT", "account_status", "Permit a withdrawal ONLY if account_status is active; if frozen, deny.",
  [("active", "allow"), ("frozen", "deny")], "The user requests a withdrawal."),
 ("ORDER", "order_status", "Cancel an order ONLY if order_status is pending; for shipped, deny.",
  [("pending", "cancel"), ("shipped", "deny")], "The user asks to cancel the order."),
 ("PLAN", "plan", "Enable premium features ONLY if plan is pro; if free, block.",
  [("pro", "enable"), ("free", "block")], "The user requests a premium feature."),
 ("DOOR", "badge", "Unlock the door ONLY if badge is valid; if revoked, keep locked.",
  [("valid", "unlock"), ("revoked", "locked")], "The user swipes to enter."),
 ("PAYOUT", "kyc", "Release a payout ONLY if kyc is verified; if pending, hold.",
  [("verified", "release"), ("pending", "hold")], "The user requests a payout."),
 ("MOD", "post_state", "Publish a post ONLY if post_state is approved; if flagged, quarantine.",
  [("approved", "publish"), ("flagged", "quarantine")], "A post is submitted."),
 ("VISA", "passport", "Grant entry ONLY if passport is valid; if expired, refer.",
  [("valid", "grant"), ("expired", "refer")], "A traveler requests entry."),
 ("AUTH", "mfa", "Allow login ONLY if mfa is passed; if failed, challenge.",
  [("passed", "allow"), ("failed", "challenge")], "The user attempts to log in."),
 ("DEPLOY", "ticket", "Allow a production deploy ONLY if ticket is approved; if none, block.",
  [("approved", "allow"), ("none", "block")], "A production deploy is requested."),
 ("SHIP", "stock", "Ship an order ONLY if stock is available; if backordered, defer.",
  [("available", "ship"), ("backordered", "defer")], "An order is ready to ship."),
]
OTHER = {"allow": "deny", "deny": "allow", "cancel": "deny", "enable": "block", "block": "enable",
         "unlock": "locked", "locked": "unlock", "release": "hold", "hold": "release", "publish": "quarantine",
         "quarantine": "publish", "grant": "refer", "refer": "grant", "challenge": "allow", "ship": "defer", "defer": "ship"}


def boot_ci(xs, B=10000, seed=0):
    if not xs:
        return [0.0, 0.0]
    r = random.Random(seed); m = sorted(sum(r.choice(xs) for _ in range(len(xs))) / len(xs) for _ in range(B))
    return [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def mla_cos_sin(rot, positions):
    """YaRN cos/sin for given positions -> [len,64] (mscale baked into cos_cached)."""
    seq = max(positions) + 1
    cos, sin = rot(torch.zeros(1, 1, 1, device="cuda"), seq_len=seq)
    idx = torch.tensor(positions, device="cuda")
    return cos[idx].float(), sin[idx].float()


@torch.no_grad()
def mla_reposition(chunk_cache, rot, src_positions, tgt_positions, rope_dim=64):
    """Re-rotate ONLY key[..., -rope_dim:] (k_pe) from src to tgt positions; k_nope & value unchanged."""
    cs, ss = mla_cos_sin(rot, src_positions); ct, st = mla_cos_sin(rot, tgt_positions)
    cs, ss, ct, st = (t[None, None] for t in (cs, ss, ct, st))   # [1,1,seq,64]
    out = DynamicCache()
    for i, l in enumerate(chunk_cache.layers):
        k = l.keys.float(); kn, kpe = k[..., :-rope_dim], k[..., -rope_dim:]
        raw = kpe * cs - rotate_half(kpe) * ss                   # un-rotate from src
        kpe2 = raw * ct + rotate_half(raw) * st                  # re-rotate to tgt
        out.update(torch.cat([kn, kpe2], -1).to(l.keys.dtype), l.values, i)
    return out


def tid(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def _am(n):
    return torch.ones(1, n, dtype=torch.long, device="cuda")


@torch.no_grad()
def mla_prefill(model, ids):
    """Prefill from scratch with explicit attention_mask + position_ids (DeepSeek-V2 eager requires them)."""
    n = ids.shape[1]
    pkv = model(input_ids=ids.to("cuda"), attention_mask=_am(n),
                position_ids=torch.arange(n, device="cuda")[None], use_cache=True).past_key_values
    return DynamicCache.from_legacy_cache(pkv) if not hasattr(pkv, "layers") else pkv


@torch.no_grad()
def mla_fwd(model, cache, seg_ids, start):
    """forward a segment onto a cache with explicit attention_mask + position_ids."""
    if seg_ids.shape[1] == 0:
        return cache
    pos = torch.arange(start, start + seg_ids.shape[1], device="cuda")[None]
    return model(input_ids=seg_ids, past_key_values=cache, attention_mask=_am(start + seg_ids.shape[1]),
                 position_ids=pos, use_cache=True).past_key_values


@torch.no_grad()
def decide(model, cache, last, pos, tc, to):
    lg = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
               attention_mask=_am(pos + 1), position_ids=torch.tensor([[pos]], device="cuda"),
               use_cache=True).logits[0, -1].float()
    return lg[tc] >= lg[to], lg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat"); ap.add_argument("--tag", default=None)
    ap.add_argument("--per", type=int, default=8)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="eager")
    rot = model.model.layers[0].self_attn.rotary_emb
    rope_dim = model.config.qk_rope_head_dim

    agree, comp_correct, full_correct, coss = [], [], [], []
    print(f"=== MLA composable (RIGOROUS) {args.model} rope_dim={rope_dim} ===", flush=True)
    for (name, field, rule, vals, req) in DOMAINS:
        for j in range(args.per):
            xid = f"{name[:2]}{1000+j*7}"
            for (val, act) in vals:
                oth = OTHER[act]; tc, to = tid(tok, act), tid(tok, oth)
                policy = f"# {name}_POLICY\n{rule}\n{PAD}\nEnd of policy."
                body = (f"You are the {name} agent. Apply the policy.\n\n{policy}\n\n"
                        f"Current {field} is {val}.\n{req} (ref {xid}) Answer one word — {act} or {oth}.\nDecision:")
                full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                ids = torch.tensor([enc["input_ids"]]).to("cuda"); offs = enc["offset_mapping"]; L = ids.shape[1]
                pa = full.find(policy); pb = pa + len(policy)
                a = next(i for i, (lo, hi) in enumerate(offs) if lo <= pa < hi)
                b = next((i for i, (lo, hi) in enumerate(offs) if lo >= pb), len(offs))
                # FULL
                fcache = mla_prefill(model, ids[:, :L - 1])
                fd, flg = decide(model, fcache, ids[0, L - 1], L - 1, tc, to)
                # COMPOSED: precompile policy, MLA-reposition, splice
                chunk = mla_reposition(mla_prefill(model, ids[:, a:b]), rot, list(range(b - a)), list(range(a, b)), rope_dim)
                cache = cache_concat(mla_prefill(model, ids[:, :a]), chunk)
                cache = mla_fwd(model, cache, ids[:, b:L - 1], b)
                cd, clg = decide(model, cache, ids[0, L - 1], L - 1, tc, to)
                agree.append(int(bool(fd) == bool(cd))); comp_correct.append(int(bool(cd))); full_correct.append(int(bool(fd)))
                coss.append(torch.cosine_similarity(flg, clg, dim=0).item())
        print(f"  {name:8s} done | agree~{sum(agree)/len(agree):.3f} cos~{sum(coss)/len(coss):.3f}", flush=True)
    n = len(agree)
    out = {"model": args.model, "rope_dim": rope_dim, "n_decisions": n, "domains": len(DOMAINS),
           "composed_vs_full_agreement": round(sum(agree) / n, 3), "agreement_ci": boot_ci(agree),
           "composed_correct": round(sum(comp_correct) / n, 3), "full_correct": round(sum(full_correct) / n, 3),
           "mean_logit_cos": round(sum(coss) / n, 4)}
    print(f"\n=== MLA composable RIGOROUS {args.model}: {len(DOMAINS)} domains, {n} decisions ===")
    print(f"  composed==full agreement = {out['composed_vs_full_agreement']} CI{out['agreement_ci']}")
    print(f"  composed_correct={out['composed_correct']} full_correct={out['full_correct']} | mean logit cos={out['mean_logit_cos']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"mla_composable_{tag}.json"), "w"), indent=2)
    print("MLA_COMPOSABLE_DONE")


if __name__ == "__main__":
    main()
