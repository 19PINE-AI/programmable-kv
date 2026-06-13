"""E5 — end-to-end systems amortization + correctness guard.

Simulates multi-turn sessions and measures per-decision TTFT for three serving strategies,
plus faithfulness of the proposed method vs a full-reprefill oracle:
  * front_reprefill : [sys][MEM][traj]; memory edit invalidates downstream -> reprefill MEM+traj.
  * end_reprefill   : [sys][traj][MEM]; traj grows each turn -> reprefill MEM every decision.
  * proposed        : [sys][traj][MEM]; memory precompiled+repositioned (re-rotate, no reprefill);
                      edit = recompile chunk once (or erratum).
Oracle = full reprefill of the whole sequence each decision (timing + decision reference).

Per decision turn we log ttft_ms for each method and (proposed vs oracle) top1_agree/cos, and a
CoT decision-agreement spot-check on a subset. Writes results/e5_<tag>.jsonl.
"""
import os, sys, json, argparse, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from data import make_persona, filler_trajectory
from app import MemoryAgent, _ids
from memkv import generate_from_cache, parse_final, decide
from composable_kv import (load_lm, prefill, forward_suffix, cache_slice, cache_concat,
                           precompute_chunk, repositioned_chunk_cache, _as_dyn)
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

SYS = "You are a careful account-management assistant. Follow the user settings exactly."


@torch.no_grad()
def time_full_reprefill(model, tok, full_text, query):
    """Oracle: reprefill [full_text][query] from scratch; return (ttft_ms, first_logits, pos, cache, first)."""
    ids = _ids(tok, full_text, special=True)
    q = _ids(tok, "\n" + query)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cache = prefill(model, ids)
    out = forward_suffix(model, cache, q, ids.shape[1])
    first = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize(); ttft = (time.perf_counter() - t0) * 1000
    return ttft, out.logits[0, -1].float(), ids.shape[1] + q.shape[1], out.past_key_values, first


@torch.no_grad()
def time_front(model, tok, sys_t, mem_md, traj_text, query, mem_dirty, front_state):
    """front_reprefill: cache [sys][MEM][traj]; if mem changed, rebuild MEM+traj; else reuse + query."""
    if front_state.get("cache") is None or mem_dirty:
        ids = _ids(tok, sys_t + "\n\n" + mem_md + "\n\n" + traj_text, special=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cache = prefill(model, ids)
        front_state["cache"] = cache; front_state["len"] = ids.shape[1]; front_state["traj"] = traj_text
    else:
        # append only the new trajectory delta
        new = traj_text[len(front_state["traj"]):]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if new:
            d = _ids(tok, new)
            front_state["cache"] = forward_suffix(model, front_state["cache"], d, front_state["len"]).past_key_values
            front_state["len"] += d.shape[1]; front_state["traj"] = traj_text
    q = _ids(tok, "\n" + query)
    out = forward_suffix(model, front_state["cache"], q, front_state["len"])
    first = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize(); ttft = (time.perf_counter() - t0) * 1000
    # roll back the query from the persistent cache (keep [sys][MEM][traj] only)
    front_state["cache"] = cache_slice(front_state["cache"], 0, front_state["len"])
    return ttft, out.logits[0, -1].float()


@torch.no_grad()
def time_end(model, tok, base_cache, base_len, mem_ids, query):
    """end_reprefill: [sys][traj] cached; reprefill MEM after it every decision, then query."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    c = forward_suffix(model, cache_slice(base_cache, 0, base_len), mem_ids, base_len).past_key_values
    pos = base_len + mem_ids.shape[1]
    q = _ids(tok, "\n" + query)
    out = forward_suffix(model, c, q, pos)
    first = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize(); ttft = (time.perf_counter() - t0) * 1000
    return ttft, out.logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--decide_every", type=int, default=3)
    ap.add_argument("--edit_rate", type=float, default=0.25)
    ap.add_argument("--mtotal", type=int, default=120)     # ~2k-token memory
    ap.add_argument("--nfacts", type=int, default=2)
    ap.add_argument("--cot_spotcheck", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    path = os.path.join(os.path.dirname(__file__), "results", f"e5_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time()
    import random
    for s in range(args.sessions):
        rng = random.Random(s)
        p = make_persona(1000 + s, args.mtotal, args.nfacts, gold_yes=(s % 2 == 0))
        cur = p
        agent = MemoryAgent(model, tok, SYS, p.memory_markdown())
        front_state = {}
        traj_text = ""
        # base cache for end_reprefill = [sys][traj]; reuse agent.base by mirroring turns
        decision_idx = 0
        for t in range(args.turns):
            # chit-chat turn
            chat = f"User: let's chat about item {t} in session {s}.\nAssistant: sure, glad to help with that."
            agent.add_turn(chat); traj_text += chat + "\n"
            # maybe a memory edit (tool flips the relevant setting)
            mem_dirty = False
            if rng.random() < args.edit_rate:
                new_enabled = not cur.settings[cur.flip_idx]["enabled"]
                cur = cur.with_toggle(cur.flip_idx, new_enabled)
                flip = p.settings[p.flip_idx]["attr"]; val = "enabled" if new_enabled else "disabled"
                agent.update_memory(cur.memory_markdown(), mode="recompile", label=flip, value=val)
                mem_dirty = True
            # decision turn?
            if (t + 1) % args.decide_every == 0:
                q = cur.decision_query(False)
                gold = "yes" if cur.gold_yes else "no"
                # MATCHED oracle: reprefill the EXACT token stream the proposed cache represents,
                # so the only difference is transplant-vs-full-prefill (no text-construction mismatch).
                exact = agent.exact_ids(q); Le = exact.shape[1]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                o_cache = prefill(model, exact[:, :Le - 1])
                o_out = model(input_ids=exact[:, Le - 1:Le].to("cuda"),
                              past_key_values=cache_slice(o_cache, 0, Le - 1),
                              cache_position=torch.tensor([Le - 1], device="cuda"))
                o_first = int(o_out.logits[0, -1].argmax())
                torch.cuda.synchronize(); o_ttft = (time.perf_counter() - t0) * 1000
                o_logits = o_out.logits[0, -1].float()
                p_res = agent.decide(q)  # proposed (timed inside); returns matched first_logits
                p_logits = p_res["first_logits"]
                f_ttft, f_logits = time_front(model, tok, SYS, cur.memory_markdown(), traj_text, q, mem_dirty, front_state)
                e_ttft, e_logits = time_end(model, tok, agent.base, agent.base_len, agent._mem_ids, q)
                rec = dict(model=args.model, session=s, turn=t, decision_idx=decision_idx,
                           edit_rate=args.edit_rate, mtotal=args.mtotal, L_traj=len(traj_text), L_total=int(Le),
                           ttft_oracle=o_ttft, ttft_front=f_ttft, ttft_end=e_ttft, ttft_proposed=p_res["ttft_ms"],
                           top1_agree=int(int(p_logits.argmax()) == int(o_logits.argmax())),
                           cos=float(F.cosine_similarity(p_logits, o_logits, 0)),
                           dec_proposed=decide(p_logits, tok), dec_oracle=decide(o_logits, tok), gold=gold)
                # CoT spot-check (matched oracle): proposed vs full-reprefill, both from identical tokens
                if args.cot_spotcheck and decision_idx % 3 == 0:
                    pr = agent.decide(q, cot=True)
                    o_txt = generate_from_cache(model, tok, o_cache, o_first, Le - 1, 400)
                    rec["cot_proposed"] = pr["decision"]; rec["cot_oracle"] = parse_final(o_txt)
                    rec["cot_agree"] = int(pr["decision"] == parse_final(o_txt))
                    rec["cot_oracle_correct"] = int(parse_final(o_txt) == gold)
                f.write(json.dumps(rec) + "\n"); f.flush()
                decision_idx += 1
        print(f"  session {s+1}/{args.sessions} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"E5_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
