"""External-validity run: oracle / field_only / erratum x reasoning / non-reasoning
across 8 diverse tasks (domains, field types, decision structures).
Decision = argmax over {correct-first-token, stale-first-token}. Reports P(correct)
and P(stale) per condition, aggregated over tasks (non-reasoning: 1/task; reasoning:
K samples/task), with Wilson CIs.
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from mech_suite import load, clone, prefill, step, ftok, wilson
from align import align_pair, _common_prefix_len
import diverse_tasks as DT
from collections import Counter


def chat(tok, content, thinking, suffix=""):
    msgs = [{"role": "user", "content": content}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:   # non-Qwen templates without the kwarg
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + suffix


def decide2(lg, toi):
    return "correct" if lg[toi["correct"]] >= lg[toi["stale"]] else "stale"


@torch.no_grad()
def gen_decide(model, tok, cache, ids, L, toi, sample=False, seed=0, max_new=1200):
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    c = clone(cache, L - 1); cur = int(ids[0, L - 1]); pos = L - 1; gen = []
    eos = tok.eos_token_id
    for _ in range(max_new):
        out = step(model, c, cur, pos); pos += 1
        if sample:
            p = torch.softmax(out.logits[0, -1].float() / 0.7, -1)
            nx = int(torch.multinomial(p, 1, generator=g))
        else:
            nx = int(out.logits[0, -1].argmax())
        gen.append(nx); cur = nx
        if "</think>" in tok.decode(gen) or nx == eos:
            break
    scaffold = tok("\nDecision:", add_special_tokens=False)["input_ids"]
    for t in [cur] + scaffold[:-1]:
        step(model, c, t, pos); pos += 1
    out = step(model, c, scaffold[-1], pos)
    return decide2(out.logits[0, -1].float(), toi)


def fieldonly(model, oid_ids, nid_ids, a, b, L):
    co, cn = prefill(model, oid_ids), prefill(model, nid_ids)
    fc = clone(co, L)
    for i in range(len(fc.layers)):
        fc.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
        fc.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
    return fc, cn, co


def run_task(model, tok, key, thinking, K):
    t = DT.TASKS[key]
    toi = {"correct": ftok(tok, t["correct"]), "stale": ftok(tok, t["stale"])}
    sfx = "" if thinking else ""    # build() already ends with 'Decision:'
    t_old = chat(tok, DT.build(key, t["vold"]), thinking)
    t_new = chat(tok, DT.build(key, t["vnew"]), thinking)
    t_err = chat(tok, DT.build(key, t["vold"], erratum_value=t["vnew"]), thinking)
    eid = torch.tensor([tok(t_err, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]
    oid = al["old_ids"]; nid = al["new_ids"]; L = al["seq_len"]   # length-aligned
    fc, cn, co = fieldonly(model, oid, nid, a, b, L)
    ew = prefill(model, eid); Le = eid.shape[1]

    if not thinking:
        dec = lambda cache, ids, Lx: decide2(step(model, clone(cache, Lx - 1), int(ids[0, Lx - 1]), Lx - 1).logits[0, -1].float(), toi)
        return {"oracle": [dec(cn, nid, L)], "field_only": [dec(fc, nid, L)], "erratum": [dec(ew, eid, Le)]}
    out = {"oracle": [], "field_only": [], "erratum": []}
    for s in range(K):
        out["oracle"].append(gen_decide(model, tok, cn, nid, L, toi, True, 100 + s))
        out["field_only"].append(gen_decide(model, tok, fc, nid, L, toi, True, 100 + s))
        out["erratum"].append(gen_decide(model, tok, ew, eid, Le, toi, True, 100 + s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--modes", default="both", choices=["both", "nonreasoning", "reasoning"])
    args = ap.parse_args()
    tok, model = load(args.model)
    res = {"model": args.model, "tasks": list(DT.TASKS), "by_mode": {}}
    mode_flags = {"both": [False, True], "nonreasoning": [False], "reasoning": [True]}[args.modes]
    for thinking in mode_flags:
        mode = "reasoning" if thinking else "nonreasoning"
        agg = {c: [] for c in ["oracle", "field_only", "erratum"]}
        per = {}
        for key in DT.TASKS:
            r = run_task(model, tok, key, thinking, args.K)
            per[key] = {c: dict(Counter(r[c])) for c in r}
            for c in agg:
                agg[c].extend(r[c])
            print(f"[{mode}] {key:14s} " + " ".join(
                f"{c}:c{sum(x=='correct' for x in r[c])}/s{sum(x=='stale' for x in r[c])}" for c in agg), flush=True)
        summ = {}
        for c in agg:
            n = len(agg[c]); kc = sum(x == "correct" for x in agg[c])
            summ[c] = {"P_correct": round(kc / n, 2), "ci": wilson(kc, n),
                       "P_stale": round(sum(x == "stale" for x in agg[c]) / n, 2), "n": n}
        res["by_mode"][mode] = {"summary": summ, "per_task": per}
        print(f"=== {mode} (n={summ['oracle']['n']}): oracle P_correct={summ['oracle']['P_correct']}{summ['oracle']['ci']} "
              f"field_only={summ['field_only']['P_correct']}{summ['field_only']['ci']} "
              f"erratum={summ['erratum']['P_correct']}{summ['erratum']['ci']}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_diverse_{args.tag}.json"), "w"), indent=2)
    print("DIVERSE_DONE", flush=True)


if __name__ == "__main__":
    main()
