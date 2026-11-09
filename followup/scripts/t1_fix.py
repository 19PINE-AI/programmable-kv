"""Repair MiMo outputs: relabel neurons whose label JSON failed to parse (variant prompt so the cache
is bypassed), re-simulate those neurons (general + task), and retry NaN simulation items."""
import json, os, sys
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_mimo as M

OUT, TAG = M.OUT, M.TAG
labels = json.load(open(os.path.join(OUT, f"labels_{TAG}.json")))
st = torch.load(os.path.join(OUT, f"corpus_stats_{TAG}.pt"), weights_only=False)
T = np.load(os.path.join(OUT, f"corpus_tokens_{TAG}.npy"))
sl = st["seqlen"]
bad = [x["rank"] for x in labels if x.get("label") == "PARSE_FAIL"]
print("bad labels:", bad, flush=True)

NUDGES = ["\n\nIMPORTANT: output strictly valid JSON (escape any double quotes inside strings, no trailing commas, keep every field under 400 characters).",
          "\n\nReturn ONLY a compact, valid JSON object. Do not quote example tokens with double quotes; use single quotes. Each field at most 300 characters.",
          "\n\nKeep it short. Valid JSON only."]


def relabel(j):
    L, i = st["neurons"][j]
    maxv = float(st["top_v"][j, 0]); minv = float(st["bot_v"][j, 0])
    pos = M.dedupe(st["top_v"][j].tolist(), st["top_i"][j].tolist(), sl, 20)
    neg = M.dedupe(st["bot_v"][j].tolist(), st["bot_i"][j].tolist(), sl, 15)
    sc = lambda v: (10 * v / maxv) if v >= 0 else (10 * v / abs(minv) if minv < 0 else 0)
    ptxt = "\n".join(f"[{sc(v):+.1f}] {M.context_str(T, sl, g)}" for v, g in pos)
    ntxt = "\n".join(f"[{sc(v):+.1f}] {M.context_str(T, sl, g)}" for v, g in neg if v < 0)
    base = M.LABEL_PROMPT.format(mean=float(st["mean"][j]), std=float(st["std"][j]), fpos=float(st["frac_pos"][j]),
                                 maxv=maxv, minv=minv, pos=ptxt, neg=ntxt or "(no negative activations)")
    for nd in NUDGES:
        txt = M.chat(base + nd)
        try:
            d = M.parse_json(txt)
            if isinstance(d, dict) and d.get("label"):
                return d
        except Exception:
            pass
    return None


for j in bad:
    d = relabel(j)
    if d:
        keep = {k: labels[j][k] for k in ("layer", "idx", "rank", "kind", "max", "min", "examples_pos", "examples_neg")}
        d.update(keep); d["relabeled"] = True
        labels[j] = d
        print(f"  relabeled {j}: {d['label']}", flush=True)
    else:
        print(f"  relabel FAILED {j}", flush=True)
json.dump(labels, open(os.path.join(OUT, f"labels_{TAG}.json"), "w"), indent=1, ensure_ascii=False)

# general sims: redo for relabeled neurons + NaN items
sg = json.load(open(os.path.join(OUT, f"sim_general_{TAG}.json")))
win = sg["win"]


def resim(it, nudge=""):
    ids = T[it["seq"], it["start"]:it["start"] + win].tolist()
    lab = labels[it["neuron"]]
    toks = "\n".join(f"{k}: {json.dumps(M.piece(t), ensure_ascii=False)}" for k, t in enumerate(ids))
    prompt = M.SIM_PROMPT.format(label=lab["label"], pp=lab.get("positive_pole", ""), np_=lab.get("negative_pole", ""),
                                 guide=lab.get("simulation_guide", ""), toks=toks, n=len(ids)) + nudge
    try:
        arr = [float(x) for x in M.parse_json(M.chat(prompt))]
    except Exception:
        arr = []
    if len(arr) != len(ids):
        arr = (arr + [0.0] * len(ids))[:len(ids)] if arr else [float("nan")] * len(ids)
    return arr


todo = [k for k, it in enumerate(sg["items"]) if it["neuron"] in bad or any(v != v for v in it["sim"])]
print("general items to redo:", len(todo), flush=True)


def redo_g(k):
    it = sg["items"][k]
    for nd in ("", "\n(Output exactly the JSON array.)", "\n(Return a JSON array of integers only, no prose.)"):
        arr = resim(it, nd)
        if not any(v != v for v in arr):
            return k, arr
    return k, arr


with ThreadPoolExecutor(8) as ex:
    for k, arr in ex.map(redo_g, todo):
        sg["items"][k]["sim"] = arr
json.dump(sg, open(os.path.join(OUT, f"sim_general_{TAG}.json"), "w"))
print("general NaN items left:", sum(any(v != v for v in it["sim"]) for it in sg["items"]), flush=True)

# task sims for relabeled neurons
tp = json.load(open(os.path.join(OUT, f"task_prompts_{TAG}.json")))
stask = json.load(open(os.path.join(OUT, f"sim_task_{TAG}.json")))
tk = [k for k, r in enumerate(stask) if r["neuron"] in bad]


def redo_t(k):
    r = stask[k]
    rec = tp[r["inst"]]; ids = rec[r["which"] + "_ids"]; agg = rec["agg8"]
    text = ""
    for q, t in enumerate(ids):
        pc = M.tok.decode([t])
        text += f"⟦A{agg.index(q)+1}:{pc}⟧" if q in agg else pc
    lab = labels[r["neuron"]]
    keys = ", ".join(f"A{a+1}" for a in range(len(agg)))
    prompt = M.TASK_PROMPT.format(label=lab["label"], pp=lab.get("positive_pole", ""), np_=lab.get("negative_pole", ""),
                                  guide=lab.get("simulation_guide", ""), text=text, keys=keys)
    try:
        d = M.parse_json(M.chat(prompt))
        return k, [float(d.get(f"A{a+1}", 0)) for a in range(len(agg))]
    except Exception:
        return k, [float("nan")] * len(agg)


with ThreadPoolExecutor(8) as ex:
    for k, v in ex.map(redo_t, tk):
        stask[k]["sim"] = v
json.dump(stask, open(os.path.join(OUT, f"sim_task_{TAG}.json"), "w"))
print("task items redone:", len(tk), "NaN left:", sum(any(v != v for v in r["sim"]) for r in stask), flush=True)
print("T1_FIX_DONE", flush=True)
