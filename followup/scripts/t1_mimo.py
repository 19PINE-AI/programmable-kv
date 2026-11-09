"""T1 steps 4-5 (LLM side): label neurons essay-style with MiMo Pro, then simulate activations.

  python t1_mimo.py label     -> labels_<tag>.json
  python t1_mimo.py simgen    -> sim_general_<tag>.json   (held-out fineweb snippets)
  python t1_mimo.py simtask   -> sim_task_<tag>.json      (task prompts, marked aggregator tokens)

Calls are split across the two MiMo boxes (max 8 concurrent each). Results are cached per
request key so the script is resumable.
"""
import json, os, re, sys, random, threading, time, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer

OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
TAG = os.environ.get("TAG", "llama31_8b")
MODEL_ID = os.environ.get("MODEL", "unsloth/Llama-3.1-8B-Instruct")
env = dict(l.strip().split("=", 1) for l in open(os.path.expanduser(os.environ.get("MIMO_ENV", "~/clean-tests/mimo.env"))) if "=" in l)
URLS = env["MIMO_URLS"].split(",")
KEYS = [env["MIMO_KEY_2"], env["MIMO_KEY_3"]]
MM = env["MIMO_MODEL"]
CLIENTS = [OpenAI(base_url=u, api_key=k, timeout=600) for u, k in zip(URLS, KEYS)]
SEMS = [threading.Semaphore(4) for _ in CLIENTS]   # lead: max 4 concurrent per box
CACHE_F = os.path.join(OUT, f"mimo_cache_{TAG}.jsonl")
_cache = {}
if os.path.exists(CACHE_F):
    for l in open(CACHE_F):
        try:
            d = json.loads(l); _cache[d["k"]] = d["v"]
        except Exception:
            pass
_lock = threading.Lock()
_rr = [0]


def chat(prompt, max_tokens=4000):
    k = hashlib.sha1(prompt.encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    for attempt in range(6):
        with _lock:
            i = _rr[0] % len(CLIENTS); _rr[0] += 1
        try:
            with SEMS[i]:
                r = CLIENTS[i].chat.completions.create(
                    model=MM, messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.0)
            v = r.choices[0].message.content or ""
            with _lock:
                _cache[k] = v
                with open(CACHE_F, "a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
            return v
        except Exception as e:
            print(f"  mimo err box{i+2} attempt {attempt}: {str(e)[:160]}", flush=True)
            time.sleep(5 + 10 * attempt)
    raise RuntimeError("MiMo unreachable")


def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, flags=re.S)
    if m:
        txt = m.group(1)
    s = txt.find("{"); s2 = txt.find("[")
    starts = [x for x in (s, s2) if x >= 0]
    if not starts:
        raise ValueError("no json")
    st = min(starts)
    end = max(txt.rfind("}"), txt.rfind("]"))
    return json.loads(txt[st:end + 1])


tok = AutoTokenizer.from_pretrained(MODEL_ID)


def piece(t):
    s = tok.decode([int(t)])
    return s.replace("\n", "\\n")


def context_str(T, seqlen, gidx, left=28, right=6):
    s, p = divmod(int(gidx), seqlen)
    row = T[s]
    lo, hi = max(1, p - left), min(seqlen, p + right + 1)
    out = ""
    for j in range(lo, hi):
        pc = piece(row[j])
        out += f"«{pc}»" if j == p else pc
    return out


def dedupe(vals, idxs, seqlen, n):
    seen, out = set(), []
    for v, g in zip(vals, idxs):
        s = int(g) // seqlen
        if s in seen:
            continue
        seen.add(s); out.append((float(v), int(g)))
        if len(out) >= n:
            break
    return out


LABEL_PROMPT = """You are an interpretability researcher studying a single MLP neuron (a SwiGLU intermediate unit) in the language model Llama-3.1-8B-Instruct. Its activation is a real number that can be POSITIVE or NEGATIVE (SwiGLU units have two poles). Typical activation on ordinary text: mean {mean:.3f}, std {std:.3f}; it is positive on {fpos:.0%} of tokens.

Below are text excerpts from a general web corpus. In each excerpt, the token where the activation was measured is wrapped in «guillemets»; the number before each excerpt is the activation at that token, scaled so the neuron's strongest positive activation is +10 and strongest negative is -10 (raw max {maxv:.2f}, raw min {minv:.2f}).

STRONGEST POSITIVE ACTIVATIONS:
{pos}

MOST NEGATIVE ACTIVATIONS (the negative pole):
{neg}

Describe what this neuron does. Consider both poles: the negative side often encodes the contrast the network needs. Consider that the neuron may respond to the token itself, to what came before it, or to what it predicts comes next.

Answer ONLY with a JSON object:
{{"label": "<concise label, <=12 words>",
  "positive_pole": "<what drives strong positive activation>",
  "negative_pole": "<what drives strong negative activation, or 'none/weak'>",
  "jobs": "<one of: one_job | two_related_jobs | multiple_unrelated | unclear>",
  "confidence": <0-1>,
  "simulation_guide": "<2-4 sentences: precise rules for predicting this neuron's activation on new text, token by token, including when it is near zero>"}}"""

SIM_PROMPT = """You are simulating a single neuron in a language model from its description. Predict its activation on EVERY token of the text below.

Neuron description:
- label: {label}
- positive pole: {pp}
- negative pole: {np_}
- rules: {guide}

Scale: integer from -10 to 10. +10 = the neuron's strongest positive activation, -10 = its strongest negative activation, 0 = inactive. Most tokens in ordinary text are near 0.

Tokens (index: token):
{toks}

Answer ONLY with a JSON array of {n} integers, one per token in order, nothing else."""

TASK_PROMPT = """You are simulating a single neuron in a language model (Llama-3.1-8B-Instruct) from its description. Below is a full prompt the model read. Some tokens are marked like ⟦A3:token⟧. Predict the neuron's activation at each marked token only (the activation at a token depends on that token and everything BEFORE it).

Neuron description:
- label: {label}
- positive pole: {pp}
- negative pole: {np_}
- rules: {guide}

Scale: integer from -10 to 10. +10 = strongest positive activation, -10 = strongest negative, 0 = inactive.

PROMPT:
<<<
{text}
>>>

Answer ONLY with a JSON object mapping each marker id to an integer, e.g. {{"A1": 0, "A2": 3, ...}} with keys {keys}."""


def do_label():
    st = torch.load(os.path.join(OUT, f"corpus_stats_{TAG}.pt"), weights_only=False)
    T = np.load(os.path.join(OUT, f"corpus_tokens_{TAG}.npy"))
    sl = st["seqlen"]
    neurons = st["neurons"]

    def one(j):
        L, i = neurons[j]
        maxv = float(st["top_v"][j, 0]); minv = float(st["bot_v"][j, 0])
        pos = dedupe(st["top_v"][j].tolist(), st["top_i"][j].tolist(), sl, 20)
        neg = dedupe(st["bot_v"][j].tolist(), st["bot_i"][j].tolist(), sl, 15)
        sc = lambda v: (10 * v / maxv) if v >= 0 else (10 * v / abs(minv) if minv < 0 else 0)
        ptxt = "\n".join(f"[{sc(v):+.1f}] {context_str(T, sl, g)}" for v, g in pos)
        ntxt = "\n".join(f"[{sc(v):+.1f}] {context_str(T, sl, g)}" for v, g in neg if v < 0)
        prompt = LABEL_PROMPT.format(mean=float(st["mean"][j]), std=float(st["std"][j]),
                                     fpos=float(st["frac_pos"][j]), maxv=maxv, minv=minv,
                                     pos=ptxt, neg=ntxt or "(no negative activations)")
        txt = chat(prompt)
        try:
            d = parse_json(txt)
        except Exception:
            d = {"label": "PARSE_FAIL", "raw": txt[:2000]}
        d.update({"layer": L, "idx": i, "rank": j, "kind": st.get("kinds", ["top"] * len(neurons))[j], "max": maxv, "min": minv,
                  "examples_pos": [context_str(T, sl, g) for _, g in pos[:5]],
                  "examples_neg": [context_str(T, sl, g) for v, g in neg[:5] if v < 0]})
        print(f"  L{L}.{i}: {d.get('label')} [{d.get('jobs')}]", flush=True)
        return d

    with ThreadPoolExecutor(16) as ex:
        labels = list(ex.map(one, range(len(neurons))))
    json.dump(labels, open(os.path.join(OUT, f"labels_{TAG}.json"), "w"), indent=1, ensure_ascii=False)


def sim_tokens(lab, ids):
    toks = "\n".join(f"{k}: {json.dumps(piece(t), ensure_ascii=False)}" for k, t in enumerate(ids))
    prompt = SIM_PROMPT.format(label=lab["label"], pp=lab.get("positive_pole", ""), np_=lab.get("negative_pole", ""),
                               guide=lab.get("simulation_guide", ""), toks=toks, n=len(ids))
    txt = chat(prompt)
    try:
        arr = parse_json(txt)
        arr = [float(x) for x in arr]
    except Exception:
        arr = []
    if len(arr) != len(ids):
        arr = (arr + [0.0] * len(ids))[:len(ids)] if arr else [float("nan")] * len(ids)
    return arr


def do_simgen(win=64, n_rand=10, n_top=5, n_bot=5, seed=0):
    st = torch.load(os.path.join(OUT, f"corpus_stats_{TAG}.pt"), weights_only=False)
    T = np.load(os.path.join(OUT, f"corpus_tokens_{TAG}.npy"))
    labels = json.load(open(os.path.join(OUT, f"labels_{TAG}.json")))
    sl, n_lab = st["seqlen"], st["n_lab"]
    H = st["held_acts"].float()          # [n_held_seq, sl, N]
    base = len(T) - H.shape[0]           # held rows start at the first full batch after n_lab (NOT n_lab)
    rng = random.Random(seed)
    nh = H.shape[0]
    # shared random windows (same for every neuron -> enables a joint substitution test)
    shared = []
    for _ in range(n_rand):
        s = rng.randrange(nh); p = rng.randrange(1, sl - win)
        shared.append((s, p))
    jobs = []
    for j, lab in enumerate(labels):
        wins = [("rand", s, p) for s, p in shared]
        a = H[:, 1:, j]
        flat = a.flatten()
        top = torch.argsort(flat, descending=True)
        bot = torch.argsort(flat)
        used = set()
        for kind, order, n in (("top", top, n_top), ("bot", bot, n_bot)):
            c = 0
            for g in order.tolist():
                s, p = divmod(g, sl - 1); p += 1
                if s in used:
                    continue
                if kind == "bot" and float(H[s, p, j]) >= 0:
                    break
                used.add(s)
                start = min(max(1, p - win + 8), sl - win)
                wins.append((kind, s, start)); c += 1
                if c >= n:
                    break
        for kind, s, start in wins:
            jobs.append((j, kind, s, start))

    def one(job):
        j, kind, s, start = job
        ids = T[base + s, start:start + win].tolist()
        sim = sim_tokens(labels[j], ids)
        real = H[s, start:start + win, j].tolist()
        return {"neuron": j, "kind": kind, "seq": int(base + s), "held_row": int(s), "start": int(start), "sim": sim, "real": real}

    print(f"simgen jobs: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    json.dump({"win": win, "shared": shared, "base": int(base), "items": res}, open(os.path.join(OUT, f"sim_general_{TAG}.json"), "w"))


def do_simtask():
    labels = json.load(open(os.path.join(OUT, f"labels_{TAG}.json")))
    tp = json.load(open(os.path.join(OUT, f"task_prompts_{TAG}.json")))   # written by t1_gao.py prep
    jobs = [(j, ti, which) for j in range(len(labels)) for ti in range(len(tp)) for which in ("safe", "unsafe")]

    def one(job):
        j, ti, which = job
        rec = tp[ti]
        ids = rec[which + "_ids"]
        agg = rec["agg8"]
        text = ""
        mk = {}
        for k, t in enumerate(ids):
            pc = tok.decode([t])
            if k in agg:
                m = f"A{agg.index(k)+1}"
                mk[m] = k
                text += f"⟦{m}:{pc}⟧"
            else:
                text += pc
        lab = labels[j]
        keys = ", ".join(sorted(mk, key=lambda x: int(x[1:])))
        prompt = TASK_PROMPT.format(label=lab["label"], pp=lab.get("positive_pole", ""), np_=lab.get("negative_pole", ""),
                                    guide=lab.get("simulation_guide", ""), text=text, keys=keys)
        txt = chat(prompt)
        try:
            d = parse_json(txt)
            vals = [float(d.get(f"A{a+1}", 0)) for a in range(len(agg))]
        except Exception:
            vals = [float("nan")] * len(agg)
        return {"neuron": j, "inst": ti, "which": which, "sim": vals}

    print(f"simtask jobs: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    json.dump(res, open(os.path.join(OUT, f"sim_task_{TAG}.json"), "w"))


if __name__ == "__main__":
    {"label": do_label, "simgen": do_simgen, "simtask": do_simtask}[sys.argv[1]]()
