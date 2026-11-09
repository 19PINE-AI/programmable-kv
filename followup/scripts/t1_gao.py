"""T1 steps 5-6 (GPU side).

  python t1_gao.py prep   -> task_prompts_<tag>.json (no GPU: token ids + aggregator positions)
  python t1_gao.py eval   -> gao_<tag>.json  (Gao simulation-substitution test)
  python t1_gao.py ov     -> ov_<tag>.json   (do the read heads read these neurons' output directions?)

Simulation substitution (Gao): replace a neuron's real activation by its label-based simulated
activation (MiMo, integer -10..10) mapped to activation units by a per-neuron linear calibration
fit on the general-text simulations; compare the downstream damage (KL to the unmodified model)
with mean-ablation and zero-ablation of the same neuron(s) at the same positions.
  damage_avoided = 1 - KL(sim) / KL(mean-ablation)       (1 = perfect explanation, <=0 = useless)
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("PKV_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "esys"))
sys.path.insert(0, os.path.join(REPO, "e1"))
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
TAG = os.environ.get("TAG", "llama31_8b")
MODEL_ID = os.environ.get("MODEL", "unsloth/Llama-3.1-8B-Instruct")


def prep():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from t1_common import build_pair_f
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    nj = json.load(open(os.path.join(OUT, f"t1_neurons_{TAG}.json")))
    out = []
    for inst in nj["instances"]:
        P = build_pair_f(tok, inst["scn"], inst["oid"], inst["fi"])
        out.append({"scn": inst["scn"], "oid": inst["oid"], "fi": inst["fi"], "agg8": inst["agg8"], "dpos": P["dpos"],
                    "toi": P["toi"], "safe_ids": P["safe_ids"][0].tolist(), "unsafe_ids": P["unsafe_ids"][0].tolist()})
    json.dump(out, open(os.path.join(OUT, f"task_prompts_{TAG}.json"), "w"))
    print("prep ok", len(out))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    return tok, model


class Sub:
    """patch spec: list of (layer, idx, pos LongTensor[P], vals Tensor[P]) applied in batch row b
    (or all rows when b is None) at mlp.down_proj input."""

    def __init__(self, model):
        self.model = model
        self.spec = []
        self.h = [l.mlp.down_proj.register_forward_pre_hook(self._mk(L)) for L, l in enumerate(model.model.layers)]

    def _mk(self, L):
        def hook(mod, a):
            mine = [s for s in self.spec if s[0] == L]
            if not mine:
                return None
            x = a[0].clone()
            for (_, i, b, pos, vals) in mine:
                x[b, pos, i] = vals.to(x.dtype)
            return (x,)
        return hook


def kl_rows(lp_ref, lp):
    """KL(ref||p) per position given log-probs [.., V]."""
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


def evaluate():
    tok, model = load_model()
    S = Sub(model)
    st = torch.load(os.path.join(OUT, f"corpus_stats_{TAG}.pt"), weights_only=False)
    T = np.load(os.path.join(OUT, f"corpus_tokens_{TAG}.npy"))
    labels = json.load(open(os.path.join(OUT, f"labels_{TAG}.json")))
    sg = json.load(open(os.path.join(OUT, f"sim_general_{TAG}.json")))
    stask = json.load(open(os.path.join(OUT, f"sim_task_{TAG}.json")))
    tp = json.load(open(os.path.join(OUT, f"task_prompts_{TAG}.json")))
    neurons = [tuple(x) for x in st["neurons"]]
    N = len(neurons)
    kinds = st.get("kinds", ["top"] * N)
    topi = [j for j in range(N) if kinds[j] == "top"]
    rndi = [j for j in range(N) if kinds[j] == "random"]
    SETS = {"top30": topi[:30], "top100": topi[:100]}
    if rndi:
        SETS["random100"] = rndi[:100]
    cmean = st["mean"].float()
    win = sg["win"]
    items = sg["items"]

    # ---------------- calibration + correlation on general text
    cal, corr = {}, {}
    for j in range(N):
        xs, ys, kinds = [], [], []
        for it in items:
            if it["neuron"] != j:
                continue
            for s_, r_ in zip(it["sim"], it["real"]):
                if s_ == s_:  # not nan
                    xs.append(s_); ys.append(r_); kinds.append(it["kind"])
        xs, ys, kinds = np.array(xs), np.array(ys), np.array(kinds)
        if len(xs) < 10 or xs.std() == 0:
            a, b = 0.0, float(cmean[j])
        else:
            a, b = np.polyfit(xs, ys, 1)
        cal[j] = (float(a), float(b))

        def pr(m):
            if m.sum() < 5 or xs[m].std() == 0 or ys[m].std() == 0:
                return None
            return float(np.corrcoef(xs[m], ys[m])[0, 1])
        corr[j] = {"all": pr(np.ones_like(xs, bool)), "random_only": pr(kinds == "rand"),
                   "top_bot": pr(kinds != "rand"), "n_tok": int(len(xs)),
                   "frac_nan_items": float(np.mean([any(v != v for v in it["sim"]) for it in items if it["neuron"] == j]))}

    # ---------------- general-text substitution KL (per neuron)
    @torch.no_grad()
    def run_rows(ids_rows, spec):
        S.spec = spec
        x = torch.tensor(ids_rows, device="cuda")
        lg = model(input_ids=x, use_cache=False).logits.float()
        S.spec = []
        return F.log_softmax(lg, -1)

    gen = {}
    for j in range(N):
        L, i = neurons[j]
        its = [it for it in items if it["neuron"] == j and not any(v != v for v in it["sim"])]
        if not its:
            continue
        rows = [T[it["seq"]].tolist() for it in its]
        a, b = cal[j]
        res = {}
        for mode in ("orig", "sim", "mean", "zero", "sim_nominal"):
            spec = []
            for r, it in enumerate(its):
                pos = torch.arange(it["start"], it["start"] + win, device="cuda")
                if mode == "orig":
                    continue
                if mode == "sim":
                    v = torch.tensor([a * s_ + b for s_ in it["sim"]], device="cuda")
                elif mode == "sim_nominal":
                    mx, mn = float(st["top_v"][j, 0]), float(st["bot_v"][j, 0])
                    v = torch.tensor([s_ / 10 * mx if s_ >= 0 else s_ / 10 * abs(mn) for s_ in it["sim"]], device="cuda")
                elif mode == "mean":
                    v = torch.full((win,), float(cmean[j]), device="cuda")
                else:
                    v = torch.zeros(win, device="cuda")
                spec.append((L, i, r, pos, v))
            lp = []
            for c in range(0, len(rows), 10):
                sub = [(LL, ii, rr - c, pp, vv) for (LL, ii, rr, pp, vv) in spec if c <= rr < c + 10]
                lp.append(run_rows(rows[c:c + 10], sub).cpu())
            res[mode] = torch.cat(lp)
        kls = {}
        for mode in ("sim", "mean", "zero", "sim_nominal"):
            per = []
            for r, it in enumerate(its):
                st_ = it["start"]
                per.append(float(kl_rows(res["orig"][r, st_:], res[mode][r, st_:]).sum()))
            kls[mode] = per
        kinds = [it["kind"] for it in its]

        def agg(mask):
            m = [k for k, x in zip(range(len(its)), mask) if x]
            if not m:
                return None
            ks = {md: float(np.sum([kls[md][k] for k in m])) for md in kls}
            return {"KL": ks, "damage_avoided_vs_mean": 1 - ks["sim"] / ks["mean"] if ks["mean"] > 0 else None,
                    "damage_avoided_vs_zero": 1 - ks["sim"] / ks["zero"] if ks["zero"] > 0 else None,
                    "nominal_damage_avoided_vs_mean": 1 - ks["sim_nominal"] / ks["mean"] if ks["mean"] > 0 else None}
        gen[j] = {"all": agg([True] * len(its)), "random": agg([k == "rand" for k in kinds]),
                  "top_bot": agg([k != "rand" for k in kinds])}
        print(f"  gen L{L}.{i} r={corr[j]['all']} DA_mean={gen[j]['all']['damage_avoided_vs_mean']}", flush=True)
        del res
        torch.cuda.empty_cache()

    # joint general-text: all neurons on the shared random windows
    shared = sg["shared"]
    n_lab = sg["base"]          # first held row in T (held_acts row 0)
    rows = [T[n_lab + s].tolist() for s, p in shared]
    joint = {}
    simmap = {}
    for it in items:
        if it["kind"] == "rand":
            simmap[(it["neuron"], it["seq"], it["start"])] = it["sim"]
    joint_res = {}
    for setname, members in SETS.items():
      joint = {}
      for mode in ("orig", "sim", "mean", "zero"):
        spec = []
        if mode != "orig":
            for r, (s, p) in enumerate(shared):
                pos = torch.arange(p, p + win, device="cuda")
                for j in members:
                    L, i = neurons[j]
                    if mode == "sim":
                        sm = simmap.get((j, n_lab + s, p))
                        if sm is None or any(v != v for v in sm):
                            v = torch.full((win,), float(cmean[j]), device="cuda")
                        else:
                            a, b = cal[j]; v = torch.tensor([a * x + b for x in sm], device="cuda")
                    elif mode == "mean":
                        v = torch.full((win,), float(cmean[j]), device="cuda")
                    else:
                        v = torch.zeros(win, device="cuda")
                    spec.append((L, i, r, pos, v))
        joint[mode] = run_rows(rows, spec).cpu()
      jk = {md: float(sum(kl_rows(joint["orig"][r, p:], joint[md][r, p:]).sum() for r, (s, p) in enumerate(shared)))
            for md in ("sim", "mean", "zero")}
      joint_res[setname] = {"KL": jk, "damage_avoided_vs_mean": 1 - jk["sim"] / jk["mean"],
                            "damage_avoided_vs_zero": 1 - jk["sim"] / jk["zero"]}
      print("joint general", setname, joint_res[setname], flush=True)

    # ---------------- task: substitution at aggregator positions
    simt = {}
    for r in stask:
        simt[(r["neuron"], r["inst"], r["which"])] = r["sim"]

    @torch.no_grad()
    def decide(ids, spec):
        S.spec = spec
        o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        S.spec = []
        return o.logits[0, -1].float()

    @torch.no_grad()
    def capture(ids, which_neurons, pos):
        cap = {}
        hs = []
        for L in set(L for L, _ in which_neurons):
            idx = torch.tensor([i for LL, i in which_neurons if LL == L], device="cuda")
            def mk(L, idx):
                def hook(mod, a):
                    cap[L] = (idx, a[0][0][pos][:, idx].float())
                return hook
            hs.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L, idx)))
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        for h in hs:
            h.remove()
        out = {}
        for L, (idx, v) in cap.items():
            for c, i in enumerate(idx.tolist()):
                out[(L, i)] = v[:, c]
        return out

    task = {"per_instance": [], "single": {}}
    real_task_acts = {}
    single_acc = {j: {"kl_sim": [], "kl_mean": [], "kl_zero": []} for j in range(N)}
    corr_task = {j: {"sim": [], "real": [], "dsim": [], "dreal": []} for j in range(N)}
    for ti, rec in enumerate(tp):
        pos = torch.tensor(rec["agg8"], device="cuda")
        toi = rec["toi"]
        reals = {w: capture(rec[w + "_ids"], neurons, pos) for w in ("safe", "unsafe")}
        lg_real = {w: decide(rec[w + "_ids"], []) for w in ("safe", "unsafe")}
        Sreal = {w: float(lg_real[w][toi["safe"]] - lg_real[w][toi["unsafe"]]) for w in lg_real}
        denom = Sreal["safe"] - Sreal["unsafe"]

        def simvals(j, w):
            a, b = cal[j]
            sv = simt.get((j, ti, w))
            if sv is None or any(v != v for v in sv):
                return torch.full((len(pos),), float(cmean[j]), device="cuda")
            return torch.tensor([a * x + b for x in sv], device="cuda")

        for j in range(N):
            for w in ("safe", "unsafe"):
                corr_task[j]["sim"] += simt.get((j, ti, w), [np.nan] * len(pos))
                corr_task[j]["real"] += reals[w][neurons[j]].tolist()
            ds = np.array(simt.get((j, ti, "safe"), [np.nan] * 8)) - np.array(simt.get((j, ti, "unsafe"), [np.nan] * 8))
            dr = (reals["safe"][neurons[j]] - reals["unsafe"][neurons[j]]).cpu().numpy()
            corr_task[j]["dsim"] += ds.tolist(); corr_task[j]["dreal"] += dr.tolist()

        row = {"scn": rec["scn"], "oid": rec["oid"], "fi": rec.get("fi"), "S_real": Sreal, "denom": denom}
        for setname, members in SETS.items():
          for w in ("safe", "unsafe"):
            lpr = F.log_softmax(lg_real[w], -1)
            for mode in ("sim", "mean", "zero", "real_other"):
                spec = []
                for j in members:
                    L, i = neurons[j]
                    if mode == "sim":
                        v = simvals(j, w)
                    elif mode == "mean":
                        v = torch.full((len(pos),), float(cmean[j]), device="cuda")
                    elif mode == "zero":
                        v = torch.zeros(len(pos), device="cuda")
                    else:  # the real activation from the OTHER prompt (= real patch)
                        v = reals["unsafe" if w == "safe" else "safe"][(L, i)]
                    spec.append((L, i, 0, pos, v))
                lg = decide(rec[w + "_ids"], spec)
                s = float(lg[toi["safe"]] - lg[toi["unsafe"]])
                row[f"{setname}/{w}/{mode}/dS_norm"] = (s - Sreal[w]) / denom
                row[f"{setname}/{w}/{mode}/KL"] = float(kl_rows(lpr, F.log_softmax(lg, -1)))
          # simulated patch: UNSAFE prompt with the set's neurons at SIM(safe) values vs real patch
          spec = [(neurons[j][0], neurons[j][1], 0, pos, simvals(j, "safe")) for j in members]
          lg = decide(rec["unsafe_ids"], spec)
          row[f"{setname}/simpatch_recovery"] = (float(lg[toi["safe"]] - lg[toi["unsafe"]]) - Sreal["unsafe"]) / denom
          row[f"{setname}/realpatch_recovery"] = row[f"{setname}/unsafe/real_other/dS_norm"]
        # single-neuron substitution on both prompts -- batched (row 0 = unpatched baseline in the SAME batch,
        # so the KL reference has identical numerics; every batch padded to 16 rows)
        BS = 16
        for w in ("safe", "unsafe"):
            jobs = [(j, mode) for j in range(N) for mode in ("sim", "mean", "zero")]
            ids_t = torch.tensor([rec[w + "_ids"]], device="cuda").expand(BS, -1).contiguous()
            for c0 in range(0, len(jobs), BS - 1):
                chunk = jobs[c0:c0 + BS - 1]
                spec = []
                for r, (j, mode) in enumerate(chunk, start=1):
                    L, i = neurons[j]
                    v = simvals(j, w) if mode == "sim" else (
                        torch.full((len(pos),), float(cmean[j]), device="cuda") if mode == "mean" else torch.zeros(len(pos), device="cuda"))
                    spec.append((L, i, r, pos, v))
                S.spec = spec
                with torch.no_grad():
                    lg = model(input_ids=ids_t, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                S.spec = []
                lp = F.log_softmax(lg, -1)
                for r, (j, mode) in enumerate(chunk, start=1):
                    single_acc[j][f"kl_{mode}"].append(float(kl_rows(lp[0], lp[r])))
        task["per_instance"].append(row)
        print(f"  task {rec['scn']}/{rec['oid']}/f{rec.get('fi')}: " + " ".join(
              f"{sn}: simpatch={row[sn + '/simpatch_recovery']:+.3f} realpatch={row[sn + '/realpatch_recovery']:+.3f}"
              for sn in SETS), flush=True)

    for j in range(N):
        a = single_acc[j]
        km, ks, kz = np.sum(a["kl_mean"]), np.sum(a["kl_sim"]), np.sum(a["kl_zero"])
        c = corr_task[j]
        sim, real = np.array(c["sim"], float), np.array(c["real"], float)
        m = ~np.isnan(sim)
        r_task = float(np.corrcoef(sim[m], real[m])[0, 1]) if m.sum() > 5 and sim[m].std() > 0 else None
        ds, dr = np.array(c["dsim"], float), np.array(c["dreal"], float)
        m2 = ~np.isnan(ds)
        r_diff = float(np.corrcoef(ds[m2], dr[m2])[0, 1]) if m2.sum() > 5 and ds[m2].std() > 0 else None
        task["single"][j] = {"KL_sum": {"sim": ks, "mean": km, "zero": kz},
                             "damage_avoided_vs_mean": 1 - ks / km if km > 0 else None,
                             "r_task_acts": r_task, "r_task_safe_minus_unsafe": r_diff}
    out = {"neurons": neurons, "kinds": kinds, "sets": SETS, "calibration": cal, "corr_general": corr, "general": gen,
           "joint_general": joint_res, "task": task}
    json.dump(out, open(os.path.join(OUT, f"gao_{TAG}.json"), "w"), indent=1, default=str)
    print("T1_GAO_DONE", flush=True)


def ov():
    tok, model = load_model()
    import circuit_common as cc
    nj = json.load(open(os.path.join(OUT, f"t1_neurons_{TAG}.json")))
    heads = json.load(open(os.path.join(REPO, "results", "circ_heads_llama31_8b.json")))["summary"]["read_heads_ranked"][:12]
    heads = [tuple(int(x) for x in h["head"].split(".")) for h in heads]
    tp = json.load(open(os.path.join(OUT, f"task_prompts_{TAG}.json")))
    toi = tp[0]["toi"]
    W_U = model.lm_head.weight.float()
    gfin = model.model.norm.weight.float()
    c = model.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = c.hidden_size // nh
    rng = np.random.default_rng(0)
    top = [tuple(x) for x in nj["top300"]]
    I = c.intermediate_size

    def dirs(neur):
        return torch.stack([model.model.layers[L].mlp.down_proj.weight[:, i].float() for L, i in neur])  # [n, d]

    # per-scenario logit direction (safe/unsafe action tokens differ by scenario)
    ldirs = {}
    for r in tp:
        ldirs[r["scn"]] = (W_U[r["toi"]["safe"]] - W_U[r["toi"]["unsafe"]]) * gfin
    res = {}
    for (Lh, h) in heads:
        layer = model.model.layers[Lh]
        g = layer.input_layernorm.weight.float()
        Wv = layer.self_attn.v_proj.weight.float()[(h // (nh // nkv)) * hd:((h // (nh // nkv)) + 1) * hd]   # [hd, d]
        Wo = layer.self_attn.o_proj.weight.float()[:, h * hd:(h + 1) * hd]                                 # [d, hd]
        OV = lambda X: (X * g) @ Wv.T @ Wo.T                                                                # [n, d]
        sets = {}
        for name, neur in (("top30", top[:30]), ("top300", top[:300])):
            neur = [n for n in neur if n[0] < Lh]
            if not neur:
                continue
            X = dirs(neur); Y = OV(X)
            rand = [(L, int(rng.integers(I))) for L, _ in neur for _ in range(5)]
            Xr = dirs(rand); Yr = OV(Xr)
            def stats(Y, X):
                gain = (Y.norm(dim=1) / X.norm(dim=1))
                dl = {s: (Y @ v / X.norm(dim=1)).abs() for s, v in ldirs.items()}
                cos = {s: ((Y @ v) / (Y.norm(dim=1) * v.norm() + 1e-9)).abs() for s, v in ldirs.items()}
                return {"ov_gain_mean": float(gain.mean()),
                        "abs_logitdiff_per_unit_mean": {s: float(x.mean()) for s, x in dl.items()},
                        "abs_cos_to_logitdir_mean": {s: float(x.mean()) for s, x in cos.items()}}
            sets[name] = {"n": len(neur), "top": stats(Y, X), "rand_layer_matched": stats(Yr, Xr)}
        res[f"{Lh}.{h}"] = sets
        print(Lh, h, json.dumps(sets)[:300], flush=True)
    json.dump({"read_heads": [f"{a}.{b}" for a, b in heads], "results": res},
              open(os.path.join(OUT, f"ov_{TAG}.json"), "w"), indent=1)
    print("T1_OV_DONE", flush=True)


if __name__ == "__main__":
    {"prep": prep, "eval": evaluate, "ov": ov}[sys.argv[1]]()
