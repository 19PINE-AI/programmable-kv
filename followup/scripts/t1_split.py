"""T1 step 2 addendum: attention-vs-MLP split of the note write at the agg8 positions, measured
with the same attribution-patching estimate as the neuron ranking:
  AtP_attn(L) = sum_{p in agg8} (o_in_safe - o_in_unsafe)[L,p] . dS/d o_in |_unsafe   (all heads)
  AtP_mlp(L)  = sum_{p in agg8} (d_in_safe - d_in_unsafe)[L,p] . dS/d d_in |_unsafe   (all neurons)
plus actual patching of all attention outputs vs all MLP outputs at agg8 (each alone)."""
import json, os, sys
import numpy as np
import torch

REPO = os.environ.get("PKV_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "esys")); sys.path.insert(0, os.path.join(REPO, "e1"))
import circuit_common as cc  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_common import build_pair_f  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
for p in model.parameters():
    p.requires_grad_(False)
nl = model.config.num_hidden_layers
nj = json.load(open(os.path.join(OUT, "t1_neurons_llama31_8b.json")))
cap = {}; patch = {}; mode = [None]


def mk(kind, L):
    def hook(m, a):
        x = a[0]
        if mode[0] == "patch" and (kind, L) in patch:
            x = x.clone(); pos, v = patch[(kind, L)]; x[0, pos] = v.to(x.dtype)
        if mode[0] == "grad":
            x.retain_grad()
        cap[(kind, L)] = x
        return (x,)
    return hook


for L, l in enumerate(model.model.layers):
    l.self_attn.o_proj.register_forward_pre_hook(mk("attn", L))
    l.mlp.down_proj.register_forward_pre_hook(mk("mlp", L))
model.model.embed_tokens.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True) if mode[0] == "grad" else o)

rows = []
for inst in nj["instances"]:
    P = build_pair_f(tok, inst["scn"], inst["oid"], inst["fi"]); toi = P["toi"]; dpos = P["dpos"]
    pos = torch.tensor(inst["agg8"], device="cuda")
    acts = {}
    for w in ("safe", "unsafe"):
        mode[0] = "grad" if w == "unsafe" else None
        with torch.enable_grad():
            lg = model(input_ids=P[w + "_ids"].cuda(), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            S = lg[toi["safe"]] - lg[toi["unsafe"]]
            if w == "unsafe":
                S.backward()
        acts[w] = {k: v[0, pos].detach().float() for k, v in cap.items()}
        acts[w]["S"] = float(S)
        if w == "unsafe":
            grads = {k: v.grad[0, pos].float() for k, v in cap.items()}
        mode[0] = None
    den = acts["safe"]["S"] - acts["unsafe"]["S"]
    a_l = [float(((acts["safe"][("attn", L)] - acts["unsafe"][("attn", L)]) * grads[("attn", L)]).sum()) / den for L in range(nl)]
    m_l = [float(((acts["safe"][("mlp", L)] - acts["unsafe"][("mlp", L)]) * grads[("mlp", L)]).sum()) / den for L in range(nl)]
    act_patch = {}
    for kind in ("attn", "mlp"):
        patch.clear()
        for L in range(nl):
            patch[(kind, L)] = (pos, acts["safe"][(kind, L)])
        mode[0] = "patch"
        with torch.no_grad():
            lg = model(input_ids=P["unsafe_ids"].cuda(), use_cache=False, logits_to_keep=1).logits[0, -1].float()
        mode[0] = None; patch.clear()
        act_patch[kind] = (float(lg[toi["safe"]] - lg[toi["unsafe"]]) - acts["unsafe"]["S"]) / den
    rows.append({"scn": inst["scn"], "oid": inst["oid"], "fi": inst["fi"], "atp_attn_by_layer": a_l, "atp_mlp_by_layer": m_l,
                 "atp_attn": sum(a_l), "atp_mlp": sum(m_l), "patch_all_attn": act_patch["attn"],
                 "patch_all_mlp": act_patch["mlp"]})
    model.zero_grad(set_to_none=True); cap.clear(); torch.cuda.empty_cache()
    print(inst["scn"], inst["oid"], round(sum(a_l), 3), round(sum(m_l), 3), act_patch, flush=True)
A = np.array([r["atp_attn"] for r in rows]); M = np.array([r["atp_mlp"] for r in rows])
summ = {"atp_attn_share_of_attn_plus_mlp": float(np.mean(A / (A + M))), "atp_attn_mean": float(A.mean()),
        "atp_mlp_mean": float(M.mean()),
        "patch_all_attn_mean": float(np.mean([r["patch_all_attn"] for r in rows])),
        "patch_all_mlp_mean": float(np.mean([r["patch_all_mlp"] for r in rows])),
        "repo_circ_components_attn_share_T14": 0.599}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(OUT, "t1_split_llama31_8b.json"), "w"), indent=1)
print(summ, "\nT1_SPLIT_DONE", flush=True)
