"""T1 step 4a: run the top-N attributed neurons over a general corpus (fineweb-edu sample).

Saves token ids, per-neuron top positive / most negative examples (labeling split), activation
stats, and the full activations of the N neurons on a held-out split (for simulation).
"""
import argparse, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_GB", "30")) / 96.0)
OUT = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_rand", type=int, default=100)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--label_tokens", type=int, default=2_500_000)
    ap.add_argument("--held_tokens", type=int, default=250_000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--keep", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    st = torch.load(os.path.join(OUT, f"t1_atp_{args.tag}.pt"))
    top = [tuple(x) for x in st["top"][:args.n]]
    # layer-matched random controls (exclude the top-300)
    rng = np.random.default_rng(12345)
    excl = set(tuple(x) for x in st["top"][:300])
    I = st["atp_tr"].shape[1]
    rnd = []
    for L, _ in top[:args.n_rand]:
        while True:
            j = int(rng.integers(I))
            if (L, j) not in excl and (L, j) not in rnd:
                rnd.append((L, j)); break
    neurons = top + rnd
    kinds = ["top"] * len(top) + ["random"] * len(rnd)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    layers = sorted(set(L for L, _ in neurons))
    col = {L: [(j, i) for j, (LL, i) in enumerate(neurons) if LL == L] for L in layers}
    buf = {}

    def mk(L):
        idx = torch.tensor([i for _, i in col[L]], device="cuda")
        cols = [j for j, _ in col[L]]

        def hook(mod, a):
            buf[L] = (cols, a[0][:, :, idx].float())
        return hook
    hs = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in layers]
    # max layer needed -> we still run full model (cheap enough) to keep it simple

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    need = args.label_tokens + args.held_tokens
    toks = []
    total = 0
    bos = tok.bos_token_id
    cur = []
    for ex in ds:
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        cur.extend(ids + [tok.eos_token_id])
        while len(cur) >= args.seqlen - 1:
            toks.append([bos] + cur[:args.seqlen - 1])
            cur = cur[args.seqlen - 1:]
            total += args.seqlen
        if total >= need:
            break
    T = np.array(toks, dtype=np.int32)
    n_lab = args.label_tokens // args.seqlen
    print(f"corpus seqs={len(T)} label={n_lab} held={len(T)-n_lab} ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(OUT, f"corpus_tokens_{args.tag}.npy"), T)

    N = len(neurons)
    K = args.keep
    top_v = torch.full((N, K), -1e9, device="cuda"); top_i = torch.zeros((N, K), dtype=torch.long, device="cuda")
    bot_v = torch.full((N, K), 1e9, device="cuda"); bot_i = torch.zeros((N, K), dtype=torch.long, device="cuda")
    s1 = torch.zeros(N, device="cuda", dtype=torch.float64); s2 = torch.zeros_like(s1); cnt = 0
    npos = torch.zeros(N, device="cuda", dtype=torch.float64)
    sample = []
    held = []
    with torch.no_grad():
        for b0 in range(0, len(T), args.bs):
            x = torch.from_numpy(T[b0:b0 + args.bs]).long().cuda()
            buf.clear()
            model(input_ids=x, use_cache=False, logits_to_keep=1)
            A = torch.zeros(x.shape[0], x.shape[1], N, device="cuda")
            for L in layers:
                cols, v = buf[L]
                A[:, :, cols] = v
            A[:, 0, :] = 0.0  # ignore BOS position
            if b0 < n_lab:
                flat = A.reshape(-1, N)                         # [B*S, N]
                gidx = torch.arange(flat.shape[0], device="cuda") + b0 * args.seqlen
                s1 += flat.sum(0).double(); s2 += (flat.double() ** 2).sum(0); cnt += flat.shape[0]
                npos += (flat > 0).sum(0).double()
                if b0 % (args.bs * 20) == 0:
                    sample.append(flat[torch.randperm(flat.shape[0], device="cuda")[:512]].cpu())
                gi = gidx[None, :].expand(N, -1)
                cv = torch.cat([top_v, flat.T], 1); ci = torch.cat([top_i, gi], 1)
                v, o = cv.topk(K, dim=1)
                top_v, top_i = v, torch.gather(ci, 1, o)
                cv = torch.cat([bot_v, flat.T], 1); ci = torch.cat([bot_i, gi], 1)
                v, o = (-cv).topk(K, dim=1)
                bot_v, bot_i = -v, torch.gather(ci, 1, o)
            else:
                held.append(A.half().cpu())
            if (b0 // args.bs) % 100 == 0:
                print(f"  batch {b0}/{len(T)} ({time.time()-t0:.0f}s)", flush=True)
    mean = (s1 / cnt).float().cpu(); std = ((s2 / cnt).float().cpu() - mean ** 2).clamp(min=0).sqrt()
    S = torch.cat(sample).numpy()
    qs = np.quantile(S, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], axis=0)
    torch.save({"neurons": neurons, "kinds": kinds, "seqlen": args.seqlen, "n_lab": n_lab,
                "top_v": top_v.cpu(), "top_i": top_i.cpu(), "bot_v": bot_v.cpu(), "bot_i": bot_i.cpu(),
                "mean": mean, "std": std, "frac_pos": (npos / cnt).float().cpu(), "quantiles": qs,
                "held_acts": torch.cat(held)}, os.path.join(OUT, f"corpus_stats_{args.tag}.pt"))
    for h in hs:
        h.remove()
    print("T1_CORPUS_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
