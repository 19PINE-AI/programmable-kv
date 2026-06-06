"""Latency/throughput: real wall-clock cost of each edit method = #token-positions recomputed.

Honest accounting. To apply an edit you must recompute KV for some token-positions, each attending to
the (cached) prefix:
  full reprefill    : recompute all L positions                      -> O(L)
  erratum (append)  : recompute only the appended suffix (~35 tok)   -> O(35), prefix cached
  field+selective@K : partial-recompute the field + K positions      -> O(K), prefix cached
                      (CacheBlend-style: run those tokens through the layers attending to cached KV)
We time a forward of n tokens given a cached prefix of length L-n, for n in {full, 35, K-grid}, at
several context lengths L, on real models. Maps each method to its measured latency + speedup vs full.
Run: python esys/latency_bench.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def prefix_cache(model, ids, upto):
    out = model(input_ids=ids[:, :upto], use_cache=True)
    return out.past_key_values


@torch.no_grad()
def time_recompute(model, ids, L, n, trials=12):
    """Time a forward of n tokens attending to a cached prefix of length L-n."""
    pre = L - n
    times = []
    for _ in range(trials):
        c = prefix_cache(model, ids, pre) if pre > 0 else DynamicCache()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(input_ids=ids[:, pre:L], past_key_values=c,
              cache_position=torch.arange(pre, L, device="cuda"), use_cache=True)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]   # median ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default=None)
    ap.add_argument("--Ls", default="256,2048,8192,32768")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    ERRATUM_TOK = 35
    KGRID = [4, 8, 16, 32, 64]
    out = {"model": args.model, "by_L": {}}
    print(f"==== LATENCY (median ms, {args.model}) ====")
    for L in [int(x) for x in args.Ls.split(",")]:
        ids = torch.randint(0, tok.vocab_size, (1, L), device="cuda")
        # warmup
        time_recompute(model, ids, L, min(8, L - 1), trials=3)
        full = time_recompute(model, ids, L, L, trials=8)                       # full reprefill (n=L, no prefix)
        err = time_recompute(model, ids, L, ERRATUM_TOK)
        ks = {k: time_recompute(model, ids, L, k) for k in KGRID}
        row = {"full_ms": round(full, 2), "erratum_ms": round(err, 2),
               "erratum_speedup": round(full / err, 1),
               "selective": {k: {"ms": round(v, 2), "speedup": round(full / v, 1)} for k, v in ks.items()}}
        out["by_L"][L] = row
        sel_str = " ".join(f"K{k}={ks[k]:.1f}ms({full/ks[k]:.0f}x)" for k in KGRID)
        print(f"  L={L:>6}: full={full:7.1f}ms | erratum(35tok)={err:6.1f}ms({full/err:.0f}x) | {sel_str}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"latency_{tag}.json"), "w"), indent=2)
    print("LATENCY_DONE")


if __name__ == "__main__":
    main()
