"""Serving-oriented latency under load: batched TTFT, editkv vs full reprefill, incl. 32K.

The cost_frontier measured single-stream wall-clock. Here we measure the serving-relevant
quantity: time-to-first-token (TTFT) to make a decode-ready cache after a mutable field
changes, as a function of context length AND batch size. full_reprefill re-prefills the whole
(grown) context for every request in the batch; editkv reuses the cached prefix and recomputes
only the field (in_place) or a short suffix (erratum). We report mean per-request TTFT and the
speedup, including a long-context (up to 32K) point where the gap is largest.

This is an HF-level measurement (CUDA events, warmup+median); a production paged-attention
engine (vLLM/SGLang) would amortize the prefix further, so these are conservative lower bounds
on the serving win. Run: MECH_ATTN=sdpa python esys/serving_bench.py
"""
import argparse, os, sys, json, time, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def synth_ids(tok, n_tokens):
    base = tok("The current order_status is pending. " * 4, add_special_tokens=False)["input_ids"]
    ids = (base * (n_tokens // len(base) + 1))[:n_tokens]
    return torch.tensor([ids])


@torch.no_grad()
def cuda_time(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@torch.no_grad()
def prefill_batch(model, ids, bs):
    batch = ids.repeat(bs, 1).to("cuda")
    return model(input_ids=batch, use_cache=True).past_key_values


@torch.no_grad()
def build_cache(model, ids, bs, upto):
    """A real KV cache of length `upto` for a batch of bs (NOT timed; amortized prefix)."""
    batch = ids.repeat(bs, 1).to("cuda")[:, :upto]
    return model(input_ids=batch, use_cache=True).past_key_values


@torch.no_grad()
def suffix_forward(model, cache, suffix_len, base_len, bs):
    """Time a forward of `suffix_len` new tokens at position base_len, attending over the full
    base_len-length cache (the real editkv suffix-recompute cost)."""
    ids = torch.zeros(bs, suffix_len, dtype=torch.long, device="cuda")
    pos = torch.arange(base_len, base_len + suffix_len, device="cuda")
    model(input_ids=ids, past_key_values=cache, cache_position=pos, use_cache=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--lengths", default="1024,4096,16384,32768")
    ap.add_argument("--batches", default="1,8,32")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    lengths = [int(x) for x in args.lengths.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    ERRATUM_SUFFIX = 32   # tokens recomputed by the erratum (trigger + decision prompt)
    FIELD_TOK = 1         # in_place recomputes ~the field tokens

    def truncate(cache, T):
        for L in cache.layers:
            L.keys = L.keys[:, :, :T, :]; L.values = L.values[:, :, :T, :]

    @torch.no_grad()
    def time_suffix(ids, bs, T, suffix_len, iters=5, warmup=2):
        cache = build_cache(model, ids, bs, T)        # real length-T cache (untimed)
        def once():
            suffix_forward(model, cache, suffix_len, T, bs); torch.cuda.synchronize(); truncate(cache, T)
        for _ in range(warmup):
            once()
        tlist = []
        for _ in range(iters):
            torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); suffix_forward(model, cache, suffix_len, T, bs); e.record(); torch.cuda.synchronize()
            tlist.append(s.elapsed_time(e)); truncate(cache, T)
        del cache; torch.cuda.empty_cache()
        return statistics.median(tlist)

    rows = []
    for T in lengths:
        ids = synth_ids(tok, T)
        for bs in batches:
            if T * bs > 32768 * 8:
                rows.append({"T": T, "bs": bs, "skipped": "T*bs too large"}); continue
            try:
                full = cuda_time(lambda: prefill_batch(model, ids, bs))
                torch.cuda.empty_cache()
                ip = time_suffix(ids, bs, T, FIELD_TOK)       # in_place: recompute field token(s) over cache
                err = time_suffix(ids, bs, T, ERRATUM_SUFFIX)  # erratum: recompute short suffix over cache
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); rows.append({"T": T, "bs": bs, "skipped": "OOM"}); continue
            row = {"T": T, "bs": bs, "full_ms": round(full, 1), "in_place_ms": round(ip, 2),
                   "erratum_ms": round(err, 2), "speedup_inplace": round(full / ip, 1),
                   "speedup_erratum": round(full / err, 1)}
            rows.append(row)
            print(f"  T={T:5d} bs={bs:2d}: full={full:8.1f}ms in_place={ip:6.2f}ms "
                  f"erratum={err:6.2f}ms | speedup {row['speedup_inplace']}x / {row['speedup_erratum']}x", flush=True)
            torch.cuda.empty_cache()
    json.dump({"model": args.model, "rows": rows}, open(os.path.join(os.path.dirname(__file__),
              "..", "results", f"serving_bench_{args.tag}.json"), "w"), indent=2)
    print("SERVING_BENCH_DONE")


if __name__ == "__main__":
    main()
