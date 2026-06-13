"""Where torch.compile DOES win: StaticCache + compiled decode (static shapes).

Contrast with compile_bench.py (DynamicCache, compile barely helped / hurt decode).
StaticCache pre-allocates a fixed buffer (so editing is an in-place slice overwrite,
no clone) and is graph-capture friendly. We time single-token decode: SDPA-eager-loop
vs torch.compile(reduce-overhead, CUDA graphs). Also shows the in-place field edit.
"""
import argparse, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


def timed(fn, trials=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(statistics.median(ts), 3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--C", type=int, default=2000)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa").eval()
    maxlen = args.C + 64
    ids = torch.randint(0, 1000, (1, args.C), device="cuda")

    def fresh_cache():
        c = StaticCache(config=model.config, max_batch_size=1, max_cache_len=maxlen,
                        device="cuda", dtype=torch.bfloat16)
        model(input_ids=ids, past_key_values=c, cache_position=torch.arange(args.C, device="cuda"), use_cache=True)
        return c

    cache = fresh_cache()
    dec_tok = torch.randint(0, 1000, (1, 1), device="cuda")
    cp = torch.tensor([args.C], device="cuda")

    def decode_eager():
        model(input_ids=dec_tok, past_key_values=cache, cache_position=cp, use_cache=True)
    t_eager = timed(decode_eager)
    print(f"[StaticCache SDPA]   decode(1 tok, C={args.C}) = {t_eager} ms", flush=True)

    # in-place field edit demonstration: overwrite a span of K/V (no clone, no realloc)
    def edit_inplace():
        for layer in cache.layers if hasattr(cache, "layers") else []:
            layer.keys[:, :, 40:43, :].zero_()    # placeholder overwrite (real: new field KV)
    try:
        t_edit = timed(edit_inplace) if hasattr(cache, "layers") else None
        print(f"[StaticCache] in-place field edit (KV slice overwrite) = {t_edit} ms (no clone, no realloc)", flush=True)
    except Exception as ex:
        print("[in-place edit] note:", repr(ex)[:120], flush=True)

    try:
        cmodel = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        def decode_comp():
            cmodel(input_ids=dec_tok, past_key_values=cache, cache_position=cp, use_cache=True)
        t_comp = timed(decode_comp, warmup=20)
        print(f"[StaticCache compiled] decode = {t_comp} ms  ({t_eager/max(t_comp,1e-3):.2f}x vs SDPA-eager)", flush=True)
    except Exception as ex:
        print("[compiled] FAILED:", repr(ex)[:300], flush=True)
    print("STATIC_COMPILE_DONE", flush=True)


if __name__ == "__main__":
    main()
