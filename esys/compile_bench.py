"""Does torch.compile speed up the EDITING partial-prefill / decode? (SDPA vs compiled)

The editing op = recompute a span of E tokens against an existing cache of length C.
We time: (a) SDPA eager-loop, (b) torch.compile(model) (fused / reduce-overhead).
Fixed shapes (no recompilation). Also times a single decode step.
"""
import argparse, os, sys, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def load(name, impl="sdpa"):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation=impl).eval()
    return tok, m


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


def timed(fn, trials=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(statistics.median(ts), 2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--C", type=int, default=2000, help="context (cache) length")
    ap.add_argument("--E", type=int, default=24, help="edited/erratum span length")
    args = ap.parse_args()
    tok, model = load(args.model)
    ids = torch.randint(0, 1000, (1, args.C + args.E), device="cuda")
    base = model(input_ids=ids[:, :args.C], use_cache=True).past_key_values  # cache of length C

    span = ids[:, args.C:args.C + args.E]
    cp_span = torch.arange(args.C, args.C + args.E, device="cuda")
    cp_dec = torch.tensor([args.C], device="cuda")
    dec_tok = ids[:, args.C:args.C + 1]

    def edit_sdpa():
        c = clone(base, args.C)
        model(input_ids=span, past_key_values=c, cache_position=cp_span, use_cache=True)

    def decode_sdpa():
        c = clone(base, args.C)
        model(input_ids=dec_tok, past_key_values=c, cache_position=cp_dec, use_cache=True)

    t_edit_sdpa = timed(edit_sdpa)
    t_dec_sdpa = timed(decode_sdpa)
    print(f"[SDPA]      edit({args.E} tok over C={args.C}) = {t_edit_sdpa} ms ; decode(1 tok) = {t_dec_sdpa} ms", flush=True)

    # compile the model forward (fused). dynamic=False -> static shapes (no recompile in loop)
    try:
        cmodel = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        def edit_comp():
            c = clone(base, args.C)
            cmodel(input_ids=span, past_key_values=c, cache_position=cp_span, use_cache=True)
        def decode_comp():
            c = clone(base, args.C)
            cmodel(input_ids=dec_tok, past_key_values=c, cache_position=cp_dec, use_cache=True)
        t_edit_c = timed(edit_comp, warmup=12)   # extra warmup to trigger compilation
        t_dec_c = timed(decode_comp, warmup=12)
        print(f"[compiled]  edit = {t_edit_c} ms ({t_edit_sdpa/max(t_edit_c,0.01):.2f}x) ; "
              f"decode = {t_dec_c} ms ({t_dec_sdpa/max(t_dec_c,0.01):.2f}x)", flush=True)
    except Exception as ex:
        print("[compiled] FAILED:", repr(ex)[:300], flush=True)
    # isolate the clone overhead
    print(f"[clone-only] clone(C={args.C}) = {timed(lambda: clone(base, args.C))} ms (the cache-copy our cost-frontier over-charges)", flush=True)
    print("COMPILE_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
