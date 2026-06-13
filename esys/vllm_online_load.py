"""Online load study on vLLM: throughput & TTFT under increasing concurrency.

Extends the offline batch result (§8c) to a *load sweep*. At each offered concurrency N we submit
N requests to a single vLLM engine (continuous batching, prefix caching ON) and measure aggregate
throughput (req/s) and per-request TTFT percentiles, for two regimes:
  BASELINE : each request carries the NEW field value EARLY in the prompt -> a unique prefix ->
             vLLM must full-prefill every request (compute-bound; saturates under load).
  ERRATUM  : all requests share one cached OLD prefix and append a short [STATE UPDATE] suffix ->
             the long prefix is an APC cache hit; only the suffix is computed (cache-bound).
The prediction: the erratum's throughput advantage GROWS with offered load (baseline is
compute-bound, erratum is cache-bound). Run: python esys/vllm_online_load.py
"""
import os, sys, time, json, statistics
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import vllm.platforms as _P  # NVML-broken-box workaround (see §10.1); harmless once NVML is fixed
_P.builtin_platform_plugins["cuda"] = lambda: "vllm.platforms.cuda.CudaPlatform"
_P._current_platform = None
from vllm import LLM, SamplingParams
TAU2 = "/home/ubuntu/tau2-bench/data/tau2/domains/retail/policy.md"


def ttfts(outs):
    vals = []
    for o in outs:
        m = getattr(o, "metrics", None)
        if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
            vals.append(m.first_token_time - m.arrival_time)
    return vals


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--loads", default="8,32,128,512")
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    args = ap.parse_args()
    policy = open(TAU2).read() if os.path.exists(TAU2) else ("RETAIL POLICY. " * 400)
    pre = "# Session header\norder_status: "                    # field placed EARLY
    post = ("\n\n" + policy + "\n\n# TASK\nDecide one word: cancel or deny.\nDecision:")
    erratum = "\n[STATE UPDATE] order_status has changed to {v}; this overrides any earlier value AND conclusion.\n"

    llm = LLM(model=args.model, enable_prefix_caching=True, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, dtype="bfloat16", trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(max_tokens=4, temperature=0.0)

    def make(N, regime):
        if regime == "baseline":
            return [pre + f"processed_{i:04d}" + post for i in range(N)]   # unique early field -> unique prefix
        return [pre + "pending" + post.replace("# TASK", erratum.format(v=f"processed_{i:04d}") + "# TASK")
                for i in range(N)]                                          # shared prefix + suffix

    rows = []
    for N in [int(x) for x in args.loads.split(",")]:
        row = {"N": N}
        for regime in ["baseline", "erratum"]:
            prompts = make(N, regime)
            llm.generate(prompts[:1], sp, use_tqdm=False)                   # warm the shared prefix (not timed)
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            tf = ttfts(outs)
            row[regime] = {"throughput_req_s": round(N / dt, 1), "total_s": round(dt, 3),
                           "ttft_ms": {"p50": round(1000 * statistics.median(tf), 1),
                                       "p90": round(1000 * sorted(tf)[int(0.9 * len(tf))], 1),
                                       "p99": round(1000 * sorted(tf)[min(len(tf) - 1, int(0.99 * len(tf)))], 1)}
                           if tf else None}
        b, e = row["baseline"], row["erratum"]
        row["throughput_speedup"] = round(e["throughput_req_s"] / b["throughput_req_s"], 2)
        rows.append(row)
        tt_b = b["ttft_ms"]["p50"] if b["ttft_ms"] else "n/a"
        tt_e = e["ttft_ms"]["p50"] if e["ttft_ms"] else "n/a"
        print(f"  N={N:4d}: baseline {b['throughput_req_s']:7.1f} req/s (TTFT p50 {tt_b} ms) | "
              f"erratum {e['throughput_req_s']:7.1f} req/s (TTFT p50 {tt_e} ms) | "
              f"throughput {row['throughput_speedup']}x", flush=True)

    json.dump({"model": args.model, "rows": rows}, open(os.path.join(os.path.dirname(__file__), "..",
              "results", f"vllm_online_load_{args.tag}.json"), "w"), indent=2)
    print("VLLM_ONLINE_LOAD_DONE")


if __name__ == "__main__":
    main()
