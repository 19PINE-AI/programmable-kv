"""Comprehensive ONLINE vLLM serving benchmark for the append-only erratum.

This replaces the offline-batch microbenchmark (vllm_editkv_serving.py) with a real
online-serving experiment on vLLM's V1 engine: AsyncLLMEngine, CUDA graphs ON (NOT
enforce_eager), continuous batching, automatic prefix caching (APC), and POISSON request
arrivals at controlled request rates — the standard vLLM serving methodology. We measure,
per arm and per offered load: TTFT (p50/p90/p99), per-output-token latency (TPOT), end-to-end
latency, achieved throughput (req/s and output tok/s), and the engine's prefix-cache hit rate.

Workload: a long shared agent policy (the real tau2 retail policy, optionally padded to a
target length) with a single MUTABLE field whose value changes per request. Two arms:

  BASELINE  — the new field value is written EARLY (in the system header, before the long
              policy). Mutating a token inside the prefix changes that block's content hash
              and invalidates every downstream APC block, so the engine re-prefills the whole
              policy on every request. (This is the de-facto "just put the new value in the
              prompt" approach.)
  ERRATUM   — the field keeps its OLD value (prefix is byte-identical across requests) and the
              new value is supplied by a short appended "[STATE UPDATE] ... overrides ..."
              suffix. The long policy prefix is an APC cache hit; only the short suffix +
              decode is new work. (editkv's append-only edit.)

Prediction: TTFT and throughput diverge as offered load rises (baseline is prefill/compute-
bound; erratum is cache-bound). Run:
  python esys/vllm_serving_online.py --model Qwen/Qwen3-8B --tag qwen3_8b \
      --rates 2,4,8,16,32 --pad_tokens 6000 --max_new 96 --n 96
"""
import os
import argparse, asyncio, json, os, time, statistics, uuid
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# NVML userspace/kernel mismatch on this box breaks vLLM's NVML platform detection; force CUDA.
import vllm.platforms as _P
_P.builtin_platform_plugins["cuda"] = lambda: "vllm.platforms.cuda.CudaPlatform"
_P._current_platform = None
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

TAU2 = os.environ.get("TAU2_POLICY", os.path.expanduser("~/tau2-bench/data/tau2/domains/retail/policy.md"))


def pctl(xs, p):
    if not xs: return None
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 1)


def build_workload(tokenizer, n, pad_tokens):
    policy = open(TAU2).read() if os.path.exists(TAU2) else ("RETAIL POLICY. " * 400)
    # pad the shared policy up to ~pad_tokens tokens with neutral, cache-shared filler
    filler = "\n".join(f"# Reference clause {i}: routine, non-binding background guidance for agents."
                       for i in range(4000))
    base = policy + "\n\n" + filler
    ids = tokenizer(base, add_special_tokens=False)["input_ids"][:pad_tokens]
    shared_policy = tokenizer.decode(ids)
    OLD = "pending"
    NEWS = [f"processed_{i:04d}" for i in range(n)]
    pre = "# Session header\norder_status: "
    post = ("\n\n" + shared_policy + "\n\n# Conversation\nuser: I'd like to cancel order #W2378156.\n"
            "assistant: Let me verify the order status against the policy.\n\n# TASK\n"
            "Decide and act. Explain briefly, then state the action.\nAssistant:")
    erratum = ("\n[STATE UPDATE] order_status has changed to {v}; this overrides any earlier "
               "value AND conclusion.\n")
    baseline = [pre + v + post for v in NEWS]                                  # unique early field
    erratum_prompts = [pre + OLD + post.replace("# TASK", erratum.format(v=v) + "# TASK")
                       for v in NEWS]                                          # shared prefix + suffix
    return baseline, erratum_prompts


async def one_request(engine, prompt, sp, rid, out):
    t0 = time.perf_counter()
    ttft = None; ntok = 0; last = t0
    async for ro in engine.generate(prompt, sp, request_id=rid):
        now = time.perf_counter()
        if ttft is None and ro.outputs and ro.outputs[0].token_ids:
            ttft = now - t0
        if ro.outputs:
            ntok = len(ro.outputs[0].token_ids)
        last = now
    e2e = last - t0
    out.append({"ttft": ttft if ttft is not None else e2e, "e2e": e2e, "ntok": max(1, ntok),
                "tpot": (e2e - (ttft or 0)) / max(1, ntok - 1) if ntok > 1 else 0.0})


async def run_arm(engine, prompts, sp, rate, warm):
    # warm the shared prefix (erratum) / one prefill (baseline) — not timed
    async for _ in engine.generate(warm, sp, request_id=f"warm-{uuid.uuid4()}"):
        pass
    out = []
    tasks = []
    rng = np.random.RandomState(0)
    t_start = time.perf_counter()
    for i, p in enumerate(prompts):
        tasks.append(asyncio.create_task(one_request(engine, p, sp, f"req-{i}-{uuid.uuid4()}", out)))
        if rate > 0 and i < len(prompts) - 1:
            await asyncio.sleep(rng.exponential(1.0 / rate))   # Poisson inter-arrival
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start
    ttfts = [r["ttft"] for r in out]; e2es = [r["e2e"] for r in out]
    tpots = [r["tpot"] for r in out if r["tpot"] > 0]; ntoks = sum(r["ntok"] for r in out)
    return {"n": len(out), "wall_s": round(wall, 3),
            "throughput_req_s": round(len(out) / wall, 2),
            "output_tok_s": round(ntoks / wall, 1),
            "ttft_ms": {"mean": round(1000 * statistics.mean(ttfts), 1),
                        "p50": pctl(ttfts, 0.5), "p90": pctl(ttfts, 0.9), "p99": pctl(ttfts, 0.99)},
            "tpot_ms": round(1000 * statistics.mean(tpots), 1) if tpots else None,
            "e2e_ms": {"mean": round(1000 * statistics.mean(e2es), 1), "p90": pctl(e2es, 0.9)}}


def _prom_counters():
    """Snapshot vLLM's global Prometheus APC counters (V1 has no engine.get_metrics)."""
    hits = queries = 0.0
    try:
        from prometheus_client import REGISTRY
        for mf in REGISTRY.collect():
            if mf.name in ("vllm:gpu_prefix_cache_hits", "vllm:prefix_cache_hits"):
                hits = sum(s.value for s in mf.samples if s.name.endswith("_total"))
            if mf.name in ("vllm:gpu_prefix_cache_queries", "vllm:prefix_cache_queries"):
                queries = sum(s.value for s in mf.samples if s.name.endswith("_total"))
    except Exception:
        pass
    return hits, queries


def hit_rate_delta(before, after):
    dh, dq = after[0] - before[0], after[1] - before[1]
    return round(dh / dq, 4) if dq > 0 else None


async def main_async(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    baseline, erratum = build_workload(tok, args.n, args.pad_tokens)
    eargs = AsyncEngineArgs(model=args.model, enable_prefix_caching=True,
                            gpu_memory_utilization=args.gpu_mem, max_model_len=args.pad_tokens + 1024,
                            dtype="bfloat16", trust_remote_code=True, enforce_eager=False)
    engine = AsyncLLMEngine.from_engine_args(eargs)
    sp = SamplingParams(max_tokens=args.max_new, temperature=0.0)
    prompt_tok = len(tok(baseline[0], add_special_tokens=False)["input_ids"])
    print(f"=== ONLINE vLLM serving ({args.model}) prompt≈{prompt_tok} tok, CUDA graphs ON, APC ON ===")
    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        arms = {}
        for name, prompts in [("baseline", baseline), ("erratum", erratum)]:
            before = _prom_counters()
            r = await run_arm(engine, prompts, sp, rate, prompts[0])
            r["prefix_hit_rate"] = hit_rate_delta(before, _prom_counters())
            arms[name] = r
        b, e = arms["baseline"], arms["erratum"]
        row = {"rate": rate, "baseline": b, "erratum": e,
               "ttft_p90_speedup": round(b["ttft_ms"]["p90"] / e["ttft_ms"]["p90"], 2) if e["ttft_ms"]["p90"] else None,
               "throughput_speedup": round(e["throughput_req_s"] / b["throughput_req_s"], 2)}
        rows.append(row)
        print(f"  rate={rate:5}/s | TTFT p90  base {b['ttft_ms']['p90']:>8} ms  err {e['ttft_ms']['p90']:>7} ms "
              f"({row['ttft_p90_speedup']}x) | tput base {b['throughput_req_s']:>6} err {e['throughput_req_s']:>6} req/s "
              f"({row['throughput_speedup']}x) | hit base {b['prefix_hit_rate']} err {e['prefix_hit_rate']}", flush=True)
    out = {"model": args.model, "prompt_tokens": prompt_tok, "max_new": args.max_new, "n": args.n, "rows": rows}
    path = os.path.join(os.path.dirname(__file__), "..", "results", f"vllm_online_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("VLLM_SERVING_ONLINE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--rates", default="2,4,8,16,32,0")   # 0 == unthrottled (max throughput)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--pad_tokens", type=int, default=6000)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
