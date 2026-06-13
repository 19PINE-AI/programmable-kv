#!/bin/bash
# Large-model sweep (14B-70B) + LoCoMo external validity. Full GPU. Sequential.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/large_run.log; : > $LOG
run() { echo "=== $* ===" | tee -a $LOG; timeout 14000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }

L14=Qwen/Qwen3-14B
L32=Qwen/Qwen3-32B-FP8
L30=Qwen/Qwen3-30B-A3B
L70=unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit

# 1) competence (direct + CoT) — does the constant-answer floor break at scale?
for M in $L14 $L32 $L30 $L70; do run run_e1.py --model $M --n 24 --facts 1,2,4 --mtotals 24 --regimes direct,cot; done

# 2) E2 faithfulness (fast) on all large models
run run_e2.py --model $L14 --n 250 --mtotal 24
run run_e2.py --model $L32 --n 250 --mtotal 24 --tag Qwen3-32B-FP8
run run_e2.py --model $L30 --n 250 --mtotal 24
run run_e2.py --model $L70 --n 150 --mtotal 24 --tag Llama-3.1-70B-4bit

# 3) LoCoMo external validity (real conversational memory)
run run_locomo.py --model unsloth/Meta-Llama-3.1-8B-Instruct --convs 10 --q_per_conv 15 --max_mem_tok 8000
run run_locomo.py --model $L14 --convs 10 --q_per_conv 15 --max_mem_tok 8000
run run_locomo.py --model $L32 --convs 10 --q_per_conv 15 --max_mem_tok 8000 --tag Qwen3-32B-FP8

# 4) E3 editing (CoT) — extend stickiness/scale ladder
run run_e3.py --model $L14 --n 48 --nfacts 2 --regime cot
run run_e3.py --model $L32 --n 48 --nfacts 2 --regime cot --tag Qwen3-32B-FP8

# 5) E5 systems — bigger TTFT speedups at scale
run run_e5.py --model $L14 --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
run run_e5.py --model $L32 --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120 --tag Qwen3-32B-FP8

# analyze + figures
python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "LARGE_ALL_DONE $(date)" | tee -a $LOG
