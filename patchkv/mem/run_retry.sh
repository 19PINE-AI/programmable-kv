#!/bin/bash
# Retry the runs that OOM'd under GPU contention (now that the GPU is free).
cd /home/ubuntu/editable-kv/patchkv/mem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/retry_run.log; : > $LOG
run() { echo "=== $* ===" | tee -a $LOG; timeout 40000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }

# E3-32B (was incomplete, 287/336) — re-run for full data
run run_e3.py --model Qwen/Qwen3-32B-FP8 --n 48 --nfacts 2 --regime cot --tag Qwen3-32B-FP8
# E5 large (were OOM)
run run_e5.py --model Qwen/Qwen3-14B --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
run run_e5.py --model Qwen/Qwen3-32B-FP8 --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120 --tag Qwen3-32B-FP8
# LoCoMo 32B (was OOM) — full, all questions, full memory, flash
run run_locomo.py --model Qwen/Qwen3-32B-FP8 --convs 10 --q_per_conv 300 --max_mem_tok 24000 --tag Qwen3-32B-FP8

python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "RETRY_ALL_DONE $(date)" | tee -a $LOG
