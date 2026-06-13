#!/bin/bash
# Full LoCoMo run: ALL answerable questions (all 10 conversations), 12k-token real memory.
# Chained after the large sweep to avoid GPU contention. Supersedes the sweep's lighter LoCoMo.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/locomo_full_run.log; : > $LOG
echo "waiting for LARGE_ALL_DONE..." | tee -a $LOG
while ! grep -q "LARGE_ALL_DONE" results/large_run.log 2>/dev/null; do sleep 60; done
echo "large sweep done; starting full LoCoMo at $(date)" | tee -a $LOG
run() { echo "=== $* ===" | tee -a $LOG; timeout 50000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }
# ALL answerable questions (q_per_conv huge), FULL conversation memory (24k covers all convs),
# flash attention (O(L) memory). Ordered small->large; run() continues past any OOM.
run run_locomo.py --model Qwen/Qwen3-4B --convs 10 --q_per_conv 300 --max_mem_tok 24000
run run_locomo.py --model unsloth/Meta-Llama-3.1-8B-Instruct --convs 10 --q_per_conv 300 --max_mem_tok 24000
run run_locomo.py --model Qwen/Qwen3-14B --convs 10 --q_per_conv 300 --max_mem_tok 24000
run run_locomo.py --model Qwen/Qwen3-32B-FP8 --convs 10 --q_per_conv 300 --max_mem_tok 24000 --tag Qwen3-32B-FP8
python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "LOCOMO_FULL_DONE $(date)" | tee -a $LOG
