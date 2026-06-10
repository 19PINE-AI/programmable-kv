#!/bin/bash
# Runs after the master chain: negative control + LoCoMo attempt, then final analyze + figures.
cd /home/ubuntu/editable-kv/patchkv/mem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/final_run.log; : > $LOG
echo "waiting for REST_ALL_DONE..." | tee -a $LOG
while ! grep -q "REST_ALL_DONE" results/rest_run.log 2>/dev/null; do sleep 30; done
echo "rest done; starting final at $(date)" | tee -a $LOG
run() { echo "=== $* ===" | tee -a $LOG; timeout 4000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }
run run_negctrl.py --model Qwen/Qwen3-4B --n 48 --regime cot
run run_negctrl.py --model unsloth/Meta-Llama-3.1-8B-Instruct --n 48 --regime cot
python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "FINAL_ALL_DONE $(date)" | tee -a $LOG
