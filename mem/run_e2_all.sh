#!/bin/bash
# E2 across models, sequential (one model in GPU at a time). Logs to results/e2_<tag>.log
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/e2_run.log; : > $LOG
# (model, n) — small models large N; 7-8B smaller N + short memory
run() { M=$1; N=$2; MT=$3; echo "=== $M (n=$N mtotal=$MT) ===" | tee -a $LOG
  timeout 3000 python run_e2.py --model "$M" --n "$N" --mtotal "$MT" >>$LOG 2>&1 && echo "ok $M" | tee -a $LOG || echo "FAIL $M" | tee -a $LOG; }
run Qwen/Qwen3-0.6B 400 24
run Qwen/Qwen3-1.7B 400 24
run Qwen/Qwen3-4B 400 24
run unsloth/gemma-2-2b-it 400 24
run mistralai/Mistral-7B-Instruct-v0.3 300 24
run unsloth/Meta-Llama-3.1-8B-Instruct 300 24
# long-memory faithfulness on 4B (mtotal=120 ~ 2k tokens)
echo "=== Qwen/Qwen3-4B LONG (mtotal=120) ===" | tee -a $LOG
timeout 3000 python run_e2.py --model Qwen/Qwen3-4B --n 250 --mtotal 120 --tag Qwen3-4B-long >>$LOG 2>&1 && echo "ok long" | tee -a $LOG || echo "FAIL long" | tee -a $LOG
echo "E2_ALL_DONE" | tee -a $LOG
