#!/bin/bash
# Re-run E5 with the matched-oracle fix (exact token stream), then final analyze + figures.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/e5redo_run.log; : > $LOG
echo "waiting for FINAL_ALL_DONE..." | tee -a $LOG
while ! grep -q "FINAL_ALL_DONE" results/final_run.log 2>/dev/null; do sleep 30; done
echo "final done; re-running E5 at $(date)" | tee -a $LOG
run() { echo "=== $* ===" | tee -a $LOG; timeout 5000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-4B unsloth/Meta-Llama-3.1-8B-Instruct; do
  run run_e5.py --model $M --sessions 24 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
done
for R in 0.1 0.5; do
  run run_e5.py --model Qwen/Qwen3-4B --sessions 16 --turns 12 --decide_every 3 --edit_rate $R --mtotal 120 --tag Qwen3-4B-r$R
done
python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "E5REDO_ALL_DONE $(date)" | tee -a $LOG
