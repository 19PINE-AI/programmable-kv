#!/bin/bash
# Master chain: wait for E2 sweep, then run E1, E3, E4, E5, analyze, figures. Fully autonomous.
cd /home/ubuntu/editable-kv/patchkv/mem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/rest_run.log; : > $LOG
echo "waiting for E2_ALL_DONE..." | tee -a $LOG
while ! grep -q "E2_ALL_DONE" results/e2_run.log 2>/dev/null; do sleep 30; done
echo "E2 done; starting rest at $(date)" | tee -a $LOG

run() { echo "=== $* ===" | tee -a $LOG; timeout 6000 python "$@" >>$LOG 2>&1 && echo "ok $*" | tee -a $LOG || echo "FAIL $*" | tee -a $LOG; }

# E1 — placement x pre-digestion (CoT competent regime + direct floored baseline)
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-4B unsloth/Meta-Llama-3.1-8B-Instruct; do
  run run_e1.py --model $M --n 40 --facts 1,2,4 --mtotals 24,120 --regimes direct,cot
done

# E3 — editing memory mid-session (CoT), scale ladder for stickiness law
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-4B unsloth/Meta-Llama-3.1-8B-Instruct; do
  run run_e3.py --model $M --n 64 --nfacts 2 --regime cot
done

# E4 — granularity / sub-chunking (fast faithfulness)
for M in Qwen/Qwen3-4B unsloth/Meta-Llama-3.1-8B-Instruct; do
  run run_e4.py --model $M --n 300 --nfacts 4 --mtotal 60 --S 1,2,4,8,16
done

# E5 — end-to-end systems amortization + working app
for M in Qwen/Qwen3-1.7B Qwen/Qwen3-4B unsloth/Meta-Llama-3.1-8B-Instruct; do
  run run_e5.py --model $M --sessions 24 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
done

# E5 edit-rate sweep on 4B (for the crossover analysis)
for R in 0.1 0.5; do
  run run_e5.py --model Qwen/Qwen3-4B --sessions 16 --turns 12 --decide_every 3 --edit_rate $R --mtotal 120 --tag Qwen3-4B-r$R
done

echo "=== analyze + figures ===" | tee -a $LOG
python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "REST_ALL_DONE $(date)" | tee -a $LOG
