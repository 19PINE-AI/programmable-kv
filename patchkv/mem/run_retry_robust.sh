#!/bin/bash
# Robust retry for runs that OOM under intermittent GPU contention: wait until enough GPU
# memory is free before launching each run, and retry on failure with backoff (the node is
# shared and other jobs grab/free 20-60GB unpredictably).
cd /home/ubuntu/editable-kv/patchkv/mem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/retry_run.log; : > $LOG

wait_gpu() {  # $1 = min free MiB; wait (up to ~3h) for a window
  for i in $(seq 1 180); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$1" ]; then return 0; fi
    echo "  [wait] ${free}MiB free, need $1 (try $i)" >> $LOG; sleep 60
  done
  echo "  [wait] gave up waiting for $1 MiB" >> $LOG; return 1
}

retry() {  # $1 = min free MiB; rest = python args
  minfree=$1; shift
  for attempt in $(seq 1 12); do
    wait_gpu "$minfree" || true
    echo "=== attempt $attempt: $* ($(date +%H:%M:%S)) ===" | tee -a $LOG
    if timeout 40000 python "$@" >>$LOG 2>&1; then echo "ok $*" | tee -a $LOG; return 0; fi
    echo "  attempt $attempt failed; backoff 120s" | tee -a $LOG; sleep 120
  done
  echo "FAIL (all attempts) $*" | tee -a $LOG; return 1
}

# thresholds sized to model weights + KV headroom
retry 40000 run_e3.py --model Qwen/Qwen3-32B-FP8 --n 48 --nfacts 2 --regime cot --tag Qwen3-32B-FP8
retry 34000 run_e5.py --model Qwen/Qwen3-14B --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
retry 42000 run_e5.py --model Qwen/Qwen3-32B-FP8 --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120 --tag Qwen3-32B-FP8
retry 55000 run_locomo.py --model Qwen/Qwen3-32B-FP8 --convs 10 --q_per_conv 300 --max_mem_tok 24000 --tag Qwen3-32B-FP8

python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "RETRY_ALL_DONE $(date)" | tee -a $LOG
