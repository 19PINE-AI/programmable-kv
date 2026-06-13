#!/bin/bash
# Second retry: E5-14B (raise mem threshold so it launches with headroom against contention)
# and the 32B LoCoMo point using the BF16 checkpoint (flash supports bf16; FP8 was rejected).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/retry2_run.log; : > $LOG

wait_gpu() {
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$1" ]; then return 0; fi
    echo "  [wait] ${free}MiB free, need $1 (try $i)" >> $LOG; sleep 60
  done; return 1
}
retry() {
  minfree=$1; shift
  for attempt in $(seq 1 12); do
    wait_gpu "$minfree" || true
    echo "=== attempt $attempt: $* ($(date +%H:%M:%S)) ===" | tee -a $LOG
    if timeout 50000 python "$@" >>$LOG 2>&1; then echo "ok $*" | tee -a $LOG; return 0; fi
    echo "  attempt $attempt failed; backoff 120s" | tee -a $LOG; sleep 120
  done
  echo "FAIL (all attempts) $*" | tee -a $LOG; return 1
}

# E5-14B: needs ~28GB weights + headroom; launch only with >=50GB free to survive contention spikes
retry 50000 run_e5.py --model Qwen/Qwen3-14B --sessions 16 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120
# 32B LoCoMo via bf16 checkpoint (64GB weights + 24k KV ~13GB) -> need a near-empty GPU
retry 82000 run_locomo.py --model Qwen/Qwen3-32B --convs 10 --q_per_conv 300 --max_mem_tok 24000 --tag Qwen3-32B

python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "RETRY2_ALL_DONE $(date)" | tee -a $LOG
