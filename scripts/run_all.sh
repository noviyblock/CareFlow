#!/usr/bin/env bash
# Start the full local experiment queue in the background; Mac stays awake while it runs.
#   scripts/run_all.sh            then monitor with: watch -n 30 scripts/status.sh
# Resumable: finished LLM calls come from the cache, so a restart only computes what is missing.
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
nohup caffeinate -dimsu scripts/run_queue.sh \
  configs/experiments/baseline_single_agent.yaml \
  configs/experiments/proxy_validity.yaml \
  configs/experiments/systems_eval.yaml \
  > outputs/logs/queue_$(date +%F_%H%M).log 2>&1 &
echo "started (pid $!); log: outputs/logs/queue_$(date +%F_%H%M).log"
