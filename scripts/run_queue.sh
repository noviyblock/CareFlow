#!/usr/bin/env bash
# Run experiment configs sequentially; survives terminal/session exit when started with nohup.
#   nohup scripts/run_queue.sh <config> [<config> ...] > outputs/logs/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export MLFLOW_DISABLE_AGENT_HINT=1
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "=== $(date '+%F %T') START $name"
  uv run python scripts/run_experiment.py --config "$cfg" > "outputs/logs/$name.log" 2>&1
  rc=$?  # capture before any command substitution resets $?
  echo "=== $(date '+%F %T') END $name (exit $rc)"
done
