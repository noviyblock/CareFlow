#!/usr/bin/env bash
# One-screen status of background experiments.  Usage: scripts/status.sh   (loop: watch -n 30 scripts/status.sh)
cd "$(dirname "$0")/.."
echo "== Queues"
cat outputs/logs/queue*.log 2>/dev/null
echo
echo "== Running"
pgrep -fl "scripts/run_experiment.py" | grep -v "uv run" | sed -E 's/.*--config ([^ ]+).*/\1/' | sort -u || true
[ -z "$(pgrep -f scripts/run_experiment.py)" ] && echo "(nothing)"
echo
echo "== Latest progress per experiment"
for f in outputs/logs/*.log; do
  case "$f" in *queue*) continue ;; esac
  line=$(grep -E "^\[[a-z_]+\]|Traceback|Error:" "$f" 2>/dev/null | tail -1)
  age=$(( $(date +%s) - $(stat -f %m "$f") ))
  [ -n "$line" ] && printf "%-28s %-90s (log updated %ss ago)\n" "$(basename "$f" .log)" "$line" "$age"
done
echo
echo "== Last LLM call"
uv run python -c "import sqlite3,time; r=sqlite3.connect('.cache/llm/llm_cache.sqlite').execute('select max(created_at) from calls').fetchone()[0]; print(f'{time.time()-r:.0f}s ago (if this keeps growing past a few minutes while something is Running, it is stuck)')" 2>/dev/null
echo
echo "== Finished results"
ls -1 outputs/*/metrics.json outputs/*/results.csv 2>/dev/null
