#!/usr/bin/env bash
# Start vLLM servers for the roles in configs/models_colab.yaml and wait until they answer.
#   scripts/colab_serve.sh gen     # Qwen3.5-9B (agent) + Qwen3.5-2B (slm): phase 1, generation
#   scripts/colab_serve.sh judge   # Gemma-4-26B-A4B (judge): phase 2, judging (and judge validation)
#   scripts/colab_serve.sh stop
# vLLM lives in its own venv (/content/vllm-env) so its torch pin does not clash with the project's.
# Revisions are read from configs/models_colab.yaml so the served weights are exactly the pinned ones.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM=${VLLM:-/content/vllm-env/bin/vllm}
LOGS=outputs/logs/vllm; mkdir -p "$LOGS"

rev() { python3 - "$1" <<'PY'
import sys, yaml
m = yaml.safe_load(open("configs/models_colab.yaml"))["models"][sys.argv[1]]
print(m["id"], m["revision"], m["base_url"].split(":")[-1].split("/")[0])
PY
}

serve() {  # role  gpu_fraction  max_len  [extra args]
  read -r id revision port < <(rev "$1")
  echo "starting $1 = $id@${revision:0:12} on :$port (gpu ${2})"
  nohup "$VLLM" serve "$id" --revision "$revision" --served-model-name "$1" --port "$port" \
    --gpu-memory-utilization "$2" --max-model-len "$3" --enable-prefix-caching "${@:4}" \
    > "$LOGS/$1.log" 2>&1 &
}

wait_up() {
  read -r _ _ port < <(rev "$1")
  for _ in $(seq 1 180); do curl -sf "http://localhost:$port/v1/models" >/dev/null && { echo "$1 up"; return; }; sleep 5; done
  echo "$1 did not start, see $LOGS/$1.log"; exit 1
}

case "${1:-}" in
  # on a 40 GB A100 pass GEN_ARGS="--quantization fp8" for the 9B (bf16 leaves too little KV cache next to the 2B)
  gen)   serve agent 0.55 8192 ${GEN_ARGS:-}; serve slm 0.25 8192; wait_up agent; wait_up slm ;;
  # 26B-A4B in bf16 needs ~52 GB: on a 40 GB A100 pass JUDGE_ARGS="--quantization fp8" (weight-only on Ampere).
  judge) serve judge 0.90 8192 ${JUDGE_ARGS:-}; wait_up judge ;;
  stop)  pkill -f "vllm serve" || true; sleep 5; echo stopped ;;
  *) echo "usage: $0 gen|judge|stop"; exit 2 ;;
esac
