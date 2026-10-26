#!/usr/bin/env bash
# Switch the model served by llama-server (systemd user service).
#
#   ./model.sh gptoss        gpt-oss-20b                       (fast, general)
#   ./model.sh qwen-coder    Qwen3-Coder-30B-A3B-Instruct      (heavier coding work)
#   ./model.sh status        what is running now
#
# Overrides (environment):
#   CTX  context per slot       NP  parallel slots (default 2: the agent and the model's
#                               own test runs share the server, see README_NEW.md)
#   The server gets CTX x NP tokens in total, which is what uses VRAM.
#   e.g.  CTX=8192 NP=1 ./model.sh qwen-coder      DRY_RUN=1 ./model.sh qwen-coder
set -euo pipefail
cd "$(dirname "$0")"

LLAMA="${LLAMA_SERVER:-$HOME/repos/llama.cpp/build/bin/llama-server}"
PORT="${LLAMA_PORT:-8080}"
NP="${NP:-2}"

status() {
  systemctl --user status llama --no-pager 2>&1 | head -n 6 || true
  echo "serving: $(curl -sf "localhost:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -n 1 || echo 'nothing (not ready)')"
}

case "${1:-}" in
  gptoss)
    REPO="ggml-org/gpt-oss-20b-GGUF";                      PER_SLOT="${CTX:-16384}" ;;
  qwen-coder)
    # ~18.6 GB at Q4_K_M. The default context keeps it inside 24 GB of VRAM; if you
    # still hit out-of-memory, lower it:  CTX=8192 ./model.sh qwen-coder
    REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"; PER_SLOT="${CTX:-12288}" ;;
  status)
    status; exit 0 ;;
  *)
    echo "usage: $0 {gptoss|qwen-coder|status}"; exit 1 ;;
esac

TOTAL=$((PER_SLOT * NP))
UNIT="[Unit]
Description=llama-server ($1)

[Service]
ExecStart=$LLAMA -hf $REPO --jinja -ngl 99 -c $TOTAL -np $NP --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target"

if [ -n "${DRY_RUN:-}" ]; then echo "$UNIT"; exit 0; fi

# deploy/llama.service is what ~/.config/systemd/user/llama.service links to
mkdir -p deploy
printf '%s\n' "$UNIT" > deploy/llama.service
systemctl --user daemon-reload
systemctl --user restart llama
echo "Starting $1: $NP slot(s) x $PER_SLOT tokens = $TOTAL total."
echo "(the first run of a model downloads it, which can take a while)"

# Wait until the server answers (up to 15 minutes, to allow for the download)
for _ in $(seq 1 450); do
  if curl -sf "localhost:$PORT/v1/models" >/dev/null; then
    echo "Ready."; status; echo
    echo "Tip: start a fresh conversation with the new model:  ./run.sh --new --chat"
    exit 0
  fi
  if ! systemctl --user is-active --quiet llama; then
    echo "Server stopped. Last log lines:"
    journalctl --user -u llama -n 25 --no-pager
    exit 1
  fi
  sleep 2
done
echo "Timed out waiting for the server. Check: journalctl --user -u llama -f"
exit 1
