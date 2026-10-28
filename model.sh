#!/usr/bin/env bash
# Switch the model served by llama-server (systemd user service).
#
#   ./model.sh <model>       switch to <model> (downloads it only if you don't have it)
#   ./model.sh status        what is running now
#   ./model.sh list          aliases and models already downloaded
#
# <model> can be:
#   an alias               gptoss | qwen-coder
#   a Hugging Face repo    org/repo-GGUF            (llama.cpp picks Q4_K_M if there is one)
#                          org/repo-GGUF:Q5_K_M     (pick a quantization)
#   a local file           /path/to/model.gguf
#
# Progress is shown in four steps: [1/4] check, [2/4] restart the server, [3/4] download
# (with percentage, speed and ETA; skipped when the model is already on disk) and
# [4/4] load the model into the GPU. A model you already have is used straight from disk,
# with no network access at all.
#
# Environment overrides:
#   CTX    context per slot (default depends on the model)
#   NP     parallel slots, default 1 (the agent and the model's own test runs share the
#          server, see README_NEW.md). The server gets CTX x NP tokens in total.
#   EXTRA  extra llama-server flags, e.g. EXTRA="-ctk q8_0 -ctv q8_0"
#   FORCE=1       go ahead even if the checks below fail
#   SKIP_CHECK=1  don't contact Hugging Face first
#   DRY_RUN=1     run the checks and print the service file, change nothing
#   WAIT          seconds to wait for the server (default 3600; big downloads are slow)
#
# Gated/private models: put  HF_TOKEN=hf_xxx  in ~/.config/llama-server.env
# (kept out of the repo on purpose; the service file reads it from there).
set -euo pipefail
cd "$(dirname "$0")"

LLAMA="${LLAMA_SERVER:-$HOME/repos/llama.cpp/build/bin/llama-server}"
PORT="${LLAMA_PORT:-8080}"
NP="${NP:-1}"
WAIT="${WAIT:-3600}"
HF="${HF_ENDPOINT:-https://huggingface.co}"
ENV_FILE="$HOME/.config/llama-server.env"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
OLD_CACHE="${LLAMA_CACHE:-$HOME/.cache/llama.cpp}"

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
step() { echo; echo "[$1/4] $2"; }

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
}

# nvidia-smi lives in /usr/lib/wsl/lib under WSL and is usually not on PATH.
NVSMI="$(command -v nvidia-smi || true)"
if [ -z "$NVSMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then NVSMI=/usr/lib/wsl/lib/nvidia-smi; fi
gpu_mib() {   # gpu_mib used|total  ->  MiB, or nothing if there is no GPU tool
  [ -n "$NVSMI" ] || return 0
  "$NVSMI" --query-gpu="memory.$1" --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true
}

# llama-server answers /v1/models while it is still loading; /health is 200 only when ready.
server_ready() { curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1; }

status() {
  systemctl --user status llama --no-pager 2>&1 | head -n 6 || true
  if server_ready; then
    echo "serving: $(curl -sf "localhost:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -n 1 | sed 's/"id":"\(.*\)"/\1/' || echo '?') (ready)"
  else
    echo "serving: nothing yet (not running, or still loading)"
  fi
}

list_models() {
  echo "aliases:"
  echo "  gptoss       ggml-org/gpt-oss-20b-GGUF"
  echo "  qwen-coder   unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
  echo "downloaded:"
  ls -1 "$HUB" 2>/dev/null | sed -n 's/^models--//p' | sed 's|--|/|; s/^/  /' || true
}

# ---- what did the user ask for? -> SPEC (HF repo[:quant]) or LOCAL (file), DEF_CTX
SPEC=""; LOCAL=""; DEF_CTX=8192
case "${1:-}" in
  "")          usage ;;
  status)      status; exit 0 ;;
  list)        list_models; exit 0 ;;
  gptoss)      SPEC="ggml-org/gpt-oss-20b-GGUF"; DEF_CTX=32768 ;;
  qwen-coder)  # ~18.6 GB at Q4_K_M; this context keeps it inside 24 GB of VRAM
               SPEC="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"; DEF_CTX=12288 ;;
  *.gguf)      [ -f "$1" ] || die "file not found: $1"; LOCAL="$(realpath "$1")" ;;
  */*)         SPEC="$1" ;;
  *)           echo "Unknown model '$1' (use an alias, org/repo[:quant], or a .gguf path)." >&2; usage ;;
esac
LABEL="$1"
PER_SLOT="${CTX:-$DEF_CTX}"
TOTAL=$((PER_SLOT * NP))
REPO="${SPEC%%:*}"; QUANT=""; [[ "$SPEC" == *:* ]] && QUANT="${SPEC#*:}"

# ---- Hugging Face token (optional; needed for gated/private repos)
TOKEN="${HF_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f "$ENV_FILE" ]; then
  TOKEN="$(grep -E '^HF_TOKEN=' "$ENV_FILE" | head -n 1 | cut -d= -f2- || true)"
fi
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

# ---- is the model already on disk? Prints the path of the (first) .gguf file, or nothing.
find_cached() {
  python3 - "$HUB" "$OLD_CACHE" "$REPO" "$QUANT" <<'PY'
import pathlib, re, sys
hub, old, repo, quant = sys.argv[1:5]
want = (quant or "q4_k_m").lower()
cands = []                                            # (path, name used for matching)
snap = pathlib.Path(hub) / ("models--" + repo.replace("/", "--")) / "snapshots"
if snap.is_dir():
    cands += [(p, p.relative_to(snap).as_posix()) for p in snap.rglob("*.gguf")]
oldp = pathlib.Path(old)
if oldp.is_dir():                                     # older llama.cpp: <org>_<repo>_<file>.gguf
    cands += [(p, p.name) for p in oldp.glob(repo.replace("/", "_") + "_*.gguf")]
ok = [(p, n) for p, n in cands if "mmproj" not in n.lower() and want in n.lower()
      and p.exists() and p.stat().st_size > 0]        # p.exists() is False for a broken symlink
shard = re.compile(r"^(.*)-(\d+)-of-(\d+)\.gguf$")
firsts = []
for p, n in sorted(ok):
    m = shard.match(p.name)
    if not m:
        firsts.append(p)
    elif int(m.group(2)) == 1 and \
            sum(1 for q, _ in ok if q.parent == p.parent and q.name.startswith(m.group(1) + "-")) >= int(m.group(3)):
        firsts.append(p)                              # all shards present: load via the first
if firsts:
    print(firsts[0])
PY
}

# ---- pre-flight: does the repo exist, is there a GGUF, will it fit? Nothing is changed
# (and the current server keeps running) if this fails. Sets TOTAL_BYTES.
TOTAL_BYTES=0
preflight() {
  local tmp code rc out vram="" disk=0 c_free
  tmp="$(mktemp)"; trap 'rm -f "$tmp"' RETURN
  code="$(curl -s -m 20 -o "$tmp" -w '%{http_code}' ${AUTH[@]+"${AUTH[@]}"} \
          "$HF/api/models/$REPO/tree/main?recursive=true" || true)"
  case "$code" in
    200) ;;
    404) die "'$REPO' was not found on Hugging Face. Check the spelling (org/name)." ;;
    401|403) die "'$REPO' is gated or private. Accept its terms on the website, then put HF_TOKEN=... in $ENV_FILE" ;;
    ""|000) warn "couldn't reach Hugging Face to verify the model; continuing without checks."; return 0 ;;
    *) warn "Hugging Face answered HTTP $code; continuing without checks."; return 0 ;;
  esac

  vram="$(gpu_mib total)"
  disk="$(df -B1 --output=avail "$HOME" | tail -n 1 | tr -d ' ')"
  # Under WSL, df shows the size of WSL's virtual disk, not the Windows drive that backs it.
  # That drive filling up is what corrupts WSL, so check C: too and use the smaller number.
  if [ -d /mnt/c ]; then
    c_free="$(df -B1 --output=avail /mnt/c 2>/dev/null | tail -n 1 | tr -d ' ' || true)"
    if [ -n "$c_free" ] && [ "$c_free" -lt "$disk" ]; then disk="$c_free"; fi
  fi

  set +e
  out="$(python3 - "$tmp" "$QUANT" "${vram:-0}" "${disk:-0}" <<'PY'
import json, sys
path, quant, vram_mb, disk = sys.argv[1], sys.argv[2].lower(), int(sys.argv[3]), int(sys.argv[4])
files = json.load(open(path))
gg = [f for f in files if f.get("type") == "file" and f["path"].lower().endswith(".gguf")
      and "mmproj" not in f["path"].lower()]
size = lambda f: (f.get("lfs") or {}).get("size") or f.get("size") or 0
gb = lambda n: f"{n / 1e9:.1f} GB"
if not gg:
    print("this repo has no .gguf files. llama.cpp needs GGUF; look for a '-GGUF' conversion of the model.")
    sys.exit(2)
want = quant or "q4_k_m"
match = [f for f in gg if want in f["path"].lower()]
if not match:
    listing = "\n".join(f"    {f['path']}  ({gb(size(f))})" for f in sorted(gg, key=lambda f: f["path"])[:20])
    if quant:
        print(f"no file matches quantization '{quant}'. Files in the repo:\n{listing}")
        sys.exit(2)
    print(f"note: no Q4_K_M here, llama.cpp will choose. Files in the repo:\n{listing}")
    sys.exit(0)
total = sum(size(f) for f in match)
names = ", ".join(f["path"].split("/")[-1] for f in match[:3]) + (" ..." if len(match) > 3 else "")
print(f"{len(match)} file(s), {gb(total)}: {names}")
code = 0
if vram_mb and total / 1048576 + 2000 > vram_mb:
    print(f"WARN: the model is {gb(total)} and the GPU has {vram_mb / 1024:.1f} GB in total. With context it probably "
          "won't fit: expect out-of-memory or very slow CPU offload. Try a smaller quantization.")
if disk and total + 5e9 > disk:
    print(f"DISK: {gb(total)} to download but only {gb(disk)} free (checked on the Windows C: drive too: a full drive corrupts WSL). Free some space first.")
    code = 3
print(f"SIZE_BYTES={total}")
sys.exit(code)
PY
)"; rc=$?
  set -e
  TOTAL_BYTES="$(printf '%s\n' "$out" | sed -n 's/^SIZE_BYTES=//p' | tail -n 1)"; TOTAL_BYTES="${TOTAL_BYTES:-0}"
  printf '%s\n' "$out" | grep -v '^SIZE_BYTES=' | sed 's/^/      /' || true
  case "$rc" in
    0) ;;
    3) [ -n "${FORCE:-}" ] || die "not enough disk space (FORCE=1 overrides)." ;;
    *) [ -n "${FORCE:-}" ] || die "can't use that model as given (FORCE=1 overrides)." ;;
  esac
}

# ---- [1/4] check
CACHED=""
if [ -n "$LOCAL" ]; then
  step 1 "Using the local file $LOCAL"
elif CACHED="$(find_cached)" && [ -n "$CACHED" ]; then
  step 1 "$LABEL is already downloaded, so no download or network access is needed:"
  echo "      $CACHED"
elif [ -n "${SKIP_CHECK:-}" ]; then
  step 1 "Skipping the Hugging Face check (SKIP_CHECK=1)"
else
  step 1 "Not downloaded yet. Checking $REPO ${QUANT:+($QUANT) }on Hugging Face ..."
  preflight
fi

# ---- build the service file (a cached model is loaded straight from disk)
if [ -n "$LOCAL" ]; then MODEL_ARGS="-m \"$LOCAL\""
elif [ -n "$CACHED" ]; then MODEL_ARGS="-m \"$CACHED\""
else MODEL_ARGS="-hf $SPEC"; fi
UNIT="[Unit]
Description=llama-server ($LABEL)

[Service]
EnvironmentFile=-%h/.config/llama-server.env
ExecStart=$LLAMA $MODEL_ARGS --jinja -ngl 99 -c $TOTAL -np $NP --host 127.0.0.1 --port $PORT ${EXTRA:-}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target"

if [ -n "${DRY_RUN:-}" ]; then echo; echo "$UNIT"; exit 0; fi

# ---- [2/4] restart
step 2 "Restarting the server ($NP slot(s) x $PER_SLOT tokens = $TOTAL total)"
# deploy/llama.service is what ~/.config/systemd/user/llama.service links to
mkdir -p deploy
printf '%s\n' "$UNIT" > deploy/llama.service
systemctl --user daemon-reload
systemctl --user restart llama

# ---- after loading: is it actually fast, and is there VRAM headroom? (a GPU that is full
# makes the driver spill into system RAM, which looks like a hung server)
speed_check() {
  local used total reply tps
  echo "Speed check (a short test generation; NO_CHECK=1 skips it) ..."
  reply="$(curl -s -m 90 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
           -d '{"messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":48}' || true)"
  tps="$(printf '%s' "$reply" | python3 -c 'import sys, json
try: print(round(json.load(sys.stdin)["timings"]["predicted_per_second"], 1))
except Exception: pass' 2>/dev/null || true)"
  used="$(gpu_mib used)"; total="$(gpu_mib total)"
  if [ -z "$tps" ]; then
    warn "no answer within 90 s: far too slow to be usable. Almost certainly the GPU is out of memory and spilling into system RAM."
  else
    echo "      generating at $tps tokens/s"
  fi
  if [ -n "$used" ] && [ -n "${total:-}" ]; then
    echo "      GPU memory: $(awk -v u="$used" -v t="$total" 'BEGIN { printf "%.1f / %.1f GB (%d%%)", u / 1024, t / 1024, u * 100 / t }')"
    if [ $((used * 100 / total)) -ge 93 ]; then
      warn "the GPU is over 93% full. Once the agent's context grows it will likely spill into system RAM and crawl. Try: NP=1 CTX=16384 EXTRA=\"-ctk q8_0 -ctv q8_0\" ./model.sh $LABEL"
    fi
  fi
  if [ -n "$tps" ] && awk -v t="$tps" 'BEGIN { exit !(t < 10) }'; then
    warn "only $tps tokens/s: this model is probably too big for the GPU as configured (see the memory line above)."
  fi
}

# ---- progress helpers
blobs_dir="$HUB/models--${REPO//\//--}/blobs"
dir_bytes() {   # bytes in the model's download folder (finished and in-progress files)
  local b=0 f
  for f in "$blobs_dir"/*; do [ -f "$f" ] && b=$((b + $(stat -c %s "$f"))); done
  echo "$b"
}
downloading() { compgen -G "$HUB/models--*/blobs/*.downloadInProgress" >/dev/null; }
tty_out=0; [ -t 1 ] && tty_out=1
show() {        # overwrite one line on a terminal; a plain line otherwise (every 10 s)
  if [ "$tty_out" = 1 ]; then
    local cols; cols="$(tput cols 2>/dev/null || echo 80)"
    printf '\r\033[K      %s' "${1:0:$((cols - 8))}"      # never wrap, or the line can't be overwritten
  elif (( ${2:-0} % 5 == 0 )); then echo "      $1"; fi
}
endline() { [ "$tty_out" = 1 ] && echo; return 0; }
fmt_eta() { awk -v s="$1" 'BEGIN { s = int(s); printf "%d:%02d", s / 60, s % 60 }'; }

# ---- [3/4] download and [4/4] load: one loop that reports whichever is happening
NEED_DL=0; [ -z "$CACHED$LOCAL" ] && NEED_DL=1
if [ "$NEED_DL" = 1 ]; then step 3 "Downloading $LABEL (llama-server fetches it on start)"
else step 3 "Download: skipped, nothing to fetch"; step 4 "Loading the model into the GPU"; fi

# While loading, GPU memory fills up to roughly the model's file size: that gives a percentage.
if [ -n "$LOCAL" ]; then model_file="$LOCAL"; else model_file="$CACHED"; fi
if [ -n "$model_file" ] && [ -f "$model_file" ]; then expect_bytes="$(stat -L -c %s "$model_file")"
else expect_bytes="$TOTAL_BYTES"; fi
vram_min=""

base="$(dir_bytes)"; prev=$base; prev_t=$SECONDS; speed=0; phase=$([ "$NEED_DL" = 1 ] && echo dl || echo load)
t0=$SECONDS; load_t0=$SECONDS
for ((i = 0; i * 2 < WAIT; i++)); do
  if server_ready; then
    endline; echo; echo "Ready. The model is loaded and serving."; status
    [ -n "${NO_CHECK:-}" ] || speed_check
    echo
    echo "Tip: start a fresh conversation with the new model:  ./run.sh --new --chat"
    exit 0
  fi
  if ! systemctl --user is-active --quiet llama; then
    endline; echo "Server stopped. Last log lines:"
    journalctl --user -u llama -n 25 --no-pager
    exit 1
  fi
  if [ "$phase" = dl ] && ! downloading && [ "$(( $(dir_bytes) - base ))" -gt 0 -o $((SECONDS - t0)) -gt 20 ]; then
    endline; echo "      download finished."; step 4 "Loading the model into the GPU"
    phase=load; load_t0=$SECONDS
  fi
  if [ "$phase" = dl ]; then
    now="$(dir_bytes)"; dt=$((SECONDS - prev_t))
    if [ "$dt" -ge 2 ]; then
      speed="$(awk -v n="$now" -v p="$prev" -v d="$dt" -v s="$speed" 'BEGIN { r = (n - p) / d; print (s == 0 ? r : 0.7 * s + 0.3 * r) }')"
      prev=$now; prev_t=$SECONDS
    fi
    got=$((now - base)); [ "$got" -lt 0 ] && got=0
    if [ "$TOTAL_BYTES" -gt 0 ]; then
      line="$(awk -v g="$got" -v t="$TOTAL_BYTES" -v s="$speed" 'BEGIN {
          p = g * 100 / t; if (p > 99.9) p = 99.9
          printf "%5.1f%%  %.1f / %.1f GB  %.0f MB/s", p, g / 1e9, t / 1e9, s / 1e6 }')"
      [ "${speed%.*}" -gt 0 ] 2>/dev/null && line="$line  ETA $(fmt_eta "$(awk -v g="$got" -v t="$TOTAL_BYTES" -v s="$speed" 'BEGIN { print (t - g) / s }')")"
    else
      line="$(awk -v g="$got" -v s="$speed" 'BEGIN { printf "%.1f GB so far  %.0f MB/s", g / 1e9, s / 1e6 }')"
    fi
    show "$line" "$i"
  else
    used="$(gpu_mib used)"; total_gpu="$(gpu_mib total)"; line="loading... $((SECONDS - load_t0))s"
    if [ -n "$used" ] && [ "${expect_bytes:-0}" -gt 0 ]; then
      if [ -z "$vram_min" ] || [ "$used" -lt "$vram_min" ]; then vram_min=$used; fi   # low point = old model freed
      line="$line  $(awk -v u="$used" -v m="$vram_min" -v e="$expect_bytes" -v t="${total_gpu:-0}" 'BEGIN {
          p = (u - m) * 1048576 * 100 / e; if (p > 99) p = 99; if (p < 0) p = 0
          printf "~%d%%  GPU %.1f / %.1f GB", p, u / 1024, t / 1024 }')"
    fi
    show "$line" "$i"
  fi
  sleep 2
done
endline; echo "Timed out after ${WAIT}s waiting for the server. Check: journalctl --user -u llama -f"
exit 1
