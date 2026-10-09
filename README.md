# worshipper

A small autonomous agent that runs against a **local LLM** and executes shell commands inside a **bubblewrap sandbox**.

```
GOAL -> REASON -> CHOOSE ACTION -> EXECUTE (sandboxed) -> OBSERVE RESULT -> REASON -> ...
```

The model runs on a local GPU via `llama-server` (llama.cpp). The agent talks to it over an OpenAI-compatible HTTP API, so any server that speaks that API will work.

## Architecture

```
Dev laptop (Arch)  --ssh-->  Windows PC (RTX 4090)
                                 └─ WSL2 (Arch)
                                     ├─ llama-server   (systemd user service, 127.0.0.1:8080)
                                     ├─ agent_local.py (the agent loop)
                                     └─ bwrap jail     (agent commands run here; only ./workspace is writable)
```

Everything runs inside WSL. The laptop is just an SSH terminal.

## Requirements

| Component | Notes |
|---|---|
| NVIDIA GPU + Windows driver | The driver lives on Windows. **Do not install an NVIDIA driver inside WSL.** `nvidia-smi` must work in WSL. |
| WSL2 with systemd | `/etc/wsl.conf` needs `[boot]` / `systemd=true` |
| CUDA toolkit (in WSL) | `sudo pacman -S cuda` (needed to build llama.cpp with GPU support) |
| llama.cpp | Built with `-DGGML_CUDA=ON` |
| bubblewrap | `sudo pacman -S bubblewrap` |
| Python 3 + `openai` client | `sudo pacman -S python-openai` (only a client library; nothing is sent to OpenAI) |
| ~20 GB free disk | On the **Windows drive that holds the WSL disk**. A full `C:` corrupts the WSL filesystem. |

## Setup

### 1. Check GPU passthrough

```bash
nvidia-smi        # should list the GPU
```

### 2. Build llama.cpp with CUDA

```bash
git clone https://github.com/ggml-org/llama.cpp ~/repos/llama.cpp
cd ~/repos/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Optional: add `~/repos/llama.cpp/build/bin` to your `PATH`.

### 3. Run the model server

Quick test in a terminal (downloads the model on first run, ~13 GB):

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF \
  --jinja -ngl 99 -c 32768 -np 1 --host 127.0.0.1 --port 8080
```

| Flag | Purpose |
|---|---|
| `--jinja` | Use the model's own chat template. Required for tool calling. |
| `-ngl 99` | Put all layers on the GPU. |
| `-c 32768` | Context size in tokens. |
| `-np 1` | One slot, so the agent gets the whole context. |
| `--host 127.0.0.1` | Local only. Don't expose this to the network. |

Verify:

```bash
curl http://localhost:8080/v1/models
```

### 4. Run the server as a service (recommended)

```bash
mkdir -p ~/.config/systemd/user
ln -sf "$PWD/deploy/llama.service" ~/.config/systemd/user/llama.service
systemctl --user daemon-reload
systemctl --user enable --now llama
sudo loginctl enable-linger "$USER"      # keep it running with no terminal open
```

Useful commands:

```bash
systemctl --user status llama
systemctl --user restart llama
journalctl --user -u llama -f            # live server log
```

The unit's `ExecStart` must use absolute paths. If the service can't find CUDA libraries, add `Environment=LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64` under `[Service]`.

### 5. Install agent dependencies

```bash
sudo pacman -S bubblewrap python-openai
```

Check the sandbox works:

```bash
./jail.sh 'echo jail works; ls /home'    # "jail works", then an error: /home is hidden
```

## Running the agent

```bash
./run.sh "Create primes.py that prints the first 20 primes, run it, and report the output"
```

The agent keeps its conversation between runs (saved in `session.json`), so you can keep building on earlier work:

```bash
./run.sh "Now change it to print the first 50"
./run.sh --chat                 # interactive: type goals one after another, "exit" to quit
./run.sh --new "fresh goal"     # discard history and start a new conversation
./reset.sh                      # wipe workspace/ and session.json for a clean slate
```

Always check the results yourself (`ls workspace`, `cat workspace/...`) instead of trusting the agent's summary.

### Configuration (environment variables)

Set in `run.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_URL` | `http://localhost:11434/v1` | Server base URL (`run.sh` sets `http://localhost:8080/v1`) |
| `LLM_MODEL` | `qwen2.5-coder:14b` | Model name (a label for llama-server; matters for Ollama) |
| `WORKSPACE` | `~/agent-workspace` | Folder the agent can write to (appears as `/workspace` in the jail) |
| `SANDBOX_MODE` | `bwrap` | `bwrap` or `docker` |
| `SESSION_FILE` | `session.json` | Where the conversation history is saved |

Hard-coded in `agent_local.py`: `MAX_STEPS = 25`, 30 s command timeout, 3 retries on server errors.

## Sandbox

Agent commands run inside bubblewrap with:

- System directories mounted **read-only**
- Only `workspace/` writable (as `/workspace`)
- **No network** (`--unshare-all`); add `--share-net` in `sandbox_cmd()` only when you deliberately want internet
- Your home directory and Windows drives (`/mnt/c`) **not visible**

This is a sensible first layer, not a hard security boundary. Run everything as a normal user (never `sudo`), and don't mount sensitive directories into the jail.

## Logs and files

| File | Purpose |
|---|---|
| `agent_run.jsonl` | Event log: goals, every tool call and observation, LLM errors, finish events |
| `session.json` | Saved conversation history (what makes multi-run memory work) |
| `workspace/` | The agent's working directory |

Both log files are gitignored.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `curl localhost:8080` fails | Server isn't running. Check `systemctl --user status llama` and `journalctl --user -u llama -n 50`. |
| `500 ... does not match the expected peg-native format` | llama-server couldn't parse the model's output (gpt-oss uses the harmony format). The agent retries automatically. If frequent: update and rebuild llama.cpp, lower the reasoning effort, or try another model. |
| Tool calls appear as plain text | `--jinja` missing from the server command. |
| `Bus error` / `Input/output error` / download fails midway | Out of disk space on the Windows drive. Free space, run `wsl --shutdown`, retry. |
| Generation very slow | Model spilled out of VRAM. Lower `-c`, or check the log for layers offloaded to GPU. |
| Agent forgets earlier work or acts oddly | Conversation exceeded the context window. Use `--new`. |
| bwrap errors about namespaces | Run the one-line jail test and check WSL kernel support for user namespaces. |

## Layout

```
agent_local.py     the agent loop
run.sh             launcher with settings
reset.sh           wipe workspace + session
jail.sh            manual sandbox shell for experiments
deploy/            systemd unit for llama-server
workspace/         agent sandbox (contents gitignored)
notes/             observations, failure patterns
```

## Roadmap

1. Stable baseline: run simple tasks, note failures
2. Better logging (full transcripts, finish reason)
3. Automated test suite: tasks with script-verified checks, multiple trials per task
4. Compare models and settings on the same suite
5. Improve prompts, tool output, context handling (one change at a time)
6. Later: memory, planning, and fine-tuning on logged successful runs


### OLD
PowerShell:
    New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -Protocol TCP -LocalPort 2222 -RemoteAddress LocalSubnet -Action Allow

laptop:
    ssh -p 2222 <wsl_username>@WINDOWS_IP
    sftp -P 2222 <wsl_username>@WINDOWS_IP
