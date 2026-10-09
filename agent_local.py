"""
Local-LLM agent sandboxed with bubblewrap, talking to a llama.cpp server.

One file, two roles
-------------------
HOST role (what YOU run):
    python agent_local.py [--new] [--chat] ["goal"]
  * Commands from the model run inside a bwrap jail: read-only system, no network,
    only <repo>/workspace writable (as /workspace).
  * A unix-socket bridge (socat) exposes ONLY your llama-server to the jail, so the
    model can verify code by making real LLM calls without getting internet access.
  * On first run this file copies itself to <repo>/workspace/agent_dev.py. The model
    edits THAT copy, never the file that builds the jail.

DEV role (what the MODEL runs, inside the jail, to test its edits):
    SANDBOX_MODE=none python /workspace/agent_dev.py ...
  * "none" = already inside the jail, so run commands directly (no nested bwrap).

Review and promote the model's work yourself:
    python agent_local.py --diff                      # show diff host vs dev copy
    cp workspace/agent_dev.py agent_local.py  # only after reading the diff
    python agent_local.py --sync                      # reset dev copy from host file

One-time setup (Arch):
    sudo pacman -S bubblewrap socat python-openai
    llama-server -m model.gguf --jinja -c 16384 -np 2 --port 8080
      (--jinja is needed for tool calling; -np 2 lets the outer agent and the model's
       inner test run share the server without queueing.)

Env vars: LLM_URL, LLM_MODEL, LLAMA_PORT, WORKSPACE, SANDBOX_MODE (bwrap|docker|none),
          EXPOSE_LLAMA (1|0), MAX_STEPS, SESSION_FILE, LOG_FILE
"""
import atexit
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from openai import APIConnectionError, APIStatusError, OpenAI

# --------------------------------------------------------------------------- config
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLM_URL = os.environ.get("LLM_URL", f"http://127.0.0.1:{LLAMA_PORT}/v1")
MODEL = os.environ.get("LLM_MODEL", "local")   # llama-server mostly ignores this
SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "bwrap")   # "bwrap" | "docker" | "none"
CONTAINER = os.environ.get("SANDBOX", "agent-sandbox")   # docker mode only
WORKSPACE = os.path.abspath(os.path.expanduser(os.environ.get(
    "WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace"))))
EXPOSE_LLAMA = os.environ.get("EXPOSE_LLAMA", "1") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))
LOG_FILE = os.environ.get("LOG_FILE", "agent_run.jsonl")
SESSION_FILE = os.environ.get("SESSION_FILE", "session.json")   # kept outside the sandbox
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300

DEV_NAME = "agent_dev.py"
HOST_FILE = os.path.abspath(__file__)
DEV_COPY = os.path.join(WORKSPACE, DEV_NAME)

os.makedirs(WORKSPACE, exist_ok=True)

# Set by start_llama_bridge(); read by sandbox_cmd().
SOCK_DIR: str | None = None

# --------------------------------------------------------------------------- prompts
def build_system() -> str:
    s = """You are an autonomous agent working toward a goal inside a Linux sandbox.
Work step by step. Each turn, call exactly ONE tool, read the result, then decide
the next step. If a command fails, read the error and try something different.
When the goal is achieved (or impossible), call `finish` with a summary.
Never reply with plain text only; always call a tool.
Your working directory is /workspace. Prefer small targeted edits (sed -i, or a short
python patch script) over rewriting whole files."""
    if SANDBOX_MODE == "bwrap" and EXPOSE_LLAMA:
        s += f"""

A llama.cpp server (OpenAI-compatible, base URL http://127.0.0.1:{LLAMA_PORT}/v1) is
reachable from your commands. There is NO other network access.

If asked to improve the agent script, edit /workspace/{DEV_NAME} (never anything else),
and verify every change before calling finish:
  1. python -m py_compile /workspace/{DEV_NAME}
  2. A real end-to-end run (pass a larger timeout, it makes LLM calls):
     run_shell with timeout=120 and command:
       cd /workspace && SANDBOX_MODE=none LLM_URL=http://127.0.0.1:{LLAMA_PORT}/v1 \\
       SESSION_FILE=/tmp/s.json LOG_FILE=/tmp/l.jsonl WORKSPACE=/workspace MAX_STEPS=6 \\
       python {DEV_NAME} --new "run: echo hello, then finish"
     It passes if the output shows [DONE] and no Traceback.
Report what you changed and what the test showed."""
    return s


SYSTEM = build_system()

TOOLS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "Return the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": ("Run a bash command inside the sandbox. Default timeout 30s; "
                        f"set `timeout` (seconds, max {MAX_TIMEOUT}) for slow commands."),
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"},
                                      "timeout": {"type": "integer"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Call when the goal is complete.",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]


# --------------------------------------------------------------------------- sandbox
def truncate(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def start_llama_bridge() -> None:
    """Host side: forward a private unix socket to the llama.cpp TCP port."""
    global SOCK_DIR
    if shutil.which("socat") is None:
        sys.exit("socat not found on the host. Install it: sudo pacman -S socat")
    SOCK_DIR = tempfile.mkdtemp(prefix="agent-sock-")      # mkdtemp is mode 0700
    sock = os.path.join(SOCK_DIR, "llama.sock")
    p = subprocess.Popen(
        ["socat", f"UNIX-LISTEN:{sock},fork,mode=600", f"TCP:127.0.0.1:{LLAMA_PORT}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(p.terminate)
    atexit.register(shutil.rmtree, SOCK_DIR, True)
    for _ in range(20):                                    # wait up to ~2s for the socket
        if os.path.exists(sock):
            return
        time.sleep(0.1)
    print("[warn] llama bridge socket did not appear; the sandbox won't reach llama-server")


def sandbox_cmd(command: str) -> list[str]:
    """Build the argv that runs `command` inside the sandbox."""
    if SANDBOX_MODE == "none":          # we're already inside the jail (dev/test role)
        return ["bash", "-c", command]
    if SANDBOX_MODE == "docker":
        return ["docker", "exec", CONTAINER, "bash", "-lc", command]

    extra: list[str] = []
    if EXPOSE_LLAMA and SOCK_DIR:
        # Only the socket dir is exposed. Inside, socat re-listens on loopback:PORT.
        extra = ["--ro-bind", SOCK_DIR, "/run/llama"]
        command = (
            f"socat TCP-LISTEN:{LLAMA_PORT},bind=127.0.0.1,fork,reuseaddr "
            f"UNIX-CONNECT:/run/llama/llama.sock >/dev/null 2>&1 &\n"
            f"sleep 0.3\n{command}"
        )

    # bubblewrap: read-only system, writable /workspace only. --unshare-all cuts off
    # network, other processes, IPC, etc. (Deliberately NO --share-net.)
    return [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/bin", "/sbin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", WORKSPACE, "/workspace",
        *extra,
        "--chdir", "/workspace",
        "--setenv", "HOME", "/workspace",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "bash", "-c", command,
    ]


def execute(name: str, args: dict) -> str:
    try:
        if name == "get_time":
            return datetime.now().isoformat()
        if name == "run_shell":
            try:
                timeout = int(args.get("timeout") or DEFAULT_TIMEOUT)
            except (TypeError, ValueError):
                timeout = DEFAULT_TIMEOUT
            timeout = max(1, min(timeout, MAX_TIMEOUT))
            r = subprocess.run(
                sandbox_cmd(args["command"]),
                capture_output=True, text=True, timeout=timeout,
            )
            return truncate(f"exit code: {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
        return f"ERROR: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- dev copy
def sync_dev_copy(force: bool = False) -> None:
    """Put a copy of this file in the workspace for the model to edit."""
    if force or not os.path.exists(DEV_COPY):
        shutil.copyfile(HOST_FILE, DEV_COPY)
        print(f"[dev copy] {'reset' if force else 'created'}: {DEV_COPY}")


def show_diff() -> None:
    if not os.path.exists(DEV_COPY):
        sys.exit("No dev copy yet. Run the agent once (or use --sync).")
    with open(HOST_FILE) as a, open(DEV_COPY) as b:
        diff = list(difflib.unified_diff(a.readlines(), b.readlines(),
                                         "agent_local.py (host)", f"{DEV_NAME} (model)"))
    sys.stdout.writelines(diff or ["No differences.\n"])


# --------------------------------------------------------------------------- logging / session
def log(entry: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_session() -> list:
    """Resume the previous conversation, or start a fresh one."""
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)
    return [{"role": "system", "content": SYSTEM}]


def save_session(messages: list) -> None:
    with open(SESSION_FILE, "w") as f:
        json.dump(messages, f)


# --------------------------------------------------------------------------- agent loop
def run_goal(client: OpenAI, messages: list, goal: str) -> None:
    """Run one goal. `messages` is the running history, so earlier work stays in context."""
    messages.append({"role": "user", "content": f"GOAL: {goal}"})
    log({"event": "start", "goal": goal, "model": MODEL})

    for step in range(1, MAX_STEPS + 1):
        print(f"\n===== step {step} =====")
        resp = None
        for attempt in range(1, 4):   # local servers sometimes fail to parse a model's output
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=TOOLS,
                    temperature=(0.2, 0.7, 1.0)[attempt - 1],   # vary the sample on retries
                )
                break
            except (APIStatusError, APIConnectionError) as e:
                print(f"[llm error] attempt {attempt}/3: {e}")
                log({"event": "llm_error", "step": step, "attempt": attempt, "error": str(e)})
        if resp is None:
            print(f"[STOPPED] the model server kept failing; see {LOG_FILE}")
            save_session(messages)
            return
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            print(f"[reason] {msg.content.strip()}")

        # Small local models sometimes forget to call a tool; nudge them.
        if not msg.tool_calls:
            messages.append({"role": "user",
                             "content": "You must call a tool. Use run_shell to act or finish if done."})
            continue

        finished = False
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None

            if args is None:
                observation = "ERROR: tool arguments were not valid JSON. Try again."
            elif name == "finish":
                print(f"\n[DONE] {args.get('summary')}")
                log({"event": "finish", "step": step, "summary": args.get("summary")})
                observation = "Goal marked complete. Wait for the next goal."
                finished = True
            else:
                print(f"[act] {name}({args})")
                observation = execute(name, args)

            print(f"[observe] {truncate(observation, 500)}")
            log({"event": "step", "step": step, "tool": name, "args": args, "observation": observation})
            # Every tool call must get an answer, or the history is invalid next turn.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

        save_session(messages)
        if finished:
            return

    print(f"\n[STOPPED] hit MAX_STEPS={MAX_STEPS}")
    log({"event": "max_steps"})
    save_session(messages)


# --------------------------------------------------------------------------- entry point
def main() -> None:
    args = sys.argv[1:]

    if "--diff" in args:
        show_diff()
        return

    def pop(flag: str) -> bool:
        if flag in args:
            args.remove(flag)
            return True
        return False

    new, chat, sync = pop("--new"), pop("--chat"), pop("--sync")

    if new and os.path.exists(SESSION_FILE):          # start a fresh conversation
        os.remove(SESSION_FILE)

    if SANDBOX_MODE == "bwrap":
        sync_dev_copy(force=sync)
        if EXPOSE_LLAMA:
            start_llama_bridge()
    elif sync:
        print("[note] --sync only applies in bwrap mode")

    client = OpenAI(base_url=LLM_URL, api_key="not-needed-for-local")
    messages = load_session()
    print(f"[session] {len(messages) - 1} earlier messages loaded" if len(messages) > 1
          else "[session] new conversation")

    if args:
        run_goal(client, messages, " ".join(args))
    if chat:
        while True:
            try:
                goal = input("\ngoal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if goal in ("exit", "quit"):
                break
            if goal:
                run_goal(client, messages, goal)
    elif not args:
        sys.exit('usage: python agent_local.py [--new] [--chat] [--sync] [--diff] ["your goal"]')


if __name__ == "__main__":
    main()
