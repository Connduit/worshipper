"""
Local-LLM agent with a Docker sandbox.

  Arch laptop (this script)  --HTTP-->  Windows PC (Ollama + GPU)
  Commands run in a throwaway Docker container, never on your host.

One-time setup
--------------
Windows PC (the LLM host):
    1. Install Ollama, then:  ollama pull qwen2.5-coder:14b      (or any tool-capable model)
    2. Make it reachable on your LAN:  setx OLLAMA_HOST "0.0.0.0:11434"  (restart Ollama)
       Only allow it on your private network in the firewall.

Sandbox (default: bubblewrap, no Docker needed):
    sudo pacman -S bubblewrap python-openai      # or: pip install openai
    # Commands run inside a bwrap jail: read-only system, no network, no access to
    # your home dir or Windows drives. Only ~/agent-workspace is writable (as /workspace).
    # Set SANDBOX_MODE=docker later to use a container instead (see DOCKER settings below).

Run (inside WSL, where llama-server is running):
    LLM_URL=http://localhost:8080/v1 LLM_MODEL=gpt-oss-20b \
        python agent_local.py "Make a file hello.txt containing the current date, then print it"
"""
import json
import os
import subprocess
import sys
from datetime import datetime

from openai import APIConnectionError, APIStatusError, OpenAI

LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5-coder:14b")
SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "bwrap")   # "bwrap" or "docker"
CONTAINER = os.environ.get("SANDBOX", "agent-sandbox")    # only used in docker mode
WORKSPACE = os.path.expanduser(os.environ.get("WORKSPACE", "~/agent-workspace"))
os.makedirs(WORKSPACE, exist_ok=True)
MAX_STEPS = 25
LOG_FILE = "agent_run.jsonl"   # every step is logged for evaluation (Phase 7)

SYSTEM = """You are an autonomous agent working toward a goal inside a Linux sandbox.
Work step by step. Each turn, call exactly ONE tool, read the result, then decide
the next step. If a command fails, read the error and try something different.
When the goal is achieved (or impossible), call `finish` with a summary.
Never reply with plain text only; always call a tool."""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "Return the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a bash command inside the sandbox container (30s timeout).",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Call when the goal is complete.",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]


def truncate(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def sandbox_cmd(command: str) -> list[str]:
    """Build the argv that runs `command` inside the sandbox."""
    if SANDBOX_MODE == "docker":
        return ["docker", "exec", CONTAINER, "bash", "-lc", command]
    # bubblewrap: read-only system, writable /workspace only, and --unshare-all
    # cuts off network, other processes, IPC, etc. Add "--share-net" to grant internet.
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
            r = subprocess.run(
                sandbox_cmd(args["command"]),
                capture_output=True, text=True, timeout=30,
            )
            return truncate(f"exit code: {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
        return f"ERROR: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def log(entry: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


SESSION_FILE = os.environ.get("SESSION_FILE", "session.json")  # kept outside the sandbox


def load_session() -> list:
    """Resume the previous conversation, or start a fresh one."""
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)
    return [{"role": "system", "content": SYSTEM}]


def save_session(messages: list) -> None:
    with open(SESSION_FILE, "w") as f:
        json.dump(messages, f)


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
            print("[STOPPED] the model server kept failing; see agent_run.jsonl")
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


def main() -> None:
    args = sys.argv[1:]
    if "--new" in args:                      # start a fresh conversation
        args.remove("--new")
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    chat = "--chat" in args
    if chat:
        args.remove("--chat")

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
        sys.exit('usage: python agent_local.py [--new] [--chat] ["your goal"]')


if __name__ == "__main__":
    main()
