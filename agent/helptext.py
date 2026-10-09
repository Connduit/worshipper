"""The text printed by -h."""

HELP = """\
agent_local.py - a local-LLM agent that runs its shell commands in a sandbox.

USAGE
  python agent_local.py [options] ["goal"]

  The goal is everything that is not an option (quotes optional). With a goal the
  agent runs it and exits; with --chat it then keeps prompting for more goals.
  With neither a goal nor --chat, this usage message is shown.

OPTIONS
  -h, --help   Show this help and exit.
  --new        Delete the saved session (and the friends' memory) first and start a
               fresh conversation. Without it, the previous conversation is resumed.
  --chat       Interactive mode: after any initial goal, prompt "goal>" repeatedly.
               Type "exit" or "quit" (or Ctrl-D / Ctrl-C) to leave. History is kept
               across goals, so later goals can build on earlier work.
  --friends    List the configured friends (kind, tools, limits) and exit.
  --sync       Reset the model's dev copy (workspace/agent_dev/) from the host code,
               discarding the model's edits. bwrap mode only.
  --diff       Show a unified diff of the host code vs. the model's dev copy, then
               exit. Use it to review the model's changes before promoting them.

HOW IT WORKS
  * The main agent can call: get_time, run_shell, ask_friend, delegate, finish.
    Each turn it calls one tool, sees the result, and continues until it calls
    finish or hits MAX_STEPS.
  * Sandbox modes (SANDBOX_MODE):
      bwrap   (default) bubblewrap jail: read-only system, no network, only
              <repo>/workspace writable (seen as /workspace).
      docker  run commands via `docker exec` in an existing container (SANDBOX).
      none    run commands directly. Used by the model inside the jail to test
              its own edits; do not use on your host unless you trust the model.
  * With EXPOSE_LLAMA=1 (bwrap mode), a socat unix-socket bridge exposes ONLY your
    llama-server to the jail on 127.0.0.1:LLAMA_PORT. No other network access.
  * On first run agent_local.py, agent/ and tests/ are copied to
    workspace/agent_dev/. The model edits that copy, never the code that builds
    the jail.
  * Session history is saved to SESSION_FILE after each step; every step is also
    appended to LOG_FILE as JSON lines.

FRIENDS
  Friends are extra "minds" the main agent can collaborate with. They are served
  by the same llama-server by default (each can point at its own url/model).
  Built in: planner, reviewer, tester (advisors); coder, inspector (workers).

  advisor  Used through ask_friend(friend, message). Text only, no tools. Has a
           persistent conversation with the agent (saved next to SESSION_FILE as
           <name>.friends.json, trimmed to the last 12 messages).
  worker   Used through delegate(friend, task). A full sub-agent: its own system
           prompt, its own tool loop and shell in the same /workspace, its own step
           limit. Starts fresh for every task, returns its finish() summary.
           Workers cannot delegate further unless MAX_DEPTH allows it AND their
           tools list contains "delegate".
  Friends cannot see the main agent's conversation; it has to give them context.
  A "readonly" worker gets /workspace mounted read-only (enforced in bwrap mode only;
  in docker/none mode it is just told not to modify files).

  Customize with a JSON file (FRIENDS_FILE, default ./friends.json). An entry
  replaces a built-in friend of the same name; null removes one:
    {
      "translator": {"kind": "advisor", "desc": "translates text",
                     "prompt": "You translate between English and German.",
                     "url": "http://127.0.0.1:8081/v1", "model": "other"},
      "builder":    {"kind": "worker", "desc": "builds and runs things",
                     "prompt": "You are a careful builder.",
                     "tools": ["run_shell", "get_time"], "max_steps": 20,
                     "readonly": false},
      "tester": null
    }
  Fields: kind (advisor|worker, required), prompt (required), desc, tools (worker;
  any of run_shell, get_time, ask_friend, delegate; finish is always added),
  max_steps (worker, default 12), readonly (worker), url, model.
  Friend LLM calls run on the host, so a friend's own url needs no bridge.
  Concurrency note: calls are sequential, so llama-server -np 2 is enough.

PROMOTING THE MODEL'S WORK
  python agent_local.py --diff                          # read the changes
  cp -r workspace/agent_dev/{agent,tests,agent_local.py} .   # only after reading the diff
  python agent_local.py --sync                          # reset dev copy from host code

ENVIRONMENT VARIABLES (default)
  LLM_URL       OpenAI-compatible base URL (http://127.0.0.1:$LLAMA_PORT/v1)
  LLM_MODEL     Model name sent to the server; llama-server mostly ignores it (local)
  LLAMA_PORT    Port of your llama-server (8080)
  WORKSPACE     Writable directory shared with the jail (<repo>/workspace)
  SANDBOX_MODE  bwrap | docker | none (bwrap)
  SANDBOX       Docker container name, docker mode only (agent-sandbox)
  EXPOSE_LLAMA  1 = bridge llama-server into the jail, 0 = no access (1)
  MAX_STEPS     Max tool calls per goal for the main agent (25)
  SESSION_FILE  Saved conversation, kept outside the sandbox (session.json)
  LOG_FILE      JSON-lines log of every step (agent_run.jsonl)
  USE_FRIENDS   1 = enable ask_friend/delegate, 0 = main agent works alone (1)
  FRIENDS_FILE  Optional JSON file that adds/replaces friends (friends.json)
  MAX_DEPTH     How deep delegation may nest; 0 disables delegate (1)

EXAMPLES
  python agent_local.py --new "list the files in /workspace and summarize them"
  python agent_local.py --chat
  python agent_local.py --friends
  MAX_STEPS=40 python agent_local.py "improve the agent: add a --version flag"
  USE_FRIENDS=0 python agent_local.py "what time is it?"
  python agent_local.py --diff

ONE-TIME SETUP (Arch)
  sudo pacman -S bubblewrap socat python-openai
  llama-server -m model.gguf --jinja -c 16384 -np 2 --port 8080
    (--jinja enables tool calling; -np 2 lets the agent and the model's inner test
     run share the server without queueing.)
"""
