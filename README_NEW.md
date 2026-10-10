# agent_local

A local-LLM agent that runs its shell commands inside a sandbox. It talks to a
[llama.cpp](https://github.com/ggerganov/llama.cpp) server and can optionally
collaborate with "friends" (other agents). Run `python agent_local.py -h` for the
flags; everything else is here.

## Setup (Arch)

```bash
sudo pacman -S bubblewrap socat python-openai
llama-server -m model.gguf --jinja -c 16384 -np 2 --port 8080
```

- `--jinja` is required for tool calling.
- `-np 2` lets the outer agent and the model's inner test run share the server
  without queueing. Friends are called one at a time, so 2 slots is still enough.

## Quick start

```bash
python agent_local.py "list the files in /workspace and summarize them"
python agent_local.py --new --chat            # interactive, fresh conversation
python agent_local.py --friends "goal"        # let the agent use friends
python agent_local.py --debug "goal"          # show every step
```

The agent works **solo by default**. Friends only exist when you pass `--friends`
(or set `USE_FRIENDS=1`).

## Output

Quiet by default: stdout gets only the agent's final result, so it is safe to pipe
(`python agent_local.py "..." > answer.txt`). If a run doesn't finish (step limit,
model server failing) a short explanation goes to stderr instead. Warnings such as an
unreadable `friends.json` also go to stderr.

`--debug` (or `DEBUG=1`) shows everything as it happens: the model's reasoning, each
tool call and its result, and friend activity (indented for workers). Steps are always
written to `LOG_FILE`, even in quiet mode, so you can inspect a run afterwards.

## How it works

Each turn the model calls exactly one tool, sees the result, and continues until it
calls `finish` or reaches `MAX_STEPS`.

| Tool | What it does |
|------|--------------|
| `get_time` | current date and time |
| `run_shell` | run a bash command in the sandbox (default 30 s timeout, max 300 s) |
| `finish` | end the run with a summary |
| `ask_friend` | ask an advisor (only with `--friends`) |
| `delegate` | hand a task to a worker (only with `--friends`) |

The conversation is saved to `SESSION_FILE` after every step, so the next run
resumes it. Use `--new` to start over. Every step is also appended to `LOG_FILE`
as JSON lines.

## Sandbox

`SANDBOX_MODE` picks where commands run:

| Mode | Behavior |
|------|----------|
| `bwrap` (default) | bubblewrap jail: read-only system, no network, only `<repo>/workspace` writable (seen as `/workspace`) |
| `docker` | `docker exec` into an existing container (`SANDBOX`, default `agent-sandbox`) |
| `none` | run directly. This is what the model uses *inside* the jail to test its own edits. Don't use it on your host unless you trust the model. |

With `EXPOSE_LLAMA=1` (bwrap mode), a `socat` unix-socket bridge exposes **only**
your llama-server to the jail, on `127.0.0.1:LLAMA_PORT`. The model can make real LLM
calls to verify its code but has no other network access.

## Friends

Friends are extra "minds" the main agent can collaborate with. Turn them on with
`--friends`; see them with `--list-friends`. They use the same llama-server by
default, and each can point at its own url/model.

**Advisors** (`ask_friend(friend, message)`): text only, no tools. Each keeps a
persistent conversation with the agent, saved next to `SESSION_FILE` as
`<name>.friends.json` and trimmed to the last 12 messages.
Built in: `planner`, `reviewer`, `tester`.

**Workers** (`delegate(friend, task)`): full sub-agents with their own system prompt,
tool loop, shell (same `/workspace`) and step limit. Each task starts fresh and
returns the worker's `finish` summary. Built in: `coder`, `inspector` (read-only).

- Friends can't see the main agent's conversation; it has to pass context along.
- Workers can't delegate further unless `MAX_DEPTH` allows it **and** their `tools`
  list contains `"delegate"`. `MAX_DEPTH=0` disables delegation entirely.
- A `readonly` worker gets `/workspace` mounted read-only. This is enforced in bwrap
  mode only; in docker/none mode it is just told not to modify files.
- Friend LLM calls run on the host, so a friend's own `url` needs no bridge.

### Customizing friends

Create `friends.json` (or point `FRIENDS_FILE` at another path). An entry replaces a
built-in friend of the same name; `null` removes one:

```json
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
```

| Field | Applies to | Notes |
|-------|-----------|-------|
| `kind` | both | `advisor` or `worker` (required) |
| `prompt` | both | the friend's system prompt (required) |
| `desc` | both | one line shown to the main agent |
| `tools` | worker | any of `run_shell`, `get_time`, `ask_friend`, `delegate`; `finish` is always added. Default: `run_shell`, `get_time` |
| `max_steps` | worker | default 12 |
| `readonly` | worker | default false |
| `url`, `model` | both | use a different llama-server / model |

Invalid entries are skipped with a warning.

## Letting the model improve the agent

On first run in bwrap mode, `agent_local.py`, `agent/` and `tests/` are copied to
`workspace/agent_dev/`. If you ask the agent to improve itself, it edits **that copy**
and never the code that builds the jail. It is told to verify every change with
`compileall`, the unit tests, and a real end-to-end run (with `--debug`) before finishing.

You review and promote its work by hand:

```bash
python agent_local.py --diff                                # read the changes
cp -r workspace/agent_dev/{agent,tests,agent_local.py} .    # only after reading the diff
python agent_local.py --sync                                # reset the dev copy from the host code
```

`--sync` wipes `workspace/agent_dev/` and recopies it.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_URL` | `http://127.0.0.1:$LLAMA_PORT/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `local` | model name sent to the server (llama-server mostly ignores it) |
| `LLAMA_PORT` | `8080` | port of your llama-server |
| `WORKSPACE` | `<repo>/workspace` | the one directory writable from the jail |
| `SANDBOX_MODE` | `bwrap` | `bwrap`, `docker` or `none` |
| `SANDBOX` | `agent-sandbox` | docker container name (docker mode only) |
| `EXPOSE_LLAMA` | `1` | `1` = bridge llama-server into the jail, `0` = no access |
| `MAX_STEPS` | `25` | max tool calls per goal for the main agent |
| `SESSION_FILE` | `session.json` | saved conversation (kept outside the sandbox) |
| `LOG_FILE` | `agent_run.jsonl` | JSON-lines log of every step |
| `USE_FRIENDS` | `0` | `1` = same as `--friends` |
| `DEBUG` | `0` | `1` = same as `--debug` |
| `FRIENDS_FILE` | `friends.json` | optional file that adds/replaces friends |
| `MAX_DEPTH` | `1` | how deep delegation may nest; `0` disables `delegate` |

## Project layout

```
agent_local.py          entry point, the only file you run
agent/
  cli.py                argument parsing, -h, main()
  helptext.py           the text printed by -h
  app.py                App: wires everything together, runs goals
  agent.py              Agent: the think -> tool -> observe loop
  tools.py              Tool base class, built-in tools, ToolPool
  friends.py            friend config, FriendService, ask_friend / delegate
  sandbox.py            bwrap / docker / none sandboxes, LlamaBridge
  llm.py                LLM client with retries
  console.py            Console: quiet by default, --debug shows every step
  prompts.py            system prompts
  storage.py            EventLog, SessionStore, FriendMemory
  devcopy.py            the model's editable copy (sync / diff)
  config.py             Config dataclass, read from env vars
  util.py               truncate()
tests/test_agent.py     unit tests
```

Created at runtime: `workspace/` (including `workspace/agent_dev/`), `session.json`,
`session.friends.json`, `agent_run.jsonl`. The session, memory and log files sit
outside `workspace/`, so the model can't read or edit them from inside the jail.

To add a tool, subclass `Tool` in `tools.py` and add an instance in `App.__init__`.
To add a sandbox, subclass `Sandbox` and extend `make_sandbox()`.

## Tests

```bash
python -m unittest discover -s tests -q
```

They use a fake LLM, so they need no llama-server, bwrap or `openai` package.
