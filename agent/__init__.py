"""Local-LLM agent that runs its shell commands in a sandbox.

Layout (arrows = "depends on"; nothing below imports anything above it)

    cli.py          argument parsing, -h, main()                -> app, helptext, config
    app.py          App: wires everything together, runs goals  -> everything below
    friends.py      FriendSpec/Registry/Service, ask_friend and delegate tools
    agent.py        Agent: the think -> call tool -> observe loop
    tools.py        Tool base class, built-in tools, ToolPool
    sandbox.py      Sandbox classes (bwrap / docker / none) + LlamaBridge
    llm.py          LLM: chat completions with retries
    prompts.py      system prompts
    storage.py      EventLog, SessionStore, FriendMemory
    devcopy.py      DevCopy: the model's editable copy of this code
    config.py       Config: every setting, read from env vars
    util.py         truncate(), say()

To add a tool: subclass Tool in tools.py, add an instance in App.__init__.
To add a sandbox: subclass Sandbox in sandbox.py and extend make_sandbox().
"""
