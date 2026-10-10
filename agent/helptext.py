"""The text printed by -h: just the flags. Everything else lives in README.md."""

HELP = """\
usage: python agent_local.py [options] ["goal"]

Run a local-LLM agent toward a goal. Everything that is not an option is the goal.
With a goal the agent runs it and exits; with --chat it then keeps asking for more.

options:
  -h, --help       Show this help and exit.
  --new            Start a fresh conversation. Deletes the saved session (and the
                   friends' memory). Without it, the previous conversation resumes.
  --chat           Interactive mode: prompt "goal>" after any initial goal.
                   "exit", "quit" or Ctrl-D leaves; history carries across goals.
  --friends        Let the agent use friends (ask_friend / delegate). Off by default:
                   the agent works solo.
  --list-friends   List the available friends and exit.
  --sync           Reset the model's dev copy (workspace/agent_dev/) from the host
                   code, discarding its edits. bwrap mode only.
  --diff           Show host code vs. the model's dev copy, then exit.

examples:
  python agent_local.py "list the files in /workspace and summarize them"
  python agent_local.py --new --chat
  python agent_local.py --friends "add a --version flag, have the reviewer check it"
  python agent_local.py --diff

More (setup, sandbox, friends, environment variables): see README.md
"""
