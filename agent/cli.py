"""Command-line interface."""
from __future__ import annotations

import argparse
import dataclasses
import sys

from .helptext import HELP

USAGE = ('usage: python agent_local.py [-h] [--new] [--chat] [--friends] [--list-friends] '
         '[--sync] [--diff] ["your goal"]\nrun with -h for details')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_local.py", add_help=False, allow_abbrev=False)
    p.add_argument("--new", action="store_true")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--friends", action="store_true")
    p.add_argument("--list-friends", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--diff", action="store_true")
    p.add_argument("goal", nargs="*")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    # Checked first, and before anything heavy is imported, so help always works.
    if any(a in ("-h", "--help") for a in argv):
        print(HELP, end="")
        return

    args = build_parser().parse_args(argv)

    from .config import ROOT, Config
    cfg = Config.from_env()
    if args.friends:
        cfg = dataclasses.replace(cfg, use_friends=True)

    if args.diff:
        from .devcopy import DevCopy
        try:
            diff = DevCopy(ROOT, cfg.dev_dir).diff()
        except FileNotFoundError as e:
            sys.exit(str(e))
        sys.stdout.writelines(diff or ["No differences.\n"])
        return

    if args.list_friends:
        from .friends import FriendRegistry
        registry = FriendRegistry.load(cfg.friends_file)
        if registry:
            print(registry.describe(config_file=cfg.friends_file, max_depth=cfg.max_depth,
                                    default_url=cfg.llm_url, default_model=cfg.model))
            print("\nThe agent only uses friends when started with --friends.")
        else:
            print("No friends configured.")
        return

    from .app import App
    app = App(cfg)
    app.setup(new=args.new, sync=args.sync)

    goal = " ".join(args.goal)
    if goal:
        app.run_goal(goal)
    if args.chat:
        app.chat()
    elif not goal:
        sys.exit(USAGE)
