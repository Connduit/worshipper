"""Command-line interface."""
from __future__ import annotations

import argparse
import dataclasses
import sys

from .helptext import HELP

USAGE = ('usage: python agent_local.py [-h] [--new] [--chat] [--friends] [--list-friends] '
         '[--debug] [--sync] [--diff] [--promote] [--rollback] [--verify] ["your goal"]\nrun with -h for details')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_local.py", add_help=False, allow_abbrev=False)
    p.add_argument("--new", action="store_true")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--friends", action="store_true")
    p.add_argument("--list-friends", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--diff", action="store_true")
    p.add_argument("--promote", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("goal", nargs="*")
    return p


def _self_improvement_command(args, cfg) -> int:
    """--promote / --rollback / --verify: the human's controls over the model's own code changes."""
    from .config import ROOT
    from .sandbox import BwrapSandbox
    from .selfimprove import Promoter, review_and_promote

    if cfg.sandbox_mode != "bwrap":
        print("These commands need SANDBOX_MODE=bwrap: the new code is tested inside the jail.")
        return 1
    sandbox = BwrapSandbox(cfg.workspace)           # no llama bridge needed for the checks
    promoter = Promoter(ROOT, cfg.dev_dir, lambda command, timeout, binds=(): sandbox.run(command, timeout, False, binds))
    if args.rollback:
        restored = promoter.rollback()
        print("Restored: " + ", ".join(restored) if restored else "Nothing to roll back.")
        return 0
    if args.verify:
        ok, text = promoter.selftest()
        print(text)
        return 0 if ok else 1
    return review_and_promote(promoter, interactive=sys.stdin.isatty())


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
    if args.debug:
        cfg = dataclasses.replace(cfg, debug=True)

    if args.diff:
        from .devcopy import DevCopy
        try:
            diff = DevCopy(ROOT, cfg.dev_dir).diff()
        except FileNotFoundError as e:
            sys.exit(str(e))
        sys.stdout.writelines(diff or ["No differences.\n"])
        return

    if args.promote or args.rollback or args.verify:
        sys.exit(_self_improvement_command(args, cfg))

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
    result = None
    if goal:
        result = app.run_goal(goal)
    if args.chat:
        app.chat()
    elif not goal:
        sys.exit(USAGE)
    elif result is not None and result.exit_code:
        sys.exit(result.exit_code)
