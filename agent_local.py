#!/usr/bin/env python3
"""Entry point. All the code lives in the `agent/` package; see agent/__init__.py.

    python agent_local.py -h
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make `agent` importable from anywhere

from agent.cli import main

if __name__ == "__main__":
    main()
