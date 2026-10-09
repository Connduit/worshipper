#!/usr/bin/env bash
cd /home/binks/repos/worshipper
[ -f workspace/agent_dev.py ] && ./run.sh --diff
rm -rf workspace && mkdir workspace
./run.sh --new "$@"
echo; echo "--- workspace ---"; ls -la workspace
