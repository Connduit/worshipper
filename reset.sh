#!/usr/bin/env bash
cd /home/binks/repos/worshipper
rm -rf workspace && mkdir workspace
./run.sh "$@"
echo; echo "--- workspace ---"; ls -la workspace
