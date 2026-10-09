#!/usr/bin/env bash
cd "$(dirname "$0")"
export WORKSPACE="$PWD/workspace"
export LLM_URL=http://localhost:8080/v1
export LLM_MODEL=gpt-oss-20b
exec python agent_local.py "$@"
