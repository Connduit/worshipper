#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export WORKSPACE="${WORKSPACE:-$PWD/workspace}"
export LLM_URL="${LLM_URL:-http://localhost:8080/v1}"
export LLM_MODEL="${LLM_MODEL:-gpt-oss-20b}"

# Launcher options precede the goal or agent options (--new / --chat).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--url)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                printf 'error: %s requires a value\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --model ]]; then
                case "$2" in
                    gpt-oss) export LLM_MODEL=gpt-oss-20b ;;
                    qwen) export LLM_MODEL=qwen3-coder ;;
                    *) export LLM_MODEL="$2" ;;
                esac
            else
                export LLM_URL="$2"
            fi
            shift 2
            ;;
        --help|-h)
            cat <<'EOF'
Usage: ./run.sh [--model gpt-oss|qwen|SERVER_MODEL_ID] [--url BASE_URL] [--new] [--chat] ["goal"]

Defaults: gpt-oss-20b at http://localhost:8080/v1.
Environment overrides: LLM_MODEL, LLM_URL, WORKSPACE.
The matching model must already be loaded in your server; --model does not load it.
EOF
            exit 0
            ;;
        --) shift; break ;;
        *) break ;;
    esac
done

exec python agent_local.py "$@"
