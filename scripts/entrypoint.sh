#!/bin/sh
# Container entrypoint.
#
# Starts the health probe server in the background, then execs the main
# command (the Streamlit UI by default). SIGTERM/SIGINT sent to the main
# process also stops the probe server.

set -e

HOST="${HEALTH__HOST:-0.0.0.0}"
PORT="${HEALTH__PORT:-8080}"

auto-rag-health --serve --host "$HOST" --port "$PORT" --no-llm &
health_pid=$!
echo "health server listening on ${HOST}:${PORT} (pid ${health_pid})"

trap 'kill "$health_pid" 2>/dev/null || true' EXIT INT TERM

exec "$@"
