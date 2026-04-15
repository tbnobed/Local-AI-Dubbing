#!/usr/bin/env bash
# Convenience wrapper — runs the DubbingStudio setup from the repo root.
exec "$(dirname "$0")/dubbing-studio/scripts/setup.sh" "$@"
