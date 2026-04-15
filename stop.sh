#!/usr/bin/env bash
# Convenience wrapper — stops DubbingStudio from the repo root.
exec "$(dirname "$0")/dubbing-studio/scripts/stop.sh" "$@"
