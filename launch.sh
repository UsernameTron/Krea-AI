#!/bin/bash
# FLUX.1 Krea — Single launcher script
# Usage: ./launch.sh [command] [options]
#
# Examples:
#   ./launch.sh info                          # System info
#   ./launch.sh web                           # Launch web UI
#   ./launch.sh web --port 8080               # Web UI on custom port
#   ./launch.sh generate -p "a cute cat"      # Generate image
#   ./launch.sh benchmark --quick             # Quick benchmark
#   ./launch.sh                               # Show help

set -euo pipefail

cd "$(dirname "$0")"

# Load environment if available
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

exec python main.py "$@"
