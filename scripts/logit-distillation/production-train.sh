#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set -a
source "$SCRIPT_DIR/production.env"
source "$SCRIPT_DIR/data.sh"
set +a
exec python3 "$SCRIPT_DIR/production.py" train "$@"
