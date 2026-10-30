#!/usr/bin/env bash
# Locate the panel relative to this launcher, regardless of the working directory.
set -euo pipefail
DEVICE_BINDINGS_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "$DEVICE_BINDINGS_DIR/camera_aliases.py" --gui "$@"
