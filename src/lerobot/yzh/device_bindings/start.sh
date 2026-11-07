#!/usr/bin/env bash
# Locate the panel relative to this launcher, regardless of the working directory.
set -euo pipefail
# GTK and device utilities belong to the system environment, not the caller's Conda/venv.
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
DEVICE_BINDINGS_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "$DEVICE_BINDINGS_DIR/camera_aliases.py" --gui "$@"
