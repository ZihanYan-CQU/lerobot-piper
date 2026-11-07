#!/usr/bin/env bash
# Compatibility entry point: discovery, validation, and activation live in Python.
set -euo pipefail
DEVICE_BINDINGS_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( EUID != 0 )); then
    exec sudo bash "$DEVICE_BINDINGS_DIR/can_activate.sh" "$@"
fi
CAN_ARGUMENTS=(--interface "${1:-can0}" --bitrate "${2:-1000000}")
if [[ -n "${3:-}" ]]; then
    CAN_ARGUMENTS+=(--usb "$3")
fi
exec /usr/bin/python3 "$DEVICE_BINDINGS_DIR/can_port_bindings.py" "${CAN_ARGUMENTS[@]}"
