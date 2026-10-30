"""Run local DAgger: python -m lerobot.async_inference_local.scripts.dagger."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import suppress
from queue import Empty, Queue
from threading import Thread

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import make_robot_from_config
from lerobot.robots.piper_follower import PIPERFollowerConfig  # noqa: F401
from lerobot.utils.import_utils import _pyrealsense2_available
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.pedal import start_pedal_listener

if _pyrealsense2_available:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from ..runtime.dagger import LocalDagger
from ..shared.dagger_configs import LocalDaggerConfig

HELP = "/start [task] /pause /correct /reset /task TEXT /status /stop"


def _read_commands(commands: Queue) -> None:
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put(None)


def _start_inputs(runtime: LocalDagger, config: LocalDaggerConfig) -> list:
    listeners = []

    def on_key(name: str) -> None:
        if name == config.stop_key:
            runtime.submit_operator_event("stop")
        elif config.input_device == "keyboard":
            if name == config.advance_key:
                runtime.submit_operator_event("advance")
            elif name == config.reset_key:
                runtime.submit_operator_event("reset")

    if config.input_device in {"keyboard", "pedal"}:
        listener = create_key_listener(
            on_key, controls_help="DAgger: Right=advance Left=discard/reset Esc=stop"
        )
        if listener is not None:
            listeners.append(listener)
        elif config.input_device == "keyboard":
            raise RuntimeError("No keyboard input available; use --input_device=prompt in this terminal")
    if config.input_device == "pedal":

        def on_pedal(code: str) -> None:
            if code == config.pedal_advance:
                runtime.submit_operator_event("advance")
            elif code == config.pedal_reset:
                runtime.submit_operator_event("reset")

        listener = start_pedal_listener(on_pedal, device_path=config.pedal_device_path)
        if listener is None:
            for item in listeners:
                item.stop()
            raise RuntimeError(
                "Pedal input requires evdev; use keyboard/prompt or install the pedal dependency"
            )
        listeners.append(listener)
    return listeners


def _dispatch_prompt(runtime: LocalDagger, line: str | None) -> None:
    if line is None or line in {"/stop", "/quit", "/exit"}:
        runtime.request_stop()
        return
    command, _, value = line.partition(" ")
    if command == "/status":
        print(json.dumps(runtime.status(), ensure_ascii=False, indent=2), flush=True)
    elif command in {"/start", "/pause", "/correct", "/reset", "/task"}:
        if not runtime.submit_operator_event(command[1:], value.strip() or None):
            logging.warning("DAgger is busy; command ignored. /status and /stop remain available.")
    elif command:
        print(HELP, flush=True)


def run(config: LocalDaggerConfig) -> None:
    """Keep operator input responsive while model inference, saving and resets run separately."""
    runtime = LocalDagger(config, make_robot_from_config(config.robot))
    listeners = []
    try:
        runtime.start()
        commands = Queue()
        if config.input_device == "prompt":
            Thread(target=_read_commands, args=(commands,), name="LocalDaggerPrompt", daemon=True).start()
            print(f"Local DAgger ready: {HELP}", flush=True)
        else:
            listeners = _start_inputs(runtime, config)
            print(
                "Local DAgger ready: advance → start / pause / correct / save+reset; reset → discard+reset",
                flush=True,
            )
        ready_at = time.monotonic()
        previous = None
        while not runtime.stop_event.wait(0.05):
            if config.duration_s is not None and time.monotonic() - ready_at >= config.duration_s:
                runtime.request_stop()
                break
            progress = (runtime.state, runtime.phase, runtime.recorder.saved_episodes)
            if progress != previous:
                logging.info(
                    "DAgger %s/%s | saved corrections: %s/%s",
                    *progress,
                    config.num_episodes or "unlimited",
                )
                previous = progress
            if config.input_device == "prompt":
                with suppress(Empty):
                    _dispatch_prompt(runtime, commands.get_nowait())
        if runtime.failure:
            raise RuntimeError(runtime.failure)
    finally:
        # Keyboard callbacks only request stop; encoding never owns this loop.
        runtime.request_stop()
        try:
            runtime.stop()
        finally:
            for listener in listeners:
                stop = getattr(listener, "stop", None)
                if callable(stop):
                    stop()
            logging.info("Local DAgger final status: %s", json.dumps(runtime.status(), ensure_ascii=False))


def main() -> None:
    """Parse the independent local DAgger command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    logging.getLogger("can").setLevel(logging.WARNING)
    config = draccus.parse(LocalDaggerConfig)
    try:
        run(config)
    except KeyboardInterrupt:
        logging.info("Local DAgger interrupted")


if __name__ == "__main__":
    main()
