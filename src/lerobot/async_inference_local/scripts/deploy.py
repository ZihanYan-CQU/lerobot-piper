"""Run one local PI05 deployment process: python -m lerobot.async_inference_local."""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import Future
from queue import Empty, Queue
from threading import Thread

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import make_robot_from_config
from lerobot.robots.piper_follower import PIPERFollowerConfig  # noqa: F401
from lerobot.utils.import_utils import _pyrealsense2_available

if _pyrealsense2_available:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from ..runtime.deployment import LocalDeployment, State
from ..shared.configs import LocalDeploymentConfig


def _read_commands(commands: Queue) -> None:
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put(None)


def _reset(runtime: LocalDeployment, result: Future[None]) -> None:
    """Run the interruptible pose restoration without blocking operator commands."""
    try:
        runtime.reset()
    except Exception as exc:
        result.set_exception(exc)
    else:
        result.set_result(None)


def run(config: LocalDeploymentConfig) -> None:
    """Run deployment and interactive lifecycle commands until exit or a worker fault."""
    robot = make_robot_from_config(config.robot)
    runtime = LocalDeployment(config, robot)
    reset_result: Future[None] | None = None
    reset_worker: Thread | None = None
    try:
        runtime.start()
        commands = Queue()
        Thread(target=_read_commands, args=(commands,), name="LocalPrompt", daemon=True).start()
        print("Local deployment ready: /start /pause /reset /task TEXT /status /quit", flush=True)
        run_started = time.monotonic() if runtime.state == State.RUNNING else None
        while not runtime.stop_event.is_set():
            if (
                run_started is not None
                and config.duration_s is not None
                and time.monotonic() - run_started >= config.duration_s
            ):
                break
            try:
                command = commands.get(timeout=0.1)
            except Empty:
                command = ""
            if reset_result is not None and reset_result.done():
                reset_result.result()
                reset_result = None
            if runtime.stop_event.is_set():
                break
            if command is None:
                if config.duration_s is None or run_started is None:
                    break
                continue
            try:
                if reset_result is not None and (
                    command in {"/start", "/resume", "/pause", "/reset"} or command.startswith("/task ")
                ):
                    raise RuntimeError("Reset in progress; /status and /quit remain available")
                if command in {"/start", "/resume"}:
                    runtime.resume()
                    run_started = time.monotonic()
                elif command == "/pause":
                    runtime.pause()
                    run_started = None
                elif command == "/reset":
                    run_started = None
                    reset_result = Future()
                    reset_worker = Thread(
                        target=_reset, args=(runtime, reset_result), name="LocalReset", daemon=True
                    )
                    reset_worker.start()
                elif command.startswith("/task "):
                    runtime.set_task(command.removeprefix("/task "))
                elif command == "/status":
                    print(json.dumps(runtime.status(), ensure_ascii=False, indent=2), flush=True)
                elif command in {"/quit", "/exit"}:
                    break
                elif command:
                    print("Commands: /start /pause /reset /task TEXT /status /quit", flush=True)
            except (ValueError, RuntimeError) as exc:
                logging.error("%s", exc)
        if runtime.failure:
            raise RuntimeError(runtime.failure)
    finally:
        runtime.stop()
        if reset_worker is not None:
            reset_worker.join(timeout=config.stop_timeout_s)
        logging.info("Local deployment final status: %s", json.dumps(runtime.status(), ensure_ascii=False))


def main() -> None:
    """Parse the standalone deployment CLI and configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    logging.getLogger("can").setLevel(logging.WARNING)
    config = draccus.parse(LocalDeploymentConfig)
    try:
        run(config)
    except KeyboardInterrupt:
        logging.info("Local deployment interrupted")


if __name__ == "__main__":
    main()
