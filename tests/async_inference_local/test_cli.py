import threading
import time

import pytest
from test_runtime import FakeEngine, FakeRobot, configuration, wait_for

from lerobot.async_inference_local.runtime.deployment import LocalDeployment, State
from lerobot.async_inference_local.scripts import deploy


def start_cli(monkeypatch, **overrides):
    config = configuration(**overrides)
    robot, engine = FakeRobot(), FakeEngine()
    runtime = LocalDeployment(config, robot, engine_factory=lambda *args: engine)
    commands = []
    errors = []
    monkeypatch.setattr(deploy, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(deploy, "LocalDeployment", lambda *args: runtime)
    monkeypatch.setattr(deploy, "_read_commands", commands.append)

    def run():
        try:
            deploy.run(config)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, name="TestCLI", daemon=True)
    thread.start()
    wait_for(lambda: bool(commands))
    return runtime, robot, commands[0], thread, errors


@pytest.mark.parametrize("command", ["/status", "/quit"])
def test_reset_keeps_status_and_quit_responsive(monkeypatch, capsys, command):
    runtime, robot, commands, thread, errors = start_cli(monkeypatch, reset_duration_s=1.0)
    try:
        robot.position = 3.0
        commands.put("/reset")
        wait_for(lambda: runtime.state == State.RESETTING)
        commands.put(command)
        if command == "/status":
            output = []

            def saw_resetting():
                output.append(capsys.readouterr().out)
                return '"state": "resetting"' in "".join(output)

            wait_for(saw_resetting, timeout=0.3)
        else:
            thread.join(timeout=0.3)
            assert not thread.is_alive(), "/quit must cancel the active reset"
            assert robot.closed == 1
        assert not errors
    finally:
        commands.put("/quit")
        runtime.stop()
        thread.join(timeout=2)


def test_reset_finishes_before_subsequent_start_and_propagates_failures(monkeypatch):
    runtime, robot, commands, thread, errors = start_cli(monkeypatch, reset_duration_s=0.03)
    try:
        robot.position = 3.0
        commands.put("/reset")
        wait_for(lambda: runtime.state == State.RESETTING)
        wait_for(lambda: runtime.state == State.READY)
        assert robot.position == pytest.approx(0.0)
        commands.put("/start")
        wait_for(lambda: runtime.state == State.RUNNING and robot.sent[-1][1] > 0)
        commands.put("/pause")
        wait_for(lambda: runtime.state == State.PAUSED)

        def fail_read():
            raise OSError("reset feedback failed")

        robot.get_observation = fail_read
        commands.put("/reset")
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert "reset feedback failed" in str(errors[0])
        assert robot.closed == 1
    finally:
        commands.put("/quit")
        runtime.stop()
        thread.join(timeout=2)


def test_reset_rejects_motion_commands_until_it_completes(monkeypatch):
    runtime, robot, commands, thread, errors = start_cli(monkeypatch, reset_duration_s=0.2)
    try:
        commands.put("/reset")
        wait_for(lambda: runtime.state == State.RESETTING)
        commands.put("/start")
        commands.put("/reset")
        commands.put("/task changed")
        time.sleep(0.3)
        assert runtime.state == State.READY
        assert runtime.status()["task"] == "old"
        assert not errors
    finally:
        commands.put("/quit")
        runtime.stop()
        thread.join(timeout=2)
