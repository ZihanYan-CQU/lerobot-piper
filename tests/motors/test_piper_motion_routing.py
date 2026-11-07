"""Verify SDK motion selection without constructing or connecting a real arm."""

from types import SimpleNamespace

import pytest

from lerobot.motors.piper import piper_PyAgxArm as driver
from lerobot.robots.piper_follower import PIPERFollower, PIPERFollowerConfig


def make_bus():
    bus = driver.PiperMotorsBus.__new__(driver.PiperMotorsBus)
    calls = []
    bus.piper = SimpleNamespace(
        move_j=lambda values: calls.append(("j", list(values))),
        move_js=lambda values: calls.append(("js", list(values))),
    )
    bus.end_effector = SimpleNamespace(move_gripper_m=lambda **kwargs: calls.append(("gripper", kwargs)))
    bus.LOG_LENGTH = 77
    bus.init_joint_position = [0.0] * 7
    bus.safe_disable_position = [0.0] * 7
    bus.read_follower = lambda: {**{f"joint_{i}": 0.0 for i in range(1, 7)}, "gripper": 0.0}
    return bus, calls


@pytest.mark.parametrize("method,mode", [("write", "j"), ("write_js", "js")])
def test_joint_modes_preserve_payload_and_gripper_command(method, mode):
    bus, calls = make_bus()
    target = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, -0.02]
    getattr(bus, method)(target)
    assert calls == [(mode, target[:6]), ("gripper", {"value": 0.02, "force": 1.0})]
    assert target[-1] == -0.02


def test_home_and_exit_use_j(monkeypatch):
    bus, calls = make_bus()
    disabled = []
    bus.connect_follower = lambda **kwargs: disabled.append(kwargs)
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    assert bus.apply_calibration_follower(log=False)
    bus.safe_disconnect_follower(wait_time=0)
    assert [mode for mode, _ in calls if mode != "gripper"] == ["j", "j"]
    assert disabled == [{"enable": False}]


def test_streaming_resumes_js_after_calibration():
    bus, calls = make_bus()
    robot = PIPERFollower.__new__(PIPERFollower)
    robot.bus = bus
    robot.config = PIPERFollowerConfig(cameras={}, teleop_joint_alpha=1, teleop_gripper_alpha=1)
    robot._is_connected = True
    robot._ema_cmd = None
    action = {**{f"joint_{i}.pos": 0.01 for i in range(1, 7)}, "gripper.pos": 0.02}
    robot.send_action(action)
    robot.calibrate()
    robot.send_action(action)
    assert [mode for mode, _ in calls if mode != "gripper"] == ["js", "j", "js"]


def test_planned_positioning_bypasses_ema_and_seeds_following_stream():
    bus, calls = make_bus()
    robot = PIPERFollower.__new__(PIPERFollower)
    robot.bus = bus
    robot.config = PIPERFollowerConfig(cameras={}, teleop_joint_alpha=0.25, teleop_gripper_alpha=0.5)
    robot._is_connected = True
    robot._ema_cmd = [0.8] * 7
    action = {**{f"joint_{i}.pos": 0.1 for i in range(1, 7)}, "gripper.pos": 0.02}
    assert robot.send_action_planned(action) == action
    assert robot._ema_cmd == [0.1] * 6 + [0.02]
    performed = robot.send_action(action)
    assert performed == pytest.approx(action)
    assert [mode for mode, _ in calls if mode != "gripper"] == ["j", "js"]
    assert calls[0][1] == [0.1] * 6
