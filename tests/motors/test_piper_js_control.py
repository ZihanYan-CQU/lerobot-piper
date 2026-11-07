# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Piper's ordinary and explicit JS entrypoints use the deployed JS driver."""

from types import SimpleNamespace

import pytest

from lerobot.motors.piper.piper_PyAgxArm import PiperMotorsBus
from lerobot.robots.piper_follower import PIPERFollower, PIPERFollowerConfig


def _bus():
    joints = []
    gripper = []
    bus = PiperMotorsBus.__new__(PiperMotorsBus)
    bus.motors = {f"joint_{i}": (i, "piper") for i in range(1, 7)} | {"gripper": (7, "piper")}
    bus.piper = SimpleNamespace(
        move_js=lambda values: joints.append(list(values)),
        move_j=lambda *_: pytest.fail("Unexpected move_j call"),
        move_mit=lambda **_: pytest.fail("Unexpected explicit MIT call"),
    )
    bus.end_effector = SimpleNamespace(move_gripper_m=lambda **values: gripper.append(values))
    return bus, joints, gripper


def test_bus_stream_preserves_move_js_and_gripper_commands():
    bus, joints, gripper = _bus()
    values = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.03]
    bus.write_js(values)
    assert joints == [values[:6]]
    assert gripper == [{"value": 0.03, "force": 1.0}]


@pytest.mark.parametrize("method", ["send_action", "send_action_js"])
@pytest.mark.parametrize("joint_alpha,gripper_alpha", [(1.0, 1.0), (0.25, 0.5)])
def test_follower_entrypoints_preserve_js_and_existing_smoothing(method, joint_alpha, gripper_alpha):
    robot = PIPERFollower.__new__(PIPERFollower)
    robot.config = PIPERFollowerConfig(
        cameras={}, teleop_joint_alpha=joint_alpha, teleop_gripper_alpha=gripper_alpha
    )
    robot._is_connected = True
    robot.bus, joints, gripper = _bus()
    robot.seed_command(dict.fromkeys(robot.action_features, 0.0))
    action = dict.fromkeys(robot.action_features, 0.4)
    action["gripper.pos"] = 0.04
    performed = getattr(robot, method)(action)
    assert joints[0] == pytest.approx([0.4 * joint_alpha] * 6)
    assert gripper[0]["value"] == pytest.approx(0.04 * gripper_alpha)
    assert performed == pytest.approx({
        **{f"joint_{i}.pos": 0.4 * joint_alpha for i in range(1, 7)},
        "gripper.pos": 0.04 * gripper_alpha,
    })
