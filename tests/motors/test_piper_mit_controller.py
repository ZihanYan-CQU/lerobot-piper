# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Follower gain routing and command validation without robot hardware."""

import time
from types import SimpleNamespace

import pytest

from lerobot.motors.piper.mit_controller import MITPositionSettings, PiperMITPositionController
from lerobot.robots.piper_follower.config_piper_follower import PIPERFollowerConfig
from lerobot.robots.piper_follower.piper_follower_PyAgxArm import PIPERFollower

LIMITS = {f"joint{index}": [-2.0, 2.0] for index in range(1, 7)}


def _settings(**overrides):
    return MITPositionSettings(**{
        "kp": [10, 11, 12, 13, 14, 15],
        "kd": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "torque_ff": [0.0] * 6,
        **overrides,
    })


class _SDK:
    def __init__(self):
        self.commands = []
        self.gripper_commands = []
        self.version = "S-V1.9-0"
        self.sample = SimpleNamespace(msg=[0.0] * 6, timestamp=1000.0)

    def get_firmware(self):
        return {"software_version": self.version}

    def get_joint_angles(self):
        return self.sample

    def move_mit(self, **kwargs):
        self.commands.append(kwargs)

    def move_gripper_m(self, **kwargs):
        self.gripper_commands.append(kwargs)


def _controller(sdk=None, **settings):
    sdk = sdk or _SDK()
    return sdk, PiperMITPositionController(sdk, sdk, LIMITS, _settings(**settings), clock=lambda: 1000.05)


@pytest.mark.parametrize("field,value", [
    ("kp", None), ("kd", [0.5] * 5), ("torque_ff", [0.0] * 7),
    ("kp", [float("nan")] * 6), ("kd", [float("inf")] * 6),
    ("kp", [0.0] * 6), ("kd", [-0.1] * 6),
    ("kp", [501.0] * 6), ("kd", [5.1] * 6), ("torque_ff", [16.1] * 6),
    ("max_tracking_error", 0), ("feedback_max_age_s", float("nan")),
])
def test_invalid_parameters_are_rejected_before_hardware(field, value):
    with pytest.raises(ValueError):
        _settings(**{field: value})


def test_settings_copy_mutable_gain_arrays():
    gains = [10.0] * 6
    settings = _settings(kp=gains)
    gains[0] = 500.0
    assert settings.kp[0] == 10.0


def test_each_joint_receives_its_own_gains_and_unscaled_feedforward():
    sdk, controller = _controller(torque_ff=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    controller.send([0.01, 0.02, -0.03, 0.04, -0.05, 0.06, 0.03])
    assert [command["joint_index"] for command in sdk.commands] == list(range(1, 7))
    assert [command["kp"] for command in sdk.commands] == [10, 11, 12, 13, 14, 15]
    assert [command["kd"] for command in sdk.commands] == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert [command["t_ff"] for command in sdk.commands] == [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert all(command["v_des"] == 0 for command in sdk.commands)
    assert sdk.gripper_commands == [{"value": 0.03, "force": 1.0}]


@pytest.mark.parametrize("command", [
    [0.0] * 6,
    [0.0] * 6 + [float("nan")],
    [0.0] * 5 + [2.01, 0.03],
    [0.0] * 5 + [0.11, 0.03],
])
def test_invalid_last_component_does_not_partially_send_earlier_joints(command):
    sdk, controller = _controller()
    with pytest.raises((ValueError, RuntimeError)):
        controller.send(command)
    assert sdk.commands == []
    assert sdk.gripper_commands == []


@pytest.mark.parametrize("sample", [
    None,
    SimpleNamespace(msg=[0.0] * 6, timestamp=999.0),
    SimpleNamespace(msg=[0.0] * 6, timestamp=1001.0),
    SimpleNamespace(msg=[0.0] * 6, timestamp=float("nan")),
    SimpleNamespace(msg=[0.0] * 5 + [float("nan")], timestamp=1000.0),
])
def test_missing_or_invalid_feedback_prevents_new_commands(sample):
    sdk, controller = _controller()
    sdk.sample = sample
    with pytest.raises((ValueError, RuntimeError)):
        controller.send([0.0] * 7)
    assert sdk.commands == []


@pytest.mark.parametrize("version", ["", "unknown", "S-V1.7-3", "S-V1.8-8"])
def test_unknown_or_incompatible_firmware_prevents_new_commands(version):
    sdk, controller = _controller()
    sdk.version = version
    with pytest.raises(RuntimeError, match="verified firmware"):
        controller.send([0.0] * 7)
    assert sdk.commands == []


def _robot(mode="native"):
    extra = {} if mode == "native" else {
        "mit_kp": [10.0] * 6,
        "mit_kd": [0.8] * 6,
        "mit_torque_ff": [0.0] * 6,
    }
    robot = PIPERFollower.__new__(PIPERFollower)
    robot.config = PIPERFollowerConfig(
        cameras={}, joint_control_mode=mode, teleop_joint_alpha=1, teleop_gripper_alpha=1, **extra
    )
    robot._is_connected = True
    robot._ema_cmd = None
    robot._mit_controller = None
    sdk = _SDK()
    # Route fresh feedback through the production controller's real clock.
    sdk.sample.timestamp = time.time()
    native = []
    robot.bus = SimpleNamespace(
        piper=sdk, end_effector=sdk, cfg={"joint_limits": LIMITS},
        motors={f"joint_{index}": (index, "piper") for index in range(1, 7)} | {"gripper": (7, "piper")},
        write=lambda values: native.append(("j", list(values))),
        write_js=lambda values: native.append(("js", list(values))),
    )
    return robot, sdk, native


@pytest.mark.parametrize("method,expected", [("send_action", "j"), ("send_action_js", "js")])
def test_native_mode_preserves_each_existing_sending_path(method, expected):
    robot, sdk, native = _robot()
    action = dict.fromkeys(robot.action_features, 0.01)
    getattr(robot, method)(action)
    assert native == [(expected, [0.01] * 7)]
    assert sdk.commands == []


@pytest.mark.parametrize("method", ["send_action", "send_action_js"])
def test_both_robot_entrypoints_use_explicit_mit_when_selected(method):
    robot, sdk, native = _robot("mit")
    action = dict.fromkeys(robot.action_features, 0.01)
    performed = getattr(robot, method)(action)
    assert performed == action
    assert len(sdk.commands) == 6
    assert native == []


def test_mit_config_requires_explicit_feedforward_choice():
    with pytest.raises(ValueError, match="six joint values"):
        PIPERFollowerConfig(cameras={}, joint_control_mode="mit", mit_kp=[10] * 6, mit_kd=[0.8] * 6)


def test_native_config_cannot_silently_ignore_gain_arguments():
    with pytest.raises(ValueError, match="native mode does not apply"):
        PIPERFollowerConfig(cameras={}, mit_kp=[10] * 6)


def test_mit_writer_does_not_apply_a_second_ema_to_shared_dagger_commands():
    robot, sdk, native = _robot("mit")
    robot.config.teleop_joint_alpha = 0.25
    robot.write_joint_command([0.04] * 7)
    assert [command["p_des"] for command in sdk.commands] == [0.04] * 6
    assert native == []
