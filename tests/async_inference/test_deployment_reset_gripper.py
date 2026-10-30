# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Reset holds a grasp during arm return, then releases it after verified arrival."""

from functools import partial
from types import SimpleNamespace

import pytest

import lerobot.async_inference.client.robot_client_deployment as deployment
from lerobot.robots.piper_follower import PIPERFollower, PIPERFollowerConfig
from tests.mocks.mock_robot import MockRobotConfig


class _GraspingRobot:
    action_features = {**{f"joint_{i}.pos": float for i in range(1, 7)}, "gripper.pos": float}
    observation_features = action_features

    def __init__(self):
        self.pose = dict.fromkeys(self.action_features, 0.0)
        self.commands = []
        self.holding_object = False
        self.stuck_gripper = False
        self.stuck_joints = False

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        self.commands.append((dict(self.pose), dict(action), self.holding_object))
        for key, target in action.items():
            if key == "gripper.pos":
                if self.stuck_gripper:
                    continue
                # A held object stops closing at 16.7 mm. Opening takes several
                # feedback samples, so sending an open command is not enough.
                delta = max(-0.01, min(0.01, target - self.pose[key]))
                actual = self.pose[key] + delta
                if self.holding_object:
                    actual = max(0.0167, actual)
                    if actual >= 0.06:
                        self.holding_object = False
                self.pose[key] = actual
            elif not self.stuck_joints:
                self.pose[key] = target
        return dict(action)


@pytest.fixture
def make_reset_client(tmp_path, monkeypatch):
    clients = []

    def make(*, piper=True, **kwargs):
        robot = _GraspingRobot()
        monkeypatch.setattr(deployment, "make_robot_from_config", lambda _: robot)
        robot_config = (
            PIPERFollowerConfig(calibration_dir=tmp_path, cameras={})
            if piper
            else MockRobotConfig(calibration_dir=tmp_path)
        )
        client = deployment.RobotClientDeployment(
            deployment.RobotClientDeploymentConfig(
                policy_type="smolvla",
                pretrained_name_or_path="checkpoint",
                robot=robot_config,
                actions_per_chunk=8,
                reset_duration_s=0,
                reset_settle_s=0,
                reset_verify_timeout_s=0.3,
                **kwargs,
            )
        )
        clients.append(client)
        robot.pose = dict.fromkeys(robot.action_features, 0.5)
        robot.pose["gripper.pos"] = 0.0167
        robot.holding_object = True
        return client, robot

    yield make
    for client in clients:
        client.stop()


@pytest.mark.parametrize("pose_source", ["startup", "explicit", "hold"])
@pytest.mark.parametrize("reset_duration_s", [0, 0.06])
def test_reset_holds_object_until_arm_arrives_then_opens(make_reset_client, pose_source, reset_duration_s):
    explicit_pose = dict.fromkeys(_GraspingRobot.action_features, 0.25)
    explicit_pose["gripper.pos"] = 0.0
    options = {}
    if pose_source == "explicit":
        options["reset_pose"] = explicit_pose
    elif pose_source == "hold":
        options["reset_to_initial_position"] = False
    client, robot = make_reset_client(**options)
    client.config.reset_duration_s = reset_duration_s

    assert client.start_rollout()
    assert client.reset_rollout()
    assert client.rollout_state is deployment.RolloutState.READY
    assert client.reset_error is None
    assert not robot.holding_object
    assert robot.pose["gripper.pos"] == pytest.approx(0.07)
    expected_joint = {"startup": 0.0, "explicit": 0.25, "hold": 0.5}[pose_source]
    assert robot.pose["joint_1.pos"] == pytest.approx(expected_joint)
    for before, command, holding_object in robot.commands:
        assert command["gripper.pos"] >= 0.0167
        if command["joint_1.pos"] != before["joint_1.pos"]:
            assert holding_object
            assert command["gripper.pos"] == pytest.approx(0.0167)
        if command["gripper.pos"] > 0.0167:
            assert before["joint_1.pos"] == pytest.approx(expected_joint)
    # Applying the open override must not overwrite the saved reset pose.
    saved_pose = client._configured_reset_pose if pose_source == "explicit" else client._initial_pose
    if saved_pose is not None:
        assert saved_pose["gripper.pos"] == 0.0
    assert explicit_pose["gripper.pos"] == 0.0
    assert client.start_rollout()


def test_reset_reseeds_real_piper_smoothing_before_first_command(make_reset_client, monkeypatch):
    client, robot = make_reset_client()
    client.config.reset_verify_timeout_s = 1.0
    send_to_motors = robot.send_action
    robot.config = SimpleNamespace(teleop_joint_alpha=0.25, teleop_gripper_alpha=0.5)
    robot._is_connected = True
    # The last commanded gripper target is fully closed, but feedback is 16.7 mm.
    robot._ema_cmd = [0.0] * 7
    robot.bus = SimpleNamespace(
        write=lambda values: send_to_motors(dict(zip(robot.action_features, values, strict=True)))
    )
    monkeypatch.setattr(robot, "seed_command", partial(PIPERFollower.seed_command, robot), raising=False)
    monkeypatch.setattr(robot, "send_action", partial(PIPERFollower.send_action, robot))

    assert client.reset_rollout()
    assert robot.commands[0][1]["gripper.pos"] == pytest.approx(0.0167)
    assert all(command["gripper.pos"] >= 0.0167 for _, command, _ in robot.commands)
    for before, command, _ in robot.commands:
        if command["gripper.pos"] > 0.0167:
            assert abs(before["joint_1.pos"]) <= client.config.reset_joint_tolerance
    assert not robot.holding_object


def test_arm_reset_failure_does_not_open_gripper(make_reset_client):
    client, robot = make_reset_client()
    robot.stuck_joints = True

    assert not client.reset_rollout()
    assert client.rollout_state is deployment.RolloutState.RESET_FAILED
    assert not client.start_rollout()
    assert robot.holding_object
    assert all(command["gripper.pos"] == pytest.approx(0.0167) for _, command, _ in robot.commands)
    assert "joint_1.pos" in client.reset_error


@pytest.mark.parametrize("feedback_arm", ["robot", "master", "follower"])
def test_gripper_hold_error_does_not_block_release(make_reset_client, monkeypatch, feedback_arm):
    client, robot = make_reset_client()
    client.config.reset_settle_s = 0.04
    hold_samples = 0

    def feedback(timeout_s):
        nonlocal hold_samples
        arms = ["robot"] if feedback_arm == "robot" else ["master", "follower"]
        poses = {arm: robot.get_observation() for arm in arms}
        if robot.commands[-1][1]["gripper.pos"] == pytest.approx(0.0167):
            # Reproduce the real 1.3 mm tracking error while holding an object.
            poses[feedback_arm]["gripper.pos"] += 0.0013
            hold_samples += 1
        return poses

    monkeypatch.setattr(client, "_read_reset_poses_unlocked", feedback)
    assert client.reset_rollout()
    assert hold_samples >= 2
    assert not robot.holding_object
    assert robot.pose["gripper.pos"] == pytest.approx(0.07)
    assert client.reset_error is None
    assert client.start_rollout()


def test_gripper_release_still_rejects_1_3_mm_tracking_error(make_reset_client, monkeypatch):
    client, robot = make_reset_client()

    def feedback(timeout_s):
        pose = robot.get_observation()
        if robot.commands[-1][1]["gripper.pos"] == pytest.approx(0.07):
            pose["gripper.pos"] -= 0.0013
        return {"robot": pose}

    monkeypatch.setattr(client, "_read_reset_poses_unlocked", feedback)
    assert not client.reset_rollout()
    assert client.rollout_state is deployment.RolloutState.RESET_FAILED
    assert "reset gripper release" in client.reset_error
    assert "gripper.pos" in client.reset_error
    assert not client.start_rollout()


@pytest.mark.parametrize("invalid_feedback", [None, float("nan"), float("inf")])
def test_gripper_hold_still_requires_valid_feedback(make_reset_client, monkeypatch, invalid_feedback):
    client, robot = make_reset_client()

    def feedback(timeout_s):
        pose = robot.get_observation()
        if invalid_feedback is None:
            del pose["gripper.pos"]
        else:
            pose["gripper.pos"] = invalid_feedback
        return {"robot": pose}

    monkeypatch.setattr(client, "_read_reset_poses_unlocked", feedback)
    assert not client.reset_rollout()
    assert client.rollout_state is deployment.RolloutState.RESET_FAILED
    assert "gripper.pos" in client.reset_error
    assert robot.holding_object
    assert all(command["gripper.pos"] == pytest.approx(0.0167) for _, command, _ in robot.commands)
    assert not client.start_rollout()


@pytest.mark.parametrize("piper", [True, False])
def test_single_stage_reset_still_checks_gripper_convergence(make_reset_client, monkeypatch, piper):
    target = dict.fromkeys(_GraspingRobot.action_features, 0.0)
    target["gripper.pos"] = 0.0167
    options = {"reset_open_gripper": False} if piper else {}
    client, robot = make_reset_client(piper=piper, reset_pose=target, **options)

    def feedback(timeout_s):
        pose = robot.get_observation()
        pose["gripper.pos"] += 0.0013
        return {"robot": pose}

    monkeypatch.setattr(client, "_read_reset_poses_unlocked", feedback)
    assert not client.reset_rollout()
    assert client.rollout_state is deployment.RolloutState.RESET_FAILED
    assert "reset pose" in client.reset_error
    assert "gripper.pos" in client.reset_error
    assert not client.start_rollout()


def test_gripper_release_failure_blocks_start_until_retry_succeeds(make_reset_client):
    client, robot = make_reset_client()
    robot.stuck_gripper = True

    assert not client.reset_rollout()
    assert robot.pose["joint_1.pos"] == 0.0
    assert client.rollout_state is deployment.RolloutState.RESET_FAILED
    assert not client.start_rollout()
    assert "reset gripper release" in client.reset_error
    assert "gripper.pos" in client.reset_error

    robot.stuck_gripper = False
    assert client.reset_rollout()
    assert not robot.holding_object
    assert client.reset_error is None
    assert client.start_rollout()


def test_release_waits_for_all_arms_in_reset_feedback(make_reset_client, monkeypatch):
    client, robot = make_reset_client()
    reads_before_master_arrives = 3

    def feedback(timeout_s):
        nonlocal reads_before_master_arrives
        follower = robot.get_observation()
        master = dict(follower)
        if reads_before_master_arrives:
            # The follower is home, but the master has not reached the target.
            master["joint_1.pos"] = 0.5
            reads_before_master_arrives -= 1
            assert follower["gripper.pos"] == pytest.approx(0.0167)
        return {"follower": follower, "master": master}

    monkeypatch.setattr(client, "_read_reset_poses_unlocked", feedback)
    assert client.reset_rollout()
    assert not reads_before_master_arrives
    assert not robot.holding_object


def test_cancel_after_arm_arrival_does_not_open_gripper(make_reset_client, monkeypatch):
    client, robot = make_reset_client()
    wait_for_pose = client._wait_for_reset_pose_unlocked

    def wait_then_cancel(target, **kwargs):
        reached = wait_for_pose(target, **kwargs)
        client.shutdown_event.set()
        return reached

    monkeypatch.setattr(client, "_wait_for_reset_pose_unlocked", wait_then_cancel)
    assert not client.reset_rollout()
    assert robot.holding_object
    assert all(command["gripper.pos"] == pytest.approx(0.0167) for _, command, _ in robot.commands)


@pytest.mark.parametrize("piper", [True, False])
def test_custom_opening_is_applied_after_arm_return(make_reset_client, piper):
    client, robot = make_reset_client(piper=piper, reset_gripper_open_position=0.08)

    assert client.reset_rollout()
    assert robot.pose["gripper.pos"] == pytest.approx(0.08)
    for before, command, _ in robot.commands:
        if command["gripper.pos"] > 0.0167:
            assert before["joint_1.pos"] == 0.0
    assert client.config.to_dict()["reset_gripper_open_position"] == 0.08


@pytest.mark.parametrize("piper", [True, False])
def test_reset_pose_gripper_is_used_when_release_is_disabled_or_not_configured(make_reset_client, piper):
    target = dict.fromkeys(_GraspingRobot.action_features, 0.0)
    target["gripper.pos"] = 0.08
    options = {"reset_open_gripper": False} if piper else {}
    client, robot = make_reset_client(piper=piper, reset_pose=target, **options)

    assert client.reset_rollout()
    assert robot.commands[0][1]["gripper.pos"] == 0.08
    assert robot.pose["gripper.pos"] == pytest.approx(0.08)
    if piper:
        assert client.config.to_dict()["reset_open_gripper"] is False
    else:
        assert client.config.reset_gripper_open_position is None


@pytest.mark.parametrize("opening", [float("nan"), float("inf"), -0.01, 0.0, 1.0])
def test_invalid_piper_opening_is_rejected_before_connect(make_reset_client, opening):
    with pytest.raises(ValueError, match="reset_gripper_open_position"):
        make_reset_client(reset_gripper_open_position=opening)
