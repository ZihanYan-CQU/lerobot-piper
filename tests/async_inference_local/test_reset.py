import time
from types import SimpleNamespace

import pytest

from lerobot.async_inference_local.runtime.deployment import LocalDeployment, State
from lerobot.async_inference_local.shared.configs import InferenceConfig, LocalDeploymentConfig


class GraspingRobot:
    name = "fake"
    action_features = {"joint.pos": float, "gripper.pos": float}
    observation_features = action_features

    def __init__(self):
        self.pose = {"joint.pos": 0.0, "gripper.pos": 0.0}
        self.commands = []
        self.seeded = []
        self.stuck_joint = False
        self.stuck_gripper = False
        self.hold_feedback_error = 0.0

    def connect(self):
        pass

    def disconnect(self):
        pass

    def seed_command(self, pose):
        self.seeded.append(dict(pose))

    def get_observation(self):
        value = dict(self.pose)
        if self.commands and self.commands[-1][2]["gripper.pos"] == pytest.approx(0.0167):
            value["gripper.pos"] += self.hold_feedback_error
        return value

    def send_action(self, action):
        self.commands.append((time.monotonic(), dict(self.pose), dict(action)))
        if not self.stuck_joint:
            self.pose["joint.pos"] = action["joint.pos"]
        if not self.stuck_gripper:
            self.pose["gripper.pos"] = action["gripper.pos"]
        return action


def setup_reset(**overrides):
    robot = GraspingRobot()
    values = {
        "robot": SimpleNamespace(type="piper_follower"),
        "pretrained_name_or_path": "unused",
        "task": "pick",
        "inference": InferenceConfig(device="cpu", compile_model=False, warmup_runs=0),
        "reset_duration_s": 0.02,
        "reset_fps": 200,
        "reset_settle_s": 0.02,
        "reset_settle_timeout_s": 0.15,
        "startup_timeout_s": 1,
        "stop_timeout_s": 0.2,
    }
    values.update(overrides)
    runtime = LocalDeployment(
        LocalDeploymentConfig(**values),
        robot,
        engine_factory=lambda *args: SimpleNamespace(warmup=lambda *args: []),
    )
    runtime.start()
    robot.pose = {"joint.pos": 0.5, "gripper.pos": 0.0167}
    return runtime, robot


@pytest.mark.parametrize("pose_source", ["startup", "explicit", "hold"])
@pytest.mark.parametrize("duration", [0, 0.03])
def test_reset_holds_grasp_until_arm_stably_arrives_then_opens(pose_source, duration):
    options = {"reset_duration_s": duration}
    target = {"joint.pos": 0.25, "gripper.pos": 0.0}
    if pose_source == "explicit":
        options["reset_pose"] = target
    elif pose_source == "hold":
        options["reset_to_initial_position"] = False
    runtime, robot = setup_reset(**options)
    try:
        runtime.reset()
        assert runtime.state == State.READY
        expected = {"startup": 0.0, "explicit": 0.25, "hold": 0.5}[pose_source]
        assert robot.pose == pytest.approx({"joint.pos": expected, "gripper.pos": 0.07})
        arrival = None
        opening = None
        for stamp, before, command in robot.commands:
            if before["joint.pos"] == pytest.approx(expected) and arrival is None:
                arrival = stamp
            if command["joint.pos"] != before["joint.pos"]:
                assert command["gripper.pos"] == pytest.approx(0.0167)
            if command["gripper.pos"] > 0.0167:
                assert before["joint.pos"] == pytest.approx(expected)
                opening = opening or stamp
        assert opening is not None and opening - arrival >= runtime.config.reset_settle_s
        assert robot.seeded[1]["gripper.pos"] == pytest.approx(0.0167)
        assert target["gripper.pos"] == 0.0
        assert runtime._initial_pose["gripper.pos"] == 0.0
    finally:
        runtime.stop()


def test_every_reset_phase_uses_planned_robot_entrypoint():
    runtime, robot = setup_reset()
    planned = []
    move = robot.send_action

    def send_planned(action):
        planned.append(dict(action))
        return move(action)

    robot.send_action_planned = send_planned
    robot.send_action = lambda action: pytest.fail("reset must not use the streaming sender")
    try:
        runtime.reset()
        assert runtime.state == State.READY
        assert planned == [command for _, _, command in robot.commands]
        assert planned[0]["gripper.pos"] == pytest.approx(0.0167)
        assert planned[-1]["gripper.pos"] == pytest.approx(0.07)
    finally:
        runtime.stop()


def test_arm_reset_failure_does_not_release_grasp():
    runtime, robot = setup_reset()
    robot.stuck_joint = True
    try:
        with pytest.raises(TimeoutError, match="joint.pos"):
            runtime.reset()
        assert runtime.state == State.FAULT
        assert all(command["gripper.pos"] == pytest.approx(0.0167) for _, _, command in robot.commands)
    finally:
        runtime.stop()


def test_gripper_release_requires_measured_arrival():
    runtime, robot = setup_reset()
    robot.stuck_gripper = True
    try:
        with pytest.raises(TimeoutError, match="gripper"):
            runtime.reset()
        assert runtime.state == State.FAULT
        assert any(command["gripper.pos"] == pytest.approx(0.07) for _, _, command in robot.commands)
    finally:
        runtime.stop()


def test_small_grasp_tracking_error_does_not_block_arm_return_or_release():
    runtime, robot = setup_reset()
    robot.hold_feedback_error = 0.0013
    try:
        runtime.reset()
        assert runtime.state == State.READY
        assert robot.pose["gripper.pos"] == pytest.approx(0.07)
    finally:
        runtime.stop()


def test_disabling_open_restores_explicit_gripper_target():
    runtime, robot = setup_reset(reset_open_gripper=False, reset_pose={"joint.pos": 0.1, "gripper.pos": 0.04})
    try:
        runtime.reset()
        assert robot.pose == pytest.approx({"joint.pos": 0.1, "gripper.pos": 0.04})
    finally:
        runtime.stop()
