import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from lerobot_robot_piper_mujoco.config import ACTION_KEYS, PiperMujocoConfig, default_model_path
from lerobot_robot_piper_mujoco.controllers import LeaderController


def test_default_model_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    """Find repository assets when direct uv commands run from another directory."""
    monkeypatch.delenv("PIPER_SIM_ASSETS", raising=False)
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1].parent.parent
    assert default_model_path() == repo_root / "outputs/piper_sim/assets/piper.xml"


def test_default_model_path_respects_explicit_asset_directory(tmp_path, monkeypatch):
    """Preserve explicit model asset overrides for direct and legacy callers."""
    assets = tmp_path / "selected-assets"
    monkeypatch.setenv("PIPER_SIM_ASSETS", str(assets))
    assert default_model_path() == assets / "piper.xml"


def test_discovery_does_not_import_hardware_or_mujoco():
    code = """
import sys
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.robots.config import RobotConfig
from lerobot.envs.configs import EnvConfig
register_third_party_plugins()
assert 'piper_mujoco' in RobotConfig.get_known_choices()
assert 'piper_mujoco' in EnvConfig.get_known_choices()
assert 'pyAgxArm' not in sys.modules
assert 'mujoco' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"physics_substeps": 0},
        {"camera_names": ["unknown"]},
        {"camera_names": ["top", "top"]},
        {"joint_signs": [1.0]},
        {"joint_offsets": [float("nan")] * 6},
        {"max_joint_step": -0.1},
    ],
)
def test_invalid_configuration_fails_before_connect(kwargs):
    with pytest.raises(ValueError):
        PiperMujocoConfig(**kwargs)


def test_factory_and_connection_lifecycle(model_path):
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

    robot = make_robot_from_config(PiperMujocoConfig(model_path=model_path))
    assert robot.name == "piper_mujoco"
    assert not hasattr(robot, "bus")
    assert tuple(robot.action_features) == ACTION_KEYS
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action(dict.fromkeys(ACTION_KEYS, 0.0))
    robot.connect()
    try:
        with pytest.raises(DeviceAlreadyConnectedError):
            robot.connect()
    finally:
        robot.disconnect()
    robot.disconnect()
    assert not robot.is_connected


def test_camera_reads_do_not_advance_physics_and_return_owned_frames(robot):
    sim = robot.simulation
    before = sim.data.time
    first = robot.get_observation()
    first["top"][:] = 0
    second = robot.get_observation()
    assert sim.data.time == before
    assert second["top"].shape == (64, 80, 3)
    assert second["top"].std() > 15
    assert not np.shares_memory(first["top"], second["top"])


def test_one_second_is_exactly_thirty_control_ticks(robot):
    pose = dict(zip(ACTION_KEYS, robot.simulation.state(), strict=True))
    for _ in range(30):
        robot.send_action_js(pose)
    assert robot.simulation.data.time == pytest.approx(1.0, abs=1e-12)


def test_invalid_actions_do_not_advance_state(robot):
    before = robot.simulation.snapshot()
    bad = dict.fromkeys(ACTION_KEYS, 0.0)
    bad["joint_1.pos"] = float("nan")
    with pytest.raises(ValueError):
        robot.send_action(bad)
    with pytest.raises(ValueError):
        robot.send_action({"joint_1.pos": 0})
    assert robot.simulation.snapshot() == before


def test_gripper_is_total_gap_and_issued_targets_are_limited(robot):
    action = dict(zip(ACTION_KEYS, robot.simulation.state(), strict=True))
    action["joint_1.pos"] = 100.0
    action["gripper.pos"] = 0.20
    issued = robot.send_action(action)
    assert issued["joint_1.pos"] == pytest.approx(0.1, abs=1e-5)
    assert issued["gripper.pos"] == pytest.approx(0.07)
    assert robot.simulation.data.ctrl[robot.simulation.gripper_actuator] == pytest.approx(0.035)
    assert action["joint_1.pos"] == 100.0  # Driver never rewrites the caller's input.


def test_seeded_reset_and_saved_initial_state_replay(robot):
    sim = robot.simulation
    robot.reset(12)
    initial = json.loads(json.dumps(sim.snapshot()))
    actions = []
    for i in range(60):
        target = dict(zip(ACTION_KEYS, sim.state(), strict=True))
        target["joint_1.pos"] = i / 300
        actions.append(robot.send_action(target))
    final = sim.state().copy()
    sim.restore(initial)
    for action in actions:
        robot.send_action(action)
    np.testing.assert_allclose(sim.state(), final, atol=2e-4)
    robot.reset(12)
    np.testing.assert_allclose(sim.data.qpos, initial["qpos"], atol=1e-9)


def test_signed_offset_mapping_is_reversible(robot):
    sim = robot.simulation
    sim.signs = np.array([-1, 1, -1, 1, 1, -1])
    sim.offsets = np.array([0.2, -0.1, 0.3, 0, 0, 0.1])
    joints = np.arange(6) / 10
    np.testing.assert_allclose(sim.from_model_joints(sim.to_model_joints(joints)), joints)


class FakeLeader:
    def __init__(self):
        self.events = []

    def connect(self):
        self.events.append("connect")

    def get_action(self):
        return dict.fromkeys(ACTION_KEYS, 0.0)

    def calibrate(self, **kwargs):
        raise AssertionError("Simulation reset must not home a real leader")

    def disconnect(self):
        self.events.append("disconnect")


def test_real_leader_adapter_reuses_contract_without_per_episode_homing(robot):
    leader = FakeLeader()
    controller = LeaderController(robot, leader=leader)
    controller.connect()
    for seed in range(3):
        robot.reset(seed)
        controller.reset()
        robot.send_action(controller.action())
    controller.close()
    assert leader.events == ["connect", "disconnect"]


def test_gym_environment_seed_and_termination(model_path):
    from lerobot_robot_piper_mujoco.environment import PiperMujocoEnv

    env = PiperMujocoEnv(model_path=model_path, width=80, height=64, max_episode_steps=2)
    try:
        observation, info = env.reset(seed=15)
        assert env.observation_space.contains(observation)
        cube = env.robot.simulation.cube_position
        _, _, terminated, truncated, _ = env.step(observation["agent_pos"])
        assert not terminated and not truncated
        _, _, _, truncated, _ = env.step(observation["agent_pos"])
        assert truncated
        env.reset(seed=15)
        np.testing.assert_allclose(env.robot.simulation.cube_position, cube)
    finally:
        env.close()


def test_learned_policy_cannot_claim_success_by_starting_in_goal(robot):
    sim = robot.simulation
    sim.data.qpos[sim.cube_index : sim.cube_index + 3] = [*sim.goal_position[:2], 0.74]
    for _ in range(30):
        robot.send_action(dict(zip(ACTION_KEYS, sim.state(), strict=True)))
    assert not sim.success  # Success requires an actual lift before placement.
