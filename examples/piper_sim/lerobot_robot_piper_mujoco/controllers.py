"""Action sources for simulation sessions. Hardware is imported only for explicit leader mode."""

import importlib
import time
from typing import Any

import numpy as np

from .config import ACTION_KEYS, TASK
from .robot import PiperMujoco


class LeaderController:
    """Reuse the original leader driver, with no real follower and no per-episode homing."""

    source = "human_piper_leader"

    def __init__(self, robot: PiperMujoco, leader: Any = None):
        self.robot = robot
        self._connected = False
        if leader is None:
            from lerobot.utils.import_utils import require_package

            require_package("pyAgxArm", "piper")
            module = importlib.import_module("lerobot.teleoperators.piper_leader")
            leader = module.PIPERLeader(module.PIPERLeaderConfig(id="piper_sim_leader"))
        self.leader = leader

    def connect(self):
        # The existing driver performs its existing homing/gripper initialization here.
        # This method is NEVER used by expert, keyboard, replay, evaluation, or training.
        try:
            self.leader.connect()
        except BaseException:
            self._connected = bool(getattr(self.leader, "is_connected", False))
            raise
        self._connected = True

    def action(self, observation: dict | None = None) -> dict[str, float]:
        action = self.leader.get_action()
        if set(action) != set(ACTION_KEYS):
            raise ValueError("Leader returned a different joint/gripper contract")
        result = {key: float(action[key]) for key in ACTION_KEYS}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Leader returned non-finite joint/gripper data")
        return result

    def reset(self):
        # Resetting virtual objects must not invoke leader.calibrate().
        pass

    @property
    def done(self) -> bool:
        return False

    def close(self):
        if self._connected:
            try:
                self.leader.disconnect()
            finally:
                self._connected = False


class ExpertController:
    source = "scripted_oracle"

    def __init__(self, robot: PiperMujoco):
        self.robot = robot
        self.expert = None

    def connect(self):
        pass

    def reset(self):
        from .expert import PickPlaceExpert

        self.expert = PickPlaceExpert(self.robot.simulation)

    def action(self, observation: dict | None = None) -> dict[str, float]:
        return self.expert.action()

    @property
    def done(self) -> bool:
        return self.expert.done

    def close(self):
        pass


class KeyboardController:
    source = "human_keyboard"

    def __init__(self, robot: PiperMujoco):
        if not robot.config.viewer:
            raise ValueError("Keyboard control requires --viewer")
        self.robot = robot
        self._target = None

    def connect(self):
        pass

    def reset(self):
        self._target = self.robot.simulation.state().astype(float)

    def handle_keys(self, keys: list[int]):
        for key in keys:
            if key in range(ord("1"), ord("7")):
                self._target[key - ord("1")] += 0.04
            elif key in map(ord, "QWERTY"):
                self._target["QWERTY".index(chr(key))] -= 0.04
            elif key == ord(" "):
                self._target[6] = 0.07 if self._target[6] < 0.035 else 0.0

    def action(self, observation: dict | None = None) -> dict[str, float]:
        return dict(zip(ACTION_KEYS, map(float, self._target), strict=True))

    @property
    def done(self) -> bool:
        return False

    def close(self):
        pass


class PolicyController:
    source = "policy"

    def __init__(self, robot: PiperMujoco, path: str, device: str):
        import torch

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.robot = robot
        self.path = path
        self.torch = torch
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select --device cpu for CPU testing")
        expected = {f"observation.images.{name}" for name in robot.cameras}
        if set(config.image_features) != expected:
            raise ValueError(
                f"Checkpoint cameras {set(config.image_features)} differ from simulation {expected}"
            )
        for feature in config.image_features.values():
            if feature.shape != (3, robot.config.height, robot.config.width):
                raise ValueError("Set --width and --height to match the checkpoint image shapes")
        if config.output_features["action"].shape != (7,):
            raise ValueError("Checkpoint must output six joint targets and one gripper width")
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config).to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def connect(self):
        pass

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def action(self, observation: dict | None = None) -> dict[str, float]:
        from lerobot.envs.utils import preprocess_observation

        obs = observation if observation is not None else self.robot.get_observation()
        batch = preprocess_observation(
            {
                "agent_pos": np.array([obs[key] for key in ACTION_KEYS], dtype=np.float32),
                "pixels": {name: obs[name] for name in self.robot.cameras},
            }
        )
        batch["task"] = [TASK]
        with self.torch.inference_mode():
            prediction = self.policy.select_action(self.preprocessor(batch))
            values = self.postprocessor(prediction).squeeze(0).detach().cpu().numpy()
        if values.shape != (7,):
            raise ValueError(f"Unexpected policy action shape {values.shape}")
        return dict(zip(ACTION_KEYS, map(float, values), strict=True))

    @property
    def done(self) -> bool:
        return False

    def close(self):
        pass


def align_virtual_follower(robot: PiperMujoco, controller: LeaderController, timeout_s: float = 8):
    """Move only the simulated follower toward the leader, outside the recorded episode."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        started = time.monotonic()
        action = controller.action()
        applied = robot.send_action(action)
        state = robot.simulation.state()
        targets = np.array([action[key] for key in ACTION_KEYS])
        issued = np.array([applied[key] for key in ACTION_KEYS])
        if np.max(np.abs(issued[:6] - targets[:6])) < 0.02 and np.max(np.abs(state[:6] - targets[:6])) < 0.03:
            return
        if not robot.simulation.viewer_running():
            raise InterruptedError("Viewer closed during leader alignment")
        time.sleep(max(0.0, 1 / robot.config.fps - (time.monotonic() - started)))
    raise TimeoutError(
        "Virtual follower did not align: check leader pose, zero offsets, limits, and camera view"
    )
