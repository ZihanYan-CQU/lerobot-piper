"""The virtual robot has its own registry keys and never overrides piper_follower."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.robots.config import RobotConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

ACTION_KEYS = tuple([f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"])
TASK = "Pick up the red cube and place it in the green target area."


def default_model_path() -> Path:
    # The uv project installs this plugin editable from examples/piper_sim.
    # Resolve its resource directory independently of uv's caller working directory.
    repo_root = Path(__file__).resolve().parents[3]
    root = Path(os.environ.get("PIPER_SIM_ASSETS", str(repo_root / "outputs/piper_sim/assets")))
    return root / "piper.xml"


@RobotConfig.register_subclass("piper_mujoco")
@dataclass
class PiperMujocoConfig(RobotConfig):
    model_path: str | None = None
    fps: int = 30
    width: int = 320
    height: int = 240
    camera_names: list[str] = field(default_factory=lambda: ["top", "wrist"])
    viewer: bool = False
    seed: int = 0
    randomize: bool = True
    physics_substeps: int = 20
    # SDK / dataset coordinates -> MJCF coordinates: q_mj = sign * q_sdk + offset.
    joint_signs: list[float] = field(default_factory=lambda: [1.0] * 6)
    joint_offsets: list[float] = field(default_factory=lambda: [0.0] * 6)
    # Maximum change of the issued target per control tick, in radians. None disables it.
    max_joint_step: float | None = 0.10

    def __post_init__(self):
        super().__post_init__()
        if self.fps <= 0 or self.physics_substeps <= 0:
            raise ValueError("fps and physics_substeps must be positive")
        if self.width < 16 or self.height < 16:
            raise ValueError("Camera dimensions must be at least 16 pixels")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must contain unique camera names")
        if not set(self.camera_names) <= {"top", "wrist", "down", "agentview", "sideview"}:
            raise ValueError("Supported cameras: top, wrist, down, agentview, sideview")
        if len(self.joint_signs) != 6 or any(x not in (-1.0, 1.0) for x in self.joint_signs):
            raise ValueError("joint_signs must contain six values, each -1 or 1")
        if len(self.joint_offsets) != 6 or not all(math.isfinite(x) for x in self.joint_offsets):
            raise ValueError("joint_offsets must contain six finite radian offsets")
        if self.max_joint_step is not None and (
            not math.isfinite(self.max_joint_step) or self.max_joint_step <= 0
        ):
            raise ValueError("max_joint_step must be positive or None")


@EnvConfig.register_subclass("piper_mujoco")
@dataclass
class PiperMujocoEnvConfig(EnvConfig):
    task: str = "PickPlace-v0"
    fps: int = 30
    episode_length: int = 600
    observation_width: int = 320
    observation_height: int = 240
    camera_names: list[str] = field(default_factory=lambda: ["top", "wrist"])
    model_path: str | None = None
    randomize: bool = True

    def __post_init__(self):
        if self.task != "PickPlace-v0":
            raise ValueError("The supported Piper task is PickPlace-v0")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        # Validate shared fields before allocating any MuJoCo resources.
        PiperMujocoConfig(
            fps=self.fps,
            width=self.observation_width,
            height=self.observation_height,
            camera_names=self.camera_names,
        )
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            **{
                f"pixels/{name}": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(self.observation_height, self.observation_width, 3),
                )
                for name in self.camera_names
            },
        }
        self.features_map = {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            **{f"pixels/{name}": f"{OBS_IMAGES}.{name}" for name in self.camera_names},
        }

    @property
    def gym_kwargs(self) -> dict:
        return {
            "model_path": self.model_path,
            "fps": self.fps,
            "width": self.observation_width,
            "height": self.observation_height,
            "camera_names": self.camera_names,
            "randomize": self.randomize,
            "max_episode_steps": self.episode_length,
        }

    def create_envs(self, n_envs: int, use_async_envs: bool = False) -> dict:
        import gymnasium as gym

        # gym.make resolves the implementation lazily, keeping MuJoCo out of plugin discovery.
        if "gym_piper_mujoco/PickPlace-v0" not in gym.registry:
            gym.register(
                "gym_piper_mujoco/PickPlace-v0",
                entry_point="lerobot_robot_piper_mujoco.environment:PiperMujocoEnv",
            )
        return super().create_envs(n_envs, use_async_envs)
