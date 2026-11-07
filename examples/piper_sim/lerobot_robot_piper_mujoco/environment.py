"""Gymnasium wrapper for lerobot-eval and optional training-time evaluation."""

import gymnasium as gym
import numpy as np

from .config import ACTION_KEYS, TASK, PiperMujocoConfig
from .robot import PiperMujoco


class PiperMujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str | None = None,
        fps: int = 30,
        width: int = 320,
        height: int = 240,
        camera_names: list[str] | None = None,
        randomize: bool = True,
        max_episode_steps: int = 600,
        render_mode: str = "rgb_array",
    ):
        if render_mode != "rgb_array":
            raise ValueError("Only rgb_array rendering is supported; use robot.viewer for a GUI")
        self.render_mode = render_mode
        self.metadata = {**type(self).metadata, "render_fps": fps}
        self._max_episode_steps = max_episode_steps
        self.task_description = TASK
        self.task = "PickPlace-v0"
        self.robot = PiperMujoco(
            PiperMujocoConfig(
                model_path=model_path,
                fps=fps,
                width=width,
                height=height,
                camera_names=camera_names or ["top", "wrist"],
                randomize=randomize,
            )
        )
        self.robot.connect()
        limits = self.robot.simulation.model.jnt_range[self.robot.simulation.joint_ids]
        self.action_space = gym.spaces.Box(
            low=np.r_[limits[:, 0], 0.0].astype(np.float32),
            high=np.r_[limits[:, 1], 0.07].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                "pixels": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
                        for name in self.robot.cameras
                    }
                ),
            }
        )
        self._steps = 0

    def _observation(self) -> dict:
        obs = self.robot.get_observation()
        return {
            "agent_pos": np.array([obs[key] for key in ACTION_KEYS], dtype=np.float32),
            "pixels": {name: obs[name] for name in self.robot.cameras},
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Respect Gym's random stream for unseeded autoresets, too.
        episode_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        self.robot.reset(episode_seed)
        self._steps = 0
        return self._observation(), {"is_success": False, "seed": episode_seed}

    def step(self, action):
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError("Expected seven Piper joint/gripper targets")
        self.robot.send_action(dict(zip(ACTION_KEYS, values, strict=True)))
        self._steps += 1
        success = bool(self.robot.simulation.success)
        return (
            self._observation(),
            float(success),
            success,
            self._steps >= self._max_episode_steps and not success,
            {"is_success": success, "sim_time": float(self.robot.simulation.data.time)},
        )

    def render(self):
        return self.robot.simulation.image("top")

    def close(self):
        self.robot.disconnect()
