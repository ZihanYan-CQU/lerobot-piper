"""MuJoCo owns virtual state only: this module never imports a hardware SDK."""

import time
from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any

import mujoco
import numpy as np

from .config import ACTION_KEYS, PiperMujocoConfig, default_model_path
from .scene import scene_xml

HOME = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])


class PiperSimulation:
    def __init__(self, config: PiperMujocoConfig):
        self.config = config
        self.lock = RLock()
        path = Path(config.model_path) if config.model_path else default_model_path()
        timestep = 1.0 / (config.fps * config.physics_substeps)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(path, timestep, config.width, config.height))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([self.model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.finger_index = self.model.jnt_qposadr[self.model.joint("joint7").id]
        self.mimic_index = self.model.jnt_qposadr[self.model.joint("joint8").id]
        self.gripper_actuator = self.model.actuator("gripper").id
        self.cube_index = self.model.jnt_qposadr[self.model.joint("cube_free").id]
        self.cube_id = self.model.body("cube").id
        self.goal_id = self.model.site("goal").id
        self.tcp_id = self.model.site("piper_grasp_site").id
        self.signs = np.asarray(config.joint_signs)
        self.offsets = np.asarray(config.joint_offsets)
        self._renderer = None
        self._viewer = None
        self._keys: deque[int] = deque()
        self._frame_cache: dict[str, np.ndarray] = {}
        self._frame_time = time.perf_counter()
        self._closed = False
        self.reset(config.seed)
        if config.viewer:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self.model, self.data, key_callback=self._keys.append)
            self._viewer.cam.lookat[:] = [0.10, 0.0, 0.90]
            self._viewer.cam.distance = 1.4
            self._viewer.cam.azimuth = 135
            self._viewer.cam.elevation = -30
            self._viewer.opt.geomgroup[3] = 0

    def to_model_joints(self, action: np.ndarray) -> np.ndarray:
        return self.signs * np.asarray(action) + self.offsets

    def from_model_joints(self, joints: np.ndarray) -> np.ndarray:
        return (np.asarray(joints) - self.offsets) / self.signs

    def _invalidate_images(self):
        self._frame_cache.clear()
        self._frame_time = time.perf_counter()

    def reset(self, seed: int | None = None):
        with self.lock:
            rng = np.random.default_rng(self.config.seed if seed is None else seed)
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qpos_indices] = HOME
            self.data.qpos[self.finger_index] = 0.035
            self.data.qpos[self.mimic_index] = -0.035
            cube = np.array([0.12, -0.06, 0.741])
            goal = np.array([0.12, 0.18, 0.721])
            if self.config.randomize:
                cube[:2] += rng.uniform(-0.025, 0.025, 2)
                goal[:2] += rng.uniform(-0.02, 0.02, 2)
            self.data.qpos[self.cube_index : self.cube_index + 7] = [*cube, 1, 0, 0, 0]
            self.model.site_pos[self.goal_id] = goal
            self.data.ctrl[self.actuator_ids] = HOME
            self.data.ctrl[self.gripper_actuator] = 0.035
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.config.fps * self.config.physics_substeps // 2)
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._last_target = self.data.qpos[self.qpos_indices].copy()
            self._initial_cube_z = float(self.cube_position[2])
            self._lifted = False
            self._success_ticks = 0
            self.success = False
            self._keys.clear()
            self._invalidate_images()
            self.sync_viewer()

    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_id].copy()

    @property
    def goal_position(self) -> np.ndarray:
        return self.data.site_xpos[self.goal_id].copy()

    def state(self) -> np.ndarray:
        with self.lock:
            joints = self.from_model_joints(self.data.qpos[self.qpos_indices])
            # pyAgxArm move_gripper_m uses total jaw width. MJCF controls a single finger.
            gap = 2.0 * float(self.data.qpos[self.finger_index])
            return np.r_[joints, np.clip(gap, 0, 0.07)].astype(np.float32)

    def snapshot(self) -> dict:
        """Own all episode initial state needed for deterministic action replay."""
        with self.lock:
            return {
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
                "goal": self.model.site_pos[self.goal_id].tolist(),
                "initial_cube_z": self._initial_cube_z,
                "time": float(self.data.time),
            }

    def restore(self, snapshot: dict):
        with self.lock:
            arrays = {}
            for key, length in (("qpos", self.model.nq), ("qvel", self.model.nv), ("ctrl", self.model.nu)):
                value = np.asarray(snapshot[key], dtype=float)
                if value.shape != (length,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid simulation snapshot field {key}")
                arrays[key] = value
            goal = np.asarray(snapshot["goal"], dtype=float)
            if goal.shape != (3,) or not np.all(np.isfinite(goal)):
                raise ValueError("Invalid goal in simulation snapshot")
            mujoco.mj_resetData(self.model, self.data)
            for key, value in arrays.items():
                getattr(self.data, key)[:] = value
            self.model.site_pos[self.goal_id] = goal
            self.data.time = float(snapshot["time"])
            self._initial_cube_z = float(snapshot["initial_cube_z"])
            self._last_target = self.data.ctrl[self.actuator_ids].copy()
            self._lifted, self._success_ticks, self.success = False, 0, False
            mujoco.mj_forward(self.model, self.data)
            self._invalidate_images()
            self.sync_viewer()

    def send(self, action: dict[str, Any]) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        if set(action) != set(ACTION_KEYS):
            raise ValueError(f"Action must contain exactly {ACTION_KEYS}")
        values = np.asarray([float(action[key]) for key in ACTION_KEYS])
        if not np.all(np.isfinite(values)):
            raise ValueError("Piper actions must be finite")
        with self.lock:
            target = self.to_model_joints(values[:6])
            limits = self.model.jnt_range[self.joint_ids]
            target = np.clip(target, limits[:, 0], limits[:, 1])
            if self.config.max_joint_step is not None:
                step = self.config.max_joint_step
                target = np.clip(target, self._last_target - step, self._last_target + step)
            gap = float(np.clip(values[6], 0.0, 0.07))
            self.data.ctrl[self.actuator_ids] = target
            self.data.ctrl[self.gripper_actuator] = gap / 2.0
            self._last_target = target.copy()
            mujoco.mj_step(self.model, self.data, nstep=self.config.physics_substeps)
            mujoco.mj_forward(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)):
                raise RuntimeError("MuJoCo produced non-finite state")
            cube, goal = self.cube_position, self.goal_position
            self._lifted |= cube[2] > self._initial_cube_z + 0.035
            in_goal = (
                np.all(np.abs(cube[:2] - goal[:2]) < 0.035)
                and 0.73 < cube[2] < 0.77
                and self.state()[6] > 0.055
            )
            self._success_ticks = self._success_ticks + 1 if self._lifted and in_goal else 0
            self.success |= self._success_ticks >= max(1, round(0.3 * self.config.fps))
            self._invalidate_images()
            self.sync_viewer()
            applied = np.r_[self.from_model_joints(target), gap]
            return dict(zip(ACTION_KEYS, map(float, applied), strict=True))

    def image(self, camera: str) -> np.ndarray:
        with self.lock:
            if self._closed:
                raise RuntimeError("Simulation is closed")
            if camera not in self._frame_cache:
                if self._renderer is None:
                    self._renderer = mujoco.Renderer(self.model, self.config.height, self.config.width)
                opt = mujoco.MjvOption()
                opt.geomgroup[3] = 0
                self._renderer.update_scene(self.data, camera=camera, scene_option=opt)
                self._frame_cache[camera] = self._renderer.render().copy()
            return self._frame_cache[camera].copy()

    def sync_viewer(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def pop_keys(self) -> list[int]:
        keys = []
        while self._keys:
            keys.append(self._keys.popleft())
        return keys

    def close(self):
        with self.lock:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            self._closed = True
