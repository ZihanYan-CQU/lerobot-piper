"""A task-specific oracle for exercising collection without moving real hardware.

Uses object poses from the simulator. These privileged inputs are never added to
the policy's image/state observation. Nothing welds or teleports a grasped object.
"""

import mujoco
import numpy as np

from .config import ACTION_KEYS
from .simulation import PiperSimulation


def solve_ik(sim: PiperSimulation, position: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Damped Jacobian IK for a downward-facing grasp site, without changing live state."""
    model = sim.model
    data = mujoco.MjData(model)
    data.qpos[:] = sim.data.qpos
    limits = model.jnt_range[sim.joint_ids]
    # A slight carry tilt stays inside the Piper wrist's +/-70 degree range.
    # At tabletop height the tool is vertical for grasp and release.
    angle = np.pi - (0.45 if position[2] > 0.78 else 0.0)
    c, s = np.cos(angle), np.sin(angle)
    desired_rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    best_error = float("inf")
    best_q = start.copy()

    def residual(q):
        data.qpos[sim.qpos_indices] = q
        mujoco.mj_forward(model, data)
        position_error = position - data.site_xpos[sim.tcp_id]
        rotation = data.site_xmat[sim.tcp_id].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (desired_rotation @ rotation.T).ravel())
        if quaternion[0] < 0:
            quaternion *= -1
        norm = np.linalg.norm(quaternion[1:])
        rotation_error = quaternion[1:] * (2 * np.arctan2(norm, quaternion[0]) / max(norm, 1e-12))
        return np.r_[position_error, 0.25 * rotation_error]

    yaw = np.arctan2(position[1], position[0] + 0.24)
    elbow_up = np.array([yaw, 1.9, -1.45, 0.0, 1.1, yaw])
    starts = [start, elbow_up]
    for initial in starts:
        q = np.clip(initial.copy(), limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
        for _ in range(250):
            error = residual(q)
            score = float(np.linalg.norm(error))
            if score < best_error:
                best_error, best_q = score, q.copy()
            if np.linalg.norm(error[:3]) < 0.001 and np.linalg.norm(error[3:]) < 0.003:
                return q
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, sim.tcp_id)
            jac = np.vstack([jac_pos[:, sim.dof_indices], 0.25 * jac_rot[:, sim.dof_indices]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.005**2 * np.eye(6), error)
            delta *= min(1.0, 0.2 / max(np.linalg.norm(delta), 1e-12))
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = np.clip(q + scale * delta, limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
                if np.linalg.norm(residual(candidate)) < score:
                    q = candidate
                    break
            else:
                break
    if best_error > 0.012:
        raise RuntimeError(f"Expert IK could not reach {position.tolist()}: residual {best_error:.4f}")
    return best_q


class PickPlaceExpert:
    def __init__(self, sim: PiperSimulation):
        self.sim = sim
        cube, goal = sim.cube_position, sim.goal_position
        q = sim.data.qpos[sim.qpos_indices].copy()
        # Each waypoint is reached through normal position actuators and contact physics.
        points = [
            (np.array([*cube[:2], 0.83]), 0.07, 2.0),
            (np.array([*cube[:2], 0.752]), 0.07, 1.5),
            (np.array([*cube[:2], 0.752]), 0.0, 1.0),
            (np.array([*cube[:2], 0.83]), 0.0, 1.5),
            (np.array([*goal[:2], 0.83]), 0.0, 2.0),
            (np.array([*goal[:2], 0.752]), 0.0, 1.5),
            (np.array([*goal[:2], 0.752]), 0.07, 1.0),
            (np.array([*goal[:2], 0.83]), 0.07, 1.5),
        ]
        self._actions: list[np.ndarray] = []
        previous = np.r_[q, 0.07]
        for position, gap, seconds in points:
            q = solve_ik(sim, position, q)
            target = np.r_[q, gap]
            count = round(seconds * sim.config.fps)
            for tick in range(count):
                # Smooth interpolation, with a short dwell at each waypoint.
                alpha = min(1.0, (tick + 1) / max(1, round(count * 0.75)))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                self._actions.append(previous * (1 - alpha) + target * alpha)
            previous = target
        self._tick = 0

    @property
    def done(self) -> bool:
        return self._tick >= len(self._actions)

    def action(self) -> dict[str, float]:
        index = min(self._tick, len(self._actions) - 1)
        values = self._actions[index]
        self._tick += 1
        canonical = np.r_[self.sim.from_model_joints(values[:6]), values[6]]
        return dict(zip(ACTION_KEYS, map(float, canonical), strict=True))
