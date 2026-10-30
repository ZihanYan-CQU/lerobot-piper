"""Local dual-Piper I/O with one shared observation/handover boundary."""

from __future__ import annotations

import time
from threading import RLock

import numpy as np

from .piper_dagger import PiperDaggerController, PiperDaggerHardwareConfig
from .robot_io import RobotIO


class DaggerRobotIO(RobotIO):
    """Mirror autonomous commands and isolate human capture from handovers."""

    def __init__(
        self,
        robot,
        hardware: PiperDaggerHardwareConfig,
        action_keys=None,
        *,
        controller_factory=PiperDaggerController,
    ):
        """Create the adapter and its synchronization state."""
        super().__init__(robot, action_keys)
        if not self._piper:
            raise TypeError("Local DAgger currently requires piper_follower")
        self.hardware = hardware
        self.capture_lock = RLock()
        self.controller = None
        self._controller_factory = controller_factory
        self._closed = False

    def connect(self) -> None:
        """Connect the follower before initializing the master and closed CAN gate."""
        super().connect()
        self.controller = self._controller_factory(self.robot, self.hardware)

    def seed(self, pose: dict[str, float]) -> None:
        """Seed the common smoother, so both arms receive identical targets."""
        with self._motor_lock:
            self.controller.seed_command(self._validate_pose(pose))

    def capture(self, max_age_s: float) -> tuple[dict, float, float]:
        """Capture owned frames without crossing a takeover or disconnect."""
        with self.capture_lock:
            if self._closed:
                raise RuntimeError("DAgger hardware is disconnected")
            return super().capture(max_age_s)

    def correction_observation(self, max_age_s: float) -> dict:
        """Reject stale camera frames instead of silently labeling them as current."""
        values, _, source = self.capture(max_age_s)
        if time.monotonic() - source > max_age_s:
            raise TimeoutError("Correction observation is stale")
        return values

    def send(self, values: np.ndarray, *, deadline: float | None = None) -> np.ndarray:
        """Apply exactly one common filtered target to both arms."""
        if values.shape != (len(self.action_keys),) or not np.isfinite(values).all():
            raise ValueError("Invalid robot command shape or values")
        requested = dict(zip(self.action_keys, map(float, values), strict=True))
        with self._motor_lock:
            if self._closed:
                raise RuntimeError("DAgger hardware is disconnected")
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Action source expired while waiting for motor I/O")
            applied = self._validate_pose(self.controller.send_same_action(requested))
        return np.asarray([applied[key] for key in self.action_keys], dtype=np.float32)

    def read_reset_poses(self, timeout_s: float) -> dict[str, dict[str, float]]:
        """Require fresh physical CAN feedback from both arms, not SDK caches."""
        with self._motor_lock:
            return {
                arm: self._validate_pose(pose)
                for arm, pose in self.controller.read_reset_feedback(timeout_s).items()
            }

    def disconnect(self, *, faulted: bool) -> None:
        """Close both arms after capture has stopped; fault cleanup never homes."""
        with self.capture_lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            if self.controller is not None:
                try:
                    if faulted:
                        self.controller.emergency_close()
                    else:
                        self.controller.close()
                except Exception as exc:
                    errors.append(exc)
                    faulted = True
            try:
                super().disconnect(faulted=faulted)
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise ExceptionGroup("DAgger hardware cleanup failed", errors)
