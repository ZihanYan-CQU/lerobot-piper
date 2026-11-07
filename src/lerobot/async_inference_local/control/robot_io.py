"""Robot adapter that keeps Piper camera waits outside the motor I/O lock."""

from __future__ import annotations

import logging
import math
import time
from threading import Event, Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RobotIO:
    """Serialize motor access while keeping Piper camera acquisition independent."""

    def __init__(self, robot: Any, action_keys: list[str] | None = None):
        """Validate action ordering and select the hardware I/O strategy."""
        self.robot = robot
        available = list(robot.action_features)
        self.action_keys = action_keys or available
        if set(self.action_keys) != set(available) or len(self.action_keys) != len(available):
            raise ValueError("action_keys must list every robot action feature exactly once")
        self._motor_lock = Lock()
        self._piper = getattr(robot, "name", None) == "piper_follower"
        self._untimestamped_cameras: set[str] = set()
        self._timestamped_robot = callable(getattr(robot, "get_observation_with_timestamps", None))

    def connect(self) -> None:
        """Connect the robot through its existing hardware driver."""
        self.robot.connect()

    def _validate_pose(self, pose: dict) -> dict[str, float]:
        result = {key: float(pose[key]) for key in self.action_keys}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Robot returned non-finite joint positions")
        return result

    def read_pose(self) -> dict[str, float]:
        """Read finite joint positions, without waiting for Piper cameras."""
        with self._motor_lock:
            if self._piper:
                values = self.robot.bus.read_action_positions(self.robot.bus.read_follower())
            else:
                values = self.robot.get_observation()
            return self._validate_pose(values)

    def read_reset_poses(self, timeout_s: float) -> dict[str, dict[str, float]]:
        """Read the feedback that must settle before a reset releases the gripper."""
        return {"robot": self.read_pose()}

    def seed(self, pose: dict[str, float]) -> None:
        """Seed optional driver smoothing from measured positions without moving."""
        with self._motor_lock:
            seed = getattr(self.robot, "seed_command", None)
            if callable(seed):
                seed(pose)

    def capture(self, max_age_s: float) -> tuple[dict, float, float]:
        """Return owned observations, acquisition time and conservative source time."""
        started = time.monotonic()
        times: dict[str, float] = {}
        if self._piper:
            images = {}
            for name, camera in self.robot.cameras.items():
                if name not in self._untimestamped_cameras:
                    try:
                        image, times[name] = camera.read_latest_with_timestamp(max_age_ms=max_age_s * 1000)
                    except (NotImplementedError, AttributeError):
                        self._untimestamped_cameras.add(name)
                        logger.warning(
                            "Camera %s has no timestamps; its age uses acquisition start time", name
                        )
                if name in self._untimestamped_cameras:
                    image = camera.async_read()
                # Copy before any other hardware call can recycle an SDK buffer.
                images[name] = self._own_value(image)
            # Cameras can wait independently while the control thread writes motors.
            values = self.read_pose()
            values.update(images)
        else:
            # Conservative fallback for drivers without independent motor/camera APIs.
            with self._motor_lock:
                if self._timestamped_robot:
                    try:
                        values, times = self.robot.get_observation_with_timestamps(
                            max_age_ms=max_age_s * 1000
                        )
                    except NotImplementedError:
                        self._timestamped_robot = False
                        values = self.robot.get_observation()
                else:
                    values = self.robot.get_observation()
                # A subsequent motor call must not overwrite returned driver buffers.
                values = {key: self._own_value(value) for key, value in values.items()}
        camera_now = time.perf_counter()
        finished = time.monotonic()
        if any(not math.isfinite(stamp) or stamp > camera_now for stamp in times.values()):
            raise ValueError("Camera timestamp must be finite and no later than the acquisition clock")
        source = min(started, finished - max(0.0, camera_now - min(times.values()))) if times else started
        return values, started, source

    @staticmethod
    def _own_value(value):
        if isinstance(value, np.ndarray):
            value = value.copy()
            value.setflags(write=False)
        return value

    def send(
        self, values: np.ndarray, *, deadline: float | None = None, stop_event: Event | None = None
    ) -> np.ndarray:
        """Apply one finite CPU command and return the driver-reported target."""
        return self._send(values, deadline=deadline, planned=False, stop_event=stop_event)

    def send_planned(self, values: np.ndarray) -> np.ndarray:
        """Use the robot's planned positioning entrypoint for reset waypoints."""
        return self._send(values, deadline=None, planned=True)

    @staticmethod
    def _check_send_guard(deadline: float | None, stop_event: Event | None) -> None:
        """Called with the motor lock held, immediately before a driver write."""
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Robot command cancelled while waiting for motor I/O")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Action source expired while waiting for motor I/O")

    def _send(
        self, values: np.ndarray, *, deadline: float | None, planned: bool, stop_event: Event | None = None
    ) -> np.ndarray:
        if values.shape != (len(self.action_keys),) or not np.isfinite(values).all():
            raise ValueError("Invalid robot command shape or values")
        requested = dict(zip(self.action_keys, map(float, values), strict=True))
        with self._motor_lock:
            sender = getattr(self.robot, "send_action_planned", None) if planned else None
            if not callable(sender):
                sender = self.robot.send_action
            self._check_send_guard(deadline, stop_event)
            applied = sender(requested)
        applied = requested if applied is None else self._validate_pose(applied)
        return np.asarray([applied[key] for key in self.action_keys], dtype=np.float32)

    def disconnect(self, *, faulted: bool) -> None:
        """Release hardware, avoiding a Piper homing trajectory after faults."""
        if self._piper and faulted:
            # Piper.disconnect() moves home. Never start a new trajectory on a fault.
            errors = []
            try:
                # The existing bus convenience method also resets the stop after
                # sleeping. Keep the stop latched until the operator handles it.
                self.robot.bus.piper.electronic_emergency_stop()
            except Exception as exc:
                errors.append(exc)
            for camera in self.robot.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception as exc:
                    errors.append(exc)
            # Match the driver's bookkeeping without its homing/reset trajectory.
            self.robot._is_connected = False
            self.robot._ema_cmd = None
            if errors:
                raise ExceptionGroup("Piper fault cleanup failed", errors)
            return
        self.robot.disconnect()
