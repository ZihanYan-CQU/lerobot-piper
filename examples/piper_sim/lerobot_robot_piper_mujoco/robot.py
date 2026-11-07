"""LeRobot Robot adapter; piper_follower and its CAN bus remain independent."""

import time
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ACTION_KEYS, PiperMujocoConfig

if TYPE_CHECKING:
    from .simulation import PiperSimulation


class VirtualCamera:
    """Minimal camera surface needed by LeRobot recording and visualization."""

    channels = 3

    def __init__(self, robot: "PiperMujoco", name: str):
        self.robot, self.name = robot, name
        self.width, self.height, self.fps = robot.config.width, robot.config.height, robot.config.fps

    @property
    def is_connected(self) -> bool:
        return self.robot.is_connected

    def async_read(self):
        return self.robot.simulation.image(self.name)

    def read_latest_with_timestamp(self, max_age_ms: float = 100):
        image = self.async_read()
        # A paused simulated scene is current; it does not become a stale physical camera frame.
        return image, time.perf_counter()

    def connect(self):
        if not self.robot.is_connected:
            raise DeviceNotConnectedError("Connect the virtual robot first")

    def disconnect(self):
        pass


class PiperMujoco(Robot):
    config_class = PiperMujocoConfig
    name = "piper_mujoco"

    def __init__(self, config: PiperMujocoConfig):
        # No hardware calibration files are needed or written by a virtual robot.
        self.config = config
        self.id = config.id
        self.robot_type = self.name
        self.calibration = {}
        self._simulation: PiperSimulation | None = None
        self.cameras = {name: VirtualCamera(self, name) for name in config.camera_names}

    @property
    def simulation(self) -> "PiperSimulation":
        if self._simulation is None:
            raise DeviceNotConnectedError("Piper simulation is not connected")
        return self._simulation

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @cached_property
    def observation_features(self) -> dict:
        return {
            **self.action_features,
            **dict.fromkeys(self.cameras, (self.config.height, self.config.width, 3)),
        }

    @property
    def is_connected(self) -> bool:
        return self._simulation is not None

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self):
        pass

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Piper simulation is already connected")
        from .simulation import PiperSimulation

        self._simulation = PiperSimulation(self.config)

    def reset(self, seed: int | None = None):
        self.simulation.reset(seed)

    def calibrate(self):
        # Explicit virtual reset only. No CAN, homing, or gripper SDK calls.
        self.reset(self.config.seed)

    def get_observation(self) -> dict:
        sim = self.simulation
        with sim.lock:
            observation = dict(zip(ACTION_KEYS, map(float, sim.state()), strict=True))
            observation.update({name: sim.image(name) for name in self.cameras})
            return observation

    def get_observation_with_timestamps(self, max_age_ms: float = 100) -> tuple[dict, dict[str, float]]:
        observation = self.get_observation()
        stamp = time.perf_counter()
        return observation, dict.fromkeys(self.cameras, stamp)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self.simulation.send(action)

    def send_action_js(self, action: dict[str, float]) -> dict[str, float]:
        """Compatibility with the existing Piper continuous-recording loop."""
        return self.send_action(action)

    def disconnect(self):
        if self._simulation is not None:
            try:
                self._simulation.close()
            finally:
                self._simulation = None
