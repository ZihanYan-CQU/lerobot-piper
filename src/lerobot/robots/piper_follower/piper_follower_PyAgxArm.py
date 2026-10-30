#!/usr/bin/env python

import logging
import threading
from collections.abc import Callable
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.piper.mit_controller import MITPositionSettings, PiperMITPositionController
from lerobot.motors.piper.piper_PyAgxArm import PiperMotorsBus, PiperMotorsBusConfig, visual_pad
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_follower import PIPERFollowerConfig

logger = logging.getLogger(__name__)

def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]

class PIPERFollower(Robot):
    config_class = PIPERFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PIPERFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name="can_follower",
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                }
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)
        self._ema_cmd: list[float] | None = None
        self._mit_controller: PiperMITPositionController | None = None

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,   
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        arm_dict = {"follower": self.bus}
        action_names = get_motor_names(arm_dict)
        state_names = get_motor_names(arm_dict)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于 record/replay 的电机动作描述"""
        arm_dict = {"follower": self.bus}
        motor_names = get_motor_names(arm_dict)
        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于 record/replay 的相机图像描述"""
        #return {f"observation.images.{cam_key}": (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        # 不做任何事,过抽象类用
        pass

    @property
    def is_connected(self) -> bool:
        """机器人和所有相机是否都已连接"""
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""
        return self._is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras"""
        print('┌' + '─' * (self.bus.LOG_START_LENGTH - 2) + '┐')
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Piper is already connected. Do not run robot.connect() twice."
            )

        self.bus.connect_follower(enable=True)
        self._is_connected = True
        self.calibrate()
        self.bus.change_to_follower_mode()
        self.bus.get_firware_version_follower()
        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self._is_connected = self._is_connected and self.cameras[name].is_connected
            print(f"│──────────{visual_pad(f'相机 {name} 已连接', self.bus.LOG_LENGTH, '^')}──────────│")
        
        
        print('└' + '─' * (self.bus.LOG_END_LENGTH -2) + '┘')
        
    def disconnect(self,disconnect_time=7) -> None:
        """move to home position, disenable piper and cameras"""
        print('┌' + '─' * (self.bus.LOG_START_LENGTH -2) + '┐')
        self.bus.safe_disconnect_follower(wait_time=disconnect_time)
        #self.bus.connect_follower(enable=False)

        if len(self.cameras) > 0:
            for name in self.cameras:
                self.cameras[name].disconnect()
                print(f"│──────────{visual_pad(f'相机 {name} 已断开', self.bus.LOG_LENGTH, '^')}──────────│")

        self._is_connected = False
        self._ema_cmd = None
        self._mit_controller = None
        print('└' + '─' * (self.bus.LOG_END_LENGTH -2) + '┘')

    def calibrate(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()

        self.bus.apply_calibration_follower()
        self._is_calibrated = True  # 标记为已标定
        self._ema_cmd = None
        self._mit_controller = None

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        obs_dict = self.bus.read_action_positions(self.bus.read_follower())

        # 读取图像
        for name, cam in self.cameras.items():
            #obs_dict[f"observation.images.{name}"] = cam.async_read()
            obs_dict[name] = cam.async_read()

        return obs_dict

    def get_observation_with_timestamps(self, max_age_ms: float = 100) -> tuple[dict, dict[str, float]]:
        """Peek cameras, then read joints; camera times use local ``perf_counter``.

        Unlike recording's ``get_observation``, this never waits for or consumes
        a camera's new-frame event. Unsupported cameras raise NotImplementedError
        so a caller can explicitly use the ordinary capture path instead.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")
        images = {}
        timestamps = {}
        for name, cam in self.cameras.items():
            images[name], timestamps[name] = cam.read_latest_with_timestamp(max_age_ms=max_age_ms)
        obs_dict = self.bus.read_action_positions(self.bus.read_follower())
        obs_dict.update(images)
        return obs_dict, timestamps

    def seed_command(self, pose: dict[str, float]) -> None:
        """Seed command smoothing from measured positions without moving the arm."""
        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        self._ema_cmd = [float(pose[f"{motor}.pos"]) for motor in motor_order]

    def write_joint_command(self, command: list[float], *, follower_mode: bool = False) -> None:
        """Send an already-filtered command through the configured follower backend.

        DAgger calls this after its shared EMA, so this method must not filter
        again. Its master arm and physical correction bridge do not use it.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")
        if getattr(self.config, "joint_control_mode", "native") == "mit":
            if self._mit_controller is None:
                self._mit_controller = PiperMITPositionController(
                    self.bus.piper,
                    self.bus.end_effector,
                    self.bus.cfg["joint_limits"],
                    MITPositionSettings(
                        self.config.mit_kp, self.config.mit_kd, self.config.mit_torque_ff,
                        self.config.mit_max_tracking_error, self.config.mit_feedback_max_age_s,
                    ),
                )
                logger.warning(
                    "Experimental follower MIT gains enabled: kp=%s kd=%s torque_ff=%s; "
                    "velocity reference is zero; constant torque_ff is not a gravity model",
                    self.config.mit_kp, self.config.mit_kd, self.config.mit_torque_ff,
                )
            self._mit_controller.send(command)
        elif follower_mode:
            self.bus.write_js(command)
        else:
            self.bus.write(command)

    def send_action_js(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        target_joints = [float(action[f"{motor}.pos"]) for motor in motor_order]

        ja = min(1.0, max(0.0, float(self.config.teleop_joint_alpha)))
        ga = min(1.0, max(0.0, float(self.config.teleop_gripper_alpha)))
        if ja <= 0.0:
            ja = 1.0
        if ga <= 0.0:
            ga = 1.0

        if ja >= 1.0 - 1e-9 and ga >= 1.0 - 1e-9:
            cmd = list(target_joints)
        else:
            if self._ema_cmd is None:
                init = self.bus.read_action_positions(self.bus.read_follower())
                self._ema_cmd = [float(init[f"{m}.pos"]) for m in motor_order]
            for i in range(6):
                self._ema_cmd[i] = ja * target_joints[i] + (1.0 - ja) * self._ema_cmd[i]
            self._ema_cmd[6] = ga * target_joints[6] + (1.0 - ga) * self._ema_cmd[6]
            cmd = self._ema_cmd

        if getattr(self.config, "joint_control_mode", "native") == "mit":
            self.write_joint_command(cmd, follower_mode=True)
        else:
            self.bus.write_js(cmd)
        return {f"{motor_order[i]}.pos": cmd[i] for i in range(len(motor_order))}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        target_joints = [float(action[f"{motor}.pos"]) for motor in motor_order]

        ja = min(1.0, max(0.0, float(self.config.teleop_joint_alpha)))
        ga = min(1.0, max(0.0, float(self.config.teleop_gripper_alpha)))
        if ja <= 0.0:
            ja = 1.0
        if ga <= 0.0:
            ga = 1.0

        if ja >= 1.0 - 1e-9 and ga >= 1.0 - 1e-9:
            cmd = list(target_joints)
        else:
            if self._ema_cmd is None:
                init = self.bus.read_action_positions(self.bus.read_follower())
                self._ema_cmd = [float(init[f"{m}.pos"]) for m in motor_order]
            for i in range(6):
                self._ema_cmd[i] = ja * target_joints[i] + (1.0 - ja) * self._ema_cmd[i]
            self._ema_cmd[6] = ga * target_joints[6] + (1.0 - ga) * self._ema_cmd[6]
            cmd = self._ema_cmd

        if getattr(self.config, "joint_control_mode", "native") == "mit":
            self.write_joint_command(cmd)
        else:
            self.bus.write(cmd)
        return {f"{motor_order[i]}.pos": cmd[i] for i in range(len(motor_order))}

    def temperature_monitor_log(self) -> dict[str,list]:
        return self.bus.temperature_monitor_log()

    def electronic_emergency_stop(self) -> None:
        self.bus.electronic_emergency_stop()

    def temperature_monitor_loop(
            self,
            events: dict,
            temperature_limit: float,
            check_interval: float,
            stop_event: threading.Event,
            on_overheat: Callable[[float, float], None] | None = None,
            display_temperature: bool = False,   # 是否持续在顶部显示温度
        ):
        self.bus.temperature_monitor_loop(events,
                            temperature_limit,
                            check_interval,
                            stop_event,
                            on_overheat,
                            display_temperature,
                            )
