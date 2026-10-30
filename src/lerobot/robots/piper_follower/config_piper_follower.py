#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors.piper.mit_controller import MITPositionSettings

from ..config import RobotConfig

@RobotConfig.register_subclass("piper_follower")
@dataclass
class PIPERFollowerConfig(RobotConfig):
    #cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top": OpenCVCameraConfig(
                index_or_path="/dev/video_top",
                fps=30,
                width=480,
                height=640,
                rotation=90,  # 无旋转（可按需调整 90/180/270）
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video_wrist",
                fps=30,
                width=640,
                height=480,
                rotation=0,
            ),
            "down": OpenCVCameraConfig(
                index_or_path="/dev/video_down",
                fps=30,
                width=640,
                height=480,
                rotation=0,
            ),
        }
    )
    # EMA on teleop targets before JointCtrl: smaller = smoother, more lag. 1.0 = off.
    teleop_joint_alpha: float = 0.25
    teleop_gripper_alpha: float = 0.5

    # "native" preserves move_j for ordinary clients and move_js for DAgger.
    # "mit" selects an experimental follower-only backend; all vectors are explicit.
    joint_control_mode: str = "native"
    mit_kp: list[float] | None = None
    mit_kd: list[float] | None = None
    # Constant SDK torque bias, NOT an automatic gravity-compensation model.
    mit_torque_ff: list[float] | None = None
    mit_max_tracking_error: float = 0.1
    mit_feedback_max_age_s: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if self.joint_control_mode not in ("native", "mit"):
            raise ValueError("joint_control_mode must be 'native' or 'mit'")
        if self.joint_control_mode == "mit":
            MITPositionSettings(
                self.mit_kp, self.mit_kd, self.mit_torque_ff,
                self.mit_max_tracking_error, self.mit_feedback_max_age_s,
            )
        elif any(value is not None for value in (self.mit_kp, self.mit_kd, self.mit_torque_ff)):
            raise ValueError("Set joint_control_mode='mit' to use MIT gains; native mode does not apply them")
