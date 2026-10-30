# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Opt-in follower MIT position control for gain experiments.

The SDK executes PD. The explicitly chosen constant torque bias is not a gravity
model or a limiter for the combined PD and feed-forward output.
"""

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _six_finite(name: str, values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None or len(values) != 6:
        raise ValueError(f"{name} must explicitly contain six joint values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


@dataclass(frozen=True)
class MITPositionSettings:
    """Explicit experimental position-control parameters, in SDK units."""

    kp: Sequence[float]
    kd: Sequence[float]
    torque_ff: Sequence[float]
    max_tracking_error: float = 0.1
    feedback_max_age_s: float = 0.2

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "torque_ff"):
            object.__setattr__(self, name, _six_finite(name, getattr(self, name)))
        if not all(0 < value <= 500 for value in self.kp):
            raise ValueError("mit_kp must be in (0, 500] for position control")
        if not all(0 < value <= 5 for value in self.kd):
            raise ValueError("mit_kd must be in (0, 5] for position control")
        if not all(abs(value) <= 16 for value in self.torque_ff):
            raise ValueError("mit_torque_ff must be within the v189 SDK range [-16, 16]")
        for name in ("max_tracking_error", "feedback_max_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class PiperMITPositionController:
    """Send validated position-impedance commands without taking CAN ownership.

    All six references are checked before sending any MIT frame. Velocity
    references are zero: Kd damps measured velocity. Keep this convention fixed
    during a gain sweep. Error/age checks reject new commands; they do not
    implement a hardware emergency stop.
    """

    def __init__(
        self,
        sdk: Any,
        gripper: Any,
        joint_limits: Mapping[str, Sequence[float]],
        settings: MITPositionSettings,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sdk = sdk
        self._gripper = gripper
        self._settings = settings
        self._clock = clock
        self._firmware_verified = False
        self._limits: list[tuple[float, float]] = []
        for index in range(1, 7):
            limits = joint_limits.get(f"joint{index}")
            if limits is None or len(limits) != 2:
                raise ValueError(f"SDK joint{index} limits are required for MIT experiments")
            lower, upper = (float(value) for value in limits)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid SDK joint{index} limits")
            self._limits.append((lower, upper))

    def _verify_firmware(self) -> None:
        if self._firmware_verified:
            return
        info = self._sdk.get_firmware()
        version = str(info.get("software_version", "")) if isinstance(info, dict) else ""
        match = re.search(r"(\d+)[._-](\d+)[._-](\d+)", version)
        if match is None or tuple(map(int, match.groups())) < (1, 8, 9):
            raise RuntimeError(f"MIT gain experiments require verified firmware >= S-V1.8-9; got {version!r}")
        self._firmware_verified = True

    def send(self, command: Sequence[float]) -> None:
        """Send six joint targets (rad) and one gripper stroke (m)."""
        if len(command) != 7:
            raise ValueError("Piper commands must contain six joints and one gripper value")
        values = [float(value) for value in command]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Piper MIT commands must be finite")
        for index, (position, (lower, upper)) in enumerate(zip(values[:6], self._limits, strict=True), 1):
            if not lower <= position <= upper:
                raise ValueError(f"MIT joint {index} target {position} outside SDK limits [{lower}, {upper}]")

        self._verify_firmware()
        sample = self._sdk.get_joint_angles()
        if sample is None:
            raise RuntimeError("No joint feedback available for MIT gain experiment")
        timestamp = float(sample.timestamp)
        age = self._clock() - timestamp
        if not math.isfinite(timestamp) or timestamp <= 0 or not 0 <= age <= self._settings.feedback_max_age_s:
            raise RuntimeError(f"MIT joint feedback is stale or invalid (age={age:.3f}s)")
        measured = _six_finite("measured joints", sample.msg)
        errors = [abs(target - actual) for target, actual in zip(values[:6], measured, strict=True)]
        if max(errors) > self._settings.max_tracking_error:
            index = errors.index(max(errors)) + 1
            raise RuntimeError(
                f"MIT joint {index} target/feedback error {max(errors):.4f} rad exceeds "
                f"mit_max_tracking_error={self._settings.max_tracking_error:.4f}; no new command sent"
            )

        for index in range(6):
            self._sdk.move_mit(
                joint_index=index + 1,
                p_des=values[index],
                v_des=0.0,
                kp=self._settings.kp[index],
                kd=self._settings.kd[index],
                t_ff=self._settings.torque_ff[index],
            )
        self._gripper.move_gripper_m(value=abs(values[6]), force=1.0)
