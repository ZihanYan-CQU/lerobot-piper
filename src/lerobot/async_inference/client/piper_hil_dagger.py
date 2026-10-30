# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Piper-specific human-in-the-loop primitives for remote RTC DAgger.

This module intentionally does not change the generic Piper robot or teleoperator
classes.  It implements the dynamic handover recommended by Piper's official
human-in-the-loop note:

* autonomous: both arms receive the same position targets; the follower can use
  its optional explicit MIT gain backend, while the master keeps ``move_js``;
* correction: computer commands stop, the master becomes a leader, and physical
  control frames (0x151--0x159) are bridged from ``can_master`` to ``can_follower``;
* resume: the bridge stops first, the master returns to follower mode, a fresh
  follower feedback frame is observed, and both arms are held at the corrected
  pose before policy execution is allowed to continue.

The CAN bridge ignores SocketCAN local-loopback messages (``Message.is_rx=False``).
That is a safety boundary: without it, SDK commands written on ``can_master`` by
this process could be reflected into the bridge and contend with normal control.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

PIPER_ACTION_KEYS = tuple([f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"])
PIPER_CONTROL_IDS = tuple(range(0x151, 0x15A))
PIPER_EXPERT_CONTROL_IDS = (0x155, 0x156, 0x157, 0x159)
PIPER_FEEDBACK_IDS = tuple(range(0x2A1, 0x2A9))
PIPER_POSE_FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A8)


class PiperDaggerHardwareError(RuntimeError):
    """Raised when a handover cannot be completed with a known-safe state."""


class PiperHandoverAlignmentError(PiperDaggerHardwareError):
    """The closed takeover gate rejected a physically misaligned leader."""


class PiperControlMode(StrEnum):
    """Internal ownership of the two Piper arms."""

    SHADOW = "shadow"
    PAUSED = "paused"
    CORRECTING = "correcting"
    CLOSED = "closed"


@dataclass(frozen=True)
class PiperDaggerHardwareConfig:
    master_can_name: str = "can_master"
    follower_can_name: str = "can_follower"
    bridge_receive_timeout_s: float = 0.01
    feedback_timeout_s: float = 1.0
    minimum_firmware: str = "1.8.9"
    require_safe_firmware: bool = True
    require_matching_firmware: bool = True
    require_master_enabled: bool = True
    master_enable_timeout_s: float = 5.0
    follower_first: bool = True
    debug_command_timing: bool = False
    handover_alignment_timeout_s: float = 2.0
    handover_joint_tolerance: float = 0.035
    handover_gripper_tolerance: float = 0.001
    handover_gripper_settle_s: float = 0.05

    def __post_init__(self) -> None:
        if not self.master_can_name:
            raise ValueError("master_can_name cannot be empty")
        if not self.follower_can_name:
            raise ValueError("follower_can_name cannot be empty")
        if self.master_can_name == self.follower_can_name:
            raise ValueError("master and follower must use different CAN interfaces")
        if self.bridge_receive_timeout_s <= 0:
            raise ValueError("bridge_receive_timeout_s must be positive")
        if self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be positive")
        if self.master_enable_timeout_s <= 0:
            raise ValueError("master_enable_timeout_s must be positive")
        if self.handover_alignment_timeout_s <= 0:
            raise ValueError("handover_alignment_timeout_s must be positive")
        if self.handover_joint_tolerance <= 0:
            raise ValueError("handover_joint_tolerance must be positive")
        if self.handover_gripper_tolerance <= 0:
            raise ValueError("handover_gripper_tolerance must be positive")
        if self.handover_gripper_settle_s < 0:
            raise ValueError("handover_gripper_settle_s must be non-negative")
        _parse_firmware_version(self.minimum_firmware)


@dataclass(frozen=True)
class DaggerDatasetConfig:
    """Correction-only LeRobot dataset settings.

    ``repo_id=None`` disables recording while retaining the complete takeover
    state machine.  This is useful for hardware validation before collecting a
    real DAgger dataset.
    """

    repo_id: str | None = None
    root: str | Path | None = None
    fps: int = 30
    use_videos: bool = True
    resume: bool = False
    push_to_hub: bool = False
    private: bool | None = None
    image_writer_processes: int = 0
    image_writer_threads_per_camera: int = 4

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("DAgger dataset fps must be positive")
        if self.image_writer_processes < 0:
            raise ValueError("image_writer_processes must be non-negative")
        if self.image_writer_threads_per_camera < 0:
            raise ValueError("image_writer_threads_per_camera must be non-negative")
        if self.resume and self.root is None:
            raise ValueError("resuming a DAgger dataset requires an explicit root")


def _parse_firmware_version(value: str) -> tuple[int, ...]:
    """Parse variants such as ``S-V1.8-9`` and ``1.9.0`` for ordering."""

    match = re.search(r"(\d+)[._-](\d+)[._-](\d+)", value)
    if match is None:
        raise ValueError(f"cannot parse Piper firmware version {value!r}")
    return tuple(int(part) for part in match.groups())


def _clone_can_message(
    can_module: Any,
    message: Any,
    *,
    data: bytearray | None = None,
) -> Any:
    """Copy only portable python-can message fields for transmission."""

    kwargs: dict[str, Any] = {
        "arbitration_id": int(message.arbitration_id),
        "is_extended_id": bool(getattr(message, "is_extended_id", False)),
        "data": bytearray(message.data) if data is None else data,
        "is_remote_frame": bool(getattr(message, "is_remote_frame", False)),
        "is_error_frame": False,
    }
    is_fd = bool(getattr(message, "is_fd", False))
    if is_fd:
        kwargs.update(
            is_fd=True,
            bitrate_switch=bool(getattr(message, "bitrate_switch", False)),
            error_state_indicator=bool(getattr(message, "error_state_indicator", False)),
        )
    return can_module.Message(**kwargs)


class PiperCANBridgeDagger:
    """One-way, phase-gated SocketCAN bridge for physical Piper control frames."""

    def __init__(
        self,
        master_channel: str,
        follower_channel: str,
        *,
        receive_timeout_s: float = 0.01,
        can_module: Any | None = None,
        bus_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.master_channel = master_channel
        self.follower_channel = follower_channel
        self.receive_timeout_s = receive_timeout_s
        self._can = can_module
        self._bus_factory = bus_factory
        self._master_bus: Any | None = None
        self._follower_bus: Any | None = None
        self._enabled = threading.Event()
        self._forward_lock = threading.Lock()
        self._receive_lock = threading.Lock()
        self._forward_generation = 0
        self._shutdown = threading.Event()
        self._started = threading.Event()
        self._error_lock = threading.Lock()
        self._last_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self.forwarded_frames = 0
        self.dropped_local_frames = 0

    @staticmethod
    def should_forward(message: Any) -> bool:
        """Accept only physical standard Piper control frames.

        SocketCAN reports frames produced by another local socket with
        ``is_rx=False``.  Requiring an explicit physical receive indication is
        intentional; an unknown backend must fail closed instead of forwarding a
        possible local command echo.
        """

        return (
            getattr(message, "is_rx", None) is True
            and not bool(getattr(message, "is_error_frame", False))
            and not bool(getattr(message, "is_extended_id", False))
            and not bool(getattr(message, "is_remote_frame", False))
            and not bool(getattr(message, "is_fd", False))
            and int(getattr(message, "arbitration_id", -1)) in PIPER_CONTROL_IDS
        )

    def _load_can(self) -> Any:
        if self._can is None:
            try:
                import can
            except ImportError as exc:  # pragma: no cover - hardware dependency
                raise PiperDaggerHardwareError(
                    "python-can is required for Piper DAgger CAN forwarding"
                ) from exc
            self._can = can
        return self._can

    def _make_bus(self, channel: str, *, receive: bool) -> Any:
        can_module = self._load_can()
        factory = self._bus_factory or can_module.Bus
        kwargs: dict[str, Any] = {
            "interface": "socketcan",
            "channel": channel,
            "receive_own_messages": False,
        }
        if receive:
            kwargs["can_filters"] = [
                {"can_id": arbitration_id, "can_mask": 0x7FF, "extended": False}
                for arbitration_id in PIPER_CONTROL_IDS
            ]
        return factory(**kwargs)

    def start(self) -> None:
        if self._thread is not None:
            return
        with self._error_lock:
            self._last_error = None
        self._master_bus = self._make_bus(self.master_channel, receive=True)
        try:
            self._follower_bus = self._make_bus(self.follower_channel, receive=False)
        except Exception:
            self._shutdown_bus(self._master_bus)
            self._master_bus = None
            raise
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="PiperDaggerCANBridge")
        self._thread.start()
        if not self._started.wait(timeout=1.0):
            self.close()
            raise PiperDaggerHardwareError("Piper CAN bridge thread did not start")

    def enable(self) -> None:
        # Use forward -> receive lock order. The worker releases receive before
        # taking forward, so an old frame may be waiting outside the queue here.
        # Advance its generation and drain/open atomically against new receives.
        with self._forward_lock, self._receive_lock:
            self._enabled.clear()
            self.raise_if_unhealthy()
            self._forward_generation += 1
            self._drain_locked()
            self._enabled.set()

    def disable(self) -> None:
        # Return only after a send already in progress has completed.  Once this
        # method returns, no further frame can cross the bridge until enable().
        with self._forward_lock:
            self._enabled.clear()

    def drain(self) -> int:
        """Drop queued master frames so a correction never starts with stale input."""

        with self._receive_lock:
            return self._drain_locked()

    def _drain_locked(self) -> int:
        """Drain while the caller owns the receive lock."""

        bus = self._master_bus
        if bus is None:
            return 0
        drained = 0
        while True:
            message = bus.recv(timeout=0.0)
            if message is None:
                return drained
            drained += 1

    @property
    def last_error(self) -> Exception | None:
        with self._error_lock:
            return self._last_error

    def _record_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._last_error is None:
                self._last_error = exc
        self._enabled.clear()

    def raise_if_unhealthy(self) -> None:
        error = self.last_error
        thread = self._thread
        if error is not None:
            raise PiperDaggerHardwareError(f"Piper CAN bridge failed: {error}") from error
        if thread is None or not thread.is_alive():
            raise PiperDaggerHardwareError("Piper CAN bridge is not running")

    def _run(self) -> None:
        self._started.set()
        while not self._shutdown.is_set():
            bus = self._master_bus
            if bus is None:
                return
            try:
                with self._receive_lock:
                    generation = self._forward_generation
                    message = bus.recv(timeout=self.receive_timeout_s)
            except Exception as exc:
                if not self._shutdown.is_set():
                    self._record_error(exc)
                    logger.exception("Piper DAgger CAN receive failed; bridge stopped")
                return
            if message is None:
                continue
            with self._forward_lock:
                if not self._enabled.is_set() or generation != self._forward_generation:
                    continue
                if not self.should_forward(message):
                    if getattr(message, "is_rx", None) is False:
                        self.dropped_local_frames += 1
                    continue
                try:
                    outgoing = _clone_can_message(self._load_can(), message)
                    self._follower_bus.send(outgoing, timeout=self.receive_timeout_s)
                    self.forwarded_frames += 1
                except Exception as exc:
                    self._record_error(exc)
                    logger.exception("Piper DAgger CAN forwarding failed; disabling bridge")
                    return

    @staticmethod
    def _shutdown_bus(bus: Any | None) -> None:
        if bus is None:
            return
        shutdown = getattr(bus, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def close(self) -> None:
        self.disable()
        self._shutdown.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
        self._started.clear()
        self._shutdown_bus(self._master_bus)
        self._shutdown_bus(self._follower_bus)
        self._master_bus = None
        self._follower_bus = None

    def _wait_for_feedback(self, channel: str, arm_name: str, timeout_s: float) -> int:
        """Wait for a fresh physical feedback frame on one Piper CAN channel."""

        can_module = self._load_can()
        factory = self._bus_factory or can_module.Bus
        monitor = factory(
            interface="socketcan",
            channel=channel,
            receive_own_messages=False,
            can_filters=[
                {"can_id": arbitration_id, "can_mask": 0x7FF, "extended": False}
                for arbitration_id in PIPER_FEEDBACK_IDS
            ],
        )
        try:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                message = monitor.recv(timeout=min(self.receive_timeout_s, remaining))
                if (
                    message is not None
                    and getattr(message, "is_rx", None) is True
                    and not bool(getattr(message, "is_error_frame", False))
                    and int(getattr(message, "arbitration_id", -1)) in PIPER_FEEDBACK_IDS
                ):
                    return int(message.arbitration_id)
        finally:
            self._shutdown_bus(monitor)
        raise PiperDaggerHardwareError(
            f"no fresh Piper {arm_name} feedback on {channel!r} within {timeout_s:.3f}s"
        )

    def wait_for_master_feedback(self, timeout_s: float) -> int:
        """Confirm that the physical master is emitting follower-mode feedback."""

        return self._wait_for_feedback(self.master_channel, "master", timeout_s)

    def wait_for_follower_feedback(self, timeout_s: float) -> int:
        """Confirm that the physical follower is emitting fresh feedback."""

        return self._wait_for_feedback(self.follower_channel, "follower", timeout_s)

    def _read_pose(
        self,
        channel: str,
        frame_ids: tuple[int, ...],
        timeout_s: float,
        *,
        trigger: Callable[[], None] | None = None,
        reference: dict[str, float] | None = None,
        joint_tolerance: float = 0.035,
        gripper_tolerance: float = 0.001,
        settle_s: float = 0.0,
    ) -> dict[str, float]:
        """Read physical CAN samples, never a pre-switch SDK cache.

        With a reference, require all seven absolute values to stay aligned
        throughout the settling interval. A leader may only publish on change;
        silence is allowed after its first complete fresh sample, while any
        subsequent out-of-tolerance frame restarts the alignment interval.
        """

        can_module = self._load_can()
        factory = self._bus_factory or can_module.Bus
        monitor = factory(
            interface="socketcan",
            channel=channel,
            receive_own_messages=False,
            can_filters=[
                {"can_id": arbitration_id, "can_mask": 0x7FF, "extended": False}
                for arbitration_id in frame_ids
            ],
        )
        seen: set[int] = set()
        pose: dict[str, float] = {}
        stable_since: float | None = None
        first_gripper: float | None = None
        joint_error = gripper_error = float("inf")
        try:
            if trigger is not None:
                trigger()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                message = monitor.recv(timeout=min(self.receive_timeout_s, remaining))
                arbitration_id = int(getattr(message, "arbitration_id", -1))
                if (
                    message is not None
                    and getattr(message, "is_rx", None) is True
                    and not bool(getattr(message, "is_error_frame", False))
                    and not bool(getattr(message, "is_extended_id", False))
                    and not bool(getattr(message, "is_remote_frame", False))
                    and not bool(getattr(message, "is_fd", False))
                    and arbitration_id in frame_ids
                ):
                    data = bytes(message.data)
                    if len(data) != 8:
                        raise PiperDaggerHardwareError(f"invalid Piper pose frame 0x{arbitration_id:03X}")
                    if arbitration_id == frame_ids[-1]:
                        if (arbitration_id == 0x2A8 and data[7] != 0) or (
                            arbitration_id == 0x159 and (data[6] & 0x04 or data[7] != 0)
                        ):
                            raise PiperDaggerHardwareError(
                                "takeover requires width-mode gripper frames without zeroing"
                            )
                        pose["gripper.pos"] = int.from_bytes(data[:4], "big", signed=True) * 1e-6
                        if first_gripper is None:
                            first_gripper = pose["gripper.pos"]
                    else:
                        joint_index = frame_ids.index(arbitration_id) * 2
                        for index in range(2):
                            raw = int.from_bytes(data[index * 4 : index * 4 + 4], "big", signed=True)
                            pose[PIPER_ACTION_KEYS[joint_index + index]] = math.radians(raw * 0.001)
                    seen.add(arbitration_id)
                    if len(seen) == len(frame_ids):
                        if reference is None:
                            return pose
                        joint_error = max(abs(pose[key] - reference[key]) for key in PIPER_ACTION_KEYS[:6])
                        gripper_error = abs(pose["gripper.pos"] - reference["gripper.pos"])
                        if joint_error <= joint_tolerance and gripper_error <= gripper_tolerance:
                            if stable_since is None:
                                stable_since = time.monotonic()
                        else:
                            stable_since = None
                if stable_since is not None and time.monotonic() - stable_since >= settle_s:
                    logger.info(
                        "Piper absolute handover aligned: follower_gripper=%.4f m, "
                        "first_leader_gripper=%.4f m, last_leader_gripper=%.4f m, "
                        "joint_error=%.4f rad, gripper_error=%.4f m",
                        reference["gripper.pos"],
                        first_gripper,
                        pose["gripper.pos"],
                        joint_error,
                        gripper_error,
                    )
                    return pose
        finally:
            self._shutdown_bus(monitor)

        if reference is not None and len(seen) == len(frame_ids):
            raise PiperHandoverAlignmentError(
                "leader moved away from the absolute handover pose; bridge remains closed "
                f"(follower_gripper={reference['gripper.pos']:.4f} m, "
                f"first_leader_gripper={first_gripper:.4f} m, "
                f"last_leader_gripper={pose['gripper.pos']:.4f} m, "
                f"joint_error={joint_error:.4f} rad, gripper_error={gripper_error:.4f} m)"
            )
        missing = sorted(set(frame_ids) - seen)
        missing_text = ", ".join(f"0x{arbitration_id:03X}" for arbitration_id in missing)
        raise PiperDaggerHardwareError(
            f"no complete fresh Piper pose on {channel!r} within {timeout_s:.3f}s; missing {missing_text}"
        )

    def read_follower_feedback_pose(self, timeout_s: float) -> dict[str, float]:
        return self._read_pose(self.follower_channel, PIPER_POSE_FEEDBACK_IDS, timeout_s)

    def read_master_feedback_pose(self, timeout_s: float) -> dict[str, float]:
        return self._read_pose(self.master_channel, PIPER_POSE_FEEDBACK_IDS, timeout_s)

    def wait_for_master_alignment(
        self,
        reference: dict[str, float],
        config: PiperDaggerHardwareConfig,
        *,
        trigger: Callable[[], None],
    ) -> dict[str, float]:
        return self._read_pose(
            self.master_channel,
            PIPER_EXPERT_CONTROL_IDS,
            config.handover_alignment_timeout_s,
            trigger=trigger,
            reference=reference,
            joint_tolerance=config.handover_joint_tolerance,
            gripper_tolerance=config.handover_gripper_tolerance,
            settle_s=config.handover_gripper_settle_s,
        )


class PiperDaggerController:
    """Own dual-arm command mirroring and safe dynamic handover sequencing."""

    def __init__(
        self,
        follower_robot: Any,
        config: PiperDaggerHardwareConfig,
        *,
        master_bus: Any | None = None,
        bridge: PiperCANBridgeDagger | Any | None = None,
    ) -> None:
        self.follower_robot = follower_robot
        self.follower_bus = getattr(follower_robot, "bus", None)
        if self.follower_bus is None:
            raise TypeError("Piper DAgger requires a follower robot exposing .bus")
        if tuple(follower_robot.action_features) != PIPER_ACTION_KEYS:
            raise ValueError(
                "Piper DAgger requires the standard seven Piper action keys in order; "
                f"got {tuple(follower_robot.action_features)!r}"
            )
        self.config = config
        self._lock = threading.RLock()
        self._mode = PiperControlMode.PAUSED
        self._ema_command: list[float] | None = None
        self._closed = False

        self.master_bus = master_bus if master_bus is not None else self._create_master_bus(config)
        self.bridge = bridge or PiperCANBridgeDagger(
            config.master_can_name,
            config.follower_can_name,
            receive_timeout_s=config.bridge_receive_timeout_s,
        )

        try:
            self.bridge.start()
            self.bridge.disable()
            self._set_master_follower_mode()
            self._validate_firmware()
            pose = self.read_follower_pose()
            self.seed_command(pose)
            self._write_same_unfiltered(pose)
            self._mode = PiperControlMode.SHADOW
        except Exception:
            # Initialization may fail before firmware/mode safety has been
            # established.  Disconnect that partial setup without commanding
            # a new motion; normal program shutdown still returns the master
            # home via close()'s default.
            self.close(return_master_home=False)
            raise

    @staticmethod
    def _create_master_bus(config: PiperDaggerHardwareConfig) -> Any:
        try:
            from lerobot.motors.piper.piper_PyAgxArm import PiperMotorsBus, PiperMotorsBusConfig
        except ImportError as exc:  # pragma: no cover - hardware dependency
            raise PiperDaggerHardwareError("pyAgxArm/Piper motor support is unavailable") from exc

        return PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.master_can_name,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                },
            )
        )

    @property
    def mode(self) -> PiperControlMode:
        with self._lock:
            return self._mode

    def _set_master_leader_mode(self) -> None:
        """Switch mode without the legacy wrapper's fixed one-second sleep."""

        direct = getattr(getattr(self.master_bus, "piper", None), "set_leader_mode", None)
        if callable(direct):
            direct()
            logger.info("Piper master switched to leader mode")
            return
        self.master_bus.change_to_leader_mode()

    def _set_master_follower_mode(self) -> None:
        """Switch mode without the legacy wrapper's fixed one-second sleep."""

        direct = getattr(getattr(self.master_bus, "piper", None), "set_follower_mode", None)
        if callable(direct):
            direct()
            logger.info("Piper master switched to follower mode")
        else:
            self.master_bus.change_to_follower_mode()
        self._ensure_master_enabled()

    def _ensure_master_enabled(self) -> None:
        """Verify the shadow arm can actually execute the mirrored MIT stream."""

        piper = getattr(self.master_bus, "piper", None)
        get_status = getattr(piper, "get_joint_enable_status", None)
        enable = getattr(piper, "enable", None)
        if not callable(get_status) or not callable(enable):
            if self.config.require_master_enabled:
                raise PiperDaggerHardwareError("Piper SDK cannot verify/enable the master follower arm")
            logger.warning("Piper master enable state could not be verified")
            return

        deadline = time.monotonic() + self.config.master_enable_timeout_s
        while time.monotonic() < deadline:
            try:
                if bool(get_status(255)):
                    return
                if bool(enable()) or bool(get_status(255)):
                    return
            except Exception:
                logger.debug("Piper master enable attempt failed", exc_info=True)
            time.sleep(0.05)
        raise PiperDaggerHardwareError(
            f"Piper master did not enable within {self.config.master_enable_timeout_s:.1f}s"
        )

    def _firmware_version(self, bus: Any) -> str | None:
        piper = getattr(bus, "piper", None)
        get_firmware = getattr(piper, "get_firmware", None)
        if not callable(get_firmware):
            return None
        for _ in range(10):
            try:
                info = get_firmware()
                if isinstance(info, dict) and info.get("software_version"):
                    return str(info["software_version"])
            except Exception:
                pass
            time.sleep(0.2)
        return None

    def _validate_firmware(self) -> None:
        minimum = _parse_firmware_version(self.config.minimum_firmware)
        versions = {
            "master": self._firmware_version(self.master_bus),
            "follower": self._firmware_version(self.follower_bus),
        }
        parsed_versions: dict[str, tuple[int, ...]] = {}
        for arm, version in versions.items():
            if version is None:
                message = f"could not read {arm} Piper firmware"
                if self.config.require_safe_firmware:
                    raise PiperDaggerHardwareError(message)
                logger.warning("%s; dynamic handover firmware was not verified", message)
                continue
            try:
                parsed = _parse_firmware_version(version)
            except ValueError as exc:
                if self.config.require_safe_firmware:
                    raise PiperDaggerHardwareError(str(exc)) from exc
                logger.warning("%s", exc)
                continue
            if parsed < minimum:
                raise PiperDaggerHardwareError(
                    f"{arm} firmware {version!r} is older than required {self.config.minimum_firmware!r}"
                )
            parsed_versions[arm] = parsed
            logger.info("Verified Piper %s firmware %s", arm, version)
        if (
            self.config.require_matching_firmware
            and len(parsed_versions) == 2
            and parsed_versions["master"] != parsed_versions["follower"]
        ):
            raise PiperDaggerHardwareError(
                "Piper master/follower firmware must match for dynamic takeover; "
                f"master={versions['master']!r}, follower={versions['follower']!r}"
            )

    def _action_values(self, action: dict[str, float]) -> list[float]:
        missing = [key for key in PIPER_ACTION_KEYS if key not in action]
        if missing:
            raise KeyError(f"Piper action is missing {missing}")
        return [float(action[key]) for key in PIPER_ACTION_KEYS]

    @staticmethod
    def _action_dict(values: list[float]) -> dict[str, float]:
        return dict(zip(PIPER_ACTION_KEYS, values, strict=True))

    def read_follower_pose(self) -> dict[str, float]:
        return self.follower_bus.read_action_positions(self.follower_bus.read_follower())

    def read_master_pose(self) -> dict[str, float]:
        """Read the master's physical feedback while it is in follower mode."""

        return self.master_bus.read_action_positions(self.master_bus.read_follower())

    def read_reset_feedback(self, timeout_s: float) -> dict[str, dict[str, float]]:
        """Read complete fresh physical feedback from both arms after reset commands."""

        with self._lock:
            if self._closed or self._mode is PiperControlMode.CORRECTING:
                raise PiperDaggerHardwareError("reset feedback requires both arms in follower mode")
            deadline = time.monotonic() + timeout_s
            poses = {}
            for arm, read in (
                ("follower", self.bridge.read_follower_feedback_pose),
                ("master", self.bridge.read_master_feedback_pose),
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("reset feedback deadline expired")
                poses[arm] = read(min(self.config.feedback_timeout_s, remaining))
            return poses

    def _read_raw_expert_action(self) -> dict[str, float]:
        """Read one complete, unmodified master control sample."""

        try_reader = getattr(self.master_bus, "try_read_leader", None)
        if not callable(try_reader):
            return self.master_bus.read_action_positions(self.master_bus.read_leader())

        deadline = time.monotonic() + self.config.feedback_timeout_s
        while time.monotonic() < deadline:
            raw_action = try_reader()
            if raw_action is not None:
                return self.master_bus.read_action_positions(raw_action)
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        raise PiperDaggerHardwareError(
            "Piper master leader-control cache did not become complete within "
            f"{self.config.feedback_timeout_s:.3f}s"
        )

    def read_expert_action(self) -> dict[str, float]:
        """Read the unmodified absolute leader command used by the CAN bridge."""

        return self._read_raw_expert_action()

    def _alignment_errors(
        self,
        reference: dict[str, float],
        candidate: dict[str, float],
    ) -> tuple[float, float]:
        joint_error = max(abs(float(candidate[key]) - float(reference[key])) for key in PIPER_ACTION_KEYS[:6])
        gripper_error = abs(float(candidate[PIPER_ACTION_KEYS[6]]) - float(reference[PIPER_ACTION_KEYS[6]]))
        return joint_error, gripper_error

    def _is_handover_aligned(
        self,
        reference: dict[str, float],
        candidate: dict[str, float],
    ) -> tuple[bool, float, float]:
        joint_error, gripper_error = self._alignment_errors(reference, candidate)
        aligned = (
            joint_error <= self.config.handover_joint_tolerance
            and gripper_error <= self.config.handover_gripper_tolerance
        )
        return aligned, joint_error, gripper_error

    def _align_master_to_follower(self) -> dict[str, float]:
        """Align the actuated master to actual follower feedback before takeover.

        Equal policy commands do not imply equal physical gripper openings: the
        follower may be blocked by a grasped object while the unloaded master
        closes fully.  Sending the follower's measured pose to both arms holds
        the follower where it is and moves the master to the true takeover
        origin, preventing the first leader command from tightening the grasp.
        """

        target = self.bridge.read_follower_feedback_pose(self.config.feedback_timeout_s)
        self.seed_command(target)
        self._write_same_unfiltered(target)

        deadline = time.monotonic() + self.config.handover_alignment_timeout_s
        joint_error = float("inf")
        gripper_error = float("inf")
        stable_since: float | None = None
        while time.monotonic() < deadline:
            master_pose = self.bridge.read_master_feedback_pose(
                min(self.config.feedback_timeout_s, max(0.0, deadline - time.monotonic()))
            )
            aligned, joint_error, gripper_error = self._is_handover_aligned(target, master_pose)
            if aligned:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= self.config.handover_gripper_settle_s:
                    logger.info(
                        "Piper physical alignment in follower mode "
                        "(follower_gripper=%.4f m, master_gripper=%.4f m, "
                        "joint_error=%.4f rad, gripper_error=%.4f m)",
                        target["gripper.pos"],
                        master_pose["gripper.pos"],
                        joint_error,
                        gripper_error,
                    )
                    return target
            else:
                stable_since = None
            # Reassert only the master's target.  The follower already holds its
            # measured pose, so repeated alignment commands cannot tighten it.
            self.master_bus.write_js(self._action_values(target))
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

        raise PiperDaggerHardwareError(
            "Piper master did not align to follower before correction within "
            f"{self.config.handover_alignment_timeout_s:.3f}s "
            f"(joint_error={joint_error:.4f} rad, gripper_error={gripper_error:.4f} m)"
        )

    def _wait_for_expert_cache(self, reference: dict[str, float]) -> None:
        """Wait for the recorder's independent SDK receiver to catch up.

        This is not the handover safety check: that check must first observe
        complete, aligned physical control frames on a fresh CAN socket.
        """

        deadline = time.monotonic() + self.config.feedback_timeout_s
        while time.monotonic() < deadline:
            if self._is_handover_aligned(reference, self._read_raw_expert_action())[0]:
                return
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        raise PiperDaggerHardwareError(
            "Piper SDK leader cache did not match the verified absolute handover pose; bridge remains closed"
        )

    def seed_command(self, pose: dict[str, float]) -> None:
        with self._lock:
            self._ema_command = self._action_values(pose)
            # Keep the legacy follower object's private filter from carrying a
            # pre-intervention command if another caller later uses it directly.
            if hasattr(self.follower_robot, "_ema_cmd"):
                self.follower_robot._ema_cmd = list(self._ema_command)

    def _smooth(self, target: list[float]) -> list[float]:
        joint_alpha = min(
            1.0, max(0.0, float(getattr(self.follower_robot.config, "teleop_joint_alpha", 1.0)))
        )
        gripper_alpha = min(
            1.0,
            max(0.0, float(getattr(self.follower_robot.config, "teleop_gripper_alpha", 1.0))),
        )
        if joint_alpha <= 0:
            joint_alpha = 1.0
        if gripper_alpha <= 0:
            gripper_alpha = 1.0
        if self._ema_command is None:
            self._ema_command = self._action_values(self.read_follower_pose())
        for index in range(6):
            self._ema_command[index] = (
                joint_alpha * target[index] + (1.0 - joint_alpha) * self._ema_command[index]
            )
        self._ema_command[6] = gripper_alpha * target[6] + (1.0 - gripper_alpha) * self._ema_command[6]
        return list(self._ema_command)

    def _write_same_values(self, values: list[float], *, use_follower_gains: bool = False) -> None:
        # Only autonomous policy commands use the experimental follower backend.
        # Native holds establish the existing JS handover/bridge mode.
        tuned = use_follower_gains and getattr(
            self.follower_robot.config, "joint_control_mode", "native"
        ) == "mit"
        follower_write = (
            self.follower_robot.write_joint_command if tuned else self.follower_bus.write_js
        )
        if self.config.debug_command_timing:
            started = time.monotonic()
        if self.config.follower_first:
            follower_write(values)
            self.master_bus.write_js(values)
        else:
            self.master_bus.write_js(values)
            follower_write(values)
        if self.config.debug_command_timing:
            skew_ms = (time.monotonic() - started) * 1000
            logger.debug("Piper dual-arm MIT command dispatched (sequential span %.3f ms)", skew_ms)

    def _write_same_unfiltered(self, action: dict[str, float]) -> dict[str, float]:
        values = self._action_values(action)
        self._write_same_values(values)
        return self._action_dict(values)

    def send_same_action(self, action: dict[str, float]) -> dict[str, float]:
        """Apply one shared EMA result and send exactly that vector to both arms."""

        with self._lock:
            if self._closed or self._mode in (PiperControlMode.CORRECTING, PiperControlMode.CLOSED):
                raise PiperDaggerHardwareError(f"computer command rejected in mode {self._mode.value}")
            values = self._smooth(self._action_values(action))
            self._write_same_values(values, use_follower_gains=self._mode is PiperControlMode.SHADOW)
            return self._action_dict(values)

    def pause_computer_output(self) -> None:
        with self._lock:
            if self._mode is PiperControlMode.CORRECTING:
                raise PiperDaggerHardwareError("cannot pause computer output during correction")
            self.bridge.disable()
            self._set_master_follower_mode()
            self._align_master_to_follower()
            self._mode = PiperControlMode.PAUSED

    def enter_correction(self) -> None:
        """Physically align, switch, then require unmodified absolute alignment.

        A transient failed switch gets one automatic realignment and retry.
        Persistent closure is rejected; no coordinate offset or manual precision
        adjustment is used to make a mismatched teaching pendant appear aligned.
        """

        with self._lock:
            if self._mode is not PiperControlMode.PAUSED:
                raise PiperDaggerHardwareError(f"cannot enter correction from {self._mode.value}")
            self.bridge.disable()
            self.bridge.drain()
            try:
                for attempt in range(2):
                    # Recheck at the actual takeover keypress, not just when
                    # autonomy was paused (possibly many seconds earlier).
                    self._set_master_follower_mode()
                    handover_pose = self._align_master_to_follower()
                    try:
                        self.bridge.wait_for_master_alignment(
                            handover_pose, self.config, trigger=self._set_master_leader_mode
                        )
                        break
                    except PiperHandoverAlignmentError as exc:
                        if attempt == 1:
                            raise
                        logger.warning(
                            "Piper takeover rejected: %s. Automatically realigning the master "
                            "in follower mode and retrying once; bridge remains closed.",
                            exc,
                        )
                self._wait_for_expert_cache(handover_pose)
                # Drop any local mode-switch traffic and only then open the
                # physical master -> follower control-frame gate.
                self.bridge.drain()
                self.bridge.enable()
                self._mode = PiperControlMode.CORRECTING
                logger.info(
                    "Piper correction ready with strict absolute alignment; "
                    "CAN commands and expert labels are unchanged"
                )
            except Exception as exc:
                self.bridge.disable()
                try:
                    self._set_master_follower_mode()
                    self._align_master_to_follower()
                except Exception as restore_exc:
                    # Do not report a recoverable rejection when physical
                    # rollback itself failed; the client must enter FAULT.
                    raise PiperDaggerHardwareError(
                        f"takeover failed ({exc}); could not restore master alignment: {restore_exc}"
                    ) from restore_exc
                self._mode = PiperControlMode.PAUSED
                raise

    def assert_correction_healthy(self) -> None:
        """Raise when the active physical-control path has stopped."""

        with self._lock:
            if self._mode is PiperControlMode.CORRECTING:
                self.bridge.raise_if_unhealthy()

    def leave_correction(self) -> dict[str, float]:
        """Close the bridge, restore follower mode, and establish a fresh hold."""

        with self._lock:
            if self._mode is not PiperControlMode.CORRECTING:
                raise PiperDaggerHardwareError(f"cannot leave correction from {self._mode.value}")
            self.bridge.disable()
            self.bridge.drain()
            self._set_master_follower_mode()
            master_feedback_id = self.bridge.wait_for_master_feedback(self.config.feedback_timeout_s)
            follower_feedback_id = self.bridge.wait_for_follower_feedback(self.config.feedback_timeout_s)
            pose = self.read_follower_pose()
            self.seed_command(pose)
            self._write_same_unfiltered(pose)
            self._mode = PiperControlMode.PAUSED
            logger.info(
                "Piper correction ended after fresh master/follower feedback 0x%03X/0x%03X",
                master_feedback_id,
                follower_feedback_id,
            )
            return pose

    def prepare_autonomy(self) -> dict[str, float]:
        """Establish follower/follower shadow mode and a common command origin."""

        with self._lock:
            if self._mode is PiperControlMode.CORRECTING:
                raise PiperDaggerHardwareError("correction must end before autonomy can resume")
            self.bridge.disable()
            # Reassert the physical master's role for every rollout.  Piper may
            # return to leader mode while READY has no computer command stream,
            # so a previous successful transition is not sufficient evidence.
            self._set_master_follower_mode()
            self.bridge.wait_for_master_feedback(self.config.feedback_timeout_s)
            self.bridge.wait_for_follower_feedback(self.config.feedback_timeout_s)
            pose = self.read_follower_pose()
            self.seed_command(pose)
            self._write_same_unfiltered(pose)
            self._mode = PiperControlMode.SHADOW
            return pose

    def abort_to_paused(self) -> dict[str, float]:
        """Best-effort correction abort used by reset and shutdown."""

        with self._lock:
            self.bridge.disable()
            self.bridge.drain()
            if self._mode is PiperControlMode.CORRECTING:
                self._set_master_follower_mode()
                self.bridge.wait_for_master_feedback(self.config.feedback_timeout_s)
                self.bridge.wait_for_follower_feedback(self.config.feedback_timeout_s)
            pose = self.read_follower_pose()
            self.seed_command(pose)
            self._write_same_unfiltered(pose)
            self._mode = PiperControlMode.PAUSED
            return pose

    def _return_master_home(self) -> None:
        """Home the master through follower control, then restore leader drag mode."""

        follower_home = getattr(self.master_bus, "apply_calibration_follower", None)
        if not callable(follower_home):
            raise PiperDaggerHardwareError(
                "Piper master bus does not expose apply_calibration_follower(); "
                "cannot return it home from follower mode"
            )
        target_values = getattr(self.master_bus, "init_joint_position", None)
        if target_values is None or len(target_values) != len(PIPER_ACTION_KEYS):
            raise PiperDaggerHardwareError("Piper master bus has no valid seven-axis home position")

        logger.info("Switching Piper master to follower mode for controlled home return")
        try:
            self._set_master_follower_mode()
            self.bridge.wait_for_master_feedback(self.config.feedback_timeout_s)
            reached = follower_home(log=False, arm_name="主臂（从臂模式）")
            actual = self.read_master_pose()
            target = self._action_dict([float(value) for value in target_values])
            joint_error, gripper_error = self._alignment_errors(target, actual)
            if reached is False or not self._is_handover_aligned(target, actual)[0]:
                raise PiperDaggerHardwareError(
                    "Piper master did not reach home in follower mode "
                    f"(joint_error={joint_error:.4f} rad, gripper_error={gripper_error:.4f} m)"
                )
            logger.info(
                "Piper master reached home in follower mode (joint_error=%.4f rad, gripper_error=%.4f m)",
                joint_error,
                gripper_error,
            )
        finally:
            self._set_master_leader_mode()
            logger.info("Piper master restored to leader zero-force drag mode")

    def close(self, *, return_master_home: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.bridge.disable()
                self.bridge.close()
            finally:
                try:
                    if return_master_home:
                        self._return_master_home()
                    elif self._mode is PiperControlMode.CORRECTING:
                        self._set_master_follower_mode()
                finally:
                    disconnect = getattr(getattr(self.master_bus, "piper", None), "disconnect", None)
                    if callable(disconnect):
                        disconnect()
                    self._mode = PiperControlMode.CLOSED


class PiperDaggerRobotProxy:
    """Delegate observations to the follower and commands to the dual-arm controller."""

    def __init__(self, follower_robot: Any, controller: PiperDaggerController) -> None:
        self._follower_robot = follower_robot
        self._controller = controller
        self._disconnected = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._follower_robot, name)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self._controller.send_same_action(action)

    def seed_command(self, pose: dict[str, float]) -> None:
        self._controller.seed_command(pose)

    def disconnect(self) -> None:
        if self._disconnected:
            return
        self._disconnected = True
        try:
            self._controller.close()
        finally:
            self._follower_robot.disconnect()


class DaggerCorrectionRecorder:
    """Sample correction frames away from the CAN/control path.

    Each ``begin``/``end(save=True)`` window becomes one LeRobot episode.  The
    label is the master command, while the observation is captured from the
    follower, matching the official corrections-only DAgger strategy.
    """

    def __init__(
        self,
        follower_robot: Any,
        expert_action_reader: Callable[[], dict[str, float]],
        config: DaggerDatasetConfig,
        *,
        observation_lock: threading.Lock | threading.RLock | None = None,
        dataset: Any | None = None,
    ) -> None:
        self.robot = follower_robot
        self.expert_action_reader = expert_action_reader
        self.config = config
        self._observation_lock = observation_lock or threading.Lock()
        self._sample_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._task = ""
        self._frames_in_window = 0
        self._saved_episodes = 0
        self._last_error: BaseException | None = None
        self.features: dict[str, dict] = {}
        self.dataset = dataset

        if config.repo_id is not None and self.dataset is None:
            self.dataset = self._create_dataset()
        if self.dataset is not None:
            self.features = dict(self.dataset.meta.features)

    @property
    def enabled(self) -> bool:
        return self.dataset is not None

    @property
    def active(self) -> bool:
        return self._active.is_set()

    @property
    def frames_in_window(self) -> int:
        with self._state_lock:
            return self._frames_in_window

    @property
    def saved_episodes(self) -> int:
        with self._state_lock:
            return self._saved_episodes

    @property
    def last_error(self) -> BaseException | None:
        with self._state_lock:
            return self._last_error

    def _dataset_features(self) -> dict[str, dict]:
        observation_features = hw_to_dataset_features(
            self.robot.observation_features,
            OBS_STR,
            use_video=self.config.use_videos,
        )
        action_features = hw_to_dataset_features(
            self.robot.action_features,
            ACTION,
            use_video=self.config.use_videos,
        )
        return {
            **observation_features,
            **action_features,
            "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        }

    def _create_dataset(self) -> Any:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - optional dataset stack
            raise RuntimeError("LeRobot dataset dependencies are required for DAgger recording") from exc

        camera_count = len(getattr(self.robot, "cameras", {}))
        writer_threads = self.config.image_writer_threads_per_camera * camera_count
        if self.config.resume:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            # Reject incompatible capture settings before opening the writer or its workers.
            meta = LeRobotDatasetMetadata(self.config.repo_id, root=self.config.root)
            self._validate_resume(meta)
            return LeRobotDataset.resume(
                self.config.repo_id,
                root=self.config.root,
                image_writer_processes=self.config.image_writer_processes,
                image_writer_threads=writer_threads,
            )

        return LeRobotDataset.create(
            self.config.repo_id,
            self.config.fps,
            root=self.config.root,
            robot_type=getattr(self.robot, "name", "piper_follower"),
            features=self._dataset_features(),
            use_videos=self.config.use_videos,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=writer_threads,
        )

    def _validate_resume(self, meta: Any) -> None:
        """Check the existing time base and feature schema before enabling writes."""
        if meta.fps != self.config.fps:
            raise ValueError(f"Dataset fps {meta.fps} does not match recording fps {self.config.fps}")
        expected = self._dataset_features()
        actual_features = set(meta.features) - set(DEFAULT_FEATURES)
        if actual_features != set(expected):
            missing = sorted(set(expected) - actual_features)
            extra = sorted(actual_features - set(expected))
            raise ValueError(
                f"Resumed DAgger dataset features do not match correction capture: {missing=}, {extra=}"
            )
        for key, feature in expected.items():
            actual = meta.features[key]
            if (
                actual["dtype"] != feature["dtype"]
                or tuple(actual["shape"]) != tuple(feature["shape"])
                or actual.get("names") != feature.get("names")
            ):
                raise ValueError(f"Resumed DAgger dataset feature does not match correction capture: {key}")

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="PiperDaggerRecorder")
        self._thread.start()

    def begin(self, task: str) -> None:
        if not self.enabled:
            return
        with self._sample_lock, self._state_lock:
            if self.dataset.has_pending_frames():
                self.dataset.clear_episode_buffer()
            self._task = task
            self._frames_in_window = 0
            self._last_error = None
            self._active.set()

    def end(self, *, save: bool) -> int:
        if not self.enabled:
            return 0
        self.pause_capture()
        return self.finish_window(save=save)

    def pause_capture(self) -> None:
        """Stop sampling and wait until a possibly active sample has completed."""

        if not self.enabled:
            return
        self._active.clear()
        with self._sample_lock:
            pass

    def finish_window(self, *, save: bool) -> int:
        """Persist or discard an already-paused correction window."""

        if not self.enabled:
            return 0
        with self._sample_lock:
            with self._state_lock:
                frames = self._frames_in_window
            if self.dataset.has_pending_frames():
                if save and frames > 0 and self.last_error is None:
                    # ProcessPoolExecutor forks the live process when multiple
                    # cameras are encoded in parallel.  Forking a process that
                    # owns Piper CAN reader threads can invalidate the SDK
                    # communication context.  Sequential camera encoding still
                    # delegates the heavy work to ffmpeg without forking this
                    # hardware process.
                    try:
                        self.dataset.save_episode(parallel_encoding=False)
                    except BaseException as exc:
                        self._active.clear()
                        with self._state_lock:
                            self._last_error = exc
                            self._frames_in_window = 0
                        # Do not retry a failed save as a partial episode during
                        # shutdown. The writer retains a marker if commit began.
                        try:
                            self.dataset.clear_episode_buffer()
                        except Exception:
                            logger.exception("Failed to discard DAgger frames after a save error")
                        raise
                    with self._state_lock:
                        self._saved_episodes += 1
                else:
                    self.dataset.clear_episode_buffer()
            with self._state_lock:
                self._frames_in_window = 0
        return frames

    def _run(self) -> None:
        period = 1.0 / self.config.fps
        while not self._shutdown.is_set():
            if not self._active.wait(timeout=0.05):
                continue
            if self._shutdown.is_set():
                break
            # Pace from the actual capture start: never repay a stall by burst-sampling.
            deadline = time.monotonic() + period
            try:
                self._record_one()
            except Exception:
                logger.exception("DAgger correction recording stopped after an error")
            precise_sleep(max(0.0, deadline - time.monotonic()))

    def _record_one(self) -> None:
        with self._sample_lock:
            if not self._active.is_set():
                return
            try:
                self._capture_frame()
            except Exception as exc:
                # Publish the failure before releasing the capture barrier, so
                # save/discard/begin cannot race a delayed error from this window.
                with self._state_lock:
                    self._last_error = exc
                self._active.clear()
                raise

    def _capture_frame(self) -> None:
        """Capture one frame while the caller holds the sample lock."""
        with self._observation_lock:
            observation = self.robot.get_observation()
            # Own camera buffers before another SDK read or the async image writer can reuse them.
            observation = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
        expert_action = self.expert_action_reader()
        observation_frame = build_dataset_frame(self.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.features, expert_action, prefix=ACTION)
        with self._state_lock:
            task = self._task
        self.dataset.add_frame(
            {
                **observation_frame,
                **action_frame,
                "task": task,
                "intervention": np.array([True], dtype=bool),
            }
        )
        with self._state_lock:
            self._frames_in_window += 1

    def close(self, *, save_partial: bool = False) -> None:
        if not self.enabled:
            return
        try:
            self.end(save=save_partial)
        finally:
            self._shutdown.set()
            self._active.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=3.0)
            self._thread = None
            self._active.clear()
            self.dataset.finalize()
        if self.config.push_to_hub and self.last_error is None:
            self.dataset.push_to_hub(private=self.config.private)
