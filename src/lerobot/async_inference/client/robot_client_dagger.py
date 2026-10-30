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

"""Canonical Piper dynamic-takeover DAgger client.

It extends the consolidated ``robot_client_deployment`` core and the
Piper-specific two-CAN handover in ``piper_hil_dagger``.  No historical RTC,
interpolated, filtered, or resettable client module is imported.

The keyboard workflow is intentionally sequential and task-reset oriented:

``READY --right--> AUTONOMOUS --right--> PAUSED --right--> CORRECTING``.

Pressing right once more saves the correction as one episode and resets the
robot to its rollout reset pose.  Pressing left in PAUSED or CORRECTING aborts
the current rollout, discards only unsaved correction frames, and performs the
same reset.  Both paths then wait in READY while the operator restores the
physical environment; they never resume policy execution from a world state
that a bad human correction may have changed.  The exception is the final
configured episode: it exits immediately after saving, and the Piper robot's
existing ``safe_disconnect_follower`` shutdown path performs the final safe
return and disable sequence exactly once.

Pausing advances the encoded RTC generation and clears all local schedule,
interpolator, latency, and pending-request state.  Therefore an inference already in
flight may return, but it cannot mutate or command the post-intervention run.
Starting the next rollout sends the first observation with an empty action
prefix, which makes the deployment server reset its episode state while keeping
the model loaded.

Example (corrections-only dataset):

.. code-block:: shell

   python -m lerobot.async_inference.client.robot_client_dagger \
       --robot.type=piper_follower --server_address=10.0.0.2:8080 \
       --policy_type=smolvla --pretrained_name_or_path=user/model \
       --actions_per_chunk=50 --fps=30 --interpolation_multiplier=2 \
       --task="pick the cube" \
       --dagger_dataset_repo_id=user/rollout_piper_dagger \
       --dagger_dataset_root=/data/rollout_piper_dagger \
       --dagger_num_episodes=20

The default keyboard controls are right arrow (advance), left arrow (abort and
reset), and Esc (stop).  Equivalent prompt controls are ``/start``, ``/pause``,
``/correct`` (start, or save and reset), ``/reset``, ``/status``, and ``/stop``.
"""

import contextlib
import logging
import sys
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lerobot.light import (
    StatusCue,
    StatusIndicator,
    StatusIndicatorConfig,
    make_status_indicator,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import log_say

from ..shared.deployment_diagnostics import configure_deployment_logging
from ..shared.helpers import get_logger, visualize_action_queue_size
from ..shared.rtc_remote import RemoteRTCActionChunk, RemoteRTCObservation
from .piper_hil_dagger import (
    DaggerCorrectionRecorder,
    DaggerDatasetConfig,
    PiperDaggerController,
    PiperDaggerHardwareConfig,
    PiperDaggerRobotProxy,
    PiperHandoverAlignmentError,
)
from .robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
    RolloutState,
    _run_interactive_prompt,
)

_THREAD_JOIN_TIMEOUT_SECONDS = 3.0
_OPERATOR_SHUTDOWN_EVENT = "__shutdown__"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD_BRIGHT_RED = "\033[1;91m"
_ANSI_BOLD_BRIGHT_GREEN = "\033[1;92m"
_ANSI_BOLD_BRIGHT_YELLOW = "\033[1;93m"
_ANSI_BOLD_BRIGHT_MAGENTA = "\033[1;95m"
_ANSI_BOLD_BRIGHT_CYAN = "\033[1;96m"


class DAggerPhase(StrEnum):
    """Observable remote-DAgger phase."""

    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    CORRECTING = "correcting"
    FAULT = "fault"


def _terminal_emphasis(text: str, color: str, *, enabled: bool | None = None) -> str:
    """Apply terminal-only bold color without polluting non-interactive output."""

    if enabled is None:
        enabled = sys.stderr.isatty()
    return f"{color}{text}{_ANSI_RESET}" if enabled else text


@dataclass
class RobotClientDaggerConfig(RobotClientDeploymentConfig):
    """Remote RTC, Piper handover, operator-input, and dataset settings."""

    dagger_master_can_name: str = field(
        default="can_master",
        metadata={"help": "SocketCAN interface connected only to the Piper master arm"},
    )
    dagger_follower_can_name: str = field(
        default="can_follower",
        metadata={"help": "SocketCAN interface connected only to the Piper follower arm"},
    )
    dagger_bridge_receive_timeout_s: float = field(
        default=0.01,
        metadata={"help": "Maximum CAN receive/send wait used by the takeover bridge"},
    )
    dagger_feedback_timeout_s: float = field(
        default=1.0,
        metadata={"help": "Timeout waiting for fresh follower feedback before policy control"},
    )
    dagger_minimum_firmware: str = field(
        default="1.8.9",
        metadata={"help": "Minimum Piper firmware accepted for dynamic mode switching"},
    )
    dagger_require_safe_firmware: bool = field(
        default=True,
        metadata={"help": "Fail startup when arm firmware cannot be verified"},
    )
    dagger_require_matching_firmware: bool = field(
        default=True,
        metadata={"help": "Require master and follower to report the same firmware version"},
    )
    dagger_require_master_enabled: bool = field(
        default=True,
        metadata={"help": "Verify that the master can execute mirrored commands in follower mode"},
    )
    dagger_master_enable_timeout_s: float = field(default=5.0)
    dagger_follower_first: bool = field(
        default=True,
        metadata={"help": "Send each shared MIT command to the follower before the master"},
    )
    dagger_handover_alignment_timeout_s: float = field(
        default=2.0,
        metadata={"help": "Timeout for aligning the master to actual follower feedback before takeover"},
    )
    dagger_handover_joint_tolerance: float = field(
        default=0.035,
        metadata={"help": "Maximum master/follower joint error allowed at takeover, in radians"},
    )
    dagger_handover_gripper_tolerance: float = field(
        default=0.001,
        metadata={"help": "Maximum master/follower gripper error allowed at takeover, in metres"},
    )
    dagger_handover_gripper_settle_s: float = field(
        default=0.05,
        metadata={"help": "Required duration of absolute alignment before opening the correction bridge"},
    )

    dagger_input_device: str = field(
        default="keyboard",
        metadata={"help": "Operator input: prompt, keyboard, or pedal"},
    )
    dagger_advance_key: str = field(
        default="right",
        metadata={"help": "Advance READY -> AUTONOMOUS -> PAUSED -> CORRECTING -> save/reset"},
    )
    dagger_reset_key: str = field(
        default="left",
        metadata={"help": "Abort PAUSED/CORRECTING, discard unsaved data, and reset the task"},
    )
    dagger_stop_key: str = field(default="esc")
    dagger_pedal_device_path: str = field(default="/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd")
    dagger_pedal_advance: str = field(default="KEY_A")
    dagger_pedal_reset: str = field(default="KEY_B")

    status_indicator: StatusIndicatorConfig = field(default_factory=StatusIndicatorConfig)

    dagger_dataset_repo_id: str | None = field(
        default=None,
        metadata={
            "help": (
                "LeRobot repo id for correction-only DAgger recording. "
                "None disables recording for takeover dry-runs."
            )
        },
    )
    dagger_dataset_root: str | Path | None = field(default=None)
    dagger_dataset_fps: int = field(
        default=0,
        metadata={"help": "Correction recording FPS; 0 inherits the policy FPS"},
    )
    dagger_dataset_video: bool = field(default=True)
    dagger_dataset_resume: bool = field(default=False)
    dagger_dataset_push_to_hub: bool = field(default=False)
    dagger_dataset_private: bool | None = field(default=None)
    dagger_dataset_image_writer_processes: int = field(default=0)
    dagger_dataset_image_writer_threads_per_camera: int = field(default=4)
    dagger_num_episodes: int | None = field(
        default=0,
        metadata={
            "help": (
                "Number of correction episodes to save during this client run. "
                "0 or None collects without a limit; a positive target exits "
                "automatically after the final episode is saved."
            )
        },
    )
    dagger_save_partial_correction_on_stop: bool = field(
        default=False,
        metadata={"help": "Save an interrupted correction window during terminal shutdown"},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dagger_input_device not in ("prompt", "keyboard", "pedal"):
            raise ValueError("dagger_input_device must be 'prompt', 'keyboard', or 'pedal'")
        if self.dagger_dataset_fps < 0:
            raise ValueError("dagger_dataset_fps must be non-negative")
        if self.dagger_num_episodes is not None and self.dagger_num_episodes < 0:
            raise ValueError("dagger_num_episodes must be non-negative or None")
        PiperDaggerHardwareConfig(
            master_can_name=self.dagger_master_can_name,
            follower_can_name=self.dagger_follower_can_name,
            bridge_receive_timeout_s=self.dagger_bridge_receive_timeout_s,
            feedback_timeout_s=self.dagger_feedback_timeout_s,
            minimum_firmware=self.dagger_minimum_firmware,
            require_safe_firmware=self.dagger_require_safe_firmware,
            require_matching_firmware=self.dagger_require_matching_firmware,
            require_master_enabled=self.dagger_require_master_enabled,
            master_enable_timeout_s=self.dagger_master_enable_timeout_s,
            follower_first=self.dagger_follower_first,
            debug_command_timing=self.debug_rtc_diagnostics,
            handover_alignment_timeout_s=self.dagger_handover_alignment_timeout_s,
            handover_joint_tolerance=self.dagger_handover_joint_tolerance,
            handover_gripper_tolerance=self.dagger_handover_gripper_tolerance,
            handover_gripper_settle_s=self.dagger_handover_gripper_settle_s,
        )
        DaggerDatasetConfig(
            repo_id=self.dagger_dataset_repo_id,
            root=self.dagger_dataset_root,
            fps=self.dagger_dataset_fps or self.fps,
            use_videos=self.dagger_dataset_video,
            resume=self.dagger_dataset_resume,
            push_to_hub=self.dagger_dataset_push_to_hub,
            private=self.dagger_dataset_private,
            image_writer_processes=self.dagger_dataset_image_writer_processes,
            image_writer_threads_per_camera=self.dagger_dataset_image_writer_threads_per_camera,
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "dagger_master_can_name": self.dagger_master_can_name,
            "dagger_follower_can_name": self.dagger_follower_can_name,
            "dagger_bridge_receive_timeout_s": self.dagger_bridge_receive_timeout_s,
            "dagger_feedback_timeout_s": self.dagger_feedback_timeout_s,
            "dagger_minimum_firmware": self.dagger_minimum_firmware,
            "dagger_require_safe_firmware": self.dagger_require_safe_firmware,
            "dagger_require_matching_firmware": self.dagger_require_matching_firmware,
            "dagger_require_master_enabled": self.dagger_require_master_enabled,
            "dagger_master_enable_timeout_s": self.dagger_master_enable_timeout_s,
            "dagger_follower_first": self.dagger_follower_first,
            "dagger_handover_alignment_timeout_s": self.dagger_handover_alignment_timeout_s,
            "dagger_handover_joint_tolerance": self.dagger_handover_joint_tolerance,
            "dagger_handover_gripper_tolerance": self.dagger_handover_gripper_tolerance,
            "dagger_handover_gripper_settle_s": self.dagger_handover_gripper_settle_s,
            "dagger_input_device": self.dagger_input_device,
            "dagger_advance_key": self.dagger_advance_key,
            "dagger_reset_key": self.dagger_reset_key,
            "dagger_stop_key": self.dagger_stop_key,
            "dagger_pedal_device_path": self.dagger_pedal_device_path,
            "dagger_pedal_advance": self.dagger_pedal_advance,
            "dagger_pedal_reset": self.dagger_pedal_reset,
            "status_indicator": asdict(self.status_indicator),
            "dagger_dataset_repo_id": self.dagger_dataset_repo_id,
            "dagger_dataset_root": self.dagger_dataset_root,
            "dagger_dataset_fps": self.dagger_dataset_fps,
            "dagger_dataset_video": self.dagger_dataset_video,
            "dagger_dataset_resume": self.dagger_dataset_resume,
            "dagger_dataset_push_to_hub": self.dagger_dataset_push_to_hub,
            "dagger_dataset_private": self.dagger_dataset_private,
            "dagger_dataset_image_writer_processes": self.dagger_dataset_image_writer_processes,
            "dagger_dataset_image_writer_threads_per_camera": (
                self.dagger_dataset_image_writer_threads_per_camera
            ),
            "dagger_num_episodes": self.dagger_num_episodes,
            "dagger_save_partial_correction_on_stop": self.dagger_save_partial_correction_on_stop,
        }


@dataclass(frozen=True)
class _OperatorEvent:
    name: str
    generation: int
    rollout: RolloutState
    phase: DAggerPhase


class RobotClientDagger(RobotClientDeployment):
    """Piper dual-arm DAgger client with deployment RTC generation fences."""

    prefix = "robot_client_dagger"
    logger = get_logger(prefix)

    def __init__(
        self,
        config: RobotClientDaggerConfig,
        *,
        dagger_controller: Any | None = None,
        correction_recorder: Any | None = None,
        status_indicator: StatusIndicator | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config

        # The parent uses a plain Lock.  This subclass adds a phase check around
        # the parent's already-locked dispatch method, so re-entrancy is needed
        # without changing the parent implementation.
        self._robot_command_lock = threading.RLock()
        self._dagger_transition_lock = threading.RLock()
        self._dagger_phase = DAggerPhase.PAUSED
        self._dagger_fault: str | None = None
        self._operator_queue: Queue[_OperatorEvent | str] = Queue(maxsize=8)
        self._operator_busy = threading.Event()
        self._operator_worker_shutdown = threading.Event()
        self.operator_stop_event = threading.Event()
        self._save_in_progress = threading.Event()
        self._operator_worker: threading.Thread | None = None
        self._input_listener: Any | None = None
        self._status_indicator = (
            status_indicator
            if status_indicator is not None
            else make_status_indicator(config.status_indicator, log=self.logger)
        )

        self._follower_robot = self.robot
        try:
            hardware_config = PiperDaggerHardwareConfig(
                master_can_name=config.dagger_master_can_name,
                follower_can_name=config.dagger_follower_can_name,
                bridge_receive_timeout_s=config.dagger_bridge_receive_timeout_s,
                feedback_timeout_s=config.dagger_feedback_timeout_s,
                minimum_firmware=config.dagger_minimum_firmware,
                require_safe_firmware=config.dagger_require_safe_firmware,
                require_matching_firmware=config.dagger_require_matching_firmware,
                require_master_enabled=config.dagger_require_master_enabled,
                master_enable_timeout_s=config.dagger_master_enable_timeout_s,
                follower_first=config.dagger_follower_first,
                debug_command_timing=config.debug_rtc_diagnostics,
                handover_alignment_timeout_s=config.dagger_handover_alignment_timeout_s,
                handover_joint_tolerance=config.dagger_handover_joint_tolerance,
                handover_gripper_tolerance=config.dagger_handover_gripper_tolerance,
                handover_gripper_settle_s=config.dagger_handover_gripper_settle_s,
            )
            self.dagger_controller = dagger_controller or PiperDaggerController(
                self._follower_robot,
                hardware_config,
            )
            self.robot = PiperDaggerRobotProxy(self._follower_robot, self.dagger_controller)

            dataset_config = DaggerDatasetConfig(
                repo_id=config.dagger_dataset_repo_id,
                root=config.dagger_dataset_root,
                fps=config.dagger_dataset_fps or config.fps,
                use_videos=config.dagger_dataset_video,
                resume=config.dagger_dataset_resume,
                push_to_hub=config.dagger_dataset_push_to_hub,
                private=config.dagger_dataset_private,
                image_writer_processes=config.dagger_dataset_image_writer_processes,
                image_writer_threads_per_camera=config.dagger_dataset_image_writer_threads_per_camera,
            )
            self.correction_recorder = correction_recorder or DaggerCorrectionRecorder(
                self._follower_robot,
                self.dagger_controller.read_expert_action,
                dataset_config,
                observation_lock=self._observation_capture_lock,
            )
        except Exception:
            self.channel.close()
            controller = getattr(self, "dagger_controller", None)
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    self.logger.exception("Piper controller cleanup failed after initialization error")
            try:
                self._follower_robot.disconnect()
            except Exception:
                self.logger.exception("Follower cleanup failed after DAgger initialization error")
            raise

        if not self.correction_recorder.enabled:
            self.logger.warning(
                "DAgger takeover is enabled but dataset recording is disabled; set "
                "--dagger_dataset_repo_id to collect corrections"
            )

    @property
    def dagger_phase(self) -> DAggerPhase:
        with self._lifecycle_lock:
            return self._dagger_phase

    @property
    def dagger_fault(self) -> str | None:
        with self._lifecycle_lock:
            return self._dagger_fault

    @property
    def dagger_episode_target(self) -> int | None:
        """Return the per-process target, or ``None`` for unlimited collection."""

        target = self.config.dagger_num_episodes
        return None if target in (None, 0) else target

    @property
    def dagger_collection_complete(self) -> bool:
        target = self.dagger_episode_target
        return target is not None and self.correction_recorder.saved_episodes >= target

    @property
    def dagger_save_in_progress(self) -> bool:
        return self._save_in_progress.is_set()

    def _start_status_indicator(self) -> None:
        try:
            self._status_indicator.start()
        except Exception:
            if self.config.status_indicator.required:
                raise
            self.logger.exception("Could not initialize the status indicator")

    def _show_status(self, cue: StatusCue) -> None:
        try:
            self._status_indicator.show(cue)
        except Exception:
            self.logger.exception("Could not show status cue %s", cue.value)

    def _close_status_indicator(self) -> None:
        try:
            self._status_indicator.close()
        except Exception:
            self.logger.exception("Could not close the status indicator")

    def _announce_environment_reset(self, *, discarded_correction: bool) -> None:
        prefix = "当前未保存 correction 已丢弃；" if discarded_correction else ""
        message = f"DAgger 已复位：{prefix}请手动恢复实验环境，完成后按右方向键开始下一轮自主推理。"
        color = _ANSI_BOLD_BRIGHT_MAGENTA if discarded_correction else _ANSI_BOLD_BRIGHT_YELLOW
        self.logger.warning(
            _terminal_emphasis(
                f"********** [DAgger] {message} **********",
                color,
            )
        )

    def _episode_progress(self) -> str:
        target = self.dagger_episode_target
        target_text = "∞" if target is None else str(target)
        return f"{self.correction_recorder.saved_episodes}/{target_text}"

    def _request_exit_if_target_reached(self) -> bool:
        """Wake the main thread for orderly shutdown once the target is reached."""

        if not self.dagger_collection_complete:
            return False
        with self._lifecycle_changed:
            self._clear_episode_state_locked()
            self._lifecycle_changed.notify_all()
        if not self.operator_stop_event.is_set():
            message = f"DAgger 已完成 episode {self._episode_progress()}，程序正在安全退出。"
            self.logger.info(message)
            self._show_status(StatusCue.EXIT)
            self.operator_stop_event.set()
        return True

    def _set_fault(self, operation: str, exc: BaseException) -> None:
        with self._lifecycle_changed:
            self._dagger_phase = DAggerPhase.FAULT
            self._dagger_fault = f"{operation}: {exc}"
            self._lifecycle_changed.notify_all()
        self._show_status(StatusCue.FAULT)
        self.logger.exception("DAgger entered FAULT during %s", operation, exc_info=exc)

    def _seed_interpolator_locked(self, pose: dict[str, float]) -> None:
        """Use corrected follower pose as the first interpolation origin."""

        origin = torch.tensor(
            [float(pose[key]) for key in self.robot.action_features],
            dtype=torch.float32,
        )
        self.action_interpolator.reset()
        self.action_interpolator.add(origin)
        # Exhaust the synthetic hold endpoint.  The first real policy endpoint
        # will now be interpolated from this pose rather than sent as a jump.
        self.action_interpolator.get()
        self._has_interpolation_origin = True

    # ------------------------------------------------------------------
    # Rollout + DAgger state machine
    # ------------------------------------------------------------------

    def start(self) -> bool:
        try:
            self._start_status_indicator()
        except Exception:
            self.logger.exception("Required status indicator is unavailable")
            self.stop()
            return False
        started = super().start()
        if started:
            self.correction_recorder.start()
            self._show_status(StatusCue.RESET_REQUIRED)
            self._request_exit_if_target_reached()
        return started

    def start_rollout(self, task: str | None = None) -> bool:
        # A start requested during reset must be rejected, never wait behind it
        # and then turn into a start once READY is published.
        if not self._dagger_transition_lock.acquire(blocking=False):
            return False
        try:
            with self._lifecycle_lock:
                if (
                    self._rollout_state is not RolloutState.READY
                    or not self.running
                    or self.operator_stop_event.is_set()
                    or self._dagger_phase is DAggerPhase.FAULT
                ):
                    return False
            if self._request_exit_if_target_reached():
                return False
            try:
                with self._observation_capture_lock, self._robot_command_lock:
                    if self.operator_stop_event.is_set() or not self.running:
                        return False
                    pose = self.dagger_controller.prepare_autonomy()
                if self.operator_stop_event.is_set() or not self.running:
                    return False
                if not super().start_rollout(task):
                    return False
                with self._lifecycle_changed:
                    if self.operator_stop_event.is_set() or not self.running:
                        return False
                    self._seed_interpolator_locked(pose)
                    self._dagger_phase = DAggerPhase.AUTONOMOUS
                    self._dagger_fault = None
                    self._lifecycle_changed.notify_all()
                self._show_status(StatusCue.AUTONOMOUS)
                return True
            except Exception as exc:
                self._set_fault("start rollout", exc)
                return False
        finally:
            self._dagger_transition_lock.release()

    def pause_autonomy(self) -> bool:
        """Invalidate RTC first, then wait out the final computer command."""

        with self._dagger_transition_lock:
            with self._lifecycle_changed:
                if (
                    self._rollout_state is not RolloutState.RUNNING
                    or self._dagger_phase is not DAggerPhase.AUTONOMOUS
                ):
                    return False
                self._dagger_phase = DAggerPhase.PAUSED
                self._advance_generation_locked()
                self._clear_episode_state_locked()
                generation = self._rollout_generation
                self._lifecycle_changed.notify_all()
            try:
                with self._robot_command_lock:
                    self.dagger_controller.pause_computer_output()
            except Exception as exc:
                self._set_fault("pause autonomy", exc)
                return False
            self._show_status(StatusCue.PAUSED)
            self.logger.info("Autonomy paused; RTC generation advanced to %d", generation)
            return True

    def start_correction(self) -> bool:
        with self._dagger_transition_lock:
            with self._lifecycle_lock:
                if (
                    self._rollout_state is not RolloutState.RUNNING
                    or self._dagger_phase is not DAggerPhase.PAUSED
                    or self.operator_stop_event.is_set()
                ):
                    return False
                task = self._task
            if self._request_exit_if_target_reached():
                return False
            try:
                with self._robot_command_lock:
                    if self.operator_stop_event.is_set() or not self.running:
                        return False
                    self.dagger_controller.enter_correction()
                # Esc can arrive during physical alignment or while begin() waits
                # for a previous sample. Do not publish a new correction afterward.
                cancelled = self.operator_stop_event.is_set() or not self.running
                if not cancelled:
                    self.correction_recorder.begin(task)
                    with self._lifecycle_changed:
                        cancelled = self.operator_stop_event.is_set() or not self.running
                        if not cancelled:
                            self._dagger_phase = DAggerPhase.CORRECTING
                            self._lifecycle_changed.notify_all()
                if cancelled:
                    self.correction_recorder.pause_capture()
                    with self._observation_capture_lock, self._robot_command_lock:
                        self.dagger_controller.abort_to_paused()
                    self.correction_recorder.finish_window(save=False)
                    return False
                self._show_status(StatusCue.CORRECTING)
                self.logger.info("Human correction started (task=%r)", task)
                return True
            except PiperHandoverAlignmentError as exc:
                # enter_correction only exposes this recoverable error after
                # restoring follower mode and verifying physical realignment.
                # Capture has not started, and policy remains paused.
                self._show_status(StatusCue.PAUSED)
                self.logger.warning(
                    _terminal_emphasis(
                        "接管未开启：切换后主从绝对开度不一致，自动重试仍未通过。"
                        "已将主臂切回从臂模式并自动恢复对齐；未采集数据，仍处于暂停。"
                        "无需手调夹爪；可按左键复位，或在排查后按右键重试。",
                        _ANSI_BOLD_BRIGHT_YELLOW,
                    )
                )
                self.logger.warning("Piper handover rejected after automatic realignment: %s", exc)
                return False
            except Exception as exc:
                try:
                    self.correction_recorder.pause_capture()
                    self.correction_recorder.finish_window(save=False)
                except Exception:
                    self.logger.exception("Could not discard a failed correction window")
                try:
                    with self._observation_capture_lock, self._robot_command_lock:
                        self.dagger_controller.abort_to_paused()
                except Exception:
                    self.logger.exception("Could not roll back a failed correction start")
                self._set_fault("start correction", exc)
                return False

    def _finish_correction(self, *, save: bool) -> bool:
        saved_before = 0
        with self._dagger_transition_lock:
            with self._lifecycle_changed:
                if self._dagger_phase is not DAggerPhase.CORRECTING:
                    return False
                # Stop new recorder samples and policy work before touching CAN.
                self._dagger_phase = DAggerPhase.PAUSED
                self._lifecycle_changed.notify_all()
            self.correction_recorder.pause_capture()
            self._show_status(StatusCue.PAUSED)
            try:
                with self._observation_capture_lock, self._robot_command_lock:
                    self.dagger_controller.leave_correction()
            except Exception as exc:
                self.correction_recorder.finish_window(save=False)
                self._set_fault("stop correction", exc)
                return False

            saved_before = self.correction_recorder.saved_episodes
            if save:
                self._save_in_progress.set()
                dropped_events = 0
                while True:
                    try:
                        self._operator_queue.get_nowait()
                    except Empty:
                        break
                    dropped_events += 1
                if dropped_events:
                    self.logger.info(
                        "Discarded %d repeated operator event(s) before episode saving",
                        dropped_events,
                    )
                self.logger.info(
                    "Correction capture stopped; saving episode in the background transition stage. "
                    "Esc remains available for safe hardware shutdown."
                )

        # Video encoding can take tens of seconds.  Do not hold the DAgger
        # transition lock here: the main thread must remain able to process an
        # out-of-band Esc and safely disconnect the hardware.
        try:
            frames = self.correction_recorder.finish_window(save=save)
        except Exception as exc:
            operation = "save correction" if save else "discard correction"
            if self.operator_stop_event.is_set():
                self.logger.exception("DAgger %s failed during requested shutdown", operation, exc_info=exc)
            else:
                self._set_fault(operation, exc)
            return False
        finally:
            self._save_in_progress.clear()

        if not save:
            self.logger.info(
                "Discarded current correction (%d frames); episode remains %s",
                frames,
                self._episode_progress(),
            )
            return True

        recording_error = self.correction_recorder.last_error
        if recording_error is not None:
            if self.operator_stop_event.is_set():
                self.logger.error(
                    "DAgger recording failed during requested shutdown: %s",
                    recording_error,
                    exc_info=recording_error,
                )
            else:
                self._set_fault("record correction", recording_error)
            return False

        saved_after = self.correction_recorder.saved_episodes
        if saved_after == saved_before:
            self.logger.warning(
                "Correction ended with %d frames but no episode was saved; episode remains %s",
                frames,
                self._episode_progress(),
            )
            return True

        target_reached = self.dagger_collection_complete
        self.logger.info(
            "Human correction stopped; saved %d frames as episode %s%s",
            frames,
            self._episode_progress(),
            "; target reached, requesting automatic safe shutdown" if target_reached else "",
        )
        return True

    def stop_correction(self) -> bool:
        """Save one correction, then either reset for another rollout or exit."""

        if not self._finish_correction(save=True):
            return False
        if self.operator_stop_event.is_set():
            return True
        if self._request_exit_if_target_reached():
            return True
        return self.reset_rollout()

    def reset_rollout(self) -> bool:
        if not self._dagger_transition_lock.acquire(blocking=False):
            return False
        try:
            with self._lifecycle_changed:
                if self._rollout_state is RolloutState.STOPPED or not self.running:
                    return False
                discarded_correction = self._dagger_phase is DAggerPhase.CORRECTING
            self._show_status(StatusCue.RESET_REQUIRED)
            reset_ok = super().reset_rollout()
            if reset_ok:
                with self._lifecycle_changed:
                    self._dagger_fault = None
                    self._dagger_phase = DAggerPhase.PAUSED
                    self._lifecycle_changed.notify_all()
                self._show_status(StatusCue.RESET_REQUIRED)
                self._announce_environment_reset(discarded_correction=discarded_correction)
            else:
                self._set_fault(
                    "reset robot pose",
                    RuntimeError(self.reset_error or "robot did not reach the configured reset pose"),
                )
            return reset_ok
        finally:
            self._dagger_transition_lock.release()

    def _prepare_reset(self) -> None:
        with self._lifecycle_changed:
            self._dagger_phase = DAggerPhase.PAUSED
            self._lifecycle_changed.notify_all()
        self.correction_recorder.pause_capture()
        try:
            with self._observation_capture_lock, self._robot_command_lock:
                self.dagger_controller.abort_to_paused()
        finally:
            self.correction_recorder.finish_window(save=False)

    def _reset_cancelled(self) -> bool:
        return super()._reset_cancelled() or self.operator_stop_event.is_set()

    def _read_reset_poses_unlocked(self, timeout_s: float) -> dict[str, dict[str, float]]:
        poses = self.dagger_controller.read_reset_feedback(timeout_s)
        if set(poses) != {"master", "follower"}:
            raise RuntimeError("reset requires fresh feedback from both master and follower")
        return poses

    def _handle_recording_failure(self, error: Exception) -> None:
        # Do not wait behind a reset/save transition. It already checks recorder
        # errors, and an old error must not poison a later correction window.
        if not self._dagger_transition_lock.acquire(blocking=False):
            return
        try:
            with self._lifecycle_lock:
                if (
                    self._dagger_phase is not DAggerPhase.CORRECTING
                    or self.correction_recorder.last_error is not error
                ):
                    return
            self._set_fault("record correction", error)
            try:
                self.correction_recorder.pause_capture()
                with self._observation_capture_lock, self._robot_command_lock:
                    self.dagger_controller.abort_to_paused()
            except Exception:
                self.logger.exception("Could not complete handover after recorder failure")
            finally:
                try:
                    self.correction_recorder.finish_window(save=False)
                except Exception:
                    self.logger.exception("Could not discard correction after recorder failure")
        finally:
            self._dagger_transition_lock.release()

    # ------------------------------------------------------------------
    # Defense-in-depth phase gates around the deployment RTC client
    # ------------------------------------------------------------------

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        if self.operator_stop_event.is_set() or not self.running:
            return None
        with self._lifecycle_lock:
            phase = self._dagger_phase
        if phase is DAggerPhase.CORRECTING:
            recording_error = self.correction_recorder.last_error
            if recording_error is not None:
                self._handle_recording_failure(recording_error)
                return None
            check_bridge = getattr(self.dagger_controller, "assert_correction_healthy", None)
            if callable(check_bridge):
                try:
                    check_bridge()
                except Exception as exc:
                    # The bridge itself fails closed.  Keep the master in
                    # leader mode so it never receives policy commands; /reset
                    # performs the explicit follower-mode recovery sequence.
                    self._set_fault("CAN correction bridge", exc)
                    try:
                        self.correction_recorder.pause_capture()
                        self.correction_recorder.finish_window(save=False)
                    except Exception:
                        self.logger.exception("Could not discard correction after bridge failure")
            return None
        try:
            with self._defer_rtc_logs(), self._robot_command_lock, self._lifecycle_lock:
                if (
                    self._dagger_phase is not DAggerPhase.AUTONOMOUS
                    or self.operator_stop_event.is_set()
                    or not self.running
                ):
                    return None
                return super().control_loop_action(verbose=verbose)
        except Exception as exc:
            try:
                with self._robot_command_lock:
                    self.dagger_controller.pause_computer_output()
            except Exception:
                self.logger.exception("Could not pause Piper output after command failure")
            self._set_fault("dispatch dual-arm command", exc)
            return None

    def _observation_collection_active_locked(self) -> bool:
        return (
            super()._observation_collection_active_locked()
            and self._dagger_phase is DAggerPhase.AUTONOMOUS
            and not self.operator_stop_event.is_set()
        )

    def _request_observation_if_ready(self, task: str | None = None) -> bool:
        with self._defer_rtc_logs(), self._lifecycle_lock:
            if (
                self._dagger_phase is not DAggerPhase.AUTONOMOUS
                or self.operator_stop_event.is_set()
                or not self.running
            ):
                return False
            return super()._request_observation_if_ready(task)

    def send_observation(self, envelope: RemoteRTCObservation) -> bool:
        with self._lifecycle_lock:
            if (
                self._dagger_phase is not DAggerPhase.AUTONOMOUS
                or self.operator_stop_event.is_set()
                or not self.running
            ):
                return False
        # The upload RPC can be slow.  Do not retain the lifecycle lock across
        # it: a concurrent pause advances the generation, and the parent then
        # rejects this upload/result without delaying physical takeover.
        return super().send_observation(envelope)

    def _handle_action_chunk(self, chunk: RemoteRTCActionChunk) -> bool:
        with self._lifecycle_lock:
            if (
                self._dagger_phase is not DAggerPhase.AUTONOMOUS
                or self.operator_stop_event.is_set()
                or not self.running
            ):
                return False
        return super()._handle_action_chunk(chunk)

    # ------------------------------------------------------------------
    # Operator input
    # ------------------------------------------------------------------

    def _operator_input_label(self, event: str) -> str:
        if self.config.dagger_input_device == "keyboard":
            key = {
                "advance": self.config.dagger_advance_key,
                "reset": self.config.dagger_reset_key,
                "stop": self.config.dagger_stop_key,
            }.get(event, event)
            return {
                "right": "右方向键",
                "left": "左方向键",
                "esc": "Esc 键",
            }.get(key, f"{key!r} 键")
        if self.config.dagger_input_device == "pedal":
            code = {
                "advance": self.config.dagger_pedal_advance,
                "reset": self.config.dagger_pedal_reset,
            }.get(event, event)
            return f"脚踏开关 {code}"
        return f"/{event}"

    def _log_operator_event(
        self,
        event: str,
        phase: DAggerPhase,
        rollout: RolloutState,
    ) -> None:
        input_label = self._operator_input_label(event)
        if event == "stop":
            message = f"检测到{input_label} -> 安全退出程序"
            color = "bright_red"
        elif event == "reset":
            if phase is DAggerPhase.CORRECTING:
                action = "丢弃当前 correction 数据并复位"
                color = "bright_magenta"
            elif phase in (DAggerPhase.PAUSED, DAggerPhase.FAULT):
                action = "丢弃当前未保存窗口并复位"
                color = "bright_magenta"
            else:
                action = f"当前 {rollout.value}/{phase.value} 阶段不接受复位"
                color = "bright_yellow"
            message = f"检测到{input_label} -> {action}"
        elif event == "advance":
            if rollout is RolloutState.READY and phase is DAggerPhase.PAUSED:
                action, color = "开始自主推理", "bright_green"
            elif rollout is RolloutState.RUNNING and phase is DAggerPhase.AUTONOMOUS:
                action, color = "暂停自主推理并对齐主从臂", "bright_yellow"
            elif rollout is RolloutState.RUNNING and phase is DAggerPhase.PAUSED:
                action, color = "进入人工接管并开始 correction", "bright_cyan"
            elif rollout is RolloutState.RUNNING and phase is DAggerPhase.CORRECTING:
                action, color = "结束 correction、保存 episode 并复位", "bright_blue"
            elif phase is DAggerPhase.FAULT:
                action, color = "FAULT 状态不可前进，请按左方向键复位", "bright_red"
            else:
                action = f"当前 {rollout.value}/{phase.value} 阶段不接受前进"
                color = "bright_yellow"
            message = f"检测到{input_label} -> {action}"
        else:
            message = f"检测到未知操作员输入 {event!r}"
            color = "bright_yellow"

        log_say(f"********** [DAgger] {message} **********", play_sounds=False, color=color)

    def submit_operator_event(self, event: str) -> bool:
        # Esc is a safety command, not an ordinary queued transition.  It must
        # remain responsive even while video encoding blocks the operator worker
        # or the ordinary transition queue is full.
        if event == "stop":
            if not self.operator_stop_event.is_set():
                # Publish the safety signal before optional UI work.  A slow
                # terminal handler or status-light bus must never delay Esc.
                self.operator_stop_event.set()
                self._log_operator_event(event, self.dagger_phase, self.rollout_state)
                self._show_status(StatusCue.EXIT)
            return True

        with self._lifecycle_lock:
            busy = (
                self._save_in_progress.is_set()
                or self._reset_in_progress.is_set()
                or self._operator_busy.is_set()
            )
            if not busy:
                if not self.running or self.operator_stop_event.is_set():
                    return False
                queued = _OperatorEvent(
                    event, self._rollout_generation, self._rollout_state, self._dagger_phase
                )
                try:
                    self._operator_queue.put_nowait(queued)
                    return True
                except Full:
                    pass
        if busy:
            self.logger.warning("Ignoring operator event %s while a transition is in progress", event)
            log_say(
                "********** [DAgger] 复位／保存／状态切换尚未完成，本次按键已忽略；"
                "完成后请重新按键，Esc 仍可退出 **********",
                play_sounds=False,
                color="bright_yellow",
            )
            return False

        self.logger.warning("Dropping operator event %s: transition queue is full", event)
        return False

    def _handle_operator_event(self, event: str, *, expected: _OperatorEvent | None = None) -> bool:
        if event == "stop":
            return self.submit_operator_event(event)
        with self._lifecycle_lock:
            if (
                self._operator_busy.is_set()
                or self._reset_in_progress.is_set()
                or self._save_in_progress.is_set()
                or not self.running
            ):
                return False
            if expected is not None and (
                expected.generation != self._rollout_generation
                or expected.rollout is not self._rollout_state
                or expected.phase is not self._dagger_phase
            ):
                return False
            self._operator_busy.set()
        try:
            return self._apply_operator_event(event)
        finally:
            self._operator_busy.clear()

    def _apply_operator_event(self, event: str) -> bool:
        """Apply one logical key/pedal event to the phase machine."""

        phase = self.dagger_phase
        rollout = self.rollout_state
        self._log_operator_event(event, phase, rollout)

        if self.operator_stop_event.is_set():
            return False

        if event == "advance":
            if phase is DAggerPhase.FAULT:
                self.logger.warning("DAgger is in FAULT; press Left Arrow to reset")
                return False
            if rollout is RolloutState.READY and phase is DAggerPhase.PAUSED:
                return self.start_rollout()
            if rollout is RolloutState.RUNNING:
                if phase is DAggerPhase.AUTONOMOUS:
                    return self.pause_autonomy()
                if phase is DAggerPhase.PAUSED:
                    return self.start_correction()
                if phase is DAggerPhase.CORRECTING:
                    return self.stop_correction()
            return False

        if event == "reset":
            if phase is DAggerPhase.FAULT:
                return self.reset_rollout()
            if rollout is RolloutState.RUNNING and phase in (
                DAggerPhase.PAUSED,
                DAggerPhase.CORRECTING,
            ):
                return self.reset_rollout()
            return False
        return False

    def _operator_loop(self) -> None:
        while not self._operator_worker_shutdown.is_set() and self.running:
            try:
                event = self._operator_queue.get(timeout=0.1)
            except Empty:
                continue
            if event == _OPERATOR_SHUTDOWN_EVENT:
                return
            assert isinstance(event, _OperatorEvent)
            self._handle_operator_event(event.name, expected=event)

    def start_operator_input(self) -> None:
        if self.config.dagger_input_device == "prompt":
            return
        if self._operator_worker is not None:
            return
        self._operator_worker_shutdown.clear()
        self._operator_worker = threading.Thread(
            target=self._operator_loop,
            daemon=True,
            name="PiperDaggerOperator",
        )
        self._operator_worker.start()

        if self.config.dagger_input_device == "keyboard":
            from lerobot.utils.keyboard_input import create_key_listener

            def on_key(name: str) -> None:
                if name == self.config.dagger_advance_key:
                    self.submit_operator_event("advance")
                elif name == self.config.dagger_reset_key:
                    self.submit_operator_event("reset")
                elif name == self.config.dagger_stop_key:
                    self.submit_operator_event("stop")

            self._input_listener = create_key_listener(
                on_key,
                controls_help=(
                    f"advance={self.config.dagger_advance_key}, "
                    f"abort/reset={self.config.dagger_reset_key}, "
                    f"stop={self.config.dagger_stop_key}"
                ),
            )
        else:
            from lerobot.utils.pedal import start_pedal_listener

            def on_press(code: str) -> None:
                if code == self.config.dagger_pedal_advance:
                    self.submit_operator_event("advance")
                elif code == self.config.dagger_pedal_reset:
                    self.submit_operator_event("reset")

            self._input_listener = start_pedal_listener(
                on_press,
                device_path=self.config.dagger_pedal_device_path,
            )

    def stop_operator_input(self) -> None:
        listener = self._input_listener
        stop_listener = getattr(listener, "stop", None)
        if callable(stop_listener):
            stop_listener()
        self._input_listener = None
        self._operator_worker_shutdown.set()
        with contextlib.suppress(Full):
            self._operator_queue.put_nowait(_OPERATOR_SHUTDOWN_EVENT)
        worker = self._operator_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        self._operator_worker = None

    def stop(self) -> None:
        self.shutdown_event.set()
        self._show_status(StatusCue.EXIT)
        self.stop_operator_input()
        try:
            with self._dagger_transition_lock:
                with self._lifecycle_changed:
                    if self._rollout_state is RolloutState.STOPPED:
                        return
                    self._dagger_phase = DAggerPhase.PAUSED
                    self._lifecycle_changed.notify_all()
                if self._save_in_progress.is_set():
                    self.logger.info(
                        "Esc accepted while an episode is saving; stopping hardware before waiting for dataset finalization"
                    )
                else:
                    self.correction_recorder.pause_capture()
                try:
                    with self._observation_capture_lock, self._robot_command_lock:
                        self.dagger_controller.abort_to_paused()
                except Exception:
                    self.logger.exception("Could not establish paused Piper state during shutdown")
                try:
                    # Disconnect CAN and both arms before any potentially slow
                    # video finalization or Hub upload.
                    super().stop()
                finally:
                    try:
                        self.correction_recorder.close(
                            save_partial=self.config.dagger_save_partial_correction_on_stop
                        )
                    except Exception:
                        # Dataset failure must not mask a completed hardware stop
                        # (or replace a hardware-stop exception already in flight).
                        self.logger.exception("DAgger dataset finalization failed during shutdown")
        finally:
            self._show_status(StatusCue.EXIT)
            self._close_status_indicator()


def _render_status(client: RobotClientDagger) -> str:
    with client.latest_action_lock:
        latest = client.latest_action
        queued = client.action_schedule.qsize(latest)
    recorder = client.correction_recorder
    fault = client.dagger_fault
    fault_text = "" if fault is None else f", fault={fault!r}"
    saving_text = ", saving_episode=True" if client.dagger_save_in_progress else ""
    return (
        f"rollout={client.rollout_state.value}, phase={client.dagger_phase.value}, "
        f"generation={client.rollout_generation}, task={client.task!r}, "
        f"executed_action_step={latest}, scheduled={queued}, "
        f"correction_frames={recorder.frames_in_window}, "
        f"episode={client._episode_progress()}{saving_text}{fault_text}"
    )


def _collection_progress_postfix(client: RobotClientDagger, *, color: bool | None = None) -> str:
    if client.dagger_save_in_progress:
        phase_text, phase_color = "episode 保存中", _ANSI_BOLD_BRIGHT_MAGENTA
    else:
        phase_text, phase_color = {
            DAggerPhase.AUTONOMOUS: ("自主推理", _ANSI_BOLD_BRIGHT_GREEN),
            DAggerPhase.PAUSED: ("已暂停", _ANSI_BOLD_BRIGHT_YELLOW),
            DAggerPhase.CORRECTING: ("人工接管", _ANSI_BOLD_BRIGHT_CYAN),
            DAggerPhase.FAULT: ("FAULT", _ANSI_BOLD_BRIGHT_RED),
        }[client.dagger_phase]
    phase_status = _terminal_emphasis(f"阶段={phase_text}", phase_color, enabled=color)
    return (
        f"{phase_status} | 当前 correction={client.correction_recorder.frames_in_window} 帧 | "
        f"episode={client._episode_progress()}"
    )


def _make_collection_progress(client: RobotClientDagger) -> Any:
    target = client.dagger_episode_target
    saved = client.correction_recorder.saved_episodes
    return tqdm(
        total=target,
        initial=saved,
        desc="\033[1;96m◆ DAgger 采集进度 ◆\033[0m",
        unit="episode",
        dynamic_ncols=True,
        colour="#00ff66",
        disable=None,
    )


@contextlib.contextmanager
def _progress_safe_logging() -> Iterator[None]:
    """Redirect console records through tqdm without widening its log level.

    ``tqdm.contrib.logging.logging_redirect_tqdm`` copies the console
    formatter and filters but not the handler level.  LeRobot deliberately
    keeps the root logger at ``NOTSET`` so its DEBUG file handler can retain
    diagnostics; an unlevelled replacement console handler would therefore
    print every DEBUG record.  Preserve the original console threshold while
    leaving the DEBUG file handler untouched.
    """

    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_handler_ids = {id(handler) for handler in original_handlers}
    console_levels = [
        handler.level
        for handler in original_handlers
        if isinstance(handler, logging.StreamHandler)
        and getattr(handler, "stream", None) in (sys.stdout, sys.stderr)
    ]
    console_level = min(console_levels, default=logging.INFO)

    with logging_redirect_tqdm(tqdm_class=tqdm):
        for handler in root_logger.handlers:
            if id(handler) not in original_handler_ids:
                handler.setLevel(console_level)
        yield


def _refresh_collection_progress(progress: Any, client: RobotClientDagger) -> None:
    saved = client.correction_recorder.saved_episodes
    progress.n = min(saved, progress.total) if progress.total is not None else saved
    progress.set_postfix_str(_collection_progress_postfix(client), refresh=False)
    progress.refresh()


def _monitor_collection_progress(
    progress: Any,
    client: RobotClientDagger,
    shutdown: threading.Event,
) -> None:
    while not shutdown.wait(timeout=0.2):
        _refresh_collection_progress(progress, client)


def _interactive_prompt(client: RobotClientDagger) -> None:
    print(
        "Piper remote RTC DAgger ready. Commands: /start [task], /pause, "
        "/correct, /task <task>, /reset, /status, /help, /stop",
        flush=True,
    )
    while client.running and not client.operator_stop_event.is_set():
        try:
            line = input("dagger> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not client.running:
            break
        if not line:
            continue
        command, _, arguments = line.partition(" ")
        command = command.lower()
        arguments = arguments.strip()

        if command == "/start":
            ok = client.start_rollout(arguments or None)
        elif command == "/pause":
            ok = client.pause_autonomy()
        elif command == "/correct":
            ok = (
                client.stop_correction()
                if client.dagger_phase is DAggerPhase.CORRECTING
                else client.start_correction()
            )
        elif command == "/reset":
            ok = client._handle_operator_event("reset")
        elif command == "/task":
            if not arguments:
                print(f"task={client.task!r}", flush=True)
                continue
            ok = client.set_task(arguments)
        elif command == "/status":
            print(_render_status(client), flush=True)
            continue
        elif command == "/help":
            print(
                "/start [task]  align both arms and start a clean autonomous rollout\n"
                "/pause         invalidate in-flight RTC work and stop computer commands\n"
                "/correct       start correction, or save it and reset the task\n"
                "/task <task>   change the language instruction\n"
                "/reset         discard unsaved correction, reset robot, then restore environment\n"
                "/status        show lifecycle, DAgger, schedule, and episode i/N progress\n"
                "/stop          finalize the dataset and disconnect",
                flush=True,
            )
            continue
        elif command == "/stop":
            break
        else:
            print("unknown command; type /help", flush=True)
            continue

        print(("ok" if ok else "rejected") + f": {_render_status(client)}", flush=True)


RobotClient = RobotClientDagger


@draccus.wrap()
def async_client(cfg: RobotClientDaggerConfig) -> None:
    configure_deployment_logging(debug=cfg.debug_rtc_diagnostics)
    logging.info(pformat(asdict(cfg)))
    client = RobotClientDagger(cfg)
    if not client.start():
        client.stop()
        return

    threads = [
        threading.Thread(target=client.receive_actions, daemon=True, name="RTCActionReceiver"),
        threading.Thread(target=client.observation_sender, daemon=True, name="RTCObservationSender"),
        threading.Thread(target=client.control_loop, daemon=True, name="RTCLocalControl"),
    ]
    for thread in threads:
        thread.start()

    progress = _make_collection_progress(client)
    progress_shutdown = threading.Event()
    progress_thread = threading.Thread(
        target=_monitor_collection_progress,
        args=(progress, client, progress_shutdown),
        daemon=True,
        name="DAggerCollectionProgress",
    )
    progress_thread.start()

    try:
        # Route every console log through tqdm.write() while the live bar is
        # active. This keeps key-transition messages and genuine warnings as
        # complete lines above the bar instead of cutting through it.
        with _progress_safe_logging():
            if cfg.dagger_input_device == "prompt":
                _run_interactive_prompt(client, _interactive_prompt)
            else:
                client.start_operator_input()
                while client.running and not client.operator_stop_event.wait(timeout=0.2):
                    pass
    finally:
        progress_shutdown.set()
        progress_thread.join(timeout=1.0)
        _refresh_collection_progress(progress, client)
        progress.close()
        client.stop()
        for thread in threads:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        if cfg.debug_visualize_queue_size and client.action_queue_size:
            visualize_action_queue_size(list(client.action_queue_size))
        client.logger.info("Piper remote RTC DAgger client stopped")

    if client.worker_error is not None:
        raise RuntimeError(f"RTC client stopped after a worker failure: {client.worker_error}")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
