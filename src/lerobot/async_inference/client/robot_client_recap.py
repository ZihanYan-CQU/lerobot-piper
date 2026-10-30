# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Independent full-rollout RECAP entry point using DAgger's Piper/RTC control.

Right: start -> pause -> intervene -> pause. S/F save a task with an explicit
result, then reset. Left discards an unconfirmed task and resets. Esc disconnects
hardware before discarding an unconfirmed task or finishing an accepted save.
"""

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import draccus

from lerobot.light import StatusCue
from lerobot.transport import services_pb2
from lerobot.utils.import_utils import register_third_party_plugins

from ..shared.deployment_diagnostics import configure_deployment_logging
from ..shared.helpers import get_logger
from ..shared.image_transport import parse_image_transport_metadata
from ..shared.metadata_recap import IDENTITY_METADATA_KEY, validate_policy_identity
from .piper_hil_recap import (
    RecapControlSample,
    RecapDatasetConfig,
    RecapOutcome,
    RecapPhase,
    RecapRolloutRecorder,
)
from .robot_client_dagger import DAggerPhase, RobotClientDagger, RobotClientDaggerConfig
from .robot_client_deployment import RobotClientDeployment, RolloutState, _run_interactive_prompt


@dataclass
class RobotClientRecapConfig(RobotClientDaggerConfig):
    recap_dataset_repo_id: str | None = None
    recap_dataset_root: str | Path | None = None
    recap_dataset_fps: int = 0
    recap_dataset_video: bool = True
    recap_dataset_resume: bool = False
    recap_dataset_push_to_hub: bool = False
    recap_dataset_private: bool | None = None
    recap_image_writer_threads_per_camera: int = 4
    recap_num_episodes: int = 0
    recap_success_key: str = "s"
    recap_failure_key: str = "f"

    def __post_init__(self):
        super().__post_init__()
        if self.dagger_dataset_repo_id or self.dagger_dataset_root or self.dagger_num_episodes:
            raise ValueError("Use recap_dataset_* / recap_num_episodes for the RECAP entry point")
        if not self.recap_dataset_repo_id or self.recap_dataset_root is None:
            raise ValueError("RECAP requires recap_dataset_repo_id and recap_dataset_root")
        if self.recap_dataset_fps < 0 or self.recap_num_episodes < 0:
            raise ValueError("RECAP fps and episode target must be non-negative")
        if self.recap_image_writer_threads_per_camera < 0:
            raise ValueError("RECAP image writer thread count must be non-negative")
        if self.dagger_input_device == "pedal":
            raise ValueError("RECAP currently requires keyboard or prompt for explicit task outcome labels")
        keys = (
            self.dagger_advance_key,
            self.dagger_reset_key,
            self.dagger_stop_key,
            self.recap_success_key,
            self.recap_failure_key,
        )
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("RECAP operator keys must be nonempty and distinct")

    def to_dict(self):
        return asdict(self)


class _ControlOnlyRecorder:
    """Satisfy the inherited handover API without opening a DAgger dataset."""

    enabled = True
    saved_episodes = 0
    frames_in_window = 0
    last_error = None

    def start(self):
        pass

    def begin(self, task):
        pass

    def pause_capture(self):
        pass

    def finish_window(self, *, save):
        return 0

    def close(self, *, save_partial=False):
        pass


class RobotClientRecap(RobotClientDagger):
    prefix = "robot_client_recap"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientRecapConfig, *, rollout_recorder=None, **kwargs):
        self.policy_identity: dict | None = None
        self.rollout_recorder: RecapRolloutRecorder | None = None
        self._recap_close_lock = threading.Lock()
        self._recap_closed = False
        self.recap_error: str | None = None
        super().__init__(config, correction_recorder=_ControlOnlyRecorder(), **kwargs)
        try:
            self.rollout_recorder = rollout_recorder or RecapRolloutRecorder(
                self._follower_robot,
                self._read_recording_sample,
                RecapDatasetConfig(
                    repo_id=config.recap_dataset_repo_id,
                    root=config.recap_dataset_root,
                    fps=config.recap_dataset_fps or config.fps,
                    use_videos=config.recap_dataset_video,
                    resume=config.recap_dataset_resume,
                    push_to_hub=config.recap_dataset_push_to_hub,
                    private=config.recap_dataset_private,
                    image_writer_threads_per_camera=config.recap_image_writer_threads_per_camera,
                ),
                observation_lock=self._observation_capture_lock,
            )
        except Exception:
            self.stop()
            raise

    @property
    def dagger_episode_target(self):
        return self.config.recap_num_episodes or None

    @property
    def dagger_collection_complete(self):
        return (
            self.rollout_recorder is not None
            and self.dagger_episode_target is not None
            and self.rollout_recorder.saved_episodes >= self.dagger_episode_target
        )

    def _episode_progress(self):
        return f"{self.rollout_recorder.saved_episodes}/{self.dagger_episode_target or '∞'}"

    def start(self) -> bool:
        self._start_status_indicator()
        self.policy_identity = None
        # Use deployment's readiness/calibration gate before starting recording.
        if not RobotClientDeployment.start(self):
            return False
        self.rollout_recorder.start()
        self._show_status(StatusCue.RESET_REQUIRED)
        self.logger.info("RECAP connected to %s", self.policy_identity["model_version"])
        return True

    def _send_policy_instructions(self, request: services_pb2.PolicySetup) -> None:
        _, call = self.stub.SendPolicyInstructions.with_call(request)
        if self.policy_setup.client_image_resize:
            self._image_transport_plan = parse_image_transport_metadata(call.initial_metadata())
        metadata = dict(call.trailing_metadata() or ())
        if IDENTITY_METADATA_KEY not in metadata:
            raise RuntimeError("Use policy_server_recap: server did not return RECAP checkpoint identity")
        identity = validate_policy_identity(json.loads(metadata[IDENTITY_METADATA_KEY]))
        if identity["policy_type"] != self.config.policy_type:
            raise ValueError("RECAP server identity does not match the requested policy type")
        self.policy_identity = identity

    def _command_snapshot_required(self) -> bool:
        """Autonomous action recording requires metadata even with debug off."""
        return True

    def _read_recording_sample(self) -> RecapControlSample:
        with self._lifecycle_lock:
            phase = self._dagger_phase
            generation = self._rollout_generation
            if phase is DAggerPhase.AUTONOMOUS:
                command = self.command_diagnostics_snapshot()
                if command is not None:
                    return RecapControlSample(
                        RecapPhase.AUTONOMOUS,
                        dict(zip(self.robot.action_features, command.command.tolist(), strict=True)),
                        generation,
                        command.policy_step,
                        command.source_observation_id,
                        command.monotonic_time,
                    )
                return RecapControlSample(RecapPhase.AUTONOMOUS, generation=generation)
            if phase is DAggerPhase.CORRECTING:
                return RecapControlSample(
                    RecapPhase.INTERVENTION,
                    self.dagger_controller.read_expert_action(),
                    generation=generation,
                )
            return RecapControlSample(RecapPhase.PAUSED, generation=generation)

    def _set_fault(self, operation, exc):
        if self.rollout_recorder is not None:
            self.rollout_recorder.request_pause()
        super()._set_fault(operation, exc)

    def set_task(self, task: str) -> bool:
        with self._dagger_transition_lock:
            if self.rollout_recorder.has_rollout or self._save_in_progress.is_set():
                return False
            return super().set_task(task)

    def start_rollout(self, task: str | None = None) -> bool:
        if not self._dagger_transition_lock.acquire(blocking=False):
            return False
        try:
            if (
                self.policy_identity is None
                or self.rollout_state is not RolloutState.READY
                or self.dagger_phase is DAggerPhase.FAULT
                or self.rollout_recorder.has_rollout
                or self._save_in_progress.is_set()
                or not self.running
                or self.operator_stop_event.is_set()
                or self._request_exit_if_target_reached()
            ):
                return False
            # JSON round trip makes configs (Paths/enums/tuples) portable sidecar values.
            config = json.loads(json.dumps(self.config.to_dict(), default=str))
            self.rollout_recorder.begin_rollout(
                task if task is not None else self.task, self.policy_identity, config
            )
            if not super().start_rollout(task):
                self.rollout_recorder.discard_rollout()
                return False
            self.rollout_recorder.resume_capture()
            return True
        except Exception as exc:
            self._set_fault("start RECAP rollout", exc)
            return False
        finally:
            self._dagger_transition_lock.release()

    def _recorded_transition(self, transition) -> bool:
        with self._dagger_transition_lock:
            if self._save_in_progress.is_set() or not self.running or self.operator_stop_event.is_set():
                return False
            self.rollout_recorder.pause_capture()
            try:
                ok = transition()
                if self.rollout_recorder.has_rollout and self.dagger_phase is not DAggerPhase.FAULT:
                    self.rollout_recorder.resume_capture()
                return ok
            except Exception as exc:
                self._set_fault("RECAP phase transition", exc)
                return False

    def pause_autonomy(self) -> bool:
        return self._recorded_transition(super().pause_autonomy)

    def start_correction(self) -> bool:
        return self._recorded_transition(super().start_correction)

    def stop_correction(self) -> bool:
        def leave():
            if self.dagger_phase is not DAggerPhase.CORRECTING:
                return False
            with self._lifecycle_changed:
                self._dagger_phase = DAggerPhase.PAUSED
                self._lifecycle_changed.notify_all()
            try:
                with self._observation_capture_lock, self._robot_command_lock:
                    self.dagger_controller.leave_correction()
            except Exception as exc:
                self._set_fault("stop correction", exc)
                return False
            self._show_status(StatusCue.PAUSED)
            self.logger.info("RECAP intervention ended; label the task with S/F (or /success, /failure)")
            return True

        return self._recorded_transition(leave)

    def finish_rollout(self, outcome: RecapOutcome | str, *, reason: str = "operator") -> bool:
        outcome = RecapOutcome(outcome)
        if not self._dagger_transition_lock.acquire(blocking=False):
            return False
        try:
            if (
                not self.running
                or self.operator_stop_event.is_set()
                or not self.rollout_recorder.has_rollout
                or self._save_in_progress.is_set()
            ):
                return False
            self.rollout_recorder.pause_capture()
            if self.dagger_phase is DAggerPhase.FAULT or self.rollout_recorder.last_error is not None:
                self.logger.warning("RECAP recording/control fault; reset or stop to discard this attempt")
                return False
            # Fence all outstanding inference before any handover or persistence.
            with self._lifecycle_changed:
                self._dagger_phase = DAggerPhase.PAUSED
                self._advance_generation_locked()
                self._clear_episode_state_locked()
                self._lifecycle_changed.notify_all()
            try:
                with self._observation_capture_lock, self._robot_command_lock:
                    self.dagger_controller.abort_to_paused()
            except Exception as exc:
                self._set_fault("finish RECAP rollout", exc)
                return False
            self.rollout_recorder.seal_rollout(outcome, reason=reason)
            self._save_in_progress.set()
            self._show_status(StatusCue.PAUSED)
        finally:
            self._dagger_transition_lock.release()
        # Esc must disconnect hardware while encoding is still in progress.
        try:
            frames = self.rollout_recorder.finish_rollout(outcome, reason=reason)
            self.logger.info(
                "Saved RECAP outcome=%s frames=%d episodes=%s", outcome, frames, self._episode_progress()
            )
        except Exception as exc:
            self.recap_error = f"RECAP save failed: {exc}"
            self._set_fault("save RECAP rollout", exc)
            self.operator_stop_event.set()
            return False
        finally:
            self._save_in_progress.clear()
        if not self.running or self.operator_stop_event.is_set() or self._request_exit_if_target_reached():
            return True
        return super().reset_rollout()

    def reset_rollout(self) -> bool:
        if not self._dagger_transition_lock.acquire(blocking=False):
            return False
        try:
            if self._save_in_progress.is_set():
                return False
            return super().reset_rollout()
        finally:
            self._dagger_transition_lock.release()

    def _prepare_reset(self) -> None:
        self.rollout_recorder.pause_capture()
        try:
            # Fence commands and leave teleoperation before clearing the buffer.
            super()._prepare_reset()
        finally:
            frames = self.rollout_recorder.discard_rollout()
            if frames:
                self.logger.info("Discarded unconfirmed RECAP attempt (%d frames)", frames)

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        error = self.rollout_recorder.last_error
        if error is not None and self.rollout_recorder.has_rollout:
            self._fail_worker("RECAP recording", error)
            return None
        if self._save_in_progress.is_set() or not self.rollout_recorder.active:
            return None
        return super().control_loop_action(verbose=verbose)

    def _announce_environment_reset(self, *, discarded_correction: bool):
        self.logger.warning("RECAP 已复位；未确认保存的本轮数据已丢弃。请恢复任务环境，再按右键开始下一轮。")

    def _log_operator_event(self, event, phase, rollout):
        self.logger.info("RECAP input=%s state=%s/%s", event, rollout.value, phase.value)

    def _apply_operator_event(self, event: str) -> bool:
        if event in ("success", "failure"):
            return self.finish_rollout(event)
        return super()._apply_operator_event(event)

    def start_operator_input(self) -> None:
        if self.config.dagger_input_device == "prompt" or self._operator_worker is not None:
            return
        from lerobot.utils.keyboard_input import create_key_listener

        self._operator_worker_shutdown.clear()
        self._operator_worker = threading.Thread(
            target=self._operator_loop, daemon=True, name="PiperRecapOperator"
        )
        self._operator_worker.start()
        mapping = {
            self.config.dagger_advance_key: "advance",
            self.config.dagger_reset_key: "reset",
            self.config.dagger_stop_key: "stop",
            self.config.recap_success_key: "success",
            self.config.recap_failure_key: "failure",
        }
        self._input_listener = create_key_listener(
            lambda name: self.submit_operator_event(mapping[name]) if name in mapping else None,
            controls_help=", ".join(f"{event}={key}" for key, event in mapping.items()),
        )

    def stop(self) -> None:
        recorder = self.rollout_recorder
        if recorder is not None:
            recorder.request_pause()
        try:
            # Base stop has only the inert correction recorder. All real dataset
            # finalization happens AFTER CAN/hardware disconnect below.
            super().stop()
        finally:
            with self._recap_close_lock:
                if recorder is not None and not self._recap_closed:
                    self._recap_closed = True
                    try:
                        recorder.close()
                    except Exception as exc:
                        self.recap_error = f"RECAP finalization failed: {exc}"
                        self.logger.exception(self.recap_error)


def _interactive_prompt(client: RobotClientRecap) -> None:
    print("RECAP: /start [task], /pause, /correct, /success, /failure, /reset, /status, /stop")
    while client.running and not client.operator_stop_event.is_set():
        try:
            line = input("recap> ").strip()
        except (EOFError, KeyboardInterrupt):
            client.submit_operator_event("stop")
            return
        if not client.running or client.operator_stop_event.is_set():
            return
        command, _, argument = line.partition(" ")
        if command == "/start":
            ok = client.start_rollout(argument.strip() or None)
        elif command == "/pause":
            ok = client.pause_autonomy()
        elif command == "/correct":
            ok = (
                client.stop_correction()
                if client.dagger_phase is DAggerPhase.CORRECTING
                else client.start_correction()
            )
        elif command == "/task":
            ok = bool(argument.strip()) and client.set_task(argument.strip())
        elif command == "/status":
            print(
                f"{client.rollout_state.value}/{client.dagger_phase.value}, saved={client._episode_progress()}, "
                f"frames={client.rollout_recorder.frames_in_window}, fault={client.dagger_fault}"
            )
            continue
        elif command in ("/success", "/failure", "/reset", "/stop"):
            ok = (
                client.submit_operator_event("stop")
                if command == "/stop"
                else client._handle_operator_event(command[1:])
            )
        else:
            print("Commands: /start, /pause, /correct, /success, /failure, /reset, /status, /stop")
            continue
        if not ok:
            print("当前状态不接受此操作；请检查 /status。")


@draccus.wrap()
def async_client(cfg: RobotClientRecapConfig) -> None:
    configure_deployment_logging(debug=cfg.debug_rtc_diagnostics)
    client = RobotClientRecap(cfg)
    threads = []
    try:
        if not client.start():
            return
        for target in (client.receive_actions, client.observation_sender, client.control_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            threads.append(thread)
        if cfg.dagger_input_device == "prompt":
            _run_interactive_prompt(client, _interactive_prompt)
        else:
            client.start_operator_input()
            while client.running and not client.operator_stop_event.wait(0.2):
                pass
    finally:
        client.stop()
        for thread in threads:
            thread.join(timeout=3.0)
    if client.worker_error or client.recap_error:
        raise RuntimeError(client.worker_error or client.recap_error)


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
