"""Local DAgger: autonomous rollout, gated human correction, save and reset."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from enum import StrEnum
from queue import Empty, Queue
from threading import Lock, RLock, Thread, current_thread

import numpy as np

from lerobot.light import StatusCue, make_status_indicator

from ..control.dagger_io import DaggerRobotIO
from ..control.piper_dagger import PiperControlMode, PiperHandoverAlignmentError
from ..inference.engine import LocalPolicyEngine
from ..recording.correction import DaggerCorrectionRecorder
from ..shared.dagger_configs import LocalDaggerConfig
from .deployment import LocalDeployment, State

logger = logging.getLogger(__name__)


class DaggerPhase(StrEnum):
    """Operator-visible ownership of the two arms."""

    PAUSED = "paused"
    AUTONOMOUS = "autonomous"
    CORRECTING = "correcting"
    FAULT = "fault"


class LocalDagger(LocalDeployment):
    """Reuse local inference while owning independent DAgger transitions and capture."""

    def __init__(
        self,
        config: LocalDaggerConfig,
        robot,
        *,
        engine_factory=LocalPolicyEngine,
        robot_io=None,
        recorder_factory=DaggerCorrectionRecorder,
        status_indicator=None,
    ):
        """Create the adapter and its synchronization state."""
        io = robot_io if robot_io is not None else DaggerRobotIO(robot, config.hardware, config.action_keys)
        # Do not start motion until recording and the DAgger watchdog are initialized.
        super().__init__(replace(config, autostart=False), robot, engine_factory=engine_factory, robot_io=io)
        self.dagger_config = config
        self.recorder = None
        self._recorder_factory = recorder_factory
        self._phase = DaggerPhase.PAUSED
        self._dagger_error: str | None = None
        self._saving = False
        self._transition_lock = RLock()
        self._dagger_stop_lock = Lock()
        self._cleanup_complete = False
        self._operator_lock = Lock()
        self._operator_pending = False
        self._operator_queue = Queue(maxsize=1)
        self._operator_thread = None
        self._watchdog_thread = None
        self._indicator = status_indicator or make_status_indicator(config.status_indicator, log=logger)

    @property
    def phase(self) -> DaggerPhase:
        """Return the current handover phase."""
        with self._condition:
            return self._phase

    def status(self) -> dict:
        """Include this run's correction count and any required recovery."""
        result = super().status()
        with self._condition:
            result.update(
                phase=str(self._phase),
                saving=self._saving,
                dagger_error=self._dagger_error,
                num_episodes=self.dagger_config.num_episodes,
            )
        recorder = self.recorder
        result.update(
            recording=bool(recorder and recorder.active),
            frames_in_window=recorder.frames_in_window if recorder else 0,
            saved_episodes=recorder.saved_episodes if recorder else 0,
        )
        return result

    def _show(self, cue: StatusCue) -> None:
        try:
            self._indicator.show(cue)
        except Exception:
            logger.exception("Could not update DAgger status indicator")

    def _ensure_live(self) -> None:
        if self.stop_event.is_set():
            if self.io.controller is not None:
                self.io.controller.bridge.disable()
            raise RuntimeError("DAgger is stopping")

    @contextmanager
    def _transition(self):
        if not self._transition_lock.acquire(blocking=False):
            raise RuntimeError("DAgger transition/save/reset in progress; stop remains available")
        try:
            self._ensure_live()
            yield
        finally:
            self._transition_lock.release()

    def start(self) -> None:
        """Warm up the sole local policy worker and initialize correction capture."""
        try:
            super().start()
            self._ensure_live()
            self.recorder = self._recorder_factory(
                self.io.robot,
                self.io.controller.read_expert_action,
                self.dagger_config.dataset,
                observation_lock=self.io.capture_lock,
                observation_reader=lambda: self.io.correction_observation(self.config.observation_max_age_s),
            )
            self.recorder.start()
            self._indicator.start()
            # The watchdog also waits for dataset discard. Hardware access is
            # fenced by _command_lock, capture_lock and stop_event; do not make
            # slow image cleanup a prerequisite for disconnecting the arms.
            self._watchdog_thread = Thread(
                target=self._correction_watchdog, name="LocalDaggerWatchdog", daemon=True
            )
            self._watchdog_thread.start()
            self._operator_thread = Thread(
                target=self._operator_loop, name="LocalDaggerOperator", daemon=True
            )
            self._operator_thread.start()
            self._show(StatusCue.PAUSED)
            if self.dagger_config.autostart:
                self.resume()
        except BaseException as exc:
            self._fault("DAgger startup", exc)
            self.stop()
            raise

    def _target_reached(self) -> bool:
        return bool(
            self.dagger_config.num_episodes
            and self.recorder
            and self.recorder.saved_episodes >= self.dagger_config.num_episodes
        )

    def resume(self) -> None:
        """Start a fresh rollout only from READY, after any correction has been reset."""
        with self._transition():
            with self._condition:
                if self._state != State.READY or self._phase != DaggerPhase.PAUSED or self._dagger_error:
                    raise RuntimeError(
                        "DAgger /start requires READY; finish or discard the correction and reset"
                    )
                if self._target_reached():
                    raise RuntimeError("DAgger episode target has been reached")
            try:
                with self._command_lock, self.io.capture_lock:
                    self._ensure_live()
                    pose = self.io.controller.prepare_autonomy()
                    self._ensure_live()
                    self.io.seed(pose)
                    with self._condition:
                        self._invalidate_locked(
                            seed=np.asarray([pose[k] for k in self.io.action_keys], dtype=np.float32)
                        )
                        self._first_action_deadline = time.monotonic() + self.config.inference_timeout_s
                        self._state = State.RUNNING
                        self._phase = DaggerPhase.AUTONOMOUS
                        self._condition.notify_all()
            except Exception as exc:
                self._recoverable_fault("start rollout", exc)
                raise
            self._show(StatusCue.AUTONOMOUS)

    def pause(self) -> None:
        """Fence policy motion before aligning the master for a possible handover."""
        with self._transition():
            if self.state != State.RUNNING or self.phase != DaggerPhase.AUTONOMOUS:
                raise RuntimeError("DAgger /pause requires autonomous motion")
            super().pause()
            with self._condition:
                self._phase = DaggerPhase.PAUSED
            try:
                with self._command_lock, self.io.capture_lock:
                    self._ensure_live()
                    self.io.controller.pause_computer_output()
            except Exception as exc:
                self._recoverable_fault("pause", exc)
                raise
            self._show(StatusCue.PAUSED)

    def start_correction(self) -> None:
        """Verify physical alignment, then allow the master CAN stream and recording."""
        with self._transition():
            if self.state != State.PAUSED or self.phase != DaggerPhase.PAUSED or self._dagger_error:
                raise RuntimeError("Pause autonomy before starting a correction")
            try:
                with self._command_lock, self.io.capture_lock:
                    self._ensure_live()
                    self.io.controller.enter_correction()
                    self._ensure_live()
                    self.recorder.begin(self._task)
                    with self._condition:
                        self._invalidate_locked()
                        self._phase = DaggerPhase.CORRECTING
            except PiperHandoverAlignmentError:
                # The controller has already closed the gate and restored PAUSED.
                self._show(StatusCue.WARNING)
                raise
            except Exception as exc:
                self._recoverable_fault("enter correction", exc)
                raise
            self._show(StatusCue.CORRECTING)

    def finish_correction(self) -> None:
        """Close capture and the CAN gate, save one window, then reset for the next rollout."""
        with self._transition():
            if self.phase != DaggerPhase.CORRECTING:
                raise RuntimeError("There is no active correction to save")
            try:
                self.recorder.pause_capture()
                if self.recorder.last_error is not None:
                    raise RuntimeError("Correction capture failed") from self.recorder.last_error
                with self._command_lock, self.io.capture_lock:
                    self._ensure_live()
                    self.io.controller.leave_correction()
                    with self._condition:
                        self._phase = DaggerPhase.PAUSED
                        self._invalidate_locked()
                        self._saving = True
                # No hardware/command lock during image flush, parquet write or ffmpeg.
                self.recorder.finish_window(save=True)
            except Exception as exc:
                self._recoverable_fault("save correction", exc)
                raise
            finally:
                with self._condition:
                    self._saving = False
            if self.stop_event.is_set():
                return
            if self._target_reached():
                self._show(StatusCue.COMPLETE)
                self.request_stop()
            else:
                self.reset()

    def reset(self) -> None:
        """Discard an unfinished window and restore both arms before reopening READY."""
        with self._transition():
            if self.state not in {State.READY, State.PAUSED}:
                raise RuntimeError("Pause autonomy before resetting DAgger")
            try:
                self.recorder.pause_capture()
                with self._command_lock, self.io.capture_lock:
                    self._ensure_live()
                    with self._condition:
                        self._phase = DaggerPhase.PAUSED
                        self._invalidate_locked()
                    self.io.controller.abort_to_paused()
                self.recorder.finish_window(save=False)
                self._show(StatusCue.RESETTING)
                super().reset()
            except Exception as exc:
                if self.phase != DaggerPhase.FAULT:
                    self._recoverable_fault("reset", exc)
                raise
            if not self.stop_event.is_set():
                self.recorder.acknowledge_error()
                with self._condition:
                    self._dagger_error = None
                    self._phase = DaggerPhase.PAUSED
                self._show(StatusCue.PAUSED)

    def set_task(self, task: str) -> None:
        """Change the task between rollouts without relabeling an active window."""
        with self._transition():
            if self.state != State.READY or self.phase != DaggerPhase.PAUSED:
                raise RuntimeError("Change the DAgger task in READY before /start")
            super().set_task(task)

    def _recoverable_fault(self, operation: str, error: BaseException, *, restore_hold: bool = True) -> None:
        """Revoke correction capture and require an explicit successful reset."""
        if self.stop_event.is_set():
            return
        if self.recorder:
            self.recorder.request_pause()
        try:
            with self._command_lock, self.io.capture_lock:
                if self.stop_event.is_set():
                    return
                with self._condition:
                    self._state = State.PAUSED
                    self._phase = DaggerPhase.FAULT
                    self._dagger_error = f"{operation}: {type(error).__name__}: {error}"
                    self._invalidate_locked()
                if restore_hold:
                    self.io.controller.abort_to_paused()
                else:
                    # A failed forwarding link can leave the human-held master
                    # far from the follower. Only an explicit reset may switch
                    # it back to follower mode and command a common hold pose.
                    self.io.controller.bridge.disable()
        except Exception as recovery_error:
            self._fault("DAgger recovery", recovery_error)
            return
        self._show(StatusCue.FAULT)
        logger.error("DAgger requires reset: %s", self._dagger_error)

    def _fault(self, operation: str, error: BaseException) -> None:
        if operation == "reset":
            self._recoverable_fault(operation, error)
            return
        if self.recorder:
            self.recorder.request_pause()
        controller = self.io.controller
        try:
            if controller is not None:
                controller.bridge.disable()
        finally:
            super()._fault(operation, error)

    def _correction_watchdog(self) -> None:
        while not self.stop_event.wait(0.02):
            if self.phase != DaggerPhase.CORRECTING:
                continue
            if not self._transition_lock.acquire(blocking=False):
                continue
            try:
                if self.phase != DaggerPhase.CORRECTING or self.stop_event.is_set():
                    continue
                error = self.recorder.last_error
                restore_hold = True
                operation = "record correction"
                if error is None:
                    try:
                        with self._command_lock, self.io.capture_lock:
                            if not self.stop_event.is_set():
                                self.io.controller.assert_correction_healthy()
                    except Exception as exc:
                        error = exc
                        restore_hold = False
                        operation = "CAN correction bridge"
                if error is not None:
                    self._recoverable_fault(operation, error, restore_hold=restore_hold)
                    # Discard the failed window outside the hardware locks. A
                    # pending image write must not postpone gate closure or stop.
                    try:
                        self.recorder.finish_window(save=False)
                    except Exception:
                        logger.exception("Could not discard the failed correction window")
            finally:
                self._transition_lock.release()

    def request_stop(self) -> None:
        """Out-of-band stop request, independent of operator queues and encoding."""
        self.stop_event.set()
        if self.recorder:
            self.recorder.request_pause()
        with self._condition:
            self._condition.notify_all()
        if self.io.controller is not None:
            try:
                self.io.controller.bridge.disable()
            except Exception as exc:
                with self._condition:
                    if self._failure is None:
                        self._failure = f"DAgger bridge stop: {exc}"

    def submit_operator_event(self, name: str, value: str | None = None) -> bool:
        """Drop busy/duplicate events and fence accepted events to their originating state."""
        if name == "stop":
            self.request_stop()
            return True
        if self.stop_event.is_set() or not self._transition_lock.acquire(blocking=False):
            return False
        try:
            with self._operator_lock, self._condition:
                if self._operator_pending:
                    return False
                self._operator_pending = True
                self._operator_queue.put_nowait((name, value, self._generation, self._state, self._phase))
            return True
        finally:
            self._transition_lock.release()

    def _operator_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                name, value, generation, state, phase = self._operator_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                with self._transition():
                    with self._condition:
                        if (generation, state, phase) != (self._generation, self._state, self._phase):
                            continue
                    self._operator_action(name, value)
            except Exception as exc:
                logger.error("DAgger operator command rejected: %s", exc)
            finally:
                with self._operator_lock:
                    self._operator_pending = False

    def _operator_action(self, name: str, value: str | None) -> None:
        if name == "advance":
            if self.state == State.READY:
                self.resume()
            elif self.phase == DaggerPhase.AUTONOMOUS:
                self.pause()
            elif self.phase == DaggerPhase.CORRECTING:
                self.finish_correction()
            else:
                self.start_correction()
        elif name == "start":
            if value:
                self.set_task(value)
            self.resume()
        elif name == "pause":
            self.pause()
        elif name == "correct":
            if self.phase == DaggerPhase.CORRECTING:
                self.finish_correction()
            else:
                self.start_correction()
        elif name == "reset":
            self.reset()
        elif name == "task":
            self.set_task(value or "")
        else:
            raise ValueError(f"Unknown DAgger event: {name}")

    def stop(self) -> None:
        """Stop hardware before waiting for encoding, and finalize each dataset once."""
        with self._dagger_stop_lock:
            if self._cleanup_complete:
                return
            self.request_stop()
            errors = []
            with self._command_lock, self.io.capture_lock:
                if self._dagger_error and self._failure is None:
                    self._failure = self._dagger_error
                controller = self.io.controller
                if (
                    controller is not None
                    and controller.mode == PiperControlMode.CORRECTING
                    and not self._failure
                ):
                    try:
                        controller.abort_to_paused()
                    except Exception as exc:
                        self._failure = f"DAgger stop: {exc}"
                        errors.append(exc)
                with self._condition:
                    self._phase = DaggerPhase.FAULT if self._failure else DaggerPhase.PAUSED
            try:
                super().stop()
            except Exception as exc:
                errors.append(exc)
            # A saver can still finish after hardware closes, but cannot resume or reset it.
            for worker in (self._operator_thread, self._watchdog_thread):
                if worker is not None and worker is not current_thread():
                    worker.join(timeout=self.config.stop_timeout_s)
            try:
                if self.recorder:
                    self.recorder.close(
                        save_partial=self.dagger_config.save_partial_on_stop and not self._failure
                    )
                    # Encoding may have failed after the stop request, while the
                    # hardware was already closing. Do not report that as success.
                    if self.recorder.last_error is not None:
                        raise RuntimeError(
                            "DAgger recording failed before finalization"
                        ) from self.recorder.last_error
            except Exception as exc:
                errors.append(exc)
                with self._condition:
                    self._failure = self._failure or f"DAgger finalization: {exc}"
                    self._phase = DaggerPhase.FAULT
            finally:
                try:
                    self._indicator.close()
                except Exception as exc:
                    errors.append(exc)
            self._cleanup_complete = not self._connected
            if errors:
                raise ExceptionGroup("Local DAgger cleanup failed", errors)
