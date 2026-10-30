"""Single-process deployment with one policy owner and fenced motor dispatch."""

from __future__ import annotations

import logging
import math
import time
from enum import StrEnum
from threading import Condition, Event, Lock, Thread, current_thread

import numpy as np

from ..control.robot_io import RobotIO
from ..control.schedule import ActionSchedule
from ..inference.engine import LocalPolicyEngine
from ..shared.configs import LocalDeploymentConfig
from ..shared.metrics import Timings
from ..shared.types import Observation, Request

logger = logging.getLogger(__name__)


class State(StrEnum):
    """Observable lifecycle of one deployment instance."""

    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"
    FAULT = "fault"
    STOPPED = "stopped"


class LocalDeployment:
    """Coordinate capture, a single policy worker and independently paced control."""

    def __init__(
        self, config: LocalDeploymentConfig, robot, *, engine_factory=LocalPolicyEngine, robot_io=None
    ):
        """Construct an idle runtime without loading weights or touching hardware."""
        self.config = config
        self.io = robot_io if robot_io is not None else RobotIO(robot, config.action_keys)
        self._engine_factory = engine_factory
        self._condition = Condition()
        # All motor writes and lifecycle revocation use this order: command -> condition.
        self._command_lock = Lock()
        self._stop_lock = Lock()
        self.stop_event = Event()
        self._state = State.STOPPED
        self._started = False
        self._connected = False
        self._ready = False
        self._failure: str | None = None
        self._generation = 0
        self._sequence = 0
        self._latest: Observation | None = None
        self._last_used_sequence = -1
        self._valid_after = 0.0
        self._pending: Request | None = None
        self._started_at = 0.0
        self._first_action_deadline = 0.0
        self._has_dispatched = False
        self._last_dispatched_source: float | None = None
        self._task = config.task
        self._threads: list[Thread] = []
        self._initial_pose: dict[str, float] | None = None
        self.schedule = ActionSchedule(config.interpolation_multiplier)
        self.timings = Timings()
        self.engine = None

    @property
    def state(self) -> State:
        """Return the current lifecycle state under its synchronization lock."""
        with self._condition:
            return self._state

    @property
    def failure(self) -> str | None:
        """Return the first terminal worker error, if any."""
        with self._condition:
            return self._failure

    def status(self) -> dict:
        """Return a CPU-only status snapshot and bounded timing statistics."""
        with self._condition:
            return {
                "state": str(self._state),
                "task": self._task,
                "generation": self._generation,
                "remaining_actions": self.schedule.remaining,
                "completed_step": self.schedule.completed_step,
                "inference_pending": self._pending is not None,
                "failure": self._failure,
                **self.timings.report(),
            }

    def _invalidate_locked(self, *, seed: np.ndarray | None = None) -> None:
        self._generation += 1
        self._latest = None
        self._valid_after = time.monotonic()
        self.schedule.clear(seed)
        self._has_dispatched = False
        self._last_dispatched_source = None
        self._condition.notify_all()

    def _fault(self, operation: str, error: BaseException) -> None:
        message = None
        with self._command_lock, self._condition:
            if self._state == State.STOPPED:
                return
            if self._failure is None:
                self._failure = f"{operation}: {type(error).__name__}: {error}"
                message = self._failure
            self._state = State.FAULT
            self._invalidate_locked()
            self.stop_event.set()
        # A slow console/file handler must neither postpone revocation nor hold
        # the motor/lifecycle locks needed by status and terminal cleanup.
        if message is not None:
            logger.error("Local deployment fault: %s", message, exc_info=error)

    def start(self) -> None:
        """Connect hardware and await model warmup before permitting autonomy."""
        with self._command_lock, self._condition:
            if self._started or self.stop_event.is_set():
                raise RuntimeError("Create a new LocalDeployment after stopping; instances cannot restart")
            self._started = True
            self._state = State.STARTING
            self._started_at = time.monotonic()
        try:
            # Connection/calibration and startup reads are hardware operations
            # too: stop() must not disconnect halfway through their initialization.
            with self._command_lock:
                if self.stop_event.is_set():
                    raise RuntimeError("Startup cancelled")
                if self.config.reset_pose is not None and set(self.config.reset_pose) != set(
                    self.io.action_keys
                ):
                    raise ValueError("reset_pose must contain every robot action feature exactly once")
                self._connected = True  # Also clean up a partially failed connection.
                self.io.connect()
                self._initial_pose = self.io.read_pose()
                self.io.seed(self._initial_pose)
                self.schedule.clear(
                    np.asarray([self._initial_pose[k] for k in self.io.action_keys], dtype=np.float32)
                )
                for name, target in (
                    ("LocalObservation", self._capture_loop),
                    ("LocalInference", self._inference_loop),
                    ("LocalControl", self._control_loop),
                ):
                    thread = Thread(name=name, target=target, daemon=True)
                    self._threads.append(thread)
                    thread.start()
            with self._condition:
                completed = self._condition.wait_for(
                    lambda: self._ready or self.stop_event.is_set(), timeout=self.config.startup_timeout_s
                )
                if not completed:
                    raise TimeoutError("Local policy startup/warmup timed out")
                if self._failure or not self._ready:
                    raise RuntimeError(self._failure or "Startup cancelled")
            if self.config.autostart:
                self.resume()
        except BaseException as exc:
            self._fault("startup", exc)
            self.stop()
            raise

    def resume(self) -> None:
        """Start a new motion generation from a fresh measured pose and observation."""
        with self._command_lock:
            with self._condition:
                if self._state not in {State.READY, State.PAUSED}:
                    raise RuntimeError(f"Cannot start from {self._state}")
            pose = self.io.read_pose()
            self.io.seed(pose)
            with self._condition:
                seed = np.asarray([pose[key] for key in self.io.action_keys], dtype=np.float32)
                self._invalidate_locked(seed=seed)
                self._first_action_deadline = time.monotonic() + self.config.inference_timeout_s
                self._state = State.RUNNING
                self._condition.notify_all()

    def pause(self) -> None:
        """Revoke queued and in-flight predictions before returning to the caller."""
        with self._command_lock, self._condition:
            if self._state not in {State.RUNNING, State.READY, State.PAUSED}:
                raise RuntimeError(f"Cannot pause from {self._state}")
            self._state = State.PAUSED
            self._invalidate_locked()

    def set_task(self, task: str) -> None:
        """Invalidate old-task motion and require a new observation for the new task."""
        if not task.strip():
            raise ValueError("Task must be nonempty")
        with self._command_lock, self._condition:
            if self._state not in {State.RUNNING, State.READY, State.PAUSED}:
                raise RuntimeError(f"Cannot change task from {self._state}")
            self._task = task
            self._invalidate_locked()
            self._first_action_deadline = time.monotonic() + self.config.inference_timeout_s

    def reset(self) -> None:
        """Invalidate old inference immediately, restore pose, then wait for /start."""
        with self._command_lock, self._condition:
            if self._state not in {State.RUNNING, State.READY, State.PAUSED}:
                raise RuntimeError(f"Cannot reset from {self._state}")
            self._state = State.RESETTING
            self._invalidate_locked()
            generation = self._generation
        try:
            target = self.config.reset_pose
            if target is None and self.config.reset_to_initial_position:
                target = self._initial_pose
            self._restore_pose(target, generation)
            with self._command_lock:
                with self._condition:
                    if self._state != State.RESETTING or self._generation != generation:
                        return
                pose = self.io.read_pose()
                self.io.seed(pose)
                with self._condition:
                    self._invalidate_locked(
                        seed=np.asarray([pose[k] for k in self.io.action_keys], dtype=np.float32)
                    )
                    self._state = State.READY
                    self._condition.notify_all()
        except Exception as exc:
            self._fault("reset", exc)
            raise

    def _restore_pose(self, target: dict[str, float] | None, generation: int) -> None:
        with self._command_lock:
            with self._condition:
                if self._state != State.RESETTING or self._generation != generation:
                    return
            start = self.io.read_pose()
            self.io.seed(start)
        target = dict(start if target is None else target)
        gripper_targets = {}
        if self.config.reset_open_gripper and self.config.reset_gripper_open_position is not None:
            gripper_targets = {
                key: self.config.reset_gripper_open_position
                for key in self.io.action_keys
                if key.endswith("gripper.pos")
            }
            # Keep a held object at its measured opening throughout arm return.
            # The previous driver target may still be trying to close against it.
            target.update({key: start[key] for key in gripper_targets})
        before = np.asarray([start[k] for k in self.io.action_keys], dtype=np.float32)
        after = np.asarray([target[k] for k in self.io.action_keys], dtype=np.float32)
        steps = max(1, math.ceil(self.config.reset_duration_s * self.config.reset_fps))
        for index in range(1, steps + 1):
            deadline = time.monotonic() + 1 / self.config.reset_fps
            with self._command_lock:
                with self._condition:
                    if self._state != State.RESETTING or generation != self._generation:
                        return
                self.io.send(before + (after - before) * (index / steps))
            if self.config.reset_duration_s and self.stop_event.wait(max(0.0, deadline - time.monotonic())):
                return
        if not self._wait_for_reset_pose(target, generation, verify_gripper=not gripper_targets):
            return
        if gripper_targets:
            target.update(gripper_targets)
            self._wait_for_reset_pose(target, generation, phase="reset gripper release")

    def _wait_for_reset_pose(
        self,
        target: dict[str, float],
        generation: int,
        *,
        verify_gripper: bool = True,
        phase: str = "reset pose",
    ) -> bool:
        after = np.asarray([target[key] for key in self.io.action_keys], dtype=np.float32)
        settle_until = time.monotonic() + self.config.reset_settle_timeout_s
        stable_since = None
        while not self.stop_event.is_set():
            with self._command_lock:
                with self._condition:
                    if self._state != State.RESETTING or generation != self._generation:
                        return False
                # Keep advancing driver smoothing, then verify actual feedback.
                self.io.send(after)
                poses = self.io.read_reset_poses(max(0.001, settle_until - time.monotonic()))
                if not poses:
                    raise ValueError("Reset requires physical feedback")
                if any(
                    not math.isfinite(pose[key]) for pose in poses.values() for key in self.io.action_keys
                ):
                    raise ValueError("Reset feedback contains non-finite positions")
                errors = {
                    f"{arm}.{key}": abs(pose[key] - target[key])
                    for arm, pose in poses.items()
                    for key in self.io.action_keys
                    if (verify_gripper or not key.endswith("gripper.pos"))
                    and abs(pose[key] - target[key])
                    > (
                        self.config.reset_gripper_tolerance
                        if key.endswith("gripper.pos")
                        else self.config.reset_tolerance
                    )
                }
                now = time.monotonic()
                if now >= settle_until:
                    raise TimeoutError(f"{phase} did not settle: {errors}")
                if errors:
                    stable_since = None
                else:
                    stable_since = now if stable_since is None else stable_since
                    if now - stable_since >= self.config.reset_settle_s:
                        return True
            self.stop_event.wait(1 / self.config.reset_fps)
        return False

    def stop(self) -> None:
        """Revoke dispatch, join hardware workers and disconnect exactly once."""
        with self._stop_lock:
            with self._command_lock, self._condition:
                self._state = State.STOPPED
                self.stop_event.set()
                self._invalidate_locked()
            hardware = [t for t in self._threads if t.name != "LocalInference"]
            for thread in hardware:
                if thread is not current_thread():
                    thread.join(timeout=self.config.stop_timeout_s)
            if any(t.is_alive() for t in hardware):
                raise RuntimeError(
                    "Robot I/O did not stop; refusing to disconnect while an I/O call is active"
                )
            if self._connected:
                self._connected = False
                self.io.disconnect(faulted=self._failure is not None)
            # A cancelled model call owns no hardware and cannot publish actions.
            # In particular, it must not postpone a faulted Piper's stop command.
            for thread in self._threads:
                if thread not in hardware and thread is not current_thread():
                    thread.join(timeout=self.config.stop_timeout_s)
            if any(t.is_alive() for t in self._threads):
                logger.warning("The cancelled GPU call is still finishing; its result cannot be dispatched")

    def _capture_loop(self) -> None:
        period = 1 / self.config.observation_capture_fps
        last_warning = -math.inf
        while not self.stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self.stop_event.is_set() or self._state in {State.STARTING, State.RUNNING}
                )
                if self.stop_event.is_set():
                    return
                generation = self._generation
            deadline = time.monotonic() + period
            try:
                values, captured, source = self.io.capture(self.config.observation_max_age_s)
                if time.monotonic() - source > self.config.observation_max_age_s:
                    raise TimeoutError("Observation is already stale after acquisition")
                with self._condition:
                    if (
                        generation == self._generation
                        and captured >= self._valid_after
                        and source >= self._valid_after
                    ):
                        self._sequence += 1
                        self._latest = Observation(self._sequence, captured, source, values)
                        self._condition.notify_all()
            except Exception as exc:
                now = time.monotonic()
                if now - last_warning > 1.0:
                    logger.warning("Waiting for fresh local sensors: %s", exc)
                    last_warning = now
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def _fresh_locked(self) -> bool:
        return (
            self._latest is not None
            and self._latest.sequence > self._last_used_sequence
            and self._latest.captured_at >= self._valid_after
            and self._latest.source_at >= self._valid_after
            and time.monotonic() - self._latest.source_at <= self.config.observation_max_age_s
        )

    def _inference_loop(self) -> None:
        try:
            self.engine = self._engine_factory(
                self.config, self.io.robot.observation_features, self.io.action_keys
            )
            with self._condition:
                self._condition.wait_for(lambda: self.stop_event.is_set() or self._fresh_locked())
                if self.stop_event.is_set():
                    return
                observation = self._latest
            samples = self.engine.warmup(observation, self.stop_event.is_set)
            with self._condition:
                if self.stop_event.is_set():
                    return
                for elapsed in samples:
                    self.timings.add("inference", elapsed)
                self._ready = True
                self._state = State.READY
                self._condition.notify_all()
            engine_generation = -1
            rejected = 0
            while not self.stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self.stop_event.is_set()
                            or (self._state == State.RUNNING and self._fresh_locked())
                        )
                    )
                    if self.stop_event.is_set():
                        return
                    request = Request(
                        self._generation,
                        self._latest,
                        # The guidance horizon controls its mask, not the amount
                        # of real prefix available for latency conditioning.
                        self.schedule.snapshot(self.config.actions_per_chunk),
                        self._task,
                        time.monotonic(),
                    )
                    self._last_used_sequence = request.observation.sequence
                    self._pending = request
                    latency = self.timings.quantile("inference", self.config.latency_quantile)
                if engine_generation != request.generation:
                    self.engine.reset()
                    engine_generation = request.generation
                    rejected = 0
                prediction = self.engine.predict(request, latency)
                elapsed = time.monotonic() - request.started_at
                with self._condition:
                    self._pending = None
                    if (
                        self.stop_event.is_set()
                        or self._state != State.RUNNING
                        or request.generation != self._generation
                    ):
                        continue
                    if elapsed > self.config.inference_timeout_s:
                        raise TimeoutError("Local inference exceeded inference_timeout_s")
                    self.timings.add("inference", elapsed)
                    self.timings.add("observation_to_chunk", time.monotonic() - request.observation.source_at)
                    trained = self.config.rtc.mode == "trained"
                    supported = not (
                        trained
                        and len(request.prefix)
                        and math.ceil(elapsed * self.config.fps) > prediction.conditioned_delay
                    )
                    fresh = time.monotonic() - prediction.chunk.source_at <= self.config.action_max_age_s
                    accepted = (
                        supported
                        and fresh
                        and self.schedule.accept(
                            request.prefix,
                            prediction.chunk,
                            trained=trained,
                            delay=prediction.conditioned_delay,
                        )
                    )
                    if accepted:
                        self.timings.accepted_chunks += 1
                        rejected = 0
                    else:
                        self.timings.rejected_chunks += 1
                        rejected += 1
                    if rejected >= self.config.max_rejected_chunks:
                        raise RuntimeError(
                            "Repeated unusable chunks: local latency exceeds the configured RTC capacity"
                        )
                    self._condition.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._fault("inference", exc)

    def _control_loop(self) -> None:
        period = 1 / (self.config.fps * self.config.interpolation_multiplier)
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self._condition:
                    if self._state != State.RUNNING:
                        previous_state = self._state
                        timeout = (
                            max(0.0, self._started_at + self.config.startup_timeout_s - time.monotonic())
                            if previous_state == State.STARTING
                            else None
                        )
                        self._condition.wait_for(
                            lambda previous_state=previous_state: (
                                self.stop_event.is_set() or self._state != previous_state
                            ),
                            timeout=timeout,
                        )
                        if (
                            self._state == State.STARTING
                            and time.monotonic() - self._started_at > self.config.startup_timeout_s
                        ):
                            raise TimeoutError("Local startup timed out")
                        deadline = time.monotonic()
                        continue
                if self.stop_event.wait(max(0.0, deadline - time.monotonic())):
                    return
                with self._command_lock:
                    now = time.monotonic()
                    with self._condition:
                        if self._state != State.RUNNING or self.stop_event.is_set():
                            continue
                        if (
                            self._pending is not None
                            and self._pending.generation == self._generation
                            and now - self._pending.started_at > self.config.inference_timeout_s
                        ):
                            raise TimeoutError("Local inference worker missed its deadline")
                        command = self.schedule.command()
                        if command is None:
                            if self._has_dispatched:
                                raise RuntimeError(
                                    "Action buffer exhausted; no stale trajectory will be replayed"
                                )
                            if now > self._first_action_deadline:
                                raise TimeoutError("No fresh initial action arrived before the deadline")
                        elif now - command.source_at > self.config.action_max_age_s:
                            raise TimeoutError("Action source observation exceeded action_max_age_s")
                    if command is not None:
                        applied = self.io.send(
                            command.values, deadline=command.source_at + self.config.action_max_age_s
                        )
                        dispatched_at = time.monotonic()
                        with self._condition:
                            self.schedule.acknowledge(command, applied)
                            self._has_dispatched = True
                            self.timings.add("control_lateness", now - deadline)
                            self.timings.add("command_dispatch", dispatched_at - now)
                            self.timings.add("action_source_age", dispatched_at - command.source_at)
                            if self._last_dispatched_source != command.source_at:
                                self.timings.add(
                                    "observation_to_first_action", dispatched_at - command.source_at
                                )
                                self._last_dispatched_source = command.source_at
                            self._condition.notify_all()
                deadline += period
                if deadline < time.monotonic():
                    deadline = time.monotonic() + period  # Never burst to catch up.
        except Exception as exc:
            if not self.stop_event.is_set():
                self._fault("control", exc)
