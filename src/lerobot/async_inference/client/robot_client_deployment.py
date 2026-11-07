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

"""Canonical deployable remote-RTC robot client.

This module contains transport, action interpolation, generation fencing, and
rollout reset, plus the guided runtime's adaptive scheduling.
Trained RTC delegates prefix transport, scheduling, and admission to rtc_trained.
Neither path imports historical robot_client_rtc* implementations.

Example:

.. code-block:: shell

   uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment \
       --config_path=src/lerobot/async_inference/config/robot_client_deployment.json

Individual CLI arguments override values from the configuration file. The original
CLI-only form is also supported:

.. code-block:: shell

   python -m lerobot.async_inference.client.robot_client_deployment \
       --robot.type=piper_follower --server_address=10.0.0.2:8080 \
       --policy_type=smolvla --pretrained_name_or_path=user/model \
       --actions_per_chunk=50 --fps=30 --interpolation_multiplier=2 \
       --task="pick the cube"
"""

import copy
import logging
import math
import os
import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any
from uuid import uuid4

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

try:
    import pyrealsense2  # noqa: F401
except ImportError as exc:
    logging.warning("realsense camera type unavailable: pyrealsense2 failed to import (%s)", exc)
else:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
    unitree_g1,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

from ..shared.configs import RobotClientConfigBase
from ..shared.deployment_diagnostics import configure_deployment_logging
from ..shared.helpers import (
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from ..shared.image_transport import (
    IMAGE_TRANSPORT_MODES,
    ImageResizeSpec,
    encode_observation_images,
    parse_image_transport_metadata,
)
from ..shared.rlt_config import RLTClientConfig
from ..shared.rtc_remote import (
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    ROLLOUT_SEQUENCE_LIMIT,
    RemoteRTCActionChunk,
    RemoteRTCCalibrationRequest,
    RemoteRTCCalibrationResult,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
    RemoteRTCRetry,
    RTCActionRef,
    decode_rollout_generation,
    decode_rollout_sequence,
    encode_rollout_observation_id,
)
from ..shared.rtc_trained import TrainedRTCClientConfig, TrainedRTCClientRuntime

_THREAD_JOIN_TIMEOUT_SECONDS = 3.0
_MAX_QUEUE_HISTORY_SAMPLES = 100_000
_RTT_ESTIMATE_DECAY = 0.2


@dataclass
class _RTCClientConfig(RobotClientConfigBase):
    """Single-request RTC transport and optional diagnostics."""

    rpc_retry_backoff: float = 0.1
    resize_images_on_client: bool = field(
        default=True,
        metadata={"help": "Negotiate Pi05 checkpoint-native image preparation before upload"},
    )
    image_transport_mode: str = field(
        default="lerobot_float32",
        metadata={
            "help": "lerobot_float32: existing resize + lossless compression; "
            "openpi_uint8: PIL resize/pad + raw RGB uint8 (requires updated server)"
        },
    )
    debug_visualize_queue_size: bool = field(
        default=False,
        metadata={"help": "Plot the most recent 100,000 command-tick queue sizes after shutdown"},
    )
    debug_rtc_diagnostics: bool = field(
        default=False,
        metadata={
            "help": "Enable RTC command/tracking/starvation checks, timing, splice diagnostics and DEBUG logs"
        },
    )
    guided_late_action_tolerances: dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Optional tolerance for every robot action feature, in robot action units. "
                "Allows late guided chunks only when their consumed overlap and first live "
                "endpoint agree with the committed plan. Empty retains the +3-step guard."
            )
        },
    )
    request_timeout_s: float = field(
        default=30.0,
        metadata={
            "help": (
                "Maximum capture-to-result request lifetime and per-RPC wait in seconds. "
                "Expired requests are replaced with a fresh observation/prefix. "
                "Set above normal capture, upload, and inference latency, including cold inference."
            )
        },
    )
    rtc_mode: str = "guided"
    trained: TrainedRTCClientConfig = field(default_factory=TrainedRTCClientConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_transport_mode not in IMAGE_TRANSPORT_MODES:
            raise ValueError(f"image_transport_mode must be one of {IMAGE_TRANSPORT_MODES}")
        if (
            self.resize_images_on_client
            and self.image_transport_mode == "openpi_uint8"
            and self.policy_type != "pi05"
        ):
            raise ValueError("openpi_uint8 image transport requires policy_type=pi05")
        if any(
            not math.isfinite(value) or value <= 0 for value in self.guided_late_action_tolerances.values()
        ):
            raise ValueError("guided_late_action_tolerances must contain finite positive values")
        if self.rtc_mode not in {"guided", "trained"}:
            raise ValueError("rtc_mode must be guided or trained")
        if self.rpc_retry_backoff < 0:
            raise ValueError("rpc_retry_backoff must be non-negative")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")


@dataclass(frozen=True)
class _ScheduledAction:
    timed_action: TimedAction
    source_observation_id: int
    raw_action_index: int


@dataclass(frozen=True)
class _ExecutionSnapshot:
    timestamp: float
    executed_action_step: int
    action_refs: tuple[RTCActionRef, ...]
    trained_prefix: torch.Tensor | None = None


def _publish_latest(queue: Queue, value: Any) -> None:
    try:
        queue.put_nowait(value)
        return
    except Full:
        pass
    with suppress(Empty):
        queue.get_nowait()
    with suppress(Full):
        queue.put_nowait(value)


class _RTCClient:
    """Robot client with local execution and asynchronous chunk transport."""

    logger: logging.Logger

    def __init__(self, config: _RTCClientConfig):
        self.config = config
        self._diagnostic_context = threading.local()
        self._trained_runtime = (
            TrainedRTCClientRuntime(config.trained, config.actions_per_chunk, config.client_device)
            if config.rtc_mode == "trained"
            else None
        )
        self.robot = make_robot_from_config(config.robot)
        tolerances = config.guided_late_action_tolerances
        if tolerances and set(tolerances) != set(self.robot.action_features):
            raise ValueError("guided_late_action_tolerances must name every robot action feature exactly")
        self._guided_late_tolerances = (
            torch.tensor([tolerances[key] for key in self.robot.action_features], device=config.client_device)
            if tolerances
            else None
        )
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
        )
        self.policy_setup = RemoteRTCPolicySetup(
            policy=self.policy_config,
            control_frequency_hz=config.fps,
            action_feature_names=tuple(self.robot.action_features),
            protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
            rtc_mode=config.rtc_mode,
            trained_queue_threshold=(
                None if self._trained_runtime is None else self._trained_runtime.queue_threshold
            ),
            client_image_resize=config.resize_images_on_client and config.policy_type == "pi05",
            image_transport_mode=config.image_transport_mode,
        )
        self._image_transport_plan: dict[str, ImageResizeSpec] = {}

        self.server_address = config.server_address
        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        # Startup calibration and the live receiver share the GetActions
        # stream. Hold ownership through setup/calibration on reconnect too.
        self._action_receive_lock = threading.Lock()
        self._startup_in_progress = threading.Event()

        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        # Diagnostic history belongs to the process, not a rollout. Keep it
        # bounded and retain it across pause/reset/stop for the exit plot.
        self.action_queue_size: deque[int] = deque(maxlen=_MAX_QUEUE_HISTORY_SAMPLES)
        self.action_chunk_size = config.actions_per_chunk

        self._transport_lock = threading.Lock()
        self._latest_latency_feedback: tuple[int, float] | None = None
        self._latest_received_chunk_id = -1
        self.start_barrier = threading.Barrier(3)
        self.logger.info("Robot connected; remote RTC server is %s", self.server_address)

    def _log_rtc(self, level: int, message: str, *args: Any) -> None:
        if level == logging.DEBUG and not self.config.debug_rtc_diagnostics:
            return
        if not self.logger.isEnabledFor(level):
            return
        records = getattr(self._diagnostic_context, "records", None)
        if records is None:
            self.logger.log(level, message, *args)
        else:
            records.append((level, message, args))

    @property
    def _timing_diagnostics_enabled(self) -> bool:
        return self.config.debug_rtc_diagnostics or getattr(self.config, "log_rtc_timing", False)

    @contextmanager
    def _defer_rtc_logs(self) -> Iterator[None]:
        """Emit diagnostic records after the outermost dispatch/admission lock exits."""

        if getattr(self._diagnostic_context, "records", None) is not None:
            yield
            return
        records = []
        self._diagnostic_context.records = records
        try:
            yield
        finally:
            del self._diagnostic_context.records
            for level, message, args in records:
                self.logger.log(level, message, *args)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        try:
            self.stub.Ready(services_pb2.Empty())
            setup = services_pb2.PolicySetup(data=pickle.dumps(self.policy_setup))  # nosec
            self._image_transport_plan = {}
            self._send_policy_instructions(setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as exc:
            self.logger.error("Failed to connect to remote RTC server: %s", exc)
            return False

    def _send_policy_instructions(self, setup: services_pb2.PolicySetup) -> None:
        if self.policy_setup.client_image_resize:
            _, call = self.stub.SendPolicyInstructions.with_call(setup)
            self._image_transport_plan = parse_image_transport_metadata(
                call.initial_metadata(), expected_mode=self.policy_setup.image_transport_mode
            )
            if self._image_transport_plan:
                description = (
                    "OpenPI uint8"
                    if self.policy_setup.image_transport_mode == "openpi_uint8"
                    else "lossless float32"
                )
                self.logger.info(
                    "Client image resize enabled (%s): %s", description, self._image_transport_plan
                )
            else:
                self.logger.info("Server did not negotiate client image resizing; using original images")
        else:
            self.stub.SendPolicyInstructions(setup)

    def stop(self) -> None:
        self.shutdown_event.set()
        self.channel.close()
        disconnect_error: Exception | None = None
        try:
            self.robot.disconnect()
        except Exception as exc:
            disconnect_error = exc
            self.logger.exception("Robot disconnect failed")
        if disconnect_error is not None:
            raise disconnect_error

    def send_observation(self, envelope: RemoteRTCObservation) -> bool:
        """Upload one already-triggered observation from the background thread."""

        if not self.running:
            return False
        try:
            timing_enabled = self._timing_diagnostics_enabled
            if timing_enabled:
                image_started = time.monotonic()
                self._on_observation_transport_phase(
                    envelope.observation_id, "serialization_started", image_started
                )
            plan = self._image_transport_plan
            if plan:
                raw, raw_bytes, image_bytes = encode_observation_images(
                    envelope.observation.get_observation(), plan
                )
                envelope = replace(envelope, observation=replace(envelope.observation, observation=raw))
            payload = pickle.dumps(envelope)  # nosec
            if plan and timing_enabled:
                self._log_rtc(
                    logging.INFO if getattr(self.config, "log_rtc_timing", False) else logging.DEBUG,
                    "RTC IMAGE UPLOAD obs=%d raw_image_bytes=%d image_bytes=%d payload_bytes=%d prepare_ms=%.3f",
                    envelope.observation_id,
                    raw_bytes,
                    image_bytes,
                    len(payload),
                    (time.monotonic() - image_started) * 1000,
                )
            if timing_enabled:
                self._on_observation_transport_phase(
                    envelope.observation_id, "serialization_finished", time.monotonic()
                )
            request_iterator = send_bytes_in_chunks(
                payload,
                services_pb2.Observation,
                log_prefix="[RTC CLIENT] Observation",
                silent=True,
            )
            if timing_enabled:
                self._on_observation_transport_phase(
                    envelope.observation_id, "upload_started", time.monotonic()
                )
            try:
                self.stub.SendObservations(request_iterator, timeout=self.config.request_timeout_s)
            finally:
                if timing_enabled:
                    self._on_observation_transport_phase(
                        envelope.observation_id, "upload_finished", time.monotonic()
                    )
            return True
        except grpc.RpcError as exc:
            if self.running:
                self.logger.error("Error sending RTC observation %d: %s", envelope.observation_id, exc)
            return False

    def _move_actions_to_client_device(self, actions: list[TimedAction]) -> list[TimedAction]:
        device = self.config.client_device
        return [
            TimedAction(
                timestamp=action.get_timestamp(),
                timestep=action.get_timestep(),
                action=action.get_action().to(device),
            )
            for action in actions
        ]

    def _chunk_round_trip(self, chunk: RemoteRTCActionChunk | RemoteRTCRetry) -> float:
        return max(0.0, time.time() - chunk.observation_timestamp)

    def _handle_action_chunk(self, chunk: RemoteRTCActionChunk) -> bool:
        """Validate and merge one full chunk received by the background RPC."""

        round_trip = self._chunk_round_trip(chunk)
        with self._transport_lock:
            if chunk.source_observation_id <= self._latest_received_chunk_id:
                self._log_rtc(
                    logging.DEBUG,
                    "Ignoring duplicate/reordered RTC chunk %d (watermark=%d)",
                    chunk.source_observation_id,
                    self._latest_received_chunk_id,
                )
                return False
            self._latest_received_chunk_id = chunk.source_observation_id
            self._latest_latency_feedback = (chunk.source_observation_id, round_trip)
        wall_delay = math.ceil(round_trip / self.config.environment_dt)

        # RTC delay is an action-consumption quantity, not a wall-clock one.
        # During a latency spike the robot may exhaust its schedule and stop;
        # wall time keeps growing in that state but no additional policy action
        # has been consumed.  Use the first endpoint that is still mutable on
        # this client as the exact remote equivalent of ActionQueue.real_delay.
        with self.latest_action_lock:
            executed = self.latest_action
            first_mutable_step = self.action_schedule.first_mutable_step(executed)
        execution_delay = max(0, first_mutable_step - chunk.chunk_start_step)
        actions = self._move_actions_to_client_device(chunk.actions)
        if chunk.conditioned_delay < 0 or chunk.prefix_length < 0 or chunk.chunk_start_step < 0:
            raise ValueError("RTC chunk delay, prefix length and start step must be non-negative")
        if actions and (
            any(
                row.get_timestep() != chunk.chunk_start_step + index
                or row.get_action().shape != (len(self.robot.action_features),)
                for index, row in enumerate(actions)
            )
            or not torch.isfinite(torch.stack([row.get_action() for row in actions])).all()
        ):
            raise ValueError("RTC actions must be contiguous, finite vectors matching robot action features")

        if (
            chunk.rtc_mode == "guided"
            and chunk.prefix_length > 0
            and execution_delay > chunk.conditioned_delay + 3
        ):
            compatible, reason, max_ratio = False, "tolerances_unset", None
            if self._guided_late_tolerances is not None:
                compatible, reason, max_ratio = self.action_schedule.check_late_overlap(
                    actions,
                    chunk_start_step=chunk.chunk_start_step,
                    executed_action_step=executed,
                    tolerances=self._guided_late_tolerances,
                )
            self._log_rtc(
                logging.INFO if compatible else logging.WARNING,
                "RTC GUIDED LATE obs=%d accepted=%s reason=%s max_error_ratio=%s "
                "execution_delay=%d wall_delay=%d conditioned_delay=%d prefix=%d "
                "chunk_start=%d",
                chunk.source_observation_id,
                compatible,
                reason,
                max_ratio,
                execution_delay,
                wall_delay,
                chunk.conditioned_delay,
                chunk.prefix_length,
                chunk.chunk_start_step,
            )

            if not compatible:
                return False

        # Preserve the full returned chunk when handing it to the schedule.
        # Elapsed/frozen rows are rejected by absolute step inside merge(), but
        # remain available there as trajectory anchors.  Crucially, the guided
        # prefix is no longer treated as an old-queue ownership interval: it is
        # the model's transition band and may replace any still-mutable future
        # endpoint, exactly as local RTC slices at the real consumed delay.
        merge_actions = actions
        merge_raw_indices = list(range(len(actions)))
        if self.config.debug_rtc_diagnostics:
            first_incoming_step = next(
                (action.get_timestep() for action in actions if action.get_timestep() >= first_mutable_step),
                None,
            )
            skipped_elapsed = sum(action.get_timestep() < first_mutable_step for action in actions)

            self._log_rtc(
                logging.DEBUG,
                "RTC MERGE LIVE obs=%d "
                "chunk_start=%d conditioned_delay=%d "
                "execution_delay=%d wall_delay=%d executed=%d first_mutable=%d "
                "first_incoming=%s skipped_elapsed=%d merge_actions=%d",
                chunk.source_observation_id,
                chunk.chunk_start_step,
                chunk.conditioned_delay,
                execution_delay,
                wall_delay,
                executed,
                first_mutable_step,
                first_incoming_step,
                skipped_elapsed,
                len(merge_actions),
            )

        stale, inserted, replaced = self.action_schedule.merge(
            merge_actions,
            source_observation_id=chunk.source_observation_id,
            raw_action_indices=merge_raw_indices,
            executed_action_step=executed,
            capture_diagnostics=self.config.debug_rtc_diagnostics and self.logger.isEnabledFor(logging.DEBUG),
        )
        self.action_chunk_size = max(self.action_chunk_size, len(actions))
        self._log_rtc(
            logging.DEBUG,
            "Merged RTC chunk %d: stale=%d inserted=%d replaced=%d RTT=%.1fms",
            chunk.source_observation_id,
            stale,
            inserted,
            replaced,
            round_trip * 1000,
        )
        return inserted + replaced > 0

    def receive_actions(self) -> None:
        """Long-poll for complete chunks on a dedicated background thread."""

        self.start_barrier.wait()
        self.logger.info("RTC action-chunk receiver started")
        while self.running:
            if self._startup_in_progress.is_set():
                self.shutdown_event.wait(0.01)
                continue
            try:
                with self._action_receive_lock:
                    if not self.running or self._startup_in_progress.is_set():
                        continue
                    response = self.stub.GetActions(
                        services_pb2.Empty(), timeout=self.config.request_timeout_s
                    )
                    if not response.data:
                        continue
                    chunk = pickle.loads(response.data)  # nosec
                    if isinstance(chunk, RemoteRTCRetry):
                        self._handle_snapshot_retry(chunk)
                        continue
                    if not isinstance(chunk, RemoteRTCActionChunk):
                        raise TypeError(f"expected RemoteRTCActionChunk, got {type(chunk)}")
                    self._handle_action_chunk(chunk)
            except grpc.RpcError as exc:
                if not self.running:
                    break
                if exc.code() == grpc.StatusCode.INTERNAL:
                    self.logger.error("Remote RTC inference worker failed: %s", exc)
                    self.shutdown_event.set()
                    break
                self.logger.error("Error receiving RTC action chunk: %s", exc)
                time.sleep(self.config.rpc_retry_backoff)
            except Exception as exc:
                self.logger.error("Invalid RTC action chunk: %s", exc, exc_info=True)
                time.sleep(self.config.rpc_retry_backoff)

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {key: action_tensor[index].item() for index, key in enumerate(self.robot.action_features)}

    def _trigger_threshold(self) -> int:
        return max(0, math.floor(self.action_chunk_size * self.config.chunk_size_threshold))

    def _snapshot_execution_state(self) -> _ExecutionSnapshot:
        """Atomically pair the execution watermark with its future RTC prefix."""

        with self.latest_action_lock:
            executed = self.latest_action
            prefix = None
            if self._trained_runtime is not None:
                prefix = self._trained_runtime.schedule.snapshot_prefix(
                    executed, len(self.robot.action_features)
                )
                refs = ()
            else:
                refs = self.action_schedule.snapshot_refs(executed, self.config.actions_per_chunk)
            timestamp = time.time()
        return _ExecutionSnapshot(
            timestamp=timestamp,
            executed_action_step=executed,
            action_refs=refs,
            trained_prefix=prefix,
        )


@dataclass(frozen=True)
class _SpliceDiagnostics:
    """Atomic before/after snapshot of one future-schedule splice."""

    source_observation_id: int
    executed_action_step: int
    inflight_step: int | None
    frozen_through_step: int
    splice_step: int | None
    queue_before: int
    queue_after: int
    stale: int
    inserted: int
    replaced: int
    old_previous: torch.Tensor | None
    old_same_step: torch.Tensor | None
    new_same_step: torch.Tensor | None


@dataclass(frozen=True)
class _CommandDiagnosticsSnapshot:
    """Last command actually handed to ``robot.send_action``."""

    wall_time: float
    monotonic_time: float
    policy_step: int
    source_observation_id: int
    raw_action_index: int
    interpolation_substep: int
    interpolation_substeps: int
    command: torch.Tensor
    delta: torch.Tensor | None
    send_duration_seconds: float


class _InterpolatedActionSchedule:
    """Separate mutable future actions from one immutable in-flight endpoint.

    ``_actions`` is exclusively the LWW future schedule.  Consuming an action
    atomically pops it into ``_inflight``; from that point onward a newer chunk
    cannot mutate it.  The interpolator owns that immutable endpoint until its
    final command is dispatched and :meth:`commit_inflight` advances the
    execution watermark.

    Although in-flight is no longer stored in the mutable schedule, queue-size
    and RTC-prefix snapshots explicitly prepend it.  This preserves the real
    execution semantics: consumed is not the same as executed.
    """

    def __init__(self, *, history_size: int = 0) -> None:
        self._actions: dict[int, _ScheduledAction] = {}
        self._lock = threading.Lock()
        self._inflight: _ScheduledAction | None = None
        self._last_committed: _ScheduledAction | None = None
        self._committed_history: deque[_ScheduledAction] = deque(maxlen=history_size)
        self._committed_step = -1
        self._last_merge_diagnostics: _SpliceDiagnostics | None = None

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._inflight = None
            self._last_committed = None
            self._committed_history.clear()
            self._committed_step = -1
            self._last_merge_diagnostics = None

    def _drop_executed_locked(self, executed_action_step: int) -> None:
        stale_steps = [step for step in self._actions if step <= executed_action_step]
        for step in stale_steps:
            del self._actions[step]

    @staticmethod
    def _action_ref(scheduled: _ScheduledAction) -> RTCActionRef:
        return RTCActionRef(
            scheduled.source_observation_id,
            scheduled.raw_action_index,
        )

    def consume_next(self, executed_action_step: int) -> _ScheduledAction | None:
        """Move the next mutable future endpoint into immutable in-flight."""

        with self._lock:
            if self._inflight is not None:
                inflight_step = self._inflight.timed_action.get_timestep()
                raise RuntimeError(f"policy action {inflight_step} is already in flight")

            executed_action_step = max(executed_action_step, self._committed_step)
            self._drop_executed_locked(executed_action_step)
            next_step = executed_action_step + 1
            scheduled = self._actions.pop(next_step, None)
            if scheduled is not None:
                self._inflight = scheduled
            return scheduled

    def inflight(self) -> _ScheduledAction | None:
        """Return the immutable endpoint currently owned by interpolation."""

        with self._lock:
            return self._inflight

    def commit_inflight(self, expected_step: int) -> None:
        """Commit an immutable endpoint after its final command was dispatched."""

        with self._lock:
            inflight_step = None if self._inflight is None else self._inflight.timed_action.get_timestep()
            if inflight_step != expected_step:
                raise RuntimeError(
                    f"cannot commit policy action {expected_step}; in-flight step is {inflight_step}"
                )
            self._last_committed = self._inflight
            self._committed_history.append(self._inflight)
            self._inflight = None
            self._committed_step = max(self._committed_step, expected_step)

    def first_mutable_step(self, executed_action_step: int) -> int:
        """Return the first step a newly arrived chunk is still allowed to own."""

        with self._lock:
            executed_action_step = max(executed_action_step, self._committed_step)
            inflight_step = -1 if self._inflight is None else self._inflight.timed_action.get_timestep()
            return max(executed_action_step, inflight_step) + 1

    def check_late_overlap(
        self,
        actions: list[TimedAction],
        *,
        chunk_start_step: int,
        executed_action_step: int,
        tolerances: torch.Tensor,
    ) -> tuple[bool, str, float | None]:
        """Check a late guided proposal against endpoints already committed.

        Compare every consumed row, including the immutable interpolation target,
        and the first live row. Never infer agreement from only the final anchor:
        a conflicting path can return to the same anchor. This checks commanded
        trajectory compatibility, not measured state, contact or task validity.
        """

        with self._lock:
            frozen = max(executed_action_step, self._committed_step)
            committed = {row.timed_action.get_timestep(): row for row in self._committed_history}
            if self._inflight is not None:
                frozen = max(frozen, self._inflight.timed_action.get_timestep())
                committed[self._inflight.timed_action.get_timestep()] = self._inflight
            incoming = {row.get_timestep(): row.get_action() for row in actions}
            if frozen + 1 not in incoming:
                return False, "no_live_endpoint", None
            old_rows, new_rows = [], []
            for step in range(chunk_start_step, frozen + 1):
                if step not in committed or step not in incoming:
                    return False, "missing_overlap", None
                old_rows.append(committed[step].timed_action.get_action())
                new_rows.append(incoming[step])
            if not old_rows:
                return False, "missing_overlap", None

            # Bound the seam against the same-step old target when it exists.
            # After starvation, only the last committed target is available;
            # require the restart step itself to fit the configured tolerance.
            old_next = self._actions.get(frozen + 1)
            first_live = incoming[frozen + 1]
            old_rows.append(old_rows[-1] if old_next is None else old_next.timed_action.get_action())
            new_rows.append(first_live)
            if any(row.shape != tolerances.shape for row in old_rows + new_rows):
                return False, "action_shape", None
            old = torch.stack(old_rows)
            new = torch.stack(new_rows)
            if not torch.isfinite(old).all() or not torch.isfinite(new).all():
                return False, "nonfinite_action", None
            max_ratio = float(((new - old).abs() / tolerances).max().item())
            return max_ratio <= 1.0, "compatible" if max_ratio <= 1.0 else "overlap_mismatch", max_ratio

    def last_merge_diagnostics(self, source_observation_id: int) -> _SpliceDiagnostics | None:
        """Return diagnostics only when they belong to the requested merge."""

        with self._lock:
            diagnostics = self._last_merge_diagnostics
            if diagnostics is None or diagnostics.source_observation_id != source_observation_id:
                return None
            return diagnostics

    def qsize(self, executed_action_step: int) -> int:
        """Count the contiguous in-flight-plus-future policy plan."""

        with self._lock:
            plan = self._execution_plan_locked(executed_action_step)
            return len(plan)

    def snapshot_refs(self, executed_action_step: int, max_actions: int) -> tuple[RTCActionRef, ...]:
        """Snapshot in-flight first, followed by contiguous mutable future refs."""

        if max_actions <= 0:
            return ()
        with self._lock:
            plan = self._execution_plan_locked(executed_action_step)
            return tuple(self._action_ref(scheduled) for scheduled in plan[:max_actions])

    def _execution_plan_locked(self, executed_action_step: int) -> list[_ScheduledAction]:
        executed_action_step = max(executed_action_step, self._committed_step)
        self._drop_executed_locked(executed_action_step)
        next_step = executed_action_step + 1
        plan: list[_ScheduledAction] = []

        if self._inflight is not None:
            inflight_step = self._inflight.timed_action.get_timestep()
            if inflight_step != next_step:
                raise RuntimeError(
                    "in-flight policy endpoint is not the next unexecuted step: "
                    f"inflight={inflight_step}, expected={next_step}"
                )
            plan.append(self._inflight)
            next_step += 1

        while next_step in self._actions:
            plan.append(self._actions[next_step])
            next_step += 1
        return plan

    def merge(
        self,
        actions: list[TimedAction],
        *,
        source_observation_id: int,
        raw_action_indices: list[int],
        executed_action_step: int,
        capture_diagnostics: bool = False,
    ) -> tuple[int, int, int]:
        """Merge live guided actions around an immutable execution boundary.

        Rows at or before the currently executed/in-flight endpoint are anchor
        context only and can never re-enter the mutable schedule.  Starting at
        the first still-mutable absolute step, a newer chunk wins pointwise.
        """

        if len(actions) != len(raw_action_indices):
            raise ValueError("each prepared action must have one raw action index")

        stale = inserted = replaced = 0
        with self._lock:
            executed_action_step = max(executed_action_step, self._committed_step)
            self._drop_executed_locked(executed_action_step)
            inflight_step_or_none = (
                None if self._inflight is None else self._inflight.timed_action.get_timestep()
            )
            inflight_step = -1 if inflight_step_or_none is None else inflight_step_or_none
            frozen_through_step = max(executed_action_step, inflight_step)
            self._last_merge_diagnostics = None
            first_incoming = None
            # Locate the first replaced endpoint only for verbose diagnostics.
            if capture_diagnostics:
                eligible = (
                    (action, raw_index)
                    for action, raw_index in zip(actions, raw_action_indices, strict=True)
                    if action.get_timestep() > frozen_through_step
                    and (
                        (existing := self._actions.get(action.get_timestep())) is None
                        or existing.source_observation_id < source_observation_id
                    )
                )
                first_incoming = min(eligible, key=lambda item: item[0].get_timestep(), default=None)
            splice_step = None if first_incoming is None else first_incoming[0].get_timestep()

            def scheduled_at(step: int) -> _ScheduledAction | None:
                if self._inflight is not None and self._inflight.timed_action.get_timestep() == step:
                    return self._inflight
                if (
                    self._last_committed is not None
                    and self._last_committed.timed_action.get_timestep() == step
                ):
                    return self._last_committed
                return self._actions.get(step)

            old_previous_scheduled = None if splice_step is None else scheduled_at(splice_step - 1)
            if capture_diagnostics:
                queue_before = len(self._execution_plan_locked(executed_action_step))
                old_same_scheduled = None if splice_step is None else self._actions.get(splice_step)
                old_previous = (
                    None
                    if old_previous_scheduled is None
                    else old_previous_scheduled.timed_action.get_action().detach().cpu().clone()
                )
                old_same_step = (
                    None
                    if old_same_scheduled is None
                    else old_same_scheduled.timed_action.get_action().detach().cpu().clone()
                )
                new_same_step = (
                    None if first_incoming is None else first_incoming[0].get_action().detach().cpu().clone()
                )

            for action, raw_index in zip(actions, raw_action_indices, strict=True):
                step = action.get_timestep()
                if step <= frozen_through_step:
                    stale += 1
                    continue
                existing = self._actions.get(step)
                if existing is not None and existing.source_observation_id >= source_observation_id:
                    stale += 1
                    continue

                self._actions[step] = _ScheduledAction(
                    timed_action=action,
                    source_observation_id=source_observation_id,
                    raw_action_index=raw_index,
                )
                if existing is None:
                    inserted += 1
                else:
                    replaced += 1
            if capture_diagnostics:
                queue_after = len(self._execution_plan_locked(executed_action_step))
                self._last_merge_diagnostics = _SpliceDiagnostics(
                    source_observation_id=source_observation_id,
                    executed_action_step=executed_action_step,
                    inflight_step=inflight_step_or_none,
                    frozen_through_step=frozen_through_step,
                    splice_step=splice_step,
                    queue_before=queue_before,
                    queue_after=queue_after,
                    stale=stale,
                    inserted=inserted,
                    replaced=replaced,
                    old_previous=old_previous,
                    old_same_step=old_same_step,
                    new_same_step=new_same_step,
                )
        return stale, inserted, replaced


@dataclass
class _InterpolatedClientConfig(_RTCClientConfig):
    """RTC client settings plus official LeRobot action interpolation."""

    interpolation_multiplier: int = field(
        default=1,
        metadata={
            "help": (
                "Robot commands per policy action. Policy/RTC remain at fps; "
                "robot commands run at fps * interpolation_multiplier."
            )
        },
    )
    command_delta_warning_threshold: float = field(
        default=2.0,
        metadata={
            "help": (
                "Log an RTC COMMAND ANOMALY when any robot command component changes by at least "
                "this amount in one local command tick. Set to 0 to disable the delta check."
            )
        },
    )
    command_reversal_warning_threshold: float = field(
        default=0.5,
        metadata={
            "help": (
                "Log a direction reversal when consecutive command deltas on a component have "
                "opposite signs and both exceed this magnitude. Set to 0 to disable."
            )
        },
    )
    command_timing_warning_factor: float = field(
        default=1.5,
        metadata={
            "help": (
                "Log a command timing anomaly when dispatch spacing exceeds this multiple of the "
                "configured local command period."
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.interpolation_multiplier < 1:
            raise ValueError(f"interpolation_multiplier must be >= 1, got {self.interpolation_multiplier}")
        if self.command_delta_warning_threshold < 0:
            raise ValueError("command_delta_warning_threshold must be non-negative")
        if self.command_reversal_warning_threshold < 0:
            raise ValueError("command_reversal_warning_threshold must be non-negative")
        if self.command_timing_warning_factor < 1:
            raise ValueError("command_timing_warning_factor must be >= 1")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "interpolation_multiplier": self.interpolation_multiplier,
            "command_delta_warning_threshold": self.command_delta_warning_threshold,
            "command_reversal_warning_threshold": self.command_reversal_warning_threshold,
            "command_timing_warning_factor": self.command_timing_warning_factor,
        }


class _InterpolatedClient(_RTCClient):
    """RTC client that interpolates locally between scheduled policy actions."""

    def __init__(self, config: _InterpolatedClientConfig):
        super().__init__(config)
        self.action_schedule = (
            _InterpolatedActionSchedule(
                history_size=config.actions_per_chunk if config.guided_late_action_tolerances else 0,
            )
            if self._trained_runtime is None
            else self._trained_runtime.schedule
        )
        self.action_interpolator = ActionInterpolator(multiplier=config.interpolation_multiplier)
        self._command_diagnostics_lock = threading.Lock()
        self._last_command_diagnostics: _CommandDiagnosticsSnapshot | None = None
        self._last_command_anomaly_log_at = float("-inf")
        self._has_interpolation_origin = False
        self._interpolation_substep = 0
        self._interpolation_substeps = 0
        self.logger.info(
            "Action interpolation: policy/RTC=%gHz, robot commands=%gHz (x%d)",
            config.fps,
            config.fps * config.interpolation_multiplier,
            config.interpolation_multiplier,
        )
        self.logger.info("RTC diagnostic action order: %s", list(self.robot.action_features))

    @staticmethod
    def _diagnostic_vector(value: torch.Tensor | None) -> list[float] | None:
        if value is None:
            return None
        return [round(float(component), 5) for component in value.detach().cpu().flatten()]

    @staticmethod
    def _max_abs(value: torch.Tensor | None) -> float | None:
        if value is None or value.numel() == 0:
            return None
        return float(value.detach().abs().max().cpu())

    def _clear_command_diagnostics(self) -> None:
        with self._command_diagnostics_lock:
            self._last_command_diagnostics = None
            self._last_command_anomaly_log_at = float("-inf")
            self._has_interpolation_origin = False
            self._interpolation_substep = 0
            self._interpolation_substeps = 0

    def command_diagnostics_snapshot(self) -> _CommandDiagnosticsSnapshot | None:
        if not self._command_snapshot_required():
            return None
        with self._command_diagnostics_lock:
            snapshot = self._last_command_diagnostics
            if snapshot is None:
                return None
            return _CommandDiagnosticsSnapshot(
                wall_time=snapshot.wall_time,
                monotonic_time=snapshot.monotonic_time,
                policy_step=snapshot.policy_step,
                source_observation_id=snapshot.source_observation_id,
                raw_action_index=snapshot.raw_action_index,
                interpolation_substep=snapshot.interpolation_substep,
                interpolation_substeps=snapshot.interpolation_substeps,
                command=snapshot.command.clone(),
                delta=None if snapshot.delta is None else snapshot.delta.clone(),
                send_duration_seconds=snapshot.send_duration_seconds,
            )

    def _handle_action_chunk(self, chunk: RemoteRTCActionChunk) -> bool:
        """Merge a chunk, then emit one atomic old/new splice diagnostic."""

        merged = super()._handle_action_chunk(chunk)
        if not self.config.debug_rtc_diagnostics or not self.logger.isEnabledFor(logging.DEBUG):
            return merged
        diagnostics = self.action_schedule.last_merge_diagnostics(chunk.source_observation_id)
        if diagnostics is None:
            return merged

        command_jump = None
        replacement_delta = None
        slope_change = None
        if diagnostics.old_previous is not None and diagnostics.new_same_step is not None:
            command_jump = diagnostics.new_same_step - diagnostics.old_previous
        if diagnostics.old_same_step is not None and diagnostics.new_same_step is not None:
            replacement_delta = diagnostics.new_same_step - diagnostics.old_same_step
        if (
            diagnostics.old_previous is not None
            and diagnostics.old_same_step is not None
            and diagnostics.new_same_step is not None
        ):
            old_slope = diagnostics.old_same_step - diagnostics.old_previous
            scheduled_slope = diagnostics.new_same_step - diagnostics.old_previous
            slope_change = scheduled_slope - old_slope

        self._log_rtc(
            logging.DEBUG,
            "RTC SPLICE DIAG obs=%d executed=%d inflight=%s frozen_through=%d splice=%s "
            "queue=%d->%d stale=%d inserted=%d replaced=%d "
            "command_jump_max=%s "
            "replacement_delta_max=%s slope_change_max=%s "
            "old_prev=%s old_same=%s new_same=%s",
            diagnostics.source_observation_id,
            diagnostics.executed_action_step,
            diagnostics.inflight_step,
            diagnostics.frozen_through_step,
            diagnostics.splice_step,
            diagnostics.queue_before,
            diagnostics.queue_after,
            diagnostics.stale,
            diagnostics.inserted,
            diagnostics.replaced,
            self._max_abs(command_jump),
            self._max_abs(replacement_delta),
            self._max_abs(slope_change),
            self._diagnostic_vector(diagnostics.old_previous),
            self._diagnostic_vector(diagnostics.old_same_step),
            self._diagnostic_vector(diagnostics.new_same_step),
        )
        return merged

    def _command_snapshot_required(self) -> bool:
        """Record commands only for diagnostics; recording clients may override."""
        return self.config.debug_rtc_diagnostics

    def _record_command_dispatch(
        self,
        scheduled,
        requested: torch.Tensor,
        performed: dict[str, Any],
        dispatch_started: float,
        dispatch_completed: float,
    ) -> None:
        diagnostics = self.config.debug_rtc_diagnostics
        previous_command = None
        if diagnostics:
            with self._command_diagnostics_lock:
                previous_command = self._last_command_diagnostics
        try:
            command_cpu = torch.tensor(
                [float(performed[key]) for key in self.robot.action_features], dtype=requested.dtype
            )
        except (KeyError, TypeError, ValueError):
            command_cpu = requested.detach().cpu().clone()
        command_delta = None if previous_command is None else command_cpu - previous_command.command
        send_duration = max(0.0, dispatch_completed - dispatch_started) if diagnostics else 0.0
        target_step = scheduled.timed_action.get_timestep()
        interpolation_substep = self._interpolation_substep
        interpolation_substeps = self._interpolation_substeps
        snapshot = _CommandDiagnosticsSnapshot(
            wall_time=time.time(),
            monotonic_time=dispatch_completed,
            policy_step=target_step,
            source_observation_id=scheduled.source_observation_id,
            raw_action_index=scheduled.raw_action_index,
            interpolation_substep=interpolation_substep,
            interpolation_substeps=interpolation_substeps,
            command=command_cpu,
            delta=command_delta,
            send_duration_seconds=send_duration,
        )
        with self._command_diagnostics_lock:
            self._last_command_diagnostics = snapshot
        if not diagnostics:
            # RECAP consumes command/time/source metadata for its dataset; no
            # command differences or anomaly checks are needed for recording.
            return
        send_interval = (
            None if previous_command is None else max(0.0, dispatch_started - previous_command.monotonic_time)
        )
        requested_to_sent = command_cpu - requested.detach().cpu()
        max_delta = self._max_abs(command_delta)
        expected_period = self.config.environment_dt / self.action_interpolator.multiplier
        timing_limit = expected_period * self.config.command_timing_warning_factor
        timing_anomaly = send_interval is not None and send_interval > timing_limit
        send_duration_anomaly = send_duration > timing_limit
        delta_anomaly = (
            max_delta is not None
            and self.config.command_delta_warning_threshold > 0
            and max_delta >= self.config.command_delta_warning_threshold
        )
        reversal_indices: list[int] = []
        reversal_threshold = self.config.command_reversal_warning_threshold
        if (
            reversal_threshold > 0
            and command_delta is not None
            and previous_command is not None
            and previous_command.delta is not None
            and command_delta.shape == previous_command.delta.shape
        ):
            reversed_components = (
                (command_delta * previous_command.delta < 0)
                & (command_delta.abs() >= reversal_threshold)
                & (previous_command.delta.abs() >= reversal_threshold)
            )
            reversal_indices = reversed_components.nonzero(as_tuple=False).flatten().tolist()

        has_anomaly = timing_anomaly or send_duration_anomaly or delta_anomaly or bool(reversal_indices)
        source_changed = (
            previous_command is not None
            and previous_command.source_observation_id != scheduled.source_observation_id
        )
        with self._command_diagnostics_lock:
            should_log_anomaly = has_anomaly and (
                source_changed or dispatch_completed - self._last_command_anomaly_log_at >= 0.1
            )
            if should_log_anomaly:
                self._last_command_anomaly_log_at = dispatch_completed

        if should_log_anomaly:
            second_difference = (
                None
                if command_delta is None
                or previous_command is None
                or previous_command.delta is None
                or command_delta.shape != previous_command.delta.shape
                else command_delta - previous_command.delta
            )
            self._log_rtc(
                logging.WARNING,
                "RTC COMMAND ANOMALY policy_step=%d source=%d raw_index=%d interp=%d/%d "
                "endpoint=%s send_dt_ms=%s send_duration_ms=%.3f max_delta=%s max_second_diff=%s "
                "requested_to_sent_max=%s reversal_indices=%s command=%s delta=%s",
                target_step,
                scheduled.source_observation_id,
                scheduled.raw_action_index,
                interpolation_substep,
                interpolation_substeps,
                self.action_interpolator.emitted_policy_action,
                None if send_interval is None else round(send_interval * 1000, 3),
                send_duration * 1000,
                max_delta,
                self._max_abs(second_difference),
                self._max_abs(requested_to_sent),
                reversal_indices,
                self._diagnostic_vector(command_cpu),
                self._diagnostic_vector(command_delta),
            )

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Send one command tick and commit progress only at policy endpoints."""

        if self.action_interpolator.needs_new_action():
            with self.latest_action_lock:
                latest = self.latest_action
                scheduled = self.action_schedule.consume_next(latest)
            if scheduled is None:
                return None

            policy_action = scheduled.timed_action
            action = policy_action.get_action().detach().cpu()
            self._interpolation_substep = 0
            self._interpolation_substeps = (
                self.action_interpolator.multiplier if self._has_interpolation_origin else 1
            )
            self.action_interpolator.add(action)
            self._has_interpolation_origin = True
        else:
            scheduled = self.action_schedule.inflight()
            if scheduled is None:
                raise RuntimeError(
                    "action interpolator has buffered commands without an in-flight policy endpoint"
                )

        interpolated = self.action_interpolator.get()
        if interpolated is None:
            return None

        action_dict = self._action_tensor_to_action_dict(interpolated)
        capture_command = self._command_snapshot_required()
        dispatch_started = time.monotonic() if self.config.debug_rtc_diagnostics else 0.0
        performed = self.robot.send_action(action_dict)
        self._interpolation_substep += 1
        target_step = scheduled.timed_action.get_timestep()
        if capture_command:
            self._record_command_dispatch(
                scheduled, interpolated, performed, dispatch_started, time.monotonic()
            )

        if self.action_interpolator.emitted_policy_action:
            # Keep the execution watermark and schedule commit atomic with
            # respect to receiver/request threads. DRTC defines this watermark
            # as the last endpoint command actually dispatched to the robot.
            with self.latest_action_lock:
                self.action_schedule.commit_inflight(target_step)
                self.latest_action = target_step

        if verbose and self.config.debug_rtc_diagnostics:
            with self.latest_action_lock:
                latest = self.latest_action
            remaining = self.action_schedule.qsize(latest)
            command_kind = (
                "policy endpoint" if self.action_interpolator.emitted_policy_action else "interpolated"
            )
            self._log_rtc(
                logging.DEBUG,
                "Sent %s command toward policy action %d (executed=%d); %d scheduled actions remain",
                command_kind,
                target_step,
                latest,
                remaining,
            )
        return performed


class RolloutState(StrEnum):
    """Logical state of the reusable client process."""

    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    RESETTING = "resetting"
    RESET_FAILED = "reset_failed"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class _SessionObservationRequest:
    observation_id: int
    generation: int
    task: str
    triggered_action_step: int


@dataclass(frozen=True)
class _CapturedObservation:
    """Owned sensor values only; the action prefix must be read at request time."""

    generation: int
    sequence: int
    observation: RawObservation
    started_at: float
    finished_at: float
    started_action_step: int
    oldest_camera_at: float | None = None

    def age(self, now: float) -> float:
        oldest = self.started_at
        if self.oldest_camera_at is not None:
            oldest = min(oldest, self.oldest_camera_at)
        return max(0.0, now - oldest)


@dataclass
class _RequestTiming:
    """Monotonic timestamps for one end-to-end inference request."""

    requested_at: float
    triggered_action_step: int
    queued_actions: int
    mailbox_dequeued_at: float | None = None
    observation_selected_at: float | None = None
    capture_started_at: float | None = None
    capture_finished_at: float | None = None
    snapshot_at: float | None = None
    serialization_started_at: float | None = None
    serialization_finished_at: float | None = None
    upload_started_at: float | None = None
    upload_finished_at: float | None = None


@dataclass
class RobotClientDeploymentConfig(_InterpolatedClientConfig):
    """Interpolated RTC settings plus reusable rollout reset."""

    rlt: RLTClientConfig = field(default_factory=RLTClientConfig)

    disable_proxy: bool = field(
        default=False,
        metadata={"help": "Clear HTTP/HTTPS/ALL/gRPC proxy variables in this process before connecting"},
    )
    startup_calibration_runs: int = field(
        default=3,
        metadata={"help": "Pi05 measured startup round trips after two excluded dry warmups; 0 disables"},
    )
    startup_calibration_timeout_s: float = field(
        default=600.0,
        metadata={"help": "Maximum wait per startup dry run, allowing first-use CUDA compilation"},
    )
    latency_history_ttl_s: float = field(
        default=300.0,
        metadata={"help": "Keep measured refill estimates across rollout reset for this many idle seconds"},
    )

    continuous_observation: bool = field(
        default=True,
        metadata={"help": "Continuously collect sensors into a latest-only cache during autonomous rollout"},
    )
    observation_capture_fps: float | None = field(
        default=None,
        metadata={
            "help": "Background observation rate; defaults to policy fps, independently of RPC latency"
        },
    )
    observation_max_age_s: float = field(
        default=0.1,
        metadata={"help": "Maximum cached sensor age at request selection; wait for fresh data when older"},
    )
    adaptive_rtt_trigger: bool = field(
        default=True,
        metadata={
            "help": (
                "Adapt the action-queue refill threshold from measured request-to-chunk latency. "
                "Besides retaining enough actions for the next RTT, keep the single-inference "
                "pipeline continuously occupied once the previous result returns. The configured "
                "chunk_size_threshold remains the minimum."
            )
        },
    )
    rtt_trigger_margin_steps: int = field(
        default=5,
        metadata={"help": ("Additional policy steps kept beyond the adaptive request-to-chunk RTT estimate")},
    )
    log_rtc_timing: bool = field(
        default=False,
        metadata={"help": "Collect and log per-request phase timings and image upload sizes at INFO"},
    )
    tracking_error_warning_threshold: float = field(
        default=2.0,
        metadata={
            "help": (
                "With debug_rtc_diagnostics, log observation/command tracking as a warning when a feature differs "
                "from the latest dispatched command by at least this amount. Set to 0 to keep "
                "tracking diagnostics at debug level only."
            )
        },
    )
    suppress_can_debug_logs: bool = field(
        default=True,
        metadata={
            "help": (
                "Raise python-can loggers to WARNING so raw CAN frames do not bury structured RTC "
                "splice, starvation, command, and timing diagnostics."
            )
        },
    )

    reset_pose: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit safe reset pose keyed by robot action feature. "
                "When omitted, reset returns to the pose captured at client startup. "
                "Gripper positions are overridden when reset_open_gripper is enabled "
                "and reset_gripper_open_position is available."
            )
        },
    )
    reset_to_initial_position: bool = field(
        default=True,
        metadata={
            "help": (
                "Return to the startup pose on /reset. If false and reset_pose is unset, "
                "hold the current joint pose and verify measured position stability."
            )
        },
    )
    reset_open_gripper: bool = field(
        default=True,
        metadata={
            "help": (
                "Hold the measured gripper opening while returning the arm, then open it "
                "after arrival is verified. Requires reset_gripper_open_position "
                "(defaults to 0.07 m for Piper). False restores the reset pose's gripper position."
            )
        },
    )
    reset_gripper_open_position: float | None = field(
        default=None,
        metadata={
            "help": (
                "Gripper opening after arm reset, in robot action units. "
                "None selects 0.07 m for Piper; other robots keep their reset-pose "
                "gripper position unless an opening is explicitly configured."
            )
        },
    )
    reset_duration_s: float = field(
        default=3.0,
        metadata={"help": "Seconds used to interpolate smoothly to the reset pose"},
    )
    reset_fps: int = field(
        default=50,
        metadata={"help": "Command frequency for the reset-pose interpolation"},
    )
    reset_verify_timeout_s: float = field(
        default=5.0, metadata={"help": "Maximum wait for measured reset-pose convergence after interpolation"}
    )
    reset_settle_s: float = field(
        default=0.1, metadata={"help": "Time measured positions must remain within the reset tolerances"}
    )
    reset_joint_tolerance: float = field(
        default=0.02, metadata={"help": "Reset tolerance in robot action units (radians for Piper joints)"}
    )
    reset_gripper_tolerance: float = field(
        default=0.001,
        metadata={
            "help": (
                "Reset gripper tolerance in robot action units (meters for Piper). "
                "With gripper release enabled, checked after opening, not while holding during arm return."
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rlt.enabled:
            if self.policy_type != "pi05" or self.rtc_mode != "guided":
                raise ValueError("RLT deployment currently requires guided PI05")
            if self.rlt.budget_margin_s >= self.rlt.prefix_steps / self.fps:
                raise ValueError("RLT timing margin must be smaller than the committed prefix duration")
            if self.actions_per_chunk < self.rlt.execution_horizon + self.rlt.prefix_steps:
                raise ValueError("RLT actions_per_chunk must cover the window and its continuation prefix")
        if self.startup_calibration_runs < 0:
            raise ValueError("startup_calibration_runs must be non-negative")
        for name in ("startup_calibration_timeout_s", "latency_history_ttl_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.observation_capture_fps is not None and (
            not math.isfinite(self.observation_capture_fps) or self.observation_capture_fps <= 0
        ):
            raise ValueError("observation_capture_fps must be finite and positive")
        if not math.isfinite(self.observation_max_age_s) or self.observation_max_age_s <= 0:
            raise ValueError("observation_max_age_s must be finite and positive")
        if self.rtt_trigger_margin_steps < 0:
            raise ValueError(
                f"rtt_trigger_margin_steps must be non-negative, got {self.rtt_trigger_margin_steps}"
            )
        if self.tracking_error_warning_threshold < 0:
            raise ValueError("tracking_error_warning_threshold must be non-negative")
        if self.reset_gripper_open_position is None and self.robot.type == "piper_follower":
            self.reset_gripper_open_position = 0.07
        if self.reset_gripper_open_position is not None:
            if not math.isfinite(self.reset_gripper_open_position):
                raise ValueError("reset_gripper_open_position must be finite")
            if self.robot.type == "piper_follower" and not 0 < self.reset_gripper_open_position <= 0.1:
                raise ValueError("Piper reset_gripper_open_position must be in (0, 0.1] meters")
        if self.reset_duration_s < 0:
            raise ValueError(f"reset_duration_s must be non-negative, got {self.reset_duration_s}")
        if self.reset_fps <= 0:
            raise ValueError(f"reset_fps must be positive, got {self.reset_fps}")
        for name in ("reset_verify_timeout_s", "reset_joint_tolerance", "reset_gripper_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.reset_settle_s)
            or not 0 <= self.reset_settle_s < self.reset_verify_timeout_s
        ):
            raise ValueError("reset_settle_s must be finite, non-negative and below reset_verify_timeout_s")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "disable_proxy": self.disable_proxy,
            "startup_calibration_runs": self.startup_calibration_runs,
            "startup_calibration_timeout_s": self.startup_calibration_timeout_s,
            "latency_history_ttl_s": self.latency_history_ttl_s,
            "rtc_mode": self.rtc_mode,
            "trained": asdict(self.trained),
            "request_timeout_s": self.request_timeout_s,
            "resize_images_on_client": self.resize_images_on_client,
            "image_transport_mode": self.image_transport_mode,
            "debug_rtc_diagnostics": self.debug_rtc_diagnostics,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "adaptive_rtt_trigger": self.adaptive_rtt_trigger,
            "continuous_observation": self.continuous_observation,
            "observation_capture_fps": self.observation_capture_fps,
            "observation_max_age_s": self.observation_max_age_s,
            "rtt_trigger_margin_steps": self.rtt_trigger_margin_steps,
            "guided_late_action_tolerances": self.guided_late_action_tolerances,
            "log_rtc_timing": self.log_rtc_timing,
            "tracking_error_warning_threshold": self.tracking_error_warning_threshold,
            "suppress_can_debug_logs": self.suppress_can_debug_logs,
            "reset_pose": self.reset_pose,
            "reset_to_initial_position": self.reset_to_initial_position,
            "reset_open_gripper": self.reset_open_gripper,
            "reset_gripper_open_position": self.reset_gripper_open_position,
            "reset_duration_s": self.reset_duration_s,
            "reset_fps": self.reset_fps,
            "reset_verify_timeout_s": self.reset_verify_timeout_s,
            "reset_settle_s": self.reset_settle_s,
            "reset_joint_tolerance": self.reset_joint_tolerance,
            "reset_gripper_tolerance": self.reset_gripper_tolerance,
        }


class RobotClientDeployment(_InterpolatedClient):
    """Remote RTC client whose process and transports survive rollout reset."""

    prefix = "robot_client_deployment"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientDeploymentConfig):
        self._rlt_runtime = None
        rlt_config = getattr(config, "rlt", None)
        rlt_reward = None
        if rlt_config is not None and rlt_config.enabled:
            if self.prefix != "robot_client_deployment":
                raise ValueError("RLT currently supports robot_client_deployment only")
            from .rlt_runtime import load_reward_provider

            if config.policy_type != "pi05" or config.rtc_mode != "guided":
                raise ValueError("RLT deployment currently requires guided PI05")
            rlt_reward = load_reward_provider(rlt_config.reward_provider)
        if config.disable_proxy:
            for name in ("http_proxy", "https_proxy", "all_proxy", "grpc_proxy"):
                os.environ.pop(name, None)
                os.environ.pop(name.upper(), None)
        super().__init__(config)
        if config.suppress_can_debug_logs:
            logging.getLogger("can").setLevel(logging.WARNING)

        # This lock is the local linearization point for start/reset, action
        # dispatch, observation snapshots, and action-chunk admission.
        self._lifecycle_lock = threading.RLock()
        self._lifecycle_changed = threading.Condition(self._lifecycle_lock)
        self._reset_operation_lock = threading.Lock()
        self._reset_in_progress = threading.Event()
        self._reset_error: str | None = None
        self._robot_command_lock = threading.Lock()
        self._observation_capture_lock = threading.Lock()
        self._rollout_state = RolloutState.READY
        self._worker_error: str | None = None
        self._rollout_generation = 0
        self._request_sequence = 0
        self._initial_task = config.task
        self._task = config.task

        # Reserve inference at request-publication time, rather than only after
        # the observation sender has captured and uploaded an observation.  The
        # slot spans mailbox/fresh-observation wait, upload, server inference, and
        # action-chunk delivery.  This is what makes the gate genuinely
        # single-in-flight even when the local action schedule is empty.
        self._inflight_request_id: int | None = None
        self._inflight_requested_at: float | None = None
        # Request ids increase within a rollout. Fence abandoned captures,
        # uploads and results even when a replacement uses the same generation.
        self._retired_request_id = -1
        self._force_fresh_snapshot = False

        # End-to-end request latency is deliberately separate from the RTC
        # observation-to-chunk RTT reported to the server.  Trigger scheduling
        # must also budget mailbox wait and camera capture, because actions keep
        # executing during both.  Slow samples raise the estimate immediately;
        # faster samples decay it gradually to avoid threshold oscillation.
        self._adaptive_rtt_steps: float | None = None
        self._latest_request_rtt_seconds: float | None = None
        self._rtt_updated_at: float | None = None
        self._consecutive_rtt_capacity_overruns = 0
        self._rtt_capacity_warning_emitted = False

        self._request_timings: dict[int, _RequestTiming] = {}

        # Starvation is tracked at local command frequency.  It begins only
        # when interpolation has no buffered command and the next immutable
        # policy endpoint cannot be consumed, not merely when mutable qsize is
        # momentarily zero while an endpoint is still in flight.
        self._starvation_started_at: float | None = None
        self._starvation_started_executed = -1
        self._starvation_missed_command_ticks = 0
        self._starvation_request_id: int | None = None

        self._request_mailbox: Queue[_SessionObservationRequest] = Queue(maxsize=1)
        self._observation_collector: threading.Thread | None = None
        self._latest_observation: _CapturedObservation | None = None
        self._observation_sequence = 0
        self._last_used_observation_sequence = -1
        self._observation_valid_after = time.monotonic()
        self._timestamped_observation_supported = callable(
            getattr(self.robot, "get_observation_with_timestamps", None)
        )

        self._configured_reset_pose = self._validate_reset_pose(config.reset_pose)
        self._initial_pose: dict[str, float] | None = None
        if self._configured_reset_pose is None and config.reset_to_initial_position:
            try:
                self._initial_pose = self._read_action_pose()
                self.logger.info("Captured startup reset pose (%d joints)", len(self._initial_pose))
            except Exception as exc:
                self._log_rtc(
                    logging.WARNING,
                    "Could not capture startup pose; configure reset_pose before using /reset: %s",
                    exc,
                )

        if rlt_config is not None and rlt_config.enabled:
            from .rlt_runtime import RemoteRLTClient

            self._rlt_runtime = RemoteRLTClient(self, rlt_reward)

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    @property
    def rollout_state(self) -> RolloutState:
        with self._lifecycle_lock:
            return self._rollout_state

    @property
    def rollout_generation(self) -> int:
        with self._lifecycle_lock:
            return self._rollout_generation

    @property
    def task(self) -> str:
        with self._lifecycle_lock:
            return self._task

    @property
    def worker_error(self) -> str | None:
        with self._lifecycle_lock:
            return self._worker_error

    @property
    def reset_error(self) -> str | None:
        with self._lifecycle_lock:
            return self._reset_error

    def set_task(self, task: str) -> bool:
        """Set the instruction used by subsequently triggered observations."""

        with self._lifecycle_lock:
            if self._rollout_state is RolloutState.STOPPED:
                return False
            changed = task != self._task
            if (
                changed
                and getattr(self, "_rlt_runtime", None) is not None
                and self._rollout_state is RolloutState.RUNNING
            ):
                return False
            self._task = task
            return changed

    def start(self) -> bool:
        """Finish dry calibration before any execution/recording threads start."""
        if getattr(self, "_rlt_runtime", None) is not None:
            return self._rlt_runtime.start()
        with self._lifecycle_lock:
            if self._rollout_state is not RolloutState.READY or self._startup_in_progress.is_set():
                return False
            self._startup_in_progress.set()
            self._rollout_state = RolloutState.STARTING
            with self._transport_lock:
                self._rtt_updated_at = None
                self._expire_rtt_estimate_locked()
        enabled = self.config.policy_type == "pi05" and self.config.startup_calibration_runs > 0
        self.policy_setup = replace(self.policy_setup, calibration_token=uuid4().hex if enabled else None)
        try:
            # Release the lifecycle lock before waiting: an outstanding live
            # poll may need it to finish admitting its previous response.
            with self._action_receive_lock:
                with self._lifecycle_lock:
                    if self._rollout_state is not RolloutState.STARTING or not self.running:
                        raise RuntimeError("policy startup was cancelled")
                if not super().start():
                    raise RuntimeError("remote policy setup failed")
                with self._lifecycle_lock:
                    if self._rollout_state is not RolloutState.STARTING:
                        # The base handshake clears shutdown_event on success. A
                        # concurrent stop must not be undone by a late RPC return.
                        self.shutdown_event.set()
                        raise RuntimeError("policy startup was cancelled")
                if enabled:
                    self._calibrate_startup()
                with self._lifecycle_lock:
                    if self._rollout_state is not RolloutState.STARTING or not self.running:
                        raise RuntimeError("policy startup was cancelled")
                    self._rollout_state = RolloutState.READY
            self.logger.info("Policy ready; startup calibration complete%s", "" if enabled else " (disabled)")
            return True
        except Exception as exc:
            with self._transport_lock:
                self._rtt_updated_at = None
            self._fail_worker("startup calibration", exc)
            return False
        finally:
            self._startup_in_progress.clear()

    def _calibration_round_trip(self, request: RemoteRTCCalibrationRequest) -> RemoteRTCCalibrationResult:
        deadline = time.monotonic() + self.config.startup_calibration_timeout_s
        if request.observation is not None and self._image_transport_plan:
            raw, _, _ = encode_observation_images(
                request.observation.get_observation(), self._image_transport_plan
            )
            request = replace(request, observation=replace(request.observation, observation=raw))
        payload = pickle.dumps(request)  # nosec

        def poll():
            return self.stub.GetActions.future(
                services_pb2.Empty(),
                timeout=min(self.config.request_timeout_s, max(0.001, deadline - time.monotonic())),
            )

        # Mirror the live background receiver: poll before uploading, so upload
        # acknowledgement does not add an extra serial network round trip.
        pending = poll()
        try:
            self.stub.SendObservations(
                send_bytes_in_chunks(payload, services_pb2.Observation, silent=True),
                timeout=max(0.001, deadline - time.monotonic()),
            )
            while self.running and time.monotonic() < deadline:
                try:
                    response = pending.result()
                except grpc.RpcError as exc:
                    if exc.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                        raise
                else:
                    if response.data:
                        result = pickle.loads(response.data)  # nosec
                        if (
                            not isinstance(result, RemoteRTCCalibrationResult)
                            or result.token != request.token
                            or result.sequence != request.sequence
                            or (result.chunk is None) != (request.observation is None)
                        ):
                            raise ValueError("unexpected startup calibration response")
                        return result
                pending = poll()
            raise TimeoutError("startup calibration timed out or was stopped")
        finally:
            pending.cancel()

    def _calibrate_startup(self) -> None:
        # No send_observation/action handler, action schedule, or recorder is
        # involved here. The explicit wrapper cannot be mistaken for live actions.
        feedback: tuple[int, float] | None = None
        token = self.policy_setup.calibration_token
        for sequence in range(self.config.startup_calibration_runs + 2):
            if not self.running:
                raise RuntimeError("startup calibration cancelled")
            requested_at = time.monotonic()
            with self._observation_capture_lock:
                raw = copy.deepcopy(self.robot.get_observation())
            if not self.running:
                raise RuntimeError("startup calibration cancelled")
            raw["task"] = self.task
            snapshot_at = time.monotonic()
            warmup = sequence < 2
            result = self._calibration_round_trip(
                RemoteRTCCalibrationRequest(
                    token=token,
                    sequence=sequence,
                    observation=TimedObservation(time.time(), 0, raw),
                    warmup=warmup,
                    use_prefix=sequence > 0,
                    feedback_source=None if feedback is None else feedback[0],
                    feedback_seconds=None if feedback is None else feedback[1],
                )
            )
            arrived_at = time.monotonic()
            rtt = arrived_at - snapshot_at
            feedback = None if warmup else (sequence, rtt)
            if not warmup and self._trained_runtime is None:
                with self._transport_lock:
                    self._update_rtt_estimate_locked(arrived_at - requested_at, arrived_at)
            self.logger.info(
                "RTC calibration %s %d: round_trip_ms=%.1f server_ms=%.1f",
                "warmup" if warmup else "measure",
                sequence,
                rtt * 1000,
                result.chunk.inference_seconds * 1000,
            )
        self._calibration_round_trip(
            RemoteRTCCalibrationRequest(
                token=token,
                sequence=self.config.startup_calibration_runs + 2,
                feedback_source=feedback[0],
                feedback_seconds=feedback[1],
            )
        )

    def start_rollout(self, task: str | None = None) -> bool:
        """Start a clean rollout while keeping process, RPCs, and threads warm."""

        with self._lifecycle_changed:
            if (
                self._rollout_state is not RolloutState.READY
                or not self.running
                or self._reset_in_progress.is_set()
                or self._startup_in_progress.is_set()
            ):
                return False
            self._advance_generation_locked()
            self._clear_episode_state_locked()
            if task is not None:
                self._task = task
            if self._rlt_runtime is not None:
                self._rlt_runtime.begin_rollout()
            self._rollout_state = RolloutState.RUNNING
            generation = self._rollout_generation
            current_task = self._task
            self._lifecycle_changed.notify_all()

        self.logger.info("Rollout generation %d started (task=%r)", generation, current_task)
        return True

    def reset_rollout(self) -> bool:
        """Fence old work and admit the next rollout only after a verified reset."""

        if not self._reset_operation_lock.acquire(blocking=False):
            return False
        try:
            with self._lifecycle_changed:
                if (
                    self._rollout_state is RolloutState.STOPPED
                    or not self.running
                    or self._startup_in_progress.is_set()
                ):
                    return False
                self._reset_in_progress.set()
                self._reset_error = None
                self._advance_generation_locked()
                invalidation_generation = self._rollout_generation
                self._rollout_state = RolloutState.RESETTING
                self._task = self._initial_task
                self._clear_episode_state_locked()
                self._lifecycle_changed.notify_all()

            try:
                self._prepare_reset()
                reset_ok = self._reset_robot_pose()
            except Exception as exc:
                self.logger.exception("Reset preparation failed")
                self._reset_error = str(exc)
                reset_ok = False

            with self._lifecycle_changed:
                if (
                    self._rollout_state is RolloutState.RESETTING
                    and self._rollout_generation == invalidation_generation
                ):
                    reset_ok = reset_ok and not self._reset_cancelled()
                    self._rollout_state = RolloutState.READY if reset_ok else RolloutState.RESET_FAILED
                    if not reset_ok and self._reset_error is None:
                        self._reset_error = "reset failed or was cancelled before arrival was verified"
                    self._lifecycle_changed.notify_all()
                else:
                    reset_ok = False

            if reset_ok:
                self.logger.info(
                    "Rollout reset complete; client is ready (boundary generation=%d)",
                    invalidation_generation,
                )
            else:
                self.logger.error(
                    "Robot reset/hold failed; /start is blocked until a successful /reset: %s",
                    self._reset_error,
                )
            return reset_ok
        finally:
            self._reset_in_progress.clear()
            self._reset_operation_lock.release()

    def _prepare_reset(self) -> None:
        """Optional hardware handover, after the rollout has entered RESETTING."""

    def _reset_cancelled(self) -> bool:
        return not self.running

    def stop(self) -> None:
        """Terminal process shutdown; unlike reset_rollout, disconnect hardware."""

        self.shutdown_event.set()
        with self._reset_operation_lock:
            with self._lifecycle_changed:
                if self._rollout_state is RolloutState.STOPPED:
                    return
                self._rollout_state = RolloutState.STOPPED
                self._advance_generation_locked()
                self._clear_episode_state_locked()
                self._lifecycle_changed.notify_all()
            with self._observation_capture_lock, self._robot_command_lock:
                super().stop()
        collector = self._observation_collector
        if collector is not None and collector is not threading.current_thread():
            collector.join(timeout=3)
        if getattr(self, "_rlt_runtime", None) is not None:
            self._rlt_runtime.close()

    def _advance_generation_locked(self) -> None:
        self._rollout_generation += 1
        self._request_sequence = 0

    def _fail_worker(self, operation: str, error: Exception) -> None:
        """Fence execution and wake the owner so it can disconnect in its finally block."""

        self.logger.error("Fatal RTC %s failure: %s", operation, error, exc_info=True)
        with self._lifecycle_changed:
            if self._rollout_state is RolloutState.STOPPED:
                return
            self._worker_error = f"{operation}: {type(error).__name__}: {error}"
            self._rollout_state = RolloutState.FAULT
            self.shutdown_event.set()
            self._advance_generation_locked()
            try:
                self._clear_episode_state_locked()
            finally:
                self._lifecycle_changed.notify_all()

    def _clear_episode_state_locked(self) -> None:
        """Clear every client state that can carry policy history across a reset."""

        if getattr(self, "_rlt_runtime", None) is not None:
            self._rlt_runtime.cancel()
        self._latest_observation = None
        self._last_used_observation_sequence = -1
        self._observation_valid_after = time.monotonic()
        while True:
            try:
                self._request_mailbox.get_nowait()
            except Empty:
                break

        with self.latest_action_lock:
            self.action_schedule.clear()
            self.action_interpolator.reset()
            self.latest_action = -1
        if self._trained_runtime is not None:
            self._trained_runtime.reset()
        self._force_fresh_snapshot = False
        self._retired_request_id = -1
        self._clear_command_diagnostics()

        self.action_chunk_size = self.config.actions_per_chunk

        with self._transport_lock:
            self._latest_latency_feedback = None
            self._latest_received_chunk_id = -1
            self._inflight_request_id = None
            self._inflight_requested_at = None
            # Execution and feedback ids never cross a generation boundary;
            # still-valid process timing estimates do.
            self._expire_rtt_estimate_locked()
            self._consecutive_rtt_capacity_overruns = 0
            self._rtt_capacity_warning_emitted = False
            self._request_timings.clear()

        self._starvation_started_at = None
        self._starvation_started_executed = -1
        self._starvation_missed_command_ticks = 0
        self._starvation_request_id = None

    # ------------------------------------------------------------------
    # Reset pose
    # ------------------------------------------------------------------

    def _validate_reset_pose(self, pose: dict[str, float] | None) -> dict[str, float] | None:
        if pose is None:
            return None
        action_keys = set(self.robot.action_features)
        pose_keys = set(pose)
        missing = sorted(action_keys - pose_keys)
        unknown = sorted(pose_keys - action_keys)
        if missing or unknown:
            raise ValueError(
                "reset_pose must contain exactly the robot action features; "
                f"missing={missing}, unknown={unknown}"
            )
        validated = {key: float(pose[key]) for key in self.robot.action_features}
        if not all(math.isfinite(value) for value in validated.values()):
            raise ValueError("reset_pose must contain finite positions")
        return validated

    def _read_action_pose_unlocked(self) -> dict[str, float]:
        observation = self.robot.get_observation()
        missing = [key for key in self.robot.action_features if key not in observation]
        if missing:
            raise KeyError(f"robot observation has no reset values for {missing}")
        return {key: float(observation[key]) for key in self.robot.action_features}

    def _read_action_pose(self) -> dict[str, float]:
        with self._observation_capture_lock:
            return self._read_action_pose_unlocked()

    def _send_reset_action(self, action: dict[str, float]) -> dict[str, float]:
        """Use planned positioning when the robot exposes a reset-specific entrypoint."""
        planned = getattr(self.robot, "send_action_planned", None)
        return planned(action) if callable(planned) else self.robot.send_action(action)

    def _reset_robot_pose(self) -> bool:
        try:
            with self._observation_capture_lock, self._robot_command_lock:
                current = self._read_action_pose_unlocked()
                target = self._configured_reset_pose
                if target is None and self.config.reset_to_initial_position:
                    target = self._initial_pose
                    if target is None:
                        raise RuntimeError("startup reset pose is unavailable; configure reset_pose")
                if target is None:
                    target = current
                if not all(math.isfinite(value) for value in (*current.values(), *target.values())):
                    raise ValueError("reset target and measured positions must be finite")

                target = dict(target)
                gripper_targets = {}
                if self.config.reset_open_gripper and self.config.reset_gripper_open_position is not None:
                    gripper_targets = {
                        key: self.config.reset_gripper_open_position
                        for key in target
                        if key.endswith("gripper.pos")
                    }
                    # Keep the measured opening, not the previous closing
                    # command or startup pose, until the arm is back in place.
                    target.update({key: current[key] for key in gripper_targets})
                if gripper_targets:
                    seed_command = getattr(self.robot, "seed_command", None)
                    if callable(seed_command):
                        # Robot-side EMA can still contain a fully closed
                        # target even though an object holds the gripper open.
                        seed_command(current)
                    self.logger.info("Resetting arm while holding measured gripper opening")

                steps = max(int(self.config.reset_duration_s * self.config.reset_fps), 1)
                for step in range(1, steps + 1):
                    if self._reset_cancelled():
                        return False
                    ratio = step / steps
                    command = {key: current[key] * (1.0 - ratio) + target[key] * ratio for key in target}
                    self._send_reset_action(command)
                    if self.config.reset_duration_s:
                        precise_sleep(1.0 / self.config.reset_fps)
                # A loaded gripper can drift slightly from its measured hold
                # target. Gate release on arm arrival; verify the gripper's
                # convergence only after commanding it open in the next phase.
                if not self._wait_for_reset_pose_unlocked(target, verify_gripper=not gripper_targets):
                    return False
                if not gripper_targets:
                    return True
                target.update(gripper_targets)
                self.logger.info("Reset arm pose verified; opening gripper to %s", gripper_targets)
                return self._wait_for_reset_pose_unlocked(target, phase="reset gripper release")
        except Exception as exc:
            self._reset_error = str(exc)
            self.logger.exception("Could not reset/hold robot pose: %s", exc)
            return False

    def _read_reset_poses_unlocked(self, timeout_s: float) -> dict[str, dict[str, float]]:
        del timeout_s
        return {"robot": self._read_action_pose_unlocked()}

    def _wait_for_reset_pose_unlocked(
        self, target: dict[str, float], *, phase: str = "reset pose", verify_gripper: bool = True
    ) -> bool:
        """Command the full target, optionally gating arrival on arm joints only."""

        deadline = time.monotonic() + self.config.reset_verify_timeout_s
        stable_since: float | None = None
        errors: dict[str, dict[str, float]] = {}
        while time.monotonic() < deadline:
            if self._reset_cancelled():
                return False
            # Refresh the planned target while checking measured arrival.
            # A successful send alone is never proof of arrival.
            self._send_reset_action(target)
            poses = self._read_reset_poses_unlocked(max(0.0, deadline - time.monotonic()))
            if not poses:
                raise RuntimeError("reset feedback contains no robot poses")
            errors = {}
            for arm, pose in poses.items():
                errors[arm] = {}
                for key, expected in target.items():
                    actual = float(pose[key])
                    if not math.isfinite(actual):
                        raise ValueError(f"non-finite reset feedback: {arm}/{key}")
                    is_gripper = key.endswith("gripper.pos")
                    if is_gripper and not verify_gripper:
                        continue
                    tolerance = (
                        self.config.reset_gripper_tolerance
                        if is_gripper
                        else self.config.reset_joint_tolerance
                    )
                    error = abs(actual - expected)
                    if error > tolerance:
                        errors[arm][key] = error
            now = time.monotonic()
            if all(not arm_errors for arm_errors in errors.values()):
                if stable_since is None:
                    stable_since = now
                if now <= deadline and now - stable_since >= self.config.reset_settle_s:
                    return not self._reset_cancelled()
            else:
                stable_since = None
            precise_sleep(min(1.0 / self.config.reset_fps, max(0.0, deadline - now)))
        raise TimeoutError(f"{phase} did not settle within {self.config.reset_verify_timeout_s}s: {errors}")

    # ------------------------------------------------------------------
    # Generation-fenced transport
    # ------------------------------------------------------------------

    def _chunk_round_trip(self, chunk: RemoteRTCActionChunk | RemoteRTCRetry) -> float:
        # The prefix is captured at snapshot_at, after camera capture. Use the
        # client's monotonic clock for feedback; NTP changes must not alter d.
        with self._transport_lock:
            timing = self._request_timings.get(chunk.source_observation_id)
            snapshot_at = None if timing is None else timing.snapshot_at
        if snapshot_at is not None:
            return max(0.0, time.monotonic() - snapshot_at)
        return super()._chunk_round_trip(chunk)

    def _trigger_threshold(self) -> int:
        """Return the refill threshold for a continuous single-inference pipeline.

        There are two independent lower bounds:

        * ``latency_guard`` retains at least one measured RTT plus the safety
          margin, preventing the currently scheduled trajectory from draining
          while the next inference runs.
        * ``continuous_refill`` accounts for the actions normally consumed by
          the request that just completed.  In steady state a returned
          ``A``-action chunk has roughly ``A - RTT`` live actions left.  Using
          that value (plus the margin) lets the next request start on the next
          control tick instead of idling until the old half-chunk threshold.

        The request-id slot remains the authoritative gate, so raising this
        threshold cannot create concurrent inference RPCs.
        """

        if self._trained_runtime is not None:
            return self._trained_runtime.queue_threshold
        configured_threshold = super()._trigger_threshold()
        if not self.config.adaptive_rtt_trigger:
            return configured_threshold

        with self._transport_lock:
            self._expire_rtt_estimate_locked()
            estimated_rtt_steps = self._adaptive_rtt_steps
        if estimated_rtt_steps is None:
            return configured_threshold

        estimated_rtt_steps = math.ceil(estimated_rtt_steps)
        margin = self.config.rtt_trigger_margin_steps
        latency_guard = estimated_rtt_steps + margin
        continuous_refill = self.action_chunk_size - estimated_rtt_steps + margin
        adaptive_threshold = max(latency_guard, continuous_refill)
        maximum_threshold = max(0, self.action_chunk_size - 1)
        return min(maximum_threshold, max(configured_threshold, adaptive_threshold))

    def _request_observation_if_ready(self, task: str | None = None) -> bool:
        """Publish a generation-bound request task when inference is due."""

        with self._defer_rtc_logs(), self._lifecycle_lock:
            if self._rollout_state is not RolloutState.RUNNING or not self.running:
                return False

            self._expire_inflight_request_locked()

            with self._transport_lock:
                # One request owns capture, upload and inference until it
                # completes, fails or expires. Queue checks are unnecessary
                # while that slot is occupied.
                if self._inflight_request_id is not None:
                    return False

            with self.latest_action_lock:
                executed = self.latest_action
            queue_size = self.action_schedule.qsize(executed)
            trigger_threshold = self._trigger_threshold()
            if not self._force_fresh_snapshot and queue_size > trigger_threshold:
                return False

            if self._request_sequence >= ROLLOUT_SEQUENCE_LIMIT:
                raise RuntimeError("rollout produced too many observation requests")
            observation_id = encode_rollout_observation_id(
                self._rollout_generation,
                self._request_sequence,
            )
            request = _SessionObservationRequest(
                observation_id=observation_id,
                generation=self._rollout_generation,
                task=self._task if task is None else task,
                triggered_action_step=executed,
            )
            with self._transport_lock:
                self._inflight_request_id = observation_id
                self._force_fresh_snapshot = False
                requested_at = time.monotonic()
                self._inflight_requested_at = requested_at
                self._request_timings[observation_id] = _RequestTiming(
                    requested_at=requested_at,
                    triggered_action_step=executed,
                    queued_actions=queue_size,
                )
            self._request_sequence += 1
            _publish_latest(self._request_mailbox, request)

        self._log_rtc(
            logging.DEBUG,
            "Triggered RTC observation %d (generation=%d, action=%d, queued=%d, threshold=%d)",
            request.observation_id,
            request.generation,
            executed,
            queue_size,
            trigger_threshold,
        )
        return True

    def _request_is_current_locked(self, request: _SessionObservationRequest) -> bool:
        return (
            self._rollout_state is RolloutState.RUNNING
            and request.generation == self._rollout_generation
            and request.observation_id > self._retired_request_id
        )

    def _expire_inflight_request_locked(self) -> None:
        """Recover a lost terminal response without reusing its observation or prefix."""

        with self._transport_lock:
            observation_id = self._inflight_request_id
            requested_at = self._inflight_requested_at
        if observation_id is None or requested_at is None:
            return
        elapsed = time.monotonic() - requested_at
        if elapsed < self.config.request_timeout_s:
            return
        self._retire_failed_request(observation_id)
        self._log_rtc(
            logging.WARNING,
            "RTC request %d timed out after %.1fs; capturing a fresh observation and prefix",
            observation_id,
            elapsed,
        )

    def _mark_request_timing(self, observation_id: int, field_name: str, timestamp: float) -> None:
        # snapshot_at also drives monotonic RTC feedback; it is not optional telemetry.
        if field_name != "snapshot_at" and not self._timing_diagnostics_enabled:
            return
        with self._transport_lock:
            timing = self._request_timings.get(observation_id)
            if timing is not None:
                setattr(timing, field_name, timestamp)

    def _on_observation_transport_phase(
        self,
        observation_id: int,
        phase: str,
        timestamp: float,
    ) -> None:
        phase_fields = {
            "serialization_started": "serialization_started_at",
            "serialization_finished": "serialization_finished_at",
            "upload_started": "upload_started_at",
            "upload_finished": "upload_finished_at",
        }
        field_name = phase_fields.get(phase)
        if field_name is not None:
            self._mark_request_timing(observation_id, field_name, timestamp)

    @staticmethod
    def _phase_ms(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round(max(0.0, end - start) * 1000, 3)

    def _log_request_timing(
        self,
        chunk: RemoteRTCActionChunk,
        arrived_at: float,
        *,
        merged: bool,
    ) -> None:
        with self._transport_lock:
            timing = self._request_timings.pop(chunk.source_observation_id, None)
        if timing is None or not self._timing_diagnostics_enabled:
            return
        with self.latest_action_lock:
            first_mutable = self.action_schedule.first_mutable_step(self.latest_action)
        snapshot_rtt = None if timing.snapshot_at is None else max(0.0, arrived_at - timing.snapshot_at)

        self._log_rtc(
            logging.INFO if self.config.log_rtc_timing else logging.DEBUG,
            "RTC RPC PHASES obs=%d accepted=%s trigger_action=%d queued_at_request=%d total_ms=%.3f "
            "mailbox_wait_ms=%s observation_wait_ms=%s capture_ms=%s capture_to_snapshot_ms=%s serialization_ms=%s "
            "upload_rpc_ms=%s post_upload_to_chunk_ms=%s server_inference_ms=%.3f "
            "snapshot_to_result_ms=%s outside_server_ms=%s consumed_steps=%d conditioned_delay=%d "
            "live_steps=%d",
            chunk.source_observation_id,
            merged,
            timing.triggered_action_step,
            timing.queued_actions,
            max(0.0, arrived_at - timing.requested_at) * 1000,
            self._phase_ms(
                timing.requested_at,
                timing.mailbox_dequeued_at
                if timing.mailbox_dequeued_at is not None
                else timing.capture_started_at,
            ),
            self._phase_ms(timing.mailbox_dequeued_at, timing.observation_selected_at),
            self._phase_ms(timing.capture_started_at, timing.capture_finished_at),
            self._phase_ms(timing.capture_finished_at, timing.snapshot_at),
            self._phase_ms(timing.serialization_started_at, timing.serialization_finished_at),
            self._phase_ms(timing.upload_started_at, timing.upload_finished_at),
            self._phase_ms(timing.upload_finished_at, arrived_at),
            chunk.inference_seconds * 1000,
            None if snapshot_rtt is None else round(snapshot_rtt * 1000, 3),
            None
            if snapshot_rtt is None
            else round(max(0.0, snapshot_rtt - chunk.inference_seconds) * 1000, 3),
            max(0, first_mutable - chunk.chunk_start_step),
            chunk.conditioned_delay,
            sum(row.get_timestep() >= first_mutable for row in chunk.actions) if merged else 0,
        )

    def _log_observation_tracking(
        self,
        request: _SessionObservationRequest,
        raw_observation: RawObservation,
        snapshot: _ExecutionSnapshot,
        capture_started_at: float,
        capture_finished_at: float,
        snapshot_at: float,
        capture_started_executed: int,
        inflight_step: int | None,
        command: _CommandDiagnosticsSnapshot | None,
    ) -> None:
        if not self.config.debug_rtc_diagnostics:
            return
        prefix_length = (
            len(snapshot.trained_prefix) if snapshot.trained_prefix is not None else len(snapshot.action_refs)
        )
        first_future_step = snapshot.executed_action_step + 1 if prefix_length else None
        first_ref = None
        if snapshot.action_refs:
            ref = snapshot.action_refs[0]
            first_ref = f"{ref.source_observation_id}:{ref.action_index}"
        elif snapshot.trained_prefix is not None:
            first_ref = "trained_tensor" if prefix_length else None

        measured: torch.Tensor | None = None
        tracking_error: torch.Tensor | None = None
        try:
            measured = torch.tensor(
                [float(raw_observation[key]) for key in self.robot.action_features],
                dtype=torch.float32,
            )
            if command is not None and measured.shape == command.command.shape:
                tracking_error = measured - command.command.to(dtype=measured.dtype)
        except (KeyError, TypeError, ValueError):
            measured = None

        max_tracking_error = self._max_abs(tracking_error)
        warning_threshold = self.config.tracking_error_warning_threshold
        level = (
            logging.WARNING
            if max_tracking_error is not None
            and warning_threshold > 0
            and max_tracking_error >= warning_threshold
            else logging.DEBUG
        )
        if level == logging.DEBUG and not self.config.debug_rtc_diagnostics:
            return
        self._log_rtc(
            level,
            "RTC OBS SNAPSHOT obs=%d trigger_executed=%d capture_started_executed=%d "
            "captured_executed=%d trigger_to_snapshot_advance=%d capture_execution_advance=%d "
            "timed_step=%d first_future=%s prefix=%d first_ref=%s inflight=%s capture_ms=%.3f "
            "capture_to_snapshot_ms=%.3f "
            "last_command_step=%s command_age_ms=%s command_after_capture=%s "
            "tracking_error_max=%s measured=%s "
            "last_command=%s tracking_error=%s",
            request.observation_id,
            request.triggered_action_step,
            capture_started_executed,
            snapshot.executed_action_step,
            snapshot.executed_action_step - request.triggered_action_step,
            snapshot.executed_action_step - capture_started_executed,
            max(snapshot.executed_action_step + 1, 0),
            first_future_step,
            prefix_length,
            first_ref,
            inflight_step,
            max(0.0, capture_finished_at - capture_started_at) * 1000,
            max(0.0, snapshot_at - capture_finished_at) * 1000,
            None if command is None else command.policy_step,
            None if command is None else round((capture_finished_at - command.monotonic_time) * 1000, 3),
            command is not None and command.monotonic_time > capture_finished_at,
            max_tracking_error,
            self._diagnostic_vector(measured),
            None if command is None else self._diagnostic_vector(command.command),
            self._diagnostic_vector(tracking_error),
        )

    def _release_inflight_request(self, observation_id: int) -> bool:
        """Release the inference slot iff it is still owned by ``observation_id``."""

        with self._transport_lock:
            if self._inflight_request_id != observation_id:
                return False
            self._inflight_request_id = None
            self._inflight_requested_at = None
            self._request_timings.pop(observation_id, None)
            if self._trained_runtime is not None:
                self._trained_runtime.forget_request(observation_id)
            return True

    def _expire_rtt_estimate_locked(self) -> None:
        if (
            self._rtt_updated_at is None
            or time.monotonic() - self._rtt_updated_at > self.config.latency_history_ttl_s
        ):
            self._adaptive_rtt_steps = None
            self._latest_request_rtt_seconds = None
            self._rtt_updated_at = None

    def _update_rtt_estimate_locked(self, seconds: float, arrived_at: float) -> None:
        self._expire_rtt_estimate_locked()
        sample_steps = math.ceil(seconds / self.config.environment_dt)
        estimate = self._adaptive_rtt_steps
        self._adaptive_rtt_steps = (
            float(sample_steps)
            if estimate is None or sample_steps >= estimate
            else (1.0 - _RTT_ESTIMATE_DECAY) * estimate + _RTT_ESTIMATE_DECAY * sample_steps
        )
        self._latest_request_rtt_seconds = seconds
        self._rtt_updated_at = arrived_at

    def _complete_inflight_request(self, observation_id: int, arrived_at: float) -> float | None:
        """Close a matching request and update its end-to-end latency estimate."""

        capacity_warning: tuple[int, int] | None = None
        with self._transport_lock:
            if self._inflight_request_id != observation_id:
                return None

            requested_at = self._inflight_requested_at
            request_rtt = None if requested_at is None else max(0.0, arrived_at - requested_at)
            if request_rtt is not None and self._trained_runtime is None:
                sample_steps = math.ceil(request_rtt / self.config.environment_dt)
                self._update_rtt_estimate_locked(request_rtt, arrived_at)

                single_flight_capacity = 2 * sample_steps + self.config.rtt_trigger_margin_steps
                if single_flight_capacity >= self.action_chunk_size:
                    self._consecutive_rtt_capacity_overruns += 1
                else:
                    self._consecutive_rtt_capacity_overruns = 0
                if self._consecutive_rtt_capacity_overruns >= 2 and not self._rtt_capacity_warning_emitted:
                    self._rtt_capacity_warning_emitted = True
                    capacity_warning = (single_flight_capacity, self.action_chunk_size)

            self._inflight_request_id = None
            self._inflight_requested_at = None
            if self._trained_runtime is not None:
                self._trained_runtime.forget_request(observation_id)

        if capacity_warning is not None:
            required_steps, chunk_size = capacity_warning
            self._log_rtc(
                logging.WARNING,
                "Single-inference RTC needs about %d actions_per_chunk steps to cover one RTT plus "
                "the next adaptive refill budget, but actions_per_chunk=%d; normal latency jitter "
                "may still starve the schedule",
                required_steps,
                chunk_size,
            )
        return request_rtt

    def _retire_failed_request(self, observation_id: int) -> None:
        """Make failed local work retryable without touching a newer request."""

        with self._lifecycle_lock, self._transport_lock:
            if self._inflight_request_id == observation_id:
                self._retired_request_id = max(self._retired_request_id, observation_id)
                self._inflight_request_id = None
                self._inflight_requested_at = None
                self._request_timings.pop(observation_id, None)
                self._force_fresh_snapshot = True
                self._latest_observation = None
                self._observation_valid_after = time.monotonic()
                self._lifecycle_changed.notify_all()
                if self._trained_runtime is not None:
                    self._trained_runtime.forget_request(observation_id)

    def _observation_collection_active_locked(self) -> bool:
        """Subclasses can suspend inference capture during human intervention."""
        return self.running and self._rollout_state is RolloutState.RUNNING

    def _observation_is_usable_locked(self, sample: _CapturedObservation) -> bool:
        return (
            sample.generation == self._rollout_generation
            and sample.started_at >= self._observation_valid_after
            and (sample.oldest_camera_at is None or sample.oldest_camera_at >= self._observation_valid_after)
            and sample.age(time.monotonic()) <= self.config.observation_max_age_s
        )

    def _start_observation_collector(self) -> None:
        with self._lifecycle_lock:
            if not self.config.continuous_observation or not self.running:
                return
            if self._observation_collector is not None:
                return
            self._observation_collector = threading.Thread(
                target=self._observation_collector_worker,
                name="rtc-observation-collector",
                daemon=True,
            )
            self._observation_collector.start()

    def _observation_collector_worker(self) -> None:
        try:
            self._collect_observations()
        except Exception as exc:
            self._fail_worker("observation collector", exc)

    def _capture_observation(self, generation: int, *, buffered_cameras: bool) -> _CapturedObservation:
        """Caller owns capture lock, never the lifecycle or motor-command lock."""
        started_action_step = -1
        if self.config.debug_rtc_diagnostics:
            with self.latest_action_lock:
                started_action_step = self.latest_action
        started_at = time.monotonic()
        camera_times: dict[str, float] = {}
        if buffered_cameras and self._timestamped_observation_supported:
            try:
                raw, camera_times = self.robot.get_observation_with_timestamps(
                    max_age_ms=self.config.observation_max_age_s * 1000
                )
            except NotImplementedError:
                self._timestamped_observation_supported = False
                self.logger.warning(
                    "Robot/camera does not support timestamped peeking; background capture will use "
                    "get_observation(), with capture age only (camera exposure age unknown)"
                )
                raw = self.robot.get_observation()
        else:
            raw = self.robot.get_observation()
        # Own the values before releasing the capture slot: no producer may
        # mutate a previously published image or state through a reused buffer.
        observation = copy.deepcopy(raw)
        camera_clock = time.perf_counter()
        finished_at = time.monotonic()
        # Camera availability clocks need not share an epoch with monotonic().
        oldest_camera_at = (
            finished_at - max(0.0, camera_clock - min(camera_times.values())) if camera_times else None
        )
        return _CapturedObservation(
            generation=generation,
            sequence=0,
            observation=observation,
            started_at=started_at,
            finished_at=finished_at,
            started_action_step=started_action_step,
            oldest_camera_at=oldest_camera_at,
        )

    def _collect_observations(self) -> None:
        """Replace one cache entry at sensor cadence, including while RPCs block."""
        period = 1.0 / (self.config.observation_capture_fps or self.config.fps)
        last_warning_at = -math.inf
        while self.running:
            with self._lifecycle_changed:
                self._lifecycle_changed.wait_for(
                    lambda: not self.running or self._observation_collection_active_locked(), timeout=0.1
                )
                if not self._observation_collection_active_locked():
                    continue
                generation = self._rollout_generation
            cycle_started_at = time.monotonic()
            try:
                with self._observation_capture_lock:
                    with self._lifecycle_lock:
                        if (
                            generation != self._rollout_generation
                            or not self._observation_collection_active_locked()
                        ):
                            continue
                    sample = self._capture_observation(generation, buffered_cameras=True)
                with self._lifecycle_changed:
                    if (
                        generation == self._rollout_generation
                        and self._observation_collection_active_locked()
                        and sample.started_at >= self._observation_valid_after
                    ):
                        self._observation_sequence += 1
                        self._latest_observation = replace(sample, sequence=self._observation_sequence)
                        self._lifecycle_changed.notify_all()
                if sample.age(time.monotonic()) > self.config.observation_max_age_s:
                    raise TimeoutError("captured observation already exceeds observation_max_age_s")
            except Exception as exc:
                # A temporarily unavailable camera is recoverable. Do not
                # re-date its previous frame or block local action dispatch.
                now = time.monotonic()
                if now - last_warning_at >= 1.0:
                    self.logger.warning("RTC observation collector waiting for fresh sensors: %s", exc)
                    last_warning_at = now
            # No catch-up bursts if acquisition took longer than one period.
            self.shutdown_event.wait(max(0.0, period - (time.monotonic() - cycle_started_at)))

    def _observation_for_request(self, request: _SessionObservationRequest) -> _CapturedObservation | None:
        if not self.config.continuous_observation:
            with self._observation_capture_lock:
                with self._lifecycle_lock:
                    if not self._request_is_current_locked(request):
                        return None
                return self._capture_observation(request.generation, buffered_cameras=False)

        with self._defer_rtc_logs(), self._lifecycle_changed:
            while self.running and self._request_is_current_locked(request):
                self._expire_inflight_request_locked()
                if not self._request_is_current_locked(request):
                    return None
                sample = self._latest_observation
                if (
                    sample is not None
                    and sample.generation == request.generation
                    and sample.sequence > self._last_used_observation_sequence
                    and self._observation_is_usable_locked(sample)
                ):
                    self._last_used_observation_sequence = sample.sequence
                    return sample
                self._lifecycle_changed.wait(timeout=0.1)
        return None

    def observation_sender(self) -> None:
        """Upload the latest sensors with a live prefix, fencing reset/timeout boundaries."""

        self.start_barrier.wait()
        self._start_observation_collector()
        self.logger.info("Resettable RTC observation sender started")
        while self.running:
            try:
                request = self._request_mailbox.get(timeout=0.1)
            except Empty:
                continue

            if self._timing_diagnostics_enabled:
                self._mark_request_timing(request.observation_id, "mailbox_dequeued_at", time.monotonic())
            with self._lifecycle_lock:
                if not self._request_is_current_locked(request):
                    continue

            try:
                sample = self._observation_for_request(request)
                if sample is None:
                    continue

                # Sensor age is bounded, but prefix/timestep always describe
                # the live execution schedule, never a cached schedule.
                with self._defer_rtc_logs(), self._lifecycle_lock:
                    if not self._request_is_current_locked(request):
                        self._log_rtc(
                            logging.DEBUG,
                            "Dropping observation %d after its request was invalidated",
                            request.observation_id,
                        )
                        continue
                    latest = self._latest_observation
                    if (
                        self.config.continuous_observation
                        and latest is not None
                        and latest.sequence > sample.sequence
                        and self._observation_is_usable_locked(latest)
                    ):
                        sample = latest
                        self._last_used_observation_sequence = latest.sequence
                    if self.config.continuous_observation and not self._observation_is_usable_locked(sample):
                        _publish_latest(self._request_mailbox, request)
                        continue
                    selected_at = time.monotonic() if self._timing_diagnostics_enabled else None
                    capture_started_at = sample.started_at
                    capture_finished_at = sample.finished_at
                    capture_started_executed = sample.started_action_step
                    # Never attach task metadata to the shared cache entry itself.
                    raw_observation = dict(sample.observation)
                    snapshot: _ExecutionSnapshot = self._snapshot_execution_state()
                    snapshot_at = time.monotonic()
                    if self._timing_diagnostics_enabled:
                        self._mark_request_timing(
                            request.observation_id, "observation_selected_at", selected_at
                        )
                        self._mark_request_timing(
                            request.observation_id, "capture_started_at", capture_started_at
                        )
                        self._mark_request_timing(
                            request.observation_id, "capture_finished_at", capture_finished_at
                        )
                    self._mark_request_timing(request.observation_id, "snapshot_at", snapshot_at)
                    if self.config.debug_rtc_diagnostics:
                        inflight = self.action_schedule.inflight()
                        inflight_step = None if inflight is None else inflight.timed_action.get_timestep()
                        command = self.command_diagnostics_snapshot()
                    with self._transport_lock:
                        feedback = self._latest_latency_feedback
                    raw_observation["task"] = request.task
                    timed_observation = TimedObservation(
                        timestamp=snapshot.timestamp,
                        timestep=max(snapshot.executed_action_step + 1, 0),
                        observation=raw_observation,
                        must_go=True,
                    )
                    envelope = RemoteRTCObservation(
                        observation=timed_observation,
                        observation_id=request.observation_id,
                        executed_action_step=snapshot.executed_action_step,
                        action_refs=snapshot.action_refs,
                        trained_prefix=snapshot.trained_prefix,
                        latency_feedback_seconds=None if feedback is None else feedback[1],
                        latency_feedback_source_id=None if feedback is None else feedback[0],
                    )

                if self._timing_diagnostics_enabled:
                    self._log_rtc(
                        logging.INFO if self.config.log_rtc_timing else logging.DEBUG,
                        "RTC OBS SNAPSHOT obs=%d continuous=%s sample=%d sensor_age_ms=%.3f "
                        "cache_age_ms=%.3f camera_age_ms=%s capture_ms=%.3f",
                        request.observation_id,
                        self.config.continuous_observation,
                        sample.sequence,
                        sample.age(snapshot_at) * 1000,
                        max(0.0, snapshot_at - sample.finished_at) * 1000,
                        self._phase_ms(sample.oldest_camera_at, snapshot_at),
                        (sample.finished_at - sample.started_at) * 1000,
                    )
                if self.config.debug_rtc_diagnostics:
                    self._log_observation_tracking(
                        request,
                        raw_observation,
                        snapshot,
                        capture_started_at,
                        capture_finished_at,
                        snapshot_at,
                        capture_started_executed,
                        inflight_step,
                        command,
                    )
                    if snapshot.executed_action_step != request.triggered_action_step:
                        self._log_rtc(
                            logging.DEBUG,
                            "RTC observation %d selected with execution advanced from %d to %d",
                            request.observation_id,
                            request.triggered_action_step,
                            snapshot.executed_action_step,
                        )
                self.send_observation(envelope)
            except Exception as exc:
                self._retire_failed_request(request.observation_id)
                self.logger.error("RTC observation capture/send failed: %s", exc, exc_info=True)

    def send_observation(self, envelope: RemoteRTCObservation) -> bool:
        """Reserve the request and upload its live observation and action prefix.

        The upload runs outside the lifecycle lock so reset never waits for a
        slow RPC. Once admitted, every snapshot is needed to refill the queue.
        """

        generation = decode_rollout_generation(envelope.observation_id)

        with self._defer_rtc_logs(), self._lifecycle_lock:
            if (
                self._rollout_state is not RolloutState.RUNNING
                or generation != self._rollout_generation
                or envelope.observation_id <= self._retired_request_id
            ):
                return False

            # The normal path reserved this id before publishing the mailbox
            # request.  Claim an empty slot here as a compatibility path for
            # direct callers, but never allow a different observation to pass
            # while the current request is awaiting its action chunk.
            with self._transport_lock:
                inflight_request_id = self._inflight_request_id
                if inflight_request_id is None:
                    self._inflight_request_id = envelope.observation_id
                    requested_at = time.monotonic()
                    self._inflight_requested_at = requested_at
                    self._request_sequence = max(
                        self._request_sequence, decode_rollout_sequence(envelope.observation_id) + 1
                    )
                    self._request_timings[envelope.observation_id] = _RequestTiming(
                        requested_at=requested_at,
                        triggered_action_step=envelope.executed_action_step,
                        queued_actions=(
                            len(envelope.trained_prefix)
                            if envelope.trained_prefix is not None
                            else len(envelope.action_refs)
                        ),
                    )
                elif inflight_request_id != envelope.observation_id:
                    self._log_rtc(
                        logging.DEBUG,
                        "Skipping RTC observation %d while request %d is in flight",
                        envelope.observation_id,
                        inflight_request_id,
                    )
                    return False

            if self._trained_runtime is not None:
                self._trained_runtime.remember_request(envelope)
            envelope.observation.must_go = True

        # The RPC may finish after reset; _handle_action_chunk fences its result.
        try:
            sent = super().send_observation(envelope)
        except Exception:
            self._retire_failed_request(envelope.observation_id)
            raise
        with self._lifecycle_lock:
            is_current = (
                self._rollout_state is RolloutState.RUNNING
                and generation == self._rollout_generation
                and envelope.observation_id > self._retired_request_id
            )
        if not sent:
            self._retire_failed_request(envelope.observation_id)
        elif not is_current:
            # Reset normally cleared this already.  The id comparison keeps a
            # late upload completion from releasing a new rollout's slot.
            self._release_inflight_request(envelope.observation_id)
        return sent and is_current

    def _handle_snapshot_retry(self, response: RemoteRTCRetry) -> bool:
        """Close a rejected request and let capture build a fresh observation/prefix."""
        with self._defer_rtc_logs(), self._lifecycle_lock:
            if (
                self._rollout_state is not RolloutState.RUNNING
                or decode_rollout_generation(response.source_observation_id) != self._rollout_generation
            ):
                return False
            with self._transport_lock:
                if self._inflight_request_id != response.source_observation_id:
                    return False
            try:
                if response.rtc_mode != self.config.rtc_mode:
                    raise ValueError("RTC retry mode does not match the client")
                if self._trained_runtime is not None:
                    self._trained_runtime.reject(response.reason)
                else:
                    round_trip = self._chunk_round_trip(response)
                    with self._transport_lock:
                        self._latest_latency_feedback = (
                            response.source_observation_id,
                            round_trip,
                        )
                self._force_fresh_snapshot = True
            except (ValueError, RuntimeError):
                self.shutdown_event.set()
                raise
            finally:
                self._complete_inflight_request(response.source_observation_id, time.monotonic())
                with self._transport_lock:
                    self._request_timings.pop(response.source_observation_id, None)
                    self._latest_received_chunk_id = max(
                        self._latest_received_chunk_id, response.source_observation_id
                    )
            self._log_rtc(
                logging.WARNING,
                "RTC request %d rejected (%s); requesting a new observation and prefix",
                response.source_observation_id,
                response.reason,
            )
            return True

    def _handle_action_chunk(self, chunk: RemoteRTCActionChunk) -> bool:
        """Reject stale generations before they can mutate any client state."""

        generation = decode_rollout_generation(chunk.source_observation_id)
        arrived_at = time.monotonic()
        with self._defer_rtc_logs(), self._lifecycle_lock:
            if (
                self._rollout_state is not RolloutState.RUNNING
                or generation != self._rollout_generation
                or chunk.source_observation_id <= self._retired_request_id
            ):
                self._release_inflight_request(chunk.source_observation_id)
                self._log_rtc(
                    logging.DEBUG,
                    "Discarding RTC chunk %d from rollout generation %d (active=%d, state=%s)",
                    chunk.source_observation_id,
                    generation,
                    self._rollout_generation,
                    self._rollout_state.value,
                )
                return False
            with self._transport_lock:
                matching_request = self._inflight_request_id == chunk.source_observation_id
            merged = False
            try:
                if chunk.rtc_mode != self.config.rtc_mode:
                    raise ValueError("RTC chunk mode does not match the client")
                if self._trained_runtime is not None:
                    with self.latest_action_lock:
                        executed = self.latest_action
                    merged = self._trained_runtime.accept(chunk, executed)
                    if matching_request:
                        with self._transport_lock:
                            self._latest_received_chunk_id = chunk.source_observation_id
                        if not merged:
                            self._log_rtc(
                                logging.WARNING,
                                "Trained RTC request %d rejected (%s); requesting a fresh snapshot",
                                chunk.source_observation_id,
                                self._trained_runtime.last_rejection,
                            )
                else:
                    merged = super()._handle_action_chunk(chunk)
                if matching_request and (
                    not merged
                    or (self._trained_runtime is None and self.config.adaptive_rtt_trigger)
                    or (self._trained_runtime is not None and self.config.trained.replan_on_result)
                ):
                    # A fast return can leave more actions than the RTT-based
                    # threshold predicted. Still take a fresh observation on the
                    # next control tick: a longer prediction is not a reason to
                    # execute it open-loop for longer. The request slot remains
                    # occupied until the finally block below completes it.
                    self._force_fresh_snapshot = True
            except (ValueError, RuntimeError):
                self.shutdown_event.set()
                raise
            finally:
                # A returned chunk completes inference even when its actions are
                # stale, under-conditioned, or otherwise rejected by merge.
                # Only the matching id may open the next request slot.
                request_rtt = self._complete_inflight_request(
                    chunk.source_observation_id,
                    arrived_at,
                )
                self._log_request_timing(chunk, arrived_at, merged=merged)

            if (
                self.config.debug_rtc_diagnostics
                and request_rtt is not None
                and self._trained_runtime is None
            ):
                with self._transport_lock:
                    estimated_rtt_steps = self._adaptive_rtt_steps
                self._log_rtc(
                    logging.DEBUG,
                    "Adaptive RTC RTT obs=%d sample=%.1fms estimate=%.2f steps threshold=%d",
                    chunk.source_observation_id,
                    request_rtt * 1000,
                    0.0 if estimated_rtt_steps is None else estimated_rtt_steps,
                    self._trigger_threshold(),
                )
            return merged

    # ------------------------------------------------------------------
    # Local action path
    # ------------------------------------------------------------------

    def _update_starvation_diagnostics(self, performed: dict[str, Any] | None) -> None:
        if not self.config.debug_rtc_diagnostics:
            return
        now = time.monotonic()
        if performed is None and self.action_interpolator.needs_new_action():
            with self.latest_action_lock:
                executed = self.latest_action
                queued = self.action_schedule.qsize(executed)
            with self._transport_lock:
                request_id = self._inflight_request_id
                requested_at = self._inflight_requested_at
                latest_chunk_id = self._latest_received_chunk_id

            if self._starvation_started_at is None:
                self._starvation_started_at = now
                self._starvation_started_executed = executed
                self._starvation_missed_command_ticks = 0
                self._starvation_request_id = request_id
                self._log_rtc(
                    logging.WARNING,
                    "RTC STARVATION START executed=%d queued=%d latest_chunk=%d pending_request=%s "
                    "request_age_ms=%s",
                    executed,
                    queued,
                    latest_chunk_id,
                    request_id,
                    None if requested_at is None else round(max(0.0, now - requested_at) * 1000, 3),
                )
            elif self._starvation_request_id is None and request_id is not None:
                self._starvation_request_id = request_id
            self._starvation_missed_command_ticks += 1
            return

        if performed is None or self._starvation_started_at is None:
            return

        command = self.command_diagnostics_snapshot()
        with self.latest_action_lock:
            executed = self.latest_action
            queued = self.action_schedule.qsize(executed)
        duration_ms = max(0.0, now - self._starvation_started_at) * 1000
        missed_policy_steps = math.ceil(
            self._starvation_missed_command_ticks / self.action_interpolator.multiplier
        )
        self._log_rtc(
            logging.WARNING,
            "RTC STARVATION END duration_ms=%.3f missed_command_ticks=%d missed_policy_steps=%d "
            "started_executed=%d resumed_executed=%d queued=%d request=%s next_source=%s "
            "next_policy_step=%s",
            duration_ms,
            self._starvation_missed_command_ticks,
            missed_policy_steps,
            self._starvation_started_executed,
            executed,
            queued,
            self._starvation_request_id,
            None if command is None else command.source_observation_id,
            None if command is None else command.policy_step,
        )
        self._starvation_started_at = None
        self._starvation_started_executed = -1
        self._starvation_missed_command_ticks = 0
        self._starvation_request_id = None

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Dispatch only while the same lifecycle lock says RUNNING."""

        with self._defer_rtc_logs(), self._robot_command_lock, self._lifecycle_lock:
            if self._rollout_state is not RolloutState.RUNNING:
                return None
            performed = super().control_loop_action(verbose=verbose)
            self._update_starvation_diagnostics(performed)
            return performed

    def control_loop(
        self,
        task: str | None = None,
        verbose: bool = False,
    ) -> tuple[Observation | None, dict[str, Any] | None]:
        """Propagate fatal control failures to the process owner instead of dying silently."""

        try:
            return self._run_control_loop(task=task, verbose=verbose)
        except Exception as exc:
            self._fail_worker("local control loop", exc)
            return None, None

    def _run_control_loop(
        self,
        task: str | None = None,
        verbose: bool = False,
    ) -> tuple[Observation | None, dict[str, Any] | None]:
        """Keep a local control thread alive across any number of rollouts."""

        if task is not None:
            self.set_task(task)
        self.start_barrier.wait()
        self.logger.info("Resettable RTC local control loop ready; waiting for /start")
        last_observation: Observation | None = None
        last_action: dict[str, Any] | None = None

        while self.running:
            with self._lifecycle_changed:
                self._lifecycle_changed.wait_for(
                    lambda: not self.running or self._rollout_state is RolloutState.RUNNING,
                    timeout=0.1,
                )
                if not self.running:
                    break
                generation = self._rollout_generation
                last_action = None

            timer = CycleTimer(
                fps=self.config.fps,
                multiplier=self.action_interpolator.multiplier,
                records_data=False,
                diagnostics=self.config.debug_rtc_diagnostics,
            )
            try:
                while self.running:
                    with self._lifecycle_lock:
                        if (
                            self._rollout_state is not RolloutState.RUNNING
                            or self._rollout_generation != generation
                        ):
                            break

                    timer.tick(new_cycle=self.action_interpolator.needs_new_action())
                    performed = self.control_loop_action(verbose=verbose)
                    if performed is not None:
                        last_action = performed
                    self._request_observation_if_ready()

                    if self.config.debug_visualize_queue_size:
                        with self._lifecycle_lock:
                            if (
                                self._rollout_state is RolloutState.RUNNING
                                and self._rollout_generation == generation
                            ):
                                with self.latest_action_lock:
                                    latest = self.latest_action
                                self.action_queue_size.append(self.action_schedule.qsize(latest))
                    timer.wait()
            finally:
                timer.log_run_summary()

        return last_observation, last_action


def _render_status(client: RobotClientDeployment) -> str:
    with client.latest_action_lock:
        latest = client.latest_action
        queued = client.action_schedule.qsize(latest)
    with client._transport_lock:
        inflight = client._inflight_request_id
        rtt_steps = client._adaptive_rtt_steps
    trigger_threshold = client._trigger_threshold()
    rtt_text = "unknown" if rtt_steps is None else f"{rtt_steps:.2f}"
    return (
        f"state={client.rollout_state.value}, generation={client.rollout_generation}, "
        f"task={client.task!r}, executed_action_step={latest}, scheduled={queued}, "
        f"inflight_request={inflight}, rtt_steps={rtt_text}, trigger_threshold={trigger_threshold}, "
        f"worker_error={client.worker_error!r}, reset_error={client.reset_error!r}"
    )


def _run_interactive_prompt(
    client: RobotClientDeployment, prompt: Callable[[RobotClientDeployment], None]
) -> None:
    """Let the main thread observe shutdown even while stdin is waiting for a line."""

    finished = threading.Event()
    errors: list[Exception] = []

    def run_prompt() -> None:
        try:
            prompt(client)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    threading.Thread(target=run_prompt, daemon=True, name="RTCInteractivePrompt").start()
    try:
        while client.running and not finished.wait(timeout=0.1):
            pass
    except KeyboardInterrupt:
        return
    if errors:
        raise errors[0]


def _interactive_prompt(client: RobotClientDeployment) -> None:
    print(
        "Remote RTC interactive rollout ready. Commands: /start [task], /task <task>, "
        "/reset, /status, /help, /stop",
        flush=True,
    )
    while client.running:
        try:
            line = input("rtc> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        if not client.running:
            break
        if not line:
            continue

        command, _, arguments = line.partition(" ")
        command = command.lower()
        arguments = arguments.strip()
        if (
            command in {"/success", "/failure", "/truncate"}
            and getattr(client, "_rlt_runtime", None) is not None
        ):
            runtime = client._rlt_runtime
            if runtime.mark_manual_outcome(command.removeprefix("/")):
                print("RLT task outcome queued for the current policy interval", flush=True)
            else:
                print("Manual RLT feedback requires a running rollout and reward_provider=manual", flush=True)
        elif command == "/start":
            if client.start_rollout(arguments or None):
                print(f"started: {_render_status(client)}", flush=True)
            else:
                print(f"cannot start: {_render_status(client)}", flush=True)
        elif command == "/reset":
            reset_ok = client.reset_rollout()
            print(
                ("reset complete" if reset_ok else "reset failed or unavailable")
                + f": {_render_status(client)}",
                flush=True,
            )
        elif command == "/task":
            if not arguments:
                print(f"task={client.task!r}", flush=True)
            elif client.set_task(arguments):
                print(f"task set to {client.task!r}", flush=True)
            else:
                print(f"task unchanged or client stopped: {client.task!r}", flush=True)
        elif command == "/status":
            print(_render_status(client), flush=True)
        elif command == "/help":
            print(
                "/start [task]  start a clean rollout\n"
                "/task <task>   change the instruction used by future observations\n"
                "/reset         stop, clear the episode, and return/hold the robot\n"
                "/status        show lifecycle and schedule progress\n"
                "/stop          disconnect and end the client\n"
                "/success /failure /truncate  finish an RLT rollout with manual task feedback",
                flush=True,
            )
        elif command == "/stop":
            break
        else:
            print("unknown command; type /help", flush=True)


RobotClient = RobotClientDeployment


@draccus.wrap()
def async_client(cfg: RobotClientDeploymentConfig) -> None:
    configure_deployment_logging(debug=cfg.debug_rtc_diagnostics)
    logging.info(pformat(asdict(cfg)))
    client = RobotClientDeployment(cfg)
    if not client.start():
        client.stop()
        return

    if client._rlt_runtime is not None:
        threads = [threading.Thread(target=client._rlt_runtime.run, daemon=True, name="RLTLocalControl")]
    else:
        threads = [
            threading.Thread(
                target=client.receive_actions,
                daemon=True,
                name="RTCActionReceiver",
            ),
            threading.Thread(
                target=client.observation_sender,
                daemon=True,
                name="RTCObservationSender",
            ),
            threading.Thread(
                target=client.control_loop,
                daemon=True,
                name="RTCLocalControl",
            ),
        ]
    for thread in threads:
        thread.start()

    try:
        _run_interactive_prompt(client, _interactive_prompt)
    finally:
        client.stop()
        for thread in threads:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        if cfg.debug_visualize_queue_size and client.action_queue_size:
            visualize_action_queue_size(list(client.action_queue_size))
        client.logger.info("Deployment robot client stopped")

    if client.worker_error is not None:
        raise RuntimeError(f"RTC client stopped after a worker failure: {client.worker_error}")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
