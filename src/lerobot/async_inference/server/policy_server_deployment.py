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

"""Canonical deployable reset-aware remote-RTC policy server.

The inference service, guided RTC, rollout reset, and warm-policy cache live
here. Trained RTC delegates to the independent rtc_trained runtime. Neither
path imports historical policy_server or policy_server_rtc* implementations.

Example:

.. code-block:: shell

   python -m lerobot.async_inference.server.policy_server_deployment \
       --host=0.0.0.0 --port=8080 --fps=30 \
       --rtc.mode=guided --rtc.execution_horizon=30
"""

import gc
import logging
import math
import os
import pickle  # nosec
import threading
import time
import traceback
from collections import OrderedDict, deque
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.lerobot_types import PolicyAction
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig, RTCProcessor, reanchor_relative_rtc_prefix
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import receive_bytes_in_chunks

from ..shared.configs import PolicyServerConfigBase
from ..shared.constants import SUPPORTED_POLICIES
from ..shared.deployment_diagnostics import (
    configure_deployment_logging,
    is_fatal_cuda_error,
    server_fault_diagnostics,
)
from ..shared.helpers import (
    Observation,
    RemotePolicyConfig,
    TimedAction,
    get_logger,
)
from ..shared.image_transport import (
    IMAGE_TRANSPORT_METADATA_KEY,
    image_transport_metadata,
    negotiate_image_transport,
    prepare_remote_observation,
)
from ..shared.rlt_config import RLTServerConfig
from ..shared.rlt_remote import RLTSetup
from ..shared.rtc_remote import (
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    RemoteRTCActionChunk,
    RemoteRTCCalibrationRequest,
    RemoteRTCCalibrationResult,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
    RemoteRTCRetry,
    RTCActionRef,
    RTCPrefixUnavailableError,
    decode_rollout_generation,
    estimate_rtc_delay,
    supports_remote_rtc,
)
from ..shared.rtc_trained import TrainedRTCServerConfig, TrainedRTCServerRuntime
from .pi05_inference import PI05InferenceConfig, warmup_pi05
from .rtc_guidance import (
    LightweightGuidedRTCProcessor as _LightweightGuidedRTCProcessor,
)

_IDLE_POLL_SECONDS = 0.01
_JOIN_TIMEOUT_SECONDS = 5.0
_MAX_CONSECUTIVE_ERRORS = 10


def _default_deployment_rtc_config() -> RTCConfig:
    """Use the paper's exponential soft mask and conservative beta."""

    return RTCConfig(
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=5.0,
    )


class _SlidingWindowLatencyTracker:
    """Bounded samples with a nearest-rank quantile (1.0 is Algorithm 1's max)."""

    def __init__(self, maxlen: int, *, ttl_s: float = 300.0):
        if maxlen <= 0:
            raise ValueError("RTC delay buffer size must be positive")
        self._values: deque[float] = deque(maxlen=maxlen)
        self._ttl_s = ttl_s
        self._updated_at: float | None = None
        self._spike_guard = 0.0

    def reset(self) -> None:
        self._values.clear()
        self._updated_at = None
        self._spike_guard = 0.0

    def _expire(self) -> None:
        if self._updated_at is not None and time.monotonic() - self._updated_at > self._ttl_s:
            self.reset()

    def add(self, latency: float) -> None:
        value = float(latency)
        if math.isfinite(value) and value >= 0:
            self._expire()
            self._values.append(value)
            self._updated_at = time.monotonic()
            # Once a stall leaves the window it must no longer inflate the
            # budget. Otherwise a very large sample can leave an EMA tail
            # above every retained measurement for many further requests.
            self._spike_guard = min(max(self._values), max(value, 0.8 * self._spike_guard + 0.2 * value))

    def max(self) -> float:
        self._expire()
        return max(self._values, default=0.0)

    def estimate(self, probability: float) -> float:
        """Keep P90, but react to a real spike before it occupies 10% of the window."""
        return max(self.quantile(probability), self._spike_guard)

    def quantile(self, probability: float) -> float:
        self._expire()
        if not 0 < probability <= 1:
            raise ValueError("latency quantile must be in (0, 1]")
        if not self._values:
            return 0.0
        rank = math.ceil(probability * len(self._values)) - 1
        return sorted(self._values)[rank]

    def __len__(self) -> int:
        self._expire()
        return len(self._values)


class _BasePolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    """Shared policy state and timestamp helpers for the deployment RTC service."""

    logger: logging.Logger

    def __init__(self, config: PolicyServerConfigBase):
        self.config = config
        self.shutdown_event = threading.Event()

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self._image_transport_plan = {}

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]


@dataclass
class _RTCServerConfig(PolicyServerConfigBase):
    """Configuration for the remote RTC policy server."""

    rtc: RTCConfig = field(default_factory=_default_deployment_rtc_config)
    trained: TrainedRTCServerConfig = field(default_factory=TrainedRTCServerConfig)
    debug_rtc_diagnostics: bool = field(
        default=False,
        metadata={
            "help": "Enable RTC setup/inference phase traces, transport logs and per-chunk diagnostics"
        },
    )
    debug_cuda_sync: bool = field(
        default=False,
        metadata={
            "help": "Log and synchronize CUDA at inference/setup phase boundaries to locate native failures; slows inference"
        },
    )
    rtc_fast_guidance: bool = field(
        default=True,
        metadata={
            "help": "Use equivalent direct lightweight RTC correction and cached weights; false selects the reference path for A/B timing"
        },
    )
    rtc_delay_buffer_size: int = field(
        default=10,
        metadata={
            "help": (
                "Number of recent client RTT samples used for guided RTC conditioning. "
                "Server processing times are kept separately as a cold-start fallback."
            )
        },
    )
    rtc_delay_quantile: float = field(
        default=0.9,
        metadata={"help": "Guided RTT nearest-rank quantile; 1.0 restores a sliding-window maximum"},
    )
    rtc_delay_margin_steps: int = field(
        default=1,
        metadata={"help": "Extra guided conditioning steps beyond the RTT quantile, not a handover wait"},
    )
    latency_history_ttl_s: float = field(
        default=300.0,
        metadata={"help": "Retain timing estimates across rollout reset until this many idle seconds"},
    )
    action_wait_timeout: float = field(
        default=5.0,
        metadata={"help": "Long-poll timeout for the background action-chunk receiver"},
    )
    action_cache_size: int = field(
        default=10,
        metadata={"help": "Number of raw action chunks retained for RTC prefix reconstruction"},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_wait_timeout < 0:
            raise ValueError("action_wait_timeout must be non-negative")
        if self.action_cache_size < 1:
            raise ValueError("action_cache_size must be positive")
        if self.rtc_delay_buffer_size < 1:
            raise ValueError("rtc_delay_buffer_size must be positive")
        if not 0 < self.rtc_delay_quantile <= 1:
            raise ValueError("rtc_delay_quantile must be in (0, 1]")
        if self.rtc_delay_margin_steps < 0:
            raise ValueError("rtc_delay_margin_steps must be non-negative")
        if not math.isfinite(self.latency_history_ttl_s) or self.latency_history_ttl_s <= 0:
            raise ValueError("latency_history_ttl_s must be finite and positive")
        if not self.rtc.enabled:
            raise ValueError("policy_server_deployment requires rtc.enabled=true")
        if self.rtc.execution_horizon < 1:
            raise ValueError("rtc.execution_horizon must be positive")


@dataclass(frozen=True)
class _CachedActionChunk:
    original: torch.Tensor
    processed: torch.Tensor


class _ActionChunkCache:
    """Small LRU cache of model-space and executable chunks."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._chunks: OrderedDict[int, _CachedActionChunk] = OrderedDict()

    def clear(self) -> None:
        self._chunks.clear()

    def put(self, source_id: int, original: torch.Tensor, processed: torch.Tensor) -> None:
        self._chunks.pop(source_id, None)
        while len(self._chunks) >= self._max_size:
            self._chunks.popitem(last=False)
        self._chunks[source_id] = _CachedActionChunk(
            original=original.detach().cpu().clone(),
            processed=processed.detach().cpu().clone(),
        )

    def get(self, source_id: int) -> _CachedActionChunk | None:
        chunk = self._chunks.get(source_id)
        if chunk is not None:
            self._chunks.move_to_end(source_id)
        return chunk


def _publish_latest(queue: Queue, value: Any) -> None:
    """Publish without blocking, replacing an obsolete one-slot value."""

    try:
        queue.put_nowait(value)
        return
    except Full:
        pass

    with suppress(Empty):
        queue.get_nowait()

    # A consumer/producer race can fill the slot between get and put.  The
    # single producer will publish a newer value on its next update.
    with suppress(Full):
        queue.put_nowait(value)


class _RTCServer(_BasePolicyServer):
    """Policy server whose inference worker owns the complete RTC state machine."""

    def __init__(self, config: _RTCServerConfig):
        super().__init__(config)
        self._observation_mailbox: Queue[RemoteRTCObservation | RemoteRTCCalibrationRequest] = Queue(
            maxsize=1
        )
        self._action_mailbox: Queue[RemoteRTCActionChunk | RemoteRTCRetry | RemoteRTCCalibrationResult] = (
            Queue(maxsize=1)
        )
        self._trained_runtime: TrainedRTCServerRuntime | None = None
        self._chunk_cache = _ActionChunkCache(config.action_cache_size)
        self._rtc_stop_event = threading.Event()
        self._worker_failed = threading.Event()
        self._worker_traceback: str | None = None
        self._inference_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._observation_publish_lock = threading.Lock()
        self._latency_tracker = _SlidingWindowLatencyTracker(
            config.rtc_delay_buffer_size, ttl_s=config.latency_history_ttl_s
        )
        self._server_latency_tracker = _SlidingWindowLatencyTracker(
            config.rtc_delay_buffer_size, ttl_s=config.latency_history_ttl_s
        )
        self._worker_ready = threading.Event()
        self._last_latency_feedback_source = -1
        self._latest_received_observation_id = -1
        self._last_observation_id = -1
        self._relative_step: RelativeActionsProcessorStep | None = None
        self._normalizer_step: NormalizerProcessorStep | None = None
        self._setup: RemoteRTCPolicySetup | None = None
        self._session_sequence = 0
        # A failed CUDA context cannot be repaired by Ready, policy.reload, or
        # empty_cache. Retain this cause until the server process is restarted.
        self._fatal_cuda_error: str | None = None

    def _remember_fatal_cuda_error(self, error: BaseException) -> None:
        if self._fatal_cuda_error is None and is_fatal_cuda_error(error):
            self._fatal_cuda_error = f"{type(error).__name__}: {error}"
            self.shutdown_event.set()
            self.logger.critical(
                "CUDA context is unusable; restart the server process before reconnecting. First failure: %s",
                self._fatal_cuda_error,
            )

    def _raise_restart_required(self, context=None) -> None:
        if self._fatal_cuda_error is None:
            return
        details = f"Fatal CUDA error; restart the server process. First failure: {self._fatal_cuda_error}"
        if context is not None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, details)
        raise RuntimeError(details)

    def _diagnostic_cuda_sync(self, phase: str, boundary: str, device: str | None) -> None:
        if (
            not self.config.debug_cuda_sync
            or self._fatal_cuda_error is not None
            or device is None
            or torch.device(device).type != "cuda"
            or not torch.cuda.is_initialized()
        ):
            return
        self.logger.info("RTC CUDA SYNC begin phase=%s boundary=%s device=%s", phase, boundary, device)
        torch.cuda.synchronize(device)
        self.logger.info("RTC CUDA SYNC end phase=%s boundary=%s device=%s", phase, boundary, device)

    def _diagnostic_phase(self, phase: str, *, verbose: bool = False, device: str | None = None, **details):
        if not (self.config.debug_cuda_sync or self.config.debug_rtc_diagnostics):
            # Per-request errors are already handled by the inference loop.
            # Setup still needs the CUDA failure guard, without phase telemetry.
            return nullcontext() if verbose else self._guard_phase_errors(phase)
        return self._logged_diagnostic_phase(phase, device=device, **details)

    @contextmanager
    def _guard_phase_errors(self, phase: str) -> Iterator[None]:
        try:
            yield
        except RTCPrefixUnavailableError:
            raise
        except Exception as exc:
            self._remember_fatal_cuda_error(exc)
            self.logger.exception("RTC PHASE failed phase=%s session=%d", phase, self._session_sequence)
            raise

    @contextmanager
    def _logged_diagnostic_phase(self, phase: str, *, device: str | None = None, **details) -> Iterator[None]:
        session = self._session_sequence
        device = self.device if device is None else device
        started = time.perf_counter()
        self.logger.info(
            "RTC PHASE begin phase=%s session=%d pid=%d tid=%#x native_tid=%d "
            "thread=%s model=%s device=%s details=%s",
            phase,
            session,
            os.getpid(),
            threading.get_ident(),
            threading.get_native_id(),
            threading.current_thread().name,
            None if self.policy is None else hex(id(self.policy)),
            device,
            details,
        )
        with self._guard_phase_errors(phase):
            self._diagnostic_cuda_sync(phase, "before", device)
            yield
            self._diagnostic_cuda_sync(phase, "after", device)
        self.logger.info(
            "RTC PHASE end phase=%s session=%d elapsed_ms=%.1f",
            phase,
            session,
            (time.perf_counter() - started) * 1000,
        )

    def _configure_policy_rtc(self) -> None:
        """Configure native trained RTC or lightweight deployment guidance.

        Training-time RTC keeps the policy's native processor because it uses
        hard prefix conditioning and performs no inference-time VJP.  Guided
        RTC uses native lightweight guidance without denoiser-grad recording;
        policy weights and the model implementation remain unchanged.
        """

        if self.policy is None:
            raise RuntimeError("policy must be loaded before configuring RTC")

        self.policy.config.rtc_config = self.config.rtc
        training_max_delay = int(getattr(self.policy.config, "rtc_training_max_delay", 0))
        if self.config.rtc.mode == "trained" and training_max_delay <= 0:
            raise ValueError("rtc.mode='trained' requires a checkpoint with rtc_training_max_delay > 0")

        init_rtc = getattr(self.policy, "init_rtc_processor", None)
        if callable(init_rtc):
            init_rtc()

        current = getattr(self.policy, "rtc_processor", None)
        if self.config.rtc.mode != "guided" or not isinstance(current, RTCProcessor):
            return

        processor = _LightweightGuidedRTCProcessor(
            self.config.rtc,
            trained_mode_supported=training_max_delay > 0,
            fast=self.config.rtc_fast_guidance,
        )
        self.policy.rtc_processor = processor

        model = getattr(self.policy, "model", None)
        if model is not None and hasattr(model, "rtc_processor"):
            model.rtc_processor = processor

        self.logger.info(
            "Installed lightweight guided RTC processor (denoiser gradients disabled, "
            "prefix_schedule=%s, fast_guidance=%s)",
            self.config.rtc.prefix_attention_schedule.value,
            self.config.rtc_fast_guidance,
        )

    @property
    def failed(self) -> bool:
        return self._worker_failed.is_set()

    @property
    def failure_traceback(self) -> str | None:
        return self._worker_traceback

    def _reset_server(self) -> None:
        """Stop the old RTC worker and flush all per-session state."""

        self._rtc_stop_event.set()
        self.shutdown_event.set()
        thread = self._inference_thread
        if thread is not None and thread.is_alive():
            # Do not synchronize here: the old worker may still be inside
            # CUDA graph capture. Its own phase boundaries own synchronization.
            self.logger.info(
                "RTC worker join begin session=%d worker_tid=%s worker_native_tid=%s",
                self._session_sequence,
                None if thread.ident is None else hex(thread.ident),
                thread.native_id,
            )
            thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    f"RTC inference worker did not stop within {_JOIN_TIMEOUT_SECONDS:.1f}s; "
                    "refusing to mix two client sessions"
                )
            self.logger.info("RTC worker join end session=%d", self._session_sequence)
        self._inference_thread = None

        self._observation_mailbox = Queue(maxsize=1)
        self._action_mailbox = Queue(maxsize=1)
        self._chunk_cache = _ActionChunkCache(self.config.action_cache_size)
        with self._state_lock:
            self._latency_tracker.reset()
            self._server_latency_tracker.reset()
            self._last_latency_feedback_source = -1
        self._latest_received_observation_id = -1
        self._last_observation_id = -1
        self._relative_step = None
        self._normalizer_step = None
        self._setup = None
        self._trained_runtime = None
        if self._fatal_cuda_error is None:
            self._worker_failed.clear()
            self._worker_traceback = None
        self._worker_ready.clear()

    def GetActions(self, request, context):  # noqa: N802
        """Long-poll for and return one complete action chunk.

        This RPC is called by the client's background receiver.  The robot's
        per-tick control loop never calls it.
        """

        del request
        if getattr(self, "_rlt_runtime", None) is not None:
            result = self._rlt_runtime.get(context)
            return services_pb2.Actions(data=pickle.dumps(result)) if result is not None else services_pb2.Actions()
        self._raise_worker_failure(context)
        wait_timeout = self.config.action_wait_timeout
        if context is not None:
            if not context.is_active():
                return services_pb2.Actions()
            remaining = context.time_remaining()
            if remaining is not None:
                if remaining <= 0:
                    return services_pb2.Actions()
                # A client may use a shorter deadline than our long poll.
                # Do not leave expired RPC handlers waiting to consume the
                # replacement request's result after the receiver reconnects.
                wait_timeout = min(wait_timeout, remaining)
        try:
            chunk = self._action_mailbox.get(timeout=wait_timeout)
        except Empty:
            self._raise_worker_failure(context)
            return services_pb2.Actions()
        return services_pb2.Actions(data=pickle.dumps(chunk))  # nosec

    def _raise_worker_failure(self, context) -> None:
        self._raise_restart_required(context)
        if not self.failed:
            return
        details = "Remote RTC inference worker failed"
        if context is not None:
            context.abort(grpc.StatusCode.INTERNAL, details)
        raise RuntimeError(details)

    def _reconstruct_prefix(
        self, envelope: RemoteRTCObservation
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Rebuild model-space and executable prefixes from cache references."""

        refs = envelope.action_refs[: self.config.rtc.execution_horizon]
        if not refs:
            return None, None

        original: list[torch.Tensor] = []
        processed: list[torch.Tensor] = []
        for ref in refs:
            resolved = self._resolve_action_ref(ref)
            if resolved is None:
                raise RTCPrefixUnavailableError("committed action reference is missing from the server cache")
            original_action, processed_action = resolved
            original.append(original_action)
            processed.append(processed_action)

        return torch.stack(original), torch.stack(processed)

    def _resolve_action_ref(
        self,
        ref: RTCActionRef,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Resolve one unchanged action row in model and executable coordinates."""

        cached = self._chunk_cache.get(ref.source_observation_id)
        if cached is None:
            if self.config.debug_rtc_diagnostics:
                self.logger.debug("RTC cache miss for source observation %d", ref.source_observation_id)
            return None
        if ref.action_index >= len(cached.original) or ref.action_index >= len(cached.processed):
            self.logger.warning(
                "Ignoring invalid RTC action reference source=%d index=%d",
                ref.source_observation_id,
                ref.action_index,
            )
            return None

        return cached.original[ref.action_index], cached.processed[ref.action_index]

    def _current_latency(self) -> float:
        with self._state_lock:
            if self._latency_tracker:
                return self._latency_tracker.estimate(self.config.rtc_delay_quantile)
            return self._server_latency_tracker.max()

    def _record_latency(self, latency: float) -> None:
        with self._state_lock:
            self._server_latency_tracker.add(latency)

    def _postprocess_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        # Match RTCInferenceEngine: process the whole trajectory in one call so
        # chunk-aware/stateful processor steps see the same tensor shape as local
        # RTC.  The ordinary async server's per-action loop predates that engine.
        processed = self.postprocessor(actions)
        if not isinstance(processed, torch.Tensor):
            raise TypeError(f"postprocessor must return a Tensor, got {type(processed)}")
        if processed.ndim == 3 and processed.shape[0] == 1:
            processed = processed.squeeze(0)
        if processed.ndim != 2:
            raise ValueError(f"postprocessed action chunk must have shape (T, A), got {processed.shape}")
        return processed.detach().cpu()

    def _predict_remote_chunk(self, envelope: RemoteRTCObservation) -> RemoteRTCActionChunk | RemoteRTCRetry:
        if self.preprocessor is None or self.postprocessor is None or self.policy is None:
            raise RuntimeError("policy and processors must be initialized before RTC inference")
        if self.config.rtc.mode == "trained":
            if self._trained_runtime is None:
                raise RuntimeError("trained runtime has not been configured by policy setup")
            return self._trained_runtime.predict(envelope)
        if envelope.trained_prefix is not None:
            raise ValueError("guided inference cannot consume a trained prefix payload")

        started = time.perf_counter()
        timed_observation = envelope.observation
        observation: Observation = prepare_remote_observation(
            timed_observation.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self._image_transport_plan,
        )
        with self._diagnostic_phase("preprocess", verbose=True):
            preprocessed = self.preprocessor(observation)
        if self.config.debug_rtc_diagnostics:
            preprocess_finished = time.perf_counter()

        previous_original, previous_processed = self._reconstruct_prefix(envelope)
        prefix_length = 0 if previous_original is None else len(previous_original)
        has_previous_actions = prefix_length > 0
        previous_actions = previous_original

        if has_previous_actions and self._relative_step is not None:
            raw_state = self._relative_step.get_cached_state()
            if raw_state is not None and previous_processed is not None:
                previous_actions = reanchor_relative_rtc_prefix(
                    prev_actions_absolute=previous_processed,
                    current_state=raw_state,
                    relative_step=self._relative_step,
                    normalizer_step=self._normalizer_step,
                    policy_device=self.device,
                )

        policy_config = self.policy.config
        training_max_delay = int(getattr(policy_config, "rtc_training_max_delay", 0))
        conditioned_delay = estimate_rtc_delay(
            latency=self._current_latency(),
            time_per_step=self.config.environment_dt,
            mode=self.config.rtc.mode,
            training_max_delay=training_max_delay,
            has_previous_actions=has_previous_actions,
        )
        if has_previous_actions:
            conditioned_delay = min(
                conditioned_delay + self.config.rtc_delay_margin_steps,
                prefix_length,
                self.actions_per_chunk - 1,
            )
        else:
            conditioned_delay = 0

        if previous_actions is not None:
            # RTC Eq. (5) guides only timesteps that genuinely overlap the old
            # and new chunks.  Do not repeat the final real action to fabricate
            # a longer prefix: RTCProcessor already zero-pads tensor shape while
            # clamping the mask horizon to this prefix's actual length.
            previous_actions = previous_actions.to(self.device)

        if self.config.debug_rtc_diagnostics:
            prefix_finished = time.perf_counter()
        with self._diagnostic_phase("predict_action_chunk", verbose=True):
            actions = self.policy.predict_action_chunk(
                preprocessed,
                inference_delay=conditioned_delay,
                prev_chunk_left_over=previous_actions,
            )
        if self.config.debug_rtc_diagnostics:
            prediction_finished = time.perf_counter()
        if actions.ndim != 3:
            actions = actions.unsqueeze(0)
        actions = actions[:, : self.actions_per_chunk, :]
        if actions.shape[0] != 1:
            raise ValueError(f"remote RTC expects batch size 1, got {actions.shape[0]}")

        original = actions.squeeze(0).detach()
        with self._diagnostic_phase("postprocess", verbose=True):
            processed = self._postprocess_chunk(actions)
        if self.config.debug_rtc_diagnostics:
            postprocess_finished = time.perf_counter()

        if (
            self.config.debug_rtc_diagnostics
            and self.logger.isEnabledFor(logging.DEBUG)
            and previous_processed is not None
            and prefix_length > 0
        ):
            compare_n = min(
                prefix_length,
                len(previous_processed),
                len(processed),
            )

            if compare_n > 0:
                old_exec = previous_processed[:compare_n].detach().cpu()
                new_exec = processed[:compare_n].detach().cpu()

                delta = new_exec - old_exec
                per_step_max = delta.abs().amax(dim=-1)
                per_step_l2 = torch.linalg.vector_norm(delta, dim=-1)

                start_step = max(timed_observation.get_timestep(), 0)

                self.logger.debug(
                    "RTC ALIGN obs=%d start_step=%d prefix=%d conditioned_delay=%d "
                    "first_max=%.6f first_l2=%.6f overall_max=%.6f mean_l2=%.6f",
                    envelope.observation_id,
                    start_step,
                    compare_n,
                    conditioned_delay,
                    per_step_max[0].item(),
                    per_step_l2[0].item(),
                    per_step_max.max().item(),
                    per_step_l2.mean().item(),
                )

                # print first 8 overlapping absolute timesteps
                for i in range(min(compare_n, 8)):
                    step = start_step + i
                    self.logger.debug(
                        "RTC ALIGN STEP=%d idx=%d max=%.6f l2=%.6f old=%s new=%s delta=%s",
                        step,
                        i,
                        per_step_max[i].item(),
                        per_step_l2[i].item(),
                        old_exec[i].tolist(),
                        new_exec[i].tolist(),
                        delta[i].tolist(),
                    )

                # Inspect the edge of the explicitly conditioned region.  The
                # client no longer treats this as a hard ownership handover;
                # its real splice is determined later from consumed actions.
                conditioned_edge_idx = min(conditioned_delay, compare_n - 1)
                conditioned_edge_kind = (
                    "first_free" if conditioned_delay < compare_n else "last_available_prefix"
                )

                self.logger.debug(
                    "RTC CONDITIONED EDGE kind=%s step=%d idx=%d max_delta=%.6f l2=%.6f old=%s new=%s",
                    conditioned_edge_kind,
                    start_step + conditioned_edge_idx,
                    conditioned_edge_idx,
                    per_step_max[conditioned_edge_idx].item(),
                    per_step_l2[conditioned_edge_idx].item(),
                    old_exec[conditioned_edge_idx].tolist(),
                    new_exec[conditioned_edge_idx].tolist(),
                )

        elapsed = time.perf_counter() - started
        remaining_latency = self.config.inference_latency - elapsed
        if remaining_latency > 0:
            time.sleep(remaining_latency)
        inference_seconds = time.perf_counter() - started
        measured_delay = math.ceil(inference_seconds / self.config.environment_dt)
        if self.config.debug_rtc_diagnostics:
            self.logger.debug(
                "RTC SERVER PHASES obs=%d total_ms=%.3f preprocess_ms=%.3f prefix_ms=%.3f "
                "predict_ms=%.3f postprocess_ms=%.3f postprocess_to_return_ms=%.3f "
                "prefix=%d conditioned_delay=%d measured_delay=%d",
                envelope.observation_id,
                inference_seconds * 1000,
                (preprocess_finished - started) * 1000,
                (prefix_finished - preprocess_finished) * 1000,
                (prediction_finished - prefix_finished) * 1000,
                (postprocess_finished - prediction_finished) * 1000,
                max(0.0, inference_seconds - (postprocess_finished - started)) * 1000,
                prefix_length,
                conditioned_delay,
                measured_delay,
            )

        # Keep this lower bound separate from client RTT. Mixing the two would
        # count one request twice, underweight transport/queueing and evict RTT
        # spikes after only half the configured number of requests.
        self._record_latency(inference_seconds)

        self._chunk_cache.put(envelope.observation_id, original, processed)

        chunk_start_step = max(timed_observation.get_timestep(), 0)
        timed_actions = self._time_action_chunk(
            timed_observation.get_timestamp(),
            list(processed),
            chunk_start_step,
        )
        return RemoteRTCActionChunk(
            source_observation_id=envelope.observation_id,
            chunk_start_step=chunk_start_step,
            observation_timestamp=timed_observation.get_timestamp(),
            actions=timed_actions,
            conditioned_delay=conditioned_delay,
            prefix_length=prefix_length,
            inference_seconds=inference_seconds,
            rtc_mode=self.config.rtc.mode,
            training_max_delay=training_max_delay,
        )

    def _initialize_worker(self) -> None:
        """Optional startup work, always on the thread that runs real inference."""

    def _inference_loop(self, worker_stop_event: threading.Event) -> None:
        consecutive_errors = 0
        pending: RemoteRTCObservation | RemoteRTCCalibrationRequest | None = None
        try:
            with self._diagnostic_phase("worker_initialize"):
                self._initialize_worker()
            self._worker_ready.set()
            while self.running and not worker_stop_event.is_set():
                if pending is None:
                    try:
                        pending = self._observation_mailbox.get(timeout=_IDLE_POLL_SECONDS)
                    except Empty:
                        continue

                if isinstance(pending, RemoteRTCCalibrationRequest):
                    with self._diagnostic_phase("calibration", sequence=pending.sequence):
                        result = self._calibrate(pending)
                    with self._observation_publish_lock:
                        self._calibration_pending = False
                        if self.running and not worker_stop_event.is_set():
                            _publish_latest(self._action_mailbox, result)
                    pending = None
                    continue

                # Only run the freshest queued snapshot. Every attempted request
                # finishes with actions or an explicit request for a NEW snapshot.
                while True:
                    try:
                        candidate = self._observation_mailbox.get_nowait()
                    except Empty:
                        break
                    if candidate.observation_id > pending.observation_id:
                        pending = candidate
                if pending.observation_id <= self._last_observation_id:
                    pending = None
                    continue

                try:
                    with self._diagnostic_phase(
                        "inference", verbose=True, observation_id=pending.observation_id
                    ):
                        chunk = self._predict_remote_chunk(pending)
                    if chunk is None:
                        raise RuntimeError("inference returned neither actions nor a retry response")
                    if self.running and not worker_stop_event.is_set():
                        _publish_latest(self._action_mailbox, chunk)
                    self._last_observation_id = pending.observation_id
                    pending = None
                    consecutive_errors = 0
                except RTCPrefixUnavailableError as exc:
                    self.logger.warning("Requesting fresh RTC prefix: %s", exc)
                    self._publish_snapshot_retry(pending, "prefix_unavailable", worker_stop_event)
                    pending = None
                except Exception as exc:
                    self._remember_fatal_cuda_error(exc)
                    if self._fatal_cuda_error is not None:
                        raise
                    consecutive_errors += 1
                    self.logger.error(
                        "Remote RTC inference error (%d/%d): %s",
                        consecutive_errors,
                        _MAX_CONSECUTIVE_ERRORS,
                        exc,
                        exc_info=True,
                    )
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                    self._publish_snapshot_retry(pending, "inference_error", worker_stop_event)
                    pending = None
        except Exception as exc:
            self._remember_fatal_cuda_error(exc)
            self._worker_traceback = traceback.format_exc()
            self._worker_failed.set()
            self.shutdown_event.set()
            self.logger.error("Remote RTC inference worker stopped: %s", exc)
            self.logger.error(self._worker_traceback)
        finally:
            self.logger.info(
                "RTC worker exiting session=%d tid=%#x native_tid=%d failed=%s restart_required=%s",
                self._session_sequence,
                threading.get_ident(),
                threading.get_native_id(),
                self.failed,
                self._fatal_cuda_error is not None,
            )
            # Also wake setup on failure; it checks _worker_failed before success.
            self._worker_ready.set()

    def _publish_snapshot_retry(
        self, request: RemoteRTCObservation, reason: str, worker_stop_event: threading.Event
    ) -> None:
        if self.running and not worker_stop_event.is_set():
            _publish_latest(
                self._action_mailbox,
                RemoteRTCRetry(
                    request.observation_id,
                    request.observation.get_timestamp(),
                    reason,
                    rtc_mode=self.config.rtc.mode,
                ),
            )
        self._last_observation_id = request.observation_id

    def stop(self) -> None:
        self._reset_server()
        self.logger.info("Remote RTC policy server stopping")


@dataclass
class PolicyServerDeploymentConfig(_RTCServerConfig):
    """Resettable RTC server settings plus a one-entry warm model cache."""

    rlt: RLTServerConfig = field(default_factory=RLTServerConfig)

    cuda_visible_devices: str | None = field(
        default=None,
        metadata={
            "help": "CUDA_VISIBLE_DEVICES override before CUDA initialization, e.g. '6' or '4,5'; "
            "null preserves the process environment"
        },
    )
    inference_latency: float = field(
        default=0.0,
        metadata={
            "help": "Optional minimum request duration for latency simulation; 0 disables artificial delay"
        },
    )
    pi05_inference: PI05InferenceConfig = field(default_factory=PI05InferenceConfig)
    enable_fault_handler: bool = field(
        default=True,
        metadata={"help": "Keep a crash log and dump Python thread stacks on fatal signals; no CUDA sync"},
    )
    fault_log_path: Path | None = field(
        default=None,
        metadata={"help": "Fatal traces and server phase logs; defaults to logs/<server>_fault_<pid>.log"},
    )

    reuse_last_policy: bool = field(
        default=True,
        metadata={
            "help": (
                "Keep the most recently loaded policy on its device and reuse it when a new "
                "client requests the same policy type, checkpoint, and device."
            )
        },
    )


@dataclass(frozen=True)
class _ModelCacheKey:
    policy_type: str
    pretrained_name_or_path: str
    device: str
    pi05_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ProcessorCacheKey:
    model: _ModelCacheKey
    rename_map: tuple[tuple[str, str], ...]


class PolicyServerDeployment(_RTCServer):
    """Policy server with rollout generations and a one-entry warm policy cache."""

    prefix = "policy_server_deployment"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerDeploymentConfig):
        if isinstance(config, _RTCServerConfig) and not isinstance(config, PolicyServerDeploymentConfig):
            # Historical callers construct the common RTC config directly.
            config = PolicyServerDeploymentConfig(**{
                item.name: getattr(config, item.name) for item in fields(config) if item.init
            })
        # Select GPUs before the server can load a policy or initialize CUDA.
        if config.cuda_visible_devices is not None:
            if (
                torch.cuda.is_initialized()
                and os.environ.get("CUDA_VISIBLE_DEVICES") != config.cuda_visible_devices
            ):
                raise RuntimeError(
                    "Cannot change CUDA_VISIBLE_DEVICES after CUDA initialization. "
                    "Restart the server with cuda_visible_devices set in its configuration."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        super().__init__(config)
        self._rlt_runtime = None
        self._active_rollout_generation = -1
        self._policy_setup_lock = threading.Lock()
        self._cached_model_key: _ModelCacheKey | None = None
        self._cached_processor_key: _ProcessorCacheKey | None = None
        self.model_cache_hits = 0
        self.model_loads = 0
        self._horizon_feedback_generation = -1
        self._consecutive_horizon_overruns = 0
        self._horizon_warning_emitted_generation = -1
        self._calibration_open = False
        self._calibration_pending = False
        self._calibration_next_sequence = 0
        self._calibration_measured_source: int | None = None
        self._calibration_previous: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def active_rollout_generation(self) -> int:
        return self._active_rollout_generation

    def _reset_server(self) -> None:
        """Forget the rollout, but deliberately retain the warm policy runtime."""

        runtime = getattr(self, "_rlt_runtime", None)
        if runtime is not None:
            runtime.close()
            self._rlt_runtime = None
        super()._reset_server()
        self._active_rollout_generation = -1
        self._horizon_feedback_generation = -1
        self._consecutive_horizon_overruns = 0
        self._horizon_warning_emitted_generation = -1
        self._calibration_open = False
        self._calibration_pending = False
        self._calibration_previous = None

    def _check_execution_horizon_feedback(self, generation: int, latency_seconds: float) -> None:
        """Warn once when normal guided RTT repeatedly exceeds the configured horizon."""

        if self.config.rtc.mode != "guided":
            return
        if generation != self._horizon_feedback_generation:
            self._horizon_feedback_generation = generation
            self._consecutive_horizon_overruns = 0

        latency_steps = math.ceil(latency_seconds / self.config.environment_dt)
        if latency_steps <= self.config.rtc.execution_horizon:
            self._consecutive_horizon_overruns = 0
            return

        self._consecutive_horizon_overruns += 1
        if self._consecutive_horizon_overruns < 2 or self._horizon_warning_emitted_generation == generation:
            return

        self._horizon_warning_emitted_generation = generation
        suggested_horizon = latency_steps + 5
        maximum_horizon = max(1, self.actions_per_chunk - 1)
        self.logger.warning(
            "RTC execution_horizon=%d is below repeated measured RTT=%d steps; "
            "guided conditioning is being prefix-limited. Restart with "
            "--rtc.execution_horizon=%d or larger (and actions_per_chunk greater than it)%s",
            self.config.rtc.execution_horizon,
            latency_steps,
            min(suggested_horizon, maximum_horizon),
            "" if suggested_horizon <= maximum_horizon else "; increase actions_per_chunk as well",
        )

    def Ready(self, request, context):  # noqa: N802
        """Serialize reconnect teardown against warm-policy installation."""

        with self._policy_setup_lock:
            self._raise_restart_required(context)
            self.logger.info(
                "RTC Ready begin peer=%s session=%d",
                context.peer() if context is not None else None,
                self._session_sequence,
            )
            self._reset_server()
            # An in-flight inference may have poisoned CUDA while reset was
            # joining the old worker. Never clear that process-level failure.
            self._raise_restart_required(context)
            self.shutdown_event.clear()
            self.logger.info("RTC Ready end session=%d", self._session_sequence)
            return services_pb2.Empty()

    @staticmethod
    def _canonical_pretrained_reference(reference: str) -> str:
        """Canonicalize an existing local checkpoint path for stable cache hits."""

        path = Path(reference).expanduser()
        if path.exists():
            return str(path.resolve())
        return reference

    def _model_cache_key(self, policy_specs: RemotePolicyConfig) -> _ModelCacheKey:
        inference = self.config.pi05_inference
        if policy_specs.policy_type == "pi05":
            overrides = inference.overrides()
        else:
            if (
                inference.profile != "checkpoint"
                or inference.compile_model is not None
                or inference.compile_mode is not None
            ):
                raise ValueError("pi05_inference overrides require policy_type=pi05")
            overrides = ()
        return _ModelCacheKey(
            policy_type=policy_specs.policy_type,
            pretrained_name_or_path=self._canonical_pretrained_reference(
                policy_specs.pretrained_name_or_path
            ),
            device=policy_specs.device,
            pi05_overrides=overrides,
        )

    @staticmethod
    def _processor_cache_key(
        model_key: _ModelCacheKey,
        policy_specs: RemotePolicyConfig,
    ) -> _ProcessorCacheKey:
        return _ProcessorCacheKey(
            model=model_key,
            rename_map=tuple(sorted(policy_specs.rename_map.items())),
        )

    def _drop_cached_policy_runtime(self) -> None:
        """Release the previous one-entry cache before loading a different model."""

        self._raise_restart_required()
        old_device = self._cached_model_key.device if self._cached_model_key is not None else self.device
        with self._diagnostic_phase("runtime_release", device=old_device):
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            self._relative_step = None
            self._normalizer_step = None
            self._cached_model_key = None
            self._cached_processor_key = None
            with self._diagnostic_phase("gc_collect", device=old_device):
                gc.collect()
            if torch.cuda.is_available():
                with self._diagnostic_phase("cuda_empty_cache", device=old_device):
                    torch.cuda.empty_cache()

    def _build_processors(self, policy_specs: RemotePolicyConfig) -> None:
        device_override = {"device": policy_specs.device}
        with self._diagnostic_phase("processor_build"):
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                },
                postprocessor_overrides={"device_processor": device_override},
            )

    def _load_or_reuse_policy(self, policy_specs: RemotePolicyConfig) -> bool:
        """Install a policy runtime and return whether its weights were reused."""

        self._raise_restart_required()
        model_key = self._model_cache_key(policy_specs)
        processor_key = self._processor_cache_key(model_key, policy_specs)
        cache_hit = (
            getattr(self.config, "reuse_last_policy", True)
            and self.policy is not None
            and self._cached_model_key == model_key
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        if cache_hit:
            if (
                self.preprocessor is None
                or self.postprocessor is None
                or self._cached_processor_key != processor_key
            ):
                self._build_processors(policy_specs)
                self._cached_processor_key = processor_key
            self.model_cache_hits += 1
            return True

        if self.policy is not None:
            self._drop_cached_policy_runtime()

        with self._diagnostic_phase("policy_load", checkpoint=policy_specs.pretrained_name_or_path):
            policy_class = get_policy_class(policy_specs.policy_type)
            if policy_specs.policy_type == "pi05":
                # Apply before construction/loading: changing compile_model on an
                # already initialized config would not compile sample_actions.
                # Even the checkpoint profile must use the requested device here:
                # PI05's constructor already moves parameters to config.device.
                policy_config = policy_class.config_class.from_pretrained(
                    policy_specs.pretrained_name_or_path,
                    cli_overrides=[*model_key.pi05_overrides, f"--device={policy_specs.device}"],
                )
                policy = policy_class.from_pretrained(
                    policy_specs.pretrained_name_or_path, config=policy_config
                )
            else:
                policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        with self._diagnostic_phase("policy_to_device"):
            policy.to(policy_specs.device)
        self.policy = policy
        self._build_processors(policy_specs)
        self._cached_model_key = model_key
        self._cached_processor_key = processor_key
        self.model_loads += 1
        return False

    def _reset_processors(self) -> None:
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _initialize_worker(self) -> None:
        self._warmup_pi05()

    def _warmup_pi05(self) -> None:
        """Warm eager/compiled inference and processors on the serving worker."""
        if self.policy_type != "pi05":
            return
        config = self.policy.config
        self.logger.info(
            "PI05 inference dtype=%s precision_profile=%s compile=%s mode=%s "
            "Gemma_attention=eager matmul_precision=%s",
            getattr(config, "dtype", "unknown"),
            getattr(config, "precision_profile", "lerobot"),
            getattr(config, "compile_model", False),
            getattr(config, "compile_mode", "max-autotune"),
            torch.get_float32_matmul_precision(),
        )
        if self.config.pi05_inference.warmup_runs == 0:
            return
        if self._trained_runtime is not None:
            prefix_length = delay = self._trained_runtime.delay_steps
        else:
            prefix_length = delay = min(self.config.rtc.execution_horizon, config.chunk_size - 1)
        raw = {"task": "Warm up the policy."}
        for key, feature in self.lerobot_features.items():
            if key.startswith("observation.images."):
                raw[key.removeprefix("observation.images.")] = np.zeros(feature["shape"], dtype=np.uint8)
            elif key.startswith("observation.") and feature["dtype"] == "float32":
                raw.update(dict.fromkeys(feature["names"], 0.0))

        def prepare_batch():
            observation = prepare_remote_observation(
                raw, self.lerobot_features, self.policy_image_features, self._image_transport_plan
            )
            return self.preprocessor(observation)

        warmup_pi05(
            self.policy,
            runs=self.config.pi05_inference.warmup_runs,
            prefix_length=prefix_length,
            inference_delay=delay,
            logger=self.logger,
            prepare_batch=prepare_batch,
            postprocess=lambda actions: self._postprocess_chunk(actions[:, : self.actions_per_chunk]),
            reset_processors=self._reset_processors,
            cancelled=lambda: not self.running or self._rtc_stop_event.is_set(),
        )

    def _calibrate(self, request: RemoteRTCCalibrationRequest) -> RemoteRTCCalibrationResult:
        """Dry-run the full path on the serving worker, outside rollout caches."""
        if request.feedback_source != self._calibration_measured_source or (
            request.feedback_seconds is None
        ) != (request.feedback_source is None):
            raise ValueError("calibration feedback must acknowledge the previous measured result")
        if request.feedback_seconds is not None:
            if not math.isfinite(request.feedback_seconds) or request.feedback_seconds < 0:
                raise ValueError("invalid calibration RTT")
            with self._state_lock:
                self._latency_tracker.add(request.feedback_seconds)
        self._calibration_measured_source = None
        if request.observation is None:
            self._reset_processors()
            self.policy.reset()
            processor = getattr(self.policy, "rtc_processor", None)
            if processor is not None:
                processor.reset_tracker()
            self._calibration_previous = None
            self._calibration_open = False
            self.logger.info(
                "RTC calibration ready: initial RTT budget=%.1f ms", self._current_latency() * 1000
            )
            return RemoteRTCCalibrationResult(request.token, request.sequence)

        started = time.perf_counter()
        observation = prepare_remote_observation(
            request.observation.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self._image_transport_plan,
        )
        with self._diagnostic_phase("calibration_preprocess", verbose=True):
            preprocessed = self.preprocessor(observation)
        prefix = None
        delay = 0
        if request.use_prefix:
            if self._calibration_previous is None:
                raise ValueError("prefix calibration requires an ordinary warmup result first")
            original, absolute = self._calibration_previous
            delay = (
                self._trained_runtime.delay_steps
                if self._trained_runtime is not None
                else min(self.config.rtc.execution_horizon, self.actions_per_chunk - 1)
            )
            prefix = original[:delay].to(self.device)
            if self._relative_step is not None:
                prefix = reanchor_relative_rtc_prefix(
                    prev_actions_absolute=absolute[:delay],
                    current_state=self._relative_step.get_cached_state(),
                    relative_step=self._relative_step,
                    normalizer_step=self._normalizer_step,
                    policy_device=self.device,
                )
        if torch.device(self.device).type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        # Guided RTC's local VJP must remain able to enable gradients.
        with self._diagnostic_phase("calibration_predict", verbose=True):
            actions = self.policy.predict_action_chunk(
                preprocessed, inference_delay=delay, prev_chunk_left_over=prefix
            )
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < self.actions_per_chunk:
            raise ValueError("calibration requires a complete [1, H, A] action chunk")
        actions = actions[:, : self.actions_per_chunk]
        with self._diagnostic_phase("calibration_postprocess", verbose=True):
            processed = self._postprocess_chunk(actions)
        if not torch.isfinite(processed).all().item():
            raise RuntimeError("calibration produced non-finite actions")
        # Own CPU storage; no CUDA graph output can survive into the next call.
        self._calibration_previous = (actions.squeeze(0).detach().cpu().clone(), processed.clone())
        remaining = self.config.inference_latency - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        elapsed = time.perf_counter() - started
        if not request.warmup:
            self._record_latency(elapsed)
            self._calibration_measured_source = request.sequence
        chunk = RemoteRTCActionChunk(
            source_observation_id=request.sequence,
            chunk_start_step=0,
            observation_timestamp=request.observation.get_timestamp(),
            actions=self._time_action_chunk(request.observation.get_timestamp(), list(processed), 0),
            conditioned_delay=delay,
            prefix_length=0 if prefix is None else len(prefix),
            inference_seconds=elapsed,
            rtc_mode=self.config.rtc.mode,
            training_max_delay=int(getattr(self.policy.config, "rtc_training_max_delay", 0)),
        )
        self.logger.info(
            "RTC calibration %s sequence=%d prefix=%d server_ms=%.1f",
            "warmup" if request.warmup else "measure",
            request.sequence,
            chunk.prefix_length,
            elapsed * 1000,
        )
        return RemoteRTCCalibrationResult(request.token, request.sequence, chunk)

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Reuse matching warm weights, configure RTC, and start one worker."""

        self._raise_restart_required(context)
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        setup = pickle.loads(request.data)  # nosec
        if isinstance(setup, RLTSetup):
            from .rlt_runtime import RemoteRLTServer

            with self._policy_setup_lock:
                if self._inference_thread is not None or self._rlt_runtime is not None:
                    raise RuntimeError("Call Ready before installing an RLT worker")
                with self._diagnostic_phase("rlt_policy_runtime", device=setup.policy.device):
                    self._rlt_runtime = RemoteRLTServer(self, setup, context)
            return services_pb2.Empty()
        if self.config.rlt.enabled:
            raise ValueError("RLT server requires an RLT client setup")
        if not isinstance(setup, RemoteRTCPolicySetup):
            raise TypeError(f"RTC policy setup must be RemoteRTCPolicySetup, got {type(setup)}")
        if setup.protocol_version != DEPLOYMENT_RTC_PROTOCOL_VERSION:
            raise ValueError("deployment RTC protocol changed; update both deployment client and server")
        if setup.rtc_mode != self.config.rtc.mode:
            raise ValueError("client --rtc_mode must match server --rtc.mode")
        if not math.isclose(setup.control_frequency_hz, self.config.fps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "RTC client/server control frequencies must match: "
                f"client={setup.control_frequency_hz:g}Hz, server={self.config.fps:g}Hz"
            )
        policy_specs = setup.policy
        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )
        if (
            self.config.rtc.mode == "guided"
            and policy_specs.actions_per_chunk <= self.config.rtc.execution_horizon
        ):
            raise ValueError(
                "actions_per_chunk must be greater than rtc.execution_horizon "
                f"({policy_specs.actions_per_chunk} <= {self.config.rtc.execution_horizon})"
            )

        with self._policy_setup_lock:
            self._raise_restart_required(context)
            if self._inference_thread is not None and self._inference_thread.is_alive():
                raise RuntimeError("call Ready before replacing the active policy worker")
            self._session_sequence += 1
            started = time.perf_counter()
            with self._diagnostic_phase("policy_runtime", device=policy_specs.device):
                cache_hit = self._load_or_reuse_policy(policy_specs)
            self.logger.info(
                "RTC runtime installed session=%d cache_hit=%s model=%#x device=%s "
                "compile=%s compile_mode=%s",
                self._session_sequence,
                cache_hit,
                id(self.policy),
                self.device,
                getattr(self.policy.config, "compile_model", False),
                getattr(self.policy.config, "compile_mode", None),
            )

            if not supports_remote_rtc(self.policy):
                raise ValueError(
                    f"Policy type {policy_specs.policy_type!r} does not implement RTC-compatible "
                    "predict_action_chunk(observation, inference_delay=..., prev_chunk_left_over=...)"
                )
            self._setup = setup
            with self._diagnostic_phase("rtc_configure"):
                self._configure_policy_rtc()
                self.policy.eval()
            self._image_transport_plan = {}
            if getattr(setup, "client_image_resize", False) and context is not None:
                image_mode = getattr(setup, "image_transport_mode", "lerobot_float32")
                self._image_transport_plan = negotiate_image_transport(
                    policy_specs.policy_type,
                    self.policy.config,
                    self.lerobot_features,
                    self.preprocessor,
                    mode=image_mode,
                )
                if image_mode == "openpi_uint8" and not self._image_transport_plan:
                    raise ValueError("openpi_uint8 requires a supported Pi05 RGB preprocessing pipeline")
                context.send_initial_metadata(
                    ((IMAGE_TRANSPORT_METADATA_KEY, image_transport_metadata(self._image_transport_plan)),)
                )
                self.logger.info("Negotiated client image resize: %s", self._image_transport_plan)

            self._relative_step = next(
                (
                    step
                    for step in getattr(self.preprocessor, "steps", ())
                    if isinstance(step, RelativeActionsProcessorStep) and step.enabled
                ),
                None,
            )
            self._normalizer_step = next(
                (
                    step
                    for step in getattr(self.preprocessor, "steps", ())
                    if isinstance(step, NormalizerProcessorStep)
                ),
                None,
            )
            if self._relative_step is not None:
                configured_names = getattr(self.policy.config, "action_feature_names", None)
                self._relative_step.action_names = list(configured_names or setup.action_feature_names)

            self._trained_runtime = None
            if self.config.rtc.mode == "trained":
                if policy_specs.policy_type != "pi05" or setup.trained_queue_threshold is None:
                    raise ValueError("trained deployment requires Pi05 and a trained client queue threshold")
                self._trained_runtime = TrainedRTCServerRuntime(
                    config=self.config.trained,
                    policy=self.policy,
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    features=self.lerobot_features,
                    device=self.device,
                    fps=self.config.fps,
                    chunk_size=self.actions_per_chunk,
                    queue_threshold=setup.trained_queue_threshold,
                    minimum_inference_seconds=self.config.inference_latency,
                    image_transport_plan=self._image_transport_plan,
                )

            self._calibration_open = setup.calibration_token is not None
            self._calibration_pending = False
            self._calibration_next_sequence = 0
            self._calibration_measured_source = None
            self._calibration_previous = None
            self._worker_ready.clear()
            self._rtc_stop_event = threading.Event()
            worker_stop_event = self._rtc_stop_event
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                args=(worker_stop_event,),
                daemon=True,
                name="RemoteRTCInference",
            )
            self._inference_thread.start()
            self.logger.info(
                "RTC worker started session=%d worker_tid=%s worker_native_tid=%s model=%#x",
                self._session_sequence,
                None if self._inference_thread.ident is None else hex(self._inference_thread.ident),
                self._inference_thread.native_id,
                id(self.policy),
            )
            warmup_started = time.monotonic()
            deadline = warmup_started + self.config.pi05_inference.warmup_timeout_s
            next_progress = warmup_started + 30.0
            while not self._worker_ready.wait(timeout=0.1):
                is_active = getattr(context, "is_active", None)
                now = time.monotonic()
                if now >= deadline or (is_active is not None and not is_active()):
                    worker_stop_event.set()
                    self.shutdown_event.set()
                    raise RuntimeError("PI05 startup warmup timed out or setup RPC was cancelled")
                if now >= next_progress:
                    self.logger.info(
                        "Waiting for policy warmup: %.0f s elapsed (timeout %.0f s)",
                        now - warmup_started,
                        self.config.pi05_inference.warmup_timeout_s,
                    )
                    next_progress = now + 30.0
            self._raise_worker_failure(context)
            if not self.running or worker_stop_event.is_set():
                raise RuntimeError("policy startup was cancelled")

        elapsed = time.perf_counter() - started
        if cache_hit:
            self.logger.info(
                "Reused warm policy %s from %s on %s in %.3fs (cache_hits=%d)",
                policy_specs.policy_type,
                policy_specs.pretrained_name_or_path,
                policy_specs.device,
                elapsed,
                self.model_cache_hits,
            )
        else:
            self.logger.info(
                "Loaded policy %s from %s onto %s in %.3fs (model_loads=%d)",
                policy_specs.policy_type,
                policy_specs.pretrained_name_or_path,
                policy_specs.device,
                elapsed,
                self.model_loads,
            )
        if self._trained_runtime is not None:
            self.logger.info(
                "Trained RTC inference started (fixed delay_steps=%d, training_max_delay=%d, "
                "client queue_threshold=%d)",
                self._trained_runtime.delay_steps,
                self._trained_runtime.training_max_delay,
                setup.trained_queue_threshold,
            )
        else:
            self.logger.info(
                "Guided RTC inference started (execution_horizon=%d)", self.config.rtc.execution_horizon
            )
        return services_pb2.Empty()

    def _begin_rollout_generation(self, generation: int) -> None:
        """Clear server episode state on the inference worker thread.

        This method is intentionally called from ``_predict_remote_chunk`` and
        not the gRPC receive thread.  It therefore cannot mutate policy or
        processor state concurrently with an older CUDA inference.
        """

        if generation == self._active_rollout_generation:
            return
        if generation < self._active_rollout_generation:
            raise RuntimeError(
                "rollout generation moved backwards: "
                f"active={self._active_rollout_generation}, received={generation}"
            )

        previous = self._active_rollout_generation

        self._chunk_cache.clear()
        # Performance history belongs to this connection/model, not an episode.
        # Trackers expire after idle TTL; never reset the feedback watermark here.

        # A queued chunk belongs to a completed older inference. A concurrent
        # GetActions may already have taken it; that is harmless because the
        # client applies the same generation fence before merging.
        while True:
            try:
                self._action_mailbox.get_nowait()
            except Empty:
                break

        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()
        if self.policy is not None:
            self.policy.reset()
            rtc_processor = getattr(self.policy, "rtc_processor", None)
            reset_tracker = getattr(rtc_processor, "reset_tracker", None)
            if callable(reset_tracker):
                reset_tracker()

        # Commit the generation only after every reset succeeds. If a stateful
        # policy/processor reset raises, the inference loop retries this whole
        # boundary instead of accidentally treating a partial reset as done.
        self._active_rollout_generation = generation

        self.logger.info(
            "Started remote RTC rollout generation %d (previous=%d); model remains loaded",
            generation,
            previous,
        )

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Accept observation and latency feedback as one generation-aware update.

        Check the globally monotonic observation id first so a delayed upload
        cannot repopulate the next rollout's latency tracker.
        """

        received_bytes = receive_bytes_in_chunks(
            request_iterator,
            None,
            self.shutdown_event,
            log_prefix=self.logger.name,
            log_diagnostics=self.config.debug_rtc_diagnostics or self.config.debug_cuda_sync,
        )
        envelope = pickle.loads(received_bytes)  # nosec
        if self._rlt_runtime is not None:
            self._rlt_runtime.submit(envelope)
            return services_pb2.Empty()
        if isinstance(envelope, RemoteRTCCalibrationRequest):
            with self._observation_publish_lock:
                if (
                    not self._calibration_open
                    or self._setup is None
                    or envelope.token != self._setup.calibration_token
                    or envelope.sequence != self._calibration_next_sequence
                    or self._calibration_pending
                    or self._latest_received_observation_id >= 0
                ):
                    raise ValueError("stale, overlapping or out-of-session calibration request")
                self._calibration_pending = True
                self._calibration_next_sequence += 1
                self._observation_mailbox.put_nowait(envelope)
            return services_pb2.Empty()
        if not isinstance(envelope, RemoteRTCObservation):
            raise TypeError(f"RTC observation must be RemoteRTCObservation, got {type(envelope)}")

        with self._observation_publish_lock:
            if self._calibration_open:
                raise RuntimeError("finish startup calibration before sending executable requests")
            if envelope.observation_id <= self._latest_received_observation_id:
                if self.config.debug_rtc_diagnostics:
                    self.logger.debug("Ignoring stale RTC observation %d", envelope.observation_id)
                return services_pb2.Empty()

            feedback = envelope.latency_feedback_seconds
            feedback_source = envelope.latency_feedback_source_id
            if (
                self.config.rtc.mode == "guided"
                and feedback is not None
                and math.isfinite(feedback)
                and feedback >= 0
                and feedback_source is not None
            ):
                envelope_generation = decode_rollout_generation(envelope.observation_id)
                feedback_generation = decode_rollout_generation(feedback_source)
                if feedback_generation != envelope_generation:
                    if self.config.debug_rtc_diagnostics:
                        self.logger.debug(
                            "Ignoring cross-generation RTC latency feedback source=%d observation=%d",
                            feedback_source,
                            envelope.observation_id,
                        )
                else:
                    with self._state_lock:
                        if feedback_source > self._last_latency_feedback_source:
                            self._latency_tracker.add(feedback)
                            self._last_latency_feedback_source = feedback_source
                            self._check_execution_horizon_feedback(
                                envelope_generation,
                                feedback,
                            )

            self._latest_received_observation_id = envelope.observation_id
            _publish_latest(self._observation_mailbox, envelope)

        if self.config.debug_rtc_diagnostics:
            self.logger.debug(
                "Received RTC observation %d (generation=%d, executed=%d, refs=%d)",
                envelope.observation_id,
                decode_rollout_generation(envelope.observation_id),
                envelope.executed_action_step,
                len(envelope.action_refs),
            )
        return services_pb2.Empty()

    def _predict_remote_chunk(self, envelope: RemoteRTCObservation) -> RemoteRTCActionChunk | RemoteRTCRetry:
        generation = decode_rollout_generation(envelope.observation_id)
        self._begin_rollout_generation(generation)
        return super()._predict_remote_chunk(envelope)


PolicyServer = PolicyServerDeployment


def _serve_policy_server(policy_server: PolicyServerDeployment) -> None:
    """Run the common gRPC lifecycle for deployment and DAgger entry points."""

    cfg = policy_server.config
    configure_deployment_logging(debug=cfg.debug_rtc_diagnostics)
    with (
        server_fault_diagnostics(policy_server.logger, policy_server.prefix, cfg.fault_log_path)
        if cfg.enable_fault_handler
        else nullcontext()
    ):
        policy_server.logger.info(pformat(asdict(cfg)))
        policy_server.logger.info(
            "RTC runtime torch=%s torch_cuda=%s grpc=%s environment=%s",
            torch.__version__,
            torch.version.cuda,
            grpc.__version__,
            {
                key: os.environ.get(key)
                for key in (
                    "CUDA_LAUNCH_BLOCKING",
                    "TORCH_SHOW_CPP_STACKTRACES",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTORCH_CUDA_ALLOC_CONF",
                )
            },
        )
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
        server.add_insecure_port(f"{cfg.host}:{cfg.port}")
        policy_server.logger.info("%s started on %s:%d", type(policy_server).__name__, cfg.host, cfg.port)
        server.start()
        try:
            server.wait_for_termination()
        finally:
            policy_server.stop()


@draccus.wrap()
def serve(cfg: PolicyServerDeploymentConfig) -> None:
    _serve_policy_server(PolicyServerDeployment(cfg))


if __name__ == "__main__":
    serve()
