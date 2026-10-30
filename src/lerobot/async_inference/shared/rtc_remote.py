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

"""Wire objects and rollout-generation identifiers shared by the RTC peers.

The existing async-inference protobuf deliberately transports opaque pickled
payloads.  Keeping the RTC metadata in small dataclasses lets the RTC pair use
that transport without changing the original protobuf service or the original
``policy_server.py`` / ``robot_client.py`` entry points.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import torch

from .helpers import RemotePolicyConfig, TimedAction, TimedObservation

# Retain the retired wire-format identifier so deployment can reject old setups
# rather than interpreting them as its scheduling/admission contract.
REMOTE_RTC_PROTOCOL_VERSION = 2
DEPLOYMENT_RTC_PROTOCOL_VERSION = 5

# High bits fence rollout generations; low bits count requests within one rollout.
ROLLOUT_SEQUENCE_BITS = 32
ROLLOUT_SEQUENCE_LIMIT = 1 << ROLLOUT_SEQUENCE_BITS
_ROLLOUT_SEQUENCE_MASK = ROLLOUT_SEQUENCE_LIMIT - 1


def encode_rollout_observation_id(generation: int, sequence: int) -> int:
    """Pack a non-negative rollout generation and local request sequence."""

    if generation < 0:
        raise ValueError(f"generation must be non-negative, got {generation}")
    if not 0 <= sequence < ROLLOUT_SEQUENCE_LIMIT:
        raise ValueError(f"sequence must be in [0, {ROLLOUT_SEQUENCE_LIMIT}), got {sequence}")
    return (generation << ROLLOUT_SEQUENCE_BITS) | sequence


def decode_rollout_generation(observation_id: int) -> int:
    """Return the rollout generation encoded in an observation/chunk id."""

    if observation_id < 0:
        raise ValueError(f"observation_id must be non-negative, got {observation_id}")
    return observation_id >> ROLLOUT_SEQUENCE_BITS


def decode_rollout_sequence(observation_id: int) -> int:
    """Return the generation-local sequence encoded in an observation id."""

    if observation_id < 0:
        raise ValueError(f"observation_id must be non-negative, got {observation_id}")
    return observation_id & _ROLLOUT_SEQUENCE_MASK


class RTCPrefixUnavailableError(RuntimeError):
    """A non-empty committed prefix could not be reconstructed."""


def supports_remote_rtc(policy: object) -> bool:
    """Return whether a policy advertises and accepts the RTC call shape."""

    supports_rtc = getattr(policy, "supports_rtc", None)
    predict = getattr(policy, "predict_action_chunk", None)
    if not callable(supports_rtc) or not supports_rtc() or not callable(predict):
        return False
    try:
        inspect.signature(predict).bind(
            object(),
            inference_delay=0,
            prev_chunk_left_over=None,
        )
    except (TypeError, ValueError):
        return False
    return True


def normalize_rtc_prefix_length(previous: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Hold-pad or truncate a non-empty ``[T, A]`` RTC model prefix.

    Action-space zero is generally the dataset mean after normalization, not
    the last command the robot holds when its real schedule runs out. Repeating
    the final real action therefore gives guided RTC a physically meaningful
    fixed-size prefix without fabricating client-side schedule entries.
    """

    if previous.ndim != 2:
        raise ValueError(f"expected a [T, A] RTC prefix, got shape={tuple(previous.shape)}")
    steps, _ = previous.shape
    if steps == target_steps:
        return previous
    if steps > target_steps:
        return previous[:target_steps]
    if steps == 0:
        raise ValueError("cannot hold-pad an empty RTC prefix")
    hold = previous[-1:].expand(target_steps - steps, -1)
    return torch.cat((previous, hold), dim=0)


def estimate_rtc_delay(
    *,
    latency: float,
    time_per_step: float,
    mode: str,
    training_max_delay: int,
    has_previous_actions: bool,
) -> int:
    """Quantize observed round-trip latency, bootstrapping trained RTC safely."""

    if latency:
        return math.ceil(latency / time_per_step)
    if mode == "trained" and has_previous_actions:
        return training_max_delay
    return 0


def clamp_trained_rtc_delay(*, conditioned_delay: int, available_steps: int, training_max_delay: int) -> int:
    """Clamp a trained prefix to both checkpoint and reconstructed-schedule support."""

    return min(conditioned_delay, available_steps, training_max_delay)


def trained_rtc_chunk_can_merge(
    *,
    conditioned_delay: int,
    measured_delay: int,
    training_max_delay: int,
    has_previous_actions: bool,
) -> bool:
    """Return whether elapsed inference still fits a trained RTC hard prefix."""

    if not has_previous_actions:
        return True
    return measured_delay <= training_max_delay and measured_delay <= conditioned_delay


@dataclass(frozen=True)
class RTCActionRef:
    """Reference to one unchanged action row cached on the policy server."""

    source_observation_id: int
    action_index: int

    def __post_init__(self) -> None:
        if self.source_observation_id < 0:
            raise ValueError("source_observation_id must be non-negative")
        if self.action_index < 0:
            raise ValueError("action_index must be non-negative")


@dataclass(frozen=True)
class RemoteRTCPolicySetup:
    """Policy setup plus the timing and action metadata RTC needs."""

    policy: RemotePolicyConfig
    control_frequency_hz: float
    action_feature_names: tuple[str, ...] = ()
    protocol_version: int = REMOTE_RTC_PROTOCOL_VERSION
    rtc_mode: str = "guided"
    trained_queue_threshold: int | None = None
    # Optional, backward-compatible capability. The server acknowledges its
    # loaded checkpoint's resize specification in gRPC response metadata.
    client_image_resize: bool = False
    # A fresh token scopes startup-only dry runs to this connection. Historical
    # RTC clients leave it unset and never use the calibration wire objects.
    calibration_token: str | None = None
    image_transport_mode: str = "lerobot_float32"

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        if self.rtc_mode not in {"guided", "trained"}:
            raise ValueError(f"unsupported RTC mode: {self.rtc_mode!r}")


@dataclass(frozen=True)
class RemoteRTCObservation:
    """Observation and an asynchronous snapshot of client execution state.

    Both modes snapshot the actions still scheduled when capture completes.
    Guided mode sends cache ``action_refs``; trained mode sends the explicit
    absolute ``trained_prefix`` so it is independent of guided action caches.
    """

    observation: TimedObservation
    observation_id: int
    executed_action_step: int
    action_refs: tuple[RTCActionRef, ...] = ()
    latency_feedback_seconds: float | None = None
    latency_feedback_source_id: int | None = None
    # Trained mode sends an explicit CPU [T, A] prefix in absolute robot-action
    # coordinates. An empty tensor means a genuinely empty execution queue;
    # None means the trained protocol was not supplied, never a cold start.
    trained_prefix: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("observation_id must be non-negative")
        if self.latency_feedback_seconds is not None and self.latency_feedback_seconds < 0:
            raise ValueError("latency_feedback_seconds must be non-negative")
        if (self.latency_feedback_seconds is None) != (self.latency_feedback_source_id is None):
            raise ValueError("latency feedback value and source id must be provided together")
        if self.latency_feedback_source_id is not None and self.latency_feedback_source_id < 0:
            raise ValueError("latency_feedback_source_id must be non-negative")


@dataclass
class RemoteRTCActionChunk:
    """A complete action chunk returned by one background ``GetActions`` RPC."""

    source_observation_id: int
    chunk_start_step: int
    observation_timestamp: float
    actions: list[TimedAction] = field(default_factory=list)
    conditioned_delay: int = 0
    prefix_length: int = 0
    inference_seconds: float = 0.0
    rtc_mode: str = "guided"
    training_max_delay: int = 0

    def __getstate__(self) -> dict:
        # Each row is usually a view of the whole model output storage. Pickling
        # those tensors separately repeats that storage H times (O(H^2 * A)).
        # Send one dense tensor plus per-row metadata, retaining the public API.
        state = self.__dict__.copy()
        actions = state.pop("actions")
        values = torch.stack([action.get_action().detach().cpu() for action in actions]) if actions else None
        state["_packed_actions"] = (
            values,
            tuple(action.get_timestamp() for action in actions),
            tuple(action.get_timestep() for action in actions),
        )
        return state

    def __setstate__(self, state: dict) -> None:
        state = state.copy()
        packed = state.pop("_packed_actions", None)
        if packed is not None:
            values, timestamps, steps = packed
            rows = () if values is None else values
            state["actions"] = [
                TimedAction(timestamp, step, value)
                for timestamp, step, value in zip(timestamps, steps, rows, strict=True)
            ]
        # Older stored chunks already contain the actions list.
        self.__dict__.update(state)

    def __post_init__(self) -> None:
        if self.source_observation_id < 0:
            raise ValueError("source_observation_id must be non-negative")
        if self.chunk_start_step < 0:
            raise ValueError("chunk_start_step must be non-negative")
        if self.conditioned_delay < 0:
            raise ValueError("conditioned_delay must be non-negative")
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be non-negative")
        if self.inference_seconds < 0:
            raise ValueError("inference_seconds must be non-negative")
        if self.rtc_mode not in {"guided", "trained"}:
            raise ValueError(f"unsupported RTC mode: {self.rtc_mode!r}")
        if self.training_max_delay < 0:
            raise ValueError("training_max_delay must be non-negative")


@dataclass(frozen=True)
class RemoteRTCCalibrationRequest:
    """Startup-only dry run; observation=None closes calibration and clears state.

    Warmup requests are explicitly excluded from latency feedback. Measured
    requests return a wrapped, non-executable chunk to exercise normal transport.
    """

    token: str
    sequence: int
    observation: TimedObservation | None = None
    warmup: bool = False
    use_prefix: bool = True
    feedback_source: int | None = None
    feedback_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.token or self.sequence < 0:
            raise ValueError("calibration requires a session token and non-negative sequence")
        if (self.feedback_source is None) != (self.feedback_seconds is None):
            raise ValueError("calibration feedback requires both source and value")
        if self.feedback_seconds is not None and (
            not math.isfinite(self.feedback_seconds) or self.feedback_seconds < 0
        ):
            raise ValueError("calibration feedback must be finite and non-negative")


@dataclass(frozen=True)
class RemoteRTCCalibrationResult:
    """Never an executable action response, even though it carries the same payload."""

    token: str
    sequence: int
    chunk: RemoteRTCActionChunk | None = None


@dataclass(frozen=True)
class RemoteRTCRetry:
    """Terminal response for one request: capture a NEW observation and prefix.

    A retry response carries no executable actions. It completes the matching
    client's single-flight slot; the server must never retry that old snapshot.
    """

    source_observation_id: int
    observation_timestamp: float
    reason: str
    rtc_mode: str = "guided"
    inference_seconds: float = 0.0
