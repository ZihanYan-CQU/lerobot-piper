# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Independent trained-RTC runtime, without guided deployment heuristics.

Deployment provides RPC, request/generation fencing, observation capture and
robot dispatch. This module owns trained prefix transport, fixed scheduling,
hard-prefix admission and queue replacement. It imports no deployment class.
"""

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from lerobot.policies.rtc import reanchor_relative_rtc_prefix
from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)

from .helpers import TimedAction
from .image_transport import ImageResizeSpec, prepare_remote_observation
from .rtc_remote import RemoteRTCActionChunk, RemoteRTCObservation, RemoteRTCRetry


@dataclass
class TrainedRTCServerConfig:
    # A fixed commitment budget, independent of guided RTT history. None uses
    # the checkpoint's training maximum; a smaller budget trades jitter cover
    # for earlier replanning and must still fit the measured end-to-end delay.
    delay_steps: int | None = None

    def __post_init__(self) -> None:
        if self.delay_steps is not None and self.delay_steps < 1:
            raise ValueError("trained.delay_steps must be positive or None")


@dataclass
class TrainedRTCClientConfig:
    # None means half the requested chunk. No RTT-adaptive trigger or cooldown.
    queue_threshold: int | None = None
    max_consecutive_rejections: int = 5
    # Take the next fresh snapshot promptly after a valid result, even when the
    # queue is still long. This never changes the trained hard-prefix budget.
    replan_on_result: bool = True

    def __post_init__(self) -> None:
        if self.queue_threshold is not None and self.queue_threshold < 1:
            raise ValueError("trained.queue_threshold must be positive or None")
        if self.max_consecutive_rejections < 1:
            raise ValueError("trained.max_consecutive_rejections must be positive")


@dataclass(frozen=True)
class _TrainedScheduledAction:
    timed_action: TimedAction
    source_observation_id: int
    raw_action_index: int


class TrainedActionSchedule:
    """One contiguous plan plus an immutable endpoint owned by interpolation.

    A valid response replaces the entire mutable suffix. There is no per-row
    latest-writer arbitration or server-cache action references.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: dict[int, _TrainedScheduledAction] = {}
        self._inflight: _TrainedScheduledAction | None = None
        self._committed_step = -1

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._inflight = None
            self._committed_step = -1

    def _plan_locked(self, executed: int) -> list[_TrainedScheduledAction]:
        step = max(executed, self._committed_step) + 1
        plan = []
        if self._inflight is not None:
            if self._inflight.timed_action.get_timestep() != step:
                raise RuntimeError("trained in-flight endpoint is not the next policy action")
            plan.append(self._inflight)
            step += 1
        while step in self._actions:
            plan.append(self._actions[step])
            step += 1
        return plan

    def qsize(self, executed: int) -> int:
        with self._lock:
            return len(self._plan_locked(executed))

    def snapshot_prefix(self, executed: int, action_dim: int) -> torch.Tensor:
        with self._lock:
            plan = self._plan_locked(executed)
            if not plan:
                return torch.empty((0, action_dim), dtype=torch.float32)
            return torch.stack([row.timed_action.get_action() for row in plan]).detach().cpu().clone()

    def consume_next(self, executed: int) -> _TrainedScheduledAction | None:
        with self._lock:
            if self._inflight is not None:
                raise RuntimeError("trained action interpolation already owns an endpoint")
            step = max(executed, self._committed_step) + 1
            self._inflight = self._actions.pop(step, None)
            return self._inflight

    def inflight(self) -> _TrainedScheduledAction | None:
        with self._lock:
            return self._inflight

    def commit_inflight(self, expected_step: int) -> None:
        with self._lock:
            if self._inflight is None or self._inflight.timed_action.get_timestep() != expected_step:
                raise RuntimeError(f"cannot commit trained action {expected_step}")
            self._committed_step = expected_step
            self._inflight = None

    def first_mutable_step(self, executed: int) -> int:
        with self._lock:
            frozen = max(executed, self._committed_step)
            if self._inflight is not None:
                frozen = max(frozen, self._inflight.timed_action.get_timestep())
            return frozen + 1

    def replace(self, chunk: RemoteRTCActionChunk, executed: int, device: str) -> None:
        with self._lock:
            frozen = max(executed, self._committed_step)
            if self._inflight is not None:
                frozen = max(frozen, self._inflight.timed_action.get_timestep())
            self._actions = {
                action.get_timestep(): _TrainedScheduledAction(
                    TimedAction(
                        action.get_timestamp(), action.get_timestep(), action.get_action().to(device)
                    ),
                    chunk.source_observation_id,
                    index,
                )
                for index, action in enumerate(chunk.actions)
                if action.get_timestep() > frozen
            }


class TrainedRTCClientRuntime:
    """Fixed trigger, explicit prefix snapshots, and strict trained admission."""

    def __init__(self, config: TrainedRTCClientConfig, chunk_size: int, action_device: str):
        self.config = config
        self.chunk_size = chunk_size
        self.action_device = action_device
        self.queue_threshold = (
            config.queue_threshold if config.queue_threshold is not None else chunk_size // 2
        )
        if not 0 < self.queue_threshold < chunk_size:
            raise ValueError("trained.queue_threshold must be between 1 and actions_per_chunk - 1")
        self.schedule = TrainedActionSchedule()
        self._requests: dict[int, RemoteRTCObservation] = {}
        self.consecutive_rejections = 0
        self.last_rejection: str | None = None

    def reset(self) -> None:
        self.schedule.clear()
        self._requests.clear()
        self.consecutive_rejections = 0
        self.last_rejection = None

    def remember_request(self, request: RemoteRTCObservation) -> None:
        prefix = request.trained_prefix
        if request.action_refs or prefix is None or prefix.ndim != 2 or not torch.isfinite(prefix).all():
            raise ValueError("trained requests require an explicit finite [T, A] prefix and no action_refs")
        if self._requests and request.observation_id not in self._requests:
            raise RuntimeError("trained RTC permits only one in-flight observation")
        self._requests[request.observation_id] = replace(
            request, trained_prefix=prefix.detach().cpu().clone()
        )

    def forget_request(self, observation_id: int) -> None:
        self._requests.pop(observation_id, None)

    def reject(self, reason: str) -> bool:
        self.consecutive_rejections += 1
        self.last_rejection = reason
        if self.consecutive_rejections >= self.config.max_consecutive_rejections:
            raise RuntimeError(f"trained RTC repeatedly rejected fresh snapshots: {reason}")
        return False

    def accept(self, chunk: RemoteRTCActionChunk, executed: int) -> bool:
        request = self._requests.get(chunk.source_observation_id)
        if request is None:
            return False
        prefix = request.trained_prefix
        assert prefix is not None
        start = max(request.observation.get_timestep(), 0)
        if (
            chunk.rtc_mode != "trained"
            or chunk.chunk_start_step != start
            or chunk.prefix_length != len(prefix)
            or not 0 <= chunk.conditioned_delay <= min(len(prefix), chunk.training_max_delay)
            or (len(prefix) > 0 and chunk.conditioned_delay == 0)
            or chunk.training_max_delay <= 0
            or len(chunk.actions) != self.chunk_size
            or any(action.get_timestep() != start + i for i, action in enumerate(chunk.actions))
        ):
            raise ValueError("trained response does not match its observation/prefix snapshot")
        values = torch.stack([action.get_action().detach().cpu() for action in chunk.actions])
        if values.shape[1:] != prefix.shape[1:] or not torch.isfinite(values).all():
            raise ValueError("trained response contains invalid action values or dimensions")
        d = chunk.conditioned_delay
        if d and not torch.allclose(values[:d].float(), prefix[:d].float(), atol=1e-4, rtol=1e-4):
            raise ValueError("trained response changed the committed hard prefix")
        consumed = max(0, self.schedule.first_mutable_step(executed) - start)
        if len(prefix) and consumed > d:
            return self.reject(f"consumed {consumed} policy steps, but conditioned on {d}")
        if consumed >= len(chunk.actions):
            return self.reject("no unconsumed actions remain")
        self.schedule.replace(chunk, executed, self.action_device)
        self.consecutive_rejections = 0
        self.last_rejection = None
        return True


class TrainedRTCServerRuntime:
    """Native trained sampling with explicit absolute prefixes and its own budget."""

    def __init__(
        self,
        *,
        config: TrainedRTCServerConfig,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        features: dict,
        device: str,
        fps: float,
        chunk_size: int,
        queue_threshold: int,
        minimum_inference_seconds: float = 0.0,
        image_transport_plan: dict[str, ImageResizeSpec] | None = None,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.features = features
        self.device = device
        self.fps = fps
        self.chunk_size = chunk_size
        self.minimum_inference_seconds = minimum_inference_seconds
        self.image_transport_plan = image_transport_plan
        self.training_max_delay = int(getattr(policy.config, "rtc_training_max_delay", 0))
        if self.training_max_delay <= 0:
            raise ValueError("trained RTC requires a checkpoint trained with rtc_training_max_delay > 0")
        self.delay_steps = config.delay_steps if config.delay_steps is not None else self.training_max_delay
        if not 0 < self.delay_steps <= self.training_max_delay:
            raise ValueError("trained.delay_steps must fit the checkpoint's rtc_training_max_delay")
        model_horizon = int(getattr(policy.config, "chunk_size", chunk_size))
        if chunk_size > model_horizon or 2 * self.delay_steps > chunk_size:
            raise ValueError("trained RTC requires 2 * delay_steps <= actions_per_chunk <= model chunk_size")
        if not self.delay_steps <= queue_threshold < chunk_size:
            raise ValueError("trained queue threshold must cover delay_steps and be smaller than the chunk")
        self.relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None
        )
        self.normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)), None
        )

    def _model_prefix(self, absolute: torch.Tensor) -> torch.Tensor:
        if self.relative_step is not None:
            state = self.relative_step.get_cached_state()
            if state is None:
                raise ValueError("trained relative prefix requires the current observation state")
            return reanchor_relative_rtc_prefix(
                absolute, state, self.relative_step, self.normalizer_step, self.device
            )
        transition = create_transition(action=absolute)
        if self.normalizer_step is not None:
            transition = self.normalizer_step(transition)
        return transition[TransitionKey.ACTION].to(self.device)

    def predict(self, request: RemoteRTCObservation) -> RemoteRTCActionChunk | RemoteRTCRetry:
        started = time.perf_counter()
        prefix = request.trained_prefix
        # Missing transport state is not an empty queue. In particular, never
        # silently convert a guided cache reference into an unconditional call.
        if request.action_refs or prefix is None:
            return self._retry(request, "prefix_unavailable", 0.0)
        if (
            prefix.ndim != 2
            or prefix.device.type != "cpu"
            or not prefix.is_floating_point()
            or len(prefix) > self.chunk_size
            or not torch.isfinite(prefix).all()
            or request.observation.get_timestep() != request.executed_action_step + 1
        ):
            raise ValueError("invalid trained observation/prefix snapshot")
        observation = prepare_remote_observation(
            request.observation.get_observation(),
            self.features,
            self.policy.config.image_features,
            self.image_transport_plan,
        )
        preprocessed = self.preprocessor(observation)
        delay = min(self.delay_steps, len(prefix))
        model_prefix = self._model_prefix(prefix) if delay else None
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(
                preprocessed, inference_delay=delay, prev_chunk_left_over=model_prefix
            )
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < self.chunk_size:
                raise ValueError("trained policy must return a complete [1, H, A] chunk")
            processed = self.postprocessor(actions[:, : self.chunk_size]).squeeze(0).detach().cpu()
        if processed.shape != (self.chunk_size, prefix.shape[1]) or not torch.isfinite(processed).all():
            raise ValueError("trained postprocessor returned invalid actions")
        remaining = self.minimum_inference_seconds - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        elapsed = time.perf_counter() - started
        if delay and math.ceil(elapsed * self.fps) > delay:
            return self._retry(request, "delay_exceeded", elapsed)
        start = request.observation.get_timestep()
        timestamp = request.observation.get_timestamp()
        return RemoteRTCActionChunk(
            source_observation_id=request.observation_id,
            chunk_start_step=start,
            observation_timestamp=timestamp,
            actions=[
                TimedAction(timestamp + i / self.fps, start + i, value) for i, value in enumerate(processed)
            ],
            conditioned_delay=delay,
            prefix_length=len(prefix),
            inference_seconds=elapsed,
            rtc_mode="trained",
            training_max_delay=self.training_max_delay,
        )

    @staticmethod
    def _retry(request: RemoteRTCObservation, reason: str, elapsed: float) -> RemoteRTCRetry:
        return RemoteRTCRetry(
            source_observation_id=request.observation_id,
            observation_timestamp=request.observation.get_timestamp(),
            reason=reason,
            rtc_mode="trained",
            inference_seconds=elapsed,
        )
