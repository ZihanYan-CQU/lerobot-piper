"""Synchronous fixed-window RLT rollout; hardware and task rewards remain explicit callbacks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

import numpy as np
import torch

from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE

from .replay import FixedWindowCollector, FixedWindowReplayBuffer, WindowTransition, replay_spec


@dataclass(frozen=True)
class ExecutedStep:
    """One POLICY step, after any motor-rate interpolation; observation must be pre-reset.

    executed_action is the actual absolute command reported by the controller,
    including smoothing/clipping or human takeover. reward belongs to this
    executed interval. Supply the final observation for terminal/time-limit steps.
    """

    observation: dict
    executed_action: torch.Tensor
    reward: float
    terminated: bool = False
    truncated: bool = False
    intervention: bool = False


@dataclass(frozen=True)
class RewardSignal:
    """Task-owned reward and episode outcome for one executed policy interval."""

    reward: float
    terminated: bool = False
    truncated: bool = False
    intervention: bool = False


class RobotWindowStep:
    """Bridge an already-connected RobotIO to fixed-window collection.

    prepare converts a captured robot observation to the saved PI05 processor's
    input convention (including task and image scale). reward is supplied by the
    task/environment; no synthetic success labels are inferred here.
    """

    def __init__(
        self,
        io,
        prepare: Callable,
        reward: Callable,
        *,
        action_names: list[str],
        fps: float = 30,
        max_observation_age_s: float = 0.2,
        stop_event=None,
    ):
        if (
            not math.isfinite(fps)
            or fps <= 0
            or not math.isfinite(max_observation_age_s)
            or max_observation_age_s <= 0
        ):
            raise ValueError("fps and max_observation_age_s must be finite and positive")
        self.io, self.prepare, self.reward = io, prepare, reward
        actual_names = getattr(io, "action_keys", None)
        if (
            not action_names
            or not actual_names
            or [key.removesuffix(".pos") for key in action_names]
            != [key.removesuffix(".pos") for key in actual_names]
        ):
            raise ValueError("Robot action order must match the Stage 1 checkpoint action_feature_names")
        self.fps, self.max_age = fps, max_observation_age_s
        self.stop_event = stop_event if stop_event is not None else Event()

    def __call__(self, action: torch.Tensor) -> ExecutedStep:
        if self.stop_event.is_set():
            raise InterruptedError("Fixed-window robot step cancelled")
        if (
            not isinstance(action, torch.Tensor)
            or action.shape != (len(self.io.action_keys),)
            or not torch.isfinite(action).all()
        ):
            raise ValueError("Robot command must be a finite vector in the checkpoint's action order")
        started = time.monotonic()
        sent = self.io.send(
            action.detach().float().cpu().numpy(),
            deadline=started + self.max_age,
            stop_event=self.stop_event,
        )
        sent_at = time.monotonic()
        sent = torch.as_tensor(np.asarray(sent), dtype=torch.float32).clone()
        if sent.shape != action.shape or not torch.isfinite(sent).all():
            raise ValueError("RobotIO did not report a finite executed command")
        if self.stop_event.wait(max(0.0, 1 / self.fps - (time.monotonic() - started))):
            raise InterruptedError("Fixed-window robot step cancelled")
        raw, captured_at, source_at = self.io.capture(self.max_age)
        now = time.monotonic()
        if (
            not all(math.isfinite(t) for t in (captured_at, source_at))
            or source_at < sent_at
            or source_at > captured_at
            or captured_at > now
            or now - source_at > self.max_age
        ):
            raise TimeoutError("Robot reward/next observation is stale or has invalid timestamps")
        signal = self.reward(raw, sent.clone())
        if not isinstance(signal, RewardSignal):
            raise TypeError("Task reward callback must return RewardSignal")
        return ExecutedStep(
            self.prepare(raw), sent, signal.reward, signal.terminated, signal.truncated, signal.intervention
        )


class FixedWindowRollout:
    """A single owner commits K rows before requesting another feature/reference.

    It never connects, resets, homes or disconnects a robot. The explicit step
    callback can call RobotIO.send; return its reported action and task reward.
    Model/reference/actor selection is made only at window boundaries.
    """

    def __init__(self, features, replay: FixedWindowReplayBuffer, *, actor=None, stop_event=None):
        if replay_spec(features.config) != replay_spec(replay.config):
            raise ValueError("Feature and replay contracts differ")
        if actor is not None and replay_spec(actor.config) != replay_spec(replay.config):
            raise ValueError("Actor and replay contracts differ")
        self.features, self.replay, self.actor = features, replay, actor
        self.config = replay.config
        self.stop_event = stop_event if stop_event is not None else Event()
        self.collector = FixedWindowCollector(self.config)
        self._state = None
        self._ended = False
        self._collecting = False

    def load_actor_weights(self, payload: dict) -> None:
        """Install a learner snapshot between windows, on the rollout owner thread."""
        if self._collecting or self.collector.active:
            raise RuntimeError("Actor weights can only change at execution-window boundaries")
        if self.actor is None:
            raise ValueError("Actor-controlled rollout requires an actor")
        self.actor.load_rollout_weights(payload)

    def begin_episode(self, observation: dict, *, episode_id: int) -> None:
        if self._collecting or self.collector.active:
            raise RuntimeError("Cannot reset during an execution window; request cancellation first")
        if self.stop_event.is_set():
            raise RuntimeError("Rollout is stopped")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id must be a nonnegative integer")
        self._state = None
        self.features.reset()
        self._state = self.features.encode(observation)
        self._episode_id, self._step, self._ended = episode_id, 0, False

    def abort(self):
        """Discard partial data; a fresh begin_episode is required to continue."""
        self.collector.abort()
        self._state = None

    def collect_window(
        self,
        step: Callable[[torch.Tensor], ExecutedStep],
        *,
        use_actor: bool = False,
        policy_version: int | None = None,
        deterministic: bool = False,
    ) -> WindowTransition | None:
        if self._collecting:
            raise RuntimeError("An execution window is already active")
        if self._state is None or self._ended:
            raise RuntimeError("begin_episode is required before collecting a new window")
        if use_actor and self.actor is None:
            raise ValueError("Actor-controlled rollout requires an actor")
        if self.stop_event.is_set():
            self.abort()
            return None
        actor_version = getattr(self.actor, "rollout_version", -1) if use_actor else -1
        if policy_version is None:
            policy_version = max(0, actor_version)
        elif actor_version >= 0 and policy_version != actor_version:
            raise ValueError("policy_version must match the loaded actor snapshot")
        self._collecting = True
        try:
            with torch.no_grad():
                if use_actor:
                    model = (
                        self.actor.predict_action_chunk(
                            {key: value[None] for key, value in self._state.items()},
                            deterministic=deterministic,
                        )[0]
                        .detach()
                        .float()
                        .cpu()
                        .clone()
                    )
                else:
                    model = self._state[OBS_REFERENCE][: self.config.execution_horizon].clone()
                commands = self.features.executable_actions(model)
            self.collector.begin(
                self._state,
                planned_action=model,
                episode_id=self._episode_id,
                start_step=self._step,
                policy_version=policy_version,
            )
            for command in commands:
                if self.stop_event.is_set():
                    self.abort()
                    return None
                result = step(command.clone())
                if not isinstance(result, ExecutedStep):
                    raise TypeError("step must return an ExecutedStep with a real reward and sent command")
                sent = self.features.normalize_executed(result.executed_action)
                self.collector.record(
                    sent,
                    result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    intervention=result.intervention,
                )
                if result.terminated or result.truncated:
                    break
            if self.stop_event.is_set():
                self.abort()
                return None
            # The same final observation supplies s' and the next window's s.
            next_state = self.features.encode(result.observation)
            if self.stop_event.is_set():
                self.abort()
                return None
            transition = self.collector.finish(next_state)
            self.replay.add(transition)
            self._state = next_state
            self._step += transition.valid_steps
            self._ended = transition.terminated or transition.truncated
            return transition
        except BaseException:
            self.abort()
            raise
        finally:
            self._collecting = False
