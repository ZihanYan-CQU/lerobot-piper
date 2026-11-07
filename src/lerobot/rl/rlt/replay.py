"""Replay complete execution windows; partial windows require a real episode boundary."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch
from torch import Tensor

from lerobot.policies.rlt_actor.configuration_rlt_actor import RLTActorConfig
from lerobot.policies.rlt_actor.modeling_rlt_actor import validate_observation


def replay_spec(config: RLTActorConfig) -> dict:
    spec = {
        key: getattr(config, key)
        for key in (
            "feature_model_id",
            "z_dim",
            "proprio_dim",
            "action_dim",
            "execution_horizon",
            "reference_horizon",
        )
    }

    # Keep existing Stage 2 replay/checkpoints loadable, with a distinct RTC spec.
    if config.rtc_prefix_steps:
        spec["rtc_prefix_steps"] = config.rtc_prefix_steps
    return spec


def feature_digest(identifier: str) -> Tensor:
    return torch.tensor(list(hashlib.sha256(identifier.encode()).digest()), dtype=torch.uint8)


def owned_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().to(device="cpu", dtype=torch.float32).clone() for key, value in state.items()}


@dataclass(frozen=True)
class WindowTransition:
    state: dict[str, Tensor]
    action: Tensor
    rewards: Tensor
    next_state: dict[str, Tensor]
    valid_steps: int
    terminated: bool
    truncated: bool
    intervention: Tensor
    feature_model_id: str
    episode_id: int
    start_step: int
    policy_version: int = 0
    planned_action: Tensor | None = None

    def validate(self, config: RLTActorConfig) -> None:
        validate_observation(self.state, config, batched=False, strict=True)
        validate_observation(self.next_state, config, batched=False, strict=True)
        k, d = config.execution_horizon, config.action_dim
        if self.feature_model_id != config.feature_model_id:
            raise ValueError("Replay feature_model_id differs from the frozen Stage 1 model/processors")
        for key in ("valid_steps", "episode_id", "start_step", "policy_version"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if key == "valid_steps" else 0)
            ):
                raise ValueError(f"Invalid transition {key}")
        if self.valid_steps > k or (self.valid_steps < k and not (self.terminated or self.truncated)):
            raise ValueError("A partial execution window must terminate or truncate the episode")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated/truncated must be booleans")
        if self.action.shape != (k, d) or self.rewards.shape != (k,):
            raise ValueError("Transition actions/rewards do not match the fixed execution window")
        if not self.action.is_floating_point() or not self.rewards.is_floating_point():
            raise ValueError("Transition actions/rewards must be floating tensors")
        if not torch.isfinite(self.action).all() or not torch.isfinite(self.rewards).all():
            raise ValueError("Transition actions/rewards must be finite")
        if self.planned_action is None:
            if self.valid_steps < k:
                raise ValueError(
                    "Partial RLT replay requires the complete planned_action; recollect legacy partial windows"
                )
        elif (
            not isinstance(self.planned_action, Tensor)
            or self.planned_action.shape != (k, d)
            or not self.planned_action.is_floating_point()
            or not torch.isfinite(self.planned_action).all()
        ):
            raise ValueError("planned_action must be a complete finite normalized execution window")
        if self.intervention.shape != (k,) or self.intervention.dtype != torch.bool:
            raise ValueError("intervention must be a boolean per-step mask")
        tail = slice(self.valid_steps, None)
        if (
            self.action[tail].count_nonzero()
            or self.rewards[tail].count_nonzero()
            or self.intervention[tail].any()
        ):
            raise ValueError("Unexecuted tail must be zero padded")

    def owned(self) -> WindowTransition:
        return WindowTransition(
            owned_state(self.state),
            self.action.detach().float().cpu().clone(),
            self.rewards.detach().float().cpu().clone(),
            owned_state(self.next_state),
            self.valid_steps,
            self.terminated,
            self.truncated,
            self.intervention.detach().cpu().clone(),
            self.feature_model_id,
            self.episode_id,
            self.start_step,
            self.policy_version,
            (self.action if self.planned_action is None else self.planned_action)
            .detach()
            .float()
            .cpu()
            .clone(),
        )


class FixedWindowCollector:
    """A single-stream window builder; cancel/reset discards unfinished transitions."""

    def __init__(self, config: RLTActorConfig):
        self.config = config
        self.abort()

    @property
    def active(self) -> bool:
        return self._state is not None

    def begin(
        self,
        state: dict[str, Tensor],
        *,
        planned_action: Tensor,
        episode_id: int,
        start_step: int,
        policy_version: int = 0,
    ) -> None:
        if self.active:
            raise RuntimeError("An execution window is already active")
        validate_observation(state, self.config, batched=False, strict=True)
        if (
            not isinstance(planned_action, Tensor)
            or planned_action.shape != (self.config.execution_horizon, self.config.action_dim)
            or not planned_action.is_floating_point()
            or not torch.isfinite(planned_action).all()
        ):
            raise ValueError("A complete finite planned_action is required before collecting a window")
        for key, value in (
            ("episode_id", episode_id),
            ("start_step", start_step),
            ("policy_version", policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        self._state = owned_state(state)
        self._planned_action = planned_action.detach().float().cpu().clone()
        self._episode_id, self._start_step, self._version = episode_id, start_step, policy_version
        self._actions, self._rewards, self._human = [], [], []
        self._terminated = self._truncated = False

    def record(
        self,
        executed_action: Tensor,
        reward: float,
        *,
        terminated: bool = False,
        truncated: bool = False,
        intervention: bool = False,
    ) -> None:
        if (
            not self.active
            or self._terminated
            or self._truncated
            or len(self._actions) >= self.config.execution_horizon
        ):
            raise RuntimeError("No open execution step in this window")
        action = torch.as_tensor(executed_action).detach().float().cpu().clone()
        reward_tensor = torch.as_tensor(reward)
        if action.shape != (self.config.action_dim,) or not torch.isfinite(action).all():
            raise ValueError("Executed action must be a finite action_dim vector")
        if reward_tensor.ndim != 0 or not torch.isfinite(reward_tensor):
            raise ValueError("Each executed policy step requires a finite scalar reward")
        if any(not isinstance(flag, bool) for flag in (terminated, truncated, intervention)):
            raise ValueError("Execution flags must be booleans")
        self._actions.append(action)
        self._rewards.append(float(reward))
        self._human.append(intervention)
        self._terminated, self._truncated = terminated, truncated

    def finish(self, next_state: dict[str, Tensor]) -> WindowTransition:
        if not self.active:
            raise RuntimeError("No execution window is active")
        n = len(self._actions)
        if n < 1 or (n < self.config.execution_horizon and not (self._terminated or self._truncated)):
            raise ValueError("Incomplete nonterminal execution window; abort it instead of replaying it")
        actions = torch.zeros(self.config.execution_horizon, self.config.action_dim)
        rewards = torch.zeros(self.config.execution_horizon)
        human = torch.zeros(self.config.execution_horizon, dtype=torch.bool)
        actions[:n], rewards[:n], human[:n] = (
            torch.stack(self._actions),
            torch.tensor(self._rewards),
            torch.tensor(self._human),
        )
        transition = WindowTransition(
            self._state,
            actions,
            rewards,
            owned_state(next_state),
            n,
            self._terminated,
            self._truncated,
            human,
            self.config.feature_model_id,
            self._episode_id,
            self._start_step,
            self._version,
            self._planned_action,
        )
        transition.validate(self.config)
        self.abort()
        return transition

    def abort(self) -> None:
        self._state = None
        self._planned_action = None
        self._actions, self._rewards, self._human = [], [], []
        self._terminated = self._truncated = False


class FixedWindowReplayBuffer:
    """Thread-safe CPU ring with owned tensors, explicit feature version and restorable sampling RNG."""

    FORMAT_VERSION = 2

    def __init__(
        self, config: RLTActorConfig, capacity: int = 100_000, *, seed: int = 0, device: str = "cpu"
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.config = copy.deepcopy(config)
        self.capacity = capacity
        self.device = torch.device(device)
        self._data: list[WindowTransition] = []
        self._position = 0
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._lock = RLock()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def add(self, transition: WindowTransition) -> None:
        transition.validate(self.config)
        transition = transition.owned()
        with self._lock:
            if len(self._data) < self.capacity:
                self._data.append(transition)
            else:
                self._data[self._position] = transition
            self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        with self._lock:
            if not self._data:
                raise RuntimeError("Cannot sample empty fixed-window replay")
            indices = torch.randint(len(self._data), (batch_size,), generator=self._rng).tolist()
            rows = [self._data[i] for i in indices]

            def stack(tensors):
                return torch.stack(tensors).to(self.device)

            valid = stack([torch.arange(self.config.execution_horizon) < row.valid_steps for row in rows])
            return {
                "state": {key: stack([row.state[key] for row in rows]) for key in rows[0].state},
                "next_state": {
                    key: stack([row.next_state[key] for row in rows]) for key in rows[0].next_state
                },
                "action": stack([row.action for row in rows]),
                "reward": stack([row.rewards for row in rows]),
                "done": torch.tensor([row.terminated for row in rows], device=self.device),
                "truncated": torch.tensor([row.truncated for row in rows], device=self.device),
                "complementary_info": {
                    "planned_action": stack([row.planned_action for row in rows]),
                    "valid_mask": valid,
                    "intervention": stack([row.intervention for row in rows]),
                    "feature_id": feature_digest(self.config.feature_model_id)
                    .expand(batch_size, -1)
                    .clone()
                    .to(self.device),
                    "episode_id": torch.tensor([row.episode_id for row in rows], device=self.device),
                    "start_step": torch.tensor([row.start_step for row in rows], device=self.device),
                    "policy_version": torch.tensor([row.policy_version for row in rows], device=self.device),
                },
            }

    def get_iterator(self, batch_size: int, async_prefetch: bool = False, queue_size: int = 2):
        # Sampling is intentionally synchronous; the ring lock protects concurrent ingestion.
        while True:
            yield self.sample(batch_size)

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "format_version": self.FORMAT_VERSION,
                "spec": replay_spec(self.config),
                "capacity": self.capacity,
                "position": self._position,
                "rng": self._rng.get_state().clone(),
                "data": [vars(row.owned()) for row in self._data],
            }

    def load_state_dict(self, state: dict) -> None:
        if (
            state["format_version"] not in (1, self.FORMAT_VERSION)
            or state["spec"] != replay_spec(self.config)
            or state["capacity"] != self.capacity
        ):
            raise ValueError("Replay schema, capacity, feature model or window shape differs")
        if state["format_version"] == 1:
            if any(row["valid_steps"] != self.config.execution_horizon for row in state["data"]):
                raise ValueError(
                    "Legacy partial RLT replay has no complete planned_action; recollect it with the fixed collector"
                )
            rows = [WindowTransition(**(row | {"planned_action": row["action"]})) for row in state["data"]]
        else:
            if any(row.get("planned_action") is None for row in state["data"]):
                raise ValueError("RLT replay v2 must contain planned_action for every window")
            rows = [WindowTransition(**row) for row in state["data"]]
        if len(rows) > self.capacity or not 0 <= state["position"] < self.capacity:
            raise ValueError("Invalid replay ring metadata")
        if len(rows) < self.capacity and state["position"] != len(rows):
            raise ValueError("Invalid replay ring position")
        for row in rows:
            row.validate(self.config)
        rng = torch.Generator(device="cpu")
        rng.set_state(state["rng"].cpu())
        with self._lock:
            self._data = [row.owned() for row in rows]
            self._position, self._rng = state["position"], rng

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(self.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, config: RLTActorConfig, *, device: str = "cpu"):
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.pt"))
            if not files:
                raise ValueError("RLT replay journal directory contains no completed windows")
            replay = cls(config, capacity=len(files), device=device)
            seen = set()
            for file in files:
                shard = cls.load(file, config)
                if len(shard) != 1:
                    raise ValueError("RLT journal shards must each contain exactly one completed window")
                row = shard._data[0]
                identity = (row.episode_id, row.start_step)
                if identity in seen:
                    raise ValueError("Duplicate episode/window in RLT replay journal")
                seen.add(identity)
                replay.add(row)
            return replay
        state = torch.load(path, map_location="cpu", weights_only=True)
        replay = cls(config, state["capacity"], device=device)
        replay.load_state_dict(state)
        return replay
