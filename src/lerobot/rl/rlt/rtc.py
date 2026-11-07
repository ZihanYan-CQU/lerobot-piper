"""Single-owner RLT planning and replay for fixed windows with committed RTC prefixes.

At each boundary the client executes P reference continuation rows while this
owner computes z, guided reference, and actor output. The boundary observation
and pending prefix form the state. Receipts, never predictions, populate replay.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.async_inference.shared.rlt_remote import (
    RLTFinished,
    RLTPlan,
    RLTRequest,
)
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE, OBS_RTC_MASK, OBS_RTC_PREFIX
from lerobot.policies.rlt_actor.modeling_rlt_actor import validate_observation

from .replay import FixedWindowCollector, FixedWindowReplayBuffer, owned_state, replay_spec


@dataclass
class _Pending:
    plan: RLTPlan
    state: dict
    coordinates: object
    planned_action: torch.Tensor


class RLTOverlapPlanner:
    """One inference worker owns this object; generation admission is owned by transport."""

    def __init__(self, features, replay, *, session: str, actor=None, journal_dir=None, policy_version=0):
        self.features, self.replay, self.actor = features, replay, actor
        self.config = features.config
        if not self.config.rtc_prefix_steps:
            raise ValueError("RTC planning requires rtc_prefix_steps > 0")
        if replay_spec(self.config) != replay_spec(replay.config):
            raise ValueError("RTC feature/replay contracts differ")
        if actor is not None and replay_spec(actor.config) != replay_spec(self.config):
            raise ValueError("RTC actor/feature contracts differ")
        if not session or type(policy_version) is not int or policy_version < 0:
            raise ValueError("RTC requires a session and nonnegative policy version")
        self.session, self.policy_version = session, policy_version
        self.journal_dir = Path(journal_dir) if journal_dir is not None else None
        self.pending = None
        self.generation = -1
        self.ended = False

    def _validate(self, request):
        cfg = self.config
        if not isinstance(request, RLTRequest) or request.session != self.session:
            raise ValueError("RLT request belongs to another session")
        for key in ("generation", "window", "start_step"):
            if type(getattr(request, key)) is not int or getattr(request, key) < 0:
                raise ValueError(f"Invalid RLT {key}")
        if type(request.final) is not bool or type(request.dry_run) is not bool:
            raise ValueError("Invalid RLT request flags")
        if request.generation < self.generation:
            raise ValueError("Stale RLT generation")
        if request.dry_run:
            if request.previous is not None or request.final or self.generation >= 0:
                raise ValueError("Dry runs are startup-only and cannot create replay")
            return
        if request.generation > self.generation:
            if (
                request.window
                or request.start_step
                or request.previous is not None
                or request.final
                or request.committed_prefix is not None
            ):
                raise ValueError("A new RLT generation must start with a cold window")
            return
        if self.ended or self.pending is None:
            raise ValueError("RLT episode has ended or has no outstanding plan")
        plan = self.pending.plan
        receipt = request.previous
        if receipt is not None and any(
            type(getattr(receipt, key)) is not int or getattr(receipt, key) < 0
            for key in ("window", "start_step", "policy_version")
        ):
            raise ValueError("Invalid RLT receipt identifiers/version")
        if (
            receipt is None
            or receipt.window != plan.window
            or receipt.start_step != plan.start_step
            or receipt.policy_version != plan.policy_version
            or request.window != plan.window + 1
        ):
            raise ValueError("RLT receipt does not match the outstanding window/version")
        count = len(receipt.steps)
        if not 1 <= count <= cfg.execution_horizon or request.start_step != plan.start_step + count:
            raise ValueError("RLT receipt has an invalid execution span")
        ended = False
        for step in receipt.steps:
            if (
                not isinstance(step.action, torch.Tensor)
                or step.action.shape != (cfg.action_dim,)
                or not step.action.is_floating_point()
                or not torch.isfinite(step.action).all()
                or not isinstance(step.reward, (int, float))
                or isinstance(step.reward, bool)
                or not torch.isfinite(torch.tensor(step.reward))
                or any(
                    type(getattr(step, key)) is not bool
                    for key in ("terminated", "truncated", "intervention")
                )
            ):
                raise ValueError("Invalid RLT execution receipt")
            if ended:
                raise ValueError("Receipt contains commands after an episode boundary")
            ended = step.terminated or step.truncated
        if request.final != ended or (count != cfg.execution_horizon and not ended):
            raise ValueError("Partial windows require a real terminal/truncation; final must match")
        expected = plan.actions[count : count + cfg.rtc_prefix_steps]
        prefix = request.committed_prefix
        if prefix is None or prefix.shape != expected.shape or not torch.equal(prefix.cpu(), expected.cpu()):
            raise ValueError("RTC prefix differs from the committed continuation")

    def process(self, request: RLTRequest, *, admit=lambda: True, commit_guard=None):
        """Compute privately, then publish replay only if transport still admits this generation."""
        self._validate(request)
        cfg = self.config
        fresh = request.generation > self.generation and not request.dry_run
        previous = None if fresh or request.dry_run else self.pending
        if fresh or request.dry_run:
            self.features.reset()
        state = self.features.encode(request.observation, committed_prefix=request.committed_prefix)
        validate_observation(state, cfg, batched=False, strict=True)
        if not request.final:
            coordinates = self.features.snapshot_coordinates()
            normalized = state[OBS_REFERENCE].clone()
            if self.actor is not None:
                with torch.no_grad():
                    batch = {key: value[None] for key, value in state.items()}
                    normalized[: cfg.execution_horizon] = self.actor.predict_action_chunk(
                        batch, deterministic=False
                    )[0].cpu()
            normalized[: cfg.execution_horizon] = torch.where(
                state[OBS_RTC_MASK].bool()[:, None],
                state[OBS_RTC_PREFIX],
                normalized[: cfg.execution_horizon],
            )
            actions = coordinates.decode(normalized)
            if request.committed_prefix is not None:
                actions[: cfg.rtc_prefix_steps] = request.committed_prefix.cpu()
            plan = RLTPlan(
                self.session,
                request.generation,
                request.window,
                request.start_step,
                actions.clone(),
                cfg.execution_horizon,
                cfg.rtc_prefix_steps,
                self.policy_version,
                replay_spec(cfg),
            )
        with commit_guard if commit_guard is not None else nullcontext():
            # admit must serialize with transport generation fencing (the server holds its lock here).
            if not admit():
                raise InterruptedError("RLT generation was invalidated during inference")
            if previous is not None:
                collector = FixedWindowCollector(cfg)
                collector.begin(
                    previous.state,
                    planned_action=previous.planned_action,
                    episode_id=request.generation,
                    start_step=previous.plan.start_step,
                    policy_version=previous.plan.policy_version,
                )
                for step in request.previous.steps:
                    collector.record(
                        previous.coordinates.normalize(step.action),
                        step.reward,
                        terminated=step.terminated,
                        truncated=step.truncated,
                        intervention=step.intervention,
                    )
                transition = collector.finish(state)
                # One atomic small shard avoids repeatedly copying/saving the entire live ring.
                if self.journal_dir is not None:
                    shard = FixedWindowReplayBuffer(cfg, capacity=1)
                    shard.add(transition)
                    path = (
                        self.journal_dir / f"g{request.generation:012d}_s{previous.plan.start_step:012d}.pt"
                    )
                    if path.exists():
                        raise FileExistsError("Duplicate RLT journal window")
                    shard.save(path)
                self.replay.add(transition)
            if request.dry_run:
                return plan
            self.generation = request.generation
            self.ended = request.final
            self.pending = (
                None
                if request.final
                else _Pending(
                    plan,
                    owned_state(state),
                    coordinates,
                    normalized[: cfg.execution_horizon].detach().cpu().clone(),
                )
            )
            if request.final:
                return RLTFinished(self.session, request.generation, request.window, len(self.replay))
            return plan


def warmstart_rtc_actor(source, target):
    """Explicit policy-only transfer; appended RTC inputs start with zero influence.

    Optimizers and critics have different contracts and are deliberately not resumed.
    """
    old, new = source.config, target.config
    old_spec, new_spec = replay_spec(old), replay_spec(new)
    new_spec.pop("rtc_prefix_steps", None)
    if (
        old.rtc_prefix_steps
        or not new.rtc_prefix_steps
        or old_spec != new_spec
        or old.hidden_dims != new.hidden_dims
    ):
        raise ValueError("RTC warm start requires the same Stage 2 actor dimensions and frozen model")
    values = target.state_dict()
    for key, value in source.state_dict().items():
        if key == "actor.0.weight":
            values[key].zero_()
            values[key][:, : value.shape[1]].copy_(value)
        else:
            values[key].copy_(value)
    target.load_state_dict(values, strict=True)
