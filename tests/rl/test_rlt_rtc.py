"""Real overlap, immutable anchors, receipts, replay/learner compatibility and fencing."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.client.rlt_runtime import RLTOverlapExecutor
from lerobot.async_inference.shared.rlt_remote import RLTReceipt, RLTRequest, RLTStepReceipt
from lerobot.policies.rlt_actor import RLTActorPolicy
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE, OBS_RTC_MASK, OBS_RTC_PREFIX
from lerobot.policies.rlt_actor.modeling_rlt_actor import validate_observation
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm, RLTACConfig
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.rl.rlt.rollout import ExecutedStep
from lerobot.rl.rlt.rtc import RLTOverlapPlanner, warmstart_rtc_actor
from tests.rl.test_rlt_replay import config, observation

SESSION = "a" * 32


class Coordinates:
    def __init__(self, anchor):
        self.anchor = anchor

    def normalize(self, absolute):
        return (absolute - self.anchor) / 2

    def decode(self, normalized):
        return normalized * 2 + self.anchor


class Features:
    def __init__(self, cfg=None):
        self.config = cfg or config(rtc_prefix_steps=3, reference_horizon=14)
        self.encoded = []
        self.policy = SimpleNamespace(config=SimpleNamespace(image_features={}))
        self.preprocessor = SimpleNamespace(steps=[])
        self.hook = None

    def reset(self):
        pass

    def encode(self, raw, *, committed_prefix=None):
        value = raw["value"]
        self.encoded.append(value)
        if self.hook:
            self.hook(value)
        self.coordinates = Coordinates(value)
        state = observation(self.config, value * 0.001)
        state[OBS_REFERENCE] = (
            torch.arange(self.config.reference_horizon).float()[:, None].repeat(1, self.config.action_dim)
            * 0.01
            + 0.1
        )
        prefix = torch.zeros(self.config.execution_horizon, self.config.action_dim)
        mask = torch.zeros(self.config.execution_horizon)
        if committed_prefix is not None:
            prefix[: self.config.rtc_prefix_steps] = self.coordinates.normalize(committed_prefix)
            mask[: self.config.rtc_prefix_steps] = 1
        state[OBS_RTC_PREFIX], state[OBS_RTC_MASK] = prefix, mask
        return state

    def snapshot_coordinates(self):
        return self.coordinates


def planner(tmp_path=None):
    features = Features()
    replay = FixedWindowReplayBuffer(features.config, capacity=32)
    owner = RLTOverlapPlanner(features, replay, session=SESSION, journal_dir=tmp_path)
    return owner, features, replay


def next_request(plan, *, count=None, terminated=False, truncated=False):
    count = plan.execution_horizon if count is None else count
    rows = tuple(
        RLTStepReceipt(
            plan.actions[index].clone() + 0.02,
            float(index + 1),
            terminated=terminated and index == count - 1,
            truncated=truncated and index == count - 1,
            intervention=index == 2,
        )
        for index in range(count)
    )
    return RLTRequest(
        SESSION,
        plan.generation,
        plan.window + 1,
        plan.start_step + count,
        {"value": plan.start_step + count},
        plan.actions[count : count + plan.prefix_steps].clone(),
        RLTReceipt(plan.window, plan.start_step, plan.policy_version, rows),
        final=terminated or truncated,
    )


def test_overlap_executes_prefix_during_next_inference_and_records_actual_rows(tmp_path):
    owner, features, replay = planner(tmp_path)
    inference_started, executed_prefix = threading.Event(), threading.Event()

    def hook(value):
        if value == 10:
            inference_started.set()
            assert executed_prefix.wait(1)

    features.hook = hook
    sent = []
    scheduler = RLTOverlapExecutor(owner.process, session=SESSION, fps=50, margin=0)

    def step(action):
        sent.append(action.clone())
        if len(sent) == 11:
            assert inference_started.wait(1)
            executed_prefix.set()
        time.sleep(0.02)
        return ExecutedStep(
            {"value": len(sent)},
            action + 0.02,
            float(len(sent)),
            terminated=len(sent) == 27,
            intervention=len(sent) == 3,
        )

    try:
        scheduler.run({"value": 0}, step, generation=1)
    finally:
        executed_prefix.set()
        scheduler.close()
    assert len(sent) == 27 and len(replay) == 3
    assert features.encoded == [0, 10, 20, 27]
    rows = replay._data
    assert [row.start_step for row in rows] == [0, 10, 20]
    assert [row.valid_steps for row in rows] == [10, 10, 7]
    for first, second in zip(rows, rows[1:], strict=False):
        for key in first.next_state:
            torch.testing.assert_close(first.next_state[key], second.state[key], rtol=0, atol=0)
    torch.testing.assert_close(rows[0].action, (torch.stack(sent[:10]) + 0.02) / 2)
    torch.testing.assert_close(rows[1].action, (torch.stack(sent[10:20]) + 0.02 - 10) / 2)
    assert rows[1].state[OBS_RTC_MASK].sum() == 3
    assert rows[0].intervention[2]
    assert rows[-1].action[7:].count_nonzero() == 0
    restored = FixedWindowReplayBuffer.load(tmp_path, features.config)
    assert len(restored) == 3
    algorithm = RLTACAlgorithm(RLTActorPolicy(features.config), RLTACConfig(policy_config=features.config))
    assert algorithm.update(iter([restored.sample(4)])).losses["critic_loss"] >= 0


@pytest.mark.parametrize("truncated", [False, True])
def test_terminal_during_committed_prefix_never_executes_remaining_actions(truncated):
    owner, features, replay = planner()
    scheduler = RLTOverlapExecutor(owner.process, session=SESSION, fps=50, margin=0)
    count = 0

    def step(action):
        nonlocal count
        count += 1
        time.sleep(0.02)
        return ExecutedStep(
            {"value": count},
            action,
            1,
            terminated=count == 11 and not truncated,
            truncated=count == 11 and truncated,
        )

    try:
        scheduler.run({"value": 0}, step, generation=1)
    finally:
        scheduler.close()
    assert count == 11 and len(replay) == 2
    assert replay._data[1].valid_steps == 1
    assert replay._data[1].truncated is truncated
    assert features.encoded == [0, 10, 11]


def test_late_inference_stops_at_prefix_budget_and_does_not_fabricate_window():
    owner, features, replay = planner()
    release = threading.Event()
    features.hook = lambda value: release.wait(1) if value == 10 else None
    scheduler = RLTOverlapExecutor(owner.process, session=SESSION, fps=100, margin=0)
    sent = []

    def step(action):
        sent.append(action.clone())
        time.sleep(0.01)
        return ExecutedStep({"value": len(sent)}, action, 0)

    try:
        with pytest.raises(TimeoutError, match="prefix"):
            scheduler.run({"value": 0}, step, generation=1)
        assert len(sent) == 13
        # The 3 executed rows have no valid complete-window next state and are never committed.
        release.set()
        scheduler.pool.submit(lambda: None).result(timeout=1)
        assert len(replay) == 1
        assert replay._data[0].valid_steps == 10
    finally:
        release.set()
        scheduler.close()


@pytest.mark.parametrize("bad", ["window", "version", "prefix", "short", "post_terminal", "nan"])
def test_bad_receipts_are_rejected_before_feature_computation_or_replay(bad):
    owner, features, replay = planner()
    plan = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
    request = next_request(plan)
    if bad == "window":
        request = replace(request, previous=replace(request.previous, window=99))
    elif bad == "version":
        request = replace(request, previous=replace(request.previous, policy_version=99))
    elif bad == "prefix":
        request = replace(request, committed_prefix=request.committed_prefix + 1)
    elif bad == "short":
        request = next_request(plan, count=2)
    else:
        steps = list(request.previous.steps)
        steps[0] = replace(
            steps[0], terminated=bad == "post_terminal", reward=float("nan") if bad == "nan" else 0
        )
        request = replace(request, previous=replace(request.previous, steps=tuple(steps)))
    with pytest.raises(ValueError):
        owner.process(request)
    assert features.encoded == [0] and len(replay) == 0


def test_dry_runs_reset_generation_and_cancelled_publication_never_make_partial_replay():
    owner, features, replay = planner()
    owner.process(RLTRequest(SESSION, 0, 0, 0, {"value": 0}, dry_run=True))
    assert len(replay) == 0 and owner.pending is None
    plan = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
    with pytest.raises(InterruptedError):
        owner.process(next_request(plan), admit=lambda: False)
    assert len(replay) == 0
    newer = owner.process(RLTRequest(SESSION, 3, 0, 0, {"value": 50}))
    assert newer.generation == 3 and len(replay) == 0
    with pytest.raises(ValueError, match="Stale"):
        owner.process(next_request(plan))


def test_actor_clamps_prefix_and_excludes_it_from_actor_updates():
    owner, features, replay = planner()
    plan = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
    plan = owner.process(next_request(plan))
    state = owner.pending.state
    actor = RLTActorPolicy(features.config)
    batch = {key: value[None] for key, value in state.items()}
    output = actor(batch, deterministic=False)
    torch.testing.assert_close(output[0, :3], state[OBS_RTC_PREFIX][:3], rtol=0, atol=0)
    output.sum().backward()
    # Committed rows do not create gradients on their output-layer parameters.
    assert actor.actor[-1].weight.grad[: 3 * features.config.action_dim].count_nonzero() == 0
    owner.process(next_request(plan, count=1, terminated=True))
    only_prefix = FixedWindowReplayBuffer(features.config, capacity=1)
    only_prefix.add(replay._data[-1])
    algorithm = RLTACAlgorithm(actor, RLTACConfig(policy_config=features.config))
    weights = {key: value.clone() for key, value in actor.state_dict().items()}
    stats = algorithm.update(iter([only_prefix.sample(2)]))
    assert "actor_loss" not in stats.losses
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_explicit_stage2_warmstart_keeps_uncommitted_actor_outputs():
    old = RLTActorPolicy(config(reference_horizon=14))
    cfg = config(reference_horizon=14, rtc_prefix_steps=3)
    new = RLTActorPolicy(cfg)
    warmstart_rtc_actor(old, new)
    state = observation(cfg)
    state[OBS_RTC_PREFIX] = torch.zeros(10, 2)
    state[OBS_RTC_MASK] = torch.zeros(10)
    output = new({key: value[None] for key, value in state.items()}, deterministic=True)
    expected = old({key: value[None] for key, value in observation(old.config).items()}, deterministic=True)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    malformed = dict(state)
    malformed[OBS_RTC_MASK] = torch.tensor([1, 0, 1, 1, 0, 0, 0, 0, 0, 0]).float()
    with pytest.raises(ValueError, match="committed prefix"):
        validate_observation(malformed, cfg, batched=False)


def test_same_shape_actor_snapshot_with_different_rtc_prefix_contract_is_rejected():
    cfg = config(reference_horizon=14, rtc_prefix_steps=3)
    actor = RLTActorPolicy(cfg)
    algorithm = RLTACAlgorithm(actor, RLTACConfig(policy_config=cfg))
    payload = algorithm.get_weights()
    other = RLTActorPolicy(config(reference_horizon=14, rtc_prefix_steps=2))
    before = {key: value.clone() for key, value in other.state_dict().items()}
    with pytest.raises(ValueError, match="RTC replay contract"):
        other.load_rollout_weights(payload)
    for key, value in other.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_cancellation_while_waiting_for_cold_plan_sends_no_commands():
    owner, features, replay = planner()
    started, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    def hook(value):
        started.set()
        release.wait(1)

    features.hook = hook
    scheduler = RLTOverlapExecutor(owner.process, session=SESSION)
    errors = []

    def run():
        try:
            scheduler.run(
                {"value": 0},
                lambda _: pytest.fail("cancelled cold plan executed"),
                generation=1,
                cancelled=cancelled,
            )
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(1)
        cancelled.set()
        worker.join(1)
        assert not worker.is_alive() and isinstance(errors[0], InterruptedError)
        assert len(replay) == 0
    finally:
        release.set()
        scheduler.close()


def test_rtc_journal_trains_saves_and_resumes_with_real_cli(tmp_path, monkeypatch):
    from examples.tutorial.pi05.train_rlt_stage2 import main
    from lerobot.rl.rlt.checkpoint import load_checkpoint

    journal = tmp_path / "journal"
    owner, _, _ = planner(journal)
    first = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
    second = owner.process(next_request(first))
    owner.process(next_request(second, count=5, truncated=True))
    output = tmp_path / "checkpoint"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rlt_stage2.py",
            "--replay",
            str(journal),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--hidden-dims",
            "16",
            "16",
        ],
    )
    main()
    algorithm, replay = load_checkpoint(output)
    assert algorithm.optimization_step == 2 and len(replay) == 2
    assert algorithm.policy.config.rtc_prefix_steps == 3
    resumed = tmp_path / "resumed"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rlt_stage2.py",
            "--resume",
            str(output),
            "--output-dir",
            str(resumed),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--batch-size",
            "2",
        ],
    )
    main()
    algorithm, replay = load_checkpoint(resumed)
    assert algorithm.optimization_step == 3 and replay.config.rtc_prefix_steps == 3
