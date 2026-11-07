"""Sparse-terminal credit, complete planned actions, migration, and RTC dispatch deadlines."""

import copy
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from examples.tutorial.pi05 import train_rlt_stage2 as cli
from lerobot.async_inference.client.rlt_runtime import RemoteRLTClient, RLTOverlapExecutor
from lerobot.async_inference.shared.rlt_remote import RLTFailure
from lerobot.policies.rlt_actor import RLTActorPolicy
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_RTC_MASK, OBS_RTC_PREFIX
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm
from lerobot.rl.rlt.checkpoint import load_checkpoint, save_checkpoint
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.rl.rlt.rollout import ExecutedStep
from tests.async_inference.test_robot_client_deployment import _make_client
from tests.rl.test_rlt_ac import batch, learner
from tests.rl.test_rlt_replay import config, observation, transition
from tests.rl.test_rlt_rtc import SESSION, next_request, planner


@pytest.mark.parametrize("prefix", [0, 3])
@pytest.mark.parametrize("planned_value", [0.0, 0.4])
def test_short_success_reward_bootstraps_into_preceding_full_window(prefix, planned_value):
    torch.manual_seed(1)
    cfg = config(rtc_prefix_steps=prefix, reference_horizon=14)
    algorithm = learner(cfg, critic_lr=0.003, target_update_weight=1.0, actor_update_freq=100000)
    with torch.no_grad():
        for parameter in algorithm.policy.parameters():
            parameter.zero_()
        algorithm.policy.actor[-1].bias.fill_(torch.atanh(torch.tensor(planned_value)).item())

    def state(value):
        result = observation(cfg, value)
        if prefix:
            result[OBS_RTC_PREFIX] = torch.zeros(10, 2)
            result[OBS_RTC_PREFIX][:prefix] = planned_value
            result[OBS_RTC_MASK] = (torch.arange(10) < prefix).float()
        return result

    # The environment terminates on step 3. Remaining planned commands are never executed.
    row = replace(
        transition(cfg, steps=3, terminated=True, value=planned_value),
        state=state(0.0),
        next_state=state(0.1),
    )
    row.rewards.zero_()
    row.rewards[2] = 1
    replay = FixedWindowReplayBuffer(cfg, capacity=2)
    replay.add(row)
    data = replay.sample(2)
    for _ in range(350):
        algorithm.update(iter([data]))
    prepared = algorithm._prepare_batch(data)
    expected = torch.full((2,), 0.99**2)
    torch.testing.assert_close(
        algorithm.critic(prepared["state"], prepared["critic_action"])[:, 0],
        expected,
        rtol=1e-4,
        atol=1e-5,
    )
    previous = replace(
        transition(cfg),
        state=state(-1.0),
        next_state=row.state,
        planned_action=torch.full((10, 2), planned_value),
    )
    previous.rewards.zero_()
    replay.add(previous)
    history = FixedWindowReplayBuffer(cfg, capacity=1)
    history.add(previous)
    target = algorithm.target_values(algorithm._prepare_batch(history.sample(2)))
    torch.testing.assert_close(target, expected * 0.99**10, rtol=1e-4, atol=1e-5)
    # The unexecuted tail is retained only as planned context, never as a real receipt.
    assert row.action[3:].count_nonzero() == 0


def test_critic_uses_sent_prefix_and_planned_tail_without_exposing_realized_duration():
    algorithm = learner()
    data = batch(algorithm, steps=3, terminated=True)
    data["action"][:, :3] = 0.25
    data["complementary_info"]["planned_action"].fill_(0.7)
    prepared = algorithm._prepare_batch(data)
    torch.testing.assert_close(prepared["critic_action"][:, :3], torch.full((2, 3, 2), 0.25))
    torch.testing.assert_close(prepared["critic_action"][:, 3:], torch.full((2, 7, 2), 0.7))
    assert data["action"][:, 3:].count_nonzero() == 0
    assert algorithm.critic.heads[0][0].in_features == 4 + 3 + 10 * 2


def test_rtc_plan_tail_keeps_its_original_normalized_coordinates():
    from lerobot.async_inference.shared.rlt_remote import RLTRequest

    owner, features, replay = planner()
    first = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))
    original = owner.pending.planned_action.clone()
    second = owner.process(next_request(first))
    torch.testing.assert_close(replay._data[0].planned_action, original, rtol=0, atol=0)
    next_plan = owner.pending.planned_action.clone()
    owner.process(next_request(second, count=2, terminated=True))
    row = replay._data[-1]
    assert features.encoded == [0, 10, 12]
    torch.testing.assert_close(row.planned_action, next_plan, rtol=0, atol=0)
    assert row.action[2:].count_nonzero() == 0 and row.planned_action[2:].count_nonzero() > 0


@pytest.mark.parametrize("problem", ["missing", "shape", "dtype", "nan"])
def test_invalid_planned_tail_is_rejected_before_training(problem):
    algorithm = learner()
    data = batch(algorithm, steps=3, terminated=True)
    info = data["complementary_info"]
    if problem == "missing":
        del info["planned_action"]
    elif problem == "shape":
        info["planned_action"] = torch.zeros(2, 3, 2)
    elif problem == "dtype":
        info["planned_action"] = info["planned_action"].long()
    else:
        info["planned_action"][0, 4, 0] = float("nan")
    before = copy.deepcopy(algorithm.state_dict())
    with pytest.raises(ValueError, match="planned_action"):
        algorithm.update(iter([data]))
    assert algorithm.optimization_step == 0
    for key, value in before.items():
        torch.testing.assert_close(value, algorithm.state_dict()[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_partial_replay_moves_planned_tail_and_resumes(tmp_path):
    cfg = config(device="cuda", exploration_std=0.002)
    algorithm = learner(cfg)
    replay = FixedWindowReplayBuffer(cfg, capacity=2)
    replay.add(transition(cfg, steps=3, terminated=True, value=0.25))
    data = replay.sample(2)
    data["complementary_info"]["planned_action"].fill_(0.7)
    prepared = algorithm._prepare_batch(data)
    assert prepared["critic_action"].device.type == "cuda"
    torch.testing.assert_close(prepared["critic_action"][:, 3:], torch.full((2, 7, 2), 0.7, device="cuda"))
    stats = algorithm.update(iter([data]))
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.to_log_dict().values())
    checkpoint = save_checkpoint(tmp_path / "partial", algorithm, replay)
    restored, restored_replay = load_checkpoint(checkpoint, device="cuda")
    restored.update(iter([restored_replay.sample(2)]))
    assert restored.optimization_step == 2


@pytest.mark.parametrize("partial", [False, True])
def test_legacy_replay_upgrades_only_complete_windows(partial):
    cfg = config()
    replay = FixedWindowReplayBuffer(cfg, capacity=2)
    replay.add(transition(steps=3 if partial else 10, terminated=partial, value=0.4))
    old = replay.state_dict()
    old["format_version"] = 1
    for row in old["data"]:
        del row["planned_action"]
    restored = FixedWindowReplayBuffer(cfg, capacity=2)
    if partial:
        with pytest.raises(ValueError, match="Legacy partial.*recollect"):
            restored.load_state_dict(old)
        assert len(restored) == 0
    else:
        restored.load_state_dict(old)
        data = restored.sample(2)
        torch.testing.assert_close(data["complementary_info"]["planned_action"], data["action"])
        assert restored.state_dict()["format_version"] == 2


def test_replay_v2_requires_and_owns_the_complete_plan(tmp_path):
    cfg = config()
    replay = FixedWindowReplayBuffer(cfg, capacity=2)
    row = transition(steps=2, terminated=True, value=0.3)
    replay.add(row)
    row.planned_action.zero_()
    data = replay.sample(1)
    assert data["complementary_info"]["planned_action"].min() == pytest.approx(0.3)
    data["complementary_info"]["planned_action"].zero_()
    assert replay.sample(1)["complementary_info"]["planned_action"].min() == pytest.approx(0.3)
    saved = replay.state_dict()
    del saved["data"][0]["planned_action"]
    with pytest.raises(ValueError, match="v2 must contain planned_action"):
        replay.load_state_dict(saved)
    with pytest.raises(ValueError, match="complete planned_action"):
        replace(row, planned_action=None).validate(cfg)


def test_old_critic_checkpoint_rejects_resume_but_actor_remains_loadable(tmp_path):
    algorithm = learner()
    replay = FixedWindowReplayBuffer(algorithm.policy.config, capacity=2)
    replay.add(transition())
    saved = save_checkpoint(tmp_path / "checkpoint", algorithm, replay)
    manifest = json.loads((saved / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    manifest["format_version"] = 1
    (saved / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Legacy RLT critic/optimizer"):
        load_checkpoint(saved)
    restored = RLTActorPolicy.from_pretrained(saved / "actor", strict=True)
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_cli_warm_starts_actor_with_fresh_corrected_critics(tmp_path, monkeypatch):
    cfg = config()
    actor = RLTActorPolicy(cfg)
    actor.save_pretrained(tmp_path / "old_actor")
    replay = FixedWindowReplayBuffer(cfg, capacity=2)
    replay.add(transition())
    replay.save(tmp_path / "replay.pt")
    captured = []

    def construct(policy, config):
        captured.append(copy.deepcopy(policy.state_dict()))
        return RLTACAlgorithm(policy, config)

    monkeypatch.setattr(cli, "RLTACAlgorithm", construct)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rlt_stage2.py",
            "--replay",
            str(tmp_path / "replay.pt"),
            "--actor-checkpoint",
            str(tmp_path / "old_actor"),
            "--output-dir",
            str(tmp_path / "new"),
            "--steps",
            "1",
            "--batch-size",
            "2",
        ],
    )
    cli.main()
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, captured[0][key], rtol=0, atol=0)
    restored, _ = load_checkpoint(tmp_path / "new")
    assert restored.optimization_step == 1
    assert restored.policy.config.hidden_dims == cfg.hidden_dims


class ImmediatePool:
    """A response that has already completed when execution reaches the next prefix tick."""

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, **kwargs):
        pass


def immediate_scheduler(exchange, **kwargs):
    scheduler = RLTOverlapExecutor(exchange, session=SESSION, **kwargs)
    scheduler.pool.shutdown(wait=True)
    scheduler.pool = ImmediatePool()
    return scheduler


@pytest.mark.parametrize("failure", ["response", "exception", "malformed"])
def test_completed_inference_failure_stops_before_any_unsent_prefix(failure):
    owner, _, _ = planner()
    sent = []

    def exchange(request):
        if request.window == 1:
            if failure == "response":
                return RLTFailure("model failed")
            if failure == "exception":
                raise OSError("RPC failed")
            return None
        return owner.process(request)

    scheduler = immediate_scheduler(exchange)

    def step(action):
        sent.append(action)
        return ExecutedStep({"value": len(sent)}, action, 0.0)

    try:
        with pytest.raises((RuntimeError, OSError, ValueError)):
            scheduler.run({"value": 0}, step, generation=1)
        assert len(sent) == 10
    finally:
        scheduler.close()


def test_failure_arriving_during_prefix_stops_at_the_next_policy_tick():
    owner, _, _ = planner()
    prefix_entered = threading.Event()
    futures, sent = {}, []

    def exchange(request):
        if request.window == 1:
            assert prefix_entered.wait(2)
            raise OSError("RPC failed during prefix")
        return owner.process(request)

    scheduler = RLTOverlapExecutor(exchange, session=SESSION)
    submit = scheduler.pool.submit

    def capture_future(function, request):
        future = submit(function, request)
        futures[request.window] = future
        return future

    scheduler.pool.submit = capture_future

    def step(action):
        sent.append(action)
        if len(sent) == 11:
            prefix_entered.set()
            assert isinstance(futures[1].exception(timeout=2), OSError)
        return ExecutedStep({"value": len(sent)}, action, 0.0)

    try:
        with pytest.raises(OSError, match="during prefix"):
            scheduler.run({"value": 0}, step, generation=1)
        assert len(sent) == 11
    finally:
        prefix_entered.set()
        scheduler.close()


def test_inflight_cold_request_is_bounded_by_sensor_deadline():
    owner, features, _ = planner()
    entered, release = threading.Event(), threading.Event()

    def hook(value):
        entered.set()
        assert release.wait(2)

    features.hook = hook
    scheduler = RLTOverlapExecutor(owner.process, session=SESSION, timeout=1, max_observation_age_s=0.03)
    try:
        with pytest.raises(TimeoutError, match="cold observation expired"):
            scheduler.run({"value": 0}, lambda action: pytest.fail("expired motion"), generation=1)
        assert entered.is_set() and not release.is_set()
    finally:
        release.set()
        scheduler.pool.submit(lambda: None).result(timeout=2)
        scheduler.close()


@pytest.mark.parametrize("delay,expired", [(0.05, False), (0.15, True)])
def test_cold_deadline_includes_sensor_age_and_inference_time(monkeypatch, delay, expired):
    owner, _, _ = planner()
    clock, sent = [10.0], []
    monkeypatch.setattr("lerobot.async_inference.client.rlt_runtime.time.monotonic", lambda: clock[0])

    def exchange(request):
        if request.window == 0:
            clock[0] += delay
        return owner.process(request)

    scheduler = immediate_scheduler(exchange, max_observation_age_s=0.2)

    def step(action):
        sent.append(action)
        return ExecutedStep({"value": 1}, action, 1.0, terminated=True)

    try:
        if expired:
            with pytest.raises(TimeoutError, match="cold observation expired"):
                scheduler.run({"value": 0}, step, generation=1, observation_started_at=9.9)
            assert not sent
        else:
            scheduler.run({"value": 0}, step, generation=1, observation_started_at=9.9)
            assert len(sent) == 1
    finally:
        scheduler.close()


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -1.0])
def test_bad_or_expired_initial_sensor_timestamp_never_submits_inference(stamp):
    scheduler = immediate_scheduler(lambda request: pytest.fail("stale inference was submitted"))
    try:
        with pytest.raises((ValueError, TimeoutError)):
            scheduler.run(
                {}, lambda action: pytest.fail("stale motion"), generation=1, observation_started_at=stamp
            )
    finally:
        scheduler.close()


def test_cold_source_deadline_is_checked_again_after_motor_lock_wait(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    client.config.observation_max_age_s = 1.0
    runtime = RemoteRLTClient.__new__(RemoteRLTClient)
    runtime.client, runtime.cancelled = client, threading.Event()
    runtime.scheduler = SimpleNamespace(close=lambda: None)
    client._rlt_runtime = runtime
    runtime.reward = lambda *args: pytest.fail("expired command reached reward")
    assert client.start_rollout()
    client.action_interpolator.add(torch.zeros(3))
    while client.action_interpolator.get() is not None:
        pass
    entered, sends = threading.Event(), []
    add = client.action_interpolator.add

    def add_action(action):
        add(action)
        entered.set()

    monkeypatch.setattr(client.action_interpolator, "add", add_action)
    client.robot.send_action = lambda action: sends.append(action) or action
    client._robot_command_lock.acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime._step,
                torch.ones(3),
                client.rollout_generation,
                runtime.cancelled,
                source_deadline=time.monotonic() + 0.02,
            )
            try:
                assert entered.wait(2)
                time.sleep(0.04)
            finally:
                client._robot_command_lock.release()
            with pytest.raises(TimeoutError, match="cold observation expired before motor dispatch"):
                future.result(timeout=2)
        assert not sends
    finally:
        client.stop()
