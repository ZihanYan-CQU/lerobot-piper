"""Post-dispatch sensor causality and atomic manual-feedback admission, with MockRobot only."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.client.rlt_runtime import ManualReward, RemoteRLTClient
from lerobot.rl.rlt.rollout import RewardSignal
from tests.async_inference.test_robot_client_deployment import _make_client


def runtime_for(client):
    runtime = RemoteRLTClient.__new__(RemoteRLTClient)
    runtime.client, runtime.reward = client, ManualReward()
    runtime.cancelled = threading.Event()
    runtime.scheduler = SimpleNamespace(close=lambda: None)
    client._rlt_runtime = runtime
    return runtime


@pytest.mark.parametrize("continuous", [False, True])
def test_step_rejects_observation_captured_during_motor_lock_wait(tmp_path, monkeypatch, continuous):
    client = _make_client(tmp_path)
    client.config.continuous_observation = continuous
    client.config.fps = 1000
    client.config.observation_max_age_s = 0.1
    runtime = runtime_for(client)
    assert client.start_rollout()
    client.action_interpolator.add(torch.zeros(3))
    while client.action_interpolator.get() is not None:
        pass
    generation, entered, sent, rewarded = client.rollout_generation, threading.Event(), [], []
    add = client.action_interpolator.add

    def add_action(action):
        add(action)
        entered.set()

    monkeypatch.setattr(client.action_interpolator, "add", add_action)
    client.robot.send_action = lambda action: sent.append(time.monotonic()) or action
    runtime.reward = lambda raw, action: rewarded.append(raw) or RewardSignal(0.0)
    client._robot_command_lock.acquire()
    locked = True
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime._step, torch.ones(3), generation, runtime.cancelled)
            try:
                assert entered.wait(2)
                sample = client._capture_observation(generation, buffered_cameras=False)
                with client._lifecycle_changed:
                    client._latest_observation = sample
                    client._lifecycle_changed.notify_all()
                if not continuous:
                    monkeypatch.setattr(client, "_capture_observation", lambda *a, **kw: sample)
            finally:
                client._robot_command_lock.release()
                locked = False
            with pytest.raises(TimeoutError, match="post-command|predates"):
                future.result(timeout=2)
        assert len(sent) == 2 and sample.started_at < sent[0]
        assert not rewarded and client.latest_action == -1
    finally:
        if locked:
            client._robot_command_lock.release()
        client.stop()


def test_step_waits_for_sensors_after_last_interpolated_send(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    runtime = runtime_for(client)
    assert client.start_rollout()
    client.action_interpolator.add(torch.zeros(3))
    while client.action_interpolator.get() is not None:
        pass
    generation = client.rollout_generation
    client._latest_observation = client._capture_observation(generation, buffered_cameras=False)
    capture_entered, selected, sent = threading.Event(), [], []
    original_capture = runtime._capture

    def capture(generation, *, after):
        capture_entered.set()
        result = original_capture(generation, after=after)
        selected.append((after, result[1]))
        return result

    monkeypatch.setattr(runtime, "_capture", capture)
    client.robot.send_action = lambda action: sent.append(time.monotonic()) or action
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime._step, torch.ones(3), generation, runtime.cancelled)
            assert capture_entered.wait(2)
            fresh = client._capture_observation(generation, buffered_cameras=False)
            with client._lifecycle_changed:
                client._latest_observation = fresh
                client._lifecycle_changed.notify_all()
            result = future.result(timeout=2)
        assert len(sent) == 2 and selected[0][0] >= sent[-1]
        assert selected[0][1] is fresh
        torch.testing.assert_close(result.executed_action, torch.ones(3))
        assert client.latest_action == 0
    finally:
        client.stop()


@pytest.mark.parametrize("stale_field", ["started_at", "oldest_camera_at"])
@pytest.mark.parametrize("continuous", [False, True])
def test_capture_enforces_post_send_bound_for_state_and_camera(
    tmp_path, monkeypatch, stale_field, continuous
):
    client = _make_client(tmp_path)
    client.config.continuous_observation = continuous
    client.config.observation_max_age_s = 0.02
    runtime = runtime_for(client)
    assert client.start_rollout()
    generation, after = client.rollout_generation, time.monotonic()
    sample = client._capture_observation(generation, buffered_cameras=False)
    stale = replace(sample, **{stale_field: after - 0.01})
    if continuous:
        client._latest_observation = stale
    else:
        monkeypatch.setattr(client, "_capture_observation", lambda *a, **kw: stale)
    try:
        with pytest.raises(TimeoutError, match="predates|post-command"):
            runtime._capture(generation, after=after)
    finally:
        client.stop()


@pytest.mark.parametrize("outcome", ["success", "failure", "truncate"])
def test_feedback_after_start_survives_worker_wakeup(tmp_path, outcome):
    client = _make_client(tmp_path)
    runtime, observed = runtime_for(client), []
    runtime._capture = lambda generation: (
        client.robot.get_observation(),
        client._capture_observation(generation, buffered_cameras=False),
    )
    client._start_observation_collector = lambda: None

    def execute(*args, **kwargs):
        observed.append(runtime.reward({}, torch.zeros(3)))
        client.shutdown_event.set()

    runtime.scheduler.run = execute
    try:
        with client._lifecycle_lock:
            worker = threading.Thread(target=runtime.run)
            worker.start()
            assert client.start_rollout()
            event = runtime.cancelled
            assert runtime.mark_manual_outcome(outcome)
        worker.join(2)
        assert not worker.is_alive() and len(observed) == 1
        assert event is runtime.cancelled
        assert observed[0] == RewardSignal(
            float(outcome == "success"), terminated=outcome != "truncate", truncated=outcome == "truncate"
        )
    finally:
        client.stop()


def test_start_clears_previous_feedback_and_reset_closes_feedback_admission(tmp_path):
    client = _make_client(tmp_path)
    runtime = runtime_for(client)
    try:
        assert not runtime.mark_manual_outcome("success")
        runtime.reward.mark("failure")  # Stale feedback left by an earlier rollout.
        assert client.start_rollout()
        assert runtime.reward({}, None) == RewardSignal(0.0)
        assert runtime.mark_manual_outcome("success")
        previous_cancelled = runtime.cancelled
        assert client.reset_rollout()
        assert previous_cancelled.is_set()
        assert not runtime.mark_manual_outcome("failure")
        assert client.start_rollout()
        assert runtime.cancelled is not previous_cancelled and not runtime.cancelled.is_set()
        assert runtime.reward({}, None) == RewardSignal(0.0)
    finally:
        client.stop()
