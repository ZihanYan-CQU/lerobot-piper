"""Execution-window boundaries, reset fencing, controller output and reward alignment."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.async_inference_local.control.robot_io import RobotIO
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_RLT
from lerobot.policies.rlt_actor.modeling_rlt_actor import RLTActorPolicy
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm, RLTACConfig
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.rl.rlt.rollout import ExecutedStep, FixedWindowRollout, RewardSignal, RobotWindowStep
from tests.rl.test_rlt_replay import config, observation


class Features:
    def __init__(self):
        self.config = config()
        self.encoded = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def encode(self, raw):
        self.encoded.append(raw["value"])
        return observation(self.config, float(raw["value"]))

    def executable_actions(self, actions):
        assert actions.shape == (10, 2)
        return actions * 2 + 100

    def normalize_executed(self, command):
        return (command - 100) / 2


def rollout():
    features = Features()
    replay = FixedWindowReplayBuffer(features.config, capacity=8)
    runner = FixedWindowRollout(features, replay)
    runner.begin_episode({"value": 0}, episode_id=1)
    return runner, features, replay


def test_commits_ten_actual_commands_before_next_feature_reference():
    runner, features, replay = rollout()
    calls = []

    def execute(command):
        calls.append(command.clone())
        return ExecutedStep(
            {"value": len(calls)}, command + 0.2, float(len(calls)), intervention=len(calls) == 2
        )

    first = runner.collect_window(execute, policy_version=4)
    assert len(calls) == 10 and features.encoded == [0, 10]
    assert first.valid_steps == 10 and first.policy_version == 4
    assert first.action.shape == (10, 2)
    torch.testing.assert_close(first.action, torch.full((10, 2), 0.6), atol=2e-6, rtol=0)
    torch.testing.assert_close(first.rewards, torch.arange(1, 11).float())
    second = runner.collect_window(execute)
    assert features.encoded == [0, 10, 20]
    for key in first.next_state:
        torch.testing.assert_close(second.state[key], first.next_state[key], rtol=0, atol=0)
    assert second.start_step == 10 and first.start_step == 0 and len(replay) == 2


@pytest.mark.parametrize("truncated", [False, True])
def test_early_boundary_keeps_only_executed_prefix_and_requires_reset(truncated):
    runner, _, replay = rollout()
    calls = []

    def execute(command):
        calls.append(1)
        end = len(calls) == 3
        return ExecutedStep(
            {"value": 3}, command, 1.0, terminated=end and not truncated, truncated=end and truncated
        )

    row = runner.collect_window(execute)
    assert len(calls) == 3 and len(replay) == 1
    assert row.valid_steps == 3
    assert row.action[3:].count_nonzero() == 0
    with pytest.raises(RuntimeError, match="begin_episode"):
        runner.collect_window(execute)
    runner.begin_episode({"value": 50}, episode_id=2)
    assert runner._state[OBS_RLT][0] == 50


@pytest.mark.parametrize("failure", ["cancel", "disconnect", "bad_reward", "bad_result"])
def test_cancel_or_fault_discards_partial_window_and_blocks_stale_resume(failure):
    runner, _, replay = rollout()
    calls = []

    def execute(command):
        calls.append(1)
        if len(calls) == 2:
            if failure == "cancel":
                runner.stop_event.set()
            elif failure == "disconnect":
                raise OSError("CAN disconnected")
            elif failure == "bad_result":
                return None
        return ExecutedStep(
            {"value": len(calls)},
            command,
            float("nan") if failure == "bad_reward" and len(calls) == 2 else 0.0,
        )

    if failure == "cancel":
        assert runner.collect_window(execute) is None
    else:
        with pytest.raises((OSError, ValueError, TypeError)):
            runner.collect_window(execute)
    assert len(replay) == 0
    assert not runner.collector.active
    runner.stop_event.clear()
    with pytest.raises(RuntimeError, match="begin_episode"):
        runner.collect_window(execute)


def test_cancel_during_final_feature_pass_does_not_publish_old_window():
    runner, features, replay = rollout()
    original = features.encode

    def encode(raw):
        runner.stop_event.set()
        return original(raw)

    features.encode = encode
    assert runner.collect_window(lambda command: ExecutedStep({"value": 1}, command, 0.0)) is None
    assert len(replay) == 0


def test_actor_selection_is_frozen_for_one_window():
    runner, features, replay = rollout()
    predictions = []
    actor = SimpleNamespace(config=features.config)
    actor.value = 0.1

    def predict(state, **kwargs):
        predictions.append(1)
        return torch.full((1, 10, 2), actor.value)

    actor.predict_action_chunk = predict
    runner.actor = actor

    def execute(command):
        actor.value = 0.9  # Learner sync cannot replace already committed actions.
        return ExecutedStep({"value": 1}, command, 0.0)

    row = runner.collect_window(execute, use_actor=True)
    assert predictions == [1]
    torch.testing.assert_close(row.action, torch.full((10, 2), 0.1), atol=2e-6, rtol=0)
    assert len(replay) == 1


def test_actor_snapshot_sync_and_replay_version_only_change_between_windows():
    runner, features, _ = rollout()
    runner.actor = RLTActorPolicy(features.config)
    learner = RLTACAlgorithm(RLTActorPolicy(features.config), RLTACConfig(policy_config=features.config))
    learner.optimization_step = 3
    runner.load_actor_weights(learner.get_weights())
    expected = runner.actor.predict_action_chunk({key: value[None] for key, value in runner._state.items()})[
        0
    ]
    learner.optimization_step = 4
    with torch.no_grad():
        for parameter in learner.policy.parameters():
            parameter.add_(0.1)
    payload = learner.get_weights()
    calls = []

    def execute(command):
        calls.append(1)
        with pytest.raises(RuntimeError, match="boundaries"):
            runner.load_actor_weights(payload)
        with pytest.raises(RuntimeError, match="Cannot reset"):
            runner.begin_episode({"value": 10}, episode_id=2)
        with pytest.raises(RuntimeError, match="already active"):
            runner.collect_window(execute, use_actor=True)
        return ExecutedStep({"value": 1}, command, 0.0)

    row = runner.collect_window(execute, use_actor=True, deterministic=True)
    assert row.policy_version == 3 and len(calls) == 10
    torch.testing.assert_close(row.action, expected, atol=2e-6, rtol=0)
    runner.load_actor_weights(payload)
    row = runner.collect_window(
        lambda command: ExecutedStep({"value": 2}, command, 0.0), use_actor=True, deterministic=True
    )
    assert row.policy_version == runner.actor.rollout_version == 4
    with pytest.raises(ValueError, match="match the loaded"):
        runner.collect_window(execute, use_actor=True, policy_version=3)


def test_invalid_window_metadata_is_rejected_before_sending_commands():
    runner, _, replay = rollout()
    calls = []
    with pytest.raises(ValueError, match="policy_version"):
        runner.collect_window(lambda command: calls.append(command), policy_version=-1)
    assert not calls and len(replay) == 0


def test_existing_robot_io_bridge_records_reported_output_and_real_reward():
    calls = []

    def send(action, *, deadline, stop_event):
        assert time.monotonic() <= deadline and not stop_event.is_set()
        calls.append(action.copy())
        return np.minimum(action, 0.25)

    def capture(age):
        stamp = time.monotonic()
        return {"measured": 4.0}, stamp, stamp

    io = SimpleNamespace(send=send, capture=capture, action_keys=["joint_1.pos", "gripper.pos"])
    step = RobotWindowStep(
        io,
        lambda raw: {"value": raw["measured"]},
        lambda raw, sent: RewardSignal(float(sent.sum()), terminated=True),
        action_names=["joint_1", "gripper"],
        fps=1e6,
    )
    result = step(torch.tensor([0.8, -0.5]))
    torch.testing.assert_close(result.executed_action, torch.tensor([0.25, -0.5]))
    assert result.reward == pytest.approx(-0.25) and result.terminated
    assert result.observation == {"value": 4.0}


def test_robot_io_bridge_rejects_stale_next_state_and_cancel_before_send():
    sends = []
    io = SimpleNamespace(
        send=lambda values, **kwargs: sends.append(values) or values,
        capture=lambda age: ({}, time.monotonic(), time.monotonic() - 2),
        action_keys=["joint_1.pos", "gripper.pos"],
    )
    event = Event()
    step = RobotWindowStep(
        io,
        lambda raw: raw,
        lambda raw, sent: RewardSignal(0.0),
        action_names=["joint_1", "gripper"],
        fps=1e6,
        stop_event=event,
    )
    event.set()
    with pytest.raises(InterruptedError):
        step(torch.zeros(2))
    assert sends == []
    event.clear()
    with pytest.raises(TimeoutError, match="stale"):
        step(torch.zeros(2))


def test_robot_io_action_order_is_checked_before_sending_any_command():
    io = SimpleNamespace(action_keys=["gripper.pos", "joint_1.pos"])
    with pytest.raises(ValueError, match="action order"):
        RobotWindowStep(
            io, lambda obs: obs, lambda obs, sent: RewardSignal(0.0), action_names=["joint_1", "gripper"]
        )


@pytest.mark.parametrize("post_command", [False, True])
def test_robot_io_freshness_is_measured_after_send_returns(monkeypatch, post_command):
    clock, rewarded = [10.0], []
    monkeypatch.setattr("lerobot.rl.rlt.rollout.time.monotonic", lambda: clock[0])

    def send(action, *, deadline, stop_event):
        assert deadline == pytest.approx(10.2) and not stop_event.is_set()
        # Camera updates during the motor lock/SDK wait must not become next_state.
        clock[0] = 10.05
        return action

    def capture(age):
        clock[0] = 10.06
        return {"value": 1}, clock[0], 10.055 if post_command else 10.02

    step = RobotWindowStep(
        SimpleNamespace(send=send, capture=capture, action_keys=["joint.pos"]),
        lambda raw: raw,
        lambda raw, action: rewarded.append(raw) or RewardSignal(1.0),
        action_names=["joint"],
        fps=1e6,
    )
    if post_command:
        assert step(torch.zeros(1)).reward == 1.0 and len(rewarded) == 1
    else:
        with pytest.raises(TimeoutError, match="stale"):
            step(torch.zeros(1))
        assert not rewarded


def test_cancel_during_motor_lock_wait_does_not_send_or_commit_replay():
    runner, _, replay = rollout()
    sent, entered = [], Event()
    robot = SimpleNamespace(
        action_features={"joint_1.pos": float, "gripper.pos": float},
        send_action=lambda action: sent.append(dict(action)) or action,
    )
    io = RobotIO(robot)
    original_send = io.send

    def send(values, **kwargs):
        entered.set()
        return original_send(values, **kwargs)

    io.send = send
    step = RobotWindowStep(
        io,
        lambda raw: raw,
        lambda raw, command: RewardSignal(0.0),
        action_names=["joint_1", "gripper"],
        stop_event=runner.stop_event,
    )
    io._motor_lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.collect_window, step)
        try:
            assert entered.wait(2), "Worker never entered send"
            runner.stop_event.set()
        finally:
            io._motor_lock.release()
        with pytest.raises(InterruptedError):
            future.result(timeout=2)
    assert not sent and len(replay) == 0
    assert runner._state is None and not runner.collector.active
