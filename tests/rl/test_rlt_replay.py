"""Fixed-window replay semantics, ownership, versioning, and persistence."""

from dataclasses import replace

import pytest
import torch

from lerobot.policies.rlt_actor import RLTActorConfig
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE, OBS_RLT
from lerobot.rl.rlt.replay import FixedWindowCollector, FixedWindowReplayBuffer, WindowTransition
from lerobot.utils.constants import OBS_STATE


def config(**overrides):
    values = {
        "feature_model_id": "tiny-stage1-and-processors-v1",
        "z_dim": 4,
        "proprio_dim": 3,
        "action_dim": 2,
        "execution_horizon": 10,
        "reference_horizon": 12,
        "hidden_dims": (16, 16),
        "exploration_std": 0.0,
        "device": "cpu",
        "push_to_hub": False,
    }
    values.update(overrides)
    return RLTActorConfig(**values)


def observation(cfg=None, value=0.0):
    cfg = config() if cfg is None else cfg
    return {
        OBS_RLT: torch.full((cfg.z_dim,), value),
        OBS_STATE: torch.full((cfg.proprio_dim,), value),
        OBS_REFERENCE: torch.full((cfg.reference_horizon, cfg.action_dim), value + 0.5),
    }


def transition(cfg=None, *, steps=None, terminated=False, truncated=False, value=0.0):
    cfg = config() if cfg is None else cfg
    n = cfg.execution_horizon if steps is None else steps
    actions = torch.zeros(cfg.execution_horizon, cfg.action_dim)
    rewards = torch.zeros(cfg.execution_horizon)
    actions[:n] = value
    rewards[:n] = 1.0
    return WindowTransition(
        observation(cfg, value),
        actions,
        rewards,
        observation(cfg, value + 0.1),
        n,
        terminated,
        truncated,
        torch.zeros(cfg.execution_horizon, dtype=torch.bool),
        cfg.feature_model_id,
        7,
        20,
        3,
        torch.full_like(actions, value),
    )


def test_default_execution_window_is_ten_policy_steps():
    assert RLTActorConfig(feature_model_id="version", device="cpu").execution_horizon == 10


@pytest.mark.parametrize(
    "steps,terminated,truncated", [(10, False, False), (3, True, False), (2, False, True)]
)
def test_collector_records_actual_commands_and_only_executed_steps(steps, terminated, truncated):
    cfg = config()
    collector = FixedWindowCollector(cfg)
    state = observation(cfg)
    planned = torch.full((10, 2), 0.7)
    collector.begin(state, planned_action=planned, episode_id=1, start_step=0, policy_version=8)
    planned.zero_()
    state[OBS_RLT].add_(100)
    sent = torch.tensor([0.25, -0.5])
    for index in range(steps):
        collector.record(
            sent,
            index + 1,
            intervention=index == 1,
            terminated=terminated and index == steps - 1,
            truncated=truncated and index == steps - 1,
        )
    sent.add_(10)
    row = collector.finish(observation(cfg, 1.0))
    assert not collector.active
    assert row.valid_steps == steps and row.policy_version == 8
    torch.testing.assert_close(row.state[OBS_RLT], torch.zeros(4))
    torch.testing.assert_close(row.action[:steps], torch.tensor([[0.25, -0.5]]).expand(steps, -1))
    torch.testing.assert_close(row.rewards[:steps], torch.arange(1, steps + 1).float())
    assert row.action[steps:].count_nonzero() == 0
    assert row.rewards[steps:].count_nonzero() == 0
    assert row.intervention.sum() == int(steps > 1)
    torch.testing.assert_close(row.planned_action, torch.full((10, 2), 0.7))


def test_incomplete_nonterminal_window_is_rejected_and_cancel_does_not_cross_reset():
    collector = FixedWindowCollector(config())
    collector.begin(observation(), planned_action=torch.zeros(10, 2), episode_id=1, start_step=0)
    collector.record(torch.zeros(2), 1.0)
    with pytest.raises(ValueError, match="Incomplete"):
        collector.finish(observation())
    with pytest.raises(RuntimeError, match="already active"):
        collector.begin(observation(), planned_action=torch.zeros(10, 2), episode_id=2, start_step=0)
    collector.abort()
    collector.begin(observation(), planned_action=torch.ones(10, 2), episode_id=2, start_step=0)
    collector.record(torch.ones(2), 0.0, terminated=True)
    row = collector.finish(observation())
    assert row.episode_id == 2 and row.valid_steps == 1
    assert row.rewards.sum() == 0


@pytest.mark.parametrize("bad", ["feature", "partial", "tail", "nan", "mask", "shape"])
def test_replay_rejects_invalid_transition_before_mutation(bad):
    replay = FixedWindowReplayBuffer(config(), capacity=2)
    replay.add(transition())
    row = transition()
    if bad == "feature":
        row = replace(row, feature_model_id="different-weights-or-stats")
    elif bad == "partial":
        row = replace(row, valid_steps=3)
    elif bad == "tail":
        row = transition(steps=3, terminated=True)
        row.action[4] = 1
    elif bad == "nan":
        row.rewards[0] = float("nan")
    elif bad == "mask":
        row = replace(row, intervention=torch.zeros(10))
    else:
        row = replace(row, action=torch.zeros(50, 2))
    with pytest.raises(ValueError):
        replay.add(row)
    assert len(replay) == 1


def test_replay_owns_inputs_and_samples_and_preserves_actual_out_of_actor_range():
    replay = FixedWindowReplayBuffer(config(), capacity=2)
    row = transition(value=2.0)
    replay.add(row)
    row.action.zero_()
    row.state[OBS_RLT].zero_()
    batch = replay.sample(1)
    assert batch["action"].min() == 2.0  # No clipping of real controller/human commands.
    assert batch["state"][OBS_RLT].min() == 2.0
    batch["action"].zero_()
    assert replay.sample(1)["action"].min() == 2.0


def test_ring_round_trip_keeps_sampling_rng_and_feature_contract(tmp_path):
    cfg = config()
    replay = FixedWindowReplayBuffer(cfg, capacity=2, seed=19)
    for value in range(5):
        replay.add(transition(cfg, value=float(value)))
    assert len(replay) == 2
    replay.sample(7)
    path = tmp_path / "replay.pt"
    replay.save(path)
    restored = FixedWindowReplayBuffer.load(path, cfg)
    for _ in range(3):
        expected, actual = replay.sample(5), restored.sample(5)
        torch.testing.assert_close(expected["action"], actual["action"], rtol=0, atol=0)
        assert set(actual["action"].flatten().tolist()) <= {3.0, 4.0}
    with pytest.raises(ValueError, match="feature model"):
        FixedWindowReplayBuffer.load(path, config(feature_model_id="other"))


def test_empty_replay_and_invalid_masks_fail_clearly():
    replay = FixedWindowReplayBuffer(config(), capacity=1)
    with pytest.raises(RuntimeError, match="empty"):
        replay.sample(1)
    with pytest.raises(ValueError, match="positive"):
        replay.sample(0)
