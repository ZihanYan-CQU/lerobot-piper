"""Chunked Bellman/BC objectives and real optimizer/checkpoint integration."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE
from lerobot.rl.algorithms.factory import make_algorithm, make_algorithm_config
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm, RLTACConfig
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.rlt.checkpoint import load_checkpoint, save_checkpoint
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.rl.trainer import RLTrainer
from tests.rl.test_rlt_replay import config, observation, transition


def learner(cfg=None, **kwargs):
    cfg = config() if cfg is None else cfg
    return RLTACAlgorithm(RLTActorPolicy(cfg), RLTACConfig(policy_config=cfg, **kwargs))


def batch(algorithm, **row_kwargs):
    replay = FixedWindowReplayBuffer(algorithm.policy.config, capacity=4)
    replay.add(transition(algorithm.policy.config, **row_kwargs))
    return replay.sample(2)


@pytest.mark.parametrize(
    "steps,terminated,truncated", [(10, False, False), (3, True, False), (3, False, True), (2, True, True)]
)
def test_bellman_discount_counts_executed_policy_steps(steps, terminated, truncated):
    algorithm = learner(discount=0.9)
    with torch.no_grad():
        for parameter in algorithm.critic_target.parameters():
            parameter.zero_()
        for head in algorithm.critic_target.heads:
            head[-1].bias.fill_(2.0)
    prepared = algorithm._prepare_batch(
        batch(algorithm, steps=steps, terminated=terminated, truncated=truncated)
    )
    target = algorithm.target_values(prepared)
    expected = sum(0.9**i for i in range(steps)) + (0 if terminated else 0.9**steps * 2)
    torch.testing.assert_close(target, torch.full((2,), expected))
    assert not target.requires_grad


def test_bc_uses_human_commands_only_on_intervened_valid_steps():
    algorithm = learner(q_weight=0, bc_weight=5, reference_dropout_prob=0)
    with torch.no_grad():
        for parameter in algorithm.policy.parameters():
            parameter.zero_()
    data = batch(algorithm, steps=3, terminated=True, value=0.0)
    data["action"][:, :3] = 0.25
    data["complementary_info"]["intervention"][:, 0] = True
    data["state"][OBS_REFERENCE][:, 3:] = 100  # Unexecuted tail cannot enter the BC objective.
    prepared = algorithm._prepare_batch(data)
    loss, metrics = algorithm.actor_loss(prepared)
    assert metrics["bc_loss"] == pytest.approx((0.25**2 + 0.5**2 * 2) / 3)
    assert loss.item() == pytest.approx(5 * metrics["bc_loss"])
    assert metrics["human_ratio"] == pytest.approx(1 / 3)


def test_reference_dropout_does_not_change_frozen_state_or_bc_targets():
    algorithm = learner()
    data = batch(algorithm)
    state = data["state"]
    altered = {key: value.clone() for key, value in state.items()}
    altered[OBS_REFERENCE].add_(10)
    first = algorithm.policy(state, deterministic=True, reference_dropout_prob=1)
    second = algorithm.policy(altered, deterministic=True, reference_dropout_prob=1)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.all(state[OBS_REFERENCE] == 0.5)
    assert not torch.equal(
        algorithm.policy(state, deterministic=True), algorithm.policy(altered, deterministic=True)
    )


@pytest.mark.parametrize("bad", ["feature", "mask", "partial", "tail", "reward", "done"])
def test_invalid_replay_is_rejected_before_any_parameter_update(bad):
    algorithm = learner()
    data = batch(algorithm)
    if bad == "feature":
        data["complementary_info"]["feature_id"].zero_()
    elif bad == "mask":
        data["complementary_info"]["valid_mask"][:, 1] = False
    elif bad == "partial":
        data["complementary_info"]["valid_mask"][:, 3:] = False
        data["reward"][:, 3:] = 0
    elif bad == "tail":
        data["complementary_info"]["valid_mask"][:, 3:] = False
        data["done"].fill_(True)
    elif bad == "reward":
        data["reward"][0, 0] = float("inf")
    else:
        data["done"] = data["done"].float()
    original = {key: value.clone() for key, value in algorithm.state_dict().items()}
    with pytest.raises(ValueError):
        algorithm.update(iter([data]))
    assert algorithm.optimization_step == 0
    for key, value in algorithm.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_registered_actor_and_algorithm_round_trip(tmp_path):
    cfg = config()
    policy = RLTActorPolicy(cfg)
    assert get_policy_class("rlt_actor") is RLTActorPolicy
    algorithm = make_algorithm(make_algorithm_config("rlt_ac", policy_config=cfg), policy)
    assert isinstance(algorithm, RLTACAlgorithm)
    path = tmp_path / "actor"
    policy.save_pretrained(path)
    loaded_cfg = PreTrainedConfig.from_pretrained(path)
    assert isinstance(loaded_cfg, RLTActorConfig)
    restored = RLTActorPolicy.from_pretrained(path)
    inputs = {key: value[None] for key, value in observation(cfg).items()}
    torch.testing.assert_close(policy.predict_action_chunk(inputs), restored.predict_action_chunk(inputs))
    pre, post = make_pre_post_processors(cfg)
    torch.testing.assert_close(
        post(policy.predict_action_chunk(pre(inputs))), policy.predict_action_chunk(inputs)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_trainer_update_and_exact_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(41)
    cfg = config(device=device, exploration_std=0.002)
    algorithm = learner(cfg)
    replay = FixedWindowReplayBuffer(cfg, capacity=8, seed=17)
    for value in (0.0, 0.1, 0.2):
        replay.add(transition(cfg, value=value))
    trainer = RLTrainer(algorithm, OnlineOfflineMixer(replay), batch_size=2)
    before_actor = copy.deepcopy(algorithm.policy.state_dict())
    before_target = copy.deepcopy(algorithm.critic_target.state_dict())
    stats = trainer.training_step()
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.to_log_dict().values())
    assert any(
        not torch.equal(value, before_actor[key]) for key, value in algorithm.policy.state_dict().items()
    )
    assert any(
        not torch.equal(value, before_target[key])
        for key, value in algorithm.critic_target.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in algorithm.critic_target.parameters())
    destination = save_checkpoint(tmp_path / "saved", algorithm, replay)
    expected_stats = trainer.training_step()
    expected_actor = copy.deepcopy(algorithm.policy.state_dict())
    expected_critic = copy.deepcopy(algorithm.state_dict())
    restored, restored_replay = load_checkpoint(destination, device=device)
    optimizer_ids = {key: id(value) for key, value in restored.get_optimizers().items()}
    resumed_trainer = RLTrainer(restored, OnlineOfflineMixer(restored_replay), batch_size=2)
    assert {key: id(value) for key, value in restored.get_optimizers().items()} == optimizer_ids
    actual_stats = resumed_trainer.training_step()
    assert restored.optimization_step == algorithm.optimization_step == 2
    assert actual_stats.to_log_dict() == expected_stats.to_log_dict()
    for key, value in restored.policy.state_dict().items():
        torch.testing.assert_close(value, expected_actor[key], rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_critic[key], rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["missing", "unexpected"])
def test_checkpoint_rejects_incomplete_or_foreign_actor_weights(tmp_path, problem):
    algorithm = learner()
    replay = FixedWindowReplayBuffer(algorithm.policy.config, capacity=4)
    replay.add(transition(algorithm.policy.config))
    algorithm.update(replay.get_iterator(2))
    saved = save_checkpoint(tmp_path / "saved", algorithm, replay)
    path = saved / "actor" / "model.safetensors"
    weights = load_file(path)
    if problem == "missing":
        weights.pop(next(iter(weights)))
    else:
        weights["unrecognized_head.weight"] = torch.ones(1)
    save_file(weights, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(saved)


def test_critic_ignores_reference_but_values_the_complete_fixed_action_plan():
    algorithm = learner()
    prepared = algorithm._prepare_batch(batch(algorithm))
    q = algorithm.critic(prepared["state"], prepared["critic_action"])
    prepared["state"][OBS_REFERENCE].fill_(100)
    torch.testing.assert_close(
        q, algorithm.critic(prepared["state"], prepared["critic_action"]), rtol=0, atol=0
    )
    different = prepared["critic_action"].clone()
    different[:, 3:] += 0.4
    assert not torch.equal(q, algorithm.critic(prepared["state"], different))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_snapshot_load_preserves_device_parameters_and_clears_queued_actions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source, destination = learner(config(device=device)), learner(config(device=device))
    inputs = {key: value[None] for key, value in observation().items()}
    destination.policy.select_action(inputs)
    parameters = list(destination.policy.parameters())
    source.optimization_step = 4
    destination.load_weights(source.get_weights())
    assert destination.policy.rollout_version == 4 and not destination.policy._action_queue
    assert all(
        before is after for before, after in zip(parameters, destination.policy.parameters(), strict=True)
    )
    torch.testing.assert_close(
        source.policy.predict_action_chunk(inputs),
        destination.policy.predict_action_chunk(inputs),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("bad", ["feature", "stale", "version", "missing", "shape", "nan"])
def test_invalid_actor_snapshot_is_rejected_without_partial_weight_replacement(bad):
    algorithm = learner()
    algorithm.optimization_step = 4
    algorithm.load_weights(algorithm.get_weights())
    before = copy.deepcopy(algorithm.policy.state_dict())
    payload = algorithm.get_weights()
    payload["version"] = 5
    for value in payload["policy"].values():
        value.add_(1)
    last = next(reversed(payload["policy"]))
    if bad == "feature":
        payload["feature_model_id"] = "other-stage1"
    elif bad == "stale":
        payload["version"] = 3
    elif bad == "version":
        payload["version"] = True
    elif bad == "missing":
        del payload["policy"][last]
    elif bad == "shape":
        payload["policy"][last] = torch.ones(1)
    else:
        payload["policy"][last].fill_(float("nan"))
    with pytest.raises(ValueError):
        algorithm.load_weights(payload)
    assert algorithm.policy.rollout_version == 4
    for key, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_outer_amp_keeps_small_rl_heads_fp32():
    algorithm = learner(config(device="cuda", exploration_std=0.002))
    data = batch(algorithm)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        stats = algorithm.update(iter([data]))
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.to_log_dict().values())
    assert all(parameter.dtype == torch.float32 for parameter in algorithm.policy.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in algorithm.critic.parameters())
