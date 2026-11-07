"""Frozen RLT representation/action parity and relative-command coordinates."""

import copy
import json
import shutil

import pytest
import torch

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_REFERENCE, OBS_RLT
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm, RLTACConfig
from lerobot.rl.rlt.features import FrozenRLTFeatures, checkpoint_fingerprint
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer
from lerobot.rl.rlt.rollout import ExecutedStep, FixedWindowRollout
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME
from tests.policies.pi0_pi05.test_pi05_peft import processors, training_batch
from tests.policies.pi0_pi05.test_pi05_rlinf import tiny_networks as tiny_networks
from tests.policies.pi0_pi05.test_pi05_rlt import rlt_config


def actor_config(device="cpu"):
    return RLTActorConfig(
        feature_model_id="tiny-saved-stage1-processors-v1",
        z_dim=16,
        proprio_dim=3,
        action_dim=3,
        execution_horizon=2,
        reference_horizon=4,
        hidden_dims=(16,),
        device=device,
        exploration_std=0.0,
        push_to_hub=False,
    )


def adapter(device="cpu"):
    cfg = rlt_config(
        device=device, use_relative_actions=True, action_feature_names=["joint_1", "joint_2", "gripper"]
    )
    policy = PI05Policy(cfg).to(device)
    pre, post = processors(cfg)
    return FrozenRLTFeatures(policy, pre, post, actor_config(device))


def raw_batch(value=0.0):
    batch = training_batch()
    batch.pop(ACTION)
    batch[OBS_STATE] = batch[OBS_STATE] + value
    return batch


def test_shared_prefix_has_same_features_and_fixed_noise_actions(tiny_networks, monkeypatch):
    policy = PI05Policy(rlt_config())
    batch = raw_batch()
    noise = torch.randn(1, 4, 4)
    expected_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    expected_z = policy.extract_rlt_features(batch)
    calls = []
    original = policy.model.build_prefix_cache

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.model, "build_prefix_cache", count)
    actions, z = policy.predict_action_chunk_with_rlt_features(batch, noise=noise.clone())
    assert len(calls) == 1
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)
    assert not z.requires_grad and not actions.requires_grad


def test_real_processor_round_trip_anchors_executed_actions_to_start_state(tiny_networks):
    features = adapter()
    state = features.encode(raw_batch())
    assert state[OBS_RLT].shape == (16,)
    assert state[OBS_REFERENCE].shape == (4, 3)
    target = torch.tensor([[-0.2, 0.1, 0.3], [0.2, -0.1, 0.5]])
    absolute = features.executable_actions(target)
    for index in range(2):
        torch.testing.assert_close(
            features.normalize_executed(absolute[index]), target[index], rtol=1e-5, atol=1e-6
        )
    # Controller-side smoothing changes what was sent. Preserve that difference in replay.
    changed = absolute[0].clone()
    changed[0] += 0.1
    normalized = features.normalize_executed(changed)
    assert normalized[0] == pytest.approx(target[0, 0].item() + 0.025, abs=1e-6)
    features.reset()
    with pytest.raises(RuntimeError, match="coordinates"):
        features.normalize_executed(absolute[0])


def test_frozen_feature_adapter_reloads_checkpoint_processors(tiny_networks, tmp_path):
    original = adapter()
    saved = tmp_path / "stage1"
    original.policy.save_pretrained(saved)
    original.preprocessor.save_pretrained(saved)
    original.postprocessor.save_pretrained(saved)
    cfg = actor_config()
    cfg.feature_model_id = checkpoint_fingerprint(saved)
    restored = FrozenRLTFeatures.from_pretrained(saved, cfg, device="cpu")
    torch.manual_seed(21)
    expected = original.encode(raw_batch())
    torch.manual_seed(21)
    actual = restored.encode(raw_batch())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert all(not p.requires_grad for p in restored.policy.parameters())


def test_checkpoint_fingerprint_catches_changed_stats_before_loading_model(tmp_path):
    source = tmp_path / "stage1"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"not loaded by metadata validation")
    (source / "stats.json").write_text('{"scale": 1}')
    first = checkpoint_fingerprint(source)
    cfg = actor_config()
    cfg.feature_model_id = first
    (source / "stats.json").write_text('{"scale": 2}')
    assert checkpoint_fingerprint(source) != first
    with pytest.raises(ValueError, match="weights/processors"):
        FrozenRLTFeatures.from_pretrained(source, cfg, device="cpu")
    with pytest.raises(ValueError, match="pin"):
        checkpoint_fingerprint("user/stage1", revision="main")


def write_tokenizer(directory, *, swapped=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0, "Task": 1, "foo": 3 if swapped else 2, "bar": 2 if swapped else 3}
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]").save_pretrained(
        directory
    )


@pytest.mark.parametrize("external_change", ["changed", "removed"])
def test_frozen_loader_keeps_bundled_tokenizer_when_external_path_changes(
    tiny_networks, tmp_path, external_change
):
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    external, saved = tmp_path / "external_tokenizer", tmp_path / "stage1"
    write_tokenizer(external)
    source = rlt_config(text_tokenizer_name=str(external))
    policy = PI05Policy(source)
    stats = {
        key: {"q01": torch.full((3,), -2.0), "q99": torch.full((3,), 6.0)} for key in (OBS_STATE, ACTION)
    }
    pre, post = make_pi05_pre_post_processors(source, stats)
    policy.save_pretrained(saved)
    pre.save_pretrained(saved)
    post.save_pretrained(saved)
    cfg = actor_config()
    cfg.feature_model_id = checkpoint_fingerprint(saved)
    first = FrozenRLTFeatures.from_pretrained(saved, cfg, device="cpu")

    def raw():
        return {**raw_batch(), "task": "foo"}

    expected_tokens = first.preprocessor(raw())[OBS_LANGUAGE_TOKENS]
    torch.manual_seed(25)
    expected = first.encode(raw())
    if external_change == "changed":
        write_tokenizer(external, swapped=True)
    else:
        shutil.rmtree(external)
    # External assets are no longer part of the effective frozen feature model.
    assert checkpoint_fingerprint(saved) == cfg.feature_model_id
    second = FrozenRLTFeatures.from_pretrained(saved, cfg, device="cpu")
    torch.testing.assert_close(second.preprocessor(raw())[OBS_LANGUAGE_TOKENS], expected_tokens)
    torch.manual_seed(25)
    actual = second.encode(raw())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def fingerprint_checkpoint(tmp_path, *, artifact="tokenizer"):
    saved = tmp_path / "stage1"
    saved.mkdir()
    (saved / "config.json").write_text("{}")
    (saved / "model.safetensors").write_bytes(b"only metadata is read")
    step = {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "/external/tokenizer"}}
    if artifact is not None:
        step["artifacts"] = {"tokenizer_name": artifact}
    (saved / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").write_text(json.dumps({"steps": [step]}))
    return saved


def test_fingerprint_covers_all_bundled_tokenizer_files(tmp_path):
    saved = fingerprint_checkpoint(tmp_path)
    (saved / "tokenizer").mkdir()
    asset = saved / "tokenizer" / "vocabulary.binary"
    asset.write_bytes(b"version 1")
    first = checkpoint_fingerprint(saved)
    asset.write_bytes(b"version 2")
    assert checkpoint_fingerprint(saved) != first
    cfg = actor_config()
    cfg.feature_model_id = first
    with pytest.raises(ValueError, match="weights/processors"):
        FrozenRLTFeatures.from_pretrained(saved, cfg, device="cpu")


@pytest.mark.parametrize("artifact", [None, "../external", "/external/tokenizer"])
def test_frozen_features_reject_unbundled_or_escaping_tokenizer_before_loading(tmp_path, artifact):
    saved = fingerprint_checkpoint(tmp_path, artifact=artifact)
    with pytest.raises(ValueError, match="bundled tokenizer|inside the Stage 1"):
        checkpoint_fingerprint(saved)


@pytest.mark.parametrize("nested", [False, True])
def test_tokenizer_directory_symlinks_cannot_escape_fingerprinting(tmp_path, nested):
    saved = fingerprint_checkpoint(tmp_path, artifact="assets/tokenizer" if nested else "tokenizer")
    external = tmp_path / "external"
    external.mkdir()
    if nested:
        (external / "tokenizer").mkdir()
    (saved / ("assets" if nested else "tokenizer")).symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinking"):
        checkpoint_fingerprint(saved)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_stage1_to_window_replay_to_actor_critic_keeps_vla_frozen(tiny_networks, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    features = adapter(device)
    before = copy.deepcopy(features.policy.state_dict())
    actor = RLTActorPolicy(actor_config(device))
    algorithm = RLTACAlgorithm(actor, RLTACConfig(policy_config=actor.config))
    replay = FixedWindowReplayBuffer(actor.config, capacity=4)
    rollout = FixedWindowRollout(features, replay, actor=actor)
    rollout.begin_episode(raw_batch(), episode_id=0)
    counter = [0]

    def execute(command):
        counter[0] += 1
        return ExecutedStep(raw_batch(counter[0] * 0.01), command * 0.9, 0.25)

    rollout.collect_window(execute, use_actor=False)
    rollout.collect_window(execute, use_actor=True)
    stats = algorithm.update(replay.get_iterator(2))
    assert "actor_loss" in stats.losses and len(replay) == 2
    assert counter == [4]
    for key, value in features.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in features.policy.parameters())
