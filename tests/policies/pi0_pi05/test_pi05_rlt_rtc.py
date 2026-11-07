"""Shared real PI05/RLT guided sampling, compile regions and independent action anchors."""

from dataclasses import replace

import pytest
import torch

from lerobot.async_inference.server.rtc_guidance import LightweightGuidedRTCProcessor
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.rlt_actor.configuration_rlt_actor import OBS_RTC_MASK, OBS_RTC_PREFIX
from lerobot.policies.rtc import RTCConfig
from lerobot.rl.rlt.features import FrozenRLTFeatures, checkpoint_fingerprint
from tests.policies.pi0_pi05.test_pi05_rlinf import tiny_networks as tiny_networks
from tests.policies.pi0_pi05.test_pi05_rlt import rlt_config
from tests.policies.pi0_pi05.test_pi05_rlt_stage2 import actor_config, adapter, raw_batch


@pytest.mark.parametrize("compile_regions", [False, True])
def test_shared_guided_sampler_matches_normal_sampler_and_keeps_single_prefix(
    tiny_networks, monkeypatch, compile_regions
):
    if compile_regions:
        compile = torch.compile
        monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compile(fn, backend="eager"))
    rtc = RTCConfig(enabled=True, execution_horizon=1)
    policy = PI05Policy(rlt_config(rtc_config=rtc, compile_model=compile_regions))
    policy.rtc_processor = LightweightGuidedRTCProcessor(rtc)
    policy.model.rtc_processor = policy.rtc_processor
    batch, noise = raw_batch(), torch.randn(1, 4, 4)
    prefix = torch.tensor([[[0.1, -0.2, 0.3]]])
    kwargs = {"prev_chunk_left_over": prefix, "inference_delay": 1, "execution_horizon": 1}
    expected = policy.predict_action_chunk(batch, noise=noise.clone(), **kwargs)
    expected_z = policy.extract_rlt_features(batch)
    calls = []
    original = policy.model._compiled_rlt_prefix if compile_regions else policy.model.build_prefix_cache

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(
        policy.model, "_compiled_rlt_prefix" if compile_regions else "build_prefix_cache", count
    )
    actions, z = policy.predict_action_chunk_with_rlt_features(batch, noise=noise.clone(), **kwargs)
    assert len(calls) == 1
    torch.testing.assert_close(actions, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)


def test_saved_coordinates_survive_later_observations_and_rtc_prefix_reanchors(tiny_networks):
    original = adapter()
    rtc = RTCConfig(enabled=True, execution_horizon=1)
    original.policy.config.rtc_config = rtc
    original.policy.rtc_processor = LightweightGuidedRTCProcessor(rtc)
    original.policy.model.rtc_processor = original.policy.rtc_processor
    cfg = replace(actor_config(), rtc_prefix_steps=1, input_features={})
    features = FrozenRLTFeatures(original.policy, original.preprocessor, original.postprocessor, cfg)
    features.encode(raw_batch())
    saved = features.snapshot_coordinates()
    target = torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
    absolute = saved.decode(target)
    state = features.encode(raw_batch(10), committed_prefix=absolute[2:3])
    new = features.snapshot_coordinates()
    assert state[OBS_RTC_MASK].tolist() == [1, 0]
    torch.testing.assert_close(state[OBS_RTC_PREFIX][0], new.normalize(absolute[2]))
    torch.testing.assert_close(saved.normalize(absolute[0]), target[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(new.normalize(absolute[0]), target[0])
    torch.testing.assert_close(saved.decode(target), absolute, rtol=0, atol=0)


def test_frozen_loader_configures_rtc_without_mutating_saved_feature_identity(tiny_networks, tmp_path):
    original = adapter()
    original.policy.save_pretrained(tmp_path)
    original.preprocessor.save_pretrained(tmp_path)
    original.postprocessor.save_pretrained(tmp_path)
    identity = checkpoint_fingerprint(tmp_path)
    cfg = replace(actor_config(), feature_model_id=identity, rtc_prefix_steps=1, input_features={})
    features = FrozenRLTFeatures.from_pretrained(tmp_path, cfg, device="cpu")
    assert features.policy.model.rtc_processor.rtc_config.enabled
    features.encode(raw_batch())
    assert checkpoint_fingerprint(tmp_path) == identity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rtc_coordinates_shared_sampler_and_actor_critic(tiny_networks):
    original = adapter("cuda")
    rtc = RTCConfig(enabled=True, execution_horizon=1)
    original.policy.config.rtc_config = rtc
    original.policy.rtc_processor = LightweightGuidedRTCProcessor(rtc)
    original.policy.model.rtc_processor = original.policy.rtc_processor
    cfg = replace(actor_config("cuda"), rtc_prefix_steps=1, input_features={})
    features = FrozenRLTFeatures(original.policy, original.preprocessor, original.postprocessor, cfg)
    state = features.encode(raw_batch())
    coordinates = features.snapshot_coordinates()
    absolute = coordinates.decode(state["observation.reference_chunk"])
    state = features.encode(raw_batch(1), committed_prefix=absolute[2:3])
    assert torch.isfinite(state[OBS_RTC_PREFIX]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inductor_compiled_rlt_prefix_and_guided_suffix(tiny_networks):
    original = adapter("cuda")
    cfg = replace(
        original.policy.config,
        rtc_config=RTCConfig(enabled=True, execution_horizon=1),
        compile_model=True,
        compile_mode="default",
    )
    policy = PI05Policy(cfg).to("cuda")
    policy.load_state_dict(original.policy.state_dict(), strict=True)
    policy.rtc_processor = LightweightGuidedRTCProcessor(cfg.rtc_config)
    policy.model.rtc_processor = policy.rtc_processor
    actor_cfg = replace(actor_config("cuda"), rtc_prefix_steps=1, input_features={})
    features = FrozenRLTFeatures(policy, original.preprocessor, original.postprocessor, actor_cfg)
    state = features.encode(raw_batch())
    actions = features.snapshot_coordinates().decode(state["observation.reference_chunk"])
    state = features.encode(raw_batch(1), committed_prefix=actions[2:3])
    torch.cuda.synchronize()
    assert all(torch.isfinite(value).all() for value in state.values())
