# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import lerobot.async_inference.server.policy_server_deployment as server_module
from lerobot.async_inference.server.pi05_inference import PI05InferenceConfig, warmup_pi05
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import RemotePolicyConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def _config():
    return PI05Config(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        image_resolution=(16, 16),
        tokenizer_max_length=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )


@pytest.mark.parametrize("profile", ["checkpoint", "lerobot", "openpi"])
@pytest.mark.parametrize(
    "saved_dtype,saved_profile,saved_compile,saved_mode",
    [
        ("float32", "lerobot", False, "default"),
        ("bfloat16", "openpi", True, "reduce-overhead"),
    ],
)
def test_device_and_profile_are_applied_before_model_loading_and_affect_cache(
    tmp_path, monkeypatch, profile, saved_dtype, saved_profile, saved_compile, saved_mode
):
    config = _config()
    config.dtype = saved_dtype
    config.precision_profile = saved_profile
    config.compile_model = saved_compile
    config.compile_mode = saved_mode
    config.device = "cuda:7"  # Deployment must override the training machine's device before construction.
    config.save_pretrained(tmp_path)
    loaded = []

    class Policy:
        config_class = PI05Config

        @classmethod
        def from_pretrained(cls, path, *, config=None):
            assert config is not None
            assert config.device == "cpu"
            loaded.append(config)
            obj = cls()
            obj.config = config
            return obj

        def to(self, device):
            return self

    monkeypatch.setattr(server_module, "get_policy_class", lambda _: Policy)
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(pi05_inference=PI05InferenceConfig(profile=profile, warmup_runs=0))
    )

    def processors(_):
        server.preprocessor = server.postprocessor = object()

    monkeypatch.setattr(server, "_build_processors", processors)
    specs = RemotePolicyConfig(
        policy_type="pi05",
        pretrained_name_or_path=str(tmp_path),
        lerobot_features={},
        actions_per_chunk=4,
        device="cpu",
    )
    try:
        assert not server._load_or_reuse_policy(specs)
        assert loaded[-1].dtype == (saved_dtype if profile == "checkpoint" else "bfloat16")
        assert loaded[-1].precision_profile == profile
        expected_compile = saved_compile if profile == "checkpoint" else profile == "openpi"
        assert loaded[-1].compile_model is expected_compile
        assert loaded[-1].compile_mode == (saved_mode if profile == "checkpoint" else "max-autotune")
        assert server._load_or_reuse_policy(specs)
        assert len(loaded) == 1
        server.config.pi05_inference.compile_model = not expected_compile
        assert not server._load_or_reuse_policy(specs)
        assert len(loaded) == 2 and loaded[-1].compile_model is not expected_compile
        # Switching precision must invalidate the cache even when compilation
        # is explicitly fixed to the same value for both profiles.
        server.config.pi05_inference.profile = "openpi" if profile == "lerobot" else "lerobot"
        assert not server._load_or_reuse_policy(specs)
        assert len(loaded) == 3
        assert loaded[-1].precision_profile == server.config.pi05_inference.profile
        assert loaded[-1].compile_model is not expected_compile
        original = PI05Config.from_pretrained(tmp_path)
        assert original.precision_profile == saved_profile and original.dtype == saved_dtype
        assert original.compile_model is saved_compile and original.compile_mode == saved_mode
    finally:
        server.stop()


@pytest.mark.parametrize("kwargs", [{"profile": "invalid"}, {"compile_mode": "invalid"}, {"warmup_runs": -1}])
def test_invalid_deployment_options_fail(kwargs):
    with pytest.raises(ValueError):
        PI05InferenceConfig(**kwargs)


class _WarmPolicy(torch.nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = _config()
        self.reset_count = 0
        self.calls = []
        self.fail = fail
        self.rtc_processor = SimpleNamespace(reset_tracker=lambda: self.calls.append("reset_tracker"))

    def reset(self):
        self.reset_count += 1

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, inference_delay, prev_chunk_left_over):
        assert not torch.is_inference_mode_enabled()
        assert batch["observation.images.top"].shape == (1, 3, 16, 16)
        prefix_shape = None if prev_chunk_left_over is None else tuple(prev_chunk_left_over.shape)
        self.calls.append((inference_delay, prefix_shape))
        actions = torch.rand(1, 4, 7)
        return actions.fill_(float("nan")) if self.fail else actions


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_preserves_rng_clears_episode_state_and_exercises_prefix(fail):
    policy = _WarmPolicy(fail=fail)
    rng = torch.get_rng_state().clone()
    kwargs = {"runs": 2, "prefix_length": 2, "inference_delay": 1, "logger": logging.getLogger(__name__)}
    if fail:
        with pytest.raises(RuntimeError, match="non-finite"):
            warmup_pi05(policy, **kwargs)
    else:
        warmup_pi05(policy, **kwargs)
        assert policy.calls[:-1] == [(0, None), (0, None), (1, (2, 7)), (1, (2, 7))]
    assert torch.equal(torch.get_rng_state(), rng)
    assert policy.reset_count == 2
    assert policy.calls[-1] == "reset_tracker"


def test_checkpoint_profile_only_overrides_parameter_layout():
    assert PI05InferenceConfig().overrides() == ("--precision_profile=checkpoint",)


@pytest.mark.parametrize(
    "options", [{}, {"profile": "lerobot"}, {"compile_model": False}, {"compile_mode": "default"}]
)
def test_default_checkpoint_profile_does_not_override_other_policy_types(options):
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(pi05_inference=PI05InferenceConfig(**options))
    )
    specs = RemotePolicyConfig(
        policy_type="smolvla",
        pretrained_name_or_path="non-pi05-checkpoint",
        lerobot_features={},
        actions_per_chunk=4,
        device="cpu",
    )
    try:
        if options:
            with pytest.raises(ValueError, match="require policy_type=pi05"):
                server._model_cache_key(specs)
        else:
            assert server._model_cache_key(specs).pi05_overrides == ()
    finally:
        server.stop()


def test_warmup_logs_first_failure_before_rng_restore_raises(monkeypatch, caplog):
    policy = _WarmPolicy()

    def predict_failure(*args, **kwargs):
        raise RuntimeError("original prediction failure")

    @contextmanager
    def failed_rng_restore(**kwargs):
        try:
            yield
        finally:
            raise RuntimeError("secondary RNG restoration failure")

    monkeypatch.setattr(policy, "predict_action_chunk", predict_failure)
    monkeypatch.setattr(torch.random, "fork_rng", failed_rng_restore)
    with pytest.raises(RuntimeError, match="secondary RNG restoration failure"):
        warmup_pi05(policy, runs=1, prefix_length=0, inference_delay=0, logger=logging.getLogger(__name__))
    assert "ordinary 1/1 failed at predict_action_chunk" in caplog.text
    assert "original prediction failure" in caplog.text
    assert policy.reset_count == 2


def test_full_pipeline_warmup_does_not_build_an_unused_synthetic_batch():
    policy = _WarmPolicy()
    # The callback owns preparation. Its valid batch need not use the fallback
    # generator's state schema (or allocate an extra image/token set on the GPU).
    policy.config.input_features.pop("observation.state")
    batch = {"observation.images.top": torch.ones(1, 3, 16, 16)}
    warmup_pi05(
        policy,
        runs=1,
        prefix_length=0,
        inference_delay=0,
        logger=logging.getLogger(__name__),
        prepare_batch=lambda: batch,
    )
    assert policy.calls[0] == (0, None)
