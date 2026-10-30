# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Configuration tests for the consolidated DAgger server entry point."""

from lerobot.async_inference.server.policy_server_dagger import (
    PolicyServerDagger,
    PolicyServerDaggerConfig,
)
from lerobot.async_inference.server.policy_server_deployment import PolicyServerDeployment
from lerobot.configs import RTCAttentionSchedule


def test_dagger_server_disables_cross_connection_model_reuse_by_default():
    config = PolicyServerDaggerConfig(host="localhost", port=9999)
    assert not config.reuse_last_policy
    assert config.rtc.prefix_attention_schedule is RTCAttentionSchedule.EXP
    assert config.rtc.max_guidance_weight == 5.0
    assert issubclass(PolicyServerDagger, PolicyServerDeployment)


def test_dagger_server_can_explicitly_reenable_model_reuse():
    config = PolicyServerDaggerConfig(host="localhost", port=9999, reuse_last_policy=True)
    assert config.reuse_last_policy
