# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Numerical and work-avoidance regressions for deployment speed improvements."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.server.rtc_guidance import LightweightGuidedRTCProcessor
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig
from lerobot.scripts.lerobot_profile_pi05 import configure_guided_rtc


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("squeezed", [False, True])
def test_direct_guidance_matches_reference_for_all_masks_shapes_times(device, schedule, dtype, squeezed):
    config = RTCConfig(prefix_attention_schedule=schedule, execution_horizon=4)
    reference = LightweightGuidedRTCProcessor(config, fast=False)
    direct = LightweightGuidedRTCProcessor(config)
    generator = torch.Generator(device).manual_seed(17)
    x = torch.randn((6, 3) if squeezed else (2, 6, 3), dtype=dtype, device=device, generator=generator)
    before = x.clone()
    for length, delay in [(2, 0), (4, 2), (1, 1), (4, 4), (6, 2)]:
        previous = torch.randn(length, 2, dtype=dtype, device=device, generator=generator)
        for time in [1.0, 0.7, 0.1, 0.0, torch.tensor(0.5, device=device)]:
            args = (x, previous, delay, time, lambda value: 0.37 * value.square())
            torch.testing.assert_close(
                direct.denoise_step(*args), reference.denoise_step(*args), rtol=0, atol=0
            )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_direct_guidance_skips_autograd_and_debug_tracking(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unused autograd or debug tracking ran")

    processor = LightweightGuidedRTCProcessor(RTCConfig())
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(processor, "track", forbidden)
    result = processor.denoise_step(torch.ones(1, 6, 3), torch.zeros(2, 3), 1, 0.5, lambda x: x * 0.5)
    assert torch.isfinite(result).all()
    assert not result.requires_grad


def test_weight_reuse_does_not_reuse_actions_and_respects_changed_config(monkeypatch):
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP, execution_horizon=4)
    processor = LightweightGuidedRTCProcessor(config)
    reference = LightweightGuidedRTCProcessor(config, fast=False)
    builds = []
    build = processor.get_prefix_weights

    def record(*args):
        builds.append(args)
        return build(*args)

    monkeypatch.setattr(processor, "get_prefix_weights", record)
    for step in range(3):
        args = (torch.ones(1, 6, 3), torch.full((4, 3), step / 10), 2, 0.5, lambda x: x * 0.5)
        torch.testing.assert_close(
            processor.denoise_step(*args), reference.denoise_step(*args), rtol=0, atol=0
        )
    assert len(builds) == 1
    config.prefix_attention_schedule = RTCAttentionSchedule.LINEAR
    config.max_guidance_weight = 1.0
    torch.testing.assert_close(processor.denoise_step(*args), reference.denoise_step(*args), rtol=0, atol=0)
    assert len(builds) == 2


def test_direct_guidance_preserves_debug_tensor_values():
    config = RTCConfig(debug=True)
    reference = LightweightGuidedRTCProcessor(config, fast=False)
    direct = LightweightGuidedRTCProcessor(config)
    args = (torch.ones(6, 3), torch.zeros(2, 3), 1, 0.5, lambda x: x * 0.5)
    reference.denoise_step(*args)
    direct.denoise_step(*args)
    expected, actual = reference.get_all_debug_steps()[0], direct.get_all_debug_steps()[0]
    for name in ("x1_t", "correction", "err", "weights", "guidance_weight"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_prepared_guidance_preserves_native_rounding_for_changing_prefixes(device, schedule, dtype):
    config = RTCConfig(prefix_attention_schedule=schedule, execution_horizon=4)
    processor = LightweightGuidedRTCProcessor(config)
    reference = LightweightGuidedRTCProcessor(config, fast=False)
    generator = torch.Generator(device).manual_seed(23)
    x = torch.randn(2, 6, 3, dtype=dtype, device=device, generator=generator)

    def denoise(value):
        return 0.37 * value.square()

    for length, delay, horizon in [(1, 0, None), (4, 2, 3), (2, 2, 4), (6, 3, 1)]:
        previous = torch.randn(2, length, 2, dtype=dtype, device=device, generator=generator)
        padded, prepared = processor.prepare_for_compilation(
            previous, delay, horizon, shape=x.shape, device=x.device, num_steps=10
        )
        for step in range(10):
            time = 1.0 + step * (-1.0 / 10)
            torch.testing.assert_close(
                prepared.denoise_step(x, padded, 0, time, denoise),
                reference.denoise_step(x, previous, delay, time, denoise, horizon),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("fast,debug", [(False, False), (True, True)])
def test_reference_and_debug_guidance_keep_the_unprepared_path(fast, debug):
    processor = LightweightGuidedRTCProcessor(RTCConfig(debug=debug), fast=fast)
    assert (
        processor.prepare_for_compilation(
            torch.zeros(2, 3), 1, None, shape=(1, 6, 3), device=torch.device("cpu"), num_steps=10
        )
        is None
    )


@pytest.mark.parametrize("fast", [True, False])
def test_profiler_uses_deployment_guidance(fast):
    policy = SimpleNamespace(config=SimpleNamespace(rtc_config=RTCConfig()), model=SimpleNamespace())
    configure_guided_rtc(policy, fast=fast)
    assert policy.rtc_processor is policy.model.rtc_processor
    assert type(policy.rtc_processor) is LightweightGuidedRTCProcessor
    assert policy.rtc_processor.fast is fast
