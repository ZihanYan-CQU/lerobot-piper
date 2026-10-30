# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Regression checks for native PI05 sampling and fused vision LoRA contracts."""

import json
from dataclasses import replace
from functools import partial

import pytest
import torch
from torch import nn
from torch._dynamo.testing import CompileCounter

pytest.importorskip("transformers")

from lerobot.async_inference.server.rtc_guidance import LightweightGuidedRTCProcessor  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402
from lerobot.policies.rtc import RTCConfig, RTCProcessor  # noqa: E402
from tests.policies.pi0_pi05.test_pi05_rlinf import (  # noqa: E402
    config,
    inputs,
    tiny_networks as tiny_networks,
)


@pytest.mark.parametrize("entry", ["wrap", "add", "reload"])
@pytest.mark.parametrize("setting,value", [("lora_dropout", 0.1), ("lora_bias", True)])
def test_fused_vision_lora_rejects_unsupported_settings_before_injection(tmp_path, entry, setting, value):
    peft = pytest.importorskip("peft")
    from lerobot.policies.pi05.rlinf.peft import prepare_lora_config

    def container():
        model = nn.Module()
        model.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        return model

    def lora_config(**kwargs):
        return prepare_lora_config(peft.LoraConfig(target_modules=["attn"], r=2, lora_alpha=4, **kwargs))

    model = container()
    if entry == "wrap":
        operation = partial(peft.get_peft_model, model, lora_config(**{setting: value}))
    else:
        wrapped = peft.get_peft_model(model, lora_config())
        if entry == "add":
            operation = partial(wrapped.add_adapter, "second", lora_config(**{setting: value}))
        else:
            wrapped.save_pretrained(tmp_path)
            filename = tmp_path / "adapter_config.json"
            saved_config = json.loads(filename.read_text())
            saved_config[setting] = value
            filename.write_text(json.dumps(saved_config))
            model = container()
            loaded_config = prepare_lora_config(peft.PeftConfig.from_pretrained(tmp_path))
            operation = partial(peft.PeftModel.from_pretrained, model, tmp_path, config=loaded_config)
    expected = "lora_dropout=0" if setting == "lora_dropout" else "lora_bias=false"
    with pytest.raises(ValueError, match=expected):
        operation()
    if entry == "add":
        assert "second" not in model.attn.lora_A
        assert "second" not in model.attn.base_layer.out_proj.lora_A
    else:
        assert isinstance(model.attn.out_proj, nn.Linear)


@pytest.mark.parametrize("profile", ["lerobot", "openpi", "checkpoint"])
@pytest.mark.parametrize("compile_model", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_profile_applies_matmul_precision(tiny_networks, monkeypatch, profile, compile_model, dtype):
    original = torch.get_float32_matmul_precision()
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    try:
        torch.set_float32_matmul_precision("highest")
        PI05Policy(replace(config(), precision_profile=profile, compile_model=compile_model, dtype=dtype))
        expected = "high" if compile_model or profile == "openpi" else "highest"
        assert torch.get_float32_matmul_precision() == expected
    finally:
        torch.set_float32_matmul_precision(original)


@pytest.mark.parametrize(
    "dtype,kind,matmul_precision",
    [
        (dtype, kind, "high")
        for dtype in ("float32", "bfloat16")
        for kind in ("fast", "reference", "native", "trained")
    ]
    + [("float32", "native", "highest")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_compiled_rtc_preserves_outputs_cache_and_graph_reuse(
    tiny_networks, monkeypatch, request, kind, dtype, matmul_precision, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(17)
    mode = "trained" if kind == "trained" else "guided"
    rtc_config = RTCConfig(mode=mode, execution_horizon=2)
    cfg = replace(config(), dtype=dtype, device=device, rtc_config=rtc_config, rtc_training_max_delay=2)
    reference = PI05Policy(cfg).model.eval()
    # Nonzero adaptive gates and image head ensure changed observations affect
    # the denoiser instead of being hidden by the native zero initialization.
    with torch.no_grad():
        reference.img.head.weight.normal_(std=0.03)
        for module in reference.llm.modules():
            if getattr(module, "adaptive", False):
                module.ada_modulation.weight.normal_(std=0.03)

    torch._dynamo.reset()
    counter = CompileCounter()
    compile_fn = torch.compile

    def compile_sampler(function, **kwargs):
        if device == "cuda":
            return compile_fn(function, backend="inductor", **kwargs)
        return compile_fn(function, backend=counter)

    monkeypatch.setattr(torch, "compile", compile_sampler)
    original_precision = torch.get_float32_matmul_precision()
    try:
        compiled = PI05Policy(replace(cfg, compile_model=True)).model.eval()
        compiled.load_state_dict(reference.state_dict())
        torch.set_float32_matmul_precision(matmul_precision)
        if kind in {"fast", "reference"}:
            for model in (reference, compiled):
                model.rtc_processor = LightweightGuidedRTCProcessor(rtc_config, fast=kind == "fast")
        elif kind == "native":
            for model in (reference, compiled):
                model.rtc_processor = RTCProcessor(rtc_config)
        images, image_masks, tokens, masks = inputs()
        image_masks = [mask.to(device) for mask in image_masks]
        masks = masks.to(device)
        noise = torch.randn(1, 4, 4, device=device)
        graph_counts = []
        errors = []
        previous_output = None
        for index, (length, delay, horizon) in enumerate(
            [(2, 0, 2), (2, 1, 2), (3, 1, 3), (3, 2, 2), (2, 0, 2)]
        ):
            image = images[0].roll(index, dims=-1).to(device)
            prompt = tokens.to(device).clone()
            prompt[:, 0] = index + 1
            prefix = torch.full((1, length, 4), index * 0.1, device=device)
            kwargs = {
                "noise": noise + index * 0.05,
                "prev_chunk_left_over": prefix,
                "inference_delay": delay,
                "execution_horizon": horizon,
            }
            arguments = ([image], image_masks, prompt, masks)
            expected = reference.sample_actions(*arguments, **kwargs)
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            actual = compiled.sample_actions(*arguments, **kwargs)
            # CUDA's high-precision mode permits TF32 matmuls; eager and
            # Inductor may choose different kernels at TF32 mantissa accuracy.
            # Keep a strict, same-weight highest-precision control alongside it.
            tolerance = (
                0.02
                if dtype == "bfloat16"
                else (1e-3 if device == "cuda" and matmul_precision == "high" else 2e-4)
            )
            errors.append((actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert torch.isfinite(actual).all()
            if kind == "trained" and delay:
                torch.testing.assert_close(actual[:, :delay], prefix[:, :delay], rtol=0, atol=0)
            if previous_output is not None:
                assert not torch.equal(previous_output, actual)
            previous_output = actual.detach().clone()
            graph_counts.append(counter.frame_count)
        if device == "cpu" and kind == "fast":
            assert graph_counts == [1, 1, 1, 1, 1]
        elif device == "cpu" and kind in {"reference", "native"}:
            assert graph_counts == [graph_counts[0]] * len(graph_counts)
        request.node.user_properties.append(("max_action_error", max(errors)))
        # A suffix pass must not append data to the cached prefix or overwrite it.
        with torch.no_grad():
            _, padding, cache = compiled.build_prefix_cache(*arguments)
            before = [(key.clone(), value.clone()) for key, value in cache]
            compiled.denoise_step(padding, cache, noise, torch.tensor([0.5], device=device))
            for (key, value), (old_key, old_value) in zip(cache, before, strict=True):
                torch.testing.assert_close(key, old_key, rtol=0, atol=0)
                torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(original_precision)
