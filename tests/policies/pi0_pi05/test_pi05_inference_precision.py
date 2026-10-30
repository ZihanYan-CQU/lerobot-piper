# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers import GemmaConfig, PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPooling  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel  # noqa: E402
from lerobot.policies.rtc import RTCConfig  # noqa: E402
from lerobot.utils.import_utils import _grpc_available  # noqa: E402

if _grpc_available:
    from lerobot.async_inference.server.pi05_inference import warmup_pi05
    from lerobot.async_inference.server.policy_server_deployment import (
        _LightweightGuidedRTCProcessor,
    )


class _TinyVisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = SiglipVisionModel(
            SiglipVisionConfig(
                image_size=16,
                patch_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                vision_use_head=False,
            )
        )
        self.multi_modal_projector = nn.Linear(16, 32)

    def get_image_features(self, image):
        hidden = self.vision_tower(image).last_hidden_state
        return BaseModelOutputWithPooling(pooler_output=self.multi_modal_projector(hidden))


def _backbone(profile, dtype="bfloat16"):
    # Exercise production precision/vision methods with real, small HF SigLIP
    # layers, without allocating the 3B-parameter language model.
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.precision_profile = profile
    model.freeze_vision_encoder = False
    model.train_expert_only = False
    model.paligemma = nn.Module()
    model.paligemma.model = _TinyVisualModel()
    model.gemma_expert = nn.Module()
    model.gemma_expert.input_layernorm = nn.Linear(16, 48)
    model.gemma_expert.projection = nn.Linear(16, 16)
    model.to_bfloat16_for_selected_params(dtype)
    model.eval()
    return model


@pytest.mark.parametrize("profile", ["lerobot", "openpi"])
def test_profiles_preserve_stable_parameter_types_and_fp32_norms(profile):
    model = _backbone(profile)
    visual = model.paligemma.model
    vision = visual.vision_tower.vision_model
    expected = torch.float32 if profile == "lerobot" else torch.bfloat16
    assert vision.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vision.embeddings.position_embedding.weight.dtype == torch.float32
    assert vision.encoder.layers[0].self_attn.q_proj.weight.dtype == expected
    assert visual.multi_modal_projector.weight.dtype == expected
    assert model.gemma_expert.input_layernorm.weight.dtype == torch.float32
    assert model.gemma_expert.projection.weight.dtype == torch.bfloat16
    original = {name: parameter.dtype for name, parameter in model.named_parameters()}
    image = torch.rand(1, 3, 16, 16)
    for _ in range(2):
        output = model.embed_image(image)
        assert output.dtype == expected
        output.float().square().mean().backward()
        assert torch.isfinite(vision.embeddings.patch_embedding.weight.grad).all()
    assert original == {name: parameter.dtype for name, parameter in model.named_parameters()}


def test_openpi_vision_matches_reference_activation_cast():
    model = _backbone("openpi")
    visual = model.paligemma.model
    vision = visual.vision_tower.vision_model
    image = torch.rand(2, 3, 16, 16)
    # OpenPI's patched SigLIP: FP32 embeddings -> BF16 encoder -> norm/projector.
    hidden = vision.embeddings(image).to(torch.bfloat16)
    expected = visual.multi_modal_projector(
        vision.post_layernorm(vision.encoder(inputs_embeds=hidden).last_hidden_state)
    )
    torch.testing.assert_close(model.embed_image(image), expected, rtol=0, atol=0)
    video = model.embed_image(image[:, None], frame_mask=torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(video, expected)


def test_default_vision_matches_unmodified_hf_path():
    model = _backbone("lerobot")
    image = torch.rand(1, 3, 16, 16)
    torch.testing.assert_close(
        model.embed_image(image), model.paligemma.model.get_image_features(image).pooler_output
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
def test_lerobot_vision_matches_official_inference_autocast(device, training, dtype):
    model = _backbone("lerobot", dtype).to(device)
    model.train(training)
    visual = model.paligemma.model
    original_dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
    assert all(parameter.dtype == torch.float32 for parameter in visual.parameters())
    image = torch.rand(2, 3, 16, 16, device=device)
    use_bf16 = device == "cuda" and dtype == "bfloat16" and not training
    # Reference the upstream inference boundary independently of our helper.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        expected = visual.get_image_features(image).pooler_output.to(image.dtype)
    compute_dtypes = []
    hook = visual.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_hook(
        lambda _module, _inputs, output: compute_dtypes.append(output.dtype)
    )
    try:
        actual = model.embed_image(image)
    finally:
        hook.remove()
    assert compute_dtypes == [torch.bfloat16 if use_bf16 else torch.float32]
    assert actual.dtype == image.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if training:
        actual.square().mean().backward()
        assert torch.isfinite(visual.vision_tower.vision_model.embeddings.patch_embedding.weight.grad).all()
    assert original_dtypes == {name: parameter.dtype for name, parameter in model.named_parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lerobot_vision_training_retains_caller_amp_and_mem_inference_stays_fp32():
    model = _backbone("lerobot").to("cuda")
    image = torch.rand(1, 3, 16, 16, device="cuda")
    compute_dtypes = []
    hook = model.paligemma.model.vision_tower.vision_model.encoder.layers[
        0
    ].self_attn.q_proj.register_forward_hook(
        lambda _module, _inputs, output: compute_dtypes.append(output.dtype)
    )
    try:
        # The deployment update must not disable a training caller's AMP.
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.embed_image(image)
        output.square().mean().backward()
        assert compute_dtypes == [torch.bfloat16]
        compute_dtypes.clear()
        model.eval()
        video = model.embed_image(
            image[:, None], frame_mask=torch.ones(1, 1, dtype=torch.bool, device="cuda")
        )
        assert compute_dtypes == [torch.float32]
        assert video.dtype == torch.float32 and torch.isfinite(video).all()
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "profile,training,video,unequal_batches,camera_count,expected_calls",
    [
        pytest.param("lerobot", False, False, False, 2, 1, id="batched-inference"),
        pytest.param("lerobot", True, False, False, 2, 2, id="checkpointed-training"),
        pytest.param("lerobot", False, True, False, 2, 2, id="mem-per-camera"),
        pytest.param("lerobot", False, False, True, 2, 2, id="unequal-shapes"),
        pytest.param("lerobot", False, False, False, 1, 1, id="single-camera"),
        pytest.param("openpi", False, False, False, 2, 2, id="openpi-unchanged"),
        pytest.param("checkpoint", False, False, False, 2, 2, id="checkpoint-unchanged"),
    ],
)
def test_camera_batching_preserves_order_outputs_and_training_gradients(
    monkeypatch, profile, training, video, unequal_batches, camera_count, expected_calls
):
    model = modeling_pi05.PI05Pytorch.__new__(modeling_pi05.PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(precision_profile=profile, memory_temporal_attention_every=1)
    model.gradient_checkpointing_enabled = training
    model.paligemma_with_expert = _backbone(profile, "float32")
    model.train(training)
    images, masks = [], []
    for index in range(camera_count):
        batch_size = 1 if unequal_batches and index else 2
        shape = (batch_size, 2, 3, 16, 16) if video else (batch_size, 3, 16, 16)
        image = torch.rand(shape)
        mask = torch.ones(shape[:2] if video else shape[:1], dtype=torch.bool)
        if video:
            mask[0, 0] = index == 0
        images.append(image)
        masks.append(mask)
    with torch.no_grad():
        expected = [model._embed_one_image(img, mask) for img, mask in zip(images, masks, strict=True)]
    calls = []
    original_embed = model.paligemma_with_expert.embed_image

    def record_embed(image, **kwargs):
        calls.append(image.shape)
        return original_embed(image, **kwargs)

    monkeypatch.setattr(model.paligemma_with_expert, "embed_image", record_embed)
    actual = model._embed_images(images, masks)
    assert len(calls) == expected_calls
    for camera_output, camera_reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(camera_output, camera_reference, rtol=1e-5, atol=1e-6)
    if training:
        sum(output.square().mean() for output in actual).backward()
        patch = (
            model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding
        )
        assert patch.weight.grad is not None and torch.isfinite(patch.weight.grad).all()
        assert len(calls) == 2 * expected_calls


@pytest.mark.parametrize("profile", ["checkpoint", "lerobot", "openpi"])
def test_float32_checkpoint_remains_float32(profile):
    model = _backbone(profile, "float32")
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.embed_image(torch.rand(1, 3, 16, 16)).dtype == torch.float32


def test_checkpoint_bf16_casts_visual_embeddings_and_norm_parameters():
    model = _backbone("checkpoint")
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    image = torch.rand(1, 3, 16, 16)
    output = model.embed_image(image)
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    video = model.embed_image(image[:, None], frame_mask=torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(video, output)
    output.float().square().mean().backward()
    assert torch.isfinite(
        model.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight.grad
    ).all()
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize("profile", ["lerobot", "openpi"])
def test_official_vision_can_be_captured_without_graph_breaks(profile):
    model = _backbone(profile)
    image = torch.rand(1, 3, 16, 16)
    # This validates Dynamo capture on CPU; it does not claim CUDA/Inductor parity.
    compiled = torch.compile(model.embed_image, backend="eager", fullgraph=True)
    try:
        torch.testing.assert_close(compiled(image), model.embed_image(image))
    finally:
        torch._dynamo.reset()


def test_legacy_config_retains_lerobot_precision():
    assert PI05Config().precision_profile == "lerobot"
    with pytest.raises(ValueError, match="precision_profile"):
        PI05Config(precision_profile="unknown")


def _tiny_policy(monkeypatch, device, *, vision_patch_size=8, **overrides):
    def tiny_paligemma():
        return PaliGemmaConfig(
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": vision_patch_size,
            }
        )

    monkeypatch.setattr(modeling_pi05, "CONFIG_MAPPING", {"paligemma": tiny_paligemma, "gemma": GemmaConfig})
    monkeypatch.setattr(
        modeling_pi05, "get_gemma_config", lambda _: modeling_pi05.GemmaConfig(32, 1, 64, 8, 1, 4)
    )
    values = {
        "device": device,
        "dtype": "bfloat16",
        "precision_profile": "openpi",
        "chunk_size": 4,
        "n_action_steps": 4,
        "max_action_dim": 8,
        "num_inference_steps": 2,
        "image_resolution": (16, 16),
        "tokenizer_max_length": 8,
        "input_features": {"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 16, 16))},
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (7,))},
    }
    values.update(overrides)
    policy = modeling_pi05.PI05Policy(PI05Config(**values)).eval()
    # Production uses a 2048-wide VLM/projector; resize only this miniature
    # fixture's projector to the 32-wide language model used above.
    projector = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector
    projector.linear = nn.Linear(32, 32, dtype=projector.linear.weight.dtype, device=device)
    return policy


def _two_camera_features():
    return {
        f"observation.images.{camera}": PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
        for camera in ("top", "wrist")
    }


def test_batched_prefix_preserves_camera_padding_and_language_order(monkeypatch):
    model = _tiny_policy(monkeypatch, "cpu", precision_profile="lerobot").model
    images = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    img_masks = [torch.tensor([True, False]), torch.tensor([False, True])]
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    masks = torch.tensor([[True, True, False], [True, False, False]])
    with torch.no_grad():
        expected_images = [
            model._embed_one_image(img, mask) for img, mask in zip(images, img_masks, strict=True)
        ]
        expected_language = model.paligemma_with_expert.embed_language_tokens(tokens)
        actual, padding, attention = model.embed_prefix(images, img_masks, tokens, masks)
    torch.testing.assert_close(actual, torch.cat([*expected_images, expected_language], dim=1))
    expected_padding = torch.cat(
        [
            mask[:, None].expand(2, embedding.shape[1])
            for mask, embedding in zip(img_masks, expected_images, strict=True)
        ]
        + [masks],
        dim=1,
    )
    torch.testing.assert_close(padding, expected_padding)
    assert attention.shape == padding.shape and not attention.any()


@pytest.mark.parametrize(
    "profile,dtype",
    [
        ("lerobot", "bfloat16"),
        ("lerobot", "float32"),
        ("openpi", "bfloat16"),
        ("checkpoint", "bfloat16"),
        ("checkpoint", "float32"),
    ],
)
@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "eager"),
        pytest.param(
            "cuda",
            "inductor",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ),
    ],
)
def test_profile_complete_sampler_preserves_outputs_and_prefix_cache(
    monkeypatch, device, backend, profile, dtype
):
    precision = torch.get_float32_matmul_precision()
    try:
        policy = _tiny_policy(
            monkeypatch, device, precision_profile=profile, dtype=dtype, input_features=_two_camera_features()
        )
        original_dtypes = {name: parameter.dtype for name, parameter in policy.named_parameters()}
        if profile == "checkpoint":
            assert set(original_dtypes.values()) == {getattr(torch, dtype)}
        batch = {
            "observation.images.top": torch.rand(1, 3, 16, 16, device=device),
            "observation.images.wrist": torch.rand(1, 3, 16, 16, device=device),
            "observation.language.tokens": torch.ones(1, 8, dtype=torch.long, device=device),
            "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        }
        noise = torch.randn(1, 4, 8, device=device)
        expected = policy.predict_action_chunk(batch, noise=noise)
        assert expected.shape == (1, 4, 7) and expected.dtype == torch.float32
        assert torch.isfinite(expected).all()
        policy.model.sample_actions = torch.compile(
            policy.model.sample_actions,
            backend=backend,
            mode="max-autotune" if device == "cuda" else None,
            fullgraph=True,
        )
        for _ in range(3):
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            actual = policy.predict_action_chunk(batch, noise=noise)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-5)
            del actual
        backbone = policy.model.paligemma_with_expert
        assert original_dtypes == {name: parameter.dtype for name, parameter in policy.named_parameters()}
        assert backbone.paligemma.model.language_model.config._attn_implementation == "eager"
        assert backbone.gemma_expert.model.config._attn_implementation == "eager"
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
def test_lerobot_trained_rtc_keeps_hard_prefix_with_batched_compiled_vision(monkeypatch, device):
    precision = torch.get_float32_matmul_precision()
    try:
        policy = _tiny_policy(
            monkeypatch,
            device,
            precision_profile="lerobot",
            input_features=_two_camera_features(),
            rtc_config=RTCConfig(mode="trained", execution_horizon=2),
            rtc_training_max_delay=2,
        )
        batch = {
            **{
                key: torch.rand(1, *feature.shape, device=device)
                for key, feature in _two_camera_features().items()
            },
            "observation.language.tokens": torch.ones(1, 8, dtype=torch.long, device=device),
            "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        }
        previous = torch.randn(1, 3, 7, device=device)
        kwargs = {
            "noise": torch.randn(1, 4, 8, device=device),
            "inference_delay": 2,
            "prev_chunk_left_over": previous,
        }
        expected = policy.predict_action_chunk(batch, **kwargs)
        policy.model._prefill_prefix = torch.compile(
            policy.model._prefill_prefix,
            backend="eager" if device == "cpu" else "inductor",
            mode="max-autotune" if device == "cuda" else None,
            fullgraph=True,
        )
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        actual = policy.predict_action_chunk(batch, **kwargs)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-3)
        torch.testing.assert_close(actual[:, :2], previous[:, :2], rtol=0, atol=0)
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpoint_uniform_parameters_support_training_and_memory(monkeypatch, dtype):
    policy = _tiny_policy(
        monkeypatch,
        "cpu",
        precision_profile="checkpoint",
        dtype=dtype,
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=2,
        memory_temporal_attention_every=1,
    ).train()
    assert {parameter.dtype for parameter in policy.parameters()} == {getattr(torch, dtype)}
    inputs = {
        "images": [torch.rand(1, 2, 3, 16, 16)],
        "img_masks": [torch.ones(1, 2, dtype=torch.bool)],
        "tokens": torch.ones(1, 8, dtype=torch.long),
        "masks": torch.ones(1, 8, dtype=torch.bool),
        "states": torch.rand(1, 2, policy.config.max_state_dim),
        "state_masks": torch.ones(1, 2, dtype=torch.bool),
    }
    noise = torch.randn(1, 4, 8)
    loss = policy.model(**inputs, actions=torch.rand_like(noise), noise=noise, time=torch.tensor([0.5]))
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all()
    loss.mean().backward()
    for projection in (
        policy.model.action_in_proj,
        policy.model.time_mlp_in,
        policy.model.proprio_history_proj,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
    policy.eval()
    actions = policy.model.sample_actions(**inputs, noise=noise)
    assert actions.dtype == torch.float32 and torch.isfinite(actions).all()
    assert {parameter.dtype for parameter in policy.parameters()} == {getattr(torch, dtype)}


@pytest.mark.skipif(not _grpc_available, reason="requires the async deployment extra")
@pytest.mark.parametrize("profile", ["lerobot", "openpi", "checkpoint"])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
def test_guided_compile_reuses_graph_across_timesteps_delays_and_prefixes(
    monkeypatch, device, fast, fullgraph, profile
):
    original_compile = torch.compile

    def compile_for_test(function, *, mode):
        return original_compile(
            function,
            backend="eager" if device == "cpu" else "inductor",
            mode=None if device == "cpu" else mode,
            # False matches production. True also verifies that each selected
            # compiled region can be captured without falling back at a break.
            fullgraph=fullgraph,
        )

    monkeypatch.setattr(torch, "compile", compile_for_test)
    precision = torch.get_float32_matmul_precision()
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    graphs_before = counters["stats"]["unique_graphs"]
    try:
        reference = _tiny_policy(
            monkeypatch,
            device,
            precision_profile=profile,
            input_features=_two_camera_features(),
            num_inference_steps=10,
            rtc_config=RTCConfig(execution_horizon=3),
        )
        policy = _tiny_policy(
            monkeypatch,
            device,
            precision_profile=profile,
            input_features=_two_camera_features(),
            num_inference_steps=10,
            rtc_config=RTCConfig(execution_horizon=3),
            compile_model=True,
            compile_mode="max-autotune",
        )
        policy.load_state_dict(reference.state_dict())
        policy.model.rtc_processor = policy.rtc_processor = _LightweightGuidedRTCProcessor(
            policy.config.rtc_config, fast=fast
        )
        batch = {
            "observation.images.top": torch.rand(1, 3, 16, 16, device=device),
            "observation.images.wrist": torch.rand(1, 3, 16, 16, device=device),
            "observation.language.tokens": torch.ones(1, 8, dtype=torch.long, device=device),
            "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        }
        noise = torch.randn(1, 4, 8, device=device)
        previous = torch.randn(3, 7, device=device)
        for index, (delay, length) in enumerate([(0, 0), (1, 3), (0, 1), (2, 3), (0, 0), (0, 2), (1, 3)]):
            # Observation values and padding change without changing shapes.
            # A compiled prefill must consume them, not reuse an older KV cache.
            batch["observation.images.top"] = torch.rand(1, 3, 16, 16, device=device)
            batch["observation.images.wrist"] = torch.rand(1, 3, 16, 16, device=device)
            batch["observation.language.tokens"].fill_(index + 1)
            batch["observation.language.attention_mask"][:, -2:] = index % 2 == 0
            kwargs = {
                "noise": noise,
                "inference_delay": delay,
                "prev_chunk_left_over": previous[:length] if length else None,
            }
            expected = reference.predict_action_chunk(batch, **kwargs)
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            actual = policy.predict_action_chunk(batch, **kwargs)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-3)
            del actual
            # Default guidance compiles the entire sampler just like ordinary
            # inference. The reference keeps separate prefill/denoiser graphs.
            # Changing observations, time, delay and length must reuse them.
            total_graphs = 2 if fast else 3
            assert counters["stats"]["unique_graphs"] - graphs_before == (1 if index == 0 else total_graphs)
        assert all(parameter.grad is None for parameter in policy.parameters())
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
@pytest.mark.parametrize("profile", ["lerobot", "openpi"])
def test_compiled_prefill_refreshes_observation_and_preserves_cache_during_denoising(
    monkeypatch, device, profile
):
    precision = torch.get_float32_matmul_precision()
    torch._dynamo.reset()
    try:
        model = _tiny_policy(monkeypatch, device, precision_profile=profile).model
        compiled = torch.compile(
            model._prefill_prefix,
            backend="eager" if device == "cpu" else "inductor",
            mode="max-autotune" if device == "cuda" else None,
            fullgraph=True,
        )
        inputs = {
            "images": [torch.rand(1, 3, 16, 16, device=device) for _ in range(2)],
            "img_masks": [torch.ones(1, dtype=torch.bool, device=device) for _ in range(2)],
            "tokens": torch.ones(1, 8, dtype=torch.long, device=device),
            "masks": torch.ones(1, 8, dtype=torch.bool, device=device),
        }
        noise = torch.randn(1, 4, 8, device=device)

        def snapshot(cache):
            return [(k.detach().cpu().clone(), v.detach().cpu().clone()) for k, v, _ in cache]

        @torch.no_grad()
        def prefill_and_denoise():
            expected_mask, expected_cache = model._prefill_prefix(**inputs)
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            mask, cache = compiled(**inputs)
            torch.testing.assert_close(mask, expected_mask)
            before = snapshot(cache)
            for actual, expected in zip(before, snapshot(expected_cache), strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-3)
            for time in (1.0, 0.5, 0.1):
                velocity = model.denoise_step(mask, cache, noise, torch.full((1,), time, device=device))
                assert torch.isfinite(velocity).all()
            for actual, expected in zip(snapshot(cache), before, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return before

        original = prefill_and_denoise()
        inputs["images"][0] = torch.rand_like(inputs["images"][0])
        changed_image = prefill_and_denoise()
        assert not torch.equal(original[0][0], changed_image[0][0])
        inputs["tokens"].fill_(2)
        inputs["masks"][:, -2:] = False
        changed_language = prefill_and_denoise()
        assert not torch.equal(changed_image[0][0], changed_language[0][0])
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(precision)


@pytest.mark.skipif(
    not _grpc_available or not torch.cuda.is_available(), reason="requires CUDA and async extra"
)
@pytest.mark.parametrize(
    "profile,compile_mode",
    [
        ("lerobot", "max-autotune"),
        ("openpi", "default"),
        ("openpi", "max-autotune"),
        ("openpi", "max-autotune-no-cudagraphs"),
        ("checkpoint", "max-autotune"),
    ],
)
def test_cuda_worker_warmup_with_generated_noise_and_deployment_shapes(profile, compile_mode):
    """Exercise the serving warmup, including CUDA RNG capture and restoration.

    Keep real input lengths (two 224px cameras, 200 tokens, 50 actions) while
    reducing model widths/layers for CI. This does not emulate A100 kernels.
    """
    # Each mode represents a fresh server process. CUDAGraph Trees in PyTorch
    # 2.11 keeps thread-local state; importing it on an earlier test's thread
    # and then changing workers does not model a fresh deployment startup.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging, pytest; "
                "logging.basicConfig(level=logging.INFO); "
                "from tests.policies.pi0_pi05.test_pi05_inference_precision "
                "import _cuda_worker_warmup; "
                f"_cuda_worker_warmup(pytest.MonkeyPatch(), {compile_mode!r}, profile={profile!r})",
            ],
            capture_output=True,
            text=True,
            # Cold compilation can exceed 240s. Match the server's default
            # warmup budget rather than failing before its supported timeout.
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = "\n".join(
            part.decode(errors="replace") if isinstance(part, bytes) else part or ""
            for part in (error.stdout, error.stderr)
        )
        pytest.fail(f"CUDA worker warmup exceeded 600s:\n{output}", pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _cuda_worker_warmup(monkeypatch, compile_mode, *, profile="openpi"):
    precision = torch.get_float32_matmul_precision()
    try:
        policy = _tiny_policy(
            monkeypatch,
            "cuda",
            precision_profile=profile,
            vision_patch_size=14,
            compile_model=True,
            compile_mode=compile_mode,
            chunk_size=50,
            n_action_steps=50,
            max_action_dim=32,
            num_inference_steps=10,
            image_resolution=(224, 224),
            tokenizer_max_length=200,
            rtc_config=RTCConfig(execution_horizon=25),
            input_features={
                "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
                "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            },
        )
        policy.rtc_processor = policy.model.rtc_processor = _LightweightGuidedRTCProcessor(
            policy.config.rtc_config
        )
        outputs = []

        def postprocess(actions):
            output = actions.detach().cpu().clone()
            outputs.append(output)
            return output

        def warm_on_worker(delay):
            warmup_pi05(
                policy,
                runs=3,
                prefix_length=25,
                inference_delay=delay,
                logger=logging.getLogger(__name__),
                postprocess=postprocess,
            )

        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        # Reconnect keeps the compiled model but creates a new inference worker.
        # Exercise CUDA graph output/cache lifetimes across both worker sessions.
        for delay in (25, 10):
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(warm_on_worker, delay).result()
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        assert len(outputs) == 12
        assert all(output.shape == (1, 50, 7) and torch.isfinite(output).all() for output in outputs)
        # CUDA graph replay must advance the internally generated sampling noise.
        assert not torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[1], outputs[2])
    finally:
        monkeypatch.undo()
        torch.set_float32_matmul_precision(precision)
