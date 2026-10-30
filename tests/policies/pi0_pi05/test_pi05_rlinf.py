# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks using real, miniature Transformers and RLinf network layers."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

from transformers import PaliGemmaConfig  # noqa: E402

from lerobot.async_inference.shared.rtc_remote import supports_remote_rtc  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    modeling_pi05 as legacy,  # noqa: E402
)
from lerobot.policies.pi05.rlinf import gemma, siglip  # noqa: E402
from lerobot.policies.pi05.rlinf.checkpoint import to_rlinf_state_dict  # noqa: E402
from lerobot.policies.rtc import RTCConfig  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


@pytest.fixture
def tiny_networks(monkeypatch):
    """Reduce dimensions, keeping the real two-stream attention and vision paths."""
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    monkeypatch.setattr(gemma, "get_config", lambda _: gemma.Config(32, 2, 64, 8, 1, 4))
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda _: {"width": 32, "depth": 2, "num_heads": 4, "mlp_dim": 64, "patch_size": (14, 14)},
    )
    monkeypatch.setattr(legacy, "get_gemma_config", lambda _: legacy.GemmaConfig(32, 2, 64, 8, 1, 4))
    original_paligemma = legacy.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = legacy.PiGemmaForCausalLM

    def vlm_config():
        return PaliGemmaConfig(
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 224,
                "patch_size": 14,
            }
        )

    def make_vlm(*, config):
        config.text_config.vocab_size = 64
        config.vision_config.intermediate_size = 64
        config.vision_config.projection_dim = 32
        return original_paligemma(config)

    def make_expert(*, config):
        config.vocab_size = 64
        return original_expert(config)

    mapping = {"paligemma": vlm_config, "gemma": legacy.CONFIG_MAPPING["gemma"]}
    monkeypatch.setattr(legacy, "CONFIG_MAPPING", mapping)
    monkeypatch.setattr(legacy, "PaliGemmaForConditionalGenerationWithPiGemma", make_vlm)
    monkeypatch.setattr(legacy, "PiGemmaForCausalLM", make_expert)


def config(**kwargs):
    return PI05Config(
        device="cpu",
        backend="rlinf",
        chunk_size=4,
        n_action_steps=2,
        num_inference_steps=2,
        max_action_dim=4,
        max_state_dim=4,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (3,))},
        **kwargs,
    )


def inputs():
    generator = torch.Generator().manual_seed(7)
    return (
        [torch.rand(1, 3, 224, 224, generator=generator) * 2 - 1],
        [torch.ones(1, dtype=torch.bool)],
        torch.tensor([[1, 2, 3, 0]]),
        torch.tensor([[True, True, True, False]]),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "invalid"},
        {"use_visual_memory": True},
        {"use_proprioceptive_memory": True},
        {"image_resolution": (256, 256)},
    ],
)
def test_invalid_rlinf_configuration(kwargs):
    with pytest.raises(ValueError):
        PI05Config(**({"backend": "rlinf"} | kwargs))


def test_legacy_configs_still_default_to_original_backend():
    assert PI05Config().backend == "lerobot"


def test_full_size_checkpoint_layout_matches_without_allocating_model_weights():
    from lerobot.policies.pi05.rlinf.modeling import RLinfPI05Pytorch

    with torch.device("meta"):
        source = legacy.PI05Pytorch(PI05Config(device="cpu"))
        target = RLinfPI05Pytorch(PI05Config(backend="rlinf", device="cpu"))
        converted = to_rlinf_state_dict(source.state_dict())
    expected = target.state_dict()
    assert converted.keys() == expected.keys()
    for name, tensor in converted.items():
        assert tensor.shape == expected[name].shape, name


def test_converted_weights_match_real_transformers_outputs(tiny_networks):
    torch.manual_seed(4)
    source = legacy.PI05Pytorch(replace(config(), backend="lerobot")).eval()
    for module in source.modules():
        if getattr(module, "dense", None) is not None and hasattr(module, "cond_dim"):
            torch.nn.init.normal_(module.dense.weight, std=0.03)
    target = PI05Policy(config()).model.eval()
    target.load_state_dict(to_rlinf_state_dict(source.state_dict()))
    # Isolate layout conversion from the intentional upstream GELU difference.
    for layer in source.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers:
        layer.mlp.activation_fn = torch.nn.GELU()
    for stream in (
        source.paligemma_with_expert.paligemma.model.language_model,
        source.paligemma_with_expert.gemma_expert.model,
    ):
        for layer in stream.layers:
            layer.mlp.act_fn = torch.nn.GELU()
    images, image_masks, tokens, masks = inputs()
    noise = torch.randn(1, 4, 4)
    actions = torch.randn(1, 4, 4)
    time = torch.tensor([0.4])
    source_loss = source(images, image_masks, tokens, masks, actions, noise, time)
    target_loss = target(images, image_masks, tokens, masks, actions, noise, time)
    torch.testing.assert_close(target_loss, source_loss, rtol=2e-4, atol=2e-4)
    target_loss.mean().backward()
    source_loss.mean().backward()
    torch.testing.assert_close(
        target.action_in_proj.weight.grad, source.action_in_proj.weight.grad, rtol=5e-4, atol=5e-5
    )
    with torch.no_grad():
        expected = source.sample_actions(images, image_masks, tokens, masks, noise=noise.clone())
        actual = target.sample_actions(images, image_masks, tokens, masks, noise=noise.clone())
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_cached_suffix_matches_full_pass_and_keeps_prefix_immutable(tiny_networks):
    model = PI05Policy(config()).model.eval()
    args = inputs()
    actions, noise, time = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.tensor([0.3])
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    _, mask, cache = model.build_prefix_cache(*args)
    before = [(key.clone(), value.clone()) for key, value in cache]
    velocity = model.denoise_step(mask, cache, x_t, time)
    expected = (velocity - (noise - actions)).square()
    actual = model(*args, actions, noise, time)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    model.denoise_step(mask, cache, x_t, time)
    for (key, value), (old_key, old_value) in zip(cache, before, strict=True):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_real_rtc_sampling_and_remote_contract(tiny_networks, mode):
    cfg = config(rtc_config=RTCConfig(mode=mode, execution_horizon=2), rtc_training_max_delay=2)
    policy = PI05Policy(cfg)
    assert supports_remote_rtc(policy)
    previous = torch.randn(1, 4, 4)
    output = policy.model.sample_actions(
        *inputs(),
        noise=torch.randn(1, 4, 4),
        inference_delay=1,
        prev_chunk_left_over=previous,
        execution_horizon=2,
    )
    assert output.shape == (1, 4, 4)
    assert torch.isfinite(output).all()
    if mode == "trained":
        torch.testing.assert_close(output[:, :1], previous[:, :1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_cuda_sampling_with_real_rtc(tiny_networks, dtype, mode):
    policy = PI05Policy(
        config(dtype=dtype, rtc_config=RTCConfig(mode=mode, execution_horizon=2), rtc_training_max_delay=2)
    ).to("cuda")
    images, image_masks, tokens, masks = inputs()
    arguments = ([images[0].cuda()], [image_masks[0].cuda()], tokens.cuda(), masks.cuda())
    previous = torch.randn(1, 4, 4, device="cuda")
    output = policy.model.sample_actions(
        *arguments,
        noise=torch.randn(1, 4, 4, device="cuda"),
        inference_delay=1,
        prev_chunk_left_over=previous,
        execution_horizon=2,
    )
    assert output.is_cuda
    assert torch.isfinite(output).all()
    if mode == "trained":
        torch.testing.assert_close(output[:, :1], previous[:, :1], rtol=0, atol=0)


def test_per_action_time_matches_scalar_and_trained_forward_has_gradients(tiny_networks):
    model = PI05Policy(config()).model
    # Activate gates so the test exercises adaptive conditioning, not zero gates.
    for module in model.modules():
        if isinstance(module, gemma.RMSNorm) and module.adaptive:
            torch.nn.init.normal_(module.ada_modulation.weight, std=0.03)
    args = inputs()
    _, mask, cache = model.build_prefix_cache(*args)
    noise, actions, time = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.tensor([0.25])
    scalar = model.denoise_step(mask, cache, noise, time)
    repeated = model.denoise_step(mask, cache, noise, time[:, None].expand(1, 4))
    torch.testing.assert_close(scalar, repeated)
    model.gradient_checkpointing_enable()
    loss = model(*args, actions, noise, time, prefix_mask=torch.tensor([[True, False, False, False]]))
    loss[:, 1:].mean().backward()
    assert torch.isfinite(model.action_in_proj.weight.grad).all()


def test_roundtrip_checkpoint_and_fail_closed_loading(tiny_networks, tmp_path):
    policy = PI05Policy(config())
    policy.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).backend == "rlinf"
    loaded = PI05Policy.from_pretrained(tmp_path)
    direct = PI05Policy.from_pretrained(tmp_path / "model.safetensors", config=config())
    torch.testing.assert_close(direct.model.action_in_proj.weight, loaded.model.action_in_proj.weight)
    noise = torch.randn(1, 4, 4)
    torch.testing.assert_close(
        policy.model.sample_actions(*inputs(), noise=noise.clone()),
        loaded.model.sample_actions(*inputs(), noise=noise.clone()),
        rtol=0,
        atol=0,
    )
    state = policy.state_dict()
    state.pop("model.action_in_proj.weight")
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="action_in_proj.weight"):
        PI05Policy.from_pretrained(tmp_path)


def test_legacy_checkpoint_loads_and_unknown_or_incomplete_weights_fail(tiny_networks, tmp_path):
    source = legacy.PI05Pytorch(replace(config(), backend="lerobot"))
    state = {"model." + key: value for key, value in source.state_dict().items()}
    save_file({key: value.clone() for key, value in state.items()}, tmp_path / "model.safetensors")
    loaded = PI05Policy.from_pretrained(tmp_path, config=config())
    replace(config(), backend="lerobot").save_pretrained(tmp_path)
    overridden = PI05Policy.from_pretrained(tmp_path, cli_overrides=["--backend=rlinf"])
    assert overridden.config.backend == "rlinf"
    torch.testing.assert_close(
        loaded.model.action_in_proj.weight, source.action_in_proj.weight, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="Unrecognized"):
        to_rlinf_state_dict(state | {"model.adapter.bad.weight": torch.ones(1)})
    state.pop(
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight"
    )
    with pytest.raises(ValueError, match="Incomplete qkv"):
        to_rlinf_state_dict(state)


def test_policy_chunk_queue_reset_and_bfloat16_precision(tiny_networks):
    policy = PI05Policy(config(dtype="bfloat16"))
    assert policy.model.img.stem.weight.dtype == torch.float32
    assert policy.model.llm.layers[0].attn.q_proj[0].weight.dtype == torch.bfloat16
    assert policy.model.action_out_proj.weight.dtype == torch.float32
    images, _, tokens, mask = inputs()
    batch = {
        OBS_STATE: torch.zeros(1, 3),
        "observation.images.front": (images[0] + 1) / 2,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    chunk = policy.predict_action_chunk(batch, noise=torch.randn(1, 4, 4))
    assert chunk.shape == (1, 4, 3)
    assert torch.isfinite(chunk).all()
    assert policy.select_action(batch).shape == (1, 3)
    assert len(policy._action_queue) == 1
    policy.reset()
    assert len(policy._action_queue) == 0


@pytest.mark.parametrize("profile", ["lerobot", "openpi", "checkpoint"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_precision_profiles_train_and_sample_with_trained_rtc(tiny_networks, profile, dtype):
    policy = PI05Policy(
        config(
            dtype=dtype,
            precision_profile=profile,
            rtc_config=RTCConfig(mode="trained", execution_horizon=2),
            rtc_training_max_delay=2,
        )
    )
    if profile == "checkpoint":
        assert {parameter.dtype for parameter in policy.parameters()} == {getattr(torch, dtype)}
    elif profile == "lerobot":
        assert {parameter.dtype for parameter in policy.model.img.parameters()} == {torch.float32}
    noise = torch.randn(1, 4, 4)
    actions = torch.randn_like(noise)
    loss = policy.model(*inputs(), actions=actions, noise=noise, time=torch.tensor([0.5])).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.model.action_in_proj.weight.grad is not None
    assert torch.isfinite(policy.model.action_in_proj.weight.grad).all()
    policy.eval()
    output = policy.model.sample_actions(
        *inputs(), noise=noise, prev_chunk_left_over=actions, inference_delay=1
    )
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :1], actions[:, :1])


@pytest.mark.parametrize("expert_only", [False, True])
def test_freezing_keeps_trainable_heads_and_frozen_modules_in_eval(tiny_networks, expert_only):
    policy = PI05Policy(config(freeze_vision_encoder=True, train_expert_only=expert_only)).train()
    assert not policy.model.img.stem.weight.requires_grad
    assert not policy.model.img.pos_embedding.requires_grad
    assert policy.model.action_in_proj.weight.requires_grad
    assert policy.model.img.head.weight.requires_grad is not expert_only
    assert policy.model.llm.layers[0].attn.q_proj[0].weight.requires_grad is not expert_only
    assert policy.model.llm.layers[0].attn.q_proj[1].weight.requires_grad
    assert not policy.model.img.encoder.training


def test_dynamo_can_trace_native_sampling_without_changing_outputs(tiny_networks):
    model = PI05Policy(config()).model.eval()
    noise = torch.randn(1, 4, 4)
    expected = model.sample_actions(*inputs(), noise=noise.clone())
    traced = torch.compile(model.sample_actions, backend="eager")
    actual = traced(*inputs(), noise=noise.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_comparison_tool_uses_same_checkpoint_and_reports_measured_differences(tiny_networks, tmp_path):
    from examples.tutorial.pi05.compare_backends import compare

    policy = PI05Policy(replace(config(), backend="lerobot"))
    policy.save_pretrained(tmp_path)
    images, _, tokens, mask = inputs()
    batch = {
        OBS_STATE: torch.zeros(1, 3),
        "observation.images.front": (images[0] + 1) / 2,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    report = compare(str(tmp_path), batch, batch_is_preprocessed=True, iterations=2, warmup=1)
    assert report["action_shape"] == [1, 4, 3]
    assert report["normalized_action_error"]["mean_absolute_error"] > 0
    assert all(data["mean_latency_ms"] > 0 for data in report["measurements"].values())
    assert report["measurements"]["rlinf"]["peak_allocated_bytes"] is None
    state = {key: value.clone() for key, value in policy.state_dict().items()}
    state.pop("model.action_in_proj.weight")
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="action_in_proj.weight"):
        compare(str(tmp_path), batch, batch_is_preprocessed=True, iterations=1)


def test_comparison_rejects_adapter_directory_even_if_full_weights_are_present(tiny_networks, tmp_path):
    from examples.tutorial.pi05.compare_backends import compare

    source = PI05Policy(replace(config(), backend="lerobot", use_peft=True))
    source.save_pretrained(tmp_path)
    images, _, tokens, mask = inputs()
    batch = {
        OBS_STATE: torch.zeros(1, 3),
        "observation.images.front": (images[0] + 1) / 2,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    with pytest.raises(ValueError, match="Merge the PI05 adapter"):
        compare(str(tmp_path), batch, batch_is_preprocessed=True, warmup=0, iterations=1)


def test_comparison_export_preserves_normalization_and_relative_action_processors(tiny_networks, tmp_path):
    from examples.tutorial.pi05.compare_backends import compare
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
    from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines

    source, exported = tmp_path / "original", tmp_path / "rlinf"
    cfg = replace(
        config(),
        backend="lerobot",
        use_relative_actions=True,
        action_feature_names=["joint_1", "joint_2", "gripper"],
    )
    policy = PI05Policy(cfg)
    policy.save_pretrained(source)
    stats = {
        key: {"q01": torch.full((3,), -2.0), "q99": torch.full((3,), 6.0)} for key in (OBS_STATE, ACTION)
    }
    steps = make_default_policy_processor_steps(cfg, stats)
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=cfg.action_feature_names
    )
    pre, post = make_policy_processor_pipelines(
        [steps.add_batch_dim, relative, steps.normalize, steps.to_device],
        [steps.unnormalize, AbsoluteActionsProcessorStep(enabled=True, relative_step=relative), steps.to_cpu],
    )
    pre.save_pretrained(source)
    post.save_pretrained(source)
    images, _, tokens, mask = inputs()
    batch = {
        OBS_STATE: torch.tensor([[0.2, -0.1, 0.5]]),
        "observation.images.front": (images[0] + 1) / 2,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
    }
    report = compare(str(source), batch, warmup=0, iterations=1, export_dir=exported)
    assert "executable_action_error" in report
    loaded = PI05Policy.from_pretrained(exported)
    pre, post = make_pre_post_processors(loaded.config, pretrained_path=str(exported))
    output = post(loaded.predict_action_chunk(pre(batch)))
    assert output.shape == (1, 4, 3)
    assert torch.isfinite(output).all()
    with pytest.raises(FileExistsError):
        compare(str(source), batch, export_dir=exported)
