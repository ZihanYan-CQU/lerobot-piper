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

"""Exercise mixed PI05 fine-tuning with real, miniature network and PEFT layers."""

import math
import re
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import PeftModel  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from tests.policies.pi0_pi05.test_pi05_rlinf import (  # noqa: E402
    config,
    inputs,
    tiny_networks as tiny_networks,  # noqa: E402
)


def training_batch():
    images, _, tokens, mask = inputs()
    return {
        OBS_STATE: torch.tensor([[0.2, -0.1, 0.5]]),
        "observation.images.front": (images[0] + 1) / 2,
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: mask,
        ACTION: torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(9)),
    }


def train_steps(policy, steps=3, batch=None, optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    policy.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(13)
        loss, _ = policy(training_batch() if batch is None else batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    return optimizer


def processors(cfg):
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
    from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines

    stats = {
        key: {"q01": torch.full((3,), -2.0), "q99": torch.full((3,), 6.0)} for key in (OBS_STATE, ACTION)
    }
    steps = make_default_policy_processor_steps(cfg, stats)
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=cfg.action_feature_names
    )
    return make_policy_processor_pipelines(
        [steps.add_batch_dim, relative, steps.normalize, steps.to_device],
        [steps.unnormalize, AbsoluteActionsProcessorStep(enabled=True, relative_step=relative), steps.to_cpu],
    )


def dataset_metadata(cfg):
    return SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (3,)},
            "observation.images.front": {
                "dtype": "image",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            },
            ACTION: {"dtype": "float32", "shape": (3,), "names": cfg.action_feature_names},
        },
        stats={},
    )


def component_lora_targets(backend):
    if backend == "rlinf":
        return {
            "language_attention": {
                f"model.llm.layers.{index}.attn.{projection}_proj.0"
                for index in range(2)
                for projection in "qkvo"
            },
            "language_mlp": {f"model.llm.layers.{index}.mlps.0" for index in range(2)},
            "vision_attention": {f"model.img.encoder.layers.{index}.attn" for index in range(2)},
            "vision_mlp": {
                f"model.img.encoder.layers.{index}.mlp.fc{projection}"
                for index in range(2)
                for projection in (1, 2)
            },
        }
    prefix = "model.paligemma_with_expert.paligemma.model."
    return {
        "language_attention": {
            f"{prefix}language_model.layers.{index}.self_attn.{projection}_proj"
            for index in range(2)
            for projection in "qkvo"
        },
        "language_mlp": {
            f"{prefix}language_model.layers.{index}.mlp.{projection}_proj"
            for index in range(2)
            for projection in ("gate", "up", "down")
        },
        "vision_attention": {
            f"{prefix}vision_tower.vision_model.encoder.layers.{index}.self_attn.{projection}_proj"
            for index in range(2)
            for projection in ("q", "k", "v", "out")
        },
        "vision_mlp": {
            f"{prefix}vision_tower.vision_model.encoder.layers.{index}.mlp.fc{projection}"
            for index in range(2)
            for projection in (1, 2)
        },
    }


@pytest.mark.usefixtures("tiny_networks")
def test_default_lora_time_projections_train_and_reload(tmp_path):
    torch.manual_seed(5)
    source = PI05Policy(replace(config(), backend="lerobot"))
    # Fresh AdaRMS modulation weights are zero; use a nonzero conditioning path
    # to exercise time-adapter gradients as in a pretrained action expert.
    for module in source.model.paligemma_with_expert.gemma_expert.modules():
        if getattr(module, "dense", None) is not None and hasattr(module, "cond_dim"):
            torch.nn.init.normal_(module.dense.weight, std=0.03)
    base = tmp_path / "base"
    source.save_pretrained(base)
    source.config.pretrained_path = base
    policy = source.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    time_layers = {
        name: policy.get_base_model().get_submodule(f"model.{name}")
        for name in ("time_mlp_in", "time_mlp_out")
    }
    for name, layer in time_layers.items():
        assert hasattr(layer, "lora_A") and hasattr(layer, "lora_B"), name
        assert layer.lora_A["default"].weight.requires_grad
        assert layer.lora_B["default"].weight.requires_grad
        assert not layer.base_layer.weight.requires_grad

    train_steps(policy)
    for name, layer in time_layers.items():
        weight = layer.lora_B["default"].weight
        assert weight.grad is not None and torch.isfinite(weight.grad).all(), name
        assert weight.grad.count_nonzero() > 0, name
        assert weight.detach().count_nonzero() > 0, name

    adapter = tmp_path / "adapter"
    policy.save_pretrained(adapter)
    reloaded = PeftModel.from_pretrained(PI05Policy.from_pretrained(base), adapter, is_trainable=True)
    for name, layer in time_layers.items():
        restored = reloaded.get_base_model().get_submodule(f"model.{name}")
        for component in ("lora_A", "lora_B"):
            torch.testing.assert_close(
                getattr(restored, component)["default"].weight,
                getattr(layer, component)["default"].weight,
                rtol=0,
                atol=0,
            )


def test_explicit_legacy_recipe_trains_saves_reloads_and_merges(tiny_networks, tmp_path):
    """The user's existing CLI recipe needs no policy implementation changes."""
    base = tmp_path / "base"
    source = PI05Policy(replace(config(), backend="lerobot"))
    source.save_pretrained(base)
    source.config.pretrained_path = base
    initial = {name: value.clone() for name, value in source.state_dict().items()}
    policy = source.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "LORA",
            "r": 4,
            "lora_alpha": 8,
            "target_modules": r".*[.]paligemma[.].*[.](q|k|v|o|out)_proj",
            "full_training_modules": [
                "model.paligemma_with_expert.gemma_expert.model",
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
            ],
        }
    )
    train_steps(policy)
    adapter = tmp_path / "adapter"
    policy.save_pretrained(adapter)
    saved = load_file(adapter / "adapter_model.safetensors")
    assert any("gemma_expert.model.layers.0.mlp" in name for name in saved)
    assert any("vision_tower" in name and "lora_B" in name and value.any() for name, value in saved.items())
    assert any("language_model" in name and "lora_B" in name and value.any() for name, value in saved.items())
    noise = torch.randn(1, 4, 4)
    expected = policy.predict_action_chunk(training_batch(), noise=noise.clone())
    reloaded = PeftModel.from_pretrained(PI05Policy.from_pretrained(base), adapter, is_trainable=True)
    torch.testing.assert_close(
        reloaded.predict_action_chunk(training_batch(), noise=noise.clone()), expected, rtol=0, atol=0
    )
    merged = reloaded.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(
        merged.predict_action_chunk(training_batch(), noise=noise.clone()), expected, rtol=1e-4, atol=1e-5
    )
    state = merged.state_dict()
    assert not torch.equal(state["model.action_out_proj.weight"], initial["model.action_out_proj.weight"])
    expert_mlp = "model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight"
    assert not torch.equal(state[expert_mlp], initial[expert_mlp])
    frozen = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    torch.testing.assert_close(state[frozen], initial[frozen], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.filterwarnings("error:Found missing adapter keys")
def test_mixed_profile_checkpoint_factory_reload_and_merged_inference(
    tiny_networks, tmp_path, backend, checkpointing, dtype, device
):
    from examples.tutorial.pi05.merge_lora import merge_adapter
    from lerobot.common.train_utils import resume_after_prepare, resume_before_prepare, save_checkpoint
    from lerobot.configs.default import DatasetConfig, PeftConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    base = tmp_path / "base"
    PI05Policy(replace(config(), backend="lerobot")).save_pretrained(base)
    cfg = replace(
        config(),
        backend=backend,
        dtype=dtype,
        device=device,
        peft_mode="vlm_lora_expert_full",
        pretrained_path=base,
        gradient_checkpointing=checkpointing,
        use_relative_actions=True,
        action_feature_names=["joint_1", "joint_2", "gripper"],
    )
    policy = PI05Policy.from_pretrained(base, config=cfg)
    initial = {name: value.clone() for name, value in policy.state_dict().items()}
    expert_pattern = (
        r"model\.llm\..*(attn\.[qkvo]_proj|pre_attention_norms|pre_ffw_norms|mlps|final_norms)\.1\."
        if backend == "rlinf"
        else r"model\.paligemma_with_expert\.gemma_expert\.model\."
    )
    expected_full = {
        name
        for name in initial
        if re.match(expert_pattern, name)
        or name.startswith(
            ("model.action_in_proj.", "model.action_out_proj.", "model.time_mlp_in.", "model.time_mlp_out.")
        )
    }
    pre, post = processors(cfg)
    policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name or ".modules_to_save.default." in name, name
            if "lora_" in name:
                assert "gemma_expert" not in name
                assert not re.search(r"(attn\.[qkvo]_proj|mlps)\.1\.", name)
        else:
            assert "lora_" not in name
    active_full = {
        name.removeprefix("base_model.model.").replace(".modules_to_save.default.", ".")
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    }
    assert active_full == expected_full  # Includes every AdaRMS and expert FFN tensor.
    if backend == "rlinf":
        language_mlp_targets = [f"model.llm.layers.{index}.mlps.0" for index in range(2)]
        vision_mlp_targets = [
            f"model.img.encoder.layers.{index}.mlp.fc{projection}"
            for index in range(2)
            for projection in (1, 2)
        ]
    else:
        prefix = "model.paligemma_with_expert.paligemma.model."
        language_mlp_targets = [
            f"{prefix}language_model.layers.{index}.mlp.{projection}_proj"
            for index in range(2)
            for projection in ("gate", "up", "down")
        ]
        vision_mlp_targets = [
            f"{prefix}vision_tower.vision_model.encoder.layers.{index}.mlp.fc{projection}"
            for index in range(2)
            for projection in (1, 2)
        ]
    for target in language_mlp_targets + vision_mlp_targets:
        layer = policy.get_base_model().get_submodule(target)
        assert "default" in layer.lora_A and "default" in layer.lora_B, target
    optimizer = train_steps(policy, batch=pre(training_batch()))
    current = {
        name.removeprefix("base_model.model.")
        .replace(".base_layer.", ".")
        .replace(".modules_to_save.default.", "."): value
        for name, value in policy.state_dict().items()
        if "lora_" not in name and ".original_module." not in name
    }
    for name in initial.keys() - expected_full:
        torch.testing.assert_close(current[name], initial[name], rtol=0, atol=0, msg=name)
    checkpoint = tmp_path / "checkpoint"
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/pi05"),
        policy=cfg,
        peft=PeftConfig(r=4, lora_alpha=8),
        batch_size=1,
    )
    train_cfg.parallelism.resolve(1)
    save_checkpoint(checkpoint, 3, train_cfg, policy, optimizer, preprocessor=pre, postprocessor=post)
    adapter = checkpoint / "pretrained_model"
    assert (adapter / "adapter_model.safetensors").is_file()
    assert (checkpoint / "training_state/optimizer_state.safetensors").is_file()

    loaded_cfg = PreTrainedConfig.from_pretrained(adapter)
    assert loaded_cfg.use_peft and loaded_cfg.peft_mode == "vlm_lora_expert_full"
    loaded_cfg.pretrained_path = adapter
    reloaded = make_policy(loaded_cfg, ds_meta=dataset_metadata(cfg))
    noise = torch.randn(1, 4, 4, device=device)
    expected = post(policy.predict_action_chunk(pre(training_batch()), noise=noise.clone()))
    loaded_pre, loaded_post = make_pre_post_processors(loaded_cfg, pretrained_path=adapter)
    actual = loaded_post(reloaded.predict_action_chunk(loaded_pre(training_batch()), noise=noise.clone()))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # Resume the optimizer as well: the next update must match uninterrupted
    # training, including full-expert moments and VLM adapter moments.
    restored_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.5)
    train_cfg.checkpoint_path = checkpoint
    assert resume_before_prepare(train_cfg) == 3
    accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    resume_after_prepare(train_cfg, accelerator, reloaded, restored_optimizer, None)
    before, restored = optimizer.state_dict(), restored_optimizer.state_dict()
    assert before["param_groups"] == restored["param_groups"]
    for index, state in before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored["state"][index][name], rtol=0, atol=0)
    train_steps(policy, steps=1, batch=pre(training_batch()), optimizer=optimizer)
    train_steps(reloaded, steps=1, batch=loaded_pre(training_batch()), optimizer=restored_optimizer)
    restored_weights = reloaded.state_dict()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored_weights[name], rtol=0, atol=0, msg=name)

    exported = merge_adapter(str(adapter), tmp_path / "merged", device=device)
    merged = PI05Policy.from_pretrained(exported)
    merged_pre, merged_post = make_pre_post_processors(merged.config, pretrained_path=exported)
    actual = merged_post(merged.predict_action_chunk(merged_pre(training_batch()), noise=noise.clone()))
    torch.testing.assert_close(
        actual, expected, rtol=2e-4 if dtype == "float32" else 0.02, atol=2e-5 if dtype == "float32" else 0.02
    )
    assert merged.config.backend == backend
    assert not merged.config.use_peft and merged.config.pretrained_path is None
    state = merged.state_dict()
    assert not any("lora_" in name or "modules_to_save" in name for name in state)
    assert not torch.equal(state["model.action_out_proj.weight"], initial["model.action_out_proj.weight"])
    if backend == "rlinf":
        expert_mlp = "model.llm.layers.0.mlps.1.w_gating"
        frozen_embedding = "model.llm.embedder.embedding.weight"
        vision_lora = "model.img.encoder.layers.0.attn.lora_B.weight"
        language_lora = "model.llm.layers.0.attn.k_proj.0.lora_B.weight"
        language_mlp_lora = [
            f"model.llm.layers.0.mlps.0.lora_B{suffix}.weight" for suffix in ("", "_up", "_down")
        ]
        merged_mlp_weights = {
            "model.llm.layers.0.mlps.0.w_gating": (0, 1),
            "model.llm.layers.0.mlps.0.w_linear": (None,),
        }
    else:
        expert_mlp = "model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight"
        frozen_embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        vision_lora = "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_B.weight"
        language_lora = "language_model.layers.0.self_attn.k_proj.lora_B.weight"
        language_mlp_lora = [f"{target}.lora_B.weight" for target in language_mlp_targets[:3]]
        merged_mlp_weights = {target + ".weight": (None,) for target in language_mlp_targets[:3]}
    assert not torch.equal(state[expert_mlp], initial[expert_mlp])
    torch.testing.assert_close(state[frozen_embedding], initial[frozen_embedding], rtol=0, atol=0)
    saved = load_file(adapter / "adapter_model.safetensors")
    vision_mlp_lora = [f"{target}.lora_B.weight" for target in vision_mlp_targets]
    for suffix in (vision_lora, language_lora, *language_mlp_lora, *vision_mlp_lora):
        assert any(name.endswith(suffix) and value.any() for name, value in saved.items()), suffix
    merged_mlp_weights.update({target + ".weight": (None,) for target in vision_mlp_targets})
    for name, indices in merged_mlp_weights.items():
        for index in indices:
            actual = state[name] if index is None else state[name][index]
            original = initial[name] if index is None else initial[name][index]
            assert not torch.equal(actual, original), name
    with pytest.raises(FileExistsError):
        merge_adapter(str(adapter), exported)


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
@pytest.mark.parametrize(
    "language,vision,device",
    [
        (language, vision, "cpu")
        for language in ("none", "attention", "attention_mlp")
        for vision in ("none", "attention", "attention_mlp")
    ]
    + [
        ("attention_mlp", "none", "cuda"),
        ("none", "attention_mlp", "cuda"),
        ("attention", "attention_mlp", "cuda"),
        ("none", "none", "cuda"),
    ],
)
@pytest.mark.filterwarnings("error:Found missing adapter keys")
def test_language_and_vision_lora_are_independent(tiny_networks, tmp_path, backend, language, vision, device):
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.constants import DUMMY_TARGET_MODULES

    from lerobot.policies.factory import make_policy

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = "bfloat16" if device == "cuda" else "float32"
    base = tmp_path / "base"
    PI05Policy(replace(config(), backend="lerobot")).save_pretrained(base)
    cfg = replace(
        config(),
        backend=backend,
        dtype=dtype,
        device=device,
        pretrained_path=base,
        peft_mode="vlm_lora_expert_full",
        peft_language_lora=language,
        peft_vision_lora=vision,
        freeze_vision_encoder=vision == "none",
        train_expert_only=language == vision == "none",
        action_feature_names=["joint_1", "joint_2", "gripper"],
    )
    policy = PI05Policy.from_pretrained(base, config=cfg)
    initial = {name: value.clone() for name, value in policy.state_dict().items()}
    expected = set()
    for component, selection in (("language", language), ("vision", vision)):
        targets = component_lora_targets(backend)
        if selection != "none":
            expected |= targets[f"{component}_attention"]
        if selection == "attention_mlp":
            expected |= targets[f"{component}_mlp"]
    policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    actual = {
        name
        for name, module in policy.get_base_model().named_modules()
        if isinstance(module, LoraLayer) and ".base_layer." not in name
    }
    assert actual == expected
    if not expected:
        assert policy.peft_config["default"].target_modules == DUMMY_TARGET_MODULES
    full = {
        name.removeprefix("base_model.model.").replace(".modules_to_save.default.", ".")
        for name, value in policy.named_parameters()
        if value.requires_grad and "lora_" not in name
    }
    assert "model.action_out_proj.weight" in full
    for name in full:
        assert "paligemma.model." not in name and "model.img." not in name, name
        if backend == "rlinf":
            assert not re.search(
                r"(attn\.[qkvo]_proj|mlps|pre_attention_norms|pre_ffw_norms|final_norms)\.0\.", name
            )
    batch = {key: value.to(device) for key, value in training_batch().items()}
    train_steps(policy, batch=batch)
    current = {
        name.removeprefix("base_model.model.")
        .replace(".base_layer.", ".")
        .replace(".modules_to_save.default.", "."): value
        for name, value in policy.state_dict().items()
        if "lora_" not in name and ".original_module." not in name
    }
    for name in initial.keys() - full:
        torch.testing.assert_close(current[name], initial[name], rtol=0, atol=0, msg=name)
    assert not torch.equal(current["model.action_out_proj.weight"], initial["model.action_out_proj.weight"])
    adapter = tmp_path / "adapter"
    policy.save_pretrained(adapter)
    cfg.save_pretrained(adapter)  # The training checkpoint writer saves this alongside the PEFT weights.
    saved = load_file(adapter / "adapter_model.safetensors")
    for component, selection in (("language", language), ("vision", vision)):
        roots = (
            {"language": "model.llm.", "vision": "model.img."}
            if backend == "rlinf"
            else {"language": ".language_model.", "vision": ".vision_tower."}
        )
        updates = [value for name, value in saved.items() if roots[component] in name and ".lora_B" in name]
        assert bool(updates) == (selection != "none")
        if updates:
            assert any(value.any() for value in updates), component
    loaded_cfg = PreTrainedConfig.from_pretrained(adapter)
    assert (loaded_cfg.peft_language_lora, loaded_cfg.peft_vision_lora) == (language, vision)
    loaded_cfg.pretrained_path = adapter
    reloaded = make_policy(loaded_cfg, ds_meta=dataset_metadata(cfg))
    noise = torch.randn(1, 4, 4, device=device)
    expected_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    torch.testing.assert_close(
        reloaded.predict_action_chunk(batch, noise=noise.clone()), expected_actions, rtol=0, atol=0
    )
    merged = reloaded.merge_and_unload(safe_merge=True)
    tolerance = 0.02 if dtype == "bfloat16" else 1e-4
    torch.testing.assert_close(
        merged.predict_action_chunk(batch, noise=noise.clone()),
        expected_actions,
        rtol=tolerance,
        atol=tolerance,
    )
    assert not any("lora_" in name or "modules_to_save" in name for name in merged.state_dict())


def test_independent_lora_scopes_parse_from_cli():
    import draccus

    parsed = draccus.parse(
        PI05Config,
        args=[
            "--device=cpu",
            "--peft_mode=vlm_lora_expert_full",
            "--peft_language_lora=attention_mlp",
            "--peft_vision_lora=none",
        ],
    )
    assert parsed.peft_language_lora == "attention_mlp" and parsed.peft_vision_lora == "none"


@pytest.mark.parametrize("field", ["peft_language_lora", "peft_vision_lora"])
def test_invalid_lora_scope_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        PI05Config(**{field: "unknown"})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_gated_mlp_lora_matches_dense_reference_and_can_be_disabled(dtype):
    from peft import LoraConfig

    from lerobot.policies.pi05.rlinf.gemma import FeedForward
    from lerobot.policies.pi05.rlinf.peft import GatedFeedForward

    torch.manual_seed(11)
    base = FeedForward(8, 12).to(dtype=dtype)
    original = deepcopy(base)
    cfg = LoraConfig(r=3, lora_alpha=6, use_rslora=True, lora_dropout=0.1, init_lora_weights="gaussian")
    layer = GatedFeedForward(base, "default", cfg, r=3, lora_alpha=6).eval()
    # Exercise nonzero updates on all three projections, including both halves
    # of the native packed gating tensor, using an independent dense reference.
    reference = deepcopy(base)
    with torch.no_grad():
        for projection, weight in (
            ("gate", reference.w_gating[0]),
            ("up", reference.w_gating[1]),
            ("down", reference.w_linear),
        ):
            suffix = "" if projection == "gate" else f"_{projection}"
            a = getattr(layer, "lora_A" + suffix)["default"].weight
            b = getattr(layer, "lora_B" + suffix)["default"].weight
            torch.nn.init.normal_(b, std=0.03)
            weight.add_((a.T @ b.T) * (6 / math.sqrt(3)))
    inputs = torch.randn(2, 5, 8, dtype=dtype)
    expected = reference(inputs)
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(layer(inputs), expected, rtol=tolerance, atol=tolerance)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(base.state_dict()[name], value, rtol=0, atol=0)
    layer.enable_adapters(False)
    torch.testing.assert_close(layer(inputs), original(inputs), rtol=0, atol=0)
    layer.enable_adapters(True)
    layer.merge(safe_merge=True)
    torch.testing.assert_close(layer(inputs), expected, rtol=tolerance, atol=tolerance)
    if dtype == torch.float32:
        layer.unmerge()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(base.state_dict()[name], value, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(layer(inputs), expected, rtol=tolerance, atol=tolerance)


def test_native_gated_mlp_safe_merge_rejects_nonfinite_without_mutating_base():
    from peft import LoraConfig

    from lerobot.policies.pi05.rlinf.gemma import FeedForward
    from lerobot.policies.pi05.rlinf.peft import GatedFeedForward

    base = FeedForward(8, 12)
    original = {name: value.clone() for name, value in base.state_dict().items()}
    layer = GatedFeedForward(base, "default", LoraConfig(r=2), r=2)
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(0.1)
        layer.lora_B_down["default"].weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite gated MLP"):
        layer.merge(safe_merge=True)
    assert not layer.merged
    for name, value in original.items():
        torch.testing.assert_close(base.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("flag", ["train_expert_only", "freeze_vision_encoder"])
def test_mixed_profile_rejects_flags_that_freeze_vlm_adapters(tiny_networks, flag):
    policy = PI05Policy(config(peft_mode="vlm_lora_expert_full", pretrained_path="base", **{flag: True}))
    with pytest.raises(ValueError, match=f"requires {flag}=false"):
        policy.wrap_with_peft()


def test_invalid_peft_mode_is_rejected():
    with pytest.raises(ValueError, match="peft_mode"):
        PI05Config(peft_mode="unknown")


def test_adapter_keeps_the_base_checkpoint_revision(tiny_networks, tmp_path):
    source = PI05Policy(config(pretrained_path=tmp_path, pretrained_revision="pinned-base-sha"))
    source.save_pretrained(tmp_path)
    policy = source.wrap_with_peft(peft_cli_overrides={"r": 4})
    assert policy.peft_config["default"].revision == "pinned-base-sha"


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
@pytest.mark.parametrize("missing", ["lora", "expert", "action"])
@pytest.mark.parametrize("entrypoint", ["factory", "merge"])
def test_incomplete_adapter_is_rejected(tiny_networks, tmp_path, backend, missing, entrypoint):
    from examples.tutorial.pi05.merge_lora import merge_adapter
    from lerobot.policies.factory import make_policy

    base, adapter = tmp_path / "base", tmp_path / "adapter"
    PI05Policy(replace(config(), backend="lerobot")).save_pretrained(base)
    cfg = replace(
        config(),
        backend=backend,
        pretrained_path=base,
        peft_mode="vlm_lora_expert_full",
        action_feature_names=["joint_1", "joint_2", "gripper"],
    )
    policy = PI05Policy.from_pretrained(base, config=cfg).wrap_with_peft(peft_cli_overrides={"r": 4})
    policy.save_pretrained(adapter)
    cfg.save_pretrained(adapter)
    pre, post = processors(cfg)
    pre.save_pretrained(adapter)
    post.save_pretrained(adapter)
    saved = load_file(adapter / "adapter_model.safetensors")
    expert = "gemma_expert.model.layers.0.mlp" if backend == "lerobot" else "llm.layers.0.mlps.1"
    marker = {"lora": "lora_B", "expert": expert, "action": "action_out_proj"}[missing]
    removed = next(name for name in saved if marker in name)
    saved.pop(removed)
    save_file(saved, adapter / "adapter_model.safetensors")
    with pytest.raises(RuntimeError, match="Incomplete PI05 adapter"):
        if entrypoint == "factory":
            loaded_cfg = PreTrainedConfig.from_pretrained(adapter)
            loaded_cfg.pretrained_path = adapter
            make_policy(loaded_cfg, ds_meta=dataset_metadata(cfg))
        else:
            merge_adapter(str(adapter), tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
