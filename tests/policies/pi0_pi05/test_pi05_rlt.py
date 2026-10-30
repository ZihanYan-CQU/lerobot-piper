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

"""RLT Stage 1 lifecycle with real miniature VLM, action expert, and RLT layers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("peft")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.policies.pi05.rlinf.rlt import RLTTokenTransformer  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from tests.policies.pi0_pi05.test_pi05_peft import (  # noqa: E402
    dataset_metadata,
    processors,
    training_batch,
)
from tests.policies.pi0_pi05.test_pi05_rlinf import (  # noqa: E402
    config,
    tiny_networks as tiny_networks,
)


def rlt_config(**kwargs):
    return replace(
        config(),
        **(
            {
                "use_rlt": True,
                "rlt_embed_dim": 16,
                "rlt_num_heads": 4,
                "rlt_num_layers": 1,
                "rlt_mlp_ratio": 2,
                "rlt_prefix_seq_len": 512,
            }
            | kwargs
        ),
    )


def train(policy, batch, *, optimizer=None, steps=2, amp=False):
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for _ in range(steps):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(41)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = policy(batch)
        assert torch.isfinite(loss)
        assert "rlt_loss" in metrics
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        optimizer.step()
    return optimizer


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "lerobot"},
        {"rlt_training_mode": "other"},
        {"rlt_embed_dim": 15},
        {"rlt_num_heads": 3},
        {"rlt_prefix_seq_len": 0},
        {"rlt_num_layers": 0},
        {"rlt_alpha": -1},
        {"rlt_alpha": float("nan")},
        {"rlt_mlp_ratio": float("inf")},
    ],
)
def test_rlt_configuration_rejects_invalid_settings(overrides):
    with pytest.raises(ValueError):
        rlt_config(**overrides)


def test_rlt_cli_fields_roundtrip(tmp_path):
    import draccus

    cfg = draccus.parse(
        PI05Config,
        args=[
            "--backend=rlinf",
            "--device=cpu",
            "--use_rlt=true",
            "--rlt_training_mode=frozen_vla",
            "--rlt_embed_dim=32",
            "--rlt_num_layers=1",
            "--rlt_num_heads=4",
            "--rlt_image_only=false",
            "--rlt_use_mask=true",
        ],
    )
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.use_rlt and restored.rlt_training_mode == "frozen_vla"
    assert restored.rlt_embed_dim == 32 and restored.rlt_use_mask


def test_rlt_mask_reconstruction_causality_and_detached_targets():
    torch.manual_seed(17)
    model = RLTTokenTransformer(
        input_dim=8, embed_dim=8, prefix_seq_len=8, num_layers=1, num_heads=2, mlp_ratio=2
    )
    features = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    changed = features.detach().clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 100
    torch.testing.assert_close(model(features, mask, mode="encode"), model(changed, mask, mode="encode"))
    loss = model(features, mask)
    assert loss.shape == (2,)
    individual = torch.stack([model(features[i : i + 1], mask[i : i + 1]).squeeze(0) for i in range(2)])
    torch.testing.assert_close(loss, individual)
    torch.testing.assert_close(loss, model(changed, mask))
    loss.mean().backward()
    assert features.grad is None  # The reconstruction objective never updates the VLA.
    assert model.encoder.rl_token_embed.grad.abs().sum() > 0
    assert model.decoder.output_proj.weight.grad.abs().sum() > 0
    targets = features.detach()
    z = model.encode(targets)
    original = model.decode(z, targets)
    future = targets.clone()
    future[:, 3:] += 100
    torch.testing.assert_close(model.decode(z, future)[:, :4], original[:, :4])
    empty = model(features.detach(), torch.zeros_like(mask))
    torch.testing.assert_close(empty, torch.zeros(2), rtol=0, atol=0)


def test_rlt_rejects_overlong_prefix_and_invalid_mask():
    model = RLTTokenTransformer(
        input_dim=8, embed_dim=8, prefix_seq_len=3, num_layers=1, num_heads=2, mlp_ratio=2
    )
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.randn(1, 4, 8))
    with pytest.raises(ValueError, match="mask"):
        model(torch.randn(1, 3, 8), torch.ones(1, 3))


@pytest.mark.parametrize("masked", [False, True])
def test_rlt_checkpointing_preserves_dropout_rng_and_detached_targets(masked):
    settings = {
        "input_dim": 8,
        "embed_dim": 8,
        "prefix_seq_len": 8,
        "num_layers": 2,
        "num_heads": 2,
        "mlp_ratio": 2,
    }
    reference = RLTTokenTransformer(**settings, dropout_rate=0.2)
    checkpointed = RLTTokenTransformer(**settings, dropout_rate=0.2)
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.encoder.gradient_checkpointing = True
    checkpointed.decoder.gradient_checkpointing = True
    features = torch.randn(2, 7, 8, requires_grad=True)
    mask = torch.arange(7)[None, :] < torch.tensor([[5], [0]]) if masked else None
    losses = []
    for model in (reference, checkpointed):
        torch.manual_seed(53)
        loss = model(features, mask).mean()
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1])
    assert features.grad is None
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(checkpointed.named_parameters())[name].grad, msg=name)


@pytest.mark.parametrize("mode", ["frozen_vla", "joint"])
def test_policy_checkpointing_recomputes_rlt_layers_and_saves_activations(tiny_networks, mode):
    cfg = rlt_config(rlt_training_mode=mode, rlt_num_layers=2)
    reference = PI05Policy(cfg)
    checkpointed = PI05Policy(replace(cfg, gradient_checkpointing=True))
    checkpointed.load_state_dict(reference.state_dict())
    losses, saved_bytes = [], []
    for model, enabled in ((reference, False), (checkpointed, True)):
        model.train()
        parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
        saved_storage = {}
        calls = {}

        def pack(tensor, parameter_storage=parameter_storage, saved_storage=saved_storage):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_storage:
                saved_storage[storage.data_ptr()] = storage.nbytes()
            return tensor

        def count(module, args, calls=calls):
            calls[module] = calls.get(module, 0) + 1

        layers = [*model.model.rlt_module.encoder.layers, *model.model.rlt_module.decoder.layers]
        handles = [layer.register_forward_pre_hook(count) for layer in layers]
        try:
            torch.manual_seed(47)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                loss, _ = model(training_batch())
            saved_bytes.append(sum(saved_storage.values()))
            losses.append(loss.detach())
            loss.backward()
            assert all(calls[layer] == (2 if enabled else 1) for layer in layers)
        finally:
            for handle in handles:
                handle.remove()
    torch.testing.assert_close(losses[0], losses[1])
    assert 0 < saved_bytes[1] < saved_bytes[0]
    for name, parameter in reference.named_parameters():
        actual = dict(checkpointed.named_parameters())[name]
        assert (parameter.grad is None) == (actual.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, actual.grad, msg=name)


@pytest.mark.parametrize("peft", [False, True])
def test_policy_checkpointing_toggle_includes_peft_rlt_copies(tiny_networks, tmp_path, peft):
    from lerobot.policies.pi05.rlinf.rlt import RLTTokenDecoder, RLTTokenEncoder

    policy = PI05Policy(
        rlt_config(
            pretrained_path=tmp_path,
            rlt_training_mode="joint" if peft else "frozen_vla",
            peft_mode="vlm_lora_expert_full" if peft else "default",
            gradient_checkpointing=True,
        )
    )
    if peft:
        policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    core = policy.get_base_model().model if peft else policy.model
    modules = [m for m in core.rlt_module.modules() if isinstance(m, RLTTokenEncoder | RLTTokenDecoder)]
    assert len(modules) == (4 if peft else 2)
    keys = set(policy.state_dict())
    for enabled in (False, True):
        if enabled:
            core.gradient_checkpointing_enable()
        else:
            core.gradient_checkpointing_disable()
        assert core.gradient_checkpointing_enabled == enabled
        assert core.llm.gradient_checkpointing == enabled
        assert core.img.encoder.gradient_checkpointing == enabled
        assert all(module.gradient_checkpointing == enabled for module in modules)
    assert set(policy.state_dict()) == keys


@pytest.mark.parametrize("inference_only", [False, True])
def test_rlt_checkpointing_is_bypassed_during_eval_and_no_grad(monkeypatch, inference_only):
    from lerobot.policies.pi05.rlinf import rlt

    model = RLTTokenTransformer(
        input_dim=8, embed_dim=8, prefix_seq_len=8, num_layers=1, num_heads=2, inference_only=inference_only
    )
    model.encoder.gradient_checkpointing = True
    if model.decoder is not None:
        model.decoder.gradient_checkpointing = True

    def unexpected_checkpoint(*args, **kwargs):
        pytest.fail("Checkpointing must only run during training with gradients enabled")

    monkeypatch.setattr(rlt, "checkpoint", unexpected_checkpoint)
    features = torch.randn(2, 5, 8)
    mode = "encode" if inference_only else "loss"
    model.eval()
    expected = model(features, mode=mode).detach()
    model.train()
    with torch.no_grad():
        torch.testing.assert_close(model(features, mode=mode), expected)


@pytest.mark.parametrize("image_only", [False, True])
def test_frozen_vla_trains_only_rlt_without_actions(tiny_networks, image_only):
    cfg = rlt_config(rlt_image_only=image_only, mask_action_padding_loss=True)
    policy = PI05Policy(cfg)
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    batch = training_batch()
    del batch[ACTION]
    train(policy, batch)
    updated = []
    for name, value in policy.state_dict().items():
        if not name.startswith("model.rlt_module."):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        elif not torch.equal(value, before[name]):
            updated.append(name)
    assert any("encoder" in name for name in updated) and any("decoder" in name for name in updated)
    assert not policy.model.img.training and not policy.model.llm.training
    for name, parameter in policy.named_parameters():
        assert parameter.requires_grad == name.startswith("model.rlt_module.")
        if not parameter.requires_grad:
            assert parameter.grad is None
    features = policy.extract_rlt_features(batch)
    assert features.shape == (1, 16) and not features.requires_grad


def test_joint_loss_preserves_rtc_padding_and_per_sample_weights(tiny_networks):
    from lerobot.policies.pi05.modeling_pi05 import _reduce_training_rtc_loss

    policy = PI05Policy(
        rlt_config(
            rlt_training_mode="joint", rlt_alpha=0.3, rtc_training_max_delay=2, mask_action_padding_loss=True
        )
    )
    batch = {name: value.repeat(2, *([1] * (value.ndim - 1))) for name, value in training_batch().items()}
    batch["action_is_pad"] = torch.tensor([[False, False, False, True], [False, False, True, True]])
    policy.train()
    captured = {}

    def capture(module, args, kwargs, output):
        captured.update(prefix_mask=kwargs["prefix_mask"], outputs=output)

    handle = policy.model.register_forward_hook(capture, with_kwargs=True)
    # PI05Policy's direct model.forward call bypasses normal hooks; capture it explicitly.
    original = policy.model.forward

    def traced(*args, **kwargs):
        output = original(*args, **kwargs)
        capture(policy.model, args, kwargs, output)
        return output

    policy.model.forward = traced
    try:
        torch.manual_seed(23)
        vector, metrics = policy(batch, reduction="none")
        expected_vla = _reduce_training_rtc_loss(
            captured["outputs"]["vla_loss"][:, :, :3],
            captured["prefix_mask"],
            reduction="none",
            action_is_pad=batch["action_is_pad"],
        )
        torch.testing.assert_close(vector, captured["outputs"]["rlt_loss"] + 0.3 * expected_vla)
        torch.manual_seed(23)
        scalar, _ = policy(batch)
        # Preserve LeRobot's global valid-action reduction in mean mode; it
        # intentionally differs from equally weighting differently padded samples.
        global_vla = _reduce_training_rtc_loss(
            captured["outputs"]["vla_loss"][:, :, :3],
            captured["prefix_mask"],
            reduction="mean",
            action_is_pad=batch["action_is_pad"],
        )
        torch.testing.assert_close(scalar, captured["outputs"]["rlt_loss"].mean() + 0.3 * global_vla)
        (vector * torch.tensor([0.25, 0.75])).sum().backward()
        assert policy.model.action_out_proj.weight.grad.abs().sum() > 0
        assert policy.model.rlt_module.encoder.rl_token_embed.grad.abs().sum() > 0
        assert metrics["rlt_loss"] > 0 and metrics["vla_loss"] > 0
    finally:
        handle.remove()
        policy.model.forward = original


@pytest.mark.parametrize("peft", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stage1_checkpoint_resume_factory_and_inference_export(tiny_networks, tmp_path, peft, device):
    from examples.tutorial.pi05.export_rlt import export_rlt
    from lerobot.common.train_utils import resume_after_prepare, resume_before_prepare, save_checkpoint
    from lerobot.configs.default import DatasetConfig, PeftConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    base = tmp_path / "base"
    PI05Policy(config()).save_pretrained(base)
    cfg = rlt_config(
        device=device,
        gradient_checkpointing=True,
        pretrained_path=base,
        rlt_training_mode="joint" if peft else "frozen_vla",
        peft_mode="vlm_lora_expert_full" if peft else "default",
        use_relative_actions=True,
        action_feature_names=["joint_1", "joint_2", "gripper"],
    )
    policy = PI05Policy.from_pretrained(base, config=cfg)
    if peft:
        policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
        assert "model.rlt_module" in policy.peft_config["default"].modules_to_save
    pre, post = processors(cfg)
    batch = pre(training_batch())
    noise = torch.randn(1, 4, 4, device=device)
    before_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    optimizer = train(policy, batch, amp=device == "cuda")
    expected_z = policy.extract_rlt_features(batch)
    expected_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    if not peft:
        torch.testing.assert_close(before_actions, expected_actions, rtol=0, atol=0)
    checkpoint = tmp_path / "checkpoint"
    training = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/rlt"),
        policy=cfg,
        peft=PeftConfig(r=4, lora_alpha=8) if peft else None,
        batch_size=1,
    )
    training.parallelism.resolve(1)
    save_checkpoint(checkpoint, 2, training, policy, optimizer, preprocessor=pre, postprocessor=post)
    saved = checkpoint / "pretrained_model"
    loaded_cfg = PreTrainedConfig.from_pretrained(saved)
    loaded_cfg.pretrained_path = saved
    restored = make_policy(loaded_cfg, ds_meta=dataset_metadata(cfg))
    restored_pre, _ = make_pre_post_processors(loaded_cfg, pretrained_path=saved)
    restored_batch = restored_pre(training_batch())
    torch.testing.assert_close(restored.extract_rlt_features(restored_batch), expected_z, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.predict_action_chunk(restored_batch, noise=noise.clone()), expected_actions, rtol=0, atol=0
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
    training.checkpoint_path = checkpoint
    assert resume_before_prepare(training) == 2
    resume_after_prepare(
        training, SimpleNamespace(unwrap_model=lambda model: model), restored, restored_optimizer, None
    )
    train(policy, batch, optimizer=optimizer, steps=1, amp=device == "cuda")
    train(restored, restored_batch, optimizer=restored_optimizer, steps=1, amp=device == "cuda")
    for name, value in policy.state_dict().items():
        if ".rlt_module.original_module." in name:
            # PEFT saves the active trained copy. The unused frozen original
            # was randomly initialized on a base that did not yet contain RLT.
            continue
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0, msg=name)
    # Export from the saved checkpoint, rather than the additional resumed update.
    exported = export_rlt(saved, tmp_path / "inference", device=device)
    inference = PI05Policy.from_pretrained(exported)
    assert inference.config.rlt_inference_only and not inference.config.use_peft
    assert inference.model.rlt_module.decoder is None
    assert not any("rlt_module.decoder" in name for name in inference.state_dict())
    inference_pre, _ = make_pre_post_processors(inference.config, pretrained_path=exported)
    inference_batch = inference_pre(training_batch())
    torch.testing.assert_close(
        inference.extract_rlt_features(inference_batch), expected_z, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        inference.predict_action_chunk(inference_batch, noise=noise.clone()),
        expected_actions,
        rtol=2e-4,
        atol=2e-5,
    )
    with pytest.raises(ValueError, match="encoder-only"):
        inference(inference_batch)
    assert not (exported / "adapter_config.json").exists()


@pytest.mark.parametrize("remove_all", [False, True])
def test_damaged_rlt_checkpoint_cannot_reinitialize(tiny_networks, tmp_path, remove_all):
    policy = PI05Policy(rlt_config())
    policy.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    names = [name for name in state if name.startswith("model.rlt_module.")]
    for name in names if remove_all else names[:1]:
        del state[name]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="RLT checkpoint"):
        PI05Policy.from_pretrained(tmp_path, strict=False)


def test_frozen_mode_rejects_new_lora_and_missing_base_weights(tiny_networks, tmp_path):
    policy = PI05Policy(rlt_config(pretrained_path=tmp_path))
    with pytest.raises(ValueError, match="Merge your trained adapter"):
        policy.wrap_with_peft(peft_cli_overrides={"r": 4})
    PI05Policy(config()).save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    del state["model.action_out_proj.weight"]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Incomplete PI05 base"):
        PI05Policy.from_pretrained(tmp_path, config=rlt_config())


def test_bare_native_base_requires_explicit_rlt_initialization(tiny_networks, tmp_path):
    raw = tmp_path / "native.safetensors"
    save_file(PI05Policy(config()).state_dict(), raw)
    with pytest.raises(ValueError, match="initialize_rlt=True"):
        PI05Policy.from_pretrained(raw, config=rlt_config())
    initialized = PI05Policy.from_pretrained(raw, config=rlt_config(), initialize_rlt=True)
    assert initialized.extract_rlt_features(training_batch()).shape == (1, 16)


@pytest.mark.parametrize("mode", ["frozen_vla", "joint"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_training_and_feature_inference_use_same_final_prefix(tiny_networks, mode, use_mask):
    policy = PI05Policy(rlt_config(rlt_training_mode=mode, rlt_use_mask=use_mask))
    captured = []
    handle = policy.model.rlt_module.register_forward_pre_hook(
        lambda module, args: captured.append(
            (args[0].detach().clone(), args[1].clone() if args[1] is not None else None)
        )
    )
    try:
        policy.train()
        policy(training_batch())
        policy.extract_rlt_features(training_batch())
    finally:
        handle.remove()
    training_prefix, training_mask = captured[0]
    inference_prefix, inference_mask = captured[1]
    if not use_mask:
        torch.testing.assert_close(training_prefix, inference_prefix, rtol=1e-5, atol=1e-6)
        return
    torch.testing.assert_close(training_mask, inference_mask)
    # Fully masked padding queries can differ with total attention length;
    # they are excluded from both readout attention and reconstruction loss.
    torch.testing.assert_close(
        training_prefix[training_mask], inference_prefix[inference_mask], rtol=1e-5, atol=1e-6
    )
    with torch.no_grad():
        torch.testing.assert_close(
            policy.model.rlt_module(training_prefix, training_mask, mode="encode"),
            policy.model.rlt_module(inference_prefix, inference_mask, mode="encode"),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("remove_all", [False, True])
def test_joint_adapter_rejects_missing_rlt_weights(tiny_networks, tmp_path, remove_all):
    from lerobot.policies.factory import make_policy

    base = tmp_path / "base"
    PI05Policy(config()).save_pretrained(base)
    cfg = rlt_config(pretrained_path=base, rlt_training_mode="joint", peft_mode="vlm_lora_expert_full")
    policy = PI05Policy.from_pretrained(base, config=cfg)
    policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    adapter = tmp_path / "adapter"
    policy.save_pretrained(adapter)
    # The standard training persistence layer saves this config beside the adapter.
    cfg.save_pretrained(adapter)
    state = load_file(adapter / "adapter_model.safetensors")
    names = [name for name in state if "rlt_module" in name]
    assert names
    for name in names if remove_all else names[:1]:
        del state[name]
    save_file(state, adapter / "adapter_model.safetensors")
    loaded_cfg = PreTrainedConfig.from_pretrained(adapter)
    loaded_cfg.pretrained_path = adapter
    with pytest.raises(RuntimeError, match="Incomplete PI05 adapter"):
        make_policy(loaded_cfg, ds_meta=dataset_metadata(cfg))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("peft", [False, True])
def test_cuda_fp32_amp_stage1_training_and_features(tiny_networks, tmp_path, peft):
    cfg = rlt_config(
        device="cuda",
        gradient_checkpointing=True,
        rlt_training_mode="joint" if peft else "frozen_vla",
        peft_mode="vlm_lora_expert_full" if peft else "default",
        pretrained_path=tmp_path,
    )
    policy = PI05Policy(cfg)
    if peft:
        policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    batch = {name: value.cuda() for name, value in training_batch().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    policy.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(p.dtype == torch.float32 for p in policy.parameters() if p.requires_grad)
    features = policy.extract_rlt_features(batch)
    assert features.shape == (1, 16) and torch.isfinite(features).all()
