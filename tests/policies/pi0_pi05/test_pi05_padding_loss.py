# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, _reduce_training_rtc_loss  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor import tokenizer_processor  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def test_padding_is_excluded_from_loss_denominator_and_gradient():
    losses = torch.tensor([[[2.0, 4.0], [100.0, 100.0], [200.0, 200.0]]], requires_grad=True)
    padding = torch.tensor([[False, True, True]])

    loss = _reduce_training_rtc_loss(losses, None, "mean", action_is_pad=padding)
    loss.backward()

    assert loss.item() == pytest.approx(3.0)
    torch.testing.assert_close(losses.grad, torch.tensor([[[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]]))


def test_rtc_and_padding_masks_intersect_and_all_masked_samples_are_finite():
    losses = torch.tensor([[[100.0], [2.0], [4.0], [100.0]], [[10.0], [20.0], [30.0], [40.0]]])
    losses.requires_grad_()
    prefix = torch.tensor([[True, False, False, False], [True, True, False, False]])
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])

    mean = _reduce_training_rtc_loss(losses, prefix, "mean", padding)
    per_sample = _reduce_training_rtc_loss(losses, prefix, "none", padding)
    per_dim = _reduce_training_rtc_loss(losses, prefix, "per_dim", padding)
    mean.backward()

    assert mean.item() == pytest.approx(3.0)
    torch.testing.assert_close(per_sample, torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(per_dim, torch.tensor([3.0]))
    torch.testing.assert_close(losses.grad, torch.tensor([[[0.0], [0.5], [0.5], [0.0]], [[0.0]] * 4]))

    empty = _reduce_training_rtc_loss(losses, prefix, "mean", torch.ones_like(padding))
    assert empty.item() == 0.0
    assert torch.isfinite(empty)


@pytest.mark.parametrize("reduction,dims", [("mean", (0, 1, 2)), ("none", (1, 2)), ("per_dim", (0, 1))])
def test_unmasked_loss_preserves_previous_reductions(reduction, dims):
    losses = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    expected = losses.mean(dim=dims)
    torch.testing.assert_close(_reduce_training_rtc_loss(losses, None, reduction), expected)
    torch.testing.assert_close(
        _reduce_training_rtc_loss(losses, None, reduction, torch.zeros(2, 4, dtype=torch.bool)), expected
    )


@pytest.mark.parametrize(
    "padding,error",
    [(torch.zeros(2, 4, 1, dtype=torch.bool), ValueError), (torch.zeros(2, 4), TypeError)],
)
def test_padding_mask_rejects_broadcastable_shapes_and_non_boolean_flags(padding, error):
    with pytest.raises(error, match="action_is_pad"):
        _reduce_training_rtc_loss(torch.zeros(2, 4, 3), None, "mean", padding)


class _LossOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The third action dimension is model padding, independently of time padding.
        self.prediction = nn.Parameter(
            torch.tensor(
                [
                    [[1.0, 2.0, 99.0], [9.0, 9.0, 99.0], [9.0, 9.0, 99.0]],
                    [[3.0, 4.0, 99.0], [5.0, 6.0, 99.0], [9.0, 9.0, 99.0]],
                ]
            )
        )

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward(self, *args, **kwargs):
        return self.prediction.square()


def _policy_and_batch(*, masking=True):
    # Exercise the real policy forward/reduction without constructing a VLA backbone.
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = PI05Config(
        device="cpu",
        chunk_size=3,
        n_action_steps=3,
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        mask_action_padding_loss=masking,
    )
    policy.model = _LossOnlyModel()
    policy._preprocess_images = lambda batch: ([], [])
    policy._prepare_memory_states = lambda batch: (None, None)
    policy.prepare_action = lambda batch: batch[ACTION]
    batch = {
        ACTION: torch.zeros(2, 3, 3),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True, True], [False, False, True]]),
    }
    return policy, batch


def test_policy_forward_masks_time_and_action_dimensions_and_reports_matching_diagnostics():
    policy, batch = _policy_and_batch()
    loss, metrics = policy(batch)
    loss.backward()

    assert loss.item() == pytest.approx(91 / 6)
    # Evo-RL divides the batch's valid-element sum by its valid-element count.
    # Equal weighting of the two per-sample means would instead give 12.0.
    assert loss.item() != pytest.approx(12.0)
    assert metrics["loss_per_dim"] == pytest.approx([35 / 3, 56 / 3])
    grad = policy.model.prediction.grad
    assert torch.count_nonzero(grad[:, :, 2]) == 0
    assert torch.count_nonzero(grad[batch["action_is_pad"]]) == 0
    torch.testing.assert_close(grad[0, 0, :2], torch.tensor([1 / 3, 2 / 3]))


def test_policy_preserves_upstream_optional_sample_weighting_interface():
    policy, batch = _policy_and_batch()
    per_sample, metrics = policy(batch, reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([2.5, 21.5]))
    assert metrics["loss"] == pytest.approx(12.0)

    # Compatibility with upstream's opt-in RA-BC path, not the Evo-RL SFT recipe.
    weights = torch.tensor([0.0, 1.0])
    weighted_loss = (per_sample * weights).sum() / weights.sum()
    weighted_loss.backward()
    assert torch.count_nonzero(policy.model.prediction.grad[0]) == 0
    assert torch.count_nonzero(policy.model.prediction.grad[1, 2]) == 0
    torch.testing.assert_close(
        policy.model.prediction.grad[1, :2, :2], torch.tensor([[1.5, 2.0], [2.5, 3.0]])
    )


def test_policy_forward_combines_trained_rtc_with_padding(monkeypatch):
    policy, batch = _policy_and_batch()
    policy.config.rtc_training_max_delay = 1
    monkeypatch.setattr(
        modeling_pi05,
        "_sample_training_rtc_prefix_mask",
        lambda *args: torch.tensor([[True, False, False], [True, False, False]]),
    )
    loss, metrics = policy(batch)
    assert loss.item() == pytest.approx(30.5)
    assert metrics["loss_per_dim"] == pytest.approx([25.0, 36.0])


def test_policy_masking_is_opt_in_and_requires_dataset_mask_when_enabled():
    assert PI05Config().mask_action_padding_loss is False
    policy, batch = _policy_and_batch(masking=False)
    batch.pop("action_is_pad")
    loss, _ = policy(batch)
    torch.testing.assert_close(loss, policy.model.prediction[:, :, :2].square().mean())

    policy.config.mask_action_padding_loss = True
    with pytest.raises(ValueError, match="requires action_is_pad"):
        policy(batch)


def test_padding_mask_setting_survives_checkpoint_config_roundtrip(tmp_path):
    config = PI05Config(device="cpu", mask_action_padding_loss=True)
    config.save_pretrained(tmp_path)
    assert PI05Config.from_pretrained(tmp_path).mask_action_padding_loss is True


def test_padding_mask_survives_pi05_preprocessing(monkeypatch):
    def tokenize(text, **kwargs):
        return {
            "input_ids": torch.ones(len(text), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
        }

    monkeypatch.setattr(tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *args: tokenize)
    policy, batch = _policy_and_batch()
    policy.config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))}
    stats = {key: {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]} for key in (OBS_STATE, ACTION)}
    preprocessor, _ = make_pi05_pre_post_processors(policy.config, stats)
    raw = {
        OBS_STATE: torch.zeros(2, 2),
        ACTION: torch.zeros(2, 3, 2),
        "task": ["match the shape", "match the shape"],
        "action_is_pad": batch["action_is_pad"].clone(),
    }

    processed = preprocessor(raw)

    torch.testing.assert_close(processed["action_is_pad"], batch["action_is_pad"])
    loss, _ = policy(processed)
    assert loss.item() == pytest.approx(91 / 6)


def test_sft_training_step_uses_valid_element_mean_without_sample_weighting():
    pytest.importorskip("datasets")
    accelerate = pytest.importorskip("accelerate")
    import draccus

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=test/merged",
            "--policy.type=pi05",
            "--policy.device=cpu",
            "--policy.mask_action_padding_loss=true",
            "--policy.rtc_training_max_delay=0",
            "--intervention_sampling_ratio=0.5",
            "--sample_weighting=null",
        ],
    )
    assert cfg.sample_weighting is None
    assert cfg.policy.mask_action_padding_loss
    assert cfg.policy.rtc_training_max_delay == 0
    policy, batch = _policy_and_batch()
    batch["intervention"] = torch.tensor([False, True])
    before = policy.model.prediction.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics = MetricsTracker(
        batch_size=2,
        num_frames=2,
        num_episodes=2,
        metrics={name: AverageMeter(name, ":.3f") for name in ("loss", "lr", "update_s", "gpu_mem_gb")},
    )

    metrics, outputs = update_policy(
        metrics, policy, batch, optimizer, grad_clip_norm=0, accelerator=accelerate.Accelerator(cpu=True)
    )

    assert metrics.loss.val == pytest.approx(91 / 6)
    assert not any(key.startswith("sample_weight_") for key in outputs)
    expected_gradient = torch.zeros_like(before)
    expected_gradient[0, 0, :2] = 2 * before[0, 0, :2] / 6
    expected_gradient[1, :2, :2] = 2 * before[1, :2, :2] / 6
    torch.testing.assert_close(policy.model.prediction, before - 0.1 * expected_gradient)
