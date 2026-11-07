"""Portable RLT/PEFT exports must retain the tokenizer and numerical behavior they saved."""

import shutil

import pytest
import torch

from examples.tutorial.pi05.export_rlt import export_rlt
from examples.tutorial.pi05.merge_lora import merge_adapter
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.policies.pi0_pi05.test_pi05_rlinf import config, tiny_networks as tiny_networks
from tests.policies.pi0_pi05.test_pi05_rlt import rlt_config, train
from tests.policies.pi0_pi05.test_pi05_rlt_stage2 import raw_batch, write_tokenizer


@pytest.mark.parametrize("export_mode", ["full", "peft_merge", "peft_encoder"])
@pytest.mark.parametrize("external_change", ["changed", "removed"])
def test_export_preserves_bundled_tokenizer_and_model_outputs(
    tiny_networks, tmp_path, export_mode, external_change
):
    external, saved, output = tmp_path / "external", tmp_path / "saved", tmp_path / "export"
    write_tokenizer(external)
    overrides = {"text_tokenizer_name": str(external)}
    if export_mode == "full":
        source = rlt_config(**overrides)
        policy = PI05Policy(source)
    else:
        base = tmp_path / "base"
        PI05Policy(config()).save_pretrained(base)
        source = rlt_config(
            **overrides,
            pretrained_path=base,
            rlt_training_mode="joint",
            peft_mode="vlm_lora_expert_full",
        )
        policy = PI05Policy.from_pretrained(base, config=source)
        policy = policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8})
    stats = {
        key: {"q01": torch.full((3,), -2.0), "q99": torch.full((3,), 6.0)} for key in (OBS_STATE, ACTION)
    }
    pre, post = make_pi05_pre_post_processors(source, stats)
    raw = {**raw_batch(), "task": "foo"}
    batch = pre(raw)
    train(policy, {**batch, ACTION: torch.zeros(1, 4, 3)}, steps=1)
    policy.eval()
    expected_tokens = batch[OBS_LANGUAGE_TOKENS].clone()
    expected_z = policy.extract_rlt_features(batch)
    noise = torch.randn(1, 4, 4)
    expected_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    policy.save_pretrained(saved)
    policy.config.save_pretrained(saved)
    pre.save_pretrained(saved)
    post.save_pretrained(saved)
    original_config = (saved / "config.json").read_bytes()
    if external_change == "changed":
        write_tokenizer(external, swapped=True)
    else:
        shutil.rmtree(external)

    if export_mode == "peft_merge":
        merge_adapter(str(saved), output)
    else:
        export_rlt(saved, output)

    assert (saved / "config.json").read_bytes() == original_config
    cfg = PreTrainedConfig.from_pretrained(output)
    assert cfg.text_tokenizer_name is None and not cfg.use_peft
    assert cfg.rlt_inference_only is (export_mode != "peft_merge")
    exported_pre, _ = make_pre_post_processors(cfg, pretrained_path=output)
    exported_batch = exported_pre(raw)
    torch.testing.assert_close(exported_batch[OBS_LANGUAGE_TOKENS], expected_tokens, rtol=0, atol=0)
    restored = PI05Policy.from_pretrained(output)
    torch.testing.assert_close(
        restored.extract_rlt_features(exported_batch), expected_z, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        restored.predict_action_chunk(exported_batch, noise=noise.clone()),
        expected_actions,
        rtol=2e-4,
        atol=2e-5,
    )
