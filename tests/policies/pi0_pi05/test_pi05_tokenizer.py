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

"""Local-tokenizer config, processor reload and portable PI05 exports, without downloads."""

from copy import deepcopy
from dataclasses import replace

import draccus
import pytest
import torch

pytest.importorskip("transformers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.processor import TokenizerProcessorStep  # noqa: E402
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


def save_tokenizer(path, marker="original"):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, marker: 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        path
    )
    return path


def tokenizer_step(pipeline):
    return next(step for step in pipeline.steps if isinstance(step, TokenizerProcessorStep))


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
def test_training_cli_accepts_local_pi05_tokenizer_and_mixed_lora(backend):
    path = "/home/yzh/具身智能/基座模型/Lerobot/paligemma-3b-pt-224"
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/task",
            "--use_pretrained_stats=true",
            "--policy.type=pi05",
            "--policy.pretrained_path=/home/yzh/具身智能/基座模型/Lerobot/pi05_base",
            "--policy.device=cpu",
            f"--policy.backend={backend}",
            f"--policy.text_tokenizer_name={path}",
            "--policy.peft_mode=vlm_lora_expert_full",
            "--policy.peft_language_lora=attention_mlp",
            "--policy.peft_vision_lora=none",
            "--peft.method_type=LORA",
            "--peft.r=16",
            "--peft.lora_alpha=32",
        ],
    )
    assert cfg.policy.text_tokenizer_name == path
    assert cfg.policy.peft_vision_lora == "none"
    assert cfg.peft.r == 16
    assert cfg.use_pretrained_stats
    assert cfg.policy.precision_profile == "lerobot"
    assert cfg.policy.mask_action_padding_loss is False


def test_local_tokenizer_checkpoint_config_roundtrip_and_cli_override(tmp_path):
    cfg = PI05Config(device="cpu", text_tokenizer_name="/original/paligemma")
    cfg.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).text_tokenizer_name == "/original/paligemma"
    loaded = PreTrainedConfig.from_pretrained(
        tmp_path, cli_overrides=["--text_tokenizer_name=/server/paligemma"]
    )
    assert loaded.text_tokenizer_name == "/server/paligemma"
    bundled = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=["--text_tokenizer_name=null"])
    assert bundled.text_tokenizer_name is None


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
@pytest.mark.parametrize("custom", [False, True])
def test_fresh_processors_use_configured_or_original_default_tokenizer(
    tmp_path, monkeypatch, backend, custom
):
    local = save_tokenizer(tmp_path / "本地 tokenizer")
    expected = str(local) if custom else "google/paligemma-3b-pt-224"
    calls = []
    original_load = AutoTokenizer.from_pretrained

    def load(name, **kwargs):
        calls.append(name)
        return original_load(local, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    cfg = PI05Config(device="cpu", backend=backend, text_tokenizer_name=str(local) if custom else None)
    pre, _ = make_pre_post_processors(cfg)
    assert calls == [expected]
    assert tokenizer_step(pre).input_tokenizer.get_vocab()["original"] == 2


@pytest.mark.parametrize("backend", ["lerobot", "rlinf"])
@pytest.mark.parametrize("selection", ["saved_artifact", "policy_override", "processor_override"])
def test_tokenizer_reload_precedence_and_portable_saved_artifact(tmp_path, backend, selection):
    original = save_tokenizer(tmp_path / "original_tokenizer")
    requested = save_tokenizer(tmp_path / "requested_tokenizer", "requested")
    explicit = save_tokenizer(tmp_path / "explicit_tokenizer", "explicit")
    cfg = PI05Config(device="cpu", backend=backend, text_tokenizer_name=str(original))
    pre, post = make_pre_post_processors(cfg)
    checkpoint = tmp_path / "checkpoint"
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    original.rename(tmp_path / "original_tokenizer_moved")
    cfg = replace(cfg, text_tokenizer_name=None if selection == "saved_artifact" else str(requested))
    overrides = {"device_processor": {"device": "cpu"}}
    if selection == "processor_override":
        overrides["tokenizer_processor"] = {"tokenizer_name": str(explicit), "max_length": 19}
    unchanged = deepcopy(overrides)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(checkpoint), preprocessor_overrides=overrides)
    marker = {"saved_artifact": "original", "policy_override": "requested", "processor_override": "explicit"}[
        selection
    ]
    assert marker in tokenizer_step(pre).input_tokenizer.get_vocab()
    assert overrides == unchanged
    if selection == "processor_override":
        assert tokenizer_step(pre).max_length == 19


@pytest.mark.parametrize("entrypoint", ["merge_lerobot", "merge_rlinf", "compare"])
def test_export_uses_bundled_tokenizer_after_original_directory_moves(tiny_networks, tmp_path, entrypoint):
    pytest.importorskip("peft")
    from examples.tutorial.pi05.compare_backends import compare
    from examples.tutorial.pi05.merge_lora import merge_adapter

    local = save_tokenizer(tmp_path / "本地 tokenizer")
    cfg = replace(
        config(),
        backend="rlinf" if entrypoint == "merge_rlinf" else "lerobot",
        text_tokenizer_name=str(local),
        peft_mode="vlm_lora_expert_full",
        peft_vision_lora="none",
    )
    base, source, exported = tmp_path / "base", tmp_path / "source", tmp_path / "exported"
    policy = PI05Policy(cfg)
    policy.save_pretrained(base)
    if entrypoint != "compare":
        policy.config.pretrained_path = base
        policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4, "lora_alpha": 8})
    policy.save_pretrained(source)
    cfg.save_pretrained(source)  # The normal training checkpoint writer also saves the policy config.
    stats = {key: {"q01": torch.zeros(3), "q99": torch.ones(3)} for key in (OBS_STATE, ACTION)}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(source)
    post.save_pretrained(source)
    if entrypoint == "compare":
        images, _, tokens, mask = inputs()
        batch = {
            OBS_STATE: torch.zeros(1, 3),
            "observation.images.front": (images[0] + 1) / 2,
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: mask,
        }
        compare(str(source), batch, batch_is_preprocessed=True, warmup=0, iterations=1, export_dir=exported)
    else:
        merge_adapter(str(source), exported)
    local.rename(tmp_path / "original_tokenizer_moved")
    exported_cfg = PreTrainedConfig.from_pretrained(exported)
    assert exported_cfg.text_tokenizer_name is None
    pre, _ = make_pre_post_processors(exported_cfg, pretrained_path=str(exported))
    assert "original" in tokenizer_step(pre).input_tokenizer.get_vocab()
