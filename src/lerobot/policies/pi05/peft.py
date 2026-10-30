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

"""PEFT recipes and complete adapter loading for PI05's two network layouts."""

import re
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _peft_available, require_package

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel

if TYPE_CHECKING:
    from .modeling_pi05 import PI05Policy


def load_pi05_adapter(policy: "PI05Policy", adapter: str, **kwargs):
    """Load a complete adapter onto a fresh base, including all full-trained tensors.

    PEFT tolerates missing LoRA weights. Inspect PyTorch's unfiltered missing
    keys during loading and normalize missing modules_to_save errors, so an
    incomplete checkpoint cannot proceed to inference or merged export.
    """
    require_package("peft", extra="peft")
    policy._validate_peft_config(kwargs["config"])

    def require_complete_adapter(module, incompatible):
        missing = [
            name for name in incompatible.missing_keys if "lora_" in name or ".modules_to_save." in name
        ]
        if missing:
            raise RuntimeError(f"Incomplete PI05 adapter; missing trained tensors: {missing[:8]}")

    handle = policy.register_load_state_dict_post_hook(require_complete_adapter)
    try:
        return PeftModel.from_pretrained(policy, adapter, **kwargs)
    except KeyError as error:
        if error.args and str(error.args[0]).startswith("base_model.model."):
            raise RuntimeError(f"Incomplete PI05 adapter; missing trained tensor: {error.args[0]}") from error
        raise
    finally:
        handle.remove()


def vlm_lora_expert_full_targets(policy: "PI05Policy") -> dict[str, Any]:
    """Select vision/language LoRA scopes; train and save the action stream.

    RLinf interleaves its VLM (index 0) and action expert (index 1) inside each
    transformer block. Its expert is not a single parent module that PEFT can
    wrap, so all expert submodules must be listed, including adaptive norms.
    """
    projections = [
        "model." + name for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    ]
    if policy.config.backend == "lerobot":
        modules = ["model.paligemma_with_expert.gemma_expert.model", *projections]
        if policy.config.use_proprioceptive_memory:
            modules.append("model.proprio_history_proj")
        prefix = r"model\.paligemma_with_expert\.paligemma\.model\."
        scopes = {
            "language": (
                prefix + r"language_model\.layers\.\d+\.self_attn\.[qkvo]_proj",
                prefix + r"language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj",
            ),
            "vision": (
                prefix + r"vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj",
                prefix + r"vision_tower\.vision_model\.encoder\.layers\.\d+\.mlp\.fc[12]",
            ),
        }
    else:
        expert = r"model\.llm\.(layers\.\d+\.(attn\.[qkvo]_proj|pre_attention_norms|pre_ffw_norms|mlps)\.1|final_norms\.1)"
        modules = [name for name, _ in policy.named_modules() if re.fullmatch(expert, name)] + projections
        scopes = {
            "language": (
                r"model\.llm\.layers\.\d+\.attn\.[qkvo]_proj\.0",
                r"model\.llm\.layers\.\d+\.mlps\.0",
            ),
            "vision": (
                # The fused MHA adapter handles both QKV and the output projection;
                # targeting out_proj alone would miss MultiheadAttention's forward.
                r"model\.img\.encoder\.layers\.\d+\.attn",
                r"model\.img\.encoder\.layers\.\d+\.mlp\.fc[12]",
            ),
        }

    targets = []
    for component, (attention, mlp) in scopes.items():
        selection = getattr(policy.config, f"peft_{component}_lora")
        if selection != "none":
            targets.append(attention)
        if selection == "attention_mlp":
            targets.append(mlp)
    return {
        # PEFT's DUMMY_TARGET_MODULES sentinel permits an expert-only checkpoint
        # with modules_to_save and no LoRA layers, without importing PEFT here.
        "target_modules": "(" + "|".join(targets) + ")" if targets else "dummy-target-modules",
        "modules_to_save": modules,
    }
