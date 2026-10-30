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

"""Strict, in-memory conversion of OpenPI/LeRobot PI05 weights to RLinf layout.

The tensor transforms follow RLinf's ``openpi_pytorch_to_openpi`` converter at
061c4cce80bbab998ef90585475d3f51290a2dfa. Unlike that utility, this also accepts
LeRobot's outer ``model.`` prefix and its saved input embedding, and rejects
unrecognized tensors instead of silently omitting them.
"""

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from ..configuration_pi05 import PI05Config

_VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
_LANGUAGE = "paligemma_with_expert.paligemma.model.language_model."
_EXPERT = "paligemma_with_expert.gemma_expert.model."
_PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
_UNUSED_HEADS = {
    "paligemma_with_expert.paligemma.lm_head.weight",
    "paligemma_with_expert.gemma_expert.lm_head.weight",
}
_WRAPPERS = ("module.", "_orig_mod.", "_fsdp_wrapped_module.", "model.")


def _bare_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    for key, value in state.items():
        original = key
        while key.startswith(_WRAPPERS):
            key = next(key[len(prefix) :] for prefix in _WRAPPERS if key.startswith(prefix))
        if key in result:
            raise ValueError(f"Checkpoint wrapper prefixes collide at {original!r}.")
        result[key] = value
    return result


def to_rlinf_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Convert action-relevant tensors without changing precision or their source.

    The unused token-output heads and SigLIP pooling head are omitted. Every other source tensor
    must be recognized, including all members of packed QKV and gate/up groups.
    Target model loading subsequently checks missing keys and tensor shapes.
    """
    state = _bare_state_dict(state)
    if any(key.startswith(("img.", "llm.")) for key in state):
        return state
    if not any(key.startswith("paligemma_with_expert.") for key in state):
        raise ValueError("Expected a LeRobot/OpenPI PI05 or native RLinf OpenPI checkpoint.")

    result = {}
    groups: dict[tuple[str, str], dict[str, Tensor]] = {}
    unknown = []
    embedding = state.get(_LANGUAGE + "embed_tokens.weight")
    if embedding is None:
        embedding = state.get("paligemma_with_expert.paligemma.lm_head.weight")
    if embedding is not None:
        result["llm.embedder.embedding.weight"] = embedding

    for key, value in state.items():
        if (
            key in _UNUSED_HEADS
            or key == _LANGUAGE + "embed_tokens.weight"
            or key.startswith(_VISION + "head.")
        ):
            continue
        if key.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")):
            result[key] = value
            continue
        if key.startswith(_PROJECTOR):
            result["img.head." + key[len(_PROJECTOR) :]] = value
            continue
        if key.startswith(_VISION):
            suffix = key[len(_VISION) :]
            if suffix.startswith("embeddings.patch_embedding."):
                result["img.stem." + suffix.removeprefix("embeddings.patch_embedding.")] = value
                continue
            if suffix == "embeddings.position_embedding.weight":
                result["img.pos_embedding"] = value.unsqueeze(0)
                continue
            if suffix.startswith("post_layernorm."):
                result["img.encoder.norm." + suffix.removeprefix("post_layernorm.")] = value
                continue
            match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", suffix)
            if match:
                layer, part = match.groups()
                target = f"img.encoder.layers.{layer}."
                qkv = re.fullmatch(r"self_attn\.([qkv])_proj\.(weight|bias)", part)
                if qkv:
                    projection, parameter = qkv.groups()
                    groups.setdefault((target + "attn.in_proj_" + parameter, "qkv"), {})[projection] = value
                    continue
                for source, destination in (
                    ("layer_norm1.", "norm1."),
                    ("layer_norm2.", "norm2."),
                    ("self_attn.out_proj.", "attn.out_proj."),
                    ("mlp.fc1.", "mlp.fc1."),
                    ("mlp.fc2.", "mlp.fc2."),
                ):
                    if part.startswith(source):
                        result[target + destination + part[len(source) :]] = value
                        break
                else:
                    unknown.append(key)
                continue
        for expert, prefix in enumerate((_LANGUAGE, _EXPERT)):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.startswith("norm."):
                norm = suffix.removeprefix("norm.")
                if norm == "weight":
                    result[f"llm.final_norms.{expert}.scale"] = value
                elif norm.startswith("dense."):
                    result[f"llm.final_norms.{expert}.ada_modulation." + norm.removeprefix("dense.")] = value
                else:
                    unknown.append(key)
                break
            match = re.fullmatch(r"layers\.(\d+)\.(.+)", suffix)
            if match is None:
                unknown.append(key)
                break
            layer, part = match.groups()
            target = f"llm.layers.{layer}."
            attention = re.fullmatch(r"self_attn\.([qkvo]_proj)\.weight", part)
            if attention:
                result[f"{target}attn.{attention[1]}.{expert}.weight"] = value
            elif part in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
                projection = "gate" if "gate_proj" in part else "up"
                groups.setdefault((f"{target}mlps.{expert}.w_gating", "mlp"), {})[projection] = value
            elif part == "mlp.down_proj.weight":
                result[f"{target}mlps.{expert}.w_linear"] = value.T.contiguous()
            else:
                norm = re.fullmatch(
                    r"(input_layernorm|post_attention_layernorm)\.(weight|dense\.(?:weight|bias))", part
                )
                if norm:
                    position, parameter = norm.groups()
                    name = "pre_attention_norms" if position == "input_layernorm" else "pre_ffw_norms"
                    parameter = (
                        "scale" if parameter == "weight" else parameter.replace("dense.", "ada_modulation.")
                    )
                    result[f"{target}{name}.{expert}.{parameter}"] = value
                else:
                    unknown.append(key)
            break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"Unrecognized PI05 checkpoint tensors: {unknown[:8]}")
    for (target, kind), members in groups.items():
        required = ("q", "k", "v") if kind == "qkv" else ("gate", "up")
        if set(members) != set(required):
            raise ValueError(
                f"Incomplete {kind} group for {target}: expected {required}, got {tuple(members)}"
            )
        result[target] = (
            torch.cat([members[name] for name in required], dim=0)
            if kind == "qkv"
            else torch.stack([members[name].T.contiguous() for name in required])
        )
    return result


def load_pretrained_policy[PolicyT](
    policy_class: type[PolicyT],
    pretrained_name_or_path: str | Path,
    *,
    config: PI05Config,
    force_download: bool = False,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    strict: bool = True,
    initialize_rlt: bool = False,
    **kwargs: Any,
) -> PolicyT:
    """Load safely on CPU, convert layout, and propagate all loading failures."""
    directory = Path(pretrained_name_or_path)
    if directory.is_dir():
        filename = directory / "model.safetensors"
    elif directory.is_file():
        filename = directory
    else:
        filename = hf_hub_download(
            str(pretrained_name_or_path),
            "model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            local_files_only=local_files_only,
        )
    state = to_rlinf_state_dict(load_file(str(filename), device="cpu"))
    policy = policy_class(config, **kwargs)
    expected_rlt = {name for name in policy.model.state_dict() if name.startswith("rlt_module.")}
    provided_rlt = {name for name in state if name.startswith("rlt_module.")}
    if provided_rlt and not config.use_rlt:
        raise ValueError("This checkpoint contains RLT weights; load it with policy.use_rlt=True.")
    if provided_rlt and provided_rlt != expected_rlt:
        raise RuntimeError(
            "Incomplete or incompatible RLT checkpoint: "
            f"missing={sorted(expected_rlt - provided_rlt)[:5]}, "
            f"unexpected={sorted(provided_rlt - expected_rlt)[:5]}"
        )
    if config.use_rlt and not provided_rlt:
        # A new RLT can be initialized from a declared ordinary PI05 base. A
        # trained RLT checkpoint missing its head must never silently reinitialize.
        if directory.is_dir() or directory.is_file():
            source_config_path = Path(filename).parent / "config.json"
        else:
            source_config_path = Path(
                hf_hub_download(
                    str(pretrained_name_or_path),
                    "config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            )
        source_config = json.loads(source_config_path.read_text()) if source_config_path.is_file() else {}
        if source_config.get("use_rlt"):
            raise RuntimeError(
                "Incomplete RLT checkpoint: its saved config enables RLT but all RLT weights are missing."
            )
        declared_base = source_config.get("type") == "pi05" and not source_config.get("use_rlt", False)
        if not (declared_base or initialize_rlt):
            raise ValueError(
                "Initializing RLT from a bare network requires initialize_rlt=True; "
                "a saved ordinary LeRobot PI05 base is recognized automatically."
            )
        incompatible = policy.model.load_state_dict(state, strict=False)
        missing_core = set(incompatible.missing_keys) - expected_rlt
        if strict and (missing_core or incompatible.unexpected_keys):
            raise RuntimeError(
                f"Incomplete PI05 base: missing={sorted(missing_core)[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        logging.info("Initialized new RLT weights from the ordinary PI05 base %s", pretrained_name_or_path)
        policy.eval()
        return policy
    # Load the core directly; the outer LeRobot prefix is normalized above.
    incompatible = policy.model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        logging.warning("Partial RLinf PI05 checkpoint load: %s", incompatible)
    policy.eval()
    return policy
