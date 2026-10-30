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

"""PEFT adapters for RLinf's fused attention and native gated MLP weights."""

import math
from dataclasses import fields
from inspect import signature
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _peft_available, require_package

from .gemma import FeedForward
from .utils import gelu_glu

if TYPE_CHECKING or _peft_available:
    from peft import LoraConfig
    from peft.tuners.lora.layer import LoraLayer, MultiheadAttention
    from peft.tuners.tuners_utils import check_adapters_to_merge
else:
    LoraLayer = object
    MultiheadAttention = nn.Module


class GatedFeedForward(nn.Module, LoraLayer):
    """Adapt gate/up/down projections while preserving RLinf's packed base weights.

    Apply low-rank projections to activations during training. Dense weight
    updates are only materialized when merging into a native checkpoint.
    """

    adapter_layer_names = ("lora_A", "lora_B", "lora_A_up", "lora_B_up", "lora_A_down", "lora_B_down")

    def __init__(
        self, base_layer, adapter_name: str, config: "LoraConfig | None" = None, r=0, lora_alpha=1, **kwargs
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        # PEFT 0.18 reads dimensions through a global helper rather than the
        # overridable method used by newer releases.
        self.in_features, self.out_features = self._get_in_out_features(self.get_base_layer())
        for name in self.adapter_layer_names[2:]:
            setattr(self, name, nn.ModuleDict())
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, config=config, **kwargs)

    def _get_in_out_features(self, module) -> tuple[int, int]:
        return module.features, module.hidden_dim

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device=None) -> None:
        weight = self.get_base_layer().w_gating
        for name in self.adapter_layer_names:
            adapters = getattr(self, name)
            if adapter_name in adapters and not any(p.is_meta for p in adapters[adapter_name].parameters()):
                adapters[adapter_name].to(
                    device=weight.device if device is None else device, dtype=weight.dtype
                )

    def update_layer(
        self, adapter_name: str, r: int, lora_alpha: int, config: "LoraConfig | None" = None, **kwargs
    ):
        if config is None:
            # PEFT <0.20 passes individual settings instead of a LoraConfig.
            if kwargs.get("use_alora"):
                raise ValueError("RLinf gated MLP LoRA does not support aLoRA.")
            config_fields = {field.name for field in fields(LoraConfig) if field.init}
            config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                **{key: value for key, value in kwargs.items() if key in config_fields},
            )
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")
        if hasattr(LoraLayer, "lora_variants"):
            self.resolve_lora_variant(config=config)
        elif any(getattr(config, name, None) for name in ("use_dora", "use_qalora", "arrow_config")):
            raise ValueError("RLinf gated MLP supports vanilla LoRA and rsLoRA only.")
        if config.fan_in_fan_out or config.lora_bias:
            raise ValueError("RLinf gated MLP LoRA requires fan_in_fan_out=false and lora_bias=false.")
        if config.init_lora_weights not in (True, False, "gaussian"):
            raise ValueError("RLinf gated MLP LoRA supports init_lora_weights=true, false or 'gaussian'.")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / (math.sqrt(r) if config.use_rslora else r)
        self.use_rslora[adapter_name] = config.use_rslora
        self.use_dora[adapter_name] = False
        self.lora_bias[adapter_name] = False
        self.lora_dropout[adapter_name] = nn.Dropout(config.lora_dropout)
        dimensions = {
            "gate": (self.in_features, self.out_features),
            "up": (self.in_features, self.out_features),
            "down": (self.out_features, self.in_features),
        }
        for name, (inputs, outputs) in dimensions.items():
            down, up = self._projection_adapters(name)
            # Keep the adapter name immediately before `.weight`, as expected
            # by PEFT's checkpoint save/load key mapping.
            down[adapter_name] = nn.Linear(inputs, r, bias=False)
            up[adapter_name] = nn.Linear(r, outputs, bias=False)
            if config.init_lora_weights:
                weight = down[adapter_name].weight
                if config.init_lora_weights == "gaussian":
                    nn.init.normal_(weight, std=1 / r)
                else:
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                nn.init.zeros_(up[adapter_name].weight)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def _projection_adapters(self, projection: str) -> tuple[nn.ModuleDict, nn.ModuleDict]:
        suffix = "" if projection == "gate" else f"_{projection}"
        return getattr(self, "lora_A" + suffix), getattr(self, "lora_B" + suffix)

    def _project(self, x: Tensor, weight: Tensor, projection: str) -> Tensor:
        result = torch.matmul(x, weight.to(x.dtype))
        result_dtype = result.dtype
        for adapter in self.active_adapters:
            if adapter in self.lora_A:
                adapters_a, adapters_b = self._projection_adapters(projection)
                down, up = adapters_a[adapter], adapters_b[adapter]
                inputs = self.lora_dropout[adapter](self._cast_input_dtype(x, down.weight.dtype))
                result = result + up(down(inputs)) * self.scaling[adapter]
        return result.to(result_dtype)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        self._check_forward_args(x, **kwargs)
        if kwargs.get("adapter_names") is not None:
            raise ValueError("RLinf gated MLP LoRA does not support mixed adapter batches.")
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x)
        if self.merged:
            return self.base_layer(x)
        base = self.get_base_layer()
        gate = self._project(x, base.w_gating[0], "gate")
        up = self._project(x, base.w_gating[1], "up")
        return self._project(gelu_glu(gate, up), base.w_linear, "down")

    def _delta_weights(self, adapter: str) -> tuple[Tensor, Tensor]:
        updates = {}
        for name in ("gate", "up", "down"):
            down, up = self._projection_adapters(name)
            updates[name] = (up[adapter].weight @ down[adapter].weight).T * self.scaling[adapter]
        return torch.stack([updates["gate"], updates["up"]]), updates["down"]

    @torch.no_grad()
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        base = self.get_base_layer()
        for adapter in check_adapters_to_merge(self, adapter_names):
            if adapter in self.lora_A:
                gating, linear = self._delta_weights(adapter)
                gating = base.w_gating + gating.to(base.w_gating.dtype)
                linear = base.w_linear + linear.to(base.w_linear.dtype)
                if safe_merge and (not torch.isfinite(gating).all() or not torch.isfinite(linear).all()):
                    raise ValueError(f"Non-finite gated MLP weights for adapter {adapter!r}.")
                base.w_gating.copy_(gating)
                base.w_linear.copy_(linear)
                self.merged_adapters.append(adapter)

    @torch.no_grad()
    def unmerge(self) -> None:
        base = self.get_base_layer()
        while self.merged_adapters:
            adapter = self.merged_adapters.pop()
            if adapter in self.lora_A:
                gating, linear = self._delta_weights(adapter)
                base.w_gating.sub_(gating.to(base.w_gating.dtype))
                base.w_linear.sub_(linear.to(base.w_linear.dtype))


class FunctionalMultiheadAttention(MultiheadAttention):
    """Keep frozen BF16 weights intact across forwards and adapter reloads.

    PEFT's MHA implementation merges/unmerges on every forward. The subtraction
    cannot undo BF16 rounding, so the frozen base drifts and cannot be recovered
    from an adapter checkpoint. Compute effective weights as temporary tensors
    instead, retaining PEFT's initialization, checkpoint and permanent-merge APIs.
    """

    def __init__(self, *args, **kwargs):
        # PEFT 0.18 passes individual settings; 0.20 passes a LoraConfig. Bind
        # the installed signature so positional settings cannot bypass checks.
        settings = signature(MultiheadAttention.__init__).bind(self, *args, **kwargs).arguments
        self._validate_settings(settings)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_settings(settings) -> None:
        values = {**settings.get("kwargs", {}), **settings}
        config = values.get("config")
        dropout = config.lora_dropout if config is not None else values.get("lora_dropout", 0.0)
        bias = config.lora_bias if config is not None else values.get("lora_bias", False)
        if dropout != 0:
            raise ValueError(
                "RLinf fused vision attention requires lora_dropout=0; "
                "activation dropout is not supported by its effective-weight LoRA path."
            )
        if bias:
            raise ValueError(
                "RLinf fused vision attention requires lora_bias=false; "
                "LoRA-B biases are not supported by its effective-weight LoRA path."
            )

    def update_layer(self, *args, **kwargs) -> None:
        settings = signature(LoraLayer.update_layer).bind(self, *args, **kwargs).arguments
        self._validate_settings(settings)
        super().update_layer(*args, **kwargs)

    @torch.no_grad()
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        base = self.get_base_layer()
        for adapter in check_adapters_to_merge(self, adapter_names):
            if adapter in self.lora_A:
                weight = base.in_proj_weight + self.get_delta_weight(adapter).to(base.in_proj_weight.dtype)
                if safe_merge and not torch.isfinite(weight).all():
                    raise ValueError(f"Non-finite fused vision attention weights for adapter {adapter!r}.")
                # PEFT 0.20's MHA merge adds the output delta itself and calls
                # out_proj.merge, applying that delta twice. Delegate it once.
                base.out_proj.merge(safe_merge=safe_merge, adapter_names=[adapter])
                base.in_proj_weight.copy_(weight)
                self.merged_adapters.append(adapter)

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        base = self.get_base_layer()
        while self.merged_adapters:
            adapter = self.merged_adapters.pop()
            if adapter in self.lora_A:
                base.in_proj_weight.sub_(self.get_delta_weight(adapter).to(base.in_proj_weight.dtype))
        base.out_proj.unmerge()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        self._check_forward_args(query, key, value, **kwargs)
        if self.disable_adapters or self.merged:
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        base = self.get_base_layer()
        if base.out_proj.active_adapters != self.active_adapters:
            raise ValueError("Fused vision attention and its output projection must use the same adapters.")
        in_weight = base.in_proj_weight
        out_weight = base.out_proj.get_base_layer().weight
        for adapter in self.active_adapters:
            if adapter in self.lora_A:
                in_weight = in_weight + self.get_delta_weight(adapter).to(in_weight.dtype)
                out_weight = out_weight + base.out_proj.get_delta_weight(adapter).to(out_weight.dtype)

        batched = query.ndim == 3
        if base.batch_first and batched:
            # Preserve aliases so PyTorch uses the same packed self-attention
            # projection as nn.MultiheadAttention rather than three matmuls.
            if query is key and key is value:
                query = key = value = query.transpose(0, 1)
            elif key is value:
                query, key = query.transpose(0, 1), key.transpose(0, 1)
                value = key
            else:
                query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            base.embed_dim,
            base.num_heads,
            in_weight,
            base.in_proj_bias,
            base.bias_k,
            base.bias_v,
            base.add_zero_attn,
            base.dropout,
            out_weight,
            base.out_proj.bias,
            training=base.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if base.batch_first and batched:
            output = output.transpose(0, 1)
        return output, weights


def prepare_lora_config(config: "LoraConfig") -> "LoraConfig":
    """Re-register custom dispatch on both initial wrapping and checkpoint load."""
    require_package("peft", extra="peft")
    if config.peft_type == "LORA":
        config._register_custom_module(
            {nn.MultiheadAttention: FunctionalMultiheadAttention, FeedForward: GatedFeedForward}
        )
    return config
