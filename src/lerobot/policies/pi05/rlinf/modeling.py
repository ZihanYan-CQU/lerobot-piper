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

"""RLinf's native OpenPI network behind LeRobot's existing PI05/RTC interface."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

from ..configuration_pi05 import PI05Config
from ..modeling_pi05 import (
    PI05Pytorch,
    _build_flow_matching_inputs,
)
from . import gemma, siglip


class RLinfPI05Pytorch(PI05Pytorch):
    """Reuse PI05's sampling helpers while replacing its entire network core.

    ``img.*`` and ``llm.*`` retain RLinf checkpoint names. Observation transforms,
    action normalization, guided/trained RTC, and the policy's action queue stay
    owned by LeRobot. No OpenPI, JAX, Ray, or RLinf package is imported.
    """

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        # Do not construct the Transformers network in PI05Pytorch.__init__.
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor
        self.gradient_checkpointing_enabled = False
        vlm_config = gemma.get_config(config.paligemma_variant)
        expert_config = gemma.get_config(config.action_expert_variant)
        self.llm = gemma.Module(
            [vlm_config, expert_config],
            embed_dtype=config.dtype,
            adarms=[False, True],
            use_gradient_checkpointing=False,
        )
        self.img = siglip.SigLIPViT(
            variant="So400m/14", pool_type="none", num_classes=vlm_config.width, dtype_mm=config.dtype
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_config.width)
        self.action_out_proj = nn.Linear(expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(expert_config.width, expert_config.width)
        self.time_mlp_out = nn.Linear(expert_config.width, expert_config.width)
        self._set_precision()
        if config.use_rlt:
            from .rlt import RLTTokenTransformer

            # Keep new trainable RLT weights in FP32; outer AMP controls compute.
            self.rlt_module = RLTTokenTransformer(
                input_dim=vlm_config.width,
                embed_dim=config.rlt_embed_dim,
                prefix_seq_len=config.rlt_prefix_seq_len,
                num_layers=config.rlt_num_layers,
                num_heads=config.rlt_num_heads,
                mlp_ratio=config.rlt_mlp_ratio,
                inference_only=config.rlt_inference_only,
            )
        self._freeze_parameters()
        if config.compile_model or config.precision_profile == "openpi":
            torch.set_float32_matmul_precision("high")
        if config.compile_model:
            # Reuse PI05's public sampler dispatch: delay-dependent RTC controls
            # stay outside Dynamo, with separate prefix/suffix regions for the
            # native/reference guidance fallback.
            self._compiled_sample_actions = torch.compile(self._sample_actions, mode=config.compile_mode)
            self._compiled_rlt_prefix = torch.compile(self.build_prefix_cache, mode=config.compile_mode)
            self._compiled_prefill_prefix = torch.compile(self._prefill_prefix, mode=config.compile_mode)
            self._compiled_denoise_step = torch.compile(self.denoise_step, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _set_precision(self) -> None:
        if self.config.dtype == "float32":
            return
        if self.config.precision_profile == "checkpoint":
            self.to(dtype=torch.bfloat16)
            return
        self.llm.to(dtype=torch.bfloat16)
        self.img.to(dtype=torch.bfloat16)
        # Mirror RLinf Pi0.to_bfloat16_for_selected_params. Projections and
        # time conditioning live outside the trunk and retain FP32 weights.
        fp32_names = ("img.stem.", "img.pos_embedding", "pre_attention_norms", "pre_ffw_norms", "final_norms")
        for name, parameter in self.named_parameters():
            if any(part in name for part in fp32_names):
                parameter.data = parameter.data.float()
        if self.config.precision_profile == "lerobot":
            self.img.float()
            self.img.dtype_mm = torch.float32

    def _frozen_modules(self) -> list[nn.Module]:
        if self.config.use_rlt and self.config.rlt_training_mode == "frozen_vla":
            return [
                self.img,
                self.llm,
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ]
        modules = []
        if self.config.train_expert_only:
            modules.append(self.img)
        elif self.config.freeze_vision_encoder:
            modules.extend([self.img.stem, self.img.encoder])
        if self.config.train_expert_only:
            modules.extend([self.llm.embedder, self.llm.final_norms[0]])
            for block in self.llm.layers:
                modules.extend([block.pre_attention_norms[0], block.pre_ffw_norms[0], block.mlps[0]])
                for projections in (
                    block.attn.q_proj,
                    block.attn.k_proj,
                    block.attn.v_proj,
                    block.attn.o_proj,
                ):
                    if projections[0] is not None:
                        modules.append(projections[0])
        return modules

    def _freeze_parameters(self) -> None:
        if self.config.freeze_vision_encoder:
            self.img.pos_embedding.requires_grad_(False)
        for module in self._frozen_modules():
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for module in self._frozen_modules():
            module.eval()
        return self

    def gradient_checkpointing_enable(self) -> None:
        self._set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        self._set_gradient_checkpointing(False)

    def _set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing_enabled = enabled
        self.llm.gradient_checkpointing = enabled
        self.img.encoder.gradient_checkpointing = enabled
        if self.config.use_rlt:
            # Include PEFT's active modules_to_save copies when toggled after wrapping.
            for module in self.rlt_module.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = enabled

    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        embeddings, padding = [], []
        for image, image_mask in zip(images, img_masks, strict=True):
            # LeRobot prepares BCHW images in [-1, 1]; RLinf's stem expects BHWC.
            image_tokens, _ = self.img(image.permute(0, 2, 3, 1).to(self.img.stem.weight.dtype))
            embeddings.append(image_tokens)
            padding.append(image_mask[:, None].expand(image.shape[0], image_tokens.shape[1]))
        embeddings.append(self.llm.embed(tokens))
        padding.append(masks)
        prefix = torch.cat(embeddings, dim=1)
        prefix_mask = torch.cat(padding, dim=1).bool()
        return prefix, prefix_mask, torch.zeros_like(prefix_mask)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        # RLinf uses FP32 time features (the legacy backend uses FP64 before
        # projecting). Also accept per-action time for LeRobot's trained RTC.
        fraction = torch.linspace(
            0, 1, self.action_in_proj.out_features // 2, device=timestep.device, dtype=torch.float32
        )
        period = self.config.min_period * (self.config.max_period / self.config.min_period) ** fraction
        angles = timestep.float()[..., None] * (2 * torch.pi / period)
        time_embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        time_embedding = time_embedding.to(self.time_mlp_in.weight.dtype)
        condition = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_embedding))))
        actions = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        padding = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        autoregressive = torch.zeros_like(padding)
        autoregressive[:, 0] = True
        return actions, padding, autoregressive, condition

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask=None,
        states=None,
        state_masks=None,
    ) -> Tensor | dict[str, Tensor]:
        if self.config.use_rlt:
            if self.config.rlt_inference_only:
                raise ValueError("An encoder-only RLT export cannot be used for Stage 1 training.")
            if self.config.rlt_training_mode == "frozen_vla":
                with torch.no_grad():
                    prefix_output, prefix_padding, _ = self.build_prefix_cache(
                        images, img_masks, tokens, masks
                    )
                return {"rlt_loss": self._rlt_forward(prefix_output, prefix_padding, tokens)}
        x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)
        prefix, prefix_padding, prefix_ar = self.embed_prefix(images, img_masks, tokens, masks)
        suffix, suffix_padding, suffix_ar, condition = self.embed_suffix(x_t, model_time)
        padding = torch.cat([prefix_padding, suffix_padding], dim=1)
        autoregressive = torch.cat([prefix_ar, suffix_ar], dim=1)
        outputs, _ = self.llm(
            [prefix, suffix],
            positions=padding.long().cumsum(dim=1) - 1,
            mask=make_att_2d_masks(padding, autoregressive),
            adarms_cond=[None, condition],
            return_prefix=self.config.use_rlt,
            prefix_requires_grad=not self.config.use_rlt,
        )
        suffix_output = outputs[1][:, -self.config.chunk_size :].to(self.action_out_proj.weight.dtype)
        velocity = self.action_out_proj(suffix_output).float()
        action_losses = F.mse_loss(velocity, noise - actions, reduction="none")
        if self.config.use_rlt:
            rlt_prefix = outputs[0]
            if not self.config.rlt_use_mask:
                # Fully masked padding queries depend on the total attention
                # length. Without an RLT mask, use an observation-only pass so
                # padding features cannot contain training action information.
                with torch.no_grad():
                    rlt_prefix, prefix_padding, _ = self.build_prefix_cache(images, img_masks, tokens, masks)
            return {
                "vla_loss": action_losses,
                "rlt_loss": self._rlt_forward(rlt_prefix, prefix_padding, tokens),
            }
        return action_losses

    def _rlt_forward(self, prefix_output, prefix_padding, tokens, *, mode="loss"):
        if not self.config.use_rlt:
            raise ValueError("RLT features require policy.use_rlt=True.")
        if self.config.rlt_image_only:
            image_length = prefix_output.shape[1] - tokens.shape[1]
            prefix_output = prefix_output[:, :image_length]
            prefix_padding = prefix_padding[:, :image_length]
        parameter = next(self.rlt_module.parameters())
        prefix_output = prefix_output.detach().to(device=parameter.device, dtype=parameter.dtype)
        mask = prefix_padding if self.config.rlt_use_mask else None
        # Call the wrapper, rather than encode_flat/loss directly, so PEFT's
        # trained modules_to_save copy and FSDP hooks participate normally.
        return self.rlt_module(prefix_output, mask, mode=mode)

    @torch.no_grad()
    def extract_rlt_features(self, images, img_masks, tokens, masks):
        prefix_output, padding, _ = self.build_prefix_cache(images, img_masks, tokens, masks)
        return self._rlt_forward(prefix_output, padding, tokens, mode="encode")

    @torch.no_grad()
    def sample_actions_with_rlt_features(
        self, images, img_masks, tokens, masks, *, noise=None, num_steps=None,
        prev_chunk_left_over=None, inference_delay=0, execution_horizon=None,
    ):
        """Extract z and a reference chunk from one observation-only prefix pass.

        Optional guided RTC conditions the action expert on a committed prefix;
        z still represents the same boundary observation. Suffix regions can be compiled.
        """
        if not self.config.use_rlt:
            raise ValueError("RLT features require policy.use_rlt=True")
        guided = prev_chunk_left_over is not None
        if self._rtc_enabled() and self.rtc_processor.rtc_config.mode != "guided":
            raise ValueError("Shared RLT sampling supports guided RTC only")
        if guided and not self._rtc_enabled():
            raise ValueError("Configure guided RTC before supplying a committed prefix")
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        shape = (tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            noise = self.sample_noise(shape, tokens.device)
        if noise.shape != shape or not torch.isfinite(noise).all():
            raise ValueError("Invalid reference chunk noise")
        prefill = self._compiled_rlt_prefix if self.config.compile_model else self.build_prefix_cache
        denoise = self._compiled_denoise_step if self.config.compile_model else self.denoise_step
        prefix, padding, cache = prefill(images, img_masks, tokens, masks)
        z = self._rlt_forward(prefix, padding, tokens, mode="encode")
        actions = euler_integrate(
            lambda x_t, timestep: denoise(padding, cache, x_t, timestep), noise, num_steps,
            rtc_processor=self.rtc_processor, rtc_enabled=guided,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay, execution_horizon=execution_horizon,
        )
        return actions, z

    def build_prefix_cache(self, images, img_masks, tokens, masks):
        prefix, padding, autoregressive = self.embed_prefix(images, img_masks, tokens, masks)
        outputs, cache = self.llm(
            [prefix, None],
            positions=padding.long().cumsum(dim=1) - 1,
            mask=make_att_2d_masks(padding, autoregressive),
        )
        return outputs[0], padding, cache

    @torch.no_grad()
    def _prefill_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        """Adapt native KV tuples to the shared PI05 compiled sampler contract."""
        _, padding, cache = self.build_prefix_cache(images, img_masks, tokens, masks)
        return padding, cache

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix, padding, autoregressive, condition = self.embed_suffix(x_t, timestep)
        prefix_attention = prefix_pad_masks[:, None, :].expand(-1, padding.shape[1], -1)
        attention = torch.cat([prefix_attention, make_att_2d_masks(padding, autoregressive)], dim=-1)
        positions = prefix_pad_masks.long().sum(dim=-1)[:, None] + padding.long().cumsum(dim=1) - 1
        # Native RLinf KV tuples are immutable inputs: each suffix call creates
        # new concatenated tensors, so guided RTC can differentiate through it.
        outputs, _ = self.llm(
            [None, suffix],
            positions=positions,
            mask=attention,
            kv_cache=past_key_values,
            adarms_cond=[None, condition],
        )
        suffix_output = outputs[1][:, -self.config.chunk_size :].to(self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_output).float()
