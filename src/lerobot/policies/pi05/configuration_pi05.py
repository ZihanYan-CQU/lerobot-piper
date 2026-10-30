#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    # Select the network implementation; both use the same LeRobot processors and RTC API.
    backend: str = "lerobot"  # "lerobot" or "rlinf"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"
    # Training keeps LeRobot's precision layout by default. Deployment's
    # "checkpoint" profile instead applies dtype to every floating VLA parameter.
    # LeRobot keeps visual weights FP32 and uses BF16 CUDA vision autocast in eval.
    # OpenPI keeps visual embeddings and Gemma norms in FP32 under BF16.
    precision_profile: str = "lerobot"  # Options: "checkpoint", "lerobot", "openpi"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # MEM short-horizon observation memory (https://arxiv.org/abs/2603.03596).
    # Historical image tokens are fused inside SigLIP and dropped before the
    # language backbone. Historical proprioceptive states become one continuous
    # backbone token per frame. Both paths are opt-in and independent.
    #
    # MEM pre-trains on six observations spaced one second apart. `memory_stride` is
    # counted in dataset frames, so the default matches that spacing only at 30 fps,
    # the usual LeRobot recording rate. Scale it with the dataset: a 10 fps dataset
    # such as `lerobot/robomme` needs `memory_stride=10` for the same one second.
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None
    # Maximum clean action-prefix length sampled during training. Zero disables trained RTC.
    rtc_training_max_delay: int = 0

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    # Keep the existing PaliGemma default; None selects a bundled tokenizer
    # when loading processors, or the same default for new processors.
    text_tokenizer_name: str | None = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    # Exclude episode-end action padding from training loss. Opt in when fine-tuning
    # on correction segments; keep older checkpoints' loss semantics by default.
    mask_action_padding_loss: bool = False
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections
    # PEFT defaults; explicit --peft.* targets can still override this recipe.
    peft_mode: str = "default"  # "default" or "vlm_lora_expert_full"
    # Independent scopes for the mixed recipe: "none", "attention", "attention_mlp".
    peft_language_lora: str = "attention_mlp"
    peft_vision_lora: str = "attention_mlp"

    # RL Token Stage 1. RLT reads detached, final-layer VLM prefix features.
    use_rlt: bool = False
    rlt_training_mode: str = "frozen_vla"  # "frozen_vla" or "joint"
    rlt_alpha: float = 1.0  # Joint loss = RLT reconstruction + alpha * action loss.
    rlt_embed_dim: int = 2048
    rlt_prefix_seq_len: int = 1024
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = False
    rlt_use_mask: bool = True
    # Encoder-only exports can extract features and sample actions, but cannot train RLT.
    rlt_inference_only: bool = False

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.backend not in {"lerobot", "rlinf"}:
            raise ValueError(f"Invalid PI05 backend: {self.backend!r}. Expected 'lerobot' or 'rlinf'.")
        if self.backend == "rlinf":
            if self.use_visual_memory or self.use_proprioceptive_memory:
                raise ValueError("The RLinf backend does not yet support MEM; use backend='lerobot' for MEM.")
            if self.image_resolution != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
                raise ValueError("The RLinf backend requires image_resolution=(224, 224).")

        if self.use_rlt and self.backend != "rlinf":
            raise ValueError("RLT Stage 1 requires policy.backend='rlinf'.")
        if self.rlt_inference_only and not self.use_rlt:
            raise ValueError("rlt_inference_only=True requires use_rlt=True.")
        if self.rlt_training_mode not in {"frozen_vla", "joint"}:
            raise ValueError("rlt_training_mode must be 'frozen_vla' or 'joint'.")
        if not math.isfinite(self.rlt_alpha) or self.rlt_alpha < 0:
            raise ValueError("rlt_alpha must be finite and nonnegative.")
        for name in ("rlt_embed_dim", "rlt_prefix_seq_len", "rlt_num_layers", "rlt_num_heads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.rlt_embed_dim % 2 or self.rlt_embed_dim % self.rlt_num_heads:
            raise ValueError("rlt_embed_dim must be even and divisible by rlt_num_heads.")
        if not math.isfinite(self.rlt_mlp_ratio) or self.rlt_mlp_ratio <= 0:
            raise ValueError("rlt_mlp_ratio must be finite and positive.")
        if int(self.rlt_embed_dim * self.rlt_mlp_ratio) < 1:
            raise ValueError("rlt_embed_dim * rlt_mlp_ratio must be at least 1.")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.precision_profile not in {"checkpoint", "lerobot", "openpi"}:
            raise ValueError(f"Invalid precision_profile: {self.precision_profile}")

        if self.peft_mode not in {"default", "vlm_lora_expert_full"}:
            raise ValueError(f"Invalid PI05 peft_mode: {self.peft_mode!r}.")
        for name in ("peft_language_lora", "peft_vision_lora"):
            if getattr(self, name) not in {"none", "attention", "attention_mlp"}:
                raise ValueError(
                    f"Invalid PI05 {name}: {getattr(self, name)!r}. "
                    "Expected 'none', 'attention' or 'attention_mlp'."
                )

        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int] | None:
        if not self.use_visual_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        if not self.use_proprioceptive_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
