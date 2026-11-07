"""Shape and feature-version contract for RLT Stage 2."""

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE

OBS_RLT = "observation.rlt"
OBS_REFERENCE = "observation.reference_chunk"
OBS_RTC_PREFIX = "observation.rtc_prefix"
OBS_RTC_MASK = "observation.rtc_prefix_mask"


@PreTrainedConfig.register_subclass("rlt_actor")
@dataclass
class RLTActorConfig(PreTrainedConfig):
    """Actions and reference chunks use the frozen VLA's normalized action coordinates."""

    feature_model_id: str = ""
    z_dim: int = 2048
    proprio_dim: int = 7
    action_dim: int = 7
    execution_horizon: int = 10
    reference_horizon: int = 50
    rtc_prefix_steps: int = 0
    hidden_dims: tuple[int, ...] = (256, 256)
    exploration_std: float = 0.002
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.feature_model_id.strip():
            raise ValueError("feature_model_id must identify an immutable Stage 1 checkpoint and processors")
        for key in ("z_dim", "proprio_dim", "action_dim", "execution_horizon", "reference_horizon"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.reference_horizon < self.execution_horizon:
            raise ValueError("reference_horizon must cover execution_horizon")
        if (
            isinstance(self.rtc_prefix_steps, bool)
            or not isinstance(self.rtc_prefix_steps, int)
            or not 0 <= self.rtc_prefix_steps < self.execution_horizon
        ):
            raise ValueError("rtc_prefix_steps must be an integer in [0, execution_horizon)")
        if self.reference_horizon < self.execution_horizon + self.rtc_prefix_steps:
            raise ValueError("RTC needs a reference continuation after the complete execution window")
        if not self.hidden_dims or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if not math.isfinite(self.exploration_std) or self.exploration_std < 0:
            raise ValueError("exploration_std must be finite and nonnegative")
        if self.n_obs_steps != 1 or self.use_peft:
            raise ValueError("RLT actor requires n_obs_steps=1 and use_peft=False")
        if any(mode != NormalizationMode.IDENTITY for mode in self.normalization_mapping.values()):
            raise ValueError("RLT actor inputs/actions are already normalized by the frozen VLA")
        self.validate_features()

    def validate_features(self):
        expected = {
            OBS_RLT: PolicyFeature(FeatureType.STATE, (self.z_dim,)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (self.proprio_dim,)),
            OBS_REFERENCE: PolicyFeature(FeatureType.STATE, (self.reference_horizon, self.action_dim)),
        }
        if self.rtc_prefix_steps:
            expected[OBS_RTC_PREFIX] = PolicyFeature(
                FeatureType.STATE, (self.execution_horizon, self.action_dim)
            )
            expected[OBS_RTC_MASK] = PolicyFeature(FeatureType.STATE, (self.execution_horizon,))
        if self.input_features and self.input_features != expected:
            raise ValueError("RLT actor input features must match its z/proprio/reference dimensions")
        output = {ACTION: PolicyFeature(FeatureType.ACTION, (self.action_dim,))}
        if self.output_features and self.output_features != output:
            raise ValueError("RLT actor output features must match action_dim")
        self.input_features = expected
        self.output_features = output

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return list(range(self.execution_horizon))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(lr=3e-4)

    def get_scheduler_preset(self):
        return None
