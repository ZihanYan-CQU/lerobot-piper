"""RLT's Q + reference/human BC objective, with per-policy-step discounting."""

import math
from dataclasses import dataclass

from lerobot.policies.rlt_actor.configuration_rlt_actor import RLTActorConfig

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("rlt_ac")
@dataclass
class RLTACConfig(RLAlgorithmConfig):
    policy_config: RLTActorConfig | None = None
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.99
    target_update_weight: float = 0.005
    q_weight: float = 0.1
    bc_weight: float = 5.0
    reference_dropout_prob: float = 0.5
    actor_update_freq: int = 1
    grad_clip_norm: float = 10.0

    def __post_init__(self):
        for key in ("actor_lr", "critic_lr", "discount", "target_update_weight"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.discount > 1 or self.target_update_weight > 1:
            raise ValueError("discount and target_update_weight must be <= 1")
        for key in ("q_weight", "bc_weight", "grad_clip_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.q_weight + self.bc_weight == 0:
            raise ValueError("At least one actor objective weight must be positive")
        if not 0 <= self.reference_dropout_prob <= 1:
            raise ValueError("reference_dropout_prob must be in [0, 1]")
        if (
            isinstance(self.actor_update_freq, bool)
            or not isinstance(self.actor_update_freq, int)
            or self.actor_update_freq < 1
        ):
            raise ValueError("actor_update_freq must be a positive integer")

    @classmethod
    def from_policy_config(cls, policy_cfg: RLTActorConfig):
        return cls(policy_config=policy_cfg)
