"""Opt-in RLT deployment settings, independent of ordinary RTC chunk refill settings."""

import math
from dataclasses import dataclass


def validate_window(window, prefix):
    if type(window) is not int or window < 2 or type(prefix) is not int or not 0 < prefix < window:
        raise ValueError("RLT requires 0 < prefix_steps < execution_horizon (default 3 < 10)")


@dataclass
class RLTServerConfig:
    enabled: bool = False
    execution_horizon: int = 10
    prefix_steps: int = 3
    actor_checkpoint: str | None = None
    warmstart_stage2_actor: bool = False
    actor_version: int = 0
    replay_dir: str = "outputs/rlt_rtc"
    replay_capacity: int = 100_000

    def __post_init__(self):
        validate_window(self.execution_horizon, self.prefix_steps)
        if type(self.actor_version) is not int or self.actor_version < 0:
            raise ValueError("RLT actor_version must be a nonnegative integer")
        if type(self.replay_capacity) is not int or self.replay_capacity < 1 or not self.replay_dir:
            raise ValueError("RLT requires a replay directory and positive ring capacity")


@dataclass
class RLTClientConfig:
    enabled: bool = False
    execution_horizon: int = 10
    prefix_steps: int = 3
    reward_provider: str | None = None
    budget_margin_s: float = 0.005

    def __post_init__(self):
        validate_window(self.execution_horizon, self.prefix_steps)
        if not math.isfinite(self.budget_margin_s) or self.budget_margin_s < 0:
            raise ValueError("RLT budget_margin_s must be finite and nonnegative")
        if self.enabled and not self.reward_provider:
            raise ValueError("RLT needs reward_provider='manual' or 'module:function' returning RewardSignal")
