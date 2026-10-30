"""Configuration for the standalone local PI05 deployment path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.policies.rtc import RTCConfig
from lerobot.robots import RobotConfig


@dataclass
class InferenceConfig:
    """Local model precision, compilation and image preparation settings."""

    device: str = "cuda:0"
    precision_profile: str = "openpi"
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    image_mode: str = "openpi_uint8"
    warmup_runs: int = 3
    # An optional local tokenizer directory overrides the saved tokenizer location.
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid or incompatible deployment settings before startup."""
        if self.precision_profile not in {"openpi", "lerobot", "checkpoint"}:
            raise ValueError("precision_profile must be openpi, lerobot or checkpoint")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("Unknown torch.compile mode")
        if self.image_mode not in {"openpi_uint8", "lerobot_float32"}:
            raise ValueError("image_mode must be openpi_uint8 or lerobot_float32")
        if not self.device or self.warmup_runs < 0:
            raise ValueError("A device and nonnegative warmup_runs are required")
        if self.compile_model and self.warmup_runs < 2:
            raise ValueError("Compiled deployment requires warmup_runs >= 2")

    def policy_overrides(self) -> list[str]:
        """Build model configuration overrides to apply before weight loading."""
        result = [
            f"--device={self.device}",
            f"--precision_profile={self.precision_profile}",
            f"--compile_model={str(self.compile_model).lower()}",
            f"--compile_mode={self.compile_mode}",
        ]
        if self.precision_profile != "checkpoint":
            result.append("--dtype=bfloat16")
        if self.tokenizer_path:
            result.append(f"--text_tokenizer_name={self.tokenizer_path}")
        return result


@dataclass(kw_only=True)
class LocalDeploymentConfig:
    """Robot, timing and lifecycle configuration for local deployment."""

    robot: RobotConfig
    pretrained_name_or_path: str
    task: str
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rtc: RTCConfig = field(default_factory=lambda: RTCConfig(execution_horizon=20))
    actions_per_chunk: int = 50
    fps: float = 30.0
    interpolation_multiplier: int = 2
    observation_capture_fps: float = 30.0
    observation_max_age_s: float = 0.1
    # Bounds source age at action dispatch, separately from input admission age.
    action_max_age_s: float = 2.0
    inference_timeout_s: float = 10.0
    startup_timeout_s: float = 1800.0
    stop_timeout_s: float = 3.0
    latency_quantile: float = 0.95
    latency_margin_steps: int = 2
    trained_delay_steps: int | None = None
    max_rejected_chunks: int = 5
    rename_map: dict[str, str] = field(default_factory=dict)
    # Explicit ordering overrides for checkpoints whose feature names are unavailable.
    action_keys: list[str] | None = None
    state_keys: list[str] | None = None
    reset_pose: dict[str, float] | None = None
    reset_to_initial_position: bool = True
    reset_open_gripper: bool = True
    reset_gripper_open_position: float | None = None
    reset_duration_s: float = 3.0
    reset_fps: float = 50.0
    reset_tolerance: float = 0.02
    reset_gripper_tolerance: float = 0.001
    reset_settle_timeout_s: float = 5.0
    reset_settle_s: float = 0.1
    autostart: bool = False
    duration_s: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid or incompatible deployment settings before startup."""
        if not self.pretrained_name_or_path or not self.task.strip():
            raise ValueError("pretrained_name_or_path and a nonempty task are required")
        for name in (
            "fps",
            "observation_capture_fps",
            "observation_max_age_s",
            "action_max_age_s",
            "inference_timeout_s",
            "startup_timeout_s",
            "stop_timeout_s",
            "reset_fps",
            "reset_tolerance",
            "reset_gripper_tolerance",
            "reset_settle_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.reset_duration_s) or self.reset_duration_s < 0:
            raise ValueError("reset_duration_s must be finite and nonnegative")
        if (
            not math.isfinite(self.reset_settle_s)
            or not 0 <= self.reset_settle_s < self.reset_settle_timeout_s
        ):
            raise ValueError("reset_settle_s must be finite, nonnegative and below reset_settle_timeout_s")
        piper = getattr(self.robot, "type", None) == "piper_follower"
        if piper and self.reset_gripper_open_position is None:
            self.reset_gripper_open_position = 0.07
        if self.reset_gripper_open_position is not None:
            if not math.isfinite(self.reset_gripper_open_position):
                raise ValueError("reset_gripper_open_position must be finite")
            if piper and not 0 < self.reset_gripper_open_position <= 0.1:
                raise ValueError("Piper reset_gripper_open_position must be in (0, 0.1] meters")
        if self.actions_per_chunk < 2 or self.interpolation_multiplier < 1:
            raise ValueError("actions_per_chunk >= 2 and interpolation_multiplier >= 1 are required")
        if not self.rtc.enabled or self.rtc.execution_horizon < 1 or self.rtc.debug:
            raise ValueError("Local deployment requires enabled RTC, a positive horizon and debug=false")
        if not math.isfinite(self.rtc.max_guidance_weight):
            raise ValueError("RTC guidance weight must be finite")
        if not 0 < self.latency_quantile <= 1 or self.latency_margin_steps < 0:
            raise ValueError("Invalid latency quantile or margin")
        if self.max_rejected_chunks < 1:
            raise ValueError("max_rejected_chunks must be positive")
        if self.trained_delay_steps is not None and self.trained_delay_steps < 1:
            raise ValueError("trained_delay_steps must be positive")
        if self.rtc.mode != "trained" and self.trained_delay_steps is not None:
            raise ValueError("trained_delay_steps requires rtc.mode=trained")
        if self.action_max_age_s < self.observation_max_age_s:
            raise ValueError("action_max_age_s must cover observation_max_age_s")
        if self.duration_s is not None and (not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError("duration_s must be finite and positive")
        for names in (self.action_keys, self.state_keys):
            if names is not None and (not names or len(set(names)) != len(names)):
                raise ValueError("Feature ordering must be nonempty and contain no duplicates")
        if self.reset_pose is not None and not all(math.isfinite(v) for v in self.reset_pose.values()):
            raise ValueError("reset_pose must contain finite positions")
