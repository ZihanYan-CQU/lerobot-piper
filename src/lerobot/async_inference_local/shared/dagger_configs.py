"""Configuration for local inference with correction-only DAgger capture."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from lerobot.light import StatusIndicatorConfig

from ..control.piper_dagger import PiperDaggerHardwareConfig
from ..recording.correction import DaggerDatasetConfig
from .configs import LocalDeploymentConfig


@dataclass(kw_only=True)
class LocalDaggerConfig(LocalDeploymentConfig):
    """Keep model settings shared with local deployment and capture settings separate."""

    hardware: PiperDaggerHardwareConfig = field(default_factory=PiperDaggerHardwareConfig)
    dataset: DaggerDatasetConfig = field(default_factory=DaggerDatasetConfig)
    input_device: str = "keyboard"
    advance_key: str = "right"
    reset_key: str = "left"
    stop_key: str = "esc"
    pedal_device_path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    pedal_advance: str = "KEY_A"
    pedal_reset: str = "KEY_B"
    status_indicator: StatusIndicatorConfig = field(default_factory=StatusIndicatorConfig)
    num_episodes: int = 0
    save_partial_on_stop: bool = False

    def __post_init__(self) -> None:
        """Validate hardware routing and resolve recording fps before opening devices."""
        super().__post_init__()
        if getattr(self.robot, "type", None) != "piper_follower":
            raise ValueError("Local DAgger currently requires robot.type=piper_follower")
        # The current common Piper driver creates its bus with this fixed channel.
        if self.hardware.follower_can_name != "can_follower":
            raise ValueError("The current Piper follower driver uses can_follower")
        if self.dataset.fps == 0:
            if not float(self.fps).is_integer():
                raise ValueError("Set an integer dataset.fps when policy fps is fractional")
            self.dataset = replace(self.dataset, fps=int(self.fps))
        if self.input_device not in {"keyboard", "pedal", "prompt"}:
            raise ValueError("input_device must be keyboard, pedal or prompt")
        keys = (self.advance_key, self.reset_key, self.stop_key)
        if any(not key for key in keys) or len(set(keys)) != 3:
            raise ValueError("advance_key, reset_key and stop_key must be nonempty and distinct")
        if not self.pedal_advance or not self.pedal_reset or self.pedal_advance == self.pedal_reset:
            raise ValueError("pedal_advance and pedal_reset must be nonempty and distinct")
        if self.num_episodes < 0:
            raise ValueError("num_episodes must be nonnegative (0 means unlimited)")
        if self.num_episodes and self.dataset.repo_id is None:
            raise ValueError("num_episodes requires dataset.repo_id")
