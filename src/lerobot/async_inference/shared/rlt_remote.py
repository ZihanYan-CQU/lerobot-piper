"""Explicit RLT deployment protocol; ordinary deployment RTC remains unchanged."""

from dataclasses import dataclass

import torch

from .helpers import RemotePolicyConfig

RLT_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class RLTSetup:
    session: str
    policy: RemotePolicyConfig
    action_names: tuple[str, ...]
    fps: float
    execution_horizon: int = 10
    prefix_steps: int = 3
    version: int = RLT_PROTOCOL_VERSION
    client_image_resize: bool = True
    image_transport_mode: str = "lerobot_float32"


@dataclass(frozen=True)
class RLTStepReceipt:
    """One policy interval, after interpolation, using the controller's actual command."""

    action: torch.Tensor
    reward: float
    terminated: bool = False
    truncated: bool = False
    intervention: bool = False


@dataclass(frozen=True)
class RLTReceipt:
    window: int
    start_step: int
    policy_version: int
    steps: tuple[RLTStepReceipt, ...]


@dataclass(frozen=True)
class RLTRequest:
    session: str
    generation: int
    window: int
    start_step: int
    observation: dict
    committed_prefix: torch.Tensor | None = None
    previous: RLTReceipt | None = None
    final: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RLTPlan:
    session: str
    generation: int
    window: int
    start_step: int
    actions: torch.Tensor
    execution_horizon: int
    prefix_steps: int
    policy_version: int
    replay_spec: dict


@dataclass(frozen=True)
class RLTFinished:
    session: str
    generation: int
    window: int
    replay_windows: int


@dataclass(frozen=True)
class RLTFailure:
    message: str
