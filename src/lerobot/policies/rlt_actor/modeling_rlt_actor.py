"""Small RLT actor; the frozen VLA is owned separately by the rollout adapter."""

from collections import deque
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from .configuration_rlt_actor import OBS_REFERENCE, OBS_RLT, OBS_RTC_MASK, OBS_RTC_PREFIX, RLTActorConfig


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def validate_observation(
    obs: dict[str, Tensor], cfg: RLTActorConfig, *, batched: bool, strict: bool = False
) -> None:
    expected = {
        OBS_RLT: (cfg.z_dim,),
        OBS_STATE: (cfg.proprio_dim,),
        OBS_REFERENCE: (cfg.reference_horizon, cfg.action_dim),
    }
    if cfg.rtc_prefix_steps:
        expected[OBS_RTC_PREFIX] = (cfg.execution_horizon, cfg.action_dim)
        expected[OBS_RTC_MASK] = (cfg.execution_horizon,)
    batch_size = None
    if not set(expected).issubset(obs) or (strict and set(obs) != set(expected)):
        raise ValueError(f"RLT observations must contain {sorted(expected)}")
    for key, shape in expected.items():
        value = obs[key]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{key} must be a floating tensor")
        if (
            value.ndim != len(shape) + int(batched)
            or tuple(value.shape[int(batched) :]) != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid RLT observation {key}: expected {'(batch, ...)' if batched else shape}"
            )
        if batched:
            if value.shape[0] < 1 or (batch_size is not None and value.shape[0] != batch_size):
                raise ValueError("Inconsistent or empty RLT observation batch")
            batch_size = value.shape[0]

    if cfg.rtc_prefix_steps:
        prefix = obs[OBS_RTC_PREFIX]
        mask = obs[OBS_RTC_MASK]
        if not batched:
            prefix, mask = prefix[None], mask[None]
        count = mask.sum(-1)
        positions = torch.arange(cfg.execution_horizon, device=mask.device)[None]
        if (
            not ((mask == 0) | (mask == 1)).all()
            or not ((count == 0) | (count == cfg.rtc_prefix_steps)).all()
            or not torch.equal(mask.bool(), positions < count[:, None])
            or prefix[~mask.bool()].count_nonzero()
        ):
            raise ValueError("RTC state requires an empty or complete committed prefix and zero tail")


class RLTActorPolicy(PreTrainedPolicy):
    """Tanh MLP with fixed Gaussian exploration before the tanh, without entropy tuning."""

    config_class = RLTActorConfig
    name = "rlt_actor"

    def __init__(self, config: RLTActorConfig, **kwargs):
        super().__init__(config)
        self.actor = mlp(
            config.z_dim
            + config.proprio_dim
            + config.execution_horizon * config.action_dim
            + (config.execution_horizon * (config.action_dim + 1) if config.rtc_prefix_steps else 0),
            config.hidden_dims,
            config.execution_horizon * config.action_dim,
        )
        self._action_queue = deque()
        self._rollout_version = -1

    def get_optim_params(self):
        return self.actor.parameters()

    def reset(self):
        self._action_queue.clear()

    @property
    def rollout_version(self) -> int:
        return self._rollout_version

    def load_rollout_weights(self, payload: dict) -> None:
        """Validate a learner snapshot before replacing any actor parameters.

        The rollout owner must call this at a window boundary. Tensors are copied
        in place, retaining the actor's device and its optimizer references.
        """
        if payload.get("feature_model_id") != self.config.feature_model_id:
            raise ValueError("Actor weights belong to a different frozen feature model")
        from lerobot.rl.rlt.replay import replay_spec

        spec = payload.get("replay_spec")
        if (self.config.rtc_prefix_steps or spec is not None) and spec != replay_spec(self.config):
            raise ValueError("Actor snapshot belongs to a different window/RTC replay contract")
        version = payload.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("Actor version must be a nonnegative integer")
        if version < self._rollout_version:
            raise ValueError("Actor snapshot is older than the active rollout version")
        incoming, current = payload.get("policy"), self.state_dict()
        if not isinstance(incoming, Mapping) or set(incoming) != set(current):
            raise ValueError("Incomplete RLT actor snapshot")
        for key, target in current.items():
            value = incoming[key]
            if (
                not isinstance(value, Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid RLT actor tensor {key}")
        self.load_state_dict(incoming, strict=True)
        self.reset()
        self._rollout_version = version

    def forward(
        self, batch: dict[str, Tensor], *, deterministic: bool = False, reference_dropout_prob: float = 0.0
    ) -> Tensor:
        validate_observation(batch, self.config, batched=True)
        if not 0 <= reference_dropout_prob <= 1:
            raise ValueError("reference_dropout_prob must be in [0, 1]")
        parameter = next(self.actor.parameters())
        reference = batch[OBS_REFERENCE][:, : self.config.execution_horizon].flatten(1).detach()
        reference = reference.to(parameter)
        if reference_dropout_prob:
            keep = torch.rand(reference.shape[0], 1, device=reference.device) >= reference_dropout_prob
            reference = reference * keep
        pieces = [reference, batch[OBS_RLT].detach().to(parameter), batch[OBS_STATE].detach().to(parameter)]
        if self.config.rtc_prefix_steps:
            pieces.extend(
                [
                    batch[OBS_RTC_PREFIX].detach().to(parameter).flatten(1),
                    batch[OBS_RTC_MASK].detach().to(parameter),
                ]
            )
        inputs = torch.cat(pieces, dim=-1)
        mean = self.actor(inputs)
        if not deterministic and self.config.exploration_std:
            mean = mean + torch.randn_like(mean) * self.config.exploration_std
        actions = mean.tanh().reshape(-1, self.config.execution_horizon, self.config.action_dim)
        if self.config.rtc_prefix_steps:
            actions = torch.where(
                batch[OBS_RTC_MASK].to(parameter.device).bool()[..., None],
                batch[OBS_RTC_PREFIX].detach().to(parameter),
                actions,
            )
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self(batch, deterministic=kwargs.pop("deterministic", True), **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self._action_queue:
            self._action_queue.extend(self.predict_action_chunk(batch, **kwargs).transpose(0, 1))
        return self._action_queue.popleft()
