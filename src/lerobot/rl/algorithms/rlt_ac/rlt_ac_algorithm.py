"""RLT Stage 2: fixed-std tanh actor, twin target Q, and reference/human BC.

The objective follows RLinf revision 061c4cce80bbab998ef90585475d3f51290a2dfa,
without its distributed worker dependency. Partial-window masks and replay
feature-version checks are local extensions. There is no entropy/alpha update.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from lerobot.policies.rlt_actor.configuration_rlt_actor import (
    OBS_REFERENCE,
    OBS_RLT,
    OBS_RTC_MASK,
    OBS_RTC_PREFIX,
)
from lerobot.policies.rlt_actor.modeling_rlt_actor import RLTActorPolicy, mlp, validate_observation
from lerobot.rl.rlt.replay import feature_digest, replay_spec
from lerobot.utils.constants import OBS_STATE

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_rlt_ac import RLTACConfig


class RLTQEnsemble(nn.Module):
    """Value a complete K-step plan; the future realized duration is never an input."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        size = config.z_dim + config.proprio_dim + config.execution_horizon * config.action_dim
        if config.rtc_prefix_steps:
            # The frozen reference determines the next committed continuation.
            size += config.reference_horizon * config.action_dim + config.execution_horizon * (
                config.action_dim + 1
            )
        self.heads = nn.ModuleList([mlp(size, config.hidden_dims, 1) for _ in range(2)])

    def forward(self, state: dict[str, Tensor], actions: Tensor) -> Tensor:
        pieces = [
            state[OBS_RLT].detach(),
            state[OBS_STATE].detach(),
            actions.flatten(1),
        ]
        if self.config.rtc_prefix_steps:
            pieces.extend(
                [
                    state[OBS_REFERENCE].detach().flatten(1),
                    state[OBS_RTC_PREFIX].detach().flatten(1),
                    state[OBS_RTC_MASK].detach(),
                ]
            )
        x = torch.cat(pieces, dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)


class RLTACAlgorithm(RLAlgorithm):
    config_class = RLTACConfig
    name = "rlt_ac"

    def __init__(self, policy: RLTActorPolicy, config: RLTACConfig):
        if not isinstance(policy, RLTActorPolicy) or config.policy_config != policy.config:
            raise ValueError("RLT AC requires a matching RLTActorPolicy and policy_config")
        self.config, self.policy = config, policy
        self.device = torch.device(policy.config.device)
        self.policy.to(device=self.device, dtype=torch.float32)
        self.critic = RLTQEnsemble(policy.config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.optimizers = {}
        self.optimization_step = 0

    def make_optimizers_and_scheduler(self):
        # RLTrainer calls this too; never replace optimizers restored from a checkpoint.
        if not self.optimizers:
            self.optimizers = {
                "actor": torch.optim.Adam(self.policy.parameters(), lr=self.config.actor_lr),
                "critic": torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr),
            }
        return self.optimizers

    def get_optimizers(self):
        return self.optimizers

    def _prepare_batch(self, batch: dict) -> dict:
        cfg = self.policy.config
        validate_observation(batch["state"], cfg, batched=True, strict=True)
        validate_observation(batch["next_state"], cfg, batched=True, strict=True)
        b = batch["action"].shape[0]
        k, d = cfg.execution_horizon, cfg.action_dim
        if batch["action"].shape != (b, k, d) or batch["reward"].shape != (b, k):
            raise ValueError("RLT replay requires fixed-window action and per-step reward tensors")
        if batch["state"][OBS_RLT].shape[0] != b or batch["next_state"][OBS_RLT].shape[0] != b:
            raise ValueError("RLT state and action batch sizes differ")
        for key in ("action", "reward"):
            if not batch[key].is_floating_point() or not torch.isfinite(batch[key]).all():
                raise ValueError(f"Replay {key} must be finite floating tensors")
        for key in ("done", "truncated"):
            if batch[key].shape != (b,) or batch[key].dtype != torch.bool:
                raise ValueError(f"Replay {key} must be boolean with shape (batch,)")
        info = batch.get("complementary_info")
        if not info or any(key not in info for key in ("valid_mask", "intervention", "feature_id")):
            raise ValueError("RLT replay requires valid_mask, intervention and feature_id")
        valid, human = info["valid_mask"], info["intervention"]
        if (
            valid.shape != (b, k)
            or valid.dtype != torch.bool
            or human.shape != valid.shape
            or human.dtype != torch.bool
        ):
            raise ValueError("Invalid fixed-window masks")
        steps = valid.sum(-1)
        prefix = torch.arange(k, device=valid.device)[None] < steps[:, None]
        if not torch.equal(prefix, valid) or (steps < 1).any():
            raise ValueError("valid_mask must be a nonempty contiguous executed prefix")
        if ((steps < k) & ~(batch["done"].to(valid.device) | batch["truncated"].to(valid.device))).any():
            raise ValueError("Partial replay windows require a terminal/truncated episode")
        expected = feature_digest(cfg.feature_model_id).to(info["feature_id"].device)
        if (
            info["feature_id"].dtype != torch.uint8
            or info["feature_id"].shape != (b, 32)
            or not (info["feature_id"] == expected).all()
        ):
            raise ValueError("Replay uses different frozen features, normalization or action coordinates")
        if (
            batch["action"].to(valid.device)[~valid].count_nonzero()
            or batch["reward"].to(valid.device)[~valid].count_nonzero()
            or (human & ~valid).any()
        ):
            raise ValueError("Unexecuted replay tail must be zero padded")
        planned = info.get("planned_action")
        if planned is None:
            if (steps < k).any():
                raise ValueError("Partial RLT replay requires complete planned_action for critic training")
            planned = batch["action"]
        if (
            not isinstance(planned, Tensor)
            or planned.shape != (b, k, d)
            or not planned.is_floating_point()
            or not torch.isfinite(planned).all()
        ):
            raise ValueError("Invalid complete planned_action in RLT replay")
        # Preserve actual sent rows, including clipping/intervention. A terminal
        # tail is planned context only; it is never relabelled as executed data.
        action = batch["action"].detach().to(self.device, torch.float32)
        valid = valid.to(self.device)
        planned = action if planned is batch["action"] else planned.detach().to(self.device, torch.float32)
        critic_action = torch.where(
            valid[..., None],
            action,
            planned,
        )
        return {
            "state": {
                key: value.detach().to(self.device, torch.float32) for key, value in batch["state"].items()
            },
            "next_state": {
                key: value.detach().to(self.device, torch.float32)
                for key, value in batch["next_state"].items()
            },
            "action": action,
            "critic_action": critic_action,
            "reward": batch["reward"].detach().to(self.device, torch.float32),
            "done": batch["done"].to(self.device),
            "valid": valid,
            "human": human.to(self.device),
        }

    @torch.no_grad()
    def target_values(self, batch: dict) -> Tensor:
        """Semi-Markov backup: sum gamma**i*r_i + (1-terminated)*gamma**n*Q'."""
        k = self.policy.config.execution_horizon
        discount = torch.tensor(self.config.discount, device=self.device, dtype=torch.float32)
        powers = discount.pow(torch.arange(k, device=self.device))
        reward = (batch["reward"] * batch["valid"] * powers).sum(-1)
        next_action = self.policy(batch["next_state"], deterministic=True)
        next_q = self.critic_target(batch["next_state"], next_action).min(-1).values
        bootstrap = discount.pow(batch["valid"].sum(-1)) * (~batch["done"]).float()
        return reward + bootstrap * next_q

    def actor_loss(self, batch: dict) -> tuple[Tensor, dict]:
        actions = self.policy(batch["state"], reference_dropout_prob=self.config.reference_dropout_prob)
        q = self.critic(batch["state"], actions)[:, 0]
        reference = batch["state"][OBS_REFERENCE][:, : self.policy.config.execution_horizon]
        target = torch.where(batch["human"][..., None], batch["action"], reference)
        error = (actions - target).square().mean(-1)
        valid = batch["valid"]
        if self.policy.config.rtc_prefix_steps:
            valid = valid & ~batch["state"][OBS_RTC_MASK].bool()
        bc = (error * valid).sum() / valid.sum().clamp_min(1)
        human = batch["human"] & valid
        ordinary = ~batch["human"] & valid
        metrics = {
            "bc_loss": bc.detach().item(),
            "bc_human_loss": (error * human).sum().detach().item() / max(human.sum().item(), 1),
            "bc_ref_loss": (error * ordinary).sum().detach().item() / max(ordinary.sum().item(), 1),
            "human_ratio": human.sum().item() / max(valid.sum().item(), 1),
            "q_pi": q.detach().mean().item(),
        }
        free_windows = valid.any(-1)
        q_loss = (q * free_windows).sum() / free_windows.sum().clamp_min(1)
        return -self.config.q_weight * q_loss + self.config.bc_weight * bc, metrics

    def _step(self, loss: Tensor, module: nn.Module, optimizer) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RLT loss; optimizer update rejected")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        maximum = self.config.grad_clip_norm or float("inf")
        norm = nn.utils.clip_grad_norm_(module.parameters(), maximum, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return norm.item()

    def update(self, batch_iterator: Iterator[dict]) -> TrainingStats:
        self.make_optimizers_and_scheduler()
        batch = self._prepare_batch(next(batch_iterator))
        stats = TrainingStats()
        # Small RL heads stay FP32, independently of any outer VLA AMP context.
        with torch.autocast(self.device.type, enabled=False):
            target = self.target_values(batch)
            q = self.critic(batch["state"], batch["critic_action"])
            loss = (q - target[:, None]).square().mean()
            stats.grad_norms["critic"] = self._step(loss, self.critic, self.optimizers["critic"])
            stats.losses["critic_loss"] = loss.item()
            free = batch["valid"]
            if self.policy.config.rtc_prefix_steps:
                free = free & ~batch["state"][OBS_RTC_MASK].bool()
            if (self.optimization_step + 1) % self.config.actor_update_freq == 0 and free.any():
                self.critic.requires_grad_(False)
                try:
                    actor_loss, metrics = self.actor_loss(batch)
                    stats.grad_norms["actor"] = self._step(actor_loss, self.policy, self.optimizers["actor"])
                    stats.losses["actor_loss"] = actor_loss.item()
                    stats.extra.update(metrics)
                finally:
                    self.critic.requires_grad_(True)
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    self.critic_target.parameters(), self.critic.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, self.config.target_update_weight)
        self.optimization_step += 1
        stats.extra.update(
            {"target_q": target.mean().item(), "executed_steps": batch["valid"].sum(-1).float().mean().item()}
        )
        return stats

    def get_weights(self):
        return {
            "policy": {key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()},
            "version": self.optimization_step,
            "feature_model_id": self.policy.config.feature_model_id,
            "replay_spec": replay_spec(self.policy.config),
        }

    def load_weights(self, weights, device="cpu"):
        self.policy.load_rollout_weights(weights)

    def state_dict(self):
        return {
            **{"critic." + key: value for key, value in self.critic.state_dict().items()},
            **{"critic_target." + key: value for key, value in self.critic_target.state_dict().items()},
        }

    def load_state_dict(self, state_dict, device="cpu"):
        expected = set(self.state_dict())
        if set(state_dict) != expected:
            raise ValueError("Incomplete RLT critic/target checkpoint")
        for prefix, module in (("critic.", self.critic), ("critic_target.", self.critic_target)):
            module.load_state_dict(
                {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)},
                strict=True,
            )
            module.to(device)
