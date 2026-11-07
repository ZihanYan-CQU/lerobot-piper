"""Frozen PI05 feature/reference adapter and reversible executed-action coordinates."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.rlt_actor.configuration_rlt_actor import (
    OBS_REFERENCE,
    OBS_RLT,
    OBS_RTC_MASK,
    OBS_RTC_PREFIX,
    RLTActorConfig,
)
from lerobot.policies.rlt_actor.modeling_rlt_actor import validate_observation
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)
from lerobot.utils.constants import OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME

from .replay import owned_state

_FEATURE_FORMAT = "rlt-features-v2-bundled-tokenizer"


def _bundled_tokenizer_dirs(processor_config: Path) -> list[Path]:
    """Require saved tokenizer artifacts before constructing any tokenizer.

    A policy's original text_tokenizer_name can reference an external mutable
    directory. Frozen features must use only assets saved with the checkpoint.
    """
    saved = json.loads(processor_config.read_text())
    directories = []
    for step in saved["steps"]:
        if (
            step.get("registry_name") != "tokenizer_processor"
            and step.get("class", "").rsplit(".", 1)[-1] != "TokenizerProcessorStep"
        ):
            continue
        artifact = step.get("artifacts", {}).get("tokenizer_name")
        if not isinstance(artifact, str):
            raise ValueError("Frozen RLT features require a bundled tokenizer; re-export Stage 1 processors")
        path = Path(artifact)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise ValueError("Frozen RLT tokenizer artifacts must be inside the Stage 1 checkpoint")
        directories.append(path)
    return directories


def checkpoint_fingerprint(checkpoint, *, revision: str | None = None) -> str:
    """Identify weights, normalization and tokenizer assets before collecting replay.

    Local assets include every file in the saved tokenizer directories. Stage 2
    loads these bundles instead of external tokenizer overrides. The v2 namespace
    rejects replay collected under the previous, incomplete identity contract.
    Hub checkpoints require a pinned commit, which identifies all their assets.
    """
    directory = Path(checkpoint).expanduser()
    if not directory.is_dir():
        if not revision or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("Use a local Stage 1 directory or pin the Hub revision to a full commit")
        return f"hub:{checkpoint}@{revision.lower()}#{_FEATURE_FORMAT}"
    tokenizer_files = set()
    processor_config = directory / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    if processor_config.is_file():
        for relative in _bundled_tokenizer_dirs(processor_config):
            artifact = directory / relative
            if not artifact.is_dir():
                raise FileNotFoundError(
                    f"Missing bundled tokenizer: {artifact}; re-export Stage 1 processors"
                )
            entries = [artifact, *artifact.rglob("*")]
            ancestors = [directory / parent for parent in relative.parents if parent.parts]
            if any(path.is_symlink() and path.is_dir() for path in [*entries, *ancestors]):
                raise ValueError("Copy tokenizer directories into the checkpoint instead of symlinking them")
            tokenizer_files.update(path for path in entries if path.is_file())
    assets = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and (path.suffix in {".json", ".safetensors", ".model", ".txt"} or path in tokenizer_files)
    )
    if not (directory / "config.json").is_file() or not any(path.suffix == ".safetensors" for path in assets):
        raise ValueError("Stage 1 directory must contain its config and weights")
    digest = hashlib.sha256(_FEATURE_FORMAT.encode())
    for path in assets:
        name = path.relative_to(directory).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ActionCoordinates:
    """An owned snapshot; later observations cannot change an earlier window's anchor."""

    action_dim: int
    anchor: torch.Tensor | None
    mask: object
    normalizer: object
    postprocessor: object
    device: torch.device

    @torch.no_grad()
    def normalize(self, absolute_action):
        action = torch.as_tensor(absolute_action).detach().to(self.device, torch.float32)
        if action.shape != (self.action_dim,) or not torch.isfinite(action).all():
            raise ValueError("Executed action must be a finite action_dim vector")
        action = action[None]
        if self.anchor is not None:
            action = to_relative_actions(action, self.anchor, self.mask)
        transition = create_transition(action=action)
        if self.normalizer is not None:
            transition = self.normalizer(transition)
        result = transition[TransitionKey.ACTION][0].detach().float().cpu().clone()
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite normalized executed action")
        return result

    @torch.no_grad()
    def decode(self, normalized):
        normalized = torch.as_tensor(normalized)
        if (
            normalized.ndim != 2
            or normalized.shape[1] != self.action_dim
            or not torch.isfinite(normalized).all()
        ):
            raise ValueError("Actions must be a finite (steps, action_dim) tensor")
        actions = self.postprocessor(normalized.to(self.device)[None])[0].detach().float().cpu().clone()
        if actions.shape != normalized.shape or not torch.isfinite(actions).all():
            raise ValueError("Postprocessed actions are invalid")
        return actions


class FrozenRLTFeatures:
    """One owner processes observations, generates references, and anchors a full window.

    feature_model_id is an explicit immutable identifier for BOTH Stage 1
    weights and processors. Never reuse that identifier after changing either.
    """

    def __init__(self, policy, preprocessor, postprocessor, config: RLTActorConfig):
        source = policy.config
        if source.backend != "rlinf" or not source.use_rlt or source.use_peft:
            raise ValueError("Stage 2 requires a full/merged RLinf PI05 RLT checkpoint")
        if source.rtc_config is not None and source.rtc_config.enabled and not config.rtc_prefix_steps:
            raise ValueError("Fixed-window RLT requires live RTC prefix splicing disabled")
        if (
            source.rlt_embed_dim != config.z_dim
            or source.chunk_size < config.reference_horizon
            or source.input_features[OBS_STATE].shape != (config.proprio_dim,)
            or source.output_features["action"].shape != (config.action_dim,)
        ):
            raise ValueError("Frozen Stage 1 dimensions differ from Stage 2 actor/replay")
        self.policy = policy.eval().requires_grad_(False)
        self.preprocessor, self.postprocessor = preprocessor, postprocessor
        self.config = config
        self.relative = next(
            (
                step
                for step in preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        if self.relative is not None and source.action_feature_names:
            self.relative.action_names = list(source.action_feature_names)
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep):
                step.relative_step = self.relative
        normalizer = next(
            (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
        )
        self.normalizer = copy.deepcopy(normalizer)
        self._has_observation = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint,
        config: RLTActorConfig | None = None,
        *,
        config_factory: Callable[[PreTrainedConfig, str], RLTActorConfig] | None = None,
        device: str = "cuda",
        revision=None,
        compile_model: bool = False,
        compile_mode: str = "default",
    ):
        """Load and identify Stage 1 once, then validate the actor/replay contract.

        Deployment may build its actor config from the inspected source config
        and fingerprint through config_factory. Explicit configs retain their
        early identity check, before any model/config contents are loaded.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Config, PI05Policy

        if (config is None) == (config_factory is None):
            raise ValueError("Provide exactly one of config or config_factory for frozen RLT features")
        directory = Path(checkpoint).expanduser()
        if directory.is_dir():
            checkpoint = directory
        identity = checkpoint_fingerprint(checkpoint, revision=revision)
        if config is not None and identity != config.feature_model_id:
            raise ValueError("Stage 1 weights/processors do not match actor/replay feature_model_id")
        source = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
        if not isinstance(source, PI05Config):
            raise ValueError("RLT features require a PI05 checkpoint")
        if config_factory is not None:
            config = config_factory(source, identity)
            if not isinstance(config, RLTActorConfig):
                raise TypeError("RLT config_factory must return an RLTActorConfig")
            if config.feature_model_id != identity:
                raise ValueError("Stage 1 weights/processors do not match actor/replay feature_model_id")
        if directory.is_dir():
            processor_config = directory / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        else:
            from huggingface_hub import hf_hub_download

            processor_config = Path(
                hf_hub_download(
                    repo_id=str(checkpoint),
                    filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                    revision=revision,
                )
            )
        _bundled_tokenizer_dirs(processor_config)
        from lerobot.policies.rtc import RTCConfig

        rtc_config = (
            RTCConfig(enabled=True, execution_horizon=config.rtc_prefix_steps, mode="guided")
            if config.rtc_prefix_steps
            else None
        )
        source = replace(
            source,
            device=device,
            compile_model=compile_model,
            compile_mode=compile_mode,
            gradient_checkpointing=False,
            rtc_config=rtc_config,
            text_tokenizer_name=None,
        )
        pre, post = make_pre_post_processors(
            source,
            pretrained_path=checkpoint,
            pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        policy = PI05Policy.from_pretrained(checkpoint, config=source, revision=revision)
        if config.rtc_prefix_steps:
            from lerobot.async_inference.server.rtc_guidance import LightweightGuidedRTCProcessor

            policy.rtc_processor = LightweightGuidedRTCProcessor(rtc_config)
            policy.model.rtc_processor = policy.rtc_processor
        return cls(policy, pre, post, config)

    def reset(self):
        self._has_observation = False
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.no_grad()
    def encode(self, raw_batch: dict, *, committed_prefix=None) -> dict:
        self._has_observation = False
        batch = self.preprocessor(raw_batch)
        coordinates = self._capture_coordinates()
        prefix = torch.zeros(self.config.execution_horizon, self.config.action_dim)
        mask = torch.zeros(self.config.execution_horizon)
        kwargs = {}
        if committed_prefix is not None:
            if not self.config.rtc_prefix_steps:
                raise ValueError("RTC prefix requires an RTC actor/replay contract")
            committed_prefix = torch.as_tensor(committed_prefix)
            if committed_prefix.shape != (self.config.rtc_prefix_steps, self.config.action_dim):
                raise ValueError("Committed prefix length must match rtc_prefix_steps")
            for index, action in enumerate(committed_prefix):
                prefix[index] = coordinates.normalize(action)
                mask[index] = 1
            kwargs = {
                "prev_chunk_left_over": prefix[: self.config.rtc_prefix_steps].to(coordinates.device)[None],
                "inference_delay": self.config.rtc_prefix_steps,
                "execution_horizon": self.config.rtc_prefix_steps,
            }
        reference, z = self.policy.predict_action_chunk_with_rlt_features(batch, **kwargs)
        if reference.shape[0] != 1 or z.shape[0] != 1 or batch[OBS_STATE].shape[0] != 1:
            raise ValueError("A fixed-window feature adapter owns exactly one environment")
        state = {
            OBS_RLT: z[0],
            OBS_STATE: batch[OBS_STATE][0],
            OBS_REFERENCE: reference[0, : self.config.reference_horizon],
        }
        if self.config.rtc_prefix_steps:
            state[OBS_RTC_PREFIX], state[OBS_RTC_MASK] = prefix, mask
        validate_observation(state, self.config, batched=False)
        self._coordinates = coordinates
        self._has_observation = True
        return owned_state(state)

    def _capture_coordinates(self) -> ActionCoordinates:
        anchor, mask = None, None
        if self.relative is not None:
            anchor = self.relative.get_cached_state()
            if anchor is None:
                raise RuntimeError("Missing start-of-window relative-action anchor")
            anchor = anchor.detach().clone()
            mask = copy.deepcopy(self.relative._build_mask(self.config.action_dim))
        return ActionCoordinates(
            self.config.action_dim,
            anchor,
            mask,
            copy.deepcopy(self.normalizer),
            copy.deepcopy(self.postprocessor),
            next(self.policy.parameters()).device,
        )

    def snapshot_coordinates(self) -> ActionCoordinates:
        if not self._has_observation:
            raise RuntimeError("Encode an observation before snapshotting action coordinates")
        return self._coordinates

    @torch.no_grad()
    def executable_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        if not self._has_observation:
            raise RuntimeError("Encode an observation before decoding a window")
        expected = (self.config.execution_horizon, self.config.action_dim)
        if normalized_actions.shape != expected:
            raise ValueError("Expected a complete finite normalized execution window")
        return self._coordinates.decode(normalized_actions)

    @torch.no_grad()
    def normalize_executed(self, absolute_action: torch.Tensor) -> torch.Tensor:
        """Map the actual sent command back using the window's original raw-state anchor."""
        if not self._has_observation:
            raise RuntimeError("No start-of-window action coordinates are available")
        return self._coordinates.normalize(absolute_action)
