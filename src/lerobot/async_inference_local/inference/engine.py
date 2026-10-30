"""PI05 runtime owned exclusively by a single local inference thread."""

from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)

from ..shared.configs import LocalDeploymentConfig
from ..shared.types import ActionChunk, Observation, PlanSlice, Prediction, Prefix, Request
from .observation import ObservationPreparer
from .rtc import LocalGuidedRTCProcessor

logger = logging.getLogger(__name__)


class LocalPolicyEngine:
    """Load, warm up and invoke PI05 exclusively on its owning worker."""

    def __init__(self, config: LocalDeploymentConfig, observation_features: dict, action_keys: list[str]):
        """Validate the checkpoint contract before constructing its model and processors."""
        self.config = config
        self.device = torch.device(config.inference.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable; refusing an implicit CPU fallback")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        elif config.inference.compile_model:
            raise ValueError("Disable compilation for CPU functional testing")

        policy_class = get_policy_class("pi05")
        policy_config = policy_class.config_class.from_pretrained(
            config.pretrained_name_or_path, cli_overrides=config.inference.policy_overrides()
        )
        if policy_config.type != "pi05":
            raise ValueError("This local deployment entry point requires a PI05 checkpoint")
        if policy_config.use_peft:
            raise ValueError("Local deployment currently requires merged PI05 weights, not a PEFT adapter")
        if config.actions_per_chunk > policy_config.chunk_size:
            raise ValueError("actions_per_chunk exceeds the checkpoint horizon")
        action_dim = policy_config.output_features["action"].shape[0]
        if action_dim != len(action_keys):
            raise ValueError("Checkpoint action dimension differs from the robot")
        checkpoint_keys = getattr(policy_config, "action_feature_names", None)
        if checkpoint_keys and [k.removesuffix(".pos") for k in checkpoint_keys] != [
            k.removesuffix(".pos") for k in action_keys
        ]:
            raise ValueError("action_keys must follow the checkpoint's action_feature_names order")
        self.trained_delay = 0
        if config.rtc.mode == "trained":
            maximum = int(getattr(policy_config, "rtc_training_max_delay", 0))
            self.trained_delay = config.trained_delay_steps or maximum
            if not 0 < self.trained_delay <= maximum or 2 * self.trained_delay > config.actions_per_chunk:
                raise ValueError("Trained RTC requires 0 < delay <= checkpoint maximum and 2*delay <= chunk")
        policy_config.rtc_config = copy.deepcopy(config.rtc)
        self.policy = policy_class.from_pretrained(config.pretrained_name_or_path, config=policy_config)
        self.policy.eval()
        if config.rtc.mode == "guided":
            guidance = LocalGuidedRTCProcessor(policy_config.rtc_config)
            self.policy.rtc_processor = guidance
            self.policy.model.rtc_processor = guidance

        pre_overrides = {
            "device_processor": {"device": str(self.device)},
            "rename_observations_processor": {"rename_map": config.rename_map},
        }
        if config.inference.tokenizer_path:
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": config.inference.tokenizer_path}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides=pre_overrides,
            # Keep the executable-space prefix on GPU; transfer once after postprocessing.
            postprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.prepare = ObservationPreparer(config, policy_config, observation_features)
        self.relative = next(
            (
                step
                for step in self.preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        if self.relative is not None:
            # Older processor files may omit names or retain stale metadata.
            # The checkpoint's declared order (validated above), or the explicit
            # robot action order, must also govern gripper exclusion in both
            # absolute postprocessing and RTC prefix reanchoring.
            self.relative.action_names = list(checkpoint_keys or action_keys)
        normalizer = next(
            (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
        )
        if (
            config.inference.image_mode == "openpi_uint8"
            and normalizer is not None
            and normalizer.norm_map.get(FeatureType.VISUAL, NormalizationMode.IDENTITY)
            != NormalizationMode.IDENTITY
        ):
            raise ValueError("openpi_uint8 requires identity visual normalization; use lerobot_float32")
        # Independent statistics prevent CPU state tokenization and GPU prefix
        # normalization from moving the same processor's statistics back and forth.
        self.prefix_normalizer = copy.deepcopy(normalizer)
        if self.prefix_normalizer is not None:
            self.prefix_normalizer.to(device=self.device)
        self.action_dim = action_dim

    def reset(self) -> None:
        """Clear policy history and processor state on the inference worker."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        if self.policy.rtc_processor is not None:
            self.policy.rtc_processor.reset_tracker()

    def _model_prefix(self, prefix: Prefix) -> torch.Tensor | None:
        if not len(prefix):
            return None
        if self.relative is None:
            return prefix.tensor("model")
        state = self.relative.get_cached_state()
        if state is None:
            raise RuntimeError("Relative actions require the current observation's raw state")
        absolute = prefix.tensor("absolute")
        relative = to_relative_actions(
            absolute, state.to(self.device), self.relative._build_mask(absolute.shape[-1])
        )
        transition = create_transition(action=relative)
        if self.prefix_normalizer is not None:
            transition = self.prefix_normalizer(transition)
        return transition[TransitionKey.ACTION]

    @torch.no_grad()
    def predict(self, request: Request, latency_s: float) -> Prediction:
        """Prepare one snapshot and return owned GPU/CPU action buffers."""
        started = time.monotonic()
        if self.device.type == "cuda" and self.config.inference.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        batch = self.preprocessor(self.prepare(request.observation.values, request.task))
        previous = self._model_prefix(request.prefix)
        if self.config.rtc.mode == "trained":
            delay = min(self.trained_delay, len(request.prefix))
        else:
            delay = min(
                math.ceil(latency_s * self.config.fps) + self.config.latency_margin_steps,
                len(request.prefix),
                self.config.actions_per_chunk - 1,
            )
        result = self.policy.predict_action_chunk(batch, inference_delay=delay, prev_chunk_left_over=previous)
        if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < self.config.actions_per_chunk:
            raise ValueError("PI05 must return a complete [1, horizon, action_dim] action chunk")
        result = result[:, : self.config.actions_per_chunk]
        # Compiled graph outputs may be reused on the next invocation. These small
        # action buffers must outlive that invocation; image/KV buffers do not escape.
        model = result[0].detach().clone()
        absolute = self.postprocessor(result).squeeze(0).detach().clone()
        cpu = absolute.to("cpu", dtype=torch.float32).numpy()
        if cpu.shape != (self.config.actions_per_chunk, self.action_dim) or not np.isfinite(cpu).all():
            raise ValueError("Postprocessed actions have an invalid shape or non-finite values")
        if (
            self.config.rtc.mode == "trained"
            and delay
            and not np.allclose(cpu[:delay], request.prefix.cpu()[:delay], rtol=1e-4, atol=1e-4)
        ):
            raise ValueError("Trained RTC changed its committed hard prefix")
        cpu.setflags(write=False)
        chunk = ActionChunk(model, absolute, cpu, request.observation.source_at)
        return Prediction(chunk, delay, time.monotonic() - started)

    def warmup(self, observation: Observation, cancelled) -> list[float]:
        """Exercise ordinary and continuation graphs without publishing any action."""
        samples = []
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            try:
                previous = None
                for branch in ("ordinary", "continuation"):
                    for index in range(self.config.inference.warmup_runs):
                        if cancelled():
                            raise RuntimeError("Local warmup cancelled")
                        if branch == "continuation" and previous is not None:
                            prefix = Prefix(0, (PlanSlice(previous, 0, len(previous)),))
                        else:
                            prefix = Prefix(0, ())
                        request = Request(0, observation, prefix, self.config.task, time.monotonic())
                        prediction = self.predict(request, max(samples, default=1 / self.config.fps))
                        previous = prediction.chunk
                        if index > 0:
                            samples.append(prediction.elapsed_s)
                        logger.info(
                            "Local warmup %s %d: %.1f ms", branch, index + 1, prediction.elapsed_s * 1000
                        )
            finally:
                self.reset()
        return samples
