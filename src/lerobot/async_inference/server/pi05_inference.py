# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""PI05 deployment overrides and warmup, independent of robot/transport code."""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


@dataclass
class PI05InferenceConfig:
    """Use the checkpoint dtype uniformly, or select LeRobot/OpenPI BF16 settings.

    ``checkpoint`` applies the saved dtype to all floating model parameters and
    retains the saved compile settings. Numerical reductions and the action
    integration/output may still use FP32; this does not force every op to BF16.
    ``lerobot`` selects BF16 with FP32 visual parameters, BF16 CUDA vision
    autocast during single-frame inference, and batched equal-shaped cameras
    on the LeRobot backend. Compilation defaults to disabled, as in LeRobot.
    ``openpi`` selects BF16 with FP32 visual embeddings/Gemma norms, high FP32
    matmul precision and max-autotune sampling. Attention stays on the official
    eager Gemma path. Explicit compile options take precedence over the profile.
    """

    profile: str = "checkpoint"
    compile_model: bool | None = None
    compile_mode: str | None = None
    warmup_runs: int = 3
    warmup_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if self.profile not in {"checkpoint", "lerobot", "openpi"}:
            raise ValueError("pi05_inference.profile must be checkpoint, lerobot or openpi")
        if self.compile_mode not in {
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"Invalid pi05_inference.compile_mode: {self.compile_mode}")
        if self.warmup_runs < 0:
            raise ValueError("pi05_inference.warmup_runs must be non-negative")
        if not math.isfinite(self.warmup_timeout_s) or self.warmup_timeout_s <= 0:
            raise ValueError("pi05_inference.warmup_timeout_s must be finite and positive")

    def overrides(self) -> tuple[str, ...]:
        """Return deterministic config overrides, also used in the warm-model cache key."""
        values: dict[str, str | bool] = {"precision_profile": self.profile}
        if self.profile in {"lerobot", "openpi"}:
            values.update(
                dtype="bfloat16",
                compile_model=self.profile == "openpi",
                compile_mode="max-autotune",
            )
        if self.compile_model is not None:
            values["compile_model"] = self.compile_model
        if self.compile_mode is not None:
            values["compile_mode"] = self.compile_mode
        return tuple(
            f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in sorted(values.items())
        )


def warmup_pi05(
    policy,
    *,
    runs: int,
    prefix_length: int,
    inference_delay: int,
    logger: logging.Logger,
    prepare_batch: Callable[[], dict[str, Any]] | None = None,
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reset_processors: Callable[[], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Warm ordinary and representative RTC shapes before accepting observations.

    Deployment supplies the full preparation/postprocessing path. No actions are
    emitted. RNG is restored and temporary episode state is cleared. Run on the
    actual inference thread. PI05 keeps guided RTC controls outside compilation;
    other input-shape changes can still require a new compiled graph.
    Keep no_grad rather than inference_mode: native/reference lightweight RTC
    still uses autograd to evaluate its identity correction.
    """
    if runs == 0:
        return
    config = policy.config
    if not 0 <= inference_delay <= prefix_length < config.chunk_size:
        raise ValueError("Warmup requires 0 <= delay <= prefix_length < chunk_size")
    device = next(policy.parameters()).device
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = "unavailable"
        logger.info(
            "PI05 warmup runtime torch=%s cuda=%s triton=%s GPU=%s capability=%d.%d SMs=%d",
            torch.__version__,
            torch.version.cuda,
            triton_version,
            properties.name,
            properties.major,
            properties.minor,
            properties.multi_processor_count,
        )
    policy.eval()
    policy.reset()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            if prepare_batch is None:
                batch = {
                    key: torch.zeros((1, 3, *config.image_resolution), device=device)
                    for key in config.image_features
                }
                batch[OBS_STATE] = torch.zeros((1, *config.input_features[OBS_STATE].shape), device=device)
                batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
                    (1, config.tokenizer_max_length), dtype=torch.long, device=device
                )
                batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
                    (1, config.tokenizer_max_length), dtype=torch.bool, device=device
                )
            cases = [("ordinary", 0, None)]
            if prefix_length:
                action_dim = config.output_features["action"].shape[0]
                prefix = torch.zeros((prefix_length, action_dim), device=device)
                cases.append(("rtc", inference_delay, prefix))
            for name, delay, prefix in cases:
                for index in range(runs):
                    if cancelled is not None and cancelled():
                        raise RuntimeError("PI05 warmup cancelled")
                    logger.info(
                        "PI05 warmup %s %d/%d starting (prefix=%d, delay=%d)",
                        name,
                        index + 1,
                        runs,
                        0 if prefix is None else prefix.shape[0],
                        delay,
                    )
                    if device.type == "cuda":
                        torch.compiler.cudagraph_mark_step_begin()
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    stage = "prepare_batch"
                    try:
                        if prepare_batch is not None:
                            batch = prepare_batch()
                        stage = "predict_action_chunk"
                        actions = policy.predict_action_chunk(
                            batch, inference_delay=delay, prev_chunk_left_over=prefix
                        )
                        stage = "postprocess"
                        if postprocess is not None:
                            actions = postprocess(actions)
                        stage = "cuda_synchronize"
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    except Exception:
                        # Log before fork_rng restores CUDA state: a poisoned
                        # context may also fail during restoration and obscure
                        # the original exception at the end of the traceback.
                        logger.exception(
                            "PI05 warmup %s %d/%d failed at %s (prefix=%d, delay=%d)",
                            name,
                            index + 1,
                            runs,
                            stage,
                            0 if prefix is None else prefix.shape[0],
                            delay,
                        )
                        raise
                    elapsed = time.perf_counter() - started
                    if not torch.isfinite(actions).all().item():
                        raise RuntimeError("PI05 warmup produced non-finite actions")
                    del actions
                    logger.info("PI05 warmup %s %d/%d: %.1f ms", name, index + 1, runs, elapsed * 1000)
    finally:
        policy.reset()
        processor = getattr(policy, "rtc_processor", None)
        if processor is not None:
            processor.reset_tracker()
        if reset_processors is not None:
            reset_processors()
