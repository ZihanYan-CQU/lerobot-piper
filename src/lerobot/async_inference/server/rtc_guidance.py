# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Deployment RTC guidance shared with the offline PI05 profiler."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from lerobot.policies.rtc import RTCConfig, RTCProcessor


def _guidance_weight(time, maximum):
    # Keep the native tensor operations/rounding, including the t=1 boundary.
    tau = 1 - time
    maximum = torch.as_tensor(maximum)
    tau = torch.as_tensor(tau)
    squared_one_minus_tau = (1 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = torch.nan_to_num((1 - tau) / tau, posinf=maximum)
    weight = torch.nan_to_num(c * inv_r2, posinf=maximum)
    return torch.minimum(weight, maximum)


@lru_cache(maxsize=128)
def _cpu_guidance_weight(time: float, maximum: float, dtype: torch.dtype) -> torch.Tensor:
    # Cache only scalar CPU constants, never request tensors or model outputs.
    # dtype is part of the key so changing the default dtype cannot reuse stale values.
    with torch.device("cpu"):
        return _guidance_weight(time, maximum)


def _correct_velocity(x_t, v_t, previous, weights, time, guidance_weight):
    x1_t = x_t - time * v_t
    # Native lightweight RTC treats v_t as constant w.r.t. x_t, so its
    # identity-VJP is the weighted error, rounded to the input gradient dtype.
    err = (previous - x1_t) * weights
    correction = err.to(dtype=x_t.dtype)
    return v_t - guidance_weight * correction, x1_t, err, correction


@dataclass(frozen=True)
class _PreparedGuidance:
    """Per-request tensor inputs for compiling the unchanged Euler sampling loop.

    Delay/length-dependent padding and masks have already been prepared outside
    Dynamo. Coefficients use the native CPU rounding before becoming constants
    in the compiled graph. No observation or model output is cached here.
    """

    weights: torch.Tensor
    guidance_weights: dict[float, float]
    coefficient_dtype: torch.dtype

    def is_debug_enabled(self):
        return False

    @torch.no_grad()
    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        x_t = x_t.clone().detach()
        v_t = original_denoise_step_partial(x_t)
        # Recreate the native zero-dimensional scalar, which has different
        # BF16 promotion/rounding from a bare Python float. Inductor can fold
        # this known constant; no Python cache or device context is traced.
        coefficient = torch.tensor(self.guidance_weights[time], dtype=self.coefficient_dtype, device="cpu")
        return _correct_velocity(x_t, v_t, prev_chunk_left_over, self.weights, time, coefficient)[0]


class LightweightGuidedRTCProcessor(RTCProcessor):
    """Evaluate the native identity-VJP correction without an autograd traversal."""

    def __init__(self, rtc_config: RTCConfig, *, trained_mode_supported: bool = False, fast: bool = True):
        super().__init__(rtc_config, trained_mode_supported=trained_mode_supported)
        self.fast = fast
        self._prefix_weight_key = None
        self._prefix_weights = None

    def _prepare_prefix(self, previous, inference_delay, execution_horizon, shape, device):
        if previous.ndim < 3:
            previous = previous.unsqueeze(0)
        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        execution_horizon = min(execution_horizon, previous.shape[1])
        batch_size, chunk_size, action_dim = shape
        if previous.shape[1] < chunk_size or previous.shape[2] < action_dim:
            padded = torch.zeros(batch_size, chunk_size, action_dim, device=device)
            padded[:, : previous.shape[1], : previous.shape[2]] = previous
            previous = padded
        assert previous.shape == shape, "The padded previous chunk must be the same size as the input tensor"
        key = (
            inference_delay,
            execution_horizon,
            chunk_size,
            self.rtc_config.prefix_attention_schedule,
            device,
            torch.get_default_dtype(),
        )
        if key != self._prefix_weight_key:
            self._prefix_weights = self.get_prefix_weights(inference_delay, execution_horizon, chunk_size)
            self._prefix_weights = self._prefix_weights.to(device)[None, :, None]
            self._prefix_weight_key = key
        return previous, self._prefix_weights, execution_horizon

    @torch.no_grad()
    def prepare_for_compilation(
        self, previous, inference_delay, execution_horizon, *, shape, device, num_steps
    ):
        if not self.fast or self.is_debug_enabled():
            return None
        previous, weights, _ = self._prepare_prefix(
            previous, inference_delay, execution_horizon, shape, device
        )
        dt = -1.0 / num_steps
        maximum, dtype = self.rtc_config.max_guidance_weight, torch.get_default_dtype()
        coefficients = {
            1.0 + step * dt: _cpu_guidance_weight(1.0 + step * dt, maximum, dtype).item()
            for step in range(num_steps)
        }
        return previous, _PreparedGuidance(weights, coefficients, dtype)

    @torch.no_grad()
    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> torch.Tensor:
        if not self.fast:
            # Reference path for A/B measurement and rollback. Native RTC
            # enables input gradients only after the velocity has been computed.
            def denoise_without_parameter_grad(input_x_t):
                with torch.no_grad():
                    return original_denoise_step_partial(input_x_t)

            return super().denoise_step(
                x_t,
                prev_chunk_left_over,
                inference_delay,
                time,
                denoise_without_parameter_grad,
                execution_horizon,
            )
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.clone().detach()
        squeezed = x_t.ndim < 3
        if squeezed:
            x_t = x_t.unsqueeze(0)
        prev_chunk_left_over, weights, execution_horizon = self._prepare_prefix(
            prev_chunk_left_over, inference_delay, execution_horizon, x_t.shape, x_t.device
        )
        v_t = original_denoise_step_partial(x_t)
        maximum = self.rtc_config.max_guidance_weight
        guidance_weight = (
            _cpu_guidance_weight(time, maximum, torch.get_default_dtype())
            if isinstance(time, (float, int))
            else _guidance_weight(time, maximum)
        )
        result, x1_t, err, correction = _correct_velocity(
            x_t, v_t, prev_chunk_left_over, weights, time, guidance_weight
        )
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)
        if self.is_debug_enabled():
            self.track(
                time=time,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )
        return result
