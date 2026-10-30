#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Offline, no-robot PI05 latency diagnostics with fixed synthetic observations.

Run in the same environment as the inference server::

    python -m lerobot.scripts.lerobot_profile_pi05 /path/to/pretrained_model --output baseline.json

Measures predict_action_chunk on device-resident images/tokens, including its
internal image preparation. Excludes cameras, transport, tokenizer, external
processors and action execution. Synthetic inputs measure cost, not task quality.
The uninstrumented benchmark and instrumented phase/trace runs are separate.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import fields
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import draccus
import torch

from lerobot.async_inference.server.rtc_guidance import LightweightGuidedRTCProcessor
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch


def configure_guided_rtc(policy, *, fast: bool = True) -> None:
    """Use the same guidance implementation as the deployment server."""
    config = policy.config.rtc_config
    if config is None or config.mode != "guided":
        raise ValueError("guided profiling requires a guided RTC config")
    processor = LightweightGuidedRTCProcessor(config, fast=fast)
    policy.rtc_processor = policy.model.rtc_processor = processor


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize milliseconds using linearly interpolated percentiles."""
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def read_config(path: Path) -> tuple[PI05Config, dict, dict]:
    """Decode model settings strictly so the benchmark uses the checkpoint's architecture."""
    raw = json.loads(path.read_text())
    if raw.get("type") != "pi05":
        raise ValueError("Expected a PI05 config.json")
    supported = {field.name for field in fields(PI05Config)}
    unused = {key: value for key, value in raw.items() if key not in supported and key != "type"}
    if unused:
        raise ValueError(f"Unsupported config fields; use the server's source checkout: {sorted(unused)}")
    config = draccus.decode(PI05Config, {key: value for key, value in raw.items() if key in supported})
    if config.use_visual_memory or config.use_proprioceptive_memory or config.use_peft:
        raise ValueError("This diagnostic supports ordinary PI05 without memory or PEFT")
    # Explicitly benchmark ordinary inference; do not edit the checkpoint on disk.
    config.rtc_config = None
    config.compile_model = False
    config.device = "cpu"
    config.validate_features()
    return config, raw, unused


def inventory(model: "PI05Pytorch") -> dict:
    """Count actual parameter dtypes instead of trusting config.dtype."""
    backbone = model.paligemma_with_expert
    modules = {
        "vision": backbone.paligemma.model.vision_tower,
        "vision_projector": backbone.paligemma.model.multi_modal_projector,
        "language": backbone.paligemma.model.language_model,
        "action_expert": backbone.gemma_expert.model,
        "time_mlp_in": model.time_mlp_in,
        "time_mlp_out": model.time_mlp_out,
    }
    result = {}
    for name, module in modules.items():
        counts = Counter()
        for parameter in module.parameters():
            counts[str(parameter.dtype)] += parameter.numel()
        result[name] = dict(counts)
    result["vision_attention"] = modules["vision"].config._attn_implementation
    result["language_attention"] = modules["language"].config._attn_implementation
    result["expert_attention"] = modules["action_expert"].config._attn_implementation
    return result


def nvidia_snapshot() -> dict:
    """Read clocks, power, utilization and sharing information without changing GPU state."""
    commands = {
        "gpus": [
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,"
            "power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        "processes": ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"],
    }
    result = {}
    for name, args in commands.items():
        try:
            completed = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=5)
            result[name] = {"returncode": completed.returncode, "text": completed.stdout + completed.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"error": str(exc)}
    return result


def synchronize(device: torch.device) -> None:
    """Wait for work on the selected device at benchmark boundaries only."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, device: torch.device) -> tuple[float, torch.Tensor]:
    """Measure complete prediction, including GPU completion but not CPU output copy."""
    if device.type == "cuda":
        torch.compiler.cudagraph_mark_step_begin()
    synchronize(device)
    started = time.perf_counter()
    output = call()
    synchronize(device)
    return (time.perf_counter() - started) * 1000, output


class PhaseRecorder:
    """Nested host/event intervals; no synchronization between model stages.

    CUDA event intervals include stream idle time from host dispatch. They are
    not sums of kernel durations. Parent intervals include their child intervals.
    """

    def __init__(self, device: torch.device, *, trace: bool = False):
        self.device = device
        self.trace = trace
        self.pending = []
        self.samples = defaultdict(list)

    def wrap(self, name, function):
        """Wrap a bound method or module function without changing its arguments."""

        @wraps(function)
        def wrapped(*args, **kwargs):
            start_event = end_event = None
            if self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
            region = torch.profiler.record_function(f"pi05/{name}") if self.trace else nullcontext()
            with region:
                if start_event is not None:
                    start_event.record()
                started = time.perf_counter()
                result = function(*args, **kwargs)
                host_ms = (time.perf_counter() - started) * 1000
                if end_event is not None:
                    end_event.record()
            self.pending.append((name, host_ms, start_event, end_event))
            return result

        return wrapped

    @contextmanager
    def instrument(self, policy):
        """Temporarily instrument eager PI05 only; restore all methods on exit."""
        model = policy.model
        backbone = model.paligemma_with_expert
        targets = [
            (policy, "_preprocess_images", "image_prepare"),
            (model, "embed_prefix", "embed_prefix_inclusive"),
            (backbone, "embed_image", "vision_per_camera"),
            (backbone.paligemma.model.language_model, "forward", "language_prefill"),
            (model, "denoise_step", "denoise_inclusive"),
            (model, "embed_suffix", "action_time_embedding"),
            (backbone.gemma_expert.model, "forward", "action_expert"),
            (modeling_pi05, "clone_past_key_values", "kv_clone"),
        ]
        with ExitStack() as stack:
            for owner, attribute, name in targets:
                stack.enter_context(
                    patch.object(owner, attribute, self.wrap(name, getattr(owner, attribute)))
                )
            yield

    def finish_sample(self):
        """Collect events after the caller has synchronized the completed prediction."""
        grouped = defaultdict(list)
        for name, host_ms, start, end in self.pending:
            grouped[name].append((host_ms, None if start is None else start.elapsed_time(end)))
        for name, intervals in grouped.items():
            self.samples[name].append(
                {
                    "calls": len(intervals),
                    "host_dispatch_ms": sum(value[0] for value in intervals),
                    "cuda_stream_ms": (
                        None if intervals[0][1] is None else sum(value[1] for value in intervals)
                    ),
                }
            )
        self.pending.clear()

    def report(self):
        """Report each stage's total per prediction, not just its per-call cost."""
        result = {}
        for name, samples in self.samples.items():
            result[name] = {
                "calls_per_prediction": [sample["calls"] for sample in samples],
                "host_dispatch": summarize([sample["host_dispatch_ms"] for sample in samples]),
            }
            if samples[0]["cuda_stream_ms"] is not None:
                result[name]["cuda_stream"] = summarize([sample["cuda_stream_ms"] for sample in samples])
        return result


def synthetic_batch(config: PI05Config, device: torch.device, image_size: str, active_tokens: int):
    """Keep batch=1, camera count, padding and action horizon fixed across runs."""
    batch = {}
    for key, feature in config.image_features.items():
        shape = (3, *config.image_resolution) if image_size == "native" else tuple(feature.shape)
        batch[key] = torch.rand((1, *shape), device=device)
    batch[OBS_STATE] = torch.zeros((1, *config.input_features[OBS_STATE].shape), device=device)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
        (1, config.tokenizer_max_length), dtype=torch.long, device=device
    )
    batch[OBS_LANGUAGE_TOKENS][:, :active_tokens] = 2
    batch[OBS_LANGUAGE_ATTENTION_MASK] = (
        torch.arange(config.tokenizer_max_length, device=device)[None, :] < active_tokens
    )
    noise = torch.randn((1, config.chunk_size, config.max_action_dim), device=device)
    return batch, noise


def load_policy(checkpoint: Path, config: PI05Config, device: torch.device):
    """Load strictly: a missing/bad weight file must never benchmark random weights."""
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Expected local, unsharded weights: {weights}")
    policy = PI05Policy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    policy.config.device = str(device)
    return policy


def main():
    """Write one reproducible runtime/latency report without starting a robot or server."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Local pretrained_model directory (or JSON for inspect-only)"
    )
    parser.add_argument(
        "--inspect-only", action="store_true", help="Inspect parameter dtypes on meta device, no weights/GPU"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--phase-runs", type=int, default=5, help="Separate eager instrumentation runs; 0 disables"
    )
    parser.add_argument(
        "--num-steps", type=int, help="Override denoising steps for an offline cost experiment"
    )
    parser.add_argument(
        "--active-tokens",
        type=int,
        default=100,
        help="Nonpadding synthetic tokens; padded length stays fixed",
    )
    parser.add_argument("--image-size", choices=["native", "checkpoint"], default="native")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], help="Override checkpoint parameter precision"
    )
    parser.add_argument(
        "--precision-profile",
        choices=["checkpoint", "lerobot", "openpi"],
        help="Select the production precision path; checkpoint applies dtype to every floating parameter",
    )
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high"], help="Optional FP32 GEMM precision experiment"
    )
    parser.add_argument(
        "--compile-mode",
        choices=["eager", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="eager",
    )
    parser.add_argument("--rtc-mode", choices=["ordinary", "guided"], default="ordinary")
    parser.add_argument("--rtc-prefix-length", type=int, default=20)
    parser.add_argument("--rtc-delay", type=int, default=10)
    parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-prefix-schedule",
        choices=[item.value for item in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument("--rtc-fast-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vision-autocast", action="store_true", help="Experimental BF16 autocast scoped to vision only"
    )
    parser.add_argument(
        "--trace", type=Path, help="Export a separate one-prediction torch.profiler Chrome trace"
    )
    parser.add_argument("--output", type=Path, default=Path("pi05_profile.json"))
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.runs < 1
        or args.phase_runs < 0
        or (args.num_steps is not None and args.num_steps < 1)
    ):
        parser.error("warmup/runs/num-steps must be positive; phase-runs must be nonnegative")
    require_package("transformers", extra="pi")
    config_path = args.checkpoint if args.checkpoint.is_file() else args.checkpoint / "config.json"
    config, original_config, unused = read_config(config_path)
    if not 1 <= args.active_tokens <= config.tokenizer_max_length:
        parser.error("active-tokens must be between 1 and tokenizer_max_length")
    if not config.image_features:
        parser.error("At least one camera feature is required")
    if args.rtc_mode == "guided":
        if (
            not 0 <= args.rtc_delay <= args.rtc_prefix_length < config.chunk_size
            or args.rtc_prefix_length == 0
        ):
            parser.error(
                "guided profiling requires 0 <= rtc-delay <= rtc-prefix-length < chunk_size, with a nonempty prefix"
            )
        config.rtc_config = RTCConfig(
            mode="guided",
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_schedule),
        )
    if args.num_steps is not None:
        config.num_inference_steps = args.num_steps
    if args.dtype is not None:
        config.dtype = args.dtype
    if args.precision_profile is not None:
        config.precision_profile = args.precision_profile
    # Use the production constructor's compile path, including its high FP32
    # matmul setting. Explicit --matmul-precision still takes precedence below.
    config.compile_model = args.compile_mode != "eager"
    if config.compile_model:
        config.compile_mode = args.compile_mode
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    report = {
        "scope": f"{args.rtc_mode} PI05, synthetic device-resident inputs, no tokenizer/network/robot/postprocessor",
        "checkpoint_config": original_config,
        "unused_tokenizer_metadata": unused,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "draccus")},
        "cuda_runtime": torch.version.cuda,
        "python": {"executable": sys.executable, "version": sys.version},
        "cpu_threads": torch.get_num_threads(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "TORCH_LOGS")
        },
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "source": {},
    }
    for obj in (
        PI05Policy,
        modeling_pi05.clone_past_key_values,
        modeling_pi05.euler_integrate,
        modeling_pi05.PiGemmaForCausalLM,
        LightweightGuidedRTCProcessor,
    ):
        path = Path(inspect.getfile(obj)).resolve()
        report["source"][path.name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    if args.inspect_only:
        with torch.device("meta"):
            model = PI05Pytorch(config)
        if args.matmul_precision:
            torch.set_float32_matmul_precision(args.matmul_precision)
        report["parameters"] = inventory(model)
        report["note"] = "Metadata inspection only; attention is overwritten to eager when sampling runs."
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            parser.error("Timing requires an available CUDA GPU; use --inspect-only for metadata")
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        report["gpu_before"] = nvidia_snapshot()
        props = torch.cuda.get_device_properties(device)
        report["device"] = {
            "name": props.name,
            "total_memory": props.total_memory,
            "multiprocessors": props.multi_processor_count,
        }
        print("Loading checkpoint strictly; all actions stay in this diagnostic process.", flush=True)
        policy = load_policy(args.checkpoint, config, device)
        if args.rtc_mode == "guided":
            configure_guided_rtc(policy, fast=args.rtc_fast_guidance)
        # Profile initialization may set OpenPI's default to high. An explicit
        # benchmark override takes precedence so precision A/Bs stay isolated.
        if args.matmul_precision:
            torch.set_float32_matmul_precision(args.matmul_precision)
        batch, noise = synthetic_batch(config, device, args.image_size, args.active_tokens)
        report["input_shapes"] = {key: list(value.shape) for key, value in batch.items()}
        previous = None
        delay = 0
        if args.rtc_mode == "guided":
            previous = torch.zeros(
                args.rtc_prefix_length, config.output_features[ACTION].shape[0], device=device
            )
            delay = args.rtc_delay
            report["input_shapes"]["rtc_prefix"] = list(previous.shape)
        if args.vision_autocast:
            backbone = policy.model.paligemma_with_expert
            original_embed_image = backbone.embed_image

            def embed_with_autocast(*positional, **keywords):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return original_embed_image(*positional, **keywords)

            backbone.embed_image = embed_with_autocast

        def predict():
            return policy.predict_action_chunk(
                batch, noise=noise, inference_delay=delay, prev_chunk_left_over=previous
            )

        report["warmup_ms"] = []
        for i in range(args.warmup):
            elapsed, _ = measure(predict, device)
            report["warmup_ms"].append(elapsed)
            print(f"warmup {i + 1}/{args.warmup}: {elapsed:.2f} ms", flush=True)
        samples = []
        for i in range(args.runs):
            elapsed, output = measure(predict, device)
            samples.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"measurement {i + 1}/{args.runs}: {elapsed:.2f} ms", flush=True)
        report["predict_sync"] = summarize(samples)
        report["predict_samples_ms"] = samples
        report["fixed_input_actions"] = output.detach().cpu().tolist()
        report["actions_finite"] = bool(torch.isfinite(output).all().item())
        report["parameters"] = inventory(policy.model)
        report["gpu_after"] = nvidia_snapshot()
        if args.phase_runs and args.compile_mode == "eager":
            recorder = PhaseRecorder(device)
            with recorder.instrument(policy):
                for _ in range(args.phase_runs):
                    measure(predict, device)
                    recorder.finish_sample()
            report["phases"] = recorder.report()
            report["phase_note"] = (
                "Separate instrumented run. vision_per_camera sums all cameras per prediction; "
                "denoise/action_expert/kv_clone sum all denoising steps. Nested intervals must not be added. "
                "CUDA events measure stream intervals including host dispatch gaps, not pure kernel time."
            )
        elif args.compile_mode != "eager":
            report["phase_note"] = "Python instrumentation skipped to avoid changing compiled graphs."
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            recorder = PhaseRecorder(device, trace=True)
            regions = recorder.instrument(policy) if args.compile_mode == "eager" else nullcontext()
            with (
                regions,
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profiler,
            ):
                measure(predict, device)
            profiler.export_chrome_trace(str(args.trace))
        print(json.dumps(report["predict_sync"], indent=2), flush=True)

    report["effective_policy"] = {
        "dtype": config.dtype,
        "precision_profile": config.precision_profile,
        "num_inference_steps": config.num_inference_steps,
        "compile_model": config.compile_model,
        "compile_mode": args.compile_mode,
    }
    report["matmul_precision"] = torch.get_float32_matmul_precision()
    report["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
