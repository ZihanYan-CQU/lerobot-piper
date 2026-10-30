# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare PI05 network backends on one shared, fixed observation and noise.

Loads models sequentially so both full-size networks need not fit on the GPU
at once. This tool never connects to a robot. See docs/source/pi05.mdx.
"""

import argparse
import gc
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

RLINF_REVISION = "061c4cce80bbab998ef90585475d3f51290a2dfa"


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shapes differ: {reference.shape} versus {candidate.shape}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Backend produced non-finite actions.")
    difference = candidate.double() - reference.double()
    return {
        "mean_absolute_error": difference.abs().mean().item(),
        "max_absolute_error": difference.abs().max().item(),
        "root_mean_square_error": difference.square().mean().sqrt().item(),
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(policy, batch, noise, *, warmup: int, iterations: int) -> tuple[torch.Tensor, dict]:
    device = noise.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    output = None
    with torch.no_grad():
        for index in range(warmup + iterations):
            _synchronize(device)
            start = time.perf_counter()
            output = policy.predict_action_chunk(batch, noise=noise.clone())
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if index >= warmup:
                durations.append(elapsed * 1000)
    metrics = {
        "mean_latency_ms": statistics.mean(durations),
        "median_latency_ms": statistics.median(durations),
        "min_latency_ms": min(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }
    return output.detach().float().cpu(), metrics


def compare(
    checkpoint: str,
    batch: dict,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    warmup: int = 2,
    iterations: int = 5,
    seed: int = 0,
    num_steps: int | None = None,
    batch_is_preprocessed: bool = False,
    revision: str | None = None,
    export_dir: Path | None = None,
) -> dict:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if num_steps is not None and num_steps < 1:
        raise ValueError("num_steps must be positive")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu to run a CPU comparison.")
    cfg = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
    if not isinstance(cfg, PI05Config):
        raise ValueError("The comparison requires a LeRobot PI05 checkpoint and its policy config.")
    if cfg.use_peft:
        raise ValueError(
            "Merge the PI05 adapter with examples/tutorial/pi05/merge_lora.py before comparing backends. "
            "Comparing an adapter directory could load stale base weights instead of the trained adapter."
        )
    if cfg.backend != "lerobot":
        raise ValueError("Use the original LeRobot/OpenPI checkpoint so both backends load the same weights.")
    if export_dir is not None:
        export_dir = Path(export_dir)
        if export_dir.exists() and any(export_dir.iterdir()):
            raise FileExistsError(f"Export directory must be new or empty: {export_dir}")
        if Path(checkpoint).is_dir() and export_dir.resolve() == Path(checkpoint).resolve():
            raise ValueError("Export must not overwrite the source checkpoint.")
    cfg = replace(cfg, device=str(target_device), dtype=dtype, compile_model=False, rtc_config=None)
    # Validate the second backend before loading a potentially multi-GB baseline.
    replace(cfg, backend="rlinf")
    if num_steps is not None:
        cfg.num_inference_steps = num_steps

    preprocessor = postprocessor = None
    if not batch_is_preprocessed or export_dir is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint,
            pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": str(target_device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
    prepared = batch if batch_is_preprocessed else preprocessor(batch)
    prepared = _to_device(prepared, target_device)
    tokens = prepared[OBS_LANGUAGE_TOKENS]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tokens.shape[0], cfg.chunk_size, cfg.max_action_dim, generator=generator).to(
        target_device
    )
    outputs, physical_outputs, measurements = {}, {}, {}

    for backend in ("lerobot", "rlinf"):
        policy = PI05Policy.from_pretrained(
            checkpoint, config=replace(cfg, backend=backend), revision=revision, strict=True
        ).eval()
        output, metrics = benchmark(policy, prepared, noise, warmup=warmup, iterations=iterations)
        outputs[backend], measurements[backend] = output, metrics
        if postprocessor is not None:
            physical_outputs[backend] = postprocessor(output.clone()).detach().float().cpu()
        if backend == "rlinf" and export_dir is not None:
            error_metrics(outputs["lerobot"], output)
            # The processor export bundles its tokenizer. Avoid retaining a local
            # override from the source machine in the standalone policy config.
            policy.config = replace(policy.config, text_tokenizer_name=None)
            policy.save_pretrained(export_dir)
            preprocessor.save_pretrained(
                export_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            )
            postprocessor.save_pretrained(
                export_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            )
        del policy
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "checkpoint": checkpoint,
        "revision": revision,
        "rlinf_source_revision": RLINF_REVISION,
        "torch_version": torch.__version__,
        "device": str(target_device),
        "dtype": dtype,
        "seed": seed,
        "num_inference_steps": cfg.num_inference_steps,
        "warmup": warmup,
        "iterations": iterations,
        "action_shape": list(outputs["lerobot"].shape),
        "normalized_action_error": error_metrics(outputs["lerobot"], outputs["rlinf"]),
        "measurements": measurements,
        "compiled": False,
        "rtc_enabled": False,
        "known_operator_difference": "RLinf exact GELU versus LeRobot tanh-approximate GELU",
    }
    if physical_outputs:
        report["executable_action_error"] = error_metrics(
            physical_outputs["lerobot"], physical_outputs["rlinf"]
        )
    if export_dir is not None:
        report["export_dir"] = str(export_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Original LeRobot PI05 checkpoint directory or Hub ID"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", type=Path, help="A batched observation dict saved with torch.save")
    source.add_argument("--dataset-repo-id", help="LeRobot dataset ID; uses one frame as input")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--batch-is-preprocessed",
        action="store_true",
        help="Batch already contains normalized state and language tokens",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--revision")
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Save a standalone RLinf-backend LeRobot checkpoint and both processors",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_backend_comparison.json"))
    args = parser.parse_args()
    if args.batch is not None:
        batch = torch.load(args.batch, map_location="cpu", weights_only=True)
        if not isinstance(batch, dict):
            parser.error("--batch must contain an observation dictionary")
    else:
        if args.batch_is_preprocessed:
            parser.error("--batch-is-preprocessed requires --batch")
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
        frame = dataset[args.frame_index]
        batch = {
            key: value for key, value in frame.items() if key.startswith("observation.") or key == "task"
        }
    report = compare(
        args.checkpoint,
        batch,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        num_steps=args.num_steps,
        batch_is_preprocessed=args.batch_is_preprocessed,
        revision=args.revision,
        export_dir=args.export_dir,
    )
    report["input"] = {
        "batch": str(args.batch) if args.batch is not None else None,
        "dataset_repo_id": args.dataset_repo_id,
        "frame_index": args.frame_index if args.dataset_repo_id is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
