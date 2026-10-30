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

"""Merge a PI05 PEFT checkpoint, including its full expert, for ordinary RTC inference."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.peft import load_pi05_adapter
from lerobot.utils.import_utils import _peft_available, require_package

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig


def merge_adapter(
    adapter: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    revision: str | None = None,
    rlt_encoder_only: bool = False,
) -> Path:
    """Export the adapter and trained expert with the checkpoint's saved processors.

    The base checkpoint referenced by adapter_config.json must remain available.
    Both network backends are supported; this operation does not switch backends.
    """
    require_package("peft", extra="peft")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    config = PreTrainedConfig.from_pretrained(adapter, revision=revision)
    if not isinstance(config, PI05Config) or not config.use_peft:
        raise ValueError("Expected a PI05 PEFT checkpoint with its saved LeRobot config and processors.")
    if rlt_encoder_only and not config.use_rlt:
        raise ValueError("rlt_encoder_only requires a checkpoint trained with use_rlt=True.")
    config = replace(config, device=device, use_peft=False, compile_model=False, gradient_checkpointing=False)
    peft_config = PeftConfig.from_pretrained(adapter, revision=revision)
    if not peft_config.base_model_name_or_path:
        raise ValueError("The adapter config must reference its pretrained base checkpoint.")
    # Load processors before merging/writing so missing stats cannot produce a
    # seemingly deployable model without its original action transforms.
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=adapter,
        pretrained_revision=revision,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    base = PI05Policy.from_pretrained(
        peft_config.base_model_name_or_path, config=config, revision=peft_config.revision
    )
    peft_config = base._prepare_peft_config(peft_config)
    policy = load_pi05_adapter(base, adapter, config=peft_config, revision=revision)
    merged = policy.merge_and_unload(safe_merge=True)
    merged.config = replace(
        config,
        use_peft=False,
        peft_mode="default",
        pretrained_path=None,
        pretrained_revision=None,
        text_tokenizer_name=None,
        rlt_inference_only=config.rlt_inference_only or rlt_encoder_only,
    )
    if rlt_encoder_only:
        merged.model.rlt_module.decoder = None
    merged.model.config = merged.config
    merged.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="Training checkpoint's pretrained_model folder or Hub ID"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rlt-encoder-only",
        action="store_true",
        help="Export RLT features without the reconstruction decoder",
    )
    parser.add_argument(
        "--revision", help="Revision of the adapter on the Hub; the base uses its own saved revision"
    )
    args = parser.parse_args()
    output = merge_adapter(
        args.adapter,
        args.output_dir,
        device=args.device,
        revision=args.revision,
        rlt_encoder_only=args.rlt_encoder_only,
    )
    print(f"Merged PI05 model and processors saved to {output}")


if __name__ == "__main__":
    main()
