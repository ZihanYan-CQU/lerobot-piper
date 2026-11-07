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

"""Export a full or PEFT RLT Stage 1 checkpoint for action and RL-feature inference."""

import argparse
from dataclasses import replace
from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy


def export_rlt(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    revision: str | None = None,
) -> Path:
    """Bundle the VLA, RLT encoder, saved normalization, and tokenizer.

    PEFT checkpoints are merged first, including the full-trained expert and
    RLT module. The training checkpoint is retained for resuming reconstruction.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    config = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
    if not isinstance(config, PI05Config) or not config.use_rlt:
        raise ValueError("Expected a PI05 RLT Stage 1 checkpoint with use_rlt=True.")
    if config.use_peft:
        if __package__:
            from .merge_lora import merge_adapter
        else:
            from merge_lora import merge_adapter

        return merge_adapter(
            str(checkpoint), output_dir, device=device, revision=revision, rlt_encoder_only=True
        )
    # Export the saved processor/tokenizer together; a training-time external
    # tokenizer path must not override the checkpoint's bundled assets.
    config = replace(
        config, device=device, compile_model=False, gradient_checkpointing=False, text_tokenizer_name=None
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        pretrained_revision=revision,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy = PI05Policy.from_pretrained(checkpoint, config=config, revision=revision)
    policy.model.rlt_module.decoder = None
    policy.config = replace(
        config,
        rlt_inference_only=True,
        pretrained_path=None,
        pretrained_revision=None,
        text_tokenizer_name=None,
    )
    policy.model.config = policy.config
    policy.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Training checkpoint's pretrained_model directory or Hub ID"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision")
    args = parser.parse_args()
    output = export_rlt(args.checkpoint, args.output_dir, device=args.device, revision=args.revision)
    print(f"PI05 and RLT encoder saved to {output}")


if __name__ == "__main__":
    main()
