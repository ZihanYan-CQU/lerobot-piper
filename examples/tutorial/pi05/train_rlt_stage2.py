"""Train/resume the compact RLT actor-critic from collected fixed-window latent replay.

This entrypoint never connects to a robot. Collection uses FrozenRLTFeatures,
FixedWindowRollout and an explicit environment/RobotIO reward callback.
"""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm, RLTACConfig
from lerobot.rl.data_sources.data_mixer import OnlineOfflineMixer
from lerobot.rl.rlt.checkpoint import load_checkpoint, save_checkpoint
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer, replay_spec
from lerobot.rl.trainer import RLTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--replay",
        type=Path,
        help="FixedWindowReplayBuffer.save() file or a completed RTC journal session directory",
    )
    source.add_argument("--resume", type=Path, help="Complete Stage 2 checkpoint directory")
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        help="Warm-start only actor weights with --replay; critics and optimizers start fresh",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000, help="Additional learner updates")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+")
    parser.add_argument("--discount", type=float, default=None, help="Per-policy-step gamma; new run only")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if args.output_dir.exists():
        parser.error("output-dir must be new")
    if args.resume and args.actor_checkpoint:
        parser.error("actor-checkpoint is a policy-only warm start with --replay, not --resume")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if args.resume:
        if args.discount is not None:
            parser.error("resume uses the saved discount")
        algorithm, replay = load_checkpoint(args.resume, device=args.device)
    else:
        metadata = args.replay
        if metadata.is_dir():
            shards = sorted(metadata.glob("*.pt"))
            if not shards:
                parser.error("replay journal contains no completed windows")
            metadata = shards[0]
        data = torch.load(metadata, map_location="cpu", weights_only=True)
        config = RLTActorConfig(
            **data["spec"],
            hidden_dims=tuple(args.hidden_dims or (256, 256)),
            device=args.device,
            push_to_hub=False,
        )
        policy = None
        if args.actor_checkpoint:
            saved_config = RLTActorConfig.from_pretrained(args.actor_checkpoint)
            if replay_spec(saved_config) != data["spec"]:
                parser.error(
                    "actor-checkpoint and replay must use the same frozen features and window/RTC contract"
                )
            if args.hidden_dims is not None and tuple(args.hidden_dims) != tuple(saved_config.hidden_dims):
                parser.error("hidden-dims must match the warm-start actor")
            saved_config.device = args.device
            saved_config.push_to_hub = False
            config = saved_config
            policy = RLTActorPolicy.from_pretrained(args.actor_checkpoint, config=config, strict=True)
        replay = FixedWindowReplayBuffer.load(args.replay, config)
        algorithm = RLTACAlgorithm(
            policy if policy is not None else RLTActorPolicy(config),
            RLTACConfig(
                policy_config=config,
                discount=0.99 if args.discount is None else args.discount,
            ),
        )
    if not len(replay):
        parser.error("replay must contain executed windows before training")
    trainer = RLTrainer(algorithm, OnlineOfflineMixer(replay), args.batch_size)
    for _ in range(args.steps):
        stats = trainer.training_step()
        print(json.dumps({"step": algorithm.optimization_step, **stats.to_log_dict()}), flush=True)
    output = save_checkpoint(args.output_dir, algorithm, replay)
    print(f"RLT actor, critics, optimizer state and replay saved to {output}")


if __name__ == "__main__":
    main()
