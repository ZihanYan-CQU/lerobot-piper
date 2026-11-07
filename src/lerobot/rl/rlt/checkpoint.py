"""Atomic Stage 2 snapshots: actor, critics/targets, optimizers, replay, and RNG."""

import json
import random
import shutil
import tempfile
from pathlib import Path

import torch

from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.rl.algorithms.rlt_ac import RLTACAlgorithm

from .replay import FixedWindowReplayBuffer, replay_spec


def save_checkpoint(
    directory: str | Path, algorithm: RLTACAlgorithm, replay: FixedWindowReplayBuffer
) -> Path:
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"Checkpoint destination must be new: {directory}")
    if replay_spec(algorithm.policy.config) != replay_spec(replay.config):
        raise ValueError("Actor/replay contracts differ")
    algorithm.make_optimizers_and_scheduler()
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".", dir=directory.parent))
    try:
        algorithm.policy.save_pretrained(temporary / "actor")
        algorithm.save_pretrained(temporary / "algorithm")
        replay.save(temporary / "replay.pt")
        torch.save(
            {
                "optimization_step": algorithm.optimization_step,
                "optimizers": {
                    key: optimizer.state_dict() for key, optimizer in algorithm.get_optimizers().items()
                },
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            },
            temporary / "training_state.pt",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "spec": replay_spec(replay.config),
                    "optimization_step": algorithm.optimization_step,
                },
                indent=2,
            )
        )
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def load_checkpoint(directory: str | Path, *, device: str = "cpu", restore_rng: bool = True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 2:
        raise ValueError(
            "Legacy RLT critic/optimizer checkpoints cannot resume the corrected planned-action critic; "
            "warm-start actor weights with --replay ... --actor-checkpoint .../actor instead"
        )
    config = RLTActorConfig.from_pretrained(directory / "actor")
    config.device = device
    if manifest["spec"] != replay_spec(config):
        raise ValueError("Checkpoint actor/replay feature contracts differ")
    policy = RLTActorPolicy.from_pretrained(directory / "actor", config=config, strict=True)
    algorithm = RLTACAlgorithm.from_pretrained(directory / "algorithm", policy=policy, device=device)
    replay = FixedWindowReplayBuffer.load(directory / "replay.pt", config)
    training = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    if training["optimization_step"] != manifest["optimization_step"]:
        raise ValueError("Checkpoint optimizer step differs from manifest")
    optimizers = algorithm.make_optimizers_and_scheduler()
    if set(training["optimizers"]) != set(optimizers):
        raise ValueError("Checkpoint must contain both actor and critic optimizer states")
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(training["optimizers"][key])
    algorithm.optimization_step = training["optimization_step"]
    if restore_rng:
        torch.set_rng_state(training["torch_rng"])
        random.setstate(training["python_rng"])
        if training["cuda_rng"] and torch.device(device).type == "cuda":
            torch.cuda.set_rng_state_all(training["cuda_rng"])
    return algorithm, replay
